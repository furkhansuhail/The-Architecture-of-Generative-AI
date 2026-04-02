"""Module: RAG Design Patterns"""

"""
RAG Design Patterns
===================

Retrieval-Augmented Generation (RAG) solves a fundamental problem with
Large Language Models: their knowledge is frozen at training time. RAG
gives LLMs access to external, up-to-date, and private knowledge at
inference time by retrieving relevant documents and injecting them into
the prompt as context.

But RAG is not one thing. As use cases have grown more complex — handling
images and video, reasoning over structured knowledge, orchestrating
multiple agents — distinct architectural patterns have emerged. Choosing
the wrong pattern costs accuracy, latency, and money.

This module covers all seven major RAG patterns:

  1. Naive RAG          — the baseline; simple and fast
  2. Retrieve-and-Rerank — two-stage precision pipeline
  3. Multimodal RAG     — documents beyond text (images, audio, video)
  4. Graph RAG          — structured knowledge with relationships
  5. Hybrid RAG         — vector + graph, best of both worlds
  6. Agentic RAG        — a router that decides HOW to retrieve
  7. Multi-Agent RAG    — parallel specialised agents for complex queries

For each pattern: how it works, when to use it, its strengths and
failure modes, and key design decisions.
"""

import re
import textwrap

TOPIC_NAME   = "08a-RAG Design Patterns"
DISPLAY_NAME = "08a-RAG Design Patterns"
ICON         = "🔍"
SUBTITLE     = "Naive → Rerank → Multimodal → Graph → Hybrid → Agentic → Multi-Agent"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — WHY RAG? THE CORE PROBLEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Knowledge Gap Problem

LLMs are trained once and frozen. They have no knowledge of:
    * Events after their training cutoff
    * Private documents (internal wikis, contracts, codebases)
    * Real-time data (stock prices, weather, live databases)
    * Domain-specific knowledge not well represented in training data

Two naive solutions both fail:

    FINE-TUNING          Expensive, slow, requires re-training for every update.
                         Catastrophic forgetting erases old knowledge.
                         Does not scale to millions of documents.

    CONTEXT STUFFING     Simply putting all documents in the prompt.
                         Context windows are finite (and costly).
                         Model attention degrades on very long contexts.
                         Still does not handle real-time updates.

RAG is the practical middle ground: retrieve only the relevant documents
at query time, inject them as context, and generate a grounded response.

### The RAG Pipeline (Universal Components)

Every RAG variant shares the same three-stage structure:

    ┌─────────────────────────────────────────────────────────┐
    │  INDEXING (offline, one-time)                           │
    │  Documents → Chunks → Embeddings → Vector Store         │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │  RETRIEVAL (online, per query)                          │
    │  Query → Embed → Search Vector Store → Top-K chunks     │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │  GENERATION (online, per query)                         │
    │  Query + Context → Prompt Template → LLM → Response     │
    └─────────────────────────────────────────────────────────┘

The seven patterns differ in how they implement and augment these stages.


### Key Metrics for Choosing a RAG Pattern

    FAITHFULNESS      Is the answer grounded in the retrieved context?
    RELEVANCE         Does the retrieved context actually answer the query?
    LATENCY           How long does the end-to-end pipeline take?
    COST              Embedding calls, LLM calls, infrastructure cost
    COMPLEXITY        Engineering effort to build and maintain
    DATA TYPE         Text-only vs multimodal vs structured/relational


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — PATTERN 1: NAIVE RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### How It Works

Naive RAG is the baseline: a single-stage vector similarity search feeding
directly into the LLM. No reranking, no graph traversal, no agents.

    INDEXING:
        1. Split documents into fixed-size chunks (e.g. 512 tokens, 50-token overlap)
        2. Embed each chunk with an embedding model (e.g. text-embedding-ada-002)
        3. Store (embedding, chunk_text, metadata) in a vector database

    RETRIEVAL:
        1. Embed the user query with the same embedding model
        2. Compute cosine similarity between query embedding and all chunk embeddings
        3. Return top-K most similar chunks (K = 3-10 typically)

    GENERATION:
        1. Insert retrieved chunks into a prompt template
        2. Send to LLM: "Answer using only the provided context: {context}\n\nQ: {query}"
        3. Return LLM response


### Pipeline Diagram

    Documents ----chunk----> Chunks
                                |
                            embed()
                                |
                         Vector Database
                                |
    Query --embed()--> similarity search ---> Top-K Context
                                                    |
                               Prompt Template <----+
                                      |
                               Generative Model
                                      |
                                  Response


### Chunking Strategies

Chunking is surprisingly impactful on Naive RAG performance:

    FIXED-SIZE CHUNKING    Split every N tokens. Simple, but may break
                           sentences, paragraphs, or code mid-way.

    SENTENCE CHUNKING      Split at sentence boundaries. Preserves semantics
                           but creates variable-length chunks.

    RECURSIVE CHUNKING     Try splitting on paragraph → sentence → word.
                           Most practical default (LangChain default).

    SEMANTIC CHUNKING      Use an embedding model to split at semantic
                           boundaries. Higher quality but slower.

    DOCUMENT-AWARE         Respect document structure: PDF headers,
                           Markdown sections, HTML tags. Best for
                           structured documents.

    Key parameters:
        chunk_size:     200–1000 tokens (shorter = more precise, longer = more context)
        chunk_overlap:  10–20% of chunk size (prevents context loss at boundaries)


### When to Use Naive RAG

    ✓  Small to medium knowledge base (< 100K documents)
    ✓  Text-only documents with dense, uniform content
    ✓  Latency-sensitive applications (fastest pipeline)
    ✓  Prototyping and initial development
    ✓  Queries are simple, well-formed, semantically clear
    ✓  Budget-constrained deployments (fewest LLM calls)

### When NOT to Use

    ✗  Queries are ambiguous or require keyword matching (use hybrid search)
    ✗  Relevance needs are strict (use reranking)
    ✗  Documents contain images, charts, or tables (use multimodal RAG)
    ✗  Knowledge has complex relationships (use graph RAG)


### Common Failure Modes

    SEMANTIC MISMATCH      Query and relevant chunk use different vocabulary.
                           Embedding similarity is high for wrong chunks.
                           Fix: use larger embedding model, add keyword search.

    CHUNK BOUNDARY LOSS    Answer spans two chunks; neither chunk alone is useful.
                           Fix: increase overlap, use semantic chunking.

    CONTEXT DILUTION       Top-K returns irrelevant chunks that confuse the LLM.
                           Fix: reduce K, use reranking, improve chunking.

    LOST IN THE MIDDLE     LLM ignores context in the middle of a long prompt.
                           Fix: put most relevant chunks first and last.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — PATTERN 2: RETRIEVE-AND-RERANK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Problem with Single-Stage Retrieval

Embedding-based similarity (ANN search) is fast but imprecise. It captures
semantic similarity but misses:
    * Exact keyword matches
    * Fine-grained relevance nuances
    * Query-document interaction signals

A reranker addresses this with a second-pass model that scores each
(query, document) pair jointly — far more accurate than embedding
similarity alone.


### How It Works

    STAGE 1 — BROAD RETRIEVAL (recall-optimised):
        Retrieve top-N candidates (e.g. N=50) using fast ANN search.
        Prioritise recall over precision — cast a wide net.

    STAGE 2 — RERANKING (precision-optimised):
        Score all N candidates with a cross-encoder reranker model.
        Cross-encoders process the full (query, document) pair jointly —
        much more powerful than separate embedding similarity.
        Return top-K after reranking (e.g. K=5 from N=50).

    GENERATION:
        Same as Naive RAG but with higher-quality top-K context.


### Cross-Encoder vs Bi-Encoder

    BI-ENCODER (Naive RAG):           CROSS-ENCODER (Reranker):
    Query  → embed → q_vec            [Query, Document] → score
    Doc    → embed → d_vec            Single model sees both together
    score  = cosine(q_vec, d_vec)     Attention across query AND document
    Pros: O(1) per query (pre-indexed) Pros: Far higher relevance accuracy
    Cons: Weaker relevance signal      Cons: O(N) per query (cannot pre-index)

    The two-stage design combines both:
        Bi-encoder for speed (retrieve N), cross-encoder for accuracy (rerank K).


### Pipeline Diagram

    Documents ----chunk----> Chunks
                                |
                            embed()
                                |
                         Vector Database
                                |
    Query --embed()--> similarity search ---> Top-50 candidates
                                                    |
                               Reranker Model <-----+
                               (cross-encoder)
                                    |
                                Top-5 Re-ranked
                                    |
                             Prompt Template
                                    |
                           Generative Model
                                    |
                               Response


### Reranker Model Options

    COHERE RERANK           Commercial API, high accuracy, pay-per-call.
    BGE-RERANKER            Open-source, strong performance, self-hosted.
    COLBERT                 Token-level late interaction, fast+accurate.
    MONOT5 / RANKT5         T5-based, good for longer documents.
    LLM-AS-RERANKER         Ask the LLM itself to score and rank candidates.
                            Most accurate but most expensive.


### When to Use Retrieve-and-Rerank

    ✓  Precision matters more than latency (e.g. legal, medical, compliance)
    ✓  Large, noisy document collections where many chunks look superficially similar
    ✓  User queries are conversational or ambiguous
    ✓  You can tolerate extra latency (typically +100–500ms)
    ✓  The cost of a wrong answer is high

### Latency/Cost Tradeoff

    Naive RAG:              1 embed call + ANN search + 1 LLM call
    Retrieve-and-Rerank:    1 embed call + ANN search + N reranker calls + 1 LLM call

    Optimisation: batch N chunks through the reranker in one API call.
    Typical overhead: 100–300ms with a hosted reranker API.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — PATTERN 3: MULTIMODAL RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Problem

Most real-world documents are not purely text:
    * PDFs contain charts, diagrams, and images
    * Product catalogs have images alongside descriptions
    * Videos have frames, transcripts, and audio
    * Slide decks carry visual information as the primary medium

Naive RAG discards all non-text content. Multimodal RAG preserves and
retrieves across modalities.


### How It Works

    INDEXING:
        1. Extract all modalities: text, images, audio, video frames
        2. Use a multimodal embedding model (CLIP, ImageBind, etc.)
           that maps all modalities into a SHARED embedding space
        3. Store all embeddings in the vector database with modality metadata
        4. Optionally: generate text captions for images using a vision LLM

    RETRIEVAL:
        1. Embed the query (text, image, or both) using the same model
        2. Cross-modal search: a text query can retrieve images;
           an image query can retrieve text documents
        3. Return top-K across all modalities

    GENERATION:
        1. Feed retrieved media (images, video frames) directly to a
           multimodal LLM (GPT-4V, Claude, Gemini) alongside the query
        2. The LLM "sees" and reasons over the retrieved visual content


### Multimodal Embedding Models

    CLIP (OpenAI)       Text + Images in a shared space.
                        Excellent zero-shot cross-modal retrieval.

    IMAGEBIND (Meta)    Text + Images + Audio + Video + IMU + Depth.
                        6 modalities in one shared space.

    E5-MISTRAL          Text-first but strong at cross-modal reasoning.

    VOYAGE-MULTIMODAL   Commercial, strong PDF + diagram support.


### When to Use Multimodal RAG

    ✓  Documents contain diagrams, charts, or images critical to the answer
    ✓  Product search (image queries → product descriptions)
    ✓  Video Q&A (query → relevant frame + transcript)
    ✓  Medical imaging reports (query → scan image + clinical notes)
    ✓  Technical manuals with figures and schematics


### Key Design Decisions

    CAPTION GENERATION   Generate text descriptions of images at index time
                         using a vision LLM. Allows text-based retrieval of
                         visual content. Adds cost; improves recall.

    LATE FUSION          Retrieve text and images separately, merge results.
    EARLY FUSION         Embed everything together into one shared space.
    HYBRID               Embed with CLIP, caption with LLM, search both.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — PATTERN 4: GRAPH RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Problem with Flat Vector Search

Vector databases treat each chunk independently. They cannot represent:
    * "Person A reports to Person B who leads Department C"
    * "Drug X inhibits enzyme Y which regulates pathway Z"
    * "Event A caused Event B which preceded Event C"
    * "Company X acquired Company Y, which owns Patent Z"

These relationships are structural knowledge — encoded in the edges and
traversal paths of a graph, not in the semantic content of individual chunks.


### How It Works

    INDEXING (two parallel paths):
        1. Standard path: Documents → Chunks → Embeddings → Vector DB
        2. Graph path: Documents → LLM extracts entities and relationships
                       → Knowledge Graph (nodes = entities, edges = relations)
                       → Stored in a graph database (Neo4j, Amazon Neptune)

    RETRIEVAL:
        1. Embed query → Vector DB search → Top-K relevant chunks
        2. Extract entities from query (or from retrieved chunks)
        3. Graph traversal: starting from those entities, follow edges
           to discover related entities and facts
        4. Combine: vector results + graph traversal results

    GENERATION:
        Context = retrieved chunks (vector) + graph facts → LLM → Response


### LLM Graph Generation

The LLM is used to extract the knowledge graph from raw text:

    Input:  "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976."
    Output: [
        ("Apple Inc.", "FOUNDED_BY", "Steve Jobs"),
        ("Apple Inc.", "FOUNDED_BY", "Steve Wozniak"),
        ("Apple Inc.", "FOUNDED_IN", "1976")
    ]

This happens at index time. The LLM acts as an information extractor.
OpenAI + LangChain GraphTransformer or Microsoft GraphRAG (2024) do this.


### Microsoft GraphRAG (2024)

Microsoft's GraphRAG is a landmark paper/system that:
    1. Extracts a hierarchical knowledge graph from all documents
    2. Creates "community summaries" at multiple granularities
    3. For local queries: traverse relevant graph subgraph
    4. For global queries: summarise across community reports

    Key result: 3-6x better performance on global, multi-hop questions
    vs standard RAG on the same corpus.


### When to Use Graph RAG

    ✓  Multi-hop questions: "Who is the CEO of the company that acquired X?"
    ✓  Relationship queries: "What drugs interact with drug X?"
    ✓  Narrative/temporal: "What events led to the 2008 financial crisis?"
    ✓  Enterprise knowledge graphs: org charts, product dependencies
    ✓  Scientific knowledge: protein interactions, citation networks

### Costs and Tradeoffs

    INDEXING COST     Extracting a knowledge graph requires many LLM calls.
                      For 1M tokens of documents: ~$10–100 in LLM API costs.

    QUERY LATENCY     Graph traversal adds 50–200ms. But answer quality
                      for complex queries is dramatically higher.

    MAINTENANCE       Updating the graph as documents change requires
                      incremental graph updates (non-trivial).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — PATTERN 5: HYBRID RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Best of Both Worlds

Hybrid RAG combines vector search (semantic) with graph traversal
(structural), fusing their results for richer context.

    Vector DB:   "What does this text mean?"  → semantic similarity
    Graph DB:    "How are these entities related?" → structural relationships

Neither alone is sufficient for complex enterprise knowledge bases.
Together they cover queries that neither can handle individually.


### Architecture

    INDEXING:
        1. Documents → Chunks → Vector DB (same as Naive RAG)
        2. Documents → LLM extracts KG → Graph DB (same as Graph RAG)
        Both indexes maintained in parallel, in sync.

    RETRIEVAL:
        Path A: Query → embed → vector search → semantic chunks
        Path B: Query → entity extraction → graph traversal → related facts
        Fusion:  Merge and deduplicate results from both paths
                 (reciprocal rank fusion, score normalisation, or LLM re-ranking)

    GENERATION:
        Combined context (vector chunks + graph facts) → LLM → Response


### Fusion Strategies

    SIMPLE UNION            Concatenate all results, pass to LLM.
                            Fast, but can dilute precision with noise.

    RECIPROCAL RANK FUSION  Score = Σ 1/(rank_i + k). Robust merging
                            of ranked lists from heterogeneous retrievers.

    LLM RE-RANKING          Ask the LLM to select the most relevant subset.
                            Highest quality, highest cost.

    CONFIDENCE-WEIGHTED     Weight vector vs graph results by query type.
                            Entity-heavy queries → weight graph more.
                            Semantic queries → weight vector more.


### When to Use Hybrid RAG

    ✓  Enterprise knowledge bases (combine documents + org structure)
    ✓  Customer support (semantic FAQ + product knowledge graph)
    ✓  Financial research (news documents + company relationship graph)
    ✓  Healthcare (clinical notes + drug/disease knowledge graph)
    ✓  When Graph RAG alone misses semantic nuance
    ✓  When Naive RAG alone misses structural relationships

### Infrastructure Requirements

    Requires maintaining BOTH a vector database AND a graph database.
    Popular combination: Weaviate or Pinecone (vector) + Neo4j (graph).
    Cloud solutions: Amazon Neptune + OpenSearch; Azure Cosmos DB.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — PATTERN 6: AGENTIC RAG (ROUTER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Problem with Static Pipelines

All previous patterns use a fixed retrieval strategy regardless of the query.
But different queries demand different retrieval approaches:

    "What is our refund policy?"        → simple vector search
    "Compare our Q3 and Q4 revenue"     → structured database query
    "What happened in the news today?"  → web search
    "Summarise this uploaded document"  → direct document analysis
    "Create a chart of sales by region" → code execution

A static pipeline uses the same strategy for all of these — poorly.
An Agentic RAG Router uses an AI agent to DECIDE which retrieval tool to call.


### How It Works

    1. User sends query to an AI Agent (LLM with tool-use capability)
    2. Agent analyses the query and DECIDES which retrieval tool(s) to use:
           - Vector database search (semantic)
           - SQL database query (structured data)
           - Web search API (real-time information)
           - Document analysis (direct file processing)
           - Code execution (computation)
           - Knowledge graph traversal (relational)
    3. Agent calls the chosen tool(s) with appropriate parameters
    4. Agent receives context from the tool
    5. Agent passes context to a multimodal LLM for response generation


### Router Decision Logic

The agent uses a combination of:

    FUNCTION CALLING      The LLM is given tool descriptions and decides
                          which function to call. OpenAI function calling,
                          Anthropic tool use, LangChain tools.

    QUERY CLASSIFICATION  A lightweight classifier categorises the query
                          (factual, comparative, real-time, computational)
                          and routes accordingly.

    CHAIN-OF-THOUGHT      The agent reasons: "This query asks for today's
                          date — I should use web search, not vector DB."


### When to Use Agentic RAG

    ✓  Knowledge base has heterogeneous data sources (docs + DB + web)
    ✓  Queries span multiple modalities and types
    ✓  You need adaptive retrieval without hardcoding rules
    ✓  Some queries don't require retrieval (agent skips it)
    ✓  Some queries need computation, not retrieval

### Tradeoffs

    FLEXIBILITY     Very high. Agent adapts to any query type.
    RELIABILITY     Lower than fixed pipelines. Agent routing can fail.
                    Mitigation: fallback to vector search if routing fails.
    LATENCY         Higher. Agent reasoning adds 200–500ms.
    COST            Higher. Extra LLM call for routing decision.
    OBSERVABILITY   Harder to debug. Why did the agent choose this tool?


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — PATTERN 7: MULTI-AGENT RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### When One Agent Is Not Enough

Some queries are inherently multi-faceted and require parallel investigation:

    "Compare our internal product performance data with competitor news,
     customer reviews, and analyst reports, and summarise the findings."

A single agent would handle this sequentially — slow and potentially
confused by the breadth of the task. Multi-Agent RAG uses specialised
parallel agents, each owning a domain.


### Architecture

    ORCHESTRATOR AGENT
        Receives the query, decomposes it into sub-tasks, assigns each
        sub-task to a specialised agent, waits for all results,
        and synthesises the final response.

    SPECIALISED AGENTS (run in parallel):
        Agent A:  Vector search engine A (internal documents)
        Agent B:  Vector search engine B (external knowledge)
        Agent C:  Web search (real-time news, competitor data)
        Agent D:  Slack/Gmail/internal tools (communication data)
        Agent E:  Code execution (data analysis, calculations)

    SYNTHESIS:
        Orchestrator receives all agent results → deduplicates →
        summarises → Generative model → Response


### Multi-Agent Communication Patterns

    PARALLEL (all agents simultaneously)
        Each agent is given the full query or a sub-query.
        Results aggregated at the end.
        Best for: independent sub-tasks, time-sensitive applications.

    SEQUENTIAL (pipeline of agents)
        Agent A retrieves → Agent B refines → Agent C synthesises.
        Best for: when later agents need results from earlier agents.

    HIERARCHICAL (tree of agents)
        Root orchestrator → domain orchestrators → leaf agents.
        Best for: very large knowledge bases with clear domain partitioning.

    COLLABORATIVE (peer agents)
        Agents share partial results and coordinate.
        Best for: complex reasoning requiring multiple perspectives.


### When to Use Multi-Agent RAG

    ✓  Knowledge base spans completely separate domains (internal docs,
       external web, CRM data, communication tools)
    ✓  Query requires synthesising information from 3+ distinct sources
    ✓  Latency can be reduced by parallelism (parallel agents run in ~same
       time as the slowest single agent)
    ✓  Domain expertise matters: each source needs different retrieval logic
    ✓  Enterprise applications: sales intelligence, competitive research,
       medical differential diagnosis

### Orchestration Frameworks

    LANGGRAPH      State machine for multi-agent workflows. Recommended.
    AUTOGEN        Microsoft's multi-agent conversation framework.
    CREWAI         Role-based agent crews. Good for structured pipelines.
    SEMANTIC KERNEL  Microsoft enterprise agent orchestration.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — DECISION GUIDE: WHICH PATTERN TO USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Decision Tree

    Is your data text-only?
      YES → Is precision critical?
              YES → Does data have relationships? → YES: Graph RAG / Hybrid RAG
                                                  → NO: Retrieve-and-Rerank
              NO  → Naive RAG (start here, always)
      NO  → Does it include images/video/audio?
              YES → Multimodal RAG
              NO  → Hybrid RAG (structured data + text)

    Does the query type vary widely?
      YES → Agentic RAG (Router)

    Do you need to synthesise 3+ distinct knowledge sources?
      YES → Multi-Agent RAG


### Pattern Comparison Table

    +------------------+----------+----------+----------+----------+----------+
    | Pattern          | Latency  | Accuracy | Cost     | Complexity| Best For|
    +------------------+----------+----------+----------+----------+----------+
    | Naive RAG        | Fastest  | Baseline | Lowest   | Low      | Prototypes|
    | Retrieve-Rerank  | Medium   | High     | Low-Med  | Low      | Precision |
    | Multimodal RAG   | Medium   | High*    | Medium   | Medium   | Mixed docs|
    | Graph RAG        | Medium   | High**   | High     | High     | Relations |
    | Hybrid RAG       | Slow     | Highest  | High     | Very High| Enterprise|
    | Agentic RAG      | Variable | Adaptive | Medium   | Medium   | Diverse Q |
    | Multi-Agent RAG  | Parallel | Highest  | Highest  | Highest  | Complex Q |
    +------------------+----------+----------+----------+----------+----------+
    * For multimodal queries only. ** For relational queries only.


### The Evolution Path

Most production systems follow this maturity path:

    START HERE:    Naive RAG
    IMPROVE FIRST: Add reranking (highest ROI for lowest effort)
    THEN ADD:      Hybrid search (keyword + semantic) within vector DB
    IF NEEDED:     Graph RAG (if multi-hop questions are failing)
    IF NEEDED:     Agentic RAG (if query types are heterogeneous)
    SCALE UP:      Multi-Agent RAG (if synthesis across sources is needed)

Do not start with Multi-Agent RAG. Start with Naive RAG and add complexity
only when you have measured evidence that simpler approaches are failing.


### Production RAG Evaluation Metrics

    RAGAS FRAMEWORK (standard):
        Context Recall:      Did we retrieve the right chunks?
        Context Precision:   Are all retrieved chunks relevant?
        Faithfulness:        Is the answer grounded in the context?
        Answer Relevance:    Does the answer address the question?

    ADDITIONAL:
        Latency p50/p99:     Median and tail latency in production
        Token Cost:          Average tokens per query × price
        Cache Hit Rate:      % of queries served from cache

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "1. Build a Naive RAG Pipeline from Scratch": {
        "description": "Implement a complete Naive RAG pipeline in pure Python: "
                       "chunking, TF-IDF embedding, vector similarity search, "
                       "and prompt assembly. No external APIs required.",
        "code": """\
import numpy as np
import re
from collections import Counter

# ── Step 1: Document corpus ────────────────────────────────────
DOCS = [
    "Retrieval-Augmented Generation (RAG) combines retrieval with language model generation. "
    "It retrieves relevant documents from a knowledge base and uses them as context for the LLM.",

    "Vector databases store high-dimensional embeddings and support approximate nearest neighbour "
    "search. Popular choices include Pinecone, Weaviate, Chroma, and FAISS.",

    "Chunking splits documents into smaller pieces before embedding. Chunk size affects retrieval "
    "quality. Typical chunk sizes are 256 to 1024 tokens with 10-20% overlap.",

    "Cross-encoder rerankers score (query, document) pairs jointly and are more accurate than "
    "bi-encoder similarity but cannot be pre-indexed. They are used in the second stage of "
    "retrieve-and-rerank pipelines.",

    "Graph RAG builds a knowledge graph from documents and uses graph traversal alongside vector "
    "search. Microsoft GraphRAG (2024) showed 3-6x improvement on multi-hop questions.",

    "The Naive RAG pipeline has three stages: indexing (chunk and embed documents), "
    "retrieval (embed query and find similar chunks), and generation (LLM + context).",

    "Embedding models convert text to dense vectors. OpenAI text-embedding-ada-002 outputs "
    "1536-dimensional vectors. Open-source alternatives include BAAI/bge and E5 models.",

    "Hallucination in LLMs is reduced by RAG because the model is grounded in retrieved facts. "
    "Faithfulness measures whether the answer is supported by the retrieved context.",
]

# ── Step 2: Chunking (sentence-level for this demo) ────────────
def chunk_document(text, chunk_size=1, overlap=0):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max(1, chunk_size - overlap)):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

chunks = []
for doc_id, doc in enumerate(DOCS):
    for chunk in chunk_document(doc, chunk_size=2, overlap=1):
        chunks.append({'doc_id': doc_id, 'text': chunk})

print(f"Corpus: {len(DOCS)} docs → {len(chunks)} chunks")
print(f"Sample chunk: '{chunks[0]['text'][:80]}...'")
print()

# ── Step 3: TF-IDF Embedding (lightweight, no API needed) ──────
def tokenise(text):
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

def build_tfidf(corpus_chunks):
    all_tokens = [tokenise(c['text']) for c in corpus_chunks]
    vocab      = sorted(set(t for tokens in all_tokens for t in tokens))
    vocab_idx  = {w: i for i, w in enumerate(vocab)}
    N          = len(corpus_chunks)

    # Document frequency
    df = Counter()
    for tokens in all_tokens:
        for t in set(tokens):
            df[t] += 1

    # TF-IDF matrix
    mat = np.zeros((N, len(vocab)), dtype=np.float32)
    for i, tokens in enumerate(all_tokens):
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        for t, count in tf.items():
            if t in vocab_idx:
                idf = np.log((N + 1) / (df[t] + 1)) + 1
                mat[i, vocab_idx[t]] = (count / total) * idf

    # L2-normalise rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat   = mat / np.where(norms == 0, 1, norms)
    return mat, vocab_idx, df, N

embeddings, vocab_idx, df, N_docs = build_tfidf(chunks)
print(f"Vocabulary size: {len(vocab_idx)} terms")
print(f"Embedding matrix: {embeddings.shape}")
print()

# ── Step 4: Query and retrieve ─────────────────────────────────
def embed_query(query_text, vocab_idx, df, N):
    tokens = tokenise(query_text)
    tf     = Counter(tokens)
    total  = max(len(tokens), 1)
    vec    = np.zeros(len(vocab_idx), dtype=np.float32)
    for t, count in tf.items():
        if t in vocab_idx:
            idf = np.log((N + 1) / (df[t] + 1)) + 1
            vec[vocab_idx[t]] = (count / total) * idf
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def retrieve(query, top_k=3):
    q_vec    = embed_query(query, vocab_idx, df, N_docs)
    scores   = embeddings @ q_vec          # cosine similarity (pre-normalised)
    top_idx  = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i]['text'], float(scores[i])) for i in top_idx]

# ── Step 5: Prompt assembly ────────────────────────────────────
def build_prompt(query, retrieved_chunks):
    context = '\\n\\n'.join(
        f"[{i+1}] {text}" for i, (text, _) in enumerate(retrieved_chunks)
    )
    return (
        f"Answer the question using ONLY the provided context.\\n\\n"
        f"CONTEXT:\\n{context}\\n\\n"
        f"QUESTION: {query}\\n\\n"
        f"ANSWER:"
    )

# ── Step 6: Demo queries ───────────────────────────────────────
queries = [
    "What is chunking and why does it matter?",
    "How does retrieve-and-rerank improve accuracy?",
    "What are the three stages of a RAG pipeline?",
]

print("Naive RAG — Retrieval Demo")
print("=" * 56)
for query in queries:
    results = retrieve(query, top_k=3)
    print(f"\\nQuery: '{query}'")
    print("Retrieved chunks:")
    for rank, (text, score) in enumerate(results, 1):
        print(f"  [{rank}] score={score:.4f}: {text[:80]}...")
    print("\\nPrompt preview (first 200 chars):")
    prompt = build_prompt(query, results)
    print(f"  {prompt[:200]}...")
    print()

print(f"Pipeline: {len(DOCS)} docs → {len(chunks)} chunks → TF-IDF embed → cosine sim → LLM")
""",
    },

    "2. Chunking Strategies — Impact on Retrieval Quality": {
        "description": "Implement four chunking strategies (fixed-size, sentence, "
                       "recursive, overlap) and measure how chunk size and overlap "
                       "affect retrieval precision on the same query set.",
        "code": """\
import numpy as np
import re
from collections import Counter

TEXT = \"\"\"
Retrieval-Augmented Generation combines retrieval systems with large language models.
The retrieval stage finds relevant documents from a knowledge base. Documents are split
into chunks and embedded as dense vectors. At query time the query is embedded and
compared against all chunk embeddings. The most similar chunks are retrieved and passed
to the language model as context. The language model generates a response grounded in
the retrieved context. This reduces hallucination because the model cites specific facts.
Chunk size is a critical hyperparameter. Smaller chunks are more precise but may miss
context. Larger chunks capture more context but reduce retrieval precision. Overlap
between chunks ensures that information at chunk boundaries is not lost. Typical overlap
is ten to twenty percent of chunk size. Sentence-level chunking preserves semantic
boundaries. Recursive chunking first splits on paragraphs then sentences then words.
\"\"\"

# ── Four chunking strategies ───────────────────────────────────
def chunk_fixed(text, size=50, overlap=0):
    words = text.split()
    step  = max(1, size - overlap)
    return [' '.join(words[i:i+size]) for i in range(0, len(words), step)
            if words[i:i+size]]

def chunk_sentence(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def chunk_recursive(text, max_words=40):
    chunks = []
    for para in re.split('\\n+', text.strip()):
        words = para.split()
        if len(words) <= max_words:
            if para.strip(): chunks.append(para.strip())
        else:
            for sent in re.split(r'(?<=[.!?])\s+', para):
                words_s = sent.split()
                if len(words_s) <= max_words:
                    if sent.strip(): chunks.append(sent.strip())
                else:
                    for i in range(0, len(words_s), max_words):
                        chunks.append(' '.join(words_s[i:i+max_words]))
    return [c for c in chunks if c]

def chunk_overlap(text, size=40, overlap=10):
    return chunk_fixed(text, size=size, overlap=overlap)

strategies = {
    'Fixed-50  (no overlap)':   chunk_fixed(TEXT, 50, 0),
    'Fixed-30  (10 overlap)':   chunk_overlap(TEXT, 30, 10),
    'Sentence  boundaries':     chunk_sentence(TEXT),
    'Recursive (max 40 words)': chunk_recursive(TEXT, 40),
}

# ── Simple TF-IDF embed + cosine sim ──────────────────────────
def tokenise(t): return re.findall(r'\b[a-z]{2,}\b', t.lower())

def tfidf_embed(corpus):
    all_toks = [tokenise(c) for c in corpus]
    vocab    = sorted(set(t for ts in all_toks for t in ts))
    vidx     = {w:i for i,w in enumerate(vocab)}
    N        = len(corpus)
    df       = Counter(t for ts in all_toks for t in set(ts))
    mat      = np.zeros((N, len(vocab)), np.float32)
    for i, tokens in enumerate(all_toks):
        tf = Counter(tokens); total = max(len(tokens),1)
        for t, cnt in tf.items():
            if t in vidx:
                idf = np.log((N+1)/(df[t]+1))+1
                mat[i, vidx[t]] = (cnt/total)*idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat  /= np.where(norms==0,1,norms)
    return mat, vidx, df, N

def query_vec(q, vidx, df, N):
    tokens = tokenise(q); tf = Counter(tokens); total = max(len(tokens),1)
    v = np.zeros(len(vidx), np.float32)
    for t, cnt in tf.items():
        if t in vidx:
            idf = np.log((N+1)/(df[t]+1))+1
            v[vidx[t]] = (cnt/total)*idf
    n = np.linalg.norm(v); return v/n if n>0 else v

# ── Evaluate on 3 queries with ground-truth keywords ──────────
test_queries = [
    ("What is chunk overlap?",        ["overlap", "boundaries"]),
    ("How does query embedding work?",["embed", "query", "vector"]),
    ("Why does RAG reduce hallucination?", ["hallucination", "grounded","facts"]),
]

print("Chunking Strategy Comparison")
print("=" * 68)
print()
for strategy, chunks in strategies.items():
    mat, vidx, df, N = tfidf_embed(chunks)
    total_recall = 0
    for query, keywords in test_queries:
        qv    = query_vec(query, vidx, df, N)
        scores = mat @ qv
        top3   = np.argsort(scores)[::-1][:3]
        top3_text = ' '.join(chunks[i].lower() for i in top3)
        hits  = sum(1 for kw in keywords if kw in top3_text)
        total_recall += hits / len(keywords)
    avg_recall = total_recall / len(test_queries)
    avg_len    = np.mean([len(c.split()) for c in chunks])
    print(f"  Strategy: {strategy}")
    print(f"    Chunks: {len(chunks):>3}  |  Avg length: {avg_len:.1f} words  |  Recall@3: {avg_recall:.2%}")

print()
print("  Interpretation:")
print("    Shorter chunks = more precise but risk boundary loss")
print("    Larger chunks  = more context but dilute similarity scores")
print("    Overlap prevents information loss at chunk boundaries")
print("    Sentence chunking preserves semantic integrity")
""",
    },

    "3. Retrieve-and-Rerank — Two-Stage Pipeline": {
        "description": "Implement the two-stage retrieve-and-rerank pipeline. Stage 1 "
                       "uses fast vector similarity to get top-N candidates. Stage 2 "
                       "uses a cross-encoder-style scorer for precision reranking.",
        "code": """\
import numpy as np
import re
from collections import Counter

CHUNKS = [
    "RAG reduces hallucination by grounding LLM responses in retrieved facts from a knowledge base.",
    "Vector databases use approximate nearest neighbour search for fast similarity retrieval.",
    "The reranker model scores query-document pairs jointly and is more accurate than bi-encoders.",
    "Chunking splits documents into overlapping windows to prevent information loss at boundaries.",
    "Cross-encoders process the full query and document together using bidirectional attention.",
    "Bi-encoders embed query and document independently, enabling pre-computation of doc embeddings.",
    "Naive RAG uses a single retrieval stage with no reranking, making it fast but less precise.",
    "Retrieve-and-rerank first retrieves N candidates then re-scores with a more powerful model.",
    "The two-stage pipeline balances speed (stage 1) and accuracy (stage 2) for optimal performance.",
    "RAG evaluation uses faithfulness, context precision, context recall, and answer relevance.",
    "Embedding models convert text into dense fixed-size vectors capturing semantic meaning.",
    "The LLM generates a response conditioned on the retrieved context in the prompt template.",
]

# ── TF-IDF setup ───────────────────────────────────────────────
def tokenise(t): return re.findall(r'\b[a-z]{2,}\b', t.lower())

def build_index(corpus):
    all_toks = [tokenise(c) for c in corpus]
    vocab    = sorted(set(t for ts in all_toks for t in ts))
    vidx     = {w:i for i,w in enumerate(vocab)}
    N        = len(corpus)
    df       = Counter(t for ts in all_toks for t in set(ts))
    mat      = np.zeros((N, len(vocab)), np.float32)
    for i, tokens in enumerate(all_toks):
        tf = Counter(tokens); total = max(len(tokens),1)
        for t, cnt in tf.items():
            if t in vidx:
                idf = np.log((N+1)/(df[t]+1))+1
                mat[i, vidx[t]] = (cnt/total)*idf
    norms = np.linalg.norm(mat,axis=1,keepdims=True)
    mat  /= np.where(norms==0,1,norms)
    return mat, vidx, df, N

mat, vidx, df, N = build_index(CHUNKS)

def embed_q(q):
    tokens = tokenise(q); tf = Counter(tokens); total = max(len(tokens),1)
    v = np.zeros(len(vidx), np.float32)
    for t, cnt in tf.items():
        if t in vidx:
            idf = np.log((N+1)/(df[t]+1))+1
            v[vidx[t]] = (cnt/total)*idf
    n = np.linalg.norm(v); return v/n if n>0 else v

# ── Stage 1: fast recall-oriented retrieval ───────────────────
def stage1_retrieve(query, top_n=8):
    qv     = embed_q(query)
    scores = mat @ qv
    idx    = np.argsort(scores)[::-1][:top_n]
    return [(i, CHUNKS[i], float(scores[i])) for i in idx]

# ── Stage 2: cross-encoder-style reranker (simulated) ─────────
# Real cross-encoders use bidirectional attention over (q, doc).
# We simulate it with: token overlap ratio * BM25-style term weight
def cross_encoder_score(query, doc):
    q_toks = set(tokenise(query))
    d_toks = tokenise(doc)
    if not q_toks or not d_toks: return 0.0
    # Weighted term match: IDF-weighted overlap
    score = 0.0
    for t in q_toks:
        if t in [w for w in d_toks]:
            idf = np.log((N+1)/(df.get(t,0)+1))+1
            count = d_toks.count(t)
            score += idf * (count / (count + 1.0))  # BM25-style saturation
    return score / len(q_toks)

def stage2_rerank(query, candidates, top_k=3):
    scored = [(idx, text, cross_encoder_score(query, text), s1) for idx, text, s1 in candidates]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]

# ── Compare naive vs reranked ──────────────────────────────────
queries = [
    "How does a two-stage pipeline improve retrieval precision?",
    "What is the difference between bi-encoder and cross-encoder?",
    "How do we evaluate RAG quality?",
]

print("Retrieve-and-Rerank vs Naive RAG")
print("=" * 64)

for query in queries:
    print(f"\\nQuery: '{query}'")
    candidates = stage1_retrieve(query, top_n=8)
    reranked   = stage2_rerank(query, candidates, top_k=3)

    print(f"  Stage 1 — Top 8 by vector similarity:")
    for rank, (idx, text, s1) in enumerate(candidates[:5], 1):
        rerank_pos = next((r+1 for r,(i,_,_,_) in enumerate(reranked) if i==idx), '-')
        print(f"    [{rank}] sim={s1:.4f} → rerank_pos={rerank_pos}: {text[:65]}...")

    print(f"  Stage 2 — Top 3 after reranking:")
    for rank, (idx, text, s2, s1) in enumerate(reranked, 1):
        shift = candidates.index(next(c for c in candidates if c[0]==idx)) + 1
        arrow = f"↑{shift-rank} promoted" if shift>rank else ("same" if shift==rank else f"↓{rank-shift}")
        print(f"    [{rank}] xenc={s2:.4f} (was rank {shift}, {arrow}): {text[:55]}...")
""",
    },

    "4. Graph RAG — Knowledge Graph Extraction and Traversal": {
        "description": "Simulate Graph RAG: extract entities and relationships from "
                       "documents to build a knowledge graph, then traverse it alongside "
                       "vector search to answer multi-hop questions.",
        "code": """\
import numpy as np
import re
from collections import Counter, defaultdict

# ── Knowledge Graph (hand-crafted to simulate LLM extraction) ─
# In real Graph RAG, an LLM extracts these triples from documents.
# Format: (subject, relation, object)
KG_TRIPLES = [
    ("RAG",              "uses",              "Vector Database"),
    ("RAG",              "uses",              "Embedding Model"),
    ("RAG",              "reduces",           "Hallucination"),
    ("Naive RAG",        "is_type_of",        "RAG"),
    ("Graph RAG",        "is_type_of",        "RAG"),
    ("Graph RAG",        "uses",              "Knowledge Graph"),
    ("Graph RAG",        "uses",              "Vector Database"),
    ("Knowledge Graph",  "contains",          "Entities"),
    ("Knowledge Graph",  "contains",          "Relationships"),
    ("Microsoft",        "developed",         "GraphRAG"),
    ("GraphRAG",         "improves",          "Multi-hop Questions"),
    ("GraphRAG",         "uses",              "Community Summaries"),
    ("Embedding Model",  "produces",          "Dense Vectors"),
    ("Dense Vectors",    "stored_in",         "Vector Database"),
    ("Cross-Encoder",    "used_in",           "Reranker"),
    ("Reranker",         "improves",          "Retrieval Precision"),
    ("Retrieve-Rerank",  "uses",              "Reranker"),
    ("Retrieve-Rerank",  "is_type_of",        "RAG"),
    ("LLM",              "used_in",           "RAG"),
    ("LLM",              "generates",         "Response"),
    ("RAG",              "grounds",           "LLM"),
]

# ── Build adjacency graph ──────────────────────────────────────
class KnowledgeGraph:
    def __init__(self, triples):
        self.out_edges  = defaultdict(list)  # node -> [(relation, target)]
        self.in_edges   = defaultdict(list)  # node -> [(relation, source)]
        self.all_nodes  = set()
        for s, r, o in triples:
            self.out_edges[s].append((r, o))
            self.in_edges[o].append((r, s))
            self.all_nodes.add(s); self.all_nodes.add(o)

    def neighbours(self, entity, depth=2):
        visited = {entity}; frontier = {entity}; facts = []
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for rel, target in self.out_edges[node]:
                    facts.append(f"{node} --[{rel}]--> {target}")
                    if target not in visited:
                        visited.add(target); next_frontier.add(target)
                for rel, source in self.in_edges[node]:
                    facts.append(f"{source} --[{rel}]--> {node}")
                    if source not in visited:
                        visited.add(source); next_frontier.add(source)
            frontier = next_frontier
        return list(dict.fromkeys(facts))  # deduplicate, preserve order

    def find_path(self, start, end, max_depth=3):
        from collections import deque
        queue = deque([(start, [start])]); visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == end: return path
            if len(path) > max_depth: continue
            for _, target in self.out_edges[node]:
                if target not in visited:
                    visited.add(target); queue.append((target, path + [target]))
        return None

kg = KnowledgeGraph(KG_TRIPLES)

# ── Entity extraction from query ──────────────────────────────
def extract_entities(query, known_entities):
    q_lower = query.lower()
    return [e for e in known_entities if e.lower() in q_lower]

# ── Hybrid retrieval: vector (TF-IDF) + graph ─────────────────
TEXT_CHUNKS = [
    "Naive RAG is the simplest form of retrieval-augmented generation.",
    "Graph RAG uses a knowledge graph to handle multi-hop questions better.",
    "The knowledge graph stores entities and relationships extracted by an LLM.",
    "Microsoft developed GraphRAG which uses community summaries for global queries.",
    "Reranking with a cross-encoder improves retrieval precision significantly.",
    "Dense vectors from embedding models are stored in vector databases.",
]

def tokenise(t): return re.findall(r'\b[a-z]{2,}\b', t.lower())

def build_tfidf(corpus):
    all_toks = [tokenise(c) for c in corpus]
    vocab    = sorted(set(t for ts in all_toks for t in ts))
    vidx     = {w:i for i,w in enumerate(vocab)}
    N        = len(corpus)
    df       = Counter(t for ts in all_toks for t in set(ts))
    mat      = np.zeros((N, len(vocab)), np.float32)
    for i, toks in enumerate(all_toks):
        tf = Counter(toks); tot = max(len(toks),1)
        for t,cnt in tf.items():
            if t in vidx: mat[i,vidx[t]] = (cnt/tot)*(np.log((N+1)/(df[t]+1))+1)
    norms = np.linalg.norm(mat,axis=1,keepdims=True)
    return mat / np.where(norms==0,1,norms), vidx, df, N

mat, vidx, df, N = build_tfidf(TEXT_CHUNKS)

def retrieve_vector(query, top_k=2):
    toks = tokenise(query); tf = Counter(toks); tot = max(len(toks),1)
    v = np.zeros(len(vidx),np.float32)
    for t,cnt in tf.items():
        if t in vidx: v[vidx[t]] = (cnt/tot)*(np.log((N+1)/(df.get(t,0)+1))+1)
    n = np.linalg.norm(v);  v = v/n if n>0 else v
    scores = mat@v
    idx    = np.argsort(scores)[::-1][:top_k]
    return [TEXT_CHUNKS[i] for i in idx if scores[i]>0]

def hybrid_rag_retrieve(query):
    # Vector results
    vector_results = retrieve_vector(query)
    # Graph results
    entities = extract_entities(query, kg.all_nodes)
    graph_facts = []
    for ent in entities[:2]:
        graph_facts.extend(kg.neighbours(ent, depth=1)[:4])
    return vector_results, graph_facts[:6]

# ── Demo ───────────────────────────────────────────────────────
questions = [
    "What does GraphRAG use to answer questions?",
    "How does Graph RAG relate to the Vector Database?",
    "What did Microsoft develop?",
]

print("Graph RAG — Knowledge Graph + Vector Retrieval")
print("=" * 58)

for q in questions:
    vec, graph = hybrid_rag_retrieve(q)
    entities   = extract_entities(q, kg.all_nodes)
    print(f"\\nQ: {q}")
    print(f"  Entities detected: {entities if entities else ['(none found)']}")
    print(f"  Vector chunks retrieved:")
    for c in vec: print(f"    → {c[:75]}...")
    print(f"  Graph facts retrieved:")
    for f in graph: print(f"    → {f}")

print("\\nPath finding demo:")
for start, end in [("RAG", "Response"), ("GraphRAG", "Vector Database")]:
    path = kg.find_path(start, end)
    print(f"  {start} → ... → {end}: {path}")
""",
    },

    "5. Hybrid RAG — Reciprocal Rank Fusion": {
        "description": "Implement Hybrid RAG that fuses results from two retrievers "
                       "(vector similarity + keyword BM25) using Reciprocal Rank Fusion. "
                       "Show how fusion outperforms either retriever alone.",
        "code": """\
import numpy as np
import re
from collections import Counter

CHUNKS = [
    "RAG pipelines combine dense vector retrieval with language model generation for grounded answers.",
    "BM25 is a term-frequency ranking function used in keyword search and full-text search engines.",
    "Reciprocal rank fusion combines ranked lists from multiple retrievers without score normalisation.",
    "Dense embeddings capture semantic meaning while sparse BM25 excels at exact keyword matching.",
    "Hybrid search uses both vector similarity and BM25 to improve recall for diverse query types.",
    "The RRF score for document d is: sum over retrievers of 1 / (rank_d + k), where k is a constant.",
    "Semantic search may miss documents with exact keywords if they are paraphrased differently.",
    "BM25 fails on synonyms and paraphrases but is reliable for product codes and proper nouns.",
    "Weaviate and Elasticsearch both support hybrid search with native BM25 and vector search.",
    "Normalising scores from different retrievers is difficult because their scales differ widely.",
]

def tokenise(t): return re.findall(r'\b[a-z]{2,}\b', t.lower())

# ── Retriever A: TF-IDF / Dense Vector ────────────────────────
def build_dense(corpus):
    all_toks = [tokenise(c) for c in corpus]
    vocab    = sorted(set(t for ts in all_toks for t in ts))
    vidx     = {w:i for i,w in enumerate(vocab)}
    N        = len(corpus)
    df       = Counter(t for ts in all_toks for t in set(ts))
    mat      = np.zeros((N, len(vocab)), np.float32)
    for i, toks in enumerate(all_toks):
        tf = Counter(toks); tot = max(len(toks),1)
        for t,cnt in tf.items():
            if t in vidx: mat[i,vidx[t]] = (cnt/tot)*(np.log((N+1)/(df[t]+1))+1)
    norms = np.linalg.norm(mat,axis=1,keepdims=True)
    return mat/np.where(norms==0,1,norms), vidx, df, N

dense_mat, vidx, df_dense, N = build_dense(CHUNKS)

def dense_retrieve(query, top_k=len(CHUNKS)):
    toks = tokenise(query); tf = Counter(toks); tot = max(len(toks),1)
    v = np.zeros(len(vidx),np.float32)
    for t,cnt in tf.items():
        if t in vidx: v[vidx[t]] = (cnt/tot)*(np.log((N+1)/(df_dense.get(t,0)+1))+1)
    n = np.linalg.norm(v); v = v/n if n>0 else v
    scores = dense_mat@v
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in ranked]

# ── Retriever B: BM25 (keyword) ────────────────────────────────
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.corpus = [tokenise(c) for c in corpus]
        self.N      = len(corpus)
        self.avgdl  = np.mean([len(d) for d in self.corpus])
        self.df     = Counter(t for d in self.corpus for t in set(d))

    def score(self, query_tokens, doc_idx):
        doc   = self.corpus[doc_idx]
        dl    = len(doc)
        tf_d  = Counter(doc)
        score = 0.0
        for t in query_tokens:
            if t not in tf_d: continue
            idf = np.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            tf_norm = (tf_d[t] * (self.k1 + 1)) / \
                      (tf_d[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += idf * tf_norm
        return score

    def retrieve(self, query, top_k=len(CHUNKS)):
        q_toks = tokenise(query)
        scores = [(i, self.score(q_toks, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

bm25 = BM25(CHUNKS)

# ── Reciprocal Rank Fusion ─────────────────────────────────────
def rrf(ranked_lists, k=60):
    \"\"\"
    RRF score = sum_over_lists 1 / (rank + k)
    k=60 is the standard constant (prevents high scores for top-1 docs)
    \"\"\"
    scores = Counter()
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: -x[1])

# ── Evaluation ─────────────────────────────────────────────────
queries_gt = [
    ("What is RRF and how is it calculated?",        [2, 5]),  # ground truth chunk indices
    ("Why does BM25 fail on semantic queries?",      [3, 7]),
    ("Which databases support hybrid search?",       [4, 8]),
    ("Why not just normalise scores from retrievers?",[9, 2]),
]

def precision_at_k(ranked_ids, ground_truth, k=3):
    top_k = set(i for i,_ in ranked_ids[:k])
    return len(top_k & set(ground_truth)) / k

print("Hybrid RAG — Reciprocal Rank Fusion")
print("=" * 58)
print(f"  k=60 (RRF constant), evaluating Precision@3")
print()
print(f"  {'Query':>42} | {'Dense P@3':>9} | {'BM25 P@3':>9} | {'RRF P@3':>9}")
print("  " + "-"*76)

dense_total=0; bm25_total=0; rrf_total=0
for query, gt in queries_gt:
    dense_ranked = dense_retrieve(query)
    bm25_ranked  = bm25.retrieve(query)
    rrf_ranked   = rrf([dense_ranked, bm25_ranked])

    p_dense = precision_at_k(dense_ranked, gt)
    p_bm25  = precision_at_k(bm25_ranked,  gt)
    p_rrf   = precision_at_k(rrf_ranked,   gt)
    dense_total+=p_dense; bm25_total+=p_bm25; rrf_total+=p_rrf
    winner = "← RRF wins" if p_rrf > max(p_dense,p_bm25) else ("tie" if p_rrf==max(p_dense,p_bm25) else "")
    print(f"  {query[:42]:>42} | {p_dense:>9.2%} | {p_bm25:>9.2%} | {p_rrf:>9.2%} {winner}")

n=len(queries_gt)
print(f"  {'AVERAGE':>42} | {dense_total/n:>9.2%} | {bm25_total/n:>9.2%} | {rrf_total/n:>9.2%}")
print()
print("  RRF formula: score(d) = Σ  1 / (rank_d_in_list_i  +  k=60)")
print("  No score normalisation needed — ranks are directly comparable.")
""",
    },

    "6. Agentic RAG Router — Query Routing Logic": {
        "description": "Implement an Agentic RAG router that classifies queries and "
                       "selects the appropriate retrieval tool. Shows how different "
                       "query types are dispatched to different data sources.",
        "code": """\
import numpy as np
import re
from collections import Counter
from datetime import datetime

# ── Tool Registry ──────────────────────────────────────────────
class VectorSearchTool:
    name = "vector_search"
    description = "Semantic search over internal knowledge base documents."
    def run(self, query):
        return f"[Vector DB] Retrieved 3 relevant chunks about: '{query[:40]}...'"

class SQLDatabaseTool:
    name = "sql_database"
    description = "Query structured data: sales, metrics, tables, numbers, comparisons."
    def run(self, query):
        if "revenue" in query.lower():
            return "[SQL DB] Q3 Revenue: $4.2M (+12% YoY), Q4 Revenue: $4.8M (+14% YoY)"
        return f"[SQL DB] Executed query for: '{query[:40]}'"

class WebSearchTool:
    name = "web_search"
    description = "Real-time web search for current events, news, and live information."
    def run(self, query):
        return f"[Web Search] Live results retrieved for: '{query[:40]}'"

class DocumentAnalysisTool:
    name = "document_analysis"
    description = "Analyse an uploaded document or file provided by the user."
    def run(self, query):
        return f"[Doc Analysis] Processed uploaded document for: '{query[:40]}'"

class CodeExecutionTool:
    name = "code_execution"
    description = "Execute code, calculations, data transformations, or generate charts."
    def run(self, query):
        return f"[Code Exec] Ran computation for: '{query[:40]}'"

TOOLS = {
    t.name: t() for t in [
        VectorSearchTool, SQLDatabaseTool, WebSearchTool,
        DocumentAnalysisTool, CodeExecutionTool
    ]
}

# ── Router: Rule-based + pattern matching (simulates LLM routing) ──
class RAGRouter:
    \"\"\"
    In production this would be an LLM with function-calling capability.
    Here we simulate routing with keyword patterns and query classification.
    \"\"\"
    ROUTING_RULES = [
        # (pattern, tool_name, confidence, reason)
        (r'\\b(today|current|latest|news|2024|2025|now|right now|live)\\b',
         'web_search', 0.95, 'real-time information required'),
        (r'\\b(revenue|sales|metrics|kpi|quarter|q[1-4]|compare|growth|trend|chart|graph|plot)\\b',
         'sql_database', 0.92, 'structured/numerical data query'),
        (r'\\b(uploaded|attached|this (doc|file|pdf)|provided document|analyse this)\\b',
         'document_analysis', 0.97, 'explicit document reference'),
        (r'\\b(calculate|compute|code|script|python|formula|convert|transform|generate chart)\\b',
         'code_execution', 0.90, 'computation or code required'),
        (r'.',  # catch-all
         'vector_search', 0.70, 'default semantic search'),
    ]

    def route(self, query):
        q_lower = query.lower()
        for pattern, tool, confidence, reason in self.ROUTING_RULES:
            if re.search(pattern, q_lower, re.IGNORECASE):
                return tool, confidence, reason
        return 'vector_search', 0.70, 'default fallback'

    def execute(self, query):
        tool_name, confidence, reason = self.route(query)
        tool   = TOOLS[tool_name]
        result = tool.run(query)
        return {
            'query':      query,
            'tool':       tool_name,
            'confidence': confidence,
            'reason':     reason,
            'result':     result,
        }

# ── Fallback strategy ──────────────────────────────────────────
class AgenticRAG:
    def __init__(self, router, fallback_tool='vector_search'):
        self.router       = router
        self.fallback     = TOOLS[fallback_tool]
        self.call_log     = []

    def query(self, question):
        result = self.router.execute(question)
        self.call_log.append(result)
        # If confidence is low, also run vector search as backup
        if result['confidence'] < 0.80 and result['tool'] != 'vector_search':
            fallback = TOOLS['vector_search'].run(question)
            result['fallback_result'] = fallback
        return result

router = RAGRouter()
agent  = AgenticRAG(router)

test_queries = [
    "What is our refund policy?",
    "What were Q3 and Q4 revenue numbers and the growth rate?",
    "What happened in the tech industry news today?",
    "Can you calculate the compound annual growth rate from 2020 to 2024?",
    "Analyse the uploaded financial report and summarise key findings.",
    "How does our product work under the hood?",
    "Compare Q3 vs Q4 sales by region and plot a chart.",
    "What is the latest version of Python released?",
]

print("Agentic RAG Router — Query Dispatch")
print("=" * 68)
print()
print(f"  {'Query':<45} | {'Tool':<20} | {'Conf':>6} | {'Reason'}")
print("  " + "-"*90)

tool_counts = Counter()
for q in test_queries:
    res = agent.query(q)
    tool_counts[res['tool']] += 1
    fb  = " + vector fallback" if 'fallback_result' in res else ""
    print(f"  {q[:45]:<45} | {res['tool']:<20} | {res['confidence']:>6.0%} | {res['reason']}{fb}")

print()
print("  Tool usage distribution:")
for tool, count in sorted(tool_counts.items(), key=lambda x:-x[1]):
    bar = '█' * count
    print(f"    {tool:<20}: {count}  {bar}")

print()
print("  Key insight: the router selects the OPTIMAL tool per query type.")
print("  Real implementation: replace routing rules with LLM function-calling.")
print("  Benefit: no retrieval at all for queries that don't need it (computation).")
""",
    },

    "7. Multi-Agent RAG — Parallel Specialised Agents": {
        "description": "Implement a Multi-Agent RAG orchestrator that decomposes a "
                       "complex query into sub-tasks, dispatches them to specialised "
                       "agents in parallel, and synthesises the results.",
        "code": """\
import numpy as np
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# ── Specialised Agent Knowledge Bases ─────────────────────────
INTERNAL_DOCS = {
    "product": "Our product uses RAG architecture with a vector database. "
               "Version 3.2 was released last quarter with multimodal support. "
               "Pricing starts at $99/month for the starter plan.",
    "policy":  "Refunds are available within 30 days. Data is stored in EU region. "
               "SLA guarantees 99.9% uptime. Support responds within 4 hours.",
}

EXTERNAL_KNOWLEDGE = {
    "rag":     "Retrieve-and-Rerank improves RAG precision by adding a cross-encoder stage. "
               "GraphRAG by Microsoft uses knowledge graphs for multi-hop reasoning.",
    "market":  "The vector database market is growing at 25% annually. "
               "Major players include Pinecone, Weaviate, Chroma, and pgvector.",
}

WEB_RESULTS = {
    "news":    "Latest AI news: OpenAI released GPT-5, Anthropic released Claude 4. "
               "Vector databases gaining enterprise adoption. RAG market projected $10B by 2026.",
    "pricing": "Competitor A: $89/month. Competitor B: $120/month. Competitor C: $75/month.",
}

CRM_DATA = {
    "customers": "Total customers: 1,240. Churn rate: 3.2%. NPS: 72. "
                 "Top segment: Enterprise (45% of revenue). SMB growing 18% QoQ.",
    "support":   "Open tickets: 23. Avg resolution time: 6.2 hours. "
                 "Top issue: integration setup (38% of tickets).",
}

# ── Specialised Agents ─────────────────────────────────────────
class Agent:
    def __init__(self, name, knowledge_base, speciality):
        self.name     = name
        self.kb       = knowledge_base
        self.speciality = speciality

    def search(self, query):
        q_lower = query.lower()
        results = []
        for key, content in self.kb.items():
            relevance = sum(1 for w in q_lower.split() if w in content.lower())
            if relevance > 0:
                results.append((relevance, content))
        results.sort(reverse=True)
        return results[0][1] if results else f"No relevant data found in {self.name}."

    def run(self, sub_query, delay_sim=0.0):
        time.sleep(delay_sim)  # simulate network latency
        result = self.search(sub_query)
        return {
            'agent':     self.name,
            'speciality': self.speciality,
            'sub_query': sub_query,
            'result':    result,
        }

agents = {
    'internal_docs': Agent("Internal Docs Agent", INTERNAL_DOCS, "Product & Policy"),
    'external_kb':   Agent("External KB Agent",   EXTERNAL_KNOWLEDGE, "Industry Knowledge"),
    'web_search':    Agent("Web Search Agent",     WEB_RESULTS, "Real-Time News"),
    'crm':           Agent("CRM Agent",            CRM_DATA, "Customer Intelligence"),
}

# ── Orchestrator ───────────────────────────────────────────────
class OrchestratorAgent:
    def __init__(self, agents):
        self.agents = agents

    def decompose(self, query):
        \"\"\"Break a complex query into sub-queries per agent.
        In production: an LLM generates this decomposition.
        Here: rule-based simulation.
        \"\"\"
        sub_queries = {}
        q = query.lower()
        if any(w in q for w in ['product', 'feature', 'version', 'plan', 'price', 'policy']):
            sub_queries['internal_docs'] = f"Find internal info about: {query}"
        if any(w in q for w in ['rag', 'vector', 'market', 'technology', 'competitor']):
            sub_queries['external_kb'] = f"Find industry knowledge about: {query}"
        if any(w in q for w in ['news', 'latest', 'recent', 'today', 'current']):
            sub_queries['web_search'] = f"Search web for: {query}"
        if any(w in q for w in ['customer', 'churn', 'support', 'nps', 'ticket', 'crm']):
            sub_queries['crm'] = f"Query CRM data for: {query}"
        # Default: send to all agents
        if not sub_queries:
            for k in self.agents:
                sub_queries[k] = query
        return sub_queries

    def synthesise(self, query, agent_results):
        \"\"\"Combine all agent results into a coherent summary.
        In production: an LLM synthesises this.
        \"\"\"
        lines = [f"Synthesised answer for: '{query}'", ""]
        for res in agent_results:
            lines.append(f"  [{res['agent']} | {res['speciality']}]")
            lines.append(f"    {res['result'][:120]}...")
            lines.append("")
        return '\\n'.join(lines)

    def run_parallel(self, query):
        sub_queries = self.decompose(query)
        t0 = time.time()

        # Run selected agents in parallel
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(self.agents[agent_key].run, sq): agent_key
                for agent_key, sq in sub_queries.items()
                if agent_key in self.agents
            }
            results = [f.result() for f in futures]

        elapsed = time.time() - t0
        synthesis = self.synthesise(query, results)
        return results, synthesis, elapsed, sub_queries

orchestrator = OrchestratorAgent(agents)

# ── Demo queries ───────────────────────────────────────────────
queries = [
    "Compare our product pricing with competitors and summarise customer satisfaction.",
    "What are the latest RAG technology developments and how does our product compare?",
    "What is our refund policy and what are customers saying about support?",
]

print("Multi-Agent RAG — Parallel Orchestration")
print("=" * 64)

for query in queries:
    results, synthesis, elapsed, sub_queries = orchestrator.run_parallel(query)
    print()
    print(f"Query: '{query}'")
    print(f"  Agents activated: {list(sub_queries.keys())}")
    print(f"  Parallel execution time: {elapsed*1000:.1f}ms")
    print(f"  Results from {len(results)} agents:")
    for res in results:
        print(f"    [{res['agent']}]: {res['result'][:80]}...")

print()
print("Key metrics:")
print(f"  Max agents in parallel: {len(agents)}")
print(f"  Parallel vs sequential speedup: ~{len(agents)}x for independent agents")
print("  Real implementation: use LangGraph, AutoGen, or CrewAI for orchestration")
""",
    },

    "8. RAG Pattern Selector — Decision Guide with Benchmarks": {
        "description": "Interactive decision guide that recommends the optimal RAG "
                       "pattern given query type, data characteristics, and constraints. "
                       "Includes a benchmark comparison across all 7 patterns.",
        "code": """\
import numpy as np

# ── Pattern definitions ────────────────────────────────────────
PATTERNS = {
    "Naive RAG": {
        "description": "Single-stage vector similarity retrieval → LLM",
        "latency_ms":  (80,  200),   # (min, max) milliseconds
        "precision":    0.65,
        "recall":       0.72,
        "setup_days":   1,
        "cost_per_1k":  0.02,        # USD per 1000 queries
        "strengths":   ["Fastest", "Simplest", "Cheapest"],
        "weaknesses":  ["Lower precision", "Keyword mismatch", "No reranking"],
        "best_for":    ["Prototypes", "Simple Q&A", "Small corpus"],
        "data_types":  ["text"],
        "query_types": ["factual", "semantic"],
    },
    "Retrieve-Rerank": {
        "description": "Vector retrieval top-N → cross-encoder reranker → LLM",
        "latency_ms":  (150, 400),
        "precision":    0.82,
        "recall":       0.75,
        "setup_days":   2,
        "cost_per_1k":  0.06,
        "strengths":   ["High precision", "Filters noise", "Robust"],
        "weaknesses":  ["Extra latency", "Reranker cost"],
        "best_for":    ["Legal", "Medical", "Compliance", "Large corpus"],
        "data_types":  ["text"],
        "query_types": ["factual", "semantic", "precise"],
    },
    "Multimodal RAG": {
        "description": "Cross-modal embedding (CLIP/ImageBind) → multimodal LLM",
        "latency_ms":  (200, 600),
        "precision":    0.78,
        "recall":       0.80,
        "setup_days":   5,
        "cost_per_1k":  0.15,
        "strengths":   ["Handles images/video", "Cross-modal search"],
        "weaknesses":  ["Complex indexing", "Expensive vision models"],
        "best_for":    ["Product catalogs", "Medical imaging", "Technical manuals"],
        "data_types":  ["text", "image", "video", "audio"],
        "query_types": ["visual", "multimodal", "factual"],
    },
    "Graph RAG": {
        "description": "KG extraction → graph traversal + vector search → LLM",
        "latency_ms":  (200, 500),
        "precision":    0.85,
        "recall":       0.70,
        "setup_days":   10,
        "cost_per_1k":  0.12,
        "strengths":   ["Multi-hop reasoning", "Relationship queries", "Structured"],
        "weaknesses":  ["High indexing cost", "Graph maintenance"],
        "best_for":    ["Enterprise KG", "Drug interactions", "Org charts"],
        "data_types":  ["text", "structured"],
        "query_types": ["relational", "multi-hop", "structural"],
    },
    "Hybrid RAG": {
        "description": "Vector DB + Graph DB → RRF fusion → LLM",
        "latency_ms":  (300, 700),
        "precision":    0.88,
        "recall":       0.84,
        "setup_days":   14,
        "cost_per_1k":  0.18,
        "strengths":   ["Best accuracy", "Covers all query types", "Robust"],
        "weaknesses":  ["Most complex", "Highest infra cost"],
        "best_for":    ["Enterprise", "Healthcare", "Financial research"],
        "data_types":  ["text", "structured", "relational"],
        "query_types": ["all"],
    },
    "Agentic RAG": {
        "description": "AI router → selects tool (vector/SQL/web/code) → LLM",
        "latency_ms":  (200, 800),
        "precision":    0.80,
        "recall":       0.82,
        "setup_days":   7,
        "cost_per_1k":  0.10,
        "strengths":   ["Adaptive routing", "Skips retrieval when not needed"],
        "weaknesses":  ["Non-deterministic", "Harder to debug"],
        "best_for":    ["Diverse query types", "Multi-source knowledge"],
        "data_types":  ["text", "structured", "web", "code"],
        "query_types": ["diverse", "adaptive", "computational"],
    },
    "Multi-Agent RAG": {
        "description": "Orchestrator → N parallel specialised agents → synthesis",
        "latency_ms":  (300, 900),
        "precision":    0.90,
        "recall":       0.88,
        "setup_days":   21,
        "cost_per_1k":  0.35,
        "strengths":   ["Parallel execution", "Domain specialisation", "Highest quality"],
        "weaknesses":  ["Highest cost", "Most complex", "Orchestration overhead"],
        "best_for":    ["Competitive intelligence", "Complex research", "Enterprise"],
        "data_types":  ["text", "structured", "web", "image", "code"],
        "query_types": ["complex", "multi-source", "synthesis"],
    },
}

def recommend(query_type, data_types, latency_budget_ms, precision_need, budget_usd):
    scores = {}
    for name, p in PATTERNS.items():
        score = 0.0
        # Query type match
        if "all" in p["query_types"] or query_type in p["query_types"]:
            score += 30
        # Data type support
        data_covered = sum(1 for d in data_types if d in p["data_types"])
        score += 20 * (data_covered / max(len(data_types), 1))
        # Latency constraint
        if p["latency_ms"][0] <= latency_budget_ms:
            score += 15 * (1 - p["latency_ms"][0] / (latency_budget_ms + 1))
        # Precision match
        score += 20 * (1 - abs(p["precision"] - precision_need))
        # Cost constraint
        if p["cost_per_1k"] <= budget_usd:
            score += 15 * (1 - p["cost_per_1k"] / (budget_usd + 0.01))
        scores[name] = score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked

# ── Pattern benchmark table ────────────────────────────────────
print("RAG Pattern Benchmark Comparison")
print("=" * 84)
print()
print(f"  {'Pattern':<22} | {'Precision':>9} | {'Recall':>7} | "
      f"{'Latency':>12} | {'Setup':>6} | {'$/1K':>6}")
print("  " + "-"*76)
for name, p in PATTERNS.items():
    lat = f"{p['latency_ms'][0]}-{p['latency_ms'][1]}ms"
    print(f"  {name:<22} | {p['precision']:>9.0%} | {p['recall']:>7.0%} | "
          f"{lat:>12} | {p['setup_days']:>4}d | {p['cost_per_1k']:>5.2f}")

# ── Decision scenarios ─────────────────────────────────────────
scenarios = [
    {
        "name":          "Startup prototype",
        "query_type":    "factual",
        "data_types":    ["text"],
        "latency_ms":    500,
        "precision":     0.70,
        "budget":        0.05,
    },
    {
        "name":          "Legal document review",
        "query_type":    "precise",
        "data_types":    ["text"],
        "latency_ms":    1000,
        "precision":     0.90,
        "budget":        0.20,
    },
    {
        "name":          "Enterprise knowledge base",
        "query_type":    "all",
        "data_types":    ["text", "structured", "relational"],
        "latency_ms":    2000,
        "precision":     0.88,
        "budget":        0.50,
    },
    {
        "name":          "Product catalog (images)",
        "query_type":    "visual",
        "data_types":    ["text", "image"],
        "latency_ms":    800,
        "precision":     0.80,
        "budget":        0.20,
    },
]

print()
print("Scenario-Based Recommendations:")
print("=" * 84)
for s in scenarios:
    ranked = recommend(s["query_type"], s["data_types"], s["latency_ms"],
                       s["precision"], s["budget"])
    print(f"\\n  Scenario: {s['name']}")
    print(f"  Constraints: query={s['query_type']}, data={s['data_types']}, "
          f"latency<{s['latency_ms']}ms, precision>{s['precision']:.0%}, cost<${s['budget']}/1K")
    print(f"  Recommendations:")
    for rank, (name, score) in enumerate(ranked[:3], 1):
        p = PATTERNS[name]
        print(f"    {rank}. {name:<22} score={score:.1f}  "
              f"(precision={p['precision']:.0%}, "
              f"latency={p['latency_ms'][0]}-{p['latency_ms'][1]}ms)")
""",
    },

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
    visual_height = 1300
    try:
        from Generative_AI.visuals.rag_pattern_visual import (   # ← match your exact folder casing
            RAG_VISUAL_HTML,
            RAG_VISUAL_HEIGHT,
        )
        visual_html   = RAG_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = RAG_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"Could not load visual: {e}", stacklevel=2)

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