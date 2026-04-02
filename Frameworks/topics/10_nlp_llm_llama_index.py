"""
LlamaIndex — The Data Framework for LLM Applications
=====================================================

LlamaIndex (originally called GPT Index) was created by Jerry Liu in
November 2022 and open-sourced the same month. It emerged from a simple
frustration: the existing tools for connecting large language models to
external data were too generic. LlamaIndex was purpose-built for one job
done extremely well — building production-grade retrieval and question-
answering systems over private, structured, and unstructured data.

Where LangChain is a general-purpose LLM application framework, LlamaIndex
is a data framework. Every architectural decision is optimised for one
question: "How do you get an LLM to reason accurately and efficiently over
YOUR data?" The result is a library with far deeper indexing primitives,
more sophisticated retrieval strategies, and richer query engines than any
general-purpose alternative.

The core insight behind LlamaIndex: the quality of an LLM application over
private data is determined primarily by the quality of its retrieval, not
the quality of its generation. A state-of-the-art LLM given bad context
will produce bad answers. A smaller model given excellent, precisely
retrieved context will outperform it. LlamaIndex focuses its entire
architecture on maximising retrieval quality.

LlamaIndex 0.10+ (released 2024) introduced a major architectural redesign
into two independent packages:
    llama-index-core:       base abstractions, pipeline components
    llama-index-readers-*:  150+ data connectors (SimpleDirectoryReader,
                            Notion, S3, Google Drive, Confluence, GitHub, ...)
    llama-index-vector-stores-*: integrations for every vector database
    llama-index-llms-*:     LLM provider wrappers (OpenAI, Anthropic, Ollama)
    llama-index-embeddings-*: embedding model wrappers
    llama-index-postprocessor-*: re-rankers and node post-processors

This module covers the complete LlamaIndex stack: the Document/Node data
model, the indexing pipeline, the five index types and when to use each,
the query engine hierarchy, the retriever abstraction, response synthesis
strategies, the QueryPipeline composition system, sub-question decomposition,
knowledge graphs, multi-modal indexing, and production evaluation with
the RAGAS framework.

"""

import textwrap
import re

TOPIC_NAME   = "LlamaIndex — The Data Framework for LLM Applications"
DISPLAY_NAME = "10 · LlamaIndex"
ICON         = "🦙"
SUBTITLE     = "From Document Ingestion to Advanced Query Engines and Agents"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — LLAMAINDEX PHILOSOPHY: DATA-CENTRIC LLM APPLICATIONS

### The Data Problem LlamaIndex Solves

    Every enterprise has data: PDFs, Word documents, wikis, code repositories,
    databases, Slack archives, CRM records. The challenge is not generating text —
    modern LLMs do that excellently. The challenge is making LLMs reason accurately
    over THIS specific data, in THIS specific organisation, updated in real time.

    Three failure modes of naive RAG (retrieve + generate):
        1. RETRIEVAL FAILURE: the right information is never retrieved because
           chunks are poorly segmented or the embedding model misses semantic intent.
        2. CONTEXT FAILURE: retrieved chunks are relevant but lack necessary
           surrounding context (the answer spans two chunks; one is retrieved).
        3. REASONING FAILURE: the LLM has the right context but fails to synthesise
           a coherent answer because the context is too long or poorly structured.

    LlamaIndex addresses all three with:
        - Sophisticated chunking strategies (sentence windows, hierarchical nodes)
        - Multiple index types optimised for different data structures
        - Re-ranking and post-processing pipelines
        - Response synthesis modes (tree summarise, accumulate, compact)
        - Structured extraction and knowledge graphs for complex reasoning

### LlamaIndex vs LangChain: The Right Tool

    LangChain:
        General-purpose LLM application framework.
        Excellent chains, agents, tool use, conversation management.
        RAG is one feature among many.

    LlamaIndex:
        Purpose-built data framework for LLMs.
        Deeper indexing, more retrieval strategies, richer query engines.
        Agents and pipelines are built on top of the data layer.

    Decision guide:
        Building a chatbot with tools, API integrations, complex agents?
        → LangChain (or LangGraph)

        Building a knowledge base Q&A, document search, or any system
        where retrieval quality is the primary concern?
        → LlamaIndex

        Complex production systems often use both:
        LlamaIndex for data/retrieval + LangChain for agent orchestration.

### The LlamaIndex Architecture Stack

    ┌─────────────────────────────────────────────────────────────────────┐
    │  APPLICATIONS                                                       │
    │  Chat engine · Query engine · Agent · Pipeline                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │  QUERY LAYER                                                        │
    │  Retriever · Node postprocessor · Response synthesiser              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  INDEX LAYER                                                        │
    │  VectorStoreIndex · SummaryIndex · TreeIndex · KnowledgeGraph       │
    ├─────────────────────────────────────────────────────────────────────┤
    │  INGESTION LAYER                                                    │
    │  Reader → Document → Transformation → Node · TextSplitter           │
    ├─────────────────────────────────────────────────────────────────────┤
    │  FOUNDATION                                                         │
    │  LLM · Embedding model · Vector store · ServiceContext              │
    └─────────────────────────────────────────────────────────────────────┘

### Global Configuration: Settings

    LlamaIndex 0.10+ uses a global Settings object instead of
    ServiceContext (which is deprecated):

        from llama_index.core import Settings
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        Settings.llm         = OpenAI(model="gpt-4o-mini", temperature=0)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.chunk_size  = 512
        Settings.chunk_overlap = 64

    Once set, ALL components created after that point use these defaults.
    Individual components can override Settings with local values.

    Using local models (no API key needed):
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.llm         = Ollama(model="llama3.2", request_timeout=120)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )


##### PART 2 — THE DATA MODEL: DOCUMENTS AND NODES

### Document: The Atomic Input Unit

    A Document is a container for raw content before it is processed:

        from llama_index.core import Document

        doc = Document(
            text       = "LlamaIndex is a data framework...",
            metadata   = {
                "source":   "wiki.internal.company.com",
                "author":   "Jerry Liu",
                "date":     "2024-01-15",
                "category": "ml_engineering",
            },
            doc_id     = "doc_001",    # unique identifier
            extra_info = {...},        # alternative metadata field
        )

    Document sources — SimpleDirectoryReader handles most common formats:
        from llama_index.core import SimpleDirectoryReader

        documents = SimpleDirectoryReader("./data/").load_data()
        # Supports: PDF, DOCX, TXT, MD, CSV, HTML, EPUB, images, ...

        # Single file:
        documents = SimpleDirectoryReader(input_files=["report.pdf"]).load_data()

        # With metadata extraction:
        documents = SimpleDirectoryReader(
            "./docs/",
            file_metadata=lambda path: {"source": str(path), "type": "internal"},
        ).load_data()

### Node: The Unit of Retrieval

    After processing, Documents are split into Nodes. Nodes are the units
    that get embedded, stored in the index, and retrieved at query time.

    TextNode is the most common Node type:
        from llama_index.core.schema import TextNode, NodeRelationship

        node = TextNode(
            text       = "chunk of text content...",
            id_        = "node_001",
            metadata   = {"source": "report.pdf", "page": 3},
            embedding  = [0.1, 0.3, ...],   # populated after embedding
        )

    Node relationships — the graph of the document:
        Each node knows its parent document and its siblings:

        NodeRelationship.SOURCE:      the parent Document
        NodeRelationship.PREVIOUS:    the preceding Node in the document
        NodeRelationship.NEXT:        the following Node in the document
        NodeRelationship.PARENT:      parent node (in hierarchical indexing)
        NodeRelationship.CHILD:       child nodes (in hierarchical indexing)

    Why relationships matter:
        When you retrieve a node, you can access its context — the nodes
        before and after it in the original document. This is the foundation
        of the SentenceWindowNodeParser and AutoMergingRetriever.

### The Ingestion Pipeline

    IngestionPipeline processes Documents into Nodes with a configurable
    transformation chain:

        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=1024, chunk_overlap=200),
                TitleExtractor(nodes=5),               # extract title metadata
                QuestionsAnsweredExtractor(questions=3),  # generate Q metadata
                embed_model,                           # compute embeddings
            ]
        )
        nodes = pipeline.run(documents=documents)

    Built-in transformations:
        SentenceSplitter:           splits on sentence boundaries (primary splitter)
        TokenTextSplitter:          splits by token count (precise for model limits)
        SentenceWindowNodeParser:   creates small nodes + large window for context
        HierarchicalNodeParser:     creates parent/child node trees
        TitleExtractor:             extracts document title via LLM
        SummaryExtractor:           generates summary per chunk via LLM
        QuestionsAnsweredExtractor: generates questions each chunk answers
        KeywordExtractor:           extracts keywords from each chunk

    IngestionPipeline with persistence (avoid reprocessing):
        pipeline.persist("./pipeline_storage")
        pipeline = IngestionPipeline.from_persist_dir("./pipeline_storage")


##### PART 3 — INDEX TYPES: FIVE WAYS TO ORGANISE YOUR DATA

### Index 1: VectorStoreIndex (Most Common)

    Stores document nodes as embedding vectors. Uses approximate nearest-neighbour
    (ANN) search to find semantically similar nodes at query time.

        from llama_index.core import VectorStoreIndex

        index = VectorStoreIndex.from_documents(documents)
        # Internally: splits → embeds → stores in default FAISS/SimpleVectorStore

    Construction:
        VectorStoreIndex.from_documents(docs)   → from Document objects
        VectorStoreIndex(nodes)                  → from pre-processed Nodes
        VectorStoreIndex.from_vector_store(vs)   → from existing vector store

    Persistent vector stores:
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb

        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

    When to use VectorStoreIndex:
        Semantic search over any unstructured text.
        The default choice for document Q&A.
        Works well for: FAQs, technical docs, knowledge bases, reports.

### Index 2: SummaryIndex (Sequential / Full Document)

    Stores all nodes as a sequential list. At query time, iterates through
    ALL nodes to synthesise an answer. Does NOT use embedding similarity.

        from llama_index.core import SummaryIndex

        index = SummaryIndex.from_documents(documents)

    Use cases:
        - Questions that require reading the FULL document (summarisation)
        - When the answer could be anywhere in the document
        - Small document collections where full scan is feasible

    Query modes:
        Default:  iterates all nodes, uses LLM to filter + summarise
        Embedding: embeds query, finds top-k nodes, then synthesises
        Async:    parallel processing of all nodes for large docs

    When to use SummaryIndex:
        "Summarise this entire document."
        "What are the key themes across these 10 papers?"
        The document is short enough to process entirely.

### Index 3: TreeIndex (Hierarchical Summarisation)

    Builds a tree of summarisations bottom-up:
        Leaf nodes: individual document chunks
        Parent nodes: summaries of their children
        Root node: summary of the entire document

    Construction:
        from llama_index.core import TreeIndex

        index = TreeIndex.from_documents(documents)

    Query traversal (top-down):
        Given a query, starts at the root summary.
        Uses LLM to decide which child subtrees are most relevant.
        Recursively descends until reaching leaf nodes.
        More token-efficient than SummaryIndex for large documents.

    When to use TreeIndex:
        Long documents where full-scan is too expensive.
        Hierarchical content (books with chapters, reports with sections).
        Multi-hop reasoning (needs to navigate document structure).

### Index 4: KeywordTableIndex

    Builds an inverted index: keyword → list of nodes containing that keyword.
    Uses LLM or RAKE/TFIDF to extract keywords from each node.

        from llama_index.core import KeywordTableIndex

        index = KeywordTableIndex.from_documents(documents)

    At query time:
        Extracts keywords from the query.
        Looks up nodes containing those keywords.
        Synthesises answer from retrieved nodes.

    When to use KeywordTableIndex:
        Exact match or keyword-heavy queries ("what does the API return for error code 404?")
        Complementary to vector search in hybrid retrieval.
        When semantic search misses precise technical terms.

### Index 5: KnowledgeGraphIndex

    Extracts entities and relationships from text, builds a knowledge graph.
    Stores: (subject, predicate, object) triplets.
    Combines graph traversal with vector search for complex reasoning.

        from llama_index.core import KnowledgeGraphIndex

        index = KnowledgeGraphIndex.from_documents(
            documents,
            max_triplets_per_chunk=10,
            include_embeddings=True,
        )

    At query time:
        Embeds the query, finds related entities.
        Traverses the graph to find relevant sub-graphs.
        Synthesises answer from graph context + text context.

    When to use KnowledgeGraphIndex:
        Questions about relationships ("Who founded X company?")
        Questions requiring multi-hop reasoning ("Who manages the team that built Y?")
        Scientific literature with entities and interactions.

### Index Selection Guide

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Task                              │ Index Type        │ Why          │
    ├──────────────────────────────────────────────────────────────────────┤
    │ General document Q&A              │ VectorStoreIndex  │ Default best │
    │ Summarise entire document         │ SummaryIndex      │ Full scan    │
    │ Questions about relationships     │ KnowledgeGraph    │ Graph search │
    │ Exact keyword / code search       │ KeywordTableIndex │ Inverted idx │
    │ Long document, hierarchical nav   │ TreeIndex         │ Top-down nav │
    │ Large scale + metadata filtering  │ VectorStore + SQL │ Hybrid       │
    └──────────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — NODE PARSERS: ADVANCED CHUNKING STRATEGIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why Chunking Strategy Matters More Than Model Choice

    The single most impactful decision in RAG quality is often NOT the LLM
    or the embedding model — it is the chunking strategy. Poor chunking
    breaks semantic coherence across chunk boundaries, losing exactly the
    information needed to answer a query.

    The fundamental tension:
        Small chunks: precise retrieval but lack context.
        Large chunks: rich context but imprecise retrieval (noisy).
        Hierarchical: retrieve small, return large = best of both.

### SentenceSplitter (The Baseline)

    Splits on sentence boundaries while respecting chunk_size.
    The most natural split — sentences are semantic units.

        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes    = splitter.get_nodes_from_documents(documents)

    Parameters:
        chunk_size:       target token count per chunk (not characters!)
        chunk_overlap:    tokens shared between consecutive chunks
        paragraph_separator: split preference (default: "\\n\\n")
        chunking_tokenizer_fn: custom tokeniser function

### SentenceWindowNodeParser (High-Quality Retrieval)

    Creates two sets of nodes from each sentence:
        1. SMALL node: just the single sentence (for embedding/retrieval)
        2. LARGE window: the sentence plus N sentences on each side (for context)

    At retrieval time, return the SMALL embedded node to the retriever
    (precise match), but inject the LARGE window into the LLM prompt
    (full context). This uses a MetadataReplacementPostProcessor.

        from llama_index.core.node_parser import SentenceWindowNodeParser
        from llama_index.core.postprocessor import MetadataReplacementPostProcessor

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size          = 3,    # 3 sentences on each side = window of 7
            window_metadata_key  = "window",
            original_text_metadata_key = "original_sentence",
        )
        nodes = node_parser.get_nodes_from_documents(documents)

        # The MetadataReplacementPostProcessor swaps the small node text
        # with the full window during response synthesis
        postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )

    Why this dramatically improves quality:
        Retrieved by semantic similarity of the sentence → high precision.
        LLM sees the surrounding context → enough information to answer.
        The best retrieval strategy for most document Q&A workloads.

### HierarchicalNodeParser (Auto-Merging)

    Creates nodes at multiple granularities simultaneously:
        - Large chunks (e.g. 2048 tokens) — for broad context
        - Medium chunks (e.g. 512 tokens) — for section-level context
        - Small chunks (e.g. 128 tokens) — for precise retrieval

    All three sizes are stored with parent-child relationships.
    At retrieval time, the AutoMergingRetriever checks: if most of a
    large chunk's children were retrieved, merge them into the parent
    and return the parent instead. This avoids sending fragmented context.

        from llama_index.core.node_parser import (
            HierarchicalNodeParser, get_leaf_nodes
        )

        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        all_nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(all_nodes)   # only the smallest (128-token) nodes

        # Index the leaf nodes for retrieval
        index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

        # At query time, AutoMergingRetriever promotes to parent if threshold met:
        from llama_index.core.retrievers import AutoMergingRetriever
        base_retriever = index.as_retriever(similarity_top_k=12)
        retriever = AutoMergingRetriever(
            base_retriever,
            storage_context,
            verbose=True,
            simple_ratio_thresh=0.5,    # merge if 50%+ of sibling nodes retrieved
        )

### MarkdownNodeParser and CodeSplitter

    MarkdownNodeParser:
        Splits on Markdown headers (##, ###).
        Preserves document hierarchy as node metadata.
        Excellent for documentation sites, README files, wikis.

        from llama_index.core.node_parser import MarkdownNodeParser
        nodes = MarkdownNodeParser().get_nodes_from_documents(md_docs)

    CodeSplitter:
        Splits code files respecting syntax boundaries (functions, classes).
        Uses tree-sitter for language-aware splitting.
        Supports: Python, JavaScript, TypeScript, Java, C++, Go, Rust, ...

        from llama_index.core.node_parser import CodeSplitter
        splitter = CodeSplitter(language="python", chunk_lines=50)
        code_nodes = splitter.get_nodes_from_documents(code_docs)

    SemanticSplitterNodeParser:
        Uses embedding similarity to detect semantic breaks between sentences.
        Splits where embedding cosine similarity drops below a threshold.
        Produces semantically coherent chunks regardless of fixed size.
        Expensive (requires embeddings during ingestion) but highest quality.

        from llama_index.core.node_parser import SemanticSplitterNodeParser
        splitter = SemanticSplitterNodeParser(
            buffer_size          = 1,        # number of sentences to consider at once
            breakpoint_percentile_threshold=95,  # similarity drop threshold
            embed_model          = embed_model,
        )


##### PART 5 — RETRIEVERS AND QUERY ENGINES

### The Retriever Abstraction

    A Retriever takes a query string and returns a list of NodeWithScore objects
    (each containing a TextNode and its relevance score):

        retriever = index.as_retriever(similarity_top_k=5)
        nodes     = retriever.retrieve("What is gradient descent?")
        for n in nodes:
            print(n.score, n.node.text[:80])

    All retrievers implement:
        .retrieve(query_str)   → list[NodeWithScore]

### Retriever Variants

    VectorIndexRetriever:
        Standard ANN similarity search. The default.
        index.as_retriever(similarity_top_k=5)

    VectorIndexAutoRetriever:
        Uses an LLM to infer vector DB metadata filters from the query.
        "Papers about transformers from 2023" → filter: {year: 2023} + embed
        Automatically structured and semantic search combined.

        from llama_index.core.retrievers import VectorIndexAutoRetriever
        from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

        vector_store_info = VectorStoreInfo(
            content_info="research papers on ML",
            metadata_info=[
                MetadataInfo(name="year",   type="int",    description="publication year"),
                MetadataInfo(name="author", type="str",    description="paper author"),
                MetadataInfo(name="venue",  type="str",    description="conference name"),
            ]
        )
        retriever = VectorIndexAutoRetriever(index, vector_store_info=vector_store_info)

    BM25Retriever:
        Keyword-based sparse retrieval. Fast, no embedding needed.
        Excellent for exact term matching, complementary to dense retrieval.

        from llama_index.retrievers.bm25 import BM25Retriever
        retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

    RouterRetriever:
        Uses an LLM to decide which retriever to use for a given query.
        Define multiple tools, each wrapping a different retriever or index.

        from llama_index.core.retrievers import RouterRetriever
        from llama_index.core.tools import RetrieverTool

        retriever = RouterRetriever.from_defaults(
            retriever_tools=[
                RetrieverTool.from_defaults(
                    retriever=vector_retriever,
                    description="Useful for general semantic questions.",
                ),
                RetrieverTool.from_defaults(
                    retriever=bm25_retriever,
                    description="Useful for exact keyword or technical queries.",
                ),
            ],
            llm=llm,
        )

    QueryFusionRetriever (Hybrid Search + Fusion):
        Generates multiple query variants, runs all against the retriever,
        deduplicates and re-ranks results using Reciprocal Rank Fusion (RRF).
        Dramatically improves recall over single-query retrieval.

        from llama_index.core.retrievers import QueryFusionRetriever

        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k     = 10,
            num_queries          = 4,        # generate 4 query variants
            mode                 = "reciprocal_rerank",  # RRF fusion
            use_async            = True,
        )

### Node Postprocessors: Refining Retrieved Nodes

    Postprocessors filter, re-rank, or transform the retrieved node list
    before it reaches the response synthesiser.

    SimilarityPostprocessor:
        Filter out nodes below a similarity threshold.
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]

    MetadataReplacementPostProcessor:
        Replace node text with a metadata field (e.g. sentence window).
        node_postprocessors=[MetadataReplacementPostProcessor(
            target_metadata_key="window")]

    LLMRerank:
        Use an LLM to re-score and re-order retrieved nodes by relevance.
        Most accurate reranker. Expensive (requires LLM call).
        node_postprocessors=[LLMRerank(choice_batch_size=5, top_n=3)]

    CohereRerank:
        Use Cohere's Rerank API for fast, accurate cross-encoder scoring.
        Far faster than LLMRerank. Excellent quality.
        node_postprocessors=[CohereRerank(api_key="...", top_n=5)]

    SentenceEmbeddingOptimizer:
        Keep only the sentences from each node most similar to the query.
        Reduces noise in context window. Improves answer quality.

    KeywordNodePostprocessor:
        Filter nodes by required or excluded keywords.
        node_postprocessors=[KeywordNodePostprocessor(
            required_keywords=["transformer", "attention"])]

    TimeWeightedPostprocessor:
        Boost recent documents in the relevance ranking.
        Useful for news, changelogs, time-sensitive domains.

### Query Engine Hierarchy

    Query engine = Retriever + Response Synthesiser + optional postprocessors.

    From index directly:
        query_engine = index.as_query_engine(
            similarity_top_k         = 5,
            response_mode            = "compact",
            node_postprocessors      = [reranker],
        )
        response = query_engine.query("What is attention?")
        print(response.response)
        print(response.source_nodes[0].node.metadata)

    Custom query engine:
        from llama_index.core.query_engine import RetrieverQueryEngine

        query_engine = RetrieverQueryEngine.from_args(
            retriever             = retriever,
            response_synthesizer  = synthesiser,
            node_postprocessors   = [similarity_cutoff, reranker],
        )

### Response Synthesis Modes

    The response synthesiser determines how the LLM uses retrieved nodes.

    "compact" (default, recommended):
        Pack as many nodes as fit into one prompt.
        Single LLM call if all nodes fit. Most efficient.

    "tree_summarize":
        Build a tree of summaries bottom-up. Summarise groups of nodes,
        then summarise the summaries. Final answer from root.
        Best for large contexts that don't fit in a single prompt.
        Most token-efficient for many nodes.

    "refine":
        Sequential: answer using node 1, then refine with node 2, etc.
        Each refine step sees previous answer + new node.
        Highest quality (most information integration) but most expensive.
        Good for: detailed technical questions requiring multiple sources.

    "accumulate":
        Generate answer for EACH node independently, then concatenate.
        Returns multiple answers (one per node). No cross-node synthesis.
        Good for: "List all X from the document" queries.

    "simple_summarize":
        Truncate all nodes into a single prompt. Fast but lossy.
        OK for: short answers, high-speed applications.

    "no_text":
        Returns nodes without any synthesis. Useful for debugging/inspection.


##### PART 6 — ADVANCED QUERY PATTERNS

### Sub-Question Query Engine

    Complex questions often require information from multiple documents or
    multiple parts of a large document. A single retrieval step misses this.

    SubQuestionQueryEngine:
        1. Decomposes the complex question into simpler sub-questions
        2. Routes each sub-question to the appropriate source engine
        3. Executes all sub-questions (optionally in parallel)
        4. Synthesises a final answer from all sub-answers

    Example:
        Question: "Compare the architecture and pretraining objectives of BERT and GPT"
        Sub-questions generated:
            → Q1: "What is the architecture of BERT?"     → BERT index
            → Q2: "What is BERT's pretraining objective?" → BERT index
            → Q3: "What is the architecture of GPT?"      → GPT index
            → Q4: "What is GPT's pretraining objective?"  → GPT index
        Final: synthesise comparison from all four answers.

        from llama_index.core.query_engine import SubQuestionQueryEngine
        from llama_index.core.tools import QueryEngineTool

        query_engine_tools = [
            QueryEngineTool.from_defaults(
                query_engine=bert_engine,
                name="bert_docs",
                description="Contains technical documentation about BERT.",
            ),
            QueryEngineTool.from_defaults(
                query_engine=gpt_engine,
                name="gpt_docs",
                description="Contains technical documentation about GPT.",
            ),
        ]

        sq_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            use_async=True,
        )

### RouterQueryEngine (Multi-Source Routing)

    Intelligently routes queries to the appropriate engine/index:

        from llama_index.core.query_engine import RouterQueryEngine
        from llama_index.core.selectors import LLMSingleSelector

        router = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                QueryEngineTool.from_defaults(
                    query_engine=summary_engine,
                    description="Use for summarisation requests.",
                ),
                QueryEngineTool.from_defaults(
                    query_engine=vector_engine,
                    description="Use for specific factual questions.",
                ),
                QueryEngineTool.from_defaults(
                    query_engine=sql_engine,
                    description="Use for numerical or structured data queries.",
                ),
            ],
        )

    Selectors:
        LLMSingleSelector:    LLM picks exactly one engine
        LLMMultiSelector:     LLM picks multiple engines (parallel)
        PydanticSingleSelector: structured output selector (more reliable)

### SQL + Vector Hybrid Query Engine

    NLSQLTableQueryEngine:
        Translates natural language to SQL, executes it, returns table results.
        Perfect for structured data alongside unstructured documents.

        from llama_index.core.query_engine import NLSQLTableQueryEngine

        sql_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=["users", "orders", "products"],
        )
        response = sql_engine.query("How many users signed up in January 2024?")

    SQLAutoVectorQueryEngine:
        Combines SQL query engine + vector query engine.
        Routes automatically to the right modality based on query type.
        "How many papers?" → SQL count query
        "What do papers about attention say?" → vector search

### QueryPipeline: Composable Query Processing

    QueryPipeline is LlamaIndex's equivalent of LCEL — a composable DAG
    of query processing components.

        from llama_index.core.query_pipeline import QueryPipeline, InputComponent

        pipeline = QueryPipeline(verbose=True)
        pipeline.add_modules({
            "input":     InputComponent(),
            "retriever": retriever,
            "reranker":  reranker,
            "synthesiser": synthesiser,
        })
        pipeline.add_link("input",     "retriever")
        pipeline.add_link("retriever", "reranker")
        pipeline.add_link("reranker",  "synthesiser")
        pipeline.add_link("input",     "synthesiser",
                           dest_key="query_str")  # also pass query to synthesiser

        response = pipeline.run(input="What is self-attention?")

### Chat Engine: Multi-Turn Conversation

    A chat engine maintains conversation history across multiple turns:

        chat_engine = index.as_chat_engine(
            chat_mode = "condense_plus_context",  # best for document Q&A
            memory    = ChatMemoryBuffer.from_defaults(token_limit=4096),
            verbose   = True,
        )

        response1 = chat_engine.chat("What is the Transformer architecture?")
        response2 = chat_engine.chat("Who invented it?")    # "it" resolved by memory
        response3 = chat_engine.chat("What year was this?")

        chat_engine.reset()   # clear conversation history

    Chat modes:
        "condense_question":   condenses chat history + question into standalone query
        "context":             retrieves context, inserts into system prompt
        "condense_plus_context": combines both — best quality, most calls
        "openai":              uses OpenAI's native function calling
        "best":                automatically picks the best mode for your LLM
        "simple":              no retrieval, just plain conversation with history


##### PART 7 — AGENTS IN LLAMAINDEX

### LlamaIndex Agents vs LangChain Agents

    Both frameworks implement the ReAct (Reason + Act) agent loop.
    The key difference: LlamaIndex agents are deeply integrated with
    the query engine and index system.

    A LlamaIndex agent can use:
        - QueryEngineTools (indexes as tools)
        - FunctionTools (arbitrary Python functions)
        - ToolSpecs (bundled multi-tool packages, e.g. Gmail, Slack, Jira)

### ReActAgent

    The standard, widely-compatible agent:

        from llama_index.core.agent import ReActAgent
        from llama_index.core.tools import QueryEngineTool, FunctionTool

        # Query engine as a tool
        rag_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="knowledge_base",
            description="Use to answer questions about company policies.",
        )

        # Python function as a tool
        def multiply(a: float, b: float) -> float:
            # "Multiply two numbers and return the result."
            return a * b

        calc_tool = FunctionTool.from_defaults(fn=multiply)

        agent = ReActAgent.from_tools(
            [rag_tool, calc_tool],
            llm     = llm,
            verbose = True,
            max_iterations = 10,
        )
        response = agent.chat("What is 5.5 times the number of employees mentioned in the policy?")

### OpenAIAgent (Function Calling)

    For OpenAI models, uses native function calling for more reliable
    tool selection compared to ReAct's text-based approach:

        from llama_index.agent.openai import OpenAIAgent

        agent = OpenAIAgent.from_tools(tools, llm=OpenAI("gpt-4o"))
        response = agent.chat("Compare revenue from Q1 and Q2 reports.")

### AgentRunner + AgentWorker (Programmatic Control)

    For fine-grained control over the agent execution:

        from llama_index.core.agent import AgentRunner

        # Step-by-step execution (for debugging or human-in-the-loop)
        task  = agent.create_task("What was the revenue trend in 2023?")
        step  = agent.run_step(task.task_id)   # runs one agent step

        while not step.is_last:
            step = agent.run_step(task.task_id)

        response = agent.finalize_response(task.task_id)


##### PART 8 — EVALUATION AND PRODUCTION BEST PRACTICES

### Why Evaluation Is Non-Negotiable

    Building a RAG pipeline without evaluation is building blind. You will
    not know whether a chunking change improved or hurt quality, whether a
    different embedding model is worth the cost, or whether your retrieval
    is finding the right documents.

    The three pillars of RAG evaluation:
        1. RETRIEVAL quality:  does retrieval find the right chunks?
        2. GENERATION quality: does the LLM produce accurate answers from context?
        3. END-TO-END quality: does the full system correctly answer questions?

### LlamaIndex Built-In Evaluators

    Faithfulness evaluator:
        Does the response contradict or add information not in the sources?
        (measures hallucination)

        from llama_index.core.evaluation import FaithfulnessEvaluator
        evaluator = FaithfulnessEvaluator(llm=llm)
        result = evaluator.evaluate_response(query="...", response=response)
        result.passing    # True/False
        result.score      # 0.0 to 1.0

    Relevancy evaluator:
        Is the retrieved context actually relevant to the query?

        from llama_index.core.evaluation import RelevancyEvaluator
        evaluator = RelevancyEvaluator(llm=llm)
        result = evaluator.evaluate_response(query="...", response=response)

    Answer Relevancy evaluator:
        Does the answer actually address the question asked?

    Correctness evaluator:
        Does the answer match the reference answer? (requires ground truth)

        from llama_index.core.evaluation import CorrectnessEvaluator
        evaluator = CorrectnessEvaluator(llm=llm)
        result = evaluator.evaluate(query="...", response="...", reference="...")

    Batch evaluation with EvaluationResult:
        from llama_index.core.evaluation import BatchEvalRunner

        runner = BatchEvalRunner(
            evaluators={"faithfulness": faithfulness_eval, "relevancy": relevancy_eval},
            workers=8,    # parallel evaluation
        )
        eval_results = await runner.aevaluate_queries(
            query_engine, queries=questions
        )

### RAGAS: The Standard RAG Evaluation Framework

    RAGAS (Retrieval Augmented Generation Assessment) is the industry-standard
    evaluation framework for RAG systems. LlamaIndex has a native integration.

    Four RAGAS metrics:
        1. Faithfulness:
            Does the answer only assert things present in the retrieved context?
            Prevents hallucination scoring.
            Score: 0.0 (complete hallucination) to 1.0 (fully grounded).

        2. Answer Relevancy:
            Does the answer actually address the question?
            High if the answer directly responds to what was asked.
            Low if the answer is relevant but doesn't answer the question.

        3. Context Precision:
            Are the retrieved chunks relevant to answering the question?
            Measures retrieval quality — how much of what was retrieved was useful.

        4. Context Recall:
            Was enough relevant context retrieved to answer the question?
            Requires a reference answer to compute.

    Usage with LlamaIndex:
        from ragas.integrations.llama_index import evaluate
        from ragas.metrics import faithfulness, answer_relevancy,
                                   context_precision, context_recall
        from datasets import Dataset

        eval_dataset = Dataset.from_dict({
            "question":  questions,
            "answer":    answers,
            "contexts":  contexts,   # list of list of retrieved chunks
            "ground_truth": ground_truths,
        })

        result = evaluate(
            query_engine = query_engine,
            metrics      = [faithfulness, answer_relevancy, context_precision],
            dataset      = eval_dataset,
        )
        print(result.to_pandas())

### Production Best Practices

    1. ALWAYS evaluate before and after any pipeline change:
        Measure faithfulness + context precision as your north star metrics.
        A single bad chunking decision can drop faithfulness by 20+ points.

    2. Use SentenceWindowNodeParser as default for text documents:
        Consistently outperforms fixed-size chunking in benchmarks.
        Set window_size=3–5 based on document density.

    3. Add a reranker after retrieval:
        Even a simple cross-encoder (BGE reranker) improves quality significantly.
        The order: retrieve top-20 → rerank → take top-5.

    4. Separate ingestion from query:
        Process and persist your index offline.
        Never rebuild an index during a query — it is slow and expensive.
        Use IngestionPipeline.persist() + VectorStoreIndex with persistent store.

    5. Metadata is your friend:
        Rich metadata (date, author, category, section) enables filtered retrieval.
        Use VectorIndexAutoRetriever for automatic metadata-aware queries.

    6. Monitor token counts:
        Track tokens_used per query for cost management.
        Set context_window limit in Settings to prevent overflow.

    7. Use async for production throughput:
        query_engine.aquery(), retriever.aretrieve(), pipeline.arun()
        Enables 10–50× higher throughput for concurrent users.

    8. Instrument with callbacks:
        from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
        debug_handler = LlamaDebugHandler(print_trace_on_end=True)
        Settings.callback_manager = CallbackManager([debug_handler])
        # Logs every step: retrieval, LLM calls, postprocessing

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Quality Improvement Levers (roughly in order of impact)              │
    ├──────────────────────────────────────────────────────────────────────┤
    │ 1. SentenceWindow or Hierarchical chunking (vs fixed-size)           │
    │ 2. Add a reranker (CohereRerank or LLMRerank)                        │
    │ 3. QueryFusionRetriever (multiple query variants + RRF fusion)       │
    │ 4. Richer metadata + AutoRetriever (structured + semantic)           │
    │ 5. Increase similarity_top_k at retrieval (20→rerank→5)              │
    │ 6. Add QuestionsAnsweredExtractor metadata during ingestion          │
    │ 7. Switch to tree_summarise response mode for multi-doc              │
    │ 8. SubQuestionQueryEngine for complex multi-part questions           │
    └──────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Documents, Nodes & Indexing — The Core Data Pipeline": {
        "description": (
            "Complete LlamaIndex ingestion and indexing pipeline. "
            "Document creation from text and with rich metadata. "
            "SimpleDirectoryReader pattern for file-based loading. "
            "SentenceSplitter and TokenTextSplitter comparison. "
            "Node structure: text, metadata, relationships, embeddings. "
            "SentenceWindowNodeParser — small nodes + large windows. "
            "HierarchicalNodeParser — multi-granularity node trees. "
            "IngestionPipeline with transformations and persistence. "
            "VectorStoreIndex creation from documents and from nodes. "
            "SummaryIndex and KeywordTableIndex patterns. "
            "Storage persistence with StorageContext. "
            "Mock LLM and embeddings for fully offline demonstration."
        ),
        "language": "python",
        "code": '''
import time
import math
import re
import json
from typing import List, Any, Optional, Dict

try:
    from llama_index.core import (
        Document, VectorStoreIndex, SummaryIndex, Settings,
        StorageContext, load_index_from_storage,
    )
    from llama_index.core.schema import (
        TextNode, NodeRelationship, RelatedNodeInfo, ObjectType,
    )
    from llama_index.core.node_parser import (
        SentenceSplitter, SentenceWindowNodeParser,
        HierarchicalNodeParser, get_leaf_nodes, get_root_nodes,
    )
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
    print("  llama-index-core: installed")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "llama-index-core", "--quiet"], check=True)
    from llama_index.core import (
        Document, VectorStoreIndex, SummaryIndex, Settings,
        StorageContext, load_index_from_storage,
    )
    from llama_index.core.schema import (
        TextNode, NodeRelationship, RelatedNodeInfo, ObjectType,
    )
    from llama_index.core.node_parser import (
        SentenceSplitter, SentenceWindowNodeParser,
        HierarchicalNodeParser, get_leaf_nodes, get_root_nodes,
    )
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr

print("=" * 65)
print("  DOCUMENTS, NODES & INDEXING — THE CORE DATA PIPELINE")
print("=" * 65)
print()

# ── Mock LLM and Embeddings for offline demonstration ─────────────────
import hashlib

class MockLLM(CustomLLM):
    context_window: int = 4096
    num_output: int     = 256
    model_name: str     = "mock-llm"
    dummy: str          = ""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        p = prompt.lower()
        if "summary" in p or "summarize" in p or "summarise" in p:
            text = "This document covers key concepts in transformer architecture, attention mechanisms, and modern language model pretraining strategies."
        elif "question" in p:
            text = "What are the key innovations of the transformer? How does self-attention work? What is the purpose of positional encoding?"
        elif "title" in p:
            text = "Technical Guide to Transformer Architectures"
        else:
            text = "Based on the provided context, the answer involves important technical concepts in machine learning and deep learning research."
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs):
        response = self.complete(prompt)
        for ch in response.text:
            yield CompletionResponse(text=ch, delta=ch)


class MockEmbedding(BaseEmbedding):
    _dim: int = PrivateAttr(default=64)

    def __init__(self, dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim

    def _get_text_embedding(self, text: str) -> List[float]:
        words = re.findall(r"[a-z]+", text.lower())
        vec   = [0.0] * self._dim
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for i in range(4):
                vec[(h >> (i * 8)) % self._dim] += 1.0
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

# Configure global Settings
Settings.llm         = MockLLM()
Settings.embed_model = MockEmbedding(dim=64)
Settings.chunk_size  = 256
Settings.chunk_overlap = 32

print(f"  Settings configured: MockLLM + MockEmbedding(dim=64)")
print(f"  chunk_size={Settings.chunk_size}, chunk_overlap={Settings.chunk_overlap}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Document creation with rich metadata
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Document creation: text, metadata, IDs")
print("━" * 65)
print()

RAW_CORPUS = [
    {
        "id": "doc_001",
        "title": "Transformer Architecture Overview",
        "text": """The Transformer architecture, introduced in 'Attention Is All You Need'
by Vaswani et al. in 2017, revolutionised natural language processing. The core
innovation was replacing recurrent layers with self-attention mechanisms.
Self-attention allows every position in a sequence to directly attend to every
other position, enabling parallelisation during training. The model consists of
an encoder stack and a decoder stack, each composed of multiple identical layers.
Each layer contains two sub-layers: a multi-head self-attention mechanism and a
position-wise fully connected feed-forward network. Residual connections and layer
normalisation are applied around each sub-layer. The original model was designed
for machine translation but the architecture has since been applied to virtually
every domain in machine learning.""",
        "source": "transformers_overview.pdf",
        "category": "architecture", "year": 2017,
    },
    {
        "id": "doc_002",
        "title": "BERT: Bidirectional Encoder Representations",
        "text": """BERT (Bidirectional Encoder Representations from Transformers) was
developed by Google AI Language and released in 2018. It uses only the encoder
portion of the Transformer architecture. The key innovation of BERT is bidirectional
pretraining using the Masked Language Model (MLM) objective: 15% of input tokens
are randomly masked, and the model is trained to predict the masked tokens using
both left and right context. This bidirectionality allows BERT to develop deep
contextual representations. BERT also uses Next Sentence Prediction (NSP) as a
second pretraining objective. BERT-base has 12 transformer layers, 768 hidden
units, and 12 attention heads (110M parameters). BERT-large has 24 layers,
1024 hidden units, and 16 attention heads (340M parameters).""",
        "source": "bert_paper_summary.pdf",
        "category": "pretraining", "year": 2018,
    },
    {
        "id": "doc_003",
        "title": "Attention Mechanism: Mathematical Foundation",
        "text": """The attention mechanism computes a weighted sum of value vectors,
where weights are determined by the compatibility between query and key vectors.
Scaled dot-product attention is defined as: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V.
The scaling factor 1/sqrt(d_k) prevents dot products from growing too large in
magnitude as the dimension increases, which would push the softmax into regions
with very small gradients. Multi-head attention applies multiple attention functions
in parallel, each with different learned projection matrices. The outputs are
concatenated and linearly projected: MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W_O.
Each head specialises in different types of relationships: syntactic dependencies,
coreference, positional patterns. The number of heads is typically 8, 12, or 16.""",
        "source": "attention_math.pdf",
        "category": "mathematics", "year": 2017,
    },
    {
        "id": "doc_004",
        "title": "LlamaIndex: Data Framework for LLMs",
        "text": """LlamaIndex is a data framework for building LLM-powered applications
over private and external data. It was created by Jerry Liu and open-sourced in
November 2022. The framework provides data connectors to ingest any data source,
data indexes structured for LLM consumption, a query interface to combine LLM with
the data, and application integrations. LlamaIndex's key differentiator is its
focus on retrieval quality over general-purpose application building. The ingestion
pipeline handles document loading, node parsing with various chunking strategies,
metadata extraction, and embedding computation. The query layer includes retrievers,
node postprocessors including rerankers, and response synthesisers with multiple
modes. Advanced features include sub-question decomposition, knowledge graphs,
multi-modal indexing, and evaluation with RAGAS metrics.""",
        "source": "llamaindex_docs.md",
        "category": "framework", "year": 2022,
    },
    {
        "id": "doc_005",
        "title": "Positional Encoding in Transformers",
        "text": """Since the Transformer has no recurrence or convolution, it has no
inherent notion of token order. Positional encoding adds position information
to the token embeddings. The original Transformer used sinusoidal positional
encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) =
cos(pos / 10000^(2i/d_model)). Modern LLMs use Rotary Position Embedding (RoPE),
which encodes position by rotating query and key vectors in 2D subspaces. RoPE
has become dominant because it naturally encodes relative positions, works with
the dot product formulation of attention, and enables length extrapolation beyond
the training context window through techniques like YaRN. ALiBi adds a linear
bias to attention scores proportional to token distance, requiring no learned
parameters and supporting inference at longer contexts than seen during training.""",
        "source": "positional_encoding.pdf",
        "category": "architecture", "year": 2021,
    },
]

# Create Document objects
documents = [
    Document(
        text     = d["text"],
        doc_id   = d["id"],
        metadata = {
            "title":    d["title"],
            "source":   d["source"],
            "category": d["category"],
            "year":     d["year"],
        },
        excluded_llm_metadata_keys   = ["source"],   # don't send source to LLM
        excluded_embed_metadata_keys = ["year"],      # don't embed year
    )
    for d in RAW_CORPUS
]

print(f"  Created {len(documents)} Documents:")
print(f"  {'ID':<10} {'Title':<42} {'Cat':<15} {'Year'}")
print(f"  {'─'*75}")
for doc in documents:
    m = doc.metadata
    print(f"  {doc.doc_id:<10} {m['title'][:40]:<42} {m['category']:<15} {m['year']}")
print()

# Show Document structure
sample_doc = documents[0]
print(f"  Document structure (doc_001):")
print(f"    doc_id:    {sample_doc.doc_id}")
print(f"    text len:  {len(sample_doc.text)} chars")
print(f"    metadata:  {sample_doc.metadata}")
print(f"    excluded from LLM:   {sample_doc.excluded_llm_metadata_keys}")
print(f"    excluded from embed: {sample_doc.excluded_embed_metadata_keys}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Text splitting — SentenceSplitter and TokenTextSplitter
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Text splitting: SentenceSplitter comparison")
print("━" * 65)
print()

splitter_small = SentenceSplitter(chunk_size=128, chunk_overlap=20)
splitter_med   = SentenceSplitter(chunk_size=256, chunk_overlap=40)
splitter_large = SentenceSplitter(chunk_size=512, chunk_overlap=64)

for label, splitter in [
    ("chunk_size=128", splitter_small),
    ("chunk_size=256", splitter_med),
    ("chunk_size=512", splitter_large),
]:
    all_nodes = []
    for doc in documents:
        all_nodes.extend(splitter.get_nodes_from_documents([doc]))
    avg_len = sum(len(n.text.split()) for n in all_nodes) / len(all_nodes)
    print(f"  {label:<18}: {len(all_nodes):>3} nodes, avg {avg_len:>5.0f} words/node")

print()
# Show a node in detail
default_nodes = splitter_med.get_nodes_from_documents(documents)
sample_node   = default_nodes[0]
print(f"  Sample TextNode (splitter_med, node 0):")
print(f"    node_id:   {sample_node.node_id[:16]}...")
print(f"    text:      '{sample_node.text[:80].strip()}...'")
print(f"    metadata:  {sample_node.metadata}")
print(f"    start_idx: {sample_node.start_char_idx}")
print(f"    end_idx:   {sample_node.end_char_idx}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: SentenceWindowNodeParser
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — SentenceWindowNodeParser: small + window pattern")
print("━" * 65)
print()

window_parser = SentenceWindowNodeParser.from_defaults(
    window_size             = 2,
    window_metadata_key     = "window",
    original_text_metadata_key = "original_sentence",
)

window_nodes = window_parser.get_nodes_from_documents([documents[0]])

print(f"  SentenceWindowNodeParser (window_size=2):")
print(f"  Applied to: '{documents[0].metadata['title']}'")
print(f"  Produced {len(window_nodes)} sentence nodes")
print()
if window_nodes:
    n = window_nodes[0]
    print(f"  Node 0 (what gets EMBEDDED):")
    print(f"    original_sentence: '{n.metadata.get('original_sentence','')[:80]}...'")
    print()
    print(f"  Node 0 window (what the LLM SEES after MetadataReplacementPostProcessor):")
    window = n.metadata.get('window', '')
    print(f"    window: '{window[:150]}...'")
    print()
    print(f"  Key insight:")
    print(f"    Embedding = precise single sentence  → high retrieval precision")
    print(f"    LLM context = ±2 sentences window    → rich answer generation context")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: HierarchicalNodeParser
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — HierarchicalNodeParser: multi-granularity nodes")
print("━" * 65)
print()

hier_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 256, 128]
)
hier_nodes = hier_parser.get_nodes_from_documents(documents)
leaf_nodes  = get_leaf_nodes(hier_nodes)
root_nodes  = get_root_nodes(hier_nodes)

# Count nodes at each level
level_counts: Dict[int, int] = {}
for node in hier_nodes:
    n_children = len(node.relationships.get(NodeRelationship.CHILD, []))
    has_parent = NodeRelationship.PARENT in node.relationships
    if not has_parent:
        level_counts[0] = level_counts.get(0, 0) + 1
    elif n_children == 0:
        level_counts[2] = level_counts.get(2, 0) + 1
    else:
        level_counts[1] = level_counts.get(1, 0) + 1

print(f"  HierarchicalNodeParser (chunk_sizes=[512, 256, 128]):")
print(f"  Applied to {len(documents)} documents")
print(f"  Total nodes: {len(hier_nodes)}")
print()
print(f"  {'Level':<8} {'Count':>8} {'Role':<30} {'Chunk size'}")
print(f"  {'─'*55}")
level_info = [
    (0, level_counts.get(0,0), "Root (largest context)",    "~512 tokens"),
    (1, level_counts.get(1,0), "Middle (medium context)",   "~256 tokens"),
    (2, level_counts.get(2,0), "Leaf (fine retrieval)",     "~128 tokens"),
]
for lvl, cnt, role, sz in level_info:
    print(f"  {lvl:<8} {cnt:>8} {role:<30} {sz}")
print()
print(f"  Leaf nodes (indexed for retrieval): {len(leaf_nodes)}")
print(f"  AutoMergingRetriever promotes to parent if ≥50% of siblings retrieved")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: IngestionPipeline with transformations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — IngestionPipeline with transformations")
print("━" * 65)
print()

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=256, chunk_overlap=40),
        Settings.embed_model,   # embed each node
    ]
)

t0             = time.perf_counter()
pipeline_nodes = pipeline.run(documents=documents)
t_pipeline     = (time.perf_counter() - t0) * 1000

print(f"  IngestionPipeline: SentenceSplitter → EmbeddingModel")
print(f"  Processed {len(documents)} documents → {len(pipeline_nodes)} nodes in {t_pipeline:.0f}ms")
print()
# Verify embeddings were computed
has_embed = sum(1 for n in pipeline_nodes if n.embedding is not None)
print(f"  Nodes with embeddings: {has_embed}/{len(pipeline_nodes)}")
if pipeline_nodes[0].embedding:
    emb = pipeline_nodes[0].embedding
    print(f"  Embedding dim: {len(emb)}, norm: {math.sqrt(sum(x*x for x in emb)):.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: VectorStoreIndex construction and storage
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — VectorStoreIndex: build, query, and persist")
print("━" * 65)
print()

import tempfile, os

# Build from documents (pipeline handled internally)
t0    = time.perf_counter()
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=40)],
    show_progress=False,
)
t_build = (time.perf_counter() - t0) * 1000

docstore_nodes = list(index.docstore.docs.values())
print(f"  VectorStoreIndex built in {t_build:.0f}ms")
print(f"  Nodes indexed: {len(docstore_nodes)}")
print()

# Persist and reload
with tempfile.TemporaryDirectory() as tmp:
    persist_dir = os.path.join(tmp, "index_storage")
    index.storage_context.persist(persist_dir=persist_dir)

    # List persisted files
    all_files = []
    for root, dirs, files in os.walk(persist_dir):
        for f in files:
            full = os.path.join(root, f)
            all_files.append((f, os.path.getsize(full) / 1024))

    print(f"  Persisted index files:")
    for fname, kb in all_files:
        print(f"    {fname:<30} {kb:.1f} KB")
    print()

    # Reload from disk
    t0       = time.perf_counter()
    storage  = StorageContext.from_defaults(persist_dir=persist_dir)
    reloaded = load_index_from_storage(storage)
    t_reload = (time.perf_counter() - t0) * 1000
    print(f"  Index reloaded from disk in {t_reload:.0f}ms  ✅")
    print()

# Query the index directly
query_engine = index.as_query_engine(similarity_top_k=3)

queries = [
    "What is the Transformer architecture and who invented it?",
    "How does BERT differ from GPT in terms of architecture?",
    "What is the mathematical formula for attention?",
]

print(f"  Index query results:")
for q in queries:
    t0  = time.perf_counter()
    resp = query_engine.query(q)
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"  Q: {q}")
    print(f"  A: {resp.response[:110].strip()}...")
    sources = [n.node.metadata.get("title","?")[:30] for n in resp.source_nodes]
    print(f"  Sources ({len(resp.source_nodes)}): {sources}")
    print(f"  Latency: {t_ms:.0f}ms")
    print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Retrievers & Response Synthesis — Quality-Focused Retrieval": {
        "description": (
            "Deep dive into LlamaIndex retrieval quality patterns. "
            "VectorIndexRetriever: similarity_top_k tuning. "
            "BM25Retriever: keyword-based sparse retrieval. "
            "QueryFusionRetriever: multi-query + Reciprocal Rank Fusion. "
            "SimilarityPostprocessor: score-based filtering. "
            "MetadataReplacementPostProcessor: sentence window swap. "
            "LLMRerank and simulated cross-encoder reranking. "
            "Five response synthesis modes benchmark: compact, refine, "
            "tree_summarize, accumulate, simple_summarize. "
            "RetrieverQueryEngine with custom postprocessors. "
            "Source attribution: source_nodes and metadata inspection. "
            "Retrieval quality metrics: precision, recall, MRR."
        ),
        "language": "python",
        "code": '''
import time
import math
import re
import hashlib
from typing import List, Any, Optional, Dict

try:
    from llama_index.core import (
        Document, VectorStoreIndex, SummaryIndex, Settings,
    )
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.node_parser import (
        SentenceSplitter, SentenceWindowNodeParser,
    )
    from llama_index.core.postprocessor import (
        SimilarityPostprocessor, MetadataReplacementPostProcessor,
    )
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import (
        get_response_synthesizer, ResponseMode,
    )
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
    print("  llama-index-core: installed")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "llama-index-core", "--quiet"], check=True)
    from llama_index.core import (
        Document, VectorStoreIndex, SummaryIndex, Settings,
    )
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.node_parser import (
        SentenceSplitter, SentenceWindowNodeParser,
    )
    from llama_index.core.postprocessor import (
        SimilarityPostprocessor, MetadataReplacementPostProcessor,
    )
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import (
        get_response_synthesizer, ResponseMode,
    )
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr

print("=" * 65)
print("  RETRIEVERS & RESPONSE SYNTHESIS — QUALITY PATTERNS")
print("=" * 65)
print()

# ── Mock classes (same as Operation 1) ───────────────────────────────
class MockLLM(CustomLLM):
    context_window: int = 4096
    num_output: int = 256
    model_name: str = "mock-llm"
    dummy: str = ""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=self.context_window,
                           num_output=self.num_output, model_name=self.model_name)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        p = prompt.lower()
        if "attention" in p and "formula" in p:
            return CompletionResponse(text="The attention formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. The scaling by 1/sqrt(d_k) prevents gradient vanishing in the softmax.")
        if "bert" in p and "gpt" in p:
            return CompletionResponse(text="BERT uses a bidirectional encoder with masked language modelling pretraining. GPT uses a unidirectional (causal) decoder for autoregressive language modelling. BERT is better for understanding tasks; GPT for generation.")
        if "transformer" in p:
            return CompletionResponse(text="The Transformer architecture was introduced by Vaswani et al. in 2017 in 'Attention Is All You Need'. It uses self-attention to replace recurrence with parallel processing.")
        if "refine" in p:
            return CompletionResponse(text="Refined answer incorporating all retrieved context: The key concepts span attention mechanisms, positional encoding, and the encoder-decoder structure.")
        if "summarize" in p or "summary" in p:
            return CompletionResponse(text="Summary: The documents cover Transformer architecture, BERT pretraining, attention mathematics, LlamaIndex data framework, and positional encoding methods.")
        return CompletionResponse(text="Based on the retrieved context, the answer involves key concepts related to transformer architectures and modern language model design.")

    def stream_complete(self, prompt, **kwargs):
        r = self.complete(prompt)
        for ch in r.text:
            yield CompletionResponse(text=ch, delta=ch)


class MockEmbedding(BaseEmbedding):
    _dim: int = PrivateAttr(default=64)

    def __init__(self, dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim

    def _get_text_embedding(self, text: str) -> List[float]:
        words = re.findall(r"[a-z]+", text.lower())
        vec   = [0.0] * self._dim
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for i in range(4):
                vec[(h >> (i*8)) % self._dim] += 1.0
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _get_query_embedding(self, q: str) -> List[float]:
        return self._get_text_embedding(q)

    async def _aget_query_embedding(self, q: str) -> List[float]:
        return self._get_query_embedding(q)

    async def _aget_text_embedding(self, t: str) -> List[float]:
        return self._get_text_embedding(t)


Settings.llm         = MockLLM()
Settings.embed_model = MockEmbedding(dim=64)
Settings.chunk_size  = 256
Settings.chunk_overlap = 32

# ── Build shared document corpus and index ────────────────────────────
TEXTS = {
    "transformer":   ("Transformer Architecture Overview",
        "The Transformer architecture was introduced in 'Attention Is All You Need' "
        "by Vaswani et al. in 2017. It replaces recurrence with self-attention, enabling "
        "full parallelisation. The architecture has an encoder and decoder, each with "
        "multi-head attention and feed-forward sub-layers plus residual connections. "
        "The scaling factor 1/sqrt(d_k) prevents gradient saturation in softmax. "
        "Positional encoding adds order information since attention is permutation-invariant."),
    "bert":          ("BERT: Bidirectional Encoder",
        "BERT uses only the encoder portion of the Transformer. It is pretrained with "
        "Masked Language Modelling (MLM): 15% of tokens are masked and predicted from "
        "bidirectional context. This creates deep bidirectional representations unlike GPT. "
        "BERT-base has 12 layers, 768 hidden units, 12 attention heads, 110M params. "
        "Fine-tuning adds a task head on top of the CLS token representation. "
        "BERT set records on GLUE, SQuAD and many other NLP benchmarks upon release."),
    "attention":     ("Attention Mechanism Mathematics",
        "Scaled dot-product attention formula: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) * V. "
        "Q, K, V are obtained by projecting input with learned weight matrices. "
        "Multi-head attention runs h attention functions in parallel with different projections. "
        "The outputs are concatenated: MultiHead = Concat(head_1,...,head_h) * W_O. "
        "Each head can specialise: some track syntax, others coreference or position. "
        "Causal attention adds a mask preventing positions from attending to future tokens."),
    "llamaindex":    ("LlamaIndex Framework",
        "LlamaIndex is a data framework for LLM-powered applications over private data. "
        "Created by Jerry Liu in November 2022, it focuses on retrieval quality. "
        "Core components: data connectors (150+ readers), indexes (vector, summary, tree, "
        "knowledge graph), query engines with multiple retrieval and synthesis strategies. "
        "Advanced features: SentenceWindowNodeParser, HierarchicalNodeParser, "
        "QueryFusionRetriever with multi-query and RRF fusion, SubQuestionQueryEngine."),
    "positional":    ("Positional Encoding Methods",
        "Sinusoidal positional encoding uses sin/cos at different frequencies to inject "
        "position. Rotary Position Embedding (RoPE) encodes position by rotating Q and K "
        "in 2D subspaces. The rotation angle depends on position, so relative positions "
        "emerge naturally from the dot product. RoPE is used in LLaMA, Mistral, and most "
        "modern LLMs. ALiBi adds linear biases to attention scores: closer tokens get "
        "smaller penalties. ALiBi enables zero-shot extrapolation to longer sequences."),
}

documents = [
    Document(
        text=t, doc_id=k,
        metadata={"title": title, "category": k, "word_count": len(t.split())}
    )
    for k, (title, t) in TEXTS.items()
]

splitter = SentenceSplitter(chunk_size=150, chunk_overlap=30)
index    = VectorStoreIndex.from_documents(
    documents,
    transformations=[splitter],
    show_progress=False,
)
all_nodes = list(index.docstore.docs.values())
print(f"  Index built: {len(all_nodes)} nodes from {len(documents)} documents")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: VectorIndexRetriever with similarity_top_k
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — VectorIndexRetriever: top-k comparison")
print("━" * 65)
print()

query = "How does attention mechanism work mathematically?"
print(f"  Query: '{query}'")
print()

for k in [1, 3, 5]:
    retriever = index.as_retriever(similarity_top_k=k)
    t0        = time.perf_counter()
    nodes     = retriever.retrieve(query)
    t_ms      = (time.perf_counter() - t0) * 1000

    titles = [n.node.metadata.get("title","?")[:25] for n in nodes]
    scores = [f"{n.score:.4f}" if n.score else "N/A" for n in nodes]
    print(f"  top_k={k} ({t_ms:.0f}ms): {list(zip(titles, scores))}")

print()
print(f"  Source node detail (top_k=3, node[0]):")
retriever = index.as_retriever(similarity_top_k=3)
retrieved = retriever.retrieve(query)
if retrieved:
    n0 = retrieved[0]
    print(f"    score:    {n0.score:.6f}")
    print(f"    node_id:  {n0.node.node_id[:20]}...")
    print(f"    title:    {n0.node.metadata.get('title','?')}")
    print(f"    text:     '{n0.node.text[:80].strip()}...'")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: BM25Retriever (keyword/sparse retrieval)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — BM25Retriever: keyword-based sparse retrieval")
print("━" * 65)
print()

# Implement a lightweight BM25 retriever for demonstration
class SimpleBM25Retriever:
    def __init__(self, nodes: List[TextNode], top_k: int = 3):
        self.nodes = nodes
        self.top_k = top_k
        # Build inverted index
        self.df: Dict[str, int] = {}
        self.tf_idf: List[Dict[str, float]] = []
        N = len(nodes)
        for node in nodes:
            terms = set(re.findall(r"[a-z]+", node.text.lower()))
            for t in terms:
                self.df[t] = self.df.get(t, 0) + 1
        for node in nodes:
            words = re.findall(r"[a-z]+", node.text.lower())
            tf: Dict[str, float] = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1
            total = len(words) or 1
            tfidf = {w: (c/total) * math.log(N / (self.df.get(w,1)))
                     for w, c in tf.items()}
            self.tf_idf.append(tfidf)

    def retrieve(self, query: str) -> List[NodeWithScore]:
        query_terms = re.findall(r"[a-z]+", query.lower())
        scores = []
        for i, tfidf in enumerate(self.tf_idf):
            score = sum(tfidf.get(t, 0) for t in query_terms)
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        results = []
        for idx, score in scores[:self.top_k]:
            if score > 0:
                results.append(NodeWithScore(node=self.nodes[idx], score=score))
        return results

# Get flat node list
flat_nodes = [n for n in all_nodes if isinstance(n, TextNode)]
bm25 = SimpleBM25Retriever(flat_nodes, top_k=3)

keyword_queries = [
    "attention formula softmax sqrt",
    "BERT masked language model pretraining",
    "sinusoidal positional encoding",
]

print(f"  BM25 (TF-IDF) keyword retrieval:")
for kq in keyword_queries:
    bm25_results = bm25.retrieve(kq)
    titles = [n.node.metadata.get("category","?") for n in bm25_results]
    print(f"  Query: '{kq[:45]}'")
    print(f"  Retrieved categories: {titles}")
print()

# Compare vector vs BM25 on different query types
print(f"  Vector vs BM25 comparison:")
print(f"  {'Query type':<30} {'Vector top-1':<25} {'BM25 top-1'}")
print(f"  {'─'*75}")
test_pairs = [
    ("semantic: context understanding", "what does attention allow the model to do"),
    ("keyword: exact technical term",   "attention formula softmax QKV"),
    ("semantic: concept",               "bidirectional language understanding"),
    ("keyword: exact names",            "BERT masked language model"),
]
vec_retriever = index.as_retriever(similarity_top_k=1)
for label, qtext in test_pairs:
    vec_top = vec_retriever.retrieve(qtext)
    bm25_top = bm25.retrieve(qtext)
    v_cat = vec_top[0].node.metadata.get("category","?") if vec_top else "none"
    b_cat = bm25_top[0].node.metadata.get("category","?") if bm25_top else "none"
    print(f"  {label:<30} {v_cat:<25} {b_cat}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: SimilarityPostprocessor — score-based filtering
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Node postprocessors: filtering and reranking")
print("━" * 65)
print()

retriever      = index.as_retriever(similarity_top_k=5)
raw_nodes      = retriever.retrieve(query)
filtered_nodes = SimilarityPostprocessor(similarity_cutoff=0.5).postprocess_nodes(
    raw_nodes, query_str=query
)

print(f"  SimilarityPostprocessor(cutoff=0.5):")
print(f"  {'─'*50}")
print(f"  {'Node':<5} {'Score':>8} {'Category':<20} {'Kept'}")
print(f"  {'─'*50}")
for i, node in enumerate(raw_nodes):
    kept = "✓" if node.score and node.score >= 0.5 else "✗"
    cat  = node.node.metadata.get("category","?")
    score_str = f"{node.score:.4f}" if node.score else "  N/A"
    print(f"  [{i}]   {score_str:>8} {cat:<20} {kept}")

print(f"\n  Before filter: {len(raw_nodes)} nodes")
print(f"  After  filter: {len(filtered_nodes)} nodes (score >= 0.5)")
print()

# Simulated re-ranking (cross-encoder concept)
class MockReranker:
    def __init__(self, top_n: int = 3):
        self.top_n = top_n

    def postprocess_nodes(self, nodes, query_str=""):
        # Simulated cross-encoder scoring: slightly different than bi-encoder
        query_words = set(re.findall(r"[a-z]+", query_str.lower()))
        rescored = []
        for node in nodes:
            node_words = set(re.findall(r"[a-z]+", node.node.text.lower()))
            overlap    = len(query_words & node_words)
            new_score  = overlap / (len(query_words) + 1e-8)
            rescored.append(NodeWithScore(node=node.node, score=new_score))
        rescored.sort(key=lambda x: -(x.score or 0))
        return rescored[:self.top_n]

reranker = MockReranker(top_n=3)
reranked = reranker.postprocess_nodes(raw_nodes, query_str=query)

print(f"  Cross-encoder reranking (top_n=3):")
print(f"  {'Rank':<6} {'Score':>8} {'Category'}")
print(f"  {'─'*30}")
for i, n in enumerate(reranked):
    print(f"  [{i+1}]   {n.score:>8.4f} {n.node.metadata.get('category','?')}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Response synthesis modes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Response synthesis modes benchmark")
print("━" * 65)
print()

query_engine_query = "What are the key innovations in the Transformer and BERT architectures?"
retriever_5        = index.as_retriever(similarity_top_k=4)
retrieved_nodes    = retriever_5.retrieve(query_engine_query)

modes_to_test = [
    (ResponseMode.COMPACT,        "compact (default)"),
    (ResponseMode.REFINE,         "refine (sequential)"),
    (ResponseMode.TREE_SUMMARIZE, "tree_summarize (hierarchical)"),
    (ResponseMode.ACCUMULATE,     "accumulate (per-node)"),
    (ResponseMode.SIMPLE_SUMMARIZE,"simple_summarize (truncate)"),
]

print(f"  Query: '{query_engine_query[:55]}...'")
print(f"  Retrieved nodes: {len(retrieved_nodes)}")
print()
print(f"  {'Mode':<25} {'LLM calls':>10} {'Latency ms':>12} {'Response preview'}")
print(f"  {'─'*80}")

for mode, label in modes_to_test:
    synthesiser = get_response_synthesizer(
        response_mode=mode, llm=Settings.llm
    )
    t0   = time.perf_counter()
    resp = synthesiser.synthesize(query_engine_query, nodes=retrieved_nodes)
    t_ms = (time.perf_counter() - t0) * 1000
    preview = resp.response[:45].strip().replace("\n"," ") if resp.response else "N/A"
    # Estimate LLM calls by mode
    llm_calls = {
        ResponseMode.COMPACT: 1, ResponseMode.REFINE: len(retrieved_nodes),
        ResponseMode.TREE_SUMMARIZE: max(1, len(retrieved_nodes)//2),
        ResponseMode.ACCUMULATE: len(retrieved_nodes),
        ResponseMode.SIMPLE_SUMMARIZE: 1,
    }.get(mode, 1)
    print(f"  {label:<25} {llm_calls:>10} {t_ms:>12.0f} '{preview}...'")

print()
print(f"  Recommendation:")
print(f"    compact:        Best default — efficient, single LLM call")
print(f"    tree_summarize: Long contexts, multiple documents")
print(f"    refine:         Highest quality, multiple sources needed")
print(f"    accumulate:     'List all X from document' queries")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Full RetrieverQueryEngine with postprocessors
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 5 — RetrieverQueryEngine with custom postprocessors")
print("━" * 65)
print()

synthesiser = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
query_engine = RetrieverQueryEngine(
    retriever            = index.as_retriever(similarity_top_k=5),
    response_synthesizer = synthesiser,
    node_postprocessors  = [
        SimilarityPostprocessor(similarity_cutoff=0.3),
        MockReranker(top_n=3),
    ],
)

test_queries = [
    "How does the attention formula scale dot products?",
    "What pretraining objective does BERT use?",
]

print(f"  RetrieverQueryEngine: retrieve(5) → filter(≥0.3) → rerank(top-3) → compact")
print()
for q in test_queries:
    t0   = time.perf_counter()
    resp = query_engine.query(q)
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"  Q: {q}")
    print(f"  A: {resp.response[:100].strip()}...")
    print(f"  Sources: {[n.node.metadata.get('category','?') for n in resp.source_nodes]}")
    print(f"  Latency: {t_ms:.0f}ms")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Retrieval quality metrics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Retrieval quality metrics: precision, recall, MRR")
print("━" * 65)
print()

# Ground truth: query → expected source categories
ground_truth = {
    "attention formula softmax QKV":         ["attention"],
    "BERT masked language modelling":         ["bert"],
    "positional encoding sinusoidal RoPE":    ["positional"],
    "LlamaIndex framework features":          ["llamaindex"],
    "transformer encoder decoder structure":  ["transformer"],
}

def precision_at_k(retrieved_cats, relevant_cats, k):
    top_k = retrieved_cats[:k]
    return sum(1 for c in top_k if c in relevant_cats) / k

def recall_at_k(retrieved_cats, relevant_cats, k):
    top_k = retrieved_cats[:k]
    return sum(1 for c in top_k if c in relevant_cats) / max(len(relevant_cats), 1)

def mrr(retrieved_cats, relevant_cats):
    for i, c in enumerate(retrieved_cats):
        if c in relevant_cats:
            return 1.0 / (i + 1)
    return 0.0

K = 3
retriever_eval = index.as_retriever(similarity_top_k=K)

p_vals, r_vals, mrr_vals = [], [], []
print(f"  Evaluation (k={K}):")
print(f"  {'Query':<42} {'P@3':>6} {'R@3':>6} {'MRR':>6}")
print(f"  {'─'*62}")

for q, expected in ground_truth.items():
    nodes = retriever_eval.retrieve(q)
    cats  = [n.node.metadata.get("category","?") for n in nodes]
    p = precision_at_k(cats, expected, K)
    r = recall_at_k(cats, expected, K)
    m = mrr(cats, expected)
    p_vals.append(p); r_vals.append(r); mrr_vals.append(m)
    print(f"  {q[:40]:<42} {p:>6.3f} {r:>6.3f} {m:>6.3f}")

print(f"  {'─'*62}")
print(f"  {'Mean':<42} {sum(p_vals)/len(p_vals):>6.3f} "
      f"{sum(r_vals)/len(r_vals):>6.3f} {sum(mrr_vals)/len(mrr_vals):>6.3f}")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Advanced Query Patterns — Sub-Questions, Router & Chat Engine": {
        "description": (
            "Advanced LlamaIndex query patterns for complex information needs. "
            "SubQuestionQueryEngine: decompose complex queries into sub-questions. "
            "RouterQueryEngine: route to specialist index by query type. "
            "SummaryIndex for full-document summarisation tasks. "
            "KeywordTableIndex for exact-match retrieval patterns. "
            "QueryPipeline: composable multi-step query processing DAG. "
            "Chat engine with conversation history and context condensing. "
            "Metadata filtering with structured search predicates. "
            "QueryFusionRetriever with multi-query generation and RRF. "
            "Custom response synthesis with streaming. "
            "Complete source attribution and citation patterns."
        ),
        "language": "python",
        "code": '''
import time
import math
import re
import hashlib
from typing import List, Optional, Any, Dict

try:
    from llama_index.core import (
        Document, VectorStoreIndex, SummaryIndex,
        KeywordTableIndex, Settings,
    )
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.query_engine import (
        SubQuestionQueryEngine, RouterQueryEngine, RetrieverQueryEngine,
    )
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.response_synthesizers import (
        get_response_synthesizer, ResponseMode,
    )
    from llama_index.core.chat_engine import CondensePlusContextChatEngine
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.llms import (
        CustomLLM, CompletionResponse, LLMMetadata, ChatMessage,
    )
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.vector_stores import (
        MetadataFilter, MetadataFilters, FilterOperator, FilterCondition,
    )
    print("  llama-index-core: installed")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "llama-index-core", "--quiet"], check=True)
    from llama_index.core import (
        Document, VectorStoreIndex, SummaryIndex,
        KeywordTableIndex, Settings,
    )
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.query_engine import (
        SubQuestionQueryEngine, RouterQueryEngine, RetrieverQueryEngine,
    )
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.response_synthesizers import (
        get_response_synthesizer, ResponseMode,
    )
    from llama_index.core.chat_engine import CondensePlusContextChatEngine
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.llms import (
        CustomLLM, CompletionResponse, LLMMetadata, ChatMessage,
    )
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr

print("=" * 65)
print("  ADVANCED QUERY PATTERNS — SUB-QUESTIONS, ROUTER & CHAT")
print("=" * 65)
print()

# ── Mock classes ──────────────────────────────────────────────────────
class MockLLM(CustomLLM):
    context_window: int = 4096
    num_output: int = 512
    model_name: str = "mock-llm"
    dummy: str = ""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=self.context_window,
                           num_output=self.num_output, model_name=self.model_name)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        p = prompt.lower()
        # Sub-question detection
        if "sub_question" in p or "generate" in p and "question" in p:
            return CompletionResponse(text=json_sub_questions(p))
        # Selector detection
        if "choose" in p or "select" in p or "which" in p:
            if "summary" in p or "entire" in p or "whole" in p:
                return CompletionResponse(text="2")   # summary engine
            return CompletionResponse(text="1")       # vector engine
        # Condense question
        if "condense" in p or "standalone" in p or "rephrase" in p:
            return CompletionResponse(text="What are the key architectural components of BERT?")
        # BERT-specific
        if "bert" in p:
            return CompletionResponse(text="BERT uses bidirectional encoder-only architecture with MLM pretraining. It has 12 layers, 768 hidden dimensions, and 12 attention heads for the base version (110M params). Fine-tuning adds a classification head on the CLS token.")
        # GPT-specific
        if "gpt" in p:
            return CompletionResponse(text="GPT uses a decoder-only architecture with causal (autoregressive) language modelling. Each token only attends to previous tokens. GPT-2 has 12 layers and 117M-1.5B parameters depending on variant. It excels at text generation tasks.")
        # Attention math
        if "attention" in p and ("formula" in p or "math" in p or "equation" in p):
            return CompletionResponse(text="Attention formula: softmax(QK^T/sqrt(d_k)) * V. The scaling prevents gradient vanishing. Multi-head attention runs h parallel attention functions and concatenates outputs.")
        # Summary mode
        if "all" in p and ("document" in p or "text" in p):
            return CompletionResponse(text="The documents collectively cover: (1) Transformer architecture and self-attention, (2) BERT bidirectional pretraining, (3) GPT autoregressive generation, (4) Mathematical foundations of attention, (5) Positional encoding methods including RoPE and ALiBi.")
        return CompletionResponse(text="Based on the available context, the key insight is that modern transformer-based models use self-attention for parallel sequence processing, with different variants (BERT, GPT, T5) optimised for understanding vs generation tasks.")

    def stream_complete(self, prompt, **kwargs):
        r = self.complete(prompt)
        for ch in r.text:
            yield CompletionResponse(text=ch, delta=ch)

    def chat(self, messages, **kwargs):
        from llama_index.core.llms import ChatResponse
        last_user = next((m.content for m in reversed(messages)
                          if m.role == "user"), "")
        response = self.complete(last_user)
        return ChatResponse(message=ChatMessage(role="assistant",
                                                content=response.text))


def json_sub_questions(prompt):
    import json
    if "compare" in prompt or "bert" in prompt and "gpt" in prompt:
        return json.dumps([
            {"sub_question": "What is the architecture of BERT?",
             "tool_name": "bert_engine"},
            {"sub_question": "What pretraining objective does BERT use?",
             "tool_name": "bert_engine"},
            {"sub_question": "What is the architecture of GPT?",
             "tool_name": "gpt_engine"},
            {"sub_question": "What pretraining objective does GPT use?",
             "tool_name": "gpt_engine"},
        ])
    return json.dumps([{"sub_question": prompt, "tool_name": "main_engine"}])


class MockEmbedding(BaseEmbedding):
    _dim: int = PrivateAttr(default=64)
    def __init__(self, dim=64, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim
    def _get_text_embedding(self, text):
        words = re.findall(r"[a-z]+", text.lower())
        vec   = [0.0] * self._dim
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for i in range(4):
                vec[(h >> (i*8)) % self._dim] += 1.0
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x/norm for x in vec]
    def _get_query_embedding(self, q): return self._get_text_embedding(q)
    async def _aget_query_embedding(self, q): return self._get_query_embedding(q)
    async def _aget_text_embedding(self, t): return self._get_text_embedding(t)


import json
Settings.llm         = MockLLM()
Settings.embed_model = MockEmbedding(dim=64)
Settings.chunk_size  = 200

# ── Build two specialist indexes ──────────────────────────────────────
BERT_DOCS = [Document(text="""BERT (Bidirectional Encoder Representations from Transformers)
uses only the encoder portion of the Transformer. Pretraining with Masked Language
Modelling (MLM): 15% of tokens randomly masked, model predicts them using bidirectional
context. Next Sentence Prediction (NSP): predict if sentence B follows sentence A.
BERT-base: 12 layers, 768 hidden, 12 heads, 110M parameters. BERT-large: 24 layers,
1024 hidden, 16 heads, 340M parameters. Fine-tuning adds a task head; CLS token
used for classification. Set records on GLUE and SQuAD benchmarks.""",
    metadata={"model": "BERT", "type": "encoder", "year": 2018})]

GPT_DOCS = [Document(text="""GPT (Generative Pre-trained Transformer) uses only the
decoder with causal attention. Each token attends only to previous tokens (autoregressive).
GPT pretraining objective: predict the next token given all previous tokens (CLM).
GPT-2 has 12-48 layers depending on size variant (117M to 1.5B parameters).
GPT-3 scaled to 175B parameters and demonstrated few-shot learning capabilities.
GPT-4 added multimodal inputs. These models excel at text generation, code, dialogue.
In-context learning allows task adaptation from examples in the prompt alone.""",
    metadata={"model": "GPT", "type": "decoder", "year": 2018})]

ALL_DOCS = BERT_DOCS + GPT_DOCS + [
    Document(text="""Attention formula: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))*V.
Scaling by 1/sqrt(d_k) prevents vanishing gradients in softmax. Multi-head attention
concatenates h parallel heads with separate learned projections W_Q, W_K, W_V, W_O.
Causal masking sets future positions to -infinity before softmax.""",
        metadata={"model": "General", "type": "mathematics", "year": 2017}),
]

splitter   = SentenceSplitter(chunk_size=200, chunk_overlap=30)
bert_index = VectorStoreIndex.from_documents(
    BERT_DOCS, transformations=[splitter], show_progress=False)
gpt_index  = VectorStoreIndex.from_documents(
    GPT_DOCS, transformations=[splitter], show_progress=False)
all_index  = VectorStoreIndex.from_documents(
    ALL_DOCS, transformations=[splitter], show_progress=False)
sum_index  = SummaryIndex.from_documents(ALL_DOCS, show_progress=False)

print(f"  Indexes built: BERT ({len(BERT_DOCS)} docs), GPT ({len(GPT_DOCS)} docs), "
      f"All ({len(ALL_DOCS)} docs), Summary")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: SubQuestionQueryEngine
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — SubQuestionQueryEngine: decompose complex queries")
print("━" * 65)
print()

bert_engine = bert_index.as_query_engine(similarity_top_k=2)
gpt_engine  = gpt_index.as_query_engine(similarity_top_k=2)

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine = bert_engine,
        name         = "bert_engine",
        description  = "Detailed technical documentation about BERT architecture and pretraining.",
    ),
    QueryEngineTool.from_defaults(
        query_engine = gpt_engine,
        name         = "gpt_engine",
        description  = "Detailed technical documentation about GPT architecture and training.",
    ),
]

sq_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools = query_engine_tools,
    use_async          = False,
    verbose            = False,
)

complex_query = "Compare the architecture and pretraining objectives of BERT and GPT"
print(f"  Complex query: '{complex_query}'")
print()

t0   = time.perf_counter()
resp = sq_engine.query(complex_query)
t_ms = (time.perf_counter() - t0) * 1000

print(f"  SubQuestion decomposition trace:")
if hasattr(resp, "metadata") and resp.metadata:
    for k, v in resp.metadata.items():
        if "sub_q" in str(k).lower():
            print(f"    {k}: {v}")
print()
print(f"  Final answer:")
print(f"    {resp.response[:200].strip()}...")
print()
print(f"  Latency: {t_ms:.0f}ms  (sub-questions can run in parallel with use_async=True)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: RouterQueryEngine
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — RouterQueryEngine: intelligent engine routing")
print("━" * 65)
print()

vec_engine_for_router = all_index.as_query_engine(similarity_top_k=3)
sum_engine_for_router = sum_index.as_query_engine(response_mode="tree_summarize")

router_tools = [
    QueryEngineTool.from_defaults(
        query_engine = vec_engine_for_router,
        name         = "vector_engine",
        description  = "Use for specific factual questions requiring precise retrieval.",
    ),
    QueryEngineTool.from_defaults(
        query_engine = sum_engine_for_router,
        name         = "summary_engine",
        description  = "Use for summarisation, overview, or questions requiring the entire document.",
    ),
]

router = RouterQueryEngine(
    selector          = LLMSingleSelector.from_defaults(llm=Settings.llm),
    query_engine_tools = router_tools,
    verbose           = False,
)

router_queries = [
    ("factual",      "What is the mathematical formula for scaled dot-product attention?"),
    ("summary",      "Give me a complete summary of all the documents about transformer models."),
    ("factual",      "How many attention heads does BERT-base have?"),
]

print(f"  RouterQueryEngine selects between:")
print(f"    vector_engine  → precise factual retrieval")
print(f"    summary_engine → full document synthesis")
print()
print(f"  {'Type':<10} {'Query':<50} {'Routed to'}")
print(f"  {'─'*80}")

for qtype, q in router_queries:
    t0   = time.perf_counter()
    resp = router.query(q)
    t_ms = (time.perf_counter() - t0) * 1000
    engine_used = getattr(resp, "metadata", {}).get("selector_result", "auto-selected")
    print(f"  {qtype:<10} {q[:48]:<50} router decision")
    print(f"  {'':>10} A: '{resp.response[:80].strip()}...'  ({t_ms:.0f}ms)")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Metadata filtering
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Metadata filtering: structured + semantic search")
print("━" * 65)
print()

# Build index with metadata-rich documents
year_docs = [
    Document(text="BERT introduced masked language modelling in 2018.",
             metadata={"year": 2018, "model_type": "encoder"}),
    Document(text="GPT-3 with 175B parameters was released in 2020.",
             metadata={"year": 2020, "model_type": "decoder"}),
    Document(text="The original Transformer was published in 2017.",
             metadata={"year": 2017, "model_type": "seq2seq"}),
    Document(text="LLaMA open-weight models were released by Meta in 2023.",
             metadata={"year": 2023, "model_type": "decoder"}),
    Document(text="GPT-2 with 1.5B parameters was released in 2019.",
             metadata={"year": 2019, "model_type": "decoder"}),
]
year_index = VectorStoreIndex.from_documents(
    year_docs,
    transformations=[SentenceSplitter(chunk_size=100, chunk_overlap=10)],
    show_progress=False,
)

# Metadata filter: only models from year >= 2020
filters_2020 = MetadataFilters(
    filters=[MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE)],
    condition=FilterCondition.AND,
)

# Filter by model type
filters_decoder = MetadataFilters(
    filters=[MetadataFilter(key="model_type", value="decoder",
                             operator=FilterOperator.EQ)],
)

print(f"  Metadata-filtered retrieval:")
for label, filt in [
    ("year >= 2020",      filters_2020),
    ("type == decoder",   filters_decoder),
]:
    retriever_f = year_index.as_retriever(
        similarity_top_k=10, filters=filt
    )
    nodes_f = retriever_f.retrieve("language model")
    years   = [n.node.metadata.get("year","?") for n in nodes_f]
    types   = [n.node.metadata.get("model_type","?") for n in nodes_f]
    print(f"  Filter [{label:<20}]: {len(nodes_f)} nodes retrieved")
    print(f"    Years: {years}")
    print(f"    Types: {types}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Chat Engine with memory
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Chat Engine: multi-turn conversation with memory")
print("━" * 65)
print()

chat_engine = all_index.as_chat_engine(
    chat_mode      = "condense_plus_context",
    memory         = ChatMemoryBuffer.from_defaults(token_limit=4096),
    verbose        = False,
    similarity_top_k = 3,
)

conversation = [
    "What is the Transformer architecture?",
    "How does its attention mechanism work?",
    "What year was it introduced and by whom?",
    "Can you now explain how BERT differs from it?",
]

print(f"  Multi-turn chat (condense_plus_context mode):")
print(f"  Each turn condenses history + new question before retrieval")
print()
for turn, msg in enumerate(conversation):
    t0   = time.perf_counter()
    resp = chat_engine.chat(msg)
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"  Turn {turn+1}: '{msg}'")
    print(f"  Answer:  '{resp.response[:90].strip()}...'  ({t_ms:.0f}ms)")
    print()

chat_engine.reset()
print(f"  chat_engine.reset() — history cleared ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: QueryFusionRetriever (multi-query + RRF)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — QueryFusionRetriever: multi-query + RRF fusion")
print("━" * 65)
print()

# Simulate QueryFusionRetriever with manual multi-query + RRF
class MockQueryFusionRetriever:
    def __init__(self, retriever, num_queries=4, top_k=5):
        self.retriever  = retriever
        self.num_queries = num_queries
        self.top_k      = top_k

    def _generate_query_variants(self, query):
        templates = [
            query,
            f"What is {query.lower().rstrip('?')}?",
            f"Explain the concept of {query.lower().rstrip('?')}",
            f"Technical overview of {query.lower().rstrip('?')}",
        ]
        return templates[:self.num_queries]

    def _rrf(self, ranked_lists, k=60):
        scores: Dict[str, float] = {}
        node_map: Dict[str, Any] = {}
        for ranked_list in ranked_lists:
            for rank, node_ws in enumerate(ranked_list):
                nid   = node_ws.node.node_id
                score = 1.0 / (k + rank + 1)
                scores[nid]   = scores.get(nid, 0) + score
                node_map[nid] = node_ws.node
        sorted_ids = sorted(scores, key=lambda x: -scores[x])
        return [NodeWithScore(node=node_map[nid], score=scores[nid])
                for nid in sorted_ids[:self.top_k]]

    def retrieve(self, query: str):
        variants = self._generate_query_variants(query)
        all_lists = []
        for v in variants:
            nodes = self.retriever.retrieve(v)
            all_lists.append(nodes)
        return self._rrf(all_lists)

base_retriever = all_index.as_retriever(similarity_top_k=3)
fusion_retriever = MockQueryFusionRetriever(base_retriever, num_queries=4, top_k=5)

fusion_query = "How does self-attention enable parallel processing in transformers?"
print(f"  Query: '{fusion_query}'")
print()

# Compare single-query vs fusion
single_nodes = base_retriever.retrieve(fusion_query)
fusion_nodes = fusion_retriever.retrieve(fusion_query)

single_ids = {n.node.node_id for n in single_nodes}
fusion_ids = {n.node.node_id for n in fusion_nodes}
new_nodes  = fusion_ids - single_ids

print(f"  Single-query retrieval:  {len(single_nodes)} nodes")
print(f"  Fusion retrieval (RRF):  {len(fusion_nodes)} nodes")
print(f"  Additional nodes found: {len(new_nodes)}  (higher recall)")
print()
print(f"  RRF score distribution (fusion_retriever):")
for i, n in enumerate(fusion_nodes[:3]):
    print(f"    [{i+1}] RRF score={n.score:.4f}  text='{n.node.text[:50].strip()}...'")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Evaluation, Production & Best Practices — RAGAS & Pipelines": {
        "description": (
            "Production-grade LlamaIndex evaluation and deployment patterns. "
            "FaithfulnessEvaluator: hallucination detection with LLM-as-judge. "
            "RelevancyEvaluator: context relevance scoring. "
            "CorrectnessEvaluator: answer accuracy against ground truth. "
            "BatchEvalRunner: parallel evaluation across a question set. "
            "RAGAS metrics: faithfulness, answer relevancy, context precision. "
            "Chunking strategy comparison: fixed vs sentence-window quality. "
            "Retrieval pipeline ablation: base vs reranker vs fusion. "
            "Async query execution for production throughput. "
            "CallbackManager for tracing and observability. "
            "Production architecture checklist and decision guide."
        ),
        "language": "python",
        "code": '''
import time
import math
import re
import hashlib
import asyncio
from typing import List, Optional, Any, Dict

try:
    from llama_index.core import (
        Document, VectorStoreIndex, Settings,
    )
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.node_parser import (
        SentenceSplitter, SentenceWindowNodeParser,
    )
    from llama_index.core.postprocessor import (
        SimilarityPostprocessor, MetadataReplacementPostProcessor,
    )
    from llama_index.core.evaluation import (
        FaithfulnessEvaluator, RelevancyEvaluator,
        CorrectnessEvaluator, EvaluationResult,
    )
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.callbacks import (
        CallbackManager, LlamaDebugHandler, CBEventType,
    )
    print("  llama-index-core: installed")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "llama-index-core", "--quiet"], check=True)
    from llama_index.core import (
        Document, VectorStoreIndex, Settings,
    )
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.node_parser import (
        SentenceSplitter, SentenceWindowNodeParser,
    )
    from llama_index.core.postprocessor import (
        SimilarityPostprocessor, MetadataReplacementPostProcessor,
    )
    from llama_index.core.evaluation import (
        FaithfulnessEvaluator, RelevancyEvaluator,
        CorrectnessEvaluator, EvaluationResult,
    )
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.callbacks import (
        CallbackManager, LlamaDebugHandler, CBEventType,
    )

print("=" * 65)
print("  EVALUATION, PRODUCTION & BEST PRACTICES")
print("=" * 65)
print()

# ── Mock classes ──────────────────────────────────────────────────────
class MockLLM(CustomLLM):
    context_window: int = 4096
    num_output: int     = 512
    model_name: str     = "mock-eval-llm"
    dummy: str          = ""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=self.context_window,
                           num_output=self.num_output, model_name=self.model_name)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        p = prompt.lower()
        # Faithfulness evaluation
        if "faithful" in p or "hallucin" in p or "contradict" in p:
            if "not mentioned" in p or "outside" in p:
                return CompletionResponse(text="NO")
            return CompletionResponse(text="YES")
        # Relevancy evaluation
        if "relevant" in p or "appropriate" in p:
            if "unrelated" in p or "irrelevant" in p:
                return CompletionResponse(text="NO")
            return CompletionResponse(text="YES")
        # Correctness evaluation
        if "correct" in p or "accurate" in p or "grade" in p:
            if "wrong" in p or "incorrect" in p:
                return CompletionResponse(text="1")
            if "partially" in p:
                return CompletionResponse(text="3")
            return CompletionResponse(text="5")
        # Standard answers
        if "attention" in p:
            return CompletionResponse(text="Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) * V. The 1/sqrt(d_k) scaling prevents gradient saturation.")
        if "bert" in p:
            return CompletionResponse(text="BERT uses the encoder-only Transformer with masked language modelling pretraining on bidirectional context.")
        if "gpt" in p:
            return CompletionResponse(text="GPT uses decoder-only Transformer with causal autoregressive language modelling.")
        if "rope" in p or "position" in p:
            return CompletionResponse(text="RoPE (Rotary Position Embedding) encodes position by rotating Q and K vectors, enabling relative position awareness and length extrapolation.")
        return CompletionResponse(text="The requested information is present in the provided context documents.")

    def stream_complete(self, prompt, **kwargs):
        r = self.complete(prompt)
        for ch in r.text:
            yield CompletionResponse(text=ch, delta=ch)


class MockEmbedding(BaseEmbedding):
    _dim: int = PrivateAttr(default=64)
    def __init__(self, dim=64, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim
    def _get_text_embedding(self, text):
        words = re.findall(r"[a-z]+", text.lower())
        vec   = [0.0] * self._dim
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for i in range(4):
                vec[(h >> (i*8)) % self._dim] += 1.0
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x/norm for x in vec]
    def _get_query_embedding(self, q): return self._get_text_embedding(q)
    async def _aget_query_embedding(self, q): return self._get_query_embedding(q)
    async def _aget_text_embedding(self, t): return self._get_text_embedding(t)


Settings.llm         = MockLLM()
Settings.embed_model = MockEmbedding(dim=64)
Settings.chunk_size  = 200

# Build evaluation corpus
EVAL_DOCS = [
    Document(text="""The Transformer architecture uses self-attention to process
sequences in parallel. It was introduced in 2017 by Vaswani et al. at Google Brain.
The attention formula is Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) * V.
The encoder processes input bidirectionally; the decoder generates output autoregressively.
Multi-head attention runs h=8 or h=12 parallel attention functions.""",
             metadata={"topic": "transformer", "year": 2017}),
    Document(text="""BERT (Bidirectional Encoder Representations from Transformers) was
released by Google AI in 2018. It uses only the encoder with masked language modelling.
BERT-base: 12 layers, 768 hidden dimensions, 12 attention heads, 110M parameters.
Pretraining on masked token prediction and next sentence prediction. Fine-tuning
involves adding task-specific heads to the pretrained encoder.""",
             metadata={"topic": "bert", "year": 2018}),
    Document(text="""Rotary Position Embedding (RoPE) encodes position by rotating
query and key vectors in 2D subspaces. The rotation angle is proportional to position.
Relative positions emerge naturally from the dot product of rotated Q and K.
RoPE is used in LLaMA, Mistral, and most modern LLMs released since 2022.
YaRN extends RoPE for length extrapolation beyond training context window.""",
             metadata={"topic": "positional", "year": 2022}),
]

splitter = SentenceSplitter(chunk_size=200, chunk_overlap=30)
index    = VectorStoreIndex.from_documents(
    EVAL_DOCS, transformations=[splitter], show_progress=False)
query_engine = index.as_query_engine(similarity_top_k=3)

print(f"  Evaluation index: {len(EVAL_DOCS)} documents, "
      f"{len(list(index.docstore.docs.values()))} nodes")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: LlamaIndex built-in evaluators
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Built-in evaluators: Faithfulness, Relevancy, Correctness")
print("━" * 65)
print()

faithfulness_eval = FaithfulnessEvaluator(llm=Settings.llm)
relevancy_eval    = RelevancyEvaluator(llm=Settings.llm)
correctness_eval  = CorrectnessEvaluator(llm=Settings.llm)

eval_cases = [
    {
        "query":     "What is the attention formula?",
        "reference": "Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) * V",
        "good_answer": "The attention formula is Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) * V. The scaling prevents gradient vanishing.",
        "bad_answer": "Attention is computed using sigmoid normalization and dot products divided by the embedding dimension.",
    },
    {
        "query":     "How many layers does BERT-base have?",
        "reference": "BERT-base has 12 transformer layers.",
        "good_answer": "BERT-base has 12 layers, 768 hidden dimensions, and 12 attention heads.",
        "bad_answer": "BERT-base has 24 transformer layers with 1024 hidden dimensions.",
    },
]

print(f"  Evaluator results (LLM-as-judge):")
print()
for case in eval_cases:
    print(f"  Query: '{case['query']}'")
    response = query_engine.query(case["query"])

    # Faithfulness: does response contain only info from context?
    faith = faithfulness_eval.evaluate_response(
        query=case["query"], response=response)
    # Relevancy: is context relevant to query?
    rel   = relevancy_eval.evaluate_response(
        query=case["query"], response=response)

    print(f"    Faithfulness: passing={faith.passing}, score={faith.score:.2f}")
    print(f"    Relevancy:    passing={rel.passing}, score={rel.score:.2f}")
    print()

    # Correctness: compare good vs bad answer against reference
    good_result = correctness_eval.evaluate(
        query=case["query"],
        response=case["good_answer"],
        reference=case["reference"],
    )
    bad_result = correctness_eval.evaluate(
        query=case["query"],
        response=case["bad_answer"],
        reference=case["reference"],
    )
    print(f"    Correctness — Good answer: score={good_result.score:.1f}/5.0")
    print(f"    Correctness — Bad answer:  score={bad_result.score:.1f}/5.0")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Chunking strategy evaluation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Chunking strategy ablation study")
print("━" * 65)
print()

# Evaluation queries with expected source topics
eval_set = [
    ("What is the scaling factor in the attention formula?", ["transformer"]),
    ("How many parameters does BERT-base have?",            ["bert"]),
    ("What year was BERT released?",                         ["bert"]),
    ("What is RoPE used for in modern LLMs?",               ["positional"]),
    ("Who introduced the Transformer architecture?",         ["transformer"]),
]

strategies = {
    "Fixed 100 tokens":  SentenceSplitter(chunk_size=100, chunk_overlap=15),
    "Fixed 200 tokens":  SentenceSplitter(chunk_size=200, chunk_overlap=30),
    "Fixed 400 tokens":  SentenceSplitter(chunk_size=400, chunk_overlap=60),
}

def recall_at_3(index, eval_set):
    retriever = index.as_retriever(similarity_top_k=3)
    hits = 0
    for q, expected_topics in eval_set:
        nodes   = retriever.retrieve(q)
        topics  = {n.node.metadata.get("topic","?") for n in nodes}
        if any(t in topics for t in expected_topics):
            hits += 1
    return hits / len(eval_set)

print(f"  Recall@3 across chunking strategies ({len(eval_set)} queries):")
print(f"  {'Strategy':<22} {'#Nodes':>8} {'Recall@3':>10} {'Avg node len'}")
print(f"  {'─'*55}")

for name, spl in strategies.items():
    strat_index = VectorStoreIndex.from_documents(
        EVAL_DOCS, transformations=[spl], show_progress=False)
    nodes_list  = list(strat_index.docstore.docs.values())
    avg_len     = sum(len(n.text.split()) for n in nodes_list) / max(len(nodes_list),1)
    recall      = recall_at_3(strat_index, eval_set)
    print(f"  {name:<22} {len(nodes_list):>8} {recall:>10.3f} {avg_len:>12.0f} words")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Retrieval pipeline ablation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Retrieval pipeline ablation: base vs augmented")
print("━" * 65)
print()

import json as _json

def evaluate_pipeline(retriever, eval_set, label):
    hits, total = 0, 0
    times       = []
    for q, expected in eval_set:
        t0    = time.perf_counter()
        nodes = retriever.retrieve(q)
        times.append((time.perf_counter() - t0)*1000)
        topics = {n.node.metadata.get("topic","?") for n in nodes}
        if any(t in topics for t in expected):
            hits += 1
        total += 1
    recall   = hits / total
    avg_time = sum(times) / len(times)
    return recall, avg_time

base_index     = VectorStoreIndex.from_documents(
    EVAL_DOCS, transformations=[SentenceSplitter(chunk_size=200, chunk_overlap=30)],
    show_progress=False)
base_retriever = base_index.as_retriever(similarity_top_k=3)

# Augmented: retrieve top-5, filter by score >= 0.3, take top-3
filtered_retriever_obj = base_index.as_retriever(similarity_top_k=5)

class FilteredRetriever:
    def __init__(self, base, cutoff=0.3, top_n=3):
        self.base   = base
        self.cutoff = cutoff
        self.top_n  = top_n
    def retrieve(self, q):
        nodes    = self.base.retrieve(q)
        filtered = [n for n in nodes if n.score and n.score >= self.cutoff]
        return filtered[:self.top_n]

filtered_retriever = FilteredRetriever(filtered_retriever_obj, cutoff=0.3, top_n=3)

pipelines = [
    ("Base (top_k=3)",          base_retriever),
    ("Filtered (k=5→score≥0.3→top3)", filtered_retriever),
]

print(f"  Pipeline comparison on {len(eval_set)} queries:")
print(f"  {'Pipeline':<38} {'Recall@3':>10} {'Avg ms':>10}")
print(f"  {'─'*62}")

for label, retr in pipelines:
    recall, avg_ms = evaluate_pipeline(retr, eval_set, label)
    print(f"  {label:<38} {recall:>10.3f} {avg_ms:>10.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Async query for production throughput
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Async query engine for production throughput")
print("━" * 65)
print()

qe = base_index.as_query_engine(similarity_top_k=3)

async def async_batch_query(queries):
    tasks   = [qe.aquery(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results

batch_queries = [
    "What is the Transformer architecture?",
    "How does BERT pretraining work?",
    "What is the attention formula?",
    "What is RoPE?",
    "Who introduced the Transformer?",
]

# Sequential (sync)
t0 = time.perf_counter()
sync_results = [qe.query(q) for q in batch_queries]
t_sync = (time.perf_counter() - t0) * 1000

# Parallel (async)
t0 = time.perf_counter()
async_results = asyncio.get_event_loop().run_until_complete(
    async_batch_query(batch_queries))
t_async = (time.perf_counter() - t0) * 1000

print(f"  Batch of {len(batch_queries)} queries:")
print(f"  {'─'*45}")
print(f"    Synchronous (.query):    {t_sync:>8.0f}ms total")
print(f"    Async (.aquery/gather):  {t_async:>8.0f}ms total")
if t_async > 0:
    print(f"    Speedup:                 {t_sync/t_async:>8.1f}×")
print()
print(f"  Production rule: always use aquery() / aretrieve() for concurrent users.")
print(f"  Async enables 10-50× higher throughput without any code restructuring.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: RAGAS evaluation concept
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — RAGAS: the standard RAG evaluation framework")
print("━" * 65)
print()

RAGAS_CODE = """
# RAGAS (Retrieval Augmented Generation Assessment) — industry standard eval

# Install: pip install ragas llama-index-core

from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # response only contains info from context
    answer_relevancy,      # response addresses the question
    context_precision,     # retrieved context is relevant to query
    context_recall,        # enough context retrieved to answer (needs ground truth)
)
from ragas.integrations.llama_index import evaluate as li_evaluate

# Prepare evaluation dataset
eval_dataset = {
    "question":   ["What is attention?", "How does BERT work?"],
    "answer":     ["Attention is...",    "BERT uses..."],
    "contexts":   [["chunk1", "chunk2"], ["chunk3"]],
    "ground_truth": ["Attention(Q,K,V)=...", "BERT uses MLM..."],
}

# Run evaluation against your query engine directly
result = li_evaluate(
    query_engine = query_engine,
    metrics      = [faithfulness, answer_relevancy, context_precision],
    questions    = eval_dataset["question"],
    ground_truth = eval_dataset["ground_truth"],
)

# Results as a dataframe
df = result.to_pandas()
print(df[["question", "faithfulness", "answer_relevancy", "context_precision"]])

# Interpret scores:
# faithfulness:      1.0 = fully grounded, 0.0 = pure hallucination
# answer_relevancy:  1.0 = directly answers question, 0.0 = off-topic
# context_precision: 1.0 = all retrieved chunks relevant, 0.0 = noisy retrieval
# context_recall:    1.0 = all needed info retrieved, 0.0 = missing context
"""
print(RAGAS_CODE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Complete production guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Production architecture and quality checklist")
print("━" * 65)
print()

PROD_ARCHITECTURE = """
LlamaIndex Production RAG Architecture:

 DATA INGESTION (offline, once)
 ┌─────────────────────────────────────────────────────────────┐
 │  Load  → SentenceWindowNodeParser (window=3)               │
 │       → SummaryExtractor + QuestionsAnsweredExtractor       │
 │       → Embed (text-embedding-3-small / BGE-large)          │
 │       → VectorStoreIndex (Chroma / Pinecone / pgvector)     │
 │       → IngestionPipeline.persist() → cached for re-use     │
 └─────────────────────────────────────────────────────────────┘

 QUERY PIPELINE (online, per request)
 ┌─────────────────────────────────────────────────────────────┐
 │  Query → [Optional] rephrase with chat history             │
 │       → QueryFusionRetriever (4 variants, RRF, top-20)     │
 │       → MetadataReplacementPostProcessor (window swap)     │
 │       → CohereRerank or LLMRerank (top-20 → top-5)         │
 │       → SimilarityPostprocessor (cutoff=0.5)               │
 │       → compact or tree_summarize synthesis                 │
 │       → Return response + source_nodes + metadata          │
 └─────────────────────────────────────────────────────────────┘
"""
print(PROD_ARCHITECTURE)

print(f"  Quality improvement levers (ranked by typical impact):")
levers = [
    ("1 (highest)", "SentenceWindowNodeParser",    "vs fixed-size: +15-25% faithfulness"),
    ("2",           "CohereRerank / LLMRerank",     "top-20→rerank→top-5: +10-20% accuracy"),
    ("3",           "QueryFusionRetriever",          "+5-15% recall from multi-query"),
    ("4",           "QuestionsAnsweredExtractor",    "metadata-enhanced retrieval"),
    ("5",           "Increase similarity_top_k",     "more candidates for reranking"),
    ("6",           "Metadata + AutoRetriever",      "structured + semantic filtering"),
    ("7",           "tree_summarize mode",           "better multi-doc synthesis"),
    ("8",           "SubQuestionQueryEngine",        "complex multi-hop questions"),
    ("9 (lowest)",  "Better embedding model",        "marginal if architecture solid"),
]
for rank, name, impact in levers:
    print(f"    [{rank:<10}] {name:<30} {impact}")
print()
print(f"  ┌───────────────────────────────────────────────────────────────────┐")
print(f"  │ Settings            │ Development      │ Production               │")
print(f"  ├───────────────────────────────────────────────────────────────────┤")
print(f"  │ LLM                 │ gpt-4o-mini      │ gpt-4o / claude-3.5      │")
print(f"  │ Embedding           │ text-emb-3-small │ text-emb-3-large / BGE   │")
print(f"  │ Vector store        │ FAISS (in-mem)   │ Pinecone / pgvector      │")
print(f"  │ Chunk strategy      │ SentenceSplitter │ SentenceWindowParser     │")
print(f"  │ Retrieval           │ top_k=3          │ top_k=20 + reranker      │")
print(f"  │ Response mode       │ compact          │ compact / tree_summarize │")
print(f"  │ Async               │ optional         │ always (aquery)          │")
print(f"  │ Evaluation          │ manual spot-check│ RAGAS automated suite    │")
print(f"  └───────────────────────────────────────────────────────────────────┘")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


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