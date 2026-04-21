"""
TOPIC TEMPLATE
==============
Copy this file and rename it:  NN_topic_name.py
Fill in each section. The app will auto-discover this module.

Naming convention:
  01_tokenization_embeddings.py
  02_language_modeling.py
  ...
"""

# ── Display name (shown in sidebar and as page title) ──────────────────────
TOPIC_NAME   = "Topic Name"
DISPLAY_NAME = "Topic Name"
ICON         = "📖"
SUBTITLE     = "One-line description of what this topic covers"

# ── Theory ─────────────────────────────────────────────────────────────────
THEORY = """

Naive RAG Core Features:

Key design decisions at each step:

    +-----------------+---------------------------------+-------------------------------------+
    | Step            | Common Choices                  | What to Watch Out For               |
    +-----------------+---------------------------------+-------------------------------------+
    | Chunking        | Fixed-size, sentence, recursive | Chunk size greatly affects quality  |
    |                 |                                 |                                     |
    | Embedding       | OpenAI, BGE, Cohere             | Query & doc model must match        |
    |                 |                                 |                                     |
    | Vector Store    | FAISS (local), Chroma, Pinecone | Scalability vs. simplicity tradeoff |
    |                 |                                 |                                     |
    | Retrieval (k)   | top-3 to top-5                  | Too many chunks = noisy context     |
    |                 |                                 |                                     |
    | LLM             | GPT-4o, Claude, Llama           | Context window limits chunk count   |
    +-----------------+---------------------------------+-------------------------------------+



    # ──────────────────────────────────────────────────────────────────────── #
    # ─────────────────────────   EMBEDDINGS:   ────────────────────────────── #
    # ──────────────────────────────────────────────────────────────────────── #


#####  1 . What Is an Embedding?

An embedding is a mathematical representation of a piece of text — or any other data — expressed 
as a fixed-length vector of floating-point numbers in a high-dimensional space. 

At its core, an embedding converts something symbolic and discrete (words, sentences, documents) 
into something continuous and geometric (coordinates in a mathematical space), where the geometric 
properties of that space mirror the semantic properties of the original data.

The fundamental insight is this: 
    
    If two pieces of text mean similar things, they should occupy nearby positions in the embedding space. 
    The distance between two vectors becomes a proxy for semantic distance between concepts.
     
This transforms the problem of 'finding relevant text' into the problem of 'finding 
nearby points in space' — a problem that mathematics, linear algebra, and efficient data 
structures have already solved elegantly.


╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              EMBEDDING METHODS — COMPLETE TAXONOMY                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

  PRE-COMPUTED VECTOR REPRESENTATIONS (stored at index time)
  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
  │       SPARSE        │  │    STATIC DENSE     │  │  CONTEXTUAL DENSE   │  │    MULTI-VECTOR     │
  │  Keyword precision  │  │  Word-level, static │  │    Context-aware    │  │    Token-level      │
  ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
  │ TF-IDF              │  │ Word2Vec            │  │ BERT (raw)          │  │ ColBERT         *   │
  │ BM25            *   │  │ GloVe               │  │ Sentence-BERT       │  │ ColBERT v2 / PLAID  │
  │ SPLADE (neural)     │  │ FastText            │  │ BGE-Large       *   │  │ ME-BERT             │
  │                     │  │                     │  │ OpenAI text-emb-3   │  │                     │
  │ → no training req.  │  │ → fixed vec/word    │  │ E5-Mistral-7B       │  │ → MaxSim scoring    │
  │ → inverted index    │  │ → 100–300 dims      │  │                     │  │ → 25–100× storage   │
  │ → 10k–500k dims     │  │ → no OOV (W2V/GloVe)│  │ → 384–4096 dims     │  │ → per-token vectors │
  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘

  RETRIEVAL STRATEGIES AND ENHANCEMENTS
  ┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐
  │       CROSS-ENCODER          │  │      HYBRID RETRIEVAL        │  │     MATRYOSHKA (MRL)         │
  │      Re-ranking only         │  │    Sparse + dense fusion     │  │     Tunable dimensions       │
  ├──────────────────────────────┤  ├──────────────────────────────┤  ├──────────────────────────────┤
  │ MS-MARCO MiniLM              │  │ BM25 + bi-encoder        *   │  │ OpenAI text-emb-3            │
  │ Cohere Rerank 3              │  │ BGE-M3 (unified)             │  │ nomic-embed-v1.5             │
  │ Jina Reranker v2             │  │ RRF fusion                   │  │ Truncate to any size         │
  │                              │  │                              │  │                              │
  │ → cannot precompute docs     │  │ → best of both worlds        │  │ → train once, resize freely  │
  │ → O(k) passes at query time  │  │ → consistently beats either  │  │ → 3072 → 256 with <5% loss   │
  │ → used for top-k rerank only │  │ → gold standard for prod.    │  │ → adaptive cascade retrieval │
  └──────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────────┘

  ★ = production gold standard


QUALITY LADDER (retrieval quality ↑ / cost ↑)
    
    0  TF-IDF             Classic sparse. No training. Exact keywords only.
    1  BM25               Probabilistic sparse. TF saturation + length norm. Industry default.
    2  SPLADE             Neural sparse. Vocabulary expansion. Inverted-index compatible.
    3  Static Dense       Word2Vec / GloVe / FastText. Fast, word-level, no context.
    4  Contextual Dense   SBERT / BGE. Bi-encoder. Sentence-level semantics. Production standard.
    5  Hybrid (BM25+Dense) Sparse + dense fused via RRF. Best single-stage quality.
    6  Multi-Vector       ColBERT / PLAID. Per-token MaxSim. High recall, high storage.
    7  + Cross-Encoder    Add reranker on top-k. Near-perfect precision on candidate set.
    8  + Contextual Retr. LLM-prepended context prefix per chunk. 47–70% fewer retrieval failures.
    9  + MRL Cascade      Full-dim rerank after cheap truncated first-pass. Best cost/quality ratio.
    

PRODUCTION TIER GUIDE
    
    TIER 1 — Standard     BM25 + Dense bi-encoder (BGE-Large or MiniLM) + RRF
    (low cost)            Token chunking + HNSW index + basic deduplication
    
    TIER 2 — High quality Dense + BM25 Hybrid + RRF + Cross-Encoder rerank
    (moderate cost)       + Contextual Retrieval (Anthropic 2024) + deduplication
    
    TIER 3 — Maximum      ColBERT or BGE-M3 (all 3 signals) + PLAID compression
    (high cost, offline)  + Contextual Retrieval + HyDE at query time + CE rerank
                        + RAPTOR tree for synthesis queries


    EMBEDDING & RETRIEVAL METHODS (UNIFIED VIEW)
    │
    ├── 📦 1. sparse_embeddings (Lexical / Exact Match)
    │   ├── 📄 tf_idf                      Frequency × rarity (baseline - Keywords) 
    │   ├── 📄 bm25                    ★   Probabilistic Industry Search Standard ranking
    │   └── 📄 splade_v3                   Neural sparse (semantic expansion), avoids "zero-hit" terms
    │
    ├── 📦 2. static_dense_embeddings (Legacy/Edge, Token-Level, No Context)
    │   ├── 📄 word2vec                   Local context prediction (CBOW/Skip-gram)
    │   ├── 📄 glove                      Global co-occurrence statistics
    │   └── 📄 fasttext                   Subword-aware (handles OOV/typos)
    │
    ├── 📦 3. contextual_dense_embeddings (Bi-Encoders)
    │   ├── 📄 bge_v1_5 / gte_large       Strong open baselines (BERT-style)
    │   ├── 📄 sentence_bert              Siamese, contrastive learning
    │   ├── 📄 bge_large                ★ Open SOTA, production-friendly
    │   ├── 📄 openai_text_embedding_3    API-based, MRL-native, Managed, high performance
    │   └── 📄 e5 / qwen3_embedding_8b  ★ LLM-backbone, long-context (32k–128k)
    │
    ├── 📦 4. instruction_based_embeddings (Task-Aware)
    │   ├── 📄 instructor_xl              Prompt-conditioned embeddings, Prefix-based (e.g., "Represent for...")
    │   ├── 📄 voyage_large           ★   Domain-specialized (legal, finance, etc.)
    │   └── 📄 bge_en_icl                 In-context learning for retrieval
    │
    ├── 📦 5. multi_vector / late_interaction (Fine-Grained Matching - Token-Granularity)
    │   ├── 📄 colbert / colbert_v2   ★   Token-level MaxSim scoring
    │   ├── 📄 colbert_plaid              Compressed late interaction
    │   ├── 📄 x_pi                       Efficient Compressed parallel interaction
    │   └── 📄 me_bert/                   Multi-vector per document
    │           mixedbread_whole_embed    Unified multimodal interaction  
    │
    ├── 📦 6. graph_knowledge_embeddings (Relational / Structured)
    │   ├── 📄 graph_sage                 Graph neighborhood aggregation
    │   ├── 📄 node2vec                   Random-walk graph embeddings
    │   └── 📄 hippo_retrieval            Hierarchical retrieval indexing
    │
    ├── 📦 7. cross_encoders (Rerankers / High Precision)
    │   ├── 📄 ms_marco_minilm            Fast, CPU-friendly reranker
    │   ├── 📄 cohere_rerank_v4       ★   Commercial SOTA reasoning
    │   ├── 📄 jina_reranker_v2           Long-context reranking - Open weights, 10k+ context
    │   └── 📄 rank_llama                 LLM-based zero-shot reranker - LLM as a zero-shot reranker
    │
    ├── 📦 8. hybrid_retrieval (Fusion Strategies)
    │   ├── 📄 bm25 + biencoder       ★   Gold standard (recall + semantics)
    │   ├── 📄 rrf_fusion                 Rank-based merging (robust)
    │   ├── 📄 weighted_combination       Score interpolation (α tuning)
    │   └── 📄 bge_m3_unified         ★   Dense + sparse + multi-vector (1 model)
    │
    ├── 📦 9. matryoshka_mrl (Adaptive Embeddings)
    │   ├── 📄 openai_text_embedding_3   Multi-resolution vectors
    │   ├── 📄 nomic_embed_v1_5          Open-source(Open weights) MRL-native
    │   └── 📄 adaptive_cascade          Cheap → expensive staged retrieval
    │
    └── 📦 10. compression & scaling (Infra Layer)
        ├── 📄 pq / ivf                  Product quantization (FAISS)
        ├── 📄 bitwise_quantization      1-bit / binary embeddings
        └── 📄 distillation              Smaller student embedding models

    
    
    ═════════════════════════════════════════════════════════════════════
     CROSS-CUTTING CONCERNS
    ═════════════════════════════════════════════════════════════════════
    
      DISTANCE METRIC     →  Cosine similarity (default) / dot product / L2
      POOLING STRATEGY    →  Mean pooling (default) / [CLS] token / max pooling
      NORMALIZATION       →  L2-normalize before storage (cosine = dot product)
      TOKEN LIMIT         →  Never chunk larger than model max — silent truncation
      QUERY/DOC SYMMETRY  →  Query and document MUST use the same embedding model
      HARD NEGATIVES      →  BM25 top-k misses + dense top-k misses + LLM-generated
      CHUNK SIZE          →  Factoid: 128–256 tok │ Standard: 256–512 │ Reasoning: 512–1024
    
    
    ═════════════════════════════════════════════════════════════════════
     AWARENESS LADDER  (bottom = simplest / cheapest → top = best / costliest)
    ═════════════════════════════════════════════════════════════════════
    
      9 ║  MRL CASCADE            Cheap truncated first-pass → full-dim rerank
      8 ║  CONTEXTUAL RETRIEVAL   LLM prepends document-context prefix per chunk
      7 ║  HYBRID + RERANK        BM25 + Dense + RRF + Cross-Encoder on top-k
      6 ║  MULTI-VECTOR           ColBERT per-token MaxSim, PLAID compression
      5 ║  CROSS-ENCODER          Joint query+doc encoding, rerank only
      4 ║  HYBRID RETRIEVAL       Sparse + dense fused via RRF — production default
      3 ║  CONTEXTUAL DENSE       SBERT / BGE bi-encoder, contrastive training
      2 ║  SPLADE                 Neural sparse with vocabulary expansion
      1 ║  BM25                   Probabilistic TF saturation + length norm
      0 ║  TF-IDF                 Classic frequency × rarity — no training needed
    
    
    ★ = production gold standard

---

#### The Core Intuition — Why Geometry Captures Meaning

Imagine you had to assign coordinates to every word in the English language on a giant map. 
Words that are related should be close together. Words that are unrelated should be far apart. 

An embedding model is a neural network that has learned exactly this map — not just for words, 
but for entire sentences and documents. The model has been trained on billions of examples and 
has discovered a geometry where semantic relationships become spatial relationships.

    Famous example (Word2Vec, 2013):  vector('King') - vector('Man') + vector('Woman')  ≈  vector('Queen')

This arithmetic works because the model learned that the direction from 'Man' to 'King' encodes 
the concept of royalty, and that same directional offset applied to 'Woman' reaches 'Queen'. 
Meaning has become direction in space.


### **Where Embeddings Fit in the RAG Pipeline**

Embeddings are the bridge between raw text and mathematical retrieval. They live
between chunking (Module 01) and the vector database. The full RAG stack:

    Raw Document
        │
        ▼
    CHUNKING (Module 01)      →  break text into retrievable units
        │
        ▼
    EMBEDDING (Module 02)     →  convert each chunk into a vector
        │
        ▼
    VECTOR DATABASE           →  store and index all vectors
        │
        ▼ (at query time)
    EMBED QUERY               →  convert user question into vector
        │
        ▼
    ANN RETRIEVAL             →  find nearest chunk vectors
        │
        ▼
    LLM GENERATION            →  answer using retrieved chunks as context


### 1.1  The Three Roles of Embeddings in RAG

In a Retrieval-Augmented Generation pipeline, embeddings serve three distinct and crucial roles:


    ┌───────────────────────┬─────────────────────────────────────────────────────┐
    │ Role                  │ Explanation                                         │
    ├───────────────────────┼─────────────────────────────────────────────────────┤
    │ Indexing (Offline)    │ At ingestion time, every chunk of your corpus is    │
    │                       │ passed through an embedding model to produce a      │  
    │                       │ dense vector.                                       │
    │                       │                                                     │  
    │                       │ These vectors are stored in a vector database       │  
    │                       │ (Pinecone, Weaviate, Chroma, FAISS).                │
    │                       │ This is done once and persisted to disk.            │
    │                       │                                                     │
    ├───────────────────────┼─────────────────────────────────────────────────────┤
    │ Retrieval (Online)    │ At query time, the user's question is embedded      │
    │                       │ using the same model (this is critical — query      │
    │                       │ and document must use the same embedding space).    │
    │                       │                                                     │
    │                       │ The query vector is compared against all stored     │
    │                       │ document vectors to find the most semantically      │  
    │                       │ similar chunks.                                     │
    │                       │                                                     │
    ├───────────────────────┼─────────────────────────────────────────────────────┤
    │ Ranking (Optional)    │ A second-stage re-ranker (often a cross-encoder)    │
    │                       │ rescores the top-k retrieved chunks with higher     │
    │                       │ accuracy to reorder them before passing to the LLM. │
    │                       │                                                     │
    │                       │ This decouples retrieval speed from ranking quality │
    │                       │                                                     │
    └───────────────────────┴─────────────────────────────────────────────────────┘
    
    ⚠️  CRITICAL RULE: The query and document MUST use the same embedding model.
        Mixing models from different families produces meaningless similarity scores.

### 1.2  The Embedding Pipeline — End to End

Before diving into specific methods, it is essential to understand the full lifecycle of an 
embedding in a RAG system. 

The following steps constitute the complete embedding pipeline:

    1.	TEXT INPUT  →   Raw text arrives — a document chunk, a sentence, or a query.
    
    2.	TOKENIZATION    →   The text is tokenized into subword units (BPE, WordPiece, or SentencePiece)
                            according to the model's vocabulary. For example, 'embeddings' might be split
                            into ['embed', '##ding', '##s'].
        
    3.	ENCODING    →   Tokens are passed through the model (an MLP, RNN, or Transformer).
                        Each token is processed with respect to all other tokens in the input (attention).
        
    4.	POOLING     →   The per-token hidden states are aggregated into a single fixed-size vector.
                        Common strategies: [CLS] token, mean pooling, max pooling, weighted pooling.
        
    5.	NORMALIZATION   →   The resulting vector is often L2-normalized so that its magnitude is
                            1.0 and cosine similarity equals the dot product. 
                            This is nearly universal in modern practice.
        
    6.	STORAGE/COMPARISON  →   The normalized vector is stored (indexing) or compared against
                                stored vectors (retrieval) using a distance metric.

### 1.3  Distance Metrics

The choice of distance metric determines how 'similarity' is measured between two embedding vectors. 
This choice is not arbitrary — it must match what the model was trained to optimize:

---

    **Cosine Similarity**
    
    cos(A, B) = (A · B) / (||A|| × ||B||)
     
      Range: [-1.0, +1.0]  |  Perfect similarity = 1.0  |  Orthogonal = 0.0  |  Opposite = -1.0
     
      Most common for text. Measures the angle between vectors, not magnitude.
      
      After L2 normalization: cos(A,B) = A · B  (dot product becomes equivalent)

---

    Euclidean Distance (L2)
    
    d(A, B) = sqrt( sum_i (A_i - B_i)^2 )
     
      Range: [0, +inf)  |  Perfect similarity = 0.0
     
      Appropriate when magnitude of vectors carries information (rare in text embeddings).
      
      Used in some image/multimodal embedding spaces.
    
---

    Dot Product
    
    score(A, B) = A · B = sum_i (A_i × B_i)
     
      Equivalent to cosine similarity when vectors are L2-normalized.
      
      Faster to compute than cosine (no normalization step needed at query time).
      
      Used in Maximum Inner Product Search (MIPS) — the retrieval problem solved by FAISS.

---

#####  2 . The Embedding Taxonomy


### **THE EMBEDDING TAXONOMY**

The landscape of embedding methods is broad and has evolved significantly over the past decade.

    ╔══════════════════════════════════════════════════════════════════════════════════════════╗
    ║                               EMBEDDING METHOD TAXONOMY                                  ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════╝
                                                │
          ┌───────────────┬────────────────┬────┴─────────────┬────────────────┬─────────────┐
          │               │                │                  │                │             │
          ▼               ▼                ▼                  ▼                ▼             ▼
    ┌──────────┐   ┌───────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────┐  ┌──────────┐
    │  SPARSE  │   │  STATIC   │   │  CONTEXTUAL  │   │   MULTI   │   │  CROSS-  │  │MATRYOSHKA│
    │ METHODS  │   │  DENSE    │   │    DENSE     │   │  VECTOR   │   │ ENCODER  │  │   MRL    │
    │          │   │           │   │              │   │           │   │          │  │          │
    │TF-IDF    │   │Word2Vec   │   │BERT          │   │ColBERT    │   │BERT      │  │OpenAI-3  │
    │BM25      │   │GloVe      │   │SBERT         │   │ME-BERT    │   │Reranker  │  │nomic-    │
    │SPLADE    │   │FastText   │   │BGE/OpenAI    │   │PLAID      │   │Cohere    │  │embed     │
    │          │   │           │   │E5-Mistral    │   │           │   │Rerank    │  │          │
    │High-dim  │   │Low-dim    │   │Mid-dim dense │   │Per-token  │   │Scalar    │  │Truncat-  │
    │sparse    │   │dense      │   │(384-4096)    │   │vectors    │   │score     │  │able dims │
    │(10k-500k)│   │(100-300)  │   │              │   │(128 each) │   │          │  │          │
    └──────────┘   └───────────┘   └──────────────┘   └───────────┘   └──────────┘  └──────────┘
    Keyword        Fast,           Semantic            Token-level     Highest       Tunable
    precision      lightweight     understanding       granularity,    accuracy,     cost/quality,
    interpretable  word-level      context-aware       high recall     very slow     flexible storage


    FULL TAXONOMY TABLE:
    ┌──────────────────────┬──────────────────────┬────────────────────┬────────────────────────┐
    │ Category             │ Methods              │ Vector Type        │ Strengths              │
    ├──────────────────────┼──────────────────────┼────────────────────┼────────────────────────┤
    │ Sparse               │ TF-IDF, BM25, SPLADE │ High-dim sparse    │ Keyword precision,     │
    │                      │                      │ (10k-100k dims)    │ interpretable, no train│
    ├──────────────────────┼──────────────────────┼────────────────────┼────────────────────────┤
    │ Static Dense         │ Word2Vec, GloVe,     │ Low-dim dense      │ Fast, lightweight,     │
    │                      │ FastText             │ (100-300 dims)     │ word-level understand. │
    ├──────────────────────┼──────────────────────┼────────────────────┼────────────────────────┤
    │ Contextual Dense     │ BERT, SBERT, OpenAI  │ Mid-dim dense      │ Semantic understanding,│
    │                      │                      │ (384-3072 dims)    │ context-aware          │
    ├──────────────────────┼──────────────────────┼────────────────────┼────────────────────────┤
    │ Multi-Vector         │ ColBERT, ME-BERT,    │ Per-token vectors  │ Token-level granularity│
    │                      │ PLAID                │ (128 dims each)    │ high recall            │
    ├──────────────────────┼──────────────────────┼────────────────────┼────────────────────────┤
    │ Cross-Encoder        │ BERT reranker,       │ Scalar score (not  │ Highest accuracy,      │
    │                      │ Cohere Rerank        │ a vector)          │ expensive              │
    ├──────────────────────┼──────────────────────┼────────────────────┼────────────────────────┤
    │ Sparse-Dense Hybrid  │ Pinecone Hybrid,     │ Two separate       │ Best of both           │
    │                      │ Weaviate BM25+       │ vectors            │ precision types        │
    ├──────────────────────┼──────────────────────┼────────────────────┼────────────────────────┤
    │ Matryoshka (MRL)     │ OpenAI text-emb-3,   │ Truncatable dense  │ Tunable cost/quality,  │
    │                      │ E5-Mistral           │ (from 3072)        │ flexible storage       │
    └──────────────────────┴──────────────────────┴────────────────────┴────────────────────────┘

---

### **3. SPARSE EMBEDDING METHODS**

Sparse embeddings are the oldest class of text representation methods. 
They produce vectors where the vast majority of dimensions are zero, with only a small 
number of non-zero values. 
The non-zero dimensions correspond directly to vocabulary terms (or term features), 
making these representations directly interpretable by humans. 
You can look at a sparse vector and immediately understand which words contributed to it.

The term 'sparse' refers to the structure of the vector: a vocabulary of 100,000 
terms produces a 100,000-dimensional vector, but a given document might only contain 
50 unique terms — meaning 99,950 dimensions are zero.


## **3.1 TF-IDF — Term Frequency-Inverse Document Frequency**

TF-IDF is the foundational sparse embedding method, introduced in the 1970s and still actively 
used today in many search systems. 

It quantifies the importance of a term to a document within a corpus using a two-component score:

    Term Frequency (TF):      How often does this word appear in THIS document?
                              Words that appear many times in a document are likely important to that document.  
    
    Inverse Document Freq (IDF): How rare is this word across the ENTIRE corpus?
                                 Words that appear in many documents (like 'the', 'is', 'and') carry very little 
                                 distinguishing information and should be downweighted.   

The product of these two components creates a score that rewards words that are frequent in a 
specific document but rare across the collection — 
exactly the words that identify what makes that document unique.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  TF-IDF FORMULA                                                          │
    │                                                                          │
    │  TF(t, d)  = count(t in d) / total_tokens(d)                             │
    │              Raw:           TF(t, d) = count(t in d)                     │
    │              Log-norm:      TF(t, d) = log(1 + count(t in d))            │
    │              Double-norm:   TF(t, d) = 0.5 + 0.5 × (count(t,d)/max(d))   │
    │                                                                          │
    │  IDF(t, D) = log( N / df(t) ) + 1                                        │
    │              N = total documents                                         │   
    │              df(t) = docs containing term t                              │ 
    │              Smooth IDF (sklearn): log( (1+N)/(1+df(t)) ) + 1            │
    │                                                                          │ 
    │  Smooth IDF variant (sklearn default):                                   │ 
    │  IDF(t) = log( (1 + N) / (1 + df(t)) ) + 1                               │ 
    │                                                                          │ 
    │  TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)                                  │
    │                                                                          │
    │  Dimensionality: |V| where V = vocabulary size (10,000 - 500,000)        │
    │  Sparsity:       99%+ zeros for any typical document                     │
    └──────────────────────────────────────────────────────────────────────────┘


### **How TF-IDF Retrieval Works**

To retrieve documents relevant to a query using TF-IDF, both the query and all documents 
are represented as TF-IDF vectors. 
The query is typically treated as a short document and scored with TF (no IDF weighting on 
query terms by default). 
Retrieval is then performed by computing the cosine similarity between the query vector and 
each document vector. 
Because the vectors are sparse, this computation is highly efficient — you only need to 
consider dimensions where both vectors are non-zero, i.e., terms that appear in both the 
query and the document.


**Practical Dimensionality:**

    Corpus Size             Typical Dimensionality
    ────────────────────────────────────────────────
    Small (<10K docs)       ~10,000 - 50,000 dims
    Medium (100K docs)      ~50,000 - 200,000 dims
    Large (1M+ docs)        ~200,000 - 500,000 dims
    
    After LSA/SVD reduction 100 - 1,000 dims


**Strengths and Limitations: Positive and Negative**

    + Fully interpretable            You can read the non-zero values and understand
                                     exactly which terms drove the score.
                                     
    + No training required           Works out-of-the-box on any corpus.
    
    + Blazing fast at scale          Inverted index structures → sub-millisecond ops.
    
    + Perfect exact keyword recall   If a document contains the query term, it will be found.


    - Vocabulary mismatch            'Car' and 'automobile' have zero vector overlap.
    
    - No semantic understanding      Cannot understand 'happy' and 'joyful' are related.
    
    - Order-blind (bag of words)     'Dog bites man' and 'Man bites dog' → identical vector.
    
    - Fails on OOV terms             New words not seen during indexing have no representation.


### **3.2 BM25 — Best Match 25**

**What Is BM25?**

BM25 (Best Match 25) is the de facto standard sparse retrieval algorithm used in essentially 
all production search engines today, including Elasticsearch, Lucene (Apache Solr), and OpenSearch. 

It was developed by Robertson and colleagues at Okapi in the 1990s as part of a series of probabilistic 
retrieval models (BM1, BM11, BM15, BM25...). 

The '25' refers to the 25th iteration of their tuning process.

BM25 fixes the most critical weaknesses of TF-IDF: 
    → it introduces saturation on term frequency 
      (so a term appearing 100 times vs. 50 times doesn't matter much — what matters is that it appears at all)
       
    → Normalizes for document length 
      (so a short, focused document about a topic outscores a long document that mentions the topic once in passing).

    Simplified:
    → Introduces TF saturation (repetition has diminishing returns)
    → Normalizes for document length (short focused docs outrank long diluted ones)
    
    
## **Mathematical Formula**

    BM25 Scoring Function
    Score(d, q) = (sum)Σ over t in q of:
     
      IDF(t) × [ tf(t,d) × (k1 + 1) ] / [ tf(t,d) + k1 × (1 - b + b x |d|/avg(dl) ]
     
    Where:
        tf(t, d) = frequency of term t in document d
        
        |d|      = length of document d in tokens
        
        avgdl    = average document length across corpus
        
        k1       = term frequency saturation parameter  (typically 1.2 - 2.0)
        
        b        = length normalization parameter        (typically 0.75)
     
    IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
     
        N    = total documents | df(t) = docs containing t
     
    Key insight — TF saturation curve:
        As tf(t,d) → infinity, the score component → (k1+1) not infinity
        
        This caps the reward for repetition. 
        
        A term appearing 1000x scores only marginally better than one appearing 10x.



**The Two Key Parameters:**

    k1  (TF saturation)           Controls how quickly term frequency effect saturates.
                                  k1=0: TF ignored entirely (pure IDF).
                                  k1=1.2: standard for general search.      
                                  k1=1.5-2.0: technical/code search (repetition matters).

    b   (length normalization)    Controls how strongly document length is penalized.
                                  b=0: length normalization disabled.
                                  b=1: full length normalization.
                                  b=0.75: standard compromise.

    BM25+ variant                Adds small constant delta (typically 1.0) to TF component.
                                  Prevents zero-frequency terms from scoring zero.

    BM25F variant                Field-weighted BM25 for structured docs (title, body).
                                  Different k1 and b per field. Match in title > match in body.


**BM25 vs TF-IDF:**

    Dimension               TF-IDF                      BM25
    ──────────────────────────────────────────────────────────────────────
    TF Treatment            Linear (100× = 10× better)  Saturating (dim. returns)
    Length Normalization    None by default              Explicit via b parameter
    Probabilistic basis     Heuristic weighting          RSJ probabilistic model
    Industry adoption       Legacy/sklearn pipelines     Elasticsearch, Lucene, Solr



**BM25 vs. TF-IDF — Detailed Key Differences**

    Dimension	                Comparison
    ────────────────────────────────────────────────────────────────────────────────────────────────────────    
    TF Treatment	            TF-IDF: linear (tf=100 is 10x better than tf=10) 
                                BM25: saturating (diminishing returns past a threshold)
                                
    Length Normalization	    TF-IDF: none by default (long docs score higher) 
                                BM25: explicit normalization via b parameter
                                
    Probabilistic Foundation	TF-IDF: heuristic weighting 
                                BM25: derived from probabilistic relevance model (RSJ model)
                                
    Industry Adoption	        TF-IDF: legacy systems, sklearn pipelines, quick prototypes 
                                BM25: Elasticsearch, Lucene, Solr, OpenSearch — industry standard

---

#### **3.3 SPLADE — Sparse Learned Embedding**

**What Is SPLADE?**

SPLADE (Sparse Lexical and Expansion model) represents a critical bridge between sparse and dense methods. 
Introduced by Formal et al. (2021), SPLADE learns sparse representations using a BERT-like transformer but 
produces sparse vectors over the vocabulary space — combining the interpretability and inverted-index 
compatibility of sparse methods with the semantic power of neural models.

The key innovation is learned query and document expansion: SPLADE learns to add vocabulary terms to a 
document's representation that are semantically related but not literally present. 
A document about 'cars' will have non-zero activations for 'automobile', 'vehicle', 'driving' even if 
those exact words never appear. 
This directly solves the vocabulary mismatch problem while maintaining sparsity.


    SPLADE Representation
    
        For document d with token representations h_1, ..., h_n from BERT:
         
        w_j = log(1 + ReLU( max_i( W_j · h_i + b_j ) ))
         
          j         = vocabulary term index (0 to |V|)
          h_i       = hidden state of token i from BERT encoder
          W_j, b_j  = learned projection weights for vocabulary term j
          ReLU      = max(0, x) — enforces non-negativity
          log(1+x)  = log saturation — controls sparsity of representation
          max over i = takes the strongest activation across all token positions
         
        The result: a sparse vector over the full vocabulary where non-zero
        dimensions represent both original terms AND semantically related terms
        learned during training on MS-MARCO or similar relevance datasets.


Why SPLADE Matters for Production RAG

    •	Inverted index compatible: Because outputs are sparse vocabulary vectors, SPLADE can be served 
        from standard Elasticsearch/Lucene infrastructure with no new infrastructure needed.
        
    •	Vocabulary expansion built-in: The 'automobile'/'car' mismatch problem is solved at indexing time, 
        not query time.
        
    •	FLOPS-efficient retrieval: Despite being transformer-based, the sparse output means retrieval uses 
        inverted index operations, not dense matrix multiplication.
        
    •	Interpretable: You can inspect which vocabulary terms have high weights and understand the 
        representation.
    
    ─────────────────────────────────────────────────────────────────────────────────────────
    
    ✓ Inverted index compatible     Served from standard Elasticsearch/Lucene, no new infra.
    ✓ Vocabulary expansion built-in 'automobile'/'car' mismatch solved at indexing time.
    ✓ FLOPS-efficient retrieval     Sparse output → inverted index ops, not dense matmul.
    ✓ Interpretable                 Inspect vocabulary term weights directly.


**Decision Matrix — Sparse Methods:**


    Use Case                        TF-IDF      BM25            SPLADE          Winner
    ──────────────────────────────────────────────────────────────────────────────────
    Low-latency keyword search      Good        ★★ Best        Good             BM25
    Legal/compliance (exact)        Good        ★★ Best        ★★ Best         BM25 or SPLADE
    No training data available      ★★ Best    ★★ Best         ✗ Needs FT      BM25
    Vocabulary mismatch             ✗ Fails     ✗ Fails         ★★ Best         SPLADE
    Elasticsearch/existing infra    Good        ★★ Built-in     Plugin avail    BM25
    Academic/scientific search      OK          Good            ★★ Best         SPLADE
    New domain, zero-shot           Good        ★★ Good         ✗ May halluc    BM25
    Hybrid retrieval component      Skip        ★★ Standard     ★★ Better      Tie


---

#### 4. Static Dense Embedding Methods

Static dense embeddings represent a paradigm shift from sparse methods. 
Instead of a high-dimensional sparse vector over a vocabulary, dense embeddings produce a 
low-dimensional vector (typically 50-300 dimensions) where every dimension is non-zero and encodes 
a distributed representation of meaning. 

The word 'static' means these models produce the same vector for a word regardless of context — 
'bank' in 'river bank' and 'bank' in 'bank account' get the same vector. 

This was the first generation of neural embedding models, and while now superseded by contextual 
models for most tasks, they remain important, fast, and practical for word-level understanding.


4.1  Word2Vec — Learning Word Geometry from Context

Word2Vec (Mikolov et al., Google, 2013) is arguably the most influential embedding paper
in NLP history. Core insight: the Distributional Hypothesis —
    "a word is characterized by the company it keeps"
Words appearing in similar contexts should have similar meanings.


**Two Architectures:**

    CBOW (Continuous Bag of Words):
        Given surrounding context words, predict the center word.
        Example: ['the', 'cat', '_', 'on', 'the'] → predict 'sat'
        Treats context as unordered bag. Faster to train. Better for frequent words.

    Skip-Gram (standard choice in practice):
        Given center word, predict each surrounding context word.
        Example: 'sat' → predict ['the', 'cat', 'on', 'the']
        More training examples per word. Better for rare words and large datasets.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  WORD2VEC TRAINING OBJECTIVE (Skip-Gram)                                 │
    │                                                                          │
    │  Maximize: Σ over all (w, c) pairs of  log P(c | w)                      │
    │                                                                          │
    │  P(c | w) = exp(v_c · v_w) / Σ_c' exp(v_c' · v_w)                        │
    │                                                                          │
    │  In practice: Negative Sampling replaces the full softmax:               │
    │  log σ(v_c · v_w) + Σ_k E[log σ(-v_k · v_w)]                             │
    │                                                                          │
    │  σ = sigmoid | k = k randomly sampled negative words (typically 5-20)    │
    │  Output: 100-300 dimensional dense vectors                               │
    └──────────────────────────────────────────────────────────────────────────┘


---

## **4.2 GloVe — Global Vectors for Word Representation**

GloVe (Pennington et al., Stanford, 2014) directly models global co-occurrence statistics.
Builds a co-occurrence matrix X where X_ij = number of times word i appears in context
of word j across the entire corpus, then factorizes this matrix to produce embeddings.

Key insight: the RATIO of co-occurrence probabilities (not raw probabilities) carries meaning.
    P(ice | solid) / P(steam | solid) is large — ice is more related to solid.
This ratio is captured by a linear relationship in embedding space.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  GLOVE OBJECTIVE                                                         │
    │                                                                          │
    │  Minimize: Σ_i Σ_j f(X_ij) × (v_i · v_j + b_i + b_j - log X_ij)²         │
    │                                                                          │
    │  f(x) = min(1, (x/x_max)^alpha)    — caps influence of frequent pairs    │
    │  alpha = 0.75 (empirically best)   |   x_max = 100                       │
    │                                                                          │
    │  GloVe vs Word2Vec:                                                      │
    │    GloVe: uses global statistics (full corpus co-occurrence matrix)      │
    │    Word2Vec: uses local statistics (sliding window predictions)          │
    │    In practice: similar performance. GloVe is simpler to train.          │
    └──────────────────────────────────────────────────────────────────────────┘


---

## **4.3 FastText — Subword-Aware Embeddings**

FastText (Bojanowski et al., Facebook, 2017) extends Word2Vec by representing each word
as a bag of character n-grams. Instead of a single vector for 'playing', it learns vectors
for: 'pla', 'lay', 'ayi', 'yin', 'ing', '<pl', 'ng>' (angle brackets = word boundaries).

The word embedding is the SUM of its n-gram embeddings.

Two crucial advantages:
    1. Handles out-of-vocabulary (OOV) words by composing n-gram representations.
    2. Captures morphological patterns: 'run', 'running', 'runner', 'ran' share n-grams.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  FASTTEXT REPRESENTATION                                                 │
    │                                                                          │
    │  Let G(w) = set of n-grams for word w  (n = 3..6 typically)              │
    │  Vector(w) = Σ_{g in G(w)} z_g                                           │
    │                                                                          │
    │  For OOV word 'unhappiness' (not in training vocab):                     │
    │    G('unhappiness') = {'unh', 'nha', 'hap', 'app', 'ppi', 'pin', ...}    │
    │    Most n-grams were seen in 'unhappy', 'happiness', etc.                │
    │    Result: a reasonable vector even for unseen words.                    │
    │                                                                          │
    │  Config: min_n=3, max_n=6, dim=100-300                                   │
    └──────────────────────────────────────────────────────────────────────────┘


**Decision Matrix — Static Dense Methods:**

    Use Case                    Word2Vec    GloVe       FastText    Recommendation
    ──────────────────────────────────────────────────────────────────────────────────
    Word analogy tasks          ★★ Best     ★★ Best     Good        Word2Vec / GloVe
    Morphologically rich lang   Poor        Poor        ★★ Best     FastText
    OOV / noisy text            ✗ Fails OOV ✗ Fails OOV ★★ Best     FastText
    Speed-critical inference    ★★ Fast     ★★ Fast     Good        Word2Vec or GloVe
    Sentence/doc tasks          Need pool.  Need pool.  Need pool.  Use SBERT instead
    Resource-constrained edge   ★★ Small    ★★ Small    Larger      Word2Vec / GloVe
    Pre-2020 legacy systems     Common      Common      Common      Any of the three


# ============================================================================================== #


### **5. CONTEXTUAL DENSE EMBEDDING METHODS**

Contextual embeddings are the dominant paradigm in modern NLP. Unlike static methods where
every word has one fixed vector, contextual models produce DIFFERENT vectors for the same
word depending on its surrounding context.

    'bank' in 'I deposited money at the bank'    → different vector than →
    'bank' in 'We sat on the river bank'

The transformer architecture makes this possible by processing the entire input sequence
simultaneously with attention mechanisms.


---

## **5.1 BERT — Bidirectional Encoder Representations from Transformers**

BERT (Devlin et al., Google, 2018) fundamentally changed NLP. Before BERT, models read
text left-to-right or concatenated L2R + R2L passes. BERT introduced bidirectional context:
every token attends to every other token simultaneously in both directions.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  TRANSFORMER SELF-ATTENTION                                              │
    │                                                                          │
    │  Input: sequence of token embeddings X = [x_1, ..., x_n]                 │
    │                                                                          │
    │  Q = X × W_Q    (queries matrix)                                         │
    │  K = X × W_K    (keys matrix)                                            │
    │  V = X × W_V    (values matrix)                                          │
    │                                                                          │
    │  Attention(Q, K, V) = softmax( Q × K^T / sqrt(d_k) ) × V                 │
    │                                                                          │
    │  d_k  = dimension of key vectors (scaling prevents gradient saturation)  │
    │  Q×K^T = attention scores: how relevant is each token to each other      │
    │  softmax = normalized attention weights summing to 1.0                   │
    │  ×V    = weighted combination of value vectors                           │
    │                                                                          │
    │  Multi-Head: MultiHead(Q,K,V) = Concat(head_1,...,head_h) × W_O          │
    │                                                                          │
    │  BERT-base:  12 layers, 12 heads, d_model=768,  110M parameters          │
    │  BERT-large: 24 layers, 16 heads, d_model=1024, 340M parameters          │
    └──────────────────────────────────────────────────────────────────────────┘


**BERT Pre-training Objectives:**

    Masked Language Model (MLM):
        15% of tokens are randomly masked. Model must predict the original token.
        Of the 15% masked: 80% → [MASK], 10% → random token, 10% → unchanged.
        Forces the model to learn bidirectional contextual representations.

    Next Sentence Prediction (NSP):
        Given two sentences, predict whether B naturally follows A.
        50% positive pairs, 50% random negative pairs.
        Note: later work showed NSP may hurt performance — RoBERTa removes it.


**Why BERT is NOT Directly Used for Retrieval:**

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  THE RETRIEVAL PROBLEM WITH RAW BERT                                     │
    │                                                                          │
    │  Cross-Encoder: encode(query + document) → score  [1 pass per pair]      │
    │  Bi-Encoder:    encode(query) → vector  +  encode(doc) → vector          │
    │                 then cosine similarity                                   │
    │                                                                          │
    │  For 1 million documents:                                                │
    │    Cross-Encoder: 1,000,000 BERT passes   [IMPOSSIBLE in real-time]      │
    │    Bi-Encoder:    1 query pass + precomputed doc vectors + ANN search    │
    │                   [Milliseconds]                                         │
    │                                                                          │
    │  Cross-encoders are reserved for RE-RANKING the top-k retrieved results  │
    │  (typically k=100) — only k forward passes, not N.                       │
    └──────────────────────────────────────────────────────────────────────────┘


---

## **5.2 Sentence-BERT (SBERT) — Bi-Encoder for Semantic Search**

SBERT (Reimers and Gurevych, 2019) fine-tunes BERT in a siamese and triplet network
architecture to produce sentence-level embeddings comparable with cosine similarity
directly, without needing both inputs simultaneously.

Key innovation: contrastive learning training objective.
    Semantically similar sentence pairs → trained to have HIGH cosine similarity.
    Semantically dissimilar pairs → trained to have LOW cosine similarity.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  SBERT TRAINING OBJECTIVES                                               │
    │                                                                          │
    │  1. Siamese Network (NLI):                                               │
    │     u = BERT(sentA) → mean_pool → L2_normalize                           │
    │     v = BERT(sentB) → mean_pool → L2_normalize                           │
    │     concat([u, v, |u-v|]) → softmax → 3-class (entail/neutral/contradict)│
    │                                                                          │
    │  2. Contrastive Loss:                                                    │
    │     L = max(0, cos(anchor,negative) - cos(anchor,positive) + margin)     │
    │                                                                          │
    │  3. Multiple Negatives Ranking Loss (modern SBERT):                      │
    │     Given batch of (query, positive_doc) pairs:                          │
    │     Treat all other positives in the batch as in-batch negatives.        │
    │     L = -log( exp(sim(q,p+)) / Σ_j exp(sim(q,d_j)) )                     │
    │                                                                          │
    │  Typical output: 384-768 dimensional dense vectors                       │
    │  Retrieval speed: ~5ms for 10M documents with FAISS index                │
    └──────────────────────────────────────────────────────────────────────────┘


**SBERT Pooling Strategies:**

    [CLS] Token Pooling:     Use the [CLS] token hidden state. Simple but not always best.

    Mean Pooling (default):  Average all token hidden states: mean(h_1, ..., h_n).
                              Empirically outperforms [CLS] for semantic similarity.
                              Most SBERT models use this. Equal influence per token.

    Max Pooling:             Element-wise maximum across all token representations.
                              Captures the 'strongest signal' per dimension.

    Weighted Mean Pooling:   Weight tokens by IDF or attention weights.
                              Downweights stopwords. Improves precision on keyword tasks.


---

## **5.3 Modern Dense Embedding Models — Current Landscape**

    ┌────────────────────────────┬──────────────┬───────┬──────────┬──────────────────────────┐
    │ Model                      │ Provider     │ Dims  │ Max Tok  │ Best For                 │
    ├────────────────────────────┼──────────────┼───────┼──────────┼──────────────────────────┤
    │ text-embedding-3-large     │ OpenAI       │ 3072  │ 8191     │ General purpose, MRL     │
    │ text-embedding-3-small     │ OpenAI       │ 1536  │ 8191     │ Cost-efficient, quality  │
    │ embed-english-v3.0         │ Cohere       │ 1024  │ 512      │ English RAG + reranking  │
    │ embed-multilingual-v3.0    │ Cohere       │ 1024  │ 512      │ Multilingual RAG (100+)  │
    │ bge-large-en-v1.5          │ BAAI / HF    │ 1024  │ 512      │ Open-source SOTA, on-prem│
    │ bge-m3                     │ BAAI / HF    │ 1024  │ 8192     │ Multi-lingual, sparse+   │
    │ e5-large-v2                │ Microsoft    │ 1024  │ 512      │ Strong RAG, text pair    │
    │ E5-Mistral-7B              │ Microsoft    │ 4096  │ 32768    │ LLM-based, top MTEB      │
    │ all-MiniLM-L6-v2           │ SBERT project│ 384   │ 256      │ Ultra-fast, tiny, local  │
    │ nomic-embed-text-v1.5      │ Nomic        │ 768   │ 8192     │ Long context, MRL        │
    │ jina-embeddings-v2         │ Jina AI      │ 768   │ 8192     │ Long document, efficient │
    └────────────────────────────┴──────────────┴───────┴──────────┴──────────────────────────┘

    ⚠️  TOKEN LIMIT CRITICAL RULE:
        Never chunk documents larger than your embedding model's max token limit.
        Truncation is SILENT and SIGNIFICANTLY degrades quality.
        bge-large / all-MiniLM → 512 max.  Store 1024-token chunks → tail invisible.


**Decision Matrix — Dense Contextual Methods:**

    Use Case / Constraint       MiniLM          BGE-Large    OpenAI-3-Lg  E5-Mistral  Recommend.
    ──────────────────────────────────────────────────────────────────────────────────────────────
    On-prem, no API             ★★ Best        ★★ Best      ✗ API only   Open wts    MiniLM or BGE
    Production cost sensitivity ★★ Free+fast   ★★ Free+good $ API cost   $$ Exp.     BGE-Large
    Maximum quality (MTEB)      OK              Good         Very good    ★★ Best     E5-Mistral/OAI
    Long docs (>512 tokens)     ✗ 256 limit     ✗ 512 limit  ✓ 8191       ★★ 32768    E5-Mistral
    Multilingual (50+ langs)    Partial         BGE-M3       Good         ★★ Best     E5 or BGE-M3
    Edge/mobile deployment      ★★ Best(22MB)  ✗ 1.3GB      ✗ Not local   ✗ 14GB+     all-MiniLM
    Rapid prototype             ★★ Best        Good         ★★ Easiest   ✗ Setup hvy  OpenAI API
    Fine-tuning domain data     ★★ Fast FT     Good         ✗ No FT opt  LoRA FT      MiniLM/BGE


# ============================================================================================== #


### **6. MULTI-VECTOR EMBEDDING METHODS**

Multi-vector embeddings represent the most sophisticated class. Rather than compressing an
entire document into a single vector (inevitably losing fine-grained information),
multi-vector methods preserve per-token or per-passage vectors and perform retrieval using
LATE INTERACTION — comparing query tokens against document tokens individually.


---

## **6.1 ColBERT — Contextualized Late Interaction over BERT**

ColBERT (Khattab and Zaharia, Stanford, 2020) introduces delayed interaction between
query and document representations.

    THE INTERACTION SPECTRUM:

    EARLY INTERACTION (Bi-encoder / SBERT):
        Encode query → single vector q
        Encode doc   → single vector d  [precomputed]
        Score = cosine(q, d)  [one similarity computation]
        Weakness: all information compressed to one vector

    FULL INTERACTION (Cross-encoder / BERT):
        Encode [query + doc] jointly → scalar score
        Full attention between all query and doc tokens
        Strength: maximum expressiveness | Weakness: cannot precompute

    LATE INTERACTION (ColBERT):
        Encode query → per-token vectors [q_1, ..., q_m]
        Encode doc   → per-token vectors [d_1, ..., d_n]  [precomputed offline]
        Score = MaxSim(Q, D) = Σ_i max_j cosine(q_i, d_j)
        Strength: token-level matching + document precomputation


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  COLBERT MAXSIM SCORING                                                  │
    │                                                                          │
    │  Q = [q_1, ..., q_m]  — m query token vectors                            │
    │  D = [d_1, ..., d_n]  — n document token vectors (precomputed)           │
    │                                                                          │
    │  Score(q, d) = Σ_{i=1}^{m}  max_{j=1..n}  q_i · d_j                      │
    │                                                                          │
    │  For each query token q_i:                                               │
    │    Find the single most similar document token: d_j* = argmax_j(q_i·d_j) │
    │    Add that maximum similarity to the total score.                       │
    │                                                                          │
    │  Linear projection (to 128 dims before MaxSim):                          │
    │    q_i = W_q × BERT_q(token_i)                                           │
    │    d_j = W_d × BERT_d(token_j)                                           │
    │                                                                          │
    │  Storage per document: 128 × n_tokens floats (n_tokens typically 200-500)│
    └──────────────────────────────────────────────────────────────────────────┘


**Storage Requirements — The ColBERT Trade-off:**

    Method                                      Storage per Document (200 tokens)
    ──────────────────────────────────────────────────────────────────────────────
    Single-vector (BGE-Large)                   1 × 1024 floats × 4 bytes = 4 KB
    ColBERT (128-dim, 200 tokens/doc)           200 × 128 floats × 4 bytes = 100 KB
    ColBERT with 2-bit quantization             200 × 128 bits × 0.25 bytes = 25 KB
    PLAID (ColBERT v2 optimized)                Centroid-based compression: ~10-20 KB


---

## **6.2 ColBERT v2 and PLAID**

ColBERT v2 (Santhanam et al., 2022) dramatically reduces storage via residual vector
quantization: instead of storing the full 128-dim float per token, it stores the
residual (difference) from the nearest centroid in a learned codebook.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  COLBERT V2 RESIDUAL COMPRESSION                                         │
    │                                                                          │
    │  Step 1: Learn K centroids C = {c_1, ..., c_K} via k-means               │
    │                                                                          │
    │  Step 2: For each token vector d_j:                                      │
    │    nearest_centroid(d_j) = argmin_k ||d_j - c_k||                        │
    │    residual(d_j) = d_j - c_nearest  (stored with low bits)               │
    │                                                                          │
    │  Step 3: Store (centroid_index, quantized_residual)                      │
    │    centroid_index: 14 bits  (K=2^14=16384)                               │
    │    residual: 2-bit quantization × 128 dims = 32 bytes                    │
    │                                                                          │
    │  Result: 10-30× storage reduction with <5% quality loss                  │
    └──────────────────────────────────────────────────────────────────────────┘

PLAID (Performance-optimized Late Interaction Driver) adds two-stage retrieval:
    Stage 1: Use centroid scores to identify candidate documents (cheap).
    Stage 2: Compute exact MaxSim only for candidates (precise).
    Reduces full MaxSim computations by 1-2 orders of magnitude.


---

## **6.3 ME-BERT — Multiple Embeddings per Document**

ME-BERT (Luan et al., Microsoft, 2021): simpler multi-vector approach.
Instead of per-token vectors, it represents each document with a small set
(k=3-5) of sentence-level vectors.

A document-level score = maximum similarity across all passage vectors.
Simpler than ColBERT and more storage-efficient, though less granular.

This maps directly to RAG design: storing multiple embeddings per document
(one per paragraph) gives more precise retrieval without full token-level indexing.


**Decision Matrix — Multi-Vector Methods:**

    Use Case                Bi-Encoder     ColBERT        ME-BERT        Cross-Encoder
    ──────────────────────────────────────────────────────────────────────────────────────
    Billion-scale corpus    ★★ Efficient   ✓ With PLAID   ★★ Good        ✗ Too slow
    Nuanced semantic retr.  Good           ★★ Best        ★★ Very good   ★★ Best (rerank)
    Long doc understanding  Poor           ★★ Best        Good           ✓ With windowing
    Storage constrained     ★★ Best        ✗ 25-100× more ✓ 3-5× more   ★★ No index needed
    First-stage retrieval   ★★ Standard    ★★ Excellent   ★★ Good        ✗ Not applicable
    Re-ranking top-k        Not ideal      Good           Good           ★★ Best
    Token-level matching    ✗ Loses signal ★★ Best        ✗ Loses signal ★★ Very good


# ============================================================================================== #


### **7. CROSS-ENCODERS AND RE-RANKING**

Cross-encoders represent the highest-accuracy text comparison approach. Unlike bi-encoders
which encode query and document independently, cross-encoders take query + document
CONCATENATED as a single input and produce a scalar relevance score.

The trade-off is severe: you CANNOT pre-compute document representations.
Every query-document pair requires a fresh forward pass.
This limits cross-encoders to re-ranking a small top-k set (typically 20-100 documents).


## **7.1 The Two-Stage Retrieval Architecture**

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  THE STANDARD PRODUCTION RETRIEVAL PIPELINE                              │
    │                                                                          │
    │  STAGE 1 — First-stage retrieval (recall-focused):                     │
    │    Method: Bi-encoder (SBERT/BGE) or BM25 or Hybrid                    │
    │    Goal:   Retrieve top-k=100 candidates from N documents              │
    │    Speed:  ~10ms with FAISS ANN index                                   │
    │    Quality: High recall, moderate precision                             │
    │                                                                          │
    │  STAGE 2 — Re-ranking (precision-focused):                             │
    │    Method: Cross-encoder on top-k=100 candidates                       │
    │    Goal:   Reorder the 100 candidates by true relevance                │
    │    Speed:  ~100-500ms (100 cross-encoder passes)                        │
    │    Quality: Near-perfect precision on the candidate set                │
    │                                                                          │
    │  FINAL OUTPUT: Top-3 to Top-10 most relevant chunks → LLM context      │
    │                                                                          │
    │  Net effect: recall of a fast retriever + precision of a cross-encoder │
    │  Both stages are complementary, not competing.                         │
    └──────────────────────────────────────────────────────────────────────────┘


## **7.2 Cross-Encoder Mathematical Formulation**

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  CROSS-ENCODER INFERENCE                                                 │
    │                                                                          │
    │  Input: [CLS] query [SEP] document [SEP]                                │
    │  h = BERT([CLS] q_1 ... q_m [SEP] d_1 ... d_n [SEP])                   │
    │  score = sigmoid( W × h_[CLS] + b )                                     │
    │                                                                          │
    │  W, b = learned classification head (fine-tuned on relevance data)     │
    │  sigmoid = maps to (0,1) for binary relevance probability               │
    │                                                                          │
    │  Inference cost: O(k) BERT forward passes for k candidates             │
    │  k=100, BERT-base: ~1-2 seconds on CPU, ~100ms on GPU                  │
    └──────────────────────────────────────────────────────────────────────────┘


## **7.3 Production Cross-Encoder Options**

    Model                               Characteristics
    ──────────────────────────────────────────────────────────────────────────────
    cross-encoder/ms-marco-MiniLM-L-6   Fast and lightweight. 12M params. CPU-feasible
                                         for k<100. Best for latency-critical systems.
    cross-encoder/ms-marco-electra-base  Higher quality than MiniLM. ELECTRA uses replaced
                                         token detection pretraining, more sample-efficient.
    Cohere Rerank-3                      State-of-the-art commercial reranker. Multilingual.
                                         ~20ms/doc API latency. Best quality via API.
    Jina Reranker v2                     Open-weights reranker. Long contexts. Text + code.
                                         Strong BEIR benchmark scores.
    FlashRank                            Ultra-lightweight (<1MB). 4ms/query. Edge deployment.
    RankLLaMA / RankVicuna               LLM-based rerankers. Highest quality but ~500ms/doc.
                                         Used for offline/batch processing only.


# ============================================================================================== #


### **8. HYBRID RETRIEVAL — COMBINING SPARSE AND DENSE**

No single embedding method dominates across all retrieval scenarios.
    Sparse → excels at exact keyword matching and interpretability.
    Dense  → excels at semantic understanding and paraphrase matching.

The practical solution: HYBRID RETRIEVAL — run both in parallel and merge results.
The empirical evidence is clear: hybrid consistently outperforms either method alone
across virtually every benchmark (BEIR, MS MARCO, MIRACL).


## **8.1 Reciprocal Rank Fusion (RRF)**

RRF (Cormack et al., 2009) is the most popular and robust fusion method.
Simple, parameter-free (almost), and empirically outperforms complex fusion methods.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  RECIPROCAL RANK FUSION FORMULA                                          │
    │                                                                          │
    │  RRF_score(d) = Σ over rankers r of:  1 / (k + rank_r(d))              │
    │                                                                          │
    │  rank_r(d) = rank position of document d in ranker r's result list      │
    │  k = 60  (empirically optimal constant)                                 │
    │                                                                          │
    │  Example:                                                               │
    │    BM25 rank: doc D is #3  → BM25 contribution = 1/(60+3) = 0.0159     │
    │    SBERT rank: doc D is #7 → Dense contribution = 1/(60+7) = 0.0149    │
    │    RRF score = 0.0159 + 0.0149 = 0.0308                                │
    │                                                                          │
    │    Doc ranked #1 by BOTH: RRF = 1/61 + 1/61 = 0.0328  (more confident) │
    │    Doc ranked #1 by ONE:  RRF = 1/61 + 0 = 0.0164     (less confident) │
    │                                                                          │
    │  Why k=60? Prevents rank-1 from being disproportionately rewarded.     │
    └──────────────────────────────────────────────────────────────────────────┘


## **8.2 Weighted Linear Combination**

    Normalize each retriever's scores to [0,1]:
        normalized_score = (score - min) / (max - min)
        OR softmax: normalized_score_i = exp(s_i) / Σ_j exp(s_j)

    Weighted combination:
        final_score(d) = alpha × sparse_score(d) + (1-alpha) × dense_score(d)

        alpha = 0.5  (equal weight, most common default)
        alpha = 0.3  (bias toward dense/semantic)
        alpha = 0.7  (bias toward sparse/keyword)

    alpha is typically tuned on a validation set.
    Many systems learn alpha per query type (keyword-heavy vs. semantic).


## **8.3 Hybrid Retrieval Decision Matrix**

    Scenario                   Sparse Only  Dense Only  Hybrid(RRF)  Hybrid+Rerank
    ──────────────────────────────────────────────────────────────────────────────────
    Exact product code search  ★★ Best      Poor        ★★ Good      ★★ Best
    Conceptual/semantic Q&A    Poor         ★★ Best     ★★ Best      ★★ Best
    Mixed keyword+semantic     OK           OK          ★★ Best      ★★ Best
    Low latency (<50ms)        ★★ Fast      ANN fast    Slightly slr ✗ Too slow
    Highest possible recall    Good         Good        ★★ Best      ★★ Best
    Simple infrastructure      ★★ Simple    ★★ Simple   More complex Most complex
    New domain, no training    ★★ Best      Zero-shot   ★★ Best      ★★ Best
    Multilingual corpus        Lang-specific ★★ Best   ★★ Best      ★★ Best


# ============================================================================================== #


### **9. MATRYOSHKA REPRESENTATION LEARNING (MRL)**

MRL (Kusupati et al., 2022) — named after Russian nesting dolls — trains a model to produce
embeddings where the FIRST d dimensions are themselves a high-quality d-dimensional embedding.

This means you can TRUNCATE the embedding to any smaller size and still get a useful
representation. Traditional embeddings are brittle: truncating a 1536-dim embedding to 256
captures arbitrary features, not the most important ones. MRL fixes this by jointly training
at multiple scales, forcing the model to pack the most important information into early dims.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  MATRYOSHKA REPRESENTATION LEARNING                                      │
    │                                                                          │
    │  Standard training:                                                     │
    │    L = loss(f(x), y)   where f(x) is the full d-dimensional embedding  │
    │                                                                          │
    │  MRL training:                                                          │
    │    L_MRL = Σ_{m in M}  c_m × loss(f(x)[1:m], y)                        │
    │                                                                          │
    │    M = set of sizes, e.g. {8, 16, 32, 64, 128, 256, 512, 1536, 3072}  │
    │    f(x)[1:m] = first m dimensions of the full embedding                │
    │    c_m = importance weight for size m  (often uniform: c_m = 1)        │
    │                                                                          │
    │  The model is simultaneously optimized at every size in M.             │
    │  Early dims MUST capture the most salient information.                  │
    │                                                                          │
    │  At inference: use f(x)[1:m] for any m ≤ d                             │
    │  No re-training, no distillation — just truncate and use.              │
    │                                                                          │
    │  OpenAI text-embedding-3-large: native 3072, usable at 256/512/1024   │
    │  nomic-embed-text-v1.5:         native 768,  usable at 64/128/256/512  │
    └──────────────────────────────────────────────────────────────────────────┘


## **9.1 The Cost-Quality Trade-off**

    Dimension Size      Quality vs. Cost
    ──────────────────────────────────────────────────────────────────────────────────────────────
    3072 (full)         Highest quality, highest MTEB. Use when quality is paramount.
    1536                ~97% of full quality. Half the storage and dot-product cost. Most production.
    512                 ~93% quality. 6× cheaper. Compatible with legacy 512-dim infrastructure.
    256                 ~88% quality. Good for first-stage retrieval + downstream reranking. 12× cheaper.
    64-128              ~75-80% quality. Suitable for classification, clustering, rough similarity.


## **9.2 Adaptive Retrieval with MRL — The Cascade Pattern**

    1. INDEX at full dimensionality (3072) for max recall,
       OR at 256 for large corpora requiring cost control.

    2. FIRST-PASS RETRIEVAL using truncated 256-dim embeddings → top-1000 candidates cheaply.

    3. RE-SCORE top-1000 with full 3072-dim embeddings for precise ranking.

    4. RERANK top-100 with a cross-encoder for maximum precision.

    5. DELIVER top-5 to LLM.

    This amortizes cost: expensive full-dim comparison only for a tiny fraction of corpus.


# ============================================================================================== #


### **10. UNIFIED EMBEDDING — BGE-M3**

BGE-M3 (Chen et al., BAAI, 2024) is the most architecturally ambitious open-source embedding
model to date. A single model covering three axes:

    Multi-Lingual       →  100+ languages
    Multi-Granularity   →  passage, document, multi-hop
    Multi-Functionality →  dense, sparse, and multi-vector retrieval from ONE model


**BGE-M3 Architecture:**

    Output Type              Description
    ───────────────────────────────────────────────────────────────────────────────────────────
    Dense Output             [CLS] token → 1024 dims. Semantic search with cosine similarity.
    Sparse Output (Learned)  Per-token logits → vocabulary space via ReLU + max-pooling (SPLADE).
    Multi-Vector (ColBERT)   Per-token embeddings → 1024 dims. MaxSim late interaction scoring.
    Context Length           8192 tokens. Enables long-document embedding without chunking.
    Languages                100+ languages. Trained on MIRACL, Mr.TYDI, synthetic multilingual.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  BGE-M3 SINGLE-PASS MULTI-SIGNAL RETRIEVAL                               │
    │                                                                          │
    │  One forward pass produces ALL THREE signal types:                      │
    │                                                                          │
    │  1. Dense vector (1024 dims)  → stored in FAISS / vector DB            │
    │  2. Sparse vector (vocab-dim) → stored in Elasticsearch / inv. index   │
    │  3. Token vectors (per-token) → stored in ColBERT-style index          │
    │                                                                          │
    │  At query time, all three fused:                                        │
    │  final_score = w1×dense_sim + w2×sparse_sim + w3×colbert_sim           │
    │                                                                          │
    │  Default weights: w1=1.0, w2=0.3, w3=1.0  (from BGE-M3 paper)        │
    │                                                                          │
    │  Result: one model replaces what previously required                   │
    │  BM25 + SBERT + ColBERT — with unified multilingual support.           │
    └──────────────────────────────────────────────────────────────────────────┘


# ============================================================================================== #


### **11. INSTRUCTION-TUNED AND TASK-SPECIFIC EMBEDDINGS**

A key limitation of standard embedding models: the same model serves all tasks equally.
Retrieval, classification, clustering, reranking, STS all have subtly different notions
of similarity. Instruction-tuned embedding models accept a TASK-SPECIFIC INSTRUCTION PREFIX.


## **11.1 E5 — Text Embeddings by Weakly Supervised Contrastive Pre-Training**

E5 (Wang et al., Microsoft, 2022) introduced the instruction prefix pattern:
every input is prefixed with 'query: ' or 'passage: ' (or more specific instructions).

    Instruction Prefix                      Effect
    ──────────────────────────────────────────────────────────────────────────────────────
    query: {question}                        Standard RAG query embedding. Optimizes for
                                             retrieval of relevant passages.
    passage: {document}                      Document/passage for indexing. Optimizes for
                                             being retrieved.
    Represent this sentence for              Clusters semantically similar sentences,
    clustering: {text}                       regardless of writing style.
    Represent this sentence for              Fine-grained similarity scoring. Used in STS.
    semantic textual similarity: {text}
    Represent this question for              More explicit; same as 'query:' but descriptive
    retrieval: {question}                    for multi-task models.


## **11.2 E5-Mistral — LLM as Embedding Model**

E5-Mistral (Wang et al., Microsoft, 2024): uses a 7B parameter Mistral LLM as backbone.
Fine-tuned with contrastive learning on synthetic hard negatives generated by GPT-4.
Achieves top-1 performance on MTEB across many task categories.

Key insight: decoder-only LLMs (Llama, Mistral, GPT) trained on far more text than
BERT-scale models. Their representations carry richer world knowledge.

    Property              Detail
    ──────────────────────────────────────────────────────────────────────────────────────
    Embedding dimension   4096 (Mistral hidden size). Often projected to 1024-2048.
    Max sequence length   Up to 32,768 tokens — far beyond BERT-class models.
    Quality               SOTA on MTEB. Strong on long docs and multi-hop reasoning.
    Cost                  7B parameters. Requires GPU (A100 or H100).
    Fine-tuning approach  Synthetic hard negatives via GPT-4 → contrastive training.


# ============================================================================================== #


### **12. FINE-TUNING EMBEDDING MODELS FOR YOUR DOMAIN**

Pre-trained models are trained on general corpora. On specialized domains (medical, legal,
financial, technical), the out-of-domain gap can significantly hurt retrieval quality.


## **12.1 When to Fine-Tune**

    Condition                   Details
    ──────────────────────────────────────────────────────────────────────────────────────────────
    High domain specificity     Medical/legal/financial jargon that general models don't understand.
                                Example: 'amortization schedule' in finance ≠ in computing.
    Low retrieval quality       You have an eval set and off-the-shelf model has poor recall@10.
    Large proprietary corpus    Millions of domain-specific docs. Domain-adaptive pre-training first.
    Limited training data       <500 labeled pairs. Consider: (1) LLM-generated synthetic pairs,
                                (2) larger model with fewer epochs, (3) LoRA.
    General Q&A use case        FAQ retrieval, Wikipedia-style general knowledge → general models OK.


## **12.2 Contrastive Fine-Tuning with Hard Negatives**

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  HARD NEGATIVE MINING AND MULTIPLE NEGATIVES RANKING LOSS               │
    │                                                                          │
    │  Training pair format: (query, positive_doc, neg_doc_1, ..., neg_doc_k) │
    │                                                                          │
    │  Hard negatives = docs superficially similar to query but NOT relevant. │
    │  Most informative type of negative — gives largest quality improvement. │
    │                                                                          │
    │  Sources of hard negatives:                                             │
    │    1. BM25 top-k hits that are NOT relevant (lexically similar, wrong)  │
    │    2. Dense retrieval top-k hits that are NOT relevant (similar but bad)│
    │    3. LLM-generated negatives: GPT-4 generates plausible-but-wrong docs │
    │                                                                          │
    │  Multiple Negatives Ranking Loss:                                       │
    │    Given batch B = {(q_i, p_i)}:                                       │
    │    Use all other positives as in-batch negatives:                      │
    │    L = -log( exp(sim(q_i,p_i)/τ) / Σ_j exp(sim(q_i,p_j)/τ) )          │
    │                                                                          │
    │    τ = temperature (0.02-0.05 typical). Lower τ = sharper distribution.│
    │    Large batch size = more in-batch negatives = better gradients.      │
    │    Recommended: batch size 256-1024, gradient checkpointing for memory. │
    └──────────────────────────────────────────────────────────────────────────┘


## **12.3 Synthetic Training Data Generation**

Labeled retrieval pairs (query, relevant document) are expensive to collect.
A powerful alternative: SYNTHETIC DATA GENERATION using LLMs.

    1. Take each document chunk in your corpus.
    2. Prompt an LLM: 'Given this passage, generate 3-5 questions whose answer
       is found in this passage.'
    3. Each (generated question, source passage) = a positive training example.
    4. Add hard negatives using BM25 retrieval of non-relevant passages.
    5. Fine-tune your embedding model on these synthetic pairs.

This approach (InPars, Promptagator, INSTRUCTOR) can generate tens of thousands
of training pairs from any corpus with NO human annotation, and consistently
improves retrieval quality by 5-15% on domain-specific benchmarks.


# ============================================================================================== #


### **13. VECTOR DATABASES AND APPROXIMATE NEAREST NEIGHBOR SEARCH**

Exact nearest neighbor search requires O(N×d) operations — infeasible at scale.
Approximate Nearest Neighbor (ANN) algorithms trade a small amount of accuracy for
orders-of-magnitude speedup by building smart data structures.


## **13.1 Core ANN Algorithms**

    HNSW — Hierarchical Navigable Small World:
    ──────────────────────────────────────────────────────────────────────────────────────────
    Builds a multi-layer graph where each node connects to its approximate neighbors.
    Upper layers sparse (fast long-range traversal) → lower layers dense (precise local search).
    Navigation from top to bottom finds approximate nearest neighbors.
    Excellent recall (>99%) with ~10ms queries on 100M vectors.
    Used by: Pinecone, Weaviate, Qdrant, Chroma.

    IVF-PQ — Inverted File with Product Quantization:
    ──────────────────────────────────────────────────────────────────────────────────────────
    k-means clustering divides space into N_list Voronoi cells (inverted file).
    Product Quantization compresses each vector by splitting into M sub-vectors,
    quantizing each independently. Query searches only nearest cells.
    Very memory-efficient. Used by FAISS. Best for: billion-scale with memory constraints.

    ScaNN — Scalable Approximate Nearest Neighbor (Google):
    ──────────────────────────────────────────────────────────────────────────────────────────
    Anisotropic vector quantization — minimizes error on high inner-product pairs specifically.
    Fastest large-scale ANN in benchmarks at high recall requirements.
    Used in Google Search.

    DiskANN — Disk-based ANN (Microsoft):
    ──────────────────────────────────────────────────────────────────────────────────────────
    Stores graph structure on DISK rather than RAM. Billion-scale on commodity hardware.
    Latency: ~5ms for 1B vectors with single SSD.


## **13.2 Vector Database Comparison**

    Database    Algorithm         Scale       Best For                    Self-Host?
    ──────────────────────────────────────────────────────────────────────────────────────────
    FAISS       IVF-PQ/HNSW/Flat  Billions    Local research, GPU-accel   Yes (library)
    Pinecone    HNSW (managed)    Billions    Managed cloud, production   No (cloud only)
    Weaviate    HNSW + BM25       100M+       Hybrid search, GraphQL API  Yes + Cloud
    Qdrant      HNSW + scalar q.  100M+       Filtering, on-prem, Rust    Yes + Cloud
    Chroma      HNSW (local)      Millions    Prototyping, local dev      Yes (local)
    Milvus      HNSW/IVF/DiskANN  Billions    Enterprise scale, cloud     Yes + Cloud
    pgvector    IVFFLAT/HNSW      Millions    Already using PostgreSQL    Yes (extension)
    Redis VSS   HNSW/FLAT         Millions    Low-latency, in-memory      Yes + Cloud


# ============================================================================================== #


### **14. EVALUATING EMBEDDING QUALITY**

Choosing and validating an embedding model requires rigorous evaluation.
MTEB (Massive Text Embedding Benchmark) is the standard comprehensive evaluation,
but production RAG should also be evaluated on domain-specific benchmarks.


## **14.1 Core Information Retrieval Metrics**

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  CORE RETRIEVAL EVALUATION METRICS                                       │
    │                                                                          │
    │  Recall@k:                                                               │
    │    Recall@10 = (#relevant docs in top 10) / (total #relevant docs)      │
    │    Optimize when: high coverage needed for downstream reranking.        │
    │                                                                          │
    │  Precision@k:                                                            │
    │    Precision@10 = (#relevant docs in top 10) / 10                       │
    │    Optimize when: LLM context window is small.                          │
    │                                                                          │
    │  MRR — Mean Reciprocal Rank:                                             │
    │    MRR = (1/|Q|) × Σ_i (1 / rank_i)                                    │
    │    Rewards models that put the FIRST relevant result higher.            │
    │                                                                          │
    │  MAP — Mean Average Precision:                                           │
    │    AP(q) = Σ_k P@k × rel(k) / #relevant docs                           │
    │    Rewards models that put ALL relevant results higher.                 │
    │                                                                          │
    │  nDCG@k — Normalized Discounted Cumulative Gain:                        │
    │    Handles graded relevance (0=irrelevant, 1=partial, 2=highly relevant)│
    │    nDCG@k = DCG@k / IDCG@k  (normalized by ideal ordering)             │
    │    Most comprehensive single-number metric. MTEB uses nDCG@10.          │
    └──────────────────────────────────────────────────────────────────────────┘


## **14.2 MTEB — Massive Text Embedding Benchmark**

MTEB (Muennighoff et al., 2022) covers 56 datasets across 8 task types:

    Task Category               Description
    ──────────────────────────────────────────────────────────────────────────────────────────────
    Retrieval (15 datasets)     nDCG@10 on BEIR: MSMARCO, HotpotQA, NQ. Most relevant for RAG.
    Semantic Textual Sim (10)   Spearman correlation between model scores and human judgments.
    Clustering (11 datasets)    V-measure on k-means clustering. ArXiv, Reddit, StackExchange.
    Classification (12)         Accuracy using embedding + logistic regression. SST2, IMDb.
    Reranking (4 datasets)      MAP on reranking retrieved lists. AskUbuntu, MindSmallReranking.
    Pair Classification (3)     Average Precision on duplicate question detection. QQP.
    Bitext Mining (1 dataset)   F1 on parallel sentence retrieval. Tatoeba multilingual.
    Summarization (1 dataset)   Spearman correlation between embedding similarity + human quality.


# ============================================================================================== #


### **15. MASTER EMBEDDING SELECTION FRAMEWORK**

## **15.1 By Primary Use Case Requirement**

    Requirement              Primary Method              Enhance With              Avoid
    ──────────────────────────────────────────────────────────────────────────────────────────────
    Exact keyword/code match BM25                        + dense for semantic      Dense-only
    Semantic / conceptual QA BGE-Large or OAI-3-Small    Cross-encoder reranking   TF-IDF only
    Long docs (>2K tokens)   E5-Mistral or jina-v2       Hierarchical chunking     512-tok models
    Multilingual / cross-lg  E5-Mistral or BGE-M3        Lang-specific BM25        English-only
    Low-latency (<20ms)      MiniLM + HNSW               BM25 for keyword          Cross-encoders
    Maximum retrieval quality OAI-3-Large or E5-Mistral  ColBERT + CE rerank      Sparse-only
    Cost-sensitive (hi vol)  Self-hosted BGE-Large        Quantized HNSW index      OAI API at vol
    Domain-specific          Fine-tuned BGE or SBERT     Synthetic training data   Zero-shot only
    Nuanced (multi-hop)      ColBERT + BGE-M3 hybrid      CE reranking, HyDE        Single-vec only
    Edge / mobile            MiniLM (22MB)               Quantized ONNX export     LLM-based models
    Rapid prototype          OAI text-embedding-3-small   Chroma or FAISS local     Complex pipelines
    Existing ES infra        BM25 (built-in) + SPLADE     ES kNN for dense vectors  Replacing stack


## **15.2 By Infrastructure and Operational Constraints**

    Constraint              Recommended Stack           Dims          Latency Profile
    ──────────────────────────────────────────────────────────────────────────────────────────────
    No GPU, CPU-only        MiniLM + FAISS Flat         384           ~5ms @10K, ~500ms @1M
    Single A100 GPU         BGE-Large + FAISS HNSW      1024          ~1ms @10M
    Managed cloud, no infra OAI-3-Small + Pinecone      1536          ~30ms end-to-end
    Multi-GPU distributed   E5-Mistral + Milvus         4096          ~5ms @100M
    Hybrid sparse + dense   BM25 (ES) + BGE + RRF       1024 + vocab  ~20ms @10M
    On-prem, air-gapped     BGE-M3 + Qdrant self-host   1024 (3 types)~5ms @10M
    Very large (1B+ docs)   Quantized BGE + DiskANN     256-512 (MRL) ~10ms @1B
    Real-time indexing       Weaviate or Qdrant + HNSW   384-768       ~50ms/doc


## **15.3 The Embedding Method Hierarchy for RAG Quality**

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║          QUALITY RANKING (Best to Fastest) for RAG Retrieval               ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    Rank 1 — ColBERT + Cross-Encoder Rerank
    Quality: ████████████████████ (highest)
    Speed:   ████████░░░░░░░░░░░░ (medium)
    Cost:    ████████████████░░░░ (expensive)
    Use when: Maximum recall matters. Medical, legal, high-stakes QA.

    Rank 2 — BGE-M3 Hybrid (Dense + Sparse + Multi-Vector)
    Quality: ████████████████████ (highest open-source)
    Speed:   ██████████████░░░░░░ (good)
    Cost:    ████████████░░░░░░░░ (moderate — 3× index storage)
    Use when: Multilingual RAG, no cross-encoder budget.

    Rank 3 — Dense Bi-Encoder + BM25 Hybrid + Cross-Encoder Rerank
    Quality: ██████████████████░░ (excellent)
    Speed:   ████████████░░░░░░░░ (moderate — 2 retrieval paths)
    Cost:    ████████████░░░░░░░░ (moderate)
    Use when: Standard production RAG. THE GOLD STANDARD CHOICE.

    Rank 4 — Dense Bi-Encoder + Cross-Encoder Rerank
    Quality: ████████████████░░░░ (very good)
    Speed:   ██████████████░░░░░░ (good)
    Cost:    ████████░░░░░░░░░░░░ (lower than hybrid)
    Use when: Semantic-heavy queries, lexical match not critical.

    Rank 5 — Dense Bi-Encoder Only (No Reranking)
    Quality: ██████████████░░░░░░ (good)
    Speed:   ██████████████████░░ (fast)
    Cost:    ████████░░░░░░░░░░░░ (low)
    Use when: Latency-critical, good enough for most semantic RAG.

    Rank 6 — BM25 Only
    Quality: █████████░░░░░░░░░░░ (keyword-dependent)
    Speed:   ████████████████████ (fastest)
    Cost:    ████░░░░░░░░░░░░░░░░ (cheapest)
    Use when: Exact keyword search, legacy systems, very low latency.


# ============================================================================================== #


### **16. ADVANCED EMBEDDING TOPICS**

## **16.1 HyDE — Hypothetical Document Embeddings**

HyDE (Gao et al., CMU, 2022) inverts the typical RAG flow. Instead of embedding the user's
question directly, HyDE first generates a hypothetical answer using an LLM, then embeds
THAT hypothetical answer and uses it as the query vector.

Intuition: a hypothetical answer is in 'document space' rather than 'question space',
closing the distribution gap between query and document embeddings.

    Approach        Flow
    ──────────────────────────────────────────────────────────────────────────────────────
    Standard RAG    query → embed(query) → search(doc_embeddings) → retrieve
    HyDE            query → LLM(generate_hypothetical_doc) → embed(hyp_doc)
                         → search(doc_embeddings) → retrieve

    When HyDE helps:  Open-domain factual QA, knowledge-intensive queries,
                      domains where questions look very different from answers.
    When HyDE hurts:  Highly ambiguous queries (wrong hypothesis misleads),
                      real-time low-latency requirements (adds LLM generation latency),
                      creative tasks with no 'correct' answer.


---

## **16.2 Contextual Retrieval (Anthropic, 2024)**

Contextual Retrieval addresses a fundamental limitation of chunk-level embeddings: when a
chunk is extracted from its document, it loses context.

    "the merger was finalized on Q3 2024"

...doesn't specify which merger — that context was in a preceding chunk.

Solution: before embedding each chunk, prepend a brief LLM-generated context summary
explaining where this chunk fits within the broader document.

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  CONTEXTUAL RETRIEVAL IMPLEMENTATION                                     │
    │                                                                          │
    │  For each chunk in document D:                                          │
    │    context = LLM(f'<document>{D}</document>                             │
    │               Please give a short succinct context to situate          │
    │               this chunk within the overall document for retrieval:    │
    │               <chunk>{chunk}</chunk>')                                   │
    │                                                                          │
    │    contextualized_chunk = context + chunk                               │
    │    embedding = embed_model(contextualized_chunk)                        │
    │    store(embedding, original_chunk)  # Store original for generation   │
    │                                                                          │
    │  Note: only the contextualized version is embedded.                    │
    │  The ORIGINAL chunk text is what gets passed to the LLM at             │
    │  inference time. This prevents context bloat in generation.            │
    │                                                                          │
    │  Reported results (Anthropic 2024):                                     │
    │    Standard RAG:                         5.7% failure rate             │
    │    + Contextual Retrieval:               3.0% (47% improvement)        │
    │    + Contextual Retrieval + BM25:        1.9% (67% improvement)        │
    │    + Above + Reranking:                  1.7% (70% improvement)        │
    └──────────────────────────────────────────────────────────────────────────┘


---

## **16.3 Binary and Quantized Embeddings**

For extreme scale or edge deployment, full-precision float32 embeddings may be too large.

    Method                  Storage & Quality
    ──────────────────────────────────────────────────────────────────────────────────────────────
    Float32 (full)          4 bytes/dim. 1024-dim = 4KB/vector. Baseline.
    Float16 (half)          2 bytes/dim. 1024-dim = 2KB/vector. ~0% quality loss. GPU native.
    Int8 (scalar quant.)    1 byte/dim.  1024-dim = 1KB/vector. ~0.5-1% quality loss. 4× smaller.
    Binary (1-bit)          1 bit/dim.   1024-dim = 128 bytes.  ~3-5% quality loss. 32× smaller.
    Product Quant. (PQ)     Sub-vector codebook. 16-64× compression. FAISS IVF-PQ. ~1-3% loss.

OpenAI recommends binary quantization + re-scoring for text-embedding-3-large:
    1. Search with binary vectors (32× faster, 32× smaller) → top-1000 candidates.
    2. Rescore with full-precision embeddings → final top-10.
    Result: 99.9% of full-precision quality at 28× the speed.


---

## **16.4 The Chunking–Embedding Interaction (Cross-Module)**

Embedding quality is DIRECTLY dependent on chunking decisions from Module 01.
These two modules are inseparable in production:

    CHUNK TOO LARGE (>model token limit):
        → Embedding model silently truncates → tail of chunk invisible to retrieval
        → Embeddings of later sections of the chunk are NEVER computed
        → Silent degradation with no error message

    CHUNK TOO SMALL (<50 tokens):
        → Embedding of "it costs $50" loses context (which product?)
        → Low-information vectors → noisy retrieval results
        → Consider small-to-large or parent-document patterns

    CHUNK WRONG SIZE FOR QUERY TYPE:
        Factoid Q&A needs   → small chunks (128-256 tokens)  → high precision
        Reasoning Q&A needs → larger chunks (512-1024 tokens) → more context

    MISMATCH BETWEEN CHUNK MODEL AND EMBED MODEL:
        If you chunk by characters but embed with a token-limited model,
        you cannot guarantee chunks stay within the model's token window.
        Always: chunk by TOKENS, not characters, when using embedding models.


    ┌──────────────────────────────────────────────────────────────────────────┐
    │  CHUNK SIZE vs. EMBEDDING MODEL TOKEN LIMIT — COMPATIBILITY MATRIX      │
    │                                                                          │
    │  Model                  Max Tokens   Safe Chunk Size (with overlap)    │
    │  ────────────────────────────────────────────────────────────────────── │
    │  all-MiniLM-L6-v2       256          ≤ 200 tokens                      │
    │  bge-large-en-v1.5      512          ≤ 400 tokens                      │
    │  Cohere embed-v3.0      512          ≤ 400 tokens                      │
    │  text-embedding-3-small 8191         ≤ 2000 tokens (safe)              │
    │  text-embedding-3-large 8191         ≤ 2000 tokens (safe)              │
    │  nomic-embed-v1.5       8192         ≤ 2000 tokens (safe)              │
    │  E5-Mistral-7B          32768        ≤ 4000 tokens (safe)              │
    │  jina-embeddings-v2     8192         ≤ 2000 tokens (safe)              │
    └──────────────────────────────────────────────────────────────────────────┘


# ============================================================================================== #


### **17. PRODUCTION EMBEDDING PIPELINE CHECKLIST**

## PHASE 1 — Model Selection

    1. Define primary use case: keyword search, semantic QA, multilingual,
       long documents, or hybrid?

    2. Evaluate 3 models on YOUR actual data: use a small labeled test set
       (50-100 query-document pairs). Never trust MTEB benchmarks alone.

    3. Check token limits: NEVER chunk documents larger than your embedding
       model's max token limit. Truncation silently degrades quality.

    4. Choose open-source vs. API: API models (OpenAI, Cohere) easiest for
       prototyping but incur per-call costs at scale.


## PHASE 2 — Infrastructure Setup

    5. Select vector database: FAISS for local/research, Qdrant/Weaviate for
       self-hosted production, Pinecone for fully managed.

    6. Choose ANN algorithm: HNSW for latency-critical (<10ms), IVF-PQ for
       memory-constrained large scale, DiskANN for billion-scale.

    7. Set index parameters:
       HNSW: ef_construction=200, M=16 (balanced) or M=32 (higher recall, 2× memory).
       IVF:  nlist = sqrt(N) cells.

    8. Implement hybrid retrieval: set up BM25 (Elasticsearch or BM25s library)
       alongside your vector index. Use RRF for fusion.


## PHASE 3 — Embedding and Indexing

    9.  Batch embedding: never embed one document at a time.
        Batch 32-256 documents per API call or GPU batch → 10-50× throughput.

    10. Handle query-document asymmetry: use 'query:' prefix for queries,
        'passage:' for documents when using E5 family models.

    11. Normalize vectors: always L2-normalize before storage if using cosine
        similarity. This lets you use dot product (faster) at retrieval time.

    12. Store metadata alongside vectors: source document, chunk position,
        date, category. Filter by metadata before ANN search for major speedup.


## PHASE 4 — Evaluation and Monitoring

    13. Build a retrieval evaluation set: minimum 100 (query, relevant chunk) pairs
        from ACTUAL user queries. Measure Recall@10 and MRR continuously.

    14. Monitor embedding drift: if corpus evolves (new products, policy updates),
        embeddings of old documents may not retrieve new ones. Re-index periodically.

    15. A/B test model updates: when upgrading embedding models, never switch cold.
        Run both indexes in parallel, split traffic, compare quality metrics.

    16. Cache query embeddings: embedding a query takes 20-100ms.
        A query embedding cache (Redis, in-memory) converts repeated queries to <1ms.


# ============================================================================================== #


### **18. GLOSSARY OF KEY TERMS**

    ANN (Approximate Nearest Neighbor):
        Algorithms that find approximately (not exactly) nearest vectors, trading small
        quality loss for orders-of-magnitude speedup over exact search.

    Bi-Encoder:
        Architecture that encodes query and document independently into vectors, enabling
        pre-computation of document embeddings. Basis of most retrieval systems.

    BM25:
        Probabilistic sparse retrieval model with TF saturation and length normalization.
        De facto standard in production search engines.

    ColBERT:
        Multi-vector embedding model using per-token representations and MaxSim late
        interaction scoring. High quality, high storage cost.

    Contrastive Learning:
        Training paradigm where the model learns to bring similar pairs (positives) closer
        and push dissimilar pairs (negatives) apart in embedding space.

    Cross-Encoder:
        Architecture that encodes query and document jointly. Highest quality but cannot
        pre-compute; used only for reranking top-k candidates.

    Dense Embedding:
        Low-dimensional (64-4096 dims) vector where all dimensions are non-zero and encode
        distributed semantic representations.

    Hard Negatives:
        Training examples superficially similar to a positive but actually irrelevant.
        The most informative type of negative for contrastive learning.

    HNSW:
        Hierarchical Navigable Small World. Graph-based ANN algorithm with excellent recall
        and latency. Most widely used in production vector databases.

    HyDE:
        Hypothetical Document Embeddings. Technique that embeds an LLM-generated
        hypothetical answer rather than the query directly.

    Late Interaction:
        ColBERT's approach — pre-compute document token vectors offline, compute
        query-document interaction at retrieval time via MaxSim.

    Matryoshka Representation Learning (MRL):
        Training technique producing embeddings where any prefix of dimensions is itself
        a high-quality smaller embedding. Enables tunable cost/quality.

    MaxSim:
        ColBERT's scoring: for each query token, find its maximum similarity to any
        document token, then sum across all query tokens.

    Mean Pooling:
        Aggregation strategy: average all token hidden states into a single fixed-size
        sentence representation. Default for most SBERT models.

    MTEB:
        Massive Text Embedding Benchmark. Standard evaluation suite covering 56 datasets
        across 8 task types for text embedding models.

    Multi-Head Attention:
        Transformer mechanism running multiple parallel attention computations, each
        attending to different aspects of the sequence.

    nDCG@k:
        Normalized Discounted Cumulative Gain at k. Primary evaluation metric for
        retrieval, rewarding relevant results ranked higher.

    RRF (Reciprocal Rank Fusion):
        score(d) = Σ(1/(k + rank_r(d))). Standard formula for combining multiple
        retrieval result lists.

    SBERT (Sentence-BERT):
        Fine-tuned BERT model producing sentence-level embeddings optimized for direct
        cosine similarity comparison.

    Sparse Embedding:
        High-dimensional vector (10K-500K dims) with mostly zeros. Non-zero dimensions
        correspond directly to vocabulary terms.

    SPLADE:
        Sparse Learned Embedding. Neural model producing sparse vocabulary-space vectors
        with learned query/document expansion.

    TF-IDF:
        Term Frequency-Inverse Document Frequency. Classic sparse embedding scoring terms
        by frequency in document vs. rarity in corpus.

    Vector Database:
        Specialized database optimized for storing and querying high-dimensional vectors
        using ANN algorithms.


# ============================================================================================== #


### **19. SUMMARY AND KEY TAKEAWAYS**

    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                  THE 10 MOST IMPORTANT PRINCIPLES FROM THIS MODULE                   ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝

    1.  No embedding method wins across all scenarios.
        Hybrid (sparse + dense) retrieval with cross-encoder reranking is the
        gold standard for production RAG.

    2.  Sparse methods (BM25) excel at exact keyword recall.
        Dense methods excel at semantic understanding.
        Neither alone is sufficient for real-world queries.

    3.  The query and document MUST use the same embedding model.
        Mixing models from different families produces meaningless similarity scores.

    4.  Never chunk documents larger than your embedding model's token limit.
        Truncation is silent and significantly degrades quality.

    5.  Cross-encoders are for re-ranking only (top-k), not first-stage retrieval.
        They cannot pre-compute document representations.

    6.  ColBERT achieves better quality than single-vector bi-encoders at the cost
        of 25-100× more storage. Use PLAID compression to make it practical.

    7.  Matryoshka embeddings (OpenAI-3, nomic-embed) let you tune the cost/quality
        trade-off post-deployment without re-training.

    8.  Domain-specific fine-tuning consistently outperforms general models on
        specialized corpora. Synthetic training data makes this accessible without
        annotation budgets.

    9.  HyDE and Contextual Retrieval are practical, high-ROI improvements to
        retrieval quality that require no changes to the underlying embedding model.

    10. Always evaluate on your own data with your own queries.
        MTEB rankings are a starting point, not a final answer.
        The right model is the one that scores best on YOUR retrieval benchmark.


    ─────────────────────────────────────────────────────────────────────────────────────────
    COMPLETE EMBEDDING METHOD TAXONOMY — AWARENESS LADDER
    ─────────────────────────────────────────────────────────────────────────────────────────

    9 ║  AGENTIC / RAPTOR TREE   →  Hierarchical summaries + LLM-structured retrieval
    8 ║  CONTEXTUAL RETRIEVAL    →  LLM prepends document-context prefix to each chunk
    7 ║  BGE-M3 UNIFIED          →  Dense + Sparse + Multi-Vector from one model pass
    6 ║  COLBERT LATE INTERACTION →  Per-token MaxSim — token-level granularity
    5 ║  CROSS-ENCODER           →  Joint query+doc encoding — highest accuracy, no precompute
    4 ║  SBERT / BGE CONTEXTUAL  →  Bi-encoder with contrastive training — production standard
    3 ║  BERT (raw)              →  Contextual representations — basis of all modern models
    2 ║  SPLADE                  →  Neural sparse with vocabulary expansion
    1 ║  BM25                    →  Probabilistic TF with saturation + length norm
    0 ║  TF-IDF                  →  Classic frequency × rarity — no training required

    ─────────────────────────────────────────────────────────────────────────────────────────
    PRODUCTION QUALITY TIERS
    ─────────────────────────────────────────────────────────────────────────────────────────

    TIER 1 — Standard quality, low cost:
        BM25 + Dense Bi-Encoder (BGE-Large or MiniLM)
        + RRF fusion + token-based chunking
        Suitable for most use cases. Correct starting point.

    TIER 2 — High quality, moderate cost:
        Dense Bi-Encoder + BM25 Hybrid + RRF + Cross-Encoder Rerank
        + Contextual Retrieval + Deduplication
        Used in serious production systems.

    TIER 3 — Maximum quality, high cost (offline indexing only):
        ColBERT or BGE-M3 Hybrid + PLAID compression
        + Contextual Retrieval + HyDE at query time + Cross-Encoder Rerank
        + RAPTOR tree for synthesis queries
        Used for high-stakes knowledge bases where precision justifies indexing cost.


---

RAG Engineering Series — Module 02: Embeddings
Covering: TF-IDF · BM25 · SPLADE · Word2Vec · GloVe · FastText · BERT · SBERT ·
BGE · OpenAI-3 · E5-Mistral · ColBERT · BGE-M3 · Cross-Encoders · Hybrid Retrieval ·
MRL · HyDE · Contextual Retrieval · ANN Search · Vector Databases · Fine-Tuning ·
Evaluation · Binary Quantization · Chunking-Embedding Interaction

"""

# ── Step-by-Step Operations ─────────────────────────────────────────────────
OPERATIONS = {

}

# ── Entry point called by topics/__init__.py ────────────────────────────────
def get_topic_data() -> dict:
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  None,
        "operations":   OPERATIONS,
    }


"""


    # EMBEDDING METHODS 
    # │
    # ├── 📦 sparse_embeddings (Lexical/Exact)
    # │   ├── 📄 tf_idf                          Classic freq × rarity weighting
    # │   ├── 📄 bm25                        ★   Probabilistic TF + length norm
    # │   └── 📄 splade                          Neural sparse, vocab expansion
    # │
    # ├── 📦 static_dense_embeddings (Legacy/Edge)
    # │   ├── 📄 word2vec                         Skip-gram / CBOW prediction
    # │   ├── 📄 glove                            Global co-occurrence matrix
    # │   └── 📄 fasttext                         Subword n-gram composition
    # │
    # ├── 📦 contextual_dense_embeddings (Bi-Encoders)
    # │   ├── 📄 bge_v1_5 / gte_large             BERT - Bidirectional transformer encoder
    # │   ├── 📄 sentence_bert                    Siamese bi-encoder, contrastive loss
    # │   ├── 📄 bge_large                    ★   Open-source SOTA, on-prem
    # │   ├── 📄 openai_text_embedding_3          Managed API, MRL-native
    # │   └── 📄 e5_mistral_7b/                   LLM backbone, top MTEB, 32k ctx
    # │                 qwen3_emebedding_8b   ★   32k-128k context
    # │
    # ├── 📦 multi_vector_embeddings
    # │   ├── 📄 colbert                      ★   Per-token vectors, MaxSim scoring
    # │   ├── 📄 colbert_v2_plaid                 Residual compression, 10–30× smaller
    # │   └── 📄 me_bert                          k sentence-level vecs per document
    # │
    # ├── 📦 cross_encoders
    # │   ├── 📄 ms_marco_minilm                  Fast, lightweight, CPU-feasible
    # │   ├── 📄 cohere_rerank_3                  Commercial SOTA, multilingual
    # │   └── 📄 jina_reranker_v2                 Open weights, long context
    # │
    # ├── 📦 hybrid_retrieval
    # │   ├── 📄 bm25_plus_biencoder          ★   RRF fusion, gold standard
    # │   ├── 📄 bge_m3_unified                   Dense + sparse + multi-vec, 1 pass
    # │   └── 📄 weighted_linear_combination      alpha-tuned score fusion
    # │
    # └── 📦 matryoshka_mrl
    #     ├── 📄 openai_text_embedding_3          3072 → 256 dims, ~5% quality loss
    #     ├── 📄 nomic_embed_v1_5                 768 dims, open weights, MRL-native
    #     └── 📄 adaptive_cascade_retrieval       Cheap truncated pass → full rerank
    
    
"""