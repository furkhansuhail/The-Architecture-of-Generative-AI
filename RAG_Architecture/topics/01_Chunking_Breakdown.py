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
TOPIC_NAME   = "Chunking Concept Breakdown"
DISPLAY_NAME = "Chunking Concept Breakdown"
ICON         = "📖"
SUBTITLE     = "Breakdown and in depth explanation for Chunking"

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



    # ----------------------------------------------------------------------------------------- #
    # -------------------------------------   Chunking:   ------------------------------------- #   
    # ----------------------------------------------------------------------------------------- #
    
##### CHUNKING STRATEGIES

Chunking is the process of breaking down a large piece of text into smaller, 
more manageable pieces (called "chunks") before turning them into math (vectors) 
and storing them in a database.

Think of it like trying to eat a steak: you can't swallow the whole thing at once, 
so you cut it into bite-sized pieces so you can digest it properly.

### **Why do we need Chunking?**

There are three main technical reasons why we can't just feed an entire 500-page book into a RAG system at once:

**1. The Context Window Limit**

    LLMs have a "memory limit" known as the context window. 
    If you try to feed it an entire library's worth of text to answer one question, 
    you will eventually run out of space, and the model will "forget" the beginning of 
    the document or fail to process it entirely.

**2. Retrieval Precision**

    If you search for a specific fact (e.g., "What is the battery life of the X-5 model?") 
    and your system retrieves a 50-page document as the "result," the AI has to sift through 
    a lot of "noise" to find that one sentence. 
    Small chunks allow the system to point exactly to the relevant paragraph.

**3. Meaningful Embeddings**

    As we discussed with tokens, text is turned into a vector (a coordinate in space).
    
    If a chunk is too long, its "meaning" becomes blurry because it covers too many topics.
    
    If a chunk is too short, it loses context (e.g., the chunk just says "it costs $50," 
    but the previous chunk had the name of the product).



### **Common Chunking Strategies**

Here is what those actually look like:

Fixed-Size Chunking: You decide on a set number of tokens (e.g., 500 tokens per chunk). 
                     It's fast but often cuts sentences in half.

Recursive Character Chunking: This is the most popular method. 
                              It tries to split at natural pauses—first at paragraphs, then sentences, 
                              then words—to keep the meaning intact while staying under a size limit.

Document-Specific Chunking: Using the structure of the file (like Markdown headers or Python 
                            function definitions) to decide where one thought ends and another begins.

**Summary of Use Cases**

    Use Case                    Why Chunking Matters
    Technical Manuals           Keeps specific instructions separate from general safety warnings.
    Legal Contracts             Ensures each "clause" is its own searchable unit.
    Customer Support            Helps the bot find the exact "FAQ" answer without reading the whole site.




    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                  TEXT CHUNKING STRATEGIES — COMPLETE TAXONOMY                                     ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
                                                                  │
          ┌──────────────┬─────────────────┬──────────────────────┼──────────────────────┬──────────────┬─────────────────┐
          │              │                 │                      │                      │              │                 │
          ▼              ▼                 ▼                      ▼                      ▼              ▼                 ▼
   ┌────────────┐  ┌─────────────┐  ┌──────────────┐   ┌───────────────────┐  ┌──────────────┐ ┌────────────┐  ┌──────────────┐
   │ FIXED-SIZE │  │ STRUCTURE-  │  │  DOCUMENT-   │   │     SEMANTIC      │  │ LLM-POWERED  │ │HIERARCHICAL│  │  CONTINUITY  │
   │  CHUNKING  │  │    AWARE    │  │   SPECIFIC   │   │     CHUNKING      │  │   CHUNKING   │ │  INDEXING  │  │  PATTERNS    │
   │            │  │             │  │              │   │                   │  │              │ │            │  │              │
   │ Blind cuts │  │ Separator-  │  │ Format-      │   │ Meaning-aware,    │  │ Intelligence │ │ Multi-level│  │ Overlap and  │
   │ by count   │  │ hierarchy   │  │ aware splits │   │ no config needed  │  │ -based       │ │ tree index │  │ late binding │
   └─────┬──────┘  └──────┬──────┘  └──────┬───────┘   └────────┬──────────┘  └──────┬───────┘ └──────┬─────┘  └──────┬───────┘
         │                │                │                     │                    │                │               │
    ┌────┼────┐      ┌─────┴──────┐   ┌────┼─────┐        ┌──────┼──────┐       ┌────┼─────┐    ┌─────┴────┐    ┌─────┴────┐
    │    │    │      │            │   │    │     │        │      │      │       │    │     │    │          │    │          │
    ▼    ▼    ▼      ▼            ▼   ▼    ▼     ▼        ▼      ▼      ▼       ▼    ▼     ▼    ▼          ▼    ▼          ▼
  ┌───┐┌───┐┌───┐ ┌───────┐  ┌───────┐ ┌────┐┌────┐┌────┐ ┌───┐┌───┐┌────┐ ┌────┐┌────┐┌────┐ ┌──────┐┌──────┐ ┌──────┐┌──────┐
  │Chr││Tok││Wrd│ │Recurs.│  │Sent.- │ │ .md││HTML││PDF │ │Fix││Pct││Std │ │Prop││Agnt││Cntx│ │RAPTOR││Small │ │Slidg.││Late  │
  │   ││ * ││   │ │Char.  │  │Based  │ │code││tags││JSON│ │Thr││ile││Dev │ │Chk.││Chk.││Rtv.│ │ Tree ││toLg  │ │Window││Chunk.│
  └───┘└───┘└───┘ └───┬───┘  └───┬───┘ │    ││    ││LaT.│ └───┘└───┘└────┘ └────┘└──┬─┘└────┘ └──────┘└──────┘ └──────┘└──────┘
  Counts:             │          │     └────┘└────┘└────┘ Splits on embedding       │
  Chr = chars         │       ┌──┴─────────────────┐      similarity drop:          │
  Wrd = words         │       │                    │      ① Fixed threshold      ┌──┴─────────────────────┐
  * Token                     ▼                    ▼      ② Percentile *         │                        │
  = gold std.      Default   Rule-based         NLP-based  ③ Std deviation       ▼                        ▼
                   text:     Split on . ? !     (trained)  ★ Percentile is   LLM-determined          LLM-generated
                   [\n\n,    Fast but fragile   ─────────  LlamaIndex default   boundaries         summaries + metadata
                    \n,      on abbrev.,        NLTK       Needs embeddings     + agentic           + Hypothetical-
                    . ,      decimals, quotes   spaCy      at index time        reasoning           Question indexing
                    " ", ""]                  stanza ★                         over full doc       (Contextual Retrieval
                   
                   + lang-aware:           ★ Best for                                              = Anthropic 2024,
                   Markdown  HTML          real-world                                              prepends LLM context
                   Python    LaTeX         multilingual                                            prefix per chunk)
                   code      JSON          text
                   
    
    TEXT CHUNKING STRATEGIES
    │
    ├── 📦 fixed_size_chunking
    │   ├── 📄 character
    │   ├── 📄 token                          ★ gold standard
    │   └── 📄 word
    │
    ├── 📦 structure_aware_chunking
    │   ├── 📄 recursive_character
    │   │   ├── 📄 default_separators         [\n\n, \n, ., " ", ""]
    │   │   ├── 📄 markdown
    │   │   ├── 📄 html
    │   │   ├── 📄 python
    │   │   └── 📄 latex
    │   └── 📄 sentence_based
    │       ├── 📄 rule_based                 splits on . ? !
    │       └── 📄 nlp_based
    │           ├── 📄 nltk
    │           ├── 📄 spacy
    │           └── 📄 stanza                 ★ best for multilingual
    │
    ├── 📦 document_specific_chunking
    │   ├── 📄 markdown_code
    │   ├── 📄 html_tags
    │   └── 📄 pdf_json_latex
    │
    ├── 📦 semantic_chunking
    │   ├── 📄 fixed_threshold
    │   ├── 📄 percentile                     ★ LlamaIndex default
    │   └── 📄 standard_deviation
    │
    ├── 📦 llm_powered_chunking
    │   ├── 📄 proposition_chunking
    │   ├── 📄 agentic_chunking
    │   └── 📄 contextual_retrieval           Anthropic 2024
    │
    ├── 📦 hierarchical_indexing
    │   ├── 📄 raptor_tree
    │   └── 📄 small_to_large
    │
    └── 📦 continuity_patterns
        ├── 📄 sliding_window
        └── 📄 late_chunking


# ============================================================================================== #


    TEXT CHUNKING STRATEGIES
    │
    ├── 📦 fixed_size_chunking
    │   ├── 📄 character
    │   ├── 📄 token                          ★ gold standard for LLMs
    │   └── 📄 word
    │
    ├── 📦 structure_aware_chunking
    │   ├── 📄 recursive_character
    │   │   ├── 📄 default_separators         [\n\n, \n, ., " ", ""]
    │   │   ├── 📄 markdown
    │   │   ├── 📄 html
    │   │   ├── 📄 python
    │   │   └── 📄 latex
    │   │
    │   ├── 📄 sentence_based
    │   │   ├── 📄 rule_based                 splits on . ? !
    │   │   └── 📄 nlp_based
    │   │       ├── 📄 nltk
    │   │       ├── 📄 spacy
    │   │       └── 📄 stanza                 ★ best for multilingual
    │   │
    │   ├── 📄 ast_based                      ★ best for code corpora
    │   │   ├── 📄 function_level             splits on def / func boundaries
    │   │   ├── 📄 class_level                splits on class declarations
    │   │   └── 📄 block_level                splits on scope blocks
    │   │
    │   └── 📄 section_header                 uses layout/font detection
    │       ├── 📄 by_title                   h1 → h2 → h3 hierarchy
    │       └── 📄 by_depth                   configurable heading depth
    │
    ├── 📦 document_specific_chunking
    │   ├── 📄 markdown_code
    │   ├── 📄 html_tags
    │   ├── 📄 pdf_json_latex
    │   └── 📄 table_figure_aware             ★ critical for multimodal RAG
    │       ├── 📄 table_isolation            tables as atomic units
    │       └── 📄 figure_caption_pairing     image + caption kept together
    │
    ├── 📦 semantic_chunking
    │   ├── 📄 embedding_similarity
    │   │   ├── 📄 fixed_threshold
    │   │   ├── 📄 percentile                 ★ LlamaIndex default
    │   │   └── 📄 standard_deviation
    │   └── 📄 topic_model_based
    │       ├── 📄 lda                        probabilistic topic assignment
    │       └── 📄 bertopic                   cluster-based neural topics
    │
    ├── 📦 llm_powered_chunking
    │   ├── 📄 proposition_chunking           atomic factual statements
    │   ├── 📄 agentic_chunking               LLM decides split points
    │   └── 📄 contextual_retrieval           Anthropic 2024 ⚠ enrichment step
    │                                         (prepends context to chunks,
    │                                          not a splitting strategy)
    │
    ├── 📦 hierarchical_indexing
    │   ├── 📄 raptor_tree                    recursive summarisation tree
    │   └── 📄 small_to_large                 retrieve small, expand to parent
    │
    ├── 📦 continuity_patterns
    │   ├── 📄 sliding_window                 overlapping fixed-size chunks
    │   └── 📄 late_chunking                  embed full doc, chunk embeddings
    │
    └── 📦 hybrid_ensemble
        ├── 📄 split_then_merge               recursive split → semantic merge
        ├── 📄 chunk_then_summarise           chunks + parent summary nodes
        └── 📄 multi_granularity              simultaneous sentence + para + doc
                                              ★ used in Voyage / Pinecone prod

    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     AWARENESS LADDER  (bottom = simplest / cheapest → top = most accurate / most expensive)
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

      8 ║  AGENTIC / RAPTOR     →  Understands full document structure + hierarchy via LLM reasoning
      7 ║  PROPOSITION          →  Extracts atomic facts; LLM resolves coreference, decontextualises
      6 ║  CONTEXTUAL RETRIEVAL →  LLM prepends document-context prefix to every chunk before embedding
      5 ║  SEMANTIC             →  Embedding similarity detects topic shifts; no size limit needed
      4 ║  SENTENCE-BASED       →  NLP tokeniser (spaCy/NLTK) finds true sentence boundaries
      3 ║  RECURSIVE / DOC-SPEC →  Separator hierarchy (paragraphs → sentences → words → chars)
      2 ║  FIXED-SIZE           →  Blind cut every N chars / tokens / words regardless of content
      1 ║  SLIDING WINDOW       →  Adds controlled overlap to any method — not a standalone strategy
      0 ║  LATE CHUNKING        →  Embeds the full doc FIRST, then pools token vecs into chunk vecs

    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     CROSS-CUTTING CONCERNS
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

      MEASUREMENT UNIT  →  Character / Word / Token (token = gold standard for all LLM pipelines)
      OVERLAP           →  Applies to any strategy; 10–20% typical; prevents boundary blind spots
      DEDUPLICATION     →  Post-processing step; removes near-identical chunks via hash / MinHash / cosine
      CHUNK SIZE        →  Factoid Q&A: 128–256 tok │ Standard RAG: 256–512 tok │ Reasoning: 512–1024 tok


## More Detailed Diagram 

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                        TEXT CHUNKING STRATEGIES                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
                                               │
           ┌───────────────────┬───────────────┼───────────────┬──────────────────┐
           │                   │               │               │                  │
           ▼                   ▼               ▼               ▼                  ▼
    ┌─────────────┐   ┌──────────────┐  ┌──────────┐   ┌─────────────┐  ┌──────────────┐
    │  FIXED-SIZE │   │  RECURSIVE   │  │SENTENCE- │   │  DOCUMENT-  │  │   SEMANTIC   │
    │  CHUNKING   │   │  CHARACTER   │  │  BASED   │   │  SPECIFIC   │  │  CHUNKING    │
    └──────┬──────┘   └──────┬───────┘  └────┬─────┘   └──────┬──────┘  └──────┬───────┘
           │                 │               │                │                │
      ┌────┼────┐            │         Uses NLP tools   ┌─────┼─────┐          │
      │    │    │    Tries separators       │           │     │     │       Embeds sentences
      ▼    ▼    ▼    in priority order:     │           ▼     ▼     ▼       compares similarity
    ┌───┐┌───┐┌───┐ ┌──────────────┐        │          ┌────┐┌────┐┌─────┐  splits at meaning
    │Chr││Tok││Wrd│ │ Paragraph \n │   ┌────┴──────┐   │ .md││HTML││ PDF │  shift
    └───┘└───┘└───┘ │ Sentence  .  │   │           │   │code││tags││LaTeX│
                    │ Space        │   ▼           ▼   └────┘└────┘└─────┘
                    │ Character    │ ┌─────┐   ┌──────┐
                    └──────┬───────┘ │NLTK │   │spaCy │      Always token-based
                           │         │sent │   │sent  │      (needs embeddings)
                    Measured in      │tok  │   │tizer │
                    chars OR tokens  └─────┘   └──────┘
    
                                      Splits at . ? !
                                      Preserves full sentences
                                      Measured in tokens


    ══════════════════════════════════════════════════════════════════════════════
     UNIT OF MEASUREMENT (cuts across all strategies)
    ══════════════════════════════════════════════════════════════════════════════

  CHARACTER-BASED    │  Counts every character including spaces
  ───────────────────┼──────────────────────────────────────────────────────
  WORD-BASED         │  Counts whitespace-separated tokens
  ───────────────────┼──────────────────────────────────────────────────────
  TOKEN-BASED        │  Counts subword units (BPE/WordPiece). Model-aware.
                     │  GOLD STANDARD for all LLM pipelines.


    ══════════════════════════════════════════════════════════════════════════════
     PRODUCTION USAGE
    ══════════════════════════════════════════════════════════════════════════════

  Strategy              Tool/System              Unit Used
  ──────────────────────────────────────────────────────────
  Fixed  (char)     →   Elasticsearch, logs    →  characters
  Fixed  (token)    →   OpenAI, Pinecone       →  tokens      ← gold std
  Recursive char    →   LangChain (default)    →  chars/tokens
  Document-specific →   Haystack, LlamaIndex   →  tokens
  Semantic          →   LlamaIndex advanced    →  tokens


    ══════════════════════════════════════════════════════════════════════════════
     KEY INSIGHT
    ══════════════════════════════════════════════════════════════════════════════

  Recursive & Document-specific = strategies for WHERE to split
  Character / Token / Word      = unit for HOW MUCH fits per chunk
  These are two separate decisions. Most production RAG systems use:
  → Recursive character splitting  +  token-based measurement

---
    
### **Fixed Size Chunking**

You decide on a set number of tokens (e.g., 500 tokens per chunk). 
It's fast but often cuts sentences in half.

In most text chunking implementations, chunk size refers to the number of characters,
and spaces count as characters too.


    Character-based chunking — counts every character including spaces, punctuation, newlines. Most common.
    
    Token-based chunking — counts tokens (used in LLM contexts). "Hello How are you?" is roughly 5 tokens,
                           not 18 characters.
                           Spaces/punctuation are handled differently by the tokenizer.
    
    Word-based chunking — counts whole words. A chunk size of 3 would give "Hello How are".


## **Character-based Chunking**


## **1.a CHARACTER BASED CHUNKING**

    "The cat sat on the mat. The dog ran fast. The bird flew high."

    chunk_size=20 chars, overlap=5

    [The cat sat on the] [n the mat. The dog] [e dog ran fast. Th] [st. The bird flew ]
    ◄─────────────────►  ◄──────────────────►
         chunk 1               chunk 2 ...

    + Simple, fast, predictable
    - Cuts mid-sentence, mid-concept — semantically blind

---
    
## **1.b Token Based Chunking**
    
    In token-based chunking, the text is first converted into tokens, then chunked by token count.
    
    The flow looks like this:
    
        Raw Text → Tokenizer → [Token IDs] → Chunked by token count → Decoded back to text
    
    Example with "Hello How are you?"
    
        1) Tokenizer converts text to tokens:
        
            "Hello"  → 15496
            " How"   → 1374
            " are"   → 389
            " you"   → 345
            "?"      → 30
        
            (token IDs vary by model/tokenizer)
        
        2) Chunker splits by token count (e.g., chunk size = 3):
        
            Chunk 1: [15496, 1374, 389]  → "Hello How are"
            Chunk 2: [345, 30]           → " you?"    ( The space before you is considered and is important ) 
        
        3) Tokens decoded back to readable text for storage/use.
    
Key things to note:

Different models use different tokenizers (GPT uses tiktoken, LLaMA uses SentencePiece, etc.),
so the same text can have different token counts across models.

A single token is not always one word — "unhappiness" might be 3 tokens: un, happi, ness.

This is why token-based chunking is more accurate for LLM context windows, since models
have token limits, not character limits.


    Tokenization in Large Language Models1. What Physically IS a Token?
    ────────────────────────────────────────────────────────────────────

    1. What Physically IS a Token?

        A token is neither a character nor a full word — it's a sub-word unit,
        sitting somewhere in between.

        The tokenizer learns a vocabulary (typically 32K–100K tokens) from a large corpus,
        breaking text into the most statistically efficient pieces.


        "unhappiness"  →  ["un", "happi", "ness"]         (3 tokens)
        "cat"          →  ["cat"]                         (1 token)
        "ChatGPT"      →  ["Chat", "G", "PT"]             (3 tokens)
        "Hello World"  →  ["Hello", " World"]             (2 tokens)

    Notice spaces are often absorbed into the next token — " World" not "World".


    2. The Full Pipeline: String → ID → Vector

    Stage 1 — Raw Text to Token IDs (Integer Lookup)

    The tokenizer has a fixed vocabulary dictionary (learned during training).
    It maps every token to a unique integer:

        "Hello"  →  15496
        "World" →  1917
        "!"      →  0

    So the sentence "Hello World!" becomes:

        [15496, 1917, 0]

    This is a simple integer array. No math yet. Just a lookup table.
    This is what gets fed into the model.

# Token-Based Chunking

When to Use:

    Scenario                                Why it fits
    RAG (Retrieval-Augmented Generation)    Chunks fed directly into LLM context
    Semantic search with embeddings         Embedding models have token limits (e.g. 512 tokens)
    LLM summarization pipelines             Respects model's context window exactly
    Question answering systems              Preserves sub-word meaning precisely
    Fine-tuning data preparation            Training data must match model's tokenizer
    Production AI applications              Gold standard for LLM-aware pipelines

When NOT to Use

    * When no LLM is involved (overkill — adds tokenizer dependency for no gain)
    * Real-time systems with extreme latency requirements (tokenization adds overhead)

Real World Example

    * LangChain, LlamaIndex — default to token-based chunking
    * OpenAI's recommended chunking strategy for embeddings
    * Pinecone, Weaviate, Chroma vector DB pipelines
    * GitHub Copilot, Claude, ChatGPT context management


    The Decision Rules:
    
        Is an LLM involved anywhere in your pipeline?
            │
           YES                          NO
            │                            │
            ▼                            ▼
      Token-based              Is your text multilingual
      (always)                 or structurally unformatted?
                                        │
                               YES              NO
                                │                │
                                ▼                ▼
                         Character-based    Word-based
                         (language-agnostic)  (simple IR/search)


Production Systems & What They Use

    System                  Chunking Method            Why
    LangChain               Token-based (tiktoken)     LLM pipeline standard
    LlamaIndex              Token-based                Respects context windows
    OpenAI Embeddings API   Token-based (cl100k)       8191 token hard limit
    Pinecone docs           Token-based                Embedding model alignment
    Elasticsearch           Word/Character             Classic IR, no LLM
    Haystack                Token-based                Production RAG framework
    Cohere RAG              Token-based                Model-aware chunking

The Golden Rule for Today's AI Tasks

    If your chunks ever touch an LLM or embedding model — use token-based chunking. Always.

The reason is precise and practical:

    GPT-4          →  128,000 token context limit
    Claude 3       →  200,000 token context limit
    text-embedding-ada-002  →  8,191 token limit

    If you chunk by words :   500 words ≈ 600–700 tokens   (unpredictable)
    If you chunk by chars :   2000 chars ≈ 400–600 tokens  (unpredictable)
    If you chunk by tokens:  500 tokens = 500 tokens       (exact, safe)

Only token-based chunking gives you exact, reliable control over what fits in a model's context window —
making it the undisputed standard for modern production AI systems.

Bonus — Advanced Chunking Used in Production Today
Beyond the basic three, production systems often layer on:

Semantic chunking — splits at meaning boundaries using sentence embeddings (LlamaIndex supports this)

Recursive character splitting — tries paragraphs → sentences → words → characters 
                                (LangChain's most popular splitter)
                                
Document-aware chunking — respects Markdown headers, HTML tags, code blocks as natural split points


## **Word-Based Chunking**

When to Use

    Scenario                                Why it fits
    Keyword search systems (BM25)           Words are the unit of search relevance
    Legal / compliance documents            Precise word count matters contractually
    Readability-focused summaries           Humans think in words, not tokens
    Low-resource environments               No tokenizer library needed
    Simple chatbots with no LLM backend     Fast, dependency-free

When NOT to Use

    * LLM pipelines — word count ≠ token count, you'll exceed context windows unpredictably
    * Multilingual text — word boundaries don't exist in Chinese/Japanese
    * Code — variable_name is one word but semantically multi-part


Real World Example

    * Classic IR systems like TF-IDF pipelines
    * Word count tools (Grammarly, Google Docs)
    * NLTK-based older NLP pipelines

---

## **Chunking Method Selection Guide**

The 3 Methods at a Glance

    Character-based  →  split every N characters
    
    Word-based       →  split every N words
    
    Token-based      →  split every N tokens (model-aware)

**Character-Based Chunking**

    When to Use

    Scenario                              Why it fits
    Simple search over raw logs     ->    Logs have no semantic structure anyway
    Fixed-width data processing     ->    CSV rows, sensor outputs, structured strings
    Language-agnostic pipelines     ->    Works on Chinese, Arabic, code — no word boundaries needed
    Storage/bandwidth constraints   ->    Predictable byte size per chunk
    Pre-processing step before NLP  ->    Just need rough splits before deeper processing

When NOT to Use

When chunks will be fed into an LLM (model doesn't think in characters)
When meaning matters — a chunk can brutally cut mid-word:

    "The patient has hypercholester"  ←  meaningless cut
    "olemia and needs treatment"      ←  context lost

Real World Example

    * Elasticsearch ingesting raw log files
    * Simple grep-style search systems
    * Data streaming pipelines where speed > semantics

---


### **SENTENCE-BASED CHUNKING**

Sentence-based chunking splits text at natural sentence boundaries rather than at a
fixed character or token count. Instead of blindly cutting every N units, it respects
the grammatical structure of language — each chunk contains one or more complete sentences.

It is smarter than fixed-size chunking but does not require embeddings like semantic
chunking does. It sits in the middle of the complexity ladder.

    Fixed-size   →  cuts every N units (blind)
    Sentence     →  cuts at sentence boundaries (grammar-aware)
    Semantic     →  cuts at meaning shifts (meaning-aware)


Split on punctuation boundaries (. ? !)

  "Einstein was born in 1879. He developed relativity. He won the Nobel Prize."

  [Einstein was born in 1879.] [He developed relativity.] [He won the Nobel Prize.]

  + Preserves grammatical units
  - Short sentences = too little context per chunk

Side Observation:

If you configure a recursive chunker with sentence-ending separators like [".", "?", "!"], 
you essentially replicate the idea of sentence-based chunking:

Recursive separators:  ["\n\n", "\n", ".", "?", "!", " ", ""]
                                        ^^^^^^^^^^^
                                   same as sentence rules



## **2.a HOW SENTENCE-BASED CHUNKING WORKS**

    "The cat sat on the mat. The dog ran fast. The bird flew high."

    Each sentence becomes a chunk (or sentences are grouped until token limit is hit):

    Chunk 1: "The cat sat on the mat."
    Chunk 2: "The dog ran fast."
    Chunk 3: "The bird flew high."

    + Preserves complete thoughts
    + No mid-sentence cuts
    - Chunks vary in size (some sentences are 5 words, some are 40)


    Compare this to character-based on the same text:

    CHARACTER-BASED (chunk_size=20):
    [The cat sat on the] [n the mat. The dog] [e dog ran fast. Th]
                          ^--- cuts mid-sentence, mid-word

    SENTENCE-BASED:
    [The cat sat on the mat.] [The dog ran fast.] [The bird flew high.]
                              ^--- clean boundary  ^--- clean boundary


---

## **2.b THE CORE PROBLEM — WHAT COUNTS AS A SENTENCE END?**

    On the surface, sentence splitting looks trivial:
    Split on  .  ?  !  and you are done.

    But natural language is full of edge cases that break this assumption:


    ABBREVIATIONS:
    ─────────────
    "Dr. Smith went to Washington D.C. He arrived late."
          ^                       ^
          These are NOT sentence ends — a naive splitter breaks here

    DECIMAL NUMBERS:
    ────────────────
    "The stock price rose to $4.99. Analysts were surprised."
                                ^
                         NOT a sentence end

    ELLIPSIS:
    ─────────
    "I was thinking... maybe we should leave."
                   ^^^
             NOT a sentence end

    QUOTES AND DIALOGUE:
    ────────────────────
    'He said "Run!" and left.'
                   ^
             Is this a sentence end? Depends on context.


    This is why sentence-based chunking requires a trained NLP tokenizer,
    not just a string split on punctuation.

    Rule-based (naive):    text.split(".")     ← breaks on Dr., $4.99, etc.
    NLP-based  (correct):  nltk / spaCy        ← handles all edge cases


---

## **2.c THE TWO APPROACHES TO SENTENCE SPLITTING**


    APPROACH 1 — RULE-BASED (naive, fast, fragile)
    ───────────────────────────────────────────────

        Split on:  [  "."  "?"  "!"  ]

        "Dr. Smith paid $4.99. He left."
           ^                 ^        ^
         wrong split   correct     correct

        Fast but produces incorrect splits on real-world text.
        Suitable only for clean, controlled text with no abbreviations.


    APPROACH 2 — NLP TOKENIZER (trained, accurate, recommended)
    ────────────────────────────────────────────────────────────

        Uses a trained model that has learned sentence boundaries from
        large corpora. Understands abbreviations, punctuation context,
        and language-specific rules.

        Libraries:

        ┌──────────┬─────────────────────────────────────────────────────┐
        │ NLTK     │ sent_tokenize() — lightweight, rule + statistical   │
        │          │ Good for clean English text                         │
        ├──────────┼─────────────────────────────────────────────────────┤
        │ spaCy    │ nlp(text).sents — deep linguistic pipeline          │
        │          │ Best accuracy, handles complex real-world text      │
        ├──────────┼─────────────────────────────────────────────────────┤
        │ stanza   │ Stanford NLP — multilingual sentence segmentation   │
        │          │ Best for non-English text                           │
        └──────────┴─────────────────────────────────────────────────────┘

        "Dr. Smith paid $4.99. He left."

        NLTK / spaCy correctly identifies:
        Sentence 1 → "Dr. Smith paid $4.99."
        Sentence 2 → "He left."


---

## **2.d SENTENCE GROUPING — HANDLING SIZE VARIATION**

    A single sentence may be too small to be a useful chunk for retrieval.
    A document about law might have sentences that are 200 tokens long.
    A children's book might have sentences that are 5 tokens.

    The solution is sentence grouping — combine N sentences per chunk
    until a token limit is reached:


    STRATEGY 1 — Fixed sentence count per chunk:

        sentences_per_chunk = 3

        Input:
        S1: "The cat sat on the mat."
        S2: "The dog ran fast."
        S3: "The bird flew high."
        S4: "The fish swam deep."
        S5: "The fox jumped quick."
        S6: "The bear slept long."

        Chunk 1: S1 + S2 + S3  →  "The cat sat on the mat.
                                    The dog ran fast.
                                    The bird flew high."

        Chunk 2: S4 + S5 + S6  →  "The fish swam deep.
                                    The fox jumped quick.
                                    The bear slept long."


    STRATEGY 2 — Token-limited sentence grouping (production standard):

        max_tokens = 100

        Keep adding sentences to current chunk until
        adding the next sentence would exceed max_tokens.
        Then start a new chunk.

        S1 (12 tok) → Chunk 1 running total: 12
        S2 (8  tok) → Chunk 1 running total: 20
        S3 (10 tok) → Chunk 1 running total: 30
        S4 (80 tok) → Adding S4 would make 110 → EXCEEDS LIMIT
                    → Close Chunk 1, start Chunk 2 with S4


    STRATEGY 3 — Sliding window with overlap:

        sentences_per_chunk = 3,  overlap = 1

        Chunk 1: S1  S2  S3
        Chunk 2:     S2  S3  S4      ← S2, S3 repeated for context
        Chunk 3:         S3  S4  S5
                         ^^^
                    overlap preserves context across chunk boundaries


---

## **2.e SENTENCE CHUNKING PIPELINE**


    Raw Text
        │
        ▼
    NLP Sentence Tokenizer (NLTK / spaCy / stanza)
        │
        ▼
    List of clean sentences
    ["S1.", "S2.", "S3.", ...]
        │
        ▼
    Grouping Strategy
    (fixed count / token limit / sliding window)
        │
        ▼
    Chunks of complete sentences
        │
        ▼
    Token counting per chunk (verify fits model context window)
        │
        ▼
    Store in Vector DB / Pass to LLM


---

## **2.f SENTENCE-BASED CHUNKING vs RECURSIVE CHUNKING**

    A common question is whether sentence-based chunking is just
    a special case of recursive chunking with sentence separators.

    The short answer: conceptually yes, practically no.


    RECURSIVE with sentence-like separators:
    ─────────────────────────────────────────
        separators = ["\n\n", "\n", ".", "?", "!"]

        This approximates sentence splitting but:

        "Dr. Smith paid $4.99. He left."
              ^              ^
         WRONG SPLIT     CORRECT

        Recursive chunking is pure pattern matching.
        It cannot distinguish a period in "Dr." from a sentence end.


    TRUE SENTENCE-BASED (NLP tokenizer):
    ──────────────────────────────────────
        spaCy / NLTK are trained models.
        They have learned from corpora that "Dr." is an abbreviation.

        "Dr. Smith paid $4.99. He left."
                              ^
                         ONLY SPLIT HERE  ← correct

    Recursive chunking ≈ sentence chunking on clean, simple text
    Recursive chunking ≠ sentence chunking on real-world text


---

## **When to Use Sentence-Based Chunking**

    Scenario                                    Why it fits
    ────────────────────────────────────────────────────────────────────────
    Q&A systems over documents                  Each answer = 1-2 sentences
    Legal / medical document retrieval          Sentences carry precise meaning
    News article RAG pipelines                  Clean sentence structure
    Summarization pipelines                     Summaries built sentence by sentence
    Chatbot knowledge bases                     Natural retrieval granularity
    Any domain where half-sentences = wrong     Preserves complete thoughts


## **When NOT to Use Sentence-Based Chunking**

    * Code — code has no sentence structure, splits will be meaningless
    * Tables and structured data — rows are not sentences
    * Very long sentences — legal contracts can have 300-token single sentences,
      making chunk sizes wildly unpredictable
    * Low-resource / real-time pipelines — NLP tokenizer adds latency overhead
      compared to simple character or recursive splitting
    * Multilingual mixed text — sentence tokenizers are language-specific,
      mixing languages in one document degrades accuracy


## **Real World Example**

    * LlamaIndex  — SentenceSplitter is a first-class chunking strategy
    * Haystack    — PreProcessor uses sentence-aware splitting by default
    * LangChain   — NLTKTextSplitter and SpacyTextSplitter both available
    * Chroma      — commonly paired with spaCy sentence splitting in tutorials


---

## **Key Properties Summary**

    Property                    Sentence-Based Chunking
    ────────────────────────────────────────────────────────────────
    Split boundary              Sentence end  . ? !
    Chunk size                  Variable (depends on sentence length)
    Measurement unit            Tokens (after grouping)
    Requires NLP library        Yes — NLTK, spaCy, or stanza
    Handles abbreviations       Yes (NLP-based), No (rule-based)
    Cuts mid-sentence           Never
    Good for LLM pipelines      Yes
    Computational cost          Medium (higher than fixed, lower than semantic)
    Context preservation        High — complete thoughts always preserved


---

## **The Key Distinction from All Other Methods**

    Character-based  →  does not care about words, sentences, or meaning
    Word-based       →  cares about words, not sentences or meaning
    Token-based      →  cares about model units, not sentences or meaning
    Sentence-based   →  cares about grammar and complete thoughts      ← unique
    Semantic         →  cares about meaning across sentences
    Recursive        →  cares about separators, approximates sentences

    Sentence-based is the only method where the split boundary
    is linguistically motivated — it respects how humans
    naturally communicate complete ideas.

---

### **RECURSIVE / HIERARCHICAL CHUNKING**

Recursive chunking splits text by trying a list of separators in order of preference,
from the most structure-preserving to the least. It does not cut blindly like fixed-size,
and it does not need a trained NLP model like sentence-based chunking.

It is the most widely used chunking strategy in production RAG pipelines today,
and is the default splitter in LangChain.

The core idea is simple:

    Try to split on the best separator first.
    If the resulting chunks are still too large,
    recurse into them and try the next separator.
    Keep going until all chunks fit within the size limit.


    Fixed-size   →  cuts every N units (blind, no rules)
    Recursive    →  tries separators in order (structure-aware)
    Sentence     →  cuts at sentence boundaries (grammar-aware)
    Semantic     →  cuts at meaning shifts (meaning-aware)

    
    Try splitting on: \n\n → \n → "." → " " → ""
    (paragraph → line → sentence → word → char)
    
    Stop when chunk fits within token limit.
    
    "Chapter 1\n\nEinstein was born...\n\nHe later moved..."
        │
        ├─► [Chapter 1\n\nEinstein was born...]   ← too big, split again
        │         │
        │         ├─► [Einstein was born...]       ← fits ✓
        │         └─► [He later moved...]          ← fits ✓
        ...
    
        + Respects natural document structure
        + Used by LangChain's RecursiveCharacterTextSplitter

---

## **3.a HOW RECURSIVE CHUNKING WORKS — STEP BY STEP**

    Default separator hierarchy (LangChain):

        Level 1  →  "\n\n"    (paragraph break)
        Level 2  →  "\n"      (line break)
        Level 3  →  ". "      (sentence end)
        Level 4  →  " "       (word boundary)
        Level 5  →  ""        (character — last resort)

    The chunker always tries the highest level separator first.
    It only falls to the next level if chunks are still too large.


    INPUT TEXT:
    ───────────
    "Machine learning is a subset of AI.
     It learns from data.

     Deep learning uses neural networks.
     It needs large datasets.

     Transformers changed NLP forever.
     Attention is the key mechanism."

    chunk_size = 60 tokens


    PASS 1 — Try "\n\n" (paragraph break):
    ────────────────────────────────────────
    Split on double newlines gives 3 chunks:

        Chunk A: "Machine learning is a subset of AI.\n It learns from data."
        Chunk B: "Deep learning uses neural networks.\n It needs large datasets."
        Chunk C: "Transformers changed NLP forever.\n Attention is the key mechanism."

        All 3 chunks are under 60 tokens → DONE. No recursion needed.


    NOW WITH SMALLER chunk_size = 10 tokens:
    ─────────────────────────────────────────
    PASS 1 — Try "\n\n":
        Chunk A: "Machine learning is a subset of AI. It learns from data."
                  ← 15 tokens. Still too large. RECURSE INTO CHUNK A.

    PASS 2 — Try "\n" on Chunk A:
        Chunk A1: "Machine learning is a subset of AI."   ← 9 tokens. OK.
        Chunk A2: "It learns from data."                  ← 5 tokens. OK.

    PASS 2 — Try "\n" on Chunk B:
        Chunk B1: "Deep learning uses neural networks."   ← 7 tokens. OK.
        Chunk B2: "It needs large datasets."              ← 6 tokens. OK.

    Final chunks: [A1, A2, B1, B2, C1, C2]
    Every chunk is within the token limit.
    Every split happened at a natural boundary.


---

## **3.b THE RECURSION VISUALIZED**


                        Full Document
                             │
                    Split on "\n\n"
                   ┌──────────────────┐
                   │                  │
              Paragraph 1        Paragraph 2
              (too large)        (fits) ✓
                   │
              Split on "\n"
           ┌───────────────┐
           │               │
        Line 1           Line 2
        (too large)      (fits) ✓
           │
        Split on ". "
      ┌────────────┐
      │            │
  Sentence 1   Sentence 2
  (fits) ✓     (fits) ✓


    The key insight:
    ─────────────────
    The chunker never goes deeper than it needs to.
    If a paragraph fits within the token limit,
    it is kept whole and never split further.
    Recursion only happens when a chunk is still too large.


---

## **3.c SEPARATOR HIERARCHY FOR DIFFERENT CONTENT TYPES**


    PLAIN TEXT (default):
    ──────────────────────
        ["\n\n", "\n", ". ", " ", ""]
         ──────  ────  ────  ───  ──
         para    line  sent  word char


    MARKDOWN:
    ──────────
        ["## ", "# ", "\n\n", "\n", ". ", " ", ""]
         ──────  ────  ──────  ────  ────  ───  ──
         H2      H1    para    line  sent  word char

        Markdown-aware recursive chunking never splits inside a section.
        It tries to keep H1/H2 sections together first.

        # Introduction           ← Try to keep this whole section together
            This is intro text.
            More intro text.

        ## Methods               ← Only split here if Introduction was too large
            Method details.


    CODE:
    ──────
        ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
         ──────────  ────────  ──────  ────  ───  ──
         class def   func def  block   line  word char

        Code-aware recursive chunking keeps functions and classes intact.
        It only splits inside a function if the function itself is too large.

        class MyModel:           ← Try to keep whole class together
            def forward(self):
                ...              ← Only split here if class is too large
            def train(self):
                ...


    HTML:
    ──────
        ["</div>", "</p>", "<br>", "\n", " ", ""]

        Splits on closing tags first, preserving HTML block structure.


---

## **3.d CHUNK OVERLAP IN RECURSIVE CHUNKING**

    Recursive chunking supports overlap just like fixed-size chunking.
    Overlap repeats the tail of one chunk at the start of the next.

    Why overlap matters:
    ─────────────────────
    Consider this text split across two chunks:

        Chunk 1: "The Eiffel Tower was built in 1889. It stands 330 metres tall."
        Chunk 2: "It is located in Paris, France."

        The word "It" in Chunk 2 refers to the Eiffel Tower from Chunk 1.
        Without overlap, Chunk 2 loses its referential context.
        A retrieval system returning only Chunk 2 gives an incomplete answer.


    WITH overlap = 20 tokens:
    ──────────────────────────
        Chunk 1: "The Eiffel Tower was built in 1889. It stands 330 metres tall."
        Chunk 2: "It stands 330 metres tall. It is located in Paris, France."
                  ◄──────── overlap ────────►

        Now Chunk 2 carries enough context to be understood independently.


    OVERLAP VISUALIZED:
    ────────────────────

        ┌──────────────────────────────────────┐
        │            Chunk 1                   │
        └──────────────────────┬───────────────┘
                               │◄── overlap ──►│
                          ┌────┴───────────────────────────┐
                          │            Chunk 2             │
                          └────────────────────────────────┘
                               │◄── overlap ──►│
                          ┌────┴───────────────────────────┐
                          │            Chunk 3             │
                          └────────────────────────────────┘

    Overlap recommendation:
    ────────────────────────
        chunk_size = 500 tokens  →  overlap = 50–100 tokens  (10–20%)
        chunk_size = 256 tokens  →  overlap = 25–50  tokens  (10–20%)

        Too little overlap  →  context lost at boundaries
        Too much overlap    →  redundant chunks, storage waste, retrieval noise


---

## **3.e RECURSIVE vs FIXED-SIZE — SIDE BY SIDE**


    TEXT:
    "Photosynthesis is how plants make food.
     They use sunlight, water, and CO2.

     The process happens in chloroplasts.
     Chlorophyll absorbs light energy."

    chunk_size = 50 tokens


    FIXED-SIZE (token-based, no awareness):
    ─────────────────────────────────────────
        Chunk 1: "Photosynthesis is how plants make food. They use sunlight,"
        Chunk 2: "water, and CO2. The process happens in chloroplasts. Chloro"
        Chunk 3: "phyll absorbs light energy."
                                        ^
                            cuts mid-word, mid-sentence
                            paragraph boundary ignored entirely


    RECURSIVE (separator-aware):
    ──────────────────────────────
        Chunk 1: "Photosynthesis is how plants make food.
                  They use sunlight, water, and CO2."

        Chunk 2: "The process happens in chloroplasts.
                  Chlorophyll absorbs light energy."

        Clean paragraph boundaries preserved.
        No mid-sentence or mid-word cuts.


---

## **3.f THE FULL RECURSIVE CHUNKING PIPELINE**


    Raw Text
        │
        ▼
    Define separator hierarchy
    ["\n\n", "\n", ". ", " ", ""]
        │
        ▼
    Split on highest-priority separator ("\n\n")
        │
        ├── Chunk fits within limit? → Keep it. Move to next chunk.
        │
        └── Chunk too large? → Recurse with next separator ("\n")
                │
                ├── Chunk fits? → Keep it.
                │
                └── Still too large? → Recurse with next separator (". ")
                        │
                        ├── Chunk fits? → Keep it.
                        │
                        └── Still too large? → Split on " " (words)
                                │
                                └── Still too large? → Split on "" (chars)
        │
        ▼
    Apply overlap between adjacent chunks
        │
        ▼
    Count tokens per chunk (verify against model limit)
        │
        ▼
    Store in Vector DB / Pass to LLM


---

## **3.g RECURSIVE CHUNKING AND SENTENCE-BASED — THE RELATIONSHIP**

    Recursive chunking can approximate sentence-based chunking
    by including sentence separators in its hierarchy.

    Recursive with sentence separators:
    ─────────────────────────────────────
        ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    This works well for clean text but fails on real-world text:

        "Dr. Smith earned $4.99. He left at 8 a.m."
               ^              ^              ^
          wrong split     correct        wrong split

    Recursive chunking is pure string pattern matching.
    It cannot distinguish "Dr." from a sentence end.
    A trained sentence tokenizer (spaCy / NLTK) handles this correctly.

    Conclusion:
    ────────────
        Recursive ≈ sentence-based  →  on clean, simple text
        Recursive ≠ sentence-based  →  on real-world text with
                                       abbreviations, decimals,
                                       and complex punctuation


---

## **When to Use Recursive Chunking**

    Scenario                                    Why it fits
    ────────────────────────────────────────────────────────────────────────
    General-purpose RAG pipelines               Works on any text type
    Mixed content (text + code + markdown)      Separator hierarchy is configurable
    When no NLP library is available            No external dependencies needed
    LangChain-based pipelines                   It is the default splitter
    Documents with clear paragraph structure    Paragraphs are the natural unit
    Production systems needing speed + quality  Faster than semantic, smarter than fixed


## **When NOT to Use Recursive Chunking**

    * Highly technical abbreviation-heavy text — periods misfire as sentence ends
    * When exact sentence boundaries are critical — use sentence-based instead
    * Purely structured data (tables, JSON, CSV) — no separator hierarchy applies
    * When meaning-level coherence is required — use semantic chunking instead
    * Languages without whitespace boundaries (Chinese, Japanese, Thai) — word
      and sentence separators do not exist in these scripts


## **Real World Example**

    * LangChain   — RecursiveCharacterTextSplitter is the default and most used
    * LlamaIndex  — SentenceSplitter uses a recursive fallback internally
    * Haystack    — RecursiveChunker available as a pipeline component
    * Most RAG tutorials online default to this method as the starting point


---

## **Key Properties Summary**

    Property                    Recursive Chunking
    ────────────────────────────────────────────────────────────────
    Split boundary              Best available separator in hierarchy
    Chunk size                  Controlled (stays within token limit)
    Measurement unit            Tokens or characters
    Requires NLP library        No
    Handles abbreviations       No (pattern matching only)
    Cuts mid-sentence           Rarely (only if sentence > chunk_size)
    Cuts mid-word               Never (space separator catches this)
    Good for LLM pipelines      Yes — production standard
    Computational cost          Low (no model needed)
    Context preservation        High — natural boundaries respected
    Configurable per content    Yes — different hierarchies for different types


---

## **The Defining Characteristic**

    Character-based  →  cuts every N characters regardless of anything
    Word-based       →  cuts every N words regardless of structure
    Token-based      →  cuts every N tokens regardless of structure
    Sentence-based   →  cuts only at sentence boundaries
    Recursive        →  cuts at the BEST available boundary  ← unique
    Semantic         →  cuts where meaning shifts

    Recursive chunking is the only method that adapts its split point
    to whatever structure is available in the text.

    If paragraphs exist  →  split on paragraphs
    If no paragraphs     →  split on lines
    If no lines          →  split on sentences
    If sentences too big →  split on words
    If words too big     →  split on characters

    It is greedy in the best possible way — it always preserves
    as much structure as the chunk size allows.

---

### **SEMANTIC CHUNKING**



Semantic chunking splits text based on meaning rather than structure or size.
Instead of asking "how many tokens fit here?" or "where is the next separator?",
it asks "where does the meaning of the text actually change?"

It is the most intelligent chunking strategy available today, and the only one
that understands the content of the text rather than just its surface form.

It works by embedding sentences into vector space, measuring the similarity
between adjacent sentences, and splitting wherever the similarity drops sharply —
indicating a topic or meaning shift.


    Fixed-size   →  cuts every N units                 (blind)
    Recursive    →  cuts at best available separator   (structure-aware)
    Sentence     →  cuts at sentence boundaries        (grammar-aware)
    Semantic     →  cuts where meaning changes         (meaning-aware)  ← unique


---

## **4.a THE CORE IDEA — EMBEDDING SIMILARITY**

    The fundamental insight behind semantic chunking is this:

        Sentences that talk about the same topic will have
        similar vector representations in embedding space.

        Sentences that shift to a new topic will have
        noticeably different vector representations.

        Split where the difference is largest.


    EXAMPLE TEXT:
    ──────────────
    S1: "The Eiffel Tower was built in 1889."
    S2: "It stands 330 metres tall in Paris."
    S3: "Gustave Eiffel designed the structure."
    S4: "Python was created by Guido van Rossum."
    S5: "It was first released in 1991."
    S6: "Python is known for its readable syntax."


    SIMILARITY SCORES between adjacent sentences:
    ───────────────────────────────────────────────
    S1 ↔ S2  :  0.91   (both about Eiffel Tower)       HIGH
    S2 ↔ S3  :  0.87   (both about Eiffel Tower)       HIGH
    S3 ↔ S4  :  0.11   (tower → Python language)        LOW  ← SPLIT HERE
    S4 ↔ S5  :  0.89   (both about Python)             HIGH
    S5 ↔ S6  :  0.84   (both about Python)             HIGH

    RESULT:
    ────────
    Chunk 1: S1 + S2 + S3   "Everything about the Eiffel Tower"
    Chunk 2: S4 + S5 + S6   "Everything about Python"

    The chunker found the topic boundary automatically —
    no separator rules, no fixed token count, no manual configuration.


---

## **4.b HOW THE ALGORITHM WORKS — STEP BY STEP**


    STEP 1 — SPLIT INTO SENTENCES
    ──────────────────────────────
    Use a sentence tokenizer (spaCy / NLTK) to get individual sentences.

        Input:  Full document text
        Output: [S1, S2, S3, S4, S5, ... Sn]


    STEP 2 — EMBED EVERY SENTENCE
    ──────────────────────────────
    Run each sentence through an embedding model.
    Each sentence becomes a high-dimensional vector.

        S1  →  [0.21, -0.83,  0.44, ...]   (384 or 768 dimensions)
        S2  →  [0.19, -0.79,  0.41, ...]
        S3  →  [0.22, -0.81,  0.46, ...]
        S4  →  [-0.61, 0.34, -0.72, ...]   ← very different direction
        S5  →  [-0.58, 0.31, -0.69, ...]
        S6  →  [-0.55, 0.29, -0.71, ...]

    Sentences on the same topic point in similar directions in vector space.
    Sentences on different topics point in very different directions.


    STEP 3 — COMPUTE COSINE SIMILARITY BETWEEN ADJACENT SENTENCES
    ──────────────────────────────────────────────────────────────
    For each consecutive pair (Si, Si+1), compute cosine similarity.

        cosine_similarity(S1, S2) = 0.91
        cosine_similarity(S2, S3) = 0.87
        cosine_similarity(S3, S4) = 0.11   ← sharp drop
        cosine_similarity(S4, S5) = 0.89
        cosine_similarity(S5, S6) = 0.84

    Cosine similarity ranges from -1 to 1:
        1.0   →  identical meaning
        0.8+  →  very similar topic
        0.5   →  loosely related
        0.1-  →  unrelated topics


    STEP 4 — DETECT BREAKPOINTS
    ────────────────────────────
    A breakpoint is placed wherever the similarity drops below a threshold,
    or wherever the drop is the sharpest relative to surrounding pairs.

        Similarity scores:  [0.91, 0.87, 0.11, 0.89, 0.84]
                                          ^^^^
                                    sharp drop → split here

    Three methods for deciding where to split:

        METHOD 1 — Fixed threshold:
            Split wherever similarity < 0.5
            Simple but sensitive to threshold choice.
            Too low  →  too few splits (chunks too large)
            Too high →  too many splits (chunks too small)

        METHOD 2 — Percentile-based:
            Compute all similarity scores.
            Split at the bottom N percentile drops.
            E.g. bottom 10% of similarity scores = split points.
            More adaptive than fixed threshold.
            Used by LlamaIndex by default.

        METHOD 3 — Standard deviation:
            Compute mean and std dev of all similarity scores.
            Split wherever score < (mean - k * std_dev)
            More statistically robust. Common in research implementations.


    STEP 5 — GROUP SENTENCES INTO CHUNKS
    ──────────────────────────────────────
    Sentences between breakpoints become one chunk.

        Breakpoint after S3:
        Chunk 1 = S1 + S2 + S3
        Chunk 2 = S4 + S5 + S6


    STEP 6 — OPTIONAL SIZE ENFORCEMENT
    ─────────────────────────────────────
    Semantic chunks can still be too large or too small.
    A post-processing step enforces token limits:

        If chunk > max_tokens  →  split further (recursive fallback)
        If chunk < min_tokens  →  merge with neighbour

    This gives semantic boundaries with practical size control.


---

## **4.c VISUALIZING THE SIMILARITY CURVE**


    Sentence index:    S1    S2    S3    S4    S5    S6    S7    S8
                                                                      1.0
    Similarity:        0.91  0.87  0.11  0.89  0.84  0.78  0.13  -    │
                                                                      │
    0.9  ──  ●────●                    ●────●────●                0.9 │
    0.8  ──               │            │                          0.8 │
    0.7  ──               │            │                          0.7 │
    0.6  ──               │            │                          0.6 │
    0.5  ──  threshold ───┼────────────┼────────────────────────  0.5 │
    0.4  ──               │            │                          0.4 │
    0.3  ──               │            │                          0.3 │
    0.2  ──               │            │                          0.2 │
    0.1  ──               ●            │              ●           0.1 │
    0.0  ──               │            │              │               │
                          ▲            │              ▲
                       SPLIT           │           SPLIT
                      (S3→S4)          │          (S7→S8)
                                       │
                                 Above threshold
                                 (no split here)

    Every sharp valley in the similarity curve = topic boundary = split point.
    Flat high regions = same topic = kept together in one chunk.


---

## **4.d THE EMBEDDING MODEL — THE ENGINE OF SEMANTIC CHUNKING**

    The quality of semantic chunking depends entirely on the
    embedding model used. Different models capture meaning differently.


    COMMON EMBEDDING MODELS FOR SEMANTIC CHUNKING:
    ────────────────────────────────────────────────

    ┌───────────────────────────────┬───────────┬──────────────────────────────┐
    │ Model                         │ Dims      │ Best for                     │
    ├───────────────────────────────┼───────────┼──────────────────────────────┤
    │ text-embedding-ada-002        │ 1536      │ General RAG (OpenAI)         │
    │ text-embedding-3-small        │ 1536      │ Fast, cheap (OpenAI)         │
    │ text-embedding-3-large        │ 3072      │ High accuracy (OpenAI)       │
    │ all-MiniLM-L6-v2              │ 384       │ Fast, local (SentenceTransf) │
    │ all-mpnet-base-v2             │ 768       │ Balanced local model         │
    │ bge-large-en-v1.5             │ 1024      │ SOTA open source (BAAI)      │
    │ cohere-embed-english-v3.0     │ 1024      │ Production Cohere pipelines  │
    └───────────────────────────────┴───────────┴──────────────────────────────┘

    The embedding model used for chunking should ideally match or be compatible
    with the embedding model used for retrieval later in the RAG pipeline.
    Mixing models introduces vector space mismatch and degrades retrieval quality.


---

## **4.e SEMANTIC CHUNKING vs ALL OTHER METHODS**


    TEXT:
    ──────
    "Newton discovered gravity in 1666.
     He published Principia Mathematica in 1687.
     His laws of motion changed physics forever.
     Einstein later revised these laws for high speeds.

     Shakespeare wrote Hamlet around 1600.
     The play explores themes of revenge and mortality.
     It is considered one of the greatest works in English literature."

    chunk_size limit = 80 tokens


    FIXED-SIZE (token-based):
    ──────────────────────────
        Chunk 1: "Newton discovered gravity in 1666. He published Principia
                  Mathematica in 1687. His laws of motion changed physics
                  forever. Einstein later revised"
                                                   ^── cuts mid-thought
        Chunk 2: "these laws for high speeds. Shakespeare wrote Hamlet
                  around 1600. The play explores themes of revenge"
                  ^── mixes physics and literature in same chunk

        Physics and literature end up mixed. Retrieval is confused.


    RECURSIVE:
    ───────────
        Chunk 1: "Newton discovered gravity in 1666.
                  He published Principia Mathematica in 1687.
                  His laws of motion changed physics forever.
                  Einstein later revised these laws for high speeds."

        Chunk 2: "Shakespeare wrote Hamlet around 1600.
                  The play explores themes of revenge and mortality.
                  It is considered one of the greatest works."

        Good — paragraph boundary preserved.
        But only works because there was a "\n\n" between topics.
        If the paragraph break was missing, recursive would merge them.


    SEMANTIC:
    ──────────
        Detects topic shift between "Einstein" and "Shakespeare"
        even if there is NO paragraph break in the text.

        Chunk 1: "Newton discovered gravity in 1666. He published Principia
                  Mathematica in 1687. His laws of motion changed physics
                  forever. Einstein later revised these laws for high speeds."

        Chunk 2: "Shakespeare wrote Hamlet around 1600. The play explores
                  themes of revenge and mortality. It is considered one of
                  the greatest works in English literature."

        Correctly separates physics from literature
        purely based on meaning — no structural cues needed.


    This is the key advantage of semantic chunking:
    ─────────────────────────────────────────────────
    It finds topic boundaries that have NO structural signal —
    no paragraph break, no heading, no separator of any kind.
    The meaning itself is the signal.


---

## **4.f SEMANTIC CHUNKING PIPELINE**


    Raw Text
        │
        ▼
    Sentence Tokenizer (spaCy / NLTK)
        │
        ▼
    [S1, S2, S3, S4 ... Sn]
        │
        ▼
    Embedding Model  (all-MiniLM / ada-002 / bge)
        │
        ▼
    Sentence vectors  [ v1, v2, v3, v4 ... vn ]
        │
        ▼
    Cosine Similarity between consecutive pairs
    [ sim(v1,v2), sim(v2,v3), sim(v3,v4) ... ]
        │
        ▼
    Breakpoint Detection
    (threshold / percentile / std dev method)
        │
        ▼
    Group sentences between breakpoints into chunks
        │
        ▼
    Size enforcement (merge small / split large chunks)
        │
        ▼
    Final semantically coherent chunks
        │
        ▼
    Store in Vector DB / Pass to LLM


---

## **4.g WINDOWED SIMILARITY — HANDLING SHORT SENTENCES**

    A weakness of pure adjacent-sentence similarity is that
    very short sentences can cause false splits:

        S1: "Newton discovered gravity."         0.88 ↔
        S2: "It changed physics forever."        0.85 ↔
        S3: "Yes."                               0.21 ↔  ← false drop
        S4: "This was a turning point."          0.86 ↔
        S5: "Einstein built on these ideas."

    "Yes." is a short, low-information sentence.
    Its embedding is generic and causes a false similarity drop.
    A naive splitter would incorrectly place a chunk boundary here.

    SOLUTION — Windowed similarity:
    ─────────────────────────────────
    Instead of comparing single sentences, compare WINDOWS of sentences.
    Average the embeddings of sentences [i-k ... i+k] around each point.

        Window size = 2:
        Compare average(S1,S2,S3) with average(S3,S4,S5)
        The "Yes." is smoothed out by its neighbors.
        False split is avoided.

    LlamaIndex uses this windowed approach by default (window_size=1 or 2).


---

## **4.h WHEN SEMANTIC CHUNKING SHINES — REAL EXAMPLES**


    EXAMPLE 1 — Wikipedia-style article (no headings):
    ────────────────────────────────────────────────────
    An article about "London" might discuss:
        - Geography and location
        - History from Roman times
        - Modern economy and finance
        - Culture and museums
        - Transport infrastructure

    If the article has no headings, recursive chunking merges topics.
    Semantic chunking correctly identifies each topic shift and creates
    one chunk per thematic section — purely from meaning.


    EXAMPLE 2 — Legal contract with dense paragraphs:
    ──────────────────────────────────────────────────
    A contract paragraph might cover:
        - Payment terms (sentences 1-4)
        - Liability clauses (sentences 5-9)
        - Termination conditions (sentences 10-13)

    All within one dense block of text with no separator.
    Recursive chunking keeps it as one giant chunk or cuts mid-clause.
    Semantic chunking splits at payment→liability and liability→termination.


    EXAMPLE 3 — Research paper discussion section:
    ────────────────────────────────────────────────
    Discussion sections flow from result to result with no headers.
    Semantic chunking follows the intellectual flow of the argument,
    placing boundaries where the researcher shifts from one finding
    to the next — exactly what a human reader would do.


---

## **4.i COST — THE MAIN TRADEOFF**

    Semantic chunking is the most expensive chunking strategy.
    Every sentence must be embedded before any split decision is made.

    COST COMPARISON for a 10,000 token document:
    ──────────────────────────────────────────────

        Method            API calls    Compute       Latency
        ──────────────────────────────────────────────────────
        Fixed-size        0            None          ~0ms
        Recursive         0            None          ~5ms
        Sentence-based    0            NLTK/spaCy    ~20ms
        Semantic          1 per sent   Embedding     ~500ms-2s
                          (~80-120
                          sentences)

    For a 1 million token corpus:
        Fixed / Recursive  →  seconds
        Semantic           →  minutes to hours + embedding API cost

    This is why semantic chunking is used at INDEXING TIME (offline),
    never at query time (real-time). You pay the cost once when
    building the vector store. Retrieval itself is fast regardless.


---

## **When to Use Semantic Chunking**

    Scenario                                     Why it fits
    ─────────────────────────────────────────────────────────────────────────
    Long documents with no structural markup     No headings/separators to rely on
    Multi-topic documents in single paragraphs   Meaning is the only boundary signal
    High-precision RAG where quality > speed     Best retrieval relevance
    Legal, medical, scientific documents         Topic coherence is critical
    Knowledge bases requiring deep understanding Each chunk = one coherent concept
    Offline indexing pipelines (batch)           Cost is acceptable at index time


## **When NOT to Use Semantic Chunking**

    * Real-time chunking pipelines — embedding latency is too high
    * Very short documents — not enough sentences to detect meaningful shifts
    * Highly structured data (tables, JSON, code) — embeddings do not capture
      structural meaning well, recursive or document-specific works better
    * Budget-constrained projects — embedding API costs add up at large scale
    * When documents already have clear structure (headings, sections) —
      document-specific chunking gives similar quality at a fraction of the cost


## **Real World Example**

    * LlamaIndex   — SemanticSplitterNodeParser (production-ready implementation)
    * LangChain    — SemanticChunker (uses embedding model + breakpoint threshold)
    * Chroma       — recommended for high-quality RAG pipelines in documentation
    * Greg Kamradt — original open source implementation that popularized the method


---

## **Key Properties Summary**

    Property                    Semantic Chunking
    ─────────────────────────────────────────────────────────────────
    Split boundary              Meaning / topic shift
    Chunk size                  Variable (content-driven)
    Measurement unit            Tokens (after grouping)
    Requires NLP library        Yes — sentence tokenizer + embedding model
    Requires embedding model    Yes — this is the core requirement
    Cuts mid-sentence           Never
    Cuts mid-topic              Never (this is the entire point)
    Good for LLM pipelines      Yes — highest quality retrieval
    Computational cost          High (embedding every sentence)
    Context preservation        Highest of all methods
    Configurable                Yes — threshold, window size, min/max tokens


---

## **The Defining Characteristic**

    Character-based  →  has no awareness of words, sentences, or meaning
    Word-based       →  aware of words, not sentences or meaning
    Token-based      →  aware of model units, not sentences or meaning
    Sentence-based   →  aware of grammar and sentence boundaries
    Recursive        →  aware of structural separators in the text
    Semantic         →  aware of MEANING                              ← unique

    Every other chunking method operates on the FORM of the text.
    Semantic chunking is the only method that operates on the CONTENT.

    It does not ask "where is the next period?"
    It does not ask "how many tokens have I counted?"
    It asks "has the author started talking about something different?"

    That question — and the ability to answer it — is what makes
    semantic chunking fundamentally different from everything else.

---

## **Agentic / Proposition Chunking**

Agentic chunking and proposition chunking are the most advanced chunking
strategies available today. They do not split text by size, structure,
or even meaning shifts — they use an LLM itself to decide how to chunk,
and what each chunk should contain.

Instead of mechanically dividing text, they instruct an LLM to read the
text and extract self-contained, atomic units of information — the way a
knowledgeable human editor would summarize and organize content.


    Fixed-size    →  cuts every N units                  (blind)
    Recursive     →  cuts at best separator              (structure-aware)
    Sentence      →  cuts at sentence boundaries         (grammar-aware)
    Semantic      →  cuts where meaning shifts           (meaning-aware)
    Proposition   →  extracts atomic facts               (knowledge-aware) ← unique
    Agentic       →  LLM decides everything              (reasoning-aware) ← unique


---

## **5.a THE CORE IDEA — WHY EXISTING METHODS STILL FAIL**

    Every chunking method so far has one shared weakness:

        The chunk boundary is determined by the TEXT SURFACE —
        characters, tokens, separators, sentence endings, or
        embedding similarity.

        None of them understand what the text is SAYING.


    EXAMPLE — A single sentence that contains multiple facts:
    ──────────────────────────────────────────────────────────
    "Albert Einstein, who was born in Ulm, Germany in 1879,
     developed the theory of relativity, won the Nobel Prize
     in Physics in 1921, and later emigrated to the United
     States in 1933."

    This is ONE sentence. All chunking methods keep it as one chunk.

    But it contains FIVE distinct retrievable facts:

        Fact 1: Einstein was born in Ulm, Germany.
        Fact 2: Einstein was born in 1879.
        Fact 3: Einstein developed the theory of relativity.
        Fact 4: Einstein won the Nobel Prize in Physics in 1921.
        Fact 5: Einstein emigrated to the United States in 1933.

    If a user asks "When did Einstein emigrate to the US?", a retrieval
    system needs Fact 5. But Fact 5 is buried in a 50-token sentence
    alongside 4 other facts. The chunk retrieved is noisy.

    Proposition chunking solves this by breaking the sentence into
    five independent, atomic, retrievable propositions.


---

## **5.b PROPOSITION CHUNKING — THE DENSE PASSAGE RETRIEVAL ORIGIN**

    Proposition chunking was formalized in the research paper:
    "Dense X Retrieval: What Retrieval Granularity Should We Use?"
    (Chen et al., 2023)

    The paper showed that retrieving atomic propositions rather than
    sentences or paragraphs consistently improved RAG accuracy across
    multiple benchmarks.

    The core definition of a proposition:

        A proposition is a single, atomic, self-contained statement
        of fact that:

        1. Expresses exactly ONE piece of information
        2. Is fully self-contained — no pronouns or references
           that require the surrounding text to resolve
        3. Is decontextualized — can be understood in isolation
        4. Is written as a minimal, clean declarative sentence


    EXAMPLE — Original text:
    ─────────────────────────
    "Marie Curie was a Polish-French physicist and chemist.
     She conducted pioneering research on radioactivity.
     She was the first woman to win a Nobel Prize, and the
     only person to win Nobel Prizes in two different sciences.
     Her husband Pierre Curie was also a physicist and frequent
     collaborator. She died in 1934 from aplastic anaemia,
     likely caused by radiation exposure."

    PROPOSITIONS extracted:
    ────────────────────────
    P1: "Marie Curie was a physicist and chemist."
    P2: "Marie Curie was of Polish-French nationality."
    P3: "Marie Curie conducted pioneering research on radioactivity."
    P4: "Marie Curie was the first woman to win a Nobel Prize."
    P5: "Marie Curie won Nobel Prizes in two different sciences."
    P6: "Marie Curie is the only person to win Nobel Prizes in two sciences."
    P7: "Pierre Curie was a physicist."
    P8: "Pierre Curie was Marie Curie's husband."
    P9: "Pierre Curie and Marie Curie frequently collaborated."
    P10: "Marie Curie died in 1934."
    P11: "Marie Curie died from aplastic anaemia."
    P12: "Marie Curie's death was likely caused by radiation exposure."

    Each proposition is:
        ✓ One fact only
        ✓ Self-contained (no "she" without "Marie Curie")
        ✓ Understandable in isolation
        ✓ Minimal and clean


---

## **5.c HOW PROPOSITION CHUNKING WORKS — STEP BY STEP**


    STEP 1 — CHUNK THE DOCUMENT INTO PASSAGES FIRST
    ──────────────────────────────────────────────────
    The full document is too large to send to an LLM at once.
    A preliminary chunking pass (recursive or sentence-based)
    creates manageable passage chunks of 200-500 tokens each.

        Full Document  →  [Passage 1, Passage 2, ... Passage N]

    These passages are NOT the final chunks. They are working units
    for the LLM to process.


    STEP 2 — SEND EACH PASSAGE TO AN LLM WITH A PROMPT
    ─────────────────────────────────────────────────────
    Each passage is sent to an LLM with a structured prompt:

        PROMPT:
        ────────
        "Decompose the following passage into the smallest possible
         set of self-contained, atomic propositions.

         Rules:
         - Each proposition must contain exactly one fact.
         - Each proposition must be fully self-contained.
           Replace all pronouns with the full noun they refer to.
         - Each proposition must be understandable without any
           surrounding context.
         - Write each proposition as a simple declarative sentence.
         - Do not add any information not present in the passage.

         Passage:
         {passage_text}

         Return the propositions as a numbered list."


    STEP 3 — LLM RETURNS ATOMIC PROPOSITIONS
    ──────────────────────────────────────────
    The LLM reads the passage, understands coreference
    (what "she", "it", "they" refers to), and returns
    clean, atomic, decontextualized statements.


    STEP 4 — STORE PROPOSITIONS AS INDIVIDUAL CHUNKS
    ──────────────────────────────────────────────────
    Each proposition becomes one chunk in the vector store.
    Each proposition is embedded individually.
    Retrieval fetches the exact proposition that answers the query.


    STEP 5 — OPTIONAL — METADATA LINKING
    ──────────────────────────────────────
    Each proposition is linked back to its source passage
    and source document via metadata.

    When a proposition is retrieved, the full source passage
    can be fetched for additional context if needed.

        Proposition: "Marie Curie died in 1934."
        Metadata: {
            source_doc:     "marie_curie_biography.pdf",
            source_passage: 4,
            passage_text:   "She died in 1934 from aplastic anaemia..."
        }


---

## **5.d THE FULL PROPOSITION CHUNKING PIPELINE**


    Raw Document
        │
        ▼
    Preliminary chunker (recursive / sentence-based)
        │
        ▼
    [Passage 1] [Passage 2] [Passage 3] ... [Passage N]
        │
        ▼  (for each passage)
    LLM  ←── Proposition extraction prompt
        │
        ▼
    [P1, P2, P3, P4, P5 ...]   (atomic propositions)
        │
        ▼
    Embedding model (embed each proposition)
        │
        ▼
    Proposition vectors stored in Vector DB
    (with metadata linking back to source passage)
        │
        ▼
    At query time:
    Query  →  embed query  →  retrieve top-K propositions
        │
        ▼
    Optionally fetch full source passages for context
        │
        ▼
    LLM generates answer from retrieved propositions


---

## **5.e AGENTIC CHUNKING — GOING FURTHER**

    Proposition chunking gives an LLM a fixed task:
    "decompose this passage into atomic facts."

    Agentic chunking gives an LLM full autonomy over
    the entire chunking strategy and structure.

    The LLM does not just extract facts —
    it decides:

        - What the natural topics in the document are
        - Where the boundaries between topics fall
        - What metadata to attach to each chunk
        - Whether chunks should be hierarchical
        - Whether some content is redundant
        - What the best summary of each chunk is
        - How chunks relate to each other


    AGENTIC CHUNKING APPROACHES:
    ──────────────────────────────

    APPROACH 1 — LLM-determined boundaries:

        Send the full document (or large sections) to an LLM.
        Ask it to identify natural section boundaries.
        Use those boundaries as chunk split points.

        PROMPT:
        ────────
        "Read the following document and identify the natural
         thematic sections. For each section, return:
         - The start sentence of the section
         - The end sentence of the section
         - A one-line description of what the section covers

         Document: {document_text}"

        The LLM returns a structured set of boundaries.
        The chunker uses those boundaries to create chunks.


    APPROACH 2 — LLM-generated summaries + chunks:

        For each chunk the LLM also generates:
        - A title for the chunk
        - A summary of the chunk
        - Keywords / entities in the chunk
        - A list of questions this chunk can answer

        These are stored as metadata alongside the chunk text.
        Retrieval can use the summary or questions for matching
        instead of (or in addition to) the raw chunk text.


    APPROACH 3 — Hypothetical questions indexing:

        For each chunk, the LLM generates 3-5 hypothetical
        questions that the chunk could answer.

        Chunk text:
        "The Treaty of Versailles was signed on June 28, 1919.
         It officially ended World War I between Germany and
         the Allied Powers. Germany was forced to accept full
         responsibility for the war under Article 231."

        Generated questions:
        Q1: "When was the Treaty of Versailles signed?"
        Q2: "What did the Treaty of Versailles end?"
        Q3: "What was Article 231 of the Treaty of Versailles?"
        Q4: "What war did the Treaty of Versailles conclude?"
        Q5: "What responsibilities did Germany accept in 1919?"

        Both the questions AND the chunk text are embedded and stored.
        At retrieval time, a user query matches against questions —
        often a better semantic match than raw text.

        This is called HyDE-style indexing (Hypothetical Document Embeddings)
        and significantly improves retrieval precision.


    APPROACH 4 — Hierarchical agentic chunking:

        The LLM creates a multi-level chunk hierarchy:

        Level 1 (document summary):
            One 2-3 sentence summary of the entire document.

        Level 2 (section summaries):
            One summary per major section of the document.

        Level 3 (proposition level):
            Atomic facts extracted from each section.

        Retrieval can happen at any level.
        A high-level query hits Level 1 or 2.
        A specific factual query hits Level 3.

        This is sometimes called RAPTOR chunking
        (Recursive Abstractive Processing for Tree-Organized Retrieval).


---

## **5.f PROPOSITION vs AGENTIC — THE DISTINCTION**


    PROPOSITION CHUNKING:
    ──────────────────────
        Task given to LLM:  fixed and narrow
        "Decompose this passage into atomic facts."

        Output:             a flat list of propositions
        Structure:          no hierarchy
        LLM role:           extractor / decomposer
        Autonomy:           low — follows strict rules
        Best for:           factoid Q&A, dense knowledge bases


    AGENTIC CHUNKING:
    ──────────────────
        Task given to LLM:  open and broad
        "Understand this document and chunk it intelligently."

        Output:             chunks + metadata + summaries + structure
        Structure:          can be hierarchical
        LLM role:           analyst / architect
        Autonomy:           high — makes judgment calls
        Best for:           complex documents, multi-level retrieval,
                            documents with varied structure and depth


    COMBINED (most powerful):
    ──────────────────────────
    Use agentic chunking to determine section structure and generate
    summaries, then use proposition chunking within each section to
    create atomic retrievable facts.

        Document
            │
            ▼
        Agentic pass  →  [Section 1, Section 2, Section 3]
            │                            + summaries
            ▼                            + titles
        Proposition pass per section
            │
            ▼
        Hierarchy:
            Document summary (1 chunk)
            Section summaries (N chunks)
            Atomic propositions (M chunks per section)


---

## **5.g COST — THE MAIN TRADEOFF**

    Proposition and agentic chunking are the most expensive
    chunking strategies by a wide margin.

    An LLM API call is required for every passage in the document.

    COST COMPARISON for a 100-page document (~50,000 tokens):
    ───────────────────────────────────────────────────────────

        Method            LLM calls    Time         Cost estimate
        ──────────────────────────────────────────────────────────
        Fixed-size        0            ~0ms         $0.00
        Recursive         0            ~10ms        $0.00
        Sentence-based    0            ~100ms       $0.00
        Semantic          0 *          ~30s         ~$0.05 (embeddings)
        Proposition       50-100       ~5-10 min    ~$0.50-$2.00
        Agentic           50-200       ~10-30 min   ~$1.00-$5.00+

        * Semantic uses embedding model, not generative LLM

    Cost is paid ONCE at index time.
    Retrieval quality improvement often justifies the cost.
    Not suitable for real-time or high-volume ingestion pipelines.


    COST REDUCTION STRATEGIES:
    ────────────────────────────
    * Use a smaller / cheaper LLM for chunking (GPT-4o-mini, Haiku)
      and a larger LLM for answer generation
    * Cache propositions — reuse for documents that rarely change
    * Only apply to high-value documents where precision matters
    * Use proposition chunking for dense factual content,
      recursive for less critical or narrative content


---

## **5.h RETRIEVAL QUALITY — WHY IT IS WORTH THE COST**

    QUERY: "What year did Einstein win the Nobel Prize?"


    FIXED-SIZE chunk retrieved (300 tokens):
    ──────────────────────────────────────────
    "Albert Einstein was born in Ulm, Germany in 1879. He developed
     special relativity in 1905 and general relativity in 1915. His
     work on the photoelectric effect, for which he won the Nobel
     Prize in Physics in 1921, was foundational to quantum mechanics.
     He later moved to the United States in 1933 after the Nazi rise
     to power. His famous equation E=mc² is perhaps the most..."

        Noise ratio: HIGH. Answer buried in 300 tokens.
        LLM must extract answer from a large noisy chunk.


    SEMANTIC chunk retrieved (~60 tokens):
    ────────────────────────────────────────
    "Albert Einstein developed the theory of relativity and conducted
     foundational work on quantum mechanics. He won the Nobel Prize
     in Physics in 1921 for his discovery of the photoelectric effect.
     He later emigrated to the United States."

        Noise ratio: MEDIUM. Better but still contains irrelevant facts.


    PROPOSITION retrieved (~10 tokens):
    ─────────────────────────────────────
    "Albert Einstein won the Nobel Prize in Physics in 1921."

        Noise ratio: ZERO. Pure signal.
        Exact answer. Nothing else.

    This is the retrieval precision advantage of proposition chunking.
    Less noise in the context = more accurate LLM answer generation.


---

## **When to Use Proposition / Agentic Chunking**

    Scenario                                     Why it fits
    ─────────────────────────────────────────────────────────────────────────
    High-stakes Q&A systems                      Precision > cost
    Medical / legal knowledge bases              Every fact must be retrievable
    Dense factual documents (encyclopedias)      One sentence = many facts
    Enterprise knowledge management              Documents indexed once, queried many
    Research paper RAG systems                   Findings must be individually findable
    Offline batch indexing pipelines             Time and cost are acceptable
    When retrieval accuracy is the top priority  No other method matches quality


## **When NOT to Use Proposition / Agentic Chunking**

    * Real-time document ingestion — LLM latency is prohibitive
    * High-volume pipelines (millions of documents) — cost scales badly
    * Narrative or creative text (novels, stories) — atomic facts are not
      the unit of meaning in narrative content
    * Code repositories — code semantics not captured well by propositions
    * Frequently changing documents — re-chunking cost is too high
    * Budget-sensitive projects at large scale — start with semantic chunking


## **Real World Example**

    * LlamaIndex  — PropositionNodeParser implements Chen et al. paper directly
    * LangChain   — agentic chunking via custom LLM chains
    * Greg Kamradt— proposition chunking popularized in open source notebooks
    * RAPTOR      — hierarchical agentic chunking, available in LlamaIndex
    * Contextual Retrieval (Anthropic) — uses Claude to add context metadata
                                         to every chunk before indexing


---

## **Key Properties Summary**

    Property                  Proposition         Agentic
    ─────────────────────────────────────────────────────────────────
    Split boundary            Atomic facts        LLM judgment
    Chunk size                Very small (1 fact) Variable
    Measurement unit          Tokens              Tokens
    Requires LLM              Yes (always)        Yes (always)
    Output type               Flat propositions   Chunks + rich metadata
    Hierarchy support         No                  Yes
    Coreference resolution    Yes                 Yes
    Cuts mid-sentence         Never               Never
    Cuts mid-fact             Never               Never
    Retrieval precision       Highest             Highest
    Computational cost        Very high           Very high
    Best use                  Factoid Q&A         Complex document RAG


---

## **The Complete Chunking Ladder**

    METHOD            AWARENESS               SPLIT SIGNAL
    ──────────────────────────────────────────────────────────────────────
    Character-based   None                    Every N characters
    Word-based        Words                   Every N words
    Token-based       Model units             Every N tokens
    Recursive         Text structure          Best available separator
    Sentence-based    Grammar                 Sentence boundaries
    Semantic          Meaning                 Embedding similarity drop
    Proposition       Facts                   Atomic knowledge units    ←
    Agentic           Full understanding      LLM reasoning             ←

    Each rung on the ladder adds a deeper layer of understanding.
    The higher you go, the better the retrieval quality.
    The higher you go, the higher the computational cost.
    The art of RAG engineering is choosing the right rung
    for your quality requirements, latency budget, and cost constraints.


---

## **The Defining Characteristic**

    Character-based  →  operates on bytes
    Word-based       →  operates on words
    Token-based      →  operates on model units
    Sentence-based   →  operates on grammatical sentences
    Recursive        →  operates on structural separators
    Semantic         →  operates on embedding similarity
    Proposition      →  operates on KNOWLEDGE UNITS        ← unique
    Agentic          →  operates on REASONING              ← unique

    Every other chunking method is fundamentally passive —
    it reacts to signals already present in the text surface.

    Proposition and agentic chunking are active —
    they bring external intelligence to the chunking process.

    They do not find boundaries in the text.
    They create the best possible boundaries
    by understanding what the text means.

    That shift — from reading text to understanding it —
    is what places proposition and agentic chunking in a
    category of their own.

---


### **SLIDING WINDOW CHUNKING**

Sliding window chunking is fixed-size chunking with deliberate, controlled overlap.
Instead of producing non-overlapping chunks, each new chunk starts before the previous
one ends — the "window" slides forward by a step size smaller than the chunk size,
guaranteeing that every chunk boundary is covered by at least one chunk.

It is not a separate splitting strategy but a fundamental retrieval pattern that can
be layered on top of any chunking method.

    Fixed-size (no overlap):   [AAAA] [BBBB] [CCCC]
    Sliding window (overlap):  [AAABB] [AABBB] [ABBBC] [BBBCC]
                                ◄──────── step ─────►
                                 window slides forward by step_size

The defining parameters are:

    chunk_size   →  how many tokens per window
    step_size    →  how far the window moves forward each time
    overlap      →  chunk_size minus step_size (the repeated region)


---

## **SW.a THE CORE PROBLEM IT SOLVES**

    Without overlap, a fact that straddles a chunk boundary gets split:

        Text: "...the patient must take 500mg of ibuprofen | twice daily before meals..."
                                                            ^
                                                    chunk boundary

        Chunk 1: "...the patient must take 500mg of ibuprofen"
        Chunk 2: "twice daily before meals..."

        Neither chunk contains a complete medical instruction.
        A query for "ibuprofen dosage" retrieves an incomplete answer.


    With sliding window overlap:

        Chunk 1: "...the patient must take 500mg of ibuprofen"
        Chunk 2: "must take 500mg of ibuprofen twice daily before meals..."
                  ◄───────────── overlap region ─────────────►

        Chunk 2 now contains the complete instruction.
        The query retrieves a coherent, complete answer.


---

## **SW.b PARAMETER SELECTION**

    CHUNK SIZE AND STEP SIZE RELATIONSHIP:
    ─────────────────────────────────────────

        step_size = chunk_size - overlap

        chunk_size=500, overlap=100  →  step_size=400  (20% overlap)
        chunk_size=500, overlap=250  →  step_size=250  (50% overlap)
        chunk_size=256, overlap=50   →  step_size=206  (20% overlap)


    OVERLAP RECOMMENDATIONS BY CONTENT TYPE:
    ──────────────────────────────────────────

        Content Type                    Recommended Overlap
        ─────────────────────────────────────────────────────
        General prose / Wikipedia       10 – 15%
        Technical manuals               15 – 20%
        Legal / medical documents       20 – 25%
        Code (function bodies)          10% or full function
        Dense factual content           20 – 30%


    TOO LITTLE OVERLAP:
    ────────────────────
        ✗ Cross-boundary facts get split
        ✗ Pronouns at chunk start lose referent ("It was discovered that...")
        ✗ Retrieval misses answers that straddle boundaries


    TOO MUCH OVERLAP:
    ──────────────────
        ✗ Duplicate content stored in vector DB
        ✗ Same chunk retrieved multiple times = noisy context
        ✗ Storage cost grows significantly
        ✗ Embedding cost increases proportionally


---

## **SW.c SLIDING WINDOW VS PARENT DOCUMENT RETRIEVAL**

    Sliding window and parent document retrieval solve the same problem
    (boundary context loss) in fundamentally different ways:


    SLIDING WINDOW:
    ────────────────
        Ensures overlap by DUPLICATING content at boundaries.
        Simple, no extra infrastructure needed.
        The same text appears in multiple chunks.
        Easy to implement with any chunker.


    PARENT DOCUMENT RETRIEVAL (small-to-large):
    ─────────────────────────────────────────────
        Keeps SMALL precise chunks for retrieval.
        At retrieval time, fetches the LARGER parent passage for context.
        No duplication — each piece of text stored once.
        Better storage efficiency. Requires metadata linking infrastructure.

        See the Small-to-Large section for the full explanation.


    WHEN TO USE WHICH:
    ────────────────────

        Sliding window  →  simple setup, no metadata infrastructure,
                           acceptable storage overhead, small corpora

        Parent document →  production systems at scale, strict deduplication
                           required, complex retrieval pipelines


---

## **SW.d KEY PROPERTIES SUMMARY**

    Property                    Sliding Window Chunking
    ────────────────────────────────────────────────────────────────
    Split boundary              Fixed size with controlled overlap
    Chunk size                  Fixed (user-defined)
    Overlap                     Explicit and configurable
    Requires NLP library        No
    Handles abbreviations       No
    Cuts mid-sentence           Sometimes (like fixed-size)
    Removes boundary blind spot Yes — this is the entire purpose
    Good for LLM pipelines      Yes
    Computational cost          Low (no model needed)
    Storage overhead            Increases with overlap %
    Implementation complexity   Very low


---


### **CONTEXTUAL RETRIEVAL**

Contextual Retrieval is a technique published by Anthropic in 2024 that dramatically
improves retrieval accuracy by solving a fundamental weakness in how chunks are stored
and retrieved.

The core problem it addresses: individual chunks, taken out of their document context,
often lose the meaning they had in context.

A chunk that reads "The defendant was found liable for damages under Section 4(b)." is
meaningless without knowing which contract, which defendant, and what Section 4(b) says.
Embeddings of decontextualized chunks match queries poorly.

Contextual Retrieval fixes this by prepending a short context summary to each chunk
before it is embedded and stored — so the chunk carries its own context into the
vector store.

    Standard RAG:
        Chunk stored:  "The defendant was found liable for damages under Section 4(b)."
        Embedding:      captures only the surface text above

    Contextual Retrieval:
        Chunk stored:  "This chunk is from the Smith vs. Acme Corp. employment
                        contract (2022). Section 4(b) governs non-compete violations.
                        The defendant was found liable for damages under Section 4(b)."
        Embedding:      now captures document identity, topic, and the fact


---

## **CR.a THE CORE MECHANISM**


    STEP 1 — CHUNK THE DOCUMENT NORMALLY
    ──────────────────────────────────────
    Use any chunking strategy (recursive, sentence, semantic) to produce chunks.
    This step is unchanged from standard RAG.

        Document  →  [Chunk 1, Chunk 2, Chunk 3 ... Chunk N]


    STEP 2 — FOR EACH CHUNK, GENERATE A CONTEXT PREFIX
    ─────────────────────────────────────────────────────
    Send the FULL DOCUMENT and the INDIVIDUAL CHUNK to an LLM.
    Ask the LLM to write a short (1-3 sentence) description of where this
    chunk sits in the document and what surrounding context matters.

        PROMPT:
        ────────
        <document>
        {full_document_text}
        </document>

        Here is a chunk from this document:
        <chunk>
        {chunk_text}
        </chunk>

        Give a short, succinct context that situates this chunk within
        the overall document. This context will be prepended to the chunk
        for the purpose of improving retrieval. Reply only with the context,
        no preamble.


    STEP 3 — PREPEND CONTEXT TO CHUNK
    ────────────────────────────────────
    Concatenate the LLM-generated context with the original chunk text.

        Contextual chunk = context_prefix + "\n\n" + original_chunk_text

        Stored in vector DB: the full contextual chunk
        Displayed to LLM at generation time: the full contextual chunk


    STEP 4 — EMBED AND STORE CONTEXTUAL CHUNKS
    ────────────────────────────────────────────
    Each contextual chunk is embedded and stored exactly as in standard RAG.
    No change to retrieval infrastructure. The improvement is purely in
    what gets embedded.


---

## **CR.b WHAT THE CONTEXT PREFIX LOOKS LIKE**

    EXAMPLE — Legal contract chunk:
    ─────────────────────────────────

        ORIGINAL CHUNK (what standard RAG would embed):
        ──────────────────────────────────────────────────
        "The termination notice period shall be no less than thirty (30)
         days. Failure to provide adequate notice shall result in the
         forfeiture of accrued benefits."

        LLM-GENERATED CONTEXT PREFIX:
        ───────────────────────────────
        "This chunk is from the Acme Corp Employee Handbook (2023),
         Section 7: Termination Procedures. It specifies the minimum
         notice period required for employee or employer termination."

        CONTEXTUAL CHUNK (what Contextual Retrieval embeds):
        ──────────────────────────────────────────────────────
        "This chunk is from the Acme Corp Employee Handbook (2023),
         Section 7: Termination Procedures. It specifies the minimum
         notice period required for employee or employer termination.

         The termination notice period shall be no less than thirty (30)
         days. Failure to provide adequate notice shall result in the
         forfeiture of accrued benefits."

    The embedding of the contextual chunk now captures:
        ✓ Document identity (Acme Corp Employee Handbook)
        ✓ Section identity (Section 7: Termination Procedures)
        ✓ Topic (termination notice periods)
        ✓ The original fact (30 days, benefit forfeiture)

    A query like "notice period for leaving Acme" now retrieves this chunk
    correctly — the standard chunk would likely be missed entirely.


---

## **CR.c CONTEXTUAL RETRIEVAL + BM25 (THE FULL SYSTEM)**

    Anthropic's published results combined Contextual Retrieval with BM25
    keyword search for the highest retrieval accuracy:

    STANDARD RAG:
    ──────────────
        Query  →  Embedding similarity search  →  Top K chunks

    CONTEXTUAL RETRIEVAL + BM25:
    ─────────────────────────────
        Query  →  Embedding similarity search  →  Top K_1 chunks
                │
                └──  BM25 keyword search       →  Top K_2 chunks
                │
                └──  Combine + deduplicate  →  Rerank  →  Final top K

        BM25 catches keyword-exact matches that embeddings miss.
        Embeddings catch semantic matches that BM25 misses.
        Reranking (e.g. Cohere Rerank) selects the best of both.


    REPORTED RESULTS (from Anthropic's 2024 publication):
    ──────────────────────────────────────────────────────
        Method                              Retrieval failure rate
        ─────────────────────────────────────────────────────────────
        Standard RAG                        5.7%
        + Contextual Retrieval              3.0%   (47% improvement)
        + Contextual Retrieval + BM25       1.9%   (67% improvement)
        + Above + Reranking                 1.7%   (70% improvement)

    These are substantial improvements for a relatively simple change
    to the indexing pipeline.


---

## **CR.d COST AND CACHING**

    The main cost of Contextual Retrieval is the LLM call per chunk.

    For a document with 100 chunks, you make 100 LLM calls at indexing time.
    Each call sends the full document + the chunk, so token costs can add up.

    COST REDUCTION — PROMPT CACHING:
    ──────────────────────────────────
    Anthropic's API supports prompt caching. Since the full document is
    the same for every chunk call, it is cached after the first call.
    Subsequent calls for the same document only charge for the chunk
    portion of the prompt.

        Without caching:  100 chunks × (full_doc + chunk) tokens each
        With caching:     1 × full_doc + 100 × chunk tokens

        Typical cost reduction: 80-90% for large documents


---

## **CR.e WHEN TO USE CONTEXTUAL RETRIEVAL**

    Scenario                                     Why it fits
    ─────────────────────────────────────────────────────────────────────────
    Documents with many similar sections         Context disambiguates sections
    Legal / financial / technical documents      Precise attribution matters
    Multi-document corpora with related content  Prevents cross-document confusion
    Long documents where chunks lose context     Each chunk becomes self-contained
    Production RAG where precision is critical   The best quality/cost ratio gain
    Any RAG system as a drop-in improvement      Works with any chunking strategy


## **When NOT to Use Contextual Retrieval**

    * Real-time ingestion — LLM call per chunk adds latency
    * Very short documents — context is already obvious
    * High-volume pipelines at cost-sensitive scale — mitigate with caching
    * Simple FAQ systems with self-contained Q&A pairs — chunks already have context


## **Key Properties Summary**

    Property                    Contextual Retrieval
    ────────────────────────────────────────────────────────────────
    What it modifies            What gets embedded (not split logic)
    Works with                  Any chunking strategy
    LLM required                Yes — at indexing time only
    Context prefix length       1-3 sentences (50-100 tokens typical)
    Improves retrieval by       ~47-70% failure rate reduction
    Cost reduction available    Yes — via prompt caching
    Infrastructure change       None (same vector DB, same retrieval)
    Published by                Anthropic (2024)


---


### **LATE CHUNKING**

Late Chunking is a technique published by Jina AI in 2024 that fundamentally
reverses the order of the embedding-then-chunking pipeline.

In standard RAG, the sequence is:

    Text → CHUNK → Embed each chunk separately → Store

The problem: when you embed a chunk in isolation, the embedding model only
attends to that chunk's text. It has no awareness of what came before or after.
The surrounding context that gives the chunk meaning is invisible at embedding time.

Late Chunking fixes this by reversing the order:

    Text → Embed the FULL DOCUMENT (with long-context model) → CHUNK the embeddings

By embedding first, the model's attention mechanism processes the entire document.
Every token's representation is enriched by every other token in the document.
Then you chunk the already-contextual embeddings — each chunk carries full-document
awareness built into its vectors.


---

## **LC.a THE CORE MECHANISM**


    STANDARD CHUNKING PIPELINE:
    ─────────────────────────────

        Document text
            │
            ▼
        Split into chunks:
        ["Einstein was born...", "He developed...", "He won..."]
            │
            ▼
        Embed EACH CHUNK independently:
        Embedding("Einstein was born...") — model sees only this chunk
        Embedding("He developed...")      — model sees only this chunk
        Embedding("He won...")            — model sees only this chunk
            │
            ▼
        Store chunk embeddings in vector DB


    LATE CHUNKING PIPELINE:
    ────────────────────────

        Document text
            │
            ▼
        Embed THE FULL DOCUMENT using a long-context embedding model
        Model attends to all tokens simultaneously — full cross-context
            │
            ▼
        Token-level embeddings produced:
        [e1, e2, e3, e4, e5, e6, ... eN]  ← one vector per token
        Each ei is influenced by ALL surrounding tokens
            │
            ▼
        CHUNK the token embeddings using the same boundaries as before:
        Chunk 1 embeddings: [e1, e2, e3, ...]    ← pool these into one vector
        Chunk 2 embeddings: [e4, e5, e6, ...]    ← pool these into one vector
        Chunk 3 embeddings: [e7, e8, e9, ...]    ← pool these into one vector
            │
            ▼  (mean pooling per chunk)
        Final chunk vectors — each carrying full-document context
            │
            ▼
        Store in vector DB exactly as normal


---

## **LC.b WHY THE ORDER MATTERS — THE PRONOUN PROBLEM**

    Consider this passage about a researcher:

        S1: "Dr. Amara Osei joined the lab in 2018."
        S2: "She published three papers on CRISPR."
        S3: "Her work was cited over 200 times."

    STANDARD CHUNKING — embedding S2 in isolation:
    ────────────────────────────────────────────────
    The model embeds "She published three papers on CRISPR."
    "She" has no referent. The embedding captures a generic
    sentence about an anonymous person publishing CRISPR research.
    The embedding quality for the specific researcher is degraded.

    LATE CHUNKING — embedding S2 within the full document:
    ───────────────────────────────────────────────────────
    The model processes the entire passage at once.
    When encoding "She" in S2, the attention mechanism has already
    processed "Dr. Amara Osei" from S1. The token embedding for "She"
    contains the full meaning: this is Dr. Amara Osei.
    The chunk embedding for S2 accurately represents the specific researcher.

    This is the core quality advantage of Late Chunking.
    Coreference, entity resolution, and cross-sentence meaning are
    all handled natively by the model's attention — not by overlap heuristics.


---

## **LC.c REQUIREMENTS AND CONSTRAINTS**

    Late Chunking requires a long-context embedding model.
    Standard 512-token models cannot encode a full document in one pass.

    COMPATIBLE MODELS:
    ───────────────────

        Model                       Max Tokens    Late Chunking Support
        ──────────────────────────────────────────────────────────────────
        jina-embeddings-v2-base     8192          ✓ (designed for this)
        jina-embeddings-v3          8192          ✓ (native support)
        nomic-embed-text-v1.5       8192          ✓
        text-embedding-3-large      8191          ✓
        all-MiniLM-L6-v2            512           ✗ (too short)
        bge-large-en-v1.5           512           ✗ (too short)

    DOCUMENT LENGTH CONSTRAINT:
    ─────────────────────────────
    The entire document must fit in the model's context window.
    For very long documents (books, 100+ page PDFs), late chunking
    must be applied section-by-section, not to the full document.

    POOLING METHOD:
    ────────────────
    After chunking the token embeddings, the token vectors within
    each chunk must be pooled into a single chunk vector.
    Mean pooling (average of all token embeddings in the chunk)
    is the standard approach and performs best in practice.


---

## **LC.d LATE CHUNKING vs CONTEXTUAL RETRIEVAL — COMPARISON**

    Both techniques solve the context loss problem. They do so differently.

    ┌───────────────────────┬──────────────────────────────┬──────────────────────────────┐
    │ Dimension             │ Contextual Retrieval         │ Late Chunking                │
    ├───────────────────────┼──────────────────────────────┼──────────────────────────────┤
    │ How context added     │ LLM writes a text prefix     │ Long-context embedding model │
    │ When context added    │ Before embedding             │ During embedding             │
    │ Requires LLM          │ Yes (generative model)       │ No (embedding model only)    │
    │ Requires long-context │ No                           │ Yes (8K+ embedding model)    │
    │ Cost                  │ LLM tokens per chunk         │ One embedding pass/document  │
    │ Context type          │ Explicit natural language    │ Implicit attention-based     │
    │ Works with any model  │ Yes                          │ Only long-context embedders  │
    │ Published by          │ Anthropic (2024)             │ Jina AI (2024)               │
    └───────────────────────┴──────────────────────────────┴──────────────────────────────┘

    They can be combined: apply late chunking to get context-rich token
    embeddings, then also prepend a text context prefix before storage.
    In practice, late chunking alone is often sufficient.


---

## **LC.e KEY PROPERTIES SUMMARY**

    Property                    Late Chunking
    ────────────────────────────────────────────────────────────────
    What it modifies            Embedding order (embed first, chunk after)
    Split strategy              Any (same as normal chunking)
    Requires long-context model Yes (8K+ tokens minimum)
    Requires generative LLM     No
    Context mechanism           Model attention over full document
    Handles coreference         Yes — natively
    Pooling method              Mean pooling per chunk
    Infrastructure change       Embedding pipeline only
    Best for                    Documents where pronouns/references matter
    Published by                Jina AI (2024)


---


### **RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)**

RAPTOR is a hierarchical indexing and retrieval framework published by researchers
at Stanford in 2024. It addresses a fundamental limitation of flat chunking:
all chunks live at the same level of abstraction, so high-level synthesis queries
perform poorly while low-level factoid queries perform well — but never both.

RAPTOR builds a tree of summaries over the chunk corpus. Leaf nodes are the
original text chunks. Each higher level is an LLM-generated summary of a
cluster of the level below. Retrieval can access any level of the tree,
matching the right abstraction depth to the query type.

    Standard RAG (flat):
        [chunk][chunk][chunk][chunk][chunk][chunk][chunk][chunk]
        All at the same level. No synthesis. No hierarchy.

    RAPTOR (tree):
                        [Document summary]         ← Level 3
                       /                 \
            [Section A summary]  [Section B summary]  ← Level 2
            /        \               /        \
        [chunk][chunk]           [chunk][chunk]       ← Level 1 (leaves)


---

## **RAP.a THE RAPTOR ALGORITHM — STEP BY STEP**


    STEP 1 — CREATE BASE CHUNKS
    ────────────────────────────
    Standard chunking (recursive or sentence-based) produces the leaf nodes.
    These are the raw text chunks, the same as any RAG system would use.

        Full Document  →  [C1, C2, C3, C4, C5, C6 ... CN]


    STEP 2 — EMBED ALL CHUNKS
    ──────────────────────────
    Each chunk is embedded. The embeddings live in a shared vector space.

        [C1, C2, C3, C4, C5, C6]  →  [v1, v2, v3, v4, v5, v6]


    STEP 3 — CLUSTER SIMILAR CHUNKS
    ─────────────────────────────────
    Chunks with similar embeddings (similar topics) are grouped using
    Gaussian Mixture Models (GMM) — a soft clustering algorithm.
    Unlike k-means, GMM allows one chunk to belong to multiple clusters,
    which is important because chunks can cover multiple topics.

        Cluster A (physics):   [C1, C2, C3]
        Cluster B (history):   [C4, C5]
        Cluster C (bio):       [C6, C7, C8]

        Note: RAPTOR uses soft clustering — a chunk about Einstein
        can appear in both a "physics" cluster and a "history" cluster.


    STEP 4 — SUMMARIZE EACH CLUSTER WITH AN LLM
    ──────────────────────────────────────────────
    The chunks within each cluster are concatenated and sent to an LLM.
    The LLM generates a coherent summary of the cluster.

        Cluster A chunks → LLM → "A summary of the key physics concepts
                                   discussed, including special relativity
                                   and the photoelectric effect."

    These summaries become the PARENT nodes — one level up in the tree.


    STEP 5 — EMBED THE SUMMARIES
    ──────────────────────────────
    The new summary nodes are embedded and added to the vector space.
    They now exist alongside the original chunks.


    STEP 6 — RECURSE
    ─────────────────
    Steps 3-5 are repeated on the summary nodes, creating a new level.
    This continues until a single root summary covers the entire document.

        Level 1 (leaves):     original chunks              — precise
        Level 2 (clusters):   paragraph-group summaries    — medium
        Level 3 (sections):   section summaries            — broad
        Level 4 (root):       full document summary        — highest


---

## **RAP.b RAPTOR RETRIEVAL**

    RAPTOR supports two retrieval modes:


    MODE 1 — TREE TRAVERSAL:
    ──────────────────────────
    Start at the root. Find the most similar summary node.
    Descend into its children. Find the most similar child.
    Continue descending until reaching the leaf level.

        Query → Root → Match Section A → Descend → Match Paragraph 2
             → Descend → Match Chunk C3  ← final retrieved chunk

        Pros:  Efficient, focused traversal
        Cons:  If wrong branch taken early, correct answer missed


    MODE 2 — COLLAPSED TREE (recommended):
    ────────────────────────────────────────
    All nodes at ALL levels are placed in a single flat vector store.
    Retrieval does one single search across all nodes simultaneously.

        Query → Search ALL nodes (leaves + all summary levels) simultaneously
             → Return the top-K nodes regardless of their tree level

        A high-level query ("summarise the physics section") matches a
        Level 3 summary node.
        A precise query ("when did Einstein win the Nobel Prize") matches
        a Level 1 leaf chunk.

        Pros:  Single retrieval step, adapts automatically to query type
        Cons:  Vector store is larger (stores all levels)

    The collapsed tree mode is what RAPTOR uses by default and is what
    most production implementations adopt.


---

## **RAP.c WHY RAPTOR OUTPERFORMS FLAT RAG ON SYNTHESIS QUERIES**

    QUERY: "What were the major themes across Einstein's contributions to physics?"

    FLAT RAG:
    ──────────
        Returns top-3 leaf chunks, e.g.:
        C1: "Einstein developed special relativity in 1905."
        C4: "Einstein won the Nobel Prize for the photoelectric effect."
        C7: "Einstein's work on Brownian motion confirmed atomic theory."

        These are facts, not a synthesis.
        The LLM must synthesize from disconnected chunks — often poorly.
        The "major themes" framing requires an overview that no single
        leaf chunk contains.

    RAPTOR (collapsed tree):
    ─────────────────────────
        Query matches a Level 3 summary node:
        "Einstein's physics contributions span three major areas:
         the special and general theories of relativity which
         redefined space-time; his quantum contributions including
         the photoelectric effect and Bose-Einstein statistics;
         and his foundational work on statistical mechanics."

        This is the answer. One retrieval. No synthesis required.

    RAPTOR doesn't just improve retrieval — it fundamentally changes
    what information is available to retrieve.


---

## **RAP.d COST AND TRADEOFFS**

    COST COMPARISON for a 100-page document:
    ──────────────────────────────────────────

        Method              LLM calls   Extra storage   Extra cost
        ─────────────────────────────────────────────────────────────
        Standard RAG         0          0               $0.00
        RAPTOR (3 levels)    N/5 +      ~40% more       ~$0.20–$1.00
                             N/25 +     vectors
                             1
        (N = number of base chunks, roughly N/5 clusters per level)

    TRADEOFFS:
    ───────────

        Pros:
            + Synthesis queries work — flat RAG cannot match this
            + High-level and low-level queries both served from one index
            + Retrieved summaries are coherent, pre-synthesized text
            + Reduces LLM burden at generation time

        Cons:
            − LLM cost at indexing time
            − More complex pipeline than standard RAG
            − Summary quality depends on the LLM used
            − Summaries can hallucinate or distort if LLM is weak


---

## **When to Use RAPTOR**

    Scenario                                     Why it fits
    ─────────────────────────────────────────────────────────────────────────
    Long documents requiring synthesis queries   Tree structure enables overview retrieval
    Research papers, textbooks, reports          Multi-level abstraction matches content
    Enterprise knowledge bases                   Documents queried at multiple depths
    Chatbots that must answer "explain X"        Summary nodes provide direct answers
    When flat RAG gives incomplete summaries     RAPTOR's design directly solves this


## **When NOT to Use RAPTOR**

    * Very short documents — tree adds cost with no benefit
    * Real-time ingestion — tree building requires offline indexing
    * Purely factoid Q&A — leaf chunks are sufficient, tree is overkill
    * Frequently updated documents — tree must be rebuilt on changes
    * Budget-constrained pipelines — LLM cost at all tree levels adds up


## **Key Properties Summary**

    Property                    RAPTOR
    ────────────────────────────────────────────────────────────────
    Structure                   Tree (leaves = chunks, parents = summaries)
    Levels                      Typically 3-4 (configurable)
    Clustering algorithm        Gaussian Mixture Models (soft clustering)
    Summarization               LLM-generated per cluster
    Retrieval mode              Collapsed tree (all levels in flat store)
    Query types served          Factoid AND synthesis (both)
    LLM required                Yes — at indexing time
    Extra storage               ~30-50% more vectors vs flat RAG
    Published by                Stanford NLP (2024)
    Available in                LlamaIndex (built-in support)


---


### **CHUNK DEDUPLICATION**

Chunk deduplication is a post-processing step applied after chunking that removes
or merges near-identical chunks before they are embedded and stored in the vector DB.

It is not a chunking strategy — it is a quality control step that prevents a common
production problem: duplicate and near-duplicate chunks degrading retrieval quality.


## **Why Deduplication Matters**

    Duplicates enter the vector store through multiple routes:

    ROUTE 1 — OVERLAP:
        Sliding window chunking deliberately creates overlapping content.
        Chunk N and Chunk N+1 share 10-20% of their text.
        With high overlap, two adjacent chunks can be >80% identical.

    ROUTE 2 — REPEATED DOCUMENT CONTENT:
        Legal documents repeat standard clauses across sections.
        Technical manuals repeat safety warnings on every page.
        FAQ documents list the same answer under multiple phrasings.

    ROUTE 3 — CORPUS INGESTION:
        The same document ingested twice (version update, re-index).
        Multiple files that contain the same boilerplate (headers, footers).
        Web scrapes that include navigation text on every page chunk.

    ROUTE 4 — MULTI-DOCUMENT CORPORA:
        Different documents quoting the same passage.
        Wikipedia article and a blog post covering the same topic.


---

## **DED.a WHAT DUPLICATE CHUNKS DO TO RETRIEVAL**

    When the vector store contains duplicates, retrieval quality degrades:


    PROBLEM 1 — RETRIEVAL DILUTION:
    ─────────────────────────────────
    A query retrieves top-5 chunks. If 3 of the 5 are near-duplicates
    of each other, the LLM receives only ~3 unique pieces of information
    instead of 5. Context window is wasted on repeated content.

        Query: "What is the refund policy?"

        Retrieved chunks:
        C1: "Refunds are processed within 5-7 business days. [...]"
        C2: "Refunds are processed within 5-7 business days. [...]"   ← duplicate
        C3: "Refunds are processed within 5-7 business days. [...]"   ← duplicate
        C4: "Items must be returned within 30 days of purchase."
        C5: "Original packaging required for all returns."

        3 out of 5 chunks carry the same information.
        C4 and C5, which have complementary policy details, compete with
        duplicates for the top-5 slots and may be pushed out by lower-ranked
        but unique content.


    PROBLEM 2 — SCORE INFLATION:
    ──────────────────────────────
    Near-duplicate chunks have nearly identical embeddings.
    They cluster together in vector space.
    A query near this cluster retrieves all of them at once,
    pushing out potentially more relevant unique chunks.


    PROBLEM 3 — LLM CONFUSION:
    ────────────────────────────
    Seeing the same information three times in the context can cause
    the LLM to over-weight that fact or structure its answer around
    the repeated content rather than synthesizing from all sources.


---

## **DED.b DEDUPLICATION METHODS**

    METHOD 1 — EXACT DEDUPLICATION (simplest)
    ──────────────────────────────────────────
    Hash each chunk text (MD5 or SHA-256).
    If two chunks have the same hash → identical text → remove one.

        hash(chunk_text) → if seen before → discard

        Cost:    O(N) hash operations
        Catches: Perfect duplicates only
        Misses:  "Refunds processed in 5-7 days." vs
                 "Refunds are processed within 5-7 business days."
                 (same meaning, different hash)


    METHOD 2 — MINHASH / LSH (near-duplicate detection)
    ─────────────────────────────────────────────────────
    MinHash estimates Jaccard similarity between chunks (word-level overlap).
    Locality Sensitive Hashing (LSH) groups similar chunks into buckets
    without comparing all pairs.

        Chunks with Jaccard similarity > threshold → deduplicate

        Common threshold: 0.8 (80% word overlap)
        Cost:    O(N) with small constant (no pairwise comparison)
        Catches: Near-duplicates with rewording at word level
        Used by: Elasticsearch, large-scale corpus cleaning pipelines


    METHOD 3 — EMBEDDING COSINE SIMILARITY
    ────────────────────────────────────────
    Embed all chunks. Compute cosine similarity between chunk embeddings.
    Remove chunks whose embedding similarity exceeds a threshold.

        cosine_similarity(embed(chunk_A), embed(chunk_B)) > 0.95
        → keep one, discard the other

        Cost:    O(N²) naive; use ANN index for large corpora
        Catches: Semantic near-duplicates (same meaning, different words)
        Most accurate but most expensive method


    METHOD 4 — CHUNK MERGING (instead of discarding)
    ──────────────────────────────────────────────────
    Instead of removing one of two near-duplicates, merge their unique
    content into one consolidated chunk.

        Chunk A: "Refunds take 5-7 days. Contact support@acme.com."
        Chunk B: "Refunds take 5-7 days. Return window is 30 days."

        Merged: "Refunds take 5-7 days. Contact support@acme.com.
                 Return window is 30 days."

        Best for: Complementary near-duplicates that each add something.


---

## **DED.c DEDUPLICATION IN THE CHUNKING PIPELINE**

    Deduplication fits as a post-processing step after chunking
    and before embedding:

        Raw Document(s)
            │
            ▼
        Chunking Strategy (any method)
            │
            ▼
        Chunk candidates  [C1, C2, C3, C4, C5, ...]
            │
            ▼
        Deduplication
        ┌──────────────────────────────────────────────┐
        │ Hash-based exact dedup   →  remove identical │
        │ MinHash near-dedup       →  remove >80% same │
        │ Embedding similarity     →  remove >0.95 sim │
        └──────────────────────────────────────────────┘
            │
            ▼
        Deduplicated chunks  [C1, C3, C5, ...]
            │
            ▼
        Embedding
            │
            ▼
        Vector DB


---

## **DED.d WHEN DEDUPLICATION IS ESSENTIAL**

    Always necessary:
        ✓ Overlap > 20% (sliding window with high overlap)
        ✓ Corpus ingested from multiple sources with shared content
        ✓ Documents with repeated boilerplate (headers, footers, disclaimers)
        ✓ Web scraped content (navigation, footers repeated on every page)
        ✓ Any corpus re-ingested after updates

    Usually not necessary:
        ✗ Single clean document chunked once with no overlap
        ✗ Corpus with strictly unique, non-overlapping content
        ✗ Proposition chunking (each proposition is inherently unique)


## **Key Properties Summary**

    Property                    Chunk Deduplication
    ────────────────────────────────────────────────────────────────
    Pipeline stage              Post-chunking, pre-embedding
    Methods available           Exact hash, MinHash/LSH, embedding sim
    What it prevents            Diluted retrieval, score inflation
    Cost                        Low (hash), Medium (MinHash), High (embed)
    Most common threshold       Jaccard > 0.8 or cosine similarity > 0.95
    When essential              High overlap, multi-source corpora


---


## **Updated Complete Chunking Taxonomy**

    METHOD                  AWARENESS               SPLIT SIGNAL               COST
    ─────────────────────────────────────────────────────────────────────────────────────
    Character-based         None                    Every N characters         Free
    Word-based              Words                   Every N words              Free
    Token-based             Model units             Every N tokens             Free
    Sliding Window          Structure + overlap     Fixed size + step size     Free
    Recursive               Text structure          Best available separator   Free
    Sentence-based          Grammar                 Sentence boundaries        Low
    Semantic                Meaning                 Embedding similarity drop  Medium
    Proposition             Facts                   Atomic knowledge units     High
    Agentic                 Full understanding      LLM reasoning              Very High
    ─────────────────────────────────────────────────────────────────────────────────────
    Post-processing:
    Deduplication           Similarity / hashing    Duplicate removal          Low–Medium
    ─────────────────────────────────────────────────────────────────────────────────────
    Indexing enhancements:
    Contextual Retrieval    Full document context   LLM-written prefix         Medium
    Late Chunking           Full document attention Embedding-level context    Medium
    RAPTOR                  Multi-level abstraction Tree of LLM summaries      High


    The three tiers of modern production RAG:

    TIER 1 — Standard quality, low cost:
        Recursive chunking + token measurement + 10-20% overlap
        Suitable for most use cases and the correct starting point.

    TIER 2 — High quality, moderate cost:
        Semantic chunking + Contextual Retrieval + BM25 hybrid retrieval
        + Deduplication. Used in serious production systems.

    TIER 3 — Maximum quality, high cost (offline indexing only):
        Proposition chunking + RAPTOR tree + Late Chunking
        + Reranking. Used for high-stakes knowledge bases where
        retrieval precision justifies the indexing cost.


---



  ┌──────────────────────────────────────────────────────────┐
  │  SWEET SPOT                                              │
  │                                                          │
  │  Token range:   128 – 512 tokens per chunk               │
  │  Overlap:       10–15% of chunk size                     │
  │  Semantic unit: 1 coherent idea / paragraph              │
  │                                                          │
  │  Too small  (<50 tokens)  → lost context, weak vectors   │
  │  Too large  (>1024 tokens) → diluted retrieval signal    │
  └──────────────────────────────────────────────────────────┘

The best real-world approach is often proposition chunking with a parent-child structure — 
store atomic facts as child chunks for precise retrieval, but return their parent paragraph 
to the LLM for full context. 
This gives you precision on the retrieval side and coherence on the generation side.

This is a genuinely nuanced question and the honest answer is — it depends, 
and neither number is universally correct. 

Here is the full picture:

    Why 512 Became the Standard
    512 was not an arbitrary choice. 
    
    It came directly from the hardware constraints of early embedding models:
    
    BERT (2018)              →  512 token hard limit
    sentence-transformers    →  512 token hard limit
    text-embedding-ada-001   →  2046 token limit but degraded after 512
    all-MiniLM-L6-v2         →  512 token hard limit

The entire first generation of embedding models was trained on 512 tokens.
Chunks had to match the model they were being embedded with.
512 was not a best practice — it was a hardware ceiling.

Why 512 Is Considered Outdated Today

    Modern embedding models have broken the 512 ceiling entirely:
    
    Model                           Max Tokens    Optimal Range
    ────────────────────────────────────────────────────────────
    text-embedding-ada-002          8191          256 - 1024
    text-embedding-3-small          8191          256 - 1024
    text-embedding-3-large          8191          256 - 2048
    bge-large-en-v1.5               512           256 - 512   ← still 512
    all-mpnet-base-v2               512           256 - 512   ← still 512
    nomic-embed-text-v1.5           8192          256 - 2048
    Cohere embed-english-v3.0       512           256 - 512   ← still 512
    jina-embeddings-v2-base         8192          512 - 2048
    
    So the answer is model-dependent — not a universal upgrade from 512 to 1024.

The Real Tradeoff — Precision vs Context

    This is the core tension that chunk size controls:
    
    SMALLER CHUNKS (128 - 512 tokens)
    ───────────────────────────────────
    + Higher retrieval precision
      → Embedding represents one focused idea
      → Less noise in retrieved chunk
      → LLM answer generation is cleaner
    
    - Less context per chunk
      → A fact may need surrounding sentences to make sense
      → Multi-sentence reasoning is harder
      → More chunks needed to cover a topic


    LARGER CHUNKS (512 - 2048 tokens)
    ───────────────────────────────────
    + More context per chunk
      → Reasoning that spans multiple sentences is preserved
      → Better for complex questions requiring synthesis
      → Fewer chunks needed
    
    - Lower retrieval precision
      → Embedding averages over many ideas
      → One irrelevant paragraph dilutes the whole embedding
      → Harder to match a specific query to a large chunk

What Research and Production Systems Show

    FINDING 1 — The precision-context curve peaks around 256-512 for factoid Q&A:
    
        Chunk size:      128    256    512    1024   2048
        Factoid Q&A:     ███    ████   ████   ███    ██
        Reasoning Q&A:   ██     ███    ████   ████   ████
    
        Factoid questions (who, what, when) → smaller chunks win
        Reasoning questions (why, how, compare) → larger chunks win
    
    
    FINDING 2 — The embedding model ceiling matters more than the number:
    
        If your embedding model is bge-large (512 limit):
        → 1024 token chunks get TRUNCATED to 512 during embedding
        → You are storing 1024 tokens but only embedding half of them
        → The tail of every chunk is invisible to retrieval
        → This is WORSE than 512 chunks, not better
    
    
    FINDING 3 — LLM context windows are now large enough that chunk size
                matters less for generation, more for retrieval:
    
        GPT-4o      →  128K context  →  can handle large chunks easily
        Claude 3.5  →  200K context  →  same
        The bottleneck is RETRIEVAL RELEVANCE not generation capacity
        Smaller precise chunks retrieve better, then get passed to large context LLM

The Modern Best Practice — Chunk Size by Use Case

    USE CASE                        RECOMMENDED CHUNK SIZE
    ────────────────────────────────────────────────────────────────
    Factoid Q&A (precise facts)     128  - 256  tokens
    Standard RAG pipelines          256  - 512  tokens
    Reasoning / synthesis tasks     512  - 1024 tokens
    Summarization pipelines         1024 - 2048 tokens
    Proposition chunking            10   - 50   tokens (atomic facts)

The Actual Modern Answer

    512 is outdated only if:
    
        ✓ Your embedding model supports > 512 tokens natively
        ✓ Your queries require multi-sentence reasoning
        ✓ Your documents have dense, long-form content
        ✓ You are doing summarization, not factoid retrieval

512 is still correct if:

    ✓ Your embedding model has a 512 token limit (bge, MiniLM, Cohere)
    ✓ Your queries are factoid (specific facts, dates, names)
    ✓ You are using proposition chunking (512 is overkill anyway)
    ✓ Retrieval precision matters more than context richness

The Production Recommendation Today

Rather than picking 512 or 1024 as a fixed number, the real answer is:

    STEP 1 — Check your embedding model's native token limit first.
              Never chunk larger than what it can embed without truncation.
    
    STEP 2 — Match chunk size to query type.
              Factoid queries  → smaller (256-512)
              Reasoning queries → larger (512-1024)
    
    STEP 3 — Use the small-to-large retrieval pattern:
              Index at small chunk size (256-512) for precision.
              At retrieval time, fetch the small chunk + its surrounding
              parent passage for context.
              Best of both worlds — precise retrieval, rich context.
    
    STEP 4 — Always evaluate empirically on YOUR data.
              The right chunk size is the one that scores best on
              your retrieval benchmark. No universal number wins.
              
The small-to-large pattern in Step 3 is what most serious production RAG systems use today — 
it makes the 512 vs 1024 debate largely irrelevant by decoupling retrieval granularity from 
context window size.


##### Is Chunking Data Saved on Disk

Yes, but where and how it is saved depends on which stage of the RAG pipeline you are looking at.

In a professional RAG setup, chunked data is typically stored in two distinct ways to ensure the 
system is both fast (for searching) and accurate (for answering).

## 1. The Vector Database (The "Searchable" Storage)

    When you chunk your data, each chunk is transformed into a Vector Embedding 
    (that long list of numbers we discussed earlier). 
    These vectors are saved to disk within a Vector Database 
    (like Pinecone, Milvus, Chroma, or Weaviate).

    What is saved: The numerical vector and a "pointer" to the original text.

Purpose: This allows the system to perform high-speed mathematical comparisons to 
         find the most relevant chunks when a user asks a question.

## 2. The Document Store (The "Readable" Storage)

Most RAG systems also save the raw text chunks themselves to disk, 
often in a standard database (like PostgreSQL) or a simple key-value store.

What is saved: The actual human-readable text 
(e.g., "The capital of France is Paris").

Purpose: The Vector Database is great for finding the right chunk, but it's hard to "read" numbers. 
         Once the system identifies the correct vector, it goes to the Document Store to grab the 
         actual text to send to the LLM so it can write a response.

The Storage Workflow

    1. Ingestion: You chunk a PDF into 100 pieces.

    2. Processing: Each of those 100 pieces is turned into a vector.

    3. Saving:

        * The 100 Vectors go into the Vector Index on the disk.
        * The 100 Text Chunks go into the Blob/Text Storage on the disk.

    4. Retrieval: When you ask a question, the system finds the best Vector, 
       looks up the corresponding Text Chunk, and hands it to the AI.

Why do we save it to disk?

Persistence: You don't want to re-chunk and re-embed your entire library 
             every time you restart your application. Embedding costs money 
             (API fees) and time (compute).

Scalability: If you have millions of chunks, they won't fit in your computer's RAM (memory). 
             Saving to disk allows the system to handle massive amounts of data by only 
             pulling into memory what it needs at that moment.

Can you skip saving to disk?

Yes. If you are building a very small, temporary demo, you can use In-Memory storage 
(like a local FAISS index). 
However, as soon as you close your code or turn off your computer, 
all those chunks and vectors are deleted.


    # ============================================================================================== #
    # ======================================  CODE SECTION  ======================================= #
    # ============================================================================================== #


5 distinct chunking strategies. Here's a breakdown of each:

1. fixed_tokens — Fixed-Size Token Chunking
    
    Slides a window of fixed token count (default 512 tokens) across the text with an overlap 
    (default 64 tokens). 
    Uses tiktoken (cl100k_base) with a character-based fallback. 
    This is the classic approach — fast but sentence-boundary-unaware.

2. sentences — Structure-Aware Sentence Chunking

    Groups a fixed number of sentences (default 5) per chunk with sentence-level overlap 
    (default 1 sentence). 
    Uses a 3-tier fallback for sentence splitting: Stanza → NLTK → regex. 
    This is the variable-size, sentence-aware method we discussed.

3. document_specific — Format-Aware Structural Chunking

    Dispatches based on file type:
    
    Markdown / HTML → splits by headings
    PDF → splits by page
    JSON → splits by record/field
    Code (py/js/ts etc.) → splits by function/class blocks
    LaTeX → splits by \section / \subsection
    txt / docx → splits by paragraph


4. semantic — Percentile-Based Semantic Chunking

    The most sophisticated content-aware method. Embeds each sentence, computes cosine distances 
    between adjacent sentences, then splits at the top-percentile breakpoints (default 95th percentile). 
    Uses sentence-transformers with TF-IDF and char-frequency fallbacks. 
    This is the LlamaIndex default approach.

5. llm_contextual — LLM-Powered Contextual Retrieval

    First chunks using any base method (defaults to fixed_tokens), 
    then sends each chunk + the full document to Claude to generate a 2–3 sentence situating context. 
    That context gets prepended to the chunk at retrieval time via chunk.retrieval_text. 
    This is Anthropic's Contextual Retrieval pattern from 2024.




    # ──────────────────────────────────────────────────────────────────────────────────────────────
    # STRATEGY 1 — FIXED-SIZE TOKEN CHUNKING
    # ──────────────────────────────────────────────────────────────────────────────────────────────
#
# CONCEPT:
#   The simplest and most widely used baseline chunking strategy.
#   The document is tokenized using a model-aware tokenizer (tiktoken for OpenAI-family models),
#   then split into overlapping windows of fixed token width.
#
#   Each chunk covers exactly `chunk_size` tokens.
#   Adjacent chunks share `overlap` tokens to prevent context loss at boundaries.
#
#   Pros:  - Extremely fast; no NLP models needed
#          - Chunk sizes are exactly predictable (safe for context window management)
#          - Works on any text regardless of language or format
#
#   Cons:  - Blind cuts — will split mid-sentence or mid-thought
#          - Embeddings of partial sentences are less semantically accurate
#          - Overlap only partially compensates for boundary context loss
#
#   Best for:  Prototypes, cost-sensitive pipelines, structured/clean documents
#
# FALLBACK CHAIN:
#   tiktoken  →  character-based approximation (1 token ≈ 4 chars)
#
# DEFAULT PARAMETERS:
#   chunk_size = 512   (tokens)
#   overlap    = 64    (tokens)
#   encoding   = "cl100k_base"  (GPT-4 / text-embedding-ada-002 tokenizer)

def chunk_fixed_tokens(
    doc,
    chunk_size: int = 512,
    overlap:    int = 64,
    encoding:   str = "cl100k_base",
) -> list:
    '''
    Split the document into fixed-size token windows with optional overlap.

    Uses tiktoken when available; falls back to character-based splitting
    (1 token ≈ 4 chars) so the function always runs without optional deps.

    Parameters
    ----------
    chunk_size : Target tokens per chunk (default 512).
    overlap    : Token overlap between adjacent chunks (default 64).
    encoding   : tiktoken encoding name (default cl100k_base).
    '''
    text = doc.raw_text
    if not text.strip():
        return []

    # ── Tokenise ─────────────────────────────────────────────────────────────
    try:
        import tiktoken
        enc        = tiktoken.get_encoding(encoding)
        token_ids  = enc.encode(text)
        use_tiktoken = True
    except Exception:
        # Fallback: treat each 4 chars as one token (positional markers only)
        use_tiktoken = False
        token_ids    = list(range(0, len(text), 4))

    # ── Slide window ─────────────────────────────────────────────────────────
    chunks = []
    step   = max(1, chunk_size - overlap)          # step < chunk_size → overlap

    for start in range(0, len(token_ids), step):
        end        = start + chunk_size
        window_ids = token_ids[start:end]

        if use_tiktoken:
            import tiktoken
            enc        = tiktoken.get_encoding(encoding)
            chunk_text = enc.decode(window_ids)
        else:
            char_start = start * 4
            char_end   = min(char_start + chunk_size * 4, len(text))
            chunk_text = text[char_start:char_end]

        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        chunks.append({
            "text":        chunk_text,
            "index":       len(chunks),
            "token_start": start,
            "token_end":   min(end, len(token_ids)),
            "token_count": len(window_ids),
            "chunk_size":  chunk_size,
            "overlap":     overlap,
            "backend":     "tiktoken" if use_tiktoken else "char_approx",
        })

        if end >= len(token_ids):
            break

    return chunks


    # ──────────────────────────────────────────────────────────────────────────────────────────────
    # STRATEGY 2 — SENTENCE-BASED (STRUCTURE-AWARE) CHUNKING
    # ──────────────────────────────────────────────────────────────────────────────────────────────

# CONCEPT:
#   Instead of blind token windows, text is first split into individual sentences
#   using a trained NLP tokenizer, then sentences are grouped into chunks.
#
#   This produces variable-size chunks that always contain complete thoughts.
#   No sentence is ever split mid-way, which makes embeddings significantly more
#   semantically accurate than fixed-size chunking.
#
#   Pros:  - Every chunk is a grammatically complete unit
#          - Embeddings encode coherent meaning (better retrieval precision)
#          - Overlap at sentence level preserves cross-boundary context
#
#   Cons:  - Variable chunk sizes (harder to predict token budget)
#          - Requires an NLP library (Stanza / NLTK) as a dependency
#          - Long single sentences can still produce oversized chunks
#
#   Best for:  Q&A systems, legal/medical RAG, news article retrieval
#
# SENTENCE SPLITTER FALLBACK CHAIN:
#   Stanza (multilingual, most accurate)
#   → NLTK sent_tokenize (lightweight English)
#   → Regex on [.!?] (fast, fragile fallback)
#
# DEFAULT PARAMETERS:
#   sentences_per_chunk = 5   (sentences grouped per chunk)
#   overlap_sentences   = 1   (last N sentences repeated in next chunk)
#   lang                = "en"

def _split_sentences(text: str, lang: str = "en") -> list:
    '''Try Stanza → NLTK → regex sentence splitter (priority order).'''
    import re

    # ── Stanza (best for multilingual) ───────────────────────────────────────
    try:
        import stanza
        stanza.download(lang, verbose=False)
        nlp         = stanza.Pipeline(lang, processors="tokenize", verbose=False)
        doc_stanza  = nlp(text)
        return [s.text for s in doc_stanza.sentences]
    except Exception:
        pass

    # ── NLTK ─────────────────────────────────────────────────────────────────
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text, language="english")
    except Exception:
        pass

    # ── Regex fallback ───────────────────────────────────────────────────────
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(
    doc,
    sentences_per_chunk: int = 5,
    overlap_sentences:   int = 1,
    lang:                str = "en",
) -> list:
    '''
    Segment text by sentence boundaries.

    Parameters
    ----------
    sentences_per_chunk : Number of sentences grouped into one chunk.
    overlap_sentences   : Sentence overlap between adjacent chunks.
    lang                : Stanza language code (default "en").
    '''
    text = doc.raw_text
    if not text.strip():
        return []

    sentences = _split_sentences(text, lang)
    if not sentences:
        return []

    chunks = []
    step   = max(1, sentences_per_chunk - overlap_sentences)

    for start in range(0, len(sentences), step):
        window     = sentences[start : start + sentences_per_chunk]
        chunk_text = " ".join(s.strip() for s in window if s.strip())
        if not chunk_text:
            continue

        chunks.append({
            "text":            chunk_text,
            "index":           len(chunks),
            "sentence_start":  start,
            "sentence_end":    start + len(window),
            "sentence_count":  len(window),
            "sents_per_chunk": sentences_per_chunk,
            "overlap":         overlap_sentences,
            "lang":            lang,
        })

        if start + sentences_per_chunk >= len(sentences):
            break

    return chunks


    # ──────────────────────────────────────────────────────────────────────────────────────────────
    # STRATEGY 3 — DOCUMENT-SPECIFIC (FORMAT-AWARE) CHUNKING
    # ──────────────────────────────────────────────────────────────────────────────────────────────

# CONCEPT:
#   Exploits the structural metadata already present in formatted documents
#   (headings, pages, code blocks, LaTeX sections) to produce chunks that align
#   with the document's own logical divisions.
#
#   Each file format is dispatched to a dedicated splitter:
#
#     .md / .html  →  heading-bounded sections (h1, h2, h3)
#     .pdf         →  page-bounded segments (one chunk per page)
#     .json        →  record / field segments
#     .py / .js    →  function and class blocks (regex AST approximation)
#     .tex         →  \section / \subsection boundaries
#     .txt / .docx →  paragraph-bounded (blank-line separation)
#
#   Pros:  - Chunks align with human reading units (a section, a page, a function)
#          - No arbitrary size decisions — structure dictates boundaries
#          - Retrieval maps directly to meaningful document parts
#
#   Cons:  - Requires structured input; poorly formatted docs produce poor chunks
#          - Heading-based chunks can be very long (dense sections)
#          - Code regex splitting is heuristic; AST-based tools are more accurate
#
#   Best for:  Technical documentation, code corpora, PDFs with clear page breaks,
#             Markdown wikis, legal contracts with numbered clauses

import re as _re

def chunk_document_specific(doc) -> list:
    '''
    Dispatch to the appropriate format-aware splitter based on doc.file_type.
    '''
    ft        = (doc.file_type or "").lower().strip(".")
    structure = doc.structure

    if ft in ("md", "html", "htm"):
        return _chunk_by_headings(doc)
    elif ft == "pdf":
        return _chunk_pdf_pages(structure)
    elif ft == "json":
        return _chunk_json_records(structure)
    elif ft in ("py", "js", "ts", "jsx", "tsx", "rb", "go", "java", "c", "cpp"):
        return _chunk_code_blocks(doc.raw_text, ft)
    elif ft in ("tex", "latex"):
        return _chunk_latex(doc.raw_text)
    else:
        return _chunk_paragraphs(structure, doc.raw_text)   # txt / docx / unknown


def _chunk_by_headings(doc) -> list:
    '''Group content between Markdown / HTML headings into one chunk each.'''
    chunks          = []
    current_heading = ""
    current_level   = 0
    current_parts   = []

    def _flush(heading, level, parts):
        text = "\n\n".join(p for p in parts if p.strip())
        if text.strip():
            chunks.append({
                "text":      text,
                "index":     len(chunks),
                "heading":   heading,
                "level":     level,
                "format":    doc.file_type,
                "seg_count": len(parts),
            })

    for seg in doc.structure:
        if seg.get("type") == "heading":
            _flush(current_heading, current_level, current_parts)
            current_heading = seg["text"]
            current_level   = seg.get("level", 1)
            current_parts   = [seg["text"]]
        else:
            current_parts.append(seg.get("text", ""))

    _flush(current_heading, current_level, current_parts)
    return chunks


def _chunk_pdf_pages(structure: list) -> list:
    '''One chunk per PDF page.'''
    chunks = []
    for seg in structure:
        if seg.get("type") == "page":
            text = seg.get("text", "").strip()
            if text:
                chunks.append({
                    "text":       text,
                    "index":      len(chunks),
                    "page":       seg.get("page"),
                    "char_count": seg.get("char_count", len(text)),
                })
    return chunks


def _chunk_json_records(structure: list) -> list:
    '''One chunk per JSON record / field.'''
    chunks = []
    for seg in structure:
        text = seg.get("text", "").strip()
        if text:
            meta = {k: v for k, v in seg.items() if k != "text"}
            chunks.append({"text": text, "index": len(chunks), **meta})
    return chunks


def _chunk_paragraphs(structure: list, raw_text: str) -> list:
    '''Paragraph-bounded fallback for txt / docx.'''
    chunks   = []
    segments = [s for s in structure if s.get("text", "").strip()]

    if not segments:
        for para in _re.split(r"\n\n+", raw_text):
            para = para.strip()
            if para:
                chunks.append({"text": para, "index": len(chunks), "type": "paragraph"})
        return chunks

    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            meta = {k: v for k, v in seg.items() if k != "text"}
            chunks.append({"text": text, "index": len(chunks), **meta})
    return chunks


def _chunk_code_blocks(raw_text: str, lang: str) -> list:
    '''Split source code on function / class boundaries (regex heuristic).'''
    if lang == "py":
        pattern = r"(?=^(?:def |class |async def ))"
        flag    = _re.MULTILINE
    elif lang in ("js", "ts", "jsx", "tsx"):
        pattern = r"(?=^(?:function |const |class |export ))"
        flag    = _re.MULTILINE
    else:
        pattern = r"\n{2,}"
        flag    = 0

    blocks = _re.split(pattern, raw_text, flags=flag)
    chunks = []
    for block in blocks:
        block = block.strip()
        if block:
            chunks.append({"text": block, "index": len(chunks),
                           "language": lang, "type": "code_block"})
    return chunks


def _chunk_latex(raw_text: str) -> list:
    '''Split LaTeX at \\section / \\subsection boundaries.'''
    section_re = _re.compile(r"(\\(?:sub)*section\*?\{[^}]*\})", _re.MULTILINE)
    parts      = section_re.split(raw_text)
    chunks     = []
    i          = 0

    while i < len(parts):
        header = parts[i].strip() if section_re.match(parts[i]) else ""
        body   = parts[i] if not header else (parts[i + 1] if i + 1 < len(parts) else "")
        text   = (f"{header}\n{body}").strip() if header else body.strip()
        if text:
            chunks.append({"text": text, "index": len(chunks),
                           "heading": header, "format": "latex"})
        i += 2 if header else 1

    return chunks or [{"text": raw_text.strip(), "index": 0, "format": "latex"}]


    # ──────────────────────────────────────────────────────────────────────────────────────────────
    # STRATEGY 4 — SEMANTIC (PERCENTILE-BASED COSINE-DISTANCE) CHUNKING
    # ──────────────────────────────────────────────────────────────────────────────────────────────

# CONCEPT:
#   The most content-aware splitting method that does not require an LLM.
#
#   Every sentence is embedded into a dense vector. The cosine distance between
#   each pair of adjacent sentence vectors is measured. A large distance signals
#   a topic shift. Sentences are grouped together as long as the topic stays
#   consistent; a new chunk is started when the distance exceeds a threshold.
#
#   The threshold is defined as the Nth percentile of all observed distances,
#   making it adaptive to each document's natural topic structure rather than
#   relying on a hard-coded constant.
#
#   Pipeline:
#     1. Split document → sentences          (via _split_sentences)
#     2. Embed all sentences                 (sentence-transformers or TF-IDF fallback)
#     3. Compute cosine distance(i, i+1)     for every adjacent pair
#     4. Compute threshold = percentile(distances, breakpoint_percentile)
#     5. At every index where distance >= threshold → start a new chunk
#     6. Group sentences within boundaries → output variable-size chunks
#
#   Pros:  - Chunks map to actual topic shifts in the document
#          - No configuration of chunk size needed — the data drives splits
#          - Better retrieval coherence than any fixed-size method
#
#   Cons:  - Requires embedding at index time (slower + memory intensive)
#          - Quality depends heavily on the embedding model
#          - Hyperparameter (percentile) still needs tuning per corpus
#
#   Best for:  Long mixed-topic documents, research papers, news corpora
#
# EMBEDDING FALLBACK CHAIN:
#   sentence-transformers (all-MiniLM-L6-v2)
#   → sklearn TF-IDF cosine (no GPU needed)
#   → char-frequency histogram (zero-dependency ultimate fallback)
#
# DEFAULT PARAMETERS:
#   breakpoint_percentile = 95   (top 5% of distances become split points)
#   embed_model           = "all-MiniLM-L6-v2"
#   min_chunk_size        = 100  (characters; prevents tiny orphan chunks)

def _embed_sentences(sentences: list, model_name: str):
    '''Return a 2-D numpy array of sentence embeddings.'''
    import numpy as np

    # ── sentence-transformers (preferred) ────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(sentences, show_progress_bar=False)
    except Exception:
        pass

    # ── TF-IDF cosine fallback ───────────────────────────────────────────────
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=4096)
        mat = vec.fit_transform(sentences).toarray()
        return mat.astype(np.float32)
    except Exception:
        pass

    # ── Char-frequency ultra-fallback ────────────────────────────────────────
    dim = 256
    out = np.zeros((len(sentences), dim), dtype=np.float32)
    for i, s in enumerate(sentences):
        for ch in s:
            out[i, ord(ch) % dim] += 1.0
    return out


def chunk_semantic(
    doc,
    breakpoint_percentile: int = 95,
    embed_model:           str = "all-MiniLM-L6-v2",
    min_chunk_size:        int = 100,
) -> list:
    '''
    Detect semantic topic shifts via percentile cosine-distance splitting.

    Parameters
    ----------
    breakpoint_percentile : Distance percentile above which a new chunk starts.
    embed_model           : sentence-transformers model name.
    min_chunk_size        : Minimum characters to keep a chunk (filters slivers).
    '''
    import numpy as np

    text = doc.raw_text
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return [{"text": text.strip(), "index": 0,
                 "method": "semantic", "reason": "too_short"}]

    # ── Embed + compute consecutive cosine distances ──────────────────────────
    embeddings = _embed_sentences(sentences, embed_model)
    distances  = []

    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        cos_sim = float(np.dot(a, b) / norm) if norm > 0 else 0.0
        distances.append(1.0 - cos_sim)              # distance = 1 − similarity

    # ── Split at percentile breakpoints ──────────────────────────────────────
    threshold = float(np.percentile(distances, breakpoint_percentile))
    chunks    = []
    current   = [sentences[0]]

    for idx, dist in enumerate(distances):
        if dist >= threshold:
            text_block = " ".join(current).strip()
            if len(text_block) >= min_chunk_size:
                chunks.append({
                    "text":            text_block,
                    "index":           len(chunks),
                    "split_distance":  round(dist, 4),
                    "threshold":       round(threshold, 4),
                    "percentile":      breakpoint_percentile,
                    "sentence_count":  len(current),
                })
            current = []
        current.append(sentences[idx + 1])

    # ── Flush final segment ───────────────────────────────────────────────────
    if current:
        text_block = " ".join(current).strip()
        if text_block:
            chunks.append({
                "text":           text_block,
                "index":          len(chunks),
                "split_distance": 0.0,
                "threshold":      round(threshold, 4),
                "percentile":     breakpoint_percentile,
                "sentence_count": len(current),
            })

    return chunks or [{"text": text.strip(), "index": 0, "method": "semantic"}]


    # ──────────────────────────────────────────────────────────────────────────────────────────────
    # STRATEGY 5 — LLM-POWERED CONTEXTUAL RETRIEVAL (ANTHROPIC 2024)
    # ──────────────────────────────────────────────────────────────────────────────────────────────

# CONCEPT:
#   This is NOT a splitting strategy — it is a chunk enrichment strategy.
#   The document is first split using any base method (default: fixed_tokens).
#   Then, for each chunk, the full document and the chunk are sent to Claude,
#   which generates a short (2–3 sentence) situating context explaining:
#     - Where in the document this chunk comes from
#     - What broader topic it belongs to
#     - Any cross-references needed to understand it in isolation
#
#   This context is prepended to the chunk before embedding, so the vector
#   encodes not just the chunk's local content but also its document-level position.
#
#   Why this helps:
#     Without context:  "The revenue increased by 12%."
#                       → What product? What quarter? Ambiguous embedding.
#
#     With context:     "This chunk is from the Q3 FY2024 earnings report,
#                        in the section covering North American product lines.
#                        The revenue increased by 12%."
#                       → Embedding is now precise and retrievable.
#
#   Pros:  - Dramatically improves retrieval accuracy on long documents
#          - Works regardless of base chunking strategy
#          - Especially effective on dense technical / financial / legal docs
#
#   Cons:  - One LLM API call per chunk → expensive at scale
#          - Latency at index time is high (not suitable for real-time ingestion)
#          - Quality depends on base chunking quality
#
#   Best for:  Production RAG over long documents where retrieval precision is critical
#
# REFERENCE:
#   Anthropic blog: "Contextual Retrieval" (2024)
#   https://www.anthropic.com/news/contextual-retrieval
#
# DEFAULT PARAMETERS:
#   base_method         = "fixed_tokens"
#   chunk_size          = 512
#   overlap             = 64
#   model               = "claude-haiku-4-5-20251001"   (fast + cheap for context gen)
#   max_context_tokens  = 256

def chunk_llm_contextual(
    doc,
    base_method:        str      = "fixed_tokens",
    chunk_size:         int      = 512,
    overlap:            int      = 64,
    model:              str      = "claude-haiku-4-5-20251001",
    max_context_tokens: int      = 256,
    api_key:            str|None = None,
) -> list:
    '''
    Anthropic Contextual Retrieval (2024):

    1. Chunk the document with base_method.
    2. For each chunk, call Claude with the full document + chunk to generate
       a short situating context (max_context_tokens).
    3. Prepend the context to the chunk text before embedding.
       Retrieval uses context + chunk together; the raw chunk text is preserved separately.

    Parameters
    ----------
    base_method         : Chunking strategy for the initial split.
    chunk_size          : Token size used by fixed_tokens base method.
    overlap             : Overlap used by fixed_tokens base method.
    model               : Claude model for context generation.
    max_context_tokens  : Maximum tokens for the generated context snippet.
    api_key             : Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
    '''
    import os
    import anthropic

    # ── Step 1: Base chunking ─────────────────────────────────────────────────
    if base_method == "fixed_tokens":
        base_chunks = chunk_fixed_tokens(doc, chunk_size=chunk_size, overlap=overlap)
    elif base_method == "sentences":
        base_chunks = chunk_sentences(doc)
    elif base_method == "document_specific":
        base_chunks = chunk_document_specific(doc)
    elif base_method == "semantic":
        base_chunks = chunk_semantic(doc)
    else:
        raise ValueError(f"Unknown base_method: {base_method!r}")

    if not base_chunks:
        return []

    client           = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
    full_doc_preview = doc.raw_text[:8000]         # stay well within context window
    contextualised   = []

    # ── Step 2: Generate context for each chunk ───────────────────────────────
    for base_chunk in base_chunks:
        system_prompt = (
            "You are a RAG pre-processing assistant. "
            "Given a document excerpt and a specific chunk from that document, "
            "write a concise 2-3 sentence situating context that explains "
            "where this chunk fits in the overall document. "
            "This context will be prepended to the chunk for retrieval. "
            "Reply with ONLY the context — no preamble, no labels."
        )
        user_prompt = (
            f"<document>\n{full_doc_preview}\n</document>\n\n"
            f"<chunk>\n{base_chunk['text']}\n</chunk>\n\n"
            f"Situating context (max {max_context_tokens} tokens):"
        )

        try:
            response     = client.messages.create(
                model      = model,
                max_tokens = max_context_tokens,
                system     = system_prompt,
                messages   = [{"role": "user", "content": user_prompt}],
            )
            context_text = response.content[0].text.strip()
        except Exception as exc:
            context_text = ""    # degrade gracefully — chunk still stored without context

        # ── Step 3: Prepend context to chunk ─────────────────────────────────
        retrieval_text = f"{context_text}\n\n{base_chunk['text']}" if context_text else base_chunk["text"]

        contextualised.append({
            "text":           base_chunk["text"],        # raw chunk (for display)
            "retrieval_text": retrieval_text,            # context + chunk (for embedding)
            "context":        context_text,              # LLM-generated prefix
            "index":          len(contextualised),
            "llm_model":      model,
            "base_method":    base_method,
            **{k: v for k, v in base_chunk.items()
               if k not in ("text", "index")},           # carry forward base metadata
        })

    return contextualised


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
        "visual_html":  None, # VISUAL_HTML,
        "operations":   OPERATIONS,
    }