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

##### Type of RAGS
        
    ├── 📦 naive_rag                            
    ├── 📦 advanced_rag                
    ├── 📦 graph_rag                   
    ├── 📦 modular_rag                 
    ├── 📦 agentic_rag                 
    ├── 📦 corrective_rag              
    ├── 📦 self_rag                    
    ├── 📦 hybrid_rag                  
    ├── 📦 hyde_rag                    
    ├── 📦 multimodal_rag              
    ├── 📦 long_context_rag            
    ├── 📦 federated_rag                


### Naive RAG Pipeline 

    ╔════════════════════════════════════════════════════════════════╗
    ║                     NAIVE RAG PIPELINE                         ║
    ╚════════════════════════════════════════════════════════════════╝
    
     ┌─────────────────────────────────────────────────────────────┐
     │                   PHASE 1: INDEXING                         │
     └─────────────────────────────────────────────────────────────┘
    
      [Raw Documents]
      .txt /.pdf / .docx / .html /.md / .json /
            │
            ▼
      ┌───────────┐
      │  LOADING  │  ← Load files from disk, URLs, databases, etc.
      └───────────┘
            │
            ▼
      ┌───────────┐     chunk_size=512 tokens
      │ CHUNKING  │  ← Split docs into smaller passages
      └───────────┘     overlap=50 tokens
            │
            ▼
      ┌───────────────┐
      │  EMBEDDING    │  ← Convert each chunk to a dense vector
      │  text → [🔢]  │    e.g. text-embedding-3-small, BGE, etc.
      └───────────────┘
            │
            ▼
      ┌───────────────┐
      │  VECTOR STORE │  ← Persist vectors + metadata
      │  [🔢][🔢][🔢] │    e.g. FAISS, Pinecone, Chroma, Weaviate
      └───────────────┘
    
    
     ┌─────────────────────────────────────────────────────────────┐
     │                   PHASE 2: RETRIEVAL                        │
     └─────────────────────────────────────────────────────────────┘
    
      [User Query]
      "What is X?"
            │
            ▼
      ┌───────────────┐
      │  EMBEDDING    │  ← Embed the query with the SAME model
      │  text → [🔢]  │
      └───────────────┘
            │
            ▼
      ┌──────────────────┐
      │  SIMILARITY      │  ← Cosine similarity / ANN search
      │  SEARCH          │    Returns top-k most relevant chunks
      │  (top-k=3~5)     │
      └──────────────────┘
            │
            ▼
      [Retrieved Chunks]
      chunk_1, chunk_2, chunk_3
    
    
     ┌─────────────────────────────────────────────────────────────┐
     │                   PHASE 3: GENERATION                       │
     └─────────────────────────────────────────────────────────────┘
    
      [User Query] + [Retrieved Chunks]
            │
            ▼
      ┌───────────────────────────────┐
      │       PROMPT ASSEMBLY         │
      │                               │
      │  "Answer using this context:  │
      │   <chunk_1>                   │
      │   <chunk_2>                   │
      │   <chunk_3>                   │
      │   Question: {user_query}"     │
      └───────────────────────────────┘
            │
            ▼
      ┌───────────┐
      │    LLM    │  ← GPT-4o, Claude, Llama, Mistral, etc.
      └───────────┘
            │
            ▼
      [Final Answer]
      "Based on the documents, X is..."
    
    
    ══════════════════════════════════════════════════════════════════
     FULL FLOW SUMMARY
    ══════════════════════════════════════════════════════════════════
    
      Docs ──► Chunk ──► Embed ──► Vector DB
                                       │
      Query ──► Embed ──► Search ──────┘
                             │
                         Top-K Chunks
                             │
                       Prompt + Query
                             │
                            LLM
                             │
                          Answer
 


"""

# ── Visual HTML ─────────────────────────────────────────────────────────────
# Return a full HTML string. It will be rendered in an iframe.
# Import from Required_Images/ or write inline.
VISUAL_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  body { background: #0d0d0d; color: #e0e0e0; font-family: monospace;
         display: flex; justify-content: center; align-items: center;
         height: 100vh; margin: 0; }
  .placeholder { text-align: center; opacity: 0.5; }
  .placeholder h2 { font-size: 2rem; }
</style>
</head>
<body>
<div class="placeholder">
  <h2>📐 Visual Breakdown</h2>
  <p>HTML diagram coming soon for this topic.</p>
</div>
</body>
</html>
"""

# ── Step-by-Step Operations ─────────────────────────────────────────────────
OPERATIONS = {
    "Step 1: Example": {
        "description": "Brief description of what this step demonstrates.",
        "language": "python",
        "code": """
# Example code — runnable in subprocess
print("Hello from Step 1!")
""".strip(),
    },
    "Step 2: Example": {
        "description": "Next step description.",
        "language": "python",
        "code": """
print("Hello from Step 2!")
""".strip(),
    },
}

# ── Entry point called by topics/__init__.py ────────────────────────────────
def get_topic_data() -> dict:
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  VISUAL_HTML,
        "operations":   OPERATIONS,
    }
