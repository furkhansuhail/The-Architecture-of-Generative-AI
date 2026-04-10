"""
Tutorial Template — Automation & Infrastructure
=================================================
Copy this file, rename it, and fill in the sections.

The TOPIC_NAME and CATEGORY are parsed by the app.
OPERATIONS entries can specify "language" as "bash", "yaml", "python", "json", etc.
"""

TOPIC_NAME = "LLM - Memory Architecture"
DISPLAY_NAME = "LLM - Memory Architecture"
ICON         = "🧠"
SUBTITLE     = "The four memory systems of an LLM — what persists, what fades, and what never existed"
CATEGORY = "LLM Architecture"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = r"""

Order:

    Module_Concepts · Memory Architecture
          ├── four_types_of_llm_memory    In-context / external / in-weights / in-cache
          ├── in_context_memory           What lives in the window — the only "live" memory
          ├── in_weights_memory           Knowledge baked in at training — frozen at inference
          ├── external_memory_stores      Databases, vector stores, files
          ├── kv_cache                    Key-Value cache — avoiding full re-reads each turn
          ├── kv_cache_invalidation       What clears the cache and why it matters for cost
          ├── episodic_vs_semantic_memory Event-specific recall vs. general world knowledge
          └── working_memory_analogy      Context window as cognitive working memory


## 1. FOUR TYPES OF LLM MEMORY — The Complete Taxonomy

When people say an LLM "remembers" something, they almost always mean something
quite different from human memory — and often mean four completely different
things without realising it.

An LLM has four distinct memory systems. Each operates by a different mechanism,
persists across a different time horizon, has a different capacity, and costs a
different amount to access. Understanding all four — and crucially how they
interact — is the foundation of reasoning correctly about what any LLM-based
system can and cannot do.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE FOUR MEMORY TYPES — OVERVIEW                                ║
    ╚══════════════════════════════════════════════════════════════════╝

      TYPE 1 — IN-CONTEXT MEMORY
      ────────────────────────────
      The tokens currently in the active context window.
      This is the model's "live" working memory — the only information
      the model can directly attend to and reason about right now.

      TYPE 2 — IN-WEIGHTS MEMORY
      ────────────────────────────
      Knowledge encoded into the model's parameters (weights) during
      training. Facts, language patterns, reasoning procedures.
      Frozen at inference — the model cannot update it.

      TYPE 3 — EXTERNAL MEMORY
      ────────────────────────────
      Information stored outside the model: databases, vector stores,
      files, APIs. Retrieved and injected into the context window
      when needed. Unlimited capacity; requires a retrieval step.

      TYPE 4 — IN-CACHE MEMORY (KV Cache)
      ────────────────────────────────────
      Pre-computed Key-Value attention matrices stored in GPU VRAM,
      avoiding re-processing of already-seen tokens. An implementation
      detail that dramatically affects inference speed and cost.


## 1.1  Properties Comparison — The Complete Matrix

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FOUR MEMORY TYPES — DETAILED PROPERTY COMPARISON               ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌────────────────────┬───────────────┬───────────────┬───────────────┬───────────────┐
      │  Property          │  In-Context   │  In-Weights   │  External     │  KV Cache     │
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  LOCATION          │  Context      │  Model file   │  DB / disk    │  GPU VRAM     │
      │                    │  window       │  (frozen)     │  (external)   │  (ephemeral)  │
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  CAPACITY          │  1k–2M tokens │  Billions of  │  Unlimited    │  Varies;      │
      │                    │  (hard limit) │  parameters   │  (TB scale)   │  ~0.5MB/token │
      │                    │               │  (~7B–405B)   │               │  per 7B model │
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  ACCESS SPEED      │  ~ns          │  ~ns          │  ~ms–s        │  ~ns          │
      │                    │  (VRAM read)  │  (VRAM read)  │  (network)    │  (VRAM read)  │
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  PERSISTENCE       │  One session  │  Permanent    │  Permanent    │  One session  │
      │                    │  only         │  (read-only)  │  (writable)   │  (ephemeral)  │
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  MUTABILITY        │  Grows during │  NOT mutable  │  Fully        │  Grows during │
      │                    │  session      │  at inference │  mutable      │  session      │
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  COST TO READ      │  Attention    │  Forward pass │  Retrieval    │  Near-zero    │
      │                    │  computation  │  computation  │  + embedding  │  (reuse)      │
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  COST TO WRITE     │  Input token  │  Training     │  Write to DB  │  Free         │
      │                    │  cost         │  (very large) │  (cheap)      │  (auto)       │
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  EXACT RECALL?     │  YES          │  NO           │  YES          │  N/A          │
      │                    │  (verbatim)   │  (fuzzy)      │  (verbatim)   │  (internal)   │
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  SHARED ACROSS     │  NO           │  YES          │  YES          │  NO           │
      │  USERS?            │  (per session)│  (same model) │  (if designed)│  (per session)│
      ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
      │  SURVIVES RESTART? │  NO           │  YES          │  YES          │  NO           │
      └────────────────────┴───────────────┴───────────────┴───────────────┴───────────────┘


## 1.2  The Memory Access Flow for One Inference Call

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW ALL FOUR MEMORY TYPES INTERACT ON A SINGLE API CALL        ║
    ╚══════════════════════════════════════════════════════════════════╝

      USER QUERY: "What are the payment terms in the NDA I uploaded last week?"

      STEP 1 — External memory is consulted (before the API call)
        Application embeds the query → searches vector DB →
        retrieves the 3 most relevant NDA chunks → injects into prompt.

      STEP 2 — In-context memory is assembled (the full prompt)
        [System prompt]
        [Retrieved NDA chunks — from external memory]
        [Conversation history from this session — in-context]
        [User question — in-context]
        → This is the context window. Everything the model "sees."

      STEP 3 — In-weights memory is activated (during the forward pass)
        The model uses its trained knowledge to:
        • Understand what "payment terms" means in legal contexts
        • Understand NDA document structure
        • Parse the retrieved chunks using language comprehension
        → The model did not need to "look up" these skills — they are in weights.

      STEP 4 — KV cache is used/populated (during generation)
        If this is turn 3 of a conversation:
        • Turns 1–2 are already in the KV cache → no recomputation
        • New tokens (retrieved NDA + current question) are computed fresh
        • New K/V pairs are added to the KV cache for future turns

      STEP 5 — Response generated
        The model synthesises: in-weights legal knowledge + retrieved NDA
        text (now in-context) → produces an answer about payment terms.

      WHAT EACH MEMORY TYPE CONTRIBUTED:
        In-weights:  Understanding of legal language, NDA conventions
        In-context:  The actual NDA text (from external memory → injected)
        External:    Storage and retrieval of the original NDA document
        KV cache:    Efficient continuation of the multi-turn conversation


## 1.3  The Critical Distinction: Reading vs. Retrieving

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE MOST COMMON MISCONCEPTION ABOUT LLM MEMORY                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      People often assume the model is "searching its memory" when it
      answers a question — as if it were doing a database lookup.

      WHAT ACTUALLY HAPPENS:

        In-weights knowledge is NOT retrieved. It is activated.

        When the model sees the word "Paris" followed by a context
        suggesting a geography question, the weight matrices mathematically
        transform the token representations such that the output
        distribution assigns high probability to "capital" and "France."

        There is no index. No search. No retrieval from a memory bank.
        The "knowledge" is distributed across millions of weight values.
        It cannot be inspected, listed, or individually addressed.

        This is why:
          • Models cannot tell you "what they know" — knowledge is implicit
          • Models confabulate (hallucinate) — probability distributions can
            produce plausible-sounding but incorrect outputs
          • Models cannot update in-weights knowledge without retraining
          • Two identical prompts always produce outputs from the same weights

      IN-CONTEXT memory IS explicitly retrieved:
        The model attends to every token in the context window.
        If the contract says "payment due in 30 days," the model will
        find those exact tokens when asked about payment terms.
        This is why injecting information into context is reliable;
        relying on in-weights knowledge for specific facts is not.

**KEY CONCEPT:** The difference between in-context and in-weights memory
is the difference between reading from a document in front of you (exact,
reliable, citable) and answering from memory (approximately right, fallible,
non-citable). For any fact that matters — use in-context memory.


---


## 2. IN-CONTEXT MEMORY — What Lives in the Window

In-context memory is the model's active, live working memory. It is the
complete set of tokens in the current context window — and it is the ONLY
information the model can directly attend to and reason about in this call.


## 2.1  What In-Context Memory Physically Is

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE PHYSICAL REPRESENTATION OF IN-CONTEXT MEMORY               ║
    ╚══════════════════════════════════════════════════════════════════╝

      In-context memory is a 2D matrix of shape (N × d_model):

        N       = number of tokens currently in the context window
        d_model = embedding dimension (e.g. 4096 for a 7B model)

      Each ROW is the current representation of one token — a dense
      vector of d_model floating-point numbers that encodes that token's
      identity, its semantic role in context, and every layer of meaning
      added by the transformer layers it has passed through.

      After processing all N tokens through all 32 (or 48, or 80) layers:
        • The first row represents the fully-contextualised meaning of token 0
        • The last row represents the fully-contextualised meaning of token N-1
        • Each row has "read" all other rows via the attention mechanism

      This matrix exists in GPU VRAM during the forward pass.
      Once the session ends, it is discarded.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW IN-CONTEXT MEMORY IS READ — THE ATTENTION MECHANISM        ║
    ╚══════════════════════════════════════════════════════════════════╝

      Reading from in-context memory is not sequential (like reading a
      file) — it is parallel and attention-weighted.

      For each token at position i, attention computes:
        1. A QUERY: "What information do I need?" (based on token i's state)
        2. KEYS for all j ≤ i: "What does each prior token offer?"
        3. SCORES: Q_i · K_j / √d_k  for all j (how well each j matches)
        4. Softmax: normalise scores to a probability distribution
        5. WEIGHTED SUM of VALUES: the "memory read" result

      The result: each token simultaneously "reads" all other tokens,
      weighted by relevance. High-relevance tokens are read more strongly;
      low-relevance tokens are de-emphasised but not ignored.

      IMPLICATION FOR MEMORY:
        All in-context tokens are potentially accessible at every step.
        But attention weighting means tokens at the beginning or end of
        the window are attended to more strongly than middle tokens.
        (The "lost in the middle" effect — see Context Window module.)


## 2.2  What Can and Cannot Live in In-Context Memory

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CONTENTS OF IN-CONTEXT MEMORY IN A PRODUCTION SYSTEM            ║
    ╚══════════════════════════════════════════════════════════════════╝

      CAN LIVE IN IN-CONTEXT MEMORY:

        ✓  System prompt — instructions, persona, constraints
        ✓  Conversation history — all prior turns in this session
        ✓  Retrieved documents — injected from external storage
        ✓  User-uploaded files — text extracted and tokenised
        ✓  Code to be reviewed — the actual source file text
        ✓  Database results — query results serialised to text
        ✓  Tool outputs — results of function calls
        ✓  Images (multimodal models) — encoded as token-like embeddings
        ✓  Structured data — JSON, CSV, tables as text
        ✓  The current user message — the trigger for this call

      CANNOT LIVE IN IN-CONTEXT MEMORY:

        ✗  Information from previous sessions (cleared on session end)
        ✗  Information that exceeds the token limit
        ✗  Binary data (audio, non-image binary files) — must be described
        ✗  Real-time data without explicit injection (stock prices, weather)
        ✗  Information not tokenisable (true video, complex diagrams)
        ✗  Private user data from other users' sessions

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WRITING TO IN-CONTEXT MEMORY                                    ║
    ╚══════════════════════════════════════════════════════════════════╝

      In-context memory grows in exactly two ways:

      1. DIRECT INJECTION (before the API call):
         The application prepends content to the prompt.
         Every token prepended is an input token — billed at input price.
         The model sees this content as if it had always been there.

         Application adds: retrieved document, conversation history,
         tool outputs, uploaded file text.

      2. MODEL GENERATION (during the API call):
         The model generates new tokens, which are appended to the context.
         Each generated token is immediately available for the next token.
         This is how "scratch-pad" reasoning and chain-of-thought work —
         the model literally writes into its own in-context memory.

         Model generates: reasoning steps, partial answers, intermediate
         computations that inform the final answer.

      CRITICAL: The model CANNOT inject information back into the context
      between turns. The application must explicitly append the model's
      last response to the conversation history before the next call.
      This is entirely an application-layer responsibility.


## 2.3  In-Context Memory Across a Multi-Turn Conversation

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW IN-CONTEXT MEMORY EVOLVES ACROSS TURNS                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      TURN 1:
        Context window: [SYS] [User: "My name is Alice."]
        Model sees: the system prompt + Alice's message.
        Model generates: "Hello Alice, how can I help?"
        Session ends. Nothing is retained in the model.

      BETWEEN TURNS (application layer):
        Application stores: [User: "My name is Alice."]
                            [Assistant: "Hello Alice, how can I help?"]

      TURN 2:
        Application assembles the new context window:
        [SYS] [User: "My name is Alice."] [Asst: "Hello Alice..."] [User: "What is my name?"]
        Model sees: everything, including the name.
        Model generates: "Your name is Alice."

      If the APPLICATION did not include turn 1 in turn 2's context:
        Context window: [SYS] [User: "What is my name?"]
        Model has no access to the name.
        Model generates: "I don't know your name."

      The model is not "remembering" — it is reading.
      Memory is simulated by the application layer re-injecting history.

      ┌─────────────────────────────────────────────────────────────────┐
      │  WHAT THE MODEL SEES vs. WHAT ACTUALLY PERSISTS                │
      │                                                                 │
      │  Turn 1:  [SYS][U1][A1]     ← full context for turn 1          │
      │           └─ discarded after generation                        │
      │                                                                 │
      │  Turn 2:  [SYS][U1][A1][U2] ← application re-injects [U1][A1] │
      │           └─ discarded after generation                        │
      │                                                                 │
      │  Turn 3:  [SYS][U1][A1][U2][A2][U3]  ← application re-injects │
      │           └─ discarded after generation                        │
      │                                                                 │
      │  The model is reading fresh tokens every turn.                 │
      │  The "memory" of prior turns is an application-layer illusion. │
      └─────────────────────────────────────────────────────────────────┘


## 2.4  The Limits of In-Context Memory

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THREE FUNDAMENTAL CONSTRAINTS                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      LIMIT 1 — CAPACITY (hard ceiling)
        The context window has a fixed maximum token count.
        Once full, new information can only enter by removing old information.
        This is non-negotiable: the architecture defines the limit.

      LIMIT 2 — ATTENTION QUALITY (soft degradation)
        Even within the limit, not all positions are attended to equally.
        Information buried in the middle of a very long context may be
        effectively "invisible" — present but insufficiently attended to.
        Effective capacity is less than nominal capacity for long contexts.

      LIMIT 3 — LINEARITY (no structure)
        In-context memory is a flat sequence of tokens.
        There is no hierarchy, no index, no tagging.
        Finding specific information requires the model to "scan"
        via attention across all positions — it cannot jump to a location.
        This is why 500 tokens of dense, well-organised text often produces
        better answers than 5,000 tokens of loosely organised context.


---


## 3. IN-WEIGHTS MEMORY — Knowledge Baked In at Training

In-weights memory is the knowledge encoded into the model's parameters during
training. Unlike in-context memory, it is not explicitly readable — it is
activated probabilistically by the input. It is permanent but frozen.


## 3.1  How Knowledge Gets Into Weights

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FROM TRAINING DATA TO WEIGHT VALUES — THE ENCODING PROCESS     ║
    ╚══════════════════════════════════════════════════════════════════╝

      During training, the model sees a sequence of tokens and learns
      to predict the next token. Over billions of examples, the weight
      values are adjusted to encode statistical patterns in the training data.

      SIMPLIFIED EXAMPLE — encoding the fact "Paris is the capital of France":

        Training sees: "...the city of Paris, capital of France..."
                       "...France, whose capital is Paris..."
                       "...Paris (French: Paris), the capital and..."
                       ...seen thousands of times across billions of tokens...

        Gradient descent adjusts weights such that:
          when the model sees [..., "capital", "of", "France"],
          the weight matrices make "Paris" the highest-probability next token.

          when the model sees [..., "Paris", "is", "the", "capital", "of"],
          the weight matrices make "France" the highest-probability next token.

        The "fact" is not stored as {"Paris": "capital of France"}.
        It is stored as millions of tiny weight adjustments that collectively
        bias the output distribution toward correct continuations.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE IMPLICIT KEY-VALUE STORE IN THE FFN LAYERS                  ║
    ╚══════════════════════════════════════════════════════════════════╝

      Research on mechanistic interpretability has shown that the Feed-Forward
      Network (FFN) sub-layers act as implicit key-value memory:

        KEYS:    the first-layer weight matrix W₁ encodes "what patterns activate me?"
        VALUES:  the second-layer weight matrix W₂ encodes "what should I contribute?"

        When a token vector h enters the FFN:
          1. h · W₁ computes similarity to all keys (which neurons activate?)
          2. Activation function (SwiGLU) gates the result
          3. The gated result · W₂ retrieves the associated values

        This is analogous to a soft, approximate lookup:
          "Capital of France" activates neurons associated with Paris
          "Python syntax for list" activates neurons associated with [ ] and append

        The analogy is imperfect — it is not a discrete lookup but a continuous,
        distributed activation — but it explains why the FFN is where factual
        associations primarily live, while attention handles relational reasoning.


## 3.2  What Kinds of Knowledge Live in Weights

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TAXONOMY OF IN-WEIGHTS KNOWLEDGE                                ║
    ╚══════════════════════════════════════════════════════════════════╝

      TYPE A — FACTUAL WORLD KNOWLEDGE
        Facts about the world learned from training data.
        "Paris is the capital of France."
        "Python was created by Guido van Rossum."
        "Water boils at 100°C at sea level."
        → Reliable for well-represented facts; unreliable for rare or recent facts.

      TYPE B — LANGUAGE KNOWLEDGE
        Grammar, syntax, morphology, pragmatics in dozens of languages.
        "The past tense of 'go' is 'went'."
        "In formal writing, avoid contractions."
        "Spanish adjectives agree with noun gender."
        → Extremely reliable; heavily represented in training data.

      TYPE C — REASONING PROCEDURES
        How to approach problems: logical deduction, mathematical operations,
        code debugging strategies, argument structure.
        "To solve a quadratic: identify a, b, c; compute discriminant..."
        → Reliable for common procedures; unreliable for novel complex reasoning.

      TYPE D — CULTURAL AND SOCIAL KNOWLEDGE
        Conventions, norms, common sense, social expectations.
        "Emails to managers should be more formal than to peers."
        "Red means stop; green means go."
        → Generally reliable; culturally biased toward training data distribution.

      TYPE E — STYLISTIC PATTERNS
        Writing styles, code patterns, domain conventions.
        "Python code uses snake_case for variables."
        "Academic papers cite sources in parentheses."
        → Very reliable; directly observable in training corpus statistics.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHAT IN-WEIGHTS MEMORY CANNOT DO                                ║
    ╚══════════════════════════════════════════════════════════════════╝

      1. EXACT RECALL OF TRAINING DATA
         In-weights knowledge is distributed and blurry — not exact.
         The model cannot reliably reproduce verbatim text from training.
         (Memorisation of very frequent sequences does occur, but unreliably.)

      2. KNOWLEDGE AFTER THE TRAINING CUTOFF
         Any event, document, or fact that postdates the training cutoff
         does not exist in the weights.
         "Who won the 2026 election?" → weights have no information.
         → Requires in-context or external memory.

      3. PRIVATE OR CUSTOM KNOWLEDGE
         Information not in the training corpus is not in the weights.
         "What does our internal API spec say?" → not in weights.
         → Requires in-context or external memory.

      4. CERTAINTY ABOUT WHAT IT KNOWS
         The model cannot reliably introspect on its own knowledge.
         "Are you sure this is correct?" — the model cannot check.
         High-confidence outputs can be wrong; low-confidence can be right.
         → In-weights knowledge should always be verified for high-stakes facts.


## 3.3  Fine-Tuning — Modifying In-Weights Memory

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FINE-TUNING AS WEIGHT MODIFICATION                              ║
    ╚══════════════════════════════════════════════════════════════════╝

      Fine-tuning is the process of continuing training on a smaller,
      task-specific dataset to shift the weight values toward desired behaviour.

      WHAT FINE-TUNING CAN DO:
        ✓  Teach the model a new writing style or response format
        ✓  Improve reliability on a specific narrow task
        ✓  Reduce hallucination on a specific domain (with good data)
        ✓  Change the model's "personality" or default tone
        ✓  Add a small set of new factual associations

      WHAT FINE-TUNING CANNOT DO:
        ✗  Add reliable arbitrary factual knowledge cheaply
           (requires large amounts of high-quality training data)
        ✗  Guarantee the model won't hallucinate on fine-tuned facts
        ✗  Remove specific knowledge (e.g. "forget" training data)
        ✗  Add real-time information (training cutoff still applies)
        ✗  Make weight memory exact or queryable

      FINE-TUNING vs. RAG — when to choose:

        ┌───────────────────────────────┬────────────────┬──────────────────┐
        │  Need                         │  Fine-tuning   │  RAG             │
        ├───────────────────────────────┼────────────────┼──────────────────┤
        │  Consistent output format     │  ✓ best        │  possible        │
        │  Domain-specific tone/style   │  ✓ best        │  partial         │
        │  Updated factual knowledge    │  ✗ expensive   │  ✓ best          │
        │  Private company documents    │  ✗ risky       │  ✓ best          │
        │  Exact fact recall needed     │  ✗ unreliable  │  ✓ best          │
        │  Low training data available  │  ✗ risky       │  ✓ workable      │
        │  Knowledge changes frequently │  ✗ not viable  │  ✓ best          │
        └───────────────────────────────┴────────────────┴──────────────────┘


## 3.4  The Hallucination Problem — When In-Weights Memory Fails

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY MODELS CONFABULATE — THE PROBABILISTIC ROOT CAUSE          ║
    ╚══════════════════════════════════════════════════════════════════╝

      Hallucination is not a bug — it is the expected consequence of how
      in-weights memory works.

      The model generates the STATISTICALLY MOST LIKELY continuation of the
      token sequence. It does not "check" whether the next token is true.

      SCENARIO: "The CEO of Acme Corp was appointed in _____"

        If "Acme Corp" is not in the training data:
          The model has no weight activations specific to Acme Corp.
          But the phrase "The CEO of [company] was appointed in ____"
          has a strong statistical pattern from thousands of similar sentences.
          The model will generate a year — perhaps "2019" or "2021" —
          because those are statistically likely completions of this pattern.
          There is no "I don't have information about this" trigger.
          The model does not know that it doesn't know.

        If "Acme Corp" had conflicting mentions in training data:
          (e.g. a fictional company that appeared in different articles
          with different fictional CEOs)
          The model may blend multiple inconsistent "facts."

      THE DISTRIBUTION COLLAPSE:
        Low-frequency facts are poorly encoded in weights.
        For common, frequently-reinforced facts: reliable.
        For obscure, rarely-seen facts: weight activations are weak.
        In the absence of strong factual activations, statistical pattern
        completion takes over — producing fluent but unreliable output.

      THE PRACTICAL IMPLICATION:
        Never rely on in-weights memory for:
          • Specific numbers (dates, prices, statistics, versions)
          • Specific names (people, organisations, products)
          • Rare or specialised facts
          • Anything that matters and could be verified

        Always ground specific claims in in-context memory
        (injected from a reliable source you control).


---


## 4. EXTERNAL MEMORY STORES — Databases, Vector Stores, Files

External memory is information stored outside the model — in systems the
application controls. It is the only memory type with unlimited capacity,
persistence across sessions, and the ability to be precisely updated.


## 4.1  The External Memory Ecosystem

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TAXONOMY OF EXTERNAL MEMORY STORES                              ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌──────────────────────────────────────────────────────────────────────────┐
      │  STORE TYPE          RETRIEVAL METHOD        BEST FOR                   │
      ├──────────────────────────────────────────────────────────────────────────┤
      │  Vector database     Semantic similarity     Long-form docs, past convos │
      │  (Pinecone, Weaviate  (cosine / dot product) Knowledge bases, research   │
      │   pgvector, Qdrant)   embedding search)       papers, unstructured text  │
      ├──────────────────────────────────────────────────────────────────────────┤
      │  Relational DB       SQL query               Structured records, facts   │
      │  (Postgres, MySQL)   Exact / range match     with known schema, CRM data │
      ├──────────────────────────────────────────────────────────────────────────┤
      │  Key-Value store     Exact key lookup        Session state, user prefs   │
      │  (Redis, DynamoDB)   O(1) retrieval          Conversation summaries      │
      ├──────────────────────────────────────────────────────────────────────────┤
      │  Document store      Full-text / field query Semi-structured docs        │
      │  (MongoDB, Elastic)  + embedding hybrid      Mixed content types         │
      ├──────────────────────────────────────────────────────────────────────────┤
      │  File system /       Filename / path         Large files (PDFs, code)    │
      │  object store        Content-addressed       Binary + text assets        │
      │  (S3, GCS, IPFS)     (hash-based)            Version history             │
      ├──────────────────────────────────────────────────────────────────────────┤
      │  Knowledge graph     Graph traversal         Entity relationships        │
      │  (Neo4j, Wikidata)   SPARQL / Cypher         Multi-hop reasoning         │
      ├──────────────────────────────────────────────────────────────────────────┤
      │  API / tool call     HTTP / function call    Real-time data              │
      │  (any live service)  On-demand               Stock prices, weather, live │
      │                                              database queries            │
      └──────────────────────────────────────────────────────────────────────────┘


## 4.2  Vector Databases — The Core of RAG Systems

Vector databases are the most important external memory type for LLM applications
because they bridge the gap between unstructured text and semantic retrieval.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW VECTOR DATABASES WORK                                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      INGESTION (one-time or periodic):

        Document:  "Our refund policy allows returns within 30 days of purchase."

        Step 1 — Chunk the document into segments of ~512 tokens.
        Step 2 — Embed each chunk using an embedding model.
                  embed("Our refund policy allows returns within 30 days...")
                  → [0.23, -0.11, 0.67, 0.09, ..., 0.44]  (1536-dimensional vector)
        Step 3 — Store the vector + original text + metadata in the vector DB.
                  { id: "doc_42_chunk_3",
                    vector: [0.23, -0.11, ...],
                    text: "Our refund policy...",
                    metadata: {source: "policy.pdf", section: "returns"} }

      RETRIEVAL (every query):

        Query: "Can I return a product I bought last month?"

        Step 1 — Embed the query.
                  embed("Can I return a product I bought last month?")
                  → [0.21, -0.09, 0.65, 0.11, ..., 0.42]

        Step 2 — Compute cosine similarity between query vector and all stored vectors.
                  similarity(query, doc_42_chunk_3) = 0.89  ← very similar
                  similarity(query, doc_17_chunk_1) = 0.34  ← unrelated
                  ...

        Step 3 — Return top-K highest-similarity chunks.
                  K=3: returns doc_42_chunk_3 (0.89), doc_42_chunk_4 (0.81), doc_11_chunk_2 (0.72)

        Step 4 — Inject retrieved text into the context window.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY COSINE SIMILARITY CAPTURES SEMANTIC MEANING                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      Embedding models (like text-embedding-3-large) map text to vectors
      such that semantically similar text maps to nearby vectors.

      "Can I return a product?"    → vector near [0.21, -0.09, ...]
      "What is your refund policy?" → vector near [0.20, -0.10, ...]
      "Cheese is made from milk."  → vector near [-0.51, 0.33, ...]

      The first two questions mean similar things → their vectors are close.
      The cheese fact is unrelated → its vector is far away.

      Cosine similarity measures the ANGLE between two vectors:
        cos(θ) = 1.0  →  identical direction → most similar
        cos(θ) = 0.0  →  perpendicular → unrelated
        cos(θ) = -1.0 →  opposite direction → semantically opposite

      The embedding model is trained so that semantic similarity ≈ vector proximity.
      This allows "return a product" to retrieve "refund policy" even though
      those exact words don't appear in the query.


## 4.3  The Full RAG Pipeline — External Memory in Action

    ╔══════════════════════════════════════════════════════════════════╗
    ║  RETRIEVAL-AUGMENTED GENERATION — END TO END                    ║
    ╚══════════════════════════════════════════════════════════════════╝

      OFFLINE (document ingestion — done once or periodically):

        ┌──────────────────────────────────────────────────────────────┐
        │  Raw documents (PDFs, web pages, database exports, ...)      │
        │          │                                                   │
        │          ▼                                                   │
        │  Text extraction (PDF parsing, HTML cleaning, etc.)          │
        │          │                                                   │
        │          ▼                                                   │
        │  Chunking (512-token chunks, 50-token overlap)               │
        │          │                                                   │
        │          ▼                                                   │
        │  Embedding (text-embedding-3-large or similar)               │
        │          │                                                   │
        │          ▼                                                   │
        │  Storage in vector DB (vectors + text + metadata)            │
        └──────────────────────────────────────────────────────────────┘

      ONLINE (query time — every user request):

        User query
            │
            ▼
        Query embedding
            │
            ▼
        Vector similarity search → top-K chunks retrieved
            │
            ▼
        (Optional) Re-ranking — score chunks against query more precisely
            │
            ▼
        (Optional) Compression — shorten long chunks, remove irrelevant parts
            │
            ▼
        Context assembly:
          [System prompt]
          [Retrieved chunks, in order of relevance]
          [Conversation history]
          [Current user query]
            │
            ▼
        LLM API call → response grounded in retrieved facts
            │
            ▼
        (Optional) Citation extraction — identify which chunks were used


## 4.4  Memory Management Patterns

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FOUR PATTERNS FOR MANAGING EXTERNAL MEMORY IN PRODUCTION        ║
    ╚══════════════════════════════════════════════════════════════════╝

      PATTERN 1 — READ-ONLY KNOWLEDGE BASE
        ──────────────────────────────────
        External memory is static (infrequently updated).
        Documents are ingested in batch; retrieval is read-only.
        Use for: product documentation, legal archives, research corpora.

      PATTERN 2 — EPISODIC CONVERSATION MEMORY
        ────────────────────────────────────────
        Each turn of each conversation is stored in the vector DB.
        Future turns retrieve the most relevant past exchanges.
        The model "remembers" past conversations by re-reading them.
        Use for: long-running assistants, customer support with history.

        Storage schema:
          { session_id, turn_number, role, content, embedding, timestamp }
        Retrieval: embed current query → find semantically similar past turns.

      PATTERN 3 — EXTRACTED MEMORY (ENTITY MEMORY)
        ──────────────────────────────────────────────
        After each turn, a background call extracts key facts.
        Facts stored in a structured store (key-value or relational).
        Future calls inject relevant extracted facts as a bullet list.

        Extraction prompt: "Extract any user preferences, names, constraints,
                            or decisions from this message as JSON."
        Storage: { user_id: { "name": "Alice", "language": "Python", ... } }
        Injection: "Known about this user: name=Alice, language=Python..."
        Use for: personalised assistants, long-term user relationship management.

      PATTERN 4 — HYBRID (vector + structured)
        ──────────────────────────────────────────
        Vector search for semantic/unstructured content (documents, past turns).
        Structured DB for exact/queryable facts (user data, pricing, inventory).
        Both retrieved and injected at query time.
        Use for: most sophisticated production applications.


---


## 5. KV CACHE — Avoiding Full Re-Reads Each Turn

The KV (Key-Value) cache is an in-memory store of pre-computed attention
matrices that eliminates redundant computation during generation. It is not
a memory the model reasons about — it is an engineering optimisation that
makes multi-turn conversations and long-context inference practical.


## 5.1  The Problem the KV Cache Solves

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WITHOUT KV CACHE — THE NAIVE APPROACH                          ║
    ╚══════════════════════════════════════════════════════════════════╝

      In auto-regressive generation, each new token is generated by:
        1. Running the FULL context through all 32 transformer layers
        2. Taking the output at the LAST position
        3. Sampling the next token

      Without any caching, for a conversation with N prior tokens:

        GENERATING TOKEN N+1:
          Compute attention for all N+1 tokens: O(N²) operations
          → Produces K and V matrices for all N+1 tokens

        GENERATING TOKEN N+2:
          Compute attention for all N+2 tokens: O((N+2)²) operations
          → Re-computes K and V for tokens 0 to N (identical to previous step!)
          → Also computes K and V for new token N+1

        GENERATING TOKEN N+3:
          Re-computes tokens 0 to N+1 AGAIN.

      At every decode step, ALL prior tokens are recomputed from scratch.
      For a 1,000-token context generating 100 tokens:
        Total recomputation: 1,000 + 1,001 + ... + 1,099 ≈ 105,000 token-computations
        Necessary computation: 1,000 (original) + 100 (new tokens) = 1,100
        Wasted computation: ~104,000 / 105,000 = 99% waste

      The KV cache eliminates this waste.


## 5.2  What the KV Cache Stores and How It Works

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE KV CACHE — WHAT IS STORED                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      During the forward pass, each transformer layer computes
      Key (K) and Value (V) matrices for every token.
      These K and V vectors are stored in GPU VRAM — one set per
      layer, per token — in the KV cache.

      KV cache structure:
        For each layer l = 0, 1, ..., n_layers-1:
          K[l] = matrix of shape (n_tokens × n_heads × d_head)
          V[l] = matrix of shape (n_tokens × n_heads × d_head)

      For LLaMA 3 8B (32 layers, 8 KV heads via GQA, d_head=128):
        K per token: 32 × 8 × 128 × 2 bytes (fp16) = 524,288 bytes = 0.5 MB
        V per token: same
        Total per token: ~1 MB

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW THE KV CACHE ELIMINATES REDUNDANT COMPUTATION              ║
    ╚══════════════════════════════════════════════════════════════════╝

      PRE-FILL PHASE (processing the full prompt at once):
        The complete prompt is processed in ONE batched forward pass.
        K and V vectors are computed for all N prompt tokens.
        ALL K and V values are stored in the KV cache.

        KV cache after pre-fill:
          K[l][0], K[l][1], ..., K[l][N-1]   ← stored for all layers
          V[l][0], V[l][1], ..., V[l][N-1]   ← stored for all layers

      DECODE PHASE (generating each new token):
        For each new token at position N+i:
          1. Compute Q for the new token only (one vector)
          2. Retrieve K[l][0..N+i-1] from the cache (no recomputation!)
          3. Retrieve V[l][0..N+i-1] from the cache (no recomputation!)
          4. Compute attention: Q · K^T → softmax → × V
          5. Compute the new token's K and V and ADD to the cache

        Result: each decode step computes Q, K, V for ONE new token only.
        Tokens 0 to N+i-1 are served from cache — never recomputed.

      SPEEDUP:
        Without cache: O(N²) operations per step
        With cache:    O(N)  operations per step  ← linear
        For N=10,000:  10,000× reduction in recomputation


## 5.3  KV Cache Memory Requirements

    ╔══════════════════════════════════════════════════════════════════╗
    ║  KV CACHE SIZE — WORKED CALCULATIONS                             ║
    ╚══════════════════════════════════════════════════════════════════╝

      FORMULA (with full MHA — Multi-Head Attention):
        KV cache = n_tokens × n_layers × 2 × n_heads × d_head × bytes_per_element

        Where the 2 accounts for both K and V.

      LLaMA 2 7B (full MHA: n_heads = n_kv_heads = 32):
        Per token: 32 × 2 × 32 × 128 × 2 bytes = 524,288 bytes ≈ 0.5 MB
        At 4,096 tokens:  2 GB KV cache  (comparable to model weights!)
        At 32,768 tokens: 16 GB KV cache (larger than the model itself)

      LLaMA 3 8B (GQA: n_kv_heads = 8, not 32):
        Per token: 32 × 2 × 8 × 128 × 2 bytes = 131,072 bytes ≈ 0.125 MB
        At 8,192 tokens:  1 GB KV cache
        At 131,072 tokens: 16 GB KV cache

      GQA (Grouped Query Attention) reduces KV cache by n_heads / n_kv_heads:
        LLaMA 3 8B: 32 heads / 8 KV heads = 4× reduction vs. full MHA.
        This is why GQA was introduced — at long contexts, KV cache size
        was the primary GPU memory bottleneck.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  KV CACHE vs. MODEL WEIGHTS — RELATIVE SIZE                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      LLaMA 3 8B on a single A100 (80 GB VRAM):

        Model weights (fp16):  ~16 GB
        Available for KV cache: ~64 GB  (after weights)

        At 0.125 MB per token (GQA):
          64 GB / 0.125 MB = 512,000 tokens of KV cache

        BUT serving 100 concurrent users:
          512,000 / 100 = 5,120 tokens per user
          → Only a 5k context window per user at 100 concurrent sessions

        This is why batch size, concurrency, and context length are
        directly competing for the same GPU VRAM budget.
        Longer contexts = fewer concurrent users for the same GPU.


## 5.4  Prefix Caching — Persistent KV Cache Across Requests

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FROM SESSION-LOCAL TO CROSS-REQUEST KV CACHING                  ║
    ╚══════════════════════════════════════════════════════════════════╝

      Standard KV cache: lives only for the duration of one API call.
      At session end → KV cache is cleared.

      Prefix caching: the provider stores the KV activations for static
      prefixes (system prompts, large documents) and reuses them across
      multiple requests from different users.

      WITHOUT PREFIX CACHING:
        Request 1: pre-fill 2,000-token system prompt → compute KV → discard
        Request 2: pre-fill same 2,000-token system prompt → compute KV → discard
        Request 3: pre-fill same 2,000-token system prompt → compute KV → discard
        → Same computation repeated for every request.

      WITH PREFIX CACHING:
        Request 1: pre-fill 2,000-token system prompt → compute KV → STORE
        Request 2: same prefix → cache HIT → load stored KV → skip pre-fill
        Request 3: same prefix → cache HIT → load stored KV → skip pre-fill
        → Computation done once; reused indefinitely until cache expires.

      This is the same KV cache mechanism as intra-session caching,
      applied across session boundaries by the serving infrastructure.


---


## 6. KV CACHE INVALIDATION — What Clears the Cache and Why It Matters

Cache invalidation is the process by which stored KV entries become invalid
and must be discarded or recomputed. Understanding when and why invalidation
occurs determines how much you benefit from caching infrastructure.


## 6.1  Session-Boundary Invalidation

    ╔══════════════════════════════════════════════════════════════════╗
    ║  INVALIDATION AT SESSION END — THE DEFAULT CASE                  ║
    ╚══════════════════════════════════════════════════════════════════╝

      The most common form of KV cache invalidation is session termination.
      When an API call completes, all KV entries for that session are freed
      from GPU VRAM. The memory is returned to the pool for new requests.

      TIMELINE:
        t=0:   API call begins. KV cache allocated.
        t=1:   Pre-fill. KV computed for all prompt tokens.
        t=2:   Decode. New tokens generated. KV extended.
        t=N:   EOS token generated. Response complete.
        t=N+1: API call returns to caller.
        t=N+2: VRAM freed. KV cache deallocated.

      At t=N+2: all KV data is gone.
      The NEXT API call, even with an identical prompt, starts from scratch.

      IMPLICATION:
        Without prefix caching, every API call pays full pre-fill cost
        for the system prompt, even if it's identical across thousands of calls.


## 6.2  Token-Level Invalidation — The Prefix Rule

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY A SINGLE CHANGED TOKEN INVALIDATES THE ENTIRE SUFFIX        ║
    ╚══════════════════════════════════════════════════════════════════╝

      The KV cache stores attention matrices that depend on every token
      in the sequence UP TO that position.

      K and V for token at position i depend on:
        • The embedding of token i itself
        • The positional encoding at position i
        • The result of attention on tokens 0 through i-1

      If ANY token at position j < i changes, then:
        • The attention computation for token i changes
        • K[i] and V[i] become stale
        • All K[j], V[j] for j ≥ changed_position are invalid

      EXAMPLE — one token change cascades:

        Original prompt: "You are a helpful assistant. User: Hello, how are you?"
        Modified prompt: "You are a POLITE assistant. User: Hello, how are you?"
                                              ↑
                               one token changed at position 6

        Position 6 KV: now invalid (different token)
        Positions 7 to N: now invalid (all depended on position 6's attention)
        Positions 0 to 5: still valid (unchanged, and causally prior)

        Cache reuse:  positions 0–5 still valid  (small prefix: "You are a ")
        Must recompute: positions 6 to N

      This is why the requirement for prefix caching is so strict:
        The prefix must be BYTE-IDENTICAL — not just semantically equivalent.

      REAL-WORLD INVALIDATION CAUSES:
        ✗  Adding the current date to the system prompt → invalidates daily
        ✗  Personalising with user's name in the system prompt → different per user
        ✗  Including a request counter ("Request #4823") → different every call
        ✗  Dynamic content anywhere in the cached prefix
        ✗  Prompt A/B testing (variants produce different prefixes)


## 6.3  Memory-Pressure Eviction

    ╔══════════════════════════════════════════════════════════════════╗
    ║  VRAM PRESSURE — EVICTION AS RESOURCE MANAGEMENT                 ║
    ╚══════════════════════════════════════════════════════────────────╝

      GPU VRAM is shared among:
        • Model weights (fixed, loaded at startup)
        • Active KV caches (all concurrent sessions)
        • Prefix cache entries (provider-side stored prefixes)
        • Forward pass activation buffers (temporary)

      When VRAM fills up, old or lower-priority KV cache entries must be
      evicted to make room for new ones.

      EVICTION POLICIES:
        LRU (Least Recently Used): evict the KV cache entry that has not been
          accessed for the longest time. Common for prefix cache stores.

        Priority-based: evict sessions that have been idle longest; prefer
          to keep recently-active session KV caches.

        PagedAttention (vLLM): divides KV cache into fixed-size blocks.
          Evicts individual BLOCKS rather than entire session caches.
          Allows partial reuse when only part of a session's KV is evicted.

      SYMPTOMS OF EVICTION:
        • Unexpected latency spikes (prefix cache miss → full pre-fill needed)
        • Throughput drops under heavy load
        • Increased TTFT (Time to First Token) when system is loaded

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TTL (TIME-TO-LIVE) EXPIRY — PROVIDER-SIDE CACHE                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      For cross-request prefix caching (as opposed to within-session),
      providers set a TTL after which the cached KV is considered stale.

        Anthropic prefix cache:  ~5 minutes of inactivity → expires
        OpenAI automatic cache:  ~1 hour (approximately)
        Google (Gemini):         Explicit TTL set at creation time

      AFTER TTL EXPIRY:
        The next request with that prefix pays full input price (cache miss).
        The KV is recomputed and cached again from the new request.

      IMPACT ON LOW-TRAFFIC APPLICATIONS:
        If your service handles 1 request per 10 minutes, the cache expires
        between requests. You pay full price for every call.
        Prefix caching is most valuable for services with frequent requests
        (multiple per minute) that consistently use the same prefix.

      KEEPING THE CACHE WARM:
        For known idle periods (e.g. overnight), send a periodic "keep-alive"
        request with the prefix to prevent cache expiry.
        Cost: negligible (short request; cache hit price if still warm).


## 6.4  Cost Consequences of Cache Invalidation

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHAT INVALIDATION COSTS — QUANTIFIED                            ║
    ╚══════════════════════════════════════════════════════════════════╝

      SCENARIO: Production assistant with 10,000-token system prompt.
                 100,000 API calls/day. Anthropic Claude 3.5 Sonnet.
                 Input: $3.00/MTok full, $0.30/MTok cache hit.

      CASE A — Perfect cache hit rate (100%):
        10,000 × 100,000 = 1B tokens @ $0.30/MTok = $300/day

      CASE B — Cache miss due to dynamic content in prefix (0% hit rate):
        10,000 × 100,000 = 1B tokens @ $3.00/MTok = $3,000/day
        EXTRA COST: $2,700/day = $81,000/month from one poor design decision.

      CASE C — Cache miss due to TTL expiry (50% hit rate):
        500M tokens @ $0.30 + 500M tokens @ $3.00 = $150 + $1,500 = $1,650/day

      CASE D — Personalisation in system prompt (0% cache hits because
               every user gets a different prefix):
        Same as Case B: $3,000/day.
        FIX: move personalisation to the user turn, not the system prompt.
             System prompt stays static → cache can apply.

      THE SINGLE MOST EXPENSIVE MISTAKE:
        Putting any dynamic, per-request content inside the cached prefix.
        Date stamps, user names, request IDs, random seeds, version numbers —
        any of these in the system prompt kills the cache entirely.


---


## 7. EPISODIC vs. SEMANTIC MEMORY — Event-Specific vs. General World Knowledge

The distinction between episodic and semantic memory comes from cognitive
neuroscience. Applied to LLMs, it reveals a fundamental asymmetry: models
have rich semantic memory but no native episodic memory — and every "memory
of a conversation" is an artificial construction by the application layer.


## 7.1  The Cognitive Science Background

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ENDEL TULVING'S MEMORY TAXONOMY (1972)                          ║
    ╚══════════════════════════════════════════════════════════════════╝

      Endel Tulving (cognitive psychologist) distinguished two forms of
      long-term declarative memory in humans:

      SEMANTIC MEMORY:
        General world knowledge, facts, concepts, language.
        Not tied to any specific personal experience.
        "Paris is the capital of France."
        "A triangle has three sides."
        "Dogs bark."
        You know these facts but cannot remember the specific occasion
        when you learned them. They are de-contextualised.

      EPISODIC MEMORY:
        Autobiographical, event-specific memory.
        Tied to a specific time, place, and context.
        "I first ate sushi at a restaurant in Kyoto in 2018."
        "In our last meeting on Tuesday, you said the deadline was March 15."
        "Earlier in this conversation, you told me your name is Alice."
        These are personal recollections of specific events.

      The key difference: WHEN and WHERE the information was encountered.
        Semantic: known, but origin forgotten.
        Episodic: known AND the original event is part of the memory.


## 7.2  LLM Memory Through This Lens

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW LLMS MAP TO THE SEMANTIC/EPISODIC TAXONOMY                  ║
    ╚══════════════════════════════════════════════════════════════════╝

      IN-WEIGHTS MEMORY → SEMANTIC MEMORY (approximate)

        The model's weights encode general world knowledge, language
        patterns, and reasoning procedures.

        "Paris is the capital of France." → in weights
        "Python uses indentation for blocks." → in weights
        "The Treaty of Versailles ended WWI." → in weights

        IMPORTANT CAVEAT: Unlike human semantic memory, in-weights memory
        is blurry and fallible. The model may "know" Paris is the capital
        of France very reliably (frequent in training data) but may
        "know" the population of a small town unreliably (rare data,
        weak weight activations). Human semantic memory is crisp for
        well-learned facts; LLM in-weights memory degrades with rarity.

      IN-CONTEXT MEMORY + EXTERNAL MEMORY → EPISODIC MEMORY (constructed)

        The model has NO native episodic memory.
        It cannot remember "the conversation we had last Tuesday."
        It does not experience the passage of time between calls.
        Every call begins with a blank slate of live memory.

        EPISODIC MEMORY IS CONSTRUCTED by:
          • Keeping conversation history in the context window
            (episodic within a session)
          • Storing past conversations in an external vector DB and
            retrieving relevant ones at query time
            (episodic across sessions — RAG-based memory)
          • Extracting key facts from past conversations and injecting
            them into future system prompts
            (episodic to semantic conversion — compressed, de-contextualised)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE EPISODIC MEMORY GAP — IMPLICATIONS FOR APPLICATION DESIGN  ║
    ╚══════════════════════════════════════════════════════════════════╝

      Because the model has no episodic memory, every "memory of a specific
      event" must be engineered by the application layer.

      WHAT THIS MEANS IN PRACTICE:

        "Remember when I told you about my project?" (user, session 3)
          → Only works if session 1's content was stored and retrieved.
          → If not retrieved → model cannot answer → disappoints user.

        "You said the deadline was March 15." (user, session 2)
          → Only accurate if session 1's statement is in context.
          → If not in context → model may generate a different date.
          → This is a RELIABILITY AND TRUST issue, not just UX.

        "Based on our previous discussions, what should I prioritise?"
          → Requires retrieving a meaningful summary of all past sessions.
          → Without this retrieval → model has zero history to draw on.
          → Response will be generic, not personalised.

      THE ENGINEERING OBLIGATION:
        Any application that promises "memory" across sessions is making
        a promise that the model cannot keep natively. The application
        MUST implement episodic memory storage and retrieval.
        Failing to do so is not just a UX shortcoming — it can lead to
        inconsistent, unreliable, or trust-damaging model behaviour.


## 7.3  Three Patterns for Simulating Episodic Memory

    ╔══════════════════════════════════════════════════════════════════╗
    ║  PATTERN 1 — FULL HISTORY IN CONTEXT (within-session episodic)  ║
    ╚══════════════════════════════════════════════════════════════════╝

      All turns of the CURRENT session are in the context window.
      The model can reference anything said earlier in this session.
      Sessions are isolated — past sessions are not accessible.

      Characteristics:
        Fidelity: PERFECT (verbatim access to all turns)
        Capacity: limited by context window (typically 100–500 turns)
        Cross-session: NO
        Cost: grows linearly with turns

      Best for: short to medium sessions where full accuracy matters.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  PATTERN 2 — VECTOR-RETRIEVED EPISODIC MEMORY                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      Every turn (or summarised turn) stored in a vector DB.
      At each new turn, retrieve the 3–5 most semantically relevant past turns.
      Inject as "Previous relevant conversations" block.

      Characteristics:
        Fidelity: APPROXIMATE (only semantically relevant memories retrieved)
        Capacity: unlimited
        Cross-session: YES
        Cost: additional embedding + vector search per turn

      Best for: long-running assistants needing cross-session memory.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  PATTERN 3 — EXTRACTED ENTITY MEMORY (episodic → semantic)      ║
    ╚══════════════════────────────────────────────────────────────────╝

      After each session, an extraction pass identifies and stores key facts:
        User preferences, stated goals, established constraints, decisions.
      These extracted facts are injected as a structured "user profile"
      in all future sessions — episodic detail converted to semantic form.

      Storage format (example):
        {
          "name": "Alice",
          "role": "backend engineer",
          "language": "Python",
          "project": "billing system for SaaS startup",
          "stack": "FastAPI + Postgres + Stripe",
          "constraints": ["no microservices", "must deploy to Railway"],
          "current_task": "webhook endpoint for payment confirmation"
        }

      Characteristics:
        Fidelity: LOSSY (raw conversations discarded; only extracted facts remain)
        Capacity: unlimited (just key-value pairs)
        Cross-session: YES
        Cost: one extraction LLM call per session end

      Best for: personalisation without full conversation replay.


---


## 8. WORKING MEMORY ANALOGY — Context Window as Cognitive Working Memory

The human concept of working memory provides a powerful and surprisingly
precise analogy for the context window. Understanding where the analogy
holds — and crucially where it breaks — gives a reliable intuition for
predicting LLM behaviour.


## 8.1  Human Working Memory — The Cognitive Model

    ╔══════════════════════════════════════════════════════════════════╗
    ║  BADDELEY'S WORKING MEMORY MODEL (1974, revised 2000)            ║
    ╚══════════════════════════════════════════════════════════════════╝

      Alan Baddeley's influential model describes working memory as
      a multi-component system for temporarily holding and manipulating
      information during active cognitive tasks.

      CENTRAL EXECUTIVE
      ─────────────────
      The "controller" — allocates attention, coordinates sub-systems,
      links working memory to long-term memory. Limited capacity;
      can only process one thing deeply at a time.

      PHONOLOGICAL LOOP
      ─────────────────
      Stores verbal/linguistic information temporarily (about 2 seconds
      of speech). An inner voice rehearses items to prevent decay.
      Example: holding a phone number in mind while dialling it.

      VISUOSPATIAL SKETCHPAD
      ───────────────────────
      Stores visual and spatial information temporarily.
      Used for mental imagery, spatial reasoning, navigating.

      EPISODIC BUFFER (added 2000)
      ─────────────────────────────
      A temporary store that integrates information from all sub-systems
      and from long-term memory into a unified, coherent episode.
      Limited capacity (~4 chunks).

    ╔══════════════════════════════════════════════════════════════════╗
    ║  MILLER'S LAW — THE CAPACITY LIMIT (1956)                        ║
    ╚══════════════════════════════════════════════════════════════════╝

      George Miller's famous paper "The Magical Number Seven, Plus or
      Minus Two" established that humans can hold approximately 7±2
      independent "chunks" of information in working memory simultaneously.

      A "chunk" is any unit of information that has been encoded as a
      meaningful whole. For a novice: individual digits are chunks.
      For a chess master: whole board configurations are single chunks.

      The capacity limit is not about bits or bytes — it is about
      meaningful units. Expertise is largely the ability to chunk more
      information into fewer, richer units.


## 8.2  The Analogy — Where It Holds

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CONTEXT WINDOW ≈ WORKING MEMORY — SIX PARALLELS                ║
    ╚══════════════════════════════════════════════════════════════════╝

      1. FINITE CAPACITY
         ─────────────────
         Human working memory: ~7±2 chunks / ~4 items in the episodic buffer.
         LLM context window: N tokens (1k–2M depending on model).
         Both have a hard limit. Information beyond the limit either pushes
         out older information or simply cannot be processed.

      2. IMMEDIATE ACCESSIBILITY
         ─────────────────────────
         Human: items in working memory are immediately available for reasoning.
         LLM: tokens in the context window are immediately available via attention.
         Both are the "live" active memory — no retrieval delay.

      3. VOLATILE / SESSION-SPECIFIC
         ────────────────────────────
         Human: working memory contents are lost when attention shifts.
                (You forget the phone number as soon as you're distracted.)
         LLM: context window is cleared at session end.
         Neither is permanent storage without deliberate transfer.

      4. INTERACTION WITH LONG-TERM MEMORY
         ─────────────────────────────────────
         Human: working memory actively retrieves from long-term memory
                to support reasoning about current information.
         LLM: in-context tokens activate in-weights knowledge via attention,
              drawing on trained patterns to interpret the current content.

      5. CAPACITY LIMITS AFFECT REASONING QUALITY
         ──────────────────────────────────────────
         Human: cognitive load theory — when working memory is full,
                complex reasoning degrades. Learning is harder when
                the learner is overloaded.
         LLM: when context is nearly full (>80% of limit), retrieval
              from middle positions degrades. "Lost in the middle" effect
              mirrors cognitive overload.

      6. CHUNKING IMPROVES EFFECTIVE CAPACITY
         ──────────────────────────────────────
         Human: organising information into meaningful chunks allows
                more content in the same 7±2 slots.
         LLM: dense, well-structured prompts carry more information per
              token than verbose, redundant ones. A tight 500-token prompt
              can be more "useful" than a bloated 5,000-token one.


## 8.3  Where the Analogy Breaks Down

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SIX WAYS THE ANALOGY IS WRONG                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      1. SCALE IS INCOMPARABLE
         Human working memory: ~7 chunks.
         LLM context window: 200,000 tokens (Claude 3) ≈ 150,000 words.
         The scale difference (roughly 5 orders of magnitude) means
         analogical reasoning about capacity must be calibrated carefully.
         The LLM's "working memory" is vastly larger than a human's.

      2. NO REHEARSAL / DECAY
         Human working memory decays within seconds without rehearsal.
         An LLM context window holds tokens perfectly and permanently
         until they are explicitly removed — there is no decay over time.
         A token at position 0 is just as present as one at position N-1.

      3. ALL POSITIONS ACCESSIBLE SIMULTANEOUSLY
         Human working memory is accessed sequentially — attention must
         focus on one chunk at a time (serial processing).
         LLM attention processes ALL positions in parallel within each layer.
         Every token simultaneously "reads" every other token.
         This is fundamentally non-human.

      4. NO VOLUNTARY DIRECTION OF ATTENTION
         Human working memory: the person CHOOSES what to focus on.
         LLM attention: weights are FIXED (frozen at training).
         The model cannot decide to "pay more attention" to a specific
         part of the context — the attention patterns emerge from the
         trained weights applied to the input. The model has no agency
         over its own attention.

      5. RECENCY EFFECT IS DIFFERENT
         Human: strong recency effect — recent items better recalled
                because they haven't decayed yet.
         LLM: recency effect exists but for different reasons (positional
              encoding, distance penalty in ALiBi/RoPE). The mechanism
              is mathematical, not temporal.

      6. NO PROSPECTIVE MEMORY / INTENTION
         Human working memory supports "prospective memory" — the intention
         to do something in the future, maintained while doing something else.
         LLM has no ongoing process between turns. There is no "between turns."
         Each call is complete and isolated; no intention persists.


## 8.4  The Analogy Applied — Practical Reasoning Tool

    ╔══════════════════════════════════════════════════════════════════╗
    ║  USING THE WORKING MEMORY ANALOGY TO PREDICT LLM BEHAVIOUR      ║
    ╚══════════════════════════════════════════════════════════════════╝

      The working memory analogy is most useful as an intuition pump
      for prompt design. Here are the analogical predictions and their
      LLM equivalents:

      ANALOGY: "You can't solve a complex problem while also trying to
                remember a 10-digit number."
      LLM EQUIVALENT: A context window packed with 80,000 tokens of
                      marginally relevant documents leaves less effective
                      capacity for the actual reasoning task.
                      LESSON: Keep context dense and relevant. Noise is
                      a cognitive load analogue — it degrades performance.

      ANALOGY: "Organise new information into familiar categories to
                remember more."
      LLM EQUIVALENT: Well-structured, clearly organised prompts (using
                      headers, numbered lists, explicit section labels)
                      are more reliably processed than unstructured prose
                      with the same information.
                      LESSON: Structure your prompts. Labels and headers
                      act like cognitive "chunk headers."

      ANALOGY: "Write it down before you forget it."
      LLM EQUIVALENT: Chain-of-thought prompting lets the model "write
                      down" intermediate reasoning steps in the context.
                      The model extends its own in-context working memory
                      with its generated text.
                      LESSON: For complex multi-step reasoning, ask the
                      model to show its work. Each step written becomes
                      available as context for the next step.

      ANALOGY: "You can only hold one conversation while truly paying
                attention."
      LLM EQUIVALENT: A prompt that simultaneously asks the model to
                      follow a complex persona, apply multiple constraints,
                      reason about a long document, AND format the output
                      in a complex structure may produce worse results on
                      each dimension than separate focused calls.
                      LESSON: For complex tasks, consider decomposing
                      into focused sub-tasks rather than one monolithic prompt.

      ANALOGY: "Important things at the beginning and end of a list are
                remembered best (serial position effect)."
      LLM EQUIVALENT: This is the "lost in the middle" effect — the most
                      direct parallel. Information at the start and end of
                      the context window receives more attention weight.
                      LESSON: Put critical instructions at the beginning
                      (system prompt) and critical data at the end (just
                      before the question).


## 8.5  A Unified Mental Model

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ALL FOUR MEMORY TYPES AS A UNIFIED COGNITIVE SYSTEM            ║
    ╚══════════════════════════════════════════════════════════════════╝

      Human Cognitive System          LLM System
      ──────────────────────────────  ─────────────────────────────────
      Long-term semantic memory    ≈  In-weights memory
      (world knowledge, language)     (frozen at training, always active)

      Long-term episodic memory    ≈  External memory stores
      (autobiographical records)      (vector DB, key-value, files)
                                      + application-layer retrieval

      Working memory               ≈  In-context memory
      (active, limited, volatile)     (context window, ephemeral)

      Procedural memory            ≈  In-weights memory (procedures)
      (skills, how-to knowledge)      (reasoning patterns, coding skills)

      (no direct human analogue)   ≈  KV cache
                                      (computational speedup artefact)

      The absence of a human analogue for the KV cache is telling:
      human working memory does not have a separate "cache" for
      re-reading prior information efficiently. Human memory is
      reconstructive — each recall is a fresh reconstruction, not
      a cached lookup. The LLM's KV cache is a fundamentally
      computational concept with no cognitive counterpart.

      This unified map makes a subtle but important point:
        The LLM has semantic memory (in weights) and can construct
        episodic memory (via external stores + application logic),
        but its working memory (context window) is orders of magnitude
        larger than a human's — while simultaneously being far less
        intelligent in how it uses that working memory.

        A human with working memory overload uses meta-cognition to
        decide what to drop, what to rehearse, and what to compress.
        An LLM simply applies fixed attention weights to whatever is in
        the context window — the "what to attend to" question is answered
        by trained pattern-matching, not by deliberate cognitive strategy.

**FINAL KEY CONCEPT:** The LLM is simultaneously a much more powerful and
much more limited working memory system than a human. It can hold 150,000
words in "mind" at once — but it cannot decide what to do with them.
That decision must come from the application design.


---


## Glossary

    +──────────────────────────────┬────────────────────────────────────────────────────────+
    │  Term                        │  Definition                                             │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Attention mechanism         │  The transformer operation where each token computes    │
    │                              │  Q·K^T scores against all other tokens and reads a      │
    │                              │  weighted sum of their V vectors.                       │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Confabulation               │  Generating plausible-sounding but incorrect content    │
    │  (hallucination)             │  because in-weights statistical patterns produce a      │
    │                              │  fluent continuation without verifying factual truth.   │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Cosine similarity           │  A measure of the angle between two vectors in high-    │
    │                              │  dimensional space. 1.0 = identical direction;           │
    │                              │  0.0 = unrelated; used in vector database retrieval.    │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Embedding                   │  A dense vector representation of text in a high-       │
    │                              │  dimensional space where semantic similarity maps to     │
    │                              │  vector proximity.                                      │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Episodic memory             │  Memory of specific events tied to a time, place, and   │
    │                              │  context. In LLMs: simulated via conversation history   │
    │                              │  or external vector stores.                             │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  External memory             │  Information stored outside the model in databases,     │
    │                              │  vector stores, or files. Retrieved and injected into   │
    │                              │  context at query time.                                 │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Fine-tuning                 │  Continued training on a smaller task-specific dataset  │
    │                              │  to modify in-weights knowledge and behaviour.          │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  GQA                         │  Grouped-Query Attention — reduces KV cache size by     │
    │                              │  sharing K/V projections across groups of query heads.  │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  In-context memory           │  The tokens currently in the active context window —    │
    │                              │  the model's only live, directly accessible memory.     │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  In-weights memory           │  Knowledge encoded into model parameters during         │
    │                              │  training. Activated probabilistically; frozen at       │
    │                              │  inference; cannot be individually queried.             │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  KV cache                    │  Pre-computed Key-Value attention matrices stored in    │
    │                              │  GPU VRAM to avoid reprocessing prior tokens during     │
    │                              │  each decode step.                                      │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  KV cache invalidation       │  The process by which cached KV entries become stale    │
    │                              │  and must be discarded. Triggered by session end,       │
    │                              │  token change, memory pressure, or TTL expiry.          │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Mechanistic interpretability│  A research field studying how specific facts and       │
    │                              │  computations are implemented in transformer weights.   │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Miller's Law                │  The capacity of human working memory is approximately  │
    │                              │  7 ± 2 independent "chunks" of information.             │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  PagedAttention              │  A KV cache management system (vLLM) that divides the  │
    │                              │  cache into fixed-size pages, enabling efficient        │
    │                              │  sharing and partial eviction.                          │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Parametric memory           │  Synonymous with in-weights memory. Knowledge that      │
    │                              │  is stored in the model's parameters (weights).         │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Pre-fill phase              │  The batched forward pass that processes all prompt     │
    │                              │  tokens simultaneously before generation begins.         │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Prefix caching              │  Provider-side storage of KV activations for repeated  │
    │                              │  token prefixes, enabling cache-hit pricing on          │
    │                              │  subsequent requests using the same prefix.             │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  RAG                         │  Retrieval-Augmented Generation — the pattern of        │
    │                              │  retrieving relevant external memory and injecting it   │
    │                              │  into the context window before generation.             │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Semantic memory             │  General world knowledge, facts, and concepts not tied  │
    │                              │  to a specific experience. In LLMs: primarily encoded   │
    │                              │  in FFN layer weights.                                  │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Session boundary            │  The point at which an API call completes and the KV   │
    │                              │  cache is deallocated. Nothing from the session         │
    │                              │  persists in the model.                                 │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  TTL (Time-To-Live)          │  The duration after which a cached entry expires and    │
    │                              │  must be recomputed. Typically 5 minutes (Anthropic)    │
    │                              │  to ~1 hour (OpenAI) for prefix caching.               │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Vector database             │  A storage system optimised for storing and querying    │
    │                              │  high-dimensional embedding vectors via approximate     │
    │                              │  nearest-neighbour (ANN) search.                       │
    +──────────────────────────────┼────────────────────────────────────────────────────────+
    │  Working memory              │  In cognitive psychology: the limited-capacity system   │
    │                              │  for actively holding and manipulating information      │
    │                              │  during a cognitive task. Analogue: context window.     │
    +──────────────────────────────┴────────────────────────────────────────────────────────+


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