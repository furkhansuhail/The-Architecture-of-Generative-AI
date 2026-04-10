"""
Tutorial Template — Automation & Infrastructure
=================================================
Copy this file, rename it, and fill in the sections.

The TOPIC_NAME and CATEGORY are parsed by the app.
OPERATIONS entries can specify "language" as "bash", "yaml", "python", "json", etc.
"""

TOPIC_NAME = "LLM - Context Window"
DISPLAY_NAME = "LLM - Context Window"
ICON         = "🪟"
SUBTITLE     = "The model's entire working memory — what it is, what fills it, and where it breaks"
CATEGORY = "LLM Architecture"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = r"""

Order:

    Module_Concepts · Context Window
          ├── context_window_definition       What the context window is and contains
          ├── context_window_sizes            GPT-2 (1k) → Claude (200k) → Gemini (1M+)
          ├── what_fills_the_window           System prompt / history / docs / tools budget
          ├── lost_in_the_middle              Why buried information gets less attention
          ├── context_window_vs_memory        Why the context window IS the model's memory
          ├── sliding_window_attention        Handling sequences beyond full attention range
          ├── context_utilisation_efficiency  Density vs. noise — not all tokens are equal
          ├── context_overflow_strategies     Truncation, summarisation, chunking, retrieval
          ├── long_context_performance        Quality degradation at very long contexts
          ├── positional_encoding             How the model knows where each token sits
          └── rope_alibi_encodings            RoPE, ALiBi — extending beyond training length


## 1. CONTEXT WINDOW DEFINITION — What It Is and What It Contains

The context window is the complete set of tokens that a language model can
"see" during a single inference call. It is the model's entire universe for
that call: everything the model knows, everything it can reason about, and
everything it can be influenced by must be inside this window.

Nothing outside the context window exists for the model.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE CONTEXT WINDOW — A FLAT SEQUENCE OF TOKENS                  ║
    ╚══════════════════════════════════════════════════════════════════╝

      The context window is not a document. Not a conversation tree.
      Not a database. It is a single, flat, ordered sequence of token IDs.

      Position:  [0]   [1]   [2]   [3]   [4]  ...  [N-2] [N-1]
      Content:   <BOS> [sys] [sys] [usr] [usr] ...  [ast] [ast]
                 ↑                                              ↑
                 position 0                           position N-1
                 (first token ever)           (most recent token — generation starts here)

      Every token at every position is processed by every transformer layer.
      Every token can (in principle) attend to every prior token.
      There is no hierarchy, no nesting, no special "important" zone.
      It is one long tape of integers.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHAT "FITS" IN ONE CONTEXT WINDOW                               ║
    ╚══════════════════════════════════════════════════════════════════╝

      A context window is measured in tokens. Approximate real-world
      equivalents (English prose at ~4 chars/token):

      ┌─────────────────┬─────────────────────────────────────────────┐
      │  Token count    │  Rough equivalent                           │
      ├─────────────────┼─────────────────────────────────────────────┤
      │  1,000          │  750 words — two or three pages of text     │
      │  4,096          │  3,000 words — a short essay or article     │
      │  16,000         │  12,000 words — a long report or chapter    │
      │  32,000         │  24,000 words — a novella chapter           │
      │  128,000        │  96,000 words — a short novel               │
      │  200,000        │  150,000 words — a long novel               │
      │  1,000,000      │  750,000 words — roughly War and Peace ×2   │
      └─────────────────┴─────────────────────────────────────────────┘

      For code (higher token density): divide these word counts by ~2.
      For non-English text: divide by the appropriate language multiplier
      (see Tokenisation module, section 5.3).

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE HARD LIMIT — WHAT HAPPENS WHEN THE WINDOW IS FULL           ║
    ╚══════════════════════════════════════════════════════════════════╝

      The context window is not a soft guideline. It is a hard architectural
      constraint. If prompt_tokens + generated_tokens > max_context, the
      model cannot proceed.

      THREE OUTCOMES when the limit is reached:

        1. API ERROR
           The provider rejects the request before any processing.
           "This model's maximum context length is 128,000 tokens. Your
            messages resulted in 131,421 tokens."

        2. SILENT TRUNCATION
           The serving layer silently removes tokens from the front or
           back of the prompt before processing. The caller may not
           know this happened. The model's response may be incoherent
           because it never saw the truncated content.

        3. GENERATION STOPS
           During streaming, generation stops mid-sentence if the
           running total of prompt + generated tokens hits the limit.

      The max_tokens parameter in API calls reserves space for the
      response: effective input budget = context_limit − max_tokens.

**KEY CONCEPT:** The context window is the model's entire working memory
for one call. There is no background awareness, no persistent state, no
long-term recall. If information is not in the context window, the model
does not have access to it.


---


## 2. CONTEXT WINDOW SIZES — The Historical Progression

Context window sizes have grown dramatically since 2019. Understanding the
progression helps explain both why the limits exist and what unlocked each
expansion.


## 2.1  The Full Timeline

    +──────────────────────────────────────────────────────────────────────────+
    │  Model              Year  Max Tokens  Training Length   Notes            │
    +──────────────────────────────────────────────────────────────────────────+
    │  GPT-2              2019    1,024          1,024         Transformer limit│
    │  GPT-3              2020    2,048          2,048         Standard limit   │
    │  GPT-3.5-turbo      2022    4,096          4,096         Chat fine-tuned  │
    │  GPT-4 (launch)     2023    8,192          8,192         First GPT-4      │
    │  GPT-4-32k          2023   32,768         32,768         Extended variant │
    │  Claude 1           2023    9,000          9,000         Anthropic        │
    │  Claude 2           2023  100,000        100,000         First 100k model │
    │  GPT-4-turbo        2023  128,000        128,000         Current GPT-4    │
    │  Gemini 1.0 Ultra   2024   32,768         32,768                          │
    │  Claude 3 (all)     2024  200,000        200,000         Sonnet/Opus/Haiku│
    │  GPT-4o             2024  128,000        128,000                          │
    │  Gemini 1.5 Flash   2024 1,000,000      1,000,000        First 1M model  │
    │  Gemini 1.5 Pro     2024 2,000,000      2,000,000        Largest public  │
    │  LLaMA 2            2023    4,096          4,096         Meta open-source │
    │  LLaMA 3 8B/70B     2024    8,192          8,192         Base models      │
    │  LLaMA 3.1 405B     2024  131,072        131,072         Extended         │
    │  Mistral 7B         2023    8,192          8,192         Sliding window   │
    │  Mixtral 8×7B       2024   32,768         32,768                          │
    +──────────────────────────────────────────────────────────────────────────+


## 2.2  Why Early Windows Were Small

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE THREE LIMITS THAT KEPT WINDOWS SMALL                        ║
    ╚══════════════════════════════════════════════════════════════════╝

      1. QUADRATIC MEMORY COST OF ATTENTION
         ────────────────────────────────────
         Standard (naive) attention requires storing an N × N matrix —
         one attention score per token pair.

         At N = 1,024:    1,024² = 1,048,576 values  ≈ 4 MB in fp16
         At N = 8,192:    8,192² = 67,108,864 values ≈ 256 MB in fp16
         At N = 128,000: 128k² = 16,384,000,000 values ≈ 30 GB in fp16

         The N × N matrix simply could not fit in GPU VRAM for long sequences.
         Flash Attention (2022) solved this by never materialising the full
         N × N matrix, computing it tile by tile within SRAM.

      2. KV CACHE MEMORY
         ──────────────────
         Each token processed generates a K and V vector for every layer.
         These must be stored in VRAM for the entire generation.

         KV cache per token ≈ 2 (K+V) × n_layers × d_head × n_heads × 2 bytes (fp16)
         For a 7B model (32 layers, 32 heads, d_head=128):
           = 2 × 32 × 128 × 32 × 2 = 524,288 bytes ≈ 0.5 MB per token

         At 128k tokens: 64 GB KV cache — more than the model weights themselves.
         GQA (Grouped Query Attention) reduced this by 4–8×.

      3. POSITIONAL ENCODING GENERALISATION
         ─────────────────────────────────────
         Original absolute positional encodings had fixed tables up to the
         training length. A model trained to 2k tokens had no positional
         encoding defined for position 2,001.

         RoPE and ALiBi (covered in sections 10–11) solved this by encoding
         position relative to other tokens, enabling extrapolation beyond
         training length.


## 2.3  What Each Context Size Tier Enables

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CONTEXT SIZE TIERS AND THEIR USE-CASE UNLOCKS                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      TIER 1 — 1k–4k tokens  (GPT-2, GPT-3, early GPT-3.5)
      ─────────────────────────────────────────────────────
      • Single document Q&A (one-page articles)
      • Short code review or completion
      • Simple chat with minimal history
      NOT possible: multi-document analysis, long code files,
                    maintaining context across a 10-turn conversation

      TIER 2 — 8k–32k tokens  (GPT-4 launch, Claude 1, Mistral)
      ─────────────────────────────────────────────────────────
      • Full research paper analysis
      • Code review of a module (200–500 lines)
      • Multi-document comparison (3–5 short documents)
      • Moderate conversation history (10–20 turns)
      NOT possible: full codebase ingestion, book analysis

      TIER 3 — 100k–200k tokens  (Claude 2/3, GPT-4-turbo)
      ─────────────────────────────────────────────────────
      • Entire codebases (small-to-medium projects)
      • Full book analysis and citation
      • Legal document review (hundreds of pages)
      • Long conversation history (100+ turns)
      • RAG-like workflows with many retrieved documents

      TIER 4 — 1M+ tokens  (Gemini 1.5)
      ─────────────────────────────────────────────────────
      • Full large codebase ingestion (Linux kernel ~17M tokens)
      • Multiple book corpus analysis
      • Hour-long video transcript + frames
      • Entire company document archives
      NOTE: Quality at extreme lengths degrades (see section 9)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE CONTEXT CAPACITY RACE — VISUALISED                          ║
    ╚══════════════════════════════════════════════════════════════════╝

      2019  GPT-2         1k   ░
      2020  GPT-3         2k   ░░
      2022  GPT-3.5       4k   ░░░░
      2023  GPT-4         8k   ░░░░░░░░
      2023  GPT-4-32k    32k   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
      2023  Claude 2    100k   ████████████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░
      2023  GPT-4-turbo 128k   ████████████████████████████████████████████████████████████████████████████████████████████████░░░░░░░░
      2024  Claude 3    200k   ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
      2024  Gemini 1.5  1M+    (bar would extend ~3 metres at this scale)


---


## 3. WHAT FILLS THE WINDOW — System Prompt / History / Docs / Tools Budget

The context window is consumed by multiple competing components. Understanding
how each component uses the budget — and how that budget evolves across a
conversation — is critical for building production applications.


## 3.1  The Four Major Consumers

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ANATOMY OF A PRODUCTION CONTEXT WINDOW                          ║
    ╚══════════════════════════════════════════════════════════════════╝

      A production API call assembles the context window from up to four
      distinct components, in this order:

      ┌─────────────────────────────────────────────────────────────────┐
      │  POSITION 0 ──────────────────────────────────── POSITION N-1  │
      │                                                                 │
      │  ┌──────────────────┐                                          │
      │  │  SYSTEM PROMPT   │  Fixed instructions, persona, rules,    │
      │  │                  │  output format, safety constraints.     │
      │  │  (static)        │  Same for every user, every request.    │
      │  └──────────────────┘                                          │
      │                                                                 │
      │  ┌──────────────────┐                                          │
      │  │  INJECTED        │  Retrieved documents, database results, │
      │  │  CONTEXT         │  code files, knowledge base articles.   │
      │  │                  │  Changes per request based on the query.│
      │  │  (dynamic)       │                                          │
      │  └──────────────────┘                                          │
      │                                                                 │
      │  ┌──────────────────┐                                          │
      │  │  CONVERSATION    │  All prior user and assistant turns,    │
      │  │  HISTORY         │  in chronological order.                │
      │  │                  │  Grows by ~50–500 tokens per turn.      │
      │  │  (grows)         │                                          │
      │  └──────────────────┘                                          │
      │                                                                 │
      │  ┌──────────────────┐                                          │
      │  │  CURRENT         │  The user's latest message.             │
      │  │  USER MESSAGE    │  Triggers this API call.                │
      │  └──────────────────┘                                          │
      │                                                                 │
      │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
      │  RESPONSE BUDGET  (reserved by max_tokens parameter)           │
      │                                                                 │
      └─────────────────────────────────────────────────────────────────┘


## 3.2  Token Budget by Application Type

Different application architectures allocate the window very differently.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  BUDGET BREAKDOWNS FOR COMMON APPLICATION TYPES                  ║
    ║  (total window: 128k tokens)                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      A — CUSTOMER SUPPORT CHATBOT
      ─────────────────────────────
      System prompt (persona, rules, tone):        2,000 tokens   1.6%
      Product knowledge base (retrieved):         15,000 tokens  11.7%
      Conversation history (last 10 turns):        5,000 tokens   3.9%
      Current user message:                          200 tokens   0.2%
      Response budget (max_tokens):                2,000 tokens   1.6%
      UNUSED:                                    103,800 tokens  81.1%

      B — CODE REVIEW ASSISTANT
      ──────────────────────────
      System prompt (reviewer persona):             1,000 tokens   0.8%
      Code file(s) being reviewed:                60,000 tokens  46.9%
      Conversation history:                        3,000 tokens   2.3%
      Current review request:                        500 tokens   0.4%
      Response budget:                             4,096 tokens   3.2%
      UNUSED:                                     59,404 tokens  46.4%

      C — DOCUMENT ANALYSIS / RAG
      ─────────────────────────────
      System prompt (analyst persona):              1,500 tokens   1.2%
      Retrieved document chunks (top 10):          50,000 tokens  39.1%
      Conversation history (current session):       8,000 tokens   6.3%
      Current query:                                  300 tokens   0.2%
      Response budget:                              4,000 tokens   3.1%
      UNUSED:                                      64,200 tokens  50.1%

      D — LONG-RUNNING AGENTIC TASK
      ──────────────────────────────
      System prompt + tool schemas:                10,000 tokens   7.8%
      Task description + retrieved context:        20,000 tokens  15.6%
      Full conversation/action history:            80,000 tokens  62.5%
      Current step output + next instruction:       4,000 tokens   3.1%
      Response budget:                              4,000 tokens   3.1%
      UNUSED:                                      10,000 tokens   7.8%
      ← approaching the limit — must manage history aggressively


## 3.3  Tool Schemas — The Hidden Budget Consumer

When using function calling / tools, each tool's JSON schema is injected into
the context window. This cost is often underestimated.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TOOL SCHEMA COST — A REALISTIC EXAMPLE                          ║
    ╚══════════════════════════════════════════════════════════════════╝

      A customer-facing assistant with 8 tools:

        get_order_status(order_id)           ~150 tokens
        update_shipping_address(...)         ~200 tokens
        process_refund(order_id, reason)     ~180 tokens
        check_inventory(product_id, sku)     ~160 tokens
        create_support_ticket(...)           ~220 tokens
        get_account_info(customer_id)        ~140 tokens
        schedule_callback(time, phone)       ~170 tokens
        cancel_subscription(account_id)      ~190 tokens
                                         ──────────────
        Total tool overhead:              ~1,410 tokens

      This overhead is paid on EVERY API call, even if only one tool
      is used. For 100,000 calls/day at $3/MTok input:

        1,410 tokens × 100,000 calls = 141M tool-overhead tokens/day
        141M × $3/MTok = $423/day = $12,690/month in tool schema tokens

      MITIGATION STRATEGIES:
        • Only inject tools relevant to the current user context
        • Use shorter descriptions — each word costs tokens
        • Split toolsets: basic users get 3 tools, advanced users get 8
        • Consider OpenAPI spec compression (remove verbose descriptions)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE GROWING HISTORY PROBLEM                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      In a multi-turn chatbot, history grows at roughly:
        50–500 tokens per turn (user message + assistant response)

      For a typical customer support session (avg. response 200 tokens,
      user message 80 tokens = 280 tokens/turn):

        Turn 1:   280 tokens of history
        Turn 5:   1,400 tokens
        Turn 10:  2,800 tokens
        Turn 20:  5,600 tokens
        Turn 50: 14,000 tokens
        Turn 100: 28,000 tokens

      For a 128k window with a 15k system+context payload, the history
      budget is ~109k tokens = ~389 turns before overflow.

      Most sessions end far before that. But long support sessions,
      coding assistants, and autonomous agents can easily run into
      overflow within a session.

      The cost also scales: turn 100 costs 28,000 more input tokens
      than turn 1, just from accumulated history.


---


## 4. LOST IN THE MIDDLE — Why Buried Information Gets Less Attention

One of the most important empirical findings about long-context LLMs is
that they do NOT process all positions equally well. Information placed
in the middle of a long context window is retrieved less accurately than
information placed at the beginning or end.


## 4.1  The Empirical Finding

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE "LOST IN THE MIDDLE" RESULT  (Liu et al., 2023)             ║
    ╚══════════════════════════════════════════════════════════════════╝

      Experiment setup:
        • Give the model K documents (e.g. 20 documents)
        • One document contains the answer to a question
        • Vary WHICH position the answer document appears at (1 through 20)
        • Measure accuracy of the model's answer

      Result (GPT-3.5-turbo-16k, 20 documents):

        Position 1  (first)   ████████████████████████  74%  ← high (primacy)
        Position 2            ████████████████████░░░░  68%
        Position 5            ████████████████░░░░░░░░  60%
        Position 10           ████████████░░░░░░░░░░░░  52%  ← worst
        Position 15           █████████████░░░░░░░░░░░  55%
        Position 18           ████████████████░░░░░░░░  63%
        Position 20 (last)    ████████████████████████  75%  ← high (recency)

      The U-shaped curve:

      Accuracy
        │
      75%│  ●                                           ●
      70%│    ●                                       ●
      65%│      ●                                   ●
      60%│        ●                               ●
      55%│          ●                           ●
      50%│            ●         ●         ●
        │              ●●●●●●●●●
        └────────────────────────────────────────────────
              1   3   5   7   9  11  13  15  17  19   Position

      This U-shape was found consistently across GPT-3.5, Claude 1.3,
      and other models tested. The effect AMPLIFIES as total context
      grows — at 30 documents, the drop in the middle was even steeper.


## 4.2  Why This Happens

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ATTENTION PATTERNS AT LONG CONTEXT                              ║
    ╚══════════════════════════════════════════════════════════════════╝

      Two mechanisms drive the U-shape:

      1. RECENCY BIAS IN TRAINING
         ─────────────────────────
         Language models are trained to predict the NEXT token.
         In natural language, what comes immediately before (recent context)
         is almost always more predictive than what came 10,000 tokens ago.

         As a result, attention heads learn to WEIGHT RECENT TOKENS MORE
         heavily during training. This is adaptive for most language tasks
         but creates a systematic disadvantage for information buried in
         the middle of long contexts.

      2. POSITIONAL ENCODING GRADIENTS
         ──────────────────────────────
         Positional encodings encode distance between tokens.
         Tokens very close to the generation point (end of context) have
         small position deltas. Tokens far back (beginning) have large
         position deltas.

         RoPE and ALiBi both introduce a distance penalty — tokens further
         away contribute less. Beginning tokens partially overcome this
         because they establish the framing for the entire sequence.
         Middle tokens have neither the recency advantage nor the
         framing advantage.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE PRIMACY AND RECENCY EFFECTS IN DETAIL                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      PRIMACY EFFECT (beginning of context performs well):
        The system prompt, task instructions, and initial framing sit at
        positions 0–N_system. These are processed first by every layer.
        More critically: instructions at the very beginning strongly
        influence how ALL subsequent tokens are interpreted. The model
        "reads" the instructions before any content, so the framing sticks.

      RECENCY EFFECT (end of context performs well):
        The most recent user message, the last retrieved document, the most
        recent conversation turn — these sit just before generation begins.
        The model's "working attention" is naturally focused here.
        This is why few-shot examples placed at the end of a prompt tend
        to be more influential than the same examples placed in the middle.


## 4.3  Practical Placement Rules

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHERE TO PUT WHAT — EXPLOITING THE U-CURVE                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      PLACE AT THE BEGINNING (positions 0 → ~10% of context):
        ✓  System prompt and core instructions
        ✓  The task definition ("Summarise the following documents")
        ✓  Output format requirements
        ✓  Critical constraints ("Never reveal pricing")
        ✓  The most important reference document

      PLACE AT THE END (positions ~80% → 100% of context):
        ✓  The current user question or task
        ✓  The most relevant retrieved document
        ✓  Few-shot examples demonstrating the desired output
        ✓  Any constraint you want the model to "remember" at generation time

      LEAST EFFECTIVE PLACEMENT (middle of context):
        ✗  Critical instructions that must be followed
        ✗  The specific document needed to answer the question
        ✗  Safety instructions that must not be forgotten
        ✗  The most important few-shot examples

      PRACTICAL CONSEQUENCE FOR RAG SYSTEMS:
        When retrieving K documents and inserting them before the question,
        the last retrieved document will be most attended to.
        Arrange documents in order of ASCENDING relevance: least relevant
        first, most relevant last — so the best document is closest to
        the question.

      PRACTICAL CONSEQUENCE FOR SYSTEM PROMPTS:
        Put the most critical instruction first AND repeat it just before
        the user message. Repetition at both ends is more effective than
        a single placement in the middle.


---


## 5. CONTEXT WINDOW vs. MEMORY — Why They Are the Same Thing

A persistent misconception: "the model remembers our previous conversations."
It does not. The context window IS the model's memory — the complete extent of it —
and this section makes that equivalence precise.


## 5.1  The Model Has No Other Memory

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHAT EXISTS vs. WHAT DOES NOT EXIST INSIDE THE MODEL            ║
    ╚══════════════════════════════════════════════════════════════════╝

      EXISTS — static, permanent, shared across all calls:
        • Model weights (W_Q, W_K, W_V, W_FFN, etc.)
          These encode factual knowledge, language patterns, and reasoning
          capabilities learned during training.
          They are IDENTICAL before and after your conversation.

      EXISTS — dynamic, per-call, discarded after each call:
        • The context window (the token sequence)
        • The KV cache (the computed K/V activations for those tokens)
        • Activations and logits (exist for microseconds during forward pass)

      DOES NOT EXIST — at any time, anywhere in the model:
        • Memory of previous conversations
        • A running "user profile" that accumulates across calls
        • Any state that persists between API calls (unless explicitly
          stored externally and injected into the context)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE COMPLETE EQUIVALENCE                                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      "What does the model know right now?"
            =
      "What is in the context window right now?"

      These two questions have identical answers. They are not related
      questions — they are the same question.

      The model's weights encode general world knowledge and language
      understanding. But WHAT THE MODEL CAN REASON ABOUT IN THIS CALL —
      what specific documents it can cite, what the user's name is, what
      was said earlier in this session — is exclusively determined by
      what is in the context window.

      An analogy: imagine someone who is extremely well-educated (weights)
      but has no long-term episodic memory. They wake up for each conversation
      having forgotten everything that came before. The only "memory" they
      have during the conversation is the written notes in front of them
      right now (the context window).

      When the conversation ends, the notes are thrown away.
      The next conversation starts with a blank page.


## 5.2  Weights vs. Context — Two Different Kinds of Knowledge

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHAT KNOWLEDGE LIVES WHERE                                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      KNOWLEDGE IN WEIGHTS (parametric memory):

        • General facts: "Paris is the capital of France"
        • Language: grammar, syntax, idiom in dozens of languages
        • Reasoning patterns: how to solve algebra, write code, structure argument
        • Cultural knowledge: history, science, literature up to training cutoff
        • No source citations — just probability distributions over text
        • Cannot be updated without retraining
        • May be wrong, outdated, or hallucinated
        • Shared identically across all users and all calls

      KNOWLEDGE IN CONTEXT WINDOW (in-context memory):

        • This specific document: "The contract states delivery by March 15"
        • This specific user: "My name is Sarah. I'm asking about my account."
        • This specific session: "Earlier I said I wanted the formal version."
        • This specific data: "The API returned: {'status': 'error', 'code': 429}"
        • Fully grounded — the model can quote exact text
        • Changes every call — perfectly tailored to the current task
        • Disappears after the call ends
        • Private to this call (not shared with other users)

      THE INTERACTION:

        The model uses parametric knowledge to INTERPRET and REASON ABOUT
        what is in the context window, but the context window provides the
        SPECIFIC FACTS for the current task.

        "What does the contract say about penalty clauses?"
         ↑ parametric: what "penalty clauses" means, how to read contracts
         ↑ context:     what THIS contract says — must be in the window

        If the contract is not in the context window, the model cannot
        answer accurately regardless of how much it knows about contracts
        in general.


## 5.3  The Application Layer Creates the Illusion of Memory

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW CHATGPT AND SIMILAR SYSTEMS SIMULATE MEMORY                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      The model has no memory. The APPLICATION does the memory work.

      FULL-HISTORY INJECTION (simplest):
        Application stores every turn in a database.
        On each new call, the full history is prepended to the new message.
        The model "remembers" because it re-reads the entire history
        from the context window every time.

        Turn 1:  context = [SYS] [Turn1_user]
        Turn 2:  context = [SYS] [Turn1_user] [Turn1_asst] [Turn2_user]
        Turn N:  context = [SYS] [all prior turns] [TurnN_user]
                                   ↑ grows by ~300 tokens per turn
                                   ↑ sent to the API in full every call

      MEMORY EXTRACTION (smarter):
        After each turn, a background call extracts key facts
        ("user's name: Alice", "preferred language: Python")
        and stores them in a structured memory store.
        Future calls inject only the relevant extracted memories, not
        the raw conversation history.
        The model "remembers" because facts are injected, not because
        it has any native memory.

      VECTOR RETRIEVAL (for very long histories):
        Past turns are embedded and stored in a vector database.
        Each new turn retrieves the 3–5 most semantically relevant
        prior exchanges and injects them.
        The model appears to "remember" relevant past conversations —
        it just only sees the relevant ones, not all of them.

      In ALL cases: the model itself does nothing. The application layer
      does all the work of creating continuity. The model processes a fresh
      context window every single call.


---


## 6. SLIDING WINDOW ATTENTION — Handling Long Sequences

Full (global) attention attends every token to every other token — but this
requires O(N²) memory and compute. For sequences far beyond the training length,
or for models that need to process very long documents cheaply, sliding window
attention provides an alternative.


## 6.1  The Problem with Full Attention at Long Context

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FULL ATTENTION — MEMORY AND COMPUTE AT SCALE                    ║
    ╚══════════════════════════════════════════════════════════════════╝

      Full attention for a sequence of N tokens requires:

        Memory:  N² attention score values per head per layer
        Compute: O(N²) multiply-accumulate operations

      At N = 8,192 (typical GPT-4 trained length):
        8,192² = 67M scores per head   manageable with Flash Attention

      At N = 128,000 (GPT-4-turbo context limit):
        128,000² = 16.4 billion scores per head
        With 32 heads: ~524 billion scores per layer  ← Flash Attention manages this
        With 32 layers: enormous memory bandwidth requirement

      At N = 1,000,000 (Gemini 1.5):
        1,000,000² = 1 trillion scores per head
        Full attention is fundamentally impractical without architectural changes.
        Gemini 1.5 uses a modified attention mechanism (likely linear attention
        or a form of sparse attention) to achieve practical 1M context.

      Even Flash Attention only reduces the MEMORY of full attention, not the
      O(N²) COMPUTE. For truly extreme contexts, the quadratic wall still applies.


## 6.2  The Sliding Window Solution

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SLIDING WINDOW ATTENTION — HOW IT WORKS                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      Instead of attending to ALL previous tokens, each token only attends
      to the W tokens immediately preceding it (its "window").

      Full attention (W = N):
        Token 100 attends to: tokens 0, 1, 2, 3, ..., 99      (100 tokens)
        Token 500 attends to: tokens 0, 1, 2, 3, ..., 499     (500 tokens)
        Token N   attends to: tokens 0, 1, 2, 3, ..., N-1     (N tokens)
                                                                ↑ grows with N

      Sliding window attention (W = 512):
        Token 100 attends to: tokens 0, 1, 2, ..., 99         (100 tokens — full, within W)
        Token 500 attends to: tokens 0, 1, 2, ..., 98, ..., 499  ← wait, only 512 back
              actually:        tokens 499-512 to 499           (512 tokens max)
        Token 10,000 attends to: tokens 9,488 to 9,999        (512 tokens)
        Token N attends to: tokens N-512 to N-1               (always 512 tokens)
                                                                ↑ FIXED cost regardless of N

      Memory cost: O(N × W) instead of O(N²)
      For W = 512, N = 10,000: 5.1M instead of 100M values — 20× cheaper.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  VISUALISING THE ATTENTION PATTERN                               ║
    ╚══════════════════════════════════════════════════════════════════╝

      Full Attention (N=10 shown):

        ████░░░░░░
        ████░░░░░░
        ██████░░░░
        ████████░░
        ██████████   ← every row attends to all prior tokens
        ██████████
        ██████████
        ██████████
        ██████████
        ██████████

      Sliding Window (W=3):

        ██░░░░░░░░
        ███░░░░░░░
        ████░░░░░░   ← only W=3 prior tokens visible per row
        ░███░░░░░░
        ░░████░░░░
        ░░░████░░░
        ░░░░████░░
        ░░░░░████░
        ░░░░░░████
        ░░░░░░░███

      ██ = attends    ░ = cannot attend (outside window)

      THE KEY LIMITATION:
        Token at position 1,000 cannot attend to token at position 0 directly.
        Information from position 0 can only reach position 1,000 by being
        propagated layer by layer through intermediate tokens.
        At W=512 and 32 layers: the "receptive field" is 512 × 32 = 16,384.
        Information from farther than 16,384 tokens back cannot influence
        the current token at all.


## 6.3  Dilated and Global Attention Patterns

    ╔══════════════════════════════════════════════════════════════════╗
    ║  EXTENSIONS TO BASIC SLIDING WINDOW                              ║
    ╚══════════════════════════════════════════════════════════════════╝

      DILATED ATTENTION (Longformer-style):
        Alternate between consecutive window attention and dilated attention
        where every dth token is attended (sampling across the full sequence).

        Layer 1:  ████░░░░████░░░░████  ← local window (W=4)
        Layer 2:  █░░░░████░░░░███░░░█  ← dilated (d=5)
        Layer 3:  ██████░░░░░░░░░░████  ← local window again

        Dilated attention gives each token a chance to attend to distant
        tokens without the full O(N²) cost. The pattern resembles how
        CNNs use dilated convolutions to increase receptive field.

      GLOBAL TOKENS (Bigbird, Longformer):
        Certain tokens are designated as "global" — they attend to AND
        are attended to by ALL tokens in the sequence.
        In practice: the [CLS] token, the first few tokens of the system
        prompt, and any explicitly flagged "anchor" tokens get global attention.

        Global token ████████████████████████  ← attends to everything
        Token 100    ████░░░░░░░░░░░░░░░░████  ← local window + global tokens
        Token 500    ░░░░████░░░░░░░░░░░░████  ← local window + global tokens

      MISTRAL'S IMPLEMENTATION:
        Mistral 7B uses sliding window attention (W=4,096) combined with
        a rolling KV cache. This allows serving very long sequences at
        inference time while training was done on 8,192 tokens.
        The tradeoff: no direct attention beyond 4,096 tokens back,
        but multi-layer propagation handles most practical dependencies.


---


## 7. CONTEXT UTILISATION EFFICIENCY — Density vs. Noise

Not all tokens in a context window contribute equally to the quality of
the output. A 10,000-token context with dense, relevant information will
consistently outperform a 50,000-token context full of padding, repetition,
and irrelevant material.

This section quantifies token density and gives practical techniques for
maximising the signal per token.


## 7.1  What Makes a Token "High Value"

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HIGH-DENSITY vs. LOW-DENSITY TOKEN PATTERNS                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      HIGH-DENSITY TOKENS — directly inform the model's output:

        • Specific facts needed to answer the question
          ("The API rate limit is 100 requests per minute")
        • Concrete examples demonstrating the desired behaviour
          (few-shot examples with input → output pairs)
        • Critical constraints
          ("The response must be under 50 words")
        • The exact content to be processed
          (the document to summarise, the code to review)

      LOW-DENSITY / NOISE TOKENS — consume budget without improving output:

        • Verbose preambles ("I would like you to please kindly...")
        • Redundant instructions repeated identically
        • Pleasantries and padding ("As an AI language model, I...")
        • Irrelevant retrieved documents (RAG noise)
        • Overly verbose tool schema descriptions
        • Blank lines, excessive whitespace in injected content
        • Boilerplate that adds no signal for the specific task

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE NOISE-DILUTION EFFECT                                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      Adding irrelevant content does not just waste tokens — it actively
      DEGRADES performance on the relevant content.

      Mechanism: attention weights are normalised over all tokens via softmax.
      If 40,000 irrelevant tokens are present alongside 5,000 relevant tokens,
      the relevant tokens receive a smaller fraction of the total attention
      weight than they would in a 5,000-token context.

      Empirically observed:
        Accuracy on a question-answering task with a 5k relevant document:
          5k relevant tokens alone:              82% accuracy
          5k relevant + 5k irrelevant noise:     78% accuracy
          5k relevant + 20k irrelevant noise:    71% accuracy
          5k relevant + 50k irrelevant noise:    63% accuracy

      Adding noise consistently degrades performance. This directly
      challenges the "just throw everything in" approach to RAG.


## 7.2  Measuring Prompt Token Efficiency

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SIGNAL-TO-TOKEN RATIO — A PRACTICAL FRAMEWORK                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      For a given task, token efficiency = useful information tokens ÷ total tokens.

      EXAMPLE — Code generation system prompt:

      INEFFICIENT VERSION (247 tokens):
      ──────────────────────────────────
      "You are an expert and highly skilled senior software engineer with
       many years of experience writing clean, readable, well-documented,
       and thoroughly tested code. You always follow best practices and
       coding standards. When writing code, please make sure that the code
       you write is clean, maintainable, and follows industry best practices.
       Please provide code that is well-commented and easy to understand.
       You are very knowledgeable about Python and you write excellent Python
       code that is pythonic and follows PEP 8 standards. Always write
       complete and working code."

      Information density: low — most tokens are filler that does not
      add precision. "expert senior engineer" and "excellent Python" are
      vague. The model cannot distinguish what matters.

      EFFICIENT VERSION (47 tokens):
      ─────────────────────────────
      "Write Python code. Follow PEP 8. Add docstrings. Include type hints.
       Write unit tests for all public functions. Raise specific exceptions."

      Same constraints, 81% fewer tokens. Each token carries a concrete,
      non-redundant constraint. Signal-to-token ratio: ~5× higher.

      At 100k calls/day, the difference is:
        Inefficient: 247 × 100k = 24.7M tokens/day
        Efficient:    47 × 100k =  4.7M tokens/day
        Savings: 20M tokens/day = $60/day at $3/MTok input pricing.


## 7.3  RAG Context Pollution

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE RETRIEVAL NOISE PROBLEM IN RAG SYSTEMS                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      In Retrieval-Augmented Generation, a retrieval step fetches K
      document chunks and inserts them before the question.

      A common mistake: retrieve K=10 chunks "to be safe", regardless
      of whether all 10 are relevant.

      Chunk relevance scores (cosine similarity to query, hypothetical):
        Chunk 1: 0.91  ← highly relevant
        Chunk 2: 0.87  ← highly relevant
        Chunk 3: 0.79  ← relevant
        Chunk 4: 0.71  ← marginally relevant
        Chunk 5: 0.65  ← borderline
        Chunk 6: 0.58  ← mostly irrelevant
        Chunk 7: 0.52  ← noise
        Chunk 8: 0.48  ← noise
        Chunk 9: 0.43  ← noise
        Chunk 10: 0.39 ← anti-correlated noise

      Chunks 6–10 are noise. Including them costs tokens and dilutes
      attention from chunks 1–3.

      BETTER STRATEGIES:
        • Use a relevance THRESHOLD rather than a fixed K
          (only include chunks with similarity > 0.65)
        • Use a re-ranker model to score chunks against the exact question
          before inserting — more expensive but higher precision
        • Set a HARD BUDGET: "inject at most 5,000 tokens of context"
          and stop retrieving once the budget is used, regardless of K
        • Compress chunks: extract only the relevant passage from each
          document (another LLM call) rather than injecting the full chunk


---


## 8. CONTEXT OVERFLOW STRATEGIES — Truncation, Summarisation, Chunking

When context exceeds the window limit — or approaches it — the application
must decide how to handle the overflow. Each strategy has specific trade-offs.


## 8.1  Detecting Overflow Before It Happens

    ╔══════════════════════════════════════════════════════════════════╗
    ║  PROACTIVE TOKEN COUNTING — BEFORE THE API CALL                  ║
    ╚══════════════════════════════════════════════════════════════════╝

      The right time to detect overflow is BEFORE sending the API request,
      not after receiving an error.

      Using tiktoken (Python) for pre-flight token counting:

        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")

        def count_tokens(messages, model="gpt-4"):
            tokens_per_message = 3  # <|im_start|>role\n ... <|im_end|>\n
            tokens_per_reply   = 3  # <|im_start|>assistant\n
            total = 0
            for msg in messages:
                total += tokens_per_message
                for key, value in msg.items():
                    total += len(enc.encode(value))
            total += tokens_per_reply
            return total

        MAX_CONTEXT = 128_000
        MAX_RESPONSE = 4_096
        INPUT_BUDGET = MAX_CONTEXT - MAX_RESPONSE  # = 123,904

        token_count = count_tokens(messages)
        if token_count > INPUT_BUDGET:
            # apply overflow strategy here


## 8.2  Strategy 1 — Truncation

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TRUNCATION — THREE VARIANTS                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      Truncation removes tokens from some part of the context to make it fit.
      The key question is: WHICH tokens to remove.

      ── VARIANT A: TAIL TRUNCATION (drop the oldest history) ─────────

        Remove the earliest conversation turns first.
        The system prompt and current message are always preserved.

        Before (130k tokens, over limit):
          [SYS 2k] [Turn1 0.5k] [Turn2 0.5k] ... [Turn100 0.5k] [Current 0.2k]

        After (drop turns 1–10, 5k saved):
          [SYS 2k] [Turn11 0.5k] ... [Turn100 0.5k] [Current 0.2k]

        PROS:  Simple. Preserves recent context which is usually most relevant.
        CONS:  May lose critical early context (initial task definition,
               user preferences stated at the start of session).

      ── VARIANT B: HEAD TRUNCATION (drop the newest history) ─────────

        Remove the most recent turns to make room.
        Rarely used in practice — usually the opposite of what you want.
        Occasionally useful when re-processing a document that has grown.

      ── VARIANT C: MIDDLE REMOVAL (keep first + last, remove middle) ──

        Preserve the beginning (framing, instructions) and the end
        (most recent turns) while removing the middle.

        [SYS 2k] [Turn1..5 retained] [TRUNCATED] [Turn95..100 retained] [Current]

        PROS:  Leverages the U-curve — you keep the primacy and recency zones.
        CONS:  Creates an artificial discontinuity. The model may behave
               oddly if it references content that was silently removed.

      ── WHAT TO NEVER TRUNCATE ────────────────────────────────────────

        • The system prompt (defines the model's behaviour for this call)
        • The current user message (the task at hand)
        • Tool schemas (if tools are needed for the response)
        • Any content the current question explicitly references


## 8.3  Strategy 2 — Summarisation

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ROLLING SUMMARISATION — THE CONVERSATION COMPRESSION PATTERN    ║
    ╚══════════════════════════════════════════════════════════════════╝

      Instead of dropping turns, compress old turns into a summary
      that retains key information in fewer tokens.

      ROLLING SUMMARY ARCHITECTURE:

        The context window contains:
          [SYS] [SUMMARY of turns 1–50] [Turns 51–60 raw] [Current message]
                 ↑                       ↑
                 compressed (500 tokens) kept verbatim (3,000 tokens)
                 summarises 15,000 tokens of prior conversation

        When turns 51–60 also overflow:
          Run a summarisation call (background):
            Input:  [Old summary] + [Turns 51–60]
            Output: [New summary covering turns 1–60]

          New context:
            [SYS] [SUMMARY of turns 1–60] [Turns 61–70 raw] [Current message]

      COMPRESSION RATIO:
        15,000 tokens of conversation → 500-token summary = 30:1 compression
        At 30:1, a 128k context window effectively handles ~3.8M tokens
        of total conversation history (at the cost of some detail loss).

      WHAT SUMMARIES LOSE:
        • Exact wording (cannot be directly quoted)
        • Low-priority details (the model decides what to summarise away)
        • Emotional tone and hedging ("I'm not sure but...")
        • Timestamps and conversational context

      WHAT SUMMARIES PRESERVE:
        • Key decisions and agreements
        • User preferences and constraints
        • Task progress and completed steps
        • Critical facts that were established


## 8.4  Strategy 3 — Chunking and RAG

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CHUNKING — SPLITTING LARGE DOCUMENTS FOR RETRIEVAL              ║
    ╚══════════════════════════════════════════════════════════════════╝

      For large document corpora that cannot fit in any context window,
      the answer is not to extend the window but to retrieve selectively.

      CHUNKING PIPELINE:

        1. INGEST DOCUMENTS
           Split each document into chunks of C tokens (e.g. C = 512).
           Overlap chunks by O tokens (e.g. O = 50) to prevent
           information loss at chunk boundaries.

           Document (10,000 tokens):
           [Chunk 1: 0–512] [Chunk 2: 462–974] [Chunk 3: 924–1436] ...
                             ↑ overlap           ↑ overlap
                             50-token overlap ensures no sentence is
                             cut off at a boundary

        2. EMBED CHUNKS
           Convert each chunk into a dense vector (embedding) using
           an embedding model (e.g. text-embedding-3-large).
           Store vectors in a vector database (Pinecone, Weaviate, pgvector).

        3. RETRIEVE AT QUERY TIME
           Convert the user query to an embedding.
           Find the K nearest chunk embeddings (cosine similarity).
           Inject those K chunks into the context window.

      CHUNK SIZE TRADE-OFFS:

        Small chunks (128–256 tokens):
          + Higher retrieval precision (each chunk is focused)
          + Less irrelevant content injected
          − May lose context that spans multiple chunks
          − Requires more chunks to cover a topic

        Large chunks (1024–2048 tokens):
          + Better local context — related sentences stay together
          + Fewer retrieval calls needed
          − Less precise: relevant paragraph mixed with irrelevant ones
          − Slower embedding and retrieval

        Common production choice: 512 tokens with 50-token overlap.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  OVERFLOW STRATEGY COMPARISON                                    ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌────────────────────┬──────────────┬──────────────┬────────────────┐
      │  Strategy          │  Loss type   │  Extra cost  │  Best for      │
      ├────────────────────┼──────────────┼──────────────┼────────────────┤
      │  Tail truncation   │  Old context │  None        │  Chatbots      │
      │  Middle removal    │  Mid-session │  None        │  Long sessions │
      │  Rolling summary   │  Detail loss │  LLM calls   │  Long sessions │
      │  Chunking + RAG    │  Precision   │  Embed + DB  │  Large corpora │
      │  Hierarchical sum  │  Detail loss │  Many calls  │  Deep archives │
      └────────────────────┴──────────────┴──────────────┴────────────────┘


---


## 9. LONG CONTEXT PERFORMANCE — Quality Degradation at Very Long Contexts

Having a long context window does not guarantee good performance at that length.
Empirical testing consistently shows that model quality degrades as context
approaches its limit, even for models nominally supporting that length.


## 9.1  The Needle-in-a-Haystack Test

    ╔══════════════════════════════════════════════════════════════════╗
    ║  NIAH — THE STANDARD LONG-CONTEXT BENCHMARK                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      The Needle-in-a-Haystack (NIAH) test is the standard evaluation
      for long-context performance. Setup:

        1. Fill the context with irrelevant but realistic text
           (e.g. Paul Graham essays, Wikipedia articles)
           from 1,000 to the full context limit

        2. Insert a "needle" — a specific factual statement —
           at a particular position in the haystack
           e.g. "The best pizza in New York is at Sal's on 5th Avenue"

        3. Ask the model: "What is the best pizza in New York?"

        4. Measure whether the model correctly retrieves the needle

        5. Repeat for all combinations of:
           • Context length (1k, 2k, 4k, ..., 128k tokens)
           • Needle position (0%, 10%, 25%, 50%, 75%, 90%, 100% of context)

      Results plotted as a 2D heatmap:
        X-axis: needle position (beginning to end of context)
        Y-axis: total context length (short to long)
        Colour: accuracy (green = 100%, red = 0%)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TYPICAL NIAH HEATMAP PATTERNS                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      STRONG model (e.g. Claude 3 Opus):

        Context length
        200k │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  ← mostly green, slight middle dip
        150k │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
        100k │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
         50k │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
             └──────────────────────────────────
               start           middle         end
               ●=pass  ○=fail

      TYPICAL model (before long-context fine-tuning):

        Context length
        128k │  ●●●●●●●●●○○○○○○○○○○○○○●●●●●●●  ← significant middle failures
         96k │  ●●●●●●●●●●●○○○○○○○●●●●●●●●●●●
         64k │  ●●●●●●●●●●●●●●○○●●●●●●●●●●●●●
         32k │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              └──────────────────────────────────
               start           middle         end
               The middle-of-long-context failure is clearly visible.

      WEAK model (long context tag but not well trained):

        Context length
        128k │  ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○  ← almost all fail at long lengths
         96k │  ●●○○○○○○○○○○○○○○○○○○○○○○○●●●●
         64k │  ●●●●●●○○○○○○○○○○●●●●●●●●●●●●●
         32k │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              └──────────────────────────────────
               A model that handles 32k well but fails badly at 128k


## 9.2  Why Performance Degrades at Long Context

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FOUR CAUSES OF LONG-CONTEXT DEGRADATION                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      1. TRAINING DATA SCARCITY
         ──────────────────────
         Most training documents are short. The distribution of training
         examples by length is heavily right-skewed:

         Length distribution of typical web-scraped training data:
           < 512 tokens:   ████████████████████████████████████  70%
           512–2k tokens:  ████████████████░░░░░░░░░░░░░░░░░░░░  25%
           2k–8k tokens:   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   4%
           8k–32k tokens:  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.9%
           > 32k tokens:   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.1%

         A model trained mostly on short sequences has seen very few examples
         of how to reason across 100k+ token contexts. It lacks "practice"
         at the task.

      2. POSITIONAL ENCODING EXTRAPOLATION
         ────────────────────────────────────
         Models are trained up to a certain sequence length. Positional
         encodings may behave unexpectedly at positions the model has not
         seen during training.
         RoPE and ALiBi address this (section 11) but extrapolation
         beyond training length always introduces some uncertainty.

      3. ATTENTION DILUTION
         ───────────────────
         With N tokens in context, each token receives a fraction 1/N of
         the total attention weight. At N=1,000 each token gets ~0.1%.
         At N=100,000 each token gets ~0.001%.
         Relevant tokens must "compete" against 100x more irrelevant tokens
         for a share of the total attention weight.

      4. ACCUMULATED APPROXIMATION ERROR
         ──────────────────────────────────
         Flash Attention and other approximations introduce tiny numerical
         errors. Across 32 layers and 100k+ tokens, these errors may
         compound. The model at very long context is operating with
         slightly less precision throughout.


## 9.3  The "Goldilocks Zone" of Context Utilisation

    ╔══════════════════════════════════════════════════════════════════╗
    ║  PRACTICAL GUIDANCE — WHERE QUALITY IS RELIABLE                  ║
    ╚══════════════════════════════════════════════════════════════════╝

      A model's context window has three zones of reliability:

      ┌──────────────────────────────────────────────────────────────────┐
      │  ZONE 1 — GOLDILOCKS (0%–60% of max context)                    │
      │  ████████████████████████████████████████                       │
      │  Reliable quality. Model behaves close to its best.              │
      │  Context is well within training distribution.                   │
      │                                                                  │
      │  ZONE 2 — CAUTION (60%–85% of max context)                      │
      │  ░░░░░░░░░░░░░░░░░░░░░░░░░                                      │
      │  Measurable quality degradation in some benchmarks.              │
      │  Lost-in-the-middle effect is significant.                       │
      │  Still usable with careful prompt engineering.                   │
      │                                                                  │
      │  ZONE 3 — WARNING (85%–100% of max context)                     │
      │  ░░░░░░░░░░░░░░░                                                 │
      │  Significant degradation for many models.                        │
      │  Retrieval of middle content is unreliable.                      │
      │  Consider overflow strategies before reaching this zone.         │
      └──────────────────────────────────────────────────────────────────┘

      Practical rule: for production applications requiring reliable recall,
      use at most 60% of the model's stated maximum context window.
      For GPT-4-turbo (128k): stay under ~75k tokens.
      For Claude 3 (200k): stay under ~120k tokens.

      Exception: state-of-the-art models (Claude 3 Opus, Gemini 1.5 Pro)
      have been specifically optimised for long context and perform well
      across more of their range.


---


## 10. POSITIONAL ENCODING — How the Model Knows Token Position

Transformer attention operates in parallel — all tokens are processed
simultaneously. Without some mechanism to communicate position, the model
would have no way to distinguish "The cat sat on the mat" from "mat the on
sat cat The." Every permutation would produce identical attention scores.

Positional encoding injects positional information into the token vectors
before they enter the attention layers.


## 10.1  The Fundamental Problem

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY POSITION IS INVISIBLE WITHOUT ENCODING                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      Attention score between tokens i and j:
        score(i, j) = Q_i · K_j

      Q_i is computed from the embedding of token i.
      K_j is computed from the embedding of token j.

      If token i and token j are the same word ("the"),
      their embeddings are the same regardless of their position.
      Therefore Q_i = Q_j and K_i = K_j.

      The attention score between any two identical words is identical
      regardless of where they appear in the sequence.

      Without positional encoding: the model cannot tell
        "The cat sat"   from   "sat cat The"
      because each "the", "cat", "sat" produces identical Q and K vectors.

      SOLUTION: add a position-dependent signal to each token's representation
      before computing Q, K, V. The signal must be:
        (a) unique per position, and
        (b) informative about the DISTANCE between positions, not just their absolute index.


## 10.2  Absolute Sinusoidal Encoding (Original Transformer)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SINUSOIDAL ENCODING — THE 2017 "ATTENTION IS ALL YOU NEED"      ║
    ║  APPROACH                                                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      For each position pos (0, 1, 2, ...) and each dimension i (0, 1, ..., d_model):

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

      Even dimensions: sine wave.   Odd dimensions: cosine wave.
      Each dimension oscillates at a different frequency.

      Intuition: think of it like a binary clock where each bit position
      oscillates at a different frequency. Position 0 = all zeros.
      Position 1 = first bit flips. Position 2 = second bit changes.
      The combination uniquely encodes any position up to 2^(d_model/2).

      WHAT THE ENCODING LOOKS LIKE (d_model=8, positions 0–5):

        pos  dim0      dim1      dim2      dim3      dim4      dim5     dim6     dim7
          0  0.000     1.000     0.000     1.000     0.000     1.000    0.000    1.000
          1  0.841     0.540     0.046     0.999     0.002     1.000    0.000    1.000
          2  0.909    -0.416     0.093     0.996     0.004     1.000    0.000    1.000
          3  0.141    -0.990     0.139     0.990     0.006     1.000    0.000    1.000
          4 -0.757    -0.654     0.185     0.983     0.008     1.000    0.000    1.000
          5 -0.959     0.284     0.231     0.973     0.010     1.000    0.000    1.000

      Low dimensions oscillate rapidly (distinguish adjacent positions).
      High dimensions oscillate very slowly (distinguish distant positions).
      Together they create a unique "fingerprint" for every position.

      PROPERTIES:
        • Deterministic — no parameters to learn
        • Works for any sequence length (theoretically unlimited)
        • Encodes relative positions: PE(pos+k) can be expressed as a
          linear function of PE(pos), so the model can learn relative
          distances

      LIMITATIONS:
        • Absolute encoding — generalises poorly to lengths not seen in training
        • Later models moved to relative position encodings for better extrapolation


## 10.3  Learned Absolute Embeddings

    ╔══════════════════════════════════════════════════════════════════╗
    ║  LEARNED POSITIONAL EMBEDDINGS — GPT-2, BERT                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      Instead of computing position encodings from a formula, maintain a
      learned embedding table of shape (max_length × d_model).

        Position 0   → learned vector p₀ = [0.23, -0.11, 0.67, ...]
        Position 1   → learned vector p₁ = [0.41, -0.09, 0.31, ...]
        ...
        Position N-1 → learned vector pₙ₋₁

      These vectors are randomly initialised and then trained end-to-end
      with the rest of the model.

      The input to the first transformer layer is the sum of:
        token_embedding(token_id) + positional_embedding(position)

      PROPERTIES:
        • Maximum flexibility — the model learns whatever positional
          signal is most useful
        • Works extremely well up to the training length

      CRITICAL LIMITATION:
        The table has exactly max_length rows. If max_length = 2048 (GPT-2),
        there are no positional embeddings defined for position 2049 or beyond.
        Attempting inference beyond max_length is undefined behaviour.

        This hard limit is why GPT-2 (1,024), GPT-3 (2,048), and early
        BERT models (512) had such small context windows. The table size
        is a hard architectural constraint baked into the model file.


## 10.4  Relative Position Encodings

The key insight motivating relative encodings: for most language tasks,
what matters is not the ABSOLUTE position of a token but its POSITION
RELATIVE TO OTHER TOKENS.

"The dog chased the cat" — knowing "cat" is at absolute position 4 is
less useful than knowing it is 3 positions after "dog."

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ABSOLUTE vs. RELATIVE — WHAT THE DIFFERENCE MEANS               ║
    ╚══════════════════════════════════════════════════════════════════╝

      ABSOLUTE ENCODING:
        Token "cat" at position 4 has encoding p₄.
        Token "cat" at position 40 has encoding p₄₀.
        p₄ ≠ p₄₀ — same word, different positions, different encoding.

        Question: does a model trained on "cat" at positions 1–2,048
        understand "cat" at position 2,049?
        Answer: not reliably — the model has never seen that absolute position.

      RELATIVE ENCODING:
        Instead of encoding position itself, encode the OFFSET between token pairs.
        "cat" at position 4 and "dog" at position 1 → offset = 3.
        "cat" at position 40 and "dog" at position 37 → offset = 3.
        Same offset → same relative position signal.

        A model trained with relative encodings generalises to any length,
        because the offsets it sees at training length are the same offsets
        it will see at any length (just more of them).

      Relative encodings are the foundation of RoPE and ALiBi (next section).


---


## 11. ROPE AND ALIBI ENCODINGS — Extending Beyond Training Length

The two dominant positional encoding schemes in modern LLMs are RoPE
(Rotary Position Embedding) and ALiBi (Attention with Linear Biases).
Both enable better generalisation beyond training length than absolute
encodings, but they achieve this through different mechanisms.


## 11.1  RoPE — Rotary Position Embedding

RoPE (Su et al., 2021) encodes position as a ROTATION applied directly to
the Query and Key vectors, rather than as an additive offset to the input embedding.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE ROPE MECHANISM — INTUITION                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      In standard attention, the dot product between Query and Key
      measures "how relevant is token j to token i?"

        score(i,j) = Q_i · K_j

      In RoPE, before computing this dot product, we rotate Q_i and K_j
      by angles proportional to their positions:

        Q̃_i = Rotate(Q_i, θ × i)
        K̃_j = Rotate(K_j, θ × j)

        score(i,j) = Q̃_i · K̃_j = Q_i · Rotate(K_j, θ × (j - i))
                                              ↑
                                     only the DIFFERENCE (j-i) matters
                                     not the absolute positions i and j

      The key algebraic identity: the dot product of two rotated vectors
      depends only on their RELATIVE ANGLE (j - i), not their absolute angles.

      This gives RoPE relative position sensitivity for free, without any
      modification to the attention formula.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ROPE — THE ROTATION IN 2D (SIMPLIFIED)                          ║
    ╚══════════════════════════════════════════════════════════════════╝

      For a 2-dimensional Q vector at position i:

        Q_i = [q₁, q₂]

      After RoPE rotation by angle θᵢ = i × θ_base:

        Q̃_i = [q₁ cos(θᵢ) - q₂ sin(θᵢ),
                q₁ sin(θᵢ) + q₂ cos(θᵢ)]

      For d_model dimensions, the vector is divided into d/2 pairs.
      Each pair (q₂ₖ, q₂ₖ₊₁) is rotated by a different base angle θₖ,
      creating a multi-frequency rotation scheme — exactly analogous to
      the multi-frequency sine/cosine scheme in absolute sinusoidal encoding,
      but applied as a rotation to Q and K rather than an addition to the input.

      Real models (d_model = 4096):
        The vector has 2048 pairs.
        Each pair rotates at a different frequency.
        Low-index pairs rotate fast (sensitive to nearby positions).
        High-index pairs rotate slowly (sensitive to long-range positions).

      This creates a continuous, smooth position signal that naturally
      extends to positions beyond training length.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ROPE PROPERTIES AND WHY IT BECAME THE DOMINANT CHOICE           ║
    ╚══════════════════════════════════════════════════════════════════╝

      ✓  Relative position sensitivity from absolute position encoding
         (the relative offset j-i is what determines the score)

      ✓  Integrates directly into the Q·K dot product — no additional
         parameters; no change to the attention formula

      ✓  Extrapolates beyond training length better than absolute encodings
         (relative offsets seen at training length generalise to longer offsets)

      ✓  Used by: LLaMA, LLaMA 2, LLaMA 3, Mistral, Gemma, Falcon, GPT-NeoX,
         and most modern open-source models

      ✗  Does not inherently favour recent tokens (unlike ALiBi)
         — this must be learned from data

      ✗  Some degradation beyond the training length (though much less
         than absolute encodings)


## 11.2  Context Window Extension with RoPE — YaRN and LongRoPE

Even RoPE degrades when used beyond its training length. Several techniques
extend RoPE to longer contexts without full retraining.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE EXTRAPOLATION PROBLEM WITH BASE ROPE                        ║
    ╚══════════════════════════════════════════════════════════════════╝

      RoPE uses a base frequency θ_base = 10,000 (original).
      At training length L, the angles seen range from 0 to L/10,000.
      At inference with length 2L, angles from L/10,000 to 2L/10,000
      are seen for the first time — slight degradation occurs.

      POSITION INTERPOLATION (Chen et al., 2023):
        Instead of extrapolating to new positions, COMPRESS the position
        indices to fit within the training range.

        Normal:        positions 0, 1, 2, ..., 32,767  (at 32k context)
        Interpolated:  positions 0, 0.5, 1, ..., 16,383 (for 64k context)
                        scaled by factor = L_train / L_new

        This maps every new position to a fractional position within
        the training range. The model has seen nearby fractional positions
        during training (the "between-integer" space is smooth).

        After 1,000 steps of fine-tuning at the extended length,
        quality at 64k becomes comparable to quality at 32k before extension.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  YARN — YET ANOTHER ROPE EXTENSION METHOD                        ║
    ╚══════════════════════════════════════════════════════════════════╝

      YaRN (Peng et al., 2023) improves on position interpolation by:

        1. FREQUENCY-DEPENDENT SCALING
           Different frequency components of RoPE are scaled differently.
           Low-frequency components (long-range) are interpolated.
           High-frequency components (short-range) are left unchanged.
           This preserves local positional precision while extending range.

        2. ATTENTION TEMPERATURE SCALING
           At long context, attention scores naturally become smaller
           (more tokens to distribute probability over). YaRN introduces
           a temperature correction to compensate.

        Results: LLaMA models extended from 4k → 128k with YaRN and
        a small fine-tuning run (0.1% of pretraining compute).

      LONGROPE (Microsoft, 2024):
        Applies non-uniform position interpolation — different groups
        of RoPE dimensions receive different rescaling factors.
        Achieves 2M token context for Phi-3 models.


## 11.3  ALiBi — Attention with Linear Biases

ALiBi (Press et al., 2021) takes a fundamentally different approach:
rather than modifying the token representations, ALiBi adds a
DISTANCE-PROPORTIONAL PENALTY directly to the attention scores.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE ALIBI MECHANISM                                              ║
    ╚══════════════════════════════════════════════════════════════════╝

      Standard attention score:
        score(i, j) = Q_i · K_j / √d_k

      ALiBi attention score:
        score(i, j) = Q_i · K_j / √d_k  −  m × (i − j)
                                           ↑
                                      linear penalty proportional to distance

      Where:
        m = head-specific slope (different for each attention head)
        (i − j) = distance from current token i to past token j
                  (always positive since j ≤ i for causal attention)

      Effect: every attention score is reduced by a penalty linear in distance.
      Tokens 1 step away: penalty = m × 1
      Tokens 10 steps away: penalty = m × 10
      Tokens 100 steps away: penalty = m × 100

      Different heads have different slopes m (ranging from 2⁻¹ to 2⁻⁸
      for 8-head models). Some heads attend locally (high m = steep penalty
      = strong recency bias). Others attend more globally (low m = gentle
      penalty = can still attend to distant tokens).

    ╔══════════════════════════════════════════════════════════════════╗
    ║  VISUALISING THE ALIBI PENALTY                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      For a head with slope m = 0.25, generating token at position 10:

        Token position  Distance  ALiBi penalty  Effect on score
        position 9       1        0.25×1 = 0.25  small penalty
        position 8       2        0.25×2 = 0.50
        position 5       5        0.25×5 = 1.25
        position 0      10        0.25×10= 2.50  large penalty — distant token less likely attended

      For a head with slope m = 2.0 (steep):

        Token position  Distance  ALiBi penalty
        position 9       1        2.00           aggressive recency bias
        position 5       5       10.00           extremely discouraged
        position 0      10       20.00           essentially zeroed out

      The slopes are chosen so that different heads cover different
      range scales — local coherence and long-range dependencies are
      handled by different specialised heads.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ALIBI vs. ROPE — KEY COMPARISON                                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌─────────────────────┬─────────────────────────┬───────────────────────────┐
      │  Property           │  RoPE                   │  ALiBi                    │
      ├─────────────────────┼─────────────────────────┼───────────────────────────┤
      │  Mechanism          │  Rotation of Q/K vectors│  Additive bias on scores  │
      │  Position type      │  Relative (via rotation)│  Relative (direct offset) │
      │  Parameters         │  None (formula-based)   │  None (formula-based)     │
      │  Extrapolation      │  Good (needs tuning)    │  Excellent by design      │
      │  Training length    │  Must be set in advance │  Trained at short, infers │
      │                     │                         │  at long naturally        │
      │  Recency bias       │  Learned from data      │  Baked in by design       │
      │  Quality at 4× ext. │  Good with fine-tuning  │  Good without fine-tuning │
      │  Used in            │  LLaMA, Mistral, Gemma  │  MPT, BLOOM, Falcon       │
      └─────────────────────┴─────────────────────────┴───────────────────────────┘

      ALiBi's extrapolation advantage: because the penalty is purely a function
      of distance, there is no fixed training-length ceiling. A model trained
      on 2,048 tokens can naturally infer on 10,000+ tokens — the penalty
      formula simply produces larger numbers for larger distances.

      ALiBi's limitation: the recency bias is hard-baked. Long-range attention
      is always penalised. For tasks that genuinely need global attention across
      very long sequences (e.g. finding a specific fact at position 0 while
      generating at position 100,000), ALiBi-based models may be systematically
      disadvantaged compared to RoPE-based models with context extension.


## 11.4  Sinusoidal vs. Learned vs. RoPE vs. ALiBi — Complete Comparison

    +────────────────────┬───────────────┬──────────────┬───────────────┬──────────────+
    │  Property          │  Sinusoidal   │  Learned     │  RoPE         │  ALiBi       │
    +────────────────────┼───────────────┼──────────────┼───────────────┼──────────────+
    │  Introduced        │  2017         │  2018 (BERT) │  2021         │  2021        │
    │  Type              │  Absolute     │  Absolute    │  Relative     │  Relative    │
    │  Where applied     │  Input embed  │  Input embed │  Q, K vectors │  Attn scores │
    │  Extra parameters  │  None         │  Yes (L×d)   │  None         │  None        │
    │  Training length   │  Soft limit   │  HARD LIMIT  │  Soft limit   │  No limit    │
    │  Extrapolation     │  Poor         │  NONE        │  Good w/ tune │  Excellent   │
    │  Recency bias      │  No           │  Learned     │  Learned      │  Yes (built) │
    │  Used in           │  Original     │  GPT-2,BERT  │  LLaMA 1/2/3  │  BLOOM, MPT  │
    │                    │  Transformer  │              │  Mistral,Gemma│  Falcon      │
    +────────────────────┴───────────────┴──────────────┴───────────────┴──────────────+


---


## Glossary

    +──────────────────────────┬────────────────────────────────────────────────────────+
    │  Term                    │  Definition                                             │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  ALiBi                   │  Attention with Linear Biases — positional encoding    │
    │                          │  that adds a distance-proportional penalty to each     │
    │                          │  attention score. Enables natural extrapolation.        │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Attention mask          │  A binary matrix that sets certain attention scores to  │
    │                          │  -∞ before softmax, preventing attention to padding     │
    │                          │  tokens or future tokens (causal mask).                 │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Causal attention        │  Attention where token i can only attend to tokens      │
    │                          │  0 through i. Required for auto-regressive generation. │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Context window          │  The maximum number of tokens (prompt + response) a    │
    │                          │  model can process in one inference call. The model's   │
    │                          │  entire working memory for that call.                   │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Dilated attention       │  Attention pattern where attended tokens are spaced     │
    │                          │  uniformly across the sequence (every d-th token),     │
    │                          │  providing long-range coverage at sub-quadratic cost.   │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Flash Attention         │  An algorithm that computes exact attention without     │
    │                          │  materialising the N×N score matrix, using tiled SRAM  │
    │                          │  computation. O(N) memory, same O(N²) compute.         │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Global token            │  A token that attends to and is attended by every      │
    │                          │  token in the sequence, used in sparse attention       │
    │                          │  patterns (Longformer, BigBird) to maintain a global   │
    │                          │  summary representation.                               │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  KV cache                │  Per-token Key and Value matrices stored in VRAM       │
    │                          │  to avoid recomputing prior tokens during generation.  │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Learned positional emb. │  A trainable embedding table of shape (max_len × d).  │
    │                          │  Hard limit: no embedding defined beyond max_len.      │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  LongRoPE                │  Microsoft's non-uniform RoPE rescaling to extend      │
    │                          │  context to 2M tokens with fine-tuning.                │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Lost in the middle      │  Empirical finding that LLMs retrieve information at  │
    │                          │  the beginning and end of long contexts more           │
    │                          │  accurately than information in the middle (U-curve).  │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  max_tokens              │  API parameter capping the number of tokens the model  │
    │                          │  will generate. Reserves space in the context window.  │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  NIAH                    │  Needle in a Haystack — benchmark measuring retrieval  │
    │                          │  accuracy across all positions in a long context.      │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Noise dilution          │  Performance degradation caused by irrelevant tokens   │
    │                          │  competing for attention with relevant ones.           │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Parametric memory       │  Knowledge encoded in model weights during training.   │
    │                          │  Shared across all users; cannot be updated per-call.  │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Position interpolation  │  Extending RoPE context by compressing position        │
    │                          │  indices to fit within the training range.             │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Primacy effect          │  The tendency for tokens at the beginning of context   │
    │                          │  to receive stronger attention than middle tokens.     │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  RAG                     │  Retrieval-Augmented Generation — injecting relevant   │
    │                          │  retrieved documents into the context at query time.   │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Recency effect          │  The tendency for tokens near the generation point     │
    │                          │  (end of context) to receive stronger attention.       │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  RoPE                    │  Rotary Position Embedding — encodes position as a    │
    │                          │  rotation of Q/K vectors. Relative by construction.   │
    │                          │  The dominant positional encoding in modern LLMs.     │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Signal-to-token ratio   │  The fraction of context window tokens that directly  │
    │                          │  inform the model's output vs. filler/noise tokens.   │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Sinusoidal encoding     │  Original 2017 positional encoding using sine/cosine   │
    │                          │  functions at different frequencies per dimension.     │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Sliding window attn.    │  Each token attends only to the W nearest prior       │
    │                          │  tokens, reducing attention cost from O(N²) to O(NW). │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  Token budget            │  The allocation of context window tokens across        │
    │                          │  system prompt, history, retrieved context, and        │
    │                          │  response reserve.                                     │
    +──────────────────────────┼────────────────────────────────────────────────────────+
    │  YaRN                    │  Yet Another RoPE extensioN — frequency-dependent     │
    │                          │  scaling and attention temperature correction for      │
    │                          │  extending RoPE to 128k+ tokens with minimal tuning.  │
    +──────────────────────────┴────────────────────────────────────────────────────────+


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