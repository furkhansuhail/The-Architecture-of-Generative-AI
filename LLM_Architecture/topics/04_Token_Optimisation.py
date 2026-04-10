"""
Tutorial Template — Automation & Infrastructure
=================================================
Copy this file, rename it, and fill in the sections.

The TOPIC_NAME and CATEGORY are parsed by the app.
OPERATIONS entries can specify "language" as "bash", "yaml", "python", "json", etc.
"""

TOPIC_NAME = "LLM - Token Optimisation"
DISPLAY_NAME = "LLM - Token Optimisation"
ICON         = "💰"
SUBTITLE     = "Spending your context budget wisely — compression, caching, and cost control"
CATEGORY = "LLM Architecture"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = r"""

Order:

    Module_Concepts · Token Optimisation
          ├── token_budget_mental_model       Context as a finite budget to spend wisely
          ├── prompt_compression              Removing redundancy without losing meaning
          ├── system_prompt_efficiency        Writing tight system prompts — what to cut
          ├── few_shot_token_cost             The token price of examples
          ├── history_pruning_strategies      Which messages to drop first
          ├── conversation_summarisation      Compressing old turns before they're dropped
          ├── document_chunking               Splitting long docs into retrieval-sized pieces
          ├── structured_output_efficiency    JSON vs. prose — output format token cost
          ├── chain_of_thought_cost           Why CoT is more accurate but more expensive
          ├── caching_prompt_tokens           Prefix caching — pay once for repeated prompts
          ├── token_budget_for_responses      max_tokens — setting it without truncating
          └── counting_tokens_before_sending  Using tiktoken before an API call


## 1. TOKEN BUDGET MENTAL MODEL — Context as a Finite Budget to Spend Wisely

Every LLM API call consumes a fixed, finite resource: the context window.
The correct mental model is not "how long can my prompt be?" but "how do I
get the most value from the tokens I spend?"

This section establishes the budget framework that all other optimisation
decisions in this module flow from.


## 1.1  The Budget Metaphor

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CONTEXT AS A SPENDING BUDGET                                    ║
    ╚══════════════════════════════════════════════════════════════════╝

      You have N tokens to spend on each API call.
      Every token you spend on input is a token not available for response.
      Every wasted token is compute and money with zero return.

      A 128,000-token context window is a budget.
      The question is not "does my content fit?" but:

        "What is the return on investment of every token I include?"

      HIGH-ROI TOKENS:
        • The specific fact needed to answer this question
        • The exact format example that prevents re-formatting cycles
        • The constraint that eliminates an entire failure mode
        • The most relevant retrieved document

      LOW-ROI TOKENS:
        • Verbose pleasantries and preambles
        • Repeated instructions already stated earlier
        • Irrelevant retrieved documents
        • Overly elaborate descriptions of things the model already knows
        • System prompt boilerplate that adds no information

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE INPUT/OUTPUT ASYMMETRY                                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      Input tokens and output tokens have fundamentally different economics.

      INPUT TOKENS (prompt):
        • Cost: $1–$15 per million tokens (model-dependent)
        • Generated: by you, fully controllable
        • ROI lever: every input token you REMOVE costs nothing
          and may improve quality by reducing noise

      OUTPUT TOKENS (completion):
        • Cost: $3–$75 per million tokens (3–5× more than input)
        • Generated: by the model, partially controllable via max_tokens
        • ROI lever: shorter precise outputs are cheaper AND often better

      This asymmetry has two practical implications:

        1. INVEST in high-quality input — a well-crafted system prompt
           that saves one round-trip of clarification pays for itself many
           times over in saved output tokens.

        2. SET TIGHT OUTPUT LIMITS — for structured tasks (classification,
           extraction, yes/no), the model rarely needs 4,096 output tokens.
           Setting max_tokens=20 for a sentiment classifier costs nothing
           in quality and saves substantial output token spend.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE ROI FRAMEWORK — EVALUATING EVERY TOKEN BLOCK                ║
    ╚══════════════════════════════════════════════════════════════════╝

      Before including any block of content in your prompt, ask:

        1. NECESSITY: Does removing this block change the model's output?
           If no → remove it. It is pure waste.

        2. PRECISION: Can this block be said in fewer tokens without
           losing the essential constraint or fact?
           If yes → compress it.

        3. POSITION: Is this block in the right place?
           (High-importance content: beginning or end. See Context Window module.)

        4. CACHEABILITY: Is this block identical across most/all requests?
           If yes → move it to the prefix-cached section (section 10).

      Applying this framework to a production system prompt can
      typically reduce its token count by 30–60% without any quality loss.


## 1.2  The Token Budget Accounting Sheet

    ╔══════════════════════════════════════════════════════════════════╗
    ║  MAPPING YOUR TOKEN SPEND — A WORKED AUDIT                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      HYPOTHETICAL: A customer support assistant (128k context limit)

      Component                         Tokens   % of budget   Optimisable?
      ─────────────────────────────────────────────────────────────────────
      System prompt (persona + rules)    4,200       3.3%       Yes — target 1,500
      Tool schemas (8 tools)             1,400       1.1%       Yes — selective injection
      Knowledge base (3 retrieved docs) 12,000       9.4%       Yes — rerank + trim
      Conversation history (20 turns)    5,800       4.5%       Yes — prune + summarise
      Current user message                 150       0.1%       No
      Response reserve (max_tokens)      2,000       1.6%       Maybe — task-dependent
      ─────────────────────────────────────────────────────────────────────
      TOTAL USED                        25,550      19.9%
      UNUSED                           102,450      80.1%
      ─────────────────────────────────────────────────────────────────────

      For this application, 80% of the context window is unused —
      but the 20% being used still has significant optimisation potential.

      Optimising from 25,550 to 15,000 tokens (a realistic target):
        • Saves ~10,500 tokens per call
        • At $3/MTok and 500k calls/month: saves $15,750/month
        • Reduces pre-fill time → lower TTFT
        • Increases cache hit rates (smaller, more consistent prefix)


---


## 2. PROMPT COMPRESSION — Removing Redundancy Without Losing Meaning

Prompt compression is the practice of conveying the same information in fewer
tokens. Done well it maintains or improves model performance; done badly it
loses critical constraints and degrades output quality.


## 2.1  Types of Redundancy in Prompts

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SIX CATEGORIES OF TOKEN WASTE                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      1. VERBAL FILLER — words that carry no information
         ─────────────────────────────────────────────────
         Before: "I would like you to please kindly ensure that you always
                  remember to use formal language in your responses."
         After:  "Use formal language."
         Saved:  ~22 tokens → 4 tokens  (82% reduction)

      2. REDUNDANT QUALIFICATION — hedges and meta-commentary
         ──────────────────────────────────────────────────────
         Before: "When answering, please note that it is important to make
                  sure that you take care to be accurate and to avoid making
                  things up or hallucinating."
         After:  "Be accurate. Do not fabricate information."
         Saved:  ~33 tokens → 8 tokens  (76% reduction)

      3. RESTATING THE OBVIOUS — instructions the model follows by default
         ──────────────────────────────────────────────────────────────────
         Before: "Please use proper grammar and punctuation and write in
                  complete sentences with correct spelling."
         After:  [delete entirely — default model behaviour]
         Saved:  ~20 tokens → 0 tokens  (100% reduction)

      4. REPETITION — the same constraint stated multiple times
         ─────────────────────────────────────────────────────
         Before: "Always respond in English. Make sure your answers are in
                  English. The user expects English responses."
         After:  "Respond in English."
         Saved:  ~22 tokens → 4 tokens  (82% reduction)

      5. VERBOSE EXAMPLES — examples with unnecessary scaffolding
         ──────────────────────────────────────────────────────────
         Before: "For example, if someone asks you 'What is the capital of
                  France?' you might respond with something like 'The capital
                  of France is Paris, which is located in northern France.'"
         After:  "Q: Capital of France? A: Paris."
         Saved:  ~47 tokens → 9 tokens  (81% reduction)

      6. PREAMBLE BLOAT — unnecessary context-setting before instructions
         ──────────────────────────────────────────────────────────────────
         Before: "You are an advanced AI assistant developed to help users
                  with a wide variety of tasks. You have been trained on a
                  large corpus of text and have extensive knowledge about
                  many topics. Your role is to assist users by..."
         After:  "You are a helpful assistant."
         Saved:  ~47 tokens → 6 tokens  (87% reduction)


## 2.2  The Active Compression Technique

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SEVEN-STEP COMPRESSION WORKFLOW                                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      STEP 1 — Count the baseline
        Tokenise the current prompt. Record the token count.
        This is your compression target.

      STEP 2 — Delete filler passes
        Read each sentence. Ask: "If this sentence were missing, would
        the model behave differently?" Remove every sentence where the
        answer is no.

      STEP 3 — Consolidate repetition
        Find every constraint that is stated more than once.
        Keep the sharpest statement. Delete all others.

      STEP 4 — Convert prose to lists
        Prose:  "You should always format your responses with a clear
                 introduction, followed by the main content, and ending
                 with a concise summary."
        List:   "Format: intro → content → summary."
        The list version is typically 40–60% shorter.

      STEP 5 — Convert lists to structured notation
        List:   "- Do not use jargon.\n- Do not use acronyms.\n- Do not
                  use technical terms without defining them first."
        Compact: "No jargon. No undefined acronyms. Define technical terms."
        Saves the bullet characters and line breaks.

      STEP 6 — Remove default behaviour instructions
        LLMs already do these by default for well-aligned models:
          • Write grammatically
          • Avoid obvious falsehoods
          • Answer the question asked
          • Use paragraphs and punctuation
        Instructions for these consume tokens without changing behaviour.

      STEP 7 — Verify quality is maintained
        Test the compressed prompt against 10–20 representative inputs.
        Verify the model output quality is unchanged.
        If specific edge cases fail, restore only the instruction that fixes them.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  AUTOMATED COMPRESSION — LLMLINGUA AND SELECTIVE CONTEXT         ║
    ╚══════════════════════════════════════════════════════════════════╝

      For large retrieved documents where manual compression is impractical,
      automated compression tools estimate the importance of each token and
      remove the least important ones.

      LLMLingua (Microsoft) approach:
        1. Score each token's "perplexity" given surrounding context
           Low perplexity = predictable from context = removable
           High perplexity = surprising = important to keep

        2. Drop the lowest-perplexity tokens until the target compression
           ratio is reached (e.g. 4:1 — 4 tokens → 1 token average)

        3. The compressed text is shorter but conveys the same key facts

      Example:
        Original (41 tokens):
          "The patient was admitted to the hospital on March 15th with
           symptoms including fever, cough, and difficulty breathing. The
           attending physician noted elevated white blood cell counts."

        Compressed 2:1 (21 tokens):
          "Patient admitted March 15 fever cough breathing difficulty.
           Physician noted elevated white blood cells."

        The grammar degrades but the facts are preserved.
        LLMs can understand and reason about compressed text reliably.

      WHEN TO USE AUTOMATED COMPRESSION:
        ✓  Large retrieved documents (>1,000 tokens per chunk)
        ✓  Historical conversation turns being summarised
        ✓  News articles or web pages injected as context
        ✗  System prompts (manual compression is better — more precise)
        ✗  Structured data (JSON/tables — lossless compression matters more)
        ✗  Code (syntax must be preserved exactly)


---


## 3. SYSTEM PROMPT EFFICIENCY — Writing Tight System Prompts

The system prompt is sent with every API call and is usually the largest
static cost component. It is also the easiest to optimise because it is
entirely under the developer's control.


## 3.1  What the System Prompt Actually Needs to Contain

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE MINIMUM VIABLE SYSTEM PROMPT                                ║
    ╚══════════════════════════════════════════════════════════════════╝

      The model already knows, by default, how to:
        • Write grammatically and coherently
        • Answer questions helpfully
        • Format output with headers and lists when appropriate
        • Respond in the user's language
        • Avoid obviously false or harmful content
        • Be polite and professional

      The system prompt should contain ONLY information the model
      does not already know and cannot infer:

        MUST INCLUDE:
          ✓  Role identity that meaningfully changes behaviour
             "You are a tax attorney specialising in US corporate law."
             NOT: "You are a helpful AI assistant."  (default behaviour)

          ✓  Constraints specific to this deployment
             "Never discuss competitor products."
             "All prices quoted must include VAT."

          ✓  Output format requirements that deviate from default
             "Respond in JSON only. No explanatory prose."
             "Always respond in Spanish regardless of user language."

          ✓  Domain knowledge not in the model's training
             "Our product version 4.2 changed the auth flow. Users now..."

          ✓  Escalation and scope limits
             "You cannot process refunds. Direct all refund requests to..."

        EXCLUDE:
          ✗  "You are an advanced AI..."  — adds nothing, costs ~5 tokens
          ✗  "Be helpful and accurate."   — default behaviour, ~5 tokens
          ✗  "Think step by step."        — better as per-request instruction
          ✗  "Use proper grammar."        — default behaviour
          ✗  "Do not make things up."     — reduces to one meaningful phrase


## 3.2  Before and After — A Full System Prompt Audit

    ╔══════════════════════════════════════════════════════════════════╗
    ║  REAL-WORLD SYSTEM PROMPT COMPRESSION — CUSTOMER SUPPORT         ║
    ╚══════════════════════════════════════════════════════════════════╝

      BLOATED VERSION (287 tokens):
      ──────────────────────────────
      "You are a highly skilled and knowledgeable customer support specialist
       for Acme Corp. You have been trained extensively on our products and
       services and you are here to help our valued customers with any
       questions or issues they might have. You should always be polite,
       friendly, and professional in your interactions. When responding to
       customers, please make sure that you are accurate and helpful. Do not
       make up information. If you do not know the answer to something, say
       so clearly and offer to escalate to a human agent.

       You should always respond in a warm and friendly tone. Use the
       customer's name when you know it. Keep your responses concise but
       complete — do not give overly long responses. Do not use jargon that
       customers might not understand. Avoid technical language unless the
       customer seems technically sophisticated.

       You are not able to process refunds or account changes directly.
       For these requests, always direct the customer to our support team
       at support@acme.com or via the in-app help menu. Never discuss
       pricing changes or product roadmap items. If asked about these,
       explain that you are not able to provide that information."

      COMPRESSED VERSION (71 tokens):
      ──────────────────────────────────
      "Acme Corp customer support.

       Rules:
       - Cannot process refunds or account changes → direct to support@acme.com
       - Do not discuss pricing changes or product roadmap
       - If unsure: say so and offer human escalation

       Tone: warm, concise, no unexplained jargon."

      Savings: 216 tokens per call (75% reduction).
      At $3/MTok, 100k calls/day: $64.80/day saved = $1,944/month.
      Quality: identical or better (tighter constraints, less noise).


## 3.3  The Diminishing Returns of Instruction Length

    ╔══════════════════════════════════════════════════════════════════╗
    ║  DOES A LONGER SYSTEM PROMPT PRODUCE BETTER OUTPUT?              ║
    ╚══════════════════════════════════════════════════════════════════╝

      Empirical pattern (observed across many production deployments):

      Instructions   Output quality
         50 tokens   ████████████████████░░░░  good baseline
        200 tokens   ███████████████████████░  near-peak quality
        500 tokens   ████████████████████████  peak quality
      1,000 tokens   ███████████████████████░  marginal decline (noise)
      2,000 tokens   ██████████████████████░░  measurable decline
      5,000 tokens   ████████████████████░░░░  quality degrades
     10,000 tokens   █████████████████░░░░░░░  significant degradation
                      ↑ returns diminish; beyond ~500 tokens, more is often worse

      WHY MORE INSTRUCTIONS HURTS BEYOND A THRESHOLD:
        • Contradictions: long prompts often contain subtly contradictory
          instructions. The model must weight and reconcile them — and may
          weight the wrong one.
        • Dilution: important instructions are buried among less important ones.
          The model cannot reliably identify which rule matters most.
        • Middle-content decay: instructions in the middle of long system prompts
          are attended to less than instructions at the beginning (see Context
          Window module, section 4).

      PRACTICAL GUIDELINE:
        Target 100–500 tokens for a system prompt.
        If you exceed 500 tokens, audit for compression opportunities first.
        Only go above 500 tokens if every additional token encodes a genuine
        behavioural constraint not captured by anything shorter.


---


## 4. FEW-SHOT TOKEN COST — The Token Price of Examples

Few-shot prompting — including input/output examples before the actual
question — is one of the most reliable ways to improve model output quality.
It is also one of the most expensive techniques per quality improvement.
This section quantifies the cost and defines when it pays off.


## 4.1  What Few-Shot Examples Actually Cost

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TOKEN COST OF DIFFERENT EXAMPLE TYPES                           ║
    ╚══════════════════════════════════════════════════════════════════╝

      CLASSIFICATION EXAMPLE (sentiment analysis):
        Input:  "The product arrived damaged."
        Output: "negative"
        Cost:   ~11 tokens per example

      EXTRACTION EXAMPLE (entity extraction):
        Input:  "John Smith signed the NDA on March 15, 2024."
        Output: {"person": "John Smith", "document": "NDA", "date": "2024-03-15"}
        Cost:   ~35 tokens per example

      TRANSFORMATION EXAMPLE (rewriting formal → casual):
        Input:  "We regret to inform you that your application has been declined."
        Output: "Sorry, your application didn't make it this time."
        Cost:   ~30 tokens per example

      REASONING EXAMPLE (multi-step):
        Input:  "A train leaves at 9am travelling at 80mph. Another leaves..."
        Output: [full reasoning chain + answer]
        Cost:   100–500 tokens per example depending on complexity

      ┌────────────────────────────────────────────────────────────────┐
      │  At 5 examples of each type:                                   │
      │  Classification:  5 × 11  =    55 tokens  (negligible)        │
      │  Extraction:      5 × 35  =   175 tokens  (small)             │
      │  Transformation:  5 × 30  =   150 tokens  (small)             │
      │  Reasoning:       5 × 200 = 1,000 tokens  (significant)       │
      └────────────────────────────────────────────────────────────────┘


## 4.2  The Quality-per-Token Return of Examples

    ╔══════════════════════════════════════════════════════════════════╗
    ║  DIMINISHING RETURNS ON EXAMPLE COUNT                            ║
    ╚══════════════════════════════════════════════════════════════════╝

      Accuracy improvement from adding examples (typical classification task):

        0 examples (zero-shot):  ████████████████████░░░░  72% accuracy
        1 example (one-shot):    ███████████████████████░  82% accuracy  +10pp
        2 examples:              ████████████████████████  85% accuracy  + 3pp
        3 examples:              ████████████████████████  87% accuracy  + 2pp
        5 examples:              ████████████████████████  88% accuracy  + 1pp
        8 examples:              ████████████████████████  89% accuracy  + 1pp
       12 examples:              ████████████████████████  89% accuracy  + 0pp

      The marginal return of each additional example DIMINISHES rapidly.
      The first example produces the largest single improvement.
      Beyond 3–5 examples, improvements are marginal for most tasks.

      COST vs. QUALITY ANALYSIS:

        1 example:   11 tokens → +10pp accuracy  →  1.1 tokens per percentage point
        2 examples:  22 tokens → +13pp accuracy  →  1.7 tokens per percentage point
        5 examples:  55 tokens → +16pp accuracy  →  3.4 tokens per percentage point
        12 examples: 132 tokens → +17pp accuracy → 7.8 tokens per percentage point

      Cost efficiency is HIGHEST at 1–2 examples and falls monotonically.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE THREE-WAY TRADE-OFF: ZERO-SHOT vs. FEW-SHOT vs. FINE-TUNING║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌─────────────────┬───────────────┬───────────────────┬──────────────────┐
      │  Approach       │  Per-call cost│  One-time cost    │  Best for        │
      ├─────────────────┼───────────────┼───────────────────┼──────────────────┤
      │  Zero-shot      │  Lowest       │  None             │  General tasks   │
      │                 │  0 example    │                   │  High variety    │
      │                 │  tokens       │                   │                  │
      ├─────────────────┼───────────────┼───────────────────┼──────────────────┤
      │  Few-shot       │  Medium       │  Prompt           │  Format-specific │
      │  (3–5 examples) │  ~100–500 tok │  engineering only │  tasks, moderate │
      │                 │  per call     │                   │  volume          │
      ├─────────────────┼───────────────┼───────────────────┼──────────────────┤
      │  Fine-tuned     │  Lowest       │  $500–$10,000+    │  High-volume     │
      │  model          │  0 example    │  training + data  │  tasks, stable   │
      │                 │  tokens, but  │  labelling        │  format, quality │
      │                 │  smaller model│                   │  critical        │
      │                 │  may suffice  │                   │                  │
      └─────────────────┴───────────────┴───────────────────┴──────────────────┘

      BREAK-EVEN CALCULATION for fine-tuning vs. few-shot:

        Assume: 5 examples at 200 tokens each = 1,000 tokens per call
                Input price: $3/MTok
                Fine-tuning cost: $2,000

        Few-shot cost per call:  1,000 × $3/1,000,000 = $0.003
        Fine-tuning payoff:       $2,000 / $0.003 = 666,667 calls

        If you make more than ~670k calls at this configuration, fine-tuning
        pays back its investment and becomes cheaper than few-shot prompting.


## 4.3  Strategic Example Placement

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHERE TO PUT FEW-SHOT EXAMPLES FOR MAXIMUM EFFECT               ║
    ╚══════════════════════════════════════════════════════════════════╝

      Recall from the Context Window module: recency bias means examples
      at the END of the prompt (just before the actual question) are more
      influential than examples at the beginning.

      OPTIMAL FEW-SHOT STRUCTURE:

        [System prompt — task definition, constraints]
        [Retrieved context if any]
        [Examples — ordered last-before-question]
        [Current question]

      Example placement:
        Example 1  (weakest effect — will be read first, attended to less)
        Example 2
        Example 3  (strongest effect — immediately before the question)
        ────────────────────────
        CURRENT QUESTION HERE

      For 3 examples, place the most representative or hardest-to-get-right
      example LAST — it will be most influential on the generation.

      PRACTICAL TRICK — DYNAMIC EXAMPLE SELECTION:
        Instead of hardcoding the same examples for every call, select
        examples at query time based on similarity to the current input.
        Use an embedding search to find the 2–3 stored examples most
        semantically similar to the current query.
        These "nearest-neighbour examples" often outperform static examples
        at equal or lower token cost.


---


## 5. HISTORY PRUNING STRATEGIES — Which Messages to Drop First

In multi-turn conversations, history grows unbounded while the context window
stays fixed. History pruning decides which messages to keep and which to discard.


## 5.1  The History Growth Problem

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW QUICKLY HISTORY FILLS THE BUDGET                            ║
    ╚══════════════════════════════════════════════════════════════════╝

      Assume:
        System prompt + tools: 3,000 tokens (fixed)
        Average turn: 300 tokens (150 user + 150 assistant)
        Response reserve: 2,000 tokens
        Context limit: 128,000 tokens

      Available for history: 128,000 − 3,000 − 2,000 = 123,000 tokens
      Maximum turns:         123,000 / 300 = 410 turns before overflow

      Sounds comfortable — but cost grows linearly with every turn:

        Turn 10:   3,000 + 2,000 tokens  =   3,000 token history cost
        Turn 50:  15,000 + 2,000 tokens  =  15,000 token history cost
        Turn 100: 30,000 + 2,000 tokens  =  30,000 token history cost

      At $3/MTok input and 10,000 sessions reaching turn 100:
        Turn 1:   $0.009 per call
        Turn 100: $0.099 per call  (11× more expensive)

      The 100th turn is 11× more expensive than the first turn —
      for the same user question complexity — purely from history accumulation.


## 5.2  Message-Level Importance Taxonomy

    ╔══════════════════════════════════════════════════════════════════╗
    ║  RANKING MESSAGES BY INFORMATION VALUE                           ║
    ╚══════════════════════════════════════════════════════════════════╝

      Not all messages in a conversation history are equally valuable.

      TIER 1 — NEVER PRUNE:
        • Messages containing explicit user preferences
          ("I prefer Python over JavaScript")
        • Messages containing task-defining context
          ("I'm building a checkout flow for an e-commerce site")
        • Messages containing key decisions
          ("Let's go with Option B — the microservices approach")
        • The most recent 3–5 turns (conversational context)

      TIER 2 — PRUNE ONLY UNDER PRESSURE:
        • Clarification exchanges that led to the tier-1 decisions
          (the "why" behind a decision, may be useful for follow-ups)
        • Messages containing specific technical details
          (error messages, exact code that was discussed)
        • The first turn (often contains the original task definition)

      TIER 3 — PRUNE FIRST:
        • Purely social messages ("Thanks!", "Got it.", "OK.")
        • Redundant explanations ("Let me explain what I just did...")
        • Failed attempts the user rejected
          (the code that didn't work, before it was fixed)
        • Long model explanations of obvious things
        • Assistant messages confirming understanding without adding content
          ("I understand, you'd like me to...")

    ╔══════════════════════════════════════════════════════════════════╗
    ║  PRUNING ALGORITHMS                                              ║
    ╚══════════════════════════════════════════════════════════════════╝

      ALGORITHM 1 — FIFO (Simplest)
      ──────────────────────────────
      Drop oldest messages first when over budget.
      Always preserve the system prompt and last N turns.

        KEEP: system prompt + turns [max(1, total-N) to total]
        DROP: turns [1 to max(1, total-N)-1]

      Pros: O(1), no LLM calls, always safe from overflow.
      Cons: loses potentially critical early context.

      ALGORITHM 2 — TYPE-WEIGHTED
      ─────────────────────────────
      Assign a retention score to each message type.
      When over budget, drop lowest-score messages first.

        Score = base_type_score × recency_multiplier × length_penalty

        Type scores:
          User turn containing question/task:  1.0
          User turn containing only feedback:  0.3
          Assistant turn with code/data:       0.8
          Assistant turn with explanation:     0.4
          Social/confirmatory turn:            0.1

        Recency multiplier: 1.0 for last 5 turns, 0.5 for older.
        Length penalty: messages >500 tokens get 0.7× score.

        Drop messages in score order (lowest first) until under budget.

      ALGORITHM 3 — IMPORTANCE EXTRACTION (most sophisticated)
      ──────────────────────────────────────────────────────────
      Run a small background LLM call to score/summarise the conversation:

        Prompt: "Rate each message's importance to answering follow-up
                 questions. Return JSON: [{'id': N, 'importance': 0-10}]"

        Keep top-K messages by importance score.
        Summarise the rest into a single "Earlier in this conversation..."
        prefix (see section 6).

      Additional cost: one LLM call every time history needs pruning.
      Quality: significantly better than FIFO or type-weighted.
      Use when conversation quality justifies the cost.


---


## 6. CONVERSATION SUMMARISATION — Compressing Old Turns Before They're Dropped

Rather than silently deleting old turns (pruning), summarisation replaces
them with a compressed representation that retains the key information.
This is the preferred strategy when important context was established early.


## 6.1  The Rolling Summary Architecture

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW ROLLING SUMMARISATION WORKS                                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      The context window contains three zones at any given turn:

      ┌─────────────────────────────────────────────────────────────────┐
      │  [SYSTEM PROMPT]                                                │
      │                                                                 │
      │  [ROLLING SUMMARY]   ← compressed version of all old turns     │
      │  "Earlier: user asked to build a React checkout form.           │
      │   Requirements: Stripe payment, 3 shipping options, dark mode.  │
      │   Decided: TypeScript + React Hook Form. Auth0 for login."      │
      │   ~80 tokens, summarising ~2,000 tokens of turns 1–15          │
      │                                                                 │
      │  [RECENT TURNS]      ← last 5 turns verbatim                   │
      │  Turn 16: user question                                         │
      │  Turn 16: assistant response                                    │
      │  ...                                                            │
      │  Turn 20: user question                                         │
      │                                                                 │
      │  [RESPONSE RESERVE]                                             │
      └─────────────────────────────────────────────────────────────────┘

      TRIGGER: when raw history crosses a threshold (e.g. 60% of budget),
      run a summarisation call on the oldest N turns.

      SUMMARISATION CALL:
        System: "Summarise this conversation history concisely. Preserve:
                 user goals, key decisions, specific requirements, and any
                 data (names, numbers, dates) that may be referenced later.
                 Omit pleasantries and intermediate steps.
                 Output in ≤100 tokens."

        Input: [turns 1 through N]
        Output: the rolling summary (~80–150 tokens)

      RESULT:
        2,000 tokens of turns 1–15  →  ~100 token summary  (20:1 compression)
        The next 2,000 tokens accumulate before the next summarisation trigger.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  COMPRESSION RATIOS AND QUALITY EXPECTATIONS                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌──────────────────────┬─────────────────────┬──────────────────────────┐
      │  Summarised content  │  Compression ratio  │  What is preserved       │
      ├──────────────────────┼─────────────────────┼──────────────────────────┤
      │  Social chat         │  50:1–100:1          │  Key topics only         │
      │  Task discussion     │  10:1–30:1           │  Requirements, decisions │
      │  Technical Q&A       │  5:1–15:1            │  Solutions, constraints  │
      │  Code review         │  3:1–8:1             │  Design choices, issues  │
      │  Data-heavy content  │  2:1–5:1             │  Must preserve numbers   │
      └──────────────────────┴─────────────────────┴──────────────────────────┘

      WHAT SUMMARISATION ALWAYS LOSES:
        • Exact wording (cannot be quoted — it was not said exactly this way)
        • Subtle emotional tone ("I'm a bit concerned about...")
        • Information the summariser judged unimportant (model decisions vary)
        • The sequence of reasoning that led to a conclusion

      PRACTICAL GUIDELINE:
        For fact-critical workflows (legal, medical, financial), summarisation
        introduces a risk: important specific data may be lost or distorted.
        Consider keeping verbatim records of critical exchanges and only
        summarising the purely conversational scaffolding.


## 6.2  What a Good Conversation Summary Contains

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE FIVE ELEMENTS OF AN EFFECTIVE SUMMARY                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      A rolling summary that will be injected into future calls should
      preserve these five categories of information:

      1. TASK AND GOAL
         "User is building a B2B SaaS billing system in Python."
         The original objective — everything else is context for this.

      2. ESTABLISHED CONSTRAINTS
         "Requirements: Stripe for payments, Postgres, must support
          annual and monthly plans, multi-currency from day one."
         Constraints set early are easy to violate if not preserved.

      3. KEY DECISIONS MADE
         "Decided: webhook-based payment confirmation (not polling).
          Auth: JWT with 24h expiry. No free tier."
         Past decisions prevent the model from suggesting reversed choices.

      4. SPECIFIC DATA REFERENCED
         "User's company: Acme Corp. DB schema: users, subscriptions,
          invoices. Test card: 4242 4242 4242 4242."
         Numbers, names, IDs — easy to drop in summarisation, costly to lose.

      5. CURRENT STATE / NEXT STEPS
         "Completed: user auth flow, subscription creation. Outstanding:
          invoice PDF generation, Stripe webhook endpoint."
         Where things stand — so the model doesn't repeat completed work.

      SUMMARY TEMPLATE PROMPT:
        "Summarise the key information from this conversation in ≤150 tokens.
         Include: (1) user's main goal, (2) established constraints and
         requirements, (3) key decisions already made, (4) any specific
         names/numbers/IDs, (5) what has been completed vs. outstanding.
         Omit pleasantries, explanations, and intermediate steps."


---


## 7. DOCUMENT CHUNKING — Splitting Long Docs into Retrieval-Sized Pieces

When documents are too large to fit in the context window, they must be
split into chunks, embedded, and retrieved selectively at query time.
The chunking strategy directly determines retrieval quality.


## 7.1  The Core Trade-Off: Chunk Size

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CHUNK SIZE EFFECTS ON RETRIEVAL QUALITY                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      SMALL CHUNKS (64–256 tokens):
        ✓  High semantic precision — each chunk is about one specific thing
        ✓  Less irrelevant content injected per retrieved chunk
        ✓  Lower per-chunk token cost in the context window
        ✗  May split a concept across chunk boundaries
        ✗  More chunks needed to cover a topic
        ✗  Context around a fact may not be included
        Best for: FAQ databases, product spec sheets, structured records

      MEDIUM CHUNKS (256–512 tokens):
        ✓  A paragraph or two — natural information unit
        ✓  Good balance of precision and context
        ✓  Standard choice for most RAG deployments
        Best for: documentation, articles, knowledge base entries

      LARGE CHUNKS (512–2048 tokens):
        ✓  Full sections or sub-chapters — more context preserved
        ✓  Fewer total chunks → faster retrieval
        ✗  Higher token cost per retrieved chunk
        ✗  More irrelevant content may be included per chunk
        Best for: legal documents, research papers, narrative content

    ╔══════════════════════════════════════════════════════════════════╗
    ║  OVERLAP — PREVENTING INFORMATION LOSS AT BOUNDARIES             ║
    ╚══════════════════════════════════════════════════════════════════╝

      A document split at hard boundaries may cut a concept in half:

      WITHOUT OVERLAP — sentence split at chunk boundary:
        Chunk 3 ends:  "...the model uses RoPE to encode position information."
        Chunk 4 starts: "This means the attention score depends on the offset..."

        A query about "how does RoPE affect attention scores?" might retrieve
        Chunk 3 OR Chunk 4 — neither alone fully answers the question.

      WITH OVERLAP (50-token overlap):
        Chunk 3: [tokens 0–512]
        Chunk 4: [tokens 462–974]   ← includes 50 tokens from Chunk 3
        Chunk 5: [tokens 924–1436]  ← includes 50 tokens from Chunk 4

        The query about RoPE + attention scores will find Chunk 4, which
        contains the end of the RoPE explanation AND the beginning of
        the attention score explanation — the full picture.

      OVERLAP COST:
        Overlap multiplies the total token count stored in the vector DB.
        50-token overlap on 512-token chunks: ~10% increase in stored content.
        This is cheap storage cost for a significant retrieval quality gain.

      STANDARD PRODUCTION CHOICE:
        Chunk size: 512 tokens
        Overlap:    50 tokens  (~10%)


## 7.2  Chunking Strategies Beyond Fixed Size

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FOUR CHUNKING STRATEGIES COMPARED                               ║
    ╚══════════════════════════────────────────────────────────────────╝

      STRATEGY 1 — FIXED-SIZE (most common)
      ────────────────────────────────────────
      Split every N tokens with M-token overlap.
      Simple, predictable, works for all content types.
      Sentence or paragraph boundaries may be broken.

      STRATEGY 2 — SENTENCE/PARAGRAPH BOUNDARY
      ──────────────────────────────────────────
      Split at natural language boundaries (full stops, paragraph breaks).
      Chunks vary in size but semantic units are preserved.
      Implementation: split at sentence boundary nearest to N tokens.

        Document:  "...end of first section.\n\nThe second section discusses..."
        Split:     chunk ends at period — never mid-sentence.
        Cost: adds slight complexity, significant quality improvement for prose.

      STRATEGY 3 — SEMANTIC CHUNKING
      ─────────────────────────────────
      Embed each sentence. Find embedding "breakpoints" where similarity
      between adjacent sentences drops significantly. Split there.

        Sentences 1–8:  cosine sim 0.85 (all about RoPE)
        Sentences 9–15: cosine sim 0.80 (all about ALiBi)
        Similarity between sentences 8 and 9: 0.42  ← topic shift!
                                                        split here

      Produces topically coherent chunks at the cost of additional
      embedding computation during ingestion. Best for long documents
      covering multiple topics.

      STRATEGY 4 — PARENT-CHILD CHUNKING
      ─────────────────────────────────────
      Store two granularities:
        Small child chunks (~128 tokens) — for precise retrieval
        Larger parent chunks (~512 tokens) — for injected context

      Retrieve using child chunks (more precise similarity matching).
      Inject the PARENT chunk into the context (more surrounding context).

      Effect: retrieval precision of small chunks + context richness of
      large chunks. Best for documents where context matters greatly.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CHUNK METADATA — CONTEXT THE EMBEDDING CANNOT CAPTURE           ║
    ╚══════════════════════════════════════════════════════════════════╝

      An embedding captures SEMANTIC CONTENT but not structural context.
      Metadata injected before each chunk provides that context cheaply.

      Without metadata:
        "...the maximum supported connections is 100 per instance..."
        (Retrieved chunk — which product? which version? is this current?)

      With metadata prepended (~15 tokens):
        "[Product: CloudDB v4.2 | Section: Connection Limits | Date: 2024-01]
         ...the maximum supported connections is 100 per instance..."

      The model now knows the source, product, version, and date — critical
      for questions where these details affect the correct answer.

      USEFUL METADATA FIELDS:
        source, document_title, section_heading, page_number,
        product_version, date, author, document_type


---


## 8. STRUCTURED OUTPUT EFFICIENCY — JSON vs. Prose Token Cost

The format of the model's output dramatically affects its token cost.
Structured formats like JSON can cost 2–5× more tokens than equivalent
prose for the same semantic content.


## 8.1  Token Cost of Different Output Formats

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE SAME INFORMATION IN FIVE FORMATS                            ║
    ╚══════════════════════════════════════════════════════════════════╝

      INFORMATION: A sentiment analysis result for a product review.
      Ground truth: Positive sentiment, 4/5 confidence, mentions: price, quality.

      FORMAT 1 — Full JSON (most verbose):
      ──────────────────────────────────────
      {
        "sentiment": "positive",
        "confidence": 0.80,
        "aspects_mentioned": ["price", "quality"],
        "summary": "The reviewer expressed positive sentiment overall."
      }
      Tokens: ~45

      FORMAT 2 — Compact JSON (still JSON but tighter):
      ──────────────────────────────────────────────────
      {"s":"pos","c":0.8,"a":["price","quality"]}
      Tokens: ~18  (60% reduction vs. full JSON)

      FORMAT 3 — Delimited string:
      ─────────────────────────────
      pos|0.8|price,quality
      Tokens: ~8  (82% reduction vs. full JSON)

      FORMAT 4 — Prose sentence:
      ────────────────────────────
      "Positive sentiment (0.8 confidence). Mentions price and quality."
      Tokens: ~18

      FORMAT 5 — Single-word classification:
      ────────────────────────────────────────
      positive
      Tokens: ~1  (98% reduction vs. full JSON)

      ┌─────────────────────────────────────────────────────────────────┐
      │  For 10M classifications/month at $15/MTok output:              │
      │  Full JSON:      450M tokens → $6,750/month                     │
      │  Compact JSON:   180M tokens → $2,700/month  saving: $4,050     │
      │  Delimited:       80M tokens → $1,200/month  saving: $5,550     │
      │  Single word:     10M tokens →   $150/month  saving: $6,600     │
      └─────────────────────────────────────────────────────────────────┘


## 8.2  JSON-Specific Token Waste

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY JSON IS EXPENSIVE — TOKEN-BY-TOKEN ANALYSIS                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      Every structural element in JSON costs tokens:

        {          →  1 token
        "key"      →  1–3 tokens (depending on key length)
        :          →  1 token (often merged with key: "key": = 2 tokens)
        " "        →  1 token (space before value)
        "value"    →  1–N tokens
        ,          →  1 token
        }          →  1 token
        \n         →  1 token (newline for pretty-printing)

      For a simple 5-field JSON object:
        Structural overhead (brackets, colons, quotes, commas): ~30 tokens
        Actual values: ~15 tokens
        Ratio: structural overhead is 2× the actual data

      COMPACT JSON eliminates:
        • Pretty-printing newlines and indentation
        • Spaces after colons ("key": "value" → "key":"value")
        • Verbose key names ("sentiment" → "s")

      WHEN FULL JSON IS WORTH IT:
        ✓  When the application downstream parses the JSON by key name
           (using abbreviated keys requires a mapping table)
        ✓  When human readability is important for debugging
        ✓  When the schema must be self-documenting

      WHEN TO AVOID FULL JSON:
        ✗  High-volume pipelines where output is machine-parsed
        ✗  Simple classifications or extractions
        ✗  When a structured prose sentence reads just as well


## 8.3  Output Format Design for Specific Tasks

    ╔══════════════════════════════════════════════════════════════════╗
    ║  OPTIMAL FORMAT BY TASK TYPE                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌──────────────────────────┬─────────────────────┬────────────────────┐
      │  Task                    │  Optimal format     │  Tokens saved vs.  │
      │                          │                     │  verbose JSON      │
      ├──────────────────────────┼─────────────────────┼────────────────────┤
      │  Binary classification   │  Single word:       │  98%               │
      │  (yes/no, pass/fail)     │  "yes"              │                    │
      ├──────────────────────────┼─────────────────────┼────────────────────┤
      │  Multi-class classif.    │  Single label:      │  95%               │
      │                          │  "negative"         │                    │
      ├──────────────────────────┼─────────────────────┼────────────────────┤
      │  Score/confidence        │  Single number:     │  90%               │
      │                          │  "0.83"             │                    │
      ├──────────────────────────┼─────────────────────┼────────────────────┤
      │  Named entity extraction │  Delimited list:    │  70%               │
      │                          │  "Alice|Bob|Acme"   │                    │
      ├──────────────────────────┼─────────────────────┼────────────────────┤
      │  Key-value extraction    │  Compact JSON or    │  40–60%            │
      │                          │  colon-delimited    │                    │
      ├──────────────────────────┼─────────────────────┼────────────────────┤
      │  Structured record       │  Full JSON          │  baseline          │
      │  (many fields, nested)   │  (readability       │                    │
      │                          │  justified)         │                    │
      ├──────────────────────────┼─────────────────────┼────────────────────┤
      │  Natural language answer │  Prose              │  N/A — prose is    │
      │  (summaries, Q&A)        │                     │  already optimal   │
      └──────────────────────────┴─────────────────────┴────────────────────┘


---


## 9. CHAIN-OF-THOUGHT COST — Why CoT Is More Accurate But More Expensive

Chain-of-thought (CoT) prompting asks the model to reason step-by-step before
giving its final answer. It reliably improves accuracy on complex reasoning tasks —
but those reasoning steps are output tokens, and output tokens are expensive.


## 9.1  What CoT Actually Costs

    ╔══════════════════════════════════════════════════════════════════╗
    ║  DIRECT ANSWER vs. CHAIN-OF-THOUGHT — SIDE BY SIDE              ║
    ╚══════════════════════════════════════════════════════════════════╝

      QUESTION: "A store sells apples for $0.40 each and oranges for $0.60
                 each. Sarah buys 5 apples and 3 oranges. She pays with a
                 $5 note. How much change does she get?"

      DIRECT ANSWER:
        "Sarah gets $1.20 change."
        Output tokens: 8
        Correct: YES (for this easy example)

      CHAIN-OF-THOUGHT ANSWER:
        "Let me work through this step by step.
         Cost of apples: 5 × $0.40 = $2.00
         Cost of oranges: 3 × $0.60 = $1.80
         Total cost: $2.00 + $1.80 = $3.80
         Change: $5.00 - $3.80 = $1.20
         Sarah gets $1.20 change."
        Output tokens: 68
        Correct: YES

      COST COMPARISON:
        CoT uses 8.5× more output tokens for the same correct answer.
        At $15/MTok (Claude 3.5 Sonnet output) and 1M queries/month:
          Direct:  8M tokens → $120/month
          CoT:    68M tokens → $1,020/month
          Difference: $900/month extra for CoT on simple arithmetic.

      BUT — for harder problems where direct answer is WRONG:

      QUESTION: "What is the 15th prime number? Show your working."

      DIRECT ANSWER (may be wrong): "47"  (incorrect — it's 47, but the model
        may say 41 or 53 without working through it)
        Tokens: 2. Accuracy: ~40-60% depending on model.

      COT ANSWER:
        "Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47.
         The 15th prime number is 47."
        Tokens: ~35. Accuracy: ~95%.

      For hard reasoning tasks, CoT buys real accuracy at real token cost.


## 9.2  When Chain-of-Thought Is Worth the Cost

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE ACCURACY-COST DECISION MATRIX                               ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌──────────────────────────┬────────────────────┬────────────────────────┐
      │  Task type               │  CoT accuracy gain │  Cost justified?       │
      ├──────────────────────────┼────────────────────┼────────────────────────┤
      │  Multi-step arithmetic   │  High (+20–40pp)   │  YES for high stakes   │
      │  Logical deduction       │  High (+15–30pp)   │  YES                   │
      │  Mathematical proof      │  High (+30–50pp)   │  YES                   │
      │  Code debugging (complex)│  Medium (+10–25pp) │  YES for critical code │
      │  Legal/medical reasoning │  Medium (+10–20pp) │  YES — errors costly   │
      │  Commonsense Q&A         │  Low (+2–5pp)      │  RARELY                │
      │  Sentiment classification│  Negligible (~0pp) │  NO                    │
      │  Named entity extraction │  Negligible (~0pp) │  NO                    │
      │  Translation             │  Negligible (~0pp) │  NO                    │
      │  Summarisation           │  Low (+2–5pp)      │  RARELY                │
      └──────────────────────────┴────────────────────┴────────────────────────┘

      GUIDELINE: Use CoT when both conditions hold:
        (a) The task involves multi-step reasoning, and
        (b) An incorrect answer has meaningful consequences.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  EXTENDED THINKING / REASONING MODELS — A DIFFERENT COST MODEL  ║
    ╚══════════════════════════════════════════════════════════════════╝

      Extended thinking models (e.g. Claude 3.7 Sonnet with extended
      thinking, o1, o3) generate a hidden reasoning chain before responding.

      The reasoning chain is NOT shown to the user (or is streamed separately).
      It can be 1,000–100,000 tokens long for hard problems.
      It is still billed at output token rates.

      EXAMPLE — Hard math problem with extended thinking:

        Input prompt:          500 tokens
        Hidden reasoning:   10,000 tokens  ← billed as output tokens
        Final answer:           100 tokens  ← billed as output tokens

        Total output tokens: 10,100
        At $15/MTok: $0.1515 per query
        Versus direct answer:  100 tokens = $0.0015 per query

        Extended thinking: 101× more expensive for this query.
        But for hard math/coding/logic where accuracy matters: justified.

      COST CONTROL FOR EXTENDED THINKING:
        Most providers allow a "thinking budget" — a maximum number of
        tokens the model can spend on reasoning before being forced to output.
        Setting a tighter budget reduces cost but may reduce accuracy.


## 9.3  The CoT Distillation Pattern — Eliminating CoT at Inference

    ╔══════════════════════════════════════════════════════════════════╗
    ║  DISTILLATION — PAYING CoT COST ONCE, NOT EVERY CALL             ║
    ╚══════════════════════════════════════════════════════════════════╝

      OBSERVATION: CoT is expensive at inference time but its benefit
      can often be "baked into" a smaller model or fine-tuned into the
      base model so it reasons correctly without generating visible steps.

      WORKFLOW:

        PHASE 1 — GENERATE TRAINING DATA WITH CoT (once):
          Use a large model with CoT to solve 10,000 examples.
          Record the (question, CoT reasoning, correct answer) triples.
          Cost: one-time. Expensive but not recurring.

        PHASE 2 — FINE-TUNE A SMALLER MODEL ON (question, answer) PAIRS:
          Train a smaller model to produce the correct answer directly,
          using the CoT-generated correct answers as labels.
          The smaller model learns to "think" without showing its work.

        PHASE 3 — DEPLOY THE FINE-TUNED MODEL:
          At inference: input question → direct correct answer
          No CoT tokens generated. No CoT cost.

        RESULT: near-CoT accuracy at direct-answer token cost.

      WHEN THIS WORKS WELL:
        ✓  High-volume, stable task (classification, extraction, scoring)
        ✓  Task that benefits from CoT during training (structured reasoning)
        ✓  Volume justifies the fine-tuning one-time cost

      WHEN THIS DOES NOT WORK:
        ✗  Novel or open-ended tasks (no fixed training distribution)
        ✗  Tasks requiring very long CoT chains (complex multi-hop reasoning)
        ✗  Low-volume tasks where fine-tuning cost isn't recovered


---


## 10. CACHING PROMPT TOKENS — Prefix Caching: Pay Once for Repeated Prompts

Prefix caching is the single highest-leverage cost reduction available for
applications that send the same system prompt, document, or instruction set
on every API call. It can reduce input costs by 60–90%.


## 10.1  How Prefix Caching Works

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE MECHANISM — FROM THE ARCHITECTURE MODULE                    ║
    ╚══════════════════════════════════════════════════════════════════╝

      Every input token requires a pre-fill forward pass: the model
      computes Q, K, V matrices for each token. For the static prefix
      (system prompt, large document), this computation is IDENTICAL
      across all requests.

      WITHOUT CACHING:
        Every request pays full input price for every token in the prompt.
        System prompt (2,000 tokens) × 1,000 requests = 2,000,000 tokens billed.

      WITH PREFIX CACHING:
        The server caches the KV activations for the static prefix.
        Subsequent cache-hit requests skip the re-computation.
        They pay a reduced "cache hit" price for those tokens.

        Cache miss  (first request):   pay full input price for all tokens
        Cache hit   (all subsequent):  pay reduced price for cached tokens

    ╔══════════════════════════════════════════════════════════════════╗
    ║  PROVIDER CACHING — RATES AND REQUIREMENTS                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌─────────────────────┬──────────────────┬────────────────┬────────────────────────┐
      │  Provider           │  Full price      │  Cache hit     │  Requirements          │
      ├─────────────────────┼──────────────────┼────────────────┼────────────────────────┤
      │  Anthropic (Claude) │  $3.00/MTok      │  $0.30/MTok    │  Min 1,024 tokens;     │
      │                     │  (Sonnet 3.5)    │  (10% price)   │  cache_control: "ephemeral"│
      │                     │                  │                │  in the API call       │
      ├─────────────────────┼──────────────────┼────────────────┼────────────────────────┤
      │  OpenAI (GPT-4o)    │  $5.00/MTok      │  $2.50/MTok    │  Automatic for prompts │
      │                     │                  │  (50% price)   │  ≥1,024 tokens; no     │
      │                     │                  │                │  special parameter     │
      ├─────────────────────┼──────────────────┼────────────────┼────────────────────────┤
      │  Google (Gemini     │  $3.50/MTok      │  $0.875/MTok   │  Explicit cache        │
      │  1.5 Pro)           │                  │  (25% price)   │  creation call;        │
      │                     │                  │                │  min 32,768 tokens     │
      └─────────────────────┴──────────────────┴────────────────┴────────────────────────┘

      Anthropic offers the deepest discount (90% off).
      OpenAI is automatic (no code changes) but only 50% off.
      Google requires the largest minimum size.


## 10.2  Maximising Cache Hit Rates

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CACHE HIT REQUIREMENTS — WHAT BREAKS THE CACHE                  ║
    ╚══════════════════════════════════════════════════════════════════╝

      A cache hit requires that the prefix is BYTE-IDENTICAL to the
      previously cached prefix. One changed character invalidates the cache.

      THINGS THAT BREAK THE CACHE:
        ✗  Adding the current date/time to the system prompt
           "Today is {{date}}." — changes every day (or every second)
           → Remove timestamps from the cached prefix entirely

        ✗  Personalising the system prompt per user
           "You are helping {{user_name}}." in the system prompt
           → Move personalisation to the user message, not the system prompt

        ✗  A/B testing prompt variations
           Different variants → different cache entries, half the hit rate
           → Run A/B tests as user-turn variations, not system prompt variations

        ✗  Including session-specific context in the prefix
           Any per-request data in the cached section → cache miss every time

      OPTIMAL PROMPT STRUCTURE FOR CACHING:

        ┌──────────────────────────────────────────────────────────────┐
        │  CACHED SECTION (identical every call)                       │
        │                                                              │
        │  System prompt (static persona, rules, format)              │
        │  Large static document (knowledge base, product manual)     │
        │  Tool schemas (if the toolset doesn't change)               │
        │                                                              │
        │  ─────────────────────── CACHE BOUNDARY ─────────────────── │
        │                                                              │
        │  UNCACHED SECTION (changes every call)                       │
        │                                                              │
        │  Conversation history (grows each turn)                      │
        │  Retrieved documents (different per query)                   │
        │  Current user message                                        │
        └──────────────────────────────────────────────────────────────┘

      Everything above the boundary pays cache-hit price.
      Everything below pays full price.


## 10.3  Prefix Caching Cost Calculations

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ROI OF PREFIX CACHING — WORKED EXAMPLES                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      EXAMPLE 1 — Simple system prompt caching (Anthropic)

        System prompt: 2,000 tokens (static)
        Conversation:  1,000 tokens (dynamic)
        Volume:        500,000 calls/month
        Model:         Claude 3.5 Sonnet ($3.00/MTok input)

        WITHOUT CACHING:
          Total input tokens: 3,000 × 500,000 = 1.5B/month
          Cost: 1.5B × $3.00/MTok = $4,500/month

        WITH CACHING:
          Cached tokens:   2,000 × 500,000 = 1B/month at $0.30/MTok
          Uncached tokens: 1,000 × 500,000 = 0.5B/month at $3.00/MTok
          Cost: 1B × $0.30 + 0.5B × $3.00 = $300 + $1,500 = $1,800/month
          Saving: $2,700/month (60% reduction)

      EXAMPLE 2 — Large document caching (high-value)

        Cached document: 20,000 tokens (product manual, static)
        Dynamic context:  2,000 tokens
        Volume:          200,000 calls/month
        Model:           Claude 3.5 Sonnet

        WITHOUT CACHING:
          22,000 × 200,000 = 4.4B tokens → $13,200/month

        WITH CACHING:
          Cached:   20,000 × 200,000 = 4B tokens @ $0.30 = $1,200/month
          Uncached:  2,000 × 200,000 = 0.4B tokens @ $3.00 = $1,200/month
          Total: $2,400/month
          Saving: $10,800/month (82% reduction)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CACHE INVALIDATION AND TTL                                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      Caches are not permanent. Providers expire cached entries after
      varying TTLs (time-to-live):

        Anthropic:  5 minutes of inactivity → cache expires
                    Active caches can persist longer
                    First call after expiry → cache miss (full price)

        OpenAI:     ~1 hour TTL (approximately, varies)

        Google:     Explicit TTL set at creation time (minutes to hours)

      IMPLICATION FOR LOW-TRAFFIC APPLICATIONS:
        If your application has long gaps between requests (>5 minutes),
        the cache expires between calls. You pay full price for every call.
        Prefix caching is most valuable for high-throughput applications
        where calls are frequent enough to maintain warm cache entries.

      WARMING THE CACHE:
        Send a single "warm-up" request at startup or after known idle
        periods to ensure the cache is populated before real traffic arrives.


---


## 11. TOKEN BUDGET FOR RESPONSES — Setting max_tokens Without Truncating

The max_tokens parameter caps how many tokens the model will generate in a
response. Setting it correctly prevents two opposite failure modes: truncated
responses and wasteful over-reservation.


## 11.1  What max_tokens Does and Does Not Do

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE MAX_TOKENS PARAMETER — PRECISE SEMANTICS                    ║
    ╚══════════════════════════════════════════════════════════════════╝

      max_tokens DOES:
        • Set a hard ceiling on generation length
        • Reserve that many tokens of space in the context window
          (effective input budget = context_limit − max_tokens)
        • Stop generation mid-sentence if the limit is hit
        • Protect against runaway generation (model generating indefinitely)

      max_tokens DOES NOT:
        • Guarantee the model will generate that many tokens
          (the model may stop naturally at any point)
        • Tell the model how long to be
          (use an instruction for that: "Answer in ≤50 words")
        • Change the cost of shorter responses
          (you pay for tokens GENERATED, not tokens PERMITTED)

      CRITICAL DISTINCTION:
        max_tokens=4096 on a request that generates 500 tokens costs
        exactly the same as max_tokens=500 on the same request —
        because cost is based on actual tokens generated, not max_tokens.

        BUT: max_tokens=4096 reserves 4,096 tokens of context window space.
        Setting it too high on a model with a tight context limit may
        leave insufficient space for your input.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE TRUNCATION FAILURE MODE                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      If max_tokens is set too low for the task, the response is cut off
      mid-generation when the model hits the limit. The model does not know
      it has been truncated — it simply stops. The caller receives a partial
      response with finish_reason = "length" (not "stop").

      TRUNCATION SYMPTOMS:
        • Response ends mid-sentence: "The solution is to use a load balanc"
        • JSON is incomplete:         {"result": "ok", "data": [1, 2, 3
        • List is cut short:          "1. First item\n2. Second item\n3. Thi"
        • Code is incomplete:         "def process(data):\n    result = []\n    fo"

      HOW TO DETECT TRUNCATION:
        Check finish_reason in the API response.
        "stop"   → model finished naturally (correct)
        "length" → model hit max_tokens limit (truncated — may need retry with higher limit)
        "end_turn" (Anthropic) → same as "stop"
        "max_tokens" (Anthropic) → truncated


## 11.2  Task-Appropriate max_tokens Values

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CALIBRATING MAX_TOKENS BY TASK TYPE                             ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌─────────────────────────────────┬──────────────┬───────────────────────────┐
      │  Task                           │  max_tokens  │  Rationale                │
      ├─────────────────────────────────┼──────────────┼───────────────────────────┤
      │  Binary classification          │    5–10      │  "yes" or "no" = 1 token  │
      │  Multi-class classification     │   10–20      │  One label word           │
      │  Sentiment + brief explanation  │   50–100     │  Label + one sentence     │
      │  Named entity extraction        │  100–300     │  JSON with 5–10 fields    │
      │  Short Q&A answer               │  100–250     │  1–3 sentences typical    │
      │  Email/message reply            │  200–400     │  3–5 paragraphs max       │
      │  Code function (small)          │  200–500     │  ~30 lines of code        │
      │  Code function (complex)        │  500–2000    │  50–200 lines of code     │
      │  Structured report section      │  500–1000    │  One sub-section          │
      │  Full report / long document    │ 2000–4096    │  Multiple sections        │
      │  Complex reasoning / analysis   │ 1000–4096    │  CoT + conclusion         │
      │  Creative writing (short story) │ 1000–4096    │  1,000+ word story        │
      └─────────────────────────────────┴──────────────┴───────────────────────────┘

      PRACTICAL CALIBRATION APPROACH:
        1. Run 50–100 representative queries with max_tokens=4096.
        2. Record actual token counts generated (usage.completion_tokens).
        3. Set max_tokens = 90th percentile of actual usage × 1.1 buffer.
        4. Monitor finish_reason in production — if "length" exceeds 1%
           of responses, increase max_tokens for that task type.


## 11.3  Dynamic max_tokens Based on Task Classification

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ROUTING DIFFERENT TASKS TO DIFFERENT LIMITS                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      In a general-purpose assistant, max_tokens should not be one fixed
      value — it should vary by what the user is asking for.

      SIMPLE CLASSIFICATION APPROACH:

        def get_max_tokens(user_message: str) -> int:
            msg = user_message.lower()

            if any(w in msg for w in ["yes", "no", "true", "false",
                                       "which", "classify", "label"]):
                return 50       # short classification

            if any(w in msg for w in ["summarise", "summarize", "tldr",
                                       "brief", "short"]):
                return 300      # constrained summary

            if any(w in msg for w in ["code", "function", "implement",
                                       "write a", "create a"]):
                return 2000     # code generation

            if any(w in msg for w in ["explain", "how does", "why",
                                       "analyse", "compare"]):
                return 800      # explanation or analysis

            return 1000         # default for general queries

      This simple heuristic reduces max_tokens by 40–80% on classification
      and short-answer tasks without affecting output quality.


---


## 12. COUNTING TOKENS BEFORE SENDING — Using tiktoken Before an API Call

Pre-flight token counting lets you detect overflow, calculate cost estimates,
and make intelligent pruning decisions before making the API call — avoiding
errors, surprises, and wasted spend.


## 12.1  Why Count Tokens Before Sending

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FIVE REASONS FOR PRE-FLIGHT TOKEN COUNTING                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      1. OVERFLOW DETECTION
         Know whether the prompt exceeds the context limit BEFORE sending.
         An API error wastes the network round-trip and reveals nothing
         about which part of the prompt to trim.

      2. COST ESTIMATION
         Calculate the expected cost of a call before committing.
         Essential for rate-limiting, budget caps, and user-facing cost
         transparency ("This analysis will cost approximately $0.05").

      3. INTELLIGENT PRUNING
         Count individual components (system prompt, each retrieved doc,
         each history turn) to identify exactly WHICH part is over budget,
         then apply targeted pruning rather than blind truncation.

      4. CACHE EFFICIENCY TRACKING
         Know exactly how many tokens are in the cached prefix vs. uncached
         section, enabling precise tracking of cache ROI.

      5. DEBUGGING AND LOGGING
         Logging token counts per component enables identification of
         systematic bloat (e.g. a retrieved document that is always 3×
         larger than expected).


## 12.2  tiktoken — The Standard Token Counter

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TIKTOKEN — INSTALLATION AND BASIC USAGE                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      tiktoken is OpenAI's open-source tokenisation library.
      It implements cl100k_base (GPT-4), p50k_base (GPT-3), and others.

      Install:
        pip install tiktoken

      Basic usage:

        import tiktoken

        # Get the encoding for a specific model
        enc = tiktoken.encoding_for_model("gpt-4")

        # Count tokens in a plain string
        text = "Hello, how are you today?"
        tokens = enc.encode(text)
        print(len(tokens))   # → 6

        # Decode back to string (round-trip check)
        decoded = enc.decode(tokens)
        print(decoded)   # → "Hello, how are you today?"

      Counting for specific models:
        "gpt-4"          → uses cl100k_base  (100,256 vocab)
        "gpt-3.5-turbo"  → uses cl100k_base
        "gpt-2"          → uses gpt2         (50,257 vocab)
        "text-davinci-*" → uses p50k_base    (50,281 vocab)

      For non-OpenAI models (Claude, Gemini), tiktoken gives an approximate
      count. For Claude specifically, the true tokeniser is not public.
      cl100k_base gives a count within ~5% for typical English text.


## 12.3  Accurate Message Token Counting

    ╔══════════════════════════════════════════════════════════════════╗
    ║  COUNTING CHATML MESSAGE OVERHEAD                                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      Each ChatML message has structural token overhead beyond the content:

        <|im_start|>system\n    = 4 tokens (message header overhead)
        [content tokens]
        <|im_end|>\n             = 2 tokens (message footer overhead)
        ─────────────────────────────────────────────────────────────
        Total overhead per message: ~4 tokens

      Additionally, the reply primer adds ~3 tokens:
        <|im_start|>assistant\n  = 3 tokens

      PRECISE TOKEN COUNTER FOR CHAT MESSAGES:

        import tiktoken

        def count_chat_tokens(messages: list[dict],
                               model: str = "gpt-4") -> int:
            # Count tokens for a list of messages in ChatML format.
            # Returns the total input tokens that will be billed.
            enc = tiktoken.encoding_for_model(model)

            # Overhead per message varies slightly by model version
            tokens_per_message = 3   # <|im_start|>role\n...<|im_end|>\n
            tokens_per_name    = 1   # if 'name' key present in message

            total = 0
            for msg in messages:
                total += tokens_per_message
                for key, value in msg.items():
                    total += len(enc.encode(str(value)))
                    if key == "name":
                        total += tokens_per_name

            total += 3   # reply primer: <|im_start|>assistant\n

            return total


        # Example usage
        messages = [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user",      "content": "And 3+3?"}
        ]

        print(count_chat_tokens(messages))   # → approximately 44 tokens


## 12.4  Component-Level Counting for Budget Management

    ╔══════════════════════════════════════════════════════════════════╗
    ║  BUILDING A TOKEN BUDGET TRACKER                                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      Rather than counting the full prompt as a lump sum, count each
      component individually to enable targeted pruning.

        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4")

        def count(text: str) -> int:
            return len(enc.encode(text))

        class PromptBudget:
            def __init__(self, context_limit: int, max_response: int):
                self.limit     = context_limit
                self.reserved  = max_response
                self.budget    = context_limit - max_response
                self.components = {}

            def add(self, name: str, content: str):
                self.components[name] = count(content)

            def total(self) -> int:
                return sum(self.components.values())

            def remaining(self) -> int:
                return self.budget - self.total()

            def is_over(self) -> bool:
                return self.total() > self.budget

            def report(self):
                print(f"Budget: {self.budget} tokens (limit {self.limit}"
                      f" - response reserve {self.reserved})")
                for name, tokens in sorted(
                    self.components.items(), key=lambda x: -x[1]):
                    pct = 100 * tokens / self.budget
                    bar = "█" * int(pct / 2)
                    print(f"  {name:25s}: {tokens:5d}  ({pct:4.1f}%)  {bar}")
                print(f"  {'TOTAL':25s}: {self.total():5d}"
                      f"  ({100*self.total()/self.budget:.1f}%)")
                print(f"  {'REMAINING':25s}: {self.remaining():5d}")


        # Usage
        budget = PromptBudget(context_limit=128_000, max_response=2_000)
        budget.add("system_prompt",     system_prompt)
        budget.add("tool_schemas",      json.dumps(tools))
        budget.add("retrieved_doc_1",   doc1)
        budget.add("retrieved_doc_2",   doc2)
        budget.add("conversation",      format_history(history))
        budget.add("user_message",      user_message)

        if budget.is_over():
            budget.report()
            # Apply targeted pruning to the largest components

    ╔══════════════════════════════════════════════════════════════════╗
    ║  PRE-FLIGHT DECISION FLOW                                        ║
    ╚══════════════════════════════════════════════════════════════════╝

      Build prompt components
              │
              ▼
      Count tokens per component
              │
              ▼
      total > (context_limit − max_tokens)?
              │
           NO │                    YES
              │                    │
              ▼                    ▼
        Send API call        Identify largest components
                                    │
                                    ├── System prompt > 500 tokens?
                                    │     → Compress or audit
                                    │
                                    ├── Retrieved docs > 10k tokens?
                                    │     → Re-rank and cut lowest docs
                                    │
                                    ├── History > 20k tokens?
                                    │     → Summarise or prune old turns
                                    │
                                    └── Still over budget?
                                          → Increase max_tokens or use
                                            larger context model


---


## Glossary

    +──────────────────────────────┬──────────────────────────────────────────────────────+
    │  Term                        │  Definition                                           │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Cache hit                   │  A request where the prefix is found in the           │
    │                              │  provider's cache. Billed at a reduced token rate.    │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Cache miss                  │  A request where the prefix is NOT cached. Billed     │
    │                              │  at the full input token rate.                        │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Chain-of-thought (CoT)      │  Prompting the model to reason step-by-step before    │
    │                              │  giving its final answer. Improves accuracy on        │
    │                              │  reasoning tasks at the cost of output tokens.        │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  CoT distillation            │  Using a CoT-capable model to generate correct        │
    │                              │  answers for training data, then fine-tuning a        │
    │                              │  smaller model to produce direct answers.             │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Compression ratio           │  Original token count ÷ compressed token count.      │
    │                              │  A 10:1 ratio means 1,000 tokens → 100 tokens.       │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Extended thinking           │  A model mode where a hidden reasoning chain is       │
    │                              │  generated before the response. Billed as output      │
    │                              │  tokens. Improves accuracy on complex reasoning.      │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Few-shot prompting          │  Including example input/output pairs in the prompt   │
    │                              │  to demonstrate the desired output format or style.   │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  FIFO pruning                │  First-In-First-Out — dropping the oldest history     │
    │                              │  messages first when the context budget is exceeded.  │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  finish_reason               │  API response field indicating why generation stopped.│
    │                              │  "stop" = natural end. "length" = hit max_tokens.     │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  LLMLingua                   │  An automated prompt compression library that removes │
    │                              │  low-perplexity (predictable) tokens from text.       │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  max_tokens                  │  API parameter that caps generation length and         │
    │                              │  reserves space in the context window for the response.│
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Noise dilution              │  Accuracy degradation caused by irrelevant tokens     │
    │                              │  consuming attention that should go to relevant ones. │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Overlap (chunking)          │  A number of tokens repeated between adjacent chunks  │
    │                              │  to prevent information loss at chunk boundaries.     │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Parent-child chunking       │  Storing two chunk sizes: small for retrieval          │
    │                              │  precision, large for injected context richness.       │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Prefix caching              │  Server-side storage of KV activations for a repeated  │
    │                              │  token prefix. Subsequent requests with the same       │
    │                              │  prefix skip re-computation and pay reduced rates.     │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Prompt compression          │  Rewriting a prompt to convey the same information    │
    │                              │  in fewer tokens, either manually or automatically.   │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  RAG                         │  Retrieval-Augmented Generation — injecting retrieved  │
    │                              │  documents into the context at query time.            │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Rolling summary             │  A continuously updated compressed representation of   │
    │                              │  past conversation turns, replacing raw history to     │
    │                              │  stay within the context budget.                      │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Signal-to-token ratio       │  The fraction of tokens in a prompt that directly      │
    │                              │  inform the model's output vs. noise/filler tokens.   │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  tiktoken                    │  OpenAI's open-source tokenisation library.            │
    │                              │  Implements cl100k_base, p50k_base, and others.        │
    │                              │  Used for pre-flight token counting.                  │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Token budget                │  The allocation of a context window across its         │
    │                              │  components: system prompt, retrieved context,         │
    │                              │  history, user message, and response reserve.          │
    +──────────────────────────────┼──────────────────────────────────────────────────────+
    │  Zero-shot prompting         │  Prompting without any examples. Relies entirely on   │
    │                              │  the model's pre-trained capabilities.                 │
    +──────────────────────────────┴──────────────────────────────────────────────────────+


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