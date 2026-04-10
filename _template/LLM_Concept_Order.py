
"""

    ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                  COMPLETE LLM SYSTEMS CURRICULUM                                │
    └─────────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐  ┌──────────────────────────────────────────┐  ┌─────────────────────┐
    │  01. CORE ARCHITECTURE   │  │    02. CONTEXT & TOKEN OPTIMISATION      │  │  03. MEMORY SYSTEMS │
    ├──────────────────────────┤  ├──────────────────────────────────────────┤  ├─────────────────────┤
    │  The Transformer Unit    │  │  Context Window Mechanics                │  │  The 4-Layer Stack  │
    │  ├─ Multi-head Attention │  │  ├─ Window size (128K to 2M+)            │  │  ├─ L1: In-Weights  │
    │  ├─ Self-Attention (QKV) │  │  ├─ Quadratic vs Linear cost             │  │  ├─ L2: System Prom.│
    │  ├─ Feed-Forward (FFN)   │  │  ├─ Lost-in-the-middle phenomena         │  │  ├─ L3: Context RAM │
    │  ├─ Layer Norm & Resid.  │  │  └─ Extension (RoPE, ALiBi)              │  │  └─ L4: Ext. Vector │
    │  └─ GQA (Grouped Query)  │  │                                          │  │                     │
    │                          │  │  Tokenization & Costs                    │  │  Session State      │
    │  The Data Pipeline       │  │  ├─ BPE & Vocabulary size                │  │  ├─ Statelessness   │
    │  ├─ Token Embeddings     │  │  ├─ Special tokens (<BOS>/[INST])        │  │  ├─ Multi-turn hist.│
    │  ├─ Positional Encoding  │  │  ├─ Counting (tiktoken/cl100k)           │  │  ├─ Session isolation│
    │  └─ Detokenization       │  │  └─ Cost/Latency calculation             │  │  └─ History pruning │
    │                          │  │                                          │  │                     │
    │  Advanced Components     │  │  Optimization Patterns                   │  │  Persistent Memory  │
    │  ├─ MoE (Expert Routing) │  │  ├─ Prompt Compression                   │  │  ├─ Fact Extraction │
    │  ├─ Flash Attention 3    │  │  ├─ Context Caching (Prefix)             │  │  ├─ Entity Linking  │
    │  └─ Cross-Attention      │  │  ├─ Summarization loops                  │  │  ├─ Memory Decay    │
    │                          │  │  └─ Chunking (Recursive/Semantic)        │  │  └─ User-level CRUD │
    └──────────────────────────┘  └──────────────────────────────────────────┘  └─────────────────────┘

    ┌─────────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────────┐
    │  04. INFERENCE & SERVING    │  │    05. TRAINING & ALIGN   │  │     06. RAG & GROUNDING       │
    ├─────────────────────────────┤  ├───────────────────────────┤  ├───────────────────────────────┤
    │  The Generation Loop        │  │  Learning Phases           │  │  Retrieval Pipeline           │
    │  ├─ Autoregressive flow     │  │  ├─ Pre-training (Next Tok)│  │  ├─ Embeddings (Dense)        │
    │  ├─ KV Cache Management     │  │  ├─ SFT (Instruction)      │  │  ├─ BM25 (Sparse/Keyword)    │
    │  ├─ TTFT vs. TPOT latency   │  │  ├─ RLHF / DPO (Feedback)  │  │  ├─ Hybrid Search             │
    │  └─ Stopping Conditions     │  │  └─ Knowledge Cutoff       │  │  └─ Reranking (Cross-Enc)     │
    │                             │  │                            │  │                               │
    │  Sampling & Controls        │  │  Efficiency & Adaptation   │  │  Advanced RAG Patterns        │
    │  ├─ Temperature (Logits)    │  │  ├─ LoRA / QLoRA           │  │  ├─ HyDE (Hypothetical)       │
    │  ├─ Nucleus (Top-P)         │  │  ├─ PEFT (Parameter Eff.)  │  │  ├─ Self-RAG (Reflection)     │
    │  ├─ Greedy vs Beam Search   │  │  ├─ Catastrophic Forget.   │  │  ├─ Corrective RAG (CRAG)     │
    │  └─ Repetition Penalty      │  │  └─ Domain Specialization  │  │  └─ GraphRAG (Entities)       │
    │                             │  │                            │  │                               │
    │  Serving Optimization       │  │  Safety & Governance       │  │  Trust & Reliability          │
    │  ├─ Speculative Decoding    │  │  ├─ Red Teaming            │  │  ├─ Citation Attribution      │
    │  ├─ Continuous Batching     │  │  ├─ Constitutional AI      │  │  ├─ Hallucination Detection   │
    │  └─ Quantization (FP8/INT4) │  │  └─ Bias Mitigation        │  │  └─ Verifiability             │
    └─────────────────────────────┘  └───────────────────────────┘  └───────────────────────────────┘

    ┌──────────────────────────────────────┐  ┌──────────────────────────────────────────┐
    │       07. AGENTS & TOOL USE          │  │     08. LIMITATIONS & FAILURES           │
    ├──────────────────────────────────────┤  ├──────────────────────────────────────────┤
    │  Agentic Logic                       │  │  Reasoning Gaps                          │
    │  ├─ ReAct (Thought/Act/Obs)          │  │  ├─ Logical Fallacies                    │
    │  ├─ Chain of Thought (CoT)           │  │  ├─ Mathematical errors                  │
    │  ├─ Reflection & Self-Correction      │  │  ├─ Sycophancy (User bias)               │
    │  └─ Plan-and-Execute                │  │  └─ Primacy/Recency bias                 │
    │                                      │  │                                          │
    │  Tooling Ecosystem                   │  │  Operational Limits                      │
    │  ├─ Function Calling (JSON)          │  │  ├─ Stale World Knowledge                │
    │  ├─ Parallel Tool Calls              │  │  ├─ Multi-step drift                     │
    │  ├─ Sandbox Execution                │  │  ├─ Prompt Injection / Jailbreaks         │
    │  └─ MPC (Model Context Protocol)     │  │  └─ Instruction Following Failures        │
    │                                      │  │                                          │
    │  Multi-Agent Patterns                │  │  Inference Limits                        │
    │  ├─ Orchestrator / Worker            │  │  ├─ KV Cache Bloat                       │
    │  ├─ Peer Review (Critic)             │  │  ├─ Compute/Cost Scaling                 │
    │  ├─ Shared State Management          │  │  ├─ Latency Bottlenecks                  │
    │  └─ Agent Handoffs                   │  │  └─ Context Degradation                   │
    └──────────────────────────────────────┘  └──────────────────────────────────────────┘











































































    ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                    HOW LLMs WORK — MODULE MAP                                   │
    └─────────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐  ┌──────────────────────────────────────────┐  ┌─────────────────────┐
    │  1. CORE ARCHITECTURE    │  │   2. CONTEXT WINDOW & TOKEN MANAGEMENT   │  │  3. MEMORY/SESSION  │
    ├──────────────────────────┤  ├──────────────────────────────────────────┤  ├─────────────────────┤
    │  Foundation              │  │  Context Window Fundamentals             │  │  Session Isolation  │
    │  ├─ Transformer overview │  │  ├─ Window definition                    │  │  ├─ Statelessness   │
    │  ├─ Tokenization         │  │  ├─ Window size limits                   │  │  ├─ No persistence  │
    │  ├─ Token embeddings     │  │  ├─ Why window size matters              │  │  ├─ Session boundary│
    │  └─ Positional encoding  │  │  ├─ Quadratic attention cost             │  │  └─ Multi-user isol.│
    │                          │  │  └─ Long-context (RoPE, ALiBi)           │  │                     │
    │  Attention Mechanism     │  │                                          │  │  In-Context Memory  │
    │  ├─ Self-attention       │  │  Token Counting & Budgeting              │  │  ├─ Convo history   │
    │  ├─ Query / Key / Value  │  │  ├─ Token counting                       │  │  ├─ Turn accumulate │
    │  ├─ Multi-head attention │  │  ├─ Input vs output token cost           │  │  ├─ Context as RAM  │
    │  ├─ Attention scores     │  │  ├─ System prompt overhead               │  │  └─ System prompt   │
    │  └─ Causal masking       │  │  └─ Tokenizer differences                │  │                     │
    │                          │  │                                          │  │  External Memory    │
    │  Layer Structure         │  │  Context Optimization                    │  │  ├─ Vector DB       │
    │  ├─ Feed-forward layers  │  │  ├─ Prompt compression                   │  │  ├─ Semantic search │
    │  ├─ Layer normalization  │  │  ├─ Context pruning/summarization        │  │  ├─ Episodic memory │
    │  ├─ Residual connections │  │  ├─ Sliding window strategy              │  │  ├─ Semantic memory │
    │  ├─ Encoder vs decoder   │  │  ├─ RAG context injection                │  │  ├─ Procedural mem. │
    │  └─ KV cache             │  │  ├─ Chunking strategies                  │  │  └─ KV store memory │
    │                          │  │  ├─ Context distillation                 │  │                     │
    │                          │  │  └─ Lost-in-the-middle problem           │  │  Memory Patterns    │
    │                          │  │                                          │  │  ├─ Running summary │
    │                          │  │                                          │  │  ├─ R/W/D lifecycle │
    │                          │  │                                          │  │  ├─ Memory decay    │
    │                          │  │                                          │  │  ├─ Store vs fetch  │
    │                          │  │                                          │  │  ├─ Shared memory   │
    │                          │  │                                          │  │  └─ Conflicts/sync  │
    └──────────────────────────┘  └──────────────────────────────────────────┘  └─────────────────────┘

    ┌─────────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────────┐
    │  4. INFERENCE & PROMPTS     │  │   5. TRAINING/KNOWLEDGE   │  │   6. RETRIEVAL & GROUNDING    │
    ├─────────────────────────────┤  ├───────────────────────────┤  ├───────────────────────────────┤
    │  Prompt Construction        │  │  Pre-Training             │  │  RAG Fundamentals             │
    │  ├─ Prompt ingestion        │  │  ├─ Next-token prediction │  │  ├─ RAG overview              │
    │  ├─ System/user/asst roles  │  │  ├─ Training data curation│  │  ├─ Dense vs sparse retrieval │
    │  ├─ Special tokens BOS/EOS  │  │  ├─ Deduplication         │  │  ├─ Embedding models          │
    │  └─ Prompt injection        │  │  ├─ Knowledge cutoff      │  │  └─ Cosine similarity         │
    │                             │  │  └─ Scaling laws          │  │                               │
    │  Generation                 │  │                           │  │  Advanced RAG Patterns        │
    │  ├─ Autoregressive gen.     │  │  Post-Training Alignment  │  │  ├─ HyDE                      │
    │  ├─ Temperature             │  │  ├─ SFT                   │  │  ├─ Re-ranking                │
    │  ├─ Top-p / nucleus         │  │  ├─ RLHF                  │  │  ├─ Hybrid search             │
    │  ├─ Top-k sampling          │  │  ├─ DPO                   │  │  ├─ Multi-hop retrieval       │
    │  ├─ Repetition penalty      │  │  └─ Constitutional AI     │  │  └─ Self-RAG                  │
    │  └─ Beam search vs greedy   │  │                           │  │                               │
    │                             │  │  Adaptation               │  │  Grounding                    │
    │  Efficiency                 │  │  ├─ LoRA fine-tuning      │  │  ├─ Hallucination & grounding │
    │  ├─ Speculative decoding    │  │  ├─ Instruction tuning    │  │  └─ Citation grounding        │
    │  ├─ Batched inference       │  │  ├─ Catastrophic forget.  │  │                               │
    │  └─ Streaming vs full resp. │  │  └─ Domain adaptation     │  │                               │
    └─────────────────────────────┘  └───────────────────────────┘  └───────────────────────────────┘

    ┌──────────────────────────────────────┐  ┌──────────────────────────────────────────┐
    │  7. AGENTS & TOOL USE                │  │   8. LIMITATIONS & FAILURE MODES         │
    ├──────────────────────────────────────┤  ├──────────────────────────────────────────┤
    │  Agent Architecture                  │  │  Common Failure Modes                    │
    │  ├─ Agent loop (perceive-plan-act)   │  │  ├─ Hallucination types                  │
    │  ├─ Tool use / function calling      │  │  ├─ Sycophancy                           │
    │  ├─ ReAct pattern                    │  │  ├─ Recency & primacy bias               │
    │  └─ Chain-of-thought reasoning       │  │  ├─ Reasoning failures                   │
    │                                      │  │  └─ Prompt sensitivity                   │
    │  Multi-Agent Systems                 │  │                                          │
    │  ├─ Orchestration patterns           │  │  Knowledge & Context Limits              │
    │  ├─ Subagent spawning                │  │  ├─ Stale knowledge                      │
    │  ├─ Shared context                   │  │  ├─ Context length limits                │
    │  └─ Agent handoffs                   │  │  ├─ Working memory limits                │
    │                                      │  │  ├─ Multi-step reasoning limits          │
    │  Planning & State                    │  │  └─ Instruction-following failures       │
    │  ├─ Task decomposition               │  │                                          │
    │  ├─ Scratchpad / working memory      │  │                                          │
    │  ├─ Plan-and-execute pattern         │  │                                          │
    │  └─ Self-reflection loops            │  │                                          │
    └──────────────────────────────────────┘  └──────────────────────────────────────────┘

                        ╔══════════════════════════════════════╗
                        ║         MEMORY LAYER STACK           ║
                        ╠══════════════════════════════════════╣
                        ║  L1 │ In-weights (pre-trained facts) ║ ← frozen
                        ║  L2 │ System prompt (persistent)     ║ ← per-deployment
                        ║  L3 │ Context window (working RAM)   ║ ← per-session
                        ║  L4 │ External store (vector / KV)   ║ ← retrieved
                        ╚══════════════════════════════════════╝





╔══════════════════════════════════════════════════════════════════════════════════════╗
║                        LLM CONCEPT CURRICULUM — TOPIC MAP                            ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

      MODULE 00 · Architecture & Inference
      ├── prompt_processing          End-to-end prompt flow and tokenisation
      ├── session_isolation          Session-scoped vs. persistent memory
      ├── inference_mental_model     Threshold/path-selection — what's right and wrong
      ├── single_neuron              Weights, bias, activation, output
      ├── neuron_connectivity        Inputs, outputs, fan-out structure
      ├── input_connection_limits    Who sets layer width and why
      ├── parallel_processing        Simultaneous vs. sequential aggregation
      └── input_scale_bounds         Layer Norm, value bounds, activation functions

      MODULE 01 · Tokenisation
      ├── what_is_a_token            Characters vs. words vs. subwords
      ├── bpe_tokenisation           Byte Pair Encoding — how vocabulary is built
      ├── tokeniser_vocabulary       Vocab size trade-offs (GPT-2: 50k, LLaMA: 32k)
      ├── token_counting             Why "1 word ≈ 1.3 tokens" and when it breaks
      ├── tokenisation_edge_cases    Numbers, code, non-English, emojis
      ├── special_tokens             <BOS>, <EOS>, <PAD>, [INST] — roles and positions
      └── tokenisation_and_cost      Token count = compute cost = API billing unit

      MODULE 02 · Context Window  ★
      ├── context_window_definition       What the context window is and contains
      ├── context_window_sizes            GPT-2 (1k) → Claude (200k) → Gemini (1M+)
      ├── what_fills_the_window           System prompt / history / docs / tools budget
      ├── lost_in_the_middle              Why buried information gets less attention
      ├── context_window_vs_memory        Why these are the same thing
      ├── sliding_window_attention        Handling sequences beyond training length
      ├── context_utilisation_efficiency  Density vs. noise — not all tokens equal
      ├── context_overflow_strategies     Truncation, summarisation, chunking
      ├── long_context_performance        Quality degradation at very long contexts
      ├── positional_encoding             How the model knows token position
      └── rope_alibi_encodings            RoPE, ALiBi — extending beyond training length

      MODULE 03 · Token Optimisation  ★
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

      MODULE 04 · Memory Architecture  ★
      ├── four_types_of_llm_memory    In-context / external / in-weights / in-cache
      ├── in_context_memory           What lives in the window — the only "live" memory
      ├── in_weights_memory           Knowledge baked in at training — frozen at inference
      ├── external_memory_stores      Databases, vector stores, files
      ├── kv_cache                    Key-Value cache — avoiding full re-reads each turn
      ├── kv_cache_invalidation       What clears the cache and why it matters for cost
      ├── episodic_vs_semantic_memory Event-specific recall vs. general world knowledge
      └── working_memory_analogy      Context window as cognitive working memory

      MODULE 05 · Session & State Management  ★
      ├── stateless_model_truth           Every API call is fully independent
      ├── session_scoped_memory           How a single conversation accumulates history
      ├── cross_session_isolation         Why new chats know nothing of old ones
      ├── how_apps_inject_history         The app layer's job: retrieve, assemble, prepend
      ├── session_state_in_apis           messages[] array — full thread passed each time
      ├── multi_turn_cost_growth          Why message #10 costs 10× more than message #1
      ├── session_id_and_routing          How platforms tie messages to a conversation
      └── conversation_reset_strategies   When and how to wipe history

      MODULE 06 · Persistent Memory Systems  ★
      ├── memory_features_overview        ChatGPT Memory, Claude Memory, MemGPT
      ├── fact_extraction_pipeline        How "User prefers Python" is extracted
      ├── memory_injection_into_prompt    Where extracted facts go in each new session
      ├── memory_retrieval_vs_injection   Inject everything vs. search for relevant
      ├── rag_for_memory                  Vector search to retrieve relevant past context
      ├── memory_staleness_problem        Old facts that are no longer true
      ├── memory_privacy_implications     What is stored, where, who can see it
      ├── user_controllable_memory        Letting users view, edit, delete memories
      └── operator_injected_context       CRM data, account history in system prompt

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
      └── hybrid_search              BM25 (keyword) + vector (semantic)

      MODULE 08 · Attention Mechanism
      ├── attention_intuition         Why looking at all tokens is needed
      ├── query_key_value             Q, K, V — what each represents
      ├── attention_score_computation Dot product → scale → softmax → weighted sum
      ├── multi_head_attention        Parallel attention across representation subspaces
      ├── causal_masking              Why future tokens are masked during training
      ├── attention_complexity        O(n²) — why long contexts are expensive
      ├── flash_attention             Memory-efficient attention rewrite
      ├── grouped_query_attention     GQA — reducing KV cache memory usage
      └── cross_attention             Attending across two different sequences

      MODULE 09 · Training Concepts
      ├── pre_training              Next-token prediction on internet-scale text
      ├── fine_tuning               Adjusting weights on a curated dataset
      ├── rlhf                      Reinforcement Learning from Human Feedback
      ├── reward_model              The model that scores helpfulness and safety
      ├── sft_vs_rlhf               Supervised fine-tuning vs. reinforcement
      ├── catastrophic_forgetting   Why fine-tuning can destroy general capability
      ├── lora_and_peft             Low-rank adapters — fine-tuning without all weights
      ├── data_quality_over_quantity 1M good examples > 1B noisy ones
      └── training_compute_scaling  Chinchilla laws — optimal token-to-param ratio

      MODULE 10 · Inference Optimisation
      ├── autoregressive_generation  Why the model generates one token at a time
      ├── greedy_vs_sampling         Top token always vs. sampling the distribution
      ├── temperature                Scaling logits — 0.0 (deterministic) to 2.0 (chaos)
      ├── top_k_top_p                Nucleus sampling — restricting candidate pool
      ├── speculative_decoding       Draft model proposes, large model verifies
      ├── quantisation               INT8/INT4 — smaller weights, same model in less RAM
      ├── batching_strategies        Continuous batching for concurrent users
      ├── kv_cache_management        PagedAttention — KV cache as virtual memory
      └── beam_search                Multiple candidate sequences explored in parallel

      MODULE 11 · Prompting Techniques
      ├── zero_shot_prompting        Asking with no examples
      ├── few_shot_prompting         In-context examples — format matters as much as content
      ├── chain_of_thought           "Think step by step" — why it improves accuracy
      ├── self_consistency           Multiple reasoning paths + majority vote
      ├── system_prompt_design       Role / instructions / constraints / format
      ├── prompt_injection           Attacks that hijack behaviour via untrusted input
      ├── xml_tagging                Separating data from instructions with XML tags
      ├── output_format_control      Getting reliable JSON, lists, tables
      └── meta_prompting             Using the model to write its own prompts

      MODULE 12 · Hallucination & Reliability
      ├── what_is_hallucination        Confident generation of false information
      ├── why_models_hallucinate       Next-token prediction has no "I don't know"
      ├── factual_vs_reasoning_errors  Wrong facts vs. flawed logic
      ├── hallucination_in_rag         When retrieval reduces vs. introduces errors
      ├── grounding_strategies         Citations and source attribution
      ├── calibration                  Whether confidence correlates with accuracy
      ├── refusal_vs_hallucination     Trading false confidence for "I don't know"
      ├── How are hallucinations detected
      ├── Steps and methods to avoid hallucinations
      └── Reasons for causes for hallcinations

      MODULE 13 · Agents & Tool Use
      ├── what_is_an_agent          Model + tools + loop
      ├── function_calling          Structured tool invocation via the API
      ├── tool_selection            How the model decides which tool to call
      ├── react_pattern             Reason + Act cycles — the dominant agentic loop
      ├── agentic_memory_management Context grows fast — strategies to control it
      ├── multi_agent_systems       Orchestrator + specialist agents
      └── agentic_failure_modes     Looping, tool misuse, context overflow

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      ★  Priority modules — deepest coverage of memory, sessions, context, token optimisation

      TOPIC COUNT
      ├── Modules 00–13         14 modules
      ├── Sections total        ~100 topics
      └── ★ Priority sections    47 topics across modules 02–06

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

