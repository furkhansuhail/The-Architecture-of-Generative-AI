"""
Tutorial Template — Automation & Infrastructure
=================================================
Copy this file, rename it, and fill in the sections.

The TOPIC_NAME and CATEGORY are parsed by the app.
OPERATIONS entries can specify "language" as "bash", "yaml", "python", "json", etc.
"""

TOPIC_NAME = "LLM - Architecture & Inference"
DISPLAY_NAME = "LLM - Architecture & Inference"
ICON         = "📖"
SUBTITLE     = "Explanation about LLM Architecture and Inference"
CATEGORY = "LLM Architecture"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = r"""

Order: 

    Module_Concepts · Architecture & Inference
          ├── prompt_processing          End-to-end prompt flow and tokenisation
          ├── session_isolation          Session-scoped vs. persistent memory
          ├── inference_mental_model     Threshold/path-selection — what's right and wrong
          ├── single_neuron              Weights, bias, activation, output
          ├── neuron_connectivity        Inputs, outputs, fan-out structure
          ├── input_connection_limits    Who sets layer width and why
          ├── parallel_processing        Simultaneous vs. sequential aggregation
          └── input_scale_bounds         Layer Norm, value bounds, activation functions


## PROMPT PROCESSING - End-to-End Prompt Flow & Tokenisation

Topics 

    •	Unicode, byte-pair encoding, and the full tokenisation pipeline
    •	Vocabulary construction, special tokens, and encoding edge cases
    •	Prompt assembly: system prompts, conversation turns, and delimiters
    •	Context windows: KV cache mechanics and memory layout
    •	Attention masks, position encodings, and positional bias
    •	Pre-fill (prompt processing) vs. auto-regressive decoding
    •	Sampling strategies: temperature, top-k, top-p, and beam search
    •	Batching, padding, and throughput optimisation
    •	Tool use, multi-modal inputs, and structured generation
    •	Detokenisation, post-processing, and output formatting


## 1. What Is a Token? — The Fundamental Unit

Before any neural network can process a sentence, every character, word, or sub-word must be 
converted into an integer — an index into a fixed vocabulary. That integer is a token. 
Understanding why tokens exist as they do, and how they are produced, is the foundation of everything 
that follows.

## Vocabulary Lookup Table Explanation:

Imagine you have a dictionary (the vocabulary) that lists every known word and assigns each one a unique number:
 
    "the"  → 0
    "cat"  → 1
    "sat"  → 2
    "on"   → 3
    "mat"  → 4

When you have a sentence like "the cat sat", the model converts it to [0, 1, 2].
Each number is an index — meaning it's a position in that lookup table, not a position in your document.
Breaking down the phrase:

"index" → the integer assigned to a word (its position in the vocabulary list)
"fixed vocabulary" → the vocabulary is decided beforehand and doesn't change — if a word isn't in it, it's treated as unknown
"into a fixed vocabulary" → means "pointing into / referring to an entry in that vocabulary table"

**Every word gets mapped to a number that represents where that word lives in a pre-built vocabulary list.**

---

## 1.1  Why Not Characters or Words?

Two obvious alternatives exist: character-level models and word-level models. 
Character models keep vocabularies tiny (roughly 100–300 symbols for Unicode), but they require sequences hundreds of 
times longer than word models for equivalent text. This dramatically increases the quadratic cost of self-attention. 
Word models keep sequences short, but their vocabularies can swell to hundreds of thousands of entries, most of which 
appear extremely rarely, causing poor generalisation and enormous embedding tables.


Sub-word tokenisation is the practical middle ground. A vocabulary of 32,000–100,000 carefully chosen pieces covers 
most natural-language text compactly while keeping sequences manageable. 
Common words like "the" become single tokens; rare words like "antidisestablishmentarianism" split into recognisable 
morphemes; digits, punctuation, and non-Latin scripts each receive appropriate coverage.


**KEY CONCEPT:**	
The compression ratio — the ratio of Unicode characters to tokens — for well-written 
English is roughly 3.5–4.5 characters per token. Code, JSON, and some non-Latin scripts can push that ratio below 2, 
making them significantly more expensive in context budget terms.

## 1.2  Unicode and Byte-Level Pre-Processing

Modern LLM tokenisers operate on UTF-8 bytes. Every input string first undergoes 
Unicode normalisation (typically NFC or NFKC) to canonicalise composed vs. decomposed character forms and remove 
confusable duplicates. After normalisation, the string is encoded to a sequence of raw bytes (0–255).

Operating at the byte level rather than the character level gives total coverage: even emojis, mathematical symbols, 
rare CJK extension characters, and malformed sequences can always be represented as byte tokens. 
Vocabularies include explicit tokens for all 256 byte values as a fallback, guaranteeing lossless round-trip encoding 
and decoding of any input.

## 1.3  Byte-Pair Encoding (BPE) — The Core Algorithm

GPT-style tokenisers (including those used by most instruction-tuned transformer LLMs) are built with Byte-Pair Encoding. 

The algorithm has two phases: 

    1. Training 
    
    2. Encoding
    
**Training Phase**

Given a large training corpus, BPE begins with a vocabulary containing every unique byte (256 entries).
 
It then repeatedly:

    •	Counts all adjacent symbol pairs in the entire corpus.
    
    •	Identifies the most frequent pair, e.g. (e, r).
    
    •	Merges that pair into a new symbol er and adds it to the vocabulary.
    
    •	Replaces all occurrences of the pair in the corpus with the merged symbol.
    
    •	Repeats until the vocabulary reaches a target size (e.g. 50,257 for GPT-2, 100,256 for cl100k).

The result is a deterministic merge table — an ordered list of merge rules. The rules encode the statistical structure 
of the training corpus. Frequent sub-strings become single high-frequency tokens; rarer ones stay as sequences of 
smaller pieces.

Encoding Phase (Inference)

At inference time, new text is encoded by:

    •	Converting the string to UTF-8 bytes.
    
    •	Applying any pre-tokenisation rules (e.g. splitting on whitespace, punctuation, or regex patterns 
        defined in the tokeniser config).
    
    •	Treating each byte in each pre-token chunk as an initial symbol.

    •	Scanning the merge table in order and applying every rule whose pair appears in the current symbol sequence.
    
    •	Mapping each resulting symbol to its integer ID via the vocabulary dictionary.


**IMPORTANT**	

The pre-tokenisation regex is often overlooked but critically important. GPT-4's cl100k_base tokeniser uses a regex 
that prevents merges across word boundaries, contraction apostrophes, and whitespace. This means "token" and " token" 
(with a leading space) are always different tokens. This design intentionally encodes the concept of word-initial 
position.

    ╔══════════════════════════════════════════════════════════════════╗
    ║              BYTE PAIR ENCODING (BPE) — HOW IT WORKS             ║
    ╚══════════════════════════════════════════════════════════════════╝
    
     CORPUS (input)
     ┌─────────────────────────────────┐
     │  "hug" ×2   "pug" ×2            │
     │  "pun" ×1   "bun" ×1            │
     └─────────────────────────────────┘
                      │
                      │  split every word into characters
                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  ITERATION 0 — character tokens                                 │
    │                                                                 │
    │  "hug" ×2  →  [h] [u] [g]                                       │
    │  "pug" ×2  →  [p] [u] [g]                                       │
    │  "pun" ×1  →  [p] [u] [n]                                       │
    │  "bun" ×1  →  [b] [u] [n]                                       │
    │                                                                 │
    │  Count adjacent pairs (weighted by word freq):                  │
    │                                                                 │
    │   u·g  ████████████████  4  ◄── winner                          │
    │   p·u  ████████████      3                                      │
    │   h·u  ████████          2                                      │
    │   u·n  ████████          2                                      │
    │   b·u  ████              1                                      │
    └─────────────────────────────────────────────────────────────────┘
                      │
                      │  merge top pair:  [u] + [g]  →  [ug]
                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  ITERATION 1 — after merging "u·g"                              │
    │                                                                 │
    │  "hug" ×2  →  [h] [ug]                                          │
    │  "pug" ×2  →  [p] [ug]                                          │
    │  "pun" ×1  →  [p] [u]  [n]                                      │
    │  "bun" ×1  →  [b] [u]  [n]                                      │
    │                                                                 │
    │  Count pairs again:                                             │
    │                                                                 │
    │   p·ug  ████████  2  ◄── winner (pick one of the ties)          │
    │   h·ug  ████████  2                                             │
    │   u·n   ████████  2                                             │
    │   p·u   ████      1                                             │
    │   b·u   ████      1                                             │
    └─────────────────────────────────────────────────────────────────┘
                      │
                      │  merge top pair:  [p] + [ug]  →  [pug]
                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  ITERATION 2 — after merging "p·ug"                             │
    │                                                                 │
    │  "hug" ×2  →  [h]   [ug]                                        │
    │  "pug" ×2  →  [pug]                                             │
    │  "pun" ×1  →  [p]   [u]  [n]                                    │
    │  "bun" ×1  →  [b]   [u]  [n]                                    │
    │                                                                 │
    │  Vocabulary so far:                                             │
    │  { h, p, u, g, b, n }  +  { ug, pug }  ← newly learned          │
    └─────────────────────────────────────────────────────────────────┘
                      │
                      │  repeat until vocab reaches target size
                      │  (e.g. 50,000 tokens for GPT-2)
                      ▼
                  [ DONE ]
    
     KEY TAKEAWAYS
     ─────────────────────────────────────────────────────────────────
      1. Start small  — initial vocab is just individual characters
      2. Merge greedily — always pick the most frequent adjacent pair
      3. Grows bottom-up — common subwords (ing, tion, est) emerge
                           naturally from data
      4. No <UNK> needed — any new word splits into known subpieces


## 1.4  SentencePiece and Unigram Tokenisation

LLaMA-family and many multilingual models use SentencePiece, which supports both BPE and an alternative Unigram 
Language Model algorithm. Unlike BPE, Unigram starts with a large vocabulary and iteratively prunes the pieces whose 
removal decreases the training corpus log-likelihood the least. This produces a probabilistic tokeniser where the same 
string can in principle be tokenised in multiple ways; the Viterbi algorithm selects the segmentation with the highest 
probability.

A key difference from BPE is that SentencePiece treats the raw text as a stream of Unicode code points, not bytes, 
and uses a special whitespace meta-symbol (▁) to mark word boundaries inside tokens rather than splitting on whitespace. 
This means whitespace is embedded within tokens rather than being a separate pre-tokenisation step.


## 1.5  Special Tokens
Every tokeniser reserves a set of non-text tokens for structural purposes. 

These typically include:


    +----------------+----------------------+--------------------------------------------------------------+
    | Token          | Common ID / Name     | Purpose                                                      |
    +----------------+----------------------+--------------------------------------------------------------+
    | <|endoftext|>  | 50256 (GPT-2)        | End-of-sequence / document separator                         |
    | <|im_start|>   | cl100k               | ChatML: begins a conversation turn                           |
    | <|im_end|>     | cl100k               | ChatML: ends a conversation turn                             |
    | <s>            | SentencePiece BOS    | Beginning-of-sequence in LLaMA models                        |
    | </s>           | SentencePiece EOS    | End-of-sequence in LLaMA models                              |
    | [PAD]          | BERT / most encoders | Padding to align batch sequences                             |
    | [MASK]         | BERT                 | Masked token prediction in pre-training                      |
    | <|tool_call|>  | Custom               | Marks start of a tool invocation in function-calling formats |
    +----------------+----------------------+--------------------------------------------------------------+

Special tokens are injected by the tokenisation wrapper, not by the base BPE/Unigram model. 
They occupy IDs in the vocabulary but their bytes are never processed by the merge rules. 
Attempting to encode the string "<|endoftext|>" as if it were regular text would tokenise it differently from injecting
the genuine special token.


### 2. Prompt Assembly — Structuring the Input

The raw tokens produced by the tokeniser are not immediately fed to the model. They must first be assembled into a 
structured prompt according to the format the model was trained on. This assembly phase is where system instructions, 
conversation history, and the current user message are concatenated into a single linear token sequence.


## 2. Prompt Assembly — Structuring the Input

The raw tokens produced by the tokeniser are not immediately fed to the model. They must first be assembled into a 
structured prompt according to the format the model was trained on. This assembly phase is where system instructions, 
conversation history, and the current user message are concatenated into a single linear token sequence.


## 2.1  The ChatML Format

OpenAI introduced the ChatML (Chat Markup Language) format, which most instruction-tuned models now use in some form. 
Each conversational turn is wrapped in delimiter tokens:


**EXAMPLE:** 
	
	<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user What is 2+2? <|im_end|> <|im_start|>assistant 
	
	
	<|im_start|>system                              ->  ChatML: begins a conversation turn      (System)
    You are a helpful assistant.
    <|im_end|>                                      ->  ChatML: ends a conversation turn
    
    <|im_start|>user                                ->  ChatML: begins a conversation turn      (User)
    What is 2+2?
    <|im_end|>                                      ->  ChatML: ends a conversation turn
    
    <|im_start|>assistant                           -> ChatML: begins a conversation turn       (Assistant)   


The model is trained to predict the assistant's response given this exact sequence. 
The opening <|im_start|>assistant  with no closing <|im_end|> signals to the model that generation should begin. 
The model learns during fine-tuning to produce relevant, role-appropriate completions that follow this structure.

Different models use different delimiter conventions. 
Anthropic's Claude uses a format with <|HUMAN_TURN|> and <|ASSISTANT_TURN|> boundaries in earlier versions, 
while Llama 3 uses [INST] / [/INST] tags within <<SYS>> blocks, and Mistral models use similar instruction tags. 
The key principle is identical across all: clear, trainable structural markers that tell the model whose voice each 
segment represents.

## 2.2  The System Prompt

The system prompt is a privileged block that precedes the conversation. It is used to establish the model's persona, 
constraints, capabilities, and any task-specific context. Because the system prompt occupies the beginning of the 
context and is processed first, its activations influence all subsequent tokens through attention.

System prompts are not inherently more "trusted" than user input at the architectural level — both are just tokens. 
The differentiation comes from fine-tuning: the model is trained to give the system role's instructions priority over 
potentially conflicting user instructions. Some deployment frameworks enforce this at the infrastructure level by 
separating system tokens from user tokens and masking them from certain attention computations, but this is an 
implementation detail, not a property of the base transformer architecture.

**IMPORTANT**
A common misconception is that the system prompt is hidden from the model or encoded separately. 
In reality, it becomes part of the same linear token sequence as everything else. Its content is fully visible to the 
attention mechanism. Some providers route system tokens through different software pathways, but from the transformer's 
perspective they are just tokens at position 0 through N.

## 2.3  Conversation History Serialisation
In multi-turn conversations, all previous turns must be re-serialised into the token sequence on every request. 
There is no persistent internal state in a standard transformer inference setup between requests. 
The model has no memory other than what is currently in its context window.

The practical consequence is that conversation cost grows linearly with history length. 
A ten-turn conversation with 500 tokens per turn requires 5,000 tokens of context before the model even sees the new
message. 
Providers typically implement conversation truncation strategies — dropping the oldest turns, 
summarising compressed history, or using sliding windows — to manage this.

## 2.4  Tool / Function Calling Formats

Tool use adds another layer of structural complexity. When a model needs to call an external function, 
it emits a structured output — typically JSON inside a special delimiter — that the inference runtime intercepts and 
executes. 
The result is then injected back into the conversation as a new turn.

The full tool-use cycle in the token sequence looks like:
    
    •	User turn with query tokens.
    
    •	Assistant turn opens, model emits <|tool_call|> delimiter.
    
    •	Model emits JSON: function name, arguments as structured tokens.
    
    •	<|tool_call_end|> delimiter.
    
    •	Runtime executes the function; result is injected as a tool-result turn.
    
    •	Assistant turn reopens; model continues generating the final response.

Each of these back-and-forth cycles adds hundreds to thousands of tokens and requires a fresh pre-fill pass over the 
growing context. 
Efficient tool use therefore critically depends on fast prompt processing and caching.

## 2.5  Multi-Modal Inputs

Vision-language models extend the linear token sequence to include image tokens. 
Images are typically passed through a separate vision encoder (e.g. a ViT patch encoder or CLIP) that converts the 
image into a grid of high-dimensional vectors. These vectors are projected into the text embedding space via a 
linear adapter and inserted into the token sequence at a placeholder position.

From the transformer's attention perspective, image patches look exactly like text tokens: they each occupy a position, 
carry an embedding vector, and participate in full bidirectional attention with surrounding text. 
The positional encoding for image patches may be 2D (encoding row and column) rather than the 1D encoding used for text, 
or images may use special learnable positional embeddings.


### 3. The Tokenisation Pipeline in Detail

The journey from raw string input to a tensor of integer IDs passes through several discrete steps. 
Understanding each step is essential for diagnosing unexpected token counts, debugging prompt injection, 
and optimising context budget usage.

## 3.1  Input Normalisation

Before the BPE or Unigram rules are applied, text undergoes normalisation. 
For BPE tokenisers this typically means NFC Unicode normalisation, which ensures that characters like 
é (é as a single code point) and e + é (e followed by a combining accent) are canonicalised to the same form. 
Without normalisation, identical-looking characters could tokenise differently depending on how they were produced 
(e.g. by different operating systems or keyboard layouts).

SentencePiece models often also apply NFKC normalisation which additionally maps compatibility characters — for example, 
the full-width digit ０ (０) — to their ASCII equivalents. 
This significantly reduces vocabulary fragmentation for multilingual text.


## 3.2  Pre-Tokenisation Splitting

Before the merge rules run, a regex splits the text into pre-token chunks. For GPT-4 / cl100k_base, the regex is:

**EXAMPLE**
	
	'(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

This regex ensures that common English contractions (don't, I've), sequences of letters, numbers in groups of up to 
three digits, punctuation strings, and whitespace are each treated as independent units before merging begins. 
Merges never cross pre-token boundaries. This is what prevents " New" (space + New) from ever partially merging with 
the preceding word to produce a nonsensical cross-boundary token.

## 3.3  Byte Encoding and Merge Application

Each pre-token chunk is encoded to UTF-8 bytes. 
In tiktoken (OpenAI's tokeniser library) this uses a Rust-based Aho-Corasick automaton to scan the byte sequence 
against the compiled merge table in a single linear pass, which is dramatically faster than the naive O(n^2) greedy 
approach. The result is a list of token IDs for each chunk.

The chunks are then concatenated to produce the final flat ID list. This concatenation is pure — no delimiter is 
inserted between pre-tokens; the split was only ever to constrain which bytes can merge together.

## 3.4  Special Token Injection

After the BPE encoding step, the tokenisation wrapper injects special tokens at structurally required positions. 
In an API call this injection is handled entirely by the serving infrastructure; the user never sends raw special token IDs. 

The wrapper:

    •	Prepends the BOS token if the model requires it.
    
    •	Wraps each turn in role-appropriate delimiters.
    
    •	Appends the EOS token after completed assistant turns.
    
    •	Inserts image placeholder tokens where multi-modal content was provided.
    
The resulting sequence is the complete prompt tensor, ready for the embedding lookup.

## 3.5  Encoding Edge Cases and Gotchas

    +-----------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
    | Edge Case                               | What Actually Happens                                                                                                                          |
    +-----------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
    | Leading spaces on words                 | " token" vs "token" are different IDs; always check whether your API strips or preserves whitespace at turn boundaries.                        |
    |_________________________________________|________________________________________________________________________________________________________________________________________________|   
    | Numbers > 3 digits                      | Large numbers like 12345 split into multiple tokens: "123", "45". Arithmetic on long numbers is therefore inherently                           | 
    |                                         | harder for transformers.                                                                                                                       |
    |_________________________________________|________________________________________________________________________________________________________________________________________________|
    | Non-ASCII in code                       | Variable names with non-ASCII chars (e.g. Greek letters as variables) can split unexpectedly, costing 2–4x the token budget.                   |
    |_________________________________________|________________________________________________________________________________________________________________________________________________|
    | Repeated characters                     | "aaaaaaa" tokenises very differently from "aaaa" — the merge table may or may not have a rule for the exact repetition length.                 | 
    |_________________________________________|________________________________________________________________________________________________________________________________________________|
    | Prompt injection strings                | Strings that contain literal delimiter tokens (e.g. the string "<|endoftext|>") may or may not be sanitised depending on tokeniser config;     | 
    |                                         | some encode them as regular text, others treat them as the actual special token.                                                               |
    |_________________________________________|________________________________________________________________________________________________________________________________________________|
    | Token healing                           | Some inference runtimes apply token healing — adjusting the last token at the boundary between prompt and generation —                         | 
    |                                         | to avoid bias from partial words.                                                                                                              |
    |_________________________________________|________________________________________________________________________________________________________________________________________________|

## 3.6  Detokenisation

After the model has produced a list of output token IDs, they must be decoded back to a string. 
For BPE tokenisers, each token maps back to a byte string via the vocabulary, and the byte strings are concatenated and 
decoded as UTF-8. For SentencePiece models, the ▁ meta-symbol is replaced with a space and leading/trailing spaces are 
trimmed.

Detokenisation is typically straightforward but can produce unexpected results at boundaries. 
Streaming APIs yield tokens one at a time, but some tokens are mid-character multi-byte sequences in UTF-8. 
A robust streaming detokeniser must buffer incomplete multi-byte sequences until enough bytes arrive to decode a 
complete Unicode code point before forwarding text to the user interface.

### Context Window Exploration 

---

## **Question**

Consider a scenario where we are on the fifth prompt so does that mean while prompt assembly the history of all 
previous prompts is added in token format to the current prompt tokens 

In Prompt Assembly 

chat begins  - [user prompt] + previous chat history ->  LLM 

## Explanation 

What actually happens on prompt 5:
The serving layer assembles a single flat token sequence that looks like this:

    [system prompt]
    [user turn 1]  [assistant turn 1]
    [user turn 2]  [assistant turn 2]
    [user turn 3]  [assistant turn 3]
    [user turn 4]  [assistant turn 4]
    [user turn 5]  ← new message, generation starts here

The entire thing — all 5 turns — is serialised into one continuous integer ID array and sent to the model in a single 
pre-fill pass. The model has no separate "memory" of previous turns. It only knows what it said in turns 1–4 because 
those tokens are literally sitting in the context window right now.
The subtle detail your diagram is missing:

The history is not just the user prompts — it includes the assistant's own previous responses too. 

So it's more accurately:

chat begins → [system] + [prompt1 + assistant1 + prompt2 + assistant2 ... + promptN] → LLM

The direct consequence of this:

Every new turn gets more expensive than the last, because the pre-fill token count keeps growing. 
By turn 10 of a verbose conversation you might be pre-filling 8,000+ tokens just to process a 20-token question. 
This is exactly why providers implement conversation truncation, summarisation, 
and prefix caching — the architecture has no native way to "remember" without re-reading.


##### **One context window per session — but with an important nuance on what "maintained" actually means.**

There is no persistent context window sitting in memory between turns. 
The context window is rebuilt from scratch on every single API call. 
What gives the illusion of a persistent session is that the client (your app, Claude.ai, ChatGPT etc.) stores the 
conversation history and re-sends it every time.

**So the flow**

    Turn 1:  [system + user1]                              → LLM → response1
    Turn 2:  [system + user1 + assistant1 + user2]         → LLM → response2
    Turn 3:  [system + user1 + assistant1 + user2 + 
              assistant2 + user3]                          → LLM → response3


    system     = the system prompt (instructions to the LLM)
    user1      = your first message / prompt
    assistantx = the LLM's response to your x message
    user2      = your second message / prompt
    
Each turn is a completely fresh, independent API call. 
The model doesn't "remember" turn 1 — it re-reads it every single time.

    +-------------------+------------------------------+------------------------------+
    |                   | What you think               | What actually happens        |
    +-------------------+------------------------------+------------------------------+
    | Memory            | Model remembers the chat     | History re-sent every call   |
    +-------------------+------------------------------+------------------------------+
    | Context window    | Persistent per session       | Rebuilt fresh each call      |
    +-------------------+------------------------------+------------------------------+
    | Cost              | Fixed per message            | Grows with every turn        |
    +-------------------+------------------------------+------------------------------+
    | State             | Stored in the model          | Stored on the client side    |
    +-------------------+------------------------------+------------------------------+

### **The one exception — KV Cache:**

This is where prefix caching comes in. Even though the context is logically rebuilt each time, the server can cache the 
pre-computed KV activations from the previous turns so it doesn't have to re-process them from scratch. 
The tokens are still "re-sent," but the compute is saved. This is the closest thing to a truly persistent context 
window in production systems.

So the short answer: one growing context window per session, 
but it's reconstructed on every single call, not held between them.


### **4. Context Windows, Attention, and the KV Cache**

The context window is the maximum number of tokens a model can process in a single forward pass. 
It is the most fundamental resource constraint in LLM deployment. 
Understanding how the context window works internally — through the key-value (KV) cache — is essential for 
understanding both inference cost and capability.

## 4.1  The Self-Attention Mechanism Refresher

In a transformer, each token at position i can attend to every token at positions 0 through i 
(in causal / decoder-only models) via the self-attention mechanism. 
For each attention head, every token generates three vectors: a query (Q), a key (K), and a value (V). 
Attention is computed as:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V  
    
    Where d_k is the key dimension. 
    
The scale factor prevents the dot products from entering the region where the softmax gradient vanishes.

The key insight is that computing attention for one token requires comparing its query against all previous keys. 
For a sequence of N tokens, this is O(N^2) in memory and O(N^2) in compute, 
which is why large context windows are expensive.

### 4.2  Pre-Fill vs. Decode Phases

Inference for a generative model has two fundamentally different phases:

## Pre-Fill (Prompt Processing)

All prompt tokens are known in advance. 
The model processes the entire sequence in a single batched matrix multiplication. 
Every token's Q, K, and V projections are computed simultaneously. 
This is highly parallelisable and GPU-utilisation-friendly. 
Pre-fill time scales roughly linearly with prompt length for the matrix multiplications, but the O(N^2) attention 
computation adds a quadratic component for very long prompts.

## Auto-Regressive Decode (Generation)

After pre-fill, the model generates one token at a time. 
For each new token, it must compute attention against all previous tokens. 
Without caching, this would require re-computing K and V for every previous token on every decode step — an enormous redundancy. 
This is where the KV cache becomes critical.

## 4.3  The KV Cache: Architecture and Memory

The KV cache stores the pre-computed K and V matrices for all previous tokens. 

When generating token N+1, the model:

    •	Computes Q, K, V only for the new token.
    •	Retrieves cached K and V for all previous tokens.
    •	Appends the new K and V to the cache.
    •	Runs attention using the full cached K/V sequence.

This reduces each decode step from O(N^2) compute to O(N) — a dramatic saving. 
However, the memory cost is substantial.
For each layer, for each attention head, for each token, two vectors (K and V) of dimension d_k must be stored. 

The total KV cache memory for a single sequence of length N is:

    Memory = 2 * num_layers * num_heads * d_head * N * bytes_per_element  

    For a 70B-parameter model with 80 layers, 
    64 heads,
    d_head=128, 
    N=128K tokens, 
    fp16: 2 * 80 * 64 * 128 * 131072 * 2 bytes ≈ 429 GB  
    
    This is why KV cache, not model weights, is often the binding memory constraint in long-context inference.
    
    
### 4.4  Multi-Query and Grouped-Query Attention

## A Little Background for K, Q, V

The model has 3 separate weight matrices, all randomly initialised then learned via backpropagation 
over billions of examples in training.
      
    Query (Q) → what you are looking for
    Key (K) → labels/index of stored items
    Value (V) → actual content of those items

---

Once training is done, W_Q, W_K, and W_V are frozen. That is the model file you download or call via API.

But it is worth understanding what "fixed" actually means here, because it is the key insight into how transformers work:

    Step 2a:
    ╔══════════════════════════════════════════════════════════════════╗
    ║  FIXED WEIGHTS  vs  DYNAMIC QKV VALUES                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      W_Q, W_K, W_V  ──────────────────────────────  FIXED after training
                                                      same for every sentence
                                                      stored in the model file
                                                      never change at inference
    
      Q, K, V        ──────────────────────────────  DYNAMIC every inference
                                                      different for every input
                                                      computed fresh each call
                                                      exist only during the
                                                      forward pass
      
So what the training actually did was teach the weights how to project meaning, not memorise specific sentences:
    
      W_Q learned:  "how to extract what a token is LOOKING FOR"
      W_K learned:  "how to extract what a token ADVERTISES"
      W_V learned:  "how to extract what a token SHARES"
    
      Then at inference:
    
      "The cat sat"   →   X · W_Q  =  Q  (unique to this sentence)
      "Dogs bark"     →   X · W_Q  =  Q  (completely different Q)
      "Rain falls"    →   X · W_Q  =  Q  (completely different Q)
    
      Same W_Q every time.
      Different X every time.
      Therefore different Q every time.
      
This is the elegant part — the weights encode general linguistic knowledge, and the input X is what makes each 
sentence's Q, K, V unique. 
The weights learned things like "nouns tend to advertise their role as subjects" and 
"verbs tend to query for their subject" — across all of language — not just for "cat" and "sat" specifically.

One more layer to this: this is also why the model has no memory between conversations. 
The weights are fixed, and Q/K/V only exist during that single forward pass. When the conversation ends, 
those values are gone. Nothing was updated. The weights are identical to before you started talking.


    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHERE THE WEIGHTS LIVE                                          ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      AFTER TRAINING
      ──────────────
    
      All weights including W_Q, W_K, W_V are serialised
      and saved to disk as a model file (or sharded files)
    
      Format examples:
      ┌──────────────────────────────────────────────────┐
      │  llama-3-70b.safetensors   (Meta / open source)  │
      │  model.bin                 (PyTorch format)      │
      │  model.gguf                (quantised local)     │
      │  weights/shard-00001.bin   (large models split   │
      │  weights/shard-00002.bin    across multiple      │
      │  weights/shard-00003.bin    files)               │
      └──────────────────────────────────────────────────┘
    
      These files are just tensors (multi-dimensional arrays
      of numbers) serialised to bytes on disk.
    
      A 7B model  ≈  14 GB  (fp16)
      A 70B model ≈ 140 GB  (fp16)
      A 405B model≈ 810 GB  (fp16)
      
    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHAT HAPPENS WHEN A USER SENDS A PROMPT                         ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      BEFORE first request ever arrives (server startup)
      ─────────────────────────────────────────────────
    
      1. Model file read from disk
            │
            ▼
      2. All weight tensors loaded into GPU VRAM
         (this is the slow part — can take 30–120 seconds)
            │
            ▼
      3. Model sits in GPU memory, IDLE, waiting
         W_Q, W_K, W_V are just sitting in VRAM
         ready to be used instantly
    
      ┌──────────────────────────────────────────────┐
      │              GPU VRAM                        │
      │                                              │
      │   Layer 1:  W_Q  W_K  W_V  W_FFN  ...        │
      │   Layer 2:  W_Q  W_K  W_V  W_FFN  ...        │
      │   Layer 3:  W_Q  W_K  W_V  W_FFN  ...        │
      │   ...                                        │
      │   Layer 32: W_Q  W_K  W_V  W_FFN  ...        │
      │                                              │
      │   All frozen. All waiting. ~140GB for 70B.   │
      └──────────────────────────────────────────────┘
    
    
      WHEN user sends "The cat sat"
      ─────────────────────────────
    
      4. Prompt is tokenised → integer IDs
            │
            ▼
      5. Token IDs looked up in embedding table
         (also stored in VRAM) → matrix X
            │
            ▼
      6. X is multiplied against W_Q, W_K, W_V
         that are already sitting in VRAM
            │
            ▼
      7. Q, K, V computed fresh for this prompt
            │
            ▼
      8. Attention → FFN → repeat for all layers
            │
            ▼
      9. Logits → sample → next token
            │
            ▼
      10. Streamed back to user
    
      Weights were NEVER reloaded from disk.
      They were READ from VRAM (nanoseconds, not milliseconds).
  
    ╔══════════════════════════════════════════════════════════════════╗
    ║  DISK  vs  VRAM  vs  COMPUTE — the three distinct stages         ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      DISK (SSD / NFS)
      ┌──────────────────────────────────────────┐
      │  model.safetensors                       │
      │  Permanent storage.                      │
      │  Weights never change here after         │
      │  training is done.                       │
      │  Read speed: ~5–10 GB/s (NVMe SSD)       │
      └───────────────────┬──────────────────────┘
                          │  loaded ONCE at startup
                          ▼
      GPU VRAM (HBM memory on the GPU chip)
      ┌──────────────────────────────────────────┐
      │  W_Q  W_K  W_V  W_FFN  ...  (all layers) │
      │  Weights live here permanently while     │
      │  server is running.                      │
      │  Read speed: ~3000 GB/s (HBM3)           │
      │  This is what makes inference fast.      │
      └───────────────────┬──────────────────────┘
                          │  multiplied against X
                          │  on every prompt
                          ▼
      CUDA CORES (actual compute)
      ┌──────────────────────────────────────────┐
      │  X · W_Q  →  Q   (matrix multiply)       │
      │  X · W_K  →  K   (matrix multiply)       │
      │  X · W_V  →  V   (matrix multiply)       │
      │                                          │
      │  Lives here for microseconds then gone.  │
      │  Q, K, V are ephemeral — they exist      │
      │  only during the forward pass.           │
      └──────────────────────────────────────────┘

The weights are loaded from disk into GPU VRAM once at server startup, and then for every user prompt they are simply 
read from VRAM — never reloaded — and multiplied against the input to produce Q, K, V fresh each time.

The reason inference is fast is precisely because that disk → VRAM load already happened. 
By the time your prompt arrives, the weights are sitting a few nanoseconds away in high-bandwidth GPU memory, 
ready to go.

---

## K, Q, V Calculation Visualized 

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║   SENTENCE: "The cat sat"                                                    ║
    ║   STEP 1 — TOKENISATION & EMBEDDING LOOKUP                                   ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
      Raw text       →   Token IDs   →   Embedding Vectors (d_model = 4, simplified)
    
      "The"          →   [ 464  ]    →   e₁ = [ 0.2,  0.5, -0.1,  0.8 ]
      "cat"          →   [ 3797 ]    →   e₂ = [ 0.9,  0.1,  0.4, -0.3 ]
      "sat"          →   [ 3332 ]    →   e₃ = [ 0.1,  0.7,  0.6,  0.2 ]
                                               ↑
                                        each number is one
                                        dimension of meaning
                                        (real models use 4096+ dims)
    
      These 3 vectors are stacked into input matrix X:
    
           dim→  [d0    d1    d2    d3 ]
      X =  "The" [0.2   0.5  -0.1   0.8]   ← row 1
           "cat" [0.9   0.1   0.4  -0.3]   ← row 2
           "sat" [0.1   0.7   0.6   0.2]   ← row 3
    
      Shape of X = (3 tokens × 4 dimensions)    
    
---

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║   STEP 2 — THREE WEIGHT MATRICES ARE LEARNED DURING TRAINING                 ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
      The model has 3 separate weight matrices, all randomly initialised
      then learned via backpropagation over billions of examples.
    
      W_Q  (4×4)  "What am I looking for?"
      ┌                         ┐
      │  0.1   0.4  -0.2   0.3  │
      │  0.5  -0.1   0.7   0.2  │
      │ -0.3   0.6   0.1  -0.4  │
      │  0.2   0.3  -0.5   0.6  │
      └                         ┘
    
      W_K  (4×4)  "What do I contain?"
      ┌                         ┐
      │  0.3  -0.2   0.5   0.1  │
      │  0.1   0.7  -0.3   0.4  │
      │  0.6   0.2   0.1  -0.5  │
      │ -0.1   0.4   0.2   0.3  │
      └                         ┘
    
      W_V  (4×4)  "What will I actually share?"
      ┌                         ┐
      │  0.4   0.1   0.3  -0.2  │
      │ -0.2   0.5   0.1   0.4  │
      │  0.3  -0.3   0.6   0.1  │
      │  0.1   0.2  -0.1   0.5  │
      └                         ┘
    
      These weights are the SAME for every sentence. They are not
      recomputed — they are parameters stored in the model file.
      What changes per sentence is X (the input).
      


---

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║   STEP 3 — COMPUTING Q, K, V  (matrix multiplication)                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
      Formula:  Q = X · W_Q
                K = X · W_K
                V = X · W_V
    
      ┌──────────────────────────────────────────────────────────────┐
      │  Q = X · W_Q                                                 │
      │                                                              │
      │  [0.2   0.5  -0.1   0.8]         [q_The_d0 .. q_The_d3]      │
      │  [0.9   0.1   0.4  -0.3]  × W_Q =  [q_cat_d0 .. q_cat_d3]    │
      │  [0.1   0.7   0.6   0.2]         [q_sat_d0 .. q_sat_d3]      │
      │                                                              │
      │  Result (simplified numbers):                                │
      │                                                              │
      │        [d0    d1    d2    d3 ]                               │
      │  "The" [0.31  0.42 -0.18  0.55]  ← "The" is asking about:    │
      │  "cat" [0.18  0.29  0.44 -0.11]  ← "cat" is asking about:    │
      │  "sat" [0.52  0.37  0.21  0.43]  ← "sat" is asking about:    │
      │         ↑                                                    │
      │   QUERY = "what information does this token NEED?"           │
      └──────────────────────────────────────────────────────────────┘
    
      ┌──────────────────────────────────────────────────────────────┐
      │  K = X · W_K                                                 │
      │                                                              │
      │        [d0    d1    d2    d3 ]                               │
      │  "The" [0.22  0.48  0.11  0.33]  ← "The" is advertising:     │
      │  "cat" [0.61  0.14  0.38  0.25]  ← "cat" is advertising:     │
      │  "sat" [0.19  0.55  0.42  0.17]  ← "sat" is advertising:     │
      │         ↑                                                    │
      │   KEY = "what information does this token HAVE?"             │
      └──────────────────────────────────────────────────────────────┘
    
      ┌──────────────────────────────────────────────────────────────┐
      │  V = X · W_V                                                 │
      │                                                              │
      │        [d0    d1    d2    d3 ]                               │
      │  "The" [0.29  0.18  0.34  0.11]  ← what "The" will share     │
      │  "cat" [0.44  0.22  0.17  0.38]  ← what "cat" will share     │
      │  "sat" [0.15  0.51  0.29  0.44]  ← what "sat" will share     │
      │         ↑                                                    │
      │   VALUE = "the actual content I will pass to other tokens"   │
      └──────────────────────────────────────────────────────────────┘
  

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║   STEP 4 — ATTENTION SCORES  (Q · Kᵀ)                                        ║
    ║   "How much should each token attend to every other token?"                  ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
      For every token's Query, we take the dot product with every token's Key.
      Dot product = how well the "question" matches the "advertisement".
    
      Score(i, j) = Q_i  ·  K_j
    
      Example: how much should "sat" attend to "cat"?
    
        Q_sat = [0.52, 0.37, 0.21, 0.43]   ← sat's query
        K_cat = [0.61, 0.14, 0.38, 0.25]   ← cat's key
    
        dot product = (0.52×0.61) + (0.37×0.14) + (0.21×0.38) + (0.43×0.25)
                   =   0.317    +    0.052    +    0.080    +    0.108
                   =   0.557
    
    
      Full Score matrix  (Q · Kᵀ),  one row per querying token,
                                      one col per key token:
    
                  "The"  "cat"  "sat"   ← which token is being attended TO
                  ─────  ─────  ─────
      "The"  →  [ 0.38   0.41   0.29 ]  ← how much "The" attends to each
      "cat"  →  [ 0.22   0.61   0.48 ]  ← how much "cat" attends to each
      "sat"  →  [ 0.31   0.56   0.44 ]  ← how much "sat" attends to each
                  ↑
             "sat" attends
             most to "cat"
             (0.56 is highest)
             — makes sense!
             the cat is the
             one doing the
             sitting
    
    
      ┌──────────────────────────────────────────────────┐
      │  SCALE: divide every score by √d_k               │
      │                                                  │
      │  d_k = 4  →  √4 = 2                              │
      │                                                  │
      │  Why? Dot products grow large as dimension grows │
      │  Large values push softmax into flat regions     │
      │  where gradients vanish. Scaling fixes this.     │
      │                                                  │
      │  Scaled scores:                                  │
      │                                                  │
      │           "The"  "cat"  "sat"                    │
      │  "The"  [ 0.19   0.21   0.15 ]                   │
      │  "cat"  [ 0.11   0.31   0.24 ]                   │
      │  "sat"  [ 0.16   0.28   0.22 ]                   │
      └──────────────────────────────────────────────────┘
  
  
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║   STEP 5 — SOFTMAX  (turn scores into probabilities)                         ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
      Softmax is applied ROW by ROW.
      Each row must sum to 1.0  (it becomes a probability distribution)
    
      softmax([0.16, 0.28, 0.22])  for "sat" row:
    
           e^0.16  e^0.28  e^0.22
         ──────────────────────────
         sum of all three e^ values
    
        = 1.174 / (1.174 + 1.323 + 1.246)
        = 1.174 / 3.743  = ...etc
    
      Attention Weights  (after softmax):
    
                  "The"   "cat"   "sat"        row sums
                  ──────  ──────  ──────        ────────
      "The"  →  [ 0.34    0.36    0.30  ]  =   1.00  ✓
      "cat"  →  [ 0.28    0.38    0.34  ]  =   1.00  ✓
      "sat"  →  [ 0.30    0.38    0.32  ]  =   1.00  ✓
    
    
      Visualised as attention intensity:
    
      "sat" attending to:
    
        "The" ████████████░░░░░░░░  0.30  (some — articles matter)
        "cat" ███████████████░░░░░  0.38  (most — subject of action)
        "sat" █████████████░░░░░░░  0.32  (self — positional awareness)
      
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║   STEP 6 — WEIGHTED SUM OF VALUES  (final output)                            ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
      Each token's output = weighted sum of ALL value vectors
      Weight comes from the attention matrix row for that token
    
      Output for "sat":
    
        weights = [0.30,  0.38,  0.32]
                   ↑       ↑      ↑
                 "The"  "cat"  "sat"
    
        V_The = [0.29, 0.18, 0.34, 0.11]
        V_cat = [0.44, 0.22, 0.17, 0.38]
        V_sat = [0.15, 0.51, 0.29, 0.44]
    
        Output_sat = 0.30 × V_The
                   + 0.38 × V_cat
                   + 0.32 × V_sat
    
        = 0.30 × [0.29, 0.18, 0.34, 0.11]
        + 0.38 × [0.44, 0.22, 0.17, 0.38]
        + 0.32 × [0.15, 0.51, 0.29, 0.44]
    
        = [0.087, 0.054, 0.102, 0.033]   (from "The")
        + [0.167, 0.084, 0.065, 0.144]   (from "cat")
        + [0.048, 0.163, 0.093, 0.141]   (from "sat")
          ───────────────────────────────────────────
        = [0.302, 0.301, 0.260, 0.318]   ← Output_sat
    
        "sat" now carries information from ALL tokens,
        blended by how relevant each one was to it.
        
        
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║   FULL PIPELINE SUMMARY                                                      ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
      "The cat sat"
           │
           ▼
      ┌─────────────┐
      │  Tokenise   │  "The"=464, "cat"=3797, "sat"=3332
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │   Embed     │  lookup table → 3 vectors → matrix X (3×4)
      └──────┬──────┘
             │
             ├──────────────────────────────────────────────┐
             │                                              │
        X · W_Q                X · W_K                    X · W_V
             │                    │                         │
             ▼                    ▼                         ▼
        Q (3×4)              K (3×4)                     V (3×4)
      "what I need"       "what I have"               "what I give"
             │                    │
             └────────┬───────────┘
                      │
                      ▼
               Q · Kᵀ  ÷ √d_k
               (3×3 score matrix)
                      │
                      ▼
                  softmax
               (3×3 weight matrix,
                each row sums to 1)
                      │
                      └──────────────────────┐
                                             │
                                        weights · V
                                             │
                                             ▼
                                       Output (3×4)
                             context-aware token vectors
                             each token now knows about
                             all other tokens it attended to


---
 
## Multi Head Atttention & Grouped-Query Attention 

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║           STANDARD MULTI-HEAD ATTENTION (MHA) — BASELINE                     ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
      Input Token Embeddings
      ┌─────────────────────┐
      │   Hidden State (X)  │
      └──────────┬──────────┘
                 │
        ┌────────┴────────┐
        │  Linear Project │  ← same input, 3 separate projections
        └─────┬─────┬─────┘
              │     │
             Q│    K│    V│
        ──────┼─────┼─────┼──────────────────────────────
       HEAD 1 ↓     ↓     ↓
              Q₁   K₁    V₁   ← each head has its OWN K and V
              └─────┴─────┘
              Attention(Q₁,K₁,V₁)
        
       HEAD 2 ↓     ↓     ↓
              Q₂   K₂    V₂   ← each head has its OWN K and V
              └─────┴─────┘
              Attention(Q₂,K₂,V₂)
        
       HEAD 3 ↓     ↓     ↓
              Q₃   K₃    V₃   ← each head has its OWN K and V
              └─────┴─────┘
              Attention(Q₃,K₃,V₃)
        
       HEAD 4 ↓     ↓     ↓
              Q₄   K₄    V₄   ← each head has its OWN K and V
              └─────┴─────┘
              Attention(Q₄,K₄,V₄)
        ─────────────────────────────────────────────────
        
        KV Cache stores: K₁ K₂ K₃ K₄ V₁ V₂ V₃ V₄  (8 matrices per token)
                         ↑
                         expensive at long context lengths



    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║           MULTI-QUERY ATTENTION (MQA)                                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
        Input Token Embeddings
        ┌─────────────────────┐
        │   Hidden State (X)  │
        └──────────┬──────────┘
                   │
          ┌────────┴─────────┐
          │  Linear Project  │
          └─────┬──────┬─────┘
                │      │     │       
               Q│     K│    V│
          ──────┼──────┼─────┼──────────────────────────────
         HEAD 1 ↓     ↓     ↓
                Q₁    ┌K┐   ┌V┐   ← K and V are SHARED
                └─────┤ ├───┤ │     (only 1 K, 1 V for ALL heads)
                      │ │   │ │
          HEAD 2 ↓    │ │   │ │
                 Q₂   │K│   │V│
                └─────┤ ├───┤ │
                      │ │   │ │
          HEAD 3 ↓    │ │   │ │
                 Q₃   │K│   │V│
                └─────┤ ├───┤ │
                      │ │   │ │
          HEAD 4 ↓    │ │   │ │
                 Q₄   └K┘   └V┘
                 └─────┴─────┘
        ─────────────────────────────────────────────────
        
        KV Cache stores: K  V  only  (2 matrices per token)
                         ↑
                         4x cheaper than MHA (with 4 heads)
        
        Trade-off: all heads see identical context → less expressive




    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║           GROUPED-QUERY ATTENTION (GQA) — used in LLaMA 3, Mistral           ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
          Input Token Embeddings
          ┌─────────────────────┐
          │   Hidden State (X)  │
          └──────────┬──────────┘
                     │
            ┌────────┴────────┐
            │  Linear Project │
            └─────┬─────┬─────┘
                  │     │
                 Q│    K│    V│
            ──────┼─────┼─────┼──────────────────────────────
        
                      GROUP A               GROUP B
                 (shared K_A, V_A)     (shared K_B, V_B)
                      ↓     ↓               ↓     ↓
            HEAD 1 → Q₁    ┌K_A┐  HEAD 3 → Q₃   ┌K_B┐
                      └────┤   │           └────┤   │
                           │   │                │   │
            HEAD 2 → Q₂    │K_A│  HEAD 4 → Q₄   │K_B│
                      └────┤   │           └────┤   │
                           └───┘                └───┘
        
                           ┌───┐                ┌───┐
                           │V_A│                │V_B│
                           └───┘                └───┘
        
                      Attn(Q₁,K_A,V_A)    Attn(Q₃,K_B,V_B)
                      Attn(Q₂,K_A,V_A)    Attn(Q₄,K_B,V_B)
            ─────────────────────────────────────────────────
        
          KV Cache stores: K_A  K_B  V_A  V_B  (4 matrices per token)
                           ↑
                           2x cheaper than MHA, 2x richer than MQA
        
          G = number of groups (tunable hyperparameter)
          G=1 → collapses to MQA
          G=H → collapses to MHA (H = number of heads)
          
          
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║           SIDE-BY-SIDE COMPARISON                                            ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
                      MHA          MQA          GQA
                    ────────     ────────     ────────
      Q projections    H            H            H      ← always H
      K projections    H            1            G      ← G between 1 and H
      V projections    H            1            G      ← G between 1 and H
    
      KV cache size   2×H          2×1          2×G
      per token
    
      Expressiveness  high         low          medium
      Memory cost     high         low          medium
      Quality         best         degraded     near-MHA
    
      Used in         BERT,        PaLM         LLaMA 3
                      GPT-2        (early)      Mistral
                                                Gemma 2
                                                GPT-4 (likely)
    
      Example (H=8):
      KV matrices     16           2            4 (G=2)
      per token                                 or 8 (G=4)
      
To reduce KV cache pressure, modern models use Multi-Query Attention (MQA) or Grouped-Query Attention (GQA). 
In MQA, all query heads share a single K and V projection, reducing KV cache size by a factor equal to the 
number of heads. 
In GQA, heads are divided into groups that share K and V projections, providing a tunable trade-off between 
cache size and model expressivity.

Llama 3, Mistral, and other recent models all use GQA. GPT-4 likely uses MQA or GQA internally. 
This is a significant practical consideration: the effective context window size that can be served in a given 
GPU memory budget is much larger with GQA than with standard multi-head attention.


## 4.5  Prefix Caching
      
A natural optimisation for repeated deployments with the same system prompt is prefix caching: storing the KV activations 
for the system prompt tokens on the server and reusing them across requests. 
This eliminates the need to re-compute pre-fill for the static system prompt on every call.

Anthropic, OpenAI, and Google all offer some form of prompt caching. 

The economic benefit is significant: if the system prompt is 10,000 tokens long and the average conversation has 
1,000 tokens of user history, caching the system prompt reduces pre-fill cost by roughly 90%. 
The technical requirement is that the cached prefix must begin at position 0 and be byte-identical across requests.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FIRST: RECALL WHAT HAPPENS WITHOUT PREFIX CACHING               ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Every single request recomputes EVERYTHING from scratch
    
      REQUEST 1
      ─────────
      Tokens:  [System Prompt 500 tokens] [User: "What is AI?"]
                    │
                    ▼
              PRE-FILL PASS
              compute Q,K,V for ALL 506 tokens
              store KV cache for all 506 tokens
                    │
                    ▼
              generate response
              KV cache discarded when request ends ←── PROBLEM
    
    
      REQUEST 2  (next user, same system prompt)
      ─────────
      Tokens:  [System Prompt 500 tokens] [User: "What is ML?"]
                    │
                    ▼
              PRE-FILL PASS
              compute Q,K,V for ALL 506 tokens AGAIN
              same 500 system prompt tokens
              recomputed from scratch AGAIN         ←── WASTED WORK
                    │
                    ▼
              generate response
              KV cache discarded again
    
    
      REQUEST 3  (another user, same system prompt)
      ─────────
      Tokens:  [System Prompt 500 tokens] [User: "Explain GPT"]
                    │
                    ▼
              PRE-FILL PASS
              compute Q,K,V for ALL 507 tokens AGAIN ←── WASTED AGAIN

      
    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHAT PREFIX CACHING DOES                                        ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      The system prompt tokens NEVER CHANGE across requests.
      So their KV activations will ALWAYS be identical.
      Therefore: compute them ONCE. Store them. Reuse forever.
    
      ┌───────────────────────────────────────────────────┐
      │                    SERVER MEMORY                  │
      │                                                   │
      │  ┌─────────────────────────────────────────────┐  │
      │  │           PREFIX CACHE STORE                │  │
      │  │                                             │  │
      │  │  key:   hash("You are a helpful assistant   │  │
      │  │               ...500 tokens of system       │  │
      │  │               prompt...")                   │  │
      │  │                                             │  │
      │  │  value: KV matrices for all 500 tokens      │  │
      │  │         Layer 1:  K[0..499]  V[0..499]      │  │
      │  │         Layer 2:  K[0..499]  V[0..499]      │  │
      │  │         ...                                 │  │
      │  │         Layer 32: K[0..499]  V[0..499]      │  │
      │  │                                             │  │
      │  │  Stored permanently while server is running │  │
      │  └─────────────────────────────────────────────┘  │
      └───────────────────────────────────────────────────┘
      
    ╔══════════════════════════════════════════════════════════════════╗
    ║  REQUEST FLOW WITH PREFIX CACHING                                ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      REQUEST 1  (cold — cache is empty)
      ─────────────────────────────────
    
      Tokens: [■■■■■■■■■■■■■■■■■■■■] [User msg 6 tokens]
               ←── 500 system tokens ──→
    
                   │
                   ▼
             hash(system tokens) → not found in cache  CACHE MISS
                   │
                   ▼
             pre-fill ALL 506 tokens  (full compute cost)
                   │
                   ├──────────────────────────────────────────┐
                   │                                          ▼
                   ▼                               SAVE to prefix cache
             generate response                     KV for tokens 0-499
                                                   stored under hash key
    
    
      REQUEST 2  (warm — cache has system prompt)
      ──────────────────────────────────────────
    
      Tokens: [■■■■■■■■■■■■■■■■■■■■] [User msg 5 tokens]
               ←── same 500 tokens ──→
    
                   │
                   ▼
             hash(system tokens) → FOUND in cache       CACHE HIT
                   │
                   ▼
             ┌─────────────────────────────────────────────┐
             │  SKIP pre-fill for tokens 0-499             │
             │  load their KV directly from cache          │
             │  takes microseconds not milliseconds        │
             └─────────────────────────────────────────────┘
                   │
                   ▼
             pre-fill ONLY the 5 new user tokens
                   │
                   ▼
             generate response     ← dramatically faster TTFT
             
             
    ╔══════════════════════════════════════════════════════════════════╗
    ║  VISUALISING WHAT GETS RECOMPUTED vs REUSED                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      WITHOUT prefix caching
      ──────────────────────
    
      Request 1  [████████████████████████████████████░░░]
      Request 2  [████████████████████████████████████░░░░]
      Request 3  [████████████████████████████████████░░░░░]
      Request 4  [████████████████████████████████████░░]
                 ↑                                   ↑
                 recomputed every time            user msg
                 (system prompt 500 tokens)
    
    
      WITH prefix caching
      ───────────────────
    
      Request 1  [████████████████████████████████████░░░]
                  ↑ computed once, saved to cache ↑
    
      Request 2  [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░]
      Request 3  [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░]
      Request 4  [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░]
                  ↑ loaded from cache — zero compute ↑  ↑
                                                   new tokens only
    
      ████  =  computed (costs time + money)
      ▓▓▓▓  =  loaded from cache (nearly free)
      ░░░░  =  user message (always computed, always different)
      
      

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE HASHING MECHANISM — HOW THE CACHE KNOWS WHAT TO REUSE       ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      The cache key is a HASH of the token IDs at that prefix position.
      If even ONE token changes, the hash changes, cache is bypassed.
    
      SAME hash → cache hit:
      ┌─────────────────────────────────────────────────────────┐
      │ Request A tokens: [9642, 318, 257, 7613, 8796, ...]     │
      │ Request B tokens: [9642, 318, 257, 7613, 8796, ...]     │
      │                    ──────────────────────────           │
      │                    identical → same hash → HIT ✓        │
      └─────────────────────────────────────────────────────────┘
    
      DIFFERENT hash → cache miss:
      ┌─────────────────────────────────────────────────────────┐
      │ Request A tokens: [9642, 318, 257, 7613, 8796, ...]     │
      │ Request C tokens: [9642, 318, 999, 7613, 8796, ...]     │
      │                              ↑                          │
      │                         one token differs               │
      │                         different hash → MISS ✗         │
      └─────────────────────────────────────────────────────────┘
    
      This is why the rule is:
      prefix must start at position 0
      and be BYTE-IDENTICAL across requests.
    
      Even a single extra space or punctuation mark
      in the system prompt = cache miss = full recompute.


    ╔══════════════════════════════════════════════════════════════════╗
    ║  PREFIX CACHING IN MULTI-TURN CONVERSATIONS                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      The cache is not just for system prompts.
      In a long conversation, earlier turns can also be cached.
    
      Turn 1  [SYS][U1][A1]
                           ↑ saved to cache after turn 1
    
      Turn 2  [SYS][U1][A1][U2][A2]
              [▓▓▓][▓▓][▓▓]           ← loaded from cache
                             [U2][A2] ← computed fresh
                                      ↑ saved to cache after turn 2
    
      Turn 3  [SYS][U1][A1][U2][A2][U3][A3]
              [▓▓▓][▓▓][▓▓][▓▓][▓▓]        ← loaded from cache
                                    [U3][A3]← computed fresh
    
      Turn 4  [SYS][U1][A1][U2][A2][U3][A3][U4][A4]
              [▓▓▓][▓▓][▓▓][▓▓][▓▓][▓▓][▓▓]       ← loaded
                                            [U4][A4]← fresh
    
    
      Cost per turn:
      ┌────────┬───────────────────┬──────────────────────┐
      │  Turn  │  Tokens computed  │  Tokens from cache   │
      ├────────┼───────────────────┼──────────────────────┤
      │   1    │  506  (100%)      │   0                  │
      │   2    │  ~50  (9%)        │   506  (91%)         │
      │   3    │  ~60  (7%)        │   ~556 (93%)         │
      │   4    │  ~55  (6%)        │   ~616 (94%)         │
      └────────┴───────────────────┴──────────────────────┘
                                    ↑
                         cost of each turn stays LOW
                         even as conversation grows LONG


    ╔══════════════════════════════════════════════════════════════════╗
    ║  FULL PICTURE — DISK → VRAM → PREFIX CACHE → INFERENCE           ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      ┌──────────┐   load at    ┌──────────────────────────────────┐
      │  Disk    │   startup    │           GPU VRAM               │
      │          │─────────────▶│                                  │
      │ model    │              │  W_Q  W_K  W_V  (frozen weights) │
      │ .safeten │              │                                  │
      │ sors     │              │  ┌────────────────────────────┐  │
      └──────────┘              │  │     PREFIX CACHE STORE     │  │
                                │  │                            │  │
                                │  │  hash_A → KV[0..499]       │  │
                                │  │  hash_B → KV[0..299]       │  │
                                │  │  hash_C → KV[0..1023]      │  │
                                │  │                            │  │
                                │  └────────────────────────────┘  │
                                └────────────────┬─────────────────┘
                                                 │
                                user prompt arrives
                                                 │
                                ┌────────────────▼──────────────────┐
                                │         INFERENCE ENGINE          │
                                │                                   │
                                │  1. hash(system + history)        │
                                │  2. cache hit? → load KV          │
                                │     cache miss? → recompute       │
                                │  3. pre-fill only NEW tokens      │
                                │  4. decode → stream to user       │
                                └───────────────────────────────────┘
                                
                                
The core idea in one sentence: the prefix cache saves the KV matrices for tokens that are identical across requests, 
so the GPU never multiplies the same numbers twice — it just reads the result it already computed from memory.


## 4.6  Position Encodings

Transformers have no inherent notion of order — all tokens are processed in parallel. 
Position encodings inject positional information. The main approaches are:

+---------------------------+-----------------------------+------------------------------------------------------------------------------------------------+
| Method                    | Models                      | Key Properties                                                                                 |
+---------------------------+-----------------------------+------------------------------------------------------------------------------------------------+
| Sinusoidal (absolute)     | Original Transformer, BERT  | Fixed sin/cos at each position; extrapolates poorly beyond training length.                    |
+---------------------------+-----------------------------+------------------------------------------------------------------------------------------------+
| Learned Absolute (ALiBi)  | GPT-2, early GPT-3          | Embedding table per position; does not extrapolate; simple to implement.                       |
+---------------------------+-----------------------------+------------------------------------------------------------------------------------------------+
| Rotary (RoPE)             | LLaMA, Mistral, GPT-NeoX    | Relative position encoded as rotation of Q/K vectors; extrapolates better with scaling tricks. |
+---------------------------+-----------------------------+------------------------------------------------------------------------------------------------+
| ALiBi                     | MPT, BLOOM                  | Attention logits penalised by linear distance; good zero-shot extrapolation.                   |
+---------------------------+-----------------------------+------------------------------------------------------------------------------------------------+
| YaRN / LongRoPE           | Extended LLaMA 3, Phi-3     | Extends RoPE base frequency or introduces non-uniform scaling to push context to 128K+.        |
+---------------------------+-----------------------------+------------------------------------------------------------------------------------------------+

RoPE (Rotary Position Embedding) encodes position as a phase rotation applied directly to the query and key vectors 
before the dot-product attention computation. 
Because the rotation depends only on the relative difference between two positions' indices, RoPE produces inherently 
relative position information, which is one reason it extrapolates better than absolute embeddings to sequence lengths 
not seen during training.


### 5. The Model Forward Pass — From Tokens to Logits

Once the prompt tensor exists and position encodings are prepared, the data flows through the transformer 
layers to produce a probability distribution over the vocabulary for the next token.

---

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SETUP: OUR INPUT                                                ║
    ║  Sentence: "The cat sat"  (3 tokens, d_model = 4, simplified)    ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      After tokenisation we have integer IDs:
    
      "The" → 464
      "cat" → 3797
      "sat" → 3332
    
      These 3 integers are the ONLY thing entering the model.
      Everything below is what happens next.

---

    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 1 — EMBEDDING LOOKUP TABLE                                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      The model has a giant lookup table  (vocab_size × d_model)
      stored in VRAM. Every token ID maps to a learned vector.
    
      Embedding Table (simplified, real = 128,256 rows × 4096 cols)
      ┌──────┬────────────────────────────────┐
      │  ID  │  Vector (4 dimensions shown)   │
      ├──────┼────────────────────────────────┤
      │  ... │  ...                           │
      │  463 │  [ 0.1,  0.3, -0.2,  0.5]      │
      │  464 │  [ 0.2,  0.5, -0.1,  0.8] ◄── "The"  looked up here
      │  465 │  [ 0.7, -0.2,  0.4,  0.1]      │
      │  ... │  ...                           │
      │ 3332 │  [ 0.1,  0.7,  0.6,  0.2] ◄── "sat"  looked up here
      │  ... │  ...                           │
      │ 3797 │  [ 0.9,  0.1,  0.4, -0.3] ◄── "cat"  looked up here
      │  ... │  ...                           │
      └──────┴────────────────────────────────┘
               ↑
               these numbers were learned during training
               they encode semantic meaning of each token
    
      Result after lookup → matrix X (3 tokens × 4 dims):
    
                 [d0    d1    d2    d3]
      "The" pos0 [0.2   0.5  -0.1   0.8]
      "cat" pos1 [0.9   0.1   0.4  -0.3]
      "sat" pos2 [0.1   0.7   0.6   0.2]

    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 2 — POSITIONAL ENCODING ADDED TO X                         ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Transformers process all tokens in parallel.
      Without positional info the model cannot tell
      "cat sat" from "sat cat" — order is invisible.
    
      A positional vector is ADDED to each token embedding.
      (RoPE rotates the Q/K vectors directly, but conceptually
       the idea is the same — inject position into the signal)
    
      Positional vectors (one per position, learned or fixed):
    
      pos0 = [ 0.0,  0.1,  0.0,  0.1]  ← position 0 signature
      pos1 = [ 0.1,  0.0,  0.1,  0.0]  ← position 1 signature
      pos2 = [ 0.0, -0.1,  0.1,  0.0]  ← position 2 signature
    
      X + positional encoding:
    
                 [d0    d1    d2    d3]
      "The" pos0 [0.2   0.6  -0.1   0.9]   ← 0.2+0.0, 0.5+0.1 ...
      "cat" pos1 [1.0   0.1   0.5  -0.3]   ← 0.9+0.1, 0.1+0.0 ...
      "sat" pos2 [0.1   0.6   0.7   0.2]   ← 0.1+0.0, 0.7-0.1 ...
    
      This is now X̃  (X with position baked in)
      This is what actually enters the first transformer layer.


    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 3 — ENTER THE TRANSFORMER LAYER STACK                      ║
    ║  (model has N identical layers, e.g. 32 for a 7B model)          ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    
      X̃ (3×4)
        │
        ▼
      ╔═════════════════════════════════════════════════════════════╗
      ║                  LAYER 1  of  32                            ║
      ║                                                             ║
      ║  ┌───────────────────────────────────────────────────────┐  ║
      ║  │              PRE-NORM (RMS Norm)                      │  ║
      ║  │                                                       │  ║
      ║  │  Normalises each token vector so values don't         │  ║
      ║  │  explode or vanish as they pass through layers.       │  ║
      ║  │                                                       │  ║
      ║  │  For each token row, rescale so RMS = 1.0             │  ║
      ║  │                                                       │  ║
      ║  │  "The" [0.2  0.6 -0.1  0.9] → [0.21  0.63 -0.10  0.94]│  ║  
      ║  │  "cat" [1.0  0.1  0.5 -0.3] → [0.85  0.08  0.42 -0.25]│  ║
      ║  │  "sat" [0.1  0.6  0.7  0.2] → [0.11  0.65  0.76  0.22]│  ║
      ║  └───────────────────────┬───────────────────────────────┘  ║
      ║                          │                                  ║
      ║                          ▼                                  ║
      ║  ┌─────────────────────────────────────────────────────┐    ║
      ║  │           SELF-ATTENTION SUB-LAYER                  │    ║
      ║  │                                                     │    ║
      ║  │  1. Project to Q, K, V via W_Q, W_K, W_V            │    ║
      ║  │     (weights loaded from VRAM)                      │    ║
      ║  │                                                     │    ║
      ║  │     Q = X̃ · W_Q    K = X̃ · W_K    V = X̃ · W_V       │    ║
      ║  │                                                     │    ║
      ║  │  2. Compute attention scores                        │    ║
      ║  │     scores = Q · Kᵀ  ÷  √d_k                        │    ║
      ║  │                                                     │    ║
      ║  │  3. Apply causal mask (future tokens → -∞)          │    ║
      ║  │     "The" can only see: "The"                       │    ║
      ║  │     "cat" can only see: "The", "cat"                │    ║
      ║  │     "sat" can only see: "The", "cat", "sat"         │    ║
      ║  │                                                     │    ║
      ║  │  4. Softmax → attention weights                     │    ║
      ║  │                                                     │    ║
      ║  │  5. Weighted sum of V                               │    ║
      ║  │     attention_out = weights · V                     │    ║
      ║  │                                                     │    ║
      ║  │  6. Project back to d_model via W_O                 │    ║
      ║  └───────────────────────┬─────────────────────────────┘    ║
      ║                          │                                  ║
      ║                          ▼                                  ║
      ║  ┌─────────────────────────────────────────────────────┐    ║
      ║  │              RESIDUAL CONNECTION  (+)               │    ║
      ║  │                                                     │    ║
      ║  │  output = X̃  +  attention_out                       │    ║
      ║  │           ↑         ↑                               │    ║
      ║  │      original    what attention                     │    ║
      ║  │       input       added/changed                     │    ║
      ║  │                                                     │    ║
      ║  │  Why? Prevents vanishing gradients.                 │    ║
      ║  │  The original signal always flows through.          │    ║
      ║  └───────────────────────┬─────────────────────────────┘    ║
      ║                          │                                  ║
      ║                          ▼                                  ║    
      ║  ┌─────────────────────────────────────────────────────┐    ║ 
      ║  │              PRE-NORM AGAIN (RMS Norm)              │    ║ 
      ║  └───────────────────────┬─────────────────────────────┘    ║ 
      ║                          │                                  ║
      ║                          ▼                                  ║
      ║  ┌─────────────────────────────────────────────────────┐    ║
      ║  │         FEED-FORWARD NETWORK (FFN) SUB-LAYER        │    ║
      ║  │                                                     │    ║
      ║  │  Applied to EACH token INDEPENDENTLY                │    ║
      ║  │  (no communication between tokens here)             │    ║
      ║  │                                                     │    ║
      ║  │  For each token vector h:                           │    ║
      ║  │                                                     │    ║
      ║  │  h → Linear(d_model → d_ff)                         │    ║
      ║  │         ↓    (expand: 4 → 16 dims in real models    │    ║
      ║  │              4096 → 16384 in a 7B model)            │    ║ 
      ║  │    SwiGLU / ReLU activation                         │    ║  
      ║  │    (introduces non-linearity so model               │    ║
      ║  │     can learn complex patterns)                     │    ║
      ║  │         ↓                                           │    ║
      ║  │  h → Linear(d_ff → d_model)                         │    ║
      ║  │         ↓   (compress back: 16 → 4)                 │    ║
      ║  │    richer token representation                      │    ║
      ║  └───────────────────────┬─────────────────────────────┘    ║ 
      ║                          │                                  ║
      ║                          ▼                                  ║    
      ║  ┌─────────────────────────────────────────────────────┐    ║ 
      ║  │              RESIDUAL CONNECTION  (+)               │    ║  
      ║  │                                                     │    ║  
      ║  │  output = prev_residual  +  ffn_out                 │    ║ 
      ║  └───────────────────────┬─────────────────────────────┘    ║
      ║                          │                                  ║
      ╚══════════════════════════╪══════════════════════════════════╝
                                 │
                                 │  same structure repeated
                                 ▼
      ╔═══════════════════════════════════════════════════════════╗
      ║                  LAYER 2  of  32                          ║
      ║  [identical structure — norm → attn → residual →          ║
      ║   norm → ffn → residual]                                  ║
      ╚══════════════════════════╪════════════════════════════════╝
                                 │
                                 ▼
                                ...
                                 │
                                 ▼
      ╔═══════════════════════════════════════════════════════════╗
      ║                  LAYER 32  of  32                         ║
      ║  [same structure]                                         ║
      ╚══════════════════════════╪════════════════════════════════╝
                                 │
                                 ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 4 — WHAT THE RESIDUAL STREAM LOOKS LIKE                    ║
    ║  (how the token representation evolves layer by layer)           ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      "sat" token representation as it flows through layers:
    
      After embedding:   [0.1,  0.7,  0.6,  0.2]
                          ↓ layer 1 attention adds context from "The", "cat"
      After layer 1:    [0.3,  0.8,  0.4,  0.5]   ← knows about "The cat"
                          ↓ layer 2 refines further
      After layer 2:    [0.5,  0.6,  0.7,  0.3]
                          ↓ ...
      After layer 8:    [0.7,  0.4,  0.9,  0.2]   ← syntax understood
                          ↓ ...
      After layer 16:   [0.6,  0.9,  0.5,  0.8]   ← semantics richer
                          ↓ ...
      After layer 32:   [0.8,  0.7,  0.6,  0.9]   ← full contextual meaning
    
      Each layer builds on the last.
      Early layers  → syntax, grammar, position
      Middle layers → semantics, coreference
      Late layers   → task-specific, next-token prediction
      

╔══════════════════════════════════════════════════════════════════╗
║  STEP 5 — FINAL LAYER NORM                                       ║
╚══════════════════════════════════════════════════════════════════╝

  After the last transformer layer, one final RMS Norm
  is applied to stabilise the output before the LM head.

  h_final (3 tokens × 4 dims):

  "The" [0.6,  0.5,  0.8,  0.4]
  "cat" [0.9,  0.3,  0.7,  0.6]
  "sat" [0.8,  0.7,  0.6,  0.9]  ← THIS row is what matters
                                      we only care about
                                      the LAST token's vector
                                      because that is where
                                      generation happens

              ┌─────────────────────────────────────┐
              │  Why only the last token?           │
              │                                     │
              │  The model is predicting what comes │
              │  AFTER "sat" — the next token.      │
              │  "sat" has already attended to      │
              │  everything before it. Its final    │
              │  hidden state encodes the full      │
              │  context of "The cat sat ___"       │
              └─────────────────────────────────────┘

    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 6 — THE LM HEAD  (hidden state → logits)                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      The LM Head is a single linear projection:
    
      h_sat (1 × 4)   ×   W_lm_head (4 × vocab_size)
                                           ↑
                                   100,256 tokens
                                   in cl100k vocab
    
      = logits (1 × 100,256)  ← one raw score per vocabulary token
    
      W_lm_head is often TIED to the embedding table
      (same matrix used in reverse — an elegant weight saving)
    
      ┌──────────────────────────────────────────────────────────┐
      │                                                          │
      │  h_sat = [0.8,  0.7,  0.6,  0.9]                         │
      │                                                          │
      │       × W_lm_head  (huge matrix multiplication)          │
      │                                                          │
      │  logits output (showing just a few of 100,256 values):   │
      │                                                          │
      │  token ID  │  token string  │  raw logit score           │
      │  ──────────┼────────────────┼─────────────────           │
      │   262      │  " the"        │   1.2                      │
      │   319      │  " on"         │   3.8  ◄── high score      │
      │   290      │  " and"        │   2.1                      │
      │   257      │  " a"          │   1.9                      │
      │   1657     │  " down"       │   2.7                      │
      │   4920     │  " quietly"    │   1.4                      │
      │   50256    │  <endoftext>   │   0.3                      │
      │   ...      │  ...           │   ...  (100,249 more rows) │
      │                                                          │
      │  These scores are NOT probabilities yet.                 │
      │  They are raw unnormalised numbers (logits).             │
      │  They can be negative, zero, or any positive value.      │
      └──────────────────────────────────────────────────────────┘
      
      
    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 7 — LOGITS → PROBABILITIES → NEXT TOKEN                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      logits = [... 1.2,  3.8,  2.1,  1.9,  2.7,  1.4 ...]
                                  │
                                  ▼
                        divide by temperature T
                        (T=1.0 means no change)
                                  │
                                  ▼
                               softmax
                        (converts to probabilities,
                         all values 0-1, sum = 1.0)
                                  │
                                  ▼
      token ID  │  token string  │  probability
      ──────────┼────────────────┼─────────────
       319      │  " on"         │   0.31  ◄── most likely
       1657     │  " down"       │   0.22
       290      │  " and"        │   0.15
       257      │  " a"          │   0.12
       262      │  " the"        │   0.09
       4920     │  " quietly"    │   0.06
       50256    │  <endoftext>   │   0.01
       ...      │  ...           │   0.04  (remaining ~100k tokens)
      ──────────┴────────────────┴─────────────
                                   1.00  ✓
    
                                  │
                                  ▼
                        sample from distribution
                        (top-p, top-k, temperature
                         all applied here)
                                  │
                                  ▼
                            next token = " on"
    
      Full sentence so far: "The cat sat on ___"
      Process repeats for next token.


    ╔══════════════════════════════════════════════════════════════════╗
    ║  COMPLETE FORWARD PASS — ONE DIAGRAM                             ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      "The cat sat"
           │
           ▼
      ┌─────────────────┐
      │  Tokenise       │  → [464, 3797, 3332]
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │  Embed          │  token IDs → vectors → matrix X (3×4096)
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │  Add Position   │  X = X + positional encoding
      └────────┬────────┘
               │
               ▼
      ┌─────────────────────────────────────────────────────────┐
      │  LAYER 1                                                │
      │  ┌──────────┐   ┌───────────────┐   ┌───────────────┐   │
      │  │ RMS Norm │──▶│ Self-Attention│──▶│  + Residual   │   │
      │  └──────────┘   └───────────────┘   └───────┬───────┘   │
      │                                             │           │
      │  ┌──────────┐   ┌───────────────┐   ┌───────▼───────┐   │
      │  │ RMS Norm │◀──│               │◀──│               │   │
      │  └────┬─────┘   └───────────────┘   └───────────────┘   │
      │       │                                                 │
      │  ┌────▼─────┐   ┌───────────────┐   ┌───────────────┐   │
      │  │          │──▶│      FFN      │──▶│  + Residual   │   │
      │  └──────────┘   └───────────────┘   └───────┬───────┘   │
      └────────────────────────────────────────────┬┘           │
                                                   │ ×32 layers
                                                   ▼
      ┌─────────────────┐
      │  Final RMS Norm │
      └────────┬────────┘
               │
               ▼  take ONLY the last token's vector
      ┌─────────────────┐
      │   LM Head       │  h (1×4096)  ×  W (4096×100256)
      └────────┬────────┘
               │
               ▼
      logits  (1 × 100,256)    ← one score per vocabulary token
               │
               ▼
      softmax + sampling
               │
               ▼
           "on"                ← next token predicted
    
      Feed "on" back in.
      Repeat until <endoftext>.
      
The key insight running through all of this: the token vector starts as a simple lookup of meaning, and each layer 
adds more context until by layer 32 it contains a full understanding of the entire sentence — and that final rich vector 
is what gets projected into a prediction over the whole vocabulary.


## 5.1  Token Embeddings
The integer token IDs are first passed through an embedding lookup table of shape [vocab_size × d_model]. 
Each ID maps to a dense vector of d_model dimensions (e.g. 4096 for a 7B model, 8192 for a 70B model). 
These vectors are the model's learned representation of each vocabulary entry.

In many architectures, the embedding matrix is tied — i.e. the same weight matrix is used for both input embedding 
and the final output projection. This weight tying reduces the number of parameters and encourages the input and 
output representation spaces to be coherent.

5.2  Transformer Layer Stack
The embedded sequence passes through a stack of identical transformer layers (typically 32–96 layers for modern LLMs). 

Each layer contains:

**Attention Sub-Layer**
Multi-head (or grouped-query) self-attention as described in Section 4. 
The output is projected back to d_model dimensionality and added to the layer input via a residual connection. 
Layer normalisation (typically RMS Norm in modern models like LLaMA) is applied — either Pre-LN (before the attention) 
or Post-LN (after), with Pre-LN being the current standard for training stability.

**Feed-Forward Sub-Layer**
A two-layer MLP (or SwiGLU / GeGLU gated variant in modern architectures) that applies a non-linearity. 
The intermediate dimension is typically 4x d_model for standard MLPs, or uses the gated variant with a smaller 
intermediate dimension. 
This layer is applied position-independently — the same MLP processes each token's representation without inter-token 
communication. 
All inter-token information flow happens exclusively in the attention sub-layer.

The feed-forward layer can be thought of as a key-value memory store: the first matrix computes which "memories" are
relevant to the current token, and the second matrix retrieves associated values. 
Research on mechanistic interpretability has shown that factual associations and linguistic transformations are largely 
stored in the MLP weights, while attention heads handle position-dependent relational reasoning.


## 5.3  The Final Layer: Logits and the LM Head

After all transformer layers, the final hidden state for the last position in the sequence 
(the token immediately before where generation will begin) is passed through the LM head — a linear projection 
from d_model to vocab_size. This produces a vector of raw scores called logits, one per vocabulary token.

These logits represent the model's unnormalised confidence that each vocabulary token should come next. 
They are not probabilities and can take any real value. Converting logits to probabilities requires a softmax, 
but in practice, sampling strategies operate on the logits directly (via temperature scaling) or on partially 
normalised subsets, to avoid the numerical instability of computing softmax over a 100,000-entry vocabulary.

## 5.4  Attention Masks

Attention masks are binary or float matrices that prevent certain token pairs from attending to each other. 
Two types matter for prompt processing:

## **Causal (Auto-Regressive) Mask**

In decoder-only models, token i is prevented from attending to token j if j > i. 
This causal mask enforces the auto-regressive property: the model can only use past information to predict the future. 
In practice this is implemented as a triangular mask that sets the attention logits for 
future positions to -infinity before the softmax, effectively zeroing out those attention weights.

## **Padding Mask**
When processing a batch of sequences of different lengths, shorter sequences must be padded to the maximum length 
in the batch. The padding tokens are meaningless and should not influence computation. 
A padding mask sets their attention logits to -infinity, preventing the model from attending to padding positions. 
It also prevents padding positions from having their loss computed during training.

## 6. Sampling Strategies — From Logits to Output Tokens

Given the logit vector for the next token position, the inference runtime must decide which token to produce. 
This decision involves several configurable strategies that together determine the character of the model's outputs.

---

### 6. Sampling Strategies — From Logits to Output Tokens

Given the logit vector for the next token position, the inference runtime must decide which token to produce. 
This decision involves several configurable strategies that together determine the character of the model's outputs.


    ╔══════════════════════════════════════════════════════════════════╗
    ║  SETUP: WHERE WE ARE                                             ║
    ║  The transformer has finished its forward pass.                  ║
    ║  We have a raw logits vector sitting in GPU memory.              ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      The last transformer layer processed "The cat sat"
      The LM head projected the final hidden state to vocab size.
    
      We now have:
    
      logits  =  one raw score for every token in the vocabulary
                 shape: (1 × 100,256)
    
      Showing just 10 of the 100,256 values:
    
      ┌──────────┬────────────────┬─────────────┐
      │  Token   │  String        │  Raw Logit  │
      ├──────────┼────────────────┼─────────────┤
      │  319     │  " on"         │    5.2      │
      │  1657    │  " down"       │    3.9      │
      │  290     │  " and"        │    3.1      │
      │  2402    │  " there"      │    2.8      │
      │  257     │  " a"          │    2.4      │
      │  1088    │  " back"       │    2.1      │
      │  262     │  " the"        │    1.8      │
      │  4920    │  " quietly"    │    1.4      │
      │  50256   │  <endoftext>   │    0.3      │
      │  2159    │  " still"      │    0.1      │
      │  ...     │  99,246 more   │  very low   │
      └──────────┴────────────────┴─────────────┘
    
      These numbers are NOT probabilities.
      Negative values are valid. No upper bound.
      They are raw unnormalised scores.
      Everything below converts them to a usable token.



    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 1 — TEMPERATURE SCALING                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Formula:  scaled_logit  =  raw_logit  ÷  T
    
      Temperature T is set by the caller (API parameter).
      It is applied BEFORE softmax.
    
      ┌──────────────────────────────────────────────────────────┐
      │  T = 0.0  →  GREEDY  (always pick highest)              │
      │  T = 1.0  →  RAW distribution (no change)               │
      │  T > 1.0  →  FLATTER  (more random)                     │
      │  T < 1.0  →  SHARPER  (more deterministic)              │
      └──────────────────────────────────────────────────────────┘
    
      Effect on our logits  (top 5 shown):
    
      Original:     [5.2,  3.9,  3.1,  2.8,  2.4]
                     ↓          ↓           ↓
      T = 0.5       [10.4, 7.8,  6.2,  5.6,  4.8]  ← gaps WIDEN
      (more focused) top token dominates even more
    
      T = 1.0       [5.2,  3.9,  3.1,  2.8,  2.4]  ← unchanged
      (neutral)
    
      T = 1.5       [3.5,  2.6,  2.1,  1.9,  1.6]  ← gaps NARROW
      (more random)  other tokens get more competitive
    
      Visualised as gap between top and second token:
    
      T=0.5   ████████████████████░░░░░░  gap: 2.6  (top dominates)
      T=1.0   ████████████░░░░░░░░░░░░░░  gap: 1.3
      T=1.5   ████████░░░░░░░░░░░░░░░░░░  gap: 0.9  (more even)
      T=2.0   ██████░░░░░░░░░░░░░░░░░░░░  gap: 0.65 (very flat)


    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 2 — SOFTMAX  (logits → probabilities)                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Formula:  P(token_i)  =  e^logit_i  ÷  Σ e^logit_j
                                              (sum over ALL tokens)
    
      Every logit gets exponentiated.
      Then divided by the total sum.
      Result: all values between 0 and 1, sum = 1.0 exactly.
    
      Working through our example at T=1.0:
    
      token " on"     logit=5.2  →  e^5.2  =  181.3
      token " down"   logit=3.9  →  e^3.9  =   49.4
      token " and"    logit=3.1  →  e^3.1  =   22.2
      token " there"  logit=2.8  →  e^2.8  =   16.4
      token " a"      logit=2.4  →  e^2.4  =   11.0
      ... (99,251 more tokens, all very small e^ values)
      total sum of all e^ values  ≈  300.1
    
      Probabilities:
    
      ┌──────────┬────────────────┬──────────┬────────────────────┐
      │  Token   │  String        │  P(T=1.0)│  P(T=0.5)          │
      ├──────────┼────────────────┼──────────┼────────────────────┤
      │  319     │  " on"         │  0.604   │  0.821  ← much     │
      │  1657    │  " down"       │  0.165   │  0.089    higher   │
      │  290     │  " and"        │  0.074   │  0.033             │
      │  2402    │  " there"      │  0.055   │  0.020             │
      │  257     │  " a"          │  0.037   │  0.011             │
      │  1088    │  " back"       │  0.027   │  0.006             │
      │  262     │  " the"        │  0.019   │  0.003             │
      │  4920    │  " quietly"    │  0.009   │  0.001             │
      │  50256   │  <endoftext>   │  0.002   │  0.000             │
      │  ...     │  99,247 more   │  0.009   │  0.016             │
      ├──────────┼────────────────┼──────────┼────────────────────┤
      │  TOTAL   │                │  1.000 ✓ │  1.000 ✓           │
      └──────────┴────────────────┴──────────┴────────────────────┘
    
      Visualised as bars:
    
      T = 1.0
      " on"      ████████████████████████░░░░░░░░░░░░  0.604
      " down"    ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.165
      " and"     ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.074
      " there"   ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.055
      others     ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.102
    
      T = 1.5
      " on"      ████████████████░░░░░░░░░░░░░░░░░░░░  0.412
      " down"    █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.201
      " and"     ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.148
      " there"   ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.121
      others     ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.118
                 ↑ many more tokens now viable
    
    
    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 3 — SAMPLING STRATEGY                                     ║
    ║  Three approaches. Only ONE is used per request.                ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    
      ── STRATEGY A:  GREEDY DECODING  (T=0 or argmax) ──────────────
    
      Just take the highest probability token. Always.
      No randomness whatsoever.
    
      probabilities: [0.604, 0.165, 0.074, 0.055, 0.037 ...]
                        ↑
                      argmax → always pick this one
    
      output: " on"  (every single time, deterministic)
    
      ┌────────────────────────────────────────────────────┐
      │  PROS: fast, deterministic, reproducible           │
      │  CONS: repetitive, boring, loops easily            │
      │  USE:  code completion, structured output          │
      └────────────────────────────────────────────────────┘
    
    
      ── STRATEGY B:  TOP-K SAMPLING ────────────────────────────────
    
      Keep only the K highest probability tokens.
      Zero out everything else.
      Sample randomly from those K tokens only.
    
      Example: K = 3
    
      Full distribution:
      " on"      0.604  ← keep  ✓
      " down"    0.165  ← keep  ✓
      " and"     0.074  ← keep  ✓
      " there"   0.055  ← ZERO OUT ✗
      " a"       0.037  ← ZERO OUT ✗
      " back"    0.027  ← ZERO OUT ✗
      ...99,250 more    ← ZERO OUT ✗
    
      Re-normalise the 3 survivors to sum to 1.0:
    
      " on"      0.604 ÷ 0.843  =  0.716
      " down"    0.165 ÷ 0.843  =  0.196
      " and"     0.074 ÷ 0.843  =  0.088
      ───────────────────────────────────
      total                         1.000 ✓
    
      Now spin the weighted wheel:
    
      ├──────────────────────────────────────┤───────────┤─────┤
      0                                     0.716      0.912  1.0
      └───────────── " on" ─────────────────┘─" down"──┘─"and"┘
    
      Random number drawn = 0.43  →  lands in " on" zone → pick " on"
      Random number drawn = 0.78  →  lands in " down" zone → pick " down"
      Random number drawn = 0.95  →  lands in " and" zone  → pick " and"
    
      ┌────────────────────────────────────────────────────┐
      │  PROS: prevents very unlikely tokens ever appearing│
      │  CONS: K is fixed regardless of distribution shape │
      │  USE:  common default in many systems              │
      └────────────────────────────────────────────────────┘
    
    
      ── STRATEGY C:  TOP-P (NUCLEUS) SAMPLING ──────────────────────
    
      Instead of a fixed count K, keep the SMALLEST SET
      of tokens whose cumulative probability exceeds P.
    
      Example: P = 0.90
    
      Sort by probability (descending), accumulate:
    
      Token      Prob    Cumulative   Keep?
      ─────────  ──────  ──────────   ─────
      " on"      0.604   0.604        ✓  (0.604 < 0.90, keep going)
      " down"    0.165   0.769        ✓  (0.769 < 0.90, keep going)
      " and"     0.074   0.843        ✓  (0.843 < 0.90, keep going)
      " there"   0.055   0.898        ✓  (0.898 < 0.90, keep going)
      " a"       0.037   0.935        ✓  (0.935 > 0.90, STOP here)
      " back"    0.027   —            ✗  zeroed out
      " the"     0.019   —            ✗  zeroed out
      ...more    ...     —            ✗  zeroed out
    
      Nucleus = [" on", " down", " and", " there", " a"]
      These 5 tokens cover 93.5% of probability mass.
    
      Re-normalise and sample from these 5 only.
    
      WHY THIS IS BETTER THAN TOP-K:
    
      Confident distribution  (model is sure):
      ┌──────────────────────────────────────────────────┐
      │  " on"  0.95,  " down"  0.03,  others 0.02      │
      │  Top-p (0.90) keeps:  1 token  (" on" alone)     │
      │  Top-k (K=50) keeps:  50 tokens (wrong!)         │
      └──────────────────────────────────────────────────┘
    
      Uncertain distribution  (model is unsure):
      ┌──────────────────────────────────────────────────┐
      │  Many tokens all near 0.02 probability           │
      │  Top-p (0.90) keeps:  45 tokens (adapts)         │
      │  Top-k (K=3)  keeps:  3 tokens (too restrictive) │
      └──────────────────────────────────────────────────┘
    
      Top-p adapts dynamically to model confidence.
      Top-k uses a rigid fixed count regardless.
    
      ┌────────────────────────────────────────────────────┐
      │  PROS: adapts to model certainty at each step     │
      │  CONS: slightly more expensive to compute         │
      │  USE:  default in most production APIs (p=0.9)    │
      └────────────────────────────────────────────────────┘



    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 4 — REPETITION PENALTY  (applied before softmax)          ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      If a token has already appeared in the context,
      its logit is divided by the penalty factor.
      This discourages the model from repeating itself.
    
      penalty = 1.3  (typical value)
    
      Tokens already seen in "The cat sat":
      " the" (ID 262) has appeared → penalise it
    
      Before penalty:   logit[262] = 1.8
      After penalty:    logit[262] = 1.8 ÷ 1.3  =  1.38
                                                  ↑
                                       lower score → less likely
                                       to repeat "the" again
    
      Tokens NOT seen yet are unaffected.
    
      ┌──────────┬───────────┬───────────────┬─────────────────────┐
      │  Token   │  Logit    │  Seen before? │  Adjusted Logit     │
      ├──────────┼───────────┼───────────────┼─────────────────────┤
      │  " on"   │  5.2      │  No           │  5.2   (unchanged)  │
      │  " down" │  3.9      │  No           │  3.9   (unchanged)  │
      │  " the"  │  1.8      │  YES          │  1.38  (penalised)  │
      │  " cat"  │  0.8      │  YES          │  0.62  (penalised)  │
      │  " sat"  │  0.5      │  YES          │  0.38  (penalised)  │
      └──────────┴───────────┴───────────────┴─────────────────────┘


    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 5 — TOKEN SELECTED, NOW DECODE IT BACK TO TEXT            ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Sampling selected token ID:  319
    
      Reverse lookup in the vocabulary table:
    
      ID 319  →  byte sequence [0x20, 0x6F, 0x6E]
               →  UTF-8 decode  →  " on"
                                   ↑
                              space included
                              (was baked into
                               the token during
                               BPE training)
    
      Append to output buffer:  "The cat sat" + " on"
                             =  "The cat sat on"
                             
                             
    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 6 — STOPPING CONDITIONS                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      The decode loop checks FOUR conditions after each token:
    
      ┌────────────────────────────────────────────────────────────┐
      │                                                            │
      │  1. EOS TOKEN?                                             │
      │     sampled token ID == 50256 (<endoftext>)                │
      │     or model-specific end token                            │
      │     → STOP immediately                                     │
      │                                                            │
      │  2. MAX TOKENS REACHED?                                    │
      │     generated_count >= max_tokens (API parameter)          │
      │     → STOP, return what we have                            │
      │                                                            │
      │  3. STOP SEQUENCE MATCHED?                                 │
      │     output buffer ends with a caller-defined string        │
      │     e.g. stop=["\n\n", "END", "###"]                       │
      │     → STOP, trim the stop sequence from output             │
      │                                                            │
      │  4. CONTEXT WINDOW FULL?                                   │
      │     prompt tokens + generated tokens >= max_context        │
      │     → STOP, cannot generate further                        │
      │                                                            │
      │  If NONE triggered → loop back, generate next token        │
      └────────────────────────────────────────────────────────────┘                 
                             
                             
                             
    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE FULL DECODE LOOP  (all steps together)                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Initial context: "The cat sat"
      ════════════════════════════════════════════════════════
    
      ITERATION 1
      ───────────
      Input tokens:  ["The", "cat", "sat"]
                            ↓
                   forward pass (all 32 layers)
                            ↓
      logits (1 × 100,256)  →  temperature scale
                            ↓
                       softmax
                            ↓
      repetition penalty applied to seen tokens
                            ↓
      top-p nucleus: keep tokens until cumulative p > 0.90
                            ↓
      re-normalise → sample → token ID 319 → " on"
                            ↓
      stopping check: not EOS, not max, no stop seq → CONTINUE
                            ↓
      stream " on" to client
      append to KV cache
    
    
      ITERATION 2
      ───────────
      Input:  only NEW token " on"  (KV cache has "The cat sat")
                            ↓
                   forward pass (all 32 layers)
                     KV cache loaded for prior tokens
                     only " on" fully computed
                            ↓
      logits → temp → softmax → penalty → top-p → sample
                            ↓
      token ID 262 → " the"
                            ↓
      stream " the" → continue
    
    
      ITERATION 3
      ───────────
      Input:  only NEW token " the"
      (KV cache now has "The cat sat on")
                            ↓
      token ID 3442 → " mat"
                            ↓
      stream " mat" → continue
    
    
      ITERATION 4
      ───────────
      Input:  only NEW token " mat"
                            ↓
      token ID 50256 → <endoftext>   ← EOS token sampled!
                            ↓
      stopping check: EOS detected → STOP
                            ↓
      final output: "The cat sat on the mat"
      usage stats returned: { prompt_tokens: 3,
                              completion_tokens: 4,
                              total_tokens: 7 }        
                             
                          
                          

    ╔══════════════════════════════════════════════════════════════════╗
    ║  COMPLETE PICTURE — LOGITS TO OUTPUT                            ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Raw logits  (1 × 100,256)
           │
           ▼
      ┌─────────────────────┐
      │  Repetition Penalty  │  divide seen token logits by penalty
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  Temperature Scale  │  divide all logits by T
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │      Softmax        │  convert to probabilities (sum=1)
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │   Sampling Strategy │
      │                     │
      │   Greedy   → argmax │  ← deterministic
      │   Top-K    → keep K │  ← fixed count cutoff
      │   Top-P    → cumul. │  ← adaptive cutoff
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  Weighted Sample    │  spin the probability wheel
      └──────────┬──────────┘
                 │
                 ▼
      Token ID  (single integer)
                 │
                 ▼
      ┌─────────────────────┐
      │   Stopping Check    │  EOS? max_tokens? stop_seq? ctx full?
      └──────────┬──────────┘
                 │
          ┌──────┴──────┐
          │             │
        STOP          CONTINUE
          │             │
       return         append to KV cache
       output         detokenise to text
                      stream to client
                      loop back for next token   
                      
                      
The key insight through all of this: logits are produced in one big parallel matrix multiply, 
but the conversion to a token is a series of small sequential filtering decisions — each one shaping the probability 
landscape before the final weighted coin flip that picks the actual word.



## 6.1  Temperature Scaling

Temperature T is applied by dividing all logits by T before the softmax. Low temperatures (T < 1) sharpen the 
distribution, making the most probable token more dominant and producing more deterministic, conservative outputs. 
High temperatures (T > 1) flatten the distribution, making lower-probability tokens more likely and producing more 
creative, diverse, but also potentially incoherent outputs. At T → 0, sampling collapses to greedy decoding.

Temperature = 0 (greedy decoding): Always picks the highest-probability token. 
Fast and deterministic, but can produce repetitive, low-diversity text. Temperature = 0.7: 
A typical creative writing setting. The top few tokens compete meaningfully. Temperature = 1.0:
The raw distribution — no sharpening or flattening. Temperature = 1.5+: 
Experimental outputs; high chance of grammatical errors or incoherence for most models.


## 6.2  Top-k Sampling

Top-k sampling restricts the sampling pool to only the k highest-probability tokens after temperature scaling. 
All other tokens are set to zero probability. Setting k=1 is identical to greedy decoding. 
Common values are k=40 to k=100. 
Top-k sampling prevents the model from ever sampling from the long tail of very low-probability tokens (which are often 
grammatically or semantically inappropriate), while preserving diversity among the plausible completions.

The drawback of top-k is that k is a fixed count regardless of the shape of the distribution. 
If the model is very confident (narrow distribution), k=50 may still include many effectively zero-probability tokens. 
If the model is very uncertain (flat distribution), k=50 may exclude many viable alternatives.


## 6.3  Top-p (Nucleus) Sampling
Top-p sampling addresses the distribution-shape problem. Instead of taking the top k tokens by count, 
it takes the smallest set of tokens whose cumulative probability mass exceeds p. 
For example, with p=0.9, if the top 3 tokens account for 90% of probability mass, only those 3 tokens are sampled. 
If the distribution is flat, perhaps 50 tokens are needed to accumulate 90%.

Nucleus sampling is generally considered superior to top-k for open-ended generation because it adapts the sampling 
pool to the model's uncertainty at each step. Most production APIs expose top-p as the primary sampling parameter. 
Typical values range from 0.85 to 0.97.

## 6.4  Repetition Penalty
A common practical addition is a repetition penalty that reduces the logits of tokens that have already appeared 
in the current generation (or prompt). This discourages the model from entering repetitive loops, which can occur 
especially at high temperatures or when generating long sequences. The penalty is applied multiplicatively: 
if a token ID has appeared, its logit is divided by the penalty factor (values > 1 penalise repetition). 
Frequency penalty variants scale the discount by how many times the token has appeared.

## 6.5  Beam Search
Greedy and sampling-based methods produce a single token at each step. 
Beam search maintains B parallel candidate sequences (beams) and at each step expands each beam by all possible 
next tokens, keeping only the B sequences with the highest cumulative log-probability. 
Beam search can find higher-probability sequences than greedy decoding but is B times more expensive.

For open-ended generative tasks, beam search often produces bland, repetitive text 
(because the highest-probability sequence tends to be the safest, most generic continuation). 
It remains valuable for structured generation tasks like translation, summarisation with length constraints, 
and code completion where the "most likely" sequence is genuinely the target.


## 6.6  Structured / Constrained Decoding
For applications that require JSON output, SQL queries, or adherence to any formal grammar, constrained decoding 
operates on the logit vector to ensure the output always conforms to the grammar at each step. 
The runtime maintains a parse state and masks out any token that would create an invalid parse state, setting those 
logits to -infinity before sampling.

Libraries like Outlines, LMQL, and Guidance implement this efficiently using finite-state automata that track which 
tokens are valid given the partial output so far. 
The masking operation runs in microseconds and adds negligible overhead, making constrained decoding a practical 
production technique.


##### Batching, Padding, and Throughput Optimisation



    ╔══════════════════════════════════════════════════════════════════╗
    ║  FIRST: WHY BATCHING EXISTS                                     ║
    ║  The GPU utilisation problem                                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      A modern GPU has thousands of parallel compute cores.
      Processing ONE request at a time wastes almost all of them.
    
      WITHOUT batching  (one request at a time):
    
      GPU CORES
      ┌────────────────────────────────────────────────────────────┐
      │ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
      │ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
      │ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
      │ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
      └────────────────────────────────────────────────────────────┘
        ↑↑↑↑
        only ~5% of cores doing useful work
        95% sitting idle — pure waste
    
      WITH batching  (multiple requests together):
    
      GPU CORES
      ┌────────────────────────────────────────────────────────────┐
      │ ████████████████████████████████████████████████████████ │
      │ ████████████████████████████████████████████████████████ │
      │ ████████████████████████████████████████████████████████ │
      │ ████████████████████████████████████████████████████████ │
      └────────────────────────────────────────────────────────────┘
        ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        ~95% of cores doing useful work
        same hardware, much more throughput



    ╔══════════════════════════════════════════════════════════════════╗
    ║  STATIC BATCHING — THE NAIVE APPROACH                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Server waits until B requests arrive.
      Groups them together.
      Runs ONE forward pass for all B requests.
      Returns all results together.
    
      Timeline:
    
      t=0   Request A arrives  (prompt: 5 tokens)
      t=1   Request B arrives  (prompt: 3 tokens)
      t=2   Request C arrives  (prompt: 8 tokens)
             ↓
      t=3   Batch is FULL (batch size = 3)
             ↓
      t=3   Forward pass begins for all 3 together
    
      ┌────────────────────────────────────────────────┐
      │  BATCH at t=3                                  │
      │                                                │
      │  Request A: [The] [cat] [sat] [on ] [the ]     │
      │  Request B: [Who] [are] [you]                  │
      │  Request C: [How] [do ] [you] [bake] [bread]   │
      │             [from] [scratch] [?  ]             │
      └────────────────────────────────────────────────┘



    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE PADDING PROBLEM                                            ║
    ║  Sequences must be the same length to batch together           ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      GPU matrix operations require rectangular tensors.
      All sequences in a batch MUST have identical length.
    
      Our 3 requests have different lengths:
    
      Request A:  5 tokens
      Request B:  3 tokens
      Request C:  8 tokens   ← longest in batch
    
      SOLUTION: pad shorter sequences to match the longest.
      Add special [PAD] token to fill the gap.
    
      BEFORE padding (jagged — cannot process):
    
      Request A: [The][cat][sat][on ][the]
      Request B: [Who][are][you]
      Request C: [How][do ][you][bak][bre][fro][scr][? ]
    
      AFTER padding (rectangular — can process):
    
      Request A: [The][cat][sat][on ][the][PAD][PAD][PAD]
      Request B: [Who][are][you][PAD][PAD][PAD][PAD][PAD]
      Request C: [How][do ][you][bak][bre][fro][scr][? ]
                  ↑                   ↑
                  real tokens         wasted compute
                  (carry meaning)     (PAD tokens do nothing
                                       useful but still cost
                                       GPU time to process)
    
      Padding waste in this batch:
    
      Total token slots:  8 × 3 = 24
      Real tokens:        5 + 3 + 8 = 16
      Wasted PAD slots:   24 - 16 = 8
      Waste percentage:   8 ÷ 24 = 33% wasted
    
      ┌─────────────────────────────────────────────────────────┐
      │  Worst case scenario:                                   │
      │                                                         │
      │  Request A: 1 token   [Hi ]                             │
      │  Request B: 1 token   [Hey]                             │
      │  Request C: 500 tokens [very long document...]          │
      │                                                         │
      │  Padded batch:                                          │
      │  Request A: [Hi ][PAD][PAD]...(499 PADs)...[PAD]        │
      │  Request B: [Hey][PAD][PAD]...(499 PADs)...[PAD]        │
      │  Request C: [very long document ... 500 tokens]         │
      │                                                         │
      │  Waste = (500+500-2) ÷ 1500 = 66% of compute wasted    │
      └─────────────────────────────────────────────────────────┘
    
      ATTENTION MASK saves the model from LEARNING on PAD tokens
      but does NOT save the GPU from COMPUTING them.
    
      ┌─────────────────────────────────────────────────────────┐
      │  Attention mask for Request A in the padded batch:      │
      │                                                         │
      │  Tokens:  [The][cat][sat][on ][the][PAD][PAD][PAD]      │
      │  Mask:    [ 1 ][ 1 ][ 1 ][ 1 ][ 1 ][ 0 ][ 0 ][ 0 ]    │
      │                                        ↑                │
      │                         0 = ignore this position        │
      │                         set attention logit to -∞       │
      │                         before softmax                  │
      │                                                         │
      │  PAD tokens cannot influence real tokens via attention. │
      │  But the matrix multiply still ran for those slots.     │
      └─────────────────────────────────────────────────────────┘



    ╔══════════════════════════════════════════════════════════════════╗
    ║  STATIC BATCHING TIMELINE — THE WAITING PROBLEM                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      In static batching, EVERY request in the batch waits
      for the SLOWEST request to finish generating.
    
      Batch of 3 requests, different generation lengths:
    
      Request A: needs  4 tokens of generation
      Request B: needs 12 tokens of generation
      Request C: needs  2 tokens of generation
    
      ──────────────────────────────────────────────────────────▶ time
    
      Pre-fill:  [████]  (all 3 processed together)
    
      Decode step 1:   [A][B][C]   ← all 3 generating
      Decode step 2:   [A][B][C]
      Decode step 3:   [A][B][C]
      Decode step 4:   [A][B][C]   ← A is DONE (4 tokens)
                                      but must wait for B
      Decode step 5:      [B][░]   ← A slot now IDLE
      Decode step 6:      [B][░]
      Decode step 7:      [B][░]
      Decode step 8:      [B][░]
      Decode step 9:      [B][░]
      Decode step 10:     [B][░]
      Decode step 11:     [B][░]
      Decode step 12:     [B][░]   ← B is DONE
    
      ░ = idle GPU slot (A and C finished but batch not released)
      Entire batch held until B finishes its 12 tokens.
    
      PROBLEMS:
      1. Requests A and C are done but user waits for B
      2. GPU slots wasted on idle positions
      3. New requests piling up in queue cannot enter
         until entire current batch completes
         


    ╔══════════════════════════════════════════════════════════════════╗
    ║  CONTINUOUS BATCHING — THE MODERN SOLUTION                      ║
    ║  (also called in-flight batching / iteration-level scheduling)  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Key insight: release finished slots IMMEDIATELY.
      Insert waiting requests at the NEXT decode step.
      Never wait for the slowest request.
    
      Same scenario: A needs 4 tokens, B needs 12, C needs 2.
      New requests D and E are waiting in queue.
    
      ──────────────────────────────────────────────────────────▶ time
    
      Pre-fill:       [A][B][C]    ← batch starts
    
      Decode step 1:  [A][B][C]
      Decode step 2:  [A][B][C]
      Decode step 3:  [A][B][C]
                        ↑
                        C is DONE after step 3
                        ↓
      Decode step 4:  [A][B][D]   ← D inserted immediately
                       ↑
                       A is DONE after step 4
                       ↓
      Decode step 5:  [E][B][D]   ← E inserted immediately
      Decode step 6:     [B][D]   ← D finishes
      Decode step 7:     [B][F]   ← F from queue
      Decode step 8:     [B][F]
      ...
      Decode step 12:    [B][..]  ← B finally finishes
    
      GPU utilisation:
    
      Static batching:
      Steps 1-4:   [███]  3 slots active
      Steps 5-12:  [█░░]  1-2 slots active  ← wasted
      Average utilisation: ~55%
    
      Continuous batching:
      Steps 1-12:  [███]  3 slots active throughout
      Average utilisation: ~95%
    
      ┌─────────────────────────────────────────────────────────┐
      │  The batch is never empty. The moment a slot frees up   │
      │  the next waiting request jumps in.                     │
      │  GPU never idles waiting for a slow request to finish.  │
      └─────────────────────────────────────────────────────────┘



    ╔══════════════════════════════════════════════════════════════════╗
    ║  PAGEDATTENTION — SOLVING THE MEMORY WASTE PROBLEM              ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Even with continuous batching, KV cache memory is wasted.
      The problem: you do not know how long a sequence will be
      before it finishes generating. So you pre-allocate the
      maximum context length for every request.
    
      NAIVE KV CACHE ALLOCATION:
    
      GPU VRAM  (say 40GB available for KV cache)
      ┌─────────────────────────────────────────────────────────┐
      │  Request A  (max 4096 tokens pre-allocated)             │
      │  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │
      │   used        wasted (might never be used)              │
      │                                                         │
      │  Request B  (max 4096 tokens pre-allocated)             │
      │  [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │
      │   used              wasted                              │
      │                                                         │
      │  Request C  (max 4096 tokens pre-allocated)             │
      │  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │
      │   used  wasted (mostly empty)                           │
      │                                                         │
      │  NO ROOM for more requests (memory exhausted)           │
      └─────────────────────────────────────────────────────────┘
    
    
      PAGEDATTENTION — virtual memory for KV cache:
    
      Inspired directly by OS virtual memory paging.
      KV cache divided into fixed-size BLOCKS (pages).
      Each block holds KV for a fixed number of tokens (e.g. 16).
      Blocks allocated ON DEMAND as sequence grows.
      Blocks returned to pool when sequence finishes.
    
      Block Pool in GPU VRAM:
      ┌─────────────────────────────────────────────────────────┐
      │  [BLK01][BLK02][BLK03][BLK04][BLK05][BLK06][BLK07]...  │
      │  each block = KV for 16 tokens, all layers              │
      └─────────────────────────────────────────────────────────┘
    
      Request A generates 34 tokens → needs 3 blocks (ceil 34/16):
      Block table for A:
      ┌──────────────────────────────────────────────┐
      │  Logical pos 0-15   → Physical Block BLK03   │
      │  Logical pos 16-31  → Physical Block BLK07   │
      │  Logical pos 32-33  → Physical Block BLK11   │
      └──────────────────────────────────────────────┘
    
      Request B generates 18 tokens → needs 2 blocks:
      Block table for B:
      ┌──────────────────────────────────────────────┐
      │  Logical pos 0-15   → Physical Block BLK01   │
      │  Logical pos 16-17  → Physical Block BLK09   │
      └──────────────────────────────────────────────┘
    
      Blocks are NOT contiguous in memory.
      The block table maps logical to physical on the fly.
      This is identical to how your OS manages RAM pages.
    
      BENEFIT:
    
      Naive allocation:
      ┌────────────────────────────────────────────────────┐
      │  3 requests × 4096 max tokens = 12,288 token slots │
      │  actual usage: 34 + 18 + 20 = 72 tokens            │
      │  waste: (12288 - 72) ÷ 12288 = 99.4%  ← terrible  │
      └────────────────────────────────────────────────────┘
    
      PagedAttention:
      ┌────────────────────────────────────────────────────┐
      │  actual blocks used: 3 + 2 + 2 = 7 blocks          │
      │  = 7 × 16 = 112 token slots allocated              │
      │  actual usage: 72 tokens                            │
      │  waste: (112 - 72) ÷ 112 = 36%  ← near-optimal    │
      │  (only partial last block per sequence wastes)     │
      └────────────────────────────────────────────────────┘
    
      RESULT: same GPU can serve 10-20× more concurrent requests
      
      
      
    ╔══════════════════════════════════════════════════════════════════╗
    ║  FLASH ATTENTION — SOLVING THE MEMORY BANDWIDTH BOTTLENECK      ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Even with all above optimisations, attention itself is slow.
      The problem is not compute — it is memory movement.
    
      STANDARD ATTENTION — memory movement:
    
      GPU chip layout:
      ┌─────────────────────────────────────────────────────────┐
      │                      GPU CHIP                           │
      │                                                         │
      │  ┌──────────────┐           ┌──────────────────────┐   │
      │  │  SRAM        │           │  HBM (VRAM)           │   │
      │  │  (on-chip)   │           │  (high bandwidth mem) │   │
      │  │              │           │                       │   │
      │  │  Very fast   │           │  Slower but huge      │   │
      │  │  ~19 TB/s    │◄─────────▶│  ~3 TB/s              │   │
      │  │  20 MB size  │           │  40-80 GB size        │   │
      │  └──────────────┘           └──────────────────────┘   │
      └─────────────────────────────────────────────────────────┘
    
      STANDARD attention for N=1024 tokens:
    
      Step 1: load Q, K from HBM to SRAM          ← read  HBM
      Step 2: compute S = QKᵀ  (1024×1024 matrix) ← compute
      Step 3: write S to HBM                      ← WRITE HBM ← slow
      Step 4: read S back from HBM                ← READ  HBM ← slow
      Step 5: compute softmax(S)                  ← compute
      Step 6: write softmax(S) to HBM             ← WRITE HBM ← slow
      Step 7: read softmax(S) back                ← READ  HBM ← slow
      Step 8: compute softmax(S) × V              ← compute
      Step 9: write output to HBM                 ← write HBM
    
      The N×N matrix S is written and read TWICE from slow HBM.
      For N=8192 tokens: S = 8192×8192 = 67M floats = 128MB
      Moving 128MB back and forth is the bottleneck, not the math.
    
    
      FLASH ATTENTION — tiled computation, never leaves SRAM:
    
      Instead of computing the full N×N matrix at once,
      break Q, K, V into small tiles that FIT in SRAM.
      Process one tile at a time, accumulate result.
      The large S matrix NEVER EXISTS in HBM.
    
      Tile size chosen to fit entirely in SRAM:
    
      ┌───────────────────────────────────────────────────────┐
      │  SRAM  (20MB)                                         │
      │                                                       │
      │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
      │  │  Q tile  │  │  K tile  │  │  V tile  │            │
      │  │ (small)  │  │ (small)  │  │ (small)  │            │
      │  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
      │       └─────────────┴─────────────┘                  │
      │                      ↓                               │
      │              compute partial S                       │
      │              apply partial softmax                   │
      │              accumulate into output                  │
      │              all stays in SRAM                       │
      └───────────────────────────────────────────────────────┘
    
      HBM reads/writes:
      Standard attention:  O(N²)  ← quadratic in sequence length
      Flash attention:     O(N)   ← linear in sequence length
    
      Speedup:  2-5× faster attention
      Memory:   O(N) instead of O(N²)  ← enables longer contexts
      


    ╔══════════════════════════════════════════════════════════════════╗
    ║  SPECULATIVE DECODING — BREAKING THE SEQUENTIAL BOTTLENECK      ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      The core problem: auto-regressive decoding is SEQUENTIAL.
      You cannot generate token 5 until token 4 is done.
      Each step takes the same time regardless of how easy the token is.
    
      NORMAL DECODING:
    
      ──────────────────────────────────────────────────────▶ time
      [BIG MODEL]  t1   t2   t3   t4   t5   t6   t7   t8
                   "The" "cat" "sat" "on " "the" "mat" "."  EOS
    
      Each tick = one full forward pass of the big model.
      8 tokens = 8 sequential forward passes.
      Cannot be parallelised.
    
    
      SPECULATIVE DECODING — draft then verify:
    
      Uses TWO models:
      ┌──────────────────────────────────────────────────┐
      │  DRAFT model  (small, fast — e.g. 1B params)     │
      │  TARGET model (large, slow — e.g. 70B params)    │
      └──────────────────────────────────────────────────┘
    
      Step 1: DRAFT model runs K steps ahead (K=4 here)
    
      ──────────────────────────────────────────────────────▶ time
      [DRAFT]  t1    t2    t3    t4
               "The" "cat" "sat" "on "   ← K=4 draft tokens
               fast! cheap! (small model)
    
      Step 2: TARGET model verifies ALL K drafts in ONE pass
              (verification is parallelisable — all drafts known)
    
      [TARGET]  [verify t1, t2, t3, t4 simultaneously]
                 ↓       ↓       ↓       ↓
                "The"✓  "cat"✓  "sat"✓  "on "✓   ← all accepted!
    
      4 tokens accepted in time of ~1 target model pass.
    
    
      What if draft makes a mistake?
    
      DRAFT:   "The" "cat" "sat" "slowly"  ← draft token 4 wrong
      TARGET:  "The"✓ "cat"✓ "sat"✓ reject "slowly"
                                            resample from target = "on "
    
      Accept t1, t2, t3.  Reject t4.
      Sample correct t4 from target.
      3 tokens accepted + 1 corrected = still faster than 1-by-1.
    
      Crucially: the OUTPUT DISTRIBUTION is IDENTICAL to running
      the target model alone. No quality loss. Pure speed gain.
    
      Speedup depends on draft acceptance rate:
    
      ┌───────────────┬──────────────────┬──────────────────────┐
      │ Acceptance    │  Effective       │  Throughput          │
      │ Rate          │  tokens/pass     │  vs baseline         │
      ├───────────────┼──────────────────┼──────────────────────┤
      │  50%          │  ~2.0            │  ~1.5×               │
      │  70%          │  ~2.8            │  ~2.0×               │
      │  85%          │  ~3.5            │  ~2.5×               │
      │  95%          │  ~4.2            │  ~3.0×               │
      └───────────────┴──────────────────┴──────────────────────┘
      (K=4 draft tokens, overhead from draft model factored in)
      


    ╔══════════════════════════════════════════════════════════════════╗
    ║  THROUGHPUT METRICS — HOW PERFORMANCE IS MEASURED               ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      Three metrics matter in production:
    
      ┌──────────────────────────────────────────────────────────┐
      │  1. TTFT — Time To First Token                           │
      │                                                          │
      │  Wall clock from API call to FIRST token streamed back.  │
      │  Dominated by:                                           │
      │  → pre-fill compute time  (scales with prompt length)    │
      │  → queue wait time        (server under load)            │
      │                                                          │
      │  API call ──────────────────────► first token            │
      │  t=0                              t=TTFT                 │
      │           [queue][pre-fill][decode step 1]               │
      │                                                          │
      │  Typical values:                                         │
      │  short prompt  (100 tokens):   50-200ms                  │
      │  long prompt   (10K tokens):   1-5 seconds               │
      │  very long     (100K tokens):  10-60 seconds             │
      └──────────────────────────────────────────────────────────┘
    
      ┌──────────────────────────────────────────────────────────┐
      │  2. TBT — Time Between Tokens  (inter-token latency)     │
      │                                                          │
      │  Time between consecutive tokens in the stream.          │
      │  Dominated by:                                           │
      │  → one decode forward pass of the model                  │
      │  → roughly constant throughout generation                │
      │  → independent of prompt length (KV cache handles it)   │
      │                                                          │
      │  token1 ──► token2 ──► token3 ──► token4                 │
      │         TBT       TBT       TBT                          │
      │                                                          │
      │  Typical values:  20-80ms per token                      │
      │  Perceived as: streaming speed the user sees             │
      └──────────────────────────────────────────────────────────┘
    
      ┌──────────────────────────────────────────────────────────┐
      │  3. THROUGHPUT — tokens per second across all users      │
      │                                                          │
      │  Total tokens generated across ALL concurrent requests   │
      │  per second. The business metric for serving cost.       │
      │                                                          │
      │  throughput = (batch_size × tokens_per_step) ÷ step_time │
      │                                                          │
      │  Batch size 1:   ~50 tokens/sec   (GPU under-utilised)   │
      │  Batch size 8:   ~350 tokens/sec                         │
      │  Batch size 32:  ~900 tokens/sec  (near GPU saturation)  │
      │  Batch size 64:  ~950 tokens/sec  (diminishing returns)  │
      └──────────────────────────────────────────────────────────┘


    ╔══════════════════════════════════════════════════════════════════╗
    ║  ALL OPTIMISATIONS TOGETHER — THE FULL SERVING STACK            ║
    ╚══════════════════════════════════════════════════════════════════╝
    
      USER REQUESTS
      ┌──────────────────────────────────────────────────────────┐
      │  Req A: "Explain quantum physics"                        │
      │  Req B: "Write a haiku"                                  │
      │  Req C: "Debug my Python code..."                        │
      │  Req D: "Translate this paragraph..."                    │
      └────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
      ┌──────────────────────────────────────────────────────────┐
      │  REQUEST SCHEDULER                                       │
      │  Groups requests into batches.                           │
      │  Manages continuous batching — inserts new requests      │
      │  the moment a slot frees up.                             │
      └────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
      ┌──────────────────────────────────────────────────────────┐
      │  PREFIX CACHE CHECK                                      │
      │  Hash system prompt prefix.                              │
      │  Hit? → skip pre-fill for cached tokens.                 │
      │  Miss? → full pre-fill, save to cache.                   │
      └────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
      ┌──────────────────────────────────────────────────────────┐
      │  PAGEDATTENTION MEMORY MANAGER                           │
      │  Allocates KV cache blocks on demand.                    │
      │  No pre-allocation waste.                                │
      │  Reclaims blocks the moment a request finishes.          │
      └────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
      ┌──────────────────────────────────────────────────────────┐
      │  FLASH ATTENTION COMPUTE KERNEL                          │
      │  Tiled attention — never materialises N×N matrix.        │
      │  O(N) HBM reads. 2-5× faster attention.                  │
      └────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
      ┌──────────────────────────────────────────────────────────┐
      │  SPECULATIVE DECODER  (if small draft model available)   │
      │  Draft model predicts K tokens ahead.                    │
      │  Target model verifies in one pass.                      │
      │  2-3× effective decode throughput.                       │
      └────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
      STREAMED TOKENS → detokenise → SSE stream → user
    
    
      COMBINED EFFECT ON THROUGHPUT:
    
      Baseline (no optimisations):          100 tokens/sec
      + Continuous batching:                400 tokens/sec   (4×)
      + PagedAttention:                     700 tokens/sec   (7×)
      + Flash Attention:                   1400 tokens/sec  (14×)
      + Speculative Decoding:              3500 tokens/sec  (35×)
      ─────────────────────────────────────────────────────────
      Same hardware. Same model. 35× more throughput.

The single insight tying everything together: every optimisation here solves the same root problem — the GPU is either 
idle, or processing wasted tokens, or moving data unnecessarily. 
Continuous batching keeps the GPU full, PagedAttention eliminates memory waste, 
Flash Attention eliminates redundant data movement, and speculative decoding breaks the sequential bottleneck. 
Each one attacks a different dimension of the same underlying waste.



##### Real-world LLM serving must handle many simultaneous requests. 
Efficient batching is what turns a capable model into a high-throughput service.


## **7.1  Static Batching**

In static batching, the server waits until B requests have arrived, pads all sequences to the same length 
(adding [PAD] tokens), runs a single batched forward pass, and returns results. 
This approach is simple to implement and achieves high GPU utilisation for homogeneous request lengths, 
but causes two problems: short sequences waste compute on padding tokens, and the entire batch must wait for the 
slowest (longest) sequence to finish generating.


## **7.2  Continuous Batching (Iteration-Level Scheduling)**

Continuous batching (also called in-flight batching) solves the waiting problem by inserting new requests into the 
batch at the token-generation step boundary rather than waiting for the entire batch to complete. 
When any request in the current batch finishes (emits EOS), its slot in the batch is immediately replaced by a new 
waiting request. This keeps GPU utilisation near 100% and dramatically reduces the average waiting time for users.

NVIDIA's TensorRT-LLM and vLLM both implement continuous batching. It is now the standard for production LLM serving.

## **7.3  PagedAttention and Virtual KV Cache**

The KV cache for different sequences has different lengths, and lengths grow unpredictably during generation. 
If you pre-allocate the maximum context window for every request, you waste most of the memory for short requests. 
vLLM introduced PagedAttention, which manages KV cache memory using a virtual memory paging scheme analogous to 
OS virtual memory.

The KV cache is divided into fixed-size blocks (pages). Each sequence maintains a logical block table mapping 
logical position ranges to physical memory pages. Pages are allocated on demand as sequences grow. 
When a sequence finishes, its pages are returned to the pool for reuse. This reduces memory waste to near zero and 
allows many more requests to share the available GPU memory simultaneously.

## **7.4  Flash Attention**

Standard attention requires materialising the full N×N attention matrix in GPU high-bandwidth memory (HBM). 
For long sequences, this dominates memory usage and memory bandwidth — not compute. Flash Attention (Dao et al., 2022) 
tiles the computation to keep intermediate results in the much faster on-chip SRAM, never fully materialising 
the N×N matrix in HBM.

Flash Attention 2 and 3 extended this to better handle causal masking, GQA, and multi-head variants. 
The result is 2–5× speedup in the attention computation alone, primarily by reducing memory reads and writes 
rather than reducing FLOPs. All major inference frameworks now use Flash Attention by default.

## **7.5  Speculative Decoding**

Auto-regressive decoding is inherently sequential — one token at a time. Speculative decoding breaks 
this sequential bottleneck by using a small, cheap draft model to speculatively generate K tokens ahead. 
The large target model then verifies all K tokens in a single forward pass (because verification is parallelisable 
unlike generation). 
If the draft is correct, K tokens are accepted in one step; if any token is rejected, the target model resamples 
from that point.

With good draft models (same family, much smaller), acceptance rates of 70–80% are achievable, 
resulting in 2–3× effective throughput with no quality degradation, since the distribution used for accepted tokens is 
identical to the original model's distribution.


##### End-to-End Flow: From API Call to Generated Text

Bringing all the preceding sections together, here is the complete step-by-step journey of a single 
API request through a production LLM serving stack.


    +----+-------------------+----------------------------------------------------------------------------------------+
    | #  | Stage             | What Happens                                                                           |
    +----+-------------------+----------------------------------------------------------------------------------------+
    | 1  | API Ingestion     | Client sends HTTP POST with JSON: model, messages array, sampling params, tools.       |
    | 2  | Input Validation  | Server validates schema, checks content policy, authenticates key, routes to cluster.  |
    | 3  | Prompt Assembly   | System prompt, history, and user message serialized into ChatML. Tool schemas appended.|
    | 4  | Tokenisation      | Text is normalised, pre-tokenised, BPE/Unigram-encoded. Output: Integer ID tensor.     |
    | 5  | KV Cache Lookup   | System prompt hash checked against prefix cache. If hit, pre-fill skips cached prefix. |
    | 6  | Pre-Fill          | Full prompt tensor processed in one batched forward pass. K/V matrices computed.       |
    | 7  | Sampling Config   | Temp, top-p, top-k, repetition penalty applied to logits of the last prompt position.  |
    | 8  | Decode Loop       | New token ID: embedded -> appended to KV cache -> logits computed -> sampled -> stream.|
    | 9  | Stopping          | Loop terminates on: EOS token, max_tokens limit, stream close, or stop sequence.       |
    | 10 | Detokenisation    | Token ID stream decoded to UTF-8. Multi-byte sequences buffered. Returned via SSE/JSON.|
    | 11 | Post-Processing   | Usage metadata computed. Logging, billing, and rate-limit counters updated.            |
    +----+-------------------+----------------------------------------------------------------------------------------+

---

        [ USER / CLIENT ]
               |
               |  1. API Ingestion (HTTP POST / JSON)
               v
    +-----------------------+
    |    INPUT VALIDATION   |  2. Schema check, Policy check,
    |      & ROUTING        |     Auth & Cluster routing
    +----------+------------+
               |
               |  3. Prompt Assembly (ChatML serialization)
               v
    +-----------------------+
    |      TOKENISATION     |  4. Normalization -> BPE/Unigram
    | (Text to ID Tensor)   |     Output: [1, N] Integer Tensor
    +----------+------------+
               |
               |  5. KV Cache Lookup (Prefix Hash Check)
               v
    +-----------------------+
    |        PRE-FILL       |  6. Parallel Batch Forward Pass
    |   (Prompt Processing) |     Compute K/V matrices for N tokens
    +----------+------------+
               |
               |  7. Sampling Config (Set Temp, Top-P, Logits)
               v
    +-----------------------+ <---------------------------+
    |      DECODE LOOP      |                             |
    | (Autoregressive Gen)  |  8. Logits -> Sample ->     |
    +----------+------------+     Append to KV Cache -----+
               |
               |  9. Stopping (EOS, max_tokens, or Stop Seq)
               v
    +-----------------------+
    |    DETOKENISATION     |  10. IDs to UTF-8
    |    & STREAMING        |      Buffered SSE / Final JSON
    +----------+------------+
               |
               |  11. Post-Processing (Usage, Billing, Logging)
               v
       [ RESPONSE SENT ]

## 8.1  Streaming and First-Token Latency

Modern LLM APIs stream output token-by-token using Server-Sent Events (SSE). 
The most user-visible latency metric is Time to First Token (TTFT) — the delay between the API call and the first 
decoded token appearing. 
TTFT is dominated by pre-fill time for long prompts and by queueing time when servers are under load.
The second key metric is Time Between Tokens (TBT) or inter-token latency.
This is roughly constant across the decode phase (assuming no batching changes mid-request) and is determined by 
the per-step compute of a single forward pass of the decode phase.



## 8.2  The Impact of Context Length on Latency

Pre-fill TTFT grows roughly quadratically with prompt length for the attention computation (O(N^2) in the naïve case, 
O(N) in Flash Attention due to better hardware utilisation, but still super-linear overall). 
A prompt of 100K tokens takes substantially longer to pre-fill than a prompt of 1K tokens — not just 100× longer 
but often several hundred times longer due to memory bandwidth constraints at long sequence lengths.

Decoding latency is largely independent of prompt length (since the KV cache absorbs the cost of attending to 
prior tokens), but grows linearly with generation length. 
This is why long-context requests have high TTFT but comparable token-generation throughput to short-context requests.

## 8.3  Context Window Limits and Truncation Strategies

When a constructed prompt exceeds the model's maximum context window, the serving layer must truncate. 
Common strategies include: truncating the oldest conversation turns first (keeping the system prompt and recent history); 
summarising mid-conversation using a secondary model call; 
or returning an error if the caller is expected to manage context length themselves.

From the model's perspective, truncation at position 0 removes tokens whose KV entries would have been at the 
beginning of the cache. The model never sees information that was removed. Importantly, the positional encodings of 
retained tokens do not shift — the model may therefore see position 5001 immediately following position 0, 
which can confuse models that learned strong positional structure. 
Some serving implementations re-index positions after truncation to avoid this.

##### 9. Advanced Topics in Prompt Processing

## 9.1  Logit Bias and Token Forcing

The API parameter logit_bias (OpenAI) or similar allows callers to add a constant offset to specific token logits 
before sampling. By setting a token's logit bias to +100, you effectively force it to always be sampled; 
by setting it to -100, you prevent it from ever being sampled. This mechanism is useful for steering generation 
toward specific outputs (e.g. requiring a response to begin with a specific word) or for implementing 
safety filters (preventing specific slurs or identifiers from being generated).


## 9.2  Prompt Caching Economics

Because pre-fill costs scale with token count, large system prompts represent a meaningful cost at scale. 
Prefix caching amortises this cost. 
The economic model is: pay full price once for the first request that establishes the cache entry, 
pay a reduced cache-hit price (often 10–25% of the full price) for all subsequent requests that share the same prefix.
For a production application with a 10,000-token system prompt serving 1,000 requests per minute, 
prompt caching can reduce total inference spend by 60–80% depending on the ratio of system prompt tokens to 
user-turn tokens.


## 9.3  Few-Shot Prompting and In-Context Learning

In-context learning (ICL) refers to the model's ability to infer and apply a task pattern from examples provided 
in the prompt, without any weight updates. A few-shot prompt appends several input-output demonstration pairs 
before the query. The model attends to these examples during pre-fill and the K/V representations of the examples 
effectively "program" the model's behaviour for the completion.

Research has shown that ICL performance is sensitive to: the ordering of examples (with recency effects favouring 
later examples), the format consistency between demonstrations and the query, and the label distribution 
(models can develop a bias toward the most common label in few-shot demonstrations). 
These are all artefacts of the attention mechanism's ability to pattern-match across the context window.


## 9.4  Chain-of-Thought as Token Budget

Chain-of-thought (CoT) prompting adds intermediate reasoning steps to the prompt (few-shot CoT) or asks the model to 
reason before answering (zero-shot CoT: "Think step by step"). 
From a token mechanics perspective, CoT works because the generated reasoning tokens become part of the context that 
the model attends to when producing the final answer — in effect giving the model more "scratch space" in the token 
sequence to process complex problems incrementally.

The token budget implications are significant: a reasoning model generating 1,000 tokens of internal monologue 
before a 100-token answer costs 10× more in generation terms than a direct-answer model. Extended thinking models 
(which generate long hidden reasoning chains) therefore have very different cost profiles from standard 
instruction-tuned models.

## 9.5  Quantisation and Its Effect on Tokenisation

Quantisation reduces the numerical precision of model weights from float16/bfloat16 to int8, int4, or lower. 
This reduces memory consumption and can increase inference throughput significantly. 
Tokenisation itself is unaffected by quantisation (it is a lookup table operation, not a matrix multiply). 
However, quantisation affects the logit distribution and therefore sampling outcomes: heavily quantised models 
may produce slightly different token distributions, affecting both quality and the statistical properties 
of streaming outputs.

## 9.6  Prompt Injection and Security

Because the system prompt and user input are concatenated into the same token sequence, a malicious user can 
attempt to inject instructions that override the system prompt by embedding delimiter tokens, role markers, 
or semantically overriding instructions in their input. Mitigations include: sanitising known delimiter strings 
from user input before tokenisation, training models to resist role-reversal attacks through RLHF or Constitutional
AI techniques, and using infrastructure-level routing to prepend system prompts that the user cannot see or modify.

From a token perspective, true prompt injection at the byte level is constrained because the special tokens used 
as delimiters are injected programmatically, not from user text. However, semantic injection — where carefully 
crafted user text influences the model to behave as if it had different instructions — operates at the embedding 
level and is much harder to prevent purely through tokenisation safeguards.

##### 10. Quick Reference: Key Numbers and Formulas


    +----------------------------+-----------------------------------------------------------------------+
    | Parameter / Formula        | Typical Values / Notes                                                |
    +----------------------------+-----------------------------------------------------------------------+
    | BPE vocabulary size        | GPT-2: 50,257 | GPT-4 / cl100k: 100,256 | LLaMA 3: 128,256            |
    +----------------------------+-----------------------------------------------------------------------+
    | Chars per token (English)  | ~3.5–4.5 characters | Code: ~2–3 | Dense JSON: ~2                     |
    +----------------------------+-----------------------------------------------------------------------+
    | Context window             | GPT-4: 128K | Claude 3.5: 200K | Gemini 1.5 Pro: 1M | LLaMA 3.1: 128K |
    +----------------------------+-----------------------------------------------------------------------+
    | KV cache memory per token  | ~0.5 MB per 1K tokens for a 7B model in fp16 with GQA                 |
    +----------------------------+-----------------------------------------------------------------------+
    | TTFT scaling               | O(N) with Flash Attention pre-fill; quadratic memory bandwidth        |
    |                            | at extremes                                                           |
    +----------------------------+-----------------------------------------------------------------------+
    | Speculative decoding speed | 2–3x typical with small draft model; acceptance rate ~70–80%          |
    +----------------------------+-----------------------------------------------------------------------+
    | Temperature range          | 0 = greedy | 0.7–1.0 = general use | 1.5+ = experimental              |
    +----------------------------+-----------------------------------------------------------------------+
    | Prefix cache savings       | 60–80% cost reduction for large fixed system prompts                  |
    +----------------------------+-----------------------------------------------------------------------+
    | BPE training merges        | Typically 30K–100K merge rules beyond base 256-byte vocab             |
    +----------------------------+-----------------------------------------------------------------------+
    | Pre-fill vs decode cost    | Pre-fill: ~3x more compute per token than decode; but parallelisable  |
    +----------------------------+-----------------------------------------------------------------------+


## Glossary of Key Terms

    +----------------------+-------------------------------------------------------------------------+
    | Term                 | Definition                                                              |
    +----------------------+-------------------------------------------------------------------------+
    | BPE                  | Byte-Pair Encoding — iterative merge-based sub-word tokenisation        |
    |                      | algorithm.                                                              |
    +----------------------+-------------------------------------------------------------------------+
    | ChatML               | Chat Markup Language — OpenAI's structured format for multi-turn        |
    |                      | conversation prompts.                                                   |
    +----------------------+-------------------------------------------------------------------------+
    | Context window       | Maximum number of tokens (prompt + generation) the model can process    |
    |                      | at once.                                                                |
    +----------------------+-------------------------------------------------------------------------+
    | GQA                  | Grouped-Query Attention — reduces KV cache size by sharing K/V          |
    |                      | projections across head groups.                                         |
    +----------------------+-------------------------------------------------------------------------+
    | Greedy decoding      | Always selecting the highest-probability next token; deterministic but  |
    |                      | prone to repetition.                                                    |
    +----------------------+-------------------------------------------------------------------------+
    | KV cache             | Memory store of pre-computed Key/Value matrices from prior tokens;      |
    |                      | enables efficient decode.                                               |
    +----------------------+-------------------------------------------------------------------------+
    | Logits               | Raw unnormalised scores output by the LM head, one per vocabulary       |
    |                      | token.                                                                  |
    +----------------------+-------------------------------------------------------------------------+
    | Nucleus sampling     | Top-p sampling — sample from the minimal token set covering probability |
    |                      | mass p.                                                                 |
    +----------------------+-------------------------------------------------------------------------+
    | PagedAttention       | Virtual memory paging for KV cache; enables high-throughput concurrent  |
    |                      | serving.                                                                |
    +----------------------+-------------------------------------------------------------------------+
    | Pre-fill             | The batched forward pass processing all prompt tokens at once before    |
    |                      | decode begins.                                                          |
    +----------------------+-------------------------------------------------------------------------+
    | RoPE                 | Rotary Position Embedding — encodes position as phase rotation of Q/K   |
    |                      | vectors.                                                                |
    +----------------------+-------------------------------------------------------------------------+
    | Speculative decoding | Draft-then-verify scheme for parallel token acceptance; increases       |
    |                      | throughput 2–3x.                                                        |
    +----------------------+-------------------------------------------------------------------------+
    | Temperature          | Logit scaling factor; low T = deterministic, high T = diverse/creative. |
    +----------------------+-------------------------------------------------------------------------+
    | Tokeniser            | Software converting text to token IDs (and back) using a vocabulary and |
    |                      | merge rules.                                                            |
    +----------------------+-------------------------------------------------------------------------+
    | TTFT                 | Time to First Token — latency from API call to first streamed output    |
    |                      | token.                                                                  |
    +----------------------+-------------------------------------------------------------------------+


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