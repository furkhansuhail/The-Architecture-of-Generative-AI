"""
SGLang — Structured Generation Language for LLM Programs
==========================================================

SGLang (Structured Generation Language) is a serving framework and
programming language for LLMs that treats multi-call LLM workflows as
first-class citizens. Where vLLM optimises single-request throughput and
TRT-LLM maximises raw GPU utilisation, SGLang's unique contribution is the
RadixAttention KV cache and an execution runtime that can co-optimise
across multiple interdependent LLM calls in the same program.

The core insight: most real-world LLM applications are NOT single prompts.
They are chains, trees, and graphs of LLM calls — agents, RAG pipelines,
chain-of-thought, multi-step reasoning, few-shot programs. SGLang gives you
a language to express these programs, and a runtime that automatically shares
KV cache across all calls in the program, schedules them efficiently, and
enforces output structure via constrained decoding.

This module covers RadixAttention, the SGLang primitives, constrained
decoding internals, and the complete deployment stack.

"""

import textwrap
import re

TOPIC_NAME   = "SGLang — Structured Generation Language & RadixAttention"
DISPLAY_NAME = "22 · SGLang"
ICON         = "🌿"
SUBTITLE     = "LLM Programs, RadixAttention, and Structured Generation"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE PROBLEM SGLANG SOLVES: MULTI-CALL LLM PROGRAMS

### Real LLM Applications are Programs, Not Single Calls

    A naive chatbot: one prompt → one response. TRT-LLM/vLLM handle this well.

    Real applications look like this:

    AGENT LOOP:
        1. LLM call: "Given the user query, decide: search or answer directly?"
        2. [if search] Tool call → retrieve documents
        3. LLM call: "Summarise these documents in 3 sentences."
        4. LLM call: "Given the summary and original query, produce a final answer."

    TREE-OF-THOUGHT:
        1. LLM call (shared prefix): "Think step by step about this problem."
        2a. LLM call (branch A): "Continue reasoning path A..."
        2b. LLM call (branch B): "Continue reasoning path B..."
        2c. LLM call (branch C): "Continue reasoning path C..."
        3. LLM call: "Which path is best? Produce final answer."

    SELF-CONSISTENCY (Wang et al., 2022):
        Same prompt → generate N independent completions → vote on best answer
        N = 10–40 parallel calls with the same long prompt prefix.

    FEW-SHOT CLASSIFICATION:
        System + examples prefix (shared) + Q1 → [score Q1]
        System + examples prefix (shared) + Q2 → [score Q2]
        ...1000 queries, all sharing the same 2000-token prefix

    The PROBLEM with naive serving for these patterns:
        Each LLM call sends the FULL prompt (including all shared context).
        The server recomputes KV for the shared prefix on EVERY call.
        For few-shot with 2000-token prefix: 99%+ of prefill is wasted.
        For tree-of-thought: every branch recomputes the root.

    SGLang solves this by making KV sharing AUTOMATIC and EXACT across calls.

### What SGLang Adds on Top of vLLM-style Serving

    vLLM prefix caching:   hash-based, best-effort, within a session window
    SGLang RadixAttention: exact prefix tree, persistent across ALL requests,
                           shared across different users/programs

    vLLM constrained output: guided decoding via outlines (separate library)
    SGLang constrained output: native first-class primitive (gen(), select(),
                                regex, JSON schema, EBNF grammar)

    vLLM program expression: write Python, call API manually
    SGLang program expression: @sgl.function decorator, fork/join primitives,
                                stream-based multi-call programs


##### PART 2 — RADIXATTENTION: THE CORE INNOVATION

### From Hash Cache to Radix Tree

    vLLM prefix caching: hash each block independently (with parent hash).
        - Exact match required for cache hit
        - LRU eviction per block
        - No explicit tree structure

    SGLang RadixAttention: builds an explicit RADIX TREE of KV blocks.
        - Tree edges = sequences of tokens
        - Each node = computed KV state for that prefix
        - New request: tree lookup finds LONGEST matching prefix automatically
        - Eviction: LRU on tree nodes (whole subtrees evicted together)

    A radix tree is a compressed prefix tree (Patricia trie):
        Common prefixes stored ONCE at the root path.
        Different suffixes branch from the deepest common node.

### Radix Tree Structure

    Example: three requests share structure

        Request 1: "You are a helpful assistant. Q: What is 2+2?"
        Request 2: "You are a helpful assistant. Q: What is the capital of France?"
        Request 3: "You are a coding assistant. Q: Write a sort function."

    Radix tree after processing all three:

                    [root]
                       │
                "You are a "
                       │
               ┌───────┴────────┐
         "helpful assistant."   "coding assistant."
               │                        │
          "Q: What is "          "Q: Write a sort function."
               │                     (Req 3 node — cached)
         ┌─────┴──────┐
       "2+2?"    "the capital of France?"
       (Req 1)        (Req 2)

    Now Request 4 arrives: "You are a helpful assistant. Q: Who is Alan Turing?"
        Tree walk: "You are a " → "helpful assistant." → "Q: " (partial match)
        Cache hit for: "You are a helpful assistant." (everything up to that node)
        Cached KV reused → skip prefill for those tokens
        Only "Q: Who is Alan Turing?" needs new prefill computation

### Why Radix Tree Beats Hash-Based Caching

    Hash cache (vLLM):
        - Block granularity: if prompt is 50 tokens and block_size=16, blocks 0,1,2,3
          might match partially — only full block matches count
        - Minimum match unit = one block (16 tokens)
        - No awareness of tree structure → can't share "almost matching" prefixes

    Radix tree (SGLang):
        - Token granularity at the DIVERGENCE POINT
        - Shares the maximum possible prefix (single token after divergence)
        - Explicit tree structure → O(prefix_length) lookup always
        - Eviction policy: LRU on subtrees (preserves shared nodes longer)

    Diagram — Shared system prompt hit rate comparison:

    Scenario: 1000 requests, 512-token shared system prompt, 50-token unique query

    Hash cache (vLLM, block_size=16):
        Prefix len 512 = 32 full blocks → all blocks match ✅
        Effective → good hit rate on this scenario

    Radix tree (SGLang):
        Prefix len 512 → all 512 tokens stored at one node → perfect hit ✅
        Additionally: diverging at token 513 means 512 tokens shared exactly

    Where Radix tree wins more decisively:
        Few-shot examples of VARYING lengths: hashes miss partial block matches
        Multi-turn conversations: tree stores entire conversation history as path
        Agent loops: each step extends the same tree path → all prior context cached

### KV Block Management in RadixAttention

    The radix tree stores LOGICAL token sequences.
    Physical KV blocks are associated with tree nodes, not individual tokens.

    Each tree node covers a SEGMENT of tokens and holds:
        - token_ids:   the tokens in this segment
        - kv_blocks:   list of physical GPU memory blocks for those tokens
        - ref_count:   number of active requests using this node
        - last_access: timestamp for LRU eviction

    When a request walks the tree:
        Matched nodes → ref_count++ (blocks pinned, won't be evicted)
        New nodes allocated for new tokens → added to tree on completion
        Request finishes → ref_count-- on all matched nodes

    Eviction: when KV pool is full, evict LRU leaf nodes first
    (leaf nodes are the least shared — safe to evict first)


##### PART 3 — CONSTRAINED DECODING IN SGLANG

### What Constrained Decoding Is

    Standard decoding: sample from ALL vocabulary tokens at each step.
    Constrained decoding: at each step, mask out tokens that would make
                          the output invalid given the constraint.

    Result: output is GUARANTEED to satisfy the constraint.
    No retry logic needed. No post-processing validation.
    No hallucinated JSON keys or invalid regex patterns.

### The Finite State Machine (FSM) Approach

    Every regular expression maps to a finite state machine (FSM):
        States = positions in the pattern
        Transitions = token characters
        Accepting state = valid complete match

    At each generation step:
        1. Know current FSM state (which part of the pattern we're in)
        2. For each vocabulary token, check: does this token keep us in a valid state?
        3. Mask out all invalid tokens
        4. Sample from the valid subset only

    The challenge: doing this efficiently.
        Naive: check every token at every step = 50,000 FSM transitions per step
        SGLang: pre-compute TRANSITION TABLES offline

    Precomputed transition table:
        For each (FSM_state, token_id) → (next_FSM_state or INVALID)
        Built once at server startup for each constraint
        At inference time: O(1) lookup per step

### JSON Schema → FSM Compilation

    JSON schema is more complex than regex — it's context-free.
    SGLang uses a specialised approach:

    1. JSON schema → grammar (EBNF-like)
    2. Grammar → pushdown automaton (PDA) — handles nested structures
    3. PDA → approximate FSM by "unrolling" recursion to bounded depth
    4. FSM → per-token transition table (pre-computed)

    For structured output like:
        {"name": string, "age": integer, "skills": [string, ...]}

    The FSM enforces:
        - Opening {
        - "name": followed by a quoted string
        - , "age": followed by digits only
        - , "skills": followed by [
        - Zero or more quoted strings separated by commas
        - Closing }

    At each step, only tokens consistent with the current FSM state are allowed.
    The model CANNOT produce {"naame": ...} or {"age": "not_a_number"}.

### Xgrammar: SGLang's Grammar Engine

    SGLang v0.3+ uses XGrammar as its constrained decoding backend:
        - Supports JSON schema, regex, EBNF grammar, Python objects
        - Pre-compiles FSM transition tables for fast per-step masking
        - Handles recursive grammars (nested JSON) correctly
        - Batch-aware: one grammar can be applied to multiple requests in parallel

    Overhead: ~0.5–2 ms per token for constraint checking (vs ~10–50 ms for generation)
    Net effect: negligible latency overhead for the guarantee of valid output.


##### PART 4 — THE SGLANG PROGRAMMING MODEL

### SGLang Functions (@sgl.function)

    SGLang provides a Python DSL for expressing LLM programs.
    Programs are Python functions decorated with @sgl.function.

    Core primitives inside an sgl.function:

        s += "text"          — append text to the prompt (no generation)
        s += sgl.gen("name") — generate text, store in s["name"]
        s += sgl.select("name", choices=["A","B","C"]) — classify to one of choices
        s.fork(N)            — create N parallel branches (tree-of-thought)
        s.join("name")       — merge branches, store all outputs

    The runtime:
        Traces the program to build a computation graph
        Identifies shared prefixes across parallel branches (fork)
        Schedules all calls to the server, sharing KV via RadixAttention
        Returns a filled-in program state (like a template)

### Example Programs

    Simple question answering:

        @sgl.function
        def qa(s, question):
            s += "Q: " + question + "\\n"
            s += "A: " + sgl.gen("answer", max_tokens=100)

    Multi-turn with structured output:

        @sgl.function
        def extract_person(s, text):
            s += "Extract person info from: " + text + "\\n"
            s += sgl.gen("info", max_tokens=200,
                         regex=r'\\{"name": ".+", "age": \\d+\\}')

    Tree-of-thought with branching:

        @sgl.function
        def tree_of_thought(s, problem):
            s += "Problem: " + problem + "\\n"
            s += "Let's think step by step.\\n"
            # Fork into 4 parallel reasoning paths — ALL SHARE the prefix above
            forks = s.fork(4)
            for i, f in enumerate(forks):
                f += f"Approach {i+1}: " + sgl.gen(f"path_{i}", max_tokens=200)
            s.join()
            # Now synthesise — ALL fork KV caches are available
            s += "The best approach is: " + sgl.gen("answer", max_tokens=50)

    Batch execution — run the same program on 100 inputs:

        results = qa.run_batch(
            [{"question": q} for q in questions],
            num_threads=32        # parallel execution
        )

### select() Primitive: Classification Without Generation

    sgl.select("label", choices=["positive", "negative", "neutral"])

    This does NOT generate free-form text.
    It computes the LIKELIHOOD of each choice given the current prefix,
    and returns the highest-probability option.

    Implementation:
        For each choice c: compute log P(c | prefix) = sum of token log-probs
        Return argmax over choices

    This is:
        - Faster than gen() (one forward pass per choice, not sequential generation)
        - More reliable (no ambiguous generation, no partial matches)
        - Equivalent to fine-tuned classification in accuracy

    Use case: sentiment analysis, intent routing, multi-choice QA, reward scoring.

### fork() and KV Sharing

    fork(N) creates N copies of the current program state.
    All N branches share the KV cache for everything BEFORE the fork.

    RadixAttention handles this naturally:
        The forked prefix is one node in the radix tree (ref_count=N)
        Each branch allocates its own NEW nodes for post-fork tokens
        No copying of shared KV blocks — O(1) fork cost

    This is the key advantage over calling the LLM N times independently:
        Independent calls: N × prefill_time for the shared prefix
        SGLang fork:       1 × prefill_time for the prefix + N × decode_time

    For tree-of-thought with 1000-token root and 4 branches:
        Naive: 4 × (1000-token prefill + 200-token decode)
        SGLang: 1 × 1000-token prefill + 4 × 200-token decode
        Savings: 3 × 1000-token prefills = ~75% of prefill compute saved


##### PART 5 — SGLANG RUNTIME INTERNALS

### Architecture

    ┌─────────────────────────────────────────────────────────────┐
    │  @sgl.function programs (Python front-end)                  │
    ├─────────────────────────────────────────────────────────────┤
    │  SGLang Interpreter (traces function, builds call graph)    │
    ├─────────────────────────────────────────────────────────────┤
    │  SGLang Runtime Client (batches & schedules calls)          │
    ├─────────────────────────────────────────────────────────────┤
    │  SGLang Server (OpenAI-compatible HTTP API)                 │
    ├─────────────────────────────────────────────────────────────┤
    │  TokenizerManager + Detokenizer (async, separate processes)  │
    ├─────────────────────────────────────────────────────────────┤
    │  Scheduler (RadixAttention, continuous batching)            │
    ├─────────────────────────────────────────────────────────────┤
    │  ModelRunner (PyTorch/CUDA, FlashInfer attention backend)   │
    ├─────────────────────────────────────────────────────────────┤
    │  RadixCache (token tree, KV block management)               │
    ├─────────────────────────────────────────────────────────────┤
    │  GPU (NVIDIA A100/H100, AMD ROCm)                           │
    └─────────────────────────────────────────────────────────────┘

### FlashInfer: SGLang's Attention Backend

    SGLang uses FlashInfer as its default attention kernel library.
    FlashInfer is optimised specifically for LLM inference scenarios
    with heterogeneous sequence lengths (mixed prefill/decode batches).

    Key FlashInfer features used by SGLang:
        Ragged attention:    efficiently handles batches with different seq lengths
        Page attention:      attention over paged KV caches (like PagedAttention)
        Cascade attention:   hierarchical attention for shared prefix (RadixAttention)
        FP8 attention:       native FP8 decode on H100

    Cascade attention — the key enabler for RadixAttention efficiency:
        Shared prefix KV → one KV block tree (computed once)
        Per-request unique KV → small per-request blocks
        Two-level attention: Q attends to shared blocks, THEN to unique blocks
        Result: shared prefix attended only ONCE per batch step (not N times)

### Tensor Parallelism in SGLang

    --tp 4  (tensor parallel degree)

    SGLang uses ZeroMQ (ZMQ) instead of Ray for worker communication:
        Lower latency: ZMQ is a direct socket, no Ray actor overhead
        Better for tight decode loops where per-step latency matters

    Communication pattern:
        Server process: scheduler, tokenizer, RadixCache
        Worker 0–3: hold model shards, receive batched inputs, run forward pass
        AllReduce via NCCL (same as vLLM) between workers

    PP (pipeline parallelism): supported from SGLang v0.3+


##### PART 6 — SGLANG vs VLLM: WHEN TO CHOOSE WHICH

### Performance Comparison

    Throughput (tokens/sec) on same hardware:
        Single-call workloads (chatbot):   SGLang ≈ vLLM (within 5–10%)
        Shared-prefix workloads (few-shot): SGLang 2–5× faster than vLLM
        Tree-of-thought (4 branches):       SGLang 3–4× faster than naive vLLM

    Latency (P50 TTFT):
        SGLang: typically 10–20% lower than vLLM on same hardware
        Reason: ZMQ vs Ray overhead, FlashInfer vs FlashAttention

    Memory efficiency:
        Both use paged KV allocation. SGLang's radix tree is strictly better
        at identifying sharing opportunities, especially for multi-call programs.

### Decision Matrix

    ┌──────────────────────────────────────────────────────────────────┐
    │  Use Case                          │ Best Choice                 │
    ├──────────────────────────────────────────────────────────────────┤
    │  Simple chatbot (single call)       │ vLLM or SGLang (≈equal)    │
    │  Agent with tool calls (multi-call) │ SGLang (RadixAttention)    │
    │  Few-shot prompts (shared prefix)   │ SGLang (prefix sharing)    │
    │  Tree-of-thought / beam             │ SGLang (fork primitive)    │
    │  Self-consistency (N completions)   │ SGLang (fork + join)       │
    │  Structured JSON output required    │ SGLang (XGrammar native)   │
    │  LoRA multi-adapter serving         │ vLLM (more mature)         │
    │  Maximum throughput, NVIDIA-only    │ TRT-LLM                    │
    │  AMD GPU / ROCm                     │ vLLM or SGLang (both OK)   │
    │  Beginner/fastest to deploy         │ vLLM (larger community)    │
    │  Research into new architectures    │ vLLM (more model support)  │
    │  Production agent frameworks        │ SGLang (purpose-built)     │
    └──────────────────────────────────────────────────────────────────┘

### OpenAI API Compatibility

    SGLang also implements the OpenAI API (/v1/chat/completions).
    Same base_url swap approach as vLLM.
    Additionally supports /v1/completions, /v1/embeddings.

    SGLang-specific API extensions:
        /v1/completions with "custom_logprob_proc" for custom logit processing
        Batch API endpoint for offline batch workloads
        Native JSON schema parameter in chat completions request body

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · RadixAttention Tree — Implement and Simulate the Radix Cache": {
        "description": (
            "Implement the RadixAttention radix tree from scratch. "
            "Build the insert, match-prefix, and LRU-eviction operations. "
            "Simulate cache hit rates for shared-prefix workloads. "
            "Compare radix tree vs vLLM-style hash cache on tree-of-thought "
            "and few-shot classification workloads."
        ),
        "language": "python",
        "code": '''
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

print("=" * 65)
print("  RADIXATTENTION — RADIX TREE KV CACHE IMPLEMENTATION")
print("=" * 65)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────
# Radix Tree Node
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class RadixNode:
    """
    One node in the RadixAttention tree.
    Represents a segment of tokens and their cached KV state.
    """
    token_ids    : List[int]     = field(default_factory=list)
    children     : Dict[int, 'RadixNode'] = field(default_factory=dict)
    ref_count    : int           = 0      # active requests pinning this node
    last_access  : int           = 0      # timestamp for LRU eviction
    kv_allocated : bool          = False  # True if KV blocks exist in GPU memory
    node_id      : int           = 0

    @property
    def num_tokens(self):
        return len(self.token_ids)

    def is_leaf(self):
        return len(self.children) == 0

    def is_pinned(self):
        return self.ref_count > 0


class RadixCache:
    """
    SGLang-style radix tree for KV cache sharing.
    Stores token sequences as paths in a compressed prefix tree.
    KV blocks associated with each node are shared across all users of that path.
    """

    def __init__(self, max_kv_tokens: int):
        self.max_kv_tokens  = max_kv_tokens
        self.root           = RadixNode(node_id=0)
        self.root.kv_allocated = True  # root is always "allocated"
        self._node_counter  = 1
        self._clock         = 0        # logical clock for LRU
        self._allocated_tokens = 0
        self.stats          = defaultdict(int)

    # ── Core operations ────────────────────────────────────────────────

    def match_prefix(self, token_ids: List[int]) -> Tuple[RadixNode, int]:
        """
        Walk the tree, return (deepest matching node, number of matched tokens).
        This is the cache LOOKUP operation.
        """
        node         = self.root
        matched_toks = 0
        i            = 0

        while i < len(token_ids):
            next_tok = token_ids[i]
            if next_tok not in node.children:
                break   # no matching child — stop here
            child = node.children[next_tok]
            # Try to match as many tokens as possible along this edge
            edge_len = 0
            for j, tok in enumerate(child.token_ids):
                if i + j >= len(token_ids) or token_ids[i + j] != tok:
                    break
                edge_len += 1
            if edge_len == 0:
                break
            # Did we match the full edge?
            if edge_len < child.num_tokens:
                # Partial match — split node (not implemented here for brevity)
                matched_toks += edge_len
                self.stats['partial_matches'] += 1
                break
            # Full edge match — move to child
            node          = child
            matched_toks += edge_len
            i            += edge_len
            child.last_access = self._clock
            self._clock += 1

        if matched_toks > 0:
            self.stats['cache_hits'] += 1
            self.stats['tokens_saved'] += matched_toks
        else:
            self.stats['cache_misses'] += 1

        return node, matched_toks

    def insert(self, token_ids: List[int]) -> RadixNode:
        """
        Insert a full token sequence into the tree.
        Creates new nodes for the unmatched suffix.
        Called when a request COMPLETES (its KV is now cached).
        """
        node, matched = self.match_prefix(token_ids)
        remaining     = token_ids[matched:]

        if not remaining:
            return node   # already fully cached

        # Create new node(s) for remaining tokens
        # Split into chunks for tree structure
        while remaining:
            first_tok = remaining[0]
            new_node  = RadixNode(
                token_ids    = remaining,
                node_id      = self._node_counter,
                kv_allocated = True,
                last_access  = self._clock,
            )
            self._node_counter  += 1
            self._clock         += 1
            self._allocated_tokens += len(remaining)
            node.children[first_tok] = new_node
            node      = new_node
            remaining = []   # all tokens in one node (simplified)
            self.stats['nodes_created'] += 1

        return node

    def evict_lru(self, tokens_needed: int) -> int:
        """
        Evict LRU leaf nodes to free tokens_needed tokens.
        Returns number of tokens actually freed.
        Pinned nodes (ref_count > 0) are skipped.
        """
        freed = 0
        while freed < tokens_needed:
            leaf = self._find_lru_leaf()
            if leaf is None:
                break
            freed                   += leaf.num_tokens
            self._allocated_tokens  -= leaf.num_tokens
            self._remove_node(leaf)
            self.stats['evictions'] += 1
        return freed

    def _find_lru_leaf(self) -> Optional[RadixNode]:
        """Find the least-recently-used leaf node that is not pinned."""
        best_node  = None
        best_time  = float('inf')

        def dfs(node):
            nonlocal best_node, best_time
            if node.is_leaf() and not node.is_pinned():
                if node.last_access < best_time:
                    best_time = node.last_access
                    best_node = node
            for child in node.children.values():
                dfs(child)

        dfs(self.root)
        return best_node

    def _remove_node(self, target: RadixNode):
        """Remove a leaf node from the tree."""
        def dfs(node):
            for key, child in list(node.children.items()):
                if child is target:
                    del node.children[key]
                    return True
                if dfs(child):
                    return True
            return False
        dfs(self.root)
        target.kv_allocated = False

    def pin(self, node: RadixNode):
        """Pin a node (request is using it — don't evict)."""
        node.ref_count += 1

    def unpin(self, node: RadixNode):
        """Unpin a node (request finished with it)."""
        node.ref_count = max(0, node.ref_count - 1)

    @property
    def utilisation(self):
        return self._allocated_tokens / self.max_kv_tokens

    def tree_stats(self):
        """Count nodes and depth."""
        nodes, depth = [0], [0]
        def dfs(node, d):
            nodes[0] += 1
            depth[0]  = max(depth[0], d)
            for child in node.children.values():
                dfs(child, d + 1)
        dfs(self.root, 0)
        return nodes[0], depth[0]


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Basic tree operations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Basic radix tree insert and prefix match")
print("━" * 65)
print()

cache = RadixCache(max_kv_tokens=10000)

# Simulate inserting requests with shared prefixes
SYSTEM_PROMPT = list(range(0, 64))          # tokens 0–63 (system prompt)
USER_QUERY_1  = list(range(100, 116))        # tokens 100–115
USER_QUERY_2  = list(range(200, 220))        # tokens 200–219
USER_QUERY_3  = list(range(300, 312))        # tokens 300–311

# Three requests: same system prompt, different queries
req1 = SYSTEM_PROMPT + USER_QUERY_1
req2 = SYSTEM_PROMPT + USER_QUERY_2
req3 = SYSTEM_PROMPT + USER_QUERY_3

# Insert completed requests
cache.insert(req1)
cache.insert(req2)
cache.insert(req3)

print(f"  Inserted 3 requests:")
print(f"    Req 1: {len(req1)} tokens (64 system + 16 query)")
print(f"    Req 2: {len(req2)} tokens (64 system + 20 query)")
print(f"    Req 3: {len(req3)} tokens (64 system + 12 query)")
print()

# Now simulate a NEW request with same system prompt
new_req = SYSTEM_PROMPT + list(range(400, 425))
node, matched = cache.match_prefix(new_req)
print(f"  New request: {len(new_req)} tokens (same system prompt + new query)")
print(f"  Cache match: {matched} tokens ({matched/len(new_req)*100:.1f}% of prompt)")
print(f"  Tokens to compute (prefill): {len(new_req) - matched}")
print(f"  Prefill savings: {matched / len(new_req) * 100:.1f}%")
print()

nodes_count, max_depth = cache.tree_stats()
print(f"  Tree stats: {nodes_count} nodes, max depth {max_depth}")
print(f"  Allocated tokens: {cache._allocated_tokens}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Workload simulation — RadixAttention vs hash cache
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Radix tree vs hash cache: hit rate comparison")
print("━" * 65)
print()

BLOCK_SIZE = 16

def simulate_hash_cache(requests, cache_size_blocks):
    """Simulate vLLM-style hash block cache."""
    cache_blocks = {}     # block_hash → True
    lru_order    = []
    hits = 0; total = 0

    def block_hash(token_ids, parent_h=0):
        return hash((parent_h, tuple(token_ids)))

    for tokens in requests:
        n_full = len(tokens) // BLOCK_SIZE
        parent = 0
        for i in range(n_full):
            blk = tuple(tokens[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
            h   = block_hash(blk, parent)
            total += 1
            if h in cache_blocks:
                hits += 1
                lru_order.remove(h); lru_order.append(h)
            else:
                if len(cache_blocks) >= cache_size_blocks:
                    evict = lru_order.pop(0)
                    del cache_blocks[evict]
                cache_blocks[h] = True
                lru_order.append(h)
            parent = h

    return hits / total if total > 0 else 0


def simulate_radix_cache(requests, max_tokens):
    """Simulate SGLang-style radix tree cache."""
    rc = RadixCache(max_kv_tokens=max_tokens)

    for tokens in requests:
        if rc._allocated_tokens + len(tokens) > max_tokens * 0.95:
            rc.evict_lru(len(tokens))
        rc.match_prefix(tokens)   # lookup (counts hits)
        rc.insert(tokens)         # insert for future reuse

    total = rc.stats['cache_hits'] + rc.stats['cache_misses']
    return rc.stats['cache_hits'] / total if total > 0 else 0


def gen_requests(n, system_len, query_len_range, sharing_ratio):
    """Generate requests with shared system prompts."""
    shared_sys = list(range(system_len))
    reqs = []
    for i in range(n):
        if np.random.random() < sharing_ratio:
            sys = shared_sys
        else:
            sys = list(range(i*1000, i*1000 + system_len))
        q_len = np.random.randint(*query_len_range)
        query = list(range(50000 + i*200, 50000 + i*200 + q_len))
        reqs.append(sys + query)
    return reqs


WORKLOADS = [
    ("Chatbot (80% shared 64-tok sys prompt)", 200, 64,  (16, 64),  0.8),
    ("Few-shot (95% shared 512-tok prefix)",   200, 512, (16, 48),  0.95),
    ("Agent (100% shared 256-tok context)",    200, 256, (32, 96),  1.0),
    ("Mixed (50% sharing)",                    200, 128, (32, 128), 0.5),
]

CACHE_TOKENS = 4096
CACHE_BLOCKS = CACHE_TOKENS // BLOCK_SIZE

print(f"  Cache capacity: {CACHE_TOKENS} tokens ({CACHE_BLOCKS} blocks)")
print(f"  Block size: {BLOCK_SIZE} tokens")
print()
print(f"  {'Workload':<42} | {'Hash cache':>12} | {'Radix tree':>12} | {'Advantage':>12}")
print(f"  {'─'*82}")

for label, n, sys_len, q_range, sharing in WORKLOADS:
    reqs = gen_requests(n, sys_len, q_range, sharing)

    h_rate = simulate_hash_cache(reqs, CACHE_BLOCKS) * 100
    r_rate = simulate_radix_cache(reqs, CACHE_TOKENS) * 100
    adv    = r_rate - h_rate
    print(f"  {label:<42} | {h_rate:11.1f}% | {r_rate:11.1f}% | {adv:+11.1f}%")

print()
print("  Radix tree wins most on few-shot and agent workloads — exactly the")
print("  patterns SGLang is designed for. Both perform similarly for pure chatbot")
print("  workloads with short uniform system prompts.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Fork savings — tree-of-thought
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Fork savings for tree-of-thought programs")
print("━" * 65)
print()

print("  Tree-of-thought: one root prompt, N parallel branches.")
print("  Naive: send full prompt N times → N × prefill_cost")
print("  SGLang fork: 1 × prefill + N × decode (shared KV via RadixAttention)")
print()

# Realistic model timing estimates (A100 80GB, LLaMA-2 13B)
PREFILL_MS_PER_TOK = 0.8    # ms per token, LLaMA-2 13B
DECODE_MS_PER_TOK  = 30     # ms per token (memory-bandwidth limited)

scenarios = [
    ("Root=256 tokens, branch=128, N=4",  256, 128, 4),
    ("Root=512 tokens, branch=64,  N=4",  512, 64,  4),
    ("Root=1024 tokens, branch=128, N=4", 1024, 128, 4),
    ("Root=256 tokens, branch=128, N=8",  256, 128, 8),
    ("Root=512 tokens, branch=64,  N=8",  512, 64,  8),
    ("Root=1024 tokens, branch=128, N=8", 1024, 128, 8),
]

print(f"  {'Scenario':<40} | {'Naive (ms)':>12} | {'SGLang (ms)':>13} | {'Speedup':>9}")
print(f"  {'─'*80}")

for label, root, branch, N in scenarios:
    t_naive   = N * (root * PREFILL_MS_PER_TOK + branch * DECODE_MS_PER_TOK)
    t_sglang  = (root * PREFILL_MS_PER_TOK +        # shared prefill once
                 N * branch * DECODE_MS_PER_TOK)     # N parallel decodes
    speedup   = t_naive / t_sglang
    print(f"  {label:<40} | {t_naive:12.0f} | {t_sglang:13.0f} | {speedup:8.2f}×")

print()
print("  Speedup increases with:")
print("  - Longer shared root (more prefill saved per fork)")
print("  - More branches N (more duplicated prefill avoided)")
print("  - Shorter branches (prefill dominates more)")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Constrained Decoding — FSM, JSON Schema, and Logit Masking": {
        "description": (
            "Implement constrained decoding via finite state machines. "
            "Build a simple regex FSM and show how it masks logits at each step. "
            "Implement JSON schema validation as a grammar. "
            "Demonstrate why constrained decoding is more reliable than prompting. "
            "Show the token-level transition table pre-computation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import re
import json
from typing import List, Dict, Set, Optional, Tuple

print("=" * 65)
print("  CONSTRAINED DECODING — FSM, JSON SCHEMA, LOGIT MASKING")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Simple FSM for constrained generation
# ─────────────────────────────────────────────────────────────────────────

class FSMState:
    """One state in a finite state machine."""
    def __init__(self, state_id: int, is_accepting: bool = False):
        self.state_id    = state_id
        self.is_accepting = is_accepting
        self.transitions : Dict[str, int] = {}   # char → next_state_id

    def __repr__(self):
        a = " [ACCEPT]" if self.is_accepting else ""
        return f"State({self.state_id}{a})"


class SimpleFSM:
    """
    Finite state machine for constrained token generation.
    Encodes what characters are valid at each position.
    """
    def __init__(self, pattern_description: str):
        self.pattern_desc = pattern_description
        self.states: List[FSMState] = []
        self.start_state = 0
        self._build()

    def _build(self):
        raise NotImplementedError("Subclass me")

    def get_valid_chars(self, state_id: int) -> Set[str]:
        """Return set of valid next characters from this state."""
        if state_id < 0 or state_id >= len(self.states):
            return set()
        return set(self.states[state_id].transitions.keys())

    def transition(self, state_id: int, char: str) -> int:
        """
        Move to the next state given current state and character.
        Returns -1 if no valid transition (invalid character).
        """
        state = self.states[state_id]
        return state.transitions.get(char, -1)

    def is_accepting(self, state_id: int) -> bool:
        return 0 <= state_id < len(self.states) and self.states[state_id].is_accepting


class IntegerFSM(SimpleFSM):
    """
    FSM that accepts an optional sign followed by one or more digits.
    Valid: "42", "-7", "0", "123"
    Invalid: "1.5", "abc", ""
    """
    def _build(self):
        # States:
        # 0 = start (accept sign or digit)
        # 1 = have at least one digit (accepting)
        # 2 = dead state (invalid)
        s0 = FSMState(0, is_accepting=False)
        s1 = FSMState(1, is_accepting=True)    # ← accepting: one or more digits seen
        s2 = FSMState(2, is_accepting=False)   # dead

        # From start: sign or digit
        s0.transitions['-'] = 1   # sign goes to digit state (needs at least one digit after)
        s0.transitions['+'] = 1
        for d in '0123456789':
            s0.transitions[d] = 1

        # From digit state: more digits allowed
        for d in '0123456789':
            s1.transitions[d] = 1
        # anything else → dead
        s0.transitions[' '] = 2
        s1.transitions['.'] = 2

        self.states = [s0, s1, s2]


class YesNoFSM(SimpleFSM):
    """
    FSM that accepts exactly 'yes' or 'no' (case insensitive).
    """
    def _build(self):
        # 0 → (y|n) → 1
        # 1 (y) → e → 2 → s → 3 [accept]
        # 1 (n) → o → 4 [accept]
        states = [FSMState(i) for i in range(6)]
        states[3].is_accepting = True   # "yes" complete
        states[4].is_accepting = True   # "no" complete
        states[5].is_accepting = False  # dead

        # Start: 'y' or 'n'
        states[0].transitions['y'] = 1
        states[0].transitions['Y'] = 1
        states[0].transitions['n'] = 2
        states[0].transitions['N'] = 2

        # 'ye' path
        states[1].transitions['e'] = 3
        states[1].transitions['E'] = 3

        # 'yes' path
        states[3].transitions['s'] = 4
        states[3].transitions['S'] = 4
        states[4].is_accepting = True

        # 'no' path
        states[2].transitions['o'] = 5
        states[2].transitions['O'] = 5
        states[5].is_accepting = True

        self.states = states


# ─────────────────────────────────────────────────────────────────────────
# Constrained generation simulator
# ─────────────────────────────────────────────────────────────────────────

SMALL_VOCAB = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-+ .,!?"\n{}[]:()')

def precompute_transition_table(fsm: SimpleFSM, vocab: List[str]) -> Dict[int, List[int]]:
    """
    Pre-compute: for each (state, token), is this token valid?
    Returns: state → list of VALID token indices.

    In production (SGLang/XGrammar), this is computed once at server startup
    for each unique constraint. Lookup is O(1) per decode step.
    """
    table = {}
    for state_id in range(len(fsm.states)):
        valid = []
        valid_chars = fsm.get_valid_chars(state_id)
        for tok_idx, tok in enumerate(vocab):
            # A token is valid if its FIRST character is a valid transition
            # (simplified — real systems handle multi-character tokens carefully)
            if tok and tok[0] in valid_chars:
                valid.append(tok_idx)
        table[state_id] = valid
    return table


def constrained_sample(logits: np.ndarray, valid_indices: List[int],
                        temperature: float = 0.8) -> int:
    """
    Sample a token from logits, restricted to valid_indices.
    Masks all invalid tokens to -inf before softmax.
    """
    masked = np.full_like(logits, -np.inf)
    if valid_indices:
        masked[valid_indices] = logits[valid_indices]
    # Softmax with temperature
    masked = masked - masked[masked != -np.inf].max()
    exp    = np.exp(masked / temperature)
    probs  = exp / exp.sum()
    return int(np.random.choice(len(probs), p=probs))


def simulate_constrained_generation(fsm: SimpleFSM, logits_sequence: np.ndarray,
                                     vocab: List[str], max_steps: int = 20):
    """
    Simulate full constrained generation.
    Shows which tokens are valid at each step and what gets generated.
    """
    table    = precompute_transition_table(fsm, vocab)
    state    = fsm.start_state
    output   = []
    steps    = []

    for step in range(max_steps):
        valid_indices = table.get(state, [])
        if not valid_indices:
            break

        token_idx = constrained_sample(logits_sequence[step % len(logits_sequence)],
                                        valid_indices)
        token     = vocab[token_idx]
        next_state = fsm.transition(state, token[0] if token else '')
        if next_state == -1:
            break

        output.append(token)
        steps.append({'step': step, 'token': token, 'valid_count': len(valid_indices),
                       'state': state, 'next_state': next_state})
        state = next_state

        if fsm.is_accepting(state):
            # Can stop here — valid complete output
            if np.random.random() < 0.6:   # simulate natural stop
                break

    return ''.join(output), steps


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: FSM demonstration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — FSM states and valid transition sets")
print("━" * 65)
print()

int_fsm = IntegerFSM("integer")
print(f"  IntegerFSM: accepts integers like '42', '-7', '123'")
print()

for state_id in range(len(int_fsm.states)):
    valid_chars = int_fsm.get_valid_chars(state_id)
    accepting   = " [ACCEPTING]" if int_fsm.is_accepting(state_id) else ""
    print(f"  State {state_id}{accepting}: valid next chars = {sorted(valid_chars)}")
print()

# Show transition table for integer FSM
table = precompute_transition_table(int_fsm, SMALL_VOCAB)
print(f"  Pre-computed transition table (state → #valid tokens in vocab):")
for state_id, valid in table.items():
    accepting = " [ACCEPT]" if int_fsm.is_accepting(state_id) else ""
    print(f"    State {state_id}{accepting}: {len(valid):3d} valid tokens out of {len(SMALL_VOCAB)}")
    if valid:
        sample_chars = [SMALL_VOCAB[i] for i in valid[:8]]
        print(f"                 Sample valid: {sample_chars}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Constrained vs unconstrained generation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Constrained vs unconstrained generation")
print("━" * 65)
print()

# Simulate a distribution that "wants" to generate something wrong
# (high probability on non-digit characters in positions where digits expected)
N_STEPS = 10
vocab_size = len(SMALL_VOCAB)
# Logits slightly favour letters over digits (simulate a confused LLM)
logits_seq = [np.random.randn(vocab_size) for _ in range(N_STEPS)]
for logits in logits_seq:
    # Boost letter probabilities
    for i, c in enumerate(SMALL_VOCAB):
        if c.isalpha():
            logits[i] += 1.5

N_TRIALS = 100
valid_unconstrained = 0
valid_constrained   = 0

for _ in range(N_TRIALS):
    # Unconstrained: sample freely
    text_free = ''
    for logits in logits_seq[:5]:
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx = np.random.choice(vocab_size, p=probs)
        text_free += SMALL_VOCAB[idx]
    try:
        int(text_free.strip())
        valid_unconstrained += 1
    except ValueError:
        pass

    # Constrained: FSM masking
    text_con, _ = simulate_constrained_generation(
        int_fsm, logits_seq, SMALL_VOCAB, max_steps=5
    )
    try:
        int(text_con.strip())
        valid_constrained += 1
    except ValueError:
        pass

print(f"  Task: generate a valid integer (using IntegerFSM)")
print(f"  Model logits: biased toward LETTERS (unfavourable for integers)")
print()
print(f"  Unconstrained: {valid_unconstrained}/{N_TRIALS} = {valid_unconstrained}% valid integers")
print(f"  Constrained:   {valid_constrained}/{N_TRIALS} = {valid_constrained}% valid integers")
print()
print("  Constrained decoding GUARANTEES valid output regardless of model bias.")
print("  No retry logic, no output validation, no prompt engineering needed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: JSON schema validation as grammar
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — JSON schema constrained generation")
print("━" * 65)
print()

SCHEMA = {
    "type": "object",
    "properties": {
        "name":  {"type": "string"},
        "age":   {"type": "integer", "minimum": 0},
        "role":  {"type": "string", "enum": ["engineer", "manager", "analyst"]},
    },
    "required": ["name", "age", "role"]
}

print(f"  JSON Schema:")
for line in json.dumps(SCHEMA, indent=4).split('\n'):
    print(f"    {line}")
print()

def validate_json_against_schema(text: str, schema: dict) -> Tuple[bool, str]:
    """Check if generated text matches the schema."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    required = schema.get("required", [])
    for field in required:
        if field not in obj:
            return False, f"Missing required field: {field}"

    props = schema.get("properties", {})
    for field, value in obj.items():
        if field not in props:
            continue
        prop_schema = props[field]
        if prop_schema.get("type") == "integer" and not isinstance(value, int):
            return False, f"Field '{field}' should be integer, got {type(value)}"
        if "enum" in prop_schema and value not in prop_schema["enum"]:
            return False, f"Field '{field}' = {value!r} not in enum {prop_schema['enum']}"

    return True, "VALID"

# Simulate outputs: some valid, some not (as an LLM without constraints might produce)
test_outputs = [
    '{"name": "Alice", "age": 30, "role": "engineer"}',           # ✅ valid
    '{"name": "Bob", "age": "thirty", "role": "manager"}',         # ❌ age is string
    '{"name": "Carol", "age": 25, "role": "developer"}',           # ❌ role not in enum
    '{"name": "Dave", "age": 40}',                                  # ❌ missing role
    '{"name": "Eve", age: 28, "role": "analyst"}',                  # ❌ invalid JSON
    '{"name": "Frank", "age": 35, "role": "analyst"}',              # ✅ valid
]

print(f"  Validating simulated model outputs:")
print()
print(f"  {'Output (truncated)':<52} | {'Valid?':>7} | {'Issue'}")
print(f"  {'─'*80}")
for out in test_outputs:
    valid, reason = validate_json_against_schema(out, SCHEMA)
    symbol = "✅" if valid else "❌"
    print(f"  {out[:50]:<52} | {symbol:>7} | {reason}")

print()
print("  Without constrained decoding: ~50% of outputs may be invalid.")
print("  With constrained decoding (XGrammar in SGLang):")
print("    → 100% of outputs are valid JSON matching the schema")
print("    → No retry, no validation layer, no error handling needed")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Overhead measurement
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Constrained decoding overhead analysis")
print("━" * 65)
print()

import time

# Simulate the cost of:
# 1. Building the transition table (done ONCE at startup)
# 2. Masking logits at each decode step (done per token)

VOCAB_SIZES  = [1000, 10000, 50000, 100000]
NUM_STATES   = [5, 10, 50, 100]

print("  Transition table BUILD time (one-time cost at startup):")
print(f"  {'Vocab size':>12} | {'States':>8} | {'Build time (ms)':>17} | {'Table entries':>15}")
print(f"  {'─'*58}")

for vocab_sz in [1000, 10000, 50257]:
    for n_states in [5, 50]:
        # Simulate building table: vocab_sz × n_states lookups
        dummy_vocab = [chr(65 + (i % 26)) for i in range(vocab_sz)]
        dummy_fsm_transitions = {
            s: {chr(65 + j % 26) for j in range(min(10, 26))}
            for s in range(n_states)
        }

        t0 = time.perf_counter()
        table = {}
        for s in range(n_states):
            valid_chars = dummy_fsm_transitions[s]
            table[s]    = [i for i, c in enumerate(dummy_vocab) if c in valid_chars]
        t_build = (time.perf_counter() - t0) * 1000

        print(f"  {vocab_sz:12,} | {n_states:8d} | {t_build:17.3f} | {vocab_sz*n_states:15,}")

print()
print("  Per-token masking time (applied at EVERY decode step):")
print(f"  {'Vocab size':>12} | {'Valid tokens':>13} | {'Mask time (μs)':>15}")
print(f"  {'─'*45}")

for vocab_sz in [1000, 10000, 50257, 100000]:
    for valid_frac in [0.01, 0.1]:
        n_valid = int(vocab_sz * valid_frac)
        valid_indices = np.random.choice(vocab_sz, size=n_valid, replace=False)
        logits = np.random.randn(vocab_sz).astype(np.float32)

        t0 = time.perf_counter()
        for _ in range(1000):
            masked = np.full(vocab_sz, -np.inf, dtype=np.float32)
            masked[valid_indices] = logits[valid_indices]
        t_mask = (time.perf_counter() - t0) / 1000 * 1e6  # microseconds

        print(f"  {vocab_sz:12,} | {n_valid:13,} | {t_mask:15.2f}")

print()
print("  Typical LLM decode step: 10–50 ms (memory bandwidth limited)")
print("  Constrained masking:     0.05–0.5 ms (1–5% overhead)")
print()
print("  The overhead is NEGLIGIBLE relative to the GPU decode time.")
print("  SGLang pre-computes tables at server startup → per-token cost is just")
print("  an array index lookup and a numpy scatter operation.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · SGLang Programming Model — sgl.function, gen, select, fork": {
        "description": (
            "Complete guide to the SGLang Python DSL. "
            "Show @sgl.function, sgl.gen(), sgl.select(), fork/join patterns. "
            "Implement tree-of-thought, self-consistency, and few-shot programs. "
            "Explain how the runtime traces and optimises multi-call programs. "
            "All examples annotated for study — no server needed."
        ),
        "language": "python",
        "code": """
print("=" * 65)
print("  SGLANG PROGRAMMING MODEL — gen, select, fork, join")
print("=" * 65)
print()
print("  This module shows the SGLang DSL patterns with annotations.")
print("  Install: pip install sglang[all]")
print("  Start server: python -m sglang.launch_server --model meta-llama/...")
print("  Code is annotated for study.")
print()

import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# SGLang primitives reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  REFERENCE — SGLang primitives")
print("━" * 65)
print()

PRIMITIVES = '''
import sglang as sgl

# ── Core primitives inside @sgl.function ──────────────────────────────

# 1. Append text (no generation — pure context building)
s += "You are a helpful assistant."
s += "\\n\\nUser: " + user_query + "\\n\\nAssistant: "

# 2. Generate text — LLM produces tokens until stop or max_tokens
s += sgl.gen(
    "output_name",          # key to retrieve result: s["output_name"]
    max_tokens=200,         # max tokens to generate
    temperature=0.7,        # sampling temperature
    top_p=0.9,              # nucleus sampling
    stop=["\\n", "User:"],   # stop sequences
    regex=r"\\d+\\.\\d+",    # optional regex constraint
)

# 3. Select — classify to one of fixed choices (no generation, one forward pass)
s += sgl.select(
    "sentiment",                        # key to retrieve result
    choices=["positive", "negative", "neutral"],  # options
    temperature=0.0                     # usually greedy for classification
)

# 4. Fork — create N parallel branches sharing the current prefix KV
forks = s.fork(N)           # returns list of N sub-states
for f in forks:
    f += "Branch content: " + sgl.gen("branch_output", max_tokens=100)
s.join()                    # sync: wait for all forks to complete

# 5. Retrieve generated text
result = s["output_name"]           # string of generated text
results = [f["branch_output"] for f in forks]  # list of branch outputs

# ── Execution ─────────────────────────────────────────────────────────

# Single execution
state = my_function.run(arg1=val1, arg2=val2)
output = state["key"]

# Batch execution (all prompts scheduled together!)
states = my_function.run_batch(
    [{"arg1": v1, "arg2": v2} for v1, v2 in zip(vals1, vals2)],
    num_threads=32,          # parallel HTTP calls to server
)
outputs = [s["key"] for s in states]

# Streaming
for text_delta in my_function.run_stream(arg1=val1):
    print(text_delta, end="", flush=True)
'''

print(PRIMITIVES)

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 1: Simple Q&A with structured output
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 1 — Simple Q&A with regex-constrained answer")
print("━" * 65)
print()

PATTERN_1 = '''
import sglang as sgl

@sgl.function
def answer_number(s, question: str):
    '''Extract a numeric answer from a question.'''
    s += "Answer the following question with ONLY a number.\\n"
    s += f"Question: {question}\\n"
    s += "Answer: "
    s += sgl.gen(
        "answer",
        max_tokens=10,
        regex=r"\\d+",      # ONLY digits allowed — guaranteed numeric output
        stop=["\\n", " "],
    )

# Usage:
state = answer_number.run(question="How many days are in a week?")
answer = state["answer"]   # guaranteed to be a digit string like "7"
print(f"Answer: {answer}")
# → "7"   (always a valid number, regardless of model behaviour)

# Batch usage:
questions = [
    "How many months in a year?",
    "How many sides does a triangle have?",
    "How many continents are there?",
]
states = answer_number.run_batch(
    [{"question": q} for q in questions],
    num_threads=len(questions)   # all run in parallel!
)
for q, s in zip(questions, states):
    print(f"  Q: {q}")
    print(f"  A: {s['answer']}")
'''
print(PATTERN_1)

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 2: Multi-step reasoning with select()
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 2 — Multi-step with select() for routing")
print("━" * 65)
print()

PATTERN_2 = '''
import sglang as sgl

@sgl.function
def agent_router(s, user_query: str):
    '''
    Step 1: Classify the query type.
    Step 2: Route to different generation strategies.
    '''
    s += f"User query: {user_query}\\n"
    s += "Query type (factual/creative/code): "

    # select() — ONE forward pass, returns highest-probability choice
    # Much faster and more reliable than generating the category as free text
    s += sgl.select(
        "query_type",
        choices=["factual", "creative", "code"],
    )

    query_type = s["query_type"]

    # Branch based on classification
    if query_type == "factual":
        s += "\\nProvide a concise, accurate answer:\\n"
        s += sgl.gen("response", max_tokens=150, temperature=0.1)

    elif query_type == "creative":
        s += "\\nBe creative and expressive:\\n"
        s += sgl.gen("response", max_tokens=300, temperature=1.1, top_p=0.95)

    elif query_type == "code":
        s += "\\nProvide working code with explanation:\\n```python\\n"
        s += sgl.gen(
            "response",
            max_tokens=500,
            temperature=0.2,
            stop=["```"],   # stop at code block end
        )
        s += "```"

# Why select() > generating the category:
# - select(): one forward pass, O(len(choices)) work, deterministic
# - gen():    sequential tokens, can produce "factuall" or "FACTUAL" or "factual answer"
# - select() is faster, more reliable, and easier to handle downstream
'''
print(PATTERN_2)

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 3: Tree-of-thought with fork/join
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 3 — Tree-of-thought (fork/join with KV sharing)")
print("━" * 65)
print()

PATTERN_3 = '''
import sglang as sgl

@sgl.function
def tree_of_thought(s, problem: str, n_branches: int = 4):
    '''
    Generate N independent reasoning paths, then synthesise.

    KV sharing via RadixAttention:
    - System prompt + problem statement KV computed ONCE
    - All N branches reuse this KV (no re-prefill)
    - Each branch adds its OWN decode tokens
    '''
    # ── Shared context (computed once for all branches) ────────────────
    s += "You are an expert problem solver.\\n\\n"
    s += f"Problem: {problem}\\n\\n"
    s += "Let's explore multiple approaches."

    # ── Fork: N parallel reasoning paths (all share KV above) ─────────
    branches = s.fork(n_branches)   # ← one prefill for shared prefix

    for i, branch in enumerate(branches):
        branch += f"\\n\\nApproach {i+1}: Let me think step by step...\\n"
        branch += sgl.gen(
            f"reasoning_{i}",
            max_tokens=200,
            temperature=0.7,
        )

    # ── join() — wait for all branches to complete ────────────────────
    s.join()   # s now has access to all branch results

    # ── Synthesise ────────────────────────────────────────────────────
    s += "\\n\\nBased on all approaches above, the best answer is:\\n"
    s += sgl.gen("final_answer", max_tokens=100, temperature=0.1)

# Usage with timing comparison:
import time

problem = "What is the most efficient algorithm for finding prime numbers?"

# SGLang (shared KV via fork)
t0 = time.time()
state = tree_of_thought.run(problem=problem, n_branches=4)
t_sglang = time.time() - t0
print(f"SGLang fork: {t_sglang:.2f}s")
print(f"Final answer: {state['final_answer'][:100]}...")

# Naive (4 independent calls, each re-prefills the full prompt)
# → ~4x slower because prompt KV is computed 4 times
t0 = time.time()
# [4 independent API calls here]
t_naive = time.time() - t0
print(f"Naive calls: {t_naive:.2f}s")
print(f"Speedup from fork: {t_naive/t_sglang:.1f}×")
'''
print(PATTERN_3)

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 4: Self-consistency (best-of-N with voting)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 4 — Self-consistency (N completions → majority vote)")
print("━" * 65)
print()

PATTERN_4 = '''
import sglang as sgl
from collections import Counter

@sgl.function
def self_consistency(s, question: str, n: int = 10):
    '''
    Generate N answers independently, then vote.
    Wang et al. 2022: majority voting improves accuracy on reasoning tasks.

    RadixAttention shares the question prefix across all N completions.
    '''
    s += f"Question: {question}\\nLet\\'s think step by step.\\n"

    # Fork into N independent completions — ALL share the question prefix KV
    completions = s.fork(n)
    for c in completions:
        c += sgl.gen("reasoning", max_tokens=150, temperature=0.7)
        c += "\\nTherefore, the answer is: "
        c += sgl.gen(
            "answer",
            max_tokens=10,
            regex=r"[A-Za-z0-9 ]+",  # constrain answer format
            stop=["\\n", "."],
        )
    s.join()

    # Majority vote on answers
    answers = [c["answer"].strip().lower() for c in completions]
    vote    = Counter(answers).most_common(1)[0][0]
    s.answer_votes = dict(Counter(answers))   # attach to state for inspection

    s += f"\\n\\nFinal answer (majority of {n} votes): {vote}"

# Why self-consistency works:
# - Correct reasoning paths: tend to converge on the same answer
# - Incorrect reasoning paths: make different errors → don't dominate the vote
# - Accuracy improvement: +5-15% on mathematical and logical reasoning benchmarks
# - Cost: N × generation_cost, but shared prefill makes it cheaper than N API calls

state = self_consistency.run(
    question="If I have 3 apples and give away 2/3 of them, how many do I have left?",
    n=8
)
print(f"Answer distribution: {state.answer_votes}")
print(f"Final answer: {state['answer_votes']}")
'''
print(PATTERN_4)

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 5: Batch few-shot classification
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 5 — Batch few-shot classification (shared examples prefix)")
print("━" * 65)
print()

PATTERN_5 = '''
import sglang as sgl

# Few-shot examples — shared across ALL classification requests
EXAMPLES = '''
Review: "The food was amazing!" → Sentiment: positive
Review: "Terrible service, waited 2 hours." → Sentiment: negative
Review: "It was okay, nothing special." → Sentiment: neutral
Review: "Best meal I've ever had!" → Sentiment: positive
Review: "Disappointing, won't return." → Sentiment: negative
'''

@sgl.function
def classify_sentiment(s, review: str):
    '''
    Few-shot sentiment classification.
    EXAMPLES prefix is shared via RadixAttention across all requests.
    '''
    s += EXAMPLES                # ← same for every request (cached after first!)
    s += f"Review: \\"{review}\\" → Sentiment: "
    s += sgl.select(
        "sentiment",
        choices=["positive", "negative", "neutral"],
    )

# Batch 1000 reviews — EXAMPLES prefix computed ONCE, reused 1000 times
reviews = [
    "Absolutely loved it!",
    "Complete waste of money.",
    "Pretty average experience.",
    "Outstanding quality, highly recommend!",
    "Left early, staff was rude.",
    # ... 995 more
]

# RadixAttention: EXAMPLES (5 × ~20 tokens = 100 tokens) prefill saved 999 times
# Savings: 999 × 100 tokens × 0.8ms/token = ~80 seconds of prefill compute
# At 1000 requests, this is a massive throughput improvement

states = classify_sentiment.run_batch(
    [{"review": r} for r in reviews],
    num_threads=64,  # 64 concurrent HTTP calls to the SGLang server
)

for review, state in zip(reviews, states):
    print(f"  {state['sentiment']:10s} | {review[:40]}")
'''
print(PATTERN_5)

print("━" * 65)
print("  SGLang PROGRAMMING PATTERNS SUMMARY")
print("━" * 65)
print()
print("  ┌────────────────────────────────────────────────────────────┐")
print("  │ Pattern             │ Primitive  │ KV Sharing     │ Gain   │")
print("  ├────────────────────────────────────────────────────────────┤")
print("  │ Simple generation   │ gen()      │ Within session │ 1×     │")
print("  │ Classification      │ select()   │ Within session │ 2–5×   │")
print("  │ Multi-step chain    │ gen()+sel()│ Previous steps │ 1.5×   │")
print("  │ Tree-of-thought     │ fork/join  │ Shared root    │ 3–4×   │")
print("  │ Self-consistency    │ fork(N)    │ Full prefix    │ N×     │")
print("  │ Few-shot batch      │ run_batch  │ Shared examples│ 10–50× │")
print("  └────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · SGLang Server — Deployment, Config & Benchmarking": {
        "description": (
            "Complete SGLang server deployment reference. "
            "All key launch flags with explanations. "
            "Quantisation options (AWQ, GPTQ, FP8). "
            "Benchmark TTFT and throughput across workload types. "
            "Checklist for production deployment."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  SGLANG SERVER — DEPLOYMENT & CONFIGURATION REFERENCE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Launch command reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Launch commands for common configurations")
print("━" * 65)
print()

CONFIGS = [
    {
        "name": "Minimal (development)",
        "cmd": [
            "python -m sglang.launch_server",
            "    --model-path meta-llama/Llama-2-7b-chat-hf",
            "    --port 30000",
        ]
    },
    {
        "name": "Single A100 80GB (production)",
        "cmd": [
            "python -m sglang.launch_server",
            "    --model-path meta-llama/Llama-2-7b-chat-hf",
            "    --dtype float16",
            "    --tp 1",
            "    --max-total-tokens 32768",     # total KV pool size
            "    --chunked-prefill-size 512",   # prefill chunk size
            "    --enable-torch-compile",        # torch.compile for ~15% speedup
            "    --attention-backend flashinfer", # FlashInfer (default, fastest)
            "    --port 30000",
        ]
    },
    {
        "name": "4x A100 80GB (LLaMA-2 70B, TP=4)",
        "cmd": [
            "python -m sglang.launch_server",
            "    --model-path meta-llama/Llama-2-70b-chat-hf",
            "    --dtype bfloat16",
            "    --tp 4",
            "    --max-total-tokens 65536",
            "    --chunked-prefill-size 2048",
            "    --enable-torch-compile",
            "    --port 30000",
        ]
    },
    {
        "name": "H100 FP8 (maximum throughput)",
        "cmd": [
            "python -m sglang.launch_server",
            "    --model-path meta-llama/Meta-Llama-3.1-70B-Instruct",
            "    --dtype float8",
            "    --quantization fp8",
            "    --tp 4",
            "    --max-total-tokens 131072",
            "    --chunked-prefill-size 4096",
            "    --enable-torch-compile",
            "    --attention-backend flashinfer",
            "    --port 30000",
        ]
    },
    {
        "name": "AWQ quantised (memory-constrained)",
        "cmd": [
            "python -m sglang.launch_server",
            "    --model-path TheBloke/Llama-2-7B-Chat-AWQ",
            "    --quantization awq",
            "    --dtype float16",
            "    --max-total-tokens 65536",
            "    --port 30000",
        ]
    },
    {
        "name": "Vision-language model (LLaVA)",
        "cmd": [
            "python -m sglang.launch_server",
            "    --model-path lmms-lab/llava-onevision-qwen2-7b-ov",
            "    --dtype bfloat16",
            "    --tp 1",
            "    --port 30000",
        ]
    },
]

for cfg in CONFIGS:
    print(f"  # {cfg['name']}")
    for line in cfg['cmd']:
        print(f"  {line} \\")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Key parameters explained
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Key parameters (SGLang vs vLLM differences)")
print("━" * 65)
print()

PARAMS = [
    ("--max-total-tokens",
     "Total KV token pool size (= max_num_blocks × block_size).\n"
     "       vLLM equivalent: max_num_seqs × max_model_len.\n"
     "       SGLang is more flexible: tokens shared across requests via radix tree."),

    ("--chunked-prefill-size",
     "Max tokens per prefill chunk (interleaved with decode).\n"
     "       Larger: better GPU utilisation.\n"
     "       Smaller: lower inter-token latency for existing requests."),

    ("--attention-backend",
     "flashinfer (default, fastest), triton, flashattn.\n"
     "       FlashInfer has cascade attention for RadixAttention efficiency.\n"
     "       Use triton as fallback on non-CUDA hardware."),

    ("--enable-torch-compile",
     "Apply torch.compile() to the model for ~10–20% speedup.\n"
     "       First request is slow (compilation). Warmup required.\n"
     "       Disable for debugging or frequent model updates."),

    ("--schedule-policy",
     "lpm (default): longest-prefix-match (maximise cache hits).\n"
     "       fcfs: first-come-first-served.\n"
       "       lpm is almost always better — prioritises requests that share cache."),

    ("--mem-fraction-static",
     "Fraction of GPU memory for model weights (default: 0.88).\n"
     "       Remainder is for KV cache (RadixAttention pool).\n"
     "       Reduce if OOM on weights; increase for larger KV pool."),

    ("--tp / --dp",
     "--tp N: tensor parallelism (split model across N GPUs, ZMQ comms).\n"
     "       --dp N: data parallelism (N replicas, each handles separate requests).\n"
     "       TP for models that don't fit; DP for throughput scaling."),

    ("--quantization",
     "awq, gptq, fp8, gguf.\n"
     "       awq: best quality weight-only INT4 (recommended for A100).\n"
     "       fp8: native H100 support, near-FP16 accuracy, 2× throughput."),
]

for flag, desc in PARAMS:
    print(f"  {flag}")
    for line in desc.split('\n'):
        print(f"         {line.strip()}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: SGLang vs vLLM benchmark simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — SGLang vs vLLM performance by workload type")
print("━" * 65)
print()

np.random.seed(10)

def simulate_ttft(framework, workload, n=200):
    '''Simulate TTFT distribution for different frameworks and workloads.'''
    if workload == "chatbot_no_cache":
        base_ms = {"sglang": 180, "vllm": 195}[framework]
        noise   = 40
    elif workload == "chatbot_shared_sys":
        # SGLang radix tree finds shared prefix faster
        base_ms = {"sglang": 45,  "vllm": 120}[framework]
        noise   = 20
    elif workload == "few_shot_512tok":
        # 512-token prefix shared → SGLang avoids 512-tok prefill on cache hits
        base_ms = {"sglang": 30,  "vllm": 310}[framework]
        noise   = 15
    elif workload == "agent_multistep":
        # Each step reuses prior context
        base_ms = {"sglang": 55,  "vllm": 200}[framework]
        noise   = 25
    elif workload == "tree_of_thought_4":
        # fork(4): SGLang shares root KV
        base_ms = {"sglang": 250, "vllm": 850}[framework]
        noise   = 60

    return np.random.exponential(1, n) * noise + base_ms

def simulate_throughput(framework, workload):
    '''Simulated tokens/sec throughput.'''
    base = {
        ("sglang", "chatbot_no_cache"):   2800,
        ("vllm",   "chatbot_no_cache"):   2650,
        ("sglang", "chatbot_shared_sys"): 5200,
        ("vllm",   "chatbot_shared_sys"): 3100,
        ("sglang", "few_shot_512tok"):    8500,
        ("vllm",   "few_shot_512tok"):    2800,
        ("sglang", "agent_multistep"):    4800,
        ("vllm",   "agent_multistep"):    2400,
        ("sglang", "tree_of_thought_4"):  6200,
        ("vllm",   "tree_of_thought_4"):  2100,
    }
    return base.get((framework, workload), 0)

WORKLOADS = [
    ("chatbot_no_cache",   "Chatbot, no shared prefix"),
    ("chatbot_shared_sys", "Chatbot, 64-tok shared sys prompt (80% hit)"),
    ("few_shot_512tok",    "Few-shot, 512-tok examples prefix (95% hit)"),
    ("agent_multistep",    "Agent loop, multi-step context reuse"),
    ("tree_of_thought_4",  "Tree-of-thought, fork(4)"),
]

print(f"  Model: LLaMA-2 13B, A100 80GB, FP16, batch=16")
print()
print(f"  {'Workload':<42} | {'SGLang TTFT P50':>16} | {'vLLM TTFT P50':>15} | {'Throughput speedup':>20}")
print(f"  {'─'*100}")

for wl_key, wl_label in WORKLOADS:
    ttft_sg = simulate_ttft("sglang", wl_key)
    ttft_vl = simulate_ttft("vllm",   wl_key)
    tput_sg = simulate_throughput("sglang", wl_key)
    tput_vl = simulate_throughput("vllm",   wl_key)
    tput_ratio = tput_sg / tput_vl if tput_vl > 0 else 0

    p50_sg = np.percentile(ttft_sg, 50)
    p50_vl = np.percentile(ttft_vl, 50)

    print(f"  {wl_label:<42} | {p50_sg:12.0f} ms | {p50_vl:11.0f} ms | "
          f"{tput_ratio:18.2f}× (SGLang)")

print()
print("  INTERPRETATION:")
print("  - Chatbot (no cache): SGLang ≈ vLLM (both fast, small gap from FlashInfer)")
print("  - Shared system prompt: SGLang 2.5× better (RadixAttention reuses prefix KV)")
print("  - Few-shot 512-token: SGLang 3× better (512 tokens skipped on cache hit)")
print("  - Agent multi-step: SGLang 2× better (full context history cached in tree)")
print("  - Tree-of-thought: SGLang 3× better (fork() shares root, N branches run free)")
print()
print("  PRODUCTION CHECKLIST:")
print("  ☐ Start server: python -m sglang.launch_server --model-path ...")
print("  ☐ Warmup: send 20 requests before benchmarking")
print("  ☐ Monitor: GET /get_server_info — shows cache hit rate")
print("  ☐ Benchmark: python -m sglang.bench_serving --backend sglang")
print("  ☐ Set --schedule-policy lpm for workloads with shared prefixes")
print("  ☐ Enable --enable-torch-compile for sustained throughput")
print("  ☐ Match --max-total-tokens to your GPU VRAM budget")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · SGLang OpenAI-Compatible API — Client Patterns & Extensions": {
        "description": (
            "Use SGLang via the standard OpenAI Python client. "
            "Show structured output, multi-modal (vision), batch requests. "
            "Cover SGLang-specific API extensions beyond OpenAI. "
            "Demonstrate how to migrate an existing OpenAI application to SGLang "
            "with minimal code changes."
        ),
        "language": "python",
        "code": '''
import json
import numpy as np

print("=" * 65)
print("  SGLANG OPENAI-COMPATIBLE API — CLIENT PATTERNS")
print("=" * 65)
print()
print("  SGLang implements /v1/chat/completions, /v1/completions,")
print("  /v1/embeddings — same as OpenAI API.")
print("  Migration: change base_url from openai.com → localhost:30000")
print()

# ─────────────────────────────────────────────────────────────────────────
# MIGRATION: OpenAI → SGLang
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  MIGRATION — OpenAI API → SGLang (one line change)")
print("━" * 65)
print()

MIGRATION = '''
# BEFORE: OpenAI API
from openai import OpenAI
client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

# AFTER: SGLang (change ONLY base_url and model name)
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:30000/v1",   # ← only change
    api_key="EMPTY",                         # ← SGLang ignores key
)
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",  # ← model name
    messages=[{"role": "user", "content": "Hello!"}]
)

# Everything else stays identical:
# - response.choices[0].message.content
# - streaming with stream=True
# - logprobs, n, temperature, top_p, max_tokens, stop, etc.
'''
print(MIGRATION)

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 1: Structured output via response_format
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 1 — Structured output (JSON schema in request)")
print("━" * 65)
print()

PATTERN_JSON = '''
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

# Method 1: response_format with JSON schema (OpenAI-compatible)
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a data extraction assistant."},
        {"role": "user",   "content": "Extract info from: Alice Smith, age 32, ML engineer."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string"},
                    "age":       {"type": "integer"},
                    "job_title": {"type": "string"},
                },
                "required": ["full_name", "age", "job_title"],
                "additionalProperties": False
            }
        }
    },
    max_tokens=200,
)

# Output is GUARANTEED to be valid JSON matching the schema:
data = json.loads(response.choices[0].message.content)
# → {"full_name": "Alice Smith", "age": 32, "job_title": "ML engineer"}

# Method 2: SGLang-specific regex in extra_body
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "What year was Python created?"}],
    extra_body={
        "regex": r"19\\d{2}|20\\d{2}",   # match a 4-digit year
    },
    max_tokens=4,
)
year = response.choices[0].message.content
# → "1991"   (guaranteed 4-digit year format)
'''
print(PATTERN_JSON)

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 2: Streaming
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 2 — Streaming token-by-token")
print("━" * 65)
print()

PATTERN_STREAM = '''
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

# Streaming — tokens arrive as Server-Sent Events
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Write a short poem about GPUs."}],
    max_tokens=200,
    stream=True,            # ← enable streaming
)

print("Streaming output: ", end="")
full_text = ""
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)   # print each token as it arrives
        full_text += delta

print()  # newline at end
print(f"Total tokens: {len(full_text.split())}")

# For FastAPI endpoints, use async streaming:
from fastapi.responses import StreamingResponse
import asyncio

async def stream_response(prompt: str):
    async_client = AsyncOpenAI(
        base_url="http://localhost:30000/v1", api_key="EMPTY"
    )
    async with async_client.chat.completions.stream(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    ) as stream:
        async for text in stream.text_stream:
            yield f"data: {text}\\n\\n"   # SSE format
'''
print(PATTERN_STREAM)

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 3: Batch API (offline batch processing)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 3 — Batch requests (offline workloads)")
print("━" * 65)
print()

PATTERN_BATCH = '''
# Method 1: Concurrent requests with asyncio (recommended for high throughput)
import asyncio
from openai import AsyncOpenAI

async def classify_batch(reviews: list[str]) -> list[str]:
    client = AsyncOpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

    async def classify_one(review: str) -> str:
        response = await client.chat.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[
                {"role": "system", "content": "Classify sentiment: positive/negative/neutral"},
                {"role": "user",   "content": review}
            ],
            max_tokens=5,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    # All requests fly to the server concurrently
    # SGLang's RadixAttention shares the system prompt KV across all of them
    tasks   = [classify_one(r) for r in reviews]
    results = await asyncio.gather(*tasks)
    return results

# Run:
reviews  = ["Loved it!", "Terrible.", "It was okay."] * 100
sentiments = asyncio.run(classify_batch(reviews))
print(f"Classified {len(sentiments)} reviews")

# Method 2: SGLang's native batch endpoint (v0.3+)
import requests

batch_request = {
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "text": [
        "Review: Loved it! Sentiment:",
        "Review: Terrible. Sentiment:",
        "Review: It was okay. Sentiment:",
    ],
    "sampling_params": {
        "temperature": 0.0,
        "max_new_tokens": 5,
    }
}

response = requests.post(
    "http://localhost:30000/generate",   # SGLang's batch generate endpoint
    json=batch_request
)
results = response.json()["text"]
'''
print(PATTERN_BATCH)

# ─────────────────────────────────────────────────────────────────────────
# SECTION: Cache hit rate monitoring
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SERVER MONITORING — Cache hit rate and performance metrics")
print("━" * 65)
print()

MONITORING = '''
# Check server status and RadixAttention cache stats:
import requests

info = requests.get("http://localhost:30000/get_server_info").json()

print(f"Model:           {info['model_path']}")
print(f"TP degree:       {info['tp_size']}")
print(f"Max KV tokens:   {info['max_total_num_tokens']:,}")
print(f"Prefix cache hit rate: {info['cache_hit_rate']:.1%}")

# Key metric to watch:
# cache_hit_rate > 0.5  → RadixAttention is working well
# cache_hit_rate < 0.1  → workload has few shared prefixes → consider vLLM instead

# Health check (for load balancer probes):
health = requests.get("http://localhost:30000/health").status_code
# → 200 if healthy

# Flush radix cache (use when switching workloads):
requests.post("http://localhost:30000/flush_cache")

# Benchmark:
# python -m sglang.bench_serving \\
#     --backend sglang \\
#     --model meta-llama/Llama-2-7b-chat-hf \\
#     --num-prompts 500 \\
#     --request-rate 20 \\
#     --tokenizer meta-llama/Llama-2-7b-chat-hf
'''
print(MONITORING)

print()
print("━" * 65)
print("  DEPLOYMENT COMPARISON: vLLM vs SGLang vs TRT-LLM")
print("━" * 65)
print()
print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Criterion          │ TRT-LLM       │ vLLM          │ SGLang      │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Deploy speed       │ Hours (build) │ Minutes       │ Minutes     │")
print("  │ Peak throughput    │ ★★★ Best     │ ★★ Good       │ ★★ Good     │")
print("  │ Shared prefix KV   │ ★ Hash cache  │ ★★ Hash cache │ ★★★ Radix  │")
print("  │ Multi-call programs│ ✗ None        │ ✗ None         │ ✅ Native   │")
print("  │ Constrained output │ ✗ Via plugin  │ ✅ outlines    │ ✅ Native   │")
print("  │ LoRA serving       │ ✅ Supported  │ ✅ Mature      │ ✅ Beta     │")
print("  │ AMD/ROCm           │ ✗ NVIDIA only │ ✅ Supported   │ ✅ Supported│")
print("  │ Model update       │ Full rebuild  │ Reload         │ Reload      │")
print("  │ Community size     │ Medium        │ Largest        │ Growing     │")
print("  └──────────────────────────────────────────────────────────────────┘")
"""
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