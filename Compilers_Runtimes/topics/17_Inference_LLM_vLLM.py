"""
vLLM — High-Throughput LLM Serving with PagedAttention
========================================================

vLLM's central contribution is PagedAttention: a KV cache memory manager
inspired by virtual memory paging in operating systems. Before vLLM (2023),
LLM serving systems pre-allocated a contiguous block of GPU memory for each
request's maximum possible context length — whether it used that memory or not.
Memory fragmentation was severe, concurrency was low, and throughput suffered.

PagedAttention treats the KV cache as a pool of fixed-size pages, allocated
on demand and freed immediately when a request finishes. The result: near-zero
memory waste, 2–4× more concurrent requests, and proportionally higher
throughput — all without changing model weights or accuracy.

This module covers PagedAttention, continuous batching, the OpenAI-compatible
server, sampling parameters, and the complete vLLM deployment stack.

"""

import textwrap
import re

TOPIC_NAME = "vLLM — PagedAttention & High-Throughput LLM Serving"
DISPLAY_NAME = "17 · vLLM"
ICON = "📄"
SUBTITLE = "Virtual Memory for the KV Cache — Maximum GPU Concurrency"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE MEMORY PROBLEM VLLM SOLVES

### Pre-vLLM KV Cache Allocation

    Before vLLM, every serving system allocated KV cache memory like this:

        request arrives → allocate MAX_SEQ_LEN slots upfront → generate tokens

    Problem 1: INTERNAL FRAGMENTATION
        Allocate for 2048 tokens, use 47 tokens, waste 2001 slots.
        For unpredictable outputs (chat), average utilisation ≈ 20–40%.

    Problem 2: EXTERNAL FRAGMENTATION
        Each request needs a CONTIGUOUS block of GPU memory.
        Memory becomes fragmented → can't fit new requests even with free memory.

        Example (8 GB KV cache, 2 GB per slot):
            [Req A: 2GB][Req B: 2GB][FREE: 2GB][Req C: 2GB]
            Req B finishes: [Req A: 2GB][FREE: 2GB][FREE: 2GB][Req C: 2GB]
            New request needs 3 GB contiguous: REJECTED (only 2 GB contiguous)
            Total free: 4 GB — but contiguous: only 2 GB.

    Problem 3: MEMORY RESERVATION
        Systems reserved memory for the WORST CASE output length.
        A system supporting 2048-token outputs couldn't serve many requests,
        even when most outputs were 50 tokens.

    These three problems together meant GPU memory utilisation was typically
    20–40% effective — paying for 80 GB but using 16–32 GB productively.

### The OS Inspiration: Virtual Memory Paging

    Modern operating systems solved analogous problems for RAM decades ago:

        Physical memory:  actual RAM chips (limited, contiguous addresses)
        Virtual memory:   each process sees a large, contiguous address space
        Pages:            fixed-size blocks (4 KB) — the unit of management
        Page table:       maps virtual page numbers → physical frame numbers

    Key insight: "contiguous" in virtual space ≠ "contiguous" in physical space.
    Processes see contiguous addresses; OS maps them to scattered physical frames.

    vLLM applies EXACTLY this idea to the KV cache:

        Physical KV memory: actual GPU VRAM blocks
        Logical KV space:   each request sees a "contiguous" KV sequence
        KV blocks:          fixed-size pages (16–32 tokens each, typically)
        Block table:        maps logical block → physical block (per request)

    Result: requests share the physical KV pool, fragments don't matter,
    allocation is O(1), deallocation is immediate.


##### PART 2 — PAGEDATTENTION: THE CORE ALGORITHM

### Block Structure

    vLLM divides the KV cache into fixed-size blocks:

        block_size:         number of tokens per block (default: 16)
        num_blocks:         total blocks in the KV pool
        block_size_bytes:   2 × num_layers × num_kv_heads × head_dim × block_size × bpe

    For LLaMA-2 7B, block_size=16, FP16:
        = 2 × 32 × 32 × 128 × 16 × 2 bytes = 8,388,608 bytes = 8 MB per block

    An A100 80GB with 14GB for model weights, 66GB for KV:
        = 66 GB / 8 MB ≈ 8,250 blocks
        = 8,250 × 16 = 132,000 token slots

### The Block Table

    Each request has a BLOCK TABLE: a list of physical block IDs.

        Request A (47 tokens, block_size=16):
            Needs: ceil(47/16) = 3 blocks
            Block table: [block_7, block_23, block_41]   ← scattered physical blocks
            Token layout:
                block_7:  tokens 0–15
                block_23: tokens 16–31
                block_41: tokens 32–46, slots 47–63 unused (last block only!)

        Request B (32 tokens):
            Block table: [block_2, block_19]

    The attention kernel walks the block table to gather K, V:

        for block_id in request.block_table:
            K_block = kv_cache[block_id, 0, :, :]   # keys
            V_block = kv_cache[block_id, 1, :, :]   # values
            attend(Q_new, K_block, V_block)

### Lifecycle of a Block

    PREFILL:
        - Allocate ceil(prompt_len / block_size) blocks from free pool
        - Fill them with K, V computed during prefill
        - Last block may be partially filled (that's OK!)

    DECODE (each step):
        - New token fits in current last block if it has a free slot
        - If last block is full, allocate ONE new block from free pool
        - Append new K, V to the last block

    COMPLETION:
        - Request finishes → ALL its blocks returned to free pool immediately
        - No GC, no defragmentation needed — O(1) deallocation

    Diagram — Block allocation as generation progresses:

    Step 0 (prompt=20 tokens, block_size=16):
        Block table: [B5, B12]
        B5:  [t0..t15]  (full)
        B12: [t16..t19, _, _, _, _, _, _, _, _, _, _, _]  (4/16 used)

    Step 1–12 (generate 12 tokens):
        Block table: [B5, B12]
        B12 now: [t16..t31]  (full!)

    Step 13 (allocate new block):
        Block table: [B5, B12, B7]
        B7:  [t32, _, _, _, _, ...]  (1/16 used)

    Step N (request finishes at token 47):
        B5, B12, B7 → returned to free pool
        Next request can immediately use these blocks.

### Copy-on-Write for Parallel Sampling

    When sampling multiple completions for the same prompt (beam search,
    best-of-N, parallel sampling), the PROMPT blocks are shared:

        Request for "generate 4 completions of this prompt":
            Prompt blocks (shared, read-only): [B5, B12]
            Completion A decode blocks: [B3, B8, ...]
            Completion B decode blocks: [B9, ...]
            Completion C decode blocks: [B14, ...]
            Completion D decode blocks: [B21, ...]

    Prompt blocks have ref_count=4 (shared by 4 completions).
    Each completion writes its own new decode blocks.
    Copy-on-Write: if a completion needs to modify a shared block → copy first.

    Memory saving: prompt tokens stored ONCE instead of 4×.
    For long prompts (512 tokens), sharing saves 32 blocks = 256 MB.


##### PART 3 — CONTINUOUS BATCHING IN VLLM

### The vLLM Scheduler

    vLLM's scheduler runs at every decode step and makes three decisions:

    1. PREEMPTION:  if no free blocks exist, which running request to pause?
    2. ADMISSION:   which waiting requests can start their prefill?
    3. BATCHING:    which requests execute in this step (prefill + decode mixed)?

    Priority: Running > Swapped > Waiting

    States a request can be in:
        WAITING:   in the queue, no KV cache allocated yet
        RUNNING:   actively generating, has KV blocks
        SWAPPED:   was running, preempted to CPU swap space (blocks freed)
        FINISHED:  done, blocks freed

### Chunked Prefill (vLLM v2)

    Problem with mixed prefill + decode batching:
        A long prefill (1024 tokens) dominates the batch — decode requests
        starve and TTFT spikes for existing users.

    Solution: CHUNKED PREFILL
        Break prefill into chunks of `max_num_chunked_tokens` (e.g., 512).
        Interleave prefill chunks with decode steps.

        Without chunked prefill:
            Step 1: [1024-token prefill for new user]     ← decode users wait 1 step
            Step 2: [decode for all users]

        With chunked prefill (chunk_size=512):
            Step 1: [512-token prefill chunk, decode for existing users]
            Step 2: [512-token prefill chunk, decode for existing users]
            ← decode users experience no stall

    Effect: more predictable inter-token latency (ITL) under load.
    Trade-off: new request TTFT slightly longer (prefill spread over 2 steps).

### Preemption: Recompute vs Swap

    When the KV cache pool is full and a new high-priority request arrives:

    Option 1: SWAP (vLLM default for short sequences)
        Copy preempted request's KV blocks to CPU RAM.
        Free GPU blocks → allocate to new request.
        Later: swap back when blocks available.
        Cost: ~10 GB/s CPU-GPU bandwidth → slow for long contexts.

    Option 2: RECOMPUTE (vLLM default for long sequences)
        Simply discard the KV blocks (free GPU memory).
        When the request resumes: run prefill again from scratch.
        Cost: re-run prefill (compute) vs swap (memory bandwidth).
        For long sequences: recompute is cheaper than swap.

    Threshold: vLLM uses sequence length heuristic to choose.
    TRT-LLM uses a similar mechanism with configurable policy.


##### PART 4 — THE VLLM RUNTIME AND EXECUTION ENGINE

### Architecture Overview

    ┌─────────────────────────────────────────────────────────────┐
    │  Client (HTTP / Python API)                                 │
    ├─────────────────────────────────────────────────────────────┤
    │  AsyncLLMEngine  (async request handler, streaming support) │
    ├─────────────────────────────────────────────────────────────┤
    │  Scheduler       (block allocation, preemption, batching)   │
    ├─────────────────────────────────────────────────────────────┤
    │  BlockManager    (PagedAttention block table management)    │
    ├─────────────────────────────────────────────────────────────┤
    │  Worker(s)       (one per GPU, executes model forward pass)  │
    │  ├── ModelRunner (PyTorch model + custom CUDA kernels)       │
    │  └── CacheEngine (manages KV cache tensors on GPU)          │
    ├─────────────────────────────────────────────────────────────┤
    │  PagedAttention CUDA Kernel  (attention over scattered KV)  │
    ├─────────────────────────────────────────────────────────────┤
    │  NVIDIA GPU                                                 │
    └─────────────────────────────────────────────────────────────┘

### The PagedAttention CUDA Kernel

    Standard FlashAttention assumes contiguous K, V tensors.
    PagedAttention has a custom CUDA kernel that:
        1. Loads the block table for the request (list of block IDs)
        2. For each block, fetches K and V from scattered GPU addresses
        3. Computes partial attention, accumulates with running softmax
        4. Writes the output token embedding

    The kernel is the key innovation — it handles scattered memory
    with minimal overhead vs contiguous FlashAttention.

    Two kernel variants:
        PagedAttention V1: one warp per sequence position
        PagedAttention V2: one warp per block (better for long sequences)

### Model Support and Backends

    vLLM supports models through a plugin architecture.
    Each model registers attention, FFN, and embedding implementations.

    Supported:
        LLaMA 1/2/3, Mistral, Mixtral (MoE), Falcon, GPT-2/J/NeoX,
        Qwen 1/2, Phi 1/2/3, Gemma 1/2, Command-R, DeepSeek, Yi,
        Whisper, LLaVA (vision-language), and 50+ more.

    Attention backends (selectable):
        Flash Attention 2 (default on most GPUs)
        Flash Attention 3 (H100, experimental)
        Xformers (fallback)
        FlashInfer (high-performance alternative)

    Quantisation backends:
        AWQ, GPTQ, SqueezeLLM, FP8, bitsandbytes (NF4/INT8)
        All integrated via the vLLM quantisation config.


##### PART 5 — THE OPENAI-COMPATIBLE SERVER

### Why the OpenAI API Compatibility Matters

    The OpenAI Chat Completions API (/v1/chat/completions) has become
    the de facto standard interface for LLM serving.
    Every client library, agent framework, and application supports it.

    vLLM implements this API exactly, meaning:
        - Drop-in replacement for OpenAI API (change base_url, keep code)
        - Works with LangChain, LlamaIndex, AutoGen, CrewAI, etc.
        - Same request/response format → no client code changes needed

### Server Launch

    # Basic launch (single GPU)
    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --port 8000

    # Production launch (multi-GPU, quantised)
    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-2-70b-chat-hf \\
        --tensor-parallel-size 4 \\
        --dtype bfloat16 \\
        --quantization awq \\
        --max-model-len 4096 \\
        --max-num-seqs 256 \\
        --gpu-memory-utilization 0.92 \\
        --enable-chunked-prefill \\
        --port 8000

### Sampling Parameters (vLLM-specific extensions)

    Standard OpenAI parameters:
        temperature:     randomness (0 = greedy, 1 = full distribution)
        top_p:           nucleus sampling (keep top p probability mass)
        max_tokens:      max output tokens
        stop:            stop sequences
        stream:          stream tokens as generated (SSE)

    vLLM extensions (beyond OpenAI):
        top_k:               keep only top-k tokens (additional filtering)
        repetition_penalty:  penalise repeating tokens (> 1.0 = less repeat)
        frequency_penalty:   penalise tokens proportional to frequency
        presence_penalty:    penalise any previously seen token (binary)
        best_of:             generate N completions, return best (uses CoW)
        use_beam_search:     beam search instead of sampling
        min_p:               minimum probability threshold (alternative to top_p)
        skip_special_tokens: strip <bos>, <eos> from output
        spaces_between_special_tokens: formatting control

### Sampling Algorithm Deep Dive

    Greedy decoding (temperature=0):
        token = argmax(logits)
        Deterministic. Best for code, structured output, factual Q&A.

    Temperature sampling:
        probs = softmax(logits / temperature)
        token = sample(probs)
        temperature < 1: sharpens distribution (more focused)
        temperature > 1: flattens distribution (more random)
        temperature → 0: approaches greedy

    Top-K sampling:
        Keep only K highest-probability tokens, renormalise, sample.
        Prevents sampling from the long tail of unlikely tokens.
        Common: top_k=50

    Top-P (nucleus) sampling:
        Keep smallest set of tokens whose cumulative probability ≥ P.
        Adaptive: fewer tokens when distribution is peaked, more when flat.
        Common: top_p=0.9 or top_p=0.95

    Min-P sampling (newer, vLLM specific):
        Keep tokens where p(token) ≥ min_p × max_p(token)
        Scales threshold relative to the best token — adapts better than top_k.
        Common: min_p=0.05

    Combined pipeline (vLLM applies in this order):
        logits → temperature → top_k filter → top_p filter → min_p filter → sample


##### PART 6 — TENSOR PARALLELISM AND MULTI-GPU IN VLLM

### vLLM's Parallelism Strategy

    Tensor Parallelism (TP):
        --tensor-parallel-size N
        Uses Ray actors under the hood (one Worker actor per GPU).
        Each worker holds a shard of the model weights.
        AllReduce via NCCL after each attention and FFN layer.
        Supports TP = 1, 2, 4, 8 (powers of 2 only, within one node).

    Pipeline Parallelism (PP):
        --pipeline-parallel-size N  (vLLM v0.4+)
        Layers split across nodes.
        Combined: TP=4, PP=2 → 8 GPUs across 2 nodes.

    Expert Parallelism (EP):
        For Mixture-of-Experts models (Mixtral, DeepSeek).
        Different experts placed on different GPUs.
        --enable-expert-parallel (vLLM v0.5+)

### Distributed Execution Model

    vLLM uses RAY for multi-GPU orchestration (same Ray from the AI Frameworks module):

        Driver process  → LLMEngine, Scheduler, BlockManager (single process)
        Worker actors   → one per GPU, managed by Ray, execute forward passes

    Communication flow each decode step:
        Driver:  compute batch assignments, block tables
        → Send: input_ids, block_tables, sampling_params to all workers
        → Workers: forward pass in parallel (NCCL AllReduce between them)
        → Workers: return sampled token IDs to driver
        Driver:  update block tables, check stop conditions, stream tokens


##### PART 7 — VLLM v2: ENGINE CORE REWRITE

### What Changed in vLLM v2 (2024–2025)

    vLLM v0.6+ introduced "V2" architectural changes addressing performance
    limits of the original design:

    1. PREFIX CACHING (Automatic KV Cache Reuse):
        If two requests share the same prefix (system prompt, few-shot examples),
        the KV cache for those tokens is computed ONCE and reused.

        Implementation: hash each block's token IDs.
        On new request: look up prefix hash → reuse matching physical blocks.
        Blocks with matching prefix hashes are ref-counted, not duplicated.

        Impact: 2–4× speedup for workloads with shared system prompts.
        Shared system prompt KV = computed once, shared across 1000s of requests.

    2. CHUNKED PREFILL (default enabled in v2):
        Interleave prefill and decode at token level (described in Part 3).

    3. DISAGGREGATED PREFILL:
        Run prefill on dedicated GPU(s), decode on separate GPU(s).
        Prefill is compute-bound; decode is memory-bandwidth-bound.
        Different GPU types optimised for each → cost efficiency.

    4. SPEC DECODE INTEGRATION:
        Draft model speculation built into the scheduler.
        Automatic fallback when batch size exceeds threshold.

### Prefix Caching Deep Dive

    Without prefix caching (every request):
        "You are a helpful assistant.\n\nUser: What is..."
        → prefill 20 system prompt tokens EVERY request
        → 20 × (num_layers × kv_heads × head_dim) bytes allocated EVERY request

    With prefix caching:
        First request:  compute and cache system prompt KV blocks (hash: 0xABCD...)
        All subsequent requests with same system prompt:
            → block lookup hits → skip prefill for those tokens
            → directly start generating from the last user token

    Block hash = hash of (token_ids in block + hash of parent block)
    This creates a HASH CHAIN: each block's identity depends on all tokens before it.

    ┌────────────────────────────────────────────────────────────┐
    │  Block 0: hash(system_prompt[:16])    → 0xABCD             │
    │  Block 1: hash(system_prompt[16:32] + 0xABCD) → 0xEF01     │
    │  Block 2: hash(user_prompt[:16]      + 0xEF01) → 0x2345     │
    │                                                            │
    │  New request: same system prompt?                          │
    │    Block 0: hash matches 0xABCD → reuse (ref_count++)       │
    │    Block 1: hash matches 0xEF01 → reuse (ref_count++)       │
    │    Block 2: different user query  → allocate new            │
    └────────────────────────────────────────────────────────────┘


##### PART 8 — VLLM DEPLOYMENT: WHEN TO USE IT AND HOW

### vLLM vs TRT-LLM: The Core Tradeoff

    vLLM:
        - PyTorch-based: fast to deploy, easy to update models
        - No compilation step: load model → serve in minutes
        - OpenAI-compatible API out of the box
        - Superior memory efficiency via PagedAttention
        - Best for: research serving, rapid deployment, heterogeneous workloads

    TRT-LLM:
        - Compiled engine: slower to deploy, maximum throughput
        - 10–30 min compilation per model/config
        - Better raw latency and throughput on NVIDIA hardware
        - Best for: production at scale, SLA-bound serving, dedicated NVIDIA infra

    Performance comparison (A100 80GB, LLaMA-2 7B, batch=32):
        Throughput:  TRT-LLM ~20% faster than vLLM (compiled vs PyTorch)
        Latency:     TRT-LLM ~15% lower P50 latency
        Memory util: vLLM better (PagedAttention vs TRT-LLM's paged KV)
        Deploy time: vLLM minutes vs TRT-LLM hours

### When to Use vLLM

    ✅  You need to serve models today, not after a 30-min compile
    ✅  Your model updates frequently (new weights, LoRA adapters)
    ✅  You need OpenAI-compatible API without infrastructure work
    ✅  Mixed GPU environments (AMD ROCm, AWS Inferentia, Google TPU)
    ✅  Long contexts where PagedAttention's memory efficiency matters most
    ✅  Research deployments: quick iteration, many model variants
    ✅  LoRA serving: multiple adapters loaded simultaneously
    ✅  Shared system prompts: prefix caching gives large speedup
    ❌  Absolute maximum throughput on H100 (TRT-LLM wins by ~20%)
    ❌  INT4 with NVIDIA-specific optimisations (TRT-LLM is more optimised)

### LoRA Serving

    vLLM supports serving MULTIPLE LoRA adapters on ONE base model:

    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-2-7b \\
        --enable-lora \\
        --lora-modules sql-lora=./sql_adapter code-lora=./code_adapter

    At request time: specify which adapter to use per request:
        "model": "sql-lora"   or   "model": "code-lora"

    vLLM dynamically swaps LoRA weights per request in the same batch.
    One base model serves many fine-tuned variants with minimal overhead.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · PagedAttention Block Manager — Simulate KV Cache Paging": {
        "description": (
            "Implement the PagedAttention block manager from scratch. "
            "Simulate block allocation, deallocation, copy-on-write for parallel sampling. "
            "Show how memory fragmentation is eliminated vs contiguous allocation. "
            "Measure effective memory utilisation for both approaches."
        ),
        "language": "python",
        "code": '''
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional

print("=" * 65)
print("  PAGEDATTENTION BLOCK MANAGER — KV CACHE PAGING SIMULATION")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Block Manager Implementation
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class KVBlock:
    """One physical KV cache block (holds block_size token KV pairs)."""
    block_id:   int
    ref_count:  int = 0       # how many sequences share this block
    token_ids:  List[int] = field(default_factory=list)   # for prefix caching

    def is_free(self):
        return self.ref_count == 0

    def is_full(self, block_size: int):
        return len(self.token_ids) >= block_size


class BlockManager:
    """
    Manages the physical KV cache block pool.
    Inspired by vLLM's BlockAllocator.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks  = num_blocks
        self.block_size  = block_size
        self.blocks      = [KVBlock(block_id=i) for i in range(num_blocks)]
        self.free_blocks = set(range(num_blocks))   # IDs of available blocks
        self.alloc_stats = defaultdict(int)          # track allocation events

    def allocate(self) -> Optional[KVBlock]:
        """Allocate one free block. Returns None if pool exhausted."""
        if not self.free_blocks:
            return None
        block_id = self.free_blocks.pop()
        block    = self.blocks[block_id]
        block.ref_count  = 1
        block.token_ids  = []
        self.alloc_stats['allocations'] += 1
        return block

    def free(self, block: KVBlock):
        """Decrement ref count. Free block if ref_count reaches 0."""
        block.ref_count -= 1
        if block.ref_count == 0:
            block.token_ids = []
            self.free_blocks.add(block.block_id)
            self.alloc_stats['frees'] += 1

    def copy_on_write(self, block: KVBlock) -> KVBlock:
        """
        If a shared block needs to be written (fork), copy it first.
        Returns the (possibly new) block that is safe to write to.
        """
        if block.ref_count == 1:
            return block   # only one owner — write in place
        # Multiple owners — must copy
        new_block = self.allocate()
        if new_block is None:
            raise RuntimeError("OOM during copy-on-write")
        new_block.token_ids = block.token_ids.copy()
        self.free(block)   # decrement original ref count
        self.alloc_stats['cow_copies'] += 1
        return new_block

    @property
    def num_free(self):
        return len(self.free_blocks)

    @property
    def utilisation(self):
        used = self.num_blocks - len(self.free_blocks)
        return used / self.num_blocks


class PagedSequence:
    """
    One request's KV cache state in PagedAttention.
    Holds a list of physical KV blocks.
    """

    def __init__(self, seq_id: int, block_manager: BlockManager):
        self.seq_id       = seq_id
        self.bm           = block_manager
        self.block_table  : List[KVBlock] = []
        self.num_tokens   = 0

    def append_tokens(self, new_tokens: List[int]):
        """Add tokens to the sequence, allocating blocks as needed."""
        for tok in new_tokens:
            # Need a new block?
            if (not self.block_table or
                    self.block_table[-1].is_full(self.bm.block_size)):
                blk = self.bm.allocate()
                if blk is None:
                    raise RuntimeError(f"OOM: Seq {self.seq_id} needs block but pool empty")
                self.block_table.append(blk)
            self.block_table[-1].token_ids.append(tok)
            self.num_tokens += 1

    def fork(self, new_seq_id: int) -> 'PagedSequence':
        """
        Fork this sequence (for parallel sampling / beam search).
        Prompt blocks are SHARED (ref_count++), not copied.
        """
        new_seq = PagedSequence(new_seq_id, self.bm)
        for blk in self.block_table:
            blk.ref_count += 1          # increment — now shared
        new_seq.block_table = self.block_table.copy()  # same physical blocks
        new_seq.num_tokens  = self.num_tokens
        return new_seq

    def free(self):
        """Release all blocks."""
        for blk in self.block_table:
            self.bm.free(blk)
        self.block_table = []
        self.num_tokens  = 0

    @property
    def num_blocks_used(self):
        return len(self.block_table)

    @property
    def wasted_slots(self):
        if not self.block_table:
            return 0
        # Waste = unused slots in last block only
        last_used = len(self.block_table[-1].token_ids)
        return self.bm.block_size - last_used


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Basic allocation and deallocation
# ─────────────────────────────────────────────────────────────────────────
BLOCK_SIZE  = 16
NUM_BLOCKS  = 50
bm = BlockManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)

print("━" * 65)
print("  SECTION 1 — Basic block allocation lifecycle")
print("━" * 65)
print()
print(f"  Pool: {NUM_BLOCKS} blocks × {BLOCK_SIZE} tokens/block = "
      f"{NUM_BLOCKS * BLOCK_SIZE} total token slots")
print()

# Simulate 3 requests
requests = {
    'req_A': PagedSequence('A', bm),
    'req_B': PagedSequence('B', bm),
    'req_C': PagedSequence('C', bm),
}

# Prefill requests with different prompt lengths
prompts = {'req_A': 47, 'req_B': 20, 'req_C': 33}
for name, seq in requests.items():
    prompt_tokens = list(range(prompts[name]))
    seq.append_tokens(prompt_tokens)

print(f"  After prefill:")
for name, seq in requests.items():
    print(f"    {name}: {seq.num_tokens:3d} tokens, "
          f"{seq.num_blocks_used} blocks, {seq.wasted_slots} wasted slots in last block")
print(f"  Free blocks: {bm.num_free}/{NUM_BLOCKS}  "
      f"(utilisation: {bm.utilisation*100:.1f}%)")
print()

# Generate 20 tokens for req_A
for tok in range(100, 120):
    requests['req_A'].append_tokens([tok])

print(f"  After req_A generates 20 tokens:")
seq = requests['req_A']
print(f"    req_A: {seq.num_tokens:3d} tokens, {seq.num_blocks_used} blocks, "
      f"{seq.wasted_slots} wasted slots")
print(f"  Free blocks: {bm.num_free}/{NUM_BLOCKS}")
print()

# req_B finishes
requests['req_B'].free()
print(f"  req_B finishes → blocks freed immediately")
print(f"  Free blocks: {bm.num_free}/{NUM_BLOCKS}  "
      f"(blocks available for next request instantly)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Copy-on-Write for parallel sampling
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Copy-on-Write (parallel sampling / best-of-N)")
print("━" * 65)
print()

bm2   = BlockManager(num_blocks=100, block_size=16)
base  = PagedSequence('base', bm2)
base.append_tokens(list(range(64)))   # 64-token prompt

print(f"  Prompt: 64 tokens → {base.num_blocks_used} full blocks")
print(f"  Generating 4 parallel completions (best-of-4)...")
print()

# Fork into 4 completions
forks = [base.fork(f'fork_{i}') for i in range(4)]

print(f"  After forking: each fork SHARES the 4 prompt blocks (no copy!)")
print(f"  Prompt block ref counts: "
      f"{[blk.ref_count for blk in base.block_table]}")
print(f"  Free blocks used for forking: 0  (shared, not copied)")
print()

# Each fork generates different tokens (diverge)
for i, fork in enumerate(forks):
    fork.append_tokens([1000 + i * 10 + j for j in range(20)])

print(f"  After each fork generates 20 tokens:")
print(f"  {'Completion':>14} | {'Total tokens':>13} | {'Total blocks':>13} | "
      f"{'New blocks (decode only)':>24}")
for fork in forks:
    decode_blocks = fork.num_blocks_used - base.num_blocks_used
    print(f"  {fork.seq_id:>14} | {fork.num_tokens:13d} | "
          f"{fork.num_blocks_used:13d} | {decode_blocks:>24d}")

prompt_blocks_total  = base.num_blocks_used * 4
prompt_blocks_shared = base.num_blocks_used
savings_pct = (1 - 1/4) * 100
print()
print(f"  Without CoW: would need {prompt_blocks_total} blocks for 4 copies of prompt")
print(f"  With CoW:    {prompt_blocks_shared} shared + "
      f"{sum(f.num_blocks_used - base.num_blocks_used for f in forks)} decode "
      f"= {prompt_blocks_shared + sum(f.num_blocks_used - base.num_blocks_used for f in forks)} total")
print(f"  Memory saving: {savings_pct:.0f}% of prompt KV cache")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Fragmentation comparison — paged vs contiguous
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Memory fragmentation: paged vs contiguous allocation")
print("━" * 65)
print()

np.random.seed(1)

def simulate_contiguous(requests_spec, max_seq_len, total_slots):
    """
    Contiguous allocator: reserve MAX_SEQ_LEN slots per request upfront.
    Returns utilisation stats.
    """
    slots_used = 0
    slots_wasted = 0
    max_concurrent = 0
    concurrent = 0
    requests_served = 0
    rejected = 0

    for actual_len in requests_spec:
        reserved = max_seq_len   # reserve max even if we use less
        if slots_used + reserved > total_slots:
            rejected += 1
            continue
        # Request served
        slots_used  += reserved
        slots_wasted += reserved - actual_len
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        requests_served += 1
        # Simulate completion: half the requests have finished by now
        if np.random.random() < 0.5:
            slots_used -= reserved
            concurrent -= 1

    total_allocated = requests_served * max_seq_len
    effective_util  = (total_allocated - slots_wasted) / total_slots if total_slots > 0 else 0
    return {
        'served': requests_served, 'rejected': rejected,
        'effective_util': effective_util,
        'waste_pct': slots_wasted / (total_allocated + 1e-9) * 100,
        'max_concurrent': max_concurrent,
    }

def simulate_paged(requests_spec, block_size, num_blocks):
    """
    PagedAttention allocator: allocate blocks on demand.
    Returns utilisation stats.
    """
    bm_sim = BlockManager(num_blocks=num_blocks, block_size=block_size)
    seqs   = {}
    requests_served = 0
    rejected = 0
    max_concurrent  = 0

    for i, actual_len in enumerate(requests_spec):
        needed_blocks = (actual_len + block_size - 1) // block_size
        if bm_sim.num_free < needed_blocks:
            rejected += 1
            continue
        seq = PagedSequence(i, bm_sim)
        seq.append_tokens(list(range(actual_len)))
        seqs[i] = seq
        requests_served += 1
        max_concurrent   = max(max_concurrent, len(seqs))

        # Simulate completion: half the requests finish
        if np.random.random() < 0.5 and seqs:
            finished_id = next(iter(seqs))
            total_wasted = sum(s.wasted_slots for s in seqs.values())
            seqs[finished_id].free()
            del seqs[finished_id]

    # Measure waste across all active seqs
    total_used   = sum(s.num_blocks_used * block_size for s in seqs.values())
    total_actual = sum(s.num_tokens for s in seqs.values())
    waste_pct    = (total_used - total_actual) / (total_used + 1e-9) * 100

    return {
        'served': requests_served, 'rejected': rejected,
        'effective_util': total_actual / (num_blocks * block_size + 1e-9),
        'waste_pct': waste_pct,
        'max_concurrent': max_concurrent,
    }

# Generate requests with log-normal output lengths (realistic distribution)
N_REQS       = 200
MAX_SEQ_LEN  = 512
TOTAL_SLOTS  = 8192   # total token slots
BLOCK_SIZE   = 16
NUM_BLOCKS   = TOTAL_SLOTS // BLOCK_SIZE

actual_lens = np.clip(
    np.random.lognormal(mean=3.5, sigma=1.2, size=N_REQS).astype(int), 10, MAX_SEQ_LEN
)

cont = simulate_contiguous(actual_lens, MAX_SEQ_LEN, TOTAL_SLOTS)
paged = simulate_paged(actual_lens, BLOCK_SIZE, NUM_BLOCKS)

print(f"  Simulation: {N_REQS} requests, max_seq_len={MAX_SEQ_LEN}")
print(f"  Total pool: {TOTAL_SLOTS} token slots ({NUM_BLOCKS} blocks × {BLOCK_SIZE} tokens)")
print(f"  Actual lengths: mean={actual_lens.mean():.0f}, P95={np.percentile(actual_lens,95):.0f}")
print()
print(f"  {'Metric':<35} {'Contiguous':>14} {'PagedAttention':>16} {'Winner':>8}")
print(f"  {'─'*78}")
metrics = [
    ("Requests served",          cont['served'],           paged['served'],           '>'),
    ("Requests rejected (OOM)",  cont['rejected'],         paged['rejected'],          '<'),
    ("Max concurrent requests",  cont['max_concurrent'],   paged['max_concurrent'],    '>'),
    ("Effective memory util (%)",cont['effective_util']*100,paged['effective_util']*100,'>'),
    ("Internal waste (%)",       cont['waste_pct'],        paged['waste_pct'],         '<'),
]

for label, cv, pv, prefer in metrics:
    winner = "Paged ✅" if (prefer=='>' and pv>cv) or (prefer=='<' and pv<cv) else "Contiguous"
    print(f"  {label:<35} {cv:>14.1f} {pv:>16.1f} {winner:>8}")

print()
print("  PagedAttention's waste ≈ block_size - 1 tokens per sequence (last block)")
print("  Contiguous allocation's waste ≈ MAX_SEQ_LEN - actual_len per sequence")
print("  For typical chat (mean output 50–150 tokens, max 512): 60–90% reduction in waste")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Prefix Caching — Hash Chains and Cache Hit Rate Simulation": {
        "description": (
            "Implement vLLM's prefix caching mechanism. Build the block hash chain, "
            "simulate cache lookups and hits for workloads with shared system prompts. "
            "Show cache hit rate vs prompt sharing ratio. "
            "Measure compute savings from prefix caching on realistic chat workloads."
        ),
        "language": "python",
        "code": '''
import numpy as np
from collections import defaultdict, OrderedDict
from typing import List, Optional, Tuple
import hashlib

print("=" * 65)
print("  PREFIX CACHING — HASH CHAINS AND CACHE HIT SIMULATION")
print("=" * 65)
print()

np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────────
# Block hash computation
# ─────────────────────────────────────────────────────────────────────────

def compute_block_hash(token_ids: List[int], parent_hash: int = 0) -> int:
    """
    Hash = f(token_ids_in_block, parent_block_hash)
    Parent hash creates a CHAIN: each block's identity depends on all prior tokens.
    Two blocks with identical content but different parents → different hashes.
    This correctly captures the full prefix context.
    """
    h = hashlib.md5()
    h.update(parent_hash.to_bytes(8, 'little', signed=False))
    h.update(bytes(token_ids))
    return int(h.hexdigest()[:16], 16)

def compute_block_hash_chain(all_tokens: List[int],
                              block_size: int) -> List[int]:
    """
    Given a full token sequence, compute the hash chain for all full blocks.
    Returns list of hashes, one per full block.
    """
    hashes = []
    parent = 0
    num_full_blocks = len(all_tokens) // block_size
    for i in range(num_full_blocks):
        block_tokens = all_tokens[i*block_size : (i+1)*block_size]
        h = compute_block_hash(block_tokens, parent)
        hashes.append(h)
        parent = h
    return hashes


# ─────────────────────────────────────────────────────────────────────────
# Prefix cache (LRU eviction)
# ─────────────────────────────────────────────────────────────────────────

class PrefixCache:
    """
    LRU prefix cache mapping block_hash → cached (simulated KV computation cost).
    Tracks hit/miss rates and compute savings.
    """

    def __init__(self, max_cached_blocks: int):
        self.max_blocks   = max_cached_blocks
        self.cache        = OrderedDict()   # LRU: hash → True
        self.stats        = defaultdict(int)

    def lookup_prefix(self, hashes: List[int]) -> int:
        """
        Find the longest cached prefix match.
        Returns number of tokens (blocks × block_size) that can be skipped.
        Walks hash chain: stops at first miss.
        """
        matched_blocks = 0
        for h in hashes:
            if h in self.cache:
                self.cache.move_to_end(h)   # LRU update
                matched_blocks += 1
                self.stats['hits'] += 1
            else:
                self.stats['misses'] += 1
                break
        return matched_blocks

    def cache_blocks(self, hashes: List[int]):
        """Add computed blocks to cache (evict LRU if full)."""
        for h in hashes:
            if h not in self.cache:
                if len(self.cache) >= self.max_blocks:
                    self.cache.popitem(last=False)   # evict LRU
                    self.stats['evictions'] += 1
                self.cache[h] = True
            else:
                self.cache.move_to_end(h)

    @property
    def hit_rate(self):
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0.0

    @property
    def cache_utilisation(self):
        return len(self.cache) / self.max_blocks


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Hash chain correctness demonstration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Hash chain: same tokens, different prefix = different hash")
print("━" * 65)
print()

BLOCK_SIZE = 16

tokens_A = list(range(48))              # tokens 0–47
tokens_B = list(range(16)) + list(range(100, 132))  # same block 0, different blocks 1–2

hashes_A = compute_block_hash_chain(tokens_A, BLOCK_SIZE)
hashes_B = compute_block_hash_chain(tokens_B, BLOCK_SIZE)

print(f"  Sequence A tokens: [0..47]  (3 blocks)")
print(f"  Sequence B tokens: [0..15, 100..131]  (same block 0, different blocks 1-2)")
print()
print(f"  {'Block':>7} | {'Hash A':>20} | {'Hash B':>20} | {'Match?':>8}")
print(f"  {'─'*60}")
for i, (ha, hb) in enumerate(zip(hashes_A, hashes_B)):
    match = "✅ same" if ha == hb else "❌ diff"
    print(f"  {i:7d} | {ha:20d} | {hb:20d} | {match}")

print()
print("  Block 0: same tokens [0..15] AND same parent (0) → identical hash ✅")
print("  Block 1: same tokens [16..31] BUT parent differs → different hash ❌")
print("  Block 2: same tokens [32..47] BUT parent differs → different hash ❌")
print()
print("  This chain property ensures:")
print("  Cache hit on block N means ALL tokens 0..N×block_size are identical.")
print("  No false positives possible — mathematically sound.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Realistic workload simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Prefix cache simulation on chat workloads")
print("━" * 65)
print()

def generate_chat_request(system_prompt: List[int],
                           conversation_history: List[int],
                           user_query: List[int]) -> List[int]:
    """
    Build a full request token sequence:
    [system_prompt] [conversation_history] [user_query]
    """
    return system_prompt + conversation_history + user_query


# Token ID constants (simulated)
SYSTEM_PROMPT_LEN = 64    # e.g., "You are a helpful coding assistant..."
BLOCK_SIZE        = 16

def run_workload_simulation(n_requests, sharing_ratio, cache_size_blocks,
                             label=""):
    """
    sharing_ratio: fraction of requests sharing the same system prompt.
    """
    cache   = PrefixCache(max_cached_blocks=cache_size_blocks)
    np.random.seed(42)

    # Fixed system prompt (shared across sharing_ratio fraction of requests)
    shared_system = list(range(SYSTEM_PROMPT_LEN))   # deterministic IDs

    total_tokens_to_compute = 0
    tokens_skipped          = 0
    n_system_tokens_saved   = 0

    for req_i in range(n_requests):
        # Decide if this request shares the system prompt
        uses_shared = np.random.random() < sharing_ratio
        if uses_shared:
            system = shared_system
        else:
            # Unique system prompt (different user/use case)
            offset = (req_i + 1) * 1000
            system = list(range(offset, offset + SYSTEM_PROMPT_LEN))

        # Variable conversation history (multi-turn chat)
        history_len = np.random.choice([0, 32, 64, 128], p=[0.4, 0.3, 0.2, 0.1])
        history     = list(range(5000 + req_i * 200, 5000 + req_i * 200 + history_len))

        # User query (always unique)
        query_len   = np.random.randint(16, 64)
        query       = list(range(10000 + req_i * 100, 10000 + req_i * 100 + query_len))

        full_tokens = generate_chat_request(system, history, query)
        total_tokens = len(full_tokens)

        # Compute hash chain for all full blocks
        hashes      = compute_block_hash_chain(full_tokens, BLOCK_SIZE)

        # Cache lookup
        matched_blocks = cache.lookup_prefix(hashes)
        matched_tokens = matched_blocks * BLOCK_SIZE
        skipped        = min(matched_tokens, total_tokens)

        tokens_skipped          += skipped
        total_tokens_to_compute += total_tokens - skipped

        if uses_shared and matched_blocks >= SYSTEM_PROMPT_LEN // BLOCK_SIZE:
            n_system_tokens_saved += SYSTEM_PROMPT_LEN

        # Cache all full blocks of this request
        cache.cache_blocks(hashes)

    total_input = (total_tokens_to_compute + tokens_skipped)
    savings_pct = tokens_skipped / total_input * 100 if total_input > 0 else 0

    return dict(
        hit_rate     = cache.hit_rate * 100,
        savings_pct  = savings_pct,
        tokens_saved = tokens_skipped,
        cache_util   = cache.cache_utilisation * 100,
        label        = label,
    )

SCENARIOS = [
    (0.0,  200, "All unique system prompts (no sharing)"),
    (0.5,  200, "50% share system prompt"),
    (0.8,  200, "80% share system prompt (typical chat API)"),
    (0.95, 200, "95% share (single chatbot product)"),
    (1.0,  200, "100% share (identical system prompt)"),
]

print(f"  Simulation: 500 requests, block_size={BLOCK_SIZE}")
print(f"  System prompt: {SYSTEM_PROMPT_LEN} tokens, variable history + unique query")
print()
print(f"  {'Scenario':<40} | {'Hit rate':>9} | {'Tokens saved':>13} | {'Compute saving':>15}")
print(f"  {'─'*83}")

for sharing, cache_sz, desc in SCENARIOS:
    r = run_workload_simulation(500, sharing, cache_sz, desc)
    print(f"  {desc:<40} | {r['hit_rate']:8.1f}% | {r['tokens_saved']:13,} | {r['savings_pct']:14.1f}%")

print()
print("  At 80% sharing (typical API product with shared system prompt):")
print("  ~40–60% of prefill tokens are skipped via cache hits.")
print("  This directly translates to 40–60% reduction in TTFT for cache-hit requests.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Cache size sensitivity
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Cache size vs hit rate (80% sharing workload)")
print("━" * 65)
print()

CACHE_SIZES = [10, 50, 100, 200, 500, 1000, 2000]
print(f"  {'Cache (blocks)':>16} | {'Cache (tokens)':>15} | {'Hit rate':>10} | "
      f"{'Compute saving':>15} | {'Cache util %':>13}")
print(f"  {'─'*76}")

for cs in CACHE_SIZES:
    r = run_workload_simulation(500, 0.8, cs)
    cache_tokens = cs * BLOCK_SIZE
    print(f"  {cs:16d} | {cache_tokens:15,} | {r['hit_rate']:9.1f}% | "
          f"{r['savings_pct']:14.1f}% | {r['cache_util']:12.1f}%")

print()
print("  Diminishing returns: most gains at 100–200 blocks for this workload.")
print("  vLLM's KV block pool automatically acts as the prefix cache pool.")
print("  Larger KV allocation → more prefix cache capacity → better hit rates.")
print()
print("  HOW TO ENABLE PREFIX CACHING IN vLLM:")
print("  python -m vllm.entrypoints.openai.api_server \\")
print("      --model meta-llama/Llama-2-7b \\")
print("      --enable-prefix-caching         # <-- one flag!")
print("  Zero accuracy impact. Zero model changes. Often 2-4x faster for")
print("  chatbot workloads with shared system prompts.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Sampling Algorithms — Temperature, Top-K, Top-P, Min-P": {
        "description": (
            "Implement and compare all major sampling algorithms from scratch. "
            "Show how temperature reshapes the probability distribution. "
            "Implement top-k, top-p (nucleus), and min-p filtering. "
            "Demonstrate the combined sampling pipeline. Measure diversity vs "
            "quality tradeoffs for different parameter combinations."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  SAMPLING ALGORITHMS — Temperature, Top-K, Top-P, Min-P")
print("=" * 65)
print()

np.random.seed(42)
VOCAB_SIZE = 50257   # GPT-2/LLaMA vocab size

# ─────────────────────────────────────────────────────────────────────────
# Helper: create a realistic-looking logit distribution
# ─────────────────────────────────────────────────────────────────────────

def make_logits(scenario="peaked"):
    """Generate a realistic logit distribution."""
    logits = np.random.randn(VOCAB_SIZE).astype(np.float32) - 5.0
    if scenario == "peaked":
        # One clear best token (code generation, factual QA)
        best_tokens = np.random.choice(VOCAB_SIZE, size=3, replace=False)
        logits[best_tokens[0]] = 8.0   # strong winner
        logits[best_tokens[1]] = 4.0
        logits[best_tokens[2]] = 2.0
    elif scenario == "flat":
        # Many equally likely tokens (creative writing)
        top_tokens = np.random.choice(VOCAB_SIZE, size=100, replace=False)
        logits[top_tokens] += 6.0
    elif scenario == "bimodal":
        # Two clusters (ambiguous context)
        cluster_a = np.random.choice(VOCAB_SIZE, size=5,  replace=False)
        cluster_b = np.random.choice(VOCAB_SIZE, size=5,  replace=False)
        logits[cluster_a] = 5.0
        logits[cluster_b] = 4.8
    return logits

def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()   # numerical stability
    exp    = np.exp(logits)
    return exp / exp.sum()

# ─────────────────────────────────────────────────────────────────────────
# Sampling implementations
# ─────────────────────────────────────────────────────────────────────────

def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Divide logits by temperature BEFORE softmax.
    temperature < 1: sharpens distribution (more confident)
    temperature > 1: flattens distribution (more random)
    temperature = 1: unchanged
    temperature → 0: approaches greedy (argmax)
    """
    if temperature == 0.0:
        # Greedy: one-hot at argmax
        probs = np.zeros_like(logits, dtype=np.float32)
        probs[np.argmax(logits)] = 1.0
        return probs
    return softmax(logits / temperature)

def apply_top_k(probs: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only top-k probability tokens. Zero out the rest. Renormalise.
    Prevents sampling from the long low-probability tail.
    """
    if k <= 0 or k >= len(probs):
        return probs
    threshold_idx = np.argpartition(probs, -k)[-k:]   # indices of top-k
    mask          = np.zeros_like(probs)
    mask[threshold_idx] = 1.0
    filtered = probs * mask
    return filtered / filtered.sum()

def apply_top_p(probs: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus sampling: keep smallest set of tokens with cumulative prob >= p.
    Adaptive: uses few tokens when distribution is peaked, many when flat.
    """
    if p >= 1.0:
        return probs
    sorted_indices  = np.argsort(probs)[::-1]   # descending
    sorted_probs    = probs[sorted_indices]
    cumulative      = np.cumsum(sorted_probs)
    # Include tokens until cumulative probability >= p
    cutoff_idx      = np.searchsorted(cumulative, p)
    cutoff_idx      = min(cutoff_idx + 1, len(probs))
    keep            = sorted_indices[:cutoff_idx]
    filtered        = np.zeros_like(probs)
    filtered[keep]  = probs[keep]
    return filtered / filtered.sum()

def apply_min_p(probs: np.ndarray, min_p: float) -> np.ndarray:
    """
    Min-P sampling: keep tokens where p(token) >= min_p × p(best_token).
    Scale threshold relative to the best token — adapts to distribution width.
    """
    if min_p <= 0:
        return probs
    threshold = min_p * probs.max()
    filtered  = np.where(probs >= threshold, probs, 0.0)
    total     = filtered.sum()
    return filtered / total if total > 0 else probs

def sample_token(probs: np.ndarray) -> int:
    """Sample one token index from a probability distribution."""
    return int(np.random.choice(len(probs), p=probs))

def full_sampling_pipeline(logits: np.ndarray,
                            temperature: float = 1.0,
                            top_k: int = 0,
                            top_p: float = 1.0,
                            min_p: float = 0.0) -> tuple:
    """
    vLLM's sampling pipeline (in order):
    logits → temperature → top_k → top_p → min_p → sample
    """
    probs  = apply_temperature(logits, temperature)
    probs  = apply_top_k(probs, top_k)
    probs  = apply_top_p(probs, top_p)
    probs  = apply_min_p(probs, min_p)
    token  = sample_token(probs)
    return token, probs

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Temperature effect
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Temperature effect on distribution")
print("━" * 65)
print()

logits = make_logits("peaked")
top5_idx = np.argsort(logits)[-5:][::-1]

print("  Raw logits (top 5 tokens):")
for i, idx in enumerate(top5_idx):
    print(f"    Token {idx:6d}: logit = {logits[idx]:7.3f}")
print()

TEMPS = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
print(f"  {'Temperature':>13} | {'Entropy (bits)':>15} | {'P(best token)':>15} | "
      f"{'Tokens w/ p>1%':>16} | {'Description'}")
print(f"  {'─'*82}")

for t in TEMPS:
    probs   = apply_temperature(logits, t)
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    p_best  = probs.max()
    n_above = np.sum(probs > 0.01)
    desc    = {0.0: "Greedy (deterministic)", 0.3: "Very focused",
               0.7: "Focused (code/factual)", 1.0: "Default",
               1.5: "Creative/diverse", 2.0: "Chaotic"}.get(t, "")
    print(f"  {t:13.1f} | {entropy:15.3f} | {p_best*100:14.1f}% | "
          f"{n_above:16,d} | {desc}")

print()
print("  Entropy measures 'how random' the sampling is.")
print("  Higher entropy = more diverse outputs = higher perplexity = less coherent.")
print("  Recommended: temperature=0.7 for code, 1.0 for chat, 1.2 for creative.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Top-K vs Top-P comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Top-K vs Top-P (nucleus) on peaked vs flat distributions")
print("━" * 65)
print()

for scenario in ["peaked", "flat"]:
    logits_s = make_logits(scenario)
    probs_s  = softmax(logits_s)

    print(f"  Distribution: '{scenario}'")
    print(f"    Base entropy: {-np.sum(probs_s * np.log2(probs_s + 1e-9)):.2f} bits")
    print()

    configs = [
        ("No filter (base)",      dict(top_k=0,   top_p=1.0)),
        ("Top-K = 10",            dict(top_k=10,  top_p=1.0)),
        ("Top-K = 50",            dict(top_k=50,  top_p=1.0)),
        ("Top-P = 0.9",           dict(top_k=0,   top_p=0.9)),
        ("Top-P = 0.95",          dict(top_k=0,   top_p=0.95)),
        ("Top-K=50 + Top-P=0.9",  dict(top_k=50,  top_p=0.9)),
    ]

    print(f"  {'Filter config':<25} | {'Tokens kept':>12} | {'New entropy':>13} | {'P(best)':>9}")
    print(f"  {'─'*65}")

    for label, params in configs:
        p = apply_top_k(probs_s, params['top_k'])
        p = apply_top_p(p, params['top_p'])
        kept    = np.sum(p > 0)
        entropy = -np.sum(p * np.log2(p + 1e-9))
        p_best  = p.max()
        print(f"  {label:<25} | {kept:12,} | {entropy:13.3f} | {p_best*100:8.2f}%")
    print()

print("  KEY INSIGHT — Top-P adapts to distribution width:")
print("    Peaked distribution: top_p=0.9 keeps ~3 tokens (focused)")
print("    Flat distribution:   top_p=0.9 keeps ~50 tokens (broad)")
print("    Top-K=50 keeps 50 tokens regardless of distribution shape.")
print("    → Top-P is generally better than Top-K for most LLM tasks.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Min-P — newer, adaptive alternative
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Min-P sampling (adaptive alternative to Top-K/Top-P)")
print("━" * 65)
print()
print("  Min-P: keep token if p(token) >= min_p × p(best_token)")
print("  Scales threshold RELATIVE to the winner — naturally adapts to sharpness.")
print()

for scenario in ["peaked", "flat", "bimodal"]:
    logits_s = make_logits(scenario)
    probs_s  = softmax(logits_s)

    print(f"  Scenario: {scenario}")
    print(f"  {'Method':<20} | {'Tokens kept':>12} | {'Entropy (bits)':>15}")
    print(f"  {'─'*52}")

    configs_min = [
        ("Top-P = 0.9",    apply_top_p(probs_s, 0.9)),
        ("Top-K = 50",     apply_top_k(probs_s, 50)),
        ("Min-P = 0.05",   apply_min_p(probs_s, 0.05)),
        ("Min-P = 0.1",    apply_min_p(probs_s, 0.1)),
    ]

    for label, p in configs_min:
        kept    = np.sum(p > 1e-9)
        entropy = -np.sum(p * np.log2(p + 1e-9))
        print(f"  {label:<20} | {kept:12,} | {entropy:15.4f}")
    print()

print("  PRACTICAL RECOMMENDATIONS (vLLM SamplingParams):")
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ Task                     │ Recommended params               │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ Code generation          │ temperature=0.2, top_p=0.95      │")
print("  │ Factual Q&A              │ temperature=0.3, top_k=20        │")
print("  │ General chat             │ temperature=0.7, top_p=0.9       │")
print("  │ Creative writing         │ temperature=1.1, top_p=0.95      │")
print("  │ Structured (JSON/SQL)    │ temperature=0.0 (greedy)         │")
print("  │ Diverse brainstorming    │ temperature=1.0, min_p=0.05      │")
print("  └──────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · vLLM Offline Inference — Python API Patterns and Best Practices": {
        "description": (
            "Complete guide to vLLM's offline (Python) inference API. "
            "LLM class, SamplingParams, generate() for batched generation. "
            "Show correct batching patterns, LoRA adapter loading, "
            "structured output with guided decoding, "
            "and async LLMEngine for production workloads."
        ),
        "language": "python",
        "code": '''
print("=" * 65)
print("  vLLM OFFLINE API — PATTERNS, LORA, STRUCTURED OUTPUT")
print("=" * 65)
print()
print("  NOTE: This module shows the vLLM Python API patterns.")
print("  Install: pip install vllm")
print("  Requires: NVIDIA GPU + CUDA (or AMD ROCm).")
print("  Code is annotated for study — outputs shown as comments.")
print()

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 1: Basic offline batch generation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 1 — Basic LLM + SamplingParams (offline batch)")
print("━" * 65)
print()

PATTERN_1 = """
from vllm import LLM, SamplingParams

# ── Initialise once ─────────────────────────────────────────────────────
# LLM() loads model weights and allocates KV cache on init.
# Do NOT create LLM() inside a loop — very expensive.

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    dtype="float16",                    # weight precision
    max_model_len=4096,                 # max context window
    gpu_memory_utilization=0.90,        # 90% of GPU VRAM for model+KV
    max_num_seqs=256,                   # max concurrent in-flight seqs
    enable_prefix_caching=True,         # cache shared system prompts
)

# ── SamplingParams ──────────────────────────────────────────────────────
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop=["</s>", "[INST]"],           # stop sequences
    repetition_penalty=1.1,            # reduce repetition
    skip_special_tokens=True,          # clean output
)

# ── Batch generation ────────────────────────────────────────────────────
# vLLM automatically schedules all prompts together as a batch.
# NEVER call generate() in a loop with one prompt at a time.
# Pass ALL prompts at once for maximum throughput.

prompts = [
    "[INST] Write a Python function to reverse a string. [/INST]",
    "[INST] Explain quantum entanglement in simple terms. [/INST]",
    "[INST] What is the capital of France? [/INST]",
    "[INST] Write a haiku about autumn leaves. [/INST]",
]

outputs = llm.generate(prompts, params)

for output in outputs:
    prompt_id = output.request_id
    text      = output.outputs[0].text        # first completion
    tokens    = output.outputs[0].token_ids   # token IDs
    finish    = output.outputs[0].finish_reason  # 'stop' or 'length'
    print(f"Request {prompt_id}: {len(tokens)} tokens, finish={finish}")
    print(f"  {text[:80]}...")
"""

print("  Code:")
print(PATTERN_1)
print()
print("  CRITICAL: Always batch prompts. vLLM schedules them together.")
print("  Calling generate([prompt]) 100 times = 100× slower than generate(prompts_100).")
print()

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 2: Multi-completion sampling (best-of-N)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 2 — Multiple completions (best-of-N, beam search)")
print("━" * 65)
print()

PATTERN_2 = """
# Best-of-N: generate N completions per prompt, return all
# Uses Copy-on-Write for prompt KV sharing

params_best_of_4 = SamplingParams(
    n=4,                    # generate 4 completions
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

outputs = llm.generate(["Write a creative tagline for a coffee shop."],
                        params_best_of_4)

output = outputs[0]
print(f"Generated {len(output.outputs)} completions:")
for i, completion in enumerate(output.outputs):
    print(f"  [{i}] ({len(completion.token_ids)} tokens): {completion.text[:60]}...")

# Beam search (deterministic, maximises probability)
params_beam = SamplingParams(
    use_beam_search=True,
    best_of=4,              # beam width = 4
    temperature=0.0,        # must be 0 for beam search
    max_tokens=128,
)

# Note: beam search is slower than sampling but more coherent for
# short, factual outputs (translation, summarisation)
"""

print("  Code:")
print(PATTERN_2)
print()

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 3: LoRA adapters
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 3 — LoRA adapters (multiple fine-tunes on one base model)")
print("━" * 65)
print()

PATTERN_3 = """
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Load base model with LoRA support enabled
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_lora_rank=64,           # max rank of any loaded adapter
    max_loras=4,                # max adapters loaded in GPU memory simultaneously
    gpu_memory_utilization=0.85,
)

params = SamplingParams(temperature=0.1, max_tokens=256)

# LoRARequest: (name, unique_int_id, path_to_adapter)
sql_lora   = LoRARequest("sql-expert",   1, "./adapters/sql_lora/")
code_lora  = LoRARequest("code-instruct", 2, "./adapters/code_lora/")

# Each request specifies which adapter to use:
outputs = llm.generate(
    prompts=[
        "Convert to SQL: Find all users who signed up after 2023",
        "Write a binary search in Python",
        "What is 2+2?",   # uses base model (no LoRA)
    ],
    sampling_params=params,
    lora_request=[sql_lora, code_lora, None],   # per-request adapter
)

# vLLM batches requests with DIFFERENT adapters in the same forward pass.
# Adapter weights are applied per-request via efficient batched LoRA matmuls.
# Overhead: ~5–10% vs single adapter. Far cheaper than running separate models.
"""

print("  Code:")
print(PATTERN_3)
print()
print("  LoRA economics: 1 base model (7B, 14 GB) + N adapters (~100 MB each)")
print("  vs N separate fine-tuned models (N × 14 GB).")
print("  With 10 adapters: 15 GB total vs 140 GB → 9× memory saving.")
print()

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 4: Guided decoding (structured output)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 4 — Guided decoding (force JSON / regex / grammar output)")
print("━" * 65)
print()

PATTERN_4 = """
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Option A: JSON Schema — force output to match a schema
json_schema = {
    "type": "object",
    "properties": {
        "name":       {"type": "string"},
        "age":        {"type": "integer", "minimum": 0, "maximum": 150},
        "occupation": {"type": "string"},
    },
    "required": ["name", "age", "occupation"]
}

params_json = SamplingParams(
    max_tokens=200,
    guided_decoding=GuidedDecodingParams(json=json_schema)
)

output = llm.generate(
    ["[INST] Extract person info: 'John Smith, 34, is a software engineer.' [/INST]"],
    params_json
)
# Output is GUARANTEED to be valid JSON matching the schema.
# Example: {"name": "John Smith", "age": 34, "occupation": "software engineer"}

# Option B: Regex — constrain output to a pattern
params_regex = SamplingParams(
    max_tokens=50,
    guided_decoding=GuidedDecodingParams(
        regex=r"[A-Z][a-z]+ [A-Z][a-z]+, \\d{2}, [A-Za-z ]+\\."
    )
)

# Option C: Grammar (EBNF) — for complex structured formats
params_grammar = SamplingParams(
    max_tokens=500,
    guided_decoding=GuidedDecodingParams(
        grammar=\\'\\'\\'
        root  ::= object
        object ::= "{" pair ("," pair)* "}"
        pair   ::= string ":" value
        \\'\\'\\'
    )
)
"""

print("  Code:")
print(PATTERN_4)
print()
print("  Guided decoding works by masking logits at each step.")
print("  Only tokens consistent with the current grammar/schema state are allowed.")
print("  Zero accuracy compromise — just constrains the output space.")
print("  Uses outlines library under the hood (automaton-based masking).")
print()

# ─────────────────────────────────────────────────────────────────────────
# PATTERN 5: AsyncLLMEngine for production servers
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  PATTERN 5 — AsyncLLMEngine (production async server)")
print("━" * 65)
print()

PATTERN_5 = """
import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams

# AsyncLLMEngine: non-blocking, streams tokens as generated
# Use this when building your own server (not the built-in OpenAI server)

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-7b-chat-hf",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate_streaming(prompt: str, request_id: str):
    params   = SamplingParams(temperature=0.7, max_tokens=200)
    # generate() returns an async generator of RequestOutput objects
    async for output in engine.generate(prompt, params, request_id):
        token_text = output.outputs[0].text   # cumulative text so far
        if output.finished:
            print(f"DONE [{output.outputs[0].finish_reason}]: {token_text}")
        else:
            # Stream each new token to the client
            yield token_text

# Integration with FastAPI (streaming endpoint):
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
app = FastAPI()

@app.post("/generate")
async def generate(request: dict):
    async def token_stream():
        prompt     = request["prompt"]
        request_id = request.get("id", "req_0")
        prev_text  = ""
        async for text in generate_streaming(prompt, request_id):
            delta = text[len(prev_text):]   # only the NEW tokens
            yield f"data: {delta}\\\\n\\\\n"   # SSE format
            prev_text = text
        yield "data: [DONE]\\\\n\\\\n"
    return StreamingResponse(token_stream(), media_type="text/event-stream")
"""

print("  Code:")
print(PATTERN_5)
print()
print("  AsyncLLMEngine vs LLM class:")
print("    LLM:              blocking, for scripts/notebooks, generates full batch")
print("    AsyncLLMEngine:   non-blocking, for servers, streams token-by-token")
print("    OpenAI server:    wraps AsyncLLMEngine with OpenAI-compatible endpoints")
print()
print("  PERFORMANCE PATTERNS SUMMARY:")
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ DO                              │ DON'T                       │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ Batch all prompts in one call   │ Call generate() per prompt  │")
print("  │ Create LLM() once, reuse        │ Create LLM() per request    │")
print("  │ Set gpu_memory_utilization=0.9  │ Leave at default 0.9 blindly│")
print("  │ Enable prefix_caching=True      │ Leave prefix caching off    │")
print("  │ Use AsyncLLMEngine for servers  │ Use LLM() for serving       │")
print("  │ Use stop sequences              │ Rely on max_tokens alone    │")
print("  └──────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · vLLM OpenAI Server — Deployment, Configuration & Benchmarking": {
        "description": (
            "Complete vLLM server deployment reference. "
            "All key launch flags with explanations. "
            "Benchmark with genai-perf style metrics: TTFT, ITL, throughput. "
            "Show how to tune max-num-seqs and chunked-prefill for different "
            "latency vs throughput tradeoffs."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import json

print("=" * 65)
print("  vLLM SERVER — DEPLOYMENT, CONFIG & BENCHMARKING REFERENCE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Server launch command reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Launch command reference (all key flags)")
print("━" * 65)
print()

LAUNCH_CONFIGS = {
    "Minimal (dev)": {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "port": 8000,
    },
    "Single A100 80GB (production)": {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "dtype": "float16",
        "max-model-len": 4096,
        "gpu-memory-utilization": 0.90,
        "max-num-seqs": 256,
        "enable-prefix-caching": True,
        "enable-chunked-prefill": True,
        "max-num-batched-tokens": 8192,
        "port": 8000,
    },
    "4× A100 80GB TP=4 (LLaMA-2 70B)": {
        "model": "meta-llama/Llama-2-70b-chat-hf",
        "dtype": "bfloat16",
        "tensor-parallel-size": 4,
        "max-model-len": 4096,
        "gpu-memory-utilization": 0.85,
        "max-num-seqs": 128,
        "enable-prefix-caching": True,
        "enable-chunked-prefill": True,
        "port": 8000,
    },
    "H100 FP8 (max throughput)": {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "dtype": "float8",
        "quantization": "fp8",
        "tensor-parallel-size": 4,
        "max-model-len": 8192,
        "gpu-memory-utilization": 0.90,
        "max-num-seqs": 512,
        "enable-prefix-caching": True,
        "enable-chunked-prefill": True,
        "max-num-batched-tokens": 32768,
        "port": 8000,
    },
    "AMD ROCm (GPU-agnostic)": {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "dtype": "float16",
        "device": "rocm",           # vLLM ROCm support
        "max-model-len": 4096,
        "port": 8000,
    },
    "Multi-LoRA serving": {
        "model": "meta-llama/Llama-2-7b-hf",
        "enable-lora": True,
        "max-lora-rank": 64,
        "max-loras": 8,
        "max-num-seqs": 256,
        "port": 8000,
    },
}

for name, flags in LAUNCH_CONFIGS.items():
    print(f"  # {name}")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    flag_lines = [f"      --{k} {v}" for k, v in flags.items()
                  if v is not True]
    bool_flags = [f"      --{k}" for k, v in flags.items() if v is True]
    all_lines  = flag_lines + bool_flags
    for i, line in enumerate(all_lines):
        cont = " \\" if i < len(all_lines) - 1 else ""
        print(f"  {line}{cont}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Key parameter explanations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Key parameters explained")
print("━" * 65)
print()

PARAMS = [
    ("--gpu-memory-utilization",
     "Fraction of GPU VRAM for model+KV cache. 0.9 = 90%.\n"
     "         Remainder for activations, CUDA context, other processes.\n"
     "         Too high (>0.95) → OOM during spikes. Too low → fewer concurrent reqs."),
    ("--max-num-seqs",
     "Max concurrent requests in the scheduler.\n"
     "         Higher = more throughput, more memory pressure.\n"
     "         Set = expected_RPS × (mean_latency_s) × 1.5 (Little's Law)."),
    ("--max-model-len",
     "Maximum context window (input + output tokens).\n"
     "         Longer = more memory per request. Reduce to increase concurrency.\n"
     "         Must be ≤ model's native context length."),
    ("--max-num-batched-tokens",
     "Max total tokens processed per scheduler step.\n"
     "         Controls prefill chunk size. Higher = better GPU utilisation.\n"
     "         Lower = more predictable inter-token latency."),
    ("--enable-chunked-prefill",
     "Break long prefills into chunks, interleaved with decode.\n"
     "         Reduces TTFT spikes when long prompts arrive.\n"
     "         Default: True in vLLM v0.5+."),
    ("--enable-prefix-caching",
     "Cache KV blocks by content hash. Reuse for repeated prefixes.\n"
     "         Near-zero cost when system prompts are shared.\n"
     "         Recommended: always enable for chat/API workloads."),
    ("--quantization",
     "awq / gptq / fp8 / bitsandbytes.\n"
     "         Reduces memory: fp8 on H100, awq on A100 (best quality)."),
    ("--tensor-parallel-size",
     "Number of GPUs to shard model across (within one node).\n"
     "         Must be power of 2. Requires NVLink for best performance.\n"
     "         Set = min GPUs needed to fit model weights + KV."),
]

for flag, desc in PARAMS:
    print(f"  {flag}")
    for line in desc.split('\n'):
        print(f"         {line.strip()}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Benchmarking simulation (realistic metrics)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Benchmark metrics: TTFT, ITL, throughput")
print("━" * 65)
print()
print("  Key LLM serving metrics:")
print()
print("  TTFT (Time To First Token):")
print("    Time from request receipt to first generated token.")
print("    Dominated by prefill time. Perceived as 'responsiveness'.")
print("    Target: < 1s for real-time chat; < 5s for long docs.")
print()
print("  ITL (Inter-Token Latency) / TPOT (Time Per Output Token):")
print("    Time between successive generated tokens.")
print("    Dominated by decode memory bandwidth.")
print("    Target: < 50ms for smooth streaming (20+ tokens/sec).")
print()
print("  Throughput (tokens/sec):")
print("    Total tokens generated per second across all requests.")
print("    The cost-efficiency metric: tokens/$.")
print()

np.random.seed(7)

def simulate_vllm_benchmark(n_requests, concurrency, model_config):
    """Simulate realistic vLLM benchmark numbers."""
    results = []
    prefill_ms_per_tok = model_config['prefill_ms_per_tok']
    decode_ms_per_tok  = model_config['decode_ms_per_tok']
    prefix_cache_hit   = model_config.get('prefix_cache_hit', 0.0)

    for _ in range(n_requests):
        prompt_len  = np.random.randint(64, 512)
        output_len  = np.random.randint(50, 300)

        # TTFT: prefill time (reduced by cache hits)
        effective_prefill = prompt_len * (1 - prefix_cache_hit)
        base_ttft = effective_prefill * prefill_ms_per_tok
        # Queue time: proportional to concurrency
        queue_ms  = np.random.exponential(concurrency * decode_ms_per_tok * 10)
        ttft      = base_ttft + queue_ms + np.random.normal(0, 5)

        # ITL: decode time per token
        # Higher concurrency → slightly higher ITL (memory bandwidth shared)
        load_factor = min(1.0, concurrency / model_config['max_concurrency'])
        itl         = decode_ms_per_tok * (1 + 0.3 * load_factor) + np.random.normal(0, 2)

        total_ms = ttft + itl * output_len
        results.append({
            'ttft': max(10, ttft),
            'itl':  max(5, itl),
            'total_ms': total_ms,
            'output_len': output_len,
        })
    return results

MODELS = {
    "LLaMA-2 7B  (A100 40GB, FP16, no cache)": dict(
        prefill_ms_per_tok=0.3, decode_ms_per_tok=15, max_concurrency=32,
        prefix_cache_hit=0.0
    ),
    "LLaMA-2 7B  (A100 40GB, FP16, prefix cache)": dict(
        prefill_ms_per_tok=0.3, decode_ms_per_tok=15, max_concurrency=32,
        prefix_cache_hit=0.7
    ),
    "LLaMA-2 70B (4×A100 80GB, BF16, TP=4)": dict(
        prefill_ms_per_tok=1.2, decode_ms_per_tok=45, max_concurrency=16,
        prefix_cache_hit=0.0
    ),
    "Mistral 7B  (A100 40GB, AWQ INT4)": dict(
        prefill_ms_per_tok=0.2, decode_ms_per_tok=9, max_concurrency=64,
        prefix_cache_hit=0.0
    ),
}

CONCURRENCY = 8

for model_name, config in MODELS.items():
    results = simulate_vllm_benchmark(200, CONCURRENCY, config)
    ttfts   = [r['ttft'] for r in results]
    itls    = [r['itl']  for r in results]
    toks    = [r['output_len'] / (r['total_ms'] / 1000) for r in results]

    print(f"  {model_name}")
    print(f"    TTFT: P50={np.percentile(ttfts,50):.0f}ms  "
          f"P95={np.percentile(ttfts,95):.0f}ms  "
          f"P99={np.percentile(ttfts,99):.0f}ms")
    print(f"    ITL:  P50={np.percentile(itls,50):.1f}ms  "
          f"P95={np.percentile(itls,95):.1f}ms  "
          f"({1000/np.median(itls):.1f} tokens/sec)")
    print(f"    Throughput: {np.mean(toks):.1f} output tokens/sec/GPU")
    print()

print("  BENCHMARK TOOLS:")
print("  genai-perf (NVIDIA):  pip install genai-perf")
print("    genai-perf -m llama-2-7b --service-kind openai --url localhost:8000")
print()
print("  vllm benchmark_serving.py:")
print("    python benchmarks/benchmark_serving.py \\")
print("        --backend vllm --model meta-llama/... \\")
print("        --num-prompts 500 --request-rate 10")
print()
print("  locust (load testing):")
print("    locust -f locustfile_openai.py --users 50 --spawn-rate 5")
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
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }