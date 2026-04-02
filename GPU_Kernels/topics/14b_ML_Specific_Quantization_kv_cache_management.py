"""
KV Cache Management — Paged Attention, Continuous Batching & Prefix Caching
============================================================================

The key-value (KV) cache is the central data structure of LLM inference.
During the prefill phase, every transformer layer computes key and value
vectors for every input token and stores them. During the decode phase,
each new token attends over all previously stored KV vectors — reading
the entire cache every step. As sequences grow longer and more requests
arrive concurrently, the KV cache becomes the primary bottleneck for
both memory capacity and access bandwidth.

Three systems-level innovations transformed LLM serving efficiency:

    PAGED ATTENTION (vLLM, 2023):
        Borrows virtual memory paging from operating systems. KV cache is
        divided into fixed-size "pages" (blocks of 16 tokens each). A
        block table maps virtual page numbers to physical GPU memory blocks.
        Pages are allocated on demand and freed when a sequence ends.
        Fragmentation drops from ~60% (pre-vLLM) to <5%, enabling 2–4×
        more concurrent requests on the same hardware.

    CONTINUOUS BATCHING (Orca, 2022):
        Static batching waits for all sequences in a batch to finish before
        starting the next batch. Continuous batching inserts new requests
        mid-batch as slots become available. The GPU never waits. Throughput
        increases 5–23× vs static batching for real workload distributions
        where sequence lengths vary widely.

    PREFIX CACHING / RADIXATTENTION (SGLang, 2023):
        Many requests share a common prefix — a system prompt, a few-shot
        example set, or a shared document. Computing the KV cache for the
        shared prefix on every new request wastes compute and bandwidth.
        RadixAttention stores the prefix KV cache in a radix tree keyed by
        token sequence hash. New requests with matching prefixes reuse the
        cached KV vectors. Time-to-first-token drops by 5–100× for matched
        prefixes.

Together these three techniques form the memory management substrate of
every production LLM serving stack: vLLM, TensorRT-LLM, SGLang, LMDeploy,
and Triton Inference Server all implement variants of these ideas.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "KV Cache Management — Paged Attention, Continuous Batching & Prefix Caching"
DISPLAY_NAME = "14b · KV Cache Management"
ICON         = "📦"
SUBTITLE     = "Paged Attention · vLLM Block Manager · Continuous Batching · RadixAttention"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE KV CACHE: WHAT IT STORES AND WHY IT DOMINATES INFERENCE

### What the KV Cache Contains

    In a transformer with L layers, H attention heads, and head dimension d:
    For a sequence of N tokens, the KV cache stores:
        K[layer, head, token, :] ∈ ℝ^d    (key vectors)
        V[layer, head, token, :] ∈ ℝ^d    (value vectors)

    TOTAL KV CACHE SIZE per sequence:
        2 × L × H × N × d × sizeof(dtype)

    For Llama-2-7B (L=32, H=32, d=128) in FP16:
        2 × 32 × 32 × N × 128 × 2 bytes = 524,288 × N bytes = 0.5 MB per token
        For N=2048: 1 GB per sequence.
        For N=4096: 2 GB per sequence.
        For 10 concurrent sequences of 4096 tokens: 20 GB — most of H100 VRAM!

    For Llama-2-70B (L=80, H=64, d=128) in FP16:
        2 × 80 × 64 × N × 128 × 2 bytes = 3.28 MB per token
        For N=4096: 13.4 GB per sequence — exceeds a single H100 for one request!
        (This is why Grouped Query Attention reduces H from 64 to 8, saving 8×)

### The KV Cache Bandwidth Problem

    During decode, each new token must ATTEND over all previous tokens:
        For each layer: output = softmax(q @ K_cache.T / √d) @ V_cache

    HBM READS per decode step:
        KV cache: 2 × L × H × N × d × 2 bytes = 0.5 MB × N (for 7B)
        Weight matrices: ~14 GB (all transformer parameters)
        Total: 14 GB + 0.5 MB × N

    For N=8192: 14 GB + 4 GB = 18 GB per token.
    At H100 bandwidth (3.35 TB/s): ~5.4 ms per token.
    KV cache access is 22% of decode bandwidth at N=8192.
    At N=32768: KV cache = 16 GB, weights = 14 GB → KV cache DOMINATES.

### GQA/MQA: Reducing KV Cache Size

    Multi-Head Attention (MHA): H_Q = H_KV = 32 heads.
    Multi-Query Attention (MQA): H_KV = 1 (all queries share one K, V head).
    Grouped-Query Attention (GQA): H_KV = H_Q / G (G queries per KV group).

    For Llama-3 70B (H_Q=64, H_KV=8, G=8):
        KV cache: 2 × L × H_KV × N × d × 2 bytes = 2 × 80 × 8 × N × 128 × 2
                = 0.41 MB per token (vs 3.28 MB for full MHA)
        8× KV cache reduction vs MHA.
        For N=4096: 1.67 GB per sequence (vs 13.4 GB for MHA).

    GQA/MQA is now standard in every frontier LLM:
        Llama-3 (GQA G=8), Mistral (GQA G=4), Falcon (MQA), Gemma (MQA).


##### PART 2 — PAGED ATTENTION: VIRTUAL MEMORY FOR KV CACHES

### The Pre-vLLM Memory Waste Problem

    Before paged attention, KV caches were allocated as CONTIGUOUS REGIONS:
        seq_A: allocated [0 .. 2047]  (2048 tokens reserved)
        seq_B: allocated [2048 .. 4095]  (2048 tokens reserved)
        ...

    Two types of waste:
        INTERNAL FRAGMENTATION: seq_A is 312 tokens long but was allocated
        2048 slots. The remaining 1736 are wasted until the sequence ends.

        EXTERNAL FRAGMENTATION: freed regions are not contiguous with free
        regions that were freed later. Allocator cannot satisfy a large
        contiguous request even though total free memory is sufficient.

    RESERVATION WASTE: operators must pre-allocate for the maximum possible
    sequence length (to guarantee space is available during generation).
    For max_len=2048, average actual len=512: 75% waste.

    Measured waste in production before vLLM: 60–80% of KV cache memory
    was wasted due to fragmentation and over-reservation.

### The Paged Attention Solution

    INSPIRED BY OPERATING SYSTEM VIRTUAL MEMORY:
        Physical RAM is divided into fixed-size pages (4 KB on most OS).
        Each process has a virtual address space mapped to physical pages.
        The OS page table maps virtual page numbers to physical page numbers.
        Pages are allocated on demand; freed pages reused by other processes.

    PAGED ATTENTION APPLIES THE SAME IDEA TO KV CACHES:
        Physical GPU memory divided into fixed-size KV blocks (e.g., 16 tokens).
        Each sequence has a virtual sequence of blocks (virtual pages).
        A per-sequence block table maps virtual block numbers to physical blocks.
        Physical blocks are allocated on demand as the sequence grows.

    BLOCK SIZE CHOICE:
        16 tokens per block is the standard (used by vLLM default).
        Smaller (e.g., 8): less internal fragmentation but more table overhead.
        Larger (e.g., 32): less overhead but more waste for short sequences.
        Flash Attention works naturally with block granularity — a KV block
        = one tile in the attention computation.

### Physical Block Layout

    Each physical block stores one KV BLOCK for all layers and all KV heads:

        block_size × L × H_KV × d_head × 2 × sizeof(FP16) bytes

    For Llama-2-7B (L=32, H_KV=32, d=128), block_size=16:
        16 × 32 × 32 × 128 × 2 × 2 = 8,388,608 bytes = 8 MB per physical block

    For H100 (80 GB VRAM):
        Available for KV cache (after model weights ~14 GB and overhead):
        ~60 GB → 60 GB / 8 MB = 7,500 physical blocks
        → 7,500 × 16 = 120,000 token slots in the KV cache pool.

    With average sequence length 1024:
        Concurrent sequences = 120,000 / 1024 ≈ 117 requests simultaneously.
        Before paged attention (60% waste): 117 × 0.4 ≈ 47 requests.
        Paged attention: 2.5× more concurrent requests.

### Virtual → Physical Block Mapping

    BLOCK TABLE per sequence: an array indexed by virtual block number,
    each entry contains the physical block number.

        seq.block_table[virtual_block] = physical_block_number

    ATTENTION COMPUTATION with paged attention:
        For query token at position t:
            virtual_block  = t // block_size
            block_offset   = t % block_size
            physical_block = seq.block_table[virtual_block]
            K_t = K_cache[physical_block, :, block_offset, :]  (all layers, all KV heads)

        The kernel dereferences the block table for each KV tile access.
        This adds one level of indirection vs contiguous KV caches — a small
        overhead (~2–5% vs contiguous) that is worth the 60% memory saving.

### Block Reference Counting and Copy-on-Write

    Blocks can be SHARED between sequences:
        Prefix blocks (same token sequence prefix) can be physically shared.
        The block reference count is incremented when shared.

    COPY-ON-WRITE (COW):
        When a sequence needs to WRITE to a block (append new token),
        and the block is shared (ref_count > 1):
        1. Allocate a new physical block.
        2. Copy the existing block content to the new block.
        3. Update the sequence's block table to point to the new block.
        4. Decrement the old block's reference count.
        This is the basis for prefix caching (Part 4).


##### PART 3 — THE vLLM BLOCK MANAGER: ALLOCATION, EVICTION & PREEMPTION

### Block Manager State

    The vLLM block manager maintains three categories of physical blocks:

    FREE BLOCKS:
        Not currently allocated to any sequence.
        Tracked in a free list (a deque for fast append/pop).
        After model startup: all blocks are free.

    ALLOCATED BLOCKS:
        Currently holding KV data for at least one active sequence.
        Each block has a reference count (>0).
        ref_count > 1: block is shared (prefix caching).

    EVICTED BLOCKS (CPU swap):
        Blocks moved to CPU memory to make room for higher-priority requests.
        Tracked per sequence (all blocks of an evicted sequence go to CPU).
        Can be swapped back to GPU when the sequence resumes.

### Block Allocation Algorithm

    When a sequence generates a new token:
        1. Determine if the last block is full (block_offset == block_size - 1).
        2. If full: allocate a new physical block from the free list.
           Append the new physical block number to seq.block_table.
        3. Write the new token's K and V to:
              K_cache[physical_block, layer, head, block_offset, :] = k_new
              V_cache[physical_block, layer, head, block_offset, :] = v_new

    FULL FREE LIST: no blocks available.
        POLICY 1 — Preemption: pause the lowest-priority active sequence.
            Free all its blocks. Resume it later (with recomputation or swap).
        POLICY 2 — Swap: move the lowest-priority sequence's blocks to CPU.
            Synchronous swap: add ~1 ms latency per sequence swap.
            Asynchronous swap: overlap swap with next decode step.

### Preemption vs Swap: When to Use Each

    PREEMPTION (recompute):
        Free blocks immediately (no CPU memory used).
        Penalty: recompute the sequence's KV cache when it resumes.
        For short sequences (N < 512): recomputation is fast (~50 ms).
        For long sequences (N > 2048): recomputation is expensive.
        USE WHEN: CPU memory is also scarce; sequence is short.

    SWAP (CPU offload):
        Copy blocks to CPU via PCIe. PCIe Gen4: 32 GB/s.
        For a 1024-token sequence (0.5 GB of KV cache):
            Swap-out time: 0.5 GB / 32 GB/s = 15.6 ms.
        Swap-in time: same 15.6 ms.
        USE WHEN: sequence is long, CPU memory is available.

### Sequence Group: Beam Search and Parallel Sampling

    vLLM supports generating MULTIPLE sequences from one request:
        BEAM SEARCH: k sequences per request, pruned by probability.
        PARALLEL SAMPLING: k independent samples from the same prompt.

    BLOCK SHARING:
        All k sequences in a group share the SAME prefix blocks (the input prompt).
        When divergence occurs (different tokens generated), COW triggers.
        Before divergence: ref_count = k for all shared blocks.
        After first divergence: only the diverged block is COW-copied.

    MEMORY EFFICIENCY:
        Beam search k=4 with 2048 prompt tokens:
        Without sharing: 4 × (2048/16) = 512 blocks for the prompt.
        With sharing:    1 × (2048/16) = 128 blocks shared + 4 COW copies.
        Savings: 75% for the prompt portion.


##### PART 4 — CONTINUOUS BATCHING: DYNAMIC REQUEST SCHEDULING

### Static Batching Inefficiency

    STATIC BATCHING groups requests into a batch, runs all to completion,
    then picks the next batch.

    PROBLEM: sequence lengths vary widely in LLM workloads.
        A batch of [seq_len = 50, 2000, 100, 1800] must wait for the
        longest sequence (2000) to complete before any slot frees.
        Sequences of length 50 and 100 finish early but their GPU slots
        (and KV cache memory) remain occupied and IDLE for the rest.

    GPU UTILISATION with static batching:
        Batch has 4 sequences: [50, 2000, 100, 1800] tokens.
        After 50 steps: 75% utilisation (3/4 sequences still running).
        After 100 steps: 50% utilisation.
        After 1800 steps: 25% utilisation.
        Average GPU utilisation: ~75% of potential.
        Effective throughput: 75% × peak.

### Continuous Batching (Iteration-Level Scheduling)

    KEY INSIGHT (Orca paper, 2022): make the scheduling decision AT EVERY
    STEP (iteration), not at every batch boundary.

    ALGORITHM:
        At each decode step:
            1. Run one decode step for all ACTIVE sequences.
            2. Check for sequences that COMPLETED in this step (EOS token generated).
            3. For each completed sequence: FREE its KV blocks, remove from batch.
            4. ADMIT new waiting requests to fill the vacated slots.
            5. For newly admitted sequences: run PREFILL (can be batched separately).
            6. Proceed to next decode step.

    EFFECT: the GPU is always running at maximum batch capacity.
    No idle slots — every slot either produces a useful token or
    admits a new request immediately.

### The Chunked Prefill Optimisation

    PROBLEM WITH NAIVE CONTINUOUS BATCHING:
        Prefill (processing a new sequence's full prompt) is compute-intensive.
        Running prefill for a 1024-token prompt takes ~50× longer than one decode step.
        During prefill: the entire batch stalls (decode latency spikes for existing sequences).

    CHUNKED PREFILL (Sarathi-Serve, 2023):
        Split the prefill of a new sequence into CHUNKS of C tokens each.
        Interleave one prefill chunk with each decode step.
        Effect: prefill is spread over C steps, each step stays short.
        New sequence takes N/C more steps to complete prefill, but
        existing decode sequences see no latency spike.

    CHUNK SIZE TRADEOFF:
        Large C (e.g., 512): faster prefill completion, longer decode step.
        Small C (e.g., 64): slower prefill, shorter decode step (lower P95 latency).
        Typical production value: C = 256 or C = 512 tokens.

### Sequence Preemption Policy

    When the KV cache is full and a new high-priority request arrives,
    one or more existing requests must be preempted.

    PRIORITY-BASED PREEMPTION:
        Priority metrics: request arrival time (FIFO), expected completion
        time, SLA tier, sequence length.
        Lowest-priority sequence is paused, its blocks freed.

    RECOMPUTE vs SWAP DECISION:
        For preempted sequence of length N:
            swap_cost = N × kv_bytes_per_token / PCIe_bandwidth
            recompute_cost = N × prefill_time_per_token
        Choose: min(swap_cost, recompute_cost).

    BLOCKING vs NON-BLOCKING:
        Blocking preemption: decode step stops, swap executes, then resumes.
        Non-blocking (async): swap happens in parallel with the decode step
        using CUDA streams — effectively free if overlap is good.

### Token Budget and Scheduling Constraints

    Modern serving systems (vLLM v0.4+, SGLang, TGI) schedule with:

    MAX_NUM_SEQS: maximum concurrent sequences (memory capacity).
    MAX_NUM_BATCHED_TOKENS: maximum total tokens per step (compute capacity).
        = sum of all sequence lengths in the current batch.
        Prevents one very long sequence from consuming the entire compute budget.

    MAX_MODEL_LEN: maximum sequence length the model supports.
        Determines how many KV blocks to reserve per sequence.

    SCHEDULING LOOP:
        1. Sort waiting requests by priority.
        2. While (active_seqs < MAX_NUM_SEQS AND
                 total_tokens + len(new_seq) <= MAX_NUM_BATCHED_TOKENS):
               admit new_seq from waiting queue.
        3. Preempt if KV cache blocks < needed.
        4. Run decode step.


##### PART 5 — PREFIX CACHING AND RADIXATTENTION

### The Repeated Prefix Problem

    Many LLM workloads exhibit SHARED PREFIXES across requests:

    SYSTEM PROMPT SHARING:
        All requests in a chat application begin with the same system prompt.
        For a 1024-token system prompt, every new request recomputes 1024
        tokens of prefill from scratch — identical work every time.

    MULTI-TURN CONVERSATION:
        Each turn in a conversation shares all previous turns as prefix.
        Turn 5 computes: [sys_prompt] + [turn1] + [turn2] + [turn3] + [turn4] + [turn5]
        Turn 4's KV cache covers all but the last turn — reuse it!

    FEW-SHOT EXAMPLES:
        API calls often include the same set of examples in every request.
        100 tokens of examples × 10,000 requests/day = 1B wasted tokens/day.

    DOCUMENT QA / RAG:
        Same 32K-token document embedded in multiple queries.
        Without prefix cache: each query reprocesses the full document.
        With prefix cache: amortise prefill cost across all queries over that document.

### RadixAttention: KV Cache as a Radix Tree

    SGLang (Zheng et al., 2023) introduced RADIXATTENTION:
        Organises the KV cache as a RADIX TREE (compressed trie) where:
            - Each edge represents a sequence of tokens.
            - Each node stores the KV blocks for its token sequence.
            - New requests find the LONGEST MATCHING PREFIX in the tree.
            - KV blocks for the matched prefix are REUSED without recomputation.

    RADIX TREE OPERATIONS:
        INSERT(token_sequence, kv_blocks):
            Walk the tree. At the first mismatch, create a new branch.
            Share all blocks up to the mismatch point.

        LOOKUP(token_sequence) → longest matching prefix:
            Walk the tree along the given token sequence.
            Return the last matched node and the matching prefix length.
            The corresponding KV blocks can be directly reused.

        EVICTION(when cache full):
            LRU eviction at the leaf level — leaf nodes that were accessed
            least recently are evicted first, freeing their KV blocks.

### The Radix Tree Cache Lookup Algorithm

    For a new request with tokens [t0, t1, t2, ..., tN-1]:
        1. Start at root.
        2. For each token: follow the edge labeled t_i.
        3. Stop when no matching edge exists (or at a leaf).
        4. Matched prefix length = depth of reached node.
        5. Matched blocks: all KV blocks on the path from root to reached node.

    PREFILL AMORTISATION:
        If matched prefix length = M:
            Prefill cost for new request = (N - M) tokens instead of N.
            Savings: M/N fraction of prefill compute.
            For M=1024, N=1100: 93% prefill savings.

    BLOCK-GRANULAR MATCHING:
        Matching happens at BLOCK GRANULARITY (16-token blocks).
        A prefix matches only if the last matched block is COMPLETE.
        Incomplete blocks cannot be shared (the sequence might still grow).
        This is why vLLM uses block_size=16: smaller blocks → finer matching.

### Cache Eviction Policy: LRU with Reference Counting

    EVICTION TRIGGER: free block count falls below threshold.

    LRU EVICTION RULE for radix tree:
        Only LEAF NODES can be evicted (non-leaves are still needed as prefixes
        for other active requests).
        Among leaf nodes: evict the node with the oldest last_access_time.
        Release all its KV blocks to the free list.

    REFERENCE COUNTING:
        Nodes in the tree have ref_count.
        Active sequences hold a reference to their blocks.
        A node can only be evicted when ref_count == 0 (no active sequence
        is currently using its blocks).

### Prefix Cache Performance Impact

    CACHE HIT RATE depends on workload characteristics:
        TTFT (Time-to-First-Token) with prefix cache hit:
            TTFT ≈ (N - M) × prefill_time_per_token + 1 decode step
            vs standard: N × prefill_time_per_token
        For M=N-1 (one new token): TTFT drops from N×prefill to ~1×decode.

    TYPICAL IMPROVEMENTS (measured on production workloads):
        Chat with system prompt (1024 tokens): 5–10× TTFT reduction.
        Document QA with 32K context: 50–100× TTFT reduction for 2nd query.
        Code generation with repo context: 3–20× depending on cache hit rate.
        API multi-turn conversation: 2–5× TTFT per turn (all but latest turn is cached).

    OVERHEAD:
        Hash computation: SHA-256 or FNV hash of the token sequence.
        CPU lookup time: O(M/block_size) per request — microseconds.
        Block allocation: O(1) per block reused.
        Negligible vs the prefill compute saved.


##### PART 6 — PAGED ATTENTION CUDA KERNEL: HOW ATTENTION READS PAGED BLOCKS

### The Attention Kernel with Paged KV

    Standard Flash Attention assumes CONTIGUOUS K and V tensors.
    Paged attention requires a non-contiguous access pattern driven by the block table.

    PAGED ATTENTION KERNEL STRUCTURE:
        Grid: (N_seqs, N_heads)  — one block per (sequence, head)
        Each block handles the full attention for one query token in one head.

        for k_block_idx in range(len(block_table)):
            physical_block = block_table[k_block_idx]   // pointer dereference
            K_block = K_cache[physical_block, head, :, :]  // (block_size, d)
            V_block = V_cache[physical_block, head, :, :]  // (block_size, d)

            // Compute attention scores for this KV block
            S_block = q @ K_block.T * scale
            // Update running (m, d, O) accumulators (Flash Attention style)

    THE INDIRECTION COST:
        Each iteration loads block_table[k_block_idx] from global memory.
        This is a SCATTER access — random physical block numbers.
        Cache misses at the block table level: each new k_block_idx may miss.
        Mitigation: block tables are small (N/16 entries) → likely L1/L2 resident.
        Measured overhead vs contiguous Flash Attention: ~5% slower.

### CUDA Thread-Block Assignments

    vLLM's PagedAttention kernel (version 2, used for long contexts):
        Grid: (N_seqs, N_heads, N_KV_blocks / PARTITION_SIZE)
        Multiple thread blocks handle different partitions of the KV sequence.
        A reduction kernel merges the partial softmax statistics.
        This is essentially Flash Decoding within the paged attention framework.

    PARTITION SIZE = 512 KV tokens per thread block.
    For N=8192: 16 partitions per (seq, head) → 16× more parallelism than V1.

### Async Prefetch with CUDA Streams

    For long-context decode (N > 32K):
        Bottleneck: reading the block table from global memory before
        loading each KV block.

    ASYNC PREFETCH PATTERN:
        Partition the block table into chunks.
        Issue async load for chunk i+1 while computing attention on chunk i.
        Block table loading latency is hidden by compute.
        cp.async (Ampere+) handles the async load to SMEM.


##### PART 7 — KV CACHE QUANTIZATION AND COMPRESSION

### Why Quantize the KV Cache

    KV cache memory is the second-largest memory consumer (after weights).
    Quantizing KV cache from FP16 to INT8 or FP8 halves the memory:
        More concurrent sequences per GPU.
        Longer maximum context length.
        Less HBM bandwidth per decode step.

    QUALITY CONCERN: unlike weights, KV cache is computed at runtime.
    Quantization error in K or V vectors affects attention output quality.
    Empirically: INT8 KV cache has < 0.1 perplexity degradation for most models.
    FP8 KV cache: even smaller degradation (8-bit float has more range than INT8).

### INT8 KV Cache Quantization

    SYMMETRIC PER-HEAD QUANTIZATION:
        scale_k = max(|K[head, :]|) / 127
        K_int8  = round(K / scale_k)

    SYMMETRIC PER-TOKEN:
        More accurate but doubles scale storage.
        scale_k[token] = max(|K[token, :]|) / 127
        One scale per (layer, head, token) → adds N × 2 × L × H × 4 bytes.
        For N=8192, L=32, H=32: 4 × L × H × N × 4 bytes = 134 MB (vs 2 GB KV cache).

    DEQUANTIZATION AT ATTENTION TIME:
        For each K block retrieved from the paged cache:
            K_fp16 = K_int8.astype(float) × scale_k  (one multiply per element)
        Fused into the paged attention kernel as an epilogue.
        Overhead: ~5% increase in attention kernel time for INT8 dequant.

### FP8 KV Cache (H100 Native)

    H100 supports FP8 E4M3/E5M2 natively in wgmma.
    FP8 KV cache enables true FP8 attention computation (no dequant needed).
    Memory: 1 byte per KV value vs 2 bytes for FP16 → 2× capacity.
    TensorRT-LLM and TransformerEngine support FP8 KV cache on H100.

### KV Cache Compression Beyond Quantization

    H2O (Heavy Hitter Oracle, 2023):
        Keep only the top-k "heavy hitter" tokens in the KV cache per head.
        Heavy hitters = tokens that accumulated the most attention weight historically.
        Evict the KV pairs of non-heavy-hitter tokens.
        Reduces KV cache to O(k × L × H × d) regardless of sequence length.
        Works because attention weight is concentrated: 20% of tokens receive 80% of weight.

    SCISSORHANDS (2023):
        Tokens that received high attention in early layers remain important in later layers.
        Evict tokens based on cumulative attention scores across layers.
        O(context_length) KV → O(budget) KV at inference time.

    SNAPKV (2024):
        Compresses the KV cache by selecting important tokens per attention head.
        Each head selects different important tokens → better coverage than H2O.
        Applied during prefill: reduces the stored KV before the decode phase begins.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · KV Cache Memory Model — Size, Bandwidth & GQA Tradeoffs": {
        "description": (
            "Compute the exact KV cache memory footprint for major LLMs "
            "(Llama-2-7B, 13B, 70B; Llama-3-8B, 70B with GQA) across context "
            "lengths from 512 to 128K. Show the KV cache as a fraction of total "
            "VRAM. Compute per-token HBM bandwidth cost for decode. Demonstrate "
            "how GQA/MQA reduces KV cache by 4–8×. Show the concurrent request "
            "capacity at different context lengths."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  KV CACHE MEMORY MODEL — Size, Bandwidth & GQA Tradeoffs")
print("=" * 68)
print()

FP16_BYTES  = 2
H100_HBM_GB = 80.0
H100_BW_GBS = 3350.0

MODELS = {
    "Llama-2-7B":   {"layers":32,"h_q":32,"h_kv":32,"d_head":128,"params":7e9},
    "Llama-2-13B":  {"layers":40,"h_q":40,"h_kv":40,"d_head":128,"params":13e9},
    "Llama-2-70B":  {"layers":80,"h_q":64,"h_kv":64,"d_head":128,"params":70e9},
    "Llama-3-8B":   {"layers":32,"h_q":32,"h_kv":8, "d_head":128,"params":8e9},
    "Llama-3-70B":  {"layers":80,"h_q":64,"h_kv":8, "d_head":128,"params":70e9},
    "Mistral-7B":   {"layers":32,"h_q":32,"h_kv":8, "d_head":128,"params":7.2e9},
}


def kv_bytes_per_token(m):
    """KV cache bytes per token for one sequence."""
    return 2 * m["layers"] * m["h_kv"] * m["d_head"] * FP16_BYTES

def weight_bytes(m):
    return m["params"] * FP16_BYTES

def available_kv_gb(m, overhead_gb=2.0):
    """Available VRAM for KV cache (after weights + overhead)."""
    return H100_HBM_GB - weight_bytes(m)/1e9 - overhead_gb


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: KV cache bytes per token per model
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — KV Cache Bytes per Token: Models Compared")
print("━" * 68)
print()

print(f"  {'Model':<16}  {'L':>4}  {'H_Q':>5}  {'H_KV':>6}  "
      f"{'KV per token':>14}  {'GQA ratio':>10}  {'Weights GB':>11}")
print("  " + "─" * 68)

mha_kv = {}
for name, m in MODELS.items():
    kv_tok = kv_bytes_per_token(m)
    w_gb   = weight_bytes(m) / 1e9
    gqa_ratio = m["h_q"] // m["h_kv"]
    mha_kv[name] = kv_tok if m["h_q"] == m["h_kv"] else None
    gqa_str = f"MHA (1×)" if gqa_ratio == 1 else f"GQA ({gqa_ratio}×)"
    print(f"  {name:<16}  {m['layers']:>4}  {m['h_q']:>5}  {m['h_kv']:>6}  "
          f"{kv_tok/1024:.2f} KB/tok  {gqa_str:>10}  {w_gb:>10.1f}")

print()
print("  GQA ratio = H_Q / H_KV = how many Q heads share each KV head.")
print("  Llama-3-70B: H_KV=8 reduces KV cache 8× vs H_KV=64 (same MHA parameters).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: KV cache size vs context length
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — KV Cache Size vs Context Length")
print("━" * 68)
print()

CTX_LENGTHS  = [512, 1024, 2048, 4096, 8192, 32768, 131072]
SHOW_MODELS  = ["Llama-2-7B", "Llama-3-70B"]

for model_name in SHOW_MODELS:
    m = MODELS[model_name]
    kv_tok = kv_bytes_per_token(m)
    avail  = available_kv_gb(m)
    print(f"  {model_name}  (KV: {kv_tok/1024:.2f} KB/tok, "
          f"avail: {avail:.1f} GB after weights)")
    print()
    print(f"  {'Context':>10}  {'KV cache (GB)':>15}  {'% of avail VRAM':>17}  "
          f"{'Max concurrent seqs':>20}  {'Decode BW (GB/s)'}")
    print("  " + "─" * 70)

    for n_ctx in CTX_LENGTHS:
        kv_gb     = kv_tok * n_ctx / 1e9
        pct_avail = kv_gb / avail * 100
        max_seqs  = max(1, int(avail * 1e9 / (kv_tok * n_ctx)))
        # Decode BW: per step, read all KV (one per seq in batch), plus weights
        decode_bw = (kv_tok * n_ctx * max_seqs + weight_bytes(m)) / 1e9  # GB per step
        decode_us = decode_bw / H100_BW_GBS * 1000  # ms
        print(f"  {n_ctx:>10,}  {kv_gb:>15.3f}  {pct_avail:>16.1f}%  "
              f"{max_seqs:>20}  {decode_bw:>8.1f} GB ({decode_us:.1f} ms/tok)")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: KV cache vs weights in bandwidth breakdown
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Decode Bandwidth: KV Cache vs Weights")
print("━" * 68)
print()

model_name = "Llama-3-70B"
m = MODELS[model_name]
kv_tok = kv_bytes_per_token(m)
w_gb   = weight_bytes(m) / 1e9
batch  = 1

print(f"  Model: {model_name}, batch={batch}, FP16")
print()
print(f"  {'Context':>10}  {'Weights GB':>12}  {'KV cache GB':>13}  "
      f"{'KV fraction':>13}  {'Total GB':>10}  {'ms/token'}")
print("  " + "─" * 66)

for n_ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    kv_gb     = kv_tok * n_ctx * batch / 1e9
    total_gb  = w_gb + kv_gb
    kv_frac   = kv_gb / total_gb * 100
    t_ms      = total_gb / H100_BW_GBS * 1000
    print(f"  {n_ctx:>10,}  {w_gb:>12.2f}  {kv_gb:>13.3f}  "
          f"{kv_frac:>12.1f}%  {total_gb:>10.2f}  {t_ms:>8.2f}")

print()
print("  At N>8192: KV cache bandwidth starts to dominate over model weights.")
print("  This is why KV quantization (INT8/FP8) is critical for long-context serving.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · vLLM Block Manager — Allocation, Fragmentation & Block Table": {
        "description": (
            "Implement a simplified vLLM-style block manager with paged KV "
            "allocation. Simulate free/allocated block tracking, per-sequence "
            "block tables, and virtual-to-physical address translation. Show "
            "the fragmentation reduction vs contiguous allocation. Simulate "
            "block eviction under memory pressure. Verify that the block table "
            "correctly tracks multi-sequence allocation with different lengths."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque

print("=" * 68)
print("  vLLM BLOCK MANAGER — Allocation, Fragmentation & Block Table")
print("=" * 68)
print()

np.random.seed(42)

# KV cache configuration (Llama-2-7B style)
BLOCK_SIZE   = 16       # tokens per KV block
LAYERS       = 32
H_KV         = 32
D_HEAD       = 128
FP16_BYTES   = 2

BYTES_PER_BLOCK = BLOCK_SIZE * LAYERS * H_KV * D_HEAD * 2 * FP16_BYTES   # K+V

# Total GPU VRAM available for KV cache
KV_CACHE_VRAM_GB = 40.0
N_PHYSICAL_BLOCKS = int(KV_CACHE_VRAM_GB * 1e9 / BYTES_PER_BLOCK)


@dataclass
class PhysicalBlock:
    block_id:    int
    ref_count:   int = 0   # 0 = free, >0 = allocated

    @property
    def is_free(self):
        return self.ref_count == 0


@dataclass
class Sequence:
    seq_id:       int
    prompt_len:   int
    max_len:      int
    block_table:  List[int] = field(default_factory=list)   # virtual → physical block id
    n_tokens:     int = 0

    @property
    def n_blocks_used(self):
        if self.n_tokens == 0:
            return 0
        return math.ceil(self.n_tokens / BLOCK_SIZE)

    @property
    def last_block_offset(self):
        return (self.n_tokens - 1) % BLOCK_SIZE if self.n_tokens > 0 else -1

    @property
    def last_block_full(self):
        return self.last_block_offset == BLOCK_SIZE - 1


class BlockManager:
    def __init__(self, n_blocks: int):
        self.n_blocks  = n_blocks
        self.blocks    = [PhysicalBlock(i) for i in range(n_blocks)]
        self.free_list = deque(range(n_blocks))   # deque for fast pop/append
        self.sequences: Dict[int, Sequence] = {}

    def n_free(self):
        return len(self.free_list)

    def n_allocated(self):
        return self.n_blocks - self.n_free()

    def alloc_block(self) -> Optional[int]:
        """Allocate one physical block. Returns block_id or None if OOM."""
        if not self.free_list:
            return None
        block_id = self.free_list.popleft()
        self.blocks[block_id].ref_count = 1
        return block_id

    def free_block(self, block_id: int):
        """Decrement ref count; return to free list if zero."""
        b = self.blocks[block_id]
        b.ref_count -= 1
        if b.ref_count == 0:
            self.free_list.append(block_id)

    def admit_sequence(self, seq: Sequence) -> bool:
        """Admit a new sequence. Allocate blocks for prompt tokens."""
        blocks_needed = math.ceil(seq.prompt_len / BLOCK_SIZE)
        if self.n_free() < blocks_needed:
            return False   # OOM

        self.sequences[seq.seq_id] = seq
        seq.n_tokens = seq.prompt_len

        for _ in range(blocks_needed):
            block_id = self.alloc_block()
            seq.block_table.append(block_id)

        return True

    def append_token(self, seq_id: int) -> bool:
        """Append one new token to a sequence. Allocate block if needed."""
        seq = self.sequences[seq_id]

        # Need a new block?
        if seq.last_block_full or seq.n_tokens == 0:
            if self.n_free() == 0:
                return False   # OOM
            block_id = self.alloc_block()
            seq.block_table.append(block_id)

        seq.n_tokens += 1
        return True

    def free_sequence(self, seq_id: int):
        """Free all blocks held by a completed sequence."""
        seq = self.sequences.pop(seq_id)
        for block_id in seq.block_table:
            self.free_block(block_id)

    def virtual_to_physical(self, seq_id: int, token_pos: int):
        """Translate token position to (physical_block, block_offset)."""
        seq            = self.sequences[seq_id]
        virtual_block  = token_pos // BLOCK_SIZE
        block_offset   = token_pos % BLOCK_SIZE
        physical_block = seq.block_table[virtual_block]
        return physical_block, block_offset

    def utilisation(self):
        """Fraction of physical blocks in use."""
        return self.n_allocated() / self.n_blocks


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Block manager setup and fragmentation comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Block Manager Setup & Memory Efficiency")
print("━" * 68)
print()

print(f"  KV cache pool: {N_PHYSICAL_BLOCKS:,} blocks × {BYTES_PER_BLOCK/1e6:.1f} MB each "
      f"= {N_PHYSICAL_BLOCKS * BYTES_PER_BLOCK / 1e9:.1f} GB")
print(f"  Block size: {BLOCK_SIZE} tokens × {LAYERS} layers × {H_KV} KV heads")
print()

bm = BlockManager(N_PHYSICAL_BLOCKS)

# Simulate 20 requests with varying lengths
np.random.seed(7)
seq_lengths = np.random.choice([128, 256, 512, 1024, 2048], size=20,
                                p=[0.3, 0.3, 0.2, 0.15, 0.05])

seqs = [Sequence(seq_id=i, prompt_len=int(seq_lengths[i]),
                  max_len=int(seq_lengths[i])) for i in range(20)]

print("  Admitting 20 requests with varying prompt lengths...")
admitted = 0
for seq in seqs:
    if bm.admit_sequence(seq):
        admitted += 1

print(f"  Admitted: {admitted}/20 requests")
print()

# Paged allocation stats
total_tokens_stored = sum(s.n_tokens for s in bm.sequences.values())
paged_blocks_used   = bm.n_allocated()
paged_waste_tokens  = paged_blocks_used * BLOCK_SIZE - total_tokens_stored

# Contiguous allocation simulation (pre-reserve max_len for each)
contiguous_tokens_reserved = sum(s.max_len for s in bm.sequences.values())
contiguous_blocks_needed   = math.ceil(contiguous_tokens_reserved / BLOCK_SIZE)
contiguous_waste           = contiguous_tokens_reserved - total_tokens_stored

print(f"  {'Metric':<40}  {'Paged':>12}  {'Contiguous':>12}")
print("  " + "─" * 66)
print(f"  {'Tokens actually stored':<40}  {total_tokens_stored:>12,}  {total_tokens_stored:>12,}")
print(f"  {'Blocks allocated':<40}  {paged_blocks_used:>12,}  {contiguous_blocks_needed:>12,}")
print(f"  {'Wasted token slots':<40}  {paged_waste_tokens:>12,}  {contiguous_waste:>12,}")
print(f"  {'Memory efficiency':<40}  "
      f"{total_tokens_stored/(paged_blocks_used*BLOCK_SIZE)*100:>11.1f}%  "
      f"{total_tokens_stored/contiguous_tokens_reserved*100:>11.1f}%")
print()
print(f"  Paged allocation: {paged_waste_tokens/(paged_blocks_used*BLOCK_SIZE)*100:.1f}% internal fragmentation.")
print(f"  Contiguous: {contiguous_waste/contiguous_tokens_reserved*100:.1f}% fragmentation.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Virtual → physical block address translation trace
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Virtual → Physical Block Address Translation")
print("━" * 68)
print()

# Pick a mid-length sequence
trace_seq = seqs[5]   # whatever the 6th sequence is
seq_id    = trace_seq.seq_id

print(f"  Sequence {seq_id}: {trace_seq.n_tokens} tokens, "
      f"{len(trace_seq.block_table)} blocks")
print(f"  Block table (virtual → physical): {trace_seq.block_table[:8]}...")
print()
print(f"  {'Token pos':>10}  {'Virtual block':>14}  {'Block offset':>14}  "
      f"{'Physical block':>16}  {'Address formula'}")
print("  " + "─" * 68)

for pos in list(range(0, min(32, trace_seq.n_tokens), 4)) + [trace_seq.n_tokens-1]:
    vb  = pos // BLOCK_SIZE
    off = pos % BLOCK_SIZE
    pb, bo = bm.virtual_to_physical(seq_id, pos)
    formula = f"block_table[{vb}]={pb}, offset {off}"
    print(f"  {pos:>10}  {vb:>14}  {off:>14}  {pb:>16}  {formula}")

print()
print("  The kernel dereferences block_table[virtual_block] to get the")
print("  physical block index, then accesses K_cache[physical, layer, head, offset, :]")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Append tokens and simulate decode steps
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Decode Simulation: Block Allocation During Generation")
print("━" * 68)
print()

# Pick a short sequence and generate 40 more tokens
sim_seq_id = seqs[0].seq_id
sim_seq    = bm.sequences[sim_seq_id]
print(f"  Simulating decode for sequence {sim_seq_id} "
      f"(prompt={sim_seq.prompt_len} tokens)")
print()
print(f"  {'Step':>5}  {'Total tokens':>13}  {'N blocks':>10}  "
      f"{'Block allocated?':>18}  {'Free blocks left':>18}")
print("  " + "─" * 62)

initial_blocks = len(sim_seq.block_table)
free_before    = bm.n_free()

for step in range(40):
    old_n_blocks = len(sim_seq.block_table)
    success = bm.append_token(sim_seq_id)
    new_block = len(sim_seq.block_table) > old_n_blocks
    if step % 4 == 0 or new_block:
        print(f"  {step:>5}  {sim_seq.n_tokens:>13}  {len(sim_seq.block_table):>10}  "
              f"{'✅ NEW BLOCK' if new_block else '  same block':>18}  "
              f"{bm.n_free():>18,}")

print()
print(f"  A new block is allocated every {BLOCK_SIZE} tokens (block_size={BLOCK_SIZE}).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Block eviction under memory pressure
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Block Eviction: Freeing Memory Under Pressure")
print("━" * 68)
print()

print(f"  Current state: {bm.n_free():,} free / {bm.n_blocks:,} total blocks")
print()

# Fill most of the remaining free blocks
fill_seqs = []
while bm.n_free() > 50:
    dummy = Sequence(seq_id=1000+len(fill_seqs), prompt_len=64, max_len=64)
    if bm.admit_sequence(dummy):
        fill_seqs.append(dummy)

print(f"  After filling: {bm.n_free():,} free blocks")
print(f"  Trying to admit a large request (1024 tokens = {1024//BLOCK_SIZE} blocks needed)...")
large_req = Sequence(seq_id=9999, prompt_len=1024, max_len=1024)
success = bm.admit_sequence(large_req)
print(f"  Admission: {'✅ SUCCESS' if success else '❌ OOM — eviction needed'}")
print()

if not success:
    # Evict smallest sequences to make room
    blocks_needed = math.ceil(1024 / BLOCK_SIZE)
    evicted = []
    while bm.n_free() < blocks_needed and fill_seqs:
        victim = fill_seqs.pop(0)   # evict FIFO (in practice: LRU or priority-based)
        blocks_freed = len(bm.sequences[victim.seq_id].block_table)
        bm.free_sequence(victim.seq_id)
        evicted.append((victim.seq_id, blocks_freed))
        print(f"  Evicted seq {victim.seq_id}: freed {blocks_freed} blocks. "
              f"Free now: {bm.n_free()}")

    success = bm.admit_sequence(large_req)
    outcome = '✅ admitted' if success else '❌ still OOM'
    print(f"  After evicting {len(evicted)} sequences: {outcome}")
    print(f"  Evicted sequences would be recomputed or swapped on resume.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Continuous Batching Simulator — Throughput vs Static Batching": {
        "description": (
            "Simulate continuous batching vs static batching on a realistic "
            "workload of 100 requests with Zipf-distributed sequence lengths. "
            "Track GPU utilisation, time-to-first-token, inter-token latency, "
            "and total throughput for both approaches. Show how continuous "
            "batching eliminates idle GPU cycles when sequences complete early. "
            "Demonstrate chunked prefill and its effect on decode latency spikes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

print("=" * 68)
print("  CONTINUOUS BATCHING SIMULATOR — Throughput vs Static Batching")
print("=" * 68)
print()

np.random.seed(99)

# Hardware and model parameters
H100_BW_GBS     = 3350.0
WEIGHTS_7B_GB   = 14.0       # FP16 weights
KV_PER_TOKEN_MB = 0.5        # Llama-2-7B KV bytes per token (MB)
MAX_BATCH_SIZE  = 16         # maximum concurrent sequences
BLOCK_SIZE      = 16         # paged attention block size

# Compute time per decode step (memory-bound model)
def step_time_ms(batch_size, avg_ctx_len):
    """Estimated ms per decode step."""
    kv_gb    = avg_ctx_len * KV_PER_TOKEN_MB / 1024 * batch_size
    total_gb = WEIGHTS_7B_GB + kv_gb
    return total_gb / H100_BW_GBS * 1000  # ms


@dataclass
class Request:
    req_id:       int
    arrival_time: float     # ms
    prompt_len:   int
    gen_len:      int       # tokens to generate
    tokens_done:  int = 0
    first_token_time: float = 0.0
    done_time:    float = 0.0
    admitted_time: float = 0.0

    @property
    def finished(self):
        return self.tokens_done >= self.gen_len


def make_workload(n_requests):
    """Generate Zipf-distributed sequence lengths, Poisson arrivals."""
    # Generation lengths: heavy-tailed (Zipf), common in real LLM workloads
    gen_lens = np.random.zipf(1.5, n_requests)
    gen_lens = np.clip(gen_lens, 10, 400).astype(int)

    # Prompt lengths: roughly 1/3 of generation length
    prompt_lens = np.maximum(32, (gen_lens * 0.5).astype(int))

    # Arrivals: Poisson process (rate = 1 per 20 ms on average)
    inter_arrivals = np.random.exponential(20.0, n_requests)
    arrival_times  = np.cumsum(inter_arrivals)

    return [Request(i, arrival_times[i], int(prompt_lens[i]), int(gen_lens[i]))
            for i in range(n_requests)]


def simulate_static_batching(requests, batch_size):
    """
    Static batching: fill a batch, run all to completion, then fill next batch.
    Returns (throughput_tps, avg_ttft_ms, avg_itl_ms, utilisation_pct).
    """
    waiting   = list(requests)
    t_ms      = 0.0
    total_tok = 0
    ttft_list = []
    itl_list  = []

    while waiting:
        # Fill batch up to batch_size
        batch = waiting[:batch_size]
        waiting = waiting[batch_size:]

        # Wait for first request in batch to arrive
        t_ms = max(t_ms, max(r.arrival_time for r in batch))

        # Prefill all sequences in batch simultaneously
        avg_prompt = np.mean([r.prompt_len for r in batch])
        prefill_ms = avg_prompt / 500 * (WEIGHTS_7B_GB * 2) / H100_BW_GBS * 1000
        t_ms += prefill_ms

        for r in batch:
            r.first_token_time = t_ms
            r.admitted_time    = t_ms
            ttft_list.append(t_ms - r.arrival_time)

        # Decode until ALL in batch finish (longest determines end)
        max_gen = max(r.gen_len for r in batch)
        idle_tokens_accumulator = 0

        for step in range(max_gen):
            active = [r for r in batch if not r.finished]
            if not active:
                break
            idle_tokens_accumulator += (len(batch) - len(active))

            step_ms = step_time_ms(len(active), avg_prompt + step)
            t_ms   += step_ms
            for r in active:
                r.tokens_done += 1
                itl_list.append(step_ms)
            total_tok += len(active)

        for r in batch:
            r.done_time = t_ms

    util_pct = total_tok / (total_tok + sum(
        (r.gen_len - 1) for r in requests) * 0) * 100  # simplified

    if not requests:
        return 0, 0, 0, 0

    end_time  = max(r.done_time for r in requests)
    start_t   = requests[0].arrival_time
    total_gen = sum(r.gen_len for r in requests)
    tps       = total_gen / (end_time - start_t) * 1000
    ttft      = np.mean(ttft_list) if ttft_list else 0
    itl       = np.mean(itl_list) if itl_list else 0
    return tps, ttft, itl


def simulate_continuous_batching(requests, max_batch):
    """
    Continuous batching: admit new requests as slots free.
    Returns (throughput_tps, avg_ttft_ms, avg_itl_ms).
    """
    waiting   = deque(sorted(requests, key=lambda r: r.arrival_time))
    active    = []
    t_ms      = 0.0
    ttft_list = []
    itl_list  = []
    total_tok = 0

    while waiting or active:
        # Admit new requests
        while waiting and len(active) < max_batch:
            r = waiting[0]
            if r.arrival_time <= t_ms + 5:   # admit requests that arrived recently
                r = waiting.popleft()
                r.admitted_time = t_ms
                active.append(r)
            else:
                break

        if not active:
            if waiting:
                t_ms = waiting[0].arrival_time
                continue
            break

        # One decode step
        avg_ctx   = np.mean([r.prompt_len + r.tokens_done for r in active])
        step_ms   = step_time_ms(len(active), avg_ctx)
        t_ms     += step_ms

        completed = []
        for r in active:
            r.tokens_done += 1
            total_tok += 1
            if r.tokens_done == 1:
                r.first_token_time = t_ms
                ttft_list.append(t_ms - r.arrival_time)
            itl_list.append(step_ms)
            if r.finished:
                r.done_time = t_ms
                completed.append(r)

        for r in completed:
            active.remove(r)

    if not requests:
        return 0, 0, 0

    end_time  = max(r.done_time for r in requests if r.done_time > 0)
    start_t   = requests[0].arrival_time
    total_gen = sum(r.gen_len for r in requests)
    tps       = total_gen / (end_time - start_t) * 1000
    ttft      = np.mean(ttft_list) if ttft_list else 0
    itl       = np.mean(itl_list) if itl_list else 0
    return tps, ttft, itl


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Workload generation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Workload Statistics (100 Requests)")
print("━" * 68)
print()

N_REQ    = 100
requests = make_workload(N_REQ)

gen_lens    = [r.gen_len    for r in requests]
prompt_lens = [r.prompt_len for r in requests]

print(f"  Requests: {N_REQ}")
print(f"  Generation length: "
      f"mean={np.mean(gen_lens):.1f}, "
      f"median={int(np.median(gen_lens))}, "
      f"p95={int(np.percentile(gen_lens, 95))}, "
      f"max={max(gen_lens)}")
print(f"  Prompt length: "
      f"mean={np.mean(prompt_lens):.1f}, "
      f"p95={int(np.percentile(prompt_lens, 95))}")
print(f"  Arrival rate: ~{N_REQ/(requests[-1].arrival_time/1000):.1f} req/sec")
print()

# Show distribution
bins = [0, 25, 50, 100, 200, 400, 1000]
hist, _ = np.histogram(gen_lens, bins=bins)
print("  Generation length distribution:")
for i in range(len(bins)-1):
    bar = '█' * min(40, hist[i])
    print(f"    {bins[i]:>4}–{bins[i+1]:<4}  {bar}  {hist[i]} reqs")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Static vs continuous batching comparison
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Static vs Continuous Batching: Key Metrics")
print("━" * 68)
print()

batch_sizes = [4, 8, 16]

print(f"  {'Method':<30}  {'Batch':>6}  {'Throughput':>12}  "
      f"{'Avg TTFT':>10}  {'Avg ITL':>9}")
print("  " + "─" * 68)

for bs in batch_sizes:
    reqs_s = make_workload(N_REQ)
    reqs_c = make_workload(N_REQ)

    tps_s, ttft_s, itl_s = simulate_static_batching(reqs_s, bs)
    tps_c, ttft_c, itl_c = simulate_continuous_batching(reqs_c, bs)

    print(f"  {'Static batching':<30}  {bs:>6}  {tps_s:>10.1f} t/s  "
          f"{ttft_s:>8.1f} ms  {itl_s:>7.2f} ms")
    print(f"  {'Continuous batching':<30}  {bs:>6}  {tps_c:>10.1f} t/s  "
          f"{ttft_c:>8.1f} ms  {itl_c:>7.2f} ms")
    if tps_s > 0:
        print(f"  {'  → Speedup':<30}  {'':>6}  {tps_c/tps_s:>10.2f}×  "
              f"{'':>10}  {'':>9}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: GPU utilisation over time
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Batch Occupancy Over Time (Static vs Continuous)")
print("━" * 68)
print()

# Simulate occupancy timeline for static batching (batch_size=8)
BATCH = 8
reqs_timeline = make_workload(40)
reqs_timeline.sort(key=lambda r: r.arrival_time)

t_ms = 0.0
batch_occupancies_static = []
batch_occupancies_cont   = []

print("  STATIC BATCHING occupancy trace (batch=8, 40 requests):")
print()
print(f"  {'Batch #':>8}  {'Batch size':>11}  {'Steps':>7}  "
      f"{'Max gen':>9}  {'Avg gen':>9}  {'Wasted steps'}")
print("  " + "─" * 60)

for batch_idx in range(0, len(reqs_timeline), BATCH):
    batch = reqs_timeline[batch_idx:batch_idx+BATCH]
    if not batch:
        break
    max_gen  = max(r.gen_len for r in batch)
    avg_gen  = np.mean([r.gen_len for r in batch])
    wasted   = sum(max_gen - r.gen_len for r in batch)
    waste_pct = wasted / (max_gen * len(batch)) * 100
    print(f"  {batch_idx//BATCH+1:>8}  {len(batch):>11}  {max_gen:>7}  "
          f"{max_gen:>9}  {avg_gen:>9.1f}  {wasted:>5} ({waste_pct:.0f}%)")

print()
print("  CONTINUOUS BATCHING: no wasted steps — completed sequences")
print("  immediately replaced by new arrivals from the wait queue.")
print("  GPU always runs at MAX_BATCH_SIZE until all requests complete.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · RadixAttention — Radix Tree, Prefix Matching & Cache Hit Rate": {
        "description": (
            "Implement a RadixAttention radix tree for prefix KV caching. "
            "Support insert, longest-prefix-match lookup, and LRU eviction. "
            "Simulate a realistic multi-turn conversation workload and a RAG "
            "workload with shared document prefixes. Compute cache hit rate, "
            "prefix reuse ratio, and TTFT improvement. Show how block-granular "
            "matching affects hit rate vs token-granular matching."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

print("=" * 68)
print("  RADIXATTENTION — Radix Tree, Prefix Matching & Cache Hit Rate")
print("=" * 68)
print()

np.random.seed(7)

BLOCK_SIZE     = 16     # tokens per KV block
KV_PER_BLOCK_MB= 8.0   # MB per KV block (Llama-2-7B)
PREFILL_MS_PER_TOKEN = 0.05  # approximate prefill cost per token


# ─────────────────────────────────────────────────────────────────────
# Radix tree for prefix caching
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RadixNode:
    token_seq:    Tuple      # tokens on the edge leading to this node
    kv_blocks:    List[int]  # physical block IDs for this node's tokens
    children:     Dict = field(default_factory=dict)  # first_token → child
    ref_count:    int = 0    # active references
    last_access:  float = 0.0   # for LRU eviction
    parent:       Optional['RadixNode'] = None


class RadixTree:
    def __init__(self, max_blocks: int = 10000):
        self.root       = RadixNode(token_seq=(), kv_blocks=[])
        self.max_blocks = max_blocks
        self.used_blocks= 0
        self.n_hits     = 0
        self.n_lookups  = 0
        self.n_tokens_reused   = 0
        self.n_tokens_computed = 0
        self._time      = 0.0

    def _allocate_blocks(self, n_tokens: int) -> List[int]:
        """Simulate allocating n_tokens worth of KV blocks."""
        n_blocks = math.ceil(n_tokens / BLOCK_SIZE)
        start    = self.used_blocks
        self.used_blocks += n_blocks
        return list(range(start, start + n_blocks))

    def _split_to_blocks(self, tokens: Tuple) -> List[Tuple]:
        """Split token sequence into complete blocks only."""
        n_complete = len(tokens) // BLOCK_SIZE
        return [tokens[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE]
                for i in range(n_complete)]

    def insert(self, tokens: Tuple) -> int:
        """
        Insert a token sequence into the radix tree.
        Returns number of NEW tokens stored (not already cached).
        """
        self._time += 1.0
        blocks = self._split_to_blocks(tokens)

        node   = self.root
        new_tokens = 0

        for block_tokens in blocks:
            first = block_tokens[0]
            if first in node.children:
                child = node.children[first]
                # Check if this child's token_seq matches
                if child.token_seq == block_tokens:
                    child.last_access = self._time
                    node = child
                    continue

            # New node needed
            kv_blocks = self._allocate_blocks(len(block_tokens))
            new_node  = RadixNode(
                token_seq=block_tokens,
                kv_blocks=kv_blocks,
                parent=node,
                last_access=self._time,
            )
            node.children[first] = new_node
            node = new_node
            new_tokens += len(block_tokens)

        return new_tokens

    def longest_prefix_match(self, tokens: Tuple) -> Tuple[int, List[int]]:
        """
        Find the longest matching prefix in the tree.
        Returns (matched_token_count, matched_kv_blocks).
        Only matches at BLOCK BOUNDARIES (complete blocks).
        """
        self._time  += 1.0
        self.n_lookups += 1
        blocks = self._split_to_blocks(tokens)

        node          = self.root
        matched_toks  = 0
        matched_blocks= []

        for block_tokens in blocks:
            first = block_tokens[0]
            if first in node.children:
                child = node.children[first]
                if child.token_seq == block_tokens:
                    matched_toks   += len(block_tokens)
                    matched_blocks += child.kv_blocks
                    child.last_access = self._time
                    node = child
                    continue
            break

        if matched_toks > 0:
            self.n_hits += 1
            self.n_tokens_reused += matched_toks

        return matched_toks, matched_blocks

    @property
    def hit_rate(self):
        return self.n_hits / max(self.n_lookups, 1)

    @property
    def reuse_ratio(self):
        total = self.n_tokens_reused + self.n_tokens_computed
        return self.n_tokens_reused / max(total, 1)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Radix tree operations trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Radix Tree Operations: Insert & Prefix Match Trace")
print("━" * 68)
print()

tree = RadixTree()

# Simulate a system prompt shared by many requests
SYSTEM_PROMPT_LEN = 64    # 64 tokens (4 complete blocks)
SYSTEM_PROMPT     = tuple(range(SYSTEM_PROMPT_LEN))  # simulate token IDs

# 5 conversations, each adding turns
print(f"  System prompt: {SYSTEM_PROMPT_LEN} tokens (4 blocks of {BLOCK_SIZE})")
print()

conversations = [
    SYSTEM_PROMPT + tuple(range(100, 100+32)),    # conv 1: sys + 32 tokens
    SYSTEM_PROMPT + tuple(range(200, 200+48)),    # conv 2: sys + 48 tokens
    SYSTEM_PROMPT + tuple(range(100, 100+64)),    # conv 3: extends conv 1
    SYSTEM_PROMPT + tuple(range(200, 200+32)),    # conv 4: prefix of conv 2
    tuple(range(300, 300+32)),                     # conv 5: no common prefix
]

print(f"  {'Request':>8}  {'Length':>8}  {'Matched':>8}  "
      f"{'Hit rate':>10}  {'Match rate':>12}  {'KV blocks reused'}")
print("  " + "─" * 64)

for i, conv in enumerate(conversations):
    matched, kv = tree.longest_prefix_match(conv)
    new_toks    = tree.insert(conv)
    tree.n_tokens_computed += len(conv) - matched

    hit_str  = f"{tree.hit_rate*100:.0f}%"
    reuse_str = f"{tree.reuse_ratio*100:.0f}%"
    print(f"  {i+1:>8}  {len(conv):>8}  {matched:>8}  "
          f"{hit_str:>10}  {reuse_str:>12}  {len(kv)} blocks")

print()
print(f"  Final tree stats: {tree.used_blocks} blocks stored, "
      f"hit rate {tree.hit_rate:.0%}, token reuse {tree.reuse_ratio:.0%}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Multi-turn conversation workload
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Multi-Turn Conversation Workload")
print("━" * 68)
print()

N_USERS    = 30
N_TURNS    = 6
SYS_LEN    = 64
TURN_LEN   = 32    # average tokens per turn
BLOCK_SIZE_sim = BLOCK_SIZE

tree2 = RadixTree(max_blocks=50000)
sys_prompt = tuple(range(1000, 1000 + SYS_LEN))

ttft_cache  = []
ttft_nocache = []
total_requests = 0

print(f"  {N_USERS} users × {N_TURNS} turns, sys_prompt={SYS_LEN} tokens, "
      f"turn_len≈{TURN_LEN} tokens")
print()

for user_id in range(N_USERS):
    conversation = sys_prompt   # start with system prompt
    for turn in range(N_TURNS):
        # Add a new user turn
        turn_tokens = tuple(range(
            user_id * 10000 + turn * 100,
            user_id * 10000 + turn * 100 + TURN_LEN
        ))
        conversation = conversation + turn_tokens

        # Lookup prefix in cache
        matched, kv = tree2.longest_prefix_match(conversation)
        new_tokens  = len(conversation) - matched

        # Insert new tokens into cache
        tree2.insert(conversation)
        tree2.n_tokens_computed += new_tokens

        # TTFT = prefill cost for uncached portion
        ttft_with    = new_tokens * PREFILL_MS_PER_TOKEN
        ttft_without = len(conversation) * PREFILL_MS_PER_TOKEN

        ttft_cache.append(ttft_with)
        ttft_nocache.append(ttft_without)
        total_requests += 1

print(f"  {'Turn':>6}  {'Avg ctx len':>13}  {'Avg matched%':>14}  "
      f"{'TTFT w/ cache':>15}  {'TTFT w/o cache':>16}  {'Speedup'}")
print("  " + "─" * 70)

for turn in range(N_TURNS):
    turn_reqs = [i for i in range(turn, total_requests, N_TURNS)]
    avg_ctx   = SYS_LEN + (turn + 1) * TURN_LEN
    # Approximate matched fraction for this turn
    avg_matched_pct = min(100, (1 - 1/(turn+2)) * 100) if turn > 0 else 0
    ttft_w    = np.mean([ttft_cache[i]   for i in range(turn, total_requests, N_TURNS)])
    ttft_wo   = np.mean([ttft_nocache[i] for i in range(turn, total_requests, N_TURNS)])
    speedup   = ttft_wo / ttft_w if ttft_w > 0 else 1
    print(f"  {turn+1:>6}  {avg_ctx:>13}  {avg_matched_pct:>13.0f}%  "
          f"{ttft_w:>15.2f} ms  {ttft_wo:>16.2f} ms  {speedup:>6.1f}×")

overall_speedup = np.mean(ttft_nocache) / np.mean(ttft_cache)
print()
print(f"  Overall hit rate: {tree2.hit_rate:.0%}")
print(f"  Token reuse ratio: {tree2.reuse_ratio:.0%}")
print(f"  Average TTFT speedup: {overall_speedup:.1f}×")
print(f"  (Turn 1 has 0× speedup: system prompt is new each user session.)")
print(f"  (Turns 2-6 benefit increasingly as conversation grows.)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: RAG workload — shared document prefix
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — RAG Workload: Shared Document Prefix Cache")
print("━" * 68)
print()

DOCUMENT_LEN   = 2048  # tokens in shared document
N_QUERIES      = 50    # queries over the same document
QUERY_LEN      = 64    # per-query addition

tree3 = RadixTree(max_blocks=100000)
document_tokens = tuple(range(50000, 50000 + DOCUMENT_LEN))

print(f"  Document: {DOCUMENT_LEN} tokens, {N_QUERIES} queries over same document")
print(f"  Query additional tokens: {QUERY_LEN} each")
print()

ttft_rag_cache   = []
ttft_rag_nocache = []

for q_idx in range(N_QUERIES):
    query_tokens = tuple(range(q_idx * 1000, q_idx * 1000 + QUERY_LEN))
    full_prompt  = document_tokens + query_tokens

    matched, _  = tree3.longest_prefix_match(full_prompt)
    new_tokens   = len(full_prompt) - matched
    tree3.insert(full_prompt)
    tree3.n_tokens_computed += new_tokens

    ttft_rag_cache.append(new_tokens * PREFILL_MS_PER_TOKEN)
    ttft_rag_nocache.append(len(full_prompt) * PREFILL_MS_PER_TOKEN)

print(f"  {'Query #':>8}  {'Full len':>10}  {'Matched':>10}  "
      f"{'TTFT cache':>12}  {'TTFT no-cache':>15}  {'Speedup'}")
print("  " + "─" * 64)

for q_idx in [0, 1, 2, 5, 10, 25, 49]:
    matched_approx = DOCUMENT_LEN if q_idx > 0 else 0
    ttft_c  = ttft_rag_cache[q_idx]
    ttft_nc = ttft_rag_nocache[q_idx]
    sx      = ttft_nc / ttft_c if ttft_c > 0 else float('inf')
    sx_str  = f"{sx:.0f}×" if sx < 1000 else "∞"
    print(f"  {q_idx+1:>8}  {DOCUMENT_LEN+QUERY_LEN:>10}  "
          f"{matched_approx:>10}  {ttft_c:>12.2f} ms  "
          f"{ttft_nc:>15.2f} ms  {sx_str:>7}")

print()
rag_speedup = np.mean(ttft_rag_nocache[1:]) / np.mean(ttft_rag_cache[1:])
print(f"  After first query: TTFT speedup = {rag_speedup:.0f}×")
print(f"  Document KV cache size: {DOCUMENT_LEN/BLOCK_SIZE * KV_PER_BLOCK_MB:.0f} MB")
print(f"  Amortised across {N_QUERIES} queries: {DOCUMENT_LEN * PREFILL_MS_PER_TOKEN:.0f} ms prefill")
print(f"  saved per query (paid only once for first query).")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · End-to-End Serving Simulator — Paged + Continuous + Prefix Cache": {
        "description": (
            "Combine all three mechanisms (paged attention, continuous batching, "
            "prefix caching) into an integrated serving simulation. Compare four "
            "configurations: baseline (static batch, no paging), paged only, "
            "paged + continuous, and paged + continuous + prefix cache. Measure "
            "throughput (tokens/sec), TTFT (ms), P95 TTFT, and GPU memory "
            "utilisation across a 200-request workload with shared system prompts."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque, OrderedDict

print("=" * 68)
print("  END-TO-END SERVING SIMULATOR — Full Stack Integration")
print("=" * 68)
print()

np.random.seed(42)

# Config
H100_HBM_GB     = 80.0
H100_BW_GBS     = 3350.0
WEIGHTS_GB      = 14.0
BLOCK_SIZE      = 16
KV_MB_PER_TOK   = 0.5    # Llama-2-7B
PREFILL_MS_PER_T= 0.04   # ms per token prefill
MAX_BATCH        = 24
SYS_PROMPT_LEN  = 128    # tokens in shared system prompt

def available_kv_gb(fragmentation_pct=0):
    used = WEIGHTS_GB + (H100_HBM_GB - WEIGHTS_GB) * fragmentation_pct / 100
    return H100_HBM_GB - used - 2.0   # 2 GB overhead

def tokens_capacity(frag_pct=0):
    return int(available_kv_gb(frag_pct) * 1024 / KV_MB_PER_TOK)

def decode_step_ms(batch_size, avg_ctx):
    kv_gb = avg_ctx * KV_MB_PER_TOK / 1024 * batch_size
    return (WEIGHTS_GB + kv_gb) / H100_BW_GBS * 1000


@dataclass
class SimRequest:
    req_id:    int
    arrival:   float
    prompt:    int      # prompt length
    gen_len:   int      # total tokens to generate
    has_prefix: bool    # shares system prompt
    tokens_done: int = 0
    ttft:      float = 0.0
    done_at:   float = 0.0
    admitted:  float = 0.0


def make_requests(n=200):
    """200 requests, 70% share the system prompt, Zipf generation lengths."""
    np.random.seed(42)
    gen_lens    = np.clip(np.random.zipf(1.5, n), 20, 300).astype(int)
    prompt_lens = np.maximum(32, (gen_lens * 0.4 + SYS_PROMPT_LEN * 0.7).astype(int))
    arrivals    = np.cumsum(np.random.exponential(15.0, n))
    has_prefix  = np.random.rand(n) < 0.7
    return [SimRequest(i, arrivals[i], int(prompt_lens[i]),
                        int(gen_lens[i]), bool(has_prefix[i]))
            for i in range(n)]


def simulate(config_name, requests, use_paging, use_cont_batch, use_prefix):
    """Run the simulation with the given configuration."""
    reqs = [SimRequest(r.req_id, r.arrival, r.prompt, r.gen_len, r.has_prefix)
            for r in requests]

    waiting     = deque(sorted(reqs, key=lambda r: r.arrival))
    active: List[SimRequest] = []

    # Capacity model
    if use_paging:
        frag_overhead_pct = 3.0   # paging reduces fragmentation
    else:
        frag_overhead_pct = 55.0  # static allocation: ~55% fragmentation

    max_tokens = tokens_capacity(frag_overhead_pct)
    prefix_cache_hits = 0

    t_ms = 0.0
    ttft_list = []
    itl_list  = []
    total_gen = 0

    while waiting or active:
        # Admit new requests (continuous batching: each step)
        if use_cont_batch:
            # Try to fill batch up to MAX_BATCH
            while waiting and len(active) < MAX_BATCH:
                r = waiting[0]
                if r.arrival <= t_ms + 5:
                    r = waiting.popleft()
                    r.admitted = t_ms

                    # Prefix cache hit: reduce effective prompt
                    if use_prefix and r.has_prefix:
                        cached_prefix = (SYS_PROMPT_LEN // BLOCK_SIZE) * BLOCK_SIZE
                        r.prompt = max(BLOCK_SIZE, r.prompt - cached_prefix)
                        prefix_cache_hits += 1

                    active.append(r)
                else:
                    break
        else:
            # Static: fill once, run to completion
            if not active and waiting:
                n_admit = min(MAX_BATCH, len(waiting))
                for _ in range(n_admit):
                    r = waiting.popleft()
                    r.admitted = max(t_ms, r.arrival)
                    active.append(r)
                t_ms = max(t_ms, max(r.arrival for r in active))

                # Prefill all in batch
                avg_p = np.mean([r.prompt for r in active])
                t_ms += avg_p * PREFILL_MS_PER_T * len(active) * 0.15
                for r in active:
                    r.ttft = t_ms
                    ttft_list.append(t_ms - r.arrival)

        if not active:
            if waiting:
                t_ms = waiting[0].arrival
            continue

        # One decode step
        avg_ctx   = np.mean([r.prompt + r.tokens_done for r in active])
        step_ms   = decode_step_ms(len(active), avg_ctx)
        t_ms     += step_ms

        completed = []
        for r in active:
            r.tokens_done += 1
            total_gen += 1
            if use_cont_batch:
                if r.tokens_done == 1:
                    # Prefill was done in the same step as first decode
                    prefill_overhead = r.prompt * PREFILL_MS_PER_T
                    r.ttft = t_ms
                    ttft_list.append(t_ms - r.arrival + prefill_overhead)
            itl_list.append(step_ms)
            if r.tokens_done >= r.gen_len:
                r.done_at = t_ms
                completed.append(r)

        for r in completed:
            active.remove(r)

    end_t     = max(r.done_at for r in reqs if r.done_at > 0)
    start_t   = reqs[0].arrival
    total_req_gen = sum(r.gen_len for r in reqs)
    tps       = total_req_gen / (end_t - start_t) * 1000
    avg_ttft  = np.mean(ttft_list) if ttft_list else 999
    p95_ttft  = np.percentile(ttft_list, 95) if ttft_list else 999
    avg_itl   = np.mean(itl_list) if itl_list else 0

    mem_util = (1.0 - frag_overhead_pct / 100)

    return {
        "config":        config_name,
        "tps":           tps,
        "ttft_avg":      avg_ttft,
        "ttft_p95":      p95_ttft,
        "itl":           avg_itl,
        "mem_util":      mem_util,
        "prefix_hits":   prefix_cache_hits,
    }


# ─────────────────────────────────────────────────────────────────────
# Run all 4 configurations
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  Running 200-request simulation across 4 configurations...")
print("━" * 68)
print()

requests = make_requests(200)

configs = [
    ("Baseline (static, no paging)",       False, False, False),
    ("Paged attention only",               True,  False, False),
    ("Paged + continuous batching",        True,  True,  False),
    ("Paged + continuous + prefix cache",  True,  True,  True),
]

results = []
for name, paging, cont, prefix in configs:
    r = simulate(name, requests, paging, cont, prefix)
    results.append(r)
    print(f"  ✅ {name}")


# ─────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  RESULTS SUMMARY")
print("━" * 68)
print()

base_tps = results[0]["tps"]

print(f"  {'Configuration':<38}  {'Tput t/s':>9}  {'vs base':>8}  "
      f"{'TTFT avg':>10}  {'TTFT P95':>10}  {'Mem util':>10}")
print("  " + "─" * 88)

for r in results:
    sx     = r["tps"] / base_tps
    prefix = f"({r['prefix_hits']} hits)" if r["prefix_hits"] > 0 else ""
    print(f"  {r['config']:<38}  {r['tps']:>9.1f}  {sx:>7.2f}×  "
          f"{r['ttft_avg']:>9.0f}ms  {r['ttft_p95']:>9.0f}ms  "
          f"{r['mem_util']:>9.0%}  {prefix}")

print()
final = results[-1]
baseline = results[0]
print(f"  End-to-end improvement (full stack vs baseline):")
print(f"    Throughput:  {final['tps']/baseline['tps']:.2f}× increase")
print(f"    TTFT avg:    {baseline['ttft_avg']/final['ttft_avg']:.1f}× reduction")
print(f"    TTFT P95:    {baseline['ttft_p95']/final['ttft_p95']:.1f}× reduction")
print(f"    Memory util: {baseline['mem_util']:.0%} → {final['mem_util']:.0%}")
print()
print("  Each layer contributes:")
print(f"    Paged attention:   eliminates fragmentation (+memory → +concurrency)")
print(f"    Continuous batch:  eliminates GPU idle time (+throughput)")
print(f"    Prefix cache:      eliminates redundant prefill (-TTFT for shared prompts)")
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
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  "",
        "visual_height": 400,
        "complexity":   None,
        "operations":   OPERATIONS,
    }