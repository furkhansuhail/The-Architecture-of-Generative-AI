"""
Inference Optimisation and Serving
=====================================

LLM inference serving is a distinct engineering discipline from training.
Where training maximises GPU throughput (tokens/second), serving must
simultaneously minimise latency (time-to-first-token, time-per-output-token)
and maximise throughput (requests/second, tokens/second) under unpredictable
request patterns. The key innovations — KV caching, continuous batching,
PagedAttention, and speculative batching — together enable a single server
to handle hundreds of concurrent users at near-interactive speeds. This module
covers each optimisation with its mathematical foundations and practical impact.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Inference Optimisation and Serving"
DISPLAY_NAME = "31 · Inference & Serving"
ICON         = "🚀"
SUBTITLE     = "KV Cache, Continuous Batching, PagedAttention"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Two Regimes of LLM Inference

LLM inference has two distinct performance profiles:

    **Prefill (prompt processing):**
    The model processes the input prompt in one forward pass. This pass is
    compute-bound: a long prompt means many tokens processed simultaneously
    in a large matrix multiplication, achieving high GPU utilisation.
    Latency metric: TTFT (time-to-first-token).
    Measured in: milliseconds to seconds.

    **Decode (token generation):**
    The model generates one token at a time. Each step is memory-bandwidth-bound:
    reading the full model's weights from HBM to compute one token's probabilities.
    Latency metric: TPOT (time-per-output-token) = 1/throughput.
    Measured in: milliseconds per token (typically 10–100 ms per token at batch=1).

Most optimisation effort focuses on the decode phase because it dominates
the total response time for long outputs.

**The bandwidth bottleneck:**
At batch=1, each decode step reads all model weights (140 GB for 70B in bf16)
from HBM to compute one token:
    TPOT = weight_bytes / HBM_bandwidth = 140 GB / 2 TB/s = 70 ms/token
    (This is a lower bound; the actual cost is 2–3× higher with overhead)

At batch=N, the same weights are reused for N tokens simultaneously:
    TPOT(N) = weight_bytes / (HBM_bandwidth × N)   (ideal linear scaling)

In practice, the compute cost also scales with N, so there is a crossover
point where the throughput bottleneck shifts from memory bandwidth to compute.


### KV Cache: The Fundamental Building Block

Every inference optimisation builds on the **KV cache**. During autoregressive
generation, the attention mechanism computes:

    Attention(Q_t, K_{1:t}, V_{1:t}) = softmax(Q_t K_{1:t}^T / √d_head) V_{1:t}

where t is the current position. The key observation: K_i and V_i for all
i < t are the same values used in every subsequent forward pass. Without
a KV cache, they would be recomputed at every step.

**KV cache:** Store K_i and V_i for all i < t in GPU memory. At each step,
only compute K_t and V_t (one new entry), then use the cached K_1:t-1 and V_1:t-1.

    Per-step savings: avoids recomputing (t-1) × 2 × d_head × n_heads values
    Forward pass: O(1) per layer instead of O(t) (attention computation still O(t))

**KV cache size:**
    Per token per layer: 2 (K and V) × n_heads × d_head × dtype_bytes
                       = 2 × d_model × dtype_bytes  (since n_heads × d_head = d_model)
    Per token total:     N_layers × 2 × d_model × dtype_bytes

For LLaMA-2 70B (d=8192, N_layers=80, dtype=bf16=2 bytes):
    Per token: 80 × 2 × 8192 × 2 = 2,621,440 bytes ≈ 2.5 MB per token

At sequence length 4096:
    KV cache: 4096 × 2.5 MB ≈ 10 GB per sequence

For 8 concurrent users at 4096 tokens: 80 GB — the FULL memory of an A100.
This is the KV cache memory crisis: it fundamentally limits concurrency.


    **Diagram 1 — KV Cache Memory Breakdown:**

    KV CACHE MEMORY PER SEQUENCE (LLaMA-2 70B)
    ════════════════════════════════════════════════════════════════

    Model weights:          140 GB  (fixed, shared across all users)
    KV cache per sequence:
        per token:          2.5 MB
        at 512 tokens:      1.3 GB
        at 2048 tokens:     5.1 GB
        at 4096 tokens:     10.2 GB
        at 8192 tokens:     20.5 GB

    On 8× A100 80GB (640 GB total):
        After loading model (140 GB, only needed once):  500 GB KV budget
        At 2048 avg seq len: 500 GB / 5.1 GB ≈ 98 concurrent sequences!

    This is why KV cache management is the central challenge of LLM serving.


### Static Batching: The Naive Approach and Its Waste

Early LLM serving systems used static batching:
    1.  Collect K requests into a batch
    2.  Pad all sequences to the same length (the longest one)
    3.  Process the batch until ALL sequences are complete
    4.  Start the next batch only when the entire current batch finishes

Problems:
    1.  **Padding waste:** Short sequences in a batch waste compute on PAD tokens.
    2.  **Blocked new requests:** If user A sends a 10-token prompt and user B
        sends a 2000-token prompt, user A must wait for user B to finish even
        though user A's response was complete long ago.
    3.  **GPU bubble:** As sequences complete at different times, the batch size
        shrinks, leaving GPU partially idle at the end of each batch.
    4.  **Head-of-line blocking:** A single very long response blocks all other
        users from starting.

At low request rates, static batching is fine. Under heavy load (the production
scenario), it causes severe throughput degradation.


### Continuous Batching: The Key Innovation

**Continuous batching** (also called iteration-level scheduling or in-flight
batching) solves static batching's problems by adding and removing sequences
from the active batch at every token generation step.

**The key insight:** At each decode step, the scheduler checks:
    1.  Are there completed sequences? → Remove them from the batch
    2.  Are there waiting requests? → Add them to the batch (up to memory limit)
    3.  Process the current batch for ONE decode step

This creates a dynamic batch where different sequences are at different stages:
    User A: generating token 47 of their response (mid-generation)
    User B: starting token 1 of their response (just admitted)
    User C: generating token 312 of a long response (late in generation)

All three process their current position in ONE GPU kernel. The KV cache
handles the different histories transparently.

**The complexity:** With different sequences at different positions, the
attention computation for each sequence has a different length. This
requires careful batching:
    •   Prefill sequences (processing the prompt) can be batched together
        if they have similar prompt lengths
    •   Decode sequences (generating tokens) can be batched trivially
        (each generates exactly 1 token per step)
    •   Prefill and decode can be mixed in the same batch (chunked prefill)


    **Diagram 2 — Static vs Continuous Batching:**

    STATIC BATCHING (4 requests, max_batch=4):
    ════════════════════════════════════════════════════════════════
    Time →  0    1    2    3    4    5    6    7    8
    Req A:  [prefill][gen][gen][gen][DONE]
    Req B:  [prefill][gen][gen][gen][gen][gen][DONE]
    Req C:  [prefill][gen][gen][DONE]
    Req D:  [prefill][gen][gen][gen][gen][DONE]
                                              ↑
    Req E:  (WAITING)       ──────────────────────► starts only here
    GPU util:  █████████████████████████░░░░░░░  ← partial util late in batch

    CONTINUOUS BATCHING:
    ════════════════════════════════════════════════════════════════
    Time →  0    1    2    3    4    5    6    7    8
    Req A:  [P]  [G]  [G]  [G]  [DONE]
    Req B:  [P]  [G]  [G]  [G]  [G]  [G]  [DONE]
    Req C:  [P]  [G]  [G]  [DONE]
    Req D:  [P]  [G]  [G]  [G]  [G]  [DONE]
    Req E:       ↑ added when C finished slot    [P]  [G]  [G]  [G]
    GPU util:  █████████████████████████████████████  ← full util throughout


### PagedAttention: Virtual KV Cache Memory

The KV cache's large, variable-per-sequence memory requirement creates
memory fragmentation: when a sequence ends, its KV cache memory is freed,
but the resulting holes may be too small for a new sequence's maximum length.

**PagedAttention** (vLLM, Kwon et al., 2023) solves this by applying
virtual memory concepts to KV cache management:

    •   KV cache is divided into fixed-size **pages** (e.g., 16 tokens each)
    •   Each sequence's KV cache is a collection of pages (not contiguous!)
    •   A **block table** maps (sequence, logical_page) → physical_page
    •   Physical pages are allocated on demand, one at a time

This is identical to virtual memory in operating systems:
    Physical memory = GPU HBM
    Virtual memory  = logical KV sequence
    Page table      = block table
    Page fault       = allocate new KV page

**Benefits:**
    1.  Near-zero fragmentation (page-sized allocation granularity only)
    2.  No pre-allocation needed (allocate as tokens are generated)
    3.  KV cache can be shared across requests with the same prefix
    4.  KV cache blocks can be swapped to CPU RAM (like paging to disk)

**Prefix sharing:** When multiple users share the same system prompt,
PagedAttention can store that prefix's KV cache ONCE and reference it
from all user sequences. For a 512-token system prompt on LLaMA-2 70B,
this saves 512 × 2.5 MB = 1.25 GB per shared user.


    **Diagram 3 — PagedAttention Block Table:**

    PAGEDATTENTION: VIRTUAL KV CACHE MEMORY
    ════════════════════════════════════════════════════════════════

    Physical KV pages (16 tokens each):
    [Page 0][Page 1][Page 2][Page 3][Page 4][Page 5][Page 6][Page 7]

    Logical sequence A (35 tokens = 3 pages):
    Block table A: {0→Page0, 1→Page3, 2→Page6}  (non-contiguous!)
    Page 0: tokens 0–15
    Page 3: tokens 16–31
    Page 6: tokens 32–35 (partial)

    Logical sequence B (20 tokens = 2 pages):
    Block table B: {0→Page1, 1→Page4}
    Page 1: tokens 0–15
    Page 4: tokens 16–20 (partial)

    Shared system prompt (15 tokens = 1 page):
    Block table C: {0→Page2} ← SHARED with sequence A if same prefix!
    Block table D: {0→Page2} ← Same physical page, different sequences ✓

    Memory utilisation: ~100% (vs ~50% with contiguous allocation)


### The GPU Memory Hierarchy in Inference

    Component           Size    Bandwidth   Contents
    ─────────────────────────────────────────────────────────────────────
    HBM (GPU)           80 GB   2 TB/s      Model weights + KV cache
    DRAM (CPU)          2 TB    300 GB/s    Overflow KV pages (swapped)
    PCIe                —       32 GB/s     KV page migration (HBM↔DRAM)
    NVLink              —       600 GB/s    Tensor parallel KV sync
    ─────────────────────────────────────────────────────────────────────

**Dynamic KV page scheduling:**
When HBM is full and a new high-priority request arrives, the scheduler
can "preempt" a low-priority sequence by swapping its KV pages to CPU DRAM.
When its priority rises, pages are swapped back. This is called KV cache
offloading (analogous to process swapping in OS scheduling).


### Chunked Prefill: Combining Prefill and Decode

Standard inference processes prefill (the whole prompt at once) and decode
(one token at a time) separately. This creates a problem under high load:

    A long prefill (2000-token prompt) blocks decode for many milliseconds.
    During this time, other users' decode steps are stalled → high TPOT.

**Chunked prefill** (Agrawal et al., 2024) splits the prefill into small
chunks that are interleaved with decode steps:
    Instead of:  [2000-token prefill block] [decode] [decode] [decode]
    Use:         [500-token chunk] [decode] [500-token chunk] [decode] ...

This caps the maximum time any single decode step is delayed by a prefill.
The TTFT increases slightly (more chunks needed for the full prompt), but
the TPOT for all other users improves substantially.

vLLM implements chunked prefill with a configurable `max_prefill_tokens`
parameter. Typical values: 512–2048 tokens per chunk.


### Speculative Batching and Asynchronous Processing

Modern LLM servers (vLLM v0.5+, TGI) combine multiple optimisations:

    1.  **Speculative decoding** (Module 30): draft model proposes, target verifies
    2.  **Continuous batching**: mix decode and prefill in each step
    3.  **PagedAttention**: fragmentation-free KV management
    4.  **Chunked prefill**: bound the prefill latency impact on decode
    5.  **Tensor parallelism**: split the model across GPUs
    6.  **CUDA graph capture**: compile the decode step to avoid Python overhead

**CUDA graph capture** is a particularly important optimisation:
Normal PyTorch dispatch through Python costs ~5 ms of Python overhead per step.
At 10–50 ms total per step, this is significant (10–50% overhead).
CUDA graphs capture the GPU operations for a fixed batch size and replay them
without Python re-interpretation, eliminating this overhead.

Requirement: the batch size and tensor shapes must be the same at every call
(padding is used to maintain this). This is why vLLM uses bucketed batch sizes.


### FlashAttention and FlashDecoding for Inference

**FlashAttention** (Module 13, inference variant):
During prefill, FlashAttention avoids materialising the T×T attention matrix,
reducing memory from O(T²) to O(T) and improving throughput by 2–4×.

**FlashDecoding** (Dao et al., 2023):
Standard attention during decode has a single query (one token) attending to
a long KV cache. With multiple GPUs and long contexts:
    •   Split the KV cache sequence dimension across GPUs
    •   Each GPU computes attention between the query and its KV segment
    •   Combine partial softmax results using the log-sum-exp trick

For a 4096-token context with TP=4:
    Each GPU handles 1024 tokens of KV → 4× smaller attention per GPU
    Reduction step: O(TP) work (negligible)
    Speedup: ~4× for the attention computation in the decode step

FlashDecoding is the inference-time complement to Ring Attention (Module 21).


### vLLM Architecture and Implementation

vLLM is the dominant open-source LLM inference engine. Its architecture:

    **Key components:**
        •   LLMEngine: central scheduler + KV manager
        •   Scheduler: continuous batching, preemption, sequence priority
        •   BlockManager: PagedAttention physical page allocation
        •   Worker: GPU compute process (one per device for TP)
        •   CacheEngine: KV cache physical pages in GPU HBM

    **The execution flow per step:**
        1.  Scheduler selects which sequences to process
        2.  Sequences are divided into prefill and decode
        3.  Block tables are updated (allocate new pages as needed)
        4.  Model forward pass with block-sparse attention (using block tables)
        5.  Sampler: select next token for each sequence
        6.  Scheduler checks for completed/preempted sequences
        7.  Return completed responses; update waiting queue

    **Throughput targets (LLaMA-2 7B on A100 40GB):**
        Token throughput: ~6,000–10,000 tokens/second
        Concurrent users: 100–300 depending on seq length
        TPOT at full load: ~10–20 ms per token


### Quantisation for Inference

Inference quantisation is even more common than training quantisation:
    •   Weights are fixed at inference time — quantisation doesn't hurt quality
      the way it might during training (no gradient updates)
    •   Lower precision → fewer bytes per weight → faster memory bandwidth
    •   INT8/INT4 weights give 2–4× throughput improvement

**GPTQ** (covered in Module 32): post-training quantisation using Hessian-
weighted rounding. Achieves near-fp16 quality at 4-bit.

**AWQ** (Activation-aware Weight Quantisation): identifies which weight
channels are most important based on activation statistics; protects them
with higher precision.

**INT8 matmul with fp16 activations:**
NVIDIA's `cublasLt` supports INT8 × INT8 → INT32 accumulate matmuls.
For LLM decode (batch=1, compute-bound), INT8 weights give:
    Memory bandwidth: 2× improvement (INT8 vs fp16 = 1 byte vs 2 bytes)
    Throughput:       ~2× (memory-bound phase) or 1.2–1.5× (mixed)


### Request Scheduling Strategies

**FCFS (First Come First Served):**
Simple queue. Fair but poor for head-of-line blocking.

**Shortest Job First (SJF) / SRPT:**
Prioritise requests estimated to complete soonest. Maximises throughput
but requires output length prediction (hard for LLMs).

**Preemptive scheduling:**
When KV memory is full, preempt the longest/lowest-priority sequence by
swapping its KV pages to CPU. Resume when memory becomes available.

**Prediction-based scheduling:**
Modern systems estimate output length from the prompt to schedule more
efficiently. Techniques:
    •   Histogram-based: track output length distribution per task type
    •   LLM-based: a small classifier predicts output length from the prompt
    •   Fixed timeout: assume max_length and preempt if exceeded


### Practical Throughput and Latency Numbers

For a single A100 80GB running LLaMA-2 70B (with tensor parallelism if needed):

    System              Batch   TPOT (ms/tok)  Throughput (tok/s)
    ──────────────────────────────────────────────────────────────────
    Naive (no batching) 1       ~70ms          ~14 tok/s
    Static batch        16      ~5ms           ~3,200 tok/s
    Continuous batch    16      ~5ms           ~3,200 tok/s + no blocking
    vLLM (continuous)   32+     ~3ms           ~5,000–8,000 tok/s
    vLLM + spec decode  32+     ~2ms           ~8,000–15,000 tok/s
    ──────────────────────────────────────────────────────────────────

For LLaMA-2 7B on A100:
    vLLM alone:         ~50,000 tok/s peak throughput
    vLLM + AWQ INT4:    ~80,000 tok/s peak throughput
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Inference Optimisation Techniques

| Technique               | What it solves             | Memory impact      | Latency impact | Throughput impact |
|-------------------------|----------------------------|--------------------|----------------|-------------------|
| KV cache                | Avoids redundant compute   | +KV cache size     | -TPOT (10×)    | +throughput       |
| Continuous batching     | GPU idle time              | Same               | -TPOT          | +2–4× throughput  |
| PagedAttention          | KV fragmentation           | -waste (90%→99%)   | Neutral        | +2× concurrency   |
| FlashAttention          | T² attention memory        | -O(T²)→O(T)        | -prefill latency| +prefill throughput|
| FlashDecoding           | Single-query long-context  | Same               | -decode TPOT   | +4× at TP=4       |
| Chunked prefill         | Prefill blocks decode      | Same               | -avg TPOT      | +throughput       |
| Speculative decoding    | Sequential bottleneck      | +draft model       | -TPOT 2–4×     | +2–4× per user    |
| INT8/INT4 quantisation  | Memory bandwidth           | -2–4× weights      | -TPOT 2–4×     | +2–4× throughput  |
| Prefix caching          | Shared prefix recompute    | -KV for shared pfx | -TTFT          | +throughput       |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "KV Cache Implementation and Memory Analysis": {
        "description": "Implement a KV cache from scratch with exact memory tracking, show the memory growth over generation, and analyse the trade-off between context length and concurrent users.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
KV CACHE: IMPLEMENTATION AND MEMORY ANALYSIS
================================================================================

Implements a KV cache with:
    1. Cache allocation and management
    2. Exact memory tracking as tokens are generated
    3. Attention with KV cache (causal, incremental)
    4. Concurrent request memory analysis
    5. Maximum concurrent users as a function of sequence length

The KV cache is the fundamental building block of all inference optimisations.
Every other technique (PagedAttention, continuous batching) builds on top of it.

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ── KV Cache data structure ───────────────────────────────────────────────────

@dataclass
class KVCacheEntry:
    """KV cache for one sequence across all layers."""
    seq_id:    int
    max_len:   int
    n_layers:  int
    n_heads:   int
    d_head:    int
    dtype:     torch.dtype = torch.float16

    k_cache: torch.Tensor = field(init=False)
    v_cache: torch.Tensor = field(init=False)
    current_len: int = field(default=0, init=False)

    def __post_init__(self):
        # Pre-allocate maximum length KV cache
        # Shape: (n_layers, n_heads, max_len, d_head)
        self.k_cache = torch.zeros(
            self.n_layers, self.n_heads, self.max_len, self.d_head,
            dtype=self.dtype
        )
        self.v_cache = torch.zeros(
            self.n_layers, self.n_heads, self.max_len, self.d_head,
            dtype=self.dtype
        )

    @property
    def memory_bytes(self) -> int:
        """Total memory used by this KV cache entry."""
        per_tensor = self.n_layers * self.n_heads * self.max_len * self.d_head
        dtype_bytes = 2 if self.dtype == torch.float16 else 4
        return 2 * per_tensor * dtype_bytes  # factor 2 for K and V

    @property
    def used_memory_bytes(self) -> int:
        """Memory actually used so far (not pre-allocated)."""
        per_tensor = self.n_layers * self.n_heads * self.current_len * self.d_head
        dtype_bytes = 2 if self.dtype == torch.float16 else 4
        return 2 * per_tensor * dtype_bytes

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Write new K, V vectors at the current position."""
        t = self.current_len
        self.k_cache[layer, :, t, :] = k
        self.v_cache[layer, :, t, :] = v

    def read(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read K, V up to current position for this layer."""
        t = self.current_len
        return self.k_cache[layer, :, :t, :], self.v_cache[layer, :, :t, :]

    def advance(self):
        """Mark one new token as processed."""
        self.current_len += 1


class KVCacheManager:
    """Manages KV caches for multiple concurrent sequences."""

    def __init__(self, gpu_memory_gb: float,
                 model_weights_gb: float,
                 n_layers: int, n_heads: int, d_head: int,
                 max_seq_len: int, dtype: torch.dtype = torch.float16):
        self.gpu_memory_gb   = gpu_memory_gb
        self.model_weights_gb = model_weights_gb
        self.n_layers        = n_layers
        self.n_heads         = n_heads
        self.d_head          = d_head
        self.max_seq_len     = max_seq_len
        self.dtype           = dtype
        self.caches: dict[int, KVCacheEntry] = {}

    @property
    def kv_budget_gb(self) -> float:
        return self.gpu_memory_gb - self.model_weights_gb

    @property
    def bytes_per_token(self) -> int:
        """KV cache bytes per token (all layers, K and V)."""
        dtype_bytes = 2 if self.dtype == torch.float16 else 4
        return 2 * self.n_layers * self.n_heads * self.d_head * dtype_bytes

    @property
    def max_tokens_in_cache(self) -> int:
        return int(self.kv_budget_gb * 1e9 / self.bytes_per_token)

    def max_concurrent_users(self, avg_seq_len: int) -> int:
        return self.max_tokens_in_cache // avg_seq_len

    def allocate(self, seq_id: int, max_len: int = None) -> KVCacheEntry:
        """Allocate a KV cache entry for a new sequence."""
        if max_len is None:
            max_len = self.max_seq_len
        entry = KVCacheEntry(seq_id, max_len, self.n_layers,
                              self.n_heads, self.d_head, self.dtype)
        self.caches[seq_id] = entry
        return entry

    def free(self, seq_id: int):
        """Free the KV cache for a completed sequence."""
        self.caches.pop(seq_id, None)

    def used_memory_gb(self) -> float:
        return sum(c.used_memory_bytes for c in self.caches.values()) / 1e9

    def allocated_memory_gb(self) -> float:
        return sum(c.memory_bytes for c in self.caches.values()) / 1e9


# ── Attention with KV cache ───────────────────────────────────────────────────

class CachedAttentionLayer(nn.Module):
    """
    Attention layer that uses a KV cache for efficient inference.
    At each decode step, only the new token's Q/K/V are computed;
    K and V are appended to the cache and the full cache is used for attention.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                 cache: Optional[KVCacheEntry] = None,
                 layer_idx: int = 0) -> torch.Tensor:
        """
        x: (B, T, d)  — B=1 for single token in decode mode
        Returns: (B, T, d) with updated attention
        """
        B, T, d = x.shape
        h       = self.n_heads
        dh      = self.d_head

        Q = self.W_Q(x).view(B, T, h, dh).transpose(1, 2)  # (B, h, T, dh)
        K = self.W_K(x).view(B, T, h, dh).transpose(1, 2)
        V = self.W_V(x).view(B, T, h, dh).transpose(1, 2)

        if cache is not None and T == 1:
            # DECODE MODE: append new K, V to cache
            cache.write(layer_idx,
                        K[0, :, 0, :],   # (h, dh)
                        V[0, :, 0, :])   # (h, dh)
            # Read full KV history from cache
            K_full, V_full = cache.read(layer_idx)  # (h, t, dh)
            K = K_full.unsqueeze(0)  # (1, h, t, dh)
            V = V_full.unsqueeze(0)
        # Else: PREFILL MODE — use full K, V as-is

        scale   = math.sqrt(dh)
        scores  = Q @ K.transpose(-2, -1) / scale   # (B, h, T, t)

        # Causal mask
        T_q, T_k = scores.shape[-2], scores.shape[-1]
        mask = torch.tril(torch.ones(T_q, T_k, dtype=torch.bool,
                                      device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_O(out)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # LLaMA-2 7B approximate config
    D_MODEL    = 4096
    N_HEADS    = 32
    D_HEAD     = D_MODEL // N_HEADS
    N_LAYERS   = 32
    GPU_MEM    = 80.0   # GB A100
    MODEL_WGTS = 14.0   # 7B × 2 bytes bf16

    manager = KVCacheManager(
        gpu_memory_gb   = GPU_MEM,
        model_weights_gb = MODEL_WGTS,
        n_layers         = N_LAYERS,
        n_heads          = N_HEADS,
        d_head           = D_HEAD,
        max_seq_len      = 4096,
    )

    print("=" * 65)
    print("  KV CACHE MEMORY ANALYSIS — LLaMA-2 7B (4096 ctx)")
    print(f"  GPU HBM:     {GPU_MEM:.0f} GB")
    print(f"  Model size:  {MODEL_WGTS:.0f} GB  (bf16 weights)")
    print(f"  KV budget:   {manager.kv_budget_gb:.0f} GB")
    print("=" * 65)
    print()

    print(f"  KV bytes per token: {manager.bytes_per_token:,}")
    print(f"  Max tokens in cache: {manager.max_tokens_in_cache:,}")
    print()

    print(f"  {'Avg seq len':>14}  {'Max concurrent users':>22}  "
          f"{'KV per user (MB)':>18}")
    print(f"  {'':─>14}  {'':─>22}  {'':─>18}")
    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        max_users = manager.max_concurrent_users(seq_len)
        kv_mb     = seq_len * manager.bytes_per_token / 1e6
        print(f"  {seq_len:>14,}  {max_users:>22,}  {kv_mb:>18.1f}")

    # Memory growth during generation
    print()
    print("=" * 65)
    print("  KV CACHE MEMORY GROWTH DURING GENERATION")
    print("  (Simulating 3 concurrent users)")
    print("=" * 65)
    print()

    # Allocate caches for 3 users
    cache_a = manager.allocate(0, max_len=1000)
    cache_b = manager.allocate(1, max_len=1000)
    cache_c = manager.allocate(2, max_len=1000)

    print(f"  {'Tokens gen':>12}  {'User A':>10}  {'User B':>10}  "
          f"{'User C':>10}  {'Total (MB)':>12}  {'Budget used':>12}")
    print(f"  {'':─>12}  {'':─>10}  {'':─>10}  {'':─>10}  "
          f"{'':─>12}  {'':─>12}")

    # Simulate different generation lengths
    for step in range(0, 201, 25):
        # Advance caches by 25 tokens each step
        for _ in range(25 if step > 0 else 0):
            cache_a.advance()
        for _ in range(20 if step > 0 else 0):
            cache_b.advance()
        for _ in range(30 if step > 0 else 0):
            cache_c.advance()

        total_mb = manager.used_memory_gb() * 1e3  # MB
        budget   = total_mb / (manager.kv_budget_gb * 1e3) * 100

        print(f"  {step:>12}  {cache_a.current_len:>10}  "
              f"{cache_b.current_len:>10}  "
              f"{cache_c.current_len:>10}  "
              f"{total_mb:>12.1f}  {budget:>11.1f}%")

    # Free a user
    manager.free(1)
    print()
    print("  After user B completes:")
    total_mb = manager.used_memory_gb() * 1e3
    print(f"  Memory freed: {cache_b.used_memory_bytes/1e6:.1f} MB reclaimed")
    print(f"  Remaining: {total_mb:.1f} MB  ({total_mb/(manager.kv_budget_gb*1e3)*100:.1f}% of budget)")
    print()

    # Functional demo with attention layer
    print("=" * 65)
    print("  CACHED ATTENTION: PREFILL vs DECODE CORRECTNESS")
    print("=" * 65)
    print()
    torch.manual_seed(0)

    DEMO_D     = 64
    DEMO_H     = 4
    DEMO_SEQLEN = 8

    attn = CachedAttentionLayer(DEMO_D, DEMO_H)
    demo_cache = KVCacheEntry(seq_id=99, max_len=DEMO_SEQLEN + 10,
                               n_layers=1, n_heads=DEMO_H,
                               d_head=DEMO_D//DEMO_H)

    # Reference: process full sequence at once (prefill mode)
    x_full = torch.randn(1, DEMO_SEQLEN, DEMO_D)
    with torch.no_grad():
        out_prefill = attn(x_full, cache=None, layer_idx=0)

    # Incremental: process tokens one at a time with KV cache
    outputs_cached = []
    for t in range(DEMO_SEQLEN):
        x_t = x_full[:, t:t+1, :]   # (1, 1, d)
        with torch.no_grad():
            out_t = attn(x_t, cache=demo_cache, layer_idx=0)
        demo_cache.advance()
        outputs_cached.append(out_t)

    out_cached = torch.cat(outputs_cached, dim=1)   # (1, T, d)

    diff = (out_prefill - out_cached).abs().max().item()
    print(f"  Prefill output shape:         {out_prefill.shape}")
    print(f"  Cached decode output shape:   {out_cached.shape}")
    print(f"  Max diff (prefill vs cached): {diff:.2e}")
    print(f"  {'✓ KV cache produces identical results!' if diff < 1e-4 else '✗ Mismatch'}")
    print()
    print(f"  Memory profile (decode after {DEMO_SEQLEN} tokens):")
    print(f"  KV cache bytes used: {demo_cache.used_memory_bytes:,}")
    print(f"  KV cache bytes allocd: {demo_cache.memory_bytes:,}")
''',
    },

    "Continuous Batching Scheduler": {
        "description": "Implement a continuous batching scheduler that dynamically adds and removes sequences from the active batch — showing GPU utilisation comparison vs static batching.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
CONTINUOUS BATCHING SCHEDULER
================================================================================

Simulates a continuous batching scheduler:
    1. Request queue: incoming requests with random prompt and output lengths
    2. Active batch: sequences currently being decoded
    3. Scheduler: adds/removes sequences each step based on memory budget
    4. GPU utilisation tracking: compares continuous vs static batching
    5. TPOT and TTFT latency analysis

Continuous batching achieves near-100% GPU utilisation by ensuring there are
always enough active sequences to fill the batch, regardless of when individual
sequences complete.

================================================================================
"""

import math
import random
import heapq
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class Request:
    """One user request."""
    request_id:     int
    prompt_tokens:  int    # number of input tokens
    max_output:     int    # maximum tokens to generate
    arrival_time:   float  # when the request arrived (seconds)

    # Set during processing
    start_time:     Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time:    Optional[float] = None
    tokens_generated: int = 0

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def tpot_avg(self) -> Optional[float]:
        if self.finish_time and self.first_token_time and self.tokens_generated > 1:
            return (self.finish_time - self.first_token_time) / (self.tokens_generated - 1)
        return None

    def is_complete(self) -> bool:
        return self.tokens_generated >= self.max_output


@dataclass
class SimConfig:
    """Simulation configuration."""
    # Hardware
    gpu_tpot_single:  float = 0.020    # seconds per token at batch=1 (20ms)
    gpu_max_batch:    int   = 64       # maximum batch size
    # Memory
    gpu_kv_budget:    int   = 500_000  # total token slots in KV cache
    # Simulation
    sim_duration:     float = 60.0     # seconds to simulate
    request_rate:     float = 10.0     # requests per second (Poisson)
    avg_prompt_len:   int   = 128
    avg_output_len:   int   = 256
    seed:             int   = 42


class ContinuousBatchingScheduler:
    """
    Simulates a continuous batching inference scheduler.

    At each decode step:
        1. Remove completed sequences
        2. Add new requests from the queue (if memory available)
        3. Process one decode step for all active sequences
        4. Track timing and utilisation
    """

    def __init__(self, cfg: SimConfig):
        self.cfg      = cfg
        self.rng      = random.Random(cfg.seed)
        self.time     = 0.0
        self.active   = []     # active Request objects
        self.queue    = []     # waiting requests (sorted by arrival time)
        self.done     = []     # completed requests
        self.kv_used  = 0      # tokens currently in KV cache

        self.stats = {
            "steps": [],
            "batch_sizes": [],
            "gpu_util": [],
            "kv_util": [],
        }

    def generate_requests(self, n: int) -> list[Request]:
        """Generate n random requests with Poisson inter-arrivals."""
        requests = []
        t = 0.0
        for i in range(n):
            t += self.rng.expovariate(self.cfg.request_rate)
            if t > self.cfg.sim_duration:
                break
            prompt_len  = max(10, int(self.rng.gauss(self.cfg.avg_prompt_len, 50)))
            output_len  = max(1,  int(self.rng.gauss(self.cfg.avg_output_len, 80)))
            requests.append(Request(i, prompt_len, output_len, arrival_time=t))
        return sorted(requests, key=lambda r: r.arrival_time)

    def tpot_for_batch(self, batch_size: int) -> float:
        """
        Time per output token as a function of batch size.
        At small batches: memory-bound → linear improvement with batch size.
        At large batches: compute-bound → saturates.
        """
        if batch_size == 0:
            return self.cfg.gpu_tpot_single
        # Memory-bound regime: TPOT ∝ 1/batch (linear speedup)
        # Compute-bound regime: saturates at ~batch=32 (approx)
        compute_bound_batch = 32
        effective_batch = min(batch_size, compute_bound_batch)
        return self.cfg.gpu_tpot_single / effective_batch

    def admit_requests(self):
        """Admit waiting requests into the active batch if memory allows."""
        admitted = 0
        while self.queue:
            req = self.queue[0]
            # Estimate memory needed: prompt + max_output
            mem_needed = req.prompt_tokens + req.max_output
            if self.kv_used + mem_needed <= self.cfg.gpu_kv_budget:
                req       = self.queue.pop(0)
                req.start_time = self.time
                self.active.append(req)
                self.kv_used += req.prompt_tokens  # prefill fills KV immediately
                admitted += 1
            else:
                break  # Queue head doesn't fit; don't skip ahead
        return admitted

    def step(self, pending_arrivals: list[Request]) -> bool:
        """
        Execute one decode step (or prefill, simplified).
        Returns True if there was work to do.
        """
        # Accept new arrivals at this time
        for req in pending_arrivals:
            if req.arrival_time <= self.time:
                self.queue.append(req)

        # Admit requests from queue
        self.admit_requests()

        if not self.active:
            return False

        # Compute one decode step for all active sequences
        batch_size = len(self.active)
        step_time  = self.tpot_for_batch(batch_size)

        for req in self.active:
            if req.first_token_time is None:
                req.first_token_time = self.time
            req.tokens_generated += 1
            self.kv_used         += 1   # one new KV token per request

        self.time += step_time

        # Track stats
        gpu_util = min(1.0, batch_size / self.cfg.gpu_max_batch)
        self.stats["steps"].append(self.time)
        self.stats["batch_sizes"].append(batch_size)
        self.stats["gpu_util"].append(gpu_util)
        self.stats["kv_util"].append(self.kv_used / self.cfg.gpu_kv_budget)

        # Remove completed sequences
        completed = [r for r in self.active if r.is_complete()]
        for req in completed:
            req.finish_time  = self.time
            self.kv_used    -= (req.prompt_tokens + req.tokens_generated)
            self.done.append(req)
        self.active = [r for r in self.active if not r.is_complete()]

        return True

    def run(self, requests: list[Request]) -> dict:
        """Run the simulation to completion."""
        remaining = sorted(requests, key=lambda r: r.arrival_time)

        while remaining or self.active or self.queue:
            # Find requests that have arrived by current time
            arrived = [r for r in remaining if r.arrival_time <= self.time + 0.001]
            remaining = [r for r in remaining if r.arrival_time > self.time + 0.001]

            if not self.step(arrived) and not remaining:
                if self.queue:
                    # Fast-forward to next arrival
                    self.time = self.queue[0].arrival_time
                else:
                    break

        return self.summary()

    def summary(self) -> dict:
        if not self.done:
            return {}
        ttfts   = [r.ttft  for r in self.done if r.ttft  is not None]
        tpots   = [r.tpot_avg for r in self.done if r.tpot_avg is not None]
        total_t = sum(r.tokens_generated for r in self.done)
        elapsed = max(r.finish_time for r in self.done) if self.done else 1.0
        bs      = self.stats["batch_sizes"]
        return {
            "n_completed":     len(self.done),
            "total_tokens":    total_t,
            "throughput_tps":  total_t / elapsed,
            "avg_ttft_ms":     1000 * sum(ttfts) / len(ttfts) if ttfts else 0,
            "avg_tpot_ms":     1000 * sum(tpots) / len(tpots) if tpots else 0,
            "p99_ttft_ms":     1000 * sorted(ttfts)[int(len(ttfts)*0.99)] if ttfts else 0,
            "avg_gpu_util":    sum(self.stats["gpu_util"]) / len(self.stats["gpu_util"])
                               if self.stats["gpu_util"] else 0,
            "avg_batch_size":  sum(bs) / len(bs) if bs else 0,
            "sim_duration_s":  elapsed,
        }


class StaticBatchingScheduler:
    """
    Simulates static batching for comparison.
    Collects requests until batch is full, then processes all until done.
    """

    def __init__(self, cfg: SimConfig, batch_size: int = 8):
        self.cfg        = cfg
        self.batch_size = batch_size
        self.rng        = random.Random(cfg.seed + 1)

    def run(self, requests: list[Request]) -> dict:
        """Simulate static batching."""
        done     = []
        time     = 0.0
        pending  = sorted(requests, key=lambda r: r.arrival_time)

        while pending:
            # Collect batch_size requests
            batch     = pending[:self.batch_size]
            pending   = pending[self.batch_size:]
            max_output = max(r.max_output for r in batch)
            max_prompt = max(r.prompt_tokens for r in batch)

            # All requests start at the batch start time
            batch_start = max(r.arrival_time for r in batch)
            time = max(time, batch_start)

            for req in batch:
                req.start_time      = time
                req.first_token_time = time   # prefill immediately

            # Process until the longest response is done
            for step in range(max_output):
                tpot = self.cfg.gpu_tpot_single / self.batch_size
                time += tpot
                for req in batch:
                    if not req.is_complete():
                        req.tokens_generated += 1

            for req in batch:
                req.finish_time = time
                done.append(req)

        ttfts   = [r.ttft for r in done if r.ttft is not None]
        tpots   = [r.tpot_avg for r in done if r.tpot_avg is not None]
        total_t = sum(r.tokens_generated for r in done)

        return {
            "n_completed":   len(done),
            "total_tokens":  total_t,
            "throughput_tps": total_t / time if time > 0 else 0,
            "avg_ttft_ms":   1000 * sum(ttfts) / len(ttfts) if ttfts else 0,
            "avg_tpot_ms":   1000 * sum(tpots) / len(tpots) if tpots else 0,
            "p99_ttft_ms":   1000 * sorted(ttfts)[int(len(ttfts)*0.99)] if ttfts else 0,
            "avg_gpu_util":  min(1.0, self.batch_size / self.cfg.gpu_max_batch),
            "avg_batch_size": self.batch_size,
            "sim_duration_s": time,
        }


if __name__ == "__main__":
    cfg      = SimConfig(request_rate=5.0, sim_duration=30.0,
                          avg_prompt_len=100, avg_output_len=200, seed=0)
    sched    = ContinuousBatchingScheduler(cfg)
    requests = sched.generate_requests(500)

    print("=" * 65)
    print("  CONTINUOUS vs STATIC BATCHING COMPARISON")
    print(f"  {len(requests)} requests, rate={cfg.request_rate}/s, "
          f"avg_output={cfg.avg_output_len} tokens")
    print("=" * 65)
    print()

    # Run both schedulers
    import copy
    reqs_cb = copy.deepcopy(requests)
    reqs_sb = copy.deepcopy(requests)

    result_cb = sched.run(reqs_cb)

    sb_sched  = StaticBatchingScheduler(cfg, batch_size=8)
    result_sb = sb_sched.run(reqs_sb)

    metrics = ["n_completed", "throughput_tps", "avg_ttft_ms", "avg_tpot_ms",
               "p99_ttft_ms", "avg_gpu_util", "avg_batch_size", "sim_duration_s"]

    print(f"  {'Metric':<25}  {'Continuous':>16}  {'Static (B=8)':>14}  {'Advantage':>12}")
    print(f"  {'':─<25}  {'':─>16}  {'':─>14}  {'':─>12}")

    for m in metrics:
        cb_val = result_cb.get(m, 0)
        sb_val = result_sb.get(m, 0)
        if m in ("throughput_tps", "avg_gpu_util", "avg_batch_size", "n_completed"):
            better = "CB ✓" if cb_val >= sb_val else "Static"
        else:
            better = "CB ✓" if cb_val <= sb_val else "Static"
        if m.endswith("_ms"):
            print(f"  {m:<25}  {cb_val:>15.1f}ms  {sb_val:>13.1f}ms  {better:>12}")
        elif m in ("avg_gpu_util",):
            print(f"  {m:<25}  {cb_val:>15.1%}  {sb_val:>13.1%}  {better:>12}")
        elif m == "throughput_tps":
            speedup = cb_val / sb_val if sb_val > 0 else 1.0
            print(f"  {m:<25}  {cb_val:>14.0f}/s  {sb_val:>12.0f}/s  "
                  f"{speedup:.1f}× faster")
        else:
            print(f"  {m:<25}  {cb_val:>16}  {sb_val:>14}  {better:>12}")

    print()
    print("  Key insight: continuous batching achieves higher throughput while")
    print("  dramatically reducing average TTFT (no head-of-line blocking).")
''',
    },

    "PagedAttention and Memory Utilisation": {
        "description": "Simulate PagedAttention's block-based KV memory management — showing fragmentation elimination, prefix sharing, and memory utilisation vs contiguous allocation.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
PAGEDATTENTION: BLOCK TABLE SIMULATION
================================================================================

Simulates PagedAttention's virtual KV memory management:
    1. Physical pages allocated from a pool (fixed-size KV blocks)
    2. Block tables mapping logical sequence positions to physical pages
    3. Prefix sharing: multiple sequences reuse the same physical pages
    4. Memory fragmentation analysis: paged vs contiguous allocation
    5. KV cache hit rate simulation with prefix sharing

PagedAttention is analogous to virtual memory in operating systems:
    Physical pages = GPU HBM pages
    Block table    = page table
    Page eviction  = KV swap to CPU (for lower-priority requests)

================================================================================
"""

import random
from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class KVBlock:
    """One physical KV cache block (page)."""
    block_id:   int
    capacity:   int       # tokens per block (e.g., 16)
    in_use:     bool = False
    ref_count:  int  = 0  # number of sequences referencing this block
    token_ids:  tuple = field(default_factory=tuple)  # content fingerprint for prefix sharing

    @property
    def is_full(self) -> bool:
        return len(self.token_ids) == self.capacity


class BlockAllocator:
    """
    GPU memory allocator for KV cache blocks.
    Implements the core of PagedAttention's memory management.
    """

    def __init__(self, total_blocks: int, block_capacity: int = 16):
        self.block_capacity = block_capacity
        self.blocks = {
            i: KVBlock(i, block_capacity)
            for i in range(total_blocks)
        }
        self.free_blocks: set[int] = set(range(total_blocks))
        self.used_blocks: dict[int, int] = {}   # block_id → sequence_id

        # Prefix sharing: token sequence hash → block_id
        self.prefix_cache: dict[str, int] = {}

    def total_blocks(self) -> int:
        return len(self.blocks)

    def free_count(self) -> int:
        return len(self.free_blocks)

    def utilisation(self) -> float:
        used = sum(1 for b in self.blocks.values() if b.in_use)
        return used / len(self.blocks)

    def allocate(self) -> Optional[int]:
        """Allocate a free physical block. Returns block_id or None."""
        if not self.free_blocks:
            return None
        block_id = next(iter(self.free_blocks))
        self.free_blocks.remove(block_id)
        self.blocks[block_id].in_use = True
        self.blocks[block_id].ref_count = 1
        return block_id

    def free(self, block_id: int) -> None:
        """Return a block to the free pool."""
        blk = self.blocks[block_id]
        blk.ref_count -= 1
        if blk.ref_count <= 0:
            blk.in_use     = False
            blk.ref_count  = 0
            blk.token_ids  = ()
            self.free_blocks.add(block_id)
            # Remove from prefix cache
            self.prefix_cache = {
                k: v for k, v in self.prefix_cache.items()
                if v != block_id
            }

    def get_or_allocate_shared(self, token_ids: tuple) -> Optional[int]:
        """
        Return an existing block if these token_ids are cached (prefix sharing),
        otherwise allocate a new block and cache it.
        """
        key = str(token_ids)
        if key in self.prefix_cache:
            block_id = self.prefix_cache[key]
            self.blocks[block_id].ref_count += 1
            return block_id
        block_id = self.allocate()
        if block_id is not None:
            self.blocks[block_id].token_ids = token_ids
            self.prefix_cache[key] = block_id
        return block_id


@dataclass
class SequenceState:
    """State of one sequence being processed."""
    seq_id:      int
    tokens:      list[int]     # all tokens so far (prompt + generated)
    block_table: list[int]     # physical block IDs, in order
    current_block: int = -1   # which physical block we're currently filling


class PagedAttentionSimulator:
    """
    Simulates PagedAttention block management for multiple sequences.
    """

    def __init__(self, n_physical_blocks: int = 100,
                 block_capacity: int = 16,
                 n_layers: int = 80, n_heads: int = 64,
                 d_head: int = 128, dtype_bytes: int = 2):
        self.allocator      = BlockAllocator(n_physical_blocks, block_capacity)
        self.block_capacity = block_capacity
        self.sequences: dict[int, SequenceState] = {}

        # Memory per block in bytes
        self.bytes_per_block = (2 * n_layers * n_heads * d_head * dtype_bytes
                                 * block_capacity)

    def start_sequence(self, seq_id: int, prompt_tokens: list[int],
                        use_prefix_sharing: bool = False) -> bool:
        """
        Begin processing a new sequence.
        Returns True if enough blocks are available.
        """
        block_table = []
        current_block = -1

        for i, tok in enumerate(prompt_tokens):
            if i % self.block_capacity == 0:
                # Need a new block
                if use_prefix_sharing and i + self.block_capacity <= len(prompt_tokens):
                    # Can we share a prefix block?
                    chunk = tuple(prompt_tokens[i:i+self.block_capacity])
                    block_id = self.allocator.get_or_allocate_shared(chunk)
                else:
                    block_id = self.allocator.allocate()

                if block_id is None:
                    # OOM: free already allocated blocks and fail
                    for bid in block_table:
                        self.allocator.free(bid)
                    return False
                block_table.append(block_id)
                current_block = block_id

        self.sequences[seq_id] = SequenceState(
            seq_id=seq_id, tokens=list(prompt_tokens),
            block_table=block_table, current_block=current_block
        )
        return True

    def add_token(self, seq_id: int, token: int) -> bool:
        """Add one generated token to a sequence."""
        seq = self.sequences[seq_id]
        seq.tokens.append(token)
        t = len(seq.tokens) - 1

        # If we've filled the current block, allocate a new one
        if t % self.block_capacity == 0:
            block_id = self.allocator.allocate()
            if block_id is None:
                return False   # OOM
            seq.block_table.append(block_id)
            seq.current_block = block_id
        return True

    def end_sequence(self, seq_id: int) -> None:
        """Release all blocks for a completed sequence."""
        if seq_id in self.sequences:
            seq = self.sequences.pop(seq_id)
            for block_id in seq.block_table:
                self.allocator.free(block_id)


# ── Memory fragmentation comparison ──────────────────────────────────────────

def simulate_contiguous_allocation(n_max_blocks: int, block_capacity: int,
                                    requests: list[dict]) -> dict:
    """
    Simulate contiguous allocation (standard approach, no PagedAttention).
    Each sequence needs a contiguous memory region of max_output_len tokens.
    Returns memory utilisation statistics.
    """
    rng = random.Random(123)
    total_capacity = n_max_blocks * block_capacity
    free_regions   = [(0, total_capacity)]   # (start, length) free regions
    active_regions = []   # (start, length, tokens_used)
    waste_total    = 0
    allocations    = 0

    for req in requests:
        max_len      = req["prompt"] + req["output"]
        needed_space = max_len   # pre-allocate max

        # Find first-fit free region
        found = False
        for i, (start, length) in enumerate(free_regions):
            if length >= needed_space:
                active_regions.append((start, needed_space, req["output"]))
                free_regions[i] = (start + needed_space, length - needed_space)
                if free_regions[i][1] == 0:
                    free_regions.pop(i)
                waste_total  += needed_space - req["output"]  # unfilled region
                allocations  += 1
                found = True
                break

        if not found:
            # Fragmentation: couldn't fit despite maybe having enough total
            pass

        # Occasionally free some sequences (simulate completions)
        if rng.random() < 0.3 and active_regions:
            idx = rng.randrange(len(active_regions))
            s, l, used = active_regions.pop(idx)
            # Return region to free list (fragmented — not merged with neighbours)
            free_regions.append((s, l))
            free_regions.sort()

    total_free = sum(l for _, l in free_regions)
    util = 1.0 - (total_free + waste_total) / total_capacity

    return {
        "utilisation":   max(0, util),
        "wasted_tokens": waste_total,
        "allocations":   allocations,
        "fragmentation": total_free / total_capacity,
    }


if __name__ == "__main__":
    import random as rng_std

    # LLaMA-2 7B KV cache configuration (approximately)
    N_LAYERS, N_HEADS, D_HEAD = 32, 32, 128
    DTYPE_BYTES               = 2
    BLOCK_CAP                 = 16   # tokens per block
    N_PHYSICAL_BLOCKS         = 200  # total physical blocks (simulated)

    print("=" * 65)
    print("  PAGEDATTENTION SIMULATION")
    print(f"  {N_PHYSICAL_BLOCKS} physical blocks × {BLOCK_CAP} tokens/block")
    print("=" * 65)
    print()

    sim = PagedAttentionSimulator(
        n_physical_blocks = N_PHYSICAL_BLOCKS,
        block_capacity    = BLOCK_CAP,
        n_layers          = N_LAYERS,
        n_heads           = N_HEADS,
        d_head            = D_HEAD,
    )

    bytes_per_block = sim.bytes_per_block
    print(f"  Bytes per KV block: {bytes_per_block/1e6:.1f} MB")
    print(f"  Total KV memory:    "
          f"{N_PHYSICAL_BLOCKS * bytes_per_block / 1e9:.1f} GB")
    print()

    # Simulate several sequences
    SYSTEM_PROMPT = list(range(10, 26))  # 16 tokens (exactly 1 block)
    rng_std.seed(42)

    print("  SCENARIO 1: No prefix sharing")
    for seq_id in range(5):
        prompt_len = rng_std.randint(20, 80)
        prompt     = SYSTEM_PROMPT + [rng_std.randint(30, 200) for _ in range(prompt_len)]
        ok = sim.start_sequence(seq_id, prompt, use_prefix_sharing=False)
        print(f"    Seq {seq_id}: {len(prompt)} tok prompt → "
              f"{'✓ allocated' if ok else '✗ OOM'} "
              f"({len(sim.sequences[seq_id].block_table) if ok else 0} blocks)")

    util1 = sim.allocator.utilisation()
    print(f"  Block utilisation (no sharing): {util1:.1%}")
    for sid in list(sim.sequences.keys()):
        sim.end_sequence(sid)

    print()
    print("  SCENARIO 2: With prefix sharing (same 16-token system prompt)")
    for seq_id in range(5, 10):
        prompt_len = rng_std.randint(20, 80)
        prompt     = SYSTEM_PROMPT + [rng_std.randint(30, 200) for _ in range(prompt_len)]
        ok = sim.start_sequence(seq_id, prompt, use_prefix_sharing=True)
        n_blocks = len(sim.sequences[seq_id].block_table) if ok else 0
        print(f"    Seq {seq_id}: {len(prompt)} tok prompt → "
              f"{'✓ allocated' if ok else '✗ OOM'} "
              f"({n_blocks} blocks, prefix shared: "
              f"{'yes' if ok and sim.allocator.blocks[sim.sequences[seq_id].block_table[0]].ref_count > 1 else 'no'})")

    util2 = sim.allocator.utilisation()
    saved_block = len([b for b in sim.allocator.blocks.values()
                        if b.in_use and b.ref_count > 1])
    print(f"  Block utilisation (with sharing): {util2:.1%}")
    print(f"  Blocks shared (ref_count > 1): {saved_block}")
    print(f"  Memory saved by prefix sharing: "
          f"{saved_block * bytes_per_block / 1e6:.0f} MB")
    for sid in list(sim.sequences.keys()):
        sim.end_sequence(sid)

    # Fragmentation comparison
    print()
    print("=" * 65)
    print("  FRAGMENTATION: CONTIGUOUS vs PAGED ALLOCATION")
    print("=" * 65)
    print()

    rng_std.seed(0)
    requests = [
        {"prompt": rng_std.randint(20, 100), "output": rng_std.randint(50, 400)}
        for _ in range(30)
    ]

    contiguous_result = simulate_contiguous_allocation(
        N_PHYSICAL_BLOCKS, BLOCK_CAP, requests
    )

    print(f"  {'Metric':<30}  {'Paged (PagedAttn)':>20}  "
          f"{'Contiguous':>18}")
    print(f"  {'':─<30}  {'':─>20}  {'':─>18}")
    print(f"  {'Memory utilisation':<30}  {'~98% typical':>20}  "
          f"{contiguous_result['utilisation']:>17.1%}")
    print(f"  {'Fragmentation':<30}  {'< 5% (1 block)':>20}  "
          f"{contiguous_result['fragmentation']:>17.1%}")
    print(f"  {'Prefix sharing':<30}  {'Yes ✓':>20}  {'No ✗':>18}")
    print(f"  {'KV swap to CPU':<30}  {'Yes (block-level)':>20}  "
          f"{'No (too complex)':>18}")
    print(f"  {'Max concurrent users':<30}  {'~100% of budget':>20}  "
          f"{'~50-70% typical':>18}")
    print()
    print("  PagedAttention nearly eliminates memory waste from fragmentation,")
    print("  increasing effective concurrency by 2–3× on real workloads.")
''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 600
    # try:
    #     from llm_training.visuals.inference_serving import (
    #         INFER_VISUAL_HTML,
    #         INFER_VISUAL_HEIGHT,
    #     )
    #     visual_html   = INFER_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = INFER_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[31_inference_optimization_serving.py] Could not load visual: {e}",
    #         stacklevel=2,
    #     )

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    COMPLEXITY,
        "operations":    OPERATIONS,
    }