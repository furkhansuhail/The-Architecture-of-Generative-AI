"""
Data Loading, Batching, and Sequence Packing
=============================================

The data pipeline between storage and the GPU is a frequently neglected
bottleneck. A poorly written DataLoader can starve a cluster of 512 A100s,
reducing GPU utilisation to 30% while terabytes of preprocessed data sit
idle on disk. This module covers efficient data loading, the mechanics of
DistributedSampler, and sequence packing — the technique that achieves
near-100% token utilisation by eliminating padding waste.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Data Loading, Batching, and Sequence Packing"
DISPLAY_NAME = "14 · Data Loading & Packing"
ICON         = "📦"
SUBTITLE     = "Efficient DataLoaders, Sequence Packing"


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

### The Data Loading Bottleneck

Modern A100 GPUs can process millions of tokens per second in the forward
pass. For this throughput to be sustained, data must arrive at the GPU
at the same rate. A DataLoader that reads from disk, tokenises on the fly,
or has insufficient prefetching will create a starvation pipeline where
the GPU sits idle waiting for the next batch.

**Measuring the bottleneck:**
Run training with `torch.profiler` or simply compare:
    •   GPU utilisation ≈ 100%: compute-bound (ideal)
    •   GPU utilisation < 80%: likely I/O or data loading bound
    •   Loss of > 10% GPU time: investigate DataLoader workers

**Key DataLoader settings for LLM training:**
    num_workers:     4–16 per GPU (depends on CPU cores available)
    prefetch_factor: 2–4  (batches to prefetch per worker)
    pin_memory:      True  (pinned CPU memory → faster GPU transfer)
    persistent_workers: True  (avoid worker respawn overhead)


### The Padded Batching Problem

Variable-length sequences in a batch must be the same length to be
processed as a matrix. The naive solution is padding:

    Sequence 1: [t0, t1, t2, PAD, PAD, PAD]   (length 3, padded to 6)
    Sequence 2: [t0, t1, t2, t3, PAD, PAD]    (length 4, padded to 6)
    Sequence 3: [t0, t1, t2, t3, t4, t5]      (length 6, no padding)

PAD tokens consume:
    •   Memory (stored in the batch tensor)
    •   Compute (attention is computed for PAD positions, then masked out)
    •   Training signal (no loss computed for PAD, but still wastes FLOPs)

For a dataset with high variance in document length, padding efficiency
can drop to 40–60% — meaning 40–60% of GPU compute is wasted.

**Sorting by length (bucketing):** Group sequences of similar length
into buckets. Sequences within a batch come from the same bucket,
minimising padding within each batch. Sorting efficiency depends on
the length distribution of the dataset.


### Sequence Packing — The LLM Standard

**Packing** eliminates padding entirely by concatenating multiple
documents end-to-end to fill each context window completely:

    Context window = 4096 tokens

    Documents:        doc1 (1024 tok)    doc2 (512 tok)    doc3 (2560 tok)
    Packed sequence:  [doc1 | EOS | doc2 | EOS | doc3_start]

    Token utilisation: 4096/4096 = 100%

Every GPU compute cycle is spent on real tokens, not padding. This is
why packing is universally used in LLM pre-training.

**The document boundary problem:**
When doc1 and doc2 are packed into the same sequence, we must ensure
that attention from doc2 tokens does not attend to doc1 tokens — the
documents are semantically unrelated. This is handled by a 2D attention
mask that blocks cross-document attention:

    Position: [0,1,2,...,1023 | 1024,...,1535 | 1536,...,4095]
    Docs:     [  doc1        |     doc2       |    doc3_start  ]

    Attention mask:
    - Tokens in doc1 attend only to other doc1 tokens (positions 0–1023)
    - Tokens in doc2 attend only to other doc2 tokens (1024–1535)
    - Tokens in doc3 attend only to other doc3 tokens (1536–4095)
    - Cross-document attention is blocked

This requires a **block-diagonal causal mask** rather than the standard
lower-triangular causal mask.


    **Diagram 1 — Padded vs Packed Batching:**

    PADDED vs PACKED: TOKEN UTILISATION
    ════════════════════════════════════════════════════════════════

    3 documents: lengths 600, 300, 1200 tokens. Context = 2048 tokens.

    PADDED APPROACH (3 separate sequences, each padded to 1200):
    ┌──────────────────────────────────────────────────────────┐
    │ doc1 (600)  │    PAD (600)   │                           │
    ├──────────────────────────────────────────────────────────┤
    │ doc2 (300)  │         PAD (900)        │                 │
    ├──────────────────────────────────────────────────────────┤
    │          doc3 (1200)              │                      │
    └──────────────────────────────────────────────────────────┘
    Useful tokens: 600+300+1200 = 2100 / (3×1200) = 58.3% utilisation

    PACKED APPROACH (1 sequence of 2048):
    ┌──────────────────────────────────────────────────────────┐
    │ doc1(600) EOS doc2(300) EOS doc3(1200) EOS doc...(rest) │
    └──────────────────────────────────────────────────────────┘
    Useful tokens: 2048/2048 = 100% utilisation ✓


    **Diagram 2 — Block-Diagonal Causal Attention Mask for Packed Sequences:**

    ATTENTION MASK FOR PACKED SEQUENCE [doc1 | doc2 | doc3]
    ════════════════════════════════════════════════════════════════

    Standard causal (lower triangular) — WRONG for packing:
    ┌─────────────────┐
    │ 1 0 0 0 0 0 0 0 │  token 0 attends to itself only
    │ 1 1 0 0 0 0 0 0 │  token 1 attends to 0,1
    │ 1 1 1 0 0 0 0 0 │  doc1 boundary
    │ 1 1 1 1 0 0 0 0 │  ← token 3 (doc2!) attends to doc1 tokens! WRONG
    │ ...             │
    └─────────────────┘

    Block-diagonal causal — CORRECT for packing (doc1=3tok, doc2=3tok):
    ┌─────────────────┐
    │ 1 0 0 0 0 0 0 0 │  doc1 tokens
    │ 1 1 0 0 0 0 0 0 │
    │ 1 1 1 0 0 0 0 0 │
    │ 0 0 0 1 0 0 0 0 │  doc2 tokens (blocked from doc1)
    │ 0 0 0 1 1 0 0 0 │
    │ 0 0 0 1 1 1 0 0 │
    │ 0 0 0 0 0 0 1 0 │  doc3
    │ 0 0 0 0 0 0 1 1 │
    └─────────────────┘
    Zeros at cross-document positions → -∞ before softmax → 0 attention weight


### DistributedSampler — Partitioning Data Across Ranks

When training with N GPUs (N ranks), each rank must process a different
shard of the data — otherwise every GPU would see identical data, learning
nothing from the additional GPUs.

**DistributedSampler** partitions the dataset by assigning each rank a
disjoint subset of indices:
    •   Dataset size D, N ranks
    •   Rank r processes indices: r, r+N, r+2N, r+3N, ...
    •   (After shuffling with the same seed across all ranks)

Every epoch, the sampler reshuffles with `seed = epoch_number`, so the
assignment is different each epoch (but deterministic for reproducibility).

**Critical:** Call `sampler.set_epoch(epoch)` at the start of each epoch.
If you forget, every epoch uses the same shuffle → data is seen in the
same order → no diversity → slower convergence.

**Shard-based loading:** For very large datasets (15T tokens), a per-file
sharding approach is used instead of DistributedSampler:
    •   Each rank is assigned a set of binary shard files
    •   Each worker within a rank streams from a different shard
    •   Shuffling happens at the file/shard level (not individual sequences)

This avoids the overhead of a shared index across N×W workers (N GPUs ×
W workers per GPU).


### Prefetching and Pipeline Depth

The data loading pipeline should be pipelined so that CPU data preparation
and GPU computation run concurrently:

    CPU Worker 0:  [preparing batch t+2]  [preparing batch t+3]  ...
    CPU Worker 1:  [preparing batch t+1]  [preparing batch t+2]  ...
    GPU:           [processing batch t]   [processing batch t+1] ...
    Transfer:      [transferring t+1]     [transferring t+2]     ...

With prefetch_factor=2 and num_workers=4, the DataLoader maintains a
queue of 8 prefetched batches (4 workers × 2 per worker). If processing
one batch takes 100 ms, the DataLoader must prepare batches in < 100 ms
to avoid stalling the GPU.

**CUDA streams for overlap:** Explicitly transfer the next batch to GPU
while the current batch is being processed:

    stream = torch.cuda.Stream()

    for batch in dataloader:
        with torch.cuda.stream(stream):
            batch_gpu = batch.to("cuda", non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream)
        loss = model(batch_gpu)
        ...


### Memory-Mapped Datasets

For datasets too large to fit in RAM (common for 15T-token corpora), use
**memory-mapped files** via `numpy.memmap`:

    data = np.memmap("tokens.bin", dtype=np.uint16, mode="r",
                     shape=(n_tokens,))

The OS manages page loading on demand — individual pages are loaded from
disk only when accessed. Unused pages are evicted from RAM as needed.

Benefits:
    •   Zero copy: data is accessed directly in the OS page cache
    •   Random access: any sequence can be accessed in O(1) time
    •   Multiple processes can share the same page cache (efficient for DDP)
    •   No startup cost: no need to load data before training begins

Limitation:
    •   Cannot shuffle at the token level without copying
    •   Sequential access is much faster than random access (disk locality)


### Online vs Offline Tokenisation

**Offline (pre-tokenised, standard for pre-training):**
    •   Tokenise the entire dataset once before training
    •   Store token IDs as binary arrays on disk
    •   DataLoader reads token IDs directly — no tokenisation overhead
    •   Requires ~2× disk space vs raw text (token IDs + raw text)

**Online (tokenise on the fly):**
    •   Tokenise each batch at training time
    •   Simpler pipeline but slower (tokenisation is CPU-bound)
    •   Acceptable for fine-tuning on small datasets (< 1M tokens)
    •   Impractical for pre-training (15T tokens at 100k tok/s = 42 hours)

Rule of thumb: pre-tokenise for anything > 1B tokens.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Batching Strategy Comparison

| Strategy              | Token utilisation | Implementation complexity | Best for                    |
|-----------------------|-------------------|---------------------------|-----------------------------|
| Padded (fixed length) | 40–80%            | Simple                    | Inference, short sequences  |
| Sorted/bucketed       | 70–90%            | Moderate                  | Fine-tuning                 |
| Sequence packing      | ~100%             | Moderate (needs 2D mask)  | Pre-training (standard)     |
| Dynamic batching      | ~95%              | High                      | Variable-length inference   |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Sequence Packing with Block-Diagonal Mask": {
        "description": "Pack multiple documents into fixed-length sequences, build the block-diagonal causal attention mask, and verify that cross-document attention is correctly blocked.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SEQUENCE PACKING WITH BLOCK-DIAGONAL ATTENTION MASK
================================================================================

Implements sequence packing:
    1. Pack multiple documents into fixed-length sequences (greedy bin-packing)
    2. Track document boundaries within each packed sequence
    3. Build block-diagonal causal attention masks
    4. Verify that cross-document attention is zero

This is the exact packing strategy used in LLaMA-3, Mistral, and most
modern LLM pre-training pipelines.
================================================================================
"""

import torch
import torch.nn.functional as F


# ── Document packing ──────────────────────────────────────────────────────────

def pack_documents(token_lists: list[list[int]],
                   seq_len: int,
                   eos_id: int = 2,
                   discard_incomplete: bool = True) -> list[dict]:
    """
    Pack variable-length token lists into fixed-length sequences.

    Uses a simple greedy algorithm:
        1. Maintain a current buffer
        2. For each document, add it to the buffer (with EOS separator)
        3. When buffer reaches seq_len, start a new sequence
        4. If a document doesn't fit in the current buffer, start a new one
           (no document spans across sequences — cleaner for attention)

    Returns:
        List of dicts with keys:
            "tokens":     list[int] of length seq_len
            "doc_ends":   list[int] — positions of EOS tokens
            "n_docs":     int — number of complete documents in this sequence
    """
    packed_seqs = []
    buffer      = []
    doc_ends    = []

    for tokens in token_lists:
        doc_with_eos = tokens + [eos_id]

        # If this document doesn't fit in the current buffer, flush first
        if len(buffer) + len(doc_with_eos) > seq_len:
            # Flush current buffer as a sequence
            if len(buffer) > 0:
                # Pad to seq_len if needed (rare edge case)
                if len(buffer) < seq_len:
                    if discard_incomplete:
                        buffer = []
                        doc_ends = []
                        continue
                    buffer = buffer + [eos_id] * (seq_len - len(buffer))

                packed_seqs.append({
                    "tokens":   buffer[:seq_len],
                    "doc_ends": [e for e in doc_ends if e < seq_len],
                    "n_docs":   len([e for e in doc_ends if e < seq_len]),
                })
                buffer   = []
                doc_ends = []

        # Add document to buffer
        start = len(buffer)
        buffer.extend(doc_with_eos)
        doc_ends.append(start + len(doc_with_eos) - 1)   # position of EOS

        # If buffer is exactly full, flush
        if len(buffer) == seq_len:
            packed_seqs.append({
                "tokens":   buffer[:],
                "doc_ends": doc_ends[:],
                "n_docs":   len(doc_ends),
            })
            buffer   = []
            doc_ends = []

    # Handle remaining buffer
    if buffer and not discard_incomplete:
        buffer = buffer + [eos_id] * (seq_len - len(buffer))
        packed_seqs.append({
            "tokens":   buffer[:seq_len],
            "doc_ends": [e for e in doc_ends if e < seq_len],
            "n_docs":   len([e for e in doc_ends if e < seq_len]),
        })

    return packed_seqs


# ── Block-diagonal causal mask ────────────────────────────────────────────────

def build_block_diagonal_mask(seq_len: int,
                               doc_ends: list[int]) -> torch.Tensor:
    """
    Build a block-diagonal causal attention mask for a packed sequence.

    The mask has value 1 where attention is allowed (same document, causal)
    and 0 where attention is blocked (cross-document or future tokens).

    Shape: (seq_len, seq_len)
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    # Compute document start/end positions
    doc_boundaries = [-1] + sorted(doc_ends)

    for b in range(len(doc_boundaries) - 1):
        doc_start = doc_boundaries[b] + 1
        doc_end   = doc_boundaries[b + 1]

        if doc_start >= seq_len:
            break
        doc_end = min(doc_end, seq_len - 1)

        # Causal lower-triangular block for this document
        for i in range(doc_start, doc_end + 1):
            for j in range(doc_start, i + 1):
                if j < seq_len:
                    mask[i, j] = True

    return mask


def mask_to_bias(mask: torch.Tensor,
                 fill_value: float = float("-inf")) -> torch.Tensor:
    """
    Convert a boolean mask (1=attend, 0=block) to an additive attention bias.
    Add this to raw attention scores before softmax.
    """
    bias = torch.zeros_like(mask, dtype=torch.float32)
    bias[~mask] = fill_value
    return bias


# ── Verification ──────────────────────────────────────────────────────────────

def verify_no_cross_doc_attention(mask: torch.Tensor,
                                   doc_ends: list[int],
                                   seq_len: int) -> bool:
    """
    Verify that no token in document B attends to any token in document A
    (where A comes before B in the packed sequence).
    """
    boundaries = [-1] + sorted(doc_ends)
    all_ok     = True

    for b in range(len(boundaries) - 1):
        doc_start = boundaries[b] + 1
        doc_end   = min(boundaries[b + 1], seq_len - 1)

        for i in range(doc_start, doc_end + 1):
            for j in range(0, doc_start):
                if j < seq_len and mask[i, j]:
                    print(f"  ✗ Cross-doc attention found: pos {i} → pos {j}")
                    all_ok = False
    return all_ok


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEQ_LEN = 20
    EOS_ID  = 0

    # Sample documents (token ID lists)
    documents = [
        list(range(10, 17)),     # doc1: 7 tokens  [10,11,12,13,14,15,16]
        list(range(20, 24)),     # doc2: 4 tokens  [20,21,22,23]
        list(range(30, 39)),     # doc3: 9 tokens  [30,...,38]
        list(range(40, 44)),     # doc4: 4 tokens  [40,41,42,43]
        list(range(50, 58)),     # doc5: 8 tokens  [50,...,57]
    ]

    total_doc_tokens = sum(len(d) for d in documents)
    print("=" * 62)
    print("  SEQUENCE PACKING DEMO")
    print(f"  seq_len={SEQ_LEN}, {len(documents)} documents, "
          f"{total_doc_tokens} total tokens")
    print("=" * 62)
    print()

    packed = pack_documents(documents, SEQ_LEN, eos_id=EOS_ID)

    print(f"  Packed into {len(packed)} sequence(s):")
    total_packed = 0
    for i, seq in enumerate(packed):
        print(f"\n  Seq {i}: {seq['tokens']}")
        print(f"    EOS positions: {seq['doc_ends']}")
        print(f"    Documents:     {seq['n_docs']}")
        total_packed += seq["n_docs"]

    utilisation = total_doc_tokens / (len(packed) * SEQ_LEN) * 100
    print(f"\n  Token utilisation: {utilisation:.1f}%  "
          f"(vs {total_doc_tokens/(len(documents)*SEQ_LEN)*100:.1f}% padded)")

    print()
    print("=" * 62)
    print("  BLOCK-DIAGONAL CAUSAL MASK")
    print("=" * 62)

    for i, seq in enumerate(packed):
        mask = build_block_diagonal_mask(SEQ_LEN, seq["doc_ends"])
        print(f"\n  Sequence {i} mask ({SEQ_LEN}×{SEQ_LEN}):")
        print(f"  Tokens: {seq['tokens']}")
        print(f"  EOS at: {seq['doc_ends']}")
        print()

        # ASCII visualisation
        header = "    " + "".join(f"{j%10}" for j in range(SEQ_LEN))
        print(f"  {header}")
        for row in range(SEQ_LEN):
            row_str = "  " + f"{row:2d}│"
            for col in range(SEQ_LEN):
                row_str += "█" if mask[row, col] else "·"
            print(row_str)

        # Verify no cross-document attention
        ok = verify_no_cross_doc_attention(mask, seq["doc_ends"], SEQ_LEN)
        print(f"\n  Cross-document attention blocked: {'✓ YES' if ok else '✗ NO'}")

    # Compare standard vs block-diagonal
    print()
    print("=" * 62)
    print("  STANDARD CAUSAL vs BLOCK-DIAGONAL (first packed seq)")
    print("=" * 62)
    seq0 = packed[0]
    std_mask   = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
    block_mask = build_block_diagonal_mask(SEQ_LEN, seq0["doc_ends"])

    extra_attention = std_mask & ~block_mask   # positions allowed by std but blocked by block
    print(f"  Positions allowed by standard mask: {std_mask.sum().item()}")
    print(f"  Positions allowed by block mask:    {block_mask.sum().item()}")
    print(f"  Cross-doc positions (blocked):      {extra_attention.sum().item()}")
    print()
    print("  The block-diagonal mask blocks cross-document attention")
    print("  that the standard mask would incorrectly allow.")
''',
    },

    "DistributedSampler and Efficient DataLoader": {
        "description": "Show how DistributedSampler partitions data across ranks, implement shard-based streaming for large datasets, and benchmark DataLoader throughput.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DISTRIBUTEDSAMPLER AND EFFICIENT DATA LOADING
================================================================================

Demonstrates:
    1. How DistributedSampler assigns indices to each rank
    2. Shard-based streaming dataset for large corpora
    3. DataLoader configuration for maximum throughput
    4. Throughput benchmarking (tokens/second from disk)
================================================================================
"""

import os
import time
import math
import struct
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ── DistributedSampler mechanics ─────────────────────────────────────────────

def explain_distributed_sampler():
    """
    Show exactly how DistributedSampler assigns indices to each rank.
    """
    print("=" * 62)
    print("  DISTRIBUTEDSAMPLER INDEX ASSIGNMENT")
    print("=" * 62)
    print()

    n_samples  = 20    # total dataset size
    n_replicas = 4     # world_size (4 GPUs)
    rank_seed  = 42    # epoch = 0 seed

    # Replicate the logic from PyTorch's DistributedSampler
    import random as rnd
    rnd.seed(rank_seed)
    indices = list(range(n_samples))
    rnd.shuffle(indices)

    # Pad to make divisible by n_replicas
    n_per_rank   = math.ceil(n_samples / n_replicas)
    total_padded = n_per_rank * n_replicas
    if total_padded > n_samples:
        padding = indices[:total_padded - n_samples]
        indices = indices + padding

    print(f"  Dataset size:  {n_samples}")
    print(f"  GPUs (ranks):  {n_replicas}")
    print(f"  Samples/rank:  {n_per_rank}")
    print(f"  Total (padded):{total_padded}")
    print()
    print(f"  Shuffled indices: {indices}")
    print()

    for rank in range(n_replicas):
        rank_indices = indices[rank:total_padded:n_replicas]
        print(f"  Rank {rank}: {rank_indices}")

    print()
    print("  Key properties:")
    print("  • Each rank sees a disjoint subset of the data ✓")
    print("  • All indices are covered across all ranks ✓")
    print("  • set_epoch(epoch) changes the shuffle for next epoch ✓")
    print()

    # Show epoch-to-epoch variation
    print("  Same rank 0 assignment across epochs:")
    for epoch in range(3):
        rnd.seed(epoch)
        idxs = list(range(n_samples))
        rnd.shuffle(idxs)
        n_padded_total = n_per_rank * n_replicas
        if n_padded_total > n_samples:
            idxs = idxs + idxs[:n_padded_total - n_samples]
        rank0 = idxs[0:n_padded_total:n_replicas]
        print(f"  Epoch {epoch}: {rank0}")
    print()
    print("  IMPORTANT: Call sampler.set_epoch(epoch) each epoch!")
    print("  If you forget, the same shuffle is used every epoch.")


# ── Memory-mapped dataset ─────────────────────────────────────────────────────

class MemmapTokenDataset(Dataset):
    """
    Memory-mapped token dataset for pre-tokenised binary files.

    Expects a flat binary file of uint16 token IDs.
    Returns fixed-length sequences for the training loop.
    """

    def __init__(self, file_path: str, seq_len: int,
                 dtype: np.dtype = np.uint16):
        self.seq_len   = seq_len
        self.data      = np.memmap(file_path, dtype=dtype, mode="r")
        self.n_seqs    = len(self.data) // seq_len   # drop incomplete last chunk
        print(f"  [MemmapDataset] {file_path}: "
              f"{len(self.data):,} tokens → {self.n_seqs:,} sequences")

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> dict:
        start  = idx * self.seq_len
        tokens = torch.from_numpy(
            self.data[start : start + self.seq_len].astype(np.int64)
        )
        # Input: tokens[0:-1], Target: tokens[1:]
        return {
            "input_ids": tokens[:-1],
            "labels":    tokens[1:],
        }


# ── Shard-based streaming dataset ────────────────────────────────────────────

class ShardStreamDataset(Dataset):
    """
    Streams from multiple binary shard files, assigning shards to ranks.

    Used for very large corpora where DistributedSampler is impractical.

    Each rank processes a deterministic subset of shards:
        rank 0 → shards [0, N, 2N, ...]
        rank 1 → shards [1, N+1, 2N+1, ...]
        etc.
    """

    def __init__(self, shard_paths: list[str], seq_len: int,
                 rank: int = 0, world_size: int = 1,
                 dtype: np.dtype = np.uint16):
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size

        # Assign shards to this rank
        self.my_shards  = shard_paths[rank::world_size]
        self.all_data   = []

        for path in self.my_shards:
            data = np.memmap(path, dtype=dtype, mode="r")
            self.all_data.append(data)

        # Build cumulative sequence counts
        self.shard_sizes = [len(d) // seq_len for d in self.all_data]
        self.cumulative  = [0]
        for sz in self.shard_sizes:
            self.cumulative.append(self.cumulative[-1] + sz)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, idx: int) -> dict:
        # Find which shard this index falls into
        shard_idx = 0
        while (shard_idx + 1 < len(self.cumulative) and
               self.cumulative[shard_idx + 1] <= idx):
            shard_idx += 1

        local_idx = idx - self.cumulative[shard_idx]
        start     = local_idx * self.seq_len
        data      = self.all_data[shard_idx]
        tokens    = torch.from_numpy(
            data[start:start + self.seq_len].astype(np.int64)
        )
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


# ── DataLoader configuration and benchmarking ─────────────────────────────────

def benchmark_dataloader(dataset: Dataset, batch_size: int,
                          num_workers: int, prefetch_factor: int,
                          n_batches: int = 50) -> dict:
    """
    Benchmark DataLoader throughput: tokens/second.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=False,    # True only when CUDA is available
        persistent_workers=num_workers > 0,
        shuffle=True,
    )

    n_tokens = 0
    t0       = time.perf_counter()

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        n_tokens += batch["input_ids"].numel()

    elapsed = time.perf_counter() - t0
    return {
        "tokens_per_sec": n_tokens / elapsed,
        "batches":        min(n_batches, len(loader)),
        "elapsed_s":      elapsed,
    }


if __name__ == "__main__":
    # 1. DistributedSampler explanation
    explain_distributed_sampler()

    # 2. Create a temporary binary dataset for benchmarking
    SEQ_LEN    = 128
    N_TOKENS   = 100_000
    BATCH_SIZE = 8

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp_path = f.name
        # Write random uint16 token IDs
        data = np.random.randint(0, 32000, size=N_TOKENS, dtype=np.uint16)
        data.tofile(f)

    print("=" * 62)
    print(f"  MEMMAP DATASET BENCHMARK")
    print(f"  {N_TOKENS:,} tokens, seq_len={SEQ_LEN}, batch={BATCH_SIZE}")
    print("=" * 62)
    print()

    dataset = MemmapTokenDataset(tmp_path, SEQ_LEN)
    print()

    print(f"  {'num_workers':>12}  {'prefetch':>10}  {'tok/s':>12}  {'speedup':>10}")
    print(f"  {'':─>12}  {'':─>10}  {'':─>12}  {'':─>10}")

    baseline_tps = None
    for nw, pf in [(0, None), (1, 2), (2, 2), (4, 2), (4, 4)]:
        pf_val = pf if nw > 0 else 1
        r  = benchmark_dataloader(dataset, BATCH_SIZE, nw,
                                   pf_val if nw > 0 else 2, n_batches=50)
        tps = r["tokens_per_sec"]
        if baseline_tps is None:
            baseline_tps = tps
        speedup = tps / baseline_tps
        pf_str  = str(pf) if pf else "N/A"
        print(f"  {nw:>12}  {pf_str:>10}  {tps:>12,.0f}  {speedup:>9.2f}×")

    print()
    print("  Recommendations:")
    print("  • Use num_workers = 4–8 per GPU (match CPU cores available)")
    print("  • prefetch_factor=2 is usually sufficient")
    print("  • pin_memory=True when CUDA is available")
    print("  • persistent_workers=True to avoid worker respawn overhead")
    print()

    # 3. Show optimal DataLoader config
    print("=" * 62)
    print("  PRODUCTION DATALOADER CONFIGURATION")
    print("=" * 62)
    config_code = """
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=DistributedSampler(       # one of these two
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,               # prevents uneven last batch
        ),
        num_workers=4,                    # 4–8 per GPU
        prefetch_factor=2,                # 2× buffer per worker
        pin_memory=True,                  # GPU transfer speed
        persistent_workers=True,          # no respawn overhead
        drop_last=True,                   # consistent batch sizes
    )

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)          # ← CRITICAL: different shuffle
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device,    non_blocking=True)
            ...
    """
    print(config_code)

    os.unlink(tmp_path)
''',
    },

    "Token Utilisation Analysis and Packing Efficiency": {
        "description": "Measure the token utilisation of padded vs packed batching on a realistic document length distribution, and find the optimal packing strategy.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
TOKEN UTILISATION: PADDED vs PACKED vs SORTED-BUCKET
================================================================================

Analyses token utilisation (fraction of non-PAD tokens) for three batching
strategies on a realistic document length distribution:
    1. Naive padded (pad to max length in batch)
    2. Bucket sorted (group by length range, pad within bucket)
    3. Packed (concatenate to fill context window)

Shows why packing is the universal choice for LLM pre-training.
================================================================================
"""

import math
import random
from collections import defaultdict


# ── Synthetic document length distribution ────────────────────────────────────

def sample_doc_lengths(n: int, seed: int = 42) -> list[int]:
    """
    Sample document lengths from a realistic distribution.
    Web text has a heavy tail: many short snippets, some very long articles.
    Models as a log-normal distribution with some very short documents.
    """
    rng = random.Random(seed)
    lengths = []
    for _ in range(n):
        r = rng.random()
        if r < 0.20:
            # Short snippet: 50–200 tokens
            l = int(rng.gauss(100, 40))
        elif r < 0.60:
            # Medium article: 200–1500 tokens
            l = int(rng.gauss(600, 300))
        elif r < 0.90:
            # Long article: 1500–4000 tokens
            l = int(rng.gauss(2500, 700))
        else:
            # Very long document: 4000–8000 tokens
            l = int(rng.gauss(6000, 1000))
        lengths.append(max(50, min(8000, l)))
    return lengths


# ── Strategy 1: Naive padding ─────────────────────────────────────────────────

def padded_utilisation(lengths: list[int], batch_size: int) -> dict:
    """Each batch pads to the length of the longest sequence in that batch."""
    n_real  = 0
    n_total = 0

    for i in range(0, len(lengths) - batch_size + 1, batch_size):
        batch    = lengths[i : i + batch_size]
        max_len  = max(batch)
        n_real  += sum(batch)
        n_total += max_len * batch_size

    return {
        "n_real":     n_real,
        "n_total":    n_total,
        "utilisation": n_real / n_total if n_total > 0 else 0,
    }


# ── Strategy 2: Sorted / bucketed ────────────────────────────────────────────

def bucketed_utilisation(lengths: list[int], batch_size: int,
                          seq_len: int) -> dict:
    """Sort by length, group into batches of similar-length sequences."""
    sorted_lengths = sorted(lengths)
    n_real  = 0
    n_total = 0

    for i in range(0, len(sorted_lengths) - batch_size + 1, batch_size):
        batch    = sorted_lengths[i : i + batch_size]
        max_len  = min(max(batch), seq_len)  # cap at seq_len
        n_real  += sum(min(l, seq_len) for l in batch)
        n_total += max_len * batch_size

    return {
        "n_real":     n_real,
        "n_total":    n_total,
        "utilisation": n_real / n_total if n_total > 0 else 0,
    }


# ── Strategy 3: Packing ───────────────────────────────────────────────────────

def packed_utilisation(lengths: list[int], seq_len: int) -> dict:
    """
    Greedy bin-packing: fill each context window as much as possible.
    """
    n_seqs  = 0
    n_real  = 0
    current = 0

    for l in lengths:
        doc_len = min(l, seq_len)   # documents longer than seq_len get split

        if current + doc_len > seq_len:
            # Start new sequence
            n_seqs  += 1
            n_real  += current
            current  = doc_len
        else:
            current += doc_len

    # Flush last sequence
    if current > 0:
        n_seqs += 1
        n_real += current

    n_total = n_seqs * seq_len
    return {
        "n_real":     n_real,
        "n_total":    n_total,
        "n_seqs":     n_seqs,
        "utilisation": n_real / n_total if n_total > 0 else 0,
    }


# ── Analysis ──────────────────────────────────────────────────────────────────

def length_distribution_stats(lengths: list[int]) -> dict:
    n     = len(lengths)
    mean  = sum(lengths) / n
    sorted_l = sorted(lengths)
    p50   = sorted_l[n//2]
    p90   = sorted_l[int(n*0.9)]
    p99   = sorted_l[int(n*0.99)]
    return {"mean": mean, "p50": p50, "p90": p90, "p99": p99,
            "min": min(lengths), "max": max(lengths)}


def bar(frac: float, width: int = 30) -> str:
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


if __name__ == "__main__":
    N_DOCS     = 10_000
    SEQ_LEN    = 2048
    BATCH_SIZE = 16

    lengths = sample_doc_lengths(N_DOCS)
    stats   = length_distribution_stats(lengths)

    print("=" * 65)
    print(f"  DOCUMENT LENGTH DISTRIBUTION ({N_DOCS:,} documents)")
    print("=" * 65)
    print()
    print(f"  Mean:  {stats['mean']:.0f} tokens")
    print(f"  p50:   {stats['p50']:.0f} tokens")
    print(f"  p90:   {stats['p90']:.0f} tokens")
    print(f"  p99:   {stats['p99']:.0f} tokens")
    print(f"  Min:   {stats['min']:.0f} tokens")
    print(f"  Max:   {stats['max']:.0f} tokens")

    # Length histogram
    print()
    buckets = [(0,200), (200,600), (600,1500), (1500,3000), (3000,8001)]
    print("  Distribution:")
    for lo, hi in buckets:
        count = sum(1 for l in lengths if lo <= l < hi)
        frac  = count / N_DOCS
        print(f"  {lo:>5}–{hi-1:<5}  {bar(frac, 25)}  {count:>5} ({frac*100:.1f}%)")

    # Utilisation comparison
    print()
    print("=" * 65)
    print(f"  TOKEN UTILISATION COMPARISON")
    print(f"  seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}")
    print("=" * 65)
    print()

    r_padded   = padded_utilisation(lengths, BATCH_SIZE)
    r_bucketed = bucketed_utilisation(lengths, BATCH_SIZE, SEQ_LEN)
    r_packed   = packed_utilisation(lengths, SEQ_LEN)

    for name, r in [("Naive padding",   r_padded),
                    ("Sorted buckets",  r_bucketed),
                    ("Sequence packing", r_packed)]:
        u = r["utilisation"]
        print(f"  {name:<22}  {bar(u)}  {u*100:.1f}%")
        print(f"    Real tokens:  {r['n_real']:>10,}")
        print(f"    Total slots:  {r['n_total']:>10,}")
        wasted = r['n_total'] - r['n_real']
        print(f"    Wasted:       {wasted:>10,}  ({wasted/r['n_total']*100:.1f}%)")
        print()

    # GPU compute waste
    pack_u = r_packed["utilisation"]
    pad_u  = r_padded["utilisation"]
    print("=" * 65)
    print("  GPU COMPUTE WASTE ANALYSIS")
    print("=" * 65)
    print()
    print(f"  With naive padding, {(1-pad_u)*100:.1f}% of GPU compute is wasted on PAD tokens.")
    print(f"  With packing,       {(1-pack_u)*100:.1f}% is wasted.")
    print()

    # Scale impact
    GPU_COST_PER_HOUR = 2.0   # $ per A100-hour
    N_GPUS            = 64
    TRAINING_DAYS     = 21

    total_gpu_hours   = N_GPUS * TRAINING_DAYS * 24
    total_cost        = total_gpu_hours * GPU_COST_PER_HOUR
    wasted_cost_pad   = total_cost * (1 - pad_u)
    wasted_cost_pack  = total_cost * (1 - pack_u)
    saved             = wasted_cost_pad - wasted_cost_pack

    print(f"  For a {TRAINING_DAYS}-day run on {N_GPUS} GPUs at ${GPU_COST_PER_HOUR}/hr:")
    print(f"    Total GPU cost:          ${total_cost:>10,.0f}")
    print(f"    Wasted on PAD (padded):  ${wasted_cost_pad:>10,.0f}")
    print(f"    Wasted on PAD (packed):  ${wasted_cost_pack:>10,.0f}")
    print(f"    Savings from packing:    ${saved:>10,.0f}")
    print()
    print(f"  → Sequence packing saves ${saved:,.0f} on this training run alone.")

    # Sensitivity to context window size
    print()
    print("=" * 65)
    print("  PACKING EFFICIENCY vs CONTEXT WINDOW SIZE")
    print("=" * 65)
    print()
    print(f"  {'Context len':>12}  {'Utilisation':>14}  {'Avg docs/seq':>14}")
    print(f"  {'':─>12}  {'':─>14}  {'':─>14}")

    for ctx in [512, 1024, 2048, 4096, 8192]:
        r   = packed_utilisation(lengths, ctx)
        avg = sum(min(l, ctx) for l in lengths) / ctx / (r["n_seqs"] if r["n_seqs"] else 1)
        print(f"  {ctx:>12,}  {r['utilisation']*100:>13.1f}%  {avg:>14.2f}")
    print()
    print("  Longer context → higher utilisation (documents fit more completely).")
    print("  Even at 512, packing is far better than naive padding.")
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
    #     from llm_training.visuals.data_loading import (
    #         DATALOAD_VISUAL_HTML,
    #         DATALOAD_VISUAL_HEIGHT,
    #     )
    #     visual_html   = DATALOAD_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = DATALOAD_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[14_data_loading_batching_packing.py] Could not load visual: {e}",
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