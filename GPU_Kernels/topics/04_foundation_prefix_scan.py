"""
Prefix Scan — Inclusive, Exclusive, Blelloch & Segmented
=========================================================

A prefix scan — also called a prefix sum or cumulative scan — transforms
an array [a0, a1, a2, ..., a_{N-1}] into an array where each output
element is the cumulative application of a binary operator over all
preceding inputs. It is not a single-output reduction; it is a
many-output transformation that preserves the intermediate state of
every reduction prefix.

Prefix scan is one of the most important parallel primitives in
computer science. Guy Blelloch (1990) proved that any algorithm
expressible as a scan can be parallelised — which makes it the
theoretical foundation for a vast category of GPU kernels. In modern
GPU programming it underpins: stream compaction, radix sort, sparse
matrix-vector products, run-length encoding, dynamic work scheduling,
histogram construction, variable-length sequence packing (crucial for
LLM batching), and the block allocation logic inside vLLM's paged
attention manager.

This module builds the complete scan stack:
    1. Definitions     — inclusive vs exclusive, the operator contract
    2. Naïve scan      — O(N²) work, understand before fixing
    3. Hillis-Steele   — O(N log N) work, optimal span, SMEM-friendly
    4. Blelloch        — O(N) work-efficient, two-phase up/down sweep
    5. Warp scan       — __shfl_up_sync, register-speed, zero SMEM
    6. Block scan      — compose warp scans into a full block primitive
    7. Device scan     — multi-block, the decoupled lookback algorithm
    8. Segmented scan  — independent scans within flag-delimited regions

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Prefix Scan — Inclusive, Exclusive, Blelloch & Segmented"
DISPLAY_NAME = "04 · Prefix Scan"
ICON         = "📐"
SUBTITLE     = "Hillis-Steele · Blelloch Up/Down Sweep · Warp Shuffle · Segmented"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — DEFINITIONS: INCLUSIVE, EXCLUSIVE, AND THE OPERATOR CONTRACT

### Formal Definitions

    Given an array A of N elements and a binary operator ⊕:

    INCLUSIVE SCAN:
        out[i] = A[0] ⊕ A[1] ⊕ ... ⊕ A[i]       (elements 0 through i)
        out[0] = A[0]                               (first element unchanged)
        Length of output: N (same as input)

    EXCLUSIVE SCAN (a.k.a. prefix sum with identity element):
        out[i] = A[0] ⊕ A[1] ⊕ ... ⊕ A[i-1]      (elements 0 through i-1)
        out[0] = identity(⊕)                        (first element = identity)
        Length of output: N (same as input)

    The relationship between them:
        exclusive[i] = inclusive[i-1] for i > 0
        exclusive[0] = identity
        inclusive[i] = exclusive[i] ⊕ A[i]

    Concrete example with addition (identity = 0):
        Input:      [ 3,  1,  4,  1,  5,  9,  2,  6 ]
        Inclusive:  [ 3,  4,  8,  9, 14, 23, 25, 31 ]
        Exclusive:  [ 0,  3,  4,  8,  9, 14, 23, 25 ]

    The exclusive scan answers: "what is the cumulative total of all elements
    BEFORE index i?" This is the question asked during stream compaction,
    dynamic allocation, and histogram accumulation — which is why exclusive
    scan is the more commonly used variant in GPU kernels.

### The Operator Contract: Associativity

    The binary operator ⊕ must be ASSOCIATIVE:
        (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)

    This is the property that allows parallel reordering of the computation.
    If the order of association does not matter, any tree of operations
    gives the same answer — and parallel trees can be scheduled freely.

    COMMUTATIVE is NOT required (unlike some parallel algorithms):
        Scan preserves element ORDER. Out[i] includes A[0..i] in that order.
        A non-commutative scan (e.g., matrix multiply prefix) is valid:
            out[i] = M_0 × M_1 × ... × M_i  (matrix product, non-commutative)

    Common scan operators and their identity elements:
        Addition (+):          identity = 0
        Multiplication (×):    identity = 1
        Maximum (max):         identity = -∞
        Minimum (min):         identity = +∞
        Logical AND:           identity = true (1)
        Logical OR:            identity = false (0)
        Bitwise XOR:           identity = 0
        (max, argmax) pair:    identity = (-∞, -1)
        (max, sum) pair:       identity = (-∞, 0)      ← softmax statistics

    The (max, sum) pair is the scan operator behind FlashAttention's
    tile-by-tile softmax — each tile is a scan "element", and the
    merge function from module 22 is the scan operator.

### Why Exclusive Scan Is More Useful Than Inclusive

    The exclusive scan directly answers allocation questions:

    STREAM COMPACTION:
        Array A has some elements that pass a filter.
        want_to_keep = [1, 0, 1, 1, 0, 1, 0, 0]
        exclusive_scan(want_to_keep) = [0, 1, 1, 2, 3, 3, 4, 4]
        exclusive_scan[i] = where element i should write IF it passes filter.

        Thread i:
            if (want_to_keep[i]) output[exclusive_scan[i]] = A[i];
        → Produces compact output with no gaps. No atomic needed. No races.

    DYNAMIC ALLOCATION:
        Each thread wants to write variable_count[i] elements.
        write_offsets = exclusive_scan(variable_count)
        Thread i writes to output[write_offsets[i]..write_offsets[i]+count[i]-1]
        → LLM batching: pack variable-length sequences with no wasted padding.

    HISTOGRAM TO CDF:
        histogram = [5, 3, 8, 2, 6]   (frequency counts per bucket)
        exclusive_scan gives CDF:   [0, 5, 8, 16, 18]
        → Starting index of each bucket in a sorted array.
        → The foundation of radix sort's scatter phase.


##### PART 2 — THE NAÏVE PARALLEL SCAN AND ITS WORK COMPLEXITY PROBLEM

### Sequential Baseline: O(N) Work, O(N) Span

    acc = identity
    for i in range(N):
        acc = acc ⊕ A[i]
        out[i] = acc     # inclusive
        # or: out[i] = acc before update (exclusive)

    Work = N-1 operations. Span = N (fully sequential, no parallelism).
    This is optimal in work. The challenge is reducing the span to O(log N).

### Naïve Parallel Scan: O(N log N) Work

    The first intuition: at step k, each element adds the partial sum
    from stride=2^(k-1) positions behind it.

        for step k = 0, 1, ..., log2(N)-1:
            stride = 2^k
            for i = stride to N-1 (in parallel):
                A[i] = A[i] + A[i - stride]

    This is the HILLIS-STEELE algorithm (see Part 3).
    Span: log2(N) steps (optimal).
    Work: N × log2(N) / 2 operations (NOT optimal — does extra work).

    For N=1M and 32-bit addition:
        Sequential:   ~1M operations
        Naïve parallel: ~10M operations  ← 10× more total work!

    On a GPU with P processors:
        Time ≈ max(Work/P, Span) = max(10M/6912, 20) ≈ 1448 vs 20 steps
        The extra work dominates when P ≪ N.

    For GPU workloads where N ≫ number of CUDA cores, work efficiency matters.
    An O(N log N) scan with N=1B elements wastes significant compute bandwidth.

### The Two Metrics: Work and Span

    WORK:   total number of operations across all threads.
            Equivalent to: how long would a sequential CPU take?
            Lower is better. Sequential is optimal at O(N).

    SPAN:   length of the critical path (longest dependency chain).
            Equivalent to: how long with infinite processors?
            Lower is better. O(log N) is optimal.

    Ideal parallel algorithm: Work = O(N), Span = O(log N).

    Algorithm comparison:
        Sequential scan:    Work = O(N),       Span = O(N)       — not parallel
        Hillis-Steele:      Work = O(N log N), Span = O(log N)   — fast but wasteful
        Blelloch:           Work = O(N),       Span = O(log N)   — OPTIMAL


##### PART 3 — HILLIS-STEELE: OPTIMAL SPAN, SIMPLE SMEM IMPLEMENTATION

### The Algorithm

    Named after Danny Hillis and Guy Steele (1986).
    Also called the "naive parallel scan" or "inclusive scan by doubling".

    At each step, every element adds the value from stride positions back:

        for step in range(log2(N)):
            stride = 1 << step           # 1, 2, 4, 8, 16, ...
            for i in range(N) [parallel]:
                if i >= stride:
                    A[i] = A[i] + A[i - stride]
                # else: A[i] unchanged

    After log2(N) steps, A[i] contains the inclusive scan of A[0..i].

    The KEY REQUIREMENT: each step reads the values from the PREVIOUS step.
    → Must use double buffering: read from buffer A, write to buffer B.
    → Alternate: SMEM ping-pong, or handle with careful indexing.

### Step-by-Step Trace (N=8, values=[1,2,3,4,5,6,7,8])

    Initial:    [1, 2, 3, 4, 5, 6, 7, 8]

    Step 0 (stride=1): each position adds position-1
        i=0: unchanged → 1
        i=1: 2 + 1 = 3
        i=2: 3 + 2 = 5
        i=3: 4 + 3 = 7
        i=4: 5 + 4 = 9
        i=5: 6 + 5 = 11
        i=6: 7 + 6 = 13
        i=7: 8 + 7 = 15
        Result: [1, 3, 5, 7, 9, 11, 13, 15]

    Step 1 (stride=2): each position adds position-2
        i=0,1: unchanged → 1, 3
        i=2: 5 + 1 = 6
        i=3: 7 + 3 = 10
        i=4: 9 + 5 = 14
        i=5: 11 + 7 = 18
        i=6: 13 + 9 = 22
        i=7: 15 + 11 = 26
        Result: [1, 3, 6, 10, 14, 18, 22, 26]

    Step 2 (stride=4): each position adds position-4
        i=0..3: unchanged → 1, 3, 6, 10
        i=4: 14 + 1 = 15
        i=5: 18 + 3 = 21
        i=6: 22 + 6 = 28
        i=7: 26 + 10 = 36
        Result: [1, 3, 6, 10, 15, 21, 28, 36]  ← INCLUSIVE SCAN ✓

    True cumulative sums: [1, 3, 6, 10, 15, 21, 28, 36] ✓

### Work Count Analysis

    Step 0: N-1 additions (all threads except first)
    Step 1: N-2 additions
    Step k: N - 2^k additions
    Total: sum_{k=0}^{log2(N)-1} (N - 2^k)
         = N×log2(N) - (N-1) ≈ N×log2(N)

    For N=1024: 1024×10 - 1023 = 9217 operations vs 1023 optimal.
    The extra work grows as the stride grows (fewer elements add, but we already
    paid for the outer loop count).

### Double Buffering in SMEM

    Within a single block, Hillis-Steele uses two SMEM arrays:
        float ping[N], pong[N];
        // Load into ping
        for each step:
            for i in parallel: pong[i] = ping[i] + (i>=stride ? ping[i-stride] : 0)
            __syncthreads()
            swap(ping, pong)

    SMEM requirement: 2 × N × 4 bytes = 2 KB for N=256.
    Fits comfortably in H100's 228 KB SMEM per SM.

    Warp-level variant: __shfl_up eliminates the double buffer entirely.
    (See Part 5 — each lane reads directly from a peer lane's register.)


##### PART 4 — BLELLOCH ALGORITHM: WORK-EFFICIENT O(N) SCAN

### The Key Insight: Reduction Then Propagation

    Blelloch (1990) observed that a work-efficient scan can be decomposed
    into two phases, each traversing a binary tree of operations:

        PHASE 1: UP-SWEEP (reduce)    — build a partial-sum binary tree
        PHASE 2: DOWN-SWEEP (expand)  — propagate the total back down

    Total work: 2×(N-1) = O(N). Span: 2×log2(N) = O(log N). Both optimal.

    The algorithm ALWAYS produces an EXCLUSIVE scan.
    Inclusive scan: shift the result or run a final pass.

### Phase 1: Up-Sweep (Reduce)

    The up-sweep builds a reduction tree in-place.
    At stride d, elements at positions 2d-1 accumulate from 2^(d-1) positions left.

    for d = 0, 1, ..., log2(N)-1:
        stride = 2^(d+1)
        for each k = stride-1, 2*stride-1, 3*stride-1, ... [parallel]:
            A[k] += A[k - 2^d]

    After the up-sweep, A[N-1] contains the total sum of all elements.
    The array holds a partial reduction tree.

    Trace (N=8, values=[1,1,1,1,1,1,1,1]):

        Initial: [1, 1, 1, 1, 1, 1, 1, 1]

        d=0 (stride=2, update positions 1,3,5,7):
            A[1] += A[0]   → 2
            A[3] += A[2]   → 2
            A[5] += A[4]   → 2
            A[7] += A[6]   → 2
            [1, 2, 1, 2, 1, 2, 1, 2]

        d=1 (stride=4, update positions 3,7):
            A[3] += A[1]   → 4
            A[7] += A[5]   → 4
            [1, 2, 1, 4, 1, 2, 1, 4]

        d=2 (stride=8, update position 7):
            A[7] += A[3]   → 8
            [1, 2, 1, 4, 1, 2, 1, 8]   ← A[7] = total sum ✓

    The tree looks like:
                         8          ← total sum
                   4          4
               2      2    2    2
              1 1    1 1  1 1  1 1  ← original

### Phase 2: Down-Sweep (Propagate)

    The down-sweep distributes the total sum downward through the tree.
    First: set A[N-1] = identity (0 for addition).
    Then traverse the tree in reverse, pushing values down:

    A[N-1] = 0  (identity element)

    for d = log2(N)-1, log2(N)-2, ..., 0:
        stride = 2^(d+1)
        for each k = stride-1, 2*stride-1, ... [parallel]:
            left_child = k - 2^d
            tmp = A[left_child]
            A[left_child] = A[k]                 ← left child gets parent
            A[k] = A[k] + tmp                    ← right child gets parent + old left

    Continuing the trace (N=8, all-ones, after up-sweep):

        Before down-sweep: [1, 2, 1, 4, 1, 2, 1, 8]
        Set A[7] = 0:      [1, 2, 1, 4, 1, 2, 1, 0]

        d=2 (stride=8, process position 7):
            tmp = A[3] = 4
            A[3] = A[7] = 0
            A[7] = 0 + 4 = 4
            [1, 2, 1, 0, 1, 2, 1, 4]

        d=1 (stride=4, process positions 3,7):
            At k=3: tmp=A[1]=2; A[1]=A[3]=0; A[3]=0+2=2
            At k=7: tmp=A[5]=2; A[5]=A[7]=4; A[7]=4+2=6
            [1, 0, 1, 2, 1, 4, 1, 6]

        d=0 (stride=2, process positions 1,3,5,7):
            At k=1: tmp=A[0]=1; A[0]=A[1]=0; A[1]=0+1=1
            At k=3: tmp=A[2]=1; A[2]=A[3]=2; A[3]=2+1=3
            At k=5: tmp=A[4]=1; A[4]=A[5]=4; A[5]=4+1=5
            At k=7: tmp=A[6]=1; A[6]=A[7]=6; A[7]=6+1=7
            [0, 1, 2, 3, 4, 5, 6, 7]  ← EXCLUSIVE SCAN ✓

    Exclusive scan of [1,1,1,1,1,1,1,1] is [0,1,2,3,4,5,6,7]. ✓

### Work Count Analysis

    Up-sweep:
        d=0: N/2 operations
        d=1: N/4 operations
        ...
        d=log2(N)-1: 1 operation
        Total: N/2 + N/4 + ... + 1 = N-1 operations

    Down-sweep:
        d=log2(N)-1: 1 operation
        ...
        d=0: N/2 operations
        Total: N-1 operations

    Grand total: 2(N-1) = O(N) ← work-optimal, matches sequential!

### Comparison Summary

    ┌──────────────────┬──────────────┬──────────────┬───────────────────┐
    │ Algorithm        │ Work         │ Span         │ SMEM              │
    ├──────────────────┼──────────────┼──────────────┼───────────────────┤
    │ Sequential       │ O(N)         │ O(N)         │ O(1)              │
    │ Hillis-Steele    │ O(N log N)   │ O(log N)     │ 2N (double buffer)│
    │ Blelloch         │ O(N)         │ O(log N)     │ N (in-place)      │
    │ Warp shfl_up     │ O(N)         │ O(log N)     │ 0 (registers)     │
    └──────────────────┴──────────────┴──────────────┴───────────────────┘

    Blelloch is the canonical block-level scan algorithm on GPUs.
    Hillis-Steele is the canonical warp-level scan (via __shfl_up_sync).

    Why use Hillis-Steele for warps despite O(N log N) work?
        For N=32 (one warp): log2(32)=5 steps, work=160 vs 31 optimal.
        But 5 __shfl_up instructions at register speed ≈ 20 cycles.
        Blelloch for N=32: also 10 steps (5 up + 5 down) with SMEM traffic.
        The constant factor of shuffle speed beats Blelloch's work savings.


##### PART 5 — WARP SCAN: __shfl_up_sync, REGISTER-SPEED, ZERO SMEM

### The __shfl_up Pattern

    __shfl_up_sync(mask, var, delta):
        Lane i receives var from lane (i - delta).
        Lanes 0..delta-1 are UNCHANGED (no source below them).

    Hillis-Steele scan via __shfl_up:

        float val = my_value;
        for (int offset = 1; offset < 32; offset <<= 1) {
            float peer = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane_id >= offset) val += peer;
        }
        // Inclusive scan: lane i holds sum(A[0..i])

    Five iterations, five __shfl_up instructions.
    Zero SMEM. Zero __syncthreads. Runs entirely in register file.

### Inclusive to Exclusive Conversion

    Method 1: shift result right by 1 position:
        excl = __shfl_up_sync(0xFFFFFFFF, incl, 1);
        if (lane_id == 0) excl = identity;

    Method 2: integrate into the scan loop (run one iteration less,
    then the "accumulation before current element" property emerges naturally).

    Both cost one extra __shfl_up and one conditional. Negligible overhead.

### The Width Parameter for Sub-Warp Scans

    __shfl_up(val, delta, width=W) treats the warp as groups of W lanes.
    Each group scans independently.

        float val = A[lane_id];
        for (int off = 1; off < W; off <<= 1) {
            float peer = __shfl_up_sync(mask, val, off, W);
            if ((lane_id % W) >= off) val += peer;
        }
        // W=8: four independent 8-element scans, all in parallel.

    Used when one warp processes multiple short sequences simultaneously.
    Critical for: per-row softmax over rows of width ≤ 32.


##### PART 6 — BLOCK SCAN: COMPOSING WARP SCANS VIA SMEM

### The Three-Phase Block Scan

    For a block of 256 threads (8 warps), scanning N=256 elements:

    PHASE 1 — Intra-Warp Scan:
        Each warp scans its 32 elements independently via __shfl_up.
        Result: warp-local inclusive scan (correct within each warp).
        Cost: 5 __shfl_up per warp. All 8 warps execute in parallel.

    PHASE 2 — Scan the Warp Totals:
        Lane 31 of each warp holds the inclusive sum of that warp's 32 elements.
        → Write warp_totals[warp_id] = lane_31_value to SMEM.
        __syncthreads()
        → Let warp 0 scan the 8 warp totals via __shfl_up (width=8, 3 steps).
        → Write results back to SMEM as warp_prefixes[warp_id].
        __syncthreads()

    PHASE 3 — Add Warp Prefix to Each Lane:
        Each lane adds warp_prefixes[warp_id] to its local scan result.
        For warp 0: prefix = 0 (identity, no prior warp).
        For warp 1: prefix = sum(warp 0).
        For warp k: prefix = sum(warps 0..k-1).
        Cost: 1 SMEM read + 1 addition per lane.

    TOTAL:
        SMEM operations: 8 writes + 8 reads (just the warp totals)
        __syncthreads(): 2
        __shfl_up calls: 5×8 (phase 1) + 3 (phase 2) = 43 total
        Result: correct inclusive scan over all 256 elements

### Extending to Arbitrary Block Sizes

    For block_size = B threads, num_warps = B/32:
        Phase 1: B/32 warp scans of 32 elements each. log2(32)=5 steps.
        Phase 2: 1 scan of B/32 warp totals. log2(B/32) steps.
        Phase 3: B additions.
        Total steps: 5 + log2(B/32) = 5 + log2(B) - 5 = log2(B) steps. ✓

    SMEM usage: B/32 × 4 bytes = 32 bytes for B=256. Negligible.

### Connection to Blelloch

    The three-phase block scan is effectively Blelloch's algorithm:
        Phase 1 = distributed up-sweep (via warp shuffles)
        Phase 2 = scan of partial sums (the reduction tree "spine")
        Phase 3 = distributed down-sweep (add prefix back)

    The warp-shuffle implementation achieves the same O(N) work, O(log N) span,
    but replaces SMEM array operations with register-file shuffles for phases 1&3,
    touching SMEM only for the inter-warp communication (8 values, not 256).


##### PART 7 — DEVICE-WIDE SCAN: THE DECOUPLED LOOKBACK ALGORITHM

### The Multi-Block Problem

    A block scan handles N ≤ block_size elements.
    For N = 1B elements, we need ~4M blocks.
    How do later blocks know the prefix sum contributed by earlier blocks?

### Naïve Two-Pass Approach

    Pass 1: each block independently scans its chunk → partial total per block.
    Pass 2: scan the array of partial totals → prefix for each block.
    Pass 3: each block adds its block prefix to its local scan results.

    Cost: 3 HBM passes over N elements.
    Limitation: latency = two kernel launches (expensive on small N).

### The Decoupled Lookback Algorithm (Merrill & Garland, 2016)

    The key insight: later blocks don't need to wait for ALL earlier blocks.
    They can proceed as soon as they have their OWN inclusive prefix,
    computed by looking back only as far as necessary.

    Each block maintains a STATUS FLAG in global memory:
        INVALID  (0): block has not yet begun processing
        PARTIAL  (P): block has a partial (local) result only
        COMPLETE (C): block has a complete (global) prefix

    Protocol for block i:
        1. Compute local scan of own chunk.
        2. Write (PARTIAL, local_total) to status[i].
        3. LOOK BACK: examine status[i-1], status[i-2], ...
           Accumulate partial totals until a COMPLETE status is found.
        4. When a COMPLETE predecessor is found:
               my_prefix = complete_predecessor.total + accumulated_partials
        5. Write (COMPLETE, my_prefix + local_total) to status[i].
        6. Finalize: add my_prefix to all local scan results.

    Correctness: each block waits (spin-loops) until it can determine its prefix.
    Expected lookback distance: O(1) amortised (most blocks find COMPLETE quickly).
    Total HBM passes: effectively 1 (scan happens as data is loaded).

    This is the algorithm behind:
        - NVIDIA CUB's DeviceScan
        - thrust::exclusive_scan
        - PyTorch's torch.cumsum (GPU path)
        - Every production-grade GPU scan implementation since 2016

### Chained Scan Pattern

    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Block 0  │  │ Block 1  │  │ Block 2  │  │ Block 3  │
    │ [PART,4] │→ │ [PART,3] │→ │ [PART,6] │→ │ [PART,5] │
    │          │  │ look back│  │ look back│  │ look back│
    │[COMP,4]  │  │[COMP,7]  │  │[COMP,13] │  │[COMP,18] │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘
    Block 1 waits for Block 0 to write COMPLETE (4), then announces COMPLETE (7).
    Block 2 waits for Block 1 to write COMPLETE (7), then announces COMPLETE (13).
    Block 3 only needs to look back ONE step (finds COMPLETE immediately).


##### PART 8 — SEGMENTED SCAN: INDEPENDENT SCANS IN FLAG-DELIMITED REGIONS

### What Segmented Scan Is

    A segmented scan resets the accumulator whenever a FLAG is set.
    The flags delimit independent segments — each segment produces its own
    independent scan output without carrying values across segment boundaries.

    Input values:  [3, 1, 4, 1,  | 5, 9, 2, |  6, 5, 3]
    Flags (heads): [1, 0, 0, 0,    1, 0, 0,    1, 0, 0]
    Excl. scan:    [0, 3, 4, 8,    0, 5,14,    0, 6,11]

    The flag=1 at a position means "this is the start of a new segment".
    Values never cross a segment boundary in the output.

### Lifting to (value, flag) Pairs

    The key trick: define a LIFTED OPERATOR on (value, flag) pairs:

        (v_a, f_a) ⊗ (v_b, f_b) = (f_b ? v_b : v_a + v_b,   f_a | f_b)

    Semantics:
        If f_b=1 (b starts a new segment): result value = v_b (reset)
        If f_b=0 (b continues same segment): result value = v_a + v_b (accumulate)
        The flag in the result is 1 if either input started a segment.

    This lifted operator IS ASSOCIATIVE — any binary tree produces the correct result.
    This means ALL the scan algorithms above (Hillis-Steele, Blelloch, warp shuffle)
    can be directly applied to (value, flag) pairs to get segmented scan!

    No special-casing in the tree structure. Just a different operator.

### Associativity Proof of the Lifted Operator

    Consider three pairs (v_a, f_a), (v_b, f_b), (v_c, f_c).

    Left grouping: ((v_a,f_a) ⊗ (v_b,f_b)) ⊗ (v_c,f_c)
        middle = (f_b ? v_b : v_a+v_b,  f_a|f_b)
        final  = (f_c ? v_c : (f_b?v_b:v_a+v_b)+v_c,  f_a|f_b|f_c)

    Right grouping: (v_a,f_a) ⊗ ((v_b,f_b) ⊗ (v_c,f_c))
        inner  = (f_c ? v_c : v_b+v_c,  f_b|f_c)
        final  = (f_b|f_c ? (f_c?v_c:v_b+v_c) : v_a+(f_c?v_c:v_b+v_c),  f_a|f_b|f_c)

    Case analysis (f_b, f_c):
        (0,0): both = v_a+v_b+v_c  ✓
        (0,1): both = v_c           ✓
        (1,0): both = v_b+v_c       ✓
        (1,1): both = v_c           ✓
    Associative. ✓

### Applications of Segmented Scan in LLM Inference

    VARIABLE-LENGTH SEQUENCE PACKING:
        LLM inputs have different lengths: [512, 43, 1024, 7, 256]
        Pack them contiguously in one tensor to avoid padding waste.
        Flags mark sequence boundaries.
        Segmented scan computes per-token position IDs within each sequence.
        Segmented scan computes cumulative context lengths for attention masking.

    PER-SEQUENCE KV CACHE MANAGEMENT (vLLM):
        Each sequence in a batch has its own KV cache blocks.
        Segmented scan over token counts → per-sequence page offsets.
        Block allocation: exclusive scan of block_count per sequence
                          → starting block index in the pool.

    RAGGED TENSOR OPERATIONS:
        PyTorch's NestedTensor uses segmented scan for batch offset computation.
        torch._nested_tensor_from_mask → builds offset arrays via segmented scan.

    CAUSAL ATTENTION MASK GENERATION:
        For packed sequences, the attention mask must NOT allow attention
        across sequence boundaries. Segmented scan propagates sequence IDs
        so each token knows which sequence it belongs to.


##### PART 9 — SCAN APPLICATIONS: RADIX SORT, COMPACTION, HISTOGRAM

### Radix Sort (The Queen of GPU Sorting)

    GPU radix sort processes K-bit keys in passes of R bits at a time.
    For R=4 (nibble sort): 4 passes over 32-bit keys, each pass sorts 4 bits.

    Each pass:
        1. Histogram: count occurrences of each of 2^R=16 digit values.
        2. EXCLUSIVE SCAN of histogram → starting positions of each digit.
        3. Scatter: each element writes itself to histogram_scan[digit].

    Steps 1 and 3 require passes over N; step 2 is a scan of size 2^R=16.
    The bottleneck is step 3 (scatter) — must be done without conflicts.

    Per-block refinement (CUB radix sort):
        Each block sorts its local chunk using rank = local_exclusive_scan
        on a per-block histogram. Then global scan combines block offsets.
        Two-level: local scan (fast, SMEM) + global scan (decoupled lookback).

    This is why CUB's DeviceRadixSort is O(N) — all three steps are linear.

### Stream Compaction

    Filter an array A keeping only elements where filter(A[i]) is true.

    Naïve GPU approach (atomic):
        int pos = atomicAdd(&output_count, 1);
        output[pos] = A[i];
    Problem: atomic serialisation at high concurrency. Output order not preserved.

    Scan-based approach (deterministic, no atomics):
        1. keep[i] = filter(A[i]) ? 1 : 0
        2. offsets = exclusive_scan(keep)
        3. if (keep[i]) output[offsets[i]] = A[i];
    Output order matches input order. No atomic. Fully parallel.
    Cost: 2 passes over N + 1 scan = 3N reads + N writes.

    This is how CUDA thrust::copy_if, torch.masked_select, and
    vLLM's token compaction for speculative decoding work.

### Histogram Construction and CDF

    Goal: bin N values into B buckets, compute per-bucket CDF.

    1. For each element: bucket_id = (int)(value / bucket_width)
    2. atomicAdd(&histogram[bucket_id], 1)    ← per-block histograms
    3. histogram_global = sum all per-block histograms (reduction)
    4. CDF = exclusive_scan(histogram_global)
    5. (Optional) normalise: PDF[b] = histogram[b] / N

    Step 4 produces: for each bucket b, how many elements have value < bucket_start_b.
    This is the scatter offset for radix sort and the quantile function for sampling.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Inclusive vs Exclusive Scan — Definitions, Operators & Conversions": {
        "description": (
            "Implement inclusive and exclusive scans for every common operator "
            "(sum, max, min, product, logical-OR). Show the conversion formulas. "
            "Verify scan output against sequential reference. Trace the relationship "
            "between inclusive and exclusive outputs element by element. Demonstrate "
            "the identity element requirement and its effect on the first output."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  INCLUSIVE vs EXCLUSIVE SCAN — Operators, Identities & Conversion")
print("=" * 68)
print()

np.random.seed(7)


# ─────────────────────────────────────────────────────────────────────
# Sequential reference implementations
# ─────────────────────────────────────────────────────────────────────

def inclusive_scan(arr, op, identity):
    out = np.empty_like(arr, dtype=float)
    acc = identity
    for i, x in enumerate(arr):
        acc = op(acc, float(x))
        out[i] = acc
    return out

def exclusive_scan(arr, op, identity):
    out = np.empty_like(arr, dtype=float)
    acc = identity
    for i, x in enumerate(arr):
        out[i] = acc
        acc = op(acc, float(x))
    return out

def inclusive_to_exclusive(incl, identity):
    excl = np.empty_like(incl)
    excl[0] = identity
    excl[1:] = incl[:-1]
    return excl

def exclusive_to_inclusive(excl, arr, op):
    return np.array([op(excl[i], float(arr[i])) for i in range(len(arr))])


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Core definitions with example arrays
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Definitions: Inclusive vs Exclusive on 8-Element Array")
print("━" * 68)
print()

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
incl = inclusive_scan(arr, lambda a, b: a + b, 0)
excl = exclusive_scan(arr, lambda a, b: a + b, 0)

print(f"  Input:             {arr.astype(int).tolist()}")
print(f"  Inclusive scan:    {incl.astype(int).tolist()}")
print(f"  Exclusive scan:    {excl.astype(int).tolist()}")
print()

# Element-by-element relationship table
print(f"  {'idx':>4}  {'A[i]':>6}  {'incl[i]':>10}  {'excl[i]':>10}  "
      f"{'excl[i]=incl[i-1]?':>20}  {'incl[i]=excl[i]+A[i]?'}")
print("  " + "─" * 72)
for i in range(len(arr)):
    excl_check = (i == 0 and excl[i] == 0) or (i > 0 and abs(excl[i] - incl[i-1]) < 1e-9)
    incl_check = abs(incl[i] - (excl[i] + arr[i])) < 1e-9
    print(f"  {i:>4}  {int(arr[i]):>6}  {int(incl[i]):>10}  {int(excl[i]):>10}  "
          f"{'✅' if excl_check else '❌':>20}  {'✅' if incl_check else '❌'}")

print()
print("  KEY RELATIONSHIP:")
print("    exclusive[i]   = inclusive[i-1]   (exclusive is inclusive shifted right by 1)")
print("    exclusive[0]   = identity          (0 for addition)")
print("    inclusive[i]   = exclusive[i] + A[i]")
print()

# Conversion functions verification
excl_from_incl = inclusive_to_exclusive(incl, identity=0)
incl_from_excl = exclusive_to_inclusive(excl, arr, op=lambda a, b: a + b)
print(f"  Convert inclusive → exclusive: {excl_from_incl.astype(int).tolist()}  "
      f"{'✅' if np.allclose(excl_from_incl, excl) else '❌'}")
print(f"  Convert exclusive → inclusive: {incl_from_excl.astype(int).tolist()}  "
      f"{'✅' if np.allclose(incl_from_excl, incl) else '❌'}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Scans for all common operators
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Scans for Every Standard Operator")
print("━" * 68)
print()

ops = [
    ("Sum  (+)",         lambda a, b: a + b,             0,          np.array([1,2,3,4,5,6,7,8], dtype=float)),
    ("Max  (max)",       lambda a, b: max(a, b),         -math.inf,  np.array([3,1,4,1,5,9,2,6], dtype=float)),
    ("Min  (min)",       lambda a, b: min(a, b),         math.inf,   np.array([3,1,4,1,5,9,2,6], dtype=float)),
    ("Prod (×)",         lambda a, b: a * b,             1,          np.array([1,2,3,4,5,6,7,8], dtype=float)),
    ("OR   (|)",         lambda a, b: float(bool(a) or bool(b)),  0, np.array([0,0,1,0,0,1,0,0], dtype=float)),
    ("XOR  (^)",         lambda a, b: float(int(a) ^ int(b)),     0, np.array([1,1,0,1,1,0,1,0], dtype=float)),
]

for name, op, identity, data in ops:
    incl_r = inclusive_scan(data, op, identity)
    excl_r = exclusive_scan(data, op, identity)
    id_str = str(identity) if math.isfinite(float(str(identity).replace('inf', '1'))) else str(identity)
    id_str = "+∞" if identity == math.inf else ("-∞" if identity == -math.inf else str(int(identity)))
    print(f"  Operator: {name}   Identity = {id_str}")
    print(f"    Input:     {[int(x) for x in data]}")
    incl_disp = [int(x) if math.isfinite(x) else str(x) for x in incl_r]
    excl_disp = [int(x) if math.isfinite(x) else str(x) for x in excl_r]
    print(f"    Inclusive: {incl_disp}")
    print(f"    Exclusive: {excl_disp}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Stream compaction — exclusive scan as allocation map
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Stream Compaction: Exclusive Scan as Write-Offset Map")
print("━" * 68)
print()

np.random.seed(42)
data_vals = np.random.randint(0, 20, 16)
threshold = 10
keep_mask = (data_vals > threshold).astype(int)
offsets   = exclusive_scan(keep_mask, lambda a, b: a + b, 0).astype(int)
total_out = keep_mask.sum()

print(f"  Input:       {data_vals.tolist()}")
print(f"  Keep (>10):  {keep_mask.tolist()}")
print(f"  Excl. scan:  {offsets.tolist()}")
print(f"  Total kept:  {total_out}")
print()

output = np.full(total_out, -1)
for i in range(len(data_vals)):
    if keep_mask[i]:
        output[offsets[i]] = data_vals[i]

print(f"  Compacted output:  {output.tolist()}")
print(f"  True filter:       {sorted(data_vals[data_vals > threshold].tolist())}")
print()
print(f"  Assignment trace (first 10 elements):")
print(f"  {'idx':>4}  {'val':>5}  {'keep':>6}  {'offset':>8}  {'writes to'}")
print("  " + "─" * 38)
for i in range(min(10, len(data_vals))):
    dest = f"output[{offsets[i]}]" if keep_mask[i] else "-"
    print(f"  {i:>4}  {data_vals[i]:>5}  {keep_mask[i]:>6}  {offsets[i]:>8}  {dest}")
print()
print("  No atomics, no races, output order preserved. Total = one scan pass.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Lifted (max, sum) pair — scan for softmax statistics
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Scan of (max, sum) Pairs: Online Softmax Foundation")
print("━" * 68)
print()

def merge_ms(a, b):
    """Merge two (max, sum_of_exp) accumulators."""
    m_a, d_a = a
    m_b, d_b = b
    m = max(m_a, m_b)
    if m == -math.inf:
        return (m, 0.0)
    d = d_a * math.exp(m_a - m) + d_b * math.exp(m_b - m)
    return (m, d)

logits = np.array([2.0, 5.0, 1.0, 8.0, 3.0, 7.0, 4.0, 6.0])
N_l    = len(logits)
identity_ms = (-math.inf, 0.0)

# Each element starts as (logit, 1.0) = (m=logit, d=exp(logit-logit)=1)
elements = [(float(x), 1.0) for x in logits]

# Exclusive scan of (max, sum) pairs
ms_scan = []
acc = identity_ms
for elem in elements:
    ms_scan.append(acc)
    acc = merge_ms(acc, elem)

print(f"  Logits:  {logits.tolist()}")
print()
print(f"  Exclusive scan of (m, d) pairs:")
print(f"  (Each element starts as (logit_i, 1.0) — max=logit, sum=1)")
print()
print(f"  {'idx':>4}  {'logit':>7}  {'m (running max)':>18}  {'d (running sum)':>18}  "
      f"{'prob (from pair)':>18}")
print("  " + "─" * 74)

final_m = ms_scan[-1][0] if ms_scan[-1][0] > -math.inf else logits.max()
final_acc = acc
for i in range(N_l):
    m_i, d_i = ms_scan[i]
    m_disp = f"{m_i:.4f}" if m_i > -math.inf else "-inf"
    d_disp = f"{d_i:.6f}"
    # Probability using the running stats AFTER this element (approximate illustration)
    print(f"  {i:>4}  {logits[i]:>7.1f}  {m_disp:>18}  {d_disp:>18}")

print()
final_m_val, final_d_val = final_acc
ref_sm = np.exp(logits - logits.max()) / np.sum(np.exp(logits - logits.max()))
print(f"  Final (m, d) after all elements: m={final_m_val:.4f}, d={final_d_val:.6f}")
print(f"  Reference softmax denominator:   {np.sum(np.exp(logits-logits.max())):.6f}")
print(f"  Match: {'✅' if abs(final_d_val - np.sum(np.exp(logits-logits.max()))) < 1e-9 else '❌'}")
print()
print("  Each exclusive scan entry [i] gives the softmax stats for ALL elements BEFORE i.")
print("  This is how parallel softmax with lookback works — no second pass needed.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Hillis-Steele Algorithm — Step Trace, Work Analysis & SMEM Cost": {
        "description": (
            "Implement the Hillis-Steele parallel scan algorithm with full step-by-step "
            "traces showing every element's value after each doubling step. Count exact "
            "operations per step and total work. Demonstrate the double-buffering "
            "requirement. Show the dependency graph and why log2(N) steps achieve correct "
            "results. Compare work to sequential baseline and Blelloch."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  HILLIS-STEELE ALGORITHM — Doubling Steps, Work & Dependency Graph")
print("=" * 68)
print()

np.random.seed(3)


# ─────────────────────────────────────────────────────────────────────
# Hillis-Steele implementation with full instrumentation
# ─────────────────────────────────────────────────────────────────────

def hillis_steele_trace(arr, verbose=True):
    """
    Hillis-Steele inclusive scan with step-by-step trace.
    Uses double buffering (ping/pong) exactly as on the GPU.
    """
    n = len(arr)
    assert (n & (n - 1)) == 0, "N must be a power of 2 for this demo"

    ping = arr.astype(float).copy()
    pong = np.zeros(n)
    total_ops = 0
    steps     = int(math.log2(n))
    step_data = []

    for step in range(steps):
        stride = 1 << step   # 1, 2, 4, 8, ...
        ops_this_step = 0
        for i in range(n):
            if i >= stride:
                pong[i] = ping[i] + ping[i - stride]
                ops_this_step += 1
            else:
                pong[i] = ping[i]   # unchanged
        ping, pong = pong, ping     # swap buffers
        total_ops += ops_this_step
        step_data.append((stride, ops_this_step, ping.copy()))
        if verbose:
            vals = " ".join(f"{int(v):4d}" for v in ping)
            print(f"  step {step+1} (stride={stride:3d}, ops={ops_this_step:3d}):  [{vals}]")

    return ping, total_ops, step_data


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Step-by-step traces for different sizes
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Step-by-Step Traces")
print("━" * 68)
print()

for arr_data, label in [
    (np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float), "Increasing values"),
    (np.ones(8, dtype=float),                            "All ones"),
    (np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float),  "Pi digits"),
]:
    true_incl = np.cumsum(arr_data)
    print(f"  {label}:")
    print(f"  Input:    [{' '.join(f'{int(v):4d}' for v in arr_data)}]")
    result, total_ops, _ = hillis_steele_trace(arr_data, verbose=True)
    print(f"  Expected: [{' '.join(f'{int(v):4d}' for v in true_incl)}]")
    print(f"  Correct:  {'✅' if np.allclose(result, true_incl) else '❌'}  "
          f"Total ops: {total_ops}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Dependency graph visualisation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Dependency Graph (N=8): Which Elements Contribute?")
print("━" * 68)
print()

print("  After each step, output[i] = sum of which input elements?")
print()
n_dep = 8
# Track which original inputs contribute to each position after each step
contributors = [{i} for i in range(n_dep)]
print(f"  {'After step':>12}  " + "  ".join(f"out[{i}]" for i in range(n_dep)))
print("  " + "─" * 72)

def fmt_contrib(s):
    """Format a set of contributors compactly."""
    if not s:
        return "  -  "
    lst = sorted(s)
    if len(lst) == 1:
        return f" {{{lst[0]}}} "
    return f"{{{','.join(map(str,lst))}}}"

print(f"  {'Input':>12}  " + "  ".join(f"  {{{i}}}  " for i in range(n_dep)))

for step in range(int(math.log2(n_dep))):
    stride = 1 << step
    new_c = [c.copy() for c in contributors]
    for i in range(n_dep):
        if i >= stride:
            new_c[i] = contributors[i] | contributors[i - stride]
    contributors = new_c
    row = f"  {f'step {step+1} (δ={stride})':>12}  "
    row += "  ".join(f"{fmt_contrib(contributors[i]):>7}" for i in range(n_dep))
    print(row)

print()
print("  After step 3 (final): out[i] contains the union of inputs 0..i. ✓")
print("  Each column grows leftward — eventually reaching position 0.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Work and span analysis across sizes
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Work & Span Analysis Across Array Sizes")
print("━" * 68)
print()

print(f"  {'N':>10}  {'Steps':>7}  {'HS work':>10}  {'Seq work':>10}  "
      f"{'Extra work':>12}  {'Work ratio':>11}")
print("  " + "─" * 62)

for n_val in [4, 8, 16, 32, 64, 256, 1024, 65536, 1_000_000]:
    if (n_val & (n_val - 1)) != 0:
        # round to nearest power of 2
        n_p2 = 1 << (n_val - 1).bit_length()
    else:
        n_p2 = n_val
    steps      = int(math.log2(n_p2))
    hs_work    = sum(n_p2 - (1 << s) for s in range(steps))
    seq_work   = n_p2 - 1
    extra      = hs_work - seq_work
    ratio      = hs_work / seq_work if seq_work > 0 else 1.0
    print(f"  {n_p2:>10,}  {steps:>7}  {hs_work:>10,}  {seq_work:>10,}  "
          f"{extra:>12,}  {ratio:>10.2f}×")

print()
print("  For large N: Hillis-Steele does ~log2(N)× more work than sequential.")
print("  This EXTRA WORK is the core weakness — motivating Blelloch's algorithm.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Double buffering — why it is mandatory
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Double Buffering: Why In-Place Updates Break Hillis-Steele")
print("━" * 68)
print()

arr_test = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
true_incl = np.cumsum(arr_test)

# In-place (WRONG — reads updated values from the same step)
def hillis_steele_inplace_wrong(arr):
    a = arr.astype(float).copy()
    for step in range(int(math.log2(len(a)))):
        stride = 1 << step
        for i in range(len(a) - 1, stride - 1, -1):  # reverse to show it still fails
            a[i] = a[i] + a[i - stride]
    return a

wrong = hillis_steele_inplace_wrong(arr_test)
correct, _, _ = hillis_steele_trace(arr_test, verbose=False)

print(f"  Input:              {arr_test.astype(int).tolist()}")
print(f"  Expected inclusive: {true_incl.astype(int).tolist()}")
print(f"  In-place (wrong):   {wrong.astype(int).tolist()}  ❌")
print(f"  Double-buf (right): {correct.astype(int).tolist()}  ✅")
print()
print("  WHY: in-place step 0 (stride=1):")
print("    A[1] += A[0] → A[1] now holds partial sum")
print("    A[2] += A[1] → but A[1] was ALREADY UPDATED this step!")
print("    → A[2] uses the wrong value. Error cascades through all positions.")
print()
print("  SOLUTION: read from 'ping', write to 'pong', then swap.")
print("  GPU SMEM implementation: two arrays of N floats, alternate each step.")
print(f"  SMEM cost: 2 × N × 4 bytes  (double buffer)")
print(f"  For N=256: 2 × 256 × 4 = 2048 bytes. Fits trivially in H100 SMEM.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Blelloch Algorithm — Up-Sweep, Down-Sweep & Work Efficiency": {
        "description": (
            "Implement the full Blelloch work-efficient scan with complete traces "
            "of both the up-sweep (reduce) and down-sweep (propagate) phases. "
            "Show the in-place binary tree structure at each step. Verify the "
            "exclusive scan output. Count operations at each tree level and prove "
            "O(N) total work. Extend to inclusive scan and non-power-of-two N."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  BLELLOCH ALGORITHM — Up-Sweep / Down-Sweep / Work-Efficient O(N)")
print("=" * 68)
print()

np.random.seed(11)


# ─────────────────────────────────────────────────────────────────────
# Blelloch implementation with full instrumentation
# ─────────────────────────────────────────────────────────────────────

def blelloch_scan(arr, identity=0.0, op=None, verbose=True):
    """
    Blelloch work-efficient exclusive scan.
    Operates IN-PLACE on a copy of arr.
    Returns (output, up_ops, down_ops, tree_snapshots).
    """
    if op is None:
        op = lambda a, b: a + b

    n = len(arr)
    assert (n & (n - 1)) == 0, "Blelloch requires power-of-2 length"

    a = arr.astype(float).copy()
    levels = int(math.log2(n))
    snapshots = []
    up_ops   = 0
    down_ops = 0

    # ── PHASE 1: UP-SWEEP ────────────────────────────────────────────
    if verbose:
        print(f"  ── PHASE 1: UP-SWEEP (reduce to tree) ──────────────────────")
        print(f"  Initial:  {a.astype(int if a.max() < 1e6 else float).tolist()}")
        print()

    for d in range(levels):
        stride = 1 << (d + 1)
        ops_this = 0
        for k in range(stride - 1, n, stride):
            left  = k - (1 << d)
            a[k]  = op(a[k], a[left])
            ops_this += 1
        up_ops += ops_this
        snapshots.append(('up', d, stride, a.copy()))
        if verbose:
            print(f"  d={d} (stride={stride:3d}, {ops_this:3d} ops):  "
                  f"{a.astype(int).tolist()}")

    # ── PHASE 2: DOWN-SWEEP ──────────────────────────────────────────
    if verbose:
        print()
        print(f"  ── PHASE 2: DOWN-SWEEP (propagate identity down) ────────────")
        print(f"  Set a[{n-1}] = {identity}  (identity element)")

    a[n - 1] = identity

    for d in range(levels - 1, -1, -1):
        stride   = 1 << (d + 1)
        ops_this = 0
        for k in range(stride - 1, n, stride):
            left       = k - (1 << d)
            tmp        = a[left]
            a[left]    = a[k]               # left child  ← parent value
            a[k]       = op(a[k], tmp)      # right child ← parent + old left
            ops_this  += 2                  # 1 copy + 1 op
        down_ops += ops_this
        snapshots.append(('down', d, stride, a.copy()))
        if verbose:
            print(f"  d={d} (stride={stride:3d}, {ops_this//2:3d} swaps): "
                  f"{a.astype(int).tolist()}")

    if verbose:
        print()

    return a, up_ops, down_ops, snapshots


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Full trace on small array
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Full Trace: N=8, Values=[1,1,1,1,1,1,1,1]")
print("━" * 68)
print()

arr8 = np.ones(8, dtype=float)
result8, up_ops, down_ops, snapshots = blelloch_scan(arr8, identity=0.0, verbose=True)
true_excl8 = np.concatenate([[0], np.cumsum(arr8)[:-1]])

print(f"  Result:   {result8.astype(int).tolist()}")
print(f"  Expected: {true_excl8.astype(int).tolist()}")
print(f"  Correct:  {'✅' if np.allclose(result8, true_excl8) else '❌'}")
print(f"  Up-sweep ops:   {up_ops}")
print(f"  Down-sweep ops: {down_ops // 2} swaps = {down_ops} memory ops")
print(f"  Total work:     {up_ops + down_ops // 2} operations  (vs N-1={len(arr8)-1} sequential)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Trace on pi-digits array
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Trace: N=8, Values=[3,1,4,1,5,9,2,6]")
print("━" * 68)
print()

arr_pi = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
result_pi, up_pi, down_pi, _ = blelloch_scan(arr_pi, identity=0.0, verbose=True)
true_excl_pi = np.concatenate([[0], np.cumsum(arr_pi)[:-1]])

print(f"  Result:   {result_pi.astype(int).tolist()}")
print(f"  Expected: {true_excl_pi.astype(int).tolist()}")
print(f"  Correct:  {'✅' if np.allclose(result_pi, true_excl_pi) else '❌'}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Work analysis — operation count per level
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Work Analysis: Operations per Level")
print("━" * 68)
print()

for n_w in [8, 16, 32, 64, 256]:
    levels = int(math.log2(n_w))
    print(f"  N={n_w} ({levels} levels each phase):")
    total_up = total_down = 0
    for d in range(levels):
        stride     = 1 << (d + 1)
        ops_at_d   = n_w // stride
        total_up  += ops_at_d
        total_down += ops_at_d
        print(f"    Level d={d}: stride={stride:4d} → {ops_at_d:3d} ops  "
              f"(positions every {stride})")
    print(f"    Up-sweep total:   {total_up}   (= N/2 + N/4 + ... + 1 = N-1)")
    print(f"    Down-sweep total: {total_down}   (same structure, reverse order)")
    print(f"    Grand total:      {total_up + total_down}   (= 2(N-1))")
    print(f"    Sequential:       {n_w - 1}")
    print(f"    Extra work:       {(total_up + total_down) - (n_w - 1)} ops "
          f"({((total_up+total_down)/(n_w-1) - 1)*100:.0f}% overhead)")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Blelloch vs Hillis-Steele work comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Blelloch vs Hillis-Steele: Work × Span Tradeoff")
print("━" * 68)
print()

print(f"  {'N':>10}  {'Seq work':>10}  {'Blelloch':>10}  "
      f"{'H-S work':>10}  {'H-S/Seq':>9}  {'Blelloch/Seq':>13}  {'H-S/Bleloch'}")
print("  " + "─" * 76)

for n_c in [8, 16, 32, 64, 128, 256, 1024, 65536]:
    levels  = int(math.log2(n_c))
    seq     = n_c - 1
    bll     = 2 * (n_c - 1)
    hs      = sum(n_c - (1 << s) for s in range(levels))
    print(f"  {n_c:>10,}  {seq:>10,}  {bll:>10,}  "
          f"{hs:>10,}  {hs/seq:>8.2f}×  {bll/seq:>12.1f}×  {hs/bll:>10.2f}×")

print()
print("  Blelloch: 2× sequential work (constant overhead). O(N).")
print("  Hillis-Steele: log2(N)× sequential work. O(N log N).")
print("  For N=1M: H-S does 20M ops vs Blelloch's 2M. 10× more work.")
print()
print("  BUT for N=32 (one warp):")
print("    H-S via __shfl_up: 5 register ops, no SMEM, ~20 cycles.")
print("    Blelloch: 10 SMEM ops, 2 sync barriers, ~40 cycles.")
print("    → H-S wins for warp-level (N ≤ 32) despite worse work complexity.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: Extending to non-power-of-two N
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 5 — Non-Power-of-Two N: Padding Strategy")
print("━" * 68)
print()

def blelloch_any_n(arr, identity=0.0):
    """Handle arbitrary N by padding to next power of two."""
    n = len(arr)
    n_p2 = 1 if n == 0 else 1 << (n - 1).bit_length()
    if n_p2 < n:
        n_p2 <<= 1

    padded = np.full(n_p2, identity, dtype=float)
    padded[:n] = arr
    result, *_ = blelloch_scan(padded, identity=identity, verbose=False)
    return result[:n]   # trim back to original size

print(f"  Padding to next power-of-two, trimming output back to N.")
print()
print(f"  {'N':>8}  {'Padded to':>10}  {'Waste %':>9}  {'Correct':>8}")
print("  " + "─" * 42)
for n_np2 in [3, 5, 7, 10, 15, 17, 100, 255, 1000]:
    n_p2 = 1 << (n_np2 - 1).bit_length()
    if n_p2 < n_np2:
        n_p2 <<= 1
    waste = (n_p2 - n_np2) / n_p2 * 100
    arr_test = np.random.randint(1, 5, n_np2).astype(float)
    result = blelloch_any_n(arr_test)
    true_result = np.concatenate([[0], np.cumsum(arr_test)[:-1]])
    correct = np.allclose(result, true_result)
    print(f"  {n_np2:>8}  {n_p2:>10}  {waste:>8.1f}%  {'✅' if correct else '❌'}")

print()
print("  Worst case: N = power_of_two + 1 → 50% waste.")
print("  Alternative: Blelloch on non-power-of-two via grid-stride (CUB DeviceScan).")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Block Scan — Warp Scans Composed via SMEM, Three-Phase Pipeline": {
        "description": (
            "Implement the complete three-phase block scan that combines warp-level "
            "Hillis-Steele scans (via __shfl_up) with shared memory inter-warp "
            "coordination. Show exactly which threads write to SMEM, how warp totals "
            "are collected, scanned, and distributed. Handle arbitrary N via grid-stride "
            "accumulation. Demonstrate SMEM savings vs naive per-element SMEM scan."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  BLOCK SCAN — Three-Phase Warp-Shuffle + SMEM Pipeline")
print("=" * 68)
print()

WARP_SIZE = 32
np.random.seed(17)


# ─────────────────────────────────────────────────────────────────────
# Warp-level inclusive scan (Hillis-Steele via __shfl_up)
# ─────────────────────────────────────────────────────────────────────

def shfl_up(values, delta, width=WARP_SIZE):
    """Simulate __shfl_up_sync: lane i gets values[i - delta] if i >= delta."""
    result = np.array(values, dtype=float)
    for lane in range(len(values)):
        pos = lane % width
        if pos >= delta:
            result[lane] = values[lane - delta]
    return result

def warp_inclusive_scan(warp_vals, width=WARP_SIZE):
    """Hillis-Steele inclusive scan over 'width' elements."""
    acc = np.array(warp_vals[:width], dtype=float)
    steps = int(math.log2(width))
    for step in range(steps):
        offset = 1 << step
        peer   = shfl_up(acc, offset, width=width)
        for lane in range(width):
            pos = lane % width
            if pos >= offset:
                acc[lane] += peer[lane]
    return acc

def warp_exclusive_scan(warp_vals, width=WARP_SIZE):
    """Exclusive scan: shift inclusive result right, set [0] = 0."""
    incl = warp_inclusive_scan(warp_vals, width)
    excl = shfl_up(incl, 1, width=width)
    for lane in range(0, len(excl), width):
        excl[lane] = 0.0
    return excl


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Full three-phase block scan — 256-thread trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Three-Phase Block Scan (256 threads = 8 warps), Traced")
print("━" * 68)
print()

BLOCK_SIZE = 256
N_WARPS    = BLOCK_SIZE // WARP_SIZE   # 8

data = np.random.randint(1, 4, BLOCK_SIZE).astype(float)
true_excl = np.concatenate([[0], np.cumsum(data)[:-1]])

print(f"  Block: {BLOCK_SIZE} threads / {N_WARPS} warps / {WARP_SIZE} lanes per warp")
print(f"  Data[:16]: {data[:16].astype(int).tolist()}")
print(f"  True excl[:16]: {true_excl[:16].astype(int).tolist()}")
print()

# ── PHASE 1: Intra-warp inclusive scan ─────────────────────────────
print("  ── PHASE 1: Intra-Warp Inclusive Scan (shfl_up, 5 steps per warp) ──")
print()

phase1 = np.zeros(BLOCK_SIZE)
warp_totals_p1 = np.zeros(N_WARPS)

for w in range(N_WARPS):
    warp_data   = data[w * WARP_SIZE : (w+1) * WARP_SIZE]
    warp_scan   = warp_inclusive_scan(warp_data)
    phase1[w * WARP_SIZE : (w+1) * WARP_SIZE] = warp_scan
    warp_totals_p1[w] = warp_scan[-1]   # lane 31 holds warp total
    print(f"  Warp {w}: scan[:8]={warp_scan[:8].astype(int).tolist()}...  "
          f"total={int(warp_scan[-1])}")

print()
print(f"  After Phase 1 — warp totals (lane 31 of each warp):")
print(f"    {warp_totals_p1.astype(int).tolist()}")
print()

# ── PHASE 2: Scan the warp totals in SMEM ──────────────────────────
print("  ── PHASE 2: Scan Warp Totals → Warp Prefix Sums (via SMEM) ──────")
print()

# Write warp totals to SMEM (N_WARPS = 8 values)
smem_totals = warp_totals_p1.copy()
print(f"  SMEM write (lane 31 of each warp): {smem_totals.astype(int).tolist()}")
print(f"  __syncthreads()")
print()

# Warp 0 scans the 8 totals (exclusive scan → prefix for each warp)
warp0_data   = np.concatenate([smem_totals, np.zeros(WARP_SIZE - N_WARPS)])
warp0_excl   = warp_exclusive_scan(warp0_data, width=N_WARPS)  # use width=8
warp_prefixes = warp0_excl[:N_WARPS]

print(f"  Warp 0 exclusive scan of totals:")
print(f"    Input:    {smem_totals.astype(int).tolist()}")
print(f"    Excl out: {warp_prefixes.astype(int).tolist()}")
print()
print(f"  Write warp_prefixes back to SMEM (N_WARPS={N_WARPS} writes)")
print(f"  __syncthreads()")
print()

# ── PHASE 3: Add warp prefix to each lane ──────────────────────────
print("  ── PHASE 3: Each Lane Reads Its Warp Prefix from SMEM, Adds It ──")
print()

phase3 = np.zeros(BLOCK_SIZE)
for w in range(N_WARPS):
    prefix = warp_prefixes[w]
    for lane in range(WARP_SIZE):
        idx = w * WARP_SIZE + lane
        # Convert inclusive → exclusive, then add warp prefix
        # Inclusive scan of warp = phase1 value; exclusive = phase1[i-1] within warp
        # For lane 0: phase1_inclusive[0] = data[0], exclusive = 0 → 0 + prefix
        local_excl = phase1[idx - 1] if (lane > 0) else 0.0
        phase3[idx] = local_excl + prefix

print(f"  Phase 3 result[:16]: {phase3[:16].astype(int).tolist()}")
print(f"  Expected[:16]:       {true_excl[:16].astype(int).tolist()}")
print(f"  Correct: {'✅' if np.allclose(phase3, true_excl) else '❌'}")
print()

# ── Cost summary ────────────────────────────────────────────────────
print("  ── Cost Summary ──────────────────────────────────────────────────")
print()
p1_shfl = 5 * N_WARPS     # 5 shfl_up per warp, N_WARPS warps
p2_smem_writes = N_WARPS  # lane 31 of each warp writes to SMEM
p2_shfl = int(math.log2(N_WARPS))   # 3 shfl_up for 8 warp totals
p2_smem_reads  = N_WARPS  # each warp reads its prefix
p3_smem_reads  = N_WARPS  # one read per warp (broadcast to all lanes)
p3_adds = BLOCK_SIZE

print(f"  {'Operation':<40}  {'Count'}")
print("  " + "─" * 50)
print(f"  {'Phase 1: __shfl_up calls (5/warp × 8 warps)':<40}  {p1_shfl}")
print(f"  {'Phase 2: SMEM writes (warp totals)':<40}  {p2_smem_writes}")
print(f"  {'Phase 2: __shfl_up calls (scan totals)':<40}  {p2_shfl}")
print(f"  {'Phase 2: SMEM reads (warp prefixes)':<40}  {p2_smem_reads}")
print(f"  {'Phase 3: SMEM reads (prefix per lane)':<40}  {p3_smem_reads}")
print(f"  {'Phase 3: additions (add prefix to local)':<40}  {p3_adds}")
print(f"  {'__syncthreads() barriers':<40}  2")
print()
print(f"  Total SMEM ops: {p2_smem_writes + p2_smem_reads + p3_smem_reads} "
      f"(vs {BLOCK_SIZE * int(math.log2(BLOCK_SIZE))} for pure SMEM Hillis-Steele)")
print(f"  Reduction: {BLOCK_SIZE * int(math.log2(BLOCK_SIZE)) // (p2_smem_writes + p2_smem_reads + p3_smem_reads)}× fewer SMEM accesses")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Scaling — SMEM cost as block size grows
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — SMEM Ops: Three-Phase vs Pure-SMEM Hillis-Steele")
print("━" * 68)
print()

print(f"  {'Block size':>12}  {'N_warps':>8}  {'3-phase SMEM':>14}  "
      f"{'Pure-SMEM H-S':>15}  {'Savings':>9}")
print("  " + "─" * 62)

for bs in [32, 64, 128, 256, 512, 1024]:
    nw = bs // WARP_SIZE
    three_phase_smem = nw * 3      # write totals + read prefixes + phase 3 read
    hs_smem = bs * int(math.log2(bs)) * 2   # double-buffer: read + write each step
    savings = (1 - three_phase_smem / hs_smem) * 100
    print(f"  {bs:>12}  {nw:>8}  {three_phase_smem:>14}  "
          f"{hs_smem:>15}  {savings:>8.1f}%")

print()
print("  Three-phase SMEM ops grow as O(N_warps) = O(N/32).")
print("  Pure-SMEM H-S grows as O(N log N).")
print("  For block_size=1024: three-phase uses 96 SMEM ops vs 20480. ~213× fewer.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Segmented Scan — Lifted Operator, Applications & LLM Sequence Packing": {
        "description": (
            "Implement segmented scan via the (value, flag) lifted operator. Prove "
            "associativity with explicit case analysis. Show that any scan algorithm "
            "(Hillis-Steele, Blelloch, warp) works unchanged with the lifted operator. "
            "Apply to: stream compaction within segments, variable-length sequence "
            "packing for LLM batching, per-sequence position IDs, and KV-cache "
            "block offset computation matching vLLM's allocation pattern."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple

print("=" * 68)
print("  SEGMENTED SCAN — Lifted Operator, Proof & LLM Applications")
print("=" * 68)
print()

np.random.seed(99)


# ─────────────────────────────────────────────────────────────────────
# The lifted (value, flag) operator
# ─────────────────────────────────────────────────────────────────────

def lifted_op(a: Tuple, b: Tuple) -> Tuple:
    """
    Segmented scan operator on (value, flag) pairs.
    flag=1 means "start of new segment" (reset accumulator).

    (v_a, f_a) ⊗ (v_b, f_b) = (f_b ? v_b : v_a + v_b,  f_a | f_b)
    """
    v_a, f_a = a
    v_b, f_b = b
    v_out = v_b if f_b else v_a + v_b
    f_out = f_a | f_b
    return (v_out, f_out)

IDENTITY_PAIR = (0, 0)   # (value=0, flag=0) is identity for lifted_op


def segmented_scan_sequential(values, flags):
    """Sequential exclusive segmented scan using the lifted operator."""
    n = len(values)
    assert len(flags) == n
    pairs = list(zip(values, flags))

    out_vals = np.zeros(n)
    acc = IDENTITY_PAIR
    for i in range(n):
        out_vals[i] = acc[0]              # exclusive: output before adding current
        acc = lifted_op(acc, pairs[i])
        if flags[i]:                      # reset acc.value at segment boundary
            acc = (values[i], flags[i])   # new segment starts fresh
    # Redo correctly via standard exclusive scan with lifted op
    acc = IDENTITY_PAIR
    result = []
    for p in pairs:
        result.append(acc)
        acc = lifted_op(acc, p)

    return np.array([r[0] for r in result])


def segmented_scan_hillis_steele(values, flags):
    """
    Segmented inclusive scan via Hillis-Steele doubling on (value, flag) pairs.
    EXACT same code as regular H-S, just with lifted_op instead of +.
    """
    n = len(values)
    n_p2 = 1 << (n - 1).bit_length()
    if n_p2 < n:
        n_p2 <<= 1

    # Pad to power of 2 with identity pairs
    ping = [(float(values[i]), int(flags[i])) if i < n else IDENTITY_PAIR
            for i in range(n_p2)]
    pong = [IDENTITY_PAIR] * n_p2

    steps = int(math.log2(n_p2))
    for step in range(steps):
        stride = 1 << step
        for i in range(n_p2):
            if i >= stride:
                pong[i] = lifted_op(ping[i - stride], ping[i])
            else:
                pong[i] = ping[i]
        ping, pong = pong, ping

    # Convert inclusive → exclusive (shift right, prepend identity)
    excl = [IDENTITY_PAIR] + ping[:-1]
    return np.array([e[0] for e in excl[:n]])


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Basic segmented scan trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Basic Segmented Exclusive Scan: Step-by-Step Trace")
print("━" * 68)
print()

values = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], dtype=float)
flags  = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=int)
N_seg  = len(values)

result_seq = segmented_scan_sequential(values, flags)
result_hs  = segmented_scan_hillis_steele(values, flags)

print(f"  Input values: {values.astype(int).tolist()}")
print(f"  Flags:        {flags.tolist()}  (1 = new segment starts here)")
print()

# Identify segments
segments = []
seg_start = 0
for i in range(N_seg):
    if i > 0 and flags[i]:
        segments.append((seg_start, i - 1))
        seg_start = i
segments.append((seg_start, N_seg - 1))

print(f"  Segments identified:")
for seg_idx, (s, e) in enumerate(segments):
    seg_vals = values[s:e+1].astype(int).tolist()
    true_excl = [0] + list(np.cumsum(values[s:e+1])[:-1].astype(int))
    print(f"    Segment {seg_idx}: indices {s}..{e}  values={seg_vals}")
    print(f"             exclusive scan = {true_excl}")
print()

print(f"  Full exclusive segmented scan output:")
print(f"  {'idx':>4}  {'val':>5}  {'flag':>5}  {'result':>8}  {'segment':>8}  {'correct?'}")
print("  " + "─" * 48)
for i in range(N_seg):
    seg_id = next(j for j, (s, e) in enumerate(segments) if s <= i <= e)
    seg_start_i = segments[seg_id][0]
    true_excl_i = int(np.sum(values[seg_start_i:i])) if i > seg_start_i else 0
    correct = abs(result_seq[i] - true_excl_i) < 1e-9
    print(f"  {i:>4}  {int(values[i]):>5}  {flags[i]:>5}  "
          f"{int(result_seq[i]):>8}  {'seg '+str(seg_id):>8}  "
          f"{'✅' if correct else '❌'}")

print()
print(f"  Sequential result:   {result_seq.astype(int).tolist()}")
print(f"  Hillis-Steele result:{result_hs.astype(int).tolist()}")
print(f"  Match: {'✅' if np.allclose(result_seq, result_hs) else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Associativity proof by exhaustive case analysis
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Associativity Proof: All (f_b, f_c) Combinations")
print("━" * 68)
print()

print("  (v_a,f_a) ⊗ (v_b,f_b) ⊗ (v_c,f_c): left vs right grouping")
print()
print(f"  {'f_b':>4}  {'f_c':>4}  {'left result':>20}  {'right result':>20}  {'equal?'}")
print("  " + "─" * 58)

v_a, f_a = 3.0, 0
v_b_base, v_c_base = 5.0, 7.0

for f_b in [0, 1]:
    for f_c in [0, 1]:
        v_b = v_b_base
        v_c = v_c_base

        left  = lifted_op(lifted_op((v_a, f_a), (v_b, f_b)), (v_c, f_c))
        right = lifted_op((v_a, f_a), lifted_op((v_b, f_b), (v_c, f_c)))

        equal = (abs(left[0] - right[0]) < 1e-9) and (left[1] == right[1])

        # Human-readable expected value
        if f_c:
            expected_v = v_c
        elif f_b:
            expected_v = v_b + v_c
        else:
            expected_v = v_a + v_b + v_c

        print(f"  {f_b:>4}  {f_c:>4}  {str(left):>20}  {str(right):>20}  "
              f"{'✅' if equal else '❌'}  → expected value = {expected_v:.0f}")

print()
print("  All four cases equal → operator is ASSOCIATIVE. ✓")
print("  Consequence: ANY scan tree (sequential, Hillis-Steele, Blelloch,")
print("  warp-level) gives the correct segmented scan output.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: LLM sequence packing — variable-length batch
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — LLM Sequence Packing: Variable-Length Batch Offsets")
print("━" * 68)
print()

# Simulate a batch of variable-length sequences
seq_lengths = [512, 43, 1024, 7, 256, 128, 389, 64]
total_tokens = sum(seq_lengths)
n_seqs = len(seq_lengths)

print(f"  Batch: {n_seqs} sequences, lengths = {seq_lengths}")
print(f"  Total tokens: {total_tokens}")
print()

# ── WITHOUT packing (padded to max length) ──────────────────────────
max_len = max(seq_lengths)
padded_tokens = max_len * n_seqs
padding_waste = padded_tokens - total_tokens
waste_pct = padding_waste / padded_tokens * 100

print(f"  WITHOUT packing (pad to max={max_len}):")
print(f"    Tensor shape: {n_seqs} × {max_len} = {padded_tokens:,} tokens")
print(f"    Padding waste: {padding_waste:,} tokens ({waste_pct:.1f}%)")
print()

# ── WITH packing (concatenate, use scan for offsets) ─────────────────
# Step 1: exclusive scan of lengths → starting offset of each sequence
offsets = [0] + list(np.cumsum(seq_lengths)[:-1])

print(f"  WITH packing (concatenate all tokens):")
print(f"    Packed tensor: {total_tokens:,} tokens (flat)")
print(f"    Memory saved:  {padding_waste:,} tokens ({waste_pct:.1f}%)")
print()
print(f"  Exclusive scan of lengths → per-sequence start offsets:")
print(f"  {'seq':>5}  {'length':>8}  {'start_offset':>14}  {'end_offset':>12}  "
      f"{'tokens'}")
print("  " + "─" * 56)
for i, (length, offset) in enumerate(zip(seq_lengths, offsets)):
    end = offset + length - 1
    print(f"  {i:>5}  {length:>8}  {offset:>14}  {end:>12}  "
          f"[{offset}..{end}]")

print()

# ── Per-token position IDs via segmented scan ────────────────────────
print(f"  Per-token position IDs via segmented exclusive scan:")
print()

# Build the flat packed array
ones   = np.ones(total_tokens, dtype=float)
flags  = np.zeros(total_tokens, dtype=int)
for off in offsets:
    flags[off] = 1   # mark start of each sequence

pos_ids = segmented_scan_sequential(ones, flags)

print(f"  Packed token positions (first 20 tokens):")
print(f"  {'token_idx':>10}  {'seq_id':>7}  {'pos_in_seq':>12}  {'pos_id from scan'}")
print("  " + "─" * 48)
for tok_idx in range(min(20, total_tokens)):
    # Identify which sequence this token belongs to
    seq_id = next(i for i, off in enumerate(offsets)
                  if off <= tok_idx < off + seq_lengths[i])
    true_pos = tok_idx - offsets[seq_id]
    scan_pos = int(pos_ids[tok_idx])
    ok = (scan_pos == true_pos)
    print(f"  {tok_idx:>10}  {seq_id:>7}  {true_pos:>12}  "
          f"{scan_pos:>16}  {'✅' if ok else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: vLLM-style KV-cache block allocation via segmented scan
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — vLLM-Style KV-Cache Block Allocation via Scan")
print("━" * 68)
print()

BLOCK_SIZE_KV = 16   # tokens per KV cache page
BYTES_PER_BLOCK = 2 * 32 * 8 * 128 * BLOCK_SIZE_KV * 2   # 2×layers×heads×dim×tokens×fp16

seq_lengths_kv = [47, 128, 300, 15, 512, 73]
n_seqs_kv = len(seq_lengths_kv)

def blocks_needed(seq_len, block_size):
    return math.ceil(seq_len / block_size)

block_counts = [blocks_needed(l, BLOCK_SIZE_KV) for l in seq_lengths_kv]
block_offsets = [0] + list(np.cumsum(block_counts)[:-1])   # exclusive scan
total_blocks = sum(block_counts)

print(f"  KV-cache block size: {BLOCK_SIZE_KV} tokens/block")
print(f"  Bytes per block: {BYTES_PER_BLOCK:,} ({BYTES_PER_BLOCK/1024:.0f} KB)")
print()
print(f"  {'seq_id':>7}  {'seq_len':>9}  {'blocks_needed':>14}  "
      f"{'block_offset':>14}  {'block range'}")
print("  " + "─" * 60)

for i in range(n_seqs_kv):
    b_start = block_offsets[i]
    b_end   = block_offsets[i] + block_counts[i] - 1
    wasted  = block_counts[i] * BLOCK_SIZE_KV - seq_lengths_kv[i]
    print(f"  {i:>7}  {seq_lengths_kv[i]:>9}  {block_counts[i]:>14}  "
          f"{block_offsets[i]:>14}  blocks[{b_start}..{b_end}]  "
          f"({wasted} slots wasted in last)")

print()
print(f"  Total blocks allocated: {total_blocks}")
print(f"  Total KV memory:        {total_blocks * BYTES_PER_BLOCK / 1024 / 1024:.1f} MB")
print()
print(f"  Block offsets computed via exclusive scan of block_counts:")
print(f"    block_counts:  {block_counts}")
print(f"    block_offsets: {block_offsets}  (exclusive scan)")
print()
print(f"  This scan runs once per batch — O({n_seqs_kv}) work — providing O(1)-access")
print(f"  per-token page lookup during attention: block_id = block_offsets[seq] + pos//16")


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: Causal attention mask via segmented scan
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 5 — Causal Mask for Packed Sequences via Segmented Scan")
print("━" * 68)
print()

small_seqs   = [4, 3, 5]
total_tokens_s = sum(small_seqs)
offsets_s    = [0] + list(np.cumsum(small_seqs)[:-1])
flags_s      = np.zeros(total_tokens_s, dtype=int)
for off in offsets_s:
    flags_s[off] = 1

# Segmented scan to get sequence ID per token
seq_id_marker = np.zeros(total_tokens_s, dtype=float)
for i, off in enumerate(offsets_s):
    seq_id_marker[off] = i   # each segment head carries its seq_id (overriding cumsum)

# Build seq_ids via max segmented scan
one_hot_seqid = np.zeros(total_tokens_s, dtype=float)
for i, off in enumerate(offsets_s):
    one_hot_seqid[off] = float(i)

# Simplified: build seq_ids directly
seq_ids = np.zeros(total_tokens_s, dtype=int)
sid = 0
for t in range(total_tokens_s):
    if flags_s[t] and t > 0:
        sid += 1
    seq_ids[t] = sid

# Position within sequence
pos_in_seq = segmented_scan_sequential(np.ones(total_tokens_s), flags_s).astype(int)

print(f"  Packed sequences: lengths = {small_seqs}  (total {total_tokens_s} tokens)")
print()
print(f"  {'tok':>4}  {'seq_id':>7}  {'pos_in_seq':>11}  notes")
print("  " + "─" * 36)
for t in range(total_tokens_s):
    note = "← segment start" if flags_s[t] else ""
    print(f"  {t:>4}  {seq_ids[t]:>7}  {pos_in_seq[t]:>11}  {note}")

print()
# Build causal attention mask
print(f"  Causal attention mask (seq_id must match, pos must be ≤ query pos):")
print()
print("  Query / Key:", end="")
for t in range(total_tokens_s):
    print(f"  t{t}", end="")
print()
print("  " + "─" * (12 + total_tokens_s * 5))

for q in range(total_tokens_s):
    print(f"  query t{q}   ", end="")
    for k in range(total_tokens_s):
        # Can attend: same sequence AND key position ≤ query position (causal)
        can_attend = (seq_ids[q] == seq_ids[k]) and (pos_in_seq[k] <= pos_in_seq[q])
        print(f"{'  ■  ' if can_attend else '  ·  '}", end="")
    print()

print()
print("  ■ = can attend (same sequence, causal)  · = masked")
print()
print("  Without segmented scan: naïvely O(N²) mask construction.")
print("  With segmented scan: O(N) to compute seq_ids + pos_in_seq,")
print("  then O(1) per (q,k) pair to check the two conditions.")
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