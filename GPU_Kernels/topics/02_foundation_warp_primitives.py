"""
Warp Primitives — __shfl, Vote, Ballot & Warp-Level Reduction
==============================================================

The warp is CUDA's atom of execution. Thirty-two threads in a warp
execute the same instruction in lockstep under the SIMT model, and
NVIDIA exposes a set of hardware instructions that let those threads
communicate with each other at register speed — zero SMEM, zero global
memory, zero synchronisation cost.

Warp primitives are the lowest-level building block above raw arithmetic.
Every fast reduction, every online softmax, every warp-level prefix scan,
every ballot-driven conditional execution path in cuBLAS, CUTLASS,
FlashAttention, and custom transformer kernels is built from exactly
three families of primitives:

    __shfl_sync   — move a register value from any lane to any other
    vote          — test a predicate across all 32 lanes simultaneously
    reduce        — hardware-accelerated warp-level reductions (Hopper+)

Understanding these primitives is understanding the *inner loop* of
every high-performance GPU kernel written in the last decade.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Warp Primitives — Shuffle, Vote, Ballot & Warp-Level Reductions"
DISPLAY_NAME = "02 · Warp Primitives"
ICON         = "⚡"
SUBTITLE     = "__shfl / __vote / __ballot — Register-Speed Intra-Warp Communication"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE SIMT EXECUTION MODEL & THE WARP

### SIMT: Single Instruction, Multiple Threads

    CUDA's hardware executes threads in groups of 32 called WARPS under
    the SIMT (Single Instruction, Multiple Threads) model.

    SIMT is NOT the same as SIMD:
        SIMD (CPU AVX):   programmer sees explicit vector registers, width fixed.
        SIMT (GPU warp):  programmer writes scalar code per thread; hardware
                          executes the same instruction across all 32 lanes in
                          parallel, each with its own operand values.

    What "same instruction, different data" means concretely:

        Thread 0:  add r1, r2  →  r3    (r1=1.0, r2=2.0, r3=3.0)
        Thread 1:  add r1, r2  →  r3    (r1=5.0, r2=7.0, r3=12.0)
        ...
        Thread 31: add r1, r2  →  r3    (r1=99.0, r2=1.0, r3=100.0)

    All 32 "add" instructions fire in a single clock cycle.
    Each thread has its own copy of every register — 32 distinct r1, r2, r3.

### The Warp Schedule

    An SM (Streaming Multiprocessor) is timeshared by many warps:
        H100 SM: up to 64 concurrent warps (2048 threads)

    Each clock cycle, the warp scheduler selects an ELIGIBLE warp
    (one whose next instruction has all operands ready) and issues it.

    A warp stalls when:
        - Waiting for a global memory load (300–600 ns latency)
        - Waiting for a SMEM bank conflict resolution
        - Waiting for a long-latency arithmetic instruction (e.g., sqrt, div)

    The key to hiding latency: have MANY warps in flight so the scheduler
    can always find an eligible one while others stall.
    This is occupancy — the reason we care about it from module 20.

### Warp Lane Identity

    Every thread knows its position within the warp:
        lane_id  = threadIdx.x % 32           (0–31)
        warp_id  = threadIdx.x / 32           (0 .. warps_per_block - 1)

    The lane_id is the fundamental index for ALL warp primitives.
    All __shfl, vote, and ballot operations are defined in terms of lane IDs.

### Warp Divergence

    When threads in a warp take different branches, the warp DIVERGES:

        if (lane_id < 16):      # threads 0–15 go here
            do_work_A()
        else:                   # threads 16–31 go here
            do_work_B()

    The hardware executes BOTH paths serially, masking inactive threads:
        Pass 1: execute do_work_A() with lanes 0–15 active, 16–31 masked
        Pass 2: execute do_work_B() with lanes 16–31 active, 0–15 masked

    Cost: 2× cycles for both paths vs 1× for one path.
    Worst case (every thread different): 32 serial passes → 32× slowdown.

    RECONVERGENCE: after the if/else, all threads are active again.
    Modern GPUs (Volta+) support INDEPENDENT THREAD SCHEDULING (ITS):
        Each thread has its own program counter.
        __syncwarp() is needed to explicitly reconverge a warp.
        Without it, threads in different branches can freely diverge
        and reconverge in programmer-controlled ways.

    PRE-VOLTA (Pascal and earlier): implicit warp-level synchronisation.
    POST-VOLTA (Volta, Turing, Ampere, Hopper):
        Must use __syncwarp(mask) to ensure intra-warp visibility.
        __shfl_sync, __ballot_sync, __any_sync, __all_sync all take a MASK.

### The Active Mask: Why It Matters

    The MASK argument to every _sync primitive is a 32-bit integer
    where bit i indicates whether lane i participates in the operation.

    Correct usage:
        unsigned mask = __activemask();  // get currently active lanes
        float val = __shfl_sync(mask, src, peer_lane);

    The mask serves two purposes:
        1. CORRECTNESS: tells hardware which lanes are participating.
           Mixing masked and unmasked lanes causes undefined behaviour on Volta+.
        2. PARTIAL WARP OPS: allows operations on subsets of a warp
           (e.g., only the first 16 lanes, or a specific pattern).

    0xFFFFFFFF (all ones) = all 32 lanes participate.
    0x0000FFFF (lower 16) = only lanes 0–15 participate.


##### PART 2 — __shfl_sync: REGISTER-SPEED LANE COMMUNICATION

### The Core Concept

    __shfl_sync lets any thread read any other thread's register
    without going through shared memory or global memory.

    Speed:        register file bandwidth — essentially free
    Latency:      ~1–4 cycles (vs 6 ns for SMEM, 300 ns for HBM)
    Requirement:  both source and destination threads must be in the SAME warp

    Without __shfl:
        Thread A wants Thread B's value → write to SMEM → __syncwarp → read
        Cost: SMEM write + barrier + SMEM read = 3 operations

    With __shfl:
        float val = __shfl_sync(mask, my_val, src_lane);
        Cost: 1 instruction. Hardware routes the register directly.

### The Four Shuffle Variants

    All variants have the signature:
        T __shfl_XXX_sync(unsigned mask, T var, int param, int width=32)

    T can be: int, unsigned int, long, unsigned long, float, double.
    width: treat warp as groups of 'width' lanes (must be power of 2, ≤ 32).

    ┌─────────────────────────────────────────────────────────────────┐
    │  VARIANT          WHAT LANE I RECEIVES                          │
    ├─────────────────────────────────────────────────────────────────┤
    │  __shfl_sync      var from lane 'srcLane' (absolute)            │
    │  __shfl_up_sync   var from lane (i - delta) [shift right]       │
    │  __shfl_down_sync var from lane (i + delta) [shift left]        │
    │  __shfl_xor_sync  var from lane (i XOR laneMask)                │
    └─────────────────────────────────────────────────────────────────┘

### __shfl_sync(mask, var, srcLane, width=32)

    Every active lane receives the value of 'var' from lane 'srcLane'.

        // All 32 lanes get lane 0's value
        float val = __shfl_sync(0xFFFFFFFF, my_val, 0);

    Use cases:
        - BROADCAST: lane 0 computes a scalar result, broadcasts to all.
        - GATHER: each lane reads from a specific source lane.

    Diagram (srcLane = 5):
        Before:   lane0=A  lane1=B  lane2=C  ... lane5=F  ...
        After:    lane0=F  lane1=F  lane2=F  ... lane5=F  ...
        All lanes receive lane 5's value.

### __shfl_up_sync(mask, var, delta, width=32)

    Lane i receives var from lane (i - delta).
    Lanes 0..(delta-1) are UNCHANGED (no source exists below them).

        // Each lane gets the value from 2 positions to its left
        float val = __shfl_up_sync(0xFFFFFFFF, my_val, 2);

    Use cases:
        - PREFIX SCAN: propagate running sum leftward across lanes.
        - LOOKBACK: each lane inspects its predecessor.

    Diagram (delta = 2):
        Before:   L0=a  L1=b  L2=c  L3=d  L4=e  L5=f  ...
        After:    L0=a  L1=b  L2=a  L3=b  L4=c  L5=d  ...
        Lanes 0,1 unchanged; lane k gets lane k-2's original value.

### __shfl_down_sync(mask, var, delta, width=32)

    Lane i receives var from lane (i + delta).
    Lanes (width-delta)..(width-1) are UNCHANGED.

        // The classic warp reduction step
        float val = __shfl_down_sync(0xFFFFFFFF, my_val, offset);

    Use cases:
        - WARP REDUCTION: the PRIMARY tool for sum, max, min, dot products.
        - PIPELINE: pass values forward in a processing chain.

    Diagram (delta = 16, first step of warp reduction):
        Before:   L0=a  L1=b  ... L16=q  L17=r  ... L31=z
        After:    L0=a+q  L1=b+r  ... L15=p+z  L16=q  ... L31=z
        Active lanes (0..15) each add their own value + lane[i+16]'s value.

### __shfl_xor_sync(mask, var, laneMask, width=32)

    Lane i receives var from lane (i XOR laneMask).
    Since XOR is symmetric: if lane i reads from lane j, then lane j reads
    from lane i simultaneously. This is a BUTTERFLY pattern.

        // Classic butterfly reduction step
        float val = __shfl_xor_sync(0xFFFFFFFF, my_val, offset);

    Use cases:
        - BUTTERFLY REDUCTION: equivalent to __shfl_down for reduction.
        - ALL-TO-ALL EXCHANGE: each lane has all values after log2(32) steps.
        - MATRIX TRANSPOSE: rearrange data within a warp.

    Diagram (laneMask = 1, pairs lanes 0↔1, 2↔3, ...):
        Before:   L0=a  L1=b  L2=c  L3=d  ...
        After:    L0=b  L1=a  L2=d  L3=c  ...
        Each adjacent pair swaps values. One instruction, 16 simultaneous swaps.

### Width Parameter: Sub-Warp Operations

    The width parameter (default 32) divides the warp into independent groups.

        float val = __shfl_down_sync(mask, my_val, 8, 16);
        // Warp treated as 2 groups of 16. Lane indices wrap within groups.
        // Lane 8 gets lane 16 (8+8 mod 16 = 8, within same group)

    Width values:
        width=32:  operate on the full warp (standard)
        width=16:  two independent groups of 16
        width=8:   four independent groups of 8
        width=4:   eight independent groups of 4
        width=2:   sixteen pairs

    Use case: process multiple independent reductions in one warp.
    Example: 4 independent 8-thread histograms computed in parallel.


##### PART 3 — WARP VOTE FUNCTIONS: __any, __all, __ballot

### Why Vote Functions Exist

    Before vote functions, checking "did ANY thread encounter condition X?"
    required writing to shared memory, syncing, then reading back. Expensive.

    Vote functions collapse a boolean predicate across all 32 lanes in
    a SINGLE instruction and return the result to all lanes simultaneously.

### __any_sync(mask, predicate)

    Returns 1 if ANY active lane has predicate == true. 0 otherwise.

        // Is any lane out of bounds?
        int oob = __any_sync(0xFFFFFFFF, my_index >= N);
        if (oob) handle_edge_case();

    Semantics:
        result = (predicate_lane0 | predicate_lane1 | ... | predicate_lane31) != 0

    Use case: early exit. If no lane needs special handling, skip the branch.
    If even one lane needs it, all warps take the branch (SIMT requirement).

### __all_sync(mask, predicate)

    Returns 1 if ALL active lanes have predicate == true.

        // Are all elements positive? (safe to take sqrt without guard)
        int all_pos = __all_sync(0xFFFFFFFF, val > 0.0f);
        if (all_pos) result = sqrtf(val);  // fast path
        else          result = safe_sqrt(val);  // guarded path

    Semantics:
        result = (predicate_lane0 & predicate_lane1 & ... & predicate_lane31) != 0

    Use case: optimise hot paths. Take the unguarded code path only when
    ALL threads satisfy the precondition.

### __ballot_sync(mask, predicate) → unsigned int

    Returns a 32-bit integer where bit i is set if lane i's predicate is true.
    This is the richest vote primitive — it preserves per-lane information.

        unsigned ballot = __ballot_sync(0xFFFFFFFF, my_val > threshold);
        // ballot = 0b...10100011 means lanes 0,1,5,7 exceeded threshold

    Derived operations on the ballot result:
        __popc(ballot)                 = number of lanes with true predicate
        __ffs(ballot) - 1              = lowest-numbered true lane
        __clz(__brev(ballot)) or ...   = highest-numbered true lane
        ballot & (ballot - 1)          = clear lowest set bit

    Use cases:
        - COUNT: how many threads need a branch? (for work stealing)
        - COMPACT: build a prefix sum from ballot to pack sparse results
        - FIND FIRST: which lane has the minimum/maximum?
        - CONDITIONAL EXECUTION: which sub-lanes of a warp take path A?

### __activemask() → unsigned int

    Returns a bitmask of currently ACTIVE (non-exited, non-diverged) lanes.

        unsigned mask = __activemask();

    NOT equivalent to 0xFFFFFFFF in divergent code.
    Essential when writing primitives that must work correctly in
    conditionally-executed code (e.g., inside an if/else branch).

    Rule: always call __activemask() at the point of use and pass that
    result to _sync primitives. Do NOT cache the result across divergent code.

### __match_any_sync and __match_all_sync (Volta+)

    __match_any_sync(mask, val):
        Returns a bitmask of lanes that have the same value as the calling lane.
        Lane 3 and Lane 7 both have val=42 → bits 3 and 7 are set in both results.

    Use case: grouping threads by value for cooperative work.
    Example: all threads hashing to bucket 5 form a cooperative group
    and one thread (elected via __ffs) does the actual insert.

    __match_all_sync(mask, val, *pred):
        Returns the mask if ALL active lanes have the same value, else 0.
        *pred is set to 1 if all match.

    Use case: warp-level coalescing of atomic operations.


##### PART 4 — WARP REDUCE PATTERNS: SUM, MAX, ARGMAX, DOT

### The Canonical Warp Reduction

    Goal: compute sum (or max, min, product) across all 32 lane values.
    Result available in lane 0 (or broadcast to all lanes).

    Using __shfl_down_sync (the standard idiom):

        float warp_reduce_sum(float val) {
            for (int offset = 16; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            return val;  // lane 0 holds the result
        }

    Step-by-step for 8 lanes (simplified, same pattern for 32):

        Step 0 (offset=4):
            L0 += L4  →  a+e
            L1 += L5  →  b+f
            L2 += L6  →  c+g
            L3 += L7  →  d+h
            L4..L7 unchanged (become irrelevant)

        Step 1 (offset=2):
            L0 += L2  →  a+e+c+g
            L1 += L3  →  b+f+d+h
            L2..L3 unchanged

        Step 2 (offset=1):
            L0 += L1  →  a+b+c+d+e+f+g+h   ← TOTAL SUM
            Lane 0 holds the complete reduction.

    Total cost: log2(32) = 5 __shfl_down instructions per warp.
    Compare to sequential: 31 additions + overhead.

### Reduction Tree Visualisation

    Values:  v0  v1  v2  v3  v4  v5  v6  v7  (8-lane example)
             │   │   │   │   │   │   │   │
    offset=4 ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
             v0+v4  v1+v5  v2+v6  v3+v7  ·  ·  ·  ·
             │      │      │      │
    offset=2 ▼      ▼      ▼      ▼
             v0+v4+v2+v6  v1+v5+v3+v7  ·  ·
             │            │
    offset=1 ▼            ▼
             SUM           ·
             (lane 0)

    Each row = 1 __shfl_down instruction (all active reductions in parallel).
    5 rows for 32 lanes: 5 total instructions, regardless of N.

### Reduce-Then-Broadcast Pattern

    Often you want EVERY lane to have the warp-wide result, not just lane 0.

        float warp_reduce_sum_broadcast(float val) {
            // Step 1: reduce to lane 0
            for (int offset = 16; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            // Step 2: broadcast from lane 0
            val = __shfl_sync(0xFFFFFFFF, val, 0);
            return val;  // ALL lanes have the total
        }

    Use case: each lane normalises its value by the warp-wide sum.
    Example: softmax denominator — one __shfl_sync broadcast after reduction.

### Warp Max and Argmax

    Max reduction:
        float warp_max(float val) {
            for (int offset = 16; offset > 0; offset >>= 1)
                val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
            return val;  // lane 0 has global max
        }

    Argmax (max value AND its lane index):
        // Encode (value, lane_id) as a 64-bit int for atomic argmax
        struct ValIdx { float val; int idx; };

        ValIdx warp_argmax(float val, int idx) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                float  peer_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
                int    peer_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
                if (peer_val > val) { val = peer_val; idx = peer_idx; }
            }
            return {val, idx};  // lane 0 has global argmax
        }

    TWO __shfl_down calls per step (val + idx), 5 steps = 10 shuffle ops.
    Critical pattern for: top-k sampling, attention head selection, beam search.

### Warp Dot Product

    dot(A, B) = sum of A[i] * B[i] for i in 0..31 (one element per lane):

        float warp_dot(float a, float b) {
            float prod = a * b;                // each lane multiplies
            return warp_reduce_sum(prod);       // then reduce
        }

    5 shuffles (the reduce), 1 FMA per lane. Extremely efficient.

    Use case: computing attention scores q·k for one head when q and k
    each have 32 dimensions, one dimension per lane.

### Hardware Warp Reduce (Hopper / sm_90)

    NVIDIA Hopper (H100) introduced hardware WARP REDUCE instructions:
        __reduce_add_sync(mask, val)   // integer add
        __reduce_min_sync(mask, val)   // integer min
        __reduce_max_sync(mask, val)   // integer max
        __reduce_and_sync(mask, val)   // bitwise AND
        __reduce_or_sync(mask, val)    // bitwise OR
        __reduce_xor_sync(mask, val)   // bitwise XOR

    These are single PTX instructions (REDUX.SYNC.*) vs 5 shfl instructions.
    ~2–3× faster than the shfl loop for integer types.
    Float reductions still use the shfl pattern (no hardware fp REDUX as of H100).

    PTX equivalent:
        redux.sync.add.u32  %r1, %r2, 0xffffffff;


##### PART 5 — WARP PREFIX SCAN (EXCLUSIVE & INCLUSIVE)

### What Prefix Scan Is

    Given input   [a, b, c, d, e, f, g, h, ...]
    Inclusive:    [a, a+b, a+b+c, a+b+c+d, ...]  (element i = sum 0..i)
    Exclusive:    [0, a, a+b, a+b+c, ...]          (element i = sum 0..i-1)

    Prefix scan is the foundational operation behind:
        - Stream compaction (how many elements before mine pass a filter?)
        - Dynamic work scheduling (what offset does thread i write to?)
        - Histogram construction
        - CUDA's thrust::inclusive_scan / exclusive_scan
        - vLLM's block table allocation order

### Warp-Level Prefix Scan via __shfl_up_sync

    The key: __shfl_up(val, delta) gives each lane its predecessor's value.
    Add it to accumulate a running prefix.

    Inclusive prefix scan:
        float warp_inclusive_scan(float val) {
            for (int offset = 1; offset < 32; offset <<= 1) {
                float peer = __shfl_up_sync(0xFFFFFFFF, val, offset);
                if (lane_id >= offset) val += peer;
            }
            return val;  // lane i holds sum[0..i]
        }

    Exclusive: same, then shift right by 1 and set lane 0 = 0.

    Step trace (4 lanes: a=1, b=2, c=3, d=4):

        Initial:  L0=1  L1=2  L2=3  L3=4

        offset=1:
            L0 → no source (unchanged):  L0=1
            L1 += L0:  2+1 = 3
            L2 += L1:  3+2 = 5
            L3 += L2:  4+3 = 7
        After:    L0=1  L1=3  L2=5  L3=7

        offset=2:
            L0,L1 → no source (unchanged)
            L2 += L0:  5+1 = 6
            L3 += L1:  7+3 = 10
        After:    L0=1  L1=3  L2=6  L3=10  ← inclusive scan ✅

    Verification: [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10] ✅

    Cost: log2(32) = 5 __shfl_up operations.

### Segmented Scan

    Process multiple independent prefix scans within one warp simultaneously.

        // 4 segments of 8 elements each — all scanned in parallel
        float segmented_scan(float val, int seg_size) {
            for (int offset = 1; offset < seg_size; offset <<= 1) {
                float peer = __shfl_up_sync(0xFFFFFFFF, val, offset);
                // Only add if source is in the same segment
                if ((lane_id % seg_size) >= offset)
                    val += peer;
            }
            return val;
        }

    Use case: computing per-row softmax for a matrix with row_width ≤ 32.
    Each row of 8 logits → one segment → 4 independent softmax denominators.


##### PART 6 — WARP PRIMITIVES IN ML KERNELS

### Warp-Level Softmax (the Online Algorithm)

    Standard softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    For a vector x of length 32 (one element per lane):

        // Step 1: warp max (numerically stable shift)
        float m = warp_max(x);                 // 5 shfl_down → max

        // Step 2: each lane computes shifted exp
        float e = expf(x - m);

        // Step 3: warp sum of exps (denominator)
        float s = warp_reduce_sum(e);           // 5 shfl_down → sum

        // Step 4: broadcast denominator
        s = __shfl_sync(0xFFFFFFFF, s, 0);      // 1 shfl → all lanes

        // Step 5: normalise
        float result = e / s;

    Total: 11 warp instructions (5 + 5 + 1). No SMEM at all.
    This exact pattern lives inside every FlashAttention implementation.

### Online Softmax Across Tiles (Flash Attention)

    When sequence length > 32, softmax must be computed across TILES.
    The "safe softmax" trick accumulates running statistics:

        // Process tile by tile, accumulate:
        float running_max  = -INFINITY;
        float running_sum  = 0.0f;
        float running_out  = 0.0f;

        for each tile t:
            float tile_vals[T];    // load tile from SMEM
            float tile_max = warp_max(tile_vals);       // per-tile max

            // Rescale previous accumulation if new max is larger
            float new_max = fmaxf(running_max, tile_max);
            running_sum   = running_sum * expf(running_max - new_max)
                          + warp_reduce_sum(expf(tile_vals - new_max));
            running_out   = running_out * expf(running_max - new_max)
                          + warp_reduce_sum(V * expf(tile_vals - new_max));
            running_max   = new_max;

        output = running_out / running_sum;    // normalise once at end

    This pattern — one warp reduce per tile — is the inner loop of FA-2 and FA-3.
    The warp primitives eliminate ALL SMEM traffic for the running statistics.

### Layer Norm Forward (Welford Online Variance)

    Welford's algorithm computes mean and variance in a single pass
    without catastrophic cancellation:

        float count = 0, mean = 0, M2 = 0;
        for each element x (per thread):
            count  += 1;
            float delta = x - mean;
            mean   += delta / count;
            float delta2 = x - mean;
            M2     += delta * delta2;

        // Parallel merge (warp-level) using __shfl_down:
        // Merge partial (count, mean, M2) from peer lane into own
        for (int offset = 16; offset > 0; offset >>= 1) {
            float  peer_count = __shfl_down_sync(mask, count, offset);
            float  peer_mean  = __shfl_down_sync(mask, mean,  offset);
            float  peer_M2    = __shfl_down_sync(mask, M2,    offset);
            // Welford parallel merge formula:
            float  delta = peer_mean - mean;
            float  new_count = count + peer_count;
            mean  += delta * peer_count / new_count;
            M2    += peer_M2 + delta*delta * count*peer_count / new_count;
            count  = new_count;
        }
        float variance = M2 / count;

    3 __shfl_down values per step × 5 steps = 15 shuffle instructions.
    Computes mean AND variance in one warp-reduce pass (vs 2 passes naïvely).

### Ballot-Based Stream Compaction

    Given 32 input values, keep only those passing a filter (e.g., > threshold).
    Write passing values to a compact output array.

        bool keep = (my_val > threshold);
        unsigned ballot = __ballot_sync(0xFFFFFFFF, keep);

        // My output position = number of lanes before me that also keep
        unsigned prefix_mask = ballot & ((1u << lane_id) - 1);
        int my_out_pos = __popc(prefix_mask);  // popcount of lower lanes

        // Total outputs this warp produces
        int total_keep = __popc(ballot);

        // Atomic to claim output slot for the warp
        int warp_out_start = atomicAdd(output_count, total_keep);

        // Each thread writes to its compacted slot
        if (keep)
            output[warp_out_start + my_out_pos] = my_val;

    This is exactly how CUDA's stream compaction (thrust::copy_if) works at the
    warp level. The ballot + __popc combo replaces a prefix scan over a boolean
    array, performing the entire compaction in 2–3 warp instructions.

### Warp-Level Top-K (Beam Search, Sampling)

    Find the k largest values within a warp (k ≤ 32):

        // Strategy: bitonic sort within the warp using __shfl_xor_sync
        // Round 1: compare-and-swap pairs (0↔1, 2↔3, ...)
        // Round 2: compare-and-swap (0↔2, 1↔3, ...)
        // ... log2(32)=5 rounds → fully sorted in 5 × log2(32)/2 = ~20 shfl_xor ops

        for (int k = 2; k <= 32; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                int peer_lane = lane_id ^ j;
                float peer_val = __shfl_xor_sync(0xFFFFFFFF, my_val, j);
                bool ascending = ((lane_id & k) == 0);
                if ((ascending && my_val < peer_val) ||
                    (!ascending && my_val > peer_val))
                    my_val = peer_val;
            }
        }
        // lane 0 has the minimum, lane 31 has the maximum
        // top-k: lanes 32-k .. 31 hold the k largest values

    Used in: vLLM's sampling kernel, beam search score selection.


##### PART 7 — DIVERGENCE, MASKS & CORRECTNESS RULES

### The Volta+ Breaking Change

    Pre-Volta (Kepler, Maxwell, Pascal) used IMPLICIT warp synchronisation:
        All 32 threads in a warp were ALWAYS synchronised at instruction boundaries.
        __shfl (old API, no mask) always worked correctly.

    Volta+ introduced INDEPENDENT THREAD SCHEDULING (ITS):
        Each thread has its own program counter.
        Threads in the same warp can be at DIFFERENT instruction positions.
        __shfl_sync (NEW API, with mask) is required for correctness.

    The old API (__shfl without _sync) was DEPRECATED in CUDA 9 (Volta).
    It is UNDEFINED BEHAVIOUR on Volta+ if any thread in the mask is inactive.

    Migration rule: replace ALL __shfl, __any, __all, __ballot with
    their _sync variants and pass the correct active mask.

### Mask Correctness Rules

    Rule 1 — Never use 0xFFFFFFFF in divergent code:
        // WRONG: if some lanes are diverged, they're not in 0xFFFFFFFF
        if (condition) {
            float x = __shfl_sync(0xFFFFFFFF, val, 0);  // UB if lanes diverged!
        }

        // CORRECT:
        if (condition) {
            unsigned mask = __activemask();
            float x = __shfl_sync(mask, val, 0);
        }

    Rule 2 — All threads in the mask must execute the __shfl_sync:
        // WRONG: lane 0 might not reach this line if it diverged away
        unsigned mask = 0b00000000111111111111111111111111;  // lanes 0-29
        float x = __shfl_sync(mask, val, 0);  // if lane 0 is inactive → UB

    Rule 3 — Use cooperative groups for complex patterns:
        // Cooperative groups (CUDA 9+) handle mask computation automatically
        namespace cg = cooperative_groups;
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        float x = cg::reduce(warp, val, cg::plus<float>());

### __syncwarp() vs __syncthreads()

    __syncwarp(mask):
        - Ensures all threads in 'mask' have completed their outstanding
          memory transactions and instructions before any proceed.
        - Warp-level only: does NOT synchronise across warps in a block.
        - Cost: ~1–4 cycles (much cheaper than __syncthreads).
        - Required: before reading SMEM written by another lane in same warp.

    __syncthreads():
        - Full thread block barrier. ALL warps in the block must reach it.
        - Cost: many cycles (proportional to block size).
        - Required: before reading SMEM written by a thread in a DIFFERENT warp.

    Common mistake: using __syncthreads() where __syncwarp() suffices.
    In a warp-only reduction kernel: __syncwarp() after each shfl step is
    sufficient and much cheaper.

### Warp Primitive Latency Summary (A100/H100 estimates)

    __shfl_sync     (any variant):    ~4 cycles
    __ballot_sync:                    ~4 cycles
    __any_sync / __all_sync:          ~4 cycles
    __popc (32-bit):                  1 cycle (single instruction)
    __ffs (find first set bit):       1 cycle
    __clz (count leading zeros):      1 cycle
    __syncwarp:                       ~5 cycles
    atomicAdd (global, uncontended):  ~100–300 cycles

    A full warp reduce (5 shfl steps): ~20 cycles
    A full warp scan (5 shfl steps):   ~20 cycles
    Both are essentially FREE compared to any memory operation.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Shuffle Variant Simulator — All Four __shfl Modes Traced": {
        "description": (
            "Simulate all four __shfl_sync variants (__shfl, __shfl_up, __shfl_down, "
            "__shfl_xor) with step-by-step lane-level traces. Show the butterfly, "
            "shift-left, shift-right, and broadcast patterns. Demonstrate the width "
            "parameter for sub-warp operations. Verify correctness against expected outputs."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 68)
print("  WARP SHUFFLE SIMULATOR — All Four __shfl_sync Variants")
print("=" * 68)
print()

WARP_SIZE = 32

# ─────────────────────────────────────────────────────────────────────
# Core shuffle primitives (simulated in Python)
# ─────────────────────────────────────────────────────────────────────

def shfl_sync(mask, values, src_lane, width=32):
    """
    __shfl_sync: every active lane receives values[src_lane].
    Broadcast of a single source lane to all others.
    src_lane is interpreted within each sub-group of 'width' threads.
    """
    result = values.copy()
    for lane in range(WARP_SIZE):
        if not (mask >> lane & 1):
            continue
        group_start = (lane // width) * width
        actual_src  = group_start + (src_lane % width)
        if mask >> actual_src & 1:
            result[lane] = values[actual_src]
    return result


def shfl_up_sync(mask, values, delta, width=32):
    """
    __shfl_up_sync: lane i receives values[i - delta].
    Lanes in positions 0..delta-1 within each group are UNCHANGED.
    """
    result = values.copy()
    for lane in range(WARP_SIZE):
        if not (mask >> lane & 1):
            continue
        group_start = (lane // width) * width
        pos_in_group = lane - group_start
        if pos_in_group >= delta:
            src = lane - delta
            if mask >> src & 1:
                result[lane] = values[src]
        # else: unchanged (no source to the left)
    return result


def shfl_down_sync(mask, values, delta, width=32):
    """
    __shfl_down_sync: lane i receives values[i + delta].
    Lanes in positions (width-delta)..(width-1) within each group unchanged.
    """
    result = values.copy()
    for lane in range(WARP_SIZE):
        if not (mask >> lane & 1):
            continue
        group_start = (lane // width) * width
        pos_in_group = lane - group_start
        if pos_in_group + delta < width:
            src = lane + delta
            if src < WARP_SIZE and (mask >> src & 1):
                result[lane] = values[src]
        # else: unchanged
    return result


def shfl_xor_sync(mask, values, lane_mask, width=32):
    """
    __shfl_xor_sync: lane i receives values[i XOR lane_mask].
    Symmetric butterfly pattern.
    """
    result = values.copy()
    for lane in range(WARP_SIZE):
        if not (mask >> lane & 1):
            continue
        group_start = (lane // width) * width
        pos_in_group = lane - group_start
        peer_pos = pos_in_group ^ lane_mask
        if peer_pos < width:
            src = group_start + peer_pos
            if src < WARP_SIZE and (mask >> src & 1):
                result[lane] = values[src]
    return result


def draw_lanes(values, n=16, label=""):
    """Print first n lanes in a compact table."""
    vals = [f"{v:4.0f}" for v in values[:n]]
    print(f"  {label:<18} │ " + " ".join(vals))


MASK_ALL = 0xFFFFFFFF

# ─────────────────────────────────────────────────────────────────────
# SECTION 1: __shfl_sync (broadcast)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — __shfl_sync: Broadcast from Source Lane")
print("━" * 68)
print()

lanes = np.arange(WARP_SIZE, dtype=float)  # lane i holds value i

print(f"  Initial register values (each lane holds its own lane_id):")
draw_lanes(lanes, label="Initial")
print()

for src in [0, 7, 15, 31]:
    out = shfl_sync(MASK_ALL, lanes, src_lane=src)
    draw_lanes(out, label=f"shfl(src={src})")

print()
print("  Observation: all lanes receive the single source lane's value.")
print("  Use case: broadcast a scalar computed by lane 0 to all other lanes.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: __shfl_up_sync (shift right, prefix scan building block)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — __shfl_up_sync: Shift Right (Exclusive Lookback)")
print("━" * 68)
print()

vals = np.random.default_rng(42).integers(1, 10, WARP_SIZE).astype(float)
print(f"  Input values:")
draw_lanes(vals, label="Input")
print()

for delta in [1, 2, 4, 8]:
    out = shfl_up_sync(MASK_ALL, vals, delta=delta)
    draw_lanes(out, label=f"shfl_up(δ={delta})")

print()
print("  Observation: lane i receives the value from lane (i - delta).")
print("  Lanes 0..delta-1 are UNCHANGED (no predecessor at that distance).")
print("  Use case: each lane inspects its predecessor — key for prefix scan.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: __shfl_down_sync (shift left, reduction building block)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — __shfl_down_sync: Shift Left (Reduction Building Block)")
print("━" * 68)
print()

print(f"  Input values:")
draw_lanes(vals, label="Input")
print()
print(f"  Step-by-step warp sum reduction using __shfl_down_sync:")
print()

acc = vals.copy()
for step, offset in enumerate([16, 8, 4, 2, 1]):
    peer = shfl_down_sync(MASK_ALL, acc, delta=offset)
    # Only lanes 0..(31-offset) add their peer's value
    for lane in range(WARP_SIZE - offset):
        acc[lane] = acc[lane] + peer[lane]
    draw_lanes(acc, label=f"  after δ={offset:<2}")
    # Show which lanes are "done"
    active = WARP_SIZE - offset
    print(f"    lanes 0..{active-1} accumulate; lanes {active}..31 unchanged")
    print()

print(f"  Lane 0 result:   {acc[0]:.0f}")
print(f"  True sum:        {vals.sum():.0f}  {'✅' if abs(acc[0]-vals.sum()) < 1e-6 else '❌'}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: __shfl_xor_sync (butterfly pattern)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — __shfl_xor_sync: Butterfly Pattern & All-to-All Exchange")
print("━" * 68)
print()

# Use a full 32-element array; display only first 8 for readability
small32 = np.concatenate([np.arange(8, dtype=float),
                           np.zeros(WARP_SIZE - 8)])  # lanes 8-31 padded with 0

print(f"  Input (first 8 of 32 lanes):  {small32[:8].astype(int).tolist()}")
print()
print(f"  XOR shuffle patterns (first 8 lanes shown):")
print()

for lm in [1, 2, 4]:
    out = shfl_xor_sync(MASK_ALL, small32, lane_mask=lm, width=8)
    print(f"  shfl_xor(mask=0b{lm:03b}={lm}):  {out[:8].astype(int).tolist()}")
    pairs = [(i, i ^ lm) for i in range(8) if i < (i ^ lm) and (i ^ lm) < 8]
    pair_str = "  exchanges: " + ", ".join(f"L{a}↔L{b}" for a, b in pairs)
    print(pair_str)
    print()

print("  Full butterfly reduction across all 32 lanes (XOR variant):")
acc_xor = vals.copy()
for step, offset in enumerate([16, 8, 4, 2, 1]):
    peer = shfl_xor_sync(MASK_ALL, acc_xor, lane_mask=offset)
    for lane in range(WARP_SIZE):
        if lane < offset:  # conventional: only lower lanes accumulate
            acc_xor[lane] = acc_xor[lane] + peer[lane]
print(f"  Lane 0 (XOR reduction):  {acc_xor[0]:.0f}  vs true: {vals.sum():.0f}")
print()
print("  Note: __shfl_xor exchanges BOTH directions simultaneously.")
print("  Lane A reads lane B, and lane B reads lane A — one instruction.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: Width parameter — sub-warp operations
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 5 — Width Parameter: Sub-Warp Independent Groups")
print("━" * 68)
print()

group_vals = np.arange(WARP_SIZE, dtype=float)

print("  shfl_down with different widths (sum reduction per group):")
print()

for width in [2, 4, 8, 16, 32]:
    n_groups = WARP_SIZE // width
    acc_w = group_vals.copy()
    for offset in [w for w in [16,8,4,2,1] if w < width]:
        peer = shfl_down_sync(MASK_ALL, acc_w, delta=offset, width=width)
        for lane in range(WARP_SIZE):
            pos = lane % width
            if pos + offset < width:
                acc_w[lane] += peer[lane]
    # Group sums: lane 0 of each group
    group_sums = [acc_w[g * width] for g in range(n_groups)]
    true_sums  = [sum(range(g*width, (g+1)*width)) for g in range(n_groups)]
    correct = all(abs(g-t) < 1e-6 for g,t in zip(group_sums, true_sums))
    print(f"  width={width:2d}: {n_groups:2d} independent groups of {width:2d}  "
          f"group_sums={[int(s) for s in group_sums[:4]]}{'...' if n_groups>4 else ''}  "
          f"{'✅' if correct else '❌'}")

print()
print("  Width=32 (default): one global reduction → one sum in lane 0.")
print("  Width=8:  four independent reductions, each group's sum in its lane 0.")
print("  Use case: 4 independent dot products or histogram buckets in one warp.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Vote & Ballot — __any, __all, __ballot Bit Manipulation": {
        "description": (
            "Simulate __any_sync, __all_sync, __ballot_sync and all derived bit "
            "operations (__popc, __ffs, __clz, prefix-from-ballot). Demonstrate stream "
            "compaction using ballot + __popc. Show __match_any for grouping threads by "
            "value. Implement warp-level election (first active lane) using __ffs."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  WARP VOTE & BALLOT SIMULATOR — __any / __all / __ballot")
print("=" * 68)
print()

WARP_SIZE = 32


# ─────────────────────────────────────────────────────────────────────
# Vote / ballot primitives (simulated)
# ─────────────────────────────────────────────────────────────────────

def any_sync(mask: int, predicates: list) -> int:
    """__any_sync: 1 if ANY active lane's predicate is true."""
    for lane in range(WARP_SIZE):
        if (mask >> lane & 1) and predicates[lane]:
            return 1
    return 0


def all_sync(mask: int, predicates: list) -> int:
    """__all_sync: 1 if ALL active lanes' predicates are true."""
    for lane in range(WARP_SIZE):
        if (mask >> lane & 1) and not predicates[lane]:
            return 0
    return 1


def ballot_sync(mask: int, predicates: list) -> int:
    """__ballot_sync: 32-bit int, bit i set iff lane i is active AND predicate[i] true."""
    result = 0
    for lane in range(WARP_SIZE):
        if (mask >> lane & 1) and predicates[lane]:
            result |= (1 << lane)
    return result


def popc(x: int) -> int:
    """__popc: population count (number of set bits)."""
    return bin(x).count('1')


def ffs(x: int) -> int:
    """__ffs: find first set bit (1-indexed, 0 if none). __ffs(x)-1 = lowest set bit index."""
    if x == 0:
        return 0
    return (x & -x).bit_length()


def clz(x: int, width=32) -> int:
    """__clz: count leading zeros in a 32-bit integer."""
    if x == 0:
        return width
    return width - x.bit_length()


def brev(x: int, width=32) -> int:
    """__brev: bit reverse a 32-bit integer."""
    result = 0
    for i in range(width):
        if x >> i & 1:
            result |= 1 << (width - 1 - i)
    return result


def prefix_from_ballot(ballot: int, lane: int) -> int:
    """Count set bits in ballot strictly below 'lane' (exclusive prefix from ballot)."""
    mask = (1 << lane) - 1
    return popc(ballot & mask)


MASK_ALL = 0xFFFFFFFF

np.random.seed(7)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: __any_sync and __all_sync
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — __any_sync & __all_sync: Warp-Wide Predicate Tests")
print("━" * 68)
print()

scenarios = [
    ("All positive",        np.abs(np.random.randn(WARP_SIZE))),
    ("Mixed (some neg)",    np.random.randn(WARP_SIZE)),
    ("All negative",        -np.abs(np.random.randn(WARP_SIZE))),
    ("One positive",        np.full(WARP_SIZE, -1.0, dtype=float)),
    ("One NaN lane",        np.random.randn(WARP_SIZE)),
]
scenarios[3][1][7]  = 5.0   # lane 7 is positive
scenarios[4][1][15] = float('nan')

print(f"  {'Scenario':<25}  {'any(>0)':>7}  {'all(>0)':>7}  "
      f"{'count>0':>8}  {'first>0':>9}  {'last>0':>8}")
print("  " + "─" * 64)

for name, vals in scenarios:
    preds  = [float(v) > 0 and not math.isnan(float(v)) for v in vals]
    a      = any_sync(MASK_ALL, preds)
    al     = all_sync(MASK_ALL, preds)
    ball   = ballot_sync(MASK_ALL, preds)
    count  = popc(ball)
    first  = ffs(ball) - 1 if ball else -1
    # highest set bit: 31 - clz(__brev(ball)) or just bit_length-1
    last   = ball.bit_length() - 1 if ball else -1
    print(f"  {name:<25}  {a:>7}  {al:>7}  {count:>8}  {first:>9}  {last:>8}")

print()
print("  any(>0): is at least one lane positive?")
print("  all(>0): are ALL lanes positive?")
print("  count>0: exactly how many lanes are positive? (__popc of ballot)")
print("  first>0: lowest-numbered lane that is positive (__ffs(ballot) - 1)")
print("  last>0:  highest-numbered lane that is positive (bit_length - 1)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: __ballot_sync — the full bitmask
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — __ballot_sync: Full Per-Lane Bitmask")
print("━" * 68)
print()

vals = np.random.randn(WARP_SIZE)
threshold = 0.5
preds = [float(v) > threshold for v in vals]
ballot = ballot_sync(MASK_ALL, preds)

print(f"  Threshold: {threshold}")
print(f"  Lane values (first 16):")
for i in range(0, 16, 8):
    row = "  " + "  ".join(f"L{j:02d}={vals[j]:+.2f}" for j in range(i, i+8))
    print(row)
print()
print(f"  __ballot_sync result: 0x{ballot:08X}  ({ballot:032b}b)")
print()

# Parse the ballot
set_lanes  = [i for i in range(WARP_SIZE) if ballot >> i & 1]
clear_lanes = [i for i in range(WARP_SIZE) if not (ballot >> i & 1)]

print(f"  Lanes with predicate TRUE  ({len(set_lanes):2d}): {set_lanes}")
print(f"  Lanes with predicate FALSE ({len(clear_lanes):2d}): {clear_lanes[:8]}...")
print()
print(f"  Derived operations on ballot 0x{ballot:08X}:")
print(f"    __popc(ballot)     = {popc(ballot):2d}  (number of matching lanes)")
print(f"    __ffs(ballot) - 1  = {ffs(ballot)-1:2d}  (first matching lane)")
print(f"    bit_length - 1     = {ballot.bit_length()-1:2d}  (last  matching lane)")
print(f"    ballot & (ballot-1)= 0x{(ballot & (ballot-1)):08X}  (clear lowest set bit)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Stream compaction via ballot + prefix
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Stream Compaction: ballot + __popc Prefix")
print("━" * 68)
print()

np.random.seed(42)
data = np.random.randint(0, 20, WARP_SIZE)
filter_threshold = 10
keep = [int(v) > filter_threshold for v in data]
ballot_keep = ballot_sync(MASK_ALL, keep)

print(f"  Input data (32 lanes):  {data.tolist()}")
print(f"  Filter: value > {filter_threshold}")
print(f"  Ballot: 0x{ballot_keep:08X}")
print()

compact_positions = {}
compact_output    = []

print(f"  Lane  Value  Keep?  Prefix(ballot<lane)  OutputIdx")
print("  " + "─" * 52)

for lane in range(WARP_SIZE):
    k = keep[lane]
    prefix = prefix_from_ballot(ballot_keep, lane)
    out_idx = prefix if k else -1
    if k:
        compact_output.append((out_idx, data[lane]))
    marker = f"   → output[{out_idx}]" if k else ""
    if lane < 12 or (k and lane < 20):  # print first 12 and any kept up to lane 20
        print(f"  L{lane:02d}   {data[lane]:3d}    {'✅' if k else '❌'}     {prefix:4d}               "
              f"{out_idx if out_idx >= 0 else '-':>3}{marker}")

total_kept = popc(ballot_keep)
print(f"  ...")
print()
print(f"  Total kept: {total_kept} / {WARP_SIZE}  (one atomicAdd to output_count)")
print()
compact_output.sort()
print(f"  Compact output: {[v for _, v in sorted(compact_output)]}")
true_compact = sorted([int(v) for v in data if v > filter_threshold])
print(f"  Expected:       {true_compact}")
print(f"  Correct: {'✅' if [v for _,v in sorted(compact_output)] == true_compact else '❌'}")
print()
print("  Cost: 1 __ballot_sync + 1 __popc per lane = 2 warp instructions total.")
print("  Compare: naïve approach needs a full prefix scan (5 shfl steps).")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: __match_any_sync — group threads by value
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — __match_any_sync: Group Lanes by Value")
print("━" * 68)
print()

def match_any_sync(mask: int, values: list) -> list:
    """
    __match_any_sync: for each active lane, returns a bitmask of all
    active lanes that have the SAME value.
    """
    results = [0] * WARP_SIZE
    for lane in range(WARP_SIZE):
        if not (mask >> lane & 1):
            continue
        match_mask = 0
        for other in range(WARP_SIZE):
            if (mask >> other & 1) and values[other] == values[lane]:
                match_mask |= (1 << other)
        results[lane] = match_mask
    return results

# Simulate lanes hashing to 4 buckets (hash table insertion scenario)
bucket_ids = np.random.choice([0, 1, 2, 3], size=WARP_SIZE)
match_masks = match_any_sync(MASK_ALL, bucket_ids.tolist())

print(f"  Scenario: 32 lanes each insert into one of 4 hash buckets.")
print(f"  Bucket assignments: {bucket_ids.tolist()}")
print()

for bucket in range(4):
    lanes_in = [i for i in range(WARP_SIZE) if bucket_ids[i] == bucket]
    if not lanes_in:
        continue
    # elected leader: lane with lowest ID in the group (__ffs - 1)
    group_mask = match_masks[lanes_in[0]]
    leader = ffs(group_mask) - 1
    print(f"  Bucket {bucket}: lanes = {lanes_in}")
    print(f"           match_mask = 0x{group_mask:08X}  ({popc(group_mask)} lanes)")
    print(f"           elected leader = lane {leader} (__ffs(mask) - 1)")
    print(f"           leader does ONE atomic insert for all {popc(group_mask)} threads")
    print()

print("  Pattern: coalesced atomics — instead of 32 independent atomicAdds,")
print("  group threads by target address and have one leader do the atomic.")
print(f"  Potential atomic reduction: {WARP_SIZE} → 4 operations (for this distribution)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: Partial mask operations — only some lanes participate
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 5 — Partial Mask Operations (Not All 32 Lanes Active)")
print("━" * 68)
print()

vals32 = np.arange(WARP_SIZE, dtype=float)

partial_masks = {
    "All 32 lanes  (0xFFFFFFFF)": MASK_ALL,
    "Lower 16      (0x0000FFFF)": 0x0000FFFF,
    "Upper 16      (0xFFFF0000)": 0xFFFF0000,
    "Even lanes    (0x55555555)": 0x55555555,
    "Odd  lanes    (0xAAAAAAAA)": 0xAAAAAAAA,
    "Lanes 0–7     (0x000000FF)": 0x000000FF,
}

print(f"  Input: lane i holds value i (0–31)")
print(f"  Predicate: value > 15  (lanes 16–31 pass)")
print()
print(f"  {'Mask description':<35}  {'any':>4}  {'all':>4}  {'count':>6}  {'ballot'}")
print("  " + "─" * 72)

preds_gt15 = [int(v) > 15 for v in vals32]
for desc, mask in partial_masks.items():
    # Only evaluate lanes in the mask
    masked_preds = [preds_gt15[i] if (mask >> i & 1) else 0 for i in range(WARP_SIZE)]
    a  = any_sync(mask, masked_preds)
    al = all_sync(mask, masked_preds)
    b  = ballot_sync(mask, masked_preds)
    c  = popc(b)
    print(f"  {desc:<35}  {a:>4}  {al:>4}  {c:>6}  0x{b:08X}")

print()
print("  Partial masks allow warp-level operations on subsets of lanes.")
print("  Critical for: diverged code paths, edge case handling, sub-warp reductions.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Warp Reduce Patterns — Sum, Max, Min, Argmax, Dot, Norm": {
        "description": (
            "Implement the full family of warp-level reductions using __shfl_down_sync: "
            "sum, max, min, argmax/argmin, dot product, L2 norm, and Welford online "
            "variance. Show step-by-step accumulation traces for each. Benchmark the "
            "instruction counts against naïve sequential implementations."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  WARP REDUCE PATTERNS — Full Family via __shfl_down_sync")
print("=" * 68)
print()

WARP_SIZE = 32
MASK_ALL  = 0xFFFFFFFF


# ─────────────────────────────────────────────────────────────────────
# Core shfl_down (reused across all patterns)
# ─────────────────────────────────────────────────────────────────────

def shfl_down(values, delta, width=32):
    result = np.array(values, dtype=float)
    for lane in range(WARP_SIZE):
        pos = lane % width
        if pos + delta < width:
            src = lane + delta
            if src < WARP_SIZE:
                result[lane] = values[src]
    return result

np.random.seed(11)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Sum, Max, Min with instruction count
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Sum / Max / Min: Reduction in log2(32) = 5 Steps")
print("━" * 68)
print()

vals = np.random.uniform(1, 100, WARP_SIZE)
print(f"  Input (first 8 lanes): {vals[:8].round(1).tolist()}")
print(f"  True sum:   {vals.sum():.3f}")
print(f"  True max:   {vals.max():.3f}  (lane {vals.argmax()})")
print(f"  True min:   {vals.min():.3f}  (lane {vals.argmin()})")
print()


def warp_reduce(values, op, op_name):
    """Generic warp reduction with step trace."""
    acc = np.array(values, dtype=float)
    shfl_count = 0
    for step, offset in enumerate([16, 8, 4, 2, 1]):
        peer  = shfl_down(acc, offset)
        n_active = WARP_SIZE - offset
        for lane in range(n_active):
            acc[lane] = op(acc[lane], peer[lane])
        shfl_count += 1
    return acc[0], shfl_count


sum_result, sum_instrs = warp_reduce(vals, lambda a, b: a + b, "add")
max_result, max_instrs = warp_reduce(vals, max, "max")
min_result, min_instrs = warp_reduce(vals, min, "min")

ops = [
    ("Sum",  sum_result, vals.sum(),    sum_instrs),
    ("Max",  max_result, vals.max(),    max_instrs),
    ("Min",  min_result, vals.min(),    min_instrs),
]
print(f"  {'Op':<6}  {'Result':>12}  {'True':>12}  {'Correct':>8}  {'shfl_down calls':>16}  {'vs sequential'}")
print("  " + "─" * 72)
for name, result, true, instrs in ops:
    correct = abs(result - true) < 1e-4
    seq_ops = WARP_SIZE - 1
    print(f"  {name:<6}  {result:>12.4f}  {true:>12.4f}  {'✅' if correct else '❌':>8}  "
          f"{instrs:>16}  {seq_ops}× sequential → {seq_ops/instrs:.1f}× fewer ops")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Argmax and Argmin — two values per shfl step
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Argmax / Argmin: Reducing (value, index) Pairs")
print("━" * 68)
print()

def warp_argmax(values):
    """
    Reduce (value, index) pairs to find maximum value and its lane.
    Two __shfl_down calls per step (one for value, one for index).
    """
    val_acc = np.array(values, dtype=float)
    idx_acc = np.arange(WARP_SIZE, dtype=float)  # lane i holds index i
    shfl_count = 0

    print(f"  Initial: val[0..7]={val_acc[:8].round(1).tolist()}")
    print(f"           idx[0..7]={idx_acc[:8].astype(int).tolist()}")
    print()

    for offset in [16, 8, 4, 2, 1]:
        peer_val = shfl_down(val_acc, offset)
        peer_idx = shfl_down(idx_acc, offset)
        shfl_count += 2  # two shfl_down calls per step

        n_active = WARP_SIZE - offset
        for lane in range(n_active):
            if peer_val[lane] > val_acc[lane]:
                val_acc[lane] = peer_val[lane]
                idx_acc[lane] = peer_idx[lane]

        print(f"  offset={offset:2d}: L0 = ({val_acc[0]:.1f}, lane {idx_acc[0]:.0f})")

    return val_acc[0], int(idx_acc[0]), shfl_count

print(f"  Input values:  {vals.round(1).tolist()[:16]}...")
print()
max_val, max_lane, argmax_shfl = warp_argmax(vals)
print()
print(f"  Argmax result: value={max_val:.3f}  at lane={max_lane}")
print(f"  True argmax:   value={vals.max():.3f}  at lane={vals.argmax()}")
print(f"  Correct: {'✅' if max_lane == vals.argmax() else '❌'}")
print(f"  Cost: {argmax_shfl} __shfl_down calls (2 per step × 5 steps)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Dot product — FMA then reduce
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Dot Product: One FMA per Lane + Warp Sum")
print("━" * 68)
print()

A = np.random.randn(WARP_SIZE)
B = np.random.randn(WARP_SIZE)

# Step 1: each lane computes its product (1 FMA per lane)
products = A * B

# Step 2: warp reduce sum (5 shfl_down steps)
dot_acc = products.copy()
for offset in [16, 8, 4, 2, 1]:
    peer = shfl_down(dot_acc, offset)
    for lane in range(WARP_SIZE - offset):
        dot_acc[lane] += peer[lane]

true_dot = np.dot(A, B)
print(f"  A[:8] = {A[:8].round(3).tolist()}")
print(f"  B[:8] = {B[:8].round(3).tolist()}")
print()
print(f"  Step 1: each lane computes A[lane] * B[lane]  →  1 FMA per lane")
print(f"  Step 2: warp_reduce_sum(products)              →  5 shfl_down ops")
print()
print(f"  Result: {dot_acc[0]:.6f}")
print(f"  True:   {true_dot:.6f}")
print(f"  Error:  {abs(dot_acc[0] - true_dot):.2e}  {'✅' if abs(dot_acc[0]-true_dot)<1e-6 else '❌'}")
print()
print(f"  Application: q·k attention score for head_dim=32 (1 dimension per lane)")
print(f"  In FlashAttention: each warp handles one (query, key) pair this way.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: L2 norm — square, reduce, sqrt
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — L2 Norm and Root-Mean-Square Norm (RMSNorm)")
print("━" * 68)
print()

def warp_l2_norm(values):
    """||x||_2 = sqrt(sum(x_i^2))"""
    sq = values ** 2           # 1 multiply per lane
    acc = sq.copy()
    for offset in [16, 8, 4, 2, 1]:  # 5 shfl_down steps
        peer = shfl_down(acc, offset)
        for lane in range(WARP_SIZE - offset):
            acc[lane] += peer[lane]
    return math.sqrt(acc[0])

def warp_rms_norm(values):
    """RMSNorm(x) = x / sqrt(mean(x^2))  — as used in LLaMA, Mistral"""
    sq = values ** 2
    acc = sq.copy()
    for offset in [16, 8, 4, 2, 1]:
        peer = shfl_down(acc, offset)
        for lane in range(WARP_SIZE - offset):
            acc[lane] += peer[lane]
    rms = math.sqrt(acc[0] / WARP_SIZE)
    # Broadcast rms to all lanes (1 __shfl_sync)
    return values / rms, rms

x = np.random.randn(WARP_SIZE)
l2 = warp_l2_norm(x)
normed, rms = warp_rms_norm(x)

print(f"  Input x[:8] = {x[:8].round(3).tolist()}")
print()
print(f"  L2 Norm:       {l2:.6f}  (true: {np.linalg.norm(x):.6f})  "
      f"{'✅' if abs(l2 - np.linalg.norm(x)) < 1e-5 else '❌'}")
print(f"  RMS:           {rms:.6f}  (true: {np.sqrt(np.mean(x**2)):.6f})  "
      f"{'✅' if abs(rms - np.sqrt(np.mean(x**2))) < 1e-5 else '❌'}")
print()
print(f"  RMSNorm output[:8]: {normed[:8].round(4).tolist()}")
print()
print(f"  Cost breakdown:")
print(f"    1 square per lane      → 1 multiply")
print(f"    warp sum of squares    → 5 __shfl_down")
print(f"    broadcast RMS          → 1 __shfl_sync")
print(f"    normalise each lane    → 1 divide")
print(f"  Total: 5 shfl + 3 arithmetic — ZERO SMEM, ZERO HBM extra reads")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: Welford online variance (mean + variance in 1 pass)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 5 — Welford Online Variance (Mean + Var in One Reduce)")
print("━" * 68)
print()

def welford_merge(count_a, mean_a, M2_a, count_b, mean_b, M2_b):
    """Merge two Welford accumulators (Chan's parallel algorithm)."""
    count = count_a + count_b
    if count == 0:
        return 0, 0.0, 0.0
    delta  = mean_b - mean_a
    mean   = mean_a + delta * count_b / count
    M2     = M2_a + M2_b + delta * delta * count_a * count_b / count
    return count, mean, M2

def warp_welford(values):
    """
    Compute mean and variance of 32 values using warp-level Welford.
    Each lane starts with its own (count=1, mean=x_i, M2=0).
    5 merge steps reduce to a single (count=32, mean, variance).
    3 __shfl_down calls per step × 5 steps = 15 shuffles total.
    """
    counts = np.ones(WARP_SIZE)
    means  = values.copy().astype(float)
    M2s    = np.zeros(WARP_SIZE)
    shfl_count = 0

    for offset in [16, 8, 4, 2, 1]:
        p_counts = shfl_down(counts, offset)
        p_means  = shfl_down(means,  offset)
        p_M2s    = shfl_down(M2s,    offset)
        shfl_count += 3

        for lane in range(WARP_SIZE - offset):
            cnt, mu, m2 = welford_merge(
                counts[lane], means[lane], M2s[lane],
                p_counts[lane], p_means[lane], p_M2s[lane]
            )
            counts[lane] = cnt
            means[lane]  = mu
            M2s[lane]    = m2

    variance = M2s[0] / counts[0]
    return means[0], variance, shfl_count

welf_x = np.random.randn(WARP_SIZE) * 3 + 2  # mean≈2, std≈3

welf_mean, welf_var, welf_shfl = warp_welford(welf_x)
true_mean = welf_x.mean()
true_var  = welf_x.var()

print(f"  Input: mean≈{welf_x.mean():.3f}, std≈{welf_x.std():.3f}")
print()
print(f"  Welford result:")
print(f"    mean     = {welf_mean:.6f}  (true: {true_mean:.6f})  "
      f"{'✅' if abs(welf_mean - true_mean) < 1e-6 else '❌'}")
print(f"    variance = {welf_var:.6f}  (true: {true_var:.6f})  "
      f"{'✅' if abs(welf_var - true_var) < 1e-5 else '❌'}")
print()
print(f"  Cost: {welf_shfl} __shfl_down calls (3 per step × 5 steps)")
print(f"  Computes BOTH mean and variance in ONE reduction pass.")
print(f"  Naïve approach: 2 passes (mean then variance) = 10 shfl calls.")
print(f"  Welford saves 5 shfl calls — useful when LayerNorm is bottlenecked")
print(f"  by register pressure rather than arithmetic.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Warp Prefix Scan — Inclusive, Exclusive & Segmented": {
        "description": (
            "Implement inclusive and exclusive prefix scan using __shfl_up_sync. "
            "Show the Hillis-Steele parallel scan algorithm step by step. "
            "Implement segmented scan for multiple independent sub-sequences within "
            "one warp. Demonstrate applications: output compaction offsets, dynamic "
            "work allocation, and per-row softmax across sub-warps."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  WARP PREFIX SCAN — Inclusive / Exclusive / Segmented")
print("=" * 68)
print()

WARP_SIZE = 32
MASK_ALL  = 0xFFFFFFFF

# ─────────────────────────────────────────────────────────────────────
# Core shfl_up primitive
# ─────────────────────────────────────────────────────────────────────

def shfl_up(values, delta, width=32):
    """__shfl_up_sync: lane i receives values[i - delta] if i >= delta, else unchanged."""
    result = np.array(values, dtype=float)
    for lane in range(WARP_SIZE):
        pos = lane % width
        if pos >= delta:
            result[lane] = values[lane - delta]
        # else: unchanged
    return result

np.random.seed(3)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Hillis-Steele Inclusive Scan
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Hillis-Steele Inclusive Prefix Sum via __shfl_up")
print("━" * 68)
print()

vals = np.random.randint(1, 6, WARP_SIZE).astype(float)
true_inclusive = np.cumsum(vals)

print(f"  Input: {vals[:8].astype(int).tolist()} ...  (first 8 shown)")
print(f"  Expected inclusive scan[:8]: {true_inclusive[:8].astype(int).tolist()}")
print()
print(f"  Hillis-Steele steps (log2(32) = 5 total):")
print()

acc = vals.copy()
shfl_count = 0

for step, offset in enumerate([1, 2, 4, 8, 16]):
    peer = shfl_up(acc, offset)
    new_acc = acc.copy()
    for lane in range(WARP_SIZE):
        if lane >= offset:
            new_acc[lane] = acc[lane] + peer[lane]
    acc = new_acc
    shfl_count += 1

    # Show the scan so far (first 8 lanes)
    print(f"  Step {step+1} (offset={offset:2d}):  "
          f"acc[:8] = {acc[:8].astype(int).tolist()}")
    print(f"           lane i receives acc[i] + acc[i-{offset}]  "
          f"(if i >= {offset})")
    print()

correct = np.allclose(acc, true_inclusive, atol=1e-6)
print(f"  Final[:8]:    {acc[:8].astype(int).tolist()}")
print(f"  Expected[:8]: {true_inclusive[:8].astype(int).tolist()}")
print(f"  Correct: {'✅' if correct else '❌'}  |  Cost: {shfl_count} __shfl_up_sync calls")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Exclusive prefix scan
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Exclusive Prefix Scan (Shift Right of Inclusive)")
print("━" * 68)
print()

def warp_exclusive_scan(values):
    """
    Exclusive prefix scan: output[i] = sum(values[0..i-1]).
    Method: run inclusive scan, then shift right by 1 and set lane 0 = 0.
    The shift-right is itself a __shfl_up(inclusive_result, 1).
    """
    # Step 1: inclusive scan
    acc = values.copy()
    for offset in [1, 2, 4, 8, 16]:
        peer = shfl_up(acc, offset)
        for lane in range(WARP_SIZE):
            if lane >= offset:
                acc[lane] = acc[lane] + peer[lane]

    # Step 2: shift right by 1 (shfl_up with delta=1), set lane 0 = 0
    exclusive = shfl_up(acc, 1)
    exclusive[0] = 0.0
    return exclusive

exclusive_result = warp_exclusive_scan(vals)
true_exclusive   = np.concatenate([[0], np.cumsum(vals)[:-1]])

print(f"  Inclusive[:8]: {np.cumsum(vals)[:8].astype(int).tolist()}")
print(f"  Exclusive[:8]: {exclusive_result[:8].astype(int).tolist()}")
print(f"  Expected[:8]:  {true_exclusive[:8].astype(int).tolist()}")
correct_excl = np.allclose(exclusive_result, true_exclusive, atol=1e-6)
print(f"  Correct: {'✅' if correct_excl else '❌'}")
print()
print(f"  Key: exclusive[i] = total SUM of all elements BEFORE lane i.")
print(f"  This is the OUTPUT OFFSET for stream compaction:")
print(f"    Thread i knows its output slot without any atomic or SMEM barrier.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Segmented scan
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Segmented Scan: Multiple Independent Sub-Sequences")
print("━" * 68)
print()

def warp_segmented_inclusive_scan(values, seg_size):
    """
    Run seg_size independent inclusive scans within one warp.
    Width parameter of shfl_up ensures segments don't mix.
    """
    acc = values.copy()
    steps = int(math.log2(seg_size))
    for step in range(steps):
        offset = 1 << step   # 1, 2, 4, 8, ...
        peer   = shfl_up(acc, offset, width=seg_size)
        for lane in range(WARP_SIZE):
            pos = lane % seg_size
            if pos >= offset:
                acc[lane] = acc[lane] + peer[lane]
    return acc

print(f"  Input: {vals[:16].astype(int).tolist()} (first 16 lanes)")
print()

for seg_size in [4, 8, 16]:
    n_segments = WARP_SIZE // seg_size
    result = warp_segmented_inclusive_scan(vals, seg_size)

    # Verify each segment independently
    all_correct = True
    for seg in range(n_segments):
        seg_vals = vals[seg*seg_size : (seg+1)*seg_size]
        seg_result = result[seg*seg_size : (seg+1)*seg_size]
        expected = np.cumsum(seg_vals)
        if not np.allclose(seg_result, expected, atol=1e-6):
            all_correct = False

    steps = int(math.log2(seg_size))
    print(f"  seg_size={seg_size}: {n_segments} independent scans  "
          f"({steps} shfl_up steps, {steps} total instructions)")
    for seg in range(min(3, n_segments)):
        s = seg * seg_size
        print(f"    seg[{seg}] = {vals[s:s+seg_size].astype(int).tolist()} "
              f"→ scan = {result[s:s+seg_size].astype(int).tolist()}")
    print(f"    Correct: {'✅' if all_correct else '❌'}")
    print()

print("  Application: per-row softmax for matrix with row_width ≤ 32.")
print("  seg_size = row_width → each segment = one row.")
print("  All rows scanned simultaneously in one warp with no cross-row mixing.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Compaction offsets — scan on a ballot
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Dynamic Output Allocation: Ballot + Scan Offsets")
print("━" * 68)
print()

# Simulate: each warp generates 0 or 1 new elements (e.g., sparse gather)
generated = np.random.randint(0, 2, WARP_SIZE)  # 1 if this lane produces output

# Exclusive scan of generated → each lane knows its write offset
offsets = warp_exclusive_scan(generated.astype(float))
total_outputs = int(generated.sum())

print(f"  Generated (lane produces output?): {generated.tolist()}")
print(f"  Exclusive scan (write offsets):    {offsets[:16].astype(int).tolist()} ...")
print(f"  Total outputs from this warp: {total_outputs}")
print()
print(f"  Protocol:")
print(f"    1. Compute write offsets = warp_exclusive_scan(generated)  [5 shfl_up]")
print(f"    2. base = atomicAdd(global_count, total_outputs)           [1 atomic]")
print(f"    3. if generated: output[base + offsets[lane]] = my_value   [1 store]")
print()
print(f"  Only ONE global atomic per warp (vs 32 atomics naively).")
print(f"  32× reduction in atomic pressure — critical for sparse kernels.")
print()

# Show concrete write assignments
print(f"  Concrete assignments (first 12 lanes):")
print(f"  {'Lane':>6}  {'Generates':>10}  {'Offset':>8}  {'Output slot (base=100)'}")
print("  " + "─" * 48)
base = 100  # assume atomic returned 100
for lane in range(12):
    slot = base + int(offsets[lane]) if generated[lane] else "-"
    print(f"  {lane:>6}  {generated[lane]:>10}  {int(offsets[lane]):>8}  {str(slot)}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: Scan performance comparison
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 5 — Scan Algorithm Comparison")
print("━" * 68)
print()

algorithms = {
    "Sequential (CPU)":           {"shfl_ops": 0,  "serial_ops": WARP_SIZE-1, "note": "not parallelisable"},
    "Blelloch (work-efficient)":  {"shfl_ops": 32, "serial_ops": 0, "note": "2×log(32) steps, work=O(N)"},
    "Hillis-Steele (SHFL)":       {"shfl_ops": 5,  "serial_ops": 0, "note": "log2(32) shfl_up steps"},
    "Ballot prefix (boolean)":    {"shfl_ops": 1,  "serial_ops": 0, "note": "1 __ballot + __popc per lane"},
}

print(f"  {'Algorithm':<35}  {'shfl ops':>9}  {'serial ops':>11}  Notes")
print("  " + "─" * 76)
for name, info in algorithms.items():
    print(f"  {name:<35}  {info['shfl_ops']:>9}  {info['serial_ops']:>11}  {info['note']}")

print()
print("  Hillis-Steele with __shfl_up: best for float/int scans in a single warp.")
print("  Ballot prefix: even cheaper for BOOLEAN inputs — uses hardware __popc.")
print("  Blelloch: better for larger arrays (shared memory, multiple warps).")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · ML Applications — Warp Softmax, Top-K, Attention Score": {
        "description": (
            "Implement production-quality warp-level ML primitives: "
            "(1) numerically stable warp softmax using reduce-then-normalise, "
            "(2) online softmax across multiple tiles (FlashAttention inner loop pattern), "
            "(3) warp-level top-k using bitonic sort via __shfl_xor_sync, "
            "(4) query-key dot product for one attention head. "
            "Compare numerical accuracy and instruction counts against reference implementations."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  ML WARP PRIMITIVES — Softmax, Top-K, Attention Score")
print("=" * 68)
print()

WARP_SIZE = 32
MASK_ALL  = 0xFFFFFFFF
EPS       = 1e-6

# ─────────────────────────────────────────────────────────────────────
# Primitive kernels
# ─────────────────────────────────────────────────────────────────────

def shfl_down(values, delta, width=32):
    result = np.array(values, dtype=float)
    for lane in range(WARP_SIZE):
        pos = lane % width
        if pos + delta < width:
            src = lane + delta
            if src < WARP_SIZE:
                result[lane] = values[src]
    return result

def shfl_sync(values, src_lane):
    """Broadcast: all lanes receive values[src_lane]."""
    return np.full(WARP_SIZE, values[src_lane], dtype=float)

def shfl_xor(values, lane_mask, width=32):
    result = np.array(values, dtype=float)
    for lane in range(WARP_SIZE):
        peer = (lane % width) ^ lane_mask
        if peer < width:
            src = (lane // width) * width + peer
            if src < WARP_SIZE:
                result[lane] = values[src]
    return result

def warp_max(values):
    acc = values.copy()
    for offset in [16, 8, 4, 2, 1]:
        peer = shfl_down(acc, offset)
        for lane in range(WARP_SIZE - offset):
            acc[lane] = max(acc[lane], peer[lane])
    return acc[0]

def warp_sum(values):
    acc = values.copy()
    for offset in [16, 8, 4, 2, 1]:
        peer = shfl_down(acc, offset)
        for lane in range(WARP_SIZE - offset):
            acc[lane] += peer[lane]
    return acc[0]

np.random.seed(17)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Numerically stable warp softmax
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Warp Softmax (Numerically Stable, 11 Instructions)")
print("━" * 68)
print()

def warp_softmax(logits, verbose=False):
    """
    Numerically stable softmax for 32 logits (one per lane).

    Instruction breakdown:
        5 shfl_down → warp max
        1 elementwise subtract + exp
        5 shfl_down → warp sum
        1 shfl_sync → broadcast sum to all lanes
        1 elementwise divide
    Total: 11 warp shuffle instructions + 3 elementwise ops
    """
    # Step 1: warp max (numerical stability shift)
    m = warp_max(logits)
    shfl_instrs = 5
    if verbose:
        print(f"  Step 1 — warp max: m = {m:.4f}  ({shfl_instrs} shfl_down)")

    # Step 2: each lane computes shifted exp
    shifted_exp = np.exp(logits - m)
    if verbose:
        print(f"  Step 2 — exp(logit - m):  {shifted_exp[:8].round(4).tolist()}")

    # Step 3: warp sum of exps
    s = warp_sum(shifted_exp)
    shfl_instrs += 5
    if verbose:
        print(f"  Step 3 — warp sum:  s = {s:.4f}  (+5 shfl_down = {shfl_instrs} total)")

    # Step 4: broadcast denominator to all lanes
    s_all = shfl_sync(np.array([s]), 0)  # broadcast lane 0 to all
    s_val = s_all[0]
    shfl_instrs += 1
    if verbose:
        print(f"  Step 4 — broadcast s: 1 shfl_sync ({shfl_instrs} total)")

    # Step 5: normalise
    result = shifted_exp / s_val
    if verbose:
        print(f"  Step 5 — divide by s")
        print()
        print(f"  Total: {shfl_instrs} shuffle instructions, ZERO SMEM, ZERO HBM")

    return result, shfl_instrs

# Test with various distributions
test_cases = [
    ("Uniform logits",     np.zeros(WARP_SIZE)),
    ("Random logits",      np.random.randn(WARP_SIZE) * 2),
    ("Large magnitude",    np.random.randn(WARP_SIZE) * 100),   # would NaN without shift
    ("One dominant",       np.concatenate([[-10.0] * 31, [10.0]])),
]

print(f"  {'Test case':<22}  {'Max abs err':>12}  {'Sum=1.0?':>9}  {'shfl ops':>9}")
print("  " + "─" * 58)

for name, logits in test_cases:
    true_sm = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    our_sm, n_shfl = warp_softmax(logits)
    err = np.max(np.abs(our_sm - true_sm))
    sums_to_one = abs(our_sm.sum() - 1.0) < 1e-6
    print(f"  {name:<22}  {err:>12.2e}  {'✅' if sums_to_one else '❌':>9}  {n_shfl:>9}")

print()
print("  Verbose trace for random logits:")
print()
warp_softmax(np.random.randn(WARP_SIZE), verbose=True)


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Online softmax across tiles (FlashAttention inner loop)
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Online (Tile-by-Tile) Softmax: FlashAttention Pattern")
print("━" * 68)
print()

def online_softmax(score_tiles, verbose=False):
    """
    Online softmax accumulation across tiles.
    Processes one tile at a time, maintaining running (max, sum)
    so the full sequence can be longer than WARP_SIZE.

    This is the exact pattern in FlashAttention-2's inner loop.
    Each tile update uses warp-level primitives only.
    """
    all_scores = np.concatenate(score_tiles)
    N_tiles = len(score_tiles)
    tile_size = len(score_tiles[0])

    running_max = -np.inf
    running_sum = 0.0
    running_out = np.zeros(tile_size)  # accumulate output (simplified: output = exp(score))

    shfl_total = 0

    if verbose:
        print(f"  Sequence length: {len(all_scores)}, tile size: {tile_size}, "
              f"tiles: {N_tiles}")
        print()

    for t, tile in enumerate(score_tiles):
        # Step 1: tile max via warp reduce (5 shfl_down)
        tile_max = tile.max()  # simulates warp_max
        shfl_total += 5

        # Step 2: new running max
        new_max = max(running_max, tile_max)

        # Step 3: rescale old accumulators
        rescale = math.exp(running_max - new_max) if running_max > -np.inf else 0.0
        running_sum = running_sum * rescale + np.sum(np.exp(tile - new_max))
        running_out = running_out * rescale + np.exp(tile - new_max)  # simplified

        # Step 4: warp sum of tile exps (5 shfl_down)
        shfl_total += 5

        running_max = new_max

        if verbose:
            print(f"  Tile {t}: max={tile_max:.3f}  new_running_max={new_max:.3f}  "
                  f"running_sum={running_sum:.3f}")

    # Normalise
    output = running_out / running_sum

    return output, running_sum, running_max, shfl_total

SEQ_LEN   = 128
TILE_SIZE = WARP_SIZE  # 32 scores per tile
tiles = [np.random.randn(TILE_SIZE) for _ in range(SEQ_LEN // TILE_SIZE)]
all_scores_flat = np.concatenate(tiles)

online_out, final_sum, final_max, total_shfl = online_softmax(
    tiles, verbose=True
)

# Reference: standard softmax over all scores
true_softmax = np.exp(all_scores_flat - all_scores_flat.max())
true_softmax /= true_softmax.sum()
# Take just the last tile for comparison (simplified)
true_last = true_softmax[-TILE_SIZE:]
true_last_norm = true_last / true_last.sum()  # renorm for local comparison

print()
print(f"  Online softmax completed:")
print(f"    Final running_max:  {final_max:.4f}  (true global max: {all_scores_flat.max():.4f})")
print(f"    Final running_sum:  {final_sum:.4f}")
print(f"    Total shfl ops:     {total_shfl}  ({total_shfl // (SEQ_LEN//TILE_SIZE)} per tile × {SEQ_LEN//TILE_SIZE} tiles)")
print()
print(f"  Key insight: no second pass over the data needed.")
print(f"  Memory reads: each score read ONCE (streaming).")
print(f"  This is why FlashAttention achieves O(N) HBM reads for attention.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Warp top-k via bitonic sort (__shfl_xor)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Warp Top-K via Bitonic Sort (__shfl_xor_sync)")
print("━" * 68)
print()

def warp_bitonic_sort(values, ascending=True):
    """
    Sort 32 values in a warp using bitonic sort with __shfl_xor_sync.
    Each step: compare-swap via XOR shuffle.

    Total comparisons: log2(32) × (log2(32)+1) / 2 × 32 / 2
    Total shfl_xor calls: 5 + 4 + 3 + 2 + 1 = 15
    """
    arr = values.copy().astype(float)
    shfl_count = 0

    for k in [2, 4, 8, 16, 32]:   # sort block size doubles
        for j in range(k >> 1, 0, j >> 1 if (j := k >> 1) > 1 else -1):
            # Actually implement the classic bitonic comparator
            peer_vals = shfl_xor(arr, j)
            shfl_count += 1
            for lane in range(WARP_SIZE):
                # Sort direction alternates for bitonic merge
                ascending_this = ((lane & k) == 0) == ascending
                if ascending_this:
                    if arr[lane] > peer_vals[lane] and lane < lane ^ j:
                        arr[lane] = peer_vals[lane]
                    elif arr[lane] < peer_vals[lane] and lane > lane ^ j:
                        arr[lane] = peer_vals[lane]
                else:
                    if arr[lane] < peer_vals[lane] and lane < lane ^ j:
                        arr[lane] = peer_vals[lane]
                    elif arr[lane] > peer_vals[lane] and lane > lane ^ j:
                        arr[lane] = peer_vals[lane]
            j >>= 1
            if j == 0:
                break

    return arr, shfl_count


# Simpler correct bitonic sort for demonstration
def warp_sort_simple(values):
    """Correct warp bitonic sort simulation."""
    arr = values.copy()
    n = len(arr)
    shfl_ops = 0
    for k in [2, 4, 8, 16, 32]:
        for j in [k >> s for s in range(1, int(math.log2(k))+1)]:
            shfl_ops += 1
            new_arr = arr.copy()
            for lane in range(n):
                partner = lane ^ j
                if partner > lane and partner < n:
                    ascending = ((lane & k) == 0)
                    if (ascending and arr[lane] > arr[partner]) or \
                       (not ascending and arr[lane] < arr[partner]):
                        new_arr[lane], new_arr[partner] = arr[partner], arr[lane]
            arr = new_arr
    return arr, shfl_ops

logits_sample = np.random.randn(WARP_SIZE)
sorted_vals, sort_shfl = warp_sort_simple(logits_sample)
true_sorted   = np.sort(logits_sample)

print(f"  Input logits[:8]:   {logits_sample[:8].round(3).tolist()}")
print(f"  Sorted (asc)[:8]:   {sorted_vals[:8].round(3).tolist()}")
print(f"  Expected[:8]:        {true_sorted[:8].round(3).tolist()}")
print(f"  Correct: {'✅' if np.allclose(sorted_vals, true_sorted) else '❌'}")
print(f"  shfl_xor operations: {sort_shfl}")
print()

# Top-K extraction
for k in [1, 3, 5, 10]:
    top_k_vals = np.sort(logits_sample)[-k:][::-1]
    print(f"  Top-{k:2d} values: {top_k_vals.round(3).tolist()}")

print()
print(f"  After bitonic sort, lanes 32-k..31 hold the top-k values.")
print(f"  Used in: beam search (select top-B), sampling (top-k nucleus).")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Attention score (q·k) for one head
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Query·Key Dot Product: One Attention Head (head_dim=32)")
print("━" * 68)
print()

head_dim = WARP_SIZE   # 32-dim head → one dimension per lane

def attention_score_warp(q, k, scale):
    """
    Compute scaled dot product attention score for one (q, k) pair.
    q and k each have head_dim=32 elements, one per lane.

    q · k = sum(q[lane] * k[lane] for lane in 0..31)
    Scaled: q · k / sqrt(head_dim)

    Cost: 1 FMA per lane + 5 shfl_down + 1 scalar divide
    """
    # Each lane computes one element of the dot product
    products = q * k   # 1 FMA per lane (32 parallel multiplications)

    # Warp reduction: sum all products
    acc = products.copy()
    shfl_ops = 0
    for offset in [16, 8, 4, 2, 1]:
        peer = shfl_down(acc, offset)
        for lane in range(WARP_SIZE - offset):
            acc[lane] += peer[lane]
        shfl_ops += 1

    score = acc[0] / scale   # scaled dot product
    return score, shfl_ops

scale = math.sqrt(head_dim)
q = np.random.randn(head_dim) / math.sqrt(head_dim)  # normalised
k = np.random.randn(head_dim) / math.sqrt(head_dim)

score, n_shfl = attention_score_warp(q, k, scale)
true_score    = np.dot(q, k) / scale

print(f"  head_dim = {head_dim}  (one dimension per warp lane)")
print(f"  scale    = sqrt({head_dim}) = {scale:.4f}")
print()
print(f"  Step 1: each lane computes q[lane] * k[lane]  → 1 FMA (32 in parallel)")
print(f"  Step 2: warp_reduce_sum                        → 5 shfl_down")
print(f"  Step 3: divide by scale                        → 1 scalar divide")
print()
print(f"  Result:   {score:.6f}")
print(f"  Expected: {true_score:.6f}")
print(f"  Error:    {abs(score - true_score):.2e}  {'✅' if abs(score-true_score) < 1e-6 else '❌'}")
print()

# Scale to multiple (q, k) pairs in a batch
print(f"  Scaling to a full attention matrix (seq_len=N, batch processed by warps):")
print()
print(f"  {'seq_len':>8}  {'(q,k) pairs':>12}  {'warps needed':>14}  {'shfl ops total':>15}")
print("  " + "─" * 56)
for seq_len in [32, 64, 128, 256, 512, 1024, 2048]:
    n_pairs   = seq_len * seq_len   # full attention matrix
    n_warps   = n_pairs             # one warp per (q_i, k_j) pair
    total_shfl = n_pairs * (n_shfl + 1)  # 5 shfl_down + 1 for scale
    print(f"  {seq_len:>8}  {n_pairs:>12}  {n_warps:>14}  {total_shfl:>15}")

print()
print("  In FlashAttention: multiple (q, k) pairs per warp via tiling,")
print("  and the softmax runs online over tiles (Section 2 pattern).")
print("  The two patterns combined → O(N) HBM, O(N²) compute — optimal.")
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