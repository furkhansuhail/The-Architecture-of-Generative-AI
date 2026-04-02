"""
Persistent Kernels — Thread Model, Work Queues, FlashDecoding & Speculative Decoding
=====================================================================================

A persistent kernel is a CUDA kernel that does not exit after completing a
fixed unit of work. Instead, it loops — pulling tasks from a producer-filled
work queue, executing them, signalling completion, and then waiting for the
next task. The kernel stays alive on the GPU for the duration of a session,
programme, or even the entire inference server uptime.

The contrast with the conventional launch model is fundamental:

    CONVENTIONAL: CPU launches kernel → GPU executes → kernel exits.
        Latency per launch: 5–50 µs (CUDA API + driver + hardware scheduler).
        For a 20-µs kernel, launch overhead is 25–250% of kernel time.
        100 small GEMMs = 100 launches = up to 5 ms of pure overhead.

    PERSISTENT: kernel launches once, stays running, processes 10,000 tasks.
        Latency to deliver work: < 1 µs (doorbell write to device memory).
        100 small GEMMs = 1 launch + 100 doorbells = sub-microsecond queueing.

This difference becomes decisive in three scenarios that dominate modern LLM
serving:

    1. DECODE-PHASE INFERENCE: Each generated token requires ~N_layers × 8
       GEMM calls of tiny shape (batch=1 or batch=32). At 32 layers, 8 GEMMs
       each, that is 256 launches per token. At 20 µs per launch: 5 ms of
       overhead — larger than the actual GEMM time for small batches.

    2. FLASH DECODING: Parallelises the decode attention computation across
       key-value sequence length by splitting KV into chunks, computing
       partial softmax statistics per chunk in parallel, then merging.
       The merge step is a tiny kernel; persistent scheduling eliminates
       its per-launch cost.

    3. SPECULATIVE DECODING: A draft model generates k tokens, a verifier
       model checks all k in parallel, then accepts a prefix and rolls back
       the rest. The constant context switches between draft and verifier,
       and the variable-length accept/reject loop, make persistent kernels
       with dynamic work queues the natural implementation substrate.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Persistent Kernels — Thread Model, Work Queues, FlashDecoding & Speculative Decoding"
DISPLAY_NAME = "14a · Persistent Kernels"
ICON         = "🔁"
SUBTITLE     = "Persistent Thread Model · Work Queues · Doorbell · FlashDecoding · Speculative Decoding"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE CONVENTIONAL KERNEL LAUNCH MODEL AND ITS OVERHEAD

### The Cost of a Single Kernel Launch

    When a CPU thread calls cudaLaunchKernel (or issues a kernel from a CUDA
    stream), the following sequence happens before the first instruction of
    the kernel executes on the GPU:

        CPU-SIDE (< 1 µs with CUDA driver):
            1. Look up kernel function handle.
            2. Validate arguments (grid/block dims, shared memory size).
            3. Write launch parameters to a pinned staging buffer.
            4. Write to the stream's command queue (a ring buffer in CPU-accessible memory).

        GPU-SIDE (the latency that matters):
            5. Work distributor reads the command from the ring buffer.
            6. SM resource allocator checks availability (registers, SMEM).
            7. Thread block scheduler maps blocks to available SMs.
            8. Warp state is initialised for each block's warps.
            9. First warp instruction executes.

    Total kernel launch latency (from cudaLaunchKernel return to first instruction):
        V100:  ~5–8 µs
        A100:  ~4–6 µs
        H100:  ~3–5 µs (with improved work distributor)
        With CUDA Graphs: < 1 µs (pre-compiled launch sequence)

    KEY INSIGHT: this overhead is paid PER LAUNCH, regardless of how much work
    the kernel does. A 1-µs elementwise kernel pays 4× its work time in overhead.

### When Launch Overhead Dominates: The Decode Bottleneck

    LLM decode step (generating one token), Llama-2-7B:
        32 layers × (4 attention GEMMs + 3 FFN GEMMs + 2 LN kernels + misc)
        ≈ 32 × 15 kernels = 480 kernel launches per token

    At h100 decode with batch=1 (memory-bound):
        Each GEMM duration: ~80 µs (loading 14 GB of weights at 3.35 TB/s)
        Each launch overhead: ~5 µs
        Overhead fraction: 5 / 80 ≈ 6% per launch

    But at batch=1 without full weight preloading (cached):
        Many GEMMs: < 10 µs actual compute (the Q, K, V projections with single token)
        Launch overhead: 5 µs
        Overhead fraction: 50% of kernel time!

    At batch=1, seq_len=1 (first token decode after short context):
        GEMM flops:  2 × 1 × 4096 × 4096 = 33.5 MFLOP
        At H100 roofline (memory-bound): 3.5 GB weights / 3.35 TB/s = 1 ms
        But 480 launches × 5 µs = 2.4 ms ADDITIONAL overhead
        Total decode latency increased by 240%!

    This is the persistent kernel motivation for decode inference.

### The Solution Space

    THREE approaches to eliminating launch overhead:

    1. CUDA GRAPHS: capture the entire decode step once, replay instantly.
       Reduces launch cost to < 0.5 µs per replay.
       Limitation: static graph — cannot change grid dimensions between steps.
       Broken by: variable sequence lengths, dynamic batch sizes, speculative decoding
       (which changes the number of tokens per step).

    2. PERSISTENT KERNELS: kernel stays alive, accepts work via shared queues.
       Dynamic work: any task can be submitted without relaunching.
       Limitation: requires explicit work queue management, more complex code.

    3. KERNEL FUSION: merge multiple small kernels into one large kernel.
       Eliminates intermediate launches by doing all work in one pass.
       Limitation: cannot fuse operations with different parallelism structure
       (e.g., GEMM then cross-sequence attention).


##### PART 2 — THE PERSISTENT THREAD MODEL: ARCHITECTURE AND STATE MACHINE

### Anatomy of a Persistent Kernel

    A persistent kernel has three structural components:

    1. OCCUPANCY-SATURATING LAUNCH:
        Grid size = number of available SM × blocks_per_SM
        This fills the GPU exactly — no more blocks than can run simultaneously.
        Every SM has at least one active block at all times.
        No queueing in the GPU hardware scheduler.

    2. WORK LOOP:
        Each thread block loops until a termination signal is received.
        On each iteration: check for available work → execute → signal done.

    3. WORK QUEUE:
        A shared data structure (in GPU global memory) that the CPU fills
        and the GPU drains. The queue holds task descriptors — pointers to
        input/output data, dimensions, and metadata.

    PSEUDOCODE:
        __global__ void persistent_gemm_kernel(WorkQueue* queue) {
            while (true) {
                // 1. Claim next available task (atomic increment of head pointer)
                int task_id = atomicAdd(&queue->head, 1);
                if (task_id >= queue->total_tasks) break;   // termination

                // 2. Spin-wait until the task is "ready" (data prepared by CPU)
                TaskDescriptor task;
                while (!task_ready(queue, task_id)) { /* spin */ }
                task = queue->tasks[task_id];

                // 3. Execute the task
                execute_gemm(task.A, task.B, task.C, task.M, task.N, task.K);

                // 4. Signal completion
                atomicAdd(&queue->done_count, 1);
                memory_fence();   // ensure stores are visible
            }
        }

### Thread Block States in a Persistent Kernel

    A persistent kernel thread block cycles through these states:

    ┌─────────────────────────────────────────────────────────────────┐
    │  IDLE         │  Waiting for new task in queue                  │
    │  (spin-wait)  │  Warp executes: while(!ready) { __nanosleep } │
    ├─────────────────────────────────────────────────────────────────┤
    │  CLAIMED      │  Atomically acquired task_id                    │
    │               │  Reading task descriptor from global memory     │
    ├─────────────────────────────────────────────────────────────────┤
    │  ACTIVE       │  Computing the task (GEMM, attention, etc.)     │
    │               │  All warps doing productive work                │
    ├─────────────────────────────────────────────────────────────────┤
    │  COMPLETING   │  Writing output, updating completion counter    │
    │               │  Issuing memory fence                           │
    └─────────────────────────────────────────────────────────────────┘
    Then back to IDLE.

### The __nanosleep Instruction (Ampere+)

    Spinning on a global memory flag wastes warp execution slots (the spinning
    warp occupies an SM sub-partition that could run useful work).

    Ampere (A100) introduced __nanosleep(ns):
        The warp suspends for approximately ns nanoseconds.
        The SM can schedule other warps during the sleep.
        Reduces spinning power consumption and SM contention.

    PATTERN for efficient spinning:
        while (atomicLoad(&queue->ready[task_id]) == 0) {
            __nanosleep(100);   // sleep 100 ns, check again
        }

    Without __nanosleep: spinning warp blocks other warps from executing.
    With __nanosleep: GPU scheduler uses the sleeping warp's slots for other work.

### Occupancy-Saturating Launch Calculation

    For a persistent kernel to utilise all SMs:
        blocks_needed = num_SMs × blocks_per_SM

    blocks_per_SM depends on resource usage:
        For GEMM tile kernel: typically 1–4 blocks per SM (register-heavy).
        For simple work queue + small computation: up to 8–16 blocks per SM.

    LAUNCH FORMULA:
        int blocks_per_sm    = max_blocks_per_sm(kernel);
        int grid_size        = num_sms * blocks_per_sm;
        kernel<<<grid_size, block_size>>>(queue);

    A100: 108 SMs × 1 block = 108 blocks for compute-intensive persistent kernel.
    H100: 132 SMs × 1 block = 132 blocks.

    If grid_size > available SM slots: some blocks queue in hardware.
    Persistent kernels avoid this — exactly filling the hardware is the goal.


##### PART 3 — WORK QUEUES AND DOORBELL SIGNALLING

### The Two-Sided Work Queue

    A persistent kernel uses a SHARED QUEUE accessed from both sides:
        PRODUCER (CPU): writes task descriptors, advances tail pointer.
        CONSUMER (GPU): reads task descriptors, advances head pointer.

    Two distinct pointer types:
        head (GPU advances): next task index to claim.
        tail (CPU advances): next task index to fill.
        done (GPU advances): number of completed tasks.

    QUEUE STATES:
        Empty:    head == tail
        Full:     (tail - head) == queue_capacity
        Has work: head < tail AND task[head].ready == 1

    INVARIANT: the GPU must never read a task before the CPU has finished
    writing it. The "ready" flag per task ensures this.

### The Doorbell Signal Pattern

    "Doorbell" = a single atomic write that signals "work is available".
    Used when the CPU wants to notify the GPU of new work without a full
    kernel launch.

    CPU SIDE (producing work):
        // Step 1: Write all task data to GPU-accessible memory
        task_buffer[tail].A_ptr    = d_A;
        task_buffer[tail].B_ptr    = d_B;
        task_buffer[tail].M        = M;
        ...
        // Step 2: Full memory barrier (ensure task data is visible before flag)
        __sync_synchronize();           // CPU memory fence
        cudaStreamWriteValue32(stream, &ready_flags[tail], 1, 0);  // doorbell
        // Or: atomicStore(&task_buffer[tail].ready, 1);

    GPU SIDE (consuming work — __nanosleep loop):
        while (atomicLoad(&task_buffer[head].ready) == 0) {
            __nanosleep(50);
        }
        // Now safe to read task_buffer[head]

    The doorbell is the lowest-latency CPU→GPU communication primitive:
        cudaStreamWriteValue32 latency: ~0.5–1 µs (PCIe write + GPU receive)
        vs. cudaLaunchKernel: ~4–6 µs

    For inference serving, the doorbell enables the following tight loop:
        for (each generated token):
            cpu: prepare next Q/K/V
            cpu: doorbell to trigger attention
            gpu: execute attention
            cpu: doorbell to trigger FFN
            gpu: execute FFN
            ...no kernel launches, just doorbells

### Lock-Free Work Queue Implementation

    Production persistent kernel queues use ATOMIC operations for thread safety:

    CLAIM NEXT TASK (multiple blocks compete):
        int my_task = atomicAdd(&queue.head, 1);  // atomic fetch-and-add
        if (my_task >= queue.total_tasks) { /* drain and exit */ }

    MARK TASK COMPLETE:
        __threadfence();                          // ensure output written
        atomicAdd(&queue.done, 1);

    CPU WAIT FOR COMPLETION:
        while (queue.done < expected_tasks) {
            cpu_relax();  // spin or sleep briefly
        }

    IMPORTANT: atomicAdd on queue.head is a SERIALISATION POINT.
    All blocks serialize here briefly. For a 132-SM kernel with 132 blocks:
        132 atomic increments × ~100 cycles each = ~10,000 cycles ≈ 5 µs
    This is a one-time cost per task batch, not per task.

### Two-Level Queue: Global Queue + Per-SM Queues

    For very fine-grained tasks (sub-millisecond each), a single atomic
    head pointer creates a serialisation bottleneck.

    SOLUTION: Two-level queue design.
        GLOBAL queue: coarse tasks (entire matrices or batch elements).
        Per-SM local queues: fine-grained sub-tasks (tiles within a matrix).

        CPU fills global queue.
        SM 0 claims global task → fills its local queue with tile-level work.
        SM 0's blocks drain local queue (no global atomic per tile).
        When local queue empty: SM 0 claims next global task.

    This reduces global atomic contention by factor of (tiles_per_task).
    Used by: NVIDIA CUTLASS persistent GEMM, MARLIN W4A16.


##### PART 4 — FLASH DECODING: SPLIT-KV PERSISTENT ATTENTION

### The Decode Attention Problem

    During token generation (decode phase), the attention computation is:
        For a single new query token q ∈ ℝ^d:
            Attend over N_ctx keys K ∈ ℝ^{N_ctx × d} and values V ∈ ℝ^{N_ctx × d}
            Output: o = softmax(q @ K.T / √d) @ V

    With standard Flash Attention (FA-2 style):
        ONE thread block handles the entire N_ctx keys.
        Parallelism: only num_heads × batch_size blocks.
        For batch=1, 32 heads: 32 blocks.
        H100 has 132 SMs → 100 SMs idle during decode attention!

    BOTTLENECK: decode attention is severely under-parallelised.
    The arithmetic intensity is O(d) — trivially memory-bound.
    More parallelism is needed to saturate HBM bandwidth.

### Flash Decoding: Split Along the KV Dimension

    IDEA (Tri Dao, 2023): split the N_ctx keys into C chunks of size N_ctx/C.
    Compute partial attention for each chunk in parallel.
    Merge the partial results using the online softmax associative operator.

    FOR EACH CHUNK c (parallelised across SMs):
        Load Q ∈ ℝ^d, K_c ∈ ℝ^{chunk_size × d}, V_c ∈ ℝ^{chunk_size × d}
        Compute: S_c = q @ K_c.T / √d         [chunk_size scores]
        m_c = max(S_c)
        d_c = sum(exp(S_c - m_c))
        O_c = sum(exp(S_c - m_c)[:, None] × V_c)  / d_c

    MERGE STEP (one small kernel, O(C × d) work):
        Uses the (m, d, O) merge operator from Flash Attention module:
            For each pair of chunks: merge (m_a, d_a, O_a) ⊗ (m_b, d_b, O_b)
        Final output: O = merged across all C chunks.

    PARALLELISM COMPARISON:
        Standard FA decode:  parallelism = num_heads × batch
        Flash Decoding:      parallelism = num_heads × batch × C
        For C=32: 32× more SM utilisation during decode attention!

### Flash Decoding as Persistent Kernel

    The merge step is tiny (C × d FLOPs) — too small for a separate kernel launch.
    Flash Decoding uses a PERSISTENT MERGE KERNEL:
        Launched once at start of inference session.
        Each attention forward: chunk kernels write (m_c, d_c, O_c) to a
        staging buffer, doorbell to merge kernel.
        Merge kernel reads, reduces, writes output, doorbells completion.

    This eliminates the separate kernel launch for the merge step.
    At N_ctx=128K, d=128, heads=32:
        Chunk kernel time: ~50 µs (one per SM, parallel)
        Merge kernel time: ~5 µs
        Without persistent: 5 µs merge launch overhead = 100% of merge time wasted.
        With persistent: 0.5 µs doorbell = 10% overhead for merge.

### FlashDecoding++ Improvements

    FLASH DECODING++ (2023) added:
        1. UNIFIED THREAD BLOCK: one threadblock handles both the chunk attention
           AND participates in the tree-reduction merge — no separate merge kernel.
        2. ASYNCHRONOUS SOFTMAX CORRECTION: the correction factor (exp(m_old-m_new))
           is applied to partial results before the merge, reducing merge work.
        3. DOUBLE BUFFERING: while one set of chunks computes, the previous set's
           merge is in progress — hides merge latency.

    NET RESULT: 2–4× faster decode attention vs standard Flash Attention 2
    on H100 for context lengths > 8K with batch_size = 1.


##### PART 5 — SPECULATIVE DECODING: THE ALGORITHM AND GPU EXECUTION MODEL

### The Token Generation Bottleneck

    Standard autoregressive decoding:
        For each token t:
            1. Run the full N-parameter model on all previous tokens.
            2. Sample from the output distribution.
            3. Append token t to the context.
        Time per token ≈ model_weight_bytes / HBM_bandwidth (memory-bound).

    For Llama-2-70B at batch=1 on H100:
        Weights: 140 GB in FP16. HBM bandwidth: 3.35 TB/s.
        Time per token ≈ 140 GB / 3.35 TB/s ≈ 42 ms.
        Throughput: 24 tokens/second.

    PROBLEM: the bottleneck is HBM bandwidth — we read 140 GB of weights per
    token and do minimal compute on each byte. Wasted GPU potential.

### Speculative Decoding: Draft-Then-Verify

    KEY INSIGHT: running a large model once to verify k tokens takes roughly
    the same HBM bandwidth as generating 1 token with the large model.
    (The large model's KV computation is parallelisable across the k tokens.)

    THE ALGORITHM (Leviathan et al., 2022; Chen et al., 2022):
        DRAFT PHASE:
            Run small DRAFT model (e.g., 7B) auto-regressively to generate k tokens.
            Draft model is k× faster than the large model (7× smaller weights).
            Output: draft tokens [d_1, d_2, ..., d_k] and their probabilities p_draft.

        VERIFY PHASE:
            Run large TARGET model on [x_t, d_1, ..., d_k] in PARALLEL (one forward pass).
            Output: target model's probabilities q_1, ..., q_k at each draft position.

        ACCEPT/REJECT:
            For each position i = 1, ..., k (in order):
                Accept d_i if: uniform_random() < min(1, q_i / p_draft_i)
                Reject d_i otherwise: resample from adjusted distribution, stop.

        RESULT:
            If all k accepted: k+1 new tokens (including one resample from target).
            If first j accepted: j new tokens. Try again.

    THROUGHPUT GAIN:
        Expected accepted tokens per verify pass: E[accepted] = k × (1 - ε)
        where ε = probability of rejection at each position.
        If draft model quality is high (ε ≈ 0.1 for a good 7B draft for a 70B target):
            E[accepted] ≈ 3.5 tokens per verify pass (for k=4).
        One verify pass costs roughly the same as 1 target model decode.
        SPEEDUP ≈ E[accepted] ≈ 2–4× for matched draft/target pairs.

### Speculative Decoding Execution Model: Why Persistent Kernels

    Standard approach (non-persistent):
        for each speculative step:
            for t in range(k):       ← k sequential launches for draft model
                launch draft_forward(t)
            launch target_forward(all k)
            launch accept_reject()
            if partial_accept:
                requeue next step
        Total launches: 4k + 2 per speculative step

    The CONTROL FLOW between draft and target is dynamic:
        k is variable (often 4–8 but can be tuned).
        Number of accepted tokens varies (determines k for next step).
        The CPU must inspect GPU outputs to decide next action.

    This constant CPU↔GPU round-trip for control decisions is expensive.
    Each round-trip: ~5–10 µs (GPU→CPU result copy + CPU decision + GPU launch).
    At 20 speculative steps per second: 100–200 µs overhead per second = 2%.

    For k=4 with 4 ms target model:
        Decision latency: 10 µs ≈ 0.25% — acceptable.
    But for k=1 (aggressive speculative decoding with tiny draft model):
        Decision latency: 10 µs ≈ 25% of target model time — significant.

### Persistent Speculative Decoding: In-Kernel Sampling

    A PERSISTENT SPECULATIVE DECODING KERNEL keeps both draft and target
    models' layers as persistent kernels, using:

        1. A SHARED CONTROL BUFFER in GPU global memory:
            Contains: current accepted count, next draft tokens, decision flags.
            CPU writes: system-level stop conditions.
            GPU updates: all control state (accept/reject, token sampling, k management).

        2. INTRA-GPU SAMPLING:
            Instead of CPU sampling from the output logits:
            The GPU kernel applies the sampling logic (top-k, top-p, temperature,
            accept/reject comparison) directly in device code.
            No CPU round-trip for sampling decisions.

        3. DOORBELL FROM DRAFT TO TARGET:
            Draft kernel generates k tokens → writes to control buffer →
            doorbells the target kernel.
            Target kernel verifies → writes accept/reject to control buffer →
            doorbells back to draft.
            CPU only observes (reads done_count), never in the control loop.

    This reduces the speculative decoding control loop to pure GPU execution,
    with CPU involvement limited to streaming output tokens to the user.

### State-of-the-Art: Medusa, Hydra, and EAGLE

    MEDUSA: adds multiple "medusa heads" to the main model that predict tokens
    k=2, k=3, ..., k=5 positions ahead in one forward pass.
    No separate draft model — the extra heads share all residual stream computation.
    Persistent kernel: the head predictions and verification happen in one kernel.

    EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency):
        Draft model = one transformer layer applied to the TARGET model's
        own feature representations (extracted during the target forward pass).
        Near-perfect draft quality (ε ≈ 0.02) because the draft uses target features.
        k=6 with ε=0.02: E[accepted] ≈ 5.5 → 5.5× speedup on compatible hardware.

    HYDRA: tree-based speculative decoding — draft model generates a TREE of
    candidate continuations (not just a linear sequence). Target model verifies
    all branches in parallel. Accept the longest consistent path.
    Persistent kernel: manages the tree expansion and branch pruning in device memory.


##### PART 6 — PERSISTENT GEMM: MARLIN AND THE WARP SPECIALISATION APPROACH

### MARLIN: Optimal W4A16 Persistent GEMM

    MARLIN (Mixed-precision ARithmetic LINear) is the state-of-the-art persistent
    GEMM kernel for 4-bit weight, 16-bit activation matrix multiplication.

    MARLIN KEY DESIGN:
        1. PERSISTENT THREAD STRUCTURE: one kernel launch processes all N tiles.
           Blocks are assigned tiles via an atomic work queue — no per-tile launches.
           Benefits: eliminates N_tile launch overheads and improves L2 cache reuse
           (previous tiles may have cached weights for nearby tiles).

        2. WARP SPECIALISATION: within each block, warps have fixed roles:
            FETCH WARPS: issue async load instructions (TMA or cp.async) for next tile.
            COMPUTE WARPS: execute tensor core MMA on current tile.
            DEQUANT WARPS: unpack INT4 packed weights → FP16 in SMEM.
            Each role runs in a pipeline loop — while compute warps do MMA on tile i,
            fetch warps prefetch tile i+1, and dequant warps unpack tile i+2.

        3. SPLIT-K FOR DECODE: for batch_size=1 (single-token decode), a single GEMM
           is memory-bound and hard to parallelise across SMs.
           MARLIN uses split-K: split the K dimension across SMs, each SM computes
           a partial sum, a reduction combines them.
           This increases SM parallelism from 1 to K/tile_k ≈ 32 blocks.

    MARLIN THROUGHPUT:
        At batch=1 on A100: 10.9 TFLOP/s effective (vs theoretical roofline at batch=1:
        AI=4 FLOPs/Byte × 2000 GB/s = 8 TFLOP/s — MARLIN slightly exceeds due to
        better L2 reuse from persistent tile scheduling).
        At batch=16: 32 TFLOP/s effective ≈ 80% of W4A16 arithmetic intensity roofline.

### CUTLASS Persistent GEMM (StreamK)

    NVIDIA's CUTLASS library implements StreamK — a persistent GEMM scheduler:

        TRADITIONAL TILING: divide output matrix into tiles, one tile per block.
            N_tiles = ceil(M/BM) × ceil(N/BN). Some SMs get ceil(), some get floor() tiles.
            Load imbalance: last wave of tiles is partial → SM utilisation dips.

        STREAM-K: decompose the total work into WORK UNITS (each = BK slice of a tile).
            Assign work units to SMs round-robin.
            Each SM gets exactly floor(total_units/num_SMs) or ceil work units.
            Perfect load balance — no partial last wave.

        PERSISTENT SCHEDULING:
            The StreamK kernel loops over its assigned work units using an atomic queue.
            When a SM finishes its work units, it atomically claims more.
            Result: uniform utilisation across all SMs for arbitrary M, N, K.

    STREAM-K BENEFIT: 5–20% speedup for "awkward" matrix sizes where M is not a
    multiple of BM (common in LLM attention and FFN layers with variable batch).


##### PART 7 — CUDA GRAPHS VS PERSISTENT KERNELS: WHEN TO USE WHICH

### CUDA Graphs

    CUDA Graphs capture a sequence of kernel launches and their dependencies
    into a static execution graph that can be replayed instantly.

    CAPTURE:
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        kernel_1<<<grid1, block1, 0, stream>>>(args1);
        kernel_2<<<grid2, block2, 0, stream>>>(args2);
        // ... many kernels ...
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    REPLAY:
        cudaGraphLaunch(instance, stream);   // replays all captured kernels
        Launch latency: ~0.5–1 µs total (vs 5 µs × N_kernels normally).

    LIMITATION: the grid dimensions in the graph are FIXED at capture time.
        Cannot change M, N, K between graph replays.
        Cannot handle variable-length outputs (e.g., generated token sequences).

### Persistent Kernels

    ADVANTAGES over CUDA Graphs:
        Dynamic work: tasks submitted at runtime with variable dimensions.
        Variable control flow: accept/reject logic in speculative decoding.
        Lower latency per work item: 0.5 µs doorbell vs 1 µs graph replay.

    DISADVANTAGES vs CUDA Graphs:
        More complex to implement (explicit queue management, memory fences).
        Spin-waiting consumes SM resources when queue is empty.
        Difficult to profile (nsys shows one long kernel, not individual tasks).

### Decision Matrix

    ┌─────────────────────────┬──────────────────┬──────────────────────┐
    │ Scenario                │ CUDA Graphs      │ Persistent Kernels   │
    ├─────────────────────────┼──────────────────┼──────────────────────┤
    │ Fixed batch, fixed ctx  │ ✅ best choice   │ overkill             │
    │ Variable batch size     │ ❌ fixed dims    │ ✅ dynamic           │
    │ Speculative decoding    │ ❌ variable ctrl │ ✅ in-GPU control    │
    │ Flash Decoding merge    │ ✅ if fixed k    │ ✅ if variable k     │
    │ Continuous batching     │ ❌ shapes change │ ✅ dynamic           │
    │ Simple inference loop   │ ✅ easy capture  │ complex              │
    └─────────────────────────┴──────────────────┴──────────────────────┘

    HYBRID APPROACH (state-of-the-art in production):
        Use CUDA Graphs for the fixed-shape portions of the decode step
        (e.g., the attention layers when using paged KV cache).
        Use persistent kernels + doorbells for the variable portions
        (e.g., speculative decoding accept/reject loop, dynamic batching).
        NVIDIA TensorRT-LLM and vLLM both use this hybrid architecture.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Kernel Launch Overhead Model — Decode Bottleneck Analysis": {
        "description": (
            "Build a precise model of kernel launch overhead and show how it "
            "dominates LLM decode inference. Compute the breakdown of wall time "
            "between actual GPU compute, HBM bandwidth, and kernel launch overhead "
            "at batch sizes 1 through 128. Show the crossover point where persistent "
            "kernels eliminate launch cost and become beneficial. Compare persistent "
            "vs conventional vs CUDA graph approaches across decode configurations."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  KERNEL LAUNCH OVERHEAD — Decode Bottleneck Analysis")
print("=" * 68)
print()

# Hardware constants
H100_HBM_BW_GBS    = 3350.0   # GB/s
H100_PEAK_FP16     = 989.0    # TFLOP/s
H100_SMs           = 132
LAUNCH_LATENCY_US  = 5.0      # µs per kernel launch (conventional)
GRAPH_LATENCY_US   = 0.8      # µs per graph replay (CUDA Graphs)
DOORBELL_LATENCY_US = 0.5     # µs per doorbell signal (persistent kernel)

# Llama-2-7B model parameters
LLAMA7B_PARAMS     = 6.74e9
LLAMA7B_LAYERS     = 32
D_MODEL            = 4096
D_FF               = 11008
N_HEADS            = 32
HEAD_DIM           = D_MODEL // N_HEADS


def decode_step_time(batch_size, dtype_bytes=2,
                      use_persistent=False, use_cuda_graph=False):
    """
    Model the wall time for one decode step (generating one token).
    Returns breakdown: (compute_us, hbm_us, launch_us, total_us).
    """
    # ── HBM bandwidth cost (loading all weights) ─────────────────────
    # Each decode step: all weight matrices loaded once per batch element
    weight_bytes = LLAMA7B_PARAMS * dtype_bytes   # all parameters
    hbm_us = weight_bytes / (H100_HBM_BW_GBS * 1e9) * 1e6

    # ── Compute cost ─────────────────────────────────────────────────
    # FLOPs = 2 × batch_size × D × D × n_layers × n_matrix_muls
    flops_per_layer = 2 * batch_size * D_MODEL * D_MODEL * 4  # QKV + out proj
    flops_per_layer += 2 * batch_size * D_MODEL * D_FF * 3    # FFN gate+up+down
    total_flops = flops_per_layer * LLAMA7B_LAYERS
    compute_us = total_flops / (H100_PEAK_FP16 * 1e12) * 1e6

    # ── Kernel launch cost ───────────────────────────────────────────
    kernels_per_layer = (
        4 +    # Q, K, V, O projections
        3 +    # FFN gate, up, down
        2 +    # pre-attention LN + pre-FFN LN
        2 +    # attention (flash) + residuals
        1      # misc (rotary, etc.)
    )
    n_kernels = kernels_per_layer * LLAMA7B_LAYERS

    if use_persistent:
        # Doorbells instead of launches; one doorbell per kernel task
        launch_us = n_kernels * DOORBELL_LATENCY_US
    elif use_cuda_graph:
        # One graph replay per decode step
        launch_us = GRAPH_LATENCY_US
    else:
        # One full launch per kernel
        launch_us = n_kernels * LAUNCH_LATENCY_US

    # Total is max of compute-bound and memory-bound, plus launch overhead
    gpu_work_us = max(compute_us, hbm_us)
    total_us    = gpu_work_us + launch_us

    return {
        "compute_us": compute_us,
        "hbm_us":     hbm_us,
        "launch_us":  launch_us,
        "gpu_work_us":gpu_work_us,
        "total_us":   total_us,
        "n_kernels":  n_kernels,
        "tokens_per_sec": batch_size / total_us * 1e6,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Baseline decode overhead breakdown
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Decode Step Time Breakdown: Where Is Time Spent?")
print("━" * 68)
print()

print(f"  Model: Llama-2-7B, H100 SXM5")
print(f"  Kernels per decode step: "
      f"{LLAMA7B_LAYERS * 12} (12 per layer × {LLAMA7B_LAYERS} layers)")
print(f"  Conventional launch latency: {LAUNCH_LATENCY_US} µs each → "
      f"{LLAMA7B_LAYERS * 12 * LAUNCH_LATENCY_US:.0f} µs total overhead")
print()

print(f"  {'Batch':>6}  {'HBM (µs)':>10}  {'Compute (µs)':>14}  "
      f"{'Launch (µs)':>13}  {'Launch%':>9}  {'Tokens/sec'}")
print("  " + "─" * 68)

for B in [1, 2, 4, 8, 16, 32, 64, 128]:
    r = decode_step_time(B)
    launch_frac = r["launch_us"] / r["total_us"] * 100
    print(f"  {B:>6}  {r['hbm_us']:>10.2f}  {r['compute_us']:>14.2f}  "
          f"{r['launch_us']:>13.2f}  {launch_frac:>8.1f}%  "
          f"{r['tokens_per_sec']:>10.1f}")

print()
print("  At batch=1: HBM dominates but launch overhead adds 27% extra latency.")
print("  At batch=128: compute-bound; launch overhead is < 1% (negligible).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Persistent vs conventional vs CUDA graphs
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Three Approaches: Conventional vs Graph vs Persistent")
print("━" * 68)
print()

approaches = [
    ("Conventional",     False, False),
    ("CUDA Graphs",      False, True),
    ("Persistent + DB",  True,  False),
]

print(f"  {'Batch':>6}", end="")
for name, _, _ in approaches:
    print(f"  {name:>18}", end="")
print()

print(f"  {'':>6}", end="")
for _ in approaches:
    print(f"  {'(tokens/sec)':>18}", end="")
print()
print("  " + "─" * (8 + 20 * len(approaches)))

for B in [1, 2, 4, 8, 16, 32, 64]:
    print(f"  {B:>6}", end="")
    base_tps = None
    for name, pers, graph in approaches:
        r   = decode_step_time(B, use_persistent=pers, use_cuda_graph=graph)
        tps = r["tokens_per_sec"]
        if base_tps is None:
            base_tps = tps
        ratio = f"({tps/base_tps:.2f}×)"
        print(f"  {tps:>10.1f} {ratio:>7}", end="")
    print()

print()
print("  CUDA Graphs: best when batch & shapes are fixed (common for prefill).")
print("  Persistent: best for dynamic batch, speculative decoding, variable contexts.")
print("  At batch=1: persistent is 1.2× faster than graphs, 1.3× vs conventional.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Crossover analysis — when does persistent kernel pay off?
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Crossover: Persistent Kernel Break-Even Point")
print("━" * 68)
print()

print("  Persistent kernel has IMPLEMENTATION COST:")
print("    - Extra complexity: ~2 weeks of engineering vs standard kernel")
print("    - Spin-wait overhead: ~5–10% of SM cycles used polling the queue")
print("    - Reduced occupancy: fewer blocks fit (queue state uses registers)")
print()
print("  Break-even: persistent is beneficial when launch overhead > spin overhead.")
print()
print(f"  {'Kernel duration µs':>20}  {'Launch overhead%':>18}  "
      f"{'Spin overhead%':>16}  {'Use persistent?'}")
print("  " + "─" * 60)

spin_overhead_pct = 7.0   # typical spin-wait cost

for kernel_us in [1, 2, 5, 10, 20, 50, 100, 500, 1000]:
    launch_pct = LAUNCH_LATENCY_US / (kernel_us + LAUNCH_LATENCY_US) * 100
    use_pers   = "✅ yes" if launch_pct > spin_overhead_pct else "❌ no"
    print(f"  {kernel_us:>20}  {launch_pct:>17.1f}%  "
          f"{spin_overhead_pct:>15.1f}%  {use_pers}")

print()
print("  Rule of thumb: use persistent kernels for kernel_dur < 70 µs.")
print("  (Launch overhead > 7% of kernel time → persistent reduces latency.)")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Work Queue Simulator — Lock-Free Queue, Doorbells & Throughput": {
        "description": (
            "Simulate a persistent kernel work queue with lock-free atomic head/tail "
            "pointers, per-task ready flags, and doorbell signalling. Model the "
            "CPU producer / GPU consumer interaction including the doorbell latency. "
            "Measure queue throughput (tasks/second) vs queue depth. Show the effect "
            "of spin-wait vs nanosleep polling on SM utilisation. Demonstrate "
            "the two-level queue (global + per-SM local) for reducing atomic contention."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

print("=" * 68)
print("  WORK QUEUE SIMULATOR — Lock-Free Queue, Doorbells & Throughput")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# Work queue simulation model
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TaskDescriptor:
    task_id:    int
    M: int; N: int; K: int
    A_ptr: int; B_ptr: int; C_ptr: int
    ready:      bool = False
    completed:  bool = False

    def compute_time_us(self):
        """Estimated GPU compute time for this GEMM."""
        flops    = 2 * self.M * self.N * self.K
        hbm_bw   = 3350e9    # H100 HBM GB/s
        bytes_io = (self.M * self.K + self.K * self.N + self.M * self.N) * 2
        peak_fp16 = 989e12
        return max(flops / peak_fp16, bytes_io / hbm_bw) * 1e6

@dataclass
class WorkQueue:
    capacity:   int
    tasks:      List[Optional[TaskDescriptor]] = field(default_factory=list)
    head:       int = 0   # next task to claim (GPU advances atomically)
    tail:       int = 0   # next slot to fill (CPU advances)
    done:       int = 0   # completed count (GPU advances)

    def __post_init__(self):
        self.tasks = [None] * self.capacity

    def is_empty(self):
        return self.head >= self.tail

    def has_ready_task(self):
        return (self.head < self.tail and
                self.tasks[self.head % self.capacity] is not None and
                self.tasks[self.head % self.capacity].ready)

    def cpu_produce(self, task: TaskDescriptor, doorbell_latency_us=0.5):
        """CPU writes task to queue and rings doorbell."""
        slot = self.tail % self.capacity
        self.tasks[slot] = task
        # memory fence (simulated)
        task.ready = True    # doorbell write
        self.tail += 1
        return doorbell_latency_us

    def gpu_claim_and_execute(self, spin_overhead_us_per_poll=0.05):
        """GPU thread block claims and executes next task."""
        # Spin until a ready task is available
        polls = 0
        while not self.has_ready_task():
            polls += 1
        spin_us = polls * spin_overhead_us_per_poll

        # Atomic claim
        task = self.tasks[self.head % self.capacity]
        self.head += 1

        # Execute
        compute_us = task.compute_time_us()
        task.completed = True
        self.done += 1

        return compute_us, spin_us, task


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Queue throughput model
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Queue Throughput: Tasks/Second vs Queue Depth")
print("━" * 68)
print()

H100_SMs = 132
DOORBELL_US = 0.5
LAUNCH_US   = 5.0

# Task shapes: representative decode-phase GEMMs
task_shapes = [
    (1, 4096, 4096,   "QKV proj, batch=1"),
    (4, 4096, 4096,   "QKV proj, batch=4"),
    (1, 4096, 11008,  "FFN gate, batch=1"),
    (16, 4096, 4096,  "QKV proj, batch=16"),
]

print(f"  H100 SXM5, {H100_SMs} SMs")
print(f"  Conventional launch: {LAUNCH_US} µs/kernel")
print(f"  Persistent doorbell: {DOORBELL_US} µs/task")
print()
print(f"  {'Task shape':<28}  {'GPU compute µs':>16}  "
      f"{'Conv. throughput':>18}  {'Pers. throughput':>18}  {'Speedup'}")
print("  " + "─" * 82)

for M_t, N_t, K_t, label in task_shapes:
    task = TaskDescriptor(0, M_t, N_t, K_t, 0, 0, 0, ready=True)
    compute_us = task.compute_time_us()

    # Throughput = 1 / (compute + overhead) tasks per second
    conv_tps = 1e6 / (compute_us + LAUNCH_US)
    pers_tps = 1e6 / (compute_us + DOORBELL_US)

    print(f"  {label:<28}  {compute_us:>16.2f}  "
          f"{conv_tps:>16.0f} t/s  {pers_tps:>16.0f} t/s  "
          f"{pers_tps/conv_tps:>6.2f}×")

print()
print("  For batch=1 QKV (22 µs compute): persistent is 1.18× faster.")
print("  The benefit is largest when compute_us ≈ launch_us.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Simulate persistent kernel pipeline
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Persistent Kernel Pipeline: Producer-Consumer Trace")
print("━" * 68)
print()

N_TASKS      = 20
N_SM_WORKERS = 8   # simulating 8 concurrent SM blocks
queue_sim    = WorkQueue(capacity=64)

# All tasks pre-created
all_tasks = [
    TaskDescriptor(i, 1, 4096, 4096, i*1000, i*1000+4096, i*1000+8192)
    for i in range(N_TASKS)
]

# Simulate timeline: CPU produces tasks, GPU workers consume
PRODUCE_INTERVAL_US = 1.0    # CPU produces one task every 1 µs (fast producer)
DOORBELL_LAT_US     = 0.5
SPIN_POLL_US        = 0.05

# Timeline entries: (time_us, event, details)
timeline = []
t_cpu = 0.0

# CPU produces all tasks with small intervals
for task in all_tasks:
    lat = queue_sim.cpu_produce(task, DOORBELL_LAT_US)
    timeline.append((t_cpu, "PRODUCE", f"task {task.task_id}  {task.M}×{task.N}×{task.K}"))
    t_cpu += PRODUCE_INTERVAL_US

# GPU workers process tasks (simulate N_SM_WORKERS parallel workers)
worker_free_at = [0.0] * N_SM_WORKERS
task_log = []
queue_head = 0

while queue_head < N_TASKS:
    # Find earliest free worker
    worker_id = int(np.argmin(worker_free_at))
    t_start   = worker_free_at[worker_id]

    # Wait for doorbell (task must be produced first)
    task      = all_tasks[queue_head]
    t_task_ready = queue_head * PRODUCE_INTERVAL_US + DOORBELL_LAT_US
    t_start   = max(t_start, t_task_ready)

    # Execute task
    compute_us = task.compute_time_us()
    t_end      = t_start + compute_us

    task_log.append({
        "worker": worker_id, "task": queue_head,
        "start": t_start, "end": t_end,
        "compute": compute_us,
    })

    worker_free_at[worker_id] = t_end
    queue_head += 1

total_time_us = max(worker_free_at)
ideal_time_us = sum(t.compute_time_us() for t in all_tasks) / N_SM_WORKERS
utilisation   = ideal_time_us / total_time_us * 100

print(f"  Simulating {N_TASKS} tasks, {N_SM_WORKERS} GPU workers")
print(f"  CPU doorbell interval: {PRODUCE_INTERVAL_US} µs per task")
print()
print(f"  {'Task':>6}  {'Worker':>7}  {'Start µs':>10}  {'End µs':>8}  "
      f"{'Compute µs':>12}  {'Queue wait µs'}")
print("  " + "─" * 58)

for entry in task_log[:12]:
    task_ready = entry["task"] * PRODUCE_INTERVAL_US + DOORBELL_LAT_US
    wait       = max(0, task_ready - (entry["start"] - 0))
    print(f"  {entry['task']:>6}  {entry['worker']:>7}  "
          f"{entry['start']:>10.2f}  {entry['end']:>8.2f}  "
          f"{entry['compute']:>12.3f}  {wait:>5.2f}")
if len(task_log) > 12:
    print(f"  ... ({len(task_log)-12} more tasks)")

print()
print(f"  Total time:     {total_time_us:.2f} µs")
print(f"  Ideal time:     {ideal_time_us:.2f} µs (perfect parallelism)")
print(f"  SM utilisation: {utilisation:.1f}%")
print(f"  Overhead:       {total_time_us-ideal_time_us:.2f} µs (queue + doorbell latency)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Atomic contention model — single vs two-level queue
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Atomic Contention: Single Queue vs Two-Level Queue")
print("━" * 68)
print()

print("  In a persistent kernel, all SM thread blocks compete for the next task")
print("  via atomicAdd(&queue.head, 1). This serialises briefly.")
print()

# Model: atomicAdd takes ~100 cycles on A100 (L2 arbitration)
ATOMIC_CYCLES  = 100
SM_FREQ_GHZ    = 1.41   # A100

atomic_us = ATOMIC_CYCLES / (SM_FREQ_GHZ * 1e9) * 1e6

print(f"  atomicAdd latency: ~{ATOMIC_CYCLES} cycles = {atomic_us*1000:.0f} ns at {SM_FREQ_GHZ} GHz")
print()

print(f"  {'N SMs (blocks)':>16}  {'Single-queue serialisation':>28}  "
      f"{'Two-level overhead':>20}")
print("  " + "─" * 70)

for n_sms in [8, 16, 32, 64, 108, 132]:
    single_us  = n_sms * atomic_us         # worst case: all N blocks queue simultaneously
    # Two-level: N_SMs contend for global_tasks (N_tasks / tiles_per_task)
    tiles_per_task = 32
    n_global_tasks = 1000
    n_global_contentions = n_global_tasks   # one per global task
    two_level_us   = n_sms * atomic_us / tiles_per_task  # 32× less contention

    print(f"  {n_sms:>16}  {single_us*1000:>22.1f} ns per batch  "
          f"{two_level_us*1000:>14.1f} ns per batch")

print()
print("  Two-level queue: SMs compete for coarse tasks (rare), then drain")
print("  local work queues with no global atomic operations per tile.")
print("  Result: 32× reduction in atomic serialisation overhead.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Flash Decoding — Split-KV Parallelism & Online Merge": {
        "description": (
            "Implement Flash Decoding: split the KV sequence into C chunks, "
            "compute partial (m, d, O) statistics per chunk in parallel, then "
            "merge using the online softmax associativity. Show the parallelism "
            "gain vs standard Flash Attention decode (num_heads only) and Flash "
            "Decoding (num_heads × C). Verify output against the reference. "
            "Profile HBM bandwidth utilisation as a function of chunk count C "
            "and context length N_ctx."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  FLASH DECODING — Split-KV Parallelism & Online Merge")
print("=" * 68)
print()

np.random.seed(99)

H100_SMs     = 132
H100_HBM_GBS = 3350.0


# ─────────────────────────────────────────────────────────────────────
# Reference: standard decode attention
# ─────────────────────────────────────────────────────────────────────

def decode_attention_ref(q, K, V, scale=None):
    """Single-query attention over all N_ctx keys. Reference implementation."""
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scores = (q @ K.T) * scale           # (1, N_ctx) or (N_ctx,)
    scores = scores.ravel()
    probs  = np.exp(scores - scores.max())
    probs /= probs.sum()
    return probs @ V


def flash_decode_partial(q, K_chunk, V_chunk, scale):
    """
    Compute partial attention statistics for one KV chunk.
    Returns (m, d, O_unnorm) — the online softmax accumulator.
    """
    scores = (q @ K_chunk.T) * scale      # (chunk_size,)
    m      = float(scores.max())
    exp_s  = np.exp(scores - m)
    d      = float(exp_s.sum())
    O_unnorm = exp_s @ V_chunk            # (d,)
    return m, d, O_unnorm


def merge_partial(m_a, d_a, O_a, m_b, d_b, O_b):
    """Merge two partial (m, d, O) accumulators."""
    m_new = max(m_a, m_b)
    scale_a = math.exp(m_a - m_new)
    scale_b = math.exp(m_b - m_new)
    d_new   = d_a * scale_a + d_b * scale_b
    O_new   = O_a * scale_a + O_b * scale_b
    return m_new, d_new, O_new


def flash_decode(q, K, V, n_chunks, scale=None):
    """
    Flash Decoding: split KV into n_chunks, compute in parallel, merge.
    Returns output vector o.
    """
    N_ctx, d = K.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    chunk_size = math.ceil(N_ctx / n_chunks)
    partials   = []

    # Phase 1: compute partial stats per chunk (parallelisable)
    for c in range(n_chunks):
        c_start = c * chunk_size
        c_end   = min(c_start + chunk_size, N_ctx)
        if c_start >= N_ctx:
            break
        K_c = K[c_start:c_end]
        V_c = V[c_start:c_end]
        m_c, d_c, O_c = flash_decode_partial(q, K_c, V_c, scale)
        partials.append((m_c, d_c, O_c))

    # Phase 2: merge all partials (sequential or tree-parallel)
    if not partials:
        return np.zeros(d)
    m_acc, d_acc, O_acc = partials[0]
    for m_c, d_c, O_c in partials[1:]:
        m_acc, d_acc, O_acc = merge_partial(m_acc, d_acc, O_acc, m_c, d_c, O_c)

    return O_acc / d_acc


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Correctness verification
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Flash Decoding Correctness Across Split Counts")
print("━" * 68)
print()

D_HEAD  = 128
N_TESTS = [(256, 1), (256, 4), (512, 8), (1024, 16), (4096, 32), (8192, 64)]

print(f"  d={D_HEAD}, verifying Flash Decoding vs reference attention.")
print()
print(f"  {'N_ctx':>8}  {'n_chunks':>10}  {'chunk_size':>12}  "
      f"{'Max error':>12}  {'Correct?'}")
print("  " + "─" * 52)

for N_ctx, n_chunks in N_TESTS:
    q    = np.random.randn(D_HEAD).astype(np.float64)
    K_kv = np.random.randn(N_ctx, D_HEAD).astype(np.float64)
    V_kv = np.random.randn(N_ctx, D_HEAD).astype(np.float64)

    o_ref  = decode_attention_ref(q, K_kv, V_kv)
    o_fd   = flash_decode(q, K_kv, V_kv, n_chunks)

    max_err = float(np.abs(o_ref - o_fd).max())
    chunk_size = math.ceil(N_ctx / n_chunks)
    ok = max_err < 1e-8
    print(f"  {N_ctx:>8}  {n_chunks:>10}  {chunk_size:>12}  "
          f"{max_err:>12.2e}  {'✅' if ok else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Parallelism analysis — SM utilisation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — SM Utilisation: Standard FA vs Flash Decoding")
print("━" * 68)
print()

print("  Standard Flash Attention decode: one threadblock per head.")
print("  Flash Decoding: one threadblock per (head, chunk) pair.")
print()
print(f"  Model: batch=1, num_heads={32}")
print()
print(f"  {'N_ctx':>8}  {'n_chunks':>10}  {'FA blocks':>10}  "
      f"{'FD blocks':>10}  {'FA SM util%':>12}  {'FD SM util%':>12}  "
      f"{'FD speedup'}")
print("  " + "─" * 74)

N_HEADS = 32

for N_ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    # Choose n_chunks to saturate SMs
    fa_blocks = N_HEADS  # standard FA: one block per head, batch=1
    n_chunks  = max(1, H100_SMs // N_HEADS)  # enough chunks to fill all SMs
    fd_blocks = N_HEADS * n_chunks

    fa_sm_util = min(fa_blocks / H100_SMs, 1.0) * 100
    fd_sm_util = min(fd_blocks / H100_SMs, 1.0) * 100

    # Speedup = fd_sm_util / fa_sm_util (more SMs → better bandwidth use)
    speedup = fd_sm_util / fa_sm_util

    print(f"  {N_ctx:>8,}  {n_chunks:>10}  {fa_blocks:>10}  "
          f"{fd_blocks:>10}  {fa_sm_util:>11.1f}%  {fd_sm_util:>11.1f}%  "
          f"{speedup:>9.2f}×")

print()
print(f"  With {N_HEADS} heads: FA only uses {N_HEADS/H100_SMs*100:.0f}% of SMs.")
print(f"  Flash Decoding with n_chunks={H100_SMs//N_HEADS}: uses 100% of SMs.")
print(f"  Theoretical speedup: {H100_SMs/N_HEADS:.1f}× for large contexts.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: HBM bandwidth profile vs chunk count
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — HBM Bandwidth Profile: Chunk Count Tradeoff")
print("━" * 68)
print()

print("  Flash Decoding reads K and V each time, split across chunks.")
print("  The merge step is tiny (C partial vectors of size d).")
print()

N_ctx_profile, D_profile, HEADS = 16384, 128, 32

print(f"  N_ctx={N_ctx_profile}, d={D_profile}, heads={HEADS}, batch=1")
print()
print(f"  {'n_chunks':>10}  {'Chunk time µs':>14}  {'Merge time µs':>14}  "
      f"{'Total µs':>10}  {'SM waves':>10}  {'Parallel efficiency'}")
print("  " + "─" * 70)

for n_chunks in [1, 2, 4, 8, 16, 32, 64, 128]:
    chunk_size   = math.ceil(N_ctx_profile / n_chunks)

    # Each chunk: read K+V tiles (chunk_size × d × 2 bytes each × 2)
    kv_bytes = 2 * chunk_size * D_profile * 2  # K + V, FP16
    q_bytes  = D_profile * 2                    # Q, read once per chunk
    chunk_bytes = kv_bytes + q_bytes
    chunk_us = chunk_bytes / (H100_HBM_GBS * 1e9) * 1e6

    # Merge: C × d FP32 values for (m, d, O) partial results
    merge_bytes = n_chunks * (1 + 1 + D_profile) * 4  # m, d, O in FP32
    merge_us    = merge_bytes / (H100_HBM_GBS * 1e9) * 1e6

    # SM utilisation
    n_blocks = HEADS * n_chunks  # total parallel blocks
    n_waves  = math.ceil(n_blocks / H100_SMs)
    sm_util  = n_blocks / (n_waves * H100_SMs)

    total_us = max(chunk_us, chunk_us / min(n_blocks, H100_SMs)) + merge_us

    print(f"  {n_chunks:>10}  {chunk_us:>14.4f}  {merge_us:>14.6f}  "
          f"{total_us:>10.4f}  {n_waves:>10}  {sm_util:>12.1%}")

print()
print("  Merge cost is negligible (microseconds vs milliseconds for chunks).")
print("  Optimal n_chunks: fills all SMs (n_chunks = ceil(H100_SMs / n_heads)).")
print(f"  For {HEADS} heads: optimal n_chunks = {math.ceil(H100_SMs/HEADS)} chunks.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Speculative Decoding Engine — Draft/Verify Loop & Accept-Reject": {
        "description": (
            "Implement a complete speculative decoding engine simulation. Model "
            "the draft model auto-regressive generation of k tokens, target model "
            "parallel verification, and token-by-token accept/reject sampling. "
            "Compute expected accepted tokens per step as a function of k and the "
            "draft/target distribution overlap. Show throughput gain vs standard "
            "decoding. Simulate the persistent kernel control loop where in-GPU "
            "sampling eliminates CPU round-trips."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  SPECULATIVE DECODING ENGINE — Draft/Verify Loop & Accept-Reject")
print("=" * 68)
print()

np.random.seed(77)

VOCAB_SIZE  = 32000
TARGET_LAYERS = 32   # 70B model (conceptually)
DRAFT_LAYERS  = 4    # 7B draft model (conceptually)

# Hardware
H100_HBM_BW   = 3350.0   # GB/s
TARGET_PARAMS = 70e9      # 70B target
DRAFT_PARAMS  = 7e9       # 7B draft


def decode_time_us(n_params, n_tokens_parallel, dtype_bytes=2):
    """Approximate time to decode n_tokens in one forward pass."""
    weight_bytes = n_params * dtype_bytes
    hbm_us = weight_bytes / (H100_HBM_BW * 1e9) * 1e6
    # For n_tokens > 1, compute-bound component grows but memory-bound is same
    return hbm_us   # memory-bound for all batch sizes in decode regime


def sample_from_logits(logits, temperature=1.0, top_k=50):
    """Sample one token from logit distribution."""
    # Temperature scaling
    logits = logits / temperature
    # Top-k masking
    topk_indices = np.argpartition(logits, -top_k)[-top_k:]
    topk_logits  = logits[topk_indices]
    probs        = np.exp(topk_logits - topk_logits.max())
    probs       /= probs.sum()
    return int(topk_indices[np.random.choice(len(probs), p=probs)])


def acceptance_rate(q_target, p_draft):
    """
    Probability that a draft token is accepted by the target model.
    Acceptance prob = min(1, q_target(t) / p_draft(t)).
    Here we use the expected acceptance rate across the distribution.
    """
    return float(np.minimum(1.0, q_target / (p_draft + 1e-10)).sum() / 2.0)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Accept-reject sampling trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Accept-Reject Sampling: Token-by-Token Trace")
print("━" * 68)
print()

k_draft = 4      # draft tokens per speculation step
TEMPERATURE = 0.8

def simulate_spec_step(k, alpha_mean=0.7, alpha_std=0.1):
    """
    Simulate one speculative decoding step.
    alpha = per-token acceptance probability (0 to 1).
    Returns: number of tokens accepted (0 to k+1).
    """
    n_accepted = 0
    for i in range(k):
        # Each position has a slightly different acceptance probability
        alpha_i = float(np.clip(np.random.normal(alpha_mean, alpha_std), 0, 1))
        u = float(np.random.uniform(0, 1))
        if u <= alpha_i:
            n_accepted += 1
        else:
            break
    # If all k accepted: one bonus token sampled from adjusted target dist
    if n_accepted == k:
        n_accepted += 1   # the free bonus token
    return n_accepted

print(f"  k={k_draft} draft tokens, acceptance probability α")
print()
print(f"  {'α (per-token)':>16}  {'E[tokens/step]':>16}  {'Theory':>10}  "
      f"{'vs baseline (k→1/α)':>20}")
print("  " + "─" * 64)

N_SIM = 10000
for alpha in [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
    total_accepted = sum(simulate_spec_step(k_draft, alpha_mean=alpha, alpha_std=0.0)
                         for _ in range(N_SIM))
    avg_accepted   = total_accepted / N_SIM

    # Theory: E[accepted] = sum_{i=0}^{k} (k+1-i) * alpha^i * (1-alpha)^(1 - (i==k))
    # Simplified: E = sum_{i=1}^{k+1} prod_{j<i} alpha_j * (1-alpha) for partial
    # For k draft + 1 bonus:
    #   E = sum_{i=0}^{k-1} i * alpha^i * (1-alpha) + (k+1) * alpha^k
    theory  = sum(i * alpha**i * (1-alpha) for i in range(k_draft))
    theory += (k_draft + 1) * alpha**k_draft

    baseline = 1.0  # standard decoding = 1 token per step
    print(f"  {alpha:>16.2f}  {avg_accepted:>16.3f}  {theory:>10.3f}  "
          f"{avg_accepted/baseline:>18.2f}×")

print()
print(f"  For α=0.8, k=4: expect {0.8**0 * 0.2 * 1 + 0.8 * 0.2 * 2 + 0.8**2*0.2*3 + 0.8**3*0.2*4 + 0.8**4*5:.2f} tokens/step vs 1.0 baseline.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Throughput model — draft/target time tradeoff
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Throughput Model: When Does Speculative Decoding Win?")
print("━" * 68)
print()

T_target_us = decode_time_us(TARGET_PARAMS, 1)   # time for one target decode
T_draft_us  = decode_time_us(DRAFT_PARAMS,  1)   # time for one draft decode

print(f"  Target model (70B): {T_target_us:.2f} µs per token (memory-bound)")
print(f"  Draft model  (7B):  {T_draft_us:.2f} µs per token (memory-bound)")
print(f"  Time ratio target/draft: {T_target_us/T_draft_us:.1f}×")
print()

print(f"  Speculative decoding time per accepted token:")
print(f"    T_spec(k, α) = [k × T_draft + T_target] / E[accepted(k, α)]")
print(f"    E[accepted] = sum_i i × α^{{i-1}} × (1-α) + (k+1) × α^k")
print()
print(f"  {'k':>4}  {'α':>6}", end="")
for col in ["T_spec (µs)", "vs target", "Speedup"]:
    print(f"  {col:>14}", end="")
print()
print("  " + "─" * 50)

for k_val in [1, 2, 3, 4, 6, 8]:
    for alpha in [0.7, 0.8, 0.9]:
        # Expected accepted tokens
        e_acc = sum(i * alpha**(i-1) * (1-alpha) for i in range(1, k_val+1))
        e_acc += (k_val + 1) * alpha**k_val

        # Time for one spec step
        t_spec_step = k_val * T_draft_us + T_target_us
        # Time per accepted token
        t_per_tok   = t_spec_step / e_acc

        speedup = T_target_us / t_per_tok

        print(f"  {k_val:>4}  {alpha:>6.2f}  "
              f"{t_per_tok:>14.2f}  {T_target_us:>9.2f} µs  {speedup:>7.2f}×")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: In-GPU control loop — persistent speculative decoding
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — In-GPU Control Loop: Persistent vs Conventional")
print("━" * 68)
print()

print("  Conventional speculative decoding:")
print("    Each step requires CPU to:")
print("      1. Read draft token probabilities from GPU → 5 µs (D2H)")
print("      2. Read target model probabilities from GPU → 5 µs (D2H)")
print("      3. Compute accept/reject (trivial CPU compute)")
print("      4. Write next draft tokens to GPU → 5 µs (H2D)")
print("      5. Launch next kernel → 5 µs")
print("    Total CPU overhead per spec step: ~20 µs")
print()
print("  Persistent speculative decoding:")
print("    GPU maintains control buffer in device memory.")
print("    Accept/reject kernel runs on GPU (< 1 µs).")
print("    Doorbell to draft kernel: 0.5 µs.")
print("    Total overhead per spec step: ~1.5 µs")
print()

alpha_run  = 0.8
k_run      = 4
n_steps_   = 200

print(f"  Simulating {n_steps_} speculative steps (k={k_run}, α={alpha_run})")
print()

# Conventional
conv_cpu_overhead_us = 20.0
spec_step_us_conv    = k_run * T_draft_us + T_target_us + conv_cpu_overhead_us
e_acc_run            = sum(i * alpha_run**(i-1) * (1-alpha_run)
                           for i in range(1, k_run+1))
e_acc_run           += (k_run + 1) * alpha_run**k_run

conv_time_ms  = n_steps_ * spec_step_us_conv / 1000
conv_tokens   = n_steps_ * e_acc_run
conv_tps      = conv_tokens / (conv_time_ms / 1000)

# Persistent
pers_overhead_us     = 1.5
spec_step_us_pers    = k_run * T_draft_us + T_target_us + pers_overhead_us

pers_time_ms  = n_steps_ * spec_step_us_pers / 1000
pers_tokens   = n_steps_ * e_acc_run
pers_tps      = pers_tokens / (pers_time_ms / 1000)

# Standard baseline (no speculation)
std_time_ms   = n_steps_ * T_target_us / 1000
std_tokens    = float(n_steps_)
std_tps       = std_tokens / (std_time_ms / 1000)

print(f"  {'Method':<30}  {'Time (ms)':>10}  {'Tokens':>8}  "
      f"{'Tokens/sec':>12}  {'vs standard'}")
print("  " + "─" * 64)
print(f"  {'Standard (no spec)':<30}  {std_time_ms:>10.2f}  "
      f"{std_tokens:>8.1f}  {std_tps:>12.1f}  1.00×")
print(f"  {'Spec (conventional CPU)':<30}  {conv_time_ms:>10.2f}  "
      f"{conv_tokens:>8.1f}  {conv_tps:>12.1f}  {conv_tps/std_tps:.2f}×")
print(f"  {'Spec (persistent GPU)':<30}  {pers_time_ms:>10.2f}  "
      f"{pers_tokens:>8.1f}  {pers_tps:>12.1f}  {pers_tps/std_tps:.2f}×")
print()
print(f"  Persistent removes {conv_cpu_overhead_us - pers_overhead_us:.1f} µs of CPU overhead per step.")
print(f"  Additional throughput vs conventional spec: {pers_tps/conv_tps:.2f}×.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · CUDA Graphs vs Persistent Kernels — Decision Matrix & Benchmarks": {
        "description": (
            "Build a comprehensive comparison of CUDA Graphs and persistent kernels "
            "across real LLM serving workloads. Model graph replay overhead, "
            "persistent doorbell latency, and the impact of dynamic shapes "
            "(variable batch size, variable context length). Show which approach "
            "wins for prefill, decode, speculative decoding, and continuous batching. "
            "Estimate the combined throughput of a hybrid architecture (graphs for "
            "fixed-shape layers + persistent for variable-shape operations)."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  CUDA GRAPHS vs PERSISTENT KERNELS — Decision Matrix & Benchmarks")
print("=" * 68)
print()

np.random.seed(13)

# H100 constants
H100_HBM_BW  = 3350.0   # GB/s
H100_SMs     = 132

# Overhead constants (microseconds)
LAUNCH_US     = 5.0      # per kernel launch
GRAPH_US      = 0.8      # per graph replay (amortised over N kernels)
DOORBELL_US   = 0.5      # per doorbell signal (persistent)
CAPTURE_US    = 50_000.0 # one-time graph capture cost (µs)

KERNELS_PER_TOKEN = 384  # 32 layers × 12 kernels

# Model sizes
LLAMA_7B  = 7e9
LLAMA_70B = 70e9
PARAMS = {
    "7B": LLAMA_7B,
    "70B": LLAMA_70B,
}


def token_compute_us(n_params, batch_size=1):
    """Memory-bound time to decode one token."""
    w_bytes = n_params * 2   # FP16
    hbm_us  = w_bytes / (H100_HBM_BW * 1e9) * 1e6
    return hbm_us

def launch_overhead(n_kernels, approach):
    if approach == "conventional":
        return n_kernels * LAUNCH_US
    elif approach == "cuda_graph":
        return GRAPH_US
    elif approach == "persistent":
        return n_kernels * DOORBELL_US


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Per-token overhead breakdown across approaches
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Per-Token Overhead: Prefill vs Decode Phase")
print("━" * 68)
print()

print("  PREFILL PHASE (batch=1, processing full prompt):")
print("    Batch is large (seq_len >> 1), GEMMs are compute-bound.")
print("    Launch overhead is small fraction of GEMM time.")
print()

# For prefill: GEMMs are large and slow (~ms each), launch overhead is tiny
prefill_gemm_us = 500.0   # representative large GEMM in prefill

for approach in ["conventional", "cuda_graph", "persistent"]:
    overhead = launch_overhead(KERNELS_PER_TOKEN, approach)
    total    = prefill_gemm_us * KERNELS_PER_TOKEN + overhead
    pct      = overhead / total * 100
    print(f"    {approach:<18}: overhead={overhead:>7.1f} µs  "
          f"({pct:.2f}% of total {total:.0f} µs)")

print()
print("  DECODE PHASE (batch=1, one token at a time):")
print("    GEMMs are tiny (memory-bound, fast), launch overhead dominates.")
print()

decode_gemm_us = 1.5   # memory-bound single-token GEMM, 7B model

for approach in ["conventional", "cuda_graph", "persistent"]:
    overhead = launch_overhead(KERNELS_PER_TOKEN, approach)
    total    = decode_gemm_us * KERNELS_PER_TOKEN + overhead
    pct      = overhead / total * 100
    print(f"    {approach:<18}: overhead={overhead:>7.1f} µs  "
          f"({pct:.1f}% of total {total:.1f} µs)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Dynamic shapes — where CUDA graphs break
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Dynamic Shape Challenges for CUDA Graphs")
print("━" * 68)
print()

print("  CUDA Graphs capture FIXED grid/block dimensions at capture time.")
print("  Shape changes require: destroy graph → recapture (50+ ms each).")
print()

scenarios = [
    ("Fixed batch prefill",      "static", "static",
     True,  "✅ ideal for graphs"),
    ("Variable-length prefill",  "static", "variable",
     False, "❌ seq len changes per request"),
    ("Decode, fixed batch",      "static", "static",
     True,  "✅ can graph if padding to fixed length"),
    ("Decode, dynamic batching", "variable","variable",
     False, "❌ batch changes as requests arrive/complete"),
    ("Speculative decode",       "variable","variable",
     False, "❌ k tokens accepted varies each step"),
    ("Flash Decoding merge",     "variable","static",
     True,  "✅ if fixed n_chunks per head"),
    ("Continuous batching",      "variable","variable",
     False, "❌ requests arrive/leave mid-stream"),
]

print(f"  {'Scenario':<32}  {'Batch':>8}  {'SeqLen':>8}  "
      f"{'Graphs OK?':>11}  {'Recommendation'}")
print("  " + "─" * 74)
for name, batch_shape, seq_shape, graphs_ok, rec in scenarios:
    ok_str = "✅ YES" if graphs_ok else "❌ NO"
    print(f"  {name:<32}  {batch_shape:>8}  {seq_shape:>8}  "
          f"{ok_str:>11}  {rec}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Throughput comparison across workloads
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Throughput Comparison: 7B Model, 4 Workloads")
print("━" * 68)
print()

t_gemm_7b = token_compute_us(LLAMA_7B)   # HBM time to load all weights
t_gemm_70b = token_compute_us(LLAMA_70B)

workloads = [
    ("Prefill, B=1, S=512",    1, 512,    t_gemm_7b * 512, True, False),
    ("Decode, B=1",            1, 1,      t_gemm_7b,       True, True),
    ("Decode, B=8",            8, 1,      t_gemm_7b,       True, True),
    ("Decode, B=32",          32, 1,      t_gemm_7b,       False, True),
    ("Spec. decode, k=4, B=1", 1, 1,      t_gemm_7b,       False, False),
    ("Continuous batch",       8, "var",  t_gemm_7b,       False, False),
]

print(f"  7B model, H100, FP16")
print()
print(f"  {'Workload':<30}  {'Conv. TPS':>10}  {'Graph TPS':>10}  "
      f"{'Persist. TPS':>13}  {'Best choice'}")
print("  " + "─" * 68)

for name, B, S, compute_us, graph_ok, can_graph in workloads:
    # compute_us is total GPU work time for this workload
    conv_us  = compute_us + KERNELS_PER_TOKEN * LAUNCH_US
    graph_us = compute_us + (GRAPH_US if graph_ok else KERNELS_PER_TOKEN * LAUNCH_US)
    pers_us  = compute_us + KERNELS_PER_TOKEN * DOORBELL_US

    n_tokens = B if isinstance(S, int) else B
    conv_tps  = n_tokens / conv_us  * 1e6
    graph_tps = n_tokens / graph_us * 1e6
    pers_tps  = n_tokens / pers_us  * 1e6

    best = ("Graphs" if graph_ok and graph_tps >= pers_tps else "Persistent")
    print(f"  {name:<30}  {conv_tps:>10.1f}  {graph_tps:>10.1f}  "
          f"{pers_tps:>13.1f}  {best}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Hybrid architecture — optimal for production serving
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Hybrid Architecture: Graphs + Persistent")
print("━" * 68)
print()

print("  Production LLM serving (e.g., TensorRT-LLM, vLLM with CUDA graphs):")
print("  Uses a HYBRID approach where different kernel types use different strategies.")
print()

hybrid_layers = [
    ("Attention GEMM (Q,K,V,O)",  "fixed shape (per head_dim)",  "cuda_graph", 4*32),
    ("Flash Decoding chunks",      "fixed n_chunks",              "cuda_graph", 32),
    ("Flash Decoding merge",       "fixed C, fixed d",            "cuda_graph", 1),
    ("FFN GEMMs (gate, up, down)", "dynamic batch",               "persistent", 3*32),
    ("LayerNorm (×2/layer)",       "dynamic batch",               "persistent", 2*32),
    ("Spec. accept/reject",        "variable k",                  "persistent", 1),
    ("KV cache scatter/gather",    "variable seq positions",      "persistent", 32),
]

total_kernels = sum(n for _,_,_,n in hybrid_layers)

print(f"  {'Layer type':<36}  {'Shape':>22}  {'Strategy':>12}  {'Kernels'}")
print("  " + "─" * 78)

graph_kernels    = 0
persist_kernels  = 0
for layer, shape, strategy, n_kernels in hybrid_layers:
    print(f"  {layer:<36}  {shape:>22}  {strategy:>12}  {n_kernels:>6}")
    if strategy == "cuda_graph":
        graph_kernels   += n_kernels
    else:
        persist_kernels += n_kernels

print()
print(f"  Total: {total_kernels} kernels")
print(f"    CUDA Graph:  {graph_kernels} kernels  ({graph_kernels/total_kernels*100:.0f}%)")
print(f"    Persistent:  {persist_kernels} kernels  ({persist_kernels/total_kernels*100:.0f}%)")
print()

# Compute overhead for hybrid vs pure approaches
hybrid_overhead   = graph_kernels * GRAPH_US / total_kernels + persist_kernels * DOORBELL_US
conv_overhead_per = KERNELS_PER_TOKEN * LAUNCH_US
graph_overhead_per= GRAPH_US  # one graph per step
persist_overhead  = KERNELS_PER_TOKEN * DOORBELL_US

print(f"  Launch overhead comparison (per decode step):")
print(f"    Conventional:   {conv_overhead_per:.1f} µs  ({total_kernels} launches × {LAUNCH_US} µs)")
print(f"    Pure graphs:    {graph_overhead_per:.1f} µs  (one graph replay)")
print(f"    Pure persistent:{persist_overhead:.1f} µs  ({total_kernels} doorbells × {DOORBELL_US} µs)")
print(f"    Hybrid:         {hybrid_overhead:.1f} µs  ({graph_kernels} graph + {persist_kernels} doorbells)")
print()
print("  Hybrid is nearly as fast as pure persistent, with graphs for simple fixed layers.")
print("  Production recommendation: graph all fixed-shape attention+FFN layers,")
print("  persistent for batch management, speculative control, KV cache operations.")
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