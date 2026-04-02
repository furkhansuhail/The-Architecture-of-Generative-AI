"""
Streams & Async — Stream Creation / Sync, cudaEvent Timing & Compute/Copy Overlap
==================================================================================

A CUDA stream is a sequence of GPU operations that execute in ISSUE ORDER.
Operations on the same stream are serialised: each waits for the previous to
finish before starting. Operations on DIFFERENT streams are independent and
may execute concurrently — overlapping on the GPU's copy engines and SM
compute units simultaneously.

This concurrency model is the foundation of GPU pipeline performance. A
GPU without streams is a sequential device: HtoD copy, then compute, then
DtoH copy. A GPU with streams is a pipelined device: while SM kernels
process batch N, the PCIe copy engines transfer batch N+1 in and batch N-1
out — achieving up to 3× effective throughput on copy-bound workloads.

Three capabilities define the streams abstraction:

    STREAM CREATION AND SYNCHRONISATION:
        Creating streams, setting their priority, submitting work to them,
        and synchronising at different granularities: device-wide (all SMs
        and copy engines), stream-level (one command queue), or event-level
        (a specific point in a stream's timeline). The null stream (stream 0)
        and its special blocking semantics. cudaStreamCreateWithPriority.

    CUDAEVENT TIMING:
        Events are timestamped markers inserted into a stream's queue.
        The elapsed time between two events — measured on the GPU clock —
        is the most accurate way to time GPU operations. CPU-side timers
        include host scheduling latency, OS jitter, and the host↔device
        protocol overhead. GPU events measure only the GPU-side time.

    COMPUTE / COPY OVERLAP:
        Modern NVIDIA GPUs have INDEPENDENT HARDWARE ENGINES:
            - Compute engines: SM-based kernel execution (1 or more)
            - HtoD copy engine: async memcpy from host to device
            - DtoH copy engine: async memcpy from device to host
            - P2P engine: GPU-to-GPU copies on NVLink
        These engines run in parallel. Overlapping them requires:
            (a) Pinned (page-locked) host memory for asynchronous copies.
            (b) Separate streams for compute and copy.
            (c) Correct dependency management to preserve correctness.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Streams & Async — Stream Creation / Sync, cudaEvent Timing & Compute/Copy Overlap"
DISPLAY_NAME = "17 · Streams & Async"
ICON         = "🌊"
SUBTITLE     = "CUDA Streams · cudaEvent · HtoD/DtoH Overlap · Pinned Memory · Double Buffer"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE CUDA EXECUTION MODEL: STREAMS, ENGINES & CONCURRENCY

### The GPU Hardware Execution Units

    A modern NVIDIA GPU (A100, H100) contains MULTIPLE INDEPENDENT HARDWARE
    ENGINES that can operate concurrently:

        SM COMPUTE ENGINE(S):
            Executes CUDA kernels. On A100: up to 108 SMs simultaneously.
            Multiple kernel launches on different streams can co-reside on
            different SMs (MPS, Volta+ multi-process service).

        COPY ENGINE — HOST-TO-DEVICE (HtoD):
            Dedicated DMA hardware for moving data from host (CPU) memory
            to device (GPU) memory via PCIe / NVLink.
            Operates independently of SM compute.
            Throughput: PCIe Gen4 x16 ≈ 32 GB/s, NVLink ≈ 300 GB/s.

        COPY ENGINE — DEVICE-TO-HOST (DtoH):
            Separate dedicated DMA for moving from device to host.
            On A100/H100: HtoD and DtoH engines are INDEPENDENT — both can
            run simultaneously with full bandwidth each direction.

        COPY ENGINE — PEER-TO-PEER (P2P):
            GPU-to-GPU data movement via NVLink or PCIe peer access.

    CONCURRENCY RULE: operations using DIFFERENT HARDWARE ENGINES can overlap.
        Kernel (SM) + HtoD copy (copy engine) = can overlap ✅
        Kernel (SM) + DtoH copy (copy engine) = can overlap ✅
        HtoD + DtoH (two separate engines)     = can overlap ✅
        Kernel A + Kernel B on same SM resource = CAN overlap if SM capacity allows
        Two operations on the SAME stream       = serialised, never overlap ❌

### Streams as Hardware Command Queues

    Each CUDA stream corresponds to a GPU-side command queue.
    The GPU's work distributor reads commands from queues in issue order
    within each queue, but reads from MULTIPLE queues interleaved.

    STREAM 0 (null stream, default stream):
        Special semantics: it SYNCHRONISES with ALL other streams.
        Any operation submitted to stream 0 waits for ALL non-null streams
        to complete before starting.
        Any operation on a non-null stream submitted AFTER stream 0 work
        waits for that stream 0 work to complete.
        CONSEQUENCE: mixing stream 0 with other streams defeats concurrency.
        RECOMMENDATION: never use stream 0 in performance-critical code.
        Create explicit streams and use them for all operations.

    PER-THREAD DEFAULT STREAM (PTDS):
        Alternative to stream 0 with weaker blocking semantics.
        Enable with: nvcc --default-stream per-thread
        Each CPU thread gets its own default stream that doesn't sync globally.
        Useful for multi-threaded inference servers.

### Stream Creation and Priority

    CREATE:
        cudaStream_t stream;
        cudaStreamCreate(&stream);

    CREATE WITH PRIORITY (for QoS — lower number = higher priority):
        int lowest_priority, highest_priority;
        cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
        cudaStreamCreateWithPriority(&stream, cudaStreamDefault, highest_priority);

        High-priority streams pre-empt lower-priority work at the warp granularity.
        Use for: latency-critical operations (e.g., online inference control vs
        background data loading).

    DESTROY:
        cudaStreamDestroy(stream);
        Blocks until all previously submitted work in the stream completes.

    QUERY WITHOUT BLOCKING:
        cudaError_t status = cudaStreamQuery(stream);
        // Returns cudaSuccess if all work done, cudaErrorNotReady if pending.
        // Does NOT block the CPU.


##### PART 2 — STREAM SYNCHRONISATION: BARRIERS, EVENTS AND WAITS

### The Synchronisation Hierarchy

    LEVEL 1 — DEVICE-WIDE: cudaDeviceSynchronize()
        Blocks the calling CPU thread until ALL GPU work across ALL streams completes.
        Highest latency (~5 µs synchronisation overhead + all pending work).
        Use sparingly. Appropriate for: profiling boundaries, end of a program,
        or after a single-threaded GPU work phase before CPU post-processing.

    LEVEL 2 — STREAM-LEVEL: cudaStreamSynchronize(stream)
        Blocks the calling CPU thread until ALL work submitted to the given stream completes.
        Other streams can continue running concurrently.
        Latency: ~2–5 µs + time for pending stream work.
        Use for: waiting for one output stream while other streams continue.

    LEVEL 3 — EVENT-LEVEL: cudaEventSynchronize(event)
        Blocks the CPU thread until the specified event is reached in its stream.
        Allows synchronising on a SPECIFIC POINT in a stream, not the full queue.
        Latency: ~2–5 µs + time to reach the event.
        Use for: fine-grained pipeline synchronisation.

    LEVEL 4 — GPU-SIDE WAIT: cudaStreamWaitEvent(stream, event)
        Inserts a WAIT on the GPU side: stream will not advance past this point
        until event is complete. The CPU is NOT blocked — it continues immediately.
        Zero CPU latency (it's a GPU-side dependency, like a semaphore in the queue).
        This is the KEY primitive for producer-consumer pipeline construction.

    RULE OF THUMB:
        cudaDeviceSynchronize → use at most once per major phase (end of epoch).
        cudaStreamSynchronize → use when you need CPU to read GPU results.
        cudaStreamWaitEvent  → use to connect streams without CPU involvement.

### Event-Based Producer-Consumer Pattern

    PRODUCER stream (stream_compute) writes output.
    CONSUMER stream (stream_copy) reads that output to copy to CPU.
    We need: consumer must wait for producer, but BOTH run without CPU blocking.

        // CPU submits BOTH operations then does other work:
        kernel<<<grid, block, 0, stream_compute>>>(d_input, d_output);
        cudaEventRecord(event_compute_done, stream_compute);   // "producer done" marker

        // stream_copy will pause until event_compute_done fires:
        cudaStreamWaitEvent(stream_copy, event_compute_done, 0);
        cudaMemcpyAsync(h_output, d_output, size, DtoH, stream_copy);

        // CPU is FREE — not blocked. GPU enforces the ordering.
        do_other_cpu_work();

        // Only block when we actually need h_output:
        cudaStreamSynchronize(stream_copy);

    The GPU hardware serialises stream_copy at the event point.
    The CPU never stalls during the GPU-GPU coordination.

### cudaStreamWaitEvent Flags

    cudaStreamWaitEvent(stream, event, flags):
        flags = 0: default, wait for event to complete.
        flags = cudaEventWaitExternal: for multi-GPU external dependencies.

    Multiple streams can ALL wait on the same event:
        cudaStreamWaitEvent(stream_A, event_producer, 0);
        cudaStreamWaitEvent(stream_B, event_producer, 0);
        cudaStreamWaitEvent(stream_C, event_producer, 0);
    All three start processing as soon as the producer event fires.
    This is the BROADCAST pattern — one signal wakes many consumers.


##### PART 3 — CUDAEVENT TIMING: PRECISE GPU MEASUREMENT

### Why GPU Events Are More Accurate Than CPU Timers

    CPU timers (std::chrono::high_resolution_clock, clock_gettime) measure
    WALL CLOCK TIME on the host. For GPU operations this includes:
        - CUDA API call overhead (~1–5 µs per call)
        - CPU thread scheduling jitter (OS may context-switch the CPU thread)
        - CPU-GPU round-trip synchronisation latency

    Example: measuring a 100 µs kernel with CPU timers:
        auto t0 = chrono::now();
        kernel<<<...>>>();
        cudaDeviceSynchronize();
        auto t1 = chrono::now();
        // Measured: 100 µs + 5 µs sync overhead = 105 µs (5% over-estimate).

    GPU EVENTS stamp the GPU's hardware timer AT THE EXACT POINT THE EVENT
    IS REACHED IN THE GPU COMMAND QUEUE. They measure:
        - Only the time between the two event positions in the GPU timeline.
        - No CPU scheduling, no synchronisation overhead, no API latency.
        - Resolution: ~0.5 µs (GPU timer tick).

### Event Creation and Timing API

    CREATE EVENTS:
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

    WITH TIMING FLAGS (more precise — disables concurrent event recording):
        cudaEventCreateWithFlags(&start, cudaEventDefault);   // default
        cudaEventCreateWithFlags(&start, cudaEventBlockingSync); // CPU blocks on sync
        cudaEventCreateWithFlags(&start, cudaEventDisableTiming); // zero overhead, not timeable

    RECORD INTO A STREAM (inserts a timestamp at this queue position):
        cudaEventRecord(start, stream);   // records when the GPU REACHES this point
        // (NOT when the CPU calls this function — the GPU may still be busy)

    QUERY GPU-SIDE STATUS (non-blocking):
        cudaError_t st = cudaEventQuery(event);
        // cudaSuccess: event has been reached. cudaErrorNotReady: pending.

    WAIT FOR EVENT ON CPU:
        cudaEventSynchronize(stop);  // CPU blocks until 'stop' event fires on GPU

    MEASURE ELAPSED TIME:
        float ms;
        cudaEventElapsedTime(&ms, start, stop);  // milliseconds, float32

### Correct Timing Pattern

    // Step 1: record start marker
    cudaEventRecord(start, stream);

    // Step 2: submit work (the work will execute AFTER start is stamped)
    kernel_A<<<grid, block, 0, stream>>>(args);
    kernel_B<<<grid, block, 0, stream>>>(args);

    // Step 3: record stop marker (AFTER the work in the queue)
    cudaEventRecord(stop, stream);

    // Step 4: wait for stop to be reached (GPU has finished all work up to here)
    cudaEventSynchronize(stop);

    // Step 5: read elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CRITICAL: cudaEventRecord is ASYNCHRONOUS from the CPU's perspective.
    It queues the timestamp operation; the GPU records the time when it
    EXECUTES that queue entry — which may be much later than the CPU call.
    The CPU call returns immediately.

### What cudaEventElapsedTime Measures

    cudaEventElapsedTime measures the GPU's internal hardware timer between
    the two event recording points. Specifically:

    GPU TIMELINE:
        [...previous work...]
        [start event stamped]   ← GPU clock tick A
        [kernel A executes]
        [kernel B executes]
        [stop event stamped]    ← GPU clock tick B
        [...subsequent work...]

    Elapsed time = (tick_B - tick_A) / GPU_timer_frequency
                 = ONLY the time for kernel A + kernel B.
                 DOES NOT include CPU launch overhead, event API time, or
                 any work that happened before start or after stop.

### Common Timing Mistakes

    MISTAKE 1 — Forgetting cudaEventSynchronize before ElapsedTime:
        cudaEventRecord(stop, stream);
        cudaEventElapsedTime(&ms, start, stop);   // ← BUG: stop not reached yet!
        // ms will contain garbage or 0.

    MISTAKE 2 — Recording events on different streams and not syncing:
        cudaEventRecord(start, stream_A);
        kernel<<<..., stream_B>>>();
        cudaEventRecord(stop, stream_A);
        // stop is on stream_A but the kernel ran on stream_B!
        // Elapsed time measures nothing useful.

    MISTAKE 3 — Using CPU timer while GPU is running asynchronously:
        auto t0 = chrono::now();
        kernel<<<..., stream>>>();
        // kernel is RUNNING ASYNCHRONOUSLY — CPU returns immediately.
        auto t1 = chrono::now();
        // t1 - t0 measures only the API call time (~µs), not the kernel!
        // FIX: add cudaStreamSynchronize(stream) between t0 and t1.


##### PART 4 — PINNED MEMORY: THE PREREQUISITE FOR ASYNC COPIES

### Why Async Copies Require Pinned Memory

    PAGEABLE (NORMAL) HOST MEMORY — cudaMalloc not involved; just malloc/new.
        Physical address can change: OS may page it out to disk or remap it
        during execution. The CPU's MMU translates virtual → physical dynamically.
        DMA hardware (the PCIe copy engine) works with PHYSICAL addresses.
        For pageable memory: CUDA must FIRST COPY to an internal pinned bounce
        buffer, THEN DMA from that buffer to the GPU. This copy is serial with
        the DMA. cudaMemcpyAsync on pageable memory becomes SYNCHRONOUS.

    PINNED (PAGE-LOCKED) HOST MEMORY — cudaMallocHost or cudaHostAlloc.
        Physical address is FIXED (OS will not page it out or remap).
        The DMA engine can read directly from the physical address.
        No bounce buffer needed. TRUE asynchronous DMA is possible.
        The CPU can execute other code while the copy engine transfers data.

    ALLOCATION:
        float *h_data;
        cudaMallocHost(&h_data, N * sizeof(float));   // pinned
        // OR:
        cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault);

    DEALLOCATION:
        cudaFreeHost(h_data);

### Pinned Memory Cost

    PROS:
        True async transfers: CPU and GPU copy engine run concurrently.
        Higher PCIe bandwidth: typically 1.5–2× faster than pageable transfers.
        Low transfer latency: no bounce-buffer copy.

    CONS:
        Slower allocation/deallocation (OS pins physical pages).
        Consumes PHYSICAL RAM — reduces available physical memory for OS and other processes.
        Too much pinned memory → physical memory pressure → OS thrashing.
        Cannot be swapped — always occupies physical RAM even if not currently in use.

    RULE OF THUMB:
        Pin memory that is repeatedly transferred (e.g., mini-batch buffers that
        receive new data every iteration).
        Do NOT pin large one-time buffers or model weight checkpoints.
        For deep learning training: PyTorch DataLoader uses pinned memory by default
        when num_workers > 0 (DataLoader(pin_memory=True) is the explicit control).

### Mapped (Zero-Copy) Memory

    cudaHostAlloc(&h_data, N, cudaHostAllocMapped):
        Host memory is BOTH pinned AND mapped into the GPU's address space.
        GPU kernels can access h_data directly via PCIe (no explicit copy needed).
        Each GPU access triggers a PCIe transaction.
        USEFUL FOR: data accessed once per kernel launch, or for CPUs that share
        physical memory with the GPU (ARM SoCs, discrete GPU with UVA).
        NOT USEFUL FOR: data accessed many times per kernel (PCIe latency repeated).

### Unified Virtual Addressing (UVA) and cudaMemcpyDefault

    With UVA (CUDA 4.0+, compute capability >= 2.0):
        Host and device allocations share a single virtual address space.
        cudaMemcpyDefault auto-detects copy direction from pointer provenance.
        No need to specify cudaMemcpyHostToDevice vs cudaMemcpyDeviceToHost.


##### PART 5 — COMPUTE / COPY OVERLAP: THE DOUBLE BUFFER PATTERN

### The Naive Pipeline (No Overlap)

    For processing N batches of data from CPU to GPU:

    NAÏVE SEQUENTIAL (single stream):
        for batch in range(N):
            HtoD(batch)          [PCIe copy: T_copy µs]
            kernel(batch)        [Compute: T_kernel µs]
            DtoH(batch)          [PCIe copy: T_copy µs]

    TOTAL TIME: N × (T_copy + T_kernel + T_copy) = N × (2*T_copy + T_kernel)

    ALL THREE OPERATIONS USE DIFFERENT HARDWARE. At any given moment, only
    ONE is running — the other two hardware engines are idle.
    GPU utilisation: T_kernel / (2*T_copy + T_kernel).
    For T_copy = T_kernel: utilisation = 33%. Wasteful.

### The Double Buffer Pattern (3-Stage Pipeline)

    ALLOCATE TWO DEVICE BUFFERS: d_buf[0] and d_buf[1] (the "double buffer").
    ALLOCATE TWO HOST PINNED BUFFERS: h_in[0] and h_in[1].
    CREATE THREE STREAMS: stream_htod, stream_compute, stream_dtoh.

    PIPELINE EXECUTION:
        Step 0 (warm-up): HtoD(batch[0]) → d_buf[0]
        Step 1 (warm-up): HtoD(batch[1]) → d_buf[1] || kernel(batch[0], d_buf[0])

        Main loop (i = 1, 2, ..., N-2):
            HtoD(batch[i+1]) → d_buf[(i+1)%2]  ┐
            kernel(batch[i], d_buf[i%2])         ├ PARALLEL
            DtoH(result[i-1], d_buf[(i-1)%2])   ┘

        Step N-1 (drain): kernel(batch[N-1]) || DtoH(result[N-2])
        Step N   (drain): DtoH(result[N-1])

    STEADY-STATE: all three engines busy simultaneously.
    TOTAL TIME ≈ N × max(T_htod, T_kernel, T_dtoh) + pipeline fill/drain overhead.
    For balanced T_htod = T_kernel = T_dtoh: speedup ≈ 3×.
    For T_kernel >> T_copy: speedup ≈ T_kernel + 2*T_copy / max = 1× (compute-bound).
    Maximum speedup is bounded by max(T_htod, T_kernel, T_dtoh).

### Event-Based Dependency Management in Double Buffer

    The critical correctness constraint: the HtoD copy for batch[i+1] must
    complete BEFORE the kernel starts reading d_buf[(i+1)%2].

    CORRECTLY WIRED DOUBLE BUFFER:

        // After HtoD(batch[i+1]) to d_buf[next]:
        cudaEventRecord(event_htod[next], stream_htod);

        // Kernel on stream_compute waits for this HtoD:
        cudaStreamWaitEvent(stream_compute, event_htod[next], 0);
        kernel<<<..., stream_compute>>>(d_buf[next]);

        // After kernel(batch[i]):
        cudaEventRecord(event_compute[cur], stream_compute);

        // DtoH stream waits for compute to finish before reading d_buf[cur]:
        cudaStreamWaitEvent(stream_dtoh, event_compute[cur], 0);
        cudaMemcpyAsync(h_out[cur], d_result[cur], size, DtoH, stream_dtoh);

    The GPU hardware enforces all ordering.
    The CPU submits all these operations then does other work (or sleeps).

### Triple Buffering

    With THREE device buffers, the pipeline can be deeper:
        While kernel processes buffer A: HtoD fills buffer B AND DtoH drains buffer C.
        When kernel finishes A: start kernel on B, start DtoH on A, start HtoD on C.
    This requires N=3 buffers but maximises pipeline depth for bursty transfers.
    Used when HtoD and DtoH are both slower than the kernel (deep pipeline needed).


##### PART 6 — MULTI-STREAM KERNEL CONCURRENCY

### When Multiple Kernels Run Concurrently

    Two kernels on DIFFERENT streams can execute concurrently on the GPU when:
        1. Both kernels have been submitted (not blocked by stream dependencies).
        2. Sufficient SM resources are available (neither kernel saturates all SMs).
        3. Neither kernel uses all available shared memory or registers per SM.
        4. The CUDA device supports concurrent kernel execution (compute capability >= 2.0).

    RESOURCE CONTENTION LIMITS CONCURRENCY:
        If kernel A uses 50% of all SMs and kernel B uses 60%: they CANNOT overlap.
        If kernel A uses 20% of SMs and kernel B uses 30%: POSSIBLE overlap (50% combined).
        Rule of thumb: kernels that are small relative to the GPU (< 50% SM occupancy)
        are good candidates for concurrent execution.

### The GPU Work Distributor

    The GPU's work distributor assigns thread blocks from different streams
    to available SM slots. It does NOT preempt running thread blocks — once a
    block starts, it runs to completion on its SM.

    CONSEQUENCE: if kernel A fills all SMs with thread blocks, no blocks from
    kernel B can start until SM slots free up.

    FIFO SCHEDULING: within each stream, blocks execute in submission order.
    ROUND-ROBIN across streams: the work distributor typically interleaves blocks
    from different streams at the thread-block granularity.

### Multi-Stream Independence: The Hyper-Q Architecture

    Pre-Kepler GPUs (before GK110): single hardware command queue.
        Only ONE stream's commands could be at the head of the queue at once.
        False dependencies: stream B would stall if stream A had a blocking op.

    Kepler+ GPUs (GK110+): HYPER-Q — 32 hardware work queues.
        Each stream gets its own queue. Work distributor reads from all 32.
        True independence between streams — no false serialisation.
        A100/H100: 128 hardware queues (even more concurrency potential).

### Practical Limits to Multi-Kernel Concurrency

    Even with Hyper-Q, several factors limit practical concurrency:

    MEMORY BANDWIDTH CONTENTION:
        Two kernels that both issue many HBM reads will contend for bandwidth.
        Combined throughput cannot exceed the GPU's HBM peak.
        Adding a second concurrent kernel often slows BOTH by 40–60%.

    L2 CACHE THRASHING:
        Two large working sets in concurrent kernels evict each other's data.
        Combined working set > L2: effective bandwidth drops for both.

    REGISTER FILE SHARING:
        More active warps per SM = more total registers needed.
        If two concurrent kernels each need 64 registers/warp and the SM has
        65536 registers, maximum active warps = 65536/64 = 1024 per SM.

    WHEN CONCURRENT KERNELS HELP:
        Kernel A is latency-bound (waiting for memory) and kernel B is compute-bound.
        A fills latency with B's compute. This is the "warp-level latency hiding"
        idea applied at the kernel level.
        Example: attention (memory-bound) + FFN (compute-bound) from different requests.

### Stream Priorities for Latency-Critical Workloads

    Creating streams with different priorities enables the GPU scheduler to
    PREEMPT lower-priority warps (at warp granularity) for high-priority ones.

    USE CASE — inference server:
        High priority stream: online inference (SLA-bound).
        Low priority stream: background tasks (loading next batch, logging).
        The background task never causes latency spikes for the inference path.

    IMPLEMENTATION:
        cudaStreamCreateWithPriority(&inference_stream, cudaStreamDefault, -1);  // higher
        cudaStreamCreateWithPriority(&background_stream, cudaStreamDefault, 0);  // lower


##### PART 7 — PROFILING STREAMS WITH NSIGHT SYSTEMS

### What Nsight Systems Shows for Streams

    Nsight Systems displays the GPU timeline with one row per CUDA stream:
        - HtoD copies: labelled "[HtoD]" on the copy engine row.
        - Kernel launches: labelled by kernel name on the SM row.
        - DtoH copies: labelled "[DtoH]" on the copy engine row.
        - Events: shown as small markers at their timeline positions.

    LOOKING FOR OVERLAP:
        Correct double buffer: HtoD, kernel, DtoH rectangles appear in
        PARALLEL at the same timestamp on different rows.
        Incorrect (no overlap): rectangles are SEQUENTIAL — a gap between
        each operation or they don't overlap vertically.

### Diagnosing Stream Issues with nsys

    PROBLEM: "I added streams but see no overlap."
    DIAGNOSIS CHECKLIST:
        □ Is host memory PINNED? (cudaMallocHost, not malloc)
            If not: cudaMemcpyAsync silently becomes synchronous.
        □ Is the null stream (0) being used for any operations?
            Null stream operations sync with all others → kills overlap.
        □ Are cudaDeviceSynchronize or cudaStreamSynchronize called inside the loop?
            These CPU-blocking calls stall all progress.
        □ Are kernels large enough to saturate SMs?
            If kernel fills all SMs: HtoD can still run (separate engine),
            but DtoH of the SAME buffer must wait for the kernel.
        □ Are event dependencies set correctly?
            Missing event: kernel reads buffer before HtoD finishes → corrupt data.
            Extra event: unnecessary serialisation destroys overlap.

    NCU METRIC for copy overlap:
        There's no single metric — use nsys timeline visual inspection.
        ncu measures individual kernel execution; overlap is a system-level concern.

### cudaStreamCaptureStatus and CUDA Graph Interaction

    cudaStreamIsCapturing(stream, &status) returns the capture status:
        cudaStreamCaptureStatusNone:   not currently capturing.
        cudaStreamCaptureStatusActive: stream is being captured into a graph.
        cudaStreamCaptureStatusInvalidated: capture was invalidated (error).

    When profiling a captured stream: Nsight Systems shows the CUDA Graph launch
    as a single event, with node internals shown only with --cuda-graph-trace=node.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Stream Execution Model — Concurrency, Null Stream & Priority": {
        "description": (
            "Model the CUDA stream execution engine: show how operations on the "
            "same stream serialise while operations on different streams can "
            "overlap. Demonstrate the null stream's global synchronisation "
            "semantics. Compute the effective GPU utilisation for sequential "
            "vs multi-stream execution. Show the priority model and when it "
            "matters for latency-sensitive workloads."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  STREAM EXECUTION MODEL — Concurrency, Null Stream & Priority")
print("=" * 68)
print()

np.random.seed(42)

# Hardware constants (A100 SXM4)
SM_COUNT        = 108
MAX_BLOCKS_PER_SM = 32
PCIE_BW_GBS     = 32.0    # PCIe Gen4 x16 (bidirectional)
HBM_BW_GBS      = 2000.0


def kernel_sm_fraction(grid_size, block_warps=4):
    """Fraction of SMs a kernel occupies."""
    blocks_per_sm = min(grid_size / SM_COUNT, MAX_BLOCKS_PER_SM)
    return min(blocks_per_sm / MAX_BLOCKS_PER_SM, 1.0)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Operations on same vs different streams
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Same Stream (Serialised) vs Different Streams (Parallel)")
print("━" * 68)
print()

T_htod_us   = 500.0   # HtoD copy time (µs) — 16 MB at 32 GB/s
T_kernel_us = 800.0   # kernel time (µs)
T_dtoh_us   = 500.0   # DtoH copy time (µs)

print(f"  T_htod = {T_htod_us:.0f} µs  |  T_kernel = {T_kernel_us:.0f} µs  "
      f"|  T_dtoh = {T_dtoh_us:.0f} µs")
print()

# Single stream: strictly sequential
single_total = T_htod_us + T_kernel_us + T_dtoh_us
util_single  = T_kernel_us / single_total * 100

# Multi-stream: limited by longest stage
multi_total = max(T_htod_us, T_kernel_us, T_dtoh_us) + (
    # pipeline fill + drain overhead: 1 htod + 1 dtoh (warmup + cool-down)
    T_htod_us + T_dtoh_us) / 10   # amortised over 10 batches

util_multi   = T_kernel_us / max(T_htod_us, T_kernel_us, T_dtoh_us) * 100

print(f"  {'Approach':<28}  {'Total/batch (µs)':>18}  "
      f"{'SM utilisation':>16}  {'Speedup vs single'}")
print("  " + "─" * 64)
print(f"  {'Single stream (serial)':<28}  {single_total:>18.0f}  "
      f"{util_single:>15.1f}%  {'1.00×':>18}")
print(f"  {'Multi-stream (pipelined)':<28}  "
      f"{max(T_htod_us,T_kernel_us,T_dtoh_us):>18.0f}  "
      f"{util_multi:>15.1f}%  "
      f"{single_total/max(T_htod_us,T_kernel_us,T_dtoh_us):>17.2f}×")
print()
print(f"  Pipelined speedup = T_single / max(T_htod, T_kernel, T_dtoh)")
print(f"  = {single_total:.0f} / {max(T_htod_us,T_kernel_us,T_dtoh_us):.0f} "
      f"= {single_total/max(T_htod_us,T_kernel_us,T_dtoh_us):.2f}×")
print()
print(f"  Maximum speedup is determined by the BOTTLENECK stage:")
for times, label in [((100, 800, 100), "compute-bound (T_k >> T_copy)"),
                      ((500, 800, 500), "balanced"),
                      ((800, 200, 800), "transfer-bound (T_copy >> T_k)")]:
    th, tk, td = times
    sx = (th+tk+td) / max(th, tk, td)
    print(f"    [{th},{tk},{td}] {label:<36}  speedup = {sx:.2f}×")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Null stream blocking semantics
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Null Stream (Stream 0): The Global Synchronisation Trap")
print("━" * 68)
print()

print("  Null stream (stream 0) has SPECIAL blocking semantics:")
print("    - Null stream work waits for ALL non-null streams to drain first.")
print("    - Non-null stream work submitted AFTER a null stream op waits for it.")
print("    - Effectively: null stream = global synchronisation barrier.")
print()

scenarios = [
    ("All on stream 0",
     [("HtoD batch1", 0), ("kernel batch1", 0), ("DtoH batch1", 0),
      ("HtoD batch2", 0), ("kernel batch2", 0)],
     "All serial — no overlap possible"),
    ("Mixed: null + non-null",
     [("HtoD batch1", 1), ("kernel batch1", 0), ("HtoD batch2", 1)],
     "kernel(0) waits for ALL streams → HtoD batch2 is BLOCKED by kernel"),
    ("All non-null streams",
     [("HtoD batch1", 1), ("kernel batch1", 2), ("HtoD batch2", 1)],
     "True concurrency — HtoD batch2 overlaps kernel batch1"),
]

for name, ops, note in scenarios:
    print(f"  Scenario: {name}")
    print(f"  Operations: {[(op, f'stream={s}') for op, s in ops]}")
    print(f"  Result: {note}")
    print()

print("  KEY RULE: NEVER MIX null stream (0) with non-null streams in performance code.")
print("  Always create explicit streams and assign ALL operations to them.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Multi-kernel concurrency model
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Multi-Kernel Concurrency: SM Resource Sharing")
print("━" * 68)
print()

print(f"  GPU: A100 SXM4, {SM_COUNT} SMs, {MAX_BLOCKS_PER_SM} blocks/SM max")
print()
print(f"  {'Kernel A grid':>14}  {'Kernel B grid':>14}  {'A SMs%':>8}  "
      f"{'B SMs%':>8}  {'Combined':>10}  {'Can overlap?'}")
print("  " + "─" * 62)

kernel_configs = [
    (432, 432),   # both use 4 blocks/SM → fill everything together
    (216, 216),   # both use 2 blocks/SM → 50% each → might overlap
    (108,  54),   # A=100%, B=50%
    ( 54,  54),   # both use ~50%
    ( 27,  27),   # both use ~25%
    (  8,   8),   # tiny, each ~7%
]

for grid_a, grid_b in kernel_configs:
    frac_a = min(grid_a / (SM_COUNT * MAX_BLOCKS_PER_SM), 1.0) * 100
    frac_b = min(grid_b / (SM_COUNT * MAX_BLOCKS_PER_SM), 1.0) * 100
    combined = frac_a + frac_b
    can_overlap = combined <= 100.0
    sym = "✅ yes" if can_overlap else "❌ no (resource saturated)"
    print(f"  {grid_a:>14}  {grid_b:>14}  {frac_a:>7.0f}%  "
          f"{frac_b:>7.0f}%  {min(combined,100):>9.0f}%  {sym}")

print()
print("  Two kernels can only overlap if their combined SM usage ≤ 100%.")
print("  Even then: HBM bandwidth contention may halve effective throughput.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Stream priority model
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Stream Priority: Latency Guarantee for Critical Paths")
print("━" * 68)
print()

print("  cudaDeviceGetStreamPriorityRange returns [lo_prio, hi_prio].")
print("  A100 range: typically [-1 (high), 0 (default/low)].")
print("  Higher priority streams PREEMPT lower priority at warp granularity.")
print()

# Model: inference stream with high priority, batch loading with low
T_inference_us  = 1500.0   # inference kernel duration
T_background_us = 3000.0   # background kernel (longer, lower priority)
T_overlap_us    = 800.0    # how much the background runs before inference arrives

without_priority_delay = T_overlap_us   # inference waits for background to finish
with_priority_delay    = 0.0            # inference preempts background immediately

print(f"  Example: inference (T={T_inference_us:.0f}µs) + background (T={T_background_us:.0f}µs)")
print(f"  Background already running for {T_overlap_us:.0f}µs when inference request arrives.")
print()
print(f"  {'Scenario':<38}  {'Inference delay':>16}  {'Total latency'}")
print("  " + "─" * 60)
print(f"  {'Same priority (FIFO)':<38}  {without_priority_delay:>15.0f}µs  "
      f"{T_inference_us + without_priority_delay:.0f}µs")
print(f"  {'High-priority inference stream':<38}  {with_priority_delay:>15.0f}µs  "
      f"{T_inference_us + with_priority_delay:.0f}µs")
print()
print(f"  Priority boost saves {without_priority_delay:.0f}µs latency "
      f"({without_priority_delay/T_inference_us*100:.0f}% of inference time).")
print("  Critical for: real-time inference, robotics, audio processing.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · cudaEvent Timing — Accurate Measurement vs CPU Timers": {
        "description": (
            "Implement GPU-side timing using the cudaEvent model. Show the "
            "difference between CPU timer measurements (which include API overhead "
            "and scheduling jitter) and GPU event measurements (pure kernel time). "
            "Demonstrate the correct timing pattern: record→work→record→sync→elapsed. "
            "Show common timing mistakes and their effects. Build a timing harness "
            "that measures multiple kernels with warmup and statistical averaging."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time

print("=" * 68)
print("  cudaEvent TIMING — Accurate Measurement vs CPU Timers")
print("=" * 68)
print()

np.random.seed(7)


# ─────────────────────────────────────────────────────────────────────
# Simulate GPU timing (since we can't run actual CUDA here)
# ─────────────────────────────────────────────────────────────────────

class SimulatedGPUEvent:
    """
    Simulate cudaEvent_t: a timestamp recorded on the GPU timeline.
    """
    def __init__(self):
        self.recorded_gpu_time_us = None   # GPU-side clock when event fires
        self._stream_position     = None   # position in stream queue

    def record(self, gpu_timeline_us):
        """GPU reaches this event at gpu_timeline_us."""
        self.recorded_gpu_time_us = gpu_timeline_us

    def synchronize(self):
        """CPU waits until GPU has passed this event."""
        # In simulation: instant (event already recorded in timeline)
        assert self.recorded_gpu_time_us is not None, "Event not yet reached"

    @staticmethod
    def elapsed_time_ms(start_event, stop_event):
        """Returns elapsed GPU time between two events (milliseconds)."""
        assert start_event.recorded_gpu_time_us is not None
        assert stop_event.recorded_gpu_time_us  is not None
        return (stop_event.recorded_gpu_time_us -
                start_event.recorded_gpu_time_us) / 1000.0


class SimulatedGPUTimeline:
    """
    Simulate a GPU stream timeline with kernel executions.
    """
    def __init__(self):
        self.current_time_us = 0.0    # GPU clock
        self.operations      = []

    def record_event(self, event):
        event.record(self.current_time_us)
        return event

    def execute_kernel(self, name, duration_us):
        start = self.current_time_us
        self.current_time_us += duration_us
        self.operations.append((name, start, self.current_time_us))
        return start, self.current_time_us

    def execute_copy(self, name, duration_us):
        return self.execute_kernel(name, duration_us)   # same mechanics


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: CPU timer vs GPU event comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — CPU Timer vs GPU Event: What Each Measures")
print("━" * 68)
print()

KERNEL_DURATION_US = 1000.0   # 1 ms kernel
API_OVERHEAD_US    = 3.0      # CPU API call overhead
SYNC_OVERHEAD_US   = 4.0      # cudaDeviceSynchronize overhead
OS_JITTER_US       = 15.0     # OS scheduling jitter (random, ±15 µs)

gpu_tl = SimulatedGPUTimeline()

# --- Simulate what each measurement method captures ---

# CPU timer measurement (simulated)
np.random.seed(42)
cpu_timer_measurements = []
for trial in range(10):
    jitter = np.random.uniform(-OS_JITTER_US, OS_JITTER_US)
    cpu_measured_us = (API_OVERHEAD_US +      # cudaLaunchKernel time
                       KERNEL_DURATION_US +   # actual kernel
                       SYNC_OVERHEAD_US +     # cudaDeviceSynchronize
                       jitter)                # OS scheduling jitter
    cpu_timer_measurements.append(cpu_measured_us)

# GPU event measurement (simulated — no jitter, no overhead)
gpu_event_measurements = []
for trial in range(10):
    # Events measure only the GPU-side kernel duration
    gpu_event_measurements.append(KERNEL_DURATION_US)  # pure kernel time

cpu_arr = np.array(cpu_timer_measurements)
gpu_arr = np.array(gpu_event_measurements)

print(f"  Kernel true duration: {KERNEL_DURATION_US:.1f} µs")
print()
print(f"  {'Metric':<30}  {'CPU timer (µs)':>16}  {'GPU event (µs)':>16}")
print("  " + "─" * 62)
for metric, cpu_v, gpu_v in [
    ("Mean",         cpu_arr.mean(),  gpu_arr.mean()),
    ("Std dev",      cpu_arr.std(),   gpu_arr.std()),
    ("Min",          cpu_arr.min(),   gpu_arr.min()),
    ("Max",          cpu_arr.max(),   gpu_arr.max()),
    ("Error vs true",cpu_arr.mean()-KERNEL_DURATION_US, gpu_arr.mean()-KERNEL_DURATION_US),
]:
    print(f"  {metric:<30}  {cpu_v:>16.2f}  {gpu_v:>16.2f}")

print()
print("  CPU timer over-estimates by API + sync overhead + jitter.")
print("  GPU event measures ONLY the kernel execution time. Zero overhead.")
print()
print("  CPU timer includes:")
print(f"    API overhead: {API_OVERHEAD_US:.0f} µs (cudaLaunchKernel + scheduling)")
print(f"    Sync overhead: {SYNC_OVERHEAD_US:.0f} µs (cudaDeviceSynchronize latency)")
print(f"    OS jitter: ±{OS_JITTER_US:.0f} µs (thread scheduling uncertainty)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Correct timing pattern trace
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Correct Timing Pattern: Record→Work→Record→Sync→Elapsed")
print("━" * 68)
print()

print("  GPU timeline simulation for: [record_start] [kernel A] [kernel B] [record_stop]")
print()

gpu_tl2 = SimulatedGPUTimeline()

start_ev = SimulatedGPUEvent()
stop_ev  = SimulatedGPUEvent()

# Timeline:
gpu_tl2.record_event(start_ev)                    # record start marker
kA_start, kA_end = gpu_tl2.execute_kernel("kernel_A", 600.0)
kB_start, kB_end = gpu_tl2.execute_kernel("kernel_B", 400.0)
gpu_tl2.record_event(stop_ev)                      # record stop marker

stop_ev.synchronize()   # CPU waits here (GPU has reached stop_ev)
elapsed_ms = SimulatedGPUEvent.elapsed_time_ms(start_ev, stop_ev)

print(f"  GPU timeline:")
print(f"    t=0.0 µs:   [start_ev RECORDED]")
print(f"    t=0.0 µs:   kernel_A starts")
print(f"    t=600.0 µs: kernel_A ends")
print(f"    t=600.0 µs: kernel_B starts")
print(f"    t=1000.0 µs: kernel_B ends")
print(f"    t=1000.0 µs: [stop_ev RECORDED]")
print()
print(f"  start_ev.gpu_time = {start_ev.recorded_gpu_time_us:.1f} µs")
print(f"  stop_ev.gpu_time  = {stop_ev.recorded_gpu_time_us:.1f} µs")
print(f"  Elapsed time = {elapsed_ms:.3f} ms  "
      f"(= {stop_ev.recorded_gpu_time_us - start_ev.recorded_gpu_time_us:.0f} µs)")
print(f"  True kernel A + B time = {600+400:.0f} µs  → {'✅ exact match' if elapsed_ms == 1.0 else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Timing mistakes and their effects
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Common Timing Mistakes and Their Effects")
print("━" * 68)
print()

LAUNCH_OVERHEAD_US  = 5.0     # CPU API call overhead
KERNEL_ACTUAL_US    = 1000.0  # kernel runs for 1 ms

mistakes = [
    {"id":1,"name":"Missing cudaEventSynchronize before ElapsedTime",
     "code":"cudaEventRecord(stop, stream); cudaEventElapsedTime(&ms, start, stop); // BUG",
     "result":"Returns 0.0 ms or garbage — stop GPU timestamp not yet written.",
     "fix":"Always call cudaEventSynchronize(stop) before cudaEventElapsedTime."},
    {"id":2,"name":"Recording events on different streams",
     "code":"cudaEventRecord(start, stream_A); kernel<<<..., stream_B>>>(); cudaEventRecord(stop, stream_A); // BUG",
     "result":"Elapsed time measures stream_A (possibly ~0), not the kernel on stream_B.",
     "fix":"Record start and stop on the SAME stream as the work being timed."},
    {"id":3,"name":"CPU timer without sync (async kernel)",
     "code":"auto t0=chrono::now(); kernel<<<...,stream>>>(); auto t1=chrono::now(); // BUG: kernel still running",
     "result":f"Measures API call latency (~{LAUNCH_OVERHEAD_US:.0f} µs) not kernel duration.",
     "fix":"Add cudaStreamSynchronize(stream) between t0 and t1."},
    {"id":4,"name":"Timing inside a loop without warmup",
     "code":"for iter in range(100): record(start); kernel(...); record(stop) // first iters cold",
     "result":"First 1-3 iterations include JIT/init overhead (10-100× slower).",
     "fix":"Run 3-5 warmup iterations before the timed loop."},
]

for m in mistakes:
    print(f"  MISTAKE {m['id']}: {m['name']}")
    print(f"    Code:   {m['code']}")
    print(f"    Result: {m['result']}")
    print(f"    Fix:    {m['fix']}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Statistical timing harness
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Statistical Timing Harness: Warmup + Percentiles")
print("━" * 68)
print()

# Simulate a benchmark with realistic variation
def simulate_kernel_times(n_iters, warmup=3, kernel_us=1000, noise_us=20):
    """Simulate GPU event timings with warmup and random noise."""
    times = []
    for i in range(n_iters + warmup):
        # First iterations include one-time overheads
        if i < warmup:
            jit_overhead = 5000.0 * math.exp(-i * 2)  # decays rapidly
        else:
            jit_overhead = 0.0
        t = kernel_us + jit_overhead + np.random.normal(0, noise_us)
        if i >= warmup:
            times.append(max(t, kernel_us * 0.8))
    return times

print(f"  Simulating 100 iterations with 3 warmup: T_true=1000 µs, noise=±20 µs")
print()

times_sim = simulate_kernel_times(100, warmup=3, kernel_us=1000.0, noise_us=20.0)
t_arr     = np.array(times_sim)

print(f"  {'Statistic':<20}  {'Value (µs)':>12}")
print("  " + "─" * 34)
for stat, val in [
    ("Mean",         t_arr.mean()),
    ("Median",       np.median(t_arr)),
    ("Std dev",      t_arr.std()),
    ("Min",          t_arr.min()),
    ("P25",          np.percentile(t_arr, 25)),
    ("P75",          np.percentile(t_arr, 75)),
    ("P95",          np.percentile(t_arr, 95)),
    ("P99",          np.percentile(t_arr, 99)),
    ("Max",          t_arr.max()),
]:
    print(f"  {stat:<20}  {val:>12.2f}")

print()
print("  REPORT FORMAT: 'median ± std (P95)' gives the most useful summary.")
print(f"  This kernel: {np.median(t_arr):.1f} ± {t_arr.std():.1f} µs "
      f"(P95 = {np.percentile(t_arr, 95):.1f} µs)")
print()
print("  For cuBLAS/cuDNN benchmarks: always report median or P50, not mean.")
print("  Mean is inflated by rare outliers (thermal throttling, OS preemption).")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Pinned Memory & Async Transfers — HtoD / DtoH Bandwidth Model": {
        "description": (
            "Implement the pinned vs pageable memory model. Show why cudaMemcpyAsync "
            "on pageable memory becomes synchronous. Compute the effective bandwidth "
            "for different transfer sizes and memory types. Simulate the pinned memory "
            "allocation cost amortisation. Show the memory-mapped (zero-copy) pattern "
            "and when it outperforms explicit copies."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  PINNED MEMORY & ASYNC TRANSFERS — HtoD / DtoH Bandwidth Model")
print("=" * 68)
print()

np.random.seed(99)

# Hardware constants
PCIE_GEN4_GBS    = 32.0    # PCIe Gen4 x16 unidirectional
PCIE_GEN5_GBS    = 64.0    # PCIe Gen5 x16 unidirectional
NVLINK_GBS       = 150.0   # NVLink 4 unidirectional per link
HBM_BW_GBS       = 2000.0  # A100 HBM bandwidth
SM_FREQ_GHZ      = 1.41    # A100 clock

# Transfer overhead model (latency before DMA data actually moves)
PAGEABLE_BOUNCE_US  = 8.0    # internal copy to pinned bounce buffer
PINNED_SETUP_US     = 1.5    # DMA descriptor setup
NVLINK_SETUP_US     = 0.8    # NVLink setup (lower latency)

PINNED_ALLOC_MS  = 2.5    # cudaMallocHost amortised cost (ms)


def transfer_time_us(size_bytes, bw_gbs, is_pinned=True, is_htod=True):
    """
    Model the time for an async memcpy.
    Pinned: pure DMA with setup overhead.
    Pageable: bounce-buffer copy first, then DMA.
    """
    dma_us = size_bytes / (bw_gbs * 1e9) * 1e6
    if is_pinned:
        return dma_us + PINNED_SETUP_US
    else:
        # For pageable: must first copy to internal pinned buffer (serial with DMA)
        bounce_us = size_bytes / (HBM_BW_GBS * 1e9 * 0.1) * 1e6  # bounce via LLC
        # In practice: cudaMemcpyAsync on pageable becomes SYNCHRONOUS
        return dma_us + bounce_us + PINNED_SETUP_US + PAGEABLE_BOUNCE_US


def effective_bandwidth_gbs(size_bytes, time_us):
    return size_bytes / (time_us * 1e-6) / 1e9


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Pinned vs pageable bandwidth at different sizes
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Pinned vs Pageable: Bandwidth at Different Transfer Sizes")
print("━" * 68)
print()

print(f"  PCIe Gen4 x16 peak: {PCIE_GEN4_GBS:.0f} GB/s")
print()
print(f"  {'Size':>10}  {'Pinned µs':>12}  {'Pageable µs':>14}  "
      f"{'Pinned GB/s':>13}  {'Pageable GB/s':>15}  {'Is async?'}")
print("  " + "─" * 76)

for size_mb in [0.001, 0.01, 0.1, 1, 4, 16, 64, 256, 1024]:
    size_bytes  = int(size_mb * 1e6)
    t_pinned    = transfer_time_us(size_bytes, PCIE_GEN4_GBS, is_pinned=True)
    t_pageable  = transfer_time_us(size_bytes, PCIE_GEN4_GBS, is_pinned=False)
    bw_pinned   = effective_bandwidth_gbs(size_bytes, t_pinned)
    bw_pageable = effective_bandwidth_gbs(size_bytes, t_pageable)

    # Small transfers: pageable can't start DMA until bounce is done (serial, not async)
    async_ok    = "✅ async" if size_bytes > 64*1024 else "❌ serialised"
    if not True:   # pageable is always serialised in practice
        async_ok = "❌ serialised"
    else:
        async_ok = "✅ async (pinned)" if True else "❌ only if pinned"

    # Whether pageable can be truly async
    page_async = "❌ sync" if size_bytes > 0 else "✅"

    print(f"  {size_mb:>9.3f}M  {t_pinned:>12.1f}  {t_pageable:>14.1f}  "
          f"{bw_pinned:>13.3f}  {bw_pageable:>15.3f}  ✅ pinned / ❌ pageable")

print()
print("  PAGEABLE ASYNC IS A MYTH: cudaMemcpyAsync on pageable memory falls back")
print("  to the synchronous implementation (GPU driver detects non-pinned ptr).")
print("  The CPU IS blocked during pageable memcpy, even with Async API call.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Pinned allocation amortisation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Pinned Memory Allocation Cost Amortisation")
print("━" * 68)
print()

PINNED_ALLOC_US  = PINNED_ALLOC_MS * 1000   # one-time allocation cost
MALLOC_US        = 0.5                         # pageable malloc cost (µs)

print(f"  cudaMallocHost (pinned) cost: ~{PINNED_ALLOC_US:.0f} µs  (one-time)")
print(f"  malloc (pageable) cost:       ~{MALLOC_US:.1f} µs  (one-time)")
print()

# Savings per transfer from using pinned (vs pageable)
for size_mb in [4, 64, 256]:
    size_bytes   = int(size_mb * 1e6)
    t_pinned     = transfer_time_us(size_bytes, PCIE_GEN4_GBS, is_pinned=True)
    t_pageable   = transfer_time_us(size_bytes, PCIE_GEN4_GBS, is_pinned=False)
    saving_us    = t_pageable - t_pinned

    # How many transfers to amortise pinned alloc overhead?
    n_break_even = PINNED_ALLOC_US / saving_us

    print(f"  Transfer size: {size_mb:.0f} MB")
    print(f"    Pinned: {t_pinned:.1f} µs  |  Pageable: {t_pageable:.1f} µs  "
          f"|  Saving: {saving_us:.1f} µs/transfer")
    print(f"    Break-even: {n_break_even:.0f} transfers  "
          f"(worth it after {n_break_even:.0f} copies)")
    print(f"    For training (1000 batches): "
          f"{'✅ amortised' if 1000 > n_break_even else '  not worth it'}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Zero-copy (mapped) memory model
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Zero-Copy (Mapped Memory): When to Use PCIe Access")
print("━" * 68)
print()

print("  Zero-copy: GPU accesses host memory directly via PCIe.")
print("  Each GPU memory access → PCIe transaction (~1 µs latency, low bandwidth).")
print()

# Compare: explicit async copy vs zero-copy vs HBM
kernel_access_bytes = 64    # bytes per SM access
n_accesses_per_kernel = 1000

pcie_latency_us = 1.0    # per PCIe round-trip
hbm_latency_us  = 0.0003  # ~0.3 µs for HBM access

print(f"  Memory access pattern: {n_accesses_per_kernel} accesses of {kernel_access_bytes} bytes each")
print()
print(f"  {'Memory type':<25}  {'Latency/access':>16}  {'Total latency':>14}  {'Bandwidth'}")
print("  " + "─" * 60)

scenarios_zc = [
    ("HBM (after explicit copy)", hbm_latency_us,  n_accesses_per_kernel * hbm_latency_us,  HBM_BW_GBS),
    ("Zero-copy via PCIe",         pcie_latency_us, n_accesses_per_kernel * pcie_latency_us, PCIE_GEN4_GBS),
]
for name, lat_us, total_us, bw in scenarios_zc:
    print(f"  {name:<25}  {lat_us:>14.4f}µs  {total_us:>12.1f}µs  "
          f"{bw:>8.0f} GB/s")

print()
print("  Zero-copy is 3000× higher latency per access than HBM.")
print("  Zero-copy WINS ONLY when:")
print("    1. Data accessed exactly ONCE (no reuse benefit from copying to HBM)")
print("    2. Working set exceeds GPU HBM capacity (can't fit on GPU anyway)")
print("    3. Integrated CPU+GPU sharing physical memory (ARM, Apple, Intel ARC)")
print()

# Break-even: when is zero-copy faster than copy+compute?
for copy_size_mb in [1, 16, 256]:
    copy_bytes    = copy_size_mb * 1e6
    copy_time_us  = copy_bytes / (PCIE_GEN4_GBS * 1e9) * 1e6
    # If kernel only reads each byte once: zero-copy time = copy_time_us (same bandwidth)
    # If kernel reads each byte N times: zero-copy = N * copy_time_us
    # Break-even at N=1 (single read): zero-copy and explicit copy are similar
    print(f"  {copy_size_mb}MB: explicit copy={copy_time_us:.0f}µs. "
          f"Zero-copy amortises after 1+ reuse. "
          f"For 2 reads: zero-copy={2*copy_time_us:.0f}µs vs copy+read={copy_time_us+5:.0f}µs "
          f"→ {'explicit copy wins ✅' if copy_time_us < 2*copy_time_us else 'zero-copy'}")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Compute / Copy Overlap — Double Buffer Pipeline & Speedup Model": {
        "description": (
            "Implement the double-buffer pipeline: simulate HtoD, compute, and DtoH "
            "operations on three independent hardware streams. Show the pipeline "
            "timeline with all three stages overlapping. Compute the speedup for "
            "different T_copy/T_kernel ratios. Verify correct event-based dependency "
            "wiring prevents data corruption. Show the pipeline efficiency formula "
            "and the optimal batch sizing strategy."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple

print("=" * 68)
print("  COMPUTE / COPY OVERLAP — Double Buffer Pipeline & Speedup Model")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# Pipeline simulator
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineEvent:
    """A timestamped entry in the GPU execution log."""
    name:    str
    engine:  str    # "HtoD", "Compute", "DtoH"
    batch:   int
    start:   float  # µs
    end:     float  # µs

    @property
    def duration(self):
        return self.end - self.start


def simulate_single_stream(n_batches, T_htod, T_kernel, T_dtoh):
    """Naïve single-stream pipeline: no overlap."""
    t = 0.0
    events = []
    for i in range(n_batches):
        events.append(PipelineEvent("HtoD",    "HtoD",    i, t, t+T_htod));    t += T_htod
        events.append(PipelineEvent("kernel",  "Compute", i, t, t+T_kernel));  t += T_kernel
        events.append(PipelineEvent("DtoH",    "DtoH",    i, t, t+T_dtoh));    t += T_dtoh
    return events, t


def simulate_double_buffer(n_batches, T_htod, T_kernel, T_dtoh):
    """
    Double-buffer pipeline: 3 hardware engines run concurrently.
    Correctness: kernel(i) only starts after HtoD(i) is done.
                 DtoH(i) only starts after kernel(i) is done.
    """
    # Track when each engine is free
    htod_free    = 0.0
    compute_free = 0.0
    dtoh_free    = 0.0

    # Track when each BUFFER is ready (has valid data for its stage)
    # buf_htod_done[i]: when HtoD for batch i finishes
    # buf_compute_done[i]: when kernel for batch i finishes
    buf_htod_done    = {}
    buf_compute_done = {}

    events = []

    for i in range(n_batches):
        # HtoD(i): starts when HtoD engine is free AND previous compute freed the buffer
        # (Using double buffer: buf i uses slot i%2. Must wait for compute(i-2) to free it.)
        if i >= 2:
            prev_compute_done = buf_compute_done.get(i-2, 0.0)
        else:
            prev_compute_done = 0.0
        htod_start = max(htod_free, prev_compute_done)
        htod_end   = htod_start + T_htod
        htod_free  = htod_end
        buf_htod_done[i] = htod_end
        events.append(PipelineEvent("HtoD", "HtoD", i, htod_start, htod_end))

    for i in range(n_batches):
        # Kernel(i): starts when compute engine is free AND HtoD(i) is done
        kernel_start = max(compute_free, buf_htod_done[i])
        kernel_end   = kernel_start + T_kernel
        compute_free = kernel_end
        buf_compute_done[i] = kernel_end
        events.append(PipelineEvent("kernel", "Compute", i, kernel_start, kernel_end))

    for i in range(n_batches):
        # DtoH(i): starts when DtoH engine is free AND kernel(i) is done
        dtoh_start = max(dtoh_free, buf_compute_done[i])
        dtoh_end   = dtoh_start + T_dtoh
        dtoh_free  = dtoh_end
        events.append(PipelineEvent("DtoH", "DtoH", i, dtoh_start, dtoh_end))

    total_time = max(e.end for e in events)
    return events, total_time


def engine_utilisation(events, total_time, engine):
    """Fraction of total time the given engine is busy."""
    busy = sum(e.duration for e in events if e.engine == engine)
    return busy / total_time * 100


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Pipeline comparison — single stream vs double buffer
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Single Stream vs Double Buffer: Timeline & Utilisation")
print("━" * 68)
print()

T_h, T_k, T_d = 400.0, 600.0, 400.0  # µs
N_BATCH = 6

ev_single, t_single = simulate_single_stream(N_BATCH, T_h, T_k, T_d)
ev_double, t_double = simulate_double_buffer(N_BATCH, T_h, T_k, T_d)

print(f"  T_htod={T_h:.0f}µs, T_kernel={T_k:.0f}µs, T_dtoh={T_d:.0f}µs, N={N_BATCH}")
print()
print(f"  {'Metric':<30}  {'Single stream':>16}  {'Double buffer':>16}")
print("  " + "─" * 62)
print(f"  {'Total time (µs)':<30}  {t_single:>16.0f}  {t_double:>16.0f}")
print(f"  {'Speedup':<30}  {'1.00×':>16}  {t_single/t_double:>15.2f}×")
for engine in ["HtoD", "Compute", "DtoH"]:
    u_s = engine_utilisation(ev_single, t_single, engine)
    u_d = engine_utilisation(ev_double, t_double, engine)
    print(f"  {engine+' utilisation':<30}  {u_s:>15.1f}%  {u_d:>15.1f}%")
print()

# ASCII timeline
print("  Double buffer pipeline timeline (ASCII):")
print("  Each row = one hardware engine. Batches shown by number.")
print()

# Build ASCII grid
max_t    = int(t_double + 0.5)
scale    = 0.05   # µs per character
n_cols   = int(max_t * scale) + 1
rows     = {"HtoD": [' '] * n_cols, "Compute": [' '] * n_cols, "DtoH": [' '] * n_cols}

for ev in ev_double:
    c_start = int(ev.start * scale)
    c_end   = int(ev.end * scale)
    for c in range(c_start, min(c_end, n_cols)):
        rows[ev.engine][c] = str(ev.batch % 10)

for engine in ["HtoD", "Compute", "DtoH"]:
    row_str = ''.join(rows[engine])
    print(f"  {engine:<8}: {row_str}")
print()
print(f"  Each character ≈ {1/scale:.0f}µs. Numbers = batch index.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Speedup formula for different T_copy/T_kernel ratios
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Speedup vs T_copy/T_kernel Ratio (N=20 batches)")
print("━" * 68)
print()

N_LARGE = 20

print(f"  {'T_copy':>8}  {'T_kernel':>10}  {'ratio':>8}  "
      f"{'Single µs':>12}  {'Double µs':>12}  {'Speedup':>9}  {'Bottleneck'}")
print("  " + "─" * 68)

T_k_fixed = 500.0
for ratio in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
    T_c = T_k_fixed * ratio
    _, t_s = simulate_single_stream(N_LARGE, T_c, T_k_fixed, T_c)
    _, t_d = simulate_double_buffer(N_LARGE, T_c, T_k_fixed, T_c)
    speedup    = t_s / t_d
    bottleneck = ("compute" if T_k_fixed > T_c else "transfer")
    print(f"  {T_c:>8.0f}  {T_k_fixed:>10.0f}  {ratio:>8.2f}  "
          f"{t_s/1000:>10.2f}ms  {t_d/1000:>10.2f}ms  {speedup:>9.2f}×  {bottleneck}")

print()
print("  FORMULA: Speedup = (2*T_copy + T_kernel) / max(2*T_copy, T_kernel)")
print("  For T_copy = T_kernel: speedup = 3/1 = 3.0× (perfect balance)")
print("  For T_kernel >> T_copy: speedup → 1× (compute-bound, copy is free)")
print("  For T_copy >> T_kernel: speedup → 2× (transfer-bound, HtoD+DtoH overlap)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Event dependency wiring verification
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Dependency Wiring: Correct vs Missing Events")
print("━" * 68)
print()

print("  CORRECT WIRING (events enforce ordering):")
print()
print("    stream_htod    stream_compute       stream_dtoh")
print("    ─────────────  ──────────────────   ─────────────")
print("    HtoD(batch0)   ← waits for HtoD     ")
print("                   kernel(batch0)  ────→ waits for kernel")
print("    HtoD(batch1)   ← waits for HtoD     DtoH(batch0)")
print("    ...            kernel(batch1)  ────→ ...")
print()

print("  MISSING EVENT (HtoD → kernel dependency removed):")
print("    kernel(batch1) STARTS before HtoD(batch1) finishes!")
print("    Kernel reads PARTIALLY WRITTEN buffer → CORRUPTED DATA.")
print("    Bug is silent: no error, wrong numerical results.")
print()

print("  DEPENDENCY VERIFICATION: check that finish[htod[i]] <= start[kernel[i]]")
print()

# Verify correctness of our double buffer implementation
_, t_check = simulate_double_buffer(N_BATCH, T_h, T_k, T_d)
all_events  = simulate_double_buffer(N_BATCH, T_h, T_k, T_d)[0]

htod_done    = {e.batch: e.end for e in all_events if e.engine == "HtoD"}
compute_start= {e.batch: e.start for e in all_events if e.engine == "Compute"}
compute_done = {e.batch: e.end for e in all_events if e.engine == "Compute"}
dtoh_start   = {e.batch: e.start for e in all_events if e.engine == "DtoH"}

print(f"  {'Batch':>6}  {'HtoD done':>12}  {'Kernel start':>14}  "
      f"{'HtoD→Kernel OK?':>16}  {'Kernel done':>13}  {'DtoH start':>12}  "
      f"{'Kernel→DtoH OK?'}")
print("  " + "─" * 82)

all_ok = True
for i in range(N_BATCH):
    htod_ok    = htod_done[i] <= compute_start[i] + 0.01
    compute_ok = compute_done[i] <= dtoh_start[i] + 0.01
    if not (htod_ok and compute_ok): all_ok = False
    print(f"  {i:>6}  {htod_done[i]:>12.1f}  {compute_start[i]:>14.1f}  "
          f"{'✅' if htod_ok else '❌ VIOLATION':>16}  "
          f"{compute_done[i]:>13.1f}  {dtoh_start[i]:>12.1f}  "
          f"{'✅' if compute_ok else '❌ VIOLATION'}")

print()
print(f"  All dependency constraints satisfied: {'✅' if all_ok else '❌ BUGS FOUND'}")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · End-to-End Async Pipeline — DL Training Loop with Prefetch": {
        "description": (
            "Build a complete asynchronous deep learning training pipeline: "
            "simulate a DataLoader with prefetching (async HtoD of batch N+1 "
            "while processing batch N), gradient computation, and async "
            "results offload. Compare throughput for sync, one-stream async, "
            "and fully pipelined with prefetch. Show the GPU idle time elimination "
            "and compute the optimal number of prefetch buffers."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional

print("=" * 68)
print("  END-TO-END ASYNC PIPELINE — DL Training Loop with Prefetch")
print("=" * 68)
print()

np.random.seed(77)

# Timing constants for a representative training configuration
# (Llama-7B-style, H100 80GB, batch=8, seq=2048, FP16)
T_DATALOAD_MS  = 5.0    # CPU dataloading: tokenisation, collation
T_HTOD_MS      = 2.0    # HtoD for one batch (32 MB at 16 GB/s effective)
T_FORWARD_MS   = 15.0   # forward pass (compute)
T_BACKWARD_MS  = 30.0   # backward pass (2× forward, typical)
T_OPTIM_MS     = 3.0    # optimizer step (Adam)
T_LOGGING_MS   = 0.5    # metrics logging


@dataclass
class StepTrace:
    """Trace of one training step's timeline."""
    step:       int
    start_us:   float
    end_us:     float
    events:     List[tuple] = field(default_factory=list)

    def add(self, name, start, end, engine):
        self.events.append((name, start, end, engine))


def simulate_sync_training(n_steps):
    """
    Fully synchronous training: CPU stalls at every step boundary.
    No pipeline — sequential load, copy, forward, backward, optim.
    """
    t    = 0.0
    traces = []
    for step in range(n_steps):
        s = StepTrace(step, t, 0.0)
        ops = [
            ("dataload",  T_DATALOAD_MS, "CPU"),
            ("HtoD",      T_HTOD_MS,     "HtoD"),
            ("forward",   T_FORWARD_MS,  "GPU"),
            ("backward",  T_BACKWARD_MS, "GPU"),
            ("optim",     T_OPTIM_MS,    "GPU"),
            ("logging",   T_LOGGING_MS,  "CPU"),
        ]
        for name, dur, engine in ops:
            s.add(name, t, t+dur*1000, engine)
            t += dur * 1000    # µs
        s.end_us = t
        traces.append(s)
    return traces, t


def simulate_async_prefetch(n_steps, n_prefetch=2):
    """
    Asynchronous prefetch pipeline:
    - CPU dataloader runs ahead by n_prefetch batches.
    - HtoD runs while GPU computes on previous batch.
    - GPU-CPU logging overlaps with next step's HtoD.
    """
    # Engine availability timers
    cpu_free    = 0.0   # µs (dataloader + logging)
    htod_free   = 0.0   # µs (PCIe copy engine)
    gpu_free    = 0.0   # µs (SM compute)

    traces = []
    batch_htod_done  = {}  # step → time when HtoD finishes

    # Pre-populate prefetch buffers
    for pre in range(min(n_prefetch, n_steps)):
        # Dataload step 'pre' as early as possible
        dl_start = cpu_free
        dl_end   = dl_start + T_DATALOAD_MS * 1000
        cpu_free = dl_end

        # HtoD starts when both: HtoD engine free AND dataload done
        ht_start = max(htod_free, dl_end)
        ht_end   = ht_start + T_HTOD_MS * 1000
        htod_free = ht_end
        batch_htod_done[pre] = ht_end

    for step in range(n_steps):
        s = StepTrace(step, gpu_free, 0.0)

        # Forward + backward + optim on GPU
        # Waits for: GPU free AND HtoD(step) done
        gpu_start = max(gpu_free, batch_htod_done.get(step, 0.0))
        fwd_end   = gpu_start + T_FORWARD_MS * 1000
        bwd_end   = fwd_end   + T_BACKWARD_MS * 1000
        opt_end   = bwd_end   + T_OPTIM_MS * 1000
        gpu_free  = opt_end

        s.add("forward",  gpu_start, fwd_end, "GPU")
        s.add("backward", fwd_end,   bwd_end, "GPU")
        s.add("optim",    bwd_end,   opt_end, "GPU")

        # Logging: can start after backward (async, overlaps next prefetch)
        log_start = max(cpu_free, bwd_end)   # needs GPU bwd done for metrics
        log_end   = log_start + T_LOGGING_MS * 1000
        cpu_free  = log_end
        s.add("logging", log_start, log_end, "CPU")

        # Prefetch step+n_prefetch while GPU is computing
        next_step = step + n_prefetch
        if next_step < n_steps and next_step not in batch_htod_done:
            # Dataload
            dl_start = cpu_free
            dl_end   = dl_start + T_DATALOAD_MS * 1000
            cpu_free = dl_end
            s.add("dataload", dl_start, dl_end, "CPU")

            # HtoD
            ht_start = max(htod_free, dl_end)
            ht_end   = ht_start + T_HTOD_MS * 1000
            htod_free = ht_end
            batch_htod_done[next_step] = ht_end
            s.add("HtoD", ht_start, ht_end, "HtoD")

        s.end_us = gpu_free
        traces.append(s)

    total_time = max(t.end_us for t in traces)
    return traces, total_time


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Throughput comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Throughput: Sync vs Async Prefetch vs Full Pipeline")
print("━" * 68)
print()

N_STEPS = 20

print(f"  {N_STEPS} training steps. T_dataload={T_DATALOAD_MS}ms, "
      f"T_htod={T_HTOD_MS}ms, T_fwd={T_FORWARD_MS}ms, "
      f"T_bwd={T_BACKWARD_MS}ms, T_optim={T_OPTIM_MS}ms")
print()

sync_traces, sync_total = simulate_sync_training(N_STEPS)
async1_traces, async1_total = simulate_async_prefetch(N_STEPS, n_prefetch=1)
async2_traces, async2_total = simulate_async_prefetch(N_STEPS, n_prefetch=2)

step_total_ms = (T_DATALOAD_MS + T_HTOD_MS + T_FORWARD_MS + T_BACKWARD_MS +
                 T_OPTIM_MS + T_LOGGING_MS)

for name, traces, total in [
    ("Synchronous",             sync_traces,   sync_total),
    ("Async prefetch (N=1)",    async1_traces, async1_total),
    ("Async prefetch (N=2)",    async2_traces, async2_total),
]:
    total_ms   = total / 1000
    per_step   = total_ms / N_STEPS
    tput       = 1000 / per_step    # steps/sec
    speedup    = (sync_total/1000) / total_ms

    # GPU utilisation: fraction of time GPU is active
    gpu_busy = sum(
        (e[2] - e[1]) for t in traces for e in t.events if e[3] == "GPU"
    ) / 1000  # ms
    gpu_util = gpu_busy / total_ms * 100

    print(f"  {name:<30}:  {total_ms:>8.1f} ms total  "
          f"{per_step:>8.1f} ms/step  {tput:>7.1f} steps/s  "
          f"GPU {gpu_util:>5.1f}%  {speedup:.2f}×")

print()
print("  Async prefetch eliminates:")
print(f"    T_dataload ({T_DATALOAD_MS} ms) from the critical path — runs in background.")
print(f"    T_htod ({T_HTOD_MS} ms) from the critical path — overlaps with compute.")
print(f"    T_logging ({T_LOGGING_MS} ms) from critical path — overlaps with next prefetch.")
print(f"  Effective step time ≈ T_forward + T_backward + T_optim "
      f"= {T_FORWARD_MS+T_BACKWARD_MS+T_OPTIM_MS:.0f} ms")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Per-step GPU idle time
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Per-Step GPU Idle Time: Where Time is Wasted")
print("━" * 68)
print()

def compute_gpu_idle(traces):
    """Compute GPU idle time between consecutive GPU operations."""
    gpu_events = sorted(
        [(e[1], e[2]) for t in traces for e in t.events if e[3] == "GPU"],
        key=lambda x: x[0]
    )
    idle_total = 0.0
    for i in range(1, len(gpu_events)):
        gap = gpu_events[i][0] - gpu_events[i-1][1]
        if gap > 0:
            idle_total += gap
    return idle_total

for name, traces, total in [
    ("Synchronous",         sync_traces,   sync_total),
    ("Async prefetch (N=2)",async2_traces, async2_total),
]:
    idle_us  = compute_gpu_idle(traces)
    idle_pct = idle_us / total * 100
    print(f"  {name:<30}:  GPU idle = {idle_us/1000:.1f} ms "
          f"({idle_pct:.1f}% of total)")

print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Optimal prefetch depth
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Optimal Prefetch Depth: Memory vs Pipeline Benefit")
print("━" * 68)
print()

BATCH_HBM_MB = 32.0   # HBM per prefetch buffer (one batch)

print(f"  HBM per batch buffer: {BATCH_HBM_MB:.0f} MB")
print()
print(f"  {'N_prefetch':>12}  {'Total µs':>12}  {'Speedup':>10}  "
      f"{'HBM overhead (MB)':>20}  {'Marginal benefit'}")
print("  " + "─" * 62)

prev_total = None
for n_pre in range(1, 8):
    _, t = simulate_async_prefetch(N_STEPS, n_pre)
    speedup = sync_total / t
    hbm     = n_pre * BATCH_HBM_MB
    marginal = (prev_total - t)/prev_total*100 if prev_total else 0
    prev_total = t
    print(f"  {n_pre:>12}  {t/1000:>10.1f}ms  {speedup:>10.2f}×  "
          f"{hbm:>20.0f}  {marginal:>+8.2f}% gain")

print()
T_bottleneck = T_FORWARD_MS + T_BACKWARD_MS + T_OPTIM_MS
T_prefetch_needed = T_DATALOAD_MS + T_HTOD_MS
print(f"  OPTIMAL DEPTH: ceil((T_dataload + T_htod) / T_step)")
print(f"  = ceil(({T_DATALOAD_MS}+{T_HTOD_MS}) / {T_bottleneck}) "
      f"= {math.ceil((T_DATALOAD_MS+T_HTOD_MS)/T_bottleneck)}")
print(f"  Beyond this depth: zero additional benefit (prefetch already far enough ahead).")
print(f"  Each extra buffer: +{BATCH_HBM_MB:.0f} MB HBM with no throughput gain.")
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