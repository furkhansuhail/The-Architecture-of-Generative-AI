"""
Nsight Systems — Timeline Profiling, CPU/GPU Overlap & Bottleneck Analysis
===========================================================================

Nsight Compute tells you what is happening inside a single kernel.
Nsight Systems tells you what is happening across your entire program.

The distinction is critical. A kernel that runs at 95% of the HBM roof
is useless if the GPU sits idle for 80% of wall-clock time because the
CPU is building the next batch, or because a D2H memory copy is blocking
the next kernel launch, or because streams are serialised through an
unnecessary synchronisation barrier. These system-level inefficiencies
are invisible to Nsight Compute — they only appear on a timeline.

Nsight Systems (nsys) is NVIDIA's system-wide timeline profiler. It
records: every CUDA API call (with CPU timestamps), every kernel launch
(with GPU start/stop times), every memory copy (direction, size, stream,
duration), every NVTX annotation, CPU thread activity, and multi-GPU
NVLink traffic — all correlated on a single wall-clock timeline.

Reading an nsys timeline requires three interlocking mental models:

    1. THE TIMELINE MODEL — understanding the rows, the gaps, and what
       "idle" means at the CPU level vs the GPU level.
    2. OVERLAP & PIPELINING — how streams, events, and async transfers
       interact; how to achieve CPU/GPU overlap and copy/compute overlap.
    3. BOTTLENECK PATTERNS — the six canonical system-level bottlenecks
       that nsys reveals, and the specific API changes that fix each one.

The payoff: engineers routinely achieve 3–10× end-to-end throughput
improvements from nsys analysis alone, before touching a single kernel.
The kernel was never the bottleneck — the pipeline was.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Nsight Systems — Timeline Profiling, CPU/GPU Overlap & Bottleneck Analysis"
DISPLAY_NAME = "06a · Nsight Systems"
ICON         = "🔭"
SUBTITLE     = "Timeline · CPU/GPU Overlap · Stream Pipelining · Bottleneck Patterns"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — NSIGHT SYSTEMS vs NSIGHT COMPUTE: CHOOSING THE RIGHT TOOL

### The Two-Level Profiling Stack

    GPU performance problems live at two distinct levels:

    SYSTEM LEVEL (Nsight Systems):
        - Is the GPU utilised over the full training/inference run?
        - Are CPU and GPU overlapping, or serialised?
        - Are memory copies blocking kernel execution?
        - Are streams being used effectively?
        - Which operations dominate wall-clock time?
        Questions about TIME and PIPELINE STRUCTURE.

    KERNEL LEVEL (Nsight Compute):
        - Is this specific kernel memory-bound or compute-bound?
        - What is the L2 hit rate and coalescing efficiency?
        - Why are warps stalling?
        Questions about EFFICIENCY WITHIN a single kernel launch.

    CORRECT WORKFLOW (always in this order):
        1. Run nsys — identify which kernels/copies dominate wall time.
        2. Pick the top bottleneck from the nsys timeline.
        3. Run ncu ON THAT SPECIFIC KERNEL — understand why it is slow.
        4. Fix the kernel. Verify with ncu. Re-run nsys to confirm system gain.

    Skipping nsys and going straight to ncu is the most common profiling
    mistake. You might spend hours optimising a kernel that contributes
    2% of wall time while a H2D copy consuming 40% of time goes unnoticed.

### What nsys Records

    CUDA API TRACE:
        Every call to the CUDA runtime/driver API on the CPU thread:
        cudaLaunchKernel, cudaMalloc, cudaMemcpyAsync, cudaStreamSynchronize,
        cudaEventRecord, etc. Recorded with CPU start/end timestamps.
        Overhead: ~1 µs per API call monitoring.

    CUDA GPU TRACE:
        Every kernel execution and memory operation on the GPU:
        kernel name, stream ID, grid/block dimensions, GPU start/stop.
        Memory copies: direction (H2D/D2H/D2D/P2P), size, stream, duration.
        Recorded via hardware GPU timestamps (nanosecond resolution).

    NVTX RANGES:
        User-annotated regions that appear as coloured bars on the timeline.
        Zero overhead when profiling is off (no-op stubs).
        Essential for correlating Python code regions with GPU activity.

    CPU SAMPLING (optional):
        Statistical CPU callstack samples at ~1 kHz.
        Identifies which Python/C++ function is consuming CPU time.
        Critical for identifying CPU-bound preprocessing bottlenecks.

    OS RUNTIME:
        Thread creation/deletion, mutex locks, condition waits.
        Helps diagnose GIL contention in Python multi-threaded dataloaders.

    NVLINK / PCIe (multi-GPU):
        Link utilisation, bandwidth, direction per link.
        For identifying inter-GPU communication bottlenecks in DDP training.

### nsys CLI Reference

    Basic profiling:
        nsys profile -o profile_name python train.py

    With all CUDA APIs traced:
        nsys profile --trace=cuda,nvtx,cudnn,cublas -o out python train.py

    Limit to specific duration (skip startup):
        nsys profile --delay=10 --duration=30 -o out python train.py

    GPU metrics (requires root or perf_event_paranoid setting):
        nsys profile --gpu-metrics-device=0 -o out python train.py

    Export to SQLite for programmatic analysis:
        nsys export --type=sqlite -o out.sqlite out.nsys-rep

    Generate text statistics report:
        nsys stats out.nsys-rep

    Profile specific CUDA context only:
        nsys profile --capture-range=cudaProfilerApi python script.py
        (call torch.cuda.cudart().cudaProfilerStart() / Stop() in code)


##### PART 2 — THE TIMELINE ANATOMY: READING EVERY ROW

### The nsys Timeline Layout

    The Nsight Systems GUI shows rows stacked vertically, time on the x-axis:

    ┌────────────────────────────────────────────────────────────────────┐
    │ NVTX       │[===== forward =====][====== backward ======][=optim=] │
    ├────────────────────────────────────────────────────────────────────┤
    │ CPU Thread │▓▓▓▓▓▓▓▓▓▓▓▓▓  ··  ▓▓▓▓▓▓▓▓▓▓▓  ···  ▓▓▓▓▓▓▓▓▓▓▓▓      │
    │ (API calls)│                ↑                  ↑                   │
    │            │           cuda sync           cuda sync               │
    ├────────────────────────────────────────────────────────────────────┤
    │ Stream 7   │       [kernel_A][kernel_B][kernel_C]    [kernel_D]    │
    │ (compute)  │                                     ↑                 │
    │            │                              gap = CPU overhead       │
    ├────────────────────────────────────────────────────────────────────┤
    │ Stream 13  │  [H2D copy]              [H2D copy]                   │
    │ (memcpy)   │                                                       │
    ├────────────────────────────────────────────────────────────────────┤
    │ GPU util   │████████████████   ░░░░   ████████████░░░░████████████ │
    └────────────────────────────────────────────────────────────────────┘
                  ← time →

### What Each Row Tells You

    NVTX ROW:
        Coloured regions you annotated in your code.
        Answers: which Python function is responsible for what GPU work?
        Without NVTX: timeline is a sea of opaque kernel names.
        With NVTX: each training phase is clearly labelled.

    CPU THREAD ROW:
        Dense bars = CPU actively making API calls (submitting work).
        Gaps = CPU is blocking (synchronisation) or computing (Python).
        A long gap before GPU work = CPU is the bottleneck.
        Tall bars covering GPU idle periods = too many API calls per kernel.

    GPU STREAM ROWS (one row per active stream):
        Coloured blocks = kernel execution or memory copy.
        Gaps between blocks = GPU idle (waiting for next launch).
        Overlap between streams = actual parallelism.
        Serialised streams = synchronisation barriers blocking overlap.

    GPU UTILISATION ROW:
        Percentage of SMs executing at least one warp, averaged per period.
        100% = fully occupied. 0% = idle.
        The single most important high-level metric in the timeline.
        Low utilisation = pipeline problem (always fixable at system level).

### The Seven Diagnostic Questions

    Reading a timeline in order:

    1. What is the average GPU utilisation %?
       → If < 80%: pipeline problem. Find the gaps.

    2. Are H2D copies overlapping with compute kernels?
       → If not: use async transfers + dedicated copy stream.

    3. Are there long CPU gaps before GPU work?
       → CPU is the bottleneck: Python overhead, dataloader, preprocessing.

    4. Are CUDA synchronisation calls (cudaStreamSynchronize) frequent?
       → Unnecessary syncs flush the pipeline. Replace with CUDA events.

    5. Are multiple streams actually executing in parallel?
       → Check stream rows. Serialised: look for barriers and dependencies.

    6. Are kernel launches < 10 µs apart?
       → Launch overhead may be limiting. Use CUDA graphs (module 35).

    7. Which kernel (by name) appears most frequently and for longest?
       → That is the kernel to hand to Nsight Compute.


##### PART 3 — CUDA STREAMS: THE MECHANISM BEHIND OVERLAP

### What a Stream Is

    A CUDA stream is an ordered queue of GPU operations.
    Within a stream: all operations execute in submission order.
    Across streams: operations may overlap if no dependency exists.

    The DEFAULT STREAM (stream 0) is special:
        Operations in stream 0 block ALL other streams until complete.
        Operations in other streams block until stream 0 is clear.
        → Never use stream 0 in production code. Always create explicit streams.

    STREAM SEMANTICS:
        cudaStreamCreate(&stream_A)
        cudaMemcpyAsync(d_in, h_in, size, H2D, stream_A)  // enqueued
        kernel<<<grid, block, 0, stream_A>>>(d_in, d_out)  // runs after copy
        cudaMemcpyAsync(h_out, d_out, size, D2H, stream_A) // runs after kernel
        cudaStreamSynchronize(stream_A)                     // CPU waits here

    The GPU executes the stream_A queue asynchronously — the CPU returns
    immediately from each cudaMemcpyAsync / kernel launch call. The CPU
    only blocks at cudaStreamSynchronize.

### Stream Parallelism: The Copy/Compute Overlap Pattern

    SERIALISED (wrong — all on one stream):
        [H2D batch_0][kernel batch_0][D2H batch_0][H2D batch_1][kernel batch_1]

    OVERLAPPED (correct — copy stream + compute stream):
        Stream 0 (compute): ················[kernel batch_0]·····[kernel batch_1]
        Stream 1 (copy):    [H2D batch_0][D2H batch_0][H2D batch_1][D2H batch_1]

    The copy engine and SM compute engine are separate hardware — they can
    run simultaneously. On H100: two copy engines (one H2D, one D2H) plus
    the compute engines all operate independently.

    For the overlap to work, three conditions must hold:
        1. Transfers use cudaMemcpyAsync (NOT cudaMemcpy, which is synchronous)
        2. Host memory is page-locked (cudaMallocHost or torch.pin_memory())
        3. The transfer and kernel are on DIFFERENT streams

    PAGE-LOCKED (PINNED) MEMORY:
        Regular malloc: OS may swap pages to disk. DMA cannot be used.
        cudaMallocHost: pages are locked, DMA transfers are possible.
        DMA: Direct Memory Access — GPU reads host memory without involving CPU.
        Speed: PCIe Gen4 x16 = 32 GB/s H2D + 32 GB/s D2H simultaneously.
        With pinned memory: achieves near-theoretical PCIe bandwidth.
        Without pinned memory: 2–5× slower (DMA not possible, CPU-staged).

### CUDA Events: Lightweight Synchronisation

    cudaStreamSynchronize: blocks the CPU thread until stream is empty.
    cudaEvent: a timestamp marker inserted into a stream.

    Pattern: make stream B wait for a specific point in stream A
        (without blocking the CPU):

        cudaEventRecord(event, stream_A)         // insert marker into A
        cudaStreamWaitEvent(stream_B, event, 0)  // B waits until A reaches marker

    This is the building block for expressing fine-grained dependencies:
        - Stream B (kernel) can start as soon as stream A (copy) finishes input data
        - Stream C (D2H) can start as soon as the specific output kernel in stream B finishes
        - CPU never blocks — full async pipeline

    Timeline appearance: stream rows with event arrows connecting them,
    showing the precise dependency points rather than full barriers.

### The Three-Stage Pipeline (Double Buffering)

    Maximum throughput: while GPU processes batch N,
    CPU prepares batch N+1, and result of batch N-1 is being copied back.

    BUFFERS: allocate 2 sets of pinned host + device buffers (A and B).

    Stage:    T=0       T=1       T=2       T=3       T=4
    CPU:      prep_0    prep_1    prep_2    prep_3    prep_4
    Copy H2D: cp_0A    cp_1B    cp_2A    cp_3B
    Compute:            krnl_0A  krnl_1B  krnl_2A  krnl_3B
    Copy D2H:                    cp_0A    cp_1B    cp_2A

    All three stages run in parallel after the first 2-stage warmup.
    Throughput ≈ max(T_cpu, T_copy, T_kernel) per batch.
    Ideal: all three stages take equal time (balanced pipeline).

    nsys timeline of a perfectly pipelined run:
        CPU row:      ████████████████████████████████████████
        H2D row:      [0][1][2][3][4][5][6][7][8][9]...
        Kernel row:   [0][1][2][3][4][5][6][7][8]...
        D2H row:      [0][1][2][3][4][5][6][7]...
    All rows fully utilised. No gaps.


##### PART 4 — NVTX ANNOTATIONS: MAKING TIMELINES READABLE

### Why NVTX Is Non-Negotiable

    Without NVTX, the nsys timeline shows anonymous kernel names:
        volta_sgemm_128x64_nn
        void at::native::vectorized_elementwise_kernel<...>
        cudnn::ops::nchwToNhwcKernel<...>

    With NVTX, the timeline shows your logical structure:
        [======= forward pass =======][====== backward ======][== optimizer ==]
        [== layer 0 ==][== layer 1 ==][== attn ==][== FFN ==]

    NVTX annotations are the first thing to add to any production training
    or inference code before profiling. They cost nothing when profiling
    is inactive and provide irreplaceable context when it is.

### NVTX API

    C++ / CUDA:
        #include <nvtx3/nvToolsExt.h>
        nvtxRangePush("forward_pass");
        // ... GPU work ...
        nvtxRangePop();

        // With colour and category:
        nvtxEventAttributes_t attr = {};
        attr.color     = 0xFF4488FF;   // ARGB blue
        attr.category  = 1;            // user-defined category ID
        attr.message.ascii = "attention_layer_3";
        nvtxRangePushEx(&attr);

    Python (via cupy or torch):
        import torch.cuda.nvtx as nvtx
        nvtx.range_push("forward_pass")
        output = model(input)
        nvtx.range_pop()

        # Context manager style:
        with torch.cuda.nvtx.range("embedding_lookup"):
            emb = embedding(token_ids)

    torch.profiler integration:
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            with torch.profiler.record_function("model_inference"):
                output = model(input)

### Colour Coding Convention (Team Standard)

    ARGB hex values for consistent colour coding across engineers:

        Data loading / preprocessing:  0xFF_FF6B6B  (red)
        H2D memory copy:               0xFF_FFA500  (orange)
        Forward pass:                  0xFF_4488FF  (blue)
        Attention layer:               0xFF_00BFFF  (deep sky blue)
        FFN layer:                     0xFF_7B68EE  (medium slate blue)
        Backward pass:                 0xFF_32CD32  (lime green)
        Optimizer step:                0xFF_FFD700  (gold)
        D2H memory copy:               0xFF_FF69B4  (hot pink)
        Idle / synchronisation wait:   0xFF_808080  (grey)

    Each NVTX range becomes a coloured bar in the nsys GUI, allowing
    instant visual identification of where wall time is going.

### Recommended Annotation Granularity

    TOO COARSE (useless):
        nvtx.range_push("training")    # entire script in one range
        # 10 minutes of training
        nvtx.range_pop()

    TOO FINE (noise, overhead):
        for i in range(N):
            nvtx.range_push(f"op_{i}")    # millions of tiny ranges
            single_multiply(a[i], b[i])
            nvtx.range_pop()

    CORRECT GRANULARITY:
        Epoch / iteration boundary         ← always
        Forward / backward / optimizer     ← always
        Per-layer or per-block (optional)  ← for architecture analysis
        Per major memory transfer          ← always
        DataLoader prefetch thread         ← always


##### PART 5 — THE SIX CANONICAL SYSTEM-LEVEL BOTTLENECKS

### Bottleneck 1: CPU Launch Overhead (kernel launch gap)

    SYMPTOM in nsys:
        GPU stream row shows gaps of 5–50 µs between consecutive kernels.
        CPU row shows dense activity (Python interpreter overhead).

    ROOT CAUSE:
        Every kernel launch calls into the CUDA runtime: ~2–5 µs per launch.
        PyTorch ops add Python dispatch overhead: ~5–50 µs per op.
        For small kernels (< 100 µs), launch overhead dominates.

    EVIDENCE in nsys stats:
        "CUDA API Statistics" table shows cudaLaunchKernel taking significant
        total time relative to kernel execution time.

    FIXES:
        a) CUDA Graphs: capture a sequence of kernels once, replay instantly
           (module 35). Eliminates per-launch overhead entirely.
        b) torch.compile: fuses Python-level ops into fewer kernel launches.
        c) Operator fusion: combine elementwise ops (LayerNorm + GELU + Add)
           into a single kernel launch.

### Bottleneck 2: Blocking Memory Copy (H2D on default stream)

    SYMPTOM in nsys:
        H2D copy blocks all kernel execution: compute stream shows gap
        exactly equal to the copy duration. Streams run serially.

    ROOT CAUSE:
        cudaMemcpy (synchronous) or memcpy on stream 0 (default stream).
        Stream 0 blocks all other streams.
        Or: non-pinned memory forces CPU-staged transfer (2–5× slower).

    EVIDENCE in nsys timeline:
        H2D copy bar on stream 0 row, kernel bars with exactly matching gap.

    FIXES:
        a) Move copy to a dedicated stream: cudaMemcpyAsync(..., copy_stream)
        b) Pin host memory: torch.Tensor.pin_memory() or cudaMallocHost
        c) Prefetch next batch while current batch is computing
        d) Use CUDA Unified Memory with prefetch hints if data access is sparse

### Bottleneck 3: Unnecessary cudaStreamSynchronize

    SYMPTOM in nsys:
        Periodic CPU rows go flat (blocking). GPU goes idle at same points.
        Pattern repeats every iteration — looks like a heartbeat of idle.

    ROOT CAUSE:
        Explicit synchronisation inserted for debugging (e.g., to check
        intermediate values). Or: Python code that reads GPU tensor values:
            loss_val = loss.item()   # implicit cudaDeviceSynchronize!
            if i % 100 == 0: print(loss.item())  # sync every 100 steps

    EVIDENCE in nsys:
        cudaStreamSynchronize or cudaDeviceSynchronize in API trace.
        GPU utilisation drops to 0% at each sync point.

    FIXES:
        a) Replace loss.item() calls inside the hot loop with deferred reads
        b) Use torch.cuda.Event for timing, not Python time.time() + .item()
        c) Log metrics asynchronously: accumulate on GPU, read once per epoch
        d) If sync is unavoidable, batch multiple tensor reads before syncing

### Bottleneck 4: CPU-Bound DataLoader

    SYMPTOM in nsys:
        GPU utilisation trace shows periodic troughs (0–20%).
        CPU sampling shows high time in augmentation / decode / collate.
        H2D copy stream shows long gaps matching the CPU troughs.

    ROOT CAUSE:
        DataLoader cannot keep up with GPU throughput. Typical causes:
        - JPEG decode on CPU (torchvision transforms)
        - Complex augmentation pipelines
        - Slow disk I/O (spinning HDD, network filesystem)
        - num_workers too low (DataLoader starvation)
        - GIL contention in Python worker processes

    EVIDENCE in nsys:
        CPU row shows worker thread activity preceding each H2D copy.
        GPU idle spans correlate with CPU decode spans in NVTX annotations.

    FIXES:
        a) Increase DataLoader num_workers (until disk/CPU saturates)
        b) DALI (NVIDIA Data Loading Library): GPU-based decode + augment
        c) Prefetch to GPU: custom prefetcher with pin_memory=True
        d) Cache decoded tensors to shared memory for repeated epochs
        e) Use NVJPEG for GPU-accelerated JPEG decode

### Bottleneck 5: Stream Serialisation via False Dependency

    SYMPTOM in nsys:
        Multiple streams exist but do not overlap — they run sequentially.
        GPU utilisation never exceeds what a single stream could achieve.

    ROOT CAUSE:
        Operations in separate streams that were INTENDED to overlap are
        serialised by an implicit dependency. Common causes:
        - cudaDeviceSynchronize() instead of stream-level sync
        - Operations on stream 0 (default stream) inserted between
          multi-stream operations
        - PyTorch autograd inserting sync barriers for gradient accumulation
        - cuBLAS/cuDNN workspace shared between streams without isolation

    EVIDENCE in nsys:
        Stream rows show sequential blocks with no overlap.
        CUDA API trace reveals cudaDeviceSynchronize calls.

    FIXES:
        a) Replace cudaDeviceSynchronize with per-stream sync or events
        b) Never mix stream 0 with explicit streams
        c) Use separate cuBLAS handles per stream (each handle has own workspace)
        d) In PyTorch DDP: verify no_sync() context is correctly scoped

### Bottleneck 6: Memory Allocation in the Hot Loop

    SYMPTOM in nsys:
        CUDA API trace shows cudaMalloc inside the iteration loop.
        Each cudaMalloc shows as a long API call (0.1–10 ms) with a
        matching GPU idle period (allocator waits for previous ops).

    ROOT CAUSE:
        Framework allocating new GPU buffers every iteration, or:
        - Intermediate tensor shapes changing every step (disables cache)
        - torch.empty / torch.zeros inside forward pass
        - Growing gradient accumulation buffers

    EVIDENCE in nsys:
        cudaMalloc / cudaFree in API trace with multi-millisecond latency.
        Periodic GPU stalls matching allocation calls.

    FIXES:
        a) Pre-allocate all buffers before the hot loop
        b) Use torch.cuda.CUDAGraph or static shapes to enable allocator reuse
        c) torch.backends.cuda.matmul.allow_tf32 = True can reduce workspace
        d) Use memory pools: torch allocator caches by default, but explicit
           torch.cuda.empty_cache() calls discard the cache — remove them


##### PART 6 — MULTI-GPU PROFILING: DDP, NCCL & NVLINK

### Profiling DDP Training

    nsys profiles a single process. For DDP with multiple processes,
    profile ONE process at a time, typically rank 0:

        # Launch with torchrun:
        nsys profile -o rank0 --trace=cuda,nvtx,nccl \
            torchrun --nproc_per_node=8 train_ddp.py

        # Or profile rank 0 only in multi-process launch:
        RANK=0 nsys profile -o rank0 python train_ddp.py &
        RANK=1 python train_ddp.py &
        ...

    Profile ALL ranks simultaneously for collective analysis:
        nsys profile -o rank_%q{RANK} --trace=cuda,nvtx,nccl python train_ddp.py

    This creates rank_0.nsys-rep, rank_1.nsys-rep, etc.
    Open all simultaneously in the nsys GUI for cross-rank timeline comparison.

### Reading NCCL Operations on the Timeline

    NCCL collectives appear as CUDA kernels on the stream timeline:
        ncclKernel_AllReduce_RING_LL_Sum_float
        ncclKernel_Broadcast_RING_LL_float
        ncclKernel_ReduceScatter_...

    KEY INSIGHT: NCCL AllReduce in DDP runs on a SEPARATE stream from
    the backward pass. nsys shows these as two parallel stream rows.

    GRADIENT COMPRESSION OPPORTUNITY:
        If NCCL AllReduce bars are LONGER than backward pass bars:
            → Communication is on the critical path → bottleneck.
            Fix: overlapped gradient communication (bucket strategy in DDP),
            gradient compression (FP16 allreduce), or ZeRO sharding.
        If AllReduce bars are SHORTER or OVERLAPPING with backward:
            → Communication is hidden → optimal.

    NCCL Timeline Patterns:
        RING AllReduce:  2×(N-1)/N × size / bandwidth per step
        TREE AllReduce:  2×log2(N) × latency + 2×size/bandwidth (better for small)
        NVLINK vs PCIe:  600 GB/s (NVLink A100×8) vs 32 GB/s (PCIe Gen4) → 19×

### NVLink Traffic in nsys

    With --gpu-metrics-device:
        NVLink row shows bandwidth per link, per direction, per time slice.
        Useful for: detecting link imbalance, identifying which GPU is sender.

    Healthy NVLink pattern: symmetric bidirectional traffic.
    Pathological pattern: one GPU sending, others idle → load imbalance.

    For A100 SXM4 (NVLink 3.0):
        Total bidirectional bandwidth: 600 GB/s across 12 NVLink links.
        An AllReduce of 1 GB at peak NVLink: ~3.3 ms.
        Same AllReduce over PCIe: ~62 ms. 19× difference.


##### PART 7 — PYTORCH INTEGRATION: torch.profiler & CAPTURE RANGES

### torch.profiler vs nsys

    torch.profiler:
        Python-native. Output: TensorBoard / JSON / text table.
        Records: PyTorch ops by name, CPU time, CUDA time, memory.
        Best for: quick iteration analysis from Python, TensorBoard integration.
        Limitation: Python-level overhead, less precise GPU timestamps.

    nsys:
        System-level. Records actual GPU hardware timestamps.
        Best for: production profiling, copy/compute overlap analysis,
        multi-GPU, finding synchronisation barriers.

    RECOMMENDED COMBINATION:
        1. torch.profiler for quick op-level breakdown (which ops dominate).
        2. nsys with NVTX for system-level pipeline analysis.
        3. ncu for kernel-level analysis of the top bottleneck from step 2.

### Selective Profiling with cudaProfilerApi

    Profiling from program start captures startup overhead (module import,
    weight loading) — typically not what you want. Selective profiling:

    Python:
        import ctypes
        _cudart = ctypes.CDLL('libcudart.so')

        # Or, simpler, via torch:
        torch.cuda.cudart().cudaProfilerStart()
        # ... hot loop iterations ...
        torch.cuda.cudart().cudaProfilerStop()

    Launch with:
        nsys profile --capture-range=cudaProfilerApi -o profile python train.py

    nsys only records between Start and Stop calls.
    Avoids profiling 30 seconds of weight loading to get 5 iterations of training.

### Identifying the Iteration Boundary

    Add NVTX markers at iteration boundaries so the nsys timeline
    shows per-iteration structure:

        for step, batch in enumerate(dataloader):
            with torch.cuda.nvtx.range(f"iteration_{step}"):
                with torch.cuda.nvtx.range("data_to_device"):
                    inputs = batch.to(device, non_blocking=True)
                with torch.cuda.nvtx.range("forward"):
                    loss = model(inputs)
                with torch.cuda.nvtx.range("backward"):
                    loss.backward()
                with torch.cuda.nvtx.range("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()

    In nsys GUI: zoom to 2–3 iterations. Check that the pattern repeats
    identically (non-identical iterations indicate warmup or GC effects).

### nsys stats Output: Key Tables

    Run: nsys stats profile.nsys-rep

    TABLE 1 — CUDA Kernel Statistics (sorted by total time):
        Time(%)  Total(ns)  Instances  Avg(ns)  Name
        Shows: which kernels dominate GPU time.

    TABLE 2 — CUDA API Statistics:
        Time(%)  Total(ns)  Instances  Avg(ns)  Name
        Shows: which API calls dominate CPU overhead.
        Red flag: cudaDeviceSynchronize with high count or total time.

    TABLE 3 — CUDA Memory Operation Statistics:
        Type  Direction  Size(MB)  Count  Avg(ns)  Bandwidth(GB/s)
        Shows: total data moved, effective transfer bandwidth.
        Red flag: pageable (non-pinned) transfers showing low bandwidth.

    TABLE 4 — OS Runtime API Statistics:
        Shows: mutex waits, thread events — useful for GIL analysis.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Timeline Gap Analyser — GPU Utilisation & Idle Classification": {
        "description": (
            "Simulate nsys timeline data for a realistic training workload. "
            "Parse kernel launch timestamps and gaps between GPU operations. "
            "Classify each gap: CPU overhead, synchronisation, copy stall, "
            "or allocator pause. Compute GPU utilisation percentage. Show "
            "the ASCII timeline view and rank bottlenecks by wall-clock impact."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict

print("=" * 68)
print("  TIMELINE GAP ANALYSER — GPU Utilisation & Idle Classification")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GpuOp:
    name:       str
    kind:       str     # kernel | h2d | d2h | d2d | sync
    stream:     int
    start_us:   float
    end_us:     float
    size_mb:    float = 0.0

    @property
    def duration_us(self):
        return self.end_us - self.start_us

@dataclass
class Gap:
    start_us:  float
    end_us:    float
    after_op:  str
    before_op: str
    cause:     str    # cpu_overhead | sync_barrier | copy_stall | allocator

    @property
    def duration_us(self):
        return self.end_us - self.start_us


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Simulate four different training pipeline profiles
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Four Pipeline Profiles (3 iterations each)")
print("━" * 68)
print()

def make_profile(name, cpu_overhead_us, sync_us, use_async_copy,
                 has_alloc_pause, copy_us, kernel_us):
    """
    Build a synthetic timeline of GPU operations for one scenario.
    Returns list of GpuOp covering 3 training iterations.
    """
    ops = []
    t = 0.0

    for iteration in range(3):
        # ── H2D copy ────────────────────────────────────────────────
        if use_async_copy:
            # Async: copy on stream 2, kernel on stream 7 — overlapped
            copy_start = t + cpu_overhead_us * 0.1   # minimal CPU before async submit
            copy_end   = copy_start + copy_us
            ops.append(GpuOp("H2D_batch", "h2d", 2, copy_start, copy_end, 0.5))
            # Kernel starts after copy (dependency), but copy already running
            kern_start = copy_end
        else:
            # Synchronous: copy then kernel strictly serial
            t += cpu_overhead_us * 0.3
            copy_start = t
            copy_end   = copy_start + copy_us
            ops.append(GpuOp("H2D_batch", "h2d", 7, copy_start, copy_end, 0.5))
            t = copy_end

        # ── Optional allocation pause ────────────────────────────────
        if has_alloc_pause and iteration == 0:
            alloc_start = copy_end
            alloc_end   = alloc_start + 4500.0
            ops.append(GpuOp("cudaMalloc", "sync", 0, alloc_start, alloc_end))
            if use_async_copy:
                kern_start = alloc_end
            else:
                t = alloc_end

        # ── CPU overhead before kernel ───────────────────────────────
        if use_async_copy:
            kern_start = kern_start + cpu_overhead_us * 0.05
        else:
            t += cpu_overhead_us
            kern_start = t

        kern_end = kern_start + kernel_us
        ops.append(GpuOp("transformer_forward", "kernel", 7, kern_start, kern_end))

        # ── Optional sync barrier ────────────────────────────────────
        if sync_us > 0:
            sync_start = kern_end + cpu_overhead_us * 0.1
            sync_end   = sync_start + sync_us
            ops.append(GpuOp("cudaStreamSync", "sync", 0, sync_start, sync_end))
            t = sync_end
        else:
            t = kern_end + cpu_overhead_us * 0.1

        # ── Backward + optimizer kernels ────────────────────────────
        bwd_start = t + cpu_overhead_us * 0.1
        ops.append(GpuOp("transformer_backward", "kernel", 7,
                          bwd_start, bwd_start + kernel_us * 1.6))
        t = bwd_start + kernel_us * 1.6 + cpu_overhead_us * 0.05

        ops.append(GpuOp("adam_update", "kernel", 7, t, t + kernel_us * 0.3))
        t = t + kernel_us * 0.3

        # ── D2H (loss scalar) ───────────────────────────────────────
        # Simulating loss.item() — implicit sync
        t += cpu_overhead_us * 0.2
        ops.append(GpuOp("D2H_loss_scalar", "d2h", 7, t, t + 8.0))
        t += 8.0 + cpu_overhead_us * 0.3

    # Sort by start time
    ops.sort(key=lambda o: o.start_us)
    return ops


profiles = {
    "A: Baseline (serial, unoptimised)": make_profile(
        "Baseline", cpu_overhead_us=800, sync_us=200,
        use_async_copy=False, has_alloc_pause=False,
        copy_us=1200, kernel_us=3000),
    "B: + Async copy (pinned memory)":  make_profile(
        "Async", cpu_overhead_us=800, sync_us=200,
        use_async_copy=True, has_alloc_pause=False,
        copy_us=1200, kernel_us=3000),
    "C: + Remove sync + async copy":    make_profile(
        "NoSync", cpu_overhead_us=800, sync_us=0,
        use_async_copy=True, has_alloc_pause=False,
        copy_us=1200, kernel_us=3000),
    "D: + Reduced CPU overhead (compiled)": make_profile(
        "Compiled", cpu_overhead_us=80, sync_us=0,
        use_async_copy=True, has_alloc_pause=False,
        copy_us=1200, kernel_us=3000),
}


def analyse_profile(ops: List[GpuOp]):
    """Compute GPU utilisation and classify idle gaps."""
    if not ops:
        return {}, []

    total_span = ops[-1].end_us - ops[0].start_us

    # Active GPU time = time at least one op is running (on any stream)
    events = []
    for op in ops:
        if op.kind in ('kernel', 'h2d', 'd2h', 'd2d'):
            events.append((op.start_us, +1))
            events.append((op.end_us,   -1))
    events.sort()

    active_us = 0.0
    depth = 0
    prev_t = None
    for t, delta in events:
        if depth > 0 and prev_t is not None:
            active_us += t - prev_t
        depth += delta
        prev_t = t

    utilisation_pct = active_us / total_span * 100 if total_span > 0 else 0

    # Classify gaps on primary compute stream (stream 7)
    compute_ops = [o for o in ops if o.stream == 7 and o.kind in ('kernel', 'h2d', 'd2h')]
    gaps = []
    for i in range(1, len(compute_ops)):
        prev_op = compute_ops[i - 1]
        curr_op = compute_ops[i]
        gap_dur = curr_op.start_us - prev_op.end_us
        if gap_dur < 1.0:
            continue
        # Classify
        if gap_dur > 3000:
            cause = "allocator"
        elif gap_dur > 150:
            cause = "sync_barrier"
        elif gap_dur > 30:
            cause = "copy_stall"
        else:
            cause = "cpu_overhead"
        gaps.append(Gap(prev_op.end_us, curr_op.start_us,
                        prev_op.name, curr_op.name, cause))

    stats = {
        "total_span_ms":   total_span / 1000,
        "active_gpu_ms":   active_us / 1000,
        "utilisation_pct": utilisation_pct,
        "n_gaps":          len(gaps),
        "gaps":            gaps,
    }
    return stats, gaps


print(f"  {'Profile':<42}  {'Span(ms)':>9}  {'GPU ms':>8}  {'Util%':>7}  {'Gaps'}")
print("  " + "─" * 72)

all_stats = {}
for pname, ops in profiles.items():
    stats, gaps = analyse_profile(ops)
    all_stats[pname] = stats
    cause_counts = {}
    for g in gaps:
        cause_counts[g.cause] = cause_counts.get(g.cause, 0) + 1
    gap_str = " ".join(f"{k[:4]}×{v}" for k, v in cause_counts.items())
    print(f"  {pname:<42}  {stats['total_span_ms']:>9.1f}  "
          f"{stats['active_gpu_ms']:>8.1f}  {stats['utilisation_pct']:>6.1f}%  {gap_str}")

baseline_span = all_stats["A: Baseline (serial, unoptimised)"]['total_span_ms']
print()
print("  Speedups relative to Baseline A:")
for pname, stats in all_stats.items():
    sx = baseline_span / stats['total_span_ms']
    print(f"    {pname[:44]}: {sx:.2f}×")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: ASCII timeline visualisation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — ASCII Timeline (Profile A vs D, 1 iteration)")
print("━" * 68)
print()

def ascii_timeline(ops, width=60, label=""):
    """Render a compact ASCII timeline for the first iteration."""
    iter_ops = [o for o in ops if o.start_us < ops[0].start_us + 15000]
    if not iter_ops:
        return
    t_start = iter_ops[0].start_us
    t_end   = max(o.end_us for o in iter_ops)
    t_span  = t_end - t_start

    streams = sorted(set(o.stream for o in iter_ops))
    kind_chars = {'kernel': '█', 'h2d': '▒', 'd2h': '░', 'sync': 'S', 'd2d': '▓'}
    kind_names = {'kernel': 'kernel', 'h2d': 'H2D', 'd2h': 'D2H', 'sync': 'sync'}

    print(f"  {label}")
    print(f"  0{'─' * (width - 2)}{t_span/1000:.0f}ms")
    for s in streams:
        row = [' '] * width
        s_ops = [o for o in iter_ops if o.stream == s]
        for op in s_ops:
            start_col = int((op.start_us - t_start) / t_span * width)
            end_col   = max(start_col + 1, int((op.end_us - t_start) / t_span * width))
            ch = kind_chars.get(op.kind, '?')
            for c in range(min(start_col, width-1), min(end_col, width)):
                row[c] = ch
        row_str = ''.join(row)
        print(f"  stream {s:2d}  |{row_str}|")
    print()
    print(f"  Legend: █=kernel  ▒=H2D copy  ░=D2H copy  S=sync/alloc")
    print()

ascii_timeline(list(profiles.values())[0], label="Profile A — Serial (unoptimised):")
ascii_timeline(list(profiles.values())[3], label="Profile D — Optimised (async+compiled):")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Gap breakdown and fix recommendations
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Gap Breakdown & Actionable Fixes")
print("━" * 68)
print()

fixes = {
    "cpu_overhead":  "torch.compile / CUDA graphs to reduce per-op Python dispatch",
    "sync_barrier":  "Remove loss.item() from hot loop; replace sync with CUDA events",
    "copy_stall":    "Async copy + pinned memory; prefetch next batch",
    "allocator":     "Pre-allocate buffers; avoid dynamic shapes; remove empty_cache()",
}

for pname, stats in all_stats.items():
    gaps = stats['gaps']
    if not gaps:
        continue
    cause_time = defaultdict(float)
    for g in gaps:
        cause_time[g.cause] += g.duration_us
    total_gap_us = sum(cause_time.values())
    print(f"  {pname}:")
    for cause, time_us in sorted(cause_time.items(), key=lambda x: -x[1]):
        pct = time_us / (stats['total_span_ms'] * 1000) * 100
        bar = '█' * int(pct / 2)
        print(f"    {cause:<16} {time_us/1000:>7.1f} ms  ({pct:4.1f}%)  {bar}")
        print(f"      Fix: {fixes[cause]}")
    print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Stream Overlap Simulator — Copy/Compute Pipeline Analysis": {
        "description": (
            "Simulate the copy/compute overlap pipeline with single stream vs "
            "multi-stream configurations. Model three-stage double-buffered "
            "pipelines (H2D copy, compute, D2H copy running in parallel). "
            "Show throughput scaling as pipeline stages balance. Quantify the "
            "speedup from pinned memory, async transfers, and double buffering."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple

print("=" * 68)
print("  STREAM OVERLAP SIMULATOR — Copy/Compute Pipeline Analysis")
print("=" * 68)
print()

np.random.seed(7)


# ─────────────────────────────────────────────────────────────────────
# Hardware model
# ─────────────────────────────────────────────────────────────────────

PCIE_GEN4_BW_GBS  = 32.0     # PCIe Gen4 x16 unidirectional GB/s
PCIE_PINNED_EFF   = 0.92     # efficiency with pinned memory
PCIE_PAGEABLE_EFF = 0.25     # efficiency with pageable memory (no DMA)


def h2d_time_us(size_mb, pinned=True):
    bw = PCIE_GEN4_BW_GBS * (PCIE_PINNED_EFF if pinned else PCIE_PAGEABLE_EFF)
    return (size_mb / 1024) / bw * 1e6


def d2h_time_us(size_mb, pinned=True):
    return h2d_time_us(size_mb, pinned)


# ─────────────────────────────────────────────────────────────────────
# Pipeline simulators
# ─────────────────────────────────────────────────────────────────────

def simulate_serial(n_batches, h2d_us, kernel_us, d2h_us):
    """All ops on one stream — strictly serial."""
    total = n_batches * (h2d_us + kernel_us + d2h_us)
    return total

def simulate_async_single_buffer(n_batches, h2d_us, kernel_us, d2h_us):
    """
    Async copy + compute on separate streams, but single buffer.
    Copy and kernel can overlap IF they target independent data.
    Kernel of batch N overlaps with H2D of batch N+1.
    """
    # Startup: copy batch 0
    # Then each iteration: max(kernel, copy) + d2h (must wait for kernel)
    startup    = h2d_us
    per_batch  = max(kernel_us, h2d_us) + d2h_us
    total      = startup + per_batch * n_batches
    return total

def simulate_double_buffer(n_batches, h2d_us, kernel_us, d2h_us):
    """
    Full three-stage double-buffered pipeline.
    Stage 1: H2D batch N+1 (copy stream A)
    Stage 2: Kernel batch N (compute stream)
    Stage 3: D2H batch N-1 (copy stream B)
    All three run simultaneously after warmup.
    Throughput = max(h2d_us, kernel_us, d2h_us) per batch.
    """
    # Warmup: 2 batches to fill pipeline
    warmup = h2d_us + max(kernel_us, h2d_us)
    # Steady state: one batch per max(stage durations)
    per_batch_steady = max(h2d_us, kernel_us, d2h_us)
    total = warmup + per_batch_steady * (n_batches - 1) + d2h_us
    return total


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Varying batch size — overlap benefit changes with ratio
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Copy/Compute Ratio vs Pipeline Speedup")
print("━" * 68)
print()

N_BATCHES = 20
KERNEL_US = 5000.0    # fixed kernel time

print(f"  Kernel time fixed at {KERNEL_US/1000:.1f} ms, N={N_BATCHES} batches")
print(f"  PCIe Gen4 x16: {PCIE_GEN4_BW_GBS} GB/s (pinned) / "
      f"{PCIE_GEN4_BW_GBS * PCIE_PAGEABLE_EFF:.1f} GB/s (pageable)")
print()

print(f"  {'Batch MB':>9}  {'H2D ms':>7}  {'Ratio':>7}  "
      f"{'Serial ms':>10}  {'Async ms':>9}  {'Dbl-buf ms':>11}  "
      f"{'Async sx':>9}  {'DblBuf sx':>10}")
print("  " + "─" * 82)

for batch_mb in [8, 32, 64, 128, 256, 512, 1024, 2048]:
    h2d_us = h2d_time_us(batch_mb, pinned=True)
    d2h_us_val = d2h_time_us(0.004, pinned=True)   # tiny D2H (loss scalar only)

    t_serial  = simulate_serial(N_BATCHES, h2d_us, KERNEL_US, d2h_us_val)
    t_async   = simulate_async_single_buffer(N_BATCHES, h2d_us, KERNEL_US, d2h_us_val)
    t_double  = simulate_double_buffer(N_BATCHES, h2d_us, KERNEL_US, d2h_us_val)
    ratio     = h2d_us / KERNEL_US

    sx_async  = t_serial / t_async
    sx_double = t_serial / t_double

    print(f"  {batch_mb:>9}  {h2d_us/1000:>7.2f}  {ratio:>7.3f}  "
          f"{t_serial/1000:>10.1f}  {t_async/1000:>9.1f}  {t_double/1000:>11.1f}  "
          f"{sx_async:>9.2f}×  {sx_double:>10.2f}×")

print()
print("  When ratio << 1 (copy fast vs kernel): double buffering gives full speedup.")
print("  When ratio > 1  (copy bottleneck):     speedup saturates at serial/h2d.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Pinned vs pageable memory — the single biggest easy win
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Pinned vs Pageable Memory: Transfer Bandwidth")
print("━" * 68)
print()

print(f"  PCIe Gen4 x16 peak one-way: {PCIE_GEN4_BW_GBS:.0f} GB/s")
print(f"  Effective with pinned memory:   {PCIE_GEN4_BW_GBS * PCIE_PINNED_EFF:.1f} GB/s  "
      f"({PCIE_PINNED_EFF*100:.0f}% efficiency)")
print(f"  Effective with pageable memory: {PCIE_GEN4_BW_GBS * PCIE_PAGEABLE_EFF:.1f} GB/s  "
      f"({PCIE_PAGEABLE_EFF*100:.0f}% efficiency, ~4× slower)")
print()
print(f"  WHY pageable is slower:")
print(f"    DMA requires physical pages to be locked (not swappable).")
print(f"    Without pinning: CUDA driver must first copy to a temporary")
print(f"    pinned staging buffer → doubles the data movement distance.")
print(f"    With pinning: direct DMA from application memory → GPU VRAM.")
print()

print(f"  {'Batch size':>12}  {'Pinned ms':>11}  {'Pageable ms':>13}  "
      f"{'Slowdown':>10}  {'Pinned BW':>11}  {'Pageable BW'}")
print("  " + "─" * 68)

for batch_mb in [16, 64, 256, 512, 1024, 4096]:
    t_pinned   = h2d_time_us(batch_mb, pinned=True)
    t_pageable = h2d_time_us(batch_mb, pinned=False)
    bw_pin   = batch_mb / (t_pinned / 1e6) / 1024
    bw_page  = batch_mb / (t_pageable / 1e6) / 1024
    print(f"  {batch_mb:>9} MB  {t_pinned/1000:>11.2f}  {t_pageable/1000:>13.2f}  "
          f"{t_pageable/t_pinned:>9.1f}×  {bw_pin:>8.1f} GB/s  {bw_page:.1f} GB/s")

print()
print("  torch usage:  dataloader = DataLoader(dataset, pin_memory=True)")
print("  Effect:       DataLoader workers allocate pinned host buffers.")
print("                .to(device, non_blocking=True) then uses async DMA.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Double-buffered pipeline ASCII timeline
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Double-Buffered Pipeline: ASCII Gantt Chart")
print("━" * 68)
print()

def render_gantt(h2d_us, kernel_us, d2h_us, n_batches=6, width=72):
    """
    Draw the double-buffered pipeline as an ASCII Gantt chart.
    Shows H2D, Compute, and D2H stages per batch.
    """
    rows = {"H2D (copy)  ": [], "Kernel      ": [], "D2H (copy)  ": []}

    # Schedule events:  batch i starts kernel when h2d[i] finishes
    # h2d and d2h streams are independent from kernel stream
    h2d_starts    = []
    kernel_starts = []
    d2h_starts    = []

    t_h2d_free    = 0.0
    t_kernel_free = 0.0
    t_d2h_free    = 0.0

    for i in range(n_batches):
        # H2D: can start when copy engine is free
        hs = max(t_h2d_free, 0.0 if i == 0 else h2d_starts[-1] + h2d_us)
        hs = t_h2d_free
        he = hs + h2d_us
        h2d_starts.append(hs)
        t_h2d_free = he

        # Kernel: must wait for its H2D to complete
        ks = max(t_kernel_free, he)
        ke = ks + kernel_us
        kernel_starts.append(ks)
        t_kernel_free = ke

        # D2H: must wait for kernel, and for D2H engine to be free
        ds = max(t_d2h_free, ke)
        de = ds + d2h_us
        d2h_starts.append(ds)
        t_d2h_free = de

    total_time = max(t_h2d_free, t_kernel_free, t_d2h_free)

    # Render
    print(f"  h2d={h2d_us/1000:.1f}ms  kernel={kernel_us/1000:.1f}ms  "
          f"d2h={d2h_us/1000:.1f}ms  | 0 to {total_time/1000:.1f}ms")
    print(f"  {'0':>3}{'─' * (width - 6)}{total_time/1000:.0f}ms")

    labels = ["H2D (copy)  ", "Kernel      ", "D2H (copy)  "]
    all_starts = [h2d_starts, kernel_starts, d2h_starts]
    dur_all    = [h2d_us, kernel_us, d2h_us]
    chars      = ['▒', '█', '░']

    for label, starts, dur, ch in zip(labels, all_starts, dur_all, chars):
        row = [' '] * width
        for i, s in enumerate(starts):
            e = s + dur
            sc = int(s / total_time * width)
            ec = max(sc + 1, int(e / total_time * width))
            batch_char = str(i % 10)
            for c in range(min(sc, width-1), min(ec, width)):
                row[c] = batch_char if (c == sc or c == ec - 1) else ch
        print(f"  {label} |{''.join(row)}|")

    serial_time = n_batches * (h2d_us + kernel_us + d2h_us)
    speedup = serial_time / total_time
    print(f"  Serial equivalent: {serial_time/1000:.1f}ms  |  "
          f"Pipeline: {total_time/1000:.1f}ms  |  Speedup: {speedup:.2f}×")
    print()

# Case 1: kernel dominates (copy is fast)
render_gantt(h2d_us=800, kernel_us=3000, d2h_us=100, n_batches=6)
# Case 2: copy and kernel balanced
render_gantt(h2d_us=2500, kernel_us=3000, d2h_us=500, n_batches=6)
# Case 3: copy dominates (GPU underutilised)
render_gantt(h2d_us=5000, kernel_us=3000, d2h_us=500, n_batches=6)


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Throughput vs num_streams sensitivity
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Multi-Stream Throughput: How Many Streams Help?")
print("━" * 68)
print()

print("  Scenario: multiple independent micro-batches, each with short kernels.")
print("  Can we keep the GPU busy with N parallel streams?")
print()

def throughput_n_streams(n_streams, kernel_us, launch_overhead_us, n_batches):
    """
    Each batch goes through its own stream (round-robin assignment).
    GPU can execute multiple streams in parallel up to SM capacity.
    """
    # Simplified model: up to 4 streams can run simultaneously on modern GPUs
    # (limited by work partitioner / MPS).
    MAX_CONCURRENT = min(4, n_streams)
    effective_kernel = kernel_us / MAX_CONCURRENT  # amortised across streams
    # But with launch overhead, very short kernels suffer
    effective_kernel_with_overhead = max(effective_kernel, launch_overhead_us * 0.5)
    total_us = n_batches * effective_kernel_with_overhead / n_streams * n_streams
    # Realistically: parallel throughput = n_batches * kernel_us / min(n_streams,max)
    actual_total = n_batches * kernel_us / MAX_CONCURRENT + launch_overhead_us * n_batches
    return actual_total

KERNEL_MICRO = 200.0    # µs — short kernel (typical small-batch inference)
LAUNCH_OH    = 15.0     # µs — Python dispatch + CUDA API overhead
N_MICRO      = 100

print(f"  Kernel: {KERNEL_MICRO} µs  Launch overhead: {LAUNCH_OH} µs  "
      f"N batches: {N_MICRO}")
print()
print(f"  {'N streams':>10}  {'Total ms':>10}  {'Throughput':>12}  "
      f"{'vs 1 stream':>12}  {'Note'}")
print("  " + "─" * 62)

t_baseline = None
for ns in [1, 2, 4, 8, 16, 32]:
    t_total = throughput_n_streams(ns, KERNEL_MICRO, LAUNCH_OH, N_MICRO)
    if t_baseline is None:
        t_baseline = t_total
    throughput = N_MICRO / (t_total / 1e6)
    sx = t_baseline / t_total
    if ns <= 4:
        note = "parallel benefit"
    else:
        note = "launch overhead dominates" if KERNEL_MICRO / ns < LAUNCH_OH else "SM saturation"
    print(f"  {ns:>10}  {t_total/1000:>10.2f}  {throughput:>10.0f}/s  "
          f"{sx:>11.2f}×  {note}")

print()
print("  Key insight: beyond 4 streams, GPU SM concurrency saturates.")
print("  With very short kernels (< 10 µs): CUDA Graphs eliminate launch overhead.")
print("  For LLM inference serving: separate streams per request up to 4-8.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · NVTX Annotation Profiler — Region Timing & Breakdown": {
        "description": (
            "Simulate NVTX range timing data for a transformer training loop. "
            "Compute per-region GPU time breakdown (forward, attention, FFN, "
            "backward, optimizer). Show how NVTX nesting produces a hierarchical "
            "timeline. Identify which sub-region dominates wall time. Generate "
            "the nsys stats equivalent text tables for CUDA API and kernel stats."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

print("=" * 68)
print("  NVTX ANNOTATION PROFILER — Region Timing & Breakdown")
print("=" * 68)
print()

np.random.seed(55)


# ─────────────────────────────────────────────────────────────────────
# Simulated NVTX data model
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NvtxRange:
    name:      str
    depth:     int
    start_us:  float
    end_us:    float
    children:  List = field(default_factory=list)
    gpu_us:    float = 0.0   # GPU time attributed to this range

    @property
    def cpu_us(self):
        return self.end_us - self.start_us

@dataclass
class KernelRecord:
    name:    str
    nvtx:    str
    gpu_us:  float
    count:   int


# ─────────────────────────────────────────────────────────────────────
# Build synthetic transformer training timeline (3 iterations)
# ─────────────────────────────────────────────────────────────────────

def build_transformer_timeline(n_layers=12, seq_len=2048, n_iter=3):
    """
    Simulate GPU kernel times for a GPT-like transformer.
    Based on approximate H100 timings for 7B model, batch=4, seq=2048.
    """
    # Per-layer GPU kernel times (microseconds)
    attn_qkv_us   = 180.0    # QKV projection
    attn_score_us  = 220.0   # Q@K attention scores
    attn_softmax_us = 45.0   # softmax over seq_len
    attn_value_us  = 210.0   # scores @ V
    attn_proj_us   = 175.0   # output projection
    ffn_gate_us    = 340.0   # SwiGLU gate + up projection
    ffn_down_us    = 340.0   # down projection
    ln_us          = 25.0    # RMSNorm (2 per layer)
    residual_us    = 12.0    # add residual (2 per layer)

    # Backward is ~2× forward for transformer
    backward_scale = 2.1

    # Optimizer (Adam): ~3 elementwise passes over params
    n_params_gb    = 14.0    # 7B params × 2 bytes (bf16) = 14 GB
    optimizer_us   = n_params_gb / 3.35 * 1e6 / 3   # 3 passes at ~HBM speed

    timeline = []
    kernel_records = defaultdict(lambda: {'gpu_us': 0.0, 'count': 0, 'nvtx': ''})
    t = 0.0

    for it in range(n_iter):
        iter_start = t
        # -- Forward pass --
        fwd_start = t
        t += 5.0   # Python overhead

        for layer in range(n_layers):
            layer_start = t

            # LayerNorm 1
            ln1_start = t; t += ln_us
            k = 'rms_norm_kernel'; kernel_records[k]['gpu_us'] += ln_us; kernel_records[k]['count'] += 1; kernel_records[k]['nvtx'] = f'layer_{layer}/attn'

            # Attention
            attn_start = t
            for kname, dur in [('attn_qkv_proj', attn_qkv_us),
                                ('attn_bmm_qk',   attn_score_us),
                                ('flash_attn_softmax', attn_softmax_us),
                                ('attn_bmm_av',   attn_value_us),
                                ('attn_out_proj',  attn_proj_us)]:
                t += dur
                kernel_records[kname]['gpu_us'] += dur; kernel_records[kname]['count'] += 1
                kernel_records[kname]['nvtx'] = f'layer_{layer}/attn'

            # Residual
            t += residual_us
            kernel_records['elementwise_add']['gpu_us'] += residual_us; kernel_records['elementwise_add']['count'] += 1

            # LayerNorm 2
            t += ln_us
            kernel_records['rms_norm_kernel']['gpu_us'] += ln_us; kernel_records['rms_norm_kernel']['count'] += 1

            # FFN
            ffn_start = t
            for kname, dur in [('ffn_swiglu_gate', ffn_gate_us), ('ffn_down_proj', ffn_down_us)]:
                t += dur
                kernel_records[kname]['gpu_us'] += dur; kernel_records[kname]['count'] += 1
                kernel_records[kname]['nvtx'] = f'layer_{layer}/ffn'
            t += residual_us
            kernel_records['elementwise_add']['gpu_us'] += residual_us; kernel_records['elementwise_add']['count'] += 1

        fwd_end = t; t += 8.0
        fwd_gpu_us = fwd_end - fwd_start

        # -- Backward pass --
        bwd_start = t; t += 10.0
        bwd_gpu_us = fwd_gpu_us * backward_scale
        t += bwd_gpu_us
        bwd_end = t

        # -- Optimizer step --
        opt_start = t; t += 5.0
        t += optimizer_us
        opt_end = t; t += 5.0

        timeline.append({
            'iter': it,
            'iter_span_us': t - iter_start,
            'fwd_gpu_us':   fwd_gpu_us,
            'bwd_gpu_us':   bwd_gpu_us,
            'opt_gpu_us':   optimizer_us,
            'fwd_start':    fwd_start,
            'bwd_start':    bwd_start,
            'opt_start':    opt_start,
        })

        t += 15.0  # iteration boundary overhead

    return timeline, dict(kernel_records)


timeline, kernel_records = build_transformer_timeline(n_layers=12, n_iter=3)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Per-iteration NVTX region breakdown
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — NVTX Region Breakdown (nsys stats equivalent)")
print("━" * 68)
print()

avg_iter_us   = np.mean([t['iter_span_us'] for t in timeline])
avg_fwd_us    = np.mean([t['fwd_gpu_us']   for t in timeline])
avg_bwd_us    = np.mean([t['bwd_gpu_us']   for t in timeline])
avg_opt_us    = np.mean([t['opt_gpu_us']   for t in timeline])
total_gpu_avg = avg_fwd_us + avg_bwd_us + avg_opt_us

regions = [
    ("forward_pass",  avg_fwd_us),
    ("backward_pass", avg_bwd_us),
    ("optimizer_step", avg_opt_us),
]

print(f"  Model: GPT-7B-like  |  Layers: 12  |  Seq: 2048  |  Batch: 4")
print(f"  Hardware model: H100 SXM5 (approximate timings)")
print()
print(f"  {'NVTX Region':<22}  {'GPU Time':>10}  {'% Total':>8}  {'Bar'}")
print("  " + "─" * 62)

for name, gpu_us in sorted(regions, key=lambda x: -x[1]):
    pct = gpu_us / total_gpu_avg * 100
    bar = '█' * int(pct / 3)
    print(f"  {name:<22}  {gpu_us/1000:>8.1f}ms  {pct:>7.1f}%  {bar}")

print()
print(f"  Total GPU time per iter:  {total_gpu_avg/1000:.1f} ms")
print(f"  Iteration wall time:      {avg_iter_us/1000:.1f} ms")
print(f"  GPU utilisation (approx): {total_gpu_avg/avg_iter_us*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Forward pass breakdown — which sub-region dominates
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Forward Pass Sub-Region Breakdown (12 layers)")
print("━" * 68)
print()

sub_regions = {
    'attn_qkv_proj':       ('attention/qkv_proj',   180.0 * 12),
    'attn_bmm_qk':         ('attention/bmm_qk',     220.0 * 12),
    'flash_attn_softmax':  ('attention/softmax',     45.0 * 12),
    'attn_bmm_av':         ('attention/bmm_av',     210.0 * 12),
    'attn_out_proj':       ('attention/out_proj',   175.0 * 12),
    'ffn_swiglu_gate':     ('ffn/gate_up_proj',     340.0 * 12),
    'ffn_down_proj':       ('ffn/down_proj',        340.0 * 12),
    'rms_norm_kernel':     ('norm (all RMSNorm)',    25.0 * 12 * 2),
    'elementwise_add':     ('residual_add (all)',    12.0 * 12 * 2),
}

total_fwd_sub = sum(v[1] for v in sub_regions.values())
print(f"  {'Operation':<28}  {'GPU Time':>10}  {'%':>6}  {'Bar'}")
print("  " + "─" * 66)

for kname, (label, gpu_us) in sorted(sub_regions.items(), key=lambda x: -x[1][1]):
    pct = gpu_us / total_fwd_sub * 100
    bar = '█' * int(pct / 2)
    print(f"  {label:<28}  {gpu_us/1000:>8.1f}ms  {pct:>5.1f}%  {bar}")

print()
print(f"  Total accounted: {total_fwd_sub/1000:.1f} ms")
print()

# Attention vs FFN breakdown
attn_total = sum(v[1] for k, v in sub_regions.items() if 'attn' in k or 'softmax' in k)
ffn_total  = sum(v[1] for k, v in sub_regions.items() if 'ffn' in k)
print(f"  Attention ops total:  {attn_total/1000:.1f} ms  ({attn_total/total_fwd_sub*100:.0f}%)")
print(f"  FFN ops total:        {ffn_total/1000:.1f} ms  ({ffn_total/total_fwd_sub*100:.0f}%)")
print(f"  Attention is {attn_total/ffn_total:.2f}× more expensive than FFN for seq={2048}")
print(f"  (FlashAttention 2 scales as O(seq²); FFN scales as O(seq))")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: nsys stats — CUDA Kernel Statistics table
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — nsys stats: CUDA Kernel Statistics Table")
print("━" * 68)
print()
print("  [Simulated output of: nsys stats profile.nsys-rep]")
print()

n_iter = 3
kernel_summary = []
for kname, rec in kernel_records.items():
    total_us = rec['gpu_us'] * n_iter
    count    = rec['count']  * n_iter
    avg_us   = total_us / count if count > 0 else 0
    kernel_summary.append((kname, total_us, count, avg_us))

kernel_summary.sort(key=lambda x: -x[1])
total_kernel_us = sum(k[1] for k in kernel_summary)

print(f"  CUDA Kernel Statistics (top 10 by total GPU time, {n_iter} iterations)")
print()
print(f"  {'Time%':>6}  {'Total(ms)':>10}  {'Count':>7}  {'Avg(µs)':>9}  {'Kernel Name'}")
print("  " + "─" * 64)
for kname, total_us, count, avg_us in kernel_summary[:10]:
    pct = total_us / total_kernel_us * 100
    print(f"  {pct:>5.1f}%  {total_us/1000:>10.2f}  {count:>7}  "
          f"{avg_us:>9.1f}  {kname}")

print()
print("  Red flags to look for:")
print("    1. cudaMalloc / cudaFree in kernel stats → memory allocation in hot loop")
print("    2. ncclKernel* with high total time → AllReduce on critical path")
print("    3. memset / memcpy with high count → inefficient buffer management")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: NVTX nesting diagram
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — NVTX Nesting: Hierarchical Timeline Structure")
print("━" * 68)
print()

print("  Recommended NVTX nesting for transformer training:")
print()

structure = [
    (0, "iteration_42",             "outer loop — 1 per step"),
    (1, "data_to_device",           "H2D transfer — always annotate"),
    (1, "forward_pass",             "entire forward"),
    (2, "embedding_lookup",         "token embeddings"),
    (2, "transformer_layer_00",     "per-layer (optional, can skip for speed)"),
    (3, "attention",                "QKV + score + softmax + proj"),
    (3, "ffn",                      "gate + up + down projections"),
    (2, "transformer_layer_01",     "(repeat for all layers)"),
    (2, "lm_head",                  "vocabulary projection"),
    (1, "loss_computation",         "cross-entropy"),
    (1, "backward_pass",            "entire backward"),
    (1, "optimizer_step",           "Adam update + zero_grad"),
    (1, "logging",                  "loss.item() — moved here, OUTSIDE hot path"),
]

for depth, name, note in structure:
    indent = "    " * depth
    bar = "├── " if depth > 0 else ""
    print(f"  {indent}{bar}{name:<30}  # {note}")

print()
print("  GPU Timeline appearance (each row = one NVTX depth level):")
print()
print("  Depth 0  [=================== iteration_42 ===================]")
print("  Depth 1  [d2d][====== forward ======][loss][==== backward ====][opt]")
print("  Depth 2          [emb][layer_00]...[layer_11][lm_head]")
print("  Depth 3                [attn][ffn]")
print()
print("  Key rule: annotate at the GRANULARITY you want to optimise at.")
print("  Too fine = profiling overhead. Too coarse = no actionable insight.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Bottleneck Pattern Detector — Six System-Level Bottlenecks": {
        "description": (
            "Implement a profiling data analyser that detects each of the six "
            "canonical system-level bottlenecks from nsys timeline statistics. "
            "Score each bottleneck type, rank by wall-clock impact, and produce "
            "a prioritised fix list. Model the expected speedup from each fix. "
            "Simulate before/after timelines to verify improvement predictions."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict

print("=" * 68)
print("  BOTTLENECK PATTERN DETECTOR — Six System-Level Bottleneck Types")
print("=" * 68)
print()

np.random.seed(13)


# ─────────────────────────────────────────────────────────────────────
# Bottleneck definitions
# ─────────────────────────────────────────────────────────────────────

BOTTLENECKS = {
    "cpu_launch_gap": {
        "display": "CPU Launch Overhead",
        "symptom": "Gaps 5–50 µs between consecutive kernels on compute stream",
        "metric":  "avg_gap_between_kernels_us",
        "threshold_us": 10.0,
        "fix": "torch.compile / CUDA Graphs (module 35)",
        "expected_speedup_fn": lambda gap_us, kernel_us:
            (gap_us + kernel_us) / (min(gap_us, 2.0) + kernel_us),
    },
    "blocking_memcpy": {
        "display": "Blocking Memory Copy",
        "symptom": "H2D/D2H on stream 0 or pageable memory stalling compute",
        "metric":  "frac_time_in_blocking_copy",
        "threshold_us": 0.05,
        "fix": "cudaMemcpyAsync + pinned memory + dedicated copy stream",
        "expected_speedup_fn": lambda copy_frac, kernel_frac:
            1.0 / max(kernel_frac, 1.0 - copy_frac),
    },
    "unnecessary_sync": {
        "display": "Unnecessary Synchronisation",
        "symptom": "cudaDeviceSynchronize / loss.item() inside hot loop",
        "metric":  "sync_calls_per_iter",
        "threshold_us": 1.0,
        "fix": "Move .item() outside loop; use deferred metric accumulation",
        "expected_speedup_fn": lambda sync_us_total, iter_us:
            iter_us / (iter_us - sync_us_total * 0.85),
    },
    "cpu_bound_dataloader": {
        "display": "CPU-Bound DataLoader",
        "symptom": "GPU utilisation drops periodically; CPU decode visible",
        "metric":  "gpu_idle_frac_correlated_with_cpu",
        "threshold_us": 0.10,
        "fix": "DALI pipeline / increase num_workers / pin_memory=True",
        "expected_speedup_fn": lambda cpu_gap_us, iter_us:
            iter_us / (iter_us - cpu_gap_us * 0.90),
    },
    "false_stream_serial": {
        "display": "False Stream Serialisation",
        "symptom": "Multiple streams exist but execute sequentially",
        "metric":  "stream_overlap_fraction",
        "threshold_us": 0.05,
        "fix": "Remove cudaDeviceSynchronize; use per-stream or event sync",
        "expected_speedup_fn": lambda overlap_possible_us, actual_overlap_us:
            1.0 / max(0.01, 1.0 - (overlap_possible_us - actual_overlap_us)
                      / (overlap_possible_us + actual_overlap_us) * 0.6),
    },
    "allocator_in_loop": {
        "display": "Allocator in Hot Loop",
        "symptom": "cudaMalloc/Free calls inside iteration causing GPU stalls",
        "metric":  "alloc_time_us_per_iter",
        "threshold_us": 100.0,
        "fix": "Pre-allocate all buffers; use static shapes for allocator reuse",
        "expected_speedup_fn": lambda alloc_us, iter_us:
            iter_us / (iter_us - alloc_us * 0.95),
    },
}


# ─────────────────────────────────────────────────────────────────────
# Build scenario profiles — each has a different dominant bottleneck
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ProfileScenario:
    name:             str
    iter_wall_us:     float
    kernel_us:        float
    avg_kernel_gap_us: float
    blocking_copy_us: float
    sync_us_per_iter: float
    cpu_gap_us:       float
    alloc_us:         float
    n_streams:        int
    actual_overlap_us: float
    possible_overlap_us: float

scenarios = [
    ProfileScenario(
        "Inference server (small batches)",
        iter_wall_us=12000, kernel_us=300, avg_kernel_gap_us=35.0,
        blocking_copy_us=0, sync_us_per_iter=0, cpu_gap_us=200,
        alloc_us=0, n_streams=2, actual_overlap_us=0, possible_overlap_us=0),
    ProfileScenario(
        "Training loop with loss.item()",
        iter_wall_us=18000, kernel_us=3000, avg_kernel_gap_us=12.0,
        blocking_copy_us=0, sync_us_per_iter=2800, cpu_gap_us=100,
        alloc_us=0, n_streams=1, actual_overlap_us=0, possible_overlap_us=0),
    ProfileScenario(
        "Vision model, pageable memory",
        iter_wall_us=25000, kernel_us=5000, avg_kernel_gap_us=8.0,
        blocking_copy_us=8500, sync_us_per_iter=100, cpu_gap_us=500,
        alloc_us=0, n_streams=1, actual_overlap_us=0, possible_overlap_us=0),
    ProfileScenario(
        "LLM training with dynamic shapes",
        iter_wall_us=30000, kernel_us=8000, avg_kernel_gap_us=6.0,
        blocking_copy_us=0, sync_us_per_iter=200, cpu_gap_us=300,
        alloc_us=5200, n_streams=1, actual_overlap_us=0, possible_overlap_us=0),
    ProfileScenario(
        "DDP with false stream serialisation",
        iter_wall_us=22000, kernel_us=4500, avg_kernel_gap_us=9.0,
        blocking_copy_us=0, sync_us_per_iter=150, cpu_gap_us=200,
        alloc_us=0, n_streams=3, actual_overlap_us=100, possible_overlap_us=6000),
]


def detect_bottlenecks(s: ProfileScenario):
    """Return list of (bottleneck_key, severity_us, expected_speedup) sorted by impact."""
    findings = []

    # 1. CPU launch gaps
    n_kernels_per_iter = max(1, s.kernel_us // 300)  # estimate ~300µs avg kernel
    total_gap_us = s.avg_kernel_gap_us * n_kernels_per_iter
    if s.avg_kernel_gap_us > BOTTLENECKS['cpu_launch_gap']['threshold_us']:
        sx = BOTTLENECKS['cpu_launch_gap']['expected_speedup_fn'](
            s.avg_kernel_gap_us, 300.0)
        findings.append(('cpu_launch_gap', total_gap_us, sx))

    # 2. Blocking memcpy
    if s.blocking_copy_us > 0:
        frac = s.blocking_copy_us / s.iter_wall_us
        sx = BOTTLENECKS['blocking_memcpy']['expected_speedup_fn'](
            frac, s.kernel_us / s.iter_wall_us)
        findings.append(('blocking_memcpy', s.blocking_copy_us, sx))

    # 3. Unnecessary sync
    if s.sync_us_per_iter > BOTTLENECKS['unnecessary_sync']['threshold_us']:
        sx = BOTTLENECKS['unnecessary_sync']['expected_speedup_fn'](
            s.sync_us_per_iter, s.iter_wall_us)
        findings.append(('unnecessary_sync', s.sync_us_per_iter, sx))

    # 4. CPU-bound dataloader
    if s.cpu_gap_us > s.iter_wall_us * BOTTLENECKS['cpu_bound_dataloader']['threshold_us']:
        sx = BOTTLENECKS['cpu_bound_dataloader']['expected_speedup_fn'](
            s.cpu_gap_us, s.iter_wall_us)
        findings.append(('cpu_bound_dataloader', s.cpu_gap_us, sx))

    # 5. False stream serialisation
    overlap_gap = s.possible_overlap_us - s.actual_overlap_us
    if overlap_gap > s.iter_wall_us * BOTTLENECKS['false_stream_serial']['threshold_us']:
        sx = BOTTLENECKS['false_stream_serial']['expected_speedup_fn'](
            s.possible_overlap_us, s.actual_overlap_us)
        findings.append(('false_stream_serial', overlap_gap, sx))

    # 6. Allocator in loop
    if s.alloc_us > BOTTLENECKS['allocator_in_loop']['threshold_us']:
        sx = BOTTLENECKS['allocator_in_loop']['expected_speedup_fn'](
            s.alloc_us, s.iter_wall_us)
        findings.append(('allocator_in_loop', s.alloc_us, sx))

    findings.sort(key=lambda x: -x[1])
    return findings


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Bottleneck detection report per scenario
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Bottleneck Detection Report: 5 Real-World Scenarios")
print("━" * 68)
print()

for s in scenarios:
    findings = detect_bottlenecks(s)
    gpu_util = s.kernel_us / s.iter_wall_us * 100

    print(f"  ┌─ Scenario: {s.name}")
    print(f"  │  Iter wall: {s.iter_wall_us/1000:.1f}ms  "
          f"Kernel time: {s.kernel_us/1000:.1f}ms  "
          f"GPU util: {gpu_util:.0f}%")

    if not findings:
        print(f"  │  No significant bottlenecks detected. ✅")
    else:
        print(f"  │  Detected bottlenecks (ranked by impact):")
        for rank, (bkey, severity_us, sx) in enumerate(findings):
            b = BOTTLENECKS[bkey]
            print(f"  │    #{rank+1}  [{b['display']}]")
            print(f"  │        Severity: {severity_us/1000:.1f}ms/iter  "
                  f"({severity_us/s.iter_wall_us*100:.0f}% of wall time)")
            print(f"  │        Expected speedup from fix: {sx:.2f}×")
            print(f"  │        Fix: {b['fix']}")
            print(f"  │        nsys symptom: {b['symptom']}")

    print(f"  └{'─' * 64}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Multi-fix cumulative improvement model
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Cumulative Improvement: Fixing in Priority Order")
print("━" * 68)
print()

target_scenario = scenarios[2]  # Vision model (worst case, multiple bottlenecks)
findings = detect_bottlenecks(target_scenario)

# Add extra bottleneck for richness
findings.append(('cpu_launch_gap',
                 target_scenario.avg_kernel_gap_us * 16, 1.22))
findings.sort(key=lambda x: -x[1])

print(f"  Scenario: {target_scenario.name}")
print(f"  Starting wall time: {target_scenario.iter_wall_us/1000:.1f} ms/iter")
print()
print(f"  {'Fix order':<5}  {'Bottleneck':<28}  {'Before':>9}  {'After':>9}  "
      f"{'This fix':>9}  {'Cumul'}")
print("  " + "─" * 68)

current_us = float(target_scenario.iter_wall_us)
cumul_sx   = 1.0
for rank, (bkey, severity_us, sx) in enumerate(findings):
    before_us  = current_us
    # Apply fix: reduce wall time by the severity (with some diminishing returns)
    reduction  = severity_us * 0.85
    after_us   = max(current_us - reduction, target_scenario.kernel_us * 1.05)
    this_sx    = before_us / after_us
    current_us = after_us
    cumul_sx   = target_scenario.iter_wall_us / current_us

    print(f"  {rank+1:<5}  {BOTTLENECKS[bkey]['display']:<28}  "
          f"{before_us/1000:>7.1f}ms  {after_us/1000:>7.1f}ms  "
          f"{this_sx:>8.2f}×  {cumul_sx:.2f}×")

print()
print(f"  Final wall time:   {current_us/1000:.1f} ms/iter  "
      f"(cumulative speedup: {cumul_sx:.2f}×)")
print(f"  Theoretical minimum (kernel only): {target_scenario.kernel_us/1000:.1f} ms/iter  "
      f"({target_scenario.iter_wall_us/target_scenario.kernel_us:.1f}× headroom)")
print()
print("  LESSON: System-level fixes (nsys) often give 2–5× speedup BEFORE")
print("          any kernel-level optimisation (ncu). Always profile system first.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Multi-GPU DDP Profiler — NCCL, AllReduce Overlap & NVLink": {
        "description": (
            "Simulate nsys timeline data for DDP training across multiple GPUs. "
            "Model NCCL AllReduce timing for different collective algorithms "
            "(ring, tree) and communication substrates (NVLink, PCIe). "
            "Show whether gradient communication is on the critical path. "
            "Analyse bucketed gradient AllReduce overlap with backward pass. "
            "Compute effective scaling efficiency vs ideal linear speedup."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Dict

print("=" * 68)
print("  MULTI-GPU DDP PROFILER — NCCL, AllReduce Overlap & Scaling")
print("=" * 68)
print()

np.random.seed(88)


# ─────────────────────────────────────────────────────────────────────
# Hardware models
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Interconnect:
    name:          str
    bw_gbs:        float   # unidirectional GB/s (per link, total)
    latency_us:    float   # base latency per hop (µs)
    topology:      str     # ring | nvswitch | hybrid

GPU_CONFIGS = {
    "H100_SXM5_8x (NVLink 4.0)": Interconnect(
        "NVLink 4.0", bw_gbs=900.0,  latency_us=1.0,  topology="nvswitch"),
    "A100_SXM4_8x (NVLink 3.0)": Interconnect(
        "NVLink 3.0", bw_gbs=600.0,  latency_us=1.5,  topology="nvswitch"),
    "A100_PCIe_8x (PCIe Gen4)":  Interconnect(
        "PCIe Gen4",  bw_gbs=32.0,   latency_us=8.0,  topology="ring"),
    "4x GPU (PCIe Gen3)":         Interconnect(
        "PCIe Gen3",  bw_gbs=16.0,   latency_us=12.0, topology="ring"),
}


# ─────────────────────────────────────────────────────────────────────
# NCCL AllReduce timing models
# ─────────────────────────────────────────────────────────────────────

def ring_allreduce_us(size_bytes, n_gpus, bw_gbs, latency_us):
    """
    Ring AllReduce: 2*(N-1)/N * size / bw + 2*(N-1) * latency
    Bandwidth-optimal for large messages.
    """
    size_gb = size_bytes / 1e9
    bw_term = 2 * (n_gpus - 1) / n_gpus * size_gb / bw_gbs * 1e6
    lat_term = 2 * (n_gpus - 1) * latency_us
    return bw_term + lat_term

def tree_allreduce_us(size_bytes, n_gpus, bw_gbs, latency_us):
    """
    Tree (recursive doubling / butterfly): 2*log2(N)*latency + 2*size/bw
    Better for small messages (latency-dominated).
    """
    size_gb = size_bytes / 1e9
    bw_term = 2 * size_gb / bw_gbs * 1e6
    lat_term = 2 * math.log2(max(2, n_gpus)) * latency_us
    return bw_term + lat_term

def nccl_allreduce_us(size_bytes, n_gpus, interconnect):
    """NCCL chooses ring or tree based on message size and topology."""
    ring_t = ring_allreduce_us(size_bytes, n_gpus, interconnect.bw_gbs, interconnect.latency_us)
    tree_t = tree_allreduce_us(size_bytes, n_gpus, interconnect.bw_gbs, interconnect.latency_us)
    # NCCL heuristic: use tree for small messages, ring for large
    threshold_bytes = interconnect.latency_us * interconnect.bw_gbs * 1e9 / 2 / 1e6
    if size_bytes < threshold_bytes:
        return tree_t, "tree"
    else:
        return ring_t, "ring"


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: AllReduce time vs message size across interconnects
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — NCCL AllReduce Latency: Message Size × Interconnect")
print("━" * 68)
print()

N_GPUS = 8
sizes_bytes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000,
               100_000_000, 1_000_000_000, 7_000_000_000]

print(f"  N_GPUS = {N_GPUS}  |  Comparing {len(GPU_CONFIGS)} interconnect configs")
print()
print(f"  {'Msg Size':>14}", end="")
for cfg_name in GPU_CONFIGS:
    short = cfg_name[:18]
    print(f"  {short:>18}", end="")
print()
print("  " + "─" * (14 + 20 * len(GPU_CONFIGS)))

for size in sizes_bytes:
    if size >= 1e9:
        size_str = f"{size/1e9:.0f} GB"
    elif size >= 1e6:
        size_str = f"{size/1e6:.0f} MB"
    elif size >= 1e3:
        size_str = f"{size/1e3:.0f} KB"
    else:
        size_str = f"{size} B"

    print(f"  {size_str:>14}", end="")
    for cfg_name, ic in GPU_CONFIGS.items():
        t_us, algo = nccl_allreduce_us(size, N_GPUS, ic)
        if t_us >= 1e6:
            t_str = f"{t_us/1e6:.1f}s"
        elif t_us >= 1000:
            t_str = f"{t_us/1000:.1f}ms"
        else:
            t_str = f"{t_us:.0f}µs"
        print(f"  {t_str:>16}{algo[0]:>2}", end="")
    print()

print()
print("  r=ring (bw-optimal for large),  t=tree (latency-optimal for small)")
print(f"  H100 NVLink 4.0 vs PCIe Gen4 speedup at 7GB:  "
      f"{ring_allreduce_us(7e9, 8, 32.0, 8.0) / ring_allreduce_us(7e9, 8, 900.0, 1.0):.0f}×")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: DDP bucketed AllReduce — overlap with backward pass
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — DDP Bucket Strategy: AllReduce Overlap with Backward")
print("━" * 68)
print()

print("  PyTorch DDP divides gradients into buckets (default ~25 MB each).")
print("  AllReduce starts as soon as each bucket is filled during backward.")
print("  Goal: overlap AllReduce time with remaining backward computation.")
print()

@dataclass
class DdpScenario:
    name:         str
    n_params_gb:  float
    bwd_us:       float
    bucket_mb:    float
    ic:           Interconnect
    n_gpus:       int

    @property
    def n_buckets(self):
        return max(1, int(self.n_params_gb * 1024 / self.bucket_mb))

    @property
    def bytes_per_bucket(self):
        return self.bucket_mb * 1024 * 1024

    def simulate_ddp_timeline(self):
        """
        Simulate bucket-by-bucket AllReduce overlap with backward.
        Buckets are filled in reverse order (last layer first).
        """
        n_b = self.n_buckets
        # Backward fills buckets at even intervals
        bwd_per_bucket_us = self.bwd_us / n_b

        t_bwd_done  = 0.0
        t_ar_done   = 0.0
        ar_times    = []

        for i in range(n_b):
            # When this bucket is filled
            t_bucket_ready = (i + 1) * bwd_per_bucket_us
            # AllReduce can start when bucket is ready AND previous AR is done
            t_ar_start     = max(t_bucket_ready, t_ar_done)
            ar_us, _       = nccl_allreduce_us(self.bytes_per_bucket,
                                               self.n_gpus, self.ic)
            t_ar_end       = t_ar_start + ar_us
            ar_times.append((t_ar_start, t_ar_end, ar_us))
            t_ar_done = t_ar_end

        t_bwd_done = self.bwd_us
        t_total    = max(t_bwd_done, t_ar_done)

        # AR on critical path = time after backward finishes
        ar_tail_us = max(0.0, t_ar_done - t_bwd_done)

        return t_total, ar_tail_us, ar_times

ddp_scenarios = [
    DdpScenario("GPT-7B, H100 NVLink, 8 GPU",
                7.0, 35000, 25.0, GPU_CONFIGS["H100_SXM5_8x (NVLink 4.0)"], 8),
    DdpScenario("GPT-7B, A100 NVLink, 8 GPU",
                7.0, 35000, 25.0, GPU_CONFIGS["A100_SXM4_8x (NVLink 3.0)"], 8),
    DdpScenario("GPT-7B, PCIe Gen4, 8 GPU",
                7.0, 35000, 25.0, GPU_CONFIGS["A100_PCIe_8x (PCIe Gen4)"], 8),
    DdpScenario("GPT-7B, PCIe, 1MB buckets",
                7.0, 35000, 1.0, GPU_CONFIGS["A100_PCIe_8x (PCIe Gen4)"], 8),
    DdpScenario("ResNet-50, A100 NVLink, 8 GPU",
                0.1, 8000,  25.0, GPU_CONFIGS["A100_SXM4_8x (NVLink 3.0)"], 8),
]

print(f"  {'Scenario':<36}  {'Bwd ms':>7}  {'AR total ms':>11}  "
      f"{'AR tail ms':>11}  {'Overhead%':>10}  {'Status'}")
print("  " + "─" * 82)

for s in ddp_scenarios:
    total_ar_us = sum(nccl_allreduce_us(s.bytes_per_bucket, s.n_gpus, s.ic)[0]
                      for _ in range(s.n_buckets))
    t_total, ar_tail_us, _ = s.simulate_ddp_timeline()
    overhead_pct = ar_tail_us / s.bwd_us * 100
    status = "✅ hidden" if ar_tail_us < s.bwd_us * 0.05 else \
             "⚠ partial" if ar_tail_us < s.bwd_us * 0.20 else "❌ critical path"

    print(f"  {s.name:<36}  {s.bwd_us/1000:>7.1f}  {total_ar_us/1000:>11.1f}  "
          f"{ar_tail_us/1000:>11.1f}  {overhead_pct:>9.1f}%  {status}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: DDP scaling efficiency
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — DDP Scaling Efficiency: 1 → 64 GPUs")
print("━" * 68)
print()

BASE_BWD_US  = 35000.0    # single-GPU backward time (µs)
MODEL_GB     = 7.0
BUCKET_MB    = 25.0

print(f"  Model: 7B params  |  Backward: {BASE_BWD_US/1000:.0f}ms single GPU")
print()
print(f"  {'N GPUs':>8}  {'NVLink time':>13}  {'PCIe time':>12}  "
      f"{'NVLink eff%':>12}  {'PCIe eff%':>11}  {'Ideal'}")
print("  " + "─" * 68)

for n_gpus in [1, 2, 4, 8, 16, 32, 64]:
    if n_gpus == 1:
        nvlink_total_us = BASE_BWD_US
        pcie_total_us   = BASE_BWD_US
        nvlink_eff = pcie_eff = 100.0
    else:
        # NVLink DDP
        s_nv = DdpScenario("", MODEL_GB, BASE_BWD_US, BUCKET_MB,
                           GPU_CONFIGS["A100_SXM4_8x (NVLink 3.0)"], n_gpus)
        nvlink_total_us, _, _ = s_nv.simulate_ddp_timeline()

        # PCIe DDP
        ic_pcie = GPU_CONFIGS["A100_PCIe_8x (PCIe Gen4)"]
        n_buckets = max(1, int(MODEL_GB * 1024 / BUCKET_MB))
        s_pc = DdpScenario("", MODEL_GB, BASE_BWD_US, BUCKET_MB, ic_pcie, n_gpus)
        pcie_total_us, _, _ = s_pc.simulate_ddp_timeline()

        ideal_us    = BASE_BWD_US / n_gpus
        nvlink_eff  = ideal_us / nvlink_total_us * n_gpus * 100
        pcie_eff    = ideal_us / pcie_total_us   * n_gpus * 100

    ideal_speedup = float(n_gpus)
    print(f"  {n_gpus:>8}  {nvlink_total_us/1000:>11.1f}ms  "
          f"{pcie_total_us/1000:>10.1f}ms  "
          f"{nvlink_eff if n_gpus>1 else 100.0:>10.1f}%  "
          f"{pcie_eff if n_gpus>1 else 100.0:>9.1f}%  {ideal_speedup:.0f}×")

print()
print("  nsys diagnostic: if scaling efficiency < 85% on NVLink,")
print("    look for: (1) uneven bucket sizes, (2) unnecessary barriers,")
print("    (3) gradient hooks adding sync points, (4) non-overlappable ops.")
print()
print("  ZeRO Stage 2 / FSDP alternative: instead of AllReduce,")
print("    scatter-reduce + all-gather → same bandwidth but 1/N memory per GPU.")
print("    Each GPU holds 1/N of parameters + gradients → memory-efficient scaling.")
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