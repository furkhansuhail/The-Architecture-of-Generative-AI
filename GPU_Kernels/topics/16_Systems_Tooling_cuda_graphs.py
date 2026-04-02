"""
CUDA Graphs — Graph Capture, Replay / Update & Conditional Nodes (CUDA 12+)
============================================================================

Every CUDA kernel launch carries overhead: the CPU must write to the GPU's
command queue, the driver must validate arguments, and the hardware work
distributor must schedule thread blocks. For a single large GEMM this
5–50 µs overhead is irrelevant — the kernel runs for milliseconds. But
modern LLM inference executes hundreds of small kernels per token, and the
cumulative launch overhead can exceed the actual GPU compute time.

CUDA Graphs solve this by eliminating repeated launch overhead entirely.
A graph is a pre-compiled representation of a kernel sequence — its nodes
are operations (kernels, memcopies, memsets), its edges are dependencies.
Once captured, the entire graph replays with a single lightweight API call
whose overhead is under 1 µs regardless of how many operations the graph
contains.

Three capabilities define CUDA Graphs for production use:

    GRAPH CAPTURE: record a sequence of CUDA operations into a graph.
        Stream capture watches everything submitted to a stream and
        assembles it into a graph. One-time cost; amortised over replays.

    GRAPH REPLAY AND UPDATE: execute the captured graph.
        cudaGraphLaunch: replay with sub-1-µs overhead.
        cudaGraphExecUpdate: patch kernel arguments (pointers, scalars)
        without rebuilding the graph — critical for inference where tensor
        addresses change each step while the computation graph is fixed.

    CONDITIONAL NODES (CUDA 12.4+): embed GPU-side control flow.
        IF nodes: execute a subgraph only if a GPU-side condition is true.
        WHILE nodes: loop over a subgraph until a GPU-side flag is cleared.
        Enables speculative decoding accept/reject, early-exit inference,
        and adaptive computation without any CPU round-trips.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "CUDA Graphs — Graph Capture, Replay / Update & Conditional Nodes (CUDA 12+)"
DISPLAY_NAME = "16 · CUDA Graphs"
ICON         = "📊"
SUBTITLE     = "Graph Capture · cudaGraphExecUpdate · Conditional IF/WHILE · Launch Overhead"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE LAUNCH OVERHEAD PROBLEM: WHY GRAPHS EXIST

### The Anatomy of a Kernel Launch

    When a CPU thread calls cudaLaunchKernel (or submits a kernel from a
    CUDA stream), the following sequence unfolds before the first GPU
    instruction executes:

    CPU SIDE (in the calling thread):
        1. Look up the kernel function handle in the CUDA driver's cache.
        2. Validate grid/block dimensions and shared memory size.
        3. Copy kernel arguments (pointers, scalars) to a pinned staging buffer.
        4. Write a launch descriptor to the stream's ring buffer.
           The ring buffer is a CPU-accessible, pinned-memory queue.

    GPU SIDE (hardware work distributor, asynchronous to CPU):
        5. Read the launch descriptor from the ring buffer.
        6. Wait for all declared stream dependencies (cudaStreamWaitEvent).
        7. Check SM resource availability (registers, SMEM).
        8. Schedule thread blocks onto available SMs.
        9. Initialise warp state. First instruction executes.

    TOTAL LATENCY from cudaLaunchKernel return to first GPU instruction:
        Driver overhead (steps 1–4): ~1–3 µs (in steady state with warm cache).
        GPU scheduling (steps 5–9): ~3–5 µs (hardware round-trip latency).
        Total observed: 4–8 µs on A100/H100 with the conventional path.

    With CUDA GRAPHS (steps 1–4 done once at capture time):
        Graph launch (cudaGraphLaunch): writes ONE descriptor → GPU.
        GPU replays the pre-compiled sequence.
        Total observed: 0.3–0.8 µs per graph launch.
        IMPROVEMENT: 8–25× lower latency per "batch" of kernels.

### When Launch Overhead Dominates

    THRESHOLD: launch overhead > ~10% of kernel execution time.
        Kernel duration > 50 µs: launch overhead ~10% → CUDA Graphs mildly useful.
        Kernel duration 10–50 µs: launch overhead 10–50% → CUDA Graphs very useful.
        Kernel duration < 10 µs: launch overhead > 50% → CUDA Graphs essential.

    LLM DECODE STEP analysis (Llama-2-7B, H100, batch=1):
        Per-token decode: ~240 kernel launches (32 layers × ~7.5 kernels/layer)
        Without CUDA Graphs: 240 × 5 µs = 1.2 ms pure overhead
        Target kernel time: ~2–5 ms total (memory-bound at batch=1)
        Overhead fraction WITHOUT graphs: 20–40%!
        With CUDA Graphs: 1 graph launch × 0.5 µs = 0.5 µs overhead (negligible)

    VISION MODEL INFERENCE analysis (ResNet-50, H100, batch=1):
        ~100 kernel launches per inference step.
        Each kernel: 50–200 µs (larger compute than LLM decode).
        Without graphs: 100 × 5 µs = 0.5 ms overhead vs ~10–20 ms total (~3–5%).
        Graphs beneficial but not as critical as for small-batch LLM.

### The Three Scenarios Where CUDA Graphs Are Essential

    SCENARIO 1 — Small-batch LLM inference (batch=1..16):
        Memory-bound GEMMs are fast; launch overhead is proportionally large.
        Static graph captures the entire decode step (fixed batch/sequence shapes).
        PyTorch 2.0+ cudagraphs mode enables this automatically.

    SCENARIO 2 — High-frequency iterative algorithms:
        Iterative solvers (PCG, GMRES): hundreds of small kernels per iteration.
        Molecular dynamics: small integration steps, many launches per timestep.
        Reinforcement learning: small model, many environment steps per batch.

    SCENARIO 3 — Low-latency serving with SLA constraints:
        Real-time audio/video processing: 10–30 ms frame budgets.
        Robotics inference: sub-10 ms latency required.
        Trading systems: µs-level latency matters.


##### PART 2 — GRAPH CAPTURE: RECORDING THE COMPUTATION

### Stream Capture Mode

    CUDA's primary graph capture mechanism intercepts all operations
    submitted to a designated stream while capture is active.

    THREE CAPTURE MODES:
        cudaStreamCaptureModeGlobal (default):
            ALL operations that synchronise with the capture stream are recorded.
            Operations on other streams that have event dependencies on the
            capture stream are "pulled in" to the graph.
            Most comprehensive. Use for complex multi-stream pipelines.

        cudaStreamCaptureModeThreadLocal:
            Only operations from the CALLING THREAD are captured.
            Operations from other threads on the same stream are NOT captured.
            Use for: single-threaded capture of explicit graph structure.

        cudaStreamCaptureModeRelaxed:
            Allows event-based synchronisation with non-capturing streams.
            Use for: partial capture where some synchronisation is intentionally
            excluded from the graph (e.g., CPU-side events for profiling).

    MINIMAL CAPTURE EXAMPLE (C API):
        cudaGraph_t graph;
        cudaGraphExec_t exec;

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // All operations submitted to 'stream' here are recorded:
        myKernel_A<<<grid, block, 0, stream>>>(d_a, d_b);
        myKernel_B<<<grid, block, 0, stream>>>(d_b, d_c);
        cudaMemcpyAsync(d_out, d_c, size, cudaMemcpyDeviceToDevice, stream);

        cudaStreamEndCapture(stream, &graph);   // build the graph
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);   // exec holds its own internal copy

    AFTER CAPTURE: the graph is a static representation of the recorded ops.
    The stream itself is NOT advanced during capture — operations are recorded,
    not executed. First REAL execution happens via cudaGraphLaunch.

### What Capture Records

    Each recorded operation becomes a GRAPH NODE:
        KERNEL NODES:    grid dims, block dims, kernel function, arguments.
        MEMCPY NODES:    source/dest pointers, size, copy direction.
        MEMSET NODES:    target pointer, value, size.
        HOST NODES:      CPU-side callbacks (cudaHostFn_t).
        CHILD GRAPH NODES: nested subgraphs (enables modular composition).
        WAIT EVENT NODES:  cudaStreamWaitEvent calls.
        RECORD EVENT NODES: cudaEventRecord calls.

    EDGES (dependencies): automatically derived from the stream ordering
    and any cross-stream event waits recorded during capture.
    The graph represents a DAG — directed acyclic graph of operations.

### What Cannot Be Captured

    BLOCKING OPERATIONS: synchronous API calls that block the CPU.
        cudaDeviceSynchronize() → forbidden during capture.
        cudaStreamSynchronize() → forbidden (use events instead).
        cudaMemcpy (synchronous) → use cudaMemcpyAsync.

    DYNAMIC MEMORY ALLOCATION: cudaMalloc/cudaFree inside the captured region.
        All allocations must happen BEFORE capture starts.
        Exception: cudaMallocAsync (stream-ordered allocator) CAN be captured
        in CUDA 11.2+.

    CPU-GPU DATA TRANSFERS with cudaMemcpy (synchronous form):
        Use cudaMemcpyAsync with the capture stream instead.

    PEER-TO-PEER (multi-GPU) operations: limited support, check documentation.

### PyTorch cudagraph() Integration

    PyTorch 2.0+ provides a high-level wrapper:

        # Method 1: torch.cuda.make_graphed_callables (eager, automatic)
        graphed_model = torch.cuda.make_graphed_callables(model, sample_inputs)
        # graphed_model() is now executed via a CUDA graph internally.

        # Method 2: manual capture (more control)
        stream = torch.cuda.Stream()
        graph  = torch.cuda.CUDAGraph()

        # Warmup (populates allocator cache, avoids capture of alloc ops)
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = model(static_input)

        # Capture
        with torch.cuda.graph(graph, stream=stream):
            static_output = model(static_input)  # static_input is a fixed tensor

        # Replay (in inference loop):
        static_input.copy_(new_input)  # update the static buffer in-place
        graph.replay()                 # launches the graph
        result = static_output.clone() # read result


##### PART 3 — GRAPH REPLAY AND UPDATE

### Graph Instantiation

    cudaGraphInstantiate transforms the graph template into an EXECUTABLE
    graph (cudaGraphExec_t). This step:
        1. Validates all kernel handles and argument types.
        2. Resolves stream dependencies into GPU timeline ordering.
        3. Compiles the optimised GPU command sequence.
        4. Allocates any internal state the graph needs at runtime.

    COST: similar to a few kernel launches (microseconds to low milliseconds).
    Done ONCE. Amortised over every subsequent graph replay.

    UPDATED API (CUDA 12):
        cudaGraphInstantiateParams params = {
            .flags = cudaGraphInstantiateFlagAutoFreeOnLaunch,
            // CUDA 12.3+: cudaGraphInstantiateFlagUploadWithoutLaunch
        };
        cudaGraphInstantiateWithParams(&exec, graph, &params);

### Graph Replay

    LAUNCH: cudaGraphLaunch(exec, stream)
        Submits the entire pre-compiled graph for execution on 'stream'.
        The stream is used for dependency purposes — the graph itself runs
        independently but other work submitted to the stream after this call
        will wait for the graph to complete.

    OVERHEAD: measured at 0.3–0.8 µs on A100/H100 (much less for newer CUDA).
    The GPU work distributor receives ONE command: "execute graph exec_id".
    Internally, the GPU's graph engine replays the compiled command sequence.

    GRAPH vs STREAM SEMANTICS:
        A graph launch is ordered in the stream: work submitted BEFORE the launch
        completes before the graph starts; work submitted AFTER waits for the graph.
        Two consecutive graph launches on the same stream execute sequentially.
        Graph launches on DIFFERENT streams can overlap (if resources allow).

### cudaGraphExecUpdate: Patching Without Re-instantiation

    A critical limitation of CUDA Graphs: once instantiated, the graph
    structure (node count, edges, kernel function handles) is FIXED.

    BUT: kernel arguments (pointers, constants) CAN be updated in-place
    without re-instantiating:

        cudaGraphExecUpdateResult result;
        cudaGraphNode_t errorNode;
        cudaGraphExecUpdate(exec, new_graph, &errorNode, &result);

    This is 10–100× faster than re-instantiation for simple argument changes.

    WHAT CAN BE UPDATED (cudaGraphExecUpdate):
        Kernel argument pointers (e.g., new input/output buffers).
        Grid/block dimensions (if they don't increase resource usage).
        Memcpy source/destination pointers and size (same direction).
        Memset target pointer, value, and size.

    WHAT FORCES RE-INSTANTIATION:
        Adding or removing nodes.
        Changing the graph topology (new edges).
        Changing kernel function handles (different kernel).
        Increasing resource requirements (larger SMEM, more blocks).

    FAILURE MODE: if cudaGraphExecUpdate fails (returns non-VALID result),
    the executable graph is invalidated. You must call cudaGraphInstantiate
    again before the next launch.

    PYTORCH USE CASE:
        In LLM inference, the decode step repeats with the same kernel sequence
        but different KV cache pointers and different output buffer addresses.
        cudaGraphExecUpdate patches the kernel arg pointers each step,
        retaining the pre-compiled graph structure.

### Static vs Dynamic Tensor Addresses in PyTorch Graphs

    PyTorch's graph integration pre-allocates STATIC BUFFERS:
        static_input  = torch.empty_like(real_input)   # fixed GPU address
        static_output = model(static_input)              # captured computation

    At inference time:
        static_input.copy_(real_input)   # overwrite static buffer
        graph.replay()                    # kernel reads from static_input's fixed address
        real_output.copy_(static_output) # copy result out

    The GPU address of static_input NEVER CHANGES between replays.
    Kernel node arguments point to this fixed address permanently.
    No cudaGraphExecUpdate needed — PyTorch manages the static buffers.

    FAILURE SCENARIO: using non-static tensors inside the captured region.
        torch.ones(N, device='cuda') allocates a NEW buffer each time.
        If captured, the graph bakes in the address from capture time.
        On replay, the address is stale → corrupt data or segfault.
        FIX: allocate all tensors BEFORE capture, reuse them in the captured code.


##### PART 4 — GRAPH TOPOLOGY: NODES, EDGES AND PARALLELISM

### Graph as a DAG

    A CUDA Graph is a DIRECTED ACYCLIC GRAPH where:
        Nodes = operations (kernels, memcpy, memset, etc.)
        Edges = "must complete before" dependencies

    IMPLICIT PARALLELISM: nodes with no dependency relationship between them
    execute in PARALLEL on the GPU. The graph engine schedules them onto
    different SMs simultaneously.

    EXAMPLE — transformer attention layer graph:
        root → [Q_proj, K_proj, V_proj]   (3 nodes run in parallel!)
        Q_proj, K_proj → flash_attention
        flash_attention, V_proj → O_proj
        O_proj → residual_add
        residual_add → layer_norm

    WITHOUT CUDA GRAPHS: these would be submitted sequentially to a single
    stream — the GPU serialises them even if they're independent.
    WITH CUDA GRAPHS: Q/K/V projections overlap! ~33% speedup for this layer.

### Multi-Stream Capture and Fork/Join

    Capturing MULTIPLE STREAMS creates a graph with fork/join structure:

        cudaStreamBeginCapture(stream_main, cudaStreamCaptureModeGlobal);

        // Fork: create work on secondary streams
        cudaEventRecord(event_fork, stream_main);
        cudaStreamWaitEvent(stream_branch1, event_fork, 0);
        cudaStreamWaitEvent(stream_branch2, event_fork, 0);

        kernel_A<<<grid, block, 0, stream_branch1>>>();  // parallel
        kernel_B<<<grid, block, 0, stream_branch2>>>();  // parallel

        // Join: wait for both branches in main stream
        cudaEventRecord(event_join1, stream_branch1);
        cudaEventRecord(event_join2, stream_branch2);
        cudaStreamWaitEvent(stream_main, event_join1, 0);
        cudaStreamWaitEvent(stream_main, event_join2, 0);

        kernel_C<<<grid, block, 0, stream_main>>>();   // sequential after join

        cudaStreamEndCapture(stream_main, &graph);

    RESULT: graph has fork→[A,B parallel]→join→C topology.
    GPU executes A and B on different SMs simultaneously.

### Child Graphs

    A CHILD GRAPH NODE embeds one entire graph as a single node within another.
        Use for: modular composition (layer A as a child of the full model graph).
        Updates to the child graph propagate up when re-instantiated.
        Child graphs can be shared across multiple parent graphs.

    EXAMPLE:
        // Create the attention subgraph once
        cudaGraphCreate(&attn_graph, 0);
        // ... add nodes to attn_graph ...
        cudaGraphInstantiate(&attn_exec, attn_graph, nullptr, nullptr, 0);

        // Use as child in the full transformer graph
        cudaGraphAddChildGraphNode(&child_node, full_graph, &deps, 1, attn_graph);


##### PART 5 — CONDITIONAL NODES: GPU-SIDE CONTROL FLOW (CUDA 12.4+)

### The Control Flow Problem in Graphs

    Traditional CUDA Graphs are STATIC: the same nodes execute in the same
    order every replay. Control flow (if/loop) is not representable.

    If you need control flow (e.g., loop until convergence, early exit, or
    speculative decoding accept/reject), pre-CUDA-12 options are:
        A) CPU round-trip: GPU → CPU read → CPU decides → CPU launches next.
           Latency: ~5–10 µs per decision (PCIe round-trip).
        B) Persistent kernel: stay-alive loop polling a device flag.
           Works but uses SM resources while idle.
        C) Multiple static graphs: CPU decides which graph to launch.
           Eliminates kernel launch overhead but not decision overhead.

    CUDA 12.4 CONDITIONAL NODES solve this with GPU-side if/while.

### The Conditional Handle

    All conditional nodes share a mechanism: the CONDITIONAL HANDLE.
    This is a 64-bit value in GPU memory that the graph runtime reads
    at execution time to decide whether to execute the conditional body.

        cudaGraphConditionalHandle handle;
        cudaGraphConditionalHandleCreate(
            &handle,
            graph,
            0,     // default value: 0 = false/don't loop
            cudaGraphCondTypeIf   // or cudaGraphCondTypeWhile
        );

    The handle is set by a SETTER KERNEL inside the graph:
        __device__ void my_condition_kernel(
            cudaGraphConditionalHandle handle, ...) {
            // Set to 1 (true/continue-loop) or 0 (false/exit)
            cudaGraphSetConditional(handle, condition ? 1 : 0);
        }

### IF Conditional Node

    An IF node contains a BODY GRAPH that executes ONCE if the handle = 1,
    and is SKIPPED entirely if handle = 0.

        cudaGraphNodeParams params = {};
        params.type = cudaGraphNodeTypeConditional;
        params.conditional.type = cudaGraphCondTypeIf;
        params.conditional.handle = cond_handle;
        params.conditional.size = 1;

        cudaGraphNode_t if_node;
        cudaGraphAddNode(&if_node, graph, &deps, n_deps, &params);

        // Get the body graph from the conditional node and add nodes to it:
        cudaGraph_t* body_graphs;
        cudaGraphNodeGetParams(if_node, &params);
        body_graphs = params.conditional.phGraph_out;
        // Add kernels to body_graphs[0]...

    EXECUTION:
        A "condition setter" kernel runs before the IF node.
        The setter writes 0 or 1 to the conditional handle.
        IF node reads the handle: if 1 → execute body; if 0 → skip body.
        No CPU involvement, no synchronisation, no PCIe round-trip.

### WHILE Conditional Node

    A WHILE node contains a BODY GRAPH that repeats until the handle = 0.

        params.conditional.type = cudaGraphCondTypeWhile;
        // Default value = 1 (loop runs at least once)

    EXECUTION ORDER:
        1. Execute body graph once.
        2. A kernel at the END of the body graph evaluates the loop condition
           and writes it to the conditional handle.
        3. If handle = 1: jump to step 1. If 0: exit.

    THIS IS A DO-WHILE LOOP semantics: body runs at least once.
    For a standard WHILE (may execute zero times): add an IF node before the WHILE.

    EXAMPLE — iterative solver:
        Body graph:
            [update_x_kernel]     // one CG iteration
            [compute_residual]    // check convergence
            [set_condition_kernel]// writes: handle = (residual > tol) ? 1 : 0

        While node wraps the above body.
        Solver runs until convergence without a single CPU check.

### Use Case: Speculative Decoding with Conditional Nodes

    STANDARD SPECULATIVE DECODING CONTROL FLOW:
        1. Draft model generates k tokens.
        2. Target model verifies k tokens in one forward pass.
        3. IF accepted_count == k: generate one bonus token (accept all).
           ELSE: resample the first rejected token.

    WITH CONDITIONAL NODES:
        // Graph structure:
        [draft_model × k steps (unrolled)]
        [target_model_verify]
        [compute_accept_count]
        [set_handle: handle = (all_accepted ? 1 : 0)]
        [IF all_accepted: bonus_token_kernel]
        [resample_kernel (always runs, produces the last accepted + resample)]
        [append_to_sequence]

        All decisions happen on GPU. CPU only reads the final sequence length.
        LATENCY SAVING: eliminates k × (5 µs GPU→CPU read + 5 µs CPU decision).
        For k=4, 100 spec steps/second: saves 4 × 100 × 10 µs = 4 ms/second.

### CUDA 12 Conditional Node Limitations

    CURRENT LIMITATIONS (as of CUDA 12.4):
        Only ONE conditional type per node (if OR while, not both).
        Nested conditionals: supported but limited depth (practical: 2–3 levels).
        Body graph must be self-contained (no external stream dependencies).
        cudaGraphExecUpdate does NOT work across conditional node topology changes.
        Performance: conditional node overhead ~2–5 µs per evaluation
                     (much less than a CPU round-trip but non-zero).

    SUPPORTED OPERATIONS IN BODY GRAPH:
        All standard node types (kernel, memcpy, memset, host, child graph).
        Nested conditional nodes.
        cudaMallocAsync (stream-ordered allocation).


##### PART 6 — PROFILING AND DEBUGGING CUDA GRAPHS

### Nsight Systems and Graphs

    Nsight Systems shows CUDA Graphs as a single "Graph Launch" event on
    the GPU timeline, with internal node executions shown as sub-events.

    TO ENABLE INTERNAL VISIBILITY:
        nsys profile --cuda-graph-trace=node ...
        Without this flag: graph internals are hidden behind one launch event.
        With the flag: each node appears separately on the timeline.

    IDENTIFYING GRAPH-RELATED EVENTS:
        "cuGraphLaunch"  — the launch call (CPU side, very short).
        "cuda graph" rectangles on GPU timeline — node execution.
        Look for parallel execution: adjacent rectangles at the same time
        indicate independent nodes running simultaneously.

### Common CUDA Graph Bugs

    BUG 1 — Stale device pointer in captured kernel:
        SYMPTOM: Correct results during capture warmup, wrong results on replay.
        CAUSE: A tensor was re-allocated between warmup and replay.
              Its GPU address changed; the graph still uses the old address.
        FIX: Use static tensors (pre-allocated before capture, never freed).
             In PyTorch: use in-place operations on static buffers.

    BUG 2 — cudaMalloc during capture:
        SYMPTOM: "cudaErrorStreamCaptureInvalidated" or capture failure.
        CAUSE: cudaMalloc called inside the captured region.
        FIX: Move all allocations before cudaStreamBeginCapture.
             Or use cudaMallocAsync (stream-ordered, supported in graphs).

    BUG 3 — cudaDeviceSynchronize during capture:
        SYMPTOM: Capture fails immediately.
        CAUSE: cudaDeviceSynchronize is forbidden during capture.
        FIX: Replace with cudaStreamSynchronize (on a non-capture stream)
             or restructure with events.

    BUG 4 — NCCL / cuDNN / cuBLAS internal sync:
        SYMPTOM: Capture works, but library calls are missing from the graph.
        CAUSE: Some library versions call cudaDeviceSynchronize internally,
               which breaks graph capture.
        FIX: Use library versions that support stream capture.
             cuBLAS ≥ 11.0, cuDNN ≥ 8.0, NCCL ≥ 2.9 support stream capture.
             Verify with: cudaStreamIsCapturing(stream, &status).

    BUG 5 — cudaGraphExecUpdate invalidation:
        SYMPTOM: "cudaErrorGraphExecUpdateFailure" returned.
        CAUSE: The new graph has different topology than the executable graph.
        FIX: Check result.lastErrorNode. Rebuild with cudaGraphInstantiate.

### Nsight Compute and Graph Nodes

    Individual kernel nodes inside a graph can be profiled with NCU:
        ncu --kernel-name <name> --graph-mode=1 ...
        Without --graph-mode: NCU may not replay graph kernel nodes correctly.

    CUDA 12.3+ cudaGraphInstantiateFlagUploadWithoutLaunch:
        Allows uploading the compiled graph to GPU memory without launching.
        Combined with NCU's multi-pass profiling (replay mode), enables
        accurate per-node metrics inside a CUDA Graph.


##### PART 7 — PRODUCTION PATTERNS: CUDA GRAPHS IN LLM INFERENCE STACKS

### vLLM + CUDA Graphs

    vLLM (v0.4+) uses CUDA Graphs for its "decoding fast path":
        1. At startup: capture graphs for a set of fixed batch sizes
           (e.g., B=1, 2, 4, 8, 16, 32, 64, 128).
        2. During serving: pad the actual batch to the next captured batch size.
        3. Launch the appropriate pre-captured graph.
        4. Trim the output to the actual batch size.

    PADDING OVERHEAD: B=9 pads to 16 (7 extra tokens). For a memory-bound
    decode step: extra HBM reads ≈ 7/16 × overhead ≈ negligible vs graph speedup.

    GRAPH REGISTRY: vLLM maintains a dict {batch_size: exec_graph}.
    MEMORY: each graph occupies ~50–200 MB of GPU memory (mostly for static
    activations and the compiled command buffer).
    For 8 batch sizes: ~1–2 GB additional VRAM.

### TensorRT-LLM and CUDA Graphs

    NVIDIA TensorRT-LLM compiles entire model inference into CUDA Graphs
    at build time (not stream capture at runtime):
        1. trtllm-build traces the computation statically.
        2. Emits a CUDA Graph with all ops as graph nodes.
        3. cudaGraphExecUpdate patches input/output pointers at runtime.

    ADVANTAGE over runtime stream capture: the graph is constructed with
    full knowledge of tensor shapes → more aggressive optimisation.
    Layer fusion at the graph level: adjacent nodes may be merged into
    fused kernels that weren't fused during streaming execution.

### The "Static Shape" Constraint and Workarounds

    CUDA Graphs require fixed shapes. For LLM inference:
        FIXED: hidden dimension d, num_heads H, head_dim d_h, num_layers L.
        VARIABLE: batch size B, sequence length S.

    SOLUTION MATRIX:
        Variable B: capture one graph per B in {1,2,4,...,max_B}. Pad to nearest.
        Variable S (prefill): prefill is compute-intensive → shape change justified,
                              use separate stream execution (no graph for prefill).
        Variable S (decode with KV cache): KV context length grows each step.
            If KV is paged (vLLM style): attention kernel argument changes but
            block table pointers are fixed. Use cudaGraphExecUpdate to patch.
            If KV is contiguous: must recapture as S changes. Usually not worth it.

### PyTorch 2.0+ and torch.compile with CUDA Graphs

    torch.compile(model, mode="reduce-overhead") automatically uses CUDA Graphs:
        - Traces the model execution.
        - Identifies stable graph regions (fixed shapes, no dynamic control flow).
        - Captures those regions as CUDA Graphs.
        - Inserts copy_ calls for input updates between graph replays.

    torch.compile(model, mode="max-autotune"):
        Additionally searches for the best kernel implementation per graph node.
        Higher compile time (~minutes for large models) but maximum performance.

    DISABLE CUDA GRAPHS in torch.compile:
        torch.compile(model, options={"enable_cuda_graph": False})

    INSPECT which regions were graphed:
        torch._dynamo.explain(model, *inputs)  # shows graph boundaries

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Launch Overhead Model — When CUDA Graphs Pay Off": {
        "description": (
            "Build a precise model of kernel launch overhead and CUDA Graph "
            "launch overhead. Compute the overhead fraction for different kernel "
            "durations, batch sizes, and kernel counts. Show the crossover point "
            "where graphs become beneficial. Model the full LLM decode step "
            "overhead (240 kernels) for batch sizes 1 to 128 comparing "
            "conventional streaming vs graph launch. Show the tokens/sec improvement."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  LAUNCH OVERHEAD MODEL — When CUDA Graphs Pay Off")
print("=" * 68)
print()

# Constants
LAUNCH_US_CONVENTIONAL = 5.0    # µs per kernel launch (conventional path)
LAUNCH_US_GRAPH        = 0.5    # µs per cudaGraphLaunch (entire graph)
H100_HBM_BW_GBS        = 3350.0
WEIGHTS_7B_GB          = 14.0


def decode_kernel_time_us(batch_size):
    """
    Approximate actual GPU compute time per decode step (µs).
    Memory-bound: load all 14 GB of weights + KV cache.
    """
    kv_per_token_gb  = 0.0005   # Llama-2-7B: 0.5 MB/token KV cache
    n_cached_tokens  = 512      # assume 512-token context
    kv_gb            = kv_per_token_gb * n_cached_tokens * batch_size
    total_gb         = WEIGHTS_7B_GB + kv_gb
    return total_gb / H100_HBM_BW_GBS * 1000   # convert GB/s to µs

def decode_overhead_breakdown(batch_size, n_kernels=240):
    """Return breakdown of decode step time for one batch."""
    kernel_time_us   = decode_kernel_time_us(batch_size)
    launch_conv_us   = n_kernels * LAUNCH_US_CONVENTIONAL
    launch_graph_us  = LAUNCH_US_GRAPH
    total_conv_us    = kernel_time_us + launch_conv_us
    total_graph_us   = kernel_time_us + launch_graph_us
    return {
        "kernel_us":      kernel_time_us,
        "launch_conv_us": launch_conv_us,
        "launch_graph_us":launch_graph_us,
        "total_conv_us":  total_conv_us,
        "total_graph_us": total_graph_us,
        "overhead_pct_conv":  launch_conv_us / total_conv_us * 100,
        "overhead_pct_graph": launch_graph_us / total_graph_us * 100,
        "speedup": total_conv_us / total_graph_us,
        "tps_conv":  batch_size / total_conv_us * 1e6,
        "tps_graph": batch_size / total_graph_us * 1e6,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Overhead fraction at different kernel durations
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Overhead Fraction vs Kernel Duration")
print("━" * 68)
print()

print(f"  Conventional launch: {LAUNCH_US_CONVENTIONAL} µs/kernel")
print(f"  Graph launch:        {LAUNCH_US_GRAPH} µs per entire graph")
print()
print(f"  {'Kernel µs':>12}  {'Conv overhead%':>16}  "
      f"{'Graph overhead%':>17}  {'Graph speedup':>14}  {'Verdict'}")
print("  " + "─" * 64)

for kernel_us in [1, 5, 10, 25, 50, 100, 200, 500, 1000, 5000]:
    conv_total  = kernel_us + LAUNCH_US_CONVENTIONAL
    graph_total = kernel_us + LAUNCH_US_GRAPH
    conv_pct    = LAUNCH_US_CONVENTIONAL / conv_total * 100
    graph_pct   = LAUNCH_US_GRAPH / graph_total * 100
    speedup     = conv_total / graph_total
    verdict     = ("✅ essential" if conv_pct > 30
                   else "✅ very useful" if conv_pct > 10
                   else "⚠ marginal" if conv_pct > 3
                   else "  skip")
    print(f"  {kernel_us:>12}  {conv_pct:>15.1f}%  "
          f"{graph_pct:>16.2f}%  {speedup:>13.2f}×  {verdict}")

print()
print("  Rule: CUDA Graphs are essential when kernel_duration < 50 µs.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: LLM decode overhead breakdown
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Llama-2-7B Decode: Overhead Breakdown by Batch Size")
print("━" * 68)
print()

print(f"  240 kernels/decode step, H100 SXM5")
print(f"  KV cache context: 512 tokens")
print()
print(f"  {'Batch':>6}  {'GPU compute µs':>16}  {'Conv launch µs':>16}  "
      f"{'Graph launch µs':>17}  {'Conv total µs':>15}  "
      f"{'Graph total µs':>16}  {'Speedup':>8}")
print("  " + "─" * 96)

for B in [1, 2, 4, 8, 16, 32, 64, 128]:
    r = decode_overhead_breakdown(B)
    print(f"  {B:>6}  {r['kernel_us']:>16.2f}  {r['launch_conv_us']:>16.1f}  "
          f"{r['launch_graph_us']:>17.1f}  {r['total_conv_us']:>15.2f}  "
          f"{r['total_graph_us']:>16.2f}  {r['speedup']:>8.2f}×")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Tokens per second comparison
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Tokens/Second: Conventional vs CUDA Graphs")
print("━" * 68)
print()

print(f"  {'Batch':>6}  {'Conv TPS':>12}  {'Graph TPS':>12}  "
      f"{'TPS gain':>10}  {'Conv overhead %':>17}  {'Graph overhead %'}")
print("  " + "─" * 72)

for B in [1, 2, 4, 8, 16, 32, 64, 128]:
    r = decode_overhead_breakdown(B)
    tps_gain = r['tps_graph'] - r['tps_conv']
    print(f"  {B:>6}  {r['tps_conv']:>12.1f}  {r['tps_graph']:>12.1f}  "
          f"{tps_gain:>+10.1f}  {r['overhead_pct_conv']:>16.1f}%  "
          f"{r['overhead_pct_graph']:>16.2f}%")

print()
print("  At batch=1: graphs add 22+ tokens/sec by eliminating 1.2 ms overhead.")
print("  At batch=128: compute-bound; graphs still help ~3% but less critical.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Graph capture + replay amortisation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Break-Even: How Many Replays to Amortise Capture Cost?")
print("━" * 68)
print()

CAPTURE_MS     = 15.0    # typical ms to capture and instantiate a graph
REPLAY_SAVE_US = 240 * (LAUNCH_US_CONVENTIONAL - LAUNCH_US_GRAPH / 240)
# savings per replay = n_kernels * conv_cost - 1 graph_launch_cost

print(f"  One-time graph capture + instantiation: ~{CAPTURE_MS} ms")
print(f"  Savings per graph replay (240 kernels): "
      f"{240*LAUNCH_US_CONVENTIONAL - LAUNCH_US_GRAPH:.1f} µs")
print()

break_even_replays = (CAPTURE_MS * 1000) / (240*LAUNCH_US_CONVENTIONAL - LAUNCH_US_GRAPH)
break_even_tokens  = break_even_replays   # one token per decode step

print(f"  Break-even at: {break_even_replays:.0f} replays")
print(f"  For a serving system at 100 tokens/sec: break-even in "
      f"{break_even_replays/100:.2f} seconds after startup.")
print(f"  For 10 requests of 200 tokens each = {10*200} replays — "
      f"{'worth it ✅' if 10*200 > break_even_replays else 'not worth it ❌'}")
print()

# Show amortised cost vs number of replays
print(f"  {'N replays':>12}  {'Cumulative savings µs':>22}  "
      f"{'Capture cost µs':>17}  {'Net benefit µs':>16}  {'Break-even?'}")
print("  " + "─" * 76)

savings_per = 240*LAUNCH_US_CONVENTIONAL - LAUNCH_US_GRAPH
for n_reps in [1, 10, 50, 100, 500, 1000, 5000]:
    cum_savings = n_reps * savings_per
    cap_cost    = CAPTURE_MS * 1000
    net         = cum_savings - cap_cost
    be          = "✅ yes" if net > 0 else "  no"
    print(f"  {n_reps:>12}  {cum_savings:>22.1f}  {cap_cost:>17.1f}  "
          f"{net:>16.1f}  {be}")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Graph Capture Simulator — Node Recording, DAG & Parallelism": {
        "description": (
            "Simulate the CUDA Graph capture mechanism: record kernel nodes, "
            "memcpy nodes, and cross-stream dependencies. Build the DAG from "
            "recorded events. Identify independent (parallel) nodes. Show the "
            "critical path length and parallel speedup vs sequential execution. "
            "Simulate a transformer layer graph with Q/K/V projection overlap. "
            "Demonstrate fork/join topology from multi-stream capture."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict

print("=" * 68)
print("  GRAPH CAPTURE SIMULATOR — Node Recording, DAG & Parallelism")
print("=" * 68)
print()


@dataclass
class GraphNode:
    node_id:   int
    name:      str
    duration_us: float    # simulated execution time
    deps:      List[int] = field(default_factory=list)   # prerequisite node IDs
    stream_id: int = 0

    @property
    def label(self):
        return f"{self.name}({self.duration_us:.0f}µs)"


class CUDAGraph:
    """Simulate CUDA Graph structure: nodes, edges, critical path analysis."""

    def __init__(self, name=""):
        self.name     = name
        self.nodes: Dict[int, GraphNode] = {}
        self._next_id = 0

    def add_node(self, name, duration_us, deps=None, stream_id=0):
        nid  = self._next_id; self._next_id += 1
        node = GraphNode(nid, name, duration_us,
                         deps=deps or [], stream_id=stream_id)
        self.nodes[nid] = node
        return nid

    def topological_order(self):
        """Kahn's algorithm for topological sort."""
        in_degree = {nid: 0 for nid in self.nodes}
        for n in self.nodes.values():
            for dep in n.deps:
                in_degree[n.node_id] += 1

        queue  = [nid for nid, deg in in_degree.items() if deg == 0]
        order  = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            # Find successors
            for other in self.nodes.values():
                if nid in other.deps:
                    in_degree[other.node_id] -= 1
                    if in_degree[other.node_id] == 0:
                        queue.append(other.node_id)
        return order

    def critical_path(self):
        """Compute the longest path through the DAG (critical path length)."""
        order     = self.topological_order()
        earliest  = {nid: 0.0 for nid in self.nodes}
        for nid in order:
            node = self.nodes[nid]
            start = max((earliest[dep] + self.nodes[dep].duration_us)
                        for dep in node.deps) if node.deps else 0.0
            earliest[nid] = start
        # Critical path = max(earliest[nid] + duration) over all nodes
        cp = max(earliest[nid] + self.nodes[nid].duration_us
                 for nid in self.nodes)
        return cp

    def sequential_time(self):
        """Total time if all nodes ran sequentially."""
        return sum(n.duration_us for n in self.nodes.values())

    def parallel_speedup(self):
        """Theoretical speedup from parallel execution."""
        return self.sequential_time() / self.critical_path()

    def schedule(self):
        """
        Simulate execution timeline: assign start times respecting deps.
        Returns list of (node_id, start_us, end_us).
        """
        order = self.topological_order()
        finish = {}   # node_id → finish time
        timeline = []
        for nid in order:
            node  = self.nodes[nid]
            start = max((finish[dep] for dep in node.deps), default=0.0)
            end   = start + node.duration_us
            finish[nid] = end
            timeline.append((nid, start, end))
        return timeline, max(finish.values())

    def print_graph(self):
        print(f"  Graph '{self.name}': {len(self.nodes)} nodes")
        for nid, node in self.nodes.items():
            dep_str = f"deps={[self.nodes[d].name for d in node.deps]}" if node.deps else "no deps"
            print(f"    [{nid}] {node.name:<30} {node.duration_us:>6.1f} µs   {dep_str}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Simple linear vs parallel graph comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Linear (Stream) vs Parallel (Graph) Execution")
print("━" * 68)
print()

# Linear graph (as if submitted to one stream)
g_linear = CUDAGraph("Linear stream")
n0 = g_linear.add_node("Q_proj",   80)
n1 = g_linear.add_node("K_proj",   80, deps=[n0])  # waits for Q
n2 = g_linear.add_node("V_proj",   80, deps=[n1])  # waits for K
n3 = g_linear.add_node("attn",    200, deps=[n2])
n4 = g_linear.add_node("O_proj",   80, deps=[n3])
n5 = g_linear.add_node("FFN",     300, deps=[n4])

# Parallel graph (multi-stream capture → Q/K/V run in parallel)
g_parallel = CUDAGraph("Parallel graph")
p0 = g_parallel.add_node("Q_proj",   80, stream_id=0)
p1 = g_parallel.add_node("K_proj",   80, stream_id=1)   # independent
p2 = g_parallel.add_node("V_proj",   80, stream_id=2)   # independent
p3 = g_parallel.add_node("attn",    200, deps=[p0, p1, p2])  # waits for all 3
p4 = g_parallel.add_node("O_proj",   80, deps=[p3])
p5 = g_parallel.add_node("FFN",     300, deps=[p4])

for g in [g_linear, g_parallel]:
    seq_time = g.sequential_time()
    cp_time  = g.critical_path()
    speedup  = g.parallel_speedup()
    print(f"  {g.name}:")
    print(f"    Sequential time: {seq_time:.0f} µs")
    print(f"    Critical path:   {cp_time:.0f} µs")
    print(f"    Parallel speedup:{speedup:.2f}×")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Full transformer layer graph
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Transformer Layer Graph: All Operations")
print("━" * 68)
print()

g_layer = CUDAGraph("Transformer layer (with parallelism)")

# LayerNorm
ln1   = g_layer.add_node("LayerNorm_pre",  10)
# Parallel Q, K, V projections
qp    = g_layer.add_node("Q_proj_GEMM",    80, deps=[ln1], stream_id=0)
kp    = g_layer.add_node("K_proj_GEMM",    80, deps=[ln1], stream_id=1)
vp    = g_layer.add_node("V_proj_GEMM",    80, deps=[ln1], stream_id=2)
# Rotary embeddings (apply to Q and K — parallel)
rq    = g_layer.add_node("RoPE_Q",         5,  deps=[qp],  stream_id=0)
rk    = g_layer.add_node("RoPE_K",         5,  deps=[kp],  stream_id=1)
# Flash Attention (waits for Q, K, V)
fa    = g_layer.add_node("FlashAttn",     120, deps=[rq, rk, vp])
# Output projection
op    = g_layer.add_node("O_proj_GEMM",    80, deps=[fa])
# Residual add
ra1   = g_layer.add_node("Residual_add",   5,  deps=[op, ln1])
# Second LayerNorm
ln2   = g_layer.add_node("LayerNorm_post", 10, deps=[ra1])
# Parallel FFN gate and up projections
gate  = g_layer.add_node("FFN_gate_GEMM",  120, deps=[ln2], stream_id=0)
up    = g_layer.add_node("FFN_up_GEMM",    120, deps=[ln2], stream_id=1)
# SwiGLU activation
swi   = g_layer.add_node("SwiGLU",          5, deps=[gate, up])
# FFN down projection
down  = g_layer.add_node("FFN_down_GEMM",  80, deps=[swi])
# Final residual
ra2   = g_layer.add_node("Residual_add2",   5, deps=[down, ra1])

g_layer.print_graph()
print()

seq_t = g_layer.sequential_time()
cp_t  = g_layer.critical_path()
sp    = g_layer.parallel_speedup()
print(f"  Sequential time: {seq_t:.0f} µs  (stream execution)")
print(f"  Critical path:   {cp_t:.0f} µs   (graph parallel execution)")
print(f"  Graph speedup:   {sp:.2f}× vs stream execution")
print()

# Print critical path
print("  Critical path nodes (sequential chain):")
timeline, total = g_layer.schedule()
timeline.sort(key=lambda x: x[2], reverse=True)   # sort by end time

# trace back the critical path
crit_end   = max(t[2] for t in timeline)
crit_nodes = []
queue_cp   = [t for t in timeline if abs(t[2] - crit_end) < 0.1]
while queue_cp:
    nid, start, end = queue_cp[0]; queue_cp = queue_cp[1:]
    crit_nodes.append((nid, start, end))
    if start == 0:
        break
    # Find predecessor on critical path
    node = g_layer.nodes[nid]
    for dep in node.deps:
        dep_end = next(t[2] for t in timeline if t[0] == dep)
        if abs(dep_end - start) < 0.1:
            queue_cp.append(next(t for t in timeline if t[0] == dep))
            break

crit_nodes.sort(key=lambda x: x[1])
for nid, start, end in crit_nodes:
    print(f"    [{start:>6.0f} – {end:>6.0f} µs] {g_layer.nodes[nid].name}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Multi-layer pipeline — cumulative savings
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Multi-Layer Pipeline: Cumulative Graph Savings")
print("━" * 68)
print()

N_LAYERS      = 32
per_layer_seq = g_layer.sequential_time()
per_layer_cp  = g_layer.critical_path()

total_seq    = per_layer_seq * N_LAYERS
total_graph  = per_layer_cp  * N_LAYERS
# Plus graph launch overhead
n_kernel_approx = len(g_layer.nodes) * N_LAYERS
launch_conv  = n_kernel_approx * LAUNCH_US_CONVENTIONAL
launch_graph = LAUNCH_US_GRAPH

total_with_conv  = total_seq  + launch_conv
total_with_graph = total_graph + launch_graph

print(f"  {N_LAYERS}-layer model, {len(g_layer.nodes)} kernels/layer = "
      f"{n_kernel_approx} total kernels")
print()
print(f"  {'Component':<35}  {'Stream (µs)':>13}  {'Graph (µs)':>12}")
print("  " + "─" * 60)
print(f"  {'GPU compute (sequential)':<35}  {total_seq:>13.0f}  {'N/A':>12}")
print(f"  {'GPU compute (parallel, graph)':<35}  {'N/A':>13}  {total_graph:>12.0f}")
print(f"  {'Kernel launch overhead':<35}  {launch_conv:>13.1f}  {launch_graph:>12.1f}")
print(f"  {'Total step time':<35}  {total_with_conv:>13.1f}  {total_with_graph:>12.1f}")
print()
print(f"  Graph speedup (parallelism + reduced launch overhead):")
print(f"    {total_with_conv/total_with_graph:.2f}× faster decode step")
print(f"    Parallelism gain alone: {total_seq/total_graph:.2f}×")
print(f"    Launch overhead removal: {(total_seq+launch_conv)/(total_seq+launch_graph):.2f}×")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Graph Replay & Update — Static Buffers, Patching & PyTorch API": {
        "description": (
            "Simulate the CUDA Graph replay and update workflow. Show the "
            "static-buffer pattern required for correct graph replay. Simulate "
            "cudaGraphExecUpdate by tracking which kernel arguments change "
            "between replays. Show what triggers a re-instantiation vs a cheap "
            "update. Benchmark the latency of replay vs conventional launch "
            "across a sequence of 1000 decode steps."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

print("=" * 68)
print("  GRAPH REPLAY & UPDATE — Static Buffers, Patching & PyTorch API")
print("=" * 68)
print()

np.random.seed(42)

LAUNCH_US_CONV  = 5.0     # µs per conventional kernel launch
LAUNCH_US_GRAPH = 0.5     # µs per graph launch
UPDATE_US       = 0.8     # µs per cudaGraphExecUpdate (arg patch)
REINSTANTIATE_US= 1200.0  # µs to re-instantiate (full rebuild)


@dataclass
class KernelArgs:
    """Represents the arguments to a CUDA kernel node in a graph."""
    input_ptr:  int    # GPU memory address (simulated as int)
    output_ptr: int
    weight_ptr: int    # fixed (weight tensor never moves)
    batch_size: int
    seq_len:    int

    def same_structure(self, other):
        """Do they have the same 'topology'? Batch/seq changes are OK to patch."""
        return (self.weight_ptr == other.weight_ptr and
                self.batch_size <= other.batch_size * 2)  # allow doubling

    def patchable(self, other):
        """Can we use cudaGraphExecUpdate (fast) vs re-instantiate (slow)?"""
        # Patchable: only pointer or constant scalar changes
        # NOT patchable: batch_size changes (different grid dims needed)
        return self.batch_size == other.batch_size


@dataclass
class GraphExecutable:
    """Simulates a cudaGraphExec_t with its baked-in kernel arguments."""
    kernel_args: List[KernelArgs]
    is_valid:    bool = True
    n_replays:   int  = 0
    n_updates:   int  = 0
    n_reinstantiations: int = 0

    def replay_us(self):
        return LAUNCH_US_GRAPH

    def update(self, new_args: List[KernelArgs]):
        """Try to update; returns (success, cost_us)."""
        if not self.is_valid:
            return False, REINSTANTIATE_US

        for old, new in zip(self.kernel_args, new_args):
            if not old.patchable(new):
                # Must re-instantiate
                self.kernel_args = new_args
                self.n_reinstantiations += 1
                return False, REINSTANTIATE_US

        # Patchable: fast update
        self.kernel_args = new_args
        self.n_updates += 1
        return True, UPDATE_US


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Static buffer pattern
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Static Buffer Pattern: Why Addresses Must Not Change")
print("━" * 68)
print()

print("  PROBLEM: CUDA Graph bakes in GPU memory addresses at capture time.")
print("  If the input tensor is re-allocated, the graph uses a stale address.")
print()

print("  WRONG PATTERN (will corrupt data on replay):")
print("    # Capture:")
print("    with cuda_graph_capture(stream):")
print("        x = torch.randn(batch, seq, hidden)   # <-- allocates NEW tensor each call!")
print("        output = model(x)")
print("    # Replay (different iteration):")
print("    graph.replay()   # x's GPU address has changed → reads garbage!")
print()

print("  CORRECT PATTERN (static buffers):")
print("    # Pre-allocate static buffers ONCE:")
print("    static_input  = torch.empty(max_batch, max_seq, hidden, device='cuda')")
print("    static_output = torch.empty(max_batch, max_seq, hidden, device='cuda')")
print()
print("    # Capture (once):")
print("    with cuda_graph_capture(stream):")
print("        static_output = model(static_input)  # addresses are fixed in graph")
print()
print("    # Replay (every inference call):")
print("    static_input.copy_(new_input)  # overwrite static buffer IN-PLACE")
print("    graph.replay()                 # kernel reads static_input's fixed address")
print("    result = static_output.clone() # copy out result")
print()

# Show address stability simulation
print("  Address stability check (simulated):")
print()
print(f"  {'Allocation strategy':<30}  {'Iter 0 addr':>14}  "
      f"{'Iter 1 addr':>14}  {'Iter 2 addr':>14}  {'Stable?'}")
print("  " + "─" * 72)

base_addr = 0x7F_0000_0000

# Dynamic allocation: new tensor each iteration
for strategy, addrs in [
    ("torch.zeros(N)  (dynamic)",   [base_addr + i*1024*4 for i in [0,1,2]]),
    ("static_buf (pre-alloc)",       [base_addr + 10*1024*4]*3),
    ("static_buf.zero_() (in-place)",[base_addr + 10*1024*4]*3),
]:
    stable = len(set(addrs)) == 1
    print(f"  {strategy:<30}  "
          f"  0x{addrs[0]:08X}  0x{addrs[1]:08X}  0x{addrs[2]:08X}  "
          f"{'✅' if stable else '❌ STALE!'}")

print()
print("  Only pre-allocated (static) buffers keep the same GPU address across calls.")
print("  in-place ops (copy_, zero_, fill_) are safe — they don't change the address.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Graph replay vs update vs re-instantiation costs
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Replay vs Update vs Re-Instantiation Latency")
print("━" * 68)
print()

N_KERNELS_PER_STEP = 240
ops_table = [
    ("Conventional launch (×240)",  N_KERNELS_PER_STEP * LAUNCH_US_CONV,   "every step"),
    ("Graph replay",                 LAUNCH_US_GRAPH,                        "same args"),
    ("GraphExecUpdate + replay",     UPDATE_US + LAUNCH_US_GRAPH,            "ptrs changed"),
    ("Re-instantiate + replay",      REINSTANTIATE_US + LAUNCH_US_GRAPH,     "topology changed"),
    ("Stream capture (new graph)",   15000.0 + LAUNCH_US_GRAPH,              "full recapture"),
]

print(f"  {'Operation':<38}  {'Cost µs':>10}  {'vs Graph replay':>16}  {'When'}")
print("  " + "─" * 72)

baseline = LAUNCH_US_GRAPH
for name, cost, when in ops_table:
    ratio = cost / baseline
    print(f"  {name:<38}  {cost:>10.1f}  {ratio:>14.0f}×  {when}")

print()
print("  cudaGraphExecUpdate is 1.3× over replay — always prefer it over re-instantiation.")
print("  Re-instantiation is 2400× more expensive than replay.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: 1000-step inference simulation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — 1000-Step Decode Simulation: Total Overhead Comparison")
print("━" * 68)
print()

N_STEPS    = 1000
BATCH_SIZE = 4   # fixed batch
base_weight_ptr = 0x7F00_0000_0000

# Simulate the sequence of steps
np.random.seed(99)

# For the graph approach: same batch size throughout (no re-instantiation)
# Pointers change every step (KV cache pointer advances)
step_args = [
    KernelArgs(input_ptr=base_weight_ptr + i*4096*4,
               output_ptr=base_weight_ptr + (N_STEPS + i)*4096*4,
               weight_ptr=0xAA00_0000,
               batch_size=BATCH_SIZE,
               seq_len=i+1)
    for i in range(N_STEPS)
]

# Simulate 3 strategies
strategies = {
    "Conventional launches": {"total_us": 0.0, "n_updates": 0, "n_reinstantiations": 0},
    "Graph (static ptrs, no update)": {"total_us": 0.0, "n_updates": 0, "n_reinstantiations": 0},
    "Graph + ExecUpdate (dynamic ptrs)": {"total_us": 0.0, "n_updates": 0, "n_reinstantiations": 0},
}

for step in range(N_STEPS):
    # Conventional: N_KERNELS_PER_STEP launches
    strategies["Conventional launches"]["total_us"] += N_KERNELS_PER_STEP * LAUNCH_US_CONV

    # Static graph: replay only (assumes input buffer is always the same address)
    strategies["Graph (static ptrs, no update)"]["total_us"] += LAUNCH_US_GRAPH

    # Dynamic graph: update pointer each step (KV pointer changes)
    # All same batch size → patchable → use ExecUpdate
    strategies["Graph + ExecUpdate (dynamic ptrs)"]["total_us"] += UPDATE_US + LAUNCH_US_GRAPH
    strategies["Graph + ExecUpdate (dynamic ptrs)"]["n_updates"] += 1

print(f"  {N_STEPS}-step decode, batch={BATCH_SIZE}, {N_KERNELS_PER_STEP} kernels/step")
print()
print(f"  {'Strategy':<38}  {'Total OH µs':>13}  "
      f"{'Per-step µs':>13}  {'vs Conventional'}")
print("  " + "─" * 72)

conv_total = strategies["Conventional launches"]["total_us"]
for name, stats in strategies.items():
    per_step = stats["total_us"] / N_STEPS
    ratio    = conv_total / stats["total_us"]
    print(f"  {name:<38}  {stats['total_us']:>13.0f}  "
          f"{per_step:>13.3f}  {ratio:>14.1f}× faster")

print()
print("  Graph with ExecUpdate: almost as fast as static graph.")
print("  ExecUpdate adds 0.8 µs/step — tiny vs 1200 µs re-instantiation.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Conditional Nodes — IF/WHILE Simulation & Speculative Decoding": {
        "description": (
            "Simulate CUDA 12.4 conditional graph nodes. Implement IF and WHILE "
            "node semantics with a GPU-side condition flag. Model the latency "
            "of conditional evaluation vs CPU round-trip. Simulate speculative "
            "decoding accept/reject using a WHILE conditional loop. Compare "
            "total latency and tokens/second with and without conditional nodes "
            "across different acceptance rates."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  CONDITIONAL NODES — IF/WHILE Simulation & Speculative Decoding")
print("=" * 68)
print()

np.random.seed(42)

# Latency constants
GPU_CPU_ROUNDTRIP_US    = 10.0   # µs for GPU → CPU read + CPU decision
CONDITIONAL_NODE_US     = 3.0    # µs for GPU-side condition evaluation (CUDA 12.4)
DRAFT_TOKEN_US          = 250.0  # µs to generate 1 draft token (7B model, H100)
TARGET_VERIFY_US        = 1400.0 # µs for 1 target forward pass (70B, H100) with k tokens
# (scales sub-linearly with k due to parallel computation in prefill mode)
TARGET_VERIFY_PER_K_US  = lambda k: TARGET_VERIFY_US * (1 + 0.15 * (k - 1))


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: IF node semantics simulation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — IF Conditional Node: Semantics and Latency Model")
print("━" * 68)
print()

print("  IF node structure:")
print("    [setter_kernel] → writes handle = condition ? 1 : 0")
print("    [IF_node]       → reads handle; executes body_graph if handle==1")
print()
print("  Latency comparison (per conditional evaluation):")
print()

conditions = ["condition=True  (body executes)", "condition=False (body skipped)"]
latencies = {
    "CPU round-trip":   {"base": GPU_CPU_ROUNDTRIP_US, "body_if_true": 5.0},
    "Graph IF node":    {"base": CONDITIONAL_NODE_US,  "body_if_true": 5.0},
}

print(f"  {'Method':<22}  {'Eval latency':>14}  {'+ body (True)':>15}  "
      f"{'+ skip (False)':>16}  {'Savings vs CPU'}")
print("  " + "─" * 68)

cpu_true  = latencies["CPU round-trip"]["base"] + latencies["CPU round-trip"]["body_if_true"]
cpu_false = latencies["CPU round-trip"]["base"]

for name, lats in latencies.items():
    total_true  = lats["base"] + lats["body_if_true"]
    total_false = lats["base"]
    savings     = ((cpu_true + cpu_false) / 2 - (total_true + total_false) / 2)
    savings_pct = savings / ((cpu_true + cpu_false) / 2) * 100
    print(f"  {name:<22}  {lats['base']:>12.1f}µs  {total_true:>13.1f}µs  "
          f"{total_false:>14.1f}µs  {savings_pct:>7.1f}% faster")

print()
print("  WHILE node: body_graph repeats until setter_kernel writes handle=0.")
print("  Enables GPU-side loop WITHOUT CPU polling or cudaDeviceSynchronize.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Speculative decoding with WHILE conditional node
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Speculative Decoding: CPU Round-Trip vs WHILE Node")
print("━" * 68)
print()

def spec_decode_step_time_cpu(k, alpha):
    """
    Time for one speculative decoding step with CPU-side control.
    Returns (total_us, expected_tokens).
    """
    # 1. Draft model generates k tokens sequentially
    draft_time = k * DRAFT_TOKEN_US

    # 2. Target model verifies k tokens in one parallel pass
    verify_time = TARGET_VERIFY_PER_K_US(k)

    # 3. CPU round-trip for accept/reject decision
    cpu_decision_time = GPU_CPU_ROUNDTRIP_US

    # 4. Expected accepted tokens (geometric distribution)
    e_accepted = sum(i * alpha**(i-1) * (1-alpha) for i in range(1, k+1))
    e_accepted += (k+1) * alpha**k

    total = draft_time + verify_time + cpu_decision_time
    return total, e_accepted


def spec_decode_step_time_cond(k, alpha):
    """
    Time for one speculative decoding step with GPU-side WHILE/IF conditional nodes.
    Replaces CPU round-trip with conditional node evaluation.
    """
    draft_time  = k * DRAFT_TOKEN_US
    verify_time = TARGET_VERIFY_PER_K_US(k)
    # GPU-side conditional evaluation (no CPU round-trip)
    cond_time   = CONDITIONAL_NODE_US

    e_accepted  = sum(i * alpha**(i-1) * (1-alpha) for i in range(1, k+1))
    e_accepted += (k+1) * alpha**k

    total = draft_time + verify_time + cond_time
    return total, e_accepted


def standard_decode_time():
    """Time for one standard decode step (no speculation)."""
    return TARGET_VERIFY_US, 1.0

std_time, _ = standard_decode_time()
std_tps     = 1.0 / std_time * 1e6   # tokens per second from target

print(f"  Standard decode (70B target, no spec): {std_time:.0f} µs, {std_tps:.1f} tok/s")
print()
print(f"  Draft: 7B model, {DRAFT_TOKEN_US:.0f} µs/token")
print(f"  Verify: 70B model, ~{TARGET_VERIFY_US:.0f} µs per k-token batch")
print(f"  CPU round-trip: {GPU_CPU_ROUNDTRIP_US:.0f} µs | GPU conditional: {CONDITIONAL_NODE_US:.0f} µs")
print()

print(f"  {'k':>4}  {'α':>6}  {'E[acc]':>8}", end="")
for method in ["CPU round-trip", "GPU conditional"]:
    print(f"  {method:>18} tok/s", end="")
print(f"  {'Cond speedup':>14}")
print("  " + "─" * 72)

for k in [1, 2, 3, 4, 6]:
    for alpha in [0.7, 0.85, 0.95]:
        t_cpu,  e_acc = spec_decode_step_time_cpu(k, alpha)
        t_cond, _     = spec_decode_step_time_cond(k, alpha)

        tps_cpu  = e_acc / t_cpu  * 1e6
        tps_cond = e_acc / t_cond * 1e6

        speedup_vs_std_cpu  = tps_cpu  / std_tps
        speedup_vs_std_cond = tps_cond / std_tps
        cond_gain = tps_cond / tps_cpu

        print(f"  {k:>4}  {alpha:>6.2f}  {e_acc:>8.2f}  "
              f"{tps_cpu:>16.1f} ({speedup_vs_std_cpu:.1f}×std)  "
              f"{tps_cond:>16.1f} ({speedup_vs_std_cond:.1f}×std)  "
              f"{cond_gain:>12.3f}×")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: WHILE node — iterative convergence simulation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — WHILE Node: Iterative Solver Without CPU Polling")
print("━" * 68)
print()

print("  Use case: conjugate gradient solver — run until residual < tolerance.")
print()
print("  WITHOUT CUDA Graphs (CPU polling):")
print("    while True:")
print("        launch_cg_iter(stream)")
print("        cudaStreamSynchronize(stream)  # CPU blocks until done")
print("        residual = d_residual.item()   # GPU→CPU copy")
print("        if residual < tol: break")
print("    Cost per iteration: kernel_us + sync_us + d2h_us")
print()
print("  WITH WHILE conditional node (CUDA 12.4):")
print("    # Body graph: one CG iteration + residual computation")
print("    # End of body: setter_kernel writes 0 to handle if converged")
print("    cudaGraphLaunch(graph_exec, stream)  # runs entire solver")
print("    cudaStreamSynchronize(stream)        # wait once for full convergence")
print("    Cost per iteration: kernel_us + cond_eval_us (no CPU polling!)")
print()

# Model the cost difference
KERNEL_CG_US  = 50.0    # one CG iteration kernel
SYNC_US       = 5.0     # cudaStreamSynchronize latency
D2H_US        = 2.0     # GPU→CPU residual copy

N_ITERATIONS_EXPECTED = [5, 10, 20, 50, 100]

print(f"  {'N iters':>8}  {'CPU polling (µs)':>18}  "
      f"{'WHILE node (µs)':>18}  {'Speedup':>9}")
print("  " + "─" * 56)

for n_iters in N_ITERATIONS_EXPECTED:
    cpu_total   = n_iters * (KERNEL_CG_US + SYNC_US + D2H_US)
    cond_total  = n_iters * (KERNEL_CG_US + CONDITIONAL_NODE_US) + SYNC_US
    speedup     = cpu_total / cond_total
    print(f"  {n_iters:>8}  {cpu_total:>18.1f}  {cond_total:>18.1f}  {speedup:>9.2f}×")

print()
print("  WHILE node eliminates SYNC+D2H per iteration:")
print(f"    Saving per iteration: {SYNC_US + D2H_US:.0f} µs  ({SYNC_US:.0f} sync + {D2H_US:.0f} D2H)")
print(f"    For 50 iterations: {50*(SYNC_US+D2H_US):.0f} µs saved = pure latency recovery.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · CUDA Graphs in Production — vLLM Pattern, Failures & PyTorch API": {
        "description": (
            "Simulate the vLLM CUDA Graph serving pattern: capture graphs for "
            "multiple batch sizes, pad incoming requests to the nearest captured "
            "size, and replay the appropriate graph. Show the padding overhead "
            "tradeoff. Simulate the five most common CUDA Graph bugs and show "
            "their symptoms. Demonstrate the PyTorch torch.cuda.CUDAGraph API "
            "pattern. Compare torch.compile reduce-overhead vs manual graphs."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

print("=" * 68)
print("  CUDA GRAPHS IN PRODUCTION — vLLM Pattern, Failures & PyTorch API")
print("=" * 68)
print()

np.random.seed(42)

LAUNCH_US_CONV  = 5.0
LAUNCH_US_GRAPH = 0.5
H100_BW_GBS     = 3350.0
WEIGHTS_7B_GB   = 14.0


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: vLLM graph registry pattern
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — vLLM Graph Registry: Batch-Size Capture + Padding")
print("━" * 68)
print()

CAPTURED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
GRAPH_CAPTURE_MS_PER = 15.0    # ms to capture one batch-size graph
GRAPH_MEM_MB_PER     = 150.0   # MB of VRAM per captured graph

print(f"  vLLM captures graphs for batch sizes: {CAPTURED_BATCH_SIZES}")
print(f"  Capture time: ~{GRAPH_CAPTURE_MS_PER} ms each")
print(f"  VRAM per graph: ~{GRAPH_MEM_MB_PER} MB")
print()

total_capture_ms  = len(CAPTURED_BATCH_SIZES) * GRAPH_CAPTURE_MS_PER
total_vram_mb     = len(CAPTURED_BATCH_SIZES) * GRAPH_MEM_MB_PER

print(f"  Total startup cost: {total_capture_ms:.0f} ms  "
      f"(amortised over millions of tokens)")
print(f"  Total VRAM for graphs: {total_vram_mb:.0f} MB")
print()

def decode_time_ms(batch_size):
    """Memory-bound decode step time in ms."""
    return WEIGHTS_7B_GB / H100_BW_GBS * 1000

def padded_batch(actual_batch, captured_sizes):
    """Find the smallest captured size >= actual_batch."""
    for s in sorted(captured_sizes):
        if s >= actual_batch:
            return s
    return max(captured_sizes)

# Show padding overhead for different actual batch sizes
print(f"  Request handling: actual batch → padded batch → graph replay")
print()
print(f"  {'Actual batch':>14}  {'Padded batch':>14}  {'Padding%':>10}  "
      f"{'Decode time ms':>16}  {'Overhead fraction'}")
print("  " + "─" * 68)

for actual in [1, 3, 5, 9, 12, 20, 33, 65, 100, 128]:
    padded   = padded_batch(actual, CAPTURED_BATCH_SIZES)
    waste    = (padded - actual) / padded * 100
    t_ms     = decode_time_ms(padded)   # memory-bound: same time regardless of batch
    # Launch overhead: graph (negligible) vs conventional
    oh_frac  = LAUNCH_US_GRAPH / (t_ms * 1000) * 100
    print(f"  {actual:>14}  {padded:>14}  {waste:>9.1f}%  "
          f"{t_ms:>14.3f}  {oh_frac:>11.3f}%")

print()
print("  Memory-bound decode: time doesn't change with batch (same weights loaded).")
print("  Padding wastes computation proportional to waste% but frees graph benefit.")
print("  Padding < 50% is acceptable given graph launch is 10× faster.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Common CUDA Graph bugs catalogue
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Common CUDA Graph Bugs: Causes, Symptoms & Fixes")
print("━" * 68)
print()

bugs = [
    {"id":1,"name":"Stale device pointer",
     "cause":"Tensor re-allocated between replays; graph bakes in old GPU address.",
     "symptom":"Wrong results or SEGFAULT on replay (works fine at capture time).",
     "fix":"Allocate all tensors BEFORE capture; use copy_/fill_/zero_ in-place.",
     "detection":"Compare d_input address before capture vs after N replays."},
    {"id":2,"name":"cudaMalloc during capture",
     "cause":"Dynamic allocation (cudaMalloc, torch.empty()) inside capture region.",
     "symptom":"cudaErrorStreamCaptureInvalidated; capture fails immediately.",
     "fix":"Move all allocations before cudaStreamBeginCapture; use cudaMallocAsync.",
     "detection":"cudaStreamGetCaptureInfo returns interrupted status."},
    {"id":3,"name":"cudaDeviceSynchronize during capture",
     "cause":"Synchronous sync call is forbidden during stream capture mode.",
     "symptom":"Capture fails with cudaErrorStreamCaptureUnsupported.",
     "fix":"Replace with event-based sync; use cudaStreamWaitEvent instead.",
     "detection":"Set CUDA_LAUNCH_BLOCKING=1 to surface the call location."},
    {"id":4,"name":"Library internal synchronisation",
     "cause":"Older cuBLAS/cuDNN/NCCL call cudaDeviceSynchronize internally.",
     "symptom":"Library calls missing from the graph; results wrong on replay.",
     "fix":"Update to cuBLAS>=11.0, cuDNN>=8.0, NCCL>=2.9 (stream-capture aware).",
     "detection":"Call cudaStreamIsCapturing() around library call; check status."},
    {"id":5,"name":"cudaGraphExecUpdate topology mismatch",
     "cause":"New graph has different node count, kernel, or grid dims than exec.",
     "symptom":"cudaGraphExecUpdate returns CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE.",
     "fix":"Check result; call cudaGraphInstantiate again on failure.",
     "detection":"Inspect result.lastUpdateStatus != cudaGraphExecUpdateSuccess."},
]

for bug in bugs:
    print(f"  BUG {bug['id']}: {bug['name']}")
    print(f"    Cause:     {bug['cause']}")
    print(f"    Symptom:   {bug['symptom']}")
    print(f"    Fix:       {bug['fix']}")
    print(f"    Detection: {bug['detection']}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: PyTorch API — manual vs torch.compile
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — PyTorch API: Manual Graph vs torch.compile")
print("━" * 68)
print()

api_comparison = [
    ("API", "Manual torch.cuda.CUDAGraph", "torch.compile(reduce-overhead)"),
    ("Code complexity", "High (explicit capture/replay)", "Low (decorator + warmup)"),
    ("Control over capture", "Full (explicit begin/end)", "Automatic (traced by Dynamo)"),
    ("Handles dynamic shapes", "No (must recapture)", "Partial (recompiles on shape change)"),
    ("Multi-graph support", "Manual (dict of graphs)", "Automatic (per shape)"),
    ("Debugging", "Harder (graph is opaque)", "torch._dynamo.explain()"),
    ("Memory overhead", "Manual (know what you allocate)", "Automatic (may use more)"),
    ("Best for", "LLM serving (known shapes, max perf)", "General models, prototyping"),
    ("Integration", "Custom serving code", "Any PyTorch model, 1-line change"),
]

col_widths = [max(len(row[i]) for row in api_comparison) for i in range(3)]
col_widths = [max(w, 10) for w in col_widths]

for i, row in enumerate(api_comparison):
    sep = "  " + "─" * (sum(col_widths) + 8) if i == 1 else ""
    if sep:
        print(sep)
    print(f"  {row[0]:<{col_widths[0]}}  {row[1]:<{col_widths[1]}}  {row[2]}")

print()
print("  MANUAL GRAPH (minimal PyTorch example):")
print()
print("    stream = torch.cuda.Stream()")
print("    graph  = torch.cuda.CUDAGraph()")
print()
print("    # Pre-allocate static buffers")
print("    static_x = torch.empty(batch, seq, hidden, device='cuda')")
print()
print("    # Warmup (3 iterations to warm up allocator caches)")
print("    with torch.cuda.stream(stream):")
print("        for _ in range(3):")
print("            _ = model(static_x)")
print()
print("    # Capture")
print("    with torch.cuda.graph(graph, stream=stream):")
print("        static_out = model(static_x)")
print()
print("    # Inference loop (fast path)")
print("    static_x.copy_(real_input)   # in-place update")
print("    graph.replay()               # sub-microsecond launch")
print("    return static_out.clone()    # copy result")
print()
print("  torch.compile EQUIVALENT (1 line):")
print("    model_fast = torch.compile(model, mode='reduce-overhead')")
print("    out = model_fast(input)   # automatic graph capture + replay")
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