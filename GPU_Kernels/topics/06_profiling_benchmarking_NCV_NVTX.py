"""
NCU / NVTX Instrumentation — CLI Metrics, Replay & Range Markers
=================================================================

Nsight Compute (ncu) and NVTX are two halves of the same discipline.

ncu is the interrogation tool. It stops a kernel, replays it multiple
times with different hardware performance counters active, and returns
a structured report answering: which hardware units are saturated, which
are idle, and exactly which source lines are responsible. Used correctly,
it closes the loop between "my kernel is slow" and "here is the exact
machine instruction costing cycles."

NVTX is the annotation layer. It lets you embed semantic labels directly
into the GPU timeline — tagging ranges of code with names, colours, and
payloads that appear in both nsys and ncu. Without NVTX, the profiler
sees anonymous kernel names. With NVTX, it sees "layer_7/attention/qkv"
projected onto the exact kernels it launched.

Together they form a precision instrument loop:
    1. Annotate hot paths with NVTX so you know what you are profiling.
    2. Run ncu CLI with the right --metrics set targeting the hypothesis.
    3. Interpret the metric values against the roofline and stall taxonomy.
    4. Fix the bottleneck. Re-run. Compare.

This module covers both halves at implementation depth:
    ncu CLI    — replay mechanics, metric namespacing, section sets,
                 filtering by kernel name, source-level correlation,
                 comparing .ncu-rep files across runs
    NVTX       — C/C++/Python APIs, nested ranges, categories, payloads,
                 domain isolation, integration with torch.profiler
    Metric set — the 25 most important ncu metrics by name, unit, and
                 interpretation, grouped by hardware unit

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "NCU / NVTX Instrumentation — CLI Metrics, Replay & Range Markers"
DISPLAY_NAME = "06 · NCU / NVTX Instrumentation"
ICON         = "🔬"
SUBTITLE     = "ncu CLI · Metric Namespacing · NVTX Ranges · Source Correlation"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — NCU INTERNALS: HOW HARDWARE REPLAY WORKS

### The Hardware Performance Counter Limitation

    NVIDIA GPUs have hundreds of hardware performance counters —
    registers that count specific events: L1 cache misses, warp
    stall cycles, tensor core utilisation, bank conflicts, etc.

    The fundamental problem: the GPU cannot measure ALL counters
    simultaneously. The hardware has limited physical routing between
    the pipeline stages and the counter registers. A typical Ampere SM
    exposes ~1200 counters but can collect only ~12 simultaneously per pass.

    CONSEQUENCE: ncu replays the kernel N times, each time with a different
    set of 12 counters active. All N passes are then correlated to produce
    a single unified report. This means:

        1. The kernel must be IDEMPOTENT — same inputs, same outputs every
           replay. If your kernel has side effects (e.g., atomic increments
           modifying shared state), replays will corrupt results.

        2. Profiling ADDS LATENCY — a kernel that takes 5 ms normally may
           take 60–150 ms under ncu (10–30 replays). Do not profile inside
           latency-sensitive serving loops.

        3. Context matters — ncu's numbers are for ONE kernel invocation.
           If you profile the first iteration (cache cold), the numbers
           differ from steady-state (cache warm). Always warm up before
           profiling with --skip-before-launch.

### Replay Modes

    KERNEL REPLAY (default):
        ncu replays the ENTIRE KERNEL for each counter group.
        Most accurate. Required for source-level correlation.
        Cost: ~N_groups × kernel_duration.
        N_groups for --set full on Ampere: ~20–30 passes.

    APP REPLAY (--replay-mode application):
        ncu runs the ENTIRE APPLICATION multiple times, each time
        with a different counter group active.
        Better for: kernels that depend on preceding state (e.g., running
        statistics, iterative algorithms).
        Cost: ~N_groups × full_application_time. Very slow.

    RANGE REPLAY (--replay-mode range):
        Profile all kernels within a cudaProfilerStart/Stop range together.
        Intermediate between kernel and app replay.

    USER RANGE REPLAY (--replay-mode userrange):
        Profile all kernels within an NVTX range.
        The ncu --nvtx flag selects the range by name.
        Most useful for production code: annotate the hot function,
        profile only that NVTX range.

### What "Correlated" Means

    After all replay passes, ncu cross-correlates the counter values by
    scheduling index (which warp, which instruction, which SM partition).
    The result is a METRIC VALUE per SM partition, aggregated across warps.

    For source-level correlation: ncu maps each metric value to a SASS
    instruction (assembly), then back to the PTX, then to the CUDA C++ source
    line. This gives per-line bottleneck attribution.

    Source correlation requires:
        -lineinfo flag in nvcc compilation:
            nvcc -O3 -lineinfo kernel.cu -o kernel
        Or in PyTorch:
            torch._dynamo.config.inline_inbuilt_nn_modules = True
            TORCH_COMPILE_DEBUG=1


##### PART 2 — NCU CLI: COMMANDS, FLAGS & METRIC SECTIONS

### Section Sets: What Gets Measured

    ncu organises counters into SECTIONS. Use --set to select a preset:

    --set default                     ~10 counters, fast, gives Roofline + high-level stalls
    --set full                        ~1200 counters, slow (20–30 replays), complete picture
    --set detailed                    intermediate, covers memory + compute + scheduler
    --section SpeedOfLight            just the roofline / speed-of-light metrics
    --section MemoryWorkloadAnalysis  L1/L2/DRAM traffic + bandwidth
    --section SchedulerStatistics     warp issue rate + stall reasons
    --section WarpStateStatistics     per-stall-type warp cycle counts
    --section InstructionStatistics   instruction mix + throughput
    --section LaunchStatistics        grid/block config + occupancy
    --section SourceCounters          per-SASS-instruction counters

    PRACTICAL RULE:
        First run: --set default (fast, reveals dominant bottleneck)
        Second run: --section <relevant section> (drill into the bottleneck)
        Final run: --set full (only when submitting kernel for review)

### Filtering: Which Kernels to Profile

    Profile ALL kernels (slow for large programs):
        ncu -o report python train.py

    Profile by kernel NAME (regex match):
        ncu --kernel-name "flash_attention" -o out python train.py
        ncu --kernel-name ".*gemm.*" -o out python train.py

    Profile EXACTLY the Nth invocation of a kernel:
        ncu --kernel-name "my_kernel" --launch-count 1 python train.py
        ncu --kernel-name "my_kernel" --launch-skip 10 --launch-count 5 python

    Profile within an NVTX range:
        ncu --nvtx --nvtx-include "forward_pass" -o out python train.py
        # Only kernels launched inside push("forward_pass")...pop() are profiled

    Profile by CUDA context or stream:
        ncu --context-id 1 --stream-id 7 -o out python train.py

    Skip initial launches (allow warm-up):
        ncu --kernel-name "my_kernel" --launch-skip 5 --launch-count 1 ...
        # Profiles the 6th invocation, skipping first 5 (cache cold)

### Output Formats

    Interactive GUI report:
        ncu -o profile.ncu-rep python train.py
        # Open in Nsight Compute GUI

    Text output to terminal:
        ncu --print-summary per-kernel python train.py
        ncu --print-details all python train.py

    CSV export for scripting:
        ncu --csv --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
            python train.py

    Import into Python:
        ncu-python: NVIDIA's official ncu Python scripting interface
        ncu --import profile.ncu-rep --print-details all > report.txt

    Compare two runs (before/after optimisation):
        ncu --import-before before.ncu-rep --import-after after.ncu-rep \
            --diff-mode relative

### Source Correlation Command

    Full pipeline for CUDA C++ source attribution:
        nvcc -O3 -lineinfo --generate-line-info kernel.cu -o kernel
        ncu --set full --source-folder /path/to/src \
            --export profile.ncu-rep ./kernel

    Per-line SASS output:
        ncu --print-details all --print-source-level all --import profile.ncu-rep


##### PART 3 — THE METRIC NAMING CONVENTION: READING ANY METRIC NAME

### The Four-Part Metric Name

    Every ncu metric follows the pattern:
        unit__metric_description.aggregation.normalization

    UNIT: which hardware block produces this counter
        sm__       — streaming multiprocessor (whole SM)
        smsp__     — SM sub-partition (one of 4 per SM on Ampere)
        l1tex__    — L1 texture/data cache
        lts__      — L2 cache slice
        dram__     — HBM (device memory)
        pcie__     — PCIe interface
        nvlrx__    — NVLink receive
        nvltx__    — NVLink transmit
        fe__       — front-end (instruction fetch)
        idc__      — L1 instruction cache
        tpc__      — texture processing cluster

    METRIC DESCRIPTION: what is counted
        throughput           — bytes or operations per second
        warps_active         — warps executing this cycle
        cycles_elapsed       — total cycles
        bytes_read           — bytes read from this level
        bytes_write          — bytes written
        requests             — number of cache/memory requests
        sectors              — 32-byte cache sectors accessed
        hit_rate             — fraction of requests that hit

    AGGREGATION: how to combine across SMs / warps
        avg    — arithmetic mean across SMs
        sum    — sum across all SMs
        max    — maximum across any SM
        min    — minimum
        pct    — percentage (sum / theoretical_max)

    NORMALIZATION: what time period
        per_second                   — rate metric
        per_cycle_elapsed            — per GPU clock cycle
        per_cycle_active             — per cycle the SM had work
        pct_of_peak_sustained_elapsed — % of theoretical peak

### Worked Examples

    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        Unit: whole SM | What: aggregate throughput
        Aggregation: avg across SMs | Normalisation: % of peak over wall time
        → The Roofline "SM utilisation" headline metric.
        → 85% means SMs were active 85% of kernel duration on average.

    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
        Unit: SM sub-partition | What: stalled warps waiting for long-latency op
        Aggregation: per-warp fraction | Normalisation: % of active warp cycles
        → Long scoreboard stall = L2/HBM miss being waited on.
        → If > 30%: memory-bound stall, need better caching or access patterns.

    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"
        Unit: L1 texture cache | What: bytes from global load pipe
        Aggregation: sum across all SMs | Normalisation: absolute count
        → Total bytes that hit the L1 data cache from global memory loads.
        → Divide by l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum × 32
          to get coalescing efficiency.

    "dram__bytes_read.sum"
        Unit: HBM | What: bytes read from device memory
        Aggregation: sum across all DRAM controllers
        → Total HBM read traffic for the kernel.
        → Compare to arithmetic minimum (matrix dimensions × element size)
          to measure traffic inflation from poor access patterns.

### List All Available Metrics

    ncu --query-metrics | grep "dram__"      # all DRAM metrics
    ncu --query-metrics | grep "stalled"     # all stall metrics
    ncu --query-metrics > all_metrics.txt    # full list (~1200 metrics)

    On H100: ~1400 metrics. On A100: ~1200 metrics. On V100: ~800 metrics.
    Metric names are architecture-specific but follow the same pattern.


##### PART 4 — THE 25 ESSENTIAL NCU METRICS: GROUPED BY HARDWARE UNIT

### GROUP 1 — Roofline / Speed-of-Light (use these first)

    sm__throughput.avg.pct_of_peak_sustained_elapsed
        What:  Fraction of peak SM throughput achieved.
        Good:  > 80% (well-utilised).  Concerning: < 50%.
        If low: check whether memory-bound or compute-bound first.

    sm__sass_thread_inst_executed_op_ffma_pred_on.sum
        What:  Total FP32 FMA instructions executed (2 FLOPs each).
        Use:   Compute achieved FLOPs = value × 2, compare to peak.

    sm__cycles_elapsed.avg.per_second
        What:  SM clock frequency during kernel. Confirms no throttling.
        H100 base: 1980 MHz. If < 1800 MHz: thermal throttling.

### GROUP 2 — Memory Bandwidth (L1 → L2 → DRAM)

    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
        What:  Sectors (32 bytes each) loaded into L1 from global memory.
        Use:   sectors × 32 bytes = total L1 read traffic.
        Compare to: minimum sectors needed (array size / 32 bytes).
        Ratio > 1 indicates uncoalesced accesses.

    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
        What:  Sectors written from L1 to global memory.
        Same coalescing analysis applies to writes.

    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
        What:  Number of load requests to L1 (one per warp instruction).
        Use:   sectors_loaded / requests_loaded = sectors per transaction.
        Ideal: 1 sector per request (perfect coalescing for 32-byte loads).
        Worst: 32 sectors per request (stride-32 access, each lane in own sector).

    lts__t_sectors_srcunit_tex_op_read.sum
        What:  Sectors read from L2 cache by texture/L1 requests.
        Use:   L2 read traffic = value × 32 bytes.

    lts__t_sectors_srcunit_tex_op_read_hit_rate.pct
        What:  Fraction of L2 reads that hit the L2 cache.
        Good:  > 80% (working set fits in L2).  Concerning: < 40%.

    dram__bytes_read.sum
        What:  Total bytes read from HBM.
        Use:   Compute achieved read bandwidth = value / kernel_duration.
        Compare to peak HBM bandwidth (H100: 3.35 TB/s).

    dram__bytes_write.sum
        What:  Total bytes written to HBM.

    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
        What:  SMEM bank conflicts on loads.
        Good:  0.  Each conflict costs 1 extra cycle per degree of conflict.

    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
        What:  SMEM bank conflicts on stores.

### GROUP 3 — Warp Scheduler & Occupancy

    sm__warps_active.avg.per_cycle_active
        What:  Average number of warps resident and active per SM cycle.
        H100 max: 64 warps/SM.  Good: > 32. Low: < 8.
        Low value ≠ bad if stall analysis shows high latency hiding.

    smsp__occupancy.avg.pct
        What:  Achieved occupancy as % of theoretical maximum.
        = (active warps) / (max warps) × 100.
        Theoretical max depends on register + SMEM usage per block.

    smsp__issue_active.avg.per_cycle_active
        What:  Instructions issued per SM sub-partition cycle.
        Max on Ampere: 1 instruction/cycle/sub-partition.
        Good: > 0.6.  Low: < 0.3 (warp scheduler often has nothing to issue).

### GROUP 4 — Stall Reasons (add these when diagnosing slow warps)

    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
        What:  % of active warp cycles stalled waiting for L2/DRAM.
        If > 30%: memory latency is the bottleneck. Fix: better caching,
        prefetch, or more warps to hide latency.

    smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
        What:  Stalled waiting for L1 or SMEM (short latency ops).
        If > 20%: SMEM bank conflicts or register dependencies.

    smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
        What:  Stalled because compute pipeline (FP32/INT/Tensor) is full.
        If > 30%: compute-bound. This is acceptable — the pipeline is busy.

    smsp__warp_issue_stalled_barrier_per_warp_active.pct
        What:  Stalled at __syncthreads() waiting for other warps.
        If > 20%: divergent block execution causing long barrier waits.
        Fix: rebalance work across warps, reduce barrier frequency.

    smsp__warp_issue_stalled_membar_per_warp_active.pct
        What:  Stalled at memory fence (__threadfence, __threadfence_block).
        If > 10%: too many memory fences. Audit fence placement.

    smsp__warp_issue_stalled_not_selected_per_warp_active.pct
        What:  Warp is eligible (not stalled) but was not selected this cycle.
        This is HEALTHY — means multiple eligible warps exist and the scheduler
        is choosing between them. Low latency hiding is working.

    smsp__warp_issue_stalled_no_instruction_per_warp_active.pct
        What:  Instruction cache miss — waiting for instruction fetch.
        If > 5%: code size too large, icache thrashing. Reduce register spills
        or split into smaller kernels.

### GROUP 5 — Tensor Core / Mixed Precision

    sm__inst_executed_pipe_tensor_op_hmma.sum
        What:  HMMA (FP16 tensor core) instructions executed.
        If = 0 when expected: cuBLAS not using TC path, or wrong dtype.

    sm__inst_executed_pipe_tensor_op_imma.sum
        What:  IMMA (INT8 tensor core) instructions executed.

    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
        What:  Fraction of cycles tensor cores were active.
        Good: > 70% for GEMM-heavy kernels.  Low: pipeline not feeding TC.


##### PART 5 — NVTX: ARCHITECTURE, API, AND INTEGRATION

### What NVTX Is

    NVTX (NVIDIA Tools Extension) is a callback-based instrumentation API.
    At runtime, NVTX calls are intercepted by NVIDIA profiling tools.
    When no tool is attached: the calls are no-ops (< 10 ns overhead each).
    When nsys or ncu is attached: the calls emit timestamped events.

    NVTX events appear in:
        nsys timeline   — as coloured range bars (the primary use case)
        ncu             -- as range boundaries for --nvtx filtering
        NSight Compute  -- kernel attribution to enclosing NVTX range

### NVTX3 vs NVTX2: API Generations

    NVTX2 (legacy, C API):
        #include <nvToolsExt.h>
        nvtxRangePushA("my_range");       // ASCII string
        nvtxRangePushW(L"my_range");      // wide string
        nvtxRangePop();
        nvtxMarkA("event_name");          // instantaneous event (no duration)

    NVTX3 (modern, header-only C++17, zero-overhead when disabled):
        #include <nvtx3/nvtx3.hpp>
        nvtx3::scoped_range range{"my_range"};   // RAII: push on construct, pop on destruct
        // Or explicit:
        nvtx3::event_attributes attr{nvtx3::rgb{0, 128, 255}, "my_range"};
        auto id = nvtx3::push_range(attr);
        nvtx3::pop_range();

    NVTX3 advantages:
        Header-only (no link dependency on profiling release)
        RAII scoped_range handles push/pop via destructor
        Compile-time domain instantiation (zero cost when disabled)
        Structured payload support (int, float, double, pointer)

### Complete API Reference: nvtxRangePushEx

    nvtxEventAttributes_t is the full attribute struct:

        nvtxEventAttributes_t attr = {};
        attr.version        = NVTX_VERSION;
        attr.size           = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

        // Colour:
        attr.colorType      = NVTX_COLOR_ARGB;
        attr.color          = 0xFF_4488FF;   // alpha=FF, R=44, G=88, B=FF

        // Message:
        attr.messageType    = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii  = "layer_7/attention";

        // Payload (structured value attached to event):
        attr.payloadType    = NVTX_PAYLOAD_TYPE_INT64;
        attr.payload.llValue = layer_index;   // visible in GUI tooltip

        // Category (user-defined integer grouping):
        attr.category       = 3;   // e.g., 1=forward, 2=backward, 3=optimizer

        nvtxRangePushEx(&attr);
        // ... GPU work ...
        nvtxRangePop();

### NVTX Domains: Namespace Isolation

    A DOMAIN groups ranges under a named namespace, preventing conflicts
    between instrumentation from different libraries in the same process.

        nvtxDomainHandle_t domain = nvtxDomainCreateA("my_model");

        nvtxEventAttributes_t attr = {};
        attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = "attention_forward";
        nvtxDomainRangePushEx(domain, &attr);
        // ... work ...
        nvtxDomainRangePop(domain);

        nvtxDomainDestroy(domain);

    Without domains: all ranges are in the default global namespace.
    With domains: nsys and ncu can filter ranges by domain name:
        ncu --nvtx --nvtx-include "domain:my_model/attention_forward"

### Python API: Three Levels

    LEVEL 1 — torch.cuda.nvtx (simplest):
        import torch.cuda.nvtx as nvtx
        nvtx.range_push("forward_pass")
        output = model(x)
        nvtx.range_pop()

    LEVEL 2 — cupy.cuda.nvtx (if using CuPy):
        import cupy.cuda.nvtx as nvtx
        nvtx.RangePush("my_range", color=0xFF4488FF)
        ...
        nvtx.RangePop()

    LEVEL 3 — ctypes direct binding (full attribute control):
        import ctypes
        libcudart = ctypes.CDLL('libcudart.so')
        # See pynvtx library for a complete wrapper.

    torch.profiler integration (records NVTX ranges automatically):
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb")
        ) as prof:
            for step, batch in enumerate(dataloader):
                with torch.profiler.record_function("iteration"):
                    output = model(batch)
                prof.step()

### ncu --nvtx Filtering: Profiling Only What You Annotated

    Profile all kernels inside a specific NVTX range:
        ncu --nvtx --nvtx-include "forward_pass" -o out python train.py

    Profile kernels inside a NESTED range:
        ncu --nvtx --nvtx-include "forward_pass/attention" -o out python train.py
        # Only kernels launched while BOTH push("forward_pass") AND push("attention")
        # are on the stack are profiled.

    Profile by domain:
        ncu --nvtx --nvtx-include "domain:my_lib/attention" -o out python train.py

    Exclude ranges (profile everything EXCEPT):
        ncu --nvtx --nvtx-exclude "data_loading" -o out python train.py

    This makes ncu --nvtx the most precise targeting mechanism available:
    you annotate exactly the operation you care about, and ncu instruments
    only those kernels. No grep for kernel names. No counting launch indices.


##### PART 6 — THE PROFILING WORKFLOW: NCU + NVTX COMBINED

### The Four-Step Instrumentation Loop

    STEP 1 — ANNOTATE:
        Add NVTX ranges at the granularity of the hypothesis:
        - If you suspect attention is slow: push("attention")..pop()
        - If you suspect a specific layer: push(f"layer_{i}")..pop()
        - If you want the full forward: push("forward")..pop()

    STEP 2 — SYSTEM PROFILE (nsys):
        nsys profile --trace=cuda,nvtx -o timeline python train.py
        Open timeline. Find which NVTX-annotated region dominates.
        Confirm GPU utilisation during that region.
        → This gives you the TARGET for step 3.

    STEP 3 — KERNEL PROFILE (ncu):
        ncu --nvtx --nvtx-include "attention" \
            --set default \
            --launch-skip 5 --launch-count 3 \
            -o attention_profile python train.py

        Check: is the kernel memory-bound or compute-bound?
        Run appropriate section next:
            Memory-bound: --section MemoryWorkloadAnalysis
            Compute-bound: --section ComputeWorkloadAnalysis
            Scheduler issues: --section SchedulerStatistics

    STEP 4 — ITERATE:
        Fix the bottleneck. Add another NVTX level for the fix:
            push("attention_v2")   ← new NVTX name
            run ncu again
            compare:
                ncu --import before.ncu-rep --import after.ncu-rep

### Reading the ncu Report: The Six Key Numbers

    When you open an ncu report, check these six numbers in order:

    1. sm__throughput.avg.pct_of_peak_sustained_elapsed
       → Is the SM working? If < 60%: find where cycles are lost.

    2. dram__bytes_read.sum / kernel_duration_ns × 1e9 → HBM bandwidth (GB/s)
       → Compare to peak. If < 50% of peak: either not enough memory traffic
         (compute-bound) or inefficient traffic (bad access patterns).

    3. l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum /
       l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
       → Sectors per request. Ideal: 1 (coalesced). Bad: > 8 (strided).

    4. lts__t_sectors_srcunit_tex_op_read_hit_rate.pct
       → L2 hit rate. < 50% means working set exceeds 50 MB (A100 L2 = 40 MB).

    5. smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
       → Long stalls. > 30% = memory latency is the dominant cost.

    6. smsp__occupancy.avg.pct
       → Achieved / theoretical occupancy. Use in conjunction with stalls:
         Low occupancy + low stalls = GOOD (latency already hidden).
         Low occupancy + high stalls = BAD (not enough warps to hide latency).


##### PART 7 — COMMON INSTRUMENTATION MISTAKES AND HOW TO AVOID THEM

### Mistake 1: Profiling Cold (Cache-Cold First Iteration)

    SYMPTOM: L2 hit rate = 2%. DRAM bandwidth = peak.
    CAUSE: First kernel run loads all weights cold from HBM.
    FIX: --launch-skip 5 (skip first 5 invocations, profile warm steady state).

### Mistake 2: Mismatched NVTX push/pop (stack imbalance)

    SYMPTOM: nsys shows range extending far beyond expected code block.
    CAUSE: nvtx.range_pop() not called on all exception paths.
    FIX: Use RAII wrapper or context manager:
        with nvtx.range("attention"):   ← Python context manager auto-pops
            attn = self_attention(q, k, v)

    RAII C++ wrapper:
        struct NvtxRange {
            NvtxRange(const char* name) { nvtxRangePushA(name); }
            ~NvtxRange()               { nvtxRangePop(); }
        };
        // Usage:
        { NvtxRange r("attention"); attention(q,k,v); }  // pop on scope exit

### Mistake 3: Profiling with --set full on Every Run

    SYMPTOM: ncu takes 3 minutes per kernel. CI profiling times out.
    FIX: --set default for initial investigation (~5 s/kernel).
         --set full only when you need source-level attribution.

### Mistake 4: Using kernel name grep instead of NVTX

    SYMPTOM: wrong kernel profiled, or multiple unrelated kernels matched.
    CAUSE: --kernel-name matches on internal compiler-mangled names.
    FIX: Wrap the code in an NVTX range. Use --nvtx-include.
         NVTX ranges are semantic and stable across compiler versions.

### Mistake 5: Comparing absolute counter values across GPU architectures

    SYMPTOM: A100 shows 2× higher dram__bytes_read than V100 — is it worse?
    CAUSE: Different SM counts, different L2 sizes, different counter granularity.
    FIX: Always normalise:
        Bandwidth (GB/s) = dram__bytes_read.sum / duration_ns × 1e9
        Bandwidth efficiency (%) = achieved_bw / peak_bw × 100
        These are architecture-independent metrics.

### Mistake 6: Trusting occupancy alone

    SYMPTOM: 12% occupancy → engineer says "we need to fix occupancy".
    REALITY: FlashAttention runs at 6–12% occupancy by design (large SMEM).
             It is fully memory-bandwidth-bound and hides latency perfectly.
    FIX: Check smsp__warp_issue_stalled_not_selected.pct.
         High "not selected" means warps ARE eligible — low occupancy is fine.
         Only low occupancy + high long_scoreboard stall is a real problem.

### Mistake 7: Forgetting to compile with -lineinfo

    SYMPTOM: ncu source-level view shows "line info unavailable".
    FIX: Add -lineinfo to nvcc flags. For PyTorch compiled extensions:
        extra_compile_args={'nvcc': ['-O3', '-lineinfo']}
    Cost: ~5% larger binary, zero runtime overhead.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Metric Name Parser & Decoder — Full Taxonomy Walkthrough": {
        "description": (
            "Parse and decode every component of ncu metric names using the "
            "four-part naming convention (unit, description, aggregation, "
            "normalisation). Build a decoder that takes any metric string and "
            "produces a human-readable interpretation. Walk through all 25 "
            "essential metrics grouped by hardware unit, showing their expected "
            "ranges, what 'good' and 'bad' values look like, and what to do next."
        ),
        "language": "python",
        "code": '''
import re
from collections import defaultdict

print("=" * 68)
print("  METRIC NAME PARSER — NCU Four-Part Naming Convention")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# Decoder tables
# ─────────────────────────────────────────────────────────────────────

UNITS = {
    "sm":     ("Streaming Multiprocessor",    "whole SM aggregate"),
    "smsp":   ("SM Sub-Partition",            "one of 4 sub-partitions per SM"),
    "l1tex":  ("L1 Texture/Data Cache",       "L1 cache (includes SMEM)"),
    "lts":    ("L2 Cache Slice",              "one L2 slice (many per GPU)"),
    "dram":   ("HBM (Device Memory)",         "main GPU memory"),
    "pcie":   ("PCIe Interface",              "CPU-GPU bus"),
    "nvlrx":  ("NVLink Receive",              "inbound NVLink traffic"),
    "nvltx":  ("NVLink Transmit",             "outbound NVLink traffic"),
    "fe":     ("Front-End",                   "instruction fetch / scheduling"),
    "idc":    ("L1 Instruction Cache",        "instruction cache"),
    "tpc":    ("Texture Processing Cluster",  "TPC-level aggregates"),
    "gpc":    ("General Processing Cluster",  "GPC-level aggregates"),
}

AGGREGATIONS = {
    "avg":  "arithmetic mean across all SMs / sub-partitions",
    "sum":  "sum across all SMs / sub-partitions",
    "max":  "maximum across any single SM",
    "min":  "minimum across any SM",
    "pct":  "expressed as percentage (0–100)",
}

NORMALIZATIONS = {
    "per_second":                    "rate: events per wall-clock second",
    "per_cycle_elapsed":             "events per total GPU clock cycle",
    "per_cycle_active":              "events per cycle the SM had active warps",
    "pct_of_peak_sustained_elapsed": "% of theoretical peak over kernel duration",
    "pct_of_peak_sustained_active":  "% of peak during only active cycles",
    "":                              "absolute count (no normalisation)",
}

def parse_metric(metric_str):
    """
    Decode an ncu metric string into its four components.
    Returns dict with unit, description, aggregation, normalization.
    """
    parts = metric_str.split(".")
    # First part: unit__description (double underscore separator)
    if "__" in parts[0]:
        unit_raw, description = parts[0].split("__", 1)
    else:
        unit_raw, description = parts[0], ""

    agg   = parts[1] if len(parts) > 1 else ""
    norm  = ".".join(parts[2:]) if len(parts) > 2 else ""

    unit_info = UNITS.get(unit_raw, ("Unknown unit", ""))
    agg_info  = AGGREGATIONS.get(agg, f"'{agg}' (non-standard aggregation)")
    norm_info = NORMALIZATIONS.get(norm, f"'{norm}' (normalisation)")

    return {
        "raw":         metric_str,
        "unit_key":    unit_raw,
        "unit_name":   unit_info[0],
        "unit_detail": unit_info[1],
        "description": description,
        "aggregation": agg,
        "agg_meaning": agg_info,
        "normalization": norm,
        "norm_meaning":  norm_info,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Parse and explain the 25 essential metrics
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — The 25 Essential NCU Metrics: Decoded")
print("━" * 68)
print()

ESSENTIAL_METRICS = [
    # (metric_string, good_threshold, concerning_threshold, interpretation, fix_if_bad)
    # GROUP 1 — Roofline
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
     "> 80%", "< 50%",
     "SM aggregate throughput as % of theoretical peak",
     "Find whether memory-bound or compute-bound"),
    ("sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
     "matches FLOPs/peak", "0",
     "Total FP32 FMA instructions (× 2 = FLOPs)",
     "If 0: check dtype, __syncwarp divergence"),
    ("sm__cycles_elapsed.avg.per_second",
     "≥ 1800 MHz (H100)", "< 1600 MHz",
     "SM clock frequency — detects thermal throttle",
     "Check thermal paste, power limits, ambient temp"),
    # GROUP 2 — Memory
    ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
     "≈ array_bytes/32", "> 8× ideal",
     "L1 sectors loaded (32 B each) from global memory",
     "Audit access patterns, use __ldg / read-only cache"),
    ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
     "= sectors (coalesced)", "sectors >> requests",
     "Number of warp load requests to L1",
     "sectors/requests ratio > 4: fix coalescing"),
    ("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
     "≈ array_bytes/32", "> 8× ideal",
     "L1 sectors written to global memory",
     "Strided writes: restructure data layout"),
    ("lts__t_sectors_srcunit_tex_op_read.sum",
     "≤ L1 sectors (filtered)", "> 3× L1",
     "L2 read traffic (sectors × 32 B = bytes)",
     "High L2 traffic: data exceeds L1 capacity"),
    ("lts__t_sectors_srcunit_tex_op_read_hit_rate.pct",
     "> 80%", "< 40%",
     "L2 cache hit rate for read requests",
     "< 40%: working set > L2 size. Tiling needed"),
    ("dram__bytes_read.sum",
     "≤ ideal traffic", "> 3× ideal",
     "Total HBM bytes read",
     "HBM inflation: improve reuse (tiling, SMEM)"),
    ("dram__bytes_write.sum",
     "≤ output size", "> 3× output",
     "Total HBM bytes written",
     "Excess writes: remove redundant stores"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
     "0", "> 0",
     "SMEM bank conflicts on loads",
     "Add +1 column padding to SMEM arrays"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
     "0", "> 0",
     "SMEM bank conflicts on stores",
     "Same fix: padding or swizzled SMEM layout"),
    # GROUP 3 — Occupancy / Scheduler
    ("sm__warps_active.avg.per_cycle_active",
     "> 32/SM", "< 8/SM",
     "Average warps resident per SM cycle",
     "Low: check register/SMEM pressure, block size"),
    ("smsp__occupancy.avg.pct",
     "> 50%", "< 25% with high stalls",
     "Achieved / theoretical occupancy %",
     "Low + high long_scoreboard stall: add warps"),
    ("smsp__issue_active.avg.per_cycle_active",
     "> 0.6", "< 0.3",
     "Instructions issued per sub-partition cycle",
     "Low: warp scheduler idle — check all stalls"),
    # GROUP 4 — Stall Reasons
    ("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
     "< 15%", "> 30%",
     "Warps stalled awaiting L2/DRAM (long latency)",
     "Improve caching, access patterns, or add warps"),
    ("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
     "< 10%", "> 20%",
     "Warps stalled awaiting L1/SMEM (short latency)",
     "Fix SMEM bank conflicts, shorten dep chains"),
    ("smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
     "OK if compute-bound", "> 30% unexpected",
     "Warps stalled because FP/INT pipes are full",
     "GOOD if intentional. If unexpected: ILP issue"),
    ("smsp__warp_issue_stalled_barrier_per_warp_active.pct",
     "< 10%", "> 20%",
     "Warps waiting at __syncthreads()",
     "Uneven work per warp; reduce barrier frequency"),
    ("smsp__warp_issue_stalled_membar_per_warp_active.pct",
     "< 5%", "> 10%",
     "Warps stalled at __threadfence() / membar",
     "Remove redundant memory fences"),
    ("smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
     "HIGH is GOOD", "0%",
     "Eligible warps not chosen this cycle (HEALTHY)",
     "Low: not enough warps to hide latency"),
    ("smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
     "< 3%", "> 8%",
     "Icache miss — waiting for instruction fetch",
     "Reduce kernel code size; avoid register spills"),
    # GROUP 5 — Tensor Cores
    ("sm__inst_executed_pipe_tensor_op_hmma.sum",
     "> 0 for FP16 GEMM", "= 0",
     "HMMA (FP16 tensor core) instructions executed",
     "0: wrong dtype or cuBLAS not using TC path"),
    ("sm__inst_executed_pipe_tensor_op_imma.sum",
     "> 0 for INT8 ops", "= 0 unexpected",
     "IMMA (INT8 tensor core) instructions executed",
     "0 with INT8 input: check quantisation config"),
    ("sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
     "> 70% (GEMM)", "< 30%",
     "% of cycles tensor cores were active",
     "Low: pipeline not feeding tensor cores fast enough"),
]

groups = [
    ("Roofline / Speed-of-Light", 0, 3),
    ("Memory Bandwidth (L1→L2→DRAM→SMEM)", 3, 12),
    ("Warp Scheduler & Occupancy", 12, 15),
    ("Stall Reasons", 15, 22),
    ("Tensor Core / Mixed Precision", 22, 25),
]

for grp_name, start, end in groups:
    print(f"  ── GROUP: {grp_name} {'─' * max(1, 50 - len(grp_name))}")
    print()
    for metric_str, good, concerning, meaning, fix in ESSENTIAL_METRICS[start:end]:
        p = parse_metric(metric_str)
        print(f"  METRIC:  {metric_str}")
        print(f"  Unit:    {p['unit_name']} ({p['unit_key']})")
        print(f"  Meaning: {meaning}")
        print(f"  Good: {good:18s}   Concerning: {concerning}")
        print(f"  Fix:     {fix}")
        print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Metric name decoder for arbitrary metric strings
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Interactive Decoder: Parse Any Metric String")
print("━" * 68)
print()

test_metrics = [
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "dram__bytes_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_hit_rate.pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "sm__inst_executed_pipe_tensor_op_hmma.sum",
]

for m in test_metrics:
    p = parse_metric(m)
    print(f"  Input:   {p['raw']}")
    print(f"  ├ Unit:  [{p['unit_key']}] → {p['unit_name']}: {p['unit_detail']}")
    print(f"  ├ Desc:  {p['description']}")
    print(f"  ├ Aggr:  [{p['aggregation']}] → {p['agg_meaning']}")
    print(f"  └ Norm:  [{p['normalization']}] → {p['norm_meaning']}")
    print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · NCU Report Interpreter — Six-Number Diagnostic Sequence": {
        "description": (
            "Simulate a realistic ncu metric report for five different kernel "
            "archetypes (naïve GEMM, tiled GEMM, uncoalesced elementwise, "
            "coalesced elementwise, LayerNorm). Apply the six-number diagnostic "
            "sequence to each. Diagnose the bottleneck type, compute achieved "
            "bandwidth and FLOP/s, and produce a prioritised fix list from the "
            "metric values exactly as an engineer would reading the ncu report."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

print("=" * 68)
print("  NCU REPORT INTERPRETER — Six-Number Diagnostic Sequence")
print("=" * 68)
print()


# ─────────────────────────────────────────────────────────────────────
# Simulated ncu metric reports for 5 kernel archetypes
# ─────────────────────────────────────────────────────────────────────
# Values represent realistic H100 SXM5 measurements

@dataclass
class NcuReport:
    kernel_name:  str
    duration_ns:  float
    # Roofline
    sm_throughput_pct:   float
    fma_instructions:    float    # total FP32 FMA ops
    sm_clock_mhz:        float
    # Memory
    l1_sectors_load:     float    # sectors loaded into L1
    l1_requests_load:    float    # warp load requests
    l2_read_sectors:     float
    l2_hit_rate_pct:     float
    dram_bytes_read:     float    # bytes
    dram_bytes_write:    float
    smem_bank_conflicts_ld: float
    smem_bank_conflicts_st: float
    # Scheduler
    warps_active_per_sm:   float
    occupancy_pct:         float
    issue_active_rate:     float
    # Stalls (% of active warp cycles)
    stall_long_scoreboard_pct: float
    stall_short_scoreboard_pct: float
    stall_math_throttle_pct:   float
    stall_barrier_pct:         float
    stall_not_selected_pct:    float
    # Tensor cores
    hmma_instructions:    float

# H100 hardware constants
H100_PEAK_FP32_TFLOPS = 67.0
H100_HBM_BW_GBS       = 3350.0
H100_SMs              = 132

REPORTS = {
    "Naïve GEMM (no tiling, global mem only)": NcuReport(
        kernel_name="naive_gemm_kernel",
        duration_ns=85_000_000,
        sm_throughput_pct=8.2,
        fma_instructions=8.59e9,
        sm_clock_mhz=1980,
        l1_sectors_load=2.1e9,
        l1_requests_load=2.1e8,
        l2_read_sectors=2.05e9,
        l2_hit_rate_pct=4.1,
        dram_bytes_read=65.6e9,
        dram_bytes_write=0.5e9,
        smem_bank_conflicts_ld=0,
        smem_bank_conflicts_st=0,
        warps_active_per_sm=48,
        occupancy_pct=75.0,
        issue_active_rate=0.12,
        stall_long_scoreboard_pct=78.0,
        stall_short_scoreboard_pct=2.0,
        stall_math_throttle_pct=1.5,
        stall_barrier_pct=1.0,
        stall_not_selected_pct=8.0,
        hmma_instructions=0,
    ),
    "Tiled GEMM (SMEM + register tiling)": NcuReport(
        kernel_name="tiled_gemm_smem",
        duration_ns=4_200_000,
        sm_throughput_pct=72.0,
        fma_instructions=8.59e9,
        sm_clock_mhz=1980,
        l1_sectors_load=98e6,
        l1_requests_load=10.8e6,
        l2_read_sectors=95e6,
        l2_hit_rate_pct=82.0,
        dram_bytes_read=3.04e9,
        dram_bytes_write=0.5e9,
        smem_bank_conflicts_ld=0,
        smem_bank_conflicts_st=0,
        warps_active_per_sm=52,
        occupancy_pct=81.0,
        issue_active_rate=0.71,
        stall_long_scoreboard_pct=8.0,
        stall_short_scoreboard_pct=4.0,
        stall_math_throttle_pct=38.0,
        stall_barrier_pct=6.0,
        stall_not_selected_pct=31.0,
        hmma_instructions=0,
    ),
    "Uncoalesced elementwise (stride-32 access)": NcuReport(
        kernel_name="elementwise_strided",
        duration_ns=18_000_000,
        sm_throughput_pct=15.0,
        fma_instructions=1.07e8,
        sm_clock_mhz=1980,
        l1_sectors_load=3.36e9,
        l1_requests_load=1.05e8,
        l2_read_sectors=3.30e9,
        l2_hit_rate_pct=2.0,
        dram_bytes_read=105.6e9,
        dram_bytes_write=3.3e9,
        smem_bank_conflicts_ld=0,
        smem_bank_conflicts_st=0,
        warps_active_per_sm=28,
        occupancy_pct=44.0,
        issue_active_rate=0.14,
        stall_long_scoreboard_pct=72.0,
        stall_short_scoreboard_pct=3.0,
        stall_math_throttle_pct=0.5,
        stall_barrier_pct=0.2,
        stall_not_selected_pct=6.0,
        hmma_instructions=0,
    ),
    "Coalesced elementwise (stride-1, GELU)": NcuReport(
        kernel_name="gelu_fwd_coalesced",
        duration_ns=1_450_000,
        sm_throughput_pct=88.0,
        fma_instructions=3.22e8,
        sm_clock_mhz=1980,
        l1_sectors_load=108e6,
        l1_requests_load=107e6,
        l2_read_sectors=104e6,
        l2_hit_rate_pct=72.0,
        dram_bytes_read=3.32e9,
        dram_bytes_write=3.32e9,
        smem_bank_conflicts_ld=0,
        smem_bank_conflicts_st=0,
        warps_active_per_sm=38,
        occupancy_pct=59.0,
        issue_active_rate=0.82,
        stall_long_scoreboard_pct=9.0,
        stall_short_scoreboard_pct=2.5,
        stall_math_throttle_pct=12.0,
        stall_barrier_pct=0.5,
        stall_not_selected_pct=52.0,
        hmma_instructions=0,
    ),
    "LayerNorm (2-pass, SMEM bank conflict)": NcuReport(
        kernel_name="layernorm_fwd_2pass",
        duration_ns=3_800_000,
        sm_throughput_pct=41.0,
        fma_instructions=1.61e8,
        sm_clock_mhz=1980,
        l1_sectors_load=420e6,
        l1_requests_load=108e6,
        l2_read_sectors=185e6,
        l2_hit_rate_pct=55.0,
        dram_bytes_read=6.08e9,
        dram_bytes_write=1.52e9,
        smem_bank_conflicts_ld=98e6,
        smem_bank_conflicts_st=48e6,
        warps_active_per_sm=22,
        occupancy_pct=34.0,
        issue_active_rate=0.38,
        stall_long_scoreboard_pct=28.0,
        stall_short_scoreboard_pct=24.0,
        stall_math_throttle_pct=4.0,
        stall_barrier_pct=18.0,
        stall_not_selected_pct=9.0,
        hmma_instructions=0,
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Six-number diagnostic function
# ─────────────────────────────────────────────────────────────────────

def diagnose(r: NcuReport):
    duration_s = r.duration_ns / 1e9

    # 1. SM throughput
    sm_ok = r.sm_throughput_pct > 70

    # 2. HBM bandwidth achieved
    hbm_bw_gbs  = (r.dram_bytes_read + r.dram_bytes_write) / duration_s / 1e9
    hbm_pct     = hbm_bw_gbs / H100_HBM_BW_GBS * 100

    # 3. Coalescing: sectors per request
    sectors_per_req = r.l1_sectors_load / max(r.l1_requests_load, 1)

    # 4. L2 hit rate
    l2_ok = r.l2_hit_rate_pct > 60

    # 5. Long scoreboard stall
    stall_mem_bad = r.stall_long_scoreboard_pct > 30

    # 6. Occupancy + stalls
    occ_stall_issue = r.occupancy_pct < 40 and r.stall_long_scoreboard_pct > 20

    # Determine bottleneck
    if r.stall_math_throttle_pct > 30:
        bottleneck = "COMPUTE-BOUND (math pipe saturated — good)"
    elif r.stall_long_scoreboard_pct > 40:
        bottleneck = "MEMORY-BOUND (HBM latency dominant)"
    elif r.smem_bank_conflicts_ld + r.smem_bank_conflicts_st > 1e6:
        bottleneck = "SMEM CONFLICT (bank conflict stalls)"
    elif sectors_per_req > 6:
        bottleneck = "UNCOALESCED (strided global memory)"
    elif r.stall_barrier_pct > 15:
        bottleneck = "BARRIER STALLS (divergent thread blocks)"
    else:
        bottleneck = "BALANCED (near-optimal)"

    # Build fix list
    fixes = []
    if sectors_per_req > 4:
        fixes.append(f"Coalescing: {sectors_per_req:.1f} sectors/req → restructure array layout")
    if r.l2_hit_rate_pct < 50:
        fixes.append(f"L2 miss rate {100-r.l2_hit_rate_pct:.0f}% → tiling, prefetch, __ldg")
    if r.smem_bank_conflicts_ld > 1e6:
        fixes.append(f"SMEM load conflicts: {r.smem_bank_conflicts_ld/1e6:.0f}M → add +1 col padding")
    if r.smem_bank_conflicts_st > 1e6:
        fixes.append(f"SMEM store conflicts: {r.smem_bank_conflicts_st/1e6:.0f}M → add +1 col padding")
    if r.stall_barrier_pct > 15:
        fixes.append(f"Barrier stall {r.stall_barrier_pct:.0f}% → balance warp work, reduce __syncthreads")
    if occ_stall_issue:
        fixes.append(f"Low occ {r.occupancy_pct:.0f}% + high mem stall → more warps or register reduction")
    if not fixes:
        fixes.append("No critical fixes — kernel is near-optimal")

    return {
        "sm_throughput_pct": r.sm_throughput_pct,
        "hbm_bw_gbs":        hbm_bw_gbs,
        "hbm_eff_pct":       hbm_pct,
        "sectors_per_req":   sectors_per_req,
        "l2_hit_pct":        r.l2_hit_rate_pct,
        "stall_long_pct":    r.stall_long_scoreboard_pct,
        "occupancy_pct":     r.occupancy_pct,
        "bottleneck":        bottleneck,
        "fixes":             fixes,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Full diagnostic for all 5 kernels
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Full Six-Number Diagnostic on 5 Kernel Archetypes")
print("━" * 68)
print()

diagnostics = {}
for name, report in REPORTS.items():
    diag = diagnose(report)
    diagnostics[name] = diag

    print(f"  ┌─ {name}")
    print(f"  │  Duration: {report.duration_ns/1e6:.2f} ms")
    print(f"  │")
    print(f"  │  [1] SM throughput:  {diag['sm_throughput_pct']:>6.1f}%  "
          f"{'✅' if diag['sm_throughput_pct'] > 70 else '❌'}")
    print(f"  │  [2] HBM bandwidth: {diag['hbm_bw_gbs']:>6.1f} GB/s  "
          f"({diag['hbm_eff_pct']:.1f}% of {H100_HBM_BW_GBS} GB/s peak)  "
          f"{'✅' if diag['hbm_eff_pct'] > 70 else '⚠' if diag['hbm_eff_pct'] > 40 else '❌'}")
    print(f"  │  [3] Sectors/req:   {diag['sectors_per_req']:>6.1f}     "
          f"{'✅ coalesced' if diag['sectors_per_req'] < 2 else '❌ strided' if diag['sectors_per_req'] > 6 else '⚠ partial'}")
    print(f"  │  [4] L2 hit rate:   {diag['l2_hit_pct']:>6.1f}%  "
          f"{'✅' if diag['l2_hit_pct'] > 70 else '⚠' if diag['l2_hit_pct'] > 40 else '❌'}")
    print(f"  │  [5] Long stall:    {diag['stall_long_pct']:>6.1f}%  "
          f"{'✅' if diag['stall_long_pct'] < 15 else '⚠' if diag['stall_long_pct'] < 30 else '❌'}")
    print(f"  │  [6] Occupancy:     {diag['occupancy_pct']:>6.1f}%  "
          f"{'✅' if diag['occupancy_pct'] > 60 else '⚠'}")
    print(f"  │")
    print(f"  │  DIAGNOSIS: {diag['bottleneck']}")
    print(f"  │  FIXES:")
    for fix in diag['fixes']:
        print(f"  │    → {fix}")
    print(f"  └{'─' * 64}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Summary comparison table
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Summary Comparison Table")
print("━" * 68)
print()

print(f"  {'Kernel':<38}  {'SM%':>5}  {'HBM%':>6}  {'Sec/Req':>8}  "
      f"{'L2%':>5}  {'Stall%':>7}  {'Occ%':>6}")
print("  " + "─" * 78)
for name, report in REPORTS.items():
    d = diagnostics[name]
    short = name[:36]
    print(f"  {short:<38}  {d['sm_throughput_pct']:>5.1f}  "
          f"{d['hbm_eff_pct']:>6.1f}  {d['sectors_per_req']:>8.1f}  "
          f"{d['l2_hit_pct']:>5.1f}  {d['stall_long_pct']:>7.1f}  "
          f"{d['occupancy_pct']:>6.1f}")

print()
print("  SM% = SM throughput (>70 good)   HBM% = HBM BW efficiency (>70 good)")
print("  Sec/Req = sectors per load request (1=perfect, >6=strided)")
print("  L2% = L2 hit rate (>70 good)      Stall% = long scoreboard stall (<15 good)")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · NVTX Range Tracker — Stack Simulation & Nesting Validation": {
        "description": (
            "Implement a full NVTX push/pop stack simulator that tracks range "
            "nesting, detects stack imbalances, computes per-range GPU time, and "
            "generates a hierarchical timing report. Show the consequences of a "
            "missing pop (range extends to end of profile). Build the context "
            "manager wrapper that guarantees pop on all code paths. Demonstrate "
            "the --nvtx-include filter logic that ncu uses to select kernels."
        ),
        "language": "python",
        "code": '''
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict

print("=" * 68)
print("  NVTX RANGE TRACKER — Stack, Nesting, Timing & ncu Filter Logic")
print("=" * 68)
print()


# ─────────────────────────────────────────────────────────────────────
# NVTX simulator
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NvtxEvent:
    kind:    str      # push | pop | kernel
    name:    str
    time_us: float
    domain:  str = "default"

@dataclass
class RangeRecord:
    name:      str
    domain:    str
    push_us:   float
    pop_us:    float = 0.0
    depth:     int   = 0
    parent:    Optional[str] = None
    children:  List = field(default_factory=list)
    kernels:   List = field(default_factory=list)

    @property
    def duration_us(self):
        return self.pop_us - self.push_us if self.pop_us > 0 else -1.0


class NvtxStackSimulator:
    def __init__(self):
        self.stack:   List[RangeRecord] = []
        self.records: List[RangeRecord] = []
        self.errors:  List[str]         = []

    def push(self, name, time_us, domain="default"):
        depth  = len(self.stack)
        parent = self.stack[-1].name if self.stack else None
        rec    = RangeRecord(name=name, domain=domain,
                             push_us=time_us, depth=depth, parent=parent)
        if self.stack:
            self.stack[-1].children.append(rec)
        self.stack.append(rec)
        self.records.append(rec)

    def pop(self, time_us, domain="default"):
        if not self.stack:
            self.errors.append(f"  POP at t={time_us:.0f}µs: stack was empty! (unmatched pop)")
            return
        rec        = self.stack.pop()
        rec.pop_us = time_us

    def kernel_launch(self, name, time_us, duration_us):
        """Record which NVTX ranges are active when a kernel is launched."""
        active_ranges = [r.name for r in self.stack]
        for rec in self.stack:
            rec.kernels.append({
                'name': name, 'start_us': time_us,
                'duration_us': duration_us, 'active_stack': list(active_ranges)
            })

    def close_profile(self, end_time_us):
        """Close any unclosed ranges at profile end."""
        while self.stack:
            rec = self.stack.pop()
            rec.pop_us = end_time_us
            self.errors.append(
                f"  UNCLOSED RANGE '{rec.name}' at depth {rec.depth}: "
                f"pop missing, auto-closed at t={end_time_us:.0f}µs")

    def gpu_time_for_range(self, range_name):
        """Total GPU kernel time attributed to a range."""
        for rec in self.records:
            if rec.name == range_name:
                return sum(k['duration_us'] for k in rec.kernels)
        return 0.0

    def kernels_matching_nvtx_filter(self, include_filter):
        """
        Simulate ncu --nvtx-include logic.
        include_filter can be a single range name or "parent/child" nested path.
        A kernel matches if the filter path is a suffix of its active range stack.
        """
        parts = include_filter.split("/")
        matched = []
        for rec in self.records:
            for kernel in rec.kernels:
                stack = kernel['active_stack']
                # Check if parts appear in order as a subsequence
                j = 0
                for s in stack:
                    if j < len(parts) and s == parts[j]:
                        j += 1
                if j == len(parts):
                    if kernel['name'] not in [m['name'] for m in matched]:
                        matched.append(kernel)
        return matched


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Well-formed transformer training iteration
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Well-Formed Transformer Iteration: Range Hierarchy")
print("━" * 68)
print()

sim = NvtxStackSimulator()
t = 0.0

def k(sim, name, t_start, dur):
    sim.kernel_launch(name, t_start, dur)
    return t_start + dur

# Iteration 42 — full annotated training step
sim.push("iteration_42", t)

t += 10
sim.push("data_to_device", t)
t = k(sim, "H2D_batch_tokens", t, 850)
sim.pop(t); t += 5

sim.push("forward_pass", t)

sim.push("embedding_lookup", t + 2)
t = k(sim, "embedding_forward", t + 2, 180)
sim.pop(t); t += 3

for layer_idx in range(4):
    sim.push(f"layer_{layer_idx}", t)
    sim.push(f"layer_{layer_idx}/attention", t)
    for kname, dur in [("attn_qkv", 180), ("attn_bmm", 220),
                        ("attn_softmax", 45), ("attn_out", 175)]:
        t = k(sim, f"{kname}_l{layer_idx}", t + 1, dur)
    sim.pop(t)    # attention
    sim.push(f"layer_{layer_idx}/ffn", t)
    for kname, dur in [("ffn_gate", 340), ("ffn_down", 340)]:
        t = k(sim, f"{kname}_l{layer_idx}", t + 1, dur)
    sim.pop(t)    # ffn
    sim.pop(t)    # layer_N
    t += 2

sim.pop(t)   # forward_pass

sim.push("backward_pass", t + 5); t += 5
t = k(sim, "transformer_bwd", t, 15_000)
sim.pop(t); t += 5

sim.push("optimizer_step", t)
t = k(sim, "adam_update", t, 3_200)
sim.pop(t)

sim.pop(t)   # iteration_42

# Report
print(f"  NVTX range hierarchy (depth-indented):")
print()
top_level = [r for r in sim.records if r.depth == 0]
def print_range(rec, indent=0):
    gpu_t = sum(k['duration_us'] for k in rec.kernels)
    dur   = rec.duration_us
    n_k   = len(rec.kernels)
    pfx   = "  " + "    " * indent + ("└─ " if indent > 0 else "")
    print(f"{pfx}{rec.name:<35}  dur={dur/1000:>6.1f}ms  "
          f"kernels={n_k:>3}  gpu={gpu_t/1000:>6.1f}ms")
    shown_children = {}
    for child in rec.children:
        cname = child.name
        if "layer_" in cname and cname not in ["layer_0"]:
            # Skip layers 1-3 for brevity, summarise
            pass
        if cname not in shown_children:
            shown_children[cname] = child
            print_range(child, indent + 1)

for rec in top_level:
    print_range(rec)

print()
if sim.errors:
    print(f"  ERRORS: {sim.errors}")
else:
    print(f"  Stack balance: ✅ all pushes matched with pops")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Missing pop — the silent timeline corruption
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Missing Pop: Stack Imbalance & Auto-Close")
print("━" * 68)
print()

bad_sim = NvtxStackSimulator()
t2 = 0.0
bad_sim.push("forward_pass", t2);  t2 += 100
bad_sim.push("attention", t2);     t2 += 50
bad_sim.kernel_launch("attn_qkv", t2, 180);  t2 += 180
# ← MISSING: bad_sim.pop(t2)  for "attention"
bad_sim.pop(t2)   # this pops "attention" but programmer thought it pops "forward_pass"
bad_sim.push("ffn", t2);  t2 += 50
bad_sim.kernel_launch("ffn_gate", t2, 340);  t2 += 340
bad_sim.pop(t2)   # pops "ffn"
# forward_pass never popped
bad_sim.close_profile(end_time_us=t2 + 5000)

print("  Code (with bug):")
print("    nvtx.range_push('forward_pass')")
print("    nvtx.range_push('attention')")
print("    attn_qkv()")
print("    # BUG: forgot nvtx.range_pop() for 'attention'")
print("    nvtx.range_pop()   ← this actually pops 'attention', not 'forward_pass'")
print("    nvtx.range_push('ffn')")
print("    ffn_gate()")
print("    nvtx.range_pop()   ← pops 'ffn' correctly")
print("    # forward_pass range NEVER CLOSED")
print()
print("  Timeline consequence:")
print("    'forward_pass' extends from t=0 all the way to end of profile!")
print("    ncu --nvtx-include 'forward_pass' matches ALL kernels in the session.")
print("    Profiling data is polluted — wrong kernels attributed to forward_pass.")
print()
for err in bad_sim.errors:
    print(f"  ⚠  {err}")
print()

# Show the fix
print("  ── FIX: Python Context Manager ──────────────────────────────────")
print()
print("  import contextlib")
print("  import torch.cuda.nvtx as nvtx")
print()
print("  @contextlib.contextmanager")
print("  def nvtx_range(name, color=None):")
print("      nvtx.range_push(name)")
print("      try:")
print("          yield")
print("      finally:")
print("          nvtx.range_pop()   # called even on exception")
print()
print("  # Usage — pop ALWAYS called, even if attention() raises:")
print("  with nvtx_range('forward_pass'):")
print("      with nvtx_range('attention'):")
print("          attn_qkv()")
print("          attn_softmax()    # even if this raises, both pops fire")
print("      with nvtx_range('ffn'):")
print("          ffn_gate()")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: ncu --nvtx-include filter logic
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — ncu --nvtx-include Filter: Which Kernels Match?")
print("━" * 68)
print()

print("  Using the well-formed simulation from Section 1.")
print()

filters_to_test = [
    "iteration_42",
    "forward_pass",
    "layer_0/attention",
    "layer_1/ffn",
    "backward_pass",
    "optimizer_step",
]

for filt in filters_to_test:
    matched = sim.kernels_matching_nvtx_filter(filt)
    names   = list({m['name'] for m in matched})
    gpu_us  = sum(m['duration_us'] for m in matched)
    print("  Filter: --nvtx-include \"" + filt + "\"")
    print(f"    Matched kernels: {len(matched)}  |  GPU time: {gpu_us/1000:.1f}ms")
    print(f"    Kernel names: {names[:4]}{' ...' if len(names) > 4 else ''}")
    print()

print("  Key insight: nested filters like 'layer_0/attention' match ONLY kernels")
print("  launched while BOTH 'layer_0' AND 'attention' are on the NVTX stack.")
print("  This lets you profile a single sub-module of a single layer precisely.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · NCU Replay Cost Estimator — Section Sets vs Profiling Budget": {
        "description": (
            "Model the ncu hardware replay cost for different section sets on "
            "Ampere and Hopper architectures. Compute the number of kernel replays "
            "required per metric group and the total profiling overhead multiplier. "
            "Show how --launch-skip, --launch-count, and NVTX filtering reduce "
            "profiling time. Build a profiling budget planner that selects the "
            "minimal section set needed to answer each diagnostic question."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

print("=" * 68)
print("  NCU REPLAY COST ESTIMATOR — Section Sets & Profiling Budget")
print("=" * 68)
print()

import math


# ─────────────────────────────────────────────────────────────────────
# Section set definitions
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Section:
    name:            str
    description:     str
    n_passes_ampere: int    # how many kernel replays needed on Ampere (A100)
    n_passes_hopper: int    # how many kernel replays needed on Hopper (H100)
    answers:         List   # which diagnostic questions this section answers
    cli_flag:        str

SECTIONS = [
    Section("LaunchStatistics",
            "Grid/block config, occupancy calculation",
            1, 1,
            ["What is the occupancy?", "Is the launch config optimal?"],
            "--section LaunchStatistics"),
    Section("SpeedOfLight",
            "Roofline: SM throughput, compute/memory bottleneck",
            2, 2,
            ["Memory-bound or compute-bound?", "SM utilisation %"],
            "--section SpeedOfLight"),
    Section("MemoryWorkloadAnalysis",
            "L1/L2/DRAM traffic, bandwidth, coalescing, hit rates",
            4, 4,
            ["Which cache level is the bottleneck?", "Is access coalesced?",
             "What is the L2 hit rate?", "Total HBM traffic?"],
            "--section MemoryWorkloadAnalysis"),
    Section("SchedulerStatistics",
            "Warp issue rate, eligible warps, issued vs stalled cycles",
            2, 2,
            ["Is the warp scheduler busy?", "What fraction of cycles issue instructions?"],
            "--section SchedulerStatistics"),
    Section("WarpStateStatistics",
            "Per-stall-type warp cycle counts",
            4, 4,
            ["Why are warps stalling?", "Long scoreboard % vs short?",
             "Barrier stalls?", "Math throttle?"],
            "--section WarpStateStatistics"),
    Section("InstructionStatistics",
            "Instruction mix, FP32/INT/tensor core breakdown",
            3, 3,
            ["Are tensor cores being used?", "Instruction throughput?",
             "IPC breakdown?"],
            "--section InstructionStatistics"),
    Section("SourceCounters",
            "Per-SASS-instruction counters for source attribution",
            8, 10,
            ["Which source line is the bottleneck?", "Per-instruction stall counts"],
            "--section SourceCounters"),
]

PRESETS = {
    "--set default":  ["LaunchStatistics", "SpeedOfLight"],
    "--set detailed": ["LaunchStatistics", "SpeedOfLight", "MemoryWorkloadAnalysis",
                       "SchedulerStatistics", "WarpStateStatistics"],
    "--set full":     [s.name for s in SECTIONS],
}

section_map = {s.name: s for s in SECTIONS}


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Replay cost per section and preset
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Replay Cost: Passes per Section and Preset")
print("━" * 68)
print()

print("  Per-section replay passes (each pass = one full kernel replay):")
print()
print(f"  {'Section':<30}  {'Ampere':>8}  {'Hopper':>8}  {'Answers'}")
print("  " + "─" * 72)
for s in SECTIONS:
    ans_short = s.answers[0][:35] if s.answers else "-"
    print(f"  {s.name:<30}  {s.n_passes_ampere:>8}  {s.n_passes_hopper:>8}  "
          f"{ans_short}...")

print()
print("  Preset replay totals (sum of sections in each preset):")
print()
print(f"  {'Preset':<18}  {'Ampere passes':>14}  {'Hopper passes':>14}  {'Sections included'}")
print("  " + "─" * 66)

for preset_name, section_names in PRESETS.items():
    a_total = sum(section_map[n].n_passes_ampere for n in section_names)
    h_total = sum(section_map[n].n_passes_hopper for n in section_names)
    print(f"  {preset_name:<18}  {a_total:>14}  {h_total:>14}  "
          f"{', '.join(section_names[:3])}{'...' if len(section_names) > 3 else ''}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Real profiling time overhead for different kernels
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Profiling Time Overhead: Wall-Clock Cost per Kernel")
print("━" * 68)
print()

# Kernels with different natural durations
kernels = [
    ("short kernel (20 µs)",    20e-6,    "elementwise op"),
    ("medium kernel (1 ms)",    1e-3,     "attention layer"),
    ("long kernel (10 ms)",     10e-3,    "GEMM 8192³"),
    ("very long (100 ms)",      100e-3,   "full forward pass"),
]

print(f"  {'Kernel':<28}  {'Natural':>9}  {'default':>10}  "
      f"{'detailed':>11}  {'full':>9}  {'full overhead'}")
print("  " + "─" * 78)

for kname, dur_s, note in kernels:
    passes_default  = sum(section_map[n].n_passes_ampere for n in PRESETS["--set default"])
    passes_detailed = sum(section_map[n].n_passes_ampere for n in PRESETS["--set detailed"])
    passes_full     = sum(section_map[n].n_passes_ampere for n in PRESETS["--set full"])

    t_default  = dur_s * passes_default
    t_detailed = dur_s * passes_detailed
    t_full     = dur_s * passes_full
    overhead   = passes_full

    def fmt(t):
        if t >= 1: return f"{t:.1f}s"
        if t >= 0.001: return f"{t*1000:.0f}ms"
        return f"{t*1e6:.0f}µs"

    print(f"  {kname:<28}  {fmt(dur_s):>9}  {fmt(t_default):>10}  "
          f"{fmt(t_detailed):>11}  {fmt(t_full):>9}  {overhead}× slower")

print()
print("  RULE: --set default for iteration (~10× overhead). --set full sparingly.")
print("  For a 10ms kernel: full set = 240ms — acceptable for debugging.")
print("  For a 100ms kernel: full set = 2.4s — OK for one-time analysis only.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Profiling budget planner
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Profiling Budget Planner: Minimal Section per Question")
print("━" * 68)
print()

print("  Given a diagnostic question, what is the MINIMUM ncu command needed?")
print()

questions = [
    ("Is my kernel memory-bound or compute-bound?",
     "--set default",
     ["SpeedOfLight"], 2),
    ("What is my L2 hit rate and HBM traffic?",
     "--section MemoryWorkloadAnalysis",
     ["MemoryWorkloadAnalysis"], 4),
    ("Why are my warps stalling?",
     "--section WarpStateStatistics --section SchedulerStatistics",
     ["WarpStateStatistics", "SchedulerStatistics"], 6),
    ("Are tensor cores being used?",
     "--section InstructionStatistics",
     ["InstructionStatistics"], 3),
    ("Which source line is the bottleneck?",
     "--set full --source-folder /src",
     ["SourceCounters"] + list(PRESETS["--set full"]), 24),
    ("What is the occupancy and can I raise it?",
     "--section LaunchStatistics",
     ["LaunchStatistics"], 1),
    ("Are my SMEM accesses causing bank conflicts?",
     "--section MemoryWorkloadAnalysis",
     ["MemoryWorkloadAnalysis"], 4),
    ("Is the warp issue rate healthy (IPC)?",
     "--section SchedulerStatistics",
     ["SchedulerStatistics"], 2),
]

for q, cmd, secs, passes in questions:
    dur_10ms = 10e-3  # 10ms reference kernel
    profiling_time_s = dur_10ms * passes
    def fmt_t(t):
        return f"{t*1000:.0f}ms" if t < 1 else f"{t:.1f}s"

    print(f"  Q: {q}")
    print(f"     Command:  ncu {cmd} -o report <program>")
    print(f"     Passes:   {passes}  |  Profiling time (10ms kernel): {fmt_t(profiling_time_s)}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: --launch-skip + --launch-count optimisation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — launch-skip / launch-count: Targeting Steady State")
print("━" * 68)
print()

# Simulate kernel timing across invocations (cold → warm)
def sim_kernel_duration(invoc, base_us=3000):
    """Cold first invocation (L2 cold), warm thereafter."""
    if invoc == 0:
        return base_us * 3.2   # cold miss — 3.2× slower
    if invoc == 1:
        return base_us * 1.4   # partial warm
    return base_us * (1.0 + 0.02 * (invoc < 5))  # steady state

BASE_DUR_US   = 3000.0
FULL_PASSES   = sum(section_map[n].n_passes_ampere for n in PRESETS["--set full"])

print(f"  Reference kernel: {BASE_DUR_US/1000:.0f}ms natural duration")
print(f"  --set full: {FULL_PASSES} passes")
print()
print(f"  {'Invoc':>6}  {'Natural µs':>12}  {'Cold?':>7}  "
      f"{'Profile time':>14}  {'Recommendation'}")
print("  " + "─" * 62)

for inv in range(8):
    dur_us  = sim_kernel_duration(inv, BASE_DUR_US)
    cold    = dur_us > BASE_DUR_US * 1.1
    prof_ms = dur_us * FULL_PASSES / 1000
    if inv == 0:
        rec = "← skip (cache cold, 3× inflated)"
    elif inv == 1:
        rec = "← skip (still partially cold)"
    elif inv < 5:
        rec = "← skip if possible"
    else:
        rec = "← PROFILE HERE (steady state)"
    print(f"  {inv:>6}  {dur_us:>12.0f}  {'COLD' if cold else 'warm':>7}  "
          f"{prof_ms:>12.0f}ms  {rec}")

print()
print("  Optimal command:")
cmd = [
    "  ncu --kernel-name 'my_kernel'",
    "      --launch-skip 5              # skip first 5 invocations (cold + warmup)",
    "      --launch-count 3             # profile invocations 6, 7, 8 (steady state)",
    "      --set detailed",
    "      -o profile.ncu-rep",
    "      python train.py",
]
for line in cmd:
    print(line)
print()
print("  With NVTX (most precise):")
cmd2 = [
    "  ncu --nvtx --nvtx-include 'forward_pass/attention'",
    "      --launch-skip 10 --launch-count 3",
    "      --set detailed -o out python train.py",
]
for line in cmd2:
    print(line)
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Before/After Metric Comparison — Quantifying Optimisation Impact": {
        "description": (
            "Simulate ncu before/after metric reports for three classic GPU "
            "optimisations: (1) fixing uncoalesced access with array transposition, "
            "(2) adding SMEM tiling to a naïve GEMM, (3) removing SMEM bank "
            "conflicts with padding. For each, show the exact metric deltas, "
            "compute the speedup, and produce the --diff-mode relative report "
            "that ncu would generate when comparing two .ncu-rep files."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Tuple

print("=" * 68)
print("  BEFORE/AFTER METRIC COMPARISON — Quantifying Optimisation Impact")
print("=" * 68)
print()

H100_HBM_BW_GBS = 3350.0
H100_PEAK_FP32  = 67_000.0   # GFLOP/s

# ─────────────────────────────────────────────────────────────────────
# Metric comparison engine
# ─────────────────────────────────────────────────────────────────────

def pct_change(before, after):
    if abs(before) < 1e-12:
        return float('inf') if after > 0 else 0.0
    return (after - before) / abs(before) * 100.0

def speedup(t_before, t_after):
    return t_before / t_after if t_after > 0 else float('inf')

def fmt_metric(val, unit=""):
    if val >= 1e9:    return f"{val/1e9:.2f}G{unit}"
    if val >= 1e6:    return f"{val/1e6:.2f}M{unit}"
    if val >= 1e3:    return f"{val/1e3:.2f}K{unit}"
    return f"{val:.2f}{unit}"

def diff_report(name, before_metrics, duration_before_ns, duration_after_ns):
    sx = speedup(duration_before_ns, duration_after_ns)
    print(f"  ╔{'═'*64}╗")
    print(f"  ║  Optimisation: {name:<49}║")
    print(f"  ║  Duration:  BEFORE={duration_before_ns/1e6:.2f}ms  "
          f"AFTER={duration_after_ns/1e6:.2f}ms  "
          f"SPEEDUP={sx:.2f}×{' '*(15-len(f'{sx:.2f}'))}║")
    print(f"  ╠{'═'*64}╣")
    print(f"  ║  {'Metric':<38}  {'Before':>8}  {'After':>8}  {'Δ%':>6}  ║")
    print(f"  ╠{'─'*64}╣")
    for mname, (before_val, after_val, unit, direction) in before_metrics.items():
        delta_pct = pct_change(before_val, after_val)
        # direction: 'lower_better' or 'higher_better'
        if direction == 'lower_better':
            ok = '✅' if delta_pct < -10 else '❌' if delta_pct > 5 else '⚠'
        else:
            ok = '✅' if delta_pct > 10 else '❌' if delta_pct < -5 else '⚠'
        b_str = fmt_metric(before_val, unit)
        a_str = fmt_metric(after_val, unit)
        d_str = f"{delta_pct:+.1f}%"
        short = mname[:36]
        print(f"  ║  {short:<38}  {b_str:>8}  {a_str:>8}  {d_str:>6}  {ok} ║")
    print(f"  ╚{'═'*64}╝")
    print()


# ─────────────────────────────────────────────────────────────────────
# OPTIMISATION 1: Fix uncoalesced access (stride-32 → stride-1)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  OPTIMISATION 1 — Coalescing Fix: Column-Major → Row-Major Access")
print("━" * 68)
print()
print("  Kernel: elementwise transform of a 16384×4096 matrix")
print("  Before: threads read down columns (stride=16384) — 32 sectors per warp")
print("  After:  threads read across rows  (stride=1)    — 1 sector per warp")
print()

# Before: stride-32 access — 32 sectors per warp request
before_dur_ns = 18_400_000
after_dur_ns  =    680_000

diff_report(
    "Coalescing fix: stride-32 → stride-1",
    {
        "l1_sectors_load (×1e9)":   (3.36e9, 0.105e9,  "",    "lower_better"),
        "l1_requests_load (×1e6)":  (0.105e9, 0.104e9, "",    "lower_better"),
        "sectors_per_request":      (32.0,    1.01,    "",    "lower_better"),
        "dram_bytes_read (GB)":     (107.5e9, 3.35e9,  " B",  "lower_better"),
        "dram_bandwidth (GB/s)":    (5.8e9,  3295e6,   "/s",  "higher_better"),
        "HBM_efficiency_%":         (0.17,    98.4,    "%",   "higher_better"),
        "l2_hit_rate_%":            (2.1,     71.0,    "%",   "higher_better"),
        "stall_long_scoreboard_%":  (74.0,    7.2,     "%",   "lower_better"),
        "sm_throughput_%":          (14.8,    89.5,    "%",   "higher_better"),
    },
    before_dur_ns, after_dur_ns,
)

print(f"  KEY LESSON: 32 sectors/request means each 128-byte warp transaction")
print(f"  touches 32 × 32 = 1024 bytes but only uses 128 bytes (12.5% efficiency).")
print(f"  After fix: 1 sector/request = 100% efficiency. Speedup: {speedup(before_dur_ns, after_dur_ns):.1f}×.")
print()


# ─────────────────────────────────────────────────────────────────────
# OPTIMISATION 2: SMEM tiling for GEMM
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  OPTIMISATION 2 — SMEM Tiling: Naïve GEMM → Tiled GEMM")
print("━" * 68)
print()
print("  Kernel: SGEMM 2048×2048×2048, FP32")
print("  Before: every warp loads from global memory every FMA iteration")
print("  After:  128×128 tile loaded into SMEM, reused 128× before eviction")
print()

before_dur_2 = 85_000_000
after_dur_2  =  4_200_000

diff_report(
    "SMEM tiling (tile=128×128, register file reuse)",
    {
        "dram_bytes_read (GB)":     (65.6e9,  3.04e9,   " B",  "lower_better"),
        "l2_hit_rate_%":            (4.1,     82.0,     "%",   "higher_better"),
        "l1_sectors_load (×1e9)":   (2.1e9,   0.098e9,  "",    "lower_better"),
        "stall_long_scoreboard_%":  (78.0,    8.0,      "%",   "lower_better"),
        "stall_math_throttle_%":    (1.5,     38.0,     "%",   "higher_better"),
        "issue_active_rate":        (0.12,    0.71,     "",    "higher_better"),
        "sm_throughput_%":          (8.2,     72.0,     "%",   "higher_better"),
        "stall_not_selected_%":     (8.0,     31.0,     "%",   "higher_better"),
    },
    before_dur_2, after_dur_2,
)

arithmetic_intensity = (2 * 2048**3) / (3 * 2048**2 * 4)  # FLOPs / bytes (ideal)
print(f"  Arithmetic intensity (ideal):  {arithmetic_intensity:.0f} FLOPs/byte")
print(f"  Before: HBM dominates — AI ≈ 0.25 (all data re-read from HBM).")
print(f"  After:  SMEM reuse brings effective AI to ~{arithmetic_intensity:.0f}.")
print(f"  stall_math_throttle rising from 1.5% → 38%: the kernel is now compute-bound. ✅")
print(f"  Speedup: {speedup(before_dur_2, after_dur_2):.1f}×")
print()


# ─────────────────────────────────────────────────────────────────────
# OPTIMISATION 3: SMEM bank conflict removal via +1 padding
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  OPTIMISATION 3 — SMEM Bank Conflict Fix: +1 Column Padding")
print("━" * 68)
print()
print("  Kernel: LayerNorm forward, SMEM accumulation array")
print("  Before: 32-bank conflicts on every warp's SMEM load (32-way conflict)")
print("  After:  +1 padding breaks bank alignment — zero conflicts")
print()

before_dur_3 = 3_800_000
after_dur_3  = 2_100_000

diff_report(
    "SMEM bank conflict removal (+1 col padding)",
    {
        "smem_bank_conflicts_ld (×1e6)": (98e6,  0.0,   "",    "lower_better"),
        "smem_bank_conflicts_st (×1e6)": (48e6,  0.0,   "",    "lower_better"),
        "stall_short_scoreboard_%":      (24.0,  3.5,   "%",   "lower_better"),
        "stall_barrier_%":               (18.0,  8.0,   "%",   "lower_better"),
        "issue_active_rate":             (0.38,  0.74,  "",    "higher_better"),
        "sm_throughput_%":               (41.0,  79.0,  "%",   "higher_better"),
        "dram_bytes_read (GB)":          (6.08e9, 6.05e9," B", "lower_better"),
        "l2_hit_rate_%":                 (55.0,  57.0,  "%",   "higher_better"),
    },
    before_dur_3, after_dur_3,
)

print(f"  SMEM padding mechanics:")
print(f"    Before: float smem[32][64];    // 64 floats/row × 4 bytes = 256 B/row")
print(f"            Row i, bank = (col × 4 / 4) % 32 = col % 32")
print(f"            Warp loading smem[0][0], smem[1][0], ..., smem[31][0]")
print(f"            All land on bank 0 → 32-way conflict → 32 serial cycles")
print()
print(f"    After:  float smem[32][65];    // +1 padding per row")
print(f"            Row i, bank = (i × 65 × 4 / 4) % 32 = (i × 65) % 32")
print(f"            Each of 32 warps maps to a DIFFERENT bank → no conflict")
print(f"    Cost:   65 × 32 × 4 = 8320 bytes vs 64 × 32 × 4 = 8192 bytes (+128 B)")
print(f"    Saving: {speedup(before_dur_3, after_dur_3):.2f}× speedup for 128 bytes of extra SMEM.")
print()

print("━" * 68)
print("  SUMMARY: Three Optimisations and Their Metric Signatures")
print("━" * 68)
print()
opts = [
    ("Coalescing fix",     before_dur_ns, after_dur_ns,
     "sectors/req 32→1, long_stall 74→7%, HBM eff 0.2→98%"),
    ("SMEM tiling",        before_dur_2,  after_dur_2,
     "dram_read 66→3GB, math_throttle 1.5→38%, L2 hit 4→82%"),
    ("Bank conflict fix",  before_dur_3,  after_dur_3,
     "smem_conflicts 98M→0, short_stall 24→3.5%, IPC 0.38→0.74"),
]
print(f"  {'Optimisation':<22}  {'Before':>9}  {'After':>8}  {'Speedup':>8}  {'Key metric change'}")
print("  " + "─" * 74)
for name, tb, ta, sig in opts:
    print(f"  {name:<22}  {tb/1e6:>7.1f}ms  {ta/1e6:>6.2f}ms  "
          f"{speedup(tb, ta):>7.1f}×  {sig}")
print()
print("  All three discovered by reading the same six-number diagnostic sequence.")
print("  ncu --diff-mode relative before.ncu-rep after.ncu-rep shows these deltas.")
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