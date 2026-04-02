"""
Profiling & Benchmarking — Nsight Compute, Roofline & Stall Analysis
=====================================================================

Writing a GPU kernel that produces the correct answer is the starting
point. Writing one that runs near the hardware's theoretical limits is
the discipline. The gap between a first-cut kernel and an optimised one
is rarely obvious from reading the code — it is revealed by a profiler.

Nsight Compute (NCU) is NVIDIA's kernel-level profiler. Unlike timeline
tools (Nsight Systems), it counts individual hardware events: how many
bytes crossed the L2 cache boundary, how many warp cycles were spent
waiting for a memory instruction, exactly which SMEM addresses caused
bank conflicts. It does this by replaying each kernel multiple times
with different hardware performance counters active, then correlating
the results back to source lines.

Understanding its output requires three interlocking mental models:

    1. THE ROOFLINE MODEL — where is your kernel relative to the
       hardware's memory-bandwidth and compute ceilings?
    2. MEMORY THROUGHPUT — which level of the hierarchy is the
       bottleneck, and how efficiently is each level being used?
    3. OCCUPANCY & STALL ANALYSIS — why are warps not executing, and
       is that costing you performance or not?

These three models answer three successive questions:
    "Am I memory-bound or compute-bound?"  ← Roofline
    "Which memory level is the bottleneck?" ← Throughput
    "Are my warps hiding that latency?"     ← Occupancy / Stalls

A kernel that scores well on all three is a production-quality kernel.
Every optimisation loop starts with a profiler run and ends with one.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Profiling & Benchmarking — Nsight Compute, Roofline & Stall Analysis"
DISPLAY_NAME = "05 · Profiling & Benchmarking"
ICON         = "📊"
SUBTITLE     = "Roofline Model · Memory Throughput · Occupancy · Stall Analysis"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — NSIGHT COMPUTE: WHAT IT IS AND HOW IT WORKS

### What Nsight Compute Measures

    Nsight Compute (ncu) instruments CUDA kernels at the hardware level.
    It reads Performance Monitor Counters (PMCs) — registers inside the
    SM that count hardware events: cache hits, memory transactions, warp
    stall cycles, arithmetic throughput, shared memory bank conflicts.

    Unlike CPU profilers that sample, NCU uses HARDWARE COUNTERS that
    count EVERY event without sampling error. The trade-off: the number
    of simultaneous counters is limited (typically 8–16), so NCU must
    replay the kernel multiple times with different counter sets.

    Key implication: the profiled kernel must be DETERMINISTIC and
    IDEMPOTENT — replaying it should produce identical behaviour.
    Kernels with data-dependent control flow may show misleading results.

### Replay Architecture

    NCU kernel profiling passes (simplified):
        Pass 1: base metrics  (cycles, active warps, instructions issued)
        Pass 2: L1 metrics    (L1 hits, L1 misses, L1 sector requests)
        Pass 3: L2 metrics    (L2 hits, L2 misses, L2 read/write bytes)
        Pass 4: HBM metrics   (DRAM reads, DRAM writes, DRAM utilisation)
        Pass 5: stall metrics (stall_mem, stall_exec, stall_sync, ...)
        Pass 6: source metrics (per-instruction cycle counts)
        ...

    Total passes: 10–30 depending on metric set.
    Total execution time: 10–30× longer than a single uninstrumented run.
    This is WHY profiling must be done on representative workloads,
    not micro-benchmarks too short to amortise pass overhead.

### NCU Invocation

    # Profile everything — full metric set, write to file
    ncu --set full -o profile_output.ncu-rep python my_kernel.py

    # Quick summary — just the key metrics
    ncu --set default python my_kernel.py

    # Specific metrics only
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
                  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
                  l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
        python my_kernel.py

    # Source-level correlation (requires -lineinfo compilation flag)
    ncu --set full --source-level-analysis global_access python my_kernel.py

    # Filter to specific kernel name
    ncu --kernel-name my_kernel_name --set full python my_kernel.py

    # Benchmark mode: run N times, take average
    ncu --replay-mode kernel --set full python my_kernel.py

### The NCU GUI vs CLI

    CLI output: plain-text table of metrics per kernel launch.
    GUI (ncu-ui): loads .ncu-rep files; shows source-correlation,
    roofline charts, memory charts, warp state timelines.

    For iterative optimisation: CLI for quick iteration, GUI for deep dives.
    The "Details" panel in the GUI maps counter values back to source lines
    and PTX instructions — essential for pinpointing hotspots.

### Important Metric Namespacing

    NCU metrics follow a hierarchical naming convention:
        unit__metric_type.aggregation

    Examples:
        sm__cycles_elapsed.avg              — average SM cycles
        l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum  — L1 global load bytes
        lts__t_bytes.avg.pct_of_peak_sustained_elapsed — L2 bandwidth utilisation
        dram__bytes_read.sum                — total HBM bytes read
        sm__warps_active.avg.pct_of_peak_sustained_active — occupancy

    The suffix matters:
        .sum        total across all SMs and all replays
        .avg        average per SM per cycle
        .pct_of_peak_sustained_elapsed — percentage of theoretical peak


##### PART 2 — THE ROOFLINE MODEL: LOCATING YOUR KERNEL ON THE PERFORMANCE MAP

### The Model

    The Roofline Model (Williams, Waterman, Patterson, 2009) is a visual
    performance model that answers: "How close is my kernel to the hardware limit,
    and which hardware limit is it hitting?"

    Two hardware ceilings:
        COMPUTE ROOF:    peak FLOPs/second  (e.g., 312 TFLOP/s on H100)
        MEMORY ROOF:     peak Bytes/second  (e.g., 3.35 TB/s on H100 HBM3)

    One kernel property:
        ARITHMETIC INTENSITY (AI):  FLOPs performed / Bytes transferred
        Units: FLOPs/Byte

    The kernel's achievable performance is:
        Perf = min(peak_FLOPs, AI × peak_BW)

    At the RIDGE POINT:  AI_ridge = peak_FLOPs / peak_BW
        H100 SXM: 312e12 / 3.35e12 = 93 FLOPs/Byte (FP16 tensor core)
        A100 SXM: 312e12 / 2.0e12  = 156 FLOPs/Byte

    Below the ridge (AI < AI_ridge):  MEMORY-BOUND.
        Increasing compute speed does nothing. Must reduce memory traffic.
    Above the ridge (AI > AI_ridge):  COMPUTE-BOUND.
        Improving memory access does nothing. Must increase arithmetic throughput.

### The Roofline Chart

    Attainable performance (GFLOPs/s)
    │
    │                     Compute roof (flat, horizontal)
    │  ─────────────────────────────────────────────────────
    │              ╱│
    │            ╱  │
    │          ╱    │         ← Kernel B: memory-bound, at roof ✅
    │        ╱  ● B │
    │      ╱    │   │● A      ← Kernel A: compute-bound, not at roof ❌
    │    ╱      │   │         (leaves compute performance on the table)
    │  ╱        │   │
    │╱    ● C   │   │         ← Kernel C: memory-bound, FAR below roof ❌
    │───────────┼───────────── Arithmetic Intensity (FLOPs/Byte)
              Ridge

    Kernel B is ideal: memory-bound, and achieving the memory bandwidth roof.
    Kernel A is compute-bound but not achieving peak FLOPs — other bottleneck.
    Kernel C is memory-bound but not achieving peak bandwidth — coalescing issue.

### Computing AI for Common ML Operations

    FLOPs and bytes for key transformer operations:

    GEMM (M×K × K×N):
        FLOPs: 2 × M × K × N
        Bytes: (M×K + K×N + M×N) × dtype_bytes
        AI = 2MKN / ((MK + KN + MN) × B)
        For M=N=K=4096, FP16: AI = 2×4096³ / (3×4096²×2) ≈ 1365 FLOPs/Byte
        Well above ridge → compute-bound ✓

    SOFTMAX (row of N elements):
        FLOPs: ~5N  (max, subtract, exp, sum, divide — ~5 ops per element)
        Bytes: 2 × N × dtype_bytes  (read + write)
        AI = 5N / (2N × 2) = 1.25 FLOPs/Byte
        Far below ridge → memory-bound (bandwidth limited)

    LAYERNORM (row of N elements):
        FLOPs: ~7N  (mean, variance, subtract, divide, scale, bias)
        Bytes: 2N × dtype_bytes
        AI = 7N / (4N) = 1.75 FLOPs/Byte
        Memory-bound

    ELEMENTWISE ReLU / Add / GELU:
        FLOPs: ~1–5 per element
        Bytes: 2 × dtype_bytes per element  (read + write)
        AI ≈ 0.25–1.25 FLOPs/Byte
        Deeply memory-bound — fusing these ops is critical

    ATTENTION (seq_len=N, head_dim=d):
        Standard: O(N²d) FLOPs, O(N²) bytes (materialise QKᵀ)
        AI = O(N²d) / O(N² × 2) = O(d)  → grows with head_dim
        FlashAttention: O(N²d) FLOPs, O(Nd) bytes (tile to SMEM)
        AI = O(N²d) / O(Nd × 2) = O(N)  → grows with seq_len!
        FlashAttention pushes attention from memory-bound to compute-bound
        for long sequences — that's the core of its performance advantage.

### Rooflines Are Hierarchical

    A single roofline is a simplification. A HIERARCHICAL ROOFLINE has
    one ceiling per memory level:

        DRAM (HBM) roof:   3.35 TB/s   (slowest, farthest)
        L2 cache roof:     ~12 TB/s    (medium)
        L1/SMEM roof:      ~19 TB/s    (fast, on-chip)
        Register roof:     ~100 TB/s   (not usually the bottleneck)

    If your kernel's AI is between the DRAM and L2 ridge points,
    improving L2 cache locality can "lift" it to the L2 roof without
    changing the algorithm's AI. Tiling is the canonical technique.

    NCU's "Roofline" section in the GUI plots the kernel against
    all hierarchy levels simultaneously, so you can see exactly which
    memory level the kernel is "resting on."

### NCU Roofline Metrics

    arithmetic_intensity (NCU computes this automatically):
        = (sm__sass_thread_inst_executed_op_fadd_pred_on.sum * 2 +
           sm__sass_thread_inst_executed_op_fmul_pred_on.sum * 2 +
           sm__sass_thread_inst_executed_op_ffma_pred_on.sum * 4)
          / dram__bytes.sum

    Or more precisely, NCU uses:
        AI = flops_executed / dram_bytes_accessed
    where flops_executed comes from instruction-count PMCs and
    dram_bytes_accessed from the DRAM unit's byte counters.


##### PART 3 — MEMORY THROUGHPUT: THE FULL HIERARCHY IN DETAIL

### The H100 Memory Hierarchy (Numbers to Know)

    ┌─────────────────────────────────────────────────────────────┐
    │  Level      Capacity   Bandwidth    Latency     Scope       │
    ├─────────────────────────────────────────────────────────────┤
    │  Registers  256 KB/SM  ~100 TB/s    0 cycles    per-thread  │
    │  L1/SMEM    228 KB/SM  ~19  TB/s    ~23 cycles  per-block   │
    │  L2 cache   50  MB     ~12  TB/s    ~200 cycles GPU-wide    │
    │  HBM3       80  GB     3.35 TB/s    ~500 cycles GPU-wide    │
    └─────────────────────────────────────────────────────────────┘

    Latency numbers in cycles assume H100 SXM @ 1.83 GHz base clock.
    In nanoseconds: L1 ≈ 13 ns, L2 ≈ 109 ns, HBM ≈ 273 ns.

### Key NCU Memory Metrics

    GLOBAL MEMORY (HBM) THROUGHPUT:
        dram__bytes_read.sum         — total bytes read from DRAM
        dram__bytes_write.sum        — total bytes written to DRAM
        l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
            — average L1 sector requests per global load instruction
            — ideal = 1 (fully coalesced), high = uncoalesced

    L2 CACHE:
        lts__t_bytes.avg.pct_of_peak_sustained_elapsed — L2 bandwidth %
        lts__t_sectors_srcunit_tex_op_read.sum          — L2 read sectors
        lts__t_sector_hit_rate.avg.pct                  — L2 hit rate

    L1 / SHARED MEMORY:
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  — L1 global reads
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  — bank conflicts
        l1tex__t_sector_hit_rate.avg.pct                — L1 hit rate

    OVERALL:
        gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
            — combined arithmetic + memory throughput utilisation

### Interpreting Global Memory Efficiency

    The critical ratio: achieved_bytes / theoretical_minimum_bytes

        Theoretical minimum: each input element read once = N × dtype_bytes
        Achieved: what NCU reports in dram__bytes_read.sum

    Efficiency = theoretical_min / achieved_bytes

    If efficiency is 50%: you're reading 2× more data than necessary.
    Causes:
        1. UNCOALESCED ACCESS: threads in a warp read non-contiguous addresses.
           Each non-contiguous group needs separate cache-line fetches.
        2. LOW L1/L2 HIT RATE: same data read multiple times without caching.
        3. REDUNDANT LOADS: algorithm reads data more times than necessary.
           (e.g., naïve softmax reads x three times; fused version reads once)

    The metric l1tex__average_t_sectors_per_request... directly tells you
    coalescing efficiency:
        Value 1.0: every load instruction is fully coalesced (ideal)
        Value 4.0: each 128-byte load fetches 4 sectors but only 1 is needed
        Value 32.0: catastrophic — stride-32 access, each thread its own sector

### L2 Hit Rate and Cache Behaviour

    L2 hit rate = fraction of L1 misses served from L2 (not DRAM).
    High L2 hit rate means: the data fits in L2, or has good temporal reuse.

    Scenarios:
        Matrix multiply (tiled):       L2 hit rate ~80–95%  ← tiles fit in L2
        Softmax (streaming):           L2 hit rate ~10–30%  ← data too large
        Small model inference:         L2 hit rate ~60–80%  ← weights cached

    Improving L2 hit rate:
        - Tile algorithms so each tile fits in L2 (same as SMEM tiling)
        - Increase locality of access patterns
        - Use __ldca (cache-at-all-levels) hints instead of __ldcs (streaming)
        - Pad data to avoid L2 set conflicts (rare but real)

### Shared Memory Bandwidth and Bank Conflicts

    SMEM bank conflicts:
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum

    Interpretation:
        Value 0:     No conflicts. Peak SMEM bandwidth.
        Value N:     N extra serialised cycles across all SMEM accesses.
        Per-access:  conflicts / total_smem_accesses = conflict rate

    The metric counts EXTRA transactions (conflicts - 1 per multi-way conflict):
        2-way conflict on 1 bank = 1 extra transaction
        32-way conflict on 1 bank = 31 extra transactions

    When this metric is non-zero:
        1. Check for stride-32 access patterns (the classic culprit)
        2. Check SMEM tile dimensions — add +1 column padding if column-access
        3. Check if block size causes all threads to map to same bank subset

### Reading the NCU Memory Chart

    The Memory Workload Analysis chart in NCU GUI shows a hierarchy diagram
    with bandwidth numbers at each level:

        [L1 hit rate: 45%] → [L2 hit rate: 72%] → [DRAM: 1.8 TB/s]

    Read it from right to left:
        DRAM number: how much data crossed the HBM bus. Compare to 3.35 TB/s peak.
        L2 hit rate: what fraction of L1 misses stayed in L2.
        L1 hit rate: what fraction of requests were served from L1.

    A "waterfall" of high numbers means efficient caching (compute-bound kernel).
    A flat line at DRAM bandwidth means the kernel is HBM-bound.


##### PART 4 — OCCUPANCY: WARPS, RESOURCES, AND THE SCHEDULER CONTRACT

### What Occupancy Is

    OCCUPANCY = active warps per SM / maximum warps per SM

    H100: max 64 warps per SM. If your kernel has 32 active warps: 50% occupancy.
    A100: max 64 warps per SM.
    V100: max 64 warps per SM.

    NCU metric:
        sm__warps_active.avg.pct_of_peak_sustained_active

    THEORETICAL OCCUPANCY: what the kernel launch configuration allows,
    given register and SMEM constraints.
    ACHIEVED OCCUPANCY: what was measured during execution.
    These differ when blocks retire before others launch (end-of-grid effects).

### The Resource Constraints on Occupancy

    Three resources limit how many blocks (and thus warps) fit on an SM:

    1. REGISTERS:
        Each SM has 65,536 registers (H100/A100).
        If your kernel uses 64 regs/thread × 256 threads/block = 16,384 regs/block.
        Max blocks = 65536 / 16384 = 4 blocks.
        4 blocks × 256 threads / 32 = 32 warps → 50% occupancy.
        Compiler: use maxrregcount pragma to force fewer registers (may spill to L1).

    2. SHARED MEMORY:
        SM has 228 KB (H100). Allocated per block.
        If your kernel uses 64 KB/block:
        Max blocks = 228 KB / 64 KB = 3 blocks (rounded down).
        3 × 8 warps = 24 warps → 37.5% occupancy.

    3. THREAD COUNT LIMITS:
        Max 2048 threads/SM (H100/A100).
        Max 32 blocks/SM.
        Your block_size × (2048 / block_size) = max blocks from thread limit.

    The binding constraint is whichever resource runs out first.

### Occupancy Is Not Performance

    THIS IS THE MOST COMMON MISCONCEPTION IN GPU PROGRAMMING.

    High occupancy DOES NOT mean high performance.
    Low occupancy DOES NOT mean poor performance.

    Why high occupancy can hurt:
        - A kernel with large SMEM tiles (FlashAttention: 128 KB/block)
          will have very low occupancy (1–2 blocks/SM).
        - But those tiles eliminate HBM round-trips entirely.
        - The kernel is compute-bound and each warp is always busy.
        - Forcing higher occupancy by shrinking tiles brings the bottleneck
          back to HBM, which is far slower. Net result: slower kernel.

    Why low occupancy can be fine:
        - If each active warp has enough independent instructions to
          keep the execution units busy (high IPC), stalls are hidden.
        - Occupancy is only critical when latency hiding is the bottleneck.

    The RIGHT question is not "what is my occupancy?" but
    "are my warps ever waiting with nothing to do?"
    That question is answered by STALL ANALYSIS (Part 5).

### The Occupancy Cliff

    As register count per thread increases, occupancy drops in steps:
    (A100, block_size=256, 65536 total regs)

        Regs/thread: 16  → blocks/SM: 16 → warps: 512 → 100% occ
        Regs/thread: 24  → blocks/SM: 10 → warps: 320 →  50% occ
        Regs/thread: 32  → blocks/SM: 8  → warps: 256 →  50% occ
        Regs/thread: 48  → blocks/SM: 5  → warps: 160 →  31% occ
        Regs/thread: 64  → blocks/SM: 4  → warps: 128 →  25% occ
        Regs/thread: 128 → blocks/SM: 2  → warps:  64 →  12.5% occ

    The jumps are discrete because register allocation is quantised.
    Reducing from 65 regs/thread to 63 can suddenly jump occupancy
    from 12.5% to 25% — a 2× improvement from removing 2 registers.

    NCU metric: launch__occupancy_limit_registers
    → tells you if registers are the binding constraint.

### CUDA Occupancy Calculator

    #include <cuda_runtime.h>
    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, my_kernel, block_size, smem_bytes);
    float occupancy = (float)max_blocks * block_size / (float)max_threads_per_sm;

    Or use the Python API:
    from numba import cuda
    occ = cuda.occupancy.max_active_blocks_per_multiprocessor(kernel, block_size, smem)


##### PART 5 — STALL ANALYSIS: WHY WARPS ARE NOT EXECUTING

### The Warp Scheduler's Perspective

    Every clock cycle, the warp scheduler selects ONE eligible warp
    from all active warps and issues its next instruction.

    An ELIGIBLE warp is one where:
        - The next instruction has all input operands ready.
        - No resource conflict prevents issue (e.g., structural hazard).

    An INELIGIBLE (STALLED) warp is waiting for something.
    The percentage of cycles where no warp is eligible = issue slots wasted.

    NCU metric: smsp__issue_active.avg.pct_of_peak_sustained_active
        → fraction of scheduler cycles where an instruction was issued.
        Below 80%: significant stall cycles exist. Investigate stall reasons.

### The Stall Taxonomy

    NCU reports stall reason breakdowns as a percentage of total stall cycles:

    smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
        MIO THROTTLE: too many in-flight memory instructions.
        Memory instruction queue is full — new memory ops can't be issued.
        Fix: reduce number of independent memory operations per thread.

    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
        LONG SCOREBOARD: waiting for an L2/HBM memory instruction result.
        The warp issued a global load and the data hasn't arrived yet.
        Fix: increase occupancy (more warps to hide latency), or reduce HBM pressure.
        This is the PRIMARY stall reason for memory-bound kernels.

    smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
        SHORT SCOREBOARD: waiting for an L1/SMEM memory instruction result.
        The warp issued an SMEM access (or L1 hit) and is waiting.
        Fix: restructure SMEM access pattern, reduce SMEM bank conflicts.

    smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
        MATH PIPE THROTTLE: arithmetic units are fully saturated.
        New arithmetic instructions can't be issued — pipeline full.
        This is the PRIMARY stall for compute-bound kernels. Often desirable.
        Fix: use different instruction types to spread across pipes.

    smsp__warp_issue_stalled_barrier_per_warp_active.pct
        BARRIER: warp is waiting at __syncthreads() or __syncwarp().
        Fix: minimise barriers, overlap computation with synchronisation.
        Often indicates SMEM algorithm structure (tiled GEMM has necessary barriers).

    smsp__warp_issue_stalled_wait_per_warp_active.pct
        WAIT: warp is waiting on a fixed-latency instruction (not memory).
        Common: waiting for results of IMAD, FP conversion, special math (sin, sqrt).
        Fix: reorder instructions to interleave dependent chains.

    smsp__warp_issue_stalled_no_instructions_per_warp_active.pct
        NO INSTRUCTIONS: warp has no more instructions to execute.
        Warp has finished and is draining — nothing to do.
        Indicates block-level load imbalance or insufficient parallelism.

    smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct
        TEX THROTTLE: texture unit pipeline is full.
        Less common for compute kernels; more common for graphics-style access.

### Interpreting Stall Profiles

    Memory-bound kernel (typical softmax, elementwise):
        long_scoreboard: 60–80%    ← waiting for HBM loads
        short_scoreboard: 5–15%
        math_pipe: <5%
        Diagnosis: not enough warps to hide HBM latency, or working set too large.
        Action: increase occupancy, use streaming loads, vectorise (float4).

    Compute-bound kernel (typical large GEMM):
        math_pipe: 40–60%          ← arithmetic units full — GOOD
        long_scoreboard: 10–20%
        barrier: 10–20%            ← necessary sync at tile boundaries
        Diagnosis: near-optimal. Stalls are unavoidable (you're at the compute roof).
        Action: little to do — you're already at peak.

    SMEM-bound kernel (poorly tiled transpose):
        short_scoreboard: 40–60%   ← waiting for SMEM reads
        bank conflicts: high
        Diagnosis: SMEM bank conflicts serialise accesses.
        Action: pad SMEM array (+1 column), restructure access pattern.

    Divergence-heavy kernel:
        no_instructions: 30–50%    ← masked threads sit idle in diverged branches
        Diagnosis: warp divergence — threads in different branches.
        Action: reorganise data so same-branch threads are in the same warp.

### The IPC Metric

    Instructions Per Cycle (IPC):
        smsp__inst_executed.avg / smsp__cycles_active.avg

    Ideal IPC depends on the instruction mix:
        H100: max ~2 instructions issued per cycle per SM sub-partition (4 per SM).
        Practical maximum: ~1.5–2.0 for compute-bound kernels.
        For memory-bound kernels: IPC may be 0.1–0.5 (many idle cycles).

    IPC < 0.5 with high long_scoreboard stalls → memory latency is dominant.
    IPC > 1.0 with high math_pipe stalls → compute is dominant (ideal state).


##### PART 6 — THE OPTIMISATION WORKFLOW: PROFILING LOOP

### The Three Questions to Answer in Order

    STEP 1 — Where on the roofline is my kernel?
        Compute AI = FLOPs / bytes.
        Compare to ridge point.
        Memory-bound → go to Step 2. Compute-bound → go to Step 3.

    STEP 2 — Which memory level is the bottleneck?
        Check: dram__bytes vs theoretical minimum.
        Check: L1 sector ratio (coalescing efficiency).
        Check: L2 hit rate.
        Check: SMEM bank conflicts.
        Fix the worst offender, re-profile.

    STEP 3 — Are my warps actually executing?
        Check: issue_active percentage.
        Check: stall reason breakdown.
        If long_scoreboard dominant: increase occupancy.
        If math_pipe dominant: you're at the compute roof — done.
        If barrier dominant: reduce sync barriers or pipeline more.
        Fix, re-profile.

### A Concrete Profiling Session Example

    KERNEL: custom attention score kernel
    First profile (ncu --set default):

        Compute (SM) throughput:   18% of peak  ← very low
        Memory throughput:         72% of peak
        Achieved occupancy:        43%
        L1 sector ratio:           4.2          ← uncoalesced (ideal = 1.0)

    DIAGNOSIS: Memory-bound (18% compute, 72% memory). But memory throughput
    at 72% means coalescing issue (L1 sector ratio 4.2 → 4× wasted bandwidth).

    FIX: Restructure global memory access to coalesce thread accesses.
    After fix, re-profile:

        Compute (SM) throughput:   24% of peak
        Memory throughput:         89% of peak
        Achieved occupancy:        43%
        L1 sector ratio:           1.1          ← nearly perfect ✅

    NEXT STEP: Memory throughput at 89% → approaching bandwidth limit.
    Check stall profile: long_scoreboard at 65%. 43% occupancy may be too low
    to hide the remaining HBM latency.

    FIX: Increase occupancy by reducing register count (fewer temp variables).
    Or: switch from three-pass to fused two-pass algorithm (reduce HBM reads).

### Occupancy vs Latency Hiding: The Sweet Spot

    Latency hiding formula:
        warps_needed_to_hide_latency = latency_cycles / throughput_cycles_per_warp

    For HBM on H100:
        HBM latency:   ~500 cycles
        Issue rate:    1 instruction/cycle/warp_scheduler
        Warps needed:  ~500 warps to fully hide HBM latency (4 schedulers × 125 each)
        In practice:   12–16 active warps per SM is sufficient to hide ~80% of stalls.

    Rule of thumb: ≥ 25% occupancy (16 warps) is enough to hide latency
    for most memory-bound kernels. Below 12 warps, long_scoreboard stalls
    start to dominate noticeably.

    The inflection point varies by:
        - HBM bandwidth fraction used (higher BW → more outstanding requests)
        - Instruction level parallelism within a warp (multiple independent loads)
        - L2 hit rate (L2 latency ≈ 200 cycles needs fewer warps than HBM)

### Source-Level Correlation

    With --set full and -lineinfo compilation, NCU maps counter values to
    source lines and PTX instructions. The "Source" tab shows:

        Line 42:  float val = A[tid + stride];   ← 65% of long_scoreboard cycles
        Line 48:  val += B[offset * N];           ← 22% of long_scoreboard cycles

    This tells you WHICH LOAD is causing the most stall cycles.
    Fix that load first (coalescing, tiling, prefetching).

    The "PTX" and "SASS" views go deeper — showing the actual machine
    instructions and their per-instruction stall cycle counts.
    Essential for micro-optimisation after algorithmic fixes are exhausted.


##### PART 7 — BENCHMARKING CORRECTLY: CLOCKS, WARMUP & STATISTICAL RIGOUR

### The Common Mistakes in GPU Benchmarking

    1. NO WARMUP
        First kernel launch includes: CUDA context creation, JIT compilation,
        page fault handling, clock ramp-up.
        Rule: discard the first 1–3 runs. Always warm up.

    2. MEASURING HOST TIME INSTEAD OF GPU TIME
        time.time() on the host includes: Python overhead, CUDA API call overhead,
        and the async kernel launch (GPU is still running after the call returns).
        Rule: use CUDA events for GPU timing, always synchronise before measuring.

        // Correct GPU timing:
        cudaEvent_t start, end;
        cudaEventRecord(start);
        my_kernel<<<grid, block>>>(args);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);

    3. CPU-GPU SYNCHRONISATION SKEW
        cudaMemcpy is synchronous; kernel launches are async.
        If you wrap a kernel with a synchronous copy, you're timing both.
        Rule: isolate what you're measuring.

    4. CLOCK FREQUENCY VARIATION
        Modern GPUs (H100) use dynamic clock boosting.
        Frequency depends on power budget, temperature, and active SM count.
        Thermal throttling: runs hot → frequency drops → slower times.
        Rule: lock clocks during benchmarking (nvidia-smi --lock-gpu-clocks).

    5. INSUFFICIENT STATISTICAL SAMPLING
        Run the kernel 100+ times and report: median, P95, P99.
        Do NOT report mean (skewed by outliers).
        Report percentiles to characterise the distribution.

    6. WRONG N (TOO SMALL)
        Small N: kernel launch overhead (10–50 μs) dominates.
        Always benchmark at N large enough that kernel time >> launch overhead.
        Rule: kernel time should be ≥ 1 ms for reliable timing.

### CUDA Events: The Right Way to Time

    # PyTorch equivalent
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(3):
        my_kernel(args)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(100):
        start.record()
        my_kernel(args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))   # milliseconds

    import numpy as np
    print(f"Median: {np.median(times):.3f} ms")
    print(f"P95:    {np.percentile(times, 95):.3f} ms")
    print(f"P99:    {np.percentile(times, 99):.3f} ms")

### The Effective Bandwidth Metric

    For memory-bound kernels, the key benchmark output is EFFECTIVE BANDWIDTH,
    not raw execution time:

        effective_BW = bytes_accessed / time_s

    Then compare to theoretical peak (e.g., 3.35 TB/s for H100):
        efficiency = effective_BW / theoretical_peak

    This normalises for problem size and immediately shows how close to the
    hardware limit you are. Report this alongside latency — it's the number
    that tells you how much room for improvement remains.

### TFLOP/s for Compute-Bound Kernels

    For compute-bound kernels, report TFLOP/s:
        tflops = (2 × M × N × K) / time_s / 1e12   (for GEMM)

    Compare to theoretical peak FP16 TFLOP/s:
        cuBLAS achieves ~95% on A100 for large GEMM.
        Custom kernels typically achieve 50–85%.
        If you're below 50%: register pressure, instruction mix, or pipeline issue.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Roofline Model — AI Calculator, Kernel Placement & Bottleneck ID": {
        "description": (
            "Build a complete roofline model for A100 and H100 GPUs. Compute exact "
            "arithmetic intensity for every common ML operation (GEMM, softmax, "
            "LayerNorm, attention, elementwise). Plot each kernel's position relative "
            "to the memory-bandwidth and compute ceilings. Determine which is the "
            "binding constraint and compute the theoretical performance gap."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  ROOFLINE MODEL — Arithmetic Intensity, Ceilings & Kernel Placement")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# GPU hardware specs
# ─────────────────────────────────────────────────────────────────────

GPUS = {
    "H100 SXM": {
        "fp16_tflops":    312.0,   # tensor-core FP16
        "fp32_tflops":     67.0,   # FP32 non-tensor
        "hbm_tbs":          3.35,  # HBM3 bandwidth TB/s
        "l2_tbs":          12.0,   # L2 bandwidth TB/s
        "l1_tbs":          19.0,   # L1/SMEM bandwidth TB/s
        "l2_size_mb":      50,
        "sm_count":       132,
    },
    "A100 SXM": {
        "fp16_tflops":    312.0,
        "fp32_tflops":     19.5,
        "hbm_tbs":          2.0,
        "l2_tbs":          12.0,
        "l1_tbs":          19.0,
        "l2_size_mb":      40,
        "sm_count":        108,
    },
    "A10G": {
        "fp16_tflops":     125.0,
        "fp32_tflops":      31.2,
        "hbm_tbs":           0.6,
        "l2_tbs":            4.0,
        "l1_tbs":           10.0,
        "l2_size_mb":        6,
        "sm_count":         72,
    },
}


def ridge_point(gpu_spec, dtype="fp16"):
    tflops = gpu_spec["fp16_tflops"] if dtype == "fp16" else gpu_spec["fp32_tflops"]
    return tflops / gpu_spec["hbm_tbs"]   # FLOPs/Byte


def attainable_perf(ai, gpu_spec, dtype="fp16"):
    """Returns attainable TFLOP/s given arithmetic intensity and GPU spec."""
    peak_compute = gpu_spec["fp16_tflops"] if dtype == "fp16" else gpu_spec["fp32_tflops"]
    peak_bw_tflops = ai * gpu_spec["hbm_tbs"]   # TFlops achievable at this AI
    return min(peak_compute, peak_bw_tflops)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Ridge points and memory / compute ceilings
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — GPU Ceilings and Ridge Points")
print("━" * 68)
print()

print(f"  {'GPU':<14}  {'FP16 TFLOP/s':>13}  {'HBM TB/s':>10}  "
      f"{'Ridge (FP16)':>13}  {'FP32 TFLOP/s':>13}  {'Ridge (FP32)'}")
print("  " + "─" * 78)
for gpu_name, spec in GPUS.items():
    r16 = ridge_point(spec, "fp16")
    r32 = ridge_point(spec, "fp32")
    print(f"  {gpu_name:<14}  {spec['fp16_tflops']:>13.1f}  {spec['hbm_tbs']:>10.2f}  "
          f"{r16:>12.1f}  {spec['fp32_tflops']:>13.1f}  {r32:>12.1f}")

print()
print("  Ridge point = peak_TFLOP/s / peak_BW_TBs = FLOPs/Byte")
print("  Below ridge → MEMORY-BOUND | Above ridge → COMPUTE-BOUND")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Arithmetic intensity for common ML operations
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Arithmetic Intensity for ML Operations (FP16)")
print("━" * 68)
print()

FP16_BYTES = 2

def ai_gemm(M, N, K):
    flops = 2 * M * N * K
    bytes_io = (M*K + K*N + M*N) * FP16_BYTES
    return flops / bytes_io

def ai_softmax_3pass(N):
    flops = 5 * N   # max + sub + exp + sum + div
    bytes_io = 3 * N * FP16_BYTES + N * FP16_BYTES  # 3 reads + 1 write
    return flops / bytes_io

def ai_softmax_2pass(N):
    flops = 5 * N
    bytes_io = 2 * N * FP16_BYTES   # 1 read (with online stats) + 1 write
    return flops / bytes_io

def ai_layernorm(N):
    flops = 7 * N   # mean, var, sub, div, mul (gamma), add (beta), sqrt
    bytes_io = (N + N) * FP16_BYTES   # 1 read + 1 write
    return flops / bytes_io

def ai_rmsnorm(N):
    flops = 5 * N
    bytes_io = (N + N) * FP16_BYTES
    return flops / bytes_io

def ai_gelu(N):
    flops = 8 * N   # polynomial approximation: ~8 ops
    bytes_io = (N + N) * FP16_BYTES
    return flops / bytes_io

def ai_attention_standard(N, d):
    """Standard attention with materialised N×N score matrix."""
    flops_qkt = 2 * N * N * d   # Q × K^T
    flops_softmax = 5 * N * N
    flops_attn_v = 2 * N * N * d  # Attn × V
    total_flops = flops_qkt + flops_softmax + flops_attn_v
    # QK^T materialised: write N×N, then re-read it
    bytes_io = (2 * N * d + N * N * 4 + N * d) * FP16_BYTES
    return total_flops / bytes_io

def ai_flash_attention(N, d):
    """FlashAttention: tiles through SMEM, no N×N materialisation."""
    flops_qkt = 2 * N * N * d
    flops_softmax = 5 * N * N
    flops_attn_v = 2 * N * N * d
    total_flops = flops_qkt + flops_softmax + flops_attn_v
    bytes_io = (3 * N * d + N * d) * FP16_BYTES  # Q,K,V read once; O written once
    return total_flops / bytes_io

def ai_elementwise(N, flops_per_elem=1):
    bytes_io = (N + N) * FP16_BYTES
    return flops_per_elem * N / bytes_io

# Compute AIs
N_seq = 2048
d_head = 128
M_gemm = N_gemm = K_gemm = 4096

operations = [
    # (name, AI_func, args, color_tier)
    ("GEMM 4096³",             ai_gemm(M_gemm, N_gemm, K_gemm)),
    ("GEMM 512³",              ai_gemm(512, 512, 512)),
    ("GEMM 128³ (small)",      ai_gemm(128, 128, 128)),
    ("Flash Attention N=2048", ai_flash_attention(N_seq, d_head)),
    ("Std Attention N=2048",   ai_attention_standard(N_seq, d_head)),
    ("Softmax 2-pass N=2048",  ai_softmax_2pass(N_seq)),
    ("Softmax 3-pass N=2048",  ai_softmax_3pass(N_seq)),
    ("LayerNorm N=4096",       ai_layernorm(4096)),
    ("RMSNorm N=4096",         ai_rmsnorm(4096)),
    ("GELU N=4096",            ai_gelu(4096)),
    ("Add (elementwise)",      ai_elementwise(4096, 1)),
    ("ReLU (elementwise)",     ai_elementwise(4096, 1)),
]

gpu_spec = GPUS["A100 SXM"]
ridge = ridge_point(gpu_spec, "fp16")

print(f"  GPU: A100 SXM  |  Ridge point: {ridge:.0f} FLOPs/Byte  "
      f"(FP16 {gpu_spec['fp16_tflops']:.0f} TFLOP/s, HBM {gpu_spec['hbm_tbs']:.1f} TB/s)")
print()
print(f"  {'Operation':<33}  {'AI (F/B)':>10}  {'vs Ridge':>9}  "
      f"{'Bound':>10}  {'Attain. TFLOP/s':>16}  {'% of Peak'}")
print("  " + "─" * 88)

for name, ai in operations:
    bound = "COMPUTE" if ai >= ridge else "MEMORY"
    attain = attainable_perf(ai, gpu_spec, "fp16")
    pct = attain / gpu_spec["fp16_tflops"] * 100
    ratio_str = f"{ai/ridge:.3f}×" if ai < ridge else f"{ai/ridge:.1f}×"
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"  {name:<33}  {ai:>10.2f}  {ratio_str:>9}  {bound:>10}  "
          f"{attain:>14.1f}  {pct:>6.1f}%  [{bar}]")

print()
print("  Attainable TFLOP/s = min(peak_compute, AI × peak_BW)")
print("  Memory-bound kernels: attainable << peak. Compute-bound: near peak.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Roofline chart (ASCII art)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — ASCII Roofline Chart (A100, FP16)")
print("━" * 68)
print()

CHART_W = 60
CHART_H = 20
peak_compute = gpu_spec["fp16_tflops"]   # 312 TFLOP/s
peak_bw      = gpu_spec["hbm_tbs"]       # 2.0 TB/s

# AI range: 0.25 to 2000 FLOPs/Byte (log scale)
ai_min = 0.1
ai_max = 2000.0
log_ai_min = math.log10(ai_min)
log_ai_max = math.log10(ai_max)

def ai_to_x(ai):
    return int((math.log10(ai) - log_ai_min) / (log_ai_max - log_ai_min) * CHART_W)

def perf_to_y(perf):
    return int((1 - perf / peak_compute) * CHART_H)

# Build chart grid
grid = [[' '] * (CHART_W + 2) for _ in range(CHART_H + 2)]

# Draw roofline
for col in range(CHART_W + 1):
    ai = 10 ** (log_ai_min + col / CHART_W * (log_ai_max - log_ai_min))
    attain = attainable_perf(ai, gpu_spec, "fp16")
    row = perf_to_y(attain)
    if 0 <= row <= CHART_H:
        if ai < ridge:
            grid[row][col] = '/'
        else:
            grid[row][col] = '─'

# Plot kernel points
kernel_points = [
    ("GEMM 4K³",     ai_gemm(4096,4096,4096), "G"),
    ("Softmax 2p",   ai_softmax_2pass(2048),  "S"),
    ("FlashAttn",    ai_flash_attention(2048,128), "F"),
    ("LayerNorm",    ai_layernorm(4096),       "L"),
    ("ElemAdd",      ai_elementwise(4096, 1),  "E"),
]
for label, ai, char in kernel_points:
    attain = attainable_perf(ai, gpu_spec, "fp16")
    x = ai_to_x(ai)
    y = perf_to_y(attain)
    if 0 <= x <= CHART_W and 0 <= y <= CHART_H:
        grid[y][x] = char

# Print chart
print(f"  TFLOP/s")
print(f"  {peak_compute:>6.0f} │" + "".join(grid[0]) + "│")
for row in range(1, CHART_H):
    perf_val = peak_compute * (1 - row / CHART_H)
    prefix = f"  {perf_val:>6.0f} │" if row % 4 == 0 else "         │"
    print(prefix + "".join(grid[row]) + "│")
print(f"       0 └" + "─" * (CHART_W + 1) + "┘")
ai_labels = [0.1, 1, 10, 100, 1000]
label_row = "         "
for lbl in ai_labels:
    x = ai_to_x(lbl)
    label_row = label_row[:9 + x] + f"{lbl:.0f}" + label_row[9 + x + len(str(int(lbl))):]
print(f"  {label_row[:74]}  FLOPs/Byte")
print()
legend = "  Legend: G=GEMM  S=Softmax(2-pass)  F=FlashAttn  L=LayerNorm  E=ElemAdd"
print(legend)
print(f"  Ridge point: {ridge:.0f} FLOPs/Byte  (vertical transition on chart)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: How fusing operations changes AI
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Kernel Fusion: How Fusing Operations Changes AI")
print("━" * 68)
print()

N = 4096
print(f"  N = {N} elements, FP16")
print()

print("  UNFUSED (separate kernels):")
ops_unfused = [
    ("LayerNorm",           ai_layernorm(N),     "read x, write norm(x)"),
    ("Linear (weight proj)",ai_gemm(1, 4096, 4096), "GEMM: 1×4096 × 4096×4096"),
    ("GELU activation",     ai_gelu(N * 4),      "read linear_out, write gelu_out"),
    ("Linear (down proj)",  ai_gemm(1, 4096, 16384), "GEMM: 1×4096 × 16384×4096"),
    ("Residual add",        ai_elementwise(N, 1), "read two tensors, write one"),
]
total_bytes_unfused = 0
total_flops_unfused = 0
for name, ai, desc in ops_unfused:
    # approx bytes per op
    b = 2 * N * FP16_BYTES if "GEMM" not in name else N * N * FP16_BYTES * 0.01
    f = ai * b
    total_bytes_unfused += b
    total_flops_unfused += f
    attain = attainable_perf(ai, gpu_spec)
    bound = "COMPUTE" if ai >= ridge else "memory "
    print(f"    {name:<28}  AI={ai:>7.2f}  {bound}  {desc}")

print()
print("  FUSED (LayerNorm + GELU + Residual in one kernel):")
# Fused: read x once, compute all ops, write output once
fused_flops = (7 + 8 + 1) * N   # LN + GELU + add
fused_bytes = (N + N) * FP16_BYTES   # read x once, write output once
ai_fused = fused_flops / fused_bytes
attain_fused = attainable_perf(ai_fused, gpu_spec)
bound_fused = "COMPUTE" if ai_fused >= ridge else "memory "
print(f"    LN + GELU + Add (fused)      AI={ai_fused:>7.2f}  {bound_fused}  "
      f"read x once, write once")
print()
print(f"  Fusion benefit:")
print(f"    Unfused LN AI:  {ai_layernorm(N):.2f} (3 HBM passes)")
print(f"    Unfused GELU AI: {ai_gelu(N):.2f} (2 HBM passes)")
print(f"    Fused AI:       {ai_fused:.2f} (combined flops, 2 HBM passes)")
print(f"    Fused is {ai_fused/ai_layernorm(N):.2f}× higher AI than unfused LN")
print(f"    Fused is {ai_fused/ai_gelu(N):.2f}× higher AI than unfused GELU")
print()
print("  Fusion increases AI by amortising the fixed HBM read/write cost")
print("  over more FLOPs. The bytes stay ~constant; the FLOPs accumulate.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Memory Throughput — Coalescing Efficiency, Cache Hierarchy & Bandwidth": {
        "description": (
            "Model the full GPU memory hierarchy: compute effective bandwidth at "
            "each level (HBM, L2, L1/SMEM). Simulate how coalescing efficiency "
            "degrades throughput. Calculate the L1 sector ratio (sectors per "
            "request) for different access patterns. Show cache miss waterfalls "
            "and how L2 hit rate relates to working set size. Identify "
            "bandwidth bottlenecks the way NCU's Memory Workload Analysis panel does."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  MEMORY THROUGHPUT — Hierarchy, Coalescing & Bandwidth Analysis")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# Hardware constants (H100 SXM)
# ─────────────────────────────────────────────────────────────────────

H100 = {
    "hbm_bw_gbs":    3350,   # GB/s
    "l2_bw_gbs":    12000,
    "l1_bw_gbs":    19000,
    "l2_size_mb":      50,
    "l1_size_kb":     228,
    "warp_size":       32,
    "sector_bytes":    32,   # minimum memory transaction unit
    "cache_line":     128,   # 128-byte L1 cache line
    "sm_count":       132,
    "clock_ghz":     1.83,
    "hbm_latency_ns":  273,  # ~500 cycles
    "l2_latency_ns":   109,  # ~200 cycles
    "l1_latency_ns":    12,  # ~22 cycles
}

FP16_BYTES = 2
FP32_BYTES = 4


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Coalescing efficiency — NCU L1 sector ratio model
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Coalescing Efficiency & L1 Sector Ratio")
print("━" * 68)
print()
print("  NCU metric: l1tex__average_t_sectors_per_request")
print("  Ideal = 1.0 (fully coalesced), bad = 32.0 (one sector per thread)")
print()

SECTOR = H100["sector_bytes"]   # 32 bytes
WARP   = H100["warp_size"]      # 32 threads

def sectors_for_warp(addrs, elem_bytes=FP32_BYTES):
    """Count unique 32-byte sectors touched by a warp's memory access."""
    unique_sectors = set(a // SECTOR for a in addrs)
    return len(unique_sectors)

def ideal_sectors(elem_bytes=FP32_BYTES):
    """Minimum sectors for 32-thread warp accessing 32 contiguous elements."""
    return max(1, (WARP * elem_bytes) // SECTOR)

access_patterns = {
    "Stride-1 FP32 (ideal)":     [i * FP32_BYTES       for i in range(WARP)],
    "Stride-1 FP16 (ideal)":     [i * FP16_BYTES       for i in range(WARP)],
    "Stride-2 FP32":             [i * 2 * FP32_BYTES   for i in range(WARP)],
    "Stride-4 FP32":             [i * 4 * FP32_BYTES   for i in range(WARP)],
    "Stride-32 FP32 (worst)":    [i * 32 * FP32_BYTES  for i in range(WARP)],
    "Broadcast (all same addr)": [0                     for _ in range(WARP)],
    "Column of 1024-wide matrix":[i * 1024 * FP32_BYTES for i in range(WARP)],
    "Random (gather)":           sorted(np.random.default_rng(7).choice(
                                   range(0, 256*FP32_BYTES, FP32_BYTES),
                                   size=WARP, replace=False).astype(int).tolist()),
}

print(f"  {'Pattern':<38}  {'Sectors':>8}  {'Ideal':>6}  "
      f"{'Ratio':>7}  {'BW eff%':>8}  {'Eff. BW (GB/s)'}")
print("  " + "─" * 78)

for name, addrs in access_patterns.items():
    n_sectors = sectors_for_warp(addrs)
    ideal_n   = ideal_sectors()
    ratio     = n_sectors / ideal_n
    bw_eff    = 1.0 / ratio * 100
    eff_bw    = H100["hbm_bw_gbs"] * (1.0 / ratio)
    print(f"  {name:<38}  {n_sectors:>8}  {ideal_n:>6}  "
          f"{ratio:>6.1f}×  {bw_eff:>7.1f}%  {eff_bw:>12.0f}")

print()
print("  Sector ratio is the NCU metric that directly measures coalescing quality.")
print("  Ratio > 2 → investigate access pattern. Ratio > 8 → severe problem.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Cache hierarchy — working set vs hit rate model
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Working Set vs Cache Hit Rate Model")
print("━" * 68)
print()

print("  For a kernel that accesses a working set of W bytes:")
print("    If W < L1 size → L1 hit rate ~95%, L2 irrelevant")
print("    If W < L2 size → L1 miss, L2 hit rate ~80-90%")
print("    If W > L2 size → L2 miss, DRAM access (cold)")
print()

l1_kb = H100["l1_size_kb"]
l2_mb = H100["l2_size_mb"]

def estimate_cache_performance(working_set_bytes, access_pattern="sequential"):
    """
    Estimate effective memory bandwidth given working set size.
    Simplified model — real behaviour depends on access pattern and reuse.
    """
    ws_kb = working_set_bytes / 1024
    ws_mb = ws_kb / 1024

    if ws_kb <= l1_kb * 0.5:
        l1_hit = 0.95
        l2_hit = 0.80
        bottleneck = "L1"
        eff_bw = H100["l1_bw_gbs"] * l1_hit + H100["l2_bw_gbs"] * (1-l1_hit) * l2_hit
    elif ws_mb <= l2_mb * 0.5:
        l1_hit = 0.20
        l2_hit = 0.85
        bottleneck = "L2"
        eff_bw = H100["l2_bw_gbs"] * l2_hit + H100["hbm_bw_gbs"] * (1-l2_hit)
    elif ws_mb <= l2_mb * 2:
        l1_hit = 0.10
        l2_hit = 0.40
        bottleneck = "L2/HBM"
        eff_bw = H100["l2_bw_gbs"] * l2_hit + H100["hbm_bw_gbs"] * (1-l2_hit)
    else:
        l1_hit = 0.05
        l2_hit = 0.10
        bottleneck = "HBM"
        eff_bw = H100["hbm_bw_gbs"] * 0.85  # ~85% of peak for streaming

    return l1_hit, l2_hit, eff_bw, bottleneck


print(f"  {'Working set':>14}  {'L1 hit':>8}  {'L2 hit':>8}  "
      f"{'Eff. BW (GB/s)':>16}  {'Bottleneck':>12}  {'vs HBM peak'}")
print("  " + "─" * 74)

working_sets = [
    ("16 KB  (tile)",           16 * 1024),
    ("64 KB  (block)",          64 * 1024),
    ("1 MB   (small model)",    1 * 1024 * 1024),
    ("10 MB  (medium)",         10 * 1024 * 1024),
    ("50 MB  (= L2 size)",      50 * 1024 * 1024),
    ("200 MB (large batch)",    200 * 1024 * 1024),
    ("1 GB   (llm layer)",      1024 * 1024 * 1024),
    ("80 GB  (full model)",     80 * 1024 * 1024 * 1024),
]

for label, ws_bytes in working_sets:
    l1h, l2h, eff_bw, bottleneck = estimate_cache_performance(ws_bytes)
    ratio_to_hbm = eff_bw / H100["hbm_bw_gbs"]
    print(f"  {label:>14}  {l1h:>7.0%}  {l2h:>7.0%}  "
          f"{eff_bw:>14.0f}  {bottleneck:>12}  {ratio_to_hbm:>10.2f}×")

print()
print("  Tiling to fit in L1/L2 → massive effective bandwidth improvement.")
print("  This is why FlashAttention tiles K,V into SMEM — L1 bandwidth is 5× HBM.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Shared memory bandwidth vs bank conflicts
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — SMEM Bandwidth Degradation from Bank Conflicts")
print("━" * 68)
print()

NUM_BANKS = 32
BANK_WIDTH = 4   # bytes per bank (FP32 mode)

def smem_conflict_degree(thread_addrs):
    """Compute max bank conflict degree for a warp's SMEM access."""
    bank_accesses = {}
    for addr in thread_addrs:
        bank = (addr // BANK_WIDTH) % NUM_BANKS
        if bank not in bank_accesses:
            bank_accesses[bank] = set()
        bank_accesses[bank].add(addr)  # broadcast: same addr in same bank = OK

    max_conflict = 1
    for bank, addrs in bank_accesses.items():
        # Conflict degree = number of DISTINCT addresses in this bank
        unique = len(addrs)
        # But: if all accesses to this bank are to same addr → broadcast (no conflict)
        count = sum(1 for a in thread_addrs if (a // BANK_WIDTH) % NUM_BANKS == bank)
        if unique > 1:
            max_conflict = max(max_conflict, count)
    return max_conflict

smem_patterns = [
    ("Stride-1 (ideal)",            [i * BANK_WIDTH for i in range(WARP)]),
    ("Stride-2 (2-way conflict)",   [i * 2 * BANK_WIDTH for i in range(WARP)]),
    ("Stride-4 (4-way conflict)",   [i * 4 * BANK_WIDTH for i in range(WARP)]),
    ("Stride-16 (16-way conflict)", [i * 16 * BANK_WIDTH for i in range(WARP)]),
    ("Stride-32 (32-way conflict)", [i * 32 * BANK_WIDTH for i in range(WARP)]),
    ("Broadcast (same addr)",       [0 for _ in range(WARP)]),
    ("Col of 32×32 tile (32-way)",  [i * 32 * BANK_WIDTH for i in range(WARP)]),
    ("Col of 33×33 tile (no conf)", [(i * 33) * BANK_WIDTH for i in range(WARP)]),
]

smem_peak_gbs = H100["l1_bw_gbs"]

print(f"  SMEM peak bandwidth: {smem_peak_gbs:,} GB/s")
print()
print(f"  {'Access Pattern':<38}  {'Conflict':>10}  "
      f"{'Cycles':>7}  {'Eff. BW (GB/s)':>15}  {'BW util%'}")
print("  " + "─" * 80)

for name, addrs in smem_patterns:
    conflict = smem_conflict_degree(addrs)
    cycles   = conflict   # conflict × 1 cycle per serialised pass
    eff_bw   = smem_peak_gbs / conflict
    util     = 100.0 / conflict
    print(f"  {name:<38}  {conflict:>9}×  "
          f"{cycles:>7}  {eff_bw:>13,.0f}  {util:>7.1f}%")

print()
print("  NCU metric: l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
print("  Fix: add +1 column padding to SMEM tile (changes bank mapping for columns).")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Latency vs bandwidth — when each matters
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Latency vs Bandwidth: Two Different Problems")
print("━" * 68)
print()

print("  LATENCY problem: few outstanding requests, serial chain of dependent loads.")
print("  BANDWIDTH problem: many parallel requests, memory bus is saturated.")
print()
print("  Key distinction:")
print("    Serial loads  → limited by LATENCY (fix: more warps to hide it)")
print("    Parallel loads → limited by BANDWIDTH (fix: coalescing, tiling)")
print()
print("  How to tell which you have:")
print("    High long_scoreboard stalls + low achieved BW → LATENCY-bound")
print("    High long_scoreboard stalls + high achieved BW → BANDWIDTH-bound")
print()

# Model: warps needed to hide HBM latency
latency_cycles = int(H100["hbm_latency_ns"] * H100["clock_ghz"])   # ~500 cycles
issue_throughput = 1   # 1 instruction per cycle per warp scheduler

print(f"  Latency hiding analysis (H100):")
print(f"  HBM latency: {H100['hbm_latency_ns']} ns = {latency_cycles} cycles @ {H100['clock_ghz']} GHz")
print()
print(f"  {'Active warps':>13}  {'Warps can issue':>16}  {'Stall hidden?':>15}  "
      f"{'Effective BW %':>15}")
print("  " + "─" * 62)

for n_warps in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]:
    # Rough model: pipeline utilisation = min(1, n_warps / latency_cycles)
    util = min(1.0, n_warps * 10 / latency_cycles)  # 10 = avg independent loads/warp
    eff_bw_pct = util * 100
    stall_hidden = "Yes ✅" if n_warps >= 16 else ("Partial ⚠️" if n_warps >= 8 else "No ❌")
    print(f"  {n_warps:>13}  {n_warps:>16}  {stall_hidden:>15}  {eff_bw_pct:>13.0f}%")

print()
print("  Rule of thumb: ≥ 16 active warps per SM hides most HBM latency.")
print("  Below 8 warps: long_scoreboard stalls dominate — increase occupancy.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Occupancy Analysis — Resource Limits, Cliffs & Configuration Tuning": {
        "description": (
            "Build an exact CUDA occupancy model for H100 and A100. Show how "
            "registers, shared memory, and thread count each independently limit "
            "active warps per SM. Compute occupancy for every kernel in this "
            "curriculum (elementwise to FlashAttention). Simulate the register-count "
            "occupancy cliff. Show when low occupancy is acceptable vs harmful."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  OCCUPANCY ANALYSIS — Resource Limits, Cliffs & When It Matters")
print("=" * 68)
print()

GPU_SPECS = {
    "H100 SXM": {
        "max_threads_sm": 2048, "max_warps_sm": 64,
        "max_blocks_sm": 32,    "total_regs_sm": 65536,
        "max_smem_sm_kb": 228,  "smem_granularity_kb": 8,
        "reg_granularity": 256, "warp_size": 32,
    },
    "A100 SXM": {
        "max_threads_sm": 2048, "max_warps_sm": 64,
        "max_blocks_sm": 32,    "total_regs_sm": 65536,
        "max_smem_sm_kb": 164,  "smem_granularity_kb": 8,
        "reg_granularity": 256, "warp_size": 32,
    },
}


def compute_occupancy(gpu_name, threads_per_block, regs_per_thread, smem_bytes_per_block):
    """
    Compute theoretical occupancy with per-resource breakdown.
    Returns dict with blocks_per_sm, active_warps, occupancy, and limiting factor.
    """
    spec = GPU_SPECS[gpu_name]
    ws   = spec["warp_size"]
    warps_per_block = math.ceil(threads_per_block / ws)

    # Align SMEM to granularity
    smem_gran = spec["smem_granularity_kb"] * 1024
    smem_alloc = max(smem_gran, math.ceil(smem_bytes_per_block / smem_gran) * smem_gran)
    smem_max   = spec["max_smem_sm_kb"] * 1024

    # Align registers to granularity
    reg_gran   = spec["reg_granularity"]
    regs_per_block = math.ceil(warps_per_block * ws * regs_per_thread / reg_gran) * reg_gran

    limits = {
        "threads": spec["max_threads_sm"] // threads_per_block if threads_per_block > 0 else spec["max_blocks_sm"],
        "smem":    smem_max // smem_alloc if smem_alloc > 0 else spec["max_blocks_sm"],
        "regs":    spec["total_regs_sm"] // regs_per_block if regs_per_block > 0 else spec["max_blocks_sm"],
        "blocks":  spec["max_blocks_sm"],
    }
    blocks_per_sm  = min(limits.values())
    blocks_per_sm  = max(0, blocks_per_sm)
    active_warps   = blocks_per_sm * warps_per_block
    occupancy      = active_warps / spec["max_warps_sm"]

    # Determine binding constraint
    min_limit = min(limits.values())
    binding = [k for k, v in limits.items() if v == min_limit and v == blocks_per_sm]
    binding_str = binding[0] if binding else "none"

    return {
        "blocks_per_sm": blocks_per_sm, "warps": active_warps,
        "occupancy": occupancy, "binding": binding_str,
        "smem_per_block_kb": smem_alloc / 1024, "regs_per_block": regs_per_block,
        "limit_threads": limits["threads"], "limit_smem": limits["smem"],
        "limit_regs": limits["regs"],
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Occupancy for curriculum kernels (H100)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Occupancy for Every Curriculum Kernel (H100 SXM)")
print("━" * 68)
print()

# (name, threads_per_block, regs_per_thread, smem_bytes, notes)
curriculum_kernels = [
    ("Elementwise ReLU",       256,  16,      0,    "No SMEM"),
    ("Warp reduce sum",        256,  24,   4096,    "Warp partial sums"),
    ("Tree reduction",         256,  32,   4096,    "SMEM tree"),
    ("Prefix scan (Blelloch)", 256,  32,   4096,    "In-place SMEM"),
    ("Tiled GEMM (T=16)",      256,  32,   2048,    "2×16×16×4B"),
    ("Tiled GEMM (T=32)",     1024,  32,   8192,    "2×32×32×4B"),
    ("Tiled GEMM (T=64)",     1024,  64,  32768,    "2×64×64×4B"),
    ("LayerNorm (N=1024)",     256,  40,  16384,    "Row in SMEM"),
    ("cuDNN conv (typical)",   256,  48,  32768,    "Filter tiles"),
    ("Flash Attn V2 (d=128)",  128,  64,  65536,    "64 KB tile"),
    ("Flash Attn V3 (d=128)",  256,  96, 131072,    "128 KB tile"),
    ("CUTLASS GEMM (T=128)",   256,  96, 131072,    "Double-buf 128"),
]

print(f"  {'Kernel':<30}  {'Threads':>7}  {'Regs':>5}  {'SMEM':>7}  "
      f"{'Blk/SM':>7}  {'Warps':>6}  {'Occ%':>6}  {'Limit':>8}  Notes")
print("  " + "─" * 100)

for name, threads, regs, smem, notes in curriculum_kernels:
    r = compute_occupancy("H100 SXM", threads, regs, smem)
    smem_kb = smem / 1024
    occ_bar = "█" * int(r["occupancy"] * 10) + "░" * (10 - int(r["occupancy"] * 10))
    print(f"  {name:<30}  {threads:>7}  {regs:>5}  "
          f"{smem_kb:>5.0f}KB  {r['blocks_per_sm']:>7}  {r['warps']:>6}  "
          f"{r['occupancy']*100:>5.1f}%  {r['binding']:>8}  {notes}")

print()
print("  Flash Attn V3: only 1 block/SM (4 warps, 6.25% occ). But achieves")
print("  near-peak FLOP/s because large SMEM tiles eliminate HBM round-trips.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Register occupancy cliff
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Register Occupancy Cliff (256 threads, no SMEM, H100)")
print("━" * 68)
print()

print(f"  Total registers/SM: {GPU_SPECS['H100 SXM']['total_regs_sm']:,}")
print(f"  Block size: 256 threads = 8 warps")
print()
print(f"  {'Regs/thread':>12}  {'Regs/block':>12}  {'Blks/SM':>9}  "
      f"{'Warps/SM':>10}  {'Occ%':>7}  {'Cliff?'}")
print("  " + "─" * 64)

prev_occ = None
for regs_t in range(8, 130, 1):
    r = compute_occupancy("H100 SXM", 256, regs_t, 0)
    occ = r["occupancy"]
    cliff = ""
    if prev_occ is not None and occ < prev_occ - 0.01:
        cliff = " ← CLIFF ▼"
    # Only print interesting rows (cliffs + bookmarks)
    regs_block = r["regs_per_block"]
    if (cliff or regs_t in [8, 16, 24, 32, 40, 48, 64, 80, 96, 128]):
        print(f"  {regs_t:>12}  {regs_block:>12,}  {r['blocks_per_sm']:>9}  "
              f"{r['warps']:>10}  {occ*100:>6.1f}%{cliff}")
    prev_occ = occ

print()
print("  Cliffs occur when (total_regs / regs_per_block) crosses an integer.")
print("  Going from 33→32 regs/thread can sometimes double blocks_per_SM.")
print("  NCU reports: launch__occupancy_limit_registers when regs are binding.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: SMEM occupancy cliff
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — SMEM Occupancy Cliff (256 threads, 32 regs, H100)")
print("━" * 68)
print()

print(f"  Max SMEM/SM: {GPU_SPECS['H100 SXM']['max_smem_sm_kb']} KB")
print(f"  Allocation granularity: {GPU_SPECS['H100 SXM']['smem_granularity_kb']} KB")
print()
print(f"  {'SMEM/blk':>10}  {'Alloc':>8}  {'Blks/SM':>9}  "
      f"{'Warps/SM':>10}  {'Occ%':>7}  Bandwidth visualisation")
print("  " + "─" * 72)

prev_occ = None
for smem_kb in [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96,
                112, 128, 160, 196, 228]:
    r = compute_occupancy("H100 SXM", 256, 32, smem_kb * 1024)
    occ = r["occupancy"]
    cliff = " ← CLIFF" if prev_occ and (occ < prev_occ - 0.1) else ""
    bar_len = int(occ * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    print(f"  {smem_kb:>7} KB  {r['smem_per_block_kb']:>6.0f}KB  "
          f"{r['blocks_per_sm']:>9}  {r['warps']:>10}  "
          f"{occ*100:>6.1f}%  [{bar}]{cliff}")
    prev_occ = occ

print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Occupancy vs performance — the key insight
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Occupancy vs Performance: When It Matters")
print("━" * 68)
print()

print("  Occupancy affects performance ONLY when latency hiding is the bottleneck.")
print("  If warps have enough independent instructions → occupancy doesn't matter.")
print()

scenarios = [
    {
        "name":          "Elementwise kernel (bandwidth-bound)",
        "occupancy_pct": 50,
        "stall_type":    "long_scoreboard (HBM latency)",
        "stall_pct":     65,
        "verdict":       "BAD — more warps needed to hide HBM latency",
        "action":        "Reduce registers, increase block size",
    },
    {
        "name":          "Large GEMM (compute-bound)",
        "occupancy_pct": 25,
        "stall_type":    "math_pipe (FPU saturated)",
        "stall_pct":     50,
        "verdict":       "OK  — warps not stalled waiting; FPUs are busy",
        "action":        "Nothing to do; you're near compute roof",
    },
    {
        "name":          "FlashAttention (SMEM-heavy)",
        "occupancy_pct": 6,
        "stall_type":    "math_pipe + barrier",
        "stall_pct":     45,
        "verdict":       "OK  — SMEM eliminates HBM traffic; FPU busy",
        "action":        "Increasing occ would shrink SMEM → slower",
    },
    {
        "name":          "Naïve reduction (few warps)",
        "occupancy_pct": 12,
        "stall_type":    "long_scoreboard (HBM + idle)",
        "stall_pct":     78,
        "verdict":       "BAD — too few warps; HBM latency exposed",
        "action":        "Fuse kernel or increase warps via grid-stride",
    },
    {
        "name":          "Triton softmax (well-tuned)",
        "occupancy_pct": 37,
        "stall_type":    "long_scoreboard (some)",
        "stall_pct":     30,
        "verdict":       "GOOD — at memory bandwidth roof",
        "action":        "Profile memory efficiency; tune tile size",
    },
]

for s in scenarios:
    bar = "█" * (s["occupancy_pct"] // 5) + "░" * (20 - s["occupancy_pct"] // 5)
    print(f"  {s['name']}")
    print(f"    Occupancy: {s['occupancy_pct']:>3}%  [{bar}]")
    print(f"    Top stall: {s['stall_type']}  ({s['stall_pct']}% of cycles)")
    print(f"    Verdict:   {s['verdict']}")
    print(f"    Action:    {s['action']}")
    print()

print("  KEY RULE: Check stall reasons BEFORE trying to increase occupancy.")
print("  If math_pipe is the top stall, you're at the compute roof — done.")
print("  If long_scoreboard dominates, try increasing occupancy or tiling.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Stall Analysis — Taxonomy, Root Cause & Fix Patterns": {
        "description": (
            "Build a complete stall analysis simulator: model the warp scheduler "
            "issuing instructions and stalling on dependencies. Show cycle-by-cycle "
            "warp scheduling with a mix of memory and arithmetic instructions. "
            "Classify stalls into the full NCU taxonomy (long_scoreboard, "
            "short_scoreboard, math_pipe, barrier, wait, no_instructions). "
            "Show the diagnosis-to-fix mapping for each stall type."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from collections import defaultdict, deque

print("=" * 68)
print("  STALL ANALYSIS — Warp Scheduler Simulation & Stall Taxonomy")
print("=" * 68)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# Instruction latencies (cycles, H100)
# ─────────────────────────────────────────────────────────────────────

LATENCIES = {
    "fadd":   4,    # FP32 add
    "fmul":   4,    # FP32 multiply
    "ffma":   4,    # FP32 fused multiply-add
    "fp16":   4,    # FP16 arithmetic
    "smem":   23,   # shared memory load (no bank conflict)
    "smem_conflict": 23 * 4,  # 4-way bank conflict
    "l1":     33,   # L1 cache hit
    "l2":    200,   # L2 cache hit
    "hbm":   500,   # HBM miss
    "sync":   20,   # __syncthreads
    "sqrt":   16,   # special function
    "recip":  16,   # reciprocal
}

STALL_TYPES = {
    "long_scoreboard":  "Waiting for L2/HBM global memory op",
    "short_scoreboard": "Waiting for L1/SMEM memory op",
    "math_pipe":        "Arithmetic pipeline full",
    "barrier":          "Waiting at __syncthreads()",
    "wait":             "Fixed-latency instruction (sqrt, recip)",
    "no_instructions":  "Warp has no more instructions",
    "not_selected":     "Warp eligible but another warp was chosen",
}


# ─────────────────────────────────────────────────────────────────────
# Simple warp scheduler simulation
# ─────────────────────────────────────────────────────────────────────

class WarpSim:
    """
    Simplified warp scheduler simulation.
    Models instruction latency, scoreboard stalls, and issue slots.
    """
    def __init__(self, n_warps, program):
        """
        n_warps: active warps per scheduler
        program: list of (instruction_type, count) — sequential instruction stream
        """
        self.n_warps  = n_warps
        self.program  = []
        for instr, count in program:
            self.program.extend([instr] * count)

        self.warp_pc        = [0] * n_warps        # program counter per warp
        self.warp_ready_at  = [0] * n_warps        # cycle when warp is ready
        self.total_cycles   = 0
        self.stall_counts   = defaultdict(int)
        self.issue_count    = 0
        self.prog_len       = len(self.program)

    def classify_stall(self, instr):
        if instr in ("hbm", "l2"):
            return "long_scoreboard"
        elif instr in ("smem", "l1", "smem_conflict"):
            return "short_scoreboard"
        elif instr in ("sqrt", "recip"):
            return "wait"
        elif instr == "sync":
            return "barrier"
        return "math_pipe"

    def run(self, max_cycles=5000):
        cycle = 0
        while cycle < max_cycles:
            # Find all warps that are ready (past their stall)
            eligible = [w for w in range(self.n_warps)
                        if self.warp_ready_at[w] <= cycle
                        and self.warp_pc[w] < self.prog_len]

            if not eligible:
                # Check if any warps are still in flight
                active = [w for w in range(self.n_warps)
                          if self.warp_pc[w] < self.prog_len]
                if not active:
                    break
                # All warps stalled — advance clock to nearest ready
                next_ready = min(self.warp_ready_at[w] for w in active)
                stall_cycles = next_ready - cycle
                # Tally stall cycles for the waiting instruction
                for w in active:
                    if self.warp_pc[w] < self.prog_len:
                        instr = self.program[self.warp_pc[w]]
                        self.stall_counts[self.classify_stall(instr)] += stall_cycles
                cycle = next_ready
                continue

            # Issue one instruction from a round-robin eligible warp
            # (real scheduler is more complex but RR approximates well)
            w_chosen = eligible[cycle % len(eligible)]
            instr     = self.program[self.warp_pc[w_chosen]]
            lat       = LATENCIES.get(instr, 4)

            self.warp_ready_at[w_chosen] = cycle + lat
            self.warp_pc[w_chosen]      += 1
            self.issue_count            += 1

            # Count not_selected stalls for other eligible warps
            self.stall_counts["not_selected"] += len(eligible) - 1

            cycle += 1

        self.total_cycles = cycle
        return self

    def report(self):
        total_stall = sum(v for k, v in self.stall_counts.items()
                          if k != "not_selected")
        issue_eff = self.issue_count / max(1, self.total_cycles) * 100
        return {
            "total_cycles": self.total_cycles,
            "issue_count":  self.issue_count,
            "issue_efficiency_pct": issue_eff,
            "stall_counts": dict(self.stall_counts),
            "top_stall": max(
                (k for k in self.stall_counts if k != "not_selected"),
                key=lambda k: self.stall_counts[k],
                default="none"
            ),
        }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Stall taxonomy and diagnostic key
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Full Stall Taxonomy: NCU Metric → Root Cause → Fix")
print("━" * 68)
print()

stall_guide = [
    (
        "long_scoreboard",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "Waiting for L2/HBM global load/store to complete.",
        ">30%",
        [
            "Increase occupancy (more warps to issue while others wait)",
            "Tile algorithm into SMEM/L2 (reduce HBM round-trips)",
            "Vectorise loads: use float4 instead of float (fewer instructions, same bytes)",
            "Reduce memory pressure: fuse kernels, eliminate redundant reads",
        ],
    ),
    (
        "short_scoreboard",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "Waiting for L1/SMEM access to complete (bank conflict or L1 miss).",
        ">20%",
        [
            "Fix SMEM bank conflicts (+1 column padding for column-major access)",
            "Reduce SMEM access frequency (accumulate in registers first)",
            "Use __ldca vs __ldcs to control L1 caching policy",
        ],
    ),
    (
        "math_pipe",
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
        "Arithmetic pipeline saturated — all FPUs/TCs busy. GOOD if high.",
        "Desired (compute-bound)",
        [
            "If here: you're at the compute roof — no action needed",
            "If unexpectedly high: check for instruction-level serialisation",
            "Interleave different instruction types (FP32 + INT + memory)",
        ],
    ),
    (
        "barrier",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "Waiting at __syncthreads() or __syncwarp() for other warps.",
        ">15%",
        [
            "Reduce synchronisation frequency (pipeline tiles, double-buffer SMEM)",
            "Overlap independent computation with sync (restructure code before barrier)",
            "Replace __syncthreads with __syncwarp where only warp-scope needed",
        ],
    ),
    (
        "wait",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "Waiting for fixed-latency instruction (sqrt, recip, type convert).",
        ">10%",
        [
            "Reorder instructions to interleave dependent chains",
            "Use fast approximations: __expf, __logf, rsqrtf (1 cycle vs 16)",
            "Precompute constants outside inner loop",
        ],
    ),
    (
        "no_instructions",
        "smsp__warp_issue_stalled_no_instructions_per_warp_active.pct",
        "Warp has finished — no more instructions to issue.",
        ">5%",
        [
            "Improve grid balance: ensure all blocks have similar workload",
            "Use tail-loop padding to keep all warps active to end of block",
            "Increase grid size to cover more SMs",
        ],
    ),
]

for stall_name, metric, desc, threshold, fixes in stall_guide:
    print(f"  ┌── {stall_name.upper()} ──────────────────────────────────────")
    print(f"  │  NCU metric: {metric}")
    print(f"  │  Meaning:    {desc}")
    print(f"  │  Concerning: {threshold}")
    print(f"  │  Fixes:")
    for fix in fixes:
        print(f"  │    · {fix}")
    print(f"  └{'─' * 58}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Scheduler simulation for typical kernel types
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Scheduler Simulation for Typical Kernel Profiles")
print("━" * 68)
print()

kernel_programs = {
    "Memory-bound elementwise (low occ, 4 warps)": {
        "n_warps": 4,
        "program": [("hbm", 2), ("ffma", 2), ("hbm", 2)],
        "description": "2 HBM loads, 2 FMAs, 1 HBM store. 4 warps (25% occupancy).",
    },
    "Memory-bound elementwise (high occ, 32 warps)": {
        "n_warps": 32,
        "program": [("hbm", 2), ("ffma", 2), ("hbm", 2)],
        "description": "Same kernel but 32 warps (50% occupancy) — latency hidden.",
    },
    "Compute-bound GEMM tile (tensor core)": {
        "n_warps": 16,
        "program": [("smem", 4), ("sync", 1), ("ffma", 32), ("smem", 4), ("sync", 1)],
        "description": "Load SMEM tile, sync, 32 FMAs, repeat. Compute-bound.",
    },
    "SMEM bank-conflict (transpose kernel)": {
        "n_warps": 16,
        "program": [("hbm", 1), ("smem_conflict", 4), ("sync", 1), ("hbm", 1)],
        "description": "Load HBM, 4-way SMEM conflict, sync, store. Short scoreboard.",
    },
    "Mixed memory+compute (layernorm)": {
        "n_warps": 8,
        "program": [("hbm", 1), ("ffma", 4), ("smem", 2), ("ffma", 8),
                    ("sync", 1), ("ffma", 4), ("hbm", 1)],
        "description": "Load, compute mean+var in SMEM, normalise, store.",
    },
}

for kernel_name, config in kernel_programs.items():
    sim = WarpSim(config["n_warps"], config["program"] * 20)  # repeat for stats
    sim.run()
    r = sim.report()

    total_stalls = sum(v for k, v in r["stall_counts"].items() if k != "not_selected")
    print(f"  Kernel: {kernel_name}")
    print(f"  {config['description']}")
    print()
    print(f"    Total cycles:      {r['total_cycles']:,}")
    print(f"    Instructions issued: {r['issue_count']:,}")
    print(f"    Issue efficiency:  {r['issue_efficiency_pct']:.1f}%")
    print(f"    Top stall type:    {r['top_stall']}")
    print()

    if total_stalls > 0:
        print(f"    Stall breakdown:")
        for stall_name, count in sorted(r["stall_counts"].items(),
                                         key=lambda x: -x[1]):
            if stall_name == "not_selected" or count == 0:
                continue
            pct = count / total_stalls * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"      {stall_name:<22} [{bar}] {pct:5.1f}%")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Issue efficiency sensitivity to warp count
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Issue Efficiency vs Warp Count (Memory-Bound Kernel)")
print("━" * 68)
print()

print("  Kernel: load from HBM → 2 FMAs → store. HBM latency = 500 cycles.")
print()
print(f"  {'Warps':>6}  {'Occ%':>6}  {'Issue eff%':>11}  {'Top stall':>20}  Performance")
print("  " + "─" * 66)

for n_warps in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]:
    program = [("hbm", 2), ("ffma", 2), ("hbm", 1)]
    sim = WarpSim(n_warps, program * 50)
    sim.run()
    r = sim.report()
    occ_pct = n_warps / 64 * 100
    top_stall = r["top_stall"] if r["stall_counts"] else "none"

    if r["issue_efficiency_pct"] < 30:
        perf = "❌ Poor  — latency exposed"
    elif r["issue_efficiency_pct"] < 60:
        perf = "⚠️ OK    — partial hiding"
    else:
        perf = "✅ Good  — latency hidden"

    print(f"  {n_warps:>6}  {occ_pct:>5.1f}%  {r['issue_efficiency_pct']:>10.1f}%  "
          f"{top_stall:>20}  {perf}")

print()
print("  Issue efficiency plateaus around 32 warps — diminishing returns above.")
print("  Below 8 warps: HBM latency dominates. Increasing occupancy helps most here.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Benchmarking Harness — Timing, Statistics & Effective Bandwidth": {
        "description": (
            "Build a complete GPU kernel benchmarking harness: correct CUDA-event "
            "timing, warmup protocol, statistical analysis (median/P95/P99), "
            "effective bandwidth calculation, and TFLOP/s measurement. Simulate "
            "common benchmarking pitfalls. Show how to interpret benchmark results "
            "and map them to roofline positions. Build a summary report table."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time
from collections import namedtuple

print("=" * 68)
print("  BENCHMARKING HARNESS — Timing, Stats & Effective Bandwidth")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# Benchmark result structure
# ─────────────────────────────────────────────────────────────────────

BenchResult = namedtuple("BenchResult", [
    "name", "n_iters", "median_ms", "p95_ms", "p99_ms",
    "mean_ms", "std_ms", "eff_bw_gbs", "tflops", "cv_pct"
])


class GPUKernelBenchmark:
    """
    Simulate correct GPU kernel benchmarking methodology.
    In a real system this wraps actual CUDA event timing.
    Here we model realistic timing distributions.
    """
    def __init__(self, name, true_ms, jitter_pct=0.5, outlier_rate=0.02):
        self.name         = name
        self.true_ms      = true_ms      # median execution time
        self.jitter_pct   = jitter_pct   # ± jitter as % of true_ms
        self.outlier_rate = outlier_rate  # fraction of outlier runs

    def _sample_run(self):
        """Sample one realistic kernel execution time."""
        if np.random.random() < self.outlier_rate:
            # Outlier: OS interrupt, thermal event, etc.
            return self.true_ms * np.random.uniform(1.5, 4.0)
        jitter = np.random.normal(0, self.true_ms * self.jitter_pct / 100)
        return max(self.true_ms * 0.95, self.true_ms + jitter)

    def run(self, n_warmup=5, n_iters=100):
        # Warmup (discarded)
        for _ in range(n_warmup):
            self._sample_run()

        # Benchmark
        times = [self._sample_run() for _ in range(n_iters)]
        return np.array(times)


def compute_stats(name, times_ms, bytes_accessed=None, flops=None):
    """Compute comprehensive benchmark statistics."""
    median = np.median(times_ms)
    p95    = np.percentile(times_ms, 95)
    p99    = np.percentile(times_ms, 99)
    mean   = np.mean(times_ms)
    std    = np.std(times_ms)
    cv     = std / mean * 100   # coefficient of variation

    eff_bw = (bytes_accessed / (median / 1000) / 1e9) if bytes_accessed else 0
    tflops = (flops / (median / 1000) / 1e12) if flops else 0

    return BenchResult(name, len(times_ms), median, p95, p99, mean, std,
                       eff_bw, tflops, cv)


def print_bench_result(r, peak_bw_gbs=3350, peak_tflops_fp16=312):
    print(f"  {r.name}")
    print(f"    Runs: {r.n_iters}  |  "
          f"Median: {r.median_ms:.3f} ms  |  "
          f"P95: {r.p95_ms:.3f} ms  |  "
          f"P99: {r.p99_ms:.3f} ms")
    print(f"    Mean: {r.mean_ms:.3f} ms  |  "
          f"StdDev: {r.std_ms:.4f} ms  |  "
          f"CV: {r.cv_pct:.2f}%  (coefficient of variation)")
    if r.eff_bw_gbs > 0:
        pct = r.eff_bw_gbs / peak_bw_gbs * 100
        print(f"    Eff. BW: {r.eff_bw_gbs:.1f} GB/s  ({pct:.1f}% of {peak_bw_gbs} GB/s peak)")
    if r.tflops > 0:
        pct = r.tflops / peak_tflops_fp16 * 100
        print(f"    TFLOP/s: {r.tflops:.2f}  ({pct:.1f}% of {peak_tflops_fp16} TFLOP/s FP16 peak)")


np.random.seed(7)
PEAK_BW   = 3350   # H100 HBM3 GB/s
PEAK_FP16 = 312    # H100 FP16 TFLOP/s
FP16_BYTES = 2


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Common benchmarking pitfalls demonstrated
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Common Benchmarking Pitfalls")
print("━" * 68)
print()

# Simulate a kernel with 2 ms median but with first-run overhead
true_ms = 2.0
first_run_ms = 80.0   # CUDA context + JIT compilation overhead

print("  Pitfall 1: No warmup — first run includes JIT and context overhead")
print()
runs_no_warmup = [first_run_ms] + [true_ms + np.random.normal(0, 0.01) for _ in range(9)]
runs_warmed    = [true_ms + np.random.normal(0, 0.01) for _ in range(10)]

print(f"  {'Run':>4}  {'No warmup (ms)':>16}  {'With warmup (ms)':>18}")
print("  " + "─" * 42)
for i in range(10):
    flag = " ← COLD RUN (wrong!)" if i == 0 else ""
    print(f"  {i+1:>4}  {runs_no_warmup[i]:>16.3f}  {runs_warmed[i]:>18.3f}{flag}")
print()
print(f"  No-warmup mean:  {np.mean(runs_no_warmup):.3f} ms  ← {np.mean(runs_no_warmup)/np.mean(runs_warmed):.1f}× inflated")
print(f"  Warmed mean:     {np.mean(runs_warmed):.3f} ms  ← correct")
print()

print("  Pitfall 2: Too few iterations — high variance, unrepresentative result")
print()
kernel_sim = GPUKernelBenchmark("test_kernel", 1.5, jitter_pct=2.0, outlier_rate=0.05)

for n_iters in [3, 10, 50, 100, 500]:
    np.random.seed(42)
    times = kernel_sim.run(n_warmup=3, n_iters=n_iters)
    cv = np.std(times) / np.mean(times) * 100
    print(f"    n_iters={n_iters:>4}: median={np.median(times):.4f} ms  "
          f"P99={np.percentile(times,99):.4f} ms  CV={cv:.2f}%  "
          f"{'✅ stable' if cv < 1 else '⚠️ noisy' if cv < 3 else '❌ unreliable'}")

print()
print("  Pitfall 3: Reporting mean instead of median (outliers skew mean)")
print()
times_with_outliers = [1.5 + np.random.normal(0, 0.01) for _ in range(95)]
times_with_outliers += [8.0, 12.0, 7.5, 9.0, 6.0]   # 5 outliers
times_arr = np.array(times_with_outliers)
print(f"    95 normal runs ≈ 1.5 ms, 5 outlier runs ≈ 6–12 ms")
print(f"    Mean:   {np.mean(times_arr):.4f} ms  ← inflated by outliers")
print(f"    Median: {np.median(times_arr):.4f} ms  ← robust, correct")
print(f"    P95:    {np.percentile(times_arr,95):.4f} ms")
print(f"    P99:    {np.percentile(times_arr,99):.4f} ms")
print(f"    → Always report median (not mean) for GPU kernel latency.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Effective bandwidth measurement
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Effective Bandwidth for Memory-Bound Kernels")
print("━" * 68)
print()

memory_kernels = [
    # (name, N_elements, dtype_bytes, n_passes, actual_bw_pct_of_peak)
    ("Naïve elementwise (no coalescing)", 1_000_000, FP16_BYTES, 2, 12),
    ("Stride-2 elementwise",              1_000_000, FP16_BYTES, 2, 25),
    ("Coalesced elementwise",             1_000_000, FP16_BYTES, 2, 88),
    ("Vectorised (float4)",               1_000_000, FP16_BYTES, 2, 94),
    ("Softmax 3-pass",                    1_000_000, FP16_BYTES, 4, 65),
    ("Softmax 2-pass (online)",           1_000_000, FP16_BYTES, 2, 78),
    ("Fused softmax (Triton)",            1_000_000, FP16_BYTES, 2, 91),
    ("cuDNN softmax",                     1_000_000, FP16_BYTES, 2, 95),
]

print(f"  {'Kernel':<40}  {'Bytes':>9}  {'Median (ms)':>12}  "
      f"{'Eff BW (GB/s)':>14}  {'% of Peak':>11}")
print("  " + "─" * 94)

for name, N, dtype_b, passes, bw_pct in memory_kernels:
    bytes_accessed = N * dtype_b * passes
    # Back-calculate median time from target bandwidth
    true_bw = PEAK_BW * bw_pct / 100
    median_ms = bytes_accessed / (true_bw * 1e9) * 1000

    bench = GPUKernelBenchmark(name, median_ms, jitter_pct=1.0)
    times = bench.run(n_warmup=5, n_iters=100)
    r = compute_stats(name, times, bytes_accessed=bytes_accessed)

    gap_to_peak = PEAK_BW - r.eff_bw_gbs
    print(f"  {name:<40}  {bytes_accessed/1e6:>7.1f}MB  {r.median_ms:>11.4f}  "
          f"{r.eff_bw_gbs:>12.1f}  {r.eff_bw_gbs/PEAK_BW*100:>10.1f}%")

print()
print("  Effective BW = bytes_accessed / median_time_s")
print("  Report this % of peak — it immediately shows room for improvement.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: TFLOP/s measurement for compute-bound kernels
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — TFLOP/s Measurement for Compute-Bound Kernels")
print("━" * 68)
print()

gemm_configs = [
    # (M, N, K, fp16_utilisation_pct)
    (64,    64,   64,  15),  # too small — launch overhead dominates
    (256,  256,  256,  42),  # small — register pressure, pipeline underutilised
    (1024, 1024, 1024, 78),  # medium — approaching cuBLAS territory
    (2048, 2048, 2048, 88),  # large — CUTLASS-quality
    (4096, 4096, 4096, 93),  # very large — cuBLAS / near-peak
    (8192, 8192, 8192, 95),  # huge — cuBLAS peak
]

print(f"  {'M×N×K':<22}  {'FLOPs':>12}  {'Median (ms)':>12}  "
      f"{'TFLOP/s':>9}  {'% FP16 peak':>12}  {'Roofline pos'}")
print("  " + "─" * 82)

ridge = 312.0 / 3.35   # ≈ 93 FLOPs/Byte

for M, N, K, tc_pct in gemm_configs:
    flops = 2 * M * N * K
    bytes_io = (M*K + K*N + M*N) * FP16_BYTES
    ai = flops / bytes_io
    true_tflops = PEAK_FP16 * tc_pct / 100
    median_ms = flops / (true_tflops * 1e12) * 1000

    bench = GPUKernelBenchmark(f"GEMM {M}³", median_ms, jitter_pct=0.5)
    times = bench.run(n_warmup=5, n_iters=100)
    r = compute_stats(f"GEMM", times, bytes_accessed=bytes_io, flops=flops)

    bound = "compute" if ai >= ridge else "memory "
    shape = f"{M}×{N}×{K}"
    print(f"  {shape:<22}  {flops/1e12:>10.3f}T  {r.median_ms:>11.4f}  "
          f"{r.tflops:>8.1f}  {r.tflops/PEAK_FP16*100:>11.1f}%  {bound}")

print()
print("  Small GEMM underperforms — launch overhead is significant fraction.")
print("  GEMM >= 2048³: approaching cuBLAS performance territory.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: The complete benchmark report
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Production Benchmark Report: All Kernels")
print("━" * 68)
print()

print("  Benchmarking methodology:")
print("    Platform: H100 SXM (simulated)")
print("    Warmup: 5 iterations (discarded)")
print("    Benchmark: 200 iterations")
print("    Metric: median ± P99")
print("    Clocks: locked (nvidia-smi --lock-gpu-clocks 1410,2619)")
print()

all_kernels = [
    # (name, bytes, flops, median_ms, bw_pct, is_mem_bound)
    ("Elementwise ReLU N=10M",    10e6*2*2,  10e6*1,   0.005,  88, True),
    ("Softmax N=8192 (online)",   8192*2*2,  8192*5,   0.0003, 80, True),
    ("LayerNorm N=4096",          4096*2*2,  4096*7,   0.0002, 75, True),
    ("GEMM 4096³ (FP16)",         4096**2*3*2, 2*4096**3, 2.1, 0,  False),
    ("FlashAttn N=2048 d=128",    4*2048*128*2, 2*2048**2*128*2, 0.8, 0, False),
    ("cuDNN Conv (typical)",      50e6*2,    200e9,    12.5,   0,   False),
]

print(f"  {'Kernel':<35}  {'Median':>10}  {'P99':>10}  "
      f"{'Eff. BW':>10}  {'% Peak BW':>10}  {'TFLOP/s':>8}  {'% Peak TF'}")
print("  " + "─" * 98)

for name, bytes_io, flops, true_median, bw_pct_target, is_mem in all_kernels:
    bench = GPUKernelBenchmark(name, true_median, jitter_pct=1.0, outlier_rate=0.02)
    times = bench.run(n_warmup=5, n_iters=200)
    r = compute_stats(name, times, bytes_accessed=bytes_io, flops=flops)

    bw_str = f"{r.eff_bw_gbs:>8.1f}" if is_mem else "     N/A"
    bw_pct_str = f"{r.eff_bw_gbs/PEAK_BW*100:>9.1f}%" if is_mem else "       N/A"
    tf_str = f"{r.tflops:>8.2f}" if not is_mem else "     N/A"
    tf_pct_str = f"{r.tflops/PEAK_FP16*100:>8.1f}%" if not is_mem else "      N/A"

    print(f"  {name:<35}  {r.median_ms:>9.4f}  {r.p99_ms:>9.4f}  "
          f"{bw_str}  {bw_pct_str}  {tf_str}  {tf_pct_str}")

print()
print("  Report format: name | median_ms | P99_ms | eff_bw_GB/s | % of peak")
print("  For memory-bound: report eff_bw. For compute-bound: report TFLOP/s.")
print()
print("  GOLDEN RULES:")
print("    1. Always warm up (≥5 runs discarded).")
print("    2. Always use CUDA events, not time.time().")
print("    3. Report median, not mean. Also report P95/P99.")
print("    4. Run ≥100 iterations for stable statistics.")
print("    5. Lock clocks for reproducibility (nvidia-smi --lock-gpu-clocks).")
print("    6. Report % of theoretical peak, not just absolute numbers.")
print("    7. Profile with NCU after benchmarking to understand the gap.")
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