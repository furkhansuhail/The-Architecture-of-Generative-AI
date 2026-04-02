"""
Multi-GPU Memory — NVLink / NVSwitch Topology, P2P Transfers & GPUDirect RDMA
===============================================================================

Modern large-model training and serving requires GPUs to share data directly
with each other at memory-system speeds. Three hardware mechanisms make this
possible:

    NVLink / NVSwitch: NVIDIA's proprietary GPU interconnect, providing
    GPU-to-GPU bandwidth of 450–900 GB/s bidirectional within a node —
    10-20× faster than PCIe. NVSwitch chips create a full non-blocking
    crossbar between all GPUs in a DGX system.

    Peer-to-Peer (P2P) transfers: the ability for one GPU to read or write
    another GPU's HBM directly, without routing data through the CPU or
    system DRAM. Enabled by NVLink or PCIe peer access (cudaDeviceEnablePeerAccess).
    Eliminates the CPU bounce buffer that would otherwise halve bandwidth.

    GPUDirect RDMA: allows a network adapter (InfiniBand HCA) to DMA data
    directly from GPU HBM to the remote node — bypassing the CPU, system RAM,
    and PCIe host bridge. Reduces inter-node communication latency by 30-50%
    and eliminates one complete copy of the data for each send/receive.

Together these three mechanisms define the memory system architecture of a
GPU cluster. Understanding them means understanding: the physical connection
topology (which GPUs share NVLink links?), the software access model
(cudaMemcpyPeer vs UVA vs IPC), the kernel-level primitives (atomics,
fabric access), and the performance limits set by link bandwidth, latency,
and concurrency.

"""

import textwrap, re, math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Multi-GPU Memory — NVLink/NVSwitch Topology, P2P Transfers & GPUDirect RDMA"
DISPLAY_NAME = "17a · Multi-GPU Memory"
ICON         = "🔗"
SUBTITLE     = "NVLink · NVSwitch · P2P Transfers · UVA · IPC Handles · GPUDirect RDMA"

THEORY = """

##### PART 1 — THE MULTI-GPU MEMORY HIERARCHY

### Memory Access Paths Between GPUs

    When GPU A needs data from GPU B's HBM, there are three possible paths:

    PATH 1 — CPU bounce (no peer access):
        GPU B HBM → PCIe → CPU system DRAM → PCIe → GPU A HBM
        Bandwidth: limited by PCIe × 2 (round trip through CPU)
        Effective: ~15-16 GB/s (PCIe Gen4 x16 = 32 GB/s, halved for bounce)
        Latency: ~10-20 µs (multiple PCIe round-trips + CPU DMA operations)

    PATH 2 — PCIe peer access (no NVLink):
        GPU B HBM → PCIe → GPU A HBM (direct DMA, bypassing CPU DRAM)
        Bandwidth: ~32 GB/s (PCIe Gen4 x16 full duplex)
        Latency: ~5-10 µs
        Requires: cudaDeviceEnablePeerAccess(peerDevice, 0)
        Available when: GPUs share the same PCIe root complex or bridge.

    PATH 3 — NVLink peer access:
        GPU B HBM → NVLink → GPU A HBM (direct, on-chip interconnect)
        Bandwidth: 50-100 GB/s per link, 12-18 links per GPU (A100/H100)
        H100 aggregate unidirectional: 450 GB/s per GPU
        Latency: < 1 µs
        Available when: GPUs are physically connected by NVLink cables.

    HIERARCHY IN TERMS OF BANDWIDTH:
        NVLink (450 GB/s) >> PCIe P2P (32 GB/s) >> CPU bounce (16 GB/s)

### The Cost of Missing Levels

    For tensor parallel training passing activations between GPU 0 and GPU 1:
        With NVLink:        1 GB tensor / 450 GB/s = 2.2 ms
        With PCIe P2P:      1 GB tensor / 32 GB/s  = 31 ms
        With CPU bounce:    1 GB tensor / 16 GB/s  = 63 ms
        Ratio NVLink/bounce: 28× faster

    At 20 tensor-parallel communication calls per transformer layer:
        Per-layer overhead (NVLink):   20 × 2.2 ms = 44 ms
        Per-layer overhead (CPU):      20 × 63 ms  = 1260 ms
        Tensor parallelism is ONLY practical with NVLink interconnect.


##### PART 2 — NVLink: PHYSICAL ARCHITECTURE AND GENERATIONS

### NVLink Physical Layer

    NVLink is a point-to-point serial interconnect between two GPUs (or a
    GPU and an NVSwitch). Each NVLink LINK consists of:
        - Two differential signal pairs (TX and RX) per lane.
        - 8 differential pairs per direction = 16 signal lanes per link.
        - Each NVLink 4.0 lane: 25 Gbps per lane.
        - Per-link unidirectional bandwidth: 25 × 8 = 200 Gbps = 25 GB/s.
        - Per-link bidirectional: 50 GB/s.

    NVLINK GENERATIONS:
        ┌──────────┬─────────────┬──────────┬───────────────┬──────────────┐
        │ Version  │ Per-link BW │ Links/GPU│ Total/GPU     │ GPU          │
        │          │ (bidir GB/s)│          │ (bidir GB/s)  │              │
        ├──────────┼─────────────┼──────────┼───────────────┼──────────────┤
        │ NVLink 1 │     40      │    4     │    160        │ P100         │
        │ NVLink 2 │     50      │    6     │    300        │ V100         │
        │ NVLink 3 │     50      │   12     │    600        │ A100         │
        │ NVLink 4 │     50*     │   18     │    900        │ H100         │
        └──────────┴─────────────┴──────────┴───────────────┴──────────────┘
        *NVLink 4 doubles signal rate but uses same physical pin count via PAM4.

    H100 NVLink 4 details:
        18 NVLink ports, each 50 GB/s bidirectional.
        Total: 900 GB/s bidirectional per GPU.
        Unidirectional: 450 GB/s send + 450 GB/s receive.

### NVLink Topology in DGX Systems

    Not all 8 GPUs in a DGX are connected to each other directly.
    The topology depends on how many NVLink ports each GPU has:

    DGX A100 (8 × A100 with NVLink 3, 12 links/GPU):
        Full 8-GPU non-blocking crossbar is impossible with 12 links.
        NVIDIA uses a hybrid topology:
        - Each GPU directly connected to 5-7 other GPUs (not all 8).
        - NVSwitch chips bridge the remaining connections.
        - In DGX A100: 6 NVSwitch chips form an any-to-any crossbar.
        - Effective any-to-any bandwidth: 600 GB/s bidirectional.

    DGX H100 (8 × H100 with NVLink 4, 18 links/GPU):
        - 4 NVSwitch chips (NVSwitch 3.0).
        - Full 8×8 non-blocking crossbar at 900 GB/s per GPU.
        - Any GPU to any GPU: same 900 GB/s regardless of distance.

### NVLink Fabric: How the Switch Routes Traffic

    An NVSwitch acts as a routing fabric between GPUs:
        - Each GPU connects to each NVSwitch via multiple NVLink ports.
        - A100 DGX: each GPU has 2 links to each of 6 NVSwitches (12 total).
        - When GPU 0 sends to GPU 5: packets route through NVSwitches.
        - The switch selects the least-congested path automatically.
        - Adaptive routing: NVSwitch 3.0 (H100) dynamically balances load.

    BISECTION BANDWIDTH:
        Bisection = total bandwidth across a cut that divides GPUs in half.
        For DGX H100 (8 GPUs, 4 NVSwitches):
            Each NVSwitch connects to all 8 GPUs at 900/4 = 225 GB/s per switch.
            4 switches × 225 GB/s = 900 GB/s bisection per GPU.
            The fabric is "full bisection" — no congestion under any traffic pattern.


##### PART 3 — PEER-TO-PEER (P2P) TRANSFERS: SOFTWARE MODEL

### Enabling P2P Access

    P2P access must be explicitly enabled in the CUDA runtime:

        cudaSetDevice(0);
        int can_access;
        cudaDeviceCanAccessPeer(&can_access, 0, 1);  // GPU 0 → GPU 1?
        if (can_access) {
            cudaDeviceEnablePeerAccess(1, 0);  // enable GPU 0 accessing GPU 1
        }
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);  // symmetric: GPU 1 accessing GPU 0

    After enabling, pointers to GPU 1's memory can be used in kernels
    running on GPU 0 (and vice versa). The hardware routes the access
    through NVLink or PCIe automatically.

    REQUIREMENTS FOR P2P:
        PCIe P2P: GPUs must be on the same PCIe root complex or bridged.
                  Platform dependent — some BIOSes disable PCIe peer access.
        NVLink P2P: GPUs must be physically connected by NVLink.
                    Always faster than PCIe P2P when available.

### P2P Copy: cudaMemcpyPeer

    cudaMemcpyPeer(dst, dstDevice, src, srcDevice, size):
        Copies memory between two GPU devices.
        With P2P enabled: single DMA operation, no CPU involvement.
        Without P2P: staged through host — two separate DMA transfers.

    ASYNC VARIANT:
        cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, size, stream);
        Queued into a CUDA stream on the source device.
        Async: CPU returns immediately; GPU DMA runs in background.

    PERFORMANCE:
        NVLink P2P: up to 300-450 GB/s (NVLink 3/4 bidirectional).
        PCIe P2P:   up to 32 GB/s (PCIe Gen4 x16).

### P2P in Kernels: Direct Cross-GPU Memory Access

    With P2P enabled, CUDA kernels can dereference pointers to remote GPU memory:

        // GPU 0 kernel reading GPU 1's memory:
        __global__ void read_remote(float* gpu0_local, float* gpu1_remote, int N) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < N)
                gpu0_local[i] = gpu1_remote[i];   // NVLink/PCIe access
        }

    PERFORMANCE CHARACTERISTICS:
        Latency per access: ~1 µs (NVLink), ~5 µs (PCIe P2P).
        This is 1000-5000× slower than local L2 cache access.
        For streaming access (sequential reads): bandwidth approaches link peak.
        For random access: latency dominates — very slow for scattered lookups.

    USE CASES:
        Streaming: tensor parallel layer outputs copied to next GPU. ✅
        Streaming: KV cache scatter for paged attention across GPUs. ✅
        Random: embedding table lookups from remote GPU. ❌ (use DtoD copy)

### Unified Virtual Addressing (UVA) and cudaPointerAttributes

    UVA (CUDA 4.0+): all GPU memory allocations share a single virtual address space.
    Any CUDA pointer can be dereferenced from any GPU (after P2P enable).
    cudaMemcpy auto-detects the direction from pointer provenance.

        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, ptr);
        // attr.device: which GPU this pointer belongs to
        // attr.type:   cudaMemoryTypeDevice, cudaMemoryTypeHost, etc.

    PRACTICAL EFFECT: code doesn't need to track "which GPU owns this tensor."
    The CUDA runtime figures it out from the virtual address.

### IPC Handles: Sharing GPU Memory Across Processes

    For multi-process GPU sharing (e.g., separate inference workers sharing a model):
        cudaIpcMemHandle_t handle;
        cudaIpcGetMemHandle(&handle, d_ptr);   // in process A
        // send handle to process B via socket/shared memory/pipe
        float* remote_ptr;
        cudaIpcOpenMemHandle(&remote_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
        // process B can now access process A's GPU memory via remote_ptr

    IPC handles work only on the SAME physical machine.
    For cross-machine sharing: use NCCL or GPUDirect RDMA.
    PyTorch uses IPC handles for multi-process DataLoader GPU memory sharing.


##### PART 4 — GPUDIRECT STORAGE AND RDMA: BYPASSING THE CPU

### GPUDirect Technology Family

    NVIDIA GPUDirect is a family of related technologies that allow GPU
    memory to be accessed directly by other hardware without CPU mediation:

    GPUDIRECT RDMA (GPU Memory ↔ Network Adapter):
        An InfiniBand or RoCE NIC can DMA directly from/to GPU HBM.
        No CPU involved in the data path for network transfers.
        Enabled by: GPU and NIC sharing the same PCIe root complex,
                    and the NIC driver supporting GPUDirect.
        Bandwidth: limited by PCIe Gen4 x16 to the NIC (32 GB/s per NIC).

    GPUDIRECT STORAGE (GPU Memory ↔ NVMe/SSD):
        GPU can read/write NVMe storage without staging through CPU DRAM.
        cuFile API: cuFileRead, cuFileWrite.
        Avoids the CPU bounce buffer for large checkpoint reads/writes.
        Bandwidth: limited by NVMe bandwidth (7-12 GB/s per drive).

    GPUDIRECT ASYNC (GPU Compute ↔ Network with minimal CPU):
        CUDA extension allowing GPU kernels to trigger DMA operations.
        Used by NCCL for low-latency collective communication.

### GPUDirect RDMA: How It Works

    WITHOUT GPUDirect RDMA:
        1. GPU DMA: GPU HBM → CPU pinned buffer (via PCIe)
        2. IB DMA: CPU pinned buffer → Network (via PCIe)
        3. Network → remote CPU pinned buffer
        4. Remote GPU DMA: CPU buffer → GPU HBM

        Total copies: 4 DMA operations, 2 CPU-side pinned buffers.
        Effective bandwidth: limited by PCIe × 2 (two PCIe traversals).

    WITH GPUDirect RDMA:
        1. IB DMA: GPU HBM → Network (via PCIe, no CPU buffer)
        2. Network → remote GPU HBM (directly)

        Total copies: 2 DMA operations, 0 CPU-side buffers.
        Effective bandwidth: limited by min(NIC bandwidth, PCIe bandwidth).
        Latency: ~1-2 µs removed (no CPU DMA setup for bounce buffer).

### PCIe Topology for GPUDirect RDMA

    The most critical requirement: the GPU and the NIC must be on the SAME
    PCIe ROOT COMPLEX (or as close as possible in the PCIe tree).
    If they're on different root complexes: traffic must cross the CPU die.
    This is called the "P2P RDMA" problem.

    IDEAL TOPOLOGY (DGX A100):
        GPU 0-3 → PCIe switch → Root Complex A → NIC 0
        GPU 4-7 → PCIe switch → Root Complex B → NIC 1
        GPU 0-3 use NIC 0 for RDMA (same root complex).
        GPU 4-7 use NIC 1.
        Traffic never crosses the CPU interconnect.

    BAD TOPOLOGY (mismatched GPU-NIC):
        GPU 0 → PCIe → CPU → PCIe → NIC 1 (different root complex).
        Bandwidth penalty: 30-50% reduction due to CPU QPI/UPI crossing.
        Latency penalty: ~2-4 µs additional CPU routing latency.

    DIAGNOSING TOPOLOGY:
        nvidia-smi topo --matrix   shows the interconnect matrix.
        Look for NV# (NVLink), PIX (same PCIe root), PXB (same bridge),
        PHB (same host bridge), SYS (across CPU interconnect).

### InfiniBand Verbs API with GPUDirect

    Standard ibv_reg_mr (register memory region):
        ibv_mr* mr = ibv_reg_mr(pd, gpu_ptr, size, IBV_ACCESS_LOCAL_WRITE |
                                  IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
        This registers the GPU memory with the IB NIC directly.
        The NIC gets the physical address of the GPU HBM pages.

    RDMA SEND with GPU memory:
        ibv_send_wr wr = {};
        wr.sg_list->addr   = (uint64_t)gpu_ptr;   // GPU HBM address
        wr.sg_list->lkey   = mr->lkey;
        wr.opcode          = IBV_WR_RDMA_WRITE;
        wr.wr.rdma.remote_addr = remote_gpu_addr;
        ibv_post_send(qp, &wr, &bad_wr);
        // NIC reads from GPU HBM directly and sends over IB fabric.


##### PART 5 — NVSwitch: THE FULL-BISECTION GPU CROSSBAR

### NVSwitch Architecture

    An NVSwitch is a standalone ASIC that acts as a fabric switch for NVLink.
    It is NOT a GPU — it has no SM compute, no HBM, no CUDA cores.
    Its sole function: route NVLink packets from any input port to any output.

    NVSWITCH GENERATIONS:
        NVSwitch 1.0 (Volta era):   18 NVLink 2.0 ports, 900 GB/s aggregate.
        NVSwitch 2.0 (Ampere era):  36 NVLink 3.0 ports, 1.8 TB/s aggregate.
        NVSwitch 3.0 (Hopper era):  64 NVLink 4.0 ports, 3.2 TB/s aggregate.

    NVSwitch 3.0 in DGX H100:
        4 NVSwitch chips × 64 ports each = 256 port-pairs total.
        Each H100 GPU connects to all 4 NVSwitches (18 NVLink / 4 = 4-5 links each).
        Any GPU can communicate with any other at full 900 GB/s (no contention).

### All-to-All Bandwidth in NVSwitch Systems

    With a full-bisection NVSwitch fabric:
        ALL GPU pairs can communicate simultaneously at full bandwidth.
        No blocking: one GPU-to-GPU transfer doesn't slow others.

    PRACTICAL IMPLICATION for training:
        AllReduce on 8 GPUs: ring algorithm uses 2(N-1) steps.
        NVSwitch allows ALL N GPUs to communicate simultaneously.
        NCCL NVLS algorithm (H100+): single-step hardware reduction.
        AllReduce for 14 GB (Llama-7B) on H100 DGX: ~31 ms (ring) vs ~20 ms (NVLS).

### Multi-Node: NVLink + InfiniBand Scale-Out

    At node boundaries, NVLink stops and InfiniBand begins:
        Within node (8 GPUs): NVLink fabric, 900 GB/s per GPU.
        Between nodes: InfiniBand HDR/NDR, 25-50 GB/s per GPU.
        Ratio: 18-36× bandwidth gap at the node boundary.

    NCCL's two-tier strategy:
        Tier 1 (intra-node): NVLink ring/NVLS for intra-node reduction.
        Tier 2 (inter-node): InfiniBand RDMA for inter-node reduction.
        The 8 GPUs within each node first reduce among themselves,
        then the 8 "result" GPUs do the inter-node AllReduce via IB.
        This minimises inter-node traffic by N× (only node-aggregated data crosses IB).

    HIERARCHICAL ALLREDUCE:
        Step 1: ReduceScatter within node (NVLink).
        Step 2: AllReduce across nodes on shard (InfiniBand).
        Step 3: AllGather within node (NVLink).
        IB traffic: (N_gpus_per_node - 1) / N_gpus_per_node reduction.


##### PART 6 — UNIFIED MEMORY AND MULTI-GPU MEMORY MANAGEMENT

### Unified Memory (UM): cudaMallocManaged

    cudaMallocManaged allocates memory accessible from ANY GPU or CPU:
        float* d;
        cudaMallocManaged(&d, size, cudaMemAttachGlobal);

    The CUDA runtime manages migration automatically:
        First access by GPU A: page faulted to GPU A's HBM.
        Subsequent access by GPU B: page migrated to GPU B's HBM via NVLink.
        Prefetching: cudaMemPrefetchAsync(d, size, device, stream) to migrate proactively.

    PASCAL+ (P100+) HARDWARE PAGE MIGRATION:
        GPU MMU can handle page faults from kernels (not just the CPU).
        Pages migrate 64 KB at a time (one UM page = multiple 4 KB OS pages).
        Migration is handled by the CUDA UVM driver; no explicit copies needed.

    WHEN UM IS USEFUL:
        Algorithms with irregular data access (graph traversal, sparse operations).
        Multi-GPU cooperative algorithms where access pattern is hard to predict.
        Prototyping — avoid manual cudaMemcpy calls.

    WHEN UM IS NOT OPTIMAL:
        Training workloads with predictable data layout — explicit copies are faster.
        Any loop where the access pattern is known ahead of time.
        UM migration adds ~10-20 µs latency per 64 KB page fault.

### Memory Oversubscription with UM

    With UM, a single GPU can "use" more memory than it physically has.
    Pages not currently needed are evicted to another GPU or system DRAM.

    EXAMPLE: 4 × A100 (80 GB each) + UM.
        Total UM capacity: 4 × 80 GB + system DRAM (terabytes).
        A single allocation of 200 GB is valid — pages migrate on demand.
        An access pattern that touches all 200 GB on GPU 0: migrations occur.
        Each migration: ~2 ms per 64 KB page (bandwidth limited).
        For 200 GB: 200 GB / 64 KB pages = 3.2M migrations × 2 µs = 6.4 seconds!

    MORAL: UM is not a substitute for careful memory management in perf-critical code.


##### PART 7 — PROFILING MULTI-GPU TRANSFERS WITH NSIGHT AND nvidia-smi

### Topology Discovery: nvidia-smi topo

    nvidia-smi topo --matrix shows the interconnect between all GPUs and NICs:
        NV# = NVLink (# = number of links between this pair).
        PIX = PCIe P2P within same PCIe switch.
        PXB = PCIe P2P across PCIe bridge.
        PHB = same PCIe host bridge (e.g., same CPU socket).
        SYS = across CPU interconnect (NUMA crossing, slowest).
        MIG = in same Multi-Instance GPU.

    For A100 DGX, expected output: all GPU pairs show NV12 (12 NVLink paths).
    For a workstation with 2 GPUs (no NVLink): may show PHB or SYS.

### Measuring P2P Bandwidth

    NCCL provides nccl-tests for benchmarking:
        git clone https://github.com/NVIDIA/nccl-tests && make
        ./build/all_reduce_perf -b 1K -e 1G -f 2 -g 8
        Output: bus bandwidth for each message size.

    NVIDIA's p2pBandwidthLatencyTest (CUDA samples):
        Measures bidirectional bandwidth between every GPU pair.
        Reports whether P2P is enabled (NVLink) or disabled (staging).

### Nsight Systems for P2P Transfers

    Nsight Systems shows P2P activity on the GPU timeline:
        DMA transfers between GPUs appear on the NVLink row.
        Kernel execution and NVLink transfers can be seen overlapping.
        Look for: GPU-A sends while GPU-A kernel runs (DtoD overlap).

    NCU for access pattern analysis:
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum: counts global memory loads.
        High cross-GPU access rate: these show up as L2 misses followed by
        NVLink transactions (visible in the GPU interconnect traffic counters).

### NCCL_DEBUG and P2P Diagnostics

    NCCL_DEBUG=INFO prints topology and algorithm decisions:
        [0] NCCL INFO Ring 0 : 0 -> 1 -> 2 -> 3 -> 0 via NVL
        Shows: which ring order NCCL chose, and the transport used (NVL vs NET).

    NCCL_P2P_DISABLE=1: force disable P2P (use staging through host).
    NCCL_NET_DISABLE_INTRA=1: force all intra-node traffic over network.
    These are useful for isolating whether P2P is helping.

"""

OPERATIONS = {

    "1 · NVLink Topology Model — Bandwidth Matrix & Switch Routing": {
        "description": (
            "Build a complete topology model for DGX A100 and H100 systems. "
            "Compute the bandwidth matrix between all GPU pairs accounting for "
            "direct NVLink connections vs NVSwitch routing. Show the bisection "
            "bandwidth. Compare A100 DGX, H100 DGX, and a PCIe-only system. "
            "Model the contention when multiple GPU pairs communicate simultaneously."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  NVLink TOPOLOGY MODEL — Bandwidth Matrix & Switch Routing")
print("=" * 68)
print()

np.random.seed(42)

# Hardware specs per generation
NVLINK_GENS = {
    "NVLink 2 (V100)":  {"links_per_gpu": 6,  "bw_per_link_gbs": 25.0},
    "NVLink 3 (A100)":  {"links_per_gpu": 12, "bw_per_link_gbs": 25.0},
    "NVLink 4 (H100)":  {"links_per_gpu": 18, "bw_per_link_gbs": 25.0},
}

TOPOLOGIES = {
    "DGX A100 (6 NVSwitch 2.0)": {
        "n_gpus": 8, "nvswitch_count": 6, "nvlink_gen": "NVLink 3 (A100)",
        "links_per_gpu_to_switch": 2,   # each GPU has 2 links to each NVSwitch
        "full_bisection": True,
    },
    "DGX H100 (4 NVSwitch 3.0)": {
        "n_gpus": 8, "nvswitch_count": 4, "nvlink_gen": "NVLink 4 (H100)",
        "links_per_gpu_to_switch": 4,   # ~4-5 links per GPU per switch
        "full_bisection": True,
    },
    "PCIe server (no NVLink)": {
        "n_gpus": 8, "nvswitch_count": 0, "nvlink_gen": None,
        "links_per_gpu_to_switch": 0,
        "full_bisection": False,
    },
}


def compute_p2p_bandwidth(topology_name, topo):
    """
    Compute the bandwidth matrix between all GPU pairs.
    Returns: N×N matrix of unidirectional bandwidth (GB/s).
    """
    n = topo["n_gpus"]
    bw = np.zeros((n, n))

    if topo["nvlink_gen"] is None:
        # PCIe only: 32 GB/s P2P if same root complex, else 15 GB/s via CPU
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Assume first 4 GPUs on one root complex, second 4 on another
                    same_root = (i < 4 and j < 4) or (i >= 4 and j >= 4)
                    bw[i][j] = 32.0 if same_root else 15.0
        return bw

    gen     = NVLINK_GENS[topo["nvlink_gen"]]
    link_bw = gen["bw_per_link_gbs"]
    n_sw    = topo["nvswitch_count"]
    lpgs    = topo["links_per_gpu_to_switch"]  # links per GPU per switch

    # Each GPU has n_sw * lpgs NVLink links going to switches
    # These links are shared among all other GPUs via the switch
    # Bandwidth to any other GPU = total switch links / (n_gpus - 1) × link_bw
    # In a full-bisection fabric, each GPU pair gets equal share

    if topo["full_bisection"]:
        # Each GPU's N-1 potential peer paths share the GPU's total switch bandwidth
        total_uplinks_per_gpu = n_sw * lpgs   # total links to switches
        per_pair_bw = total_uplinks_per_gpu * link_bw / (n - 1)
        for i in range(n):
            for j in range(n):
                if i != j:
                    bw[i][j] = per_pair_bw
    return bw


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Bandwidth matrices for each topology
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Bandwidth Matrix: GPU-to-GPU Unidirectional GB/s")
print("━" * 68)
print()

for topo_name, topo in TOPOLOGIES.items():
    n = topo["n_gpus"]
    bw = compute_p2p_bandwidth(topo_name, topo)

    gen = NVLINK_GENS.get(topo["nvlink_gen"], {})
    links = gen.get("links_per_gpu", 0)
    link_bw = gen.get("bw_per_link_gbs", 0)

    print(f"  {topo_name}:")
    if topo["nvlink_gen"]:
        print(f"    {topo['nvlink_gen']}: {links} links/GPU × {link_bw} GB/s = "
              f"{links*link_bw} GB/s total/GPU")
        print(f"    NVSwitch count: {topo['nvswitch_count']}")
    print(f"    Max GPU-to-GPU bandwidth: {bw[0][1]:.1f} GB/s (unidirectional)")
    print()

    # Print bandwidth matrix (first 5 GPUs for readability)
    print(f"    BW matrix (GB/s, GPU 0-{min(4,n-1)}):")
    print(f"      {'':>6}", end="")
    for j in range(min(5, n)):
        print(f"  GPU{j}", end="")
    print()
    for i in range(min(5, n)):
        print(f"      GPU{i}:", end="")
        for j in range(min(5, n)):
            if i == j:
                print(f"  {'—':>4}", end="")
            else:
                print(f"  {bw[i][j]:>4.0f}", end="")
        print()
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Bisection bandwidth analysis
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Bisection Bandwidth Analysis")
print("━" * 68)
print()

print("  Bisection bandwidth: total BW across the cut dividing GPUs in half.")
print("  Full bisection = no congestion under any traffic pattern.")
print()

for topo_name, topo in TOPOLOGIES.items():
    n = topo["n_gpus"]
    bw = compute_p2p_bandwidth(topo_name, topo)

    # Bisection: GPUs 0..N/2-1 → GPUs N/2..N-1 simultaneously
    # Each source GPU sends to its counterpart on the other side
    n_half = n // 2
    bisection_per_pair = bw[0][n_half]   # representative pair
    total_bisection    = bisection_per_pair * n_half  # all n/2 pairs simultaneously

    gen    = NVLINK_GENS.get(topo["nvlink_gen"], {})
    links  = gen.get("links_per_gpu", 0)
    lb     = gen.get("bw_per_link_gbs", 0)
    total_switch_bw = topo["nvswitch_count"] * 64 * lb if topo.get("nvswitch_count") else 0

    print(f"  {topo_name}:")
    print(f"    Per GPU-pair bandwidth:   {bisection_per_pair:>8.1f} GB/s")
    print(f"    Total bisection ({n_half} pairs): {total_bisection:>8.1f} GB/s")
    print(f"    Full bisection?           {'Yes ✅' if topo['full_bisection'] else 'No  ❌'}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Contention model — simultaneous transfers
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Contention Model: Multiple Simultaneous Transfers")
print("━" * 68)
print()

print("  DGX H100: full-bisection NVSwitch fabric.")
print("  All GPU pairs can communicate simultaneously WITHOUT contention.")
print()

topo_h100 = TOPOLOGIES["DGX H100 (4 NVSwitch 3.0)"]
bw_h100   = compute_p2p_bandwidth("DGX H100 (4 NVSwitch 3.0)", topo_h100)
N_H100    = topo_h100["n_gpus"]

transfer_patterns = [
    ("1 pair (GPU 0→1)", [(0, 1)]),
    ("2 pairs (0→1, 2→3)", [(0,1), (2,3)]),
    ("4 pairs (all-to-all halves)", [(0,4),(1,5),(2,6),(3,7)]),
    ("Ring (0→1→2→...→7→0)", [(i,(i+1)%N_H100) for i in range(N_H100)]),
    ("8 simultaneous random", [(i,(i+3)%N_H100) for i in range(N_H100)]),
]

print(f"  {'Transfer pattern':<42}  {'BW per pair':>12}  "
      f"{'Total aggregate':>16}  {'Contention?'}")
print("  " + "─" * 74)

per_gpu_total_bw = topo_h100["nvswitch_count"] * topo_h100["links_per_gpu_to_switch"] * \
                    NVLINK_GENS["NVLink 4 (H100)"]["bw_per_link_gbs"]

for name, pairs in transfer_patterns:
    # In full bisection: each pair gets its nominal bandwidth
    # (NVSwitch has enough routing bandwidth for all simultaneously)
    bw_per_pair = bw_h100[0][1] if bw_h100[0][1] > 0 else 0
    total_bw    = bw_per_pair * len(pairs)
    contention  = "None ✅" if topo_h100["full_bisection"] else "Yes ❌"
    print(f"  {name:<42}  {bw_per_pair:>10.1f}G  "
          f"{total_bw:>14.1f}G  {contention}")

print()
print("  Full-bisection switch: total aggregate scales linearly with pairs.")
print("  PCIe server: bottleneck at root complex → contention for 3+ pairs.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: NVLink vs PCIe vs CPU-bounce transfer time
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Transfer Time: NVLink vs PCIe P2P vs CPU Bounce")
print("━" * 68)
print()

transfer_paths = [
    ("NVLink 4 (H100)",  450.0, 0.8,   "GPU-to-GPU via NVSwitch"),
    ("NVLink 3 (A100)",  300.0, 1.0,   "GPU-to-GPU via NVSwitch"),
    ("PCIe Gen4 P2P",    32.0,  5.0,   "GPU-to-GPU via PCIe, no NVLink"),
    ("PCIe + CPU bounce",16.0,  12.0,  "Staged via host DRAM (no P2P)"),
]

sizes_gb = [0.001, 0.01, 0.1, 0.5, 1.0, 14.0]

print(f"  {'Size (GB)':>12}", end="")
for name, _, _, _ in transfer_paths:
    print(f"  {name:>22}", end="")
print()
print(f"  {'':>12}", end="")
for _ in transfer_paths:
    print(f"  {'(ms)':>22}", end="")
print()
print("  " + "─" * (14 + 24 * len(transfer_paths)))

for sz in sizes_gb:
    print(f"  {sz:>12.3f}", end="")
    for name, bw, lat_us, desc in transfer_paths:
        t_ms = lat_us / 1000 + sz * 1e9 / (bw * 1e9) * 1000
        print(f"  {t_ms:>22.3f}", end="")
    print()

print()
print("  Llama-2-7B model (14 GB) transfer time:")
for name, bw, lat_us, desc in transfer_paths:
    t_s = lat_us/1e6 + 14e9/(bw*1e9)
    print(f"    {name:<28}: {t_s*1000:.0f} ms ({desc})")
''',
    },

    "2 · P2P Transfer Simulator — Enable / Disable, Routing & Bandwidth": {
        "description": (
            "Simulate the P2P enable/disable workflow, routing decisions, and "
            "effective bandwidth for different GPU topologies. Show the difference "
            "between enabled P2P (direct DMA) and disabled (CPU-staged copy). "
            "Implement cudaDeviceCanAccessPeer logic based on topology. "
            "Model the UVA pointer lookup mechanism. Show how IPC handles enable "
            "cross-process GPU memory sharing."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

print("=" * 68)
print("  P2P TRANSFER SIMULATOR — Enable/Disable, Routing & Bandwidth")
print("=" * 68)
print()

np.random.seed(42)


@dataclass
class GPUNode:
    gpu_id:      int
    hbm_gb:      float
    pcie_root:   int     # which PCIe root complex this GPU belongs to
    nvlink_peers: List[int] = field(default_factory=list)  # GPU IDs with direct NVLink


@dataclass
class P2PTransfer:
    src_gpu:  int
    dst_gpu:  int
    size_gb:  float
    p2p_enabled: bool

    def effective_bandwidth_gbs(self, topo: Dict[int, GPUNode]) -> float:
        src = topo[self.src_gpu]
        dst = topo[self.dst_gpu]
        if self.dst_gpu in src.nvlink_peers:
            return 300.0  # NVLink 3 (simulated A100)
        elif self.p2p_enabled and src.pcie_root == dst.pcie_root:
            return 32.0   # PCIe Gen4 P2P
        elif self.p2p_enabled:
            return 16.0   # PCIe P2P across root complex (CPU-assisted)
        else:
            return 15.0   # CPU bounce (staging through host)

    def latency_us(self, topo: Dict[int, GPUNode]) -> float:
        src = topo[self.src_gpu]
        dst = topo[self.dst_gpu]
        if self.dst_gpu in src.nvlink_peers:
            return 0.8   # NVLink 4 latency
        elif self.p2p_enabled and src.pcie_root == dst.pcie_root:
            return 5.0
        else:
            return 12.0

    def transfer_time_ms(self, topo: Dict[int, GPUNode]) -> float:
        bw  = self.effective_bandwidth_gbs(topo)
        lat = self.latency_us(topo) / 1000
        return lat + self.size_gb * 1e9 / (bw * 1e9) * 1000

    def path_description(self, topo: Dict[int, GPUNode]) -> str:
        src = topo[self.src_gpu]
        if self.dst_gpu in src.nvlink_peers:
            return "NVLink (direct)"
        elif self.p2p_enabled and src.pcie_root == topo[self.dst_gpu].pcie_root:
            return "PCIe P2P (same root)"
        elif self.p2p_enabled:
            return "PCIe P2P (cross root)"
        else:
            return "CPU bounce (staged)"


def make_dgx_a100_topology() -> Dict[int, GPUNode]:
    """
    Simulate DGX A100 topology:
    - 8 GPUs, all connected via NVSwitch (full any-to-any NVLink)
    - Two PCIe root complexes: GPUs 0-3 on RC0, GPUs 4-7 on RC1
    """
    topo = {}
    for i in range(8):
        nvlink_peers = [j for j in range(8) if j != i]  # all-to-all via NVSwitch
        rc = 0 if i < 4 else 1
        topo[i] = GPUNode(gpu_id=i, hbm_gb=80.0, pcie_root=rc, nvlink_peers=nvlink_peers)
    return topo


def make_pcie_server_topology() -> Dict[int, GPUNode]:
    """
    Simulate PCIe-only server (no NVLink):
    - 8 GPUs, 2 root complexes (GPUs 0-3 on RC0, 4-7 on RC1)
    - No NVLink connections
    """
    topo = {}
    for i in range(8):
        rc = 0 if i < 4 else 1
        topo[i] = GPUNode(gpu_id=i, hbm_gb=24.0, pcie_root=rc, nvlink_peers=[])
    return topo


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: P2P access matrix for different topologies
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — P2P Access Matrix: cudaDeviceCanAccessPeer Results")
print("━" * 68)
print()

for topo_name, topo_fn in [("DGX A100 (NVLink)", make_dgx_a100_topology),
                             ("PCIe server (no NVLink)", make_pcie_server_topology)]:
    topo = topo_fn()
    N    = len(topo)
    print(f"  {topo_name}:")
    print(f"  cudaDeviceCanAccessPeer matrix (first 6 GPUs):")
    print()
    print(f"    {'':>8}", end="")
    for j in range(min(6, N)):
        print(f"  GPU{j}", end="")
    print()

    for i in range(min(6, N)):
        print(f"    GPU{i}:", end="")
        for j in range(min(6, N)):
            if i == j:
                access = "  —"
            elif j in topo[i].nvlink_peers:
                access = " NVL"   # NVLink → can access
            elif topo[i].pcie_root == topo[j].pcie_root:
                access = " PIX"   # same PCIe root
            else:
                access = " PHB"   # different root (can still P2P, slower)
            print(f"  {access}", end="")
        print()
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Transfer path and bandwidth comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Transfer Path and Effective Bandwidth")
print("━" * 68)
print()

topo_dgx   = make_dgx_a100_topology()
topo_pcie  = make_pcie_server_topology()

test_transfers = [
    (0, 1, 1.0, "1 GB, adjacent GPUs"),
    (0, 4, 1.0, "1 GB, cross root complex"),
    (0, 7, 14.0,"14 GB (Llama-7B), GPU 0→7"),
]

for src, dst, size, label in test_transfers:
    print(f"  Transfer: GPU{src} → GPU{dst}, {size} GB  ({label})")
    print()
    print(f"  {'Topology':<30}  {'P2P?':>6}  {'Path':<22}  "
          f"{'BW (GB/s)':>10}  {'Time (ms)':>12}")
    print("  " + "─" * 70)

    for topo_name, topo in [("DGX A100 (NVLink)", topo_dgx),
                              ("PCIe server", topo_pcie)]:
        for p2p in [True, False]:
            xfer = P2PTransfer(src, dst, size, p2p)
            bw   = xfer.effective_bandwidth_gbs(topo)
            t_ms = xfer.transfer_time_ms(topo)
            path = xfer.path_description(topo)
            p2p_str = "yes" if p2p else "no"
            print(f"  {topo_name:<30}  {p2p_str:>6}  {path:<22}  "
                  f"{bw:>10.1f}  {t_ms:>12.3f}")

    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: UVA pointer lookup simulation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — UVA Pointer Attributes: Identifying GPU Ownership")
print("━" * 68)
print()

print("  Unified Virtual Addressing (UVA): all GPU allocations share one address space.")
print("  cudaPointerGetAttributes reveals which device owns a pointer.")
print()

# Simulate UVA address ranges for 4 GPUs
GPU_VADDR_BASE = 0x7F_0000_0000_0000
GPU_VADDR_SIZE = 0x0001_0000_0000_0000   # 1 PiB per GPU

# Simulated allocations
allocations = [
    (0, GPU_VADDR_BASE + 0 * GPU_VADDR_SIZE + 0x1000, 4 * 1024**3,  "model weights GPU 0"),
    (1, GPU_VADDR_BASE + 1 * GPU_VADDR_SIZE + 0x1000, 4 * 1024**3,  "activations GPU 1"),
    (2, GPU_VADDR_BASE + 2 * GPU_VADDR_SIZE + 0x2000, 1 * 1024**3,  "KV cache GPU 2"),
    (0, GPU_VADDR_BASE + 0 * GPU_VADDR_SIZE + 0x5000, 512 * 1024,   "attention scores GPU 0"),
]

print(f"  {'Pointer (hex)':>20}  {'Owner GPU':>10}  {'Size':>12}  {'Description'}")
print("  " + "─" * 62)
for gpu_id, vaddr, size_bytes, desc in allocations:
    size_str = (f"{size_bytes//1024**3} GB" if size_bytes >= 1024**3
                else f"{size_bytes//1024**2} MB")
    print(f"  0x{vaddr:014X}  {gpu_id:>10}  {size_str:>12}  {desc}")

print()
print("  With UVA: a kernel on GPU 3 can dereference any of these pointers.")
print("  The GPU's IOMMU routes the access to the correct physical device.")
print("  Without P2P enabled: access causes a fault and falls back to CPU.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: IPC handle workflow for multi-process sharing
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — IPC Handles: Cross-Process GPU Memory Sharing")
print("━" * 68)
print()

print("  USE CASE: inference server with multiple worker processes sharing one model.")
print()
print("  Process A (model loader):")
print("    cudaMalloc(&d_weights, 14GB);")
print("    load_weights(d_weights);")
print("    cudaIpcGetMemHandle(&handle, d_weights);   // get shareable handle")
print("    send_handle_over_socket(handle);            // send to worker processes")
print()
print("  Process B (inference worker):")
print("    recv_handle_from_socket(&handle);")
print("    float* weights_remote;")
print("    cudaIpcOpenMemHandle(&weights_remote, handle,")
print("                         cudaIpcMemLazyEnablePeerAccess);")
print("    // Now weights_remote points to process A's GPU memory!")
print("    inference_kernel<<<..., stream>>>(weights_remote, input, output);")
print()
print("  PROPERTIES:")

ipc_props = [
    ("Same machine only",   "IPC handles cannot cross the network (use NCCL for multi-node)"),
    ("Same device only",    "Both processes must use the same GPU device"),
    ("Reference counting",  "Memory freed when ALL IPC handles are closed"),
    ("Zero-copy sharing",   "No data is copied: both processes map the same physical HBM pages"),
    ("PyTorch support",     "torch.multiprocessing uses IPC for shared CUDA tensors"),
]

for prop, desc in ipc_props:
    print(f"    {prop:<24}: {desc}")

print()
print("  MEMORY COST COMPARISON (14 GB model):")
print("    Without IPC: each of 4 workers loads model → 4 × 14 GB = 56 GB VRAM")
print("    With IPC:    one process loads, all share → 1 × 14 GB = 14 GB VRAM")
print("    Saving: 42 GB (75% reduction) — enables more concurrent workers")
''',
    },

    "3 · GPUDirect RDMA Model — Path Analysis, Latency & Topology Fit": {
        "description": (
            "Model GPUDirect RDMA: compare the four-DMA path (no GPUDirect) "
            "against the two-DMA path (with GPUDirect). Compute effective bandwidth "
            "and latency for both. Show the PCIe topology constraint — GPU and NIC "
            "must share the same root complex for full bandwidth. Simulate "
            "hierarchical AllReduce using NVLink (intra-node) and RDMA (inter-node) "
            "and compute the bandwidth savings from GPUDirect."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  GPUDirect RDMA MODEL — Path Analysis, Latency & Topology Fit")
print("=" * 68)
print()

np.random.seed(7)

# Hardware constants
PCIE_GEN4_GBS    = 32.0   # PCIe Gen4 x16 unidirectional
IB_NDR_GBS       = 50.0   # InfiniBand NDR-400 unidirectional (50 GB/s)
IB_HDR_GBS       = 25.0   # InfiniBand HDR-200 unidirectional
NVLINK3_GBS      = 300.0  # NVLink 3.0 bidirectional per GPU
NVLINK4_GBS      = 450.0  # NVLink 4.0 bidirectional per GPU

# Latency components
PCIE_LAT_US      = 1.0    # per PCIe round-trip
IB_FABRIC_US     = 1.0    # InfiniBand fabric latency
NCCL_PROTO_US    = 3.0    # NCCL protocol overhead
CPU_BOUNCE_US    = 5.0    # CPU DMA setup for bounce buffer (each direction)
GPUDIRECT_SETUP_US = 0.5  # GPUDirect registration overhead (amortised)


def without_gpudirect(size_bytes, ib_gbs=IB_HDR_GBS):
    """
    Four-step DMA path (no GPUDirect):
    GPU HBM → PCIe → CPU pinned → IB NIC → fabric →
    remote CPU pinned → PCIe → remote GPU HBM
    """
    # Step 1: GPU DMA to CPU pinned buffer
    t_gpu_to_cpu = size_bytes / (PCIE_GEN4_GBS * 1e9) * 1e6   # µs
    lat_gpu_cpu  = CPU_BOUNCE_US

    # Step 2: IB NIC DMA from CPU buffer to remote
    t_ib_send    = size_bytes / (ib_gbs * 1e9) * 1e6
    lat_ib       = IB_FABRIC_US + NCCL_PROTO_US

    # Step 3: Remote IB NIC to remote CPU pinned
    # (symmetric to send, but pipelined — latency already counted)

    # Step 4: Remote GPU DMA from CPU
    t_cpu_to_gpu = size_bytes / (PCIE_GEN4_GBS * 1e9) * 1e6
    lat_cpu_gpu  = CPU_BOUNCE_US

    # Total: latencies add, bandwidth limited by bottleneck
    total_lat_us = (lat_gpu_cpu + lat_ib + lat_cpu_gpu)
    # Bandwidth bottleneck: min of PCIe (both ends) and IB
    eff_bw_gbs   = min(PCIE_GEN4_GBS, ib_gbs, PCIE_GEN4_GBS)
    bw_us        = size_bytes / (eff_bw_gbs * 1e9) * 1e6

    return total_lat_us + bw_us, eff_bw_gbs


def with_gpudirect(size_bytes, ib_gbs=IB_HDR_GBS, same_root_complex=True):
    """
    Two-step DMA path (with GPUDirect RDMA):
    GPU HBM → PCIe → IB NIC → fabric → remote IB NIC → PCIe → remote GPU HBM
    """
    # If GPU and NIC are on different root complexes: bandwidth degraded
    pcie_bw = PCIE_GEN4_GBS if same_root_complex else PCIE_GEN4_GBS * 0.6
    lat_ib  = IB_FABRIC_US + NCCL_PROTO_US + GPUDIRECT_SETUP_US

    eff_bw_gbs = min(pcie_bw, ib_gbs)
    bw_us      = size_bytes / (eff_bw_gbs * 1e9) * 1e6

    return lat_ib + bw_us, eff_bw_gbs


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: GPUDirect RDMA path comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — GPUDirect RDMA: 4-DMA vs 2-DMA Path")
print("━" * 68)
print()

print("  WITHOUT GPUDirect RDMA (4 DMA operations):")
print("    GPU HBM → PCIe → CPU pinned → IB NIC → fabric → CPU pinned → PCIe → GPU HBM")
print()
print("  WITH GPUDirect RDMA (2 DMA operations):")
print("    GPU HBM → PCIe → IB NIC → fabric → IB NIC → PCIe → GPU HBM")
print()

sizes_mb = [1, 10, 100, 500, 1000, 14000]

print(f"  IB HDR-200 ({IB_HDR_GBS:.0f} GB/s), GPU and NIC on same PCIe root complex")
print()
print(f"  {'Size (MB)':>12}  {'Without RDMA (ms)':>20}  {'With RDMA (ms)':>16}  "
      f"{'Speedup':>10}  {'Latency saved'}")
print("  " + "─" * 64)

for size_mb in sizes_mb:
    size_bytes = size_mb * 1e6
    t_no_gd, bw_no = without_gpudirect(size_bytes, IB_HDR_GBS)
    t_gd, bw_gd    = with_gpudirect(size_bytes, IB_HDR_GBS, same_root_complex=True)
    speedup    = t_no_gd / t_gd
    lat_saved  = (t_no_gd - t_gd) / t_no_gd * 100
    print(f"  {size_mb:>12.0f}  {t_no_gd/1000:>20.4f}  {t_gd/1000:>16.4f}  "
          f"{speedup:>10.2f}×  {lat_saved:>8.1f}%")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: PCIe topology impact on GPUDirect bandwidth
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — PCIe Topology: GPU-NIC Same vs Different Root Complex")
print("━" * 68)
print()

print("  GPU and NIC must share the SAME PCIe root complex for full RDMA bandwidth.")
print("  If on different root complexes: traffic crosses CPU interconnect (QPI/UPI).")
print()

configs = [
    ("GPU + NIC: same root complex",       True,  IB_HDR_GBS, "Ideal DGX layout"),
    ("GPU + NIC: different root complex",  False, IB_HDR_GBS, "Suboptimal server"),
    ("GPU + NIC: same root, IB NDR-400",   True,  IB_NDR_GBS, "H100 DGX with NDR"),
]

SIZE_TEST = 1e9  # 1 GB

print(f"  Test: 1 GB message, IB transport")
print()
print(f"  {'Configuration':<42}  {'BW (GB/s)':>10}  {'Time (ms)':>12}  {'Notes'}")
print("  " + "─" * 70)

for name, same_root, ib_bw, note in configs:
    t_us, bw = with_gpudirect(SIZE_TEST, ib_bw, same_root)
    print(f"  {name:<42}  {bw:>10.1f}  {t_us/1000:>12.3f}  {note}")

print()
print("  DIAGNOSING YOUR TOPOLOGY:")
print("    nvidia-smi topo --matrix  (look for GPU-NIC affinity)")
print("    ibv_devinfo               (show IB device properties)")
print("    NCCL_DEBUG=INFO           (NCCL reports interconnect type used)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Hierarchical AllReduce — NVLink + RDMA pipeline
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Hierarchical AllReduce: NVLink (intra) + RDMA (inter)")
print("━" * 68)
print()

print("  Multi-node AllReduce strategy:")
print("  1. ReduceScatter within each node (NVLink — fast)")
print("  2. AllReduce across nodes (IB RDMA — inter-node bottleneck)")
print("  3. AllGather within each node (NVLink — fast)")
print()
print("  This reduces inter-node traffic by N_intra × (1 - 1/N_intra):")
print("  8 GPUs per node: only 7/8 of data sent inter-node vs flat ring.")
print()

MODELS = {
    "GPT-2 (1.5B)":  1.5e9 * 2,
    "Llama-2-7B":    7e9  * 2,
    "Llama-2-70B":   70e9 * 2,
}

NODE_CONFIGS = [
    (2,  8, NVLINK3_GBS, IB_HDR_GBS, "16 A100 (2 nodes × 8 GPUs)"),
    (4,  8, NVLINK3_GBS, IB_HDR_GBS, "32 A100 (4 nodes × 8 GPUs)"),
    (8,  8, NVLINK4_GBS, IB_NDR_GBS, "64 H100 (8 nodes × 8 GPUs)"),
    (16, 8, NVLINK4_GBS, IB_NDR_GBS, "128 H100 (16 nodes × 8 GPUs)"),
]

for model_name, param_bytes in [("Llama-2-7B", 7e9*2), ("Llama-2-70B", 70e9*2)]:
    print(f"  Model: {model_name} ({param_bytes/1e9:.0f} GB BF16)")
    print()
    print(f"  {'Config':<44}  {'Intra-node':>12}  {'Inter-node':>12}  "
          f"{'Total (ms)':>12}  {'Bottleneck'}")
    print("  " + "─" * 88)

    for n_nodes, n_intra, nvl_bw, ib_bw, label in NODE_CONFIGS:
        n_total = n_nodes * n_intra
        # Intra-node ReduceScatter (NVLink)
        intra_bytes = (n_intra - 1) / n_intra * param_bytes
        t_intra_us  = intra_bytes / (nvl_bw * 1e9) * 1e6 * 2  # RS + AG
        # Inter-node AllReduce (IB RDMA): each node sends 1/n_intra of tensor
        inter_bytes = param_bytes / n_intra  # shard that crosses network
        t_inter_us, _ = with_gpudirect(inter_bytes * 2, ib_bw)  # AllReduce over net
        t_total_ms = (t_intra_us + t_inter_us) / 1000

        bottleneck = "IB" if t_inter_us > t_intra_us else "NVLink"
        print(f"  {label:<44}  {t_intra_us/1000:>12.1f}  {t_inter_us/1000:>12.1f}  "
              f"{t_total_ms:>12.1f}  {bottleneck}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: GPUDirect vs no-GPUDirect training throughput impact
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Training Throughput: GPUDirect RDMA Impact")
print("━" * 68)
print()

N_NODES    = 8
N_PER_NODE = 8
COMPUTE_MS = 400.0   # ms per step (forward + backward)

print(f"  {N_NODES} nodes × {N_PER_NODE} GPUs = {N_NODES*N_PER_NODE} total GPUs")
print(f"  Compute per step: {COMPUTE_MS} ms (Llama-2-7B, H100, batch=64)")
print()
print(f"  {'Model':<18}  {'No GPUDirect':>14}  {'GPUDirect':>12}  "
      f"{'Speedup':>10}  {'Comm hidden?'}")
print("  " + "─" * 64)

for mname, p_bytes in [("Llama-2-7B", 7e9*2), ("Llama-2-70B", 70e9*2)]:
    # Inter-node portion (with hierarchical AllReduce)
    shard = p_bytes / N_PER_NODE

    t_no_gd, _ = without_gpudirect(shard * 2, IB_NDR_GBS)
    t_gd, _    = with_gpudirect(shard * 2, IB_NDR_GBS, same_root_complex=True)

    # Add intra-node NVLink time (same for both)
    t_nvl_us = (N_PER_NODE-1)/N_PER_NODE * p_bytes / (NVLINK4_GBS * 1e9) * 1e6 * 2
    t_total_no_gd = (t_no_gd + t_nvl_us) / 1000   # ms
    t_total_gd    = (t_gd + t_nvl_us) / 1000       # ms

    speedup       = t_total_no_gd / t_total_gd
    hidden_gd     = t_total_gd < COMPUTE_MS

    print(f"  {mname:<18}  {t_total_no_gd:>12.1f}ms  {t_total_gd:>10.1f}ms  "
          f"{speedup:>10.2f}×  {'✅ hidden' if hidden_gd else '❌ visible'}")

print()
print("  GPUDirect RDMA is essential for large-scale multi-node training.")
print("  Without it: CPU bounce doubles IB latency and wastes PCIe bandwidth.")
''',
    },

    "4 · Multi-GPU Memory Bandwidth Budget — P2P, Tensor Parallel & KV Cache": {
        "description": (
            "Build a complete multi-GPU memory bandwidth budget for LLM inference. "
            "Model tensor parallelism AllGather/ReduceScatter costs across NVLink. "
            "Compute the KV cache scatter/gather cost for paged attention. "
            "Show the bandwidth requirements for pipeline parallelism (activation "
            "transfers between pipeline stages). Find the bottleneck and the "
            "optimal GPU count for each parallelism dimension."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  MULTI-GPU MEMORY BANDWIDTH BUDGET — P2P, Tensor Parallel & KV Cache")
print("=" * 68)
print()

np.random.seed(42)

# Hardware: H100 DGX
NVL4_GBS     = 450.0   # NVLink4 unidirectional per GPU
H100_HBM_GBS = 3350.0  # H100 HBM bandwidth per GPU
H100_TFLOPS  = 989.0   # H100 FP16 tensor core TFLOPS

# Model: Llama-2-7B (representative)
D_MODEL      = 4096
N_HEADS      = 32
HEAD_DIM     = D_MODEL // N_HEADS
N_LAYERS     = 32
N_KV_HEADS   = 32   # (MHA; for GQA use 8)
VOCAB_SIZE   = 32000


def allgather_ms(n_gpus, tensor_bytes, bw_gbs, lat_us=0.8):
    """AllGather time: (N-1)/N × bytes / bw + lat."""
    vol  = (n_gpus - 1) / n_gpus * tensor_bytes
    t_us = (n_gpus - 1) * lat_us + vol / (bw_gbs * 1e9) * 1e6
    return t_us / 1000

def reduce_scatter_ms(n_gpus, tensor_bytes, bw_gbs, lat_us=0.8):
    return allgather_ms(n_gpus, tensor_bytes, bw_gbs, lat_us)  # same volume


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Tensor parallelism communication per layer
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Tensor Parallelism: AllGather/RS per Layer (Llama-2-7B)")
print("━" * 68)
print()

BATCH     = 1      # decode batch size
SEQ_LEN   = 1     # decode step: 1 token
FEAT_BYTES = 2    # BF16

print(f"  Decode step: batch={BATCH}, seq={SEQ_LEN}, BF16")
print(f"  Model: Llama-2-7B ({D_MODEL}d, {N_HEADS}h, {N_LAYERS}L)")
print(f"  NVLink4: {NVL4_GBS} GB/s unidirectional")
print()

print(f"  {'TP degree':>10}  {'GPUs/node':>10}  {'AG (attention)':>16}  "
      f"{'RS (attention)':>16}  {'AG (FFN gate)':>16}  {'Total/layer':>14}")
print("  " + "─" * 84)

for tp_degree in [1, 2, 4, 8]:
    # Tensor parallel: each GPU holds d/TP columns of weight matrices
    # Forward attention: input allgather over TP GPUs
    # Input to attn: [batch, seq, d_model] = batch × seq × d_model
    attn_input_bytes = BATCH * SEQ_LEN * D_MODEL * FEAT_BYTES
    t_ag_attn = allgather_ms(tp_degree, attn_input_bytes, NVL4_GBS)

    # After attention output projection: reduce-scatter
    attn_out_bytes = BATCH * SEQ_LEN * D_MODEL * FEAT_BYTES
    t_rs_attn = reduce_scatter_ms(tp_degree, attn_out_bytes, NVL4_GBS)

    # FFN gate projection input: allgather
    ffn_input_bytes = BATCH * SEQ_LEN * D_MODEL * FEAT_BYTES
    t_ag_ffn = allgather_ms(tp_degree, ffn_input_bytes, NVL4_GBS)

    t_total = t_ag_attn + t_rs_attn + t_ag_ffn   # one layer comm

    print(f"  {tp_degree:>10}  {min(tp_degree,8):>10}  "
          f"{t_ag_attn*1000:>14.3f}µs  {t_rs_attn*1000:>14.3f}µs  "
          f"{t_ag_ffn*1000:>14.3f}µs  {t_total*1000:>12.3f}µs")

print()
print("  At decode batch=1: all communications are tiny (1 token input).")
print("  TP communication time << HBM weight loading time → NVLink hides it.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: KV cache P2P scatter/gather for paged attention
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — KV Cache P2P: Paged Attention Scatter Across GPUs")
print("━" * 68)
print()

print("  KV cache distributed across GPUs for long-context inference:")
print("  Each GPU holds 1/N of the total KV context.")
print("  Attention on GPU i needs KV from all other GPUs → AllGather.")
print()

KV_BYTES_PER_TOKEN = 2 * N_KV_HEADS * HEAD_DIM * N_LAYERS * FEAT_BYTES

print(f"  Llama-2-7B: {KV_BYTES_PER_TOKEN/1024:.1f} KB per token (both K and V, all layers)")
print()
print(f"  {'Context':>10}  {'KV size (GB)':>14}  {'N_GPUs':>8}  "
      f"{'KV/GPU (GB)':>12}  {'AllGather ms':>14}  {'H100 HBM ms'}")
print("  " + "─" * 72)

HBM_BW_GBS = H100_HBM_GBS

for n_ctx in [4096, 8192, 16384, 32768, 65536, 131072]:
    total_kv_bytes = KV_BYTES_PER_TOKEN * n_ctx
    for n_gpus in [1, 4, 8]:
        if n_gpus == 1 and n_ctx > 8192:
            continue   # won't fit
        kv_per_gpu   = total_kv_bytes / n_gpus

        if n_gpus > 1:
            t_ag_ms  = allgather_ms(n_gpus, total_kv_bytes, NVL4_GBS)
        else:
            t_ag_ms  = 0.0

        # HBM time: read full KV cache from local HBM
        t_hbm_ms = kv_per_gpu / (HBM_BW_GBS * 1e9) * 1000

        print(f"  {n_ctx:>10,}  {total_kv_bytes/1e9:>14.3f}  {n_gpus:>8}  "
              f"{kv_per_gpu/1e9:>12.3f}  {t_ag_ms:>14.4f}  {t_hbm_ms:>11.3f}")

print()
print("  AllGather for KV < HBM read time → NVLink fully hides KV distribution.")
print("  For extremely long contexts: distributing KV across GPUs helps HBM bandwidth.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Pipeline parallelism activation transfer
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Pipeline Parallelism: Activation Transfer Between Stages")
print("━" * 68)
print()

print("  Pipeline parallelism: different layer groups on different GPUs.")
print("  Activations must be sent between pipeline stages via P2P.")
print("  Micro-batch pipelining fills the pipeline bubble.")
print()

MICRO_BATCH = 8
SEQ_PP      = 2048

# Activation at each layer boundary: [micro_batch, seq, d_model] BF16
activation_bytes = MICRO_BATCH * SEQ_PP * D_MODEL * FEAT_BYTES

print(f"  Activation tensor: {MICRO_BATCH}×{SEQ_PP}×{D_MODEL} BF16 = "
      f"{activation_bytes/1e6:.1f} MB per pipeline boundary")
print()

pipeline_configs = [
    (2,  8, NVL4_GBS,  "2-stage PP within node, NVLink4"),
    (4,  4, NVL4_GBS,  "4-stage PP within node, NVLink4"),
    (8,  2, NVL4_GBS,  "8-stage PP within node, NVLink4"),
    (2,  8, 50.0,      "2-stage PP inter-node, IB NDR-400"),
    (4,  4, 50.0,      "4-stage PP inter-node, IB NDR-400"),
]

print(f"  {'Config':<44}  {'Layers/stage':>14}  {'Transfer time':>14}  "
      f"{'Bubble fraction'}")
print("  " + "─" * 82)

for pp_degree, layers_per_stage, bw, desc in pipeline_configs:
    t_ms  = activation_bytes / (bw * 1e9) * 1000

    # Bubble fraction: idle time at pipeline boundaries
    # For N_micro microbatches: bubble = (pp_degree-1) / (pp_degree-1 + N_micro)
    N_micro = 4   # typical micro-batch count
    bubble  = (pp_degree - 1) / (pp_degree - 1 + N_micro) * 100

    print(f"  {desc:<44}  {layers_per_stage:>14}  "
          f"{t_ms:>12.3f}ms  {bubble:>8.1f}%")

print()
print("  NVLink4: inter-stage transfer < 1 ms → compute completely hides it.")
print("  IB NDR-400: ~40-200 ms for cross-node stages → pipeline bubble visible.")
print("  Pipeline parallelism across nodes requires large micro-batch counts.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: End-to-end multi-GPU bandwidth budget for LLM inference
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Full Inference Bandwidth Budget: 8×H100 DGX")
print("━" * 68)
print()

TP = 8          # tensor parallel degree = 8 GPUs
N_CTX = 4096   # context length
DECODE_BATCH = 1

print(f"  Config: TP={TP}, context={N_CTX}, decode batch={DECODE_BATCH}")
print(f"  All on H100 DGX (NVLink4, 900 GB/s bidirectional)")
print()

components = []

# 1. Weight loading from HBM (per layer, per token)
weight_per_layer = (D_MODEL * D_MODEL * 4 + D_MODEL * 11008 * 3) * FEAT_BYTES / TP
t_weights_ms = weight_per_layer * N_LAYERS / (H100_HBM_GBS * 1e9 / TP) * 1000
components.append(("Weight loading (HBM)", t_weights_ms, "H100 HBM"))

# 2. KV cache read (per decode step)
kv_bytes = KV_BYTES_PER_TOKEN * N_CTX * DECODE_BATCH / TP
t_kv_ms  = kv_bytes / (H100_HBM_GBS * 1e9) * 1000
components.append(("KV cache read (HBM)", t_kv_ms, "H100 HBM"))

# 3. TP AllGather per layer (attention input)
attn_in = DECODE_BATCH * 1 * D_MODEL * FEAT_BYTES
t_ag_ms  = allgather_ms(TP, attn_in, NVL4_GBS) * N_LAYERS
components.append(("TP AllGather (attention, all L)", t_ag_ms * 1000, "NVLink4 µs"))

# 4. TP ReduceScatter per layer (after projection)
t_rs_ms  = reduce_scatter_ms(TP, attn_in, NVL4_GBS) * N_LAYERS
components.append(("TP RS (attention output, all L)", t_rs_ms * 1000, "NVLink4 µs"))

print(f"  {'Component':<40}  {'Time':>12}  {'Engine'}")
print("  " + "─" * 58)

total_ms = 0.0
for name, t, engine in components:
    if "µs" in engine:
        t_str = f"{t:.2f} µs"
        total_ms += t / 1000
    else:
        t_str = f"{t:.2f} ms"
        total_ms += t
    print(f"  {name:<40}  {t_str:>12}  {engine}")

print()
print(f"  Total estimated decode step: ~{total_ms:.1f} ms")
print(f"  TP communication: {(t_ag_ms+t_rs_ms)/total_ms*100:.1f}% of total step time")
print(f"  (NVLink4 hides TP communication under HBM weight loading.)")
''',
    },

    "5 · Multi-GPU Topology Query — nvidia-smi Matrix & Performance Impact": {
        "description": (
            "Simulate the nvidia-smi topology matrix output for different GPU "
            "configurations. Show how to interpret NV#, PIX, PHB, and SYS codes. "
            "Model the performance impact of topology: two GPUs on different NUMA "
            "domains communicating. Show NCCL's topology-aware ring construction. "
            "Compute the penalty for GPU-NIC misalignment in GPUDirect RDMA. "
            "Provide a complete topology diagnosis checklist."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  MULTI-GPU TOPOLOGY QUERY — nvidia-smi Matrix & Performance Impact")
print("=" * 68)
print()

np.random.seed(42)

# Topology codes and their bandwidth implications
TOPO_CODES = {
    "NV1":  {"desc": "1 NVLink",         "bw_gbs": 25.0,  "lat_us": 1.0},
    "NV2":  {"desc": "2 NVLinks",        "bw_gbs": 50.0,  "lat_us": 1.0},
    "NV4":  {"desc": "4 NVLinks",        "bw_gbs": 100.0, "lat_us": 1.0},
    "NV6":  {"desc": "6 NVLinks (A100)",  "bw_gbs": 150.0, "lat_us": 0.9},
    "NV12": {"desc": "12 NVLinks (A100)", "bw_gbs": 300.0, "lat_us": 0.8},
    "NV18": {"desc": "18 NVLinks (H100)", "bw_gbs": 450.0, "lat_us": 0.7},
    "PIX":  {"desc": "PCIe same switch",  "bw_gbs": 32.0,  "lat_us": 3.0},
    "PXB":  {"desc": "PCIe cross bridge", "bw_gbs": 32.0,  "lat_us": 4.0},
    "PHB":  {"desc": "PCIe same host br.","bw_gbs": 32.0,  "lat_us": 5.0},
    "SYS":  {"desc": "Cross NUMA/QPI",   "bw_gbs": 15.0,  "lat_us": 10.0},
    "MIG":  {"desc": "Same MIG instance", "bw_gbs": 900.0, "lat_us": 0.3},
}


def make_topology_matrix(config_name, n_gpus, connections):
    """
    Create a topology matrix for a GPU configuration.
    connections: dict of (i,j) -> code (symmetric).
    """
    matrix = [["—" if i == j else "SYS" for j in range(n_gpus)]
               for i in range(n_gpus)]
    for (i, j), code in connections.items():
        matrix[i][j] = code
        matrix[j][i] = code
    return matrix


def print_topology_matrix(name, matrix, n_gpus):
    n = min(n_gpus, 8)
    print(f"  {name}  ({n_gpus} GPUs)")
    print(f"  nvidia-smi topo --matrix:")
    print()
    header = "        "
    for j in range(n):
        header += f"  GPU{j}"
    print(header)
    for i in range(n):
        row = f"  GPU{i}  "
        for j in range(n):
            code = matrix[i][j]
            row += f"  {code:>4}"
        print(row)
    print()


def topology_bandwidth(matrix, i, j):
    code = matrix[i][j]
    return TOPO_CODES.get(code, TOPO_CODES["SYS"])["bw_gbs"]

def topology_latency(matrix, i, j):
    code = matrix[i][j]
    return TOPO_CODES.get(code, TOPO_CODES["SYS"])["lat_us"]


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Topology matrices for different configurations
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — nvidia-smi Topology Matrices for Common Configurations")
print("━" * 68)
print()

# DGX A100: all GPUs via NVSwitch (show as NV12 — through switch)
dgx_a100_connections = {(i,j): "NV12" for i in range(8) for j in range(8) if i != j}
dgx_a100_matrix = make_topology_matrix("DGX A100", 8, dgx_a100_connections)
print_topology_matrix("DGX A100 (NVLink3, NVSwitch)", dgx_a100_matrix, 8)

# PCIe server: 2 root complexes
pcie_connections = {}
for i in range(4):
    for j in range(4):
        if i != j:
            pcie_connections[(i,j)] = "PIX"   # same root complex
for i in range(4):
    for j in range(4, 8):
        pcie_connections[(i,j)] = "SYS"   # different NUMA

pcie_matrix = make_topology_matrix("PCIe server (no NVLink)", 8, pcie_connections)
print_topology_matrix("PCIe server (2 root complexes)", pcie_matrix, 8)

# Partial NVLink (V100 server with 4 NVLink connections)
v100_connections = {}
# Pairs: 0-1, 2-3, 4-5, 6-7 share NVLink within pairs; cross-pair via PCIe
for i in range(0, 8, 2):
    v100_connections[(i, i+1)] = "NV2"
for i in range(8):
    for j in range(8):
        if i != j and (i, j) not in v100_connections and (j, i) not in v100_connections:
            v100_connections[(i,j)] = "PHB"
v100_matrix = make_topology_matrix("V100 (NVLink pairs)", 8, v100_connections)
print_topology_matrix("V100 server (NVLink pairs only)", v100_matrix, 8)


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Topology code meaning and bandwidth impact
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Topology Code Reference: Bandwidth & Latency")
print("━" * 68)
print()

print(f"  {'Code':<8}  {'Meaning':<28}  {'BW (GB/s)':>10}  "
      f"{'Latency (µs)':>14}  {'Suitable for'}")
print("  " + "─" * 76)

USE_CASES = {
    "NV1":  "Small P2P transfers",
    "NV2":  "V100 standard P2P",
    "NV4":  "Good P2P bandwidth",
    "NV6":  "A100 4-GPU subgraph",
    "NV12": "DGX A100 full mesh",
    "NV18": "DGX H100 full mesh",
    "PIX":  "Adequate for workloads",
    "PXB":  "Acceptable if no NVLink",
    "PHB":  "Suboptimal, avoid",
    "SYS":  "⚠ NUMA penalty, slow",
    "MIG":  "MIG instance sharing",
}

for code, info in TOPO_CODES.items():
    use = USE_CASES.get(code, "")
    print(f"  {code:<8}  {info['desc']:<28}  {info['bw_gbs']:>10.1f}  "
          f"{info['lat_us']:>14.1f}  {use}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: NCCL ring construction from topology
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — NCCL Ring Construction from Topology")
print("━" * 68)
print()

print("  NCCL builds its communication rings based on the topology matrix.")
print("  Goal: maximise bandwidth by preferring NVLink links in the ring.")
print()

topologies = [
    ("DGX A100 (all NVLink)",        dgx_a100_matrix, 8),
    ("V100 (NVLink pairs)",           v100_matrix,     8),
    ("PCIe server (no NVLink)",       pcie_matrix,     8),
]

for topo_name, matrix, n in topologies:
    # Simple greedy ring construction: prefer highest bandwidth links
    ring = [0]
    remaining = list(range(1, n))

    while remaining:
        last = ring[-1]
        # Choose the neighbour with the highest bandwidth
        best = max(remaining, key=lambda j: topology_bandwidth(matrix, last, j))
        ring.append(best)
        remaining.remove(best)
    ring.append(ring[0])  # close the ring

    # Compute ring bandwidth = minimum link bandwidth in the ring
    ring_bws = [topology_bandwidth(matrix, ring[i], ring[i+1])
                for i in range(len(ring)-1)]
    ring_lats = [topology_latency(matrix, ring[i], ring[i+1])
                 for i in range(len(ring)-1)]
    min_bw  = min(ring_bws)
    max_lat = max(ring_lats)

    ring_str = " → ".join(str(g) for g in ring[:-1]) + " → 0"
    print(f"  {topo_name}:")
    print(f"    Ring: {ring_str}")
    print(f"    Bottleneck link BW: {min_bw:.0f} GB/s (weakest link)")
    print(f"    Max latency in ring: {max_lat:.1f} µs")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Topology diagnosis checklist
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Topology Diagnosis Checklist")
print("━" * 68)
print()

checklist = [
    ("1. Run nvidia-smi topo --matrix",
     "Verify all GPU pairs show NV# (not SYS or PHB) for multi-GPU training.",
     "SYS between training GPUs: severe bandwidth penalty (~15 GB/s vs 450 GB/s)."),
    ("2. Check GPU-NIC affinity",
     "Run: nvidia-smi topo --matrix with NICs shown. GPU and NIC should share root.",
     "Mismatched GPU-NIC: GPUDirect RDMA bandwidth drops 30-50%."),
    ("3. Verify P2P access",
     "Run p2pBandwidthLatencyTest (CUDA samples). Check 'Unidirectional P2P=Enabled'.",
     "Disabled P2P: communication routed through CPU, 2× slower."),
    ("4. Check NCCL algorithm",
     "NCCL_DEBUG=INFO shows: [0] NCCL INFO Ring 0: 0→1→...→0 via NVL",
     "Via NET instead of NVL: not using NVLink. Fix P2P or topology."),
    ("5. Measure NCCL bus bandwidth",
     "Run nccl-tests: ./all_reduce_perf -b 1G -e 1G -f 2 -g 8",
     "Low busbw vs NVLink peak: check NCCL_ALGO, NCCL_NTHREADS settings."),
    ("6. Verify GPUDirect RDMA",
     "Run ib_write_bw --use_cuda (OFED tools). Should achieve ~22-25 GB/s.",
     "Low bandwidth: check ibdev2netdev, GDR_LEVEL, PCIe root placement."),
    ("7. Profile with Nsight Systems",
     "Look for NVLink traffic and DMA overlap on GPU timeline.",
     "All-sequential (no overlap): missing stream or event dependency."),
]

for step, action, red_flag in checklist:
    print(f"  {step}")
    print(f"    Action:    {action}")
    print(f"    Red flag:  {red_flag}")
    print()
''',
    },
}

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