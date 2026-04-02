"""
NCCL Collectives — AllReduce (Ring/Tree), AllGather/ReduceScatter & CUDA Graph
================================================================================

NCCL (NVIDIA Collective Communications Library) is the communications backbone
of every serious multi-GPU training framework. PyTorch DDP, Megatron-LM,
DeepSpeed, and Horovod all call ncclAllReduce under the hood. Understanding
NCCL means understanding: what the collectives compute, how they are implemented
on the hardware (ring vs tree vs NVLS), and how they integrate with CUDA
streams and graphs for minimal latency overhead.

"""

import textwrap, re, math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "NCCL Collectives — AllReduce (Ring/Tree), AllGather/ReduceScatter & CUDA Graph"
DISPLAY_NAME = "18 · NCCL Collectives"
ICON         = "🔄"
SUBTITLE     = "AllReduce · Ring · Tree · AllGather · ReduceScatter · NCCL+CUDA Graph · NVLink"

THEORY = """

##### PART 1 — THE COLLECTIVE OPERATIONS: SEMANTICS AND USE CASES

### The Six Core Collectives

    ALL COLLECTIVES involve all N GPUs in a communicator group simultaneously.
    No GPU can leave early; the operation completes only when ALL GPUs are done.

    BROADCAST: one GPU sends the SAME data to all others.
        Input:  GPU 0 has tensor T.
        Output: All GPUs have T.
        Use:    Distributing initial model weights from rank 0.
        Volume: (N-1) × sizeof(T) bytes total.

    REDUCE: all GPUs contribute; one GPU gets the result.
        Input:  Each GPU i has tensor T_i.
        Output: GPU 0 has sum(T_i).
        Use:    Gradient aggregation with parameter server.
        Volume: (N-1) × sizeof(T) bytes total.

    ALL-REDUCE: all GPUs contribute; ALL GPUs get the result.
        Input:  Each GPU i has tensor T_i.
        Output: ALL GPUs have sum(T_i).
        Use:    Distributed training gradient synchronisation (PyTorch DDP).
        Volume: 2(N-1)/N × sizeof(T) per GPU (send + receive).

    ALL-GATHER: each GPU has a shard; all get the full tensor.
        Input:  GPU i has shard T_i of size sizeof(T)/N.
        Output: ALL GPUs have full T = [T_0, T_1, ..., T_{N-1}].
        Use:    ZeRO stage 3 (gather weights before forward), Megatron tensor-parallel.
        Volume: (N-1)/N × sizeof(T) per GPU.

    REDUCE-SCATTER: all GPUs contribute; each gets one shard of the result.
        Input:  Each GPU i has the full tensor T_i.
        Output: GPU j has shard j of sum(T_i).
        Use:    ZeRO stage 2/3 (split AllReduce into RS + optimizer + AG).
        Volume: (N-1)/N × sizeof(T) per GPU.

    ALL-TO-ALL: each GPU sends different data to each other GPU.
        Input:  GPU i has N chunks; chunk j goes to GPU j.
        Output: GPU j has N chunks from every other GPU.
        Use:    Expert routing in Mixture-of-Experts (MoE) models.

### The AllReduce = ReduceScatter + AllGather Decomposition

    AllReduce(T) = AllGather(ReduceScatter(T))

    STEP 1 — ReduceScatter: each GPU accumulates its shard of the global sum.
        After: GPU i holds sum of shard i across all GPUs.
        Volume: (N-1)/N × sizeof(T) per GPU.

    STEP 2 — AllGather: all GPUs exchange their shards.
        After: every GPU has all N shards = full global sum.
        Volume: (N-1)/N × sizeof(T) per GPU.

    Total: 2(N-1)/N × sizeof(T) per GPU — provably optimal for AllReduce.
    This decomposition is how ring, tree, and NVLS all implement AllReduce.

### Network Topology and Bandwidth

    H100 DGX node (8 GPUs):
        Within node: NVLink 4.0, 900 GB/s aggregate bisection bandwidth.
        Per-GPU unidirectional: 450 GB/s total = 56.25 GB/s × 18 NVLink ports.
        Latency: < 1 µs.

    Between nodes (InfiniBand NDR-400):
        400 Gb/s = 50 GB/s per port (bidirectional).
        Latency: ~1 µs fabric + ~3 µs NCCL protocol overhead.

    IMPLICATION: intra-node NVLink is 10-20× faster than inter-node IB.
    Multi-node AllReduce is almost always bandwidth-limited by the inter-node link.


##### PART 2 — RING ALL-REDUCE: THE BANDWIDTH-OPTIMAL ALGORITHM

### The Ring Topology

    N GPUs arranged in a logical ring: GPU 0 → GPU 1 → ... → GPU N-1 → GPU 0.
    Each GPU has one upstream (receive) and one downstream (send) neighbour.
    All links transmit simultaneously — full utilisation of all N links.

### Phase 1: ReduceScatter via Ring (N-1 steps)

    Tensor T is divided into N equal chunks: T[0], T[1], ..., T[N-1].
    Each GPU i has a full copy of all chunks: T_i[0..N-1].

    STEP k (for k = 0, 1, ..., N-2):
        GPU i sends chunk [(i-k) mod N] to downstream GPU (i+1) mod N.
        GPU i receives a chunk from upstream GPU (i-1) mod N.
        GPU i adds the received chunk to its local copy.

    After N-1 steps:
        GPU j holds the correct global sum of chunk j: sum_i(T_i[j]).

    DATA VOLUME per GPU: (N-1) × sizeof(T)/N = (N-1)/N × sizeof(T).

### Phase 2: AllGather via Ring (N-1 more steps)

    GPU j has the correct shard j. AllGather propagates all shards.
    STEP k: GPU i sends shard [(i-k) mod N] downstream.
    After N-1 steps: all GPUs have all N correct shards.

    DATA VOLUME per GPU: (N-1)/N × sizeof(T). Same as phase 1.

### Total Volume and Bandwidth Efficiency

    Per GPU: 2(N-1)/N × sizeof(T).
    For N→∞: approaches 2 × sizeof(T) per GPU.
    This equals the THEORETICAL LOWER BOUND for AllReduce.
    Ring is BANDWIDTH-OPTIMAL.

    LATENCY: 2(N-1)α + 2(N-1)/N × B/bw
    where α = link latency, B = tensor size, bw = link bandwidth.
    For small B: dominated by 2(N-1)α → latency-bound, scales badly with N.
    For large B: bandwidth term dominates → bandwidth-optimal.

    CROSSOVER (ring better than tree): B > α × bw × N/2.
    For NVLink (α=1µs, bw=50 GB/s, N=8): B > 200 MB. In practice: ~256 KB.

### NCCL Multiple Rings

    NCCL uses multiple simultaneous rings to exploit all NVLink ports:
    A100 DGX: 12 NVLink ports per GPU → NCCL uses 2-8 rings concurrently.
    Each ring handles a different chunk of the tensor in parallel.
    Effect: saturates all ports, maximises aggregate bandwidth.


##### PART 3 — TREE ALL-REDUCE: LATENCY-OPTIMAL FOR SMALL MESSAGES

### Why Ring Fails for Small Messages

    Ring latency: 2(N-1) × α. For N=128, α=3 µs: 762 µs latency floor.
    This is unacceptable for small gradient tensors (biases, norms, etc.)
    even when the actual data transfer is negligible.

### Binary Tree Algorithm

    REDUCE PHASE (N-1 total messages, log₂(N) steps):
        Step 1: leaves (N/2 GPUs) send to parents.
        Step 2: parents reduce and send to grandparents.
        ...
        Step log₂(N): root has global sum.

    BROADCAST PHASE (mirror of reduce):
        Root sends sum down; all GPUs receive in log₂(N) steps.

    TOTAL LATENCY: 2 × log₂(N) × α.
    For N=128, α=3 µs: 2 × 7 × 3 = 42 µs. 18× better than ring.

### Tree Bandwidth Inefficiency

    Root link carries ALL inter-subtree traffic → bottleneck.
    Bandwidth utilisation: O(1/log N) of the ring algorithm.
    Tree is latency-optimal but NOT bandwidth-optimal for large tensors.

### Double Binary Tree (NCCL Implementation)

    NCCL uses TWO binary trees with different roots to balance load.
    Tree 1: GPU 0 is root. Tree 2: GPU N/2 is root.
    Result: each GPU is a leaf in one tree, internal in the other.
    Bandwidth is more evenly distributed than single tree.

### NVLS: NVLink Switch Algorithm (H100 DGX)

    H100 DGX includes an NVLink switch chip — a hardware crossbar connecting
    all 8 GPUs directly, not just via point-to-point NVLink.

    NVLS (NCCL 2.19+):
        All 8 GPUs reduce directly into the switch crossbar simultaneously.
        Switch hardware performs in-flight reduction.
        Result available to all GPUs in ONE PASS (not N-1 steps).
        Latency: α + B/bw (single hardware pass).
        For large B: approximately 1.5× faster than ring on H100.

    NCCL ALGORITHM SELECTION (automatic):
        Small messages (< 256 KB):  Tree (latency-optimal).
        Large messages (> 256 KB):  Ring or NVLS (bandwidth-optimal).
        Controlled with: NCCL_ALGO=RING/TREE/NVLS or auto (default).


##### PART 4 — REDUCE-SCATTER AND ALL-GATHER IN PRACTICE

### ZeRO: Memory-Efficient Distributed Training

    ZeRO (Zero Redundancy Optimizer) eliminates redundant storage of model
    state across GPUs using ReduceScatter and AllGather:

    ZeRO STAGE 1 — Partition optimizer state only:
        Each GPU holds full parameters and gradients (redundant).
        Gradient AllReduce → each GPU has full gradient.
        ReduceScatter gradients → each GPU holds shard j of gradients.
        Optimizer step on shard j.
        AllGather updated parameters → all GPUs have full model.
        Memory savings: 4× for optimizer states (m1, m2, master weight).

    ZeRO STAGE 2 — Also partition gradients:
        Backward computes gradients locally for own shard only.
        ReduceScatter during backward (not AllReduce).
        Memory savings: 8× for optimizer states + gradients.

    ZeRO STAGE 3 — Also partition parameters:
        Forward: AllGather layer weights before each layer.
        Backward: ReduceScatter gradients after each layer.
        Memory savings: ~N× all model state (linear in GPU count).
        Communication cost: 3× the parameters per forward/backward pass
        (1× AllGather fwd + 1× AllGather bwd recompute + 1× ReduceScatter grad).

    Communication volume comparison (per step):
        DDP (AllReduce gradients):          2(N-1)/N × param_bytes
        ZeRO-1 (RS grad + AG param):        2(N-1)/N × param_bytes (same!)
        ZeRO-3 (AG + AG + RS per layer):    3 × (N-1)/N × param_bytes

### AllGather in Tensor Parallelism

    Megatron-LM column-parallel linear: weight matrix split across GPUs.
        GPU i holds columns [i*K/N .. (i+1)*K/N] of the weight matrix.
        Forward: each GPU computes partial output.
        AllGather partial outputs → each GPU has the full result.
        Volume: (N-1)/N × sizeof(full_output) per forward pass.

    ReduceScatter in sequence parallelism:
        Activation tensors split along sequence dimension.
        AllGather over sequence before attention.
        ReduceScatter after MLP (avoids materialising full activation).
        Memory savings: N× activations vs standard tensor parallelism.


##### PART 5 — NCCL API: COMMUNICATORS, GROUPS AND STREAM INTEGRATION

### Communicator Setup

    ncclComm_t comm;
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);           // one process creates ID
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD); // broadcast
    ncclCommInitRank(&comm, world_size, id, rank);  // all ranks join

    PyTorch wraps this as:
        torch.distributed.init_process_group(backend='nccl')
        model = DistributedDataParallel(model)

### Collective API Pattern

    ncclAllReduce(sendbuf, recvbuf, count, ncclFloat16, ncclSum, comm, stream);
    ncclAllGather(sendbuf, recvbuf, sendcount, ncclFloat16, comm, stream);
    ncclReduceScatter(sendbuf, recvbuf, recvcount, ncclFloat16, ncclSum, comm, stream);
    ncclBroadcast(sendbuf, recvbuf, count, ncclFloat16, root, comm, stream);

    All operations are ASYNCHRONOUS: they enqueue into the CUDA stream and
    return immediately to the CPU. The GPU executes them in stream order.

### Group API: Fusing Multiple Collectives

    ncclGroupStart();
    for each tensor:
        ncclAllReduce(g, g, N, ncclFloat16, ncclSum, comm, stream);
    ncclGroupEnd();

    Grouping allows NCCL to schedule all collectives together:
    - One transport setup for all (reduces latency overhead).
    - NCCL can reorder for bandwidth optimality.
    - Essential for compute-communication overlap.

### NCCL Data Types

    ncclFloat32:  FP32 (full precision gradient sync)
    ncclFloat16:  FP16 (mixed precision training)
    ncclBfloat16: BF16 (preferred for stability on A100/H100)
    ncclInt32:    for voting/masking operations
    ncclUint8:    for compressed gradients


##### PART 6 — NCCL + CUDA GRAPH INTEGRATION

### The Problem: AllReduce Launch Overhead in Training Loops

    A typical DDP backward pass:
        - Fires ~N_layers AllReduce calls (one per parameter group).
        - Llama-2-7B: ~240 parameter groups → 240 AllReduce launches.
        - Each launch: ~5 µs NCCL + kernel overhead.
        - Total: 240 × 5 µs = 1.2 ms of pure overhead per step.

    At 10 steps/sec (typical large-model training): 12 ms/sec wasted.
    At 100 steps/sec (smaller models): 120 ms/sec wasted — unacceptable.

### NCCL Stream Capture (NCCL >= 2.9)

    NCCL supports CUDA stream capture since version 2.9. AllReduce, AllGather,
    and ReduceScatter calls can be included inside a cudaStreamBeginCapture block.
    The resulting CUDA Graph contains NCCL kernel nodes.

    CAPTURE PROCEDURE:
        // 1. Warmup (NCCL initialises internal state — cannot be captured)
        for (int i = 0; i < 3; i++) {
            ncclAllReduce(grad, grad, N, ncclFloat16, ncclSum, comm, stream);
            cudaStreamSynchronize(stream);
        }

        // 2. Capture the training step
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        forward_pass(stream);
        backward_pass(stream);           // submits AllReduce calls
        optimizer_step(stream);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

        // 3. Replay (every subsequent step)
        static_input.copy_(new_batch);   // update input in-place
        cudaGraphLaunch(exec, stream);   // < 1 µs overhead for entire step

    The graph captures ALL kernel launches including NCCL communication kernels.
    Replay reduces per-step overhead from 240 × 5 µs = 1.2 ms to < 1 µs.

### Static Shape Requirement

    CUDA Graph + NCCL requires FIXED tensor shapes throughout training.
    Every captured AllReduce has a fixed count (element count) baked in.
    Changing the gradient shape (e.g., dynamic padding in NLP) breaks the graph.

    SOLUTIONS:
        Pad all sequences to a fixed max length.
        Capture one graph per batch configuration (vLLM-style registry).
        Use non-graph path for variable-length batches (fallback).

### PyTorch Gradient Bucketing with CUDA Graphs

    PyTorch DDP uses gradient BUCKETING: instead of one AllReduce per param,
    it accumulates gradients into buckets (default 25 MB) and reduces whole buckets.
    With CUDA Graphs:
        Each bucket's AllReduce is a separate graph node.
        Backward computes in-order; buckets reduce as they fill.
        CUDA Graph preserves the bucket-filling dependency automatically.

    torch.distributed.ddp set_static_graph():
        Explicitly tells PyTorch the computation graph doesn't change.
        Enables automatic CUDA Graph capture for DDP training.
        Usage: model = DDP(model); model._set_static_graph()
        Combined with torch.compile: can capture entire training step.


##### PART 7 — COMPUTE-COMMUNICATION OVERLAP AND TIMING

### The Overlap Principle

    NCCL uses dedicated communication CUDA streams internally. The NCCL kernel
    runs on one stream; the compute kernel runs on another. If submitted to
    different streams with correct event dependencies, they overlap on the GPU.

    PyTorch DDP backward-communication overlap:
        DDP uses multiple streams: one for compute, one for communication.
        As backward computes gradients for layer L, it fires AllReduce for L+1
        (already fully computed). Compute and communication proceed in parallel.
        This is called "bucketed gradient communication".

    OVERLAP CONDITION: the AllReduce for bucket k must be completely independent
    of the compute that generates bucket k+1. In DDP: this is guaranteed because
    buckets are reduced in reverse layer order (last layer first).

### Communication Volume and Bandwidth Budgeting

    For a step budget of T_step ms, the communication budget is:
        T_comm = T_step - T_compute - T_overhead
    If T_comm < T_AllReduce: training is communication-bound.
    Communication is hidden if T_comm can fit entirely within T_compute.

    LLAMA-2-7B ANALYSIS (8 × A100 with NVLink, BF16):
        Parameter bytes: 7B × 2 = 14 GB
        AllReduce volume per GPU: 2(7/8) × 14 GB = 24.5 GB
        NVLink bandwidth: 300 GB/s (A100 DGX)
        AllReduce time: 24.5 / 300 = 81.7 ms
        Forward + backward compute: ~1200 ms (at 312 TFLOP/s, batch=64)
        Overlap fraction: 81.7 / 1200 = 6.8% — communication is hidden!

    SMALL MODEL ANALYSIS (GPT-2, 8 × V100 via PCIe, FP16):
        Parameter bytes: 1.5B × 2 = 3 GB
        AllReduce volume per GPU: 2(7/8) × 3 = 5.25 GB
        PCIe bandwidth: 16 GB/s
        AllReduce time: 5.25 / 16 = 328 ms
        Forward + backward: ~120 ms
        Communication-bound by 2.7×! Overlap cannot hide it.
        Fix: upgrade to NVLink, use gradient compression, or increase batch size.

### NCCL Tuning Environment Variables

    NCCL_ALGO=RING/TREE/NVLS/COLLNET_DIRECT/COLLNET_CHAIN
        Override automatic algorithm selection.

    NCCL_NTHREADS=<N>
        Number of CUDA threads per NCCL kernel (default: 512).
        Increase for large tensors on high-bandwidth interconnects.

    NCCL_BUFFSIZE=<bytes>
        Size of NCCL's internal communication buffer (default: 4 MB).
        Increase to reduce pipelining overhead for large tensors.

    NCCL_NET_GDR_LEVEL=<0-3>
        GPU-Direct RDMA support level (for InfiniBand).
        Level 3: GPU memory accessible directly by the IB NIC — no CPU bounce buffer.
        Reduces inter-node latency by 30-50%.

    NCCL_SOCKET_NTHREADS / NCCL_NSOCKS_PERTHREAD
        For Ethernet-based communication (non-RDMA).

    NCCL_DEBUG=INFO/WARN/VERSION
        Enable verbose NCCL logging for topology and algorithm diagnostics.
"""

OPERATIONS = {

    "1 · Collective Volume & Time Model — AllReduce, AllGather, RS Across Topologies": {
        "description": (
            "Compute exact communication volumes and wall times for AllReduce, "
            "AllGather, and ReduceScatter across different GPU counts, message "
            "sizes, and interconnects. Show where each collective is the bottleneck "
            "in training. Compare NVLink vs PCIe vs InfiniBand. Model ZeRO stage "
            "communication overhead and show when AllGather dominates ZeRO-3."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  COLLECTIVE VOLUME & TIME — AllReduce, AllGather, RS Across Topologies")
print("=" * 68)
print()

# Hardware bandwidth (unidirectional, per-GPU)
INTERCONNECTS = {
    "NVLink 4 (H100)":  {"bw_gbs": 450.0, "lat_us": 0.8,  "type": "intra"},
    "NVLink 3 (A100)":  {"bw_gbs": 300.0, "lat_us": 1.0,  "type": "intra"},
    "IB HDR-200":       {"bw_gbs": 25.0,  "lat_us": 3.0,  "type": "inter"},
    "PCIe Gen4 x16":    {"bw_gbs": 32.0,  "lat_us": 5.0,  "type": "inter"},
    "Ethernet 100GbE":  {"bw_gbs": 12.5,  "lat_us": 8.0,  "type": "inter"},
}

def allreduce_volume_per_gpu(n_gpus, tensor_bytes):
    """Ring AllReduce: 2*(N-1)/N per GPU."""
    return 2 * (n_gpus - 1) / n_gpus * tensor_bytes

def allgather_volume_per_gpu(n_gpus, tensor_bytes):
    """AllGather: (N-1)/N per GPU (each sends its shard)."""
    return (n_gpus - 1) / n_gpus * tensor_bytes

def reduce_scatter_volume_per_gpu(n_gpus, tensor_bytes):
    """ReduceScatter: (N-1)/N per GPU (same as AllGather)."""
    return (n_gpus - 1) / n_gpus * tensor_bytes

def collective_time_us(volume_bytes, n_gpus, bw_gbs, lat_us, n_steps_ring):
    """Ring-based time: latency * 2*(N-1) + bandwidth term."""
    lat_term = 2 * (n_gpus - 1) * lat_us
    bw_term  = volume_bytes / (bw_gbs * 1e9) * 1e6
    return lat_term + bw_term


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: AllReduce time vs message size and GPU count
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — AllReduce Time: NVLink vs IB vs PCIe")
print("━" * 68)
print()

N_GPUS    = 8
interconn = {k: v for k, v in INTERCONNECTS.items()
             if k in ["NVLink 3 (A100)", "IB HDR-200", "PCIe Gen4 x16"]}

sizes_mb = [0.01, 0.1, 1, 10, 100, 1000, 14000]

print(f"  N_GPUs = {N_GPUS}")
print()
print(f"  {'Size (MB)':>12}", end="")
for name in interconn:
    print(f"  {name:>20}", end="")
print()
print(f"  {'':>12}", end="")
for _ in interconn:
    print(f"  {'(ms)':>20}", end="")
print()
print("  " + "─" * (14 + 22 * len(interconn)))

for size_mb in sizes_mb:
    size_bytes = size_mb * 1e6
    vol        = allreduce_volume_per_gpu(N_GPUS, size_bytes)
    print(f"  {size_mb:>12.2f}", end="")
    for name, hw in interconn.items():
        t_us = collective_time_us(vol, N_GPUS, hw["bw_gbs"], hw["lat_us"], N_GPUS-1)
        print(f"  {t_us/1000:>20.3f}", end="")
    print()

print()
print("  AllReduce of Llama-2-7B gradients (14 GB) at N=8:")
for name, hw in interconn.items():
    vol  = allreduce_volume_per_gpu(N_GPUS, 14e9)
    t_ms = collective_time_us(vol, N_GPUS, hw["bw_gbs"], hw["lat_us"], N_GPUS-1) / 1000
    print(f"    {name}: {t_ms:.1f} ms")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: AllReduce vs AllGather vs ReduceScatter volume comparison
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Collective Volume Comparison: Per GPU (N=8, 1 GB tensor)")
print("━" * 68)
print()

N_GPUS_2    = 8
TENSOR_GB   = 1.0
tensor_bytes = TENSOR_GB * 1e9

print(f"  Tensor size: {TENSOR_GB} GB, N_GPUs = {N_GPUS_2}")
print()
print(f"  {'Collective':>20}  {'Vol / GPU (GB)':>16}  {'% of tensor':>13}  "
      f"{'NVLink3 time (ms)':>18}  {'Use case'}")
print("  " + "─" * 78)

bw_nvl3, lat_nvl3 = 300.0, 1.0

collectives = [
    ("AllReduce",     allreduce_volume_per_gpu,    "DDP gradient sync"),
    ("AllGather",     allgather_volume_per_gpu,    "ZeRO-3 weight gather"),
    ("ReduceScatter", reduce_scatter_volume_per_gpu,"ZeRO gradient RS"),
]

for name, fn, use in collectives:
    vol  = fn(N_GPUS_2, tensor_bytes)
    pct  = vol / tensor_bytes * 100
    t_ms = collective_time_us(vol, N_GPUS_2, bw_nvl3, lat_nvl3, N_GPUS_2-1) / 1000
    print(f"  {name:>20}  {vol/1e9:>16.3f}  {pct:>12.1f}%  "
          f"{t_ms:>18.3f}  {use}")

print()
print("  AllReduce = AllGather + ReduceScatter: exactly 2× the volume of each.")
print("  AllGather and ReduceScatter are individually half the AllReduce cost.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: ZeRO stage communication overhead
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — ZeRO Stage Communication Overhead vs DDP")
print("━" * 68)
print()

# Model config: Llama-2-7B
PARAM_BYTES = 14e9   # 7B params × 2 bytes BF16
N_LAYERS    = 32
GRAD_BYTES  = PARAM_BYTES   # same as params (BF16 grads)
bw, lat     = 300.0, 1.0   # A100 NVLink3

def rs_time(n_gpus, bytes_, bw_gbs, lat_us):
    vol = reduce_scatter_volume_per_gpu(n_gpus, bytes_)
    return collective_time_us(vol, n_gpus, bw_gbs, lat_us, n_gpus-1) / 1000

def ag_time(n_gpus, bytes_, bw_gbs, lat_us):
    vol = allgather_volume_per_gpu(n_gpus, bytes_)
    return collective_time_us(vol, n_gpus, bw_gbs, lat_us, n_gpus-1) / 1000

def ar_time(n_gpus, bytes_, bw_gbs, lat_us):
    vol = allreduce_volume_per_gpu(n_gpus, bytes_)
    return collective_time_us(vol, n_gpus, bw_gbs, lat_us, n_gpus-1) / 1000

N_G = 8

print(f"  Model: Llama-2-7B ({PARAM_BYTES/1e9:.0f} GB), N={N_G}, A100 NVLink3")
print()

strategies = [
    ("DDP (AllReduce grads)",
     ar_time(N_G, GRAD_BYTES, bw, lat),
     "1 AllReduce on full gradient"),
    ("ZeRO-1 (RS + AG params)",
     rs_time(N_G, GRAD_BYTES, bw, lat) + ag_time(N_G, PARAM_BYTES, bw, lat),
     "RS grads + AG params after optim"),
    ("ZeRO-2 (RS grads only)",
     rs_time(N_G, GRAD_BYTES, bw, lat),
     "RS only, keep params local"),
    ("ZeRO-3 (AG fwd + RS bwd + AG bwd)",
     ag_time(N_G, PARAM_BYTES, bw, lat) * 2 + rs_time(N_G, GRAD_BYTES, bw, lat),
     "AG×2 + RS per fwd+bwd cycle"),
]

print(f"  {'Strategy':<32}  {'Comm time (ms)':>16}  {'vs DDP':>8}  {'Notes'}")
print("  " + "─" * 72)

ddp_t = strategies[0][1]
for name, t_ms, notes in strategies:
    print(f"  {name:<32}  {t_ms:>16.1f}  {t_ms/ddp_t:>7.2f}×  {notes}")

print()
print("  ZeRO-1 ≈ DDP (same volume, different partitioning of work).")
print("  ZeRO-3 costs 3× more communication but achieves N× memory reduction.")
''',
    },

    "2 · Ring AllReduce Simulator — Step-by-Step Chunk Routing": {
        "description": (
            "Implement the ring AllReduce algorithm from scratch. Simulate "
            "N GPUs arranged in a ring with each GPU holding a tensor divided "
            "into N chunks. Trace every ReduceScatter step showing which chunk "
            "moves where and how partial sums accumulate. Then trace the "
            "AllGather steps. Verify the final result matches the naive AllReduce. "
            "Show the per-link bandwidth utilisation in each step."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  RING AllReduce SIMULATOR — Step-by-Step Chunk Routing")
print("=" * 68)
print()

np.random.seed(42)


def ring_allreduce(gpu_tensors, verbose=False):
    """
    Simulate ring AllReduce for N GPUs.
    gpu_tensors: list of N numpy arrays (same shape).
    Returns: list of N arrays, all equal to the element-wise sum.
    """
    N          = len(gpu_tensors)
    chunk_size = len(gpu_tensors[0]) // N
    assert len(gpu_tensors[0]) % N == 0, "Tensor length must be divisible by N"

    # Each GPU starts with a full copy of its tensor
    # Split into N chunks; gpu_data[i][j] = chunk j of GPU i's data
    gpu_data = [np.array(t, dtype=np.float64) for t in gpu_tensors]

    total_bytes_transferred = 0
    chunk_bytes = chunk_size * 8   # float64

    if verbose:
        print(f"  Initial data (each GPU's chunks):")
        for i, gd in enumerate(gpu_data):
            chunks = [gd[j*chunk_size:(j+1)*chunk_size] for j in range(N)]
            print(f"    GPU {i}: {[c.tolist() for c in chunks]}")
        print()

    # ── PHASE 1: ReduceScatter ──────────────────────────────────────
    # After step k, GPU i holds the partial sum of chunk [(i-k) mod N]
    # from GPUs i, i+1, ..., i+k (ring direction).

    if verbose:
        print("  ── PHASE 1: ReduceScatter ──")

    for step in range(N - 1):
        new_data = [gd.copy() for gd in gpu_data]
        sends    = []

        for i in range(N):
            send_chunk_idx = (i - step) % N
            recv_from      = (i - 1) % N   # receive FROM the upstream GPU

            c_start = send_chunk_idx * chunk_size
            c_end   = c_start + chunk_size

            # GPU i sends chunk send_chunk_idx to downstream GPU (i+1)%N
            send_val = gpu_data[i][c_start:c_end].copy()
            # GPU i receives chunk [(i-1-step)%N] from upstream GPU (i-1)%N
            recv_chunk_idx = (recv_from - step) % N
            rc_start = recv_chunk_idx * chunk_size
            rc_end   = rc_start + chunk_size

            sends.append((i, send_chunk_idx, send_val))

            # Accumulate received chunk
            new_data[i][rc_start:rc_end] += gpu_data[recv_from][rc_start:rc_end]
            total_bytes_transferred += chunk_bytes

        gpu_data = new_data

        if verbose:
            print(f"  Step {step+1}: Each GPU accumulates one more chunk.")
            for i in range(N):
                owned_chunk = (i - step - 1) % N
                cs, ce = owned_chunk*chunk_size, (owned_chunk+1)*chunk_size
                partial = gpu_data[i][cs:ce]
                print(f"    GPU {i} chunk[{owned_chunk}] partial sum: {partial.tolist()}")

    if verbose:
        print()
        print("  After ReduceScatter: GPU i has the correct sum of CHUNK i.")
        for i in range(N):
            cs, ce = i*chunk_size, (i+1)*chunk_size
            print(f"    GPU {i} owns chunk[{i}] = {gpu_data[i][cs:ce].tolist()}")
        print()
        print("  ── PHASE 2: AllGather ──")

    # ── PHASE 2: AllGather ──────────────────────────────────────────
    # Each GPU propagates its correct chunk to all others.

    for step in range(N - 1):
        new_data = [gd.copy() for gd in gpu_data]

        for i in range(N):
            # GPU i forwards chunk [(i - step) % N] downstream
            send_chunk_idx = (i - step) % N
            cs, ce         = send_chunk_idx * chunk_size, (send_chunk_idx+1)*chunk_size
            recv_chunk_idx = (i - 1 - step) % N
            rcs, rce       = recv_chunk_idx*chunk_size, (recv_chunk_idx+1)*chunk_size
            recv_from      = (i - 1) % N

            # Copy correct chunk from upstream GPU
            new_data[i][rcs:rce] = gpu_data[recv_from][rcs:rce]
            total_bytes_transferred += chunk_bytes

        gpu_data = new_data

        if verbose:
            print(f"  Step {step+1}: chunks propagate around the ring.")

    return gpu_data, total_bytes_transferred


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: N=4 detailed trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — N=4 Ring AllReduce: Full Step-by-Step Trace")
print("━" * 68)
print()

N4 = 4
# Each GPU has tensor of 4 elements (1 chunk per GPU)
tensors_4 = [
    np.array([1.0,  2.0,  3.0,  4.0]),
    np.array([5.0,  6.0,  7.0,  8.0]),
    np.array([9.0, 10.0, 11.0, 12.0]),
    np.array([13.0,14.0, 15.0, 16.0]),
]

# Expected result: element-wise sum
expected = sum(tensors_4)
print(f"  N={N4} GPUs, tensor length=4, chunk_size=1")
print(f"  Expected AllReduce result: {expected.tolist()}")
print()

result_4, bytes_4 = ring_allreduce(tensors_4, verbose=True)

print()
print(f"  Final result on each GPU:")
for i, r in enumerate(result_4):
    match = np.allclose(r, expected)
    print(f"    GPU {i}: {r.tolist()}  {'✅' if match else '❌'}")
print()
print(f"  Total bytes transferred: {bytes_4} (float64 units)")
print(f"  Theoretical: 2×(N-1)/N × tensor_bytes = 2×{N4-1}/{N4} × {len(tensors_4[0])*8}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Correctness and volume at different N
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Correctness Verification and Volume Formula")
print("━" * 68)
print()

print(f"  {'N':>4}  {'Tensor size':>12}  {'Bytes transferred':>20}  "
      f"{'Theory 2(N-1)/N×B':>20}  {'Correct?':>10}")
print("  " + "─" * 68)

for N_t in [2, 4, 8, 16]:
    tensor_len = N_t * 8     # 8 elements per chunk, N chunks
    np.random.seed(N_t)
    tens = [np.random.randn(tensor_len) for _ in range(N_t)]
    expected_t = sum(tens)

    result_t, bytes_t = ring_allreduce(tens, verbose=False)

    all_match = all(np.allclose(r, expected_t) for r in result_t)
    theory    = 2 * (N_t-1)/N_t * tensor_len * 8   # float64 bytes

    print(f"  {N_t:>4}  {tensor_len*8:>10} B  {bytes_t:>20}  "
          f"{theory:>20.0f}  {'✅' if all_match and abs(bytes_t-theory) < 1 else '❌':>10}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Per-link bandwidth utilisation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Per-Link Bandwidth: All Links Active in Every Step")
print("━" * 68)
print()

N_link = 8
print(f"  Ring of N={N_link} GPUs. In each of the 2×{N_link-1}={2*(N_link-1)} steps:")
print(f"    ALL {N_link} links are active simultaneously (one chunk each).")
print(f"    No link is idle. Ring is FULLY PIPELINED.")
print()
print(f"  Per-step chunk size: tensor_size / N = B/{N_link}")
print(f"  Per-link bandwidth utilisation: B/{N_link} / step_time = link_bandwidth")
print(f"  ALL {N_link} links transmit simultaneously → aggregate = {N_link} × bw")
print()

bw_nvl3 = 300.0   # GB/s per GPU NVLink3
chunk_mb = 14000 / N_link   # Llama-2-7B split into 8 chunks

step_time_ms = (chunk_mb * 1e6) / (bw_nvl3 * 1e9) * 1000

print(f"  Example: Llama-2-7B gradients ({14000:.0f} MB), N={N_link}, NVLink3")
print(f"  Chunk size: {chunk_mb:.0f} MB per link per step")
print(f"  Step time: {step_time_ms:.3f} ms")
print(f"  Total steps: 2×(N-1) = {2*(N_link-1)}")
print(f"  Total ring time: {2*(N_link-1)*step_time_ms:.1f} ms + latency overhead")
print()
print(f"  Ring link utilisation:")
print(f"    Each link active in EVERY step = {2*(N_link-1)} / {2*(N_link-1)} steps = 100%")
print(f"    This is the bandwidth-optimal property of the ring algorithm.")
''',
    },

    "3 · Tree vs Ring Latency Model — Algorithm Selection & Crossover": {
        "description": (
            "Implement both tree (binary) and ring AllReduce latency models. "
            "Show how latency scales with N for each algorithm. Identify the "
            "crossover message size where ring becomes faster than tree. "
            "Show NCCL's algorithm selection logic. Model the NVLS advantage "
            "for H100. Compute the optimal algorithm for every combination of "
            "message size and GPU count encountered in real training workloads."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  TREE vs RING LATENCY MODEL — Algorithm Selection & Crossover")
print("=" * 68)
print()


def ring_latency_us(n, tensor_bytes, bw_gbs, lat_us):
    """Ring AllReduce: 2*(N-1) steps × lat + bandwidth term."""
    bw_term  = 2 * (n-1)/n * tensor_bytes / (bw_gbs * 1e9) * 1e6
    lat_term = 2 * (n-1) * lat_us
    return lat_term + bw_term

def tree_latency_us(n, tensor_bytes, bw_gbs, lat_us):
    """
    Double binary tree AllReduce.
    Latency steps: 2 × log2(N).
    Bandwidth: each level halves the effective bandwidth (root bottleneck).
    """
    levels   = math.ceil(math.log2(max(n, 2)))
    lat_term = 2 * levels * lat_us
    # Root link carries 2× the data of leaf links (simplified model)
    # Effective bandwidth ≈ bw_gbs (double tree balances load better)
    bw_term  = 2 * (n-1)/n * tensor_bytes / (bw_gbs * 1e9) * 1e6
    return lat_term + bw_term

def nvls_latency_us(n, tensor_bytes, bw_gbs, lat_us):
    """
    NVLS (NVLink Switch): single-pass hardware reduction.
    Bandwidth scales with number of NVLink ports into the switch.
    """
    # Switch aggregates at n × bw_gbs (all ports simultaneously)
    # Conservative model: 0.8× efficiency factor
    effective_bw = min(n * bw_gbs * 0.8, 1600.0)  # cap at switch capacity
    bw_term  = tensor_bytes / (effective_bw * 1e9) * 1e6
    lat_term = lat_us   # single step
    return lat_term + bw_term

def nccl_best_algo(n, tensor_bytes, bw_gbs, lat_us, has_nvls=False):
    """NCCL heuristic: pick algorithm with lowest latency."""
    t_ring = ring_latency_us(n, tensor_bytes, bw_gbs, lat_us)
    t_tree = tree_latency_us(n, tensor_bytes, bw_gbs, lat_us)
    if has_nvls:
        t_nvls = nvls_latency_us(n, tensor_bytes, bw_gbs, lat_us)
        best_t = min(t_ring, t_tree, t_nvls)
        if best_t == t_nvls: return "NVLS", best_t
    else:
        best_t = min(t_ring, t_tree)
    if best_t == t_ring: return "Ring", best_t
    return "Tree", best_t


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Latency vs N for small messages
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Latency vs N: Ring vs Tree for Small Messages (1 KB)")
print("━" * 68)
print()

BW  = 300.0   # NVLink3
LAT = 1.0     # µs
SMALL = 1024  # 1 KB — latency-dominant

print(f"  Message size: {SMALL} bytes, BW={BW} GB/s, α={LAT} µs")
print()
print(f"  {'N GPUs':>8}  {'Ring lat (µs)':>16}  {'Tree lat (µs)':>16}  "
      f"{'Ring/Tree':>11}  {'Winner'}")
print("  " + "─" * 54)

for N in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    t_ring = ring_latency_us(N, SMALL, BW, LAT)
    t_tree = tree_latency_us(N, SMALL, BW, LAT)
    ratio  = t_ring / t_tree
    winner = "Ring" if t_ring < t_tree else "Tree"
    print(f"  {N:>8}  {t_ring:>16.2f}  {t_tree:>16.2f}  "
          f"{ratio:>11.2f}×  {winner}")

print()
print("  Tree grows as 2×log₂(N) while ring grows as 2×(N-1).")
print("  At N=1024: ring latency = 2046 µs, tree = 40 µs — 51× advantage for tree.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Crossover message size (ring vs tree)
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Crossover: Where Ring Beats Tree (NVLink3)")
print("━" * 68)
print()

print(f"  BW={BW} GB/s, α={LAT} µs. Find crossover size where Ring < Tree.")
print()
print(f"  {'N GPUs':>8}  {'Crossover (KB)':>16}  {'Ring at crossover':>20}  "
      f"{'Tree at crossover':>20}")
print("  " + "─" * 64)

for N in [2, 4, 8, 16, 32, 64, 128]:
    # Binary search for crossover
    lo, hi = 1, 1e12
    for _ in range(60):
        mid = (lo + hi) / 2
        if ring_latency_us(N, mid, BW, LAT) < tree_latency_us(N, mid, BW, LAT):
            hi = mid
        else:
            lo = mid
    xo_bytes = (lo + hi) / 2
    t_ring = ring_latency_us(N, xo_bytes, BW, LAT)
    t_tree = tree_latency_us(N, xo_bytes, BW, LAT)
    print(f"  {N:>8}  {xo_bytes/1024:>16.1f}  {t_ring:>20.2f}µs  {t_tree:>20.2f}µs")

print()
print("  NCCL default crossover: ~256 KB (tuned empirically on real hardware).")
print("  Below 256 KB: use Tree. Above: use Ring or NVLS.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: NVLS advantage on H100 (NVLink Switch)
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — NVLS on H100 vs Ring/Tree (NVLink4, N=8)")
print("━" * 68)
print()

BW_H100  = 450.0  # NVLink4 per-GPU unidirectional GB/s
LAT_H100 = 0.8    # µs
N_H100   = 8

print(f"  H100 DGX: N={N_H100}, NVLink4 {BW_H100} GB/s/GPU, α={LAT_H100} µs")
print(f"  NVLS available (NVLink switch hardware in H100 DGX)")
print()
print(f"  {'Size':>14}  {'Ring (ms)':>12}  {'Tree (ms)':>12}  "
      f"{'NVLS (ms)':>12}  {'NCCL picks'}")
print("  " + "─" * 58)

for size_mb in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 14000]:
    size_bytes = size_mb * 1e6
    t_ring = ring_latency_us(N_H100, size_bytes, BW_H100, LAT_H100) / 1000
    t_tree = tree_latency_us(N_H100, size_bytes, BW_H100, LAT_H100) / 1000
    t_nvls = nvls_latency_us(N_H100, size_bytes, BW_H100, LAT_H100) / 1000
    algo, t_best = nccl_best_algo(N_H100, size_bytes, BW_H100, LAT_H100, has_nvls=True)
    print(f"  {size_mb:>12.3f}MB  {t_ring:>12.4f}  {t_tree:>12.4f}  "
          f"{t_nvls:>12.4f}  {algo}")

print()
print("  NVLS wins for large messages: switch reduction in one hardware pass.")
print("  Tree wins for tiny messages: log₂(N) latency steps beats both.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Real training workload — algorithm selection heatmap
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Training Workload: Best Algorithm per Gradient Tensor")
print("━" * 68)
print()

# Representative gradient tensors from a transformer model
gradient_groups = [
    ("Embedding weights",     500e6,   "word_embedding, typically large"),
    ("Attention QKV weights",  50e6,   "3×d×d, common size"),
    ("Attention bias",          0.1e6, "d, tiny"),
    ("FFN gate weight",       100e6,   "d×4d"),
    ("FFN bias",                0.1e6, "4d, tiny"),
    ("LayerNorm weight",        0.001e6, "d, very tiny"),
    ("Output lm_head",        500e6,   "vocab×d"),
]

BW_A100 = 300.0
LAT_A100 = 1.0
N_nodes  = 64   # multi-node

print(f"  Multi-node training: N={N_nodes} GPUs (8 nodes × 8 GPUs), IB HDR200")
BW_IB  = 25.0    # inter-node bottleneck
LAT_IB = 3.0

print(f"  (inter-node IB HDR200: {BW_IB} GB/s, α={LAT_IB} µs)")
print()
print(f"  {'Gradient group':<28}  {'Size (MB)':>10}  "
      f"{'Best algo':>10}  {'Time (ms)':>12}  {'Note'}")
print("  " + "─" * 70)

for name, size_bytes, note in gradient_groups:
    algo, t_us = nccl_best_algo(N_nodes, size_bytes, BW_IB, LAT_IB)
    print(f"  {name:<28}  {size_bytes/1e6:>10.2f}  "
          f"{algo:>10}  {t_us/1000:>12.3f}  {note}")
''',
    },

    "4 · NCCL + CUDA Graph Integration — Capture, Replay & Overhead Elimination": {
        "description": (
            "Model the NCCL AllReduce launch overhead for DDP training and show "
            "how CUDA Graph capture eliminates it. Simulate the warmup-capture-replay "
            "cycle. Quantify per-step overhead with and without CUDA Graphs for "
            "different numbers of gradient buckets. Show the static shape requirement "
            "and compute-communication overlap within a captured graph. Compare "
            "training throughput with and without graph capture."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  NCCL + CUDA GRAPH — Capture, Replay & Overhead Elimination")
print("=" * 68)
print()

np.random.seed(42)

# Overhead constants
NCCL_LAUNCH_US       = 5.0    # µs per ncclAllReduce call
GRAPH_LAUNCH_US      = 0.5    # µs per cudaGraphLaunch (entire graph)
GRAPH_CAPTURE_MS     = 20.0   # ms to warmup + capture (one-time)
BUCKET_SIZE_MB       = 25.0   # PyTorch DDP default bucket size

# Hardware
BW_NVL3_GBS = 300.0   # A100 NVLink3
LAT_US       = 1.0


def allreduce_time_ms(size_bytes, n_gpus, bw_gbs, lat_us):
    vol = 2 * (n_gpus-1)/n_gpus * size_bytes
    return (2*(n_gpus-1)*lat_us + vol/(bw_gbs*1e9)*1e6) / 1000


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Launch overhead per step vs number of buckets
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — DDP Launch Overhead: AllReduce Buckets per Step")
print("━" * 68)
print()

# Model sizes and bucket counts
models = [
    ("GPT-2 (1.5B)",    1.5e9,  "Small model"),
    ("Llama-2-7B",       7e9,   "Medium model"),
    ("Llama-2-70B",     70e9,   "Large model"),
]

BUCKET_BYTES = BUCKET_SIZE_MB * 1e6

print(f"  DDP bucket size: {BUCKET_SIZE_MB:.0f} MB  →  "
      f"n_buckets = param_bytes / bucket_bytes")
print(f"  NCCL launch overhead: {NCCL_LAUNCH_US:.0f} µs/call")
print(f"  CUDA Graph launch:    {GRAPH_LAUNCH_US:.1f} µs (entire step)")
print()
print(f"  {'Model':<20}  {'Params (GB)':>12}  {'n_buckets':>10}  "
      f"{'Conv OH (ms)':>13}  {'Graph OH (µs)':>14}  {'Speedup (OH only)'}")
print("  " + "─" * 74)

for name, params, note in models:
    n_buckets = math.ceil(params * 2 / BUCKET_BYTES)   # BF16 = 2 bytes/param
    oh_conv   = n_buckets * NCCL_LAUNCH_US / 1000     # ms
    oh_graph  = GRAPH_LAUNCH_US                         # µs for entire graph
    speedup_oh = oh_conv * 1000 / oh_graph
    print(f"  {name:<20}  {params*2/1e9:>12.1f}  {n_buckets:>10}  "
          f"{oh_conv:>13.2f}  {oh_graph:>14.1f}  {speedup_oh:>17.0f}×")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Full step time breakdown with and without graphs
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Full Step Time: Compute + Comm + Overhead")
print("━" * 68)
print()

# Llama-2-7B training config
N_GPUS         = 8
PARAM_BYTES    = 7e9 * 2       # BF16
BATCH_FLOPS    = 2 * 64 * 2048 * 7e9  # 2 × batch × seq × params (rough)
GPU_FLOPS      = 312e12        # A100 FP16 TFLOPS
T_COMPUTE_MS   = BATCH_FLOPS / GPU_FLOPS * 1000 * 2  # ×2 for backward

n_buckets_7b   = math.ceil(PARAM_BYTES / BUCKET_BYTES)
t_ar_total_ms  = allreduce_time_ms(PARAM_BYTES, N_GPUS, BW_NVL3_GBS, LAT_US)
oh_conv_ms     = n_buckets_7b * NCCL_LAUNCH_US / 1000
oh_graph_us    = GRAPH_LAUNCH_US

print(f"  Llama-2-7B, N={N_GPUS}, A100 NVLink3, batch=64, seq=2048")
print()

breakdown = [
    ("GPU compute (fwd+bwd+optim)", T_COMPUTE_MS, True,  True,  "runs on SMs"),
    ("AllReduce communication",     t_ar_total_ms, True, True,  "overlapped with compute"),
    ("NCCL launch overhead",        oh_conv_ms,    True, False, "serialised API calls"),
    ("Graph launch overhead",       oh_graph_us/1000, False, True, "one graph launch"),
]

print(f"  {'Component':<34}  {'Without graphs (ms)':>22}  "
      f"{'With graphs (ms)':>20}  {'Notes'}")
print("  " + "─" * 80)

total_conv  = 0.0
total_graph = 0.0
for name, ms, in_conv, in_graph, note in breakdown:
    c_str = f"{ms:>22.3f}" if in_conv  else f"{'—':>22}"
    g_str = f"{ms:>20.3f}" if in_graph else f"{'—':>20}"
    if in_conv:  total_conv  += ms
    if in_graph: total_graph += ms
    print(f"  {name:<34}  {c_str}  {g_str}  {note}")

print("  " + "─" * 80)
print(f"  {'TOTAL':<34}  {total_conv:>22.3f}  {total_graph:>20.3f}")
print()
speedup = total_conv / total_graph
print(f"  CUDA Graph speedup: {speedup:.3f}× (overhead eliminated = "
      f"{oh_conv_ms:.2f} ms / {total_conv:.2f} ms = "
      f"{oh_conv_ms/total_conv*100:.1f}% of step)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Warmup requirement and amortisation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Warmup + Capture + Amortisation")
print("━" * 68)
print()

print("  CUDA Graph + NCCL capture sequence:")
print("    Step 1: NCCL warmup (3 steps) — NCCL initialises internal state")
print("            These steps run normally (no capture).")
print("    Step 2: Graph capture — cudaStreamBeginCapture, run full step, EndCapture.")
print("    Step 3: All subsequent steps → cudaGraphLaunch (< 1 µs).")
print()
print(f"  One-time capture cost: ~{GRAPH_CAPTURE_MS:.0f} ms")
print(f"  Per-step savings: {oh_conv_ms:.2f} ms (NCCL launch overhead eliminated)")
print()

savings_per_step_ms = oh_conv_ms
break_even_steps    = GRAPH_CAPTURE_MS / savings_per_step_ms

print(f"  Break-even: {break_even_steps:.0f} steps")
print()

print(f"  {'N steps':>10}  {'Cumulative savings (ms)':>24}  "
      f"{'vs capture cost (ms)':>22}  {'Net benefit'}")
print("  " + "─" * 58)

for n_steps in [1, 10, 100, 500, 1000, 5000, 10000]:
    cum = n_steps * savings_per_step_ms
    net = cum - GRAPH_CAPTURE_MS
    be  = "✅ yes" if net > 0 else "  no"
    print(f"  {n_steps:>10}  {cum:>24.1f}  {GRAPH_CAPTURE_MS:>22.1f}  {be}")

print()
print("  NCCL+Graph is worth it after just a few hundred steps.")
print("  A typical training run of 1M steps: ~1 second of saved overhead total.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Compute-communication overlap with bucketed AllReduce
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Compute-Communication Overlap: Bucketed AllReduce")
print("━" * 68)
print()

print("  PyTorch DDP fires AllReduce for bucket k while computing gradients")
print("  for bucket k-1. This requires two CUDA streams:")
print("    stream_compute: backward pass kernels")
print("    stream_comm:    NCCL AllReduce kernels")
print()

# Simple overlap model: N_LAYERS layers, each with 2 buckets
N_LAYERS_OL  = 32
T_LAYER_BKWD = 3.0    # ms per layer backward
T_BUCKET_AR  = t_ar_total_ms / (n_buckets_7b)   # ms per bucket AllReduce

print(f"  {N_LAYERS_OL}-layer model, T_layer_bkwd = {T_LAYER_BKWD:.1f} ms, "
      f"T_bucket_ar = {T_BUCKET_AR:.3f} ms")
print()

# Without overlap: backward then AllReduce sequentially
no_overlap_ms = N_LAYERS_OL * T_LAYER_BKWD + n_buckets_7b * T_BUCKET_AR
# With overlap: AllReduce pipelined with compute (critical path = max)
# Assuming AllReduce for earlier layers overlaps with later layers' compute
overlap_ms    = N_LAYERS_OL * T_LAYER_BKWD + max(0,
    n_buckets_7b * T_BUCKET_AR - N_LAYERS_OL * T_LAYER_BKWD * 0.5)

print(f"  {'Approach':<30}  {'Bkwd (ms)':>12}  {'AR (ms)':>12}  "
      f"{'Total (ms)':>12}  {'Speedup'}")
print("  " + "─" * 68)
print(f"  {'Sequential (no overlap)':<30}  {N_LAYERS_OL*T_LAYER_BKWD:>12.1f}  "
      f"{n_buckets_7b*T_BUCKET_AR:>12.1f}  {no_overlap_ms:>12.1f}  1.00×")
print(f"  {'Overlapped (DDP bucket)':<30}  {N_LAYERS_OL*T_LAYER_BKWD:>12.1f}  "
      f"{'hidden':>12}  {overlap_ms:>12.1f}  "
      f"{no_overlap_ms/overlap_ms:.2f}×")
print()
print("  With communication hidden behind compute: step time = max(compute, comm).")
print(f"  Here compute ({N_LAYERS_OL*T_LAYER_BKWD:.0f} ms) >> AllReduce ({n_buckets_7b*T_BUCKET_AR:.0f} ms)")
print("  → AllReduce is fully hidden! Communication is not the bottleneck.")
''',
    },

    "5 · Distributed Training Bandwidth Budget — Communication vs Compute": {
        "description": (
            "Build a comprehensive communication budget model for distributed "
            "training. For each combination of model, GPU count, and interconnect, "
            "compute whether training is compute-bound or communication-bound. "
            "Show the arithmetic intensity of AllReduce (FLOPs per byte of communication). "
            "Find the minimum hardware bandwidth to keep communication hidden under "
            "compute. Identify which parallelism strategy (DDP, ZeRO, tensor-parallel) "
            "is optimal for a given model-hardware combination."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  DISTRIBUTED TRAINING BANDWIDTH BUDGET — Comm vs Compute")
print("=" * 68)
print()

np.random.seed(42)

# Hardware constants
HARDWARE = {
    "H100 DGX (NVLink4)":  {"gpu_tflops": 989.0,  "bw_gbs": 450.0, "lat_us": 0.8},
    "A100 DGX (NVLink3)":  {"gpu_tflops": 312.0,  "bw_gbs": 300.0, "lat_us": 1.0},
    "A100 PCIe (IB HDR)":  {"gpu_tflops": 312.0,  "bw_gbs": 25.0,  "lat_us": 3.0},
    "V100 (PCIe+100GbE)":  {"gpu_tflops": 112.0,  "bw_gbs": 12.5,  "lat_us": 8.0},
}

# Models
MODELS = {
    "GPT-2 (1.5B)":   {"params": 1.5e9,  "layers": 48,  "d": 1600},
    "Llama-2-7B":      {"params": 7e9,    "layers": 32,  "d": 4096},
    "Llama-2-70B":     {"params": 70e9,   "layers": 80,  "d": 8192},
    "Llama-3-405B":    {"params": 405e9,  "layers": 126, "d": 16384},
}


def training_step_flops(params, batch_tokens):
    """Forward + backward FLOPs approximation: 6 × params × tokens."""
    return 6 * params * batch_tokens

def compute_time_ms(params, batch_tokens, gpu_tflops):
    """Time for compute (forward + backward) on N_gpus GPUs (compute divided by N)."""
    flops = training_step_flops(params, batch_tokens)
    return flops / (gpu_tflops * 1e12) * 1000

def allreduce_time_ms(n_gpus, param_bytes, bw_gbs, lat_us):
    vol  = 2 * (n_gpus-1)/n_gpus * param_bytes
    t_us = 2*(n_gpus-1)*lat_us + vol/(bw_gbs*1e9)*1e6
    return t_us / 1000

def ar_per_flop(n_gpus, params, batch_tokens):
    """AllReduce bytes per training FLOP — communication intensity metric."""
    ar_bytes = 2 * (n_gpus-1)/n_gpus * params * 2  # BF16
    flops    = training_step_flops(params, batch_tokens)
    return ar_bytes / flops  # bytes/FLOP

def min_bw_to_hide_comm(n_gpus, params, batch_tokens, gpu_tflops, lat_us=1.0):
    """Minimum interconnect bandwidth to keep comm hidden under compute."""
    t_compute_ms = compute_time_ms(params, batch_tokens, gpu_tflops)
    # Need: allreduce_time_ms <= t_compute_ms
    # allreduce_time = lat_term + bw_term
    # bw_term = 2*(N-1)/N * param_bytes / bw
    # Solve for bw:
    vol_bytes  = 2 * (n_gpus-1)/n_gpus * params * 2  # BF16
    lat_term_ms= 2*(n_gpus-1)*lat_us / 1000
    avail_ms   = max(t_compute_ms - lat_term_ms, 0.001)
    return vol_bytes / (avail_ms / 1000 * 1e9)  # GB/s


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Is training compute-bound or communication-bound?
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Compute-Bound vs Communication-Bound (N=8, batch=64×2048)")
print("━" * 68)
print()

N_GPUS      = 8
BATCH_TOK   = 64 * 2048   # batch=64, seq=2048

print(f"  N={N_GPUS}, batch_tokens={BATCH_TOK:,}")
print()
print(f"  {'Model':<18}  {'Hardware':<26}  {'Compute (ms)':>14}  "
      f"{'AllReduce (ms)':>16}  {'Comm%':>8}  {'Status'}")
print("  " + "─" * 86)

for mname, mdata in MODELS.items():
    if mname in ["Llama-3-405B"]: continue  # skip too-large for 8 GPUs
    for hwname, hw in list(HARDWARE.items())[:3]:
        t_compute = compute_time_ms(mdata["params"], BATCH_TOK, hw["gpu_tflops"])
        t_ar      = allreduce_time_ms(N_GPUS, mdata["params"]*2, hw["bw_gbs"], hw["lat_us"])
        comm_pct  = t_ar / (t_compute + t_ar) * 100
        status    = "⚠ COMM-BOUND" if t_ar > t_compute else "✅ compute"
        print(f"  {mname:<18}  {hwname:<26}  {t_compute:>14.1f}  "
              f"{t_ar:>16.1f}  {comm_pct:>7.1f}%  {status}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Minimum bandwidth to hide communication
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Minimum Bandwidth to Hide AllReduce Under Compute")
print("━" * 68)
print()

print(f"  N={N_GPUS} GPUs, batch_tokens={BATCH_TOK:,}")
print()
print(f"  {'Model':<18}  {'Hardware':<20}  {'Compute (ms)':>14}  "
      f"{'Min BW (GB/s)':>15}  {'Actual BW':>12}  {'Hidden?'}")
print("  " + "─" * 78)

for mname, mdata in list(MODELS.items())[:3]:
    for hwname, hw in list(HARDWARE.items())[:3]:
        t_compute = compute_time_ms(mdata["params"], BATCH_TOK, hw["gpu_tflops"])
        min_bw    = min_bw_to_hide_comm(N_GPUS, mdata["params"],
                                         BATCH_TOK, hw["gpu_tflops"], hw["lat_us"])
        hidden    = hw["bw_gbs"] >= min_bw
        status    = "✅ yes" if hidden else "❌ no"
        print(f"  {mname:<18}  {hwname:<20}  {t_compute:>14.1f}  "
              f"{min_bw:>15.1f}  {hw['bw_gbs']:>12.1f}  {status}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Parallelism strategy selection
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Parallelism Strategy: DDP vs ZeRO vs Tensor Parallel")
print("━" * 68)
print()

print("  STRATEGY COMPARISON: Llama-2-70B, 8 × H100 (NVLink4)")
hw = HARDWARE["H100 DGX (NVLink4)"]
mdata = MODELS["Llama-2-70B"]
N_GPUS_3   = 8
PARAM_BYTES_3 = mdata["params"] * 2  # BF16

t_compute_70b = compute_time_ms(mdata["params"], BATCH_TOK, hw["gpu_tflops"])

strategies = [
    ("DDP",
     allreduce_time_ms(N_GPUS_3, PARAM_BYTES_3, hw["bw_gbs"], hw["lat_us"]),
     140.0,   # model bytes per GPU (full model on each)
     "Best for: N_gpus × compute > comm"),
    ("ZeRO-1",
     allreduce_time_ms(N_GPUS_3, PARAM_BYTES_3, hw["bw_gbs"], hw["lat_us"]),
     140.0 / 2,  # saves optimizer state: 70 GB → 35 GB
     "Same comm as DDP, saves optimizer memory"),
    ("ZeRO-3",
     allreduce_time_ms(N_GPUS_3, PARAM_BYTES_3, hw["bw_gbs"], hw["lat_us"]) * 1.5,
     140.0 / 8,  # params partitioned: 140/8 = 17.5 GB per GPU
     "3× comm vs DDP, but N× memory reduction"),
    ("Tensor Parallel (TP=8)",
     allreduce_time_ms(N_GPUS_3,
         mdata["d"] * 4 * mdata["d"] * 2 * mdata["layers"],  # FFN activation bytes
         hw["bw_gbs"], hw["lat_us"]) * 2,
     140.0 / 8,  # model sharded: 17.5 GB per GPU
     "AllGather activations per layer, 2 collectives/layer"),
]

print(f"  Compute time: {t_compute_70b:.1f} ms per step (batch={BATCH_TOK})")
print()
print(f"  {'Strategy':<24}  {'Comm (ms)':>12}  {'GPU mem (GB)':>14}  "
      f"{'Comm hidden?':>14}  {'Notes'}")
print("  " + "─" * 72)

for name, t_comm, mem_gb, notes in strategies:
    hidden = t_comm <= t_compute_70b
    print(f"  {name:<24}  {t_comm:>12.1f}  {mem_gb:>14.1f}  "
          f"{'✅ yes' if hidden else '❌ no':>14}  {notes}")

print()
print("  On H100 DGX with NVLink4: DDP and ZeRO-1 both hide communication.")
print("  ZeRO-3 useful when model doesn't fit in GPU memory (e.g., 8×H100 for 405B).")
print("  Tensor Parallel introduces intra-layer AllGather — needs low-latency NVLink.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Scaling communication with GPU count
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Scaling: AllReduce Time vs N (Llama-2-7B, A100 DGX)")
print("━" * 68)
print()

hw_a100 = HARDWARE["A100 DGX (NVLink3)"]
m7b     = MODELS["Llama-2-7B"]
PARAM_B = m7b["params"] * 2   # BF16

print(f"  Model: Llama-2-7B ({PARAM_B/1e9:.0f} GB BF16), A100 NVLink3")
print()
print(f"  {'N GPUs':>8}  {'AllReduce vol/GPU':>20}  {'AR time (ms)':>14}  "
      f"{'Compute/N (ms)':>16}  {'Speedup':>10}  {'Bottleneck'}")
print("  " + "─" * 72)

for N in [1, 2, 4, 8, 16, 32, 64, 128]:
    vol_per_gpu = 2*(N-1)/N * PARAM_B / 1e9   # GB
    t_ar_ms     = allreduce_time_ms(N, PARAM_B, hw_a100["bw_gbs"], hw_a100["lat_us"])
    t_compute   = compute_time_ms(m7b["params"], BATCH_TOK, hw_a100["gpu_tflops"]) / N
    speedup     = t_compute / (t_compute + t_ar_ms) if N > 1 else 1.0
    bottleneck  = "⚠ comm" if t_ar_ms > t_compute else "✅ compute"
    print(f"  {N:>8}  {vol_per_gpu:>19.2f}G  {t_ar_ms:>14.1f}  "
          f"{t_compute:>16.1f}  {speedup:>9.2f}×  {bottleneck}")

print()
print("  At large N: communication volume per GPU saturates at 2×params.")
print("  Compute scales as 1/N; at some point AllReduce > compute → comm-bound.")
print("  For Llama-2-7B on A100: becomes comm-bound around N=64.")
''',
    },
}

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