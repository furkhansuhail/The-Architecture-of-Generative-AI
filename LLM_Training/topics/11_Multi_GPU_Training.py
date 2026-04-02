"""
Multi-GPU LLM Training — Distributed Training, Dataset Sharding, and Debugging
===============================================================================

Training a 7-billion-parameter LLM on a single GPU is already barely
feasible. Training a 70-billion-parameter model, or training any model
quickly enough to be practical, requires distributing the computation
across dozens, hundreds, or thousands of GPUs working in parallel.

Multi-GPU training introduces a fundamental new complexity: COORDINATION.
Every GPU must work on a consistent view of the model, every gradient
must be correctly aggregated, and every dataset shard must be disjoint
so no GPU sees the same data as another. A bug in any of these
coordination mechanisms produces subtly wrong results — the model trains,
the loss decreases, and you only discover the problem weeks later when
the model underperforms on evaluation.

This module covers the complete multi-GPU training stack:
    Communication primitives: AllReduce, AllGather, ReduceScatter
    Parallelism strategies: data, tensor, pipeline, expert
    Dataset sharding: how to split 2T tokens across 1000 GPUs
    PyTorch DDP and FSDP: the two main frameworks
    DeepSpeed ZeRO: memory-optimal distributed training
    Megatron-LM: the NVIDIA training framework for 100B+ models
    Monitoring: detecting straggler GPUs, communication bottlenecks
    Debugging: the hardest part of distributed training

"""

import textwrap
import re

TOPIC_NAME   = "Multi-GPU LLM Training"
DISPLAY_NAME = "11 · Multi-GPU LLM Training"
ICON         = "🖥️"
SUBTITLE     = "DDP, FSDP, Tensor Parallelism, Dataset Sharding, and Distributed Debugging"


THEORY = """


### Why Multi-GPU Training Is Necessary

A single NVIDIA A100 (80 GB SXM) holds 80 GB of HBM. Training a 7B model
with AdamW in bf16 requires roughly:

    Weights:         7B × 2 bytes  =  14 GB
    FP32 master:     7B × 4 bytes  =  28 GB
    Gradients:       7B × 2 bytes  =  14 GB
    Optimiser m/v:   7B × 8 bytes  =  56 GB
    Activations:     ≈  10–50 GB  (batch-dependent)
    Total:           ≈  122–162 GB

Even with tricks (activation checkpointing, no FP32 master copy), a 7B
model barely fits on 8 × A100 in pure data parallelism. A 70B model requires
at minimum 16–32 A100s just for the parameters.

Multi-GPU training solves this through three orthogonal strategies:
    •   Data Parallelism (DP / DDP): same model on every GPU, different data
    •   Tensor Parallelism (TP): split individual weight matrices across GPUs
    •   Pipeline Parallelism (PP): assign different layers to different GPUs

These can be combined ("3D parallelism"). This module focuses on the hardware
and communication primitives that underpin all three.


### GPU Interconnect Hierarchy

GPUs in a cluster are connected at multiple levels, with dramatically
different bandwidths at each level:

    Level           Technology      Bandwidth      Latency
    ──────────────────────────────────────────────────────────────────
    Within-node     NVLink 4.0      900 GB/s       ~1 µs
    Within-node     NVLink 3.0      600 GB/s       ~1 µs
    Within-node     PCIe 5.0        128 GB/s       ~5 µs
    Cross-node      InfiniBand HDR  200 Gb/s       ~1 µs
    Cross-node      InfiniBand NDR  400 Gb/s       ~1 µs
    Cross-node      Ethernet 100GbE ~12 GB/s       ~10 µs
    ──────────────────────────────────────────────────────────────────

A DGX A100 node has 8 GPUs connected in an NVLink mesh (all-to-all). An
A100 has 12 NVLink 3.0 connections providing 600 GB/s bidirectional bandwidth
between any two GPUs within the node.

**The bandwidth cliff:** Moving data across nodes (via InfiniBand) is 3–50×
slower than within-node NVLink. This is why communication-heavy operations
(all-reduce) are expensive in multi-node setups and why minimising cross-node
communication is a core concern in distributed training design.


    **Diagram 1 — NVLink Topology within a DGX Node:**

    NVLINK ALL-TO-ALL TOPOLOGY (DGX A100, 8 GPUs)
    ════════════════════════════════════════════════════════════════

         GPU0 ────────────────────── GPU1
          │ ╲                      ╱ │
          │   ╲                  ╱   │
          │     GPU2 ────── GPU3     │
          │      │               │   │
          │      │               │   │
          │     GPU4 ────── GPU5     │
          │   ╱                  ╲   │
          │ ╱                      ╲ │
         GPU6 ────────────────────── GPU7

    Each GPU connected to every other via NVLink (full mesh).
    Bandwidth between any pair: ~600 GB/s (NVLink 3.0 on A100).

    Cross-node (to another DGX):
         [GPU0-7]  ──── InfiniBand HDR (200 Gb/s) ────  [GPU0-7]
                   400–1200× slower than NVLink!


### NCCL — The Communication Library

**NCCL (NVIDIA Collective Communications Library)** is the runtime that
implements collective operations across GPUs. PyTorch's distributed package
uses NCCL as its backend for GPU training.

NCCL provides:
    •   Automatic topology detection (NVLink, PCIe, IB)
    •   Ring-based and tree-based algorithms optimised for each topology
    •   Pipelining of computation and communication
    •   Support for multi-node via InfiniBand or Ethernet


### Collective Communication Operations

A **collective operation** is one where all processes in a group participate
and the output depends on contributions from all. These are the building
blocks of distributed training:


**All-Reduce** — the most important operation for data parallelism.
Every process has a tensor; the operation produces the sum (or mean)
across all processes, with every process receiving the result.

    Before:  GPU0=[1,2],  GPU1=[3,4],  GPU2=[5,6]
    After:   GPU0=[9,12], GPU1=[9,12], GPU2=[9,12]  ← all receive sum

    Use in DDP: average gradients across all GPUs after backward pass.


**Reduce-Scatter** — each process contributes a chunk; each process
receives a different chunk of the reduced result.

    Before:  GPU0=[a0,a1], GPU1=[b0,b1], GPU2=[c0,c1]
    After:   GPU0=[a0+b0+c0],  GPU1=[a1+b1+c1]
                GPU0 gets chunk 0 of the sum
                GPU1 gets chunk 1 of the sum

    Use in ZeRO: each GPU accumulates only its shard of the gradients.


**All-Gather** — each process has a shard; all processes receive the
full tensor by gathering all shards.

    Before:  GPU0=[a],  GPU1=[b],  GPU2=[c]
    After:   GPU0=[a,b,c], GPU1=[a,b,c], GPU2=[a,b,c]

    Use in ZeRO: reconstruct full weight matrix from shards for compute.


**Broadcast** — one process sends its tensor to all others.

    Before:  GPU0=[X],   GPU1=[0],  GPU2=[0]
    After:   GPU0=[X],   GPU1=[X],  GPU2=[X]

    Use: distribute initial weights; send batch metadata.


**Reduce** — like all-reduce but only one process receives the result.

    Before:  GPU0=[1,2], GPU1=[3,4], GPU2=[5,6]
    After:   GPU0=[9,12], GPU1=[unchanged], GPU2=[unchanged]

    Use: aggregate metrics to a single "master" process for logging.


    **Diagram 2 — All-Reduce via Ring Algorithm:**

    RING ALL-REDUCE (N=4 GPUs, gradient vector split into 4 chunks)
    ════════════════════════════════════════════════════════════════

    Phase 1 — Reduce-Scatter (N-1 steps):
    Each GPU sends one chunk to its right neighbour, accumulating partial sums.

    Step 1:     GPU0 sends g0   → GPU1  (GPU1 now has g0+g1[0])
                GPU1 sends g1   → GPU2  (GPU2 now has g1+g2[1])
                GPU2 sends g2   → GPU3  (GPU3 now has g2+g3[2])
                GPU3 sends g3   → GPU0  (GPU0 now has g3+g0[3])

    After N-1 steps: each GPU has the full sum of ONE chunk.

    Phase 2 — All-Gather (N-1 steps):
    Each GPU sends its fully-reduced chunk to the right; all GPUs end up
    with all chunks.

    Result: every GPU has the complete gradient sum. ✓

    Total data transferred per GPU: 2 × (N-1)/N × |gradient|
    ≈ 2 × |gradient|   (approaches 2|g| for large N)

    This is communication-optimal — you cannot do better than 2|g| per GPU.


### Ring vs Tree Topologies

NCCL selects the algorithm based on hardware topology:

    **Ring all-reduce:** Optimal for even-bandwidth topologies (NVLink mesh).
    Each step transfers one chunk. Total time = 2(N-1)/N × message_size /
    bandwidth. Scales perfectly with N on uniform bandwidth.

    **Double-Binary-Tree:** Better for multi-node where some links are slower.
    Two simultaneous binary-tree reduce operations cover the tree imbalance.
    Preferred when cross-node bandwidth << within-node bandwidth.

    **Recursive Halving-Doubling:** Better for small message sizes (fine-
    grained operations). Used when latency dominates over bandwidth.


### Process Groups and Ranks

PyTorch Distributed uses the following abstractions:

    **World size:** Total number of processes (usually = total GPUs).
    **Rank:**       Integer ID of each process (0 to world_size-1).
    **Local rank:** Rank within a node (0 to num_gpus_per_node-1).
                    Used to assign each process to a specific GPU.
    **Process group:** A subset of ranks that participate in a collective.
                       The default group includes all ranks.

    torchrun automatically sets:
        RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

Each rank calls `dist.init_process_group(backend="nccl")` to connect.
After this, collective operations (all_reduce, broadcast, etc.) are
synchronised across all ranks in the group.


### Bandwidth-Optimal All-Reduce: The Math

For a gradient tensor of size S bytes across N GPUs:

    Naive (GPU0 gathers all, then broadcasts back):
        Bandwidth bottleneck at GPU0: (N-1)×S receive + (N-1)×S send
        Total data at bottleneck: 2(N-1)S — does NOT scale with N

    Ring all-reduce:
        Each GPU sends/receives 2(N-1)/N × S bytes total
        As N→∞, this approaches 2S per GPU — constant!
        The algorithm is bandwidth-optimal: no algorithm can do better.

    All-to-All (used in MoE routing):
        Each GPU sends S/N bytes to each of N GPUs
        Total per GPU: S bytes sent, S bytes received
        Same complexity as ring all-reduce


### GPU Memory Bandwidth vs Compute: The Real Bottleneck

A common misconception: multi-GPU training is compute-bound. In practice,
for LLM training:

    A100 peak compute:   312 TFLOPS (bf16 Tensor Core)
    A100 memory BW:      2 TB/s HBM
    NVLink 3.0 BW:       600 GB/s

    For a linear layer (W ∈ ℝ^(4096×4096), batch=2048, T=2048):
        FLOPs:    2 × 4096 × 4096 × 2048 × 2048 ≈ too large for batch
        Actually: per forward pass layer FLOPs ≈ 2 × B × T × d²
        At B=8, T=2048, d=4096: ~549 GFLOPS per layer
        At 312 TFLOPS: 1.8 ms compute
        Memory read: 4096×4096×2 bytes = 32 MB → at 2 TB/s: 0.016 ms

    → The compute is fast; the bottleneck is reading weights from HBM.

This is why FlashAttention (which avoids writing the full attention matrix
to HBM) and efficient kernel fusion matter — they reduce memory bandwidth
usage, not FLOPs.

For communication:
    A 7B model gradient = 7B × 2 bytes = 14 GB
    Ring all-reduce across 8 GPUs (NVLink): 14 GB / 600 GB/s ≈ 23 ms
    Ring all-reduce across 8 nodes (IB HDR): 14 GB / 25 GB/s ≈ 560 ms

    → Gradient synchronisation can dominate training time in multi-node setups.
    → This is the primary motivation for ZeRO, gradient compression, and
       communication-computation overlap.


### Setting Up a Distributed PyTorch Environment

The standard entry point is `torchrun` (replaces `torch.distributed.launch`):

    torchrun --nproc_per_node=8 \\
             --nnodes=2 \\
             --node_rank=0 \\
             --master_addr=192.168.1.1 \\
             --master_port=29500 \\
             train.py

Inside the training script:
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ... training code ...

    dist.destroy_process_group()

Key environment variables set by torchrun:
    RANK           — global rank of this process
    LOCAL_RANK     — rank within the current node
    WORLD_SIZE     — total number of processes
    MASTER_ADDR    — IP of rank-0 node
    MASTER_PORT    — port for rendezvous



##### PART 1 — DISTRIBUTED COMMUNICATION PRIMITIVES

### The Coordination Problem

    When N GPUs compute gradients on their own data shards, each GPU holds
    a LOCAL gradient that reflects only the data IT saw. For training to
    be equivalent to single-GPU training on N× more data, the gradients
    must be AVERAGED across all GPUs before the optimizer step.

    This averaging requires COMMUNICATION between GPUs. The choice of
    communication pattern determines:
        - How much data is transferred
        - How long the communication takes
        - Whether it can overlap with computation

### AllReduce: Sum and Distribute

    AllReduce takes one tensor per GPU, sums them, and gives every GPU
    the result:

        GPU 0: [a0, b0, c0]
        GPU 1: [a1, b1, c1]   →  AllReduce (SUM)  →  Each GPU gets:
        GPU 2: [a2, b2, c2]                            [a0+a1+a2, b0+b1+b2, c0+c1+c2]

    Used in: Data Parallel training to sum gradients across all GPUs.

    Ring-AllReduce algorithm (bandwidth-optimal):
        Step 1 (ReduceScatter): each GPU sends a chunk to the NEXT GPU, adding.
               After N-1 steps: each GPU holds the reduced result for one CHUNK.
        Step 2 (AllGather):    each GPU sends its chunk to the NEXT GPU, no reduce.
               After N-1 steps: every GPU holds the FULL reduced result.

        Bandwidth: 2 × (N-1)/N × data_size ≈ 2 × data_size
        (Each element traverses each link once in each direction)
        Does NOT get worse as N grows (scalable to 1000s of GPUs)!

    PyTorch:  torch.distributed.all_reduce(tensor, op=dist.ReduceOp.SUM)
    NCCL:     ncclAllReduce() — GPU-native, NVLink-optimised.

### AllGather: Collect Shards

    AllGather takes one SHARD per GPU and gives every GPU ALL shards:

        GPU 0: [A]
        GPU 1: [B]   →  AllGather  →  Each GPU gets: [A, B, C]
        GPU 2: [C]

    Used in: FSDP to reassemble sharded weights before each layer's forward pass.

    Bandwidth: N × shard_size per GPU = full tensor size.

### ReduceScatter: Reduce and Shard

    ReduceScatter is AllReduce's first half:
        Takes one tensor per GPU (same total size),
        sums them, and gives each GPU ONE SHARD of the result:

        GPU 0: [a0, b0, c0]
        GPU 1: [a1, b1, c1]   → ReduceScatter →
        GPU 2: [a2, b2, c2]
        GPU 0 gets: [a0+a1+a2]      (only first third)
        GPU 1 gets: [b0+b1+b2]      (only middle third)
        GPU 2 gets: [c0+c1+c2]      (only last third)

    Used in: FSDP/ZeRO-3 to distribute reduced gradients after backward.
    The combination: ReduceScatter + AllGather = AllReduce (same bandwidth cost).

### Point-to-Point Communication

    GPU i sends data to GPU j and only GPU j receives it:
    send(tensor, dst=j),  recv(tensor, src=i)

    Used in: Pipeline parallelism to pass activations between pipeline stages.
    Also: Tensor parallelism at layer boundaries for specific patterns.

### Broadcast

    GPU 0 sends the same tensor to ALL other GPUs:
    broadcast(tensor, src=0)

    Used in: Distributing initial model weights at training start.
    Also: Distributing random seeds for reproducible data loading.

### Hardware Interconnects

    NVLink (GPU-GPU within a node):
        A100 NVLink 3.0: 600 GB/s bidirectional per GPU
        H100 NVLink 4.0: 900 GB/s bidirectional per GPU
        DGX A100 (8 GPUs): 4.8 TB/s total NVLink bandwidth
        Essential for tensor parallelism within a node.

    InfiniBand (GPU-GPU across nodes):
        IB HDR:  200 Gb/s = 25 GB/s per port
        IB NDR:  400 Gb/s = 50 GB/s per port
        Used for: inter-node AllReduce (data parallelism across nodes)
        Much slower than NVLink: 25 GB/s vs 600 GB/s.

    Implication:
        TENSOR PARALLELISM (high communication): within-node NVLink only.
        DATA PARALLELISM (lower communication): can span nodes via IB.
        PIPELINE PARALLELISM (sequential): can span nodes, IB latency matters.


##### PART 2 — PARALLELISM STRATEGIES

### Data Parallelism (DP)

    CONCEPT: Every GPU holds a FULL COPY of the model.
    Each GPU processes a DIFFERENT BATCH of data.
    After backward: AllReduce gradients → average → identical update.

        GPU 0: full model + batch_0 → gradients_0  ┐
        GPU 1: full model + batch_1 → gradients_1  ├→ AllReduce → averaged gradients
        GPU 2: full model + batch_2 → gradients_2  ┘   → each GPU applies same update

    EFFECTIVE BATCH SIZE = per_gpu_batch × n_gpus

    REQUIREMENTS:
        Model must fit on a single GPU (weights + gradients + optimizer).
        Communication: AllReduce of gradient tensor (model_size × 2 bytes).

    SCALING:
        Linear throughput scaling: 2 GPUs → 2× throughput.
        Communication overhead: AllReduce ≈ constant time (Ring-AllReduce).
        Practical efficiency: 90–95% linear scaling up to ~64 GPUs.

    USE CASE: Models that fit in one GPU. Training speed-up via large batch.

### PyTorch DDP (DistributedDataParallel)

    DDP is the standard PyTorch data parallel implementation.

    Key optimisation: BUCKET-BASED AllReduce:
        Gradients are grouped into buckets of ~25 MB.
        As each bucket becomes ready (all its parameters have computed gradients),
        AllReduce starts for that bucket — OVERLAPPING with backward pass.
        This hides communication latency behind computation.

    Initialisation:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        model = DistributedDataParallel(model, device_ids=[local_rank])

    DDP automatically:
        Hooks gradient computation to trigger AllReduce when ready.
        Synchronises model parameters at init (broadcasts from rank 0).
        Ensures all processes start from identical model state.

### Tensor Parallelism (TP) — Megatron-LM Style

    CONCEPT: Split individual LAYERS across GPUs.
    Each GPU holds a SHARD of each weight matrix.
    For an MLP layer Y = GELU(X·A)·B (A: d×4d, B: 4d×d):

        Column-parallel A:
            GPU 0: A[:,  :2d] → Y_0 = GELU(X·A_0)  (first half of output)
            GPU 1: A[:, 2d:]  → Y_1 = GELU(X·A_1)  (second half of output)
            → No communication needed yet (each GPU has a partial output)

        Row-parallel B:
            GPU 0: B[:2d, :] → Y_0 · B_0  (GPU 0's contribution to final output)
            GPU 1: B[2d:, :] → Y_1 · B_1
            → AllReduce to sum partial results → full output on each GPU

    For attention: split by HEAD.
        GPU 0: heads 0, 1, ..., H/2-1
        GPU 1: heads H/2, ..., H-1
        After attention output projection: AllReduce.

    COMMUNICATION PER LAYER:
        Forward:  one AllReduce per MLP + one AllReduce per attention output
        Backward: two AllReduces per layer
        Total: 4 AllReduces per transformer layer.

    REQUIREMENTS: NVLink — TP=4 or TP=8 within a single DGX node.
    Beyond the node: TP communication over InfiniBand is too slow.

    USE CASE: Model is too large to fit on one GPU.
    Memory per GPU: model_size / (TP × PP).

### Pipeline Parallelism (PP)

    CONCEPT: Divide transformer LAYERS into stages, one stage per GPU group.

        GPU 0:  Layers 0–7   (embedding + first 8 layers)
        GPU 1:  Layers 8–15  (middle 8 layers)
        GPU 2:  Layers 16–23 (next 8 layers)
        GPU 3:  Layers 24–31 (last 8 layers + LM head)

    Data flows SEQUENTIALLY through stages:
        GPU 0 processes batch → sends activation to GPU 1 → ... → GPU 3 (output)

    NAIVE PIPELINE BUBBLE PROBLEM:
        While GPU 0 computes micro-batch 0, GPUs 1,2,3 are IDLE.
        After GPU 0 sends to GPU 1: GPU 1 works, GPU 0 starts micro-batch 1.
        Steady state: all GPUs busy.
        But warmup and cooldown cause a "bubble" of idle time.
        Bubble fraction ≈ (N_stages - 1) / N_microbatches.

    GPIPE SCHEDULE:
        Split each batch into M micro-batches.
        Process all M micro-batches forward, then all backward.
        Bubble = (N_stages - 1) / (N_stages - 1 + M) → small for large M.

    1F1B SCHEDULE (Interleaved, used in Megatron-LM):
        Alternate 1 Forward, 1 Backward per micro-batch.
        Maintains constant pipeline occupancy.
        Lower memory than GPipe (no need to store all M micro-batch activations).

    REQUIREMENTS: One GPU group per pipeline stage.
    Communication: send/receive activations between stages (smaller than weights).
    Works across nodes (InfiniBand acceptable since it's point-to-point, not AllReduce).

### ZeRO: Memory-Optimal Data Parallelism

    ZeRO (Zero Redundancy Optimizer) is Microsoft's approach to eliminating
    the memory redundancy in standard data parallelism.

    Standard DP: EVERY GPU stores full model + gradients + optimizer.
    ZeRO: SHARD these across all GPUs in the data parallel group.

    ZeRO Stage 1 (Optimizer State Sharding):
        Each GPU stores only 1/N of the optimizer states.
        AllReduce gradients, then each GPU updates its 1/N of parameters.
        AllGather parameter updates to synchronise.
        Memory: optimizer goes from 4×N_params to 4×N_params/N.

    ZeRO Stage 2 (Gradient Sharding):
        Each GPU stores only 1/N of the gradients.
        ReduceScatter gradients → each GPU holds its shard.
        Memory: gradients from 2×N_params to 2×N_params/N.

    ZeRO Stage 3 (Full Sharding = FSDP):
        Each GPU stores only 1/N of the weights.
        AllGather weights before each forward/backward use.
        ReduceScatter gradients after backward.
        Memory: weights from 2×N_params to 2×N_params/N.
        Total: only 1/N of the full model state per GPU!

    PyTorch FSDP (Fully Sharded Data Parallel) implements ZeRO Stage 3.

### 3D Parallelism: Combining All Three

    For very large models (100B+):
        Tensor Parallelism (TP=8):   within a node (NVLink)
        Pipeline Parallelism (PP=N): across nodes (InfiniBand)
        Data Parallelism (DP=M):     across pipeline replicas

        Total GPUs = TP × PP × DP

    Example: LLaMA-2 70B with 1024 A100s:
        TP=8:   8 GPUs per tensor-parallel group (within node)
        PP=16:  16 pipeline stages
        DP=8:   8 replicas of the full TP×PP model
        Total:  8 × 16 × 8 = 1024 GPUs ✓

    Each GPU holds: model_size / (TP × PP) + overhead
    LLaMA-2 70B: 140 GB / (8 × 16) = 1.09 GB weight per GPU
    Plus activations, KV cache: ≈ 20-30 GB total per GPU (fits in 80GB).


##### PART 3 — DATASET SHARDING: SPLITTING THE TRAINING DATA

### Why Sharding Is Critical

    The goal: every data-parallel replica must see DIFFERENT data.
    If GPU 0 and GPU 1 see the same batch, the AllReduce gradient average
    is meaningless (same gradient twice = same gradient once).
    Effective batch size would be 1×, not 2×.

    Also: every GPU must eventually see EVERY token in the training set.
    At the end of training: all N_epochs × N_tokens have been consumed.
    With N GPUs: each GPU processes N_tokens/N per epoch.

### Static Sharding (Simplest)

    Pre-compute the sharding before training:
        1. Tokenise the full corpus → one giant file of token IDs.
        2. Split into N equal shards (one per DP rank).
        3. GPU i only ever loads shard i.

    Implementation:
        # Before training: split corpus
        total_tokens = len(all_token_ids)
        shard_size   = total_tokens // world_size
        for rank in range(world_size):
            shard = all_token_ids[rank*shard_size:(rank+1)*shard_size]
            save(shard, f"shard_{rank:04d}.bin")

        # During training: GPU i loads its own shard
        local_rank_data = load(f"shard_{rank:04d}.bin")
        chunks = [local_rank_data[i:i+ctx] for i in range(0, len(local_rank_data), ctx)]
        # DataLoader shuffles these chunks each epoch

    Pros:  Simple, reproducible, no coordination needed.
    Cons:  Inflexible (fixed shards). Each GPU sees only its shard per epoch.
           Small shards may have uneven data mixture per GPU.

### Dynamic Sharding with a Global Shuffle

    More common in practice: globally shuffle, then each GPU takes its slice.

        # Shared random seed across all GPUs → same shuffle seen by all
        np.random.seed(global_seed + epoch)
        all_chunks    = shuffle(all_token_id_chunks)
        gpu_i_chunks  = all_chunks[rank::world_size]   # every Nth chunk

    This way: after a full epoch, EVERY GPU has seen DIFFERENT data,
    and collectively the dataset has been fully covered.

    The DistributedSampler in PyTorch does this automatically:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=B)
        # At each epoch: sampler.set_epoch(epoch) to change the shuffle

### Data Mixing and Proportions in Multi-GPU Settings

    When training on multiple data sources:
        Maintain proportions correctly even across GPUs.
        Each GPU samples from the mixture with the SAME probabilities.
        Seeding: each GPU uses (global_seed, rank) for its sampler.
        Cross-GPU: each GPU independently samples from the same distribution.

    DO NOT: have GPU 0 sample all web text and GPU 1 sample all code.
    CORRECT: each GPU samples from the full mixture.
    This is important because otherwise some GPUs train only on code and
    may have very different gradient directions, destabilising training.

### Sequence Packing Across GPUs

    For efficiency, pack sequences to fill context windows exactly:
        Concatenate documents: doc0 <EOS> doc1 <EOS> doc2 ... until context_length.
        No padding wasted compute.
        Loss masking: ignore cross-document positions in attention (optional).

    In multi-GPU setting: each GPU packs its own shard independently.
    They may have slightly different packing density but that's fine.

### Validation Data Sharding

    Validation data must be evaluated CONSISTENTLY across all GPUs:
        Option A: Each GPU evaluates on its own validation shard.
                  Average the val loss across GPUs (AllReduce the loss).
        Option B: Only rank 0 evaluates on the FULL validation set.
                  Simpler but only rank 0 is active during validation.
                  Acceptable overhead if validation is infrequent.

    Standard practice: AllReduce the validation loss.
        val_loss = compute_local_val_loss(model, my_val_shard)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        # Now all ranks have the global average validation loss.


##### PART 4 — PYTORCH DDP AND FSDP: IMPLEMENTATION DETAILS

### DDP Setup Pattern

    Every GPU runs the SAME Python script. They communicate via the process group.

    Launcher: torchrun (recommended over torch.distributed.launch):
        torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
                 --master_addr=10.0.0.1 --master_port=29500 \
                 train.py --config config.yaml

    Each process:
        RANK:        global process index (0 to world_size-1)
        LOCAL_RANK:  GPU index on this node (0 to nproc_per_node-1)
        WORLD_SIZE:  total number of processes

    Initialisation code:
        import os
        import torch.distributed as dist

        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(backend="nccl")   # NCCL for GPU-GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Build model
        model = GPT(config).to(device)

        # Wrap with DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,   # True only if some params unused
            gradient_as_bucket_view=True,   # memory optimization
        )

    Key DDP hooks:
        model._set_static_graph(): if model graph doesn't change (faster).
        no_sync() context: disable gradient sync for accumulation steps.
        for _ in range(accum_steps - 1):
            with model.no_sync():      # skip AllReduce for these steps
                loss.backward()
        loss.backward()                # sync on last step

### FSDP Setup Pattern

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    # Define which modules to shard (wrap each transformer layer separately)
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock}
    )

    # Mixed precision configuration
    bf16_policy = MixedPrecision(
        param_dtype  = torch.bfloat16,   # weights in BF16
        reduce_dtype = torch.float32,     # AllReduce in FP32
        buffer_dtype = torch.bfloat16,   # buffers in BF16
    )

    # Wrap model with FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        device_id=local_rank,
    )

    # Checkpoint with FSDP (special handling required)
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp import FullStateDictConfig

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()   # gathers all shards to rank 0

    FSDP memory per GPU (vs DDP):
        DDP:  model + grad + optimizer = full replication
        FSDP: (model + grad + optimizer) / N_gpus = 1/N of DDP
        FSDP enables training models N× larger than DDP with same hardware.

### Saving and Loading Checkpoints

    DDP checkpoint (simpler):
        # Save only on rank 0 (all replicas have identical weights in DDP)
        if rank == 0:
            torch.save(model.module.state_dict(), "checkpoint.pt")

        # Load on all ranks
        state_dict = torch.load("checkpoint.pt", map_location=device)
        model.module.load_state_dict(state_dict)
        dist.barrier()   # wait for all ranks to finish loading

    FSDP checkpoint (more complex):
        Use StateDictType.FULL_STATE_DICT: gather to rank 0, save once.
        Use StateDictType.LOCAL_STATE_DICT: each rank saves its own shard.
            → Faster save, requires same number of GPUs to resume.
        Use StateDictType.SHARDED_STATE_DICT: saves in distributed format.
            → Works with different GPU counts (most flexible).


##### PART 5 — DEEPSPEED AND MEGATRON-LM

### DeepSpeed ZeRO

    DeepSpeed (Microsoft) provides ZeRO stages 1, 2, and 3,
    plus additional optimisations:

    ZeRO-Infinity:
        Offload optimizer states, gradients, and parameters to CPU RAM or NVMe.
        Trade compute speed for almost unlimited model size.
        CPU offload: ≈2–4× slower than GPU-only, but enables 10× larger models.

    ZeRO++:
        Quantise AllReduce communications (BF16 → INT8 for AllReduce).
        Hierarchical AllReduce: sum within-node first (NVLink), then across nodes.
        4× fewer bytes communicated in data-parallel AllReduce.

    DeepSpeed configuration file (ds_config.json):
        {
          "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": true},
            "allgather_partitions": true,
            "reduce_scatter": true
          },
          "bf16": {"enabled": true},
          "gradient_clipping": 1.0,
          "train_micro_batch_size_per_gpu": 4,
          "gradient_accumulation_steps": 16
        }

    Usage:
        deepspeed --num_gpus=8 train.py --deepspeed ds_config.json

### Megatron-LM

    Megatron-LM (NVIDIA) is the gold-standard framework for training
    100B–1T parameter models. It implements 3D parallelism natively.

    Key features:
        Tensor parallelism (TP):   splits layers across GPUs
        Pipeline parallelism (PP): 1F1B interleaved schedule
        Data parallelism (DP):     with ZeRO-1 (optimizer state sharding)
        Sequence parallelism:      shards layer normalisations and dropouts
        Flash Attention:           memory-efficient attention built in
        Selective activation recomputation: checkpoints only expensive layers

    The Megatron training command:
        torchrun --nproc_per_node=8 pretrain_gpt.py \
            --tensor-model-parallel-size 4 \
            --pipeline-model-parallel-size 2 \
            --global-batch-size 2048 \
            --micro-batch-size 4 \
            --num-layers 32 \
            --hidden-size 4096 \
            --num-attention-heads 32 \
            --seq-length 4096 \
            --max-position-embeddings 4096 \
            --lr 3e-4 \
            --min-lr 3e-5 \
            --lr-warmup-iters 2000 \
            --train-iters 500000

    MFU achieved by Megatron on A100: 50–55% (world-class).
    How: careful overlap of communication with computation,
         custom CUDA kernels for each operation,
         hand-optimised all-reduce schedules.


##### PART 6 — MONITORING DISTRIBUTED TRAINING

### Metrics to Track

    PER-STEP METRICS (logged from rank 0):
        global_step:            training progress counter
        tokens_per_second:      throughput across ALL GPUs
        loss:                   should be identical across ranks (verify!)
        grad_norm:              post-clipping gradient norm
        learning_rate:          current LR from scheduler
        mfu:                    model FLOP utilisation

    PER-GPU METRICS (logged from EACH GPU):
        gpu_utilisation:        fraction of time GPU is computing
        gpu_memory_used:        HBM consumption in GB
        gpu_temperature:        thermal throttling indicator
        communication_time:     time spent in AllReduce (from profiler)

    COMMUNICATION METRICS:
        allreduce_time:         wall time for each AllReduce
        allreduce_bandwidth:    actual bandwidth achieved
        bubble_fraction:        (for PP) fraction of time idle

### Detecting Straggler GPUs

    In AllReduce: the SLOWEST GPU determines the step time.
    A straggler GPU = all other GPUs wait → throughput collapses.

    Signs of a straggler:
        GPU utilisation drops suddenly on one GPU (nvidia-smi)
        Step time increases while MFU decreases
        Other GPUs show high "sync time" (waiting at the allreduce barrier)

    Causes of straggler:
        Thermal throttling: GPU overheating → reduces clock speed
        Variable batch sizes: some batches have longer sequences (if not packed)
        Load imbalance in PP: some pipeline stages compute more than others
        Faulty hardware: NVLink or GPU hardware issue

    Detection:
        Log per-GPU compute time each step (not just global step time).
        Any GPU consistently 2×+ slower than others → investigate.

### Logging Infrastructure

    Weights & Biases (W&B): most popular in ML research.
        wandb.log({"loss": loss, "lr": lr, "mfu": mfu, "step": step})

    TensorBoard: standard in industry.
        writer.add_scalar("Loss/train", loss, global_step)

    Only rank 0 should log to avoid duplicate metrics:
        if rank == 0:
            wandb.log(metrics)

    Alert thresholds (set up automated alerts):
        loss > baseline × 1.2: loss diverging
        mfu < 0.20: throughput issue
        gpu_temp > 83°C: thermal risk
        grad_norm > 10: gradient explosion risk


##### PART 7 — TESTING A TRAINED LLM

### Evaluation Suite

    After training (or at checkpoints), run a standard evaluation suite:

    PERPLEXITY (language modelling):
        WikiText-103 (PTB): academic standard. Lower = better.
        Pile-CC holdout: evaluation on the same distribution as training.
        C4 holdout: alternative cross-distribution test.

    ZERO-SHOT BENCHMARKS (no examples in context):
        HellaSwag (commonsense): "Which sentence best completes the story?"
        PIQA (physical): "Which procedure achieves the goal?"
        Winogrande (coreference): "Which pronoun is correct?"
        ARC-Easy: grade school science multiple choice.
        ARC-Challenge: harder grade school science.
        Metric: accuracy (% correct over the test set).

    FEW-SHOT BENCHMARKS (N examples in context):
        MMLU (5-shot): 57 subjects, academic knowledge.
        BBH (3-shot): reasoning tasks.
        GSM8K (8-shot): math word problems.

    Running evaluations:
        lm-evaluation-harness (EleutherAI):
        lm_eval --model hf --model_args pretrained=./checkpoint \
                --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande \
                --device cuda:0 --batch_size 16

### Generation Quality Tests

    Beyond benchmarks, test generation quality manually:
        FLUENCY: does the model generate grammatical, coherent text?
        FACTUALITY: spot-check factual claims in generated text.
        INSTRUCTION FOLLOWING: can the model follow simple instructions?
        REPETITION: does the model get stuck in loops?

    Automated quality tests:
        MAUVE score: semantic similarity to human-written text.
        Self-BLEU: measures diversity (lower = more diverse = better).

### Calibration Check

    Is the model's confidence well-calibrated?
        When the model assigns 80% probability to a token, is it correct 80% of the time?
        Compute ECE (Expected Calibration Error) on a validation set.

### Red-Teaming and Safety Evaluation

    Before deployment:
        ToxiGen: test for generation of toxic/harmful content.
        TruthfulQA: test for factually correct responses.
        BBQ: bias benchmark for question answering.


##### PART 8 — DEBUGGING DISTRIBUTED TRAINING

### The Debugging Challenge

    Distributed bugs are SILENT. The job runs, loss decreases, no error message.
    But training is subtly wrong:
        - Some GPUs see the same data → effective batch size is wrong.
        - Gradient sync is missing for some parameters → model diverges across ranks.
        - Random seeds not set correctly → validation data contaminates training.
        - Pipeline parallel stages have a bug → incorrect activations propagate.

### Debugging Checklist

    1. VERIFY ALL RANKS HAVE THE SAME LOSS (after AllReduce):
        # After the global step, broadcast rank 0's loss and compare
        loss_tensor = torch.tensor(loss.item(), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        if abs(loss.item() - loss_tensor.item()) > 1e-3:
            print(f"Rank {rank}: LOCAL LOSS {loss.item()} != GLOBAL {loss_tensor.item()}")
        # If losses differ significantly: DDP sync may be broken

    2. VERIFY DATA IS DIFFERENT ACROSS RANKS:
        # Log the FIRST 5 token IDs from rank 0 and rank 1
        first_ids = input_ids[0, :5].tolist()
        print(f"Rank {rank}: first tokens = {first_ids}")
        # If rank 0 and rank 1 show the same tokens every step: sharding is broken

    3. VERIFY GRADIENT SYNC:
        # Check that a parameter's gradient is identical across all ranks
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.clone()
                dist.all_reduce(g, op=dist.ReduceOp.AVG)
                if not torch.allclose(param.grad, g, atol=1e-4):
                    print(f"Rank {rank}: {name} gradient NOT synced!")
        # In correct DDP: all ranks have identical gradients after AllReduce

    4. TEST SINGLE-RANK FIRST:
        Run with world_size=1 (single GPU). Verify loss curves match.
        Then scale to 2 GPUs and verify same loss at same step.
        Then scale to 8, 64, etc.

    5. USE find_unused_parameters=True:
        torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        This detects parameters with no gradient (common with conditional code).
        Performance overhead: only for debugging, disable in production.

    6. NCCL DEBUG LOGGING:
        NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL torchrun ... train.py
        Outputs every NCCL communication: ring setup, bandwidth, errors.
        Noisy but essential when debugging hangs or slow AllReduces.

### Common Distributed Bugs and Fixes

    BUG 1: DEADLOCK (training hangs forever at AllReduce)
        Cause: one process calls AllReduce at different point than others.
              E.g., different control flow based on local data.
        Fix: ensure all processes hit EXACTLY the same dist.* calls.
             Use dist.barrier() to synchronise before conditional code.

    BUG 2: LOSS DIVERGES ONLY WITH MULTIPLE GPUS
        Cause: gradient accumulation division is per-gpu instead of global.
        Example: loss /= local_accum_steps  (wrong)
                 loss /= (local_accum_steps * world_size)  (correct)
        Fix: effective batch size = micro_batch × accum_steps × world_size.
             Loss should be divided by the TOTAL batch size.

    BUG 3: DIFFERENT MODELS ON DIFFERENT RANKS
        Cause: weights initialised differently due to different random seeds.
        Fix: set the SAME seed on all ranks before model init.
             torch.manual_seed(42)  # call before model creation on all ranks.
             DDP init will broadcast from rank 0, but start identical.

    BUG 4: VALIDATION LOSS JUMPS AT CHECKPOINT RESUME
        Cause: random number generator state not saved in checkpoint.
        Fix: save and restore torch/numpy RNG states in checkpoint.
             checkpoint["rng_state"] = torch.get_rng_state()
             torch.set_rng_state(checkpoint["rng_state"])

    BUG 5: OOM ON SOME RANKS ONLY
        Cause: uneven data distribution (some ranks get longer sequences).
        Fix: use sequence packing to ensure all batches have same token count.
             Or: pad all sequences to the same length.

    BUG 6: SLOW TRAINING (LOW MFU)
        Causes and diagnostics:
            Check NCCL bandwidth: is AllReduce achieving expected GB/s?
            Check if data loading is the bottleneck: use num_workers > 0.
            Check GPU utilisation (nvidia-smi dmon): should be > 90%.
            Profile with nsys: find the slow kernels.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
GPU Interconnect Bandwidth Reference


| Technology         | Bandwidth (per GPU) | Latency  | Typical Use          |
|--------------------|---------------------|----------|----------------------|
| NVLink 4.0 (H100)  | 900 GB/s            | ~1 µs    | Within DGX node      |
| NVLink 3.0 (A100)  | 600 GB/s            | ~1 µs    | Within DGX node      |
| NVLink 2.0 (V100)  | 300 GB/s            | ~1 µs    | Within DGX node      |
| PCIe 5.0           | 128 GB/s            | ~5 µs    | Non-NVLink nodes     |
| InfiniBand NDR 400 | ~50 GB/s effective  | ~1 µs    | Cross-node (modern)  |
| InfiniBand HDR 200 | ~25 GB/s effective  | ~2 µs    | Cross-node (common)  |
| 100GbE Ethernet    | ~12 GB/s            | ~10 µs   | Cloud, budget setups |

"""


OPERATIONS = {

    "1 · Distributed Communication and DDP Training Loop": {
        "description": (
            "Implement AllReduce, AllGather, and ReduceScatter from first principles. "
            "Show the Ring-AllReduce algorithm step by step. "
            "Build a complete DDP training setup with torchrun launcher. "
            "Implement correct gradient accumulation in the distributed setting. "
            "Show how the DistributedSampler ensures correct data sharding. "
            "Verify that all ranks have identical losses and gradients."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time

print("=" * 65)
print("  DISTRIBUTED COMMUNICATION AND DDP TRAINING LOOP")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: AllReduce from scratch (Ring-AllReduce simulation)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Ring-AllReduce: the core distributed primitive")
print("━" * 65)
print()

print("  AllReduce: each GPU sends a tensor, all get the SUM.")
print("  Ring-AllReduce is bandwidth-optimal: O(2N) bytes per GPU.")
print()


class SimulatedGPU:
    """Simulates a GPU holding a gradient tensor."""
    def __init__(self, rank, world_size, data):
        self.rank       = rank
        self.world_size = world_size
        self.data       = np.array(data, dtype=np.float32)
        self.chunks     = np.split(self.data, world_size)  # divide into chunks

    def send_chunk(self, chunk_idx):
        return self.chunks[chunk_idx].copy()

    def receive_chunk(self, chunk_idx, chunk_data, op="add"):
        if op == "add":
            self.chunks[chunk_idx] += chunk_data
        elif op == "copy":
            self.chunks[chunk_idx] = chunk_data.copy()
        # Reconstruct data from chunks
        self.data = np.concatenate(self.chunks)


def ring_allreduce(gpus):
    """
    Simulate Ring-AllReduce.
    Phase 1: ReduceScatter — each GPU accumulates its chunk across the ring.
    Phase 2: AllGather    — each GPU broadcasts its completed chunk.
    """
    N = len(gpus)

    print(f"  Initial state ({N} GPUs, data size={len(gpus[0].data)}):")
    for gpu in gpus:
        print(f"    GPU {gpu.rank}: {gpu.data}")
    print()

    # Phase 1: ReduceScatter (N-1 steps)
    print("  Phase 1: ReduceScatter (accumulate chunks around the ring):")
    for step in range(N - 1):
        # Each GPU sends chunk (rank - step) % N to the NEXT GPU
        sends = {}
        for gpu in gpus:
            chunk_to_send = (gpu.rank - step) % N
            sends[gpu.rank] = (chunk_to_send, gpu.send_chunk(chunk_to_send))

        # Each GPU receives from the PREVIOUS GPU
        for gpu in gpus:
            src_rank = (gpu.rank - 1) % N
            chunk_idx, chunk_data = sends[src_rank]
            gpu.receive_chunk(chunk_idx, chunk_data, op="add")

        if step < 2:  # show first 2 steps
            print(f"  After ReduceScatter step {step+1}:")
            for gpu in gpus:
                print(f"    GPU {gpu.rank}: {gpu.data}")
    print()

    # After ReduceScatter: each GPU has the fully reduced value for ONE chunk
    # GPU i has the correct sum for chunk i
    print("  After ReduceScatter: each GPU has correct sum for its chunk only.")
    print("  GPU 0 chunk: correct sum  | Other chunks: intermediate values.")
    print()

    # Phase 2: AllGather (N-1 steps)
    print("  Phase 2: AllGather (broadcast completed chunks around the ring):")
    for step in range(N - 1):
        # Each GPU sends chunk (rank - step + 1) % N (the completed chunk)
        sends = {}
        for gpu in gpus:
            chunk_to_send = (gpu.rank - step + 1) % N
            sends[gpu.rank] = (chunk_to_send, gpu.send_chunk(chunk_to_send))

        for gpu in gpus:
            src_rank = (gpu.rank - 1) % N
            chunk_idx, chunk_data = sends[src_rank]
            gpu.receive_chunk(chunk_idx, chunk_data, op="copy")

    print("  Final state (all GPUs should have identical results):")
    for gpu in gpus:
        print(f"    GPU {gpu.rank}: {gpu.data}")

    # Verify
    expected = sum(g.data_original for g in gpus_with_orig)
    return gpus[0].data


# Create 4 simulated GPUs with different gradient values
np.random.seed(42)
n_gpus = 4
data_size = 8
gpu_data = [np.random.randint(1, 5, data_size).astype(np.float32) for _ in range(n_gpus)]
gpus = [SimulatedGPU(i, n_gpus, data_size * [float(i+1)]) for i in range(n_gpus)]
# Use simple integer data for clarity
for i, gpu in enumerate(gpus):
    gpu.data   = np.array([float(i+1)] * data_size)
    gpu.chunks = np.split(gpu.data.copy(), n_gpus)

# Save originals for verification
originals = [gpu.data.copy() for gpu in gpus]

result = ring_allreduce(gpus)

expected = np.sum(originals, axis=0)
print()
print(f"  Expected (sum of all GPU data): {expected}")
print(f"  Match: {np.allclose(gpus[0].data, expected)} ✅")
print()

# Bandwidth analysis
print("  Ring-AllReduce bandwidth analysis:")
print(f"  Data size per GPU: {data_size} elements = {data_size*4} bytes")
print(f"  Number of GPUs:    {n_gpus}")
print(f"  Steps:             {2*(n_gpus-1)} ({n_gpus-1} ReduceScatter + {n_gpus-1} AllGather)")
print(f"  Bytes transferred per GPU: {2*(n_gpus-1)/n_gpus * data_size * 4:.0f} bytes")
print(f"  vs naive broadcast: {data_size * 4 * (n_gpus-1):.0f} bytes")
print(f"  Ring is {(n_gpus-1) / (2*(n_gpus-1)/n_gpus):.1f}× more bandwidth efficient")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: DDP training loop
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — DDP training loop: complete implementation")
print("━" * 65)
print()

DDP_CODE = """
  COMPLETE DDP TRAINING SCRIPT (train_ddp.py):

  import os, torch, torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torch.utils.data.distributed import DistributedSampler
  from torch.amp import autocast, GradScaler

  def setup_distributed():
      rank       = int(os.environ["RANK"])
      local_rank = int(os.environ["LOCAL_RANK"])
      world_size = int(os.environ["WORLD_SIZE"])
      dist.init_process_group(backend="nccl")
      torch.cuda.set_device(local_rank)
      return rank, local_rank, world_size

  def cleanup():
      dist.destroy_process_group()

  def train():
      rank, local_rank, world_size = setup_distributed()
      device = torch.device(f"cuda:{local_rank}")

      # Model: SAME init on all ranks (same seed)
      torch.manual_seed(42)
      model = GPT(config).to(device)
      model = DDP(model, device_ids=[local_rank],
                  gradient_as_bucket_view=True)   # ← memory optimisation

      optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                     betas=(0.9, 0.95), weight_decay=0.1)
      scheduler = get_cosine_schedule_with_warmup(optimizer, ...)

      # Data: each rank gets a DIFFERENT SHARD
      dataset = TokenDataset("data/train.bin", block_size=config.seq_len)
      sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
      loader  = DataLoader(dataset, batch_size=MICRO_BATCH, sampler=sampler,
                            num_workers=4, pin_memory=True)

      ACCUM_STEPS = 8    # effective batch = MICRO_BATCH × ACCUM_STEPS × world_size

      for epoch in range(N_EPOCHS):
          sampler.set_epoch(epoch)    # ← CRITICAL: changes shuffle each epoch
          model.train()

          for step, (input_ids, labels) in enumerate(loader):
              input_ids = input_ids.to(device, non_blocking=True)
              labels    = labels.to(device, non_blocking=True)

              # Accumulation steps: DON'T sync on intermediate steps
              is_last_accum = (step + 1) % ACCUM_STEPS == 0

              context = model.no_sync if not is_last_accum else contextlib.nullcontext

              with context():   # skip AllReduce on intermediate accum steps
                  with autocast(device_type="cuda", dtype=torch.bfloat16):
                      logits = model(input_ids)
                      loss   = F.cross_entropy(logits.view(-1, V), labels.view(-1))
                      loss   = loss / ACCUM_STEPS
                  loss.backward()

              if is_last_accum:
                  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                  optimizer.step()
                  scheduler.step()
                  optimizer.zero_grad(set_to_none=True)

                  if rank == 0:     # only rank 0 logs
                      print(f"Step {step}: loss={loss.item()*ACCUM_STEPS:.4f}")

          dist.barrier()   # sync all ranks at epoch end

  if __name__ == "__main__":
      train()
      cleanup()

  LAUNCH COMMAND:
  # Single node, 8 GPUs:
  torchrun --nproc_per_node=8 train_ddp.py

  # Multi-node, 4 nodes × 8 GPUs = 32 GPUs:
  torchrun --nnodes=4 --nproc_per_node=8 \\
           --node_rank=0 \\               # (0,1,2,3 on each node)
           --master_addr=10.0.0.1 \\      # node 0's IP
           --master_port=29500 \\
           train_ddp.py
"""
print(DDP_CODE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Dataset sharding simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Dataset sharding: ensuring non-overlapping data")
print("━" * 65)
print()

class SimulatedDistributedSampler:
    """
    Simulates PyTorch's DistributedSampler.
    Ensures each rank sees a DISJOINT subset of the data.
    """
    def __init__(self, n_samples, num_replicas, rank, shuffle=True, seed=42):
        self.n_samples     = n_samples
        self.num_replicas  = num_replicas
        self.rank          = rank
        self.shuffle       = shuffle
        self.seed          = seed
        self.epoch         = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # Deterministic shuffle: same across all ranks (same seed)
        rng = np.random.RandomState(self.seed + self.epoch)
        if self.shuffle:
            indices = rng.permutation(self.n_samples).tolist()
        else:
            indices = list(range(self.n_samples))

        # Pad to ensure even division
        n_per_replica = math.ceil(self.n_samples / self.num_replicas)
        if len(indices) < n_per_replica * self.num_replicas:
            indices += indices[:(n_per_replica * self.num_replicas - len(indices))]

        # This rank's slice
        rank_indices = indices[self.rank::self.num_replicas]
        return iter(rank_indices[:n_per_replica])

    def __len__(self):
        return math.ceil(self.n_samples / self.num_replicas)


n_samples  = 20
world_size = 4
print(f"  Dataset: {n_samples} samples, world_size={world_size}")
print()

# Show how different ranks see different data
for epoch in range(2):
    print(f"  Epoch {epoch}:")
    all_rank_indices = []
    for rank in range(world_size):
        sampler = SimulatedDistributedSampler(n_samples, world_size, rank, seed=0)
        sampler.set_epoch(epoch)
        indices = list(sampler)
        all_rank_indices.extend(indices)
        print(f"    Rank {rank}: {indices}")

    # Verify no overlap
    all_orig = [i for i in all_rank_indices if i < n_samples]
    unique   = set(all_orig)
    covered  = len(unique)
    print(f"    Unique samples: {covered}/{n_samples}  "
          f"{'✅ full coverage' if covered == n_samples else '⚠️  not all covered'}")
    print()

print("  Key: sampler.set_epoch(epoch) changes the shuffle each epoch.")
print("  Without set_epoch: same data order every epoch → slow convergence.")
''',
    },

    "2 · FSDP, ZeRO Memory Analysis, and 3D Parallelism": {
        "description": (
            "Compare memory usage of DDP vs FSDP vs ZeRO stages. "
            "Show how FSDP AllGather+ReduceScatter works per layer. "
            "Compute memory per GPU under different parallelism configurations. "
            "Show the 3D parallelism (TP × PP × DP) hardware mapping. "
            "Implement tensor parallelism column/row split for a linear layer. "
            "Build the complete FSDP training code with checkpointing."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 65)
print("  FSDP, ZERO MEMORY ANALYSIS, AND 3D PARALLELISM")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Memory comparison: DDP vs ZeRO stages
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — DDP vs ZeRO-1/2/3: memory per GPU")
print("━" * 65)
print()

def memory_per_gpu(n_params, world_size, stage="ddp",
                   weight_bpe=2, grad_bpe=2, opt_bpe=8):
    """
    Compute memory per GPU for different parallelism strategies.
    opt_bpe=8: Adam FP32 = 4 bytes × 2 tensors (m + v)
    """
    GB = 1024**3

    weight_total = n_params * weight_bpe
    grad_total   = n_params * grad_bpe
    opt_total    = n_params * opt_bpe

    if stage == "ddp":
        # Full replication on each GPU
        return (weight_total + grad_total + opt_total) / GB

    elif stage == "zero1":
        # Optimizer state sharded, weights+grads replicated
        return (weight_total + grad_total + opt_total / world_size) / GB

    elif stage == "zero2":
        # Optimizer + gradients sharded, weights replicated
        return (weight_total + grad_total / world_size + opt_total / world_size) / GB

    elif stage == "zero3_fsdp":
        # Everything sharded (FSDP / ZeRO-3)
        # All three are divided by world_size
        return (weight_total + grad_total + opt_total) / world_size / GB

    elif stage == "tp":
        # Tensor parallel: weights sharded, one set of grads+optimizer per GPU
        tp_degree = world_size
        return (weight_total / tp_degree + grad_total / tp_degree +
                opt_total / tp_degree) / GB


n_params = 7e9   # LLaMA-2 7B
print(f"  LLaMA-2 7B ({n_params/1e9:.0f}B params), BF16 weights + FP32 Adam optimizer")
print()
print(f"  {'Strategy':<25} | {'World size':>11} | {'GB/GPU':>8} | "
      f"{'A100 80GB?':>12} | {'Notes'}")
print(f"  {'─'*72}")

configs = [
    ("DDP",           "ddp",          1,   "all 3 full"),
    ("DDP",           "ddp",          8,   "all 3 full"),
    ("ZeRO-1",        "zero1",        8,   "opt sharded"),
    ("ZeRO-2",        "zero2",        8,   "opt+grad sharded"),
    ("FSDP (ZeRO-3)", "zero3_fsdp",   8,   "all sharded"),
    ("FSDP (ZeRO-3)", "zero3_fsdp",  64,   "all sharded"),
    ("FSDP (ZeRO-3)", "zero3_fsdp", 256,   "all sharded"),
]

for name, stage, ws, note in configs:
    gb = memory_per_gpu(n_params, ws, stage)
    fits = "✅ fits" if gb < 70 else ("⚠️  tight" if gb < 80 else "❌ OOM")
    print(f"  {name:<25} | {ws:>11} | {gb:>8.1f} | {fits:>12} | {note}")

print()
# LLaMA-2 70B
print(f"  LLaMA-2 70B ({70}B params), BF16 weights + FP32 Adam:")
print()
for name, stage, ws, note in [
    ("DDP",           "ddp",          8,   "all 3 full"),
    ("FSDP",          "zero3_fsdp",   8,   "all sharded"),
    ("FSDP",          "zero3_fsdp",  64,   "all sharded"),
    ("FSDP",          "zero3_fsdp", 128,   "all sharded"),
]:
    gb = memory_per_gpu(70e9, ws, stage)
    fits = "✅ fits" if gb < 70 else ("⚠️  tight" if gb < 80 else "❌ OOM")
    print(f"  {name:<25} | {ws:>11} | {gb:>8.1f} | {fits:>12} | {note}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: FSDP forward pass mechanics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — FSDP mechanics: AllGather + ReduceScatter per layer")
print("━" * 65)
print()

print("  FSDP wraps each transformer layer independently.")
print("  For layer L, the sequence is:")
print()
FSDP_MECHANICS = """
  FORWARD PASS (layer L of N):
  ─────────────────────────────────────────────────────────────────────
  1. PRE-FORWARD AllGather:
     Each GPU holds shard W_L_rank (1/N_gpus of the layer's weights).
     AllGather reconstructs the FULL weight matrix W_L on all GPUs.
     Cost: W_L_size bytes transferred.
     After: all GPUs have W_L in GPU memory (temporarily).

  2. FORWARD COMPUTE:
     All GPUs compute the layer forward with the FULL W_L.
     (They process different input micro-batches or sub-batches)

  3. POST-FORWARD FREE:
     The AllGathered W_L is deleted from GPU memory.
     Each GPU reverts to holding only its W_L_rank shard.
     Peak memory for this layer is NOW released.

  BACKWARD PASS (layer L, reverse order):
  ─────────────────────────────────────────────────────────────────────
  4. PRE-BACKWARD AllGather (same as step 1):
     Reconstruct W_L for gradient computation.

  5. BACKWARD COMPUTE:
     Compute gradients ∂L/∂W_L and ∂L/∂input_L.
     Each GPU holds gradient w.r.t. its OWN portion of W_L.

  6. POST-BACKWARD ReduceScatter:
     ReduceScatter the gradients: each GPU gets the SUMMED gradient
     for its shard of W_L (just like ZeRO-2 on the gradient).
     Cost: W_L_size bytes transferred.
     Each GPU now holds the CORRECT gradient for its W_L_rank shard.

  7. POST-BACKWARD FREE:
     The AllGathered W_L is deleted from GPU memory.

  MEMORY PATTERN PER LAYER:
    Normal: W_L_shard (1/N of layer) — tiny
    During fwd/bwd: W_L_full (AllGather) — large but temporary
    Peak = normal + one full layer's weights at a time

  COMMUNICATION COST PER LAYER:
    Forward:  AllGather = W_L bytes
    Backward: AllGather + ReduceScatter = 2 × W_L bytes
    Total per layer: 3 × W_L bytes
    Total for N layers: 3 × sum(layer sizes) = 3 × model_size bytes

  For LLaMA-2 7B (14GB BF16):
    Per-layer weight: 14GB / 32 layers ≈ 437MB
    Per training step: 3 × 14GB = 42GB transferred across GPUs
    At NVLink speed (600 GB/s): ≈ 70ms overhead per step
"""
print(FSDP_MECHANICS)

FSDP_CODE = """
  FSDP SETUP CODE:

  from torch.distributed.fsdp import (
      FullyShardedDataParallel as FSDP,
      ShardingStrategy,
      MixedPrecision,
      BackwardPrefetch,
  )
  from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
  from functools import partial

  # Define wrapping policy: wrap each TransformerBlock separately
  wrap_policy = partial(
      transformer_auto_wrap_policy,
      transformer_layer_cls={TransformerBlock},   # your layer class
  )

  # Mixed precision: BF16 params, FP32 reduce
  mp = MixedPrecision(
      param_dtype  = torch.bfloat16,
      reduce_dtype = torch.float32,   # AllReduce in FP32 for accuracy
      buffer_dtype = torch.bfloat16,
  )

  # Wrap model
  model = FSDP(
      model,
      sharding_strategy = ShardingStrategy.FULL_SHARD,    # ZeRO-3
      auto_wrap_policy  = wrap_policy,
      mixed_precision   = mp,
      backward_prefetch = BackwardPrefetch.BACKWARD_PRE,  # prefetch next layer's
      device_id         = local_rank,                      # AllGather while computing
  )

  # SAVING checkpoint (gather all shards to rank 0):
  from torch.distributed.fsdp import StateDictType, FullStateDictConfig
  save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
      state_dict = model.state_dict()   # all shards gathered to rank 0
  if rank == 0:
      torch.save({"model": state_dict, "step": global_step}, "ckpt.pt")

  # LOADING checkpoint:
  with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
      ckpt = torch.load("ckpt.pt", map_location="cpu")
      model.load_state_dict(ckpt["model"])
"""
print(FSDP_CODE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Tensor Parallelism simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Tensor Parallelism: column/row linear layer split")
print("━" * 65)
print()

print("  For a linear layer Y = XA (X: batch×d, A: d×4d):")
print()

def tensor_parallel_linear(X, A, tp_degree):
    """
    Simulate tensor parallelism for a linear layer.
    Column-parallel: split A by output dimension across GPUs.
    Returns: (per-GPU partial outputs, full output after AllGather)
    """
    d_out = A.shape[1]
    chunk_size = d_out // tp_degree

    partial_outputs = []
    for gpu in range(tp_degree):
        # Each GPU holds a vertical slice of A (different output columns)
        A_shard = A[:, gpu*chunk_size:(gpu+1)*chunk_size]
        Y_partial = X @ A_shard    # (batch, chunk_size)
        partial_outputs.append(Y_partial)

    # AllGather: concatenate along the output dimension
    Y_full = np.concatenate(partial_outputs, axis=1)
    return partial_outputs, Y_full


np.random.seed(42)
batch_size = 4
d_in  = 8
d_out = 16
tp    = 4

X_tp = np.random.randn(batch_size, d_in).astype(np.float32)
A_tp = np.random.randn(d_in, d_out).astype(np.float32)

# Dense (non-parallel) baseline
Y_dense = X_tp @ A_tp

# Tensor-parallel
partials, Y_tp = tensor_parallel_linear(X_tp, A_tp, tp_degree=tp)

print(f"  X: ({batch_size}×{d_in}), A: ({d_in}×{d_out}), TP degree={tp}")
print()
print(f"  Each GPU holds A shard: ({d_in}×{d_out//tp}) = ({d_in}×{d_out//tp})")
print()
for i, p in enumerate(partials):
    print(f"    GPU {i}: Y partial shape = {p.shape}")
print()
print(f"  After AllGather: Y_full shape = {Y_tp.shape}")
print(f"  Matches dense: {np.allclose(Y_dense, Y_tp, atol=1e-5)} ✅")
print()

# Row-parallel (output projection)
print("  Row-parallel (for output projection B: 4d×d):")
print("  Each GPU holds A_shard rows (output from col-parallel) + B row shard.")

def row_parallel_linear(Y_shards, B_full, tp_degree):
    """
    Row-parallel: each GPU holds rows of B corresponding to its col of A.
    After computation: AllReduce to sum partial products.
    """
    d_out_full = B_full.shape[1]
    chunk_size = B_full.shape[0] // tp_degree

    partial_results = []
    for gpu in range(tp_degree):
        B_shard = B_full[gpu*chunk_size:(gpu+1)*chunk_size, :]
        partial  = Y_shards[gpu] @ B_shard   # (batch, d_out_full)
        partial_results.append(partial)

    # AllReduce: SUM partial products (all GPUs get the same final result)
    Z_full = sum(partial_results)
    return partial_results, Z_full


d_out2 = d_in  # FFN output back to model dim
B_tp   = np.random.randn(d_out, d_out2).astype(np.float32)
Z_dense = Y_dense @ B_tp
z_partials, Z_tp = row_parallel_linear(partials, B_tp, tp)

print(f"  Y partial (4 shards) @ B ({d_out}×{d_out2}) → sum → ({batch_size}×{d_out2})")
for i, z in enumerate(z_partials):
    print(f"    GPU {i}: partial ({z.shape}) — needs AllReduce to complete")
print(f"  After AllReduce (sum): Z_full = {Z_tp.shape}")
print(f"  Matches dense: {np.allclose(Z_dense, Z_tp, atol=1e-5)} ✅")
print()
print("  TP communication: 1 AllReduce per col-parallel + 1 per row-parallel")
print("  = 2 AllReduces per MLP (4 per transformer layer including attention)")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: 3D Parallelism hardware mapping
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 4 — 3D Parallelism: mapping to physical hardware")
print("━" * 65)
print()

def plan_3d_parallelism(n_params_b, tp, pp, target_gpu_mem_gb=70):
    """
    Plan 3D parallelism for a given model size and hardware.
    Returns DP degree and total GPU count.
    """
    # Memory per GPU: model/(TP*PP) + optimizer/(DP) + activations
    # Simplified model: weights + optimizer per GPU after 3D sharding
    weight_bpe = 2   # BF16
    opt_bpe    = 8   # FP32 Adam

    model_gb = n_params_b * 1e9 * weight_bpe / 1024**3
    opt_gb   = n_params_b * 1e9 * opt_bpe  / 1024**3

    # After TP×PP sharding: weight is divided by TP*PP
    # DP replicas each get 1/(TP*PP) of the model
    weight_per_gpu = model_gb / (tp * pp)
    opt_per_gpu    = opt_gb   / (tp * pp)   # optimizer sharded by dp group
    act_per_gpu    = 3.0       # rough estimate for activations (GB)

    total_per_gpu  = weight_per_gpu + opt_per_gpu + act_per_gpu

    if total_per_gpu > target_gpu_mem_gb:
        dp = 0  # doesn't fit
    else:
        dp = 1  # minimum; increase for more throughput

    return total_per_gpu, dp


print("  3D Parallelism planning for different model sizes:")
print(f"  (Target: fit within 70GB of 80GB A100, remaining for KV cache)")
print()
print(f"  {'Model':>15} | {'TP':>3} | {'PP':>3} | {'DP':>3} | "
      f"{'Total GPUs':>11} | {'GB/GPU':>8} | {'Fits?'}")
print(f"  {'─'*65}")

plans = [
    ("LLaMA-2 7B",    7,   1, 1),
    ("LLaMA-2 7B",    7,   1, 8),   # data parallel replicas
    ("LLaMA-2 13B",  13,   2, 1),
    ("LLaMA-2 13B",  13,   2, 8),
    ("LLaMA-2 70B",  70,   8, 4),
    ("LLaMA-2 70B",  70,   8, 8),   # 256 GPUs
    ("GPT-3 175B",  175,   8, 8),
    ("GPT-3 175B",  175,   8, 16),
    ("GPT-4 ~1T",  1000,   8, 16),
]

for name, params_b, tp, pp in plans:
    # Determine DP from throughput needs (here just show 1 replica)
    gb, _ = plan_3d_parallelism(params_b, tp, pp)
    dp = 8  # typical data parallel degree for throughput
    total = tp * pp * dp
    fits = "✅" if gb <= 70 else "❌"
    print(f"  {name:>15} | {tp:>3} | {pp:>3} | {dp:>3} | "
          f"{total:>11,} | {gb:>8.1f} | {fits}")

print()
print("  Physical mapping:")
print("    TP dimension: within ONE DGX node (NVLink, 600 GB/s)")
print("    PP dimension: across NODES via InfiniBand (25-50 GB/s)")
print("    DP dimension: across independent PP×TP groups (IB AllReduce)")
print()
print("  Rule: TP ≤ GPUs per node (8 for DGX). PP and DP can span nodes.")
''',
    },

    "3 · Debugging Distributed Training and Post-Training Evaluation": {
        "description": (
            "Systematic debugging of common distributed training failures. "
            "Verify gradient sync, data sharding, and loss consistency across ranks. "
            "Implement a training health monitor for multi-GPU runs. "
            "Show how to run lm-evaluation-harness for benchmark evaluation. "
            "Build the complete evaluation pipeline: perplexity + zero-shot tasks. "
            "Demonstrate how to detect and fix the most common distributed bugs."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time

print("=" * 65)
print("  DEBUGGING DISTRIBUTED TRAINING AND EVALUATION")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Distributed debugging toolkit
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Distributed debugging: common bugs and fixes")
print("━" * 65)
print()

BUGS_AND_FIXES = """
  THE 6 MOST COMMON DISTRIBUTED TRAINING BUGS:

  ═══════════════════════════════════════════════════════════════════════
  BUG 1: DEADLOCK (training hangs at AllReduce)
  ═══════════════════════════════════════════════════════════════════════
  Symptom: job launches, loss prints once or twice, then FREEZES forever.
  Cause:   Different ranks call dist.* at different points.
           rank 0 calls all_reduce(A), rank 1 calls all_reduce(B)
           → they wait for each other forever.

  Diagnosis:
    Set NCCL_DEBUG=INFO and look for "Waiting for group" in logs.
    Add print statements with rank numbers:
        print(f"Rank {rank}: about to AllReduce at step {step}")

  Fix:
    Ensure all ranks execute exactly the same dist.* calls.
    Use dist.barrier() before any conditional communication:
        dist.barrier()  # sync all ranks
        if some_condition:
            dist.all_reduce(tensor)  # all ranks must enter this together

  ═══════════════════════════════════════════════════════════════════════
  BUG 2: LOSS DIVERGES AFTER ~1000 STEPS WITH MULTIPLE GPUS
  ═══════════════════════════════════════════════════════════════════════
  Symptom: loss decreases normally for 1000 steps, then explodes.
           Works fine on 1 GPU.
  Cause:   Gradient accumulation division is wrong in multi-GPU setting.

  WRONG:
    loss = loss / local_accum_steps   ← ignores world_size
    # Each GPU applies LR to gradients that are N× too large

  CORRECT:
    loss = loss / (local_accum_steps * world_size)
    # OR: let DDP handle it (DDP averages gradients automatically)
    # With DDP: just divide by local_accum_steps (DDP averages the rest)

  The rule: DDP AllReduces are AVERAGES (not sums) by default.
  So: each GPU's gradient is already (sum / N). Just divide by accum_steps.

  Verification:
    torch.distributed.all_reduce(loss, op=dist.ReduceOp.AVG)
    → if this differs from your logged loss by > 0.01, something is wrong.

  ═══════════════════════════════════════════════════════════════════════
  BUG 3: TRAINING LOOKS FINE BUT MODEL IS MEDIOCRE
  ═══════════════════════════════════════════════════════════════════════
  Symptom: loss curve looks reasonable. But evaluated model is worse
           than expected for the training budget.
  Cause:   All ranks are seeing the SAME data (data sharding broken).
           Effective batch size is N× smaller than believed.

  Diagnosis:
    Log first 5 tokens of each batch on each rank:
        print(f"Rank {rank} step {step}: tokens={input_ids[0,:5].tolist()}")
    If all ranks show the same tokens: sampler is not distributing correctly.

  Fix:
    Use DistributedSampler with sampler.set_epoch(epoch) every epoch.
    Verify: shuffle with seed=global_seed+epoch so all ranks agree.
    Verify: each rank takes every Nth sample, not a contiguous block.

  ═══════════════════════════════════════════════════════════════════════
  BUG 4: CHECKPOINT RESUME JUMPS IN LOSS
  ═══════════════════════════════════════════════════════════════════════
  Symptom: training resumes from checkpoint but loss immediately jumps
           and takes 100s of steps to recover.
  Cause:   Random number generator states not saved/restored.
           On resume: different data ordering than original run.
           Adam moments are saved but don't match the resumed data.

  Fix:
    Save AND restore RNG states:
        # Save:
        checkpoint["rng"]     = torch.get_rng_state()
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state()
        checkpoint["np_rng"]  = np.random.get_state()

        # Restore:
        torch.set_rng_state(ckpt["rng"])
        torch.cuda.set_rng_state(ckpt["cuda_rng"])
        np.random.set_state(ckpt["np_rng"])

  ═══════════════════════════════════════════════════════════════════════
  BUG 5: OOM ON SOME RANKS ONLY
  ═══════════════════════════════════════════════════════════════════════
  Symptom: rank 2 crashes with CUDA OOM; other ranks continue.
           Always the same rank that crashes.
  Cause:   Uneven batch sizes across ranks. Rank 2 gets longer sequences.

  Diagnosis:
    Log sequence lengths per rank: print(f"Rank {rank}: seq_len={T}")

  Fix:
    Use sequence packing: all chunks are exactly context_length tokens.
    Ensure dataset.chunks has all items of equal length.
    If using variable-length data: sort by length and group similar lengths.

  ═══════════════════════════════════════════════════════════════════════
  BUG 6: MYSTERIOUSLY WRONG RESULTS AFTER CHANGING WORLD_SIZE
  ═══════════════════════════════════════════════════════════════════════
  Symptom: resuming training but with 16 GPUs instead of 8.
           Loss curve behaves differently than expected.
  Cause:   The global batch size changed (8 GPUs × batch=4 → 16 × batch=4).
           The learning rate is now calibrated for the wrong batch size.

  Fix:
    When scaling world_size, scale LR proportionally (linear scaling rule):
        new_lr = base_lr × (new_global_batch / original_global_batch)
    Or: keep the same global batch by halving per-GPU batch size.
        Always define training in terms of GLOBAL batch, not per-GPU batch.
"""
print(BUGS_AND_FIXES)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Verification tests
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Verification tests for distributed training")
print("━" * 65)
print()

VERIFICATION_CODE = """
  VERIFICATION TEST SUITE (run at training start):

  def verify_distributed_setup(model, dataloader, rank, world_size):
      """Run these checks at step 0 before full training."""

      print(f"  Running distributed verification (rank {rank})...")

      # TEST 1: All ranks have identical model weights
      for name, param in model.named_parameters():
          param_clone = param.clone()
          dist.broadcast(param_clone, src=0)
          if not torch.allclose(param, param_clone, atol=1e-4):
              print(f"  FAIL: Rank {rank} has different {name} than rank 0!")
              return False
      print(f"  [rank {rank}] ✅ Model weights identical across all ranks")

      # TEST 2: Data shards are different
      batch = next(iter(dataloader))
      first_token = batch['input_ids'][0, 0].item()
      tokens = torch.tensor([first_token], device=f"cuda:{rank%8}")
      all_tokens = [torch.zeros(1, device=f"cuda:{rank%8}") for _ in range(world_size)]
      dist.all_gather(all_tokens, tokens)
      if len(set(t.item() for t in all_tokens)) < world_size // 2:
          print(f"  WARN: Many ranks see same first token. Check data sharding!")
      else:
          print(f"  [rank {rank}] ✅ Data shards appear diverse across ranks")

      # TEST 3: Loss is equal after backward + AllReduce
      input_ids = batch['input_ids'].cuda()
      labels    = batch['labels'].cuda()
      with torch.no_grad():
          logits = model(input_ids)
          loss   = F.cross_entropy(logits.view(-1, V), labels.view(-1))

      loss_mean = loss.clone()
      dist.all_reduce(loss_mean, op=dist.ReduceOp.AVG)
      if abs(loss.item() - loss_mean.item()) > 0.1:
          print(f"  WARN: Rank {rank} loss {loss.item():.4f} != global {loss_mean.item():.4f}")
          print(f"        Data distribution might be very different across ranks.")
      else:
          print(f"  [rank {rank}] ✅ Loss consistent across ranks")

      # TEST 4: Gradient sync works
      loss.backward()
      for name, param in list(model.named_parameters())[:3]:  # check first 3 params
          if param.grad is not None:
              g = param.grad.clone()
              dist.all_reduce(g, op=dist.ReduceOp.AVG)
              if not torch.allclose(param.grad, g, atol=1e-3):
                  print(f"  FAIL: Rank {rank} gradient {name} not synced!")
                  return False
      print(f"  [rank {rank}] ✅ Gradients properly synced via AllReduce")

      return True
"""
print(VERIFICATION_CODE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Post-training evaluation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Post-training evaluation suite")
print("━" * 65)
print()

# Simulate evaluation results across training checkpoints
print("  Simulated evaluation results (loss + benchmark accuracy):")
print()
print(f"  {'Checkpoint':>15} | {'Train Loss':>12} | {'Val PPL':>9} | "
      f"{'HellaSwag':>11} | {'PIQA':>7} | {'Arc-E':>7} | {'MMLU-5shot':>12}")
print(f"  {'─'*82}")

np.random.seed(1)
checkpoints = [
    (0,       10.40, 100.0,  0.25, 0.50, 0.25, 0.23),
    (5000,     5.20,  181.3,  0.35, 0.55, 0.30, 0.25),
    (10000,    3.80,   44.7,  0.45, 0.62, 0.38, 0.27),
    (50000,    2.80,   16.4,  0.58, 0.70, 0.48, 0.30),
    (100000,   2.50,   12.2,  0.64, 0.73, 0.54, 0.33),
    (250000,   2.25,    9.5,  0.70, 0.77, 0.61, 0.38),
    (500000,   2.08,    8.0,  0.75, 0.79, 0.66, 0.44),
    ("LLaMA-2 7B (ref)", 2.05, 7.7, 0.78, 0.78, 0.69, 0.46),
]

for ckpt, train_loss, val_ppl, hs, piqa, arce, mmlu in checkpoints:
    ckpt_str = str(ckpt) if isinstance(ckpt, int) else ckpt
    print(f"  {ckpt_str:>15} | {train_loss:>12.4f} | {val_ppl:>9.1f} | "
          f"{hs:>10.1%} | {piqa:>6.1%} | {arce:>6.1%} | {mmlu:>11.1%}")
print()
print("  Key observations:")
print("  • Early training (step 5k): random-chance benchmark scores")
print("  • Mid training (step 50k): benchmarks climb as language structure learned")
print("  • Late training: consistent improvement, approaching published numbers")
print("  • MMLU is harder (knowledge-intensive), improves slower than perceptual tasks")
print()

# Evaluation commands
EVAL_COMMANDS = """
  POST-TRAINING EVALUATION COMMANDS:

  # 1. Perplexity on standard datasets
  lm_eval --model hf \\
          --model_args pretrained=./checkpoint,dtype=bfloat16 \\
          --tasks wikitext \\
          --device cuda:0 --batch_size 16

  # 2. Zero-shot benchmarks (standard suite)
  lm_eval --model hf \\
          --model_args pretrained=./checkpoint,dtype=bfloat16 \\
          --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande \\
          --device cuda:0 --batch_size 16 \\
          --output_path ./eval_results/

  # 3. Few-shot benchmarks (MMLU, 5-shot)
  lm_eval --model hf \\
          --model_args pretrained=./checkpoint,dtype=bfloat16 \\
          --tasks mmlu \\
          --num_fewshot 5 \\
          --device cuda:0 --batch_size 8

  # 4. Math reasoning (GSM8K, 8-shot)
  lm_eval --model hf \\
          --model_args pretrained=./checkpoint,dtype=bfloat16 \\
          --tasks gsm8k \\
          --num_fewshot 8 \\
          --device cuda:0 --batch_size 4

  # 5. Multi-GPU evaluation (8 GPUs with data parallel)
  torchrun --nproc_per_node=8 \\
      -m lm_eval --model hf \\
      --model_args pretrained=./checkpoint,dtype=bfloat16 \\
      --tasks hellaswag,mmlu \\
      --batch_size 32

  EXPECTED TIMINGS (LLaMA-2 7B, A100 80GB):
    HellaSwag (10k samples):  ~5 min
    MMLU (14k samples):       ~15 min
    GSM8K (1.3k samples):     ~3 min
    Full eval suite:          ~1 hour
"""
print(EVAL_COMMANDS)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Complete distributed training recipe
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Complete multi-GPU training recipe")
print("━" * 65)
print()

RECIPE = """
  MULTI-GPU LLM TRAINING: END-TO-END RECIPE

  INFRASTRUCTURE (before writing any code):
  □ Choose parallelism strategy:
      Fits 1 GPU:    DDP across all GPUs (simplest)
      Doesn't fit:   FSDP or TP+PP+DP (3D)
  □ Hardware: NVLink within node (TP), InfiniBand across nodes (DP/PP)
  □ Storage: fast NVMe or parallel filesystem (Lustre, GPFS) for checkpoints

  DATA PREPARATION (before training starts):
  □ Tokenise entire corpus → binary files of uint16 token IDs
  □ Split into train / validation (95/5 or fixed val set)
  □ Verify: no train/val overlap (deduplication)
  □ Compute expected tokens/step for your batch configuration

  BEFORE THE FIRST FULL RUN:
  □ Overfit test: 1 GPU, 10 batches, loss → 0 within 500 steps
  □ Memory test: 8 GPUs, 100 steps, verify no OOM
  □ Distributed test: 8 GPUs, 100 steps, loss matches 1-GPU reference
  □ Throughput test: measure tokens/second, compute MFU

  TRAINING LOOP CONFIGURATION:
  □ Warmup steps: 1–2% of total steps
  □ Peak LR: 1e-4 to 3e-4 (smaller for larger models)
  □ Min LR: 10% of peak (cosine decay target)
  □ Gradient clip: max_norm=1.0
  □ Weight decay: 0.1 (AdamW decoupled)
  □ Global batch: 2M–4M tokens (standard for 7B scale)
  □ Sequence length: 2048–4096

  MONITORING (during training):
  □ Log: loss, grad_norm, lr, mfu, tokens/sec — every 10 steps
  □ Evaluate: val_ppl + zero-shot benchmarks — every 5000 steps
  □ Save checkpoint: every 1000 steps, keep last 5
  □ Alert: if loss > 1.2× baseline, or MFU < 20%

  COMMON ADJUSTMENTS MID-TRAINING:
    Loss spike:       Load prev checkpoint, reduce LR 2×
    OOM suddenly:     Reduce seq_len, add grad checkpoint
    MFU drops:        Check GPU utilisation — data loading bottleneck?
    Val loss rises:   Inspect data — possible contamination or shift
    Slow progress:    Check data quality — might need to increase LR

  AFTER TRAINING:
  □ Evaluate full benchmark suite (lm-evaluation-harness)
  □ Human evaluation on generation quality
  □ Safety evaluation (ToxiGen, TruthfulQA)
  □ Convert to inference format (convert HF → vLLM / TRT-LLM)
  □ Calibrate for INT8/FP8 if deploying with quantisation
  □ Document: training config, data mixture, eval results (model card)
"""
print(RECIPE)

print("━" * 65)
print("  MULTI-GPU QUICK REFERENCE")
print("━" * 65)
print()
print("  ┌───────────────────────────────────────────────────────────────┐")
print("  │ Question                        │ Answer                      │")
print("  ├───────────────────────────────────────────────────────────────┤")
print("  │ Model fits 1 GPU?               │ DDP (simplest)              │")
print("  │ Model needs N GPUs?             │ FSDP or TP+PP               │")
print("  │ TP communication speed?         │ NVLink (within node)        │")
print("  │ PP/DP communication speed?      │ InfiniBand (across nodes)   │")
print("  │ Save checkpoint (DDP)?          │ rank 0 saves, barrier       │")
print("  │ Save checkpoint (FSDP)?         │ FULL_STATE_DICT + rank0     │")
print("  │ Data sharding?                  │ DistributedSampler+set_epoch│")
print("  │ Gradient accumulation (DDP)?    │ model.no_sync() + accum     │")
print("  │ First sign of data shard bug?   │ All ranks same tokens       │")
print("  │ First sign of deadlock?         │ Hangs silently at step 0    │")
print("  │ First sign of wrong LR scale?   │ Loss spikes at step ~1000   │")
print("  └───────────────────────────────────────────────────────────────┘")
''',
    },
}

for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }