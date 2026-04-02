"""
Multi-GPU Training & Heterogeneous GPU Strategies
===================================================

Training large models requires distributing work across multiple GPUs.
Understanding how that distribution works — and what goes wrong when
those GPUs are not identical — is essential for anyone building or
debugging large-scale training pipelines.

"""

import textwrap
import re

TOPIC_NAME   = "Multi-GPU Training & Heterogeneous GPU Strategies"
DISPLAY_NAME = "17b · Multi-GPU Training"
ICON         = "🖥️"
SUBTITLE     = "Data Parallel · Tensor Parallel · Pipeline Parallel · ZeRO · Heterogeneous GPUs"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY MULTI-GPU TRAINING

### Scaling Laws and the Compute Demand

Kaplan et al. (2020) showed that model performance improves smoothly
as a power law of:
    - Number of parameters N
    - Dataset size D
    - Compute budget C  (measured in FLOPs)

Optimal compute allocation: C ≈ 6 × N × D  (Chinchilla scaling law)

For a 70B parameter model trained on 2T tokens (LLaMA 2):
    C = 6 × 70×10⁹ × 2×10¹² = 8.4×10²³ FLOPs

An H100 GPU delivers roughly 3×10¹⁵ FLOPs/s (BF16):
    Time on 1 GPU = 8.4×10²³ / 3×10¹⁵ ≈ 280,000 seconds ≈ 3.2 years

LLaMA 2 was trained on 2,000 GPUs for ~3 weeks.
Multi-GPU training is not optional at this scale — it is the only way.


### The Memory Wall

Before compute time, there is a hard memory constraint.
A model with P parameters requires:

    Parameters alone:         P × dtype_size
    Gradients (same shape):   P × dtype_size
    Optimiser states (AdamW): P × 8 bytes  (2 FP32 values: m and v)
    Activations (training):   scales with batch size and model depth

    Total for FP16/BF16 mixed precision (Module 15):
    Parameters:     2 bytes/param  (FP16)
    Gradients:      2 bytes/param  (FP16)
    Master params:  4 bytes/param (FP32 copy for optimiser)
    Optimiser:      8 bytes/param  (FP32 m + v)
    ─────────────────────────────────────────────────────── 
    Total:          16 bytes/param  (model states only)

    For a 70B model: 70×10⁹ × 16 = 1,120 GB just for model states.
    An A100 has 80 GB of VRAM. You need at least 14 A100s just for states,
    before counting activations.

    This is why distributing the model across GPUs is necessary.


### The Three Axes of Parallelism

    Every multi-GPU strategy falls on one or more of three axes:

    AXIS 1 — DATA PARALLELISM (DP):
        Each GPU holds the FULL MODEL but processes a DIFFERENT MINI-BATCH.
        Gradients are averaged across GPUs after each backward pass.
        Works when the model fits on a single GPU.
        Scales well: adding more GPUs directly increases batch throughput.

    AXIS 2 — TENSOR PARALLELISM (TP) / MODEL PARALLELISM:
        Each GPU holds a DIFFERENT PORTION OF THE WEIGHTS.
        All GPUs see the SAME MINI-BATCH but compute different parts.
        Necessary when the model is too large for a single GPU.
        Requires frequent inter-GPU communication within each layer.

    AXIS 3 — PIPELINE PARALLELISM (PP):
        Each GPU holds DIFFERENT LAYERS of the model.
        Data flows through GPUs sequentially (like an assembly line).
        Reduces per-GPU memory but introduces pipeline idle time ("bubbles").

    Diagram 1 — The Three Axes of Parallelism:

    DATA PARALLEL (4 GPUs):      TENSOR PARALLEL (4 GPUs):
    ┌──────┐ ┌──────┐            ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
    │Full  │ │Full  │            │W1/4 │ │W2/4 │ │W3/4 │ │W4/4 │
    │model │ │model │            │     │ │     │ │     │ │     │
    │Batch │ │Batch │            │Same │ │Same │ │Same │ │Same │
    │ 1    │ │ 2    │            │batch│ │batch│ │batch│ │batch│
    └──────┘ └──────┘            └─────┘ └─────┘ └─────┘ └─────┘
    Different data, same model.  Same data, different weights.

    PIPELINE PARALLEL (4 GPUs):
    ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
    │Layers  │→ │Layers  │→ │Layers  │→ │Layers  │
    │ 1-6    │  │ 7-12   │  │ 13-18  │  │ 19-24  │
    └────────┘  └────────┘  └────────┘  └────────┘
    Data flows left to right through the pipeline.


##### PART 2 — DATA PARALLELISM

### Core Concept

In data parallelism, N GPUs each hold an identical copy of the full model.
The training dataset is split into N non-overlapping shards.
Each GPU processes its own shard independently during the forward pass.

After backward, each GPU has a LOCAL gradient tensor that reflects
only its own data shard. To compute the GLOBAL gradient (what you would
get from the full batch), all gradients must be averaged across GPUs.

Only the model weights need to stay synchronised — activations are
local to each GPU and are discarded after the backward pass.


### Naive Data Parallelism: The Parameter Server

    Early approach (2012-2018, used by DistBelief, early TensorFlow):

    Diagram 2 — Parameter Server Architecture:

            ┌──────────────────────────────────┐
            │       PARAMETER SERVER           │
            │  holds authoritative copy of W   │
            └────┬──────┬──────┬──────┬────────┘
                 │      │      │      │
              PUSH    PUSH    PULL   PULL
           (grads) (grads)     (W)   (W)
                 │      │      │      │
            ┌────┘  ┌───┘  ┌───┘   ┌──┘
            ▼       ▼      ▼       ▼
         [GPU 0] [GPU 1] [GPU 2] [GPU 3]
         Mini-  Mini-   Mini-   Mini-
         batch1 batch2  batch3  batch4

    Bottleneck: the parameter server becomes a communication choke point.
    All N GPUs send gradients TO and receive weights FROM one server.
    Bandwidth at the server scales as O(N). At N=100 GPUs, the server
    is saturated and all GPUs wait for it.

    Additionally: gradients arrive at different times (asynchronous),
    so different GPUs may train on STALE weights — hurting convergence.


### Ring-AllReduce: The Scalable Alternative

    AllReduce goal: every GPU ends up with the SUM (or MEAN) of all
    gradients from all GPUs, without a central bottleneck.

    Ring topology: GPUs are arranged in a logical ring.
    Each GPU sends to its right neighbour, receives from its left neighbour.

    Diagram 3 — Ring-AllReduce with 4 GPUs:

    GPU 0 ──sends──► GPU 1 ──sends──► GPU 2 ──sends──► GPU 3
      ▲                                                    │
      └────────────────receives────────────────────────────┘

    The algorithm runs in two phases:

    PHASE 1 — Reduce-Scatter (N-1 steps):
        Each GPU starts with its full gradient tensor, split into N chunks.
        After N-1 steps, each GPU holds one chunk that is the SUM of that
        chunk across all GPUs. Different GPUs hold different chunks.

    PHASE 2 — AllGather (N-1 steps):
        Each GPU has the "fully reduced" chunk for its portion.
        In N-1 more steps, each GPU sends its chunk around the ring
        until every GPU has assembled the full reduced gradient.

    Total data transferred per GPU: 2(N-1)/N × G
        where G = total gradient size
    As N → ∞: approaches 2G regardless of N → bandwidth-optimal!

    Compare to Parameter Server: each GPU transfers G/N per step
    but the server must receive N × G/N = G from all GPUs simultaneously
    → server bandwidth bottleneck grows with N.


### PyTorch DDP: How It Works Internally

    PyTorch's DistributedDataParallel (DDP) is the standard data
    parallel implementation. Its key design choices:

    1. GRADIENT BUCKETS:
       Gradients are not communicated one-by-one as they are computed.
       Instead, gradients are accumulated into "buckets" of fixed size
       (default 25 MB). When a bucket is full, AllReduce is launched for
       that bucket. This amortises the per-communication overhead.

    2. OVERLAP OF COMPUTE AND COMMUNICATION:
       DDP uses PyTorch's autograd hooks to start AllReduce for layer L
       WHILE the backward pass is still computing gradients for earlier
       layers (L-1, L-2, ...).

       Diagram 4 — Compute-Communication Overlap in DDP:

       Time  ──────────────────────────────────────────────►
       GPU:  [Backward L24][Backward L23][Backward L22]...
       Net:          [AllReduce bucket 3][AllReduce bucket 2]...

       Without overlap: must finish ALL backward before starting communication.
       With overlap: communication for later layers happens while
       computing backward for earlier layers.
       This hides most of the communication latency.

    3. GRADIENT AVERAGING:
       After AllReduce, each GPU divides the summed gradient by N to get
       the mean. This is equivalent to computing the gradient on a batch
       N× larger (the full effective batch). Learning rate should scale
       accordingly (linear scaling rule, see Module 15).

    4. SYNCHRONISATION ACROSS RANKS:
       DDP ensures all GPUs start each iteration with identical weights
       by broadcasting from rank 0 at initialisation. From that point,
       since all GPUs compute the same AllReduce and apply the same
       optimiser update, weights stay in sync.


### Batch Size Scaling with Data Parallelism

    N GPUs each process a micro-batch of size B.
    Effective batch size = N × B.

    The gradient is the average over the effective batch:
    g_effective = (1/N) Σᵢ gᵢ  where each gᵢ is the gradient on micro-batch i.

    This is mathematically equivalent to computing the gradient on a
    single batch of size N × B.

    Learning rate scaling (Goyal et al., 2017):
        Linear rule:    lr_new = lr_base × N       (for moderate N, up to ~32)
        Sqrt rule:      lr_new = lr_base × sqrt(N)  (more conservative for large N)
        Warmup required: use lower lr for first 5-10% of training to allow
                         the large effective batch to stabilise.


##### PART 3 — TENSOR PARALLELISM (MODEL PARALLELISM)

### When Data Parallelism Is Not Enough

    Data parallelism requires the FULL MODEL to fit on each GPU.
    For GPT-3 (175B params): 175×10⁹ × 16 bytes ≈ 2,800 GB of model states.
    An A100 has 80 GB. Even with ZeRO (which we cover later), a 175B model
    is extremely difficult to fit on a small number of GPUs.

    Tensor parallelism distributes the model itself: each GPU holds
    only a SLICE of each weight matrix. All GPUs cooperate to compute
    the output of every layer on the same input.


### Column-Wise and Row-Wise Weight Splitting

    Consider a linear layer: Y = X @ W + b

    COLUMN-WISE SPLIT (split output dimension):
        W is split into [W₁ | W₂ | W₃ | W₄] by columns.
        GPU k computes Yₖ = X @ Wₖ  (partial output, full input)
        Each GPU sees the FULL input X — it must be broadcast first.
        Each GPU produces a PARTIAL output (different columns of Y).

    ROW-WISE SPLIT (split input dimension):
        W is split into [W₁; W₂; W₃; W₄] by rows.
        GPU k receives a PARTIAL input Xₖ (already split from previous layer).
        GPU k computes partial output: Yₖ = Xₖ @ Wₖ
        Final output: Y = Y₁ + Y₂ + Y₃ + Y₄  (AllReduce to sum partials)

    Diagram 5 — Tensor Parallel MLP (Megatron-LM style):

    Input X (full, on all GPUs)
         │
    ┌────┴────┐  Column-wise split
    │ GPU 0   │  GPU 1   GPU 2   GPU 3
    │ W[:,:d/4] W[:,d/4:d/2] ...
    │         │
    ▼         ▼
    Y₀(partial) Y₁(partial)  Y₂    Y₃   (no communication yet — GeLU local)
         │
    ┌────┴────┐  Row-wise split of next weight
    │ GPU 0   │  GPU 1   GPU 2   GPU 3
    │ W[:d/4,:] W[d/4:d/2,:] ...
    │         │
    ▼         ▼
    Z₀(partial) Z₁(partial)  Z₂    Z₃
         │
    AllReduce (sum all partials)
         │
    Z (full output, identical on all GPUs)

    Communication: only ONE AllReduce per transformer block (per MLP).
    Compare to naive model parallel: communication at EVERY layer boundary.


### Megatron-LM: Splitting Attention Heads

    In a multi-head attention layer with H heads:
    With T GPUs, assign H/T heads to each GPU.

    GPU k computes:
        Qₖ = X @ WQₖ   (k-th slice of query projection)
        Kₖ = X @ WKₖ   (k-th slice of key projection)
        Vₖ = X @ WVₖ   (k-th slice of value projection)
        Attn_k = softmax(QₖKₖᵀ/√d) @ Vₖ   (local attention for H/T heads)
        Outₖ = Attn_k @ WOₖ   (k-th slice of output projection)

    Then AllReduce over all GPU outputs to get the full attention output.

    Memory saving: each GPU stores only 1/T of all attention weight matrices.
    Compute saving: each GPU computes 1/T of the attention heads.
    Communication cost: 2 AllReduce operations per transformer block
    (one for the MLP, one for attention output).


### Communication Cost of Tensor Parallelism

    Each AllReduce transfers 2(T-1)/T × A bytes per GPU,
    where A = activation tensor size (sequence × model dim × batch).

    For a 1B parameter model, batch=1, seq=2048, dim=2048:
    A ≈ 2048 × 2048 × 2 bytes ≈ 8 MB per AllReduce.
    With 24 transformer blocks and 2 AllReduces per block: 24 × 2 × 8 MB = 384 MB
    at each forward + backward.

    This communication must happen every layer → REQUIRES high-bandwidth
    interconnect (NVLink: 900 GB/s intra-node, vs InfiniBand: 400 Gb/s inter-node).

    Critical rule: tensor parallelism is only efficient WITHIN a node
    (where NVLink bandwidth is available). Across nodes, the communication
    cost becomes prohibitive. This is why tensor parallel degree is
    typically 4 or 8 (matching the number of GPUs per server node).


##### PART 4 — PIPELINE PARALLELISM

### The Idea: Layers on Different GPUs

    Pipeline parallelism assigns consecutive layers to different GPUs.
    A model with L layers and P pipeline stages assigns L/P layers per stage.

    GPU 0: layers 1 to L/P
    GPU 1: layers L/P+1 to 2L/P
    ...
    GPU P-1: layers (P-1)L/P+1 to L

    Data flows like an assembly line:
        GPU 0 processes the input → sends activations to GPU 1
        GPU 1 processes further → sends to GPU 2
        ... and so on until GPU P-1 produces the output.

    Memory saving: each GPU stores only 1/P of the model weights.
    No weight broadcast needed — each GPU's weights are fixed.
    Communication: only activations at layer boundaries (small tensors).


### Naive Pipeline: The GPU Bubble Problem

    With naive pipelining, only ONE GPU is active at a time.
    While GPU 0 is processing, GPUs 1 through P-1 are idle.

    Diagram 6 — Naive Pipeline Bubble:

    Time ─────────────────────────────────────────────────────►
    GPU0 [Forward B1][bubble  ][bubble  ][bubble  ][Bwd B1]
    GPU1 [bubble  ][Forward B1][bubble  ][bubble  ][Wait][Bwd B1]
    GPU2 [bubble  ][bubble  ][Forward B1][bubble  ][Wait ][Bwd B1]
    GPU3 [bubble  ][bubble  ][bubble  ][Forward B1][Loss ][Bwd B1]

    GPU utilisation: (1/P) × 100%.
    For P=8 pipeline stages: only 12.5% utilisation. Catastrophically bad.


### GPipe: Micro-Batches to Fill the Pipeline

    GPipe (Huang et al., 2019) splits each mini-batch into M micro-batches.
    Micro-batches flow through the pipeline overlapping with each other.

    Diagram 7 — GPipe with 4 GPUs, 4 Micro-batches:

    Time ──────────────────────────────────────────────────────────►
    GPU0 [F m1][F m2][F m3][F m4][bubble][B m4][B m3][B m2][B m1]
    GPU1       [F m1][F m2][F m3][F m4  ][B m4][B m3][B m2][B m1]
    GPU2             [F m1][F m2][F m3  ][F m4][B m4][B m3][B m2][B m1]
    GPU3                   [F m1][F m2  ][F m3][F m4][B m4][B m3][B m2][B m1]

    (F = forward, B = backward, m1-m4 = micro-batches)

    Bubble fraction = (P-1) / (M + P - 1)

    As M grows, the bubble fraction shrinks:
        P=4, M=4:   bubble = 3/7  ≈ 43%  (bad)
        P=4, M=8:   bubble = 3/11 ≈ 27%  (moderate)
        P=4, M=32:  bubble = 3/35 ≈ 9%   (acceptable)
        P=4, M=128: bubble = 3/131 ≈ 2%  (very efficient)

    Trade-off: more micro-batches → better GPU utilisation but MORE
    ACTIVATIONS to store simultaneously (one per micro-batch per stage).


### PipeDream: 1F1B Schedule

    PipeDream (Narayanan et al., 2019) interleaves forward and backward
    passes more aggressively with the 1-Forward-1-Backward (1F1B) schedule.

    Diagram 8 — 1F1B Schedule (4 GPUs, 4 Micro-batches):

    Time ──────────────────────────────────────────────────────────►
    GPU0 [F1][F2][F3][F4][B1][B2][B3][B4]
    GPU1    [F1][F2][F3][B1][F4][B2][B3][B4]
    GPU2       [F1][F2][B1][F3][B2][F4][B3][B4]
    GPU3          [F1][B1][F2][B2][F3][B3][F4][B4]

    Key difference: backward passes begin as soon as possible, freeing
    pipeline stages to start new micro-batches rather than waiting.

    Memory advantage: at steady state, only a FIXED NUMBER of activations
    need to be stored (one per micro-batch in flight, not all at once).
    GPipe: must store M × layers activations simultaneously.
    1F1B: stores only P micro-batches' worth of activations.

    The bubble fraction is identical to GPipe: (P-1)/(M+P-1).
    But peak memory is much lower, enabling more micro-batches → smaller bubbles.


##### PART 5 — ZeRO AND FSDP

### The Memory Breakdown: What Uses GPU RAM

    For a model with P parameters trained with AdamW in mixed precision:

    Model states:
        FP16 parameters:        P × 2 bytes
        FP16 gradients:         P × 2 bytes
        FP32 master parameters: P × 4 bytes
        FP32 Adam m (momentum): P × 4 bytes
        FP32 Adam v (variance): P × 4 bytes
    Total model states:         P × 16 bytes

    Residual (batch and sequence dependent):
        Activations:            large, scales with batch × seq × layers
        Temporary buffers:      ~constant overhead

    For N GPUs in data parallel (DDP), EVERY GPU holds ALL P × 16 bytes.
    ZeRO eliminates this redundancy by sharding model states across GPUs.


### ZeRO Stage 1: Optimiser State Partitioning

    Insight: optimiser states (FP32 m and v) are only needed during the
    optimiser step, not during forward or backward pass.

    Each GPU owns 1/N of the optimiser states.
    After AllReduce of gradients, each GPU applies the optimiser update
    ONLY for its owned 1/N portion of parameters, then broadcasts those
    updated parameters to all GPUs.

    Memory saved per GPU: P × 8 / N bytes (optimiser states)
    Communication added: AllGather of parameters after each step.

    ZeRO Stage 1 is transparent: weights, gradients, and activations
    look the same as vanilla DDP. Only the optimiser step differs.


### ZeRO Stage 2: Gradient Partitioning

    After backward: each GPU only needs the gradients for its OWN
    1/N shard of parameters (to apply its optimiser update).

    Instead of AllReduce (which gives every GPU the full gradient):
    Use Reduce-Scatter: GPU k ends up with the summed gradient for
    only the k-th shard of parameters. The rest is discarded.

    Memory saved per GPU: optimiser states + gradients = P × 10 / N bytes.
    Communication: SAME as vanilla DDP! (Reduce-Scatter has same total
    data as AllReduce; we just don't do the AllGather phase.)

    No additional communication overhead vs DDP. Memory is halved
    (or more, for large N). This is why ZeRO Stage 2 is the default.


### ZeRO Stage 3: Parameter Partitioning

    The most aggressive stage: parameters themselves are sharded.
    Each GPU permanently stores only 1/N of the model weights.

    To compute a forward pass on any layer:
    1. AllGather: all GPUs contribute their shard → assemble full layer.
    2. Compute forward with full layer on local data.
    3. Discard the gathered parameters (they can be re-fetched later).
    4. Repeat for each layer.

    Memory per GPU: P × 16 / N bytes total.
    For N=64 GPUs: 64× reduction in memory per GPU!
    A 70B model that requires 1,120 GB normally fits in ~17.5 GB per GPU.

    Communication overhead:
        Forward:  AllGather parameters for each layer
        Backward: AllGather parameters for each layer (again, for backprop)
                  Reduce-Scatter gradients
        Total: 3× the communication of DDP.

    ZeRO Stage 3 trades communication for memory.
    Only beneficial when the communication overhead is justified by
    the ability to fit a much larger model.


### ZeRO-Infinity: Offloading to CPU and NVMe

    Extension of ZeRO Stage 3: parameters, gradients, and optimiser states
    can be OFFLOADED to CPU RAM or NVMe SSD when not in use.

    NVMe capacity: multiple TB (vs 80 GB for GPU, 1-2 TB for CPU RAM).
    Bandwidth: GPU→CPU ≈ 48 GB/s (PCIe 4.0 x16), GPU→NVMe ≈ 7 GB/s.

    Enables training models of ARBITRARY SIZE (limited only by NVMe).
    In practice, NVMe bandwidth is the bottleneck — training is slow.
    Used for model sizes that would otherwise be completely impossible.


### PyTorch FSDP: ZeRO Stage 3 with Flat Parameters

    PyTorch's Fully Sharded Data Parallel (FSDP) implements ZeRO Stage 3:

    Key design choice — FLAT PARAMETERS:
        Rather than tracking individual parameter tensors, FSDP flattens
        all parameters of a "FSDP unit" (typically a transformer block)
        into one large 1D tensor. This enables efficient bulk AllGather
        operations with minimal Python overhead.

    FSDP auto-wrapping: the user specifies which sub-modules form
    FSDP units (typically via transformer_auto_wrap_policy).
    FSDP handles the AllGather/Reduce-Scatter communication automatically.

    Memory management:
        FSDP can keep parameters in memory ("full shard" by default)
        or offload to CPU between uses (CPU offload mode).

    Communication scheduling:
        FSDP prefetches parameters for the NEXT FSDP unit during the
        current unit's forward pass, hiding AllGather latency.


##### PART 6 — 3D PARALLELISM: COMBINING ALL THREE

### Why No Single Strategy Scales to 1000+ GPUs

    DATA PARALLEL alone:
        Requires full model on each GPU → memory bottleneck.
        Communication scales with gradient size, not number of GPUs.
        Works well up to ~512 GPUs for moderate models.

    TENSOR PARALLEL alone:
        Requires very high bandwidth between all TP-group GPUs.
        Bandwidth per GPU drops as TP degree increases.
        Only efficient within a single node (NVLink bandwidth).
        Practical limit: TP degree = 4 or 8.

    PIPELINE PARALLEL alone:
        Reduces per-GPU memory but introduces idle bubbles.
        More stages → more bubbles → lower efficiency.
        Practical limit: PP degree = 4 to 16.

    The solution: combine all three in a 3D grid.


### The 3D Parallelism Grid

    Assign GPUs to a 3D logical grid:
        Data parallel degree: D
        Tensor parallel degree: T
        Pipeline parallel degree: P
        Total GPUs: D × T × P

    Diagram 9 — 3D Parallelism Assignment for 16 GPUs (D=2, T=4, P=2):

    PIPELINE STAGE 0              PIPELINE STAGE 1
    ┌────────────────────────┐    ┌─────────────────────────┐
    │ Tensor Parallel Group  │    │ Tensor Parallel Group   │
    │ GPU0 GPU1 GPU2 GPU3    │    │ GPU8  GPU9  GPU10 GPU11 │
    │ (DP group 0, rank 0-3) │    │ (DP group 0, rank 8-11) │
    ├────────────────────────┤    ├─────────────────────────┤
    │ GPU4 GPU5 GPU6 GPU7    │    │ GPU12 GPU13 GPU14 GPU15 │
    │ (DP group 1, rank 4-7) │    │ (DP group 1, rank 12-15)│
    └────────────────────────┘    └─────────────────────────┘

    Communication groups:
    - TENSOR PARALLEL: {GPU0,1,2,3}, {GPU4,5,6,7}, {GPU8,9,10,11}, {GPU12,13,14,15}
      High-frequency, small tensors → must be on NVLink (same node).

    - PIPELINE PARALLEL: {GPU0,GPU8}, {GPU1,GPU9}, ...
      Medium-frequency, medium tensors (activations at stage boundaries).
      Can cross nodes via InfiniBand.

    - DATA PARALLEL: {GPU0,GPU4}, {GPU1,GPU5}, ... (one per TP rank in each PP stage)
      Low-frequency, large tensors (full gradient AllReduce).
      Can cross nodes via InfiniBand.

    Assignment of communication to hardware:
        Intra-node (NVLink, 900 GB/s):  Tensor Parallel (requires highest BW)
        Inter-node (InfiniBand, 50 GB/s): Pipeline Parallel and Data Parallel


### How GPT-3, LLaMA, and Megatron Assign Ranks

    GPT-3 (175B, 96 layers, 96 heads):
        D=64, T=8, P=8 → total 4,096 V100 GPUs
        Each pipeline stage: 96/8 = 12 layers
        Each tensor parallel group: 96/8 = 12 heads each

    LLaMA 2 (70B, 80 layers):
        D=2000, T=8, P=4 (approximately) → ~64,000 GPU-equivalents
        (Meta used different hardware configurations)

    The ranks (GPU IDs) are assigned so that:
        Tensor parallel GPUs have consecutive ranks within a node.
        Pipeline parallel ranks are on different nodes.
        Data parallel ranks span across all nodes.


### Bandwidth Hierarchy and Its Impact

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Link              │ Bandwidth     │ Suitable for                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ NVLink 3.0        │ 600 GB/s      │ Tensor parallel (intra-node)    │
    │ NVLink 4.0        │ 900 GB/s      │ Tensor parallel (intra-node)    │
    │ PCIe Gen 4 x16    │ 32 GB/s       │ GPU-to-CPU, ZeRO offload        │
    │ InfiniBand HDR    │ 50 GB/s       │ Data parallel, pipeline (inter) │
    │ InfiniBand NDR    │ 100 GB/s      │ Data parallel, pipeline (inter) │
    │ Ethernet 100GbE   │ 12.5 GB/s     │ Data parallel only (slow!)      │
    └─────────────────────────────────────────────────────────────────────┘

    Rule: always assign the most communication-intensive parallelism
    dimension to the highest-bandwidth link.


##### PART 7 — HETEROGENEOUS GPUS: DIFFERENT SPEEDS AND CAPABILITIES

### The Straggler Problem

    In data parallelism, the SLOWEST GPU determines the throughput of
    every iteration. All GPUs must complete their forward and backward
    pass before the AllReduce barrier. The fast GPUs sit idle waiting
    for the slow one.

    Diagram 10 — The Straggler Problem:

    Time ─────────────────────────────────────────────────────────────────►
    GPU0 (A100): [Forward & Backward ────────────────][Wait][AllReduce]
    GPU1 (A100): [Forward & Backward ────────────────][Wait][AllReduce]
    GPU2 (RTX 4090): [Forward & Backward ──────────────────────────][AllReduce]
    GPU3 (A100): [Forward & Backward ────────────────][Wait][AllReduce]

                                        ← IDLE TIME →

    The RTX 4090 (faster than some consumer GPUs but slower than A100
    on large batch BF16 workloads) causes A100s to waste cycles.

    Quantifying the loss:
        If GPU_slow takes Ts seconds and GPU_fast takes Tf seconds:
        Wasted fraction = (Ts - Tf) / Ts per GPU-iteration.
        With N GPUs: total wasted compute = (N-1) × (Ts - Tf) / Ts.
        For N=8 and Ts/Tf = 2: 87.5% of fast GPU cycles are wasted.


### Memory Asymmetry: Different VRAM Capacities

    Scenario: GPU 0 has 80 GB (A100), GPU 1 has 24 GB (RTX 3090).

    DATA PARALLEL: fails if the full model + activations require more
    than 24 GB. Both GPUs must hold the same model with the same batch.

    PIPELINE PARALLEL: each GPU holds different layers, so different
    stages can have different memory requirements. Assign MORE layers
    to the GPU with more VRAM.

    FSDP / ZeRO Stage 3: shards parameters across GPUs. The shard
    size is usually equal — the smaller GPU gets the same fraction.
    If the shard size itself exceeds the smaller GPU's VRAM (possible
    for very large models), FSDP fails.

    TENSOR PARALLEL: each GPU holds a fraction of each weight matrix.
    The fraction is equal by default. If the equal fraction does not
    fit on the smaller GPU, you need asymmetric splits.


### Compute Asymmetry: Different TFLOPS and Generations

    Scenario: GPU 0 is an A100 (312 TFLOPS BF16), GPU 1 is a V100
    (125 TFLOPS FP16). The V100 also does not support BF16.

    DTYPE MISMATCH:
        A100 runs BF16. V100 only supports FP16.
        Running the whole job in FP16 penalises the A100 (BF16 is faster on A100).
        Running in BF16 is impossible for the V100.
        Mixing dtypes per GPU: gradients have different precision →
        AllReduce averages across different float representations →
        possible precision loss at the summation boundary.

        Practical solution: use FP16 for the whole job (works on both).
        The A100 is underutilised (running FP16 instead of BF16),
        but at least training is numerically consistent.

    TENSOR CORE MISMATCH:
        A100 has 3rd-gen Tensor Cores (FP16/BF16/TF32/INT8/FP8).
        V100 has 1st-gen Tensor Cores (FP16 only).
        If the job uses TF32 precision, the V100 falls back to FP32,
        which is ~4× slower → severe straggler effect.

        Solution: use FP16 explicitly; avoid TF32-only features.


### Static Load Balancing: Assign Larger Shards to Faster GPUs

    For pipeline parallelism with heterogeneous GPUs, assign more
    layers to faster/larger GPUs to equalise stage time.

    Example: 48-layer model, GPU 0 is 2× faster than GPU 1.
        EQUAL SPLIT:    GPU 0: layers 1-24, GPU 1: layers 25-48
        GPU 0 finishes in T seconds. GPU 1 finishes in 2T seconds.
        GPU 0 idle for T seconds every iteration. Bubble = 50%.

        LOAD-BALANCED:  GPU 0: layers 1-32, GPU 1: layers 17-48
        GPU 0 has 2/3 of layers but runs 2× faster → T seconds.
        GPU 1 has 1/3 of layers and runs 1× → T seconds.
        Both finish at the same time! Bubble minimised.

    Formula for balanced assignment:
        Layers assigned to GPU k ∝ compute_speed(k) / sum(compute_speed)

    For memory-limited cases, combine compute speed and memory capacity:
        Balance: layers assigned to GPU k ∝ min(VRAM(k)/mem_per_layer,
                                                  TFLOPS(k)/TFLOP_per_layer)

    Diagram 11 — Balanced vs Unbalanced Pipeline:

    UNBALANCED (equal layers, unequal speed):
    GPU0 (fast): [Layers 1-24 ──────][IDLE IDLE IDLE IDLE]
    GPU1 (slow): [Layers 25-48 ──────────────────────────]

    BALANCED (more layers on fast GPU):
    GPU0 (fast): [Layers 1-32 ────────────────────────────]
    GPU1 (slow): [Layers 33-48 ──────────────────────────]
                 Both finish at the same time!


### Dynamic Load Balancing: Measure and Adapt at Runtime

    Static assignment assumes you know the relative speeds in advance.
    In practice, speeds can vary:
        Thermal throttling: a GPU running hot may slow down mid-training.
        Memory pressure: more activations in a busy stage → slower.
        Network congestion: communication time varies.

    Dynamic approaches:
    1. PROFILING-BASED:
       Run a warmup period measuring actual time per stage.
       Rebalance by migrating layers between GPUs at a checkpoint.
       Cost: migration requires saving and loading model shards.
       Frequency: once at warmup, or periodically (every N hours).

    2. WORK STEALING (for data parallel):
       Monitor each GPU's micro-batch completion time.
       Faster GPUs "steal" extra micro-batches from slower GPUs.
       Requires a dynamic data loading queue rather than pre-split datasets.
       Works well when the bottleneck is data throughput, not memory.

    3. GRADIENT STALENESS (asynchronous DP):
       Allow GPUs to apply gradients independently without waiting.
       Faster GPUs update more frequently; slower GPUs update less.
       Risk: training on stale parameters reduces convergence quality.
       Used by Hogwild!, downpour SGD — rarely used in production today.


### Asymmetric Tensor Parallel Splits

    In tensor parallelism, the default split is equal (1/T of each
    matrix per GPU). With unequal GPUs, use asymmetric splits.

    Example: T=2, GPU 0 has 2× the compute and VRAM of GPU 1.
        GPU 0: 2/3 of each weight matrix
        GPU 1: 1/3 of each weight matrix

    Challenge: the AllReduce after each layer must still synchronise
    GPUs with different-sized result tensors. This requires a
    VARIABLE-SPLIT AllReduce (not standard in most libraries).

    Workaround using standard AllReduce:
        GPU 1 pads its partial output with zeros to match GPU 0's size.
        AllReduce sums padded tensors.
        Both GPUs strip the padding after AllReduce.
        Overhead: extra memory and communication for padding.

    Practical reality: asymmetric tensor parallel is rarely implemented
    in standard frameworks. The more common solution is to exclude the
    weak GPU from tensor parallel groups and use it only in data parallel.


### Gradient Synchronisation Across Different Dtypes

    If GPU 0 computes gradients in BF16 and GPU 1 in FP16,
    AllReduce cannot directly sum tensors of different types.
    NCCL (the GPU communication library) requires matching dtypes.

    Two solutions:

    SOLUTION 1 — Upcast before AllReduce:
        Both GPUs upcast their gradients to FP32 before AllReduce.
        AllReduce in FP32 (lossless combination).
        Downcast back to each GPU's native dtype after.
        Cost: 2× gradient communication size (FP32 is twice FP16/BF16).

    SOLUTION 2 — Uniform dtype enforcement:
        Force all GPUs to use the same dtype (typically FP16,
        the lower common denominator).
        A100 is penalised (cannot use BF16) but no casting overhead.
        Simpler, recommended unless the performance difference matters.


### Practical Strategies for Heterogeneous Clusters

    Strategy 1 — EXCLUDE THE WEAK GPU ENTIRELY:
        If one GPU is significantly slower (>50% slower), it may be
        more efficient to simply not use it.
        The remaining fast GPUs operate at full efficiency.
        "Training with a slow GPU" can cause more time loss (idle
        cycles) than training with one fewer GPU entirely.
        Rule of thumb: if the slowest GPU is <60% the speed of the
        fastest, exclude it from the main training group.

    Strategy 2 — USE THE WEAK GPU AS A PARAMETER SERVER:
        Assign the slow GPU to host the parameter server (ZeRO Stage 1/2).
        The slow GPU holds and updates optimiser states.
        The fast GPUs do forward/backward and send gradients to the slow GPU.
        The slow GPU applies optimiser updates and sends back parameters.
        The bottleneck becomes communication, not compute.

    Strategy 3 — WEAK GPU FOR EMBEDDING TABLES:
        Embedding tables (common in recommendation models) are
        memory-intensive but compute-light (just lookup, no heavy matmul).
        Assign embeddings to the slower/smaller GPU.
        Assign the compute-heavy transformer/MLP layers to the fast GPU.
        This is a form of heterogeneous pipeline parallelism.

    Strategy 4 — DATA PARALLEL WITH UNEQUAL MICRO-BATCHES:
        Assign a larger micro-batch to the faster GPU.
        Both GPUs take the same wall-clock time per step.
        The effective batch is N_fast × B_fast + N_slow × B_slow.
        Gradient averaging: weight each GPU's gradient by its actual
        batch size, not uniformly:
            g_effective = (B_fast × g_fast + B_slow × g_slow) / (B_fast + B_slow)
        Requires custom gradient weighting — not automatic in DDP.

    Strategy 5 — SEPARATE TRAINING JOBS:
        Train the model separately on each GPU cluster (fast and slow).
        Use knowledge distillation or model merging to combine results.
        Federated learning techniques apply here.


### Torch Distributed Elasticity: Fault Tolerance

    In large heterogeneous clusters, GPUs fail during training.
    A GPU that goes offline mid-training would crash the entire job
    without fault tolerance.

    TorchElastic (integrated as torchrun):
        Monitors GPU health across all ranks.
        If a GPU fails, the job PAUSES, saves a checkpoint, and RESTARTS
        with the remaining healthy GPUs.
        Ranks are reassigned: the world size changes dynamically.
        Training resumes from the checkpoint as if nothing happened.

    Requirements:
        Checkpoint every N steps (N small enough that losing N steps
        is acceptable — typically every 30-60 minutes).
        Checkpoints must be saved atomically and accessible by all GPUs
        (shared filesystem or object storage).

    Interaction with heterogeneous GPUs:
        If the failing GPU was the "bottleneck GPU" (the slowest one),
        removing it and restarting may IMPROVE training throughput.
        This is a surprising but real silver lining of fault tolerance.


### When to Just Not Use the Weaker GPU

    Sometimes the right answer is: do not use it.

    Scenarios where excluding the weak GPU is correct:
    1. Speed mismatch > 50%: the straggler overhead exceeds the added compute.
    2. Dtype incompatibility: cannot run the required precision.
    3. VRAM too small: cannot hold even 1/N shard of the model.
    4. Driver or library version incompatibility (different CUDA versions).
    5. The weak GPU introduces communication topology problems
       (e.g., it's only connected via PCIe when others have NVLink).

    How to exclude it in PyTorch:
        Set CUDA_VISIBLE_DEVICES to exclude the weak GPU before launching.
        Or use torch.distributed with a custom device mesh that only
        includes the desired GPUs.

    A useful perspective: a cluster of homogeneous fast GPUs is
    almost always more efficient than a mixed cluster of fast + slow GPUs
    of the same total TFLOP count, because heterogeneity introduces
    synchronisation overhead, complexity, and idle cycles.


##### PART 8 — GRADIENT SYNCHRONISATION IN DEPTH

### What Must Be Synchronised

    In data parallelism:
    MUST synchronise: GRADIENTS (so all GPUs apply the same weight update)
    MUST NOT synchronise: ACTIVATIONS (local to each GPU's data shard)
    IMPLICIT synchronisation: WEIGHTS (stay in sync because the same
    gradient is applied by each GPU)

    In tensor parallelism:
    MUST synchronise: ACTIVATIONS (partial sums from different weight shards)
    MUST NOT wait: WEIGHTS (each GPU permanently owns its shard)

    In pipeline parallelism:
    MUST synchronise: ACTIVATIONS at stage boundaries (point-to-point)
    NOT synchronised: GRADIENTS of the same layer (different stages)


### AllReduce Operation: Sum Then Average

    AllReduce(tensor) → every GPU ends up with the same result,
    which is the SUM (or MEAN) of all tensors.

    For data parallel gradient averaging:
        AllReduce([g₀, g₁, g₂, g₃]) → each GPU gets (g₀+g₁+g₂+g₃)/4

    The divide by N is implemented either:
        BEFORE AllReduce: each GPU divides its loss by N before backward
        AFTER AllReduce: each GPU divides the summed gradient by N
    Both are numerically equivalent; DDP does the latter.


### Ring-AllReduce Step-by-Step

    4 GPUs, gradient tensor G split into 4 chunks: G = [G₀, G₁, G₂, G₃]

    Initial state (each GPU has the full tensor, partially computed):
    GPU 0: [G₀⁰, G₁⁰, G₂⁰, G₃⁰]
    GPU 1: [G₀¹, G₁¹, G₂¹, G₃¹]
    GPU 2: [G₀², G₁², G₂², G₃²]
    GPU 3: [G₀³, G₁³, G₂³, G₃³]
    (Gᵢʲ = chunk i of gradient on GPU j)

    PHASE 1 — REDUCE-SCATTER (3 steps):
    After 3 steps, GPU k has the fully-reduced version of chunk k:
    GPU 0: chunk 0 fully reduced = G₀⁰+G₀¹+G₀²+G₀³
    GPU 1: chunk 1 fully reduced = G₁⁰+G₁¹+G₁²+G₁³
    GPU 2: chunk 2 fully reduced
    GPU 3: chunk 3 fully reduced

    PHASE 2 — ALLGATHER (3 steps):
    Each GPU sends its fully-reduced chunk around the ring.
    After 3 more steps:
    GPU 0: [G₀_sum, G₁_sum, G₂_sum, G₃_sum] = full AllReduce result
    GPU 1: [G₀_sum, G₁_sum, G₂_sum, G₃_sum] = same!
    GPU 2: same
    GPU 3: same

    Total data sent per GPU: 2 × (N-1)/N × |G|
    For N=1000: ≈ 2|G| → nearly bandwidth-optimal regardless of N.


### Gradient Compression

    For very large clusters or slow interconnects, gradient compression
    reduces the communication volume at the cost of some precision.

    POWERSGD (Vogels et al., 2019):
        Approximates the gradient matrix G with a low-rank factorisation:
        G ≈ P × Q (where P and Q have much smaller dimensions)
        Compress ratio: (m×n) → (m×r + r×n)  where r << min(m,n)
        Typical r=1 or r=4 gives 10-100× compression.
        Residual error is accumulated and added to the next iteration's gradient.

    1-BIT ADAM (Seide et al.):
        Quantise each gradient value to 1 bit (just the sign).
        AllReduce of 1-bit tensors is 32× smaller than FP32.
        Error compensation: keep the quantisation error and add to next step.
        Works surprisingly well for LLM fine-tuning.

    TOP-K SPARSIFICATION:
        Each GPU sends only the K largest-magnitude gradient elements.
        Other elements are accumulated as "error" for the next step.
        Communication: K × (value + index) per element.
        Effective when K << N (high sparsity ratio).

    WARNING: gradient compression introduces approximation error that
    can slow convergence or hurt final accuracy. Only use when the
    communication bottleneck is severe and compression is necessary.


### Communication-Computation Overlap

    The single most important performance technique in distributed training:
    compute gradients for one part of the network while communicating
    gradients for another part.

    How DDP achieves this:
    1. The backward pass computes gradients layer by layer, from last to first.
    2. DDP registers a "hook" on each parameter that fires when its gradient
       is computed.
    3. When enough gradients accumulate to fill a bucket, the hook triggers
       an AllReduce for that bucket — while the rest of the backward pass
       continues.
    4. By the time the forward pass of the next iteration begins, most
       AllReduce operations are already complete.

    Diagram 12 — Overlap in DDP:

    Time ──────────────────────────────────────────────────────────►
    CPU: [Forward]──[Backward  layer 24..18]──[Backward 17..12]──[Backward 11..1]
    GPU: [Forward]──[Backward  L24..18    ]──[Backward 17..12]──[Backward 11..1]
    Net:              [AllReduce: bucket3   ]──[AllReduce: b2  ]──[AllReduce: b1]

    Overlap hides communication latency completely when backward compute
    time > AllReduce communication time (common in large models).

    When there is NO overlap (bucket size too large, or computation too fast):
    The optimizer step is blocked waiting for AllReduce to complete.
    Tuning bucket size (--ddp-bucket-cap-mb) controls this trade-off.


##### PART 9 — PRACTICAL SETUP, FAILURE MODES & DEBUGGING

### Communication Backends

    NCCL (NVIDIA Collective Communications Library):
        Optimised for NVIDIA GPU-to-GPU communication.
        Automatically selects the fastest path: NVLink > PCIe > InfiniBand.
        Supports AllReduce, AllGather, ReduceScatter, Broadcast, P2P.
        Default and recommended for any NVIDIA GPU cluster.

    Gloo:
        CPU-based collective communication.
        Used for CPU tensors or as a fallback on non-NVIDIA hardware.
        Much slower than NCCL for GPU training.
        Mainly used for testing and debugging on a single machine.

    MPI (Message Passing Interface):
        High-performance computing standard, supports any hardware.
        Can be used as a backend for PyTorch distributed.
        Useful in HPC clusters with InfiniBand where NCCL is unavailable.

    Heterogeneous GPU note: NCCL requires all GPUs to support the
    same CUDA version and driver. Mixed architectures (e.g., NVIDIA + AMD)
    require different backends (NCCL for NVIDIA, RCCL for AMD, Gloo as bridge).


### Initialising the Process Group

    Every distributed training run initialises a process group:

        import torch.distributed as dist
        dist.init_process_group(
            backend="nccl",
            init_method="env://",   # or "tcp://<master_ip>:<port>"
            rank=rank,              # this process's global rank (0 to world_size-1)
            world_size=world_size   # total number of processes
        )

    Environment variables (set by torchrun or SLURM):
        MASTER_ADDR: IP of the master node (rank 0)
        MASTER_PORT: free port for the rendezvous
        RANK:        global rank of this process
        WORLD_SIZE:  total number of processes
        LOCAL_RANK:  rank within this node (used to select CUDA device)

    Key concepts:
        RANK: unique identifier for each process (0 to world_size-1)
        LOCAL_RANK: rank within a single node (0 to GPUs_per_node-1)
        WORLD_SIZE: total number of processes across all nodes
        PROCESS GROUP: a subset of ranks that communicate (for TP/PP sub-groups)


### Common Failure Modes

    NCCL TIMEOUT:
        AllReduce hangs because one rank never reaches the barrier.
        Cause: a GPU crashed, a process was killed, or one rank's
        backward pass takes much longer (heterogeneous straggler).
        Fix: check NCCL_DEBUG=INFO for which rank is missing;
        check GPU health; reduce NCCL timeout or add heartbeating.

    OUT OF MEMORY ON RANK 0:
        Rank 0 often has extra memory usage: it saves checkpoints,
        logs, and may receive broadcast tensors.
        Fix: explicitly move objects off rank 0 after use;
        use rank 0 only for I/O, not for extra computation.

    GRADIENT EXPLOSION ACROSS RANKS:
        Gradients are correctly averaged by AllReduce, but if ONE rank
        has a very large gradient (numerical issue on its data shard),
        the average is pulled up, destabilising training.
        Fix: per-rank gradient clipping BEFORE AllReduce; inspect
        per-rank gradient norms in logging.

    DEADLOCK FROM UNBALANCED COMMUNICATION:
        If some ranks call AllReduce and others do not (e.g., due to
        conditional branching in the model), the AllReduce hangs.
        All ranks must call collective operations in the SAME ORDER.
        Fix: remove rank-dependent control flow; use torch.no_grad
        consistently across ranks.

    CHECKSUM / WEIGHT DIVERGENCE:
        After a long training run, weights on different ranks diverge
        due to floating point non-determinism in parallel operations.
        Fix: periodically verify that rank 0 and rank N-1 have identical
        weights by broadcasting a hash of the weight tensor.


### Monitoring Distributed Training

    PER-RANK METRICS to log:
        Gradient norm (per rank and global): detect straggler outliers.
        Step time (per rank): identify which GPU is the bottleneck.
        Memory usage (per rank): detect memory imbalance.
        Loss (per rank): detect if one shard has anomalous data.

    DCGM (NVIDIA Data Center GPU Manager):
        Hardware-level monitoring: GPU utilisation, temperature, memory
        bandwidth, power, error counts (ECC errors signal hardware failure).
        Essential for large clusters where silent GPU failures are common.

    NCCL_DEBUG=INFO or NCCL_DEBUG=WARN:
        Enables detailed NCCL logging: which operations are executed,
        which path (NVLink/PCIe/InfiniBand) is chosen, timeout information.
        Very verbose — only enable when debugging communication issues.


### Checkpointing in Distributed Context

    NAIVE APPROACH (rank 0 saves):
        Only rank 0 saves the checkpoint.
        Works but creates a bottleneck: rank 0 must gather all parameters
        (if using ZeRO/FSDP sharding) before saving.
        For a 70B model: gathering 1TB+ of parameters is slow and may OOM.

    SHARDED CHECKPOINTING (recommended for ZeRO/FSDP):
        Each rank saves its OWN shard of the model.
        Checkpoint consists of N shard files.
        Loading: each rank loads its own shard — no gathering needed.
        PyTorch provides torch.distributed.checkpoint for this.

    LOADING WITH DIFFERENT WORLD SIZE:
        If you resume training with fewer GPUs (e.g., after a failure),
        the N shards must be redistributed to the new world size M.
        PyTorch's checkpoint sharding library handles this automatically.
        Manual approach: load all N shards on one machine, re-shard to M.

    ATOMIC SAVING:
        Write to a temporary path, then rename to the final path.
        This prevents a partial checkpoint from looking complete
        if the job crashes mid-save.
        Use a global barrier (dist.barrier()) after saving to ensure
        all ranks have finished before proceeding.


### Summary: Choosing the Right Strategy

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Scenario                  │ Recommended Strategy                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Model fits on 1 GPU       │ DDP (data parallel) only                │
    │                           │                                         │
    │ Model fits on 1 GPU,      │ FSDP (ZeRO Stage 2) + DDP               │
    │ but OOM from optimizer    │                                         │
    │                           │                                         │
    │ Model does NOT fit        │ FSDP (ZeRO Stage 3) + DDP               │
    │ on 1 GPU                  │ or TP + DDP (within/across node)        │
    │                           │                                         │
    │ Very large model,         │ 3D: TP (intra-node) +                   │
    │ 1000+ GPUs                │     PP (inter-node) +                   │
    │                           │     DP (across all)                     │
    │                           │                                         │
    │ Heterogeneous GPUs,       │ Use fast GPUs only; or assign           │
    │ large speed difference    │ slow GPU as parameter server            │
    │                           │                                         │
    │ Heterogeneous GPUs,       │ Pipeline parallel with unequal          │
    │ different VRAM            │ layer assignment (more layers to        │
    │                           │ GPU with more memory)                   │
    │                           │                                         │
    │ GPU failure mid-training  │ TorchElastic (torchrun) with            │
    │                           │ periodic checkpointing                  │
    └─────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# No OPERATIONS (theory-only module)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {}


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