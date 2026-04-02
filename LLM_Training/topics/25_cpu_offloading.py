"""
CPU Offloading
===============

CPU offloading extends GPU training beyond the physical limits of GPU HBM
by spilling tensors to CPU RAM — and beyond that, to NVMe SSDs. Where
ZeRO-3 divides memory across N GPUs, CPU offloading multiplies effective
GPU memory by using the vastly larger CPU memory pool. A single A100 has
80 GB of HBM; a DGX node has 2 TB of CPU RAM — 25× more. Understanding
the PCIe bandwidth constraints, prefetching strategies, and the precise
decision of which tensors to offload is what separates a working offloading
setup from one that grinds to a halt waiting for data transfers.

"""

import base64
import os
import textwrap
import re

TOPIC_NAME = "CPU Offloading"
DISPLAY_NAME = "25 · CPU Offloading"
ICON = "💿"
SUBTITLE = "Moving States to CPU/NVMe"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Memory Hierarchy for Deep Learning

Modern computing systems have multiple layers of memory with radically
different capacities, bandwidths, and latencies:

    Tier          Technology      Capacity (DGX A100)   Bandwidth      Latency
    ────────────────────────────────────────────────────────────────────────────
    GPU HBM       HBM2e           80 GB per GPU          2 TB/s        ~100 ns
    GPU L2 cache  SRAM             40 MB per GPU          12 TB/s       ~10 ns
    CPU RAM       DDR4/DDR5        2 TB (shared)          300 GB/s      ~100 ns
    GPU↔CPU       PCIe 4.0 × 16   N/A (interconnect)     32 GB/s       ~5 µs
    NVMe SSD      PCIe 4.0 × 4    15 TB per node         7 GB/s        ~100 µs
    ────────────────────────────────────────────────────────────────────────────

The fundamental tension in CPU offloading:
    GPU HBM is 25× smaller than CPU RAM but 62× faster bandwidth.
    PCIe (32 GB/s) is the bottleneck connecting these two worlds.

CPU offloading is profitable when:
    **GPU compute time per layer** >> **PCIe transfer time for that layer's data**

If a layer takes 100 ms to compute but its weights transfer over PCIe in
5 ms, we can hide the transfer completely behind computation. If the transfer
takes 200 ms, offloading costs 100 ms net overhead per step.

**The core arithmetic:** for offloading to break even, you need:
    transfer_time ≤ compute_time
    tensor_bytes / PCIe_bandwidth ≤ flops / GPU_FLOPS

Rearranging: **arithmetic intensity** (flops / byte) ≥ GPU_FLOPS / PCIe_BW
    = 312e12 FLOPS / 32e9 B/s ≈ 9750 FLOPS/byte

A 7B model layer (d=4096) processing batch_size=1, seq_len=2048:
    compute: 2 × d² × B×T = 2 × 4096² × 1×2048 ≈ 68.7 GFLOPS
    weight:  d × d × 2 bytes = 33.6 MB
    intensity: 68.7e9 / 33.6e6 ≈ 2044 FLOPS/byte

At batch_size=1, intensity (2044) < breakeven (9750) → offloading will hurt.
At batch_size=5, intensity ≈ 10,220 → offloading breaks even!

This is why CPU offloading requires large batch sizes or long sequences to
be efficient — it is a fundamentally throughput-trading strategy.


### What Can Be Offloaded

Not all GPU memory is equally good to offload. The three categories:

**Category 1 — Optimizer State (best candidate)**
    Size: 12 bytes/param (fp32 master + m + v)
    Access pattern: once per step (not per layer)
    PCIe cost: full model transfer per step
    GPU need: only briefly during the parameter update step

    The optimizer state is updated once per gradient step and sits idle
    the rest of the time. On CPU RAM, it occupies CPU memory but doesn't
    need to be on GPU except during the update. DeepSpeed's ZeRO-Infinity
    and ZeRO CPU offload both exploit this.

    Example (7B model): 7B × 12 bytes = 84 GB optimizer state
    → Kept on CPU, transferred to GPU only for optimizer.step()
    → CPU RAM needs: 84 GB (feasible; DGX node has 2 TB CPU RAM)
    → PCIe transfer: 84 GB / 32 GB/s ≈ 2.6 seconds per step ← EXPENSIVE

**Category 2 — Parameters (conditional)**
    Size: 2 bytes/param (bf16)
    Access pattern: twice per layer (forward and backward)
    PCIe cost: full model per forward + backward = 2× model size per step

    Parameters are accessed every forward and backward pass. For a 70B
    model (140 GB), transferring the entire model twice per step at 32 GB/s
    takes 2 × 140 / 32 ≈ 8.75 seconds. This is only acceptable with very
    long computation per step (large batch, long sequences).

    ZeRO-Infinity and DeepSpeed CPU offload can offload parameters, but
    require careful prefetching to hide the transfer latency.

**Category 3 — Activations (most fragile)**
    Size: highly variable; scales with B×T×d
    Access pattern: stored in forward, consumed in backward
    PCIe cost: every layer, both passes

    Activation offloading (moving activations from GPU to CPU during forward,
    fetching back during backward) is PCIe-intensive and generally outweighed
    by activation checkpointing. Activation checkpointing recomputes for free
    (just compute cost); activation offloading pays PCIe bandwidth. Unless
    you have enormous batch sizes and fast PCIe (PCIe 5.0), activation
    checkpointing is almost always preferred over activation offloading.


    **Diagram 1 — Memory and Transfer Hierarchy:**

    GPU HBM (80 GB)             CPU RAM (2 TB)            NVMe SSD (15 TB)
    ═══════════════════════════════════════════════════════════════════════
    ┌──────────────────┐        ┌────────────────────┐    ┌─────────────┐
    │ Active params    │◄──────►│ Offloaded optim    │◄──►│ Large model │
    │ Activations      │PCIe    │ state (m, v, fp32) │NVMe│ shards      │
    │ Current grads    │32 GB/s │ Next-layer params  │    │ Checkpoint  │
    │ KV-cache         │        │ Inactive grads     │7GB/s│ archive    │
    └──────────────────┘        └────────────────────┘    └─────────────┘
         ↕ HBM              ↕ DDR4/5           ↕ PCIe 4.0
        2 TB/s             300 GB/s             7 GB/s
                 ↕ PCIe 4.0 x16 (32 GB/s) ↕

    Bandwidth cliff: GPU HBM→CPU PCIe is 62× slower than GPU HBM itself.
    Every offload operation must hide its PCIe cost behind GPU compute.


### DeepSpeed CPU Offload: ZeRO-Offload and ZeRO-Infinity

DeepSpeed provides two distinct offloading modes:

**ZeRO-Offload (2021):**
    Offloads optimizer state and gradients to CPU RAM.
    Parameters stay on GPU (no parameter offloading).
    The backward pass runs on GPU; only the optimizer.step() runs on CPU.

    Memory split:
        GPU:  16-bit params (2P) + 16-bit grads (2P) + activations
        CPU:  32-bit master weights (4P) + m (4P) + v (4P)

    The gradient flow:
        GPU backward → produces fp16 gradients → PCIe to CPU
        CPU optimizer step using fp32 master weights
        CPU → PCIe → GPU: updated fp16 parameters

    Communication: 2 × 2P bytes per step (grads down + params up) = 4P bytes
    For 7B model: 4 × 7B × 2 bytes = 56 GB per step → 56/32 = 1.75 seconds

    This is only acceptable with very long compute steps (large batch) or
    when the alternative is not training at all (model doesn't fit on GPU).

**ZeRO-Infinity (2021):**
    Extends ZeRO-3 to CPU RAM and NVMe SSD for parameters, gradients,
    and optimizer state.

    Key innovations:
        1. NVMe-aware data movement (uses I/O threads to hide NVMe latency)
        2. CPU → GPU parameter prefetching (overlapped with forward/backward)
        3. Adaptive tensor fusion (combines small tensors into large I/O ops)
        4. Memory-centric tiling (GPU computes in tiles that fit in HBM)

    This allows training models of essentially unlimited size:
        1T model: ~2 TB params + ~6 TB optimizer state = 8 TB total
        Storage: NVMe SSDs (commodity 30 TB NVMe available)
        Feasibility: 1T model on 8 GPUs (128 GB GPU memory needed for compute)


### The Prefetching Strategy

Naive offloading: when the backward pass of layer L needs layer L's parameters,
it issues a PCIe transfer → GPU stalls waiting for data.

**Prefetch**: while GPU computes layer L, CPU pre-sends layer L+1's parameters
to a staging buffer in GPU memory. When layer L+1's computation starts, its
parameters are already in GPU memory.

    Without prefetch:
    L0: [stall: wait PCIe]→[compute]
    L1:                      [stall]→[compute]
    L2:                               [stall]→[compute]
    Throughput: compute_time + PCIe_time per layer

    With prefetch:
    L0:                [compute + PCIe(L1)]
    L1:                [compute + PCIe(L2)]
    L2:                [compute + PCIe(L3)]
    Throughput: max(compute_time, PCIe_time) per layer

For prefetching to eliminate overhead:
    compute_time ≥ PCIe_time  (arithmetic intensity condition again)

**Prefetch buffer size:** To prefetch N layers ahead requires holding N+1
layers' parameters in GPU HBM simultaneously. There is a direct trade-off
between prefetch depth and GPU memory usage. Typical: prefetch 1-2 layers.


### PyTorch Memory Management for Offloading

Python's `torch.Tensor.to("cpu")` creates a CPU tensor as a NumPy-compatible
buffer. For efficient PCIe transfers, the CPU tensor must be in **pinned
(page-locked) memory**:

    # Slow: pageable memory (OS can swap this page)
    cpu_tensor = tensor.to("cpu")

    # Fast: pinned memory (PCIe DMA can access directly, no OS copy)
    cpu_tensor = tensor.to("cpu", non_blocking=True)
    # To pre-allocate pinned memory:
    cpu_pinned = torch.empty(size, pin_memory=True)

**Non-blocking transfers:** Using `non_blocking=True` starts the DMA transfer
and returns immediately. The CPU can queue the next operation while the
transfer completes. Synchronisation happens when the GPU actually reads
the transferred data.

    # Compute while transferring (overlap)
    cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
    # ... CPU work here ...
    torch.cuda.synchronize()  # ensure transfer complete before using cpu_tensor

**CUDA streams for overlap:**
Using separate CUDA streams for compute and transfers allows the GPU and
PCIe bus to work simultaneously:

    compute_stream  = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    with torch.cuda.stream(transfer_stream):
        # Start transfer for next layer
        next_layer_params.to("cuda", non_blocking=True)

    with torch.cuda.stream(compute_stream):
        # Compute current layer (GPU busy)
        output = current_layer(input)

    # Synchronise: ensure transfer is done before using next_layer_params
    compute_stream.wait_stream(transfer_stream)


    **Diagram 2 — Prefetch Timeline with CUDA Streams:**

    PREFETCH TIMELINE: COMPUTE + TRANSFER OVERLAP
    ════════════════════════════════════════════════════════════════

    Without overlap:
    t: [PCIe L0→GPU][compute L0][PCIe L1→GPU][compute L1][PCIe L2→GPU][compute L2]
       ← 2× compute time due to PCIe stalls

    With overlap (prefetch 1 layer ahead):
    Compute stream: [compute L0─────────][compute L1─────────][compute L2]
    Transfer stream:[L0 done][PCIe L1→GPU][PCIe L2→GPU][PCIe L3→GPU]
    GPU memory:       {L0}         {L0,L1}      {L1,L2}       {L2,L3}
                              ↑ peak: 2 layers in GPU simultaneously

    Overhead: max(compute, PCIe) instead of compute + PCIe ✓
    GPU memory cost: +1 layer's worth of parameters


### Gradient Checkpointing vs Activation Offloading

These two techniques both reduce activation memory:

    **Gradient checkpointing (recomputation):**
    •   Cost: extra forward FLOPs (~33% overhead)
    •   Memory: O(√N) activations stored per N layers
    •   PCIe: zero (no data movement)
    •   Best for: almost all scenarios

    **Activation offloading (PCIe movement):**
    •   Cost: PCIe bandwidth for every activation
    •   Memory: activations on CPU RAM (abundant) not GPU (scarce)
    •   PCIe: 2 × activation_size per layer (forward: GPU→CPU, backward: CPU→GPU)
    •   Best for: only when PCIe is not the bottleneck AND activations
        dominate AND recomputation cost is prohibitive (e.g., with MoE
        experts where recomputation requires re-routing and is expensive)

For a typical Transformer block (B=4, T=2048, d=4096):
    activation_size ≈ 500 MB per block
    recompute_time  ≈ 50 ms (one extra forward pass through the block)
    PCIe transfer   ≈ 500 MB / 32 GB/s = 15.6 ms

At first glance, PCIe (15.6 ms) < recompute (50 ms) → offloading seems better!
But the comparison is unfair: recomputation is hidden behind backward compute,
while PCIe transfer also consumes bandwidth shared with other transfers.

In practice: **use activation checkpointing first; only consider activation
offloading if you have spare PCIe bandwidth and recomputation is expensive**.


### ZeRO-Infinity: NVMe-Based Parameter Storage

ZeRO-Infinity extends the offloading hierarchy to NVMe storage:

**NVMe bandwidth: 7 GB/s vs PCIe x16: 32 GB/s**
NVMe is 4.6× slower than PCIe. For NVMe offloading to be viable:
    compute_time >> NVMe_transfer_time
    layer_flops / GPU_FLOPS >> layer_bytes / 7_GB/s

    For a 7B model layer (33.6 MB, 68.7 GFLOPS at B=1, T=2048):
    NVMe transfer: 33.6 MB / 7 GB/s = 4.8 ms
    GPU compute:   68.7 GFLOPS / 312 TFLOPS ≈ 0.22 ms

    NVMe is 21× too slow to hide behind compute at B=1!
    At B=200: compute ≈ 44 ms >> NVMe transfer 4.8 ms ✓

NVMe offloading requires batch sizes an order of magnitude larger than
PCIe offloading. It is designed for inference serving with very large batches
or for parameter storage during checkpointing, not for general training.

**Tensor parallelism interaction with NVMe:**
Each TP rank's parameter shard is only 1/TP of the full layer. For TP=8,
each GPU's NVMe reads only 1/8 of the layer → 8× smaller NVMe transfers.
This makes NVMe offloading more feasible in TP+NVMe configurations.


### The Complete Offloading Decision Tree

                    Can model fit on GPU HBM with activation checkpointing?
                                    │
                           ┌────────┴────────┐
                           YES               NO
                           │                 │
                    Use DDP/FSDP       Can model fit with ZeRO-3?
                    (no offloading)            │
                                      ┌───────┴───────┐
                                      YES             NO
                                      │               │
                              Use ZeRO-3 FSDP    Is PCIe bandwidth sufficient
                              (no offloading)    for compute/transfer ratio?
                                                       │
                                              ┌────────┴────────┐
                                              YES               NO
                                              │                 │
                                     ZeRO-Offload CPU    ZeRO-Infinity (NVMe)
                                     (optimizer+grad)    Reduce batch/seq? Use
                                                         model compression?


### Practical Offloading Recipes

**Recipe 1: DeepSpeed ZeRO-Offload (7B on 1 GPU)**
    Works for: 7B model that won't fit on 80GB A100 with optimiser state
    Configuration:
        zero_optimization:
          stage: 2
          offload_optimizer:
            device: "cpu"
            pin_memory: True
          offload_param: False   # parameters stay on GPU
    Memory: GPU = 28 GB (weights+grads), CPU = 84 GB (optimizer)
    Throughput: ~0.7× vs GPU-only (PCIe bottleneck for optimizer step)

**Recipe 2: ZeRO-3 with CPU offload (13B on 1 GPU)**
    Works for: 13B model on 1× A100 80GB
    Configuration:
        offload_optimizer.device: "cpu"
        offload_param.device: "cpu"
    Memory: GPU = ~2 GB (active layers), CPU = ~206 GB
    Throughput: ~0.3-0.4× (significant PCIe overhead)
    Requires: large batch for good utilisation

**Recipe 3: ZeRO-Infinity NVMe (175B on 8 GPUs)**
    Works for: models that don't fit in CPU RAM
    Configuration:
        offload_optimizer.device: "nvme"
        offload_param.device: "nvme"
        nvme_path: "/mnt/nvme/"
    Memory: GPU = active layers, CPU = optimizer state, NVMe = params
    Throughput: 0.1-0.2× (NVMe is very slow)
    Use case: research, exploring model capabilities, not production training

**Recipe 4: PyTorch FSDP with CPU offload**
    from torch.distributed.fsdp import CPUOffload
    model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
    Equivalent to DeepSpeed ZeRO-Offload but native PyTorch.


### Measuring Offloading Efficiency

Key metrics to monitor when using CPU offloading:

    Metric                  Healthy value        Warning signal
    ──────────────────────────────────────────────────────────────────────
    GPU utilisation         > 85%               < 70% → PCIe bottleneck
    PCIe bandwidth usage    < 80% of max         = 100% → saturated
    Tokens/second           > 70% of no-offload  < 50% → too slow
    CPU RAM utilisation     < 90% of capacity    > 95% → OOM risk
    GPU HBM utilisation     70-95%               > 95% → OOM, < 60% → waste
    ──────────────────────────────────────────────────────────────────────

If GPU utilisation drops below 80% during offloaded training, the PCIe
transfer is the bottleneck. Solutions:
    1.  Increase batch size to improve arithmetic intensity
    2.  Enable prefetching (larger prefetch buckets)
    3.  Reduce precision of offloaded tensors (fp16 instead of fp32)
    4.  Use gradient accumulation to amortise transfer cost
    5.  Accept the overhead (if the only alternative is not training)
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
CPU Offloading Strategy Comparison

| Strategy              | GPU memory  | CPU RAM     | Throughput | Best for                              |
|-----------------------|-------------|-------------|------------|---------------------------------------|
| No offloading (DDP)   | 16P bytes   | 0           | 1× base    | Model fits on GPU                     |
| ZeRO-Offload (optim)  | 4P bytes    | 12P bytes   | ~0.7×      | Large optimizer state, fast PCIe      |
| ZeRO-Offload (param+) | ~0 bytes*   | 16P bytes   | ~0.3–0.5×  | Model >> GPU memory, big batches      |
| ZeRO-Infinity (NVMe)  | active only | active+opt  | ~0.1–0.2×  | Exploring very large models           |
| FSDP CPUOffload       | active only | 16P/N bytes | ~0.5×      | Native PyTorch, ZeRO-3 semantics      |

*"~0 bytes" means only the currently-active layer's parameters are on GPU
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "PCIe Bandwidth and Arithmetic Intensity Analysis": {
        "description": "Compute the arithmetic intensity breakeven point for CPU offloading at different batch sizes, sequence lengths, and model sizes — showing exactly when offloading becomes profitable.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
PCIE BANDWIDTH AND ARITHMETIC INTENSITY ANALYSIS
================================================================================

Answers the central question: when is CPU offloading profitable?

The key condition:
    compute_time ≥ pcie_transfer_time
    ⟺  layer_flops / GPU_FLOPS  ≥  layer_bytes / PCIe_bandwidth
    ⟺  arithmetic_intensity ≥ GPU_FLOPS / PCIe_bandwidth  (breakeven)

This module computes the breakeven intensity and shows at what batch sizes
and sequence lengths each model configuration crosses it.

================================================================================
"""

import math


# ── Hardware constants ────────────────────────────────────────────────────────

GPU_PEAK_TFLOPS    = 312.0    # A100 bf16 Tensor Core TFLOPS
PCIE_BANDWIDTH_GBS = 32.0     # PCIe 4.0 x16 bidirectional GB/s
HBM_BANDWIDTH_TBS  = 2.0      # A100 HBM bandwidth TB/s

# Breakeven: GPU must be doing at least this many FLOP per byte transferred
BREAKEVEN_FLOPS_PER_BYTE = GPU_PEAK_TFLOPS * 1e12 / (PCIE_BANDWIDTH_GBS * 1e9)


def flops_per_forward_layer(d_model: int, d_ff: int, seq_len: int,
                              batch: int, n_heads: int) -> float:
    """
    Estimate FLOPs for one Transformer block's forward pass.
    Includes: QKV projections, attention, FFN, LayerNorms.
    Dominant terms: 2 × d_model² × B×T  (attention projections + FFN).
    """
    B, T = batch, seq_len
    # Attention: 3 projections QKV + output
    attn_proj  = 4 * 2 * d_model * d_model * B * T   # 4 matmuls, each 2×d²×BT
    # Attention scores: 2 × B × H × T × T × d_head
    d_head = d_model // n_heads
    attn_scores = 2 * B * n_heads * T * T * d_head
    # FFN: 2 matmuls (W1 and W2)
    ffn        = 2 * 2 * d_model * d_ff * B * T
    return attn_proj + attn_scores + ffn


def bytes_to_transfer_layer(d_model: int, d_ff: int,
                              n_heads: int, dtype_bytes: int = 2) -> int:
    """
    Bytes to transfer one Transformer block's parameters over PCIe.
    Includes: QKV, O projections + FFN W1/W2 + LayerNorm.
    """
    qkv_o   = 4 * d_model * d_model * dtype_bytes    # 4 d×d matrices
    ffn_w   = 2 * d_model * d_ff   * dtype_bytes    # W1 and W2
    ln_w    = 2 * 2 * d_model      * dtype_bytes    # 2 LayerNorms, each 2d params
    return qkv_o + ffn_w + ln_w


def arithmetic_intensity(d: int, d_ff: int, h: int, B: int, T: int) -> float:
    """FLOP per byte for one Transformer block."""
    flops = flops_per_forward_layer(d, d_ff, T, B, h)
    nbytes = bytes_to_transfer_layer(d, d_ff, h)
    return flops / nbytes


def compute_time_ms(d: int, d_ff: int, h: int, B: int, T: int,
                     mfu: float = 0.45) -> float:
    flops = flops_per_forward_layer(d, d_ff, T, B, h)
    return flops / (GPU_PEAK_TFLOPS * 1e12 * mfu) * 1000


def pcie_transfer_time_ms(d: int, d_ff: int, h: int,
                            dtype_bytes: int = 2) -> float:
    nbytes = bytes_to_transfer_layer(d, d_ff, h, dtype_bytes)
    return nbytes / (PCIE_BANDWIDTH_GBS * 1e9) * 1000


def breakeven_batch(d: int, d_ff: int, h: int, T: int,
                     dtype_bytes: int = 2) -> int:
    """
    Minimum batch size at which compute_time ≥ pcie_transfer_time.
    Below this batch, offloading will reduce throughput.
    """
    transfer_ms = pcie_transfer_time_ms(d, d_ff, h, dtype_bytes)
    # compute_time ≥ transfer_time
    # flops(B) / (TFLOPS × mfu) ≥ transfer_time
    # flops(B) ≥ transfer_time × TFLOPS × mfu
    target_flops = transfer_ms / 1000 * GPU_PEAK_TFLOPS * 1e12 * 0.45
    # flops ≈ 2 × d_model² × B × T  (dominant term)
    approx_flops_per_sample = 2 * d * d * T
    return max(1, math.ceil(target_flops / approx_flops_per_sample))


def fmt_ms(ms: float) -> str:
    if ms >= 1000: return f"{ms/1000:.2f}s"
    if ms >= 1:   return f"{ms:.2f}ms"
    return f"{ms*1000:.2f}µs"


if __name__ == "__main__":
    print("=" * 70)
    print("  PCIe OFFLOADING ARITHMETIC INTENSITY ANALYSIS")
    print(f"  Breakeven intensity: {BREAKEVEN_FLOPS_PER_BYTE:.0f} FLOP/byte")
    print(f"  ({GPU_PEAK_TFLOPS} TFLOPS GPU / {PCIE_BANDWIDTH_GBS} GB/s PCIe)")
    print("=" * 70)
    print()

    models = [
        ("GPT-2 Small",   768,  3072, 12),
        ("LLaMA-2 7B",   4096, 11008, 32),
        ("LLaMA-2 70B",  8192, 28672, 64),
    ]

    for name, d, d_ff, h in models:
        transfer_ms  = pcie_transfer_time_ms(d, d_ff, h)
        layer_bytes  = bytes_to_transfer_layer(d, d_ff, h)

        print(f"  {name}  (d={d}, d_ff={d_ff})")
        print(f"  Layer weight bytes: {layer_bytes/1e6:.1f} MB  "
              f"PCIe transfer: {fmt_ms(transfer_ms)}")
        print()
        print(f"  {'B×T':>10}  {'Intensity':>14}  {'Compute':>12}  "
              f"{'PCIe':>10}  {'Profitable?':>14}  {'Overhead %':>12}")
        print(f"  {'':─>10}  {'':─>14}  {'':─>12}  "
              f"{'':─>10}  {'':─>14}  {'':─>12}")

        for B_T in [64, 256, 512, 1024, 2048, 4096, 8192, 16384]:
            B, T = 1, B_T
            intensity = arithmetic_intensity(d, d_ff, h, B, T)
            comp_ms   = compute_time_ms(d, d_ff, h, B, T)
            pcie_ms   = transfer_ms  # constant, independent of B and T
            profitable = intensity >= BREAKEVEN_FLOPS_PER_BYTE
            overhead   = max(0, pcie_ms - comp_ms) / comp_ms * 100 if not profitable else 0
            flag = "✓ profitable" if profitable else f"✗ {overhead:.0f}% slower"
            print(f"  {B_T:>10,}  {intensity:>14,.0f}  "
                  f"{fmt_ms(comp_ms):>12}  {fmt_ms(pcie_ms):>10}  "
                  f"{flag:>14}  {overhead:>11.0f}%")
        print()

        # Breakeven batch size
        for seq in [512, 2048, 4096]:
            be = breakeven_batch(d, d_ff, h, seq)
            print(f"  Breakeven batch (T={seq:>5}): B ≥ {be}")
        print()

    # Summary: transformer operation balance
    print("=" * 70)
    print("  DEVICE BALANCE ANALYSIS (A100)")
    print("=" * 70)
    print()
    print(f"  A100 compute:    {GPU_PEAK_TFLOPS:.0f} TFLOPS (bf16)")
    print(f"  A100 HBM BW:     {HBM_BANDWIDTH_TBS*1024:.0f} GB/s")
    print(f"  PCIe BW:         {PCIE_BANDWIDTH_GBS:.0f} GB/s")
    print()
    print(f"  Compute/HBM ratio:  {GPU_PEAK_TFLOPS*1e12/(HBM_BANDWIDTH_TBS*1e12):.0f} FLOP/byte")
    print(f"  Compute/PCIe ratio: {BREAKEVEN_FLOPS_PER_BYTE:.0f} FLOP/byte (offloading breakeven)")
    print()
    print("  Typical Transformer operation intensities:")
    ops = [
        ("MatMul (B=1, d=4096, d_ff=11008)",  2*4096*11008*1 / (4096*11008*2)),
        ("MatMul (B=32, d=4096, d_ff=11008)", 2*4096*11008*32 / (4096*11008*2)),
        ("LayerNorm",                          "~1 FLOP/byte (memory-bound)"),
        ("Element-wise ops",                   "~1 FLOP/byte (memory-bound)"),
    ]
    for name, intensity in ops:
        if isinstance(intensity, float):
            color = "✓" if intensity >= BREAKEVEN_FLOPS_PER_BYTE else "⚠️"
            print(f"  {color}  {name:<45}: {intensity:.0f} FLOP/byte")
        else:
            print(f"  ⚠️  {name:<45}: {intensity}")
''',
    },

    "CPU Offloading Simulator with Prefetching": {
        "description": "Simulate parameter offloading with and without prefetching — showing exact memory usage, PCIe transfer times, and how prefetching hides transfer latency behind compute.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
CPU OFFLOADING SIMULATOR WITH PREFETCHING
================================================================================

Simulates a forward pass with CPU offloading:
    - Parameters live on CPU
    - Before each layer: transfer to GPU (All-Gather equivalent)
    - Compute layer on GPU
    - After layer: return parameters to CPU (or keep if prefetch)

Two modes:
    1. Naive: compute waits for each transfer
    2. Prefetch: start next layer's transfer during current layer's compute

Measures:
    - Peak GPU HBM usage
    - Total wall-clock time
    - Throughput improvement from prefetching

================================================================================
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from threading import Thread
import queue


# ── Simulated hardware timing ─────────────────────────────────────────────────

# We simulate timing rather than actual PCIe (no real CPU offloading needed)
PCIE_SPEED_GBS   = 32.0    # GB/s
COMPUTE_TFLOPS   = 312.0   # TFLOPS peak (A100)
MFU              = 0.45    # model FLOP utilisation

def sim_pcie_time(bytes_: int) -> float:
    """Simulated PCIe transfer time in seconds."""
    return bytes_ / (PCIE_SPEED_GBS * 1e9)


def sim_compute_time(flops: float) -> float:
    """Simulated compute time in seconds."""
    return flops / (COMPUTE_TFLOPS * 1e12 * MFU)


# ── Layer definition ──────────────────────────────────────────────────────────

class CPUOffloadLayer(nn.Module):
    """
    A Transformer FFN block whose weights start on CPU.
    On forward, parameters are moved to GPU, computed, then freed.
    """
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.d, self.d_ff = d, d_ff
        # Store parameters on CPU
        self.W1 = nn.Parameter(torch.randn(d_ff, d), requires_grad=True)
        self.W2 = nn.Parameter(torch.randn(d, d_ff), requires_grad=True)
        # Move to CPU explicitly (usually already there at init)
        self.W1 = nn.Parameter(self.W1.cpu())
        self.W2 = nn.Parameter(self.W2.cpu())

        self._gpu_W1 = None
        self._gpu_W2 = None

    @property
    def param_bytes(self) -> int:
        return (self.W1.numel() + self.W2.numel()) * 2   # bf16

    def prefetch_to_gpu(self, device: torch.device) -> None:
        """Transfer parameters to GPU (non-blocking)."""
        self._gpu_W1 = self.W1.to(device, non_blocking=True).half()
        self._gpu_W2 = self.W2.to(device, non_blocking=True).half()

    def release_from_gpu(self) -> None:
        """Free GPU copies of parameters."""
        self._gpu_W1 = None
        self._gpu_W2 = None

    def forward_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using GPU copies of parameters."""
        assert self._gpu_W1 is not None, "Parameters not on GPU! Call prefetch_to_gpu first."
        return F.linear(F.gelu(F.linear(x.half(), self._gpu_W1)), self._gpu_W2).float()


# ── Naive offloading (no prefetch) ───────────────────────────────────────────

def run_naive_offload(layers: list, x: torch.Tensor,
                       device: torch.device, track_memory: bool = True):
    """
    Naive offloading: transfer layer to GPU, compute, release, repeat.
    GPU holds at most ONE layer's parameters at a time.
    """
    total_transfer_time  = 0.0
    total_compute_time   = 0.0
    peak_gpu_bytes       = 0

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    h = x
    for i, layer in enumerate(layers):
        # Transfer: CPU → GPU
        t0 = time.perf_counter()
        layer.prefetch_to_gpu(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        transfer_time = time.perf_counter() - t0
        total_transfer_time += transfer_time

        # Compute
        t0 = time.perf_counter()
        with torch.no_grad():
            h = layer.forward_gpu(h)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_time = time.perf_counter() - t0
        total_compute_time += compute_time

        # Release GPU memory
        layer.release_from_gpu()

        if track_memory and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            peak_gpu_bytes = max(peak_gpu_bytes, peak)

    return h, total_transfer_time, total_compute_time, peak_gpu_bytes


# ── Prefetch offloading ───────────────────────────────────────────────────────

def run_prefetch_offload(layers: list, x: torch.Tensor,
                          device: torch.device, prefetch_depth: int = 1):
    """
    Prefetch offloading: start transferring next layer during current compute.
    GPU holds at most (prefetch_depth + 1) layers at once.
    """
    total_wall_time  = 0.0
    peak_gpu_bytes   = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Prefetch the first layer
    layers[0].prefetch_to_gpu(device)

    h = x
    t_start = time.perf_counter()

    for i, layer in enumerate(layers):
        # Start prefetch for upcoming layers
        for pf_offset in range(1, prefetch_depth + 1):
            pf_idx = i + pf_offset
            if pf_idx < len(layers):
                layers[pf_idx].prefetch_to_gpu(device)

        # Compute current layer (overlap with prefetch)
        with torch.no_grad():
            h = layer.forward_gpu(h)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Release current layer's GPU parameters
        layer.release_from_gpu()

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            peak_gpu_bytes = max(peak_gpu_bytes, peak)

    total_wall_time = time.perf_counter() - t_start

    return h, total_wall_time, peak_gpu_bytes


# ── Memory budget tracker ─────────────────────────────────────────────────────

class MemoryBudget:
    """Track how GPU memory is allocated during offloaded forward pass."""

    def __init__(self, gpu_mem_gb: float):
        self.limit_bytes = gpu_mem_gb * 1e9
        self.alloc       = {}  # name → bytes
        self.timeline    = []

    def add(self, name: str, bytes_: int):
        self.alloc[name] = bytes_
        self.timeline.append(("alloc", name, bytes_, self.total))

    def free(self, name: str):
        if name in self.alloc:
            b = self.alloc.pop(name)
            self.timeline.append(("free", name, b, self.total))

    @property
    def total(self) -> int:
        return sum(self.alloc.values())

    def print_timeline(self):
        print(f"  {'Event':<30}  {'Bytes':>12}  {'Total GPU':>14}  {'% of 80GB':>12}")
        print(f"  {'':─<30}  {'':─>12}  {'':─>14}  {'':─>12}")
        for action, name, b, total in self.timeline:
            sign = "+" if action == "alloc" else "-"
            pct  = total / self.limit_bytes * 100
            print(f"  {action.upper()} {name:<25}  {sign}{b/1e6:>10.1f}MB  "
                  f"{total/1e9:>12.2f}GB  {pct:>11.1f}%")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import copy

    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D        = 512    # scaled down for demo
    D_FF     = 2048
    N_LAYERS = 6
    B, T     = 2, 32

    torch.manual_seed(0)
    layers = [CPUOffloadLayer(D, D_FF) for _ in range(N_LAYERS)]
    x      = torch.randn(B, T, D).to(DEVICE)

    print("=" * 65)
    print(f"  CPU OFFLOADING SIMULATION (simulated PCIe = {PCIE_SPEED_GBS} GB/s)")
    print(f"  {N_LAYERS} layers, d={D}, d_ff={D_FF}, B={B}, T={T}")
    print(f"  Each layer parameters: {layers[0].param_bytes/1e6:.1f} MB")
    print("=" * 65)
    print()

    # Naive offloading
    print("  [1] Naive offloading (no prefetch):")
    layers_naive = [copy.deepcopy(l) for l in layers]
    out_naive, t_transfer, t_compute, peak_naive = run_naive_offload(
        layers_naive, x.clone(), DEVICE
    )
    t_naive_total = t_transfer + t_compute
    print(f"    Total transfer time: {t_transfer*1000:.1f} ms")
    print(f"    Total compute time:  {t_compute*1000:.1f} ms")
    print(f"    Total wall time:     {t_naive_total*1000:.1f} ms")
    print(f"    Peak GPU memory:     {peak_naive/1e6:.1f} MB" if peak_naive > 0
          else f"    Peak GPU memory:     (CPU simulation)")
    print(f"    Efficiency:          {t_compute/(t_naive_total)*100:.1f}% "
          f"(rest is PCIe overhead)")
    print()

    # Prefetch offloading
    print("  [2] Prefetch offloading (1 layer ahead):")
    layers_pf = [copy.deepcopy(l) for l in layers]
    out_pf, t_pf_wall, peak_pf = run_prefetch_offload(
        layers_pf, x.clone(), DEVICE, prefetch_depth=1
    )
    speedup = t_naive_total / t_pf_wall if t_pf_wall > 0 else 1.0
    print(f"    Total wall time:     {t_pf_wall*1000:.1f} ms")
    print(f"    Speedup vs naive:    {speedup:.2f}×")
    print(f"    Peak GPU memory:     {peak_pf/1e6:.1f} MB  (2 layers in GPU)" if peak_pf > 0
          else f"    Peak GPU memory:     (CPU simulation)")
    print()

    # Simulate the GPU memory timeline
    print("=" * 65)
    print("  GPU MEMORY TIMELINE (naive vs prefetch)")
    print("=" * 65)
    print()

    GPU_MEM_GB  = 80.0
    budget      = MemoryBudget(GPU_MEM_GB)
    LAYER_BYTES = layers[0].param_bytes
    ACT_BYTES   = B * T * D * 4   # activations, fp32

    # Add activations (always present)
    budget.add("activations", ACT_BYTES)

    print("  NAIVE OFFLOADING (1 layer at a time in GPU):")
    for i in range(3):  # show first 3 layers
        budget.add(f"L{i}_params", LAYER_BYTES)
        budget.add(f"L{i}_output", ACT_BYTES)
        budget.free(f"L{i}_params")
        if i > 0:
            budget.free(f"L{i-1}_output")

    budget.print_timeline()
    print()

    budget2 = MemoryBudget(GPU_MEM_GB)
    budget2.add("activations", ACT_BYTES)
    print("  PREFETCH OFFLOADING (2 layers in GPU: current + next):")
    budget2.add("L0_params",  LAYER_BYTES)    # first prefetch
    for i in range(3):
        budget2.add(f"L{i+1}_params", LAYER_BYTES)  # prefetch next
        budget2.add(f"L{i}_output",   ACT_BYTES)
        budget2.free(f"L{i}_params")             # free after compute
        if i > 0:
            budget2.free(f"L{i-1}_output")

    budget2.print_timeline()
    print()
    print("  Trade-off: prefetch uses 2 layers' parameters simultaneously")
    print("  but hides PCIe transfer behind compute.")

    # Breakeven batch size demo
    print()
    print("=" * 65)
    print("  BREAKEVEN BATCH SIZE (when prefetch perfectly hides PCIe)")
    print("=" * 65)
    print()
    for layer_mb in [4, 16, 64, 256]:
        transfer_s   = layer_mb * 1e6 / (PCIE_SPEED_GBS * 1e9)
        # Compute time needed to hide transfer: must ≥ transfer_s
        # compute ≈ 2 × d × d_ff × B × T / (TFLOPS × MFU)
        # B × T ≥ transfer_s × TFLOPS × MFU / (2 × d × d_ff)
        required_bt = transfer_s * COMPUTE_TFLOPS * 1e12 * MFU / (2 * 512 * 2048)
        print(f"  Layer params = {layer_mb:>6} MB:  PCIe = {transfer_s*1000:>6.1f} ms  "
              f"  Breakeven B×T ≥ {required_bt:>8,.0f}  "
              f"(e.g. B={max(1,int(required_bt/512)):>3}×T=512)")
''',
    },

    "DeepSpeed-style ZeRO-Offload Implementation": {
        "description": "Implement the core mechanics of ZeRO-Offload: keep optimizer state on CPU, run the backward pass on GPU, offload gradients, run optimizer.step() on CPU, upload updated parameters.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ZERO-OFFLOAD: OPTIMIZER STATE ON CPU
================================================================================

Implements ZeRO-Offload's core pattern:
    - Model parameters and gradients live on GPU (during training step)
    - Optimizer state (fp32 master weights, m, v) lives on CPU
    - After backward: gradients transferred CPU → optimizer step → params to GPU

Shows the full data flow and memory savings.

================================================================================
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeROOffloadOptimizer:
    """
    ZeRO-Offload style optimizer wrapper.

    The model's parameters are in bf16 on GPU.
    Optimizer state (fp32 master weights + m + v) lives on CPU.

    On each step:
        1.  Gradients are transferred from GPU to CPU (fp32)
        2.  AdamW update runs on CPU using fp32 master weights
        3.  Updated bf16 parameters are transferred CPU → GPU

    This saves GPU memory at the cost of PCIe bandwidth.
    """

    def __init__(self, model: nn.Module, lr: float = 3e-4,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.model        = model
        self.lr           = lr
        self.beta1, self.beta2 = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0

        # Build CPU optimizer state
        self.cpu_params = {}  # name → fp32 master param on CPU
        self.cpu_m      = {}  # name → fp32 1st moment on CPU
        self.cpu_v      = {}  # name → fp32 2nd moment on CPU

        for name, param in model.named_parameters():
            if param.requires_grad:
                # fp32 master weight on CPU
                self.cpu_params[name] = param.data.float().cpu()
                self.cpu_m[name]      = torch.zeros_like(self.cpu_params[name])
                self.cpu_v[name]      = torch.zeros_like(self.cpu_params[name])

    @property
    def cpu_memory_bytes(self) -> int:
        """Total CPU memory used by optimizer state."""
        total = 0
        for name in self.cpu_params:
            total += (self.cpu_params[name].numel() +
                      self.cpu_m[name].numel() +
                      self.cpu_v[name].numel()) * 4  # fp32 = 4 bytes
        return total

    @property
    def gpu_param_bytes(self) -> int:
        """GPU memory used by bf16 model parameters."""
        return sum(p.numel() * 2   # bf16 = 2 bytes
                   for p in self.model.parameters() if p.requires_grad)

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None

    def step(self, verbose: bool = False) -> dict:
        """
        Execute ZeRO-Offload optimizer step.
        Returns timing breakdown.
        """
        self.t += 1
        t_gpu_to_cpu = 0.0
        t_cpu_update = 0.0
        t_cpu_to_gpu = 0.0

        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            # ── Phase 1: Transfer gradient GPU → CPU ──────────────────────────
            t0 = time.perf_counter()
            grad_cpu = param.grad.detach().float().cpu()
            t_gpu_to_cpu += time.perf_counter() - t0

            # ── Phase 2: AdamW update on CPU (fp32) ───────────────────────────
            t0 = time.perf_counter()
            p32  = self.cpu_params[name]
            m    = self.cpu_m[name]
            v    = self.cpu_v[name]

            # Weight decay applied to fp32 master weights
            p32.mul_(1 - self.lr * self.weight_decay)

            # Moment updates
            m.mul_(self.beta1).add_(grad_cpu, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad_cpu, grad_cpu, value=1 - self.beta2)

            # Bias correction
            bias_c1 = 1 - self.beta1 ** self.t
            bias_c2 = 1 - self.beta2 ** self.t
            step_size = self.lr / bias_c1
            denom = v.sqrt().div_(bias_c2 ** 0.5).add_(self.eps)

            # Parameter update (in-place on CPU fp32)
            p32.addcdiv_(m, denom, value=-step_size)
            t_cpu_update += time.perf_counter() - t0

            # ── Phase 3: Transfer updated bf16 params CPU → GPU ───────────────
            t0 = time.perf_counter()
            with torch.no_grad():
                param.data.copy_(p32.half().to(param.device,
                                               non_blocking=True))
            t_cpu_to_gpu += time.perf_counter() - t0

        return {
            "t_gpu_to_cpu_ms": t_gpu_to_cpu * 1000,
            "t_cpu_update_ms": t_cpu_update * 1000,
            "t_cpu_to_gpu_ms": t_cpu_to_gpu * 1000,
            "t_total_ms":      (t_gpu_to_cpu + t_cpu_update + t_cpu_to_gpu) * 1000,
        }


# ── Comparison: standard AdamW vs ZeRO-Offload ───────────────────────────────

class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 256, d: int = 128,
                 n_layers: int = 3, max_len: int = 32):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d * 4, bias=False),
                nn.GELU(),
                nn.Linear(d * 4, d, bias=False),
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(h)


def compare_optimizers(n_steps: int = 30):
    """Compare standard AdamW (GPU) vs ZeRO-Offload (CPU optimizer)."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB, D, N_LAYERS = 256, 128, 3
    B, T = 4, 16
    LR   = 3e-4

    torch.manual_seed(42)
    model_std  = TinyTransformer(VOCAB, D, N_LAYERS).to(DEVICE)
    model_off  = copy.deepcopy(model_std)

    # Cast to bf16 for realistic ZeRO-Offload scenario
    model_std  = model_std.half().to(DEVICE)
    model_off  = model_off.half().to(DEVICE)

    opt_std    = torch.optim.AdamW(model_std.parameters(), lr=LR,
                                    weight_decay=0.01)
    opt_off    = ZeROOffloadOptimizer(model_off, lr=LR, weight_decay=0.01)

    x = torch.randint(0, VOCAB, (B, T)).to(DEVICE)
    y = torch.randint(0, VOCAB, (B, T)).to(DEVICE)

    std_times  = []
    off_times  = []
    off_phases = {"t_gpu_to_cpu_ms": [], "t_cpu_update_ms": [], "t_cpu_to_gpu_ms": []}

    for step in range(n_steps):
        # Standard AdamW
        opt_std.zero_grad()
        t0   = time.perf_counter()
        loss = F.cross_entropy(model_std(x).view(-1, VOCAB).float(), y.view(-1))
        loss.backward()
        opt_std.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        std_times.append((time.perf_counter() - t0) * 1000)

        # ZeRO-Offload
        opt_off.zero_grad()
        t0   = time.perf_counter()
        loss = F.cross_entropy(model_off(x).view(-1, VOCAB).float(), y.view(-1))
        loss.backward()
        timing = opt_off.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        off_times.append((time.perf_counter() - t0) * 1000)
        for k in off_phases:
            off_phases[k].append(timing[k])

    avg_std = sum(std_times[5:]) / len(std_times[5:])
    avg_off = sum(off_times[5:]) / len(off_times[5:])
    avg_phases = {k: sum(v[5:]) / len(v[5:]) for k, v in off_phases.items()}

    n_params = sum(p.numel() for p in model_std.parameters())

    print("=" * 65)
    print("  ZERO-OFFLOAD vs STANDARD ADAMW COMPARISON")
    print(f"  {n_params:,} parameters, device={DEVICE}")
    print("=" * 65)
    print()
    print("  Memory footprint:")
    print(f"    Standard AdamW GPU: {n_params * (2+4+4+4) / 1e6:.1f} MB")
    print(f"    ZeRO-Offload GPU:   {opt_off.gpu_param_bytes / 1e6:.1f} MB (bf16 only)")
    print(f"    ZeRO-Offload CPU:   {opt_off.cpu_memory_bytes / 1e6:.1f} MB (fp32 master+m+v)")
    print(f"    GPU memory saved:   {(n_params * (4+4+4) - 0) / 1e6:.1f} MB  "
          f"({(4+4+4)/(2+4+4+4)*100:.0f}% of total)")
    print()
    print(f"  Throughput (avg over steps 5–{n_steps}):")
    print(f"    Standard AdamW:     {avg_std:.2f} ms/step")
    print(f"    ZeRO-Offload:       {avg_off:.2f} ms/step")
    print(f"    Overhead:           {(avg_off/avg_std - 1)*100:.1f}%")
    print()
    print(f"  ZeRO-Offload step breakdown:")
    for k, v in avg_phases.items():
        label = k.replace("_ms", "").replace("_", " → ")
        print(f"    {label:<25}: {v:.2f} ms  "
              f"({v/avg_phases['t_total_ms']*100:.0f}% of optimizer time)")
    print()
    print("  Note: overhead is small here (tiny model).")
    print("  For 7B model: optimizer state = 84 GB → PCIe cost is substantial.")


if __name__ == "__main__":
    compare_optimizers(n_steps=30)

    # Projected timings for large models
    print()
    print("=" * 65)
    print("  PROJECTED ZERO-OFFLOAD COST FOR LARGE MODELS (A100)")
    print("  (PCIe 4.0 x16 = 32 GB/s)")
    print("=" * 65)
    print()
    print(f"  {'Model':<15} {'Optim state':>14}  {'PCIe time/step':>16}  "
          f"{'Approx overhead':>18}")
    print(f"  {'':─<15} {'':─>14}  {'':─>16}  {'':─>18}")

    for name, n_params in [("7B",  7e9), ("13B", 13e9), ("70B", 70e9), ("175B", 175e9)]:
        optim_gb    = n_params * 12 / 1e9    # fp32 master + m + v
        pcie_time_s = optim_gb * 2 / 32      # 2× (grads down + params up)
        # Typical step time at reasonable batch size
        step_time_s = n_params * 6 * 4096 / (312e12 * 0.45)  # B*T=4096
        overhead    = pcie_time_s / step_time_s * 100
        print(f"  {name:<15} {optim_gb:>13.0f}GB  {pcie_time_s*1000:>14.0f}ms  "
              f"{overhead:>17.0f}%")

    print()
    print("  Overhead grows at larger models because optimizer state grows with P")
    print("  but compute grows with P × batch. Use large batches to amortise!")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html = ""
    visual_height = 600
    # try:
    #     from llm_training.visuals.cpu_offloading import (
    #         OFFLOAD_VISUAL_HTML,
    #         OFFLOAD_VISUAL_HEIGHT,
    #     )
    #     visual_html = OFFLOAD_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = OFFLOAD_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[25_cpu_offloading.py] Could not load visual: {e}",
    #         stacklevel=2,
    #     )

    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": visual_html,
        "visual_height": visual_height,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }