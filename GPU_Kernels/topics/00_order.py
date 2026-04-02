import textwrap
import re

TOPIC_NAME = "Order for GPU Kernels"
DISPLAY_NAME = "00 · Order for GPU Kernels"
ICON = "⚡"
SUBTITLE = "GPU Kernels curriculum order with priorities"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### GPU Kernels in AI/ML

A GPU kernel is a function that runs simultaneously across thousands of hardware threads.
Understanding how to write, optimize, and compose kernels is the difference between a model
that trains in days and one that trains in hours — and between inference that costs a dollar
per thousand requests and one that costs a cent. Every major efficiency breakthrough in
deep learning, from FlashAttention to quantized inference to fused operators, came from
someone understanding the GPU well enough to write a better kernel.

## The Core Problem

Modern GPUs contain thousands of cores that can execute in parallel, but they are not
fast for arbitrary code — they are fast for code that respects their memory hierarchy,
their threading model, and their instruction pipeline. A naive implementation of matrix
multiplication on a GPU can be 10–100× slower than an optimized one, even though both
produce identical results. The gap comes entirely from how the computation maps onto
hardware. Writing GPU kernels means learning to think in the hardware's terms: warps,
shared memory banks, register pressure, memory coalescing, occupancy, and pipeline
utilization.

## Two Levels of the Stack

Low-level kernel programming means writing CUDA C++, PTX assembly, or Triton DSL code
that directly expresses thread behavior, memory access patterns, and synchronization.
You control everything — and bugs are subtle, silent, and performance-destroying.
High-level libraries sit on top: cuBLAS, cuDNN, CUTLASS, and FlashAttention give you
hand-tuned kernels for standard operations without writing a line of CUDA. The skill
is knowing when each level is appropriate, and understanding the lower level well enough
to use the higher level correctly.

## Foundation: Memory and Threads

**Memory coalescing and shared memory** are the two most important concepts in GPU
programming. Coalescing means that threads in a warp access consecutive memory addresses
simultaneously — when they do, the hardware serves all 32 threads in a single transaction
rather than 32 separate ones. Shared memory is a small, fast, programmer-managed cache
(~48–100 KB per SM) that lives on-chip. The canonical optimization pattern is to load
a tile of global memory into shared memory once, process it with many threads, and write
results back — amortizing the expensive global memory access across many computations.
Every kernel optimization eventually traces back to these two primitives.

**Warp primitives** are CUDA's hardware-level communication intrinsics for threads within
a single warp (a group of 32 threads that execute in lockstep). Functions like
__shfl_sync, __ballot_sync, and __reduce_sync let threads exchange values and aggregate
results without going through shared memory at all — they operate on the register file
directly, which is faster and uses no shared memory bandwidth. Warp primitives are the
foundation of efficient reductions, prefix scans, and any operation where 32 threads need
to cooperate tightly.

**Reduction kernels** solve the problem of combining N values into one — computing a sum,
maximum, or any associative operation over an array. A naive parallel reduction leaves
most threads idle most of the time; an optimized one uses warp shuffle instructions,
avoids bank conflicts in shared memory, and keeps all threads busy through every phase.
Reductions appear everywhere: softmax requires a maximum reduction and a sum reduction;
layer normalization requires a mean and variance; loss computation requires a sum over
a batch. Getting reductions right is a prerequisite for getting everything else right.

**Prefix scan** (also called parallel prefix or inclusive/exclusive scan) computes the
running cumulative sum of an array in parallel. It underlies sorting algorithms, stream
compaction, load balancing, and sparse tensor operations. The challenge is that each
output depends on all prior inputs — a seemingly sequential dependency that must be
resolved in O(log N) parallel steps using an up-sweep and down-sweep tree structure.
Mastering prefix scan teaches the general technique of resolving data dependencies
through parallel tree reductions.

## Profiling and Benchmarking

**Nsight Compute** is NVIDIA's kernel-level profiler. It collects hardware performance
counters for individual kernel launches: memory bandwidth utilization, compute throughput,
warp efficiency, pipeline stalls, cache hit rates, and occupancy. It presents these
through a "roofline model" — a visual showing whether your kernel is memory-bound or
compute-bound and how far it sits from the hardware's theoretical peak. Nsight Compute
is how you move from "this kernel is slow" to "this kernel is slow because it has 40%
memory bandwidth utilization, and the bottleneck is uncoalesced L2 accesses from line 87."

**NVTX and NCV** (NVIDIA Tools Extension and NVIDIA CUDA-X Video) give you the ability
to annotate your own code with named ranges and markers that appear in profiler timelines.
Where Nsight Compute looks inside a single kernel, NVTX annotations let you mark which
part of your Python or C++ training loop corresponds to which GPU activity. When a
profiler timeline shows an unexpected gap, an NVTX range tells you exactly which
high-level operation caused it. This is the bridge between application logic and
hardware-level profiling.

**Nsight Systems** is the system-level profiler, one step above Nsight Compute. Where
Nsight Compute dives deep into a single kernel, Nsight Systems shows the full timeline:
CPU threads, GPU streams, PCIe transfers, CUDA API calls, kernel launches, and NCCL
collectives — all in a synchronized view. It answers questions like "why is there a 20ms
gap between these two kernel launches?" or "is my data pipeline saturating the CPU before
the GPU even starts?" Most performance investigations start here before drilling into
Nsight Compute for individual kernels.

## Dense Linear Algebra

**cuBLAS** is NVIDIA's hand-optimized library for dense linear algebra — matrix-matrix
multiplication (GEMM), matrix-vector multiplication (GEMV), triangular solves, and
factorizations. For standard precision (FP32, FP16, BF16) on standard shapes, cuBLAS
achieves close to theoretical peak performance. Virtually every deep learning framework
calls cuBLAS under the hood for fully connected layers and attention projections. Using
it directly gives you control over workspace memory, stream assignment, algorithm
selection, and batched variants that frameworks may not expose.

**Tensor cores and WMMA** are the hardware units on Ampere and later GPUs specifically
designed for small matrix multiplications (16×16×16 tiles in FP16/BF16, with
accumulation in FP32). Tensor cores deliver 8× or more throughput compared to CUDA cores
for the same operation. The WMMA (Warp Matrix Multiply Accumulate) API is the low-level
C++ interface for programming tensor cores directly — you load matrix fragments into
registers, call wmma::mma_sync, and store the result. Understanding WMMA is essential
for writing custom kernels that exploit tensor core throughput for non-standard operations.

**Mixed precision training** uses FP16 or BF16 for the forward pass and gradient computation
(exploiting tensor core throughput) while maintaining FP32 "master weights" for the
optimizer step (where numerical precision matters). The critical technique is loss scaling:
multiplying the loss by a large constant before the backward pass to prevent FP16 gradients
from underflowing to zero, then unscaling before the optimizer update. Getting mixed
precision right requires understanding where precision matters (optimizer state, gradient
accumulation) versus where it does not (activations, weight matrices).

**cuDNN** is NVIDIA's library for deep learning primitives beyond basic linear algebra —
convolutions, pooling, batch normalization, dropout, RNNs, and attention. cuDNN selects
the best algorithm for each operation based on input shape, hardware generation, and
workspace budget, using an auto-tuning search at first use. It's the performance backbone
of TensorFlow and PyTorch's convolution and RNN layers. For inference deployment,
cuDNN's graph API lets you fuse sequences of operations into a single optimized kernel.

**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's open-source
C++ template library for writing custom GEMM kernels. Where cuBLAS is a black box,
CUTLASS exposes the full tiling hierarchy — threadblock tiles, warp tiles, and instruction
tiles — as composable C++ templates. It's the right tool when you need a matrix
multiplication with custom epilogues (applying a nonlinearity, adding a bias, or computing
a fused softmax directly in the GEMM), or when you need GEMM on shapes or data types
that cuBLAS doesn't support. Most ML frameworks use CUTLASS internally for their
custom operator kernels.

## ML-Specific Kernels

**FlashAttention** (Dao et al., 2022; Dao, 2023) is the algorithmic and kernel-level
rewrite of the attention mechanism that reduced attention's memory complexity from O(N²)
to O(N) and made long-context training practical. Standard attention materializes the full
N×N attention matrix in HBM (high bandwidth memory), which becomes the bottleneck for
sequences beyond a few thousand tokens. FlashAttention avoids this by computing attention
in tiles that fit entirely in SRAM — the fast on-chip shared memory — and using online
softmax to accumulate the correct result across tiles without ever writing the full
attention matrix to HBM. The result is 2–4× faster attention with the same numerical
output. FlashAttention is now the default attention implementation in virtually every
serious training infrastructure.

**Fused kernels** combine operations that would normally be separate kernel launches into
a single kernel that keeps data on-chip between steps. The classic example is the
"LayerNorm + linear" fusion: rather than writing activations to HBM, reading them back
for the normalization, writing again, and reading again for the linear, you do all three
inside one kernel using registers and shared memory. Fusion is the single most impactful
optimization technique for inference: a typical transformer forward pass can go from
dozens of kernel launches to a handful, with dramatic reductions in memory bandwidth and
kernel launch overhead. PyTorch 2.0's `torch.compile` and XLA's fusion pass automate
some of this, but hand-written fused kernels still outperform compiler output for
performance-critical paths.

**Quantization kernels** implement inference in INT8, INT4, or lower precision, reducing
memory bandwidth and exploiting integer arithmetic units for throughput gains. The
challenge is implementing quantization correctly at the kernel level: loading packed
integer weights, dequantizing them on-the-fly before or during the matrix multiply,
and handling the scaling and zero-point arithmetic without precision loss. GPTQ, AWQ,
and bitsandbytes each implement different quantization schemes with different kernel
strategies. Understanding quantization kernels is the key to deploying large models on
memory-constrained hardware without unacceptable accuracy degradation.

**Persistent kernels** keep a kernel running across many batches of work rather than
launching a new kernel for each one. Normal kernel launches have overhead: the CUDA
driver schedules the launch, initializes thread state, and tears down after completion.
For small operations that launch frequently — like the token-by-token generation in LLM
inference — this overhead can dominate runtime. A persistent kernel sits on the GPU
continuously, pulling new work from a queue in global memory. Persistent kernels are the
foundation of continuous batching in LLM serving systems.

**KV cache management** is the memory management problem specific to autoregressive LLM
inference. At each generation step, the attention keys and values for all prior tokens
must be stored and reused — the KV cache. For long sequences and large batches, the KV
cache can consume tens of gigabytes. Efficient KV cache management involves paged
allocation (PagedAttention, used in vLLM), prefix sharing, and quantizing the cache
itself. The kernel-level challenge is implementing attention that reads from a
non-contiguous, paged memory layout efficiently — a substantially harder problem than
standard attention over a contiguous tensor.

## FFT and Sparse Operations

**cuFFT** is NVIDIA's Fast Fourier Transform library. FFT operations appear in signal
processing, convolutional layers (convolution via FFT for large kernels), spectral
methods in physics simulations, and increasingly in ML architectures that use frequency-
domain representations. cuFFT handles 1D, 2D, and 3D transforms in single and double
precision, with batched variants for processing many sequences simultaneously. The
performance of cuFFT is highly sensitive to transform size — powers of two and highly
composite numbers are fast; prime sizes are much slower.

**cuSPARSE** provides sparse matrix operations for matrices where most entries are zero.
Sparse formats like CSR, CSC, and COO store only non-zero values, dramatically reducing
memory and compute for sufficiently sparse problems. cuSPARSE's SpMM (sparse matrix ×
dense matrix) and SpGEMM (sparse × sparse) operations are fundamental to graph neural
networks, sparse attention mechanisms, and pruned model inference. The efficiency of
sparse operations on GPU depends critically on sparsity patterns — structured sparsity
(entire rows or blocks of zeros) is much faster than unstructured sparsity.

**Sparse GEMM and GEMV** are the sparse counterparts of dense matrix operations, relevant
for both explicitly sparse computations and for the emerging area of sparse weight
matrices in pruned or mixture-of-experts models. NVIDIA's Ampere architecture introduced
hardware support for 2:4 structured sparsity (exactly 2 non-zeros in every group of 4
values) through its sparse tensor cores, delivering up to 2× throughput for eligible weight
matrices without changing the programming model.

## Systems and Tooling

**CUDA graphs** capture a sequence of kernel launches and memory operations into a
replayable graph object. Instead of re-issuing each kernel launch through the CUDA driver
every iteration — each with its own CPU overhead and scheduling latency — you capture the
graph once and replay it with a single API call. For training workloads with fixed
computation structure (fixed batch size, fixed sequence length), CUDA graphs can reduce
per-iteration CPU overhead from milliseconds to microseconds and eliminate the PCIe
synchronization bubbles between kernel launches. PyTorch's `torch.compile` and `make_graphed_callables`
expose CUDA graphs to Python users.

**CUDA streams and asynchronous execution** are the mechanisms for overlapping independent
GPU operations. A stream is a queue of ordered operations; operations in different streams
can execute concurrently if hardware resources allow. The canonical use case is overlapping
computation with data transfer: while the GPU processes batch N, the CPU asynchronously
copies batch N+1 to GPU memory using a separate stream, so no GPU time is wasted waiting
for data. Streams are also essential for overlapping communication with computation in
multi-GPU training — the all-reduce for gradient synchronization runs in a separate stream
from the backward pass computation.

**Multi-GPU memory management** covers the strategies for distributing model state across
multiple devices. Tensor parallelism shards weight matrices across GPUs so each holds
a slice. Pipeline parallelism assigns different layers to different GPUs. ZeRO (Zero
Redundancy Optimizer) partitions optimizer state, gradients, and parameters across data
parallel replicas to eliminate redundancy. Each strategy has different communication
patterns, memory footprints, and kernel implications — sharded operations require gather
and all-reduce kernels at boundaries, and the placement of those boundaries determines
whether communication and computation can overlap.

**Multi-GPU training** brings together all of the above — data parallelism, model
parallelism, gradient synchronization, and stream management — into a coherent distributed
training strategy. At the kernel level, the key concerns are minimizing communication
volume, overlapping communication with backward computation, and ensuring that gradient
buckets are all-reduced in the right order. Libraries like DeepSpeed and Megatron-LM
implement specific combinations of these strategies; understanding the kernel primitives
behind them is essential for tuning or debugging at scale.

**NCCL collectives** (NVIDIA Collective Communications Library) implement the communication
primitives for multi-GPU and multi-node training: all-reduce, broadcast, scatter, gather,
all-gather, and reduce-scatter. NCCL automatically chooses communication algorithms based
on topology — ring algorithms for single-node multi-GPU, tree algorithms across nodes —
and uses NVLink or InfiniBand depending on what hardware is available. For distributed
training, NCCL's all-reduce is the most critical call: every gradient synchronization
in data-parallel training goes through it, and its latency and bandwidth directly set
the floor on per-step training time.

**Triton** is an open-source Python DSL and compiler for writing GPU kernels at a level
between CUDA and Python. You write kernels as Python functions decorated with @triton.jit,
use Triton's block-level abstractions (tl.load, tl.store, tl.dot), and the compiler
handles warp-level details, shared memory allocation, and instruction selection. Triton
makes it feasible to write high-performance fused kernels in a day rather than a week —
FlashAttention 2 and 3 provide Triton implementations that achieve near-CUDA-C performance.
It's the right tool for custom ML kernels that don't fit neatly into cuBLAS/cuDNN but
aren't worth the full engineering cost of hand-written CUDA.

**PTX and SASS** are the assembly languages of NVIDIA GPUs. PTX (Parallel Thread Execution)
is a virtual ISA that NVIDIA's compiler targets — it's portable across GPU generations
and gets compiled to machine code at load time. SASS is the actual machine code executed
by the hardware. Reading PTX and SASS lets you verify that the compiler generated the
instructions you intended, diagnose register spills and bank conflicts at the instruction
level, use inline PTX for operations that have no high-level API (like reading performance
counters or using specialized hardware instructions), and understand why the compiler made
the choices it did. PTX/SASS analysis is the deepest layer of kernel optimization — the
last resort when Nsight Compute says you're leaving performance on the table but the
source code looks correct.

## How the Layers Connect

A production ML inference system uses every layer of this stack simultaneously. A
transformer model forward pass calls cuBLAS for projection layers, FlashAttention for
self-attention (itself a fused kernel built on warp primitives and shared memory tiling),
cuDNN for layer normalization, and custom quantization kernels for INT8 weight loading.
Persistent kernels manage the generation loop, KV cache management handles memory
for long sequences, and CUDA graphs eliminate per-step launch overhead. NCCL all-reduce
synchronizes gradients during training, with communication overlapping the backward pass
via separate CUDA streams. Nsight Systems and Nsight Compute profile the system end-to-end
and per-kernel respectively, and Triton enables the team to prototype custom fused kernels
quickly before committing to a hand-tuned CUDA implementation.

Understanding GPU kernels isn't about memorizing CUDA syntax. It's about building a
mental model of the hardware — how threads are organized, where data lives at each level
of the memory hierarchy, what limits throughput, and how to express computation so the
hardware can execute it at maximum efficiency. Every abstraction in this stack, from
cuBLAS to NCCL to Triton, is a set of decisions made by people who understood the hardware
deeply enough to encode best practices into a reusable form. Understanding why those
decisions were made is how you use the abstractions well — and how you know when to
break through them.


    GPU_Kernels/
    └── topics/
        ├── 00_order.py
        ├── 01_foundation_memory_coalescing_shared_mem.py
        ├── 02_foundation_warp_primitives.py
        ├── 03_foundation_reduction_kernels.py
        ├── 04_foundation_prefix_scan.py
        ├── 05_profiling_benchmarking_NsightCompute.py
        ├── 06_profiling_benchmarking_NCV_NVTX.py
        ├── 06a_profiling_benchmarking_NsightSystems.py
        ├── 07_Dense_Linear_cublas.py
        ├── 08_Dense_Linear_tensor_cores_wmma.py
        ├── 09_Dense_Linear_mixed_precision.py
        ├── 10_Dense_Linear_cuDNN.py
        ├── 11_Dense_Linear_CUTLASS.py
        ├── 12_ML_Specific_Flash_Attention.py
        ├── 13_ML_Specific_Fused_Kernels.py
        ├── 14_ML_Specific_Quantization_Kernels.py
        ├── 14a_ML_Specific_Quantization_Persistent_kernels.py
        ├── 14b_ML_Specific_Quantization_kv_cache_management.py
        ├── 15_FFT_SPARSE_cufft.py
        ├── 15a_FFT_SPARSE_cusparse.py
        ├── 15b_FFT_SPARSE_GEMM_GEMV.py
        ├── 16_Systems_Tooling_cuda_graphs.py
        ├── 17_Systems_Tooling_streams_async.py
        ├── 17a_Systems_Tooling_multi_gpu_memory.py
        ├── 17b_Multi_GPU_Training.py
        ├── 18_Systems_Tooling_nccl_collectives.py
        ├── 19_Systems_Tooling_Triton.py
        └── 20_Systems_Tooling_ptx_sass.py

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }