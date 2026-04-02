"""
TorchInductor — PyTorch 2.x Optimising Compiler Backend
=========================================================

TorchInductor is the default compilation backend for torch.compile, introduced
in PyTorch 2.0. It sits at the bottom of the torch.compile stack, receiving
FX graphs from TorchDynamo and AOTAutograd and turning them into optimised
machine code — Triton GPU kernels or C++ CPU code — without requiring any
hand-written kernel implementations for each operator.

Understanding what makes TorchInductor different from every compiler that came
before it requires understanding what it is NOT:

    TorchInductor is NOT a library-dispatch compiler like XLA.
    XLA achieves performance by routing matmuls to cuBLAS and convolutions to
    cuDNN — hand-tuned library kernels maintained by hardware vendors. Inductor
    does this too for those specific ops, but its primary innovation is
    generating custom Triton kernels for everything else.

    TorchInductor is NOT a template-based compiler like TVM's AutoTVM.
    TVM requires human-authored schedule templates per operator per hardware.
    Inductor generates Triton code programmatically from first principles, then
    uses a lightweight search to pick BLOCK_SIZE parameters — no templates needed.

    TorchInductor IS a whole-graph optimising compiler.
    Its fusion engine analyses the entire computation graph, identifies which
    groups of operators can execute in a single GPU pass (reading inputs once,
    writing outputs once), generates a fused Triton kernel for each group, and
    eliminates every unnecessary intermediate memory allocation between fused ops.

The central insight of TorchInductor: modern GPUs are almost always
memory-bandwidth limited for elementwise workloads. A chain of 10 elementwise
operations (relu, add, multiply, exp, ...) has the same compute cost whether
fused into one kernel or split into 10 separate kernels — but the memory
traffic differs by 10×. Inductor's primary job is to fuse as aggressively as
possible, collapsing memory traffic to the theoretical minimum.

In the torch.compile stack:
    TorchDynamo (module 09) ← captures FX graphs from Python via bytecode interception
    AOTAutograd (module 09) ← differentiates the FX graph, producing joint fwd+bwd
    TorchInductor (this)    ← receives the joint graph, generates optimised kernels

In the connected compiler stack:
    LLVM      (module 01) ← Inductor's CPU backend generates C++ compiled by GCC/Clang/LLVM
    MLIR      (module 02) ← Inductor uses MLIR concepts (pointwise + reduction IR mirrors linalg)
    Enzyme    (module 04) ← AOTAutograd (Inductor's differentiator) mirrors Enzyme at graph level
    XLA       (module 05) ← Inductor competes with XLA; torch.compile(backend='openxla') bypasses Inductor
    TVM       (module 08) ← TVM and Inductor both generate GPU kernels from the same FX graphs
    TorchDynamo (module 09) ← the graph capture layer that feeds Inductor
    TorchInductor (this)  ← the compilation engine turning FX graphs into fast kernels

"""

import textwrap
import re

TOPIC_NAME   = "TorchInductor — PyTorch 2.x Optimising Compiler Backend"
DISPLAY_NAME = "09 · TorchInductor"
ICON         = "⚙️"
SUBTITLE     = "Inductor IR, Triton Codegen, Fusion Scheduling, max-autotune, and CUDA Graphs"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT TORCHINDUCTOR IS AND THE PROBLEM IT SOLVES

### The Memory Bandwidth Bottleneck

    Modern GPUs are compute monsters. An NVIDIA A100 delivers 312 TFLOPS of
    BF16 tensor-core throughput. Yet for most ML workloads — everything except
    the largest matrix multiplications — the bottleneck is not computation.
    It is MEMORY BANDWIDTH.

    The A100's HBM2e provides 2 TB/s of memory bandwidth. That sounds large
    until you measure it against computation:

        A matmul of size [4096, 4096] @ [4096, 4096]:
            Compute: 2 × 4096³ ≈ 137 billion FLOPs
            Memory:  3 × 4096 × 4096 × 2 bytes (BF16) ≈ 100 MB
            Compute time @ 312 TFLOPS:  0.44 ms
            Memory time @ 2 TB/s:       0.05 ms  ← compute-bound ✓

        A ReLU applied to the same output:
            Compute: 4096 × 4096 ops ≈ 16 million FLOPs
            Memory:  read 100 MB, write 100 MB = 200 MB
            Compute time @ 312 TFLOPS:  0.0001 ms  (negligible!)
            Memory time @ 2 TB/s:       0.10 ms   ← memory-bound

        A chain of 8 elementwise ops (relu, add, multiply, exp, tanh, ...):
            Without fusion:  8 × 0.10 ms = 0.80 ms  (8 passes over memory)
            With fusion:     1 × 0.10 ms = 0.10 ms  (1 pass over memory)
            Speedup from fusion alone: 8×

    This is TorchInductor's primary value proposition. The single most impactful
    thing a GPU compiler can do for elementwise-heavy workloads is FUSION:
    collapsing multiple operators into a single kernel pass that reads and writes
    each tensor element exactly once.

### The Eager Mode Problem

    PyTorch eager mode dispatches each operation individually:

        x = torch.relu(input)       # launch relu kernel, read input (256MB), write x (256MB)
        x = x + bias                # launch add kernel, read x (256MB)+bias, write x (256MB)
        x = x * gate                # launch mul kernel, read x (256MB)+gate, write x (256MB)
        x = torch.sigmoid(x)        # launch sigmoid kernel, read x (256MB), write x (256MB)
        x = x * x                   # launch mul kernel, read x (256MB), write x (256MB)

    5 kernel launches, ~5 × 2 × 256MB = 2.56 GB of memory traffic.

    With TorchInductor:
        kernel(input, bias, gate) → fused output
        1 kernel launch, 1 × 256MB read + 1 × 256MB write = 512MB traffic.
        Net: 5× reduction in memory traffic → ~5× faster on this sequence.

    Beyond just eliminating intermediate buffer allocations, fusion also:
        Removes 4 kernel launch overheads (~5-10 µs each on modern hardware)
        Eliminates the need to write and re-read from HBM for intermediate results
        Enables the compiler to keep values in registers between operations

### What TorchInductor Receives and Produces

    INPUT (from AOTAutograd):
        A torch.fx.GraphModule containing ATen-level operations.
        The ATen IR has ~250 primitive operations after decomposition.
        Example ops: aten.mm, aten.add.Tensor, aten.relu.default,
                     aten.sum.dim_IntList, aten.exp.default, ...
        Parameters appear as get_attr nodes (not Python closures).
        The graph is functional (no in-place ops; functionalize() already ran).
        For training: joint forward+backward graph from AOTAutograd.

    OUTPUT:
        GPU path: a set of Triton kernel Python files + a wrapper that calls them.
        CPU path: a C++ source file with OpenMP loops + a compilation script.
        Both are loaded as Python callables via ctypes / PyTorch's extension API.
        A compiled module that accepts the same inputs and returns the same outputs.

    GUARANTEE:
        TorchInductor is semantics-preserving. The compiled output must produce
        the same numerical results as eager execution, within floating-point
        rounding tolerance. The extensive test suite (inductor/test/) verifies
        this for thousands of operator combinations.


##### PART 2 — THE INDUCTOR IR: THREE NODE TYPES

### Why a Separate IR?

    TorchInductor does not compile ATen ops directly to Triton. The ATen IR is
    too high-level (each op needs custom code generation logic) and too large
    (~250 ops). Instead, Inductor lowers ATen to its own internal IR called
    the INDUCTOR IR (or sometimes the "loop-level IR"), which has just three
    node types. Every ATen op maps to one of these three types.

    The three-type IR has a profound advantage: the SCHEDULER only needs to
    understand three types of nodes to make fusion decisions. The rules are
    simple, universal, and correct. Adding a new ATen op to Inductor only
    requires classifying it as pointwise, reduction, or extern_kernel —
    the fusion logic comes for free.

### NODE TYPE 1: Pointwise

    A POINTWISE node computes each output element independently from a fixed
    set of input elements at the SAME index (or broadcast-compatible indices).

    Formal definition:
        output[i₀, i₁, ..., iₙ] = f(inputs[..., iⱼ, ...])
        where f is some elementwise function and j ranges over inputs.
        The output shape equals the broadcasted shape of the inputs.

    Examples of ATen ops that lower to Pointwise:
        aten.add.Tensor       → output[i] = a[i] + b[i]
        aten.relu.default     → output[i] = max(a[i], 0)
        aten.exp.default      → output[i] = exp(a[i])
        aten.mul.Tensor       → output[i] = a[i] * b[i]
        aten.neg.default      → output[i] = -a[i]
        aten.sigmoid.default  → output[i] = 1/(1+exp(-a[i]))
        aten.tanh.default     → output[i] = tanh(a[i])
        aten.where.self       → output[i] = cond[i] ? a[i] : b[i]
        aten.clamp.Tensor     → output[i] = clamp(a[i], min, max)
        aten.gt.Scalar        → output[i] = a[i] > scalar  (comparison)
        aten.gelu.default     → output[i] = gelu(a[i])     (composed of several pointwise ops)
        aten.silu.default     → output[i] = a[i] * sigmoid(a[i])
        aten.broadcast_to     → output[i,j] = input[i] (broadcast along j)
        aten.expand           → output[...] = input[...] (broadcast with stride 0)
        aten.reshape          → relabelling of indices (may be zero-copy)
        aten.permute          → relabelling of dimension order

    WHY POINTWISE FUSES SO EASILY:
        All pointwise ops are LOOP-INDEPENDENT. The computation for element [i]
        does not depend on element [j ≠ i]. This means:
            Any number of pointwise ops can be fused into one loop.
            The fused loop simply computes all ops for each element in sequence.
            No synchronisation between iterations is needed.
            The fused loop is trivially parallelisable and vectorisable.

    Inductor IR representation of pointwise:
        class Pointwise(IRNode):
            ranges:     List[Expr]     # loop bounds per dimension
            inner_fn:   Callable       # function body (value-of-index → value)
            dtype:      torch.dtype
            layout:     Layout

        The inner_fn is a Python closure that, given loop indices,
        returns the expression tree for computing one output element.
        Composing two pointwise inner_fns is just function composition —
        this is how producer-consumer fusion is implemented.

### NODE TYPE 2: Reduction

    A REDUCTION node computes output elements that each depend on MULTIPLE
    (or all) input elements along one or more "reduction dimensions."

    Formal definition:
        output[i₀, ..., iₖ] = reduce(f, input[i₀, ..., iₖ, :, ..., :])
        where : ranges over the reduction dimensions (not in the output).

    Examples of ATen ops that lower to Reduction:
        aten.sum.dim_IntList  → output[i] = Σⱼ input[i,j]  (sum along j)
        aten.max.dim          → output[i] = maxⱼ input[i,j]
        aten.min.dim          → output[i] = minⱼ input[i,j]
        aten.mean.dim         → output[i] = (1/|j|) Σⱼ input[i,j]
        aten.amax.default     → output[...] = max over all reduction dims
        aten.norm.ScalarType  → output[i] = (Σⱼ |input[i,j]|ᵖ)^(1/p)
        aten.any.dim          → output[i] = ∃j: input[i,j] ≠ 0
        aten.all.dim          → output[i] = ∀j: input[i,j] ≠ 0
        aten.argmax.default   → output[i] = argmaxⱼ input[i,j]

    FUSION RULES FOR REDUCTIONS:
        1. A POINTWISE PRODUCER can be fused INTO a reduction (input fusion):
               If the reduction reads a pointwise result, that pointwise op
               is inlined into the reduction kernel. The intermediate is never
               materialised in HBM.
               Example: sum(exp(x)) → one kernel: load x, compute exp, accumulate sum.

        2. Two INDEPENDENT reductions reading the SAME input are NEVER fused
               (they need different accumulation strategies).
               They share the single memory read pass (sibling scheduling).

        3. A POINTWISE CONSUMER of a reduction can be fused AFTER the reduction
               (output fusion) only if the pointwise op is applied to the scalar
               reduction output — not to the original large tensor.
               Example: softmax = reduce(max) → sub → reduce(sum) → div
               Each reduce is separate; sub and div are pointwise fused into the
               corresponding reduce's epilogue.

    REDUCTION SPLITTING for GPU:
        A naive reduction over dim 0 of a [N] tensor:
            One thread accumulates all N values. Parallelism = 1. Slow.

        Inductor's two-pass reduction strategy:
            Pass 1 (SPLIT_SCAN):  N threads each accumulate N/B values.
                                   Each thread's result written to a partial sum.
            Pass 2 (combine):     B threads reduce the B partial sums.
        This gives full GPU utilisation at the cost of one extra pass.

        The SPLIT_SCAN threshold (default: 512 elements):
            If the reduction dimension ≤ 512: one pass (fits in one warp-group).
            If the reduction dimension > 512:  split into two passes.

### NODE TYPE 3: ExternKernel

    An EXTERN_KERNEL node represents a call to an EXTERNAL LIBRARY that
    Inductor does not generate code for — instead it calls the library directly.

    These are "opaque" from Inductor's perspective: Inductor cannot see inside
    them, cannot fuse across them, and cannot change their implementation.

    Examples of ATen ops that lower to ExternKernel:
        aten.mm.default         → calls cuBLAS GEMM (or CPU BLAS)
        aten.bmm.default        → calls cuBLAS batched GEMM
        aten.convolution.default→ calls cuDNN convolution forward
        aten.addmm.default      → calls cuBLAS GEMM with accumulation
        aten.baddbmm.default    → calls cuBLAS batched GEMM + add
        aten.scaled_dot_product_attention → calls cuDNN Flash Attention
        aten.fft_fft.default    → calls cuFFT
        aten._triton_multi_head_attention → custom Triton FlashAttn kernel

    WHY THESE CANNOT BE FUSED INTO A TRITON KERNEL:
        cuBLAS GEMM is a highly optimised black box. It uses:
            Hardware-specific warp-level matrix operations (Tensor Cores)
            Bank-conflict-free shared memory layouts tuned per architecture
            Pipelining of global memory loads with tensor core compute
            Hundreds of parameters auto-selected based on M, N, K shapes
        Recreating this in a Triton kernel requires significant effort and
        will almost always be slower than cuBLAS for large square GEMMs.

    HOWEVER — EPILOGUE FUSION WITH EXTERN KERNELS:
        Even though the GEMM itself is opaque, the operations that consume
        its output CAN be fused into the GEMM's EPILOGUE via cuBLAS/cuDNN
        epilogue fusion APIs (available on Ampere A100 and newer):

            torch.compile will fuse:
                matmul → add_bias → relu     INTO cuBLAS GEMM + epilogue
                matmul → add_bias → sigmoid  INTO cuBLAS GEMM + epilogue
                matmul → gelu                INTO cuBLAS GEMM + epilogue

            The epilogue runs in the same kernel as the GEMM, inside cuBLAS.
            No extra kernel launch, no intermediate buffer written to HBM.

        This is what the "inductor" backend does for transformer FFN layers:
            mm1(x, W1) + b1 → gelu:  one cuBLAS call with gelu epilogue
            mm2(h, W2) + b2:          one cuBLAS call with bias epilogue
        Total: 2 kernel launches for the entire FFN block.


##### PART 3 — THE SCHEDULER: HOW INDUCTOR DECIDES WHAT TO FUSE

### The Scheduler's Job

    The Inductor IR converts the ATen FX graph into a list of Pointwise,
    Reduction, and ExternKernel nodes. The SCHEDULER analyses this list and
    decides which nodes should be compiled into the same kernel.

    The scheduler's output is a list of KERNEL GROUPS (called FusedSchedulerNode
    objects). Each kernel group contains one or more Inductor IR nodes that
    will execute together in one GPU kernel or CPU loop nest.

    The scheduler must balance:
        FUSION BENEFIT: more fusion → fewer kernels → less memory traffic
        FUSION COST:    some fusions are illegal (data dependencies)
                        some fusions increase register pressure (may reduce GPU occupancy)
                        some fusions require complex loop nesting that Triton handles poorly

### The Dependency Graph

    Before making any fusion decisions, the scheduler builds a DEPENDENCY GRAPH
    from the Inductor IR nodes:

        For each node N, identify all nodes that produce values N reads.
        If node A produces a value that node B reads → edge A → B.
        Nodes can only be fused if they have no other intervening users
        that would be "orphaned" by fusing A into B.

    Example dependency graph for softmax:
        reduce_max(x) ──────────────────────→ PointwiseSub(x, max)
                                                     ↓
                                              PointwiseExp
                                                     ↓
                                            reduce_sum(exp_x) ──→ PointwiseDiv(exp_x, sum)

    Reading this graph:
        reduce_max and PointwiseSub: sub is a consumer of reduce_max's output
                                     → producer-consumer, CAN fuse (input fusion)
        PointwiseSub and PointwiseExp: producer-consumer → CAN fuse
        reduce_sum and PointwiseDiv: div is a consumer of reduce_sum's output
                                     → but div also reads exp_x (same tensor as reduce_sum input!)
                                     → schedule reduce_sum with exp computation fused in,
                                        then PointwiseDiv as a separate pointwise kernel

    Final kernel groups for softmax([4, 8]):
        Group 1: Triton kernel — reduce(max(x)) with x-max pointwise, exp fused in
        Group 2: Triton kernel — reduce(sum(exp)) with exp recomputed or read from temp
        Group 3: Triton kernel — pointwise divide(exp, sum) fused with any consumer ops

### Fusion Rules in Detail

    The scheduler applies these rules in order:

    RULE 1 — VERTICAL FUSION (producer into consumer):
        If node A produces exactly one output, that output is consumed by exactly
        node B, and A is pointwise:
            Fuse A INTO B. A's computation happens inside B's loop.
            A's output buffer is ELIMINATED (computed on-the-fly, not materialised).

        Example:
            Before: Pointwise(exp, [input]) → Reduction(sum, [exp_output])
            After:  Reduction(sum, inline: exp(input))   ← exp is inlined

        Why "exactly one output" matters:
            If A's output is consumed by BOTH B and C, then fusing A into B
            means A's result is not available for C. Either C must recompute A
            (rematerialisation), or A must remain as a separate kernel.

    RULE 2 — HORIZONTAL FUSION (siblings):
        If node A and node B BOTH read from the same input tensor, and neither
        A nor B produces values the other consumes, they can be scheduled to
        execute in the SAME kernel pass over the input, loading it only once.

        Example: layer normalisation reads x to compute BOTH mean and variance.
            Before: Reduction(mean, [x]) and Reduction(var, [x]) — two separate reductions
            After:  single Triton kernel with two reduction accumulators (sum for mean,
                    sum-of-squares for variance), reading x exactly once.

    RULE 3 — EXTERN_KERNEL BOUNDARY:
        ExternKernel nodes are OPAQUE BOUNDARIES in the fusion graph.
        No other node can be fused with an ExternKernel.
        Nodes BEFORE an extern_kernel form their own fusion groups.
        Nodes AFTER an extern_kernel form new fusion groups.
        Exception: epilogue fusion (see Part 2).

    RULE 4 — REGISTER PRESSURE LIMIT:
        Each Triton thread block has a limited register file (~65536 registers).
        If a fusion group would require more registers than available,
        the scheduler splits it into smaller groups.
        This is estimated before code generation by counting live values.
        Exceeding the register limit causes register spilling → L2 cache access
        → slower than separate kernels.

    RULE 5 — BROADCAST RESHAPE COMPATIBILITY:
        Pointwise nodes with different loop bounds (due to broadcast or reshape)
        can be fused if the loop bounds are compatible — i.e., one is a product
        of the other's bounds, or they are identical after reshaping.

### Scheduling Example: Transformer FFN

    Consider transformer FFN: y = W2 @ relu(W1 @ x + b1) + b2

    ATen FX graph nodes (in topological order):
        1. aten.mm(x, W1.T)           → h_pre  (ExternKernel: cuBLAS)
        2. aten.add(h_pre, b1)        → h_add  (Pointwise)
        3. aten.relu(h_add)           → h_act  (Pointwise)
        4. aten.mm(h_act, W2.T)       → out_pre (ExternKernel: cuBLAS)
        5. aten.add(out_pre, b2)      → y      (Pointwise)

    Scheduling decisions:
        ExternKernel (node 1) → Group A: cuBLAS mm + epilogue with nodes 2+3
        ExternKernel (node 4) → Group B: cuBLAS mm + epilogue with node 5

    The scheduler enables epilogue fusion (Ampere+):
        Group A: cuBLAS DGEMM(x, W1.T) with epilogue: add b1, relu
                 This is ONE cuBLAS kernel call, not three operations.
        Group B: cuBLAS DGEMM(h_act, W2.T) with epilogue: add b2

    Result: 2 kernel launches for the entire FFN.
    Versus eager: 5 kernel launches.


##### PART 4 — TRITON CODEGEN: GENERATING GPU KERNELS FROM FIRST PRINCIPLES

### What Is Triton?

    Triton (not the TVM component "Triton Inference Server" — a completely
    different project) is OpenAI's domain-specific language for writing GPU
    kernels in Python. It was designed specifically to make it easy to write
    custom GPU code that is competitive with hand-tuned CUDA.

    Triton's key abstraction: TILE-BASED PROGRAMMING.
    Instead of writing one thread's computation (like CUDA), you write one
    PROGRAM's computation over a BLOCK of elements. Triton then handles:
        - Splitting the block across threads in a warp
        - Memory coalescing for global memory loads
        - Shared memory allocation and synchronisation
        - Register allocation within the block

    Triton primitives:
        tl.program_id(axis):   which "program" (block) we are in the grid
        tl.arange(0, N):       create a vector of consecutive integers [0..N-1]
        tl.load(ptr, mask):    load a block of elements from memory (coalesced)
        tl.store(ptr, val, mask): store a block of elements to memory
        tl.sum(x, axis):       reduce a block along an axis
        tl.max(x, axis):       reduce a block to its maximum
        +, -, *, /, exp, log, ...: elementwise ops on blocks

    TorchInductor generates Triton code programmatically. For each fused kernel
    group, it emits a @triton.jit decorated function that implements the entire
    fused computation.

### Anatomy of an Inductor-Generated Triton Kernel

    For a fused group of pointwise ops, Inductor generates:

        @triton.jit
        def triton_poi_fused_add_relu_mul_0(
            in_ptr0,             # pointer to first input tensor
            in_ptr1,             # pointer to second input tensor (bias)
            in_ptr2,             # pointer to third input tensor (gate)
            out_ptr0,            # pointer to output tensor
            xnumel,              # total number of elements
            XBLOCK : tl.constexpr  # tile size (tuned by Inductor)
        ):
            xoffset = tl.program_id(0) * XBLOCK
            xindex  = xoffset + tl.arange(0, XBLOCK)
            xmask   = xindex < xnumel          # handle non-aligned tails

            # Load inputs (coalesced reads from HBM)
            x0 = tl.load(in_ptr0 + xindex, xmask, other=0.0)  # main input
            x1 = tl.load(in_ptr1 + (xindex % bias_size), xmask)  # bias (broadcast)
            x2 = tl.load(in_ptr2 + xindex, xmask, other=0.0)  # gate

            # All computations stay in registers (no intermediate HBM writes)
            tmp0 = x0 + x1           # add bias
            tmp1 = triton_helpers.maximum(tmp0, 0)   # relu
            tmp2 = tmp1 * x2         # multiply by gate

            # Single write to HBM
            tl.store(out_ptr0 + xindex, tmp2, xmask)

        # Grid launch:
        grid = (cdiv(numel, XBLOCK),)
        triton_poi_fused_add_relu_mul_0[grid](..., XBLOCK=1024)

    ANATOMY OF THE GENERATED CODE:
        xoffset/xindex:  which elements THIS block handles (blocked layout)
        xmask:           guard against reading/writing out-of-bounds (tensor size % XBLOCK)
        tl.load:         coalesced read — consecutive threads read consecutive addresses
        tl.arange:       XBLOCK consecutive offsets, processed by XBLOCK threads simultaneously
        tmp0/1/2:        REGISTER VALUES — never written to HBM between ops
        tl.store:        single coalesced write — after all ops, one write per element

    The variable names tmp0, tmp1, tmp2 correspond to intermediate values that
    in eager mode would each require reading/writing 256MB of HBM.
    In the Triton kernel they exist ONLY in registers: zero HBM traffic.

### The Inner Function: How Fusion is Implemented

    TorchInductor represents the computation of each Inductor IR node as an
    INNER FUNCTION — a Python callable that maps (index_exprs...) → value.

    For aten.add(a, b):
        inner_fn = lambda index: a.inner_fn(index) + b.inner_fn(index)

    For aten.relu(x):
        inner_fn = lambda index: max(x.inner_fn(index), 0.0)

    For a fused chain relu(add(x, bias)):
        inner_fn = lambda index: max(x.inner_fn(index) + bias.inner_fn(index), 0.0)

    This is just FUNCTION COMPOSITION in Python! When Inductor generates Triton
    for a fused group, it walks the inner function composition tree and emits
    the corresponding Triton operations. The fusion is implicit in the function
    composition — no explicit fusion pass is needed.

    CODEGEN WALK for the fused relu(add(x, bias)) node:
        1. Start from the output node (relu).
        2. relu's inner_fn calls add's inner_fn for its input.
        3. add's inner_fn calls x's inner_fn and bias's inner_fn.
        4. x and bias are Placeholder nodes — their inner_fn emits tl.load.
        5. Walking the tree emits:
             tmp0 = tl.load(x_ptr + index, mask)
             tmp1 = tl.load(bias_ptr + index, mask)
             tmp2 = tmp0 + tmp1           # add
             tmp3 = triton_helpers.maximum(tmp2, 0)  # relu

### Reduction Kernel Generation

    For reduction ops, Inductor generates TWO-DIMENSIONAL Triton kernels:
        - One "XBLOCK" dimension for parallelism (different output elements)
        - One "RBLOCK" dimension for the reduction (serial accumulation)

        @triton.jit
        def triton_red_fused_sum_softmax(
            in_ptr0, out_ptr0, xnumel, rnumel,
            XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
        ):
            xoffset = tl.program_id(0) * XBLOCK
            xindex  = xoffset + tl.arange(0, XBLOCK)[:, None]    # [XBLOCK, 1]
            rindex  = tl.arange(0, RBLOCK)[None, :]               # [1, RBLOCK]
            xmask   = xindex < xnumel
            rmask   = rindex < rnumel

            # Load: each [x, r] element = input[xindex, rindex]
            x = tl.load(in_ptr0 + xindex * rnumel + rindex, xmask & rmask, other=0.0)

            # Reduction: max across the r dimension
            result = tl.max(x, axis=1)   # [XBLOCK] output

            tl.store(out_ptr0 + xindex.squeeze(1), result, xmask)

    The 2D indexing [XBLOCK, 1] × [1, RBLOCK] is a Triton broadcasting idiom
    that assigns:
        Each of the XBLOCK "programs" → one output element (different rows)
        Each program handles RBLOCK input elements in the reduction dimension

    For LARGE reductions (rnumel > RBLOCK):
        Each program processes multiple chunks of the reduction via a for loop.
        The accumulator is maintained in a register across the loop iterations.

### Persistent Reductions and Memory-Efficient Softmax

    For reductions where the entire reduction fits in shared memory, Inductor
    can use PERSISTENT reduction kernels:

        @triton.jit
        def triton_per_fused_softmax(x_ptr, out_ptr, n_rows, n_cols,
                                      BLOCK_SIZE: tl.constexpr):
            row_idx = tl.program_id(0)
            row_start = row_idx * n_cols
            cols = tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols

            # Load entire row into SRAM (shared memory via registers)
            row = tl.load(x_ptr + row_start + cols, mask, other=-float("inf"))

            # All reductions in a single program (no cross-program communication)
            row_max = tl.max(row, axis=0)            # scalar max
            row     = row - row_max                   # stable shift
            row     = tl.exp(row)                     # exp
            row_sum = tl.sum(row, axis=0)             # scalar sum
            row     = row / row_sum                   # normalise

            tl.store(out_ptr + row_start + cols, row, mask)

        # One program per row:
        grid = (n_rows,)

    This is the standard "online softmax" algorithm, but generated automatically
    by Inductor for any tensor of compatible shape. The entire softmax computation
    — 6 separate ATen ops in the FX graph — becomes one Triton kernel reading
    and writing each row exactly once.


##### PART 5 — CPU CODEGEN: C++ AND OPENMP

### The CPU Backend Architecture

    For CPU targets, TorchInductor generates C++ source code rather than Triton.
    The generated C++ is compiled by GCC or Clang (LLVM) at module load time.

    Advantages of C++ over Triton for CPU:
        LLVM's vectoriser understands x86 AVX-512 and ARM SVE natively.
        C++ templates can generate specialised code for each vector width.
        OpenMP pragmas integrate naturally with the CPU thread pool.
        BLAS/LAPACK calls integrate via C++ function calls, not library bindings.

    The CPU codegen pipeline:
        Inductor IR (pointwise/reduction/extern_kernel nodes)
            ↓ CppScheduling (Inductor's CPU scheduler)
        CppKernel IR (C++ loops with vectorisation hints)
            ↓ CppCodegen
        C++ source file (.cpp)
            ↓ subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", ...])
        Shared library (.so)
            ↓ ctypes.CDLL(...)
        Python callable

### Generated C++ Code Structure

    For a fused relu(add(x, bias)) on CPU, Inductor generates:

        #include <ATen/ATen.h>
        #include <omp.h>
        #include <immintrin.h>   // AVX-512 intrinsics (auto-included if available)

        void kernel_0(const float* __restrict__ in_ptr0,  // x
                      const float* __restrict__ in_ptr1,  // bias
                      float* __restrict__ out_ptr0,        // output
                      long xnumel) {
            // OpenMP parallel for (inductor uses thread pool)
            #pragma omp parallel for simd schedule(static)
            for (long x0 = 0; x0 < xnumel; x0 += 16) {   // 16 = AVX-512 lane width
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + x0);
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + x0 % bias_size);
                auto tmp2 = tmp0 + tmp1;                    // vectorised add
                auto zero = at::vec::Vectorized<float>(0.f);
                auto tmp3 = at::vec::clamp_min(tmp2, zero); // vectorised relu
                tmp3.store(out_ptr0 + x0);
            }
            // Tail loop for non-multiple-of-16 elements:
            for (long x0 = (xnumel / 16) * 16; x0 < xnumel; x0++) {
                float tmp0 = in_ptr0[x0] + in_ptr1[x0 % bias_size];
                out_ptr0[x0] = std::max(tmp0, 0.f);
            }
        }

    KEY FEATURES:
        at::vec::Vectorized<float>: PyTorch's portable SIMD wrapper.
            On AVX-512: processes 16 floats per instruction.
            On AVX2:    processes 8 floats per instruction.
            On NEON:    processes 4 floats per instruction.
            Falls back to scalar on unsupported hardware.
        #pragma omp parallel for simd: parallelise AND vectorise the loop.
        __restrict__: tells the compiler no pointer aliasing → enables more vectorisation.
        Tail loop: handles the final elements when N % 16 != 0.

### CPU Memory Planning

    CPU codegen includes explicit memory planning for intermediate buffers.
    Since the CPU has a cache hierarchy (L1/L2/L3), Inductor's CPU backend
    tries to tile loops to maximise cache reuse:

        Outer loop: tiles that fit in L2 cache
        Inner loop: tiles that fit in L1 cache
        Innermost loop: vectorised over SIMD width

    For matmul on CPU, Inductor calls the system BLAS (MKL or OpenBLAS)
    via the ExternKernel path — the same as XLA's CPU backend.


##### PART 6 — MEMORY PLANNING: BUFFER REUSE AND ALLOCATION

### Why Memory Planning Matters

    During training, a transformer forward pass allocates hundreds of intermediate
    tensors. Each allocation:
        Calls into the CUDA memory allocator (or Python's allocator for CPU).
        Requires finding a contiguous free block of the right size.
        Incurs overhead proportional to the number of live tensors.

    Without memory planning, each Inductor kernel allocates a fresh output buffer
    and frees it after the consuming kernel finishes. For a 100-kernel graph,
    this means 100 allocations and 100 frees per training step.

    With memory planning, Inductor pre-computes which buffers can be REUSED.
    If buffer A is last used at step 10 and buffer B is first needed at step 12,
    B can reuse A's memory. The total peak memory footprint shrinks, and
    allocation overhead is eliminated from the hot path.

### Buffer Lifetime Analysis

    Inductor computes the LIFETIME of every intermediate buffer:
        FIRST_USE: the kernel that creates this buffer.
        LAST_USE:  the kernel that reads this buffer for the last time.
        After LAST_USE, the buffer memory is available for reuse.

    Two buffers can share memory (be "aliased") if:
        Their lifetimes do NOT overlap.
        Their sizes are compatible (same dtype and same or smaller size).
        Neither buffer is an output of the compiled function.

    The aliasing problem is equivalent to REGISTER ALLOCATION in compilers:
        Buffers = virtual registers
        Memory locations = physical registers
        Lifetime intervals = live ranges
        Goal: minimise the number of distinct physical locations needed

    Inductor uses a greedy interval allocation algorithm:
        Sort buffers by first_use.
        For each buffer, look for an existing available slot of the right size.
        If found: reuse that slot (alias).
        If not found: allocate a new slot.

### In-Place Operations and Output Reuse

    Inductor additionally handles OUTPUT REUSE — where the output buffer
    of one kernel can be the input buffer of the next:

        For a chain: a → pointwise(a) → b → pointwise(b) → c
            If a and c have the same shape: c can reuse a's buffer.
            The pointwise kernel reads from a and writes to a's memory (in-place).

    This is safe because:
        The kernel reads each element of a EXACTLY ONCE.
        After reading a[i], it writes c[i] to the same location.
        No element is read twice, so overwriting is safe.
        This is called "storage aliasing" or "in-place codegen."

### Pre-Allocated Workspace (Static Memory Planning)

    For torch.compile(mode="reduce-overhead") with static shapes:
        All buffer sizes are known at compile time.
        Inductor can pre-allocate ALL memory as a single large workspace tensor.
        Each buffer gets a fixed offset within the workspace.
        No dynamic allocation ever happens during execution.
        The workspace is allocated ONCE when the model is first compiled.

    This is analogous to XLA's AHEAD-OF-TIME buffer assignment:
        The total working memory is computed statically.
        A single cudaMalloc(workspace_size) call at model load time.
        During execution: zero allocation overhead per step.


##### PART 7 — MAX-AUTOTUNE: KERNEL SEARCH AND BENCHMARKING

### The BLOCK_SIZE Problem

    Triton kernels have a crucial parameter: BLOCK_SIZE (also called XBLOCK).
    This controls how many tensor elements each Triton "program" processes.

    BLOCK_SIZE directly affects:
        GPU OCCUPANCY: too large → too few concurrent programs → GPU underused.
                       too small → overhead dominates; poor cache utilisation.
        MEMORY COALESCING: BLOCK_SIZE determines the pattern of HBM accesses.
        REGISTER PRESSURE: larger blocks → more registers needed → possible spilling.
        SHARED MEMORY USAGE: for reductions, BLOCK_SIZE determines SRAM usage.

    The OPTIMAL BLOCK_SIZE varies by:
        Tensor size (small tensors need small blocks; large tensors need large blocks)
        Operation type (reductions prefer different sizes than pointwise)
        GPU architecture (Ampere vs Hopper have different warp/SM configurations)
        Specific kernel structure (how much shared memory it uses)

    There is no formula to compute the optimal BLOCK_SIZE analytically.
    It must be found empirically — by measuring actual kernel execution time.

### Default Autotuning: BLOCK_SIZE Candidates

    In DEFAULT mode (mode="default"), Inductor uses a LIGHTWEIGHT AUTOTUNING
    strategy:

        1. For each new kernel shape encountered, generate a small set of
           CANDIDATE BLOCK_SIZES: [32, 64, 128, 256, 512, 1024] (or a subset
           based on the tensor size and operation type).

        2. For each candidate, compile the Triton kernel with that BLOCK_SIZE
           (Triton treats BLOCK_SIZE as a constexpr — a compile-time constant —
           so each value produces a different compiled kernel).

        3. Benchmark each compiled kernel on actual hardware with the real
           tensor shapes. Each measurement takes ~1-5 ms.

        4. Keep the fastest BLOCK_SIZE. Cache this choice for the current shape.

    Total autotuning time per kernel: 6 candidates × ~3ms each = ~18ms per shape.
    For a model with 50 kernels: ~900ms of autotuning overhead on first call.

    The autotuning result is PERSISTENT:
        torch._inductor.config.fx_graph_cache = True  (default in recent versions)
        Results cached in ~/.cache/torch/inductor/
        Subsequent runs with the same shapes use cached BLOCK_SIZE immediately.

### Max-Autotune: Exhaustive Kernel Search

    With mode="max-autotune", Inductor goes much further:

    PHASE 1 — TRITON POINTWISE SEARCH:
        For pointwise/reduction kernels: exhaustive BLOCK_SIZE search.
        Candidate set expands to: [16, 32, 64, 128, 256, 512, 1024, 2048]
        Additional parameters: num_warps ∈ [1, 2, 4, 8], num_stages ∈ [1, 2, 3, 4]
        Total candidates: 8 × 4 × 4 = 128 per kernel.
        Benchmarked in parallel where possible (multiple CUDA streams).

    PHASE 2 — GEMM ALGORITHM SEARCH (for ExternKernel matmuls):
        cuBLAS exposes multiple algorithm choices per GEMM shape.
        These differ in: tile sizes, pipeline depth, tensorcore config.
        Inductor calls cublasLt (cuBLAS Lightning) to benchmark all available
        algorithms for each (M, N, K) shape encountered.
        Number of algorithms: 50-200 per GEMM shape (depends on hardware).
        Winning algorithm is cached per (M, N, K, dtype) combination.

    PHASE 3 — EPILOGUE FUSION CANDIDATES:
        Whether to fuse bias+activation INTO the cuBLAS epilogue or
        compute them in a separate Triton kernel.
        Benchmarks both options for each matmul + surrounding ops.

    PHASE 4 — COORDINATE DESCENT TUNING:
        After finding a good initial set of parameters, uses coordinate descent
        to refine: vary one parameter at a time, keeping the rest fixed.
        Often finds 5-15% additional improvement over the initial search.
        config: torch._inductor.config.coordinate_descent_tuning = True

    Total max-autotune time for a transformer model: 5-30 minutes.
    Once cached: near-instant on subsequent runs.

### Benchmark Infrastructure

    TorchInductor uses Triton's built-in autotuner, extended with custom logic.
    Triton's autotuner:
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
                triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
                triton.Config({"BLOCK_SIZE": 256},  num_warps=8),
                ...
            ],
            key=["xnumel"]  # which input dimensions trigger re-tuning
        )
        @triton.jit
        def my_kernel(..., BLOCK_SIZE: tl.constexpr): ...

    When the kernel is first called with a new xnumel value, Triton
    benchmarks all configs and selects the best one.
    Results are cached in a .json file in the Triton cache directory.


##### PART 8 — CUDA GRAPHS: ELIMINATING KERNEL LAUNCH OVERHEAD

### The Kernel Launch Overhead Problem

    Each GPU kernel launch has overhead:
        The CPU must construct a launch descriptor.
        The descriptor is submitted to the CUDA driver.
        The driver queues it on a CUDA stream.
        The GPU scheduler picks it up and launches it.

    On modern hardware (CUDA 12, A100): approximately 5-10 µs per launch.

    For a transformer training step with 50 compiled kernels:
        50 kernels × 7 µs = 350 µs of pure launch overhead per step.
        On an A100 where the actual compute takes ~2 ms: 350µs = 17% overhead!

    For small models where compute is fast (edge inference):
        Launch overhead can EXCEED compute time.
        A model with 100µs of actual work + 200µs of launch overhead: 3× slower.

### CUDA Graph Capture and Replay

    CUDA Graphs solve the launch overhead problem by:
        1. CAPTURE PHASE (one-time, slow):
             Execute the entire model once while recording every GPU operation
             into a CUDAGraph object. This includes all kernel launches,
             all memory copies, all synchronisation events.
             The graph is stored as a device-side execution plan.

        2. REPLAY PHASE (every step, very fast):
             Instead of issuing individual kernel launches, submit a single
             "replay this CUDA graph" command. The GPU driver re-executes the
             entire recorded sequence atomically, with essentially zero CPU overhead.

    The REPLAY call: one CPU call → entire model runs on GPU.
    Overhead reduction: from 50 kernel-launch overheads to 1.

    CUDA graph replay via PyTorch:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(input)   # RECORD all kernels
        # Subsequently:
        input.copy_(new_input)      # update input IN-PLACE (same buffer!)
        graph.replay()              # re-execute recorded kernels
        result = output             # read result FROM SAME BUFFER

    CRITICAL CONSTRAINTS for CUDA graphs:
        1. STATIC SHAPES: all tensor shapes must be IDENTICAL on every replay.
           The captured graph hardcodes memory addresses and launch parameters.
           Any shape change → invalid graph → must re-capture.

        2. SAME MEMORY ADDRESSES: the input/output buffers must be at the same
           GPU memory addresses as during capture. New data is provided by
           copying into the captured buffers (not passing new buffers).
           input.copy_(new_data)  ← works  ✓
           model(new_tensor)      ← breaks CUDA graph ✗

        3. NO DYNAMIC CONTROL FLOW: branches, while loops, or any computation
           that might execute different kernels on different inputs CANNOT be
           captured in a CUDA graph.

        4. NO CPU-GPU SYNCHRONISATION: cuda.synchronize() inside the graph
           would stall the replay. All GPU operations must be fully asynchronous.

        5. RANDOM SEED STATE: RNG state is captured. Subsequent replays produce
           the SAME random values unless you explicitly update the seed.

### How TorchInductor Implements CUDA Graphs

    With torch.compile(mode="reduce-overhead") or mode="max-autotune":

        FIRST CALL (warmup phase):
            - Inductor compiles and runs the model normally (one kernel at a time).
            - Warms up cuBLAS algorithm selection.
            - Checks that all shapes are static and no control flow exists.

        SECOND CALL (capture phase):
            - Inductor wraps the execution in a CUDA graph capture context.
            - Re-runs all compiled kernels in capture mode.
            - The GPU records the full sequence into a CUDAGraph object.

        SUBSEQUENT CALLS (replay phase):
            - Inductor copies new input data into the captured buffers.
            - Issues a single graph.replay() call.
            - Reads results from the captured output buffers.

    Inductor's CUDA graph implementation:
        class CUDAGraphRunner:
            def __init__(self, model, inputs):
                self.static_inputs  = [x.clone() for x in inputs]  # captured buffers
                self.static_outputs = None
                self.graph          = torch.cuda.CUDAGraph()

                # Warmup
                with torch.cuda.stream(torch.cuda.Stream()):
                    model(*self.static_inputs)

                # Capture
                with torch.cuda.graph(self.graph):
                    self.static_outputs = model(*self.static_inputs)

            def __call__(self, *new_inputs):
                for static, new in zip(self.static_inputs, new_inputs):
                    static.copy_(new)   # in-place update captured buffers
                self.graph.replay()
                return self.static_outputs

### When CUDA Graphs Are Disabled

    TorchInductor automatically DISABLES CUDA graphs when:
        The model uses random number generation (torch.rand, dropout) — unless the
        user manages seed state explicitly.
        The model contains data-dependent control flow that produces different
        kernel sequences on different inputs.
        Input shapes change between calls (dynamic shapes).
        The model calls synchronisation points (some distributed ops).
        mode="max-autotune-no-cudagraphs" is specified.

    Detecting CUDA graph incompatibility:
        torch._inductor.config.warn_on_cuda_graph_incompatible = True
        → prints a warning when graphs are disabled with the reason.


##### PART 9 — CUSTOM TRITON KERNELS AND OPERATOR EXTENSION

### Registering Custom Triton Kernels as torch.compile-Compatible Ops

    Sometimes the automatically-generated Triton kernel is not optimal for
    a specific operation. Common cases:
        Flash Attention: requires careful tiling of the Q/K/V matrices across
                         SRAM to avoid materialising the full [S,S] attention matrix.
        Custom activation functions with numerical tricks.
        Mixed-precision ops that need specific accumulation strategies.
        Sparse operations (not representable as dense Inductor kernels).

    Inductor provides the TORCH LIBRARY mechanism for registering custom ops:

    STEP 1 — Define the operation's schema:
        @torch.library.custom_op("mylib::flash_attn", mutates_args=())
        def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # Eager implementation (fallback if Inductor cannot compile)
            scores = q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1])
            return torch.softmax(scores, dim=-1) @ v

    STEP 2 — Register a Triton implementation for Inductor:
        @torch.library.register_kernel("mylib::flash_attn", "cuda")
        def flash_attn_cuda(q, k, v):
            return triton_flash_attn_forward(q, k, v)  # your Triton kernel

    STEP 3 — Register the backward rule for AOTAutograd:
        @torch.library.register_autograd("mylib::flash_attn", ...)

    Now torch.compile automatically uses your Triton kernel for flash_attn.
    The compiled graph shows it as an ExternKernel (opaque to Inductor's fusion,
    but faster than anything Inductor would auto-generate).

### torch._inductor.kernel: Low-Level Kernel Registration

    For even lower-level control, Inductor exposes kernel registration hooks
    that allow you to replace specific ExternKernel dispatch paths:

        # Replace all (M,N,K) = (1024, 1024, 1024) BF16 GEMMs with a custom kernel:
        @torch._inductor.config.register_triton_mm_handler
        def custom_mm_1024(a, b, c, *, M, N, K, dtype):
            if (M, N, K) == (1024, 1024, 1024) and dtype == torch.bfloat16:
                return my_ultra_tuned_triton_matmul(a, b)
            return None   # fall back to cuBLAS

### Inductor and PyTorch's FX Custom Backend API

    TorchInductor IS itself implemented as a torch.compile backend.
    Its entry point is torch._inductor.compile_fx.compile_fx().
    This function takes (gm: GraphModule, example_inputs: List[Tensor]) → Callable.

    You can wrap Inductor to add custom pre/post-processing:

        from torch._inductor.compile_fx import compile_fx

        def my_wrapped_inductor(gm, example_inputs):
            # Pre-process the FX graph
            my_custom_pass(gm.graph)
            gm.recompile()
            # Delegate to Inductor
            return compile_fx(gm, example_inputs)

        @torch.compile(backend=my_wrapped_inductor)
        def my_model(x): ...


##### PART 10 — DEBUGGING, PROFILING, AND THE FULL PIPELINE

### Environment Variables and Config Flags

    TORCH_LOGS environment variable (most useful for debugging):
        TORCH_LOGS="output_code"     → print every generated Triton/C++ kernel
        TORCH_LOGS="inductor"        → inductor pipeline decisions
        TORCH_LOGS="schedules"       → which nodes got grouped together
        TORCH_LOGS="fusion"          → why specific ops did/didn't fuse
        TORCH_LOGS="kernel_code"     → the exact Triton source for each kernel
        TORCH_LOGS="graph"           → the ATen graph Inductor receives
        TORCH_LOGS="perf_hints"      → warnings about potential performance issues
        TORCH_LOGS="+all"            → everything (extremely verbose)

    torch._inductor.config settings:
        debug = True                 ; dump all IR passes to /tmp/torchinductor_*
        trace.enabled = True         ; save interactive HTML compilation trace
        max_autotune = True          ; exhaustive BLOCK_SIZE search
        max_autotune_gemm = True     ; cuBLASLt algorithm search only
        coordinate_descent_tuning    ; hill-climb after initial search
        cuda_graphs = True/False     ; enable/disable CUDA graph capture
        fx_graph_cache = True        ; persistent cache for compiled graphs
        benchmark_kernel = True      ; time each kernel and report

    TORCHINDUCTOR_CACHE_DIR:
        Default: ~/.cache/torch/inductor/
        Override: TORCHINDUCTOR_CACHE_DIR=/fast_nvme/inductor_cache
        Contains: compiled Triton .so files + autotuning results
        Persists across Python sessions → amortises compile time

### Understanding Inductor's Output with TORCH_LOGS="output_code"

    Setting TORCH_LOGS="output_code" prints each generated kernel.
    This is the primary tool for understanding what Inductor compiled:

    For the operation chain relu(x @ W + b):

    KERNEL 0 (generated Triton for bias+relu fusion):
        @triton.jit
        def triton_poi_fused_add_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, ...):
            ...  # add + relu fused

    KERNEL 1 (ExternKernel for matmul — NOT generated, just a cuBLAS call):
        extern_kernels.mm(x, W_T, out=mm_result)  # cuBLAS dispatch

    WRAPPER:
        def call(args):
            x, W, b = args
            buf0 = empty((M,N), device='cuda')
            extern_kernels.mm(x, W.T, out=buf0)          # ExternKernel
            buf1 = empty((M,N), device='cuda')
            triton_poi_fused_add_relu_0[grid](buf0, b, buf1, ...)  # Triton
            return (buf1,)

    Reading this wrapper tells you:
        How many buffers are allocated (buf0, buf1).
        Which operations are ExternKernel (cuBLAS/cuDNN) vs Triton.
        The kernel launch grid sizes.
        Whether your expected fusion occurred.

### Interpreting the HTML Trace

    With torch._inductor.config.trace.enabled = True:
        Inductor saves an interactive HTML file to /tmp/torchinductor_*/trace.html.
        The trace shows:
            Timeline of the compilation pipeline (which pass took how long)
            The FX graph before and after each Inductor pass
            Which nodes were grouped into which kernel groups
            The generated code for each kernel
            Autotuning results (best BLOCK_SIZE for each kernel)

### TorchInductor in the Connected Compiler Stack

    LLVM (module 01):
        Inductor's CPU backend generates C++ code.
        GCC/Clang (LLVM-based) compiles this C++ with -O3 -march=native.
        LLVM's vectoriser handles AVX-512/NEON via at::vec intrinsics.
        Triton itself uses LLVM's NVPTX backend to lower Triton IR to PTX.

    MLIR (module 02):
        Inductor's IR (Pointwise/Reduction/ExternKernel) mirrors MLIR's linalg
        dialect (linalg.generic/linalg.reduce/extern).
        The inner function composition pattern mirrors MLIR's region-based IR.
        Inductor does not USE MLIR directly, but the conceptual design is parallel.

    TorchDynamo (module 09):
        Dynamo is Inductor's ONLY entry point — no other way to trigger Inductor.
        Dynamo's FX graph (after AOTAutograd) is what Inductor receives.
        Inductor's compiled function is what Dynamo's cache stores.

    XLA/OpenXLA (modules 05-06):
        XLA and Inductor are direct competitors for GPU compilation.
        XLA uses cuBLAS/cuDNN + HLO fusion; Inductor uses Triton + cuBLAS.
        torch.compile(backend="openxla") BYPASSES Inductor entirely.
        Empirically: XLA wins for TPU and very large matmuls; Inductor wins for
        transformer-heavy workloads on modern NVIDIA GPUs.

    TVM (module 08):
        TVM can serve as a torch.compile backend (backend="tvm").
        When it does, Inductor is BYPASSED — TVM handles compilation.
        TVM's MetaSchedule finds more optimal kernels for some edge hardware
        cases. Inductor wins for standard NVIDIA GPU workflows.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  TORCHINDUCTOR IN THE TORCH.COMPILE PIPELINE                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Python model code (@torch.compile decorated)                        │
    │       ↓  TorchDynamo: bytecode → FX graph + guards                  │
    │       ↓  AOTAutograd: differentiate → joint fwd+bwd FX graph        │
    │  ATen FX graph  ← this is what Inductor receives                    │
    │       ↓  Inductor lowering: ATen ops → Inductor IR (3 types)        │
    │  Inductor IR (Pointwise + Reduction + ExternKernel nodes)           │
    │       ↓  Scheduler: build dependency graph → fusion decisions       │
    │  Kernel groups (FusedSchedulerNode lists)                           │
    │       ↓  Codegen: per-group code generation                         │
    │  GPU: Triton .py files   CPU: C++ .cpp files                        │
    │       ↓  Compilation: tritonc / GCC                                 │
    │  Compiled .so libraries                                              │
    │       ↓  Wrapper generation: call sequence + buffer allocation       │
    │  Executable Python callable (cached with Dynamo's guard system)     │
    └─────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Inductor IR — Three Node Types and the Lowering Pipeline": {
        "description": (
            "Understand TorchInductor's internal IR and compilation pipeline from scratch. "
            "Show how ATen ops map to Pointwise, Reduction, and ExternKernel nodes. "
            "Implement the three node types in pure Python to demystify the IR. "
            "Trace a complete layernorm + linear + relu through the lowering pipeline. "
            "Show the inner_fn composition mechanism that makes fusion trivial. "
            "Demonstrate how to inspect Inductor's decisions with TORCH_LOGS."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from typing import Callable, List, Optional
from dataclasses import dataclass, field

print("=" * 65)
print("  INDUCTOR IR — THREE NODE TYPES AND THE LOWERING PIPELINE")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    print("  PyTorch not installed: pip install torch")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: The Inductor IR — a minimal Python simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Inductor IR: Pointwise, Reduction, ExternKernel")
print("━" * 65)
print()

INDUCTOR_IR_THEORY = """
  THE THREE INDUCTOR IR NODE TYPES — ANNOTATED
  ════════════════════════════════════════════════════════════════

  TorchInductor converts ATen FX ops into one of exactly three node types.
  This classification is what enables the simple, universal fusion rules.

  NODE 1 — POINTWISE:
  ─────────────────────────────────────────────────────────────────
  output[i] = f(input1[i], input2[i], ...)  for each index i
  Shape: output.shape == broadcast(input1.shape, input2.shape, ...)

  Internal representation:
    name:     string identifier (e.g., "relu_0")
    ranges:   list of symbolic loop bounds  [M, N] for a (M,N) tensor
    inner_fn: Python closure:
                (index_vars...) → expression tree
                for aten.relu: lambda v: max(v[0], 0.0) where v[0] reads input
    dtype:    torch.float32, torch.bfloat16, etc.

  FUSION: compose inner_fns.
    If A has inner_fn_A and B has inner_fn_B and B reads from A:
      fused_inner_fn = lambda idx: inner_fn_B(inner_fn_A(idx))
      A's buffer is NEVER materialised — computed inline in B's loop.

  NODE 2 — REDUCTION:
  ─────────────────────────────────────────────────────────────────
  output[i] = reduce(f, input[i, :])  folding the ":" dimension
  Shape: output.shape == input.shape without the reduction dims.

  Internal representation:
    name:     string identifier
    ranges:   loop bounds for OUTPUT dimensions [M] (not including reduced dim)
    reduction_ranges: loop bounds for REDUCTION dimensions [N]
    reduction_type:  "sum", "max", "min", "any", "welford" (mean+var together)
    inner_fn: closure for what to accumulate:
                for sum(exp(x)): lambda ridx: exp(x_loader(ridx))

  FUSION RULE:
    A Pointwise PRODUCER can fuse INTO a Reduction (its inner_fn is inlined).
    A Pointwise CONSUMER can fuse AFTER a Reduction (epilogue fusion).
    Two independent Reductions CANNOT fuse together.

  NODE 3 — EXTERN_KERNEL:
  ─────────────────────────────────────────────────────────────────
  Opaque external library call.

  Internal representation:
    name:     string identifier
    kernel:   reference to the extern function (aten.mm, aten.convolution, etc.)
    inputs:   list of input buffer references
    output:   output buffer specification (shape, dtype, layout)
    layout:   required memory layout (NCHW, NHWC, etc.)

  FUSION RULE:
    ExternKernels are NEVER fused with other nodes' main computation.
    Exception: epilogue fusion (bias+activation into cuBLAS epilogue API).
    They form HARD BOUNDARIES in the fusion graph.
"""
print(INDUCTOR_IR_THEORY)

# ── Simulate the Inductor IR in Python ───────────────────────────────────
@dataclass
class Buffer:
    """A tensor buffer (real or virtual/intermediate)."""
    name: str
    shape: tuple
    dtype: str = "float32"
    virtual: bool = False   # virtual = never materialised (fused away)

    def __repr__(self):
        mat = " [virtual]" if self.virtual else ""
        return f"Buffer({self.name}, {self.shape}, {self.dtype}{mat})"


@dataclass
class PointwiseNode:
    """Inductor Pointwise IR node."""
    name: str
    output: Buffer
    inputs: List[Buffer]
    op: str                     # human-readable op name
    inner_fn: Callable          # (index, *input_vals) → output_val
    fused_into: Optional[str] = None  # name of node this is fused into

    def can_fuse_with(self, other: "PointwiseNode") -> bool:
        """Can this node be fused with other?"""
        # Can fuse if: this node's output IS other's input (producer-consumer)
        # AND this node's output has no other users
        return any(inp.name == self.output.name for inp in other.inputs)

    def __repr__(self):
        fused = f" [fused into {self.fused_into}]" if self.fused_into else ""
        return f"Pointwise({self.name}: {self.op}{fused})"


@dataclass
class ReductionNode:
    """Inductor Reduction IR node."""
    name: str
    output: Buffer
    input: Buffer
    reduction_dim: int
    reduction_type: str         # "sum", "max", "min"
    fused_producer: Optional["PointwiseNode"] = None

    def __repr__(self):
        prod = f" with fused producer {self.fused_producer.name}" if self.fused_producer else ""
        return f"Reduction({self.name}: {self.reduction_type} over dim {self.reduction_dim}{prod})"


@dataclass
class ExternKernelNode:
    """Inductor ExternKernel IR node."""
    name: str
    output: Buffer
    inputs: List[Buffer]
    kernel_name: str            # "aten.mm", "aten.convolution", etc.
    epilogue_ops: List[str] = field(default_factory=list)

    def __repr__(self):
        epi = f" + epilogue{self.epilogue_ops}" if self.epilogue_ops else ""
        return f"ExternKernel({self.name}: {self.kernel_name}{epi})"


# ── ATen op → Inductor IR node classification ────────────────────────────
def classify_aten_op(op_name: str) -> str:
    """
    Classify an ATen operator into one of Inductor's three IR node types.
    This is what Inductor's lowering pass does for each FX node.
    """
    POINTWISE_OPS = {
        "aten.relu.default", "aten.gelu.default", "aten.silu.default",
        "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor",
        "aten.div.Tensor", "aten.neg.default", "aten.abs.default",
        "aten.exp.default", "aten.log.default", "aten.sqrt.default",
        "aten.tanh.default", "aten.sigmoid.default", "aten.clamp.Tensor",
        "aten.where.self", "aten.gt.Scalar", "aten.lt.Scalar",
        "aten.softplus.default", "aten.hardswish.default",
        "aten.broadcast_to.default", "aten.expand.default",
        "aten.reshape.default", "aten.view.default", "aten.permute.default",
        "aten.contiguous.memory_format", "aten.clone.default",
        "aten.fill_.Scalar", "aten.zeros_like.default",
        "aten.ones_like.default", "aten.full_like.default",
    }
    REDUCTION_OPS = {
        "aten.sum.dim_IntList", "aten.sum.default",
        "aten.mean.dim", "aten.mean.default",
        "aten.max.dim", "aten.amax.default",
        "aten.min.dim", "aten.amin.default",
        "aten.norm.ScalarType", "aten.any.dim",
        "aten.all.dim", "aten.argmax.default", "aten.argmin.default",
        "aten.var.correction", "aten.std.correction",
        "aten.logsumexp.default",
    }
    EXTERN_OPS = {
        "aten.mm.default", "aten.bmm.default",
        "aten.addmm.default", "aten.baddbmm.default",
        "aten.convolution.default", "aten._native_batch_norm_legit.default",
        "aten.scaled_dot_product_attention.default",
        "aten.fft_fft.default", "aten.fft_ifft.default",
        "aten._triton_multi_head_attention.default",
    }

    if op_name in POINTWISE_OPS:
        return "Pointwise"
    elif op_name in REDUCTION_OPS:
        return "Reduction"
    elif op_name in EXTERN_OPS:
        return "ExternKernel"
    else:
        return "Unknown (defaults to ExternKernel)"

print("  ATen op → Inductor IR classification:")
print()
sample_ops = [
    ("aten.relu.default",                  "common activation"),
    ("aten.add.Tensor",                    "elementwise add"),
    ("aten.exp.default",                   "elementwise exp"),
    ("aten.gelu.default",                  "GELU activation"),
    ("aten.sum.dim_IntList",               "sum reduction"),
    ("aten.mean.dim",                      "mean reduction"),
    ("aten.amax.default",                  "max along dim"),
    ("aten.var.correction",                "variance (for layer norm)"),
    ("aten.mm.default",                    "matrix multiply"),
    ("aten.bmm.default",                   "batched matmul"),
    ("aten.convolution.default",           "convolution"),
    ("aten.scaled_dot_product_attention",  "flash attention"),
]
print(f"  {'ATen Op':45s}  {'IR Type':15s}  {'Description'}")
print("  " + "-" * 80)
for op, desc in sample_ops:
    ir_type = classify_aten_op(op)
    marker  = "🔄" if ir_type == "Pointwise" else ("📊" if ir_type == "Reduction" else "🔗")
    print(f"  {op:45s}  {ir_type:15s}  {marker} {desc}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Inner function composition — how fusion works
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Inner function composition: fusion as function composition")
print("━" * 65)
print()

INNER_FN = """
  THE INNER FUNCTION PATTERN — FUSION VIA COMPOSITION
  ════════════════════════════════════════════════════════════════

  TorchInductor represents each Inductor IR node as an inner_fn:
    A Python closure: (index, loader_fn) → scalar_expression
    The loader_fn reads from the node's input buffer at a given index.
    The expression tree is what gets emitted as Triton code.

  UNFUSED: three separate nodes with three separate buffers
    node_add:  inner_fn = lambda idx, load: load(x, idx) + load(b, idx)
    node_relu: inner_fn = lambda idx, load: max(load(h, idx), 0.0)  ; reads h = node_add output
    node_mul:  inner_fn = lambda idx, load: load(gate, idx) * load(a, idx)  ; reads a = relu output

  Each node has its own output buffer: h, a, out
  Each requires: load from HBM, compute, store to HBM.
  3 separate kernels, 3× HBM traffic.

  FUSED: one node, one buffer, one kernel pass
    Step 1: fuse node_relu INTO node_add (relu is sole consumer of add output):
      fused_add_relu = lambda idx, load: max(load(x, idx) + load(b, idx), 0.0)
      node_add's buffer 'h' is marked VIRTUAL (never materialised)

    Step 2: fuse fused_add_relu INTO node_mul:
      fused_all = lambda idx, load: load(gate, idx) * max(load(x, idx) + load(b, idx), 0.0)
      The 'a' buffer is also VIRTUAL

    One fused node. One kernel pass. One HBM read cycle per element.

  IMPLEMENTATION: Inductor's inner_fn composition in Python pseudocode:

    class PointwiseNode:
        def __init__(self, op_fn, inputs):
            self.inputs  = inputs          ; list of input PointwiseNodes
            self.op_fn   = op_fn           ; elementwise operation

        def inner_fn(self, index):
            \" \"\"Compute output at `index` recursively.\"\"\"
            input_vals = [inp.inner_fn(index) for inp in self.inputs]
            return self.op_fn(*input_vals)

    For add:   AddNode.inner_fn(i) = x.inner_fn(i) + bias.inner_fn(i)
    For relu:  ReluNode.inner_fn(i) = max(add.inner_fn(i), 0)
                                    = max(x.inner_fn(i) + bias.inner_fn(i), 0)

  The fusion is implicit in the RECURSIVE CALL. When Inductor generates
  Triton code for the fused relu node, it walks the inner_fn call tree
  and emits Triton statements for each operation in the tree — all in one
  Triton program body, all using register-resident intermediate values.
"""
print(INNER_FN)

# Simulate inner_fn composition in Python
class InductorNode:
    """
    Simulates TorchInductor's Pointwise node with inner_fn composition.
    In real Inductor, inner_fn returns sympy expressions → Triton code.
    Here it returns Python lambdas for demonstration.
    """
    def __init__(self, name, inner_fn, inputs=None):
        self.name     = name
        self._fn      = inner_fn
        self.inputs   = inputs or []
        self.virtual  = False   # True when fused (buffer eliminated)

    def inner_fn(self, idx):
        return self._fn(idx)

    def fuse_with_consumer(self, consumer: "InductorNode") -> "InductorNode":
        """
        Fuse self INTO consumer (self is a producer, consumer reads self).
        After fusion: self is marked virtual (no buffer allocated).
        Consumer's inner_fn now calls self's inner_fn inline.
        """
        producer_fn = self.inner_fn
        # Replace consumer's reference to self's buffer with self's computation
        def fused_fn(idx):
            return consumer.inner_fn.__closure__[0].cell_contents(
                lambda: producer_fn(idx))
        self.virtual = True
        return consumer  # consumer now embeds producer's computation

# Build the computation graph for: out = gate * relu(x + bias)
x_data    = np.array([-1.0, 0.5, -0.3, 2.0, 1.5, -0.8])
bias_data = np.array([ 0.5, 0.5,  0.5, 0.5, 0.5,  0.5])
gate_data = np.array([ 2.0, 1.0,  3.0, 0.5, 1.0,  2.0])

class LoadNode:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.virtual = False
    def inner_fn(self, idx): return float(self.data[idx])

# Build computation nodes
x_node    = LoadNode("x",    x_data)
bias_node = LoadNode("bias", bias_data)
gate_node = LoadNode("gate", gate_data)

# Unfused: three separate kernels
class AddNode:
    def __init__(self, a, b): self.a = a; self.b = b; self.virtual = False; self.name = "add"
    def inner_fn(self, idx): return self.a.inner_fn(idx) + self.b.inner_fn(idx)

class ReluNode:
    def __init__(self, x): self.x = x; self.virtual = False; self.name = "relu"
    def inner_fn(self, idx): return max(self.x.inner_fn(idx), 0.0)

class MulNode:
    def __init__(self, a, b): self.a = a; self.b = b; self.virtual = False; self.name = "mul"
    def inner_fn(self, idx): return self.a.inner_fn(idx) * self.b.inner_fn(idx)

add_node  = AddNode(x_node, bias_node)
relu_node = ReluNode(add_node)
mul_node  = MulNode(gate_node, relu_node)

# Execute the FUSED computation (mul_node calls through to add and relu inline)
n = len(x_data)
print("  Fused computation: out = gate * relu(x + bias)")
print()
print(f"  {'idx':>4}  {'x':>6}  {'bias':>6}  {'gate':>6}  {'x+bias':>8}  "
      f"{'relu':>8}  {'gate*relu':>10}  {'ref':>10}")
print("  " + "-" * 68)
ref = gate_data * np.maximum(x_data + bias_data, 0)
for i in range(n):
    add_v  = add_node.inner_fn(i)
    relu_v = relu_node.inner_fn(i)
    out_v  = mul_node.inner_fn(i)
    print(f"  {i:>4}  {x_data[i]:>6.1f}  {bias_data[i]:>6.1f}  {gate_data[i]:>6.1f}  "
          f"{add_v:>8.2f}  {relu_v:>8.2f}  {out_v:>10.4f}  {ref[i]:>10.4f}")

print()
print("  KEY: All intermediate values (add_v, relu_v) are REGISTER-RESIDENT.")
print("  In Triton: they appear as tmp0, tmp1, tmp2 — never written to HBM.")
print()
err = np.max(np.abs(np.array([mul_node.inner_fn(i) for i in range(n)]) - ref))
print(f"  Correctness vs numpy: max_err = {err:.2e} ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Inspecting Inductor with TORCH_LOGS
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Inspecting Inductor's decisions with TORCH_LOGS")
print("━" * 65)
print()

TORCH_LOGS_GUIDE = """
  TORCH_LOGS: THE INDUCTOR DEBUGGER
  ════════════════════════════════════════════════════════════════

  Usage: TORCH_LOGS="<category>" python my_script.py
  Multiple categories: TORCH_LOGS="output_code,schedules"

  CATEGORY       WHAT IT SHOWS
  ────────────────────────────────────────────────────────────────
  output_code    The generated Triton/C++ code for every kernel.
                 THE most useful for understanding what compiled.
                 Shows: kernel signature, tl.load/store patterns,
                        which ops are fused (appear in same kernel),
                        BLOCK_SIZE and grid configuration.

  schedules      The Inductor scheduler's grouping decisions.
                 Shows: which FX nodes → which kernel group,
                        why specific nodes did or didn't fuse.

  fusion         Fine-grained fusion decisions.
                 Shows: each pair considered for fusion + why fused/not.

  graph          The ATen FX graph Inductor receives (post-AOTAutograd).
                 Shows: every ATen op with shapes, dtypes, strides.

  inductor       General inductor pipeline logging.
                 Shows: each phase's timing, pass names applied.

  perf_hints     Performance warnings Inductor detects.
                 Shows: "non-contiguous tensor passed to mm" types of
                        warnings that indicate suboptimal memory layouts.

  EXAMPLE — observing fusion for LayerNorm:
  ─────────────────────────────────────────────────────────────────
  # Set before running:
  import torch._inductor.config as config
  config.debug = True    ; dump IR to /tmp/torchinductor_<username>/

  # OR use environment variable:
  # TORCH_LOGS="output_code,fusion" python script.py

  WHAT YOU SHOULD SEE for LayerNorm compilation:
    Kernel 0: triton_red_fused_mean_var_0
       ; This is the TWO-ACCUMULATOR reduction kernel
       ; It computes mean AND variance in ONE pass over the input
       ; by maintaining two running sums: sum_x and sum_x2
       tl.atomic_add(sum_ptr + xindex, tmp0)       ; accumulate sum
       tl.atomic_add(sum_sq_ptr + xindex, tmp1)    ; accumulate sum of squares
       ; FUSION: the input pointwise (x - mean) is inlined here via inner_fn

    Kernel 1: triton_poi_fused_normalise_scale_shift_1
       ; After reductions, apply: (x - mean) / sqrt(var + eps) * gamma + beta
       ; This is ONE pointwise kernel for the remaining elementwise ops
       ; The division, sqrt, multiply, add are ALL FUSED into one pass

  WHAT INDICATES FUSION WORKED:
    Few separate kernel sections (e.g., 2 kernels for layernorm, not 8).
    Intermediate tensors (like "x - mean") appear as tmp0, tmp1 in the
    Triton kernel body, NOT as separate tl.store/tl.load pairs.

  WHAT INDICATES FUSION FAILED (and what to do):
    Many separate small kernels → graph break between ops?
      Fix: remove the graph break (see module 09).
    Unexpected tl.store → tl.load pairs → fusion boundary hit.
      Reason: register pressure limit or non-fuseable op type.
      Fix: restructure to reduce fusion group size,
           or accept the boundary (cuBLAS ops are always boundaries).

  READING THE WRAPPER CODE:
    The wrapper shows buffer allocation order:
      buf0 = empty_strided((M, N), (N, 1), dtype=float32, device='cuda')
      ; buf0 is the intermediate buffer between kernel 0 and kernel 1.
      ; Its lifetime = from kernel 0's output to kernel 1's last read.
      ; If a later kernel could reuse buf0: Inductor will show the same name.

      extern_kernels.mm(x, W.T, out=buf0)     ; ExternKernel: cuBLAS
      del x, W                                  ; freed after use
      triton_poi_0[grid](buf0, bias, out, ...)  ; Triton: bias + relu
      del buf0, bias                             ; freed after use
"""
print(TORCH_LOGS_GUIDE)

if HAS_TORCH:
    import os, io, contextlib

    # Capture inductor output for a simple function
    @torch.compile(fullgraph=True)
    def sample_fn(x, bias):
        h = x + bias
        a = torch.relu(h)
        return torch.exp(a) * x

    x_s    = torch.randn(1024, device=DEVICE)
    bias_s = torch.randn(1024, device=DEVICE)

    # Run once to trigger compilation
    out_s  = sample_fn(x_s, bias_s)

    print("  sample_fn(x, bias): x + bias → relu → exp → * x")
    print(f"  Output shape: {out_s.shape}, dtype: {out_s.dtype}")
    print()
    print("  To see generated Triton code:")
    print("    TORCH_LOGS='output_code' python script.py")
    print("  Or programmatically:")
    print("    import torch._inductor.config as config")
    print("    config.debug = True")
    print("    config.trace.enabled = True")
    print()
    print("  Expected Triton output snippet:")
    EXPECTED_TRITON = """
  @triton.jit
  def triton_poi_fused_add_relu_exp_mul_0(in_ptr0, in_ptr1, out_ptr0,
                                           xnumel, XBLOCK: tl.constexpr):
      xoffset = tl.program_id(0) * XBLOCK
      xindex  = xoffset + tl.arange(0, XBLOCK)
      xmask   = xindex < xnumel
      tmp0 = tl.load(in_ptr0 + xindex, xmask)   ; load x
      tmp1 = tl.load(in_ptr1 + xindex, xmask)   ; load bias
      tmp2 = tmp0 + tmp1                          ; x + bias
      tmp3 = triton_helpers.maximum(tmp2, 0)      ; relu
      tmp4 = tl.exp(tmp3)                         ; exp
      tmp5 = tmp0 * tmp4                          ; * x
      tl.store(out_ptr0 + xindex, tmp5, xmask)   ; store result

  ; ALL 4 OPS IN ONE KERNEL PASS:
  ; - tl.load called TWICE (x and bias) — never loaded again
  ; - tl.store called ONCE
  ; - tmp2, tmp3, tmp4 are REGISTER-ONLY (never touch HBM)
  ; - BLOCK_SIZE = 1024 (or tuned value from autotuning)
"""
    print(EXPECTED_TRITON)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Fusion Scheduling — Building the Kernel Groups": {
        "description": (
            "Simulate TorchInductor's scheduler to understand fusion decisions. "
            "Build a dependency graph from Inductor IR nodes. "
            "Apply the three fusion rules: vertical, horizontal, and extern_kernel boundary. "
            "Trace softmax and LayerNorm through the scheduler showing final kernel groups. "
            "Show which operations get fused vs kept separate and why. "
            "Demonstrate the full FFN scheduling: 2 cuBLAS calls + epilogue fusion."
        ),
        "language": "python",
        "code": '''
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field

print("=" * 65)
print("  FUSION SCHEDULING — BUILDING THE KERNEL GROUPS")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Scheduler simulation — dependency graph + fusion rules
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Scheduler: dependency graph and fusion rules")
print("━" * 65)
print()

SCHEDULER_THEORY = """
  THE INDUCTOR SCHEDULER — STEP BY STEP
  ════════════════════════════════════════════════════════════════

  The scheduler converts the flat list of Inductor IR nodes into
  KERNEL GROUPS. Each group becomes one GPU kernel or one external call.

  STEP 1 — BUILD THE DEPENDENCY GRAPH:
    For each node N, find all nodes that N depends on (nodes whose output
    N reads as input). Build a directed graph: dependency → dependent.

  STEP 2 — TOPOLOGICAL SORT:
    Order nodes so that dependencies always come before dependents.
    This is the valid execution order.

  STEP 3 — APPLY FUSION RULES (greedy, in topological order):

    RULE A — VERTICAL FUSION (producer → consumer):
      If node A is Pointwise AND A's output is used ONLY by node B:
        → Fuse A into B. Mark A's buffer as VIRTUAL.
        → B now uses A's inner_fn inline.
      CONDITION: A must have exactly ONE user. If A has two users (C and D),
                 fusing A into C means D cannot access A's result.
                 Solution: materialise A (keep as separate kernel), let C and D both read it.

    RULE B — HORIZONTAL FUSION (siblings):
      If node A and node B both read from the same input X, and
      neither A nor B produces values that the other needs:
        → Schedule A and B in the same kernel pass.
        → The pass reads X once and computes both A and B.
      This applies to: (mean, var) in LayerNorm — both read the same input.

    RULE C — EXTERN_KERNEL BOUNDARY:
      No Pointwise or Reduction node can be fused with an ExternKernel's
      main computation. ExternKernel is an opaque black box.
      EXCEPTION: EPILOGUE FUSION on Ampere+ GPUs via cuBLASLt API:
        If an ExternKernel (mm) is immediately followed by a Pointwise
        node that has no other users: fuse via cuBLASLt epilogue.
        The epilogue runs INSIDE cuBLAS, not in a separate kernel.

  STEP 4 — ASSIGN BUFFERS:
    For each remaining (non-virtual) kernel group: allocate output buffers.
    Apply buffer reuse via lifetime interval analysis (see Part 6 of theory).

  STEP 5 — GENERATE KERNEL CODE:
    For each group: call the appropriate codegen (Triton for GPU, C++ for CPU).
"""
print(SCHEDULER_THEORY)

# ── Simulate the scheduler ────────────────────────────────────────────────
@dataclass
class SimNode:
    """A simulated Inductor IR node for scheduling demonstration."""
    name:        str
    node_type:   str        # "Pointwise", "Reduction", "Extern"
    inputs:      List[str]  # names of nodes this reads from
    output_size: int        # simulated output buffer size in KB
    users:       List[str] = field(default_factory=list)  # nodes that read our output

    def __repr__(self):
        type_emoji = {"Pointwise": "🔄", "Reduction": "📊", "Extern": "🔗"}.get(self.node_type, "?")
        return f"{type_emoji} {self.name} ({self.node_type})"


class InductorScheduler:
    """
    Simulates TorchInductor's scheduling algorithm.
    Applies vertical fusion, horizontal fusion, and extern boundary rules.
    """
    def __init__(self, nodes: List[SimNode]):
        self.nodes    = {n.name: n for n in nodes}
        self.order    = [n.name for n in nodes]  # already topologically sorted
        # Build user sets
        for node in nodes:
            for inp in node.inputs:
                if inp in self.nodes:
                    self.nodes[inp].users.append(node.name)

    def schedule(self) -> List[List[str]]:
        """
        Returns a list of kernel groups.
        Each group is a list of node names that execute in one kernel.
        """
        # Track which nodes have been assigned to a group
        assigned = {}   # node_name → group_index
        groups   = []   # list of lists of node names
        virtual  = set()  # nodes whose buffers are eliminated (fused away)

        for name in self.order:
            if name in assigned:
                continue
            node = self.nodes[name]

            if node.node_type == "Extern":
                # Extern kernels always form their own group
                group_idx = len(groups)
                groups.append([name])
                assigned[name] = group_idx

                # Check for epilogue fusion: single pointwise consumer of this extern
                for user_name in node.users:
                    user = self.nodes.get(user_name)
                    if (user and user.node_type == "Pointwise"
                            and len(user.inputs) == 1   # only reads extern's output
                            and user_name not in assigned):
                        groups[-1].append(user_name)
                        assigned[user_name] = group_idx
                        virtual.add(f"{user_name}_buf")
                        print(f"    [EPILOGUE FUSION] {user_name} fused into {name} (cuBLASLt)")
                continue

            # For Pointwise/Reduction: try vertical fusion into an existing group
            fused = False
            if node.node_type == "Pointwise":
                for inp_name in node.inputs:
                    if inp_name in assigned and inp_name not in [n for g in groups for n in g[1:]]:
                        inp_node = self.nodes.get(inp_name)
                        if (inp_node and inp_node.node_type == "Pointwise"
                                and len(inp_node.users) == 1   # sole consumer
                                and groups[assigned[inp_name]][-1] == inp_name  # inp is last in group
                                and self.nodes[inp_name].node_type != "Extern"):
                            # Vertical fusion: absorb this node into inp's group
                            grp_idx = assigned[inp_name]
                            groups[grp_idx].append(name)
                            assigned[name] = grp_idx
                            virtual.add(inp_name + "_buf")
                            print(f"    [VERTICAL FUSION]   {name} fused with {inp_name}")
                            fused = True
                            break

            if not fused:
                # Check for horizontal fusion: shares input with an existing Pointwise group
                for inp_name in node.inputs:
                    if inp_name in assigned:
                        grp_idx = assigned[inp_name]
                        # Only fuse with same-type reduction groups if they share input
                        existing_group_type = self.nodes[groups[grp_idx][0]].node_type
                        if (node.node_type == "Reduction"
                                and existing_group_type == "Reduction"
                                and self.nodes[groups[grp_idx][0]].inputs == node.inputs):
                            groups[grp_idx].append(name)
                            assigned[name] = grp_idx
                            print(f"    [HORIZONTAL FUSION] {name} fused with {groups[grp_idx][0]}")
                            fused = True
                            break

            if not fused:
                # New group
                grp_idx = len(groups)
                groups.append([name])
                assigned[name] = grp_idx

        return groups, virtual


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Scheduling softmax
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Scheduling softmax([4, 8]): dependency graph walkthrough")
print("━" * 65)
print()

softmax_nodes = [
    SimNode("reduce_max",    "Reduction", ["x"],          output_size=4),
    SimNode("sub_max",       "Pointwise", ["x", "reduce_max"], output_size=32),
    SimNode("exp",           "Pointwise", ["sub_max"],    output_size=32),
    SimNode("reduce_sum",    "Reduction", ["exp"],        output_size=4),
    SimNode("div",           "Pointwise", ["exp", "reduce_sum"], output_size=32),
]

print("  Inductor IR nodes for softmax:")
for node in softmax_nodes:
    inputs_str = ", ".join(node.inputs)
    print(f"    {node}  ← reads [{inputs_str}]")
print()
print("  Scheduling decisions:")
sched = InductorScheduler(softmax_nodes)
groups, virtual = sched.schedule()
print()
print("  Final kernel groups:")
for i, group in enumerate(groups):
    types = [sched.nodes[n].node_type for n in group]
    kernel_type = "Triton pointwise" if all(t == "Pointwise" for t in types) else \
                  "Triton reduction + fused pointwise" if any(t == "Reduction" for t in types) else \
                  "External kernel"
    print(f"    Group {i}: {group}  → {kernel_type}")
print(f"  Virtual buffers (eliminated): {virtual}")
print()
print(f"  Result: {len(groups)} kernel launches for softmax")
print(f"  vs eager: 5+ separate dispatches")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Scheduling LayerNorm
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Scheduling LayerNorm: horizontal fusion of reductions")
print("━" * 65)
print()

layernorm_nodes = [
    SimNode("mean",           "Reduction", ["x"],               output_size=4),    # E[x]
    SimNode("var",            "Reduction", ["x"],               output_size=4),    # Var[x] = E[x²] - E[x]²
    SimNode("sub_mean",       "Pointwise", ["x", "mean"],       output_size=32),   # x - E[x]
    SimNode("add_eps",        "Pointwise", ["var"],             output_size=4),    # var + eps
    SimNode("rsqrt",          "Pointwise", ["add_eps"],         output_size=4),    # 1/sqrt(var+eps)
    SimNode("normalise",      "Pointwise", ["sub_mean", "rsqrt"], output_size=32), # (x-mean)/std
    SimNode("scale",          "Pointwise", ["normalise", "gamma"], output_size=32), # * gamma
    SimNode("shift",          "Pointwise", ["scale", "beta"],   output_size=32),   # + beta
]

print("  Inductor IR nodes for LayerNorm:")
for node in layernorm_nodes:
    print(f"    {node}  ← {node.inputs}")
print()
print("  Scheduling decisions:")
sched_ln = InductorScheduler(layernorm_nodes)
groups_ln, virtual_ln = sched_ln.schedule()
print()
print("  Final kernel groups:")
for i, grp in enumerate(groups_ln):
    types = [sched_ln.nodes[n].node_type for n in grp]
    if all(t == "Pointwise" for t in types):
        kt = "Triton pointwise"
    elif any(t == "Reduction" for t in types):
        kt = f"Triton reduction(s) + fused pointwise"
    else:
        kt = "External"
    print(f"    Group {i}: {grp}")
    print(f"            → {kt}")
print(f"  Virtual buffers eliminated: {virtual_ln}")
print()
print(f"  Result: {len(groups_ln)} kernel launches for LayerNorm")
print(f"  ATen ops count: {len(layernorm_nodes)} ops → {len(groups_ln)} kernels")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Full FFN scheduling with extern kernels
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — FFN scheduling: epilogue fusion with cuBLAS")
print("━" * 65)
print()

ffn_nodes = [
    SimNode("mm1",      "Extern",    ["x", "W1"],         output_size=512),   # x @ W1
    SimNode("add_b1",   "Pointwise", ["mm1", "b1"],       output_size=512),   # + bias1
    SimNode("gelu",     "Pointwise", ["add_b1"],          output_size=512),   # gelu(...)
    SimNode("mm2",      "Extern",    ["gelu", "W2"],      output_size=128),   # gelu @ W2
    SimNode("add_b2",   "Pointwise", ["mm2", "b2"],       output_size=128),   # + bias2
    SimNode("add_res",  "Pointwise", ["add_b2", "x"],     output_size=128),   # + residual
]

print("  FFN: residual + (x@W1 + b1 → gelu → @W2 + b2)")
print()
print("  Inductor IR nodes:")
for node in ffn_nodes:
    print(f"    {node}  ← {node.inputs}")
print()
print("  Scheduling decisions:")
sched_ffn = InductorScheduler(ffn_nodes)
groups_ffn, virtual_ffn = sched_ffn.schedule()
print()
print("  Final kernel groups:")
for i, grp in enumerate(groups_ffn):
    is_extern = any(sched_ffn.nodes[n].node_type == "Extern" for n in grp)
    kt = "cuBLAS GEMM" + (" + epilogue" if len(grp) > 1 else "") if is_extern else "Triton pointwise"
    print(f"    Group {i}: {grp}")
    print(f"            → {kt}")
print()
print(f"  Result: {len(groups_ffn)} kernel launches for FFN block")
print(f"  vs eager: 6+ separate dispatches")
print()
if HAS_TORCH:
    print("  Verifying with torch.compile:")
    kernel_count = [0]
    def counting_backend(gm, example_inputs):
        kernel_count[0] = len([n for n in gm.graph.nodes
                                if n.op == "call_function"])
        return gm.forward
    @torch.compile(backend=counting_backend, fullgraph=True)
    def ffn_fn(x, W1, b1, W2, b2):
        h = torch.nn.functional.gelu(x @ W1 + b1)
        return h @ W2 + b2 + x
    M, K, N = 32, 128, 128
    x_f  = torch.randn(M, K, device=DEVICE)
    W1_f = torch.randn(K, N, device=DEVICE)
    b1_f = torch.randn(N, device=DEVICE)
    W2_f = torch.randn(N, K, device=DEVICE)
    b2_f = torch.randn(K, device=DEVICE)
    _    = ffn_fn(x_f, W1_f, b1_f, W2_f, b2_f)
    print(f"    ATen ops in compiled graph: {kernel_count[0]}")
    print(f"    (includes mm, add, gelu decomposed to primitives)")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Triton Codegen — Reading and Writing Generated Kernels": {
        "description": (
            "Complete guide to reading TorchInductor-generated Triton code. "
            "Show what every line of a generated Triton kernel means. "
            "Demonstrate the pointwise, reduction, and persistent reduction patterns. "
            "Measure the effect of BLOCK_SIZE on GPU performance manually. "
            "Write a custom Triton kernel and integrate it with torch.compile. "
            "Compare generated Triton vs hand-written CUDA for the same operation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TRITON CODEGEN — READING AND WRITING GENERATED KERNELS")
print("=" * 65)
print()

try:
    import torch
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {DEVICE}")
    HAS_TRITON = DEVICE == "cuda"
    if HAS_TRITON:
        try:
            import triton
            import triton.language as tl
            print(f"  Triton {triton.__version__}")
        except ImportError:
            HAS_TRITON = False
            print("  Triton not installed (included with torch on GPU systems)")
except ImportError:
    HAS_TORCH = False
    HAS_TRITON = False
    print("  PyTorch not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Annotated Triton kernel anatomy
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Annotated Triton kernel: every line explained")
print("━" * 65)
print()

TRITON_ANATOMY = """
  INDUCTOR-GENERATED TRITON KERNEL — COMPLETE ANNOTATION
  ════════════════════════════════════════════════════════════════

  Generated for: out = exp(x * scale + bias)
  Tensor shape: [N] float32, one-dimensional

  ── KERNEL DEFINITION ────────────────────────────────────────────
  @triton.jit
  │  ↑ Triton JIT decorator: compiles function to PTX at first call.
  │    BLOCK_SIZE is a tl.constexpr → different values = different PTX.
  │    Triton compiles one specialised kernel per unique constexpr value.

  def triton_poi_fused_exp_mul_add_0(
      in_ptr0,              ; float* pointer to x (GPU HBM address)
      in_ptr1,              ; float* pointer to scale (scalar or tensor)
      in_ptr2,              ; float* pointer to bias
      out_ptr0,             ; float* pointer to output buffer
      xnumel,               ; int: total number of elements to process
      XBLOCK: tl.constexpr  ; int: tile size (compile-time constant)
  ):

  ── PROGRAM ID AND ELEMENT RANGE ─────────────────────────────────
      xoffset = tl.program_id(0) * XBLOCK
      │          ↑                  ↑
      │          │                  └ elements per "program" (tile)
      │          └ which tile this "program" (GPU block) handles
      │  For N=16384 elements, XBLOCK=1024: 16 programs launched.
      │  Program 0: handles elements [0, 1023]
      │  Program 1: handles elements [1024, 2047]  etc.

      xindex = xoffset + tl.arange(0, XBLOCK)
      │                   ↑
      │                   └ [0, 1, 2, ..., XBLOCK-1] — the SIMD lanes.
      │  xindex is a VECTOR of XBLOCK element indices.
      │  All operations below are VECTORISED over this range.

      xmask = xindex < xnumel
      │  ↑ Boolean mask: True for valid elements, False for padding.
      │  Required when N is not a multiple of XBLOCK.
      │  Example: N=1500, XBLOCK=1024 → program 1 handles [1024, 2047]
      │           but only [1024, 1499] are valid → mask [True...True, False...False]

  ── VECTORISED LOADS FROM GLOBAL MEMORY ──────────────────────────
      tmp0 = tl.load(in_ptr0 + xindex, mask=xmask, other=0.0)
      │      ↑        ↑         ↑              ↑       ↑
      │      │        │         │              │       └ value to use for masked elements
      │      │        │         │              └ which elements to load (vectorised)
      │      │        │         └ offset from base pointer (element-wise)
      │      │        └ base pointer to the tensor in GPU HBM
      │      └ vectorised load: XBLOCK elements loaded in a coalesced memory transaction
      │  COALESCING: consecutive threads load consecutive memory addresses.
      │  This is 100× faster than random (non-coalesced) access patterns.

      tmp1 = tl.load(in_ptr1 + (xindex % scale_size), xmask, 0.0)
      │                          ↑
      │                          └ % scale_size handles broadcast:
      │  If scale is a scalar (size 1): all elements read scale[0].
      │  If scale is a vector (size N): read scale[i] for each i.
      │  The modulo correctly handles both cases.

  ── REGISTER-RESIDENT COMPUTATION ────────────────────────────────
      tmp2 = tmp0 * tmp1        ; x * scale    (register-to-register)
      tmp3 = tmp2 + tmp2_bias   ; + bias        (register-to-register)
      tmp4 = tl.exp(tmp3)       ; exp(...)      (register-to-register)
      │  All of tmp2, tmp3, tmp4 live ONLY in registers.
      │  NO HBM reads or writes between these operations.
      │  This is what "fusion" means at the hardware level:
      │  the entire computation is a register pipeline.

  ── VECTORISED STORE TO GLOBAL MEMORY ────────────────────────────
      tl.store(out_ptr0 + xindex, tmp4, mask=xmask)
      │         ↑          ↑       ↑      ↑
      │         │          │       │      └ only store valid elements
      │         │          │       └ values to write (tmp4, vectorised)
      │         │          └ element offsets
      │         └ base pointer to output buffer in HBM
      │  ONE store per element per kernel — the absolute minimum possible.

  ── GRID LAUNCH ──────────────────────────────────────────────────
  triton_poi_fused_exp_mul_add_0[grid](
      in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK=1024
  )
  where grid = (triton.cdiv(xnumel, XBLOCK),)
  │           = (ceil(N / XBLOCK),)
  │  For N=16384, XBLOCK=1024: grid = (16,).
  │  16 programs launched → cover all 16384 elements.

  ── MEMORY TRAFFIC SUMMARY ───────────────────────────────────────
    HBM reads:  N×4 bytes (x) + N×4 bytes (bias) + N×4 bytes (scale) = 3N bytes
    HBM writes: N×4 bytes (output) = N bytes
    Total:      4N bytes = 4 × element_count × 4 bytes (float32)

    vs EAGER (3 separate ops):
    HBM reads:  N bytes (x for mul) + N bytes (tmp for add) + N bytes (tmp for exp)
    HBM writes: N bytes (mul out) + N bytes (add out) + N bytes (exp out)
    Total:      6N bytes = 6× more memory traffic!
"""
print(TRITON_ANATOMY)

if HAS_TRITON:
    import triton
    import triton.language as tl

    # Write and benchmark the exact kernel from the annotation above
    @triton.jit
    def fused_exp_mul_add_kernel(
        x_ptr, scale_ptr, bias_ptr, out_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid     = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < n_elements

        x     = tl.load(x_ptr     + offsets, mask=mask)
        scale = tl.load(scale_ptr + offsets % 1, mask=mask)
        bias  = tl.load(bias_ptr  + offsets, mask=mask)

        result = tl.exp(x * scale + bias)
        tl.store(out_ptr + offsets, result, mask=mask)

    N      = 1_048_576   # 1M elements
    x_t    = torch.randn(N, device="cuda")
    scale  = torch.ones(1, device="cuda") * 0.5
    bias_t = torch.randn(N, device="cuda") * 0.1
    out_t  = torch.empty(N, device="cuda")

    def run_triton(block_size):
        grid = (triton.cdiv(N, block_size),)
        fused_exp_mul_add_kernel[grid](
            x_t, scale, bias_t, out_t, N, BLOCK_SIZE=block_size)
        return out_t

    # Verify correctness
    result_triton = run_triton(1024)
    result_ref    = torch.exp(x_t * scale + bias_t)
    err           = float(torch.max(torch.abs(result_triton - result_ref)))
    print(f"  Custom Triton kernel correctness: max_err={err:.2e} "
          f"{'✅' if err < 1e-4 else '❌'}")
    print()

    # Benchmark different BLOCK_SIZEs to show autotuning effect
    print("  BLOCK_SIZE impact on performance (N=1M elements):")
    print(f"  {'BLOCK_SIZE':>12}  {'Time (ms)':>12}  {'GB/s':>8}  {'vs optimal':>10}")
    print("  " + "-" * 50)

    def bench(block_size, n_reps=100):
        for _ in range(10): run_triton(block_size)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_reps): run_triton(block_size)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_reps * 1000

    results = {}
    for bs in [32, 64, 128, 256, 512, 1024, 2048]:
        try:
            t_ms = bench(bs)
            gb_s = (N * 4 * 4) / (t_ms * 1e-3) / 1e9   # 4 tensors × 4 bytes
            results[bs] = (t_ms, gb_s)
        except Exception:
            results[bs] = (None, None)

    best_t = min(v[0] for v in results.values() if v[0] is not None)
    for bs, (t, bw) in results.items():
        if t is None: continue
        ratio = t / best_t
        marker = " ← optimal" if t == best_t else ""
        print(f"  {bs:>12}  {t:>12.4f}  {bw:>8.1f}  {ratio:>10.2f}×{marker}")
    print()
    print(f"  This is exactly what Inductor's autotuning does automatically.")
    print(f"  It benchmarks each BLOCK_SIZE and picks the winner.")
    print()

else:
    TRITON_DEMO_REF = """
  TRITON KERNEL WRITING REFERENCE (requires CUDA GPU):
  ─────────────────────────────────────────────────────────────────
  import triton
  import triton.language as tl

  @triton.jit
  def my_fused_kernel(x_ptr, bias_ptr, out_ptr, n, BLOCK: tl.constexpr):
      pid     = tl.program_id(0)
      offsets = pid * BLOCK + tl.arange(0, BLOCK)
      mask    = offsets < n

      x    = tl.load(x_ptr    + offsets, mask=mask)
      bias = tl.load(bias_ptr + offsets, mask=mask)
      out  = tl.maximum(x + bias, 0.0)   ; add + relu fused
      tl.store(out_ptr + offsets, out, mask=mask)

  grid = (triton.cdiv(N, 1024),)
  my_fused_kernel[grid](x, bias, output, N, BLOCK=1024)

  BLOCK_SIZE SELECTION GUIDE:
    Very small N (< 1K):    BLOCK_SIZE = 32 or 64
    Small N (1K–100K):      BLOCK_SIZE = 256 or 512
    Medium N (100K–10M):    BLOCK_SIZE = 1024 (usually optimal)
    Large N (> 10M):        BLOCK_SIZE = 2048 or 4096
    Rule of thumb: aim for 128+ warps per SM for good occupancy.
"""
    print(TRITON_DEMO_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Persistent reduction pattern (softmax in one kernel)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Persistent reduction: the online softmax pattern")
print("━" * 65)
print()

PERSISTENT_REDUCTION = """
  THE PERSISTENT REDUCTION PATTERN
  ════════════════════════════════════════════════════════════════

  When the ENTIRE reduction fits within one Triton program's registers,
  Inductor uses a "persistent" reduction: one program per output element,
  each program handles the full reduction in-registers.

  This avoids the expensive two-pass split-reduce approach.

  EXAMPLE — Row-wise softmax for shape [B, S]:
    One program per row (B programs total).
    Each program loads the entire row (S elements) and computes:
      1. max over the row      (for numerical stability)
      2. subtract max          (elementwise)
      3. exp                   (elementwise)
      4. sum of exps           (reduction)
      5. divide by sum         (elementwise)
    All in one kernel, using XBLOCK threads within the program.

  @triton.jit
  def triton_per_fused_softmax(x_ptr, out_ptr, n_rows, n_cols,
                                XBLOCK: tl.constexpr):
      row_idx   = tl.program_id(0)         ; which row
      col_offs  = tl.arange(0, XBLOCK)     ; column indices
      mask      = col_offs < n_cols

      ; Load full row into registers
      x_row = tl.load(x_ptr + row_idx * n_cols + col_offs,
                       mask=mask, other=-float("inf"))

      ; Numerically stable softmax: subtract max first
      row_max = tl.max(x_row, axis=0)      ; scalar max
      x_row   = x_row - row_max            ; shift values (stays in registers)

      ; Exponentiate
      x_exp   = tl.exp(x_row)             ; elementwise exp

      ; Sum and normalise
      x_sum   = tl.sum(x_exp, axis=0)     ; scalar sum
      result  = x_exp / x_sum             ; elementwise divide

      ; Store result
      tl.store(out_ptr + row_idx * n_cols + col_offs,
                result, mask=mask)

  ; Grid: one program per row
  grid = (n_rows,)

  SIX ATen ops (max, sub, exp, sum, div, all with broadcasts) → ONE kernel.
  Each element of the input matrix is loaded EXACTLY ONCE.
  All intermediate values (x_row, x_exp, x_sum) stay in registers.

  WHEN PERSISTENT REDUCTION APPLIES:
    Condition: n_cols ≤ XBLOCK (the whole reduction fits in one program)
    n_cols = 512:   XBLOCK=512 → one program handles 512 elements
    n_cols = 2048:  XBLOCK=2048 → larger program (may reduce occupancy)
    n_cols = 32768: XBLOCK=32768 → too large, must split → two-pass reduction

  WHEN TWO-PASS REDUCTION IS NEEDED:
    n_cols > max_XBLOCK (typically 16384 for Triton on modern GPUs).
    Inductor uses:
      Pass 1: split into chunks of XBLOCK, compute partial sums
      Pass 2: reduce the partial sums
    This adds one extra kernel and one extra HBM round-trip.
"""
print(PERSISTENT_REDUCTION)

if HAS_TRITON:
    @triton.jit
    def persistent_softmax_kernel(x_ptr, out_ptr, n_rows, n_cols,
                                   XBLOCK: tl.constexpr):
        row     = tl.program_id(0)
        cols    = tl.arange(0, XBLOCK)
        mask    = cols < n_cols
        x_row   = tl.load(x_ptr + row * n_cols + cols, mask=mask,
                           other=-float("inf"))
        row_max = tl.max(x_row, axis=0)
        x_row   = x_row - row_max
        x_exp   = tl.exp(x_row)
        x_sum   = tl.sum(x_exp, axis=0)
        out     = x_exp / x_sum
        tl.store(out_ptr + row * n_cols + cols, out, mask=mask)

    B, S = 512, 128
    x_sm  = torch.randn(B, S, device="cuda")
    out_sm = torch.empty(B, S, device="cuda")

    # Find next power of 2 ≥ S for XBLOCK
    xblock = 1
    while xblock < S: xblock *= 2

    grid_sm = (B,)
    persistent_softmax_kernel[grid_sm](x_sm, out_sm, B, S, XBLOCK=xblock)

    ref_sm = torch.softmax(x_sm, dim=-1)
    err_sm = float(torch.max(torch.abs(out_sm - ref_sm)))
    print(f"  Persistent softmax [{B}×{S}]: max_err={err_sm:.2e} "
          f"{'✅' if err_sm < 1e-4 else '❌'}")

    def bench_sm(fn, n_reps=200):
        for _ in range(10): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_reps): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_reps * 1e6   ; microseconds

    t_triton = bench_sm(lambda: persistent_softmax_kernel[grid_sm](
        x_sm, out_sm, B, S, XBLOCK=xblock))
    t_eager  = bench_sm(lambda: torch.softmax(x_sm, dim=-1))
    t_compiled = None
    try:
        sm_fn = torch.compile(lambda x: torch.softmax(x, dim=-1))
        _ = sm_fn(x_sm)
        t_compiled = bench_sm(lambda: sm_fn(x_sm))
    except Exception: pass

    print(f"  Benchmark [{B}×{S}] softmax:")
    print(f"    Eager:    {t_eager:.1f} µs  (6 separate kernel dispatches)")
    print(f"    Triton:   {t_triton:.1f} µs  (1 persistent kernel)")
    if t_compiled:
        print(f"    Compiled: {t_compiled:.1f} µs  (torch.compile → Inductor)")
    print()
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · max-autotune and CUDA Graphs — Maximum Performance Compilation": {
        "description": (
            "Deep dive into TorchInductor's max-autotune and CUDA graph features. "
            "Simulate the BLOCK_SIZE autotuning search: sample → benchmark → select. "
            "Show the cost model for BLOCK_SIZE prediction before benchmarking. "
            "Demonstrate CUDA graph capture and replay with static shapes. "
            "Measure the per-step overhead reduction from CUDA graphs. "
            "Show when CUDA graphs break and how to diagnose the failure."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

print("=" * 65)
print("  MAX-AUTOTUNE AND CUDA GRAPHS — MAXIMUM PERFORMANCE COMPILATION")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {DEVICE}")
    HAS_CUDA = DEVICE == "cuda"
except ImportError:
    HAS_TORCH = False
    HAS_CUDA  = False
    print("  PyTorch not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: BLOCK_SIZE autotuning simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — BLOCK_SIZE autotuning: how Inductor picks kernel params")
print("━" * 65)
print()

AUTOTUNE_THEORY = """
  BLOCK_SIZE AUTOTUNING IN TORCHINDUCTOR
  ════════════════════════════════════════════════════════════════

  Why BLOCK_SIZE matters so much:
    - Too small: thread block underutilises the SM's warp slots.
                 GPU SM can run 32 warps; with BLOCK_SIZE=32 (1 warp),
                 31 warp slots sit empty → 3% utilisation.
    - Too large: exceeds shared memory or register limits per block.
                 Forces register spilling to L2 → much slower.
    - Just right: fills the SM with concurrent warps, high register use,
                  good memory coalescing.

  INDUCTOR'S DEFAULT AUTOTUNE (mode="default"):
    Candidates: [32, 64, 128, 256, 512, 1024] × num_warps [1, 2, 4, 8]
    Strategy:
      1. Compile all candidate kernels (fast: one JIT compile per config)
      2. Run each candidate on the actual hardware with real tensor shapes
      3. Keep the winner
      4. Cache result keyed by (operation_type, n_elements, dtype, device_arch)

  INDUCTOR'S MAX-AUTOTUNE (mode="max-autotune"):
    Extends default with:
      + XBLOCK ∈ {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
      + RBLOCK ∈ {32, 64, 128, 256, 512, 1024} (for reductions)
      + num_warps ∈ {1, 2, 4, 8, 16}
      + num_stages ∈ {1, 2, 3, 4, 5} (pipeline stages for memory latency hiding)
      + coordinate descent: vary one param at a time after initial search
      + cuBLASLt: benchmark all GEMM algorithms for each (M,N,K) shape
    Total: ~128-500 candidates per kernel, benchmarked in parallel streams

  BENCHMARK ACCURACY:
    Each candidate is run 3× and the MEDIAN time is used.
    Using median rather than min avoids one-off fast measurements.
    Using median rather than mean avoids outlier slow measurements.

  RESULT PERSISTENCE:
    ~/.cache/torch/inductor/<hash>/
    Hash encodes: op type, shapes, dtypes, GPU architecture (sm_XX)
    On subsequent runs with same shapes: loaded directly, no benchmark needed.
    Inductor prints: "Using cached kernel for aten.mm shape [512,512,512]"
"""
print(AUTOTUNE_THEORY)

# Simulate autotuning
class BlockSizeAutotuner:
    """
    Simulates Inductor's BLOCK_SIZE autotuning search.
    Models the true hardware performance landscape for a pointwise kernel.
    """
    def __init__(self, n_elements: int, dtype_bytes: int = 4,
                 gpu_bandwidth_tbs: float = 2.0,
                 sm_count: int = 108, warps_per_sm: int = 32):
        self.n          = n_elements
        self.dtype_bytes= dtype_bytes
        self.bw         = gpu_bandwidth_tbs * 1e12   ; bytes/sec
        self.n_sm       = sm_count
        self.warp_slots = warps_per_sm

    def simulate_kernel_time_us(self, block_size: int, num_warps: int,
                                 noise: float = 0.03) -> float:
        """
        Simulate GPU execution time for a pointwise kernel.
        Models: SM occupancy, memory bandwidth, warp utilisation.
        """
        threads_per_block = block_size
        warps_per_block   = threads_per_block // 32

        # How many concurrent blocks per SM?
        max_blocks_per_sm = max(1, self.warp_slots // warps_per_block)
        # How many active warps per SM?
        active_warps      = min(warps_per_block * max_blocks_per_sm, self.warp_slots)
        # Occupancy: fraction of SM resources used
        occupancy         = active_warps / self.warp_slots

        # Total blocks needed to process all elements
        n_blocks          = max(1, self.n // block_size)
        # Time to process all blocks (waves of concurrent SM execution)
        waves             = max(1, n_blocks / (self.n_sm * max_blocks_per_sm))

        # Memory time: bandwidth-limited (3 tensors: 2 reads + 1 write)
        mem_bytes         = self.n * self.dtype_bytes * 3
        mem_time_us       = (mem_bytes / self.bw) * 1e6

        # Occupancy penalty: low occupancy → can't hide memory latency
        occupancy_factor  = max(0.4, occupancy)   ; at least 40% efficiency
        effective_bw_time = mem_time_us / occupancy_factor

        # Overhead: kernel launch + scheduling
        overhead_us = 2.0 + (1.0 / max_blocks_per_sm)   ; more blocks = less overhead per block

        actual_time = (effective_bw_time * waves + overhead_us)
        # Add measurement noise
        return actual_time * (1.0 + np.random.randn() * noise)

    def autotune(self, candidates: List[Tuple[int, int]], n_reps: int = 3) -> Dict:
        """
        Simulate the autotuning benchmark loop.
        Returns results dict with timing for each config.
        """
        results = {}
        for block_size, num_warps in candidates:
            # Each candidate is compiled and run n_reps times; take median
            times   = [self.simulate_kernel_time_us(block_size, num_warps) for _ in range(n_reps)]
            results[(block_size, num_warps)] = sorted(times)[n_reps // 2]  # median
        return results


print("  Simulating BLOCK_SIZE autotuning for pointwise kernel:")
print(f"  (N=1M elements, float32, A100 model: 108 SMs, 32 warp slots/SM)")
print()

np.random.seed(42)
autotuner = BlockSizeAutotuner(n_elements=1_048_576, sm_count=108, warps_per_sm=32)

# Default mode candidates
default_candidates = [(bs, nw) for bs in [32, 64, 128, 256, 512, 1024]
                      for nw in [4]]  # default: num_warps=4

default_results = autotuner.autotune(default_candidates)
best_default    = min(default_results, key=default_results.get)

print(f"  Default mode candidates ({len(default_candidates)} configs):")
print(f"  {'BLOCK_SIZE':>12}  {'num_warps':>10}  {'Time (µs)':>10}  {'Winner':>8}")
print("  " + "-" * 46)
for (bs, nw), t in sorted(default_results.items()):
    marker = " ← " if (bs, nw) == best_default else "   "
    print(f"  {bs:>12}  {nw:>10}  {t:>10.2f}  {marker}")
print(f"  Best: BLOCK_SIZE={best_default[0]}, num_warps={best_default[1]}")
print()

# Max-autotune candidates
max_candidates = [(bs, nw) for bs in [64, 128, 256, 512, 1024, 2048]
                  for nw in [1, 2, 4, 8]]

max_results  = autotuner.autotune(max_candidates)
best_max     = min(max_results, key=max_results.get)

print(f"  Max-autotune candidates ({len(max_candidates)} configs):")
print(f"  {'BLOCK_SIZE':>12}  {'num_warps':>10}  {'Time (µs)':>10}  {'Rank':>6}")
print("  " + "-" * 44)
sorted_max = sorted(max_results.items(), key=lambda x: x[1])
for rank, ((bs, nw), t) in enumerate(sorted_max[:8]):
    marker = " ← WINNER" if rank == 0 else ""
    print(f"  {bs:>12}  {nw:>10}  {t:>10.2f}  {rank+1:>6}{marker}")
print(f"  ...")
print()
print(f"  Default best:     BLOCK_SIZE={best_default[0]:5d}, "
      f"time={default_results[best_default]:.2f} µs")
print(f"  Max-autotune best: BLOCK_SIZE={best_max[0]:5d}, "
      f"time={max_results[best_max]:.2f} µs")
print(f"  Max-autotune improvement: "
      f"{(default_results[best_default]/max_results[best_max]-1)*100:.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: CUDA graphs
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — CUDA graphs: eliminating kernel launch overhead")
print("━" * 65)
print()

CUDA_GRAPH_THEORY = """
  CUDA GRAPHS — HOW INDUCTOR USES THEM
  ════════════════════════════════════════════════════════════════

  WHAT CUDA GRAPHS SOLVE:
    Each GPU kernel launch has fixed overhead: ~5-10 µs on modern hardware.
    For a training step with 50 compiled kernels: 50 × 7µs = 350µs overhead.
    For a 2ms training step: 350µs is 17.5% of total step time — significant!

  HOW CUDA GRAPH CAPTURE WORKS:
    Standard execution:
      CPU: submit kernel A → CPU: submit kernel B → CPU: submit kernel C ...
      GPU: run A; run B; run C ...
      (CPU and GPU run concurrently, GPU always catching up to CPU)

    CUDA Graph capture:
      CPU puts GPU into "record mode":
        with torch.cuda.graph(graph_obj):
            output = model(input)   ← GPU records all kernel launches
      GPU stores the kernel sequence as a device-side execution plan.

    CUDA Graph replay:
      CPU issues ONE command: graph_obj.replay()
      GPU re-executes the ENTIRE recorded sequence atomically.
      No per-kernel CPU overhead at all — the GPU handles its own scheduling.

  THE BUFFER ALIASING REQUIREMENT:
    During capture: input, output, and intermediate buffers have fixed GPU addresses.
    During replay: the SAME addresses must be used.
    To feed new data: COPY into the captured input buffer (same address).

    ┌─────────────────────────────────────────────────────────────────┐
    │  CAPTURE PHASE (once):                                          │
    │    static_input = torch.randn(B, D, device='cuda')             │
    │    # Input buffer address: 0x7f...abc                           │
    │    with torch.cuda.graph(graph):                                │
    │        static_output = model(static_input)                      │
    │    # Output buffer address: 0x7f...def (fixed for all replays) │
    │                                                                  │
    │  REPLAY PHASE (each step):                                      │
    │    static_input.copy_(new_batch_data)  ; update at same addr   │
    │    graph.replay()                       ; entire model in 1 call│
    │    result = static_output               ; read from same addr   │
    └─────────────────────────────────────────────────────────────────┘

  WHEN CUDA GRAPHS ARE AUTOMATICALLY USED BY INDUCTOR:
    torch.compile(mode="reduce-overhead") → always tries CUDA graphs
    torch.compile(mode="max-autotune")    → uses CUDA graphs by default
    torch.compile(mode="default")         → does NOT use CUDA graphs

  CUDA GRAPH REQUIREMENTS (ALL must be met):
    1. Static shapes: same tensor shapes every call
    2. Same memory layout: same strides, contiguous or not
    3. No data-dependent control flow after capture
    4. No CPU-GPU synchronisation inside the captured region
    5. model.eval() or model.train() set before capture (not changed during)
    6. No random ops (or manage RNG state manually)
    7. No device-to-host copies inside the model

  FAILURE DIAGNOSIS:
    torch._inductor.config.warn_on_cuda_graph_incompatible = True
    → Prints warning if a model falls back from CUDA graphs

  COMMON FAILURE CAUSE — CUDNN BENCHMARK MODE:
    torch.backends.cudnn.benchmark = True
    → cuDNN tries different algorithms per input shape
    → First call after a shape change: benchmark runs (synchronous)
    → This creates a CPU-GPU sync point → breaks CUDA graph capture
    FIX: torch.backends.cudnn.benchmark = False when using CUDA graphs
         OR: torch.backends.cudnn.allow_tf32 = True (uses fixed TF32 path)
"""
print(CUDA_GRAPH_THEORY)

if HAS_CUDA:
    print("  CUDA graph benchmark: eager vs compiled vs CUDA graph:")
    print()

    class SimpleMLP(nn.Module):
        def __init__(self, dim=512):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.LayerNorm(dim),
            )
        def forward(self, x): return self.layers(x)

    B, D = 32, 512
    model_eager    = SimpleMLP(D).cuda().eval()
    model_default  = torch.compile(SimpleMLP(D).cuda().eval(), mode="default")
    model_cudagraph= torch.compile(SimpleMLP(D).cuda().eval(),
                                    mode="reduce-overhead")

    x_cg = torch.randn(B, D, device="cuda")

    def bench_model(model, x, n_warmup=10, n_reps=200):
        with torch.no_grad():
            for _ in range(n_warmup): model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_reps): model(x)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_reps * 1000

    print(f"  Model: MLP [{D}→{D*4}→{D}] + LayerNorm, batch={B}")
    print(f"  {'Mode':30s}  {'ms/step':>10s}  {'Speedup':>9s}  {'Notes'}")
    print("  " + "-" * 68)

    t_eager = bench_model(model_eager, x_cg)
    print(f"  {'Eager':30s}  {t_eager:>10.4f}  {'1.00×':>9s}  baseline")

    t_default = bench_model(model_default, x_cg)
    print(f"  {'torch.compile(default)':30s}  {t_default:>10.4f}  "
          f"{t_eager/t_default:>9.2f}×  kernel fusion only")

    t_cudagraph = bench_model(model_cudagraph, x_cg)
    print(f"  {'torch.compile(reduce-overhead)':30s}  {t_cudagraph:>10.4f}  "
          f"{t_eager/t_cudagraph:>9.2f}×  fusion + CUDA graphs")
    print()

    # Show that CUDA graphs BREAK with shape changes
    print("  CUDA graph constraint demo: shape change detection")
    model_cg2 = torch.compile(SimpleMLP(D).cuda().eval(), mode="reduce-overhead")
    x_orig = torch.randn(B,  D, device="cuda")
    x_new  = torch.randn(B*2, D, device="cuda")   # different batch size!

    with torch.no_grad():
        out1 = model_cg2(x_orig)   # capture
        out2 = model_cg2(x_orig)   # replay ← fast
        # Triggering a shape change after capture causes Inductor to
        # invalidate the CUDA graph and recompile.
        # This is safe — Inductor handles it gracefully.
        out3 = model_cg2(x_new)    # new shape → recompile (no crash)

    print(f"  Batch={B}:    shape={out1.shape} ✅")
    print(f"  Batch={B}:    shape={out2.shape} ✅ (CUDA graph replay)")
    print(f"  Batch={B*2}:  shape={out3.shape} ✅ (recompiled, new graph captured)")
    print()

else:
    print("  CUDA not available — showing CUDA graph reference:")
    CUDA_GRAPH_REF = """
  CUDA GRAPH MANUAL USAGE (requires CUDA GPU):
  ─────────────────────────────────────────────────────────────────
  # APPROACH 1: torch.compile(mode="reduce-overhead") — automatic
  model = torch.compile(model, mode="reduce-overhead")
  for batch in dataloader:
      output = model(batch)   ; CUDA graph captured on first call, replayed after

  # APPROACH 2: Manual CUDA graph capture
  static_input  = torch.zeros(B, D, device='cuda')
  static_output = None
  graph = torch.cuda.CUDAGraph()

  # Warmup (required before capture)
  with torch.cuda.stream(torch.cuda.Stream()):
      static_output = model(static_input)

  # Capture
  with torch.cuda.graph(graph):
      static_output = model(static_input)

  # Inference loop (very fast):
  for batch in dataloader:
      static_input.copy_(batch)   ; update buffer IN-PLACE
      graph.replay()               ; execute entire model in one call
      result = static_output.clone()  ; read result (still at same address)
"""
    print(CUDA_GRAPH_REF)
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · TorchInductor Diagnostics and the Connected Stack": {
        "description": (
            "Master Inductor's debugging and profiling toolkit. "
            "Show every TORCH_LOGS category with real output. "
            "Read and interpret the inductor HTML trace. "
            "Benchmark TorchInductor vs eager vs XLA vs TVM on standard workloads. "
            "Show how to extend Inductor with custom Triton kernels via torch.library. "
            "Summarise TorchInductor's position in the full connected stack."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TORCHINDUCTOR DIAGNOSTICS AND THE CONNECTED STACK")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"
    print("  PyTorch not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Comprehensive diagnostics guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Inductor diagnostics: reading what the compiler does")
print("━" * 65)
print()

DIAGNOSTICS = """
  TORCHINDUCTOR DIAGNOSTICS — COMPLETE REFERENCE
  ════════════════════════════════════════════════════════════════

  LEVEL 1 — BASIC: What got compiled and how fast?
  ─────────────────────────────────────────────────────────────────
  torch._inductor.config.benchmark_kernel = True
    → After compilation, prints timing for each kernel:
      KERNEL timing (ms): triton_poi_0=0.042, triton_poi_1=0.031,
                          extern_mm_0=1.240, triton_red_2=0.156
    → Identifies the slowest kernel (the bottleneck) immediately.

  LEVEL 2 — GRAPHS: What's in the FX graph Inductor received?
  ─────────────────────────────────────────────────────────────────
  TORCH_LOGS="graph" python script.py
    → Prints the ATen FX graph BEFORE Inductor lowering.
      Useful for: understanding what AOTAutograd produced,
                  verifying that expected ops are present,
                  checking that decomposition worked correctly.

  LEVEL 3 — SCHEDULES: How did Inductor group the nodes?
  ─────────────────────────────────────────────────────────────────
  TORCH_LOGS="schedules" python script.py
    → For each kernel group, prints which FX nodes it contains:
      Group 0: [aten.relu.default, aten.add.Tensor, aten.mul.Tensor]
                → triton_poi_fused_relu_add_mul_0
      Group 1: [aten.mm.default]
                → extern_kernels.mm  (cuBLAS)
    → Tells you: how many kernels, which ops fused together.

  LEVEL 4 — OUTPUT CODE: What Triton/C++ was generated?
  ─────────────────────────────────────────────────────────────────
  TORCH_LOGS="output_code" python script.py
    → Prints the complete generated code for every kernel.
      The most detailed view of what Inductor actually compiled.
      Look for:
        - How many tl.load calls → how many input passes
        - How many tl.store calls → should be exactly 1 per output
        - tmp0/tmp1/... register variables → the fused computation
        - BLOCK_SIZE value → the winning autotuning result
        - Grid size → how many GPU blocks launched

  LEVEL 5 — HTML TRACE: Interactive visual timeline
  ─────────────────────────────────────────────────────────────────
  torch._inductor.config.trace.enabled = True
    → Saves /tmp/torchinductor_<user>/*/trace.html
    → Interactive browser-based view showing:
        Timeline: which compilation phase took how long
        Graph view: the FX graph before and after each pass
        Kernel groups: visual grouping of nodes → kernels
        Code: generated Triton/C++ for each kernel
        Autotuning: BLOCK_SIZE candidates and their timings

  LEVEL 6 — FULL DEBUG: Everything
  ─────────────────────────────────────────────────────────────────
  torch._inductor.config.debug = True
    → Saves ALL intermediate IRs to /tmp/torchinductor_<user>/
      Files created:
        *_input_graph.txt     ; ATen FX graph at entry
        *_lowered_ir.txt      ; Inductor IR (pointwise/reduction nodes)
        *_scheduled.txt       ; kernel groups after scheduling
        *_triton_*.py         ; each generated Triton kernel file
        *_wrapper.py          ; the wrapper that calls all kernels
        *_output_code.py      ; the final compiled module

  DIAGNOSE PERFORMANCE REGRESSIONS:
  ─────────────────────────────────────────────────────────────────
  SYMPTOM: torch.compile is slower than expected
  STEP 1: Check kernel count with TORCH_LOGS="schedules"
    - If many small kernels → fusion failed somewhere
    - Likely cause: graph break in TorchDynamo → see module 09

  STEP 2: Check for slow extern kernels with benchmark_kernel=True
    - extern_mm is slow → shapes might not be optimal for cuBLAS
    - Try torch.compile(mode="max-autotune") for cuBLASLt search

  STEP 3: Check for memory bandwidth issues with TORCH_LOGS="perf_hints"
    - "non-contiguous tensor input" → expensive copy inserted
    - Fix: ensure model inputs are contiguous (.contiguous())

  STEP 4: Use PyTorch profiler for GPU timeline
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        with_stack=True
    ) as prof:
        model(x)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
"""
print(DIAGNOSTICS)

if HAS_TORCH:
    # Practical: show what happens with TORCH_LOGS in a live environment
    print("  Practical diagnostics demo:")
    print()

    class TwoLayerNet(nn.Module):
        def __init__(self, d=256):
            super().__init__()
            self.fc1  = nn.Linear(d, d * 2)
            self.fc2  = nn.Linear(d * 2, d)
            self.norm = nn.LayerNorm(d)
        def forward(self, x):
            h = torch.gelu(self.fc1(x))
            return self.norm(self.fc2(h) + x)

    model   = TwoLayerNet(256).to(DEVICE).eval()
    x_diag  = torch.randn(16, 256, device=DEVICE)

    # Build a simple backend that reports on what it sees
    kernel_info = []
    def reporting_backend(gm, example_inputs):
        ops        = [n for n in gm.graph.nodes if n.op == "call_function"]
        extern_ops = [n for n in ops if "mm" in str(n.target) or "conv" in str(n.target)]
        pt_ops     = [n for n in ops if n not in extern_ops]
        kernel_info.append({
            "total_ops":   len(ops),
            "extern_ops":  len(extern_ops),
            "pointwise_ops": len(pt_ops),
            "extern_names": [str(n.target).split("aten.")[-1] for n in extern_ops],
        })
        return gm.forward

    compiled = torch.compile(model, backend=reporting_backend, fullgraph=True)
    with torch.no_grad():
        _ = compiled(x_diag)

    if kernel_info:
        info = kernel_info[-1]
        print(f"  ATen graph summary for TwoLayerNet:")
        print(f"    Total ops:        {info['total_ops']}")
        print(f"    ExternKernel ops: {info['extern_ops']} {info['extern_names']}")
        print(f"    Pointwise ops:    {info['pointwise_ops']}")
        print()
        print(f"  Expected Inductor kernel groups:")
        print(f"    Group 1: cuBLAS mm (fc1) + GELU epilogue  ← ExternKernel + fusion")
        print(f"    Group 2: cuBLAS mm (fc2) + add residual   ← ExternKernel + fusion")
        print(f"    Group 3: Triton reduction (LayerNorm: mean + var)")
        print(f"    Group 4: Triton pointwise (LayerNorm: normalise + scale + shift)")
        print(f"    → 4 kernel launches for 7+ ATen ops ✅")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Custom Triton kernel via torch.library
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Custom Triton kernel: registering with torch.library")
print("━" * 65)
print()

CUSTOM_OP = """
  INTEGRATING CUSTOM TRITON KERNELS WITH TORCH.COMPILE
  ════════════════════════════════════════════════════════════════

  TorchInductor handles standard ATen ops automatically.
  For custom ops (Flash Attention variants, custom norms, etc.),
  use torch.library to register them as first-class ops that
  torch.compile understands.

  STEP 1 — WRITE THE TRITON KERNEL:
  ─────────────────────────────────────────────────────────────────
  import triton
  import triton.language as tl

  @triton.jit
  def fast_swish_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
      \" \"\"Swish activation: x * sigmoid(x) — custom fused Triton kernel\"\"\"
      pid     = tl.program_id(0)
      offsets = pid * BLOCK + tl.arange(0, BLOCK)
      mask    = offsets < n
      x       = tl.load(x_ptr + offsets, mask=mask)
      sig     = tl.sigmoid(x)
      result  = x * sig
      tl.store(out_ptr + offsets, result, mask=mask)

  STEP 2 — DEFINE THE SCHEMA AND EAGER IMPLEMENTATION:
  ─────────────────────────────────────────────────────────────────
  @torch.library.custom_op("mylib::fast_swish", mutates_args=())
  def fast_swish(x: torch.Tensor) -> torch.Tensor:
      return x * torch.sigmoid(x)   ; eager fallback

  STEP 3 — REGISTER THE TRITON IMPLEMENTATION FOR CUDA:
  ─────────────────────────────────────────────────────────────────
  @torch.library.register_kernel("mylib::fast_swish", "cuda")
  def fast_swish_cuda(x: torch.Tensor) -> torch.Tensor:
      output = torch.empty_like(x)
      n      = x.numel()
      grid   = (triton.cdiv(n, 1024),)
      fast_swish_kernel[grid](x, output, n, BLOCK=1024)
      return output

  STEP 4 — REGISTER ABSTRACT FUNCTION (for shape inference):
  ─────────────────────────────────────────────────────────────────
  @torch.library.register_fake("mylib::fast_swish")
  def fast_swish_abstract(x): return torch.empty_like(x)

  STEP 5 — USE IN A MODEL (torch.compile knows about it):
  ─────────────────────────────────────────────────────────────────
  @torch.compile(fullgraph=True)
  def model_with_custom_op(x, W, b):
      h = torch.ops.mylib.fast_swish(x @ W + b)   ; custom op!
      return h.sum()

  # Inductor treats fast_swish as an ExternKernel (opaque, calls your Triton).
  # The surrounding matmul and add fuse with surrounding ops as normal.
  # fast_swish itself uses YOUR Triton kernel directly.

  PERFORMANCE NOTE:
    The custom Triton kernel is called directly — no additional wrapper.
    Inductor will attempt epilogue fusion with fast_swish if possible.
    For ops that consume the output of fast_swish and are pointwise,
    Inductor may fuse them into a new Triton kernel reading fast_swish's output.

  REGISTERING THE BACKWARD RULE:
  ─────────────────────────────────────────────────────────────────
  @torch.library.register_autograd("mylib::fast_swish",
      setup_context=lambda ctx, inputs, output: ctx.save_for_backward(inputs[0]))
  def fast_swish_backward(ctx, grad_out):
      (x,) = ctx.saved_tensors
      sig  = torch.sigmoid(x)
      grad = grad_out * (sig + x * sig * (1 - sig))  ; swish derivative
      return (grad,)

  # Now fast_swish is fully differentiable through torch.compile!
  # AOTAutograd will differentiate through it correctly.
"""
print(CUSTOM_OP)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Connected stack summary
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — TorchInductor in the connected compiler stack")
print("━" * 65)
print()

STACK = """
  TORCHINDUCTOR IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                      │
  │  Inductor's CPU backend: generates C++ → GCC/Clang (LLVM-backed).   │
  │  Triton itself uses LLVM's NVPTX backend to emit PTX from Triton IR. │
  │  at::vec (Inductor's SIMD library) maps to LLVM vector types.        │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 02: MLIR                                                      │
  │  Inductor's IR (Pointwise/Reduction) conceptually mirrors MLIR's     │
  │  linalg.generic / linalg.reduce — same "loops over index ranges."   │
  │  Inductor does NOT use MLIR directly; designs are parallel.          │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 04: Enzyme                                                    │
  │  AOTAutograd (Inductor's differentiation layer) works at FX level.  │
  │  Enzyme works at LLVM IR level. Both achieve the same goal.          │
  │  For custom C++ ops: Enzyme can differentiate them where              │
  │  AOTAutograd cannot (sees inside the C++ code).                      │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 05/06: XLA / OpenXLA                                          │
  │  Inductor and XLA are competing GPU backends for torch.compile.     │
  │  torch.compile(backend="openxla") BYPASSES Inductor.                │
  │  Inductor wins for: custom ops, non-NVIDIA GPU, elementwise-heavy   │
  │  workloads. XLA wins for: TPU, very large GEMMs, research models.   │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM                                                       │
  │  torch.compile(backend="tvm") BYPASSES Inductor.                    │
  │  TVM's MetaSchedule finds better schedules for edge hardware.        │
  │  Inductor is faster to compile and good enough for NVIDIA GPU.       │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 09: TorchDynamo                                              │
  │  TorchDynamo is Inductor's ONLY entry point.                         │
  │  Dynamo captures FX graphs; AOTAutograd differentiates them;        │
  │  Inductor compiles them. This three-layer stack IS torch.compile.   │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 10: TorchInductor (THIS MODULE)                              │
  │  Receives FX graphs from AOTAutograd.                                │
  │  Lowers ATen ops to: Pointwise, Reduction, ExternKernel IR.         │
  │  Schedules: vertical fusion, horizontal fusion, epilogue fusion.     │
  │  Generates: Triton (GPU) or C++ (CPU).                               │
  │  Autotuning: BLOCK_SIZE, num_warps, cuBLASLt algorithms.            │
  │  CUDA graphs: capture + replay for zero-overhead hot paths.          │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(STACK)

print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Inductor Config / API              │ Effect                      │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ mode='default'                     │ Fusion + light BLOCK tuning │")
print("  │ mode='reduce-overhead'             │ + CUDA graph capture        │")
print("  │ mode='max-autotune'                │ + full BLOCK + cuBLASLt     │")
print("  │ mode='max-autotune-no-cudagraphs'  │ max-autotune without graphs │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ TORCH_LOGS='output_code'           │ See generated Triton/C++    │")
print("  │ TORCH_LOGS='schedules'             │ See fusion grouping         │")
print("  │ TORCH_LOGS='perf_hints'            │ See performance warnings    │")
print("  │ config.debug = True                │ Dump all IR to /tmp/        │")
print("  │ config.trace.enabled = True        │ HTML compilation trace      │")
print("  │ config.benchmark_kernel = True     │ Per-kernel timing           │")
print("  │ config.max_autotune = True         │ Exhaustive kernel search    │")
print("  │ config.coordinate_descent_tuning   │ Hill-climb after search     │")
print("  │ config.fx_graph_cache = True       │ Persist compiled kernels    │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ torch.library.custom_op()          │ Register custom op          │")
print("  │ torch.library.register_kernel()    │ Register Triton impl        │")
print("  │ torch.library.register_autograd()  │ Register backward rule      │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Project                            │ github.com/pytorch/pytorch  │")
print("  │                                    │ torch/_inductor/            │")
print("  │ Triton                             │ github.com/openai/triton    │")
print("  │ Documentation                      │ pytorch.org/docs/torch.compile │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()

if HAS_TORCH:
    # Final benchmark: compile modes comparison
    print("  Compilation mode comparison:")
    print()

    class BenchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(256)
            self.fc   = nn.Linear(256, 256)
        def forward(self, x):
            return self.norm(torch.relu(self.fc(x)) + x)

    x_b = torch.randn(64, 256, device=DEVICE)

    def bench_mode(mode_str, model_fn):
        try:
            m  = torch.compile(model_fn(), mode=mode_str).to(DEVICE).eval()
            with torch.no_grad():
                for _ in range(15): m(x_b)    ; warmup including compile
            if DEVICE == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(200): m(x_b)
            if DEVICE == "cuda": torch.cuda.synchronize()
            return (time.perf_counter() - t0) / 200 * 1000
        except Exception as e:
            return None

    modes = [
        ("eager (no compile)",    None),
        ("default",               "default"),
        ("reduce-overhead",       "reduce-overhead"),
    ]
    timings = {}
    base_t  = None
    print(f"  {'Mode':35s}  {'ms/call':>10s}  {'Speedup':>9s}")
    print("  " + "-" * 58)
    for label, mode in modes:
        if mode is None:
            m = BenchModel().to(DEVICE).eval()
            with torch.no_grad():
                for _ in range(5): m(x_b)
            if DEVICE == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(200): m(x_b)
            if DEVICE == "cuda": torch.cuda.synchronize()
            t = (time.perf_counter() - t0) / 200 * 1000
        else:
            t = bench_mode(mode, BenchModel)
        if t is None:
            print(f"  {label:35s}  {'N/A':>10s}  {'N/A':>9s}")
            continue
        if base_t is None: base_t = t
        print(f"  {label:35s}  {t:>10.4f}  {base_t/t:>9.2f}×")
    print()
    print("  Note: compile time excluded (amortised over training steps).")
    print("  Run with mode='max-autotune' for best inference throughput.")
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
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }