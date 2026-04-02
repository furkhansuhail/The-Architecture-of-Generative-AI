"""
MLIR — Multi-Level Intermediate Representation
================================================

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure
project from Google, now part of the LLVM umbrella, that solves a problem
LLVM itself cannot: the gap between high-level machine learning operations
(matrix multiply, convolution, attention) and low-level machine instructions
is too large to bridge in a single IR layer.

Before MLIR, every ML framework built its own bespoke IR stack:
TensorFlow had HLO → XLA → LLVM IR (three custom IRs).
TVM had Relay → TIR → LLVM IR (three more custom IRs).
Each one reinvented the same infrastructure: type systems, pass managers,
pattern rewriting, serialisation. All incompatible with each other.

MLIR's insight: don't build ONE intermediate representation — build a
framework for BUILDING intermediate representations, called "dialects."
Each dialect is a set of operations, types, and constraints at one level
of abstraction. Dialects can be progressively lowered into each other,
from high-level ML semantics all the way down to LLVM IR.

In the connected compiler stack:
    LLVM (previous module)  ← MLIR lowers into LLVM dialect → LLVM IR
    MLIR (this module)      ← the multi-level IR framework itself
    XLA (next module)       ← uses MLIR's StableHLO as its primary IR
    TVM (final module)      ← converges with MLIR via Relay/TOSA dialects

"""

import textwrap
import re

TOPIC_NAME   = "MLIR — Multi-Level Intermediate Representation"
DISPLAY_NAME = "02 · MLIR"
ICON         = "🧱"
SUBTITLE     = "Dialects, Progressive Lowering, and the Modern Compiler Stack"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY MLIR EXISTS: THE PROBLEM WITH ONE-SIZE-FITS-ALL IR

### The Abstraction Gap Problem

    LLVM IR operates at a level that is close to assembly:
        - Individual loads, stores, add instructions
        - Scalar or small SIMD operations
        - No concept of "tensor", "convolution", or "attention head"

    A machine learning graph operates at a completely different level:
        - Operations on N-dimensional tensors (shapes, dtypes, layouts)
        - Fused operations (conv + bias + relu as one thing)
        - Data layout decisions (NCHW vs NHWC for convolutions)
        - Memory planning across an entire model graph

    The gap between these two levels is enormous. Crossing it in one step
    means losing information that could enable optimisations:

        If you lower "conv2d" directly to LLVM IR scalar loops,
        you lose the knowledge that this IS a convolution —
        and therefore can't call cuDNN, can't tile for cache, can't
        pick Winograd vs direct vs FFT-based implementations.

    The solution: MULTIPLE levels of IR, each preserving the right amount
    of information for the optimisations at that level.

### The Pre-MLIR IR Proliferation Problem

    Before MLIR, every project built its own IR stack from scratch:

    TensorFlow:    TF Graph → HLO → XLA IR → LLVM IR
    TVM:           Relay → TE → TIR → LLVM IR
    ONNX Runtime:  ONNX graph → internal IR → code generators
    Halide:        Halide func → schedule → LLVM IR
    Torch:         ATen graph → TorchScript → JIT → LLVM IR

    Each project:
        - Built its own type system (how to represent tensor shapes/dtypes)
        - Built its own pass infrastructure (how to chain transformations)
        - Built its own pattern matching engine (how to find and rewrite ops)
        - Built its own serialisation format (how to save/load the IR)

    These IRs were ALL INCOMPATIBLE. A pattern matching pass written for
    TVM could not be reused in XLA. A type checker written for HLO could
    not be reused in Relay.

    Estimated cost: thousands of engineer-years of duplicated effort across
    the industry — all building the same infrastructure with slight variations.

### MLIR's Proposal: A Framework for IRs

    MLIR doesn't define ONE IR. It defines the INFRASTRUCTURE for building IRs:

        ┌─────────────────────────────────────────────────────────────┐
        │  MLIR provides:                                              │
        │  • A common type system (extensible by dialects)             │
        │  • A common operation format (with verification hooks)       │
        │  • A pass manager (reusable for ANY dialect)                 │
        │  • A pattern rewriting engine (works on any op)              │
        │  • A serialisation format (textual + binary)                 │
        │  • A testing infrastructure (filecheck-based)                │
        │  • Traits and interfaces (structural contracts on ops)        │
        │                                                              │
        │  Each "dialect" defines:                                     │
        │  • A namespace of operations (e.g., linalg.matmul)          │
        │  • Types specific to that level (e.g., memref<4x4xf32>)     │
        │  • Lowering passes to other dialects                          │
        └─────────────────────────────────────────────────────────────┘

    The key insight: infrastructure is shared, domain knowledge is pluggable.
    A pass written for one dialect can be applied to any operation that
    implements the right interface, regardless of which dialect it belongs to.


##### PART 2 — DIALECTS: MLIR'S CORE ABSTRACTION

### What a Dialect Is

    A dialect is a namespace of operations, types, and attributes that
    collectively represent one level of abstraction or one domain.

    Every operation in MLIR lives in a dialect:
        tf.MatMul             — TensorFlow dialect (highest level)
        linalg.matmul         — Linalg dialect (algebraic, shape-aware)
        affine.for            — Affine dialect (loop with affine bounds)
        memref.load           — MemRef dialect (explicit memory access)
        vector.outerproduct   — Vector dialect (SIMD-like vectors)
        llvm.mlir.alloca      — LLVM dialect (direct LLVM IR in MLIR)
        arith.addf            — Arithmetic dialect (basic scalar math)

    A single MLIR program can contain operations from multiple dialects
    simultaneously — this is the "multi-level" in MLIR's name.

### The Dialect Hierarchy for ML

    High level (semantics-rich, hardware-agnostic):
        TF dialect:       tf.Conv2D, tf.MatMul, tf.Relu
        TOSA:             Tensor Operator Set Architecture (cross-framework standard)
        StableHLO:        Stable version of XLA's HLO ops (XLA's primary IR)
        Torch dialect:    torch.aten.conv2d (PyTorch ATen ops)

    Mid level (shape-aware, memory-layout-agnostic):
        Linalg dialect:   linalg.matmul, linalg.conv_2d_nhwc_hwcf
                          Generic named ops with operand regions
        Tensor dialect:   tensor.extract_slice, tensor.insert_slice
        Bufferization:    Planning how tensors map to memory buffers

    Low level (loop and memory explicit):
        Affine dialect:   affine.for, affine.load, affine.store
                          Loops with affine (linear + constant) bounds
                          Enables polyhedral analysis and tiling
        SCF dialect:      scf.for, scf.if, scf.parallel
                          Structured control flow (still higher than LLVM)
        MemRef dialect:   memref.alloc, memref.load, memref.store
                          Explicit N-dimensional memory references

    Vector level:
        Vector dialect:   vector.contract, vector.transfer_read/write
                          Target-agnostic SIMD operations
                          Maps to SSE/AVX/NEON/SVE instructions

    Hardware-specific level:
        LLVM dialect:     llvm.mlir.alloca, llvm.call — direct LLVM IR ops
        NVVM dialect:     NVIDIA GPU-specific ops (warp sync, shared mem)
        ROCDL dialect:    AMD GPU-specific ops
        GPU dialect:      target-agnostic GPU ops (launch, barrier)

    ┌────────────────────────────────────────────────────────────────┐
    │  tf.Conv2D  →  linalg.conv  →  affine.for loops  →  vector.*  │
    │           →  memref.*  →  llvm.*  →  LLVM IR  →  x86/ARM     │
    └────────────────────────────────────────────────────────────────┘

### MLIR Operation Syntax

    Every MLIR operation follows the same syntax:

        %result = dialect.operation %operand1, %operand2 : type_signature

    Real examples:

        ; Linalg matmul: C += A @ B
        linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>)
                       outs(%C     : memref<4x4xf32>)

        ; Affine for loop: for i = 0 to 10 step 1
        affine.for %i = 0 to 10 {
            affine.store %val, %buf[%i] : memref<10xf32>
        }

        ; Vector transfer: load a 4-element vector from memref
        %v = vector.transfer_read %buf[%i] : memref<?xf32>, vector<4xf32>

        ; Arithmetic: floating point add
        %r = arith.addf %a, %b : f32

        ; Function definition
        func.func @relu(%x: f32) -> f32 {
            %zero = arith.constant 0.0 : f32
            %r    = arith.maxf %x, %zero : f32
            func.return %r : f32
        }

### MLIR Types

    MLIR has a rich, extensible type system:

    Scalar types:
        i1, i8, i16, i32, i64    — integers
        f16, bf16, f32, f64      — floats (bf16 is first-class!)
        index                     — used for loop variables and sizes

    Tensor types (value semantics — immutable, no aliasing):
        tensor<4x4xf32>           — static 4×4 float tensor
        tensor<?x?xf32>           — dynamic shape (? = unknown at compile time)
        tensor<4x?x8xf32>         — mixed static/dynamic

    MemRef types (buffer semantics — mutable, explicit memory):
        memref<4x4xf32>           — row-major 4×4 float buffer
        memref<4x4xf32, strided<[4,1]>>    — explicit strides (any layout)
        memref<4x4xf32, 1>        — memory space 1 (e.g., GPU shared memory)
        memref<?x?xf32>           — dynamic shape buffer

    Vector types:
        vector<4xf32>             — 4-element float vector (fits in SSE)
        vector<8xf32>             — 8-element float vector (AVX2)
        vector<4x4xf32>           — 2D vector tile (matrix register)

    The tensor/memref distinction is crucial:
        Tensors:  immutable values, like in functional programming.
                  Safe to reason about algebraically. No aliasing.
        MemRefs:  mutable buffers with explicit layout. Like C arrays.
                  Needed for code generation and memory planning.

    Bufferisation converts tensors to memrefs.


##### PART 3 — PROGRESSIVE LOWERING: THE COMPILATION STRATEGY

### What Progressive Lowering Means

    Instead of one big transformation (ML graph → machine code), MLIR
    advocates many SMALL, well-defined transformations between adjacent
    dialect levels.

    Each lowering step:
        - Is a well-tested, reusable pass
        - Preserves semantics (verifiable with MLIR's verifier)
        - Narrows the abstraction gap by one level only

    This is the compiler-engineering principle of "separation of concerns"
    applied to IR design.

### The ML Compilation Lowering Chain

    Level 5: Framework ops (Python-level ML)
        tf.Relu(x)
        torch.relu(x)

    Level 4: High-level ML ops (framework-neutral)
        tosa.clamp(x, 0, +inf)         — TOSA dialect
        stablehlo.maximum(x, zeros)    — StableHLO dialect
              ↓ --convert-to-linalg
    Level 3: Named algebraic ops (shape-aware, layout-free)
        linalg.generic {map=identity} ins(%x) outs(%result)
              ↓ --linalg-bufferize
    Level 3b: Bufferised (tensor → memref)
        linalg.generic ... ins(%x: memref) outs(%result: memref)
              ↓ --convert-linalg-to-affine
    Level 2: Affine loops (polyhedral representation)
        affine.for %i = 0 to %n {
            %xi = affine.load %x[%i]
            %ri = affine.apply max(%xi, 0)
            affine.store %ri, %result[%i]
        }
              ↓ --lower-affine
    Level 1: SCF + MemRef (structured control flow)
        scf.for %i = 0 to %n step 1 {
            %xi = memref.load %x[%i]
            %ri = arith.maxf %xi, %zero
            memref.store %ri, %result[%i]
        }
              ↓ --convert-scf-to-cf  --convert-arith-to-llvm
    Level 0: LLVM dialect (direct LLVM IR in MLIR form)
        llvm.mlir.alloca ...
        llvm.load ...
        llvm.call @llvm.maxnum.f32 ...
        llvm.store ...
              ↓ --mlir-translate --mlir-cpu-runner
    LLVM IR → LLVM passes → native machine code

### Why This Strategy Is Powerful

    At each level, passes can exploit level-specific information:

    Level 4 (ML ops): Fuse conv + bias + relu into one operation.
        The fusion pass knows these are ML ops with known semantics.
        It cannot work at Level 0 (already lowered to scalar loops).

    Level 3 (Linalg): Tile the matmul for cache efficiency.
        The tiling pass knows this is a matrix multiply.
        It knows which loops correspond to M, N, K dimensions.

    Level 2 (Affine): Apply polyhedral transformations (loop interchange,
        skewing, parallelisation). Possible because affine loops have
        mathematically analysable bounds.

    Level 1 (SCF): Generate OpenMP parallel loops.
        Mark loops with omp.parallel for CPU parallelism.

    Level 0 (LLVM): Register allocation, SIMD, branch prediction.
        LLVM handles all hardware-specific details.

    Each level "owns" its optimisations. Lower levels can't recover
    information that was lost at upper levels — this is why the
    ordering matters enormously.


##### PART 4 — THE LINALG DIALECT: MLIR'S HEART FOR ML

### What Linalg Is

    The Linalg (Linear Algebra) dialect is MLIR's primary target for ML
    operations. It was designed specifically for expressing tensor
    computations in a form that enables tiling, vectorisation, and fusion.

    Linalg operations are defined by:
        1. INDEXING MAPS:   how output indices map to input indices (math)
        2. ITERATOR TYPES:  parallel (independent) vs reduction (accumulated)
        3. A REGION:        the scalar computation per output element

    This mathematical specification is what makes transformations provably
    correct — the compiler can reason about loop reordering, tiling,
    and fusion from the indexing maps alone.

### Named Linalg Ops

    For common operations, MLIR provides named ops:
        linalg.matmul:    matrix multiplication C += A @ B
        linalg.matvec:    matrix-vector product y += A @ x
        linalg.dot:       dot product acc += sum(a * b)
        linalg.fill:      initialise a tensor with a value
        linalg.conv_2d_nhwc_hwcf:  2D convolution (NHWC input, HWCF filter)
        linalg.pool_nhwc_max:      max pooling
        linalg.batch_matmul:       batched matrix multiplication

### The Generic Op — Universal Computation

    linalg.generic expresses ANY elementwise-like operation:

        ; ElementWise add: C[i,j] = A[i,j] + B[i,j]
        linalg.generic
            {indexing_maps = [affine_map<(i,j)->(i,j)>,   ; A indexed by (i,j)
                              affine_map<(i,j)->(i,j)>,   ; B indexed by (i,j)
                              affine_map<(i,j)->(i,j)>],  ; C indexed by (i,j)
             iterator_types = ["parallel", "parallel"]}    ; both loops are parallel
            ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
            outs(%C    : tensor<4x4xf32>) {
          ^bb0(%a: f32, %b: f32, %c: f32):
            %sum = arith.addf %a, %b : f32
            linalg.yield %sum : f32             ; scalar body: one output per element
        } -> tensor<4x4xf32>

        ; Matrix multiply: C[i,k] += A[i,j] * B[j,k]
        linalg.generic
            {indexing_maps = [affine_map<(i,j,k)->(i,j)>,   ; A[i,j]
                              affine_map<(i,j,k)->(j,k)>,   ; B[j,k]
                              affine_map<(i,j,k)->(i,k)>],  ; C[i,k]
             iterator_types = ["parallel", "reduction", "parallel"]}
             ;                    i            j (sum)       k
            ins(%A, %B : tensor<MxKxf32>, tensor<KxNxf32>)
            outs(%C    : tensor<MxNxf32>) {
          ^bb0(%a: f32, %b: f32, %c_acc: f32):
            %prod  = arith.mulf %a, %b : f32
            %accum = arith.addf %c_acc, %prod : f32
            linalg.yield %accum : f32
        } -> tensor<MxNxf32>

    The indexing maps COMPLETELY DESCRIBE the access patterns.
    From these, the compiler knows:
        - Which dimensions are independent (can parallelise/vectorise)
        - Which dimensions are reductions (must accumulate)
        - Where tiling is safe (any parallel dimension)
        - What fusion is possible (producer's output = consumer's input)

### Linalg Transformations

    Because operations are defined by indexing maps, these transformations
    are GENERIC and work on ANY linalg.generic op:

    linalg.tiling:
        Splits a loop nest into tiles for cache efficiency.
        tile_sizes = [8, 8, 0] means tile i×j in 8×8 blocks, don't tile k.

    linalg.vectorize:
        Maps the inner loop body to vector operations.
        If loop body is f32 add and target has AVX2: emit <8 x float> add.

    linalg.fusion:
        If producer's output feeds directly into consumer's input,
        eliminate the intermediate tensor — compute both together.
        conv + bias + relu → single fused loop nest.

    linalg.parallelise:
        Mark parallel iterator loops as parallel_for (for OpenMP or GPU).


##### PART 5 — THE PASS MANAGER AND PATTERN REWRITING

### MLIR's Pass Manager

    MLIR's pass manager is shared infrastructure across ALL dialects.
    Any pass written for any dialect can be managed by the same framework:

        Types of passes:
            FunctionPass:   operates on one function at a time
            ModulePass:     operates on the whole module
            OperationPass:  operates on any arbitrary op

        Pass composition (nested pass managers):
            pm.addPass(createConvertLinalgToAffineLoopsPass())
            pm.addNestedPass<func::FuncOp>(createAffineVectorizePass())
            pm.addPass(createLowerVectorToLLVMPass())
            pm.run(module)

    The key advantage: ONE pass manager for the entire lowering pipeline,
    from TF graph to LLVM IR. No per-dialect infrastructure to maintain.

### Pattern Rewriting — The Workhorse of MLIR

    Most MLIR passes are implemented as pattern rewriters:
        A pattern matches a SPECIFIC op (or op combination).
        If matched, the pattern replaces it with semantically equivalent ops.

    Pattern anatomy:
        struct ReLUToLinalg : public OpRewritePattern<tosa::ClampOp> {
            LogicalResult matchAndRewrite(tosa::ClampOp op,
                                           PatternRewriter &rewriter) const {
                // 1. Check: is this a relu (clamp with min=0, max=inf)?
                if (op.getMinFp() != 0.0 || op.hasFiniteMax())
                    return failure();   // pattern doesn't match

                // 2. Rewrite: replace with linalg.generic relu
                auto result = rewriter.create<linalg::GenericOp>(...);
                rewriter.replaceOp(op, result);
                return success();
            }
        };

    Pattern sets are combined and applied with:
        RewritePatternSet patterns(ctx);
        patterns.add<ReLUToLinalg, MatMulToLinalg, ...>(ctx);
        applyPatternsAndFoldGreedily(module, std::move(patterns));

    The greedy driver applies patterns repeatedly until convergence.
    This is how multi-step lowering is implemented: each pass applies
    one set of patterns that lowers ops one level.

### Interfaces — Structural Contracts

    Interfaces are abstract contracts that ops can implement:

        LinalgOp interface:   provides indexing maps, iterator types.
                              ANY op implementing this can be tiled/vectorised.

        MemoryEffects interface: declares reads/writes to memory.
                                  Enables dead code elimination, alias analysis.

        CallOpInterface:    marks ops that perform calls.
                            Enables inlining regardless of which dialect.

        RegionBranchOpInterface: marks ops with region-based control flow.
                                  Enables generic analyses across dialects.

    Interfaces decouple pass logic from specific ops:
        A tiling pass that operates on the LinalgOp interface can tile
        linalg.matmul, linalg.conv, linalg.generic, AND any future op
        that implements the interface — without changes to the tiling pass.

    This is MLIR's primary extensibility mechanism.


##### PART 6 — STABLEHLO AND MHLO: THE XLA CONNECTION

### What StableHLO Is

    HLO (High Level Operations) is XLA's computation IR — the language that
    describes a TensorFlow or JAX computation to the XLA compiler.

    MHLO (Meta HLO) was the first attempt to express HLO as an MLIR dialect.
    StableHLO is its stabilised successor — a compatibility layer between
    ML frameworks and XLA/compiler backends.

    StableHLO defines ~100 operations matching XLA's HLO:
        stablehlo.add, stablehlo.multiply, stablehlo.dot_general
        stablehlo.convolution, stablehlo.reduce, stablehlo.pad
        stablehlo.reshape, stablehlo.transpose, stablehlo.while

    StableHLO's role in the stack:
        JAX → JAX lowering → StableHLO → XLA → LLVM/GPU code
        PyTorch → torch-mlir → StableHLO → XLA/IREE → code

    Stability guarantee: StableHLO ops from 5 years ago still work.
    This portability is critical for model serialisation.

### torch-mlir: PyTorch → MLIR

    torch-mlir is an official project bringing PyTorch into the MLIR stack:

        PyTorch model → TorchScript/FX → Torch MLIR dialect
            → TOSA dialect (or StableHLO)
                → Linalg dialect
                    → LLVM dialect → native code (via IREE or LLVM)

    The Torch dialect has direct mappings for ATen ops:
        torch.aten.relu → tosa.clamp or stablehlo.maximum
        torch.aten.matmul → linalg.matmul
        torch.aten.conv2d → linalg.conv_2d_nhwc_hwcf

### IREE: MLIR-Native Deployment Runtime

    IREE (Intermediate Representation Execution Environment) is a compiler
    and runtime built entirely on MLIR for deploying ML models:

        Compilation: StableHLO → Linalg → Spirv/LLVM → native code
        Runtime:     loads compiled artifacts, manages execution

    IREE targets: CPU (x86, ARM, RISC-V), GPU (Vulkan, Metal, CUDA, ROCm).
    Unlike TensorFlow or PyTorch, IREE has NO Python runtime dependency —
    just a compiled artifact and a minimal C runtime.

    ┌────────────────────────────────────────────────────────────────────┐
    │  ML Framework → torch-mlir/tf-mlir → StableHLO →                 │
    │  MLIR Linalg → IREE compiler → HAL (Hardware Abstraction Layer) → │
    │  Vulkan/Metal/CUDA dispatch → GPU/CPU execution                   │
    └────────────────────────────────────────────────────────────────────┘


##### PART 7 — THE AFFINE DIALECT: POLYHEDRAL TRANSFORMATIONS

### What the Affine Dialect Provides

    The Affine dialect expresses loops where bounds and memory accesses
    are AFFINE FUNCTIONS of loop variables:
        affine function: a linear combination + constant offset.
        Example: a[2*i + j + 3] — affine (linear in i and j)
        Counterexample: a[i*j] — NOT affine (product of variables)

    Why affine functions? Because polyhedral mathematics can analyse them:
        - When is loop interchange safe? (check affine dependencies)
        - How to tile for cache? (find affine hyperplanes)
        - Which iterations are independent? (solve affine integer programs)

### Key Affine Operations

    affine.for %i = 0 to 64 step 4 {        ; loop with constant bounds
        affine.for %j = 0 to %n {            ; loop with symbolic bound
            %val = affine.load %A[%i, %j]    ; load at affine index
            ...
            affine.store %result, %B[%j, %i] ; store at transposed index
        }
    }

    affine.apply:
        %idx = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%i, %j)
        ; Computes i*8 + j — an affine expression over %i and %j

    affine.if (polyhedral region guard):
        affine.if affine_set<(d0, d1) : (d0 - d1 >= 0, d1 >= 0)>(%i, %j) {
            ; only executes when i >= j >= 0 (lower triangle)
        }

### Polyhedral Transformations on Affine Loops

    Loop tiling (for cache efficiency):
        Before: affine.for %i = 0 to M { affine.for %j = 0 to N { ... }}
        After tiling by 32:
            affine.for %i0 = 0 to M step 32 {    ; outer tile loop i
              affine.for %j0 = 0 to N step 32 {  ; outer tile loop j
                affine.for %i = %i0 to min(%i0+32, M) {   ; inner
                  affine.for %j = %j0 to min(%j0+32, N) {
                    ...

    Loop interchange (reorder for better access pattern):
        matrix A[i][j]: row-major, access A[i][j] in i-then-j is good
        matrix B[j][k]: for matmul, access B[j][k] in j-then-k is good
        Interchange j and k loops if the dependency analysis allows it.

    These transformations are the theoretical basis for:
        - Cache-efficient matrix multiplication
        - Conv2D as GEMM (image-to-column transformation)
        - Memory access pattern optimisation in TVM schedules


##### PART 8 — MLIR IN THE CONNECTED COMPILER STACK

### How MLIR Connects to LLVM (previous module)

    LLVM IR is the final target for MLIR's CPU lowering path:

        MLIR's LLVM dialect is a 1:1 mapping of LLVM IR into MLIR operations.
        linalg.matmul → (tiling, vectorise, bufferise) → llvm.* operations
        mlir-translate --mlir-to-llvmir converts LLVM dialect → LLVM IR text
        Then the standard LLVM pipeline (opt, llc) takes over.

    MLIR uses LLVM's:
        - JIT compiler (ORC JIT) for on-demand execution
        - Target machine for code emission
        - Intrinsics for FMA, maxnum, vector operations
        - Pass manager (for the final LLVM-level optimisations)

    MLIR adds above LLVM:
        - Multi-level IR with dialect extensibility
        - ML-specific operations (tensor, linalg, tosa, stablehlo)
        - Polyhedral analysis (affine dialect)
        - Graph-level fusion and layout transformations

### How MLIR Connects to XLA (next module)

    XLA is increasingly adopting MLIR as its internal IR:

    Old XLA: TF/JAX → HLO proto → XLA's internal graph passes → LLVM/GPU
    New XLA: TF/JAX → StableHLO → MLIR passes (in MLIR infrastructure) → GPU

    The XLA team has:
        - Defined MHLO/StableHLO as MLIR dialects
        - Migrated XLA's core passes to MLIR's pass manager
        - Use MLIR's pattern rewriting for HLO algebraic simplifications
        - Use MLIR's affine/linalg dialects for loop optimisations

    StableHLO is now the portable serialisation format between:
        JAX → XLA, PyTorch → IREE, TF → XLA, any framework → any compiler

### How MLIR Connects to TVM (final module)

    TVM and MLIR are converging but remain architecturally distinct:

    TVM path:    model → Relay IR → TIR → TVM schedules → LLVM
    MLIR path:   model → Linalg → Affine → LLVM

    Convergence:
        - TVM added an MLIR import path (model → Relay → StableHLO → TVM)
        - TOSA dialect (in MLIR) overlaps with TVM's operator set
        - Both use LLVM as the final backend for CPU targets
        - TVM's TE schedules are conceptually similar to MLIR's Linalg transforms

    Key difference:
        TVM has AutoTVM/Ansor for auto-scheduling (search-based optimisation).
        MLIR relies on deterministic analytical transforms.
        These approaches are complementary, not competing.

### The Full ML Compiler Stack (Connected View)

    ┌─────────────────────────────────────────────────────────────────┐
    │  Python model (PyTorch / JAX / TF / ONNX)                      │
    ├─────────────────────────────────────────────────────────────────┤
    │  Framework IR (torch.fx / HLO / ONNX graph)                    │
    ├─────────────────────────────────────────────────────────────────┤
    │  StableHLO / MHLO / TOSA   ← MLIR HIGH-LEVEL DIALECTS          │
    │  Linalg dialect            ← MLIR MID-LEVEL (algebraic)        │
    │  Affine + SCF + MemRef     ← MLIR LOW-LEVEL (loop + memory)    │
    │  Vector dialect            ← MLIR (SIMD abstraction)           │
    │  LLVM dialect              ← MLIR BRIDGE TO LLVM               │
    ├─────────────────────────────────────────────────────────────────┤
    │  LLVM IR  (previous module)                                     │
    │  LLVM pass pipeline (instcombine, loop-vectorize, ...)          │
    ├─────────────────────────────────────────────────────────────────┤
    │  Target code: x86 / ARM / RISC-V / NVPTX / AMDGCN              │
    └─────────────────────────────────────────────────────────────────┘

    XLA lives at the StableHLO → GPU code generation layer.
    TVM lives at the Linalg/TIR → LLVM/CUDA layer.
    MLIR is the INFRASTRUCTURE that makes all of this composable.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · MLIR Dialects & IR — Reading, Writing and Progressive Lowering": {
        "description": (
            "Hands-on MLIR IR. Read and understand the textual format. "
            "Trace a ReLU and matmul from TOSA → Linalg → Affine → LLVM dialect. "
            "Show the type system: tensor vs memref, static vs dynamic shapes. "
            "Use the mlir Python bindings (mlir-core) to construct IR programmatically. "
            "Connect to LLVM: show how the LLVM dialect bridges to LLVM IR."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  MLIR DIALECTS & IR — READING, WRITING AND LOWERING")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: MLIR IR textual format — annotated examples
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — MLIR IR textual format guide")
print("━" * 65)
print()

MLIR_FORMAT = """
  MLIR SYNTAX GUIDE
  ════════════════════════════════════════════════════════════════

  BASIC STRUCTURE:
    Every MLIR program is a module containing functions and operations.
    Operations follow: %result = dialect.opname args : types

  TYPES:
    i32, i64           integer
    f32, f64, bf16     float (bf16 is first-class in MLIR!)
    index              loop/size variable (target word size)
    tensor<4x4xf32>    immutable 4x4 float tensor (value semantics)
    tensor<?x?xf32>    dynamic shape (? = unknown at compile time)
    memref<4x4xf32>    mutable buffer (row-major by default)
    memref<4x4xf32, strided<[8,1], offset:0>>   explicit layout
    vector<8xf32>      8-element SIMD vector
    vector<4x8xf32>    2D SIMD tile (for matrix register files)

  REGIONS AND BLOCKS:
    Operations can contain regions (lambdas / loop bodies).
    Each region has basic blocks labelled with ^bb0, ^bb1, etc.
    Basic block arguments are the SSA values entering that block.

  ATTRIBUTES:
    dense<[[1,2],[3,4]]> : tensor<2x2xi32>    constant tensor literal
    affine_map<(d0,d1)->(d0,d1)>             affine map (access pattern)
    affine_set<(d0,d1): (d0>=0, d1>=0)>      polyhedral constraint set
    #gpu.address_space<workgroup>             enum-like named attribute
"""
print(MLIR_FORMAT)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: ReLU — full lowering from TOSA to LLVM dialect
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — ReLU: progressive lowering TOSA → LLVM dialect")
print("━" * 65)
print()

RELU_LOWERING = """
  ── LEVEL 4: TOSA dialect (framework-neutral ML op) ──────────────────

  func.func @relu(%x: tensor<8xf32>) -> tensor<8xf32> {
    %zero = "tosa.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %result = "tosa.clamp"(%x) {
        min_fp = 0.000000e+00 : f32,
        max_fp = 3.4028235e+38 : f32    ; FLT_MAX = no upper clamp
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }

  ── LEVEL 3a: Linalg dialect (after --tosa-to-linalg) ────────────────

  func.func @relu(%x: tensor<8xf32>) -> tensor<8xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<8xf32>          ; output buffer
    %result = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>, ; input: A[d0]
                          affine_map<(d0) -> (d0)>], ; output: B[d0]
         iterator_types = ["parallel"]}              ; 1 parallel loop
        ins(%x    : tensor<8xf32>)
        outs(%init : tensor<8xf32>) {
      ^bb0(%xi: f32, %_: f32):
        %relu_xi = arith.maxf %xi, %zero : f32       ; scalar body
        linalg.yield %relu_xi : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }

  ── LEVEL 3b: After bufferisation (tensor → memref) ──────────────────

  func.func @relu(%x: memref<8xf32>, %out: memref<8xf32>) {
    %zero = arith.constant 0.0 : f32
    linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
        ins(%x   : memref<8xf32>)
        outs(%out : memref<8xf32>) {
      ^bb0(%xi: f32, %_: f32):
        %r = arith.maxf %xi, %zero : f32
        linalg.yield %r : f32
    }
    return
  }

  ── LEVEL 2: Affine dialect (after --convert-linalg-to-affine-loops) ──

  func.func @relu(%x: memref<8xf32>, %out: memref<8xf32>) {
    %zero = arith.constant 0.0 : f32
    affine.for %i = 0 to 8 {                         ; explicit loop
      %xi = affine.load %x[%i]    : memref<8xf32>    ; load input
      %r  = arith.maxf %xi, %zero : f32              ; compute relu
      affine.store %r, %out[%i]  : memref<8xf32>     ; store output
    }
    return
  }
  ; At this level: polyhedral analysis can prove loop is independent
  ; → safe to vectorise all 8 iterations as <8 x float> maxf

  ── LEVEL 1: SCF (after --lower-affine) ──────────────────────────────

  func.func @relu(%x: memref<8xf32>, %out: memref<8xf32>) {
    %zero = arith.constant 0.0 : f32
    %c0   = arith.constant 0   : index
    %c8   = arith.constant 8   : index
    %c1   = arith.constant 1   : index
    scf.for %i = %c0 to %c8 step %c1 {
      %xi = memref.load  %x[%i]   : memref<8xf32>
      %r  = arith.maxf   %xi, %zero : f32
      memref.store %r, %out[%i]    : memref<8xf32>
    }
    return
  }

  ── LEVEL 0: LLVM dialect (after --convert-to-llvm) ──────────────────

  llvm.func @relu(%x_ptr: !llvm.ptr, %out_ptr: !llvm.ptr) {
    %zero  = llvm.mlir.constant(0.0 : f32) : f32
    %c0    = llvm.mlir.constant(0 : i64)   : i64
    %c8    = llvm.mlir.constant(8 : i64)   : i64
    %c1    = llvm.mlir.constant(1 : i64)   : i64
    llvm.br ^loop(%c0 : i64)
  ^loop(%i : i64):
    %done  = llvm.icmp "eq" %i, %c8 : i64
    llvm.cond_br %done, ^exit, ^body
  ^body:
    %gep_x   = llvm.getelementptr %x_ptr[%i]   : (!llvm.ptr, i64) -> !llvm.ptr
    %gep_out = llvm.getelementptr %out_ptr[%i]  : (!llvm.ptr, i64) -> !llvm.ptr
    %xi      = llvm.load %gep_x   : !llvm.ptr -> f32
    %r       = llvm.intr.maxnum(%xi, %zero) : (f32, f32) -> f32   ; maxnum intrinsic
    llvm.store %r, %gep_out       : f32, !llvm.ptr
    %i_next  = llvm.add %i, %c1  : i64
    llvm.br ^loop(%i_next : i64)
  ^exit:
    llvm.return
  }
  ; → mlir-translate converts this to LLVM IR text
  ; → LLVM pipeline (loop-vectorize) converts scalar to <8 x float> ops
"""

print(RELU_LOWERING)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Matmul — linalg.generic with indexing maps
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — MatMul: linalg.generic indexing maps explained")
print("━" * 65)
print()

MATMUL_LINALG = """
  Matrix multiply C[M,N] += A[M,K] * B[K,N]
  expressed as linalg.generic with 3 index dimensions: (i, j, k)

  linalg.generic {
    indexing_maps = [
      affine_map<(i,j,k) -> (i,k)>,   ; A: row i, col k
      affine_map<(i,j,k) -> (k,j)>,   ; B: row k, col j
      affine_map<(i,j,k) -> (i,j)>    ; C: row i, col j (output)
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
    ;                     i           j         k (sum over k)
  }

  What the indexing maps tell the compiler:
    ┌──────────────────────────────────────────────────────────────┐
    │ Dimension │ Type      │ Meaning                              │
    ├──────────────────────────────────────────────────────────────┤
    │ i         │ parallel  │ rows of A,C — independent, vectorise │
    │ j         │ parallel  │ cols of B,C — independent, vectorise │
    │ k         │ reduction │ inner dim — must accumulate, no tile │
    └──────────────────────────────────────────────────────────────┘

  From these maps, the compiler derives:
    Safe to tile: i, j (parallel dimensions)
    Safe to vectorise: i or j inner loop (parallel)
    Must accumulate: k loop (reduction → initialise C to 0 before loop)
    Fusion: if C feeds directly into another linalg op → fuse loops

  After tiling (tile_sizes = [8, 8, 0]):
    affine.for %i0 = 0 to M step 8 {
      affine.for %j0 = 0 to N step 8 {
        linalg.matmul               ; operates on 8×8 tile of C
            ins(%A_slice, %B_slice)
            outs(%C_slice)
        ; tile stays in L1 cache during inner k loop
      }
    }

  After vectorisation (vector_size = 8 on j-dimension):
    affine.for %i = 0 to M {
      affine.for %k = 0 to K {
        %a_ik = affine.load %A[%i,%k]     ; scalar
        %b_k  = vector.transfer_read %B[%k, 0], ...  ; load <8 x f32> row of B
        %c    = vector.transfer_read %C[%i, 0], ...  ; load <8 x f32> of C
        %new_c = vector.fma %a_ik_splat, %b_k, %c   ; fused multiply-add
        vector.transfer_write %new_c, %C[%i, 0]
      }
    }
  ; vector.fma → llvm.intr.fma → x86 vfmadd231ps (AVX2 FMA instruction)
"""
print(MATMUL_LINALG)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Python MLIR bindings
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Python MLIR bindings (mlir-core)")
print("━" * 65)
print()

try:
    from mlir.ir import Context, Module, InsertionPoint, Location
    from mlir.ir import F32Type, RankedTensorType, IndexType
    from mlir.dialects import func, arith, linalg, tensor
    import mlir

    print(f"  mlir-core available ✅  (version: {mlir.__version__ if hasattr(mlir, '__version__') else 'installed'})")
    print()

    with Context() as ctx, Location.unknown():
        # Create a module
        module = Module.create()
        f32    = F32Type.get()
        tensor_8xf32 = RankedTensorType.get([8], f32)

        with InsertionPoint(module.body):
            # Define function: relu(tensor<8xf32>) -> tensor<8xf32>
            @func.func_op(
                name="relu_tensor",
                input_types=[tensor_8xf32],
                result_types=[tensor_8xf32],
            )
            def relu_tensor(x):
                # arith.constant for zero
                zero = arith.ConstantOp(f32, 0.0).result
                # maxf elementwise (simplification — real impl uses linalg.generic)
                result = arith.MaxFOp(x, x).result   # placeholder
                func.ReturnOp([result])

        print("  Generated MLIR module:")
        print()
        for line in str(module).split('\n'):
            print(f"    {line}")

except ImportError:
    print("  mlir-core not installed.")
    print("  Install: pip install mlir-core   (or  pip install mlir-python-bindings)")
    print()

    MLIR_PYTHON_CODE = """
  Python MLIR bindings example:

  from mlir.ir import Context, Module, InsertionPoint, Location
  from mlir.ir import F32Type, RankedTensorType, IntegerType
  from mlir.dialects import func, arith, linalg

  with Context() as ctx, Location.unknown():
      ctx.enable_multithreading(False)
      module = Module.create()

      f32    = F32Type.get()
      i32    = IntegerType.get_signless(32)
      t_4xf32 = RankedTensorType.get([4], f32)

      with InsertionPoint(module.body):
          # Define @dot(a: tensor<4xf32>, b: tensor<4xf32>) -> f32
          @func.func_op("dot", [t_4xf32, t_4xf32], [f32])
          def dot(a, b):
              zero = arith.ConstantOp(f32, 0.0).result
              # linalg.dot computes sum(a*b) into acc
              init = tensor.EmptyOp([], f32).result   ; scalar output
              result = linalg.DotOp(
                  result_tensors=[f32],
                  inputs=[a, b],
                  outputs=[init],
              ).result
              func.ReturnOp([result])

      # Print the generated IR
      print(module)
      # Output:
      # module {
      #   func.func @dot(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> f32 {
      #     %cst  = arith.constant 0.000000e+00 : f32
      #     %0    = tensor.empty() : tensor<f32>
      #     %1    = linalg.dot ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
      #                        outs(%0 : tensor<f32>) -> f32
      #     return %1 : f32
      #   }
      # }

  # Run the generated IR through passes:
  from mlir.passmanager import PassManager
  pm = PassManager.parse("builtin.module(convert-linalg-to-affine-loops,lower-affine)")
  pm.run(module)
  print("After lowering:")
  print(module)
"""
    print(MLIR_PYTHON_CODE)

print()
print("  MLIR TOOLCHAIN (command-line):")
print()
TOOLS = """
  mlir-opt:        apply passes to MLIR IR
    mlir-opt --tosa-to-linalg --linalg-bufferize \\
              --convert-linalg-to-affine-loops \\
              --lower-affine --convert-to-llvm  input.mlir -o output.mlir

  mlir-translate:  convert between MLIR and other formats
    mlir-translate --mlir-to-llvmir output.mlir -o output.ll

  mlir-cpu-runner: JIT-run MLIR on CPU directly
    mlir-cpu-runner output.mlir --entry-point-result=void

  mlir-reduce:     reduce failing test cases
  mlir-lsp-server: language server for MLIR IR files (IDE support)
"""
print(TOOLS)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Pattern Rewriting & Pass Pipeline — MLIR's Transformation Engine": {
        "description": (
            "MLIR's core transformation mechanism. "
            "Pattern matching: how ops are found and replaced. "
            "The greedy rewrite driver. Folding vs rewriting. "
            "Implement a simplified Python-level pattern rewriter "
            "that mirrors MLIR's RewritePattern API. "
            "Trace a real lowering pipeline: TOSA → Linalg → Affine. "
            "Show how interfaces decouple passes from specific ops."
        ),
        "language": "python",
        "code": '''
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from collections import defaultdict

print("=" * 65)
print("  PATTERN REWRITING & PASS PIPELINE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Simulate MLIR's operation and pattern system in Python
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Simulating MLIR's Operation and Pattern model")
print("━" * 65)
print()

# ── Simplified MLIR operation model ───────────────────────────────────

@dataclass
class MLIRType:
    """Simplified MLIR type."""
    name: str
    def __repr__(self): return self.name

@dataclass
class MLIRValue:
    """An SSA value in MLIR."""
    name: str
    type: MLIRType
    def __repr__(self): return f"%{self.name}: {self.type}"

@dataclass
class MLIROp:
    """
    Simplified MLIR operation.
    In real MLIR: every op has a dialect prefix (e.g., tosa.clamp).
    Here we simulate the core properties.
    """
    dialect:    str
    opname:     str
    operands:   List[MLIRValue] = field(default_factory=list)
    results:    List[MLIRValue] = field(default_factory=list)
    attributes: Dict            = field(default_factory=dict)
    regions:    List            = field(default_factory=list)  # nested ops

    @property
    def qualified_name(self):
        return f"{self.dialect}.{self.opname}"

    def __repr__(self):
        res  = ", ".join(str(r) for r in self.results)
        ops  = ", ".join(str(o) for o in self.operands)
        attr = f" {self.attributes}" if self.attributes else ""
        return f"  {res} = {self.qualified_name}({ops}){attr}"


class MLIRModule:
    """Container for a sequence of MLIR operations."""
    def __init__(self, name="module"):
        self.name = name
        self.ops: List[MLIROp] = []

    def add(self, op: MLIROp):
        self.ops.append(op)

    def print(self, title=""):
        if title:
            print(f"  [{title}]")
        for op in self.ops:
            print(op)
        print()


# ── Pattern rewriting system ───────────────────────────────────────────

class RewritePattern:
    """
    Abstract base for MLIR RewritePattern.
    In C++ MLIR:
        struct MyPattern : public OpRewritePattern<tosa::ClampOp> {
            LogicalResult matchAndRewrite(tosa::ClampOp op, PatternRewriter& r);
        };
    """
    def __init__(self, target_dialect: str, target_op: str, benefit: int = 1):
        self.target = f"{target_dialect}.{target_op}"
        self.benefit = benefit   # higher = applied first when multiple patterns match

    def match_and_rewrite(self, op: MLIROp, module: MLIRModule) -> bool:
        """
        Return True if pattern applied (op was replaced), False otherwise.
        """
        raise NotImplementedError


class PatternRewriter:
    """
    Greedy pattern rewrite driver.
    Applies patterns repeatedly until convergence (no more patterns fire).
    Mirrors: applyPatternsAndFoldGreedily() in MLIR C++.
    """
    def __init__(self, patterns: List[RewritePattern]):
        self.patterns = sorted(patterns, key=lambda p: -p.benefit)
        self.stats    = defaultdict(int)

    def run(self, module: MLIRModule) -> int:
        """
        Apply patterns until convergence.
        Returns total number of rewrites performed.
        """
        total_rewrites = 0
        changed        = True

        while changed:
            changed = False
            i = 0
            while i < len(module.ops):
                op      = module.ops[i]
                matched = False
                for pattern in self.patterns:
                    if op.qualified_name == pattern.target:
                        if pattern.match_and_rewrite(op, module):
                            self.stats[type(pattern).__name__] += 1
                            total_rewrites += 1
                            changed = True
                            matched = True
                            break  # restart from beginning of this op
                if not matched:
                    i += 1

        return total_rewrites


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Concrete patterns — lowering TOSA to Linalg
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Concrete patterns: TOSA → Linalg lowering")
print("━" * 65)
print()

# Type helpers
f32      = MLIRType("f32")
tensor8  = MLIRType("tensor<8xf32>")
memref8  = MLIRType("memref<8xf32>")
void_t   = MLIRType("void")
idx_t    = MLIRType("index")

_val_counter = [0]
def fresh(name: str, typ: MLIRType) -> MLIRValue:
    _val_counter[0] += 1
    return MLIRValue(f"{name}_{_val_counter[0]}", typ)


class TosaClampToLinalgGeneric(RewritePattern):
    """
    Pattern: tosa.clamp(x, min=0, max=INF) → linalg.generic {maxf body}

    In real MLIR (C++), this is --tosa-to-linalg pass.
    The pattern recognises a clamp with min=0 as relu and replaces it
    with a linalg.generic op containing a maxf in its body region.
    """
    def __init__(self):
        super().__init__("tosa", "clamp", benefit=10)

    def match_and_rewrite(self, op: MLIROp, module: MLIRModule) -> bool:
        # Match: is this a relu? (min=0, max=FLT_MAX)
        if op.attributes.get("min_fp", -1) != 0.0:
            return False   # not a relu

        print(f"    Pattern fired: TosaClampToLinalgGeneric on {op.qualified_name}")
        idx = module.ops.index(op)

        # Create replacement ops
        zero      = fresh("zero", f32)
        zero_op   = MLIROp("arith", "constant", [], [zero],
                            {"value": "0.0 : f32"})

        init_out  = fresh("init", tensor8)
        init_op   = MLIROp("tensor", "empty", [], [init_out])

        # linalg.generic with maxf body (represented as nested ops)
        body_in   = fresh("xi",  f32)
        body_acc  = fresh("acc", f32)
        body_out  = fresh("relu_xi", f32)
        body_op   = MLIROp("arith", "maxf",
                            [body_in, zero],
                            [body_out])
        yield_op  = MLIROp("linalg", "yield", [body_out], [])

        result    = fresh("linalg_result", tensor8)
        linalg_op = MLIROp("linalg", "generic",
                            op.operands + [init_out],
                            [result],
                            {"indexing_maps":
                             "affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>",
                             "iterator_types": '["parallel"]'},
                            regions=[[body_op, yield_op]])

        # Splice: remove tosa.clamp, insert the new ops at same position
        module.ops[idx:idx+1] = [zero_op, init_op, linalg_op]
        # Update uses: replace old result with new
        for subsequent_op in module.ops[idx+3:]:
            subsequent_op.operands = [
                result if v is op.results[0] else v
                for v in subsequent_op.operands
            ]
        return True


class LinalgGenericToAffineFor(RewritePattern):
    """
    Pattern: linalg.generic {1D parallel} → affine.for loop

    In real MLIR: --convert-linalg-to-affine-loops pass.
    Generates an affine.for loop for each parallel dimension
    and inlines the linalg region body inside.
    """
    def __init__(self):
        super().__init__("linalg", "generic", benefit=5)

    def match_and_rewrite(self, op: MLIROp, module: MLIRModule) -> bool:
        if "parallel" not in op.attributes.get("iterator_types", ""):
            return False   # not a parallel op we handle

        print(f"    Pattern fired: LinalgGenericToAffineFor on {op.qualified_name}")
        idx = module.ops.index(op)

        i_var  = fresh("i",  idx_t)
        xi_val = fresh("xi_loaded", f32)
        ri_val = fresh("ri", f32)

        load_op  = MLIROp("affine", "load",  [op.operands[0], i_var], [xi_val])
        body_ops = op.regions[0] if op.regions else []
        store_op = MLIROp("affine", "store", [ri_val, op.operands[-1], i_var], [])

        for_op = MLIROp("affine", "for", [], [],
                         {"lb": "0", "ub": "8", "step": "1"},
                         regions=[  [load_op] + body_ops + [store_op] ])

        module.ops[idx:idx+1] = [for_op]
        return True


class AffineForToSCFFor(RewritePattern):
    """
    Pattern: affine.for → scf.for (structured control flow)

    In real MLIR: --lower-affine pass.
    Strips the affine constraints, lowers to generic scf.for
    with explicit index arithmetic.
    """
    def __init__(self):
        super().__init__("affine", "for", benefit=3)

    def match_and_rewrite(self, op: MLIROp, module: MLIRModule) -> bool:
        print(f"    Pattern fired: AffineForToSCFFor on {op.qualified_name}")
        idx = module.ops.index(op)

        c0   = fresh("c0",  idx_t);  c0_op  = MLIROp("arith","constant",[],[c0], {"value":"0:index"})
        c8   = fresh("c8",  idx_t);  c8_op  = MLIROp("arith","constant",[],[c8], {"value":"8:index"})
        c1   = fresh("c1",  idx_t);  c1_op  = MLIROp("arith","constant",[],[c1], {"value":"1:index"})

        scf_op = MLIROp("scf", "for", [c0, c8, c1], [],
                         {}, regions=op.regions)

        module.ops[idx:idx+1] = [c0_op, c8_op, c1_op, scf_op]
        return True


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Run the lowering pipeline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Running the lowering pipeline step by step")
print("━" * 65)
print()

# Build initial TOSA module
x_val    = MLIRValue("x",      tensor8)
result0  = MLIRValue("result", tensor8)
clamp_op = MLIROp("tosa", "clamp", [x_val], [result0],
                   {"min_fp": 0.0, "max_fp": 3.4e38})
ret_op   = MLIROp("func", "return", [result0], [])

module = MLIRModule("relu_module")
module.add(clamp_op)
module.add(ret_op)

print("  Initial IR (TOSA dialect):")
module.print()

# Stage 1: TOSA → Linalg
print("  Applying --tosa-to-linalg (TosaClampToLinalgGeneric pattern):")
rewriter1 = PatternRewriter([TosaClampToLinalgGeneric()])
n1 = rewriter1.run(module)
print(f"  → {n1} rewrite(s) applied")
print()
print("  After --tosa-to-linalg:")
module.print()

# Stage 2: Linalg → Affine
print("  Applying --convert-linalg-to-affine (LinalgGenericToAffineFor pattern):")
rewriter2 = PatternRewriter([LinalgGenericToAffineFor()])
n2 = rewriter2.run(module)
print(f"  → {n2} rewrite(s) applied")
print()
print("  After --convert-linalg-to-affine:")
module.print()

# Stage 3: Affine → SCF
print("  Applying --lower-affine (AffineForToSCFFor pattern):")
rewriter3 = PatternRewriter([AffineForToSCFFor()])
n3 = rewriter3.run(module)
print(f"  → {n3} rewrite(s) applied")
print()
print("  After --lower-affine (SCF level — ready for LLVM lowering):")
module.print()

print(f"  Total rewrites across all stages: {n1+n2+n3}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: The greedy driver — convergence demonstration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — The greedy driver: multi-pattern convergence")
print("━" * 65)
print()

print("  Real MLIR lowering pipelines run ALL patterns simultaneously.")
print("  The greedy driver applies patterns repeatedly until convergence.")
print()
print("  Example: convert-linalg-to-affine-loops runs patterns for:")
print("    - linalg.matmul  → affine.for nest (3 loops)")
print("    - linalg.conv    → affine.for nest (5+ loops)")
print("    - linalg.generic → affine.for nests (N loops)")
print("    - linalg.fill    → affine.for with store")
print("    - linalg.dot     → affine.for with reduce")
print()
print("  Each pattern knows exactly what it matches (one op type).")
print("  The driver handles: ordering, iteration, re-application on changed IR.")
print()

GREEDY_PROPERTIES = """
  Properties of the greedy rewrite driver:

  1. CONVERGENCE: stops when no pattern fires in a full scan.
     Guaranteed to terminate if patterns are "terminating":
     each rewrite strictly reduces complexity (no cycles).

  2. LOCAL: each pattern sees one op at a time.
     Cannot match across multiple non-adjacent ops directly.
     (Use DAG patterns for multi-op matching.)

  3. BENEFIT ORDER: patterns with higher benefit are tried first.
     Allows expressing preference without global search.

  4. FOLDING: constant folding is interleaved with pattern rewriting.
     If %c = arith.addi %3, %4 where %3=%4=2, fold to %c = 4.
     Simplifies subsequent patterns.

  5. SIDE EFFECTS: patterns must be semantics-preserving.
     MLIR's type system and verifier catch invalid rewrites.
"""
print(GREEDY_PROPERTIES)

print("  Key MLIR pass commands (mlir-opt flags):")
PASS_COMMANDS = [
    ("--tosa-to-linalg",                  "TOSA dialect → Linalg"),
    ("--linalg-bufferize",                "Tensor → MemRef (bufferisation)"),
    ("--convert-linalg-to-affine-loops",  "Linalg → affine.for loops"),
    ("--lower-affine",                    "Affine → SCF + arith"),
    ("--convert-scf-to-cf",               "SCF → raw control flow"),
    ("--convert-arith-to-llvm",           "arith ops → llvm ops"),
    ("--convert-memref-to-llvm",          "memref ops → llvm ops"),
    ("--convert-func-to-llvm",            "func dialect → llvm functions"),
    ("--reconcile-unrealized-casts",      "clean up type cast bookkeeping"),
    ("--mlir-to-llvmir",                  "(mlir-translate) emit LLVM IR text"),
]
for flag, desc in PASS_COMMANDS:
    print(f"    {flag:<45} # {desc}")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · MLIR in the ML Stack — StableHLO, torch-mlir, and IREE": {
        "description": (
            "How MLIR connects PyTorch/JAX to compilation and deployment. "
            "StableHLO: the portable ML op set and its role between XLA/IREE. "
            "torch-mlir: tracing PyTorch → MLIR dialect pipeline. "
            "IREE: MLIR-native runtime for CPU/GPU/mobile. "
            "Demonstrate model serialisation in StableHLO format. "
            "Show the full connected stack: PyTorch → MLIR → LLVM → native code."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  MLIR IN THE ML STACK — StableHLO, torch-mlir, IREE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: StableHLO — the portable ML operation set
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — StableHLO: the lingua franca between frameworks")
print("━" * 65)
print()

print("  StableHLO = Stable HLO = XLA's HLO ops expressed as MLIR dialect")
print("  Stability guarantee: ops from version N work in version N+5 compilers")
print("  Purpose: portable serialisation format between ML frameworks and backends")
print()

STABLEHLO_EXAMPLES = """
  ── StableHLO IR for common operations ───────────────────────────────────

  ; Element-wise ReLU: stablehlo.maximum(x, zeros_like(x))
  func.func @relu(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %zeros = stablehlo.broadcast_in_dim %zero, dims=[]
               : (tensor<f32>) -> tensor<4x8xf32>
    %result = stablehlo.maximum %x, %zeros
               : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    func.return %result : tensor<4x8xf32>
  }

  ; Matrix multiply: stablehlo.dot_general
  func.func @matmul(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>)
                    -> tensor<4x4xf32> {
    %C = stablehlo.dot_general %A, %B,
           contracting_dims = [1] x [0]     ; sum over dim 1 of A, dim 0 of B
           : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    func.return %C : tensor<4x4xf32>
  }

  ; Softmax (decomposed into primitives)
  func.func @softmax(%x: tensor<4xf32>) -> tensor<4xf32> {
    %max    = stablehlo.reduce_window %x ... max : f32        ; max(x)
    %x_sub  = stablehlo.subtract %x, %max : tensor<4xf32>    ; x - max
    %exp_x  = stablehlo.exponential %x_sub : tensor<4xf32>   ; exp(x-max)
    %sum    = stablehlo.reduce %exp_x ... add : f32           ; sum(exp)
    %result = stablehlo.divide %exp_x, %sum : tensor<4xf32>  ; exp / sum
    func.return %result : tensor<4xf32>
  }

  ; Convolution: stablehlo.convolution (rich attribute set)
  func.func @conv(%input: tensor<1x28x28x1xf32>,
                  %filter: tensor<3x3x1x32xf32>) -> tensor<1x26x26x32xf32> {
    %out = stablehlo.convolution(%input, %filter)
           dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f]
           ;             NHWC input      HWIO filter     NHWC output
           window = {stride = [1, 1], pad = [[0,0],[0,0]]}
           : (tensor<1x28x28x1xf32>, tensor<3x3x1x32xf32>)
           -> tensor<1x26x26x32xf32>
    func.return %out : tensor<1x26x26x32xf32>
  }
"""
print(STABLEHLO_EXAMPLES)

print("  StableHLO compatibility table:")
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ Framework     │ Produces StableHLO    │ Consumes StableHLO  │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ JAX           │ ✅ primary export      │ N/A                │")
print("  │ TensorFlow    │ ✅ via tf2xla          │ N/A                │")
print("  │ PyTorch       │ ✅ via torch-mlir      │ N/A                │")
print("  │ XLA           │ N/A                   │ ✅ primary input   │")
print("  │ IREE          │ N/A                   │ ✅ primary input   │")
print("  │ TVM           │ N/A                   │ ✅ via import path │")
print("  │ OpenXLA       │ N/A                   │ ✅ primary input   │")
print("  └──────────────────────────────────────────────────────────────┘")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: torch-mlir — PyTorch → MLIR
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — torch-mlir: PyTorch → MLIR dialect pipeline")
print("━" * 65)
print()

try:
    import torch
    import torch_mlir
    from torch_mlir import torchscript

    print(f"  torch-mlir available ✅")
    print()

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2, bias=False)

        def forward(self, x):
            return torch.relu(self.linear(x))

    model    = SimpleModel().eval()
    example  = torch.randn(1, 4)

    # Lower to TOSA (generic hardware-neutral dialect)
    module_tosa = torchscript.compile(
        model, example,
        output_type=torch_mlir.OutputType.TOSA,
    )

    print("  Lowered to TOSA dialect:")
    tosa_ir = str(module_tosa)
    for line in tosa_ir.split('\n')[:25]:
        print(f"    {line}")
    print("    ...")
    print()

    # Lower to StableHLO
    module_hlo = torchscript.compile(
        model, example,
        output_type=torch_mlir.OutputType.STABLEHLO,
    )
    print("  Lowered to StableHLO dialect:")
    hlo_ir = str(module_hlo)
    for line in hlo_ir.split('\n')[:25]:
        print(f"    {line}")
    print("    ...")

except ImportError as e:
    print(f"  torch-mlir not installed: {e}")
    print("  Install: pip install torch-mlir-core")
    print()

    TORCH_MLIR_PIPELINE = """
  torch-mlir LOWERING PIPELINE:

  Step 1: PyTorch model → TorchScript / FX graph
    import torch, torch_mlir
    module = torchscript.compile(model, example_input,
                                  output_type=torch_mlir.OutputType.TOSA)

  Step 2: TorchScript → Torch dialect (ATen-level ops)
    torch.aten.linear(%x, %w, %b) → Torch dialect
    torch.aten.relu(%x)           → Torch dialect

  Step 3: Torch dialect → TOSA or StableHLO
    torch.aten.relu → tosa.clamp(x, 0, FLT_MAX)
    torch.aten.matmul → tosa.matmul
    (or equivalently: stablehlo.maximum, stablehlo.dot_general)

  Step 4: TOSA → Linalg → Affine → LLVM
    (standard MLIR lowering pipeline described in operations 1 & 2)

  Example output (TOSA dialect for relu(linear(x))):

    func.func @forward(%x: tensor<1x4xf32>) -> tensor<1x2xf32> {
      %w = "tosa.const"() {value = dense<...> : tensor<2x4xf32>} : () -> tensor<2x4xf32>

      ; Linear layer: y = x @ W^T
      %matmul = "tosa.matmul"(%x, %w_transposed)
                : (tensor<1x4xf32>, tensor<4x2xf32>) -> tensor<1x2xf32>

      ; ReLU: clamp with min=0
      %relu = "tosa.clamp"(%matmul) {
        min_fp = 0.0 : f32, max_fp = 3.4e38 : f32
      } : (tensor<1x2xf32>) -> tensor<1x2xf32>

      return %relu : tensor<1x2xf32>
    }
"""
    print(TORCH_MLIR_PIPELINE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: IREE — MLIR-native runtime
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 3 — IREE: MLIR-native compilation and runtime")
print("━" * 65)
print()

IREE_OVERVIEW = """
  IREE (Intermediate Representation Execution Environment) is a
  compiler + runtime built entirely on MLIR.

  Design goals:
    - MLIR all the way: from StableHLO input to final binary
    - Zero Python runtime dependency (just a C runtime)
    - Multi-target: CPU (x86/ARM), GPU (Vulkan/Metal/CUDA), mobile
    - Streaming execution: pipelines computation with memory management

  Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │  StableHLO / TOSA  (input from any ML framework)                │
    ├──────────────────────────────────────────────────────────────────┤
    │  IREE Compiler                                                   │
    │    ↓ --iree-input-transformation-pipeline                        │
    │  Flow dialect    (data flow between dispatch regions)            │
    │    ↓ --iree-flow-transformation-pipeline                         │
    │  Stream dialect  (execution streams, asynchrony)                 │
    │    ↓ --iree-hal-transformation-pipeline                          │
    │  HAL dialect     (Hardware Abstraction Layer)                    │
    │    ↓ Target-specific lowering                                    │
    │  LLVM / SPIR-V / Metal / ROCm  (code generation)               │
    ├──────────────────────────────────────────────────────────────────┤
    │  .vmfb artifact  (VM FlatBuffer = compiled binary)              │
    ├──────────────────────────────────────────────────────────────────┤
    │  IREE Runtime   (iree-run-module, C/Python API)                  │
    └──────────────────────────────────────────────────────────────────┘

  Key IREE concepts:

  Dispatch regions:
    Computation is split into "dispatch regions" — units that can be
    scheduled as one GPU/CPU kernel launch. IREE identifies dispatch
    boundaries automatically from the dataflow graph.

  HAL (Hardware Abstraction Layer):
    Uniform interface for CPU, Vulkan GPU, Metal GPU, CUDA.
    The same compiled module can dispatch to different backends
    at runtime by selecting the right HAL driver.

  Asynchronous execution:
    IREE uses asynchronous execution by default.
    Result "semaphores" are returned immediately;
    the caller waits only when the result is actually needed.
    This enables overlapping GPU and CPU work.
"""
print(IREE_OVERVIEW)

try:
    import iree.runtime as iree_rt
    import iree.compiler as iree_cc

    print("  IREE available — demonstrating compilation:")
    MLIR_INPUT = """
func.func @add(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %result = stablehlo.add %x, %y : tensor<4xf32>
  func.return %result : tensor<4xf32>
}
"""
    compiled = iree_cc.compile_str(
        MLIR_INPUT,
        target_backends=["llvm-cpu"],
        input_type="stablehlo",
    )
    config  = iree_rt.Config("local-task")
    ctx     = iree_rt.SystemContext(config=config)
    vm_mod  = iree_rt.VmModule.from_flatbuffer(ctx.instance, compiled)
    ctx.add_vm_module(vm_mod)

    add_fn  = ctx.modules.module["add"]
    x       = np.array([1., 2., 3., 4.], dtype=np.float32)
    y       = np.array([10., 20., 30., 40.], dtype=np.float32)
    result  = add_fn(x, y)[0]

    print(f"  add([1,2,3,4], [10,20,30,40]) = {result}")
    print(f"  Expected:                       {x + y}")

except ImportError:
    print("  IREE not installed: pip install iree-runtime iree-compiler")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: The connected compiler stack summary
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — The connected compiler stack")
print("━" * 65)
print()

CONNECTED_STACK = """
  HOW LLVM → MLIR → XLA → TVM CONNECT:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                        USER LEVEL                                   │
  │  PyTorch model  │  JAX function  │  ONNX model  │  TF SavedModel   │
  └──────────┬──────┴────────┬───────┴──────┬───────┴────────┬─────────┘
             │               │              │                │
        torch-mlir      JAX lowering    onnx-mlir       tf.SavedModel
             │               │              │                │
             ▼               ▼              ▼                ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                   MLIR (this module)                                │
  │  StableHLO / TOSA / MHLO  (high-level ML ops, framework-neutral)   │
  │        ↓ tosa-to-linalg / stablehlo-to-linalg                      │
  │  Linalg dialect            (algebraic, tiling/fusion here)          │
  │        ↓ linalg-bufferize + convert-linalg-to-affine               │
  │  Affine + SCF + MemRef     (polyhedral, explicit loops)             │
  │        ↓ lower-affine + convert-to-llvm                             │
  │  LLVM dialect              (direct LLVM IR in MLIR)                 │
  └──────────┬──────────────────────────────────────────────────────────┘
             │ mlir-translate --mlir-to-llvmir
             ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                   LLVM (previous module)                            │
  │  LLVM IR (SSA form, scalar ops)                                     │
  │        ↓ opt: loop-vectorize, instcombine, LICM, GVN               │
  │  Optimised LLVM IR                                                  │
  │        ↓ llc: instruction selection, register allocation            │
  │  x86-64 / ARM64 / RISC-V / WebAssembly machine code               │
  └─────────────────────────────────────────────────────────────────────┘

  XLA (next module) sits at the StableHLO → GPU code generation layer.
  XLA uses MLIR infrastructure internally but adds:
    - GPU kernel code generation (NVPTX, AMDGCN)
    - TPU-specific compilation
    - HLO-level algebraic simplifications

  TVM (final module) provides:
    - Auto-scheduling (search-based, vs MLIR's analytical transforms)
    - Hardware-specific tuning (loop tile sizes, memory hierarchy)
    - Broader hardware support (Mali GPU, FPGA, custom accelerators)

  The convergence: StableHLO is now the import format for both XLA and
  TVM, making MLIR the shared interface between ALL of them.
"""
print(CONNECTED_STACK)

print("  MLIR PROJECT RESOURCES:")
print("  mlir.llvm.org              — official documentation")
print("  github.com/llvm/llvm-project/mlir  — source code")
print("  github.com/openxla/stablehlo       — StableHLO spec")
print("  github.com/iree-org/iree           — MLIR-native runtime")
print("  github.com/llvm/torch-mlir         — PyTorch → MLIR")
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