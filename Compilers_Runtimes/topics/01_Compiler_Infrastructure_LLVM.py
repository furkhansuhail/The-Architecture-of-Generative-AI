"""
LLVM — The Universal Compiler Infrastructure
=============================================

LLVM is not a single compiler. It is a collection of reusable compiler
components — a modular toolkit for building compilers, runtimes, static
analysers, JIT engines, and code generators — that has become the shared
foundation beneath nearly every modern language and ML framework.

Clang (C/C++/Objective-C), Rust, Swift, Julia, Kotlin Native, WebAssembly,
CUDA's PTX backend, Apple's Metal shaders, AMD's ROCm, NVIDIA's NVVM,
Google's XLA, Apache TVM, MLIR — all are built on or target LLVM.

Understanding LLVM means understanding the intermediate representation (IR)
that ties all of these together: a strongly-typed, SSA-form, platform-
independent assembly language that any front end can produce and any back
end can consume. It is the lingua franca of modern compilation.

In the context of ML systems, LLVM is the lowest layer of every major
compiler stack. When torch.compile(), XLA, or TVM generate machine code
for CPUs, they do so by emitting LLVM IR and letting LLVM handle the
platform-specific details. When you understand LLVM, you understand why
ML compilers can target x86, ARM, RISC-V, and WASM from the same codebase.

"""

import textwrap
import re

TOPIC_NAME   = "LLVM — The Universal Compiler Infrastructure"
DISPLAY_NAME = "01 · LLVM"
ICON         = "🔩"
SUBTITLE     = "The IR, Pass Pipeline, and Code Generation Foundation of Modern ML"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT LLVM IS AND WHY IT EXISTS

### The Problem Before LLVM

    In the early 2000s, building a compiler meant:
        - Writing a front end (parser, type checker, AST)
        - Writing optimisation passes FROM SCRATCH
        - Writing a back end for EACH target CPU architecture

    For N languages and M target architectures, you needed N×M compiler back
    ends — all hand-written, all duplicating the same optimisation logic.

    GCC existed but was a monolithic tangle — adding a new language or
    target required understanding the entire codebase.

    LLVM (originally "Low Level Virtual Machine", now just LLVM) was
    Chris Lattner's PhD thesis at UIUC (2000–2004). The core insight:

        ┌─────────────────────────────────────────────────────────────────┐
        │  Define ONE intermediate representation (IR).                   │
        │  Any front end compiles TO that IR.                             │
        │  Any back end compiles FROM that IR.                            │
        │  Optimisations operate ON that IR.                              │
        │  N front ends + M back ends = N+M components, not N×M.          │
        └─────────────────────────────────────────────────────────────────┘

### LLVM's Place in the Compiler Stack

    Every compilation goes through three phases:

    FRONT END        MIDDLE END           BACK END
    (language-       (optimisation,       (machine code
     specific)        IR-level)            generation)

    C/C++/ObjC  ──► │              │ ──► x86-64 assembly
    Rust        ──► │  LLVM IR     │ ──► ARM64 assembly
    Swift       ──► │  (SSA form)  │ ──► RISC-V assembly
    Julia       ──► │              │ ──► WebAssembly
    XLA kernel  ──► │  OptPipeline │ ──► NVPTX (NVIDIA GPU)
    TVM relay   ──► │              │ ──► AMDGCN (AMD GPU)
    MLIR/Linalg ──► │              │ ──► AVX-512 intrinsics

    The IR is the contract. The front end and back end are independent.
    Any language that generates LLVM IR runs on any target LLVM supports.

### LLVM in the ML World

    Every major ML framework's CPU path goes through LLVM:

        PyTorch + torch.compile:
            TorchDynamo captures graph → TorchInductor generates Triton/C++ →
            Triton compiles via LLVM → x86/ARM machine code

        TensorFlow + XLA:
            TF graph → XLA HLO → XLA LLVM backend → native machine code

        Apache TVM:
            Model (ONNX/MXNet/TF) → Relay IR → TIR → LLVM IR → any CPU

        MLIR:
            High-level ML ops → progressively lower dialects →
            LLVM IR dialect → LLVM → machine code

        CUDA (NVVM):
            CUDA C → Clang → NVVM IR (LLVM variant) → PTX → cubin (GPU binary)

    Understanding LLVM is prerequisite to understanding ALL of these.


##### PART 2 — LLVM IR: THE INTERMEDIATE REPRESENTATION

### What LLVM IR Is

    LLVM IR is a strongly-typed, low-level language that looks like a
    portable assembly language. It has three equivalent representations:

        1. Human-readable text (.ll files):    for debugging and learning
        2. Binary bitcode (.bc files):         compact, fast to load
        3. In-memory data structures (LLVM C++ API): used by compilers

    You will see LLVM IR when ML compilers like TVM or XLA emit code
    for debugging (with -dump-ir or -print-passes).

### SSA Form — Static Single Assignment

    LLVM IR is in SSA (Static Single Assignment) form.
    The defining property: every variable is assigned EXACTLY ONCE.

    Normal code (variable reused):
        x = 1
        x = x + 2     ← x assigned twice
        x = x * 3

    SSA form (each assignment creates a new "version"):
        %x1 = 1
        %x2 = add %x1, 2    ← new variable %x2 (not %x1)
        %x3 = mul %x2, 3    ← new variable %x3

    Why SSA? It makes data flow analysis trivial:
        - Uses of %x2 all come from exactly one definition
        - Dead code: if %x3 is never used, eliminate it
        - Value numbering: if %a = add %b, %c appears twice, merge
        - Constant propagation: %x1 is always 1 → substitute

    SSA is how LLVM makes optimisation clean and composable.

### The Structure of LLVM IR

    Hierarchy: Module → Function → BasicBlock → Instruction

    MODULE:
        A translation unit. Contains functions, global variables, metadata.

    FUNCTION:
        define i32 @relu(float %x) {
            ...
        }
        Named with @. Has a return type, argument types, argument names.

    BASIC BLOCK:
        A sequence of instructions that executes straight-through.
        NO branches in the middle — only the last instruction can branch.
        Starts with an optional label, ends with a terminator (br, ret, switch).

    INSTRUCTION:
        Every SSA instruction. Types: arithmetic, memory (load/store),
        control flow (br, call, ret), vector ops, metadata.

### LLVM IR Type System

    Integer types:
        i1    (boolean)
        i8    (byte, used for chars and int8 quantisation)
        i16, i32, i64, i128

    Floating point types:
        half (fp16)
        float (fp32)
        double (fp64)
        bfloat (bfloat16 — added for ML workloads!)

    Pointer types:
        ptr   (opaque pointer, modern LLVM)
        i32*  (typed pointer, legacy)

    Vector types (SIMD):
        <4 x float>     — 4 floats packed (maps to SSE/AVX register)
        <8 x float>     — 8 floats packed (AVX2 register, __m256)
        <16 x float>    — 16 floats (AVX-512 register, __m512)
        <n x i8>        — INT8 quantised vector

    These vector types are how ML compilers generate SIMD code.
    TVM's vectorise pass turns loops into <n x float> operations.

### A Concrete LLVM IR Example

    C function:
        float relu(float x) {
            return x > 0.0f ? x : 0.0f;
        }

    LLVM IR (simplified):
        define float @relu(float %x) {
        entry:
            %cmp   = fcmp ogt float %x, 0.000000e+00   ; x > 0.0 ?
            br i1 %cmp, label %true_bb, label %false_bb

        true_bb:
            br label %merge                              ; goto merge

        false_bb:
            br label %merge                              ; goto merge

        merge:
            %result = phi float [ %x, %true_bb ],       ; if came from true: use x
                                 [ 0.000000e+00, %false_bb ]  ; else 0.0
            ret float %result
        }

    Key instructions:
        fcmp ogt:   float compare, ordered greater-than
        br i1 cond: conditional branch on boolean
        phi:        SSA merge point — value depends on which predecessor ran
        ret:        return the value

    The phi instruction is the signature of SSA form.
    It merges values from different control flow paths.


##### PART 3 — THE PASS PIPELINE: HOW LLVM OPTIMISES

### What a Pass Is

    A pass is a transformation or analysis that operates on the LLVM IR.
    Passes are the core unit of composition in LLVM.

    Two categories:
        ANALYSIS passes:    read IR, compute information, store it for
                            other passes to use. No IR modification.
                            Examples: DominatorTree, LoopInfo, AliasAnalysis

        TRANSFORM passes:   read IR, modify it to produce better IR.
                            Use analysis results from analysis passes.
                            Examples: mem2reg, instcombine, loop-vectorize

### The Standard Optimisation Pipeline

    LLVM's -O2 optimisation pipeline runs hundreds of passes. Key ones:

    1. mem2reg (Memory to Register):
        Converts stack-allocated variables (alloca) to SSA virtual registers.
        This is usually the FIRST pass — it's what creates proper SSA form.
        alloca + load/store → phi nodes + register values.

    2. instcombine (Instruction Combining):
        Algebraic simplifications on single instructions.
        Examples:
            x + 0 → x
            x * 1 → x
            x * 2 → x << 1   (faster)
            (x + c1) + c2 → x + (c1+c2)   (constant folding)
        Runs many times — each run enables more simplifications.

    3. simplifycfg (Simplify Control Flow):
        Removes unreachable blocks, merges duplicate branches,
        eliminates dead basic blocks.

    4. gvn (Global Value Numbering):
        Eliminates redundant computations across the whole function.
        If %a = add i32 %x, %y computed twice → eliminate the second.

    5. licm (Loop Invariant Code Motion):
        Moves computations OUT of loops if they produce the same result
        every iteration.
            for i in range(N): y = x * 3 + offset  → hoist x*3+offset

    6. loop-vectorize:
        Converts scalar loops to SIMD vector operations.
            for i: a[i] += b[i]  → <4 x float> vector add (SSE/AVX)
        This is how ML matrix kernels get SIMD acceleration.

    7. SLP vectorize (Superword Level Parallelism):
        Groups adjacent scalar operations into vector instructions.
        Complements loop-vectorize for non-loop code.

    8. inline:
        Replaces function calls with the function body at the call site.
        Critical for ML kernels — enables subsequent optimisations that
        cross function boundaries (e.g., constant propagation through inlined args).

### How ML Compilers Use the Pass Pipeline

    TVM emits LLVM IR and then runs the full LLVM optimisation pipeline:

        # TVM's LLVM code generation
        with tvm.target.Target("llvm -mcpu=cascadelake -mattr=+avx512f"):
            lib = relay.build(mod, target)

    TVM's code is often unoptimised LLVM IR with scalar operations.
    LLVM's loop-vectorize then turns those scalar loops into AVX-512
    instructions automatically — without TVM needing to know the target.

    XLA uses LLVM similarly for CPU backends:
        XLA HLO → XLA LLVM emitter → LLVM IR → LLVM passes → x86/ARM

    The magic: XLA and TVM write GENERIC LLVM IR.
    LLVM's back end handles the platform-specific details.

### Pass Managers

    New Pass Manager (NPM, default since LLVM 14):
        Uses a hierarchical structure: ModulePassManager contains
        FunctionPassManagers which contain individual passes.
        Analyses are cached and shared between passes.

    Running passes manually (llvm-opt tool):
        opt -passes="mem2reg,instcombine,gvn" input.ll -o output.ll
        opt -passes="default<O2>" input.ll -o output.ll   # standard -O2 pipeline

    TVM uses LLVM's pass pipeline programmatically:
        target_machine->addPassesToEmitFile(pass_manager, output_stream,
                                             nullptr, CGFT_ObjectFile)


##### PART 4 — BACKENDS: LLVM'S MACHINE CODE GENERATION

### The Backend Pipeline

    After the IR-level optimisations, the backend converts IR to machine code:

    LLVM IR
        ↓  Instruction Selection (SelectionDAG or GlobalISel)
    Machine IR (MIR) — abstract machine instructions, infinite virtual regs
        ↓  Register Allocation (LLVM's greedy allocator)
    MIR with physical registers assigned
        ↓  Instruction Scheduling (reduce pipeline stalls)
    MIR scheduled
        ↓  Code Emission
    Assembly text or ELF object file

### Supported Targets (ML-relevant)

    x86-64:    Primary desktop/server target. AVX2 and AVX-512 for SIMD.
    ARM AArch64: Mobile, Apple M-series, AWS Graviton. NEON/SVE SIMD.
    ARM 32-bit:  Raspberry Pi, embedded. NEON SIMD.
    RISC-V:    Emerging server and embedded. TVM targets this extensively.
    WebAssembly: Browser deployment. TFLite.js, ONNX.js.
    NVPTX:     NVIDIA GPU pseudo-assembly. Used by CUDA's Clang front end.
    AMDGCN:    AMD GPU ISA. Used by ROCm/HIP.
    SPIR-V:    Vulkan/OpenCL GPU IR. Via LLVM SPIR-V backend.

### Target Triples and Feature Flags

    LLVM targets are identified by "triples": arch-vendor-OS-env.
    ML frameworks specify targets with feature flags for SIMD:

        "x86_64-linux-gnu"                   — generic x86-64 Linux
        "x86_64-linux-gnu" + "+avx2"         — enable AVX2 (256-bit SIMD)
        "x86_64-linux-gnu" + "+avx512f"      — enable AVX-512 (512-bit SIMD)
        "aarch64-linux-gnu"                  — ARM 64-bit Linux
        "aarch64-linux-gnu" + "+sve"         — ARM Scalable Vector Extension
        "aarch64-apple-macosx" + "+neon"     — Apple Silicon

    TVM example:
        tvm.target.Target("llvm -mtriple=x86_64-linux-gnu -mattr=+avx2,+fma")

    The feature flags tell LLVM which SIMD instruction sets are available.
    loop-vectorize then uses those to generate wider vector operations.

### LLVM JIT: ORC and LLJIT

    LLVM has a JIT compiler that ML systems use for on-demand compilation:

        LLVM ORC JIT (On-Request Compilation):
            Lazily compiles IR functions when first called.
            Used by: Julia (all computation JIT-compiled), PyPy, XLA.

        LLJIT:
            Simpler layer on top of ORC.
            Load an LLVM module → execute it directly in the current process.

    XLA's CPU backend uses LLVM JIT:
        XLA compiles an HLO computation → LLVM IR → JIT compiles → calls it.
        The result: model inference happens as native machine code in the
        same process as your Python, with zero serialisation overhead.

    TVM uses LLVM JIT for auto-tuning:
        For each candidate schedule, TVM JIT-compiles it and benchmarks it.
        Hundreds of candidates are compiled and timed per operator.
        Without LLVM JIT, this would require writing to disk and spawning processes.


##### PART 5 — LLVM FOR VECTORISATION: HOW ML GETS SIMD FOR FREE

### What SIMD Is

    SIMD (Single Instruction, Multiple Data) applies one instruction to
    multiple data elements simultaneously.

    Scalar addition (1 result per instruction):
        fadd float %a, %b    → computes a + b

    Vector addition (8 results per instruction with AVX2):
        fadd <8 x float> %va, %vb  → computes [a₀+b₀, a₁+b₁, ..., a₇+b₇]

    This is 8× throughput for the same clock cycle count.
    For elementwise operations (relu, softmax numerator), SIMD is near-linear speedup.

### LLVM Auto-Vectorisation

    LLVM's loop-vectorize pass automatically converts loops to SIMD:

    Input IR (scalar loop):
        for (int i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }

    After loop-vectorize with AVX-512:
        ; Process 16 floats at a time
        %va = load <16 x float>, ptr %a_ptr
        %vb = load <16 x float>, ptr %b_ptr
        %vc = fadd <16 x float> %va, %vb
        store <16 x float> %vc, ptr %c_ptr

    The pass handles:
        Loop bounds that aren't multiples of vector width (epilogue loop)
        Memory alignment analysis (aligned loads are faster)
        Aliasing analysis (can a and c overlap? if so, can't vectorise)
        Reduction patterns (sum of array → horizontal add intrinsic)

### Vectorisation in the ML Context

    When TVM generates code for an elementwise kernel like ReLU:

        TVM Compute definition:
            B = te.compute(A.shape, lambda *i: tvm.te.max(A(*i), 0))

        TVM emits scalar LLVM IR for the loop body.
        LLVM's loop-vectorize converts it to AVX-512 instructions.
        Result: 16 ReLU operations per cycle on Skylake.

    This is why ML compilers don't need to write SIMD intrinsics manually:
    they emit clean scalar IR and let LLVM handle the vectorisation.

    For matrix multiplication, LLVM's auto-vectoriser is NOT sufficient —
    matmul has a 3-loop structure that requires tiling, reordering, and
    explicit GEMM calls. TVM and XLA handle matmul manually (or via BLAS).
    Auto-vectorisation is the accelerator for EVERYTHING ELSE.

### LLVM Intrinsics for ML

    LLVM intrinsics are built-in functions with hardware-specific semantics:

        @llvm.fma.f32(float %a, %b, %c)    → fused multiply-add: a*b+c
        @llvm.maxnum.f32(float %a, %b)     → max(a, b) respecting NaN semantics
        @llvm.sqrt.f32(float %a)            → hardware sqrt
        @llvm.vector.reduce.fadd(...)       → horizontal sum of vector

    Fused multiply-add (FMA) is critical for ML:
        Normal: y = a*b + c  → two instructions, one round-off error each
        FMA:    y = a*b + c  → one instruction, one round-off (more accurate!)
        2× throughput for operations of the form a*b+c (common in matmul)

    TVM and XLA emit FMA intrinsics explicitly to ensure hardware FMA
    units are used.


##### PART 6 — CLANG: THE C/C++ FRONT END

### Why Clang Matters for ML

    Most ML framework kernels are written in C/C++ or CUDA.
    Clang compiles all of them.

    CUDA compilation:
        CUDA C source → Clang parses → device code → NVVM (LLVM variant)
        → PTX (GPU IR) → cubin (GPU binary)
        The "host" C++ code → Clang → LLVM → x86/ARM

    TVM's TIR to C compilation:
        TVM can lower to C (via TIR → C codegen) → Clang → LLVM → binary.
        This is the fallback path when native LLVM codegen is not available.

### Clang as a Library

    Unlike GCC, Clang is designed to be used as a LIBRARY.
    libclang provides a C API for parsing and analysing C/C++ code.

    ML applications:
        - TVM uses Clang/libclang to compile generated C code
        - Static analysis tools check kernel code for correctness
        - IDE tools (clangd) provide autocomplete for ML kernel development
        - LLVM's TableGen DSL describes target-specific instructions

### Clang Attributes Relevant to ML

    Clang attributes control compilation behaviour:

        __attribute__((noinline)):     prevent inlining (debugging, profiling)
        __attribute__((always_inline)): force inlining (critical inner loops)
        __attribute__((aligned(64))):  align data to cache line boundary
        __attribute__((vectorize)):    hint to vectorise this loop
        __attribute__((unroll)):       hint to unroll this loop
        __attribute__((hot)):          mark as performance-critical path
        __attribute__((target("avx512f"))):  compile this function for AVX-512
                                             even if the rest uses AVX2


##### PART 7 — LLVM IN THE ML COMPILER STACK

### The Full Stack for a PyTorch Model on CPU

    python model(x)
        ↓
    torch.compile() → TorchDynamo captures Python bytecode
        ↓
    FX graph (PyTorch's intermediate graph representation)
        ↓
    TorchInductor → generates C++ or Triton code
        ↓  (for CPU path: generates C++ with loop nests)
    Clang/LLVM front end → LLVM IR
        ↓
    LLVM pass pipeline (instcombine, licm, loop-vectorize, ...)
        ↓
    LLVM back end (x86 or ARM) → assembly
        ↓
    Assembler → ELF object → linked into Python process
        ↓
    Native machine code executing in Python process via ctypes

### The Full Stack for a TF Model on CPU

    tf.function decorated function
        ↓
    XLA HLO (High Level Optimizer) — ML-level graph
        ↓
    XLA LLVM emitter → LLVM IR
        ↓
    LLVM pass pipeline
        ↓
    LLVM x86/ARM back end → object code
        ↓
    JIT-compiled, called from TensorFlow runtime

### The Full Stack for TVM on Any Target

    Neural network model (ONNX/PyTorch/TF)
        ↓
    TVM Relay IR → optimised Relay IR (graph-level fusion, layout)
        ↓
    TVM TE (Tensor Expressions) / TIR (Tensor IR)
        ↓
    TVM LLVM codegen → LLVM IR for compute kernels
        ↓
    LLVM target-specific compilation (x86/ARM/RISC-V)
        ↓
    Executable binary (.so or .tar) for target device

### What LLVM Does That ML Compilers Don't Want to Do

    ML compilers emit GENERIC LLVM IR — clean, unoptimised scalar code.
    They delegate to LLVM:

        Register allocation:     LLVM knows the target's register count/types.
        Instruction scheduling:  LLVM knows which ops can execute in parallel.
        SIMD vectorisation:      LLVM knows the target's vector width.
        ABI compliance:          LLVM knows calling conventions (cdecl, etc.)
        Peephole optimisations:  LLVM combines instruction sequences.
        Debug info generation:   LLVM emits DWARF for debuggers.

    Without LLVM, each ML compiler would need to replicate this for every CPU.
    With LLVM, they write one IR → run everywhere.


##### PART 8 — KEY LLVM TOOLS AND HOW TO USE THEM

### The LLVM Toolchain

    clang:          C/C++/ObjC compiler (produces LLVM IR or native code)
    clang++:        C++ compiler
    llc:            LLVM static compiler (LLVM IR → assembly/object)
    opt:            LLVM optimiser (runs passes on LLVM IR)
    llvm-dis:       LLVM disassembler (bitcode → human-readable IR)
    llvm-as:        LLVM assembler (human-readable IR → bitcode)
    llvm-link:      LLVM linker (merges multiple .bc files)
    llvm-nm:        List symbols in an object file
    llvm-objdump:   Disassemble machine code
    llvm-mc:        Machine code toolkit (assemble/disassemble)

### Working with LLVM IR Directly

    Emit IR from C code:
        clang -S -emit-llvm -O0 relu.c -o relu.ll   # human-readable IR
        clang -c -emit-llvm -O0 relu.c -o relu.bc   # bitcode

    Run optimisation passes:
        opt -passes="mem2reg,instcombine" relu.ll -S -o relu_opt.ll
        opt -passes="default<O2>"         relu.ll -S -o relu_O2.ll

    Compile IR to native code:
        llc -filetype=asm relu_opt.ll -o relu.s       # assembly
        llc -filetype=obj relu_opt.ll -o relu.o       # object file

    View passes that ran:
        opt -passes="default<O2>" -debug-pass-manager relu.ll

### Using LLVM from Python (via llvmlite)

    llvmlite is a Python binding for LLVM used by Numba and other tools:

        from llvmlite import ir, binding

        # Create a module
        module = ir.Module(name="relu_module")

        # Define a function: float relu(float x)
        float_t = ir.FloatType()
        func_ty = ir.FunctionType(float_t, [float_t])
        func    = ir.Function(module, func_ty, name="relu")
        x,      = func.args
        x.name  = "x"

        # Build the basic blocks
        entry_bb  = func.append_basic_block("entry")
        true_bb   = func.append_basic_block("positive")
        false_bb  = func.append_basic_block("zero")
        merge_bb  = func.append_basic_block("merge")

        # entry block: compare x > 0
        builder   = ir.IRBuilder(entry_bb)
        zero_f    = ir.Constant(float_t, 0.0)
        cond      = builder.fcmp_ordered(">", x, zero_f, name="cmp")
        builder.cbranch(cond, true_bb, false_bb)

        # positive block: just branch to merge
        builder   = ir.IRBuilder(true_bb)
        builder.branch(merge_bb)

        # zero block: just branch to merge
        builder   = ir.IRBuilder(false_bb)
        builder.branch(merge_bb)

        # merge block: phi + return
        builder   = ir.IRBuilder(merge_bb)
        phi       = builder.phi(float_t, name="result")
        phi.add_incoming(x,     true_bb)
        phi.add_incoming(zero_f, false_bb)
        builder.ret(phi)

        print(str(module))   # prints human-readable LLVM IR

    llvmlite is used by:
        Numba:   JIT compiles Python/NumPy code via LLVM
        Some TVM targets: llvmlite for Python-side IR construction

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · LLVM IR — Reading, Writing and Understanding SSA Form": {
        "description": (
            "Hands-on LLVM IR. Use llvmlite to construct LLVM IR programmatically. "
            "Build a ReLU function, a dot product, and a simple loop in LLVM IR. "
            "Show SSA form with phi nodes. Run the IR through the optimisation "
            "pipeline and compare before/after. Inspect what loop-vectorize does "
            "to a scalar elementwise loop."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  LLVM IR — READING, WRITING AND SSA FORM")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# llvmlite: Python bindings for LLVM
# ─────────────────────────────────────────────────────────────────────────
try:
    from llvmlite import ir, binding
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()
    HAS_LLVMLITE = True
    print(f"  llvmlite available ✅")
    print(f"  LLVM version: {binding.llvm_version_info}")
except ImportError:
    HAS_LLVMLITE = False
    print("  llvmlite not installed: pip install llvmlite")
    print("  (also installed automatically with: pip install numba)")
    print()

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: LLVM IR Reference — reading the notation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — LLVM IR notation guide")
print("━" * 65)
print()

IR_GUIDE = """
  LLVM IR CHEAT SHEET
  ════════════════════════════════════════════════════════════════

  IDENTIFIERS:
    %name     — local (function-scoped) value. SSA register.
    @name     — global (module-level). Functions, global variables.
    i32, i64  — integer types (8, 16, 32, 64 bit)
    float     — 32-bit floating point (f32)
    double    — 64-bit floating point (f64)
    half      — 16-bit float (f16)
    bfloat    — bfloat16 (added for ML workloads!)
    ptr       — opaque pointer (modern LLVM ≥14)
    <4 x f32> — SIMD vector: 4 floats packed

  COMMON INSTRUCTIONS:
    %r = add i32 %a, %b          integer add
    %r = fadd float %a, %b       float add
    %r = fmul float %a, %b       float multiply
    %r = fcmp ogt float %a, %b   float compare ordered greater-than → i1 (bool)
    %r = icmp sgt i32 %a, %b     integer compare signed greater-than → i1
    %r = select i1 %c, i32 %t, i32 %f  ternary: if %c then %t else %f
    %r = phi i32 [%v1, %bb1], [%v2, %bb2]  SSA merge
    %r = call float @foo(float %x)    function call
    ret float %r                      return value
    br label %target                  unconditional branch
    br i1 %cond, label %t, label %f   conditional branch

  MEMORY:
    %r = alloca i32                   allocate local variable on stack
    %r = load float, ptr %p           load from memory
         store float %v, ptr %p       write to memory
    %r = getelementptr i32, ptr %p, i32 %i  pointer arithmetic (array index)

  VECTOR OPS:
    %r = fadd <4 x float> %va, %vb   vector add (4 floats at once)
    %r = load <8 x float>, ptr %p    vector load
    %r = extractelement <4 x float> %v, i32 2  get element at index 2
    %r = shufflevector ...             rearrange vector elements

  ATTRIBUTES:
    define float @f(float %x) nounwind { ... }
      nounwind:   no exceptions → enables more optimisations
      readnone:   no memory side effects
      alwaysinline: force inlining

  METADATA:
    !dbg → debug information (line numbers, variable names)
    !loop → loop metadata (vectorisation hints, unroll counts)
"""

print(IR_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Annotated LLVM IR for common ML operations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Annotated LLVM IR for ML operations")
print("━" * 65)
print()

RELU_IR = """
  ; ── ReLU: max(x, 0) ──────────────────────────────────────────────────
  ; C equivalent: float relu(float x) { return x > 0.0f ? x : 0.0f; }

  define float @relu(float %x) nounwind readnone {
  entry:
    %cmp = fcmp ogt float %x, 0.000000e+00  ; %cmp = (x > 0.0)  → i1
    ; --- two branches: positive vs zero ---
    br i1 %cmp, label %pos, label %zero

  pos:                                       ; x > 0: return x
    br label %merge

  zero:                                      ; x ≤ 0: return 0
    br label %merge

  merge:                                     ; phi merges the two paths
    %result = phi float [ %x,              %pos  ],
                        [ 0.000000e+00,    %zero ]
    ret float %result
  }

  ; After instcombine + simplifycfg, this collapses to:
  define float @relu_opt(float %x) nounwind readnone {
    %r = call float @llvm.maxnum.f32(float %x, float 0.0)
    ret float %r
  }
  ; → maxnum intrinsic compiles to a single VMAXSS instruction on x86
"""

print("  ReLU — SSA form and optimised form:")
print(RELU_IR)

LOOP_IR = """
  ; ── Elementwise add loop ─────────────────────────────────────────────
  ; C equivalent: for (int i=0; i<N; i++) c[i] = a[i] + b[i];

  define void @add_arrays(ptr %a, ptr %b, ptr %c, i32 %N) {
  entry:
    %n64 = sext i32 %N to i64           ; extend N to 64-bit for GEP
    br label %loop

  loop:
    %i = phi i64 [ 0, %entry ],         ; i starts at 0, increments each iter
                  [ %i_next, %loop ]
    ; Load a[i] and b[i]
    %a_ptr = getelementptr float, ptr %a, i64 %i   ; &a[i]
    %b_ptr = getelementptr float, ptr %b, i64 %i   ; &b[i]
    %ai    = load float, ptr %a_ptr                 ; a[i]
    %bi    = load float, ptr %b_ptr                 ; b[i]
    ; Compute and store
    %ci    = fadd float %ai, %bi                    ; a[i] + b[i]
    %c_ptr = getelementptr float, ptr %c, i64 %i   ; &c[i]
    store float %ci, ptr %c_ptr                     ; c[i] = a[i]+b[i]
    ; Increment and check loop bound
    %i_next = add i64 %i, 1
    %done   = icmp eq i64 %i_next, %n64
    br i1 %done, label %exit, label %loop

  exit:
    ret void
  }

  ; After loop-vectorize (with AVX-512, VF=16):
  ; The loop body becomes one iteration over <16 x float> vectors:
  ;   %va = load <16 x float>, ptr %a_ptr_aligned
  ;   %vb = load <16 x float>, ptr %b_ptr_aligned
  ;   %vc = fadd <16 x float> %va, %vb
  ;   store <16 x float> %vc, ptr %c_ptr_aligned
  ; → 16× throughput improvement per iteration
"""

print("  Elementwise loop — scalar and vectorised forms:")
print(LOOP_IR)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Build LLVM IR with llvmlite
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Building LLVM IR programmatically with llvmlite")
print("━" * 65)
print()

if HAS_LLVMLITE:
    float_t = ir.FloatType()
    int32_t = ir.IntType(32)
    int64_t = ir.IntType(64)

    module = ir.Module(name="ml_kernels")
    module.triple = binding.get_default_triple()

    # ── Function 1: ReLU ────────────────────────────────────────────────
    relu_fn = ir.Function(module,
                           ir.FunctionType(float_t, [float_t]),
                           name="relu")
    relu_fn.args[0].name = "x"
    x = relu_fn.args[0]

    entry_bb = relu_fn.append_basic_block("entry")
    pos_bb   = relu_fn.append_basic_block("positive")
    zero_bb  = relu_fn.append_basic_block("zero")
    merge_bb = relu_fn.append_basic_block("merge")

    b = ir.IRBuilder(entry_bb)
    cmp = b.fcmp_ordered(">", x, ir.Constant(float_t, 0.0), "cmp")
    b.cbranch(cmp, pos_bb, zero_bb)

    b = ir.IRBuilder(pos_bb)
    b.branch(merge_bb)

    b = ir.IRBuilder(zero_bb)
    b.branch(merge_bb)

    b = ir.IRBuilder(merge_bb)
    phi = b.phi(float_t, "result")
    phi.add_incoming(x,                      pos_bb)
    phi.add_incoming(ir.Constant(float_t, 0.0), zero_bb)
    b.ret(phi)

    # ── Function 2: dot product (scalar) ────────────────────────────────
    ptr_t   = ir.PointerType(float_t)
    dot_fn  = ir.Function(module,
                           ir.FunctionType(float_t, [ptr_t, ptr_t, int32_t]),
                           name="dot_product")
    dot_fn.args[0].name = "a"
    dot_fn.args[1].name = "b"
    dot_fn.args[2].name = "n"
    a_arg, b_arg, n_arg = dot_fn.args

    entry_d = dot_fn.append_basic_block("entry")
    loop_d  = dot_fn.append_basic_block("loop")
    exit_d  = dot_fn.append_basic_block("exit")

    b = ir.IRBuilder(entry_d)
    n64 = b.sext(n_arg, int64_t, "n64")
    b.branch(loop_d)

    b = ir.IRBuilder(loop_d)
    i_phi   = b.phi(int64_t, "i")
    acc_phi = b.phi(float_t, "acc")
    i_phi.add_incoming(ir.Constant(int64_t, 0),   entry_d)
    acc_phi.add_incoming(ir.Constant(float_t, 0.0), entry_d)

    a_ptr_i = b.gep(a_arg, [i_phi], inbounds=True, name="a_i_ptr")
    b_ptr_i = b.gep(b_arg, [i_phi], inbounds=True, name="b_i_ptr")
    ai      = b.load(a_ptr_i, "ai")
    bi      = b.load(b_ptr_i, "bi")
    prod    = b.fmul(ai, bi, "prod")
    new_acc = b.fadd(acc_phi, prod, "new_acc")
    i_next  = b.add(i_phi, ir.Constant(int64_t, 1), "i_next")
    done    = b.icmp_unsigned("==", i_next, n64, "done")

    i_phi.add_incoming(i_next,  loop_d)
    acc_phi.add_incoming(new_acc, loop_d)

    b.cbranch(done, exit_d, loop_d)

    b = ir.IRBuilder(exit_d)
    b.ret(acc_phi)

    print("  Generated LLVM IR module:")
    print()
    for line in str(module).split('\n'):
        print(f"    {line}")
    print()

    # ── Compile and run ──────────────────────────────────────────────────
    llvm_module = binding.parse_assembly(str(module))
    llvm_module.verify()

    # Create target machine and optimise
    target       = binding.Target.from_default_triple()
    target_machine = target.create_target_machine(opt=2)  # -O2

    # JIT compile
    backing_mod = binding.parse_assembly("")
    engine      = binding.create_mcjit_compiler(backing_mod, target_machine)
    mod_copy    = binding.parse_assembly(str(module))
    mod_copy.verify()
    engine.add_module(mod_copy)
    engine.finalize_object()
    engine.run_static_constructors()

    # Call the JIT-compiled relu
    import ctypes
    relu_ptr  = engine.get_function_address("relu")
    relu_cfn  = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)(relu_ptr)

    test_inputs = [-3.0, -0.5, 0.0, 0.5, 2.7]
    print("  JIT-compiled relu results:")
    for x_val in test_inputs:
        result = relu_cfn(x_val)
        expected = max(0.0, x_val)
        match = abs(result - expected) < 1e-6
        print(f"    relu({x_val:5.1f}) = {result:.4f}  expected={expected:.4f}  {'✅' if match else '❌'}")
    print()

    # Call JIT-compiled dot product
    dot_ptr  = engine.get_function_address("dot_product")
    dot_cfn  = ctypes.CFUNCTYPE(
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    )(dot_ptr)

    a_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_arr = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    a_c   = a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_c   = b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dot_result   = dot_cfn(a_c, b_c, 4)
    dot_expected = np.dot(a_arr, b_arr)
    print(f"  JIT dot([1,2,3,4], [5,6,7,8]) = {dot_result:.2f}  "
          f"numpy: {dot_expected:.2f}  "
          f"{'✅' if abs(dot_result-dot_expected)<1e-4 else '❌'}")
    print()

    # ── Benchmark JIT-compiled vs Python ────────────────────────────────
    N = 256
    a_bench = np.random.randn(N).astype(np.float32)
    b_bench = np.random.randn(N).astype(np.float32)
    a_c2    = a_bench.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_c2    = b_bench.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    REPS = 50000
    t0 = time.perf_counter()
    for _ in range(REPS): dot_cfn(a_c2, b_c2, N)
    t_jit = (time.perf_counter() - t0) / REPS * 1e6

    t0 = time.perf_counter()
    for _ in range(REPS): float(np.dot(a_bench, b_bench))
    t_np = (time.perf_counter() - t0) / REPS * 1e6

    print(f"  Dot product (N={N}), {REPS} reps:")
    print(f"    LLVM JIT:  {t_jit:.2f} μs")
    print(f"    numpy:     {t_np:.2f} μs")
    print(f"  (numpy uses BLAS which is highly tuned; raw LLVM IR dot")
    print(f"   is unoptimised — a baseline for comparison)")

else:
    print("  Install llvmlite to run live LLVM IR generation:")
    print("    pip install llvmlite   or   pip install numba")
    print()
    print("  llvmlite is used by:")
    print("    Numba:      JIT-compiles Python/NumPy → LLVM → native code")
    print("    Some TVM:   llvmlite for lightweight LLVM integration")
    print("    Research:   building custom compilers in Python")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · LLVM Passes — Optimisation Pipeline and Vectorisation": {
        "description": (
            "How LLVM's pass pipeline transforms code. "
            "Implement a simplified SSA construction (mem2reg) manually. "
            "Show what key passes do: instcombine, LICM, loop-vectorize. "
            "Demonstrate how ML compilers benefit from LLVM auto-vectorisation. "
            "Measure speedup from vectorised vs scalar loops on CPU."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import ctypes
import struct

print("=" * 65)
print("  LLVM PASSES — OPTIMISATION PIPELINE AND VECTORISATION")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: What key LLVM passes do — illustrated in Python
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Key LLVM passes illustrated with Python equivalents")
print("━" * 65)
print()

print("  LLVM passes transform code without changing its semantics.")
print("  Each pass has a specific narrow goal. They compose:")
print()

# ── mem2reg ───────────────────────────────────────────────────────────
print("  1. mem2reg — promote stack variables to SSA registers")
print()
print("  Before mem2reg (alloca + load/store pattern):")
print("    %x_addr = alloca float          ; reserve stack slot for x")
print("    store float %input, %x_addr     ; write to stack")
print("    %x_val  = load float, %x_addr   ; read from stack")
print("    %y      = fadd float %x_val, 1.0")
print()
print("  After mem2reg (stack variable promoted to virtual register):")
print("    %y = fadd float %input, 1.0     ; direct use — no memory ops")
print()
print("  WHY: loads/stores to local stack are unnecessary memory traffic.")
print("  mem2reg turns them into SSA phi nodes and virtual registers.")
print("  This pass runs FIRST in almost every pipeline.")
print()

# ── instcombine ───────────────────────────────────────────────────────
print("  2. instcombine — algebraic simplifications")
print()

class InstCombine:
    """
    Simplified Python simulation of instcombine transformations.
    Shows the pattern of simplification, not the implementation.
    """
    @staticmethod
    def simplify(expr):
        transformations = [
            ("x + 0",        "x"),
            ("x * 1",        "x"),
            ("x * 0",        "0"),
            ("x - x",        "0"),
            ("x * 2",        "x << 1   (integer shift is faster)"),
            ("x / 2.0",      "x * 0.5  (multiply cheaper than divide)"),
            ("(x+1)+2",      "x+3      (constant folding)"),
            ("!(x > y)",     "x <= y   (de Morgan: fewer instructions)"),
            ("x * -1",       "-x       (negation instead of multiply)"),
            ("(x > 0)?x:0",  "max(x,0) → maxnum intrinsic"),
        ]
        return [(before, after) for before, after in transformations
                if before == expr]

    @staticmethod
    def show_examples():
        optimisations = [
            ("x * 1",        "x",                "eliminate multiply"),
            ("x + 0",        "x",                "eliminate add"),
            ("(c1 + x) + c2","x + (c1+c2)",      "constant folding"),
            ("x * 2",        "x << 1",           "strength reduction"),
            ("x / 4.0",      "x * 0.25",         "division → multiply"),
            ("a == a",       "true (i1 1)",       "tautology elimination"),
            ("trunc(i64→i32) + trunc(i64→i32)",
             "trunc(i64 + i64)",                  "sink truncation"),
        ]
        print(f"  {'Before':35s}  →  {'After':25s}  ({'{}'}).".format("reason"))
        print(f"  {'─'*80}")
        for before, after, reason in optimisations:
            print(f"  {before:35s}  →  {after:25s}  ({reason})")

InstCombine.show_examples()
print()

# ── LICM ─────────────────────────────────────────────────────────────
print("  3. LICM — Loop Invariant Code Motion")
print()
print("  Before LICM:")
print("    for i in range(N):")
print("        scale = base_lr * decay_factor        # same every iteration!")
print("        weights[i] -= scale * gradients[i]")
print()
print("  After LICM (invariant computation hoisted out of loop):")
print("    scale = base_lr * decay_factor            # computed ONCE")
print("    for i in range(N):")
print("        weights[i] -= scale * gradients[i]")
print()

# Quantify LICM speedup
N = 100_000
base_lr = 0.01
decay   = 0.99

# Without LICM (recompute inside loop)
t0 = time.perf_counter()
weights_no_licm = np.random.randn(N).astype(np.float32)
grads           = np.random.randn(N).astype(np.float32)
for _ in range(1000):
    for i in range(min(N, 1000)):
        scale              = base_lr * decay    # "not hoisted"
        weights_no_licm[i] -= scale * grads[i]
t_no_licm = time.perf_counter() - t0

# With LICM (compute outside loop)
t0 = time.perf_counter()
weights_licm = np.random.randn(N).astype(np.float32)
for _ in range(1000):
    scale = base_lr * decay                     # hoisted
    for i in range(min(N, 1000)):
        weights_licm[i] -= scale * grads[i]
t_licm = time.perf_counter() - t0

print(f"  Benchmark (1000 × 1000-element weight update):")
print(f"    Without LICM: {t_no_licm*1000:.2f} ms")
print(f"    With LICM:    {t_licm*1000:.2f} ms")
print(f"    Speedup:      {t_no_licm/t_licm:.2f}×")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Auto-vectorisation — scalar to SIMD
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Auto-vectorisation: scalar loops to SIMD")
print("━" * 65)
print()

print("  LLVM's loop-vectorize converts scalar loops to SIMD instructions.")
print("  VF (Vector Factor) = how many elements per SIMD instruction.")
print()
print("  Hardware SIMD widths:")
print("    SSE2:    128-bit → VF=4  for float32 (4 floats at once)")
print("    AVX2:    256-bit → VF=8  for float32 (8 floats at once)")
print("    AVX-512: 512-bit → VF=16 for float32 (16 floats at once)")
print("    ARM NEON:128-bit → VF=4  for float32")
print("    ARM SVE: scalable → VF up to 16+ for float32")
print()

# Demonstrate vectorisation speedup with NumPy (which uses SIMD via BLAS/MKL)
# vs pure Python loops

N_VEC = 1_000_000
a = np.random.randn(N_VEC).astype(np.float32)
b = np.random.randn(N_VEC).astype(np.float32)

# "Scalar" path (Python loop — equivalent to unvectorised code)
SMALL = 10_000
a_small = a[:SMALL]
b_small = b[:SMALL]

t0 = time.perf_counter()
result_py = [a_small[i] + b_small[i] for i in range(SMALL)]
t_python = time.perf_counter() - t0

# "Vectorised" path (NumPy → SIMD via LLVM or BLAS)
t0 = time.perf_counter()
for _ in range(100):
    result_np = a + b
t_numpy = (time.perf_counter() - t0) / 100

# Elementwise ops NumPy uses
ops = ["relu (max)", "sigmoid (exp+div)", "add", "multiply", "sqrt"]
print(f"  Benchmark (N={N_VEC:,} float32 elements):")
print(f"  {'Operation':<25} | {'Scalar Python (ms)':>20} | {'NumPy SIMD (ms)':>17} | {'Speedup':>9}")
print(f"  {'─'*80}")

benchmarks = [
    ("elementwise add",      lambda: [a_small[i]+b_small[i] for i in range(SMALL)],
                              lambda: a + b),
    ("relu (max with 0)",     lambda: [max(a_small[i], 0.0) for i in range(SMALL)],
                              lambda: np.maximum(a, 0)),
    ("multiply-add",          lambda: [a_small[i]*b_small[i]+1 for i in range(SMALL)],
                              lambda: a * b + 1),
    ("sum reduction",         lambda: sum(float(x) for x in a_small),
                              lambda: a.sum()),
]

for label, scalar_fn, vector_fn in benchmarks:
    t0 = time.perf_counter()
    for _ in range(3): scalar_fn()
    t_s = (time.perf_counter() - t0) / 3 * 1000

    t0 = time.perf_counter()
    for _ in range(100): vector_fn()
    t_v = (time.perf_counter() - t0) / 100 * 1000

    speedup = t_s / t_v if t_v > 0 else 0
    print(f"  {label:<25} | {t_s:20.3f} | {t_v:17.4f} | {speedup:8.0f}×")

print()
print("  NumPy achieves these speedups via SIMD — the same mechanism LLVM")
print("  auto-vectorisation uses when TVM/XLA emit scalar LLVM IR.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Using llvmlite for JIT compilation with optimisation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — JIT compilation and optimisation levels")
print("━" * 65)
print()

try:
    from llvmlite import ir as llir, binding as llbind

    def build_relu_loop_module(optimise: bool = False):
        """Build an LLVM module with a vectorisable relu loop."""
        float_t = llir.FloatType()
        int64_t = llir.IntType(64)
        int32_t = llir.IntType(32)
        ptr_t   = llir.PointerType(float_t)

        module  = llir.Module(name="relu_loop")
        module.triple = llbind.get_default_triple()

        fn_ty   = llir.FunctionType(
            llir.VoidType(), [ptr_t, ptr_t, int32_t])
        fn      = llir.Function(module, fn_ty, name="relu_batch")
        fn.attributes.add("nounwind")
        inp, out, n_arg = fn.args
        inp.name, out.name, n_arg.name = "input", "output", "n"

        entry_bb = fn.append_basic_block("entry")
        loop_bb  = fn.append_basic_block("loop")
        exit_bb  = fn.append_basic_block("exit")

        b = llir.IRBuilder(entry_bb)
        n64 = b.sext(n_arg, int64_t)
        b.branch(loop_bb)

        b = llir.IRBuilder(loop_bb)
        i = b.phi(int64_t, "i")
        i.add_incoming(llir.Constant(int64_t, 0), entry_bb)

        inp_ptr_i = b.gep(inp, [i], inbounds=True)
        xi        = b.load(inp_ptr_i, "xi")
        relu_xi   = b.select(
            b.fcmp_ordered(">", xi, llir.Constant(float_t, 0.0), "cmp"),
            xi, llir.Constant(float_t, 0.0), "relu_xi")
        out_ptr_i = b.gep(out, [i], inbounds=True)
        b.store(relu_xi, out_ptr_i)

        i_next = b.add(i, llir.Constant(int64_t, 1), "i_next")
        i.add_incoming(i_next, loop_bb)
        done = b.icmp_unsigned("==", i_next, n64, "done")
        b.cbranch(done, exit_bb, loop_bb)

        b = llir.IRBuilder(exit_bb)
        b.ret_void()

        return module

    def compile_and_run(module, opt_level: int):
        target         = llbind.Target.from_default_triple()
        target_machine = target.create_target_machine(opt=opt_level)
        llvm_mod       = llbind.parse_assembly(str(module))
        llvm_mod.verify()

        # Apply passes
        pm = llbind.create_module_pass_manager()
        target_machine.add_analysis_passes(pm)
        pmb = llbind.create_pass_manager_builder()
        pmb.opt_level = opt_level
        pmb.populate(pm)
        pm.run(llvm_mod)

        # JIT
        back  = llbind.parse_assembly("")
        eng   = llbind.create_mcjit_compiler(back, target_machine)
        eng.add_module(llvm_mod)
        eng.finalize_object()
        eng.run_static_constructors()
        return eng

    N_TEST = 65536
    inp_arr = np.random.randn(N_TEST).astype(np.float32)
    out_arr = np.zeros(N_TEST,         dtype=np.float32)
    inp_c   = inp_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_c   = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    print(f"  ReLU batch (N={N_TEST:,}) at different optimisation levels:")
    print(f"  {'Opt level':>12} | {'Time (μs)':>12} | {'vs O0':>8}")
    print(f"  {'─'*38}")

    t_base = None
    for opt in [0, 1, 2, 3]:
        mod  = build_relu_loop_module()
        eng  = compile_and_run(mod, opt)
        fn_p = eng.get_function_address("relu_batch")
        fn_c = ctypes.CFUNCTYPE(None,
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.c_int)(fn_p)
        # Warmup
        for _ in range(5): fn_c(inp_c, out_c, N_TEST)
        # Time
        REPS = 200
        t0   = time.perf_counter()
        for _ in range(REPS): fn_c(inp_c, out_c, N_TEST)
        elapsed_us = (time.perf_counter() - t0) / REPS * 1e6
        if t_base is None: t_base = elapsed_us
        speedup = t_base / elapsed_us
        print(f"  {'-O' + str(opt):>12} | {elapsed_us:12.2f} | {speedup:7.2f}×")

    # Verify correctness
    expected = np.maximum(inp_arr, 0)
    fn_c(inp_c, out_c, N_TEST)
    correct = np.allclose(out_arr, expected, atol=1e-6)
    print()
    print(f"  Output correct: {correct} ✅")
    print()
    print("  -O3 includes: instcombine, GVN, LICM, loop-vectorize,")
    print("  SLP-vectorize, inlining, loop unrolling, and more.")
    print("  On x86 with AVX2: loop-vectorize generates 8-wide fadd at -O2+.")

except ImportError:
    print("  llvmlite not available — install with: pip install llvmlite")
    print()
    print("  JIT COMPILATION CONCEPT:")
    print("  opt_level=0: no optimisation (O0). Baseline. Debug mode.")
    print("  opt_level=1: basic (O1). mem2reg, instcombine, simplifycfg.")
    print("  opt_level=2: standard (O2). +LICM, GVN, loop-vectorize.")
    print("  opt_level=3: aggressive (O3). +inlining, unrolling, SLP.")
    print("  Typical ML speedup O0→O3 on a scalar loop: 5–20×.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · LLVM in the ML Stack — Numba, TVM Integration & Deployment": {
        "description": (
            "How ML systems use LLVM in practice. "
            "Numba: JIT-compiling Python/NumPy functions via LLVM. "
            "Inspect the LLVM IR Numba generates. "
            "Show the path from TVM's TIR to LLVM IR to native code. "
            "Cross-compilation for ARM from x86. "
            "How torch.compile's TorchInductor uses LLVM. "
            "Full stack diagram and component interaction."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  LLVM IN THE ML STACK — NUMBA, TVM AND TORCH.COMPILE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Numba — Python to LLVM to native code
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Numba: JIT-compiling Python via LLVM")
print("━" * 65)
print()

print("  Numba is the most accessible way to use LLVM from Python.")
print("  It JIT-compiles Python functions to native machine code.")
print("  Pipeline: Python bytecode → Numba IR → LLVM IR → native code")
print()

try:
    from numba import njit, prange, vectorize, float32
    import numba

    print(f"  Numba version: {numba.__version__}")
    print()

    # ── Basic JIT compilation ────────────────────────────────────────────
    @njit(cache=True)
    def relu_numba(x):
        """Numba JIT-compiled ReLU. Compiles to LLVM IR on first call."""
        return max(x, 0.0)

    @njit(cache=True)
    def dot_numba(a, b):
        """Dot product — compiled with SIMD by LLVM."""
        acc = 0.0
        for i in range(len(a)):
            acc += a[i] * b[i]
        return acc

    @njit(parallel=True, cache=True)
    def relu_batch_parallel(arr):
        """
        Parallel ReLU using prange.
        Numba splits the loop across CPU cores AND vectorises within each core.
        Two levels of parallelism: thread-level + SIMD.
        """
        result = np.empty_like(arr)
        for i in prange(len(arr)):           # prange = parallel range
            result[i] = max(arr[i], 0.0)
        return result

    # Trigger compilation (first call is slow — compiling)
    print("  Compiling Numba functions (first call triggers LLVM JIT)...")
    t0 = time.perf_counter()
    relu_numba(1.0)   # compile
    dot_numba(np.array([1.0]), np.array([1.0]))
    relu_batch_parallel(np.array([1.0, -1.0]))
    t_compile = time.perf_counter() - t0
    print(f"  Compilation time: {t_compile:.3f}s")
    print()

    # Benchmark
    N = 1_000_000
    arr = np.random.randn(N).astype(np.float64)
    a_v = np.random.randn(N).astype(np.float64)
    b_v = np.random.randn(N).astype(np.float64)

    # Warmup (use cached compilation)
    relu_batch_parallel(arr)
    np.maximum(arr, 0)

    REPS = 50
    t0   = time.perf_counter()
    for _ in range(REPS): _ = np.maximum(arr, 0)
    t_np = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS): _ = relu_batch_parallel(arr)
    t_numba = (time.perf_counter() - t0) / REPS * 1000

    print(f"  ReLU (N={N:,}):")
    print(f"    NumPy:           {t_np:.3f} ms")
    print(f"    Numba parallel:  {t_numba:.3f} ms")
    print()

    # Show the LLVM IR Numba generates
    print("  Inspecting Numba's LLVM IR output:")
    relu_numba_typed = numba.core.registry.cpu_target.typing_context

    # Get LLVM IR via Numba's inspection
    # Trigger float64 compilation
    relu_numba(1.0)  # float
    try:
        ir_text = relu_numba.inspect_llvm(sig=(numba.float64,))
        # Show just first few lines
        lines = [l for l in ir_text.split('\n') if l.strip()][:15]
        print()
        print("  First 15 lines of generated LLVM IR:")
        for line in lines:
            print(f"    {line}")
        print("    ...")
    except Exception:
        print("  (inspect_llvm() requires specific Numba version)")
        print("  Use: relu_numba.inspect_llvm(sig=(numba.float64,))")
    print()

    # ── Numba @vectorize — ufunc compilation ─────────────────────────────
    @vectorize(['float64(float64)', 'float32(float32)'], target='cpu')
    def relu_ufunc(x):
        """
        Numba vectorize: creates a NumPy ufunc backed by LLVM.
        Works on arrays, scalars, and broadcasts like NumPy.
        """
        return x if x > 0.0 else 0.0

    test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result_ufunc = relu_ufunc(test)
    print(f"  Numba ufunc ReLU([-2,-1,0,1,2]) = {result_ufunc}")
    print()

except ImportError:
    print("  Numba not installed: pip install numba")
    print()
    NUMBA_PATTERNS = """
  NUMBA PATTERNS (reference code):

  from numba import njit, prange, vectorize

  # 1. Basic JIT — just add @njit
  @njit
  def matrix_norm(A):
      total = 0.0
      for i in range(A.shape[0]):
          for j in range(A.shape[1]):
              total += A[i, j] ** 2
      return total ** 0.5

  # 2. Parallel loops — prange splits across CPU threads
  @njit(parallel=True)
  def batch_relu(x):
      out = np.empty_like(x)
      for i in prange(len(x)):
          out[i] = x[i] if x[i] > 0.0 else 0.0
      return out

  # 3. CUDA GPU kernel via Numba
  from numba import cuda
  @cuda.jit
  def relu_gpu(x, out):
      i = cuda.grid(1)
      if i < x.shape[0]:
          out[i] = x[i] if x[i] > 0.0 else 0.0

  # Numba pipeline:
  # Python function
  #   → Numba type inference (infer array shapes, dtypes)
  #   → Numba IR (Numba's own IR, like simplified Python bytecode)
  #   → LLVM IR (via llvmlite)
  #   → LLVM optimisation passes (-O3 equivalent)
  #   → Native machine code (x86, ARM, or CUDA PTX)
"""
    print(NUMBA_PATTERNS)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: TVM's use of LLVM
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — TVM → LLVM: the code generation path")
print("━" * 65)
print()

TVM_LLVM = """
  TVM CODEGEN PIPELINE (CPU path):

  ┌──────────────────────────────────────────────────────────────────┐
  │  Neural network model (ONNX / PyTorch / TF)                      │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓ Frontend import
  ┌──────────────────────────────────────────────────────────────────┐
  │  TVM Relay IR  (computation graph: ops + types)                  │
  │  Relay optimisations: op fusion, layout transform, constant fold │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓ Lowering
  ┌───────────────────────────────────────────────────────────────────┐
  │  TVM TIR (Tensor IR)  — loop nests, memory access patterns        │
  │  Schedule primitives: tile, vectorize, unroll, parallelize        │
  └──────────────────────┬────────────────────────────────────────────┘
                         ↓ LLVM Codegen
  ┌──────────────────────────────────────────────────────────────────┐
  │  LLVM IR  — scalar loops with getelementptr + load + store       │
  │  (TVM emits UNOPTIMISED scalar LLVM IR deliberately)             │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓ LLVM passes (opt_level=3)
  ┌──────────────────────────────────────────────────────────────────┐
  │  Optimised LLVM IR:                                              │
  │  - loop-vectorize: scalar loops → <8 x float> AVX2 vectors       │
  │  - LICM: loop-invariant code hoisted                             │
  │  - instcombine: algebraic simplifications                        │
  └──────────────────────┬───────────────────────────────────────────┘
                         ↓ LLVM backend
  ┌──────────────────────────────────────────────────────────────────┐
  │  Target-specific machine code:                                   │
  │  x86-64:   vmovaps ymm0, [mem]  (AVX2 loads)                     │
  │            vaddps  ymm0, ymm1   (AVX2 float adds)                │
  │  ARM:      ld1     {v0.4s}, [x0](NEON loads)                     │
  │            fadd    v0.4s, v1.4s (NEON adds)                      │
  │  RISC-V:   vle32.v v0, (a0)    (RVV vector loads)                │
  └──────────────────────────────────────────────────────────────────┘

  Key: TVM writes GENERIC scalar LLVM IR.
  LLVM handles ALL platform-specific vectorisation.
  This is how TVM supports 20+ hardware targets from one codebase.
"""
print(TVM_LLVM)

# Show a TVM LLVM usage example
try:
    import tvm
    from tvm import te, target as tvm_target

    print("  TVM available — generating a simple kernel and LLVM IR:")
    print()

    # Define a simple elementwise relu in TVM TE
    n   = te.var("n")
    A   = te.placeholder((n,), dtype="float32", name="A")
    B   = te.compute((n,), lambda i: te.max(A[i], 0), name="B")

    # Build for x86 with AVX2
    tgt = tvm_target.Target("llvm -mattr=+avx2")
    s   = te.create_schedule(B.op)

    # Apply vectorise schedule
    x    = B.op.axis[0]
    xo, xi = s[B].split(x, factor=8)      # split loop for vectorisation
    s[B].vectorize(xi)                      # vectorise inner loop (width=8)
    s[B].parallel(xo)                       # parallelise outer loop

    # Compile
    func = tvm.build(s, [A, B], target=tgt, name="relu_avx2")
    print(f"  TVM kernel compiled for: {tgt}")
    print(f"  Schedule: split(8) → vectorize inner → parallel outer")
    print()

    # Get LLVM IR
    llvm_ir = func.get_source("ll")  # get LLVM IR as text
    lines   = [l for l in llvm_ir.split('\n') if l.strip() and not l.startswith(';')][:20]
    print("  Generated LLVM IR (first 20 non-comment lines):")
    for line in lines:
        print(f"    {line}")
    print("    ...")
    print()

    # Benchmark
    dev = tvm.cpu(0)
    N_K = 1_000_000
    a_tvm = tvm.nd.array(np.random.randn(N_K).astype(np.float32), dev)
    b_tvm = tvm.nd.array(np.zeros(N_K, dtype=np.float32), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=100)
    t_tvm     = evaluator(a_tvm, b_tvm).mean * 1000

    np_arr = a_tvm.numpy()
    t0     = time.perf_counter()
    for _ in range(100): np.maximum(np_arr, 0)
    t_np   = (time.perf_counter() - t0) / 100 * 1000

    print(f"  ReLU (N={N_K:,}) benchmark:")
    print(f"    TVM (AVX2, parallel): {t_tvm:.4f} ms")
    print(f"    NumPy:               {t_np:.4f} ms")

except ImportError:
    print("  TVM not available (pip install apache-tvm)")
    print("  See TVM module for full TVM → LLVM code generation details.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: torch.compile and LLVM
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — torch.compile's use of LLVM (TorchInductor)")
print("━" * 65)
print()

TORCH_INDUCTOR = """
  torch.compile() pipeline (CPU path):

  Python model code
       ↓  TorchDynamo captures bytecode
  FX Graph (PyTorch's IR — computation graph)
       ↓  TorchInductor backend selected (default for CPU)
  TorchInductor generates C++ (with OpenMP for parallelism)
  OR   TorchInductor generates Triton IR (for GPU)
       ↓  For C++ path:
  Clang compiles C++ → LLVM IR
       ↓  LLVM pass pipeline (O2)
  Native machine code (x86 / ARM)

  Inspecting what torch.compile generates:

    import torch, os
    os.environ["TORCH_COMPILE_DEBUG"] = "1"     # enable debug output
    os.environ["TORCHINDUCTOR_TRACE"] = "1"     # trace inductor

    @torch.compile
    def my_model(x):
        return torch.relu(x) + x

    my_model(torch.randn(1024))
    # This prints the generated C++ or Triton code AND the LLVM IR

  Or use torch._dynamo.explain() to see the captured graphs:
    explanation = torch._dynamo.explain(my_model)(torch.randn(1024))
    print(explanation)
"""
print(TORCH_INDUCTOR)

try:
    import torch

    @torch.compile(backend="inductor")
    def fused_ops(x):
        return torch.relu(x) * 0.5 + x

    X_tc = torch.randn(100_000)
    # Warmup (compiles on first call)
    _ = fused_ops(X_tc)

    REPS = 500
    t0 = time.perf_counter()
    for _ in range(REPS): _ = fused_ops(X_tc)
    t_compiled = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS): _ = torch.relu(X_tc) * 0.5 + X_tc
    t_eager = (time.perf_counter() - t0) / REPS * 1000

    print(f"  fused_ops(N={X_tc.numel():,}): relu(x)*0.5 + x")
    print(f"    Eager:    {t_eager:.4f} ms")
    print(f"    Compiled: {t_compiled:.4f} ms")
    print(f"    Speedup:  {t_eager/t_compiled:.2f}×")
    print()
    print("  torch.compile fuses relu + multiply + add into ONE kernel,")
    print("  eliminating intermediate tensor allocations.")
    print("  On CPU this goes through TorchInductor → C++ → Clang → LLVM.")
except ImportError:
    print("  PyTorch not available for this demo.")

print()
print("━" * 65)
print("  LLVM SUMMARY: THE UNIVERSAL COMPILER SUBSTRATE")
print("━" * 65)
print()
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ Framework           │ How LLVM is used                       │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ PyTorch             │ TorchInductor → C++/Triton → Clang/LLVM│")
print("  │ TensorFlow / XLA    │ XLA LLVM emitter → LLVM → x86/ARM      │")
print("  │ Apache TVM          │ TIR → LLVM codegen → any CPU target    │")
print("  │ MLIR                │ Lowers to LLVM dialect → LLVM backend  │")
print("  │ Numba               │ Python → llvmlite → LLVM JIT           │")
print("  │ CUDA (NVCC/Clang)   │ Device code → NVVM (LLVM) → PTX        │")
print("  │ ROCm (AMD)          │ HIP code → LLVM AMDGCN backend → GCN   │")
print("  │ Julia               │ Julia JIT → LLVM IR → native code      │")
print("  │ Rust                │ MIR → LLVM IR → native code            │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ What LLVM provides to ALL of them:                           │")
print("  │   Register allocation for target CPU/GPU registers           │")
print("  │   SIMD auto-vectorisation (SSE/AVX/NEON/SVE)                 │")
print("  │   Loop optimisations (LICM, unroll, vectorize)               │")
print("  │   Algebraic simplifications (instcombine)                    │")
print("  │   JIT compilation (ORC JIT / MCJIT)                          │")
print("  │   Target portability (20+ ISAs from one IR)                  │")
print("  └──────────────────────────────────────────────────────────────┘")
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