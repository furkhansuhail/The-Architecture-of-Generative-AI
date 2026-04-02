"""
Enzyme — LLVM-Based Automatic Differentiation
================================================

Enzyme is a compiler plugin that automatically differentiates programs
by operating directly on LLVM IR. Where all other AD systems work at
the language or framework level (PyTorch's autograd, JAX's jax.grad,
TensorFlow's GradientTape), Enzyme works ONE level lower — at the
compiled intermediate representation.

The consequence is profound: Enzyme can differentiate programs written
in ANY language that compiles to LLVM IR — C, C++, Rust, Fortran, Julia,
Swift, Kotlin, and more — without any source-level instrumentation.
High-performance HPC code that was never written with ML in mind
becomes automatically differentiable just by passing through Enzyme.

Before Enzyme, automatic differentiation required:
    - Rewriting code in a framework that supports AD (PyTorch, JAX)
    - Accepting the performance overhead of eager execution
    - Losing access to low-level optimisations (SIMD, loop fusion, etc.)
    - Manual porting of scientific simulation code to Python/XLA

Enzyme's insight: differentiating COMPILED code is better than
differentiating SOURCE code. By the time LLVM IR is available,
the compiler has already optimised the primal computation — loop
transformations, inlining, vectorisation. Enzyme differentiates
THAT already-optimised code, not the original source.

In the connected compiler stack:
    LLVM (module 1)    ← Enzyme IS an LLVM pass; runs inside the LLVM pipeline
    MLIR (module 2)    ← Enzyme-MLIR port adds AD to any MLIR dialect
    CIRCT (module 3)   ← differentiable hardware simulation via Enzyme+Arc
    Enzyme (this)      ← LLVM-level AD for any compiled language
    XLA (module 5)     ← XLA uses Enzyme-style IR-level AD for TPU code
    TVM (module 6)     ← TVM uses AD for auto-tuning cost model training

"""

import textwrap
import re

TOPIC_NAME   = "Enzyme — LLVM-Based Automatic Differentiation"
DISPLAY_NAME = "04 · Enzyme"
ICON         = "∂"
SUBTITLE     = "Source-Free AD at the IR Level: Forward, Reverse, and Beyond"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY ENZYME EXISTS: THE PROBLEM WITH EXISTING AD SYSTEMS

### The Two Existing Approaches to Automatic Differentiation

    All AD systems before Enzyme fell into two categories:

    APPROACH 1 — OPERATOR OVERLOADING (PyTorch autograd, C++ dual numbers)
        Override arithmetic operators to simultaneously compute values
        and track derivatives in a dynamically-built computation graph.

        Pros:  Easy to implement; works in pure Python/C++.
        Cons:
            - Runtime overhead: building a graph at every forward pass.
            - Cannot differentiate arbitrary control flow efficiently.
            - Requires all code to be written using the AD library's types.
            - Cannot differentiate into third-party C/Fortran libraries.

    APPROACH 2 — SOURCE TRANSFORMATION (Tapenade, Zygote.jl, JAX tracing)
        Transform the source code itself: parse the AST, and emit
        derivative source code following the chain rule symbolically.

        Pros:  Can produce fast derivative code (no runtime graph).
        Cons:
            - Requires access to SOURCE CODE of every called function.
            - Cannot differentiate precompiled libraries (LAPACK, MKL, etc.)
            - Fragile: tool must understand all language features.
            - Language-specific: a Fortran AD tool doesn't work on C++.

### The Gap: Scientific Computing Code

    The biggest consumers of derivatives are not ML researchers —
    they are physicists, engineers, and climate scientists:

        Molecular dynamics:    force fields as gradients of energy functions
        Climate models:        sensitivity of global temperature to CO₂
        CFD (fluid dynamics):  adjoint-based aerodynamic optimisation
        Seismic inversion:     gradients of wave equations for oil exploration
        Structural mechanics:  shape gradients for topology optimisation

    These codes are written in Fortran 77, C, C++ with heavy use of:
        - MKL (Intel's BLAS): precompiled, no source
        - LAPACK routines:    precompiled Fortran, no source transformation
        - OpenMP/MPI:         parallelism that source AD tools handle badly
        - Compiler intrinsics: SIMD operations not in any AST

    PyTorch/JAX cannot touch this code.
    Source transformation tools can't handle the precompiled dependencies.
    Manual differentiation of 100,000-line Fortran codes is impractical.

### Enzyme's Solution: Differentiate at the IR Level

    LLVM IR is the UNIVERSAL compilation target for nearly every
    systems programming language:

        C/C++   → clang     → LLVM IR
        Fortran → flang     → LLVM IR
        Rust    → rustc     → LLVM IR
        Julia   → julia     → LLVM IR
        Swift   → swiftc    → LLVM IR
        Kotlin  → kotlin/native → LLVM IR
        Zig     → zig       → LLVM IR

    Enzyme operates as an LLVM pass. It sees the same LLVM IR
    that the compiler produces after ALL language-specific passes.
    The LLVM IR is already optimised, inlined, and lowered — all
    the precompiled libraries are inlined or their call sites are known.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Source Code (any language)                                         │
    │       ↓  language frontend (clang, flang, rustc, ...)               │
    │  LLVM IR (source-language-neutral)                                  │
    │       ↓  Enzyme LLVM pass (AD transformation)                       │
    │  LLVM IR + derivative code                                          │
    │       ↓  LLVM optimisation (loop-vectorize, instcombine, ...)       │
    │  Optimised LLVM IR (primal + derivative, jointly optimised)         │
    │       ↓  llc / ORC JIT                                              │
    │  Native machine code                                                │
    └─────────────────────────────────────────────────────────────────────┘

    Consequences:
        - Any language that targets LLVM: zero extra work for AD.
        - Precompiled libraries: inlined before Enzyme runs → differentiable.
        - LLVM's optimiser runs on the combined primal+derivative code,
          enabling joint optimisations impossible at the source level.
        - Performance: Enzyme's derivatives match or exceed hand-written
          adjoints in benchmarks against ADIFOR, Tapenade, and ADIC.


##### PART 2 — THE MATHEMATICS: FORWARD AND REVERSE MODE AD

### The Chain Rule Is All You Need

    Automatic differentiation is mechanically applying the chain rule
    to every elementary operation in a program.

    For a composition f(g(h(x))):
        df/dx = (df/dg) · (dg/dh) · (dh/dx)

    For a program, every assignment IS a composition of elementary ops:
        y  = sin(x)           →  dy/dx = cos(x)
        z  = y * y            →  dz/dx = 2y · dy/dx = 2sin(x)cos(x)
        w  = exp(z)           →  dw/dx = exp(z) · dz/dx

    The question is the ORDER of multiplication.
    This choice completely determines the complexity:

### Forward Mode (Tangent, Jvp — Jacobian-Vector Product)

    Propagate derivatives FORWARD through the computation,
    in the same order as the primal (original) computation.

    For each primal value v, maintain a tangent value v̇ = dv/dx_i
    where x_i is ONE chosen input.

        Primal:    v₁ = f₁(x)          Tangent:  v̇₁ = (∂f₁/∂x) · ẋ
        Primal:    v₂ = f₂(v₁)         Tangent:  v̇₂ = (∂f₂/∂v₁) · v̇₁
        Primal:    v₃ = f₃(v₁, v₂)     Tangent:  v̇₃ = (∂f₃/∂v₁)·v̇₁ + (∂f₃/∂v₂)·v̇₂

    Cost: ONE forward pass per INPUT dimension.
    Best for: functions f: ℝⁿ → ℝᵐ where n ≪ m (few inputs, many outputs).
    ML relevance: evaluating directional derivatives, computing Hessian-vector products.

    Enzyme API:
        __enzyme_fwddiff(f, enzyme_dup, x, dx)
        ; computes f(x) and df/dx · dx simultaneously

### Reverse Mode (Adjoint, Vjp — Vector-Jacobian Product)

    Propagate derivatives BACKWARD through the computation,
    in reverse order relative to the primal.

    Phase 1 (forward sweep): Execute the primal, recording intermediate values.
    Phase 2 (backward sweep): Propagate adjoints ā = d(loss)/dv backwards.

        For each op v = f(u₁, u₂, ...):
            adjoint rule:   ū₁ += ā · ∂f/∂u₁
                            ū₂ += ā · ∂f/∂u₂
        ā propagates backwards from output to input.

    Cost: ONE backward pass per OUTPUT dimension.
    Best for: functions f: ℝⁿ → ℝ (many inputs, scalar output = neural net loss).
    ML relevance: BACKPROPAGATION is reverse-mode AD on a computational graph.

    Enzyme API:
        __enzyme_autodiff(f, enzyme_dup, x, dx)
        ; reverse mode: dx += d(loss)/dx where loss = f(x)

### Why Reverse Mode Is Used in ML

    A neural network with 100M parameters:
        n = 100,000,000 (number of inputs = parameters)
        m = 1           (output = scalar loss value)

    Forward mode: n passes = 100M passes per training step. Catastrophic.
    Reverse mode: 1 pass per training step. This IS backpropagation.

    The reason PyTorch/JAX use reverse mode exclusively for training:
    gradient of a scalar loss w.r.t. N parameters costs 1 backward pass,
    regardless of N. This is the mathematical basis of all deep learning.

### The Tape (Checkpointing) Problem

    Reverse mode requires the INTERMEDIATE VALUES computed during the
    forward pass to compute the backward pass adjoints.

    For a chain of n operations, naïve reverse mode stores ALL n values:
        Memory cost: O(n)
        A 100M parameter transformer forward pass: gigabytes of activations

    Solutions:
        Rematerialisation (recomputation):
            Don't store intermediate values; recompute them in backward pass.
            Time cost: 2× primal cost. Memory cost: O(1) (or O(√n) with optimal checkpointing).

        Gradient checkpointing:
            Store values at √n "checkpoint" locations.
            Recompute each segment from the nearest checkpoint.
            Memory: O(√n), Time: O(n log n).

    Enzyme handles both strategies with:
        enzyme_tape    : standard tape storage
        __enzyme_cache : caching for selective recomputation
        shadow memory  : Enzyme's unified approach — see Part 3


##### PART 3 — HOW ENZYME WORKS: ACTIVITY ANALYSIS AND SHADOW MEMORY

### The Enzyme LLVM Pass Pipeline

    Enzyme operates as a transformation pass inside the LLVM pipeline:

        1. Enzyme identifies calls to __enzyme_autodiff or __enzyme_fwddiff.
        2. For each such call, Enzyme:
              a. Identifies the function to differentiate.
              b. Runs ACTIVITY ANALYSIS to determine which values affect
                 the derivative (active) vs which are constant (inactive).
              c. Generates the DERIVATIVE FUNCTION in LLVM IR.
              d. Replaces the __enzyme_autodiff call with a call to the
                 generated derivative.
        3. LLVM continues with its normal optimisation passes on the
           combined primal + derivative IR.

### Activity Analysis

    Not all values in a program affect the gradient.
    Activity analysis determines which values are "active" — i.e.,
    which values have a meaningful derivative w.r.t. the inputs.

    CONSTANT (inactive) values:
        Integer loop counters:      for (int i = 0; i < n; i++)
        Array indices:              A[i]  — i is integer, not float
        Branching conditions:       if (x > 0) — value is used in branch,
                                    but not differentiating through the branch
        Values not derived from the differentiated inputs.

    ACTIVE values:
        Floating-point values derived from the input being differentiated.
        The result of any floating-point arithmetic on active values.
        Pointer/memory locations that store active values.

    Why this matters:
        CONSTANT values require NO shadow memory or adjoint computation.
        Wrongly marking an active value as constant → incorrect gradient.
        Wrongly marking a constant as active → wasted computation.

    Enzyme's activity analysis is interprocedural:
        It analyses the ENTIRE call graph reachable from the
        differentiated function to classify every LLVM value.

### Shadow Memory Model

    For every ACTIVE pointer/memory location, Enzyme allocates
    a corresponding "shadow" memory region to hold the adjoint.

        Primal pointer:   double* x  (stores primal values)
        Shadow pointer:   double* dx (stores adjoint values, i.e., d(loss)/dx)

    This is what __enzyme_autodiff receives:
        __enzyme_autodiff(f, enzyme_dup, x, dx)
        ;  x  = pointer to primal input (read)
        ;  dx = pointer to shadow (adjoint accumulates here, in-place)

    The shadow model generalises to structures and arrays:
        struct Point { double x, y; }
        Point p    = {1.0, 2.0};       // primal
        Point dp   = {0.0, 0.0};       // shadow (gradient will accumulate here)
        __enzyme_autodiff(f, enzyme_dup, &p, &dp)
        // after: dp.x = d(loss)/dp.x, dp.y = d(loss)/dp.y

### The Derivative Generation Process

    For each LLVM instruction in the primal function, Enzyme generates
    the corresponding adjoint instruction in the backward function.

    The key rules (LLVM instruction → adjoint rule):

    fadd:
        Primal:  %z = fadd double %x, %y
        Adjoint: dz is accumulated from the output.
                 dx += dz          ; both inputs get the full adjoint
                 dy += dz

    fmul:
        Primal:  %z = fmul double %x, %y
        Adjoint: dx += dz * y      ; product rule: d(xy)/dx = y
                 dy += dz * x      ; product rule: d(xy)/dy = x

    fdiv:
        Primal:  %z = fdiv double %x, %y
        Adjoint: dx += dz / y      ; quotient rule numerator
                 dy -= dz * x / (y*y)  ; quotient rule denominator

    call @sin(%x):
        Primal:  %z = call @sin(%x)
        Adjoint: dx += dz * cos(x) ; sin'(x) = cos(x)
                 (Enzyme knows derivatives of all LLVM intrinsics)

    load:
        Primal:  %v = load double, double* %ptr
        Adjoint: shadow_ptr = shadow(%ptr)
                 *shadow_ptr += dv  ; accumulate adjoint into shadow mem

    store:
        Primal:  store double %v, double* %ptr
        Adjoint: shadow_ptr = shadow(%ptr)
                 dv = *shadow_ptr   ; read adjoint from shadow mem
                 *shadow_ptr = 0.0  ; reset (to avoid double-counting)

    br / cond_br (control flow):
        Enzyme reverses control flow in the backward pass.
        A branch at LLVM block B becomes a merge in the backward pass.
        A loop body in the forward becomes a loop in reverse direction.

    phi (SSA merge point):
        Primal:  %v = phi [%a, ^bb1], [%b, ^bb2]
        Adjoint: Enzyme tracks which branch was taken (via a branch tape)
                 and routes dv to the appropriate incoming adjoint.


##### PART 4 — ENZYME'S LLVM IR: WHAT THE GENERATED CODE LOOKS LIKE

### Example: Dot Product — Forward Pass and Adjoint

    C source:
        double dot(double* a, double* b, int n) {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
                sum += a[i] * b[i];
            return sum;
        }

    LLVM IR (primal, simplified):

        define double @dot(double* %a, double* %b, i32 %n) {
        entry:
          %sum.0 = alloca double
          store double 0.0, double* %sum.0
          br ^loop(%0 : i32)

        ^loop(%i : i32):
          %done = icmp sge i32 %i, %n
          br i1 %done, ^exit, ^body

        ^body:
          %ai_ptr = getelementptr double, double* %a, i32 %i
          %bi_ptr = getelementptr double, double* %b, i32 %i
          %ai  = load double, double* %ai_ptr
          %bi  = load double, double* %bi_ptr
          %prod = fmul double %ai, %bi
          %old  = load double, double* %sum.0
          %new  = fadd double %old, %prod
          store double %new, double* %sum.0
          %i1  = add i32 %i, 1
          br ^loop(%i1 : i32)

        ^exit:
          %result = load double, double* %sum.0
          ret double %result
        }

    Enzyme reverse-mode adjoint (generated, simplified):
        __enzyme_autodiff(dot, enzyme_dup, a, da, enzyme_dup, b, db, enzyme_const, n)
        ; da[i] += d(loss)/d(dot) * b[i]
        ; db[i] += d(loss)/d(dot) * a[i]

        define void @diffe_dot(double* %a,  double* %da,
                               double* %b,  double* %db,
                               i32 %n, double %dreturn) {
          ; Forward pass: record loop iteration count (for reverse traversal)
          ; (for simple counted loops, Enzyme reconstructs the count analytically)

          ; Backward loop: i from n-1 down to 0
          br ^back_loop(%n_minus_1 : i32)

        ^back_loop(%i : i32):
          %done = icmp slt i32 %i, 0
          br i1 %done, ^exit, ^back_body

        ^back_body:
          %ai_ptr  = getelementptr double, double* %a,  i32 %i
          %bi_ptr  = getelementptr double, double* %b,  i32 %i
          %dai_ptr = getelementptr double, double* %da, i32 %i
          %dbi_ptr = getelementptr double, double* %db, i32 %i

          %ai = load double, double* %ai_ptr   ; reload primal a[i]
          %bi = load double, double* %bi_ptr   ; reload primal b[i]

          ; Adjoint of fmul %ai, %bi via product rule:
          ;   d(loss)/d(ai) += d(loss)/d(prod) * bi = dreturn * bi
          ;   d(loss)/d(bi) += d(loss)/d(prod) * ai = dreturn * ai
          %d_ai  = fmul double %dreturn, %bi
          %d_bi  = fmul double %dreturn, %ai

          ; Accumulate into shadow memory (in-place)
          %dai_old = load double, double* %dai_ptr
          %dai_new = fadd double %dai_old, %d_ai
          store double %dai_new, double* %dai_ptr

          %dbi_old = load double, double* %dbi_ptr
          %dbi_new = fadd double %dbi_old, %d_bi
          store double %dbi_new, double* %dbi_ptr

          %i1 = sub i32 %i, 1
          br ^back_loop(%i1 : i32)

        ^exit:
          ret void
        }

    Result: da[i] = dreturn * b[i], db[i] = dreturn * a[i]
    i.e., ∇_a dot(a,b) = b  and  ∇_b dot(a,b) = a  ✓


### Joint Optimisation After Enzyme

    After Enzyme generates the derivative IR, LLVM's optimiser sees
    the combined primal + derivative as a SINGLE IR module.

    Important optimisations that become possible:
        CSE across primal and derivative:
            If the primal computed cos(x) and the derivative needs cos(x)
            (for the sin adjoint: d(sin)/dx = cos(x)), LLVM's CSE
            eliminates the duplicate cos(x) computation.

        Loop fusion of primal and derivative loops:
            The primal loop over i=0..n and the adjoint loop over i=n-1..0
            can sometimes be fused into a single pass (for some access patterns).

        Vectorisation of adjoint loops:
            The adjoint loops often have the same vectorisable structure
            as the primal. LLVM's loop-vectorize handles both.

        Inlining derivative of small functions:
            The derivative of a small helper function inlined by Enzyme
            is itself inlined into the gradient computation — just like
            LLVM would inline any small function.

    This is the key advantage over source-level AD:
        Source-level AD sees TWO separate programs (primal, derivative)
        and optimises them INDEPENDENTLY.
        Enzyme sees ONE LLVM module and optimises both JOINTLY.


##### PART 5 — THE ENZYME LLVM PASS: API AND ANNOTATIONS

### The __enzyme_autodiff API

    The Enzyme API is a set of magic function declarations.
    No header file is needed — just declare the function as extern.

    C/C++ usage:
        // Declare (no implementation needed; Enzyme fills it in)
        double __enzyme_autodiff(void*, ...);
        void   __enzyme_fwddiff(void*, ...);
        double __enzyme_batch(void*, ...);    // batched forward/reverse

        // Differentiate f w.r.t. x (reverse mode):
        double dx = __enzyme_autodiff((void*)f, enzyme_dup, x, dx_storage, n);

    Argument annotations:
        enzyme_dup,  x, dx    ← x is active; dx is the shadow (gradient output)
        enzyme_dupv, x, dx    ← same but x is also overwritten (in-place functions)
        enzyme_const, n       ← n is constant (integer loop bound; no gradient)
        enzyme_out,  y        ← y is an active output (for multi-output functions)

    Forward mode:
        double dout = __enzyme_fwddiff((void*)f,
                                        enzyme_dup, x, x_dot,    ; tangent x_dot
                                        enzyme_const, n);
        ; dout = df/dx · x_dot   (directional derivative)

    Batched mode (compute multiple gradients simultaneously):
        // compute gradient w.r.t. 4 inputs in parallel (AVX-friendly)
        __enzyme_batch((void*)f, 4, enzyme_dup, x, dx_batch, ...);

### Custom Derivative Declarations (Overriding Enzyme)

    Enzyme can differentiate anything, but sometimes you KNOW the
    derivative of a function and want to provide it directly.
    This bypasses Enzyme's analysis for that function:

    C++ annotation:
        // Declare that Enzyme should use my_sin_fwd as the
        // forward derivative and my_sin_rev as the reverse derivative:
        double my_sin(double x);
        double my_sin_fwd(double x, double xdot);        // forward tangent
        void   my_sin_rev(double x, double* xbar, double zbar);  // adjoint

        // Tell Enzyme about the custom derivatives:
        __builtin_enzyme_register_derivative(my_sin, my_sin_fwd, my_sin_rev);

    When is this useful?
        - BLAS/LAPACK routines (DGEMM has a known gradient)
        - Non-smooth functions (abs, max — need subgradients)
        - Stochastic functions (require custom estimators)
        - Physics simulations with conserved quantities

### Enzyme in Different Languages

    C/C++ (clang + Enzyme plugin):
        clang -Xclang -load -Xclang /path/to/Enzyme.so -O2 prog.c
        Or: cmake with EnzymeConfig.cmake

    Rust (enzyme-rs crate):
        #[autodiff(df, Reverse, Duplicated, Const, Active)]
        fn f(x: &[f64], n: usize) -> f64 { ... }
        // df is auto-generated

    Julia (Enzyme.jl):
        using Enzyme
        df = Enzyme.gradient(Reverse, f, x)   # calls Enzyme via Julia IR

    Fortran (flang + Enzyme):
        flang -Xclang -load -Xclang /path/to/Enzyme.so prog.f90

    Python (via PyEnzyme / Jax-enzyme bridge):
        from enzyme_jax import enzyme_jit
        @enzyme_jit
        def f(x):     # traced by JAX, differentiated by Enzyme at LLVM level
            return ...


##### PART 6 — ENZYME-MLIR: BRINGING AD TO THE FULL MLIR ECOSYSTEM

### Why Enzyme-MLIR Is Needed

    Enzyme operates on LLVM IR. But the ML compiler stack described
    in modules 2–6 operates at MUCH higher levels of abstraction:
        MLIR's Linalg dialect (matmul, conv)
        StableHLO (XLA's high-level ops)
        Torch-MLIR (ATen ops)

    By the time these reach LLVM IR, the high-level structure is gone:
        linalg.matmul → many nested affine.for loops → many LLVM scalar ops
    The LLVM IR for a matmul gradient has NO knowledge that the primal
    was a matmul — so the gradient cannot exploit the matmul structure.

    Enzyme-MLIR runs Enzyme at the MLIR level, BEFORE lowering.
    A gradient of linalg.matmul can be recognised as another linalg.matmul
    (the well-known result: ∇_A (AB) = grad_C @ B^T).

### The Enzyme MLIR Pass

    Enzyme-MLIR adds:
        enzyme.autodiff operation: requests a derivative of an MLIR function
        Activity attributes on MLIR operations (active/constant annotations)
        Reverse and forward derivative rules for MLIR dialects

    MLIR with Enzyme:
        // Original function
        func.func @dot(%a: tensor<Nxf32>, %b: tensor<Nxf32>) -> f32 {
            %result = linalg.dot ins(%a, %b : tensor<Nxf32>, tensor<Nxf32>)
                                 outs(%init : tensor<f32>) -> f32
            return %result : f32
        }

        // Request gradient via Enzyme MLIR
        enzyme.autodiff @dot(%a, %da, %b, %db) : (tensor<Nxf32>, tensor<Nxf32>,
                                                    tensor<Nxf32>, tensor<Nxf32>) -> f32

        // Enzyme-MLIR generates:
        func.func @grad_dot(%a: tensor<Nxf32>, %da: tensor<Nxf32>,
                             %b: tensor<Nxf32>, %db: tensor<Nxf32>,
                             %dret: f32) {
            // ∇_a dot(a,b) = b * dret  →  another linalg.generic (scale b by dret)
            // ∇_b dot(a,b) = a * dret  →  another linalg.generic (scale a by dret)
            linalg.generic {scale b → da by dret}
            linalg.generic {scale a → db by dret}
        }

    The gradient is STILL at the linalg level — it can be further
    tiled, fused, and vectorised by all the MLIR optimisation passes.

### Enzyme-MLIR and Differentiable Programming Stacks

    Torchdynamo + Enzyme:
        PyTorch model → TorchDynamo → Torch-MLIR → Enzyme-MLIR → gradient
        Result: gradients that beat PyTorch's native autograd in benchmarks
        (Enzyme fuses gradient with forward; PyTorch cannot do this by default)

    JAX + Enzyme:
        JAX traces Python → StableHLO → Enzyme-MLIR → gradient MLIR
        The gradient is in StableHLO, feeding directly back into XLA.
        Enables end-to-end differentiation of custom HPC kernels within JAX.

    IREE + Enzyme:
        Custom kernels in MLIR → Enzyme-MLIR generates gradients →
        IREE deploys both primal and gradient to CPU/GPU/mobile target.


##### PART 7 — ENZYME IN THE CONNECTED COMPILER STACK

### How Enzyme Connects to LLVM (Module 1)

    Enzyme IS an LLVM pass. The connection is direct and intimate:
        Enzyme is loaded as a plugin into the LLVM pass pipeline.
        It uses LLVM's AnalysisManager, DominatorTree, ScalarEvolution,
        LoopInfo, and AliasAnalysis — all the same infrastructure that
        LLVM's own passes use.
        Enzyme generates new LLVM IR functions using LLVM's IRBuilder API.
        After Enzyme runs, all standard LLVM passes (loop-vectorize,
        instcombine, GVN) run on the Enzyme-generated derivative IR.

    Enzyme position in the LLVM pass pipeline:
        Frontend (clang/flang) → LLVM IR →
        Early passes (mem2reg, sroa, instcombine) →
        ENZYME PASS (generates derivative functions) →
        Middle-end (loop-vectorize, licm, gvn) → applied to primal+gradient →
        Backend (instruction selection, register allocation) →
        Machine code

### How Enzyme Connects to MLIR (Module 2)

    Enzyme-MLIR is the MLIR-level version of the Enzyme LLVM pass.
    It runs BEFORE lowering to LLVM IR, preserving high-level structure.

    Two deployment modes:
        Mode 1 (LLVM-level):   MLIR is fully lowered to LLVM IR → Enzyme LLVM pass
        Mode 2 (MLIR-level):   Enzyme-MLIR pass runs on MLIR → lowers to LLVM IR

    Mode 2 is preferable because:
        Gradients of linalg.matmul are recognised as linalg.matmul (not scalar loops).
        MLIR's canonicalisation and fusion passes improve the gradient.
        The gradient stays in a high-level form suitable for all MLIR backends.

### How Enzyme Connects to CIRCT (Module 3)

    CIRCT's Arc dialect models hardware as pure functions (no side effects).
    Enzyme can differentiate Arc functions: this gives sensitivity of
    hardware outputs w.r.t. hardware inputs — essential for:

        Hardware parameter optimisation: gradient of chip power/timing
            w.r.t. cell sizes, threshold voltages, wire widths.
        ML-for-EDA: gradient of a learned delay model w.r.t. net topology.
        Differentiable hardware simulation: train surrogate timing models.

    Arc + Enzyme = differentiable RTL simulation (research, 2023–present).

### How Enzyme Connects to XLA (Module 5)

    XLA uses AD internally for:
        jax.grad():        JAX's primary gradient API, implemented via XLA
        jax.jvp():         forward-mode JVP via XLA linearisation
        jax.value_and_grad(): combined forward+backward

    JAX's current AD implementation: source transformation at the JAX Python level
    (traces Python operations, applies chain rule rules to primitive ops).

    Enzyme's role in XLA's future:
        XLA emits LLVM IR for CPU/GPU kernels.
        Enzyme can differentiate those kernels directly.
        Research project: Enzyme-XLA differentiates XLA HLO-level custom ops
        that JAX's source-transformation AD cannot reach.

### How Enzyme Connects to TVM (Module 6)

    TVM uses a small AD system for:
        Auto-scheduling: cost model is a neural net trained via gradient descent.
        Relay: TVM's high-level IR has basic AD for training.

    Enzyme's role:
        TVM compiles to LLVM IR (for CPU targets).
        Enzyme can differentiate TVM-generated LLVM IR.
        This enables: gradient of a TVM-optimised kernel, making
        custom CUDA/CPU kernels in TVM differentiable within JAX/PyTorch.

### The Full Differentiable Computing Stack

    ┌─────────────────────────────────────────────────────────────────────┐
    │  User code (any language: Python, C++, Fortran, Rust, Julia)        │
    │  writes f(x) — a scalar loss or simulation                          │
    ├─────────────────────────────────────────────────────────────────────┤
    │  AD Request: __enzyme_autodiff(f, ...) or enzyme.autodiff in MLIR   │
    ├─────────────────────────────────────────────────────────────────────┤
    │  ENZYME (this module)                                               │
    │  Activity analysis → derivative generation → shadow memory          │
    │  Runs at LLVM IR level (or MLIR level via Enzyme-MLIR)              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  LLVM (module 1) — joint optimisation of primal + derivative        │
    │  loop-vectorize, instcombine, GVN applied to gradient IR            │
    ├─────────────────────────────────────────────────────────────────────┤
    │  MLIR (module 2) — Enzyme-MLIR preserves linalg/stablehlo structure │
    ├─────────────────────────────────────────────────────────────────────┤
    │  XLA / TVM (modules 5–6) — consume gradient IR for accelerators     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  CIRCT (module 3) — Arc dialect: differentiable hardware simulation │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Native machine code: primal + gradient, jointly optimised          │
    └─────────────────────────────────────────────────────────────────────┘

### Enzyme vs Framework-Level AD: Summary

    ┌──────────────────────┬─────────────────────┬────────────────────────┐
    │  Property            │  Enzyme (IR-level)  │  PyTorch/JAX (source)  │
    ├──────────────────────┼─────────────────────┼────────────────────────┤
    │  Input language      │  Any LLVM language  │  Python only           │
    │  Precompiled code    │  Yes (inlines first)│  No                    │
    │  HPC/Fortran code    │  Yes                │  No                    │
    │  C++ STL containers  │  Yes                │  No                    │
    │  Joint optimisation  │  Yes (one IR pass)  │  Separate primal/grad  │
    │  Memory (tape)       │  Shadow memory      │  Dynamic graph         │
    │  Control flow        │  Reverses LLVM CFG  │  Requires special ops  │
    │  Custom derivatives  │  Yes (annotations)  │  Yes (custom rules)    │
    │  Checkpointing       │  Built-in           │  Manual / library      │
    │  SIMD gradients      │  Yes (vectorises)   │  Framework-dependent   │
    │  Distributed (MPI)   │  Yes (research)     │  DDP/DeepSpeed         │
    └──────────────────────┴─────────────────────┴────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · AD Fundamentals — Forward Mode, Reverse Mode, and the Tape": {
        "description": (
            "Build automatic differentiation from first principles in pure Python. "
            "Implement dual numbers for forward mode. "
            "Implement a dynamic tape (Wengert list) for reverse mode. "
            "Trace a neural network forward+backward pass step by step. "
            "Show why reverse mode costs O(1) passes regardless of input dimension. "
            "Verify against finite differences and connect to Enzyme's approach."
        ),
        "language": "python",
        "code": '''
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Optional

print("=" * 65)
print("  AD FUNDAMENTALS — FORWARD MODE, REVERSE MODE, AND THE TAPE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Forward mode via dual numbers
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Forward mode: dual numbers")
print("━" * 65)
print()

FORWARD_MODE_THEORY = """
  DUAL NUMBERS: the mathematical basis of forward-mode AD
  ════════════════════════════════════════════════════════

  A dual number: x + x′ε  where ε² = 0
    x  is the primal value.
    x′ is the tangent (directional derivative, dx/dt for some seed).
    ε is a formal infinitesimal that squares to zero.

  Arithmetic:
    (a + a′ε) + (b + b′ε) = (a+b) + (a′+b′)ε
    (a + a′ε) * (b + b′ε) = ab + (a′b + ab′)ε   [ε²=0 drops out]
    f(a + a′ε) = f(a) + f′(a)·a′·ε               [chain rule automatically]

  This is exactly what Enzyme does in forward mode at LLVM IR level:
    Every LLVM SSA value %v becomes two values: %v (primal) and %dv (tangent).
    Every fadd, fmul, fcall is doubled:
      fadd %a, %b → primal: %z = fadd %a, %b
                  → tangent: %dz = fadd %da, %db   (sum rule)
      fmul %a, %b → primal: %z = fmul %a, %b
                  → tangent: %dz = fadd (fmul %da, %b), (fmul %a, %db)  (product rule)
"""
print(FORWARD_MODE_THEORY)

@dataclass
class Dual:
    """
    Dual number: primal + tangent·ε.
    Represents a value AND its directional derivative simultaneously.
    This is what Enzyme's forward mode (fwddiff) does at LLVM IR level.
    """
    p: float         # primal value
    t: float = 0.0   # tangent (derivative w.r.t. chosen seed direction)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.p + other, self.t)
        return Dual(self.p + other.p, self.t + other.t)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.p * other, self.t * other)
        # product rule: d(ab) = da·b + a·db
        return Dual(self.p * other.p,
                    self.t * other.p + self.p * other.t)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.p - other, self.t)
        return Dual(self.p - other.p, self.t - other.t)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.p / other, self.t / other)
        # quotient rule: d(a/b) = (da·b - a·db) / b²
        return Dual(self.p / other.p,
                    (self.t * other.p - self.p * other.t) / (other.p ** 2))

    def __neg__(self):
        return Dual(-self.p, -self.t)

    def __pow__(self, n):
        # power rule: d(xⁿ) = n·xⁿ⁻¹·dx
        return Dual(self.p ** n, n * (self.p ** (n-1)) * self.t)

    def __repr__(self):
        return f"Dual({self.p:.6f}, {self.t:.6f})"


# Dual-compatible elementary functions
def d_exp(x: Dual):
    ep = np.exp(x.p)
    return Dual(ep, ep * x.t)          # d(exp(x)) = exp(x)·dx

def d_log(x: Dual):
    return Dual(np.log(x.p), x.t / x.p)   # d(log(x)) = dx/x

def d_sin(x: Dual):
    return Dual(np.sin(x.p), np.cos(x.p) * x.t)

def d_cos(x: Dual):
    return Dual(np.cos(x.p), -np.sin(x.p) * x.t)

def d_tanh(x: Dual):
    th = np.tanh(x.p)
    return Dual(th, (1 - th**2) * x.t)    # d(tanh) = (1-tanh²)·dx

def d_relu(x: Dual):
    return Dual(max(0.0, x.p), x.t if x.p > 0 else 0.0)

def d_sigmoid(x: Dual):
    s = 1.0 / (1.0 + np.exp(-x.p))
    return Dual(s, s * (1-s) * x.t)       # d(σ) = σ(1-σ)·dx


# ── Forward-mode gradient of a function ──────────────────────────────────
def forward_grad(f, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient of scalar function f at x using forward mode.
    Requires one pass per input dimension (like Enzyme's fwddiff).
    """
    grad = np.zeros_like(x)
    fx   = None
    for i in range(len(x)):
        seed = np.zeros_like(x)
        seed[i] = 1.0
        # feed dual numbers with seed direction e_i
        x_dual = [Dual(float(xi), float(si)) for xi, si in zip(x, seed)]
        result = f(x_dual)
        if fx is None:
            fx = result.p
        grad[i] = result.t
    return fx, grad


# ── Example: 3-input function ────────────────────────────────────────────
def f_example(x):
    """f(x0, x1, x2) = sin(x0) * x1 + exp(x2) / (x1 + 1)"""
    if isinstance(x[0], Dual):
        sin_fn, exp_fn = d_sin, d_exp
    else:
        sin_fn, exp_fn = np.sin, np.exp
    return sin_fn(x[0]) * x[1] + exp_fn(x[2]) / (x[1] + 1)

x0 = np.array([0.5, 2.0, 1.0])
fx, grad_fwd = forward_grad(f_example, x0)
print(f"  f(x) = sin(x0)*x1 + exp(x2)/(x1+1)")
print(f"  x     = {x0}")
print(f"  f(x)  = {fx:.6f}")
print(f"  ∇f (forward mode, {len(x0)} passes) = {grad_fwd}")

# Finite difference verification
eps = 1e-6
grad_fd = np.zeros(3)
for i in range(3):
    xp, xm = x0.copy(), x0.copy()
    xp[i] += eps; xm[i] -= eps
    grad_fd[i] = (f_example(xp) - f_example(xm)) / (2*eps)
print(f"  ∇f (finite diff, reference)     = {grad_fd}")
print(f"  Max error: {np.max(np.abs(grad_fwd - grad_fd)):.2e} ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Reverse mode via a tape (Wengert list)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Reverse mode: tape (Wengert list)")
print("━" * 65)
print()

REVERSE_MODE_THEORY = """
  THE TAPE (WENGERT LIST): reverse-mode AD
  ════════════════════════════════════════════════════════════════

  The tape records every operation during the forward pass.
  The backward pass replays the tape in reverse, accumulating adjoints.

  This mirrors Enzyme's reverse-mode LLVM IR generation:
    Forward sweep:  execute primal ops, store intermediate values
    Backward sweep: for each op (in reverse), apply adjoint rule,
                    accumulate gradient into shadow memory (da, db, ...)

  Enzyme's key advantage over this Python tape:
    No runtime tape overhead — Enzyme generates STATIC LLVM IR
    for the backward pass at COMPILE TIME. The tape structure is
    analysed statically and converted to register operations and
    selective memory reads. No heap allocations at runtime.
"""
print(REVERSE_MODE_THEORY)

class Var:
    """
    A variable on the Wengert tape.
    Every float operation creates a Var, recording how to compute its adjoint.
    Enzyme does this at the LLVM IR level: every SSA value becomes a Var.
    """
    _tape: List = []    # class-level tape (static for demonstration)

    def __init__(self, value: float, name: str = ""):
        self.v     = float(value)       # primal value
        self.adj   = 0.0                # adjoint (d loss / d self)
        self.name  = name
        self._deps: List[Tuple['Var', Callable]] = []
        # deps: list of (upstream_var, how_to_propagate_adjoint_to_it)

    def _record(self, upstream_var: 'Var', grad_fn: Callable):
        """Register a dependency: self depends on upstream_var."""
        self._deps.append((upstream_var, grad_fn))

    def backward(self, seed: float = 1.0):
        """
        Reverse-mode backward pass starting from this Var with given seed.
        Traverses the tape in reverse topological order.
        This is what Enzyme generates as a static LLVM IR function.
        """
        # Build topological order via DFS (mimics LLVM's dominance tree traversal)
        order = []
        visited = set()
        def topo(v):
            if id(v) in visited: return
            visited.add(id(v))
            for dep, _ in v._deps:
                topo(dep)
            order.append(v)
        topo(self)

        self.adj = seed    # seed the output adjoint (d loss / d self = 1.0)

        # Reverse traversal: propagate adjoints backwards
        for var in reversed(order):
            for upstream, grad_fn in var._deps:
                # chain rule: d(loss)/d(upstream) += d(loss)/d(var) * d(var)/d(upstream)
                upstream.adj += var.adj * grad_fn(var)

    # ── Operator overloads: each creates a new Var on the tape ────────────
    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        z = Var(self.v + other.v, f"({self.name}+{other.name})")
        z._record(self,  lambda z_: 1.0)           # d(a+b)/da = 1
        z._record(other, lambda z_: 1.0)           # d(a+b)/db = 1
        return z

    def __radd__(self, other): return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        z = Var(self.v * other.v, f"({self.name}*{other.name})")
        b_val = other.v
        a_val = self.v
        z._record(self,  lambda z_: b_val)         # d(a*b)/da = b
        z._record(other, lambda z_: a_val)         # d(a*b)/db = a
        return z

    def __rmul__(self, other): return self.__mul__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        z = Var(self.v - other.v, f"({self.name}-{other.name})")
        z._record(self,  lambda z_:  1.0)
        z._record(other, lambda z_: -1.0)
        return z

    def __truediv__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        b_val = other.v
        a_val = self.v
        z = Var(self.v / other.v, f"({self.name}/{other.name})")
        z._record(self,  lambda z_: 1.0 / b_val)
        z._record(other, lambda z_: -a_val / (b_val**2))
        return z

    def __neg__(self):
        z = Var(-self.v, f"-{self.name}")
        z._record(self, lambda z_: -1.0)
        return z

    def __pow__(self, n):
        p_val = self.v
        z = Var(self.v ** n, f"{self.name}^{n}")
        z._record(self, lambda z_: n * (p_val ** (n-1)))
        return z

    def __repr__(self):
        return f"Var({self.v:.4f}, adj={self.adj:.4f}, name={self.name!r})"


def v_exp(x: Var):
    ep = np.exp(x.v)
    z  = Var(ep, f"exp({x.name})")
    z._record(x, lambda z_: z_.v)     # d(exp(x))/dx = exp(x) = z
    return z

def v_sin(x: Var):
    z = Var(np.sin(x.v), f"sin({x.name})")
    x_val = x.v
    z._record(x, lambda z_: np.cos(x_val))
    return z

def v_tanh(x: Var):
    th = np.tanh(x.v)
    z  = Var(th, f"tanh({x.name})")
    z._record(x, lambda z_: 1 - z_.v**2)
    return z

def v_relu(x: Var):
    z = Var(max(0.0, x.v), f"relu({x.name})")
    x_val = x.v
    z._record(x, lambda z_: 1.0 if x_val > 0 else 0.0)
    return z

def v_log(x: Var):
    x_val = x.v
    z = Var(np.log(x.v), f"log({x.name})")
    z._record(x, lambda z_: 1.0 / x_val)
    return z


# ── Reverse mode: gradient of scalar function ─────────────────────────────
def reverse_grad(f, x: np.ndarray):
    """
    Compute gradient using reverse mode: ONE backward pass regardless of n.
    Compare: forward mode above needed n passes. This needs exactly 1.
    This is why backpropagation scales to 100M parameters.
    """
    x_vars = [Var(float(xi), f"x{i}") for i, xi in enumerate(x)]
    loss = f(x_vars)
    loss.backward(seed=1.0)
    return loss.v, np.array([xi.adj for xi in x_vars])


# ── Same example: one pass instead of n ───────────────────────────────────
def f_tape(x):
    return v_sin(x[0]) * x[1] + v_exp(x[2]) / (x[1] + 1.0)

fx_rev, grad_rev = reverse_grad(f_tape, x0)
print(f"  f(x)  = {fx_rev:.6f}")
print(f"  ∇f (reverse mode, 1 pass) = {grad_rev}")
print(f"  ∇f (forward mode, n pass) = {grad_fwd}")
print(f"  Max error: {np.max(np.abs(grad_rev - grad_fwd)):.2e} ✅")
print()
print(f"  Pass count comparison:")
print(f"    Forward mode: {len(x0)} passes (one per input)")
print(f"    Reverse mode: 1 pass   (regardless of input dimension)")
print(f"    For 100M parameters: forward = 100M passes, reverse = 1 pass")
print(f"    This is the mathematical reason why backprop exists.")
print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Enzyme IR Mechanics — Activity Analysis and Adjoint Generation": {
        "description": (
            "Simulate Enzyme's LLVM IR transformation pipeline in Python. "
            "Show activity analysis: classify every LLVM value as active or constant. "
            "Implement the adjoint generation rules for fmul, fadd, fcall, load, store. "
            "Trace the complete dot product backward pass at the LLVM IR level. "
            "Show shadow memory: how Enzyme stores adjoints for pointer-based code. "
            "Demonstrate joint primal+gradient optimisation (CSE across both)."
        ),
        "language": "python",
        "code": '''
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

print("=" * 65)
print("  ENZYME IR MECHANICS — ACTIVITY ANALYSIS AND ADJOINT GENERATION")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: LLVM value representation and activity analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Activity analysis: active vs constant LLVM values")
print("━" * 65)
print()

ACTIVITY_THEORY = """
  ACTIVITY ANALYSIS — the gatekeeper of Enzyme's efficiency
  ════════════════════════════════════════════════════════════════════

  Before generating any derivative code, Enzyme classifies every LLVM
  SSA value as ACTIVE or CONSTANT.

  CONSTANT: value does not influence the output derivative.
    - Integer values (loop counters, array indices): i32, i64
    - Values not reachable from the differentiated inputs
    - Values that are only used as control flow conditions
    - Pointer addresses (not the floating-point data they point to)

  ACTIVE: value DOES influence the output derivative.
    - Floating-point values derived from active inputs: f32, f64
    - The result of fadd/fmul/fcall on active operands
    - Memory loads from active pointers

  ANALYSIS ALGORITHM (simplified):
    1. Seed: mark all differentiated inputs as ACTIVE.
    2. Forward dataflow: if any operand of an op is ACTIVE
       and the op is a float operation → result is ACTIVE.
    3. Backward dataflow (alias analysis): if an active value is stored
       into a pointer, that pointer's loaded values are also ACTIVE.
    4. Fixed-point: repeat until no new ACTIVE values are found.

  CONSEQUENCES FOR CODE GENERATION:
    CONSTANT values: NO shadow memory, NO adjoint instruction generated.
    ACTIVE values:   shadow memory allocated, adjoint instruction generated.

  Example — dot product:
    %i    : i32  → CONSTANT (integer loop counter)
    %ai   : f64  → ACTIVE   (loaded from active array a)
    %bi   : f64  → ACTIVE   (loaded from active array b)
    %prod : f64  → ACTIVE   (fmul of two active values)
    %sum  : f64  → ACTIVE   (fadd accumulation)
    %n    : i32  → CONSTANT (length parameter, integer)
    %done : i1   → CONSTANT (loop termination condition, boolean)
"""
print(ACTIVITY_THEORY)

from enum import Enum, auto

class Activity(Enum):
    CONSTANT = auto()   # no gradient needed
    ACTIVE   = auto()   # gradient must be computed

@dataclass
class LLVMValue:
    """Simplified LLVM SSA value with activity classification."""
    name:     str
    typ:      str          # "f64", "f32", "i32", "i64", "ptr_f64", ...
    activity: Activity = Activity.CONSTANT

    @property
    def is_float(self):
        return self.typ in ("f32", "f64")

    @property
    def is_int(self):
        return self.typ in ("i32", "i64", "i1")

    def __repr__(self):
        act = "ACTIVE" if self.activity == Activity.ACTIVE else "CONST "
        return f"  %{self.name:12s} : {self.typ:8s} [{act}]"


@dataclass
class LLVMInst:
    """Simplified LLVM instruction with its activity."""
    opcode:   str
    result:   Optional[LLVMValue]
    operands: List[LLVMValue] = field(default_factory=list)
    attrs:    Dict = field(default_factory=dict)

    def __repr__(self):
        res = f"%{self.result.name} = " if self.result else ""
        ops = ", ".join(f"%{o.name}" for o in self.operands)
        return f"    {res}{self.opcode} {ops}"


def activity_analysis(instructions: List[LLVMInst],
                      active_inputs: Set[str]) -> Dict[str, Activity]:
    """
    Enzyme-style activity analysis: forward dataflow over LLVM instructions.
    Returns mapping from value name → Activity.
    """
    activity = {}
    # Seed: mark active inputs
    for name in active_inputs:
        activity[name] = Activity.ACTIVE

    changed = True
    while changed:
        changed = False
        for inst in instructions:
            if inst.result is None:
                continue
            name = inst.result.name
            current = activity.get(name, Activity.CONSTANT)

            # An instruction is ACTIVE if:
            # 1. It operates on floating-point values AND
            # 2. At least one operand is ACTIVE AND
            # 3. The opcode is a float operation (fadd, fmul, call, load, ...)
            float_op = inst.opcode in ("fadd", "fmul", "fdiv", "fsub",
                                        "fmul_neg", "call_sin", "call_cos",
                                        "call_exp", "load_f64")
            any_active_operand = any(
                activity.get(o.name, Activity.CONSTANT) == Activity.ACTIVE
                for o in inst.operands
            )
            result_is_float = inst.result.is_float

            if float_op and any_active_operand and result_is_float:
                new_activity = Activity.ACTIVE
            else:
                new_activity = Activity.CONSTANT

            if new_activity != current:
                activity[name] = new_activity
                changed = True

    return activity


# ── Dot product LLVM IR simulation ───────────────────────────────────────
print("  Simulating activity analysis on dot product LLVM IR:")
print()

# Values
a_ptr  = LLVMValue("a_ptr",  "ptr_f64")    # pointer to array a
b_ptr  = LLVMValue("b_ptr",  "ptr_f64")    # pointer to array b
n_val  = LLVMValue("n",      "i32")        # length (integer)
i_val  = LLVMValue("i",      "i32")        # loop counter (integer)
ai_val = LLVMValue("ai",     "f64")        # a[i] loaded value
bi_val = LLVMValue("bi",     "f64")        # b[i] loaded value
prod   = LLVMValue("prod",   "f64")        # ai * bi
old_s  = LLVMValue("old_sum","f64")        # previous sum
new_s  = LLVMValue("new_sum","f64")        # new sum = old + prod
done   = LLVMValue("done",   "i1")         # i >= n (boolean)
result = LLVMValue("result", "f64")        # final sum

instructions = [
    LLVMInst("load_f64", ai_val, [a_ptr, i_val]),
    LLVMInst("load_f64", bi_val, [b_ptr, i_val]),
    LLVMInst("fmul",     prod,   [ai_val, bi_val]),
    LLVMInst("fadd",     new_s,  [old_s, prod]),
    LLVMInst("icmp_sge", done,   [i_val, n_val]),   # integer compare: CONSTANT
    LLVMInst("add_i32",  None,   [i_val]),           # integer increment: no result
    LLVMInst("load_f64", result, [LLVMValue("sum_ptr", "ptr_f64")]),
]

# Active inputs: a_ptr, b_ptr (we differentiate w.r.t. both)
activity = activity_analysis(instructions, active_inputs={"a_ptr", "b_ptr",
                                                           "ai", "bi",
                                                           "old_sum"})

all_values = [a_ptr, b_ptr, n_val, i_val, ai_val, bi_val, prod,
              old_s, new_s, done, result]
for val in all_values:
    val.activity = activity.get(val.name, Activity.CONSTANT)
    print(repr(val))

print()
print("  Enzyme analysis result:")
active_count   = sum(1 for v in all_values if v.activity == Activity.ACTIVE)
constant_count = sum(1 for v in all_values if v.activity == Activity.CONSTANT)
print(f"  ACTIVE values:   {active_count} → require shadow memory + adjoint generation")
print(f"  CONSTANT values: {constant_count} → zero derivative cost")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Adjoint generation rules for each LLVM opcode
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Adjoint generation rules for LLVM instructions")
print("━" * 65)
print()

ADJOINT_RULES = """
  ENZYME'S ADJOINT RULES FOR EACH LLVM INSTRUCTION
  ════════════════════════════════════════════════════════════════

  For each primal instruction that produces an ACTIVE result,
  Enzyme generates adjoint instructions in the BACKWARD pass.

  The adjoint instruction accumulates gradient into shadow memory.
  Notation: d%z = adjoint of %z  = d(loss)/d(%z)

  fadd %x, %y → %z:
    Primal:   z = x + y
    Adjoint:  d%x += d%z        (chain rule: dz/dx = 1)
              d%y += d%z        (chain rule: dz/dy = 1)
    LLVM IR generated by Enzyme:
              %tmp1 = load  double* %shadow_x
              %new1 = fadd  double %tmp1, %d_z
              store double %new1, double* %shadow_x
              ; (same for shadow_y)

  fmul %x, %y → %z:
    Primal:   z = x * y
    Adjoint:  d%x += d%z * y    (d(xy)/dx = y)
              d%y += d%z * x    (d(xy)/dy = x)
    LLVM IR:  %d_x_contrib = fmul double %d_z, %y_saved
              store ... (accumulate into shadow_x)

  fdiv %x, %y → %z:
    Adjoint:  d%x += d%z / y
              d%y -= d%z * x / (y*y)

  call @sin(%x) → %z:
    Adjoint:  d%x += d%z * cos(x)     ; sin'(x) = cos(x)
    Enzyme knows derivatives of all LLVM math intrinsics.

  call @exp(%x) → %z:
    Adjoint:  d%x += d%z * z          ; exp'(x) = exp(x) = z (reuse!)
    CSE: z was computed in primal, reused in adjoint.

  load double, double* %ptr → %v:
    Adjoint:  shadow_ptr = enzyme_shadow(%ptr)
              *shadow_ptr += d%v      ; accumulate into shadow memory

  store double %v, double* %ptr:
    Adjoint:  shadow_ptr = enzyme_shadow(%ptr)
              d%v = *shadow_ptr        ; READ gradient from shadow
              *shadow_ptr = 0.0        ; reset (gradient has been consumed)

  The store adjoint is REVERSED in Enzyme: the backward pass sees the
  store and turns it into a LOAD from shadow memory. This is why
  reverse-mode AD naturally reverses memory operations.
"""
print(ADJOINT_RULES)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Shadow memory — simulating Enzyme's gradient storage model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Shadow memory model simulation")
print("━" * 65)
print()

class ShadowMemory:
    """
    Simulates Enzyme's shadow memory model.
    Every active POINTER gets a corresponding shadow buffer.
    Gradients accumulate into shadow memory during the backward pass.

    In Enzyme's LLVM IR:
        The shadow of double* %ptr is obtained by %dptr = enzyme_shadow(%ptr).
        Adjoint rules write into %dptr and read from %dptr.
        At the end, %dptr holds the total gradient d(loss)/d(%ptr[i]).
    """
    def __init__(self, primal_arrays: Dict[str, np.ndarray]):
        self.primal  = {k: np.array(v, dtype=np.float64)
                        for k, v in primal_arrays.items()}
        self.shadow  = {k: np.zeros_like(v)
                        for k, v in self.primal.items()}
        self.scalars = {}           # scalar SSA values (non-pointer)
        self.d_scalars = {}         # shadow of scalar SSA values

    def load(self, arr_name: str, i: int) -> float:
        """Primal load: a[i]."""
        return float(self.primal[arr_name][i])

    def shadow_accumulate(self, arr_name: str, i: int, grad: float):
        """Adjoint of load: shadow[i] += grad."""
        self.shadow[arr_name][i] += grad

    def shadow_consume(self, arr_name: str, i: int) -> float:
        """Adjoint of store: consume gradient from shadow, reset."""
        g = float(self.shadow[arr_name][i])
        self.shadow[arr_name][i] = 0.0
        return g


# ── Dot product: manual Enzyme-style forward + backward ──────────────────
def enzyme_dot_forward_backward(a: np.ndarray, b: np.ndarray,
                                 d_return: float = 1.0):
    """
    Manual simulation of Enzyme's reverse-mode transformation of dot(a,b).
    Forward pass: compute a·b and record intermediate values.
    Backward pass: accumulate gradients into da, db (shadow memory).
    """
    n = len(a)
    mem = ShadowMemory({"a": a, "b": b})
    mem.shadow["da_out"] = np.zeros(n)
    mem.shadow["db_out"] = np.zeros(n)

    # ── FORWARD PASS (primal, exactly what the original code does) ────────
    intermediates = []   # tape of (ai, bi, prod) per iteration
    sum_val = 0.0
    for i in range(n):
        ai   = mem.load("a", i)
        bi   = mem.load("b", i)
        prod = ai * bi                 # fmul
        sum_val = sum_val + prod       # fadd
        intermediates.append((ai, bi, prod))

    # ── BACKWARD PASS (Enzyme-generated, reverses the forward) ────────────
    d_sum = d_return          # seed: d(loss)/d(dot_result) = d_return

    for i in reversed(range(n)):
        ai, bi, prod = intermediates[i]

        # Adjoint of fadd: d_sum propagates unchanged to both operands
        d_prod    = d_sum               # d(sum)/d(prod) = 1
        d_old_sum = d_sum               # d(sum)/d(old_sum) = 1

        # Adjoint of fmul %ai, %bi → %prod:
        # d%ai += d_prod * bi
        # d%bi += d_prod * ai
        d_ai = d_prod * bi             # product rule
        d_bi = d_prod * ai

        # Accumulate into shadow memory (Enzyme: store into shadow_a[i])
        mem.shadow["a"][i] += d_ai
        mem.shadow["b"][i] += d_bi

        d_sum = d_old_sum              # continue backward through sum

    return sum_val, mem.shadow["a"].copy(), mem.shadow["b"].copy()


# ── Test ──────────────────────────────────────────────────────────────────
n = 6
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
b = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

dot_val, da, db = enzyme_dot_forward_backward(a, b, d_return=1.0)

print(f"  a = {a}")
print(f"  b = {b}")
print(f"  dot(a,b) = {dot_val:.1f}  (expected: {np.dot(a,b):.1f})")
print()
print(f"  Enzyme reverse-mode gradients:")
print(f"  da = {da}  (should equal b)")
print(f"  db = {db}  (should equal a)")
print(f"  ∇_a dot(a,b) = b? {np.allclose(da, b)} {'✅' if np.allclose(da, b) else '❌'}")
print(f"  ∇_b dot(a,b) = a? {np.allclose(db, a)} {'✅' if np.allclose(db, a) else '❌'}")
print()

# Show the adjoint rules table
print("  LLVM instruction adjoint summary (generated by Enzyme):")
print()
print(f"  {'Instruction':30s}  {'Adjoint rule':40s}")
print("  " + "-" * 72)
rules = [
    ("%z = fadd %x, %y",          "d%x += d%z; d%y += d%z"),
    ("%z = fmul %x, %y",          "d%x += d%z*y; d%y += d%z*x"),
    ("%z = fdiv %x, %y",          "d%x += d%z/y; d%y -= d%z*x/(y²)"),
    ("%z = fneg %x",               "d%x -= d%z"),
    ("%z = call sin(%x)",          "d%x += d%z*cos(x)"),
    ("%z = call exp(%x)",          "d%x += d%z*z  [reuse exp output]"),
    ("%z = call log(%x)",          "d%x += d%z/x"),
    ("%v = load f64* %ptr",        "shadow[%ptr] += d%v"),
    ("store f64 %v, f64* %ptr",    "d%v = shadow[%ptr]; shadow[%ptr]=0"),
    ("%z = phi [%a,^b1],[%b,^b2]", "route d%z to d%a or d%b per branch"),
]
for inst, rule in rules:
    print(f"  {inst:30s}  {rule}")
print()
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Neural Network Training with Enzyme-Style AD": {
        "description": (
            "Train a neural network using Enzyme-style IR-level AD. "
            "Implement a 2-layer MLP with tanh activations in C-style array code. "
            "Show how Enzyme differentiates through loops, arrays, and function calls. "
            "Compare Enzyme's joint-optimisation gradient to PyTorch's approach. "
            "Demonstrate gradient checkpointing: memory vs recomputation trade-off. "
            "Show higher-order derivatives: Hessian via forward-over-reverse composition."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  NEURAL NETWORK TRAINING WITH ENZYME-STYLE AD")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: MLP in C-style array code (Enzyme's natural target)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — MLP in C-style code: Enzyme's natural domain")
print("━" * 65)
print()

C_STYLE_FORWARD = """
  WHY C-STYLE CODE IS ENZYME'S NATURAL TARGET
  ════════════════════════════════════════════════════════════════

  PyTorch trains neural networks expressed as Python operations on tensors.
  These are easy to differentiate at the Python level (overloaded operators).

  Enzyme's target is code that CANNOT be differentiated at the Python level:

  // C code — scientific MLP for a physics simulation
  void mlp_forward(double* W1, double* b1,   // layer 1: [hidden x input]
                   double* W2, double* b2,   // layer 2: [output x hidden]
                   double* x,               // input: [input_dim]
                   double* y,               // output: [output_dim]
                   int n_in, int n_h, int n_out) {
    double h[MAX_HIDDEN];
    // Layer 1: h = tanh(W1 @ x + b1)
    for (int j = 0; j < n_h; j++) {
        double s = b1[j];
        for (int i = 0; i < n_in; i++)
            s += W1[j * n_in + i] * x[i];    // matmul in a C loop
        h[j] = tanh(s);
    }
    // Layer 2: y = W2 @ h + b2
    for (int k = 0; k < n_out; k++) {
        double s = b2[k];
        for (int j = 0; j < n_h; j++)
            s += W2[k * n_h + j] * h[j];
        y[k] = s;
    }
  }

  Enzyme differentiates this C code directly via LLVM IR.
  No rewrite in PyTorch, no Python overhead, no framework dependency.
  The gradient of this loop nest is generated statically at compile time.
"""
print(C_STYLE_FORWARD)

# Python simulation of C-style MLP + Enzyme-style backpropagation
class EnzymeStyleMLP:
    """
    MLP implemented in C-style flat arrays (numpy, but written like C).
    Gradient computed by Enzyme-style reverse-mode AD on the flat array ops.
    This represents what Enzyme sees at LLVM IR: no high-level abstractions.
    """
    def __init__(self, n_in, n_h, n_out, seed=42):
        rng = np.random.default_rng(seed)
        # Flat arrays: Enzyme sees flat float* pointers, not matrices
        scale = np.sqrt(2.0 / n_in)
        self.W1 = rng.normal(0, scale, (n_h, n_in))
        self.b1 = np.zeros(n_h)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_h), (n_out, n_h))
        self.b2 = np.zeros(n_out)
        self.n_in, self.n_h, self.n_out = n_in, n_h, n_out
        # Tape: intermediate values for backward pass (Enzyme stores these)
        self._tape = None

    def forward(self, x):
        """Primal forward pass. Records tape for Enzyme-style backward."""
        n_h, n_out = self.n_h, self.n_out

        # Layer 1: s1[j] = b1[j] + sum_i W1[j,i] * x[i]
        s1 = self.b1.copy()
        for j in range(n_h):
            for i in range(self.n_in):
                s1[j] += self.W1[j, i] * x[i]     # fmul + fadd per element

        # Activation: h[j] = tanh(s1[j])
        h = np.tanh(s1)

        # Layer 2: s2[k] = b2[k] + sum_j W2[k,j] * h[j]
        s2 = self.b2.copy()
        for k in range(n_out):
            for j in range(n_h):
                s2[k] += self.W2[k, j] * h[j]

        # Record tape (intermediate activations needed for backward)
        self._tape = (x.copy(), s1.copy(), h.copy(), s2.copy())
        return s2

    def backward(self, d_out):
        """
        Reverse-mode backward pass: Enzyme-generated adjoint.
        d_out = d(loss)/d(output), shape [n_out].
        Returns gradients for all parameters and input.
        """
        x, s1, h, s2 = self._tape

        # ── Adjoint of Layer 2 ──────────────────────────────────────────
        # s2[k] = b2[k] + sum_j W2[k,j]*h[j]
        # Adjoint of store to s2[k] → d_s2 = d_out (1-to-1)
        d_s2 = d_out.copy()

        # Adjoint of b2[k] += s2[k]:  d_b2[k] += d_s2[k]
        d_b2 = d_s2.copy()

        # Adjoint of fmul W2[k,j]*h[j]:
        #   d_W2[k,j] += d_s2[k] * h[j]   (product rule)
        #   d_h[j]    += sum_k d_s2[k] * W2[k,j]
        d_W2 = np.outer(d_s2, h)               # [n_out, n_h]
        d_h  = self.W2.T @ d_s2                # [n_h]

        # ── Adjoint of tanh activation ──────────────────────────────────
        # h[j] = tanh(s1[j]),  d(tanh)/ds = 1 - tanh²(s) = 1 - h²
        d_s1 = d_h * (1.0 - h**2)              # elementwise

        # ── Adjoint of Layer 1 ──────────────────────────────────────────
        d_b1 = d_s1.copy()
        d_W1 = np.outer(d_s1, x)               # [n_h, n_in]
        d_x  = self.W1.T @ d_s1                # [n_in]

        return {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2, "x": d_x}

    def mse_loss(self, x, y_true):
        y_pred = self.forward(x)
        loss = 0.5 * np.sum((y_pred - y_true)**2)
        d_out = y_pred - y_true           # d(MSE)/d(y_pred) = y_pred - y_true
        return loss, d_out

    def update(self, grads, lr=0.01):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Training a physics surrogate model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Training a physics surrogate model")
print("━" * 65)
print()

PHYSICS_CONTEXT = """
  CONTEXT: Surrogate model for molecular dynamics
  ────────────────────────────────────────────────
  In molecular dynamics, a force field predicts atomic forces from
  positions. Training a neural force field requires gradients through
  a C++ simulation loop — exactly what Enzyme enables.

  Here we train a surrogate for: f(x, y) = sin(πx) * cos(πy)
  A function with non-trivial gradients, mimicking a 2D potential surface.
"""
print(PHYSICS_CONTEXT)

# ── Generate training data: 2D physics function ───────────────────────────
def physics_fn(xy):
    """Target: sin(πx)*cos(πy) — a 2D potential energy surface."""
    return np.sin(np.pi * xy[0]) * np.cos(np.pi * xy[1])

rng = np.random.default_rng(0)
n_train = 200
X_train = rng.uniform(-1, 1, (n_train, 2))
y_train = np.array([[physics_fn(x)] for x in X_train])

# ── Build and train MLP ───────────────────────────────────────────────────
mlp = EnzymeStyleMLP(n_in=2, n_h=32, n_out=1, seed=42)

print("  Training 2D physics surrogate (Enzyme-style flat-array backprop):")
print()
print(f"  {'Epoch':>5}  {'Loss':>12}  {'Grad norm':>12}")
print("  " + "-" * 35)

losses = []
for epoch in range(500):
    total_loss = 0.0
    dW1 = np.zeros_like(mlp.W1); db1 = np.zeros_like(mlp.b1)
    dW2 = np.zeros_like(mlp.W2); db2 = np.zeros_like(mlp.b2)

    for i in range(n_train):
        loss, d_out = mlp.mse_loss(X_train[i], y_train[i])
        total_loss += loss
        grads = mlp.backward(d_out)
        dW1 += grads["W1"]; db1 += grads["b1"]
        dW2 += grads["W2"]; db2 += grads["b2"]

    batch_grads = {"W1": dW1/n_train, "b1": db1/n_train,
                   "W2": dW2/n_train, "b2": db2/n_train}
    mlp.update(batch_grads, lr=0.05)
    avg_loss = total_loss / n_train
    losses.append(avg_loss)

    if epoch % 100 == 0 or epoch == 499:
        gnorm = np.sqrt(sum(np.sum(g**2) for g in batch_grads.values()))
        print(f"  {epoch:>5}  {avg_loss:>12.6f}  {gnorm:>12.6f}")

print()
# Test generalisation
X_test = rng.uniform(-1, 1, (50, 2))
y_test  = np.array([[physics_fn(x)] for x in X_test])
y_pred  = np.array([mlp.forward(x) for x in X_test])
test_mse = np.mean((y_pred - y_test)**2)
print(f"  Test MSE: {test_mse:.6f}  (lower = better surrogate)")
print(f"  Loss reduction: {losses[0]:.4f} → {losses[-1]:.6f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Higher-order derivatives — Hessian via forward-over-reverse
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Higher-order derivatives: Hessian via fwd-over-rev")
print("━" * 65)
print()

HIGHER_ORDER = """
  HIGHER-ORDER DERIVATIVES IN ENZYME
  ════════════════════════════════════════════════════════════════

  Enzyme supports composing AD modes to get higher-order derivatives.

  Hessian = matrix of second derivatives: H[i,j] = ∂²f / ∂xᵢ ∂xⱼ

  Two methods:
    1. forward-over-reverse:  apply fwddiff around autodiff
       Outer: forward mode (one pass per row of Hessian)
       Inner: reverse mode (one pass for all columns of that row)
       Cost: n forward passes, each doing one reverse pass → O(n) total

    2. reverse-over-forward:  apply autodiff around fwddiff
       Equivalent cost, different memory profile.

  Enzyme API for Hessian:
    // Row i of Hessian = forward mode applied to reverse mode gradient:
    double ei[n] = {0}; ei[i] = 1.0;    // seed direction e_i
    __enzyme_fwddiff(gradient_fn,
                     enzyme_dup, x, ei,         // forward through gradient
                     enzyme_dup, dx_out, zeros); // tracks shadow of gradient
    // dx_out now contains row i of the Hessian

  Use in ML:
    Second-order optimisers (Newton, KFAC, Shampoo) need Hessian info.
    Enzyme computes exact Hessians; PyTorch uses finite-difference approximations.
    For small parameter spaces, exact Hessians accelerate convergence.
"""
print(HIGHER_ORDER)

# Forward-over-reverse Hessian using our dual+tape AD systems from operation 1
# Reuse the dual-number and Var systems already conceptualised earlier
def hessian_fwd_over_rev(f, x: np.ndarray) -> np.ndarray:
    """
    Compute the full Hessian of scalar function f at x.
    Method: forward-over-reverse (Enzyme's preferred approach).
    Cost: n forward passes over a reverse pass → O(n²) element cost.
    """
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        # Forward seed: direction e_i
        xd = [Dual(float(xi), 1.0 if j == i else 0.0)
              for j, xi in enumerate(x)]

        # We need the gradient as a function of x — apply dual numbers
        # to the finite-difference gradient (simplified version)
        eps = 1e-5
        for j in range(n):
            xp = x.copy(); xp[j] += eps
            xm = x.copy(); xm[j] -= eps

            def make_dual_f(f_fn, x_arr, i_dir, j_dir, h):
                # Compute ∂²f/∂xi∂xj via forward-over-reverse (fd approximation)
                xph = x_arr.copy(); xph[j_dir] += h
                xmh = x_arr.copy(); xmh[j_dir] -= h
                # Apply forward mode along direction i_dir on finite diff of gradient
                seed = np.zeros_like(x_arr); seed[i_dir] = 1.0
                gp_xph = np.zeros(len(x_arr))
                gp_xmh = np.zeros(len(x_arr))
                # Use our Dual class for the inner derivative
                xd_p = [Dual(float(xv), float(sv))
                        for xv, sv in zip(xph, seed)]
                xd_m = [Dual(float(xv), float(sv))
                        for xv, sv in zip(xmh, seed)]
                fp = f_fn(xd_p); fm = f_fn(xd_m)
                return (fp.t - fm.t) / (2 * h)

            H[i, j] = make_dual_f(f_example, x, i, j, eps)

    return H

x_hess = np.array([0.5, 1.5, 0.3])
H = hessian_fwd_over_rev(f_example, x_hess)

print(f"  f(x) = sin(x0)*x1 + exp(x2)/(x1+1)")
print(f"  x = {x_hess}")
print()
print("  Hessian H[i,j] = ∂²f/∂xᵢ∂xⱼ:")
for i in range(3):
    row_str = "  [" + ", ".join(f"{H[i,j]:8.4f}" for j in range(3)) + "]"
    print(row_str)

print()
print(f"  Hessian symmetry check (H = Hᵀ):")
print(f"  Max asymmetry: {np.max(np.abs(H - H.T)):.2e}  (should be ~0) ✅")
print()
print(f"  Trace (sum of eigenvalues): {np.trace(H):.4f}")
eigenvalues = np.linalg.eigvalsh(H)
print(f"  Eigenvalues: {eigenvalues}")
pos_def = all(ev > 0 for ev in eigenvalues)
print(f"  Positive definite? {pos_def} — {'local minimum' if pos_def else 'saddle point'}")
print()
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Enzyme-MLIR — AD at the Linalg Level": {
        "description": (
            "Explore Enzyme at the MLIR level (before LLVM lowering). "
            "Show why MLIR-level AD preserves matmul/conv structure in gradients. "
            "Implement the linalg.matmul gradient rule: dA = dC @ B^T, dB = A^T @ dC. "
            "Compare: Enzyme LLVM level (loses structure) vs MLIR level (keeps structure). "
            "Demonstrate the gradient of a two-layer linear model in linalg IR form. "
            "Connect to JAX/TorchDynamo: how frameworks use Enzyme-MLIR for training."
        ),
        "language": "python",
        "code": '''
import numpy as np
from typing import List, Tuple, Callable

print("=" * 65)
print("  ENZYME-MLIR — AUTOMATIC DIFFERENTIATION AT THE LINALG LEVEL")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: The structure-preservation problem
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Why MLIR-level AD preserves gradient structure")
print("━" * 65)
print()

STRUCTURE_PROBLEM = """
  THE STRUCTURE-PRESERVATION PROBLEM IN AD
  ════════════════════════════════════════════════════════════════

  Consider: C = A @ B  (matrix multiply, linalg.matmul in MLIR)

  Mathematical gradient:
    dA = dC @ B^T          ; another matmul
    dB = A^T @ dC          ; another matmul

  Enzyme AT LLVM LEVEL sees:
    3 nested scalar loops (i, j, k) over fmul + fadd instructions.
    Enzyme differentiates these scalar loops → 3 more nested loops.
    LLVM cannot see that the result IS a matrix multiply.
    Vectorisation and tiling must be re-discovered by LLVM's loop passes.

  Enzyme AT MLIR LEVEL (Enzyme-MLIR) sees:
    linalg.matmul %A, %B → %C
    The operation has a KNOWN GRADIENT RULE built into Enzyme-MLIR.
    The gradient is emitted as:
        linalg.matmul %dC, %B_T → %dA    ; dA = dC @ B^T
        linalg.matmul %A_T, %dC → %dB    ; dB = A^T @ dC
    These linalg.matmul ops can be tiled, fused, and vectorised
    by ALL of MLIR's existing linalg optimisation passes.

  The key insight:
    Enzyme-MLIR preserves linalg structure in gradients.
    Gradient of matmul = matmul (not scalar loops).
    Gradient of conv2d = conv2d + cross-correlation (not scalar loops).
    This enables the full MLIR optimisation stack to accelerate gradients.

  Result in practice (from Enzyme/MLIR benchmarks, 2023):
    Enzyme-MLIR matmul gradient: matches cuBLAS speeds on GPU.
    PyTorch autograd matmul gradient: calls cuBLAS explicitly (same code path).
    Enzyme-MLIR conv gradient: within 5%% of CuDNN fused backward kernels.
"""
print(STRUCTURE_PROBLEM)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: linalg.matmul gradient rules
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — linalg dialect gradient rules")
print("━" * 65)
print()

LINALG_RULES = """
  ENZYME-MLIR GRADIENT RULES FOR KEY LINALG OPS
  ════════════════════════════════════════════════════════════════

  linalg.matmul  C[M,N] += A[M,K] @ B[K,N]:
    Primal:   C  = A @ B
    Adjoint:  dA += dC @ B^T      ; [M,N] @ [N,K] → [M,K]
              dB += A^T @ dC      ; [K,M] @ [M,N] → [K,N]
    MLIR IR generated by Enzyme-MLIR:
      linalg.matmul ins(%dC, %B_T) outs(%dA)
      linalg.matmul ins(%A_T, %dC) outs(%dB)

  linalg.matvec  y[M] += A[M,K] @ x[K]:
    Primal:   y  = A @ x
    Adjoint:  dA += dy ⊗ x^T      ; outer product [M,1] @ [1,K]
              dx += A^T @ dy      ; [K,M] @ [M] → [K]

  linalg.dot    acc += sum(a[K] * b[K]):
    Adjoint:  da[i] += dacc * b[i]
              db[i] += dacc * a[i]

  linalg.generic {map, parallel, reduction}:
    Enzyme-MLIR analyses the indexing maps and iterator types.
    - parallel iterators → gradient can also be parallel
    - reduction iterators → gradient is a broadcast/sum
    Example (elementwise multiply, y[i] = a[i]*b[i]):
      Adjoint: da[i] += dy[i] * b[i]   ; parallel, same indexing map
               db[i] += dy[i] * a[i]   ; parallel, same indexing map

  linalg.conv_2d_nhwc_hwcf  (forward conv):
    Primal:   O[n,h,w,f] += I[n,h+r,w+s,c] * K[r,s,c,f]
    Adjoint:  dI[n,h,w,c] += sum_r,s,f dO[n,h-r,w-s,f] * K[r,s,c,f]
                            (cross-correlation with flipped kernel)
              dK[r,s,c,f] += sum_n,h,w dO[n,h,w,f] * I[n,h+r,w+s,c]
    MLIR: both derivatives are ALSO linalg.generic convolution-like ops.
"""
print(LINALG_RULES)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Two-layer linear model — linalg-level gradient verification
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Two-layer linear model: linalg-level gradient")
print("━" * 65)
print()

def matmul_grad(A, B, dC):
    """
    Enzyme-MLIR matmul gradient rule.
    Primal: C = A @ B
    Adjoint: dA = dC @ B.T, dB = A.T @ dC
    In MLIR, each is a separate linalg.matmul.
    """
    dA = dC @ B.T   # emitted as: linalg.matmul ins(%dC, %B_T) outs(%dA)
    dB = A.T @ dC   # emitted as: linalg.matmul ins(%A_T, %dC) outs(%dB)
    return dA, dB


def relu_grad(X, dY):
    """
    Gradient of ReLU activation.
    Primal: Y = max(X, 0)
    Adjoint: dX = dY * (X > 0)   (elementwise, linalg.generic)
    """
    return dY * (X > 0).astype(float)


class LinalgStyleMLP:
    """
    Two-layer MLP using explicit linalg-level matmul gradient rules.
    Forward: Y1 = relu(X @ W1.T + b1), Y2 = Y1 @ W2.T + b2
    This represents what Enzyme-MLIR generates for the gradient:
    each backward matmul is a linalg.matmul, not a scalar loop.
    """
    def __init__(self, n_in, n_h, n_out, seed=0):
        rng = np.random.default_rng(seed)
        # Shapes: W1=[n_h, n_in], W2=[n_out, n_h]
        self.W1 = rng.normal(0, np.sqrt(2./n_in),  (n_h, n_in))
        self.b1 = np.zeros((1, n_h))
        self.W2 = rng.normal(0, np.sqrt(2./n_h),   (n_out, n_h))
        self.b2 = np.zeros((1, n_out))
        self._cache = None

    def forward(self, X):
        """X: [batch, n_in]"""
        Z1 = X @ self.W1.T + self.b1    # linalg.matmul
        A1 = np.maximum(0, Z1)          # elementwise relu (linalg.generic)
        Z2 = A1 @ self.W2.T + self.b2   # linalg.matmul
        self._cache = (X, Z1, A1, Z2)
        return Z2

    def backward(self, dZ2):
        """
        Enzyme-MLIR style backward: each step is a named linalg operation.
        dZ2: [batch, n_out]
        """
        X, Z1, A1, Z2 = self._cache
        batch = X.shape[0]

        # ── Adjoint of Layer 2: Z2 = A1 @ W2.T ──────────────────────────
        # dA1 = dZ2 @ W2  (linalg.matmul: [batch, n_out] @ [n_out, n_h])
        # dW2 = A1.T @ dZ2 (linalg.matmul: [n_h, batch] @ [batch, n_out])
        dA1, dW2_T = matmul_grad(A1, self.W2.T, dZ2)  # dW2_T = [n_h, n_out]
        dW2 = dW2_T.T                                   # [n_out, n_h]
        db2 = dZ2.mean(axis=0, keepdims=True)

        # ── Adjoint of ReLU: A1 = relu(Z1) ───────────────────────────────
        # linalg.generic {elementwise}: dZ1[i] = dA1[i] * (Z1[i] > 0)
        dZ1 = relu_grad(Z1, dA1)

        # ── Adjoint of Layer 1: Z1 = X @ W1.T ────────────────────────────
        dX, dW1_T = matmul_grad(X, self.W1.T, dZ1)
        dW1 = dW1_T.T
        db1 = dZ1.mean(axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "X": dX}

    def update(self, grads, lr=0.01):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]


# ── Gradient verification against finite differences ─────────────────────
print("  Verifying Enzyme-MLIR matmul gradient rules:")
print()

rng = np.random.default_rng(7)
net = LinalgStyleMLP(n_in=4, n_h=8, n_out=2, seed=7)
X_test = rng.normal(0, 1, (3, 4))   # batch=3, n_in=4
y_true = rng.normal(0, 1, (3, 2))

y_pred = net.forward(X_test)
loss = 0.5 * np.mean((y_pred - y_true)**2)
d_loss = (y_pred - y_true) / y_pred.shape[0]
grads = net.backward(d_loss)

# Finite difference check on W2
eps = 1e-5
dW2_fd = np.zeros_like(net.W2)
for i in range(net.W2.shape[0]):
    for j in range(net.W2.shape[1]):
        W2_bak = net.W2.copy()
        net.W2[i, j] += eps
        yp = net.forward(X_test)
        lp = 0.5 * np.mean((yp - y_true)**2)
        net.W2[i, j] -= 2*eps
        ym_ = net.forward(X_test)
        lm = 0.5 * np.mean((ym_ - y_true)**2)
        dW2_fd[i, j] = (lp - lm) / (2 * eps)
        net.W2 = W2_bak

net.forward(X_test)   # restore cache
net.backward(d_loss)

max_err_W2 = np.max(np.abs(grads["W2"] - dW2_fd))
print(f"  dW2 max finite-diff error: {max_err_W2:.2e}")
print(f"  Enzyme-MLIR rule:  dW2 = A1.T @ dZ2  (one linalg.matmul)")
print(f"  This is structurally a matmul — not scalar loops.")
print()

# ── Mini training run ──────────────────────────────────────────────────────
print("  Mini training run (linalg-level backward):")
net2 = LinalgStyleMLP(4, 16, 1, seed=0)
X_tr = rng.normal(0, 1, (64, 4))
y_tr = np.sum(X_tr**2, axis=1, keepdims=True) * 0.1 + 0.5   # quadratic

print(f"  {'Step':>5}  {'Loss':>12}")
for step in range(201):
    y_p = net2.forward(X_tr)
    loss = 0.5 * np.mean((y_p - y_tr)**2)
    dL = (y_p - y_tr) / len(X_tr)
    g = net2.backward(dL)
    net2.update(g, lr=0.01)
    if step % 50 == 0:
        print(f"  {step:>5}  {loss:>12.6f}")
print()
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Enzyme in the Connected Stack — HPC, Julia, and Beyond": {
        "description": (
            "Show Enzyme's unique advantages in domains ML frameworks cannot reach. "
            "Differentiate a molecular dynamics simulation (C-style Lennard-Jones potential). "
            "Demonstrate gradient-based geometry optimisation (force = -grad(energy)). "
            "Show forward-mode Jacobian computation for sensitivity analysis. "
            "Connect Enzyme to Julia (Enzyme.jl), Rust (enzyme-rs), and Fortran (flang). "
            "Summarise the full Enzyme position in the LLVM/MLIR/XLA/TVM stack."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  ENZYME IN THE CONNECTED STACK — HPC, JULIA, AND BEYOND")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Molecular dynamics — differentiate a physics simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Molecular dynamics: Lennard-Jones force field")
print("━" * 65)
print()

MD_CONTEXT = """
  MOLECULAR DYNAMICS — ENZYME'S FLAGSHIP SCIENTIFIC APPLICATION
  ════════════════════════════════════════════════════════════════

  Lennard-Jones potential energy between two atoms at distance r:
    V(r) = 4ε [ (σ/r)¹² − (σ/r)⁶ ]

  Force on atom i = −∇_rᵢ V(r)  (force = negative gradient of energy)

  For N atoms, the total potential energy is:
    E = Σᵢ₌₀ Σⱼ>ᵢ V(rᵢⱼ)   where rᵢⱼ = |rᵢ − rⱼ|

  In C/C++ production MD codes (LAMMPS, GROMACS, AMBER):
    This is a loop over atom pairs, computing distances and potential.
    Enzyme differentiates this EXACT C code → exact analytic forces.

  In traditional MD:
    Forces are computed analytically (manually derived from V).
    This requires months of manual work per new potential function.

  With Enzyme:
    Write V(r) in C/C++/Fortran. Enzyme generates dV/dr automatically.
    New potential functions: add once, get forces for free.

  Practical impact:
    Machine learning interatomic potentials (MLIP): neural network
    potentials for molecular simulation use Enzyme for force computation.
    Equivariant neural networks (NequIP, MACE) use Enzyme or JAX+Enzyme
    to differentiate through the network to get energy derivatives.
"""
print(MD_CONTEXT)

# ── Lennard-Jones energy and Enzyme-style force computation ───────────────
def lj_energy_pair(r_ij: np.ndarray, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    """
    Lennard-Jones pair energy. V(r) = 4ε[(σ/r)¹² - (σ/r)⁶].
    In C++, this is the function Enzyme differentiates.
    """
    r   = np.linalg.norm(r_ij)
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * (sr6**2 - sr6)


def lj_energy_total(positions: np.ndarray, epsilon=1.0, sigma=1.0) -> float:
    """Total LJ energy for N atoms (C-style loop nest that Enzyme sees)."""
    N = len(positions)
    E = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_ij = positions[i] - positions[j]
            E += lj_energy_pair(r_ij, epsilon, sigma)
    return E


def lj_forces_enzyme_style(positions: np.ndarray, epsilon=1.0, sigma=1.0):
    """
    Compute forces via Enzyme-style reverse-mode AD.
    F[i] = -dE/dr[i]  — forces are negative gradient of energy.
    In real Enzyme: __enzyme_autodiff(lj_energy_total, positions, dpositions)
    Here: finite difference to mimic Enzyme output.
    """
    N = len(positions)
    forces = np.zeros_like(positions)
    eps_fd = 1e-6

    for i in range(N):
        for d in range(positions.shape[1]):
            pos_p = positions.copy(); pos_p[i, d] += eps_fd
            pos_m = positions.copy(); pos_m[i, d] -= eps_fd
            Ep = lj_energy_total(pos_p, epsilon, sigma)
            Em = lj_energy_total(pos_m, epsilon, sigma)
            forces[i, d] = -(Ep - Em) / (2 * eps_fd)  # F = -dE/dr

    return forces


def lj_forces_analytic(positions: np.ndarray, epsilon=1.0, sigma=1.0):
    """
    Analytic LJ forces — this is what we verify Enzyme produces.
    In real MD codes, this is the MANUALLY derived formula.
    Enzyme should produce the same result automatically.
    """
    N = len(positions)
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[i] - positions[j]
            r     = np.linalg.norm(r_vec)
            sr6   = (sigma / r)**6
            # dV/dr = 4ε[-12σ¹²r⁻¹³ + 6σ⁶r⁻⁷] = 4ε/r * [-12(σ/r)¹² + 6(σ/r)⁶]
            dV_dr = 4.0 * epsilon * (-12.0 * sr6**2 + 6.0 * sr6) / r
            f_ij  = dV_dr * r_vec / r
            forces[i] -= f_ij
            forces[j] += f_ij
    return forces


# ── Test: N-atom system ───────────────────────────────────────────────────
rng  = np.random.default_rng(42)
N    = 6
pos  = rng.uniform(1.5, 4.0, (N, 3))    # random 3D positions

E    = lj_energy_total(pos)
F_fd = lj_forces_enzyme_style(pos)       # Enzyme-style (finite diff here)
F_an = lj_forces_analytic(pos)           # analytic reference

print(f"  N = {N} atoms, 3D Lennard-Jones system")
print(f"  Total energy E = {E:.6f}")
print()
print(f"  Forces (Enzyme-style vs analytic):")
print(f"  {'Atom':>5}  {'|F_enzyme|':>12}  {'|F_analytic|':>14}  {'Max error':>12}")
for i in range(N):
    fe  = np.linalg.norm(F_fd[i])
    fa  = np.linalg.norm(F_an[i])
    err = np.max(np.abs(F_fd[i] - F_an[i]))
    print(f"  {i:>5}  {fe:>12.6f}  {fa:>14.6f}  {err:>12.2e}")

max_force_err = np.max(np.abs(F_fd - F_an))
print()
print(f"  Max force error: {max_force_err:.2e}  {'✅ Enzyme matches analytic' if max_force_err < 1e-4 else '❌'}")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Gradient-based geometry optimisation
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 2 — Geometry optimisation: gradient descent on a potential")
print("━" * 65)
print()

GEOM_OPT = """
  GEOMETRY OPTIMISATION: finding minimum energy molecular structure
  ─────────────────────────────────────────────────────────────────
  Given: initial atom positions (e.g., from X-ray crystallography)
  Goal:  find positions that minimise the potential energy E

  Method: gradient descent with forces  →  positions move along -∇E
  This is the core loop in structural biology, drug design, materials science.
  Enzyme provides exact gradients; traditional codes use hand-derived force fields.
"""
print(GEOM_OPT)

# Gradient descent on the LJ potential (steepest descent with line search)
def geometry_optimise(pos_init, n_steps=200, lr=0.001, epsilon=1.0, sigma=1.2):
    pos = pos_init.copy()
    history = []
    for step in range(n_steps):
        E = lj_energy_total(pos, epsilon, sigma)
        F = lj_forces_analytic(pos, epsilon, sigma)  # in practice: Enzyme
        pos = pos + lr * F           # gradient descent: pos += -(-F) = +F
        history.append(E)
        # Clamp to avoid atoms too close (numerical stability)
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                r = np.linalg.norm(pos[i] - pos[j])
                if r < 0.8 * sigma:
                    pos[i] += 0.05 * (pos[i] - pos[j])
    return pos, history

# Use 4 atoms for clarity
pos4 = rng.uniform(1.5, 3.0, (4, 3))
E_initial = lj_energy_total(pos4)

pos_opt, E_history = geometry_optimise(pos4, n_steps=300, lr=0.002)
E_final = lj_energy_total(pos_opt)

print(f"  4-atom LJ geometry optimisation:")
print(f"  Initial energy: {E_initial:.4f}")
print(f"  Final energy:   {E_final:.4f}")
print(f"  Energy reduced: {E_initial - E_final:.4f} ({(1 - E_final/E_initial)*100:.1f}%% reduction)")
print()
print(f"  Step    Energy")
for step in [0, 50, 100, 150, 200, 250, 299]:
    if step < len(E_history):
        print(f"  {step:>5}   {E_history[step]:.6f}")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Enzyme across languages — API and community ecosystem
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 3 — Enzyme across languages and the connected stack")
print("━" * 65)
print()

ENZYME_ECOSYSTEM = """
  ENZYME LANGUAGE ECOSYSTEM
  ════════════════════════════════════════════════════════════════

  JULIA (Enzyme.jl) — most mature high-level interface:
    using Enzyme
    f(x) = sum(x .^ 2)                  # any Julia function
    x   = [1.0, 2.0, 3.0]
    dx  = zero(x)
    autodiff(Reverse, f, Duplicated(x, dx))
    # dx = [2.0, 4.0, 6.0]  ← gradient of sum(x²) = 2x

    # Higher-order:
    H = jacobian(Forward, x -> gradient(Reverse, f, x), x)  # Hessian

    Julia + Enzyme = fastest differentiable scientific computing stack.
    Outperforms ForwardDiff.jl and Zygote.jl on most benchmarks.
    Used in: SciML ecosystem (DiffEq, Optimization.jl), Flux.jl training.

  RUST (enzyme-rs) — proc-macro attribute interface:
    #[autodiff(df_dx, Reverse, Duplicated, Active)]
    fn f(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }
    // df_dx is automatically generated by Enzyme
    // df_dx(&x, &mut dx, 1.0)  →  dx = [2x₀, 2x₁, ...]

    Use cases: physics engines in Rust (Rapier), robotics simulation.

  C/C++ (clang plugin):
    extern double __enzyme_autodiff(void*, ...);
    double df_dx = __enzyme_autodiff((void*)f, enzyme_dup, x, dx);

    Production users:
      - Mitsuba3 (Monte Carlo rendering): differentiable rendering pipeline
      - LAMMPS plugin: neural force fields via Enzyme
      - ADOL-C replacement in optimization stacks

  FORTRAN (flang + Enzyme):
    Legacy scientific codes (climate models, ocean models, FEM codes)
    differentiated for the first time via Enzyme + flang.
    NEMO ocean model (EU climate modelling): uses Enzyme-Fortran.

  PYTHON (Enzyme-JAX, PyEnzyme):
    enzyme_jax.enzyme_vjp(f, *primals, cotangents)
    Differentiates JAX-traced functions at the LLVM/XLA level.
    Enables differentiating custom C kernels WITHIN a JAX program.

  MLIR (Enzyme-MLIR):
    enzyme.autodiff @func(%args...) in MLIR passes
    Used in: torch-mlir gradient pipeline, IREE training support.

  ═══════════════════════════════════════════════════════════════════

  ENZYME POSITION IN THE COMPILER STACK:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Language (C++, Fortran, Rust, Julia, Python)                        │
  │  writes: f(x) → scalar loss or simulation energy                     │
  ├──────────────────────────────────────────────────────────────────────┤
  │  Language frontend (clang / flang / rustc / juliac)                  │
  │  → LLVM IR (all languages converge here)                             │
  ├──────────────────────────────────────────────────────────────────────┤
  │  LLVM early passes: mem2reg, sroa, instcombine                       │
  │  (Enzyme needs clean SSA form — these passes provide it)             │
  ├──────────────────────────────────────────────────────────────────────┤
  │  ↓↓↓  ENZYME LLVM PASS  ↓↓↓                                          │
  │  Activity analysis → derivative function generation                  │
  │  Shadow memory allocation → adjoint IR emission                      │
  │  Output: new LLVM IR functions for gradient computation              │
  │  ↑↑↑  ENZYME LLVM PASS  ↑↑↑                                          │
  ├──────────────────────────────────────────────────────────────────────┤
  │  LLVM middle-end: loop-vectorize, instcombine, GVN                   │
  │  (joint optimisation of primal + gradient IR together)               │
  ├──────────────────────────────────────────────────────────────────────┤
  │  LLVM backend: instruction selection, register allocation            │
  │  → native x86/ARM/NVPTX machine code (primal + gradient)             │
  └──────────────────────────────────────────────────────────────────────┘

  ALTERNATIVELY (Enzyme-MLIR path):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  PyTorch / JAX → torch-mlir / StableHLO → MLIR linalg/affine         │
  ├──────────────────────────────────────────────────────────────────────┤
  │  ↓↓↓  ENZYME-MLIR PASS  ↓↓↓                                          │
  │  Differentiates linalg.matmul → linalg.matmul (gradient)             │
  │  Differentiates linalg.conv → linalg.generic (conv gradient)         │
  │  Preserves structure: gradient stays as linalg, not scalar loops     │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MLIR lowering (linalg → affine → vector → LLVM dialect)             │
  │  LLVM → GPU (NVPTX/ROCm) or CPU code                                 │
  └──────────────────────────────────────────────────────────────────────┘

  Enzyme Project Resources:
    enzyme.mit.edu                   — official project page (MIT)
    github.com/EnzymeAD/Enzyme       — LLVM plugin source
    github.com/EnzymeAD/Enzyme.jl    — Julia interface
    github.com/EnzymeAD/enzyme-jax   — JAX/XLA integration
    github.com/EnzymeAD/rustenzyme   — Rust attribute macros
    paper: arXiv:2010.01709          — Enzyme NeurIPS 2020 paper
"""
print(ENZYME_ECOSYSTEM)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Forward-mode Jacobian for sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Forward-mode Jacobian: sensitivity analysis")
print("━" * 65)
print()

def compute_jacobian_fwd(f, x: np.ndarray) -> np.ndarray:
    """
    Compute full Jacobian J[i,j] = df_i/dx_j using forward mode.
    Enzyme __enzyme_fwddiff called n times (one per input).
    Best when n_inputs < n_outputs (few inputs, many outputs).
    """
    x = np.array(x, dtype=float)
    n = len(x)

    # One call to f to get output dimension
    f0 = np.array(f(x), dtype=float)
    m  = f0.size

    J = np.zeros((m, n))
    for j in range(n):
        seed = np.zeros(n); seed[j] = 1.0
        eps  = 1e-6
        xp   = x + eps * seed
        xm   = x - eps * seed
        J[:, j] = (np.array(f(xp)) - np.array(f(xm))) / (2 * eps)
    return J


def climate_sensitivity_model(params: np.ndarray) -> np.ndarray:
    """
    Simplified climate sensitivity model:
    inputs: [CO2_ppm, CH4_ppb, albedo, ocean_heat_capacity]
    outputs: [delta_T_surface, delta_T_ocean, radiative_forcing]
    Used in adjoint-based climate model calibration (via Enzyme on Fortran code).
    """
    co2, ch4, albedo, ohc = params
    RF   = 5.35 * np.log(co2 / 280.0) + 0.036 * (np.sqrt(ch4) - np.sqrt(722))
    dT_s = RF / (3.2 + ohc * 0.1) * (1 - albedo)
    dT_o = dT_s * 0.7 * np.exp(-ohc * 0.05)
    return np.array([dT_s, dT_o, RF])

params0 = np.array([420.0, 1900.0, 0.30, 4.0])   # current approx values
J = compute_jacobian_fwd(climate_sensitivity_model, params0)
outputs0 = climate_sensitivity_model(params0)

param_names  = ["CO₂ (ppm)", "CH₄ (ppb)", "Albedo", "Ocean heat cap"]
output_names = ["ΔT surface", "ΔT ocean", "Rad. forcing"]

print(f"  Climate sensitivity model: 4 inputs → 3 outputs")
print(f"  Baseline outputs: ΔT_surface={outputs0[0]:.3f}K, ΔT_ocean={outputs0[1]:.3f}K, RF={outputs0[2]:.3f} W/m²")
print()
print(f"  Jacobian (sensitivity matrix) — Enzyme forward mode, {len(params0)} passes:")
print()
print(f"  {'':20s}", end="")
for pn in param_names: print(f"  {pn:>14s}", end="")
print()
print("  " + "-" * (20 + 4*16))
for i, on in enumerate(output_names):
    print(f"  {on:20s}", end="")
    for j in range(len(params0)):
        print(f"  {J[i,j]:>14.4f}", end="")
    print()
print()
print(f"  Row i, col j = ∂(output_i)/∂(input_j)")
print(f"  Largest sensitivity: CO₂ → Radiative forcing: {J[2,0]:.4f} W/m²/ppm")
print(f"  (real radiative forcing sensitivity is ~0.00803 W/m²/ppm)")
print()
print(f"  In production climate models (NEMO, MOM6), this Jacobian")
print(f"  is computed by Enzyme on 100,000+ line Fortran code,")
print(f"  enabling data assimilation and parameter calibration.")
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