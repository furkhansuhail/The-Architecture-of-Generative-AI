"""
TorchDynamo — PyTorch 2.x Graph Capture
=========================================

TorchDynamo is the graph capture engine at the heart of PyTorch 2.0's
torch.compile system. It solves a problem that defeated every previous
attempt at compiling PyTorch: how do you extract a clean, compilable
computation graph from arbitrary Python code that mixes tensor operations
with control flow, data-dependent branching, Python objects, closures, and
library calls — without requiring the user to rewrite anything?

The answer is radical: intercept Python at the bytecode level.

TorchDynamo does not trace Python the way JAX or TorchScript do. It does not
ask you to write your model in a restricted DSL. It does not attempt static
analysis of your Python source. Instead, it installs itself as a FRAME
EVALUATION FUNCTION — a CPython hook that intercepts the execution of every
Python frame (function call) and rewrites its bytecode on the fly, replacing
tensor operations with graph-recorded versions. When the Python code does
something Dynamo cannot represent in a graph (calls a random library, branches
on a tensor value, uses a Python list that changes length), Dynamo issues a
GRAPH BREAK: it stops recording the current graph, falls back to normal Python
execution for that fragment, then resumes recording for the next compilable
region.

This design — compile what you can, fall back when you can't — is what makes
torch.compile work on real-world PyTorch code with essentially zero user changes.

The complete torch.compile stack has three layers:
    TorchDynamo (this module):  graph capture via Python bytecode interception.
                                Produces FX graphs from real PyTorch model runs.
    AOTAutograd (this module):  ahead-of-time differentiation.
                                Differentiates the captured FX graph before
                                compilation, producing joint forward+backward graphs.
    TorchInductor / backends:   compilation to fast machine code.
                                Lowers the joint graph to Triton (GPU) or
                                C++ (CPU), or dispatches to external compilers.

In the connected compiler stack:
    LLVM      (module 01) ← TorchInductor's CPU backend emits C++ that GCC/clang compile
    MLIR      (module 02) ← torch-mlir lowers Dynamo's FX graphs to MLIR dialects
    CIRCT     (module 03) ← hardware-targeting flows consume torch-mlir's StableHLO output
    Enzyme    (module 04) ← Enzyme-JAX differentiates at LLVM level; AOTAutograd does this at graph level
    XLA       (module 05) ← torch.compile with backend="openxla" routes through XLA
    OpenXLA   (module 06) ← Dynamo's FX graph can be exported as StableHLO
    StableHLO (module 07) ← torch.export → StableHLO is the deployment path from Dynamo
    TVM       (module 08) ← TVM can serve as a torch.compile backend via torch_tvm
    TorchDynamo (this)    ← the graph capture layer enabling all the above

"""

import textwrap
import re

TOPIC_NAME   = "TorchDynamo — PyTorch 2.x Graph Capture"
DISPLAY_NAME = "08 · TorchDynamo"
ICON         = "🔥"
SUBTITLE     = "Bytecode Interception, FX Graphs, Guards, AOTAutograd, and torch.compile"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE PROBLEM: WHY PYTORCH GRAPH CAPTURE IS HARD

### Why PyTorch Resisted Compilation for Eight Years

    PyTorch launched in 2016 with a bold design choice: eager execution.
    Every operation — every matmul, every relu, every indexing expression —
    executes immediately when the Python line runs. There is no deferred
    computation, no graph to build, no separate compile step.

    This was revolutionary. TensorFlow 1.x required you to define a static
    computation graph using special tf.* APIs, then hand that graph to a
    session to execute. Debugging was a nightmare: you could not set Python
    breakpoints inside a model's forward pass, could not use Python print()
    to inspect intermediate values, could not conditionally branch on tensor
    values using normal Python if statements.

    PyTorch's eager mode gave back all of that:
        x = torch.relu(x)      # executes NOW, result available now
        if x.sum() > 0:        # branch on a real tensor value
            x = x * 2          # conditionally applied
        print(x[:3])           # inspect intermediate tensors freely

    The cost: no graph = no compilation = no fusion = lower performance.
    Each operation dispatches separately to a CUDA kernel. Each kernel launch
    costs ~5–10 microseconds of overhead. A transformer layer has hundreds
    of operations: the overhead alone can exceed the computation time.

### The Four Previous Attempts (and Why They Failed)

    ATTEMPT 1 — TorchScript (2018):
        TorchScript is a statically typed subset of Python that PyTorch can
        JIT-compile into a graph. You annotate your model with @torch.jit.script
        and the compiler parses the decorated function's source code.

        FAILURE MODE:
            TorchScript does not support most Python features:
                No Python closures over non-tensor objects.
                No arbitrary Python class methods.
                No NumPy interoperability.
                No third-party libraries (huggingface transformers, etc.).
                No data-dependent shapes (dynamic-length sequences).
            Real-world models in production use all of these.
            Porting a non-trivial model to TorchScript takes days or weeks.
            The BERT implementation in huggingface transformers: 98% failure
            rate when trying to TorchScript it.

    ATTEMPT 2 — torch.fx (2020):
        torch.fx is a pure-Python graph capture tool that uses Python's
        __torch_function__ protocol to intercept tensor operations and record
        them into an FX Graph (a DAG of nodes representing operations).

        FAILURE MODE:
            torch.fx is a SYMBOLIC TRACER. It traces through the model once
            with fake tensor inputs, recording every operation.
            This fails when:
                Code branches on tensor VALUES (if x > 0: ...)
                Code uses Python control flow that depends on tensor shapes
                Code calls non-tensor Python code (print(), logging, etc.)
                Code creates tensors with shapes computed at runtime
            The symbolic trace either traces the wrong branch (the one taken
            with the fake input) or crashes entirely.

    ATTEMPT 3 — torch.jit.trace (2017):
        Like torch.fx but at the C++ level: run the model once with concrete
        inputs, record what actually happened, replay that recording.

        FAILURE MODE:
            The recorded trace is valid for ONE specific input shape and value.
            If batch size changes → crash or silently wrong results.
            If input values change a conditional → wrong branch executed.
            Loops with data-dependent bounds → unrolled for the specific run.
            This makes traced models brittle in production.

    ATTEMPT 4 — LazyTensor (2021):
        Defers tensor operations into an IR that is compiled later.
        Used in PyTorch/XLA to run PyTorch on TPUs.

        FAILURE MODE:
            Very high overhead when model contains Python-level operations
            that cannot be deferred (printing, numpy interop, etc.).
            Requires the user to explicitly "mark steps" to trigger compilation.
            Difficult to debug: errors appear long after the Python line.
            Does not handle Python data structures (lists, dicts) well.

### The Fundamental Tension

    ALL previous approaches faced the same impossible requirement:
        PyTorch's power = Python's full expressiveness in model code.
        Compilation's requirement = a clean, complete computation graph.

    These are fundamentally in conflict:
        Full Python expressiveness → arbitrary control flow, data structures,
                                     external library calls, mutable state.
        Computation graph = a DAG → no control flow, no external state,
                                     no Python objects, pure tensor operations.

    The question is: is there any way to get a compilable graph from
    arbitrary Python code WITHOUT requiring the user to constrain their code?

### TorchDynamo's Answer: Partial Compilation with Graph Breaks

    TorchDynamo's insight: you do not need to compile EVERYTHING.
    You need to compile the TENSOR COMPUTATION parts — the parts that
    actually cost time on GPU. The Python scaffolding (loop indices,
    condition checks, print statements) is fast enough to run in Python.

    TorchDynamo's strategy:
        1. Intercept Python execution at the frame level.
        2. Extract the LARGEST POSSIBLE contiguous tensor computation region.
        3. Compile that region into a fast kernel.
        4. When Python does something uncompilable: GRAPH BREAK.
           Stop recording, execute that fragment in normal Python,
           then resume recording the next tensor computation region.
        5. Each compiled region is CACHED with GUARDS that describe what
           must remain true for the cached code to be correct.

    The result: models compile automatically, even if imperfectly.
    A model with 10 graph breaks still gets 10 separately-compiled regions,
    each faster than pure eager. The goal is to drive graph break count to
    near zero for common cases.


##### PART 2 — TORCHDYNAMO'S MECHANISM: PYTHON BYTECODE INTERCEPTION

### CPython's Frame Evaluation API

    Python 3.6 added PEP 523: "Adding a frame evaluation API to CPython."
    This PEP added a C-level hook: _PyInterpreterState_SetEvalFrameFunc().

    This hook allows a C extension to register a CUSTOM FRAME EVALUATOR —
    a function that CPython calls instead of its default bytecode interpreter
    whenever it is about to execute a Python function.

    The custom evaluator receives:
        The frame object (containing the bytecode, local variables, closure)
        The "throwflag" (whether an exception is being thrown)

    It can:
        Execute the frame normally (delegate to CPython's default evaluator)
        Transform the frame's bytecode before execution
        Build a completely different execution from the same bytecode
        Return synthetic results without executing the bytecode at all

    TorchDynamo registers itself as this custom frame evaluator.
    Every Python function call in a torch.compile-decorated scope goes
    through Dynamo's evaluator before any bytecode executes.

### What Dynamo Does With Each Frame

    When Dynamo's frame evaluator receives a frame, it:

    STEP 1 — CHECK THE CACHE:
        Does this frame have a previously compiled version?
        Check: do all GUARDS for the cached version pass?
        If yes → skip Dynamo entirely, execute the compiled code directly.
        This is the hot path — zero overhead once compiled.

    STEP 2 — SYMBOLIC EVALUATION (first time or cache miss):
        Dynamo re-executes the frame's bytecode using a SYMBOLIC INTERPRETER.
        Instead of executing operations with real tensors, it runs them with
        SYMBOLIC TENSORS (objects that record what operation was performed).

        The symbolic interpreter handles every CPython bytecode instruction:
            LOAD_FAST:     load local variable → create symbolic proxy if tensor
            BINARY_OP:     tensor + tensor → record Add node in graph
            CALL_FUNCTION: function call → analyse callee
            JUMP_IF_TRUE:  branching → evaluate condition concretely
            BUILD_LIST:    Python list construction → track as symbolic list

    STEP 3 — GRAPH BREAK DETECTION:
        During symbolic evaluation, Dynamo monitors for graph breaks.
        A graph break occurs when:
            The code calls a function Dynamo cannot trace through.
            The code branches on a tensor's VALUE (not shape).
            The code uses a Python object Dynamo cannot model.
            The code has a side effect that must happen immediately.
        At a graph break, Dynamo:
            a. Stops the current graph recording.
            b. Compiles what has been recorded so far.
            c. Emits bytecode that calls the compiled graph.
            d. Resumes normal Python execution for the uncompilable fragment.
            e. Resumes graph recording after the break.

    STEP 4 — BYTECODE REWRITING:
        After tracing, Dynamo rewrites the frame's bytecode.
        The new bytecode:
            Calls the compiled backend functions instead of individual ops.
            Evaluates guards at runtime to detect when recompilation is needed.
            Handles graph breaks seamlessly with fallback code.

### The Symbolic Interpreter: Tracking Tensor Operations

    Dynamo's symbolic interpreter uses PROXY OBJECTS to trace through
    Python code without actually executing tensor operations.

    When Dynamo sees:    x = torch.relu(input)
    It does NOT:         call the CUDA relu kernel
    It DOES:             create a Proxy(relu, [input_proxy]) node in the FX graph

    The Proxy object:
        Wraps a node in the FX Graph being built.
        Implements all tensor methods (forward to graph recording).
        Knows its shape and dtype (inferred from the input's shape/dtype).
        Raises GraphBreak if asked to do something unrepresentable.

    Python control flow:
        if x > 0:         ← x is a tensor, > compares a tensor to 0
            y = x * 2

        Dynamo evaluates this CONCRETELY using the real input value:
            If the real input x IS > 0:
                Record: y = x * 2 in the graph.
                Add a GUARD: "x > 0" must be True at runtime.
            If the real input x is NOT > 0:
                Do NOT record y = x * 2.
                Add a GUARD: "x > 0" must be False at runtime.

    This is the key distinction from torch.fx's symbolic trace:
        torch.fx traces with fake values → gets the wrong branch
        Dynamo traces with real values → gets the ACTUAL branch
                                       → records guards to detect when it changes

### Proxies, VariableTracker, and the State Machine

    Internally, Dynamo represents Python objects as VariableTracker subclasses.
    Every type of Python object has a corresponding VariableTracker:

        TensorVariable:      tracks a tensor (records ops to FX graph)
        ConstantVariable:    tracks a Python constant (int, float, bool, str)
        ListVariable:        tracks a Python list (elements are VariableTrackers)
        TupleVariable:       tracks a tuple
        NNModuleVariable:    tracks a nn.Module (special handling for parameters)
        UserFunctionVariable:tracks a user-defined Python function (trace into it)
        BuiltinVariable:     tracks built-in functions (torch.relu, math.sqrt, etc.)
        SkipVariable:        marks code that Dynamo should NOT trace through

    The VariableTracker system is a complete abstract interpreter.
    It handles every Python operator and control flow construct, either by:
        Translating it to an FX graph node (if it's a tensor operation).
        Evaluating it concretely (if it's a Python scalar operation).
        Issuing a graph break (if it's incompatible with graph recording).

    NNModuleVariable special behaviour:
        nn.Module has many Python-level attributes (training flag, buffers,
        named_parameters, etc.) that Dynamo must handle specially.
        Parameters of the module are lifted as graph INPUTS.
        Module attributes that are constants are INLINED into the graph.
        Module attributes that could change (training=True/False) become GUARDS.


##### PART 3 — THE GUARDS SYSTEM: WHEN TO RECOMPILE

### What Guards Are

    A GUARD is a runtime predicate that must be True for a compiled code
    cache entry to be valid.

    Guards are generated automatically during tracing. Every assumption
    that the tracer makes about Python objects must be recorded as a guard,
    so that if that assumption changes, Dynamo knows to recompile.

    Guards are checked BEFORE executing the compiled code.
    Guard checking is designed to be extremely fast (just comparisons).
    Only when a guard fails does Dynamo trigger recompilation.

### The Complete Guard Taxonomy

    TENSOR GUARDS:
        TENSOR_MATCH: tensor has a specific dtype and device.
            guard: "x.dtype == torch.float32 and x.device == 'cuda:0'"
        TENSOR_STRIDE: tensor has specific strides (memory layout).
            guard: "x.stride() == (512, 1)"
            Important because: kernels may assume contiguous layout.
        TENSOR_SHAPE: tensor has specific static dimensions.
            guard: "x.shape == torch.Size([32, 512])"
            Only generated when shapes are assumed static.
            With dynamic shapes enabled: fewer shape guards.

    SCALAR GUARDS:
        EQUALS: a Python scalar equals a specific value.
            guard: "n_heads == 8"
            Generated when a Python int controls loop count or tensor size.
        TYPE_MATCH: an object has a specific Python type.
            guard: "type(activation_fn) == torch.nn.GELU"

    OBJECT IDENTITY GUARDS:
        ID_MATCH: a Python object IS a specific object (same id()).
            guard: "id(model.layer1) == 140234567"
            Generated for closures over Python objects.
        GLOBAL_STATE: PyTorch global flags have specific values.
            guard: "torch.is_grad_enabled() == True"
            guard: "torch.get_default_dtype() == torch.float32"
            guard: "torch.are_deterministic_algorithms_enabled() == False"

    NN.MODULE GUARDS:
        NN_MODULE_TYPE: module has a specific class.
            guard: "type(model) == MyTransformerBlock"
        NN_MODULE_PARAM_SHAPE: module parameters have specific shapes.
            guard: "model.weight.shape == (512, 512)"
        NN_MODULE_TRAINING: module.training flag.
            guard: "model.training == True"

    FUNCTION GUARDS:
        FUNCTION_MATCH: a callable is a specific function object.
            guard: "activation_fn == torch.nn.functional.gelu"
            Generated when a function is called that was captured in a closure.

### Guard Checking at Runtime

    Each compiled cache entry stores its guard list as a Python function.
    When the compiled function is called, Dynamo runs this guard function first:

    # Generated guard function (simplified example):
    def check_guards(x, model, n_heads):
        return (
            x.dtype == torch.float32           # tensor dtype guard
            and x.device.type == 'cuda'        # tensor device guard
            and x.shape[1] == 512              # tensor shape guard (dimension 1)
            and x.is_contiguous()              # tensor stride guard
            and type(model) == TransformerBlock # module type guard
            and model.training == True          # module state guard
            and n_heads == 8                    # scalar value guard
            and torch.is_grad_enabled() == True # global state guard
        )

    if check_guards(x, model, n_heads):
        return compiled_fn(x, model, n_heads)   # fast path
    else:
        return recompile_and_run(x, model, n_heads)  # slow path

    Guard evaluation is O(number_of_guards) short-circuit evaluation.
    For typical models: 10–50 guards checked in ~1 microsecond.
    This overhead is negligible compared to the GPU kernel execution time.

### The Guard Specialisation Dilemma

    There is a fundamental tension in guard design:
        MORE SPECIFIC guards → compiled code is MORE OPTIMISED
                              → MORE recompilations when things change
        LESS SPECIFIC guards → compiled code works for MORE inputs
                              → LESS recompilations

    Examples:

    VERY SPECIFIC (full static shape):
        guard: x.shape == (32, 512, 512)
        Benefit: compiler knows all loop bounds → perfectly tiled kernel
        Cost: recompile for every new batch size

    LESS SPECIFIC (dynamic batch dimension):
        guard: x.shape[1] == 512 and x.shape[2] == 512   (only check last 2 dims)
        Benefit: one compilation works for any batch size
        Cost: kernel cannot assume batch size → less optimal tiling

    DYNAMIC SHAPES mode (torch.compile(dynamic=True)):
        Symbolic shapes propagated through the graph.
        Guards replaced by shape relationships: "dim_0 > 0", "dim_0 == dim_1"
        Full dynamic shape support with shape inference.
        Cost: runtime shape inference overhead; less optimal kernel.

### Recompilation and the Cache

    When guards fail, Dynamo RECOMPILES the function with the new inputs.
    The new compiled version is added to the CACHE.
    Next time, the guard function checks ALL cached versions in order.

    Cache structure (per Python function):
        [
            (guard_fn_1, compiled_fn_1),  # first compilation
            (guard_fn_2, compiled_fn_2),  # recompiled for different input
            (guard_fn_3, compiled_fn_3),  # third compilation
        ]

    Cache lookup: O(n_cached_versions * guards_per_version).
    Default cache limit: 8 compilations per function.
    If exceeded: falls back to eager without warning (configurable).

    RECOMPILATION WARNINGS:
        torch._dynamo.config.verbose = True  → logs every recompilation
        TORCH_LOGS="recompiles" python script.py  → logs at INFO level
        Common causes of unexpected recompiles:
            Different batch sizes without dynamic=True
            Model in training vs eval mode (different guard on .training)
            Changing Python closure variables between calls
            Input on different devices between calls


##### PART 4 — FX GRAPHS: THE OUTPUT OF TORCHDYNAMO

### What an FX Graph Is

    torch.fx (Functional Transformations) is PyTorch's graph representation
    library. An FX Graph is a DIRECTED ACYCLIC GRAPH where:
        Nodes represent operations (function calls, attribute access, etc.)
        Edges represent tensor values flowing between operations
        The graph is a LINEAR sequence of nodes (topological order)

    FX Graph is NOT the same as TorchScript's IR. It is:
        More Pythonic: nodes correspond to Python calls, not bytecode
        More flexible: supports dynamic control flow via graph breaks
        More inspectable: graphs are Python objects, not compiled bytecode
        More transformable: passes are written as Python, not C++

### FX Graph Node Types

    Every node in an FX Graph has one of these opcodes:

    placeholder:
        A graph INPUT — a tensor passed to the compiled function.
        Created for: model parameters, input tensors, captured constants.
        Example: %x : [#users=1] = placeholder[target=x]

    get_attr:
        Accesses a MODULE ATTRIBUTE (parameter or buffer).
        Example: %weight : [#users=1] = get_attr[target=layer.weight]
        Parameters are "lifted" to get_attr nodes rather than captured
        as Python closures (enables weight sharing analysis).

    call_function:
        Calls a TORCH FUNCTION or aten operator.
        Example: %relu : [#users=1] = call_function[target=torch.nn.functional.relu](args=(%x,))
        Example: %mm   : [#users=1] = call_function[target=torch.ops.aten.mm.default](args=(%x, %w))
        The target is a Python callable.

    call_method:
        Calls a METHOD on a tensor.
        Example: %t : [#users=1] = call_method[target=t](args=(%weight,))
        Example: %reshape : [#users=1] = call_method[target=reshape](args=(%x, -1, 512))

    call_module:
        Calls a PyTorch MODULE (nn.Module subclass).
        Example: %layer : [#users=1] = call_module[target=transformer_block](args=(%h,))
        Present before lowering; removed after module inlining pass.

    output:
        The graph OUTPUT — what the compiled function returns.
        Example: %output : [#users=0] = output[args=((%logits,),)]

### The ATen IR: Lowered FX Graphs

    When Dynamo sends a graph to AOTAutograd or a backend compiler,
    the graph goes through DECOMPOSITION — high-level ATen ops are
    broken down into lower-level ATen primitives.

    ATen (A Tensor Library) is PyTorch's low-level tensor operation
    dispatch layer. Every high-level nn.* function ultimately calls
    ATen operators (torch.ops.aten.*).

    HIGH-LEVEL (what Dynamo captures):
        call_function[torch.nn.functional.layer_norm](..., eps=1e-05)

    LOWERED to ATen (after decomposition):
        call_function[aten.mean.dim](input, [2], True)          ; mean
        call_function[aten.sub.Tensor](input, mean)              ; x - mean
        call_function[aten.var.correction](input, [2], ...)      ; variance
        call_function[aten.add.Tensor](var, 1e-05)               ; var + eps
        call_function[aten.rsqrt.default](var_eps)               ; 1/sqrt(var+eps)
        call_function[aten.mul.Tensor](x_centered, rstd)         ; normalise
        call_function[aten.mul.Tensor](normalised, gamma)        ; scale
        call_function[aten.add.Tensor](scaled, beta)             ; shift

    This decomposition is intentional: backends (TorchInductor, TVM, XLA)
    work better with a small set of primitive ops than a large set of
    high-level ops. The ATen IR has ~2,500 operators; the decomposed IR
    used by most backends has ~250 "core" ATen ops.

### FX Graph Transformations: Passes

    FX Graphs are amenable to graph-level transformations (passes).
    These are Python functions that walk the graph and modify it.

    BUILT-IN PASSES:
        InliningTracer:     inlines call_module nodes (expands nn.Module)
        ConstantFolding:    evaluates operations with all-constant inputs
        DeadCodeElimination: removes nodes whose outputs are never used
        ShapeInference:     propagates symbolic shapes through the graph
        Decomposition:      expands high-level ops to ATen primitives

    WRITING A CUSTOM FX PASS:
        def fuse_relu_add(graph: fx.Graph) -> None:
            for node in graph.nodes:
                if (node.op == "call_function"
                        and node.target == torch.relu
                        and len(node.users) == 1):
                    relu_user = next(iter(node.users))
                    if (relu_user.op == "call_function"
                            and relu_user.target == torch.add):
                        # Replace relu → add with fused op
                        with graph.inserting_after(relu_user):
                            fused = graph.call_function(
                                fused_relu_add,
                                args=(node.args[0], relu_user.args[1]))
                        relu_user.replace_all_uses_with(fused)
                        graph.erase_node(relu_user)
                        graph.erase_node(node)

    BACKEND INTEGRATION via fx.passes:
        Most torch.compile backends (TorchInductor, OpenXLA, TVM)
        receive the FX graph and apply their own backend-specific passes
        before generating machine code.


##### PART 5 — AOTAUTOGRAD: AHEAD-OF-TIME DIFFERENTIATION

### The Problem: Lazy Autograd vs Compiled Autograd

    PyTorch's standard autograd builds the backward graph LAZILY:
        During the forward pass, each operation records a closure on
        a dynamically-built Autograd graph (the "tape").
        When .backward() is called, the tape is traversed in reverse,
        calling each closure to compute gradients.

    This lazy approach is incompatible with ahead-of-time compilation:
        The backward graph is not known until the forward pass runs.
        The closures hold references to Python objects (tensors, functions).
        The backward pass itself dispatches operations eagerly.
        There is NO compiled backward graph — it is always interpreted.

    PROBLEM: if you compile the forward pass but not the backward,
    you get a fast forward and a slow, interpreted backward.
    For training workloads, backward is often SLOWER than forward
    (gradients must be computed for every weight).

### AOTAutograd's Solution: Differentiate the Graph, Not the Code

    AOTAutograd (Ahead-Of-Time Autograd, functorch-based) differentiates
    the FX graph BEFORE compilation:

        1. Receive the captured FX graph from TorchDynamo.
        2. Use FUNCTORCH's functional autograd (vmap/grad) to compute
           the SYMBOLIC backward pass of the graph.
        3. Produce a JOINT GRAPH: forward + backward in one FX graph.
        4. Split the joint graph into:
               FORWARD GRAPH: runs during model.forward(), saves activations
               BACKWARD GRAPH: runs during loss.backward(), computes gradients
        5. Send BOTH graphs to the backend compiler.
        6. The backend compiles BOTH as a pair, enabling cross-pass optimisations.

    The joint forward+backward graph:
        ┌─────────────────────────────────────────────────────────────────┐
        │  JOINT GRAPH                                                    │
        │  Inputs: x, W1, b1, W2, b2                                      │
        │  Forward section:                                               │
        │    h1 = x @ W1 + b1                                             │
        │    a1 = relu(h1)                                                │
        │    logits = a1 @ W2 + b2                                        │
        │    loss = cross_entropy(logits, targets)                        │
        │  Saved activations: x, h1, a1                                   │
        │  Backward section:                                              │
        │    d_logits = softmax_cross_entropy_backward(...)               │
        │    d_W2 = a1.T @ d_logits                                       │
        │    d_b2 = d_logits.sum(0)                                       │
        │    d_a1 = d_logits @ W2.T                                       │
        │    d_h1 = relu_backward(d_a1, h1)                               │
        │    d_W1 = x.T @ d_h1                                            │
        │    d_b1 = d_h1.sum(0)                                           │
        │    d_x  = d_h1 @ W1.T                                           │
        │  Outputs: loss, d_W1, d_b1, d_W2, d_b2                          │
        └─────────────────────────────────────────────────────────────────┘

### Why AOTAutograd Matters for Performance

    With AOTAutograd, BOTH forward and backward are compiled:
        Forward: fused GPU kernels (no per-op kernel launch overhead)
        Backward: fused GPU kernels (same benefit as forward)
        JOINT: activations saved from forward are the same buffers
               read by backward — no extra copies, no double allocation

    The activation memory layout:
        AOTAutograd decides WHICH activations to save for the backward pass.
        It can apply REMATERIALISATION (recompute rather than store):
            Cheap ops (relu): recompute in backward instead of storing
            Expensive ops (attention): store the activation
        This is the gradient checkpointing decision, but made automatically
        by analysing the joint graph's memory pressure vs recompute cost.

    SELECTIVE ACTIVATION CHECKPOINTING:
        AOTAutograd can automatically identify which activations to save
        vs recompute by solving a minimum-memory problem on the joint graph.
        No manual torch.utils.checkpoint() calls needed.

### Functorch: Functional Transforms Underpinning AOTAutograd

    Functorch (merged into PyTorch 2.0 as torch.func) provides:
        grad:       compute gradient of a scalar-valued function
        vmap:       vectorise a function over a batch dimension
        jacfwd:     forward-mode Jacobian (like Enzyme's fwddiff)
        jacrev:     reverse-mode Jacobian (like Enzyme's autodiff)
        functionalize: remove in-place operations from a function

    AOTAutograd uses functorch.grad and functorch.functionalize internally:
        functionalize(fn):  converts all in-place ops (add_, relu_) to
                            out-of-place versions (add, relu) so the graph
                            is purely functional (required for compilation).
        grad(fn):           traces through fn symbolically to produce the
                            backward graph as an FX graph.

    FUNCTIONALIZE is critical for compilation:
        PyTorch models are full of in-place ops:
            x.add_(bias)    # adds bias to x in-place
            x.relu_()       # relu in-place
            out.copy_(x)    # copy into pre-allocated buffer
        In-place ops are incompatible with functional graph representation
        (they mutate values that earlier nodes may still reference).
        functionalize() rewrites them:
            x.add_(bias) → x_new = x + bias; aliasing handled explicitly


##### PART 6 — TORCHINDUCTOR: THE DEFAULT BACKEND

### What TorchInductor Is

    TorchInductor is the DEFAULT backend for torch.compile, introduced in
    PyTorch 2.0. It receives the FX graph from AOTAutograd and compiles it
    to optimised machine code.

    TorchInductor is NOT like cuBLAS or cuDNN — it does not use hand-tuned
    library kernels. Instead it generates TRITON kernels (for GPU) or
    C++ code (for CPU) from first principles, using a loop-level IR.

    Triton is OpenAI's kernel-level programming language that lets you
    write GPU kernels in Python-like syntax. TorchInductor generates Triton
    code automatically, which is then compiled by the Triton compiler to PTX.

### TorchInductor's Compilation Pipeline

    FX Graph (ATen ops)
        ↓  Decomposition passes
    Lowered ATen Graph (primitive ops: aten.add, aten.mm, aten.convolution, ...)
        ↓  Lowering to Inductor IR (pointwise, reduction, extern_kernel nodes)
    Inductor IR (high-level loop description)
        ↓  Scheduling (fuse and order operations)
    Scheduled Groups (sets of ops to compute together in one kernel)
        ↓  Codegen (per target)
    GPU → Triton kernel (.py)   CPU → C++ with OpenMP
        ↓  Compilation
    GPU → tritonc → PTX → cubin  CPU → GCC/Clang → .so
        ↓
    Loaded and cached for repeated execution

### Inductor IR: The Three Node Types

    Inductor's IR has just three node types:

    POINTWISE nodes:
        Compute one output element per input element (or per fixed-size tile).
        Fuseable with other pointwise nodes: all ops in one kernel pass.
        Examples: relu, add, multiply, exp, sigmoid, bias_add, layer_norm elements
        Generated Triton: one kernel with all ops inside the main loop body.

    REDUCTION nodes:
        Reduce one or more dimensions (sum, max, argmax, softmax normaliser).
        Cannot be fused with arbitrary ops (need multiple passes).
        Examples: sum, mean, max, logsumexp, norm
        Generated Triton: split into "sum per row" + "broadcast" pattern.

    EXTERN_KERNEL nodes:
        Operations dispatched to external libraries (cuBLAS, cuDNN).
        Cannot be fused into Triton kernels (opaque library call).
        Examples: aten.mm (→ cuBLAS), aten.convolution (→ cuDNN)
        TorchInductor generates the library call + surrounding Triton fusions.

### TorchInductor's Fusion Strategy

    Inductor's scheduler analyses the Inductor IR and decides which nodes
    to fuse together into one kernel:

    FUSION RULES:
        Two pointwise nodes are fused if:
            One's output is the other's only consumer (producer-consumer fusion)
            OR they read the same input (sibling fusion, reads merged)

        A reduction fuses with its pointwise PRODUCER:
            For each output element of the reduction, compute the pointwise
            op on the fly rather than materialising it first.
            Example: sum(exp(x)) → one kernel that computes exp AND accumulates

        An extern_kernel node does NOT fuse but:
            Pointwise nodes using its output are fused into an EPILOGUE.
            This epilogue executes in the same kernel as the extern op returns.
            Example: mm + add_bias + relu → cuBLAS mm with Triton epilogue

    CONCRETE EXAMPLE — transformer FFN:
        x → linear(W1) → gelu → linear(W2) → add(residual) → layer_norm

        Fusion groups:
            Group 1: call cublas mm(x, W1) with pointwise epilogue → gelu output
            Group 2: call cublas mm(gelu_out, W2) with pointwise epilogue → add residual
            Group 3: triton kernel for layer_norm (reduce mean + reduce var + pointwise)

        Kernel launches: 3 (down from 7+ in eager mode)

### Triton Kernel Generation

    For a fused set of pointwise ops, TorchInductor generates Triton Python:

    # TorchInductor-generated Triton kernel for: out = relu(x + bias)
    @triton.jit
    def fused_add_relu_kernel(
        x_ptr, bias_ptr, out_ptr,
        n_elements, BLOCK_SIZE: tl.constexpr):

        pid   = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < n_elements

        x    = tl.load(x_ptr    + offsets, mask=mask)
        bias = tl.load(bias_ptr + offsets % n_bias, mask=mask)
        out  = tl.maximum(x + bias, 0.0)   ; relu(x + bias) in one op
        tl.store(out_ptr + offsets, out, mask=mask)

    BLOCK_SIZE = 1024  ; tuned by Inductor's autotune pass

    Key features of the generated Triton:
        Fused computation: x + bias + relu in ONE memory pass (not three)
        Blocked execution: BLOCK_SIZE tuned for L1 cache and GPU occupancy
        Masked loads/stores: correct handling of non-aligned tensor sizes
        Compiled by Triton → PTX → cubin for the specific GPU architecture

### TorchInductor Autotuning

    TorchInductor has its own lightweight autotuning for BLOCK_SIZE choices.
    It does NOT do the full search that TVM's MetaSchedule does, but:
        For each new kernel shape: tries a small set of BLOCK_SIZE values.
        Measures each on the actual GPU.
        Caches the winning value for this shape.

    This is much faster than TVM's full search (< 1 second vs minutes)
    but finds less optimal configurations for unusual shapes.

    Autotune can be disabled for faster first-compilation:
        torch._inductor.config.max_autotune = False   # no shape search
        torch._inductor.config.max_autotune = True    # full search (default)


##### PART 7 — THE TORCH.COMPILE INTERFACE AND BACKEND ECOSYSTEM

### The torch.compile API

    torch.compile is the single entry point that orchestrates the entire stack:

    BASIC USAGE:
        @torch.compile
        def my_model(x, W):
            return torch.relu(x @ W)

        # Equivalent:
        my_model = torch.compile(my_model)

        # First call: traces with Dynamo + compiles with backend
        # Subsequent calls: guard-check + execute compiled code

    FULL SIGNATURE:
        torch.compile(
            model,
            backend="inductor",           # which backend to use
            mode="default",               # optimisation preset
            fullgraph=False,              # require zero graph breaks
            dynamic=None,                 # dynamic shape handling
            options={},                   # backend-specific options
            disable=False,                # emergency off switch
        )

    MODE PRESETS:
        mode="default":
            Balanced: moderate compile time, good runtime speedup.
            Enables fusion, basic Triton kernels, moderate autotuning.
            Typical: 1.5–2× speedup, 30–60 second compile time.

        mode="reduce-overhead":
            Prioritises reducing kernel launch overhead.
            Enables CUDA graphs (record all kernel launches once,
            replay as one atomic CUDA graph call).
            Best for: batch inference where graphs are static.
            Typical: 2–3× speedup over eager.

        mode="max-autotune":
            Maximum performance, long compile time.
            Full Triton kernel search for all shapes.
            cuBLAS and cuBLASLt algorithm search.
            Typical: 3–5× speedup, 10+ minute compile time.
            Best for: production training with fixed shapes.

        mode="max-autotune-no-cudagraphs":
            max-autotune without CUDA graphs.
            Useful when model has dynamic control flow.

    DYNAMIC SHAPES:
        dynamic=None:    infer from usage (static shapes if consistent)
        dynamic=True:    always use dynamic shapes (symbolic dimensions)
        dynamic=False:   always assume static shapes (more recompilations)

    FULLGRAPH MODE:
        fullgraph=True requires zero graph breaks.
        If any graph break occurs: raise an error (don't silently fall back).
        Use this to guarantee full compilation and debug breaks.
        production training always uses fullgraph=True after debugging.

### The Backend Plugin API

    torch.compile supports pluggable backends. Any function that accepts
    an FX graph and returns a callable can be a backend:

    # Custom backend example:
    def my_backend(gm: torch.fx.GraphModule,
                    example_inputs: List[torch.Tensor]) -> Callable:
        # gm: the FX GraphModule (graph + module context)
        # example_inputs: sample inputs for shape inference

        print(gm.graph)   # inspect the FX graph
        # ... compile gm to fast code ...
        return optimised_function

    # Register and use:
    @torch.compile(backend=my_backend)
    def my_model(x):
        ...

    BUILT-IN BACKENDS:
        "inductor":      TorchInductor → Triton/C++ (default, recommended)
        "eager":         No compilation, just graph capture (debugging)
        "aot_eager":     AOTAutograd + eager, no Inductor (debugging)
        "cudagraphs":    CUDA graphs only, no kernel fusion
        "onnxrt":        Export to ONNX, run with ONNX Runtime
        "openxla":       Route through OpenXLA/XLA backend
        "tvm":           Route through TVM (requires torch_tvm package)
        "ipex":          Intel Extension for PyTorch (Intel CPU/GPU)
        "torchscript":   Convert to TorchScript (legacy interop)

    DEBUGGING BACKENDS:
        "eager":         Run captured ops in Python eagerly (no compilation).
                         Useful to verify Dynamo's graph is correct.
        "aot_eager":     Apply AOTAutograd decompositions but no compilation.
                         Useful to verify backward graph is correct.

### Graph Break Debugging and Minimisation

    Graph breaks are the primary source of torch.compile inefficiency.
    PyTorch provides extensive tooling for diagnosing and fixing them:

    DETECTING GRAPH BREAKS:
        torch._dynamo.explain(fn)(inputs):
            Returns a detailed breakdown: which breaks occurred, where,
            and why. Each break includes the source file and line number.

        TORCH_LOGS="graph_breaks" python script.py:
            Prints a log line for every graph break as it occurs.

        torch.compile(fullgraph=True):
            Raises a BackendCompilerFailed exception at the first graph break.
            Error message explains exactly what caused the break.

    COMMON GRAPH BREAK CAUSES AND FIXES:
        Python print() inside model:
            Cause:  side effect, Dynamo cannot represent in graph
            Fix:    use TORCH_LOGS or remove print()

        .numpy() call inside model:
            Cause:  exits PyTorch tensor world
            Fix:    replace with .detach().cpu() or restructure

        len() on a tensor:
            Cause:  returns Python int, but tensor.size() is symbolic
            Fix:    use tensor.size(0) and guard it, or use dynamic=True

        data-dependent if on a tensor value:
            Cause:  branches on runtime value Dynamo cannot symbolically evaluate
            Fix:    restructure with torch.where(), or accept graph break

        Non-PyTorch library call (scipy, sklearn):
            Cause:  Dynamo cannot trace into arbitrary Python libraries
            Fix:    move the call outside the compiled region, or mark as skip

        Calling a Python function that modifies a global:
            Cause:  side effect
            Fix:    refactor to pure function


##### PART 8 — TORCH.EXPORT: STATIC GRAPH EXTRACTION FOR DEPLOYMENT

### torch.export vs torch.compile

    torch.compile: EAGER-COMPATIBLE compilation.
        Runs in training loops. Handles graph breaks via fallback.
        Graph can change on different inputs (recompilation).
        Cannot be saved to disk for deployment separately.
        Optimal for: training + development.

    torch.export: DEPLOYMENT-ORIENTED static graph extraction.
        Requires ZERO graph breaks (fullgraph equivalent).
        Produces a SERIALISABLE ExportedProgram object.
        Can be saved to disk and loaded on a different machine.
        Can be consumed by IREE, TVM, OpenXLA, or TensorRT.
        Optimal for: inference deployment on diverse hardware.

### The ExportedProgram

    torch.export.export(fn, args, kwargs, constraints) produces:

        ExportedProgram:
            graph_module:   the FX GraphModule (the computation)
            graph_signature: input/output specifications
            state_dict:     model parameters and buffers
            range_constraints: symbolic shape constraints (dynamic dims)
            equality_constraints: shape equality constraints

    The graph in ExportedProgram uses the ATen IR (decomposed operators)
    and is guaranteed to have ZERO Python fallback — every operation is
    representable in the exported graph.

    SAVING AND LOADING:
        # Save
        ep = torch.export.export(model, (x,))
        torch.export.save(ep, "model.pt2")

        # Load on any machine with PyTorch 2.x
        ep_loaded = torch.export.load("model.pt2")
        result = ep_loaded.module()(x)

    DYNAMIC SHAPES IN EXPORT:
        # Specify which dimensions are dynamic:
        from torch.export import Dim
        batch = Dim("batch", min=1, max=1024)
        seq   = Dim("seq",   min=1, max=4096)

        ep = torch.export.export(
            model, (x,),
            dynamic_shapes={"x": {0: batch, 1: seq}})

        # The exported graph contains shape constraints:
        # 1 <= batch <= 1024, 1 <= seq <= 4096
        # Backends can use these for more optimal code generation.

### ExportedProgram → Backend Compilation

    The ExportedProgram can be compiled by any ATen-consuming backend:

    → torch.compile(backend="inductor"):
        Standard path for NVIDIA GPU deployment.

    → tvm.relax.from_exported_program(ep):
        Compile with TVM for edge/mobile deployment.

    → openxla.compile(ep, target="cuda"):
        Compile with OpenXLA for GPU/TPU.

    → torch_tensorrt.compile(ep, ...):
        Compile with TensorRT for maximum NVIDIA inference throughput.

    → export to StableHLO (via torch-mlir):
        torch_mlir.compile(ep, output_type=OutputType.STABLEHLO)
        Enables deployment on any StableHLO consumer (IREE, TVM, etc.)


##### PART 9 — DYNAMIC SHAPES: SYMBOLIC REASONING AT COMPILE TIME

### The Static vs Dynamic Shape Problem

    A fundamental tension in ML compilation:
        STATIC shapes: compiler knows all dimensions → optimal kernels
                       → recompile for every new batch size / sequence length
        DYNAMIC shapes: one compiled version handles any size → less recompilation
                        → compiler cannot assume specific loop bounds → less optimal

    For language model inference, this is critical:
        Training: fixed batch size, padded to fixed sequence length → static OK
        Inference: variable input lengths, streaming generation → need dynamic

### TorchDynamo's Symbolic Shape System

    With dynamic=True (or inferred), Dynamo replaces concrete dimensions
    with SYMBOLIC INTEGERS (SymInt objects) that represent unknown values.

    HOW IT WORKS:
        1. First call: x has shape [32, 512].
           Dynamo marks dim 0 as DYNAMIC (varies across calls).
           Replaces concrete 32 with symbolic SymInt("s0").

        2. Shapes propagate symbolically through the graph:
           x.shape  = [s0, 512]
           x.T      : shape [512, s0]
           x @ W    : shape [s0, 256]   (W is [512, 256])
           x.reshape(-1) : shape [s0 * 512]

        3. When shapes are used in control flow:
           for i in range(x.shape[0]):   ; x.shape[0] = SymInt s0
               This generates a guard: "s0 > 0" and loops s0 times
               symbolically, not unrolled for a specific value.

        4. The backend receives the graph with SymInt dimensions.
           TorchInductor generates Triton with variable grid sizes.
           The generated kernel has s0 as a runtime parameter.

    SYMINT ARITHMETIC:
        s0 + 1  → SymInt("s0 + 1")       (symbolic expression)
        s0 * 2  → SymInt("2*s0")
        s0 % 8  → SymInt("s0 % 8")
        s0 // 2 → SymInt("s0 // 2")

        s0 == s1 → bool False (cannot determine at compile time)
                   BUT: guards can assert s0 == s1 if needed

    SHAPE GUARDS for dynamic shapes:
        Instead of: "x.shape[0] == 32"  (recompile for each batch size)
        Generate:   "x.shape[0] >= 1"   (valid for any non-empty batch)
        OR:         "x.shape[0] % 8 == 0" (must be multiple of 8 for vectorisation)
        OR:         "x.shape[0] == x.shape[2]" (two dims must be equal)

### The Shape Environment

    TorchDynamo maintains a SHAPE ENVIRONMENT — a constraint solver that
    tracks all known relationships between symbolic dimensions.

    When a shape operation produces a constraint:
        guard: x.shape[0] > 0        → s0 > 0  (stored in shape env)
        guard: x.shape[1] == W.shape[0]  → s1 == s2  (equality constraint)

    The shape environment uses the Z3 SMT solver to:
        Determine if two shapes are DEFINITELY EQUAL
        Determine if a dimension is DEFINITELY > 0
        Simplify symbolic expressions (s0 - s0 → 0)
        Detect contradictions (s0 > 10 AND s0 < 5 → error)

    This enables correct dynamic shape handling without unsafe assumptions.


##### PART 10 — PERFORMANCE, PROFILING, AND THE CONNECTED STACK

### Performance Expectations

    torch.compile performance depends heavily on model architecture:

    BEST CASE — small pointwise-heavy models:
        MobileNet-style networks: 2–4× speedup (many elementwise fuseable ops)
        Simple MLP training:      1.5–3× speedup

    TYPICAL CASE — transformer training:
        GPT-2 (124M params):      ~1.5× speedup (matmuls dominate, already fast)
        BERT fine-tuning:         ~1.5-2× speedup
        LLaMA training:           ~1.4–1.8× speedup

    FORWARD PASS vs BACKWARD:
        The gains are typically larger for the backward pass.
        Backward has more elementwise ops (gradient multiplications, additions).
        These fuse well → bigger reduction in kernel launches.

    FIRST CALL OVERHEAD (compilation):
        "default" mode:          30–120 seconds (first-time compile)
        "max-autotune":          5–15 minutes
        Subsequent calls:        <1ms overhead (just guard checking)
        Persistent cache:        enabled via TORCHINDUCTOR_CACHE_DIR
        Saves compiled kernels:  avoids recompilation on restart

### Profiling and Debugging Tools

    torch._dynamo.explain(fn)(*args):
        Returns ExplainOutput with:
            graphs:          list of FX graphs captured
            graph_count:     number of separate graphs (= 1 + graph_breaks)
            graph_break_count: number of graph breaks
            break_reasons:   why each graph break occurred (source line + reason)
            ops_per_graph:   operator count per graph

    TORCH_LOGS environment variable:
        TORCH_LOGS="dynamo"          → Dynamo tracing decisions
        TORCH_LOGS="graph_breaks"    → each graph break with reason
        TORCH_LOGS="recompiles"      → each recompilation with guard failure
        TORCH_LOGS="output_code"     → generated Triton/C++ code
        TORCH_LOGS="schedules"       → Inductor scheduling decisions
        TORCH_LOGS="fusion"          → which ops fused into which kernels
        TORCH_LOGS="+all"            → everything (very verbose)

    torch._inductor.config settings:
        debug = True:              dump all intermediate Inductor IRs
        trace.enabled = True:      save HTML trace of compilation
        max_autotune = True:       exhaustive kernel search
        coordinate_descent_tuning: tune BLOCK_SIZE with hill climbing

    PyTorch profiler integration:
        with torch.profiler.profile(activities=[ProfilerActivity.CUDA]):
            output = compiled_model(x)
        # Shows which compiled kernels ran, their duration, memory usage

### How TorchDynamo Connects to the Full Stack

    LLVM (module 01):
        TorchInductor generates C++ for CPU targets.
        The C++ uses intrinsics that GCC/Clang lower to AVX-512 via LLVM.
        Dynamo itself is compiled as a C++ extension using LLVM toolchain.
        torch-mlir's StableHLO output is passed through LLVM by IREE/XLA.

    MLIR (module 02):
        torch-mlir converts Dynamo's FX graphs to MLIR dialects.
        Path: FX Graph → Torch dialect → StableHLO or linalg → LLVM IR
        This enables deploying PyTorch models with IREE or TVM.

    Enzyme (module 04):
        Enzyme-JAX differentiates at the LLVM IR level.
        AOTAutograd (TorchDynamo's differentiator) works at the FX level.
        They are parallel approaches to the same problem.
        For differentiating through custom C++ kernels called from PyTorch:
            Enzyme is the right tool (differentiates the C++ code itself).
            AOTAutograd cannot differentiate through opaque C++ extensions.

    XLA/OpenXLA (modules 05-06):
        torch.compile(backend="openxla") routes Dynamo's FX graph to XLA.
        This enables PyTorch models on Google TPUs via torch_xla.
        torch.export → StableHLO → XLA is the deployment path.
        OpenXLA's PJRT provides the device backend for torch.compile on TPUs.

    StableHLO (module 07):
        torch.export → torch-mlir → StableHLO is the portable export path.
        Any StableHLO consumer (IREE, TVM, OpenVINO) can then deploy
        the PyTorch model on any supported hardware.

    TVM (module 08):
        TVM can serve as a torch.compile backend via the torch_tvm package.
        TVM's MetaSchedule finds better schedules than Inductor for edge hardware.
        Dynamo captures the graph; TVM compiles it with full autotuning.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  torch.compile COMPLETE STACK                                       │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Python model code (any valid PyTorch)                              │
    │       ↓  @torch.compile                                             │
    │  TORCHDYNAMO (this module)                                          │
    │    CPython frame hook → bytecode interception                       │
    │    Symbolic evaluation → FX graph extraction                        │
    │    Guards generation → cache management                             │
    │       ↓  FX graph (ATen ops)                                        │
    │  AOTAUTOGRAD (this module)                                          │
    │    functionalize → remove in-place ops                              │
    │    grad(fn) → symbolic backward graph                               │
    │    joint forward+backward FX graph                                  │
    │       ↓  joint FX graph                                             │
    │  BACKEND (pluggable)                                                │
    │    TorchInductor → Triton (GPU) / C++ (CPU)    [DEFAULT]            │
    │    OpenXLA       → HLO → cubin/TPU                                  │
    │    TVM           → MetaSchedule → Triton/LLVM                       │
    │    ONNX Runtime  → cuDNN/OpenVINO                                   │
    │       ↓  compiled kernels                                           │
    │  CUDA / CPU execution                                               │
    └─────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · TorchDynamo Internals — Tracing, Guards, and Graph Breaks": {
        "description": (
            "Explore TorchDynamo's core mechanisms from the inside. "
            "Show how to inspect the FX graph Dynamo captures for any function. "
            "Trace through guards: what they are, which get generated, and why. "
            "Demonstrate graph breaks: causes, detection, and debugging. "
            "Show torch._dynamo.explain() to understand compilation decisions. "
            "Simulate guard checking: the fast path and the recompile path."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TORCHDYNAMO INTERNALS — TRACING, GUARDS, AND GRAPH BREAKS")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    import torch._dynamo as dynamo
    print(f"  PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("  PyTorch not installed: pip install torch")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: How Dynamo captures FX graphs
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Dynamo FX graph capture: what gets recorded")
print("━" * 65)
print()

GRAPH_CAPTURE = """
  HOW TORCHDYNAMO CAPTURES FX GRAPHS — ANNOTATED
  ════════════════════════════════════════════════════════════════

  When @torch.compile (or torch._dynamo.optimize) decorates a function,
  Dynamo intercepts EVERY call to that function via CPython's frame hook.

  WHAT DYNAMO SEES vs WHAT GETS RECORDED:

  Python code:                 FX graph node produced:
  ────────────────────────────────────────────────────────────────
  x @ W                     → call_function[aten.mm.default](x, W)
  x + b                     → call_function[aten.add.Tensor](x, b)
  torch.relu(x)             → call_function[aten.relu.default](x)
  x.view(32, -1)            → call_function[aten.view.default](x, [32,-1])
  model.weight              → get_attr[target=weight]   ← lifted parameter
  n = len(x)                → NOT in graph (Python scalar, becomes guard)
  if x.shape[0] > 1:        → NOT in graph (becomes guard: x.shape[0] > 1)
  print(x.mean())           → GRAPH BREAK (side effect)
  x.numpy()                 → GRAPH BREAK (leaves tensor world)

  THE FX GRAPH TEXT FORMAT:
    graph():
      %x    : [#users=2] = placeholder[target=x]
      %w    : [#users=1] = get_attr[target=weight]
      %mm   : [#users=1] = call_function[target=torch.ops.aten.mm.default]
                            (args=(%x, %w))
      %b    : [#users=1] = get_attr[target=bias]
      %add  : [#users=1] = call_function[target=torch.ops.aten.add.Tensor]
                            (args=(%mm, %b), kwargs={alpha: 1})
      %relu : [#users=1] = call_function[target=torch.ops.aten.relu.default]
                            (args=(%add,))
      return %relu

  READING THE FX GRAPH:
    %name  : SSA value name (like LLVM IR)
    [#users=N]: how many other nodes use this value
    placeholder: a graph INPUT (tensor passed in or lifted parameter)
    get_attr:    access model attribute (parameter/buffer)
    call_function: call a function (aten op, torch.nn.functional.*, etc.)
    output:      the graph's return value(s)
"""
print(GRAPH_CAPTURE)

if HAS_TORCH:
    # ── Capture and print FX graphs for various functions ─────────────────
    backend_graphs = []

    def capturing_backend(gm: torch.fx.GraphModule, example_inputs):
        """A backend that just captures the graph without compiling."""
        backend_graphs.append(gm)
        return gm.forward   # return unchanged (eager execution for now)

    # Test 1: Simple linear + relu
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8)

        def forward(self, x):
            return torch.relu(self.fc(x))

    model1   = SimpleLinear()
    compiled1 = torch.compile(model1, backend=capturing_backend, fullgraph=True)
    x1       = torch.randn(4, 16)
    out1     = compiled1(x1)

    print("  Graph 1: SimpleLinear (Linear + ReLU)")
    if backend_graphs:
        g1 = backend_graphs[-1]
        print(f"  Node count: {len(list(g1.graph.nodes))}")
        for node in g1.graph.nodes:
            if node.op != "output":
                print(f"    {str(node.op):15s} {str(node.target)[:45]:45s}"
                      f"  users={len(node.users)}")
    print()

    # Test 2: Model with a skip connection
    backend_graphs.clear()

    class SkipBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return x + self.W(x)   # residual connection

    model2    = SkipBlock()
    compiled2 = torch.compile(model2, backend=capturing_backend, fullgraph=True)
    x2        = torch.randn(4, 8)
    out2      = compiled2(x2)

    print("  Graph 2: Skip connection (x + W(x))")
    if backend_graphs:
        g2 = backend_graphs[-1]
        for node in g2.graph.nodes:
            if node.op != "output":
                print(f"    {str(node.op):15s} {str(node.target)[:45]:45s}"
                      f"  users={len(node.users)}")
    print()
    print(f"  Key: residual (x) appears as placeholder used TWICE "
          f"(once as skip, once as mm input)")
    print()

else:
    FX_GRAPH_REF = """
  TYPICAL FX GRAPH OUTPUT for Linear + ReLU:

  graph():
    %x           : [#users=1] = placeholder[target=x]
    %weight      : [#users=1] = get_attr[target=fc.weight]
    %bias        : [#users=1] = get_attr[target=fc.bias]
    %linear      : [#users=1] = call_function[target=torch.ops.aten.linear.default]
                                (args=(%x, %weight, %bias))
    %relu        : [#users=1] = call_function[target=torch.ops.aten.relu.default]
                                (args=(%linear,))
    return %relu

  HOW TO SEE THIS YOURSELF:
    backend_graphs = []
    def capturing_backend(gm, example_inputs):
        backend_graphs.append(gm)
        return gm.forward

    @torch.compile(backend=capturing_backend)
    def my_fn(x): ...
    my_fn(sample_input)
    print(backend_graphs[0].graph)
"""
    print(FX_GRAPH_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Guards — what gets generated and why
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Guards: the assumptions backing a compilation")
print("━" * 65)
print()

GUARDS_GUIDE = """
  GUARDS: EVERY ASSUMPTION DYNAMO MAKES BECOMES A GUARD
  ════════════════════════════════════════════════════════════════

  A GUARD is a runtime predicate that must be True for the cached
  compiled code to be valid. If any guard fails → recompile.

  GUARD TYPES AND WHEN THEY APPEAR:
  ─────────────────────────────────────────────────────────────────

  1. TENSOR_MATCH (always generated for every tensor input):
     guard: "isinstance(x, torch.Tensor)"
     guard: "x.dtype == torch.float32"
     guard: "x.device == device(type='cpu')"  (or 'cuda:0')
     guard: "x.requires_grad == False"

  2. TENSOR_SHAPE (for static shape compilation):
     guard: "x.size()[0] == 32"
     guard: "x.size()[1] == 16"
     Only for shapes that affected the compiled code structure.
     With dynamic=True: replaced by range constraints.

  3. TENSOR_STRIDE (for memory layout):
     guard: "x.stride() in {(16, 1)}"
     Generated when the kernel assumes contiguous layout.
     x.is_contiguous() → guard: stride = (N, 1) for a 2D tensor.

  4. GLOBAL_STATE (PyTorch global flags):
     guard: "torch.is_grad_enabled()"
     guard: "torch.are_deterministic_algorithms_enabled() == False"
     guard: "torch.get_default_dtype() == torch.float32"
     Generated for EVERY compilation (these affect kernel behaviour).

  5. PYTHON_SCALAR (when a Python value is used in the graph):
     guard: "n_heads == 8"
     guard: "dropout_rate == 0.1"
     Generated when Python values control tensor shapes or ops.
     These cause recompilation if the value changes!

  6. NN_MODULE (module structure and state):
     guard: "type(model) == MyTransformerBlock"
     guard: "model.training == True"
     guard: "model.weight.shape == torch.Size([512, 512])"

  7. FUNCTION_MATCH (when a callable is captured in a closure):
     guard: "activation_fn is <function gelu at 0x...>"
     Generated when a function is passed as an argument or captured.

  GUARD FAILURE = RECOMPILE:
  ─────────────────────────────────────────────────────────────────
  Scenario:
    @torch.compile
    def fn(x, n):
        return x * n   ; n is a Python int

    fn(torch.randn(4), 2)   ; COMPILE #1: guard n==2
    fn(torch.randn(4), 2)   ; CACHE HIT: guard passes ✅
    fn(torch.randn(4), 3)   ; GUARD FAIL: n != 2 → RECOMPILE
                              ; COMPILE #2: guard n==3
    fn(torch.randn(4), 3)   ; CACHE HIT: guard passes ✅
    fn(torch.randn(4), 5)   ; GUARD FAIL: n != 3 → RECOMPILE
                              ; COMPILE #3: guard n==5

  FIX: mark n as static to prevent recompile on every value:
    @torch.compile
    @torch._dynamo.mark_static_address  ; or restructure
    def fn(x, n: int):
        return x * n

  OR: treat n as a constant by using it in a way that inlines it:
    def make_fn(n):
        @torch.compile
        def fn(x): return x * n    ; n is closed over, constant per closure
        return fn

    fn2 = make_fn(2); fn3 = make_fn(3)   ; separate compilations, no guards
"""
print(GUARDS_GUIDE)

if HAS_TORCH:
    # Demonstrate guard-based recompilation counting
    compile_count = [0]

    def count_backend(gm, example_inputs):
        compile_count[0] += 1
        print(f"    [compile #{compile_count[0]}] triggered for shape={example_inputs[0].shape}, "
              f"dtype={example_inputs[0].dtype}")
        return gm.forward

    @torch.compile(backend=count_backend)
    def simple_fn(x):
        return x * 2 + 1

    print("  Recompilation demo: different shapes/dtypes trigger recompile")
    print()
    calls = [
        (torch.randn(4),    "same shape as first"),
        (torch.randn(4),    "same → cache hit"),
        (torch.randn(8),    "different shape → recompile"),
        (torch.randn(8),    "same as previous → cache hit"),
        (torch.randn(4, 4), "new rank → recompile"),
        (torch.randn(4).half(), "different dtype → recompile"),
        (torch.randn(4),    "back to original → cache hit"),
    ]
    compile_count[0] = 0
    print(f"  {'Call':>4}  {'Shape':>12}  {'Note'}")
    print("  " + "-" * 50)
    for i, (x, note) in enumerate(calls):
        prev = compile_count[0]
        _ = simple_fn(x)
        compiled = compile_count[0] > prev
        flag = "🔄 RECOMPILE" if compiled else "✅ cache hit"
        print(f"  {i+1:>4}  {str(tuple(x.shape)):>12}  {flag}  {note}")
    print(f"\\n  Total compilations: {compile_count[0]} (from {len(calls)} calls)")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Graph breaks — detection and debugging
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Graph breaks: causes, detection, and fixes")
print("━" * 65)
print()

GRAPH_BREAKS = """
  GRAPH BREAKS — THE ENEMY OF FULL COMPILATION
  ════════════════════════════════════════════════════════════════

  A graph break occurs when Dynamo encounters code it cannot represent
  in a compilable FX graph. The effect:
    1. Compile the graph collected so far (region 1).
    2. Execute the incompatible code in eager Python.
    3. Start collecting a new graph (region 2).

  This means: N graph breaks → N+1 compiled regions.
  Each region is compiled and fast. The Python "seams" between regions
  are executed eagerly (usually very cheap Python code).

  COMMON GRAPH BREAK CAUSES:
  ─────────────────────────────────────────────────────────────────

  CAUSE 1: Calling unsupported Python builtins/functions
    def fn(x):
        print(x.mean())   ; side effect → GRAPH BREAK HERE
        return x + 1      ; new graph starts after the print

    FIX: remove print() or use TORCH_LOGS for debugging

  CAUSE 2: Branching on a TENSOR VALUE (not shape)
    def fn(x):
        if x.sum() > 0:   ; x.sum() is a tensor, comparing to Python 0
            return x + 1  ; Dynamo evaluates this concretely with real x
        else:             ; records the taken branch + a guard
            return x - 1  ; this branch is NOT in the graph

    RESULT: no graph break, but guards "x.sum() > 0" added.
    If the branch changes between calls → RECOMPILE (not break).

  CAUSE 3: .numpy() calls
    def fn(x):
        n = x.numpy()       ; GRAPH BREAK — exits tensor world
        return torch.from_numpy(n + 1)  ; new graph after

    FIX: stay in PyTorch: return x + 1

  CAUSE 4: Calling non-traceable Python functions
    import scipy.linalg as la
    def fn(x):
        # Dynamo cannot trace into scipy
        vals = la.eigh(x.numpy())   ; GRAPH BREAK
        return torch.tensor(vals[0])

    FIX: move la.eigh outside the compiled scope,
         or use torch.linalg.eigh instead.

  CAUSE 5: len() on a tensor (if Dynamo can't track it symbolically)
    def fn(x):
        n = len(x)          ; len() on a tensor returns Python int
        for i in range(n):  ; Dynamo unrolls this for the specific n
            x = x + i       ; fine if n is small and constant
        return x

    RESULT: no break if n stays constant. Recompile if n changes.
    FIX with dynamic shapes: torch.compile(dynamic=True)

  CAUSE 6: Data-dependent tensor creation
    def fn(x):
        mask = x > 0
        indices = mask.nonzero()  ; output SHAPE depends on x values
        return x[indices]         ; GRAPH BREAK: dynamic shape

    FIX: restructure with torch.where() for static-shape equivalent.

  HOW TO FIND GRAPH BREAKS:
  ─────────────────────────────────────────────────────────────────
  METHOD 1: torch._dynamo.explain()
    explanation = torch._dynamo.explain(fn)(example_input)
    print(explanation.break_reasons)

  METHOD 2: Environment variable
    TORCH_LOGS="graph_breaks" python my_script.py
    → Prints each break with file:line and reason

  METHOD 3: fullgraph=True
    @torch.compile(fullgraph=True)
    def fn(x): ...
    → Raises error at FIRST graph break with explanation

  METHOD 4: dynamo.config
    torch._dynamo.config.suppress_errors = False  ; raise on all errors
    torch._dynamo.config.verbose = True            ; log everything
"""
print(GRAPH_BREAKS)

if HAS_TORCH:
    # Demonstrate graph break detection
    print("  torch._dynamo.explain() demo:")
    print()

    def fn_with_break(x):
        y = x * 2         ; compilable
        z = y.item()      ; BREAK: .item() returns a Python scalar
        return x + z      ; new graph after the break

    def fn_no_break(x):
        return torch.relu(x * 2 + 1)  ; fully compilable

    try:
        # Test no-break function
        exp_nb = torch._dynamo.explain(fn_no_break)(torch.randn(8))
        print(f"  fn_no_break: {exp_nb.graph_count} graph(s), "
              f"{exp_nb.graph_break_count} break(s)")
        if hasattr(exp_nb, "ops_per_graph"):
            print(f"    ops: {exp_nb.ops_per_graph}")
    except Exception as e:
        print(f"  fn_no_break explain: {e}")

    try:
        # Test function with break
        exp_wb = torch._dynamo.explain(fn_with_break)(torch.randn(8))
        print(f"  fn_with_break: {exp_wb.graph_count} graph(s), "
              f"{exp_wb.graph_break_count} break(s)")
        for i, reason in enumerate(exp_wb.break_reasons or []):
            print(f"    Break {i+1}: {str(reason)[:80]}")
    except Exception as e:
        print(f"  fn_with_break explain: {e}")
    print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · FX Graph Manipulation — Passes, Transformations, and Inspection": {
        "description": (
            "Deep dive into torch.fx: the graph representation underlying TorchDynamo. "
            "Build FX graphs manually and inspect their structure. "
            "Write a custom FX pass: find, replace, and insert nodes. "
            "Show decomposition: how high-level ops lower to ATen primitives. "
            "Demonstrate the difference between Dynamo-captured graphs and fx.symbolic_trace. "
            "Implement dead code elimination, constant folding, and operator fusion as FX passes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  FX GRAPH MANIPULATION — PASSES, TRANSFORMATIONS, AND INSPECTION")
print("=" * 65)
print()

try:
    import torch
    import torch.fx as fx
    import torch.nn as nn
    from torch.fx import symbolic_trace, GraphModule
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
except ImportError:
    HAS_TORCH = False
    print("  PyTorch not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: FX Graph structure deep dive
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — FX Graph structure: nodes, edges, and metadata")
print("━" * 65)
print()

FX_STRUCTURE = """
  TORCH.FX GRAPH REPRESENTATION — COMPLETE REFERENCE
  ════════════════════════════════════════════════════════════════

  torch.fx.Graph: the graph container
  torch.fx.Node:  one computation step (call, placeholder, get_attr, output)
  torch.fx.GraphModule:  nn.Module wrapping a Graph (holds parameters too)

  NODE ANATOMY:
  ─────────────────────────────────────────────────────────────────
  node.op:           opcode string — one of:
                       "placeholder"    ; graph input
                       "get_attr"       ; attribute access (parameters)
                       "call_function"  ; free function call
                       "call_method"    ; method call on a value
                       "call_module"    ; nn.Module forward call
                       "output"         ; graph return value

  node.name:         SSA name string, e.g. "relu", "mm_1", "add_2"
                     unique within the graph; used as %name in text

  node.target:       what is being called:
    call_function:   a Python callable  (torch.ops.aten.relu.default)
    call_method:     a string           ("t", "reshape", "contiguous")
    call_module:     a string key       ("layers.0", "attn")
    get_attr:        a string attr path ("weight", "encoder.layer.0.weight")
    placeholder:     a string name      ("x", "input_ids")
    output:          None

  node.args:         tuple of positional arguments (can contain other Nodes)
  node.kwargs:       dict of keyword arguments
  node.users:        dict of {consuming_node: None}  — who uses this node
  node.meta:         metadata dict  {"val": FakeTensor, "stack_trace": ...}

  GRAPH INVARIANTS:
    - Nodes are stored in topological order (a node appears after its args)
    - SSA form: each node defines exactly one value
    - The output node is always last
    - Placeholder nodes are always first
    - No cycles (it IS a DAG)

  GRAPH AS CODE:
    Every FX Graph has a .code property that generates equivalent Python:

    def forward(self, x):
        weight = self.weight
        mm = torch.ops.aten.mm.default(x, weight)
        relu = torch.ops.aten.relu.default(mm)
        return relu

    This code property is used for debugging and for generating
    Python implementations of the compiled graph.
"""
print(FX_STRUCTURE)

if HAS_TORCH:
    # Build a graph and inspect every property
    class InspectableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.W1 = nn.Parameter(torch.randn(8, 16))
            self.b1 = nn.Parameter(torch.zeros(8))

        def forward(self, x):
            h  = x @ self.W1.t() + self.b1
            return torch.relu(h)

    model  = InspectableModel()
    traced = fx.symbolic_trace(model)

    print("  FX graph for: h = relu(x @ W.T + b)")
    print()
    for node in traced.graph.nodes:
        print(f"  ┌─ {node.name}")
        print(f"  │  op:     {node.op}")
        print(f"  │  target: {str(node.target)[:60]}")
        if node.args:
            args_str = ", ".join(
                f"%{a.name}" if isinstance(a, fx.Node)
                else str(a)[:20]
                for a in node.args)
            print(f"  │  args:   ({args_str})")
        if node.users:
            users_str = ", ".join(f"%{u.name}" for u in node.users)
            print(f"  │  users:  {{{users_str}}}")
        print()

    # Show the .code property
    print("  traced.code:")
    for line in traced.code.strip().split("\\n"):
        print(f"    {line}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Writing FX passes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Writing FX passes: DCE, constant folding, op fusion")
print("━" * 65)
print()

FX_PASSES = """
  FX PASS ANATOMY
  ════════════════════════════════════════════════════════════════

  An FX pass is a function: GraphModule → GraphModule (or mutates in-place).
  Three patterns:

  PATTERN 1 — ITERATOR PASS (read-only analysis):
    def count_ops(gm: fx.GraphModule) -> dict:
        counts = {}
        for node in gm.graph.nodes:
            if node.op == "call_function":
                name = str(node.target)
                counts[name] = counts.get(name, 0) + 1
        return counts

  PATTERN 2 — MUTATION PASS (modify nodes in-place):
    def replace_relu_with_gelu(gm: fx.GraphModule) -> None:
        for node in gm.graph.nodes:
            if (node.op == "call_function"
                    and node.target == torch.ops.aten.relu.default):
                node.target = torch.ops.aten.gelu.default
        gm.graph.lint()   ; verify graph is still valid
        gm.recompile()    ; regenerate .code from modified graph

  PATTERN 3 — STRUCTURAL PASS (insert / delete / replace nodes):
    def insert_timing(gm: fx.GraphModule) -> None:
        for node in list(gm.graph.nodes):   ; list() to avoid mutation-iter
            if node.op == "call_function":
                with gm.graph.inserting_before(node):
                    start = gm.graph.call_function(time.perf_counter, ())
                # after the node:
                with gm.graph.inserting_after(node):
                    elapsed = gm.graph.call_function(
                        lambda s: time.perf_counter() - s, (start,))
        gm.recompile()

  KEY GRAPH MUTATION METHODS:
    gm.graph.inserting_before(node):  context manager: insert before node
    gm.graph.inserting_after(node):   context manager: insert after node
    gm.graph.call_function(fn, args, kwargs): create a call_function node
    gm.graph.placeholder(name):              create a new input
    node.replace_all_uses_with(new_node):    redirect all users to new_node
    gm.graph.erase_node(node):              delete a node (must have 0 users)
    gm.graph.eliminate_dead_code():         bulk remove unused nodes
    gm.graph.lint():                        verify graph invariants
    gm.recompile():                         rebuild .code from graph
"""
print(FX_PASSES)

if HAS_TORCH:
    # ── DCE (Dead Code Elimination) ─────────────────────────────────────────
    def dead_code_elimination(gm: fx.GraphModule) -> fx.GraphModule:
        """
        Remove nodes whose outputs are never used.
        FX has a built-in version (graph.eliminate_dead_code()),
        but we implement it manually to show the algorithm.
        """
        changed = True
        removed = 0
        while changed:
            changed = False
            for node in list(gm.graph.nodes):
                if (node.op not in ("output", "placeholder")
                        and len(node.users) == 0):
                    gm.graph.erase_node(node)
                    removed += 1
                    changed = True
        gm.recompile()
        return gm, removed

    # ── Constant folding ─────────────────────────────────────────────────────
    def constant_fold(gm: fx.GraphModule) -> fx.GraphModule:
        """
        Evaluate operations with all-constant inputs at compile time.
        Result is inserted as a new get_attr node.
        """
        folded = 0
        for node in list(gm.graph.nodes):
            if node.op == "call_function":
                # Check if ALL args are get_attr nodes (constants)
                all_const = all(
                    isinstance(a, fx.Node) and a.op == "get_attr"
                    for a in node.args
                    if isinstance(a, fx.Node)
                )
                if all_const and len(node.args) > 0:
                    try:
                        # Evaluate with actual constant values
                        const_args = tuple(
                            getattr(gm, a.target) if isinstance(a, fx.Node)
                            else a for a in node.args)
                        result = node.target(*const_args)
                        if isinstance(result, torch.Tensor):
                            # Insert result as a new constant
                            const_name = f"_folded_{node.name}"
                            setattr(gm, const_name, result)
                            with gm.graph.inserting_before(node):
                                new_node = gm.graph.get_attr(const_name)
                            node.replace_all_uses_with(new_node)
                            gm.graph.erase_node(node)
                            folded += 1
                    except Exception:
                        pass  # cannot fold; leave as is
        gm.graph.lint()
        gm.recompile()
        return gm, folded

    # ── Op counting pass ─────────────────────────────────────────────────────
    def count_ops(gm: fx.GraphModule) -> dict:
        counts = {}
        for node in gm.graph.nodes:
            if node.op == "call_function":
                name = str(node.target).split(".")[-1].replace("default", "").strip(".")
                counts[name] = counts.get(name, 0) + 1
        return counts

    # Test on the model from Section 1
    print("  Testing FX passes on InspectableModel:")
    print()
    model2  = InspectableModel()
    traced2 = fx.symbolic_trace(model2)

    ops_before = count_ops(traced2)
    print(f"  Before passes: {len(list(traced2.graph.nodes))} nodes")
    print(f"  Ops: {ops_before}")

    _, n_folded = constant_fold(traced2)
    print(f"  After constant_fold: {n_folded} ops folded (constants evaluated)")

    _, n_dce = dead_code_elimination(traced2)
    print(f"  After DCE: {n_dce} dead nodes removed")

    ops_after = count_ops(traced2)
    print(f"  After passes: {len(list(traced2.graph.nodes))} nodes")
    print(f"  Ops: {ops_after}")
    print()

    # Verify correctness after passes
    x_test = torch.randn(4, 16)
    with torch.no_grad():
        out_orig  = model2(x_test)
        out_transformed = traced2(x_test)
    err = float(torch.max(torch.abs(out_orig - out_transformed)))
    print(f"  Correctness after passes: max_err={err:.2e} "
          f"{'✅' if err < 1e-5 else '❌'}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: ATen decomposition and the IR lowering chain
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — ATen decomposition: high-level ops → primitives")
print("━" * 65)
print()

DECOMP_GUIDE = """
  ATen DECOMPOSITION — THE LOWERING CHAIN
  ════════════════════════════════════════════════════════════════

  torch.compile lowers operations through multiple ATen layers:

  LAYER 1 — HIGH-LEVEL (what Dynamo captures):
    nn.functional.layer_norm(x, gamma, beta, eps)
    nn.functional.softmax(x, dim=-1)
    nn.functional.cross_entropy(logits, targets)

  LAYER 2 — COMPOSITE (decompose into core ATen ops):
    layer_norm decomposes into:
      mean = aten.mean.dim(x, [last_dim], keepdim=True)
      diff = aten.sub.Tensor(x, mean)
      var  = aten.var.correction(x, [last_dim], keepdim=True)
      rstd = aten.rsqrt.default(aten.add.Tensor(var, eps))
      norm = aten.mul.Tensor(diff, rstd)
      out  = aten.add.Tensor(aten.mul.Tensor(norm, gamma), beta)

    softmax decomposes into:
      max_vals = aten.amax.default(x, [-1], keepdim=True)
      shifted  = aten.sub.Tensor(x, max_vals)
      exps     = aten.exp.default(shifted)
      sum_exps = aten.sum.dim_IntList(exps, [-1], keepdim=True)
      out      = aten.div.Tensor(exps, sum_exps)

  LAYER 3 — CORE ATen (~250 ops):
    The minimal set of ops that ALL backends must handle.
    These correspond to primitive CUDA/CPU kernel implementations.
    TorchInductor knows how to generate Triton for each of these.

  WHY DECOMPOSE?
    1. Backend implementors only need to handle ~250 ops (not 2500+).
    2. Opportunities for cross-op fusion become visible:
       exp + sum can be fused (reduce(exp(x))) → one kernel.
       Before decomposition: softmax is opaque (one call_function node).
       After decomposition:  exp node + sum node → FUSEABLE.
    3. AOTAutograd can differentiate any combination of core ops.
       It doesn't need backward rules for high-level composites.

  HOW TO SEE DECOMPOSITION:
    from torch._decomp import get_decompositions
    from torch._inductor.decomposition import decompose_ops

    # Register decompositions and trace
    with torch._dispatch.python.enable_python_dispatcher():
        with torch._dispatch.python.enable_torch_dispatch_mode(
            torch._dynamo.utils.fake_mode):
            out = nn.functional.layer_norm(x, ...)

    # Or use torch.compile with backend="aot_eager":
    @torch.compile(backend="aot_eager")   ; AOT only, no Inductor
    def fn(x, gamma, beta):
        return nn.functional.layer_norm(x, x.shape[-1:], gamma, beta)

    # The aot_eager backend prints the decomposed graph.
"""
print(DECOMP_GUIDE)

if HAS_TORCH:
    decomp_graphs = []

    def aot_capturing_backend(gm, example_inputs):
        decomp_graphs.append(gm)
        return gm.forward

    try:
        @torch.compile(backend=aot_capturing_backend)
        def layer_norm_fn(x, gamma, beta):
            return torch.nn.functional.layer_norm(
                x, x.shape[-1:], gamma, beta, eps=1e-5)

        x     = torch.randn(4, 8)
        gamma = torch.ones(8)
        beta  = torch.zeros(8)
        _     = layer_norm_fn(x, gamma, beta)

        if decomp_graphs:
            g = decomp_graphs[-1]
            ops = [n for n in g.graph.nodes
                   if n.op == "call_function"]
            print(f"  layer_norm decomposed to {len(ops)} ATen ops:")
            for node in ops:
                tgt = str(node.target)
                tgt_short = tgt.split(".")[-2] + "." + tgt.split(".")[-1] if "." in tgt else tgt
                print(f"    {tgt_short[:55]}")
        print()
    except Exception as e:
        print(f"  Decomposition demo: {e}")
        print()
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · AOTAutograd — Ahead-of-Time Differentiation": {
        "description": (
            "Deep exploration of AOTAutograd: differentiating FX graphs before compilation. "
            "Show the joint forward+backward graph that AOTAutograd produces. "
            "Demonstrate functionalize: converting in-place ops for compilation. "
            "Compare lazy autograd tape (standard PyTorch) vs ahead-of-time graph. "
            "Show activation checkpointing handled automatically by AOTAutograd. "
            "Implement a minimal autograd in pure Python to understand the tape mechanism."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  AOTAUTOGRAD — AHEAD-OF-TIME DIFFERENTIATION")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    from functorch import make_fx
    from torch._functorch.aot_autograd import aot_function
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
except ImportError:
    try:
        import torch
        HAS_TORCH = True
        print(f"  PyTorch {torch.__version__} (functorch may not be available)")
    except ImportError:
        HAS_TORCH = False
        print("  PyTorch not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Understanding the lazy autograd tape
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The lazy autograd tape vs ahead-of-time graph")
print("━" * 65)
print()

AUTOGRAD_COMPARE = """
  LAZY AUTOGRAD (standard PyTorch) vs AOTAUTOGRAD
  ════════════════════════════════════════════════════════════════

  STANDARD PYTORCH — LAZY TAPE (per-operation):
  ─────────────────────────────────────────────────────────────────

  Forward pass (builds tape dynamically):
    x.requires_grad = True

    Step 1:  h = x @ W + b
             → creates MmBackward node, stores (x, W) reference
             → creates AddmmBackward node, stores (1.0,) reference
    Step 2:  a = relu(h)
             → creates ReluBackward0 node, stores (h > 0) mask
    Step 3:  loss = a.sum()
             → creates SumBackward0 node, stores (a.shape,)

  Tape structure (lives in Python heap):
    SumBackward0
      ↑ ReluBackward0(saved: relu_mask)
        ↑ AddmmBackward(saved: x, W, bias_shape)
          ↑ input x (leaf)

  Backward pass (traverses tape):
    SumBackward0.backward(grad=1.0)
      → d_a = ones_like(a)
    ReluBackward0.backward(d_a)
      → d_h = d_a * relu_mask  (mask reloaded from closure)
    AddmmBackward.backward(d_h)
      → d_W = x.T @ d_h        (x reloaded from closure)
      → d_x = d_h @ W.T        (W reloaded from closure)

  PROBLEM: the backward traversal happens in Python at runtime.
           Each backward node is a Python closure → Python overhead.
           Cannot be compiled because the structure is built dynamically.

  AOTAUTOGRAD — AHEAD-OF-TIME GRAPH (pre-computed):
  ─────────────────────────────────────────────────────────────────

  AOTAutograd traces the forward function symbolically using functorch.
  functorch.grad(fn) computes the symbolic backward at TRACE TIME.
  The result is a JOINT GRAPH containing both forward and backward.

  joint_graph:
    Inputs: x, W, b, targets
    ── FORWARD SECTION ──────────────────────────────────────────────
    h     = aten.addmm(b, x, W.T)     ; x @ W + b
    mask  = aten.gt(h, 0)             ; h > 0 (relu mask, SAVED for backward)
    a     = aten.mul(h, mask)         ; relu(h) = h * (h>0)
    loss  = aten.cross_entropy(a, targets)
    ── SAVED ACTIVATIONS ────────────────────────────────────────────
    saved = (x, mask, a)              ; minimum needed for backward
    ── BACKWARD SECTION ─────────────────────────────────────────────
    d_a   = aten.cross_entropy_backward(loss_grad, a, targets)
    d_h   = aten.mul(d_a, mask)       ; relu backward: d_h = d_a * mask
    d_W   = aten.mm(x.T, d_h)        ; d_W = x^T @ d_h
    d_b   = aten.sum(d_h, dim=0)     ; d_b = sum(d_h)
    d_x   = aten.mm(d_h, W)          ; d_x = d_h @ W^T
    return loss, d_x, d_W, d_b       ; all in one graph

  KEY DIFFERENCES:
    Lazy:    backward built at runtime    AOTAutograd: backward built at compile
    Lazy:    each node is a Python obj    AOTAutograd: entire backward is an FX graph
    Lazy:    cannot fuse across backward  AOTAutograd: backend fuses fwd+bwd together
    Lazy:    activation reloaded per-use  AOTAutograd: planner decides what to save
"""
print(AUTOGRAD_COMPARE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Building a minimal autograd in Python (tape-based)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Minimal tape-based autograd (like PyTorch's lazy tape)")
print("━" * 65)
print()

TAPE_THEORY = """
  TAPE-BASED AUTOGRAD — HOW PYTORCH BUILDS THE BACKWARD GRAPH
  ════════════════════════════════════════════════════════════════

  The key insight: EVERY differentiable operation registers a BACKWARD
  FUNCTION on the tape. The backward function knows:
    - How many upstream values this op produced (for gradient seeding)
    - How many downstream values this op consumed (for gradient output)
    - What intermediate values to save (the "saved tensors")
    - The gradient formula for this specific op

  This is what makes PyTorch's autograd work for ANY Python code:
    You don't write backward rules for your model — only for PRIMITIVES.
    Composing primitives automatically composes their backward rules.

  Standard backward functions for common ops:

    MatMulBackward:  z = x @ W
      backward(dz):  dx = dz @ W.T;  dW = x.T @ dz

    AddBackward:     z = x + b
      backward(dz):  dx = dz;  db = dz (broadcast-summed if b has fewer dims)

    ReluBackward:    z = relu(x)   [saves: x > 0 mask]
      backward(dz):  dx = dz * (x > 0)

    ExpBackward:     z = exp(x)    [saves: z itself]
      backward(dz):  dx = dz * z

    LogBackward:     z = log(x)    [saves: x]
      backward(dz):  dx = dz / x

    SumBackward:     z = sum(x)
      backward(dz):  dx = broadcast(dz, x.shape)
"""
print(TAPE_THEORY)

class Value:
    """
    Minimal autograd Value (like a scalar PyTorch Tensor with grad).
    Implements forward + backward for scalar operations.
    This is the conceptual model behind PyTorch's autograd tape.
    """
    def __init__(self, data, _children=(), _op="", label=""):
        self.data     = float(data)
        self.grad     = 0.0
        self._backward = lambda: None
        self._prev    = set(_children)
        self._op      = _op
        self.label    = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():           ; AddBackward
            self.grad  += out.grad      ; dx = dz (chain rule: d(x+y)/dx = 1)
            other.grad += out.grad      ; dy = dz
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():           ; MulBackward
            self.grad  += other.data * out.grad   ; d(xy)/dx = y
            other.grad += self.data  * out.grad   ; d(xy)/dy = x
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():           ; ReluBackward
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _backward():           ; ExpBackward (saves output e)
            self.grad += e * out.grad      ; d(exp(x))/dx = exp(x) = e
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data + 1e-8), (self,), "log")
        def _backward():           ; LogBackward
            self.grad += (1.0 / (self.data + 1e-8)) * out.grad
        out._backward = _backward
        return out

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def backward(self):
        """Reverse-mode backward: topological sort + call each _backward."""
        topo, visited = [], set()
        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev: build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

# Test the minimal autograd
print("  Minimal autograd: f(x) = relu(x * 2 + 1) * exp(x)")
print()
x = Value(0.5, label="x")
y = (x * Value(2.0) + Value(1.0)).relu() * x.exp()

print(f"  x = {x.data:.4f}")
print(f"  forward: y = relu(x*2 + 1) * exp(x) = {y.data:.6f}")
y.backward()
print(f"  backward: dy/dx = {x.grad:.6f}")

# Verify with autograd
if HAS_TORCH:
    x_t = torch.tensor(0.5, requires_grad=True)
    y_t = torch.relu(x_t * 2 + 1) * torch.exp(x_t)
    y_t.backward()
    print(f"  PyTorch reference: dy/dx = {x_t.grad.item():.6f}")
    print(f"  Match: {abs(x.grad - x_t.grad.item()) < 1e-5} ✅")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: AOTAutograd in practice
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — AOTAutograd in practice: joint graph inspection")
print("━" * 65)
print()

if HAS_TORCH:
    fwd_graphs = []
    bwd_graphs = []

    def capturing_fwd_bwd(fw_module, bw_module, args, **kwargs):
        """
        AOTAutograd provides both forward and backward graphs
        to the backend via this callback.
        """
        fwd_graphs.append(fw_module)
        bwd_graphs.append(bw_module)
        import torch._functorch.aot_autograd as aot
        return aot.make_boxed_func(fw_module, bw_module)

    try:
        from torch._functorch import aot_autograd

        fwd_captured = []
        bwd_captured = []

        def simple_fwd_compiler(fx_module, _):
            fwd_captured.append(fx_module)
            return fx_module.forward

        def simple_bwd_compiler(fx_module, _):
            bwd_captured.append(fx_module)
            return fx_module.forward

        fn = lambda x, W, b: torch.nn.functional.relu(x @ W + b)

        from torch._functorch.aot_autograd import aot_function
        compiled_fn = aot_function(fn,
                                    fw_compiler=simple_fwd_compiler,
                                    bw_compiler=simple_bwd_compiler)

        x = torch.randn(4, 8,  requires_grad=True)
        W = torch.randn(8, 4,  requires_grad=True)
        b = torch.randn(4,     requires_grad=True)

        out  = compiled_fn(x, W, b)
        loss = out.sum()
        loss.backward()

        if fwd_captured:
            print("  AOTAutograd FORWARD graph:")
            fwd = fwd_captured[0]
            fwd_ops = [n for n in fwd.graph.nodes if n.op == "call_function"]
            for node in fwd_ops:
                tgt = str(node.target).split("aten.")[-1] if "aten." in str(node.target) else str(node.target)[:40]
                print(f"    {tgt[:50]}")
            print()

        if bwd_captured:
            print("  AOTAutograd BACKWARD graph:")
            bwd = bwd_captured[0]
            bwd_ops = [n for n in bwd.graph.nodes if n.op == "call_function"]
            for node in bwd_ops:
                tgt = str(node.target).split("aten.")[-1] if "aten." in str(node.target) else str(node.target)[:40]
                print(f"    {tgt[:50]}")
            print()

    except Exception as e:
        AOT_REF = f"""
  Note: aot_function API varies by PyTorch version ({torch.__version__}).
  ({e})

  CONCEPTUAL AOTAutograd OUTPUT for relu(x @ W + b):

  FORWARD GRAPH (what runs on the forward pass):
    placeholder x, W, b
    mm    = aten.mm.default(x, W)
    add   = aten.add.Tensor(mm, b)
    relu  = aten.relu.default(add)
    ;; Saved for backward:
    save_for_bwd = (add,)     ; only save the pre-relu value (for mask)
    output (relu, save_for_bwd)

  BACKWARD GRAPH (what runs on loss.backward()):
    placeholder grad_out, saved_add
    ; Relu backward: d = grad * (pre_relu > 0)
    mask  = aten.gt.Scalar(saved_add, 0)
    d_add = aten.mul.Tensor(grad_out, mask)
    ; Add backward: identity for each input
    d_mm  = d_add   ; d(mm+b)/d_mm = 1
    d_b   = d_add   ; d(mm+b)/d_b = 1
    ; MatMul backward: dW = x^T @ d_mm, dx = d_mm @ W^T
    d_W   = aten.mm.default(aten.t.default(x), d_mm)
    d_x   = aten.mm.default(d_mm, aten.t.default(W))
    output (d_x, d_W, d_b)

  Both graphs are then compiled by TorchInductor into Triton kernels.
  The saved activation (add) is shared between forward and backward
  without any extra copy — AOTAutograd plans the memory layout.
"""
        print(AOT_REF)
    print()

    # Verify gradients are correct
    print("  Gradient correctness verification:")
    x_v = torch.randn(4, 8)
    W_v = torch.randn(8, 4)
    b_v = torch.randn(4)

    # Eager reference
    x_e = x_v.clone().requires_grad_(True)
    W_e = W_v.clone().requires_grad_(True)
    b_e = b_v.clone().requires_grad_(True)
    torch.nn.functional.relu(x_e @ W_e + b_e).sum().backward()

    # torch.compile
    x_c = x_v.clone().requires_grad_(True)
    W_c = W_v.clone().requires_grad_(True)
    b_c = b_v.clone().requires_grad_(True)
    compiled_v = torch.compile(
        lambda x,W,b: torch.nn.functional.relu(x@W+b).sum(),
        backend="aot_eager")
    compiled_v(x_c, W_c, b_c).backward()

    for name, g_eager, g_compiled in [
        ("d_x", x_e.grad, x_c.grad),
        ("d_W", W_e.grad, W_c.grad),
        ("d_b", b_e.grad, b_c.grad),
    ]:
        if g_eager is not None and g_compiled is not None:
            err = float(torch.max(torch.abs(g_eager - g_compiled)))
            print(f"    {name}: max_err={err:.2e} {'✅' if err < 1e-4 else '❌'}")
        else:
            print(f"    {name}: gradient not computed")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · torch.compile Performance — Benchmarking, Profiling, and Tuning": {
        "description": (
            "Comprehensive torch.compile benchmarking across model types. "
            "Measure the effect of each compilation mode: default, reduce-overhead, max-autotune. "
            "Show CUDA graph capture and when it helps vs hurts. "
            "Profile TorchInductor's Triton kernel generation. "
            "Benchmark transformer training: eager vs compile on forward, backward, and total. "
            "Show the first-call overhead and how to amortise it with persistent cache."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TORCH.COMPILE PERFORMANCE — BENCHMARKING AND PROFILING")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
except ImportError:
    HAS_TORCH = False
    print("  PyTorch not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Benchmarking infrastructure
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Benchmarking infrastructure")
print("━" * 65)
print()

BENCH_THEORY = """
  CORRECT PYTORCH BENCHMARKING RULES
  ════════════════════════════════════════════════════════════════

  RULE 1 — WARMUP: always run at least 3–5 warmup iterations.
    First call: compilation overhead (seconds for torch.compile).
    Second-Nth call: CUDA graph capture, kernel caching (can be slow).
    Nth+ call: true steady-state performance.
    Measure ONLY after warmup!

  RULE 2 — SYNCHRONISE: GPU ops are asynchronous.
    Without synchronisation:  time.perf_counter() measures scheduling, not execution.
    With torch.cuda.synchronize(): waits for all queued GPU work to finish.
    Failure to synchronise → measurements up to 100× too fast (optimistic).

  RULE 3 — DISABLE GRAD for inference benchmarks:
    with torch.no_grad():  → skips building the autograd tape.
    autocast:              → may use lower precision (bf16/fp16) automatically.

  RULE 4 — USE torch.utils.benchmark.Timer (not time.perf_counter directly):
    Handles: warmup, synchronisation, adaptive measurement count.
    Reports: mean, median, std, IQR — not just a single number.

  RULE 5 — MEASURE WHAT MATTERS:
    Throughput:  items/second (for batch inference)
    Latency:     ms/call (for interactive inference)
    Memory:      peak allocated GPU memory
    FLOP/s utilisation: actual vs theoretical peak

  EXAMPLE (correct GPU benchmark):
    WARMUP_ITERS = 10
    BENCH_ITERS  = 100

    # Warmup
    for _ in range(WARMUP_ITERS):
        out = model(x)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        out = model(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / BENCH_ITERS * 1000   ; ms/call
"""
print(BENCH_THEORY)

if HAS_TORCH:
    def benchmark(fn, args, warmup=5, reps=50, sync=True):
        """Correct GPU benchmark with warmup and synchronisation."""
        for _ in range(warmup):
            out = fn(*args)
        if sync and device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn(*args)
        if sync and device == "cuda":
            torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / reps * 1000
        return t

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: Elementwise chain — where fusion wins the most
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 2 — Elementwise fusion: the biggest win for torch.compile")
    print("━" * 65)
    print()

    N = 1_000_000   ; 1M elements = 4MB of float32

    def elementwise_chain(x):
        """7 ops: each reads from and writes to GPU HBM in eager mode."""
        x = torch.exp(x)
        x = torch.log1p(x)               ; log(1+x)
        x = x * 0.5
        x = torch.tanh(x)
        x = x + 0.1
        x = torch.sqrt(torch.abs(x) + 1e-6)
        x = torch.sigmoid(x)
        return x

    x_bench = torch.randn(N, device=device)
    compiled_chain = torch.compile(elementwise_chain)

    # Warmup (triggers compilation on first call)
    _ = compiled_chain(x_bench)
    if device == "cuda": torch.cuda.synchronize()

    t_eager    = benchmark(elementwise_chain, (x_bench,))
    t_compiled = benchmark(compiled_chain,    (x_bench,))

    print(f"  Elementwise chain: 7 ops on {N//1000}K floats ({N*4//1024//1024}MB)")
    print(f"  {'Mode':25s}  {'Time (ms)':>10s}  {'Speedup':>9s}  {'Notes'}")
    print("  " + "-" * 65)
    print(f"  {'Eager':25s}  {t_eager:>10.3f}  {'1.00×':>9s}  "
          f"7 kernel launches, 7× HBM read+write")
    print(f"  {'torch.compile':25s}  {t_compiled:>10.3f}  "
          f"{t_eager/t_compiled:>9.2f}×  1 fused kernel, 1× HBM read+write")
    print()
    print(f"  Why fusion wins here: ops are MEMORY-BANDWIDTH LIMITED.")
    print(f"  GPU peak bandwidth: ~2 TB/s (A100). Each elem = 4 bytes.")
    print(f"  7 passes × {N*4//1024//1024}MB = {7*N*4//1024//1024}MB data moved in eager.")
    print(f"  1 pass   × {N*4//1024//1024}MB = {1*N*4//1024//1024}MB data moved in compiled.")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: Transformer training speedup
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 3 — Transformer training: forward + backward speedup")
    print("━" * 65)
    print()

    class TransformerBlock(nn.Module):
        def __init__(self, d_model=256, n_heads=4, d_ff=512):
            super().__init__()
            self.attn   = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.ff1    = nn.Linear(d_model, d_ff)
            self.ff2    = nn.Linear(d_ff, d_model)
            self.norm1  = nn.LayerNorm(d_model)
            self.norm2  = nn.LayerNorm(d_model)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.norm1(x + attn_out)
            ff = self.ff2(torch.relu(self.ff1(x)))
            x = self.norm2(x + ff)
            return x

    B, S, D = 8, 64, 256
    model_eager   = TransformerBlock(D).to(device)
    model_compiled = torch.compile(TransformerBlock(D).to(device),
                                    mode="default")

    x_tr     = torch.randn(B, S, D, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(model_compiled.parameters(), lr=1e-3)

    def train_step_eager(model, x):
        x = x.detach().requires_grad_(True)
        out  = model(x)
        loss = out.sum()
        loss.backward()
        return loss.item()

    def train_step_compiled(model, x):
        x = x.detach().requires_grad_(True)
        out  = model(x)
        loss = out.sum()
        loss.backward()
        return loss.item()

    # Warmup both
    for _ in range(5):
        train_step_eager(model_eager, x_tr)
        train_step_compiled(model_compiled, x_tr)
    if device == "cuda": torch.cuda.synchronize()

    t_eager_train    = benchmark(lambda: train_step_eager(model_eager, x_tr),
                                  (), warmup=3, reps=30)
    t_compiled_train = benchmark(lambda: train_step_compiled(model_compiled, x_tr),
                                  (), warmup=3, reps=30)

    print(f"  TransformerBlock training: B={B}, S={S}, D={D}")
    print(f"  {'Mode':30s}  {'ms/step':>10s}  {'Speedup':>9s}")
    print("  " + "-" * 55)
    print(f"  {'Eager (no compile)':30s}  {t_eager_train:>10.3f}  {'1.00×':>9s}")
    print(f"  {'torch.compile(default)':30s}  {t_compiled_train:>10.3f}  "
          f"{t_eager_train/t_compiled_train:>9.2f}×")
    print()
    print(f"  Forward + backward both benefit from:")
    print(f"    - LayerNorm decomposed into fuseable ATen ops")
    print(f"    - Attention score + softmax kernel fused")
    print(f"    - All elementwise backward ops fused per layer")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: Compilation modes and their trade-offs
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 4 — Compilation modes: compile time vs runtime")
    print("━" * 65)
    print()

    MODES_GUIDE = """
  TORCH.COMPILE MODES — TRADE-OFFS
  ════════════════════════════════════════════════════════════════

  mode="default":
    What it does: minimal fusion + basic Triton kernels + light autotuning.
    Compile time:  30–90 seconds (first call; cached after).
    Runtime gain:  1.3–2.0× over eager (model-dependent).
    Best for:      development, iterative training, first deployment.

  mode="reduce-overhead":
    What it does: CUDA GRAPH capture on top of default.
    CUDA graphs record all kernel launches as a single replay-able script.
    Eliminates ALL Python overhead from the training loop.
    Compile time:  30–90s (default) + graph capture overhead.
    Runtime gain:  2–3× over eager (especially for small batch sizes).
    Best for:      batch inference with fixed shapes; tight training loops.
    CAVEAT: CUDA graphs require STATIC input shapes and sizes!
            Any in-place modification of persistent buffers needs care.
            Model.training must not change between compilations.

  mode="max-autotune":
    What it does: exhaustive Triton kernel BLOCK_SIZE search.
    For each kernel shape: tries many BLOCK_SIZE configs, keeps best.
    Compile time:  5–30 minutes (per model, done once; cached).
    Runtime gain:  2–5× over eager (best results for matmul-heavy models).
    Best for:      production training with FIXED batch/sequence sizes.
                   One-time compile cost amortised over days of training.

  mode="max-autotune-no-cudagraphs":
    Same as max-autotune but without CUDA graphs.
    Use when model has:  variable input sizes, in-place buffer updates,
                         Python-side control flow between steps.
    Compile time:  same as max-autotune.
    Runtime gain:  slightly less than max-autotune for static-shape cases.

  CHOOSING A MODE:
    Debugging / development:        mode="default" (fast compile, reasonable perf)
    Inference (static batch size):  mode="reduce-overhead" (CUDA graphs win)
    Production training:            mode="max-autotune" (worth the wait)
    Custom ops / dynamic shapes:    mode="default" dynamic=True

  CUDA GRAPH REQUIREMENTS:
    1. Input tensors must have the same shape every call.
    2. Output tensors must be written to the SAME memory location.
    3. No random number generation that changes between calls.
    4. No CPU-GPU synchronisation points inside the captured region.
    5. model.train() / model.eval() must be set BEFORE capture.
"""
    print(MODES_GUIDE)

    print("  Compilation overhead measurement:")
    print()

    def fn_to_compile(x, W):
        return torch.relu(x @ W + x)

    x_comp = torch.randn(64, 128, device=device)
    W_comp = torch.randn(128, 128, device=device)

    # Time first call (includes compilation)
    compiled_fn = torch.compile(fn_to_compile, mode="default")
    t0 = time.perf_counter()
    _  = compiled_fn(x_comp, W_comp)
    if device == "cuda": torch.cuda.synchronize()
    t_first = (time.perf_counter() - t0) * 1000

    # Time subsequent calls
    t_subsequent = benchmark(compiled_fn, (x_comp, W_comp), warmup=2, reps=50)

    # Eager reference
    t_eager_ref = benchmark(fn_to_compile, (x_comp, W_comp), warmup=2, reps=50)

    print(f"  fn: relu(x @ W + x),  shape={x_comp.shape}")
    print(f"  First call (includes compile): {t_first:.1f} ms")
    print(f"  Subsequent calls (cached):     {t_subsequent:.4f} ms")
    print(f"  Eager reference:               {t_eager_ref:.4f} ms")
    print(f"  Steady-state speedup:          {t_eager_ref/t_subsequent:.2f}×")
    print(f"  Break-even after: {t_first/max(t_eager_ref-t_subsequent, 0.001):.0f} calls")
    print(f"  (torch.compile pays off if you run > that many steps)")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · torch.export and the Connected Stack — Deployment Workflows": {
        "description": (
            "Complete torch.export workflow: from PyTorch model to portable deployment artifact. "
            "Show dynamic shapes with Dim constraints: variable batch and sequence lengths. "
            "Export to StableHLO via torch-mlir for cross-compiler deployment. "
            "Demonstrate the backend plugin API: writing a custom torch.compile backend. "
            "Show the full connected stack: how TorchDynamo feeds XLA, TVM, and IREE. "
            "Summarise performance expectations and when to choose each backend."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TORCH.EXPORT AND THE CONNECTED STACK — DEPLOYMENT WORKFLOWS")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
except ImportError:
    HAS_TORCH = False
    print("  PyTorch not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: torch.export — portable static graph extraction
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — torch.export: the deployment-grade export path")
print("━" * 65)
print()

EXPORT_GUIDE = """
  TORCH.EXPORT VS TORCH.COMPILE — THE FULL PICTURE
  ════════════════════════════════════════════════════════════════

  torch.compile:  for TRAINING and DEVELOPMENT
    ✅ Handles graph breaks via eager fallback
    ✅ Supports arbitrary Python in model code
    ✅ Recompiles on shape change (with caching)
    ✅ Zero user code changes required
    ❌ Graph not serialisable to disk
    ❌ Requires PyTorch at inference time
    ❌ Can silently fall back to eager if compilation fails

  torch.export:  for DEPLOYMENT and SERVING
    ✅ Produces a serialisable ExportedProgram
    ✅ Can be run without the original model class definition
    ✅ Consumed by IREE, TVM, TensorRT, OpenXLA
    ✅ Supports dynamic dimensions via Dim() constraints
    ❌ Requires ZERO graph breaks (fullgraph=True equivalent)
    ❌ Requires model to be expressible as ATen IR
    ❌ Dynamic shapes need explicit Dim() annotation

  THE EXPORTEDPROGRAM OBJECT:
  ─────────────────────────────────────────────────────────────────
  ep = torch.export.export(model, (x,))

  ep.graph_module:        the FX GraphModule (the computation graph)
  ep.graph_signature:
    .input_specs:         list of (kind, arg_name) for each graph input
                          kind ∈ {USER_INPUT, PARAMETER, BUFFER, ...}
    .output_specs:        list of (kind, arg_name) for each output
  ep.state_dict:          dict of {param_name: tensor} (weights)
  ep.range_constraints:   {symbol: ValueRanges} for dynamic dims
  ep.equality_constraints: list of (symbol_a, symbol_b) for shape equalities

  ep.module():            returns an nn.Module that runs the graph
  ep.graph:               the underlying fx.Graph object
  torch.export.save(ep, "file.pt2"):  serialise to disk
  torch.export.load("file.pt2"):      deserialise from disk

  WORKFLOW COMPARISON:
  ─────────────────────────────────────────────────────────────────
  Training:    @torch.compile               ; fast, iterative
  Dev check:   torch.export.export(model, example_inputs)  ; verify exportability
  Deployment:  torch.export → save .pt2 → deploy with desired runtime

  The .pt2 format contains:
    - The serialised FX graph (ATen IR)
    - Model parameters (state_dict)
    - Dynamic shape constraints
    - Module metadata
  It does NOT contain: the original Python model class, PyTorch runtime
"""
print(EXPORT_GUIDE)

if HAS_TORCH:
    # Build a small model and export it
    class MLP(nn.Module):
        def __init__(self, n_in=16, n_h=32, n_out=4):
            super().__init__()
            self.fc1 = nn.Linear(n_in, n_h)
            self.fc2 = nn.Linear(n_h, n_out)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = MLP().eval()
    x     = torch.randn(8, 16)

    try:
        ep = torch.export.export(model, (x,))

        print("  Exported model graph:")
        call_nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
        param_nodes = [n for n in ep.graph.nodes if n.op in ("placeholder", "get_attr")]
        print(f"    Ops:    {len(call_nodes)} call_function nodes")
        print(f"    Inputs: {len([n for n in ep.graph.nodes if n.op=='placeholder'])}")
        print()
        for node in call_nodes:
            tgt = str(node.target).split(".")[-1].replace("default", "").strip(".")
            print(f"    {tgt}")
        print()

        # Verify round-trip
        with torch.no_grad():
            out_original = model(x)
            out_exported = ep.module()(x)
        err = float(torch.max(torch.abs(out_original - out_exported)))
        print(f"  Round-trip correctness: max_err={err:.2e} ✅")
        print()

        # Show graph signature
        print(f"  Graph signature:")
        for spec in ep.graph_signature.input_specs[:4]:
            print(f"    {spec}")
        print()

    except Exception as e:
        print(f"  torch.export: {e}")
        print()

    # Dynamic shapes demo
    print("  Dynamic shapes with Dim() constraints:")
    print()

    try:
        from torch.export import Dim

        batch = Dim("batch", min=1, max=256)
        ep_dynamic = torch.export.export(
            model, (x,),
            dynamic_shapes={"x": {0: batch}})

        print(f"  Exported with dynamic batch dimension: {ep_dynamic.range_constraints}")
        # This single export works for any batch size 1-256

        for batch_sz in [1, 4, 32, 128]:
            x_dyn = torch.randn(batch_sz, 16)
            with torch.no_grad():
                out_dyn = ep_dynamic.module()(x_dyn)
            print(f"    batch={batch_sz:3d}: output shape = {out_dyn.shape} ✅")
        print()

    except Exception as e:
        print(f"  Dynamic export: {e}")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Custom backend API
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Writing a custom torch.compile backend")
print("━" * 65)
print()

BACKEND_API = """
  TORCH.COMPILE BACKEND PLUGIN API
  ════════════════════════════════════════════════════════════════

  Any function with this signature is a valid backend:

  def my_backend(
      gm: torch.fx.GraphModule,      ; the captured FX graph
      example_inputs: List[Tensor],  ; sample inputs for shape info
  ) -> Callable:                     ; return an optimised callable

  The backend receives:
    gm.graph:       the fx.Graph (ATen ops, in topological order)
    gm.code:        Python source code of the graph
    example_inputs: a list of tensors with the right shapes and dtypes

  The backend returns a CALLABLE that:
    Takes the same positional args as the original function.
    Returns the same outputs as the original function.
    Ideally executes faster than gm.forward.

  EXAMPLE — a logging backend (for debugging):
    def logging_backend(gm, example_inputs):
        # Print the captured graph
        print("Captured graph:")
        print(gm.graph)
        print(f"Nodes: {len(list(gm.graph.nodes))}")
        # Just return the original forward (no optimisation)
        return gm.forward

    @torch.compile(backend=logging_backend)
    def my_fn(x): return torch.relu(x) + 1

  EXAMPLE — an ONNX export backend:
    import torch.onnx
    def onnx_backend(gm, example_inputs):
        torch.onnx.export(gm, tuple(example_inputs), "model.onnx")
        import onnxruntime as ort
        sess = ort.InferenceSession("model.onnx")
        def run(*inputs):
            np_inputs = {f"arg{i}": x.numpy() for i,x in enumerate(inputs)}
            return [torch.from_numpy(o) for o in sess.run(None, np_inputs)]
        return run

  EXAMPLE — a TVM backend (simplified):
    def tvm_backend(gm, example_inputs):
        import tvm, tvm.relay as relay
        # Convert FX graph to TVM Relay
        mod, params = relay.frontend.from_pytorch(gm, ...)
        # Compile with TVM MetaSchedule
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target="cuda", params=params)
        # Return a callable that uses TVM runtime
        dev = tvm.cuda(0)
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        def tvm_run(*inputs):
            for i, inp in enumerate(inputs):
                module.set_input(f"arg{i}", inp.numpy())
            module.run()
            return [torch.from_numpy(module.get_output(i).numpy())
                    for i in range(module.get_num_outputs())]
        return tvm_run
"""
print(BACKEND_API)

if HAS_TORCH:
    # Demonstrate a simple custom backend that counts operations
    op_counts_log = []

    def op_counting_backend(gm: torch.fx.GraphModule,
                              example_inputs) -> callable:
        """
        Custom backend: counts ops per call and runs eagerly.
        Shows how to introspect the FX graph in a backend.
        """
        op_types = {}
        for node in gm.graph.nodes:
            if node.op == "call_function":
                name = str(node.target).split("aten.")[-1] if "aten." in str(node.target) else str(node.target)
                name = name.split(".")[0]
                op_types[name] = op_types.get(name, 0) + 1
        op_counts_log.append(op_types)

        # Return the original forward (not compiled, just for demo)
        return gm.forward

    @torch.compile(backend=op_counting_backend, fullgraph=True)
    def model_to_inspect(x, W1, W2):
        h = torch.relu(x @ W1)
        return torch.softmax(h @ W2, dim=-1)

    if HAS_TORCH:
        x_i  = torch.randn(4, 8)
        W1_i = torch.randn(8, 16)
        W2_i = torch.randn(16, 4)
        _    = model_to_inspect(x_i, W1_i, W2_i)

        if op_counts_log:
            print("  Custom backend op count for relu(x@W1) → softmax(@W2):")
            for op, count in sorted(op_counts_log[-1].items()):
                print(f"    {op:30s}: {count}")
            print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Connected stack summary
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — TorchDynamo in the full connected compiler stack")
print("━" * 65)
print()

STACK_SUMMARY = """
  TORCHDYNAMO / TORCH.COMPILE CONNECTED STACK POSITION
  ════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                     │
  │  TorchInductor generates C++ for CPU targets → GCC/Clang → LLVM.     │
  │  Triton (GPU) uses LLVM's NVPTX backend to emit PTX from Triton IR.  │
  │  torch.compile backend="inductor" is the primary LLVM consumer.      │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 02: MLIR                                                     │
  │  torch-mlir converts Dynamo FX graphs to MLIR's Torch dialect.       │
  │  Path: FX → Torch dialect → linalg or StableHLO → IREE/XLA.          │
  │  torch.compile with torch-mlir backend routes through MLIR.          │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 04: Enzyme                                                   │
  │  AOTAutograd (Dynamo's differentiator) works at the FX graph level.  │
  │  Enzyme works at the LLVM IR level (after lowering).                 │
  │  For custom C++ ops called from PyTorch: use Enzyme to differentiate.│
  │  For standard PyTorch ops: AOTAutograd differentiates the FX graph.  │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 05/06: XLA / OpenXLA                                         │
  │  torch.compile(backend="openxla") → Dynamo FX → XLA compilation.     │
  │  torch.export → StableHLO → XLA via jax.export compatibility.        │
  │  torch_xla package runs Dynamo-captured graphs on Google TPUs.       │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 07: StableHLO                                                │
  │  torch.export → torch_mlir.compile(OutputType.STABLEHLO).            │
  │  The StableHLO output can go to IREE, TVM, OpenXLA, OpenVINO.        │
  │  This is the portable deployment path for PyTorch models.            │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM                                                      │
  │  torch.compile(backend="tvm") → Dynamo FX → TVM Relax.               │
  │  TVM's MetaSchedule finds better kernels for edge hardware.          │
  │  Best for: deploy Dynamo-trained models on mobile/MCU via TVM.       │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 09: TorchDynamo (THIS MODULE)                                │
  │  Layer 1: Dynamo — Python bytecode interception → FX graph + guards  │
  │  Layer 2: AOTAutograd — FX graph differentiation → joint fwd+bwd     │
  │  Layer 3: Backend — TorchInductor (default) or any plugin            │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(STACK_SUMMARY)

print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Task                    │ API / Tool                             │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Compile a model         │ torch.compile(model)                   │")
print("  │ Inspect captured graph  │ torch._dynamo.explain(fn)(*args)       │")
print("  │ Find graph breaks       │ TORCH_LOGS='graph_breaks' python ...   │")
print("  │ Require zero breaks     │ torch.compile(fullgraph=True)          │")
print("  │ Maximum performance     │ torch.compile(mode='max-autotune')     │")
print("  │ CUDA graphs             │ torch.compile(mode='reduce-overhead')  │")
print("  │ Dynamic shapes          │ torch.compile(dynamic=True)            │")
print("  │ Custom backend          │ torch.compile(backend=my_fn)           │")
print("  │ Export for deployment   │ torch.export.export(model, example)    │")
print("  │ Save exported model     │ torch.export.save(ep, 'model.pt2')     │")
print("  │ Load exported model     │ torch.export.load('model.pt2')         │")
print("  │ Debug recompilations    │ TORCH_LOGS='recompiles' python ...     │")
print("  │ Dump Triton kernels     │ TORCH_LOGS='output_code' python ...    │")
print("  │ Disable compilation     │ torch.compile(disable=True)            │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Key concepts            │                                        │")
print("  │  Graph break            │ Dynamo fallback to eager Python        │")
print("  │  Guard                  │ Runtime predicate for cache validity   │")
print("  │  AOTAutograd            │ Compile the backward pass too          │")
print("  │  TorchInductor          │ Default backend → Triton/C++           │")
print("  │  FX Graph               │ DAG of ATen ops from Dynamo capture    │")
print("  │  SymInt                 │ Symbolic integer for dynamic shapes    │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()
print("  RESOURCES:")
print("  pytorch.org/docs/stable/torch.compile.html")
print("  pytorch.org/docs/stable/export.html")
print("  github.com/pytorch/pytorch/tree/main/torch/_dynamo")
print("  github.com/pytorch/pytorch/tree/main/torch/_inductor")
print("  dev-discuss.pytorch.org — PyTorch developer forum")
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