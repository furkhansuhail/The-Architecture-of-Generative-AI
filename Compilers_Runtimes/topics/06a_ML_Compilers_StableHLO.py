"""
StableHLO — The Stable Portable IR for Machine Learning
=========================================================

StableHLO is an MLIR dialect that defines the semantics of roughly 100
tensor operations used in machine learning. It is the universal interchange
format that sits between ML frameworks (JAX, PyTorch, TensorFlow) and ML
compilers (OpenXLA, IREE, TVM, OpenVINO), playing the same role that LLVM IR
plays between programming languages and CPU hardware targets.

The word "stable" is the entire point. XLA's original HLO had no compatibility
guarantee — a program produced by XLA version N might silently misbehave on
version N+1. StableHLO replaces that with a formal, versioned specification:
any StableHLO program produced today must compile correctly on any StableHLO
consumer released within the next five years.

Understanding StableHLO means understanding three distinct things:

    1. THE SPECIFICATION: a mathematical description of what each operation
       computes, precise enough to write a reference interpreter from scratch.
       Every op has defined semantics for inputs, outputs, shapes, types,
       constraints, and error conditions.

    2. THE MLIR DIALECT: the concrete MLIR implementation of that specification.
       Operations, types, attributes, verification, serialisation, and passes
       that form the actual software artefact consumed and produced by tools.

    3. THE ECOSYSTEM ROLE: the portability contract between producers (JAX,
       TF, torch-mlir) and consumers (XLA, IREE, TVM). StableHLO is the
       handshake at the centre of the OpenXLA project.

In the connected compiler stack:
    LLVM   (module 01) ← StableHLO consumers (XLA, IREE) lower to LLVM IR
    MLIR   (module 02) ← StableHLO IS an MLIR dialect; uses all MLIR infrastructure
    CIRCT  (module 03) ← StableHLO is the entry point for hardware-targeting flows
    Enzyme (module 04) ← Enzyme-MLIR differentiates StableHLO programs
    XLA    (module 05) ← XLA compiles StableHLO via StableHLO → MHLO → HLO
    OpenXLA(module 06) ← StableHLO is the portability core of OpenXLA
    StableHLO (this)   ← the IR specification, dialect, and ecosystem role
    TVM    (module 08) ← TVM's Relax IR imports StableHLO as one input path

"""

import textwrap
import re

TOPIC_NAME   = "StableHLO — The Stable Portable IR for Machine Learning"
DISPLAY_NAME = "06a · StableHLO"
ICON         = "📐"
SUBTITLE     = "Op Semantics, Type System, Broadcasting Rules, and Portability Contract"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE PORTABILITY PROBLEM AND WHY STABLEHLO EXISTS

### The Gap Between Frameworks and Compilers

    ML practitioners work at two very different abstraction levels:

    FRAMEWORK LEVEL (Python, dynamic, shape-flexible):
        model = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        output = model(input_tensor)        # runs immediately
        loss   = criterion(output, target)
        loss.backward()                     # gradients computed lazily

    COMPILER LEVEL (static, shape-known, hardware-targeted):
        All shapes are fixed at compile time.
        No Python objects — just typed tensor operations.
        The compiler must know exactly what to generate for GPU/TPU/CPU.

    Between these levels, every framework historically invented its own IR:
        TensorFlow → TF GraphDef → then XLA HLO
        PyTorch    → TorchScript → then FX Graph → then TorchDynamo IR
        JAX        → Jaxpr → then XLA HLO
        ONNX       → ONNX protobuf graph

    Each of these IRs was incompatible with the others and with each other's
    compiler backends. The consequence:

        A model trained in JAX → exported to HLO → can ONLY be compiled by XLA.
        The same model → cannot be compiled by TVM, IREE, or OpenVINO directly.
        The ONNX alternative requires a lossy conversion losing expressivity.

### Why HLO Alone Was Not Enough

    XLA's HLO (High Level Operations) came close to being that universal IR.
    It already captured the right level of abstraction: tensor operations with
    static shapes, no Python control flow, typed operands. But HLO had
    three fatal limitations as a portability format:

    LIMITATION 1 — No stability guarantee:
        HLO is defined by C++ protobuf structs internal to XLA's codebase.
        These protos change with every XLA release.
        A model serialised to HLO in January may not parse in March.
        No public commitment to keeping old programs working.

    LIMITATION 2 — Not an MLIR dialect:
        HLO predates MLIR. It is not expressed as MLIR operations.
        Tools in the MLIR ecosystem cannot process HLO without a translation.
        Transformations, analysis, serialisation — all require custom code.
        Pattern matching, type inference, verification: all reimplemented.

    LIMITATION 3 — Underspecified semantics:
        Many HLO operations had behaviour that was only documented as "whatever
        XLA does". Edge cases in convolution dimension ordering, reduction
        semantics, scatter update ordering — all defined by implementation.
        A third-party compiler reimplementing HLO had no authoritative spec to
        validate against.

### StableHLO's Founding Principles (OpenXLA RFC, 2022)

    StableHLO was designed to fix all three limitations simultaneously:

    PRINCIPLE 1 — Formal stability contract:
        Once an operation is defined in a released version, its semantics
        never change in a backward-incompatible way.
        A program produced by any version N works on any consumer ≥ N for
        at least five years after N's release.
        Additive changes only: new ops can be added; old ops cannot be removed
        or semantically altered within the compatibility window.

    PRINCIPLE 2 — MLIR dialect (not a separate IR):
        Every StableHLO operation is an MLIR op.
        StableHLO programs are MLIR modules using MLIR's standard SSA form,
        type system, region semantics, attribute model, and serialisation.
        The entire MLIR ecosystem (pass manager, pattern rewriter, FileCheck
        tests, LSP server, binary bytecode) works on StableHLO programs
        without any additional infrastructure.

    PRINCIPLE 3 — Machine-verifiable specification:
        Every operation's semantics are written as a formal specification
        (in the openxla/stablehlo repository) covering:
            - Argument and result types
            - Constraints (what combinations of shapes/types are legal)
            - Semantics (the mathematical definition of what is computed)
            - Error conditions (what constitutes an invalid program)
        A reference interpreter implements exactly this spec, making every
        edge case testable and verifiable.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  WHAT STABLEHLO IS:                                                  │
    │    An MLIR dialect with ~100 ops                                     │
    │    A versioned stability guarantee (5-year forward compat)           │
    │    A formal specification with a reference interpreter               │
    │    The universal ML interchange format                               │
    │                                                                      │
    │  WHAT STABLEHLO IS NOT:                                              │
    │    A runtime (it does not execute programs)                          │
    │    A replacement for XLA's internal HLO (XLA still uses HLO protos)  │
    │    A framework IR (you don't write ML models in StableHLO directly)  │
    │    The same as MHLO (MHLO is XLA-internal, unstable, different ops)  │
    └──────────────────────────────────────────────────────────────────────┘


##### PART 2 — THE STABLEHLO TYPE SYSTEM

### The Fundamental Type: Tensor

    Every value in StableHLO is a TENSOR — a multi-dimensional array of
    elements of a uniform element type. There are no scalars, no lists,
    no dynamic structures. This restriction is deliberate: tensors with
    known element types and shapes can be perfectly planned in memory
    and compiled to efficient hardware kernels.

    A tensor type is written: tensor<{shape}x{element_type}>

    The shape describes zero or more dimensions. The element type
    describes what each element is.

### Tensor Shapes: Static, Dynamic, and Ranked

    STATIC SHAPE (all dimensions known at compile time):
        tensor<4xf32>             1D, 4 elements, float32
        tensor<8x16xf32>          2D, 8 rows × 16 columns, float32
        tensor<2x3x4xbf16>        3D tensor, bfloat16 elements
        tensor<f32>               0D scalar (rank-0 tensor, one element)
        tensor<0xf32>             empty tensor (zero elements, valid!)

    DYNAMIC SHAPE (some dimensions unknown at compile time):
        tensor<?xf32>             1D, unknown number of elements
        tensor<?x16xf32>          2D, dynamic rows, static 16 columns
        tensor<?x?xf32>           2D, both dimensions dynamic
        tensor<2x?x4xf32>         3D, middle dimension dynamic

    UNRANKED TENSOR (rank unknown at compile time):
        tensor<*xf32>             rank unknown, element type known
        (rarely used in practice; most operations require ranked tensors)

    The ? syntax is critical for ML deployment:
        Training: typically uses static shapes (padded to fixed sizes).
        Export:   jax.export supports polymorphic shapes (symbolic 'b').
        Inference:batch size is often dynamic (serving different request sizes).

    StableHLO's dynamic shape support is more expressive than XLA's original
    HLO (which required fully static shapes) and enables:
        - A single exported model handling any batch size
        - Variable-length sequence processing without padding
        - Shape inference through the computation graph

### Element Types: The Complete Catalogue

    FLOATING POINT:
        f64        64-bit IEEE 754 double precision
        f32        32-bit IEEE 754 single precision
        f16        16-bit IEEE 754 half precision
        bf16       16-bit brain float (Google's format: 8e7m = 1 sign + 8 exp + 7 mantissa)
                   Same exponent range as f32, less mantissa precision.
                   Preferred for ML training: no overflow on f32-range gradients.
        f8e4m3fn   8-bit float, 4 exponent bits, 3 mantissa, NaN encoding
                   Used in H100 FP8 tensor cores for fast training
        f8e5m2     8-bit float, 5 exponent bits, 2 mantissa
                   Wider range than f8e4m3fn; better for weight quantisation
        f8e4m3b11fnuz, f8e4m3fnuz, f8e5m2fnuz: FP8 variants without negative zero

    SIGNED INTEGERS:
        si4        4-bit signed integer  (-8 to 7)   — for int4 weight quantisation
        si8        8-bit signed integer  (-128 to 127)
        si16       16-bit signed integer
        si32       32-bit signed integer
        si64       64-bit signed integer

    UNSIGNED INTEGERS:
        ui4        4-bit unsigned  (0 to 15)
        ui8        8-bit unsigned  (0 to 255)
        ui16, ui32, ui64

    COMPLEX:
        c64        complex with f32 real and imaginary parts
        c128       complex with f64 real and imaginary parts

    BOOLEAN:
        i1         1-bit boolean (0 or 1). Used for masks, predicates.

    WHY bf16 DOMINATES ML TRAINING:
        f32: 1 sign + 8 exponent + 23 mantissa = range [-3.4e38, 3.4e38]
        f16: 1 sign + 5 exponent + 10 mantissa = range [-65504, 65504]
             ← too small! gradients overflow during training
        bf16:1 sign + 8 exponent + 7  mantissa = range [-3.4e38, 3.4e38]
             ← same range as f32, just less precision in the mantissa
        bf16 has become the default training dtype for LLMs because:
            - No overflow (same exponent range as f32)
            - Hardware support: Google TPUs, AMD MI300X, NVIDIA A100/H100
            - 2× memory vs f32, enabling larger models

### Tuple Types: Multi-Output Operations

    Some StableHLO operations return multiple tensors.
    These are represented as TUPLE types:
        tuple<tensor<4xf32>, tensor<8xi32>>   two outputs: f32 and i32 tensors
        tuple<tensor<2x3xf32>>               one-element tuple (rarely used)

    Tuple elements are accessed with stablehlo.get_tuple_element:
        %t = stablehlo.tuple %a, %b : (tensor<4xf32>, tensor<8xi32>)
              → tuple<tensor<4xf32>, tensor<8xi32>>
        %first = stablehlo.get_tuple_element %t[0]
              → tensor<4xf32>

    In practice, tuples appear at function boundaries (multiple return values)
    but are rare inside computation bodies (most ops return a single tensor).

### Token Types: Ordering Side Effects

    StableHLO has a special !stablehlo.token type for operations with
    ordering constraints (send/receive, infeed/outfeed):
        %token0 = stablehlo.create_token : !stablehlo.token
        %token1, %result = stablehlo.infeed %token0 : (!stablehlo.token) -> (tensor<4xf32>, !stablehlo.token)
        %token2 = stablehlo.outfeed %result, %token1 : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token

    Tokens enforce ordering without data dependency:
        token1 must complete before token2 starts (by the token chain).
        This models the ordering constraints of device I/O operations.
        Without tokens, a purely functional IR cannot express "write before read".

### Quantised Types (Uniform Quantisation)

    StableHLO supports quantised tensors for deployment optimisation:

        !quant.uniform<i8:f32, 0.1:10>
            Integer storage type:   i8
            Expressed type:         f32  (the "real" float the int represents)
            Scale:                  0.1  (real = (int - zero_point) × scale)
            Zero point:             10   (int value that represents real 0.0)

        !quant.uniform<i8:f32, 0.05>       (zero_point = 0 when omitted)
        !quant.uniform<i4:f32, 0.02:2>     4-bit quantised weights

    Quantisation-aware ops:
        stablehlo.uniform_quantize:   float → quantised int
        stablehlo.uniform_dequantize: quantised int → float

    A quantised matmul in StableHLO:
        %W_q = stablehlo.uniform_quantize %W_f32 : (tensor<KxNxf32>)
                → tensor<KxNx!quant.uniform<i8:f32, 0.01>>
        %out = stablehlo.dot_general %A, %W_q ...
                : (tensor<MxKxf32>, tensor<KxNx!quant.uniform<i8:f32,0.01>>)
                → tensor<MxNxf32>


##### PART 3 — THE STABLEHLO OPERATION VOCABULARY

### Op Categories Overview

    StableHLO defines ~100 operations in these major categories:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Category              │ Count │ Key ops                         │
    ├──────────────────────────────────────────────────────────────────┤
    │  Tensor creation       │   5   │ constant, iota, create_token    │
    │  Elementwise unary     │  16   │ exp, log, tanh, negate, abs...  │
    │  Elementwise binary    │  16   │ add, mul, maximum, compare...   │
    │  Elementwise ternary   │   1   │ select (conditional)            │
    │  Linear algebra        │   2   │ dot_general, convolution        │
    │  Shape manipulation    │  12   │ reshape, transpose, broadcast…  │
    │  Reduction             │   4   │ reduce, reduce_window, argmax   │
    │  Scatter/gather        │   3   │ gather, scatter, dynamic_slice  │
    │  Sorting               │   2   │ sort, top_k                     │
    │  Control flow          │   4   │ while, if, case, call           │
    │  Communication         │   8   │ all_reduce, all_gather, send…   │
    │  Precision/format      │   4   │ convert, bitcast, clamp, pad    │
    │  Quantisation          │   2   │ uniform_quantize, dequantize    │
    └──────────────────────────────────────────────────────────────────┘

### Tensor Creation Ops

    stablehlo.constant {value = dense<...>}:
        Creates a constant tensor from a literal value.
            %c = stablehlo.constant dense<1.0>       : tensor<f32>
            %c = stablehlo.constant dense<[1,2,3,4]> : tensor<4xi32>
            %c = stablehlo.constant dense<[[1.0,0.0],[0.0,1.0]]> : tensor<2x2xf32>
        The value attribute contains a DenseElementsAttr — a compact
        representation of the tensor data, supporting splat (all-same) values:
            dense<0.0>   — all elements are 0.0 (any shape)

    stablehlo.iota {iota_dimension = N}:
        Creates a tensor filled with sequential integers along one dimension.
        Result shape determines the output; iota_dimension selects which
        dimension counts up.
            %r = stablehlo.iota, iota_dimension = 0 : tensor<4xi32>
            ; → [0, 1, 2, 3]

            %r = stablehlo.iota, iota_dimension = 1 : tensor<3x4xi32>
            ; → [[0,1,2,3],
            ;    [0,1,2,3],
            ;    [0,1,2,3]]

            %r = stablehlo.iota, iota_dimension = 0 : tensor<3x4xi32>
            ; → [[0,0,0,0],
            ;    [1,1,1,1],
            ;    [2,2,2,2]]
        Used to generate position embeddings, attention masks, arange-like ops.

### Elementwise Unary Ops: Full List

    All take tensor<*xT> and return tensor<*xT> of the same shape and type.
    Applied independently to each element.

    Arithmetic:
        stablehlo.abs         absolute value: |x|
        stablehlo.negate      negation: -x
        stablehlo.sign        sign function: -1, 0, or +1
        stablehlo.ceil        ceiling: ⌈x⌉  (towards +∞)
        stablehlo.floor       floor:   ⌊x⌋  (towards -∞)
        stablehlo.round_nearest_even  banker's rounding (round half to even)

    Transcendental:
        stablehlo.exponential       e^x
        stablehlo.exponential_minus_one  e^x - 1  (numerically stable for small x)
        stablehlo.log               natural log: ln(x)
        stablehlo.log_plus_one      ln(1+x)      (numerically stable for small x)
        stablehlo.sqrt              square root: √x
        stablehlo.rsqrt             reciprocal square root: 1/√x
        stablehlo.cbrt              cube root: ∛x
        stablehlo.logistic          sigmoid: 1/(1+e^-x)  ← note: logistic, not sigmoid
        stablehlo.tanh              hyperbolic tangent

    Complex:
        stablehlo.real          real part of complex → float
        stablehlo.imag          imaginary part of complex → float
        stablehlo.is_finite     returns i1 mask: true where element is finite
        stablehlo.not           bitwise NOT (integers) or logical NOT (i1)

    Bitwise:
        stablehlo.not           bitwise NOT: ~x
        stablehlo.count_leading_zeros    number of leading zero bits
        stablehlo.popcnt        population count: number of 1 bits

    TYPE-SPECIFIC NOTES:
        stablehlo.logistic computes σ(x) = 1/(1+e^{-x}). This IS sigmoid.
        JAX's jax.nn.sigmoid → stablehlo.logistic
        The name "logistic" comes from the logistic function in statistics.

        stablehlo.round_nearest_even uses IEEE 754 "round half to even":
        0.5 → 0, 1.5 → 2, 2.5 → 2, 3.5 → 4  (not always round-up!)
        This is different from Python's round() and numpy's np.round for .5 cases.

### Elementwise Binary Ops: Full List

    All take two tensors of compatible shapes (see Broadcasting, Part 4)
    and return a tensor of the same shape.

    Arithmetic:
        stablehlo.add           x + y
        stablehlo.subtract      x - y
        stablehlo.multiply      x * y
        stablehlo.divide        x / y
        stablehlo.remainder     x % y  (sign follows dividend, like C fmod)
        stablehlo.power         x^y
        stablehlo.maximum       max(x, y)  elementwise
        stablehlo.minimum       min(x, y)  elementwise

    Bitwise:
        stablehlo.and           x & y
        stablehlo.or            x | y
        stablehlo.xor           x ^ y
        stablehlo.shift_left         x << y
        stablehlo.shift_right_arithmetic  arithmetic right shift (sign-extends)
        stablehlo.shift_right_logical     logical right shift (zero-fills)

    Comparison (returns tensor<same_shape×i1>):
        stablehlo.compare %x, %y {comparison_direction = #stablehlo.comparison_direction<EQ>}
        Directions: EQ, NE, LT, LE, GT, GE
        For floats: NaN comparisons follow IEEE 754 (NaN != NaN, etc.)
        The compare_type attribute controls how to compare:
            FLOAT:   IEEE 754 floating-point comparison
            SIGNED_INT:   signed integer comparison
            UNSIGNED_INT: unsigned integer comparison

    SEMANTICS NOTE on stablehlo.remainder:
        For integers: result has the sign of the dividend (like C %).
            7 % 3  =  1
            -7 % 3 = -1
            7 % -3 =  1
        For floats: IEEE 754 remainder (can differ from Python's % operator).

    SEMANTICS NOTE on stablehlo.power:
        For integers: 0^0 = 1 (defined; not an error).
        For floats:   IEEE 754 pow (0.0^0.0 = 1.0, (-1.0)^inf = 1.0, etc.)

### The Select Op (Ternary)

    stablehlo.select %pred, %on_true, %on_false:
        Elementwise ternary: result[i] = on_true[i] if pred[i] else on_false[i]
        Types: pred must be tensor<*xi1> (boolean mask)
               on_true and on_false must have the same type as result.

        %pred   = stablehlo.compare %x, %zero {direction=GT} : tensor<8xi1>
        %result = stablehlo.select %pred, %x, %zero : tensor<8xf32>
        ; This IS relu: max(x, 0)

    NOTE: select on the ENTIRE value (not per-element) → use stablehlo.if.
    select is ALWAYS fully evaluated: both on_true and on_false are computed
    before the selection. This differs from an if-else branch.

### The Clamp Op

    stablehlo.clamp %min, %operand, %max:
        Elementwise clamp: result[i] = min(max(operand[i], min_val[i]), max_val[i])
        min and max can be scalars (tensor<T>) or same shape as operand.
        Requires: min ≤ max elementwise (undefined behaviour otherwise).

        %result = stablehlo.clamp %zero, %x, %one : (tensor<f32>, tensor<8xf32>, tensor<f32>) → tensor<8xf32>
        ; Clamps x to [0.0, 1.0]: relu with upper bound

    DIFFERENCE from select:
        clamp saturates: values below min → min, values above max → max.
        select picks: from two existing tensors based on a boolean mask.


##### PART 4 — BROADCASTING SEMANTICS: THE COMPLETE RULES

### Why Broadcasting Exists

    ML computations constantly mix tensors of different shapes:
        Adding a bias vector of shape [128] to a batch matrix of shape [32, 128].
        Multiplying attention scores [B, H, S, S] by a scalar scale factor.
        Subtracting row-wise means [B, S] from activations [B, S, D].

    Broadcasting lets a single element "stand in for" many. Without it, every
    such operation would require explicit reshape + tile/repeat, polluting the
    graph with shape manipulation ops.

    StableHLO has TWO DISTINCT broadcasting mechanisms with different rules:

### Broadcasting Mechanism 1: Implicit Numpy-Style Broadcasting

    WHICH OPS USE IT: stablehlo.add, subtract, multiply, divide, maximum, minimum,
    compare, select, and all other elementwise binary ops.

    THE RULE (identical to NumPy broadcasting):
        1. Right-align the shapes.
        2. For each dimension position, the sizes must either be equal,
           or at least one of them must be 1.
        3. The output size for each position is the maximum of the two sizes.
        4. A size-1 dimension is "stretched" to match the other.

    EXAMPLES:
        tensor<4xf32>   +  tensor<4xf32>    → tensor<4xf32>    (exact match)
        tensor<4xf32>   +  tensor<1xf32>    → tensor<4xf32>    (1 broadcasts to 4)
        tensor<4xf32>   +  tensor<f32>      → tensor<4xf32>    (scalar broadcasts)
        tensor<3x4xf32> +  tensor<4xf32>    → tensor<3x4xf32>  (right-align: 3×4 + 4 → 3×4)
        tensor<3x1xf32> +  tensor<1x4xf32>  → tensor<3x4xf32>  (both dims broadcast)
        tensor<2x3x4xf32>+ tensor<3x4xf32>  → tensor<2x3x4xf32>(right-align: 2×3×4 + 3×4)

        ILLEGAL:
        tensor<3xf32>  +  tensor<4xf32>     → ERROR (neither is 1, sizes differ)
        tensor<2x3xf32>+  tensor<3x2xf32>   → ERROR (both dims differ and neither is 1)

    KEY DISTINCTION from MHLO/HLO:
        XLA's original binary ops required EXPLICIT reshape before broadcasting.
        StableHLO allows implicit broadcasting, matching NumPy/JAX semantics.
        This makes JAX → StableHLO lowering straightforward.

### Broadcasting Mechanism 2: broadcast_in_dim (Explicit Axis Mapping)

    WHICH OPS USE IT: stablehlo.broadcast_in_dim is the EXPLICIT broadcast op.
    Also used internally: stablehlo.reduce, stablehlo.gather dimension specs.

    SIGNATURE:
        %result = stablehlo.broadcast_in_dim %operand, dims = [d0, d1, ...]
                  : (tensor<S0xS1x...xT>) → tensor<R0xR1x...xT>

    The dims attribute maps each dimension of the INPUT to a dimension of the
    OUTPUT. Dimensions of the output not listed are "new" dimensions (broadcast).

    EXAMPLES:

    Case 1: Scalar → any shape (most common):
        %zero_scalar = stablehlo.constant dense<0.0> : tensor<f32>
        %zero_4x8    = stablehlo.broadcast_in_dim %zero_scalar, dims=[]
                       : (tensor<f32>) → tensor<4x8xf32>
        ; dims=[] means the scalar has no dimensions to map;
        ; the output shape determines the broadcast extent.
        ; Equivalent to: np.zeros((4,8))

    Case 2: Bias vector [128] → batch matrix [32, 128]:
        %bias_bc = stablehlo.broadcast_in_dim %bias, dims=[1]
                   : (tensor<128xf32>) → tensor<32x128xf32>
        ; dims=[1]: the single dim of bias maps to dim 1 of the output.
        ; dim 0 (size 32) is the new broadcast dimension.
        ; Equivalent to: np.broadcast_to(bias, (32, 128))

    Case 3: Row vector [1, 4] → matrix [3, 4]:
        %r = stablehlo.broadcast_in_dim %row, dims=[0, 1]
             : (tensor<1x4xf32>) → tensor<3x4xf32>
        ; dims=[0,1]: size-1 dim 0 is stretched from 1→3.
        ; Equivalent to: np.broadcast_to(row, (3,4))

    Case 4: Attention mask generation — iota + broadcast:
        ; Generate position indices [0,1,...,S-1] for each batch item
        %positions = stablehlo.iota, iota_dimension=0 : tensor<Sxi32>
        %pos_batch = stablehlo.broadcast_in_dim %positions, dims=[1]
                     : (tensor<Sxi32>) → tensor<Bx SxSxi32>
        ; Each batch item has the same position pattern.

    CONSTRAINTS on broadcast_in_dim:
        dims must be monotonically non-decreasing (cannot permute dimensions).
        If you need to both broadcast AND permute, use transpose first, then broadcast_in_dim.
        The sizes at dims positions must match the input sizes OR be size 1.

    THE SIZE-1 SPECIAL CASE:
        If input dim i has size 1 AND output dim dims[i] has size > 1:
            → broadcast: replicate the element across that dimension.
        If input dim i has size S AND output dim dims[i] also has size S:
            → copy: elements pass through unchanged.
        If input dim i has size S AND output dim dims[i] has size 1:
            → ILLEGAL (cannot squeeze via broadcast_in_dim; use reduce or slice).

### Broadcasting vs Transpose: A Critical Distinction

    broadcast_in_dim can ONLY map output dims in monotonically increasing order.
    It CANNOT permute dimensions.

    WRONG:
        %t = stablehlo.broadcast_in_dim %x, dims=[1, 0]
             : (tensor<3x4xf32>) → tensor<4x3xf32>
        ; ERROR: dims must be increasing (0 < 1 required, but 1 > 0 here)

    CORRECT:
        %t = stablehlo.transpose %x, dims=[1, 0]
             : (tensor<3x4xf32>) → tensor<4x3xf32>
        ; transpose handles permutation

    COMBINED permute + broadcast:
        %t = stablehlo.transpose %x, dims=[1, 0]     ; permute first
             : (tensor<3x4xf32>) → tensor<4x3xf32>
        %b = stablehlo.broadcast_in_dim %t, dims=[1, 2]   ; then broadcast
             : (tensor<4x3xf32>) → tensor<5x4x3xf32>


##### PART 5 — DOT_GENERAL AND CONVOLUTION: THE CORE COMPUTE OPS

### stablehlo.dot_general: The Universal Matrix Multiplication

    dot_general is the single most important operation in modern ML.
    It unifies: vector dot product, matrix-vector multiply, matrix multiply,
    batched matrix multiply, and arbitrary einsum-like contractions.

    SIGNATURE:
        %result = stablehlo.dot_general %lhs, %rhs,
                  lhs_batching_dimensions = [list],
                  rhs_batching_dimensions = [list],
                  lhs_contracting_dimensions = [list],
                  rhs_contracting_dimensions = [list],
                  precision_config = [DEFAULT, DEFAULT]

    THE FOUR DIMENSION ROLES:
        BATCHING dims:     paired (lhs[b0] and rhs[b0] contract independently).
                           Must be same size. Appear in the output.
        CONTRACTING dims:  paired (summed over). Must be same size. Disappear.
        FREE dims (lhs):   remaining dims of lhs. Appear in output (lhs-side).
        FREE dims (rhs):   remaining dims of rhs. Appear in output (rhs-side).

    OUTPUT SHAPE:
        [batching dims] + [lhs free dims] + [rhs free dims]
        In that order always.

    SEMANTICS (mathematical):
        result[b..., i..., j...] = sum_{k...} lhs[b..., i..., k...] * rhs[b..., k..., j...]
        where b... = batching indices, k... = contracting indices,
              i... = lhs free indices, j... = rhs free indices.

    CASE CATALOGUE:

    Vector dot product:  y = sum(a * b)  for a,b : [N]
        lhs_batching=[], rhs_batching=[]
        lhs_contracting=[0], rhs_contracting=[0]
        Input:  tensor<Nxf32>, tensor<Nxf32>
        Output: tensor<f32>   (rank-0 scalar)

    Matrix-vector product:  y[i] = sum_k A[i,k]*x[k]  for A:[M,K], x:[K]
        lhs_batching=[], rhs_batching=[]
        lhs_contracting=[1], rhs_contracting=[0]
        Input:  tensor<MxKxf32>, tensor<Kxf32>
        Output: tensor<Mxf32>

    Matrix multiply:  C[i,j] = sum_k A[i,k]*B[k,j]  for A:[M,K], B:[K,N]
        lhs_batching=[], rhs_batching=[]
        lhs_contracting=[1], rhs_contracting=[0]
        Input:  tensor<MxKxf32>, tensor<KxNxf32>
        Output: tensor<MxNxf32>

    Batched matmul:  C[b,i,j] = sum_k A[b,i,k]*B[b,k,j]  for A,B:[batch,...]
        lhs_batching=[0], rhs_batching=[0]
        lhs_contracting=[2], rhs_contracting=[1]
        Input:  tensor<BxMxKxf32>, tensor<BxKxNxf32>
        Output: tensor<BxMxNxf32>

    Multi-head attention QK^T:  scores[b,h,i,j] = sum_d Q[b,h,i,d]*K[b,h,j,d]
        lhs_batching=[0,1], rhs_batching=[0,1]
        lhs_contracting=[3], rhs_contracting=[3]
        Input:  tensor<BxHxSxDxf32>, tensor<BxHxSxDxf32>
        Output: tensor<BxHxSxSxf32>

    Outer product:  C[i,j] = a[i]*b[j]  (no contracting dims!)
        lhs_batching=[], rhs_batching=[]
        lhs_contracting=[], rhs_contracting=[]
        Input:  tensor<Mxf32>, tensor<Nxf32>
        Output: tensor<MxNxf32>

    PRECISION CONFIG:
        precision_config=[DEFAULT, DEFAULT]:    hardware default precision
        precision_config=[HIGHEST, HIGHEST]:    maximum precision (slower)
        precision_config=[HIGH, HIGH]:          intermediate
        For TPU: HIGHEST uses f32 accumulators for bf16 matmul.
        For GPU: DEFAULT uses tensor cores (Tensor Float 32 or FP16 accumulation).

### stablehlo.convolution: The Generalised N-D Convolution

    convolution is the second most important compute op (for CNNs, audio, etc.).
    It generalises 1D, 2D, and 3D convolution with configurable dimension numbers.

    SIGNATURE (simplified):
        %result = stablehlo.convolution(%input, %kernel)
                  dim_numbers = #stablehlo.conv<[n,f,0,1]x[o,i,0,1]->[n,f,0,1]>
                  window_strides = [1, 1],
                  padding = [[0,0],[0,0]],
                  lhs_dilation = [1, 1],   ; for transposed conv
                  rhs_dilation = [1, 1],   ; for atrous/dilated conv
                  feature_group_count = 1, ; for depthwise conv
                  batch_group_count = 1

    DIMENSION NUMBERS — the crucial attribute:
        The dim_numbers attribute maps the abstract "batch", "output_feature",
        "input_feature", and "spatial" dimensions to the actual tensor axes.

        Format: [input_batch, input_features, spatial_0, spatial_1, ...]
              x [kernel_output_features, kernel_input_features, spatial_0, spatial_1, ...]
             -> [output_batch, output_features, spatial_0, spatial_1, ...]

        NHWC (TensorFlow convention):
            Input:  [N=0, H=1, W=2, C=3]  (batch, height, width, channels)
            Kernel: [H=0, W=1, I=2, O=3]  (spatial_h, spatial_w, in_chan, out_chan)
            Output: [N=0, H=1, W=2, F=3]  (batch, height, width, filters)
            Notation: #stablehlo.conv<[n,0,1,f]x[0,1,i,o]->[n,0,1,f]>

        NCHW (PyTorch/cuDNN convention):
            Input:  [N=0, C=1, H=2, W=3]
            Kernel: [O=0, I=1, H=2, W=3]
            Output: [N=0, F=1, H=2, W=3]
            Notation: #stablehlo.conv<[n,f,0,1]x[o,i,0,1]->[n,f,0,1]>

    KEY ATTRIBUTES:
        window_strides:      how far the kernel slides per step (downsampling).
        padding:             [[pad_low, pad_high], ...] per spatial dimension.
                             VALID padding = [[0,0],...], SAME = computed.
        lhs_dilation:        dilation of the input (inserts zeros between elements).
                             lhs_dilation=[2,2] = transposed/fractionally-strided conv.
        rhs_dilation:        dilation of the kernel (atrous/dilated convolution).
                             rhs_dilation=[2,2] = dilated conv (receptive field grows).
        feature_group_count: groups for depthwise separable conv.
                             feature_group_count = in_channels → depthwise conv.
        batch_group_count:   groups over the batch dimension (rare).

    SEMANTICS:
        output[n, f, p0, p1, ...] = sum_{i, q0, q1, ...}
            input[n,
                  n_group*i_group + i,
                  p0*stride_0 + q0*rhs_dilation_0 - pad_low_0,
                  p1*stride_1 + q1*rhs_dilation_1 - pad_low_1, ...]
            * kernel[f, i, q0, q1, ...]


##### PART 6 — REDUCTION AND SHAPE MANIPULATION OPS

### stablehlo.reduce: Folding Dimensions

    reduce collapses one or more dimensions by repeatedly applying a
    binary function (associative and commutative in practice).

    SIGNATURE:
        %result = stablehlo.reduce(%inputs..., %inits...)
                  applies @computation across dimensions=[d0, d1, ...]
                  : (tensor<...xT>..., tensor<T>...) → (tensor<...xT>...)

    The computation is a region that takes two values of the element type
    and returns one. The initial values serve as the identity element.

    REDUCTION SEMANTICS:
        For reduce(%x, %init) across dimension d:
            result[i0, ..., i_{d-1}, i_{d+1}, ..., iN]
              = fold(init, x[i0, ..., i_{d-1}, :, i_{d+1}, ..., iN])
        where fold applies the computation along the : (full range of dim d).

    COMMON PATTERNS:

    Sum reduction (jnp.sum):
        %init  = stablehlo.constant dense<0.0> : tensor<f32>
        %sum   = stablehlo.reduce(%x, %init) applies @add_fn across dim 1
                 : (tensor<3x4xf32>, tensor<f32>) → tensor<3xf32>
        ; @add_fn: lhs + rhs

    Max reduction (jnp.max):
        %init  = stablehlo.constant dense<-inf> : tensor<f32>
        %mx    = stablehlo.reduce(%x, %init) applies @max_fn across dim 1
                 : (tensor<3x4xf32>, tensor<f32>) → tensor<3xf32>
        ; @max_fn: max(lhs, rhs)

    Argmax (jnp.argmax):
        Combine values AND indices in a single reduce (multi-output reduce):
        %val_init = stablehlo.constant dense<-inf> : tensor<f32>
        %idx_init = stablehlo.constant dense<0>    : tensor<i32>
        %max_val, %max_idx = stablehlo.reduce(%x, %indices, %val_init, %idx_init)
                             applies @argmax_fn across dim 0
        ; @argmax_fn: returns (max_val, argmax_idx) from two candidates

    Reducing multiple dimensions at once:
        %result = stablehlo.reduce(%x, %init) applies @add_fn across dims=[0,2]
                  : (tensor<2x3x4xf32>, tensor<f32>) → tensor<3xf32>
        ; Reduces dimensions 0 and 2, keeps dimension 1.

### stablehlo.reduce_window: Sliding Window Reduction (Pooling)

    reduce_window applies a reduction over a sliding window, implementing
    max pooling, average pooling, and their generalisations.

    SIGNATURE:
        %result = stablehlo.reduce_window(%input, %init) applies @computation
                  window_dimensions = [1, 2, 2, 1],    ; pool over 2x2 spatial
                  window_strides    = [1, 2, 2, 1],    ; stride 2
                  padding           = [[0,0],[0,0],[0,0],[0,0]]  ; no padding

    For a 2D max pool on NHWC input [N=0, H=1, W=2, C=3]:
        window_dimensions = [1, 2, 2, 1]  (pool over 2×2 spatial window)
        window_strides    = [1, 2, 2, 1]  (stride 2 in both spatial dims)
        padding           = [[0,0],[0,0],[0,0],[0,0]]  (no padding)
        computation       = max_fn
        init              = -inf

### Shape Manipulation Ops: The Complete Set

    stablehlo.reshape %x : (tensor<S0x...xSN×T>) → tensor<R0x...×RMxT>
        Reinterprets the element layout. All elements in row-major order;
        new shape must have the same total number of elements.
        tensor<2x3xf32> → tensor<6xf32> → tensor<3x2xf32>

    stablehlo.transpose %x, dims=[p0, p1, ..., pN]:
        Permutes dimensions. dims[i] = j means: output dim i comes from input dim j.
        tensor<2x3x4xf32> with dims=[2,0,1] → tensor<4x2x3xf32>
        Common: dims=[1,0] for matrix transpose.

    stablehlo.slice %x, start_indices=[s...], limit_indices=[l...], strides=[st...]:
        Extract a sub-tensor. Equivalent to Python slicing x[s0:l0:st0, s1:l1:st1, ...].
        All indices are constants (static slicing).
        tensor<8x4xf32>[2:6:1, 0:4:2] → tensor<4x2xf32>

    stablehlo.dynamic_slice %x, %offsets..., slice_sizes=[s...]:
        Slice with a runtime-computed start offset (indices are tensors, not constants).
        slice_sizes are still static (must be constants).
        Used for: indexed attention, beam search, gather-like ops.

    stablehlo.dynamic_update_slice %operand, %update, %offsets...:
        Write %update into %operand at position %offsets.
        Returns a new tensor (purely functional, no in-place mutation).
        Used for: scatter into a tensor at a computed position.

    stablehlo.pad %x, %padding_value, edge_padding_low=[...], edge_padding_high=[...], interior_padding=[...]:
        Pad a tensor. edge_padding adds to the outside; interior_padding
        inserts between elements (creating "holes" filled with padding_value).
        interior_padding=[1,1] on a [3,3] tensor → [3+2, 3+2] = [5,5]
        (one zero inserted between every pair of elements).

    stablehlo.concatenate %a, %b, ..., dimension = d:
        Concatenate tensors along dimension d.
        tensor<3x4xf32> ++ tensor<2x4xf32> along dim 0 → tensor<5x4xf32>

    stablehlo.reverse %x, dimensions=[d...]:
        Reverse (flip) elements along specified dimensions.
        tensor<4xf32> reversed → [x3, x2, x1, x0]

    stablehlo.broadcast_in_dim: (covered in Part 4)

### Gather and Scatter: Indexed Access

    stablehlo.gather: Read from a tensor at dynamically-specified positions.
    This is the general form of embedding lookups, index selects, and gather ops.

    SIMPLIFIED FORM (embedding lookup):
        %emb = stablehlo.gather %table, %indices,
               offset_dims=[1], collapsed_slice_dims=[0],
               start_index_map=[0], index_vector_dim=1,
               slice_sizes=[1, D]
               : (tensor<VxDxf32>, tensor<Nxi32>) → tensor<NxDxf32>
        ; Looks up N rows from a vocabulary table of shape [V, D].
        ; This IS what embedding layers compile to.

    stablehlo.scatter: Write values to dynamically-specified positions.
    Adds updates to specific indices of a base tensor:
        %result = stablehlo.scatter %base, %indices, %updates
                  applies @add_fn
                  scatter_dims_to_operand_dims=[0], index_vector_dim=1
        ; Accumulates gradient updates back to embedding table.
        ; @add_fn: base[idx] += update (atomics-free, purely functional)

    SEMANTICS NOTE on scatter ordering:
        If multiple updates target the same index, the order of accumulation
        is IMPLEMENTATION-DEFINED (StableHLO makes no guarantee).
        For a commutative+associative combiner (like add), the result is
        deterministic regardless of ordering. For non-associative ops, results
        may vary between compilers and hardware.


##### PART 7 — CONTROL FLOW: REGIONS, WHILE, IF, CASE

### MLIR Regions: The Building Block of Control Flow

    StableHLO control flow ops use MLIR's REGION mechanism — nested
    code blocks that are part of the operation.

    Every region contains at least one basic block (^bb0).
    Every basic block has block arguments (like φ-nodes in SSA).
    The last operation in a block is a terminator.
    StableHLO uses stablehlo.return as the region terminator.

    Example — a generic region structure in StableHLO:
        // An inline computation used by stablehlo.reduce
        ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
            %result = stablehlo.add %lhs, %rhs : tensor<f32>
            stablehlo.return %result : tensor<f32>

### stablehlo.while: Iterative Computation

    while executes a body region repeatedly while a condition region returns true.

    SIGNATURE:
        %results... = stablehlo.while(%init_values...)
                      cond = {
                        ^bb0(%state...:):
                          ...  ; compute condition
                          stablehlo.return %bool_scalar : tensor<i1>
                      }
                      body = {
                        ^bb0(%state...:):
                          ...  ; update state
                          stablehlo.return %new_state... : ...types...
                      }

    SEMANTICS:
        1. Start with init_values as the initial state.
        2. Evaluate cond region with current state → bool tensor<i1>.
        3. If false: return current state. Done.
        4. If true:  evaluate body region with current state → new state.
        5. Go to step 2 with new state.

    EXAMPLE — loop summing 1 to N:
        %zero = stablehlo.constant dense<0>   : tensor<i32>  ; accumulator
        %init_i = stablehlo.constant dense<0> : tensor<i32>  ; counter
        %N    = stablehlo.constant dense<10>  : tensor<i32>

        %final_sum, %final_i = stablehlo.while(%zero, %init_i)
          cond = {
            ^bb0(%acc: tensor<i32>, %i: tensor<i32>):
              %cond = stablehlo.compare %i, %N {direction=LT} : tensor<i1>
              stablehlo.return %cond : tensor<i1>
          }
          body = {
            ^bb0(%acc: tensor<i32>, %i: tensor<i32>):
              %one    = stablehlo.constant dense<1> : tensor<i32>
              %new_acc = stablehlo.add %acc, %i    : tensor<i32>
              %new_i   = stablehlo.add %i,   %one  : tensor<i32>
              stablehlo.return %new_acc, %new_i : tensor<i32>, tensor<i32>
          }
        ; final_sum = 0+1+2+...+9 = 45

    IMPORTANT CONSTRAINT: while loop bodies must be FUNCTIONAL.
        No mutable state outside the loop — state passes through arguments.
        Both cond and body regions must accept and return the same types.
        This enables the compiler to safely optimise loop iterations.

    ML USE CASES:
        Recurrent neural networks (RNN/LSTM unrolling).
        Beam search (iterative token generation).
        Iterative solvers (gradient descent inner loop in hyperparameter opt).
        Dynamic-length sequence processing.

### stablehlo.if: Conditional Computation

    if executes one of two regions based on a scalar boolean.

    SIGNATURE:
        %results... = stablehlo.if %condition
                      true_branch = {
                        ^bb0():    ; no block arguments for if branches
                          ...
                          stablehlo.return %result... : types
                      }
                      false_branch = {
                        ^bb0():
                          ...
                          stablehlo.return %result... : types
                      }

    SEMANTICS:
        if condition is true:  evaluate true_branch, return its results.
        if condition is false: evaluate false_branch, return its results.
        Both branches must return the same types.

    KEY DIFFERENCE from stablehlo.select:
        select: BOTH tensors are evaluated, then one is selected.
                This is SAFE because select is purely functional (no side effects).
                Use for: simple elementwise conditional (cheaper, vectorisable).
        if: ONLY the taken branch is evaluated.
            This is necessary for: side effects (send/recv), very large tensors
            where computing the unused branch wastes resources, or when branches
            have different shapes.

    EXAMPLE — conditional normalisation:
        %is_train = ...  ; tensor<i1> scalar boolean
        %output   = stablehlo.if %is_train
          true_branch = {
            %normed = call @batch_norm_train(%x, %gamma, %beta, %mean, %var)
            stablehlo.return %normed : tensor<NxCxHxWxf32>
          }
          false_branch = {
            %normed = call @batch_norm_infer(%x, %gamma, %beta, %running_mean, %running_var)
            stablehlo.return %normed : tensor<NxCxHxWxf32>
          }

### stablehlo.case: Multi-Way Conditional

    case is a switch-statement generalisation: select one of N branches
    based on an integer index.

    SIGNATURE:
        %results... = stablehlo.case %index
                      branches = [
                        {  ; branch 0
                          ^bb0():
                            stablehlo.return %result0... : types
                        },
                        {  ; branch 1
                          ^bb0():
                            stablehlo.return %result1... : types
                        },
                        ...
                      ]

    SEMANTICS:
        If index ∈ [0, N-1]: execute branches[index].
        If index < 0 or index ≥ N: execute branches[N-1] (last branch = default).

    ML USE CASES:
        Mixture-of-experts routing: select expert N based on routing decision.
        Multi-task models: select task head based on task ID.
        Configuration switching: select precision mode based on flag.

### stablehlo.call: Invoking Sub-Computations

    call invokes a named function defined elsewhere in the module.
    This is how StableHLO programs decompose into reusable sub-computations
    (attention head, FFN block, loss function, etc.).

    %results = stablehlo.call @attention_fn(%q, %k, %v, %mask)
               : (tensor<BxHxSxDxf32>, ...) → tensor<BxHxSxDxf32>

    Unlike reduce/while/if which take inline regions, call references a
    func.func defined at module scope.


##### PART 8 — DISTRIBUTED OPS: COMMUNICATION PRIMITIVES

### The SPMD Distributed Computing Model in StableHLO

    StableHLO programs can run on multiple devices simultaneously in an
    SPMD (Single Program Multiple Data) model:
        - Every device runs the SAME StableHLO program.
        - Each device has different data (a shard of the full tensor).
        - Communication ops synchronise and exchange data between devices.
        - The program is correct: each device sees logically complete results
          after communication ops complete.

    Device organisation:
        Devices are organised into a REPLICA GROUP (a set of devices that
        communicate with each other). Each op's replica_groups attribute
        specifies which devices participate.

        replica_groups = [[0, 1, 2, 3]]       ; all 4 devices form one group
        replica_groups = [[0, 1], [2, 3]]     ; two independent groups of 2

### all_reduce: Gradient Synchronisation

    all_reduce aggregates values across a group of devices, distributing
    the result to every device in the group.

    SIGNATURE:
        %result = stablehlo.all_reduce(%operand) applies @add_fn,
                  channel_handle = {handle=1, type=CROSS_REPLICA},
                  replica_groups = [[0, 1, 2, 3]]

    SEMANTICS:
        Each device d holds operand[d].
        result[d] = combine(operand[0], operand[1], ..., operand[D-1])
        for all d in the replica group.
        (All devices get the same combined result.)

    ML USE CASE — data-parallel gradient averaging:
        Each device computed gradients for its data shard.
        all_reduce(@add) accumulates all gradients.
        Divide by number of devices → global mean gradient.
        All devices update their parameters with the same global gradient.

    CHANNEL TYPES:
        CROSS_REPLICA:    communicate across replicas (data-parallel groups).
        CROSS_PARTITION:  communicate across model-parallel partitions.
        CROSS_REPLICA_AND_PARTITION: both.

### all_gather: Collecting Sharded Tensors

    all_gather collects shards from all devices into a larger tensor
    on every device (inverse of scatter).

    SEMANTICS:
        Device d holds shard of shape [S/D, N].
        After all_gather on dimension 0:
        Every device holds the FULL tensor of shape [S, N].

    ML USE CASE — model-parallel attention:
        Attention heads are sharded: device d holds heads [d*H/D, (d+1)*H/D].
        After attending to local heads, all_gather collects all head outputs.
        Every device now has the full [B, S, H, D] output.
        Subsequent layers can be applied without communication.

### reduce_scatter: ZeRO-Style Parameter Sharding

    reduce_scatter combines all_reduce and scatter in one operation:
    reduces across devices AND shards the result, so each device gets
    only its slice of the combined result.

    SEMANTICS:
        Device d holds operand[d] of shape [S, N].
        Step 1: add across devices → combined [S, N].
        Step 2: shard combined along scatter_dimension.
        Device d gets combined[d*S/D : (d+1)*S/D, :] of shape [S/D, N].

    ML USE CASE — ZeRO optimizer sharding:
        Gradient tensors are huge (billions of parameters).
        reduce_scatter averages gradients AND shards them.
        Each device only stores and optimises S/D parameters.
        Enables training models too large to fit on one GPU.

### all_to_all: Dimension Transposition Across Devices

    all_to_all is the most complex collective: it transposes the data/model
    parallelism dimensions, enabling 4D (batch×sequence×model×expert) parallelism.

    SEMANTICS:
        Device d has operand of shape [..., S, ...] where S is the split dim.
        All devices contribute their shard along split_dimension.
        Each device receives the part that was sharded along concat_dimension.
        Used to transpose which dimension is distributed.

    ML USE CASE — sequence parallelism in transformers:
        Before attention: batch is distributed, sequence is local.
        all_to_all swaps: sequence becomes distributed, batch is local.
        Now attention can be computed over different sequence positions per device.
        After attention: another all_to_all restores original distribution.

### send and recv: Point-to-Point Communication

    send and recv implement point-to-point messaging between specific devices.
    Unlike collectives (which involve all group members), send/recv are between
    a designated sender and receiver.

    %token1 = stablehlo.send %data, %token0,
              channel_handle = {handle=1, type=DEVICE_TO_DEVICE}
              : (tensor<4xf32>, !stablehlo.token) → !stablehlo.token

    %data, %token2 = stablehlo.recv %token1,
                     channel_handle = {handle=1, type=DEVICE_TO_DEVICE}
                     : (!stablehlo.token) → (tensor<4xf32>, !stablehlo.token)

    The token threading enforces ordering: the recv cannot execute before
    the corresponding send has sent the data.
    handle values match sends to their corresponding receives.


##### PART 9 — THE STABLEHLO SPECIFICATION AND REFERENCE INTERPRETER

### The Specification Document

    Every StableHLO operation is documented in the specification:
        github.com/openxla/stablehlo/blob/main/docs/spec.md

    For each operation, the spec provides:
        INPUTS:       every argument with its type constraints
        OUTPUTS:      result types
        CONSTRAINTS:  invariants that must hold for the op to be valid
                      (e.g., "lhs and rhs must have the same element type")
        SEMANTICS:    the mathematical definition of what is computed
                      (e.g., "result[i0,...] = operand[i0,...] + rhs[i0,...]")
        EXAMPLES:     concrete input/output pairs verifying the semantics

    This specification-first approach means:
        Ambiguities have authoritative resolutions.
        A new compiler can validate against the spec, not against XLA's behaviour.
        Test suites can be auto-generated from examples in the spec.

### The Reference Interpreter

    StableHLO ships a REFERENCE INTERPRETER (in Python):
        from stablehlo import reference_interpreter as interp

        result = interp.interpret(mlir_module, [input_tensor_1, input_tensor_2])

    The reference interpreter is the GROUND TRUTH for op semantics.
    It is NOT optimised for performance — it is correct by construction.
    Compilers (XLA, IREE, TVM) can validate their outputs against the
    reference interpreter on any StableHLO program.

    Testing workflow:
        Generate random inputs → run reference interpreter → record outputs.
        Compile with XLA → run on GPU → compare outputs.
        Any discrepancy = compiler bug.

    The reference interpreter also serves as:
        Documentation for spec writers (the code IS the spec).
        Correctness oracle for fuzzing campaigns.
        Debugging tool for compiler developers.

### Verification and the MLIR Verifier

    StableHLO leverages MLIR's built-in verification infrastructure.
    Every StableHLO op registers a verifier that checks constraints at parse time.

    What the verifier checks:
        Type constraints (e.g., compare requires same types for lhs and rhs).
        Shape constraints (e.g., transpose requires perm to be a permutation).
        Attribute constraints (e.g., dot_general contracting dims must be valid).
        Region type constraints (e.g., while body must match cond argument types).

    Verification is automatic:
        mlir-opt --verify-diagnostics input.mlir
        Illegal programs are caught before compilation.

    Example of a verification error:
        %r = stablehlo.add %a, %b : tensor<4xf32>
        ; where %a : tensor<4xf32> and %b : tensor<8xf32>
        ; VERIFIER ERROR: "operand element types must match" or shape mismatch
        ; (caught at parse time, not at runtime)


##### PART 10 — VERSIONING, COMPATIBILITY, AND SERIALISATION

### The Compatibility Policy in Detail

    StableHLO versions use SEMANTIC VERSIONING (major.minor.patch):
        Major:  reserved for breaking changes (currently at 1.x.x)
        Minor:  backward-compatible additions (new ops, new attributes)
        Patch:  bug fixes, clarifications, documentation

    FORWARD COMPATIBILITY (producer → older consumer):
        A program produced by version A must be usable by consumers
        released at version B where A was released within 6 months of B.

    BACKWARD COMPATIBILITY (consumer → older programs):
        A consumer at version B must correctly handle any program produced
        by any version A where A was released within 5 years before B.

    PRACTICAL CONSEQUENCE:
        Serialise a model in 2024 → guaranteed to work on any StableHLO
        consumer released until 2029. This is the "five-year window."

    HOW BACKWARD COMPAT IS MAINTAINED:
        A "StableHLO version upgrader" pass transforms old programs:
            Deprecated op X (version 0.9) → replacement op Y (version 1.0)
            Changed attribute encoding → updated encoding
        These upgrade passes are auto-generated from the compatibility spec
        and run transparently when loading older programs.

### Serialisation Formats

    StableHLO programs can be stored in three formats:

    FORMAT 1: Textual MLIR (.mlir files)
        Human-readable SSA text.
        Useful for debugging, diffing, code review.
        Not the primary serialisation format (verbose, slow to parse).
        Parse: mlir::parseSourceFile("model.mlir", context)
        Print: module->print(llvm::outs())

    FORMAT 2: Binary MLIR bytecode (.mlirbc files)
        Compact binary encoding of the MLIR module.
        10–100× smaller than text for large models.
        Includes version number in a header.
        Parse: mlir::parseSourceFile("model.mlirbc", context)
               (same API — MLIR detects binary vs text automatically)
        Produce: mlir::writeBytecodeToFile(*module, output_file)

    FORMAT 3: JAX export blob (via jax.export)
        Wraps the binary bytecode with additional metadata:
            - StableHLO version tag
            - Function signature (names, abstract values)
            - Sharding annotations
            - Serialised as bytes via FlatBuffers
        Produced: exported.serialize() → bytes
        Consumed: jax.export.deserialize(bytes)

    THE VERSION TAG:
        Binary bytecode embeds: producer_version, minimum_consumer_version.
        Consumer checks: can_read(program_version) before attempting to load.
        If version too old: apply upgrade passes, then continue.
        If version too new: error (consumer predates the producer).

### Passes Operating on StableHLO

    Since StableHLO is an MLIR dialect, the MLIR pass manager handles it.
    Key passes in the openxla/stablehlo repository:

    CANONICALIZATION (--stablehlo-canonicalize):
        Applies constant folding and algebraic simplification.
        Same identities as XLA's algebraic simplifier but at StableHLO level:
            add(x, 0) → x
            mul(x, 1) → x
            transpose(transpose(x, p), q) → transpose(x, compose(p,q))
            reshape(reshape(x,s1),s2) → reshape(x,s2)

    SHAPE INFERENCE (--stablehlo-infer-shapes):
        Propagates static shape information through the program.
        If a reduce input has shape [4,8] and we reduce dim 1 → output is [4].
        This converts tensor<?xf32> → tensor<4xf32> where the shape is known.

    CONVERT_TO_LINALG (--stablehlo-legalize-to-linalg):
        Lowers StableHLO ops to the MLIR linalg dialect.
        The bridge between StableHLO portability layer and IREE/MLIR backends.
        dot_general → linalg.matmul
        elementwise → linalg.generic
        reduce → linalg.reduce (partial)

    LEGALIZE_TO_HLO (--stablehlo-legalize-to-hlo):
        Converts StableHLO → XLA's internal MHLO dialect.
        Used inside XLA's compilation pipeline:
            StableHLO → MHLO → HLO protos → XLA codegen

    BATCH_NORM_EXPANDER (--stablehlo-expand-ops):
        Expands complex ops (batch_norm_training) into simpler primitives.
        batch_norm_training → reduce + subtract + multiply + etc.
        Helps compilers that don't natively support batch norm.

    VERSION_UPGRADER (--stablehlo-upgrade-version):
        Upgrades programs from old StableHLO versions to current.
        Applied automatically by consumers when loading old programs.

    INTERPRETER (for testing):
        --stablehlo-interpret: run the program through the reference interpreter.
        Used in lit tests to verify semantics:
            // CHECK: result = [1.0, 2.0, 3.0]
            // RUN: stablehlo-interpreter input.mlir | FileCheck input.mlir


##### PART 11 — PRODUCERS AND CONSUMERS: THE ECOSYSTEM MAP

### StableHLO Producers (frameworks that emit StableHLO)

    JAX (primary producer):
        jax.export.export(jax.jit(fn))(*abstract_args)
        Every jax.jit-compiled function can be exported as StableHLO.
        Supports: polymorphic shapes, multi-device SPMD sharding,
                  forward-mode and reverse-mode AD gradients.

    TensorFlow:
        tf.saved_model with jit_compile=True exports XLA computations
        which are lowered to StableHLO via the TF-to-MHLO-to-StableHLO path.
        Google's production TF models use this path for deployment.

    torch-mlir (PyTorch → StableHLO):
        torch_mlir.compile(model, inputs, output_type=OutputType.STABLEHLO)
        Lowers PyTorch's ATen operations to StableHLO via:
            PyTorch FX → Torch dialect → StableHLO dialect
        Covers: all major transformer operations (linear, attention, norm, act).

    ONNX:
        onnx-mlir with StableHLO output (experimental converter).
        Bridges the large ONNX ecosystem to the OpenXLA stack.

### StableHLO Consumers (compilers that ingest StableHLO)

    OpenXLA/XLA (primary consumer):
        StableHLO → MHLO (via --stablehlo-legalize-to-hlo)
               → XLA HLO protos (via MlirToHloTranslate)
               → XLA compilation pipeline
               → GPU cubin / TPU executable / CPU binary

    IREE:
        StableHLO → linalg (via --stablehlo-legalize-to-linalg)
               → IREE Flow/Stream/HAL dialects
               → SPIR-V / LLVM IR / CUDA
               → Vulkan / CPU / Android / Web

    TVM:
        StableHLO → TVM Relax IR (via relax.from_stablehlo)
               → TVM auto-tuning → TVM codegen
               → Compiled TVM module for CPU/GPU/edge

    OpenVINO:
        StableHLO → OpenVINO IR (via conversion pass)
               → OpenVINO inference engine
               → Intel CPU/GPU/VPU/FPGA execution

    Triton (research):
        StableHLO → Triton IR (via triton-linalg bridge)
               → Triton GPU kernel generation

### The Complete Ecosystem Diagram

    ┌─────────────────────────────────────────────────────────────────────┐
    │  PRODUCERS                                                          │
    │  JAX         torch-mlir    TensorFlow    ONNX (experimental)        │
    │    ↓              ↓              ↓              ↓                   │
    │  jax.export   TorchScript→  tf.saved_model  onnx-mlir               │
    │               STABLEHLO     +jit_compile                            │
    └──────────────────────────┬──────────────────────────────────────────┘
                               │
                   StableHLO (.mlir / .mlirbc / jax blob)
                   Stable, versioned, MLIR-based
                               │
    ┌──────────────────────────┴──────────────────────────────────────────┐
    │  CONSUMERS                                                          │
    │  OpenXLA/XLA              IREE              TVM         OpenVINO    │
    │  (GPU/TPU training)  (mobile/edge/web) (custom HW)   (Intel HW)     │
    │         ↓                   ↓               ↓              ↓        │
    │  MHLO→HLO protos     linalg→Flow→HAL  Relax IR      OpenVINO IR     │
    │         ↓                   ↓               ↓              ↓        │
    │  cubin/TPU blob      vmfb artifact    TVM module    OpenVINO blob   │
    └─────────────────────────────────────────────────────────────────────┘

    TOOLING:
        mlir-opt:           apply passes, inspect IR
        stablehlo-opt:      StableHLO-specific passes
        stablehlo-interpreter: reference execution for testing
        mlir-lsp-server:    IDE language server (hover, goto-def in .mlir files)
        filecheck + lit:    MLIR-native testing infrastructure

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · StableHLO IR — Reading Every Syntactic Element": {
        "description": (
            "Complete guide to reading and writing StableHLO IR text. "
            "Break down every syntactic element: operations, types, attributes, regions. "
            "Trace a full transformer attention block from JAX to StableHLO op by op. "
            "Show the module structure: func.func, block arguments, terminators. "
            "Demonstrate constant and iota creation ops. "
            "Annotate every line of a softmax in StableHLO explaining what each op does."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  STABLEHLO IR — READING EVERY SYNTACTIC ELEMENT")
print("=" * 65)
print()

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
    print(f"  JAX {jax.__version__} | backend: {jax.default_backend()}")
except ImportError:
    HAS_JAX = False
    print("  JAX not installed: pip install jax")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Module and function structure
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Module / function / block / op structure")
print("━" * 65)
print()

STRUCTURE_GUIDE = """
  STABLEHLO PROGRAM STRUCTURE — EVERY ELEMENT EXPLAINED
  ════════════════════════════════════════════════════════════════

  module @module_name {
  │  ↑ MLIR module = the top-level container
  │  ↑ @module_name = optional symbolic name (used in stablehlo.call)
  │
  │  func.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<2x4xf32>)
  │  │         │      │     │                     │
  │  │         │      │     └── SSA value names   └── MLIR types
  │  │         │      └── function name (must match callers)
  │  │         └── visibility: public/private/nested
  │  └── func.func = MLIR standard dialect function op
  │
  │              -> (tensor<4xf32>, tensor<4xi32>) {
  │                  ↑ return types (tuple of tensors)
  │
  │    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<2x4xf32>):
  │    │     ↑ SSA values bound at block entry (block arguments)
  │    └── label for basic block (^bb0, ^bb1, ...)
  │
  │      %0 = stablehlo.constant dense<0.0> : tensor<f32>
  │      │    │                   │           │
  │      │    │                   │           └── type of the result
  │      │    │                   └── attribute: the value
  │      │    └── operation name: dialect.op_name
  │      └── SSA result name (must be used exactly once or discarded)
  │
  │      %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
  │           │                          │   │            │               │
  │           │                          │   │            └── input type  └── output type
  │           │                          │   └── attribute: which dims to map
  │           │                          └── operand (refers to %0 above)
  │           └── operation with attribute
  │
  │      %2 = stablehlo.maximum %arg0, %1 : tensor<4xf32>
  │           │                 │      │    └── type applies to ALL operands and result
  │           │                 │      └── second operand
  │           │                 └── first operand
  │           └── elementwise op: result shape = max(arg0.shape, 1.shape) = [4]
  │
  │      return %2, %3 : tensor<4xf32>, tensor<4xi32>
  │      └── terminator: returns multiple values from the function
  │
  }

  KEY RULES:
  1. SSA form: every %value is defined exactly once, used zero or more times.
  2. Types appear after the colon on each operation.
  3. Block arguments are the function inputs (^bb0 of the entry block).
  4. Operations in a block are executed top to bottom.
  5. The entry block always has label ^bb0 and takes the function args.
  6. Non-entry blocks have explicit arguments used for phi-node-like merges.
  7. Attributes ({} or keyword=value) are compile-time constants.
  8. Operands (unnamed %values) are runtime SSA values.
"""
print(STRUCTURE_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Full annotated softmax in StableHLO
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Fully annotated softmax in StableHLO")
print("━" * 65)
print()

SOFTMAX_ANNOTATED = """
  jax.nn.softmax(x, axis=-1) for x: tensor<4x8xf32>
  ════════════════════════════════════════════════════════════════

  func.func @softmax(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {

    // ── 1. Define the initial value for the max reduction ────────────────
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    //         │                   │                  │
    //         │                   │                  └── scalar float (rank-0 tensor)
    //         │                   └── hex encoding of -infinity in IEEE 754 f32
    //         └── result: a rank-0 tensor containing -infinity

    // ── 2. Find the maximum of each row (reduce along axis=1) ────────────
    %row_max = stablehlo.reduce(%arg0, %neg_inf)
               applies @max_computation across dimensions = [1]
               : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    //         │  ↑                 ↑              │
    //         │  operand (4x8)     init value     └── output: [4] (dim 1 collapsed)
    //         └── MULTI-OP: takes both the operand AND the init value
    //
    // reduce semantics: result[i] = fold(neg_inf, arg0[i,0], arg0[i,1], ..., arg0[i,7])
    // where fold uses @max_computation: max(a, b)
    // This gives the max of each of the 4 rows.

    // ── 3. Broadcast row_max back to [4, 8] for subtraction ──────────────
    %max_bc = stablehlo.broadcast_in_dim %row_max, dims = [0]
              : (tensor<4xf32>) -> tensor<4x8xf32>
    //                             ↑                ↑
    //         input [4] dim 0     maps to          output dim 0
    //         dim 1 of output ([8]) is the new broadcast dimension
    //
    // Result: each of the 4 max values is replicated 8 times along dim 1.
    // max_bc[i, j] = row_max[i]  for all j in [0, 8)

    // ── 4. Numerically stable shift: x_shifted = x - max(x, axis=-1) ────
    %x_shifted = stablehlo.subtract %arg0, %max_bc : tensor<4x8xf32>
    //            │                  │      │
    //            │                  │      └── max_bc, broadcast-expanded
    //            │                  └── original input
    //            └── elementwise subtraction (shapes must match exactly after broadcast)
    //
    // x_shifted[i,j] = arg0[i,j] - row_max[i]
    // All values ≤ 0; the maximum in each row becomes exactly 0.
    // exp(x_shifted) will now be in (0, 1] → no overflow.

    // ── 5. Exponentiate each element ─────────────────────────────────────
    %x_exp = stablehlo.exponential %x_shifted : tensor<4x8xf32>
    //       └── unary op: result[i,j] = e^(x_shifted[i,j])

    // ── 6. Sum each row (reduce along axis=1) ────────────────────────────
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %row_sum = stablehlo.reduce(%x_exp, %zero)
               applies @add_computation across dimensions = [1]
               : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    // result[i] = sum over j of x_exp[i,j]  (the partition function)

    // ── 7. Broadcast row sums back to [4, 8] ─────────────────────────────
    %sum_bc = stablehlo.broadcast_in_dim %row_sum, dims = [0]
              : (tensor<4xf32>) -> tensor<4x8xf32>

    // ── 8. Divide: normalise each row ────────────────────────────────────
    %result = stablehlo.divide %x_exp, %sum_bc : tensor<4x8xf32>
    //         └── elementwise: result[i,j] = x_exp[i,j] / sum_bc[i,j]
    //             = e^(x[i,j] - max_i) / sum_j e^(x[i,j] - max_i)
    //             = e^x[i,j] / sum_j e^x[i,j]  (the max cancels)
    //             = softmax(x)[i,j]  ✓

    return %result : tensor<4x8xf32>
  }

  // ── Sub-computations referenced by reduce ────────────────────────────
  func.func private @max_computation(%arg0: tensor<f32>, %arg1: tensor<f32>)
                                      -> tensor<f32> {
    %r = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %r : tensor<f32>
  }

  func.func private @add_computation(%arg0: tensor<f32>, %arg1: tensor<f32>)
                                      -> tensor<f32> {
    %r = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %r : tensor<f32>
  }

  INSTRUCTION COUNT: 8 ops (constant×2, reduce×2, broadcast_in_dim×2,
                             subtract×1, exponential×1, divide×1)
  FUSED BY XLA:      exp+subtract → one kernel
                     reduce+divide → one kernel (input fusion)
  FINAL KERNEL COUNT (GPU): ~2-3 kernels for the whole softmax
"""
print(SOFTMAX_ANNOTATED)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Extract StableHLO from JAX
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Extracting StableHLO from JAX computations")
print("━" * 65)
print()

if HAS_JAX:
    def softmax(x):
        x_max = jnp.max(x, axis=-1, keepdims=True)
        x_exp = jnp.exp(x - x_max)
        return x_exp / jnp.sum(x_exp, axis=-1, keepdims=True)

    def gelu(x):
        """GeLU activation: x * Φ(x) where Φ is the normal CDF."""
        return x * 0.5 * (1.0 + jnp.tanh(
            0.7978845608 * (x + 0.044715 * x**3)))

    def layer_norm(x, gamma, beta, eps=1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var  = jnp.var(x,  axis=-1, keepdims=True)
        x_n  = (x - mean) / jnp.sqrt(var + eps)
        return gamma * x_n + beta

    for fn_name, fn, args in [
        ("softmax",    softmax,    (jnp.ones((4, 8), jnp.float32),)),
        ("gelu",       gelu,       (jnp.ones((8,), jnp.float32),)),
        ("layer_norm", layer_norm, (jnp.ones((4, 8), jnp.float32),
                                    jnp.ones((8,), jnp.float32),
                                    jnp.zeros((8,), jnp.float32))),
    ]:
        abs_args = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args)
        try:
            exp = jax.export.export(jax.jit(fn))(*abs_args)
            mlir = exp.mlir_module()
            ops  = [l.strip() for l in mlir.split("\\n")
                    if "stablehlo." in l and "=" in l]
            print(f"  {fn_name}: {len(ops)} stablehlo ops")
            for op in ops[:8]:
                clean = op.split("=")[1].strip().split(":")[0].strip()
                print(f"    {clean[:70]}")
            if len(ops) > 8:
                print(f"    ... ({len(ops)-8} more)")
        except Exception:
            comp = jax.xla_computation(fn)(*args)
            hlo_lines = [l.strip() for l in comp.as_hlo_text().split("\\n")
                         if "=" in l and "parameter" not in l
                         and "ENTRY" not in l and "HloModule" not in l
                         and "{" not in l and "}" not in l and l.strip()]
            print(f"  {fn_name}: {len(hlo_lines)} HLO ops (StableHLO needs jax>=0.4.14)")
        print()
else:
    EXTRACT_REF = """
  JAX → StableHLO extraction APIs:
  ─────────────────────────────────────────────────────────────────
  # Modern API (jax >= 0.4.14):
  abs_args = jax.ShapeDtypeStruct((4,8), jnp.float32)
  exported = jax.export.export(jax.jit(softmax))(abs_args)
  print(exported.mlir_module())   # prints full StableHLO text

  # Legacy API (all JAX versions):
  comp = jax.xla_computation(softmax)(jnp.ones((4,8)))
  print(comp.as_hlo_text())       # XLA HLO (not StableHLO, but similar)

  # Inspect output info:
  print(exported.in_avals)        # input abstract shapes/types
  print(exported.out_avals)       # output abstract shapes/types
  print(exported.fun_name)        # function name
"""
    print(EXTRACT_REF)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Broadcasting Semantics — Rules, Examples, and Pitfalls": {
        "description": (
            "Complete exploration of StableHLO's two broadcasting mechanisms. "
            "Implement numpy-style implicit broadcasting rules from first principles. "
            "Work through every broadcast_in_dim case with visual shape diagrams. "
            "Show which ops use implicit vs explicit broadcasting. "
            "Demonstrate the monotonicity constraint and how to work around it. "
            "Show common pitfalls: scalar broadcast, rank mismatch, the size-1 rule."
        ),
        "language": "python",
        "code": '''
import numpy as np
from typing import List, Tuple, Optional

print("=" * 65)
print("  BROADCASTING SEMANTICS — RULES, EXAMPLES, AND PITFALLS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Implicit broadcasting rules (numpy-style)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Implicit broadcasting: the numpy rules")
print("━" * 65)
print()

def broadcast_shape(shape_a: tuple, shape_b: tuple) -> Optional[tuple]:
    """
    Compute the output shape for StableHLO implicit broadcasting.
    Returns None if the shapes are incompatible.
    Implements the numpy/StableHLO rule: right-align, stretch size-1 dims.
    """
    ndim = max(len(shape_a), len(shape_b))
    # Pad to same rank with leading 1s (right-align)
    a = (1,) * (ndim - len(shape_a)) + tuple(shape_a)
    b = (1,) * (ndim - len(shape_b)) + tuple(shape_b)
    result = []
    for da, db in zip(a, b):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            return None   # incompatible sizes, neither is 1
    return tuple(result)

def explain_broadcast(shape_a, shape_b):
    """Print a detailed explanation of how two shapes broadcast."""
    result = broadcast_shape(shape_a, shape_b)
    ndim = max(len(shape_a), len(shape_b))
    a = (1,) * (ndim - len(shape_a)) + tuple(shape_a)
    b = (1,) * (ndim - len(shape_b)) + tuple(shape_b)
    print(f"    A: {str(shape_a):15s}  right-aligned: {a}")
    print(f"    B: {str(shape_b):15s}  right-aligned: {b}")
    if result is None:
        print(f"    Result: INCOMPATIBLE ❌")
        # Find first mismatch
        for i, (da, db) in enumerate(zip(a, b)):
            if da != db and da != 1 and db != 1:
                print(f"    Conflict at dim {i}: A={da}, B={db} "
                      f"(neither is 1, neither is equal)")
    else:
        arrows = []
        for da, db, dr in zip(a, b, result):
            if da == db:          arrows.append(f"{dr}=={dr}")
            elif da == 1:         arrows.append(f"1→{dr}")
            else:                 arrows.append(f"{dr}←1")
        print(f"    Per-dim:  {arrows}")
        print(f"    Result:  {result} ✅")
    print()

test_cases = [
    # (shape_a, shape_b, description)
    ((4,),        (4,),        "Exact match: [4] + [4]"),
    ((4,),        (1,),        "Size-1 broadcasts: [4] + [1]"),
    ((),          (4,),        "Scalar + vector: [] + [4]"),
    ((3,4),       (4,),        "Matrix + row vector: [3,4] + [4]"),
    ((3,4),       (3,1),       "Matrix + column vector: [3,4] + [3,1]"),
    ((3,1),       (1,4),       "Both broadcast: [3,1] + [1,4]"),
    ((2,3,4),     (3,4),       "3D + 2D: [2,3,4] + [3,4]"),
    ((1,1,4),     (3,1,1),     "Broadcast 3 dims: [1,1,4] + [3,1,1]"),
    ((3,),        (4,),        "INCOMPATIBLE: [3] + [4]"),
    ((2,3),       (3,2),       "INCOMPATIBLE: [2,3] + [3,2]"),
]

print("  Implicit broadcast shape inference:")
print()
for shape_a, shape_b, desc in test_cases:
    print(f"  ── {desc}")
    explain_broadcast(shape_a, shape_b)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: broadcast_in_dim — explicit axis mapping
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — broadcast_in_dim: explicit axis mapping")
print("━" * 65)
print()

BROADCAST_IN_DIM = """
  broadcast_in_dim — the EXPLICIT broadcast op
  ════════════════════════════════════════════════════════════════

  Syntax:
    %out = stablehlo.broadcast_in_dim %in, dims = [d0, d1, ...]
           : (input_type) → output_type

  The dims attribute: dims[i] = j means "input dim i maps to output dim j".
  Output dims NOT in dims are NEW broadcast dimensions.

  VISUALISED EXAMPLES:
  ─────────────────────────────────────────────────────────────────

  Case 1: scalar → [4,8]  (dims=[])
    Input:  tensor<f32>       shape: ()        value: 3.14
    dims:   []                (no input dims to map)
    Output: tensor<4x8xf32>  shape: (4,8)
    Effect: fill 4×8=32 elements all with 3.14

    Input axes:  (none)
    Output axes: [4] [8]   ← both are broadcast axes

  Case 2: bias [D] → batch [B,D]  (dims=[1])
    Input:  tensor<4xf32>      shape: (4,)     bias values
    dims:   [1]                input dim 0 maps to output dim 1
    Output: tensor<8x4xf32>   shape: (8,4)
    Effect: each bias value replicated 8 times (one per batch item)

    Input axes:  [4]
    Output axes: [8] [4]
                  ↑   ↑
                  new  mapped from input dim 0

  Case 3: column [M,1] → matrix [M,N]  (dims=[0,1])
    Input:  tensor<3x1xf32>   shape: (3,1)    column vector
    dims:   [0,1]              input dim 0→out dim 0, input dim 1→out dim 1
    Output: tensor<3x4xf32>   shape: (3,4)
    Effect: size-1 dim 1 is stretched from 1 to 4 (broadcast along dim 1)

    Input axes:  [3] [1]
    Output axes: [3] [4]
                  ↑   ↑
                  same stretch (1→4)

  Case 4: attention mask [S] → [B,H,S,S]  (dims=[2])
    Input:  tensor<Sxf32>          shape: (S,)    per-position values
    dims:   [2]                    maps to dim 2 of output
    Output: tensor<BxHxSxSxf32>   shape: (B,H,S,S)
    Effect: same position values for every batch item and head

    Input axes:  [S]
    Output axes: [B] [H] [S] [S]
                              ↑
                  mapped here (dims=[2] = output dim 2)

  MONOTONICITY CONSTRAINT:
    dims must be strictly increasing (non-decreasing after StableHLO 0.x, strict in 1.0).
    dims=[0,1] ✅    dims=[1,0] ❌  (not increasing)
    dims=[0,2] ✅    dims=[2,0] ❌

    To broadcast with a permutation, use stablehlo.transpose FIRST:
      %t = stablehlo.transpose %x, dims=[1,0] : (tensor<3x4>) → tensor<4x3>
      %b = stablehlo.broadcast_in_dim %t, dims=[0,1] : (tensor<4x3>) → tensor<5x4x3>

  SIZE-1 RULE:
    If input dim i has size 1 AND output dim dims[i] has size > 1:
      → stretch (replicate values along that dimension)
    If input dim i has size S AND output dim dims[i] also has size S:
      → pass through (values copied without replication)
    If input dim i has size S AND output dim dims[i] has size T, S != T, S != 1:
      → ILLEGAL (verification error)
"""
print(BROADCAST_IN_DIM)

def simulate_broadcast_in_dim(data: np.ndarray,
                               dims: List[int],
                               output_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Simulate stablehlo.broadcast_in_dim in numpy.
    dims[i] = j means input axis i maps to output axis j.
    """
    ndim_out = len(output_shape)
    # Build the target shape for numpy broadcasting
    # Start with a shape of all 1s, then fill in the mapped dims
    intermed_shape = [1] * ndim_out
    for i, d in enumerate(dims):
        assert data.shape[i] == output_shape[d] or data.shape[i] == 1, \
            f"Dim mismatch: input[{i}]={data.shape[i]} vs output[{d}]={output_shape[d]}"
        intermed_shape[d] = data.shape[i]
    # Reshape to intermed_shape, then broadcast to output_shape
    reshaped = data.reshape(intermed_shape)
    return np.broadcast_to(reshaped, output_shape).copy()

# Demonstrate each case
print("  Simulation of broadcast_in_dim cases:")
print()

# Case 1: scalar → [3,4]
scalar = np.array(3.14, dtype=np.float32)
out1 = simulate_broadcast_in_dim(scalar, dims=[], output_shape=(3,4))
print(f"  Case 1: scalar → [3,4]  (dims=[])")
print(f"    Input shape:  {scalar.shape}")
print(f"    Output shape: {out1.shape}")
print(f"    Output[0,:]:  {out1[0,:]}")
print()

# Case 2: bias [4] → [8,4]
bias = np.arange(4, dtype=np.float32)
out2 = simulate_broadcast_in_dim(bias, dims=[1], output_shape=(8,4))
print(f"  Case 2: bias [4] → [8,4]  (dims=[1])")
print(f"    Input shape:  {bias.shape}  values: {bias}")
print(f"    Output shape: {out2.shape}")
print(f"    Output[0,:]:  {out2[0,:]}  (same as input)")
print(f"    Output[3,:]:  {out2[3,:]}  (same! replicated across dim 0)")
print()

# Case 3: column [3,1] → [3,4]
col = np.array([[10.],[20.],[30.]], dtype=np.float32)
out3 = simulate_broadcast_in_dim(col, dims=[0,1], output_shape=(3,4))
print(f"  Case 3: column [3,1] → [3,4]  (dims=[0,1])")
print(f"    Input:  {col.ravel()}")
print(f"    Output:\\n{out3}")
print()

# Case 4: position [S] → [B,H,S,S] — attention mask broadcast
S = 4; B = 2; H = 3
pos = np.arange(S, dtype=np.float32)
out4 = simulate_broadcast_in_dim(pos, dims=[2], output_shape=(B,H,S,S))
print(f"  Case 4: position [{S}] → [{B},{H},{S},{S}]  (dims=[2])")
print(f"    Input:  {pos}")
print(f"    Output shape: {out4.shape}")
print(f"    Output[0,0,:,:] =\\n{out4[0,0,:,:]}")
print(f"    (same for ALL batch and head indices)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Common pitfalls
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Common pitfalls in broadcasting")
print("━" * 65)
print()

PITFALLS = """
  PITFALL 1: Confusing implicit and explicit broadcast
  ─────────────────────────────────────────────────────
  Implicit (elementwise ops): shapes are RIGHT-ALIGNED automatically.
    stablehlo.add %x, %bias where x:[3,4] and bias:[4] → works (right-aligned)
    stablehlo.add %x, %col  where x:[3,4] and col:[3]  → FAILS
        (right-aligned: 3,4 vs ,3 → col becomes [1,3], not [3,1])
        Fix: stablehlo.broadcast_in_dim %col, dims=[0]→[3,4]
             then stablehlo.add

  Explicit (broadcast_in_dim): axis mapping is LEFT-TO-RIGHT of input.
    broadcast_in_dim %bias (dims=[1]) → places bias at dim 1 of output.
    This is NOT the same as numpy broadcasting of [4] → [3,4]:
      numpy just right-aligns; stablehlo.broadcast_in_dim uses explicit dims.

  PITFALL 2: Forgetting the monotonicity constraint
  ─────────────────────────────────────────────────────
  Want to broadcast [A,B] to [C,B,A]:
    WRONG:  broadcast_in_dim %x, dims=[2,1]  → ERROR (not increasing)
    CORRECT: transpose first: %t = transpose %x, dims=[1,0]  → [B,A]
             then broadcast:  %r = broadcast_in_dim %t, dims=[1,2] → [C,B,A]

  PITFALL 3: select vs broadcast for scalar fill
  ─────────────────────────────────────────────────────
  Fill tensor with a scalar conditional value:
    WRONG: stablehlo.select %cond_scalar, %val_a_tensor, %val_b_tensor
           → only works if %cond_scalar is the SAME SHAPE as the tensors.
             A scalar i1 does NOT implicitly broadcast in select.
    CORRECT: broadcast the scalar condition first:
             %cond_bc = broadcast_in_dim %cond_scalar, dims=[]
                        : tensor<i1> → tensor<4x8xi1>
             %result  = stablehlo.select %cond_bc, %val_a, %val_b

  PITFALL 4: broadcast_in_dim with size > 1 inputs
  ─────────────────────────────────────────────────────
  Trying to "squeeze" dims via broadcast_in_dim:
    WRONG: broadcast_in_dim %x (3,4), dims=[0,1], output=(3,2)
           → ERROR: input dim 1 has size 4 but output dim 1 has size 2 (not 1, not 4)
    broadcast_in_dim can only GROW dimensions (size 1 → larger), never SHRINK.
    For shrinking: use stablehlo.slice, stablehlo.reduce, or stablehlo.reshape.
"""
print(PITFALLS)

# Demonstrate pitfall 1
print("  Demonstrating Pitfall 1 — column vector broadcasting:")
print()
x   = np.random.randn(3, 4).astype(np.float32)
col = np.arange(3, dtype=np.float32).reshape(3, 1)   # column [3,1]
row = np.arange(4, dtype=np.float32).reshape(1, 4)   # row    [1,4]

# Correct way: column needs explicit dims=[0] as broadcast_in_dim
col_bc = simulate_broadcast_in_dim(col.ravel(), dims=[0], output_shape=(3,4))
result = x + col_bc

print(f"  x shape: {x.shape}, col shape: {col.ravel().shape}")
print(f"  To add col [3] to x [3,4]:")
print(f"  broadcast_in_dim(col, dims=[0]) → shape {col_bc.shape}")
print(f"  (x + col_bc)[0,:] = x[0,:] + {col.ravel()[0]:.1f}")
print(f"  actual: {result[0,:3]}...")
print(f"  numpy reference: {(x + col)[0,:3]}...")
print(f"  Match: {np.allclose(result, x + col)} ✅")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · dot_general and reduce — Core Compute Op Semantics": {
        "description": (
            "Complete exploration of dot_general: every case from dot product to multi-head attention. "
            "Implement dot_general semantics in numpy and verify against JAX. "
            "Show every reduce pattern: sum, max, argmax, multi-output reduce. "
            "Trace a full attention layer: QKV projection → scores → softmax → output. "
            "Demonstrate reduce_window for pooling. "
            "Show precision_config and how it affects accumulation on TPU/GPU."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  DOT_GENERAL AND REDUCE — CORE COMPUTE OP SEMANTICS")
print("=" * 65)
print()

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: dot_general — implementing all cases from scratch
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — dot_general: implementing all cases in numpy")
print("━" * 65)
print()

def dot_general(lhs: np.ndarray, rhs: np.ndarray,
                lhs_batching: list, rhs_batching: list,
                lhs_contracting: list, rhs_contracting: list) -> np.ndarray:
    """
    Implement stablehlo.dot_general semantics in numpy.

    This is the reference implementation matching StableHLO's spec:
        result[b..., i..., j...] = sum_{k...} lhs[b..., i..., k...] * rhs[b..., k..., j...]
    where b = batching, i = lhs free, j = rhs free, k = contracting dims.
    """
    # Compute free dimensions (not batching, not contracting)
    all_lhs = set(range(lhs.ndim))
    all_rhs = set(range(rhs.ndim))
    lhs_free = sorted(all_lhs - set(lhs_batching) - set(lhs_contracting))
    rhs_free = sorted(all_rhs - set(rhs_batching) - set(rhs_contracting))

    # Build einsum string: unique letters per dimension role
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    idx = 0
    lhs_str = [''] * lhs.ndim
    rhs_str = [''] * rhs.ndim
    out_str = []

    # Batch dims → same letter for lhs and rhs → appear in output
    for lb, rb in zip(lhs_batching, rhs_batching):
        c = alphabet[idx]; idx += 1
        lhs_str[lb] = c; rhs_str[rb] = c; out_str.append(c)

    # Lhs free dims → unique letters → appear in output (lhs side)
    for lf in lhs_free:
        c = alphabet[idx]; idx += 1
        lhs_str[lf] = c; out_str.append(c)

    # Rhs free dims → unique letters → appear in output (rhs side)
    for rf in rhs_free:
        c = alphabet[idx]; idx += 1
        rhs_str[rf] = c; out_str.append(c)

    # Contracting dims → same letter for lhs and rhs → NOT in output (summed)
    for lc, rc in zip(lhs_contracting, rhs_contracting):
        c = alphabet[idx]; idx += 1
        lhs_str[lc] = c; rhs_str[rc] = c

    einsum_str = f"{''.join(lhs_str)},{''.join(rhs_str)}->{''.join(out_str)}"
    return np.einsum(einsum_str, lhs, rhs)


rng = np.random.default_rng(42)

# All cases from the theory
cases = [
    {
        "name": "Vector dot product  [N] · [N] → scalar",
        "lhs":  rng.normal(size=(6,)).astype(np.float32),
        "rhs":  rng.normal(size=(6,)).astype(np.float32),
        "lb": [], "rb": [], "lc": [0], "rc": [0],
        "ref": lambda l,r: np.dot(l, r),
    },
    {
        "name": "Matrix multiply  [M,K] @ [K,N] → [M,N]",
        "lhs":  rng.normal(size=(3,4)).astype(np.float32),
        "rhs":  rng.normal(size=(4,5)).astype(np.float32),
        "lb": [], "rb": [], "lc": [1], "rc": [0],
        "ref": lambda l,r: l @ r,
    },
    {
        "name": "Batched matmul  [B,M,K] @ [B,K,N] → [B,M,N]",
        "lhs":  rng.normal(size=(2,3,4)).astype(np.float32),
        "rhs":  rng.normal(size=(2,4,5)).astype(np.float32),
        "lb": [0], "rb": [0], "lc": [2], "rc": [1],
        "ref": lambda l,r: np.einsum("bik,bkj->bij", l, r),
    },
    {
        "name": "MH-Attn scores  Q[B,H,S,D] @ K[B,H,S,D] → [B,H,S,S]",
        "lhs":  rng.normal(size=(2,4,8,16)).astype(np.float32),
        "rhs":  rng.normal(size=(2,4,8,16)).astype(np.float32),
        "lb": [0,1], "rb": [0,1], "lc": [3], "rc": [3],
        "ref": lambda l,r: np.einsum("bhid,bhjd->bhij", l, r),
    },
    {
        "name": "Outer product  [M] ⊗ [N] → [M,N]  (no contraction)",
        "lhs":  rng.normal(size=(3,)).astype(np.float32),
        "rhs":  rng.normal(size=(4,)).astype(np.float32),
        "lb": [], "rb": [], "lc": [], "rc": [],
        "ref": lambda l,r: np.outer(l, r),
    },
]

print(f"  {'Case':55s}  {'Output shape':15s}  {'Max error':>10s}")
print("  " + "-" * 85)
for c in cases:
    result = dot_general(c["lhs"], c["rhs"], c["lb"], c["rb"], c["lc"], c["rc"])
    ref    = c["ref"](c["lhs"], c["rhs"])
    err    = float(np.max(np.abs(result - ref)))
    ok     = "✅" if err < 1e-4 else "❌"
    print(f"  {c['name']:55s}  {str(result.shape):15s}  {err:>10.2e} {ok}")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: reduce — all patterns
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — reduce: sum, max, argmax, multi-output, multi-dim")
print("━" * 65)
print()

def stablehlo_reduce(x: np.ndarray, init: float,
                     op: str, dims: list) -> np.ndarray:
    """Implement stablehlo.reduce in numpy."""
    result = x.copy()
    for d in sorted(dims, reverse=True):  # reduce from highest dim down
        if op == "add":
            result = np.sum(result, axis=d)
        elif op == "max":
            result = np.max(result, axis=d)
        elif op == "min":
            result = np.min(result, axis=d)
        elif op == "mul":
            result = np.prod(result, axis=d)
        elif op == "and":
            result = np.all(result, axis=d)
        elif op == "or":
            result = np.any(result, axis=d)
    return result

x = rng.normal(size=(3, 4, 5)).astype(np.float32)
print(f"  Input tensor shape: {x.shape}")
print()
cases_reduce = [
    ("sum, dim=[1]",    "add", [1],    np.sum(x,  axis=1)),
    ("sum, dim=[0,2]",  "add", [0,2],  np.sum(x,  axis=(0,2))),
    ("max, dim=[2]",    "max", [2],    np.max(x,  axis=2)),
    ("min, dim=[0,1]",  "min", [0,1],  np.min(x,  axis=(0,1))),
    ("prod, dim=[1,2]", "mul", [1,2],  np.prod(x, axis=(1,2))),
]
for desc, op, dims, ref in cases_reduce:
    result = stablehlo_reduce(x, 0.0, op, dims)
    err    = float(np.max(np.abs(result - ref)))
    print(f"  reduce({desc}): {str(result.shape):12s}  max_err={err:.2e} {'✅' if err<1e-4 else '❌'}")

print()

# Multi-output reduce: argmax (values + indices together)
ARGMAX_THEORY = """
  MULTI-OUTPUT REDUCE: Argmax via stablehlo.reduce
  ─────────────────────────────────────────────────────────────────
  Finding argmax requires tracking both the max VALUE and its INDEX.
  StableHLO reduce supports multiple operands simultaneously:

    // Create index tensor: [[0,1,2,...,N-1], [0,1,...], ...]
    %idx = stablehlo.iota, iota_dimension = 1 : tensor<3x4xi32>

    // Reduce both values and indices together
    %max_val, %max_idx = stablehlo.reduce(
        %x, %idx,                 ; two inputs
        %neg_inf, %zero_i32       ; two init values
    ) applies @argmax_fn across dimensions=[1]
    : (tensor<3x4xf32>, tensor<3x4xi32>, tensor<f32>, tensor<i32>)
    → (tensor<3xf32>, tensor<3xi32>)

    // @argmax_fn compares by value, keeps index from winner:
    func.func @argmax_fn(%val_lhs: f32, %idx_lhs: i32,
                          %val_rhs: f32, %idx_rhs: i32)
                          → (f32, i32) {
      %cond   = stablehlo.compare %val_lhs, %val_rhs {GE}
      %max_v  = stablehlo.select %cond, %val_lhs, %val_rhs
      %max_i  = stablehlo.select %cond, %idx_lhs, %idx_rhs
      return %max_v, %max_i
    }
"""
print(ARGMAX_THEORY)

# Implement multi-output reduce (argmax) in numpy
x2d = rng.normal(size=(4, 8)).astype(np.float32)
argmax_result = np.argmax(x2d, axis=1)
argmax_vals   = np.max(x2d, axis=1)
print(f"  Multi-output reduce (argmax) on [{x2d.shape[0]}, {x2d.shape[1]}]:")
for i in range(x2d.shape[0]):
    print(f"    row {i}: max={argmax_vals[i]:.4f} at index {argmax_result[i]}")
    assert x2d[i, argmax_result[i]] == argmax_vals[i]
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Full attention layer in StableHLO IR
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Full attention layer: StableHLO IR walkthrough")
print("━" * 65)
print()

ATTENTION_SHLO = """
  SCALED DOT-PRODUCT ATTENTION in StableHLO
  ════════════════════════════════════════════════════════════════
  Input: Q, K, V each tensor<BxHxSxDxf32>, mask tensor<1x1xSxSxf32>
  Output: tensor<BxHxSxDxf32>

  func.func @scaled_dot_product_attention(
      %Q:    tensor<BxHxSxDxf32>,
      %K:    tensor<BxHxSxDxf32>,
      %V:    tensor<BxHxSxDxf32>,
      %mask: tensor<1x1xSxSxf32>
  ) → tensor<BxHxSxDxf32> {

    // ── 1. QK^T via dot_general (batch & heads are batching dims) ─────────
    %scores = stablehlo.dot_general %Q, %K,
              lhs_batching_dimensions = [0, 1],    ; B, H are batched
              rhs_batching_dimensions = [0, 1],
              lhs_contracting_dimensions = [3],    ; D is contracted
              rhs_contracting_dimensions = [3]     ; D is contracted (K^T)
              : (tensor<BxHxSxDxf32>, tensor<BxHxSxDxf32>) → tensor<BxHxSxSxf32>
    // result[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]
    // This IS Q @ K^T — note both contract on the SAME (last) dimension

    // ── 2. Scale by 1/sqrt(D) ─────────────────────────────────────────────
    %scale_val = stablehlo.constant dense<0.25> : tensor<f32>     ; 1/sqrt(D) if D=16
    %scale_bc  = stablehlo.broadcast_in_dim %scale_val, dims=[]
                 : (tensor<f32>) → tensor<BxHxSxSxf32>
    %scores_scaled = stablehlo.multiply %scores, %scale_bc
                     : tensor<BxHxSxSxf32>

    // ── 3. Add causal mask (before softmax) ──────────────────────────────
    %mask_bc = stablehlo.broadcast_in_dim %mask, dims=[0,1,2,3]
               : (tensor<1x1xSxSxf32>) → tensor<BxHxSxSxf32>
    // mask contains 0.0 for positions to keep, -inf for positions to mask
    %scores_masked = stablehlo.add %scores_scaled, %mask_bc
                     : tensor<BxHxSxSxf32>

    // ── 4. Softmax along last dimension (attend-to positions axis) ────────
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %row_max  = stablehlo.reduce(%scores_masked, %neg_inf)
                applies @max_fn across dimensions=[3]
                : (tensor<BxHxSxSxf32>, tensor<f32>) → tensor<BxHxSxf32>
    %max_bc   = stablehlo.broadcast_in_dim %row_max, dims=[0,1,2]
                : (tensor<BxHxSxf32>) → tensor<BxHxSxSxf32>
    %shifted  = stablehlo.subtract %scores_masked, %max_bc
                : tensor<BxHxSxSxf32>
    %exp_     = stablehlo.exponential %shifted : tensor<BxHxSxSxf32>
    %zero_f   = stablehlo.constant dense<0.0> : tensor<f32>
    %row_sum  = stablehlo.reduce(%exp_, %zero_f)
                applies @add_fn across dimensions=[3]
                : (tensor<BxHxSxSxf32>, tensor<f32>) → tensor<BxHxSxf32>
    %sum_bc   = stablehlo.broadcast_in_dim %row_sum, dims=[0,1,2]
                : (tensor<BxHxSxf32>) → tensor<BxHxSxSxf32>
    %attn_weights = stablehlo.divide %exp_, %sum_bc : tensor<BxHxSxSxf32>

    // ── 5. Weighted sum: attn_weights @ V ────────────────────────────────
    %output = stablehlo.dot_general %attn_weights, %V,
              lhs_batching_dimensions = [0, 1],    ; B, H are batched
              rhs_batching_dimensions = [0, 1],
              lhs_contracting_dimensions = [3],    ; contract over the key-sequence dim
              rhs_contracting_dimensions = [2]     ; contract over the value-sequence dim
              : (tensor<BxHxSxSxf32>, tensor<BxHxSxDxf32>) → tensor<BxHxSxDxf32>
    // output[b,h,q,d] = sum_k attn_weights[b,h,q,k] * V[b,h,k,d]

    return %output : tensor<BxHxSxDxf32>
  }

  OP COUNT: 13 StableHLO ops for one attention layer
  XLA FUSION: exp+subtract → 1 kernel
              reduce+divide → 2 kernels
              dot_general   → 2 cuBLAS calls
  TOTAL: ~5-6 kernel launches on GPU (down from ~12+ eager)
"""
print(ATTENTION_SHLO)

# Verify the attention semantics in numpy
B,H,S,D = 2,4,8,16
Q = rng.normal(size=(B,H,S,D)).astype(np.float32)
K = rng.normal(size=(B,H,S,D)).astype(np.float32)
V = rng.normal(size=(B,H,S,D)).astype(np.float32)

scale   = 1.0 / np.sqrt(D)
scores  = np.einsum("bhid,bhjd->bhij", Q, K) * scale
scores  -= scores.max(axis=-1, keepdims=True)
weights = np.exp(scores)
weights /= weights.sum(axis=-1, keepdims=True)
output  = np.einsum("bhij,bhjd->bhid", weights, V)

if HAS_JAX:
    ref_out = jax.nn.dot_product_attention(
        jnp.array(Q), jnp.array(K), jnp.array(V), scale=scale)
    err = float(np.max(np.abs(output - np.array(ref_out))))
    print(f"  Attention verification vs JAX reference: max_err={err:.2e} "
          f"{'✅' if err < 1e-4 else '❌'}")
else:
    print(f"  Attention output shape: {output.shape}")
    print(f"  Output norm: {np.linalg.norm(output):.4f}")
    print(f"  Attention weights sum to 1.0: {np.allclose(weights.sum(-1), 1.0)} ✅")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Control Flow and Distributed Ops — while, if, all_reduce": {
        "description": (
            "Implement StableHLO control flow ops in Python: while, if, case. "
            "Show the region model: how inline computations work in MLIR. "
            "Trace a recurrent computation (RNN cell) through a while loop. "
            "Simulate SPMD distributed ops: all_reduce, all_gather, reduce_scatter. "
            "Verify gradient averaging via all_reduce against manual implementation. "
            "Show send/recv token ordering: why tokens are necessary in a pure IR."
        ),
        "language": "python",
        "code": '''
import numpy as np
from typing import Callable, List, Tuple, Any

print("=" * 65)
print("  CONTROL FLOW AND DISTRIBUTED OPS — WHILE, IF, ALL_REDUCE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: stablehlo.while — the loop semantics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — stablehlo.while: loop semantics and functional state")
print("━" * 65)
print()

WHILE_THEORY = """
  STABLEHLO.WHILE — KEY PROPERTIES
  ════════════════════════════════════════════════════════════════

  1. PURELY FUNCTIONAL: state passes through arguments, not mutations.
     All loop state is explicit as block arguments.
     No side-effecting access to "outer scope" variables.
     This enables: unrolling, loop inversion, pipelining.

  2. TYPE HOMOGENEITY: cond and body must take/return the SAME types.
     The type signature is fixed at definition time.
     This enables: static memory planning (buffer sizes known at compile time).

  3. UNKNOWN TRIP COUNT: the compiler cannot statically determine the
     number of iterations (unless static analysis succeeds).
     This differs from stablehlo.reduce or stablehlo.map which have known
     sizes and can always be fully parallelised.

  4. CONDITION IS EVALUATED BEFORE THE FIRST ITERATION.
     while(init, cond, body): if cond(init) is false, immediately returns init.
     No "do-while" in StableHLO; a do-while requires first calling body once.

  EXAMPLE: computing e^x via Taylor series
    e^x ≈ 1 + x + x^2/2! + x^3/3! + ... until term < epsilon

  State carried: (accumulated_sum, current_term, current_n)
    init_sum  = 1.0          (the first term: x^0/0! = 1)
    init_term = x            (the second term: x^1/1! = x)
    init_n    = 1

  cond: current_term > epsilon
  body:
    new_sum  = accumulated_sum + current_term
    new_n    = current_n + 1
    new_term = current_term * x / new_n
    return (new_sum, new_term, new_n)
"""
print(WHILE_THEORY)

def stablehlo_while(init_state: tuple,
                    cond: Callable,
                    body: Callable) -> tuple:
    """
    Simulate stablehlo.while semantics.
    - init_state: initial tuple of tensors (here, Python floats for simplicity)
    - cond: function state → bool scalar tensor
    - body: function state → new state (same types)
    Both cond and body take and return TUPLES to model StableHLO's
    multiple-operand while loops.
    """
    state = init_state
    iterations = 0
    while cond(state):
        state = body(state)
        iterations += 1
    return state, iterations

# Taylor series for exp(x)
def exp_while(x: float, eps: float = 1e-8) -> float:
    """Compute e^x via stablehlo.while with Taylor series."""
    def cond(state):
        _, term, _ = state
        return abs(term) > eps

    def body(state):
        s, term, n = state
        new_s    = s + term
        new_n    = n + 1
        new_term = term * x / new_n
        return (new_s, new_term, new_n)

    (result, _, n), iters = stablehlo_while(
        init_state=(1.0, x, 1),
        cond=cond,
        body=body
    )
    return result + 0, iters   # final accumulation + last term already in state

print("  Taylor series exp(x) via stablehlo.while semantics:")
print()
print(f"  {'x':>8s}  {'while result':>15s}  {'np.exp(x)':>12s}  {'iterations':>12s}  {'error':>10s}")
print("  " + "-" * 65)
for x in [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]:
    result, iters = exp_while(x)
    ref = np.exp(x)
    err = abs(result - ref)
    print(f"  {x:>8.1f}  {result:>15.8f}  {ref:>12.8f}  {iters:>12d}  {err:>10.2e}")

print()

# RNN cell via while loop
print("  RNN unrolling via stablehlo.while:")
print()

RNN_SHLO = """
  Simple RNN: h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
  Unrolled via stablehlo.while over a sequence of length T.

  State: (h, x_sequence, step_idx)
    h:          current hidden state  tensor<Hxf32>
    x_sequence: full input sequence   tensor<TxDxf32>
    step_idx:   current time step     tensor<i32>

  func.func @rnn_forward(%x_seq: tensor<TxDxf32>, %h_init: tensor<Hxf32>,
                          %W_h: tensor<HxHxf32>, %W_x: tensor<HxDxf32>,
                          %b: tensor<Hxf32>) → tensor<Hxf32> {
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %T    = stablehlo.constant dense<T> : tensor<i32>

    %final_h, _, _ = stablehlo.while(%h_init, %x_seq, %zero)
      cond = {
        ^bb0(%h: tensor<Hxf32>, %xs: tensor<TxDxf32>, %t: tensor<i32>):
          %cond = stablehlo.compare %t, %T {LT} : tensor<i1>
          stablehlo.return %cond
      }
      body = {
        ^bb0(%h: tensor<Hxf32>, %xs: tensor<TxDxf32>, %t: tensor<i32>):
          // Extract x_t via dynamic_slice
          %x_t = stablehlo.dynamic_slice %xs, %t, %zero, sizes=[1,D]
                 : ... → tensor<1xDxf32>
          %x_t_sq = stablehlo.reshape %x_t : tensor<Dxf32>

          // h_new = tanh(W_h@h + W_x@x_t + b)
          %Wh_h  = stablehlo.dot_general %W_h, %h ...  → tensor<Hxf32>
          %Wx_x  = stablehlo.dot_general %W_x, %x_t_sq ... → tensor<Hxf32>
          %pre   = stablehlo.add (stablehlo.add %Wh_h, %Wx_x), %b
          %h_new = stablehlo.tanh %pre : tensor<Hxf32>

          %one   = stablehlo.constant dense<1> : tensor<i32>
          %t_new = stablehlo.add %t, %one : tensor<i32>
          stablehlo.return %h_new, %xs, %t_new
      }
    return %final_h
  }
"""
print(RNN_SHLO)

# Implement and verify in numpy
T, D, H = 5, 4, 8
rng = np.random.default_rng(0)
W_h  = rng.normal(0, 0.1, (H, H)).astype(np.float32)
W_x  = rng.normal(0, 0.1, (H, D)).astype(np.float32)
b    = np.zeros(H, dtype=np.float32)
h0   = np.zeros(H, dtype=np.float32)
x_seq = rng.normal(size=(T, D)).astype(np.float32)

def rnn_step(h, x, W_h, W_x, b):
    return np.tanh(W_h @ h + W_x @ x + b)

def rnn_body(state, W_h, W_x, b, x_seq):
    h, _, t = state
    x_t     = x_seq[t]
    h_new   = rnn_step(h, x_t, W_h, W_x, b)
    return (h_new, x_seq, t + 1)

(h_final, _, _), steps = stablehlo_while(
    init_state=(h0, x_seq, 0),
    cond=lambda s: s[2] < T,
    body=lambda s: rnn_body(s, W_h, W_x, b, x_seq)
)

# Reference: unrolled loop
h_ref = h0.copy()
for t in range(T):
    h_ref = rnn_step(h_ref, x_seq[t], W_h, W_x, b)

print(f"  RNN (T={T}, D={D}, H={H}):")
print(f"    while steps: {steps}")
print(f"    h_final norm: {np.linalg.norm(h_final):.6f}")
print(f"    h_ref norm:   {np.linalg.norm(h_ref):.6f}")
print(f"    Match: {np.allclose(h_final, h_ref)} ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Distributed ops — all_reduce, all_gather, reduce_scatter
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Distributed ops: all_reduce, all_gather, reduce_scatter")
print("━" * 65)
print()

class SPMDSimulator:
    """
    Simulates SPMD execution across N devices.
    Each device holds its own shard; collective ops exchange data.
    """
    def __init__(self, n_devices: int):
        self.n  = n_devices
        self.log = []

    def all_reduce(self, shards: List[np.ndarray], op: str = "add") -> List[np.ndarray]:
        """
        stablehlo.all_reduce: each device contributes operand,
        every device receives the combined result.
        """
        if op == "add":
            combined = sum(shards)
        elif op == "max":
            combined = np.maximum.reduce(shards)
        elif op == "min":
            combined = np.minimum.reduce(shards)
        # All devices get the same result
        self.log.append(f"all_reduce({op}): {shards[0].shape} × {self.n} → combined {combined.shape}")
        return [combined.copy() for _ in range(self.n)]

    def all_gather(self, shards: List[np.ndarray], concat_dim: int = 0) -> List[np.ndarray]:
        """
        stablehlo.all_gather: concatenate all shards along concat_dim.
        Every device gets the full concatenated tensor.
        """
        full = np.concatenate(shards, axis=concat_dim)
        self.log.append(f"all_gather(dim={concat_dim}): "
                        f"{shards[0].shape} × {self.n} → {full.shape}")
        return [full.copy() for _ in range(self.n)]

    def reduce_scatter(self, shards: List[np.ndarray],
                       op: str = "add", scatter_dim: int = 0) -> List[np.ndarray]:
        """
        stablehlo.reduce_scatter: reduce across devices, then scatter result.
        Device d receives combined[d::n] along scatter_dim.
        """
        combined = sum(shards) if op == "add" else np.maximum.reduce(shards)
        chunk    = combined.shape[scatter_dim] // self.n
        result   = []
        for d in range(self.n):
            slices = [slice(None)] * combined.ndim
            slices[scatter_dim] = slice(d * chunk, (d+1) * chunk)
            result.append(combined[tuple(slices)].copy())
        self.log.append(f"reduce_scatter({op}, dim={scatter_dim}): "
                        f"{shards[0].shape} × {self.n} → {result[0].shape} each")
        return result

    def all_to_all(self, shards: List[np.ndarray],
                   split_dim: int, concat_dim: int) -> List[np.ndarray]:
        """
        stablehlo.all_to_all: split each shard along split_dim,
        distribute pieces, concat along concat_dim.
        """
        chunk = shards[0].shape[split_dim] // self.n
        pieces = []
        for shard in shards:
            row = []
            for d in range(self.n):
                slices = [slice(None)] * shard.ndim
                slices[split_dim] = slice(d*chunk, (d+1)*chunk)
                row.append(shard[tuple(slices)])
            pieces.append(row)
        # Device d collects piece d from each device
        result = []
        for d in range(self.n):
            result.append(np.concatenate([pieces[src][d] for src in range(self.n)],
                                          axis=concat_dim))
        self.log.append(f"all_to_all(split={split_dim}, concat={concat_dim}): "
                        f"{shards[0].shape} → {result[0].shape} each")
        return result


N_DEV = 4
sim = SPMDSimulator(n_devices=N_DEV)

print(f"  Simulating SPMD with {N_DEV} devices:")
print()

# all_reduce: gradient averaging (data-parallel training)
grads = [rng.normal(size=(16,)).astype(np.float32) for _ in range(N_DEV)]
avg_grads = sim.all_reduce(grads, op="add")
avg_grads = [g / N_DEV for g in avg_grads]
expected_avg = sum(grads) / N_DEV

print(f"  1. all_reduce (gradient averaging):")
print(f"     {sim.log[-1]}")
print(f"     Gradient avg matches: {np.allclose(avg_grads[0], expected_avg)} ✅")
print(f"     All devices have same result: {all(np.allclose(avg_grads[0], g) for g in avg_grads)} ✅")
print()

# all_gather: collect sharded weights for output projection
weight_shards = [rng.normal(size=(32//N_DEV, 16)).astype(np.float32) for _ in range(N_DEV)]
full_weights  = sim.all_gather(weight_shards, concat_dim=0)
expected_full = np.concatenate(weight_shards, axis=0)
print(f"  2. all_gather (collect sharded weights):")
print(f"     {sim.log[-1]}")
print(f"     Full weights match: {np.allclose(full_weights[0], expected_full)} ✅")
print()

# reduce_scatter: ZeRO optimizer step
grad_shards   = [rng.normal(size=(32,)).astype(np.float32) for _ in range(N_DEV)]
rs_results    = sim.reduce_scatter(grad_shards, op="add", scatter_dim=0)
combined_grad = sum(grad_shards)
print(f"  3. reduce_scatter (ZeRO optimizer):")
print(f"     {sim.log[-1]}")
for d in range(N_DEV):
    chunk   = len(combined_grad) // N_DEV
    expected_chunk = combined_grad[d*chunk:(d+1)*chunk]
    ok = np.allclose(rs_results[d], expected_chunk)
    print(f"     Device {d}: shard[{d*chunk}:{(d+1)*chunk}] correct: {ok} ✅")
print()

# all_to_all: sequence parallelism
# Before: batch sharded (device d has batch[d]), sequence is local [B//N, S]
seq_shards = [rng.normal(size=(4//N_DEV, 8)).astype(np.float32) for _ in range(N_DEV)]
# After: sequence sharded (device d has sequence[d]), batch is local [B, S//N]
a2a_results = sim.all_to_all(seq_shards, split_dim=1, concat_dim=0)
print(f"  4. all_to_all (sequence parallelism transpose):")
print(f"     {sim.log[-1]}")
print(f"     Original (batch-sharded): {seq_shards[0].shape} × {N_DEV}")
print(f"     Transposed (seq-sharded): {a2a_results[0].shape} × {N_DEV}")
print()
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Versioning, Passes, and the Ecosystem — End to End": {
        "description": (
            "Deep dive into StableHLO's versioning and compatibility system. "
            "Simulate the version upgrade pass: deprecated ops → replacement ops. "
            "Show every important StableHLO pass: canonicalize, shape-inference, legalise-to-linalg. "
            "Trace the full producer→consumer pipeline: JAX export → StableHLO → IREE/XLA. "
            "Compare StableHLO vs MHLO vs HLO with concrete IR examples. "
            "Summarise the complete connected stack position of StableHLO."
        ),
        "language": "python",
        "code": '''
import numpy as np
import re

print("=" * 65)
print("  VERSIONING, PASSES, AND THE ECOSYSTEM — END TO END")
print("=" * 65)
print()

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
    print(f"  JAX {jax.__version__}")
except ImportError:
    HAS_JAX = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Version compatibility simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Versioning: the 5-year compatibility window")
print("━" * 65)
print()

COMPAT_DETAIL = """
  STABLEHLO COMPATIBILITY POLICY — FULL DETAIL
  ════════════════════════════════════════════════════════════════

  StableHLO versions use CALENDAR VERSIONING embedded in the MLIR bytecode:

  BINARY FORMAT HEADER (first bytes of .mlirbc):
    magic:    0x4D4C4952  ("MLIR")
    version:  {major=0, minor=1, patch=0}      ; e.g. StableHLO 0.1.0
    producer: "stablehlo-tools-1.3.2"

  COMPATIBILITY WINDOWS:
    ┌──────────────────────────────────────────────────────────────────┐
    │ TODAY (v1.0, Jan 2024)                                           │
    │   A → saved to disk                                              │
    │          │                                                       │
    │          │   6 months        ← forward compat window             │
    │          ├──────────────────                                     │
    │          │   (consumer at v1.3, Jul 2024 must read v1.0 programs)│
    │          │                                                       │
    │          │   5 years         ← backward compat window            │
    │          ├──────────────────────────────────────────────────     │
    │          │   (consumer at v3.x, Jan 2029 must read v1.0 programs)│
    │                                                                  │
    │ AFTER WINDOW EXPIRES (Jan 2029+):                                │
    │   No guarantee — but upgrade passes may still work as best-effort│
    └──────────────────────────────────────────────────────────────────┘

  HOW BACKWARD COMPAT IS IMPLEMENTED:
    When consumer loads a program from older version V:
      1. Check: is V within the backward compat window?
      2. If yes: apply stablehlo-upgrade-version pass
         This pass contains rules like:
           deprecated_op_from_V_0_9 → replacement_op_in_V_1_0
           old_attribute_encoding  → new_attribute_encoding
         Generated from the compat spec in openxla/stablehlo/docs/compatibility.md
      3. Now the program uses only current-version ops → compile normally.

  REAL CHANGES THAT HAVE REQUIRED COMPAT PASSES:
    v0.9 → v1.0: stablehlo.custom_call got renamed attributes
    v0.9 → v1.0: stablehlo.sort result ordering clarification
    v0.8 → v0.9: FP8 types added (f8e4m3fn, f8e5m2) → consumers < 0.9 reject
    v0.7 → v0.8: stablehlo.reduce semantics tightened for empty reductions
"""
print(COMPAT_DETAIL)

# Simulate version checking and upgrade
class StableHLOVersion:
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __le__(self, other):
        return (self.major, self.minor, self.patch) <= \
               (other.major, other.minor, other.patch)

    def __str__(self):
        return f"v{self.major}.{self.minor}.{self.patch}"

    def months_from(self, other: "StableHLOVersion") -> int:
        """Approximate months between two minor version bumps."""
        return abs((self.minor - other.minor) * 2 +
                   (self.major - other.major) * 24)

def check_compatibility(producer_ver: StableHLOVersion,
                         consumer_ver: StableHLOVersion) -> str:
    months_apart = producer_ver.months_from(consumer_ver)
    if producer_ver <= consumer_ver:
        if months_apart <= 60:   # 5 years ≈ 60 months
            return f"✅ COMPATIBLE (backward compat, {months_apart}mo apart)"
        else:
            return f"⚠️  OUTSIDE WINDOW ({months_apart}mo > 60mo, may still work)"
    else:
        if months_apart <= 6:
            return f"✅ COMPATIBLE (forward compat, {months_apart}mo apart)"
        else:
            return f"❌ INCOMPATIBLE (program too new, {months_apart}mo apart)"

scenarios = [
    (StableHLOVersion(1,0,0), StableHLOVersion(1,3,0),   "Saved Jan 2024, loaded Jul 2024"),
    (StableHLOVersion(1,0,0), StableHLOVersion(2,6,0),   "Saved Jan 2024, loaded Jan 2029"),
    (StableHLOVersion(1,0,0), StableHLOVersion(3,0,0),   "Saved Jan 2024, loaded Jan 2031"),
    (StableHLOVersion(1,5,0), StableHLOVersion(1,3,0),   "Saved now, loaded older consumer"),
    (StableHLOVersion(1,8,0), StableHLOVersion(1,3,0),   "Saved far future, loaded old consumer"),
]

print("  Compatibility check matrix:")
print()
for prod, cons, desc in scenarios:
    result = check_compatibility(prod, cons)
    print(f"  Producer {prod} → Consumer {cons}:  {result}")
    print(f"    Scenario: {desc}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: StableHLO passes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Key StableHLO passes")
print("━" * 65)
print()

PASSES_GUIDE = """
  STABLEHLO PASSES — WHAT THEY DO AND WHEN TO USE THEM
  ════════════════════════════════════════════════════════════════

  All passes are invoked via mlir-opt or stablehlo-opt:
    stablehlo-opt --stablehlo-canonicalize input.mlir -o output.mlir
    stablehlo-opt --stablehlo-legalize-to-linalg input.mlir
    mlir-opt --stablehlo-refine-shapes --convert-stablehlo-to-hlo input.mlir

  ── CANONICALIZATION (--stablehlo-canonicalize) ──────────────────────
  Applies algebraic simplifications and constant folding:

  BEFORE:
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %bc   = stablehlo.broadcast_in_dim %zero, dims=[] : ... → tensor<4xf32>
    %r    = stablehlo.add %x, %bc : tensor<4xf32>

  AFTER:
    %r    = %x   ; add(x, 0) = x → entire sub-expression eliminated

  Rules applied:
    add(x, 0.0)  → x
    mul(x, 1.0)  → x
    mul(x, 0.0)  → broadcast(0.0)
    subtract(x,x)→ broadcast(0.0)
    transpose(transpose(x, p), inv(p)) → x
    reshape(reshape(x,a),b) → reshape(x, b)
    reduce(broadcast(x), dim) → x    [if reduce is identity over that dim]
    slice(x, [0:n]) → x              [if n = size of that dim]

  ── SHAPE REFINEMENT (--stablehlo-refine-shapes) ─────────────────────
  Propagates static shapes through the computation.

  BEFORE:
    func.func @main(%x: tensor<?x4xf32>) → tensor<?x4xf32> {
      %r = stablehlo.add %x, %x : tensor<?x4xf32>

  AFTER (if we know the input shape is [8,4]):
    func.func @main(%x: tensor<8x4xf32>) → tensor<8x4xf32> {
      %r = stablehlo.add %x, %x : tensor<8x4xf32>

  Required before: lowering to linalg (which needs static sizes for tiling).

  ── LEGALISATION TO LINALG (--stablehlo-legalize-to-linalg) ─────────
  The key lowering for IREE and MLIR-based backends.
  Converts StableHLO compute ops to linalg dialect:

  BEFORE:
    %r = stablehlo.add %a, %b : tensor<4x8xf32>

  AFTER:
    %r = linalg.generic
         {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,
                          affine_map<(d0,d1)->(d0,d1)>,
                          affine_map<(d0,d1)->(d0,d1)>],
          iterator_types=["parallel","parallel"]}
         ins(%a, %b : tensor<4x8xf32>, tensor<4x8xf32>)
         outs(%init : tensor<4x8xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out: f32):
        %sum = arith.addf %in0, %in1 : f32
        linalg.yield %sum : f32
    } : tensor<4x8xf32>

  Key conversions:
    stablehlo.add/mul/exp/...  → linalg.generic (elementwise)
    stablehlo.dot_general      → linalg.matmul / linalg.batch_matmul
    stablehlo.reduce           → linalg.reduce
    stablehlo.convolution      → linalg.conv_Nd_...

  ── HLO LEGALISATION (--stablehlo-legalize-to-hlo) ───────────────────
  Converts StableHLO → MHLO for use inside XLA's compiler.
  Used ONLY inside the XLA compilation pipeline.
  Output is MHLO (unstable, XLA-internal dialect).

  ── BATCH NORM EXPANSION (--stablehlo-expand-ops) ─────────────────────
  Expands high-level ops into primitive sequences:
    stablehlo.batch_norm_training → reduce + subtract + divide + multiply + ...
    stablehlo.batch_norm_inference → the inference form (simpler)
  Required for backends that don't natively handle batch norm.

  ── VERSION UPGRADE (--stablehlo-upgrade-version) ────────────────────
  Upgrades old StableHLO programs to current version.
  Applied automatically by jax.export.deserialize() and iree-compile.
  Manual use: stablehlo-opt --stablehlo-upgrade-version old.mlir -o new.mlir
"""
print(PASSES_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: StableHLO vs MHLO vs HLO — concrete comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — StableHLO vs MHLO vs HLO: the disambiguation")
print("━" * 65)
print()

COMPARISON = """
  THREE CLOSELY RELATED IRs — CONCRETE COMPARISON
  ════════════════════════════════════════════════════════════════

  Same computation: relu(x) for x: f32[8]

  ── STABLEHLO (external, stable, MLIR) ──────────────────────────────
  module @relu_stable {
    func.func @main(%x: tensor<8xf32>) → tensor<8xf32> {
      %zero = stablehlo.constant dense<0.0> : tensor<f32>
      %bc   = stablehlo.broadcast_in_dim %zero, dims=[]
              : (tensor<f32>) → tensor<8xf32>
      %r    = stablehlo.maximum %x, %bc : tensor<8xf32>
      return %r : tensor<8xf32>
    }
  }
  Properties: STABLE (guaranteed 5-year compat), MLIR-based, external API.
  File: openxla/stablehlo repo, stablehlo.* op namespace.

  ── MHLO (internal, unstable, MLIR) ────────────────────────────────
  module @relu_mhlo {
    func.func @main(%x: tensor<8xf32>) → tensor<8xf32> {
      %zero = mhlo.constant dense<0.0> : tensor<f32>
      %bc   = "mhlo.broadcast_in_dim"(%zero) {broadcast_dimensions = dense<> : tensor<0xi64>}
              : (tensor<f32>) → tensor<8xf32>
      %r    = mhlo.maximum %x, %bc : tensor<8xf32>
              return %r : tensor<8xf32>
    }
  }
  Properties: UNSTABLE (changes with XLA releases), MLIR-based, internal to XLA.
  File: tensorflow/compiler/mlir/hlo/include repo, mhlo.* op namespace.
  Note: attribute encoding and some op names differ from StableHLO.

  ── XLA HLO (internal, unstable, C++ proto) ────────────────────────
  HloModule relu_hlo

  ENTRY relu {
    x    = f32[8] parameter(0)
    zero = f32[] constant(0)
    bc   = f32[8] broadcast(zero), dimensions={}
    ROOT r = f32[8] maximum(x, bc)
  }
  Properties: UNSTABLE, NOT MLIR (C++ protobuf), XLA-internal codegen target.
  File: tensorflow/compiler/xla/service repo, defined by HloInstruction C++ class.

  ── KEY DIFFERENCES SUMMARISED ──────────────────────────────────────
  ┌────────────────┬──────────────┬──────────────┬───────────────────┐
  │ Property       │ StableHLO    │ MHLO         │ XLA HLO           │
  ├────────────────┼──────────────┼──────────────┼───────────────────┤
  │ Stable API?    │ YES (5yr)    │ NO           │ NO                │
  │ MLIR dialect?  │ YES          │ YES          │ NO (C++ proto)    │
  │ External?      │ YES          │ NO           │ NO                │
  │ Namespace      │ stablehlo.*  │ mhlo.*       │ HloInstruction    │
  │ Repository     │ openxla/     │ tensorflow/  │ tensorflow/       │
  │                │ stablehlo    │ mlir/hlo     │ compiler/xla      │
  │ Use for...     │ portability  │ XLA-internal │ codegen input     │
  │ Consumers      │ All tools    │ XLA only     │ XLA backends only │
  │ Dynamic shapes │ YES          │ YES          │ NO (static only)  │
  └────────────────┴──────────────┴──────────────┴───────────────────┘

  MIGRATION PATH INSIDE XLA:
    JAX program
      ↓ jax.export / JAX tracing
    StableHLO (external, stable)
      ↓ --stablehlo-legalize-to-hlo  (inside XLA, not public)
    MHLO (unstable, XLA-internal MLIR)
      ↓ MlirToHloTranslate (inside XLA)
    XLA HLO protos (C++ objects, not MLIR)
      ↓ HLO passes (fusion, layout, buffer assignment)
    GPU/CPU/TPU binary
"""
print(COMPARISON)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Connected stack summary
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — StableHLO in the connected compiler stack")
print("━" * 65)
print()

STACK = """
  STABLEHLO POSITION IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌───────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                      │
  │  StableHLO is never lowered DIRECTLY to LLVM IR.                      │
  │  Path: StableHLO → linalg (IREE path) → LLVM dialect → LLVM IR        │
  │     or: StableHLO → MHLO → HLO protos → XLA CPU emitter → LLVM IR     │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 02: MLIR                                                      │
  │  StableHLO IS an MLIR dialect. It inherits:                           │
  │    - MLIR's SSA form and type system                                  │
  │    - MLIR's pass manager (all StableHLO passes use PassManager)       │
  │    - MLIR's pattern rewriting (canonicalization uses RewritePatterns) │
  │    - MLIR's binary bytecode format (.mlirbc serialisation)            │
  │    - MLIR's FileCheck+lit testing infrastructure                      │
  │    - MLIR's LSP server (hover, goto-definition in .mlir files)        │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 03: CIRCT                                                     │
  │  StableHLO → hardware: the research path                              │
  │  StableHLO → Calyx dialect (CIRCT) → firtool → Verilog → FPGA/ASIC    │
  │  Enables: hardware accelerator design from JAX-trained model          │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 04: ENZYME                                                    │
  │  Enzyme differentiates StableHLO via Enzyme-MLIR.                     │
  │  Gradient of stablehlo.dot_general → stablehlo.dot_general (exact).   │
  │  JAX's jax.grad operates at this level: produces differentiated SHLO. │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 05: XLA                                                       │
  │  XLA compiles StableHLO via: StableHLO → MHLO → HLO protos            │
  │  StableHLO is XLA's INTAKE FORMAT (what JAX/TF produce for XLA).      │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 06: OpenXLA                                                   │
  │  StableHLO is the PORTABILITY CORE of OpenXLA.                        │
  │  OpenXLA governs the StableHLO spec (openxla/stablehlo repo).         │
  │  IREE, Shardy, AutoSharding all operate on StableHLO.                 │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 07: StableHLO (THIS MODULE)                                   │
  │  The specification, type system, op semantics, broadcasting rules,    │
  │  versioning policy, serialisation, and reference interpreter.         │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM                                                       │
  │  TVM's Relax IR can import StableHLO as an input format.              │
  │  tvm.relay.from_stablehlo("model.mlir") → TVM Relax IR                │
  │  Enables: train with JAX → export StableHLO → deploy with TVM.        │
  └───────────────────────────────────────────────────────────────────────┘

  QUICK REFERENCE:
  ─────────────────────────────────────────────────────────────────────────
"""
print(STACK)

print("  ┌─────────────────────────────────────────────────────────────────────────────┐")
print("  │ Task                     │ Tool / API                                       │")
print("  ├─────────────────────────────────────────────────────────────────────────────┤")
print("  │ Produce StableHLO (JAX)  │ jax.export.export(jit(fn))(args)                 │")
print("  │ Inspect MLIR text        │ exported.mlir_module()                           │")
print("  │ Serialise                │ exported.serialize() → bytes                     │")
print("  │ Load and run             │ jax.export.deserialize(bytes)                    │")
print("  │ Produce (PyTorch)        │ torch_mlir.compile(OutputType.STABLEHLO)         │")
print("  │ Canonicalize             │ stablehlo-opt --stablehlo-canonicalize           │")
print("  │ To linalg (for IREE)     │ stablehlo-opt --stablehlo-legalize-to-linalg     │")
print("  │ Shape refine             │ stablehlo-opt --stablehlo-refine-shapes          │")
print("  │ Upgrade old version      │ stablehlo-opt --stablehlo-upgrade-version        │")
print("  │ Reference interpreter    │ stablehlo-opt --stablehlo-interpret              │")
print("  │ Verify constraints       │ mlir-opt --verify-diagnostics                    │")
print("  ├─────────────────────────────────────────────────────────────────────────────┤")
print("  │ Key ops (compute)        │ dot_general, convolution, reduce                 │")
print("  │ Key ops (shape)          │ reshape, transpose, broadcast_in_dim             │")
print("  │ Key ops (control flow)   │ while, if, case                                  │")
print("  │ Key ops (distributed)    │ all_reduce, all_gather, reduce_scatter           │")
print("  │ Element types            │ bf16, f32, si8, ui4, f8e4m3fn                    │")
print("  ├─────────────────────────────────────────────────────────────────────────────┤")
print("  │ Repository               │ github.com/openxla/stablehlo                     │")
print("  │ Specification            │ openxla/stablehlo/blob/main/docs/spec.md         │")
print("  │ Compat policy            │ openxla/stablehlo/blob/main/docs/compatibility.md│")
print("  └─────────────────────────────────────────────────────────────────────────────┘")
print()

if HAS_JAX:
    def layer_norm_and_relu(x, gamma, beta):
        """Combined layer norm + relu."""
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var  = jnp.var(x,  axis=-1, keepdims=True)
        x_n  = (x - mean) / jnp.sqrt(var + 1e-5)
        return jnp.maximum(gamma * x_n + beta, 0.0)

    x_abs     = jax.ShapeDtypeStruct((4, 16), jnp.float32)
    gamma_abs = jax.ShapeDtypeStruct((16,),   jnp.float32)
    beta_abs  = jax.ShapeDtypeStruct((16,),   jnp.float32)

    try:
        exp = jax.export.export(jax.jit(layer_norm_and_relu))(
            x_abs, gamma_abs, beta_abs)
        mlir_text = exp.mlir_module()
        shlo_ops = [l.strip() for l in mlir_text.split("\\n")
                    if "stablehlo." in l]
        unique_ops = sorted(set(re.sub(r" %.*", "", s).strip()
                                for s in shlo_ops if "=" in s))
        print(f"  layer_norm_and_relu StableHLO op types used ({len(unique_ops)}):")
        for op in unique_ops:
            op_clean = op.split("=")[1].strip().split(" ")[0] if "=" in op else op
            print(f"    {op_clean}")
    except Exception as e:
        print(f"  (StableHLO export: {e})")
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