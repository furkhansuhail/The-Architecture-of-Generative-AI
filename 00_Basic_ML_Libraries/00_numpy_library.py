"""
NumPy — Numerical Computing with Python
========================================

NumPy (Numerical Python) is the foundational library for scientific computing
in Python and the backbone of virtually every machine learning framework.

"""

import os
import re
import textwrap


TOPIC_NAME   = "NumPy: Numerical Computing with Python"
DISPLAY_NAME = "03 · NumPy"
ICON         = "🔢"
SUBTITLE     = "Arrays, Broadcasting, and the Mathematics of ML Data"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is NumPy?

NumPy (Numerical Python) is the foundational library for scientific computing in Python.
It provides a powerful N-dimensional array object — the `ndarray` — along with a vast
collection of mathematical functions to operate on those arrays efficiently.

Almost every major machine learning library (PyTorch, TensorFlow, scikit-learn, pandas)
is built on top of NumPy arrays under the hood. Understanding NumPy means understanding
the data representation layer of all of modern ML.

Think of it this way: all data in machine learning — images, text embeddings, weight
matrices, loss values — ultimately live as grids of numbers. NumPy is the language for
working with those grids.


### Why NumPy Over Pure Python?

Pure Python is slow for numerical computation because:

    1. Python lists store pointers to arbitrary objects scattered in memory.
       CPU caches cannot predict the next memory access → cache misses → slow.

    2. Python's type system is dynamic. Every operation requires runtime type-checking.

    3. Python loops execute one element at a time. Modern CPUs can process 4–8
       numbers simultaneously using SIMD (Single Instruction, Multiple Data) instructions.

NumPy solves all three problems:

    1. NumPy arrays store homogeneous data in a single contiguous block of memory.
       The CPU can prefetch the next element before it's needed → fast.

    2. NumPy arrays have a fixed dtype. No type-checking at runtime.

    3. NumPy's operations are implemented in C and Fortran, which the CPU executes
       at native speed with SIMD vectorization. This is called *vectorization*.

The result: NumPy is typically 10-100× faster than equivalent Python loops,
and for large arrays the gap grows wider.


### Vectorization and Broadcasting

**Vectorization** means expressing operations as whole-array transforms rather than
element-by-element loops. Instead of:

    result = []
    for x in data:
        result.append(x * 2)

you write:

    result = data * 2          # operates on all elements simultaneously

This is not just syntactic sugar — the underlying computation is parallelized at the
hardware level.

**Broadcasting** is NumPy's mechanism for performing operations between arrays of
different (but compatible) shapes without copying data. Instead of manually tiling a
smaller array to match a larger one, NumPy "stretches" it conceptually during the
computation. No extra memory is allocated.

Broadcasting rules (evaluated from the trailing dimension leftward):
    1. If arrays differ in number of dimensions, prepend 1s to the smaller shape.
    2. Dimensions of size 1 are stretched to match the other array's dimension.
    3. Dimensions of different sizes (neither = 1) → error.

    Example:
        A has shape (5, 3)   — 5 rows, 3 columns
        B has shape    (3,)  — a 1-D vector of 3 values

        Step 1: B is treated as (1, 3)
        Step 2: B is stretched to (5, 3) — B's single row is "broadcast" to all 5 rows

    Result: A + B adds B's 3 values to every row of A. Zero extra memory used.


### The ndarray — NumPy's Core Object

Every array in NumPy is an instance of `numpy.ndarray`. It has these key attributes:

    Attribute       Meaning
    ──────────────────────────────────────────────────────
    .shape          Tuple of dimension sizes, e.g. (3, 4) for a 3×4 matrix
    .ndim           Number of dimensions (len(shape))
    .dtype          Data type of elements: float64, int32, bool, etc.
    .size           Total number of elements (product of shape)
    .itemsize       Bytes per element
    .nbytes         Total bytes (= size × itemsize)
    .T              Transposed view (axes reversed)


### Array Dimensions — Scalar, Vector, Matrix, Tensor

The number of dimensions determines how we think about the data:

    Dimensions    Name       Example Shape    ML Meaning
    ──────────────────────────────────────────────────────────────────────────
    0-D           Scalar     ()               A single loss value: 0.342
    1-D           Vector     (n,)             One training example, a 1-D signal
    2-D           Matrix     (m, n)           A batch of examples (rows=samples)
    3-D           Tensor     (d, m, n)        A sequence of matrices (e.g. time steps)
    4-D           Tensor     (b, h, w, c)     A batch of images (batch, height, width, channels)
    N-D           Tensor     (...)            Any higher-order structure

NumPy uses the same `ndarray` type for all of them. Every operation available on a
2-D matrix is also available on a 4-D image batch — the API is consistent.


    ANATOMY OF A NUMPY ARRAY
    ══════════════════════════════════════════════════════════════

    SCALAR         VECTOR          MATRIX              3-D TENSOR
    (shape: ())    (shape: (3,))   (shape: (2, 3))     (shape: (2, 2, 3))

       7           [1, 2, 3]       [[1,  2,  3],        [[[1, 2, 3],
                                    [4,  5,  6]]          [4, 5, 6]],
                                                          [[7, 8, 9],
                                                           [0, 1, 2]]]

    0 axes         1 axis          2 axes               3 axes
                   ↑               ↑     ↑               ↑   ↑   ↑
                   rows            rows  cols          depth rows cols


### Data Types (dtype)

NumPy arrays are typed. Every element is the same type, stored as raw binary.
Common dtypes in ML:

    dtype           Bytes   Range / Precision       Common Use
    ──────────────────────────────────────────────────────────────────────────
    float64 (f8)      8     ~15 decimal digits       Default, science, loss values
    float32 (f4)      4     ~7 decimal digits        GPU training (half the memory)
    float16 (f2)      2     ~3 decimal digits        Mixed-precision training
    int64   (i8)      8     ±9.2 × 10¹⁸              Large integer labels
    int32   (i4)      4     ±2.1 × 10⁹               Common integer labels
    bool    (b)       1     True / False             Masks, comparison results

Type promotion: when arrays of different dtypes are combined, NumPy upcasts to the
type that can represent both (e.g. int32 + float64 → float64).


### Creating Arrays

NumPy provides many constructors:

    Function                    What it creates
    ──────────────────────────────────────────────────────────────────────────
    np.array([1, 2, 3])         From a Python list or tuple
    np.zeros((m, n))            All zeros, shape (m, n), dtype float64
    np.ones((m, n))             All ones
    np.full((m, n), val)        All elements = val
    np.eye(n)                   Identity matrix n×n
    np.arange(start, stop, step)  Like Python range(), returns ndarray
    np.linspace(start, stop, n)   n evenly-spaced values between start and stop
    np.random.randint(low, high, size)  Random integers in [low, high)
    np.random.random(size)      Random floats in [0, 1)
    np.random.randn(m, n)       Standard normal distribution (mean=0, std=1)
    np.random.seed(n)           Fix the random seed for reproducibility


### Indexing and Slicing

NumPy indexing is an extension of Python list indexing to multiple dimensions.
Each dimension gets its own index/slice separated by a comma:

    a[row, col]             Single element
    a[start:stop:step, :]   Slice rows; all columns
    a[:, j]                 All rows, column j → returns 1-D array
    a[..., -1]              Last element along the last axis (ellipsis)
    a[[0, 2, 4], :]         Fancy indexing — select specific rows by index list
    a[a > 0]                Boolean indexing — select elements where condition is True

Indexing starts at 0. Negative indices count from the end (-1 = last element).
Slices follow Python semantics: stop is exclusive.

    For a 3-D array with shape (2, 3, 4):
        a[0]           → shape (3, 4)  — first "depth slice"
        a[0, 1]        → shape (4,)    — second row of first slice
        a[0, 1, 2]     → scalar       — single element
        a[:, :, :2]    → shape (2, 3, 2) — first 2 columns of every slice


### Arithmetic and Universal Functions (ufuncs)

Standard operators (+, -, *, /, //, **, %) all work element-wise on arrays.
NumPy also provides *universal functions* (ufuncs) — vectorized implementations of
mathematical operations that operate element-wise and respect broadcasting:

    Ufunc               Operation
    ──────────────────────────────────────────────────────────────────────────
    np.add, np.subtract  Element-wise + / -
    np.multiply          Element-wise ×
    np.divide            Element-wise ÷
    np.power             Element-wise exponentiation
    np.sqrt              Square root
    np.exp               e^x
    np.log               Natural log (ln)
    np.sin, np.cos       Trigonometric functions
    np.abs               Absolute value
    np.sign              Sign of each element (-1, 0, +1)
    np.clip              Clamp values to [min, max]


### Aggregation Functions

Aggregation reduces an array to summary statistics along one or more axes:

    Function            Reduces to
    ──────────────────────────────────────────────────────────────────────────
    np.sum(a, axis=k)   Sum along axis k (omit axis → sum all elements)
    np.mean(a, axis=k)  Arithmetic mean
    np.std(a, axis=k)   Standard deviation  (√variance)
    np.var(a, axis=k)   Variance  (mean of squared deviations from mean)
    np.min / np.max     Minimum / maximum value
    np.argmin/argmax    Index of minimum / maximum value
    np.cumsum           Cumulative sum (does not reduce — same shape as input)
    np.prod             Product of elements

The `axis` argument controls which dimension is collapsed:
    - axis=0 collapses rows → result has shape (cols,)
    - axis=1 collapses cols → result has shape (rows,)
    - axis=None (default) collapses everything → scalar


### Reshaping, Transposing, and Stacking

Shape manipulation is the bread-and-butter of ML data pipelines:

    Operation               Effect
    ──────────────────────────────────────────────────────────────────────────
    a.reshape(new_shape)    Change shape; total elements must be conserved
    a.flatten()             1-D copy of all elements, row-major order
    a.ravel()               1-D view (no copy if possible)
    a.T                     Reverse all axes (transpose)
    a.transpose(axes)       Permute axes in specified order
    np.expand_dims(a, k)    Insert new axis of size 1 at position k
    a[np.newaxis, :]        Same as expand_dims at axis 0
    np.squeeze(a)           Remove all size-1 axes
    np.concatenate([a,b], axis=k)  Join arrays along an existing axis
    np.stack([a,b], axis=k)        Join arrays along a NEW axis
    np.vstack / np.hstack          Vertical / horizontal stacking shorthands


### Dot Product and Matrix Multiplication

The dot product is the fundamental operation of neural networks — every linear layer
is a matrix multiplication. NumPy provides two equivalent ways to express it:

    np.dot(A, B)   — works for 1-D (dot product) and 2-D (matrix multiply)
    A @ B          — the matmul operator, preferred for clarity (Python 3.5+)

Rules for matrix multiplication (inner dimensions must match):
    (m, k) @ (k, n)  →  (m, n)

The result shape comes from the OUTER dimensions; the INNER dimension (k) is consumed.

    Intuition: A is m questions, each of length k.
               B is k features, each mapped to n outputs.
               Result: m answers, each of length n.

Dot product vs. element-wise multiply:
    A @ B      — dot product      (reduces the inner dimension)
    A * B      — Hadamard product (element-wise; shapes must broadcast)


### Statistical Concepts: Mean, Variance, Standard Deviation

These appear constantly in ML — in normalization, loss functions, and diagnostics:

**Mean (μ)** — The arithmetic average. The "center" of the distribution.

        μ = (1/n) × Σxᵢ

**Variance (σ²)** — The average squared distance from the mean.
    Measures how "spread out" the values are.

        σ² = (1/n) × Σ(xᵢ - μ)²

**Standard Deviation (σ)** — Square root of variance.
    Same units as the original data, easier to interpret than variance.

        σ = √σ²

    High std → values are spread far from the mean.
    Low std  → values are clustered near the mean.

In ML, standard deviation is used in:
    - Feature normalization: subtract mean, divide by std → zero mean, unit variance
    - Weight initialization: Xavier/He init uses std to scale random weights
    - Gradient diagnostics: exploding/vanishing gradients show up as std extremes


### Sorting and Searching

    np.sort(a, axis)    Returns sorted copy (does not modify original)
    np.argsort(a, axis) Returns indices that would sort the array
    np.argmax(a, axis)  Index of maximum value (per axis or globally)
    np.argmin(a, axis)  Index of minimum value
    np.unique(a)        Sorted array of unique elements
    np.where(cond, x, y) Element-wise: pick x where condition is True, else y


### Images as NumPy Arrays — The ML Use Case

In computer vision, an image is just a 3-D array:

    shape: (height, width, channels)

    channels = 3 for RGB (Red, Green, Blue)
    channels = 4 for RGBA (with Alpha transparency)
    channels = 1 for Grayscale

Each pixel is a number (0–255 for uint8, or 0.0–1.0 for float32).
A batch of images for training is a 4-D array: (batch_size, height, width, channels).

This is why NumPy is the natural language of image ML pipelines — the math is just
arithmetic on large grids of numbers.


### Why All of This Matters for Machine Learning

Every core ML operation maps directly to NumPy:

    ML Operation                   NumPy Operation
    ──────────────────────────────────────────────────────────────────────────
    Linear layer forward pass      Y = X @ W + b
    Dot-product attention          scores = Q @ K.T / sqrt(d_k)
    Batch normalization            (x - x.mean()) / x.std()
    Loss function (MSE)            np.mean((predictions - targets) ** 2)
    Gradient update                W -= lr * gradient
    Data augmentation              array slicing + random transforms
    Feature extraction from image  image[y1:y2, x1:x2, :]

Mastering NumPy means you can read and reason about any of the above — whether it's
written as raw NumPy or wrapped in PyTorch/TensorFlow abstractions.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ──────────────────────────────────────────────────────────────────
    "01 · Importing NumPy": {
        "description": (
            "The universal convention for importing NumPy. Every ML codebase uses "
            "the alias `np`. You will see this in PyTorch, TensorFlow, and scikit-learn "
            "source code alike."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            # Confirm version
            print(f"NumPy version: {np.__version__}")

            # np is now the gateway to everything: np.array, np.zeros, np.dot, etc.
            # Using any other alias (e.g. `import numpy as numpy`) is non-standard
            # and will confuse readers of your code.
        ''',
    },

    # ── 2 ──────────────────────────────────────────────────────────────────
    "02 · ndarray Attributes": {
        "description": (
            "Every NumPy array carries metadata that describes its structure. "
            "These attributes are used constantly to validate shapes before operations, "
            "debug mismatches, and understand memory usage."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            # 1-D array — called a Vector
            a1 = np.array([1, 2, 3])

            # 2-D array — called a Matrix
            a2 = np.array([[1, 2.0, 3.3],
                           [4, 5,   6.5]])

            # 3-D array — called a Tensor
            a3 = np.array([[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]],
                           [[10, 11, 12],
                            [13, 14, 15],
                            [16, 17, 18]]])

            for name, arr in [("a1 (Vector)", a1), ("a2 (Matrix)", a2), ("a3 (Tensor)", a3)]:
                print(f"--- {name} ---")
                print(f"  .shape    = {arr.shape}       # dimensions tuple")
                print(f"  .ndim     = {arr.ndim}           # number of axes")
                print(f"  .dtype    = {arr.dtype}     # element type")
                print(f"  .size     = {arr.size}          # total elements")
                print(f"  .itemsize = {arr.itemsize} bytes   # bytes per element")
                print(f"  .nbytes   = {arr.nbytes} bytes   # total memory")
                print()

            # Key insight: a2 has mixed int/float literals → NumPy upcasts to float64
            # to safely represent every element. This is called type promotion.
        ''',
    },

    # ── 3 ──────────────────────────────────────────────────────────────────
    "03 · Creating Arrays — Constructors": {
        "description": (
            "NumPy provides a family of constructors for creating arrays without "
            "manually typing every value. These are the building blocks of data "
            "pipelines and weight initialization in neural networks."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            # --- From Python sequences ---
            from_list  = np.array([1, 2, 3])            # from list
            from_tuple = np.array((1, 2, 3))            # tuples also work

            # --- Filled arrays ---
            zeros  = np.zeros((3, 4))                   # (3,4) of 0.0 (float64)
            ones   = np.ones((2, 3))                    # (2,3) of 1.0 (float64)
            filled = np.full((2, 2), 7)                 # (2,2) of 7

            # --- Identity matrix ---
            eye = np.eye(3)                             # 3×3 identity

            # --- Ranges ---
            arange  = np.arange(0, 10, 2)              # [0, 2, 4, 6, 8]  (like range())
            linspace = np.linspace(0, 1, 5)            # [0, 0.25, 0.5, 0.75, 1.0]

            # --- Type casting ---
            as_int = ones.astype(int)                   # convert float → int

            for name, arr in [
                ("from_list", from_list), ("zeros(3,4)", zeros),
                ("ones(2,3)", ones), ("eye(3)", eye),
                ("arange(0,10,2)", arange), ("linspace(0,1,5)", linspace),
                ("ones as int", as_int),
            ]:
                print(f"{name:20s}  shape={str(arr.shape):10s}  dtype={arr.dtype}")

            print()
            print("zeros:\\n", zeros)
            print("eye:\\n", eye)
        ''',
    },

    # ── 4 ──────────────────────────────────────────────────────────────────
    "04 · Random Arrays and Reproducibility": {
        "description": (
            "Random arrays are essential for weight initialization, data shuffling, "
            "and simulation. NumPy uses a pseudo-random number generator — the sequence "
            "is deterministic given a seed, enabling reproducible experiments."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            # --- Random integers ---
            int_array = np.random.randint(low=0, high=10, size=(3, 4))
            print("randint(0, 10, (3,4)):\\n", int_array)

            # --- Random floats in [0, 1) ---
            float_array = np.random.random((3, 3))
            print("\\nrandom((3,3)):\\n", float_array)

            # --- Standard normal distribution (mean=0, std=1) ---
            normal = np.random.randn(3, 3)
            print("\\nrandn(3,3):\\n", normal)

            # --- Reproducibility with random seed ---
            print("\\n--- Without seed (changes every run) ---")
            print(np.random.randint(10, size=5))

            print("\\n--- With seed=42 (same every run) ---")
            np.random.seed(42)
            print(np.random.randint(10, size=5))
            np.random.seed(42)           # reset to same seed
            print(np.random.randint(10, size=5))   # identical output ✓

            # Why this matters: if you randomly split train/test data and don't fix
            # the seed, collaborators (or future-you) will get different splits.
            # np.random.seed() ensures "random but repeatable" experiments.
        ''',
    },

    # ── 5 ──────────────────────────────────────────────────────────────────
    "05 · Indexing and Slicing": {
        "description": (
            "Selecting subsets of arrays is fundamental to data pipelines — extracting "
            "a batch of samples, a single feature column, or a region of an image. "
            "NumPy's multi-dimensional slicing extends Python's slice syntax with a "
            "comma-separated index per axis."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            a1 = np.array([1, 2, 3])
            a2 = np.array([[1, 2.0, 3.3],
                           [4, 5,   6.5]])
            a3 = np.array([[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]],
                           [[10, 11, 12],
                            [13, 14, 15],
                            [16, 17, 18]]])

            print("a1[0]         =", a1[0])           # first element → scalar
            print("a2[0]         =", a2[0])           # first row of a2 → 1-D array
            print("a2[1]         =", a2[1])           # second row
            print("a2[1, 2]      =", a2[1, 2])        # row 1, col 2 → scalar 6.5
            print("a2[:, 1]      =", a2[:, 1])        # all rows, column 1 → [2., 5.]
            print("a3[0]         =\\n", a3[0])         # first depth slice → (3,3) matrix
            print("a3[:2, :2, :2]=\\n", a3[:2, :2, :2])  # sub-cube of first 2 in each dim

            # --- Negative indexing ---
            print("a1[-1]        =", a1[-1])          # last element = 3

            # --- Boolean (mask) indexing ---
            mask = a1 > 1
            print("a1 > 1        =", mask)            # [False  True  True]
            print("a1[a1 > 1]    =", a1[a1 > 1])     # [2  3]

            # Shape rule: indexing with an integer reduces the dimension by 1.
            # Indexing with a slice keeps that dimension.
            print("\\na2[0].shape    =", a2[0].shape)     # (3,) — row, dimension reduced
            print("a2[0:1].shape  =", a2[0:1].shape)   # (1,3) — slice, dimension kept
        ''',
    },

    # ── 6 ──────────────────────────────────────────────────────────────────
    "06 · Arithmetic Operations (Element-Wise)": {
        "description": (
            "All standard Python operators work element-wise on NumPy arrays. "
            "This is vectorization in action — no loops needed. The CPU applies "
            "each operation to all elements in parallel."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            a1 = np.array([1, 2, 3])
            ones = np.ones(3)

            print("a1          =", a1)
            print("ones        =", ones)
            print()
            print("a1 + ones   =", a1 + ones)   # [2. 3. 4.]
            print("a1 - ones   =", a1 - ones)   # [0. 1. 2.]
            print("a1 * ones   =", a1 * ones)   # [1. 2. 3.]  (element-wise, not dot)
            print("a1 / ones   =", a1 / ones)   # [1. 2. 3.]
            print("a1 // ones  =", a1 // ones)  # floor division → [1. 2. 3.]
            print("a1 ** 2     =", a1 ** 2)     # [1  4  9]
            print("a1 % 2      =", a1 % 2)      # modulus → [1 0 1]

            # Scalar broadcast — the scalar is "stretched" to match the array
            print()
            a2 = np.array([[1, 2, 3], [4, 5, 6]])
            print("a2 + 10    =\\n", a2 + 10)    # adds 10 to every element
            print("a2 * 0.5   =\\n", a2 * 0.5)  # halves every element

            # Math functions — also element-wise
            print()
            print("np.square(a1) =", np.square(a1))   # [1  4  9]
            print("np.sqrt(a1)   =", np.sqrt(a1))     # [1. 1.414 1.732]
            print("np.log(a1)    =", np.log(a1))      # natural log
            print("np.exp(a1)    =", np.exp(a1))      # e^1, e^2, e^3
        ''',
    },

    # ── 7 ──────────────────────────────────────────────────────────────────
    "07 · Broadcasting — Shape-Mismatched Operations": {
        "description": (
            "Broadcasting lets NumPy operate on arrays with different shapes without "
            "copying data. It's the mechanism behind adding a bias vector to a batch "
            "of activations in a neural network forward pass."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            a1 = np.array([1, 2, 3])         # shape (3,)
            a2 = np.array([[1, 2.0, 3.3],    # shape (2, 3)
                           [4, 5,   6.5]])

            # a1 (3,) broadcasts to (2, 3) by prepending a 1 → (1, 3) → (2, 3)
            print("a1 shape:", a1.shape)
            print("a2 shape:", a2.shape)
            print("a1 + a2 =\\n", a1 + a2)   # a1 is added to every row of a2

            # Scalar → broadcast to any shape
            print("\\na2 + 2 =\\n", a2 + 2)

            # Column vector → broadcast across columns
            col = np.array([[10], [20]])     # shape (2, 1)
            print("\\ncol shape:", col.shape)
            print("a2 + col =\\n", a2 + col)  # 10 added to row 0, 20 to row 1

            # Shape mismatch that CANNOT broadcast
            a3 = np.array([[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]],
                           [[10,11,12],
                            [13,14,15],
                            [16,17,18]]])   # shape (2, 3, 3)
            print("\\na2 shape:", a2.shape, "  a3 shape:", a3.shape)
            try:
                _ = a2 + a3
            except ValueError as e:
                print(f"Error: {e}")
                print("(2, 3) vs (2, 3, 3): trailing dimensions don't all match or equal 1")

            # ML use case: adding bias vector to a batch
            batch     = np.random.randn(32, 128)   # (batch=32, features=128)
            bias      = np.random.randn(128)        # (128,)
            activated = batch + bias                # broadcasts → (32, 128)
            print("\\nbatch + bias shape:", activated.shape)
        ''',
    },

    # ── 8 ──────────────────────────────────────────────────────────────────
    "08 · Aggregation Functions": {
        "description": (
            "Aggregation functions collapse one or more axes of an array into a "
            "summary statistic. The `axis` parameter controls which dimension is "
            "reduced — critical for computing per-sample or per-feature statistics "
            "in ML pipelines."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            a2 = np.array([[1, 2.0, 3.3],
                           [4, 5,   6.5]])
            print("a2 =\\n", a2)
            print("shape:", a2.shape, "  (2 rows, 3 cols)\\n")

            # --- Global aggregation (all elements) ---
            print("np.sum(a2)  =", np.sum(a2))     # 21.8
            print("np.mean(a2) =", np.mean(a2))    # 3.633
            print("np.max(a2)  =", np.max(a2))     # 6.5
            print("np.min(a2)  =", np.min(a2))     # 1.0
            print("np.std(a2)  =", np.std(a2))     # standard deviation
            print("np.var(a2)  =", np.var(a2))     # variance
            print()

            # --- Axis-specific aggregation ---
            # axis=0 → collapse rows → result has shape (cols,)
            print("np.sum(a2, axis=0) =", np.sum(a2, axis=0))   # sum per column
            # axis=1 → collapse cols → result has shape (rows,)
            print("np.sum(a2, axis=1) =", np.sum(a2, axis=1))   # sum per row
            print()

            # --- argmin / argmax — return indices ---
            print("np.argmax(a2)       =", np.argmax(a2))        # flat index of max
            print("np.argmax(a2,axis=1)=", np.argmax(a2, axis=1)) # max col per row

            # ML meaning: if a2 were a (batch=2, logits=3) array,
            # np.argmax(a2, axis=1) gives the predicted class for each sample.
            print("\\nPredicted class per sample:", np.argmax(a2, axis=1))
        ''',
    },

    # ── 9 ──────────────────────────────────────────────────────────────────
    "09 · Performance — np.sum() vs Python sum()": {
        "description": (
            "A concrete demonstration of why NumPy is fast. On large arrays, "
            "np.sum() is orders of magnitude faster than Python's built-in sum() "
            "because NumPy executes the loop in compiled C rather than interpreted Python."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import random
            import time

            N = 100_000

            # NumPy array
            np_array = np.random.random(N)

            # Equivalent Python list
            py_list = [random.random() for _ in range(N)]

            # --- Time np.sum() on ndarray ---
            t0 = time.perf_counter()
            for _ in range(100):
                np.sum(np_array)
            t_np_array = (time.perf_counter() - t0) / 100 * 1e6

            # --- Time Python sum() on ndarray ---
            t0 = time.perf_counter()
            for _ in range(100):
                sum(np_array)
            t_py_array = (time.perf_counter() - t0) / 100 * 1e6

            # --- Time Python sum() on list ---
            t0 = time.perf_counter()
            for _ in range(100):
                sum(py_list)
            t_py_list = (time.perf_counter() - t0) / 100 * 1e6

            # --- Time np.sum() on list ---
            t0 = time.perf_counter()
            for _ in range(100):
                np.sum(py_list)
            t_np_list = (time.perf_counter() - t0) / 100 * 1e6

            print(f"N = {N:,} elements")
            print()
            print(f"np.sum(ndarray)  : {t_np_array:8.1f} µs  ← use this")
            print(f"sum(ndarray)     : {t_py_array:8.1f} µs")
            print(f"sum(list)        : {t_py_list:8.1f} µs  ← use this for plain lists")
            print(f"np.sum(list)     : {t_np_list:8.1f} µs  (np must convert list first)")
            print()
            print("Rule of thumb:")
            print("  Use np.sum() on ndarrays.  Use sum() on Python lists.")
            print("  The speedup from np.sum() grows with array size.")
        ''',
    },

    # ── 10 ─────────────────────────────────────────────────────────────────
    "10 · Variance and Standard Deviation — Deep Dive": {
        "description": (
            "Variance and standard deviation quantify how spread out data is. "
            "They appear in feature normalization, loss diagnostics, and weight "
            "initialization. This operation builds intuition by contrasting "
            "high- and low-variance arrays with histograms."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # --- Two contrasting arrays ---
            high_var = np.array([1, 100, 200, 300, 4000, 5000])
            low_var  = np.array([2, 4, 6, 8, 10])

            print("High-variance array:", high_var)
            print(f"  mean = {np.mean(high_var):.2f}")
            print(f"  var  = {np.var(high_var):.2f}")
            print(f"  std  = {np.std(high_var):.2f}")
            print()
            print("Low-variance array:", low_var)
            print(f"  mean = {np.mean(low_var):.2f}")
            print(f"  var  = {np.var(low_var):.2f}")
            print(f"  std  = {np.std(low_var):.2f}")
            print()

            # Verify: std == sqrt(var)
            print("Verification: sqrt(var) == std?")
            print(f"  high_var: {np.sqrt(np.var(high_var)):.4f} == {np.std(high_var):.4f}")
            print()

            # --- Manual variance calculation (step by step) ---
            arr = high_var.astype(float)
            mu = arr.mean()
            deviations        = arr - mu
            squared_devs      = deviations ** 2
            variance_manual   = squared_devs.mean()
            print("Manual variance for high_var:")
            print(f"  μ               = {mu:.2f}")
            print(f"  deviations      = {deviations}")
            print(f"  squared devs    = {squared_devs}")
            print(f"  variance (mean of squared devs) = {variance_manual:.2f}")
            print(f"  np.var() result = {np.var(high_var):.2f}  ✓")
            print()

            # --- Histogram comparison ---
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(high_var, bins=10, color="steelblue", edgecolor="white")
            axes[0].set_title(f"High Variance  (std={np.std(high_var):.1f})")
            axes[0].set_xlabel("Value")
            axes[1].hist(low_var, bins=5, color="darkorange", edgecolor="white")
            axes[1].set_title(f"Low Variance   (std={np.std(low_var):.2f})")
            axes[1].set_xlabel("Value")
            plt.tight_layout()
            plt.savefig("variance_comparison.png", dpi=100)
            plt.show()
            print("Histogram saved to variance_comparison.png")
        ''',
    },

    # ── 11 ─────────────────────────────────────────────────────────────────
    "11 · Reshaping Arrays": {
        "description": (
            "Reshaping changes how the same block of memory is interpreted as "
            "a multi-dimensional structure. It is used constantly in ML to match "
            "the shape requirements of layers, loss functions, and operations."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            a2 = np.array([[1, 2.0, 3.3],
                           [4, 5,   6.5]])   # shape (2, 3)
            a3 = np.array([[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]],
                           [[10,11,12],
                            [13,14,15],
                            [16,17,18]]])    # shape (2, 3, 3)

            print("a2 shape:", a2.shape)

            # reshape: total elements must be conserved (2×3 = 6)
            r1 = a2.reshape(6)           # (2,3) → (6,)    flatten to 1-D
            r2 = a2.reshape(3, 2)        # (2,3) → (3,2)   different 2-D layout
            r3 = a2.reshape(2, 3, 1)     # (2,3) → (2,3,1) add trailing dim
            r4 = a2.reshape(1, -1)       # -1 means "infer this dimension"

            print("reshape(6)      :", r1.shape, r1)
            print("reshape(3,2)    :\\n", r2)
            print("reshape(2,3,1)  shape:", r3.shape)
            print("reshape(1,-1)   :", r4.shape, r4)

            # Use case: making a2 broadcastable with a3 (shape 2,3,3)
            print("\\na2 + a3 → shape mismatch:", a2.shape, "vs", a3.shape)
            try:
                _ = a2 + a3
            except ValueError as e:
                print("Error:", e)

            # After reshape, broadcasting rules align them
            result = a2.reshape(2, 3, 1) + a3    # (2,3,1) + (2,3,3) → (2,3,3)
            print("a2.reshape(2,3,1) + a3  shape:", result.shape)
            print(result)

            # flatten() and ravel()
            print("\\na2.flatten() =", a2.flatten())   # always returns a copy
            print("a2.ravel()   =", a2.ravel())       # returns a view if possible
        ''',
    },

    # ── 12 ─────────────────────────────────────────────────────────────────
    "12 · Transposing Arrays": {
        "description": (
            "Transposing reverses the order of array axes. For 2-D arrays this swaps "
            "rows and columns. In ML this is used to align dimensions before matrix "
            "multiplication — e.g. converting (features, samples) to (samples, features)."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            a2 = np.array([[1, 2.0, 3.3],
                           [4, 5,   6.5]])   # shape (2, 3)

            print("a2:\\n", a2)
            print("a2.shape     :", a2.shape)

            # .T reverses all axes
            print("\\na2.T:\\n", a2.T)
            print("a2.T.shape   :", a2.T.shape)    # (3, 2)

            # .transpose() is equivalent for 2-D
            print("a2.transpose():\\n", a2.transpose())

            # --- Higher-dimensional transpose ---
            matrix = np.random.randint(10, size=(5, 3, 3))
            print("\\nmatrix.shape :", matrix.shape)
            print("matrix.T.shape:", matrix.T.shape)   # (3, 3, 5) — first and last swapped

            # Verify: T reverses the shape tuple
            print("shape[::-1] ==", matrix.shape[::-1], "==", matrix.T.shape)

            # swapaxes — swap any two axes (more targeted than .T)
            print("\\nswapaxes(0,-1):", matrix.swapaxes(0, -1).shape)  # same as .T here

            # --- ML context: aligning for matrix multiply ---
            X = np.random.randn(64, 128)    # (samples=64, features=128)
            W = np.random.randn(10, 128)    # (outputs=10, features=128) — weight matrix
            print("\\nX shape:", X.shape, "  W shape:", W.shape)
            print("X @ W.T  →  shape:", (X @ W.T).shape)   # (64, 10) ✓
        ''',
    },

    # ── 13 ─────────────────────────────────────────────────────────────────
    "13 · Dot Product and Matrix Multiplication": {
        "description": (
            "The dot product is the most important operation in neural networks. "
            "Every linear (Dense) layer is a matrix multiplication. This operation "
            "covers the mechanics, shape rules, and the distinction between dot product "
            "and element-wise multiplication."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            np.random.seed(0)

            # --- Shape rules for matrix multiplication ---
            # (m, k) @ (k, n) → (m, n)
            # Inner dimensions must match; outer dimensions form the result shape.

            mat1 = np.random.randint(10, size=(3, 3))   # (3, 3)
            mat2 = np.random.randint(10, size=(3, 2))   # (3, 2)

            print("mat1 shape:", mat1.shape)
            print("mat2 shape:", mat2.shape)
            print("mat1 @ mat2 shape:", (mat1 @ mat2).shape)   # (3, 2)
            print()
            print("mat1:\\n", mat1)
            print("mat2:\\n", mat2)
            print("np.dot(mat1, mat2):\\n", np.dot(mat1, mat2))
            print("mat1 @ mat2 (same):\\n", mat1 @ mat2)

            # --- Shape mismatch example ---
            np.random.seed(0)
            mat3 = np.random.randint(10, size=(4, 3))
            mat4 = np.random.randint(10, size=(4, 3))
            print("\\nmat3:", mat3.shape, "  mat4:", mat4.shape)
            try:
                np.dot(mat3, mat4)
            except ValueError as e:
                print("np.dot(mat3, mat4) →", e)

            # Fix: transpose one so inner dims match → (3,4) @ (4,3) → (3,3)
            print("mat3.T @ mat4  shape:", (mat3.T @ mat4).shape)

            # --- Element-wise vs dot product ---
            print()
            print("mat3 * mat4 (element-wise / Hadamard):\\n", mat3 * mat4)
            print("shape:", (mat3 * mat4).shape, "  ← same shape as inputs")

            # --- 1-D dot product: scalar result ---
            v1 = np.array([1, 2, 3])
            v2 = np.array([4, 5, 6])
            print("\\nv1 · v2 =", np.dot(v1, v2), " = 1×4 + 2×5 + 3×6 = 32")
        ''',
    },

    # ── 14 ─────────────────────────────────────────────────────────────────
    "14 · Dot Product — Real-World Example (Nut Butter Sales)": {
        "description": (
            "A practical dot product problem: computing daily revenue from a matrix "
            "of units sold and a price vector. This is structurally identical to "
            "computing the output of a neural network's linear layer: Y = X @ W."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd

            np.random.seed(0)

            # sales_amounts: 5 days × 3 products (units sold per day)
            sales_amounts = np.random.randint(20, size=(5, 3))
            weekly_sales  = pd.DataFrame(
                sales_amounts,
                index=["Mon", "Tues", "Wed", "Thurs", "Fri"],
                columns=["Almond butter", "Peanut butter", "Cashew butter"]
            )
            print("Weekly sales (units):")
            print(weekly_sales)
            print("Shape:", sales_amounts.shape)  # (5, 3)

            # prices: price per jar for each product (1-D vector)
            prices = np.array([10, 8, 12])   # shape (3,)
            print("\\nPrices per jar:", prices, "  shape:", prices.shape)

            # Goal: daily revenue = Σ (price_i × units_i)  for each day
            # prices (3,) @ sales_amounts (5,3) → inner mismatch!
            print("\\nprices.shape:", prices.shape, "  sales.shape:", sales_amounts.shape)
            print("Direct dot → inner dims (3) vs (5) mismatch. Need to transpose sales.")

            # Transpose sales: (5,3) → (3,5), then prices (3,) @ (3,5) → (5,)
            total_sales = prices.dot(sales_amounts.T)
            print("\\nDaily revenue:", total_sales)

            weekly_sales["Total ($)"] = total_sales
            print()
            print("Weekly sales with totals:")
            print(weekly_sales)

            # --- Connection to neural networks ---
            print()
            print("=" * 55)
            print("Neural network analogy:")
            print("  sales_amounts ≈ input X         (5 samples, 3 features)")
            print("  prices        ≈ weight vector W  (3 weights)")
            print("  total_sales   ≈ output Y = X @ W (5 predictions)")
            print("  The math is IDENTICAL.")
        ''',
    },

    # ── 15 ─────────────────────────────────────────────────────────────────
    "15 · Comparison Operators and Boolean Masks": {
        "description": (
            "Comparison operators return boolean arrays that can be used as masks "
            "to filter or modify values. This underpins operations like thresholding, "
            "clipping activations, and counting how many predictions exceed a score."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            a1 = np.array([1, 2, 3])
            a2 = np.array([[1, 2.0, 3.3],
                           [4, 5,   6.5]])

            print("a1:", a1)
            print("a2:\\n", a2)
            print()

            # --- Comparisons between arrays (broadcast) ---
            print("a1 > a2  (a1 broadcast over a2):\\n", a1 > a2)
            print("a1 >= a2:\\n", a1 >= a2)
            print("a1 == a1:", a1 == a1)    # element-wise equality
            print("a1 == a2:\\n", a1 == a2)

            # --- Scalar comparison ---
            print("\\na1 > 1:", a1 > 1)         # [False  True  True]
            print("a2 > 3:\\n", a2 > 3)          # boolean matrix

            # --- Use boolean mask to filter values ---
            print("\\na2[a2 > 3]:", a2[a2 > 3])  # 1-D array of matching elements

            # --- Count matching elements ---
            print("Number of elements > 3:", np.sum(a2 > 3))  # True = 1, False = 0

            # --- np.where — conditional element selection ---
            result = np.where(a2 > 3, a2, 0)   # keep value if > 3, else 0 (ReLU-like)
            print("\\nnp.where(a2 > 3, a2, 0):\\n", result)

            # ML use case: ReLU activation
            activations = np.array([-1.5, 0.3, -0.2, 2.1, -0.8])
            relu_out    = np.where(activations > 0, activations, 0)
            print("\\nReLU demo:")
            print("  input :", activations)
            print("  output:", relu_out)
        ''',
    },

    # ── 16 ─────────────────────────────────────────────────────────────────
    "16 · Sorting Arrays": {
        "description": (
            "Sorting gives ordered arrays; argsort gives the indices that produce "
            "the sorted order. In ML, argmax/argmin are used at inference time to "
            "extract predicted class labels from logit vectors."
        ),
        "language": "python",
        "code": '''
            import numpy as np

            np.random.seed(42)
            random_array = np.random.randint(10, size=(3, 5))
            print("random_array:\\n", random_array)

            # --- np.sort — returns sorted copy (does not modify original) ---
            print("\\nnp.sort(random_array):\\n", np.sort(random_array))  # sort each row

            # --- np.argsort — indices that would produce sorted order ---
            a1 = np.array([5, 1, 3, 2])
            print("\\na1:", a1)
            print("np.argsort(a1):", np.argsort(a1))   # [1 3 2 0] → index of smallest first
            print("a1[np.argsort(a1)]:", a1[np.argsort(a1)])  # sorted a1

            # --- np.argmax / np.argmin — index of extreme value ---
            print("\\nnp.argmax(a1):", np.argmax(a1))   # 0 (index of 5)
            print("np.argmin(a1):", np.argmin(a1))   # 1 (index of 1)

            # --- axis argument ---
            print("\\nrandom_array:\\n", random_array)
            print("np.argmax(axis=0):", np.argmax(random_array, axis=0))  # max row per col
            print("np.argmax(axis=1):", np.argmax(random_array, axis=1))  # max col per row

            # --- np.unique ---
            a3 = np.array([[[1,2,3],[4,5,6],[7,8,9]],
                           [[10,11,12],[13,14,15],[16,17,18]]])
            print("\\nnp.unique(a3):", np.unique(a3))   # all unique values, sorted

            # --- ML context: class prediction ---
            logits = np.array([[2.1, 0.3, -1.2],    # sample 0: class 0
                               [0.1, 3.4,  0.8],    # sample 1: class 1
                               [-0.5, 0.2, 2.9]])   # sample 2: class 2
            preds  = np.argmax(logits, axis=1)
            print("\\nLogits:\\n", logits)
            print("Predicted classes:", preds)       # [0 1 2]
        ''',
    },

    # ── 17 ─────────────────────────────────────────────────────────────────
    "17 · Images as NumPy Arrays": {
        "description": (
            "Images are 3-D NumPy arrays of shape (height, width, channels). "
            "A batch of images is 4-D: (batch, height, width, channels). "
            "This is the representation used by every computer vision model. "
            "This operation generates a synthetic image array and demonstrates "
            "how pixel manipulation is just array arithmetic."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # --- Simulate a small RGB image ---
            np.random.seed(0)
            image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

            print("Image shape         :", image.shape)    # (height, width, channels)
            print("Image dtype         :", image.dtype)    # uint8  (0–255 per pixel)
            print("Total pixels        :", image.shape[0] * image.shape[1])
            print("Memory (bytes)      :", image.nbytes)

            # --- Accessing channels ---
            red_channel   = image[:, :, 0]    # shape (64, 64)
            green_channel = image[:, :, 1]
            blue_channel  = image[:, :, 2]
            print("\\nRed channel shape   :", red_channel.shape)
            print("Red channel mean    :", red_channel.mean())

            # --- Grayscale conversion (weighted average of channels) ---
            # Standard luminosity formula: 0.299R + 0.587G + 0.114B
            weights   = np.array([0.299, 0.587, 0.114])
            grayscale = (image @ weights).astype(np.uint8)   # (64,64,3) @ (3,) → (64,64)
            print("\\nGrayscale shape     :", grayscale.shape)

            # --- Region of interest (crop) ---
            crop = image[16:48, 16:48, :]   # centre 32×32 region
            print("Crop shape          :", crop.shape)

            # --- Batch of images ---
            batch = np.random.randint(0, 256, size=(8, 64, 64, 3), dtype=np.uint8)
            print("\\nBatch shape         :", batch.shape)   # (batch, H, W, C)
            print("Batch mean pixel val:", batch.mean())

            # --- Visualise ---
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            axes[0].imshow(image);          axes[0].set_title("RGB image")
            axes[1].imshow(red_channel,   cmap="Reds");   axes[1].set_title("Red channel")
            axes[2].imshow(grayscale,     cmap="gray");   axes[2].set_title("Grayscale")
            axes[3].imshow(crop);           axes[3].set_title("Cropped region")
            for ax in axes: ax.axis("off")
            plt.tight_layout()
            plt.savefig("numpy_image_demo.png", dpi=100)
            plt.show()
            print("\\nSaved numpy_image_demo.png")
        ''',
    },

    # ── 18 ─────────────────────────────────────────────────────────────────
    "18 · NumPy as the Backbone of pandas": {
        "description": (
            "pandas DataFrames are built on top of NumPy arrays. Any time you "
            "need raw numerical speed or want to pass data into a scikit-learn or "
            "PyTorch model, you convert a DataFrame to a NumPy array. This operation "
            "shows how the two libraries interoperate."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd

            # --- Creating a DataFrame from a NumPy array ---
            np.random.seed(0)
            data_array = np.random.randint(10, size=(5, 3))
            df         = pd.DataFrame(data_array, columns=["a", "b", "c"])
            print("NumPy array:\\n", data_array)
            print("\\npandas DataFrame:\\n", df)

            # --- The DataFrame stores the data as NumPy arrays under the hood ---
            print("\\ndf.values type   :", type(df.values))       # numpy.ndarray
            print("df.values shape  :", df.values.shape)
            print("df['a'].values   :", df["a"].values)          # 1-D ndarray

            # --- Round-trip: ndarray → DataFrame → ndarray ---
            a2 = np.array([[1, 2.0, 3.3], [4, 5, 6.5]])
            df2 = pd.DataFrame(a2, columns=["x", "y", "z"])
            back_to_np = df2.to_numpy()                          # or df2.values
            print("\\ndf2:\\n", df2)
            print("df2.to_numpy():", back_to_np)

            # --- ML pipeline pattern ---
            # (typical: load data with pandas, extract features as ndarray, feed to model)
            np.random.seed(42)
            n_samples = 100
            df_ml = pd.DataFrame({
                "height_cm" : np.random.normal(170, 10, n_samples),
                "weight_kg" : np.random.normal(70,  15, n_samples),
                "label"     : np.random.randint(0, 2, n_samples),
            })

            # Feature matrix X and label vector y as numpy arrays
            X = df_ml[["height_cm", "weight_kg"]].to_numpy()
            y = df_ml["label"].to_numpy()
            print("\\nFeature matrix X shape:", X.shape)    # (100, 2)
            print("Label vector   y shape:", y.shape)      # (100,)
            print("X[:3]:\\n", X[:3])
        ''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings
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
    visual_html   = ""
    visual_height = 600

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None,
        "operations":    OPERATIONS,
    }