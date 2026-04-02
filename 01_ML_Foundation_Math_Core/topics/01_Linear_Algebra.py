"""
Linear Algebra
==============

The language of machine learning. Every neural network forward pass is
a chain of matrix multiplications. Every dimensionality reduction is a
change of basis. Every optimisation step lives in a vector space.
This module builds the complete geometric and algebraic intuition —
from vectors to the Singular Value Decomposition.

"""

import textwrap
import re

TOPIC_NAME   = "Linear Algebra"
DISPLAY_NAME = "01 · Linear Algebra"
ICON         = "🔢"
SUBTITLE     = "Vectors, Matrices, Decompositions — The Language of ML"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### PART 1 — VECTORS & VECTOR SPACES


### What Is a Vector?

Three equivalent views — all are correct and all are useful:

    1. An arrow in space:   has magnitude and direction
    2. An ordered list:     v = [v₁, v₂, ..., vₙ] ∈ ℝⁿ
    3. A function:          a linear map from ℝ to ℝⁿ

In ML, the "ordered list" view dominates. A data point with n features
is a vector in ℝⁿ. A set of weights in a neural network layer is a vector.
The gradient of a loss function is a vector.

Column vector (default convention):
         ⎡v₁⎤
    v  = ⎢v₂⎥  ∈ ℝⁿ
         ⎣vₙ⎦

Row vector: vᵀ = [v₁, v₂, ..., vₙ] — the transpose of a column vector.


### Vector Operations

**Addition:**      u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]
**Scalar mult:**   αv   = [αv₁, αv₂, ..., αvₙ]
**Dot product:**   u·v  = uᵀv = Σᵢ uᵢvᵢ  (a scalar)
**Cross product:** u×v  (only in ℝ³) — vector perpendicular to both

Geometric interpretation of the dot product:

    u · v = ‖u‖ · ‖v‖ · cos θ

    where θ is the angle between u and v.
    u · v > 0:  angle < 90°  (same general direction)
    u · v = 0:  angle = 90°  (orthogonal — perpendicular)
    u · v < 0:  angle > 90°  (opposite general direction)


### Vector Spaces

A vector space V over ℝ is a set with two operations (addition and scalar
multiplication) satisfying 8 axioms. The key ones:

    Closure:        u, v ∈ V → u+v ∈ V  and  αv ∈ V
    Zero vector:    ∃ 0 ∈ V such that v + 0 = v for all v
    Inverse:        ∀ v ∈ V, ∃ -v such that v + (-v) = 0
    Distributivity: α(u+v) = αu + αv

Examples:
    ℝⁿ — the standard n-dimensional real vector space
    Polynomials of degree ≤ k — a vector space over ℝ
    Continuous functions on [0,1] — an infinite-dimensional vector space
    Solutions to Ax = 0 — a vector space (the null space of A)


### Subspaces

A subset S ⊆ V is a SUBSPACE if it is itself a vector space:
    1. 0 ∈ S
    2. u, v ∈ S → u+v ∈ S    (closed under addition)
    3. v ∈ S, α ∈ ℝ → αv ∈ S (closed under scalar multiplication)

    Example: the xy-plane in ℝ³ (all vectors with z=0) is a subspace.
    Example: any line through the origin is a subspace.
    Non-example: a line NOT through the origin (fails condition 1).

    Key insight: the solution sets of homogeneous linear systems Ax=0
    are always subspaces. Non-homogeneous systems Ax=b (b≠0) produce
    AFFINE subspaces (shifted subspaces), not true subspaces.


### Span, Linear Independence, Basis, Dimension

**Span:**  span{v₁,...,vₖ} = {α₁v₁ + ... + αₖvₖ : αᵢ ∈ ℝ}
    The set of ALL linear combinations. Always a subspace.

**Linear independence:**  {v₁,...,vₖ} is linearly independent iff
    α₁v₁ + ... + αₖvₖ = 0  implies  α₁=...=αₖ=0
    No vector in the set is a linear combination of the others.

**Basis:**  A linearly independent set that spans V.
    The "minimal spanning set" — remove any vector and you lose coverage.
    The "maximal independent set" — add any vector and it becomes dependent.

**Dimension:**  dim(V) = number of vectors in any basis of V.
    All bases of V have the same size — this is well-defined.

    Standard basis of ℝⁿ:
        e₁ = [1,0,...,0]ᵀ,  e₂ = [0,1,...,0]ᵀ,  ...,  eₙ = [0,0,...,1]ᵀ
    These are the columns of the n×n identity matrix I.

    Diagram 1 — Basis Intuition in ℝ²:

    ↑ v₂                ↑ e₂
    │  ╱                │
    │ ╱  (non-standard  │        (standard
    │╱   basis)         │         basis)
    └──────→ v₁         └──────→ e₁

    Any vector in ℝ² can be written uniquely as α₁v₁ + α₂v₂ (left)
    or uniquely as x₁e₁ + x₂e₂ (right). Both are valid bases.


### PART 2 — NORMS, DISTANCES & INNER PRODUCTS

### Norms

A norm ‖·‖ on ℝⁿ assigns a non-negative "size" to each vector:
    Non-negativity:   ‖v‖ ≥ 0,  ‖v‖ = 0 iff v = 0
    Homogeneity:      ‖αv‖ = |α| · ‖v‖
    Triangle ineq.:   ‖u + v‖ ≤ ‖u‖ + ‖v‖

**ℓ₁ norm (Manhattan / Taxicab):**

    ‖v‖₁ = Σᵢ |vᵢ|

    Geometry: the "city block" distance. The unit ball is a diamond (◇).
    ML use: L1 regularisation (Lasso), MAE loss, sparse solutions.
    Key property: promotes SPARSITY — the ℓ₁ ball has corners at the
    coordinate axes, where the optimal solution touches the constraint set.

**ℓ₂ norm (Euclidean):**

    ‖v‖₂ = √(Σᵢ vᵢ²) = √(vᵀv)

    Geometry: standard Euclidean length. The unit ball is a sphere (○).
    ML use: L2 regularisation (Ridge), MSE loss, Euclidean distance.
    Key property: smooth, differentiable everywhere, unique minimum.

**ℓ∞ norm (Chebyshev / max norm):**

    ‖v‖∞ = maxᵢ |vᵢ|

    Geometry: the unit ball is a hypercube (□).
    ML use: adversarial robustness (ε-ball attacks bounded by ℓ∞).

**ℓₚ norm (general):**

    ‖v‖ₚ = (Σᵢ |vᵢ|ᵖ)^(1/p)     for p ≥ 1

    p→1: approaches ℓ₁;  p=2: Euclidean;  p→∞: approaches ℓ∞

    Diagram 2 — Unit Balls of Different Norms in ℝ²:

    ℓ₁ (diamond)    ℓ₂ (circle)    ℓ∞ (square)
         ◇               ○               □

    Sparsity intuition: the corners of the ℓ₁ diamond lie on coordinate
    axes. When you minimise a loss function subject to ‖w‖₁ ≤ t,
    the solution tends to hit a corner → many components = 0 (sparse).
    The smooth ℓ₂ ball has no corners, so solutions are rarely exactly 0.


### Inner Products

An inner product ⟨·,·⟩ on a vector space V is a function V×V → ℝ:
    Linearity:      ⟨αu+βv, w⟩ = α⟨u,w⟩ + β⟨v,w⟩
    Symmetry:       ⟨u, v⟩ = ⟨v, u⟩
    Positive def.:  ⟨v, v⟩ ≥ 0,  ⟨v,v⟩=0 iff v=0

The standard inner product on ℝⁿ is the dot product: ⟨u,v⟩ = uᵀv.

Every inner product induces a norm: ‖v‖ = √⟨v,v⟩.
The Cauchy-Schwarz inequality then follows:

    |⟨u, v⟩| ≤ ‖u‖ · ‖v‖

    with equality iff u and v are collinear (one is a scalar multiple of the other).

Pearson correlation is just the normalised dot product:

    cos θ = ⟨u,v⟩ / (‖u‖·‖v‖) = uᵀv / (‖u‖·‖v‖) ∈ [−1, 1]


### Matrix Norms

**Frobenius norm:**

    ‖A‖_F = √(Σᵢⱼ aᵢⱼ²) = √(tr(AᵀA)) = √(Σᵢ σᵢ²)

    where σᵢ are the singular values of A.
    The "entry-wise ℓ₂ norm" — treats A as a long vector.

**Spectral norm (ℓ₂ operator norm):**

    ‖A‖₂ = σ_max(A)   (largest singular value)

    The maximum factor by which A can stretch a vector:
    ‖A‖₂ = max_{‖v‖=1} ‖Av‖

    ML use: Lipschitz constant of a linear layer, spectral normalisation
    for GAN training stability.

**Nuclear norm:**

    ‖A‖_* = Σᵢ σᵢ    (sum of all singular values)

    ML use: matrix completion, convex surrogate for matrix rank
    minimisation (analogous to ℓ₁ as a surrogate for ℓ₀ sparsity).


### PART 3 — MATRICES & LINEAR TRANSFORMATIONS


### The Fundamental View: Matrices ARE Linear Maps

A matrix A ∈ ℝ^{m×n} is not just a table of numbers. It defines a
linear transformation T: ℝⁿ → ℝᵐ  by  T(x) = Ax.

Linear means:  T(αu + βv) = αT(u) + βT(v)

Equivalently, a matrix is completely determined by what it does to
the standard basis vectors e₁,...,eₙ:

    Column j of A = A·eⱼ = T(eⱼ)

This is the COLUMN PICTURE: Ax is a linear combination of the columns of A,
with coefficients given by x:

    ⎡a₁₁ a₁₂⎤ ⎡x₁⎤         ⎡a₁₁⎤        ⎡a₁₂⎤
    ⎣a₂₁ a₂₂⎦ ⎣x₂⎦  = x₁ · ⎣a₂₁⎦  + x₂ · ⎣a₂₂⎦

The ROW PICTURE: each row of Ax is a dot product of a row of A with x.

Both pictures are always simultaneously true. Choosing which to use
depends on what you are trying to understand.


### Geometric Effect of Common Linear Transformations (2D)

    Scaling:      ⎡s  0⎤  stretches x-axis by s, y-axis by t
                  ⎣0  t⎦

    Rotation:     ⎡cos θ  -sin θ⎤  rotates by angle θ
                  ⎣sin θ   cos θ⎦

    Shear:        ⎡1  k⎤  slides horizontal lines, preserving area
                  ⎣0  1⎦

    Reflection:   ⎡1  0⎤  reflects across the x-axis
                  ⎣0 -1⎦

    Projection:   ⎡1  0⎤  projects onto the x-axis (collapses y)
                  ⎣0  0⎦   rank=1, not invertible, loses info

    Diagram 3 — What Rotation Does to the Unit Square:

    BEFORE            AFTER θ=45°
    ┌──┐              ╱╲
    │  │     →      ╱    ╲
    └──┘            ╲    ╱
                      ╲╱
    Corners [1,0],[0,1] rotate to [cos45°,sin45°],[−sin45°,cos45°]


### Matrix Multiplication

(AB)ᵢⱼ = (row i of A) · (col j of B) = Σₖ Aᵢₖ Bₖⱼ

This is COMPOSITION of linear transformations: apply B first, then A.

    AB ≠ BA in general  (matrix multiplication is NOT commutative)
    (AB)C = A(BC)       (it IS associative)
    A(B+C) = AB + AC    (distributive)

Four ways to compute AB (all equivalent, each reveals different structure):

    1. Row × column:   standard dot product definition above
    2. Matrix × cols:  each column of AB = A × (column of B)
    3. Rows × matrix:  each row of AB = (row of A) × B
    4. Sum of rank-1:  AB = Σₖ (col k of A)(row k of B)^T

    The rank-1 decomposition (4) is the deepest view — it shows that
    matrix multiplication is a sum of outer products.


### Special Matrices

**Symmetric:**       A = Aᵀ          (aᵢⱼ = aⱼᵢ)
    All eigenvalues are real. Has an orthogonal eigenbasis.
    Examples: covariance matrices, graph Laplacians, Hessians of smooth functions.

**Orthogonal:**      QᵀQ = QQᵀ = I   (equivalently, Qᵀ = Q⁻¹)
    Columns are orthonormal: Qᵢ·Qⱼ = δᵢⱼ.
    Preserves lengths and angles: ‖Qv‖ = ‖v‖, ⟨Qu, Qv⟩ = ⟨u,v⟩.
    det(Q) = ±1. Rotations (det=+1) and reflections (det=−1).
    Examples: rotation matrices, Householder reflectors, DFT matrix/√n.

**Positive Definite (PD):**  vᵀAv > 0 for all v ≠ 0
    Equivalently: all eigenvalues > 0.
    Equivalently: A = BᵀB for some invertible B (Cholesky).
    Examples: covariance matrices, Gram matrices XᵀX (when X full rank),
              Hessians at strict local minima.

    Positive semidefinite (PSD): vᵀAv ≥ 0, eigenvalues ≥ 0.
    XᵀX is always PSD (may have zero eigenvalues when X not full rank).

**Diagonal:**        aᵢⱼ = 0 for i≠j
    Easy to invert: D⁻¹ has diagonal entries 1/dᵢᵢ.
    Eigenvalues are the diagonal entries.
    Powers: Dᵏ has diagonal entries dᵢᵢᵏ.

**Triangular:**      Upper: aᵢⱼ = 0 for i > j. Lower: aᵢⱼ = 0 for i < j.
    Determinant = product of diagonal entries.
    Systems Lx=b and Ux=b solvable by back/forward substitution in O(n²).

**Idempotent (Projection):**  P² = P
    Eigenvalues are 0 or 1. Range and null space are complementary.
    P = A(AᵀA)⁻¹Aᵀ projects onto the column space of A.


### Transpose, Trace, and Inverse

**Transpose:**  (Aᵀ)ᵢⱼ = Aⱼᵢ
    (AB)ᵀ = BᵀAᵀ   (reverses order)
    (Aᵀ)ᵀ = A
    (A+B)ᵀ = Aᵀ + Bᵀ

**Trace:**  tr(A) = Σᵢ Aᵢᵢ   (sum of diagonal entries)
    tr(AB) = tr(BA)          (cyclic property — powerful trick!)
    tr(A) = Σᵢ λᵢ           (sum of eigenvalues)
    tr(AᵀA) = Σᵢⱼ Aᵢⱼ² = ‖A‖_F²

**Inverse:**  A⁻¹ exists iff A is square AND det(A) ≠ 0 (full rank).
    AA⁻¹ = A⁻¹A = I
    (AB)⁻¹ = B⁻¹A⁻¹    (reverses order)
    (Aᵀ)⁻¹ = (A⁻¹)ᵀ

    Computing A⁻¹ via Gaussian elimination: O(n³) operations.
    Never compute A⁻¹ to solve Ax=b — use factorisation instead.
    The condition number κ(A) = ‖A‖·‖A⁻¹‖ = σ_max/σ_min measures
    how sensitive A⁻¹ is to numerical errors (high κ = ill-conditioned).


### PART 4 — THE FOUR FUNDAMENTAL SUBSPACES

### Gilbert Strang's "Big Picture" of Linear Algebra

For a matrix A ∈ ℝ^{m×n}, there are four fundamental subspaces that
together describe EVERYTHING about the linear map x ↦ Ax:

    ┌──────────────────────────────────────────────────────────────────┐
    │              A : ℝⁿ ──────────────────────→ ℝᵐ                    │
    │                                                                  │
    │  In ℝⁿ (domain):             In ℝᵐ (codomain):                    │
    │  ┌─────────────────┐         ┌─────────────────┐                 │
    │  │  Row space C(Aᵀ)│ ──A──→ │  Column space    │                 │
    │  │  dim = r        │         │  C(A)  dim = r  │                 │
    │  ├─────────────────┤         ├─────────────────┤                 │
    │  │  Null space N(A)│ ──A──→ │  Left null space │                 │
    │  │  dim = n−r      │   0    │  N(Aᵀ) dim = m−r │                 │
    │  └─────────────────┘         └─────────────────┘                 │
    │  (orthogonal complement)     (orthogonal complement)             │
    └──────────────────────────────────────────────────────────────────┘

    r = rank(A)  (number of independent rows = number of independent columns)


### Column Space C(A)

    C(A) = {Ax : x ∈ ℝⁿ} = span of the columns of A

    The set of all RIGHT-HAND SIDES b for which Ax = b HAS a solution.
    dim(C(A)) = rank(A) = r.

    Geometric meaning: C(A) is the image (range) of the transformation A.
    A maps all of ℝⁿ into the column space — it cannot reach anything outside.


### Null Space N(A) — Kernel

    N(A) = {x ∈ ℝⁿ : Ax = 0}

    All vectors that A maps to ZERO. Always a subspace (contains 0).
    dim(N(A)) = n − r.

    Why it matters: if N(A) ≠ {0}, then A is not injective (not one-to-one).
    Multiple inputs map to the same output — the transformation is lossy.
    In the system Ax = b: if x₀ is ONE solution, then x₀ + n is also a
    solution for EVERY n ∈ N(A). The complete solution set is:
        x = x_particular + N(A)  (an affine subspace)


### Row Space C(Aᵀ)

    C(Aᵀ) = span of the rows of A  ⊆ ℝⁿ

    The orthogonal complement of the null space:
        C(Aᵀ) ⊥ N(A)    and    C(Aᵀ) ⊕ N(A) = ℝⁿ

    Every vector x ∈ ℝⁿ splits uniquely as:
        x = x_row + x_null,  where x_row ∈ C(Aᵀ) and x_null ∈ N(A)

    A maps x_row isomorphically onto C(A), and maps x_null to 0.
    The "invertible part" of A lives on the row space.


### Left Null Space N(Aᵀ)

    N(Aᵀ) = {y ∈ ℝᵐ : Aᵀy = 0} = {y : yᵀA = 0}

    The orthogonal complement of the column space:
        C(A) ⊥ N(Aᵀ)    and    C(A) ⊕ N(Aᵀ) = ℝᵐ

    When b ∉ C(A), the system Ax = b has NO solution.
    The "obstruction" to solvability lives in N(Aᵀ).
    If yᵀA = 0 but yᵀb ≠ 0, then Ax = b is inconsistent
    (and least squares gives the best approximate solution).


### The Rank-Nullity Theorem

    rank(A) + nullity(A) = n

    dim(C(A)) + dim(N(A)) = n    (number of columns)
    dim(C(Aᵀ)) + dim(N(Aᵀ)) = m (number of rows)
    rank(A) = rank(Aᵀ)           (column rank = row rank — always!)

    For a square n×n matrix A:
        A is invertible iff rank(A) = n iff N(A) = {0}
        iff all eigenvalues ≠ 0 iff det(A) ≠ 0


### PART 5 — DETERMINANTS & SYSTEMS OF EQUATIONS

### Determinant — Geometric Meaning

The determinant det(A) of a square n×n matrix is a scalar that
measures the SIGNED VOLUME scaling factor of the transformation A:

    The image of the unit n-cube under A has volume |det(A)|.
    Positive det: orientation preserved (no reflection).
    Negative det: orientation reversed (reflection).
    det(A) = 0:  the transformation collapses at least one dimension.
                 Volume → 0. Matrix is singular (not invertible).

    For 2×2:  det ⎡a b⎤ = ad − bc
                  ⎣c d⎦

    Geometric: |ad−bc| is the area of the parallelogram spanned by
    the two column vectors [a,c]ᵀ and [b,d]ᵀ.

    For 3×3:  det is the triple product — volume of parallelepiped.


### Properties of Determinants

    det(AB) = det(A) · det(B)        (multiplicative)
    det(Aᵀ) = det(A)
    det(A⁻¹) = 1/det(A)
    det(cA) = cⁿ det(A)             (n×n matrix, scalar c)
    det(A) = Π λᵢ                   (product of eigenvalues)

Row operation effects on det:
    Swap two rows:          det flips sign
    Multiply row by c:      det multiplies by c
    Add multiple of one row to another: det unchanged

These properties make Gaussian elimination useful for computing det:
    Reduce A to upper triangular U via row ops, tracking sign flips.
    det(A) = (−1)^(swaps) × Π diagonal entries of U


### Solving Linear Systems: Ax = b

    CASE 1 — Unique solution:   A is n×n, det(A) ≠ 0
        x = A⁻¹b  (conceptually)
        In practice: Gaussian elimination (LU factorisation), O(n³).

    CASE 2 — No solution (overdetermined, inconsistent):
        m > n  or  b ∉ C(A).
        Best we can do: LEAST SQUARES solution:
            min_x ‖Ax − b‖₂²  →  x̂ = (AᵀA)⁻¹Aᵀb
        This is the orthogonal projection of b onto C(A).

    CASE 3 — Infinitely many solutions (underdetermined):
        m < n  or  rank(A) < n.
        Minimum-norm least squares: x̂ = A⁺b  (Moore-Penrose pseudoinverse)
        This gives the solution in the ROW SPACE of A — the one
        with smallest ‖x‖₂ among all solutions.


### Gaussian Elimination and LU Factorisation

Gaussian elimination reduces A to row echelon form by elementary row
operations. This produces the LU factorisation:

    A = LU

    L: lower triangular with 1s on diagonal (stores the multipliers)
    U: upper triangular (the row echelon form)

    Solving Ax = b via LU:
        1. Ly = b   (forward substitution, O(n²))
        2. Ux = y   (back substitution, O(n²))
        Total: O(n³) for the factorisation, O(n²) per right-hand side.

With partial pivoting (LAPACK standard):
    PA = LU,  P permutation matrix (reorders rows for numerical stability)

The condition number κ(A) = σ_max / σ_min measures how amplified errors
become. κ >> 1 means the system is ill-conditioned and numerical solutions
may be wildly inaccurate.


### PART 6 — EIGENVALUES & EIGENVECTORS

### The Eigenvalue Equation

    Av = λv,    v ≠ 0

    v is an EIGENVECTOR of A, λ is the corresponding EIGENVALUE.
    A eigenvector is a direction that A does NOT rotate — only stretches
    (by factor λ) or reflects (if λ < 0).

Finding eigenvalues: solve the CHARACTERISTIC POLYNOMIAL:

    det(A − λI) = 0

    This is a degree-n polynomial in λ. Its n roots (possibly complex,
    possibly repeated) are the eigenvalues.

    For 2×2:  det ⎡a−λ  b ⎤ = (a−λ)(d−λ) − bc = 0
                  ⎣ c  d−λ⎦

    Eigenspace for λ:  E_λ = N(A − λI)  (null space of A−λI)
    The eigenspace may be multi-dimensional (geometric multiplicity > 1).


### Properties of Eigenvalues

    tr(A) = Σᵢ λᵢ   (trace = sum of eigenvalues)
    det(A) = Πᵢ λᵢ  (determinant = product of eigenvalues)
    det(A) = 0 iff at least one eigenvalue is 0 (A is singular)

    Similar matrices have the same eigenvalues:
        B = P⁻¹AP  →  B and A have the same characteristic polynomial.

    Eigenvalues of Aᵏ: λᵢᵏ    (eigenvectors unchanged)
    Eigenvalues of A⁻¹: 1/λᵢ  (eigenvectors unchanged)
    Eigenvalues of A+cI: λᵢ+c  (eigenvectors unchanged) — "shift"

    Diagram 4 — Geometric Interpretation:

    Before A:    After A (λ₁=2):   After A (λ₂=0.5):
    ───v₁──→    ───────v₁───────→  →v₂
    ──v₂──→                        (squashed)
    Eigenvectors only stretch/shrink — never rotate.


### Diagonalisation

A matrix A is diagonalisable if it has n linearly independent eigenvectors:

    A = PDP⁻¹

    P: matrix of eigenvectors as columns [v₁ | v₂ | ... | vₙ]
    D: diagonal matrix of eigenvalues diag(λ₁,...,λₙ)

    Powers become trivial:  Aᵏ = PDᵏP⁻¹  (just raise diagonal entries to k-th power)
    Matrix exponential:     eᴬ = PeᴰP⁻¹  (eᴰ has diagonal entries eˡᵢ)

    Sufficient condition: n distinct eigenvalues → always diagonalisable.
    SYMMETRIC matrices: A = Aᵀ → always diagonalisable with ORTHOGONAL P.

    When A is NOT diagonalisable (defective): Jordan normal form
    exists but eigenvectors don't span ℝⁿ. Repeated eigenvalues without
    enough eigenvectors. Rare in practice but important theoretically.


### Spectral Theorem (Symmetric Matrices)

For A ∈ ℝ^{n×n} with A = Aᵀ:

    1. ALL eigenvalues are real
    2. Eigenvectors for DISTINCT eigenvalues are orthogonal
    3. A has an ORTHONORMAL basis of eigenvectors
    4. A = QΛQᵀ  where Q is orthogonal, Λ = diag(λ₁,...,λₙ)

This is the eigendecomposition for symmetric matrices. The columns of Q
are the principal directions; eigenvalues tell you how much A stretches
in each direction.

Written as a sum of rank-1 matrices (spectral decomposition):

    A = Σᵢ λᵢ qᵢqᵢᵀ

Each term λᵢ qᵢqᵢᵀ is a rank-1 projection onto eigenvector qᵢ.
The full matrix is the sum of n "pure-direction" components.


### Eigenvalues in ML

    Covariance matrix Σ:
        Symmetric, PSD. Eigenvalues = variances along principal axes.
        PCA finds eigenvectors (principal components) sorted by eigenvalue.

    Graph Laplacian L:
        Symmetric, PSD. Zero is always an eigenvalue (constant vector).
        Spectral clustering uses the Fiedler vector (2nd smallest eigenvalue).

    Hessian H at a critical point:
        All λᵢ > 0: local minimum (positive definite)
        All λᵢ < 0: local maximum (negative definite)
        Mixed signs: saddle point
        λᵢ = 0:     degenerate (second-order test fails)

    Iteration convergence rate:
        Power iteration converges at rate |λ₁/λ₂| (ratio of largest eigenvalues).
        Gradient descent on quadratic f: converges at rate (κ−1)/(κ+1)
        where κ = λ_max/λ_min is the condition number.


### PART 7 — ORTHOGONALITY & PROJECTIONS

### Orthogonality

u ⊥ v means uᵀv = 0. The Pythagorean theorem generalises:

    u ⊥ v  →  ‖u + v‖² = ‖u‖² + ‖v‖²

Orthogonal set: every pair of vectors is orthogonal.
Orthonormal set: orthogonal AND each vector has unit length (‖vᵢ‖=1).

An orthonormal basis is the "nicest" possible basis:
    - Coordinates in an ONB are just dot products: xᵢ = qᵢᵀx
    - Change of basis to ONB: no matrix inversion needed (Q⁻¹ = Qᵀ)


### Projection onto a Vector

The projection of b onto a direction a:

    proj_a(b) = (aᵀb / aᵀa) · a = (aᵀb / ‖a‖²) · a

    The scalar aᵀb/‖a‖ is the COMPONENT of b along a.
    The vector proj_a(b) is the "shadow" of b onto the line spanned by a.
    The residual e = b − proj_a(b) is orthogonal to a: aᵀe = 0.


### Projection onto a Subspace

The projection of b onto the column space of A ∈ ℝ^{m×n}:

    p = A(AᵀA)⁻¹Aᵀb

    Projection matrix:  P = A(AᵀA)⁻¹Aᵀ
    Properties of P:
        P² = P          (idempotent — projecting twice = projecting once)
        Pᵀ = P          (symmetric)
        Eigenvalues: 0 (for N(Aᵀ)) and 1 (for C(A))
        p is the closest point in C(A) to b: minimises ‖p − b‖₂

    Error vector:  e = b − p = (I − P)b  ⊥ C(A)  (error is in N(Aᵀ))

    This is LEAST SQUARES: Aᵀe = 0 → Aᵀ(b−Ax̂) = 0 → AᵀAx̂ = Aᵀb
    The NORMAL EQUATIONS. Solution x̂ = (AᵀA)⁻¹Aᵀb minimises ‖Ax−b‖₂².

    When A has orthonormal columns (Q):
        P = QQᵀ   (much simpler! No matrix inverse needed)
        x̂ = Qᵀb  (just dot products with each column)


### Gram-Schmidt Orthogonalisation

Given any basis {a₁,...,aₙ}, produce an orthonormal basis {q₁,...,qₙ}:

    Step 1: q₁ = a₁ / ‖a₁‖

    Step 2: q₂ = (a₂ − (q₁ᵀa₂)q₁) / ‖a₂ − (q₁ᵀa₂)q₁‖
            (subtract projection onto q₁, then normalise)

    Step k: subtract projections onto q₁,...,qₖ₋₁, then normalise.

    The projections being subtracted are:  (q₁ᵀaₖ)q₁ + ... + (qₖ₋₁ᵀaₖ)qₖ₋₁

This produces the QR factorisation:

    A = QR

    Q: m×n matrix with orthonormal columns  (Q^T Q = I)
    R: n×n upper triangular matrix with positive diagonal
       R_{ij} = qᵢᵀaⱼ for i≤j  (stores the Gram-Schmidt coefficients)



### PART 8 — MATRIX DECOMPOSITIONS


### Why Decompositions?

A decomposition rewrites A as a PRODUCT of structured, simple matrices.
Each factorisation reveals different structure, enables efficient algorithms,
and connects to different geometric interpretations.

    ┌──────────────┬────────────────┬────────────────────────────────────┐
    │ Factorisation│ Applies to     │ Primary use                        │
    ├──────────────┼────────────────┼────────────────────────────────────┤
    │ LU           │ Square A       │ Solving Ax=b (Gaussian elim.)      │
    │ QR           │ Any A          │ Least squares, eigenvalue algs.    │
    │ Cholesky     │ PSD symmetric  │ Faster linear systems, sampling    │
    │ Eigendecomp  │ Square (sym.)  │ PCA, spectral methods, powers      │
    │ SVD          │ Any A          │ Low-rank approx., PCA, pseudoinv.  │
    └──────────────┴────────────────┴────────────────────────────────────┘


### LU Factorisation

    A = LU  (or PA = LU with pivoting)

    L: unit lower triangular (1s on diagonal, below is Gaussian multipliers)
    U: upper triangular (the row echelon form of A)

    Algorithm: Gaussian elimination. O(n³) once, then O(n²) per new b.
    Use when: solving Ax=b for many right-hand sides b. Standard in LAPACK.

    Geometric view: L encodes the shearing operations of elimination,
    U encodes the triangular result.


### QR Factorisation

    A = QR

    Q: m×n orthonormal columns (Qᵀ Q = I_n)
    R: n×n upper triangular with positive diagonal

    Algorithms: Gram-Schmidt (unstable), Householder reflections (stable),
    Givens rotations (sparse A).

    Least squares via QR (numerically preferred over normal equations):
        Ax̂ = b  ≈  QRx̂ = b  →  Rx̂ = Qᵀb   (one back-substitution)
        More numerically stable than computing (AᵀA)⁻¹Aᵀb explicitly.

    Eigenvalue computation: the QR algorithm (unrelated to factorisation)
    iteratively applies QR decompositions to converge to the Schur form.


### Cholesky Factorisation

    A = LLᵀ  (or A = RᵀR)

    ONLY for symmetric positive definite (SPD) matrices.
    L: lower triangular with positive diagonal.

    Algorithm: O(n³/3) — TWICE as fast as LU for the same matrix size.
    Use when: solving normal equations AᵀAx = Aᵀb, sampling from
    multivariate Gaussians (x = μ + Lz where z~N(0,I)), Kalman filters.

    Existence theorem: A is SPD if and only if the Cholesky factorisation
    exists (all diagonal entries of L are real and positive).


### The Singular Value Decomposition (SVD)

    A = UΣVᵀ

    U ∈ ℝ^{m×m}: left singular vectors (orthonormal columns)
    Σ ∈ ℝ^{m×n}: diagonal, σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0 (singular values)
    V ∈ ℝ^{n×n}: right singular vectors (orthonormal columns)

    The SVD EXISTS for EVERY matrix — no symmetry or invertibility needed.
    It is the most important decomposition in applied linear algebra.

    Geometric interpretation — a 3-step transformation:
        1. Vᵀ: rotate/reflect in the input space (change to input basis)
        2. Σ: scale each dimension independently (stretch/shrink)
        3. U: rotate/reflect in the output space (change to output basis)

    ANY linear transformation = rotate → scale → rotate.

    Diagram 5 — SVD Geometry:

    Input ℝⁿ        scale       Output ℝᵐ
    ────────→  v₁ ──σ₁──→  u₁  ──────────→
              v₂ ──σ₂──→  u₂
              v₃ ──σ₃──→  u₃
              v₄ ──0───→  (0)   ← killed (in null space)

    The vᵢ are orthonormal right singular vectors (basis of ℝⁿ).
    The uᵢ are orthonormal left singular vectors (basis of ℝᵐ).
    A maps vᵢ to σᵢuᵢ — each input direction to a scaled output direction.

**Connection to eigenvalues:**

    AᵀA = VΣᵀΣVᵀ → right singular vectors = eigenvectors of AᵀA
    AAᵀ = UΣΣᵀUᵀ → left singular vectors = eigenvectors of AAᵀ
    σᵢ = √λᵢ(AᵀA)

    For symmetric PSD matrix A = AᵀA: SVD = eigendecomposition (U=V=Q).

**Truncated SVD (best low-rank approximation):**

    Aₖ = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ

    By the Eckart-Young theorem, Aₖ is the best rank-k approximation to A
    under BOTH the Frobenius norm AND the spectral norm:

        min_{rank(B)≤k} ‖A − B‖_F = ‖A − Aₖ‖_F = √(Σᵢ₌ₖ₊₁ σᵢ²)
        min_{rank(B)≤k} ‖A − B‖₂ = ‖A − Aₖ‖₂ = σₖ₊₁

    Applications:
        Image compression: store only k×(m+n+1) numbers instead of m×n
        Noise removal: small singular values encode noise → truncate them
        Latent semantic analysis: topic modelling via document-word matrix SVD
        Collaborative filtering: Netflix-style recommender systems

**Moore-Penrose Pseudoinverse:**

    A⁺ = VΣ⁺Uᵀ    where Σ⁺ replaces each σᵢ > 0 by 1/σᵢ

    A⁺ is the unique matrix satisfying the four Moore-Penrose conditions.
    For full column rank: A⁺ = (AᵀA)⁻¹Aᵀ  (left inverse)
    For full row rank:    A⁺ = Aᵀ(AAᵀ)⁻¹  (right inverse)
    Solution to min-norm least squares: x̂ = A⁺b


### PART 9 — PRINCIPAL COMPONENT ANALYSIS & LOW-RANK METHODS

### PCA as a Linear Algebra Problem

Given data matrix X ∈ ℝ^{n×d} (n samples, d features):

    Step 1: Centre — subtract column means: X̃ = X − 1μᵀ
    Step 2: Compute covariance matrix: C = X̃ᵀX̃ / (n−1) ∈ ℝ^{d×d}
    Step 3: Eigendecompose C = QΛQᵀ  (C is symmetric, so this always exists)
    Step 4: Sort eigenvectors q₁,...,qd by eigenvalue (λ₁ ≥ λ₂ ≥ ... ≥ λd)

    Principal components: q₁, q₂, ... are the directions of maximum variance.
    Projected scores: Z = X̃Q_k  (n × k scores in the PC space)

**PCA via SVD (preferred numerically):**

    X̃ = UΣVᵀ

    Right singular vectors V = Q  (the eigenvectors of X̃ᵀX̃ = (n−1)C)
    Singular values σᵢ and eigenvalues λᵢ: σᵢ² = (n−1)λᵢ
    Scores Z = U_kΣ_k = X̃V_k

    Why SVD is numerically better: avoids computing X̃ᵀX̃ explicitly
    (squaring the matrix doubles the condition number).


### Explained Variance

    Variance explained by PC k:     λₖ / Σᵢ λᵢ = σₖ² / Σᵢ σᵢ²

    Cumulative explained variance:  Σᵢ₌₁ᵏ λᵢ / Σᵢ λᵢ

    Rule of thumb: choose k such that cumulative variance ≥ 95% (or
    "scree plot elbow" — the point where adding a component gives
    diminishing returns).

    Total variance in data = tr(C) = Σᵢ λᵢ = Σᵢ σᵢ² / (n−1)
    Frobenius reconstruction error: ‖X̃ − X̃ₖ‖_F² = Σᵢ₌ₖ₊₁ σᵢ²


### Interpretations of PCA

    Geometric:   Find the k-dimensional affine subspace closest to the data
                 (in Euclidean distance) — the "best fit flat".

    Statistical: Find k directions of maximum variance, uncorrelated with each other.

    Compression: Find the projection that minimises average reconstruction error.

    All three interpretations are equivalent — they give the same answer.


### Limitations of PCA (and when NOT to use it)

    Linear only: PCA finds linear structure. Non-linear manifolds
                 require kernel PCA, t-SNE, UMAP, autoencoders.

    Variance ≠ information: the highest-variance direction is not
                 necessarily the most predictive of y (see LDA).

    Scale-sensitive: PCA is dominated by high-variance features.
                 ALWAYS standardise features before PCA (unless you want
                 variance in the original units to determine importance).

    Gaussian assumption: PCA is optimal for Gaussian data. Non-Gaussian
                 structure (clusters, non-linearities) is not captured.


### PART 10 — MATRIX CALCULUS & GRADIENTS

### Notation Convention

We use the denominator layout (Jacobian is ∂y/∂xᵀ convention):

    Scalar f, vector x ∈ ℝⁿ:  ∂f/∂x ∈ ℝⁿ     (column vector, same shape as x)
    Vector f ∈ ℝᵐ, vector x ∈ ℝⁿ: ∂f/∂x ∈ ℝ^{n×m}  (Jacobian, columns = ∂f/∂xᵢ)

Note: PyTorch/NumPy use numerator layout for Jacobians. Be consistent
within a derivation and always check against the shape you expect.


### Essential Gradient Identities

    ∂(aᵀx)/∂x = a                           (linear form)
    ∂(xᵀAx)/∂x = (A + Aᵀ)x = 2Ax  if A symmetric   (quadratic form)
    ∂(Ax)/∂x = Aᵀ                           (matrix-vector product)
    ∂(xᵀx)/∂x = 2x                          (squared norm)
    ∂‖Ax−b‖₂²/∂x = 2Aᵀ(Ax−b)              (least squares gradient)
    ∂(tr(AB))/∂A = Bᵀ                       (trace gradient)
    ∂(det(A))/∂A = det(A) · A⁻ᵀ            (determinant gradient)
    ∂(log det(A))/∂A = A⁻ᵀ = (A⁻¹)ᵀ       (used in Gaussian MLE)


### The Jacobian and Hessian

**Jacobian** of f: ℝⁿ → ℝᵐ at point x:

    J(x) ∈ ℝ^{m×n}  where  Jᵢⱼ = ∂fᵢ/∂xⱼ

    The Jacobian is the best LINEAR APPROXIMATION to f at x:
        f(x + δ) ≈ f(x) + J(x)δ    for small δ

    det(J(x)) is the local volume scaling factor of f (change of variables).

**Hessian** of scalar f: ℝⁿ → ℝ at point x:

    H(x) ∈ ℝ^{n×n}  where  Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ

    H is SYMMETRIC (Schwarz theorem, if f is twice continuously differentiable).
    The second-order Taylor expansion:
        f(x + δ) ≈ f(x) + ∇f(x)ᵀδ + ½ δᵀH(x)δ

    At a critical point ∇f(x)=0:
        H PD   → local minimum     (bowl curves up)
        H ND   → local maximum     (bowl curves down)
        Mixed  → saddle point      (both up and down directions)


### The Chain Rule in Matrix Form

For composite f(g(x)):

    df/dx = (∂g/∂x)ᵀ · (∂f/∂g)

This is the foundation of backpropagation in neural networks:

    Layer output: h = σ(Wx + b)
    Chain rule: ∂L/∂W = (∂L/∂h) · (∂h/∂W)   [accumulates across layers]

    Backpropagation is simply the chain rule applied recursively,
    using the matrix operations of forward-pass to efficiently compute
    gradients in reverse order.


### Gradient Descent as Linear Algebra

For a quadratic loss f(x) = ½xᵀAx − bᵀx (A symmetric PD):

    Gradient: ∇f = Ax − b
    Minimum:  x* = A⁻¹b  (solving the linear system)

    Gradient descent: xₜ₊₁ = xₜ − η(Axₜ − b)
    Convergence rate: error at step t is proportional to (1 − ηλ_min)^t

    The OPTIMAL learning rate is η* = 2/(λ_max + λ_min)
    The CONDITION NUMBER κ = λ_max/λ_min determines convergence speed:
        κ ≈ 1:   fast convergence (spherical loss landscape)
        κ >> 1:  slow convergence (elongated elliptical landscape)

    Newton's method: xₜ₊₁ = xₜ − H⁻¹∇f = xₜ − A⁻¹(Axₜ−b)
    Converges in 1 step for quadratic f (preconditioning with H⁻¹).
    Expensive (O(n³) per step) but condition-number-independent.


### PART 11 — TENSORS & MULTILINEAR ALGEBRA

### What Is a Tensor?

In the deep learning sense, a tensor is a multi-dimensional array with
a consistent mathematical transformation rule:

    Scalar:    0-dimensional tensor (order 0)  — a single number
    Vector:    1-dimensional tensor (order 1)  — shape (n,)
    Matrix:    2-dimensional tensor (order 2)  — shape (m, n)
    3-tensor:  shape (m, n, p)  — e.g., batch of matrices, RGB image
    4-tensor:  shape (b, c, h, w) — batch of images in a CNN

In rigorous mathematics, a tensor is a multilinear map. In ML, the
programming abstraction (n-dimensional array with automatic differentiation)
is almost always what is meant.


### Broadcasting and Einsum

**Broadcasting**: implicit extension of tensor dimensions for element-wise ops.

    Rule: dimensions are aligned from the right. A dimension of 1 is
    treated as if it were the size of the other tensor.

    (batch=32, d=256) + (d=256,) → (32, 256)  ✓
    (32, 256) · (256, 512) → (32, 512)         ✓  (matmul)

**Einstein summation (einsum)**: compact notation for contraction.

    Matrix multiply:  C = einsum("ij,jk→ik", A, B)  i.e. Cᵢₖ = Σⱼ AᵢⱼBⱼₖ
    Dot product:      s = einsum("i,i→", u, v)
    Outer product:    M = einsum("i,j→ij", u, v)
    Trace:            t = einsum("ii→", A)
    Batch matmul:     C = einsum("bij,bjk→bik", A, B)

    einsum is universal — any tensor contraction expressible this way.


### Kronecker Product and Vectorisation

**Kronecker product** A ⊗ B: block matrix where each entry of A is scaled by B.

    (A ⊗ B)ᵢⱼ = aᵢⱼ · B    (block structure)
    (A ⊗ B)(C ⊗ D) = AC ⊗ BD    (mixed product property)
    Used in: KFAC optimiser (approximate Fisher information matrix),
    multi-output regression, quantum computing.

**vec(A)** operator: stacks columns of A into a long vector.

    vec(ABC) = (Cᵀ ⊗ A) vec(B)
    This connects matrix equations to vector equations with Kronecker products.


### PART 12 — LINEAR ALGEBRA IN MACHINE LEARNING: THE UNIFIED VIEW

### Every Major ML Algorithm Through Linear Algebra

    ┌──────────────────────────┬───────────────────────────────────────────┐
    │ Algorithm / Concept      │ Core Linear Algebra                       │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ Linear regression        │ Least squares: x̂ = (XᵀX)⁻¹Xᵀy             │
    │                          │ Normal equations; projection onto C(X)    │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ Ridge regression         │ (XᵀX + λI)⁻¹Xᵀy — regularised inverse     │
    │                          │ Adds λ to eigenvalues → well-conditioned  │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ PCA                      │ SVD of centred data X̃ = UΣVᵀ              │
    │                          │ Eigendecomp of covariance C = QΛQᵀ        │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ LDA                      │ Generalised eigenvalue problem:           │
    │                          │ S_W⁻¹S_B w = λw  (between vs within)      │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ Neural network fwd pass  │ h = σ(Wx + b): matrix-vector multiply     │
    │                          │ per layer; composed linear maps           │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ Backpropagation          │ Chain rule in matrix form; Jacobian       │
    │                          │ products accumulate right-to-left         │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ Attention mechanism      │ Attention(Q,K,V) = softmax(QKᵀ/√d)V       │
    │                          │ Three projections + scaled dot product    │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ Gaussian processes       │ Kernel matrix K (PSD) + Cholesky solve    │
    │                          │ Posterior: (K + σ²I)⁻¹ via Cholesky        │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ SVMs (kernel)            │ Dual: maximise ½αᵀKα − 1ᵀα                │
    │                          │ K = kernel matrix (PSD, Gram matrix)      │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ Spectral clustering      │ Eigendecomp of normalised Laplacian L     │
    │                          │ Second eigenvector = Fiedler vector       │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ Collaborative filtering  │ Low-rank SVD: A ≈ UΣVᵀ (matrix complete)  │
    │                          │ Alternating least squares factorisation   │
    ├──────────────────────────┼───────────────────────────────────────────┤
    │ Batch normalisation      │ Standardise by μ,σ; learnable scale γ,β   │
    │                          │ Covariance whitening (ZCA transform)      │
    └──────────────────────────┴───────────────────────────────────────────┘


### Numerical Linear Algebra: Key Practical Points

**Never compute the matrix inverse explicitly:**
    x = A⁻¹b  → slow and numerically unstable
    Instead: solve Ax = b via LU (general), Cholesky (PSD), QR (least squares)

**Condition number matters:**
    κ(A) = σ_max/σ_min ≈ 10^d → you lose ~d digits of precision
    Ill-conditioned systems (large κ) require regularisation or preconditioning

**Memory and complexity:**
    Dense matrix-vector multiply: O(n²) time, O(n²) space
    Dense matrix-matrix multiply: O(n³) — bottleneck of deep learning
    Sparse matrices: O(nnz) where nnz = number of non-zeros
    GPUs exploit highly parallel matrix operations via BLAS/cuBLAS

**The solve-don't-invert principle:**
    Instead of computing (AᵀA)⁻¹Aᵀb, solve the n×n system AᵀAx = Aᵀb
    Or better: use QR factorisation, which avoids AᵀA entirely.

**Low-rank structure is everywhere in ML:**
    Weight matrices in over-parameterised networks tend to be low-rank
    LoRA (Low-Rank Adaptation): finetune W₀ + AB where rank(AB) << n
    The effective rank of learned representations reflects task complexity.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · The Four Fundamental Subspaces — Visualised": {
        "description": (
            "Construct a concrete matrix and compute all four subspaces. "
            "Verify the rank-nullity theorem. Show the orthogonal decomposition "
            "of a vector into row space and null space components. "
            "Verify that C(A) ⊥ N(Aᵀ) and C(Aᵀ) ⊥ N(A) numerically."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from scipy import linalg

np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)

print("=" * 65)
print("  THE FOUR FUNDAMENTAL SUBSPACES")
print("=" * 65)
print()

# \u2500\u2500 A rank-2 matrix with known null space \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# A \u2208 \u211d^{4\u00d75}, rank 2: lots of null space to examine
A = np.array([
    [1,  2,  3,  1, 0],
    [2,  4,  7,  0, 1],
    [0,  0,  1, -2, 1],
    [1,  2,  4, -1, 1],
], dtype=float)

m, n = A.shape
rank = np.linalg.matrix_rank(A)

print(f"  Matrix A: shape {m}\u00d7{n},  rank = {rank}")
print()
print(f"  A =")
for row in A:
    print(f"    {row}")
print()

# \u2500\u2500 Compute all four subspaces via SVD \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
U, s, Vt = np.linalg.svd(A, full_matrices=True)
V = Vt.T
tol = 1e-9

print("  SINGULAR VALUES:")
print(f"  {s.round(4)}")
print()

# Column space: first r columns of U
col_space  = U[:, :rank]
# Left null space: last m-r columns of U
left_null  = U[:, rank:]
# Row space: first r rows of Vt = first r columns of V
row_space  = V[:, :rank]
# Null space: last n-r columns of V
null_space = V[:, rank:]

dim_col  = col_space.shape[1]
dim_left = left_null.shape[1]
dim_row  = row_space.shape[1]
dim_null = null_space.shape[1]

print("  DIMENSIONS (verify Rank-Nullity Theorem):")
print(f"  {'Subspace':30s} | {'Dimension':>10} | {'Space'}")
print(f"  {'\u2500'*56}")
print(f"  {'Column space C(A)':30s} | {dim_col:10d} | \u211d^{m}")
print(f"  {'Left null space N(A\u1d40)':30s} | {dim_left:10d} | \u211d^{m}")
print(f"  {'Row space C(A\u1d40)':30s} | {dim_row:10d} | \u211d^{n}")
print(f"  {'Null space N(A)':30s} | {dim_null:10d} | \u211d^{n}")
print()
print(f"  Rank-Nullity (cols): dim C(A\u1d40) + dim N(A) = {dim_row} + {dim_null} = {dim_row+dim_null} = n={n} \u2713")
print(f"  Rank-Nullity (rows): dim C(A)  + dim N(A\u1d40) = {dim_col} + {dim_left} = {dim_col+dim_left} = m={m} \u2713")
print()

# \u2500\u2500 Verify orthogonality numerically \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  ORTHOGONALITY VERIFICATION:")
print()
print("  C(A) \u22a5 N(A\u1d40):  max |col_space\u1d40 \u00b7 left_null| = "
      f"{np.abs(col_space.T @ left_null).max():.2e}  (should be \u22480)")
print("  C(A\u1d40) \u22a5 N(A): max |row_space\u1d40 \u00b7 null_space| = "
      f"{np.abs(row_space.T @ null_space).max():.2e}  (should be \u22480)")
print()

# \u2500\u2500 Null space verification \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  NULL SPACE VERIFICATION:  A \u00b7 N(A) \u2248 0")
print(f"  \u2016A \u00b7 null_space\u2016_F = {np.linalg.norm(A @ null_space):.2e}  (should be \u22480)")
print()

# \u2500\u2500 Decompose a vector into row space + null space components \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  VECTOR DECOMPOSITION: x = x_row + x_null")
print()
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# Project x onto row space
x_row  = row_space @ (row_space.T @ x)
# Null space component
x_null = null_space @ (null_space.T @ x)

print(f"  x       = {x.round(4)}")
print(f"  x_row   = {x_row.round(4)}   (in row space)")
print(f"  x_null  = {x_null.round(4)}   (in null space)")
print(f"  x_row + x_null = {(x_row + x_null).round(4)}")
print(f"  Error   = {np.linalg.norm(x - x_row - x_null):.2e}")
print()
print(f"  x_row \u00b7 x_null = {x_row @ x_null:.2e}  (orthogonal \u2713)")
print()

# Verify: A maps x_null to 0 and x_row faithfully
print(f"  A\u00b7x_null = {(A @ x_null).round(4)}   (should be zero \u2713)")
print(f"  A\u00b7x      = {(A @ x).round(4)}")
print(f"  A\u00b7x_row  = {(A @ x_row).round(4)}    (same! null part contributes nothing)")
print()

# \u2500\u2500 Plot: singular value spectrum \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("The Four Fundamental Subspaces of A \u2208 \u211d\u2074\u02e3\u2075",
             fontsize=12, fontweight="bold")

# Plot 1: Singular value spectrum
sv_colors = ["steelblue"]*rank + ["tomato"]*(len(s)-rank)
axes[0].bar(range(1, len(s)+1), s, color=sv_colors, alpha=0.85)
axes[0].axhline(tol, color="gray", linestyle="--", lw=1.5, label=f"Threshold={tol:.0e}")
axes[0].set_xlabel("Singular value index")
axes[0].set_ylabel("\u03c3\u1d62")
axes[0].set_title("Singular Value Spectrum\\n(blue=rank, red=null)")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Plot 2: Subspace dimensions diagram
labels = ["C(A)\\n(col space)", "N(A\u1d40)\\n(left null)", "C(A\u1d40)\\n(row space)", "N(A)\\n(null space)"]
dims   = [dim_col, dim_left, dim_row, dim_null]
spaces = [f"\u211d^{m}", f"\u211d^{m}", f"\u211d^{n}", f"\u211d^{n}"]
colors = ["steelblue", "tomato", "seagreen", "orange"]
bars = axes[1].bar(labels, dims, color=colors, alpha=0.85)
axes[1].set_ylabel("Dimension")
axes[1].set_title("Four Subspace Dimensions\\n(pairs sum to m or n)")
for bar, d, sp in zip(bars, dims, spaces):
    axes[1].text(bar.get_x()+bar.get_width()/2, d+0.05, f"dim={d}\\nin {sp}",
                 ha="center", va="bottom", fontsize=9)
axes[1].set_ylim(0, max(dims)+1.5)
axes[1].grid(alpha=0.3, axis="y")

# Plot 3: Vector decomposition
axes[2].bar(["x_row", "x_null", "x"],
            [np.linalg.norm(x_row), np.linalg.norm(x_null), np.linalg.norm(x)],
            color=["steelblue", "tomato", "seagreen"], alpha=0.85)
axes[2].set_ylabel("\u2016\u00b7\u2016\u2082")
axes[2].set_title(f"Vector x Decomposition\\n\u2016x\u2016\u00b2 = \u2016x_row\u2016\u00b2 + \u2016x_null\u2016\u00b2 (Pythagoras)")
check = np.linalg.norm(x_row)**2 + np.linalg.norm(x_null)**2
axes[2].text(0.5, 0.05, f"\u2016x_row\u2016\u00b2+\u2016x_null\u2016\u00b2={check:.4f}\\n\u2016x\u2016\u00b2={np.linalg.norm(x)**2:.4f} \u2713",
             ha="center", transform=axes[2].transAxes, fontsize=9)
axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "four_subspaces.png", dpi=120)
print("  Plot saved \u2192 four_subspaces.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · SVD Deep Dive — Geometry, Low-Rank Approximation & Image Compression": {
        "description": (
            "Compute the SVD and visualise the three geometric transformations "
            "(Vᵀ, Σ, U). Demonstrate the Eckart-Young theorem: show that the "
            "rank-k truncation is the best rank-k approximation. "
            "Apply SVD compression to an image and plot reconstruction error vs rank."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)

print("=" * 65)
print("  SVD: GEOMETRY, LOW-RANK APPROXIMATION & COMPRESSION")
print("=" * 65)
print()

# \u2500\u2500 Part 1: SVD of a simple 3\u00d72 matrix \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 1 \u2014 MANUAL SVD INSPECTION")
print()
A = np.array([[3, 2],
              [2, 3],
              [1, 0]], dtype=float)

U, s, Vt = np.linalg.svd(A, full_matrices=True)
V = Vt.T
Sigma = np.zeros_like(A)
for i, sv in enumerate(s):
    Sigma[i, i] = sv

print(f"  A (3\u00d72):\\n  {A}")
print()
print(f"  U (3\u00d73, left singular vectors):\\n  {U.round(4)}")
print(f"  \u03a3 diagonal (singular values): {s.round(4)}")
print(f"  V (2\u00d72, right singular vectors):\\n  {V.round(4)}")
print()

# Verify reconstruction
A_reconstructed = U @ Sigma @ Vt
print(f"  Reconstruction error \u2016A \u2212 U\u03a3V\u1d40\u2016_F: {np.linalg.norm(A - A_reconstructed):.2e}")
print()

# Column and row spaces from SVD
print("  Column space basis (first r cols of U):")
print(f"  u\u2081 = {U[:, 0].round(4)},  u\u2082 = {U[:, 1].round(4)}")
print("  Right singular vectors (rows of V\u1d40 = cols of V):")
print(f"  v\u2081 = {V[:, 0].round(4)},  v\u2082 = {V[:, 1].round(4)}")
print()
print("  A maps: v\u2081 \u2500\u2500\u03c3\u2081\u2500\u2500\u2192 u\u2081  (stretches most)")
print(f"                 {s[0]:.4f}")
print(f"          v\u2082 \u2500\u2500\u03c3\u2082\u2500\u2500\u2192 u\u2082  (stretches less)")
print(f"                 {s[1]:.4f}")
print()

# \u2500\u2500 Part 2: Rank-1 outer product decomposition \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 2 \u2014 RANK-1 DECOMPOSITION (A\u2096 = \u03a3\u1d62\u208c\u2081\u1d4f \u03c3\u1d62 u\u1d62v\u1d62\u1d40)")
print()
print(f"  {'k':>4} | {'\u2016A\u2212A\u2096\u2016_F':>12} | {'% var explained':>17} | {'Eckart-Young bound':>20}")
print(f"  {'\u2500'*60}")
total_sq = np.sum(s**2)
for k in range(1, len(s)+1):
    Ak = sum(s[i] * np.outer(U[:, i], V[:, i]) for i in range(k))
    err_F = np.linalg.norm(A - Ak)
    var_exp = np.sum(s[:k]**2) / total_sq * 100
    ey_bound = np.sqrt(np.sum(s[k:]**2))   # = \u2016A\u2212A\u2096\u2016_F from Eckart-Young
    print(f"  {k:4d} | {err_F:12.6f} | {var_exp:16.2f}% | {ey_bound:20.6f}  \u2190 same \u2713")

print()

# \u2500\u2500 Part 3: Image compression via SVD \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 3 \u2014 IMAGE COMPRESSION VIA TRUNCATED SVD")
print()

# Create a synthetic "image" with structure
np.random.seed(7)
size = 100
# Smooth low-rank background + some texture
x_coord = np.linspace(0, 2*np.pi, size)
y_coord = np.linspace(0, 2*np.pi, size)
XX, YY = np.meshgrid(x_coord, y_coord)
img = (np.sin(XX) * np.cos(YY)
       + 0.5 * np.sin(2*XX) * np.sin(YY)
       + 0.3 * np.random.randn(size, size))
img = (img - img.min()) / (img.max() - img.min())   # normalise to [0,1]

U_img, s_img, Vt_img = np.linalg.svd(img, full_matrices=False)

m_img, n_img = img.shape
total_params = m_img * n_img

print(f"  Image size: {m_img}\u00d7{n_img} = {total_params:,} pixels (parameters)")
print()
print(f"  {'Rank k':>8} | {'Storage':>12} | {'% of full':>11} | "
      f"{'RMSE':>8} | {'PSNR (dB)':>10}")
print(f"  {'\u2500'*58}")

rank_vals = [1, 2, 5, 10, 20, 50, 100]
rmse_vals, psnr_vals, storage_vals = [], [], []

for k in rank_vals:
    Ak_img = U_img[:, :k] @ np.diag(s_img[:k]) @ Vt_img[:k, :]
    Ak_img = np.clip(Ak_img, 0, 1)
    storage = k * (m_img + n_img + 1)
    pct     = storage / total_params * 100
    rmse    = np.sqrt(np.mean((img - Ak_img)**2))
    psnr    = 20 * np.log10(1.0 / (rmse + 1e-10))
    rmse_vals.append(rmse)
    psnr_vals.append(psnr)
    storage_vals.append(storage)
    print(f"  {k:8d} | {storage:12,} | {pct:10.1f}% | {rmse:8.4f} | {psnr:10.2f}")

print()
explained_90 = np.searchsorted(-np.cumsum(s_img**2)/np.sum(s_img**2), -0.90) + 1
print(f"  Rank needed for 90% variance explained: {explained_90}")
print(f"  Rank needed for 99% variance explained: "
      f"{np.searchsorted(-np.cumsum(s_img**2)/np.sum(s_img**2), -0.99) + 1}")
print()

# \u2500\u2500 Plots \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("SVD: Geometry, Low-Rank Approximation & Image Compression",
             fontsize=12, fontweight="bold")

# Row 1: reconstructions at different ranks
show_ranks = [1, 5, 20, 100]
for ax, k in zip(axes[0], show_ranks):
    Ak = U_img[:, :k] @ np.diag(s_img[:k]) @ Vt_img[:k, :]
    Ak = np.clip(Ak, 0, 1)
    rmse = np.sqrt(np.mean((img - Ak)**2))
    storage = k * (m_img + n_img + 1)
    ax.imshow(Ak, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Rank {k}\\nRMSE={rmse:.3f}, {storage/total_params*100:.1f}% storage",
                 fontsize=9)
    ax.axis("off")

# Row 2: analysis plots
# Singular value spectrum (log scale)
axes[1, 0].semilogy(s_img[:50], "steelblue", lw=2.5, marker="o", ms=4)
axes[1, 0].axvline(explained_90-1, color="tomato", linestyle="--", lw=2,
                   label=f"90% var @ k={explained_90}")
axes[1, 0].set_xlabel("Index i"); axes[1, 0].set_ylabel("\u03c3\u1d62 (log scale)")
axes[1, 0].set_title("Singular Value Spectrum (first 50)")
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=0.3)

# Cumulative explained variance
cum_var = np.cumsum(s_img**2) / np.sum(s_img**2) * 100
axes[1, 1].plot(cum_var[:60], "seagreen", lw=2.5)
axes[1, 1].axhline(90, color="tomato", linestyle="--", lw=1.5, label="90%")
axes[1, 1].axhline(99, color="orange", linestyle="--", lw=1.5, label="99%")
axes[1, 1].set_xlabel("Rank k"); axes[1, 1].set_ylabel("Cumulative variance (%)")
axes[1, 1].set_title("Cumulative Explained Variance")
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=0.3)

# RMSE vs rank
axes[1, 2].semilogy(rank_vals, rmse_vals, "tomato", lw=2.5, marker="o", ms=6)
axes[1, 2].set_xlabel("Rank k"); axes[1, 2].set_ylabel("RMSE (log scale)")
axes[1, 2].set_title("Reconstruction Error vs Rank")
axes[1, 2].grid(alpha=0.3)

# Storage vs rank
axes[1, 3].plot(rank_vals, [s/total_params*100 for s in storage_vals],
                "purple", lw=2.5, marker="s", ms=6)
axes[1, 3].set_xlabel("Rank k"); axes[1, 3].set_ylabel("Storage (% of original)")
axes[1, 3].set_title("Storage Cost vs Rank")
axes[1, 3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "svd_compression.png", dpi=120)
print("  Plot saved \u2192 svd_compression.png")
print()
print("  KEY TAKEAWAYS:")
print("  SVD = rotate (V\u1d40) \u2192 scale (\u03a3) \u2192 rotate (U)")
print("  Every rank-1 term \u03c3\u1d62u\u1d62v\u1d62\u1d40 captures one 'mode' of the data")
print("  Eckart-Young: rank-k truncation is the BEST rank-k approximation")
print("  Singular values decay fast for structured data \u2192 huge compression")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Eigenvalues — Geometry, Power Iteration & Hessian Analysis": {
        "description": (
            "Visualise eigenvalues as the stretching directions of a matrix. "
            "Implement power iteration from scratch and show convergence rate "
            "= |λ₁/λ₂|. Connect Hessian eigenvalues to the loss landscape: "
            "condition number, gradient descent convergence, and saddle points."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


np.random.seed(42)
np.set_printoptions(precision=5, suppress=True)

print("=" * 65)
print("  EIGENVALUES: GEOMETRY, POWER ITERATION & HESSIAN ANALYSIS")
print("=" * 65)
print()

# \u2500\u2500 Part 1: Geometric view \u2014 what eigenvalues DO \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 1 \u2014 GEOMETRIC INTERPRETATION OF EIGENVALUES")
print()

A = np.array([[3., 1.],
              [1., 2.]])
eigenvalues, eigenvectors = np.linalg.eig(A)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"  A = [[3, 1],\\n       [1, 2]]  (symmetric \u2014 real eigenvalues, orthogonal eigenvectors)")
print()
print(f"  Eigenvalue \u03bb\u2081 = {eigenvalues[0]:.4f},  eigenvector v\u2081 = {eigenvectors[:,0].round(4)}")
print(f"  Eigenvalue \u03bb\u2082 = {eigenvalues[1]:.4f},  eigenvector v\u2082 = {eigenvectors[:,1].round(4)}")
print()

# Verify: Av = \u03bbv
for i in range(2):
    lhs = A @ eigenvectors[:, i]
    rhs = eigenvalues[i] * eigenvectors[:, i]
    print(f"  A\u00b7v{i+1} = {lhs.round(4)},  \u03bb{i+1}\u00b7v{i+1} = {rhs.round(4)},  "
          f"match: {np.allclose(lhs, rhs)} \u2713")

print()
print(f"  Trace check: tr(A) = {np.trace(A)} = \u03bb\u2081+\u03bb\u2082 = {eigenvalues.sum():.4f} \u2713")
print(f"  Det check:   det(A) = {np.linalg.det(A):.4f} = \u03bb\u2081\u00d7\u03bb\u2082 = {eigenvalues.prod():.4f} \u2713")
print()

# Spectral decomposition
P = eigenvectors
D = np.diag(eigenvalues)
A_reconstructed = P @ D @ P.T
print(f"  Spectral decomp: A = P\u039bP\u1d40")
print(f"  Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e} \u2713")
print()

# \u2500\u2500 Part 2: Power Iteration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 2 \u2014 POWER ITERATION FROM SCRATCH")
print()
print("  Algorithm: b_{k+1} = A\u00b7b_k / \u2016A\u00b7b_k\u2016   \u2192 converges to dominant eigenvector")
print(f"  Convergence rate: |\u03bb\u2081/\u03bb\u2082| = |{eigenvalues[0]:.4f}/{eigenvalues[1]:.4f}| = "
      f"{abs(eigenvalues[0]/eigenvalues[1]):.4f}")
print()

# Use a matrix with larger spectral gap for clearer convergence demo
B = np.array([[5., 1., 0.5],
              [1., 3., 0.2],
              [0.5, 0.2, 1.]])
B_eigs = np.sort(np.linalg.eigvalsh(B))[::-1]
rate = abs(B_eigs[1] / B_eigs[0])

b = np.random.randn(3)
b = b / np.linalg.norm(b)
true_eigvec = np.linalg.eigh(B)[1][:, -1]   # largest eigenvector
true_eigval = B_eigs[0]

print(f"  Matrix B eigenvalues: {B_eigs.round(4)}")
print(f"  Convergence rate |\u03bb\u2082/\u03bb\u2081|: {rate:.4f}")
print()
print(f"  {'Iteration':>10} | {'Rayleigh quotient':>18} | {'Error to \u03bb\u2081':>14} | {'Angle to v\u2081 (deg)':>18}")
print(f"  {'\u2500'*68}")

errors, angles = [], []
for k in range(20):
    Ab = B @ b
    rayleigh = b @ Ab
    err = abs(rayleigh - true_eigval)
    angle = np.degrees(np.arccos(np.clip(abs(np.dot(b, true_eigvec)), 0, 1)))
    errors.append(err)
    angles.append(angle)
    if k in [0, 1, 2, 4, 9, 19]:
        print(f"  {k:10d} | {rayleigh:18.6f} | {err:14.6f} | {angle:18.4f}")
    b = Ab / np.linalg.norm(Ab)

print()
print(f"  True \u03bb\u2081 = {true_eigval:.6f}")
print()

# \u2500\u2500 Part 3: Hessian and Loss Landscape \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 3 \u2014 HESSIAN EIGENVALUES AND GRADIENT DESCENT CONVERGENCE")
print()

scenarios = [
    ("Well-conditioned (\u03ba\u22481)",  np.diag([2.0, 1.8])),
    ("Moderate condition (\u03ba=10)", np.diag([10.0, 1.0])),
    ("Ill-conditioned (\u03ba=100)",  np.diag([100.0, 1.0])),
    ("Saddle point",             np.diag([3.0, -1.0])),
]

def gd_on_quadratic(H, n_steps=200, lr=None):
    """Gradient descent on f(x) = 0.5 x\u1d40Hx, starting at x0=[1,1]."""
    eigvals = np.linalg.eigvalsh(H)
    lam_min = eigvals.min(); lam_max = eigvals.max()
    if lr is None:
        lr = 2.0 / (lam_max + max(lam_min, 0.01))
    x = np.ones(H.shape[0])
    losses = []
    for _ in range(n_steps):
        f = 0.5 * x @ H @ x
        losses.append(f)
        grad = H @ x
        x = x - lr * grad
        if np.linalg.norm(x) > 1e6:   # diverged
            break
    return losses

print(f"  {'Scenario':28s} | {'\u03bb_min':>8} | {'\u03bb_max':>8} | {'\u03ba=\u03bb_max/\u03bb_min':>14} | {'Converges?'}")
print(f"  {'\u2500'*72}")

conv_losses = []
for name, H in scenarios:
    eigvals = np.linalg.eigvalsh(H)
    lmin, lmax = eigvals.min(), eigvals.max()
    kappa = lmax / max(abs(lmin), 1e-6)
    converges = "YES" if lmin > 0 else "NO (saddle)"
    print(f"  {name:28s} | {lmin:8.2f} | {lmax:8.2f} | {kappa:14.1f} | {converges}")
    losses = gd_on_quadratic(H)
    conv_losses.append((name, losses))

print()

# \u2500\u2500 Plots \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Eigenvalues: Geometry, Power Iteration & Loss Landscape",
             fontsize=12, fontweight="bold")

# Plot 1: Unit circle transformed by A (shows stretching by eigenvalues)
theta = np.linspace(0, 2*np.pi, 300)
unit_circle = np.vstack([np.cos(theta), np.sin(theta)])
transformed = A @ unit_circle
axes[0].plot(unit_circle[0], unit_circle[1], "steelblue", lw=2, label="Unit circle", alpha=0.7)
axes[0].plot(transformed[0], transformed[1], "tomato", lw=2, label="A \u00d7 unit circle")
for i in range(2):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    axes[0].annotate("", xy=lam*v, xytext=[0,0],
                     arrowprops=dict(arrowstyle="->", color="black", lw=2))
    axes[0].annotate("", xy=-lam*v, xytext=[0,0],
                     arrowprops=dict(arrowstyle="->", color="black", lw=2))
    axes[0].text(*(lam*v*1.1), f"\u03bb{i+1}={lam:.2f}\\nv{i+1}", fontsize=8, ha="center")
axes[0].set_aspect("equal"); axes[0].grid(alpha=0.3)
axes[0].set_title("Eigenvalues = Stretching Factors\\n(eigenvectors = no-rotation dirs)")
axes[0].legend(fontsize=8)

# Plot 2: Power iteration convergence
axes[1].semilogy(errors, "steelblue", lw=2.5, label="|Rayleigh - \u03bb\u2081|")
theory_rate = [errors[0] * rate**(2*k) for k in range(len(errors))]
axes[1].semilogy(theory_rate, "tomato", linestyle="--", lw=2,
                 label=f"Theory: rate={rate:.3f}")
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Error (log scale)")
axes[1].set_title(f"Power Iteration Convergence\\nrate = |\u03bb\u2082/\u03bb\u2081| = {rate:.3f}")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

# Plot 3: GD convergence under different condition numbers
colors_gd = ["seagreen", "steelblue", "orange", "tomato"]
for (name, losses), color in zip(conv_losses, colors_gd):
    valid = [l for l in losses if l < 1e6 and l >= 0]
    if valid:
        axes[2].semilogy(valid, color=color, lw=2, label=name[:20])
axes[2].set_xlabel("Gradient descent step"); axes[2].set_ylabel("Loss (log scale)")
axes[2].set_title("GD Convergence: Condition Number Dominates")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eigenvalues_geometry.png", dpi=120)
print("  Plot saved \u2192 eigenvalues_geometry.png")
print()
print("  KEY TAKEAWAYS:")
print("  Eigenvectors = directions A does not rotate, just stretches")
print("  Power iteration converges at rate |\u03bb\u2082/\u03bb\u2081|: slow if spectrum is dense")
print("  GD convergence rate = (\u03ba-1)/(\u03ba+1): high \u03ba \u2192 slow convergence")
print("  Hessian eigenvalues diagnose minima (all +), maxima (all -), saddles (mixed)")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Projections & Least Squares — The Normal Equations Derived": {
        "description": (
            "Derive and demonstrate least squares as orthogonal projection. "
            "Show that the error vector is orthogonal to the column space. "
            "Compare solving via Normal Equations, QR, and SVD. "
            "Visualise the projection geometry in 3D."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


np.random.seed(42)
np.set_printoptions(precision=5, suppress=True)

print("=" * 65)
print("  PROJECTIONS & LEAST SQUARES: FROM GEOMETRY TO ALGORITHM")
print("=" * 65)
print()

# \u2500\u2500 Part 1: Projection onto a subspace \u2014 concrete example \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 1 \u2014 ORTHOGONAL PROJECTION ONTO COLUMN SPACE")
print()
# A \u2208 \u211d^{3\u00d72}: 3D space, 2D column space (a plane)
A = np.array([[1., 0.],
              [0., 1.],
              [1., 1.]])   # columns: e\u2081 and e\u2081+e\u2083 and e\u2082+e\u2083

b = np.array([1., 2., 3.])   # a vector NOT in the column space of A

print(f"  A (3\u00d72):\\n  {A}")
print(f"  b = {b}  (target, NOT in C(A))")
print()

# Projection matrix P = A(A\u1d40A)\u207b\u00b9A\u1d40
AtA = A.T @ A
AtA_inv = np.linalg.inv(AtA)
P = A @ AtA_inv @ A.T

# Projected vector
p = P @ b
e = b - p   # error / residual

print(f"  Projection matrix P = A(A\u1d40A)\u207b\u00b9A\u1d40:")
print(f"  {P.round(4)}")
print()
print(f"  p = Pb = {p.round(5)}   (projected vector, in C(A))")
print(f"  e = b-p = {e.round(5)}  (residual, orthogonal to C(A))")
print()

# Verify orthogonality: A\u1d40e = 0 (normal equations)
print(f"  Normal equation check \u2014 A\u1d40e = {(A.T @ e).round(10)}")
print(f"  (should be \u2248 0 \u2014 residual \u22a5 every column of A) \u2713")
print()

# Verify P\u00b2 = P and P\u1d40 = P
print(f"  Idempotent: P\u00b2 = P?  Max |P\u00b2\u2212P| = {np.abs(P@P - P).max():.2e} \u2713")
print(f"  Symmetric:  P\u1d40 = P?  Max |P\u1d40\u2212P| = {np.abs(P.T - P).max():.2e} \u2713")
print()

# \u2500\u2500 Part 2: Least Squares \u2014 Solving Three Ways \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 2 \u2014 LEAST SQUARES: THREE ALGORITHMS COMPARED")
print()

# Overdetermined system: 100 equations, 5 unknowns
n_pts, n_feat = 100, 5
X = np.random.randn(n_pts, n_feat)
true_w = np.array([3., -1., 2., 0.5, -2.])
b_ls = X @ true_w + 0.5*np.random.randn(n_pts)

print(f"  Overdetermined system: {n_pts} equations, {n_feat} unknowns")
print(f"  True weights: {true_w}")
print()

# Method 1: Normal equations (A\u1d40A)w = A\u1d40b \u2014 direct, can be numerically bad
XtX = X.T @ X
Xtb = X.T @ b_ls
w_normal = np.linalg.solve(XtX, Xtb)

# Method 2: QR factorisation \u2014 numerically stable
Q, R = np.linalg.qr(X)
w_qr = np.linalg.solve(R, Q.T @ b_ls)

# Method 3: SVD pseudoinverse \u2014 most general, handles rank deficiency
U_ls, s_ls, Vt_ls = np.linalg.svd(X, full_matrices=False)
w_svd = Vt_ls.T @ np.diag(1/s_ls) @ U_ls.T @ b_ls

# Method 4: numpy least squares (gold standard)
w_np, _, _, _ = np.linalg.lstsq(X, b_ls, rcond=None)

print(f"  {'Method':22s} | {'\u2016w\u2212w_true\u2016\u2082':>14} | {'\u2016Xw\u2212b\u2016\u2082':>12} | {'Residual \u22a5 X?':>14}")
print(f"  {'\u2500'*68}")
for name, w in [("Normal equations", w_normal), ("QR factorisation", w_qr),
                ("SVD pseudoinverse", w_svd), ("numpy lstsq", w_np)]:
    err      = np.linalg.norm(w - true_w)
    res_norm = np.linalg.norm(X @ w - b_ls)
    xt_res   = np.linalg.norm(X.T @ (X @ w - b_ls))
    print(f"  {name:22s} | {err:14.8f} | {res_norm:12.6f} | {xt_res:14.2e}")

print()
print("  All methods give the same answer. Differences:")
print("  Normal equations: O(n\u00b2p + p\u00b3). Bad conditioning: \u03ba(X\u1d40X) = \u03ba(X)\u00b2.")
print("  QR: O(np\u00b2). Stable, avoids squaring condition number. Preferred.")
print("  SVD: O(np\u00b2). Most robust, handles rank-deficient X. Most expensive.")
print()

# \u2500\u2500 Part 3: Residuals are orthogonal to predictions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 3 \u2014 RESIDUAL PROPERTIES")
print()
y_hat = X @ w_np
residuals = b_ls - y_hat

print(f"  \u2016residuals\u2016\u2082 = {np.linalg.norm(residuals):.5f}  (minimised by LS)")
print(f"  y_hat \u00b7 residuals = {y_hat @ residuals:.6f}  (\u22480: predictions \u22a5 residuals \u2713)")
print(f"  mean(residuals) = {residuals.mean():.6f}  (\u22480 since X has intercept) \u2713")
print()

# \u2500\u2500 Plots \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig = plt.figure(figsize=(16, 5))
fig.suptitle("Projections & Least Squares: Geometry and Algorithms",
             fontsize=12, fontweight="bold")

# Plot 1: 3D projection diagram
ax1 = fig.add_subplot(131, projection="3d")
# Visualise the column space (a plane) and projection
# Column space spanned by A[:, 0] and A[:, 1]
t1, t2 = np.linspace(-2, 2, 5), np.linspace(-2, 2, 5)
T1, T2 = np.meshgrid(t1, t2)
plane_pts = np.array([T1.ravel()*A[0,0] + T2.ravel()*A[0,1],
                      T1.ravel()*A[1,0] + T2.ravel()*A[1,1],
                      T1.ravel()*A[2,0] + T2.ravel()*A[2,1]])
ax1.plot_surface(plane_pts[0].reshape(5,5), plane_pts[1].reshape(5,5),
                 plane_pts[2].reshape(5,5), alpha=0.25, color="steelblue")
ax1.quiver(*[0,0,0], *b, color="tomato", arrow_length_ratio=0.1, linewidth=2, label="b")
ax1.quiver(*[0,0,0], *p, color="seagreen", arrow_length_ratio=0.1, linewidth=2, label="p=Pb")
ax1.quiver(*p, *e, color="purple", arrow_length_ratio=0.15, linewidth=2,
           linestyle="--", label="e=b-p")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
ax1.set_title("Projection onto\\nColumn Space (plane)")
ax1.legend(fontsize=8, loc="upper left")

# Plot 2: LS residual analysis
ax2 = fig.add_subplot(132)
ax2.scatter(y_hat, residuals, alpha=0.5, s=20, color="steelblue")
ax2.axhline(0, color="tomato", lw=2, linestyle="--", label="Residuals \u22a5 \u0177")
ax2.set_xlabel("Fitted values \u0177"); ax2.set_ylabel("Residuals e = y \u2212 \u0177")
ax2.set_title("Residual Plot\\n(random scatter confirms correct fit)")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
corr = np.corrcoef(y_hat, residuals)[0,1]
ax2.text(0.05, 0.95, f"corr(\u0177, e) = {corr:.6f}", transform=ax2.transAxes, fontsize=9)

# Plot 3: Condition number comparison (normal eqs vs QR)
np.random.seed(0)
kappas, err_normal, err_qr = [], [], []
for _ in range(30):
    # Create matrices with varying condition numbers
    U_r, _, Vt_r = np.linalg.svd(np.random.randn(20, 5))
    kappa = 10 ** np.random.uniform(0, 12)
    sv = np.geomspace(1, kappa, 5)
    X_test = U_r[:, :5] @ np.diag(sv) @ Vt_r
    b_test = np.random.randn(20)
    w_true_t = np.random.randn(5)
    b_test = X_test @ w_true_t + 0.01*np.random.randn(20)
    try:
        w_n = np.linalg.solve(X_test.T@X_test, X_test.T@b_test)
        Q_t, R_t = np.linalg.qr(X_test)
        w_q = np.linalg.solve(R_t, Q_t.T@b_test)
        kappas.append(kappa)
        err_normal.append(np.linalg.norm(w_n - w_true_t))
        err_qr.append(np.linalg.norm(w_q - w_true_t))
    except:
        pass

ax3 = fig.add_subplot(133)
ax3.loglog(kappas, err_normal, "tomato", marker="o", ms=4, alpha=0.7,
           linestyle="none", label="Normal equations \u03ba\u00b2")
ax3.loglog(kappas, err_qr, "steelblue", marker="s", ms=4, alpha=0.7,
           linestyle="none", label="QR factorisation \u03ba")
ax3.set_xlabel("Condition number \u03ba(X)")
ax3.set_ylabel("\u2016w_est \u2212 w_true\u2016\u2082")
ax3.set_title("Numerical Stability: Normal Eq. vs QR\\n\u03ba\u00b2 squaring hurts normal eqs at high \u03ba")
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "projections_least_squares.png", dpi=120)
print("  Plot saved \u2192 projections_least_squares.png")
print()
print("  KEY TAKEAWAYS:")
print("  Least squares = orthogonal projection of b onto C(A)")
print("  Normal equations A\u1d40e=0 are the orthogonality condition")
print("  QR avoids squaring \u03ba \u2014 always prefer QR over normal equations")
print("  SVD handles rank deficiency and gives minimum-norm solution")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Matrix Decompositions — LU, QR, Cholesky & Condition Numbers": {
        "description": (
            "Implement LU factorisation step-by-step via Gaussian elimination. "
            "Compare LU, QR, and Cholesky in terms of cost and numerical stability. "
            "Demonstrate condition number: how small perturbations to b "
            "can cause large changes in x = A⁻¹b for ill-conditioned A."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib as _pl, os as _os
_src = globals().get("__file__") or _os.path.abspath(".")
OUTPUT_DIR = _pl.Path(_src).resolve().parent.parent / "Resultant_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from scipy import linalg

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)

print("=" * 65)
print("  MATRIX DECOMPOSITIONS: LU, QR, CHOLESKY & CONDITION NUMBERS")
print("=" * 65)
print()

# \u2500\u2500 Part 1: LU Factorisation from scratch \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 1 \u2014 LU FACTORISATION (GAUSSIAN ELIMINATION)")
print()

A = np.array([[2., 1., 1.],
              [4., 3., 3.],
              [8., 7., 9.]], dtype=float)

print(f"  A:\\n  {A}")
print()

# Manual LU
L = np.eye(3)
U = A.copy()

# Step 1: eliminate column 1
m21 = U[1,0] / U[0,0];  U[1] -= m21 * U[0];  L[1,0] = m21
m31 = U[2,0] / U[0,0];  U[2] -= m31 * U[0];  L[2,0] = m31

# Step 2: eliminate column 2
m32 = U[2,1] / U[1,1];  U[2] -= m32 * U[1];  L[2,1] = m32

print(f"  L (lower triangular, stores multipliers):\\n  {L}")
print(f"  U (upper triangular, row echelon form):\\n  {U}")
print()
print(f"  Verification: L\u00b7U =\\n  {(L@U).round(4)}")
print(f"  Reconstruction error: {np.linalg.norm(L@U - A):.2e} \u2713")
print()
print(f"  det(A) = \u03a0 diag(U) = {U[0,0]:.1f} \u00d7 {U[1,1]:.1f} \u00d7 {U[2,2]:.1f} = "
      f"{U[0,0]*U[1,1]*U[2,2]:.1f}")
print(f"  np.linalg.det(A) = {np.linalg.det(A):.1f} \u2713")
print()

# Solve Ax = b via L and U
b_vec = np.array([4., 10., 20.])
# Forward substitution: Ly = b
y = np.linalg.solve(L, b_vec)
# Back substitution: Ux = y
x = np.linalg.solve(U, y)
print(f"  Solving Ax=b where b={b_vec}:")
print(f"  x = {x.round(4)}")
print(f"  Residual \u2016Ax\u2212b\u2016 = {np.linalg.norm(A@x - b_vec):.2e} \u2713")
print()

# \u2500\u2500 Part 2: QR Factorisation via Gram-Schmidt \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 2 \u2014 QR FACTORISATION (GRAM-SCHMIDT)")
print()

M = np.array([[1., 1., 0.],
              [1., 0., 1.],
              [0., 1., 1.]], dtype=float)

# Gram-Schmidt
a1, a2, a3 = M[:,0], M[:,1], M[:,2]
q1 = a1 / np.linalg.norm(a1)
a2_perp = a2 - (q1@a2)*q1
q2 = a2_perp / np.linalg.norm(a2_perp)
a3_perp = a3 - (q1@a3)*q1 - (q2@a3)*q2
q3 = a3_perp / np.linalg.norm(a3_perp)
Q = np.column_stack([q1, q2, q3])
R = Q.T @ M

print(f"  Q (orthonormal columns):\\n  {Q.round(4)}")
print(f"  R (upper triangular):\\n  {R.round(4)}")
print()
print(f"  Q\u1d40Q = I?  Max |Q\u1d40Q\u2212I| = {np.abs(Q.T@Q - np.eye(3)).max():.2e} \u2713")
print(f"  QR \u2248 M?   Max |QR\u2212M|  = {np.abs(Q@R - M).max():.2e} \u2713")
print()

# \u2500\u2500 Part 3: Cholesky Factorisation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 3 \u2014 CHOLESKY FACTORISATION (SPD MATRICES ONLY)")
print()
# Create SPD matrix
S = np.array([[4., 2., 1.],
              [2., 5., 2.],
              [1., 2., 6.]], dtype=float)
L_chol = linalg.cholesky(S, lower=True)
print(f"  S (SPD matrix):\\n  {S}")
print(f"  L (Cholesky factor, LL\u1d40 = S):\\n  {L_chol.round(4)}")
print(f"  LL\u1d40 \u2248 S?  Max |LL\u1d40\u2212S| = {np.abs(L_chol@L_chol.T - S).max():.2e} \u2713")
print()
print("  Cholesky is 2\u00d7 faster than LU for SPD matrices (O(n\u00b3/3) vs O(n\u00b3/2)).")
print("  Use for: covariance matrices, kernel matrices (GPs), normal equations.")
print()

# \u2500\u2500 Part 4: Condition Number and Numerical Stability \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("  PART 4 \u2014 CONDITION NUMBER: HOW BAD DATA AMPLIFIES TO BAD ANSWERS")
print()
print("  If \u03ba(A) \u2248 10^d, you lose roughly d digits of precision in x = A\u207b\u00b9b")
print()
print(f"  {'\u03ba(A)':>12} | {'\u2016\u0394b\u2016/\u2016b\u2016':>10} | {'\u2016\u0394x\u2016/\u2016x\u2016':>10} | {'Ratio (\u2264\u03ba?)':>12}")
print(f"  {'\u2500'*52}")

kappa_vals = [1e1, 1e3, 1e6, 1e9, 1e12]
for kappa_target in kappa_vals:
    # Create matrix with known condition number
    U_c, _, Vt_c = np.linalg.svd(np.random.randn(4, 4))
    sv_c = np.geomspace(kappa_target, 1, 4)
    A_c  = U_c @ np.diag(sv_c) @ Vt_c
    kappa_actual = sv_c[0] / sv_c[-1]
    x_true = np.random.randn(4)
    b_c    = A_c @ x_true
    # Small perturbation to b
    delta_b = 1e-6 * np.random.randn(4)
    x_perturbed = np.linalg.solve(A_c, b_c + delta_b)
    rel_b = np.linalg.norm(delta_b) / np.linalg.norm(b_c)
    rel_x = np.linalg.norm(x_perturbed - x_true) / np.linalg.norm(x_true)
    ratio = rel_x / rel_b if rel_b > 0 else np.inf
    print(f"  {kappa_actual:12.1e} | {rel_b:10.2e} | {rel_x:10.2e} | {ratio:12.2e}")

print()
print("  \u2016\u0394x\u2016/\u2016x\u2016 \u2264 \u03ba(A) \u00b7 \u2016\u0394b\u2016/\u2016b\u2016  \u2014 the condition number is the AMPLIFICATION FACTOR")
print()

# \u2500\u2500 Plots \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Matrix Decompositions: LU, QR, Cholesky & Condition Numbers",
             fontsize=12, fontweight="bold")

# Plot 1: L and U heatmaps
for ax_idx, (mat, title) in enumerate([(L, "L (lower)"), (U, "U (upper)")]):
    im = axes[0].imshow(np.abs(mat) if ax_idx == 0 else np.abs(mat),
                        cmap="Blues" if ax_idx == 0 else "Oranges",
                        aspect="auto", alpha=0.8 if ax_idx == 0 else 1.0)
axes[0].imshow(np.abs(L), cmap="Blues", aspect="auto")
axes[0].set_title("L factor (LU decomp)\\n(lower triangular)")
axes[0].set_xlabel("Column"); axes[0].set_ylabel("Row")
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f"{L[i,j]:.2f}", ha="center", va="center", fontsize=11)

# Plot 2: Singular values of Q showing orthogonality
Q_svs = np.linalg.svd(Q, compute_uv=False)
axes[1].bar(range(1, len(Q_svs)+1), Q_svs, color="steelblue", alpha=0.85)
axes[1].axhline(1.0, color="tomato", linestyle="--", lw=2, label="\u03c3=1 (orthonormal cols)")
axes[1].set_xlabel("Singular value index")
axes[1].set_ylabel("Singular value")
axes[1].set_title("Q is Orthonormal: All \u03c3\u1d62 = 1\\n(from Gram-Schmidt QR)")
axes[1].set_ylim(0, 1.5)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# Plot 3: Condition number vs error amplification
kappas_plot = np.logspace(1, 13, 50)
np.random.seed(1)
ratios = []
for kp in kappas_plot:
    U_t, _, Vt_t = np.linalg.svd(np.random.randn(5, 5))
    sv_t = np.geomspace(kp, 1, 5)
    A_t  = U_t @ np.diag(sv_t) @ Vt_t
    x_t  = np.random.randn(5)
    b_t  = A_t @ x_t
    db   = 1e-6 * np.random.randn(5)
    xp   = np.linalg.solve(A_t, b_t + db)
    rb   = np.linalg.norm(db) / np.linalg.norm(b_t)
    rx   = np.linalg.norm(xp - x_t) / (np.linalg.norm(x_t) + 1e-15)
    ratios.append(rx / rb if rb > 0 else np.nan)

axes[2].loglog(kappas_plot, ratios, "steelblue", lw=2, alpha=0.8, label="Actual amplification")
axes[2].loglog(kappas_plot, kappas_plot, "tomato", linestyle="--", lw=2, label="Bound: \u03ba(A)")
axes[2].set_xlabel("Condition number \u03ba(A)")
axes[2].set_ylabel("\u2016\u0394x\u2016/\u2016x\u2016 \u00f7 \u2016\u0394b\u2016/\u2016b\u2016")
axes[2].set_title("Condition Number = Error Amplification\\n(stays below \u03ba bound)")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "decompositions.png", dpi=120)
print("  Plot saved \u2192 decompositions.png")
print()
print("  KEY TAKEAWAYS:")
print("  LU: O(n\u00b3), general square systems; stores elimination multipliers")
print("  QR: numerically stable for least squares; avoids squaring \u03ba")
print("  Cholesky: 2\u00d7 faster for SPD; requires positive definite input")
print("  \u03ba(A) = \u03c3_max/\u03c3_min: the fundamental measure of numerical difficulty")
print("  \u03ba >> 10^d \u2192 d digits of precision lost in double precision")
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

# -------------------------------------------------------------------------
# Standalone runner
# Run:  python 01_Linear_Algebra.py
# Generates all plots and saves them to Resultant_Graphs/
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    _run_globals = {"__file__": __file__}
    for _op_name, _op in OPERATIONS.items():
        sep = "=" * 65
        print(f"\n{sep}\n  Running: {_op_name}\n{sep}")
        try:
            exec(_op["code"], _run_globals)
        except Exception as _e:
            print(f"  ERROR in {_op_name}: {_e}")
            traceback.print_exc()
    _out = _run_globals.get("OUTPUT_DIR", "(OUTPUT_DIR not set)")
    print(f"\n  All done. Plots saved to: {_out}")