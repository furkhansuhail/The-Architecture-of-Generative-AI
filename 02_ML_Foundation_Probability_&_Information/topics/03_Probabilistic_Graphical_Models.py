"""
Probabilistic Graphical Models
===============================

The language of structured uncertainty. Whenever variables are not
independent — disease affects symptoms, words cluster into topics, hidden
states drive observations — raw probability tables become unmanageably
large. PGMs encode conditional independence structure as a graph,
compressing exponentially large joint distributions into tractable,
interpretable pieces.

This module builds the complete theoretical foundation — from graph
theory and d-separation through exact inference, belief propagation,
variational methods, and the full learning problem.

"""

import textwrap
import re

TOPIC_NAME   = "Probabilistic Graphical Models"
DISPLAY_NAME = "03 · Probabilistic Graphical Models"
ICON         = "𝒢"
SUBTITLE     = "Structure, Inference, Learning — The Language of Structured Uncertainty"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### PART 1 — GRAPH THEORY FOUNDATIONS

### Graphs as Mathematical Objects

A GRAPH G = (V, E) consists of:
    V — a set of NODES (vertices), one per random variable
    E — a set of EDGES (links), encoding relationships

DIRECTED GRAPH (Digraph): edges are ordered pairs (u → v).
UNDIRECTED GRAPH:          edges are unordered pairs {u, v}.

Key graph concepts:

    PATH: a sequence of nodes v₁, v₂, ..., vₖ with edges between consecutive pairs.
    CYCLE: a path where v₁ = vₖ (returns to start).
    DAG (Directed Acyclic Graph): a directed graph with NO directed cycles.
          Every Bayesian network is a DAG.
    TREE: a connected undirected graph with no cycles (|E| = |V| − 1).
    CLIQUE: a subset C ⊆ V where every pair of nodes is connected by an edge.
    MAXIMAL CLIQUE: a clique that cannot be extended by adding another node.

    Diagram 1 — Graph Vocabulary:

    DIRECTED (DAG)        UNDIRECTED           FACTOR GRAPH
    A → B → D            A ─── B              ┌f₁┐   ┌f₂┐
    ↓       ↓            │ ╲   │              │  │   │  │
    C ──────╯            C ─── D              A──┤   ├─B──┤  ├─C
                                              └──┘   └──┘
    (no directed cycles)  (bidirectional)    (factors as squares)

For a directed graph, the PARENTS of node v:
    pa(v) = {u ∈ V : u → v ∈ E}

The CHILDREN of node v:
    ch(v) = {u ∈ V : v → u ∈ E}

The ANCESTORS of v: all nodes from which v is reachable.
The DESCENDANTS of v: all nodes reachable from v.
The NON-DESCENDANTS of v: V ∖ ({v} ∪ descendants(v)).

MORAL GRAPH of a DAG: the undirected graph obtained by:
    1. Connecting all pairs of parents that share a common child ("marrying" parents)
    2. Dropping edge directions

Moralisation converts directed to undirected independence structure,
used in the junction tree algorithm for exact inference.


### Why Graphs Encode Independence

The JOINT DISTRIBUTION over n binary variables requires 2ⁿ − 1 parameters.
    n = 30: over 10⁹ parameters — completely infeasible.

A PGM uses graph structure to decompose the joint into LOCAL FACTORS:
    p(x₁,...,xₙ) = ∏ᵢ p(xᵢ | pa(xᵢ))   (Bayesian network)
    p(x₁,...,xₙ) ∝ ∏_C ψ_C(x_C)          (Markov random field)

Each factor depends on only a small subset of variables. If each node
has at most k parents, the total parameters scale as n · 2ᵏ, which is
POLYNOMIAL rather than exponential in n.

    ┌──────────────────────────────────────────────────────────┐
    │  The graph is not just a drawing — it IS the model.      │
    │  Every missing edge is an INDEPENDENCE ASSUMPTION,       │
    │  every present edge is a DEPENDENCE allowed by the model.│
    └──────────────────────────────────────────────────────────┘


### PART 2 — BAYESIAN NETWORKS (DIRECTED GRAPHICAL MODELS)

### Definition and Factorisation

A BAYESIAN NETWORK is a pair (G, P) where:
    G = (V, E) is a DAG over variables X₁,...,Xₙ
    P is a set of conditional probability distributions (CPDs)

The joint distribution FACTORISES as:

    p(x₁,...,xₙ) = ∏ᵢ₌₁ⁿ p(xᵢ | pa(xᵢ))

Each variable is conditionally independent of its non-descendants
given its parents. This is the MARKOV CONDITION, the defining property
of Bayesian networks.

The factorisation is VALID for any joint distribution if the graph
captures all dependences — but a specific graph may impose independence
assumptions that are empirically verifiable.

    Example — Medical Diagnosis Network:
    (Classic "Asia" network, simplified)

    Smoking → LungCancer → Dyspnoea
         ↓          ↓        ↑
    Bronchitis   XRay     (both)

    p(S, L, B, X, D) = p(S) · p(L|S) · p(B|S) · p(X|L) · p(D|L,B)

    With 5 binary variables:
    Naive table: 2⁵ − 1 = 31 parameters
    Factorised: 1 + 2 + 2 + 2 + 4 = 11 parameters

The reduction from 31 to 11 comes entirely from independence assumptions
encoded in the graph. With many variables and sparse graphs, the savings
are exponential.


### Conditional Independence in Bayesian Networks

Three CANONICAL STRUCTURES (for three nodes A, B, C):

    1. CHAIN:       A → B → C
       A ⊥⊥ C | B   (B "blocks" the path — C's past is irrelevant given present)
       A not ⊥⊥ C (marginally dependent)

    2. FORK:        A ← B → C
       A ⊥⊥ C | B   (B is the common cause; conditioning removes dependence)
       A not ⊥⊥ C (marginally dependent — correlated through B)

    3. COLLIDER:    A → B ← C
       A ⊥⊥ C       (marginally independent — no common cause)
       A not ⊥⊥ C | B  (conditioning on the EFFECT creates dependence!)

    ┌──────────────────────────────────────────────────────────┐
    │  COLLIDERS ARE COUNTERINTUITIVE:                         │
    │  Example: Intelligence → ExamScore ← Difficulty          │
    │  Intelligence and Difficulty are independent.            │
    │  But given ExamScore (observed), knowing Difficulty      │
    │  tells you about Intelligence (explaining away).         │
    │  This is "Berkson's paradox" in epidemiology.            │
    └──────────────────────────────────────────────────────────┘


### D-Separation — The Fundamental Independence Criterion

A path π between nodes A and B is BLOCKED by evidence set Z if:

    (a) The path contains a CHAIN (→ M →) or FORK (← M →) where M ∈ Z
        (conditioning on M blocks the flow of information)

    (b) The path contains a COLLIDER (→ M ←) where M ∉ Z AND no
        descendant of M is in Z
        (a non-observed collider blocks the path)

    Note: if M is a COLLIDER and M ∈ Z (or a descendant of M is in Z),
    the path is UNBLOCKED — this is the "explaining away" effect.

D-SEPARATION: A and B are d-separated by Z (written A ⊥_d B | Z) if
EVERY path between A and B is blocked by Z.

    D-SEPARATION ⟹ CONDITIONAL INDEPENDENCE in any distribution
    consistent with the graph:

        A ⊥_d B | Z  ⟹  A ⊥⊥ B | Z

FAITHFULNESS: The converse — independence implies d-separation — holds
under the faithfulness assumption (generic parameter values). Most
structure learning algorithms assume faithfulness.

    Diagram 2 — D-Separation Examples:

    Graph: A → B → C ← D

    Is A ⊥_d D | ∅ ?
        Path A → B → C ← D:
        C is a COLLIDER (→ C ←), not in Z=∅, and no descendant in Z.
        → Path BLOCKED. So A ⊥_d D | ∅ → A ⊥⊥ D ✓

    Is A ⊥_d D | C ?
        Path A → B → C ← D:
        C is a COLLIDER, C ∈ Z → path UNBLOCKED.
        So A NOT d-sep from D given C → A not ⊥⊥ D | C ✓


### The I-Map and Markov Equivalence

The set of all conditional independences encoded by a DAG G is I(G).

A DAG G is an I-MAP of distribution P if:
    I(G) ⊆ I(P)   (every independence in G holds in P)

A DAG G is a PERFECT MAP if  I(G) = I(P)  (captures exactly the right independences).
Not every distribution has a perfect DAG map.

MARKOV EQUIVALENCE: Two DAGs G₁ and G₂ are Markov equivalent if
they encode exactly the same set of conditional independences.
They have the same SKELETON (undirected edges) and the same IMMORALITIES
(v-structures: A → C ← B with no A−B edge).

Equivalent DAGs form a MARKOV EQUIVALENCE CLASS (MEC),
represented by a COMPLETED PARTIALLY DIRECTED ACYCLIC GRAPH (CPDAG).
Structure learning from observational data can identify the MEC,
not the exact DAG — interventional data is needed to orient edges further.


### PART 3 — MARKOV RANDOM FIELDS (UNDIRECTED GRAPHICAL MODELS)

### Definition and Gibbs Distribution

A MARKOV RANDOM FIELD (MRF), also called a Markov network, is an
undirected graph G = (V, E) with potential functions on cliques.

The joint distribution is the GIBBS DISTRIBUTION:

    p(x₁,...,xₙ) = (1/Z) · ∏_C ψ_C(x_C)

    ψ_C(x_C) — CLIQUE POTENTIAL (non-negative function on clique C)
    Z = Σ_x ∏_C ψ_C(x_C) — PARTITION FUNCTION (normalisation constant)

The log potentials  φ_C(x_C) = log ψ_C(x_C)  are called ENERGY TERMS.
The energy of a configuration:

    E(x) = −Σ_C φ_C(x_C) = −log[∏_C ψ_C(x_C)]

So:   p(x) = (1/Z) · exp(−E(x))   — the BOLTZMANN distribution.

    ┌──────────────────────────────────────────────────────────┐
    │  Computing Z requires summing over all configurations:   │
    │  Z = Σ_x exp(−E(x))  — exponentially many terms.         │
    │  THIS is the central computational challenge in MRFs.    │
    │  Most inference algorithms avoid computing Z directly.   │
    └──────────────────────────────────────────────────────────┘


### Markov Properties of MRFs

An MRF satisfies three equivalent Markov properties:

    PAIRWISE MARKOV: Xᵢ ⊥⊥ Xⱼ | X_{V ∖ {i,j}}
        Non-adjacent variables are conditionally independent given all others.

    LOCAL MARKOV: Xᵢ ⊥⊥ X_{V ∖ ne(i)} | X_{ne(i)}
        A variable is conditionally independent of all non-neighbours
        given its NEIGHBOURHOOD ne(i).

    GLOBAL MARKOV: XA ⊥⊥ XB | XS
        for all disjoint sets A, B, S where S SEPARATES A from B in G.
        (No path from A to B in G ∖ S)

    Pairwise ⟹ Local ⟹ Global (in general)
    For positive distributions (Hammersley-Clifford): all three are equivalent.

HAMMERSLEY-CLIFFORD THEOREM: A strictly positive distribution P factorises
according to undirected graph G (as a Gibbs distribution with clique potentials)
if and only if P satisfies the global Markov property with respect to G.

This theorem is the undirected analogue of the Bayesian network factorisation.


### Comparing Directed and Undirected Models

    ┌────────────────────┬───────────────────────┬────────────────────────┐
    │                    │  BAYESIAN NETWORK     │  MARKOV RANDOM FIELD   │
    ├────────────────────┼───────────────────────┼────────────────────────┤
    │ Graph type         │ DAG (directed)        │ Undirected             │
    │ Factorisation      │ ∏ p(xᵢ|pa(xᵢ))        │ (1/Z) ∏ ψ_C(x_C)       │
    │ Normalisation      │ Automatic (CPDs)      │ Z required (hard!)     │
    │ Independence       │ D-separation          │ Graph separation       │
    │ Causal direction   │ Natural (arrows)      │ None                   │
    │ Cyclic structure   │ Not allowed           │ Allowed                │
    │ Strengths          │ Generative models,    │ Symmetric relations,   │
    │                    │ causal reasoning      │ image models, CRFs     │
    │ Examples           │ Naive Bayes, HMM,     │ Ising model, CRF,      │
    │                    │ VAE, LDA              │ Boltzmann Machine      │
    └────────────────────┴───────────────────────┴────────────────────────┘

CONVERTING BETWEEN REPRESENTATIONS:
    BN → MRF: Moralise (marry parents, drop directions). Information may be lost.
    MRF → BN: Triangulate, then orient. Always possible but may add spurious edges.
    Neither direction is lossless in general.


### Factor Graphs — A Unified Representation

A FACTOR GRAPH is a bipartite graph between VARIABLE NODES and FACTOR NODES:

    Variables: circles ○
    Factors:   squares □

    A factor fₛ connects to all variables in its scope S.

    p(x) = (1/Z) ∏ₛ fₛ(x_S)

Factor graphs make the factorisation EXPLICIT. Both BNs and MRFs are
special cases of factor graphs. The sum-product algorithm (belief propagation)
is most naturally stated on factor graphs.

    Diagram 3 — Factor Graph for p(a,b,c,d) = f₁(a,b) · f₂(b,c) · f₃(c,d):

    ○_a — □_{f₁} — ○_b — □_{f₂} — ○_c — □_{f₃} — ○_d

    (a chain factor graph — belief propagation is exact on trees)


### PART 4 — EXACT INFERENCE: VARIABLE ELIMINATION

### The Inference Problem

Given a PGM, the fundamental inference tasks are:

    MARGINAL INFERENCE:  Compute p(xᵢ) or p(x_A) for a subset A ⊆ V.
    CONDITIONAL INFERENCE: Compute p(x_Q | x_E = e) — query given evidence.
    MAP/MPE INFERENCE: Find x* = argmax_x p(x | e).

Exact inference is #P-hard in general (as hard as counting satisfying
assignments). Tractable only for special graph structures (trees, bounded
treewidth).

### Variable Elimination Algorithm

The KEY IDEA: push summations inside products using distributivity.

Instead of computing Σ_{x ∖ {xᵢ}} ∏ⱼ p(xⱼ|pa(xⱼ)) naively (exponential),
we sum out variables one at a time:

    VARIABLE ELIMINATION (VE) for marginal p(x_Q):

    1. Pick an elimination ORDER π = (v₁, v₂, ..., v_{n-|Q|}) of non-query vars.
    2. For each vᵢ in order:
       a. Collect all factors that mention vᵢ → {f₁, ..., fₖ}
       b. Compute new factor:  τ(x_{scope ∖ vᵢ}) = Σ_{xᵥᵢ} ∏ⱼ fⱼ
       c. Remove {f₁,...,fₖ}; add τ to the factor set.
    3. Multiply remaining factors; normalise.

    ┌──────────────────────────────────────────────────────────┐
    │  VE is the foundational algorithm — it is EXACT.         │
    │  Complexity: exponential in the INDUCED WIDTH (treewidth)│
    │  of the graph under elimination ordering π.              │
    │  Finding the optimal ordering is NP-hard.                │
    └──────────────────────────────────────────────────────────┘

Example: Chain A → B → C. Query: p(C).
    p(C) = Σ_A Σ_B p(A) · p(B|A) · p(C|B)
         = Σ_B p(C|B)   · [Σ_A p(A) · p(B|A)]
                           ──────────────────
                              = τ(B) = p(B)
         = Σ_B p(C|B) · p(B) = p(C)

Two operations instead of the naive product-then-sum.

MAP INFERENCE (max-product / Viterbi):
    Replace Σ with max in the VE algorithm.
    Backtracking recovers the argmax configuration.

    The Viterbi algorithm for HMMs is exactly this — max-product VE
    on a chain graph.


### Treewidth and Complexity

The INDUCED WIDTH of an elimination ordering π is:
    max over all eliminated nodes vᵢ of the number of earlier
    neighbours vᵢ has when it is eliminated.

The TREEWIDTH of a graph G:
    w(G) = min over all orderings π of the induced width under π.

Inference complexity is O(n · r^{w+1}) where r = max domain size.
    Treewidth 1: tree — O(n) (belief propagation)
    Treewidth 2: e.g., grid with one extra connection
    Treewidth k: exponential in k — quickly intractable
    Random dense graphs: treewidth ≈ n (fully intractable)

For images, grids, and social networks: treewidth is typically large.
→ APPROXIMATE inference is necessary.


### PART 5 — BELIEF PROPAGATION

### Sum-Product on Trees (Exact)

BELIEF PROPAGATION (BP) passes MESSAGES along edges.
On TREES, it computes exact marginals in O(n) time.

For a factor graph with variable nodes x and factor nodes f:

    MESSAGE from variable xᵢ to factor fₛ:
        μ_{xᵢ→fₛ}(xᵢ) = ∏_{fₜ ∈ ne(xᵢ) ∖ {fₛ}} μ_{fₜ→xᵢ}(xᵢ)

    MESSAGE from factor fₛ to variable xᵢ:
        μ_{fₛ→xᵢ}(xᵢ) = Σ_{x_{S ∖ {i}}} fₛ(x_S) · ∏_{xⱼ ∈ ne(fₛ) ∖ {xᵢ}} μ_{xⱼ→fₛ}(xⱼ)

    BELIEF (unnormalised marginal) at variable xᵢ:
        b(xᵢ) ∝ ∏_{fₛ ∈ ne(xᵢ)} μ_{fₛ→xᵢ}(xᵢ)

SCHEDULE on a tree (two passes):
    1. Choose a root. Messages flow INWARD (leaves → root).
    2. Messages flow OUTWARD (root → leaves).
    After both passes: all beliefs are exact marginals.

    Diagram 4 — Message Passing on a Tree:

                    ○ B (root)
                   ╱ ╲
                  ○A   ○C
                 ╱         ╲
                ○D           ○E

    Leaf D sends μ_{D→A}, then A sends μ_{A→B} (inward pass).
    B sends μ_{B→A} (outward pass), then A sends μ_{A→D}.
    Each node computes its belief using all received messages.

For undirected trees (two-variable factors ψ_{ij}):

    μ_{i→j}(xⱼ) = Σ_{xᵢ} ψ_{ij}(xᵢ,xⱼ) · ψᵢ(xᵢ) · ∏_{k ∈ ne(i) ∖ {j}} μ_{k→i}(xᵢ)

    bᵢ(xᵢ) ∝ ψᵢ(xᵢ) · ∏_{j ∈ ne(i)} μ_{j→i}(xᵢ)


### Loopy Belief Propagation (Approximate)

For graphs with CYCLES, run BP anyway — it is no longer exact but
often converges to good approximate marginals.

LOOPY BP schedule:
    1. Initialise all messages to 1 (or randomly).
    2. Repeat until convergence (message change < ε):
       Update each message using the BP equations.
    3. Compute beliefs from converged messages.

Properties of loopy BP:
    · Not guaranteed to converge (but often does in practice).
    · If it converges, fixed points are BETHE FREE ENERGY stationary points.
    · Exact on trees; approximate on graphs with cycles.
    · Good approximation for sparse graphs, weak coupling.
    · Used in LDPC decoding (turbo codes), Ising models, vision.

The BETHE FREE ENERGY (approximation to the true free energy):

    F_Bethe(b) = Σ_i H_i(bᵢ) + Σᵢⱼ Hᵢⱼ(bᵢⱼ) − Σ_ij I_ij(bᵢ,bⱼ)

Loopy BP minimises F_Bethe — it is a variational method in disguise.

MAX-PRODUCT BP: Replace Σ with max in factor-to-variable messages.
    Computes approximate MAP (exact on trees, where it is the Viterbi algorithm).


### The Junction Tree Algorithm (Exact on Any Graph)

For graphs where loopy BP would be inexact, the JUNCTION TREE ALGORITHM
(also called Clique Tree, or Junction Graph) computes EXACT marginals:

STEPS:
    1. MORALISE the DAG (or skip for MRF).
    2. TRIANGULATE the graph (add edges to make every cycle of length ≥4
       have a chord; the resulting graph is CHORDAL).
    3. Identify MAXIMAL CLIQUES of the triangulated graph.
    4. Build a CLIQUE TREE (junction tree) connecting cliques.
       The RUNNING INTERSECTION PROPERTY must hold:
       For any two cliques Cᵢ, Cⱼ, every clique on the path between them
       contains Cᵢ ∩ Cⱼ.
    5. Run belief propagation on the (now acyclic) clique tree.

    ┌──────────────────────────────────────────────────────────┐
    │  The junction tree transforms ANY inference problem      │
    │  into exact BP on a tree of CLIQUES.                     │
    │  Complexity: exponential in the TREEWIDTH.               │
    │  The algorithm is optimal: no exact algorithm can be     │
    │  significantly better for high-treewidth graphs.         │
    └──────────────────────────────────────────────────────────┘

Triangulation step is key: adds edges to make the graph chordal.
The width of the resulting junction tree = treewidth + 1.
Minimum-width triangulation is NP-hard; greedy heuristics are used.


### PART 6 — APPROXIMATE INFERENCE: VARIATIONAL METHODS

### The Variational Principle

When exact inference is intractable, VARIATIONAL INFERENCE replaces
the intractable posterior p(z | x) with a simpler distribution q(z)
from a tractable family Q, by minimising:

    D_KL(q(z) ‖ p(z | x)) = ∫ q(z) log [q(z) / p(z|x)] dz

Since D_KL ≥ 0 with equality iff q = p, minimising this finds the
best approximation q* ∈ Q to the true posterior.

The EVIDENCE LOWER BOUND (ELBO):

    log p(x) = ELBO(q) + D_KL(q(z) ‖ p(z|x))

    ELBO(q) = 𝔼_q[log p(x, z)] − 𝔼_q[log q(z)]
            = 𝔼_q[log p(x|z)] − D_KL(q(z) ‖ p(z))

    ┌──────────────────────────────────────────────────────────┐
    │  Maximising ELBO ⟺ Minimising D_KL(q ‖ p(z|x))          │
    │  because log p(x) is constant w.r.t. q.                  │
    │                                                          │
    │  ELBO = (reconstruction term) − (KL regularisation)      │
    │  This is the VAE objective — VI at the heart of          │
    │  modern generative modelling.                            │
    └──────────────────────────────────────────────────────────┘


### Mean Field Variational Inference

The MEAN FIELD family: q factorises completely across variables:

    q(z) = ∏ᵢ qᵢ(zᵢ)

All latent variables are treated as independent in the approximation.
This is the crudest but most tractable factorisation.

COORDINATE ASCENT VARIATIONAL INFERENCE (CAVI):
    Update each factor qᵢ while holding others fixed:

    log q*ᵢ(zᵢ) = 𝔼_{q_{-i}}[log p(x, z)] + const

    q*ᵢ(zᵢ) ∝ exp(𝔼_{q_{-i}}[log p(x, z)])

CAVI iterates these updates until the ELBO converges.
Guaranteed to converge to a LOCAL maximum of ELBO.

For exponential family models, CAVI updates are CLOSED FORM:
    The optimal qᵢ is in the same exponential family as the complete
    conditional p(zᵢ | z_{-i}, x).

    ┌──────────────────────────────────────────────────────────┐
    │  Mean field: captures the "shape" of each marginal       │
    │  but CANNOT capture posterior correlations.              │
    │  Structured mean field: factorises over groups of vars,  │
    │  preserving within-group correlations.                   │
    │  Full-rank VI (ADVI): q is a multivariate Gaussian.      │
    └──────────────────────────────────────────────────────────┘


### Stochastic Variational Inference

For large datasets, CAVI requires iterating over all data points.
STOCHASTIC VI uses mini-batches and stochastic gradient ascent:

    ĝ = ∇_λ ELBO(λ)  ≈  (n/|S|) Σ_{i∈S} ∇_λ [𝔼_q[log p(xᵢ, z)]] − ∇_λ 𝔼_q[log q(z)]

    where S is a mini-batch of size |S|.

The REPARAMETERISATION TRICK (for continuous z):
    Sample z ~ q_φ(z|x) by writing z = g(ε, φ) where ε ~ p(ε) is noise.
    Then: ∇_φ 𝔼_{q_φ}[f(z)] = 𝔼_{ε}[∇_φ f(g(ε,φ))]  (differentiable!)

Example for q = N(μ, σ²): z = μ + σε, ε ~ N(0,1).
This makes the VAE encoder differentiable end-to-end.

SCORE FUNCTION ESTIMATOR (REINFORCE):
    ∇_φ 𝔼_{q_φ}[f(z)] = 𝔼_{q_φ}[f(z) ∇_φ log q_φ(z)]
    Works for DISCRETE z; higher variance than reparameterisation.


### PART 7 — HIDDEN MARKOV MODELS

### The HMM as a Directed PGM

A HIDDEN MARKOV MODEL has:
    Hidden states:  Z₁, Z₂, ..., Z_T    (unobserved, Markov chain)
    Observations:   X₁, X₂, ..., X_T    (observed, conditionally independent given Z)

PARAMETERS:
    π = {πₖ = p(Z₁=k)}                  INITIAL DISTRIBUTION
    A = {aₖₗ = p(Zₜ=l | Zₜ₋₁=k)}      TRANSITION MATRIX
    B = {bₖ(x) = p(Xₜ=x | Zₜ=k)}       EMISSION DISTRIBUTION

JOINT DISTRIBUTION:
    p(x₁:T, z₁:T) = π_{z₁} · ∏ₜ₌₂ᵀ a_{zₜ₋₁,zₜ} · ∏ₜ₌₁ᵀ b_{zₜ}(xₜ)

    Diagram 5 — HMM as a Directed PGM:

    Z₁ ──→ Z₂ ──→ Z₃ ──→ Z₄    (Markov chain, hidden)
    │      │      │      │
    ↓      ↓      ↓      ↓
    X₁     X₂     X₃     X₄    (observations)

The Markov property holds: Zₜ ⊥⊥ Z₁:ₜ₋₂ | Zₜ₋₁
(the past is summarised by the current state).


### The Forward-Backward Algorithm (Belief Propagation on the Chain)

FORWARD VARIABLE: αₜ(k) = p(x₁:ₜ, Zₜ=k)

    INITIALISATION: α₁(k) = πₖ · bₖ(x₁)
    RECURSION:      αₜ(k) = bₖ(xₜ) · Σⱼ αₜ₋₁(j) · aⱼₖ

BACKWARD VARIABLE: βₜ(k) = p(xₜ₊₁:T | Zₜ=k)

    INITIALISATION: βᵀ(k) = 1
    RECURSION:      βₜ(k) = Σⱼ aₖⱼ · bⱼ(xₜ₊₁) · βₜ₊₁(j)

SMOOTHED MARGINAL (posterior):
    p(Zₜ=k | x₁:T) ∝ αₜ(k) · βₜ(k)

PAIRWISE MARGINAL:
    p(Zₜ₋₁=j, Zₜ=k | x₁:T) ∝ αₜ₋₁(j) · aⱼₖ · bₖ(xₜ) · βₜ(k)

LIKELIHOOD:   p(x₁:T) = Σₖ αᵀ(k)   (sum over final forward variables)

The forward-backward algorithm is exactly sum-product BP on the chain.
Complexity: O(T·K²) where K is the number of states.

    ┌──────────────────────────────────────────────────────────┐
    │  SCALING: αₜ values decay exponentially. In practice,     │
    │  normalise at each time step and track log-likelihoods.  │
    │  NUMERICAL STABILITY: always work in log space for       │
    │  long sequences. Use log-sum-exp trick:                  │
    │      log Σᵢ exp(aᵢ) = max(a) + log Σᵢ exp(aᵢ − max(a))   │
    └──────────────────────────────────────────────────────────┘


### The Viterbi Algorithm (MAP Inference)

Finds the most probable state sequence:

    z* = argmax_{z₁:T} p(z₁:T | x₁:T)

Replace Σ (sum) with max in the forward pass:

    VITERBI VARIABLE: δₜ(k) = max_{z₁:ₜ₋₁} p(z₁:ₜ, Zₜ=k | x₁:ₜ)

    INITIALISATION: δ₁(k) = πₖ · bₖ(x₁)
    RECURSION:      δₜ(k) = bₖ(xₜ) · max_j [δₜ₋₁(j) · aⱼₖ]

    BACKPOINTER:    ψₜ(k) = argmax_j [δₜ₋₁(j) · aⱼₖ]

BACKTRACKING: z*ᵀ = argmax_k δᵀ(k), then z*ₜ = ψₜ₊₁(z*ₜ₊₁).

Note: The Viterbi path ≠ sequence of most likely states.
    The argmax of the joint ≠ sequence of marginal argmaxes.


### The Baum-Welch Algorithm (EM for HMMs)

Learns parameters (π, A, B) from UNLABELLED observations x₁:T.

EXPECTATION-MAXIMISATION (EM):
    E-STEP: Run forward-backward to compute posterior state probabilities.
        γₜ(k) = p(Zₜ=k | x₁:T; θ_old)
        ξₜ(j,k) = p(Zₜ₋₁=j, Zₜ=k | x₁:T; θ_old)

    M-STEP: Re-estimate parameters using posterior expectations as soft counts.
        π̂ₖ = γ₁(k)
        Âⱼₖ = Σₜ₌₂ᵀ ξₜ(j,k) / Σₜ₌₁ᵀ⁻¹ γₜ(j)
        B̂ₖ(v) = Σₜ:xₜ=v γₜ(k) / Σₜ γₜ(k)

EM GUARANTEES: Likelihood p(x|θ) is non-decreasing at every iteration.
Converges to a LOCAL maximum of the likelihood.


### PART 8 — PARAMETER LEARNING IN PGMs

### MLE for Bayesian Networks (Complete Data)

With COMPLETE DATA (all variables observed in every sample):

The log-likelihood of a BN DECOMPOSES over nodes:

    ℓ(θ) = Σᵢ Σ_{xᵢ, pa(xᵢ)} N(xᵢ, pa(xᵢ)) · log θ_{xᵢ | pa(xᵢ)}

where N(xᵢ, pa(xᵢ)) is the empirical count of configuration (xᵢ, pa(xᵢ)).

Each CPD can be optimised INDEPENDENTLY. For a discrete variable:

    θ̂_{xᵢ | pa(xᵢ)} = N(xᵢ, pa(xᵢ)) / N(pa(xᵢ))  = empirical conditional frequency

    KEY: Complete data → CLOSED FORM MLE. One pass through the data suffices.

For Gaussian BNs (linear Gaussian):
    p(xᵢ | pa(xᵢ)) = N(μᵢ + βᵢᵀ pa(xᵢ), σᵢ²)
    MLE = ordinary least squares regression of xᵢ on pa(xᵢ).


### Bayesian Parameter Estimation

With Dirichlet priors on multinomial CPDs:

    θ_{xᵢ|pa(xᵢ)} ~ Dirichlet(α)   (prior)

    Posterior (Dirichlet-Multinomial conjugacy):
    θ_{xᵢ|pa(xᵢ)} | data ~ Dirichlet(α + N(xᵢ, pa(xᵢ)))

    Predictive (posterior mean):
    p̂(xᵢ | pa(xᵢ)) = (N(xᵢ, pa(xᵢ)) + αᵢ) / (N(pa(xᵢ)) + Σⱼ αⱼ)

LAPLACE SMOOTHING: α = 1 for all values (uniform Dirichlet).
    Prevents zero probabilities for unseen combinations.
    Equivalent to adding one pseudo-count to each cell.


### EM for Incomplete Data / Latent Variables

When some variables are UNOBSERVED (latent), the likelihood no longer
decomposes and MLE has no closed form.

EM ALGORITHM (general):

    E-STEP: Compute Q(θ | θ_old) = 𝔼_{p(z|x,θ_old)}[log p(x,z|θ)]
    M-STEP: θ_new = argmax_θ Q(θ | θ_old)

EM GUARANTEES:
    · ℓ(θ_new) ≥ ℓ(θ_old): likelihood is non-decreasing.
    · Converges to a STATIONARY POINT (local max or saddle).
    · Not guaranteed to find the global maximum.
    · Convergence can be slow near the optimum (linear rate).

WHY EM WORKS (Jensen's inequality):
    log p(x|θ) = log Σ_z p(x,z|θ)
               ≥ Σ_z q(z) log [p(x,z|θ)/q(z)]  (by Jensen, concavity of log)
               = Q(θ | θ_old) + H(q)  =: ELBO(θ, q)

Setting q(z) = p(z|x,θ_old) makes the bound TIGHT (E-step).
The M-step then maximises the bound — guaranteed ascent.

    ┌──────────────────────────────────────────────────────────┐
    │  EM = Coordinate Ascent on the ELBO.                     │
    │  E-step: optimise q (making bound tight at θ_old)        │
    │  M-step: optimise θ (raising the bound for fixed q)      │
    │  Variational EM: E-step uses approximate inference.      │
    └──────────────────────────────────────────────────────────┘


### Structure Learning

Given data, learn the GRAPH STRUCTURE G as well as parameters.

SCORE-BASED METHODS: Define a score S(G, data) and search over DAGs.
    BIC (Bayesian Information Criterion):
        BIC(G) = ℓ(G; data) − (d/2) · log n
        Penalises complexity; consistent (recovers true G as n→∞).

    BDe SCORE (Bayesian Dirichlet equivalent):
        Marginal likelihood under Dirichlet priors; score-decomposable.

Search strategy: greedy hill climbing (add/remove/reverse edge),
tabu search, or dynamic programming (exact for small n).

CONSTRAINT-BASED METHODS: Test conditional independences and
build the CPDAG consistent with observed independences.
    PC ALGORITHM:
        1. Start with complete undirected graph.
        2. Remove edge (i,j) if ∃S: i ⊥⊥ j | S (test CI).
        3. Orient v-structures (immoralities).
        4. Apply Meek rules to orient remaining edges.
    FCI ALGORITHM: Handles hidden common causes and selection bias.

HYBRID METHODS (MMHC, GES): combine CI tests with score optimisation.


### PART 9 — LATENT DIRICHLET ALLOCATION (LDA)

### LDA as a Bayesian PGM

LATENT DIRICHLET ALLOCATION (Blei, Ng, Jordan 2003) is a hierarchical
Bayesian model for document collections.

GENERATIVE PROCESS for document d with Nᵈ words:
    1. Draw topic proportions: θᵈ ~ Dirichlet(α)  (distribution over K topics)
    2. For each word position n:
       a. Draw topic:         zₙᵈ ~ Multinomial(θᵈ)
       b. Draw word:          wₙᵈ ~ Multinomial(βᵢ)  where i = zₙᵈ
    Global: Each topic k is a distribution over vocabulary: βₖ ~ Dirichlet(η)

JOINT DISTRIBUTION:
    p(w, z, θ, β | α, η) = [∏_d p(θᵈ|α) · ∏_n p(zₙᵈ|θᵈ) · p(wₙᵈ|β_{zₙᵈ})]
                           · ∏_k p(βₖ|η)

INFERENCE (intractable): The posterior p(z, θ, β | w) has no closed form
because the normaliser involves summing over all topic assignments.

    LDA INFERENCE ALGORITHMS:
    · Variational EM (original paper): mean field q(θ,z,β).
    · Collapsed Gibbs sampling: integrate out θ and β analytically,
      sample z only. Conditional:
          p(zₙᵈ = k | z_{-n}, w) ∝ (N_{-n,k}ᵈ + α) · (N_{-n,kv} + η) / (N_{-n,k} + Vη)
      where N_{-n,k}ᵈ = #{words in doc d assigned topic k, excluding n}
            N_{-n,kv} = #{times word v assigned topic k, excluding n}
    · Online VB (Hoffman 2010): processes mini-batches, scales to millions of docs.

    ┌──────────────────────────────────────────────────────────┐
    │  LDA as a PGM:                                           │
    │                                                          │
    │  α ──→ θᵈ ──→ zₙᵈ ──→ wₙᵈ ←── βₖ ←── η                     │
    │  (plate over D docs)   (plate over Nᵈ words per doc)     │
    │                                                          │
    │  Plates denote replication — a concise notation for      │
    │  repeated substructure in Bayesian PGMs.                 │
    └──────────────────────────────────────────────────────────┘



### PART 10 — MODERN EXTENSIONS & CONNECTIONS TO DEEP LEARNING

### Variational Autoencoders (VAEs)

The VAE (Kingma & Welling, 2014) is a deep latent-variable model:

    GENERATIVE MODEL:  p_θ(x, z) = p_θ(x | z) · p(z)
        p(z) = N(0, I)   (isotropic Gaussian prior)
        p_θ(x | z): decoder network (outputs mean/variance of p(x|z))

    INFERENCE MODEL:   q_φ(z | x)  — encoder network
        q_φ(z|x) = N(μ_φ(x), diag(σ_φ(x)²))   (Gaussian mean field)

OBJECTIVE: Maximise the ELBO:
    ℒ(θ, φ; x) = 𝔼_{q_φ(z|x)}[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z))

    Reconstruction term:  𝔼[log p_θ(x|z)]  (decoder quality)
    KL regularisation:    D_KL(q_φ ‖ p(z))  (prior matching)

REPARAMETERISATION: z = μ_φ(x) + σ_φ(x) ⊙ ε, ε ~ N(0,I)
    Makes gradient ∇_φ ELBO tractable via standard backpropagation.

The VAE is VI applied to deep generative models — the encoder is
the variational distribution, the decoder is the likelihood.


### Conditional Random Fields (CRFs)

A CRF is a DISCRIMINATIVE undirected model that directly models p(y | x):

    p(y | x) = (1/Z(x)) · exp(Σₜ Σₖ λₖ fₖ(yₜ, yₜ₋₁, xₜ))

    fₖ — FEATURE FUNCTIONS (encode label-observation compatibility)
    λₖ — LEARNED WEIGHTS

CRFs vs HMMs:
    HMM: generative  p(x, y) = p(y)·p(x|y). Models both x and y.
    CRF: discriminative p(y|x). Directly optimises labelling accuracy.
    CRF can use GLOBAL features of x (e.g., future context).
    CRFs generalise logistic regression to structured outputs.

Applications: Named entity recognition, part-of-speech tagging, image segmentation.

Inference in CRF = BP/Viterbi on the (y, label) chain.
Learning = gradient ascent on log p(y|x) — requires computing Z(x) at each step.


### Normalising Flows

A NORMALISING FLOW transforms a simple distribution (e.g., N(0,I))
through a sequence of invertible differentiable maps fₖ:

    z₀ ~ p₀(z₀)   (base distribution)
    zₖ = fₖ(zₖ₋₁)  (invertible transformation)
    log p(x) = log p₀(z₀) − Σₖ log |det Jₖ|   (change of variables)

Each map fₖ must have a tractable Jacobian determinant.
Coupling layers, autoregressive flows, and residual flows are common choices.

Flows allow EXACT likelihood computation — unlike VAEs which only bound log p(x).


### Neural Belief Propagation & Graph Neural Networks

Modern PGMs connect to GNNs:
    MESSAGE PASSING in GNNs = generalised BP on a graph.
    Each GNN layer computes new node features from neighbourhood:

        hᵥ⁽ˡ⁺¹⁾ = UPDATE(hᵥ⁽ˡ⁾, AGGREGATE({hᵤ⁽ˡ⁾: u ∈ ne(v)}))

    This is BP where messages are learned (not hand-designed).
    GNNs are universal function approximators on graphs (in the WL hierarchy).

NEURAL BELIEF PROPAGATION: Learn message-passing functions from data.
    Replaces hand-designed potential functions with neural networks.
    Preserves the graph inductive bias but gains expressive power.

    ┌──────────────────────────────────────────────────────────┐
    │  PGM Hierarchy in Modern ML:                             │
    │                                                          │
    │  Classical PGMs          Deep Extensions                 │
    │  ─────────────────        ─────────────────              │
    │  Naive Bayes          →  Text classifiers (logistic reg) │
    │  HMM                  →  Sequence-to-sequence + CTC      │
    │  Gaussian MRF         →  Diffusion models (denoising)    │
    │  LDA                  →  Transformer attention           │
    │  VI + BN              →  VAE, β-VAE, VQ-VAE              │
    │  Factor graphs + BP   →  Graph Neural Networks           │
    │  Kalman filter        →  SSMs: Mamba, S4, LSTM           │
    └──────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Bayesian Networks — Joint, Marginals & D-Separation": {
        "description": (
            "Build a discrete Bayesian network from scratch using only NumPy. "
            "Compute the full joint distribution, all marginals by enumeration, "
            "and conditional queries (exact inference by variable elimination). "
            "Verify d-separation claims numerically. Demonstrate the 'explaining "
            "away' (Berkson's paradox) collider effect with before/after conditioning."
        ),
        "language": "python",
        "code": '''
import numpy as np
from itertools import product

print("=" * 65)
print("  BAYESIAN NETWORKS: JOINT, MARGINALS & D-SEPARATION")
print("=" * 65)
print()

# ── Define the network ────────────────────────────────────────────────────
# Graph:  Smoking → LungCancer → Dyspnoea
#                 ↘             ↑
#                  Bronchitis ──╯
#
# Variables (all binary: 0=False, 1=True):
#   S = Smoking,  L = LungCancer,  B = Bronchitis,  D = Dyspnoea

# CPTs stored as arrays indexed [parent_val, ..., node_val]
p_S    = np.array([0.7, 0.3])          # p(S)
p_L_S  = np.array([[0.99, 0.01],       # p(L|S=0)
                   [0.90, 0.10]])       # p(L|S=1)
p_B_S  = np.array([[0.97, 0.03],       # p(B|S=0)
                   [0.55, 0.45]])       # p(B|S=1)
p_D_LB = np.array([[[0.90, 0.10],      # p(D|L=0,B=0)
                    [0.20, 0.80]],      # p(D|L=0,B=1)
                   [[0.30, 0.70],       # p(D|L=1,B=0)
                    [0.05, 0.95]]])     # p(D|L=1,B=1)

# ── Build the full joint distribution ─────────────────────────────────────
print("  NETWORK: S → L → D ← B ← S  (Smoking/Lung/Bronchitis/Dyspnoea)")
print()
print("  p(S,L,B,D) = p(S)·p(L|S)·p(B|S)·p(D|L,B)")
print()

joint = np.zeros((2, 2, 2, 2))  # shape [S, L, B, D]
for s, l, b, d in product(range(2), repeat=4):
    joint[s, l, b, d] = (p_S[s] * p_L_S[s, l] *
                         p_B_S[s, b] * p_D_LB[l, b, d])

print(f"  Joint table sum = {joint.sum():.8f}  (should be 1.0)")
print()

# ── Marginal distributions ─────────────────────────────────────────────────
def marginal(jt, keep_axes):
    """Sum out all axes not in keep_axes."""
    all_axes = tuple(range(jt.ndim))
    sum_axes  = tuple(a for a in all_axes if a not in keep_axes)
    m = jt.sum(axis=sum_axes)
    # Restore original axis order
    return m

p_S_marg = marginal(joint, (0,))
p_L_marg = marginal(joint, (1,))
p_B_marg = marginal(joint, (2,))
p_D_marg = marginal(joint, (3,))

print("  MARGINAL DISTRIBUTIONS:")
print(f"  p(S=1) = {p_S_marg[1]:.4f}")
print(f"  p(L=1) = {p_L_marg[1]:.4f}")
print(f"  p(B=1) = {p_B_marg[1]:.4f}")
print(f"  p(D=1) = {p_D_marg[1]:.4f}")
print()

# ── Conditional queries ────────────────────────────────────────────────────
def conditional(jt, query_axes, evidence_dict):
    """
    Compute p(query | evidence) by:
    1. Set non-evidence, non-query variables to be summed out.
    2. Slice evidence variables to their observed values.
    3. Normalise.
    """
    # Restrict to evidence
    slices = [slice(None)] * jt.ndim
    for ax, val in evidence_dict.items():
        slices[ax] = val
    jt_e = jt[tuple(slices)]

    # Sum out non-query axes relative to restricted tensor
    # Need to track which axes remain after slicing
    remaining = [a for a in range(jt.ndim) if a not in evidence_dict]
    query_in_remaining = [remaining.index(q) for q in query_axes]
    sum_in_remaining   = [i for i in range(len(remaining))
                          if i not in query_in_remaining]
    if sum_in_remaining:
        unnorm = jt_e.sum(axis=tuple(sum_in_remaining))
    else:
        unnorm = jt_e
    return unnorm / unnorm.sum()

# Variable indices: S=0, L=1, B=2, D=3

print("  CONDITIONAL QUERIES:")
# p(L|S=1)
pL_S1 = conditional(joint, [1], {0: 1})
print(f"  p(L=1 | S=1)         = {pL_S1[1]:.4f}  (vs prior p(L=1)={p_L_marg[1]:.4f})")

# p(L|D=1)
pL_D1 = conditional(joint, [1], {3: 1})
print(f"  p(L=1 | D=1)         = {pL_D1[1]:.4f}")

# p(S|D=1)
pS_D1 = conditional(joint, [0], {3: 1})
print(f"  p(S=1 | D=1)         = {pS_D1[1]:.4f}  (vs prior p(S=1)={p_S_marg[1]:.4f})")

# p(L|D=1,B=1) — explaining away
pL_D1B1 = conditional(joint, [1], {3: 1, 2: 1})
print(f"  p(L=1 | D=1, B=1)   = {pL_D1B1[1]:.4f}  (explaining away: B explains D, L less likely)")

# p(L|D=1,B=0)
pL_D1B0 = conditional(joint, [1], {3: 1, 2: 0})
print(f"  p(L=1 | D=1, B=0)   = {pL_D1B0[1]:.4f}  (without B, L must explain D)")
print()

# ── D-Separation verification ──────────────────────────────────────────────
print("  D-SEPARATION VERIFICATION (checking independence numerically):")
print()

def total_variation(p, q):
    """Total variation distance between two marginal arrays."""
    return 0.5 * np.sum(np.abs(p - q))

# Claim 1: S ⊥⊥ D | {L, B}   (S and D are d-separated by {L,B})
# Check: p(S|L,B) should equal p(S|L,B,D)
errors = []
for l, b in product(range(2), repeat=2):
    pS_LB    = conditional(joint, [0], {1: l, 2: b})
    pS_LB_D0 = conditional(joint, [0], {1: l, 2: b, 3: 0})
    pS_LB_D1 = conditional(joint, [0], {1: l, 2: b, 3: 1})
    errors.append(total_variation(pS_LB, pS_LB_D0))
    errors.append(total_variation(pS_LB, pS_LB_D1))
print(f"  S ⊥⊥ D | {{L,B}} ?  Max TV distance = {max(errors):.6f}  "
      + ("→ CONFIRMED ✓" if max(errors) < 1e-8 else "→ NOT independent"))

# Claim 2: S NOT ⊥⊥ D (marginally)
pS_D0 = conditional(joint, [0], {3: 0})
pS_D1 = conditional(joint, [0], {3: 1})
tv_SD = total_variation(pS_D0, pS_D1)
print(f"  S ⊥⊥ D (marginal)?  TV(p(S|D=0), p(S|D=1)) = {tv_SD:.4f}  "
      + ("→ DEPENDENT (correct)" if tv_SD > 0.01 else "→ independent"))

# Claim 3: L ⊥⊥ B | S  (both children of S, d-separated given S)
errors2 = []
for s in range(2):
    pL_S    = conditional(joint, [1], {0: s})
    pL_S_B0 = conditional(joint, [1], {0: s, 2: 0})
    pL_S_B1 = conditional(joint, [1], {0: s, 2: 1})
    errors2.append(total_variation(pL_S, pL_S_B0))
    errors2.append(total_variation(pL_S, pL_S_B1))
print(f"  L ⊥⊥ B | S ?       Max TV distance = {max(errors2):.6f}  "
      + ("→ CONFIRMED ✓" if max(errors2) < 1e-8 else "→ NOT independent"))

# Claim 4: L NOT ⊥⊥ B (marginally, both caused by S)
pL_B0 = conditional(joint, [1], {2: 0})
pL_B1 = conditional(joint, [1], {2: 1})
tv_LB = total_variation(pL_B0, pL_B1)
print(f"  L ⊥⊥ B (marginal)?  TV = {tv_LB:.4f}  "
      + ("→ DEPENDENT (correct, both caused by S)" if tv_LB > 0.01 else "→ independent"))

# Claim 5: COLLIDER — S ⊥⊥ B initially but NOT given D
pS_B0    = conditional(joint, [0], {2: 0})
pS_B1    = conditional(joint, [0], {2: 1})
tv_SB    = total_variation(pS_B0, pS_B1)
pS_B0D1  = conditional(joint, [0], {2: 0, 3: 1})
pS_B1D1  = conditional(joint, [0], {2: 1, 3: 1})
tv_SBgD  = total_variation(pS_B0D1, pS_B1D1)
print()
print(f"  COLLIDER EFFECT (D is collider of L←D→B path via common parent structure):")
print(f"  S ⊥⊥ B (marginal)?     TV = {tv_SB:.4f}")
print(f"  S ⊥⊥ B | D=1?          TV = {tv_SBgD:.4f}  "
      + "← conditioning on effect creates dependence!")
print()
print("  KEY TAKEAWAYS:")
print("  Missing edges in a BN are INDEPENDENCE CLAIMS, not just simplifications.")
print("  D-separation correctly predicts every conditional independence above.")
print("  Collider effect (explaining away): observing the effect creates")
print("  spurious correlation between independent causes.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Belief Propagation — Exact on Trees, Loopy Approximation": {
        "description": (
            "Implement sum-product belief propagation from scratch on a chain "
            "and a tree factor graph. Verify exact marginals against brute-force "
            "enumeration. Extend to a graph with cycles (loopy BP) and measure "
            "approximation error vs ground truth. Track convergence of messages "
            "and demonstrate the effect of coupling strength on loopy BP quality."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as iproduct

np.random.seed(42)

print("=" * 65)
print("  BELIEF PROPAGATION: EXACT ON TREES, LOOPY ON GRAPHS")
print("=" * 65)
print()

# ── Helper: brute force marginals ─────────────────────────────────────────
def brute_force_marginals(unary, pairwise_list, edges, n_vars, n_states):
    """Compute exact marginals by full enumeration."""
    joint = np.ones([n_states] * n_vars)
    for i in range(n_vars):
        idx = [np.newaxis] * n_vars
        idx[i] = slice(None)
        joint *= unary[i][tuple(idx)]
    for (i, j), psi in zip(edges, pairwise_list):
        idx = [np.newaxis] * n_vars
        idx[i] = slice(None)
        idx[j] = slice(None)
        joint *= psi[tuple(idx)]
    Z = joint.sum()
    marginals = []
    for i in range(n_vars):
        ax = tuple(k for k in range(n_vars) if k != i)
        marginals.append(joint.sum(axis=ax) / Z)
    return marginals, Z

# ── Part 1: Exact BP on a Chain ───────────────────────────────────────────
print("  PART 1 — EXACT BELIEF PROPAGATION ON A CHAIN (n=6, K=3 states)")
print()

n_chain  = 6
K        = 3
np.random.seed(10)

# Random unary and pairwise potentials
unary_c = [np.random.dirichlet(np.ones(K)) for _ in range(n_chain)]
# Pairwise: symmetric, moderate coupling
psi_c = []
for _ in range(n_chain - 1):
    raw = np.random.uniform(0.5, 2.0, (K, K))
    raw = (raw + raw.T) / 2   # symmetrise
    psi_c.append(raw)
edges_c = [(i, i+1) for i in range(n_chain - 1)]

# Run forward-backward sum-product (BP on chain)
def chain_bp(unary, psi, n, K):
    """Sum-product BP on a chain: returns marginals."""
    # Forward messages: mu_fwd[t] = message from node t to t+1
    mu_fwd = [np.ones(K)]   # dummy for t=0
    for t in range(n - 1):
        m = unary[t] * mu_fwd[t]          # incorporate incoming + unary
        m_new = psi[t].T @ m              # send to t+1
        mu_fwd.append(m_new / m_new.sum())

    # Backward messages
    mu_bwd = [np.ones(K)]
    for t in range(n - 1, 0, -1):
        m = unary[t] * mu_bwd[0]
        m_new = psi[t-1] @ m
        mu_bwd.insert(0, m_new / m_new.sum())

    # Beliefs
    beliefs = []
    for t in range(n):
        b = unary[t] * mu_fwd[t] * mu_bwd[t]
        beliefs.append(b / b.sum())
    return beliefs

bp_marginals = chain_bp(unary_c, psi_c, n_chain, K)
bf_marginals, Z_c = brute_force_marginals(unary_c, psi_c, edges_c, n_chain, K)

print(f"  {'Node':>6} | {'BP marginal (state 0)':>22} | {'BF marginal (state 0)':>22} | {'Error':>10}")
print(f"  {'─'*68}")
for i in range(n_chain):
    err = np.max(np.abs(bp_marginals[i] - bf_marginals[i]))
    print(f"  {i:>6} | {bp_marginals[i][0]:22.8f} | {bf_marginals[i][0]:22.8f} | {err:10.2e}")
print()
print("  BP is EXACT on trees/chains. Errors are numerical only (< 1e-12).")
print()

# ── Part 2: Loopy BP on a graph with cycles ────────────────────────────────
print("  PART 2 — LOOPY BP ON A CYCLIC GRAPH")
print()

# Graph: 4-node square   0─1
#                        │ │
#                        3─2
n_loop  = 4
edges_l = [(0,1), (1,2), (2,3), (3,0)]   # a cycle

def loopy_bp(unary, pairwise, edges, n, K, n_iters=100, damp=0.5):
    """
    Loopy sum-product BP with damping.
    Returns beliefs and message history.
    """
    # messages[i][j] = message from node i to node j
    msgs = {(i,j): np.ones(K)/K for (i,j) in edges}
    msgs.update({(j,i): np.ones(K)/K for (i,j) in edges})
    psi_dict = {}
    for (i,j), psi in zip(edges, pairwise):
        psi_dict[(i,j)] = psi
        psi_dict[(j,i)] = psi.T

    # Build adjacency
    adj = {v: [] for v in range(n)}
    for (i,j) in edges:
        adj[i].append(j); adj[j].append(i)

    msg_hist = []

    for it in range(n_iters):
        new_msgs = {}
        for (i,j) in list(msgs.keys()):
            # Compute incoming product at i from all neighbours except j
            incoming = unary[i].copy()
            for k in adj[i]:
                if k != j:
                    incoming = incoming * msgs[(k, i)]
            # Pass through factor
            psi_ij = psi_dict[(i,j)]
            m_new = psi_ij.T @ incoming
            m_new = m_new / m_new.sum()
            # Damping: blend new and old
            m_damp = damp * m_new + (1 - damp) * msgs[(i,j)]
            new_msgs[(i,j)] = m_damp / m_damp.sum()
        msgs = new_msgs
        # Track max message change
        msg_vec = np.concatenate([msgs[k] for k in sorted(msgs)])
        msg_hist.append(msg_vec)

    # Compute beliefs
    beliefs = []
    for i in range(n):
        b = unary[i].copy()
        for j in adj[i]:
            b = b * msgs[(j, i)]
        beliefs.append(b / b.sum())
    return beliefs, msg_hist

# Test across coupling strengths
np.random.seed(7)
unary_l = [np.random.dirichlet(2*np.ones(K)) for _ in range(n_loop)]
coupling_strengths = [0.2, 1.0, 3.0]

print(f"  4-node cycle graph, K={K} states, evaluating coupling strength effect")
print()
print(f"  {'Coupling':>10} | {'Node':>6} | {'Loopy BP':>12} | {'Brute Force':>14} | {'Max Error':>12}")
print(f"  {'─'*60}")

all_errors = {}
for coupling in coupling_strengths:
    psi_l = []
    for _ in range(n_loop):
        raw = np.exp(coupling * np.random.randn(K, K))
        raw = (raw + raw.T) / 2
        psi_l.append(raw)
    lbp_marg, mhist = loopy_bp(unary_l, psi_l, edges_l, n_loop, K,
                                n_iters=200, damp=0.5)
    bf_marg, _ = brute_force_marginals(unary_l, psi_l, edges_l, n_loop, K)
    errors = [np.max(np.abs(lbp_marg[i] - bf_marg[i])) for i in range(n_loop)]
    all_errors[coupling] = (lbp_marg, bf_marg, errors, mhist)
    print(f"  {coupling:>10.1f} | {'avg':>6} | {'':>12} | {'':>14} | {np.mean(errors):12.5f}")
    for i in range(n_loop):
        print(f"  {'':>10} | {i:>6} | {lbp_marg[i][0]:12.5f} | {bf_marg[i][0]:14.5f} | {errors[i]:12.5f}")
    print()
print("  Low coupling → accurate loopy BP.  High coupling → larger errors.")
print()

# ── Part 3: Convergence of loopy BP messages ──────────────────────────────
print("  PART 3 — LOOPY BP CONVERGENCE (message change per iteration)")
print()
for coupling in coupling_strengths:
    _, _, _, mhist = all_errors[coupling]
    changes = [np.max(np.abs(mhist[t] - mhist[t-1])) for t in range(1, len(mhist))]
    convg_iter = next((t for t, c in enumerate(changes) if c < 1e-6), len(changes))
    print(f"  Coupling {coupling:.1f}: converged at iter ≈ {convg_iter}, "
          f"final change = {changes[-1]:.2e}")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Belief Propagation: Exact on Trees, Loopy on Graphs",
             fontsize=12, fontweight="bold")

# Plot 1: BP vs BF marginals on chain
x_pos = np.arange(n_chain)
width = 0.35
for state in range(K):
    bp_vals = [bp_marginals[i][state] for i in range(n_chain)]
    bf_vals = [bf_marginals[i][state] for i in range(n_chain)]
    axes[0].bar(x_pos + (state - 1)*width/K, bp_vals, width/K,
                label=f"BP state {state}", alpha=0.8)
    axes[0].bar(x_pos + (state - 1)*width/K + width, bf_vals, width/K,
                label=f"BF state {state}", alpha=0.5, hatch="//")
axes[0].set_xlabel("Node"); axes[0].set_ylabel("Marginal probability")
axes[0].set_title("Exact BP on Chain\\n(BP vs Brute Force — identical)")
axes[0].set_xticks(x_pos + width/2)
axes[0].set_xticklabels([f"x{i}" for i in range(n_chain)])
axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3, axis="y")

# Plot 2: Approximation error vs coupling
for node in range(n_loop):
    node_errors = [all_errors[c][2][node] for c in coupling_strengths]
    axes[1].plot(coupling_strengths, node_errors, marker="o", lw=2,
                 label=f"Node {node}")
axes[1].set_xlabel("Coupling strength"); axes[1].set_ylabel("Max marginal error")
axes[1].set_title("Loopy BP Error vs Coupling Strength\\n(4-cycle, K=3)")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
axes[1].set_yscale("log")

# Plot 3: Message convergence curves
colors_conv = ["steelblue", "tomato", "seagreen"]
for coupling, col in zip(coupling_strengths, colors_conv):
    _, _, _, mhist = all_errors[coupling]
    changes = [np.max(np.abs(np.array(mhist[t]) - np.array(mhist[t-1])))
               for t in range(1, len(mhist))]
    axes[2].semilogy(changes, lw=2, color=col, label=f"Coupling={coupling}")
axes[2].axhline(1e-6, color="gray", linestyle="--", lw=1.5, label="Threshold 1e-6")
axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Max message change (log scale)")
axes[2].set_title("Loopy BP Message Convergence\\n(Higher coupling = slower/worse)")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("belief_propagation.png", dpi=120)
print("  Plot saved → belief_propagation.png")
print()
print("  KEY TAKEAWAYS:")
print("  BP on trees is EXACT — errors are purely numerical (< 1e-12).")
print("  Loopy BP on cycles approximates marginals — accuracy degrades")
print("  with coupling strength (stronger interactions → harder problem).")
print("  Message damping aids convergence at the cost of slower updates.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Hidden Markov Models — Forward-Backward & Viterbi": {
        "description": (
            "Implement the HMM forward-backward algorithm (sum-product BP on the "
            "chain) and the Viterbi algorithm (max-product) from scratch. Generate "
            "synthetic sequences, run inference, then learn parameters with the "
            "Baum-Welch EM algorithm. Compare the most probable PATH (Viterbi) to "
            "the sequence of most probable STATES (marginal argmax). "
            "Visualise posterior state probabilities and likelihood convergence."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

print("=" * 65)
print("  HIDDEN MARKOV MODELS: FORWARD-BACKWARD & VITERBI")
print("=" * 65)
print()

# ── Define a 3-state HMM ──────────────────────────────────────────────────
K = 3   # hidden states (0=low, 1=mid, 2=high)
V = 4   # observation symbols

# True parameters
pi_true = np.array([0.6, 0.3, 0.1])
A_true  = np.array([[0.7, 0.2, 0.1],   # transition matrix
                    [0.1, 0.6, 0.3],
                    [0.2, 0.3, 0.5]])
B_true  = np.array([[0.5, 0.3, 0.1, 0.1],  # emission matrix B[k,v]
                    [0.1, 0.4, 0.4, 0.1],
                    [0.1, 0.1, 0.3, 0.5]])

# ── Generate a synthetic sequence ─────────────────────────────────────────
T = 60

def generate_hmm(pi, A, B, T):
    K = len(pi)
    z = np.zeros(T, dtype=int)
    x = np.zeros(T, dtype=int)
    z[0] = np.random.choice(K, p=pi)
    x[0] = np.random.choice(B.shape[1], p=B[z[0]])
    for t in range(1, T):
        z[t] = np.random.choice(K, p=A[z[t-1]])
        x[t] = np.random.choice(B.shape[1], p=B[z[t]])
    return z, x

z_true, x_obs = generate_hmm(pi_true, A_true, B_true, T)
print(f"  Generated sequence of length T={T} from 3-state, 4-symbol HMM.")
print(f"  True states: {z_true[:20]}... (showing first 20)")
print(f"  Observations:{x_obs[:20]}...")
print()

# ── Forward-Backward Algorithm ────────────────────────────────────────────
def forward_backward(x, pi, A, B):
    T = len(x)
    K = len(pi)

    # Forward pass (with scaling for numerical stability)
    alpha = np.zeros((T, K))
    scale = np.zeros(T)
    alpha[0] = pi * B[:, x[0]]
    scale[0] = alpha[0].sum(); alpha[0] /= scale[0]
    for t in range(1, T):
        alpha[t] = B[:, x[t]] * (alpha[t-1] @ A)
        scale[t] = alpha[t].sum(); alpha[t] /= scale[t]

    # Backward pass
    beta = np.zeros((T, K))
    beta[-1] = 1.0
    for t in range(T-2, -1, -1):
        beta[t] = A @ (B[:, x[t+1]] * beta[t+1])
        beta[t] /= scale[t+1]   # use same scaling

    # Posterior marginals
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    # Pairwise posteriors (for EM)
    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        xi[t] = (alpha[t][:, np.newaxis] * A *
                 B[:, x[t+1]][np.newaxis, :] * beta[t+1][np.newaxis, :])
        xi[t] /= xi[t].sum()

    log_lik = np.sum(np.log(scale))
    return gamma, xi, log_lik

gamma, xi, log_lik = forward_backward(x_obs, pi_true, A_true, B_true)
print(f"  FORWARD-BACKWARD RESULTS (true parameters):")
print(f"  Log-likelihood: {log_lik:.4f}")
print()
print(f"  Posterior p(Zₜ=k | x₁:T) for first 10 time steps:")
print(f"  {'t':>4} | {'State 0':>10} | {'State 1':>10} | {'State 2':>10} | "
      f"{'MAP state':>12} | {'True state':>12}")
print(f"  {'─'*65}")
for t in range(10):
    map_state = gamma[t].argmax()
    print(f"  {t:>4} | {gamma[t,0]:10.4f} | {gamma[t,1]:10.4f} | "
          f"{gamma[t,2]:10.4f} | {map_state:>12} | {z_true[t]:>12}")
print()

# ── Viterbi Algorithm ─────────────────────────────────────────────────────
def viterbi(x, pi, A, B):
    T = len(x)
    K = len(pi)
    log_A = np.log(A + 1e-300)
    log_B = np.log(B + 1e-300)
    log_pi = np.log(pi + 1e-300)

    delta = np.zeros((T, K))
    psi   = np.zeros((T, K), dtype=int)

    delta[0] = log_pi + log_B[:, x[0]]
    for t in range(1, T):
        trans = delta[t-1][:, np.newaxis] + log_A   # (K, K)
        psi[t]   = trans.argmax(axis=0)
        delta[t] = trans.max(axis=0) + log_B[:, x[t]]

    # Backtrack
    z_star = np.zeros(T, dtype=int)
    z_star[-1] = delta[-1].argmax()
    for t in range(T-2, -1, -1):
        z_star[t] = psi[t+1, z_star[t+1]]

    return z_star, delta[-1].max()

z_viterbi, log_prob_viterbi = viterbi(x_obs, pi_true, A_true, B_true)
z_marginal_map = gamma.argmax(axis=1)   # argmax of marginals at each step

# Compare Viterbi path vs marginal MAP
agree_viterbi = (z_viterbi == z_true).mean()
agree_margmap = (z_marginal_map == z_true).mean()
agree_each_other = (z_viterbi == z_marginal_map).mean()

print(f"  VITERBI vs MARGINAL MAP:")
print(f"  Viterbi MAP-path accuracy (vs true):  {agree_viterbi:.4f}")
print(f"  Marginal MAP accuracy (vs true):       {agree_margmap:.4f}")
print(f"  Viterbi == Marginal MAP (same answer): {agree_each_other:.4f}")
print(f"  (They differ because joint MAP ≠ sequence of marginal MAPs)")
print()

# ── Baum-Welch EM ────────────────────────────────────────────────────────
print("  BAUM-WELCH EM: LEARNING FROM OBSERVATIONS")
print()

def baum_welch(x, K, V, n_iters=100):
    T = len(x)
    # Random initialisation
    pi = np.random.dirichlet(np.ones(K))
    A  = np.array([np.random.dirichlet(np.ones(K)) for _ in range(K)])
    B  = np.array([np.random.dirichlet(np.ones(V)) for _ in range(K)])
    ll_hist = []

    for it in range(n_iters):
        # E-step
        gamma, xi, log_lik = forward_backward(x, pi, A, B)
        ll_hist.append(log_lik)

        # M-step
        pi = gamma[0]
        A_new = xi.sum(axis=0)                     # (K, K)
        A_new /= A_new.sum(axis=1, keepdims=True)

        B_new = np.zeros((K, V))
        for v in range(V):
            B_new[:, v] = gamma[x == v].sum(axis=0)
        B_new /= B_new.sum(axis=1, keepdims=True)

        A = A_new; B = B_new

    return pi, A, B, ll_hist

pi_est, A_est, B_est, ll_hist = baum_welch(x_obs, K, V, n_iters=80)
final_gamma, _, final_ll = forward_backward(x_obs, pi_est, A_est, B_est)

print(f"  True log-likelihood (true params):      {log_lik:.4f}")
print(f"  Learned log-likelihood (Baum-Welch):    {ll_hist[-1]:.4f}")
print()
print(f"  Learned transition matrix A_est:")
for k in range(K):
    print(f"    State {k}: {A_est[k]}")
print()
print(f"  True transition matrix A_true:")
for k in range(K):
    print(f"    State {k}: {A_true[k]}")
print()
print("  (Row/column permutations expected due to label switching.)")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(17, 10))
fig.suptitle("Hidden Markov Models: Forward-Backward & Viterbi",
             fontsize=12, fontweight="bold")

# Plot 1: Observations and true states
t_range = np.arange(T)
axes[0, 0].step(t_range, x_obs, where="mid", color="steelblue", lw=1.5, label="Observation")
axes[0, 0].step(t_range, z_true, where="mid", color="tomato", lw=2, label="True state", linestyle="--")
axes[0, 0].set_xlabel("Time t"); axes[0, 0].set_ylabel("Symbol / State")
axes[0, 0].set_title(f"Observations and True Hidden States (T={T})")
axes[0, 0].legend(fontsize=9); axes[0, 0].grid(alpha=0.3)

# Plot 2: Posterior state probabilities
palette = ["steelblue", "tomato", "seagreen"]
for k in range(K):
    axes[0, 1].fill_between(t_range, gamma[:, k], alpha=0.5,
                            color=palette[k], label=f"p(Zₜ=State {k}|x)")
    axes[0, 1].plot(t_range, gamma[:, k], color=palette[k], lw=1.2)
axes[0, 1].step(t_range, z_true / (K-1) * 0.9, color="black", lw=1.5,
                where="mid", linestyle=":", label="True state (normalised)")
axes[0, 1].set_xlabel("Time t"); axes[0, 1].set_ylabel("Posterior probability")
axes[0, 1].set_title("Posterior State Probabilities\\n(Forward-Backward smoothing)")
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

# Plot 3: Viterbi path vs marginal MAP
axes[1, 0].step(t_range, z_true,       color="black", lw=2.5, where="mid", label="True states", alpha=0.7)
axes[1, 0].step(t_range, z_viterbi,    color="steelblue", lw=2, where="mid", linestyle="--", label="Viterbi (joint MAP)")
axes[1, 0].step(t_range, z_marginal_map, color="tomato", lw=1.5, where="mid", linestyle=":", label="Marginal MAP (pointwise)")
disagreements = t_range[z_viterbi != z_marginal_map]
for td in disagreements:
    axes[1, 0].axvline(td, color="orange", alpha=0.4, lw=1)
axes[1, 0].set_xlabel("Time t"); axes[1, 0].set_ylabel("State")
axes[1, 0].set_title(f"Viterbi vs Marginal MAP\\n({len(disagreements)} disagreements highlighted)")
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

# Plot 4: Baum-Welch likelihood convergence
axes[1, 1].plot(ll_hist, "steelblue", lw=2)
axes[1, 1].axhline(log_lik, color="tomato", lw=1.5, linestyle="--",
                   label=f"True param LL = {log_lik:.1f}")
axes[1, 1].set_xlabel("EM Iteration"); axes[1, 1].set_ylabel("Log-likelihood")
axes[1, 1].set_title("Baum-Welch EM Convergence\\nNon-decreasing (EM guarantee)")
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("hmm_fb_viterbi.png", dpi=120)
print("  Plot saved → hmm_fb_viterbi.png")
print()
print("  KEY TAKEAWAYS:")
print("  Forward-backward = sum-product BP on the chain (exact).")
print("  Viterbi = max-product BP on the chain = exact MAP decoding.")
print("  Joint MAP (Viterbi) ≠ sequence of marginal MAPs — these")
print("  are different optimisation problems with different answers.")
print("  Baum-Welch is EM: likelihood is guaranteed non-decreasing.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Variational Inference — ELBO, Mean Field & the VAE Objective": {
        "description": (
            "Derive and optimise the ELBO for a Gaussian mixture model using "
            "coordinate ascent mean-field VI. Track the ELBO gap (true vs lower "
            "bound). Implement the VAE ELBO (reconstruction + KL) for a simple "
            "1D latent model and show the reparameterisation trick in action. "
            "Compare the forward KL (mean-seeking) and reverse KL (mode-seeking) "
            "approximations to a bimodal distribution."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, special

np.random.seed(42)

print("=" * 65)
print("  VARIATIONAL INFERENCE: ELBO, MEAN FIELD & VAE OBJECTIVE")
print("=" * 65)
print()

# ── Part 1: ELBO for a Gaussian model with a Gaussian variational family ───
print("  PART 1 — ELBO: GAUSSIAN POSTERIOR APPROXIMATION")
print()
# True model: p(z) = N(0,1), p(x|z) = N(z, σ_lik²)
# Observed x, want q(z) ≈ p(z|x)
# True posterior: p(z|x) = N(μ_post, σ_post²)
#   σ_post² = 1/(1 + 1/σ_lik²),  μ_post = σ_post² * x/σ_lik²

sigma_lik = 0.5   # likelihood noise
x_obs_vi  = 2.0   # single observation

# True posterior
sigma_post_sq = 1 / (1 + 1/sigma_lik**2)
mu_post = sigma_post_sq * x_obs_vi / sigma_lik**2
sigma_post = np.sqrt(sigma_post_sq)

print(f"  Model: p(z)=N(0,1), p(x|z)=N(z, σ²={sigma_lik**2}), x={x_obs_vi}")
print(f"  True posterior: p(z|x) = N({mu_post:.4f}, {sigma_post**2:.4f})")
print()

# ELBO for q(z) = N(μ_q, σ_q²):
# ELBO = 𝔼_q[log p(x|z)] + 𝔼_q[log p(z)] - 𝔼_q[log q(z)]
# = 𝔼_q[log p(x|z)] - D_KL(q||p(z))
def elbo_gaussian(mu_q, log_sigma_q, x, sigma_lik):
    sigma_q = np.exp(log_sigma_q)
    # 𝔼_q[log p(x|z)] = -0.5*log(2π σ_lik²) - 0.5/σ_lik² * 𝔼_q[(x-z)²]
    #                  = -0.5*log(2π σ_lik²) - 0.5/σ_lik² * [(x-μ_q)² + σ_q²]
    recon = -0.5*np.log(2*np.pi*sigma_lik**2) - 0.5/sigma_lik**2 * ((x - mu_q)**2 + sigma_q**2)
    # D_KL(N(μ_q, σ_q²) || N(0,1)) = 0.5*(σ_q² + μ_q² - 1 - log σ_q²)
    kl = 0.5*(sigma_q**2 + mu_q**2 - 1 - 2*log_sigma_q)
    return recon - kl

# True log-evidence: log p(x) = N(x; 0, 1 + σ_lik²)
log_px_true = stats.norm.logpdf(x_obs_vi, 0, np.sqrt(1 + sigma_lik**2))

# Sweep μ_q for fixed σ_q = σ_post
mu_grid = np.linspace(-1, 4, 200)
elbo_mu  = [elbo_gaussian(m, np.log(sigma_post), x_obs_vi, sigma_lik) for m in mu_grid]

print(f"  True log p(x) = {log_px_true:.4f}")
print(f"  Max ELBO (optimised μ_q) = {max(elbo_mu):.4f}")
print(f"  ELBO gap = {log_px_true - max(elbo_mu):.4f}  (= D_KL(q* || p(z|x)) at optimum)")
print()

# Gradient ascent on ELBO
mu_q     = 0.0
lsig_q   = 0.0   # log σ_q
lr_vi    = 0.05
elbo_hist = []
for step in range(300):
    sig_q = np.exp(lsig_q)
    grad_mu  = (x_obs_vi - mu_q) / sigma_lik**2 - mu_q
    grad_lsig = -sig_q**2 / sigma_lik**2 + 1 - sig_q**2
    mu_q    += lr_vi * grad_mu
    lsig_q  += lr_vi * 0.5 * grad_lsig
    elbo_hist.append(elbo_gaussian(mu_q, lsig_q, x_obs_vi, sigma_lik))

print(f"  Gradient Ascent on ELBO (300 steps):")
print(f"  Converged μ_q = {mu_q:.5f}  (true posterior μ = {mu_post:.5f})")
print(f"  Converged σ_q = {np.exp(lsig_q):.5f}  (true posterior σ = {sigma_post:.5f})")
print()

# ── Part 2: Forward KL vs Reverse KL on bimodal target ────────────────────
print("  PART 2 — FORWARD KL (MEAN-SEEKING) vs REVERSE KL (MODE-SEEKING)")
print()

# Target: bimodal p(z) = 0.5 N(-2, 0.5²) + 0.5 N(2, 0.5²)
z_grid = np.linspace(-6, 6, 2000)
p_target = 0.5*stats.norm.pdf(z_grid, -2, 0.5) + 0.5*stats.norm.pdf(z_grid, 2, 0.5)
p_target /= np.trapezoid(p_target, z_grid)   # renormalise (already normalised here)

def kl_forward(mu_q, sig_q):
    """D_KL(p || q) — forward KL (mean-seeking)."""
    q = stats.norm.pdf(z_grid, mu_q, sig_q) + 1e-300
    p = p_target + 1e-300
    integrand = p * (np.log(p) - np.log(q))
    return np.trapezoid(integrand, z_grid)

def kl_reverse(mu_q, sig_q):
    """D_KL(q || p) — reverse KL (mode-seeking), minimised by VI."""
    q = stats.norm.pdf(z_grid, mu_q, sig_q) + 1e-300
    p = p_target + 1e-300
    integrand = q * (np.log(q) - np.log(p))
    return np.trapezoid(integrand, z_grid)

# Grid search for optimal Gaussian approximation under each KL
best_fwd = (None, None, np.inf)
best_rev = (None, None, np.inf)
for mu in np.linspace(-3, 3, 30):
    for sig in np.linspace(0.2, 4.0, 30):
        kf = kl_forward(mu, sig)
        kr = kl_reverse(mu, sig)
        if kf < best_fwd[2]: best_fwd = (mu, sig, kf)
        if kr < best_rev[2]: best_rev = (mu, sig, kr)

print(f"  Target: 0.5 N(-2, 0.5²) + 0.5 N(2, 0.5²)")
print(f"  Best Gaussian under D_KL(p||q) [forward]: μ={best_fwd[0]:.2f}, σ={best_fwd[1]:.2f}, KL={best_fwd[2]:.4f}")
print(f"  Best Gaussian under D_KL(q||p) [reverse]: μ={best_rev[0]:.2f}, σ={best_rev[1]:.2f}, KL={best_rev[2]:.4f}")
print()
print("  Forward KL: must cover all of p → broad Gaussian (mean-seeking)")
print("  Reverse KL: concentrates where q>0 on mass of p → mode-seeking")
print()

# ── Part 3: VAE ELBO decomposition ────────────────────────────────────────
print("  PART 3 — VAE ELBO: RECONSTRUCTION + KL DECOMPOSITION")
print()
# Simple 1D latent: x = z + noise, z ~ N(0,1), encoder q(z|x) = N(μ(x), σ²)
# Analytical ELBO:  𝔼_q[log p(x|z)] - D_KL(q(z|x) || p(z))
# = -0.5*(x - μ_enc)²/σ_dec² - 0.5*(μ_enc² + σ_enc² - 1 - log σ_enc²)

n_data = 1000
z_data = np.random.randn(n_data)
x_data = z_data + 0.3 * np.random.randn(n_data)   # noisy observations

# Encoder: mu = w*x, log_sigma² = log_sig (learned scalars)
w_enc = 0.5; lsig_enc = -0.5   # initialise
sigma_dec = 0.3                 # decoder noise (fixed)
lr_vae = 0.01

recon_hist_vae = []; kl_hist_vae = []; elbo_hist_vae = []

for step in range(500):
    sig_enc = np.exp(lsig_enc)
    mu_enc  = w_enc * x_data

    # Reparameterisation: z_sample = mu_enc + sig_enc * eps
    eps = np.random.randn(n_data)
    z_sample = mu_enc + sig_enc * eps

    # Reconstruction loss: 𝔼[log p(x|z)]
    recon = -0.5 * np.log(2*np.pi*sigma_dec**2) - 0.5*((x_data - z_sample)**2)/sigma_dec**2
    recon_term = recon.mean()

    # KL term: D_KL(q||p) = 0.5*(σ² + μ² - 1 - log σ²) (per datapoint, averaged)
    kl_term = 0.5 * (sig_enc**2 + mu_enc**2 - 1 - 2*lsig_enc).mean()

    elbo_vae = recon_term - kl_term

    # Gradients (via reparameterisation)
    d_recon_dz   = (x_data - z_sample) / sigma_dec**2
    grad_w_recon = (d_recon_dz * x_data).mean()       # ∂recon/∂w
    grad_lsig_recon = (d_recon_dz * sig_enc * eps).mean()  # ∂recon/∂log_sig

    grad_w_kl    = mu_enc.mean() * w_enc   # ∂KL/∂w via μ = w*x
    grad_lsig_kl = sig_enc**2 - 0.5   # ∂KL/∂log_sig per sample (mean)

    # Actually compute proper gradient: ∂KL/∂w = (1/n)Σ μ_enc * x (chain rule)
    grad_w_kl  = (mu_enc * x_data).mean()

    w_enc    += lr_vae * (grad_w_recon - grad_w_kl)
    lsig_enc += lr_vae * (grad_lsig_recon - grad_lsig_kl)

    recon_hist_vae.append(recon_term)
    kl_hist_vae.append(kl_term)
    elbo_hist_vae.append(elbo_vae)

print(f"  VAE 1D Model: x = z + ε, z~N(0,1), ε~N(0,{sigma_dec}²)")
print(f"  Encoder: q(z|x) = N(w·x, σ²)")
print()
print(f"  After 500 steps:")
print(f"  Learned w = {w_enc:.4f}  (optimal ≈ 1/(1+σ_dec²) = {1/(1+sigma_dec**2):.4f})")
print(f"  Learned σ_enc = {np.exp(lsig_enc):.4f}  (optimal ≈ √(σ_dec²/(1+σ_dec²)) = {np.sqrt(sigma_dec**2/(1+sigma_dec**2)):.4f})")
print(f"  Final ELBO = {elbo_hist_vae[-1]:.4f}")
print(f"  Final Recon = {recon_hist_vae[-1]:.4f},  KL = {kl_hist_vae[-1]:.4f}")
print()

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Variational Inference: ELBO, Mean Field & VAE",
             fontsize=12, fontweight="bold")

# Plot 1: ELBO surface and convergence
axes[0].plot(mu_grid, elbo_mu, "steelblue", lw=2, label="ELBO(μ_q)")
axes[0].axvline(mu_post, color="tomato", lw=2, linestyle="--",
                label=f"True posterior μ={mu_post:.2f}")
axes[0].axhline(log_px_true, color="gray", lw=1.5, linestyle=":",
                label=f"log p(x)={log_px_true:.2f}")
axes[0].scatter([mu_q], [elbo_hist[-1]], color="seagreen", s=120, zorder=5,
                label=f"Converged μ={mu_q:.2f}")
axes[0].set_xlabel("μ_q"); axes[0].set_ylabel("ELBO")
axes[0].set_title("ELBO as Function of Variational Mean\\n(Gaussian posterior approximation)")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Plot 2: Forward vs Reverse KL approximations
axes[1].fill_between(z_grid, p_target, alpha=0.3, color="black", label="Target p(z)")
axes[1].plot(z_grid, p_target, "k", lw=2)
axes[1].plot(z_grid, stats.norm.pdf(z_grid, best_fwd[0], best_fwd[1]),
             "steelblue", lw=2.5, linestyle="--",
             label=f"Forward KL opt: N({best_fwd[0]:.1f}, {best_fwd[1]:.1f}²)")
axes[1].plot(z_grid, stats.norm.pdf(z_grid, best_rev[0], best_rev[1]),
             "tomato", lw=2.5, linestyle="-.",
             label=f"Reverse KL opt: N({best_rev[0]:.1f}, {best_rev[1]:.1f}²)")
axes[1].set_xlabel("z"); axes[1].set_ylabel("Density")
axes[1].set_title("Forward vs Reverse KL Approximations\\nBimodal target (Gaussian family)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

# Plot 3: VAE ELBO decomposition
iters = np.arange(len(elbo_hist_vae))
axes[2].plot(iters, elbo_hist_vae,    "steelblue", lw=2.5, label="ELBO (total)")
axes[2].plot(iters, recon_hist_vae,   "seagreen",  lw=1.8, linestyle="--", label="Reconstruction term")
axes[2].plot(iters, [-k for k in kl_hist_vae], "tomato", lw=1.8, linestyle=":",
             label="−KL term")
axes[2].set_xlabel("Step"); axes[2].set_ylabel("ELBO components")
axes[2].set_title("VAE ELBO Decomposition\\n(Reconstruction − KL regularisation)")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("variational_inference.png", dpi=120)
print("  Plot saved → variational_inference.png")
print()
print("  KEY TAKEAWAYS:")
print("  ELBO = log p(x) − D_KL(q || p(z|x)): maximising ELBO minimises KL.")
print("  Forward KL (p||q): mass-covering, mean-seeking (spreads q over all modes).")
print("  Reverse KL (q||p): mode-seeking (q concentrates on one mode of p).")
print("  VI in practice (VAE) uses reverse KL — the ELBO objective.")
print("  Reparameterisation makes gradients flow through z samples.")
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