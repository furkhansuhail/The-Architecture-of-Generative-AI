"""Module: 03 · Deep Q-Network (DQN)"""

"""
Deep Q-Network (DQN)
====================

DQN is the landmark algorithm that first demonstrated superhuman performance
on a diverse set of tasks (49 Atari games) from raw pixel input alone —
using the same network architecture, hyperparameters, and learning algorithm
for every game. Published by Mnih et al. at DeepMind in Nature (2015).

It is the direct bridge between tabular Q-Learning and modern deep RL.
Every major deep RL algorithm that followed — DDPG, TD3, SAC, Rainbow —
inherits one or both of DQN's two core ideas: experience replay and the
target network.

Understanding DQN deeply means understanding:
  * Why function approximation is hard (the deadly triad)
  * What experience replay actually fixes (and what it doesn't)
  * Why target networks work (and when they fail)
  * How the loss and gradient flow differ from supervised learning
  * The full family of improvements that culminated in Rainbow
"""

import re

TOPIC_NAME   = "Deep Q-Network (DQN)"
DISPLAY_NAME = "03 · Deep Q-Network (DQN)"
ICON         = "🕹️"
SUBTITLE     = "Neural Q-value approximation — from pixels to superhuman Atari performance"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — WHY TABULAR Q-LEARNING BREAKS DOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Dimensionality Wall

Tabular Q-Learning requires one row per state in Q ∈ R^{|S| × |A|}.

For toy environments this is fine:
    4×4 Grid World:   |S| = 16        →    Q ∈ R^{16 × 4}     (trivial)

For real environments it is impossible:
    Atari Pong (84×84 greyscale, 4 frames stacked):
        |S| = 256^{84×84×4} ≈ 10^{67970}   (astronomically large)

    MuJoCo continuous control (28-dimensional joint state):
        |S| = ∞  (continuous)

Even if the table could be stored, each state would be visited at most
once — no generalisation across similar states.

Solution: replace the table with a parametric function:

        Q(s, a)  →  Q_θ(s, a)    where θ are learnable parameters

The function approximator (neural network) generalises:
updates from one state automatically inform nearby states.


### The Deadly Triad

When combining three properties, value-based RL can diverge:

    1. FUNCTION APPROXIMATION   (neural network, linear features)
    2. BOOTSTRAPPING            (TD target uses own Q estimates)
    3. OFF-POLICY LEARNING      (behaviour ≠ target policy)

Baird's counterexample (1995) proved that even linear function
approximation with TD and off-policy data can diverge to infinity.

Tabular Q-Learning avoids divergence because:
    * With a table, each Q(s,a) is updated independently
    * There is no generalisation to corrupt neighbouring values

With a neural network:
    * Updating Q(s,a) changes Q(s',a') for all (s',a') — cannot be isolated
    * A single bad bootstrap target poisons the gradient for many states

DQN's two key contributions (experience replay + target network)
are engineering solutions to keep the deadly triad stable.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — NEURAL NETWORK Q-VALUE APPROXIMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Architecture Design

Two natural architectures exist:

    ARCHITECTURE A — separate Q per action:
        Q_θ(s, a₁) = f_θ₁(s)
        Q_θ(s, a₂) = f_θ₂(s)   (one network per action)

    ARCHITECTURE B — all Q-values in one forward pass (DQN uses this):
        Q_θ(s, ·) : R^d → R^|A|
        One forward pass produces Q(s, a) for all actions simultaneously.

Architecture B is far more efficient. One forward pass computes all Q-values
needed for ε-greedy action selection and for finding max_a' Q(s', a').


### DQN Architecture for Atari

    INPUT            84×84×4 greyscale frames (4 stacked)
    Conv1            32 filters, 8×8 kernel, stride 4, ReLU
    Conv2            64 filters, 4×4 kernel, stride 2, ReLU
    Conv3            64 filters, 3×3 kernel, stride 1, ReLU
    Flatten          3136-dimensional vector
    FC1              512 units, ReLU
    FC2 (output)     |A| units, linear activation (Q-values, no softmax)

Total parameters: ~1.7M

Key choices:
    * Three conv layers extract spatial features from pixel input
    * No pooling (preserves spatial resolution for RL)
    * Linear output — Q-values can be any real number, not probabilities
    * 4 stacked frames provide temporal information (velocity, direction)


### Why Linear Output?

Q-values represent expected returns, which are unbounded real numbers.
A softmax or sigmoid output would constrain them to [0, 1] or (0, 1),
which is wrong — Q-values can be large positive, large negative, or near 0
depending on the reward scale and discount factor.

Compare to classification:
    Classification output: softmax → probabilities → cross-entropy loss
    Q-value output:        linear  → real numbers  → mean squared error


### Generalisation and the Forgetting Problem

Function approximation enables generalisation:
    States that look similar get similar Q-values automatically.
    The agent learns "objects that look like pacman power pellets
    are good to collect" from a few examples.

But it also introduces catastrophic forgetting:
    Gradient updates on one (s, a) shift Q for ALL (s', a') pairs
    that share weight activations. The network "forgets" old estimates
    when trained heavily on new data.

Mitigations: experience replay (revisit old data), target network
(stabilise supervision), larger replay buffers.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — EXPERIENCE REPLAY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Definition

Experience replay (Lin, 1992) stores agent transitions in a circular
buffer D (the replay memory):

        D = { (s₁, a₁, r₁, s₁', done₁),
              (s₂, a₂, r₂, s₂', done₂),
              ...
              (sₙ, aₙ, rₙ, sₙ', doneₙ) }

At each training step, a random mini-batch B is sampled uniformly:

        B ~ Uniform(D)    with replacement, |B| = 32 (typical)

The network is trained on this mini-batch using gradient descent.


### Problem 1 — Correlated Samples

Without replay, consecutive transitions (sₜ, sₜ₊₁, sₜ₊₂, ...) are
heavily correlated — they come from the same trajectory, same part of the
environment, same time in training.

Training on correlated sequences causes:
    * Gradients pointing in similar directions → inefficient optimisation
    * Distribution shift: network specialises on recent experience,
      forgets how to handle states from earlier in training
    * Oscillation: small changes in a run propagate through all recent data

Random sampling from a large buffer breaks these correlations — each
mini-batch is approximately i.i.d., like a supervised learning dataset.


### Problem 2 — Sample Efficiency

Online learning discards each transition after a single gradient update.
With experience replay, each transition participates in many updates:

    Expected times each transition is used = buffer_size / batch_size

    Buffer = 1,000,000,  Batch = 32  →  each transition used ~31,250 times

This is the same intuition as shuffling a dataset and training for
multiple epochs in supervised learning.


### Replay Buffer Details

    CAPACITY        Typically 10⁵ to 10⁶ transitions.
                    Larger is generally better (more diverse, less correlated).
                    Limited by available RAM: 1M transitions × (s,a,r,s',done)
                    ≈ 10GB for Atari (raw pixels). Compressed with uint8.

    CIRCULAR BUFFER When full, oldest transitions are overwritten.
                    Recent transitions are naturally more likely to be used.

    WARMUP PERIOD   Collect N_warmup transitions (e.g. 50K) with random
                    policy before training begins. Ensures the buffer has
                    diverse initial content before any Q-updates.

    OFF-POLICY BIAS The replay buffer contains transitions from past policies
                    (behaviour policies that may differ from current π).
                    Q-Learning tolerates this because it is off-policy.
                    On-policy methods (SARSA, PPO) cannot use vanilla replay.


### What Replay Does NOT Fix

Replay addresses temporal correlation and sample efficiency.
It does not address:
    * The moving target problem (bootstrapping with a changing network)
    * Overestimation bias (still present without Double DQN)
    * Catastrophic interference (network updates still corrupt old estimates)

The target network addresses the moving target problem.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — THE TARGET NETWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Moving Target Problem

Without a target network, the DQN loss is:

        L(θ) = E [ (r + γ max_a' Q_θ(s', a') − Q_θ(s, a))² ]

Both the prediction Q_θ(s,a) and the target r + γ max_a' Q_θ(s',a')
are functions of the SAME parameters θ.

Every gradient step on θ moves BOTH the prediction AND the target.
The target is a moving goalposts problem:

    The agent is trying to hit a target that moves every time
    it takes a step toward it. This creates oscillation and can
    diverge, especially with nonlinear function approximators.

Analogy: trying to measure your own height using a ruler that grows
taller every time you stand taller.


### Solution: Frozen Target Network

Maintain two networks with identical architecture:

    ONLINE NETWORK  Q_θ      updated by gradient descent at every step
    TARGET NETWORK  Q_θ⁻     frozen for C steps (periodic copy)

The target is computed using the frozen network:

        y = r + γ max_a' Q_θ⁻(s', a')

Loss:   L(θ) = E [ (y − Q_θ(s, a))² ]

The target y is treated as a constant (stop-gradient), like a
supervised regression label. The gradient flows only through Q_θ(s,a).

Periodic copy (hard update):
        Every C steps:   θ⁻ ← θ
        Between copies:  θ⁻ is frozen

DQN (2015) used C = 10,000 steps.


### Soft Target Update (Polyak Averaging)

An alternative to periodic hard copies, used in DDPG, TD3, SAC:

        θ⁻ ← τ θ + (1 − τ) θ⁻    at every step

    τ ≈ 0.005   (slow exponential moving average)

The target network parameters lag the online network with an
exponential smoothing factor. Provides a more continuously stable
target compared to sudden hard copies.


### What the Target Network Does NOT Fix

The target network reduces the moving target problem but does not
eliminate it entirely:
    * Hard updates still cause a sudden jump in targets every C steps
    * Systematic overestimation persists (requires Double DQN)
    * With very small C (e.g. 1), degenerates to no target network


### Ablation Evidence

DeepMind's original ablation study (2015):
    Full DQN (replay + target):           Median score ratio 79.2%
    DQN without replay:                   Median score ratio 30.1%
    DQN without target network:           Median score ratio 27.8%
    DQN without either:                   Median score ratio  1.8%

Neither replay nor target alone is sufficient. Both are necessary.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — THE DQN LOSS AND GRADIENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Semi-Gradient Update

The DQN loss is a mean squared error:

        L(θ) = E_{(s,a,r,s')~D} [ (y − Q_θ(s, a))² ]

        y = r + γ max_a' Q_θ⁻(s', a')    (target, treated as constant)

Taking the gradient with respect to θ:

        ∇_θ L(θ) = −2 E [ (y − Q_θ(s, a)) · ∇_θ Q_θ(s, a) ]

This is a SEMI-GRADIENT: the gradient is taken only through Q_θ(s, a),
not through y (which also depends on θ via θ⁻). The target y is detached.

Full gradient (not used):
        ∇_θ L_full(θ) = −2 E [ (y − Q_θ(s,a)) · (∇_θ Q_θ(s,a) − γ ∇_θ max_a' Q_θ(s',a')) ]

Using the full gradient is mathematically correct but empirically unstable.
The semi-gradient update corresponds to treating RL as a supervised
regression problem with pseudo-labels y.


### Gradient Clipping

DQN clips the semi-gradient before applying it:

        g ← clip(y − Q_θ(s, a), −1, 1)

This clips the TD error (Huber loss equivalent), not the parameter gradients.

The clipped update corresponds to using the Huber loss:
        L_δ(δ) = { 0.5 δ²          if |δ| ≤ 1
                 { |δ| − 0.5        if |δ| > 1

Benefits:
    * Prevents large TD errors from causing extreme weight updates
    * Robust to reward scale differences across Atari games
    * Equivalent to L2 for small errors, L1 for large errors

Reward clipping (r ← clip(r, −1, 1)) complements this: normalises
the reward scale across all 49 Atari games without game-specific tuning.


### Optimiser Choice

DQN (2015) used RMSProp:
        g² ← ρ g² + (1−ρ) (∇L)²
        θ  ← θ − (η / √(g² + ε)) · ∇L

    η = 0.00025,  ρ = 0.95,  ε = 0.01,  gradient clipping at 10.0

Modern implementations use Adam, which generally trains faster:
        m ← β₁ m + (1−β₁) ∇L
        v ← β₂ v + (1−β₂) (∇L)²
        θ ← θ − η · m̂ / (√v̂ + ε)

    Typical: η = 1e-4,  β₁ = 0.9,  β₂ = 0.999


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — FULL DQN ALGORITHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Pseudocode (Mnih et al., 2015)

    Initialise online network Q_θ with random weights θ
    Initialise target network Q_θ⁻ ← θ
    Initialise replay buffer D with capacity N

    For episode = 1 to M:
        s₁ ← preprocess(initial_frame)
        For t = 1 to T:
            ── Interact ─────────────────────────────────────────
            With probability ε:  aₜ ← random action
            Otherwise:           aₜ ← argmax_a Q_θ(sₜ, a)
            Execute aₜ, observe rₜ, frame xₜ₊₁
            sₜ₊₁ ← preprocess(xₜ₊₁, sₜ)
            Store (sₜ, aₜ, rₜ, sₜ₊₁, done) in D

            ── Train ────────────────────────────────────────────
            Sample random mini-batch of B transitions from D
            For each transition j:
                If terminal:  yⱼ = rⱼ
                Else:         yⱼ = rⱼ + γ max_a' Q_θ⁻(sⱼ', a')
            Loss:  L = (1/B) Σⱼ Huber(yⱼ − Q_θ(sⱼ, aⱼ))
            Gradient step on θ

            ── Update target ────────────────────────────────────
            Every C steps:  θ⁻ ← θ


### Atari Preprocessing Pipeline

Raw Atari frames are 210×160 RGB at 60Hz. DQN preprocesses them:

    1. FRAME SKIPPING       Every action is repeated for 4 frames.
                            Reduces temporal redundancy; speeds training.

    2. GRAYSCALE            Convert RGB → single-channel greyscale.
                            Colour rarely matters for Atari game dynamics.

    3. DOWNSAMPLING         Resize to 84×84 pixels using bilinear interpolation.

    4. FRAME STACKING       Stack the last 4 processed frames: 84×84×4.
                            Provides velocity/direction (Markov property).

    5. REWARD CLIPPING      r ← clip(r, −1, +1).
                            Normalises across 49 games without game-specific
                            knowledge. Loses magnitude information but
                            enables a single set of hyperparameters.

    6. PIXEL NORMALISATION  Divide by 255 → [0, 1] float32. (in practice)


### Hyperparameters (Nature 2015)

    REPLAY BUFFER SIZE      N = 1,000,000 transitions
    MINI-BATCH SIZE         B = 32
    DISCOUNT FACTOR         γ = 0.99
    TARGET UPDATE FREQ      C = 10,000 steps (hard copy)
    LEARNING RATE           η = 0.00025 (RMSProp)
    INITIAL EPSILON         ε₀ = 1.0
    FINAL EPSILON           ε_f = 0.1
    EPSILON DECAY STEPS     1,000,000 steps
    WARMUP STEPS            50,000 (random policy, no training)
    TRAINING FRAMES         50,000,000 total frames
    FRAME SKIP              4
    HISTORY LENGTH          4 frames


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — THE DQN FAMILY: MAJOR IMPROVEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Double DQN (van Hasselt et al., 2015)

Problem:  max_a' Q_θ⁻(s', a') overestimates because Jensen's inequality
          applies to the maximum of noisy estimates.

Fix:      Decouple action selection from evaluation:
          yⱼ = rⱼ + γ Q_θ⁻(s', argmax_a' Q_θ(s', a'))

    The online network selects the action.
    The target network evaluates its value.

Result:   Reduces overestimation by ~50% on Atari.
          Improves median score ratio from 79.2% to 117.5%.


### Prioritised Experience Replay (Schaul et al., 2015)

Problem:  Uniform sampling wastes time on transitions the network
          has already learned well (small TD error, little to learn).

Fix:      Sample transitions proportional to |TD error|:
          P(i) = (|δᵢ| + ε)^α / Σⱼ (|δⱼ| + ε)^α

    High TD error → transition is surprising → sample more often.
    ε prevents zero probability for well-learned transitions.
    α ∈ [0,1] controls degree of prioritisation (α=0 → uniform).

Importance sampling weights correct the bias:
    wᵢ = (1 / (N · P(i)))^β

β annealed from 0.4 to 1.0 over training to correct for the
distribution shift introduced by non-uniform sampling.

Implementation: a sum-tree (binary heap) gives O(log N) sampling
and O(log N) priority update.

Result:  Median improvement ~40% over Double DQN. Especially large
         gains on sparse-reward environments.


### Dueling DQN (Wang et al., 2016)

Problem:  In many states, the choice of action barely matters
          (e.g. nothing dangerous nearby). The agent still needs
          to update Q for each action separately, wasting capacity.

Fix:      Decompose Q into state-value V and advantage A:

        Q(s, a; θ, α, β) = V(s; θ, β) + [ A(s, a; θ, α) − (1/|A|) Σ_a' A(s,a';θ,α) ]

    Shared CNN trunk → splits into two streams:
        β stream:  outputs V(s) — scalar
        α stream:  outputs A(s, ·) — |A|-vector

    The mean centering ensures identifiability (V and A cannot compensate
    each other by shifting together).

Result:  The V stream can update purely from reward signal regardless
         of which action was taken, accelerating value learning in states
         where actions are equivalent. 24% improvement over Double DQN.


### Noisy Networks (Fortunato et al., 2017)

Problem:  ε-greedy is undirected — it explores uniformly with no
          relationship to uncertainty in the Q-estimate.

Fix:      Replace deterministic weights with stochastic weights:
          wᵢⱼ = μᵢⱼ + σᵢⱼ · εᵢⱼ    where εᵢⱼ ~ N(0, 1)

    μ and σ are learned parameters. σᵢⱼ → 0 means the network
    is certain about that weight. The exploration is guided by
    parameter uncertainty — uncertain weights explore more.

    Can remove ε entirely: the network learns when to explore.

Result:  Particularly effective in hard exploration games
         (Montezuma's Revenge) where ε-greedy fails.


### Distributional RL — C51 (Bellemare et al., 2017)

Problem:  Q(s,a) = E[G] only captures the mean return. Two policies
          with the same mean but different variances are indistinguishable.

Fix:      Learn the full distribution of returns Z(s,a):
          Model Z as a discrete distribution over 51 fixed atoms
          z₁, z₂, ..., z₅₁ ∈ [V_min, V_max]

    The network outputs probabilities P(Z = zᵢ | s, a).
    The Bellman update becomes a distributional update with projection.

Result:  Captures risk (variance), multi-modal returns.
         Median improvement ~70% over Double DQN. State of the art at time.

QR-DQN (2018): uses quantile regression instead of fixed atoms.
IQN (2019):    implicit quantile networks — sample quantiles from
               a learned quantile function.


### Rainbow (Hessel et al., 2017)

Combines all five improvements into a single agent:
    1. Double Q-Learning     (anti-overestimation)
    2. Prioritised Replay    (important transitions first)
    3. Dueling Architecture  (V and A decomposition)
    4. Multi-step Returns    (3-step TD instead of 1-step)
    5. Distributional RL     (C51 return distributions)
    6. Noisy Networks        (parametric exploration)

Results on 57 Atari games:
    Rainbow:        Median human-normalised score  223%
    Best prior:     ~100% (roughly human level)
    DQN (2015):     ~78%

Ablation shows all six components contribute. The most important
single component is distributional RL; the least important is noisy nets.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — LIMITATIONS AND SUCCESSORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Key Limitations of DQN

    1. DISCRETE ACTIONS ONLY
       DQN outputs Q(s, a) for each discrete action.
       Continuous action spaces require actor-critic (DDPG, TD3, SAC).

    2. POOR SAMPLE EFFICIENCY
       DQN requires ~50M Atari frames (200 hours of play) for good
       performance. Humans learn games in minutes.
       Improvement: model-based RL (Dreamer, MuZero) achieves comparable
       performance with 100x fewer environment interactions.

    3. HARD EXPLORATION
       ε-greedy random exploration fails in sparse-reward environments
       (e.g. Montezuma's Revenge requires specific long-horizon sequences).
       Improvement: intrinsic motivation (RND, ICM), go-explore.

    4. PARTIAL OBSERVABILITY
       DQN uses frame stacking as a crude memory. Full POMDPs require
       recurrent networks (DRQN — DQN with LSTM instead of FC).

    5. OVERESTIMATION
       Even with Double DQN, systematic positive bias remains.
       Improvement: distributional RL learns the full return distribution.

    6. SINGLE AGENT, STATIONARY ENVIRONMENT
       Multi-agent extensions (QMIX, QPLEX) required for cooperative tasks.


### Successors Using DQN Ideas

    DRQN (2015)        DQN + LSTM for partial observability
    DDPG (2015)        Continuous actions via deterministic actor-critic
    Ape-X (2018)       Distributed DQN with prioritised replay at scale
    R2D2 (2019)        Recurrent Ape-X with stored hidden states
    Agent57 (2020)     First agent to beat human baseline on all 57 Atari
    EfficientZero (2021) Model-based, matches DQN at 100x fewer samples


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Core Equations

    DQN LOSS (Huber):
        L(θ) = E_{B~D} [ Huber(yⱼ − Q_θ(sⱼ, aⱼ)) ]

    DQN TARGET:
        yⱼ = rⱼ + γ (1 − doneⱼ) max_a' Q_θ⁻(sⱼ', a')

    DOUBLE DQN TARGET:
        yⱼ = rⱼ + γ Q_θ⁻( sⱼ', argmax_a' Q_θ(sⱼ', a') )

    SOFT TARGET UPDATE:
        θ⁻ ← τ θ + (1 − τ) θ⁻

    DUELING DECOMPOSITION:
        Q(s, a) = V(s) + A(s, a) − mean_a' A(s, a')

    PRIORITISED SAMPLING:
        P(i) = (|δᵢ| + ε)^α / Σⱼ (|δⱼ| + ε)^α

    IS WEIGHT CORRECTION:
        wᵢ = ( N · P(i) )^{−β}


### DQN Family Summary

    +-------------------------+-----------+--------------------------------------------+
    | Algorithm               | Year      | Key Contribution                           |
    +-------------------------+-----------+--------------------------------------------+
    | DQN                     | 2013/2015 | Replay + target network                    |
    | Double DQN              | 2015      | Decouple select/evaluate, less bias        |
    | Prioritised DQN         | 2015      | Sample by |TD error|                       |
    | Dueling DQN             | 2016      | V + A streams, better V estimation         |
    | DRQN                    | 2015      | LSTM, handles partial observability        |
    | Noisy DQN               | 2017      | Parametric stochastic exploration          |
    | C51 / Distributional    | 2017      | Learn return distribution, not mean        |
    | Rainbow                 | 2017      | All improvements combined                  |
    | QR-DQN                  | 2018      | Quantile regression for distributions      |
    | IQN                     | 2019      | Implicit quantile network                  |
    | Ape-X                   | 2018      | Distributed actors, centralised learning   |
    | Agent57                 | 2020      | Beats human on all 57 Atari games          |
    +-------------------------+-----------+--------------------------------------------+

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "1. Build a Q-Network in NumPy (Forward and Backward Pass)": {
        "description": "Implement a 2-layer neural network for Q-value approximation "
                       "using only NumPy. Demonstrates forward pass, Huber loss, "
                       "semi-gradient backpropagation, and parameter update.",
        "code": """\
import numpy as np

class QNetwork:
    \"\"\"Two-layer MLP: input_dim -> hidden_dim -> n_actions (linear output).\"\"\"
    def __init__(self, input_dim, hidden_dim, n_actions, rng):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, scale1, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (hidden_dim, n_actions)).astype(np.float32)
        self.b2 = np.zeros(n_actions, dtype=np.float32)
        # Adam state
        self.mW1=np.zeros_like(self.W1); self.vW1=np.zeros_like(self.W1)
        self.mb1=np.zeros_like(self.b1); self.vb1=np.zeros_like(self.b1)
        self.mW2=np.zeros_like(self.W2); self.vW2=np.zeros_like(self.W2)
        self.mb2=np.zeros_like(self.b2); self.vb2=np.zeros_like(self.b2)
        self.t = 0

    def forward(self, x):
        \"\"\"x: (batch, input_dim) -> (batch, n_actions)\"\"\"
        self.x  = x
        self.z1 = x @ self.W1 + self.b1          # (batch, hidden)
        self.h1 = np.maximum(0, self.z1)          # ReLU
        self.z2 = self.h1 @ self.W2 + self.b2     # (batch, n_actions)
        return self.z2

    def backward_and_update(self, dL_dz2, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        \"\"\"Semi-gradient update: dL_dz2 = -(y - Q(s,a)) for selected actions.\"\"\"
        B = self.x.shape[0]
        dW2 = self.h1.T @ dL_dz2 / B
        db2 = dL_dz2.mean(axis=0)
        dh1 = dL_dz2 @ self.W2.T
        dz1 = dh1 * (self.z1 > 0)               # ReLU grad
        dW1 = self.x.T @ dz1 / B
        db1 = dz1.mean(axis=0)

        self.t += 1
        for (p, g, m, v) in [(self.W1,dW1,self.mW1,self.vW1),
                               (self.b1,db1,self.mb1,self.vb1),
                               (self.W2,dW2,self.mW2,self.vW2),
                               (self.b2,db2,self.mb2,self.vb2)]:
            m[:] = beta1*m + (1-beta1)*g
            v[:] = beta2*v + (1-beta2)*g**2
            m_hat = m / (1 - beta1**self.t)
            v_hat = v / (1 - beta2**self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_from(self, other):
        for a, b in [(self.W1,other.W1),(self.b1,other.b1),
                     (self.W2,other.W2),(self.b2,other.b2)]:
            a[:] = b

def huber_loss(delta, clip=1.0):
    abs_d = np.abs(delta)
    return np.where(abs_d <= clip, 0.5*delta**2, clip*(abs_d - 0.5*clip))

# ── Demonstrate forward/backward on a random batch ─────────────
rng        = np.random.default_rng(0)
INPUT_DIM  = 16     # one-hot state for 4x4 grid
HIDDEN_DIM = 32
N_ACTIONS  = 4

net    = QNetwork(INPUT_DIM, HIDDEN_DIM, N_ACTIONS, rng)
target = QNetwork(INPUT_DIM, HIDDEN_DIM, N_ACTIONS, rng)
target.copy_from(net)

BATCH  = 8
GAMMA  = 0.99

# Fake batch of transitions
s_idx  = rng.integers(0, INPUT_DIM, BATCH)
a_idx  = rng.integers(0, N_ACTIONS, BATCH)
r      = rng.uniform(-0.1, 1.0, BATCH).astype(np.float32)
s2_idx = rng.integers(0, INPUT_DIM, BATCH)
done   = (rng.random(BATCH) < 0.2).astype(np.float32)

# One-hot encode states
S  = np.eye(INPUT_DIM, dtype=np.float32)[s_idx]
S2 = np.eye(INPUT_DIM, dtype=np.float32)[s2_idx]

# Forward passes
Q_pred   = net.forward(S)                              # (B, A)
Q_next   = target.forward(S2)                          # (B, A)
y        = r + GAMMA * Q_next.max(axis=1) * (1-done)  # TD targets

# Semi-gradient: loss only through Q_pred
Q_sa     = Q_pred[np.arange(BATCH), a_idx]            # selected Q-values
delta    = Q_sa - y                                    # prediction - target
loss     = huber_loss(delta).mean()

# Gradient of loss w.r.t. z2 (only for the selected action)
dL_dz2   = np.zeros_like(Q_pred)
dL_dz2[np.arange(BATCH), a_idx] = delta               # dHuber/ddelta * ddelta/dQ

print("Q-Network Forward/Backward Pass")
print("=" * 46)
print(f"  Arch:   {INPUT_DIM} -> {HIDDEN_DIM} (ReLU) -> {N_ACTIONS}")
print(f"  Params: {net.W1.size + net.b1.size + net.W2.size + net.b2.size} total")
print()
print(f"  Batch size:  {BATCH}")
print(f"  Q_pred shape: {Q_pred.shape}")
print()
print("  Transition details (first 4):")
print(f"  {'s':>4} {'a':>4} {'r':>8} {'s2':>4} {'done':>6} {'y':>8} {'Q(s,a)':>8} {'delta':>8}")
print("  " + "-"*56)
for i in range(4):
    print(f"  {s_idx[i]:>4} {a_idx[i]:>4} {r[i]:>8.4f} {s2_idx[i]:>4} "
          f"{done[i]:>6.0f} {y[i]:>8.4f} {Q_sa[i]:>8.4f} {delta[i]:>8.4f}")

print()
print(f"  Huber loss before update: {loss:.6f}")

# Update and check loss decreased
net.backward_and_update(dL_dz2, lr=1e-3)
Q_pred2 = net.forward(S)
Q_sa2   = Q_pred2[np.arange(BATCH), a_idx]
loss2   = huber_loss(Q_sa2 - y).mean()
print(f"  Huber loss after  update: {loss2:.6f}  (should decrease)")
print(f"  Loss reduced: {loss > loss2}")
""",
    },

    "2. Replay Buffer — Uniform and Prioritised": {
        "description": "Implement both a standard uniform replay buffer and a prioritised "
                       "replay buffer using a sum-tree. Compare sampling distributions "
                       "and demonstrate importance-sampling weight correction.",
        "code": """\
import numpy as np

# ── Uniform Replay Buffer ──────────────────────────────────────
class UniformReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.cap  = capacity
        self.ptr  = 0
        self.size = 0
        self.s    = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a    = np.zeros(capacity, dtype=np.int32)
        self.r    = np.zeros(capacity, dtype=np.float32)
        self.s2   = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def push(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i] = s; self.a[i] = a; self.r[i] = r
        self.s2[i] = s2; self.done[i] = done
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.size, batch)
        return (self.s[idx], self.a[idx], self.r[idx],
                self.s2[idx], self.done[idx], idx,
                np.ones(batch, dtype=np.float32))  # uniform weights = 1


# ── Sum-Tree for Prioritised Replay ───────────────────────────
class SumTree:
    def __init__(self, capacity):
        self.cap  = capacity
        self.tree = np.zeros(2*capacity - 1, dtype=np.float64)
        self.ptr  = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx, priority):
        tree_idx = idx + self.cap - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, s):
        idx = 0
        while True:
            left  = 2*idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx - (self.cap - 1)
            idx = left if s <= self.tree[left] else right
            if idx == left: pass
            else: s -= self.tree[left]

    @property
    def total(self):
        return self.tree[0]


class PrioritisedReplayBuffer:
    def __init__(self, capacity, obs_dim, alpha=0.6, beta=0.4, eps=1e-6):
        self.cap   = capacity
        self.tree  = SumTree(capacity)
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        self.size  = 0
        self.s     = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a     = np.zeros(capacity, dtype=np.int32)
        self.r     = np.zeros(capacity, dtype=np.float32)
        self.s2    = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done  = np.zeros(capacity, dtype=np.float32)
        self.max_p = 1.0

    def push(self, s, a, r, s2, done):
        idx = self.tree.ptr % self.cap
        self.s[idx]=s; self.a[idx]=a; self.r[idx]=r
        self.s2[idx]=s2; self.done[idx]=done
        self.tree.update(idx, self.max_p ** self.alpha)
        self.tree.ptr = (self.tree.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch, rng):
        total = self.tree.total
        segment = total / batch
        idx = np.array([self.tree.get(rng.uniform(i*segment, (i+1)*segment))
                        for i in range(batch)], dtype=np.int32)
        priorities = np.array([self.tree.tree[i + self.cap - 1] for i in idx])
        probs = priorities / total
        weights = (self.size * probs) ** -self.beta
        weights /= weights.max()
        return (self.s[idx], self.a[idx], self.r[idx],
                self.s2[idx], self.done[idx], idx,
                weights.astype(np.float32))

    def update_priorities(self, idx, td_errors):
        for i, e in zip(idx, td_errors):
            p = (abs(e) + self.eps) ** self.alpha
            self.tree.update(i, p)
            self.max_p = max(self.max_p, p)


# ── Fill buffers and compare ───────────────────────────────────
rng = np.random.default_rng(42)
OBS_DIM  = 16
CAPACITY = 1000
BATCH    = 32

uniform_buf = UniformReplayBuffer(CAPACITY, OBS_DIM)
prior_buf   = PrioritisedReplayBuffer(CAPACITY, OBS_DIM)

# Push 1000 transitions with varying TD errors
td_errors_pushed = rng.exponential(0.5, CAPACITY)  # some high, most low
for i in range(CAPACITY):
    s    = rng.integers(0, OBS_DIM)
    s_oh = np.eye(OBS_DIM, dtype=np.float32)[s]
    s2   = rng.integers(0, OBS_DIM)
    s2oh = np.eye(OBS_DIM, dtype=np.float32)[s2]
    a    = rng.integers(0, 4)
    r    = float(rng.uniform(-0.1, 1.0))
    done = float(rng.random() < 0.1)
    uniform_buf.push(s_oh, a, r, s2oh, done)
    prior_buf.push(s_oh, a, r, s2oh, done)
    prior_buf.update_priorities([i % CAPACITY], [td_errors_pushed[i]])

# Sample and compare
_, _, _, _, _, u_idx, u_w = uniform_buf.sample(BATCH, rng)
_, _, _, _, _, p_idx, p_w = prior_buf.sample(BATCH, rng)

print("Replay Buffer Comparison")
print("=" * 52)
print(f"  Capacity: {CAPACITY}  |  Batch: {BATCH}")
print()
print(f"  Uniform buffer sample:")
print(f"    Idx range:     [{u_idx.min()}, {u_idx.max()}]  (uniform across buffer)")
print(f"    Weights min/max: {u_w.min():.3f} / {u_w.max():.3f}  (all = 1)")
print()
print(f"  Prioritised buffer sample (alpha=0.6):")
print(f"    Idx range:     [{p_idx.min()}, {p_idx.max()}]")
print(f"    IS weights min: {p_w.min():.4f}  max: {p_w.max():.4f}")
print(f"    IS weights > 0.9: {(p_w > 0.9).sum()} (low-priority samples get high IS weight)")
print()

# Show that high-TD transitions are sampled more
# Bucket td_errors into quartiles and count priority samples in each
quartiles = np.percentile(td_errors_pushed, [25, 50, 75])
labels    = ['Q1 (low TD)', 'Q2', 'Q3', 'Q4 (high TD)']
buckets   = np.digitize(td_errors_pushed[p_idx], quartiles)
print("  Prioritised sample distribution by TD-error quartile:")
for q in range(4):
    cnt = (buckets == q).sum()
    bar = '█' * cnt
    print(f"    {labels[q]:>15}: {cnt:>3}  {bar}")

print()
print("  Key insight: high-TD transitions sampled far more often,")
print("  IS weights compensate to prevent biased gradient estimates.")
""",
    },

    "3. Target Network — Hard vs Soft Update Comparison": {
        "description": "Demonstrate the effect of target network update strategy "
                       "on training stability. Compare hard copy (every C steps) "
                       "vs soft Polyak update on Q-value convergence and loss variance.",
        "code": """\
import numpy as np

# ── Minimal Q-Network ─────────────────────────────────────────
class TinyQNet:
    def __init__(self, n_s, n_a, rng, hidden=32):
        s1 = np.sqrt(2/n_s); s2 = np.sqrt(2/hidden)
        self.W1 = rng.normal(0, s1, (n_s, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = rng.normal(0, s2, (hidden, n_a)).astype(np.float32)
        self.b2 = np.zeros(n_a, np.float32)

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def copy_from(self, other):
        self.W1[:]=other.W1; self.b1[:]=other.b1
        self.W2[:]=other.W2; self.b2[:]=other.b2

    def soft_update_from(self, other, tau=0.005):
        self.W1[:] = tau*other.W1 + (1-tau)*self.W1
        self.b1[:] = tau*other.b1 + (1-tau)*self.b1
        self.W2[:] = tau*other.W2 + (1-tau)*self.W2
        self.b2[:] = tau*other.b2 + (1-tau)*self.b2

    def sgd_step(self, x, a_idx, y, lr=5e-4):
        \"\"\"Single minibatch SGD step (simplified backprop).\"\"\"
        B = x.shape[0]; n_a = self.W2.shape[1]
        h  = np.maximum(0, x @ self.W1 + self.b1)
        z2 = h @ self.W2 + self.b2
        # Loss only on selected actions
        Qsa  = z2[np.arange(B), a_idx]
        delt = Qsa - y
        dz2  = np.zeros_like(z2)
        dz2[np.arange(B), a_idx] = delt
        dW2  = h.T @ dz2 / B
        dh   = dz2 @ self.W2.T
        dz1  = dh * (h > 0)
        dW1  = x.T @ dz1 / B
        db1  = dz1.mean(0); db2 = dz2.mean(0)
        self.W1 -= lr*dW1; self.b1 -= lr*db1
        self.W2 -= lr*dW2; self.b2 -= lr*db2
        return np.mean(delt**2)


# ── Grid World Env (reused) ────────────────────────────────────
ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

def train(update_mode, C=50, tau=0.005, n_steps=10000, seed=0):
    rng  = np.random.default_rng(seed)
    online = TinyQNet(N_S, N_A, rng)
    tgt    = TinyQNet(N_S, N_A, rng)
    tgt.copy_from(online)

    EPS=1.0; BUF=[]
    losses=[]; target_diffs=[]
    s=0; step_cnt=0

    while step_cnt < n_steps:
        x = np.eye(N_S,dtype=np.float32)[[s]]
        if rng.random() < EPS:
            a = rng.integers(N_A)
        else:
            a = online.forward(x)[0].argmax()
        s2,r,done = step(s,a)
        BUF.append((s,a,r,s2,float(done)))
        if len(BUF) > 2000: BUF.pop(0)
        s = 0 if done else s2
        EPS = max(0.05, EPS*0.9995)
        step_cnt += 1

        if len(BUF) < 64: continue
        idx = rng.integers(0, len(BUF), 32)
        batch = [BUF[i] for i in idx]
        ss  = np.eye(N_S,dtype=np.float32)[[b[0] for b in batch]]
        aa  = np.array([b[1] for b in batch])
        rr  = np.array([b[2] for b in batch], dtype=np.float32)
        ss2 = np.eye(N_S,dtype=np.float32)[[b[3] for b in batch]]
        dd  = np.array([b[4] for b in batch], dtype=np.float32)

        q_next = tgt.forward(ss2).max(axis=1)
        y      = rr + GAMMA * q_next * (1-dd)
        loss   = online.sgd_step(ss, aa, y)
        losses.append(loss)

        # Record |online - target| norm
        diff = (np.abs(online.W2 - tgt.W2).mean())
        target_diffs.append(diff)

        if update_mode == 'hard' and step_cnt % C == 0:
            tgt.copy_from(online)
        elif update_mode == 'soft':
            tgt.soft_update_from(online, tau)

    return losses, target_diffs

def smooth(x, w=200):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

print("Target Network Update Strategy Comparison")
print("=" * 52)

loss_hard, diff_hard = train('hard', C=50)
loss_soft, diff_soft = train('soft', tau=0.01)
loss_none, diff_none = train('hard', C=1)  # C=1 means copy every step (same as no target)

sm_hard = smooth(loss_hard); sm_soft = smooth(loss_soft); sm_none = smooth(loss_none)

print(f"  {'Step':>6} | {'No target (C=1)':>16} | {'Hard (C=50)':>12} | {'Soft (tau=0.01)':>16}")
print("  " + "-" * 58)
checkpoints = [99, 299, 499, 999, 1999, 4999, 9999]
for i in checkpoints:
    if i < len(sm_hard):
        print(f"  {i+1:>6} | {sm_none[i]:>16.5f} | {sm_hard[i]:>12.5f} | {sm_soft[i]:>16.5f}")

print()
print("  Final loss (avg last 500 steps):")
print(f"    No target:   {np.mean(loss_none[-500:]):.5f}  (highest — unstable targets)")
print(f"    Hard (C=50): {np.mean(loss_hard[-500:]):.5f}")
print(f"    Soft:        {np.mean(loss_soft[-500:]):.5f}")
print()
print("  |Online-Target| weight diff (avg last 500 steps):")
print(f"    Hard: {np.mean(diff_hard[-500:]):.5f}  (spikes at copy, then drifts)")
print(f"    Soft: {np.mean(diff_soft[-500:]):.5f}  (steady-state lag)")
""",
    },

    "4. Full DQN Training Loop on CartPole (NumPy Simulation)": {
        "description": "Train a full DQN with replay buffer and target network on a "
                       "simplified CartPole-like balance task simulated in NumPy. "
                       "Tracks episode length, loss, and Q-value convergence.",
        "code": """\
import numpy as np

# ── Simplified Pole Balance Task (continuous state, discrete actions) ─
# State: [cart_pos, cart_vel, pole_angle, pole_vel]  (4-dim continuous)
# Actions: 0 = push left, 1 = push right
# Episode ends when |angle| > 12 deg or |cart_pos| > 2.4

class PoleEnv:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4).astype(np.float32)
        return self.state.copy()

    def step(self, action):
        x, xd, th, thd = self.state
        F = 10.0 if action == 1 else -10.0
        mc, mp, l, g = 1.0, 0.1, 0.5, 9.8
        total_m = mc + mp
        tmp  = (F + mp*l*thd**2*np.sin(th)) / total_m
        th_acc = (g*np.sin(th) - np.cos(th)*tmp) / \
                 (l*(4/3 - mp*np.cos(th)**2/total_m))
        x_acc  = tmp - mp*l*th_acc*np.cos(th)/total_m
        dt = 0.02
        x   += dt*xd;   xd  += dt*x_acc
        th  += dt*thd;  thd += dt*th_acc
        self.state = np.array([x,xd,th,thd], dtype=np.float32)
        done = bool(abs(th) > 0.2094 or abs(x) > 2.4)  # 12 degrees
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done


# ── Q-Network ─────────────────────────────────────────────────
class QNet:
    def __init__(self, rng, in_d=4, hid=64, out_d=2):
        self.W1 = rng.normal(0, np.sqrt(2/in_d),  (in_d, hid)).astype(np.float32)
        self.b1 = np.zeros(hid, np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/hid),   (hid, hid)).astype(np.float32)
        self.b2 = np.zeros(hid, np.float32)
        self.W3 = rng.normal(0, np.sqrt(2/hid),   (hid, out_d)).astype(np.float32)
        self.b3 = np.zeros(out_d, np.float32)
        self.lr = 5e-4
        self._init_adam()

    def _init_adam(self):
        params = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def forward(self, x):
        self.x  = x
        self.z1 = x  @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = np.maximum(0, self.z2)
        return self.h2 @ self.W3 + self.b3

    def backward(self, dout):
        B = self.x.shape[0]
        dW3 = self.h2.T @ dout / B;  db3 = dout.mean(0)
        dh2 = dout @ self.W3.T
        dz2 = dh2 * (self.h2 > 0)
        dW2 = self.h1.T @ dz2 / B;   db2 = dz2.mean(0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self.h1 > 0)
        dW1 = self.x.T @ dz1 / B;    db1 = dz1.mean(0)
        grads = [dW1,db1,dW2,db2,dW3,db3]
        params= [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
        self.t += 1
        b1,b2,eps = 0.9,0.999,1e-8
        for i,(p,g) in enumerate(zip(params,grads)):
            self.m[i] = b1*self.m[i] + (1-b1)*g
            self.v[i] = b2*self.v[i] + (1-b2)*g**2
            mh = self.m[i]/(1-b1**self.t)
            vh = self.v[i]/(1-b2**self.t)
            p -= self.lr*mh/(np.sqrt(vh)+eps)

    def copy_from(self, other):
        for a,b in [(self.W1,other.W1),(self.b1,other.b1),
                    (self.W2,other.W2),(self.b2,other.b2),
                    (self.W3,other.W3),(self.b3,other.b3)]:
            a[:] = b


# ── DQN Training ──────────────────────────────────────────────
rng    = np.random.default_rng(42)
env    = PoleEnv(rng)
online = QNet(rng)
target = QNet(rng)
target.copy_from(online)

BUF_CAP=10000; BATCH=64; GAMMA=0.99; TARGET_C=100
EPS=1.0; EPS_MIN=0.05; EPS_DECAY=0.998
buf=[]; ep_lens=[]; losses=[]; step_cnt=0

for ep in range(400):
    s = env.reset(); ep_len=0
    while True:
        q = online.forward(s[None])[0]
        a = rng.integers(2) if rng.random() < EPS else q.argmax()
        s2, r, done = env.step(a)
        buf.append((s.copy(),a,r,s2.copy(),float(done)))
        if len(buf)>BUF_CAP: buf.pop(0)
        s = s2; ep_len+=1; step_cnt+=1

        if len(buf)>=BATCH:
            idx  = rng.integers(0,len(buf),BATCH)
            ss   = np.array([buf[i][0] for i in idx])
            aa   = np.array([buf[i][1] for i in idx])
            rr   = np.array([buf[i][2] for i in idx],dtype=np.float32)
            ss2  = np.array([buf[i][3] for i in idx])
            dd   = np.array([buf[i][4] for i in idx],dtype=np.float32)

            q_next = target.forward(ss2).max(axis=1)
            y      = rr + GAMMA*q_next*(1-dd)
            q_pred = online.forward(ss)
            Qsa    = q_pred[np.arange(BATCH),aa]
            delta  = Qsa - y
            # Huber clipping
            delta_clipped = np.clip(delta,-1,1)
            dout = np.zeros_like(q_pred)
            dout[np.arange(BATCH),aa] = delta_clipped
            online.backward(dout)
            losses.append(np.mean(delta**2))

        if step_cnt % TARGET_C == 0:
            target.copy_from(online)
        if done: break

    EPS = max(EPS_MIN, EPS*EPS_DECAY)
    ep_lens.append(ep_len)

def smooth(x, w=20):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

sm_len = smooth(ep_lens)
print("DQN on Pole-Balance Task (NumPy)")
print("=" * 48)
print(f"  Network: 4 -> 64 -> 64 -> 2  |  Batch: {BATCH}  |  Buffer: {BUF_CAP}")
print(f"  Target update: every {TARGET_C} steps  |  γ={GAMMA}")
print()
print(f"  {'Episode':>8} | {'Ep length':>10} | {'Smooth-20':>10} | {'ε':>8}")
print("  " + "-"*44)
for ep_i in [0,19,49,99,149,199,299,399]:
    eps_at = max(EPS_MIN, 1.0*(EPS_DECAY**ep_i))
    print(f"  {ep_i+1:>8} | {ep_lens[ep_i]:>10} | {sm_len[ep_i]:>10.1f} | {eps_at:>8.4f}")

print()
print(f"  Max episode length achieved: {max(ep_lens)}")
print(f"  Mean last 50 episodes:       {np.mean(ep_lens[-50:]):.1f}")
print(f"  Mean training loss (last 500 steps): {np.mean(losses[-500:]):.5f}")
print()
print("  (Untrained: ~8-12 steps. Trained: 100-200 steps.)")
""",
    },

    "5. Dueling DQN Architecture": {
        "description": "Implement the Dueling DQN network architecture with separate "
                       "value (V) and advantage (A) streams. Compare learning speed "
                       "and Q-value quality against a standard DQN on the grid world.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)


class StandardDQN:
    def __init__(self, n_s, n_a, rng, hidden=32):
        s1,s2=np.sqrt(2/n_s),np.sqrt(2/hidden)
        self.W1=rng.normal(0,s1,(n_s,hidden)).astype(np.float32)
        self.b1=np.zeros(hidden,np.float32)
        self.W2=rng.normal(0,s2,(hidden,n_a)).astype(np.float32)
        self.b2=np.zeros(n_a,np.float32)

    def forward(self,x):
        self.x=x; self.h=np.maximum(0,x@self.W1+self.b1)
        return self.h@self.W2+self.b2

    def update(self,dout,lr=3e-3):
        B=self.x.shape[0]
        dW2=self.h.T@dout/B; db2=dout.mean(0)
        dh=dout@self.W2.T; dz1=dh*(self.h>0)
        dW1=self.x.T@dz1/B; db1=dz1.mean(0)
        self.W1-=lr*dW1; self.b1-=lr*db1
        self.W2-=lr*dW2; self.b2-=lr*db2

    def copy_from(self,o):
        self.W1[:]=o.W1;self.b1[:]=o.b1
        self.W2[:]=o.W2;self.b2[:]=o.b2


class DuelingDQN:
    \"\"\"V-stream and A-stream share a trunk, then combine: Q = V + (A - mean(A)).\"\"\"
    def __init__(self, n_s, n_a, rng, hidden=32):
        s1,s2=np.sqrt(2/n_s),np.sqrt(2/hidden)
        # Shared trunk
        self.W1=rng.normal(0,s1,(n_s,hidden)).astype(np.float32)
        self.b1=np.zeros(hidden,np.float32)
        # Value stream: hidden -> 1
        self.Wv=rng.normal(0,s2,(hidden,1)).astype(np.float32)
        self.bv=np.zeros(1,np.float32)
        # Advantage stream: hidden -> n_a
        self.Wa=rng.normal(0,s2,(hidden,n_a)).astype(np.float32)
        self.ba=np.zeros(n_a,np.float32)

    def forward(self,x):
        self.x=x
        self.h=np.maximum(0,x@self.W1+self.b1)  # trunk
        V=self.h@self.Wv+self.bv                 # (B,1)
        A=self.h@self.Wa+self.ba                 # (B,n_a)
        self.V=V; self.A=A
        return V + (A - A.mean(axis=1,keepdims=True))

    def update(self,dout,lr=3e-3):
        B=self.x.shape[0]
        # dQ/dV = 1, dQ/dA_i = 1 - 1/|A| * delta_{ij}
        n_a=self.A.shape[1]
        dA = dout - dout.mean(axis=1,keepdims=True)
        dV = dout.sum(axis=1,keepdims=True)
        dWa=self.h.T@dA/B; dba=dA.mean(0)
        dWv=self.h.T@dV/B; dbv=dV.mean(0)
        dh=(dA@self.Wa.T + dV@self.Wv.T)*(self.h>0)
        dW1=self.x.T@dh/B; db1=dh.mean(0)
        for (p,g) in [(self.W1,dW1),(self.b1,db1),(self.Wv,dWv),
                      (self.bv,dbv),(self.Wa,dWa),(self.ba,dba)]:
            p -= lr*g

    def copy_from(self,o):
        self.W1[:]=o.W1;self.b1[:]=o.b1
        self.Wv[:]=o.Wv;self.bv[:]=o.bv
        self.Wa[:]=o.Wa;self.ba[:]=o.ba


def train_dqn(NetClass, seed=42, n_ep=2000):
    rng=np.random.default_rng(seed)
    net=NetClass(N_S,N_A,rng); tgt=NetClass(N_S,N_A,rng); tgt.copy_from(net)
    buf=[]; EPS=1.0; returns=[]; step_cnt=0
    for ep in range(n_ep):
        s=0; G=0.0; gt=1.0
        for _ in range(200):
            x=np.eye(N_S,dtype=np.float32)[[s]]
            a=rng.integers(N_A) if rng.random()<EPS else net.forward(x)[0].argmax()
            s2,r,done=step(s,a)
            buf.append((s,a,r,s2,float(done)))
            if len(buf)>5000: buf.pop(0)
            G+=gt*r; gt*=GAMMA; s=s2; step_cnt+=1
            if len(buf)>=32:
                idx=rng.integers(0,len(buf),32)
                b=[buf[i] for i in idx]
                ss=np.eye(N_S,dtype=np.float32)[[x[0] for x in b]]
                aa=np.array([x[1] for x in b])
                rr=np.array([x[2] for x in b],dtype=np.float32)
                ss2=np.eye(N_S,dtype=np.float32)[[x[3] for x in b]]
                dd=np.array([x[4] for x in b],dtype=np.float32)
                yn=tgt.forward(ss2).max(1); y=rr+GAMMA*yn*(1-dd)
                qp=net.forward(ss); Qsa=qp[np.arange(32),aa]
                d=np.clip(Qsa-y,-1,1); dout=np.zeros_like(qp)
                dout[np.arange(32),aa]=d; net.update(dout)
            if step_cnt%100==0: tgt.copy_from(net)
            if done: break
        EPS=max(0.01,EPS*0.996); returns.append(G)
    return returns, net

def smooth(x,w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

r_std, net_std = train_dqn(StandardDQN)
r_due, net_due = train_dqn(DuelingDQN)
sm_std=smooth(r_std); sm_due=smooth(r_due)

print("Dueling DQN vs Standard DQN")
print("=" * 52)
print()
print(f"  {'Episode':>8} | {'Standard (sm)':>14} | {'Dueling (sm)':>14}")
print("  " + "-"*42)
for i in [99,299,499,999,1499,1999]:
    print(f"  {i+1:>8} | {sm_std[i]:>14.4f} | {sm_due[i]:>14.4f}")

print()
print("  Final mean return (last 200 eps):")
print(f"    Standard DQN: {np.mean(r_std[-200:]):.4f}")
print(f"    Dueling DQN:  {np.mean(r_due[-200:]):.4f}")

# Show V and A separation on a few states
print()
print("  Value V(s) and max Advantage max A(s,a) for 4 states (Dueling):")
ACTIONS=['U','D','L','R']
for s in [0, 1, 10, 14]:
    x  = np.eye(N_S,dtype=np.float32)[[s]]
    q  = net_due.forward(x)[0]
    V  = net_due.V[0,0]
    A  = net_due.A[0]
    pi = ACTIONS[q.argmax()]
    print(f"    State {s:>2}: V={V:>7.3f}  best A={A.max():>7.3f}  Q_max={q.max():>7.3f}  π={pi}")
""",
    },

    "6. Distributional DQN — Learning Return Distributions": {
        "description": "Implement a simplified C51-style distributional Q-learning. "
                       "Instead of learning E[G], learn the full distribution P(G=z) "
                       "over a fixed set of atoms. Compare mean estimates with standard DQN.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)


# ── C51: 21 atoms from V_min to V_max ─────────────────────────
N_ATOMS = 21
V_MIN, V_MAX = -2.0, 2.0
atoms = np.linspace(V_MIN, V_MAX, N_ATOMS, dtype=np.float32)  # support
dz    = (V_MAX - V_MIN) / (N_ATOMS - 1)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

class C51Net:
    \"\"\"Outputs a probability distribution over N_ATOMS for each action.\"\"\"
    def __init__(self, n_s, n_a, rng, hidden=64):
        s1,s2=np.sqrt(2/n_s),np.sqrt(2/hidden)
        self.W1=rng.normal(0,s1,(n_s,hidden)).astype(np.float32)
        self.b1=np.zeros(hidden,np.float32)
        # Output: n_a * N_ATOMS logits
        self.W2=rng.normal(0,s2,(hidden,n_a*N_ATOMS)).astype(np.float32)
        self.b2=np.zeros(n_a*N_ATOMS,np.float32)
        self.n_a=n_a

    def forward(self, x):
        \"\"\"Returns (B, n_a, N_ATOMS) probability distributions.\"\"\"
        self.x=x
        self.h=np.maximum(0,x@self.W1+self.b1)
        logits=self.h@self.W2+self.b2  # (B, n_a*N_ATOMS)
        logits=logits.reshape(-1,self.n_a,N_ATOMS)
        return softmax(logits, axis=-1)  # (B, n_a, N_ATOMS)

    def q_values(self, x):
        \"\"\"Expected Q-value = sum_z z * P(Z=z).\"\"\"
        probs=self.forward(x)  # (B, n_a, N_ATOMS)
        return (probs * atoms[None,None,:]).sum(axis=-1)  # (B, n_a)

    def copy_from(self,o):
        self.W1[:]=o.W1;self.b1[:]=o.b1
        self.W2[:]=o.W2;self.b2[:]=o.b2

    def update(self, x, a_idx, target_probs, lr=1e-3):
        \"\"\"Cross-entropy loss on selected action distribution.\"\"\"
        B=x.shape[0]
        probs=self.forward(x)   # (B, n_a, N_ATOMS)
        sel  = probs[np.arange(B), a_idx]   # (B, N_ATOMS) — selected action
        # Cross-entropy gradient: d/d_logits = pred - target (after softmax)
        dsel = sel - target_probs            # (B, N_ATOMS)
        # Build full gradient (only for selected action)
        dlogits=np.zeros((B,self.n_a,N_ATOMS),dtype=np.float32)
        dlogits[np.arange(B),a_idx]=dsel
        dlogits=dlogits.reshape(B,-1)
        dW2=self.h.T@dlogits/B; db2=dlogits.mean(0)
        dh=(dlogits@self.W2.T)*(self.h>0)
        dW1=x.T@dh/B; db1=dh.mean(0)
        for p,g in [(self.W1,dW1),(self.b1,db1),(self.W2,dW2),(self.b2,db2)]:
            p -= lr*g
        return -np.sum(target_probs * np.log(sel+1e-8), axis=1).mean()


def project_distribution(r, done, next_probs_best, gamma=GAMMA):
    \"\"\"Bellman distributional projection onto the atom support.\"\"\"
    # next_probs_best: (B, N_ATOMS) probs of greedy action at s'
    B=r.shape[0]
    m = np.zeros((B, N_ATOMS), dtype=np.float32)
    for j in range(N_ATOMS):
        Tzj = np.clip(r + gamma*(1-done)*atoms[j], V_MIN, V_MAX)
        bj  = (Tzj - V_MIN) / dz         # fractional index
        lj  = np.floor(bj).astype(int)
        uj  = np.ceil(bj).astype(int)
        lj  = np.clip(lj, 0, N_ATOMS-1)
        uj  = np.clip(uj, 0, N_ATOMS-1)
        frac_u = bj - np.floor(bj)
        frac_l = 1.0 - frac_u
        for b in range(B):
            m[b, lj[b]] += next_probs_best[b, j] * frac_l[b]
            m[b, uj[b]] += next_probs_best[b, j] * frac_u[b]
    return m


# ── Training loop ──────────────────────────────────────────────
rng=np.random.default_rng(42)
net=C51Net(N_S,N_A,rng); tgt=C51Net(N_S,N_A,rng); tgt.copy_from(net)
buf=[]; EPS=1.0; returns=[]; losses=[]; step_cnt=0

for ep in range(1500):
    s=0; G=0.0; gt=1.0
    for _ in range(200):
        x=np.eye(N_S,dtype=np.float32)[[s]]
        a=rng.integers(N_A) if rng.random()<EPS else net.q_values(x)[0].argmax()
        s2,r,done=step(s,a)
        buf.append((s,a,r,s2,float(done)))
        if len(buf)>5000: buf.pop(0)
        G+=gt*r; gt*=GAMMA; s=s2; step_cnt+=1

        if len(buf)>=32:
            idx=rng.integers(0,len(buf),32)
            b=[buf[i] for i in idx]
            ss=np.eye(N_S,dtype=np.float32)[[x[0] for x in b]]
            aa=np.array([x[1] for x in b])
            rr=np.array([x[2] for x in b],dtype=np.float32)
            ss2=np.eye(N_S,dtype=np.float32)[[x[3] for x in b]]
            dd=np.array([x[4] for x in b],dtype=np.float32)

            # Target distribution via Bellman projection
            next_probs=tgt.forward(ss2)          # (B,n_a,N_ATOMS)
            next_q=tgt.q_values(ss2)             # (B,n_a)
            best_a=next_q.argmax(1)              # (B,)
            best_probs=next_probs[np.arange(32),best_a]  # (B,N_ATOMS)
            m=project_distribution(rr,dd,best_probs)
            loss=net.update(ss,aa,m)
            losses.append(loss)

        if step_cnt%100==0: tgt.copy_from(net)
        if done: break

    EPS=max(0.01,EPS*0.997); returns.append(G)

def smooth(x,w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]
sm=smooth(returns)

print("C51 Distributional DQN")
print("="*50)
print(f"  Atoms: {N_ATOMS}, V_min={V_MIN}, V_max={V_MAX}")
print()
print(f"  {'Episode':>8} | {'Return':>10} | {'Smooth-100':>12}")
print("  "+"-"*36)
for i in [99,299,499,999,1499]:
    print(f"  {i+1:>8} | {returns[i]:>10.4f} | {sm[i]:>12.4f}")

print()
# Inspect return distribution at a state near goal
s_near = 14
x = np.eye(N_S,dtype=np.float32)[[s_near]]
probs = net.forward(x)[0]   # (n_a, N_ATOMS)
ANAMES=['U','D','L','R']
print(f"  Return distributions at state {s_near} (adjacent to goal):")
for a_i in range(N_A):
    p = probs[a_i]
    best_atom = atoms[p.argmax()]
    ev = (p*atoms).sum()
    print(f"    {ANAMES[a_i]}: E[G]={ev:>7.3f}  mode_atom={best_atom:>6.3f}  "
          f"{'← best action' if ev==max((p*atoms).sum() for p in probs) else ''}")
""",
    },

    "7. DQN Ablation — Replay Buffer Size vs Performance": {
        "description": "Systematically vary replay buffer size (100, 500, 2000, 10000) "
                       "and measure the effect on learning speed and final policy quality. "
                       "Illustrates why large buffers are critical for DQN stability.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

class Net:
    def __init__(self, rng, hid=32):
        self.W1=rng.normal(0,np.sqrt(2/N_S),(N_S,hid)).astype(np.float32)
        self.b1=np.zeros(hid,np.float32)
        self.W2=rng.normal(0,np.sqrt(2/hid),(hid,N_A)).astype(np.float32)
        self.b2=np.zeros(N_A,np.float32)
    def forward(self,x):
        self.x=x; self.h=np.maximum(0,x@self.W1+self.b1)
        return self.h@self.W2+self.b2
    def update(self,dout,lr=3e-3):
        B=self.x.shape[0]
        dW2=self.h.T@dout/B; db2=dout.mean(0)
        dh=dout@self.W2.T; dz1=dh*(self.h>0)
        dW1=self.x.T@dz1/B; db1=dz1.mean(0)
        self.W1-=lr*dW1; self.b1-=lr*db1
        self.W2-=lr*dW2; self.b2-=lr*db2
    def copy_from(self,o):
        self.W1[:]=o.W1;self.b1[:]=o.b1
        self.W2[:]=o.W2;self.b2[:]=o.b2

def train(buf_cap, seed=0, n_ep=1500):
    rng=np.random.default_rng(seed)
    net=Net(rng); tgt=Net(rng); tgt.copy_from(net)
    buf=[]; EPS=1.0; returns=[]; step_cnt=0
    for ep in range(n_ep):
        s=0; G=0.0; gt=1.0
        for _ in range(200):
            x=np.eye(N_S,dtype=np.float32)[[s]]
            a=rng.integers(N_A) if rng.random()<EPS else net.forward(x)[0].argmax()
            s2,r,done=step(s,a)
            buf.append((s,a,r,s2,float(done)))
            if len(buf)>buf_cap: buf.pop(0)
            G+=gt*r; gt*=GAMMA; s=s2; step_cnt+=1
            if len(buf)>=32:
                idx=rng.integers(0,len(buf),32)
                b=[buf[i] for i in idx]
                ss=np.eye(N_S,dtype=np.float32)[[x[0] for x in b]]
                aa=np.array([x[1] for x in b])
                rr=np.array([x[2] for x in b],dtype=np.float32)
                ss2=np.eye(N_S,dtype=np.float32)[[x[3] for x in b]]
                dd=np.array([x[4] for x in b],dtype=np.float32)
                yn=tgt.forward(ss2).max(1); y=rr+GAMMA*yn*(1-dd)
                qp=net.forward(ss); Qsa=qp[np.arange(32),aa]
                d=np.clip(Qsa-y,-1,1); dout=np.zeros_like(qp)
                dout[np.arange(32),aa]=d; net.update(dout)
            if step_cnt%100==0: tgt.copy_from(net)
            if done: break
        EPS=max(0.01,EPS*0.996); returns.append(G)
    return returns

def smooth(x,w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

buf_sizes = [50, 200, 1000, 5000]
all_sm    = {}
final_perf= {}
for cap in buf_sizes:
    rets = train(cap)
    sm   = smooth(rets)
    all_sm[cap]    = sm
    final_perf[cap] = np.mean(rets[-200:])

print("DQN Ablation: Replay Buffer Size")
print("=" * 62)
print()
header = f"  {'Episode':>8} | " + " | ".join(f"buf={c:>5}" for c in buf_sizes)
print(header)
print("  " + "-"*(len(header)-2))
for ep_i in [99,299,499,999,1499]:
    vals=" | ".join(f"{all_sm[c][ep_i]:>9.4f}" for c in buf_sizes)
    print(f"  {ep_i+1:>8} | {vals}")

print()
print("  Final mean return (last 200 eps):")
for cap in buf_sizes:
    bar_len = int((final_perf[cap]+1.5)/2.5 * 25)
    bar = '█'*max(0,bar_len)
    print(f"    buf={cap:>5}: {final_perf[cap]:>7.4f}  {bar}")

print()
print("  Interpretation:")
print("    Small buffer: transitions overwritten quickly → correlated mini-batches")
print("    Large buffer: diverse, decorrelated samples  → stable gradient updates")
print("    Very small (50): agent effectively re-learns same transitions repeatedly")
""",
    },

    "8. DQN vs Q-Learning — Q-Value Divergence Stress Test": {
        "description": "Demonstrate Q-value divergence that occurs without the target "
                       "network when using function approximation. Shows why the deadly "
                       "triad is real and how the target network prevents Q-value explosion.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

class Net:
    def __init__(self,rng,hid=64):
        self.W1=rng.normal(0,np.sqrt(2/N_S),(N_S,hid)).astype(np.float32)
        self.b1=np.zeros(hid,np.float32)
        self.W2=rng.normal(0,np.sqrt(2/hid),(hid,N_A)).astype(np.float32)
        self.b2=np.zeros(N_A,np.float32)
    def forward(self,x):
        self.x=x; self.h=np.maximum(0,x@self.W1+self.b1)
        return self.h@self.W2+self.b2
    def update(self,dout,lr=1e-2):   # high LR to stress test
        B=self.x.shape[0]
        dW2=self.h.T@dout/B; db2=dout.mean(0)
        dh=dout@self.W2.T; dz1=dh*(self.h>0)
        dW1=self.x.T@dz1/B; db1=dz1.mean(0)
        self.W1-=lr*dW1; self.b1-=lr*db1
        self.W2-=lr*dW2; self.b2-=lr*db2
    def copy_from(self,o):
        self.W1[:]=o.W1;self.b1[:]=o.b1
        self.W2[:]=o.W2;self.b2[:]=o.b2

def run(use_target, use_replay, seed=0, n_steps=8000):
    rng=np.random.default_rng(seed)
    net=Net(rng)
    tgt=Net(rng); tgt.copy_from(net)
    buf=[]; s=0; EPS=0.3; step_cnt=0
    q_max_hist=[]; loss_hist=[]

    for _ in range(n_steps):
        x=np.eye(N_S,dtype=np.float32)[[s]]
        a=rng.integers(N_A) if rng.random()<EPS else net.forward(x)[0].argmax()
        s2,r,done=step(s,a)
        buf.append((s,a,r,s2,float(done)))
        if len(buf)>2000: buf.pop(0)
        s=0 if done else s2; step_cnt+=1

        if len(buf)>=32:
            if use_replay:
                idx=rng.integers(0,len(buf),32)
            else:
                idx=list(range(max(0,len(buf)-32),len(buf)))  # last 32 (correlated)
            b=[buf[i] for i in idx]
            ss=np.eye(N_S,dtype=np.float32)[[x[0] for x in b]]
            aa=np.array([x[1] for x in b])
            rr=np.array([x[2] for x in b],dtype=np.float32)
            ss2=np.eye(N_S,dtype=np.float32)[[x[3] for x in b]]
            dd=np.array([x[4] for x in b],dtype=np.float32)

            evaluator = tgt if use_target else net
            yn=evaluator.forward(ss2).max(1); y=rr+GAMMA*yn*(1-dd)
            qp=net.forward(ss); Qsa=qp[np.arange(len(b)),aa]
            d=Qsa-y; dout=np.zeros_like(qp)
            dout[np.arange(len(b)),aa]=d; net.update(dout)
            loss_hist.append(np.mean(d**2))

        if use_target and step_cnt%200==0:
            tgt.copy_from(net)

        # Track max Q-value (divergence indicator)
        all_x=np.eye(N_S,dtype=np.float32)
        q_max_hist.append(net.forward(all_x).max())

    return q_max_hist, loss_hist

configs = [
    (False, False, "No target, no replay  (worst)"),
    (False, True,  "Replay only           (unstable)"),
    (True,  False, "Target only           (unstable)"),
    (True,  True,  "Target + Replay (DQN) (stable)"),
]

print("Q-Value Divergence Stress Test (lr=0.01, high stress)")
print("=" * 60)
print(f"  Monitoring max Q-value across all states over 8000 steps")
print()

results = {}
for use_tgt, use_rep, label in configs:
    qmax, loss = run(use_tgt, use_rep)
    results[label] = (qmax, loss)

def smooth(x, w=200):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

checkpoints = [499, 999, 1999, 3999, 5999, 7999]
print(f"  {'Config':<32} | " + " | ".join(f"step {c+1:>5}" for c in checkpoints))
print("  " + "-"*(34 + len(checkpoints)*12))
for label,(qmax,_) in results.items():
    sm=smooth(qmax)
    vals=" | ".join(f"{sm[c]:>10.2f}" for c in checkpoints if c < len(sm))
    print(f"  {label:<32} | {vals}")

print()
print("  Final max |Q| (avg last 500 steps):")
for label,(qmax,_) in results.items():
    final=np.mean([abs(q) for q in qmax[-500:]])
    diverged = final > 20
    status = "DIVERGED" if diverged else "stable"
    print(f"    {label:<32} → {final:>8.2f}  [{status}]")

print()
print("  Key insight: only Target + Replay (DQN) avoids divergence.")
print("  The deadly triad (approx + bootstrap + off-policy) requires both fixes.")
""",
    },

}


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