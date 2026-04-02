"""Module: 06 · Actor-Critic"""

"""
Actor-Critic Methods
====================

Actor-Critic is the architectural family that underpins virtually every
modern deep RL algorithm — from A3C to PPO to SAC to RLHF. It sits at
the intersection of value-based and policy gradient methods, taking the
best of each:

  FROM POLICY GRADIENT:  direct policy optimisation, continuous actions,
                         stochastic policies, theoretical guarantees
  FROM VALUE LEARNING:   bootstrap estimates, lower variance, online updates,
                         no need for full episode rollouts

The core structure is simple: two components, two objectives, one loop.

  ACTOR   π_θ(a|s)   — the policy. Decides what to do.
  CRITIC  V_w(s)     — the value function. Evaluates how good the situation is.

The critic provides the baseline that reduces the variance of the actor's
gradient estimates. The actor's experience drives the critic toward better
value estimates. They improve together — the classic Generalised Policy
Iteration (GPI) pattern implemented with neural networks.

This module covers the full actor-critic family: from the one-step
formulation through advantage actor-critic (A2C/A3C), to the continuous
control variants (DDPG, TD3, SAC), shared-network architectures,
and the multi-agent extensions.
"""

import re

TOPIC_NAME   = "Actor-Critic Methods"
DISPLAY_NAME = "06 · Actor-Critic"
ICON         = "🎭"
SUBTITLE     = "Policy + value function together — the backbone of modern deep RL"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — WHY ACTOR-CRITIC?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Three Families of RL

RL algorithms occupy a spectrum defined by what they learn:

    VALUE-BASED ONLY     (Q-Learning, DQN)
        Learn Q*(s,a). Derive policy implicitly: π(s) = argmax Q(s,a).
        ✓ Sample efficient (off-policy replay)
        ✗ Discrete actions only, no stochastic policies

    POLICY GRADIENT ONLY (REINFORCE)
        Learn π_θ directly. No value function.
        ✓ Continuous actions, stochastic policies
        ✗ High variance — full episode MC returns required

    ACTOR-CRITIC         (A2C, PPO, SAC, TD3)
        Learn BOTH π_θ and V_w (or Q_w) simultaneously.
        ✓ Continuous actions, stochastic policies (from PG)
        ✓ Lower variance via bootstrapped advantage (from value learning)
        ✓ Online updates — no full episode needed


### The Variance-Bias Problem

REINFORCE updates:
        θ ← θ + α · Gₜ · ∇_θ log π_θ(aₜ|sₜ)

Gₜ is unbiased but has very high variance — it depends on all future
rewards, which are stochastic. Large variance → noisy gradients → slow,
unstable learning.

Actor-Critic replaces Gₜ with the advantage Âₜ = δₜ (TD error):

        δₜ = rₜ₊₁ + γ V_w(sₜ₊₁) − V_w(sₜ)

        θ ← θ + α · δₜ · ∇_θ log π_θ(aₜ|sₜ)

The TD error δₜ:
    * Is biased (V_w is an approximation, not the true value)
    * Has much lower variance (only one reward step + two value estimates)
    * Can be computed after EVERY step — no episode boundary required

The bias-variance tradeoff: replace unbiased high-variance MC returns
with biased low-variance TD estimates. In practice, the variance reduction
far outweighs the bias, leading to dramatically faster and more stable
learning.


### Generalised Policy Iteration with Neural Networks

Actor-Critic is precisely the neural network implementation of GPI:

    GPI (tabular):
        Alternate between:
          1. Policy Evaluation  →  compute V^π exactly
          2. Policy Improvement →  act greedily w.r.t. V

    Actor-Critic (neural):
        Simultaneously:
          1. Critic update  →  V_w(s) → r + γV_w(s')  [approximate V^π]
          2. Actor update   →  θ ← θ + α δ ∇ log π_θ  [move toward greedy]

The two updates happen concurrently on every step, rather than sequentially
in full sweeps. This is the essence of online RL.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — THE ONE-STEP ACTOR-CRITIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Simplest Actor-Critic

After each transition (sₜ, aₜ, rₜ₊₁, sₜ₊₁):

    1. Compute TD error (advantage estimate):
          δₜ = rₜ₊₁ + γ V_w(sₜ₊₁)(1 − done) − V_w(sₜ)

    2. Update critic (minimise squared TD error):
          w ← w + α_w · δₜ · ∇_w V_w(sₜ)

    3. Update actor (policy gradient with TD advantage):
          θ ← θ + α_θ · δₜ · ∇_θ log π_θ(aₜ|sₜ)

The TD error δₜ serves double duty:
    * As the regression target error for the critic (how wrong was V?)
    * As the advantage signal for the actor (was this action better than expected?)


### Why δₜ Is an Unbiased Advantage Estimate (at the true value function)

The advantage function:
        A^π(s,a) = Q^π(s,a) − V^π(s)
                 = E[rₜ₊₁ + γV^π(sₜ₊₁) | s,a] − V^π(s)

The TD error with true values:
        E[rₜ₊₁ + γV^π(sₜ₊₁) − V^π(sₜ) | sₜ=s, aₜ=a]
        = Q^π(s,a) − V^π(s)
        = A^π(s,a)

So E[δₜ | sₜ, aₜ] = A^π(sₜ, aₜ) when V_w = V^π (true values).

In practice V_w ≈ V^π, so δₜ is a biased but low-variance approximation.


### Learning Rates for Actor and Critic

In the one-step formulation, α_θ and α_w must be set carefully:

    α_w >> α_θ  (common practice):
        The critic should learn faster than the actor. If the actor updates
        based on an inaccurate critic, the gradient signal is misleading.
        Typical: α_w = 0.1, α_θ = 0.01.

    Two-timescale convergence theorem (Borkar 1997):
        Under mild conditions, two-timescale actor-critic converges to a
        local optimum if α_θ/α_w → 0 as both go to zero. The critic
        operates on the "fast timescale", the actor on the "slow timescale".


### Pseudocode

    Initialise θ (actor), w (critic)
    Repeat for each episode:
        s ← initial state
        Repeat for each step t:
            a ~ π_θ(·|s)
            s', r, done ← env.step(a)
            δ = r + γ V_w(s')(1-done) - V_w(s)
            w ← w + α_w δ ∇_w V_w(s)            ← critic update
            θ ← θ + α_θ δ ∇_θ log π_θ(a|s)      ← actor update
            s ← s'
        Until done


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — ADVANTAGE ACTOR-CRITIC (A2C AND A3C)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The N-Step Return

The one-step TD advantage has high bias (V_w is far from V^π early in training).
N-step returns trade some variance for lower bias:

        Rₜ^(n) = rₜ + γrₜ₊₁ + ... + γⁿ⁻¹rₜ₊ₙ₋₁ + γⁿ V_w(sₜ₊ₙ)

Advantage: Âₜ^(n) = Rₜ^(n) − V_w(sₜ)

    n = 1:   Standard TD advantage (low variance, high bias)
    n = T:   Monte Carlo (unbiased, high variance)
    n = 5-20: Typical practice — balances bias and variance


### A3C: Asynchronous Advantage Actor-Critic (Mnih et al., 2016)

A3C was a landmark result: demonstrated that asynchronous parallelism
with shared networks could achieve stability equivalent to experience
replay, without the memory cost.

Architecture:
    * 1 global network (θ_global, w_global)
    * N worker processes, each with local copy (θ_local, w_local)

Each worker:
    1. Copy global to local: θ_local ← θ_global
    2. Collect t_max steps in its own environment copy
    3. Compute n-step returns and gradients locally
    4. Push gradients to global (asynchronous, no lock)
    5. Global applies gradients (RMSProp with shared statistics)

Why asynchronous?
    * Different workers explore different parts of the environment
    * Concurrent exploration naturally decorrelates experience
    * No replay buffer needed — parallelism provides diversity
    * Scales linearly with number of CPU cores

A3C update (per worker):

    Compute: Rₜ = rₜ + γrₜ₊₁ + ... + γⁿ⁻¹rₜ₊ₙ₋₁ + γⁿ V_w(sₜ₊ₙ)
    Advantage: Âₜ = Rₜ − V_w(sₜ)
    Policy gradient: dθ = Σₜ ∇_θ log π_θ(aₜ|sₜ) · Âₜ + β ∇_θ H(π_θ(·|sₜ))
    Value gradient:  dw = Σₜ ∇_w (Rₜ − V_w(sₜ))²


### A2C: Synchronous Advantage Actor-Critic

A2C (synchronous version): wait for all N workers to finish their rollouts,
aggregate gradients, then apply one synchronous update.

    Advantages of A2C over A3C:
        * Reproducible: no race conditions or non-determinism
        * Better GPU utilisation: all workers update the same batch simultaneously
        * Simpler implementation
        * Empirically similar or better performance than A3C

    Modern practice: A2C on GPU with N=8-32 parallel environments.
    Most "A3C implementations" are actually A2C.


### Shared Network Architecture

A2C/A3C use a single network with two output heads:

        Input s
           ↓
    Shared encoder φ(s)      [CNN for pixels, MLP for state]
        ↙          ↘
    π head           V head
    π_θ(a|s)         V_w(s)
    (softmax)        (linear)

Loss function:

    L = L_policy + c_v · L_value + c_e · L_entropy

    L_policy  = −E[ log π_θ(a|s) · Â ]           (maximise)
    L_value   = E[ (Rₜ − V_w(s))² ]               (minimise)
    L_entropy = −H(π_θ(·|s))                       (maximise)

    Typical: c_v = 0.5, c_e = 0.01


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — CONTINUOUS CONTROL: DDPG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Continuous Action Problem

For continuous actions, the actor-critic architecture must handle:

        π*( s ) = argmax_a Q(s, a)      where a ∈ R^m

argmax over a continuous domain has no closed form. Two approaches:

    1. STOCHASTIC POLICY GRADIENT (REINFORCE / PPO):
       Sample a ~ π_θ(·|s), use log π gradient.
       Works for both discrete and continuous.

    2. DETERMINISTIC POLICY GRADIENT (DDPG):
       Use a deterministic policy μ_θ(s) ∈ R^m.
       Gradient is taken directly through Q.


### Deterministic Policy Gradient Theorem (Silver et al., 2014)

For a deterministic policy μ_θ: S → A:

        ∇_θ J(μ_θ) = E_s [ ∇_θ μ_θ(s) · ∇_a Q^μ(s,a)|_{a=μ_θ(s)} ]

This is the "deterministic policy gradient" — the gradient of J with
respect to the actor parameters θ, computed via the chain rule through
the Q function with respect to the action.

The key insight: ∇_a Q(s,a) tells the actor "in which direction should
I change my output action to increase Q?" The actor then adjusts its
weights in the direction that moves actions toward higher Q values.


### DDPG: Deep Deterministic Policy Gradient (Lillicrap et al., 2015)

DDPG combines DPG with DQN's stabilisation tricks:

    ACTOR   μ_θ(s) → deterministic action a ∈ R^m
    CRITIC  Q_w(s, a) → scalar Q-value

    REPLAY BUFFER  D  (same as DQN)
    TARGET NETWORKS  μ_θ⁻, Q_w⁻  (same as DQN)

Update rules:

    Critic:  L_Q = E [ (r + γ Q_w⁻(s', μ_θ⁻(s')) − Q_w(s, a))² ]
             w ← w − α_w ∇_w L_Q

    Actor:   J_μ = E_s [ Q_w(s, μ_θ(s)) ]
             θ ← θ + α_θ ∇_θ J_μ
                 = θ + α_θ E [ ∇_a Q_w(s,a)|_{a=μ} · ∇_θ μ_θ(s) ]

    Soft target updates (Polyak):
             θ⁻ ← τθ + (1−τ)θ⁻,    τ = 0.005
             w⁻ ← τw + (1−τ)w⁻

Exploration: deterministic policy has no built-in exploration.
Add Ornstein-Uhlenbeck (OU) noise for temporally correlated exploration:

        aₜ = μ_θ(sₜ) + Nₜ
        dNₜ = θ_OU(μ_OU − Nₜ)dt + σ_OU dWₜ

Or simply add Gaussian noise: aₜ = clip(μ_θ(sₜ) + ε, a_min, a_max).


### DDPG Limitations

    OVERESTIMATION BIAS
        Q_w(s, a) overestimates Q* because the argmax target uses the
        same network for selection and evaluation (same as DQN).

    HYPERPARAMETER SENSITIVITY
        Performance varies dramatically with LR, noise scale, network size.
        Hard to tune reliably.

    CORRELATED TARGETS
        Both actor and critic targets change with every gradient step,
        creating a moving target problem even with soft updates.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — TD3: ADDRESSING DDPG'S FAILURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### TD3: Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al., 2018)

TD3 fixes DDPG's three main problems with three targeted modifications:

    FIX 1 — TWIN CRITICS (reduces overestimation bias)
        Train TWO independent Q networks: Q_w1, Q_w2.
        Use the MINIMUM for the target:

            y = r + γ min(Q_w1⁻(s', a'), Q_w2⁻(s', a'))

        The minimum is a conservative estimate. It underestimates slightly
        (pessimistic bias) rather than overestimates (optimistic bias).
        Conservative estimates lead to more stable policy updates.

    FIX 2 — DELAYED POLICY UPDATES (reduces variance from correlated updates)
        Update the actor less frequently than the critic.
        Typical: update actor every 2 critic updates.

        The critic needs many updates to be accurate before the actor
        should trust its gradient. This prevents the actor from chasing
        noisy early critic estimates.

    FIX 3 — TARGET POLICY SMOOTHING (reduces sensitivity to sharp Q peaks)
        Add small noise to the target action:

            a' = clip(μ_θ⁻(s') + clip(ε, −c, c),  a_low, a_high)
            where ε ~ N(0, σ)

        This smooths out sharp peaks in the Q function where the policy
        might exploit approximation errors. Forces Q to be smooth over
        a neighbourhood of actions.

TD3 is typically 2-3x more sample efficient and stable than DDPG.
It matches or exceeds DDPG on almost all continuous control benchmarks.


### TD3 Algorithm

    For each step:
        a = μ_θ(s) + clip(N(0,σ_exploration), -c, c)  [exploration noise]
        s', r ← env.step(a)
        D ← D ∪ {(s, a, r, s', done)}

        If buffer ready:
            Sample batch from D
            a' = clip(μ_θ⁻(s') + clip(N(0,σ_policy), -c, c), a_low, a_high)
            y  = r + γ(1-done) · min(Q_w1⁻(s',a'), Q_w2⁻(s',a'))

            Update both critics:
                L_Q = (Q_w1(s,a) - y)² + (Q_w2(s,a) - y)²

            If step % policy_delay == 0:
                Update actor (using only Q_w1):
                    J = -Q_w1(s, μ_θ(s))
                Soft update targets


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — SAC: THE ENTROPY-REGULARISED ACTOR-CRITIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Maximum Entropy RL Objective

SAC (Soft Actor-Critic, Haarnoja et al., 2018) uses the MaxEnt RL objective:

        J(π) = Σₜ E_{(sₜ,aₜ)~ρ_π} [ r(sₜ, aₜ) + α H(π(·|sₜ)) ]

    α (temperature):  controls the entropy-reward tradeoff.
    H(π):             policy entropy — encourages exploration.

This augmented objective has a critical advantage: the optimal policy
is stochastic even for deterministic tasks. Multiple near-optimal
behaviours are all maintained, making the policy robust to perturbations.


### Soft Bellman Equations

The soft value functions under the MaxEnt objective:

        V^π_soft(s) = E_π [ Q^π_soft(s,a) − α log π(a|s) ]
        Q^π_soft(s,a) = r(s,a) + γ E_{s'} [ V^π_soft(s') ]

Substituting:
        Q^π_soft(s,a) = r(s,a) + γ E_{s'} [ E_π [ Q_soft(s',a') − α log π(a'|s') ] ]


### SAC Architecture and Updates

SAC uses:
    ACTOR   π_θ(a|s): stochastic Gaussian policy (reparameterised)
    CRITICS Q_w1(s,a), Q_w2(s,a): twin critics (like TD3)

Three updates per step:

    1. CRITIC UPDATE (both critics):
       y = r + γ(1-done) · [min(Q_w1⁻(s',a'̃), Q_w2⁻(s',a'̃)) − α log π_θ(a'̃|s')]
       where a'̃ ~ π_θ(·|s')   (sampled from CURRENT policy — no target actor!)

       L_Q = (Q_w1(s,a) − y)² + (Q_w2(s,a) − y)²

    2. ACTOR UPDATE:
       J_π = E_s,ε [ α log π_θ(ã|s) − min(Q_w1(s,ã), Q_w2(s,ã)) ]
       where ã = μ_θ(s) + σ_θ(s)·ε,  ε ~ N(0,I)  [reparameterisation]

       Minimise J_π (equivalently: maximise entropy + Q value)

    3. TEMPERATURE UPDATE (automatic entropy tuning):
       Target entropy: H_target = −dim(A)   (heuristic)
       L_α = −α · (log π_θ(a|s) + H_target).mean()
       α ← α − α_lr ∇_α L_α   [α is log_α in practice for positivity]


### Why SAC Outperforms TD3 and DDPG

    OFF-POLICY + STOCHASTIC POLICY:
        SAC is off-policy (uses replay buffer) but learns a stochastic policy.
        This gives the sample efficiency of off-policy methods with the
        robustness of stochastic exploration.

    AUTOMATIC ENTROPY TUNING:
        No need to manually tune exploration noise. α adapts to maintain
        a target entropy, providing consistent exploration throughout training.

    NO TARGET ACTOR:
        SAC uses the CURRENT actor to generate a' for the critic target,
        not a lagged target actor. This is key to its off-policy nature.

    TWIN CRITICS:
        Same as TD3 — prevents overestimation bias.

SAC consistently achieves state-of-the-art on MuJoCo continuous control
benchmarks with significantly fewer environment interactions than PPO/A3C.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — ACTOR-CRITIC DESIGN PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Critic Target Variants

Different algorithms use different target formulations for the critic:

    ONE-STEP TD (one-step A-C):
        y = r + γ V_w(s')         or    y = r + γ Q_w(s', a')

    N-STEP RETURN (A3C):
        y = Σₖ γᵏ rₜ₊ₖ + γⁿ V_w(sₜ₊ₙ)

    GAE RETURN (PPO):
        y = Âₜ^GAE + V_w(sₜ)   [TD-λ return]

    SOFT TD (SAC):
        y = r + γ (V_soft(s')) = r + γ (Q_soft(s',a') − α log π(a'|s'))

    DOUBLE Q (TD3, SAC):
        y = r + γ min(Q_w1(s', a'), Q_w2(s', a'))


### On-Policy vs Off-Policy Actor-Critic

    ON-POLICY (A2C, A3C, PPO):
        Data collected under CURRENT policy only.
        No replay buffer. Data discarded after each update.
        Lower sample efficiency, higher stability.
        The policy gradient theorem applies directly.

    OFF-POLICY (DDPG, TD3, SAC):
        Data collected under EXPLORATION policy.
        Large replay buffer. Old data reused many times.
        Higher sample efficiency, requires importance corrections.
        The policy gradient still applies for the actor update
        (the gradient is E_s~ρ_β [∇_θ J], averaged over the state
        distribution of the BEHAVIOUR policy β).


### Separate vs Shared Network Architecture

    SHARED TRUNK (A2C, PPO, A3C):
        Input → shared layers → [policy head, value head]
        Shared representations, fewer parameters.
        Suitable when states are high-dimensional (pixels) and
        feature extraction should be shared.

    SEPARATE NETWORKS (DDPG, TD3, SAC):
        Actor:  s → μ_θ(s)
        Critic: (s, a) → Q_w(s, a)   [action is an input]
        Different architectures appropriate for each.
        No gradient interference between actor and critic.

For off-policy methods the critic must take ACTION as input (Q-function),
not just state (V-function), because it needs to evaluate arbitrary actions
from the replay buffer, not just the current policy's action.


### The Role of Entropy in Actor-Critic

Entropy regularisation appears in most practical actor-critic methods:

    EXPLICIT (A3C, A2C):
        L_entropy = c_e · H(π)   as an auxiliary loss term.
        Prevents premature policy collapse.

    IMPLICIT (SAC):
        Entropy is part of the objective — the optimal policy is
        explicitly entropy-maximising. Temperature α trades off
        entropy vs reward automatically.

    NONE (DDPG, TD3):
        Exploration via external noise (OU or Gaussian). No explicit
        entropy objective. More fragile exploration.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — CONVERGENCE AND STABILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Two-Timescale Convergence

The tabular actor-critic converges to a local optimum under conditions:

    C1. Critic faster than actor: α_w / α_θ → ∞  (two-timescale)
    C2. Policy gradient conditions: Σ α_θ = ∞, Σ α_θ² < ∞
    C3. Feature matrix for V_w is full rank (linear critic)
    C4. Bounded rewards

In practice: α_w = 0.1, α_θ = 0.01 works well for tabular and linear.
For neural networks, the convergence guarantees do not hold, but the
heuristic of a faster critic remains practically important.


### The Deadly Triad in Actor-Critic

Function approximation + bootstrapping + off-policy = potential divergence.

On-policy actor-critic (A2C, PPO):
    Uses on-policy data → no off-policy instability
    Still combines FA + bootstrapping → some instability possible
    The entropy bonus and clipping provide practical stability

Off-policy actor-critic (SAC, DDPG, TD3):
    Full deadly triad. Managed by:
        * Target networks (slow the moving target)
        * Replay buffer (reduce correlation)
        * Twin critics (reduce overestimation)
        * Soft updates (smooth target changes)

Despite the lack of convergence guarantees, off-policy actor-critic
methods are empirically the most sample-efficient RL algorithms known.


### Actor-Critic Gradient Interference

In shared-network actor-critic, the policy gradient and value gradient
can interfere:

    dθ_shared ← α_π · ∇L_policy + α_v · ∇L_value

The value loss is typically much larger in magnitude early in training,
which can suppress useful policy gradient signal.

Mitigations:
    * Coefficient c_v = 0.5 reduces value gradient magnitude
    * Gradient norm clipping (max 0.5)
    * Separate learning rates for the two heads
    * Separate optimisers (less common)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — SUMMARY AND ALGORITHM COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Core Equations

    ONE-STEP TD ADVANTAGE:
        δₜ = rₜ₊₁ + γ V_w(sₜ₊₁)(1-done) − V_w(sₜ)

    ACTOR UPDATE (stochastic policy):
        θ ← θ + α_θ · δₜ · ∇_θ log π_θ(aₜ|sₜ)

    ACTOR UPDATE (deterministic policy, DDPG):
        θ ← θ + α_θ · ∇_θ μ_θ(s) · ∇_a Q_w(s,a)|_{a=μ_θ(s)}

    CRITIC UPDATE:
        w ← w − α_w · ∇_w (y − Q_w(s,a))²

    TD3 TARGET:
        y = r + γ min(Q_w1⁻(s',a'̃), Q_w2⁻(s',a'̃))  where a'̃ = μ⁻(s') + noise

    SAC CRITIC TARGET:
        y = r + γ [min(Q_w1(s',ã), Q_w2(s',ã)) − α log π_θ(ã|s')]

    SAC ACTOR UPDATE:
        θ ← argmin_θ E [α log π_θ(ã|s) − min(Q_w1(s,ã), Q_w2(s,ã))]

    SAC TEMPERATURE UPDATE:
        ∇_α L_α = −E [log π_θ(a|s) + H_target]


### Algorithm Comparison

    +----------+-----------+---------+----------+---------+---------+---------+
    | Alg.     | On/Off    | Actions | Critic   | Explore | Replay  | Year    |
    +----------+-----------+---------+----------+---------+---------+---------+
    | A2C      | On-policy | Disc.   | V(s)     | Entropy | No      | 2016    |
    | A3C      | On-policy | Both    | V(s)     | Entropy | No      | 2016    |
    | PPO      | On-policy | Both    | V(s)     | Clip+Ent| No      | 2017    |
    | DDPG     | Off       | Cont.   | Q(s,a)   | OU noise| Yes     | 2015    |
    | TD3      | Off       | Cont.   | 2×Q(s,a) | Gauss.  | Yes     | 2018    |
    | SAC      | Off       | Both    | 2×Q(s,a) | Entropy | Yes     | 2018    |
    | SAC-Auto | Off       | Both    | 2×Q(s,a) | Auto α  | Yes     | 2019    |
    +----------+-----------+---------+----------+---------+---------+---------+

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "1. One-Step Actor-Critic — Complete Implementation": {
        "description": "Implement the full one-step actor-critic with separate actor "
                       "and critic. Verify that the critic converges to V^pi and the "
                       "actor improves the policy over time. Track both learning curves.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]; ANAMES=['U','D','L','R']

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

class TabularActor:
    def __init__(self, rng):
        self.theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    def probs(self, s):
        e=np.exp(self.theta[s]-self.theta[s].max()); return e/e.sum()
    def sample(self, s, rng): return rng.choice(N_A,p=self.probs(s))
    def log_prob(self, s, a): return float(np.log(self.probs(s)[a]+1e-8))
    def update(self, s, a, delta, alpha):
        p=self.probs(s)
        g=np.zeros(N_A,np.float32); g[a]+=1.0; g-=p
        self.theta[s]+=alpha*delta*g

class TabularCritic:
    def __init__(self): self.V=np.zeros(N_S,np.float32)
    def value(self, s): return float(self.V[s])
    def update(self, s, target, alpha):
        self.V[s]+=alpha*(target-self.V[s])

def train_one_step_ac(seed=42, n_steps=100_000, alpha_a=0.01, alpha_c=0.1):
    rng=np.random.default_rng(seed)
    actor=TabularActor(rng); critic=TabularCritic()
    s=0; step=0; ep_ret=0.0; ep_gamma=1.0
    ep_returns=[]; td_errors=[]; entropy_hist=[]

    while step < n_steps:
        a=actor.sample(s,rng)
        s2,r,done=env_step(s,a)
        ep_ret+=ep_gamma*r; ep_gamma*=GAMMA; step+=1

        # One-step TD error (advantage estimate)
        v_next = critic.value(s2)*(1-done)
        target  = r + GAMMA*v_next
        delta   = target - critic.value(s)

        # Critic update: move V(s) toward target
        critic.update(s, target, alpha_c)
        # Actor update: reinforce action proportional to advantage
        actor.update(s, a, delta, alpha_a)

        td_errors.append(abs(delta))
        if step % 100 == 0:
            ent = -sum(actor.probs(st)@np.log(actor.probs(st)+1e-8) for st in range(N_S)) / N_S
            entropy_hist.append(ent)

        if done:
            ep_returns.append(ep_ret)
            s=0; ep_ret=0.0; ep_gamma=1.0
        else:
            s=s2

    return ep_returns, td_errors, entropy_hist, actor, critic

def smooth(x,w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

rets, tds, ents, actor, critic = train_one_step_ac()
sm_rets = smooth(rets)
sm_tds  = smooth(tds,w=1000)

print("One-Step Actor-Critic")
print("=" * 56)
print(f"  alpha_actor=0.01  alpha_critic=0.1  gamma={GAMMA}")
print()
print(f"  {'Episode':>8} | {'Return':>10} | {'Smooth-100':>12}")
print("  "+"-"*36)
for i in [0,49,99,199,399,min(599,len(sm_rets)-1),len(sm_rets)-1]:
    if i < len(sm_rets):
        print(f"  {i+1:>8} | {rets[i]:>10.4f} | {sm_rets[i]:>12.4f}")

print()
print("  Learned V(s) — critic output (grid layout):")
for row in critic.V.reshape(ROWS,COLS):
    print("    "+"  ".join(f"{v:6.3f}" for v in row))

print()
print("  Learned policy — actor greedy actions (grid layout):")
pi=[ANAMES[actor.probs(s).argmax()] for s in range(N_S)]
for row in np.array(pi).reshape(ROWS,COLS):
    print("    "+"  ".join(row))

print()
print(f"  Mean |TD error| (first 10K steps): {np.mean(tds[:10000]):.5f}")
print(f"  Mean |TD error| (last  10K steps): {np.mean(tds[-10000:]):.5f}")
print(f"  Policy entropy (start):           {ents[0]:.4f} nats")
print(f"  Policy entropy (end):             {ents[-1]:.4f} nats")
print("  (entropy decreases as policy becomes more deterministic)")
""",
    },

    "2. Two-Timescale Learning Rates — Convergence Analysis": {
        "description": "Demonstrate the two-timescale principle: the critic must learn "
                       "faster than the actor. Compare convergence under four learning "
                       "rate combinations and show why alpha_c >> alpha_a is critical.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

# Get true V* for comparison
def true_v_star():
    P=np.zeros((N_S,N_A,N_S)); R=np.zeros((N_S,N_A,N_S))
    for s in range(N_S):
        if s in (GOAL,TRAP): P[s,:,s]=1.0; continue
        r,c=s//COLS,s%COLS
        for a,(dr,dc) in enumerate(MOVES):
            nr,nc=r+dr,c+dc
            s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
            P[s,a,s2]=1.0
            R[s,a,s2]=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    R_sa=np.sum(P*R,axis=2); V=np.zeros(N_S)
    for _ in range(100_000):
        Vn=np.max(R_sa+GAMMA*np.einsum('san,n->sa',P,V),axis=1)
        if np.max(np.abs(Vn-V))<1e-12: break
        V=Vn
    return V
V_star = true_v_star()

def train_ac(alpha_a, alpha_c, seed=0, n_steps=80_000):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    V=np.zeros(N_S,np.float32)
    s=0; ep_ret=0.0; ep_gamma=1.0; ep_rets=[]; v_errors=[]

    for step in range(n_steps):
        logits=theta[s]-theta[s].max()
        probs=np.exp(logits); probs/=probs.sum()
        a=rng.choice(N_A,p=probs)
        s2,r,done=env_step(s,a)
        ep_ret+=ep_gamma*r; ep_gamma*=GAMMA

        v_next=V[s2]*(1-done)
        target=r+GAMMA*v_next; delta=target-V[s]
        V[s]+=alpha_c*delta

        g=np.zeros(N_A,np.float32); g[a]+=1.0; g-=probs
        theta[s]+=alpha_a*delta*g

        if done: ep_rets.append(ep_ret); ep_ret=0.0; ep_gamma=1.0; s=0
        else: s=s2

        if step % 1000 == 0:
            v_errors.append(np.mean(np.abs(V - V_star)))

    return ep_rets, v_errors

def smooth(x,w=50):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

configs=[
    (0.001, 0.001, "alpha_a=alpha_c=0.001 (equal, slow)"),
    (0.01,  0.01,  "alpha_a=alpha_c=0.01  (equal, fast)"),
    (0.05,  0.01,  "alpha_c < alpha_a     (WRONG order)"),
    (0.01,  0.1,   "alpha_c > alpha_a     (correct — 2-timescale)"),
]

print("Two-Timescale Learning Rates Analysis")
print("="*62)
print(f"  Reference: true V* computed by value iteration")
print()

results={}
for aa,ac,label in configs:
    rets,verrs=train_ac(aa,ac)
    results[label]=(smooth(rets),verrs,np.mean(rets[-100:]))

for aa,ac,label in configs:
    sm,verrs,final=results[label]
    print(f"  {label}")
    print(f"    Episodes: {len(sm)}  |  Final return: {final:.4f}")
    print(f"    ||V - V*|| (start→end): {verrs[0]:.4f} → {verrs[-1]:.4f}")
    print()

print("  Convergence at key episodes:")
print(f"  {'Episode':>8} | "+" | ".join(f"{label[:20]:>22}" for _,_,label in configs))
print("  "+"-"*(10+len(configs)*25))
max_len=min(len(results[configs[0][2]][0]) for _,_,_ in configs)
for ep_i in [9,49,99,199,499]:
    if ep_i < max_len:
        vals=" | ".join(f"{results[label][0][ep_i]:>22.4f}" for _,_,label in configs)
        print(f"  {ep_i+1:>8} | {vals}")

print()
print("  Key insight:")
print("    alpha_c >> alpha_a: critic converges fast → actor sees accurate signals")
print("    alpha_c << alpha_a: actor changes too fast for critic → noise amplification")
""",
    },

    "3. N-Step Returns — Bias-Variance in Actor-Critic": {
        "description": "Compare one-step TD, 5-step, 10-step, and Monte Carlo advantage "
                       "estimates in the actor-critic framework. Measure convergence speed, "
                       "final performance, and gradient variance for each.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

def n_step_return(rewards, values, dones, n, gamma=GAMMA):
    T=len(rewards)
    returns=np.zeros(T,np.float32)
    for t in range(T):
        G=0.0
        for k in range(n):
            if t+k >= T: break
            G+=gamma**k * rewards[t+k]
            if dones[t+k]: break
        else:
            if t+n < T:
                G += gamma**n * values[t+n] * (1-dones[min(t+n-1,T-1)])
        returns[t]=G
    return returns

def train_nstep_ac(n, seed=42, n_episodes=2000, rollout=50):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    V=np.zeros(N_S,np.float32)
    ALPHA_A=0.03; ALPHA_C=0.1
    ep_returns=[]; grad_norms=[]

    for ep in range(n_episodes):
        # Collect episode in segments of rollout length
        s=0; ep_ret=0.0; ep_gamma=1.0
        traj_s=[]; traj_a=[]; traj_r=[]; traj_d=[]

        for _ in range(200):
            logits=theta[s]-theta[s].max()
            probs=np.exp(logits); probs/=probs.sum()
            a=rng.choice(N_A,p=probs); s2,r,done=env_step(s,a)
            traj_s.append(s); traj_a.append(a); traj_r.append(r); traj_d.append(float(done))
            ep_ret+=ep_gamma*r; ep_gamma*=GAMMA; s=s2
            if done: break

        T=len(traj_r)
        # Get value estimates
        vals_arr=np.array([V[st] for st in traj_s],np.float32)

        # Use MC or n-step
        if n == 0:  # MC
            G=0.0; returns=np.zeros(T,np.float32)
            for t in reversed(range(T)):
                G=traj_r[t]+GAMMA*G*(1-traj_d[t]); returns[t]=G
        else:
            returns=n_step_return(np.array(traj_r), vals_arr, np.array(traj_d), n)

        advantages=returns-vals_arr
        total_grad=np.zeros_like(theta)

        for t,(st,at) in enumerate(zip(traj_s,traj_a)):
            V[st]+=ALPHA_C*(returns[t]-V[st])
            logits=theta[st]-theta[st].max()
            probs=np.exp(logits); probs/=probs.sum()
            g=np.zeros(N_A,np.float32); g[at]+=1.0; g-=probs
            total_grad[st]+=ALPHA_A*advantages[t]*g/T

        theta+=total_grad
        grad_norms.append(np.linalg.norm(total_grad))
        ep_returns.append(ep_ret)

    return ep_returns, grad_norms

def smooth(x,w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

ns=[1, 5, 10, 0]  # 0 = MC
labels={1:"n=1 (TD)", 5:"n=5", 10:"n=10", 0:"MC (n=T)"}

print("N-Step Returns in Actor-Critic")
print("="*58)
print()

results={}
for n in ns:
    rets,gnorms=train_nstep_ac(n)
    results[n]=(smooth(rets),gnorms)

print(f"  {'Episode':>8} | "+" | ".join(f"{labels[n]:>12}" for n in ns))
print("  "+"-"*(10+len(ns)*15))
for i in [49,99,199,499,999,1499,1999]:
    vals=" | ".join(f"{results[n][0][i]:>12.4f}" for n in ns)
    print(f"  {i+1:>8} | {vals}")

print()
print("  Final performance and gradient statistics (last 200 eps):")
for n in ns:
    sm,gnorms=results[n]
    final=np.mean([sm[i] for i in range(len(sm)-200,len(sm))])
    gvar=np.var(gnorms[-500:])
    print(f"    {labels[n]:<18}: return={final:.4f}  grad_norm_var={gvar:.6f}")

print()
print("  Interpretation:")
print("    n=1:  lowest variance, highest bias, converges first but to lower value")
print("    n=5:  balanced — typical choice")
print("    MC:   unbiased but noisiest gradients, slowest convergence")
""",
    },

    "4. Shared Network Actor-Critic — Gradient Interference": {
        "description": "Build a shared-trunk actor-critic network. Measure and compare "
                       "the gradient magnitudes of the policy and value heads. Show "
                       "how coefficient c_v controls gradient interference.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]; ANAMES=['U','D','L','R']

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

class SharedAC:
    def __init__(self, n_s, n_a, hidden=32, c_v=0.5, c_ent=0.01, rng=None):
        if rng is None: rng=np.random.default_rng(0)
        s1,s2=np.sqrt(2/n_s),np.sqrt(2/hidden)
        self.W1=rng.normal(0,s1,(n_s,hidden)).astype(np.float32)
        self.b1=np.zeros(hidden,np.float32)
        self.W2=rng.normal(0,s2,(hidden,hidden)).astype(np.float32)
        self.b2=np.zeros(hidden,np.float32)
        self.Wp=rng.normal(0,0.01,(hidden,n_a)).astype(np.float32)
        self.bp=np.zeros(n_a,np.float32)
        self.Wv=rng.normal(0,s2,(hidden,1)).astype(np.float32)
        self.bv=np.zeros(1,np.float32)
        self.c_v=c_v; self.c_ent=c_ent

    def forward(self, x):
        self.x=x
        self.h1=np.tanh(x@self.W1+self.b1)
        self.h=np.tanh(self.h1@self.W2+self.b2)
        logits=self.h@self.Wp+self.bp
        logits-=logits.max(-1,keepdims=True)
        self.lp=logits-np.log(np.exp(logits).sum(-1,keepdims=True))
        self.p=np.exp(self.lp)
        self.v=(self.h@self.Wv+self.bv).squeeze(-1)
        return self.lp, self.v

    def compute_gradients(self, s_idx, a_idx, adv, v_target):
        B=self.x.shape[0]
        # Policy gradient
        dL_pi=np.zeros_like(self.lp)
        for i,a in enumerate(a_idx):
            dL_pi[i]=-(adv[i]/B)*(np.eye(N_A)[a]-self.p[i])
        # Entropy gradient (negative entropy = maximize entropy)
        dL_ent=self.c_ent*(self.lp+1)*self.p/B
        # Value gradient
        dL_v=self.c_v*2*(self.v-v_target)/B

        # Backprop
        dlogits = dL_pi + dL_ent
        dWp=self.h.T@dlogits; dbp=dlogits.sum(0)
        dWv=(self.h.T@dL_v[:,None]); dbv=np.array([dL_v.sum()])

        dh = dlogits@self.Wp.T + dL_v[:,None]@self.Wv.T
        dh *= (1-self.h**2)
        dW2=self.h1.T@dh; db2=dh.sum(0)
        dh1=dh@self.W2.T; dh1*=(1-self.h1**2)
        dW1=self.x.T@dh1; db1=dh1.sum(0)

        pi_grad_norm=np.sqrt(sum(np.sum(g**2) for g in [dWp,dbp]))
        v_grad_norm =np.sqrt(sum(np.sum(g**2) for g in [dWv,dbv]))
        trunk_from_pi=np.sqrt(np.sum((dlogits@self.Wp.T)**2))
        trunk_from_v =np.sqrt(np.sum((dL_v[:,None]@self.Wv.T)**2))
        return dW1,db1,dW2,db2,dWp,dbp,dWv,dbv, pi_grad_norm,v_grad_norm,trunk_from_pi,trunk_from_v

    def apply(self, grads, lr=3e-3):
        dW1,db1,dW2,db2,dWp,dbp,dWv,dbv,*_=grads
        for p,g in [(self.W1,dW1),(self.b1,db1),(self.W2,dW2),(self.b2,db2),
                    (self.Wp,dWp),(self.bp,dbp),(self.Wv,dWv),(self.bv,dbv)]:
            norm=np.sqrt(np.sum(g**2))
            if norm>0.5: g=g*0.5/norm
            p-=lr*g

def train(c_v, seed=42, n_ep=1500, rollout=64):
    rng=np.random.default_rng(seed)
    net=SharedAC(N_S,N_A,c_v=c_v,rng=rng)
    buf_s=[]; buf_a=[]; buf_r=[]; buf_d=[]; buf_v=[]
    pi_gnorms=[]; v_gnorms=[]; trunk_pi=[]; trunk_v=[]; ep_rets=[]

    for ep in range(n_ep):
        s=0; traj_s=[]; traj_a=[]; traj_r=[]; traj_d=[]
        for _ in range(200):
            x=np.eye(N_S,dtype=np.float32)[[s]]
            lp,v=net.forward(x)
            probs=np.exp(lp[0]); a=rng.choice(N_A,p=probs)
            s2,r,done=env_step(s,a)
            traj_s.append(s); traj_a.append(a); traj_r.append(r); traj_d.append(float(done))
            s=s2
            if done: break

        T=len(traj_r); G=0.0; Gs=np.zeros(T)
        for t in reversed(range(T)):
            G=traj_r[t]+GAMMA*G*(1-traj_d[t]); Gs[t]=G

        ss=np.eye(N_S,dtype=np.float32)[np.array(traj_s)]
        lp,v=net.forward(ss)
        adv=(Gs-v); adv=(adv-adv.mean())/(adv.std()+1e-8)
        grads=net.compute_gradients(traj_s,traj_a,adv,Gs.astype(np.float32))
        pi_gnorms.append(grads[8]); v_gnorms.append(grads[9])
        trunk_pi.append(grads[10]); trunk_v.append(grads[11])
        net.apply(grads)
        ep_rets.append(sum(GAMMA**t*r for t,r in enumerate(traj_r)))

    return ep_rets, pi_gnorms, v_gnorms, trunk_pi, trunk_v

def smooth(x,w=50):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

c_vs=[0.1, 0.5, 1.0, 2.0]
print("Shared Network AC — Value Gradient Coefficient Analysis")
print("="*58)
print(f"  Measuring gradient norms at policy head and value head")
print()

for c_v in c_vs:
    rets,pgn,vgn,tpi,tv=train(c_v)
    print(f"  c_v = {c_v:.1f}:")
    print(f"    Final return (avg last 100 eps): {np.mean(rets[-100:]):.4f}")
    print(f"    Policy head grad norm:  {np.mean(pgn[-200:]):.5f}")
    print(f"    Value  head grad norm:  {np.mean(vgn[-200:]):.5f}")
    print(f"    Trunk grad from policy: {np.mean(tpi[-200:]):.5f}")
    print(f"    Trunk grad from value:  {np.mean(tv[-200:]):.5f}")
    ratio=np.mean(tv[-200:])/(np.mean(tpi[-200:])+1e-8)
    print(f"    Value/Policy trunk ratio: {ratio:.2f}x  "
          f"{'(value dominates!)' if ratio > 3 else '(balanced)' if ratio < 2 else '(slightly high)'}")
    print()

print("  Recommendation: c_v = 0.5 is standard — balances learning signals.")
print("  c_v too high: value gradients overwhelm policy in shared trunk.")
""",
    },

    "5. Deterministic Policy Gradient (DPG) — DDPG Core": {
        "description": "Implement the deterministic policy gradient theorem. Show how the "
                       "actor gradient flows through the Q-function. Compare DPG to "
                       "stochastic PG on a simple continuous-action task.",
        "code": """\
import numpy as np

# ── Simple Continuous Control Task ───────────────────────────
# State: x position in [-1,1]. Action: force f in [-1,1].
# Reward: -(x^2 + 0.1*f^2). Optimal: f=-k*x (proportional control).
# Dynamics: x' = x + 0.1*f + noise

class Env1D:
    def __init__(self, rng): self.rng=rng; self.x=0.0
    def reset(self): self.x=self.rng.uniform(-0.5,0.5); return np.array([self.x],np.float32)
    def step(self, f):
        f=float(np.clip(f,-1,1))
        self.x=np.clip(self.x+0.1*f+self.rng.normal(0,0.02),-1.5,1.5)
        r=-(self.x**2+0.1*f**2)
        done=abs(self.x)>1.4
        return np.array([self.x],np.float32), r, done


class LinearActor:
    def __init__(self, rng, stochastic=False):
        self.W=rng.normal(0,0.1,(1,1)).astype(np.float32)
        self.stochastic=stochastic
        self.log_std=np.array([-1.0],np.float32)  # only for stochastic
    def mu(self, s): return float(np.clip(float(s[0]*self.W[0,0]),-1,1))
    def sample(self, s, rng):
        if self.stochastic:
            std=np.exp(self.log_std[0]); a=self.mu(s)+rng.normal(0,std)
        else:
            a=self.mu(s)+rng.normal(0,0.1)   # exploration noise for DPG
        return float(np.clip(a,-1,1))

class LinearQ:
    def __init__(self, rng):
        # Q(s,a) = [s, a, s*a, 1] @ w  (bilinear approximation)
        self.w=rng.normal(0,0.01,(4,)).astype(np.float32)
    def features(self, s, a):
        return np.array([float(s[0]), a, float(s[0])*a, 1.0], np.float32)
    def value(self, s, a): return float(self.features(s,a)@self.w)
    def grad_a(self, s, a):
        # dQ/da = w[1] + w[2]*s
        return float(self.w[1] + self.w[2]*s[0])
    def update(self, s, a, target, alpha=0.1):
        f=self.features(s,a)
        delta=target-self.value(s,a)
        self.w+=alpha*delta*f

def train_dpg(seed=42, n_steps=30_000, gamma=0.99):
    rng=np.random.default_rng(seed)
    env=Env1D(rng); actor=LinearActor(rng,False); critic=LinearQ(rng)
    # Target networks
    actor_t=LinearActor(rng,False); actor_t.W[:]=actor.W
    critic_t=LinearQ(rng); critic_t.w[:]=critic.w
    BUF=[]; LR_A=0.01; LR_C=0.1; TAU=0.01; returns=[]

    s=env.reset(); ep_ret=0.0; ep_gamma=1.0
    for step in range(n_steps):
        a=actor.sample(s,rng)
        s2,r,done=env.step(a)
        BUF.append((s.copy(),a,r,s2.copy(),float(done)))
        if len(BUF)>5000: BUF.pop(0)
        ep_ret+=ep_gamma*r; ep_gamma*=gamma
        if done: returns.append(ep_ret); ep_ret=0.0; ep_gamma=1.0; s=env.reset()
        else: s=s2

        if len(BUF)<128: continue
        idx=rng.integers(0,len(BUF),32)
        batch=[BUF[i] for i in idx]
        for (sb,ab,rb,s2b,db) in batch:
            a_t=actor_t.mu(s2b)
            y=rb+gamma*(1-db)*critic_t.value(s2b,a_t)
            critic.update(sb,ab,y,LR_C)

        # DPG actor update: theta += alpha * dQ/da * da/dtheta
        dpg_grad=0.0
        for (sb,ab,rb,s2b,db) in batch:
            a_curr=actor.mu(sb)  # current det. action
            dQda=critic.grad_a(sb,a_curr)
            # da/dtheta = s (linear actor)
            dpg_grad+=dQda*float(sb[0])
        actor.W+=LR_A*dpg_grad/len(batch)

        # Soft updates
        actor_t.W[:]=TAU*actor.W+(1-TAU)*actor_t.W
        critic_t.w[:]=TAU*critic.w+(1-TAU)*critic_t.w

    return returns, actor, critic

def smooth(x,w=50):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

rets,actor,critic=train_dpg()
sm=smooth(rets)
print("Deterministic Policy Gradient (DPG) — Continuous Control")
print("="*58)
print(f"  Task: 1D balance. Action: force in [-1,1].")
print(f"  Optimal: f = -k*x (proportional feedback control)")
print()
print(f"  {'Episode':>8} | {'Return':>10} | {'Smooth':>10}")
print("  "+"-"*34)
for i in [0,19,49,99,199,min(299,len(sm)-1),len(sm)-1]:
    if i >= 0 and i < len(sm) and i < len(rets):
        print(f"  {i+1:>8} | {rets[i]:>10.4f} | {sm[i]:>10.4f}")

print()
print(f"  Learned actor W (da/ds): {actor.W[0,0]:.5f}")
print(f"  Q-weights: {critic.w.round(4)}")
print()
print("  DPG gradient formula:")
print("    dJ/dtheta = E_s [ dQ(s,a)/da |_{a=mu(s)} * dmu(s)/dtheta ]")
print("    = E_s [ dQ/da * s ]   (for linear actor mu(s) = W*s)")
print()
print("  Deterministic PG advantages vs stochastic PG:")
print("    * No need to sample and estimate log-prob gradient")
print("    * Lower variance (no stochastic action sampling)")
print("    * Off-policy capable (replay buffer + IS correction)")
print("    * Works naturally for continuous action spaces")
""",
    },

    "6. Twin Critics — Overestimation Bias Reduction": {
        "description": "Demonstrate the overestimation problem in single-critic AC "
                       "and show how twin critics (TD3-style) fix it. Measure Q-value "
                       "estimates against ground-truth returns.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

# True Q* via value iteration
def true_q_star():
    P=np.zeros((N_S,N_A,N_S)); R=np.zeros((N_S,N_A,N_S))
    for s in range(N_S):
        if s in (GOAL,TRAP): P[s,:,s]=1.0; continue
        r,c=s//COLS,s%COLS
        for a,(dr,dc) in enumerate(MOVES):
            nr,nc=r+dr,c+dc
            s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
            P[s,a,s2]=1.0
            R[s,a,s2]=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    R_sa=np.sum(P*R,axis=2); V=np.zeros(N_S)
    for _ in range(100_000):
        Q=R_sa+GAMMA*np.einsum('san,n->sa',P,V)
        Vn=Q.max(1)
        if np.max(np.abs(Vn-V))<1e-12: break
        V=Vn
    return Q
Q_STAR=true_q_star()

def train(use_twin, seed=42, n_steps=15_000, lr_q=0.05, lr_pi=0.01):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    Q1=np.zeros((N_S,N_A),np.float32)
    Q2=np.zeros((N_S,N_A),np.float32) if use_twin else None
    BUF=[]; s=0; ep_ret=0.0; ep_gamma=1.0; ep_rets=[]
    bias_hist=[]; step_cnt=0

    while step_cnt < n_steps:
        logits=theta[s]-theta[s].max()
        probs=np.exp(logits); probs/=probs.sum()
        a=rng.choice(N_A,p=probs)
        s2,r,done=env_step(s,a)
        BUF.append((s,a,r,s2,float(done)))
        if len(BUF)>2000: BUF.pop(0)
        ep_ret+=ep_gamma*r; ep_gamma*=GAMMA; step_cnt+=1
        if done: ep_rets.append(ep_ret); ep_ret=0.0; ep_gamma=1.0; s=0
        else: s=s2

        if len(BUF)<64: continue
        idx=rng.integers(0,len(BUF),32)
        batch=[BUF[i] for i in idx]

        sb_arr=np.array([b[0] for b in batch])
        ab_arr=np.array([b[1] for b in batch])
        rb_arr=np.array([b[2] for b in batch],np.float32)
        s2b_arr=np.array([b[3] for b in batch])
        db_arr=np.array([b[4] for b in batch],np.float32)

        # Vectorised critic update
        logits2 = theta[s2b_arr] - theta[s2b_arr].max(1,keepdims=True)
        probs2  = np.exp(logits2); probs2 /= probs2.sum(1,keepdims=True)
        a2_arr  = probs2.argmax(1)
        if use_twin:
            q_next = np.minimum(Q1[s2b_arr, a2_arr], Q2[s2b_arr, a2_arr])
        else:
            q_next = Q1[s2b_arr, a2_arr]
        y_arr = rb_arr + GAMMA*(1-db_arr)*q_next
        for i in range(len(batch)):
            Q1[sb_arr[i], ab_arr[i]] += lr_q*(y_arr[i]-Q1[sb_arr[i], ab_arr[i]])
            if use_twin:
                Q2[sb_arr[i], ab_arr[i]] += lr_q*(y_arr[i]-Q2[sb_arr[i], ab_arr[i]])

        # Vectorised actor update
        adv_arr = Q1[sb_arr, ab_arr] - Q1[sb_arr].mean(1)
        probs_b = np.exp(theta[sb_arr] - theta[sb_arr].max(1,keepdims=True))
        probs_b /= probs_b.sum(1,keepdims=True)
        g_arr = -probs_b.copy()
        g_arr[np.arange(len(batch)), ab_arr] += 1.0
        updates = (lr_pi/len(batch)) * adv_arr[:,None] * g_arr
        for i in range(len(batch)):
            theta[sb_arr[i]] += updates[i]

        if step_cnt % 2000 == 0:
            bias_hist.append(float(np.mean(Q1 - Q_STAR)))

    return ep_rets, bias_hist, Q1

def smooth(x,w=50):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

rets_single, bias_single, Q1_single = train(False, n_steps=15_000)
rets_twin,   bias_twin,   Q1_twin   = train(True,  n_steps=15_000)

print("Twin Critics — Overestimation Bias Comparison")
print("="*56)
print()
sm_s=smooth(rets_single); sm_t=smooth(rets_twin)
print(f"  {'Episode':>8} | {'Single Q':>12} | {'Twin Q (min)':>14}")
print("  "+"-"*40)
N=min(len(sm_s),len(sm_t))
for i in [49,99,199,299,min(399,N-1),N-1]:
    if i < N:
        print(f"  {i+1:>8} | {sm_s[i]:>12.4f} | {sm_t[i]:>14.4f}")

print()
print("  Q-value bias vs Q* (positive = overestimation):")
print(f"  {'Measurement':>12} | {'Single Q bias':>15} | {'Twin Q bias':>13}")
print("  "+"-"*46)
n_pts=min(len(bias_single),len(bias_twin))
for i in [0, n_pts//4, n_pts//2, 3*n_pts//4, n_pts-1]:
    if i < n_pts:
        print(f"  {i+1:>12} | {bias_single[i]:>15.5f} | {bias_twin[i]:>13.5f}")

print()
print(f"  Final bias (avg last 10 measurements):")
print(f"    Single Q: {np.mean(bias_single[-10:]):>+.5f}  ← positive = overestimating")
print(f"    Twin Q:   {np.mean(bias_twin[-10:]):>+.5f}  ← negative = conservative (desired)")
print()
print("  Key: min(Q1,Q2) is pessimistic but stable.")
print("  Pessimistic bias → slower but more reliable policy improvement.")
""",
    },

    "7. SAC Temperature (α) — Automatic Entropy Tuning": {
        "description": "Implement SAC's automatic entropy tuning. Show how the temperature "
                       "parameter α adapts to maintain a target entropy, providing "
                       "consistent exploration throughout training.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

def train_sac_alpha(target_entropy_frac=0.5, seed=42, n_steps=80_000):
    \"\"\"
    target_entropy = target_entropy_frac * log(|A|)
    SAC adjusts alpha to maintain this entropy automatically.
    \"\"\"
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    Q1=np.zeros((N_S,N_A),np.float32)
    Q2=np.zeros((N_S,N_A),np.float32)

    log_alpha=np.array([0.0],np.float32)  # learnable log-temperature
    target_entropy=target_entropy_frac*np.log(N_A)  # target H

    BUF=[]; s=0; ep_ret=0.0; ep_gamma=1.0; ep_rets=[]
    alpha_hist=[]; entropy_hist=[]; step=0
    LR_Q=0.05; LR_PI=0.01; LR_ALPHA=0.01

    while step < n_steps:
        alpha=float(np.exp(log_alpha[0]))
        logits=theta[s]-theta[s].max()
        probs=np.exp(logits); probs/=probs.sum()
        a=rng.choice(N_A,p=probs)
        s2,r,done=env_step(s,a)
        BUF.append((s,a,r,s2,float(done)))
        if len(BUF)>2000: BUF.pop(0)
        ep_ret+=ep_gamma*r; ep_gamma*=GAMMA; step+=1
        if done: ep_rets.append(ep_ret); ep_ret=0.0; ep_gamma=1.0; s=0
        else: s=s2

        if len(BUF)<64: continue
        idx=rng.integers(0,len(BUF),32)
        batch=[BUF[i] for i in idx]
        alpha=float(np.exp(log_alpha[0]))

        # SAC Critic: y = r + gamma * (V_soft(s') - alpha*H)
        # V_soft(s') = E_pi[Q(s',a') - alpha*log_pi(a'|s')]
        for (sb,ab,rb,s2b,db) in batch:
            logits2=theta[s2b]-theta[s2b].max()
            probs2=np.exp(logits2); probs2/=probs2.sum()
            lp2=np.log(probs2+1e-8)
            # Soft value: E[Q - alpha*log_pi]
            v_soft=float(np.sum(probs2*(np.minimum(Q1[s2b],Q2[s2b])-alpha*lp2)))
            y=rb+GAMMA*(1-db)*v_soft
            Q1[sb,ab]+=LR_Q*(y-Q1[sb,ab])
            Q2[sb,ab]+=LR_Q*(y-Q2[sb,ab])

        # Actor: maximise E[Q(s,a) - alpha*log_pi(a|s)]
        for (sb,_,_,_,_) in batch:
            logits_b=theta[sb]-theta[sb].max()
            probs_b=np.exp(logits_b); probs_b/=probs_b.sum()
            lp_b=np.log(probs_b+1e-8)
            # Soft Q advantage
            soft_q=np.minimum(Q1[sb],Q2[sb])
            adv=soft_q-alpha*lp_b
            adv=adv-adv.mean()
            for ai in range(N_A):
                # d/dtheta_sb: sum_a pi(a|s)[adv_a*(delta_ai - pi_a)]
                g=np.zeros(N_A,np.float32)
                g[ai]=1.0; g-=probs_b
                theta[sb]+=LR_PI*adv[ai]*probs_b[ai]*g/len(batch)

        # Temperature update: maximize E[-alpha*(log_pi + H_target)]
        log_pi_avg=np.mean([
            float(np.sum(np.exp(theta[sb]-theta[sb].max())*
                         (theta[sb]-theta[sb].max()-
                          np.log(np.exp(theta[sb]-theta[sb].max()).sum(keepdims=True)))))
            for (sb,_,_,_,_) in batch
        ])  # this approximates E[log pi]
        alpha_grad = -(log_pi_avg + target_entropy)
        log_alpha[0] += LR_ALPHA * float(alpha_grad)

        if step % 2000 == 0:
            alpha_hist.append(float(np.exp(log_alpha[0])))
            # Current entropy
            avg_h=np.mean([-np.sum(p*np.log(p+1e-8))
                           for s in range(N_S)
                           for p in [np.exp(theta[s]-theta[s].max())/
                                     np.exp(theta[s]-theta[s].max()).sum()]])
            entropy_hist.append(avg_h)

    return ep_rets, alpha_hist, entropy_hist

def smooth(x,w=50):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

print("SAC Automatic Temperature Tuning")
print("="*56)
H_max=np.log(N_A)
target_fracs=[0.3, 0.5, 0.8]

for frac in target_fracs:
    target_H=frac*H_max
    rets,alphas,entropies=train_sac_alpha(frac, n_steps=25_000)
    sm=smooth(rets)
    print(f"  Target entropy: {target_H:.3f} nats ({frac:.0%} of max={H_max:.3f})")
    print(f"    Final alpha:  {alphas[-1]:.5f}  (higher = more exploration)")
    print(f"    Final entropy:{entropies[-1]:.4f} nats  "
          f"({'OK' if abs(entropies[-1]-target_H)<0.2 else 'MISSED'})")
    print(f"    Final return: {np.mean(rets[-100:]):.4f}")
    print()

print("  Alpha dynamics (target_frac=0.5):")
rets2,alphas2,entropies2=train_sac_alpha(0.5, n_steps=25_000)
print(f"  {'Checkpoint':>12} | {'alpha':>10} | {'entropy':>10} | {'target H':>10}")
print("  "+"-"*48)
H_tgt=0.5*H_max
for i in range(len(alphas2)):
    print(f"  {(i+1)*2000:>12} | {alphas2[i]:>10.5f} | {entropies2[i]:>10.4f} | {H_tgt:>10.4f}")

print()
print("  Mechanism:")
print("    If entropy > target: alpha decreases (less pressure to explore)")
print("    If entropy < target: alpha increases (more pressure to explore)")
print("    Alpha stabilises when actual entropy matches target entropy.")
""",
    },

    "8. Full Algorithm Comparison — A2C vs DDPG vs SAC": {
        "description": "Train A2C (on-policy, shared net), tabular DDPG-style "
                       "(off-policy, deterministic), and tabular SAC-style (off-policy, "
                       "entropic) on the same task. Compare sample efficiency and final "
                       "performance across the three paradigms.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]
TOTAL_STEPS=25_000

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

def softmax_lp(theta, s):
    logits=theta[s]-theta[s].max()
    return logits-np.log(np.exp(logits).sum())

def softmax_probs(theta, s):
    return np.exp(softmax_lp(theta, s))

# ── A2C: on-policy, rollout of T=16 steps ─────────────────────
def train_a2c(seed=0):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    V=np.zeros(N_S,np.float32)
    LR_PI=0.05; LR_V=0.1; ROLLOUT=16; GAM=GAMMA; LAM=0.95
    s=0; step=0; ep_rets=[]; ep_ret=0.0; ep_gamma=1.0

    while step < TOTAL_STEPS:
        traj_s=[]; traj_a=[]; traj_r=[]; traj_d=[]
        for _ in range(ROLLOUT):
            probs=softmax_probs(theta,s)
            a=rng.choice(N_A,p=probs); s2,r,done=env_step(s,a)
            traj_s.append(s); traj_a.append(a); traj_r.append(r); traj_d.append(float(done))
            ep_ret+=ep_gamma*r; ep_gamma*=GAM; step+=1
            if done: ep_rets.append(ep_ret); ep_ret=0.0; ep_gamma=1.0; s=0
            else: s=s2

        T=len(traj_r); v_boot=V[s]*(1-traj_d[-1])
        vs=np.array([V[st] for st in traj_s]+[v_boot],np.float32)
        A=np.zeros(T,np.float32); la=0.0
        for t in reversed(range(T)):
            delta=traj_r[t]+GAM*vs[t+1]*(1-traj_d[t])-vs[t]
            A[t]=la=delta+GAM*LAM*(1-traj_d[t])*la
        rets=A+vs[:T]
        for t,(st,at) in enumerate(zip(traj_s,traj_a)):
            V[st]+=LR_V*(rets[t]-V[st])
            probs=softmax_probs(theta,st)
            g=np.zeros(N_A,np.float32); g[at]+=1.0; g-=probs
            theta[st]+=LR_PI*A[t]*g/T
    return ep_rets

# ── DDPG-style: off-policy, deterministic (greedy) actor ──────
def train_ddpg_style(seed=0):
    rng=np.random.default_rng(seed)
    Q=np.zeros((N_S,N_A),np.float32)
    Q_t=np.zeros((N_S,N_A),np.float32)
    # Deterministic actor = argmax Q
    BUF=[]; s=0; ep_ret=0.0; ep_gamma=1.0; ep_rets=[]; step=0
    LR_Q=0.05; TAU=0.01; EPS=0.3

    while step < TOTAL_STEPS:
        # Noisy greedy action (deterministic + epsilon exploration)
        a=rng.integers(N_A) if rng.random()<EPS else int(Q[s].argmax())
        s2,r,done=env_step(s,a)
        BUF.append((s,a,r,s2,float(done)))
        if len(BUF)>5000: BUF.pop(0)
        ep_ret+=ep_gamma*r; ep_gamma*=GAMMA; step+=1
        if done: ep_rets.append(ep_ret); ep_ret=0.0; ep_gamma=1.0; s=0
        else: s=s2

        if len(BUF)<64: continue
        idx=rng.integers(0,len(BUF),32)
        for (sb,ab,rb,s2b,db) in [BUF[i] for i in idx]:
            y=rb+GAMMA*(1-db)*float(Q_t[s2b].max())
            Q[sb,ab]+=LR_Q*(y-Q[sb,ab])
        # Soft update
        Q_t[:]=TAU*Q+(1-TAU)*Q_t
        EPS=max(0.05,EPS*0.9998)
    return ep_rets

# ── SAC-style: off-policy, stochastic + entropy ───────────────
def train_sac_style(seed=0):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    Q1=np.zeros((N_S,N_A),np.float32)
    Q2=np.zeros((N_S,N_A),np.float32)
    ALPHA=0.1  # fixed temperature
    BUF=[]; s=0; ep_ret=0.0; ep_gamma=1.0; ep_rets=[]; step=0
    LR_Q=0.05; LR_PI=0.01

    while step < TOTAL_STEPS:
        probs=softmax_probs(theta,s)
        a=rng.choice(N_A,p=probs); s2,r,done=env_step(s,a)
        BUF.append((s,a,r,s2,float(done)))
        if len(BUF)>5000: BUF.pop(0)
        ep_ret+=ep_gamma*r; ep_gamma*=GAMMA; step+=1
        if done: ep_rets.append(ep_ret); ep_ret=0.0; ep_gamma=1.0; s=0
        else: s=s2

        if len(BUF)<64: continue
        idx=rng.integers(0,len(BUF),32)
        for (sb,ab,rb,s2b,db) in [BUF[i] for i in idx]:
            probs2=softmax_probs(theta,s2b)
            lp2=softmax_lp(theta,s2b)
            v_soft=float(np.sum(probs2*(np.minimum(Q1[s2b],Q2[s2b])-ALPHA*lp2)))
            y=rb+GAMMA*(1-db)*v_soft
            Q1[sb,ab]+=LR_Q*(y-Q1[sb,ab])
            Q2[sb,ab]+=LR_Q*(y-Q2[sb,ab])
            probs_b=softmax_probs(theta,sb)
            adv=np.minimum(Q1[sb],Q2[sb])-ALPHA*softmax_lp(theta,sb)
            adv-=adv.mean()
            for ai in range(N_A):
                g=np.zeros(N_A,np.float32); g[ai]+=1.0; g-=probs_b
                theta[sb]+=LR_PI*adv[ai]*probs_b[ai]*g/32
    return ep_rets

def smooth(x,w=80):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

print("Full Algorithm Comparison — A2C vs DDPG-style vs SAC-style")
print(f"  Total steps: {TOTAL_STEPS:,} each")
print("="*64)

r_a2c  = train_a2c()
r_ddpg = train_ddpg_style()
r_sac  = train_sac_style()
sm_a   = smooth(r_a2c); sm_d=smooth(r_ddpg); sm_s=smooth(r_sac)

N=min(len(sm_a),len(sm_d),len(sm_s))
print()
print(f"  {'Episode':>8} | {'A2C (on-pol)':>14} | {'DDPG-style':>12} | {'SAC-style':>11}")
print("  "+"-"*54)
for frac in [0.02,0.05,0.1,0.2,0.4,0.7,1.0]:
    i=min(int(frac*N),N-1)
    print(f"  {i+1:>8} | {sm_a[i]:>14.4f} | {sm_d[i]:>12.4f} | {sm_s[i]:>11.4f}")

print()
print("  Final performance (last 20% of episodes):")
n20=max(1,N//5)
print(f"    A2C  (on-policy):  {np.mean(r_a2c[-n20:]):.4f}  grad var: {np.var(r_a2c[-n20:]):.4f}")
print(f"    DDPG (off-policy): {np.mean(r_ddpg[-n20:]):.4f}  grad var: {np.var(r_ddpg[-n20:]):.4f}")
print(f"    SAC  (off+entrop): {np.mean(r_sac[-n20:]):.4f}  grad var: {np.var(r_sac[-n20:]):.4f}")
print()
print("  Key differences demonstrated:")
print("    A2C:   on-policy, high data turnover, stable but less efficient")
print("    DDPG:  off-policy replay, deterministic, sample-efficient")
print("    SAC:   off-policy + entropy, most robust, consistent exploration")
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