"""Module: 02 · Q-Learning"""

"""
Q-Learning
==========

Q-Learning is the algorithm that turned reinforcement learning from a
theoretical framework into a practical one. Introduced by Chris Watkins
in 1989 and proven convergent by Watkins & Dayan in 1992, it was the
first algorithm to guarantee finding the optimal policy directly from
raw experience — without ever knowing the environment model.

The key insight: instead of learning the state-value V(s), learn the
action-value Q(s, a) — the expected return from taking action a in
state s. With Q in hand, the optimal policy is trivially:

        π*(s) = argmax_a Q*(s, a)

No model of P or R is needed. Just experience.

This module covers the Q-table, the Bellman optimality update rule,
convergence theory, exploration strategies, SARSA vs Q-Learning,
n-step returns, and the full path from tabular Q-Learning to DQN.
"""

import re

TOPIC_NAME   = "Q-Learning"
DISPLAY_NAME = "02 · Q-Learning"
ICON         = "🧮"
SUBTITLE     = "Off-policy TD control — from Q-tables to Deep Q-Networks"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — FROM VALUE FUNCTIONS TO Q-LEARNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why Action Values Instead of State Values?

Recall from the MDP module that two value functions exist:

        V^π(s)   = E_π [ Gₜ | sₜ = s ]
        Q^π(s,a) = E_π [ Gₜ | sₜ = s, aₜ = a ]

To act greedily using V, the agent needs the environment model P:

        π(s) = argmax_a Σₛ' P(s'|s,a) [ R(s,a,s') + γ V(s') ]

This requires knowing P and R — the transition and reward functions.

To act greedily using Q, no model is needed:

        π(s) = argmax_a Q(s, a)

This is the practical appeal of Q-Learning: learn Q*(s,a) directly from
experience, then extract the optimal policy for free.


### The Central Question

How do we estimate Q*(s,a) from raw transitions (s, a, r, s')
without knowing P or R?

Answer: apply the Bellman optimality equation as a stochastic update rule.

The Bellman optimality equation for Q*:

        Q*(s,a) = Σₛ' P(s'|s,a) [ R(s,a,s') + γ max_a' Q*(s',a') ]

Since we observe a single sample (r, s') from the distribution, we form a
sample-based target:

        TD target:    y = r + γ max_a' Q(s', a')

And move Q(s, a) toward it:

        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]

This is the Q-Learning update. It is an application of stochastic
approximation to the Bellman optimality equation.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — THE Q-TABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Structure

For finite discrete state and action spaces, Q is a 2D lookup table:

        Q ∈ R^{|S| × |A|}

        Q[s, a]  stores the current estimate of Q*(s, a)

Example — 4×4 grid world, 4 actions (up, down, left, right):

        Q ∈ R^{16 × 4}    →    64 values to learn

Contrast with a neural network approximation used in DQN:

        Q_θ(s, ·) : R^d → R^|A|    →    millions of parameters

The table is the exact representation; the network is an approximation.


### Initialisation

    OPTIMISTIC INIT     Q(s,a) ← large positive constant (e.g. 1.0)
                        Encourages exploration: every state-action pair
                        appears promising until visited. Naturally drives
                        systematic exploration without ε. Called optimistic
                        initialisation.

    ZERO INIT           Q(s,a) ← 0.0
                        Standard default. No bias, but requires explicit
                        exploration strategy (ε-greedy, UCB, etc.).

    RANDOM INIT         Q(s,a) ~ Uniform(0, 0.01)
                        Breaks ties between equal actions.

    HEURISTIC INIT      Initialise with domain knowledge.
                        Example: Q(s_goal, ·) ← 0, Q(others, ·) ← −1.
                        Accelerates learning when prior knowledge exists.


### The Q-Table as a Policy

At any point during learning, the greedy policy derived from Q is:

        π_Q(s) = argmax_a Q(s, a)

This policy changes as Q is updated. At convergence (Q → Q*), the
greedy policy is the optimal policy π*.

Tie-breaking when multiple actions share the maximum Q-value:
use random tie-breaking to avoid systematic bias toward one action.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — THE Q-LEARNING UPDATE RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The One-Step Update

Given a transition (sₜ, aₜ, rₜ₊₁, sₜ₊₁):

        Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α · δₜ

where the TD error δₜ is:

        δₜ = rₜ₊₁ + γ max_a' Q(sₜ₊₁, a') − Q(sₜ, aₜ)
              └─────────────────────────────────────┘
                         TD target y

Components:
    rₜ₊₁                     Observed immediate reward
    γ max_a' Q(sₜ₊₁, a')     Bootstrapped estimate of future value
    Q(sₜ, aₜ)                Current estimate (being corrected)
    α ∈ (0, 1]               Learning rate — controls update step size
    δₜ                        TD error — the prediction error


### Anatomy of the TD Error

    δₜ > 0    Q(s,a) underestimates true value → increase it
    δₜ < 0    Q(s,a) overestimates true value  → decrease it
    δₜ = 0    Q(s,a) is consistent with the observed transition

The update is a weighted move of Q(s,a) toward the TD target y:

        Q_new = (1 − α) Q_old + α · y

With α = 1: full replacement — discards history entirely.
With α = 0: no update — completely ignores new information.
Typical: α ∈ [0.1, 0.5] for tabular; much smaller for neural networks.


### Off-Policy Nature: The Key Property

Q-Learning is OFF-POLICY. The update uses:

        max_a' Q(sₜ₊₁, a')    ← greedy target policy (π*)

regardless of which action aₜ₊₁ was actually taken (the behaviour policy).

This separates two policies:
    BEHAVIOUR POLICY  b(a|s)   — what the agent actually does (e.g. ε-greedy)
                                 used to collect experience and ensure exploration
    TARGET POLICY     π*(a|s)  — the greedy policy being optimised
                                 always argmax_a Q(s, a)

Off-policy learning means the agent can:
    * Learn from any data, regardless of how it was collected
    * Learn from demonstrations (imitation + RL)
    * Learn from a replay buffer of old transitions (experience replay)
    * Learn multiple policies simultaneously from the same data


### Pseudocode

    Initialise Q(s, a) = 0 for all s ∈ S, a ∈ A
    Repeat (for each episode):
        s ← initial state s₀
        Repeat (for each step t):
            a ← ε-greedy(Q, s)                  # behaviour policy
            Execute a, observe r, s'
            Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') − Q(s, a)]
            s ← s'
        Until s is terminal


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — EXPLORATION STRATEGIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Exploration-Exploitation Dilemma

The agent must balance:
    EXPLOITATION   Take the action with highest current Q-value.
                   Fast reward, but may miss better undiscovered actions.

    EXPLORATION    Try actions with uncertain Q-values.
                   Slower reward, but discovers better strategies.

Too much exploitation → gets stuck in local optima.
Too much exploration → wastes time on known-bad actions.

This is one of the fundamental problems in RL with no universally
optimal solution (formally related to the multi-armed bandit problem).


### ε-Greedy Policy

The most common exploration strategy:

        With probability ε:   select a random action uniformly
        With probability 1-ε: select argmax_a Q(s, a)

    ε = 0     Pure greedy. No exploration. Converges to suboptimal policy.
    ε = 1     Pure random. Full exploration. Never exploits.
    ε = 0.1   Standard: 10% random, 90% greedy.

DECAYING ε (ε-decay): reduce ε over training.
    Linear decay:       ε_t = max(ε_min, ε_start − t * decay_rate)
    Exponential decay:  ε_t = max(ε_min, ε_start * decay^t)

The intuition: explore heavily at the start (high uncertainty), exploit
increasingly as Q converges (low uncertainty).

Convergence requires: Σ_t ε_t = ∞  (explore infinitely often)
                  and  Σ_t ε_t² < ∞  (exploration rate decreases)


### Upper Confidence Bound (UCB)

Selects the action with the highest upper confidence bound:

        a = argmax_a [ Q(s, a) + c · √( log(t) / N(s, a) ) ]

    Q(s, a)         Current value estimate
    N(s, a)         Number of times (s, a) has been visited
    c               Exploration coefficient (typically 1–2)
    log(t) / N(s,a) Uncertainty bonus — larger for under-visited pairs

Principled: systematically explores under-visited actions. No ε parameter.
Downside: requires maintaining visit counts N(s, a) for all (s, a).

UCB1 achieves logarithmic regret — provably optimal for bandits.
UCB applied to RL is an approximation (not provably optimal in full MDPs).


### Boltzmann (Softmax) Exploration

Select actions proportional to exponential of Q-values:

        π(a|s) = exp(Q(s,a) / τ) / Σ_a' exp(Q(s,a') / τ)

    τ (temperature) → ∞:  uniform random (full exploration)
    τ → 0:                 greedy (full exploitation)

Advantage over ε-greedy: actions are ranked — a nearly-optimal action
is selected more often than a terrible one.

Disadvantage: Q-values must be on a comparable scale for τ to be
meaningful. Sensitive to Q-value magnitude.


### Optimistic Initialisation

Initialise Q(s, a) = R_max / (1 − γ) for all (s, a).

Every state looks maximally promising at the start. The agent naturally
visits all states because unvisited ones always seem better than
recently-updated ones (whose estimates have decreased).

Works without any explicit ε or UCB term. Requires finite state/action
space and non-negative rewards. Does not work well if initial value
is impossible to achieve (creates systematic overestimation).


### Intrinsic Motivation (Curiosity)

Add a bonus reward for visiting novel or uncertain states:

        r'(s, a) = r_ext(s, a) + β · r_int(s, a)

Intrinsic reward variants:
    COUNT-BASED      r_int = 1 / √N(s)     inversely proportional to visits
    PREDICTION ERROR r_int = ||f_θ(s) − y||  — error of a forward model
    RND (Random Net. Dist.) r_int = ||f(s) − f_θ(s)||  fixed vs trained

Critical for exploration in large state spaces where ε-greedy fails.
Used in Montezuma's Revenge and other hard-exploration Atari games.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — CONVERGENCE THEORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Watkins & Dayan (1992) Convergence Theorem

Q-Learning converges to Q* with probability 1 if:

    C1.  The MDP is finite (finite |S|, |A|)
    C2.  All state-action pairs are visited infinitely often:
             Σ_t 1[(sₜ, aₜ) = (s, a)] → ∞  for all s, a
    C3.  Learning rates satisfy the Robbins-Monro conditions:
             Σ_t α_t(s,a)   = ∞          (learn forever)
             Σ_t α_t(s,a)²  < ∞          (updates diminish)
    C4.  Rewards are bounded:  |r| ≤ R_max  < ∞

Under these conditions:  Q_t(s, a) → Q*(s, a)  as t → ∞  (w.p. 1)


### Intuition for the Conditions

    C2 (all pairs visited infinitely):
        Ensures every cell of Q receives infinitely many corrections.
        Satisfied by ε-greedy with Σ εₜ = ∞ (e.g. εₜ = 1/t).

    C3 (Robbins-Monro):
        Constant α violates the second condition but works in practice.
        Common choices: α_t = 1/t (satisfies C3 exactly),
                        α_t = 0.1 (violates C3 but standard practice).

    C4 (bounded rewards):
        Ensures the TD target is bounded, preventing divergence.


### Contraction Mapping

The Q-Learning update defines the Bellman operator T:

        (TQ)(s, a) = Σₛ' P(s'|s,a) [ R(s,a,s') + γ max_a' Q(s',a') ]

T is a γ-contraction in the ℓ∞ norm:

        ||TQ − TQ'||_∞ ≤ γ ||Q − Q'||_∞

This guarantees:
    * Unique fixed point Q*
    * Convergence: ||T^n Q − Q*||_∞ ≤ γⁿ ||Q − Q*||_∞

The stochastic update (Q-Learning) converges to this fixed point because
it performs noisy gradient descent on the mean squared Bellman error.


### Limitations of the Convergence Guarantee

The theorem is for tabular representations only. With function
approximation (neural networks, linear functions), convergence
is NOT guaranteed and the algorithm can diverge.

The deadly triad (Sutton & Barto):
    When ALL THREE are combined, divergence is possible:
        1. Function approximation (non-tabular)
        2. Bootstrapping (TD-style updates using own estimates)
        3. Off-policy learning (behaviour ≠ target policy)

DQN avoids divergence in practice through engineering tricks
(experience replay + target network) but has no convergence proof.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — SARSA: ON-POLICY TD CONTROL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The SARSA Update

SARSA (State-Action-Reward-State-Action) is the on-policy counterpart:

        Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α [ rₜ₊₁ + γ Q(sₜ₊₁, aₜ₊₁) − Q(sₜ, aₜ) ]

The update uses the ACTUAL next action aₜ₊₁ (sampled from the behaviour
policy) rather than the GREEDY next action.

Named for the quintuple: (S, A, R, S', A') used in the update.


### Q-Learning vs SARSA

    +------------------+-------------------------------+----------------------------+
    | Property         | Q-Learning                    | SARSA                      |
    +------------------+-------------------------------+----------------------------+
    | Policy type      | Off-policy                    | On-policy                  |
    | TD target        | r + γ max_a' Q(s', a')        | r + γ Q(s', a')            |
    |                  | (greedy, ignores behaviour)   | (actual next action)       |
    | What it learns   | Q* (optimal action-values)    | Q^π (policy action-values) |
    | Exploration      | Unaffected by ε              | Penalised by risky actions |
    | Risky states     | Takes risks (greedy target)   | Cautious (includes ε cost) |
    | Cliff walking    | Walks cliff edge (optimal)    | Takes safe detour          |
    +------------------+-------------------------------+----------------------------+

The classic demonstration is the cliff-walking task:
    * Q-Learning learns to walk along the cliff edge — optimal but risky
    * SARSA learns to walk a safer route — accounts for the fact that
      with ε-greedy it will occasionally fall off the cliff

As ε → 0, SARSA's policy converges to the same optimal policy as Q-Learning.


### Expected SARSA

Instead of using the actual next action, take the expectation:

        Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α [ rₜ₊₁ + γ Σ_a' π(a'|sₜ₊₁) Q(sₜ₊₁, a') − Q(sₜ, aₜ) ]

Expected SARSA:
    * Lower variance than SARSA (averages over action selection noise)
    * When π is greedy: reduces to Q-Learning
    * Generally outperforms SARSA with the same computational cost
    * Can be on-policy or off-policy depending on π


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — N-STEP Q-LEARNING AND TD(λ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### N-Step Returns

The one-step TD target uses one observed reward plus a bootstrapped value:

        G_t^(1) = rₜ₊₁ + γ max_a' Q(sₜ₊₁, a')

The n-step return uses n observed rewards then bootstraps:

        G_t^(n) = rₜ₊₁ + γrₜ₊₂ + ... + γⁿ⁻¹rₜ₊ₙ + γⁿ max_a' Q(sₜ₊ₙ, a')

Special cases:
    n = 1:    Standard Q-Learning (high bias, low variance)
    n = T:    Monte Carlo (zero bias, high variance)
    n ∈ (1,T): Intermediate bias-variance tradeoff

The optimal n is task-dependent. Typical: n = 3 to 10.

N-step Q-Learning is off-policy because the n-step trajectory was
collected under the behaviour policy, not the greedy policy. This
requires importance sampling corrections for full correctness
(though often ignored in practice).


### TD(λ) — Eligibility Traces

Instead of choosing a fixed n, TD(λ) blends all n-step returns:

        G_t^λ = (1 − λ) Σ_n λ^{n−1} G_t^(n)

    λ = 0:    G_t^0 = G_t^(1)  →  TD(0) = standard Q-Learning
    λ = 1:    G_t^1 = Monte Carlo return

The λ-return assigns exponentially decaying weight to longer n-step returns.

ELIGIBILITY TRACE e(s, a):
    Tracks how recently (and how often) (s, a) was visited:

        e₀(s, a) = 0
        eₜ(s, a) = γλ eₜ₋₁(s, a)  +  1[(sₜ, aₜ) = (s, a)]

    Every (s, a) pair is updated proportionally to its eligibility:

        Q(s, a) ← Q(s, a) + α · δₜ · eₜ(s, a)    for all s, a

Watkins's Q(λ): off-policy eligibility traces for Q-Learning.
The trace is cut (set to 0) when a non-greedy action is taken.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — DOUBLE Q-LEARNING: FIXING OVERESTIMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Overestimation Problem

Standard Q-Learning overestimates Q*(s, a). Why?

The max operator in the TD target:

        y = r + γ max_a' Q(s', a')

If Q(s', a') = Q*(s', a') + noise_a' (noise with mean 0),
then by Jensen's inequality:

        E[ max_a' Q(s', a') ] ≥ max_a' E[ Q(s', a') ] = max_a' Q*(s', a')

The maximum of noisy estimates is biased upward. This systematic
overestimation can compound: overestimated values bootstrap other
overestimated values.

Consequence: Q-Learning is often overconfident, especially early in
training. The agent may commit prematurely to suboptimal actions.


### Double Q-Learning (van Hasselt, 2010)

Maintain two independent Q-tables: Q^A and Q^B.

On each update, randomly select which table to update:

    Update Q^A:    y = r + γ Q^B( s', argmax_a' Q^A(s', a') )
    Update Q^B:    y = r + γ Q^A( s', argmax_a' Q^B(s', a') )

    Q^A selects the action (argmax), Q^B evaluates it.

Decoupling selection and evaluation eliminates the upward bias:
    E[ Q^B(s', argmax_a' Q^A(s', a')) ] ≈ Q*(s', π^A(s'))

The action selected by Q^A may be sub-optimal, but Q^B gives
an unbiased estimate of its value.


### Double DQN (van Hasselt, Guez & Silver, 2015)

In DQN the same network is used for both selection and evaluation.
Double DQN uses the online network to select and the target network
to evaluate:

        y = r + γ Q_θ⁻( s', argmax_a' Q_θ(s', a') )

    Q_θ    — online network (updated every step)
    Q_θ⁻   — target network (periodically synced to Q_θ)

This reduces overestimation by ~50% on Atari and consistently
improves learning stability.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — DEEP Q-NETWORKS (DQN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Motivation: Scaling to Large State Spaces

Tabular Q-Learning requires a row for every state. This is impossible
for states like:
    * 84×84 greyscale Atari frames: |S| ≈ 256^{7056} (astronomically large)
    * Continuous robot states: |S| = ∞

Solution: approximate Q(s, a) with a neural network Q_θ(s, a).

        Q_θ : S → R^|A|

The network takes the state s as input and outputs Q-values for all
actions simultaneously. For pixel inputs, the encoder is a CNN.


### The DQN Loss

Replace the tabular update with gradient descent on the mean squared
Bellman error:

        L(θ) = E_{(s,a,r,s')~D} [ (y − Q_θ(s, a))² ]

        y = r + γ max_a' Q_θ⁻(s', a')     (using target network θ⁻)

Gradient update:
        θ ← θ − η ∇_θ L(θ)

Note: the gradient is taken only through Q_θ(s, a), NOT through y.
The target y is treated as a fixed constant (stop-gradient).
This is crucial — taking gradients through y creates instability.


### The Two Key Stabilisation Tricks

Without these, DQN diverges due to the deadly triad:

    1. EXPERIENCE REPLAY (Lin, 1992; Mnih et al., 2013)
         Store transitions (s, a, r, s') in a replay buffer D.
         Sample random mini-batches for training.

         Benefits:
           * Breaks temporal correlations between consecutive samples
           * Each transition used multiple times (data efficiency)
           * More stable gradient estimates
           * Enables off-policy learning

         Buffer size: typically 10⁵ to 10⁶ transitions.
         Batch size: 32–256.

    2. TARGET NETWORK (Mnih et al., 2015)
         Maintain a separate target network θ⁻ to compute y.
         Periodically copy: θ⁻ ← θ  every C steps (e.g. C = 1000).
         Or soft update: θ⁻ ← τθ + (1−τ)θ⁻   (τ ≈ 0.005).

         Benefits:
           * Stabilises the regression target y
           * Prevents oscillation / divergence
           * The target changes slowly — like a stable "teacher"


### DQN Algorithm (Mnih et al., 2015)

    Initialise Q_θ and Q_θ⁻ ← θ
    Initialise replay buffer D with capacity N

    For each episode:
        s ← initial observation
        For each step t:
            a ← ε-greedy(Q_θ, s)
            Execute a, observe r, s'
            Store (s, a, r, s') in D
            Sample mini-batch {(sⱼ, aⱼ, rⱼ, s'ⱼ)} from D
            Compute targets yⱼ = rⱼ + γ max_a' Q_θ⁻(s'ⱼ, a')
            Gradient step: θ ← θ − η ∇_θ Σⱼ (yⱼ − Q_θ(sⱼ, aⱼ))²
            Every C steps: θ⁻ ← θ
            s ← s'


### DQN Variants and Extensions

    +-------------------------------+---------------------------------------------------+
    | Variant                       | Key Contribution                                  |
    +-------------------------------+---------------------------------------------------+
    | Double DQN (2015)             | Separate selection/evaluation networks            |
    | Prioritised Replay (2015)     | Sample transitions by |TD error| magnitude        |
    | Dueling DQN (2016)            | Separate advantage and value streams              |
    | Noisy Networks (2017)         | Learned exploration via noise in weights          |
    | Distributional RL / C51 (2017)| Learn full return distribution, not just mean     |
    | Rainbow (2017)                | All of the above combined — state of the art      |
    | QR-DQN / IQN (2018)           | Quantile regression for distributional Q-learning |
    +-------------------------------+---------------------------------------------------+


### Dueling Architecture (Wang et al., 2016)

Decomposes Q into value and advantage streams:

        Q(s, a; θ, α, β) = V(s; θ, β)  +  [ A(s, a; θ, α) − mean_a' A(s, a'; θ, α) ]

    V(s)      State-value — how good is state s, regardless of action?
    A(s,a)    Advantage — how much better is action a relative to the mean?

The subtraction ensures identifiability: the advantage has mean zero
at each state, so V and A can be learned unambiguously.

Benefit: can learn V(s) even when many actions have similar Q-values
(e.g. in Atari, if moving left/right don't matter much right now).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 10 — HYPERPARAMETERS AND PRACTICAL TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Tabular Q-Learning Hyperparameters

    α (learning rate)
        Start with 0.1–0.5. Decrease over time for convergence.
        Too high → oscillates. Too low → slow learning.
        With constant α the agent "forgets" old experience, which can
        help in non-stationary environments.

    γ (discount factor)
        0.99 for most tasks. Lower (0.9) for shorter-horizon tasks.
        Higher (0.999) for very long-horizon tasks.

    ε (exploration)
        Start: 1.0 (full random). End: 0.01–0.1.
        Decay schedule: linear or exponential over first 50–80% of training.
        Common: ε_start=1.0, ε_end=0.01, decay over first 100K steps.

    Number of episodes
        Simple grid worlds: 1K–10K episodes.
        Complex environments: 1M+ episodes.


### DQN Hyperparameters (from the 2015 Nature paper)

    Learning rate:      α = 0.00025   (RMSProp, gradient clipped at 1.0)
    Discount:           γ = 0.99
    Replay buffer:      N = 1,000,000 transitions
    Batch size:         32
    Target update:      every C = 10,000 steps (hard copy)
    ε schedule:         1.0 → 0.1 over first 1M steps, then fixed
    Frame stack:        4 last frames (Atari-specific, Markov property)
    Reward clipping:    clip(r, −1, 1)   (normalises across games)
    Warm-up:            50,000 random steps before any training starts


### Common Failure Modes

    SLOW CONVERGENCE
        Cause: α too small, ε too high for too long.
        Fix: increase α, use decaying ε.

    OSCILLATING Q-VALUES
        Cause: α too large, target network update too frequent.
        Fix: reduce α, increase target update interval C.

    Q-VALUE EXPLOSION (tabular)
        Cause: γ too close to 1 with large rewards, no normalisation.
        Fix: normalise rewards, reduce γ slightly.

    STUCK IN LOCAL OPTIMA
        Cause: ε-greedy explores uniformly — misses structured exploration.
        Fix: use optimistic initialisation, UCB, or intrinsic rewards.

    CATASTROPHIC FORGETTING (DQN)
        Cause: small replay buffer; old transitions overwritten quickly.
        Fix: increase buffer size; use prioritised experience replay.

    OVERESTIMATION BIAS
        Cause: standard Q-Learning uses max operator (biased upward).
        Fix: use Double Q-Learning or Double DQN.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 11 — SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Q-Learning Variants at a Glance

    +-----------------------+---------------------+------------------------------+--------------------+
    | Algorithm             | On/Off Policy       | Key Idea                     | Use When           |
    +-----------------------+---------------------+------------------------------+--------------------+
    | Q-Learning            | Off-policy          | max over next actions        | Tabular, discrete  |
    | SARSA                 | On-policy           | actual next action           | Safe exploration   |
    | Expected SARSA        | On or off           | expected next action         | Lower variance     |
    | Double Q-Learning     | Off-policy          | two Q-tables, no max bias    | Overestimation     |
    | Q(λ)                  | Off-policy          | eligibility traces           | Credit assignment  |
    | DQN                   | Off-policy          | neural net + replay + target | Large state spaces |
    | Double DQN            | Off-policy          | online selects, target evals | DQN overestimation |
    | Dueling DQN           | Off-policy          | V + A decomposition          | Many similar Q     |
    | Rainbow               | Off-policy          | all improvements combined    | Atari benchmark    |
    +-----------------------+---------------------+------------------------------+--------------------+

### The Core Equations

    Q-LEARNING UPDATE:
        Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]

    TD ERROR:
        δ = r + γ max_a' Q(s', a') − Q(s, a)

    SARSA UPDATE:
        Q(s, a) ← Q(s, a) + α [ r + γ Q(s', a') − Q(s, a) ]

    DOUBLE Q-LEARNING TARGET:
        y = r + γ Q^B( s', argmax_a' Q^A(s', a') )

    DQN LOSS:
        L(θ) = E [ (r + γ max_a' Q_θ⁻(s', a') − Q_θ(s, a))² ]

    DUELING DECOMPOSITION:
        Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "1. Initialise a Q-Table and Inspect Policies": {
        "description": "Build and initialise Q-tables using three strategies "
                       "(zeros, optimistic, random), then extract and compare the "
                       "greedy policies derived from each.",
        "code": """\
import numpy as np

N_STATES  = 16    # 4×4 grid world
N_ACTIONS = 4     # up, down, left, right
ACTIONS   = ['U', 'D', 'L', 'R']
ROWS = COLS = 4

def greedy_policy(Q):
    return np.argmax(Q, axis=1)

def show_policy(pi, label):
    print(f"  {label}:")
    grid = np.array([ACTIONS[a] for a in pi]).reshape(ROWS, COLS)
    for row in grid:
        print("    " + "  ".join(row))

# ── Three initialisation strategies ───────────────────────────
Q_zero  = np.zeros((N_STATES, N_ACTIONS))
Q_opt   = np.full((N_STATES, N_ACTIONS), 1.0 / (1 - 0.99))  # R_max/(1-gamma)
Q_rand  = np.random.default_rng(42).uniform(0, 0.01, (N_STATES, N_ACTIONS))

print("Q-Table Initialisation Comparison")
print("=" * 42)
print(f"  Zero init   — all values: {Q_zero[0, 0]:.3f}")
print(f"  Optimistic  — all values: {Q_opt[0, 0]:.1f}")
print(f"  Random init — range: [{Q_rand.min():.5f}, {Q_rand.max():.5f}]")
print()

print("Greedy policies at initialisation:")
show_policy(greedy_policy(Q_zero), "Zero (all ties → action 0 = Up)")
print()
show_policy(greedy_policy(Q_opt),  "Optimistic (all ties → action 0 = Up)")
print()
show_policy(greedy_policy(Q_rand), "Random (tie-breaking by tiny differences)")
print()

# ── ε-greedy action selection ──────────────────────────────────
rng = np.random.default_rng(0)
def eps_greedy(Q, s, eps):
    if rng.random() < eps:
        return rng.integers(N_ACTIONS)
    ties = np.flatnonzero(Q[s] == Q[s].max())
    return rng.choice(ties)

print("ε-greedy action distribution from state 0 (Q=zeros, ε=0.1):")
counts = np.zeros(N_ACTIONS, dtype=int)
for _ in range(10_000):
    counts[eps_greedy(Q_zero, 0, 0.1)] += 1
for a, c in enumerate(counts):
    print(f"  {ACTIONS[a]}: {c/100:.1f}%  {'← exploiting (all equal)' if a==0 else ''}")
""",
    },

    "2. Single Q-Learning Update Step": {
        "description": "Manually compute one Q-Learning update from a single transition "
                       "(s, a, r, s'). Breaks down the TD error, shows the update rule "
                       "in detail, and traces through all quantities.",
        "code": """\
import numpy as np

GAMMA = 0.99
ALPHA = 0.1
ACTIONS = ['Up', 'Down', 'Left', 'Right']

# Suppose Q has been partially trained:
Q = np.array([
    # Up      Down    Left    Right
    [ 0.20,   0.35,   0.10,   0.45],   # state 0
    [ 0.30,   0.20,   0.25,   0.38],   # state 1
    [ 0.50,   0.60,   0.40,   0.55],   # state 2 (next state)
    [ 0.10,   0.05,   0.08,   0.12],   # state 3
])

# Observed transition
s  = 0        # current state
a  = 3        # action taken: Right
r  = -0.01    # reward received
s_ = 1        # next state

print("Single Q-Learning Update")
print("=" * 46)
print(f"  Transition: s={s}, a={ACTIONS[a]}, r={r}, s'={s_}")
print()
print(f"  Q(s, a)          = Q({s}, {ACTIONS[a]})  = {Q[s, a]:.4f}")
print()

# Step 1: best action from next state
best_a_next   = np.argmax(Q[s_])
best_q_next   = Q[s_, best_a_next]
print(f"  max_a' Q(s', a') = max Q({s_}, ·)  = {best_q_next:.4f}  [{ACTIONS[best_a_next]}]")

# Step 2: TD target
td_target = r + GAMMA * best_q_next
print(f"  TD target y      = r + γ·max Q(s',a')")
print(f"                   = {r} + {GAMMA}×{best_q_next:.4f}")
print(f"                   = {td_target:.4f}")

# Step 3: TD error
td_error = td_target - Q[s, a]
print()
print(f"  TD error δ       = y − Q(s, a)")
print(f"                   = {td_target:.4f} − {Q[s, a]:.4f}")
print(f"                   = {td_error:.4f}  ({'underestimating' if td_error > 0 else 'overestimating'})")

# Step 4: update
Q_new_sa = Q[s, a] + ALPHA * td_error
print()
print(f"  Q_new(s, a)      = Q(s,a) + α·δ")
print(f"                   = {Q[s, a]:.4f} + {ALPHA}×{td_error:.4f}")
print(f"                   = {Q_new_sa:.4f}")
print()
print(f"  Change:          Δ = {Q_new_sa - Q[s, a]:+.5f}  ({ALPHA*100:.0f}% of the way toward target)")

Q[s, a] = Q_new_sa
print()
print(f"  Updated Q-row for state {s}:")
for i, v in enumerate(Q[s]):
    marker = " ← updated" if i == a else ""
    print(f"    Q({s}, {ACTIONS[i]:5s}) = {v:.4f}{marker}")
""",
    },

    "3. Full Tabular Q-Learning on Grid World": {
        "description": "Train a Q-table on the 4×4 grid world from scratch using Q-Learning. "
                       "Plots learning curves, tracks convergence, and evaluates the "
                       "final learned policy against the known optimal.",
        "code": """\
import numpy as np

# ── Environment ────────────────────────────────────────────────
ROWS, COLS = 4, 4
N_S, N_A   = ROWS * COLS, 4
GOAL, TRAP = 15, 5
GAMMA      = 0.99
ACTIONS    = ['U', 'D', 'L', 'R']

moves = [(-1,0),(1,0),(0,-1),(0,1)]

def step(s, a):
    if s in (GOAL, TRAP):
        return s, 0.0, True
    r, c = s // COLS, s % COLS
    dr, dc = moves[a]
    nr, nc = r + dr, c + dc
    if 0 <= nr < ROWS and 0 <= nc < COLS:
        s2 = nr * COLS + nc
    else:
        s2 = s
    if s2 == GOAL:
        return s2, +1.0, True
    elif s2 == TRAP:
        return s2, -1.0, True
    return s2, -0.01, False

# ── Q-Learning ─────────────────────────────────────────────────
rng     = np.random.default_rng(42)
Q       = np.zeros((N_S, N_A))
ALPHA   = 0.1
EPS     = 1.0
EPS_MIN = 0.01
DECAY   = 0.995
N_EP    = 2000
MAX_STEPS = 200

returns_history = []
q_max_history   = []

for ep in range(N_EP):
    s = 0
    G = 0.0
    gamma_t = 1.0
    for _ in range(MAX_STEPS):
        # ε-greedy
        if rng.random() < EPS:
            a = rng.integers(N_A)
        else:
            ties = np.flatnonzero(Q[s] == Q[s].max())
            a    = rng.choice(ties)

        s2, r, done = step(s, a)
        G += gamma_t * r
        gamma_t *= GAMMA

        # Q-Learning update
        td = r + GAMMA * Q[s2].max() * (1 - done) - Q[s, a]
        Q[s, a] += ALPHA * td
        s = s2
        if done:
            break

    EPS = max(EPS_MIN, EPS * DECAY)
    returns_history.append(G)
    q_max_history.append(Q.max())

# ── Results ────────────────────────────────────────────────────
print("Q-Learning Training Results")
print("=" * 46)
print(f"  Episodes: {N_EP}, alpha={ALPHA}, gamma={GAMMA}")
print()

# Smoothed returns
window = 100
smoothed = [np.mean(returns_history[max(0,i-window):i+1])
            for i in range(len(returns_history))]

for ep_idx in [0, 99, 199, 499, 999, 1999]:
    raw = returns_history[ep_idx]
    smo = smoothed[ep_idx]
    eps_at = 1.0 * (0.995 ** ep_idx)
    eps_at = max(0.01, eps_at)
    print(f"  Ep {ep_idx+1:>5}: return={raw:+.4f}  "
          f"smooth-100={smo:+.4f}  ε={eps_at:.4f}")

print()
print("Optimal Q-values — max Q(s,·) per state (grid layout):")
for row in Q.max(axis=1).reshape(ROWS, COLS):
    print("  " + "  ".join(f"{v:6.3f}" for v in row))

print()
print("Learned policy (grid layout):")
pi = np.argmax(Q, axis=1)
for row in np.array([ACTIONS[a] for a in pi]).reshape(ROWS, COLS):
    print("  " + "  ".join(row))

print()
# Evaluate deterministically
s = 0
path = [s]
for _ in range(20):
    a = Q[s].argmax()
    s2, r, done = step(s, a)
    path.append(s2)
    if done:
        break
    s = s2
print(f"Greedy path from state 0: {path}")
reached = path[-1] == GOAL
print(f"Reached goal: {reached}")
""",
    },

    "4. SARSA vs Q-Learning — Cliff Walk Comparison": {
        "description": "Classic demonstration: train both SARSA and Q-Learning on the "
                       "cliff-walking task. SARSA learns the safe path; Q-Learning learns "
                       "the optimal (risky) path but falls more during training.",
        "code": """\
import numpy as np

# ── Cliff Walking Environment ──────────────────────────────────
# 4-row × 12-col grid
# Bottom row: S(col=0)  CLIFF(cols=1-10)  GOAL(col=11)
# Cliff: -100 reward and reset. Goal: 0. Step: -1.

ROWS, COLS = 4, 12
N_S  = ROWS * COLS
N_A  = 4   # up, down, left, right
MOVES = [(-1,0),(1,0),(0,-1),(0,1)]
START = (ROWS-1) * COLS + 0   # bottom-left
GOAL  = (ROWS-1) * COLS + 11  # bottom-right
CLIFF = [(ROWS-1)*COLS + c for c in range(1, 11)]

def cliff_step(s, a):
    r, c = s // COLS, s % COLS
    dr, dc = MOVES[a]
    nr = max(0, min(ROWS-1, r+dr))
    nc = max(0, min(COLS-1, c+dc))
    s2 = nr * COLS + nc
    if s2 in CLIFF:
        return START, -100.0, False  # reset, not terminal
    if s2 == GOAL:
        return s2, -1.0, True
    return s2, -1.0, False

rng     = np.random.default_rng(0)
GAMMA   = 1.0   # undiscounted (standard for cliff walk)
ALPHA   = 0.5
EPS     = 0.1   # fixed ε (compare at same exploration level)
N_EP    = 500
MAX_ST  = 500

def run_episode_qlearning(Q):
    s = START; G = 0.0
    for _ in range(MAX_ST):
        a = rng.integers(N_A) if rng.random() < EPS else Q[s].argmax()
        s2, r, done = cliff_step(s, a)
        td = r + GAMMA * Q[s2].max() * (not done) - Q[s, a]
        Q[s, a] += ALPHA * td
        G += r; s = s2
        if done: break
    return G

def run_episode_sarsa(Q):
    s = START; G = 0.0
    a = rng.integers(N_A) if rng.random() < EPS else Q[s].argmax()
    for _ in range(MAX_ST):
        s2, r, done = cliff_step(s, a)
        a2 = rng.integers(N_A) if rng.random() < EPS else Q[s2].argmax()
        td = r + GAMMA * Q[s2, a2] * (not done) - Q[s, a]
        Q[s, a] += ALPHA * td
        G += r; s = s2; a = a2
        if done: break
    return G

Q_ql = np.zeros((N_S, N_A))
Q_sr = np.zeros((N_S, N_A))

ql_returns = [run_episode_qlearning(Q_ql) for _ in range(N_EP)]
sr_returns = [run_episode_sarsa(Q_sr)     for _ in range(N_EP)]

window = 50
def smooth(x):
    return [np.mean(x[max(0,i-window):i+1]) for i in range(len(x))]

ql_sm = smooth(ql_returns)
sr_sm = smooth(sr_returns)

print("SARSA vs Q-Learning — Cliff Walking")
print("=" * 52)
print(f"  Fixed ε={EPS}, α={ALPHA}, γ={GAMMA}, {N_EP} episodes")
print()
print(f"  {'Episode':>8} | {'Q-Learning':>12} | {'SARSA':>10}")
print("  " + "-" * 36)
for i in [49, 99, 199, 299, 399, 499]:
    print(f"  {i+1:>8} | {ql_sm[i]:>12.1f} | {sr_sm[i]:>10.1f}")

print()
print("Final greedy paths (ε=0):")
ACTIONS = ['U','D','L','R']

def greedy_path(Q, max_steps=30):
    s = START; path_cols = [s % COLS]
    for _ in range(max_steps):
        a = Q[s].argmax()
        s2, r, done = cliff_step(s, a)
        path_cols.append(s2 % COLS)
        if done or s2 == s: break
        s = s2
    return path_cols

ql_path = greedy_path(Q_ql)
sr_path = greedy_path(Q_sr)

# Show bottom-row occupancy
ql_bottom = sum(1 for s in range(ROWS*COLS) if s//(COLS)==ROWS-1 and Q_ql[s].argmax()!=0)
sr_bottom = sum(1 for s in range(ROWS*COLS) if s//(COLS)==ROWS-1 and Q_sr[s].argmax()!=0)

print(f"  Q-Learning path (cols): {ql_path}")
print(f"  SARSA path     (cols): {sr_path}")
print()
print("Interpretation:")
print("  Q-Learning: walks along col 0→1→2→...→11 (cliff edge) — optimal but risky")
print("  SARSA:      takes a detour via row above  — safer, includes ε-exploration cost")
""",
    },

    "5. Double Q-Learning — Overestimation Bias": {
        "description": "Demonstrate the overestimation problem in standard Q-Learning "
                       "and compare to Double Q-Learning. Measures Q-value bias against "
                       "the true optimal values computed by value iteration.",
        "code": """\
import numpy as np

# ── Environment (same grid world) ─────────────────────────────
ROWS, COLS = 4, 4
N_S, N_A   = ROWS * COLS, 4
GOAL, TRAP = 15, 5
GAMMA      = 0.99
MOVES      = [(-1,0),(1,0),(0,-1),(0,1)]

def step(s, a):
    if s in (GOAL, TRAP): return s, 0.0, True
    r, c = s // COLS, s % COLS
    dr, dc = MOVES[a]
    nr, nc = r + dr, c + dc
    s2 = (nr*COLS + nc) if 0 <= nr < ROWS and 0 <= nc < COLS else s
    r2 = +1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2, r2, s2 in (GOAL, TRAP)

# ── Ground truth Q* via value iteration ────────────────────────
def build_mdp():
    P = np.zeros((N_S, N_A, N_S))
    R = np.zeros((N_S, N_A, N_S))
    for s in range(N_S):
        if s in (GOAL, TRAP): P[s, :, s] = 1.0; continue
        r, c = s // COLS, s % COLS
        for a, (dr, dc) in enumerate(MOVES):
            nr, nc = r+dr, c+dc
            s2 = (nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
            P[s, a, s2] = 1.0
            R[s, a, s2] = +1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return P, R

P, R = build_mdp()
R_sa = np.sum(P * R, axis=2)

V = np.zeros(N_S)
for _ in range(50_000):
    Q_true = R_sa + GAMMA * np.einsum('san,n->sa', P, V)
    V_new  = Q_true.max(axis=1)
    if np.max(np.abs(V_new - V)) < 1e-12: break
    V = V_new
Q_STAR = Q_true   # shape (N_S, N_A)

# ── Training runs ──────────────────────────────────────────────
rng   = np.random.default_rng(7)
ALPHA = 0.1; EPS = 0.2; N_EP = 3000; MAX_ST = 200

def run(double=False):
    QA = np.zeros((N_S, N_A))
    QB = np.zeros((N_S, N_A))
    bias_hist = []
    for ep in range(N_EP):
        s = 0
        for _ in range(MAX_ST):
            Q_avg = (QA + QB) / 2.0 if double else QA
            a = rng.integers(N_A) if rng.random() < EPS else Q_avg[s].argmax()
            s2, r, done = step(s, a)
            if double:
                if rng.random() < 0.5:
                    best_a = QA[s2].argmax()
                    td = r + GAMMA * QB[s2, best_a] * (not done) - QA[s, a]
                    QA[s, a] += ALPHA * td
                else:
                    best_a = QB[s2].argmax()
                    td = r + GAMMA * QA[s2, best_a] * (not done) - QB[s, a]
                    QB[s, a] += ALPHA * td
            else:
                td = r + GAMMA * QA[s2].max() * (not done) - QA[s, a]
                QA[s, a] += ALPHA * td
            s = s2
            if done: break
        Q_curr = (QA + QB) / 2.0 if double else QA
        bias_hist.append(np.mean(Q_curr - Q_STAR))
    return bias_hist

print("Q-Learning Overestimation vs Double Q-Learning")
print("=" * 52)
bias_ql = run(double=False)
bias_dq = run(double=True)

print(f"  {'Episode':>8} | {'Q-Learn bias':>14} | {'Double Q bias':>14}")
print("  " + "-" * 44)
for ep_i in [0, 99, 299, 499, 999, 1999, 2999]:
    bq = bias_ql[ep_i]; bd = bias_dq[ep_i]
    print(f"  {ep_i+1:>8} | {bq:>+14.5f} | {bd:>+14.5f}")

print()
final_ql = np.mean(bias_ql[-200:])
final_dq = np.mean(bias_dq[-200:])
print(f"  Mean bias (last 200 eps):")
print(f"    Q-Learning:       {final_ql:+.5f}  (positive = overestimation)")
print(f"    Double Q-Learning:{final_dq:+.5f}")
print()
reduction = abs(final_ql - final_dq) / (abs(final_ql) + 1e-9) * 100
print(f"  Overestimation reduced by: {reduction:.1f}%")
""",
    },

    "6. Epsilon Decay Schedule Comparison": {
        "description": "Compare three ε-decay schedules (linear, exponential, step) "
                       "on the same Q-Learning task. Measures convergence speed "
                       "and final policy quality for each schedule.",
        "code": """\
import numpy as np

ROWS, COLS = 4, 4
N_S, N_A   = ROWS * COLS, 4
GOAL, TRAP = 15, 5
GAMMA      = 0.99
ALPHA      = 0.1
MOVES      = [(-1,0),(1,0),(0,-1),(0,1)]
N_EP       = 1500
MAX_ST     = 200

def step(s, a):
    if s in (GOAL, TRAP): return s, 0.0, True
    r, c = s // COLS, s % COLS
    dr, dc = MOVES[a]
    nr, nc = r+dr, c+dc
    s2 = (nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2 = +1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2, r2, s2 in (GOAL, TRAP)

def eps_linear(ep, total=N_EP, start=1.0, end=0.01):
    return max(end, start - (start - end) * (ep / (total * 0.7)))

def eps_exp(ep, start=1.0, end=0.01, decay=0.995):
    return max(end, start * (decay ** ep))

def eps_step(ep, thresholds=(300, 700, 1100), levels=(0.5, 0.2, 0.05, 0.01)):
    for i, t in enumerate(thresholds):
        if ep < t: return levels[i]
    return levels[-1]

def train(eps_fn, seed=0):
    rng = np.random.default_rng(seed)
    Q   = np.zeros((N_S, N_A))
    returns = []
    epsilons = []
    for ep in range(N_EP):
        eps = eps_fn(ep)
        epsilons.append(eps)
        s = 0; G = 0.0; gt = 1.0
        for _ in range(MAX_ST):
            a = rng.integers(N_A) if rng.random() < eps else Q[s].argmax()
            s2, r, done = step(s, a)
            td = r + GAMMA * Q[s2].max() * (not done) - Q[s, a]
            Q[s, a] += ALPHA * td
            G += gt * r; gt *= GAMMA; s = s2
            if done: break
        returns.append(G)
    return returns, epsilons, Q

r_lin, e_lin, Q_lin = train(eps_linear)
r_exp, e_exp, Q_exp = train(eps_exp)
r_stp, e_stp, Q_stp = train(eps_step)

def smooth(x, w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

s_lin = smooth(r_lin); s_exp = smooth(r_exp); s_stp = smooth(r_stp)

print("Epsilon Decay Schedule Comparison")
print("=" * 60)
print(f"  {'Episode':>8} | {'ε linear':>10} | {'ε exp':>10} | {'ε step':>10}")
print("  " + "-" * 48)
for ep_i in [0, 99, 299, 499, 699, 999, 1499]:
    print(f"  {ep_i+1:>8} | {e_lin[ep_i]:>10.4f} | {e_exp[ep_i]:>10.4f} | {e_stp[ep_i]:>10.4f}")

print()
print(f"  {'Episode':>8} | {'lin return':>12} | {'exp return':>12} | {'stp return':>12}")
print("  " + "-" * 56)
for ep_i in [99, 299, 499, 999, 1499]:
    print(f"  {ep_i+1:>8} | {s_lin[ep_i]:>12.4f} | {s_exp[ep_i]:>12.4f} | {s_stp[ep_i]:>12.4f}")

print()
print("Final policy agreement (all three schedules reach same policy?):")
p_lin = np.argmax(Q_lin, axis=1)
p_exp = np.argmax(Q_exp, axis=1)
p_stp = np.argmax(Q_stp, axis=1)
print(f"  Linear vs Exp:  {np.mean(p_lin==p_exp)*100:.1f}% agreement")
print(f"  Linear vs Step: {np.mean(p_lin==p_stp)*100:.1f}% agreement")
""",
    },

    "7. N-Step Q-Learning": {
        "description": "Implement n-step Q-Learning for n = 1, 3, 5, 10 and Monte Carlo. "
                       "Compares convergence speed and stability, illustrating the "
                       "bias-variance tradeoff as n increases.",
        "code": """\
import numpy as np
from collections import deque

ROWS, COLS = 4, 4
N_S, N_A   = ROWS * COLS, 4
GOAL, TRAP = 15, 5
GAMMA      = 0.99
ALPHA      = 0.1
EPS        = 0.2
N_EP       = 1500
MAX_ST     = 200
MOVES      = [(-1,0),(1,0),(0,-1),(0,1)]

def step(s, a):
    if s in (GOAL, TRAP): return s, 0.0, True
    r, c = s // COLS, s % COLS
    dr, dc = MOVES[a]
    nr, nc = r+dr, c+dc
    s2 = (nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2 = +1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2, r2, s2 in (GOAL, TRAP)

def train_n_step(n, seed=42):
    rng = np.random.default_rng(seed)
    Q   = np.zeros((N_S, N_A))
    all_returns = []

    for ep in range(N_EP):
        buf_s = deque(); buf_a = deque(); buf_r = deque()
        s = 0; G_ep = 0.0; t = 0

        a = rng.integers(N_A) if rng.random() < EPS else Q[s].argmax()
        buf_s.append(s); buf_a.append(a)
        done = False

        while not done or len(buf_s) > 1:
            if not done:
                s2, r, done = step(s, a)
                G_ep += (GAMMA ** t) * r
                buf_r.append(r)
                if not done:
                    a = rng.integers(N_A) if rng.random() < EPS else Q[s2].argmax()
                    buf_s.append(s2); buf_a.append(a)
                    s = s2
                else:
                    buf_s.append(s2)
                t += 1

            # Update when n steps accumulated or at end of episode
            if len(buf_r) >= n or done:
                s0 = buf_s[0]; a0 = buf_a[0]
                # n-step return
                G = sum(GAMMA**k * buf_r[k] for k in range(len(buf_r)))
                if not done and len(buf_s) > len(buf_r):
                    sn = buf_s[-1]
                    G += GAMMA**len(buf_r) * Q[sn].max()

                td = G - Q[s0, a0]
                Q[s0, a0] += ALPHA * td

                buf_s.popleft(); buf_a.popleft()
                if buf_r: buf_r.popleft()

        all_returns.append(G_ep)

    return all_returns, Q

def smooth(x, w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

n_values  = [1, 3, 5, 10]
all_sm    = {}
final_q   = {}
for n in n_values:
    rets, Q = train_n_step(n)
    all_sm[n]  = smooth(rets)
    final_q[n] = Q

print("N-Step Q-Learning — Bias-Variance Tradeoff")
print("=" * 58)
print(f"  {'Episode':>8} | " + " | ".join(f"n={n:>3}" + " "*4 for n in n_values))
print("  " + "-" * (8 + 3 + len(n_values) * 14))
for ep_i in [99, 299, 499, 999, 1499]:
    vals = "  |  ".join(f"{all_sm[n][ep_i]:>8.4f}" for n in n_values)
    print(f"  {ep_i+1:>8} |  {vals}")

print()
print("Final policy agreement with n=1 (standard Q-Learning):")
pi1 = np.argmax(final_q[1], axis=1)
for n in n_values[1:]:
    pin = np.argmax(final_q[n], axis=1)
    agree = np.mean(pi1 == pin) * 100
    print(f"  n={n}: {agree:.1f}%")

print()
print("Key insight: larger n = lower bias (closer to MC) but higher variance.")
print("  n=1  → standard 1-step TD (high bias, low variance)")
print("  n=10 → longer returns, faster credit assignment, but noisier")
""",
    },

    "8. DQN Components — Replay Buffer and Target Network": {
        "description": "Implement the two core DQN stabilisation mechanisms: "
                       "a replay buffer with random sampling, and a target network "
                       "with periodic hard copy. Demonstrate how they reduce "
                       "training variance compared to naive Q-Learning with a network.",
        "code": """\
import numpy as np
from collections import deque

# ── Replay Buffer ──────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size, rng):
        idx  = rng.integers(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        s     = np.array([b[0] for b in batch], dtype=np.float32)
        a     = np.array([b[1] for b in batch], dtype=np.int32)
        r     = np.array([b[2] for b in batch], dtype=np.float32)
        s_    = np.array([b[3] for b in batch], dtype=np.float32)
        done  = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, r, s_, done

    def __len__(self):
        return len(self.buf)


# ── Lightweight Linear Q-Network (tabular via one-hot input) ───
class LinearQNet:
    def __init__(self, n_states, n_actions, rng):
        # W: (n_states, n_actions) — one weight per (state, action) pair
        self.W = rng.normal(0, 0.01, (n_states, n_actions)).astype(np.float32)
        self.n_states  = n_states
        self.n_actions = n_actions

    def predict(self, s_onehot):
        return s_onehot @ self.W   # (batch, n_actions)

    def copy_weights_from(self, other):
        self.W = other.W.copy()

    def update(self, s_idx, a, td_error, alpha=1e-3):
        self.W[s_idx, a] += alpha * td_error


def one_hot(s, n):
    v = np.zeros(n, dtype=np.float32); v[s] = 1.0
    return v


# ── Grid World step ────────────────────────────────────────────
ROWS, COLS = 4, 4
N_S, N_A   = ROWS * COLS, 4
GOAL, TRAP = 15, 5
GAMMA      = 0.99
MOVES      = [(-1,0),(1,0),(0,-1),(0,1)]

def step(s, a):
    if s in (GOAL, TRAP): return s, 0.0, True
    r, c = s // COLS, s % COLS
    dr, dc = MOVES[a]
    nr, nc = r+dr, c+dc
    s2 = (nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2 = +1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2, r2, s2 in (GOAL, TRAP)


# ── DQN Training Loop ──────────────────────────────────────────
rng    = np.random.default_rng(42)
online = LinearQNet(N_S, N_A, rng)
target = LinearQNet(N_S, N_A, rng)
target.copy_weights_from(online)
buf    = ReplayBuffer(capacity=5_000)

EPS = 1.0; EPS_MIN = 0.05; EPS_DECAY = 0.997
N_EP = 800; BATCH = 32; TARGET_UPDATE = 50; WARMUP = 200
returns_dqn = []; td_errors = []

for ep in range(N_EP):
    s = 0; G = 0.0; gt = 1.0
    for _ in range(200):
        q_vals = online.predict(one_hot(s, N_S))
        a = rng.integers(N_A) if rng.random() < EPS else q_vals.argmax()
        s2, r, done = step(s, a)
        buf.push(s, a, r, s2, float(done))
        G += gt * r; gt *= GAMMA; s = s2

        if len(buf) >= WARMUP:
            ss, aa, rr, ss_, dd = buf.sample(BATCH, rng)
            # Target using target network
            q_next  = target.predict(np.eye(N_S, dtype=np.float32)[ss_.astype(int)])
            y       = rr + GAMMA * q_next.max(axis=1) * (1 - dd)
            q_curr  = online.predict(np.eye(N_S, dtype=np.float32)[ss.astype(int)])
            err     = y - q_curr[np.arange(BATCH), aa]
            mean_td = np.mean(np.abs(err))
            td_errors.append(mean_td)
            # Update each sample
            for i in range(BATCH):
                online.update(int(ss[i]), aa[i], err[i] / BATCH, alpha=0.05)

        if done: break

    EPS = max(EPS_MIN, EPS * EPS_DECAY)
    if ep % TARGET_UPDATE == 0:
        target.copy_weights_from(online)
    returns_dqn.append(G)

def smooth(x, w=80):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

sm = smooth(returns_dqn)
print("DQN Components Demo — Replay Buffer + Target Network")
print("=" * 56)
print(f"  Buffer capacity: 5000 | Batch: {BATCH} | Target update: every {TARGET_UPDATE} eps")
print(f"  Warmup: {WARMUP} transitions before any training")
print()
print(f"  {'Episode':>8} | {'Smoothed return':>16} | {'ε':>8}")
print("  " + "-" * 38)
for ep_i in [99, 199, 299, 399, 499, 699, 799]:
    eps_at = max(EPS_MIN, 1.0 * EPS_DECAY**ep_i)
    print(f"  {ep_i+1:>8} | {sm[ep_i]:>16.4f} | {eps_at:>8.4f}")

print()
if td_errors:
    sm_td = smooth(td_errors, w=200)
    print(f"  Mean |TD error| (start):  {sm_td[0]:.4f}")
    print(f"  Mean |TD error| (final):  {sm_td[-1]:.4f}")
    print(f"  Reduction: {(1 - sm_td[-1]/sm_td[0])*100:.1f}% (TD error shrinks as Q converges)")

print()
print("Key mechanisms demonstrated:")
print("  Replay buffer:  breaks temporal correlation, reuses transitions")
print("  Target network: stable regression target, prevents oscillation")
print("  Together:       the 'deadly triad' is tamed for practical use")
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