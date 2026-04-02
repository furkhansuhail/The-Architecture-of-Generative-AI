"""Module: 04 · Policy Gradient"""

"""
Policy Gradient Methods
=======================

Policy gradient methods are the second great family of deep RL algorithms,
alongside value-based methods (Q-Learning, DQN). Where Q-Learning learns
an action-value function and derives a policy implicitly, policy gradient
methods optimise the policy DIRECTLY — parameterising π_θ and moving θ
in the direction that increases expected return.

This direct approach unlocks three capabilities that value-based methods
cannot provide:
  * Continuous action spaces (no argmax over uncountable actions)
  * Stochastic policies (required in partially observable / game-theoretic settings)
  * Rich policy classes (any differentiable parameterisation)

The core idea traces back to Williams (1992) with the REINFORCE algorithm.
It remained niche for decades, then became the backbone of modern deep RL
through actor-critic methods (A3C, PPO, SAC) and — crucially — RLHF,
the technique used to align large language models.

This module covers: the policy gradient theorem, REINFORCE, variance
reduction via baselines, advantage estimation, actor-critic foundations,
and the bridge to PPO.
"""

import re

TOPIC_NAME   = "Policy Gradient"
DISPLAY_NAME = "04 · Policy Gradient"
ICON         = "🎲"
SUBTITLE     = "REINFORCE and beyond — directly optimising the policy via gradient ascent"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — WHY POLICY GRADIENT?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Limitations of Value-Based Methods

Q-Learning and DQN learn Q*(s, a) then derive the policy implicitly:

        π*(s) = argmax_a Q*(s, a)

This works well for discrete action spaces but fails for:

    CONTINUOUS ACTIONS
        argmax_a Q(s, a) has no closed form over continuous a ∈ R^m.
        The max requires an inner optimisation loop at every step.

    STOCHASTIC POLICIES
        The greedy policy is always deterministic.
        In partially observable environments or multi-agent games,
        optimal policies can be stochastic — pure strategies are exploitable.

    HIGH-DIMENSIONAL DISCRETE ACTIONS
        argmax over 10,000+ discrete actions is expensive and degrades
        with the function approximation errors of the Q-network.

    STRUCTURED OUTPUT
        Policies over sequences (language), graphs, or combinatorial
        structures are natural probability distributions — not Q-tables.

Policy gradient methods sidestep all of these by parameterising π_θ
directly and optimising θ using gradient ascent on expected return.


### The Core Idea

Define the policy objective — the expected return under π_θ:

        J(θ) = E_π [ G₀ ] = E_π [ Σₜ γᵗ rₜ₊₁ ]

Goal: find θ* = argmax_θ J(θ)

How? Compute ∇_θ J(θ) and take gradient ascent steps:

        θ ← θ + α ∇_θ J(θ)

The challenge: J(θ) involves an expectation over trajectories τ,
and the distribution of τ depends on θ (via the policy). The environment
dynamics P(s'|s,a) also affect τ but are unknown. How do we compute
∇_θ J(θ) without knowing the environment model?

The answer is the Policy Gradient Theorem.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — THE POLICY GRADIENT THEOREM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Trajectory Formulation

A trajectory τ = (s₀, a₀, r₁, s₁, a₁, r₂, ..., sₜ) has probability:

        p(τ; θ) = d₀(s₀) · Π_t π_θ(aₜ|sₜ) · P(sₜ₊₁|sₜ, aₜ)

The objective in trajectory space:

        J(θ) = E_τ [ R(τ) ] = Σ_τ p(τ; θ) R(τ)

where R(τ) = Σₜ γᵗ rₜ₊₁ is the discounted return of trajectory τ.

Taking the gradient:

        ∇_θ J(θ) = Σ_τ ∇_θ p(τ; θ) · R(τ)

Using the log-derivative trick (REINFORCE trick):

        ∇_θ p(τ; θ) = p(τ; θ) · ∇_θ log p(τ; θ)

Therefore:

        ∇_θ J(θ) = E_τ [ ∇_θ log p(τ; θ) · R(τ) ]

Expanding log p(τ; θ):

        log p(τ; θ) = log d₀(s₀) + Σₜ [ log π_θ(aₜ|sₜ) + log P(sₜ₊₁|sₜ,aₜ) ]

The terms log d₀(s₀) and log P(sₜ₊₁|sₜ,aₜ) do not depend on θ.
Their gradients vanish. Only the policy terms survive:

        ∇_θ log p(τ; θ) = Σₜ ∇_θ log π_θ(aₜ|sₜ)

The environment model DROPS OUT. The gradient depends only on the policy!


### The Policy Gradient Theorem (Sutton et al., 1999)

For any differentiable policy π_θ, any start state distribution,
and any Q-function Q^π:

        ∇_θ J(θ) = E_π [ Σₜ ∇_θ log π_θ(aₜ|sₜ) · Q^π(sₜ, aₜ) ]

Intuition:
    ∇_θ log π_θ(aₜ|sₜ)   — the "direction" that increases the
                             probability of action aₜ from state sₜ
    Q^π(sₜ, aₜ)            — the "weight" — how good was this action?

    Update θ to make good actions (high Q) more likely
    and bad actions (low Q) less likely.

This theorem holds for:
    * Episodic and continuing tasks
    * Any policy parameterisation
    * Discounted or undiscounted returns
    * Without knowing P(s'|s,a) — model-free


### The Log-Derivative Trick

The trick ∇ log p = ∇p / p is fundamental to score function estimators.
It converts a gradient of a distribution into an expectation:

        ∇_θ E_x~p(x;θ) [ f(x) ] = E_x~p [ f(x) · ∇_θ log p(x; θ) ]

This identity is used everywhere in ML:
    * REINFORCE (policy gradients)
    * Variational autoencoders (ELBO gradient, with reparameterisation)
    * Attention mechanisms (straight-through estimator for discrete)
    * Black-box variational inference


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — REINFORCE: THE MONTE CARLO POLICY GRADIENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Algorithm

REINFORCE (Williams, 1992) instantiates the policy gradient theorem
using Monte Carlo returns Gₜ as the Q estimate:

        ∇_θ J(θ) ≈ (1/N) Σₙ Σₜ ∇_θ log π_θ(aₙₜ|sₙₜ) · Gₙₜ

where Gₙₜ = Σₖ₌ₜ^T γᵏ⁻ᵗ rₙₖ₊₁ is the return from time t in episode n.

Pseudocode:

    Initialise policy parameters θ
    Repeat for each episode:
        Generate trajectory τ = (s₀,a₀,r₁,...,sₜ) using π_θ
        For each timestep t = 0, ..., T-1:
            Compute Gₜ = Σₖ₌ₜ^T γᵏ⁻ᵗ rₖ₊₁
            θ ← θ + α · Gₜ · ∇_θ log π_θ(aₜ|sₜ)


### Why Gₜ Not G₀?

Using Gₜ (return from time t) rather than G₀ (total episode return) is
critical. Actions at time t are only responsible for rewards from t onward,
not for rewards before t (causality). Using G₀ for all steps is correct
in expectation but adds unnecessary variance.

Formally, the causal version is equivalent to the full sum because:
    E_π [ Σ_t ∇_θ log π_θ(aₜ|sₜ) · (Σ_k<t rₖ) ] = 0

Past rewards have zero covariance with future policy gradients.


### Policy Parameterisation

For DISCRETE actions:
    π_θ(a|s) = softmax( f_θ(s) )_a = exp(f_θ(s)_a) / Σ_a' exp(f_θ(s)_a')

    Score function: ∇_θ log π_θ(a|s) = ∇_θ f_θ(s)_a − Σ_a' π_θ(a'|s) ∇_θ f_θ(s)_a'
                                       = ∇_θ f_θ(s)_a − E_π [ ∇_θ f_θ(s)_a' ]

    In other words: one-hot action vector minus softmax probabilities,
    multiplied by the network Jacobian — exactly the gradient of cross-entropy.

For CONTINUOUS actions (diagonal Gaussian policy):
    π_θ(a|s) = N(μ_θ(s), σ_θ(s)²)

    Score function: ∇_θ log π_θ(a|s) = ∇_θ [ −log σ_θ(s) − (a−μ_θ(s))²/(2σ_θ(s)²) ]

    The gradient pushes μ toward high-return actions and adjusts σ
    (smaller σ for actions the policy is confident about).


### High Variance: The Core Problem

REINFORCE is unbiased — E[Gₜ · ∇_θ log π] = ∇_θ J(θ) exactly.

But Gₜ is a very noisy estimate of Q^π(sₜ, aₜ):
    * Full episodes required before any update
    * Return Gₜ has high variance due to stochastic policy and environment
    * The same action can receive very different Gₜ on different episodes
      simply due to randomness later in the trajectory

Variance of REINFORCE scales with:
    * Episode length T (longer → more random future rewards)
    * Discount γ (higher → rewards far in future contribute)
    * Policy entropy (more random → more variable trajectories)

High variance → noisy gradients → slow convergence or divergence.
Variance reduction is the central engineering problem of policy gradients.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — VARIANCE REDUCTION: BASELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Subtracting a Baseline

Adding any baseline b(sₜ) that does not depend on aₜ:

        ∇_θ J(θ) = E_π [ Σₜ ∇_θ log π_θ(aₜ|sₜ) · (Gₜ − b(sₜ)) ]

is still unbiased because:

        E_π [ ∇_θ log π_θ(aₜ|sₜ) · b(sₜ) ] = b(sₜ) E_π [ ∇_θ log π_θ(aₜ|sₜ) ]
                                               = b(sₜ) · ∇_θ Σ_a π_θ(a|sₜ) = b(sₜ) · 0 = 0

The expectation is zero because Σ_a π_θ(a|s) = 1 for all θ → gradient is zero.

Variance DOES depend on b. The optimal baseline minimises variance:

        b*(sₜ) = E[ Gₜ² · ||∇_θ log π||² ] / E[ ||∇_θ log π||² ]

In practice, this is approximated by V^π(sₜ), the state-value function.


### The State-Value Baseline V^π(s)

The most common baseline: set b(sₜ) = V^π(sₜ)

        Gₜ − V^π(sₜ)   = advantage estimate

Intuition:
    Gₜ > V^π(sₜ):   this episode went better than expected → increase π(aₜ|sₜ)
    Gₜ < V^π(sₜ):   this episode went worse than expected  → decrease π(aₜ|sₜ)
    Gₜ ≈ V^π(sₜ):   about as expected → small update

This is the "advantage" view: not just "was the return good?" but "was
it better than what you'd expect on average?"

The signal (Gₜ − V^π(sₜ)) is an unbiased estimate of the advantage:

        A^π(sₜ, aₜ) = Q^π(sₜ, aₜ) − V^π(sₜ)


### How Much Variance Reduction?

In practice, good baselines reduce gradient variance by 10–100x, leading
to dramatically faster and more stable learning. The value baseline
is so effective that REINFORCE without a baseline is rarely used.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — ADVANTAGE ESTIMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Monte Carlo Advantage (REINFORCE with baseline)

        Â_t^MC = Gₜ − V_w(sₜ)    where Gₜ = Σₖ₌ₜ γᵏ⁻ᵗ rₖ₊₁

    Unbiased estimate of A^π(s,a): exact in expectation.
    High variance: Gₜ depends on every future reward in the episode.
    Requires full episodes: no online updates during an episode.


### TD Residual (One-Step Advantage)

Replace Gₜ with the one-step TD target:

        Â_t^TD = rₜ₊₁ + γ V_w(sₜ₊₁) − V_w(sₜ)
               = δₜ      (the TD error)

    Biased: the TD target is only a one-step approximation of Q^π.
    Low variance: only one reward step + value function, no long rollout.
    Online: can be computed after each single step.


### Generalised Advantage Estimation (GAE, Schulman et al., 2015)

GAE interpolates between TD(0) and Monte Carlo using λ ∈ [0,1]:

        δₜ = rₜ₊₁ + γ V_w(sₜ₊₁) − V_w(sₜ)          (TD error)

        Â_t^GAE(γ,λ) = Σₖ₌₀^∞ (γλ)ᵏ δₜ₊ₖ
                      = δₜ + γλδₜ₊₁ + (γλ)²δₜ₊₂ + ...

Special cases:
    λ = 0:    Â = δₜ               (TD residual — low var, high bias)
    λ = 1:    Â = Gₜ − V_w(sₜ)    (MC advantage — high var, low bias)

GAE is the standard in modern policy gradient implementations (PPO, A3C).
Typical values: λ = 0.95 with γ = 0.99.

Efficient computation (backward pass through episode):

        Â_T = δ_T
        Â_t = δₜ + γλ Â_{t+1}


### Bias-Variance Tradeoff Summary

    +---------------------+-------------+-------------+---------------------+
    | Estimator           | Bias        | Variance    | Requires            |
    +---------------------+-------------+-------------+---------------------+
    | MC return Gₜ        | None (exact)| Very high   | Full episode        |
    | TD(0) residual δₜ   | High        | Low         | Single step         |
    | GAE (λ=0.95)        | Low         | Medium      | Fixed rollout       |
    | n-step return       | Medium      | Medium      | n steps             |
    +---------------------+-------------+-------------+---------------------+


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — ACTOR-CRITIC METHODS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Actor-Critic Architecture

Actor-Critic methods maintain two function approximators:

    ACTOR    π_θ(a|s)     — the policy, updated by policy gradient
    CRITIC   V_w(s)       — the value function, updated by TD learning

The actor uses the critic's value estimates as a baseline / advantage:

        ∇_θ J(θ) ≈ E [ ∇_θ log π_θ(aₜ|sₜ) · (δₜ) ]

where δₜ = rₜ₊₁ + γ V_w(sₜ₊₁) − V_w(sₜ) is the TD error computed by the critic.

This is the Generalised Policy Iteration (GPI) pattern applied to neural nets:
    Critic (evaluation): learn V^π using TD
    Actor (improvement): update π toward higher-value actions


### Why Actor-Critic is Better Than REINFORCE

    ONLINE LEARNING     Updates every step, not just at episode end.
    LOWER VARIANCE      TD advantage vs Monte Carlo return.
    WORKS FOR CONTINUING TASKS  No episode boundary needed.
    SHARED REPRESENTATIONS   Actor and critic share a feature encoder
                             (common in practice), saving compute.


### One-Step Actor-Critic

The simplest actor-critic, updating after every step:

    After transition (sₜ, aₜ, rₜ₊₁, sₜ₊₁):
        δₜ = rₜ₊₁ + γ V_w(sₜ₊₁) − V_w(sₜ)
        θ  ← θ + α_θ · δₜ · ∇_θ log π_θ(aₜ|sₜ)   (actor update)
        w  ← w + α_w · δₜ · ∇_w V_w(sₜ)           (critic update)

The critic update minimises the squared TD error:
        L_critic(w) = (rₜ₊₁ + γ V_w(sₜ₊₁) − V_w(sₜ))²


### Shared Network with Two Heads

In practice, actor and critic share a convolutional or MLP trunk:

                        shared encoder f_φ(s)
                       /                      \\
              π head                          V head
        π_θ(a|s) softmax               V_w(s) linear
              ↓                                ↓
          action dist                     scalar value

Total loss:
    L(φ, θ, w) = L_policy + c_value · L_value − c_entropy · H(π_θ(·|s))

    L_policy  = −E[ log π_θ(a|s) · Âₜ ]     (policy gradient, maximise)
    L_value   = E[ (Vₜ_target − V_w(s))² ]  (value regression, minimise)
    H(π)      = −Σ_a π(a|s) log π(a|s)       (entropy bonus, encourages exploration)

    c_value ≈ 0.5,  c_entropy ≈ 0.01


### A3C: Asynchronous Advantage Actor-Critic (Mnih et al., 2016)

Key innovation: run N actors in parallel in separate environment copies.
Each actor collects experience locally, computes gradients, and pushes
them to a shared global network asynchronously.

Benefits:
    * N actors generate decorrelated experience (replaces replay buffer)
    * Asynchronous updates naturally provide data diversity
    * Scales linearly with number of CPU cores
    * Works for both discrete and continuous action spaces

Update rule:
    Each actor collects n_steps transitions, then:
        Computes n-step returns: Rₜ = rₜ + γrₜ₊₁ + ... + γⁿ⁻¹rₜ₊ₙ₋₁ + γⁿ V(sₜ₊ₙ)
        Advantage: Âₜ = Rₜ − V_w(sₜ)
        Actor gradient: ∇_θ Σₜ log π_θ(aₜ|sₜ) · Âₜ + c_ent · ∇_θ H(π)
        Critic gradient: ∇_w Σₜ (Rₜ − V_w(sₜ))²
        Apply to global network (asynchronously, no lock)

A2C (synchronous): wait for all actors to finish, aggregate, then update.
More stable than A3C; used in practice more often today.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — ENTROPY REGULARISATION AND EXPLORATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Policy Entropy

Shannon entropy of the policy at state s:

        H(π_θ(·|s)) = −Σ_a π_θ(a|s) log π_θ(a|s)

A deterministic policy (one action probability = 1) has H = 0.
A uniform policy has H = log|A| (maximum entropy).

High entropy → exploration. Low entropy → exploitation.


### Entropy Bonus

Add a scaled entropy term to the policy gradient objective:

        J_total(θ) = J(θ) + c_ent · E_π [ H(π_θ(·|sₜ)) ]

        ∇_θ J_total = ∇_θ J + c_ent · ∇_θ H

This encourages the policy to maintain diversity, preventing:
    * Premature collapse to a deterministic policy
    * Getting stuck in local optima
    * Mode-seeking behaviour in multi-modal reward landscapes

Typical value: c_ent = 0.01. Too high → policy stays random forever.

In continuous action space (Gaussian policy):
    H(N(μ, σ)) = 0.5 log(2πe σ²)   →   encourages larger σ (more exploration).


### Maximum Entropy RL (MaxEnt RL)

Augment the reward with entropy:

        r'(sₜ, aₜ) = r(sₜ, aₜ) + α H(π(·|sₜ))

The optimal policy maximises this augmented objective.

Benefits:
    * Multiple near-optimal behaviours are all maintained (hedging)
    * More robust to model errors and distribution shift
    * Foundation of Soft Actor-Critic (SAC)

SAC (Haarnoja et al., 2018) combines MaxEnt with off-policy actor-critic,
achieving SOTA on continuous control with remarkable stability.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — TRUST REGION METHODS AND THE BRIDGE TO PPO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Policy Update Instability Problem

Standard gradient ascent on J(θ):

        θ_{k+1} = θ_k + α ∇_θ J(θ_k)

can take too large a step, moving θ into a bad region of policy space
where performance collapses. In contrast to supervised learning, a bad
policy generates bad data, which causes even worse policy updates —
a destructive feedback loop.

This is particularly dangerous because:
    * The objective surface of J(θ) is non-convex
    * Data distribution changes with each policy update
    * A step that looks small in parameter space can be large in policy space


### TRPO: Trust Region Policy Optimisation (Schulman et al., 2015)

Constrain how much the policy can change per update:

        max_θ  E_π_old [ π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ) · Âₜ ]
        s.t.   E_s [ KL(π_θ_old(·|s) || π_θ(·|s)) ] ≤ δ

The probability ratio r(θ) = π_θ(a|s) / π_θ_old(a|s) is the importance
sampling correction. Maximising the ratio-weighted advantage stays off-policy
while the KL constraint ensures the new policy remains close to the old one.

TRPO guarantees monotone policy improvement:
    J(π_{k+1}) ≥ J(π_k) − ε / (1 − γ)

Implementation requires solving a constrained optimisation problem using
conjugate gradient + line search — expensive but theoretically sound.


### PPO: Proximal Policy Optimisation (Schulman et al., 2017)

PPO achieves TRPO-like stability with a much simpler update rule.
Clip the probability ratio to prevent large updates:

        L_CLIP(θ) = E [ min( r(θ) Âₜ, clip(r(θ), 1-ε, 1+ε) Âₜ ) ]

where r(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)  and  ε = 0.2 (typical).

The clip acts as a pessimistic trust region:
    * If r(θ) > 1+ε (policy increased the probability too much), clip
    * If r(θ) < 1-ε (policy decreased the probability too much), clip
    * Only take the improvement if it's within the trust region

No KL constraint, no conjugate gradient, no line search.
Just a clipped surrogate loss + SGD.

PPO full objective:
    L_total = L_CLIP − c_value · L_value + c_ent · H
    L_value = (V_θ(sₜ) − Gₜ)²

PPO is the DOMINANT on-policy algorithm:
    * Default RL backbone for LLM alignment (RLHF)
    * Game-playing agents (OpenAI Five, AlphaStar)
    * Robotics locomotion
    * Continuous control benchmarks


### Why PPO Won

    SIMPLICITY      Only hyperparameter is ε (clip ratio). No KL constraint.
    STABILITY       Monotone improvement in practice without TRPO overhead.
    VERSATILITY     Works for discrete and continuous, on-policy and with RLHF.
    SCALABILITY     Easy to parallelise (multiple actors, same as A3C/A2C).
    SAMPLE REUSE    Uses each batch of experience for multiple gradient steps
                    (unlike pure on-policy which discards after one update).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — POLICY GRADIENT IN RLHF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Three Stages of RLHF

Reinforcement Learning from Human Feedback (Christiano et al., 2017;
Ziegler et al., 2019; Ouyang et al., 2022) trains language models to
align with human preferences using policy gradient methods:

    STAGE 1 — SUPERVISED FINE-TUNING (SFT)
        Fine-tune a pretrained LLM on high-quality (prompt, response) pairs.
        This is standard supervised learning, not RL.
        Output: SFT model π_SFT.

    STAGE 2 — REWARD MODELLING (RM)
        Collect preference data: humans compare pairs of responses
        and indicate which is better.
        Train a reward model R_φ(prompt, response) → scalar score.
        R_φ learns to predict human preference from the comparison data.

    STAGE 3 — RL FINE-TUNING (PPO)
        Use PPO to fine-tune the LLM to maximise the learned reward:

        max_θ  E_{x~D, y~π_θ(·|x)} [ R_φ(x, y) − β KL(π_θ || π_SFT) ]

        The KL penalty prevents the policy from deviating too far from π_SFT.
        Without it, the model would reward-hack — producing gibberish that
        maximises R_φ but is useless in practice.


### The LLM as a Policy

In RLHF, the LLM is literally a policy in the RL sense:

    STATE    s = prompt + tokens generated so far
    ACTION   a = next token to generate
    REWARD   r = R_φ(full response) at the end of generation (sparse)
             or per-token reward model output
    POLICY   π_θ(token | context)

The action space is the vocabulary (50,000+ tokens — large discrete).
The policy is the language model's next-token probability distribution.
PPO updates the weights of the LLM to maximise the learned reward.


### Direct Preference Optimisation (DPO)

DPO (Rafailov et al., 2023) derives a closed-form solution that
eliminates the separate RL stage entirely:

        L_DPO(θ) = −E_{(x, y_w, y_l)} [
            log σ( β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x) )
        ]

where y_w is the preferred response and y_l is the rejected response.

DPO is equivalent to RLHF with an optimal reward model, but:
    * No separate reward model training
    * No PPO rollouts
    * Direct supervised-style update on preference data

DPO is now widely adopted over PPO-based RLHF for its simplicity.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 10 — SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Core Equations

    POLICY GRADIENT THEOREM:
        ∇_θ J(θ) = E_π [ ∇_θ log π_θ(aₜ|sₜ) · Q^π(sₜ, aₜ) ]

    REINFORCE UPDATE:
        θ ← θ + α · Gₜ · ∇_θ log π_θ(aₜ|sₜ)

    REINFORCE WITH BASELINE:
        θ ← θ + α · (Gₜ − b(sₜ)) · ∇_θ log π_θ(aₜ|sₜ)

    TD ADVANTAGE:
        Â_t = rₜ₊₁ + γ V_w(sₜ₊₁) − V_w(sₜ)

    GAE:
        Â_t^GAE = Σₖ (γλ)ᵏ δₜ₊ₖ

    ACTOR-CRITIC (POLICY HEAD):
        L_actor = −E [ log π_θ(aₜ|sₜ) · Âₜ ]

    ACTOR-CRITIC (VALUE HEAD):
        L_critic = E [ (rₜ₊₁ + γ V_w(sₜ₊₁) − V_w(sₜ))² ]

    PPO CLIP:
        L_CLIP = E [ min( r(θ) Âₜ, clip(r(θ), 1−ε, 1+ε) Âₜ ) ]
        where r(θ) = π_θ(a|s) / π_θ_old(a|s)


### Algorithm Comparison

    +---------------------+------------+------------+---------------+------------------+
    | Algorithm           | On/Off     | Variance   | Key Feature   | Use Case         |
    +---------------------+------------+------------+---------------+------------------+
    | REINFORCE           | On-policy  | Very high  | MC returns    | Baseline         |
    | REINFORCE+baseline  | On-policy  | High       | V(s) subtracted| Simple tasks    |
    | One-step AC         | On-policy  | Medium     | TD advantage  | Continuing tasks |
    | A3C / A2C           | On-policy  | Medium     | Parallel actors| Scale with CPUs  |
    | TRPO                | On-policy  | Low        | KL constraint | Stability-critical|
    | PPO                 | On-policy  | Low        | Clipped ratio | Default choice   |
    | SAC                 | Off-policy | Low        | Max entropy   | Continuous ctrl  |
    | PPO + RLHF          | On-policy  | Low        | Reward model  | LLM alignment    |
    +---------------------+------------+------------+---------------+------------------+

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "1. Log-Derivative Trick and Score Function": {
        "description": "Demonstrate the log-derivative trick numerically. Verify that "
                       "the score function estimator gives the correct gradient of "
                       "E[f(x)] with respect to the parameters of a Gaussian.",
        "code": """\
import numpy as np

# ── Verify the log-derivative trick numerically ────────────────
# E_{x~N(mu,1)} [f(x)]  where f(x) = x^2
# Analytical gradient: d/d_mu E[x^2] = d/d_mu (mu^2 + 1) = 2*mu

rng = np.random.default_rng(0)

def f(x):
    return x ** 2

def score_function(x, mu, sigma=1.0):
    # d/d_mu log N(x; mu, sigma) = (x - mu) / sigma^2
    return (x - mu) / sigma**2

mu     = 2.0
sigma  = 1.0
N_true = 1_000_000   # large for ground truth
N_est  = 1_000       # small for typical single estimate

# Ground truth gradient via finite difference
eps    = 1e-4
E_plus  = np.mean(f(rng.normal(mu + eps, sigma, N_true)))
E_minus = np.mean(f(rng.normal(mu - eps, sigma, N_true)))
grad_fd = (E_plus - E_minus) / (2 * eps)

# Score function estimator (single sample)
def score_estimator(n_samples, seed):
    rng_l = np.random.default_rng(seed)
    x     = rng_l.normal(mu, sigma, n_samples)
    return np.mean(f(x) * score_function(x, mu, sigma))

# Analytical gradient
grad_analytical = 2.0 * mu   # d/d_mu (mu^2 + 1) = 2*mu

print("Log-Derivative Trick — Verifying Score Function Estimator")
print("=" * 58)
print(f"  f(x) = x^2,  x ~ N(mu={mu}, sigma={sigma})")
print(f"  Analytical gradient d/d_mu E[f(x)] = 2*mu = {grad_analytical:.4f}")
print(f"  Finite difference estimate          = {grad_fd:.4f}")
print()
print("  Score function estimator convergence:")
print(f"  {'N samples':>10} | {'Estimate':>12} | {'Error':>10}")
print("  " + "-" * 38)

for n in [1, 10, 100, 1_000, 10_000, 100_000]:
    est = np.mean([score_estimator(n, seed) for seed in range(20)])
    err = abs(est - grad_analytical)
    print(f"  {n:>10} | {est:>12.4f} | {err:>10.4f}")

print()
print("  Key insight: unbiased but high variance at small N.")
print("  Policy gradient = score function estimator applied to RL.")
print()

# ── Discrete action example ────────────────────────────────────
# Policy: softmax. Action space: {0,1,2}. Reward: r(a) known.
# Gradient of J(theta) = E_pi[r(a)] analytically and via score fn.
print("  Discrete action example:")
theta  = np.array([1.0, 0.5, -0.3])  # policy logits
probs  = np.exp(theta - theta.max()); probs /= probs.sum()
r_vals = np.array([1.0, 0.5, -1.0])  # reward per action

# Analytical: d/d_theta_i J = sum_a r(a) * grad_theta_i pi(a)
# d/d_theta_i softmax(theta)_a = pi_a * (delta_{ia} - pi_i)
grad_analytical_disc = np.array([
    sum(r_vals[a] * probs[a] * ((1 if a == i else 0) - probs[i])
        for a in range(3))
    for i in range(3)
])

# Score fn estimate
samples = rng.choice(3, p=probs, size=50_000)
scores  = (np.eye(3)[samples] - probs)  # d log pi / d theta_i = x_i - pi_i
sf_est  = np.mean(r_vals[samples, None] * scores, axis=0)

print(f"    Policy probs:       {probs}")
print(f"    Analytical gradient:{grad_analytical_disc.round(5)}")
print(f"    Score fn estimate:  {sf_est.round(5)}")
print(f"    Max error: {abs(grad_analytical_disc - sf_est).max():.5f}")
""",
    },

    "2. REINFORCE on Grid World": {
        "description": "Implement full REINFORCE with a softmax policy. Collect complete "
                       "episodes, compute Monte Carlo returns, and apply the policy gradient "
                       "update. Track convergence and compare to random policy.",
        "code": """\
import numpy as np

# ── Grid World ─────────────────────────────────────────────────
ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]; ANAMES=['U','D','L','R']

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)


# ── Softmax Policy (linear) ────────────────────────────────────
class SoftmaxPolicy:
    def __init__(self, n_s, n_a, rng):
        # Linear policy: logits = theta[s, a]
        self.theta = rng.normal(0, 0.1, (n_s, n_a)).astype(np.float32)

    def probs(self, s):
        logits = self.theta[s] - self.theta[s].max()
        e = np.exp(logits); return e / e.sum()

    def sample(self, s, rng):
        return rng.choice(N_A, p=self.probs(s))

    def log_prob(self, s, a):
        p = self.probs(s)
        return np.log(p[a] + 1e-8)

    def grad_log_prob(self, s, a):
        # d log pi(a|s) / d theta[s, :] = one_hot(a) - pi(·|s)
        p = self.probs(s)
        g = np.zeros_like(self.theta)
        g[s] = -p; g[s, a] += 1.0
        return g

    def update(self, grad, alpha):
        self.theta += alpha * grad


# ── REINFORCE Algorithm ────────────────────────────────────────
def run_episode(policy, rng, max_steps=200):
    s=0; traj=[]
    for _ in range(max_steps):
        a   = policy.sample(s, rng)
        s2,r,done = env_step(s,a)
        traj.append((s,a,r))
        s = s2
        if done: break
    return traj

def compute_returns(traj, gamma=GAMMA):
    T = len(traj); Gs = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        G = traj[t][2] + gamma * G
        Gs[t] = G
    return Gs

rng     = np.random.default_rng(42)
policy  = SoftmaxPolicy(N_S, N_A, rng)
ALPHA   = 0.05
N_EP    = 3000
returns = []

for ep in range(N_EP):
    traj = run_episode(policy, rng)
    Gs   = compute_returns(traj)

    # Accumulate gradient over all steps in episode
    total_grad = np.zeros_like(policy.theta)
    for t, (s, a, r) in enumerate(traj):
        total_grad += Gs[t] * policy.grad_log_prob(s, a)

    policy.update(total_grad / len(traj), ALPHA)
    G0 = sum(GAMMA**t * r for t, (_,_,r) in enumerate(traj))
    returns.append(G0)

def smooth(x, w=200):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

sm = smooth(returns)

print("REINFORCE on Grid World")
print("=" * 48)
print(f"  Policy: tabular softmax  |  alpha={ALPHA}  |  {N_EP} episodes")
print()
print(f"  {'Episode':>8} | {'Return':>10} | {'Smooth-200':>12}")
print("  " + "-"*36)
for i in [0, 99, 299, 499, 999, 1499, 1999, 2999]:
    print(f"  {i+1:>8} | {returns[i]:>10.4f} | {sm[i]:>12.4f}")

print()
# Evaluate greedy policy
print("  Greedy policy (argmax prob) after training:")
pi_grid = np.array([ANAMES[policy.probs(s).argmax()] for s in range(N_S)])
for row in pi_grid.reshape(ROWS, COLS):
    print("    " + "  ".join(row))

print()
# Greedy rollout
s=0; path=[s]
for _ in range(30):
    a=policy.probs(s).argmax(); s2,r,done=env_step(s,a)
    path.append(s2)
    if done: break
    s=s2
print(f"  Greedy path from state 0: {path}")
print(f"  Reached goal: {path[-1]==GOAL}")
""",
    },

    "3. REINFORCE with Baseline — Variance Reduction": {
        "description": "Compare REINFORCE with and without a value function baseline. "
                       "Measure gradient variance directly and show convergence speed "
                       "improvement when subtracting V(s) from returns.",
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

class Policy:
    def __init__(self, rng):
        self.theta = rng.normal(0, 0.1, (N_S, N_A)).astype(np.float32)
    def probs(self, s):
        e=np.exp(self.theta[s]-self.theta[s].max()); return e/e.sum()
    def sample(self, s, rng): return rng.choice(N_A, p=self.probs(s))
    def grad_log(self, s, a):
        p=self.probs(s); g=np.zeros_like(self.theta)
        g[s]=-p; g[s,a]+=1.0; return g
    def update(self, g, alpha): self.theta += alpha*g

def run_episode(pi, rng, max_st=200):
    s=0; traj=[]
    for _ in range(max_st):
        a=pi.sample(s,rng); s2,r,done=env_step(s,a)
        traj.append((s,a,r)); s=s2
        if done: break
    return traj

def returns(traj, gamma=GAMMA):
    T=len(traj); G=0.0; Gs=np.zeros(T)
    for t in reversed(range(T)):
        G=traj[t][2]+gamma*G; Gs[t]=G
    return Gs

def train(use_baseline, seed=42, n_ep=2000, alpha=0.05):
    rng  = np.random.default_rng(seed)
    pi   = Policy(rng)
    V    = np.zeros(N_S, dtype=np.float32)   # tabular baseline
    alpha_v = 0.1
    grad_norms = []; ep_returns = []

    for ep in range(n_ep):
        traj = run_episode(pi, rng)
        Gs   = returns(traj)
        total_grad = np.zeros_like(pi.theta)

        for t, (s, a, _) in enumerate(traj):
            advantage = (Gs[t] - V[s]) if use_baseline else Gs[t]
            total_grad += advantage * pi.grad_log(s, a)
            if use_baseline:
                V[s] += alpha_v * (Gs[t] - V[s])   # TD-0 style update

        g = total_grad / len(traj)
        grad_norms.append(np.linalg.norm(g))
        pi.update(g, alpha)
        G0 = sum(GAMMA**t * r for t,(_,_,r) in enumerate(traj))
        ep_returns.append(G0)

    return ep_returns, grad_norms, V

print("REINFORCE: With vs Without Baseline")
print("=" * 54)

r_no,  gn_no,  _   = train(False)
r_yes, gn_yes, V_b = train(True)

def smooth(x, w=200):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

sm_no  = smooth(r_no);  sm_yes = smooth(r_yes)
gn_sm_no  = smooth(gn_no);  gn_sm_yes = smooth(gn_yes)

print(f"  {'Episode':>8} | {'No baseline':>14} | {'With baseline':>14}")
print("  " + "-"*44)
for i in [99, 299, 499, 999, 1499, 1999]:
    print(f"  {i+1:>8} | {sm_no[i]:>14.4f} | {sm_yes[i]:>14.4f}")

print()
print("  Gradient norm (lower = less noisy signal):")
print(f"  {'Episode':>8} | {'No baseline':>14} | {'With baseline':>14}")
print("  " + "-"*44)
for i in [99, 299, 499, 999, 1999]:
    print(f"  {i+1:>8} | {gn_sm_no[i]:>14.4f} | {gn_sm_yes[i]:>14.4f}")

print()
var_no  = np.var(gn_no[-500:])
var_yes = np.var(gn_yes[-500:])
print(f"  Gradient norm variance (last 500 eps):")
print(f"    No baseline:   {var_no:.5f}")
print(f"    With baseline: {var_yes:.5f}")
print(f"    Reduction:     {(1 - var_yes/var_no)*100:.1f}%")
print()
print("  Learned baseline V(s) for each state:")
for row in V_b.reshape(ROWS,COLS):
    print("    " + "  ".join(f"{v:6.3f}" for v in row))
""",
    },

    "4. Generalised Advantage Estimation (GAE)": {
        "description": "Implement GAE and compare advantage estimates for different "
                       "values of λ. Show the bias-variance tradeoff: λ=0 (TD) vs "
                       "λ=1 (MC) vs λ=0.95 (typical PPO setting).",
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

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv   = 0.0
    for t in reversed(range(T)):
        next_val = values[t+1] if t < T-1 else 0.0
        mask     = 1.0 - dones[t]
        delta    = rewards[t] + gamma * next_val * mask - values[t]
        advantages[t] = last_adv = delta + gamma * lam * mask * last_adv
    returns = advantages + values[:T]
    return advantages, returns

# ── Get V* via value iteration (ground truth baseline) ─────────
def build_mdp():
    P=np.zeros((N_S,N_A,N_S)); R=np.zeros((N_S,N_A,N_S))
    for s in range(N_S):
        if s in (GOAL,TRAP): P[s,:,s]=1.0; continue
        r,c=s//COLS,s%COLS
        for a,(dr,dc) in enumerate(MOVES):
            nr,nc=r+dr,c+dc
            s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
            P[s,a,s2]=1.0
            R[s,a,s2]=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return P,R

P,R=build_mdp(); R_sa=np.sum(P*R,axis=2)
V=np.zeros(N_S)
for _ in range(100_000):
    Q=R_sa+GAMMA*np.einsum('san,n->sa',P,V)
    V_new=Q.max(1)
    if np.max(np.abs(V_new-V))<1e-12: break
    V=V_new
V_true=V.copy()

# ── Collect rollout with optimal policy ────────────────────────
rng = np.random.default_rng(0)
def rollout(n_steps=100):
    s=0; sss=[]; aaa=[]; rrr=[]; ddd=[]; vvv=[]
    for _ in range(n_steps):
        Q_s=R_sa[s]+GAMMA*np.einsum('an,n->a',P[s],V_true)
        a=Q_s.argmax()
        sss.append(s); aaa.append(a); vvv.append(V_true[s])
        s2,r,done=env_step(s,a)
        rrr.append(r); ddd.append(float(done))
        s=0 if done else s2
    vvv_arr=np.array(vvv, dtype=np.float32)
    # Append next value for last step
    vvv_ext=np.append(vvv_arr, V_true[s])
    return np.array(rrr), vvv_arr, vvv_ext, np.array(ddd), np.array(sss)

rewards, values, values_ext, dones, states = rollout(200)
T = len(rewards)
# True A^pi
Q_pi = R_sa + GAMMA * np.einsum('san,n->sa', P, V_true)
true_adv = np.array([Q_pi[states[t], int((R_sa[states[t]]+GAMMA*P[states[t]]@V_true).argmax())] - V_true[states[t]]
                     for t in range(T)])

lambdas = [0.0, 0.5, 0.95, 1.0]
print("Generalised Advantage Estimation (GAE)")
print("=" * 56)
print(f"  Rollout: {T} steps with near-optimal policy + true V*")
print()
print(f"  {'Lambda':>8} | {'Mean Â':>12} | {'Var Â':>12} | {'Corr w/ true A':>16}")
print("  " + "-"*56)
for lam in lambdas:
    adv, ret = compute_gae(rewards, values_ext[:T], dones, GAMMA, lam)
    adv_z = adv - adv.mean(); ta_z = true_adv - true_adv.mean()
    corr = np.corrcoef(adv_z, ta_z)[0,1] if ta_z.std() > 0 else 0.0
    label = ""
    if lam == 0.0:   label = "TD residual"
    elif lam == 0.95: label = "← PPO default"
    elif lam == 1.0:  label = "MC advantage"
    print(f"  {lam:>8.2f} | {adv.mean():>12.5f} | {adv.var():>12.5f} | {corr:>16.4f}  {label}")

print()
print("  First 10 advantage estimates (lambda=0.95 vs lambda=0.0):")
adv_95, _ = compute_gae(rewards, values_ext[:T], dones, GAMMA, 0.95)
adv_0,  _ = compute_gae(rewards, values_ext[:T], dones, GAMMA, 0.0)
print(f"  {'t':>4} | {'r_t':>8} | {'delta_t (lam=0)':>16} | {'GAE (lam=0.95)':>16}")
print("  " + "-"*50)
for t in range(10):
    print(f"  {t:>4} | {rewards[t]:>8.4f} | {adv_0[t]:>16.5f} | {adv_95[t]:>16.5f}")

print()
print("  Key insight: lambda=0 is low-variance but biased (one-step only).")
print("  lambda=0.95 blends many TD steps — lower bias, moderate variance.")
""",
    },

    "5. Actor-Critic with Separate Value Network": {
        "description": "Implement a full actor-critic with separate actor and critic networks. "
                       "The critic learns V(s) via TD; the actor uses TD advantage. "
                       "Compare convergence against REINFORCE.",
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

class Actor:
    def __init__(self, rng): self.theta=rng.normal(0,.1,(N_S,N_A)).astype(np.float32)
    def probs(self,s):
        e=np.exp(self.theta[s]-self.theta[s].max()); return e/e.sum()
    def sample(self,s,rng): return rng.choice(N_A,p=self.probs(s))
    def update(self,s,a,adv,alpha):
        p=self.probs(s)
        self.theta[s] += alpha*adv*(-p); self.theta[s,a] += alpha*adv

class Critic:
    def __init__(self): self.V=np.zeros(N_S,dtype=np.float32)
    def value(self,s): return self.V[s]
    def update(self,s,target,alpha): self.V[s] += alpha*(target-self.V[s])

# ── One-Step Actor-Critic ──────────────────────────────────────
def train_ac(seed=42, n_steps=80_000, alpha_a=0.03, alpha_c=0.1):
    rng=np.random.default_rng(seed)
    actor=Actor(rng); critic=Critic()
    s=0; returns=[]; ep_G=0.0; ep_gamma=1.0; step=0
    ep_returns=[]
    for _ in range(n_steps):
        a=actor.sample(s,rng)
        s2,r,done=env_step(s,a)
        ep_G += ep_gamma*r; ep_gamma*=GAMMA; step+=1

        # TD error (advantage estimate)
        v_next = critic.value(s2)*(1-done)
        target = r + GAMMA*v_next
        adv    = target - critic.value(s)

        actor.update(s, a, adv, alpha_a)
        critic.update(s, target, alpha_c)

        if done:
            ep_returns.append(ep_G)
            s=0; ep_G=0.0; ep_gamma=1.0
        else:
            s=s2
    return ep_returns, actor, critic

# ── REINFORCE (for comparison) ─────────────────────────────────
def train_reinforce(seed=42, n_ep=3000, alpha=0.05):
    rng=np.random.default_rng(seed)
    actor=Actor(rng); V=np.zeros(N_S,dtype=np.float32)
    ep_returns=[]
    for ep in range(n_ep):
        s=0; traj=[]
        for _ in range(200):
            a=actor.sample(s,rng); s2,r,done=env_step(s,a)
            traj.append((s,a,r)); s=s2
            if done: break
        T=len(traj); G=0.0; Gs=np.zeros(T)
        for t in reversed(range(T)):
            G=traj[t][2]+GAMMA*G; Gs[t]=G
        for t,(s,a,_) in enumerate(traj):
            adv=Gs[t]-V[s]; V[s]+=0.1*(Gs[t]-V[s])
            actor.update(s,a,adv/T,alpha)
        G0=sum(GAMMA**t*r for t,(_,_,r) in enumerate(traj))
        ep_returns.append(G0)
    return ep_returns

def smooth(x,w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

r_ac, actor_final, critic_final = train_ac()
r_reinf = train_reinforce()
sm_ac=smooth(r_ac); sm_rf=smooth(r_reinf)

print("Actor-Critic vs REINFORCE")
print("=" * 54)
print(f"  AC: one-step TD advantage  |  REINFORCE: MC returns+baseline")
print()
print(f"  {'Episode':>8} | {'REINFORCE':>12} | {'Actor-Critic':>14}")
print("  " + "-"*42)
for i in [49, 99, 199, 299, 499, 999, 1999, min(2999, len(sm_rf)-1)]:
    ac_val = sm_ac[i] if i < len(sm_ac) else sm_ac[-1]
    rf_val = sm_rf[i] if i < len(sm_rf) else sm_rf[-1]
    print(f"  {i+1:>8} | {rf_val:>12.4f} | {ac_val:>14.4f}")

print()
print("  Learned V(s) from Actor-Critic critic (grid layout):")
for row in critic_final.V.reshape(ROWS,COLS):
    print("    " + "  ".join(f"{v:6.3f}" for v in row))

print()
print("  Final actor-critic policy:")
for row in np.array([ANAMES[actor_final.probs(s).argmax()] for s in range(N_S)]).reshape(ROWS,COLS):
    print("    " + "  ".join(row))

print()
print("  Actor-Critic advantages: online updates, faster convergence,")
print("  no need to wait for episode completion.")
""",
    },

    "6. Entropy Bonus — Exploration Regularisation": {
        "description": "Train an actor-critic with and without an entropy bonus. "
                       "Show that the entropy regulariser prevents premature policy "
                       "collapse and maintains exploration, especially on sparse reward tasks.",
        "code": """\
import numpy as np

# ── Harder grid world with sparse reward ──────────────────────
# 5x5 grid, reward only on reaching goal (no dense shaping)
ROWS,COLS=5,5; N_S=25; N_A=4; GOAL=24; TRAP=12; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    if s2==GOAL: return s2,+1.0,True
    if s2==TRAP: return s2,-1.0,True
    return s2,0.0,False   # sparse: no step penalty

class Actor:
    def __init__(self, rng): self.theta=rng.normal(0,.05,(N_S,N_A)).astype(np.float32)
    def probs(self,s):
        e=np.exp(self.theta[s]-self.theta[s].max()); return e/e.sum()
    def sample(self,s,rng): return rng.choice(N_A,p=self.probs(s))
    def entropy(self,s):
        p=self.probs(s)
        return -np.sum(p*np.log(p+1e-8))
    def update(self,s,a,adv,ent_coef,alpha):
        p=self.probs(s)
        # Policy gradient + entropy gradient
        pg_grad = adv * (np.eye(N_A)[a] - p)
        ent_grad = (np.log(p+1e-8) + 1.0)  # d H / d logits
        self.theta[s] += alpha * (pg_grad + ent_coef * ent_grad)

def train(ent_coef, seed=0, n_ep=2000, alpha_a=0.05, alpha_c=0.1):
    rng=np.random.default_rng(seed)
    actor=Actor(rng); V=np.zeros(N_S,dtype=np.float32)
    returns=[]; entropies=[]; success_count=0

    for ep in range(n_ep):
        s=0; traj=[]; ep_ent=[]
        for _ in range(100):
            a=actor.sample(s,rng); ep_ent.append(actor.entropy(s))
            s2,r,done=env_step(s,a)
            traj.append((s,a,r)); s=s2
            if done: break

        T=len(traj); G=0.0; Gs=np.zeros(T)
        for t in reversed(range(T)):
            G=traj[t][2]+GAMMA*G; Gs[t]=G
        for t,(s,a,_) in enumerate(traj):
            adv=Gs[t]-V[s]
            V[s]+=alpha_c*(Gs[t]-V[s])
            actor.update(s,a,adv,ent_coef,alpha_a)

        G0=sum(GAMMA**t*r for t,(_,_,r) in enumerate(traj))
        returns.append(G0)
        entropies.append(np.mean(ep_ent))
        if traj[-1][0]==GOAL: success_count+=1

    return returns, entropies, success_count/n_ep*100

def smooth(x,w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

print("Entropy Bonus — Effect on Exploration")
print("=" * 58)
print("  Sparse reward grid (5x5, reward only on goal state)")
print()

coefs = [0.0, 0.01, 0.1]
labels= ["No entropy (c=0)", "Light (c=0.01)", "Strong (c=0.1)"]
all_returns=[]; all_entropy=[]; all_success=[]
for c, label in zip(coefs, labels):
    rets, ents, succ = train(c)
    all_returns.append(smooth(rets)); all_entropy.append(smooth(ents))
    all_success.append(succ)
    print(f"  {label}: success rate = {succ:.1f}%")

print()
print(f"  {'Episode':>8} | " + " | ".join(f"{l[:14]:>14}" for l in labels))
print("  " + "-"*(10 + len(coefs)*17))
for i in [49, 99, 199, 499, 999, 1499, 1999]:
    vals = " | ".join(f"{all_returns[j][i]:>14.4f}" for j in range(len(coefs)))
    print(f"  {i+1:>8} | {vals}")

print()
print(f"  Average policy entropy (last 200 eps):")
for j, label in enumerate(labels):
    avg_ent = np.mean(all_entropy[j][-200:])
    print(f"    {label}: {avg_ent:.4f} nats")

print()
print("  Key insight: zero entropy → premature collapse to suboptimal greedy policy.")
print("  Moderate entropy → sustained exploration → better coverage of sparse reward env.")
""",
    },

    "7. PPO Clipped Objective — Mechanics": {
        "description": "Implement the PPO clipped surrogate loss. Visualise how the clip "
                       "prevents destructive large policy updates. Compare unclipped "
                       "REINFORCE updates vs PPO-clipped updates on the same batch.",
        "code": """\
import numpy as np

# ── PPO Clipped Objective Mechanics ───────────────────────────
# Demonstrates the clip on a batch of (advantage, ratio) pairs

EPSILON = 0.2   # PPO clip parameter

def ppo_clip_loss(ratios, advantages, epsilon=EPSILON):
    # Unclipped term
    unclipped = ratios * advantages
    # Clipped term
    clipped   = np.clip(ratios, 1-epsilon, 1+epsilon) * advantages
    # Pessimistic minimum
    return -np.minimum(unclipped, clipped)   # negative for gradient ascent

def policy_gradient_loss(ratios, advantages):
    return -ratios * advantages   # standard PG loss (no clip)


# ── Grid World Environment ─────────────────────────────────────
ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

class Policy:
    def __init__(self, rng):
        self.theta=rng.normal(0,.1,(N_S,N_A)).astype(np.float32)
    def probs(self,s):
        e=np.exp(self.theta[s]-self.theta[s].max()); return e/e.sum()
    def log_prob(self,s,a):
        return np.log(self.probs(s)[a]+1e-8)
    def sample(self,s,rng): return rng.choice(N_A,p=self.probs(s))
    def copy(self):
        p2=Policy.__new__(Policy)
        p2.theta=self.theta.copy(); return p2

# ── Compare PG vs PPO on multiple gradient steps per batch ──
def train(use_ppo, seed=42, n_ep=2000, n_epochs=4, alpha=0.05):
    rng=np.random.default_rng(seed)
    policy=Policy(rng); V=np.zeros(N_S,dtype=np.float32)
    returns=[]; ratio_violations=[]

    for ep in range(n_ep):
        # Collect episode with OLD policy
        old_policy=policy.copy()
        s=0; traj=[]
        for _ in range(200):
            a=old_policy.sample(s,rng); s2,r,done=env_step(s,a)
            traj.append((s,a,r)); s=s2
            if done: break

        T=len(traj); G=0.0; Gs=np.zeros(T)
        for t in reversed(range(T)):
            G=traj[t][2]+GAMMA*G; Gs[t]=G

        # Advantages
        advs=np.array([Gs[t]-V[traj[t][0]] for t in range(T)])
        advs=(advs-advs.mean())/(advs.std()+1e-8)
        for t,(s,_,_) in enumerate(traj):
            V[s]+=0.1*(Gs[t]-V[s])

        # Multiple gradient steps on same batch (reuse data)
        ep_violations=0
        for epoch in range(n_epochs):
            for t,(s,a,_) in enumerate(traj):
                log_prob_old=old_policy.log_prob(s,a)
                log_prob_new=policy.log_prob(s,a)
                ratio=np.exp(log_prob_new - log_prob_old)

                if use_ppo:
                    clipped_ratio=np.clip(ratio, 1-EPSILON, 1+EPSILON)
                    adv=advs[t]
                    # Use clipped ratio if it would be worse
                    if (adv >= 0 and ratio > 1+EPSILON) or (adv < 0 and ratio < 1-EPSILON):
                        effective_ratio=clipped_ratio
                        ep_violations+=1
                    else:
                        effective_ratio=ratio
                else:
                    effective_ratio=ratio
                    if abs(ratio-1.0) > EPSILON: ep_violations+=1

                # Gradient update
                p=policy.probs(s)
                g=np.zeros_like(policy.theta)
                g[s]=-p; g[s,a]+=1.0
                policy.theta += alpha * effective_ratio * advs[t] * g / T

        G0=sum(GAMMA**t*r for t,(_,_,r) in enumerate(traj))
        returns.append(G0)
        ratio_violations.append(ep_violations/T)

    return returns, ratio_violations

def smooth(x,w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

r_pg,  rv_pg  = train(False, n_epochs=4)
r_ppo, rv_ppo = train(True,  n_epochs=4)
sm_pg  = smooth(r_pg);   sm_ppo = smooth(r_ppo)
sm_rv_pg = smooth(rv_pg); sm_rv_ppo = smooth(rv_ppo)

print("PPO Clip vs Standard Policy Gradient")
print(f"  epsilon={EPSILON}, n_epochs=4 (reuse each batch 4 times)")
print("=" * 58)
print()
print(f"  {'Episode':>8} | {'PG return':>12} | {'PPO return':>12} | {'PG violations':>15} | {'PPO violations':>15}")
print("  " + "-"*72)
for i in [99,299,499,999,1499,1999]:
    print(f"  {i+1:>8} | {sm_pg[i]:>12.4f} | {sm_ppo[i]:>12.4f} | {sm_rv_pg[i]:>15.4f} | {sm_rv_ppo[i]:>15.4f}")

print()
print("  Clip ratio mechanics — example batch:")
adv_examples  = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, 2.0])
ratio_examples= np.array([ 0.5,  0.8, 1.0, 1.2, 1.5, 2.0])
print(f"  {'ratio r(θ)':>12} | {'advantage':>12} | {'PG loss':>10} | {'PPO loss':>10} | {'clipped?':>10}")
print("  " + "-"*60)
for r,a in zip(ratio_examples, adv_examples):
    pg_l  = -(r*a)
    ppo_l = -min(r*a, np.clip(r,1-EPSILON,1+EPSILON)*a)
    clipped = abs(r-1)>EPSILON and abs(pg_l)>abs(ppo_l)
    print(f"  {r:>12.2f} | {a:>12.2f} | {pg_l:>10.4f} | {ppo_l:>10.4f} | {'YES' if clipped else 'no':>10}")
""",
    },

    "8. Policy Gradient on Continuous Actions (Gaussian Policy)": {
        "description": "Implement REINFORCE with a Gaussian policy for a continuous "
                       "action 1D control task. Shows how the mean and log-std are "
                       "optimised via the score function gradient.",
        "code": """\
import numpy as np

# ── 1D Continuous Control Task ────────────────────────────────
# State: position x in [-1, 1]
# Action: force f in [-2, 2] (continuous)
# Reward: -x^2 - 0.1*f^2  (penalise deviation from origin + large forces)
# Dynamics: x' = x + 0.1*f + noise
# Goal: drive x to 0 and stay there

class ContinuousEnv:
    def __init__(self, rng):
        self.rng=rng; self.reset()
    def reset(self):
        self.x=self.rng.uniform(-1.0,1.0); return np.array([self.x],dtype=np.float32)
    def step(self, f):
        f = np.clip(f,-2,2)
        self.x = np.clip(self.x + 0.1*f + self.rng.normal(0,0.05), -2,2)
        r = -self.x**2 - 0.01*f**2
        done = abs(self.x) > 1.9
        return np.array([self.x],dtype=np.float32), float(r), done


# ── Gaussian Policy (linear mean + log_std) ───────────────────
class GaussianPolicy:
    def __init__(self, rng):
        # theta_mu:   weights for mean  mu(s) = theta_mu * s
        # log_std:    scalar log standard deviation (state-independent)
        self.theta_mu  = rng.normal(0, 0.1, (1,)).astype(np.float32)
        self.log_std   = np.array([0.0], dtype=np.float32)  # std=1 initially

    def mean_std(self, s):
        mu  = (self.theta_mu * s).sum()
        std = np.exp(self.log_std[0])
        return float(mu), float(std)

    def sample(self, s, rng):
        mu, std = self.mean_std(s)
        return float(rng.normal(mu, std))

    def log_prob(self, s, a):
        mu, std = self.mean_std(s)
        return -0.5*((a-mu)/std)**2 - np.log(std) - 0.5*np.log(2*np.pi)

    def score(self, s, a):
        mu, std = self.mean_std(s)
        # d log pi / d theta_mu = (a - mu) / std^2 * s
        d_mu  = np.array([(a - mu) / std**2 * float(s[0])], dtype=np.float32)
        # d log pi / d log_std  = (a - mu)^2 / std^2 - 1
        d_std = float((a - mu)**2 / std**2 - 1.0)
        return d_mu, d_std

    def update(self, grads_mu, grads_std, alpha_mu=0.01, alpha_std=0.005):
        self.theta_mu += alpha_mu * np.atleast_1d(grads_mu).ravel()[:1]
        self.log_std  += alpha_std * float(np.atleast_1d(grads_std).ravel()[0])


GAMMA = 0.99

def run_episode(policy, env, rng, max_steps=50):
    s=env.reset(); traj=[]
    for _ in range(max_steps):
        a=policy.sample(s,rng)
        s2,r,done=env.step(a)
        traj.append((s.copy(),a,r))
        s=s2
        if done: break
    return traj

def compute_returns(traj, gamma=GAMMA):
    T=len(traj); G=0.0; Gs=np.zeros(T)
    for t in reversed(range(T)):
        G=traj[t][2]+gamma*G; Gs[t]=G
    return Gs

rng    = np.random.default_rng(42)
env    = ContinuousEnv(rng)
policy = GaussianPolicy(rng)
N_EP   = 2000; V_base=0.0; ep_returns=[]

for ep in range(N_EP):
    traj = run_episode(policy, env, rng)
    Gs   = compute_returns(traj)
    G0   = Gs[0]

    # Running baseline (simple EMA of returns)
    V_base = 0.99*V_base + 0.01*G0

    # Accumulate policy gradients
    grad_mu  = np.zeros_like(policy.theta_mu)
    grad_std = np.zeros_like(policy.log_std)
    T=len(traj)
    for t,(s,a,_) in enumerate(traj):
        advantage = Gs[t] - V_base
        d_mu, d_std = policy.score(s, a)
        grad_mu  += advantage * d_mu / T
        grad_std += advantage * d_std / T

    policy.update(grad_mu, np.array([grad_std]))
    ep_returns.append(G0)

def smooth(x,w=100):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]
sm=smooth(ep_returns)

print("REINFORCE with Gaussian Policy — Continuous Control")
print("=" * 56)
print(f"  Task: 1D balance. Action: continuous force in [-2, 2].")
print(f"  Initial mu(s) ≈ 0  |  Initial std = exp(0) = 1.0")
print()
print(f"  {'Episode':>8} | {'Return':>10} | {'Smooth':>10} | {'mu weight':>12} | {'log std':>10} | {'std':>8}")
print("  " + "-"*68)
for i in [0,49,99,199,499,999,1499,1999]:
    mu_w = policy.theta_mu[0]
    ls   = policy.log_std[0]
    std  = np.exp(ls)
    print(f"  {i+1:>8} | {ep_returns[i]:>10.4f} | {sm[i]:>10.4f} | {mu_w:>12.5f} | {ls:>10.5f} | {std:>8.5f}")

print()
print("  Final policy test (10 episodes, deterministic mean action):")
test_returns=[]
for _ in range(10):
    s=env.reset(); G=0.0; gamma=1.0
    for __ in range(50):
        mu,_=policy.mean_std(s); a=mu   # deterministic
        s2,r,done=env.step(a); G+=gamma*r; gamma*=GAMMA; s=s2
        if done: break
    test_returns.append(G)
print(f"    Mean return: {np.mean(test_returns):.4f}  (higher = better balance)")
print()
print("  Score function for Gaussian:")
print("  d log N(a;mu,sig) / d mu  = (a-mu)/sig^2  (push mu toward good actions)")
print("  d log N(a;mu,sig) / d sig = (a-mu)^2/sig^2 - 1  (shrink sig if return high)")
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