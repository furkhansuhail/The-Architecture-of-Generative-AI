"""Module: 01 · Markov Decision Processes"""

"""
Markov Decision Processes
=========================

The Markov Decision Process is the mathematical language of reinforcement
learning. Every RL algorithm — from tabular Q-learning to PPO to RLHF —
is ultimately an attempt to solve some form of MDP. Understanding the
formalism precisely means understanding what RL can and cannot do,
why algorithms are designed the way they are, and where the hard problems
actually live.

This module builds the complete MDP foundation: the 5-tuple formalism,
the Markov property, state and action spaces, transition dynamics,
reward design, the discount factor, policies, value functions,
the Bellman equations, and extensions to POMDPs and CMDPs.
"""

import re
import textwrap

TOPIC_NAME   = "Markov Decision Processes"
DISPLAY_NAME = "01 · Markov Decision Processes"
ICON         = "🎯"
SUBTITLE     = "States, actions, rewards — the mathematical framework for all of RL"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — MOTIVATION: WHY WE NEED THE MDP FORMALISM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Core Problem RL Solves

Reinforcement learning addresses sequential decision making under uncertainty.
An agent must choose actions over time, each action changes the state of the
world, and the agent receives feedback in the form of a scalar reward.

Three features distinguish this from supervised/unsupervised learning:

    1. SEQUENTIAL — decisions have long-term consequences; the effect of
       an action may not be felt until many steps later (credit assignment).

    2. INTERACTIVE — the agent's actions affect the distribution of future
       observations; unlike supervised learning, data is not fixed.

    3. EVALUATIVE — feedback is a scalar reward, not a correct answer;
       the agent must discover what is good through exploration.

The MDP gives us a precise, tractable mathematical framework for this problem.
It is not the only framework (POMDPs, Dec-MDPs, BAMDPs all exist) but it is
the right starting point: clean, well-studied, and solvable.


### Historical Context

    1953   Richard Bellman — dynamic programming, the optimality principle
    1957   Bellman — stochastic MDPs, the Bellman equation
    1960s  Howard — policy iteration algorithm
    1988   Sutton — temporal difference learning
    1992   Watkins & Dayan — Q-learning convergence proof
    1994   Puterman — "Markov Decision Processes" (the definitive textbook)
    1998   Sutton & Barto — "Reinforcement Learning: An Introduction"

Every modern RL paper traces its mathematical lineage back to these results.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — THE MDP TUPLE: (S, A, P, R, γ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Formal Definition

A Markov Decision Process is a 5-tuple:

        M = (S, A, P, R, γ)

    S   — State space
    A   — Action space
    P   — Transition function
    R   — Reward function
    γ   — Discount factor

At each discrete time step t = 0, 1, 2, ...:
    1. The environment is in state  sₜ ∈ S
    2. The agent takes action       aₜ ∈ A
    3. The environment transitions  sₜ₊₁ ~ P(·|sₜ, aₜ)
    4. The agent receives reward    rₜ₊₁ = R(sₜ, aₜ, sₜ₊₁)

The agent's goal is to find a policy π that maximises:

        E_π [ Σₜ γᵗ rₜ₊₁ ]

This section examines each component of the tuple in detail.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — THE MARKOV PROPERTY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Definition

The Markov property states that the future is independent of the past
given the present:

        P(sₜ₊₁ | sₜ, aₜ, sₜ₋₁, aₜ₋₁, ..., s₀, a₀)
                = P(sₜ₊₁ | sₜ, aₜ)

The current state sₜ is a sufficient statistic for predicting all future
states and rewards. No additional history is required.

Formally, a state sₜ is Markov if and only if:

        P(sₜ₊₁, rₜ₊₁ | s₀, a₀, r₁, ..., sₜ, aₜ) = P(sₜ₊₁, rₜ₊₁ | sₜ, aₜ)


### Why the Markov Property Matters

Without it, the agent would need to maintain an ever-growing history of all
past transitions — computationally intractable. The Markov property allows:

    * Value functions to depend only on current state (not history)
    * Bellman equations to be written recursively
    * Dynamic programming and TD algorithms to be correct
    * Policy search to be tractable


### Satisfying the Markov Property in Practice

In many real problems, the raw observations do not satisfy the Markov
property. Solutions:

    1. STATE AUGMENTATION — include enough history in the state.
       Example: Atari frames stacked (last 4 frames) to capture velocity.

    2. RECURRENT NETWORKS — maintain hidden state hₜ = f(hₜ₋₁, oₜ).
       The hidden state approximates a sufficient statistic.

    3. POMDP FRAMEWORK — explicitly model partial observability via
       belief states b(s) = P(s | o₁, a₁, ..., oₜ). See Part 11.

    4. REPRESENTATION LEARNING — learn a Markov state representation
       z = f(history) such that P(sₜ₊₁|zₜ, aₜ) = P(sₜ₊₁|history, aₜ).


### The Markov Chain (No Actions)

Before the full MDP, the simpler Markov Chain is instructive:

        P(sₜ₊₁ = s' | sₜ = s) = P_{ss'}

A Markov Chain is an MDP with a single (dummy) action — pure dynamics.
The transition matrix P ∈ R^{|S|×|S|} satisfies Σₛ' P_{ss'} = 1 for all s.

The stationary distribution d(s) satisfies d = Pᵀd — the left eigenvector
of P for eigenvalue 1. Relevant to policy gradient theory (the state
visitation distribution).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — STATE SPACE S
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Types of State Spaces

    FINITE DISCRETE       |S| < ∞. States are enumerable integers or symbols.
                          Example: grid world cells {0, 1, ..., N-1},
                          board game positions, traffic light states.
                          Enables exact tabular algorithms.

    INFINITE DISCRETE     |S| = ∞ but countable. Rare in practice.
                          Example: number of items in a queue.

    CONTINUOUS            S ⊆ R^n. Uncountably infinite.
                          Example: robot joint angles and velocities,
                          financial portfolio weights, physical simulations.
                          Requires function approximation.

    STRUCTURED / HYBRID   Combination of discrete and continuous components.
                          Example: robot gripper state (open/closed) + 6-DOF
                          joint positions. Common in robotics.

    HIGH-DIMENSIONAL RAW  S = image pixels, text tokens, audio waveforms.
                          Requires deep neural network state encoders.
                          Example: Atari (84×84×4), MineRL (64×64×3).


### State Space Design Principles

The choice of state representation critically affects learning efficiency:

    INFORMATION COMPLETENESS
        The state must contain all information needed to predict future
        rewards and transitions. Omitting relevant features breaks the
        Markov property and biases value estimates.

    COMPACT REPRESENTATION
        Unnecessary dimensions inflate the state space, slow learning,
        and increase sample complexity. Feature selection and
        dimensionality reduction are important.

    DISTINGUISHABILITY
        Two states that require different optimal actions must be
        distinguishable in the representation. Aliasing (two different
        situations mapped to the same state) is a key failure mode.

    SMOOTHNESS
        For function approximation, nearby states should have similar
        values. Raw pixels and structured features differ greatly here.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — ACTION SPACE A
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Types of Action Spaces

    FINITE DISCRETE       A = {a₁, a₂, ..., aₖ}. k possible actions.
                          Example: {up, down, left, right}, Atari joystick (18),
                          chess moves (variable), discrete dosing levels.
                          Enables tabular Q-tables and softmax policies.

    CONTINUOUS            A ⊆ R^m. Unbounded or bounded ranges.
                          Example: motor torques ∈ [-1, 1]^6, steering angle,
                          force magnitude, portfolio allocations.
                          Requires policy gradient or actor-critic methods.

    MULTI-DIMENSIONAL     A = A₁ × A₂ × ... × Aₘ. Each dimension independent.
                          Example: joint torques for a 7-DOF robot arm.
                          Combinatorial for discrete; handled naturally for
                          continuous with diagonal covariance policies.

    PARAMETERISED /       Discrete action types, each with continuous parameters.
    MIXED                 Example: soccer: {kick, run, pass} × {direction, force}.
                          Requires hybrid policy architectures.

    STATE-DEPENDENT       The valid action set A(s) ⊆ A depends on state s.
    (LEGAL ACTION MASK)   Example: chess moves, valid API calls.
                          Mask invalid actions before softmax in policy network.


### Action Space Design

    GRANULARITY
        Fine-grained action spaces allow precise control but increase
        exploration difficulty. Coarse spaces are easier to explore but
        may not express the optimal policy.

    ABSTRACTION
        Temporal abstraction (options/skills) defines macro-actions that
        last multiple timesteps. See Hierarchical RL.

    NORMALISATION
        Continuous actions should typically be normalised to [-1, 1]
        or [0, 1]. Unnormalised action spaces with very different scales
        across dimensions cause gradient and stability issues.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — TRANSITION FUNCTION P
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Definition

The transition function (also: transition kernel, dynamics model) specifies
the probability of moving to state s' after taking action a in state s:

        P : S × A × S → [0, 1]

        P(s' | s, a)  = Prob(sₜ₊₁ = s' | sₜ = s, aₜ = a)

Normalisation constraint:    Σₛ' P(s' | s, a) = 1    for all s, a


### Tabular Representation

For finite S and A:
    P is a 3D tensor:    P ∈ [0,1]^{|S| × |A| × |S|}
    or equivalently, for each (s,a) pair, a probability distribution over S.

    Example — 4×4 Grid World:
        S = {0, ..., 15},  A = {up, down, left, right}
        P(s+4 | s, down) = 1  if s not on bottom row
        P(s   | s, down) = 1  if s is on bottom row (wall bounce)


### Stochastic vs Deterministic Dynamics

    DETERMINISTIC         P(s'|s,a) ∈ {0, 1}.  One certain next state.
                          sₜ₊₁ = f(sₜ, aₜ) for some function f.
                          Simpler to plan in; not representative of real world.

    STOCHASTIC            P(s'|s,a) is a genuine distribution over S.
                          Sources: sensor noise, wind, opponent actions,
                          biological variability, quantum effects.
                          Requires expectation over outcomes.


### Known vs Unknown Dynamics

    MODEL-BASED RL        P is known (or learned). Agent can simulate
                          transitions and plan ahead (Dyna, MCTS, MuZero).
                          More sample efficient but model errors compound.

    MODEL-FREE RL         P is not explicitly represented. Agent learns
                          value functions or policies directly from samples.
                          More robust to model errors; more data hungry.

    LEARNED MODEL         P_θ(s'|s,a) is approximated by a neural network.
                          Errors in P_θ cause model bias — a key challenge.
                          Used in Dreamer, PETS, PILCO, MuZero.


### Transition Sparsity

Most real MDPs have sparse transitions: from state s with action a, only
a small fraction of states s' are reachable. Sparse representations
(adjacency lists rather than dense matrices) are critical for scalability.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — REWARD FUNCTION R
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Definition

The reward function maps transitions to a scalar feedback signal:

        R : S × A × S → R

        R(s, a, s')  =  expected immediate reward after transition (s, a, s')

Common simplified forms:
        R(s, a)      — depends only on state and action
        R(s)         — depends only on current state
        R(s')        — depends only on next state (arrival bonus)


### Types of Reward Signals

    DENSE REWARDS         Every step gives a non-zero reward signal.
                          Example: distance-to-goal decreases each step.
                          Advantage: rich learning signal at every step.
                          Risk: reward shaping can distort the true objective.

    SPARSE REWARDS        Non-zero reward only on task completion.
                          Example: +1 for winning, 0 otherwise.
                          Advantage: unambiguous objective.
                          Challenge: exploration problem — agent must
                          discover the reward by chance before it can learn.

    SHAPED REWARDS        Augmented reward: R'(s,a,s') = R(s,a,s') + F(s,a,s')
                          where F is a potential-based shaping function.
                          Theorem (Ng et al. 1999): if F(s,a,s') = γΦ(s') − Φ(s)
                          for some potential Φ, then the optimal policy is
                          preserved. Any other F can corrupt the objective.

    NEGATIVE REWARDS      Step penalties (−0.01 per step) encourage efficiency.
                          Dangerous rewards (−100 for falling) penalise
                          catastrophic failures.

    INTRINSIC REWARDS     Internally generated curiosity or novelty bonuses:
                          r_int(s) = f(visit count, prediction error, novelty).
                          Added to extrinsic reward for exploration. See RND, ICM.


### Reward Design: The Critical Skill

The reward function is the complete specification of the task. It is also
one of the hardest parts of RL in practice.

    REWARD HACKING
        The agent finds unintended behaviours that maximise reward without
        achieving the intended goal. Classic example: boat racing game where
        the agent circles endlessly collecting bonuses instead of racing.
        Goodhart's Law: "When a measure becomes a target, it ceases to be
        a good measure."

    PARTIAL OBSERVABILITY OF REWARD
        If the agent cannot observe all factors affecting reward, the
        effective MDP is not Markov and learning is degraded.

    REWARD MISSPECIFICATION
        Even with dense rewards, slight misspecification can produce
        undesirable policies. RLHF addresses this by learning R from
        human preference comparisons.

    DELAYED REWARD & CREDIT ASSIGNMENT
        When rewards are sparse and delayed, it is hard to determine
        which actions in a long trajectory caused the eventual reward.
        This is the fundamental credit assignment problem.


### Reward Normalisation

In practice, reward normalisation is critical for stability:

    PER-EPISODE NORMALISATION   Divide by running std of returns.
    CLIPPING                    r ← clip(r, −1, 1). Used in DQN for Atari.
    Z-SCORE NORMALISATION       r ← (r − μ_r) / σ_r.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — DISCOUNT FACTOR γ AND RETURN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Discount Factor

γ ∈ [0, 1) is a scalar that weights future rewards:

        Gₜ = rₜ₊₁ + γ rₜ₊₂ + γ² rₜ₊₃ + ... = Σₖ₌₀^∞ γᵏ rₜ₊ₖ₊₁

    γ = 0     Myopic agent. Maximises only immediate reward rₜ₊₁.
              No multi-step planning. Often suboptimal.

    γ → 1     Far-sighted agent. Nearly equal weight to all future rewards.
              Requires γ < 1 strictly for convergence with infinite horizon.

    γ = 0.99  Common practical choice. Effective horizon ≈ 1/(1−γ) = 100 steps.


### Why Discount?

Three complementary justifications:

    1. MATHEMATICAL CONVERGENCE
       For infinite-horizon tasks, Gₜ could be infinite without discounting.
       With γ < 1 and bounded rewards |r| ≤ R_max:
           |Gₜ| ≤ R_max / (1 − γ)    <  ∞

    2. ECONOMIC PREFERENCE FOR IMMEDIACY
       Future rewards are intrinsically less certain. Discounting reflects
       time preference — a reward now is worth more than the same reward later.

    3. MODELLING TERMINATION
       An episode that terminates at a random time τ ~ Geometric(1−γ) has
       expected return equal to the discounted infinite-horizon return.
       γ controls the mean episode length.


### Effective Planning Horizon

A key intuition: with discount γ, rewards beyond H ≈ 1/(1−γ) steps
contribute less than e⁻¹ ≈ 37% of their full value.

    γ = 0.9    → H ≈ 10 steps
    γ = 0.99   → H ≈ 100 steps
    γ = 0.999  → H ≈ 1000 steps

For short-horizon tasks (games with ≤ 100 steps), γ = 0.99 is usually fine.
For long-horizon tasks (robotics, long games), use γ ≥ 0.999.


### Episodic vs Continuing Tasks

    EPISODIC            The agent-environment interaction breaks into episodes.
                        Each episode starts from an initial state s₀ ~ d₀
                        and ends when a terminal state is reached.
                        Return is the sum over the finite episode.
                        Terminal states have V = 0 by convention.

    CONTINUING          The interaction never ends (t = 0, 1, 2, ..., ∞).
                        Return must be discounted (γ < 1) or averaged.
                        Example: industrial process control, server management.

    AVERAGE REWARD      Alternative to discounting for continuing tasks:
                        Maximise r̄ = lim_{T→∞} (1/T) Σₜ E[rₜ]
                        Differential value functions: V(s) = V_average(s) − r̄.
                        Avoids sensitivity to γ; used in some robotics work.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — POLICIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Definition

A policy π is the agent's decision-making rule — it maps states to actions
(or distributions over actions):

    DETERMINISTIC POLICY      π : S → A
                              aₜ = π(sₜ).  One action per state.

    STOCHASTIC POLICY         π : S × A → [0, 1]
                              aₜ ~ π(·|sₜ).  Distribution over actions.
                              Satisfies Σₐ π(a|sₜ) = 1 for all sₜ.


### Why Stochastic Policies?

    1. EXPLORATION
       A deterministic policy cannot explore. Adding randomness allows the
       agent to visit new (state, action) pairs and discover better behaviours.

    2. GAME THEORY
       In adversarial settings (multi-agent, competitive), stochastic policies
       (mixed strategies) may be necessary for Nash equilibria.

    3. POMDP / PARTIAL OBSERVABILITY
       When the agent cannot distinguish two states that require different
       actions, a stochastic policy can hedge by randomising.

    4. POLICY GRADIENT OPTIMISATION
       Stochastic policies are differentiable with respect to their parameters
       (e.g. softmax temperature, Gaussian mean/std). Deterministic policies
       require the deterministic policy gradient theorem (DPG).


### Policy Classes

    TABULAR POLICY        π stored as a |S| × |A| table.
                          Feasible only for small, discrete MDPs.

    LINEAR POLICY         π_θ(a|s) = softmax(θᵀ φ(s))
                          where φ(s) is a hand-crafted feature vector.

    NEURAL NETWORK POLICY π_θ(a|s) = network_θ(s)
                          θ are millions of parameters.
                          Discrete output: softmax over actions.
                          Continuous output: Gaussian N(μ_θ(s), Σ_θ(s)).

    STATIONARY POLICY     π(a|s) does not depend on time t.
                          Optimal policies for MDPs are always stationary.

    NON-STATIONARY        π_t(a|s) varies with time. Generally suboptimal
                          for MDPs but necessary for time-limited problems.


### Policy Parameterisation for Continuous Actions

For continuous action spaces A ⊆ R^m, the most common policy:

        Diagonal Gaussian:   π_θ(a|s) = N(μ_θ(s), diag(σ_θ(s)²))

        μ_θ(s)  — mean action, output of neural network
        σ_θ(s)  — standard deviation (can be state-dependent or global)

        Sample:  aₜ = μ_θ(sₜ) + σ_θ(sₜ) ⊙ ε,    ε ~ N(0, I)

The reparameterisation trick (above) makes the sample differentiable
with respect to θ — crucial for SAC and VAE-based methods.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 10 — VALUE FUNCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### State-Value Function V^π(s)

The state-value function under policy π is the expected return when starting
from state s and following π thereafter:

        V^π(s) = E_π [ Gₜ | sₜ = s ]
               = E_π [ Σₖ₌₀^∞ γᵏ rₜ₊ₖ₊₁ | sₜ = s ]

Intuition: V^π(s) answers "how good is it to be in state s, if I follow π?"

Properties:
    * V^π : S → R.  A scalar for every state.
    * V^π(terminal) = 0 by convention for episodic tasks.
    * Bounded: −R_max/(1−γ) ≤ V^π(s) ≤ R_max/(1−γ) for |r| ≤ R_max.


### Action-Value Function Q^π(s, a)

The action-value function (Q-function) is the expected return when starting
from state s, taking action a, then following π:

        Q^π(s, a) = E_π [ Gₜ | sₜ = s, aₜ = a ]
                  = E_π [ rₜ₊₁ + γ Gₜ₊₁ | sₜ = s, aₜ = a ]

Intuition: Q^π(s, a) answers "how good is action a from state s, then π?"

Relationship to V^π:

        V^π(s) = Σₐ π(a|s) Q^π(s, a)

        Q^π(s, a) = Σₛ' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]


### Advantage Function A^π(s, a)

The advantage measures how much better action a is compared to the
average action under π:

        A^π(s, a) = Q^π(s, a) − V^π(s)

Properties:
    * Σₐ π(a|s) A^π(s, a) = 0    (advantage averages to zero under π)
    * A^π(s, π*(s)) ≥ 0          (optimal action has non-negative advantage)
    * Used in policy gradient methods to reduce variance (Actor-Critic, PPO)


### Optimal Value Functions

        V*(s)    = max_π V^π(s)     for all s
        Q*(s, a) = max_π Q^π(s, a) for all s, a

The optimal policy can be derived from Q*:
        π*(s) = argmax_a Q*(s, a)

Key result: there always exists a deterministic stationary optimal policy
for finite MDPs (and for many infinite MDPs under regularity conditions).


### Greedy Policy w.r.t. a Value Function

Given any value function V, the greedy policy is:

        π_V(s) = argmax_a Σₛ' P(s'|s, a) [ R(s,a,s') + γ V(s') ]
               = argmax_a Q(s, a)

If V = V*, then π_V = π*. Policy improvement is monotone: the greedy policy
w.r.t. V^π is always at least as good as π.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 11 — BELLMAN EQUATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Bellman Expectation Equation

The most important recursive decomposition in RL:

    For V^π:
        V^π(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]

    For Q^π:
        Q^π(s,a) = Σₛ' P(s'|s,a) [ R(s,a,s') + γ Σₐ' π(a'|s') Q^π(s',a') ]

In words: the value of a state now equals the expected immediate reward
plus the discounted value of the next state (under the same policy).

    BACKUP DIAGRAM (for V^π):
                    s
                   / \
                  a₁  a₂    ← agent chooses (π)
                 / \
               s'₁  s'₂    ← environment transitions (P)
              +r    +r
                 ↓
              γV^π(s')

This recursive structure is the key insight: you only need to know the
value of the next state, not the entire future trajectory.


### Matrix Form (Finite MDPs)

For a fixed policy π, the Bellman expectation equations form a linear system:

        V^π = R^π + γ P^π V^π

where:
    V^π ∈ R^|S|              vector of state values
    R^π ∈ R^|S|              R^π(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) R(s,a,s')
    P^π ∈ [0,1]^{|S|×|S|}   P^π(s,s') = Σₐ π(a|s) P(s'|s,a)

Closed-form solution (for small MDPs):
        V^π = (I − γ P^π)⁻¹ R^π

The matrix (I − γP^π) is invertible since all eigenvalues of γP^π
have modulus ≤ γ < 1.

Computational cost: O(|S|³) for exact inversion — intractable for large |S|.
Iterative methods (policy evaluation) avoid the inversion. Cost: O(|S|²|A|)
per sweep, O(1/(1−γ)) sweeps for ε-convergence.


### Bellman Optimality Equation

The nonlinear equations characterising V* and Q*:

    For V*:
        V*(s) = max_a Σₛ' P(s'|s,a) [ R(s,a,s') + γ V*(s') ]

    For Q*:
        Q*(s,a) = Σₛ' P(s'|s,a) [ R(s,a,s') + γ max_a' Q*(s',a') ]

The max operator makes these equations nonlinear — they cannot be solved
by linear algebra. They are solved by:
    * Dynamic programming (value/policy iteration) — model known
    * Q-learning, DQN — model unknown, learn from samples
    * Fitted value iteration — with function approximation

Unique fixed point: V* is the unique solution to the Bellman optimality
equation. The Bellman operator T is a γ-contraction:

        ||TV − TW||_∞ ≤ γ ||V − W||_∞

This guarantees convergence of value iteration: Vₙ → V* as n → ∞.


### Bellman Error and Temporal Difference Error

    BELLMAN ERROR (full expectation):
        δ_BE(s) = V(s) − [R^π(s) + γ Σₛ' P^π(s,s') V(s')]
        Zero only when V = V^π.

    TD ERROR (single sample estimate):
        δₜ = rₜ₊₁ + γ V(sₜ₊₁) − V(sₜ)
        Unbiased estimate of Bellman error.
        Used as the update signal for all TD algorithms.

The TD error δₜ is the stochastic equivalent of the Bellman error —
it approximates the full expectation using a single observed transition.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 12 — DYNAMIC PROGRAMMING: SOLVING KNOWN MDPs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Overview

When P and R are known, the MDP can be solved exactly using dynamic
programming. DP algorithms apply the Bellman equations as update rules
until convergence.


### Policy Evaluation (Prediction)

Given π, compute V^π by iteratively applying the Bellman expectation equation:

    Initialise V₀(s) = 0 for all s
    Repeat until ||Vₖ₊₁ − Vₖ||_∞ < ε:
        Vₖ₊₁(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) [ R(s,a,s') + γ Vₖ(s') ]

Convergence: ||Vₖ − V^π||_∞ ≤ γᵏ ||V₀ − V^π||_∞ / (1−γ)
Rate: linear, factor γ per sweep. About log(1/ε(1−γ)) / log(1/γ) sweeps.


### Policy Improvement

Given V^π, construct a better policy:

    π'(s) = argmax_a Σₛ' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]

Policy improvement theorem: V^{π'}(s) ≥ V^π(s) for all s.
If V^π = V^{π'}, then V^π = V* (the policy is already optimal).


### Policy Iteration

Alternate between evaluation and improvement until convergence:

    Initialise π₀ arbitrarily
    Repeat until π is stable:
        1. Evaluate:  V^πₖ ← Policy-Evaluation(πₖ)
        2. Improve:   πₖ₊₁ ← Greedy(V^πₖ)

Convergence: finite MDPs converge in at most |A|^|S| policy updates.
In practice: typically converges in tens of iterations.
Bottleneck: inner policy evaluation loop (O(|S|²|A|) per sweep × many sweeps).


### Value Iteration

Fuse evaluation and improvement into a single update:

    Initialise V₀(s) = 0 for all s
    Repeat until ||Vₖ₊₁ − Vₖ||_∞ < ε:
        Vₖ₊₁(s) = max_a Σₛ' P(s'|s,a) [ R(s,a,s') + γ Vₖ(s') ]
    Extract policy: π*(s) = argmax_a Σₛ' P(s'|s,a) [ R(s,a,s') + γ V*(s') ]

Equivalent to policy iteration with one evaluation sweep per improvement.
Typically faster in practice than full policy iteration.
Convergence: same rate as policy evaluation, factor γ per sweep.


### Generalised Policy Iteration (GPI)

The unifying view: any combination of policy evaluation and policy
improvement converges to (π*, V*), regardless of the granularity of
interleaving. Both value iteration (one step evaluation) and policy
iteration (full evaluation) are special cases of GPI.

This abstraction is the theoretical backbone of actor-critic methods,
where the actor performs improvement and the critic performs evaluation
simultaneously.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 13 — EXTENSIONS OF THE MDP FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Partially Observable MDP (POMDP)

When the agent cannot directly observe the true state, the full MDP
tuple is extended with an observation space and observation model:

        M = (S, A, O, P, Z, R, γ)

    O    — Observation space
    Z    — Observation model:  Z(o | s', a) = P(oₜ | sₜ, aₜ₋₁)

The agent receives observation oₜ ∈ O, not state sₜ ∈ S.

    BELIEF STATE
        The sufficient statistic for a POMDP is the belief bₜ ∈ Δ(S):
            bₜ(s) = P(sₜ = s | o₁, a₁, ..., oₜ)

        Belief update (Bayes filter):
            bₜ₊₁(s') = η · Z(oₜ₊₁|s', aₜ) · Σₛ P(s'|s, aₜ) bₜ(s)
            where η is a normalisation constant.

    CHALLENGES
        * Belief state is a distribution — infinite-dimensional.
        * Exact POMDP solving is PSPACE-complete.
        * Practical approaches: particle filters, RNNs as belief encoders,
          point-based value iteration (PBVI).


### Constrained MDP (CMDP)

Extends the MDP with one or more constraint functions:

        M = (S, A, P, R, γ, C₁, ..., Cₘ, d₁, ..., dₘ)

    Cᵢ : S × A × S → R    cost function for constraint i
    dᵢ ∈ R                 budget limit for constraint i

Objective:
        max_π  E_π [ Gₜ ]
        s.t.   E_π [ Σₜ γᵗ cₜᵢ ] ≤ dᵢ    for i = 1, ..., m

Used in safe RL, autonomous driving, medical treatment. Solved via:
    * Primal-dual / Lagrangian methods (Lagrangian PPO)
    * Constrained Policy Optimisation (CPO)
    * Interior Point Policy Optimisation (IPO)


### Multi-Agent MDP (Markov Game)

Extends the MDP to n agents:

        M = (S, A₁, ..., Aₙ, P, R₁, ..., Rₙ, γ)

Each agent i has its own action space Aᵢ and reward Rᵢ.
The joint action a = (a₁, ..., aₙ) affects transitions and all rewards.

    COOPERATIVE         All agents share a single reward: Rᵢ = R for all i.
    COMPETITIVE         Zero-sum: Σᵢ Rᵢ = 0 (two-player games).
    MIXED               Partial cooperation and competition.

Solution concepts: Nash equilibrium, correlated equilibrium.
Algorithms: QMIX, MADDPG, MAPPO, CTDE (centralised training, decentralised
execution).


### Goal-Conditioned MDP

The agent must reach different goals g ∈ G on different episodes:

        π_θ(a | s, g)    — policy conditioned on current goal g

Reward is typically:   r(s, a, g) = −||s − g|| or sparse +1 on reaching g.

Used in robotics (reach different target positions) and language-conditioned
control (follow instruction strings).

Hindsight Experience Replay (HER): relabels failed trajectories with the
goal the agent actually reached — turning failures into learning signal.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 14 — MDP SUMMARY & RELATIONSHIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Key Equations at a Glance

    RETURN:
        Gₜ = Σₖ₌₀^∞ γᵏ rₜ₊ₖ₊₁

    BELLMAN EXPECTATION  (V^π):
        V^π(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]

    BELLMAN EXPECTATION  (Q^π):
        Q^π(s,a) = Σₛ' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]

    ADVANTAGE:
        A^π(s,a) = Q^π(s,a) − V^π(s)

    BELLMAN OPTIMALITY   (V*):
        V*(s) = max_a Σₛ' P(s'|s,a) [ R(s,a,s') + γ V*(s') ]

    BELLMAN OPTIMALITY   (Q*):
        Q*(s,a) = Σₛ' P(s'|s,a) [ R(s,a,s') + γ max_a' Q*(s',a') ]

    OPTIMAL POLICY:
        π*(s) = argmax_a Q*(s,a)

    TD ERROR:
        δₜ = rₜ₊₁ + γ V(sₜ₊₁) − V(sₜ)


### MDP Framework Comparison

    +---------------------+------------+------------+------------+--------------+
    | FRAMEWORK           | OBSERV.    | ACTIONS    | AGENTS     | CONSTRAINTS  |
    +---------------------+------------+------------+------------+--------------+
    | Markov Chain        | Full       | None       | 1          | None         |
    | Bandit              | None       | Discrete   | 1          | None         |
    | MDP                 | Full       | Any        | 1          | None         |
    | POMDP               | Partial    | Any        | 1          | None         |
    | CMDP                | Full       | Any        | 1          | Budget       |
    | Markov Game         | Full       | Any        | N          | None         |
    | Dec-POMDP           | Partial    | Any        | N          | None         |
    | BAMDP               | Full       | Any        | 1          | None (prior) |
    +---------------------+------------+------------+------------+--------------+

    BAMDP = Bayes-Adaptive MDP: uncertainty about P and R encoded in state.


### What RL Algorithms Learn

    +------------------------+----------------------------------+---------------------+
    | ALGORITHM FAMILY       | WHAT IS LEARNED                  | WHAT IS USED        |
    +------------------------+----------------------------------+---------------------+
    | Dynamic Programming    | V*, Q*  (exact)                  | Model P, R required |
    | Monte Carlo            | V^π, Q^π  (from episodes)        | No model needed     |
    | Temporal Difference    | V^π, Q^π  (bootstrapped)         | No model needed     |
    | Q-Learning / DQN       | Q*  (off-policy)                 | No model needed     |
    | Policy Gradients       | Policy π_θ directly              | No model needed     |
    | Actor-Critic           | π_θ  and  V^π_w                  | No model needed     |
    | Model-Based RL         | P_θ, R_θ + planning              | Learned model       |
    +------------------------+----------------------------------+---------------------+

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "1. Define a Grid World MDP": {
        "description": "Construct a 4x4 grid world as a complete MDP tuple (S, A, P, R, gamma). "
                       "Includes wall-bouncing transitions, a goal state, and a trap state.",
        "code": """\
import numpy as np

# ── Grid World MDP ────────────────────────────────────────────────
# Layout (4×4):
#   0  1  2  3
#   4  5  6  7
#   8  9 10 11
#  12 13 14 15*   * = goal (+1), 5 = trap (-1)

ROWS, COLS = 4, 4
N_STATES   = ROWS * COLS        # 16 states
N_ACTIONS  = 4                  # 0=up 1=down 2=left 3=right
GOAL_STATE = 15
TRAP_STATE = 5
GAMMA      = 0.99

def rc(s):
    return s // COLS, s % COLS

def state(r, c):
    return r * COLS + c

# Build transition and reward tables
# P[s, a, s'] = probability of reaching s' from s via a
P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
R = np.zeros((N_STATES, N_ACTIONS, N_STATES))

moves = [(-1,0), (1,0), (0,-1), (0,1)]   # up, down, left, right

for s in range(N_STATES):
    if s in (GOAL_STATE, TRAP_STATE):
        P[s, :, s] = 1.0     # absorbing states
        continue
    r, c = rc(s)
    for a, (dr, dc) in enumerate(moves):
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            s2 = state(nr, nc)
        else:
            s2 = s             # bounce off wall
        P[s, a, s2] = 1.0
        if s2 == GOAL_STATE:
            R[s, a, s2] = +1.0
        elif s2 == TRAP_STATE:
            R[s, a, s2] = -1.0
        else:
            R[s, a, s2] = -0.01  # small step penalty

# Immediate reward for (s, a): expected R before knowing s'
R_sa = np.sum(P * R, axis=2)   # shape (16, 4)

print("MDP defined:")
print(f"  States: {N_STATES}, Actions: {N_ACTIONS}, gamma={GAMMA}")
print(f"  Goal: state {GOAL_STATE} | Trap: state {TRAP_STATE}")
print()
print("Transition check (state 14, action right → state 15):")
print(f"  P[14, right, 15] = {P[14, 3, 15]:.1f}  (should be 1.0)")
print(f"  R[14, right, 15] = {R[14, 3, 15]:.1f}  (should be +1.0)")
print()
print("Reward matrix R_sa (first 4 states, all actions):")
actions = ['up', 'down', 'left', 'right']
header  = f"{'state':>6} | " + " | ".join(f"{a:>8}" for a in actions)
print(header)
print("-" * len(header))
for s in range(min(4, N_STATES)):
    row = f"{s:>6} | " + " | ".join(f"{R_sa[s, a]:>8.3f}" for a in range(N_ACTIONS))
    print(row)
""",
    },

    "2. Compute Return from a Trajectory": {
        "description": "Given a sequence of rewards from an episode, compute the discounted "
                       "return Gₜ = Σ γᵏ rₜ₊ₖ₊₁ at every timestep using the recursive formula.",
        "code": """\
import numpy as np

GAMMA = 0.99

# Example trajectory rewards (e.g. agent navigating grid world)
rewards = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, +1.0]

# ── Method 1: Recursive from the end ───────────────────────────
# G_T = 0 (terminal)
# G_t = r_{t+1} + gamma * G_{t+1}
T = len(rewards)
G = np.zeros(T + 1)
for t in reversed(range(T)):
    G[t] = rewards[t] + GAMMA * G[t + 1]

print("Trajectory rewards:", rewards)
print()
print(f"{'t':>4} | {'r_{t+1}':>10} | {'G_t':>12} | {'gamma^t':>10}")
print("-" * 46)
for t in range(T):
    print(f"{t:>4} | {rewards[t]:>10.3f} | {G[t]:>12.6f} | {GAMMA**t:>10.6f}")

print()
print(f"G_0 (total return from start) = {G[0]:.6f}")
print()

# ── Method 2: Direct summation ─────────────────────────────────
G0_direct = sum(GAMMA**k * rewards[k] for k in range(T))
print(f"Cross-check (direct sum)       = {G0_direct:.6f}")
print()

# ── Effect of gamma on return ──────────────────────────────────
print("Effect of discount factor on G_0:")
print(f"{'gamma':>8} | {'G_0':>12}")
print("-" * 24)
for g in [0.5, 0.9, 0.95, 0.99, 0.999]:
    g0 = sum(g**k * rewards[k] for k in range(T))
    print(f"{g:>8.3f} | {g0:>12.6f}")
""",
    },

    "3. Policy Evaluation — Iterative Bellman": {
        "description": "Implement iterative policy evaluation: given a fixed policy π, "
                       "apply the Bellman expectation equation repeatedly until V^π converges. "
                       "Demonstrates convergence rate and Bellman error decay.",
        "code": """\
import numpy as np

# ── Reuse grid world from Operation 1 ─────────────────────────
ROWS, COLS = 4, 4
N_STATES   = ROWS * COLS
N_ACTIONS  = 4
GOAL_STATE = 15
TRAP_STATE = 5
GAMMA      = 0.99

moves = [(-1,0),(1,0),(0,-1),(0,1)]

def build_mdp():
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        if s in (GOAL_STATE, TRAP_STATE):
            P[s, :, s] = 1.0; continue
        r, c = s // COLS, s % COLS
        for a, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            s2 = (nr * COLS + nc) if 0 <= nr < ROWS and 0 <= nc < COLS else s
            P[s, a, s2] = 1.0
            R[s, a, s2] = (+1.0 if s2==GOAL_STATE else -1.0 if s2==TRAP_STATE else -0.01)
    return P, R

P, R = build_mdp()
R_sa = np.sum(P * R, axis=2)           # E[r | s, a]

# ── Define a uniform random policy ─────────────────────────────
pi = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS   # shape (16, 4)

# ── Policy evaluation loop ─────────────────────────────────────
def policy_evaluation(pi, P, R_sa, gamma, theta=1e-8):
    V = np.zeros(N_STATES)
    history = []
    for iteration in range(10_000):
        # Bellman expectation update (vectorised)
        # V_new(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
        V_new = np.einsum('sa,sa->s', pi,
                    R_sa + gamma * np.einsum('san,n->sa', P, V))
        delta = np.max(np.abs(V_new - V))
        history.append(delta)
        V = V_new
        if delta < theta:
            print(f"Converged in {iteration + 1} iterations (delta={delta:.2e})")
            break
    return V, history

V_pi, history = policy_evaluation(pi, P, R_sa, GAMMA)

print()
print("V^pi (uniform random policy) — grid layout:")
grid = V_pi.reshape(ROWS, COLS)
for row in grid:
    print("  " + "  ".join(f"{v:7.3f}" for v in row))

print()
print("Convergence (Bellman error per iteration):")
for i in [0, 1, 2, 5, 10, 20, 50, 100, len(history)-1]:
    if i < len(history):
        print(f"  iter {i:>4}: max|delta| = {history[i]:.6e}")
""",
    },

    "4. Policy Improvement — Greedy Update": {
        "description": "Given a value function V^π, extract the greedy policy. "
                       "Demonstrates the policy improvement theorem: the greedy "
                       "policy is always at least as good as the original.",
        "code": """\
import numpy as np

ROWS, COLS = 4, 4
N_STATES   = ROWS * COLS
N_ACTIONS  = 4
GOAL_STATE = 15
TRAP_STATE = 5
GAMMA      = 0.99
ACTION_LABELS = ['U', 'D', 'L', 'R']

moves = [(-1,0),(1,0),(0,-1),(0,1)]

def build_mdp():
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        if s in (GOAL_STATE, TRAP_STATE):
            P[s, :, s] = 1.0; continue
        r, c = s // COLS, s % COLS
        for a, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            s2 = (nr * COLS + nc) if 0 <= nr < ROWS and 0 <= nc < COLS else s
            P[s, a, s2] = 1.0
            R[s, a, s2] = (+1.0 if s2==GOAL_STATE else -1.0 if s2==TRAP_STATE else -0.01)
    return P, R

P, R = build_mdp()
R_sa = np.sum(P * R, axis=2)

# Assume we have V_pi from some prior evaluation (use random policy here)
pi_uniform = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS

def policy_evaluation(pi, P, R_sa, gamma, theta=1e-8):
    V = np.zeros(N_STATES)
    for _ in range(10_000):
        V_new = np.einsum('sa,sa->s', pi,
                    R_sa + gamma * np.einsum('san,n->sa', P, V))
        if np.max(np.abs(V_new - V)) < theta: break
        V = V_new
    return V

V_pi = policy_evaluation(pi_uniform, P, R_sa, GAMMA)

# ── Greedy policy improvement ──────────────────────────────────
# Q(s, a) = R(s,a) + gamma * Σ_s' P(s'|s,a) V(s')
Q = R_sa + GAMMA * np.einsum('san,n->sa', P, V_pi)    # shape (16, 4)

pi_greedy = np.zeros((N_STATES, N_ACTIONS))
greedy_actions = np.argmax(Q, axis=1)
pi_greedy[np.arange(N_STATES), greedy_actions] = 1.0   # deterministic

# Evaluate the improved policy
V_pi_new = policy_evaluation(pi_greedy, P, R_sa, GAMMA)

print("Policy improvement:")
print(f"  Max V^pi  (uniform)  = {V_pi.max():.4f}")
print(f"  Max V^pi' (improved) = {V_pi_new.max():.4f}")
print(f"  Improvement (all states >= ): {np.all(V_pi_new >= V_pi - 1e-9)}")

print()
print("Greedy policy — grid layout (U/D/L/R):")
policy_grid = np.array([ACTION_LABELS[a] for a in greedy_actions]).reshape(ROWS, COLS)
for row in policy_grid:
    print("  " + "  ".join(row))

print()
print("Q-values at state 14 (adjacent to goal 15):")
for a, label in enumerate(ACTION_LABELS):
    print(f"  Q(14, {label}) = {Q[14, a]:.4f}")
""",
    },

    "5. Policy Iteration — Full Algorithm": {
        "description": "Implements the complete policy iteration algorithm: repeatedly alternate "
                       "policy evaluation and greedy policy improvement until the policy stabilises. "
                       "Finds the exact optimal policy for the known MDP.",
        "code": """\
import numpy as np

ROWS, COLS = 4, 4
N_STATES   = ROWS * COLS
N_ACTIONS  = 4
GOAL_STATE = 15
TRAP_STATE = 5
GAMMA      = 0.99
ACTION_LABELS = ['U', 'D', 'L', 'R']

moves = [(-1,0),(1,0),(0,-1),(0,1)]

def build_mdp():
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        if s in (GOAL_STATE, TRAP_STATE):
            P[s, :, s] = 1.0; continue
        r, c = s // COLS, s % COLS
        for a, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            s2 = (nr * COLS + nc) if 0 <= nr < ROWS and 0 <= nc < COLS else s
            P[s, a, s2] = 1.0
            R[s, a, s2] = (+1.0 if s2==GOAL_STATE else -1.0 if s2==TRAP_STATE else -0.01)
    return P, R

P, R = build_mdp()
R_sa = np.sum(P * R, axis=2)

def evaluate(pi, P, R_sa, gamma, theta=1e-9):
    V = np.zeros(N_STATES)
    for _ in range(50_000):
        V_new = np.einsum('sa,sa->s', pi,
                    R_sa + gamma * np.einsum('san,n->sa', P, V))
        if np.max(np.abs(V_new - V)) < theta: break
        V = V_new
    return V

def improve(V, P, R_sa, gamma):
    Q = R_sa + gamma * np.einsum('san,n->sa', P, V)
    pi = np.zeros((N_STATES, N_ACTIONS))
    pi[np.arange(N_STATES), np.argmax(Q, axis=1)] = 1.0
    return pi

# ── Policy Iteration ───────────────────────────────────────────
pi = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS   # random start

print("Policy Iteration:")
print(f"{'Iter':>5} | {'Max V(s)':>12} | {'Policy changed':>15}")
print("-" * 40)

for i in range(100):
    V = evaluate(pi, P, R_sa, GAMMA)
    pi_new = improve(V, P, R_sa, GAMMA)
    changed = not np.allclose(pi, pi_new)
    print(f"{i+1:>5} | {V.max():>12.6f} | {str(changed):>15}")
    pi = pi_new
    if not changed:
        print()
        print(f"Converged after {i+1} iteration(s).")
        break

print()
print("Optimal value function V* — grid layout:")
for row in V.reshape(ROWS, COLS):
    print("  " + "  ".join(f"{v:7.4f}" for v in row))

print()
print("Optimal policy pi* — grid layout:")
actions = np.argmax(pi, axis=1)
for row in np.array([ACTION_LABELS[a] for a in actions]).reshape(ROWS, COLS):
    print("  " + "  ".join(row))
""",
    },

    "6. Value Iteration — Bellman Optimality": {
        "description": "Implements value iteration: apply the Bellman optimality equation "
                       "until convergence. Often faster in practice than policy iteration. "
                       "Tracks convergence and compares to policy iteration's result.",
        "code": """\
import numpy as np

ROWS, COLS = 4, 4
N_STATES   = ROWS * COLS
N_ACTIONS  = 4
GOAL_STATE = 15
TRAP_STATE = 5
GAMMA      = 0.99
ACTION_LABELS = ['U', 'D', 'L', 'R']

moves = [(-1,0),(1,0),(0,-1),(0,1)]

def build_mdp():
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        if s in (GOAL_STATE, TRAP_STATE):
            P[s, :, s] = 1.0; continue
        r, c = s // COLS, s % COLS
        for a, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            s2 = (nr * COLS + nc) if 0 <= nr < ROWS and 0 <= nc < COLS else s
            P[s, a, s2] = 1.0
            R[s, a, s2] = (+1.0 if s2==GOAL_STATE else -1.0 if s2==TRAP_STATE else -0.01)
    return P, R

P, R = build_mdp()
R_sa = np.sum(P * R, axis=2)

# ── Value Iteration ────────────────────────────────────────────
# V_{k+1}(s) = max_a [R(s,a) + gamma * Σ_s' P(s'|s,a) V_k(s')]
V = np.zeros(N_STATES)
theta = 1e-9
history = []

for k in range(10_000):
    Q = R_sa + GAMMA * np.einsum('san,n->sa', P, V)   # (S, A)
    V_new = np.max(Q, axis=1)                          # (S,)
    delta = np.max(np.abs(V_new - V))
    history.append(delta)
    V = V_new
    if delta < theta:
        print(f"Value iteration converged in {k + 1} sweeps (delta={delta:.2e})")
        break

# Extract optimal policy
Q_star = R_sa + GAMMA * np.einsum('san,n->sa', P, V)
pi_star = np.argmax(Q_star, axis=1)

print()
print("Optimal V* — grid layout:")
for row in V.reshape(ROWS, COLS):
    print("  " + "  ".join(f"{v:7.4f}" for v in row))

print()
print("Optimal policy pi* — grid layout:")
for row in np.array([ACTION_LABELS[a] for a in pi_star]).reshape(ROWS, COLS):
    print("  " + "  ".join(row))

print()
print("Convergence (Bellman error per iteration):")
checkpoints = [0, 1, 2, 5, 10, 20, 50, 100, len(history)-1]
for i in checkpoints:
    if i < len(history):
        print(f"  sweep {i:>4}: max delta = {history[i]:.6e}")

print()
print("Theoretical convergence bound:")
print(f"  After k sweeps: ||V_k - V*|| <= gamma^k * ||V_0 - V*||")
print(f"  gamma = {GAMMA}, gamma^{len(history)} = {GAMMA**len(history):.2e}")
""",
    },

    "7. Verify Bellman Equations Numerically": {
        "description": "Compute V^pi analytically (matrix inverse) and iteratively, "
                       "then verify the Bellman expectation and optimality equations "
                       "hold at convergence. A sanity check for any RL implementation.",
        "code": """\
import numpy as np

ROWS, COLS = 4, 4
N_STATES   = ROWS * COLS
N_ACTIONS  = 4
GOAL_STATE = 15
TRAP_STATE = 5
GAMMA      = 0.99

moves = [(-1,0),(1,0),(0,-1),(0,1)]

def build_mdp():
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        if s in (GOAL_STATE, TRAP_STATE):
            P[s, :, s] = 1.0; continue
        r, c = s // COLS, s % COLS
        for a, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            s2 = (nr * COLS + nc) if 0 <= nr < ROWS and 0 <= nc < COLS else s
            P[s, a, s2] = 1.0
            R[s, a, s2] = (+1.0 if s2==GOAL_STATE else -1.0 if s2==TRAP_STATE else -0.01)
    return P, R

P, R = build_mdp()
R_sa = np.sum(P * R, axis=2)

pi = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS    # uniform policy

# 1. Analytical solution: V = (I - gamma * P_pi)^{-1} R_pi
P_pi = np.einsum('sa,san->sn', pi, P)              # (S, S) transition under pi
R_pi = np.einsum('sa,sa->s', pi, R_sa)             # (S,) expected reward under pi
V_exact = np.linalg.solve(np.eye(N_STATES) - GAMMA * P_pi, R_pi)

# 2. Iterative solution
V_iter = np.zeros(N_STATES)
for _ in range(100_000):
    V_new = np.einsum('sa,sa->s', pi,
                R_sa + GAMMA * np.einsum('san,n->sa', P, V_iter))
    if np.max(np.abs(V_new - V_iter)) < 1e-12: break
    V_iter = V_new

print("Verification of Bellman Expectation Equation")
print("=" * 52)
print(f"Max |V_exact - V_iter| = {np.max(np.abs(V_exact - V_iter)):.2e}")
print()

# 3. Verify Bellman equation holds: V = R_pi + gamma * P_pi @ V
lhs = V_exact
rhs = R_pi + GAMMA * P_pi @ V_exact
bellman_error = np.max(np.abs(lhs - rhs))
print(f"Bellman expectation residual ||V - (R + gamma P_pi V)||_inf = {bellman_error:.2e}")
print(f"  (should be ~0 for exact solution)")
print()

# 4. Verify Bellman optimality
# Value iteration result
V_vi = np.zeros(N_STATES)
for _ in range(100_000):
    Q = R_sa + GAMMA * np.einsum('san,n->sa', P, V_vi)
    V_new = np.max(Q, axis=1)
    if np.max(np.abs(V_new - V_vi)) < 1e-12: break
    V_vi = V_new

Q_star = R_sa + GAMMA * np.einsum('san,n->sa', P, V_vi)
V_reconstructed = np.max(Q_star, axis=1)
opt_residual = np.max(np.abs(V_vi - V_reconstructed))

print(f"Bellman optimality residual ||V* - max_a Q*||_inf = {opt_residual:.2e}")
print(f"  (should be ~0 for optimal value function)")
print()

# 5. TD error should be 0 at convergence
td_errors = []
for s in range(N_STATES):
    for a in range(N_ACTIONS):
        for s2 in range(N_STATES):
            if P[s, a, s2] > 0:
                td = R[s, a, s2] + GAMMA * V_exact[s2] - V_exact[s]
                td_errors.append(abs(td * P[s, a, s2]))  # weighted by probability

print(f"Expected |TD error| under uniform policy = {sum(td_errors)/len(td_errors):.2e}")
print("  (non-zero because V is averaged over actions, not per-action)")
""",
    },

    "8. POMDP Belief State Update": {
        "description": "Implement the Bayes filter belief state update for a simple POMDP. "
                       "Shows how the agent maintains a probability distribution over "
                       "hidden states given observations, and how it converges toward certainty.",
        "code": """\
import numpy as np

# ── Simple 3-state POMDP (Tiger Problem variant) ───────────────
# States:    0 = Tiger-Left, 1 = Tiger-Right, 2 = Empty
# Actions:   0 = Listen, 1 = Open-Left, 2 = Open-Right
# Obs:       0 = Hear-Left, 1 = Hear-Right, 2 = Silence

N_STATES  = 3
N_ACTIONS = 3
N_OBS     = 3

# Transition model P(s' | s, a)
P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
# Listen: state unchanged
P[:, 0, :] = np.eye(N_STATES)
# Open Left / Open Right: reset uniformly (new trial)
P[:, 1, :] = 1.0 / N_STATES
P[:, 2, :] = 1.0 / N_STATES

# Observation model Z(o | s', a)
Z = np.zeros((N_OBS, N_STATES, N_ACTIONS))
# After Listen: 85% correct + 15% confused
Z[0, 0, 0] = 0.85; Z[1, 0, 0] = 0.15   # Tiger-Left, heard correctly/wrong
Z[1, 1, 0] = 0.85; Z[0, 1, 0] = 0.15   # Tiger-Right
Z[2, 2, 0] = 1.0                         # Empty -> Silence always
# After opening: uniformly observe anything
Z[:, :, 1] = 1.0 / N_OBS
Z[:, :, 2] = 1.0 / N_OBS

STATE_NAMES  = ['Tiger-Left', 'Tiger-Right', 'Empty']
ACTION_NAMES = ['Listen',     'Open-Left',   'Open-Right']
OBS_NAMES    = ['Hear-Left',  'Hear-Right',  'Silence']

def belief_update(b, action, obs):
    # Bayes filter belief update:
    #   b'(s') = eta * Z(o|s',a) * Σ_s P(s'|s,a) b(s)
    b_pred = np.zeros(N_STATES)
    for s_prime in range(N_STATES):
        b_pred[s_prime] = sum(P[s, action, s_prime] * b[s] for s in range(N_STATES))
    b_new = np.array([Z[obs, s_prime, action] * b_pred[s_prime]
                      for s_prime in range(N_STATES)])
    b_new /= b_new.sum()   # normalise (eta)
    return b_new

# ── Simulate a sequence of observations ───────────────────────
np.random.seed(42)
b = np.ones(N_STATES) / N_STATES    # uniform prior

# Sequence: Listen 5 times, receiving 'Hear-Left' 4 times
actions  = [0, 0, 0, 0, 0]         # all Listen
observations = [0, 0, 0, 1, 0]     # mostly Hear-Left

print("POMDP Belief State Update (Tiger Problem variant)")
print("=" * 58)
print()
print(f"Initial belief: {dict(zip(STATE_NAMES, b.round(3)))}")
print()
print(f"{'Step':>4} | {'Action':>12} | {'Obs':>12} | {'B[0] TL':>8} | {'B[1] TR':>8} | {'B[2] Em':>8}")
print("-" * 62)

for t, (a, o) in enumerate(zip(actions, observations)):
    b = belief_update(b, a, o)
    print(f"{t+1:>4} | {ACTION_NAMES[a]:>12} | {OBS_NAMES[o]:>12} | "
          f"{b[0]:>8.4f} | {b[1]:>8.4f} | {b[2]:>8.4f}")

print()
print(f"Final belief: {dict(zip(STATE_NAMES, b.round(4)))}")
dominant = STATE_NAMES[np.argmax(b)]
print(f"Agent is most confident the tiger is: {dominant}")

print()
print("Key insight: each 'Hear-Left' observation shifts probability mass")
print("toward Tiger-Left via Bayes' theorem. Four observations in a row")
print("give the agent high confidence despite initial uncertainty.")
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