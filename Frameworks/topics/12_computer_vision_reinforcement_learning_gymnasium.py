"""
Reinforcement Learning & Gymnasium — Learning Through Interaction
=================================================================

Reinforcement Learning (RL) is the third paradigm of machine learning,
sitting alongside supervised and unsupervised learning. Where supervised
learning learns from labelled examples and unsupervised learning finds
structure in unlabelled data, reinforcement learning learns by doing —
an agent interacts with an environment, receives reward signals, and
discovers through experience what actions lead to good outcomes.

The field draws from psychology (Skinner's operant conditioning), control
theory (optimal control, dynamic programming), neuroscience (dopamine
reward signals in the brain), game theory, and statistics. The fundamental
insight — that intelligence can emerge from reward maximisation through
interaction — is one of the most profound ideas in all of science.

RL has produced some of the most spectacular AI achievements in history:
    1997:  TD-Gammon plays backgammon at world-champion level (Tesauro)
    2013:  DQN learns Atari games from pixels (DeepMind)
    2016:  AlphaGo defeats world champion Lee Sedol
    2017:  AlphaZero masters Chess, Go, Shogi from self-play alone
    2019:  AlphaStar reaches Grandmaster at StarCraft II
    2020:  OpenAI Five wins The International (Dota 2)
    2022:  ChatGPT uses RLHF to align language models to human preferences
    2023:  RT-2 uses RL for generalised robot manipulation
    2024:  AlphaProof solves International Math Olympiad problems

Gymnasium (formerly OpenAI Gym, forked and maintained by the Farama
Foundation since 2022) is the standard interface for RL environments.
Every RL algorithm in the literature is evaluated on Gymnasium environments.
It provides 50+ built-in environments spanning classic control, box2D
physics, Atari games, MuJoCo robotics, and toy text problems.

This module covers the complete RL stack: the mathematical framework
(MDP formalism, Bellman equations, value functions), the taxonomy of
algorithms (model-free vs model-based, on-policy vs off-policy, value-
based vs policy-based), the Gymnasium API and environment design, tabular
methods (Q-learning, SARSA), deep RL foundations (DQN and its variants),
policy gradient methods (REINFORCE, Actor-Critic), and modern algorithms
(PPO), plus practical considerations for training stability.

"""

import textwrap
import re

TOPIC_NAME   = "Reinforcement Learning & Gymnasium — Learning Through Interaction"
DISPLAY_NAME = "12 · Reinforcement Learning & Gymnasium"
ICON         = "🎮"
SUBTITLE     = "From Markov Decision Processes to Deep Policy Optimisation"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE REINFORCEMENT LEARNING PROBLEM

### The Agent-Environment Interface

    Reinforcement learning is formally defined by the interaction loop between
    an AGENT and an ENVIRONMENT:

        At each discrete time step t:
            1. Agent observes state  sₜ ∈ S
            2. Agent selects action  aₜ ∈ A(sₜ)  according to policy π
            3. Environment transitions to new state  sₜ₊₁ ~ P(·|sₜ, aₜ)
            4. Environment emits reward  rₜ₊₁ = R(sₜ, aₜ, sₜ₊₁)
            5. Repeat from step 1

    The agent's goal is to find a POLICY π (a mapping from states to actions)
    that maximises the expected cumulative future reward.

    This is fundamentally different from supervised learning:
        Supervised: labelled (input, output) pairs, no interaction
        RL:         only reward signals, must discover good actions by trying

    The exploration-exploitation dilemma:
        To maximise reward, the agent should EXPLOIT its best known actions.
        But to discover better actions, it must EXPLORE unknown options.
        This tension is central to all RL algorithms:
            Too much exploitation: gets stuck in local optima
            Too much exploration:  wastes time on known-bad actions

### The Markov Decision Process (MDP)

    The MDP is the mathematical framework for RL. It is a tuple (S, A, P, R, γ):

        S:   State space (can be discrete or continuous)
        A:   Action space (can be discrete or continuous)
        P:   Transition function: P(s'|s, a) = probability of reaching s'
             from s by taking action a
        R:   Reward function: R(s, a, s') = immediate reward
        γ:   Discount factor ∈ [0, 1]: how much to value future rewards

    The Markov Property:
        The future depends only on the CURRENT state, not on the history.
        P(sₜ₊₁|sₜ, aₜ, sₜ₋₁, aₜ₋₁, ...) = P(sₜ₊₁|sₜ, aₜ)

        This is a modelling assumption — the state must encode all
        relevant information. In practice, we often have partial
        observability (POMDP), handled by using history or recurrent networks.

    Discount factor γ:
        γ = 0:    agent is completely myopic — only cares about immediate reward
        γ = 0.99: agent values rewards ~100 steps in the future nearly as much
        γ = 1.0:  undiscounted — only valid for episodic tasks that always end

        γ < 1 ensures mathematical convergence of infinite-horizon returns.
        Also models uncertainty about the future (the further away, the less certain).

### Return: The Total Discounted Reward

    The RETURN Gₜ is the total discounted reward from time step t:

        Gₜ = rₜ₊₁ + γ·rₜ₊₂ + γ²·rₜ₊₃ + ... = Σₖ₌₀^∞ γᵏ·rₜ₊ₖ₊₁

    The return can be written recursively (fundamental for RL algorithms):

        Gₜ = rₜ₊₁ + γ·Gₜ₊₁

    This Bellman decomposition — current reward + discounted future return —
    is the foundation of every dynamic programming algorithm in RL.

    The agent's goal: find π* = argmax_π  Eπ[G₀]

### Episodic vs Continuing Tasks

    Episodic tasks:
        Have a natural terminal state (episode ends).
        The agent restarts from an initial state after each episode.
        Examples: games (Chess, Atari), navigation tasks, robotic manipulation.
        Return is a finite sum.

    Continuing tasks:
        No terminal state — the interaction goes on indefinitely.
        Examples: process control, resource management, trading algorithms.
        Must use γ < 1 for finite return.

    In Gymnasium:
        Episodic: env.reset() starts a new episode; done=True ends it.
        The done flag can be True for:
            - Terminal state reached (natural end)
            - Truncation (max time steps exceeded — not a true terminal state)


##### PART 2 — VALUE FUNCTIONS AND THE BELLMAN EQUATIONS

### State Value Function V(s)

    The state value function V^π(s) is the expected return when starting
    in state s and following policy π:

        V^π(s) = Eπ[Gₜ | sₜ = s]
               = Eπ[rₜ₊₁ + γ·Gₜ₊₁ | sₜ = s]

    The Bellman Expectation Equation for V^π:

        V^π(s) = Σₐ π(a|s) · Σₛ' P(s'|s,a) · [R(s,a,s') + γ·V^π(s')]

    Reading this equation:
        For each action a that policy π might take (weighted by π(a|s)):
            For each possible next state s' (weighted by transition probability):
                Take the immediate reward plus discounted future value

    The OPTIMAL value function V*(s) = max_π V^π(s) satisfies:

        V*(s) = max_a  Σₛ' P(s'|s,a) · [R(s,a,s') + γ·V*(s')]

    This is the Bellman Optimality Equation. It says: the value of the best
    state equals the reward from the best action plus discounted best future value.

### Action-Value Function Q(s, a)

    The action-value (Q) function is the expected return starting from state s,
    taking action a, then following policy π:

        Q^π(s, a) = Eπ[Gₜ | sₜ = s, aₜ = a]

    The Bellman Expectation Equation for Q^π:

        Q^π(s, a) = Σₛ' P(s'|s,a) · [R(s,a,s') + γ · Σₐ' π(a'|s')·Q^π(s',a')]

    The OPTIMAL Q function satisfies:

        Q*(s, a) = Σₛ' P(s'|s,a) · [R(s,a,s') + γ · max_a' Q*(s',a')]

    CRITICAL INSIGHT: if we know Q*, the optimal policy is trivial:
        π*(s) = argmax_a Q*(s, a)
        "In state s, take the action with the highest Q-value."

    Relationship between V and Q:
        V^π(s) = Σₐ π(a|s) · Q^π(s, a)
        Q^π(s, a) = Σₛ' P(s'|s,a) · [R(s,a,s') + γ·V^π(s')]
        V*(s) = max_a Q*(s, a)

### The Advantage Function A(s, a)

    The advantage function measures how much better action a is compared
    to the average action under the current policy:

        A^π(s, a) = Q^π(s, a) − V^π(s)

    A^π(s, a) > 0:  action a is better than average — policy should increase π(a|s)
    A^π(s, a) < 0:  action a is worse than average — policy should decrease π(a|s)
    A^π(s, a) = 0:  action a is exactly average

    The advantage function is central to modern policy gradient algorithms
    (PPO, A3C, GAE) because it has lower variance than raw Q-values.

### Temporal Difference Learning: The Bootstrap Principle

    Dynamic programming requires knowing P and R — the model.
    Monte Carlo methods average complete episode returns — high variance.
    TD learning combines both: bootstrap from incomplete trajectories.

    TD(0) update rule:
        V(sₜ) ← V(sₜ) + α · [rₜ₊₁ + γ·V(sₜ₊₁) − V(sₜ)]

    The TD target:     δₜ = rₜ₊₁ + γ·V(sₜ₊₁)
    The TD error:      δₜ − V(sₜ) = rₜ₊₁ + γ·V(sₜ₊₁) − V(sₜ)

    The TD error δₜ is the "surprise" signal:
        δₜ > 0:  the outcome was better than expected → increase V(sₜ)
        δₜ < 0:  the outcome was worse than expected → decrease V(sₜ)

    This mirrors dopamine signals in the brain (Schultz et al., 1997):
    dopamine neurons fire in response to prediction errors, not rewards themselves.
    This is considered strong evidence that the brain implements TD learning.

    Bias-variance trade-off in TD:
        TD(0):    low variance, high bias (bootstraps from estimated values)
        MC:       zero bias, high variance (uses full actual returns)
        TD(λ):    interpolates between TD(0) and MC via λ ∈ [0,1]


##### PART 3 — TABULAR METHODS: Q-LEARNING AND SARSA

### Q-Learning: Off-Policy TD Control

    Q-learning (Watkins, 1989) is the classic off-policy algorithm that
    directly learns Q* without needing a model of the environment.

    Update rule:
        Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α · [rₜ₊₁ + γ·max_a' Q(sₜ₊₁, a') − Q(sₜ, aₜ)]

    The TARGET is: rₜ₊₁ + γ·max_a' Q(sₜ₊₁, a')
    This uses the GREEDY action from the next state — regardless of what
    the behaviour policy actually did. This is why Q-learning is OFF-POLICY.

    Off-policy means: the policy that GENERATES data (behaviour policy)
    can be different from the policy being IMPROVED (target policy).
    This enables:
        - Learning from human demonstrations
        - Experience replay (reusing old data)
        - Parallel data collection with different exploration strategies

    Exploration strategy — ε-greedy:
        With probability ε:   take a random action (explore)
        With probability 1-ε: take argmax_a Q(s,a) (exploit)

        ε-decay schedule (common practice):
            Start: ε = 1.0 (fully random = full exploration)
            End:   ε = 0.01 (mostly greedy)
            Decay: ε = max(ε_min, ε_start × ε_decay^episode)

    Convergence guarantee:
        Q-learning converges to Q* if:
        1. All state-action pairs are visited infinitely often
        2. Learning rate satisfies Robbins-Monro conditions:
           Σα = ∞ (enough total learning) and Σα² < ∞ (eventually stabilises)
        3. The environment is stationary (fixed P and R)

### SARSA: On-Policy TD Control

    SARSA (State-Action-Reward-State-Action) is the ON-POLICY counterpart:

    Update rule:
        Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α · [rₜ₊₁ + γ·Q(sₜ₊₁, aₜ₊₁) − Q(sₜ, aₜ)]

    SARSA uses the ACTUAL next action aₜ₊₁ (sampled from the current policy)
    rather than the greedy action. This makes it on-policy.

    On-policy means: the policy used to generate data IS the policy being improved.

    Q-learning vs SARSA:
        Q-learning:  learns the optimal Q regardless of exploration. Can be
                     overoptimistic in dangerous environments (ignores what the
                     exploration policy might do at sₜ₊₁).
        SARSA:       learns the Q of the CURRENT policy. More conservative —
                     if the policy might take risky actions, SARSA accounts for this.

        Example — Cliff Walking:
            A path along a cliff exists: safe but long.
            A faster path exists right along the cliff edge.
            Q-learning: learns the optimal (cliff-edge) path because it
                        assumes greedy action selection at next state.
            SARSA:      learns the safe (longer) path because with ε-greedy
                        exploration, the agent might fall off the cliff.

### Hyperparameters in Tabular RL

    Learning rate α:
        Too large: oscillates, never converges
        Too small: converges very slowly
        Typical: 0.1–0.5 for tabular; 3e-4 for neural network (Adam)

    Discount factor γ:
        Higher γ: agent is more far-sighted (values delayed rewards)
        Lower γ: agent is more myopic (only near-term rewards matter)
        Typical: 0.95–0.999 for most environments

    Exploration ε:
        Starts high (1.0), decays over training
        Final value depends on task: 0.01 for most, 0.05 for stochastic envs


##### PART 4 — DEEP Q-NETWORKS AND VALUE-BASED DEEP RL

### Why Function Approximation?

    Tabular methods store Q(s, a) in a table. This requires:
        - Discrete state and action spaces
        - Small enough to enumerate all states
        - Visiting each state many times

    Real-world problems have:
        - Continuous state spaces (CartPole: 4D continuous)
        - High-dimensional observations (Atari: 84×84×4 pixels = 28,224 dims)
        - State spaces far too large to tabulate (Go: ~10^170 states)

    Function approximation: parameterise Q as a neural network:
        Q(s, a; θ) ≈ Q*(s, a)

    The neural network takes the state as input and outputs Q-values
    for all actions simultaneously. One forward pass → all Q(s,a) values.

### DQN: Deep Q-Network (Mnih et al., 2013/2015)

    DQN was the first algorithm to successfully combine deep learning and RL,
    learning to play 49 Atari games at superhuman level from raw pixels.

    DQN's two key innovations that made training stable:

    1. EXPERIENCE REPLAY:
        Problem: consecutive experiences (s_t, a_t, r_t, s_{t+1}) are highly
        correlated — violates the i.i.d. assumption that neural network
        gradient descent requires.
        Solution: store experiences in a replay buffer D of fixed size N.
        During training, sample RANDOM mini-batches from D.
        Benefits:
            - Breaks temporal correlations → stable gradients
            - Each experience used multiple times → data efficiency
            - Old experiences can influence recent learning

    2. TARGET NETWORK:
        Problem: the target rₜ₊₁ + γ·max_a' Q(sₜ₊₁,a';θ) uses the same
        network θ being updated. This creates a moving target — like a dog
        chasing its own tail — leading to oscillations and divergence.
        Solution: maintain TWO networks:
            Online network: θ  — updated every step via gradient descent
            Target network: θ⁻ — updated every C steps by copying θ→θ⁻
        The target uses θ⁻ (fixed for C steps):
            yₜ = rₜ₊₁ + γ·max_a' Q(sₜ₊₁, a'; θ⁻)
        Benefits:
            - Target is stable for C steps → stable gradients
            - Prevents divergence and oscillation

    DQN loss function (Huber loss for robustness to outliers):
        L(θ) = E[(yₜ − Q(sₜ, aₜ; θ))²]
        or Huber: L(θ) = Σ ℓδ(yₜ − Q(sₜ, aₜ; θ))
        where ℓδ(e) = e²/2 if |e|≤δ, else δ(|e|−δ/2)

    DQN training loop:
        for each step:
            1. Observe sₜ, select aₜ via ε-greedy
            2. Execute aₜ, observe rₜ₊₁, sₜ₊₁
            3. Store (sₜ, aₜ, rₜ₊₁, sₜ₊₁) in replay buffer D
            4. If len(D) > batch_size:
               a. Sample random batch from D
               b. Compute targets yₜ using target network θ⁻
               c. Compute loss L(θ)
               d. Update θ via gradient descent
            5. Every C steps: θ⁻ ← θ  (copy to target)
            6. Decay ε

### DQN Variants (The "Rainbow" of Improvements)

    Double DQN (van Hasselt et al., 2016):
        Problem: DQN overestimates Q-values because max_a Q(s',a';θ⁻) always
        picks the highest value, even for noisy estimates.
        Solution: use online network θ to SELECT action, target network θ⁻
        to EVALUATE it:
            a* = argmax_a' Q(s', a'; θ)          ← selection by online net
            y  = r + γ · Q(s', a*; θ⁻)           ← evaluation by target net
        Reduces overestimation bias significantly.

    Dueling DQN (Wang et al., 2016):
        Separates the Q-function into two streams:
            Q(s, a; θ) = V(s; θᵥ) + A(s, a; θₐ) − mean_a' A(s, a'; θₐ)
            (subtract mean to ensure identifiability)
        Benefits: V(s) is updated for every action, not just the one taken.
        Faster learning when the choice of action doesn't matter much.

    Prioritised Experience Replay (Schaul et al., 2015):
        Sample experiences with probability ∝ |TD error|^α
        Experiences with higher TD error are more surprising → more informative.
        Corrects sampling bias with importance sampling weights.

    N-step returns (reduces bias):
        Instead of 1-step TD target:  y = r₁ + γ·V(s₂)
        Use n-step return:            y = r₁ + γr₂ + γ²r₃ + ... + γⁿV(sₙ₊₁)
        Reduces bias at cost of higher variance.

    Noisy Networks (replaces ε-greedy):
        Add learnable noise to network weights → stochastic exploration.
        The noise adapts: more uncertainty → more exploration automatically.


##### PART 5 — POLICY GRADIENT METHODS

### The Policy Gradient Theorem

    Value-based methods (Q-learning, DQN) learn Q*(s,a) and derive the policy
    implicitly. Policy gradient methods DIRECTLY parameterise the policy:

        π(a|s; θ) = probability of action a in state s under parameters θ

    The objective is to maximise expected return:
        J(θ) = Eπ_θ[G₀] = Eπ_θ[Σₜ γᵗ rₜ₊₁]

    The Policy Gradient Theorem (Sutton et al., 2000):
        ∇_θ J(θ) = Eπ_θ [ Σₜ ∇_θ log π(aₜ|sₜ; θ) · Qπ(sₜ, aₜ) ]

    Reading this: the gradient of the objective = expected value of
    (the gradient of log-probability of the taken action) × (how good that action was)

    Intuition:
        If action a led to high return: increase π(a|s) (positive push)
        If action a led to low return:  decrease π(a|s) (negative push)
        The magnitude of the push scales with how good/bad the outcome was.

### REINFORCE: Monte Carlo Policy Gradient

    REINFORCE (Williams, 1992) is the simplest policy gradient algorithm.
    Use the actual return Gₜ as an estimate of Qπ(sₜ, aₜ):

        θ ← θ + α · Gₜ · ∇_θ log π(aₜ|sₜ; θ)

    Algorithm:
        for each episode:
            1. Collect trajectory: s₀,a₀,r₁, s₁,a₁,r₂, ..., sT
            2. For each step t, compute Gₜ = Σₖ γᵏ⁻ᵗ rₖ
            3. Update: θ ← θ + α·Gₜ·∇_θ log π(aₜ|sₜ;θ)

    Problems with REINFORCE:
        HIGH VARIANCE: Gₜ is a noisy estimate (a single trajectory).
        SAMPLE INEFFICIENCY: each episode used once, then discarded.
        SLOW LEARNING: especially for sparse rewards.

    Variance reduction — baselines:
        We can subtract a baseline b(s) from Gₜ without biasing the gradient:
        ∇_θ J(θ) = Eπ [ (Gₜ - b(sₜ)) · ∇_θ log π(aₜ|sₜ; θ) ]

        The best baseline: b(s) = V^π(s) (the value function).
        Then Gₜ - V^π(sₜ) ≈ A^π(sₜ, aₜ) (the advantage function).
        This is the basis of Actor-Critic methods.

### Actor-Critic: Combining Policy and Value Learning

    Actor-Critic methods maintain two networks:
        ACTOR:   π(a|s; θ) — the policy (takes actions)
        CRITIC:  V(s; w)   — estimates V^π(s) (evaluates actions)

    The actor updates using the advantage estimated by the critic:
        δₜ = rₜ₊₁ + γ·V(sₜ₊₁; w) − V(sₜ; w)    (TD error = advantage estimate)
        θ ← θ + α · δₜ · ∇_θ log π(aₜ|sₜ; θ)   (policy gradient update)
        w ← w + β · δₜ · ∇_w V(sₜ; w)           (value function update)

    Actor-Critic fixes REINFORCE's problems:
        Low variance: uses TD error δₜ instead of full return Gₜ
        Online updates: can learn after every step (not just after episodes)

    A2C (Advantage Actor-Critic):
        Synchronous, deterministic version. Multiple workers collect
        experiences in parallel, then all update the shared network.

    A3C (Asynchronous Advantage Actor-Critic, Mnih et al., 2016):
        Multiple independent agents run in parallel, each exploring different
        parts of the state space. Asynchronous gradient updates to shared network.
        The parallelism breaks correlations without a replay buffer.

### Generalised Advantage Estimation (GAE)

    GAE (Schulman et al., 2016) provides a way to trade off bias vs variance
    in the advantage estimate using a hyperparameter λ:

        Aᴳᴬᴱ(t) = Σₗ (γλ)ˡ δₜ₊ₗ

    where δₜ = rₜ₊₁ + γV(sₜ₊₁) − V(sₜ) is the TD error.

    λ = 0:  Aᴳᴬᴱ = δₜ = TD(0) advantage (low variance, high bias)
    λ = 1:  Aᴳᴬᴱ = Gₜ − V(sₜ) = Monte Carlo advantage (high variance, low bias)
    λ ≈ 0.95: the sweet spot for most tasks


##### PART 6 — PPO: PROXIMAL POLICY OPTIMISATION

### The Policy Update Problem

    Policy gradient methods update θ by gradient ascent on J(θ).
    The challenge: how large should the update step be?

    Too small:  slow learning, wasted samples
    Too large:  catastrophic policy degradation — a bad update can destroy
                a previously good policy, and recovery is slow

    This is worse than in supervised learning because:
        In supervised learning: a bad step just reduces accuracy temporarily.
        In RL: a bad policy collects bad data → learns from bad data → 
               worse policy → worse data → catastrophic collapse.

### Trust Region Policy Optimisation (TRPO, Schulman et al., 2015)

    TRPO constrains the policy update to stay within a "trust region":

        max_θ  Eπ_old [ (π_new(a|s) / π_old(a|s)) · Â(s,a) ]
        subject to  KL(π_old || π_new) ≤ δ

    The ratio π_new / π_old is the IMPORTANCE RATIO — it allows using
    data from the old policy to update the new policy (sample efficiency).

    TRPO guarantees monotonic policy improvement but requires computing
    second-order derivatives (the Fisher information matrix) → computationally expensive.

### PPO: Practical Approximation of TRPO

    PPO (Schulman et al., 2017) achieves TRPO's stability with first-order methods.
    Two variants:

    PPO-KL (adaptive KL penalty):
        L(θ) = E[r_t(θ)·Â_t] − β·KL[π_old || π_new]
        β is adapted: if KL too large, increase β; if too small, decrease β.

    PPO-Clip (the standard, widely used variant):
        r_t(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)   (probability ratio)

        L^CLIP(θ) = E[ min(r_t·Â_t,  clip(r_t, 1-ε, 1+ε)·Â_t) ]

    The clipping mechanism:
        If Â > 0 (good action): r_t is clipped at 1+ε → can't increase
                                probability too much
        If Â < 0 (bad action):  r_t is clipped at 1-ε → can't decrease
                                probability too much

        ε = 0.2 is the standard hyperparameter.

    Full PPO objective (three components):
        L(θ) = L^CLIP − c₁·L^VF + c₂·S[π_θ]

        L^CLIP:  clipped policy gradient (as above)
        L^VF:    value function loss = (V(s;θ) − Vₜarget)²
        S:       entropy bonus = -Σ π log π (encourages exploration)
        c₁ ≈ 0.5, c₂ ≈ 0.01

### PPO Training Loop

    PPO is an on-policy algorithm. The loop:

        for each iteration:
            1. COLLECT: run current policy π_θ_old for N steps
               Store: (s, a, r, s', done, log π_old(a|s), V(s))
            2. COMPUTE advantages using GAE with λ=0.95
            3. OPTIMISE: for K epochs over the collected data:
                  Sample minibatches
                  Compute L^CLIP + L^VF + L^ENT
                  Update θ by gradient ascent
            4. θ_old ← θ  (update reference policy)

    Key hyperparameters:
        N (rollout length):         2048 steps typically
        K (epochs per iteration):   4–10 epochs
        batch_size:                 64–256
        γ:                          0.99
        λ (GAE):                    0.95
        ε (clip):                   0.2
        learning rate:              3×10⁻⁴ (often annealed to 0)
        c₁ (value loss coeff):      0.5
        c₂ (entropy coeff):         0.01

    Why PPO dominates in practice:
        Simple to implement (just a few lines of loss code)
        Works well across many different environments
        Robust to hyperparameter choices
        Scales to large networks and parallel environments
        Used in: ChatGPT, OpenAI Five, many robotics systems


##### PART 7 — THE GYMNASIUM API AND ENVIRONMENT DESIGN

### The Gymnasium Interface

    Every Gymnasium environment exposes a consistent interface:

        import gymnasium as gym

        env = gym.make("CartPole-v1")
        obs, info  = env.reset(seed=42)      # start new episode

        for step in range(1000):
            action = env.action_space.sample()            # random policy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                obs, info = env.reset()

        env.close()

    The five return values of env.step(action):
        observation:  the new observation (state representation)
        reward:       scalar reward signal for this transition
        terminated:   True if reached a terminal state (game over, goal reached)
        truncated:    True if hit a time limit (not a true terminal state)
        info:         auxiliary diagnostic info (not for learning, for debugging)

    Gymnasium 0.26+ separated "done" into "terminated" and "truncated"
    because they require different handling:
        terminated: true end of episode (use bootstrap value = 0)
        truncated:  artificial end (use bootstrap value = V(s_last))

### Observation and Action Spaces

    Spaces define the format of observations and actions:

    Discrete(n):
        n discrete integers: {0, 1, ..., n-1}
        CartPole actions: Discrete(2) → {push left, push right}
        Atari actions: Discrete(18) → {no-op, fire, up, down, ...}

    Box(low, high, shape, dtype):
        Continuous n-dimensional box
        CartPole observation: Box(-inf, inf, shape=(4,), dtype=float32)
        → [cart position, cart velocity, pole angle, pole angular velocity]
        Atari pixels: Box(0, 255, shape=(210,160,3), dtype=uint8)

    MultiBinary(n):
        n-dimensional binary array: each element is 0 or 1
        Used for multi-label actions (e.g. press multiple buttons simultaneously)

    MultiDiscrete(nvec):
        Multiple discrete components: [Discrete(n1), Discrete(n2), ...]
        e.g. [0,1,2] × [0,1] for (movement direction) × (fire/no-fire)

    Checking spaces:
        env.action_space       → the action space
        env.observation_space  → the observation space
        env.action_space.n     → number of actions (Discrete)
        env.observation_space.shape → obs shape (Box)

### Key Gymnasium Environments

    Classic Control (no pixels, small state spaces — ideal for learning):
        CartPole-v1:    Balance a pole on a cart. 4D state, 2 actions.
                        Reward: +1 every step pole stays up. Max 500.
        MountainCar-v0: Drive a car up a hill. Sparse reward (-1 each step).
                        Hard for tabular methods due to sparse rewards.
        Acrobot-v1:     Swing up a two-link chain. 6D state, 3 actions.
        Pendulum-v1:    Swing up and balance pendulum. Continuous actions.

    Box2D (physics simulation):
        LunarLander-v3:    Land a spacecraft. 8D state, 4 discrete actions.
        BipedalWalker-v3:  Make a biped walk. 24D state, 4 continuous actions.
        CarRacing-v3:      Race a car from top-down view (pixel or vector obs).

    Atari (from pixel observations — classic DRL benchmarks):
        Pong-v5, Breakout-v5, SpaceInvaders-v5, Qbert-v5, ...
        Observations: 210×160×3 RGB pixels
        Requires preprocessing: greyscale, resize to 84×84, frame stacking

    MuJoCo (physics-based robotics — continuous control):
        HalfCheetah-v5, Hopper-v5, Humanoid-v5, Ant-v5, Walker2d-v5
        High-dimensional continuous state and action spaces
        Requires MuJoCo physics engine licence or mujoco-py

    Toy Text (discrete, tiny — for algorithm debugging):
        FrozenLake-v1:  Navigate an 8×8 grid lake. Slippery variant is hard.
        Blackjack-v1:   Classic card game. Partial observability.
        CliffWalking-v0: The classic SARSA vs Q-learning comparison env.
        Taxi-v3:        Pick up and drop off passengers. 500 states.

### Wrappers: Transforming Environments

    Wrappers modify an environment without changing its core:

    TimeLimit(env, max_episode_steps=500):
        Truncate episodes after max steps.

    RecordEpisodeStatistics(env):
        Adds episode total reward and length to the info dict.
        info['episode']['r']  → total episodic reward
        info['episode']['l']  → episode length

    RecordVideo(env, video_folder, episode_trigger):
        Records episodes as MP4 video files.

    NormalizeObservation(env):
        Running-mean normalises observations (z-score).
        Critical for stable neural network training.

    NormalizeReward(env, gamma=0.99):
        Normalises rewards using running statistics of discounted returns.

    FrameStack(env, n_frames=4):
        Stacks n consecutive frames as the observation.
        Allows the network to perceive velocity/motion from frames.
        Standard for Atari DQN: stack 4 greyscale frames.

    FlattenObservation(env):
        Flattens multi-dimensional observations to 1D.

    GrayScaleObservation(env):
        Converts RGB observations to greyscale.

    ResizeObservation(env, shape=(84,84)):
        Resizes pixel observations.

    Custom wrapper pattern:
        class RewardShapingWrapper(gym.Wrapper):
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                reward += 0.1 * obs[2]   # bonus for being upright (CartPole)
                return obs, reward, terminated, truncated, info


##### PART 8 — PRACTICAL RL: TRAINING STABILITY AND ALGORITHM SELECTION

### The RL Debugging Challenge

    RL is notoriously difficult to debug because:
        1. Reward is sparse: the agent may receive no reward for thousands of steps
        2. Non-stationarity: the data distribution changes as the policy improves
        3. Sensitivity: tiny hyperparameter changes can cause complete failure
        4. High variance: runs with identical hyperparameters but different seeds
           can produce radically different results (run multiple seeds!)
        5. Reward hacking: the agent finds unintended ways to maximise reward

### Reward Shaping

    Sparse rewards make RL extremely hard (the agent never finds the goal).
    Reward shaping adds dense auxiliary rewards to guide exploration:

        r_shaped = r_env + φ(s') − φ(s)   (potential-based shaping)

    Potential-based shaping guarantees the optimal policy is unchanged.
    Examples:
        CartPole: bonus for pole upright angle (guides early learning)
        Navigation: negative distance to goal (reward for getting closer)
        Robotics: bonus for hand-object proximity (encourages reaching)

    Danger: reward hacking. An agent given a bonus for proximity to the goal
    might oscillate near the goal without actually completing the task.
    Always evaluate on the ORIGINAL reward, not the shaped one.

### Key Training Stability Techniques

    Gradient clipping:
        Clip gradient norms to prevent explosive updates.
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)

    Entropy regularisation:
        Add entropy bonus to encourage exploration throughout training.
        Without it, the policy can collapse to deterministic too early.
        L = L_policy − c · H(π)   where H(π) = -Σ π log π

    Value function normalisation:
        Normalise advantages to zero mean, unit variance per minibatch.
        Prevents extreme gradient magnitudes from large rewards.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    Learning rate scheduling:
        Linear decay from initial lr to 0 over training is common for PPO.
        Ensures the policy stabilises by the end of training.

    Orthogonal initialisation:
        Initialise network weights with orthogonal matrices.
        Shown to improve performance in deep RL (used in PPO implementations).

    Multiple random seeds:
        ALWAYS run ≥5 seeds and report mean ± std.
        A single seed result is meaningless in RL.

### Algorithm Selection Guide

    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Scenario                          │ Algorithm      │ Reason              │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ Discrete actions, small state     │ Q-learning     │ Tabular, exact      │
    │ Discrete actions, function approx │ DQN/DuelingDQN │ Value-based         │
    │ Continuous actions                │ PPO/SAC        │ Policy-based        │
    │ Sample efficiency critical        │ SAC/TD3        │ Off-policy, replay  │
    │ Stable, reproducible results      │ PPO            │ Robust, simple      │
    │ Parallel environments available   │ PPO/A3C        │ On-policy parallel  │
    │ Learning from demonstrations      │ GAIL/DAgger    │ Imitation learning  │
    │ Partial observability             │ R2D2/LSTM      │ Recurrent networks  │
    │ Multi-agent                       │ MAPPO/QMIX     │ Multi-agent RL      │
    │ LLM alignment (RLHF)              │ PPO+reward     │ Human preference    │
    └──────────────────────────────────────────────────────────────────────────┘

### Evaluation Best Practices

    Training return vs evaluation return:
        Training return includes exploration noise (ε-greedy, entropy).
        Evaluation return: run greedy policy (ε=0) for N episodes.
        Always separate training and evaluation.

    Smoothing:
        Plot moving average of episode returns over 100 episodes.
        Raw returns are too noisy for learning curves.

    Statistical significance:
        Run N=5–10 random seeds.
        Report mean ± standard deviation or confidence interval.
        Use a paired statistical test if comparing algorithms.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Gymnasium API & MDP Fundamentals — Environments and Spaces": {
        "description": (
            "Complete Gymnasium API tour from first principles. "
            "Environment creation, reset, step, and close lifecycle. "
            "Observation and action space inspection for multiple envs. "
            "Random agent rollout with statistics collection. "
            "Terminated vs truncated distinction and proper handling. "
            "RecordEpisodeStatistics wrapper for metrics. "
            "Custom GridWorld environment implementing the Gym interface. "
            "MDP formalisation: state, action, transition, reward extraction. "
            "Return computation with different discount factors. "
            "Bellman equation verification on a simple MDP. "
            "Policy evaluation demo: computing V^pi via iterative methods."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from collections import defaultdict

try:
    import gymnasium as gym
    print(f"  Gymnasium version: {gym.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "gymnasium", "--quiet"], check=True)
    import gymnasium as gym
    print(f"  Gymnasium version: {gym.__version__}")

print("=" * 65)
print("  GYMNASIUM API & MDP FUNDAMENTALS")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: The Gymnasium API lifecycle
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The Gymnasium API: create, reset, step, close")
print("━" * 65)
print()

env = gym.make("CartPole-v1")

print(f"  Environment: CartPole-v1")
print(f"  Observation space: {env.observation_space}")
print(f"    shape = {env.observation_space.shape}")
print(f"    dtype = {env.observation_space.dtype}")
print(f"    low   = {env.observation_space.low}")
print(f"    high  = {env.observation_space.high[:4]}")  # inf values
print()
print(f"  Action space: {env.action_space}")
print(f"    n       = {env.action_space.n}  (0=push left, 1=push right)")
print()

# Reset — starts a new episode
obs, info = env.reset(seed=42)
print(f"  env.reset() →")
print(f"    obs:  {obs.round(4)}  (cart_pos, cart_vel, pole_angle, pole_vel)")
print(f"    info: {info}")
print()

# Step — apply an action
action = 1   # push right
obs_new, reward, terminated, truncated, info = env.step(action)
print(f"  env.step(action=1) →")
print(f"    obs:        {obs_new.round(4)}")
print(f"    reward:     {reward}  (always +1 per step until failure)")
print(f"    terminated: {terminated}  (pole fell or cart out of bounds)")
print(f"    truncated:  {truncated}   (500-step time limit exceeded)")
print(f"    info:       {info}")
print()

# Four observations explained
CART_POLE_FEATURES = [
    ("cart_pos",   "Cart position (m). |x| > 2.4 → terminated"),
    ("cart_vel",   "Cart velocity (m/s)"),
    ("pole_angle", "Pole angle (rad). |θ| > 12° = 0.209 rad → terminated"),
    ("pole_vel",   "Pole angular velocity (rad/s)"),
]
print(f"  CartPole-v1 observation components:")
for feat, desc in CART_POLE_FEATURES:
    print(f"    {feat:<14}: {desc}")
print()
env.close()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Space inspection across different environments
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Space inspection across multiple environments")
print("━" * 65)
print()

env_configs = [
    ("CartPole-v1",       "Classic control, balance pole"),
    ("MountainCar-v0",    "Classic control, sparse reward"),
    ("Acrobot-v1",        "Classic control, 6D continuous"),
    ("LunarLander-v3",    "Box2D physics, land spacecraft"),
    ("FrozenLake-v1",     "Toy text, slippery grid world"),
    ("Taxi-v3",           "Toy text, navigation + pickup"),
]

print(f"  {'Environment':<22} {'Obs space':<30} {'Act space':<20} {'Goal'}")
print(f"  {'─'*85}")
for env_name, goal in env_configs:
    try:
        e = gym.make(env_name)
        obs_sp = str(e.observation_space)[:28]
        act_sp = str(e.action_space)[:18]
        print(f"  {env_name:<22} {obs_sp:<30} {act_sp:<20} {goal}")
        e.close()
    except Exception as ex:
        print(f"  {env_name:<22} (not installed: {str(ex)[:30]})")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Random agent with statistics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Random agent rollout and episode statistics")
print("━" * 65)
print()

from gymnasium.wrappers import RecordEpisodeStatistics

N_EPISODES = 200
env = RecordEpisodeStatistics(gym.make("CartPole-v1"))

episode_returns = []
episode_lengths = []

for ep in range(N_EPISODES):
    obs, info = env.reset(seed=ep)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    episode_returns.append(info["episode"]["r"])
    episode_lengths.append(info["episode"]["l"])

env.close()

returns = np.array(episode_returns)
lengths = np.array(episode_lengths)

print(f"  Random policy on CartPole-v1 ({N_EPISODES} episodes):")
print(f"  {'Metric':<25} {'Value'}")
print(f"  {'─'*40}")
print(f"  {'Mean return':<25} {returns.mean():.2f}")
print(f"  {'Std return':<25} {returns.std():.2f}")
print(f"  {'Min return':<25} {returns.min():.1f}")
print(f"  {'Max return':<25} {returns.max():.1f}")
print(f"  {'Mean episode length':<25} {lengths.mean():.2f}")
print(f"  {'% episodes > 50 steps':<25} {(returns > 50).mean()*100:.1f}%")
print(f"  {'% perfect (500 steps)':<25} {(returns >= 500).mean()*100:.1f}%")
print()

# Return distribution
bins = [1, 10, 25, 50, 100, 200, 500]
print(f"  Return distribution:")
for i in range(len(bins)-1):
    n = ((returns >= bins[i]) & (returns < bins[i+1])).sum()
    bar = "█" * (n // max(N_EPISODES//50, 1))
    print(f"    [{bins[i]:>4}, {bins[i+1]:>4}):  {n:>4}  {bar}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Custom MDP — GridWorld implementation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Custom GridWorld: implementing the Gym interface")
print("━" * 65)
print()

class GridWorldEnv(gym.Env):
    """
    A 4×4 GridWorld MDP.

    Layout:
        S . . .     S = Start (0,0)
        . H . .     H = Hole  (1,1), (2,3), (3,0)
        . . . H
        H . . G     G = Goal  (3,3)

    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    Rewards: +1.0 at goal, -1.0 at hole, -0.01 per step (encourages speed)
    Terminal: reaching goal or hole
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=4, slippery=False):
        super().__init__()
        self.size       = size
        self.slippery   = slippery
        self.holes      = {(1,1), (2,3), (3,0)}
        self.goal       = (3, 3)
        self.start      = (0, 0)
        self._pos       = None

        # Spaces
        self.observation_space = gym.spaces.Discrete(size * size)
        self.action_space      = gym.spaces.Discrete(4)   # U R D L

        self._action_to_delta = {
            0: (-1, 0),   # Up    (row-1)
            1: ( 0, 1),   # Right (col+1)
            2: ( 1, 0),   # Down  (row+1)
            3: ( 0,-1),   # Left  (col-1)
        }

    def _pos_to_obs(self, r, c):
        return r * self.size + c

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = self.start
        return self._pos_to_obs(*self._pos), {}

    def step(self, action):
        r, c = self._pos

        # Slippery: 33% chance to go perpendicular
        if self.slippery and self.np_random.random() < 0.33:
            action = (action + self.np_random.choice([-1, 1])) % 4

        dr, dc  = self._action_to_delta[action]
        nr, nc  = np.clip(r + dr, 0, self.size-1), np.clip(c + dc, 0, self.size-1)
        self._pos = (nr, nc)

        terminated = False
        reward     = -0.01   # small step penalty

        if self._pos in self.holes:
            reward, terminated = -1.0, True
        elif self._pos == self.goal:
            reward, terminated = +1.0, True

        return self._pos_to_obs(*self._pos), reward, terminated, False, {}

    def render(self, mode="ansi"):
        SYMBOLS = {self.goal: "G", self.start: "S"}
        for h in self.holes: SYMBOLS[h] = "H"
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                sym = "P" if (r,c) == self._pos else SYMBOLS.get((r,c), ".")
                row.append(sym)
            rows.append(" ".join(row))
        return "\n".join(rows)

    @property
    def n_states(self):  return self.size * self.size

    @property
    def n_actions(self): return 4


# Demonstrate the environment
gw = GridWorldEnv(size=4, slippery=False)
obs, _ = gw.reset(seed=0)
print(f"  GridWorld (4×4, non-slippery):")
print(f"    States:  {gw.n_states}  (0-15, row-major)")
print(f"    Actions: {gw.n_actions}  (0=Up, 1=Right, 2=Down, 3=Left)")
print(f"    Holes:   {sorted(gw.holes)}")
print(f"    Goal:    {gw.goal}")
print()
print(f"  Initial state:")
print(gw.render())
print(f"  obs = {obs}")
print()

# Show a few steps
path = [1, 2, 1, 2, 1, 2, 1]   # Right, Down, Right, Down, Right, Down, Right
gw.reset()
print(f"  Following path [Right×1, Down×1, Right×1, Down×1, Right×1, Down×1, Right×1]:")
total_reward = 0
for action_id in path:
    action_names = ["Up", "Right", "Down", "Left"]
    obs, reward, term, trunc, info = gw.step(action_id)
    total_reward += reward
    r, c = obs // gw.size, obs % gw.size
    print(f"    action={action_names[action_id]:<6} → pos=({r},{c})  "
          f"reward={reward:>+6.2f}  terminated={term}")
    if term: break
print(f"  Total reward: {total_reward:.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: MDP formalisation and Bellman verification
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — MDP → Bellman equation: policy evaluation")
print("━" * 65)
print()

# Small 3-state MDP for exact Bellman demonstration
# States: s0, s1, s2 (s2 is terminal/goal)
# Actions: a0 (stay-ish), a1 (go forward)
# P[s][a] = list of (s', prob, reward)

P_mdp = {
    0: {
        0: [(0, 0.7, -0.1), (1, 0.3, -0.1)],   # stay: usually stay, sometimes advance
        1: [(1, 0.8, -0.1), (0, 0.2, -0.1)],   # advance: usually go to s1
    },
    1: {
        0: [(1, 0.6, -0.1), (0, 0.4, -0.1)],   # stay in s1
        1: [(2, 0.9, +1.0), (1, 0.1, -0.1)],   # advance to goal
    },
    2: {  # terminal state
        0: [(2, 1.0, 0.0)],
        1: [(2, 1.0, 0.0)],
    }
}

gamma = 0.9

def policy_evaluation(policy, P, gamma=0.9, theta=1e-8):
    """Iterative policy evaluation: compute V^pi."""
    n_states  = len(P)
    V = np.zeros(n_states)
    iteration = 0
    while True:
        delta = 0
        for s in range(n_states - 1):   # skip terminal state 2
            v_old = V[s]
            v_new = 0.0
            for a in range(len(P[s])):
                pi_a = policy[s][a]   # probability of action a in state s
                for s_next, prob, reward in P[s][a]:
                    v_new += pi_a * prob * (reward + gamma * V[s_next])
            V[s]  = v_new
            delta = max(delta, abs(v_old - v_new))
        iteration += 1
        if delta < theta:
            break
    return V, iteration

# Policy 1: always take action 0 (conservative)
pi_conservative = {0: [1.0, 0.0], 1: [1.0, 0.0]}
V_cons, iters_cons = policy_evaluation(pi_conservative, P_mdp, gamma)

# Policy 2: always take action 1 (aggressive)
pi_aggressive = {0: [0.0, 1.0], 1: [0.0, 1.0]}
V_agg, iters_agg = policy_evaluation(pi_aggressive, P_mdp, gamma)

# Policy 3: mixed (50/50)
pi_mixed = {0: [0.5, 0.5], 1: [0.5, 0.5]}
V_mix, iters_mix = policy_evaluation(pi_mixed, P_mdp, gamma)

print(f"  3-state MDP (γ={gamma}):")
print(f"  States: s0 (start) → s1 (mid) → s2 (goal)")
print(f"  Transitions:")
print(f"    s0 a0 (conservative): 70% → s0, 30% → s1 (r=-0.1)")
print(f"    s0 a1 (aggressive):   80% → s1, 20% → s0 (r=-0.1)")
print(f"    s1 a1 (aggressive):   90% → s2, 10% → s1 (r=+1.0 at goal)")
print()

print(f"  Policy evaluation results:")
print(f"  {'Policy':<15} {'V(s0)':>10} {'V(s1)':>10} {'Iterations':>12}")
print(f"  {'─'*50}")
for name, V, iters in [
    ("Conservative", V_cons, iters_cons),
    ("Aggressive",   V_agg,  iters_agg),
    ("Mixed 50/50",  V_mix,  iters_mix),
]:
    print(f"  {name:<15} {V[0]:>10.4f} {V[1]:>10.4f} {iters:>12}")
print()

# Verify Bellman equation manually for aggressive policy at s1
print(f"  Manual Bellman check for aggressive policy at s1:")
a = 1  # aggressive action
bellman_s1 = sum(prob * (reward + gamma * V_agg[s_next])
                 for s_next, prob, reward in P_mdp[1][a])
print(f"    V^pi(s1) computed:  {V_agg[1]:.6f}")
print(f"    Bellman equation:   {bellman_s1:.6f}")
print(f"    Match:              {abs(V_agg[1] - bellman_s1) < 1e-6} ✅")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Tabular RL — Q-Learning, SARSA, and Policy Iteration": {
        "description": (
            "Complete tabular reinforcement learning on discrete environments. "
            "Policy iteration: policy evaluation + greedy improvement. "
            "Value iteration: Bellman optimality applied directly. "
            "Q-learning on FrozenLake: convergence to optimal policy. "
            "SARSA on CliffWalking: on-policy vs off-policy comparison. "
            "ε-greedy exploration with decay schedule visualisation. "
            "Q-table inspection: learned values and derived policy. "
            "Learning curve analysis: episode return over training. "
            "Hyperparameter sensitivity: α and γ ablation study. "
            "Q-learning vs SARSA safety comparison on cliff environment. "
            "Convergence diagnostics: TD error over training."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from collections import defaultdict

try:
    import gymnasium as gym
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "gymnasium", "--quiet"], check=True)
    import gymnasium as gym

print("=" * 65)
print("  TABULAR RL — Q-LEARNING, SARSA, AND POLICY ITERATION")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Value iteration on FrozenLake
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Value iteration on FrozenLake-v1")
print("━" * 65)
print()

env_vi = gym.make("FrozenLake-v1", is_slippery=True)
P_fl   = env_vi.unwrapped.P   # dict: P[s][a] = list of (prob, next_s, reward, done)
n_s, n_a = env_vi.observation_space.n, env_vi.action_space.n

def value_iteration(P, n_states, n_actions, gamma=0.99, theta=1e-8):
    V = np.zeros(n_states)
    iteration = 0
    history   = []
    while True:
        delta = 0
        for s in range(n_states):
            q_vals = []
            for a in range(n_actions):
                q = sum(prob * (r + gamma * V[s2] * (not done))
                        for prob, s2, r, done in P[s][a])
                q_vals.append(q)
            v_new  = max(q_vals)
            delta  = max(delta, abs(V[s] - v_new))
            V[s]   = v_new
        history.append(delta)
        iteration += 1
        if delta < theta:
            break
    # Extract greedy policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_vals = [sum(prob*(r+gamma*V[s2]*(not done))
                      for prob, s2, r, done in P[s][a])
                  for a in range(n_actions)]
        policy[s] = np.argmax(q_vals)
    return V, policy, history

t0 = time.perf_counter()
V_star, pi_star, vi_history = value_iteration(P_fl, n_s, n_a, gamma=0.99)
t_vi = (time.perf_counter() - t0) * 1000

print(f"  FrozenLake-v1 (4×4, slippery): {n_s} states, {n_a} actions")
print(f"  Value iteration converged in {len(vi_history)} sweeps ({t_vi:.1f}ms)")
print()
print(f"  Optimal V* values (4×4 grid, row-major):")
action_symbols = ["↑", "→", "↓", "←"]
for row in range(4):
    v_row = "  "
    p_row = "  "
    for col in range(4):
        s = row * 4 + col
        v_row += f"{V_star[s]:>6.3f} "
        p_row += f"  {action_symbols[pi_star[s]]}   "
    print(v_row)
print()
print(f"  Optimal policy π* (4×4 grid):")
for row in range(4):
    p_row = "  "
    for col in range(4):
        s = row * 4 + col
        p_row += f"  {action_symbols[pi_star[s]]}  "
    print(p_row)
print()

# Evaluate the optimal policy
def evaluate_policy(env, policy, n_eval=1000):
    returns = []
    for ep in range(n_eval):
        obs, _ = env.reset(seed=ep)
        total_r = 0.0
        done    = False
        for _ in range(100):
            a = policy[obs]
            obs, r, term, trunc, _ = env.step(a)
            total_r += r
            done = term or trunc
            if done: break
        returns.append(total_r)
    return np.mean(returns)

win_rate = evaluate_policy(env_vi, pi_star, n_eval=1000)
print(f"  Optimal policy win rate (1000 episodes): {win_rate*100:.1f}%")
print(f"  (Random policy win rate ≈ 1–2% on slippery FrozenLake)")
print()
env_vi.close()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Q-learning on FrozenLake
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Q-learning: convergence and Q-table inspection")
print("━" * 65)
print()

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
        self.Q         = np.zeros((n_states, n_actions))
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.td_errors = []

    def select_action(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(self.Q.shape[1])   # explore
        return np.argmax(self.Q[state])                  # exploit

    def update(self, s, a, r, s_next, done):
        target    = r + (0.0 if done else self.gamma * np.max(self.Q[s_next]))
        td_error  = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        self.td_errors.append(abs(td_error))

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

def train_agent(AgentClass, env_name, n_episodes=5000, **kwargs):
    env     = gym.make(env_name)
    n_s     = env.observation_space.n
    n_a     = env.action_space.n
    agent   = AgentClass(n_s, n_a, **kwargs)
    returns = []

    for ep in range(n_episodes):
        obs, _   = env.reset(seed=ep % 100)
        total_r  = 0.0
        done     = False
        for _ in range(100):
            action = agent.select_action(obs)
            next_obs, r, term, trunc, _ = env.step(action)
            done   = term or trunc
            agent.update(obs, action, r, next_obs, done)
            obs    = next_obs
            total_r += r
            if done: break
        agent.decay_epsilon()
        returns.append(total_r)
    env.close()
    return agent, returns

t0 = time.perf_counter()
ql_agent, ql_returns = train_agent(
    QLearning, "FrozenLake-v1",
    n_episodes=8000, alpha=0.1, gamma=0.99,
    eps_start=1.0, eps_end=0.01, eps_decay=0.9994
)
t_ql = time.perf_counter() - t0

print(f"  Q-learning on FrozenLake-v1 (8000 episodes, {t_ql:.1f}s):")
ql_arr = np.array(ql_returns)
for window, label in [(100,"first 100"), (500,"500"), (1000,"last 1000")]:
    if label.startswith("last"):
        chunk = ql_arr[-window:]
    elif label.startswith("first"):
        chunk = ql_arr[:window]
    else:
        idx = (len(ql_arr) - window) // 2
        chunk = ql_arr[idx:idx+window]
    print(f"    {label:>15} eps: win_rate={chunk.mean()*100:.1f}%  "
          f"eps={ql_agent.eps:.4f}")
print()

# Final epsilon
print(f"  Final epsilon: {ql_agent.eps:.4f}")
print(f"  TD error (mean last 1000 steps): "
      f"{np.mean(ql_agent.td_errors[-1000:]):.5f}")
print()

# Q-table inspection (agreement with value iteration)
ql_policy = np.argmax(ql_agent.Q, axis=1)
agreement = (ql_policy == pi_star).mean()
print(f"  Q-learning policy agreement with optimal (value iteration): "
      f"{agreement*100:.1f}%")
print()
print(f"  Learned Q-table (first 4 states, max Q per state):")
print(f"  {'State':>6} {'↑':>8} {'→':>8} {'↓':>8} {'←':>8} "
      f"{'Best action':>14} {'Optimal':>10}")
print(f"  {'─'*65}")
for s in range(4):
    q_row = ql_agent.Q[s]
    best  = action_symbols[np.argmax(q_row)]
    opt   = action_symbols[pi_star[s]]
    match = "✓" if np.argmax(q_row) == pi_star[s] else "✗"
    print(f"  {s:>6} {q_row[0]:>8.4f} {q_row[1]:>8.4f} "
          f"{q_row[2]:>8.4f} {q_row[3]:>8.4f} "
          f"{best:>12} {match}  {opt:>6}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: SARSA vs Q-learning comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — SARSA vs Q-learning: safety comparison")
print("━" * 65)
print()

class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.1, eps_decay=0.9993):
        self.Q         = np.zeros((n_states, n_actions))
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.td_errors = []

    def select_action(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, a_next, done):
        target = r + (0.0 if done else self.gamma * self.Q[s_next, a_next])
        td_err = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_err
        self.td_errors.append(abs(td_err))

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

def train_sarsa(env_name, n_episodes=5000, **kwargs):
    env     = gym.make(env_name)
    n_s, n_a = env.observation_space.n, env.action_space.n
    agent   = SARSA(n_s, n_a, **kwargs)
    returns = []

    for ep in range(n_episodes):
        obs, _   = env.reset(seed=ep % 200)
        action   = agent.select_action(obs)
        total_r  = 0.0
        done     = False
        for _ in range(200):
            next_obs, r, term, trunc, _ = env.step(action)
            done    = term or trunc
            a_next  = agent.select_action(next_obs) if not done else 0
            agent.update(obs, action, r, next_obs, a_next, done)
            obs    = next_obs
            action = a_next
            total_r += r
            if done: break
        agent.decay_epsilon()
        returns.append(total_r)
    env.close()
    return agent, returns

# Train both on CliffWalking (the classic comparison environment)
print(f"  Training on CliffWalking-v0:")
print(f"  (Safe path = along top, risky path = along cliff bottom)")
print()

np.random.seed(42)
t0 = time.perf_counter()
ql_cliff,   ql_cliff_ret   = train_agent(
    QLearning, "CliffWalking-v0", n_episodes=500,
    alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.1, eps_decay=0.993)
sarsa_cliff, sarsa_cliff_ret = train_sarsa(
    "CliffWalking-v0", n_episodes=500,
    alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.1, eps_decay=0.993)
t_cliff = time.perf_counter() - t0

ql_arr  = np.array(ql_cliff_ret)
sa_arr  = np.array(sarsa_cliff_ret)

print(f"  Training completed in {t_cliff:.1f}s")
print()
print(f"  {'Metric':<35} {'Q-learning':>12} {'SARSA':>12}")
print(f"  {'─'*62}")

for metric, ql_val, sa_val in [
    ("Final 100-ep mean return",
     ql_arr[-100:].mean(), sa_arr[-100:].mean()),
    ("Final 100-ep std return",
     ql_arr[-100:].std(), sa_arr[-100:].std()),
    ("Best single episode return",
     ql_arr.max(), sa_arr.max()),
    ("Worst single episode return",
     ql_arr.min(), sa_arr.min()),
    ("# episodes hitting cliff (r<-100)",
     (ql_arr < -100).sum(), (sa_arr < -100).sum()),
]:
    print(f"  {metric:<35} {ql_val:>12.2f} {sa_val:>12.2f}")

print()
print(f"  Interpretation:")
print(f"    Q-learning: learns optimal (risky cliff-edge) path but")
print(f"                during training with ε-greedy, falls often")
print(f"    SARSA:      learns safer (longer) path because ε-greedy")
print(f"                exploration is factored into the Q-updates")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Hyperparameter ablation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Hyperparameter sensitivity: α and γ ablation")
print("━" * 65)
print()

print(f"  Effect of learning rate α (γ=0.99, FrozenLake, 3000 eps):")
print(f"  {'α':>6} {'Final win rate':>16} {'Convergence ep':>16}")
print(f"  {'─'*42}")

for alpha in [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]:
    np.random.seed(42)
    agent_a, ret_a = train_agent(
        QLearning, "FrozenLake-v1", n_episodes=3000,
        alpha=alpha, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.999)
    ret_a = np.array(ret_a)
    final_wr = ret_a[-500:].mean() * 100
    # Find episode where 100-ep window first exceeds 10%
    conv_ep = 3000
    for i in range(100, 3000):
        if ret_a[i-100:i].mean() > 0.1:
            conv_ep = i
            break
    print(f"  {alpha:>6.2f} {final_wr:>16.1f}% {conv_ep:>16}")

print()
print(f"  Effect of discount factor γ (α=0.1, FrozenLake, 3000 eps):")
print(f"  {'γ':>6} {'Final win rate':>16} {'Interpretation'}")
print(f"  {'─'*60}")

for gamma, interp in [(0.5, "very myopic"), (0.8, "moderately far-sighted"),
                       (0.9, "far-sighted"), (0.99, "very far-sighted"),
                       (0.999, "near-undiscounted")]:
    np.random.seed(42)
    agent_g, ret_g = train_agent(
        QLearning, "FrozenLake-v1", n_episodes=3000,
        alpha=0.1, gamma=gamma, eps_start=1.0, eps_end=0.01, eps_decay=0.999)
    final_wr = np.array(ret_g)[-500:].mean() * 100
    print(f"  {gamma:>6.3f} {final_wr:>16.1f}% {interp}")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Deep Q-Network — DQN on CartPole from Scratch": {
        "description": (
            "Complete DQN implementation from scratch with PyTorch. "
            "Neural network Q-function: architecture and forward pass. "
            "Experience replay buffer: circular buffer with random sampling. "
            "Target network: hard update every C steps. "
            "Epsilon-greedy with decay schedule. "
            "Huber loss for stable Q-value regression. "
            "Full DQN training loop on CartPole-v1. "
            "Training diagnostics: loss, Q-values, epsilon, returns. "
            "Double DQN variant: action selection vs evaluation. "
            "Dueling DQN architecture: V(s) + A(s,a) streams. "
            "Training stability analysis: target network update frequency. "
            "Evaluation: greedy policy performance benchmarking."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from collections import deque

try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    print(f"  Gymnasium: {gym.__version__} | PyTorch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "gymnasium", "torch", "--quiet"], check=True)
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

print("=" * 65)
print("  DEEP Q-NETWORK — DQN ON CARTPOLE FROM SCRATCH")
print("=" * 65)
print()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")
print()

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: DQN Network architectures
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — DQN network architectures")
print("━" * 65)
print()

class QNetwork(nn.Module):
    """Standard DQN network: state → Q-values for all actions."""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN: separate streams for V(s) and A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
    Subtracting mean ensures identifiability (unique V and A decomposition).
    """
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
        )
        self.value_stream    = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),          # scalar V(s)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),   # A(s,a) for each action
        )

    def forward(self, x):
        features  = self.shared(x)
        V         = self.value_stream(features)           # (B, 1)
        A         = self.advantage_stream(features)        # (B, n_actions)
        Q         = V + (A - A.mean(dim=1, keepdim=True)) # identifiability
        return Q


# Show architecture
obs_dim, n_actions = 4, 2   # CartPole
standard = QNetwork(obs_dim, n_actions, hidden=128).to(DEVICE)
dueling  = DuelingQNetwork(obs_dim, n_actions, hidden=128).to(DEVICE)

print(f"  Standard DQNetwork:")
for name, p in standard.named_parameters():
    print(f"    {name:<25} {tuple(p.shape)}")
n_std = sum(p.numel() for p in standard.parameters())
print(f"    Total params: {n_std:,}")
print()

print(f"  Dueling QNetwork:")
for name, p in dueling.named_parameters():
    print(f"    {name:<35} {tuple(p.shape)}")
n_dual = sum(p.numel() for p in dueling.parameters())
print(f"    Total params: {n_dual:,}")
print()

# Test forward pass
dummy_obs = torch.randn(1, obs_dim).to(DEVICE)
with torch.no_grad():
    q_std  = standard(dummy_obs)
    q_dual = dueling(dummy_obs)
print(f"  Forward pass (batch=1):")
print(f"    Standard Q(s, ·): {q_std[0].cpu().numpy().round(4)}")
print(f"    Dueling  Q(s, ·): {q_dual[0].cpu().numpy().round(4)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Replay Buffer
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Replay buffer: random sampling and statistics")
print("━" * 65)
print()

class ReplayBuffer:
    """
    Circular replay buffer. When full, oldest experiences are overwritten.
    Stores transitions as arrays for efficient batch sampling.
    """
    def __init__(self, capacity, obs_dim):
        self.capacity  = capacity
        self.ptr       = 0
        self.size      = 0
        self.obs       = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions   = np.zeros(capacity, dtype=np.int64)
        self.rewards   = np.zeros(capacity, dtype=np.float32)
        self.dones     = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.obs[idxs]).to(DEVICE),
            torch.LongTensor(self.actions[idxs]).to(DEVICE),
            torch.FloatTensor(self.rewards[idxs]).to(DEVICE),
            torch.FloatTensor(self.next_obs[idxs]).to(DEVICE),
            torch.FloatTensor(self.dones[idxs]).to(DEVICE),
        )

    def __len__(self): return self.size

# Demonstrate buffer
buf = ReplayBuffer(10000, obs_dim=4)
# Fill with dummy data
for i in range(500):
    buf.push(np.random.randn(4), np.random.randint(2), 1.0,
             np.random.randn(4), False)
obs_b, act_b, rew_b, nobs_b, done_b = buf.sample(32)

print(f"  ReplayBuffer (capacity=10000, filled with 500 transitions):")
print(f"    Buffer size: {len(buf)}")
print(f"    Batch obs shape:      {obs_b.shape}   dtype={obs_b.dtype}")
print(f"    Batch actions shape:  {act_b.shape}   dtype={act_b.dtype}")
print(f"    Batch rewards shape:  {rew_b.shape}   dtype={rew_b.dtype}")
print(f"    Batch reward stats:   mean={rew_b.mean():.2f}, std={rew_b.std():.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Full DQN training loop
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Full DQN training on CartPole-v1")
print("━" * 65)
print()

# Hyperparameters
BUFFER_SIZE    = 10_000
BATCH_SIZE     = 64
LR             = 1e-3
GAMMA          = 0.99
EPS_START      = 1.0
EPS_END        = 0.05
EPS_DECAY      = 0.995
TARGET_UPDATE  = 50      # hard update every N episodes
WARMUP_STEPS   = 500     # start training after this many steps
N_EPISODES     = 400
EVAL_EVERY     = 50

print(f"  Hyperparameters:")
print(f"    Buffer size:   {BUFFER_SIZE:,}")
print(f"    Batch size:    {BATCH_SIZE}")
print(f"    Learning rate: {LR}")
print(f"    Gamma:         {GAMMA}")
print(f"    ε: {EPS_START} → {EPS_END} (decay {EPS_DECAY}/ep)")
print(f"    Target update: every {TARGET_UPDATE} episodes")
print()

env     = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_acts  = env.action_space.n

online_net = QNetwork(obs_dim, n_acts, hidden=128).to(DEVICE)
target_net = QNetwork(obs_dim, n_acts, hidden=128).to(DEVICE)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

optimiser = optim.Adam(online_net.parameters(), lr=LR)
buffer    = ReplayBuffer(BUFFER_SIZE, obs_dim)
eps       = EPS_START

episode_returns = []
losses          = []
q_val_history   = []

t_start = time.perf_counter()

for ep in range(N_EPISODES):
    obs_np, _ = env.reset(seed=ep % 500)
    obs_t     = torch.FloatTensor(obs_np).to(DEVICE)
    total_r   = 0.0
    done      = False

    while not done:
        # ε-greedy action selection
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = online_net(obs_t.unsqueeze(0)).argmax().item()

        next_obs_np, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        buffer.push(obs_np, action, reward, next_obs_np, done)

        obs_np = next_obs_np
        obs_t  = torch.FloatTensor(obs_np).to(DEVICE)
        total_r += reward

        # Training step
        if len(buffer) > WARMUP_STEPS:
            obs_b, act_b, rew_b, nobs_b, done_b = buffer.sample(BATCH_SIZE)

            with torch.no_grad():
                # Standard DQN target
                next_q    = target_net(nobs_b).max(dim=1).values
                target_q  = rew_b + GAMMA * next_q * (1.0 - done_b)

            current_q = online_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            loss      = F.smooth_l1_loss(current_q, target_q)   # Huber loss

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
            optimiser.step()

            losses.append(loss.item())
            if len(losses) % 100 == 0:
                q_val_history.append(current_q.mean().item())

    episode_returns.append(total_r)
    eps = max(EPS_END, eps * EPS_DECAY)

    # Hard target network update
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(online_net.state_dict())

t_train = time.perf_counter() - t_start
env.close()

# Training summary
ret_arr = np.array(episode_returns)
print(f"  Training complete: {N_EPISODES} episodes in {t_train:.1f}s")
print()
print(f"  Training progress:")
print(f"  {'Episodes':>12} {'Mean return':>14} {'Max return':>12} {'ε':>8}")
print(f"  {'─'*50}")
for start in [0, 100, 200, 300]:
    end   = min(start + 100, len(ret_arr))
    chunk = ret_arr[start:end]
    eps_at = EPS_START * (EPS_DECAY ** start)
    eps_at = max(EPS_END, eps_at)
    print(f"  {start:>4}–{end:<4}    {chunk.mean():>14.1f} {chunk.max():>12.1f} "
          f"{eps_at:>8.4f}")
print()

if losses:
    print(f"  Loss statistics:")
    loss_arr = np.array(losses)
    print(f"    Early (first 500):  mean={loss_arr[:500].mean():.4f}")
    print(f"    Late  (last 500):   mean={loss_arr[-500:].mean():.4f}")
    if q_val_history:
        q_arr = np.array(q_val_history)
        print(f"  Mean Q-values over time: {q_arr[0]:.2f} → {q_arr[-1]:.2f}")
        print(f"  (Should increase then stabilise as Q approaches Q*)")
print()

# Evaluate greedy policy
def evaluate_greedy(net, env_name, n_eval=50):
    e = gym.make(env_name)
    net.eval()
    returns = []
    with torch.no_grad():
        for seed in range(n_eval):
            obs, _ = e.reset(seed=seed)
            total_r = 0
            for _ in range(500):
                q   = net(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
                act = q.argmax().item()
                obs, r, term, trunc, _ = e.step(act)
                total_r += r
                if term or trunc: break
            returns.append(total_r)
    e.close()
    net.train()
    return np.array(returns)

eval_returns = evaluate_greedy(online_net, "CartPole-v1", n_eval=50)
print(f"  Greedy policy evaluation ({len(eval_returns)} episodes):")
print(f"    Mean return: {eval_returns.mean():.1f} (max=500)")
print(f"    Std return:  {eval_returns.std():.1f}")
print(f"    % perfect (500 steps): {(eval_returns >= 500).mean()*100:.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Double DQN comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Double DQN: reducing overestimation bias")
print("━" * 65)
print()

# Quick Double DQN demonstration
ddqn_net    = QNetwork(obs_dim, n_acts, hidden=128).to(DEVICE)
ddqn_target = QNetwork(obs_dim, n_acts, hidden=128).to(DEVICE)
ddqn_target.load_state_dict(ddqn_net.state_dict())
ddqn_target.eval()
ddqn_opt  = optim.Adam(ddqn_net.parameters(), lr=LR)
ddqn_buf  = ReplayBuffer(BUFFER_SIZE, obs_dim)

# Pre-fill buffer from standard DQN
for s_idx in range(min(len(buffer), BUFFER_SIZE)):
    ddqn_buf.push(buffer.obs[s_idx], buffer.actions[s_idx],
                  buffer.rewards[s_idx], buffer.next_obs[s_idx],
                  buffer.dones[s_idx])

# Compare Q-value estimates: DQN vs Double DQN on the same batch
obs_b, act_b, rew_b, nobs_b, done_b = ddqn_buf.sample(BATCH_SIZE)

with torch.no_grad():
    # Standard DQN target (overestimates)
    dqn_target_q = target_net(nobs_b).max(dim=1).values

    # Double DQN target (less overestimation)
    # Select action with online net, evaluate with target net
    ddqn_actions    = online_net(nobs_b).argmax(dim=1, keepdim=True)
    ddqn_target_q   = ddqn_target(nobs_b).gather(1, ddqn_actions).squeeze(1)

print(f"  Q-value estimation comparison (batch of {BATCH_SIZE}):")
print(f"  {'Metric':<35} {'DQN':>12} {'Double DQN':>12}")
print(f"  {'─'*62}")
print(f"  {'Mean next-state Q-estimate':<35} {dqn_target_q.mean():.4f} "
      f"{ddqn_target_q.mean():>12.4f}")
print(f"  {'Max next-state Q-estimate':<35} {dqn_target_q.max():.4f} "
      f"{ddqn_target_q.max():>12.4f}")
print(f"  {'Std next-state Q-estimate':<35} {dqn_target_q.std():.4f} "
      f"{ddqn_target_q.std():>12.4f}")
print()
overest = (dqn_target_q > ddqn_target_q).float().mean() * 100
print(f"  DQN overestimates vs DDQN in {overest:.1f}% of transitions")
print(f"  Double DQN systematically produces lower (less biased) estimates")
print()

print(f"  DQN SUMMARY TABLE:")
print(f"  ┌──────────────────────────────────────────────────────────────────┐")
print(f"  │ Variant           │ Key innovation           │ Improvement over  │")
print(f"  ├──────────────────────────────────────────────────────────────────┤")
print(f"  │ DQN               │ Replay + target network  │ Tabular Q-learning│")
print(f"  │ Double DQN        │ Decouple select/evaluate │ Reduces bias 15%  │")
print(f"  │ Dueling DQN       │ V(s) + A(s,a) streams    │ Faster value learn│")
print(f"  │ Prioritised ER    │ Sample by TD error       │ 2× sample effic.  │")
print(f"  │ N-step DQN        │ Multi-step returns       │ Reduces bias      │")
print(f"  │ Rainbow DQN       │ All 5 improvements       │ SOTA on Atari     │")
print(f"  └──────────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Policy Gradients & PPO — REINFORCE to Proximal Policy Optimisation": {
        "description": (
            "Policy gradient methods from REINFORCE to PPO. "
            "REINFORCE: Monte Carlo policy gradient with baseline. "
            "Variance reduction: baseline subtraction impact. "
            "Actor-Critic: joint policy and value network training. "
            "TD error as advantage estimate: online AC on CartPole. "
            "Generalised Advantage Estimation (GAE): λ sweep. "
            "PPO-Clip: clipped surrogate objective implementation. "
            "PPO full training loop: rollout → GAE → K-epoch update. "
            "Entropy bonus: exploration regularisation effect. "
            "Training diagnostics: policy loss, value loss, entropy. "
            "Algorithm comparison benchmark on CartPole-v1. "
            "Reward shaping: shaped vs unshaped return comparison."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from typing import List, Tuple

try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    print(f"  Gymnasium: {gym.__version__} | PyTorch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "gymnasium", "torch", "--quiet"], check=True)
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical

print("=" * 65)
print("  POLICY GRADIENTS & PPO — REINFORCE TO PROXIMAL OPTIMISATION")
print("=" * 65)
print()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")
print()
torch.manual_seed(0)
np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: REINFORCE with baseline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — REINFORCE: Monte Carlo policy gradient")
print("━" * 65)
print()

class PolicyNetwork(nn.Module):
    """Stochastic policy: outputs action probabilities."""
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return Categorical(logits=self.net(x))   # returns distribution

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        dist  = self(obs_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(0)


class ValueNetwork(nn.Module):
    """Baseline V(s): reduces variance of policy gradient."""
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted returns from episode rewards."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train_reinforce(env_name, n_episodes=500, gamma=0.99, lr=3e-3,
                    use_baseline=True):
    env      = gym.make(env_name)
    obs_dim  = env.observation_space.shape[0]
    n_acts   = env.action_space.n

    policy  = PolicyNetwork(obs_dim, n_acts).to(DEVICE)
    value   = ValueNetwork(obs_dim).to(DEVICE) if use_baseline else None
    p_opt   = optim.Adam(policy.parameters(), lr=lr)
    v_opt   = optim.Adam(value.parameters(), lr=lr) if value else None

    returns_hist = []
    for ep in range(n_episodes):
        obs, _  = env.reset(seed=ep % 300)
        log_probs, rewards, obs_list = [], [], []
        total_r = 0.0

        done = False
        while not done:
            action, lp = policy.select_action(obs)
            obs_list.append(obs.copy())
            log_probs.append(lp)
            obs, r, term, trunc, _ = env.step(action)
            rewards.append(r)
            total_r += r
            done = term or trunc

        # Compute discounted returns
        Gs    = torch.FloatTensor(compute_returns(rewards, gamma)).to(DEVICE)
        obs_t = torch.FloatTensor(np.array(obs_list)).to(DEVICE)

        # Baseline subtraction (variance reduction)
        if use_baseline:
            with torch.no_grad():
                baseline = value(obs_t)
            advantages = Gs - baseline
        else:
            advantages = Gs

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient update
        log_p = torch.stack(log_probs)
        p_loss = -(log_p * advantages.detach()).mean()
        p_opt.zero_grad()
        p_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        p_opt.step()

        # Value function update
        if use_baseline:
            v_pred = value(obs_t)
            v_loss = F.mse_loss(v_pred, Gs)
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

        returns_hist.append(total_r)
    env.close()
    return policy, np.array(returns_hist)

print(f"  Training REINFORCE on CartPole-v1:")
print(f"  (500 episodes each — comparing with and without baseline)")
print()

t0 = time.perf_counter()
pol_base,  ret_base  = train_reinforce("CartPole-v1", n_episodes=500,
                                        use_baseline=True)
pol_nobase, ret_nobase = train_reinforce("CartPole-v1", n_episodes=500,
                                          use_baseline=False)
t_rf = time.perf_counter() - t0

print(f"  Training time: {t_rf:.1f}s")
print()
print(f"  {'Metric':<35} {'With baseline':>16} {'No baseline':>16}")
print(f"  {'─'*70}")

for label, start, end in [("First 100 ep",0,100), ("Middle 100 ep",200,300),
                            ("Last 100 ep",400,500)]:
    wb = ret_base[start:end].mean()
    nb = ret_nobase[start:end].mean()
    print(f"  {label:<35} {wb:>16.1f} {nb:>16.1f}")

wb_std = ret_base[-100:].std()
nb_std = ret_nobase[-100:].std()
print(f"  {'Last 100 ep std (variance)':<35} {wb_std:>16.1f} {nb_std:>16.1f}")
print()
print(f"  Baseline reduces variance: "
      f"{(nb_std - wb_std)/nb_std * 100:.1f}% variance reduction")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Actor-Critic with TD error
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Actor-Critic: online updates with TD advantage")
print("━" * 65)
print()

class ActorCritic(nn.Module):
    """
    Shared network with two heads:
    - Actor: policy π(a|s)
    - Critic: value V(s)
    Sharing lower layers is computationally efficient and often works well.
    """
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor_head  = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        features   = self.shared(x)
        action_dist = Categorical(logits=self.actor_head(features))
        value       = self.critic_head(features).squeeze(-1)
        return action_dist, value


def train_actor_critic(env_name, n_steps_total=50000, gamma=0.99, lr=3e-3):
    env     = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts  = env.action_space.n

    ac       = ActorCritic(obs_dim, n_acts).to(DEVICE)
    opt      = optim.Adam(ac.parameters(), lr=lr)
    ep_ret   = 0.0
    returns  = []
    td_errors = []

    obs, _ = env.reset(seed=0)

    for step in range(n_steps_total):
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        dist, V = ac(obs_t)
        action  = dist.sample()
        log_p   = dist.log_prob(action)

        next_obs, reward, term, trunc, _ = env.step(action.item())
        done    = term or trunc
        ep_ret += reward

        # Compute TD error (advantage estimate)
        with torch.no_grad():
            _, V_next = ac(torch.FloatTensor(next_obs).unsqueeze(0).to(DEVICE))
        td_target = reward + gamma * V_next * (1 - float(done))
        td_error  = td_target - V

        # Actor loss: policy gradient with TD advantage
        actor_loss  = -log_p * td_error.detach()
        # Critic loss: MSE with TD target
        critic_loss = td_error.pow(2)
        # Entropy bonus: encourage exploration
        entropy     = dist.entropy().mean()
        loss        = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
        opt.step()

        td_errors.append(abs(td_error.item()))

        if done:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    env.close()
    return ac, returns, td_errors

t0 = time.perf_counter()
ac_model, ac_returns, ac_td = train_actor_critic(
    "CartPole-v1", n_steps_total=80000)
t_ac = time.perf_counter() - t0

ac_ret = np.array(ac_returns)
print(f"  Actor-Critic training: {len(ac_ret)} episodes in {t_ac:.1f}s")
print(f"  TD error trend (|δ|): "
      f"{np.mean(ac_td[:1000]):.4f} → {np.mean(ac_td[-1000:]):.4f}")
print()

for window_label, idx_start in [("First 20 ep", 0), ("Middle 20 ep", len(ac_ret)//2-10),
                                  ("Last 20 ep", max(0, len(ac_ret)-20))]:
    chunk = ac_ret[idx_start:idx_start+20]
    if len(chunk) > 0:
        print(f"    {window_label}: mean={chunk.mean():.1f}, max={chunk.max():.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: PPO-Clip implementation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — PPO-Clip: clipped surrogate objective")
print("━" * 65)
print()

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation."""
    advantages  = []
    last_adv    = 0.0
    values_ext  = values + [last_value]

    for t in reversed(range(len(rewards))):
        not_done  = 1.0 - dones[t]
        delta     = rewards[t] + gamma * values_ext[t+1] * not_done - values_ext[t]
        last_adv  = delta + gamma * lam * not_done * last_adv
        advantages.insert(0, last_adv)
    return advantages


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor  = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Orthogonal initialisation (standard for PPO)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def get_action(self, obs):
        dist   = Categorical(logits=self.actor(obs))
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def evaluate(self, obs, actions):
        dist    = Categorical(logits=self.actor(obs))
        log_p   = dist.log_prob(actions)
        entropy = dist.entropy()
        value   = self.critic(obs).squeeze(-1)
        return log_p, entropy, value


def train_ppo(env_name, total_timesteps=100000,
              n_steps=2048, n_epochs=4, batch_size=64,
              gamma=0.99, lam=0.95, clip_eps=0.2,
              lr=3e-4, c1=0.5, c2=0.01):
    env     = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts  = env.action_space.n

    net = PPOActorCritic(obs_dim, n_acts).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    returns_hist  = []
    losses_hist   = []
    episode_ret   = 0.0
    episode_count = 0
    obs_np, _     = env.reset(seed=0)

    iteration = 0
    step      = 0

    while step < total_timesteps:
        # ROLLOUT COLLECTION
        obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
        val_buf, logp_buf = [], []

        for _ in range(n_steps):
            obs_t  = torch.FloatTensor(obs_np).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, log_p, _ = net.get_action(obs_t)
                value             = net.get_value(obs_t)

            next_obs, reward, term, trunc, _ = env.step(action.item())
            done = term or trunc

            obs_buf.append(obs_np.copy())
            act_buf.append(action.item())
            rew_buf.append(reward)
            done_buf.append(float(done))
            val_buf.append(value.item())
            logp_buf.append(log_p.item())

            episode_ret += reward
            step         += 1

            if done:
                returns_hist.append(episode_ret)
                episode_ret   = 0.0
                episode_count += 1
                obs_np, _     = env.reset()
            else:
                obs_np = next_obs

        # Bootstrap last value
        with torch.no_grad():
            obs_t      = torch.FloatTensor(obs_np).unsqueeze(0).to(DEVICE)
            last_val   = net.get_value(obs_t).item()

        # Compute GAE advantages
        adv   = compute_gae(rew_buf, val_buf, done_buf, last_val, gamma, lam)
        adv_t = torch.FloatTensor(adv).to(DEVICE)
        ret_t = adv_t + torch.FloatTensor(val_buf).to(DEVICE)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        obs_t_all  = torch.FloatTensor(np.array(obs_buf)).to(DEVICE)
        act_t_all  = torch.LongTensor(act_buf).to(DEVICE)
        logp_old   = torch.FloatTensor(logp_buf).to(DEVICE)

        # MULTIPLE EPOCHS OF OPTIMISATION
        iter_losses = []
        for epoch in range(n_epochs):
            perm  = torch.randperm(n_steps)
            for i in range(0, n_steps, batch_size):
                idx  = perm[i:i+batch_size]
                o_b  = obs_t_all[idx]
                a_b  = act_t_all[idx]
                lp_old_b = logp_old[idx]
                adv_b    = adv_t[idx]
                ret_b    = ret_t[idx]

                log_p_new, entropy, values = net.evaluate(o_b, a_b)
                ratio = torch.exp(log_p_new - lp_old_b)

                # Clipped surrogate objective
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1-clip_eps, 1+clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(values, ret_b)
                entropy_loss = -entropy.mean()

                loss = policy_loss + c1 * value_loss + c2 * entropy_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                iter_losses.append(loss.item())

        losses_hist.append(np.mean(iter_losses))
        iteration += 1

    env.close()
    return net, np.array(returns_hist), np.array(losses_hist)


t0 = time.perf_counter()
ppo_net, ppo_returns, ppo_losses = train_ppo(
    "CartPole-v1",
    total_timesteps=80000,
    n_steps=512, n_epochs=4, batch_size=64,
    gamma=0.99, lam=0.95, clip_eps=0.2,
    lr=3e-4, c1=0.5, c2=0.01,
)
t_ppo = time.perf_counter() - t0

print(f"  PPO training: {len(ppo_returns)} episodes in {t_ppo:.1f}s")
print(f"  Hyperparameters: clip_ε=0.2, λ=0.95, n_steps=512, n_epochs=4")
print()

ppo_arr = np.array(ppo_returns)
for label, chunk in [
    ("First 50 ep", ppo_arr[:50] if len(ppo_arr)>=50 else ppo_arr),
    ("Last  50 ep", ppo_arr[-50:] if len(ppo_arr)>=50 else ppo_arr),
]:
    print(f"    {label}: mean={chunk.mean():.1f}, "
          f"std={chunk.std():.1f}, max={chunk.max():.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Algorithm comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Algorithm comparison benchmark")
print("━" * 65)
print()

def final_performance(returns_arr, last_n=100):
    if len(returns_arr) < last_n:
        arr = returns_arr
    else:
        arr = returns_arr[-last_n:]
    return arr.mean(), arr.std(), arr.max()

print(f"  Algorithm performance on CartPole-v1 (last 100 episodes):")
print(f"  {'Algorithm':<20} {'Mean':>8} {'Std':>8} {'Max':>8} {'Episodes':>10}")
print(f"  {'─'*58}")

results = [
    ("REINFORCE+baseline", ret_base,     500),
    ("REINFORCE (no base)", ret_nobase,   500),
    ("Actor-Critic",        ac_ret,       len(ac_ret)),
    ("PPO-Clip",            ppo_arr,      len(ppo_arr)),
]
for name, ret, n_ep in results:
    if len(ret) > 0:
        mean, std, mx = final_performance(ret)
        print(f"  {name:<20} {mean:>8.1f} {std:>8.1f} {mx:>8.1f} {n_ep:>10}")
print()

print(f"  GAE λ sensitivity (PPO, last-ep mean return impact):")
print(f"  λ = 0.0 → TD(0):  low variance, high bias")
print(f"  λ = 0.5 → middle:  balanced")
print(f"  λ = 0.95 → standard PPO: near-MC advantage")
print(f"  λ = 1.0 → MC:     high variance, no bias")
print()

print(f"  PPO TRAINING LOOP SUMMARY:")
print(f"  ┌─────────────────────────────────────────────────────────────────┐")
print(f"  │ Phase          │ What happens                                   │")
print(f"  ├─────────────────────────────────────────────────────────────────┤")
print(f"  │ Rollout        │ Run π_old for N steps, collect (s,a,r,s',done) │")
print(f"  │ GAE            │ Compute advantages using λ-weighted TD errors  │")
print(f"  │ Normalise      │ Adv = (Adv - mean) / std per minibatch         │")
print(f"  │ K-epoch update │ Optimise L^CLIP + L^VF + L^ENT for K epochs    │")
print(f"  │ θ_old ← θ     │ Update reference policy for next iteration      │")
print(f"  └─────────────────────────────────────────────────────────────────┘")
print()
print(f"  WHY PPO WORKS WELL IN PRACTICE:")
print(f"    1. Clip prevents destructive large updates (stable)")
print(f"    2. Multiple epochs per rollout (sample efficient)")
print(f"    3. Shared network for policy + value (fast)")
print(f"    4. GAE balances bias-variance automatically (λ=0.95)")
print(f"    5. Entropy bonus prevents premature convergence (exploration)")
print(f"    6. No replay buffer needed (simple implementation)")
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