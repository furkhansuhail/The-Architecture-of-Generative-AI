"""Module 00 · Reinforcement Learning — Architecture Reference Table"""

"""
Reinforcement Learning — Complete Algorithm Reference (ASIC Table)
==================================================================

A single-source reference for every major reinforcement learning algorithm,
method, and concept. Each entry follows the ASIC format:

    A — Algorithm / Architecture name and family
    S — Summary (what problem it solves and how it works)
    I — Input / Output types
    C — Characteristics (assumptions, strengths, and limitations)

Use this as a map before diving into any individual algorithm module.
"""

import os
import re
import textwrap

TOPIC_NAME    = "Learning Path: Bandits to World Models"
DISPLAY_NAME  = "00 · Reinforcement Learning Path"
ICON          = "🤖"
SUBTITLE      = "Algorithm · Summary · Input/Output · Characteristics — the complete reinforcement learning map"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
What is Reinforcement Learning?

Reinforcement Learning (RL) is the branch of machine learning in which an agent learns to make
decisions by interacting with an environment. Unlike supervised learning (labelled examples)
or unsupervised learning (finding structure), RL learns from the consequences of actions —
a scalar reward signal that arrives after each step.

The agent's goal: find a policy π that maximises expected cumulative (discounted) reward.

Core loop:

    At each time step t:
        1. Agent observes state sₜ from the environment
        2. Agent selects action aₜ ~ π(·|sₜ)
        3. Environment transitions to sₜ₊₁ ~ P(·|sₜ, aₜ)
        4. Agent receives reward rₜ = R(sₜ, aₜ, sₜ₊₁)
        5. Agent updates its policy / value estimates

Key quantities:
    * Return Gₜ         — cumulative discounted reward: Gₜ = Σ γᵏ rₜ₊ₖ₊₁  (k=0..)
    * State value V(s)  — expected return from state s under policy π
    * Action value Q(s,a) — expected return from taking action a in state s
    * Advantage A(s,a)  — how much better a is than the average:  A(s,a) = Q(s,a) − V(s)
    * Policy π(a|s)     — probability of taking action a in state s

Key enabling components:
    * Bellman equations     — recursive decomposition of value functions
    * Temporal Difference   — bootstrapped value learning from incomplete episodes
    * Policy gradients      — direct gradient-based policy optimisation
    * Function approximation — neural networks to generalise across large state spaces
    * Exploration strategies — mechanisms to discover better behaviours

The ASIC Table below maps every major RL family across four dimensions:
    A — Algorithm         what the method is called and which family it belongs to
    S — Summary           what problem it solves and the core idea behind it
    I — Input / Output    what it takes as input and what it produces
    C — Characteristics   key assumptions, strengths, and limitations

## Basic simple breakdown

RL is learning by doing. An agent sits in a loop:

It looks at the world (the state)
It takes an action
The world gives it a reward signal (good or bad) and moves to a new state
It uses that signal to update its policy (its strategy)

Repeat millions of times — and a capable agent emerges, without anyone ever telling it what the right answer was.
The key insight that makes it different from supervised learning: there are no labeled examples. A chess engine doesn't 
get told "move the knight here." It just wins or loses, and works backwards to figure out which moves contributed to 
which outcomes.

Where you see it today:

Game AI — AlphaGo, Atari-playing DQN
Robotics — teaching robot arms to grasp objects
LLMs — RLHF (Reinforcement Learning from Human Feedback) is how ChatGPT, Claude and others are fine-tuned to be helpful
Finance — algorithmic trading, portfolio management
Data center cooling — DeepMind cut Google's cooling costs


##### FAMILY 1 — CORE FRAMEWORK: MDPs & BELLMAN EQUATIONS

    The mathematical foundation of nearly all RL. The Markov Decision Process (MDP) formalises
    the agent-environment interaction. The Bellman equations express the recursive relationship
    between values at successive states — they are the backbone of every RL algorithm.


    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | ALGORITHM                            | SUMMARY                                                  | INPUT / OUTPUT                           | CHARACTERISTICS                                                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Markov Decision Process (MDP)        | Formal tuple (S, A, P, R, γ). Assumes the Markov         | State space S, action space A →          | Assumes full observability (Markov property); provides the mathematical              |
    |                                      | property: future depends only on present state.          | Optimal policy π* and value V*           | foundation for all RL; transition function P may be unknown                          |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Partially Observable MDP (POMDP)     | Extension of MDP where the agent cannot directly         | Observations o, belief state b(s) →      | More realistic than MDP; belief state must be maintained; exact                      |
    |                                      | observe the true state; receives observations instead.   | Policy over belief states                | solutions intractable for large problems; approximated via RNNs or filters           |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Bellman Expectation Equation         | Expresses V^π(s) = Σₐ π(a|s) [R(s,a) + γ Σₛ' P V^π(s')]   | Policy π, MDP model →                    | Basis for policy evaluation; system of linear equations for finite MDPs;             |
    |                                      | Recursively defines the value of a state under π.        | State-value function V^π                 | can be solved exactly when model is known; too large for most real problems          |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Bellman Optimality Equation          | V*(s) = maxₐ [R(s,a) + γ Σₛ' P(s'|s,a) V*(s')]            | MDP model →                              | Defines the optimal value function; nonlinear due to max operator;                   |
    |                                      | Characterises the unique fixed point of the optimal π.   | Optimal value V*, optimal policy π*      | solved by dynamic programming when model is known; basis for Q-learning              |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Reward Function R(s, a, s')          | Scalar signal specifying what the agent should maximise. | (state, action, next state) →            | Design is critical — misspecified rewards cause reward hacking; sparse               |
    |                                      | Can be sparse, dense, shaped, or learned from data.      | Scalar reward                            | rewards are hard to learn from; dense rewards guide faster but can mislead           |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Discount Factor γ                    | Weights future rewards: return = Σ γᵏ rₜ₊ₖ.                | Hyperparameter ∈ [0, 1) →                | γ → 0 = myopic agent; γ → 1 = far-sighted; must be < 1 for infinite                 |
    |                                      | Controls the trade-off between immediate and future r.   | Controls effective planning horizon      | horizon convergence; affects exploration-exploitation balance                        |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Policy π(a|s)                        | The agent's decision rule — maps states to a             | State s →                                | Deterministic policies: a = π(s); stochastic: a ~ π(·|s);                            |
    |                                      | distribution over actions (or a single action).          | Action a (or distribution over actions)  | stochastic policies needed for exploration and in game-theoretic settings            |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+


##### FAMILY 2 — DYNAMIC PROGRAMMING

    Dynamic programming (DP) methods assume a perfect model of the environment (P and R are known).
    They use the Bellman equations as update rules to compute exact value functions and optimal policies.
    Not directly applicable when the model is unknown — but provide the theoretical foundation for all RL.


    ALGORITHM                       SUMMARY                                                      INPUT / OUTPUT                          CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Policy Evaluation               Iteratively applies the Bellman expectation equation         Policy π, MDP model →                   Converges to V^π in the limit; requires known P and R;
                                    until V^π converges: V(s) ← Σₐ π(a|s)[R + γΣ P V(s')].        State-value function V^π                exact for tabular MDPs; too slow for large state spaces

    Policy Improvement              Greedily updates the policy by acting greedily w.r.t.        Value function V^π →                    Guaranteed not to decrease policy quality; used as the
                                    the current value function: π'(s) = argmax Q(s, a).          Improved policy π'                      improvement step in policy iteration; monotone improvement

    Policy Iteration                Alternates between policy evaluation (compute V^π)           MDP model →                             Converges in finite steps for finite MDPs; inner evaluation
                                    and policy improvement (greedy update), until π is stable.   Optimal policy π*, optimal V*           loop can be expensive; basis for actor-critic methods

    Value Iteration                 Collapses policy evaluation to one sweep and immediately     MDP model →                             Faster than policy iteration in practice; converges to V*;
                                    applies the Bellman optimality update at every step.         Optimal value V*, derived π*            policy implicit: π*(s) = argmax Q*(s,a); simplest DP method

    Generalised Policy Iteration    Unifying view: any interaction between evaluation and        Any (policy, value function) pair →     Captures both policy iteration and value iteration as special
    (GPI)                           improvement converges to the optimal pair (π*, V*).          Optimal (π*, V*)                        cases; intuition behind actor-critic architectures

    Asynchronous DP                 Updates states in any order rather than full sweeps;         MDP model, state priority →             Allows focusing on important states; can handle very large
                                    prioritised sweeping updates high-Bellman-error states.      Value function V                        state spaces more efficiently; basis for prioritised replay


##### FAMILY 3 — MONTE CARLO METHODS

    Monte Carlo (MC) methods learn from complete episodes of experience. They estimate values by
    averaging the actual returns observed — no bootstrapping, no model required.
    Good for episodic tasks; high variance but unbiased estimates.


    ALGORITHM                       SUMMARY                                                       INPUT / OUTPUT                          CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    First-Visit MC                  Estimates V(s) as the average return the first time s         Complete episodes →                     Unbiased; high variance; requires episodic tasks; simple;
                                    is visited in each episode.                                   State-value estimates V(s)              convergence guaranteed; does not bootstrap from V estimates

    Every-Visit MC                  Averages the return every time s is visited in an episode     Complete episodes →                     Biased for finite samples but consistent; often lower variance
                                    (not just the first visit).                                   State-value estimates V(s)              than first-visit; equally commonly used in practice

    MC Control (ε-soft policies)    Combines MC evaluation with ε-greedy policy improvement       Complete episodes →                     Explores with probability ε; converges to near-optimal policy;
                                    to learn Q(s,a) without a model.                              Action-value estimates Q(s,a)           can be slow for large state/action spaces

    Off-Policy MC with              Uses importance sampling ratio to correct for the             Behaviour policy b, target policy π →   Unbiased but very high variance with ordinary IS;
    Importance Sampling             mismatch between the behaviour policy and target policy.      Value estimates for π using b's data    weighted IS lower variance but biased; foundational for off-policy


##### FAMILY 4 — TEMPORAL DIFFERENCE LEARNING

    TD methods combine the sampling idea of MC with the bootstrapping of DP.
    They update estimates after every single step — no need for complete episodes.
    This makes them applicable to continuing tasks and much more sample efficient than MC.


    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | ALGORITHM                            | SUMMARY                                                  | INPUT / OUTPUT                           | CHARACTERISTICS                                                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | TD(0)                                | Simplest TD: updates V(s) after each step using the      | Single transition (s,r,s') →             | Biased but lower variance than MC; converges to V^π under tabular, linear            |
    |                                      | TD target: r + γV(s'). One-step bootstrapping.           | Updated V(s)                             | approximation; the foundation of nearly all RL algorithms                            |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | TD(λ) / Eligibility Traces           | Blends TD(0) and MC via a λ parameter. Eligibility       | Episode experience, λ ∈ [0,1] →          | λ=0 → TD(0), λ=1 → MC; intermediate λ often best; traces assign credit               |
    |                                      | traces credit states proportional to their recency.      | Updated value function                   | backwards in time; forward and backward views are equivalent                         |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | SARSA (on-policy TD)                 | On-policy TD control. Updates Q(s,a) using the actual    | (s, a, r, s', a') tuples →               | On-policy: learns Q for the policy being followed including exploration;             |
    |                                      | next action a' taken: Q ← Q + α[r + γQ(s',a') − Q].      | Action-value Q(s,a)                      | safe in cliff-walking problems; converges with decaying ε-greedy                     |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Q-Learning (off-policy TD)           | Off-policy TD control. Updates Q using the greedy action | (s, a, r, s') tuples →                   | Off-policy: learns optimal Q regardless of exploration policy; the most              |
    |                                      | regardless of what was actually taken: max_a' Q(s', a'). | Optimal action-value Q*(s,a)             | widely used tabular RL algorithm; can diverge with function approximation            |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Expected SARSA                       | Like SARSA but uses the expected value over all next     | (s, a, r, s') tuples →                   | Lower variance than SARSA; more computationally expensive per step;                  |
    |                                      | actions under π: Σₐ' π(a'|s') Q(s', a').                  | Action-value Q(s,a)                      | generalises Q-learning (greedy π) and SARSA (same π)                                 |
    +--------------------------------------+-----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Double Q-Learning                    | Uses two separate Q-networks to decouple action           | (s, a, r, s') tuples →                   | Reduces overestimation bias of standard Q-learning; often improves stability;        |
    |                                      | selection from value estimation, reducing overestimation. | Two Q-function estimates                 | one network selects the action, the other evaluates it                               |
    +--------------------------------------+-----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | SARSA(λ)                             | SARSA with eligibility traces — propagates TD errors      | Episode experience, λ →                  | Credits all recently visited (s,a) pairs; faster propagation of reward signal;       |
    |                                      | to all recent state-action pairs via λ-weighted traces.   | Updated Q(s,a) with trace memory         | particularly effective in long-horizon tasks; watkins Q(λ) is off-policy version     |
    +--------------------------------------+-----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+


##### FAMILY 5 — DEEP VALUE-BASED METHODS

    Extend tabular Q-learning to high-dimensional state spaces (e.g. raw pixels) by replacing
    the Q-table with a deep neural network. Key innovations: experience replay (breaks correlations)
    and target networks (stabilise training). The DQN family is the dominant approach for
    discrete-action problems with visual observations.


    ALGORITHM                       SUMMARY                                                      INPUT / OUTPUT                          CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Deep Q-Network (DQN)            Approximates Q*(s,a) with a CNN. Uses a replay buffer        Raw pixels / state vector →             Achieved human-level Atari performance (Mnih 2015); unstable
                                    and a periodically updated target network for stability.     Q-values for all actions                without replay + target net; discrete actions only

    Double DQN                      Decouples action selection from evaluation: online net       Raw pixels / state vector →             Reduces systematic overestimation of Q values; often more
                                    picks the action, target net evaluates it.                   Improved Q-values for all actions       stable than standard DQN; simple drop-in modification

    Dueling DQN                     Splits Q(s,a) = V(s) + A(s,a) using two network              Raw pixels / state vector →             Better estimates of V(s) even for unimportant actions;
                                    streams (value stream + advantage stream) merged at output.  Q-values with shared backbone           particularly beneficial when many actions have similar values

    Prioritised Experience Replay   Samples transitions from the replay buffer with              Experience buffer with TD-error prioriy → Focuses learning on surprising / high-error transitions;
    (PER)                           probability proportional to their TD error magnitude.        Faster convergence of Q-values          corrected with importance sampling weights to remove bias

    Noisy Networks (NoisyNet)       Replaces linear layers with noisy linear layers              Noisy network weights →                 Learnable, state-dependent exploration; replaces ε-greedy;
                                    (μ + σ⊙ε) for parametric exploration.                        Stochastic Q-values                     noise fades as training progresses; compatible with any DQN

    C51 / Categorical DQN           Models the full return distribution Z(s,a) as a              State →                                 First distributional RL algorithm; learns 51-atom distribution;
                                    categorical distribution over fixed support atoms.           Distribution over returns               captures risk and multimodality; Bellman update via projection

    Quantile Regression DQN         Represents return distribution implicitly via quantiles.     State →                                 No fixed support needed; learns arbitrary distribution shapes;
    (QR-DQN)                        Minimises the quantile Huber loss.                           Quantile function of Z(s,a)             outperforms C51 on most benchmarks; cleaner formulation

    Implicit Quantile Network (IQN)  Samples random quantile levels τ ~ U[0,1] and learns        State, τ ~ U[0,1] →                     Expressive distributional RL; risk-sensitive policies possible;
                                    the corresponding quantile values via an implicit network.   Z_τ(s,a) (conditional quantile)         better performance than QR-DQN; single network for all τ

    Rainbow                         Combines 6 DQN improvements: Double DQN, Dueling,            Raw pixels →                            State-of-the-art on Atari at publication; ablations show all
                                    PER, NoisyNets, C51, and Multi-step returns.                 Distributional Q across actions         components contribute; establishes a strong baseline for DRL

    Multi-step DQN / n-step returns  Uses n-step returns for TD target instead of 1-step:        Experience buffer →                     Propagates reward signal faster; reduces bias at cost of
                                    rₜ + γrₜ₊₁ + ... + γⁿ⁻¹rₜ₊ₙ₋₁ + γⁿ V(sₜ₊ₙ).                         Better Q-values for long horizons       variance; n is a hyperparameter; used in Rainbow and Ape-X

    Ape-X                           Distributed DQN: many actors collect experience with         Distributed actors + central learner →  Massive parallelism; actors use different ε for diversity;
                                    different exploration policies; one learner trains.          Very high sample throughput             central replay with priority; achieves very fast wall-clock time


##### FAMILY 6 — POLICY GRADIENT METHODS

    Policy gradient methods directly parameterise and optimise the policy π_θ by gradient ascent
    on the expected return. They work for continuous and discrete action spaces, handle stochastic
    policies naturally, and are the dominant approach for robotics and continuous control.
    Key challenge: high variance of gradient estimates.


    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | ALGORITHM                            | SUMMARY                                                  | INPUT / OUTPUT                           | CHARACTERISTICS                                                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | REINFORCE (Monte Carlo PG)           | Computes the policy gradient as ∇θ log π_θ(a|s) · Gₜ.     | Complete episodes →                      | Unbiased but very high variance; slow convergence; foundational result;              |
    |                                      | Uses full-episode returns as the learning signal.        | Updated policy parameters θ              | impractical without a baseline; Williams (1992)                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | REINFORCE with Baseline              | Subtracts a state-dependent baseline b(s) from the       | Complete episodes, baseline b(s) →       | Reduces variance without introducing bias (if baseline independent of a);            |
    |                                      | return: ∇θ log π_θ(a|s) · (Gₜ − b(s)).                    | Updated policy parameters θ              | b(s) = V(s) is the standard choice; significant practical improvement                |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Actor-Critic (A2C)                   | Actor updates π_θ using advantage A(s,a) = r+γV(s')−V(s).| Transitions (s,a,r,s') →                 | Lower variance than REINFORCE; one-step updates (no full episode needed);            |
    |                                      | Critic learns V_φ(s) to compute the TD baseline.         | Policy θ, value function φ               | bias from approximate critic; workhorse of modern deep RL                            |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Asynchronous Advantage               | Runs many actor-learners in parallel on CPU;             | Parallel environments →                  | Efficient CPU-based training; asynchronous updates decorrelate experience;           |
    | Actor-Critic (A3C)                   | global parameter server accumulates async gradients.     | Shared policy and value network          | fast wall-clock time; replaced by synchronous A2C in practice; Mnih 2016             |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Natural Policy Gradient (NPG)        | Uses Fisher information matrix to precondition gradient: | Policy gradient, Fisher matrix F →       | Invariant to policy parameterisation; accounts for geometry of probability           |
    |                                      | Δθ = F⁻¹ ∇θ J. Moves equal distance in distribution space.| Updated policy in natural space          | manifold; computationally expensive to invert F; basis for TRPO                     |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Trust Region Policy Optimisation     | Maximises policy improvement subject to a KL divergence  | Old policy πold, advantage estimates →   | Monotonic improvement guarantee (theoretically); complex implementation              |
    | (TRPO)                               | constraint: KL[π_old || π_new] ≤ δ. Uses conjugate grad. | Updated policy with bounded KL           | with conjugate gradient + line search; largely superseded by PPO                    |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Proximal Policy Optimisation (PPO)   | Clips the probability ratio r_t(θ) = π_θ/π_θold          | Rollout data (s,a,r,A) →                 | Simple, robust, widely used; no conjugate gradient; strong empirical                |
    |                                      | to [1-ε, 1+ε], preventing large policy updates.          | Updated policy and value network         | performance; PPO-Clip and PPO-Penalty variants; default algorithm for RLHF          |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Deep Deterministic Policy Gradient   | Off-policy actor-critic for continuous actions.          | Continuous state + action →              | Deterministic policy; off-policy (sample efficient); sensitive to                    |
    | (DDPG)                               | Actor outputs deterministic μ(s); critic Q(s,a).         | Continuous action μ(s)                   | hyperparameters; target networks + replay buffer; Lilicrap 2016                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Twin Delayed DDPG (TD3)              | Three improvements on DDPG: (1) two critics (take min),  | Continuous state + action →              | More stable than DDPG; reduces overestimation; delayed updates stabilise             |
    |                                      | (2) delayed policy updates, (3) target policy smoothing. | Continuous action (deterministic)        | training; strong baseline for continuous control; Fujimoto 2018                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Soft Actor-Critic (SAC)              | Off-policy actor-critic that maximises entropy-augmented | Continuous state + action →              | State-of-the-art continuous control; automatic temperature tuning;                   |
    |                                      | reward: J = E[Σ γᵏ(r + α H(π))]. Entropy regularisation. | Stochastic continuous policy             | sample efficient; robust to hyperparameters; handles multi-modal actions             |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Deterministic Policy Gradient (DPG)  | Theoretical basis for DDPG. Policy gradient theorem for  | Continuous state + action →              | Off-policy update is efficient; deterministic policies have zero entropy;            |
    |                                      | deterministic policies: ∇θ J = E[∇θ μ(s) ∇ₐ Q(s,a)].      | Gradient for deterministic actor         | Silver 2014; exploration must be injected separately via noise                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Maximum a-Posteriori Policy          | Frames policy search as an EM problem with a KL          | State distribution, advantage →          | Principled EM-style updates; strong performance on locomotion; separates             |
    | Optimisation (MPO)                   | constraint between old and new policy in E and M steps.  | Updated policy parameters                | sample collection (E-step) from policy fitting (M-step); DeepMind 2018               |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+


##### FAMILY 7 — MODEL-BASED REINFORCEMENT LEARNING

    Model-based RL learns or uses a model of the environment (transition dynamics P and reward R)
    to plan ahead or generate synthetic experience. This dramatically improves sample efficiency
    but introduces model bias — errors in the model propagate to the policy.


    ALGORITHM                       SUMMARY                                                       INPUT / OUTPUT                          CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Dyna-Q                          Integrates model learning and planning with Q-learning.       Real transitions + learned model →      Simple and elegant; n additional simulated updates per real step;
                                    After each real step, n planning steps use the learned model. Q-values improved by simulation        foundational model-based algorithm; Sutton 1990

    Dyna-Q+                         Extends Dyna-Q with an exploration bonus for transitions      Real transitions, recency bonus →       Addresses stale model problem; encourages re-visiting less-explored
                                    that have not been tried for a long time (recency bonus).     Q-values + exploration bonus            transitions; useful in non-stationary environments

    PILCO                           Gaussian Process-based model + analytic policy gradient.      Real transitions →                      Extremely sample efficient (can solve tasks in seconds);
                                    Propagates uncertainty through the model exactly.             Probabilistic trajectory predictions    exact GP inference scales poorly; limited to low-dimensional spaces

    World Model (Ha & Schmidhuber)  Learns a compact latent world model (V-MDN, M-LSTM, C).       Raw observations →                      Agent trained entirely inside the dream; generative vision model;
                                    Policy (controller) trained in imagined rollouts.             Imagined trajectories + policy          can overfit to the model; separates perception from control

    PlaNet                          Learns a recurrent state-space model (RSSM) in latent         Raw images →                            Highly sample efficient; latent planning via cross-entropy method;
                                    space. Plans by optimising action sequences with CEM.         Action sequences minimising cost        learns dynamics model in pixel space; Hafner 2019

    Dreamer / DreamerV2 / V3        Extends PlaNet with actor-critic trained on latent            Raw images / observations →             State-of-the-art model-based; learns long-horizon behaviours in
                                    imagination. Backpropagates through imagined trajectories.    Policy + value function in latent space latent space; DreamerV3 works across domains with fixed HPs

    MCTS (Monte Carlo Tree Search)  Builds a lookahead tree by simulating trajectories.           Tree search policy → Best action        Strong when a model is available; used in AlphaGo, AlphaZero;
                                    Uses UCB to balance exploration vs exploitation in the tree.                                         computationally expensive; less suited to continuous domains

    AlphaGo / AlphaZero             Combines MCTS with a value network and policy network.       Board state →                           Superhuman in Go, Chess, Shogi from self-play alone (Zero);
                                    Self-play generates training data; no human knowledge.       Policy + value for tree search          requires massive compute; fixed discrete action space

    MuZero                          Extends AlphaZero to unknown environments. Learns a          Observations (no rules needed) →        First model-based method to match DQN on Atari without knowing
                                    latent dynamics model jointly with value and policy.         Policy + value + dynamics model         game rules; bridges model-based and model-free deep RL

    Ensemble Methods (PETS, MBPO)   Uses an ensemble of neural network dynamics models to        Transitions →                           Quantifies epistemic uncertainty via disagreement; MBPO uses
                                    quantify uncertainty and guide exploration / policy updates. Policy with uncertainty-aware planning  short model rollouts for on-policy updates; competitive with SAC


##### FAMILY 8 — MULTI-ARMED BANDITS & EXPLORATION

    Multi-armed bandits (MABs) are a simplified RL problem: a single state, K actions (arms),
    and the goal of maximising cumulative reward. They isolate the exploration-exploitation tradeoff.
    Many bandit algorithms generalise to full RL as exploration strategies.


    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | ALGORITHM                            | SUMMARY                                                  | INPUT / OUTPUT                           | CHARACTERISTICS                                                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | ε-Greedy                             | Select greedy action with probability 1−ε;               | Q-value estimates, ε ∈ [0,1] →           | Simple and widely used; ε-decay schedules improve performance;                       |
    |                                      | explore uniformly at random with probability ε.          | Action (exploit or random)               | wastes exploration budget on suboptimal arms uniformly                               |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Upper Confidence Bound (UCB1)        | Select arm with highest Q(a) + c√(ln t / N(a)).          | Q-estimates, visit counts N(a) →         | Logarithmic regret (optimal for stochastic bandits); confidence bonus                |
    |                                      | Optimism in the face of uncertainty principle.           | Action with highest UCB score            | automatically decays; c controls exploration strength                                |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Thompson Sampling                    | Maintains Beta(α, β) posterior over each arm's reward.   | Prior distributions →                    | Bayesian; often best empirical performance; naturally adapts exploration              |
    |                                      | Samples from posteriors and selects argmax.              | Action (sampled from posterior)          | to uncertainty; generalises to Gaussian and more complex posteriors                  |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Boltzmann / Softmax Exploration      | Selects action a with probability ∝ exp(Q(a)/τ).         | Q-value estimates, temperature τ →       | Temperature τ controls exploration: τ→∞ = uniform, τ→0 = greedy;                    |
    |                                      | Temperature τ controls exploration randomness.           | Stochastic action selection              | calibrated to value scale; can be inefficient in high-dimensional action spaces      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Linear Bandits / LinUCB              | Assumes reward is linear in a feature vector: r = θᵀx.   | Context vector x, feature space →        | Efficient contextual bandits; used in recommendation systems; closed-form            |
    |                                      | Maintains ridge-regression estimate of θ with UCB bonus. | Action with highest linear UCB           | updates; assumes linearity which may not hold in practice                            |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Gaussian Process (GP) Bandits        | Models reward function as a GP. Uses GP posterior        | Observations, GP kernel →                | Principled uncertainty; used in Bayesian optimisation; expensive (O(n³))             |
    |                                      | mean + variance to form UCB acquisition function.        | Action maximising GP-UCB                 | computationally; works well for continuous action spaces with few evaluations        |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Posterior Sampling for RL (PSRL)     | Samples a full MDP from a posterior at each episode;     | Posterior over MDPs →                    | Bayesian optimal exploration in the limit; excellent sample efficiency;              |
    |                                      | acts optimally under the sampled MDP for that episode.   | Policy optimised for sampled MDP         | intractable for complex environments; approximated via ensemble methods              |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+


##### FAMILY 9 — EXPLORATION STRATEGIES (DEEP RL)

    In deep RL, naive ε-greedy exploration is often insufficient — especially with sparse rewards
    or hard exploration problems (Montezuma's Revenge). These methods use novelty, uncertainty,
    or learned internal motivation to drive effective exploration.


    ALGORITHM                       SUMMARY                                                         INPUT / OUTPUT                          CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Count-Based Exploration          Adds exploration bonus inversely proportional to visit         State visit counts N(s) →               Theoretically motivated; works well in tabular settings;
                                    count: r+ = β / √N(s). Approximate counts for large spaces.     Modified reward with novelty bonus      approximate counting (hash / pseudocounts) needed for DRL

    Intrinsic Curiosity Module (ICM)  Trains a forward model and inverse model. Prediction          Observations, actions →                 Encourages visiting novel states via prediction error; works in
                                    error of the forward model becomes an intrinsic reward.         Intrinsic reward + policy               Montezuma's Revenge; can get distracted by stochastic elements

    Random Network Distillation (RND) Fixed random target network; predictor tries to match it.     Observations →                          Deterministic novelty signal; not distracted by environmental
                                    High prediction error on novel states gives intrinsic reward.   Intrinsic reward (prediction error)     stochasticity; computationally cheap; used with PPO; OpenAI 2018

    Never Give Up (NGU)              Combines episodic (within-episode) and life-long novelty       Observations + episodic memory →        Separate controllers for different exploration levels;
                                    signals with a mixture of policies (different β weights).       Exploration-focused mixed policy        strong on hard Atari; computationally expensive; DeepMind 2020

    Agent57                          Adds adaptive meta-controller to NGU that selects between      Observations, sliding window stats →    First agent to beat human on all 57 Atari games; handles
                                    different behavioural policies (exploration vs. exploitation).  Super-human policy on all Atari         very long-horizon exploration; very high compute cost

    Go-Explore                       Phase 1: returns to promising states from an archive,          State archive →                         Separates exploration from exploitation; highly effective on
                                    explores from there. Phase 2: robustify with imitation.         Policy from archive-based exploration   Pitfall and Montezuma; requires a resettable simulator

    Noisy Networks (NoisyNet)        Adds learned noise parameters to network weights.              Noisy parameters (μ + σ⊗ε) →            State-dependent exploration; replaces ε-greedy; compatible
                                    Policy is stochastic via weight noise rather than action noise. Stochastic policy                       with any value-based or policy-gradient algorithm; Fortunato 2017


##### FAMILY 10 — MULTI-AGENT REINFORCEMENT LEARNING (MARL)

    In MARL, multiple agents interact in a shared environment simultaneously. The environment
    becomes non-stationary from each agent's perspective. Settings range from fully cooperative
    (shared reward) to fully competitive (zero-sum) to mixed. Centralised Training with
    Decentralised Execution (CTDE) is the dominant paradigm.


    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | ALGORITHM                            | SUMMARY                                                  | INPUT / OUTPUT                           | CHARACTERISTICS                                                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Independent Q-Learning (IQL)         | Each agent runs independent Q-learning; treats other     | Each agent's local observations →        | Simple baseline; ignores other agents; non-stationary environment from each          |
    |                                      | agents as part of the environment.                       | Per-agent Q-values and policy            | agent's view; can work surprisingly well empirically despite no guarantees           |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | QMIX                                 | Centralised training; mixes per-agent Q-values via a     | Local Q-values, global state →           | Monotonic mixing ensures decentralised execution; strong cooperative MARL;           |
    |                                      | monotone mixing network into a global Q for training.    | Global Q for training, local Q for exec  | IGM principle; cannot represent all cooperative strategies; Rashid 2018             |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | MADDPG                               | Extends DDPG to multi-agent: centralised critics that    | All agents' observations + actions →     | Works for cooperative and competitive tasks; critics observe all agents;             |
    | (Multi-Agent DDPG)                   | observe all agents; decentralised actors for execution.  | Per-agent continuous policies            | scales poorly with number of agents; Lowe 2017                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | COMA (Counterfactual Multi-Agent PG) | Centralised critic with counterfactual baseline:         | Joint observations, joint actions →      | Reduces credit assignment problem; counterfactual measures each agent's              |
    |                                      | A(sᵢ, aᵢ) = Q(s, a) − Σ π(a'ᵢ|sᵢ) Q(s, (a'ᵢ, a₋ᵢ)).      | Per-agent advantage estimates            | marginal contribution; on-policy; expensive to compute; Foerster 2018              |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | MAPPO                                | Multi-agent extension of PPO with a centralised critic.  | Local obs (actors) + global state →      | Simple, scalable; strong competitive baseline; works in cooperative and mixed        |
    |                                      | Shares value function across agents during training.     | Per-agent policies (decentralised exec)  | settings; Yu 2021 showed surprisingly strong results vs. specialised MARL methods  |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Mean Field RL                        | Approximates multi-agent interactions via the mean field | Local obs + mean action of neighbours →  | Scales to very large agent populations; tractable computation;                       |
    |                                      | of neighbouring agents' action distributions.            | Per-agent policy based on mean field     | approximation quality depends on agent homogeneity; Yang 2018                       |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Self-Play                            | Agent trains by playing against past versions of itself. | Environment with opponent →              | Generates auto-curriculum; used in AlphaGo, OpenAI Five, Pluribus;                  |
    |                                      | League training extends to diverse set of past policies. | Policy that improves vs. itself          | can converge to Nash in zero-sum games; population diversity prevents cycling       |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+


##### FAMILY 11 — INVERSE RL & IMITATION LEARNING

    Instead of designing a reward function by hand, inverse RL (IRL) infers it from expert
    demonstrations. Imitation learning methods learn the policy directly from demonstrations
    without explicitly recovering a reward. Crucial for robotics, autonomous driving, and
    any domain where reward specification is difficult.


    ALGORITHM                       SUMMARY                                                      INPUT / OUTPUT                          CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Behavioural Cloning (BC)         Treats imitation as supervised learning: train π_θ(a|s)     Expert demonstrations (s, a) →          Simple and fast; distribution shift at test time (compounding
                                    to maximise likelihood of expert actions.                    Cloned policy π_θ                       errors); performance degrades outside training distribution

    DAgger (Dataset Aggregation)    Interactive imitation: rolls out learner policy, queries     Learner policy + expert labeller →      Reduces distribution shift by including on-policy states;
                                    expert for labels on visited states, aggregates dataset.     Improved policy with diverse states     requires online access to expert; Ross & Bagnell 2011

    Maximum Entropy IRL (MaxEntIRL)  Infers reward R(s,a) such that expert demonstrations are    Expert demonstrations →                 Principled probabilistic framework; handles suboptimal
                                    the maximum entropy distribution over trajectories.          Inferred reward function R              demonstrations; requires solving RL as inner loop; Ziebart 2008

    Guided Cost Learning (GCL)       Iteratively refines reward by training cost function to     Demonstrations + sampled trajectories → Uses importance sampling to avoid inner RL loop; trains reward
                                    distinguish demonstrations from sampled trajectories.        Reward function + policy                and policy jointly; Finn 2016; connects to GAN training

    Generative Adversarial           Frames imitation as occupancy measure matching.             Expert demonstrations →                  No explicit reward; discriminator distinguishes expert vs.
    Imitation Learning (GAIL)        Discriminator: expert vs. agent; generator: policy.         Policy matching expert behaviour        learner trajectories; sample efficient; Ho & Ermon 2016

    Adversarial IRL (AIRL)           Disentangles reward from dynamics using a structured        Expert demonstrations →                  Transferable reward function (not entangled with dynamics);
                                    discriminator: D(s,a,s') = exp(f(s,a,s')) / (exp(f)+π).      Transferable reward + policy            enables reward transfer to new environments; Fu 2018

    Reward Modelling                 Trains a reward model R_θ from human preference labels.     Trajectory pairs + human labels →       Core component of RLHF; reward model captures human intent;
                                    Human picks which of two trajectories is better.             Scalar reward model R_θ                 reward model can be misspecified; Christiano 2017


##### FAMILY 12 — HIERARCHICAL REINFORCEMENT LEARNING

    Hierarchical RL organises decision-making across multiple levels of abstraction and timescale.
    High-level policies set subgoals or select skills; low-level policies execute them.
    Addresses the long-horizon credit assignment problem and encourages skill reuse.


    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | ALGORITHM                            | SUMMARY                                                  | INPUT / OUTPUT                           | CHARACTERISTICS                                                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Options Framework                    | Defines options as (I, π, β): initiation set I,          | MDP + option definitions →               | Theoretical foundation for HRL; options can be learnt or hand-designed;              |
    |                                      | intra-option policy π, and termination condition β.      | Policy over options + intra-option policy | temporal abstraction at multiple timescales; Sutton 1999                             |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Feudal Networks (FuN)                | Manager sets goals in latent space every c steps.        | Observations →                           | Manager operates at longer timescales; workers rewarded for moving towards            |
    |                                      | Worker tries to reach those goals with primitive actions.| Goal vector → primitive action           | goals; learns without extrinsic reward on goal achievement; Vezhnevets 2017         |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | HIRO (Data-Efficient HRL)            | Off-policy HRL with goal-conditioned lower policy.       | State, high-level goal →                 | Uses off-policy correction for the high-level policy; sample efficient;              |
    |                                      | High-level relabels goals using hindsight for stability. | Subgoal + primitive action               | hindsight goal relabelling prevents non-stationarity; Nachum 2018                   |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | HAC (Hierarchical Actor-Critic)       | Multi-level actor-critic trained jointly with hindsight.| Multi-level goals →                      | Three-level hierarchy; each level learns independently; hindsight transitions        |
    |                                      | Each level penalises subgoal failures during training.   | Actions at each hierarchy level          | for subgoal relabelling; Levy 2019; supports arbitrary hierarchy depth               |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Skill Discovery (DIAYN, VALOR)        | Discovers diverse, distinguishable skills without reward| Observations, latent skill z →           | Unsupervised skill pre-training; downstream tasks use discovered skills;             |
    |                                      | Maximises mutual information I(S; Z) between state and z. | Diverse skill library                    | DIAYN uses discriminator; VALOR uses causal InfoMax; Eysenbach 2019                 |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+


##### FAMILY 13 — OFFLINE / BATCH REINFORCEMENT LEARNING

    Offline RL learns a policy from a fixed pre-collected dataset without any further interaction
    with the environment. Critical for domains where online data collection is expensive or risky
    (healthcare, robotics). Key challenge: distribution shift and out-of-distribution (OOD) actions.


    ALGORITHM                       SUMMARY                                                        INPUT / OUTPUT                          CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Batch-Constrained Q-Learning    Constrains the policy to actions present in the dataset:       Offline dataset D →                     Prevents OOD action extrapolation; VAE generates candidate actions;
    (BCQ)                           generates actions via VAE + perturbation, selects max Q.       Policy π with dataset support           strong baseline; effective on diverse offline datasets; Fujimoto 2019

    Conservative Q-Learning (CQL)   Adds a penalty to Q-function training that minimises Q         Offline dataset D →                     Underestimates Q for OOD actions; no need to explicitly model
                                    for OOD actions while maximising it for in-dataset actions.    Conservative Q-function + policy       behaviour policy; strong results; hyperparameter sensitivity; Kumar 2020

    TD3+BC                          Combines TD3 with a behavioural cloning regularisation         Offline dataset D →                     Simple and effective; single hyperparameter α balances BC vs TD3;
                                    term in the actor loss: π = argmax[λ Q − (a − π_BC)²].         Policy close to BC but Q-improved       strong baseline; Fujimoto & Gu 2021; easy to implement

    Implicit Q-Learning (IQL)        Avoids OOD actions entirely by fitting Q in-sample only.      Offline dataset D →                     Advantage-weighted regression for policy extraction; no OOD query
                                    Uses expectile regression to approximate max Q operator.       Q-function without OOD queries          of Q during training; scalable; Kostrikov 2021

    Decision Transformer (DT)        Treats offline RL as a sequence modelling problem.            Offline trajectories (R, s, a) →       No dynamic programming; uses GPT architecture; conditioned on
                                    GPT-style transformer predicts next action given return-to-go. Action conditioned on desired return  desired return (reward conditioning); Chen 2021; scalable

    Trajectory Transformer           Full trajectory modelling using beam search at test time.     Offline trajectories →                  Joint model of states, actions, rewards; flexible planning via
                                    Discretises continuous state-action-reward sequences.          Complete trajectory distribution        beam search; computationally expensive; Janner 2021

    BEAR (Bootstrapping Error        Constrains policy to be close to the behaviour policy via     Offline dataset D →                     Maximum mean discrepancy (MMD) constraint keeps policy in-support;
    Accumulation Reduction)         a support-based constraint using kernel-based MMD.             Policy with support constraint          principled bound on distribution shift; Kumar 2019


##### FAMILY 14 — META-REINFORCEMENT LEARNING

    Meta-RL learns to learn: the agent is trained across a distribution of tasks so that it can
    rapidly adapt to new tasks with only a few environment interactions at test time.
    "Learning an RL algorithm" — the inner loop adapts, the outer loop optimises for fast adaptation.


    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | ALGORITHM                            | SUMMARY                                                  | INPUT / OUTPUT                           | CHARACTERISTICS                                                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | MAML (Model-Agnostic Meta-Learning)  | Finds an initialisation θ* such that a few gradient      | Distribution of tasks →                  | Task-agnostic; works for any gradient-based model; computationally expensive         |
    |                                      | steps on a new task yield a good policy.                 | Initialisation θ* + fast-adapted policy  | (second-order gradients); FOMAML approximation used in practice; Finn 2017          |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | RL² (RL Squared)                     | Encodes the RL algorithm itself inside an LSTM.          | Task episodes →                          | Algorithm is implicit in LSTM weights; fast adaptation via hidden state;             |
    |                                      | The RNN hidden state tracks task-specific information.   | Policy (hidden state encodes task)       | requires training on many related tasks; Duan 2016                                  |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | SNAIL                                | Uses temporal convolutions (dilated) + attention to      | Task episodes →                          | Combines fast aggregation (TCN) with selective attention over past experience;       |
    |                                      | meta-learn the algorithm from task experience.           | Policy conditioned on task history       | strong on few-shot classification and RL; Mishra 2018                               |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | PEARL (Probabilistic Embeddings      | Infers a latent task variable z from context tuples      | Context transitions + task distribution →| Efficient off-policy meta-RL; task posterior inferred online; decouples              |
    | for RL)                              | (s,a,r,s'); conditions policy on posterior q(z|context). | Task-conditioned policy π(a|s,z)         | exploration of z from policy learning; Rakelly 2019                                 |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | MAML-PPO / Meta-PPO                  | Applies MAML outer loop with PPO as the inner-loop       | Task distribution →                      | Combines stability of PPO with meta-learning objective; commonly used in             |
    |                                      | optimiser; adapts policy parameters via PPO gradient.    | Meta-initialised policy                  | locomotion meta-RL; slower than RL² but more interpretable                          |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+


##### FAMILY 15 — SAFE & CONSTRAINED REINFORCEMENT LEARNING

    Safe RL aims to learn policies that satisfy safety constraints — either during training
    (safe exploration) or at deployment (constraint satisfaction). Constrained MDPs (CMDPs)
    extend MDPs with a cost function C(s,a) and constraint: E[cost] ≤ d.


    ALGORITHM                       SUMMARY                                                      INPUT / OUTPUT                          CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Constrained MDP (CMDP)           Formal framework: optimise Jᵣ(π) subject to Jᵢ(π) ≤ dᵢ.     Policy π →                              Mathematical foundation for safe RL; multiple constraint types;
                                    Separate reward and cost functions; separate limits.         Policy satisfying constraints          Lagrangian and primal-dual methods solve CMDPs; Altman 1999

    Constrained Policy Optimisation  TRPO-style update that enforces cost constraint explicitly. Policy + cost advantage estimates →     Theoretical guarantees on constraint satisfaction at each update;
    (CPO)                            Projects the policy update to satisfy the constraint.       Safe policy update                     complex implementation; approximations needed in practice; Achiam 2017

    Lagrangian Methods (Lagrangian  Adds a Lagrange multiplier λ for each constraint.            Policy + multipliers →                  Simple adaptation of any policy gradient algorithm; multiplier
    PPO / SAC)                       Alternates: update policy maximising J − λC, update λ.      Constrained policy                     automatically increases when constraint is violated; can oscillate

    Reward Constrained Policy        Augments reward with a constraint penalty term and          Policy, constraint budget →             Scales to multiple constraints; smoothly trades off reward vs
    Optimisation (RCPO)              propagates the constraint signal via TD methods.            Policy with penalised reward            cost; simpler than CPO; supports discount constraints; Tessler 2019

    Safety Layer                     Learns an analytic safety model (linear approximation)      Proposed action →                       Closed-form safety projection; negligible runtime overhead;
                                    and projects unsafe actions to the nearest safe action.      Corrected safe action                  requires knowing constraint structure; Dalal 2018

    Risk-Sensitive RL (CVaR, VaR)    Optimises a risk measure of the return distribution         Distribution over returns →              Avoids catastrophic outcomes; CVaR optimises the worst tail;
                                    rather than the mean: e.g. CVaR_α(G).                        Risk-penalised policy                  conservative policies; risk measure is a hyperparameter


##### FAMILY 16 — REWARD SHAPING, HINDSIGHT & CURRICULUM

    Sparse rewards make learning extremely difficult. These techniques either supplement
    the reward signal (shaping), relabel experience to extract learning signal from failures
    (HER), or progressively structure the difficulty of the learning problem (curriculum).


    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | ALGORITHM                            | SUMMARY                                                  | INPUT / OUTPUT                           | CHARACTERISTICS                                                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Potential-Based Reward Shaping       | Adds auxiliary reward: F(s,s') = γΦ(s') − Φ(s).          | Original reward + potential Φ →          | Does not change the optimal policy (policy-invariant); Φ can encode domain           |
    |                                      | Potential function Φ(s) encodes domain knowledge.        | Shaped reward                            | knowledge; choice of Φ is critical and requires careful design; Ng 1999             |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Hindsight Experience Replay (HER)    | Replays failed trajectories with the achieved goal       | Failed episode + achieved goal g' →      | Turns failures into successes; extremely effective for sparse goal-conditioned       |
    |                                      | as the desired goal — extracting learning signal anyway. | Relabelled transitions in replay buffer  | tasks (robotic manipulation); Andrychowicz 2017; pairs with DDPG / TD3             |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Curriculum Learning                  | Gradually increases task difficulty as agent improves.   | Ordered task distribution →              | Accelerates learning; prevents getting stuck on hard tasks early;                    |
    |                                      | Manually designed or automatically generated curriculum. | Sequence of training tasks               | requires knowing task difficulty ordering or automatic estimation                    |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Automatic Curriculum Generation      |Automatically selects the next training task to maximise  | Agent progress metrics →                 | ALP-GMM, PAIRED, and teacher-student methods; no manual curriculum design;           |
    | (ALP-GMM, PAIRED)                    | learning progress (e.g. change in task success rate).    | Next task from curriculum                | PAIRED uses adversarial environment generation; Portelas / Dennis 2020             |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Reward Machines                      | Specifies the reward function as a finite state machine  | Task specification as automaton →        | Enables automatic reward shaping from the machine structure; provides dense         |
    |                                      | that maps event sequences to reward values.              | Reward signal from automaton states      | signals for complex temporal tasks; facilitates transfer across tasks                |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+


##### FAMILY 17 — RLHF & LLM ALIGNMENT

    Reinforcement Learning from Human Feedback (RLHF) applies RL to fine-tune large language
    models to follow instructions, be helpful, and avoid harmful outputs. It is the core
    technique behind ChatGPT, Claude, and Gemini. Currently the most commercially important
    application of RL.


    ALGORITHM                       SUMMARY                                                     INPUT / OUTPUT                          CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    RLHF (Reward Model + PPO)        1. Collect human preferences on output pairs.              Human preference labels →                Three-stage pipeline: SFT → RM → RL; reward model can be
                                    2. Train reward model R_θ. 3. Fine-tune LLM with PPO.       Aligned LLM policy                     misspecified; reward hacking via RL possible; Ouyang 2022 (InstructGPT)

    Supervised Fine-Tuning (SFT)    Pre-trains the policy on expert demonstrations before       Expert demonstrations (prompt, response) Provides behavioural prior for RLHF; reduces variance of RL
                                    RL. First step in the RLHF pipeline.                        → Fine-tuned LLM                        optimisation; acts as the reference policy for KL divergence

    KL Penalty                      Adds KL divergence penalty between RL policy and SFT        RL policy + SFT reference policy →      Prevents the LLM from drifting too far from the SFT model;
                                    model to the reward: r' = R_θ(x,y) − β KL(π || π_SFT).      Penalised reward for RL update          β controls the trade-off between reward and language quality

    RLAIF (RL from AI Feedback)      Replaces human labellers with an AI judge (e.g. Claude     AI-labelled preference pairs →           Scalable: no human annotation bottleneck; quality depends on the
                                    or GPT-4) to generate preference labels at scale.           Aligned model policy                    judge LLM's calibration and values; Constitutional AI uses RLAIF

    Process Reward Models (PRM)      Rewards the model at each reasoning step rather than       Step-level labels (correct / incorrect)  Combats reward hacking on final answer; improves chain-of-thought
                                    only on the final output (outcome-supervised).              → Step-level reward signal              reasoning; requires step-level annotations (expensive); Lightman 2023

    Direct Preference Optimisation   Directly optimises the LLM on preference data without      Human preference pairs →                Eliminates the explicit reward model and RL loop; DPO derives a
    (DPO)                           training a separate reward model or running PPO.            Aligned LLM policy directly             closed-form objective from the Bradley-Terry model; simpler; Rafailov 2023

    Constitutional AI (CAI)         Chain of critique and revision guided by a set of           AI-generated critiques + principles →    Reduces reliance on human feedback; model critiques itself;
                                    principles; combined with RLAIF for alignment.              Aligned model policy                    used in Claude; balances helpfulness with harmlessness; Bai 2022


##### FAMILY 18 — FUNCTION APPROXIMATION & NETWORK ARCHITECTURES IN RL

    Function approximation allows RL to generalise across large or continuous state/action spaces.
    Neural network architecture choices significantly impact stability, sample efficiency,
    and performance. These are the building blocks inside modern deep RL agents.


    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | COMPONENT                            | SUMMARY                                                  | INPUT / OUTPUT                           | CHARACTERISTICS                                                                      |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Linear Function Approximation        | V(s) = wᵀφ(s) or Q(s,a) = wᵀφ(s,a).                      | Feature vector φ(s) →                    | Convergence guarantees with TD; interpretable; requires hand-crafted features;       |
    |                                      | Simplest parameterised value function.                   | Scalar value estimate                    | limited expressivity; basis for many theoretical results                             |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | CNN in RL (Visual Observations)      | Extracts features from raw pixel observations.           | Raw image frames →                       | Used in DQN, Atari agents; stacked frames for temporal info; shared CNN              |
    |                                      | Typically several conv layers followed by MLP.           | Feature embeddings for Q / policy        | for actor + critic can improve stability; requires large replay buffer               |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | LSTM / GRU in RL (Partial Obs.)      | Maintains hidden state across timesteps to handle POMDPs | Observation sequence →                   | Enables memory-based policies; training via BPTT; must handle episode boundaries;   |
    |                                      | or tasks requiring memory of past observations.          | Policy / value conditioned on history    | R2D2 and DRQN are DQN variants with RNNs; burn-in stabilises training               |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Transformer in RL                    | Self-attention over past observations / tokens.          | Observation sequence (context) →         | Long context without gradient vanishing; Decision Transformer (offline RL);          |
    |                                      | Gato and Decision Transformer use Transformer policies.  | Action conditioned on full context       | Gato is a multi-task generalist agent; quadratic memory in context length            |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Target Networks                      | Separate network for computing TD targets; updated       | Online Q-network →                       | Critical for stability in DQN; eliminates moving-target problem; Polyak              |
    |                                      | slowly (Polyak averaging) or periodically (hard copy).   | Stable TD targets for Q-learning         | (soft) updates τθ + (1−τ)θ_old smoother than hard copy; used universally            |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Experience Replay Buffer             | Stores past transitions (s,a,r,s'); samples mini-batches.| Stream of transitions →                  | Breaks temporal correlations; reuses data; enables off-policy learning;              |
    |                                      | Off-policy algorithms (DQN, SAC, TD3) require replay.    | Mini-batches for gradient updates        | buffer size is a key hyperparameter; not compatible with on-policy methods           |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+
    | Dueling Architecture                 | Separates Q into V(s) and A(s,a) streams merged via:     | State features →                         | More efficient V(s) updates (shared for all actions); advantage is mean-subtracted;  |
    |                                      | Q(s,a) = V(s) + A(s,a) − mean_a' A(s,a').                | Q-values via two streams                 | especially beneficial for many similar-valued actions; Wang 2016                    |
    +--------------------------------------+----------------------------------------------------------+------------------------------------------+--------------------------------------------------------------------------------------+


### COMPLETE ALGORITHM QUICK-REFERENCE

    Family                          Algorithms & Methods
    ─────────────────────────────────────────────────────────────────────────────────────────────

    Core Framework                  MDP, POMDP, Bellman Expectation, Bellman Optimality,
                                    Reward Function, Discount Factor γ, Policy π(a|s)

    Dynamic Programming             Policy Evaluation, Policy Improvement, Policy Iteration,
                                    Value Iteration, Generalised Policy Iteration, Async DP

    Monte Carlo Methods             First-Visit MC, Every-Visit MC, MC Control,
                                    Off-Policy MC with Importance Sampling

    Temporal Difference             TD(0), TD(λ), Eligibility Traces, SARSA, Q-Learning,
                                    Expected SARSA, Double Q-Learning, SARSA(λ)

    Deep Value-Based                DQN, Double DQN, Dueling DQN, PER, NoisyNet,
                                    C51, QR-DQN, IQN, Rainbow, Multi-step DQN, Ape-X

    Policy Gradient                 REINFORCE, REINFORCE+Baseline, A2C, A3C, NPG,
                                    TRPO, PPO, DDPG, TD3, SAC, DPG, MPO

    Model-Based RL                  Dyna-Q, Dyna-Q+, PILCO, World Models, PlaNet,
                                    Dreamer/V2/V3, MCTS, AlphaGo, AlphaZero, MuZero, PETS, MBPO

    Multi-Armed Bandits             ε-Greedy, UCB1, Thompson Sampling, Boltzmann,
                                    LinUCB, GP Bandits, Posterior Sampling for RL (PSRL)

    Exploration (Deep RL)           Count-Based, ICM, RND, NGU, Agent57,
                                    Go-Explore, NoisyNets

    Multi-Agent RL                  IQL, QMIX, MADDPG, COMA, MAPPO,
                                    Mean Field RL, Self-Play, League Training

    Inverse RL / Imitation          Behavioural Cloning, DAgger, MaxEnt IRL,
                                    Guided Cost Learning, GAIL, AIRL, Reward Modelling

    Hierarchical RL                 Options Framework, Feudal Networks (FuN),
                                    HIRO, HAC, Skill Discovery (DIAYN, VALOR)

    Offline RL                      BCQ, CQL, TD3+BC, IQL, Decision Transformer,
                                    Trajectory Transformer, BEAR

    Meta-RL                         MAML, RL², SNAIL, PEARL, MAML-PPO

    Safe / Constrained RL           CMDP, CPO, Lagrangian PPO/SAC, RCPO,
                                    Safety Layer, Risk-Sensitive RL (CVaR)

    Reward Shaping & Curriculum     Potential-Based Shaping, HER, Curriculum Learning,
                                    ALP-GMM, PAIRED, Reward Machines

    RLHF & LLM Alignment            RLHF, SFT, KL Penalty, RLAIF, PRM,
                                    DPO, Constitutional AI (CAI)

    Neural Architectures in RL      Linear FA, CNN in RL, LSTM/GRU in RL, Transformer in RL,
                                    Target Networks, Experience Replay, Dueling Architecture
    ─────────────────────────────────────────────────────────────────────────────────────────────


## The Road from Bandits to World Models

This learning path walks you through the complete evolution of reinforcement learning,
starting from the simplest exploration problem and building all the way to modern
world models and LLM alignment.

---

### Phase 1 — Mathematical & Probabilistic Foundations
Before any RL algorithm, you need:
- **Probability**: distributions, conditional probability, expectation, variance
- **Linear Algebra**: vectors, matrix operations, dot products
- **Calculus**: derivatives, partial derivatives, chain rule, gradients
- **Optimisation**: gradient ascent/descent, convergence, learning rates
- **Statistics**: estimators, bias-variance tradeoff, law of large numbers

> *RL is essentially optimisation under uncertainty — these tools are the language.*

---

### Phase 2 — The Reinforcement Learning Problem
The core formalism:
- **MDP** (Markov Decision Process): states S, actions A, transitions P, rewards R, discount γ
- **Agent-environment loop**: observe → act → receive reward → repeat
- **Return Gₜ**: cumulative discounted future reward
- **Bellman equations**: V(s) expressed recursively in terms of V(s')
- **Exploration vs. Exploitation**: the fundamental tension in all of RL

Key concepts: *episode, trajectory, policy, value function, Q-function, advantage*

---

### Phase 3 — Multi-Armed Bandits
The simplest RL problem — one state, K actions:
- **ε-Greedy**: simplest exploration strategy, epsilon controls exploration rate
- **UCB** (Upper Confidence Bound): optimism in the face of uncertainty
- **Thompson Sampling**: Bayesian posterior sampling over arm values
- **Contextual Bandits**: feature-conditioned actions (LinUCB)
- **Regret analysis**: how to measure and bound exploration inefficiency

Key concepts: *exploration bonus, confidence bound, regret, posterior sampling*

---

### Phase 4 — Dynamic Programming
Exact solutions when the model (P, R) is known:
- **Policy Evaluation**: compute V^π by iterating the Bellman expectation equation
- **Policy Improvement**: make the policy greedy with respect to V^π
- **Policy Iteration**: alternate evaluation and improvement until convergence
- **Value Iteration**: collapse to one-step updates of the Bellman optimality equation
- **Generalised Policy Iteration (GPI)**: unifying framework for all DP and RL methods

Key concepts: *convergence, fixed point, sweep, bootstrapping, model-based planning*

---

### Phase 5 — Monte Carlo & Temporal Difference Learning
Model-free learning from experience:
- **Monte Carlo**: learn from complete episodes; average actual returns; no bootstrapping
- **TD(0)**: update after every step using r + γV(s'); combines sampling with bootstrapping
- **SARSA**: on-policy TD control — learns Q for the policy being followed (including exploration)
- **Q-Learning**: off-policy TD control — learns Q* regardless of exploration policy
- **TD(λ)** and **eligibility traces**: interpolate between TD and MC via λ parameter

Key concepts: *TD error, on-policy vs off-policy, tabular RL, eligibility traces, λ-returns*

---

### Phase 6 — Deep Q-Networks (Value-Based Deep RL)
Scaling Q-learning to high-dimensional observations:
- **DQN**: Q-learning + neural network + experience replay + target network
- **Double DQN**: decouple action selection from evaluation to reduce overestimation
- **Dueling DQN**: split Q into value + advantage streams for better learning signal
- **Prioritised Experience Replay**: focus on surprising / high-error transitions
- **Rainbow**: all DQN improvements combined; strong Atari baseline
- **Distributional RL** (C51, QR-DQN, IQN): model the full return distribution not just mean

Key concepts: *replay buffer, target network, overestimation bias, distributional RL*

---

### Phase 7 — Policy Gradient Methods
Directly optimising the policy π_θ:
- **REINFORCE**: Monte Carlo policy gradient; high variance, foundational
- **Baseline**: subtract V(s) from returns to reduce variance without bias
- **Actor-Critic (A2C/A3C)**: policy gradient + value function baseline; one-step updates
- **TRPO**: update policy in a trust region (KL constraint) — monotonic improvement
- **PPO**: clip the probability ratio — simpler, more widely used than TRPO
- **SAC**: entropy-augmented off-policy actor-critic; state-of-the-art continuous control
- **TD3**: double critics + delayed updates; robust DDPG successor

Key concepts: *policy gradient theorem, advantage function, entropy bonus, clipping ratio*

---

### Phase 8 — Model-Based RL
Using learned or known environment models:
- **Dyna-Q**: combine real and imagined experience for planning
- **World Models** (Dreamer): learn a compact latent model; train policy in imagination
- **MCTS**: lookahead tree search guided by value network + policy network
- **MuZero**: learn the model jointly — no rules required
- **PILCO / PETS**: probabilistic models for sample-efficient control

Key concepts: *planning, model bias, latent space, imagination rollouts, sample efficiency*

---

### Phase 9 — Advanced Topics
Pushing the frontier:
- **Multi-Agent RL**: QMIX (cooperative), MADDPG, MAPPO, self-play
- **Inverse RL / Imitation**: BC, DAgger, GAIL, AIRL — learning from demonstrations
- **Hierarchical RL**: temporal abstraction; options, Feudal Networks, HIRO
- **Offline RL**: CQL, BCQ, IQL, Decision Transformer — learn from fixed datasets
- **Meta-RL**: MAML, RL² — learn to learn; fast adaptation to new tasks
- **Safe RL**: CPO, Lagrangian PPO, risk-sensitive — constraints and safety during learning

---

### Phase 10 — RLHF & LLM Alignment
The current frontier of RL in practice:
- **SFT**: supervised fine-tuning on expert demonstrations (first RLHF stage)
- **Reward Modelling**: train a reward model from human preference comparisons
- **PPO-based RLHF**: fine-tune LLMs with RL using the learned reward model
- **DPO**: skip the reward model — optimise directly on preference data
- **Constitutional AI (CAI)**: self-critique loops + RLAIF for scalable alignment
- **Process Reward Models**: reward at each reasoning step, not just the final answer

Key concepts: *reward hacking, KL penalty, preference data, alignment, RLAIF*

---

### Suggested Study Order

| #  | Topic                           | Builds On                          |
|----|---------------------------------|------------------------------------|
| 1  | Probability & Expectations      | —                                  |
| 2  | MDP Formalism                   | Probability                        |
| 3  | Bellman Equations               | MDP                                |
| 4  | Multi-Armed Bandits             | Probability, MDP basics            |
| 5  | Dynamic Programming             | Bellman Equations                  |
| 6  | Monte Carlo Methods             | MDP, Return Gₜ                      |
| 7  | TD Learning (TD0, SARSA)        | DP, Monte Carlo                    |
| 8  | Q-Learning                      | TD Learning                        |
| 9  | DQN & Deep Value-Based          | Q-Learning, Neural Networks        |
| 10 | REINFORCE & Policy Gradients    | MDP, Neural Networks               |
| 11 | Actor-Critic (A2C/A3C)          | Policy Gradients, TD Learning      |
| 12 | PPO                             | Actor-Critic, TRPO                 |
| 13 | DDPG / TD3                      | Actor-Critic, Off-Policy RL        |
| 14 | SAC                             | TD3, Entropy Regularisation        |
| 15 | Model-Based RL (Dyna, Dreamer)  | Q-Learning, Policy Gradients       |
| 16 | Exploration (RND, ICM, NGU)     | DQN, Curiosity                     |
| 17 | Inverse RL & Imitation (GAIL)   | Policy Gradients, RL basics        |
| 18 | Hierarchical RL (Options, HIRO) | Policy Gradients, Goal-Conditioned |
| 19 | Multi-Agent RL (QMIX, MADDPG)   | Policy Gradients, Q-Learning       |
| 20 | Offline RL (CQL, BCQ, DT)       | Q-Learning, Policy Gradients       |
| 21 | Meta-RL (MAML, RL²)             | Policy Gradients, Fine-Tuning      |
| 22 | Safe RL (CPO, Lagrangian)       | Policy Gradients, CMDP             |
| 23 | Reward Shaping & HER            | Q-Learning, Goal-Conditioned RL    |
| 24 | RLHF (Reward Model + PPO)       | PPO, Reward Modelling              |
| 25 | DPO & Constitutional AI         | RLHF, LLMs                         |

---

"""


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
    visual_height = 400
    try:
        from Reinforcement_learning.visuals.RL_path_visuals import (
            RL_PATH_VISUAL_HTML,
            RL_PATH_VISUAL_HEIGHT,
        )
        visual_html   = RL_PATH_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = RL_PATH_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[00_reinforcement_learning_path.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None,
        "operations":    None,
    }