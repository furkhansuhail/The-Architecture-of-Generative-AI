"""Module: 05 · PPO"""

"""
Proximal Policy Optimisation (PPO)
====================================

PPO (Schulman et al., OpenAI 2017) is the algorithm that turned policy
gradient methods into a practical, reliable workhorse. It achieved what
TRPO (Trust Region Policy Optimisation) achieved theoretically — stable,
monotonically improving policy updates — but with a fraction of the
implementation complexity.

The central insight: instead of enforcing a KL constraint (TRPO), simply
CLIP the probability ratio r(θ) = π_θ/π_old so that updates that move
too far from the old policy are suppressed. One hyperparameter (ε ≈ 0.2),
one loss term, no second-order optimisation.

The result was an algorithm that:
  * Matches TRPO on continuous control benchmarks
  * Scales to multi-billion parameter LLMs (RLHF)
  * Runs stably with mini-batch SGD on commodity hardware
  * Handles discrete and continuous actions identically

PPO is today the default on-policy algorithm for everything from robotics
locomotion to language model alignment. This module builds it completely
from first principles: the motivation from TRPO, the clipped surrogate
objective, GAE, the full training loop, hyperparameter guidance,
and the implementation details that determine whether PPO actually works.
"""

import re

TOPIC_NAME   = "Proximal Policy Optimisation"
DISPLAY_NAME = "05 · PPO"
ICON         = "⚡"
SUBTITLE     = "Proximal Policy Optimisation — stable on-policy training at scale"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — THE PROBLEM PPO SOLVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Standard Policy Gradient Instability

Standard REINFORCE / vanilla policy gradient:

        θ ← θ + α · ∇_θ J(θ)

suffers from two coupled problems:

    1. STEP SIZE SENSITIVITY
       A too-large step in θ-space can correspond to a huge change in
       policy behaviour. A policy that walked left now runs right.
       The gradient estimated at π_old gives no useful information about
       the objective at π_new once the two policies are far apart.

    2. DESTRUCTIVE UPDATES
       Bad steps are not just wasteful — they are catastrophic. A collapsed
       policy generates bad experience, which causes worse gradient estimates,
       leading to further collapse. There is no self-correcting mechanism
       in vanilla PG. Once training diverges, it rarely recovers.

    3. SAMPLE INEFFICIENCY
       On-policy methods must discard each batch of experience after one
       gradient step (the data was collected under π_old, now we have π_new).
       With a small step size, most of each batch's information is wasted.

The challenge: we want to take the largest possible step that is still safe.
This is the trust region concept.


### The Trust Region Idea (TRPO, Schulman et al. 2015)

Define a trust region around π_old — a region of policy space where
the surrogate objective accurately predicts the true objective change.

TRPO optimises a surrogate objective subject to a KL constraint:

        max_θ  L^CPI(θ)      (Conservative Policy Improvement surrogate)
        s.t.   KL( π_old || π_θ ) ≤ δ

where:
        L^CPI(θ) = E_t [ π_θ(aₜ|sₜ) / π_old(aₜ|sₜ) · Âₜ ]
                 = E_t [ r_t(θ) · Âₜ ]

and r_t(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ) is the probability ratio.

TRPO guarantees monotone improvement:
        J(π_new) ≥ J(π_old)  −  ε_clip / (1 − γ)

The KL constraint ensures the surrogate remains a valid lower bound.

TRPO's problem: the KL constraint requires second-order optimisation
(conjugate gradient + line search). Expensive, complex, doesn't work
with shared network architectures.


### PPO's Insight: Clip Instead of Constrain

PPO replaces the KL constraint with a clipped surrogate:

        L^CLIP(θ) = E_t [ min( r_t(θ) · Âₜ,  clip(r_t(θ), 1−ε, 1+ε) · Âₜ ) ]

The clip acts as a soft trust region without any constraint:
    * If r_t(θ) is within [1−ε, 1+ε]: unclipped, full gradient
    * If r_t(θ) > 1+ε  and Âₜ > 0:   action is already being reinforced;
                                        clip prevents making it even more likely
    * If r_t(θ) < 1−ε  and Âₜ < 0:   action is already being penalised;
                                        clip prevents making it even less likely

The min takes the worse of clipped and unclipped — a pessimistic lower bound
on the improvement. This prevents the agent from being over-optimistic
about the benefit of large ratio changes.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — THE PROBABILITY RATIO AND IMPORTANCE SAMPLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Probability Ratio r_t(θ)

        r_t(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)

This is an importance sampling ratio. It allows PPO to treat the objective
as an off-policy one: data was collected under π_old, but we compute
expectations under π_θ by reweighting.

    r_t(θ) = 1.0:    π_θ = π_old. No change. Gradient is the standard PG gradient.
    r_t(θ) > 1.0:    π_θ assigns higher probability to aₜ than π_old did.
    r_t(θ) < 1.0:    π_θ assigns lower probability to aₜ than π_old did.

Computing r_t(θ) requires:
    * log π_old(aₜ|sₜ): logged during rollout collection, stored in buffer
    * log π_θ(aₜ|sₜ):   recomputed during training from current network θ

        r_t(θ) = exp( log π_θ(aₜ|sₜ) − log π_old(aₜ|sₜ) )

Using log probabilities prevents numerical underflow with long sequences
or many independent action dimensions.


### Importance Sampling Validity

Importance sampling is valid when π_θ and π_old are close. When the two
policies are far apart, the importance weights have high variance and the
reweighted gradient estimate is unreliable.

The clip ensures π_θ stays close to π_old: r_t ∈ [1−ε, 1+ε]
guarantees the two policies assign similar probabilities to each action.

This is why PPO can take multiple gradient steps on the same batch
(K epochs): as long as the ratio stays within the clip bounds, the
importance sampling reweighting remains valid.


### Relationship to Standard Policy Gradient

When θ = θ_old (first gradient step from old policy):
        r_t(θ_old) = 1   for all t

        L^CLIP(θ_old) = E_t [ Âₜ ]   (not useful by itself)
        ∇_θ L^CLIP |_{θ=θ_old} = E_t [ ∇_θ log π_θ(aₜ|sₜ) · Âₜ ]

This is exactly the standard policy gradient! PPO degenerates to vanilla
PG at θ = θ_old. The clip only activates when r_t moves away from 1.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — THE COMPLETE PPO OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Three-Component Objective

PPO combines three terms into a single loss function:

        L^PPO(θ) = L^CLIP(θ) − c₁ · L^VF(θ) + c₂ · S(θ)

    L^CLIP(θ)   Policy surrogate with clipping (MAXIMISED)
    L^VF(θ)     Value function regression loss (MINIMISED)
    S(θ)        Entropy bonus (MAXIMISED — encourages exploration)


### 1. Clipped Policy Loss

        L^CLIP(θ) = E_t [ min( r_t(θ) Âₜ, clip(r_t(θ), 1−ε, 1+ε) Âₜ ) ]

    ε:       Clip ratio, typically 0.1 to 0.3. Standard: 0.2.

    Advantage Âₜ:   Computed via GAE (see Part 5).
                     Normalised per mini-batch: Â ← (Â − mean(Â)) / std(Â)
                     Normalisation stabilises gradient magnitudes significantly.


### 2. Value Function Loss

        L^VF(θ) = E_t [ (V_θ(sₜ) − Vₜ^target)² ]

    Vₜ^target:   TD-λ return (GAE + baseline): Vₜ^target = Âₜ + V_θ_old(sₜ)

Some implementations also clip the value function update:

        L^VF_clipped(θ) = max( (V_θ − V^target)²,
                               (clip(V_θ, V_old−ε, V_old+ε) − V^target)² )

This prevents the value function from moving too fast, matching the
spirit of the clipped policy update.

    c₁:   Value loss coefficient. Typically 0.5.


### 3. Entropy Bonus

        S(θ) = H(π_θ(·|sₜ)) = −Σ_a π_θ(a|sₜ) log π_θ(a|sₜ)

    c₂:   Entropy coefficient. Typically 0.01.
           Tuned down if the policy is too stochastic; up if it collapses.

Continuous actions (Gaussian policy):
    H(N(μ, σ)) = 0.5 · log(2πe σ²)
               = 0.5 + 0.5 · log(2π) + log(σ)

Increasing σ increases entropy — entropy bonus encourages exploration.


### Training Loop Structure

    Phase 1 — ROLLOUT COLLECTION (interact with environment):
        Run N actors for T timesteps each, using π_θ_old.
        Store: (sₜ, aₜ, rₜ, sₜ₊₁, doneₜ, log_π_old(aₜ|sₜ), V_old(sₜ))
        Compute: GAE advantages Âₜ and returns Vₜ^target

    Phase 2 — POLICY UPDATE (gradient steps on collected batch):
        For K epochs:
            Shuffle data into mini-batches of size M
            For each mini-batch:
                Recompute log π_θ(aₜ|sₜ) and V_θ(sₜ) from current θ
                Compute r_t(θ) = exp(log π_θ − log π_old)
                Compute L^CLIP, L^VF, S
                Gradient step: θ ← θ − α · ∇_θ(L^VF − L^CLIP − c₂S)

    Phase 3 — UPDATE OLD POLICY:
        θ_old ← θ     (copy current network as new old policy)
        Repeat from Phase 1.

Typical values: N=8 parallel actors, T=2048 steps, K=10 epochs, M=64.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — GENERALISED ADVANTAGE ESTIMATION IN PPO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why GAE for PPO

PPO requires an advantage estimate Âₜ for each timestep. GAE
(Schulman et al. 2015) provides the right bias-variance tradeoff.

        δₜ = rₜ + γ V(sₜ₊₁) − V(sₜ)       (TD residual)

        Âₜ^GAE = Σₖ₌₀^∞ (γλ)ᵏ δₜ₊ₖ
               = δₜ + γλ δₜ₊₁ + (γλ)² δₜ₊₂ + ...

Efficient backward computation through the rollout buffer:

        Â_{T-1} = δ_{T-1}
        Â_t     = δₜ + γλ · (1 − doneₜ₊₁) · Â_{t+1}

The (1 − done) mask resets the trace at episode boundaries.


### GAE-Lambda and Returns

The TD-λ return (used as value function target):

        Vₜ^target = Âₜ^GAE + V_old(sₜ)

This decomposes naturally: the value target is the baseline V(sₜ) plus
the GAE advantage estimate. Good targets = good critic = good advantages.


### Advantage Normalisation

After computing GAE, normalise per mini-batch:

        Â_normalised = (Â − mean(Â)) / (std(Â) + ε)

This ensures the policy gradient always has a reasonable magnitude
regardless of reward scale. Without normalisation, large rewards cause
large gradients; small rewards cause vanishing gradients.

Critical detail: normalise WITHIN the mini-batch, not across the full
rollout. This provides a consistent gradient scale.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — NETWORK ARCHITECTURE AND INITIALISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Shared vs Separate Networks

    SHARED (common):
        Single network with two heads:
        shared_trunk → [policy head, value head]

        Pros: efficient, shared representation, fewer parameters
        Cons: gradient interference (policy/value gradients can conflict)

    SEPARATE:
        Two independent networks, one for π, one for V.
        Pros: no gradient interference
        Cons: twice the parameters, no shared representation

Most implementations use shared networks with an orthogonal initialisation
and a small policy head initialisation (see below).


### Orthogonal Initialisation

Standard practice for PPO networks:
    * Hidden layers: orthogonal init, scale = √2
    * Policy head (action logits): orthogonal init, scale = 0.01
      (small init → near-uniform initial policy → exploration)
    * Value head: orthogonal init, scale = 1.0

The small policy head scale is critical. A large initial policy head
means some actions are heavily preferred before any experience is seen,
suppressing exploration and causing convergence to local optima.

Orthogonal matrices have unit singular values → gradient norms are
preserved during initialisation, preventing vanishing/exploding gradients.


### Action Distribution Details

    DISCRETE (Softmax Categorical):
        logits = policy_head(trunk)
        π(a|s) = softmax(logits)_a
        log π(a|s) = logits_a − log_sum_exp(logits)
        entropy   = −Σ_a π(a|s) log π(a|s)
        sample    = argmax or categorical sample

    CONTINUOUS (Diagonal Gaussian):
        μ = policy_head_mu(trunk)          # learnable mean
        log σ = learnable_parameter         # state-independent log std
                                            # (or: head outputs log σ)
        π(a|s) = N(μ(s), diag(exp(log σ)²))
        log π(a|s) = −0.5 Σ_i [(aᵢ − μᵢ)/σᵢ]² + log σᵢ + const
        entropy    = 0.5 Σ_i [1 + log(2π σᵢ²)]
        sample     = μ + σ ⊙ ε,  ε ~ N(0, I)   (reparameterisation)

    SQUASHED CONTINUOUS (Tanh-Gaussian, used in SAC):
        a = tanh(μ + σε)    (bounded to (−1, 1))
        log π adjusted for the tanh Jacobian:
            log π_squashed(a|s) = log π_Gaussian(tanh⁻¹(a)|s) − Σᵢ log(1 − aᵢ²)


### Observation Normalisation

Crucial for continuous state spaces:
    * Maintain running mean μ_obs and std σ_obs
    * Normalise: ŝ = (s − μ_obs) / (σ_obs + ε)
    * Update statistics online during rollout collection

Without normalisation, states with very different scales cause
gradient imbalances and slow convergence.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — IMPLEMENTATION DETAILS THAT MATTER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The 37 Implementation Details (Huang et al., 2022)

A well-known empirical study found that many small implementation details
significantly affect PPO performance. Key findings:

    CRITICAL DETAILS (large effect):
        1. Advantage normalisation (per mini-batch)
        2. Orthogonal weight initialisation (scale 0.01 for policy head)
        3. Value function clipping (same ε as policy clip)
        4. Global gradient norm clipping (max 0.5)
        5. Observation normalisation (running mean/std)
        6. Reward normalisation / clipping
        7. Learning rate annealing (linear decay to 0)

    IMPORTANT DETAILS (moderate effect):
        8. Separate log-std parameter (not from network head)
        9. Tanh activation in hidden layers (instead of ReLU)
        10. GAE lambda = 0.95 (not 0.98 or 1.0)
        11. Mini-batch shuffling between epochs
        12. Adam epsilon = 1e-5 (not default 1e-8)


### Gradient Clipping

PPO clips the global gradient norm:

        if ||∇_θ L|| > max_grad_norm:
            ∇_θ L ← ∇_θ L · max_grad_norm / ||∇_θ L||

Typical value: max_grad_norm = 0.5.

This prevents rare large gradients (from unusual rollouts) from
destabilising training. Complementary to the ratio clip.


### Learning Rate Annealing

Linear decay of learning rate over training:

        α_t = α_init · (1 − t / t_total)

Helps final convergence and reduces oscillation late in training.
Common for both continuous control and LLM fine-tuning.


### Early Stopping on KL

Optional: monitor the average KL divergence after each epoch:

        KL_approx = mean( log π_old(a|s) − log π_new(a|s) )

If KL exceeds a target threshold (e.g. 0.01), stop training early
and proceed to the next rollout. Prevents over-updating on one batch.


### The Clip Fraction Diagnostic

A useful diagnostic: what fraction of ratios are being clipped?

        clip_frac = mean( |r_t(θ) − 1| > ε )

    clip_frac ≈ 0:     update is too small; increase LR or reduce clip ε
    clip_frac ≈ 0.1:   healthy; policy is changing meaningfully
    clip_frac > 0.5:   update is too large; reduce LR or increase clip ε

Monitoring clip_frac provides a real-time signal on update stability.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — PPO HYPERPARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Standard Hyperparameter Ranges

    +----------------------------+------------------+---------------------------+
    | Hyperparameter             | Typical Range    | Notes                     |
    +----------------------------+------------------+---------------------------+
    | Clip ratio ε               | 0.1 – 0.3        | 0.2 default               |
    | GAE lambda λ               | 0.9 – 0.99       | 0.95 default              |
    | Discount γ                 | 0.99 – 0.999     | 0.99 most tasks           |
    | Learning rate              | 1e-4 – 3e-4      | Adam, linearly decayed    |
    | Value loss coeff c₁        | 0.25 – 1.0       | 0.5 default               |
    | Entropy coeff c₂           | 0.0 – 0.1        | 0.01 default              |
    | N rollout steps T          | 128 – 2048       | 2048 for MuJoCo           |
    | N parallel envs            | 4 – 64           | 8 default                 |
    | N epochs K                 | 3 – 30           | 10 default                |
    | Mini-batch size M          | 32 – 256         | 64 default                |
    | Max grad norm              | 0.5              | Critical!                 |
    | Adam epsilon               | 1e-5             | Not 1e-8                  |
    +----------------------------+------------------+---------------------------+


### Task-Specific Defaults

    ATARI (discrete, pixels):
        T=128, envs=8, epochs=4, batch=32, ε=0.1, γ=0.99, λ=0.95,
        lr=2.5e-4 (annealed), clip_vloss=True, entropy=0.01

    MUJOCO (continuous, low-dim):
        T=2048, envs=1, epochs=10, batch=64, ε=0.2, γ=0.99, λ=0.95,
        lr=3e-4 (annealed), clip_vloss=True, entropy=0.0, obs_norm=True

    RLHF / LLM (large language models):
        T=varies (sequence length), epochs=1-4, ε=0.2, γ=1.0, λ=0.95,
        KL penalty: β=0.01–0.1, usually DPO preferred now


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — PPO FOR RLHF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The RLHF Objective

        max_θ  E_{x~D, y~π_θ(·|x)} [ R_φ(x, y) ]
               − β · KL( π_θ(·|x) || π_ref(·|x) )

    R_φ(x, y)         Learned reward model (trained on human preferences)
    π_ref             Reference policy (SFT model, frozen)
    β ∈ [0.01, 0.1]   KL penalty weight

The KL term serves the same role as the clip in standard PPO:
it prevents the policy from drifting too far from the SFT model.
Without it, the model reward-hacks — producing outputs that score highly
under R_φ but are nonsensical or harmful.


### Per-Token Reward

The LLM generates a sequence token by token:
    State:  x + generated tokens so far
    Action: next token aₜ ∈ vocabulary
    Reward: R_φ(x, y) given at the END of the sequence (sparse)

Per-token KL penalty converts this to a dense reward:

        rₜ = −β · log( π_θ(aₜ|sₜ) / π_ref(aₜ|sₜ) )

    Final token: rₜ = R_φ(x, y) − β · KL_t

This makes the per-step reward dense while still penalising deviation
from the reference policy.


### Practical RLHF PPO Differences from Standard PPO

    1. SEQUENCE = EPISODE
       Each generated response is one episode.
       The episode length = response length (variable).

    2. VALUE HEAD ON LLM
       The same LLM backbone has an additional linear value head.
       This critic estimates V(s) = V(prompt + partial response).

    3. LARGE BATCH SIZE
       Mini-batch sizes of 512–8192 sequences (GPU memory permitting).

    4. SMALL LR, FEW EPOCHS
       1–4 PPO epochs per batch (vs 10 for standard tasks).
       LR ≈ 1e-6 to 1e-5 (large networks are sensitive).

    5. REWARD CLIPPING
       R_φ(x,y) clipped to a fixed range to prevent reward hacking.

    6. WHITENING ADVANTAGES
       Whitening (normalising advantages) is even more critical for LLMs
       because reward magnitudes vary widely across different prompts.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — FAILURE MODES AND DIAGNOSTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Common Failure Modes

    POLICY COLLAPSE (entropy → 0)
        Symptom: policy becomes deterministic quickly; stops exploring.
        Cause: entropy coeff too low, or LR too high.
        Fix: increase c₂ (entropy coeff), reduce LR.

    REWARD HACKING
        Symptom: reward model score improves but actual quality degrades.
        Cause: reward model is imperfect; policy exploits its blind spots.
        Fix: increase KL penalty β, use conservative reward model,
             reduce number of PPO epochs.

    VALUE FUNCTION COLLAPSE
        Symptom: value loss explodes; advantages become meaningless.
        Cause: c₁ too small, LR too high for value head.
        Fix: increase c₁, clip value function update, reduce LR.

    OSCILLATING RETURNS
        Symptom: return fluctuates wildly, no monotone improvement.
        Cause: clip ratio too large, LR too high, or batch too small.
        Fix: reduce LR, reduce ε, increase batch size.

    SLOW CONVERGENCE
        Symptom: learning is stable but very slow.
        Cause: LR too small, too few epochs, rollout too short.
        Fix: increase LR, increase K epochs, increase T.


### Key Diagnostic Metrics

    clip_frac:    Fraction of transitions with |r_t − 1| > ε
                  Healthy: 0.05 – 0.15

    approx_kl:    E[ log π_old − log π_new ] per update
                  Healthy: 0.01 – 0.05

    entropy:      Policy entropy (should decrease gradually, not collapse)

    value_loss:   Critic regression MSE (should decrease monotonically)

    explained_var: 1 − Var(V^target − V_pred) / Var(V^target)
                   Healthy: > 0.9 after training


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 10 — SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Core Equations

    PROBABILITY RATIO:
        r_t(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)
               = exp( log π_θ(aₜ|sₜ) − log π_old(aₜ|sₜ) )

    CLIPPED POLICY LOSS (maximise):
        L^CLIP(θ) = E_t [ min( r_t Âₜ,  clip(r_t, 1−ε, 1+ε) Âₜ ) ]

    VALUE LOSS (minimise):
        L^VF(θ) = E_t [ (V_θ(sₜ) − V^target_t)² ]

    ENTROPY BONUS (maximise):
        S(θ) = E_t [ H(π_θ(·|sₜ)) ]

    FULL OBJECTIVE (maximise):
        L^PPO(θ) = L^CLIP − c₁ L^VF + c₂ S

    GAE (backward pass):
        δₜ = rₜ + γ V(sₜ₊₁)(1−done) − V(sₜ)
        Âₜ = δₜ + γλ(1−done) Â_{t+1}

    RETURN TARGET:
        V^target_t = Âₜ + V_old(sₜ)

    APPROX KL:
        KL ≈ E_t [ log π_old − log π_new ]

    EXPLAINED VARIANCE:
        EV = 1 − Var(V^target − V_pred) / Var(V^target)


### PPO vs TRPO vs Vanilla PG

    +------------------------+---------------+------------------+------------------+
    | Property               | Vanilla PG    | TRPO             | PPO              |
    +------------------------+---------------+------------------+------------------+
    | Trust region           | None          | KL constraint    | Ratio clip       |
    | Monotone improvement   | No            | Yes (theoretical)| Usually          |
    | Complexity             | Simple        | High (2nd order) | Simple           |
    | Multiple epochs/batch  | No            | Yes              | Yes              |
    | Scales to large models | Poorly        | No               | Yes              |
    | Used in RLHF           | No            | No               | Yes              |
    +------------------------+---------------+------------------+------------------+

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "1. PPO Clip Loss — Mechanics and Visualisation": {
        "description": "Compute the clipped PPO loss for a range of probability ratios "
                       "and advantages. Show exactly when and why clipping activates, "
                       "and compare to the unclipped surrogate.",
        "code": """\
import numpy as np

EPSILON = 0.2

def ppo_loss(ratio, advantage, eps=EPSILON):
    unclipped = ratio * advantage
    clipped   = np.clip(ratio, 1 - eps, 1 + eps) * advantage
    return np.minimum(unclipped, clipped)   # pessimistic min

def gradient_ppo(ratio, advantage, eps=EPSILON):
    \"\"\"Effective gradient signal from PPO (sign of loss gradient).\"\"\"
    r_clipped = np.clip(ratio, 1 - eps, 1 + eps)
    unclipped = ratio * advantage
    clipped   = r_clipped * advantage
    active    = unclipped <= clipped   # True = unclipped is active
    # Gradient of unclipped term: advantage; clipped term: 0 when at boundary
    return np.where(active, advantage, np.where(
        (ratio > 1 + eps) | (ratio < 1 - eps), 0.0, advantage))

print("PPO Clip Loss Mechanics  (ε = 0.2)")
print("=" * 70)
print()

# ── Table: loss values across ratios for two advantage signs ──
ratios = np.array([0.5, 0.7, 0.8, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0])

print("Positive advantage (Â = +1.0):")
print(f"  {'ratio r':>10} | {'unclipped':>12} | {'clipped':>12} | "
      f"{'PPO loss':>10} | {'gradient':>10} | {'clip active?':>13}")
print("  " + "-"*68)
for r in ratios:
    A     = 1.0
    unc   = r * A
    cl    = np.clip(r, 1-EPSILON, 1+EPSILON) * A
    L     = ppo_loss(r, A)
    g     = gradient_ppo(r, A)
    active= abs(L - unc) > 1e-8
    print(f"  {r:>10.2f} | {unc:>12.4f} | {cl:>12.4f} | "
          f"{L:>10.4f} | {g:>10.4f} | {'YES ← clip' if active else 'no':>13}")

print()
print("Negative advantage (Â = -1.0):")
print(f"  {'ratio r':>10} | {'unclipped':>12} | {'clipped':>12} | "
      f"{'PPO loss':>10} | {'gradient':>10} | {'clip active?':>13}")
print("  " + "-"*68)
for r in ratios:
    A     = -1.0
    unc   = r * A
    cl    = np.clip(r, 1-EPSILON, 1+EPSILON) * A
    L     = ppo_loss(r, A)
    g     = gradient_ppo(r, A)
    active= abs(L - unc) > 1e-8
    print(f"  {r:>10.2f} | {unc:>12.4f} | {cl:>12.4f} | "
          f"{L:>10.4f} | {g:>10.4f} | {'YES ← clip' if active else 'no':>13}")

print()
print("Key observations:")
print("  Positive Â, r > 1+ε: action being overconfidently reinforced → clipped (gradient=0)")
print("  Positive Â, r < 1-ε: action being penalised despite high return → clipped (gradient=0)")
print("  Negative Â, r < 1-ε: action being overconfidently penalised → clipped (gradient=0)")
print("  Safe zone r ∈ [0.8, 1.2]: full gradient signal, no clipping")
""",
    },

    "2. GAE — Full Backward Computation": {
        "description": "Implement the full GAE backward pass on a rollout buffer. "
                       "Compute advantages and value targets. Show the effect of "
                       "λ and demonstrate advantage normalisation.",
        "code": """\
import numpy as np

GAMMA = 0.99

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=0.95):
    \"\"\"
    GAE backward pass.
    rewards, values, dones: arrays of length T
    values has length T+1 (includes bootstrap value at end)
    Returns: advantages (T,), returns (T,)
    \"\"\"
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv   = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * next_non_terminal - values[t]
        advantages[t] = last_adv = delta + gamma * lam * next_non_terminal * last_adv
    returns = advantages + values[:T]   # V^target = Â + V_old
    return advantages, returns

# ── Example rollout (simulated) ────────────────────────────────
rng = np.random.default_rng(42)
T   = 20    # rollout length

# Simulate: mostly -0.01 step cost, terminal +1 or -1 at end
rewards = np.full(T, -0.01, dtype=np.float32)
rewards[9]  = -1.0   # falls into trap at step 9 (episode end)
rewards[19] = +1.0   # reaches goal at step 19

dones   = np.zeros(T, dtype=np.float32)
dones[9]  = 1.0
dones[19] = 1.0

# Simulated V(s) estimates from critic (near-true values)
values_true = np.linspace(0.3, 0.9, T+1, dtype=np.float32)
values_true[9]  = 0.0   # terminal
values_true[10] = 0.1   # reset, new episode

print("GAE Backward Computation")
print("=" * 70)
print(f"  T={T} steps, γ={GAMMA}")
print()

# Compare different lambda values
lambdas = [0.0, 0.5, 0.95, 1.0]
all_advs = {}
for lam in lambdas:
    adv, ret = compute_gae(rewards, values_true, dones, GAMMA, lam)
    all_advs[lam] = adv

print(f"  {'t':>4} | {'r_t':>7} | {'done':>5} | {'V(s)':>7} | " +
      " | ".join(f"{'A(l='+str(l)+')':>10}" for l in lambdas))
print("  " + "-" * (30 + len(lambdas)*13))
for t in range(T):
    v_str  = f"{values_true[t]:>7.3f}"
    adv_str = " | ".join(f"{all_advs[l][t]:>10.4f}" for l in lambdas)
    done_str = "END" if dones[t] else "   "
    print(f"  {t:>4} | {rewards[t]:>7.3f} | {done_str:>5} | {v_str} | {adv_str}")

print()
print("Advantage statistics (λ=0.95):")
adv95, ret95 = compute_gae(rewards, values_true, dones, GAMMA, 0.95)
print(f"  mean = {adv95.mean():.5f}   std = {adv95.std():.5f}")
print(f"  min  = {adv95.min():.5f}   max = {adv95.max():.5f}")

# Normalisation
adv_norm = (adv95 - adv95.mean()) / (adv95.std() + 1e-8)
print()
print("After normalisation (Â_norm = (Â - mean) / std):")
print(f"  mean = {adv_norm.mean():.6f}   std = {adv_norm.std():.5f}")
print(f"  (mean ≈ 0, std ≈ 1 — consistent gradient scale regardless of reward magnitude)")
print()
print("Value targets (returns = Â + V_old):")
print(f"  First 5: {ret95[:5].round(4)}")
print(f"  Used as regression targets for the critic (V_θ(s) → ret)")
""",
    },

    "3. PPO Actor-Critic Network — Forward Pass and Loss": {
        "description": "Build a PPO actor-critic network in NumPy. Implement the full "
                       "forward pass, compute all three loss components (clip + value + entropy), "
                       "and verify the gradient flows correctly through each term.",
        "code": """\
import numpy as np

# ── Orthogonal initialisation ──────────────────────────────────
def ortho_init(shape, scale, rng):
    rows, cols = shape
    flat = rng.normal(0, 1, (max(rows, cols), min(rows, cols)))
    Q, _ = np.linalg.qr(flat)
    W = Q[:cols, :rows].T if rows < cols else Q[:rows, :cols]
    return (W * scale).astype(np.float32)

# ── Shared actor-critic network ────────────────────────────────
class ActorCritic:
    def __init__(self, n_s, n_a, hidden=64, rng=None):
        if rng is None: rng = np.random.default_rng(0)
        # Shared trunk (tanh activation, scale sqrt(2))
        self.W1 = ortho_init((n_s, hidden),  np.sqrt(2), rng)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = ortho_init((hidden, hidden), np.sqrt(2), rng)
        self.b2 = np.zeros(hidden, np.float32)
        # Policy head — SMALL init (0.01) for near-uniform initial policy
        self.W_pi = ortho_init((hidden, n_a), 0.01, rng)
        self.b_pi = np.zeros(n_a, np.float32)
        # Value head — scale 1.0
        self.W_v  = ortho_init((hidden, 1), 1.0, rng)
        self.b_v  = np.zeros(1, np.float32)
        self.n_a  = n_a

    def trunk(self, x):
        h1 = np.tanh(x   @ self.W1 + self.b1)
        h2 = np.tanh(h1  @ self.W2 + self.b2)
        return h2

    def policy_head(self, h):
        logits = h @ self.W_pi + self.b_pi
        logits = logits - logits.max(axis=-1, keepdims=True)  # numerical stability
        log_p  = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        return log_p

    def value_head(self, h):
        return (h @ self.W_v + self.b_v).squeeze(-1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)

    def entropy(self, log_p):
        p = np.exp(log_p)
        return -(p * log_p).sum(axis=-1)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W_pi, self.b_pi, self.W_v, self.b_v]

    def n_params(self):
        return sum(p.size for p in self.params())


# ── Compute PPO loss on a mini-batch ──────────────────────────
EPSILON  = 0.2
C1       = 0.5    # value loss coeff
C2       = 0.01   # entropy coeff
N_S, N_A = 16, 4
BATCH    = 32

rng = np.random.default_rng(7)
net = ActorCritic(N_S, N_A, hidden=64, rng=rng)

# Fake mini-batch
s_idx       = rng.integers(0, N_S, BATCH)
states      = np.eye(N_S, dtype=np.float32)[s_idx]
actions     = rng.integers(0, N_A, BATCH)
advantages  = rng.normal(0, 1, BATCH).astype(np.float32)  # pre-normalised
log_pi_old  = rng.uniform(-2, -0.5, BATCH).astype(np.float32)  # stored from rollout
v_old       = rng.uniform(0, 1, BATCH).astype(np.float32)
v_target    = v_old + advantages   # returns = adv + V_old

# Forward pass
log_pi_new, v_new = net.forward(states)
log_pi_a    = log_pi_new[np.arange(BATCH), actions]  # log π(aₜ|sₜ)
entropy     = net.entropy(log_pi_new)

# Probability ratio
ratio       = np.exp(log_pi_a - log_pi_old)

# Policy loss (clipped surrogate, MAXIMISE → negate for min)
L_clip_unclipped = ratio * advantages
L_clip_clipped   = np.clip(ratio, 1-EPSILON, 1+EPSILON) * advantages
L_clip           = np.minimum(L_clip_unclipped, L_clip_clipped).mean()

# Value loss (MSE)
L_value = np.mean((v_new - v_target)**2)

# Entropy bonus
H_mean  = entropy.mean()

# Combined PPO loss (for gradient descent: MAXIMISE L^CLIP + c2*H - c1*L_V)
L_total = -L_clip + C1 * L_value - C2 * H_mean   # sign for minimisation

# Diagnostics
clip_frac   = np.mean(np.abs(ratio - 1.0) > EPSILON)
approx_kl   = np.mean(log_pi_old - log_pi_a)

print("PPO Actor-Critic Network — Loss Components")
print("=" * 58)
print(f"  Architecture: {N_S} → 64 → 64 → [π(·|s), V(s)]")
print(f"  Params: {net.n_params():,}")
print(f"  Mini-batch: {BATCH} transitions")
print()
print(f"  Policy head logits range: [{log_pi_new.min():.3f}, {log_pi_new.max():.3f}]")
print(f"  Approx initial policy uniformity:")
initial_max_ratio = np.exp(log_pi_new.max(axis=1) - log_pi_new.min(axis=1))
print(f"    max(π)/min(π) per state: {initial_max_ratio.mean():.3f}  "
      f"(≈1.0 = near-uniform, small init is working)")
print()
print("  Loss components:")
print(f"    L^CLIP    = {L_clip:>10.5f}  (policy surrogate, maximise)")
print(f"    L^VF      = {L_value:>10.5f}  (value MSE, minimise)")
print(f"    H(π)      = {H_mean:>10.5f}  (entropy, maximise)")
print(f"    L^PPO     = {L_total:>10.5f}  (combined loss, minimise)")
print(f"    c₁={C1}, c₂={C2}:  L = -L^CLIP + {C1}·L^VF - {C2}·H")
print()
print("  Diagnostics:")
print(f"    ratio range:  [{ratio.min():.3f}, {ratio.max():.3f}]")
print(f"    clip_frac:    {clip_frac:.3f}  (fraction of ratios clipped)")
print(f"    approx_kl:    {approx_kl:.5f}  (KL between old and new policy)")
print(f"    V range:      [{v_new.min():.3f}, {v_new.max():.3f}]")
print(f"    Entropy:      {H_mean:.4f} nats  (max = {np.log(N_A):.3f} for uniform)")
""",
    },

    "4. Full PPO Training Loop on Grid World": {
        "description": "Train a complete PPO agent on the grid world: rollout collection, "
                       "GAE computation, mini-batch SGD across K epochs. Tracks all "
                       "diagnostics: clip fraction, KL, explained variance, entropy.",
        "code": """\
import numpy as np

# ── Environment ────────────────────────────────────────────────
ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]; ANAMES=['U','D','L','R']

def env_step(s,a):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2,r2,s2 in (GOAL,TRAP)

# ── Minimal ActorCritic ────────────────────────────────────────
def ortho(shape, scale, rng):
    rows, cols = shape
    flat = rng.normal(0, 1, (max(rows, cols), min(rows, cols)))
    Q, _ = np.linalg.qr(flat)
    W = Q[:cols, :rows].T if rows < cols else Q[:rows, :cols]
    return (W * scale).astype(np.float32)

class AC:
    def __init__(self, rng, hid=64):
        s2=np.sqrt(2)
        self.W1=ortho((N_S,hid),s2,rng);  self.b1=np.zeros(hid,np.float32)
        self.W2=ortho((hid,hid),s2,rng);  self.b2=np.zeros(hid,np.float32)
        self.Wp=ortho((hid,N_A),0.01,rng);self.bp=np.zeros(N_A,np.float32)
        self.Wv=ortho((hid,1),1.0,rng);   self.bv=np.zeros(1,np.float32)

    def forward(self, x):
        h1=np.tanh(x@self.W1+self.b1)
        h2=np.tanh(h1@self.W2+self.b2)
        self.h1=h1; self.h=h2; self.x=x
        logits=h2@self.Wp+self.bp
        logits-=logits.max(-1,keepdims=True)
        log_p=logits-np.log(np.exp(logits).sum(-1,keepdims=True))
        v=(h2@self.Wv+self.bv).squeeze(-1)
        return log_p, v

    def sgd(self, grad_Wp, grad_bp, grad_Wv, grad_bv,
            grad_W2, grad_b2, grad_W1, grad_b1, lr=3e-4):
        for p,g in [(self.W1,grad_W1),(self.b1,grad_b1),
                    (self.W2,grad_W2),(self.b2,grad_b2),
                    (self.Wp,grad_Wp),(self.bp,grad_bp),
                    (self.Wv,grad_Wv),(self.bv,grad_bv)]:
            norm=np.sqrt(np.sum(g**2))
            if norm>0.5: g=g*0.5/norm    # grad clip
            p -= lr*g

def compute_gae(rs,vs,ds,gamma=GAMMA,lam=0.95):
    T=len(rs); A=np.zeros(T,np.float32); la=0.0
    for t in reversed(range(T)):
        delta=rs[t]+gamma*vs[t+1]*(1-ds[t])-vs[t]
        A[t]=la=delta+gamma*lam*(1-ds[t])*la
    return A, A+vs[:T]

# ── PPO Training ───────────────────────────────────────────────
rng=np.random.default_rng(0)
ac=AC(rng)
ROLLOUT_T=256; EPOCHS=4; BATCH_SZ=64; EPSILON=0.2
C1=0.5; C2=0.01; LR=3e-4
N_UPDATES=200; all_returns=[]; all_diag=[]

for update in range(N_UPDATES):
    # ── Phase 1: collect rollout ───────────────────────────────
    buf_s=[]; buf_a=[]; buf_r=[]; buf_d=[]; buf_lp=[]; buf_v=[]
    s=0; ep_ret=0.0; ep_returns=[]
    for _ in range(ROLLOUT_T):
        oh=np.eye(N_S,dtype=np.float32)[[s]]
        lp,v=ac.forward(oh)
        probs=np.exp(lp[0]); a=rng.choice(N_A,p=probs)
        s2,r,done=env_step(s,a)
        buf_s.append(s); buf_a.append(a); buf_r.append(r)
        buf_d.append(float(done)); buf_lp.append(float(lp[0,a]))
        buf_v.append(float(v[0]))
        ep_ret+=r
        if done: ep_returns.append(ep_ret); ep_ret=0.0; s=0
        else: s=s2
    # Bootstrap value
    oh_last=np.eye(N_S,dtype=np.float32)[[s]]
    _,v_last=ac.forward(oh_last)
    buf_v_ext=np.array(buf_v+[float(v_last[0])],np.float32)

    rs=np.array(buf_r,np.float32); ds=np.array(buf_d,np.float32)
    advs,rets=compute_gae(rs,buf_v_ext,ds)

    # ── Phase 2: K epochs of mini-batch updates ───────────────
    SS=np.eye(N_S,dtype=np.float32)[np.array(buf_s)]
    AA=np.array(buf_a); LP_old=np.array(buf_lp); V_old=np.array(buf_v,np.float32)

    clip_fracs=[]; approx_kls=[]; entropies=[]; vlosses=[]

    for epoch in range(EPOCHS):
        idx=rng.permutation(ROLLOUT_T)
        for start in range(0,ROLLOUT_T,BATCH_SZ):
            mb=idx[start:start+BATCH_SZ]
            x=SS[mb]; aa=AA[mb]; lp_old=LP_old[mb]
            adv_mb=advs[mb]; ret_mb=rets[mb]; v_old_mb=V_old[mb]

            # Normalise advantages
            adv_mb=(adv_mb-adv_mb.mean())/(adv_mb.std()+1e-8)

            lp_new,v_new=ac.forward(x)
            lp_a=lp_new[np.arange(len(mb)),aa]
            ratio=np.exp(lp_a-lp_old)
            p_new=np.exp(lp_new)

            # Policy loss
            L1=ratio*adv_mb
            L2=np.clip(ratio,1-EPSILON,1+EPSILON)*adv_mb
            Lpol=-np.minimum(L1,L2).mean()

            # Value loss
            Lval=np.mean((v_new-ret_mb)**2)

            # Entropy
            H=-np.sum(p_new*lp_new,axis=1).mean()

            # Diagnostics
            clip_fracs.append(np.mean(np.abs(ratio-1)>EPSILON))
            approx_kls.append(np.mean(lp_old-lp_a))
            entropies.append(H)
            vlosses.append(Lval)

            # Compute gradients (simplified: finite difference on policy head)
            # Use direct weight update via score function for tabular part
            B=len(mb)
            # dL/d lp_a (policy gradient part)
            dL_dratio = np.where(L1<=L2, adv_mb, np.where(
                (ratio>1+EPSILON)|(ratio<1-EPSILON), 0.0, adv_mb))
            dL_dlp_a  = -dL_dratio * ratio / B  # chain: dratio/dlp = ratio

            # Policy head gradient (score function)
            dLp_dlogits=np.zeros_like(lp_new)
            for i,ai in enumerate(aa):
                dLp_dlogits[i] += dL_dlp_a[i]*(np.eye(N_A)[ai]-p_new[i])

            # Value head gradient
            dLv_dv = 2*(v_new-ret_mb)*C1/B

            # Entropy gradient
            dH_dlogits = C2*(lp_new+1)*p_new/B

            # Combined gradient through output
            dout_pi  = dLp_dlogits - dH_dlogits
            dout_val = dLv_dv

            # Backprop through heads
            dWp = ac.h.T @ dout_pi
            dbp = dout_pi.sum(0)
            dWv = (ac.h.T @ dout_val[:,None])
            dbv = dout_val.sum(0,keepdims=True)

            # Backprop through trunk
            dh2 = dout_pi @ ac.Wp.T + dout_val[:,None] @ ac.Wv.T
            dh2 *= (1.0 - ac.h**2)           # tanh' at layer 2
            dW2 = ac.h1.T @ dh2; db2 = dh2.sum(0)
            dh1 = dh2 @ ac.W2.T
            dh1 *= (1.0 - ac.h1**2)          # tanh' at layer 1
            dW1 = ac.x.T @ dh1; db1 = dh1.sum(0)

            ac.sgd(dWp,dbp,dWv,dbv.ravel(),dW2,db2,dW1,db1,lr=LR)

    all_returns.append(np.mean(ep_returns) if ep_returns else 0.0)
    all_diag.append({
        'clip_frac': np.mean(clip_fracs),
        'kl':        np.mean(approx_kls),
        'entropy':   np.mean(entropies),
        'vloss':     np.mean(vlosses)
    })

def smooth(x,w=20):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

sm=smooth(all_returns)
print("PPO Training on Grid World")
print("=" * 72)
print(f"  rollout={ROLLOUT_T}, epochs={EPOCHS}, batch={BATCH_SZ}, ε={EPSILON}, K={EPOCHS}")
print()
print(f"  {'Update':>7} | {'Return':>8} | {'Smooth':>8} | "
      f"{'clip_f':>7} | {'KL':>9} | {'entropy':>9} | {'vloss':>9}")
print("  " + "-"*70)
for i in [4,9,19,39,79,119,159,199]:
    d=all_diag[i]
    print(f"  {i+1:>7} | {all_returns[i]:>8.4f} | {sm[i]:>8.4f} | "
          f"{d['clip_frac']:>7.3f} | {d['kl']:>9.5f} | "
          f"{d['entropy']:>9.4f} | {d['vloss']:>9.5f}")

print()
# Evaluate final policy
oh=np.eye(N_S,dtype=np.float32)
lp,_=ac.forward(oh)
pi_greedy=np.argmax(lp,axis=1)
print("  Final greedy policy:")
for row in np.array([ANAMES[a] for a in pi_greedy]).reshape(ROWS,COLS):
    print("    "+"  ".join(row))
""",
    },

    "5. Clip Ratio ε — Ablation Study": {
        "description": "Train PPO with five values of ε (0.05, 0.1, 0.2, 0.3, 0.5). "
                       "Measure convergence speed, final performance, and clip fraction. "
                       "Shows why ε=0.2 is the sweet spot for most tasks.",
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

class MinimalAC:
    \"\"\"Lightweight tabular AC for ablation speed.\"\"\"
    def __init__(self, rng):
        # tabular policy and value
        self.theta = rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
        self.V     = np.zeros(N_S,np.float32)

    def log_probs(self, s):
        logits=self.theta[s]-self.theta[s].max()
        return logits-np.log(np.exp(logits).sum())

    def value(self, s): return self.V[s]

def train_ppo(epsilon, seed=42, n_updates=300,
              rollout_t=128, epochs=4, batch=32, gamma=GAMMA):
    rng=np.random.default_rng(seed)
    ac=MinimalAC(rng)
    LR_PI=0.05; LR_V=0.1; LAM=0.95

    all_returns=[]; all_clips=[]

    for update in range(n_updates):
        # Collect rollout
        buf_s=[]; buf_a=[]; buf_r=[]; buf_d=[]
        buf_lp_old=[]; buf_v=[]
        s=0; ep_ret=0.0; ep_rets=[]

        for _ in range(rollout_t):
            lp=ac.log_probs(s); probs=np.exp(lp)
            a=rng.choice(N_A,p=probs)
            s2,r,done=env_step(s,a)
            buf_s.append(s); buf_a.append(a); buf_r.append(r)
            buf_d.append(float(done)); buf_lp_old.append(float(lp[a]))
            buf_v.append(float(ac.value(s)))
            ep_ret+=r
            if done: ep_rets.append(ep_ret); ep_ret=0.0; s=0
            else: s=s2

        v_boot=ac.value(s)
        vs_ext=np.array(buf_v+[v_boot],np.float32)
        rs=np.array(buf_r,np.float32); ds=np.array(buf_d,np.float32)

        # GAE
        T=len(rs); advs=np.zeros(T,np.float32); la=0.0
        for t in reversed(range(T)):
            delta=rs[t]+gamma*vs_ext[t+1]*(1-ds[t])-vs_ext[t]
            advs[t]=la=delta+gamma*LAM*(1-ds[t])*la
        rets=advs+vs_ext[:T]

        # K-epoch mini-batch update
        SS=np.array(buf_s); AA=np.array(buf_a)
        LP_old=np.array(buf_lp_old)
        clip_fracs=[]

        for _ in range(epochs):
            idx=rng.permutation(T)
            for i0 in range(0,T,batch):
                mb=idx[i0:i0+batch]
                adv_mb=advs[mb]; adv_mb=(adv_mb-adv_mb.mean())/(adv_mb.std()+1e-8)
                for j,t_idx in enumerate(mb):
                    s_t=SS[t_idx]; a_t=AA[t_idx]
                    lp_new_t=ac.log_probs(s_t)
                    ratio=np.exp(lp_new_t[a_t]-LP_old[t_idx])
                    adv_t=float(adv_mb[j])
                    clip_fracs.append(abs(ratio-1)>epsilon)

                    # Clip gradient signal
                    if (ratio > 1+epsilon and adv_t > 0) or \
                       (ratio < 1-epsilon and adv_t < 0):
                        eff_ratio=np.clip(ratio,1-epsilon,1+epsilon)
                    else:
                        eff_ratio=ratio

                    # Policy update (score function)
                    probs=np.exp(lp_new_t)
                    g=np.zeros(N_A,np.float32)
                    g[a_t]+=1.0; g-=probs
                    ac.theta[s_t] += LR_PI * eff_ratio * adv_t * g

                    # Value update
                    ac.V[s_t] += LR_V * (rets[t_idx] - ac.V[s_t])

        all_returns.append(np.mean(ep_rets) if ep_rets else 0.0)
        all_clips.append(np.mean(clip_fracs))

    return all_returns, all_clips

def smooth(x,w=30):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

epsilons=[0.05, 0.1, 0.2, 0.3, 0.5]
results={}
for eps in epsilons:
    rets, clips = train_ppo(eps)
    results[eps]=(smooth(rets), np.mean(clips[-50:]), np.mean(rets[-50:]))

print("PPO Clip Ratio ε Ablation")
print("=" * 70)
print()
hdr="  "+f"{'Update':>7} | "+" | ".join(f"{'e='+str(e):>9}" for e in epsilons)
print(hdr)
print("  "+"-"*(9+len(epsilons)*12))
for i in [9, 29, 59, 99, 149, 199, 249, 299]:
    vals=" | ".join(f"{results[e][0][i]:>9.4f}" for e in epsilons)
    print(f"  {i+1:>7} | {vals}")

print()
print("  Final performance (avg last 50 updates):")
for e in epsilons:
    sm_ret, avg_clip, final_ret = results[e]
    stars=""
    if e==0.2: stars=" ← standard"
    print(f"    ε={e:.2f}: return={final_ret:>7.4f}  avg_clip_frac={avg_clip:.3f}{stars}")

print()
print("  Interpretation:")
print("    ε too small (0.05): over-constrained, slow learning, low clip fraction")
print("    ε = 0.2: balanced — meaningful updates, ~10% clip fraction")
print("    ε too large (0.5): ~vanilla PG, unstable, high clip fraction but not helping")
""",
    },

    "6. K Epochs Per Update — Reuse vs Staleness": {
        "description": "Vary the number of PPO optimisation epochs K per rollout batch "
                       "(1, 2, 4, 10, 20). Show that too many epochs cause the policy "
                       "to move too far from π_old, breaking importance sampling validity.",
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

def train(K_epochs, seed=0, n_updates=250, rollout=128, batch=32):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    V=np.zeros(N_S,np.float32)
    EPSILON=0.2; LR_PI=0.05; LR_V=0.1; LAM=0.95; C=GAMMA

    all_returns=[]; all_kl=[]; all_clip=[]

    for update in range(n_updates):
        buf_s=[]; buf_a=[]; buf_r=[]; buf_d=[]; buf_lp=[]; buf_v=[]
        s=0; ep_ret=0.0; ep_rets=[]
        for _ in range(rollout):
            logits=theta[s]-theta[s].max()
            lp=logits-np.log(np.exp(logits).sum())
            probs=np.exp(lp); a=rng.choice(N_A,p=probs)
            s2,r,done=env_step(s,a)
            buf_s.append(s); buf_a.append(a); buf_r.append(r)
            buf_d.append(float(done)); buf_lp.append(float(lp[a]))
            buf_v.append(float(V[s]))
            ep_ret+=r
            if done: ep_rets.append(ep_ret); ep_ret=0.0; s=0
            else: s=s2

        vs=np.array(buf_v+[V[s]],np.float32)
        rs=np.array(buf_r,np.float32); ds=np.array(buf_d,np.float32)
        T=len(rs); A=np.zeros(T,np.float32); la=0.0
        for t in reversed(range(T)):
            d=rs[t]+C*vs[t+1]*(1-ds[t])-vs[t]
            A[t]=la=d+C*LAM*(1-ds[t])*la
        rets=A+vs[:T]
        SS=np.array(buf_s); AA=np.array(buf_a); LP_old=np.array(buf_lp)

        update_kl=[]; update_clip=[]
        for k in range(K_epochs):
            idx=rng.permutation(T)
            for i0 in range(0,T,batch):
                mb=idx[i0:i0+batch]
                adv_mb=A[mb]; adv_mb=(adv_mb-adv_mb.mean())/(adv_mb.std()+1e-8)
                for j,ti in enumerate(mb):
                    s_t=SS[ti]; a_t=AA[ti]; lp_old=LP_old[ti]
                    logits=theta[s_t]-theta[s_t].max()
                    lp_new=logits-np.log(np.exp(logits).sum())
                    ratio=np.exp(lp_new[a_t]-lp_old)
                    adv_t=float(adv_mb[j])
                    kl=float(lp_old-lp_new[a_t])
                    update_kl.append(kl)
                    update_clip.append(abs(ratio-1)>EPSILON)
                    if (ratio>1+EPSILON and adv_t>0) or (ratio<1-EPSILON and adv_t<0):
                        eff=np.clip(ratio,1-EPSILON,1+EPSILON)
                    else: eff=ratio
                    probs=np.exp(lp_new); g=np.zeros(N_A,np.float32)
                    g[a_t]+=1.0; g-=probs
                    theta[s_t]+=LR_PI*eff*adv_t*g
                    V[s_t]+=LR_V*(rets[ti]-V[s_t])

        all_returns.append(np.mean(ep_rets) if ep_rets else 0.0)
        all_kl.append(np.mean(update_kl))
        all_clip.append(np.mean(update_clip))

    return all_returns, all_kl, all_clip

def smooth(x,w=30):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

K_vals=[1,2,4,10,20]
res={}
for K in K_vals:
    rets,kls,clips=train(K)
    res[K]=(smooth(rets),np.mean(kls[-50:]),np.mean(clips[-50:]),np.mean(rets[-50:]))

print("PPO Epochs per Update (K) Ablation")
print("=" * 68)
print()
print(f"  {'Update':>7} | "+" | ".join(f"{'K='+str(k):>9}" for k in K_vals))
print("  "+"-"*(9+len(K_vals)*12))
for i in [9,29,59,99,149,199,249]:
    vals=" | ".join(f"{res[k][0][i]:>9.4f}" for k in K_vals)
    print(f"  {i+1:>7} | {vals}")

print()
print("  Final stats (avg last 50 updates):")
print(f"  {'K':>4} | {'return':>9} | {'avg KL':>10} | {'avg clip_f':>12}")
print("  "+"-"*40)
for K in K_vals:
    _,avg_kl,avg_clip,final_ret=res[K]
    note=" ← default" if K==4 else ""
    print(f"  {K:>4} | {final_ret:>9.4f} | {avg_kl:>10.5f} | {avg_clip:>12.4f}{note}")

print()
print("  Interpretation:")
print("    K=1:  no data reuse — sample inefficient (like vanilla PG)")
print("    K=4:  good balance — each transition used ~4x without staleness")
print("    K=20: over-optimised — policy drifts far from π_old, high KL, degrades")
""",
    },

    "7. Advantage Normalisation — Effect on Stability": {
        "description": "Compare PPO training with and without advantage normalisation "
                       "under different reward scales. Shows how normalisation makes "
                       "the gradient scale invariant to the reward magnitude.",
        "code": """\
import numpy as np

ROWS,COLS=4,4; N_S=16; N_A=4; GOAL=15; TRAP=5; GAMMA=0.99
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]

def env_step_scaled(s, a, scale=1.0):
    if s in (GOAL,TRAP): return s,0.0,True
    r,c=s//COLS,s%COLS; dr,dc=MOVES[a]
    nr,nc=r+dr,c+dc
    s2=(nr*COLS+nc) if 0<=nr<ROWS and 0<=nc<COLS else s
    r2=+1.0 if s2==GOAL else -1.0 if s2==TRAP else -0.01
    return s2, r2*scale, s2 in (GOAL,TRAP)

def train(reward_scale, normalise_adv, seed=0, n_updates=200,
          rollout=128, epochs=4, batch=32):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    V=np.zeros(N_S,np.float32)
    EPSILON=0.2; LR_PI=0.05; LR_V=0.1; LAM=0.95; C=GAMMA

    all_returns=[]; grad_norms=[]; value_losses=[]

    for update in range(n_updates):
        buf_s=[]; buf_a=[]; buf_r=[]; buf_d=[]; buf_lp=[]; buf_v=[]
        s=0; ep_ret=0.0; ep_rets=[]
        for _ in range(rollout):
            logits=theta[s]-theta[s].max()
            lp=logits-np.log(np.exp(logits).sum())
            probs=np.exp(lp); a=rng.choice(N_A,p=probs)
            s2,r,done=env_step_scaled(s,a,reward_scale)
            buf_s.append(s); buf_a.append(a); buf_r.append(r)
            buf_d.append(float(done)); buf_lp.append(float(lp[a]))
            buf_v.append(float(V[s]))
            ep_ret+=r
            if done: ep_rets.append(ep_ret/reward_scale); ep_ret=0.0; s=0
            else: s=s2

        vs=np.array(buf_v+[V[s]],np.float32)
        rs=np.array(buf_r,np.float32); ds=np.array(buf_d,np.float32)
        T=len(rs); A=np.zeros(T,np.float32); la=0.0
        for t in reversed(range(T)):
            d=rs[t]+C*vs[t+1]*(1-ds[t])-vs[t]
            A[t]=la=d+C*LAM*(1-ds[t])*la
        rets=A+vs[:T]
        SS=np.array(buf_s); AA=np.array(buf_a); LP_old=np.array(buf_lp)

        ep_grad_norm=[]; ep_vloss=[]
        for _ in range(epochs):
            idx=rng.permutation(T)
            for i0 in range(0,T,batch):
                mb=idx[i0:i0+batch]
                adv_mb=A[mb].copy()
                if normalise_adv:
                    adv_mb=(adv_mb-adv_mb.mean())/(adv_mb.std()+1e-8)
                batch_grad=np.zeros_like(theta)
                vloss_sum=0.0
                for j,ti in enumerate(mb):
                    s_t=SS[ti]; a_t=AA[ti]; lp_old=LP_old[ti]
                    logits=theta[s_t]-theta[s_t].max()
                    lp_new=logits-np.log(np.exp(logits).sum())
                    ratio=np.exp(lp_new[a_t]-lp_old)
                    adv_t=float(adv_mb[j])
                    if (ratio>1+EPSILON and adv_t>0) or (ratio<1-EPSILON and adv_t<0):
                        eff=np.clip(ratio,1-EPSILON,1+EPSILON)
                    else: eff=ratio
                    probs=np.exp(lp_new); g=np.zeros(N_A,np.float32)
                    g[a_t]+=1.0; g-=probs
                    batch_grad[s_t]+=LR_PI*eff*adv_t*g
                    vloss_sum+=(rets[ti]-V[s_t])**2
                    V[s_t]+=LR_V*(rets[ti]-V[s_t])
                theta+=batch_grad
                ep_grad_norm.append(np.linalg.norm(batch_grad))
                ep_vloss.append(vloss_sum/len(mb))

        all_returns.append(np.mean(ep_rets) if ep_rets else 0.0)
        grad_norms.append(np.mean(ep_grad_norm))
        value_losses.append(np.mean(ep_vloss))

    return all_returns, grad_norms, value_losses

def smooth(x,w=30):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

scales=[1.0, 10.0, 100.0]
print("Advantage Normalisation — Effect on Stability")
print("=" * 68)
print()

for scale in scales:
    r_norm,   gn_norm,  vl_norm  = train(scale, True)
    r_nonorm, gn_nonorm, vl_nonorm = train(scale, False)
    print(f"  Reward scale = {scale:.0f}x")
    print(f"  {'Update':>7} | {'norm return':>12} | {'no-norm return':>14} | "
          f"{'norm gn':>9} | {'no-norm gn':>11}")
    print("  "+"-"*60)
    for i in [9,49,99,149,199]:
        print(f"  {i+1:>7} | {smooth(r_norm)[i]:>12.4f} | {smooth(r_nonorm)[i]:>14.4f} | "
              f"{smooth(gn_norm)[i]:>9.3f} | {smooth(gn_nonorm)[i]:>11.3f}")
    print(f"  Final: norm={np.mean(r_norm[-50:]):.4f}  "
          f"no-norm={np.mean(r_nonorm[-50:]):.4f}  "
          f"grad_norm ratio={np.mean(gn_nonorm[-50:])/max(np.mean(gn_norm[-50:]),1e-5):.1f}x larger without norm")
    print()

print("  Key insight: without normalisation, reward scale×10 → gradient×10.")
print("  With normalisation, gradient scale is always O(1) regardless of reward scale.")
""",
    },

    "8. PPO vs REINFORCE vs Actor-Critic — Full Comparison": {
        "description": "Train all three algorithms side-by-side on the same task with "
                       "the same number of environment interactions. Compare convergence "
                       "speed, final performance, gradient variance, and stability.",
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

# ── Shared tabular setup ───────────────────────────────────────
TOTAL_STEPS=50_000   # same env interactions for all methods

def lp_from_theta(theta, s):
    logits=theta[s]-theta[s].max()
    return logits-np.log(np.exp(logits).sum())

def probs_from_theta(theta, s):
    return np.exp(lp_from_theta(theta, s))

# ── REINFORCE with baseline ────────────────────────────────────
def train_reinforce(seed=0):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    V=np.zeros(N_S,np.float32); LR=0.05; LR_V=0.1
    returns=[]; total_steps=0; grad_norms=[]

    while total_steps < TOTAL_STEPS:
        s=0; traj=[]
        for _ in range(200):
            probs=probs_from_theta(theta,s)
            a=rng.choice(N_A,p=probs); s2,r,done=env_step(s,a)
            traj.append((s,a,r)); total_steps+=1; s=s2
            if done: break
        T=len(traj); G=0.0; Gs=np.zeros(T)
        for t in reversed(range(T)):
            G=traj[t][2]+GAMMA*G; Gs[t]=G
        g_total=np.zeros_like(theta)
        for t,(s,a,_) in enumerate(traj):
            adv=Gs[t]-V[s]; V[s]+=LR_V*(Gs[t]-V[s])
            probs=probs_from_theta(theta,s)
            g=np.zeros(N_A,np.float32); g[a]+=1.0; g-=probs
            g_total[s]+=LR*adv*g/T
        theta+=g_total
        grad_norms.append(np.linalg.norm(g_total))
        G0=sum(GAMMA**t*r for t,(_,_,r) in enumerate(traj))
        returns.append(G0)
    return returns, grad_norms

# ── One-Step Actor-Critic ──────────────────────────────────────
def train_ac(seed=0):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    V=np.zeros(N_S,np.float32); LR_PI=0.03; LR_V=0.1
    returns=[]; grad_norms=[]; ep_ret=0.0; ep_gamma=1.0; s=0

    for step in range(TOTAL_STEPS):
        probs=probs_from_theta(theta,s); a=rng.choice(N_A,p=probs)
        s2,r,done=env_step(s,a)
        ep_ret+=ep_gamma*r; ep_gamma*=GAMMA
        adv=r+GAMMA*V[s2]*(1-done)-V[s]
        V[s]+=LR_V*adv
        probs2=probs_from_theta(theta,s)
        g=np.zeros(N_A,np.float32); g[a]+=1.0; g-=probs2
        update=LR_PI*adv*g
        theta[s]+=update; grad_norms.append(np.linalg.norm(update))
        if done: returns.append(ep_ret); ep_ret=0.0; ep_gamma=1.0; s=0
        else: s=s2
    return returns, grad_norms

# ── PPO (on-policy, K epochs) ─────────────────────────────────
def train_ppo(seed=0):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,(N_S,N_A)).astype(np.float32)
    V=np.zeros(N_S,np.float32)
    ROLLOUT=128; EPOCHS=4; BATCH=32; EPS=0.2; LR=0.05; LR_V=0.1; LAM=0.95

    returns=[]; grad_norms=[]; total_steps=0

    while total_steps < TOTAL_STEPS:
        buf_s=[]; buf_a=[]; buf_r=[]; buf_d=[]; buf_lp=[]; buf_v=[]
        s=0; ep_ret=0.0; ep_rets=[]
        for _ in range(ROLLOUT):
            lp=lp_from_theta(theta,s); probs=np.exp(lp); a=rng.choice(N_A,p=probs)
            s2,r,done=env_step(s,a)
            buf_s.append(s); buf_a.append(a); buf_r.append(r)
            buf_d.append(float(done)); buf_lp.append(float(lp[a])); buf_v.append(float(V[s]))
            ep_ret+=r; total_steps+=1
            if done: ep_rets.append(ep_ret); ep_ret=0.0; s=0
            else: s=s2

        vs=np.array(buf_v+[V[s]],np.float32)
        rs=np.array(buf_r,np.float32); ds=np.array(buf_d,np.float32)
        T=len(rs); A=np.zeros(T,np.float32); la=0.0
        for t in reversed(range(T)):
            d=rs[t]+GAMMA*vs[t+1]*(1-ds[t])-vs[t]
            A[t]=la=d+GAMMA*LAM*(1-ds[t])*la
        rets=A+vs[:T]
        SS=np.array(buf_s); AA=np.array(buf_a); LP_old=np.array(buf_lp)

        g_ep=[]
        for _ in range(EPOCHS):
            idx=rng.permutation(T)
            for i0 in range(0,T,BATCH):
                mb=idx[i0:i0+BATCH]
                adv_mb=A[mb].copy()
                adv_mb=(adv_mb-adv_mb.mean())/(adv_mb.std()+1e-8)
                g_total=np.zeros_like(theta)
                for j,ti in enumerate(mb):
                    s_t=SS[ti]; a_t=AA[ti]; lp_old=LP_old[ti]
                    lp_new=lp_from_theta(theta,s_t)
                    ratio=np.exp(lp_new[a_t]-lp_old); adv_t=float(adv_mb[j])
                    if (ratio>1+EPS and adv_t>0) or (ratio<1-EPS and adv_t<0):
                        eff=np.clip(ratio,1-EPS,1+EPS)
                    else: eff=ratio
                    probs=np.exp(lp_new); g=np.zeros(N_A,np.float32)
                    g[a_t]+=1.0; g-=probs; g_total[s_t]+=LR*eff*adv_t*g
                    V[s_t]+=LR_V*(rets[ti]-V[s_t])
                theta+=g_total; g_ep.append(np.linalg.norm(g_total))

        if ep_rets: returns.extend(ep_rets)
        grad_norms.extend(g_ep)

    return returns, grad_norms

def smooth(x,w=50):
    return [np.mean(x[max(0,i-w):i+1]) for i in range(len(x))]

r_rf,  gn_rf  = train_reinforce()
r_ac,  gn_ac  = train_ac()
r_ppo, gn_ppo = train_ppo()

sm_rf=smooth(r_rf); sm_ac=smooth(r_ac); sm_ppo=smooth(r_ppo)

# Align by episode count
N=min(len(sm_rf),len(sm_ac),len(sm_ppo))
print("Algorithm Comparison: REINFORCE vs Actor-Critic vs PPO")
print(f"  Total environment steps: {TOTAL_STEPS:,} each")
print("=" * 66)
print()
print(f"  {'Episode':>8} | {'REINFORCE':>12} | {'Actor-Critic':>14} | {'PPO':>10}")
print("  "+"-"*50)
checkpoints=[int(N*f) for f in [0.02,0.1,0.2,0.4,0.6,0.8,1.0] if int(N*f)<N]
for i in checkpoints:
    print(f"  {i+1:>8} | {sm_rf[i]:>12.4f} | {sm_ac[i]:>14.4f} | {sm_ppo[i]:>10.4f}")

print()
print("  Final performance (last 20% of episodes):")
n20=max(1,N//5)
print(f"    REINFORCE:    {np.mean(r_rf[-n20:]):.4f}  (grad norm std: {np.std(gn_rf):.4f})")
print(f"    Actor-Critic: {np.mean(r_ac[-n20:]):.4f}  (grad norm std: {np.std(gn_ac):.4f})")
print(f"    PPO:          {np.mean(r_ppo[-n20:]):.4f}  (grad norm std: {np.std(gn_ppo):.4f})")
print()
print("  Summary:")
print("    REINFORCE:    high variance, full episodes needed, slow convergence")
print("    Actor-Critic: online updates, faster, lower variance than REINFORCE")
print("    PPO:          K epochs per batch, most sample-efficient, lowest variance")
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