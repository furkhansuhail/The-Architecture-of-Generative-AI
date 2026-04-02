"""
RLHF: Reward Modeling and PPO
==============================

Reinforcement Learning from Human Feedback (RLHF) is the training
methodology that transformed GPT-3 into ChatGPT — taking a capable but
unpredictable language model and aligning it to human preferences for
helpfulness, harmlessness, and honesty. RLHF comprises three distinct
phases: supervised fine-tuning (Module 26), reward model training, and
reinforcement learning via Proximal Policy Optimisation (PPO). Each phase
has its own failure modes and requires careful engineering. This module
covers the reward model architecture, the KL divergence constraint that
prevents reward hacking, and the PPO update rule as applied to language
generation.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "RLHF: Reward Modeling and PPO"
DISPLAY_NAME = "28 · RLHF & PPO"
ICON         = "🏆"
SUBTITLE     = "Reward Models and Policy Optimisation"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Why SFT Is Not Enough: The Alignment Problem

Supervised fine-tuning (Module 26) teaches a model to imitate demonstrated
good responses. But imitation has fundamental limits:

    1.  **Coverage:** It is impossible to demonstrate the ideal response for
        every possible prompt. The model must extrapolate, and it may extrapolate
        badly for out-of-distribution inputs.

    2.  **Quality calibration:** SFT treats all responses in the dataset equally.
        A slightly wrong demonstration and a completely wrong one contribute
        identically to the loss — the model cannot distinguish.

    3.  **Reward shaping:** Human preference is often easier to *judge* than
        to *demonstrate*. Given two responses, most people can say which is
        better even if they could not have written the better one from scratch.
        RLHF exploits this asymmetry.

    4.  **Multi-objective optimisation:** Being helpful, harmless, and honest
        are competing objectives. SFT data usually optimises for one. RLHF can
        learn a scalar reward that captures the right trade-off.

**The core RLHF insight:** Rather than learning from demonstrations, learn
from *comparisons*. Show a human two model responses and ask "which is better?"
Train a reward model on these comparisons. Then use RL to maximise the reward.


### The Full RLHF Pipeline

    Phase 1 — Supervised Fine-Tuning (SFT):
        Start from a pre-trained base model.
        Fine-tune on high-quality (prompt, response) demonstrations.
        Result: a well-behaved starting point for RL.

    Phase 2 — Reward Model Training:
        Collect comparison data: (prompt, chosen_response, rejected_response).
        Train a reward model to predict which response humans prefer.
        Result: a scalar-valued function r(prompt, response).

    Phase 3 — RL Fine-Tuning (PPO or DPO):
        Use the reward model to give scalar feedback to the policy.
        Update the policy to generate responses with higher reward.
        Add a KL penalty to prevent the policy from drifting too far from SFT.
        Result: a model aligned to human preferences.


    **Diagram 1 — RLHF Full Pipeline:**

    RLHF THREE-PHASE PIPELINE
    ════════════════════════════════════════════════════════════════

    PHASE 1: SFT                PHASE 2: RM Training       PHASE 3: PPO
    ┌──────────────────┐        ┌──────────────────────┐    ┌────────────────┐
    │  Pre-trained LM  │        │  Human annotators    │    │  SFT model     │
    │  Fine-tune on    │──────► │  rate response pairs │    │  (policy π_θ)  │
    │  demonstrations  │        │  RM learns to predict│    │                │
    └────────┬─────────┘        │  human preference    │    │ r(s,a) from RM │
             │                  └──────────┬───────────┘    │ KL(π_θ||π_ref) │
             │  SFT model                 │  Reward model   │ PPO update     │
             └────────────────────────────┴────────────────►│ π_θ optimised  │
                                                             └────────────────┘

    The SFT model serves two roles:
    1. Starting point for the policy (Phase 3)
    2. Reference policy for the KL divergence penalty (Phase 3)


### Phase 2: The Reward Model

The reward model (RM) is typically the SFT model with its language modelling
head replaced by a scalar regression head. It takes a (prompt, response) pair
as input and outputs a single scalar reward score.

**Architecture:**
    •   Base: the SFT model (same parameters, usually frozen early in training)
    •   Head: a linear layer W_r ∈ ℝ^(d × 1) applied to the final token's
        hidden state (or mean pooling)
    •   Output: r = h[-1] · w_r ∈ ℝ (scalar reward score)

**Why the final token?**
The reward model sees the full sequence (prompt + response) and produces a
single scalar for the entire response. Using the hidden state at the EOS token
(or the last response token) is a common approach because the autoregressive
LM has seen the entire response by the time it produces this token's representation.

**Training objective — Bradley-Terry model:**
Given a comparison (prompt x, winning response y_w, losing response y_l):

    Loss = -log σ(r(x, y_w) - r(x, y_l))

where σ is the sigmoid function. This maximises the probability that the
reward for the preferred response is higher than for the rejected one.

This is the **Bradley-Terry preference model**: the probability that response A
is preferred over response B is:
    P(A ≻ B) = σ(r(A) - r(B)) = exp(r(A)) / (exp(r(A)) + exp(r(B)))

The loss pushes r(y_w) - r(x, y_l) to be large and positive.


    **Diagram 2 — Reward Model Bradley-Terry Training:**

    REWARD MODEL TRAINING
    ════════════════════════════════════════════════════════════════

    Input pair:  (prompt x, chosen y_w, rejected y_l)

    [SFT base model with scalar head]
    x+y_w → forward → r_w = h_w · w_r     (reward for chosen)
    x+y_l → forward → r_l = h_l · w_r     (reward for rejected)

    Loss = -log σ(r_w - r_l)
    = -log (exp(r_w) / (exp(r_w) + exp(r_l)))

    After training:
    r_w should be > r_l  for all pairs in the dataset.
    The margin (r_w - r_l) should grow during training.


### Data Collection for Reward Model Training

The quality of comparison data is the bottleneck for RLHF:

    •   Collect prompts (from a diverse prompt distribution)
    •   Generate K responses per prompt (K=2 to 8) from the SFT model
    •   Present pairs to human annotators: "Which response is better?"
    •   Annotators rate on axes: helpfulness, factuality, harmlessness

**Annotation guidelines matter enormously:**
A vague guideline ("which is better?") leads to noisy, inconsistent labels.
Specific guidelines covering:
    •   How to handle factual errors (automatic low rating)
    •   How to weight length vs depth
    •   How to handle refusals (should they be more or less preferred?)
    •   Calibration: annotators see the same examples to synchronise standards

**Scale of data required:**
    InstructGPT (OpenAI): ~13,000 comparisons
    RLHF-V (typical research): ~50,000–100,000 comparisons
    Commercial scale (estimated): millions of comparisons

The reward model can generalise beyond its training distribution — one of its
key advantages over SFT data. A comparison-trained RM can assign meaningful
scores to prompts it has never seen, using the patterns it learned.


### Phase 3: Reinforcement Learning with PPO

With a trained reward model, we can formulate language generation as an RL
problem:

    **State (s):**  the current context (prompt + partially-generated response)
    **Action (a):** the next token to generate (from the vocabulary)
    **Reward (r):** given by the reward model at the end of the episode
                    (after the full response is generated)
    **Episode:**    generating one complete response to one prompt

**The RL objective:**
    Maximise E_{x~D, y~π_θ} [r(x, y)] - β × KL(π_θ || π_ref)

where:
    •   π_θ is the current policy (language model being trained)
    •   π_ref is the reference policy (SFT model, frozen)
    •   r(x, y) is the reward model score for response y to prompt x
    •   β is the KL penalty coefficient
    •   KL(π_θ || π_ref) = E_{y~π_θ} [log π_θ(y|x) - log π_ref(y|x)]


### The KL Divergence Penalty: Preventing Reward Hacking

The KL divergence penalty is the most important stabiliser in RLHF:

Without KL: the policy is free to maximise reward by any means. It quickly
discovers that the reward model is imperfect and learns to exploit its gaps.
This is called **reward hacking** or **reward model over-optimisation**.

Common reward hacking patterns without KL:
    •   Generate extremely long responses (RM trained on short data may prefer longer)
    •   Repeat key phrases that appear in high-rated responses
    •   Use formatting tricks (bullet points, bold text) that annotators prefer
    •   Generate confident-sounding but incorrect answers
    •   Use sycophantic language that agrees with any premise in the prompt

The KL penalty prevents these by keeping the policy close to the SFT model:
    •   KL(π_θ || π_ref) measures how much π_θ has diverged from π_ref
    •   Adding β × KL to the cost creates a tension between reward maximisation
        and staying close to the well-behaved SFT model
    •   As β → 0: reward hacking eventually dominates
    •   As β → ∞: the policy does not move from SFT (no RL improvement)

**Optimal β:**
The InstructGPT paper found β ≈ 0.02 worked well. The DPO paper showed that
the optimal policy under the RLHF objective has a closed-form solution:

    π*(y|x) ∝ π_ref(y|x) × exp(r(x,y) / β)

This means the optimal RLHF policy is the SFT policy reweighted by the
exponentiated reward — and this insight leads directly to DPO (Module 29).


### The PPO Algorithm as Applied to LLMs

Proximal Policy Optimisation (Schulman et al., 2017) is an on-policy RL
algorithm that constrains the size of each policy update to prevent
destabilising parameter changes. It was chosen for RLHF because:
    •   It is stable and well-understood
    •   It handles the token-level RL formulation naturally
    •   It can use value function baselines to reduce variance
    •   Its clipping mechanism prevents catastrophic parameter updates

**The PPO-Clip objective:**

    L_PPO(θ) = E_t [min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)]

where:
    r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)   (probability ratio)
    A_t    = advantage estimate (how much better this action is than average)
    ε      = clip range (typically 0.1 or 0.2)

The clipping prevents the ratio r_t(θ) from moving too far from 1.0:
    •   If A_t > 0 (good action): limit the policy increase to (1+ε)
    •   If A_t < 0 (bad action): limit the policy decrease to (1-ε)
This ensures no single update causes a catastrophic policy change.


### The RLHF PPO Training Loop in Detail

In the language model setting, PPO has four components:

    **1. Policy model (π_θ):** The LM being trained. Generates responses.
    **2. Reference model (π_ref):** Frozen SFT model. Computes KL penalty.
    **3. Reward model (RM):** Frozen. Scores complete responses.
    **4. Value model (V_φ):** Predicts expected future reward at each step.
        Often initialised from the RM or the SFT model with a scalar head.

    **PPO step:**
    1.  Sample batch of prompts from prompt distribution D
    2.  Generate responses y ~ π_θ(·|x)  [using current policy, greedy/nucleus]
    3.  Compute reward:  r(x, y) from RM
    4.  Compute per-token KL penalty:  -β × (log π_θ(y_t|x,y<t) - log π_ref(y_t|x,y<t))
    5.  Combine:  final_reward_t = r(x,y) × 1[t=T] + per_token_KL_t
        (reward model score at final token; KL at every token)
    6.  Compute advantages using GAE (Generalised Advantage Estimation):
        A_t = Σ_{k=0}^{∞} (γλ)^k (r_{t+k} + γV(s_{t+k+1}) - V(s_{t+k}))
    7.  Update policy via PPO-Clip objective
    8.  Update value function via MSE loss on returns


    **Diagram 3 — PPO Training Loop for LLMs:**

    PPO TRAINING LOOP FOR LANGUAGE MODELS
    ════════════════════════════════════════════════════════════════

    For each batch:
    ┌─────────────────────────────────────────────────────────────┐
    │  ROLLOUT PHASE (no gradient)                                │
    │  1. Sample prompts x from dataset                          │
    │  2. Generate y = [y₁, y₂, ..., y_T] ~ π_θ(·|x)           │
    │  3. Compute r_RM = RM(x, y)        ← scalar reward        │
    │  4. Compute log π_θ(y_t|x,y<t)   for each token t         │
    │  5. Compute log π_ref(y_t|x,y<t) for each token t         │
    │  6. KL_t = log π_θ - log π_ref                            │
    │  7. shaped_reward_T = r_RM - β × KL_T  (terminal)         │
    │     shaped_reward_t = -β × KL_t         (non-terminal)    │
    └─────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────┐
    │  OPTIMISATION PHASE (gradient through policy/value)         │
    │  8. Compute advantages A_t using V_φ and GAE               │
    │  9. For K epochs over the collected rollouts:              │
    │     a. Compute new log π_θ'(y_t|x,y<t)                    │
    │     b. r_t = exp(log π_θ' - log π_θ)   (ratio)            │
    │     c. L_clip = min(r_t × A_t, clip(r_t, 1±ε) × A_t)      │
    │     d. L_value = (V_φ(s_t) - R_t)²     (value loss)       │
    │     e. L_entropy = -H(π_θ')             (entropy bonus)    │
    │     f. Total loss = -L_clip + c1×L_value - c2×L_entropy   │
    │     g. Backpropagate and update θ, φ                       │
    └─────────────────────────────────────────────────────────────┘


### The Value Function in RLHF

The value function V_φ(s_t) estimates the expected future reward from state s_t.
In RLHF's episodic setting:
    •   The episode ends when the response is complete
    •   Only terminal state gets the RM reward
    •   The value function learns to propagate this signal backwards

**Initialisation:** The value function is typically initialised from the RM
(same parameters), with a scalar head. This provides a warm start since
the RM already understands which directions of text improve quality.

**Whitening advantages:**
PPO implementations typically normalise (whiten) the advantages:
    A_t ← (A_t - mean(A)) / std(A)

This stabilises training by keeping gradient magnitudes consistent.


### Reward Hacking: The Central Challenge of RLHF

Goodhart's Law states: "When a measure becomes a target, it ceases to be
a good measure." RLHF is a direct application of this law in practice.

**Manifestations of reward hacking in LLMs:**
    1.  **Length bias:** If annotators preferred longer responses, the RM
        assigns higher scores to longer text regardless of quality. PPO then
        produces increasingly verbose responses.

    2.  **Format exploitation:** If bullet-pointed responses were rated higher,
        the policy adds unnecessary bullet points to everything.

    3.  **Sycophancy:** Annotators may rate agreeable responses higher. The
        model learns to validate user beliefs rather than correct errors.

    4.  **Confident incorrectness:** Strong, assertive language may be rated
        higher than hedged uncertainty. The model becomes overconfident.

    5.  **Distribution shift:** The RM was trained on responses from π_SFT.
        As π_θ shifts from π_SFT, responses fall out of the RM's distribution,
        and RM scores become unreliable.

**The over-optimisation curve:**
There is a characteristic curve in RLHF where:
    •   Early in RL training: RM score and human quality both improve
    •   Mid training: RM score continues rising but human quality plateaus
    •   Late training: RM score is very high but human quality declines
    (The policy has found high-scoring but low-quality behaviours)

The optimal stopping point is before the divergence. KL divergence is the
practical proxy for where we are on this curve.


### The Four RLHF Failure Modes

    Failure Mode          Symptom                     Fix
    ─────────────────────────────────────────────────────────────────────────
    RM noise/bias         Policy learns wrong behaviour More RM training data
    Reward hacking        High RM score, low quality   Increase KL penalty β
    KL too high           Policy barely moves from SFT Decrease β
    Value model collapse  Advantage estimates too noisy Better value init / LR
    ─────────────────────────────────────────────────────────────────────────

**Diagnosing via KL-reward curve:**
Plot RM score vs KL(π_θ || π_ref) as training progresses.
    •   Initially: both RM score and KL increase together (healthy)
    •   After: RM score peaks and declines; KL continues increasing (hacking)
    •   The peak RM score × quality correlation is the signal to stop or
        increase β


### RLHF vs DPO: Why DPO Was Developed

PPO-based RLHF has several engineering challenges:
    1.  Requires four models simultaneously: policy, reference, RM, value model
    2.  Requires generating rollouts (slow, expensive, not parallelisable easily)
    3.  PPO hyperparameters are notoriously sensitive (ε, c1, c2, GAE λ)
    4.  Training instability: value function collapse ruins the advantage signal
    5.  Memory-intensive: 4 models × model_size

**DPO (Module 29)** eliminates the reward model and PPO entirely by:
    •   Showing that the RLHF objective has a closed-form solution in terms
        of the reference policy (bypassing explicit RM)
    •   Reducing RLHF to a supervised learning problem on comparison data
    •   Requiring only 2 models (policy + reference) instead of 4

However, PPO retains advantages over DPO:
    •   More flexible reward specification (any scalar signal, not just preferences)
    •   Can incorporate online feedback (generate → rate → update in real time)
    •   Better performance on tasks where reward is verifiable (code, math)
    •   More expressive: can incorporate non-pairwise reward signals
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
RLHF Components and Their Roles

| Component       | Architecture             | Training data          | Frozen during PPO? | Memory         |
|-----------------|--------------------------|------------------------|--------------------|----------------|
| SFT model       | Full LM                  | (prompt, response)     | Becomes policy     | Full model     |
| Reference model | Full LM (= SFT model)    | None (frozen copy)     | Yes                | Full model     |
| Reward model    | LM + scalar head         | (prompt, chosen, rej.) | Yes                | Full model     |
| Value model     | LM + scalar head         | Bootstrapped from RM   | No (updated)       | Full model     |
| Policy          | Full LM (= SFT start)    | PPO rollouts + rewards | No (updated)       | Full model     |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Reward Model Training — Bradley-Terry Loss": {
        "description": "Train a scalar reward model using the Bradley-Terry preference model — implement the comparison loss, train on synthetic preference data, and visualise the reward distribution of chosen vs rejected responses.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
REWARD MODEL TRAINING — BRADLEY-TERRY LOSS
================================================================================

Implements reward model training:
    1. Bradley-Terry preference loss: -log σ(r_w - r_l)
    2. Reward model architecture: LM base + scalar head
    3. Training on (prompt, chosen, rejected) triples
    4. Evaluation: accuracy (was chosen rated higher?), margin distribution
    5. Reward distribution analysis: chosen vs rejected reward histograms

The Bradley-Terry model is the theoretical foundation for reward learning
from comparisons. It assumes the probability of preferring A over B is
σ(r(A) - r(B)), where σ is the sigmoid function.

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ── Model architecture ────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Reward model: language model base + scalar head.

    Architecture follows InstructGPT and Anthropic's Constitutional AI:
        - Start from the SFT model (or the pre-trained base)
        - Replace the language modelling head with a scalar regression head
        - The scalar head reads from the hidden state at the LAST TOKEN position
          (the EOS/end-of-response token, which has attended to the full sequence)

    In real implementations:
        - The LM backbone is often frozen initially and unfrozen gradually
        - Gradient checkpointing is used (same model as SFT, same memory needs)
        - Typically, the RM has the same parameter count as the policy (LLaMA-7B RM)
    """

    def __init__(self, vocab: int = 512, d: int = 128,
                 n_layers: int = 3, max_len: int = 64):
        super().__init__()
        self.embed    = nn.Embedding(vocab, d, padding_idx=0)
        self.pos      = nn.Embedding(max_len, d)
        dec_layer     = nn.TransformerDecoderLayer(
            d_model=d, nhead=4, dim_feedforward=d * 4,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.body     = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.norm     = nn.LayerNorm(d)

        # Scalar head: maps d-dim hidden state to a single reward scalar
        # This replaces the language modelling head
        self.reward_head = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)

        self.max_len  = max_len
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def get_reward(self, input_ids: torch.Tensor,
                   response_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute scalar reward for each sequence in the batch.

        input_ids:        (B, T) token IDs for prompt + response
        response_lengths: (B,) length of each response (to find EOS position)
                         If None, use the last token in each sequence.

        Returns: (B,) scalar reward scores
        """
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.embed(input_ids) + self.pos(pos)
        m    = self.causal_mask[:T, :T]
        h    = self.norm(self.body(x, x, tgt_mask=m, memory_mask=m,
                                    tgt_is_causal=True, memory_is_causal=True))

        if response_lengths is not None:
            # Use the hidden state at the last response token for each sequence
            idx      = (response_lengths - 1).clamp(0, T - 1)
            h_last   = h[torch.arange(B), idx, :]  # (B, d)
        else:
            # Use the hidden state at the final token
            h_last   = h[:, -1, :]  # (B, d)

        return self.reward_head(h_last).squeeze(-1)   # (B,)


# ── Bradley-Terry loss ────────────────────────────────────────────────────────

def bradley_terry_loss(r_chosen: torch.Tensor,
                        r_rejected: torch.Tensor) -> torch.Tensor:
    """
    Compute the Bradley-Terry pairwise preference loss.

    Loss = -E [ log σ(r_w - r_l) ]
         = -E [ log (exp(r_w) / (exp(r_w) + exp(r_l))) ]
         = E [ log(1 + exp(r_l - r_w)) ]
         = E [ softplus(r_l - r_w) ]

    Args:
        r_chosen:   (B,) reward scores for the preferred response
        r_rejected: (B,) reward scores for the less-preferred response

    The loss is minimised when r_chosen >> r_rejected for all pairs.
    """
    loss = -F.logsigmoid(r_chosen - r_rejected)
    return loss.mean()


def preference_accuracy(r_chosen: torch.Tensor,
                          r_rejected: torch.Tensor) -> float:
    """Fraction of pairs where the reward model correctly prefers chosen."""
    return (r_chosen > r_rejected).float().mean().item()


# ── Synthetic preference data ─────────────────────────────────────────────────

def make_preference_batch(
    batch_size: int, prompt_len: int, response_len: int,
    vocab: int, seed: int = 0
) -> tuple:
    """
    Synthesise a batch of (prompt, chosen, rejected) triples.

    The "quality signal" is simulated by making chosen responses
    contain more tokens from the 10–200 range (simulating informative content)
    and rejected responses contain more tokens from the 201–400 range
    (simulating low-quality content).

    In real RLHF: prompts are real user queries; responses are sampled
    from the SFT model; annotators provide the preference labels.
    """
    torch.manual_seed(seed)
    prompts   = torch.randint(3, 50, (batch_size, prompt_len))

    # "Chosen" responses: tokens more likely to be from a "quality" range
    chosen_r  = torch.cat([
        torch.randint(10, 100, (batch_size, response_len // 2)),
        torch.randint(3,  50,  (batch_size, response_len // 2)),
    ], dim=1)

    # "Rejected" responses: tokens more likely to be from a "low-quality" range
    rejected_r = torch.cat([
        torch.randint(200, 300, (batch_size, response_len // 2)),
        torch.randint(300, 400, (batch_size, response_len // 2)),
    ], dim=1)

    # Concatenate prompt + response for each
    chosen   = torch.cat([prompts, chosen_r],   dim=1)
    rejected = torch.cat([prompts, rejected_r], dim=1)

    return chosen, rejected


# ── Training loop ─────────────────────────────────────────────────────────────

def train_reward_model(
    model: RewardModel,
    n_steps: int = 150,
    batch_size: int = 16,
    lr: float = 1e-4,
    vocab: int = 512,
    prompt_len: int = 10,
    response_len: int = 20,
) -> dict:
    """Train the reward model on synthetic preference data."""
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses  = []
    accs    = []
    margins = []   # r_chosen - r_rejected

    for step in range(n_steps):
        chosen, rejected = make_preference_batch(
            batch_size, prompt_len, response_len, vocab, seed=step
        )

        r_w = model.get_reward(chosen)
        r_l = model.get_reward(rejected)

        loss = bradley_terry_loss(r_w, r_l)
        acc  = preference_accuracy(r_w, r_l)
        margin = (r_w - r_l).mean().item()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        accs.append(acc)
        margins.append(margin)

    return {"losses": losses, "accs": accs, "margins": margins}


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    torch.manual_seed(42)
    VOCAB, D = 512, 64
    N_LAYERS = 2
    N_STEPS  = 150

    model  = RewardModel(VOCAB, D, N_LAYERS, max_len=64)
    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 65)
    print("  REWARD MODEL TRAINING — BRADLEY-TERRY LOSS")
    print(f"  Model: {n_params:,} parameters")
    print("=" * 65)
    print()

    # Show initial predictions (before training)
    chosen_demo, rejected_demo = make_preference_batch(32, 10, 20, VOCAB, seed=999)
    with torch.no_grad():
        r_w_init = model.get_reward(chosen_demo)
        r_l_init = model.get_reward(rejected_demo)
    acc_init = preference_accuracy(r_w_init, r_l_init)
    margin_init = (r_w_init - r_l_init).mean().item()

    print(f"  Before training:")
    print(f"    Accuracy (chosen rated > rejected): {acc_init:.1%}")
    print(f"    Mean margin (r_w - r_l):            {margin_init:.3f}")
    print()

    # Train
    print(f"  {'Step':>6}  {'Loss':>10}  {'Accuracy':>10}  {'Mean margin':>14}")
    print(f"  {'':─>6}  {'':─>10}  {'':─>10}  {'':─>14}")

    history = train_reward_model(model, N_STEPS, batch_size=16,
                                   lr=1e-4, vocab=VOCAB,
                                   prompt_len=10, response_len=20)

    for step in range(0, N_STEPS, 15):
        print(f"  {step+1:>6}  {history['losses'][step]:>10.4f}  "
              f"{history['accs'][step]:>10.1%}  "
              f"{history['margins'][step]:>14.3f}")

    # After training evaluation
    with torch.no_grad():
        r_w_final   = model.get_reward(chosen_demo)
        r_l_final   = model.get_reward(rejected_demo)
    acc_final    = preference_accuracy(r_w_final, r_l_final)
    margin_final = (r_w_final - r_l_final).mean().item()

    print()
    print(f"  After training:")
    print(f"    Accuracy (chosen rated > rejected): {acc_final:.1%}  "
          f"(was {acc_init:.1%})")
    print(f"    Mean margin (r_w - r_l):            {margin_final:.3f}  "
          f"(was {margin_init:.3f})")
    print()

    # Reward distribution analysis
    print("=" * 65)
    print("  REWARD DISTRIBUTION: CHOSEN vs REJECTED")
    print("=" * 65)
    print()
    bars = " ▁▂▃▄▅▆▇█"
    for name, rewards in [("Chosen (preferred)",  r_w_final),
                            ("Rejected (worse)",    r_l_final)]:
        r_list   = rewards.tolist()
        r_min    = min(r_list)
        r_max    = max(r_list)
        r_mean   = sum(r_list) / len(r_list)
        n_bins   = 8
        span     = r_max - r_min + 1e-8
        hist     = [0] * n_bins
        for r in r_list:
            bin_idx = min(int((r - r_min) / span * n_bins), n_bins - 1)
            hist[bin_idx] += 1
        hist_str = "".join(bars[min(7, int(h / max(hist) * 7))] for h in hist)
        print(f"  {name:<26}: [{hist_str}]  mean={r_mean:.2f}  range=[{r_min:.2f},{r_max:.2f}]")

    print()
    print("  The reward model successfully assigns higher scores to chosen responses.")
    print("  The margin (r_chosen - r_rejected) reflects model confidence.")
    print()
    print("  Bradley-Terry interpretation:")
    print(f"  P(chosen ≻ rejected) = σ(margin) = σ({margin_final:.2f}) = "
          f"{torch.sigmoid(torch.tensor(margin_final)).item():.1%}")
''',
    },

    "PPO Training Loop for Language Models": {
        "description": "Implement the complete PPO training loop for RLHF: rollout phase (generate + score), advantage estimation with GAE, PPO-clip policy update, and KL penalty tracking.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
PPO TRAINING LOOP FOR RLHF
================================================================================

Implements the complete PPO loop for language model RLHF:
    1. Rollout: sample prompts → generate responses → compute rewards + KL
    2. Advantage estimation: Generalised Advantage Estimation (GAE)
    3. PPO-Clip policy update (multiple epochs over the rollout buffer)
    4. Value function update (MSE on returns)
    5. KL divergence tracking and reward hacking detection
    6. The characteristic KL vs reward curve

This is a simplified but complete implementation showing all key concepts.

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


# ── Policy and Value models ───────────────────────────────────────────────────

class PolicyModel(nn.Module):
    """Language model policy π_θ."""
    def __init__(self, vocab: int = 128, d: int = 64, n_layers: int = 2,
                 max_len: int = 32):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d),
                          nn.Linear(d, d*4, bias=False), nn.GELU(),
                          nn.Linear(d*4, d, bias=False))
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.d      = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(self.norm(h))

    def generate(self, prompt: torch.Tensor, max_new: int = 8,
                  temperature: float = 1.0) -> tuple:
        """
        Autoregressive generation from prompt.
        Returns (full_sequence, per_token_log_probs).
        """
        tokens = prompt.clone()
        log_probs = []
        for _ in range(max_new):
            logits  = self.forward(tokens)[:, -1, :]   # (B, V)
            if temperature != 1.0:
                logits = logits / temperature
            lp      = F.log_softmax(logits, dim=-1)
            probs   = torch.exp(lp)
            # Sample next token
            next_t  = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Record log probability of chosen token
            chosen_lp = lp.gather(1, next_t).squeeze(1)       # (B,)
            log_probs.append(chosen_lp)
            tokens  = torch.cat([tokens, next_t], dim=1)
        return tokens, torch.stack(log_probs, dim=1)   # (B, max_new)


class ValueModel(nn.Module):
    """
    Value function V_φ(s): estimates expected return at each position.
    Typically initialised from the reward model.
    """
    def __init__(self, vocab: int = 128, d: int = 64, n_layers: int = 2,
                 max_len: int = 32):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d),
                          nn.Linear(d, d*4, bias=False), nn.GELU(),
                          nn.Linear(d*4, d, bias=False))
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, 1, bias=False)   # scalar head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns value estimates for each position: (B, T)"""
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(self.norm(h)).squeeze(-1)   # (B, T)


# ── Reward model (simplified) ─────────────────────────────────────────────────

def simulate_reward(tokens: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """
    Simulated reward function.
    In production: a trained reward model (Module 28, Operation 1).
    Here: reward = fraction of response tokens in "high quality" token range.
    """
    responses = tokens[:, prompt_len:]   # (B, T_resp)
    B = responses.shape[0]
    rewards = []
    for b in range(B):
        resp = responses[b]
        # "High quality" = tokens in range 20–60 (simulated quality criterion)
        quality = ((resp >= 20) & (resp <= 60)).float().mean().item()
        # Add some random noise to simulate RM imperfection
        noise   = torch.randn(1).item() * 0.1
        rewards.append(quality + noise)
    return torch.tensor(rewards, dtype=torch.float32)


# ── GAE: Generalised Advantage Estimation ─────────────────────────────────────

def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                 gamma: float = 1.0, lam: float = 0.95) -> tuple:
    """
    Compute advantages using GAE (Schulman et al., 2015).

    GAE trades off bias and variance:
    λ=0: A_t = r_t + γV(s_{t+1}) - V(s_t)  (low variance, biased)
    λ=1: A_t = Σ_k γ^k r_{t+k} - V(s_t)   (unbiased, high variance)

    Args:
        rewards: (B, T) shaped reward at each step
        values:  (B, T) value estimates from V_φ
        gamma:   discount factor (usually 1.0 for episodic LM tasks)
        lam:     GAE lambda

    Returns:
        advantages: (B, T) GAE advantage estimates
        returns:    (B, T) empirical returns for value function target
    """
    B, T         = rewards.shape
    advantages   = torch.zeros_like(rewards)
    gae          = torch.zeros(B)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = torch.zeros(B)  # no future value at end
        else:
            next_val = values[:, t + 1]

        # TD residual
        delta       = rewards[:, t] + gamma * next_val - values[:, t]
        # GAE recursive formula
        gae         = delta + gamma * lam * gae
        advantages[:, t] = gae

    returns = advantages + values
    return advantages, returns


# ── PPO update ─────────────────────────────────────────────────────────────────

def ppo_loss(
    new_log_probs: torch.Tensor,   # (B, T) new policy log-probs
    old_log_probs: torch.Tensor,   # (B, T) old policy log-probs (from rollout)
    advantages:    torch.Tensor,   # (B, T) GAE advantages
    clip_eps:      float = 0.2,
) -> torch.Tensor:
    """
    PPO-Clip policy loss.

    The clipping prevents any single update from changing the policy too much:
        - If A > 0 (good action): boost probability, but not more than (1+ε) ratio
        - If A < 0 (bad action): reduce probability, but not more than (1-ε) ratio

    This is the core of PPO's stability guarantee.
    """
    # Probability ratio: new_prob / old_prob
    ratio       = torch.exp(new_log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1       = ratio * advantages
    surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

    # Take the minimum (pessimistic): only update when both versions agree
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss


# ── Full PPO training loop ────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    n_steps:       int   = 40         # number of PPO gradient steps
    rollout_size:  int   = 16         # prompts per rollout
    max_new:       int   = 8          # max tokens to generate per response
    prompt_len:    int   = 4          # length of input prompt
    kl_beta:       float = 0.05       # KL penalty coefficient
    clip_eps:      float = 0.2        # PPO clip ratio
    gamma:         float = 1.0        # discount factor
    gae_lam:       float = 0.95       # GAE lambda
    policy_lr:     float = 1e-4       # policy learning rate
    value_lr:      float = 5e-4       # value function learning rate
    entropy_coef:  float = 0.01       # entropy bonus coefficient
    value_coef:    float = 0.5        # value loss coefficient
    ppo_epochs:    int   = 2          # optimisation epochs per rollout
    temperature:   float = 0.9        # generation temperature


def run_ppo_training(
    cfg: PPOConfig,
    vocab: int = 128,
    d: int = 64,
    n_layers: int = 2,
) -> dict:
    """
    Full PPO RLHF training loop.
    Returns training history.
    """
    torch.manual_seed(42)

    # Initialise models
    policy     = PolicyModel(vocab, d, n_layers)
    ref_policy = copy.deepcopy(policy)      # frozen reference
    value_fn   = ValueModel(vocab, d, n_layers)

    for p in ref_policy.parameters():
        p.requires_grad_(False)

    opt_policy = torch.optim.Adam(policy.parameters(),  lr=cfg.policy_lr)
    opt_value  = torch.optim.Adam(value_fn.parameters(), lr=cfg.value_lr)

    history = {
        "rm_scores": [], "kl_divs": [], "policy_loss": [], "value_loss": [],
        "entropy": [], "advantages_mean": [], "clip_fracs": []
    }

    # Pre-generate prompts
    torch.manual_seed(10)
    prompt_pool = torch.randint(3, vocab, (100, cfg.prompt_len))

    for step in range(cfg.n_steps):
        # ── ROLLOUT PHASE ─────────────────────────────────────────────────────
        policy.eval()
        ref_policy.eval()

        # Sample prompts
        perm    = torch.randperm(len(prompt_pool))[:cfg.rollout_size]
        prompts = prompt_pool[perm]   # (B, P)

        with torch.no_grad():
            # Generate responses from current policy
            full_seq, old_log_probs = policy.generate(
                prompts, cfg.max_new, cfg.temperature
            )  # full_seq: (B, P+T), old_lp: (B, T)

            # Get reference model log-probs for KL computation
            ref_lp_list = []
            for t in range(cfg.max_new):
                prefix     = full_seq[:, :cfg.prompt_len + t]
                ref_logits = ref_policy(prefix)[:, -1, :]
                ref_lp     = F.log_softmax(ref_logits, dim=-1)
                # Token actually chosen
                chosen_tok = full_seq[:, cfg.prompt_len + t].unsqueeze(1)
                ref_lp_list.append(ref_lp.gather(1, chosen_tok).squeeze(1))
            ref_log_probs = torch.stack(ref_lp_list, dim=1)  # (B, T)

            # Compute KL per token: KL_t = log π_θ - log π_ref
            kl_per_token  = old_log_probs - ref_log_probs   # (B, T)

            # Compute RM reward (scalar per sequence, at final token)
            rm_rewards     = simulate_reward(full_seq, cfg.prompt_len)  # (B,)

            # Build token-level reward signal:
            # shaped_reward = RM_reward at terminal token - β × KL at every token
            shaped_rewards = -cfg.kl_beta * kl_per_token.clone()  # (B, T)
            shaped_rewards[:, -1] += rm_rewards   # add RM reward at final step

            # Value estimates for current sequences
            values = value_fn(full_seq)[:, cfg.prompt_len:]  # (B, T)

            # GAE advantage estimation
            advantages, returns = compute_gae(
                shaped_rewards, values, cfg.gamma, cfg.gae_lam
            )

            # Whiten advantages for stable training
            adv_mean  = advantages.mean()
            adv_std   = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # ── OPTIMISATION PHASE ────────────────────────────────────────────────
        policy.train()
        value_fn.train()

        for ppo_epoch in range(cfg.ppo_epochs):
            # Recompute current policy log-probs
            new_lp_list = []
            for t in range(cfg.max_new):
                prefix     = full_seq[:, :cfg.prompt_len + t]
                new_logits = policy(prefix)[:, -1, :]
                new_lp     = F.log_softmax(new_logits, dim=-1)
                chosen_tok = full_seq[:, cfg.prompt_len + t].unsqueeze(1)
                new_lp_list.append(new_lp.gather(1, chosen_tok).squeeze(1))
            new_log_probs = torch.stack(new_lp_list, dim=1)   # (B, T)

            # Policy loss
            p_loss = ppo_loss(new_log_probs, old_log_probs.detach(),
                               advantages.detach(), cfg.clip_eps)

            # Value loss
            new_values = value_fn(full_seq.detach())[:, cfg.prompt_len:]
            v_loss     = F.mse_loss(new_values, returns.detach())

            # Entropy bonus (encourage exploration)
            all_logits   = policy(full_seq[:, :-1])[:, cfg.prompt_len-1:, :]
            all_log_probs = F.log_softmax(all_logits, dim=-1)
            entropy      = -(all_log_probs * torch.exp(all_log_probs)).sum(-1).mean()

            # Combined loss
            total_loss = p_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

            opt_policy.zero_grad()
            opt_value.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(),  0.5)
            nn.utils.clip_grad_norm_(value_fn.parameters(), 0.5)
            opt_policy.step()
            opt_value.step()

        # Clip fraction (how often the ratio was clipped)
        with torch.no_grad():
            ratio     = torch.exp(new_log_probs - old_log_probs)
            clip_frac = ((ratio < 1 - cfg.clip_eps) |
                          (ratio > 1 + cfg.clip_eps)).float().mean().item()

        history["rm_scores"].append(rm_rewards.mean().item())
        history["kl_divs"].append(kl_per_token.abs().mean().item())
        history["policy_loss"].append(p_loss.item())
        history["value_loss"].append(v_loss.item())
        history["entropy"].append(entropy.item())
        history["advantages_mean"].append(adv_mean.item())
        history["clip_fracs"].append(clip_frac)

    return history


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = PPOConfig(n_steps=50, rollout_size=16, max_new=8,
                    kl_beta=0.05, clip_eps=0.2, ppo_epochs=2)

    print("=" * 65)
    print("  PPO RLHF TRAINING LOOP")
    print(f"  KL beta={cfg.kl_beta}, clip_eps={cfg.clip_eps}")
    print("=" * 65)
    print()

    history = run_ppo_training(cfg, vocab=128, d=64, n_layers=2)

    print(f"  {'Step':>5}  {'RM score':>10}  {'KL div':>10}  "
          f"{'Policy loss':>12}  {'Clip frac':>12}  {'Entropy':>10}")
    print(f"  {'':─>5}  {'':─>10}  {'':─>10}  "
          f"{'':─>12}  {'':─>12}  {'':─>10}")

    for step in range(0, cfg.n_steps, 5):
        print(f"  {step+1:>5}  "
              f"{history['rm_scores'][step]:>10.4f}  "
              f"{history['kl_divs'][step]:>10.4f}  "
              f"{history['policy_loss'][step]:>12.4f}  "
              f"{history['clip_fracs'][step]:>12.1%}  "
              f"{history['entropy'][step]:>10.4f}")

    # KL vs Reward curve analysis
    print()
    print("=" * 65)
    print("  KL vs REWARD CURVE (reward hacking diagnostic)")
    print("=" * 65)
    print()
    print("  This curve shows the trade-off between RM reward and KL divergence.")
    print("  Reward should increase while KL remains controlled (< 0.5 is healthy).")
    print()

    rm_scores = history["rm_scores"]
    kl_divs   = history["kl_divs"]

    print(f"  {'Step':>6}  {'RM Score':>10}  {'KL Div':>10}  Status")
    print(f"  {'':─>6}  {'':─>10}  {'':─>10}  {'':─}")
    for i in range(0, cfg.n_steps, 10):
        rm  = rm_scores[i]
        kl  = kl_divs[i]
        if kl > 0.5 and rm < rm_scores[max(0, i-5)]:
            status = "⚠️ reward hacking?"
        elif kl < 0.1:
            status = "ℹ️ low KL (may need lower β)"
        else:
            status = "✓ healthy"
        print(f"  {i+1:>6}  {rm:>10.4f}  {kl:>10.4f}  {status}")

    print()
    print("  To diagnose reward hacking: human eval should validate RM score gains.")
    print("  If RM score rises but human ratings plateau → reward hacking is occurring.")
    print("  Fix: increase KL penalty β, or use more diverse preference data.")
''',
    },

    "KL Divergence Penalty: Reward Hacking Analysis": {
        "description": "Demonstrate reward hacking at different KL penalty strengths — show the over-optimisation curve, compare quality vs reward score, and find the optimal KL coefficient.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
KL PENALTY AND REWARD HACKING ANALYSIS
================================================================================

Demonstrates the central trade-off in RLHF:
    - Without KL penalty: reward maximisation leads to reward hacking
    - With small β:  fast improvement but eventual quality degradation
    - With large β:  stable but slow alignment
    - Optimal β:     reward and quality both improve together

Also shows:
    - The KL-reward Pareto curve
    - Per-token KL distribution during training
    - The "sycophancy" failure mode: a language model that optimises reward
      by generating verbose, repetitive, or sycophantic text

================================================================================
"""

import math
import copy
import torch
import torch.nn.functional as F


# ── Simulated RLHF environment ────────────────────────────────────────────────

class SimulatedRLHFEnv:
    """
    A simplified RLHF environment where:
    - The "policy" chooses tokens from a distribution
    - The "reward model" is biased: it over-rewards certain patterns
      (like high token diversity — simulating a bias toward verbose responses)
    - True quality is measured differently (penalises excessive verbosity)
    - KL penalty limits how far the policy can drift from the reference

    This directly demonstrates reward hacking: optimising the RM without
    KL constraint produces high RM score but low true quality.
    """

    def __init__(self, vocab: int = 64, prompt_len: int = 4,
                 response_len: int = 10):
        self.vocab        = vocab
        self.prompt_len   = prompt_len
        self.response_len = response_len
        # Reference distribution: uniform over [3, vocab/2)
        self.ref_probs    = torch.zeros(vocab)
        self.ref_probs[3:vocab//2] = 1.0 / (vocab//2 - 3)

    def rm_reward(self, response_tokens: torch.Tensor) -> float:
        """
        Biased reward model: rewards high token diversity.
        This is a simulated reward model bias.
        Reward hacking: the policy can get high RM reward by generating
        diverse tokens (sampling widely) even if the content is nonsensical.
        """
        unique_frac = len(set(response_tokens.tolist())) / len(response_tokens)
        high_tokens = (response_tokens >= self.vocab // 2).float().mean().item()
        # RM is biased: rewards diversity + high-range tokens
        return 0.5 * unique_frac + 0.5 * high_tokens

    def true_quality(self, response_tokens: torch.Tensor) -> float:
        """
        True quality (what humans actually want): prefer tokens in [3, vocab/4).
        This diverges from the RM for tokens in [vocab/4, vocab).
        """
        in_quality_range = ((response_tokens >= 3) &
                             (response_tokens < self.vocab // 4)).float().mean().item()
        return in_quality_range

    def kl_divergence(self, probs: torch.Tensor) -> float:
        """KL divergence from reference distribution (simplified)."""
        safe_probs = probs.clamp(min=1e-8)
        safe_ref   = self.ref_probs.clamp(min=1e-8)
        kl = (safe_probs * (safe_probs / safe_ref).log()).sum().item()
        return max(0.0, kl)


def optimise_policy(env: SimulatedRLHFEnv, kl_beta: float,
                     n_steps: int = 200) -> dict:
    """
    Optimise a policy (token distribution) to maximise RM reward - β*KL.
    Returns history of {rm_score, true_quality, kl_div}.

    The policy is parameterised as softmax(logits).
    """
    vocab     = env.vocab
    # Initialise logits = reference distribution
    logits    = torch.log(env.ref_probs.clone().clamp(min=1e-8))
    logits    = logits.requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=0.02)

    history   = {"rm_score": [], "true_quality": [], "kl_div": [], "step": []}

    B = 64   # batch size for gradient estimation
    T = env.response_len

    for step in range(n_steps):
        optimizer.zero_grad()
        probs = F.softmax(logits, dim=0)

        # Sample responses from current policy
        samples = torch.multinomial(probs.expand(B * T, -1), 1).reshape(B, T)

        # RM reward (differentiable approximation via REINFORCE-style)
        rm_rewards     = torch.tensor([env.rm_reward(samples[b])
                                         for b in range(B)], dtype=torch.float32)

        # KL divergence penalty
        kl = env.kl_divergence(probs)

        # REINFORCE: ∇E[r] ≈ E[r × ∇ log π]  (policy gradient)
        log_probs_sum = torch.stack([
            torch.log(probs[samples[b]].clamp(min=1e-8)).sum()
            for b in range(B)
        ])
        reward_baseline = rm_rewards.mean()

        # Policy gradient loss: maximise E[r] - β×KL
        pg_loss  = -(rm_rewards - reward_baseline) * log_probs_sum
        kl_penalty = kl_beta * kl * T * B   # scale KL to match reward scale

        loss     = pg_loss.mean() + kl_penalty
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            with torch.no_grad():
                test_samples = torch.multinomial(probs.expand(128*T, -1), 1).reshape(128, T)
                rm_score     = torch.tensor([env.rm_reward(test_samples[b])
                                              for b in range(128)]).mean().item()
                quality      = torch.tensor([env.true_quality(test_samples[b])
                                              for b in range(128)]).mean().item()
                kl_now       = env.kl_divergence(probs)
            history["rm_score"].append(rm_score)
            history["true_quality"].append(quality)
            history["kl_div"].append(kl_now)
            history["step"].append(step)

    return history


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  KL PENALTY AND REWARD HACKING ANALYSIS")
    print("  Varying β (KL penalty coefficient)")
    print("=" * 70)
    print()
    print("  Scenario:")
    print("  - RM rewards: diversity + high-range tokens (biased RM)")
    print("  - True quality: tokens in low range [3, vocab/4)")
    print("  - Without KL: policy exploits RM by generating diverse/high tokens")
    print("  - With KL: policy cannot deviate too far from reference")
    print()

    env = SimulatedRLHFEnv(vocab=64, response_len=10)
    betas = [0.0, 0.01, 0.05, 0.2, 1.0]

    all_histories = {}
    for beta in betas:
        h = optimise_policy(env, kl_beta=beta, n_steps=200)
        all_histories[beta] = h

    print(f"  {'β':>8}  {'RM Score (final)':>18}  {'True Quality (final)':>22}  "
          f"{'KL Div (final)':>16}  {'Hacking?':>12}")
    print(f"  {'':─>8}  {'':─>18}  {'':─>22}  {'':─>16}  {'':─>12}")

    for beta in betas:
        h     = all_histories[beta]
        rm_f  = h["rm_score"][-1]
        q_f   = h["true_quality"][-1]
        kl_f  = h["kl_div"][-1]
        hacking = (rm_f > q_f + 0.2)   # RM score much higher than true quality
        flag  = "⚠️ YES" if hacking else "✓ No"
        print(f"  {beta:>8.2f}  {rm_f:>18.4f}  {q_f:>22.4f}  {kl_f:>16.4f}  {flag:>12}")

    print()
    print("  β=0.0:   RM score is very high, true quality is low → reward hacking")
    print("  β=0.05:  Balanced: RM score improves, true quality also improves")
    print("  β=1.0:   Very conservative: minimal improvement in either metric")

    # Show KL-Reward frontier
    print()
    print("=" * 70)
    print("  KL vs REWARD TRAJECTORY (reward hacking 'knee')")
    print("=" * 70)
    print()
    print("  The characteristic RLHF over-optimisation curve:")
    print("  RM score peaks and then quality declines while KL keeps growing.")
    print()
    print(f"  {'Step':>6}", end="")
    for beta in [0.0, 0.05, 0.2]:
        print(f"  {'β='+str(beta)+' RM':>12}  {'TrueQ':>8}", end="")
    print()
    print(f"  {'':─>6}" + "".join(f"  {'':─>12}  {'':─>8}" for _ in [0,0,0]))

    for i in range(len(all_histories[0.0]["step"])):
        print(f"  {all_histories[0.0]['step'][i]:>6}", end="")
        for beta in [0.0, 0.05, 0.2]:
            h  = all_histories[beta]
            rm = h["rm_score"][i]
            q  = h["true_quality"][i]
            print(f"  {rm:>12.4f}  {q:>8.4f}", end="")
        print()

    print()
    print("  Key insight: β=0 RM score rises fast but true quality declines.")
    print("  β=0.05 is the 'sweet spot': both RM score AND true quality improve.")
    print("  This demonstrates why the KL penalty is not just regularisation —")
    print("  it is what makes RLHF actually improve human-judged quality.")
    print()

    # Show optimal policy characteristics
    print("=" * 70)
    print("  POLICY TOKEN DISTRIBUTION: NO KL vs OPTIMAL β")
    print("=" * 70)
    print()
    print("  How the policy's token preferences changed after training:")
    print()

    for beta in [0.0, 0.05]:
        # Reconstruct final policy
        ref_logits = torch.log(env.ref_probs.clamp(min=1e-8))
        l          = ref_logits.clone().requires_grad_(True)
        opt        = torch.optim.Adam([l], lr=0.02)
        T, B_      = env.response_len, 32
        for _ in range(200):
            opt.zero_grad()
            probs = F.softmax(l, dim=0)
            samps = torch.multinomial(probs.expand(B_*T, -1), 1).reshape(B_, T)
            rms   = torch.tensor([env.rm_reward(samps[b]) for b in range(B_)])
            lp    = torch.stack([torch.log(probs[samps[b]].clamp(1e-8)).sum() for b in range(B_)])
            loss  = -(rms - rms.mean()) * lp
            kl_   = beta * env.kl_divergence(probs) * T * B_
            (loss.mean() + kl_).backward()
            opt.step()
        final_probs = F.softmax(l.detach(), dim=0)

        low_range  = final_probs[3:16].sum().item()   # quality tokens
        mid_range  = final_probs[16:32].sum().item()  # neutral
        high_range = final_probs[32:].sum().item()    # "exploitable" tokens

        print(f"  β={beta}: quality [3,16)={low_range:.2%}  "
              f"neutral [16,32)={mid_range:.2%}  high [32,64)={high_range:.2%}")

    print()
    print("  β=0.0 policy heavily upweights high-range tokens (reward hacking).")
    print("  β=0.05 policy stays closer to reference (quality tokens preserved).")
''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


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
    visual_height = 600
    # try:
    #     from llm_training.visuals.rlhf_ppo import (
    #         RLHF_VISUAL_HTML,
    #         RLHF_VISUAL_HEIGHT,
    #     )
    #     visual_html   = RLHF_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = RLHF_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[28_rlhf_reward_modeling_ppo.py] Could not load visual: {e}",
    #         stacklevel=2,
    #     )

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    COMPLEXITY,
        "operations":    OPERATIONS,
    }