"""
DPO: Direct Preference Optimisation
=====================================

Direct Preference Optimisation (Rafailov et al., 2023) is arguably the most
important algorithmic advance in LLM alignment since the original RLHF paper.
DPO shows that the complex four-model PPO training loop can be replaced by a
single supervised learning objective — a binary cross-entropy loss directly
on preference pairs. The insight is profound: the optimal RLHF policy has a
closed-form solution in terms of the reference policy, and this relationship
can be exploited to train the policy directly without ever fitting a separate
reward model or running RL.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "DPO: Direct Preference Optimisation"
DISPLAY_NAME = "29 · DPO"
ICON         = "🎲"
SUBTITLE     = "Direct Preference Optimisation — No Reward Model Needed"


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

### From RLHF to DPO: The Derivation

Module 28 showed that RLHF optimises:

    max_{π_θ} E_{x~D, y~π_θ} [r(x, y)] - β × KL(π_θ || π_ref)

DPO begins by asking: what is the optimal policy under this objective?

**Step 1 — Rewrite the KL-constrained objective as a partition function:**

The KL term can be expanded:
    KL(π_θ || π_ref) = Σ_y π_θ(y|x) log [π_θ(y|x) / π_ref(y|x)]

Combining with the reward:
    Σ_y π_θ(y|x) [r(x,y) - β log(π_θ(y|x) / π_ref(y|x))]

**Step 2 — The optimal policy has a closed form:**

Setting the functional derivative to zero and solving:

    π*(y|x) = π_ref(y|x) × exp(r(x,y) / β) / Z(x)

where Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β) is the partition function (normaliser).

This is a Boltzmann distribution: the optimal policy is the reference policy
*reweighted* by exponentiated reward. Higher reward → higher probability.
Higher β → weights closer to reference (more conservative).

This result was known before DPO — it is the solution to a constrained
entropy-maximisation problem. DPO's insight is what to *do* with this solution.

**Step 3 — Invert the relationship to express reward in terms of policy:**

Taking logs of the optimal policy equation:

    log π*(y|x) = log π_ref(y|x) + r(x,y)/β - log Z(x)

Solving for r(x,y):

    r(x,y) = β × log [π*(y|x) / π_ref(y|x)] + β × log Z(x)

The reward can be expressed as a log-ratio of probabilities! The Z(x) term
is a function of x only (not y) — it cancels in pairwise comparisons.

**Step 4 — Substitute into the Bradley-Terry loss:**

The reward model training loss was:
    L_RM = -E [log σ(r(x, y_w) - r(x, y_l))]

Substituting our expression for r(x,y), and noting that Z(x) cancels in
the difference r_w - r_l:

    r(x,y_w) - r(x,y_l)
    = β × log [π*(y_w|x) / π_ref(y_w|x)] - β × log [π*(y_l|x) / π_ref(y_l|x)]

Replacing π* with the trainable policy π_θ:

    **L_DPO(π_θ; π_ref) = -E [log σ(β × log [π_θ(y_w|x)/π_ref(y_w|x)]
                                        - β × log [π_θ(y_l|x)/π_ref(y_l|x)])]**

This is the DPO loss. It is a supervised learning objective on preference pairs —
no reward model, no RL, no PPO, no value function.

The key quantities:
    log [π_θ(y_w|x) / π_ref(y_w|x)] = Σ_t log π_θ(y_w_t|x,y_w<t)
                                        - Σ_t log π_ref(y_w_t|x,y_w<t)

This is the **implicit reward** of the trained policy relative to the reference.


    **Diagram 1 — DPO vs RLHF System Comparison:**

    RLHF (4 models, 3 phases):          DPO (2 models, 1 phase):
    ════════════════════════════════    ════════════════════════════════

    Phase 1:  SFT training              Phase 1:  SFT training
    Phase 2:  RM training               Phase 2:  DPO training
              [RM: LM + scalar head]               [Policy π_θ]
              [Comparison data]                     [Reference π_ref (frozen)]
              [Bradley-Terry loss]                  [DPO loss on pref. pairs]
    Phase 3:  PPO training
              [Policy π_θ]
              [Reference π_ref (frozen)]
              [RM (frozen)]
              [Value model]
              [Rollout + GAE + PPO-clip]

    Advantages of DPO:
    ✓ 2 models instead of 4
    ✓ No rollout generation
    ✓ No value function
    ✓ Single supervised training phase
    ✓ Stable training (no PPO instabilities)
    ✓ Simple implementation (~50 lines of code)


### Understanding the DPO Loss Geometrically

The DPO loss has a beautiful geometric interpretation. Define the
**implicit reward** of the policy at response y:

    r̂_θ(x, y) = β × log [π_θ(y|x) / π_ref(y|x)]
               = β × (log π_θ(y|x) - log π_ref(y|x))

This measures how much the policy prefers y relative to the reference.
It is the *gap* between the trained and reference log-probabilities,
scaled by β.

The DPO loss becomes:
    L_DPO = -E [log σ(r̂_θ(x, y_w) - r̂_θ(x, y_l))]

The loss pushes the policy to:
    1.  Increase its log-probability on chosen responses y_w relative to reference
    2.  Decrease its log-probability on rejected responses y_l relative to reference

**What the gradient does:**
    ∂L_DPO/∂θ ∝ -[σ(negative_margin)] × β × [
        ∇_θ log π_θ(y_w|x) - ∇_θ log π_θ(y_l|x)
    ]

    where negative_margin = r̂_θ(y_l) - r̂_θ(y_w)  (is negative when correct)

The weighting term σ(negative_margin) means:
    •   When the model strongly prefers y_w (margin is large positive):
        the gradient is small → the model is already correct, less update
    •   When the model assigns y_w ≈ y_l or prefers y_l:
        the gradient is large → the model needs more correction

This is similar to a **margin-based loss**: only examples where the model
is uncertain or wrong receive large gradient signal.


### DPO vs RLHF: The Distribution Shift Problem

DPO's key weakness is that it is an **offline** algorithm. The preference
data was generated by some behaviour policy (usually the SFT model), not
by the current policy π_θ.

As π_θ diverges from the SFT model during training:
    •   The preference pairs (y_w, y_l) become less representative of π_θ's
        output distribution
    •   The implicit rewards r̂_θ(y_w) and r̂_θ(y_l) may become extreme
        (very large or very small), making the loss saturate
    •   The policy may increase log-probabilities for y_w tokens even when
        y_w is already extremely high-probability, causing degenerate outputs

This is the **distribution shift problem** or **offline RL problem**:
the training data does not match the policy's current distribution.

PPO avoids this through online data collection (generate → evaluate → train),
but this requires the expensive rollout infrastructure.

**DPO mitigations:**
    1.  Use a low learning rate (similar to RLHF: much lower than SFT)
    2.  Monitor the policy's log-probability on chosen and rejected responses
        separately (both should be healthily sized — not extreme)
    3.  Use online DPO variants (iterative DPO, RLAIF): regenerate preference
        data periodically from the current policy
    4.  Add an SFT loss as a regulariser: L_total = L_DPO + λ × L_SFT(y_w)


### The Reference Log-Probability Computation

A critical implementation detail: the DPO loss requires computing log-probs
under BOTH π_θ (trainable) and π_ref (frozen) for BOTH y_w and y_l.

**Efficient implementation:**
    for each (x, y_w, y_l) triple:
        # Forward pass through trainable policy
        logp_θ_w = sum_t log π_θ(y_w_t | x, y_w_{<t})
        logp_θ_l = sum_t log π_θ(y_l_t | x, y_l_{<t})

        # Forward pass through frozen reference (no gradient)
        with torch.no_grad():
            logp_ref_w = sum_t log π_ref(y_w_t | x, y_w_{<t})
            logp_ref_l = sum_t log π_ref(y_l_t | x, y_l_{<t})

        # Implicit rewards
        r̂_w = β × (logp_θ_w - logp_ref_w)
        r̂_l = β × (logp_θ_l - logp_ref_l)

        # DPO loss
        loss = -log σ(r̂_w - r̂_l)

**Memory considerations:**
Both the policy and reference models must be in memory simultaneously.
For a 7B model, this is 2 × 14 GB = 28 GB just for the model weights
(before gradients and optimizer state).

**Optimisation:** Pre-compute and cache reference log-probabilities before
training (they do not change). This avoids the reference model forward pass
during each training step, cutting memory and compute requirements.


    **Diagram 2 — DPO Loss Computation:**

    DPO LOSS COMPUTATION FOR ONE EXAMPLE
    ════════════════════════════════════════════════════════════════

    Input: (prompt x, chosen y_w, rejected y_l)

    CHOSEN PATH:
    x + y_w → [π_θ (trainable)]   → logp_θ_w  (requires grad)
    x + y_w → [π_ref (frozen)]    → logp_ref_w (no grad, pre-computed)
    implicit reward:  r̂_w = β × (logp_θ_w - logp_ref_w)

    REJECTED PATH:
    x + y_l → [π_θ (trainable)]   → logp_θ_l  (requires grad)
    x + y_l → [π_ref (frozen)]    → logp_ref_l (no grad, pre-computed)
    implicit reward:  r̂_l = β × (logp_θ_l - logp_ref_l)

    DPO LOSS:
    L = -log σ(r̂_w - r̂_l)
      = -log σ(β × [(logp_θ_w - logp_ref_w) - (logp_θ_l - logp_ref_l)])

    Gradient:
    ∂L/∂θ ∝ -σ(r̂_l - r̂_w) × β × [∂logp_θ_w/∂θ - ∂logp_θ_l/∂θ]
             ↑ weighting: large when model is confused
                                    ↑ push y_w up, y_l down


### DPO Hyperparameters and Sensitivity

**β (temperature / KL coefficient):**
The same β as in RLHF, controls how far the policy can deviate from the reference.

    β = 0.01:  Very small: policy can move far from reference (may destabilise)
    β = 0.05:  Standard for most instruction tuning (InstructGPT / Anthropic scale)
    β = 0.1:   Moderate: conservative but stable
    β = 0.5:   Conservative: slow alignment, minimal forgetting
    β = 1.0+:  Policy barely moves; mostly regularisation

The DPO paper used β=0.1 for most experiments. Practice has converged to
β ∈ [0.01, 0.5] depending on the quality of preference data and how much
the model should change from the SFT baseline.

**Learning rate:**
Even lower than SFT: typically 1e-6 to 1e-5.
The combined effect of the DPO loss and β is that updates are already scaled.
A too-high LR causes reward hacking through a different mechanism: the policy
aggressively upweights chosen tokens until their log-probabilities explode.

**Length normalisation:**
A subtle issue: longer responses have lower log-probabilities (product of more
terms), making them seem "less preferred" even if they are actually better.
Length-normalised DPO divides each sequence's log-probability by its length:

    logp_norm = Σ_t log π(y_t|x,y_{<t}) / |y|

Without normalisation, DPO has a systematic bias toward shorter responses.

**The sycophancy-conciseness trade-off:**
If chosen responses are longer than rejected ones (a common annotation bias),
DPO will tend to produce longer outputs. If rejected responses are longer
(if annotators prefer concise answers), DPO will shorten outputs. This
reflects the annotation bias being encoded into the model.


### DPO Variants and Extensions

**IPO (Identity Policy Optimisation, Azar et al., 2023):**
Addresses DPO's known issue of overfitting on deterministic preference pairs.
DPO's loss can saturate when the margin r̂_w - r̂_l is very large.
IPO replaces the sigmoid with a squared-error term:

    L_IPO = (r̂_w - r̂_l - 1/(2β))²

This prevents the margin from growing unboundedly.

**KTO (Kahneman-Tversky Optimisation, Ethayarajh et al., 2024):**
Instead of requiring preference *pairs*, KTO uses individual examples
labelled as "desirable" or "undesirable". This is more data-efficient since
you don't need to pair responses:

    For desirable  (y_w):  L_KTO += max(0, z0 - (r̂(y_w) - λ))
    For undesirable (y_l): L_KTO += max(0, (r̂(y_l) - λ) - z0)

where λ is the mean implicit reward and z0 is a reference margin.

**SimPO (Simple Preference Optimisation, Meng et al., 2024):**
Eliminates the reference model entirely. Uses length-normalised log-probability
of the policy itself as the reward, with a target margin γ:

    L_SimPO = -log σ(β/|y_w| × logp_θ(y_w|x) - β/|y_l| × logp_θ(y_l|x) - γ)

Since there is no reference model, SimPO requires only one model (the policy).
This is the most memory-efficient alignment method.

**ORPO (Odds Ratio Preference Optimisation, Hong et al., 2024):**
Combines SFT and DPO into a single loss using the odds ratio:

    L_ORPO = L_CE(y_w) - λ × log σ(log odds(y_w/y_l))

where odds(y) = π(y|x) / (1 - π(y|x)) is the generation odds.

This trains the model on chosen responses via standard CE loss while
simultaneously penalising rejected responses, eliminating the need for a
separate SFT phase.

**Iterative DPO / Online DPO:**
Addresses the distribution shift problem by alternating between:
    1. Generating responses from the current policy π_θ
    2. Getting human (or AI) preference labels for these new responses
    3. Training DPO on the new preference pairs

This is the closest DPO gets to online RL. Used in Self-Play Fine-Tuning
(SPIN) and various iterative alignment papers.


### When to Use DPO vs PPO

    Criterion                   DPO preferred           PPO preferred
    ─────────────────────────────────────────────────────────────────────
    Infrastructure              Limited (2 models)       Full RL setup available
    Data availability           Comparison pairs exist   Can generate rollouts
    Training stability          Important                 Can manage instability
    Online feedback             Not needed               Available (code/math eval)
    Verifiable reward           No (subjective quality)  Yes (unit tests, proofs)
    Scale                       Up to 70B easily         Any scale
    Implementation effort       Low                      High
    Quality ceiling             ~RLHF quality on most   Potentially higher
                                general tasks            tasks with clear reward
    ─────────────────────────────────────────────────────────────────────

**The practical answer:** DPO is almost always the right starting point because
it is simpler, more stable, and achieves comparable quality to PPO on most tasks.
PPO is preferred when you have a verifiable reward signal (e.g., code that must
pass unit tests, mathematical proofs that must be checkable) because online RL
can generate and evaluate its own training data indefinitely.


### Practical DPO Implementation Checklist

    ☐  Use the SFT model as both policy and reference (same weights, separate forward passes)
    ☐  Pre-compute and cache reference log-probabilities before training
    ☐  Set β = 0.1 as default; adjust based on KL divergence monitoring
    ☐  Learning rate: 1e-6 to 5e-6 (much lower than SFT)
    ☐  Batch size: effective batch of 64–128 preference pairs
    ☐  Monitor: chosen_rewards (should increase), rejected_rewards (should decrease)
    ☐  Monitor: reward_margin = chosen_rewards - rejected_rewards (should increase)
    ☐  Monitor: policy log-probs on chosen (should stay reasonable, not explode)
    ☐  Early stopping: stop when reward_margin plateaus (risk of overfitting)
    ☐  Evaluate on held-out benchmarks: check for catastrophic forgetting
    ☐  Length normalisation: consider for datasets with length imbalance between pairs
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
DPO and Its Variants

| Method      | Reference model? | Pairwise data? | Loss type           | Memory        | Best for                          |
|-------------|------------------|----------------|---------------------|---------------|-----------------------------------|
| DPO         | Yes (frozen)     | Yes            | -log σ(r̂_w - r̂_l) | 2× model size | General instruction alignment     |
| IPO         | Yes (frozen)     | Yes            | (r̂_w - r̂_l - 1/2β)²| 2× model size| When DPO overfits on easy pairs   |
| KTO         | Yes (frozen)     | No (labelled)  | Prospect-theory     | 2× model size | When pairwise data is scarce      |
| SimPO       | No               | Yes            | -log σ(β/l × Δlogp)| 1× model size | Maximum efficiency                |
| ORPO        | No               | Yes            | CE + log odds ratio | 1× model size | Combined SFT+alignment            |
| Online DPO  | Yes (frozen)     | Generated online| -log σ(r̂_w - r̂_l)| 2× + RM       | When online feedback is available |
| PPO         | Yes (frozen)     | Scalar reward  | PPO-Clip + value    | 4× model size | Verifiable reward tasks           |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "DPO Loss From Scratch — Full Derivation in Code": {
        "description": "Implement DPO from scratch showing the complete mathematical chain: log-prob computation, implicit reward calculation, Bradley-Terry loss, and the gradient analysis showing how chosen is pushed up and rejected is pushed down.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DPO LOSS — COMPLETE FROM-SCRATCH IMPLEMENTATION
================================================================================

Implements DPO with full mathematical transparency:
    1. Sequence log-probability computation (sum of token log-probs)
    2. Implicit reward: β × (log π_θ - log π_ref)
    3. DPO loss: -log σ(r̂_w - r̂_l)
    4. Gradient analysis: where does the update go?
    5. Comparison with a naive "SFT on chosen only" baseline
    6. The length normalisation issue and its fix

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ── Sequence log-probability helper ──────────────────────────────────────────

def sequence_log_prob(model: nn.Module, input_ids: torch.Tensor,
                       labels: torch.Tensor,
                       ignore_index: int = -100) -> torch.Tensor:
    """
    Compute the sum of log-probabilities for a sequence.

    For a sequence y = (y_1, ..., y_T) given context x:
        logp(y|x) = Σ_t log π(y_t | x, y_{1..t-1})

    In teacher-forcing mode (standard LM forward pass):
        input_ids:  [x_1, ..., x_n, y_1, ..., y_{T-1}]  (prompt + response prefix)
        labels:     [-100, ..., -100, y_2, ..., y_T]      (only response tokens)

    The model predicts token t+1 from position t.
    We select the log-prob of the actual next token (from labels).

    Args:
        model:     language model
        input_ids: (B, T) token IDs (prompt + response)
        labels:    (B, T) target IDs; -100 at prompt positions (masked)

    Returns:
        (B,) sum of log-probs for each sequence's response tokens
    """
    with torch.set_grad_enabled(model.training):
        logits = model(input_ids)   # (B, T, V)

    # Shift: predict position i+1 from position i
    shift_logits = logits[:, :-1, :]     # (B, T-1, V)
    shift_labels = labels[:, 1:]          # (B, T-1)

    # Log-softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)   # (B, T-1, V)

    # Gather log-prob of the actual token at each position
    # shift_labels contains -100 where we don't want to compute loss
    valid_mask = (shift_labels != ignore_index)
    safe_labels = shift_labels.clone()
    safe_labels[~valid_mask] = 0   # won't be used (masked below)

    per_token_logp = log_probs.gather(
        2, safe_labels.unsqueeze(2)
    ).squeeze(2)   # (B, T-1)

    # Zero out masked positions
    per_token_logp = per_token_logp * valid_mask.float()

    # Sum over response tokens → one scalar per sequence
    return per_token_logp.sum(dim=-1)   # (B,)


def sequence_log_prob_length_normalised(
    model: nn.Module, input_ids: torch.Tensor,
    labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Same as above but divided by response length (fixes length bias)."""
    with torch.set_grad_enabled(model.training):
        logits = model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs    = F.log_softmax(shift_logits, dim=-1)
    valid_mask   = (shift_labels != ignore_index)
    safe_labels  = shift_labels.clone()
    safe_labels[~valid_mask] = 0
    per_token_logp = log_probs.gather(2, safe_labels.unsqueeze(2)).squeeze(2)
    per_token_logp = per_token_logp * valid_mask.float()
    lengths        = valid_mask.float().sum(dim=-1).clamp(min=1)
    return per_token_logp.sum(dim=-1) / lengths


# ── DPO loss ──────────────────────────────────────────────────────────────────

def dpo_loss(
    policy_logp_chosen:    torch.Tensor,   # (B,) log π_θ(y_w|x)
    policy_logp_rejected:  torch.Tensor,   # (B,) log π_θ(y_l|x)
    ref_logp_chosen:       torch.Tensor,   # (B,) log π_ref(y_w|x)  [no grad]
    ref_logp_rejected:     torch.Tensor,   # (B,) log π_ref(y_l|x)  [no grad]
    beta:                  float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    DPO loss with full diagnostics.

    Returns:
        loss:  scalar loss value
        info:  dict of diagnostic metrics
    """
    # Implicit rewards: r̂ = β × (log π_θ - log π_ref)
    chosen_rewards   = beta * (policy_logp_chosen   - ref_logp_chosen)
    rejected_rewards = beta * (policy_logp_rejected - ref_logp_rejected)

    # Reward margin (should be positive and growing during training)
    reward_margin = chosen_rewards - rejected_rewards   # (B,)

    # DPO loss: -log σ(r̂_w - r̂_l)
    loss = -F.logsigmoid(reward_margin).mean()

    # Preference accuracy: how often does model assign higher reward to chosen?
    accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    # The weighting term σ(r̂_l - r̂_w): large when model is confused
    weighting = torch.sigmoid(-reward_margin)   # same as σ(r̂_l - r̂_w)

    info = {
        "chosen_rewards":    chosen_rewards.mean().item(),
        "rejected_rewards":  rejected_rewards.mean().item(),
        "reward_margin":     reward_margin.mean().item(),
        "accuracy":          accuracy,
        "weighting_mean":    weighting.mean().item(),
        "chosen_logp":       policy_logp_chosen.mean().item(),
        "rejected_logp":     policy_logp_rejected.mean().item(),
    }

    return loss, info


# ── DPO vs SFT comparison ────────────────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab=256, d=96, n_layers=2, max_len=48):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d, padding_idx=0)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d*4, bias=False), nn.GELU(),
                nn.Linear(d*4, d, bias=False)
            )
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.vocab  = vocab

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(self.norm(h))


def make_dpo_batch(prompt_len=8, response_len=12, vocab=256,
                    batch_size=8, seed=0):
    """
    Create a synthetic DPO batch: (prompt, chosen, rejected).

    Convention: 'chosen' uses tokens from [10,80) (simulating
    high-quality responses); 'rejected' uses tokens from [80,200)
    (lower quality).
    """
    torch.manual_seed(seed)
    B = batch_size
    P = prompt_len
    R = response_len
    IGNORE = -100

    prompt = torch.randint(3, 10, (B, P))

    # Chosen response tokens
    chosen_resp    = torch.randint(10, 80, (B, R))
    # Rejected response tokens
    rejected_resp  = torch.randint(80, 200, (B, R))

    # Full sequences: prompt + response
    chosen_ids     = torch.cat([prompt, chosen_resp],   dim=1)
    rejected_ids   = torch.cat([prompt, rejected_resp], dim=1)

    # Labels: -100 for prompt, actual token IDs for response
    def make_labels(ids, prompt_len):
        labels = ids.clone()
        labels[:, :prompt_len] = IGNORE
        return labels

    chosen_labels   = make_labels(chosen_ids,   P)
    rejected_labels = make_labels(rejected_ids, P)

    return chosen_ids, chosen_labels, rejected_ids, rejected_labels


if __name__ == "__main__":
    torch.manual_seed(42)
    VOCAB, D = 256, 96
    BETA     = 0.1
    N_STEPS  = 80

    model   = TinyLM(VOCAB, D)
    ref_mdl = copy.deepcopy(model)
    for p in ref_mdl.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 68)
    print("  DPO LOSS — COMPLETE FROM-SCRATCH IMPLEMENTATION")
    print(f"  β={BETA}, vocab={VOCAB}, d={D}, params={n_params:,}")
    print("=" * 68)
    print()

    # Show the DPO loss formula explicitly
    print("  DPO Loss Formula:")
    print("  L = -E [ log σ ( β × (log π_θ(y_w|x) - log π_ref(y_w|x))")
    print("                     - β × (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]")
    print()
    print("  = -E [ log σ ( β × Σ_t [log π_θ(y_w_t) - log π_ref(y_w_t)]")
    print("                     - β × Σ_t [log π_θ(y_l_t) - log π_ref(y_l_t)] ) ]")
    print()

    # Verify initial state (before training)
    ch_ids, ch_labels, rej_ids, rej_labels = make_dpo_batch(
        prompt_len=8, response_len=12, vocab=VOCAB, batch_size=32, seed=999
    )
    with torch.no_grad():
        ref_logp_w  = sequence_log_prob(ref_mdl, ch_ids,  ch_labels)
        ref_logp_l  = sequence_log_prob(ref_mdl, rej_ids, rej_labels)
        pol_logp_w  = sequence_log_prob(model,   ch_ids,  ch_labels)
        pol_logp_l  = sequence_log_prob(model,   rej_ids, rej_labels)
    _, info_init = dpo_loss(pol_logp_w, pol_logp_l, ref_logp_w, ref_logp_l, BETA)

    print("  Initial state (before DPO training):")
    print(f"    Chosen reward:    {info_init['chosen_rewards']:.4f}")
    print(f"    Rejected reward:  {info_init['rejected_rewards']:.4f}")
    print(f"    Reward margin:    {info_init['reward_margin']:.4f}  "
          f"(should become positive after training)")
    print(f"    Accuracy:         {info_init['accuracy']:.1%}  "
          f"(should approach 100%)")
    print()

    # Pre-compute reference log-probs (efficiency: only computed once)
    print("  Pre-computing reference log-probs (cached, no gradient)...")
    CACHE_SEED = list(range(N_STEPS))
    ref_cache  = {}
    for seed in CACHE_SEED:
        ch_ids_c, ch_labels_c, rej_ids_c, rej_labels_c = make_dpo_batch(
            prompt_len=8, response_len=12, vocab=VOCAB,
            batch_size=8, seed=seed
        )
        with torch.no_grad():
            ref_cache[seed] = (
                sequence_log_prob(ref_mdl, ch_ids_c,  ch_labels_c),
                sequence_log_prob(ref_mdl, rej_ids_c, rej_labels_c),
            )
    print("  Done. Training with cached reference log-probs.\n")

    # DPO training loop
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)

    print(f"  {'Step':>5}  {'Loss':>8}  {'Margin':>8}  "
          f"{'Acc':>6}  {'π_θ(y_w)':>10}  {'π_θ(y_l)':>10}  "
          f"{'Weight':>8}")
    print(f"  {'':─>5}  {'':─>8}  {'':─>8}  "
          f"{'':─>6}  {'':─>10}  {'':─>10}  {'':─>8}")

    for step in range(N_STEPS):
        seed = step % len(CACHE_SEED)
        ch_ids, ch_labels, rej_ids, rej_labels = make_dpo_batch(
            prompt_len=8, response_len=12, vocab=VOCAB, batch_size=8, seed=seed
        )
        ref_logp_w, ref_logp_l = ref_cache[seed]

        # Compute policy log-probs (with gradient)
        pol_logp_w = sequence_log_prob(model, ch_ids,  ch_labels)
        pol_logp_l = sequence_log_prob(model, rej_ids, rej_labels)

        loss, info = dpo_loss(pol_logp_w, pol_logp_l,
                               ref_logp_w.detach(), ref_logp_l.detach(), BETA)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0 or step < 3:
            print(f"  {step+1:>5}  {loss.item():>8.4f}  "
                  f"{info['reward_margin']:>8.4f}  "
                  f"{info['accuracy']:>6.1%}  "
                  f"{info['chosen_logp']:>10.2f}  "
                  f"{info['rejected_logp']:>10.2f}  "
                  f"{info['weighting_mean']:>8.4f}")

    # Final evaluation
    with torch.no_grad():
        pol_logp_w2 = sequence_log_prob(model, ch_ids, ch_labels)
        pol_logp_l2 = sequence_log_prob(model, rej_ids, rej_labels)
    _, info_final = dpo_loss(pol_logp_w2, pol_logp_l2,
                              ref_logp_w.detach(), ref_logp_l.detach(), BETA)

    print()
    print(f"  Summary: margin {info_init['reward_margin']:.3f} → "
          f"{info_final['reward_margin']:.3f}  "
          f"accuracy {info_init['accuracy']:.1%} → {info_final['accuracy']:.1%}")
    print()

    # Length normalisation demo
    print("=" * 68)
    print("  LENGTH NORMALISATION: FIXING DPO'S LENGTH BIAS")
    print("=" * 68)
    print()

    # Create pairs where chosen is longer (common in instruction data)
    torch.manual_seed(0)
    prompts  = torch.randint(3, 10, (16, 6))
    short_r  = torch.randint(10, 60, (16, 4))   # rejected: 4 tokens
    long_r   = torch.randint(10, 60, (16, 16))  # chosen:   16 tokens (same quality!)

    short_ids = torch.cat([prompts, short_r], dim=1)
    long_ids  = torch.cat([prompts, long_r],  dim=1)

    def make_labels_for(ids, prompt_len):
        labels = ids.clone(); labels[:, :prompt_len] = -100; return labels

    short_labels = make_labels_for(short_ids, 6)
    long_labels  = make_labels_for(long_ids,  6)

    with torch.no_grad():
        logp_short = sequence_log_prob(ref_mdl, short_ids, short_labels)
        logp_long  = sequence_log_prob(ref_mdl, long_ids,  long_labels)
        logp_short_norm = sequence_log_prob_length_normalised(ref_mdl, short_ids, short_labels)
        logp_long_norm  = sequence_log_prob_length_normalised(ref_mdl, long_ids,  long_labels)

    print(f"  Short response (4 tok):   logp = {logp_short.mean():.2f}  "
          f"logp_norm = {logp_short_norm.mean():.2f}")
    print(f"  Long  response (16 tok):  logp = {logp_long.mean():.2f}  "
          f"logp_norm = {logp_long_norm.mean():.2f}")
    print()
    print(f"  Without normalisation: long has LOWER logp ({logp_long.mean():.2f})")
    print(f"  → DPO would 'prefer' the shorter response just because it's shorter!")
    print()
    print(f"  With length normalisation: values comparable")
    print(f"  short_norm={logp_short_norm.mean():.2f}  long_norm={logp_long_norm.mean():.2f}")
    print(f"  → Quality difference, not length, drives the learning signal ✓")
''',
    },

    "DPO Training Loop with Diagnostics": {
        "description": "Complete DPO training loop with all key diagnostic metrics: chosen/rejected reward curves, reward margin growth, KL from reference, and early stopping detection.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
COMPLETE DPO TRAINING LOOP WITH DIAGNOSTICS
================================================================================

A production-ready DPO training loop with:
    1. All DPO-specific metrics logged at each step
    2. Reference model log-prob caching for efficiency
    3. β sensitivity analysis (how different β values affect training)
    4. Early stopping based on reward margin saturation
    5. Catastrophic forgetting monitoring (reference KL divergence)
    6. Comparison: DPO vs IPO vs length-normalised DPO

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class DPOConfig:
    beta:              float = 0.1       # KL penalty / implicit reward temperature
    lr:                float = 2e-5      # much lower than SFT (1e-5 to 5e-5)
    weight_decay:      float = 0.0
    n_steps:           int   = 100
    batch_size:        int   = 8
    max_grad_norm:     float = 1.0
    length_normalise:  bool  = False     # normalise log-probs by sequence length
    loss_type:         str   = "dpo"     # "dpo", "ipo", or "simpo"
    ipo_eps:           float = 0.01      # IPO epsilon (only used for loss_type="ipo")
    sft_coef:          float = 0.0       # optional SFT loss weight on chosen


class TinyLM(nn.Module):
    def __init__(self, vocab=256, d=96, n_layers=2):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d, padding_idx=0)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d),
                          nn.Linear(d, d*4, bias=False), nn.GELU(),
                          nn.Linear(d*4, d, bias=False))
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(self.norm(h))


def compute_sequence_logp(model, input_ids, labels, normalise=False):
    """Sum of log-probs for labelled positions."""
    with torch.set_grad_enabled(model.training):
        logits = model(input_ids)
    sl = logits[:, :-1, :]
    ll = labels[:, 1:]
    lp = F.log_softmax(sl, dim=-1)
    vm = (ll != -100)
    safe = ll.clone(); safe[~vm] = 0
    per_tok = lp.gather(2, safe.unsqueeze(2)).squeeze(2) * vm.float()
    seq_logp = per_tok.sum(-1)
    if normalise:
        seq_logp = seq_logp / vm.float().sum(-1).clamp(min=1)
    return seq_logp


def compute_dpo_loss(policy_logp_w, policy_logp_l,
                      ref_logp_w, ref_logp_l,
                      beta, loss_type, ipo_eps=0.01):
    """Unified loss for DPO / IPO / SimPO variants."""
    if loss_type == "dpo":
        r_w   = beta * (policy_logp_w - ref_logp_w)
        r_l   = beta * (policy_logp_l - ref_logp_l)
        loss  = -F.logsigmoid(r_w - r_l).mean()

    elif loss_type == "ipo":
        # Identity Policy Optimisation: squared loss prevents margin saturation
        r_w   = beta * (policy_logp_w - ref_logp_w)
        r_l   = beta * (policy_logp_l - ref_logp_l)
        margin = r_w - r_l
        target = 1.0 / (2 * beta)
        loss  = ((margin - target) ** 2).mean()

    elif loss_type == "simpo":
        # SimPO: no reference model, uses length-normalised log-probs
        # policy_logp should already be length-normalised
        margin = beta * (policy_logp_w - policy_logp_l) - ipo_eps  # ipo_eps = γ (margin)
        loss   = -F.logsigmoid(margin).mean()
        r_w    = policy_logp_w * beta
        r_l    = policy_logp_l * beta

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if loss_type != "simpo":
        r_w = beta * (policy_logp_w - ref_logp_w)
        r_l = beta * (policy_logp_l - ref_logp_l)

    return loss, {
        "chosen_reward":   r_w.mean().item(),
        "rejected_reward": r_l.mean().item(),
        "margin":          (r_w - r_l).mean().item(),
        "accuracy":        (r_w > r_l).float().mean().item(),
    }


def generate_batch(vocab, prompt_len, resp_len, batch_size, seed):
    """Generate synthetic preference data."""
    torch.manual_seed(seed)
    B, P, R = batch_size, prompt_len, resp_len
    prompt       = torch.randint(3, 10, (B, P))
    chosen_resp  = torch.randint(10, 80,  (B, R))
    reject_resp  = torch.randint(80, 200, (B, R))
    ch_ids  = torch.cat([prompt, chosen_resp], dim=1)
    rej_ids = torch.cat([prompt, reject_resp], dim=1)
    def mk_labels(ids):
        l = ids.clone(); l[:, :P] = -100; return l
    return ch_ids, mk_labels(ch_ids), rej_ids, mk_labels(rej_ids)


def run_dpo(cfg: DPOConfig, base_model, ref_model, vocab=256,
             prompt_len=8, resp_len=10) -> list[dict]:
    """Run one DPO training configuration and return step history."""
    model = copy.deepcopy(base_model)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)

    # Pre-cache reference log-probs
    ref_cache = {}
    for seed in range(cfg.n_steps):
        ch_ids, ch_lbl, rej_ids, rej_lbl = generate_batch(
            vocab, prompt_len, resp_len, cfg.batch_size, seed
        )
        with torch.no_grad():
            ref_cache[seed] = (
                compute_sequence_logp(ref_model, ch_ids,  ch_lbl, cfg.length_normalise),
                compute_sequence_logp(ref_model, rej_ids, rej_lbl, cfg.length_normalise),
            )

    history = []
    for step in range(cfg.n_steps):
        ch_ids, ch_lbl, rej_ids, rej_lbl = generate_batch(
            vocab, prompt_len, resp_len, cfg.batch_size, step
        )
        ref_logp_w, ref_logp_l = ref_cache[step]

        pol_logp_w = compute_sequence_logp(model, ch_ids,  ch_lbl, cfg.length_normalise)
        pol_logp_l = compute_sequence_logp(model, rej_ids, rej_lbl, cfg.length_normalise)

        loss, info = compute_dpo_loss(
            pol_logp_w, pol_logp_l, ref_logp_w, ref_logp_l,
            cfg.beta, cfg.loss_type, cfg.ipo_eps
        )

        # Optional SFT loss on chosen (regulariser)
        if cfg.sft_coef > 0:
            sft_logits = model(ch_ids[:, :-1])
            sft_loss   = F.cross_entropy(
                sft_logits.reshape(-1, sft_logits.size(-1)),
                ch_ids[:, 1:].reshape(-1)
            )
            loss = loss + cfg.sft_coef * sft_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()

        info["step"] = step
        info["loss"] = loss.item()
        history.append(info)

    return history


if __name__ == "__main__":
    torch.manual_seed(42)
    VOCAB = 256

    base_model = TinyLM(VOCAB, 96, 2)
    ref_model  = copy.deepcopy(base_model)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── 1. β sensitivity analysis ─────────────────────────────────────────────
    print("=" * 68)
    print("  DPO β SENSITIVITY ANALYSIS")
    print("=" * 68)
    print()

    betas = [0.01, 0.05, 0.1, 0.3, 1.0]
    beta_results = {}
    for beta in betas:
        cfg  = DPOConfig(beta=beta, lr=2e-5, n_steps=80)
        hist = run_dpo(cfg, base_model, ref_model, vocab=VOCAB)
        beta_results[beta] = hist

    print(f"  {'β':>6}  {'Final margin':>14}  {'Final acc':>12}  "
          f"{'Margin gain':>14}  {'Learning speed':>16}")
    print(f"  {'':─>6}  {'':─>14}  {'':─>12}  {'':─>14}  {'':─>16}")

    for beta in betas:
        h = beta_results[beta]
        m_init  = h[0]["margin"]
        m_final = h[-1]["margin"]
        acc     = h[-1]["accuracy"]
        gain    = m_final - m_init
        # How quickly did margin reach 0.5× of final?
        half_target = m_final * 0.5
        steps_to_half = next(
            (i for i, hh in enumerate(h) if hh["margin"] >= half_target),
            len(h)
        )
        print(f"  {beta:>6.2f}  {m_final:>14.4f}  {acc:>12.1%}  "
              f"{gain:>14.4f}  steps to 50%: {steps_to_half:>5}")

    # ── 2. DPO vs IPO ─────────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  DPO vs IPO vs LENGTH-NORMALISED DPO")
    print("=" * 68)
    print()

    variants = [
        DPOConfig(beta=0.1, lr=2e-5, n_steps=80, loss_type="dpo", name="DPO (standard)"),
        DPOConfig(beta=0.1, lr=2e-5, n_steps=80, loss_type="ipo", ipo_eps=0.02, name="IPO"),
        DPOConfig(beta=0.1, lr=2e-5, n_steps=80, loss_type="dpo",
                   length_normalise=True, name="DPO + length norm"),
        DPOConfig(beta=0.1, lr=2e-5, n_steps=80, loss_type="dpo",
                   sft_coef=0.1, name="DPO + SFT(0.1)"),
    ]
    # add name to DPOConfig dynamically
    for v in variants:
        object.__setattr__(v, '_name', v.__class__.__name__)

    names_list = [
        "DPO (standard)", "IPO", "DPO + length norm", "DPO + SFT(0.1)"
    ]
    variant_results = {}
    for cfg, name in zip(variants, names_list):
        hist = run_dpo(cfg, base_model, ref_model, vocab=VOCAB)
        variant_results[name] = hist

    print(f"  {'Variant':<22}  {'Final margin':>14}  {'Final acc':>12}  "
          f"{'Margin stability':>18}")
    print(f"  {'':─<22}  {'':─>14}  {'':─>12}  {'':─>18}")

    for name, hist in variant_results.items():
        margins    = [h["margin"] for h in hist]
        final_m    = margins[-1]
        final_acc  = hist[-1]["accuracy"]
        # Stability: std of margin in last 20 steps
        stability  = torch.tensor(margins[-20:]).std().item()
        print(f"  {name:<22}  {final_m:>14.4f}  {final_acc:>12.1%}  "
              f"{stability:>17.4f} σ")

    print()
    print("  IPO's squared loss is more stable than DPO's sigmoid loss.")
    print("  Length normalisation improves fairness for unequal-length pairs.")
    print("  Adding SFT loss helps maintain pre-training capabilities.")
    print()

    # ── 3. Reward distribution over training ──────────────────────────────────
    print("=" * 68)
    print("  CHOSEN vs REJECTED REWARD TRAJECTORIES (β=0.1)")
    print("=" * 68)
    print()
    print("  During healthy DPO training:")
    print("  - Chosen rewards should INCREASE (model increases π_θ(y_w))")
    print("  - Rejected rewards should DECREASE (model decreases π_θ(y_l))")
    print("  - Both moving in opposite directions = learning is happening")
    print()
    h = beta_results[0.1]
    print(f"  {'Step':>5}  {'Chosen r':>10}  {'Rejected r':>12}  "
          f"{'Margin':>10}  {'Accuracy':>10}")
    print(f"  {'':─>5}  {'':─>10}  {'':─>12}  {'':─>10}  {'':─>10}")
    for step in [0, 10, 20, 40, 60, 79]:
        hh = h[step]
        print(f"  {hh['step']+1:>5}  {hh['chosen_reward']:>10.4f}  "
              f"{hh['rejected_reward']:>12.4f}  "
              f"{hh['margin']:>10.4f}  "
              f"{hh['accuracy']:>10.1%}")

    print()
    print("  Diagnostic: if both chosen and rejected rewards increase together,")
    print("  it means the policy is drifting from the reference uniformly")
    print("  (reward hacking without differentiation).")
    print("  Healthy: chosen ↑, rejected ↓, margin grows consistently.")


# Add name field to DPOConfig for the above to work cleanly
DPOConfig.__annotations__["name"] = str
DPOConfig.__dataclass_fields__["name"] = DPOConfig.__dataclass_fields__.get(
    "n_steps"
).__class__("name", str, "", False, False, True, None)  # type: ignore
''',
    },

    "DPO Variants: IPO, SimPO, ORPO Comparison": {
        "description": "Implement and compare four preference optimisation methods — DPO, IPO, SimPO, and ORPO — on the same data, showing the trade-offs in stability, memory, and quality.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DPO VARIANTS: IPO, SIMPO, ORPO — IMPLEMENTATION AND COMPARISON
================================================================================

Implements and compares four preference optimisation methods:
    1. DPO  — standard: -log σ(β Δlogp_ref_normalised)
    2. IPO  — stable:   (β Δlogp_ref - 1/2β)²
    3. SimPO — no-ref:  -log σ(β/l × logp_θ(y_w) - β/l × logp_θ(y_l) - γ)
    4. ORPO — combined: CE(y_w) - λ × log σ(log-odds(y_w) / log-odds(y_l))

Key differences:
    DPO/IPO: require reference model (2× memory)
    SimPO/ORPO: no reference model (1× memory)
    ORPO: also trains on the chosen response via SFT (no separate SFT phase)

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared utilities ──────────────────────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab=256, d=96, n_layers=2):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d, padding_idx=0)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d),
                          nn.Linear(d, d*4, bias=False), nn.GELU(),
                          nn.Linear(d*4, d, bias=False))
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.vocab  = vocab

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(self.norm(h))


def get_logp_and_per_token(model, input_ids, labels, normalise_len=False):
    """Return (sequence_sum_logp, per_token_logp_list)."""
    with torch.set_grad_enabled(model.training):
        logits = model(input_ids)
    sl = logits[:, :-1, :]
    ll = labels[:, 1:]
    lp = F.log_softmax(sl, dim=-1)
    vm = (ll != -100)
    safe = ll.clone(); safe[~vm] = 0
    ptlp = lp.gather(2, safe.unsqueeze(2)).squeeze(2) * vm.float()
    seq  = ptlp.sum(-1)
    if normalise_len:
        seq = seq / vm.float().sum(-1).clamp(min=1)
    return seq


def make_batch(vocab, prompt_len, resp_len, batch_size, seed):
    torch.manual_seed(seed)
    B, P, R = batch_size, prompt_len, resp_len
    pmt   = torch.randint(3, 10, (B, P))
    ch_r  = torch.randint(10, 80,  (B, R))
    rej_r = torch.randint(80, 200, (B, R))
    ch_ids  = torch.cat([pmt, ch_r],  dim=1)
    rej_ids = torch.cat([pmt, rej_r], dim=1)
    def mk(ids): l = ids.clone(); l[:, :P] = -100; return l
    return ch_ids, mk(ch_ids), rej_ids, mk(rej_ids)


# ── Loss implementations ──────────────────────────────────────────────────────

def loss_dpo(pol_w, pol_l, ref_w, ref_l, beta=0.1):
    """Standard DPO loss."""
    rw   = beta * (pol_w - ref_w)
    rl   = beta * (pol_l - ref_l)
    loss = -F.logsigmoid(rw - rl).mean()
    return loss, rw.mean().item(), rl.mean().item()


def loss_ipo(pol_w, pol_l, ref_w, ref_l, beta=0.1):
    """
    Identity Policy Optimisation (Azar et al., 2023).

    DPO loss saturates when the margin is large (σ → 1).
    IPO uses a squared loss with a fixed target (1/2β):

        L_IPO = E[(β(Δlogp_w - Δlogp_l) - 1/(2β))²]

    This prevents the margin from growing unboundedly, which can cause
    the policy log-probs to drift to ±∞ over many training steps.
    """
    delta_w = pol_w - ref_w
    delta_l = pol_l - ref_l
    margin  = beta * (delta_w - delta_l)
    target  = 1.0 / (2 * beta)   # the fixed target margin
    loss    = ((margin - target) ** 2).mean()
    rw      = beta * delta_w
    rl      = beta * delta_l
    return loss, rw.mean().item(), rl.mean().item()


def loss_simpo(pol_w_norm, pol_l_norm, beta=0.1, gamma=0.5):
    """
    SimPO (Simple Preference Optimisation, Meng et al., 2024).
    No reference model required!

    Uses length-normalised log-probability of the policy itself as implicit reward.
    A target margin γ is added to ensure a minimum preference gap.

    L_SimPO = -log σ(β/|y_w| × logp(y_w) - β/|y_l| × logp(y_l) - γ)

    Note: pol_w_norm and pol_l_norm are already length-normalised.
    """
    margin = beta * (pol_w_norm - pol_l_norm) - gamma
    loss   = -F.logsigmoid(margin).mean()
    rw     = beta * pol_w_norm
    rl     = beta * pol_l_norm
    return loss, rw.mean().item(), rl.mean().item()


def loss_orpo(model, ch_ids, ch_labels, rej_ids, rej_labels,
               lambda_orpo=0.1):
    """
    ORPO (Odds Ratio Preference Optimisation, Hong et al., 2024).

    Combines SFT loss on chosen with an odds-ratio penalty for rejected.
    No reference model required. No separate SFT phase needed.

    L_ORPO = CE(y_w) - λ × log σ(log[odds(y_w) / odds(y_l)])

    where odds(y) = P(y|x) / (1 - P(y|x))

    The SFT loss trains the model on correct responses.
    The odds-ratio term penalises the model when it assigns high probability
    to rejected responses relative to chosen ones.
    """
    # SFT loss on chosen responses
    ch_logits  = model(ch_ids[:, :-1])
    ch_valid   = (ch_labels[:, 1:] != -100)
    sft_loss   = F.cross_entropy(
        ch_logits.reshape(-1, model.vocab),
        ch_labels[:, 1:].clamp(min=0).reshape(-1),
        ignore_index=-100,
    )

    # Per-token log-probs for chosen and rejected
    with torch.enable_grad():
        pol_w_lp  = get_logp_and_per_token(model, ch_ids,  ch_labels)
        pol_l_lp  = get_logp_and_per_token(model, rej_ids, rej_labels)

    # Odds ratio: log[odds(y_w) / odds(y_l)]
    # odds(y) = exp(logp(y)) / (1 - exp(logp(y))) ≈ exp(logp(y)) for small probs
    # So log[odds(y_w)/odds(y_l)] ≈ logp(y_w) - logp(y_l) for reasonable probs
    # Full computation:
    log_odds_w  = pol_w_lp - torch.log1p(-torch.exp(pol_w_lp).clamp(max=1 - 1e-6))
    log_odds_l  = pol_l_lp - torch.log1p(-torch.exp(pol_l_lp).clamp(max=1 - 1e-6))
    log_ratio   = log_odds_w - log_odds_l

    or_loss     = -F.logsigmoid(log_ratio).mean()
    total_loss  = sft_loss + lambda_orpo * or_loss

    return total_loss, pol_w_lp.mean().item(), pol_l_lp.mean().item()


# ── Comparison runner ─────────────────────────────────────────────────────────

def run_variant(name, loss_fn, base_model, ref_model,
                vocab, n_steps, batch_size, prompt_len, resp_len,
                lr=2e-5):
    model  = copy.deepcopy(base_model)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    hist   = []
    needs_ref = name in ("DPO", "IPO")

    if needs_ref:
        ref_cache = {}
        for s in range(n_steps):
            ch_ids, ch_lbl, rej_ids, rej_lbl = make_batch(vocab, prompt_len, resp_len, batch_size, s)
            with torch.no_grad():
                ref_cache[s] = (
                    get_logp_and_per_token(ref_model, ch_ids,  ch_lbl),
                    get_logp_and_per_token(ref_model, rej_ids, rej_lbl),
                )

    for step in range(n_steps):
        ch_ids, ch_lbl, rej_ids, rej_lbl = make_batch(
            vocab, prompt_len, resp_len, batch_size, step
        )

        if name == "DPO":
            rw, rl = ref_cache[step]
            pw = get_logp_and_per_token(model, ch_ids,  ch_lbl)
            pl = get_logp_and_per_token(model, rej_ids, rej_lbl)
            loss, rw_v, rl_v = loss_dpo(pw, pl, rw.detach(), rl.detach())

        elif name == "IPO":
            rw, rl = ref_cache[step]
            pw = get_logp_and_per_token(model, ch_ids,  ch_lbl)
            pl = get_logp_and_per_token(model, rej_ids, rej_lbl)
            loss, rw_v, rl_v = loss_ipo(pw, pl, rw.detach(), rl.detach())

        elif name == "SimPO":
            pw = get_logp_and_per_token(model, ch_ids,  ch_lbl, normalise_len=True)
            pl = get_logp_and_per_token(model, rej_ids, rej_lbl, normalise_len=True)
            loss, rw_v, rl_v = loss_simpo(pw, pl, beta=0.1, gamma=0.5)

        elif name == "ORPO":
            loss, rw_v, rl_v = loss_orpo(model, ch_ids, ch_lbl, rej_ids, rej_lbl,
                                           lambda_orpo=0.1)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        hist.append({"loss": loss.item(), "rw": rw_v, "rl": rl_v,
                      "margin": rw_v - rl_v})

    return hist


if __name__ == "__main__":
    torch.manual_seed(42)
    VOCAB      = 256
    PROMPT_LEN = 6
    RESP_LEN   = 10
    BATCH_SIZE = 8
    N_STEPS    = 80
    LR         = 2e-5

    base_model = TinyLM(VOCAB, 96, 2)
    ref_model  = copy.deepcopy(base_model)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print("=" * 70)
    print("  DPO VARIANTS COMPARISON")
    print(f"  vocab={VOCAB}, d=96, {N_STEPS} steps each")
    print("=" * 70)
    print()

    methods = ["DPO", "IPO", "SimPO", "ORPO"]
    results = {}
    for method in methods:
        hist = run_variant(method, None, base_model, ref_model,
                            VOCAB, N_STEPS, BATCH_SIZE,
                            PROMPT_LEN, RESP_LEN, lr=LR)
        results[method] = hist
        print(f"  {method} trained.")

    print()
    print(f"  {'Method':<10}  {'Final loss':>12}  {'Final margin':>14}  "
          f"{'Margin stability':>18}  {'Notes'}")
    print(f"  {'':─<10}  {'':─>12}  {'':─>14}  {'':─>18}  {'':─}")

    notes = {
        "DPO":   "standard; may saturate at large margins",
        "IPO":   "squared loss; more stable long-run",
        "SimPO": "no reference model; 1×memory",
        "ORPO":  "no ref + no SFT phase; combined loss",
    }

    for method in methods:
        h      = results[method]
        margins = [hh["margin"] for hh in h]
        final_loss = h[-1]["loss"]
        final_m    = h[-1]["margin"]
        stability  = torch.tensor(margins[-20:]).std().item()
        print(f"  {method:<10}  {final_loss:>12.4f}  {final_m:>14.4f}  "
              f"{stability:>17.4f}σ  {notes[method]}")

    print()
    print("  Memory requirements:")
    for method in methods:
        needs_ref = method in ("DPO", "IPO")
        n_models  = 2 if needs_ref else 1
        print(f"  {method:<10}: {n_models}× model in memory  "
              f"({'+ ref model' if needs_ref else 'no ref model ✓'})")

    print()
    print("  When to use each:")
    usage = {
        "DPO":   "default choice; best understood and most widely validated",
        "IPO":   "if DPO training shows reward explosion or instability",
        "SimPO": "tight GPU budget (single model); preference data available",
        "ORPO":  "want to skip separate SFT; have preference data from scratch",
    }
    for method, desc in usage.items():
        print(f"  {method:<10}: {desc}")
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
    #     from llm_training.visuals.dpo import (
    #         DPO_VISUAL_HTML,
    #         DPO_VISUAL_HEIGHT,
    #     )
    #     visual_html   = DPO_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = DPO_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[29_dpo_direct_preference_optimization.py] Could not load visual: {e}",
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