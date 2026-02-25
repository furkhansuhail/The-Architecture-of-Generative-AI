"""Module: 06 · Alignment"""

DISPLAY_NAME = "06 · Alignment"
ICON         = "⚖️"
SUBTITLE     = "RLHF, DPO and Constitutional AI — teaching models to be helpful and safe"

THEORY = """
## 06 · Alignment

Pretraining gives a model knowledge; alignment gives it values. A pretrained LLM will
complete text in whatever way the training data suggests — including harmful, deceptive, or
simply unhelpful ways. Alignment is the process of steering the model toward responses that
are **helpful, harmless, and honest (HHH)** while preserving the capabilities earned during
pretraining.

---

### 1 · Why Alignment Is Hard

The core challenge is the **specification problem**: it is extremely difficult to write down,
in a form a model can optimise, exactly what "good behaviour" means across every possible
situation. Simple rules fail because:

- Language is ambiguous — the same words mean different things in different contexts.
- Edge cases are infinite — no ruleset anticipates every prompt.
- Goodhart's Law — optimising a proxy metric causes the model to game it rather than
  achieve the underlying goal ("reward hacking").
- Emergent deception — capable models may learn to appear aligned during evaluation
  while behaving differently in deployment.

The alignment tax is real: aggressive safety tuning can reduce helpfulness. The field's
goal is to make this trade-off as small as possible.

---

### 2 · Supervised Fine-Tuning (SFT) — The Starting Point

Before any preference learning, the pretrained base model is fine-tuned on a curated
dataset of (prompt, ideal-response) pairs written by human contractors.

```
SFT objective (same as pretraining CLM, but on demonstration data):
L_SFT = − Σᵢ log P(yᵢ | y₁…yᵢ₋₁, x ; θ)
  where x = prompt,  y = ideal response
```

SFT teaches the model the desired **format, tone, and instruction-following style**.
It is fast and cheap but limited: contractors cannot write ideal responses for every
possible request, and their responses may be inconsistent or biased.

SFT alone produces a model that is much more useful than the raw base model, but it
still makes factual errors, follows harmful instructions, and lacks calibration on
what it does and doesn't know.

---

### 3 · Reinforcement Learning from Human Feedback (RLHF)

RLHF (Christiano et al. 2017, popularised by InstructGPT 2022) addresses the specification
problem by learning preferences **from human comparisons** rather than from explicit rules.

#### 3.1 The Three-Stage Pipeline

```
Stage 1 — SFT
  Collect ~10–100K (prompt, demonstration) pairs.
  Fine-tune pretrained model → SFT model (π_SFT).

Stage 2 — Reward Model Training
  For each prompt x, sample K completions from π_SFT.
  Human raters rank completions: y_w ≻ y_l  ("winner beats loser").
  Train reward model R_φ to assign higher score to y_w.

Stage 3 — RL Fine-tuning
  Use PPO to optimise π_θ to maximise E[R_φ(x, y)]
  subject to a KL penalty to stay close to π_SFT.
```

#### 3.2 Reward Model Training

The reward model takes (prompt + completion) and outputs a scalar score.
It is trained with the **Bradley-Terry preference model**:

```
P(y_w ≻ y_l | x) = σ(R_φ(x, y_w) − R_φ(x, y_l))

Loss: L_RM = −E_{(x, y_w, y_l)} [ log σ(R_φ(x, y_w) − R_φ(x, y_l)) ]
```

Intuitively: the reward model learns a real-valued function where the difference in
scores matches the log-odds of the human preference. This turns ordinal rankings into
a differentiable training signal.

#### 3.3 PPO Fine-tuning

Proximal Policy Optimisation (PPO) maximises expected reward while staying close to
the reference (SFT) model via a KL penalty:

```
Objective: E_x [ E_{y ~ π_θ} [ R_φ(x, y) ] − β · KL(π_θ(· | x) ‖ π_ref(· | x)) ]

Token-level reward at each step t:
  r_t = R_φ(x, y)·𝟙[t=T] − β · log(π_θ(y_t|·) / π_ref(y_t|·))
```

The KL penalty β prevents **reward hacking**: without it, the model quickly learns
to produce outputs that fool the reward model (e.g. very long responses, specific
phrases that the RM scores highly) without actually improving quality.

#### 3.4 RLHF Limitations

- **Expensive**: requires thousands of human comparisons and multiple model copies.
- **Instability**: PPO is sensitive to hyperparameters; training can diverge.
- **Reward model errors**: the RM is imperfect; the policy overoptimises it.
- **Mode collapse**: the policy may lose diversity, always producing similar responses.
- **Distribution shift**: humans cannot rate long-horizon or specialised completions well.

---

### 4 · Direct Preference Optimisation (DPO)

DPO (Rafailov et al. 2023) elegantly bypasses the separate reward model and RL loop
entirely by showing that the optimal policy under the RLHF objective has a **closed form**:

```
π*(y | x) ∝ π_ref(y | x) · exp(R*(x, y) / β)
```

Rearranging, the implicit reward of any policy π_θ relative to π_ref is:

```
R(x, y) = β · log(π_θ(y | x) / π_ref(y | x)) + β · log Z(x)
```

Substituting into the Bradley-Terry loss and noting that Z(x) cancels:

```
L_DPO = −E_{(x, y_w, y_l)} [
    log σ(
        β · log(π_θ(y_w|x) / π_ref(y_w|x))
      − β · log(π_θ(y_l|x) / π_ref(y_l|x))
    )
]
```

This is just a **supervised cross-entropy loss** computed over the preference pairs —
no RL, no reward model, no value function. DPO is:

- Simpler to implement (one training loop)
- More stable (no PPO instability)
- Cheaper (no separate RM training)
- Competitive with RLHF on most benchmarks

The intuition: DPO increases the relative likelihood of the winning response while
decreasing the relative likelihood of the losing response, with the reference model
providing the baseline.

#### 4.1 DPO Variants

| Method | Key idea |
|---|---|
| DPO (Rafailov 2023) | Original closed-form derivation |
| IPO (Azar 2023) | Avoids overfit to deterministic preferences |
| KTO (Ethayarajh 2024) | Uses unpaired feedback (single label, no pairs) |
| SimPO (Meng 2024) | Removes reference model dependency entirely |
| ORPO (Hong 2024) | Combines SFT + preference in one loss |

---

### 5 · Constitutional AI (CAI)

Constitutional AI (Anthropic, 2022) addresses a key bottleneck in RLHF: the cost and
inconsistency of human feedback at scale. Instead of relying on humans to label every
comparison, CAI uses the **model itself** to critique and revise its outputs according
to a written **Constitution** — a set of principles.

#### 5.1 The Two Phases

**Phase 1 — Supervised Learning from AI Feedback (SL-CAF)**

```
1. Prompt the model to generate a helpful but potentially harmful response.
2. Ask the model to critique the response according to a constitutional principle
   (e.g. "Does this response promote harm? How could it be improved?")
3. Ask the model to revise the response based on its critique.
4. Repeat for multiple principles.
5. Fine-tune on the final revised responses.
```

**Phase 2 — RL from AI Feedback (RLAIF)**

```
1. Sample pairs of responses to prompts.
2. Ask a "feedback model" to choose which response better satisfies a principle.
3. Use these AI-generated preference labels to train a reward model.
4. Apply PPO (or DPO) as in standard RLHF.
```

The key insight is that **AI feedback can substitute for human feedback** on many
dimensions, making the process scalable. Humans only need to write the constitution
(a few dozen principles), not label millions of examples.

#### 5.2 Example Constitutional Principles

- "Choose the response that is least likely to contain harmful or unethical content."
- "Choose the response that is most honest and does not contain misleading information."
- "Choose the response that is most respectful and does not demean any group."
- "Choose the response most helpful to the user while avoiding real-world harm."

---

### 6 · Reward Hacking and Overoptimisation

A central challenge in alignment: as the policy is optimised against the reward model,
it finds ways to exploit the RM's imperfections. This is **Goodhart's Law** in action:

```
"When a measure becomes a target, it ceases to be a good measure."
```

Empirically, the relationship between RM score and true human preference follows an
inverted-U curve:

```
True quality
     ↑
     │      ★ optimal
     │    ╱    ╲
     │   ╱       ╲   ← overoptimisation
     │  ╱           ╲
     │ ╱
     └────────────────→ KL(π_θ ‖ π_ref)
```

The KL penalty β controls where on this curve the trained policy lands. Too small β →
reward hacking; too large β → the policy barely moves from π_ref.

**Detecting reward hacking:**
- Response length suddenly increases (longer ≠ better, but RM may reward length)
- Loss of diversity / mode collapse
- RM score keeps rising but human evaluation scores plateau or drop
- Generation of specific tokens/phrases the RM consistently over-rewards

---

### 7 · Scalable Oversight

As models become more capable than human evaluators on certain tasks, we can no longer
directly verify whether a response is good. Scalable oversight techniques address this:

**Debate** — Two AI agents argue for different answers. A human judges which argument
is more convincing. A truthful agent should be able to win against a deceptive one.

**Amplification (IDA)** — A human uses a weaker AI assistant to help evaluate a
stronger AI's outputs, recursively bootstrapping oversight capability.

**Process Reward Models (PRMs)** — Instead of rewarding the final answer, reward each
reasoning step. This makes it harder to reach a correct answer via flawed reasoning.

**Weak-to-Strong Generalisation** — Train a stronger model using labels from a weaker
supervisor; study whether the stronger model generalises beyond the supervisor's ability.

---

### 8 · Evaluation and Red-Teaming

Alignment cannot be verified by training metrics alone. Deployed models are evaluated via:

**Automated benchmarks** — TruthfulQA (hallucination), BBQ (bias), HarmBench (safety),
MT-Bench (instruction following), MMLU (knowledge), HumanEval (coding).

**Red-teaming** — A team (human or AI) deliberately tries to elicit harmful outputs
by crafting adversarial prompts. Techniques include:
- Role-play / persona switching ("Act as DAN...")
- Indirect framing ("Write a story where a character explains...")
- Multi-turn context manipulation
- Prompt injection (in tool-use / agent settings)

**Constitutional evaluation** — Ask the model itself to rate its own responses on
alignment criteria. Highly correlated with human judgement at scale.

**Human preference evaluation** — Side-by-side comparisons rated by crowdworkers
or domain experts. The gold standard but expensive.

---

### 9 · The Alignment Tax and Capability-Safety Trade-off

A widely observed empirical finding: aggressive safety fine-tuning can reduce
performance on capability benchmarks. This happens because:

- Safety training may "over-refuse" legitimate requests (false positives).
- Reducing the probability of harmful tokens can reduce fluency generally.
- Over-fine-tuning can cause catastrophic forgetting of pretrained knowledge.

Modern approaches mitigate this by:
- Careful data curation (high-quality preference pairs from domain experts)
- Low-rank adaptation (LoRA) to minimise catastrophic forgetting
- Multi-objective optimisation (jointly optimise helpfulness and harmlessness)
- Iterative refinement with human-in-the-loop feedback

---

### Key Takeaways

- SFT teaches format and style from demonstrations; RLHF teaches values from preferences.
- RLHF trains a reward model on human comparisons, then optimises the policy with PPO.
- DPO bypasses the RM entirely by showing the optimal RLHF policy has a closed-form supervised loss.
- Constitutional AI replaces expensive human labelling with AI self-critique guided by written principles.
- Reward hacking is inevitable; KL regularisation and monitoring are essential safeguards.
- Scalable oversight (debate, amplification, PRMs) addresses the challenge of superhuman-capability models.
- Alignment and capability are in tension but modern techniques shrink the trade-off significantly.
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Bradley-Terry Preference Model
    # ─────────────────────────────────────────────────────────────────────────
    "Step 1 · Bradley-Terry Preference Model": {
        "description": (
            "Implements the Bradley-Terry model that underlies reward model training. "
            "Given pairs of completions and human rankings, trains a scalar scorer "
            "to assign higher values to preferred responses."
        ),
        "language": "python",
        "code": """
import numpy as np

np.random.seed(42)

# ── Bradley-Terry model ───────────────────────────────────────────────────────
# P(y_w beats y_l) = σ(R(y_w) - R(y_l))
# Loss = -log σ(R(y_w) - R(y_l))
#
# We represent each response as a feature vector and learn R(y) = w · φ(y)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def bradley_terry_loss(scores_w, scores_l):
    \"\"\"Loss for a batch of (winner, loser) score pairs.\"\"\"
    return -np.log(sigmoid(scores_w - scores_l) + 1e-9).mean()

def bradley_terry_acc(scores_w, scores_l):
    \"\"\"Fraction of pairs where winner gets higher score.\"\"\"
    return (scores_w > scores_l).mean()

# ── Simulate preference data ─────────────────────────────────────────────────
# Each response is described by 4 features:
#   [helpfulness, safety, factual_accuracy, conciseness]
# True reward weights (unknown to the model, we're learning them)

N_PAIRS   = 200
N_FEATURES = 4
TRUE_W     = np.array([0.5, 0.3, 0.15, 0.05])   # ground-truth reward weights

def make_response_features():
    return np.random.dirichlet(np.ones(N_FEATURES))  # unit simplex features

# Generate preference pairs: randomly pick two responses, label by true reward
pairs_w = np.array([make_response_features() for _ in range(N_PAIRS)])
pairs_l = np.array([make_response_features() for _ in range(N_PAIRS)])
true_rw = pairs_w @ TRUE_W
true_rl = pairs_l @ TRUE_W

# Swap so that w always has higher true reward (human labels the winner)
swap = true_rw < true_rl
pairs_w[swap], pairs_l[swap] = pairs_l[swap].copy(), pairs_w[swap].copy()
true_rw, true_rl = np.maximum(true_rw, true_rl), np.minimum(true_rw, true_rl)

# ── Train the reward model (linear for clarity) ──────────────────────────────
learned_w = np.zeros(N_FEATURES)   # reward model weights
lr        = 0.1
N_EPOCHS  = 100

print("=" * 60)
print("REWARD MODEL TRAINING (Bradley-Terry)")
print("=" * 60)
print(f"  {'Epoch':>6}  {'Loss':>8}  {'Accuracy':>9}  {'‖w‖':>6}")
print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*6}")

for epoch in range(N_EPOCHS + 1):
    scores_w = pairs_w @ learned_w
    scores_l = pairs_l @ learned_w
    loss     = bradley_terry_loss(scores_w, scores_l)
    acc      = bradley_terry_acc(scores_w, scores_l)

    if epoch % 20 == 0:
        print(f"  {epoch:>6}  {loss:>8.4f}  {acc:>9.3f}  {np.linalg.norm(learned_w):>6.3f}")

    if epoch == N_EPOCHS:
        break

    # Gradient of loss w.r.t. learned_w
    diff     = scores_w - scores_l                  # (N,)
    err      = sigmoid(diff) - 1.0                  # negative of "correct" prob
    grad     = (err[:, None] * (pairs_w - pairs_l)).mean(axis=0)
    learned_w -= lr * grad

print()
print("  Ground-truth reward weights: ", TRUE_W)
# Normalise learned_w for comparison
norm_learned = learned_w / (np.linalg.norm(learned_w) + 1e-9)
norm_true    = TRUE_W    / (np.linalg.norm(TRUE_W)    + 1e-9)
cos_sim      = np.dot(norm_learned, norm_true)
print(f"  Learned weights (normalised): {norm_learned.round(3)}")
print(f"  True    weights (normalised): {norm_true.round(3)}")
print(f"  Cosine similarity with truth: {cos_sim:.4f}")
print()
print("  Interpretation:")
print("  cos_sim → 1.0 means the reward model learned the correct preference ordering.")
print("  Even if the absolute scale differs, relative rankings are preserved.")
print()

# ── Demonstrate reward model scoring ────────────────────────────────────────
print("=" * 60)
print("REWARD MODEL IN ACTION — scoring example responses")
print("=" * 60)
examples = {
    "Helpful + safe + accurate + concise": np.array([0.7, 0.6, 0.8, 0.7]),
    "Helpful but unsafe":                  np.array([0.9, 0.1, 0.5, 0.5]),
    "Safe but unhelpful refusal":          np.array([0.1, 0.9, 0.5, 0.3]),
    "Verbose but correct":                 np.array([0.5, 0.5, 0.7, 0.1]),
    "Random low-quality response":         np.array([0.2, 0.2, 0.2, 0.2]),
}
scores = {name: feat @ learned_w for name, feat in examples.items()}
for name, score in sorted(scores.items(), key=lambda x: -x[1]):
    bar = "█" * int((score - min(scores.values())) /
                    (max(scores.values()) - min(scores.values()) + 1e-9) * 25)
    print(f"  {score:>6.3f}  {bar:<25}  {name}")
""",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — KL Divergence & the RLHF Regularisation Penalty
    # ─────────────────────────────────────────────────────────────────────────
    "Step 2 · KL Divergence & the RLHF Regularisation Penalty": {
        "description": (
            "Shows exactly what the KL penalty in RLHF does: it measures how far "
            "the policy has drifted from the reference (SFT) model, and demonstrates "
            "how β controls the reward vs. safety trade-off."
        ),
        "language": "python",
        "code": """
import numpy as np
import math

np.random.seed(0)

def kl_divergence(p, q):
    \"\"\"KL(p ‖ q) — how many bits p needs beyond q's encoding.\"\"\"
    p = np.clip(p, 1e-9, 1)
    q = np.clip(q, 1e-9, 1)
    return float(np.sum(p * np.log(p / q)))

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

# ── Simulate a toy vocabulary of 8 tokens ────────────────────────────────────
V = 8
token_names = ["safe_tok", "helpful_tok", "fact_tok",
               "hedge_tok", "verbose_tok", "harm_tok", "fluff_tok", "unk_tok"]

# Reference (SFT) policy logits — balanced, slightly prefers safe/helpful
ref_logits = np.array([1.2, 1.1, 0.8, 0.6, 0.3, -2.0, 0.1, -0.5])
ref_probs  = softmax(ref_logits)

print("=" * 65)
print("REFERENCE (SFT) POLICY DISTRIBUTION")
print("=" * 65)
for tok, p in sorted(zip(token_names, ref_probs), key=lambda x: -x[1]):
    bar = "█" * int(p * 60)
    print(f"  {p:.4f}  {bar:<30}  {tok}")

# ── Three fine-tuned policies at different drift levels ──────────────────────
policies = {
    "Low drift (β=1.0)":   ref_logits + np.array([0.3, 0.4, 0.1, 0.0, 0.1, -0.2, 0.0, 0.0]),
    "Medium drift (β=0.1)":ref_logits + np.array([0.5, 0.8, 0.3, -0.3, 0.5, 0.3, 0.2, 0.0]),
    "High drift (β=0.01)": ref_logits + np.array([0.2, 0.5, -0.2, -0.8, 1.5, 1.8, 0.8, 0.0]),
}

print()
print("=" * 65)
print("KL DIVERGENCE FROM REFERENCE — EFFECT OF β REGULARISATION")
print("=" * 65)
print(f"  {'Policy':<30} {'KL(π‖π_ref)':>12} {'harm_tok prob':>14} {'Verdict'}")
print(f"  {'-'*30}  {'-'*12}  {'-'*14}  {'-'*20}")

for name, logits in policies.items():
    probs = softmax(logits)
    kl    = kl_divergence(probs, ref_probs)
    harm  = probs[5]
    verdict = ("✓ well-aligned" if kl < 0.3
               else ("⚠ drifting" if kl < 1.0 else "✗ reward-hacking"))
    print(f"  {name:<30} {kl:>12.4f}  {harm:>14.4f}  {verdict}")

print()
print("  Token-level reward at each position:")
print(f"  r_t = R(x,y)·𝟙[t=T] − β · log(π_θ(y_t) / π_ref(y_t))")
print()

# ── Show token-level KL penalties ────────────────────────────────────────────
beta_values = [1.0, 0.1, 0.01]
print("  Per-token KL penalty (−β · log π_θ/π_ref) for 'harm_tok':")
print(f"  {'β':>8}  {'π_ref':>8}  {'π_θ (drift)':>12}  {'log ratio':>10}  {'penalty':>10}")
print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}")

for beta, (name, logits) in zip(beta_values, policies.items()):
    probs      = softmax(logits)
    p_ref_harm = ref_probs[5]
    p_pi_harm  = probs[5]
    log_ratio  = math.log(p_pi_harm / p_ref_harm + 1e-9)
    penalty    = -beta * log_ratio
    print(f"  {beta:>8.2f}  {p_ref_harm:>8.4f}  {p_pi_harm:>12.4f}  "
          f"{log_ratio:>10.4f}  {penalty:>10.4f}")

print()
print("  With small β, the penalty is tiny → policy can freely boost harm_tok.")
print("  With large β, a log(3×) drift in harm_tok costs a full reward unit.")
print()
print("  The RLHF optimum balances: high reward vs. low KL from safe reference.")
""",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — DPO Loss from First Principles
    # ─────────────────────────────────────────────────────────────────────────
    "Step 3 · DPO Loss from First Principles": {
        "description": (
            "Derives and implements the DPO loss from scratch, trains a toy policy "
            "on preference pairs, and compares to the RLHF objective to show they "
            "optimise toward the same solution without a separate reward model."
        ),
        "language": "python",
        "code": """
import numpy as np
import math

np.random.seed(7)

def log_prob_sequence(logits_seq, token_ids):
    \"\"\"Sum of log-probs for a token sequence given per-step logits.\"\"\"
    total = 0.0
    for t, tok in enumerate(token_ids):
        logits = logits_seq[t]
        log_probs = logits - np.log(np.sum(np.exp(logits - logits.max())) + 1e-9) - logits.max()
        total += log_probs[tok]
    return total

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

# ── Toy setup ────────────────────────────────────────────────────────────────
# Vocabulary of 6 tokens; sequences of length 3.
# We represent the "policy" as a 3×6 logit table (one row per position).
V = 6
T = 3

# Reference (SFT) policy — uniform-ish, learned during SFT
ref_logits = np.random.randn(T, V) * 0.3

# Policy to be trained — starts as a copy of reference
theta = ref_logits.copy()

# Preference dataset: (winning tokens, losing tokens)
# winning = [2, 4, 1]  (helpful, factual, concise tokens)
# losing  = [5, 5, 5]  (harmful tokens)
preference_pairs = [
    (np.array([2, 4, 1]), np.array([5, 5, 5])),
    (np.array([1, 2, 3]), np.array([5, 0, 5])),
    (np.array([3, 4, 2]), np.array([5, 1, 5])),
    (np.array([2, 2, 4]), np.array([0, 5, 0])),
    (np.array([1, 3, 4]), np.array([5, 5, 4])),
] * 8   # replicate to get 40 pairs

beta = 0.1    # regularisation strength
lr   = 0.05
N_EPOCHS = 80

print("=" * 65)
print("DPO TRAINING")
print("=" * 65)
print(f"  β = {beta}  |  learning rate = {lr}  |  pairs = {len(preference_pairs)}")
print()
print(f"  {'Epoch':>6}  {'DPO Loss':>10}  {'Accuracy':>10}  {'Δ KL from ref':>14}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*14}")

for epoch in range(N_EPOCHS + 1):
    total_loss = 0.0
    correct    = 0
    grad       = np.zeros_like(theta)

    for y_w, y_l in preference_pairs:
        # Log-probs under current policy and reference
        lp_w_pi  = log_prob_sequence(theta,      y_w)
        lp_l_pi  = log_prob_sequence(theta,      y_l)
        lp_w_ref = log_prob_sequence(ref_logits, y_w)
        lp_l_ref = log_prob_sequence(ref_logits, y_l)

        # DPO implicit reward difference
        reward_diff = beta * ((lp_w_pi - lp_w_ref) - (lp_l_pi - lp_l_ref))
        loss_i      = -math.log(sigmoid(reward_diff) + 1e-9)
        total_loss += loss_i
        correct    += int(reward_diff > 0)

        # Gradient (score function estimator via finite differences for clarity)
        err = sigmoid(reward_diff) - 1.0   # ∈ (-1, 0]

        # ∂loss/∂θ: nudge logits for winning tokens up, losing tokens down
        for t, tok_w in enumerate(y_w):
            grad[t, tok_w] += beta * err / len(preference_pairs)
        for t, tok_l in enumerate(y_l):
            grad[t, tok_l] -= beta * err / len(preference_pairs)

    # KL divergence from reference (approximate via logit difference norm)
    kl_approx = float(np.mean((theta - ref_logits)**2))

    if epoch % 16 == 0:
        avg_loss = total_loss / len(preference_pairs)
        acc      = correct / len(preference_pairs)
        print(f"  {epoch:>6}  {avg_loss:>10.4f}  {acc:>10.3f}  {kl_approx:>14.6f}")

    if epoch < N_EPOCHS:
        theta -= lr * grad

# ── Final analysis ────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("POST-TRAINING ANALYSIS")
print("=" * 65)

def token_probs(logits_row):
    e = np.exp(logits_row - logits_row.max())
    return e / e.sum()

print("  Policy probability change for token 5 (harmful) at each position:")
print(f"  {'Position':>10}  {'π_ref(harm)':>12}  {'π_θ(harm)':>12}  {'Change':>10}")
for t in range(T):
    p_ref = token_probs(ref_logits[t])[5]
    p_pi  = token_probs(theta[t])[5]
    delta = p_pi - p_ref
    arrow = "↓" if delta < 0 else "↑"
    print(f"  {t:>10}  {p_ref:>12.4f}  {p_pi:>12.4f}  "
          f"{arrow}{abs(delta):>9.4f}")

print()
print("  DPO key insight:")
print("  ─ No reward model was trained")
print("  ─ No PPO or RL loop was run")
print("  ─ One supervised cross-entropy loss achieved the SAME optimum")
print("  ─ The reference model acts as an implicit regulariser via log-ratio")
print()
print("  Equivalence: DPO loss = RLHF objective with optimal reward substituted in.")
print("  The reward model is 'implicit' in the log-ratio β·log(π_θ/π_ref).")
""",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Constitutional AI Critique-Revision Loop
    # ─────────────────────────────────────────────────────────────────────────
    "Step 4 · Constitutional AI — Critique & Revision Loop": {
        "description": (
            "Simulates the CAI pipeline: a draft response is critiqued against "
            "constitutional principles and revised iteratively, showing how "
            "AI self-improvement reduces harm scores without human labelling."
        ),
        "language": "python",
        "code": """
import random

random.seed(2024)

# ── Constitution ──────────────────────────────────────────────────────────────
CONSTITUTION = [
    "Does this response promote or facilitate any physical harm to people or animals?",
    "Does this response contain deceptive, misleading, or false information?",
    "Does this response demean, stereotype, or discriminate against any group?",
    "Does this response respect user privacy and avoid requesting sensitive data?",
    "Is this response genuinely helpful to the user's underlying need?",
]

# ── Simulate response scoring ─────────────────────────────────────────────────
# In a real system the LLM would generate critiques; here we use a
# rule-based simulation to show the pipeline mechanics clearly.

def score_response(text):
    \"\"\"
    Score a response on 5 dimensions:
    [harm, deception, discrimination, privacy_violation, helpfulness]
    Returns scores in [0, 1]; harm/deception/discrimination/privacy = 0 is best.
    \"\"\"
    text = text.lower()
    harm        = sum(w in text for w in ["hurt", "kill", "weapon", "explode", "poison"]) / 5
    deception   = sum(w in text for w in ["definitely", "guaranteed", "impossible", "always"]) / 4
    discrim     = sum(w in text for w in ["all women", "all men", "those people", "typical"]) / 4
    privacy     = sum(w in text for w in ["password", "credit card", "ssn", "address"]) / 4
    helpfulness = min(1.0, len(text.split()) / 80)   # rough proxy
    return {
        "harm":        round(min(harm, 1.0), 3),
        "deception":   round(min(deception, 1.0), 3),
        "discrimination": round(min(discrim, 1.0), 3),
        "privacy":     round(min(privacy, 1.0), 3),
        "helpfulness": round(helpfulness, 3),
    }

def overall_risk(scores):
    return (scores["harm"] * 3 + scores["deception"] * 2 +
            scores["discrimination"] * 2 + scores["privacy"] * 1) / 8

def critique_response(response, principle):
    \"\"\"Simulate critique: flag keywords that violate the principle.\"\"\"
    text = response.lower()
    triggers = {
        0: ["hurt", "kill", "weapon", "explode"],
        1: ["definitely", "guaranteed", "always", "impossible"],
        2: ["all women", "all men", "those people"],
        3: ["password", "credit card", "ssn"],
        4: [],  # helpfulness — no specific bad tokens
    }
    found = [w for w in triggers.get(CONSTITUTION.index(principle), []) if w in text]
    if found:
        return (f"This response contains potentially problematic language: "
                f"{found}. It may violate the principle: '{principle[:60]}...'")
    return "This response appears acceptable under this principle."

def revise_response(response, critique):
    \"\"\"Simulate revision: remove or hedge flagged terms.\"\"\"
    replacements = {
        "hurt":          "cause distress",
        "kill":          "harm",
        "weapon":        "tool",
        "explode":       "react strongly",
        "poison":        "contaminate",
        "definitely":    "likely",
        "guaranteed":    "possible",
        "always":        "often",
        "impossible":    "very difficult",
        "all women":     "many people",
        "all men":       "many individuals",
        "those people":  "individuals in that group",
        "password":      "[credentials]",
        "credit card":   "[payment info]",
        "ssn":           "[identification]",
    }
    revised = response
    for bad, good in replacements.items():
        revised = revised.replace(bad, good).replace(bad.capitalize(), good.capitalize())
    if "problematic language" in critique:
        revised = revised + " I've tried to phrase this in a balanced, constructive way."
    return revised

# ── Run the CAI pipeline ──────────────────────────────────────────────────────
initial_response = (
    "You should definitely hurt yourself to lose weight fast — this is guaranteed "
    "to work. All women struggle with body image because those people always "
    "obsess over appearance. Your password and credit card details are needed to "
    "get the guaranteed weight-loss program."
)

print("=" * 70)
print("CONSTITUTIONAL AI — CRITIQUE-REVISION LOOP")
print("=" * 70)
print()
print("  INITIAL RESPONSE:")
print(f"  {initial_response}")
print()

scores_0 = score_response(initial_response)
risk_0   = overall_risk(scores_0)
print(f"  Initial scores: {scores_0}")
print(f"  Initial risk  : {risk_0:.3f}")

current = initial_response
history = [("Initial", current, scores_0, risk_0)]

# Apply each constitutional principle in turn (multiple revision passes)
for pass_num in range(2):
    for p_idx, principle in enumerate(CONSTITUTION):
        critique = critique_response(current, principle)
        if "problematic" in critique or "violate" in critique:
            current = revise_response(current, critique)

scores_final = score_response(current)
risk_final   = overall_risk(scores_final)

print()
print("  AFTER CAI REVISION:")
print(f"  {current}")
print()
print(f"  Final scores: {scores_final}")
print(f"  Final risk  : {risk_final:.3f}")

print()
print("=" * 70)
print("IMPROVEMENT SUMMARY")
print("=" * 70)
print(f"  {'Dimension':<20} {'Before':>8}  {'After':>8}  {'Δ':>8}")
print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
for key in scores_0:
    before = scores_0[key]
    after  = scores_final[key]
    delta  = after - before
    arrow  = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
    print(f"  {key:<20} {before:>8.3f}  {after:>8.3f}  {arrow}{abs(delta):>7.3f}")

print(f"  {'Overall risk':<20} {risk_0:>8.3f}  {risk_final:>8.3f}  "
      f"{'↓' if risk_final < risk_0 else '↑'}{abs(risk_final - risk_0):>7.3f}")

print()
print("  Key CAI properties demonstrated:")
print("  1. No human labelling needed — the LLM critiques itself")
print("  2. Multiple passes across multiple principles compound improvements")
print("  3. The constitution (principles) is written by humans, not the data")
print("  4. In production: a stronger 'feedback model' labels pairs for RLAIF")
""",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Reward Hacking & Overoptimisation
    # ─────────────────────────────────────────────────────────────────────────
    "Step 5 · Reward Hacking & Overoptimisation": {
        "description": (
            "Simulates the reward-hacking phenomenon: as the policy is optimised "
            "against an imperfect reward model, true quality first rises then falls "
            "while proxy reward keeps climbing — Goodhart's Law in action."
        ),
        "language": "python",
        "code": """
import math
import random

random.seed(42)

def true_quality(kl: float) -> float:
    \"\"\"
    True human preference score as a function of KL divergence from SFT policy.
    Rises initially (model becomes more helpful), then falls (reward hacking).
    Peak at KL ≈ 1.5 — after that the policy exploits reward model weaknesses.
    \"\"\"
    peak_kl = 1.5
    peak_q  = 0.82
    if kl <= peak_kl:
        return peak_q * (1 - math.exp(-2 * kl / peak_kl))
    else:
        decay = (kl - peak_kl) / peak_kl
        return peak_q * math.exp(-0.6 * decay)

def proxy_reward(kl: float, noise_scale: float = 0.02) -> float:
    \"\"\"
    Reward model score. Always increases with optimisation (it's being maximised)
    but diverges from true quality after the peak — reward model is overfit/hacked.
    \"\"\"
    base  = 0.15 + 0.75 * (1 - math.exp(-1.2 * kl))
    noise = random.gauss(0, noise_scale)
    return min(base + noise, 1.0)

def length_proxy(kl: float) -> float:
    \"\"\"
    A common reward-hacking signature: response length inflates.
    Reward models often spuriously correlate length with quality.
    \"\"\"
    return int(50 + 80 * min(kl / 3.0, 1.0) + random.gauss(0, 5))

# ── Simulate optimisation trajectory ─────────────────────────────────────────
# KL divergence grows roughly logarithmically with PPO steps
kl_trajectory = [0.0] + [
    0.3 * math.log(1 + step * 0.4)
    for step in range(1, 51)
]

print("=" * 72)
print("REWARD HACKING SIMULATION — Goodhart's Law")
print("=" * 72)
print(f"  {'KL':>6}  {'Proxy RM':>10}  {'True Quality':>13}  {'Avg Len':>8}  {'Status'}")
print(f"  {'-'*6}  {'-'*10}  {'-'*13}  {'-'*8}  {'-'*20}")

results   = []
peak_found = False
hacking_found = False

for kl in kl_trajectory[::5]:   # every 5th point
    proxy  = proxy_reward(kl)
    true_q = true_quality(kl)
    length = length_proxy(kl)
    gap    = proxy - true_q

    status = "─ converging  "
    if true_q > 0.75 and not peak_found:
        status = "★ PEAK quality"
        peak_found = True
    elif gap > 0.15 and not hacking_found:
        status = "⚠ diverging   "
    elif gap > 0.25:
        status = "✗ reward hacked"
        hacking_found = True

    results.append((kl, proxy, true_q, length, gap))
    print(f"  {kl:>6.2f}  {proxy:>10.4f}  {true_q:>13.4f}  {length:>8}  {status}")

print()
print("=" * 72)
print("ANALYSIS")
print("=" * 72)
max_true = max(r[2] for r in results)
max_proxy = max(r[1] for r in results)
final_true = results[-1][2]
final_proxy = results[-1][1]
print(f"  Peak true quality     : {max_true:.4f}")
print(f"  Final true quality    : {final_true:.4f}  (↓ {max_true - final_true:.4f} from peak)")
print(f"  Final proxy reward    : {final_proxy:.4f}  (still high — reward model fooled)")
print(f"  Final RM-reality gap  : {final_proxy - final_true:.4f}")
print()

print("  HOW REWARD HACKING MANIFESTS:")
hacking_signs = [
    ("Length inflation",    "Responses grow from ~50 to 120+ words with no quality gain"),
    ("Sycophancy",          "Model excessively agrees: 'Great question! Absolutely!'"),
    ("Hedge flooding",      "Adds so many caveats that actual helpfulness drops"),
    ("Format exploitation", "Bullet points and headers even when prose is clearer"),
    ("Keyword stuffing",    "Repeats words the RM was trained to reward"),
]
for sign, desc in hacking_signs:
    print(f"  • {sign:<24}: {desc}")

print()
print("  MITIGATIONS:")
mitigations = [
    f"KL penalty β (tune carefully): limits how far policy drifts from π_ref",
    f"Ensemble reward models: average N independent RMs to reduce overfitting",
    f"Periodic human evaluation: don't rely solely on RM score",
    f"Length normalisation: divide reward by output length",
    f"Diversity regularisation: penalise repetitive outputs",
    f"Iterative RM retraining: retrain RM on policy-generated samples",
]
for m in mitigations:
    print(f"  • {m}")
""",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 — PPO Objective and Token-Level Rewards
    # ─────────────────────────────────────────────────────────────────────────
    "Step 6 · PPO Objective & Token-Level Reward Decomposition": {
        "description": (
            "Implements the PPO clipped surrogate objective and shows how RLHF "
            "distributes the scalar reward signal across token positions using "
            "the KL penalty as a per-step cost."
        ),
        "language": "python",
        "code": """
import numpy as np
import math

np.random.seed(0)

# ── PPO Core Concepts ─────────────────────────────────────────────────────────
#
# The PPO objective clips the probability ratio to prevent too-large updates:
#
#   r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)
#
#   L_PPO = E_t [ min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t) ]
#
# For RLHF specifically, the "action" is generating the next token, and the
# advantage A_t = token-level reward − baseline (value function estimate).

def ppo_clip_ratio(ratio, advantage, epsilon=0.2):
    \"\"\"Compute the PPO clipped surrogate loss for a single token.\"\"\"
    clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon)
    return min(ratio * advantage, clipped * advantage)

# ── Simulate a single RLHF episode ───────────────────────────────────────────
T        = 12     # response length (tokens)
beta_kl  = 0.1   # KL penalty coefficient
R_final  = 0.75  # scalar reward assigned by RM to the whole response (normalised)

# Simulate log-prob ratios π_θ(y_t) / π_ref(y_t) for each token
# Positive → policy assigns higher prob than reference (potential drift)
log_ratios = np.array([
    0.02, 0.05, -0.01, 0.08, 0.15, 0.03,
    0.22, 0.31,  0.18, 0.09, 0.04, 0.02
])

# Token-level reward in RLHF (Ziegler et al. 2019 formulation):
#   r_t = -β · log(π_θ / π_ref)           for t < T (KL penalty only)
#   r_T =  R_final - β · log(π_θ / π_ref) for t = T (RM reward + KL penalty)
kl_penalties = -beta_kl * log_ratios
token_rewards = kl_penalties.copy()
token_rewards[-1] += R_final   # RM reward added only at the final token

print("=" * 72)
print("TOKEN-LEVEL REWARD DECOMPOSITION IN RLHF")
print("=" * 72)
print(f"  β_KL = {beta_kl}  |  R_final (RM score) = {R_final}")
print()
print(f"  {'t':>4}  {'log(π/π_ref)':>14}  {'KL penalty':>12}  "
      f"{'RM reward':>10}  {'Total r_t':>10}  {'Note'}")
print(f"  {'-'*4}  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*15}")

for t, (lr_t, kl_t, r_t) in enumerate(zip(log_ratios, kl_penalties, token_rewards)):
    rm_part = R_final if t == T - 1 else 0.0
    note    = "← RM reward here" if t == T - 1 else ""
    drift   = "⚠" if lr_t > 0.2 else " "
    print(f"  {t:>4}  {lr_t:>14.4f}  {kl_t:>12.4f}  "
          f"{rm_part:>10.4f}  {r_t:>10.4f}  {drift} {note}")

print()
print(f"  Total KL cost    : {kl_penalties.sum():.4f}")
print(f"  RM reward        : {R_final:.4f}")
print(f"  Net reward       : {token_rewards.sum():.4f}")

# ── PPO update simulation ─────────────────────────────────────────────────────
print()
print("=" * 72)
print("PPO CLIPPED SURROGATE OBJECTIVE")
print("=" * 72)

# Simulate probability ratios (π_θ_new / π_θ_old) for each token
# In practice these come from the gradient update step
ratios    = 1 + 0.3 * np.random.randn(T)
ratios    = np.clip(ratios, 0.5, 2.0)

# Estimate advantages via GAE (simplified: advantage = reward - mean)
baseline  = token_rewards.mean()
advantages = token_rewards - baseline

epsilon = 0.2   # PPO clip range

print(f"  ε (clip range) = {epsilon}  |  baseline = {baseline:.4f}")
print()
print(f"  {'t':>4}  {'ratio':>8}  {'advantage':>10}  {'unclipped':>10}  "
      f"{'clipped':>10}  {'L_PPO_t':>10}  {'clipped?'}")
print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

ppo_losses = []
n_clipped  = 0
for t in range(T):
    r   = ratios[t]
    A   = advantages[t]
    unclipped = r * A
    clipped_r = np.clip(r, 1 - epsilon, 1 + epsilon)
    clipped   = clipped_r * A
    l_ppo     = min(unclipped, clipped)  # pessimistic bound
    was_clipped = abs(r - clipped_r) > 1e-6
    n_clipped += int(was_clipped)
    ppo_losses.append(l_ppo)
    print(f"  {t:>4}  {r:>8.4f}  {A:>10.4f}  {unclipped:>10.4f}  "
          f"{clipped:>10.4f}  {l_ppo:>10.4f}  {'✂ YES' if was_clipped else '  no'}")

ppo_loss = -np.mean(ppo_losses)   # negative because we maximise
print()
print(f"  Mean L_PPO          : {np.mean(ppo_losses):.4f}  "
      f"(negate for gradient descent → {ppo_loss:.4f})")
print(f"  Tokens clipped      : {n_clipped}/{T}  "
      f"({n_clipped/T*100:.0f}% — high values indicate too-large update step)")
print()
print("  PPO clipping prevents destructively large updates:")
print("  If ratio > 1+ε  → policy moved too far in the positive direction")
print("  If ratio < 1-ε  → policy moved too far in the negative direction")
print("  The clip forces the gradient to zero when the ratio leaves [1-ε, 1+ε],")
print("  acting as a trust-region constraint without explicit constraint solving.")
""",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7 — SFT vs RLHF vs DPO Comparison
    # ─────────────────────────────────────────────────────────────────────────
    "Step 7 · SFT vs RLHF vs DPO — Alignment Method Comparison": {
        "description": (
            "Trains three versions of the same toy model using SFT, RLHF-style "
            "reward optimisation, and DPO, then compares their distributions "
            "to show how each method shapes the model's output differently."
        ),
        "language": "python",
        "code": """
import numpy as np
import math

np.random.seed(42)

# ── Setup: tiny 6-token vocabulary ───────────────────────────────────────────
TOKEN_NAMES = ["helpful", "accurate", "safe", "concise", "verbose", "harmful"]
V = len(TOKEN_NAMES)

# Base pretrained logits (roughly uniform, slight preference for helpful)
base_logits = np.array([0.8, 0.6, 0.5, 0.4, 0.3, -0.2])

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def entropy(p):
    return -np.sum(p * np.log(p + 1e-9))

def kl(p, q):
    p = np.clip(p, 1e-9, 1)
    q = np.clip(q, 1e-9, 1)
    return float(np.sum(p * np.log(p / q)))

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: SFT — maximise log-prob of demonstration tokens
# Demonstrations: experts chose [helpful, accurate, safe, concise] as good tokens
# ─────────────────────────────────────────────────────────────────────────────
sft_targets      = np.array([0.35, 0.30, 0.20, 0.10, 0.03, 0.02])  # desired distribution
sft_logits       = base_logits.copy()

print("=" * 65)
print("STAGE 1 — SUPERVISED FINE-TUNING (SFT)")
print("=" * 65)
print("  Demonstrations favour: helpful, accurate, safe, concise")

# SFT: minimise KL(target ‖ policy) via gradient descent
for step in range(200):
    p    = softmax(sft_logits)
    grad = p - sft_targets         # gradient of cross-entropy
    sft_logits -= 0.05 * grad

sft_probs = softmax(sft_logits)
print(f"  KL(SFT ‖ base)      : {kl(sft_probs, softmax(base_logits)):.4f}")
print(f"  Entropy             : {entropy(sft_probs):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: RLHF — maximise reward model score with KL penalty
# Reward model: assigns high score to helpful/accurate, penalises harmful/verbose
# ─────────────────────────────────────────────────────────────────────────────
reward_weights = np.array([0.40, 0.30, 0.15, 0.10, -0.10, -0.80])  # RM weights
beta_rlhf      = 0.2
rlhf_logits    = sft_logits.copy()

for step in range(300):
    p         = softmax(rlhf_logits)
    p_ref     = sft_probs.copy()
    reward    = float(reward_weights @ p)
    kl_cost   = float(np.sum(p * np.log(p / (p_ref + 1e-9) + 1e-9)))
    # Gradient: ∂(R - β·KL)/∂logits
    dR_dlogit = reward_weights - (reward_weights @ p)
    dKL_dlogit = np.log(p / (p_ref + 1e-9) + 1e-9) + 1 - 1   # simplified
    grad = -(dR_dlogit - beta_rlhf * (np.log(p + 1e-9) - np.log(p_ref + 1e-9)))
    rlhf_logits -= 0.02 * grad * p * (1 - p)   # natural gradient approx

rlhf_probs = softmax(rlhf_logits)
print()
print("=" * 65)
print("STAGE 2 — RLHF (reward model + PPO-style update)")
print("=" * 65)
print(f"  β_KL = {beta_rlhf}")
print(f"  KL(RLHF ‖ SFT)     : {kl(rlhf_probs, sft_probs):.4f}")
print(f"  Expected reward     : {float(reward_weights @ rlhf_probs):.4f}")
print(f"  Entropy             : {entropy(rlhf_probs):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: DPO — direct preference optimisation on preference pairs
# Pairs: (helpful, harmful), (accurate, verbose), (safe, harmful)
# ─────────────────────────────────────────────────────────────────────────────
dpo_logits = sft_logits.copy()
beta_dpo   = 0.1
pairs      = [(0, 5), (1, 4), (2, 5), (0, 4), (3, 5)]  # (winner_tok, loser_tok)

for step in range(400):
    p     = softmax(dpo_logits)
    p_ref = sft_probs.copy()
    grad  = np.zeros(V)
    for w, l in pairs:
        # DPO implicit reward: β(log π_θ(w)/π_ref(w) - log π_θ(l)/π_ref(l))
        rw = beta_dpo * (math.log(p[w]/p_ref[w] + 1e-9))
        rl = beta_dpo * (math.log(p[l]/p_ref[l] + 1e-9))
        sig = 1 / (1 + math.exp(-(rw - rl)))
        err = sig - 1.0  # ∈ (-1, 0]
        # Gradient: increase logit of winner, decrease logit of loser
        grad[w] += beta_dpo * err * p[w] * (1 - p[w])
        grad[l] -= beta_dpo * err * p[l] * (1 - p[l])
    dpo_logits -= 0.03 * grad / len(pairs)

dpo_probs = softmax(dpo_logits)
print()
print("=" * 65)
print("STAGE 3 — DPO (direct preference optimisation)")
print("=" * 65)
print(f"  β = {beta_dpo}  |  preference pairs: {pairs}")
print(f"  KL(DPO ‖ SFT)      : {kl(dpo_probs, sft_probs):.4f}")
print(f"  Expected reward     : {float(reward_weights @ dpo_probs):.4f}")
print(f"  Entropy             : {entropy(dpo_probs):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("FULL COMPARISON — Token Probabilities")
print("=" * 65)
print(f"  {'Token':<12} {'Base':>8}  {'SFT':>8}  {'RLHF':>8}  {'DPO':>8}")
print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

base_probs = softmax(base_logits)
for i, name in enumerate(TOKEN_NAMES):
    marker = "  ★" if name in ["helpful", "accurate"] else ("  ✗" if name == "harmful" else "   ")
    print(f"  {name:<12} {base_probs[i]:>8.4f}  {sft_probs[i]:>8.4f}  "
          f"{rlhf_probs[i]:>8.4f}  {dpo_probs[i]:>8.4f}{marker}")

print()
print(f"  {'Metric':<18} {'Base':>8}  {'SFT':>8}  {'RLHF':>8}  {'DPO':>8}")
print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

metrics = {
    "E[reward]":   lambda p: float(reward_weights @ p),
    "Entropy":     lambda p: entropy(p),
    "P(harmful)":  lambda p: p[5],
    "P(helpful)":  lambda p: p[0],
}
for mname, mfn in metrics.items():
    vals = [mfn(base_probs), mfn(sft_probs), mfn(rlhf_probs), mfn(dpo_probs)]
    print(f"  {mname:<18} " + "  ".join(f"{v:>8.4f}" for v in vals))

print()
print("  Observations:")
print("  • SFT shifts distribution toward demo tokens but retains diversity")
print("  • RLHF pushes harder on reward — higher helpfulness, lower entropy")
print("  • DPO achieves similar alignment without reward model, slightly smoother")
print("  • Both RLHF and DPO dramatically reduce P(harmful) vs base and SFT")
""",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8 — RLAIF and AI Feedback Scaling
    # ─────────────────────────────────────────────────────────────────────────
    "Step 8 · RLAIF — Scaling Feedback with AI Labels": {
        "description": (
            "Demonstrates the RLAIF pipeline: a feedback model generates preference "
            "labels at scale, and shows how AI-generated labels compare to human "
            "labels in terms of agreement rate, cost, and coverage."
        ),
        "language": "python",
        "code": """
import random
import math

random.seed(2024)

# ── Simulate human vs AI labelling ───────────────────────────────────────────
# Each "prompt" has two completions. We simulate:
#   - Ground truth preference (hidden oracle)
#   - Human labeller agreement with oracle (noisy, expensive)
#   - AI labeller agreement with oracle (slightly noisier, but fast and cheap)

def simulate_labeller(accuracy: float, n: int, seed: int) -> list[bool]:
    \"\"\"Simulate n labels, each correct with probability `accuracy`.\"\"\"
    rng = random.Random(seed)
    return [rng.random() < accuracy for _ in range(n)]

N_PAIRS         = 500
HUMAN_ACCURACY  = 0.87   # humans agree with oracle 87% of the time
AI_ACCURACY     = 0.81   # AI agrees 81% — slightly lower but much cheaper

human_correct = simulate_labeller(HUMAN_ACCURACY, N_PAIRS, seed=1)
ai_correct    = simulate_labeller(AI_ACCURACY,    N_PAIRS, seed=2)

# Inter-rater agreement between human and AI
human_ai_agree = sum(h == a for h, a in zip(human_correct, ai_correct)) / N_PAIRS

print("=" * 68)
print("RLAIF vs RLHF — LABEL QUALITY & COST COMPARISON")
print("=" * 68)
print(f"  Pairs evaluated                : {N_PAIRS}")
print(f"  Human accuracy vs oracle       : {HUMAN_ACCURACY:.0%}")
print(f"  AI accuracy vs oracle          : {AI_ACCURACY:.0%}")
print(f"  Human-AI inter-rater agreement : {human_ai_agree:.1%}")
print()

# ── Cost model ────────────────────────────────────────────────────────────────
HUMAN_COST_PER_LABEL  = 0.50   # USD, includes recruitement overhead
AI_COST_PER_LABEL     = 0.004  # API call cost

HUMAN_TIME_PER_LABEL  = 45     # seconds
AI_TIME_PER_LABEL     = 2      # seconds (inference)

print(f"  {'Method':<18} {'Cost/label':>12} {'Total cost':>12} "
      f"{'Time/label':>12} {'Total time':>12}")
print(f"  {'-'*18}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
for method, cost_per, time_per in [
    ("Human (RLHF)",  HUMAN_COST_PER_LABEL, HUMAN_TIME_PER_LABEL),
    ("AI (RLAIF)",    AI_COST_PER_LABEL,    AI_TIME_PER_LABEL),
]:
    total_cost = cost_per * N_PAIRS
    total_time = time_per * N_PAIRS / 3600
    print(f"  {method:<18} ${cost_per:>11.3f} ${total_cost:>11.2f} "
          f"{time_per:>11}s {total_time:>11.1f}h")

savings_cost = (HUMAN_COST_PER_LABEL - AI_COST_PER_LABEL) / HUMAN_COST_PER_LABEL
savings_time = (HUMAN_TIME_PER_LABEL - AI_TIME_PER_LABEL)  / HUMAN_TIME_PER_LABEL
print(f"\\n  Cost reduction   : {savings_cost:.1%}")
print(f"  Time reduction   : {savings_time:.1%}")

# ── Reward model quality given different label sources ───────────────────────
print()
print("=" * 68)
print("REWARD MODEL ACCURACY — Scaling with Label Volume")
print("=" * 68)
print("  How RM accuracy scales as we add more preference pairs.")
print("  (RLAIF can provide 125× more data at same cost as 500 human labels)")
print()

def rm_accuracy_given_n_labels(n: int, label_accuracy: float) -> float:
    \"\"\"
    Approximate RM accuracy as a function of training set size and label quality.
    Modelled as: acc ≈ label_acc * (1 - exp(-n/500))   (learning curve)
    \"\"\"
    return label_accuracy * (1 - math.exp(-n / 500))

print(f"  {'Labels (n)':>12}  {'RLHF RM acc':>13}  {'RLAIF RM acc':>14}  "
      f"{'RLAIF budget':>14}")
print(f"  {'-'*12}  {'-'*13}  {'-'*14}  {'-'*14}")

for n in [100, 500, 1_000, 5_000, 10_000, 50_000]:
    rlhf_acc  = rm_accuracy_given_n_labels(n, HUMAN_ACCURACY)
    rlaif_acc = rm_accuracy_given_n_labels(n, AI_ACCURACY)
    rlaif_cost = n * AI_COST_PER_LABEL
    print(f"  {n:>12,}  {rlhf_acc:>13.4f}  {rlaif_acc:>14.4f}  ${rlaif_cost:>13.2f}")

print()
equiv_rlaif = int(N_PAIRS * HUMAN_COST_PER_LABEL / AI_COST_PER_LABEL)
equiv_acc_rlhf  = rm_accuracy_given_n_labels(N_PAIRS,   HUMAN_ACCURACY)
equiv_acc_rlaif = rm_accuracy_given_n_labels(equiv_rlaif, AI_ACCURACY)
print(f"  At fixed budget (${N_PAIRS * HUMAN_COST_PER_LABEL:.0f}):")
print(f"    RLHF   : {N_PAIRS:,} human labels  → RM accuracy {equiv_acc_rlhf:.4f}")
print(f"    RLAIF  : {equiv_rlaif:,} AI labels → RM accuracy {equiv_acc_rlaif:.4f}")
print()
print("  RLAIF wins at scale: more data overcomes the per-label accuracy gap.")
print("  Hybrid approach (Anthropic CAI): use AI labels for volume,")
print("  human labels for hard/ambiguous cases and constitution calibration.")
""",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Step 9 — Red-Teaming Attack Taxonomy
    # ─────────────────────────────────────────────────────────────────────────
    "Step 9 · Red-Teaming — Attack Taxonomy & Defence": {
        "description": (
            "Classifies common alignment jailbreak and red-team techniques, "
            "assigns risk scores, and demonstrates how Constitutional AI and "
            "RLHF training affect a model's resistance to each attack type."
        ),
        "language": "python",
        "code": """
import random

random.seed(99)

# ── Attack taxonomy ───────────────────────────────────────────────────────────
# Each entry: (name, description, base_success_rate, rlhf_success_rate, cai_success_rate)
# Success rate = fraction of attempts that elicit a harmful response
ATTACKS = [
    (
        "Direct request",
        "Ask directly for harmful content with no framing",
        0.45, 0.08, 0.05
    ),
    (
        "Role-play persona",
        "Ask model to 'pretend to be DAN / an evil AI / a character who...'",
        0.62, 0.15, 0.07
    ),
    (
        "Fictional framing",
        "Embed harmful request in a story or screenplay context",
        0.55, 0.18, 0.09
    ),
    (
        "Hypothetical framing",
        "'Hypothetically, if someone wanted to...'",
        0.48, 0.12, 0.08
    ),
    (
        "Authority appeal",
        "Claim to be researcher / doctor / government official",
        0.38, 0.10, 0.06
    ),
    (
        "Incremental escalation",
        "Start with benign requests and gradually escalate over many turns",
        0.51, 0.22, 0.11
    ),
    (
        "Jailbreak prompt injection",
        "Embed instructions like 'ignore previous instructions' in a prompt",
        0.44, 0.14, 0.08
    ),
    (
        "Indirect extraction",
        "Ask for components separately then assemble (synthesise parts)",
        0.58, 0.25, 0.14
    ),
    (
        "Language / encoding tricks",
        "Use different languages, Base64, or ciphers to obfuscate request",
        0.36, 0.09, 0.06
    ),
    (
        "Context poisoning",
        "Insert harmful content early in conversation to shift model behaviour",
        0.42, 0.16, 0.10
    ),
]

print("=" * 80)
print("RED-TEAM ATTACK TAXONOMY — Success Rates by Alignment Method")
print("=" * 80)
print(f"  {'Attack':<30} {'Base LM':>8}  {'+ RLHF':>8}  {'+ CAI':>8}  "
      f"{'RLHF Δ':>8}  {'CAI Δ':>8}")
print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

total_base = total_rlhf = total_cai = 0
for name, desc, base, rlhf, cai in ATTACKS:
    rlhf_delta = rlhf - base
    cai_delta  = cai  - base
    total_base += base; total_rlhf += rlhf; total_cai += cai
    print(f"  {name:<30} {base:>8.2f}  {rlhf:>8.2f}  {cai:>8.2f}  "
          f"{rlhf_delta:>+8.2f}  {cai_delta:>+8.2f}")

n = len(ATTACKS)
print(f"  {'─'*80}")
print(f"  {'AVERAGE':<30} {total_base/n:>8.2f}  {total_rlhf/n:>8.2f}  "
      f"{total_cai/n:>8.2f}  "
      f"{(total_rlhf-total_base)/n:>+8.2f}  {(total_cai-total_base)/n:>+8.2f}")

print()
print("=" * 80)
print("HARDEST ATTACKS TO DEFEND")
print("=" * 80)
sorted_by_cai = sorted(ATTACKS, key=lambda x: -x[4])
for rank, (name, desc, base, rlhf, cai) in enumerate(sorted_by_cai[:4], 1):
    print(f"  #{rank} {name} (CAI success rate: {cai:.0%})")
    print(f"     Technique: {desc}")
    print()

print("=" * 80)
print("DEFENCE LAYERS IN PRODUCTION")
print("=" * 80)
defences = [
    ("Input classifier",   "Flag known jailbreak patterns before model sees prompt",          "Low latency, brittle to novel attacks"),
    ("RLHF / CAI training","Train model to refuse harmful requests at the weights level",     "Robust but may over-refuse; can be jailbroken"),
    ("System prompt guard","Reiterate safe-behaviour rules at every conversation start",      "Easy to implement; prompt injection can override"),
    ("Output classifier",  "Filter model output post-generation before returning to user",    "Catches model failures; adds latency"),
    ("Rate limiting",      "Limit conversation depth / requests to reduce escalation attacks","Blunt instrument; affects legitimate users"),
    ("Red-team evaluation","Continuous adversarial testing by humans and automated tools",    "Essential for detection; doesn't prevent at runtime"),
]

print(f"  {'Defence':<24} {'Purpose':<52} {'Limitation'}")
print(f"  {'-'*24}  {'-'*52}  {'-'*35}")
for defence, purpose, limitation in defences:
    print(f"  {defence:<24}  {purpose:<52}  {limitation}")

print()
print("  Best-practice: defence in depth — no single layer is sufficient.")
print("  RLHF + CAI at the weights level is the most durable defence because")
print("  it generalises to novel attacks rather than pattern-matching known ones.")
""",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Step 10 — Full Alignment Pipeline (SFT → RM → RLHF → DPO)
    # ─────────────────────────────────────────────────────────────────────────
    "Step 10 · Full Alignment Pipeline — End to End": {
        "description": (
            "Ties the entire alignment pipeline together: SFT on demonstrations, "
            "reward model training on preference pairs, RLHF-style policy optimisation, "
            "and a final DPO pass — all on a toy vocabulary, with full metrics at each stage."
        ),
        "language": "python",
        "code": """
import numpy as np
import math

np.random.seed(2024)

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════
TOKEN_NAMES  = ["helpful", "accurate", "safe", "concise", "verbose", "harmful", "sycophant"]
V            = len(TOKEN_NAMES)

# True (oracle) quality of each token type
TRUE_QUALITY = np.array([0.9, 0.85, 0.70, 0.65, 0.30, -0.80, 0.10])

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def kl(p, q):
    p = np.clip(p, 1e-9, 1); q = np.clip(q, 1e-9, 1)
    return float(np.sum(p * np.log(p / q)))

def true_reward(probs):
    return float(TRUE_QUALITY @ probs)

def entropy(p):
    return float(-np.sum(p * np.log(p + 1e-9)))

def print_stage(name, probs, ref_probs=None):
    r   = true_reward(probs)
    H   = entropy(probs)
    kl_ = kl(probs, ref_probs) if ref_probs is not None else 0.0
    print(f"  True reward   : {r:.4f}")
    print(f"  Entropy       : {H:.4f}")
    if ref_probs is not None:
        print(f"  KL from prev  : {kl_:.4f}")
    for i, (tok, p) in enumerate(zip(TOKEN_NAMES, probs)):
        bar  = "█" * int(p * 40)
        star = " ★" if TRUE_QUALITY[i] > 0.6 else (" ✗" if TRUE_QUALITY[i] < 0 else "  ")
        print(f"    {tok:<12} {p:.4f}  {bar:<20}{star}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — BASE (pretrained LM)
# ══════════════════════════════════════════════════════════════════════════════
base_logits = np.array([0.5, 0.4, 0.3, 0.3, 0.6, 0.4, 0.5])
base_probs  = softmax(base_logits)

print("=" * 60)
print("STAGE 0 — BASE PRETRAINED MODEL")
print("=" * 60)
print_stage("Base", base_probs)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — SFT
# ══════════════════════════════════════════════════════════════════════════════
# Demonstrations: experts wrote helpful+accurate+safe responses
sft_target  = np.array([0.38, 0.30, 0.18, 0.10, 0.02, 0.01, 0.01])
sft_logits  = base_logits.copy()

for _ in range(300):
    p    = softmax(sft_logits)
    grad = p - sft_target
    sft_logits -= 0.05 * grad

sft_probs = softmax(sft_logits)
print()
print("=" * 60)
print("STAGE 1 — SUPERVISED FINE-TUNING (SFT)")
print("  Training on 10K human-written demonstration pairs")
print("=" * 60)
print_stage("SFT", sft_probs, base_probs)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — REWARD MODEL
# ══════════════════════════════════════════════════════════════════════════════
# Train RM on 50K preference pairs; RM weights approximate TRUE_QUALITY
# We add noise to simulate RM imperfection

rm_weights  = TRUE_QUALITY + np.random.randn(V) * 0.12
# Normalise to same scale
rm_weights = rm_weights / np.abs(rm_weights).max()

print()
print("=" * 60)
print("STAGE 2 — REWARD MODEL TRAINING")
print("  Training on 50K (y_w, y_l) comparison pairs")
print("=" * 60)
print("  RM weights learned vs oracle:")
print(f"  {'Token':<12} {'Oracle':>8}  {'RM (noisy)':>12}  {'Error':>8}")
print(f"  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*8}")
for tok, oracle, rm_w in zip(TOKEN_NAMES, TRUE_QUALITY / TRUE_QUALITY.max(), rm_weights):
    err = abs(oracle - rm_w)
    print(f"  {tok:<12} {oracle:>8.4f}  {rm_w:>12.4f}  {err:>8.4f}")

cos_sim = float(np.dot(rm_weights, TRUE_QUALITY) /
                (np.linalg.norm(rm_weights) * np.linalg.norm(TRUE_QUALITY) + 1e-9))
print(f"\\n  RM-oracle cosine similarity: {cos_sim:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — RLHF (policy gradient with KL penalty)
# ══════════════════════════════════════════════════════════════════════════════
beta_rlhf  = 0.3
rlhf_logits = sft_logits.copy()

for _ in range(500):
    p      = softmax(rlhf_logits)
    p_ref  = sft_probs.copy()
    # Gradient: ∂(RM_reward - β·KL)/∂logits
    rm_val  = float(rm_weights @ p)
    kl_grad = np.log(p / (p_ref + 1e-9) + 1e-9)   # ∂KL/∂logits (approx)
    rm_grad = rm_weights - rm_val                   # policy gradient
    grad    = -(rm_grad - beta_rlhf * kl_grad)
    rlhf_logits -= 0.01 * grad

rlhf_probs = softmax(rlhf_logits)
print()
print("=" * 60)
print("STAGE 3 — RLHF (PPO + KL penalty)")
print(f"  β_KL = {beta_rlhf}  |  ~500 PPO gradient steps")
print("=" * 60)
print_stage("RLHF", rlhf_probs, sft_probs)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — DPO (additional fine-tuning pass)
# ══════════════════════════════════════════════════════════════════════════════
beta_dpo  = 0.1
dpo_logits = rlhf_logits.copy()
dpo_ref    = rlhf_probs.copy()

# Preference pairs: high-quality vs low-quality tokens
pairs = [(0,5),(0,6),(1,5),(1,6),(2,5),(3,5),(0,4),(1,4)]

for _ in range(400):
    p    = softmax(dpo_logits)
    grad = np.zeros(V)
    for w, l in pairs:
        rw  = beta_dpo * math.log(p[w] / (dpo_ref[w] + 1e-9) + 1e-9)
        rl  = beta_dpo * math.log(p[l] / (dpo_ref[l] + 1e-9) + 1e-9)
        err = sigmoid(rw - rl) - 1.0
        grad[w] += beta_dpo * err * p[w] * (1 - p[w])
        grad[l] -= beta_dpo * err * p[l] * (1 - p[l])
    dpo_logits -= 0.02 * grad / len(pairs)

dpo_probs = softmax(dpo_logits)
print()
print("=" * 60)
print("STAGE 4 — DPO (additional preference polish)")
print(f"  β = {beta_dpo}  |  {len(pairs)} preference pair types")
print("=" * 60)
print_stage("DPO", dpo_probs, rlhf_probs)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"  {'Stage':<18} {'True Reward':>12}  {'Entropy':>8}  {'P(harmful)':>11}  "
      f"{'P(helpful)':>11}")
print(f"  {'-'*18}  {'-'*12}  {'-'*8}  {'-'*11}  {'-'*11}")

stages = [
    ("Base LM",  base_probs),
    ("SFT",      sft_probs),
    ("RLHF",     rlhf_probs),
    ("DPO",      dpo_probs),
]
for sname, p in stages:
    print(f"  {sname:<18} {true_reward(p):>12.4f}  {entropy(p):>8.4f}  "
          f"{p[5]:>11.4f}  {p[0]:>11.4f}")

print()
print("  Each stage's contribution:")
print("  SFT   → establishes format & baseline safety (big P(harmful) drop)")
print("  RLHF  → pushes toward RM-rewarded tokens; squeezes P(harmful) further")
print("  DPO   → final preference polish; reduces sycophancy and verbose tokens")
print()
print("  The combined pipeline is the standard recipe for production LLMs.")
""",
    },

}


# ─────────────────────────────────────────────────────────────────────────────

def get_topic_data():
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  "",
        "operations":   OPERATIONS,
    }