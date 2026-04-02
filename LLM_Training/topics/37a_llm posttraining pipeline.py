"""
LLM Post-Training Pipeline — SFT, Reward Modeling, RLHF, DPO, and RLVR
=======================================================================

Pre-training gives a model raw language capability: it can predict the
next token over trillions of examples. But a pre-trained model is NOT
an assistant. It completes text in the style of its training corpus —
which includes forum spam, toxic content, and hallucinated facts.

Post-training is the process of SHAPING that raw capability into a
model that follows instructions, is helpful, is honest, and is safe.
It is a sequence of supervised and reinforcement learning stages, each
building on the last, with the goal of aligning the model's behaviour
to human intent.

The pipeline involves several interacting components:
    SFT (Supervised Fine-Tuning):          teach the format of good responses
    Reward Modeling (RM):                  compress human judgment into a scalar
    RLHF / PPO:                            optimise for that scalar signal
    DPO / IPO / KTO:                       direct preference optimisation (no RM)
    RLVR:                                  RL with verifiable rewards (reasoning)
    Constitutional AI / RLAIF:             AI feedback instead of human labels
    Iterative refinement:                  loop the pipeline on live model outputs
    Agentic fine-tuning:                   teach tool use and multi-step planning

Each stage has distinct failure modes. Understanding them is essential
for anyone working on or reasoning about frontier model development.

This module covers the complete post-training stack:
    The SFT data pipeline: formats, quality filtering, diversity
    Reward model architecture and the Bradley-Terry preference model
    PPO training loop adapted for LLMs: the KL penalty, clipping, and value head
    DPO derivation: why you don't need a reward model at all
    RLVR: how reasoning models (o1, DeepSeek-R1) are trained with ground truth
    Constitutional AI: replacing humans with model-generated critiques
    Safety fine-tuning: harmlessness as a separate preference objective
    Agentic fine-tuning: tool calls, multi-step trajectories, and sparse rewards

"""

import textwrap
import re

TOPIC_NAME   = "LLM Post-Training Pipeline"
DISPLAY_NAME = "37a · LLM Post-Training Pipeline"
ICON         = "🎯"
SUBTITLE     = "SFT, Reward Modeling, RLHF, DPO, RLVR, and Agentic Fine-Tuning"


THEORY = """

##### PART 1 — THE ALIGNMENT GAP AND WHY POST-TRAINING EXISTS

### The Pre-Trained Model: Capable but Unaligned

    A pre-trained LLM learns P(next_token | context) across a massive corpus.
    Given the prompt "The capital of France is", it predicts " Paris" correctly.
    But given "Write me a summary of this document:", it may predict a NEWS
    ARTICLE HEADLINE — because that is the most likely continuation in its
    training data, not an actual summary.

    The pre-trained model has three fundamental misalignments:
        FORMAT:   It generates text in corpus style, not response style.
        INTENT:   It completes prompts; it does not interpret user goals.
        VALUES:   It replicates the full distribution of human text,
                  including harmful, deceptive, or low-quality content.

    Post-training resolves all three — in stages.

### The Canonical Post-Training Pipeline

    Pre-training (raw capability)
            ↓
    SFT — Supervised Fine-Tuning
        Trains the model on (instruction, ideal response) pairs.
        Teaches format, tone, and basic instruction following.
            ↓
    Preference Data Collection
        Humans compare two responses to the same prompt and pick the better one.
        This creates datasets of (prompt, chosen, rejected) triples.
            ↓
    Reward Modeling (optional path)
        Train a scalar model: RM(prompt, response) → quality score.
        The RM distils thousands of human comparisons into one number.
            ↓
    RLHF / PPO  ────────────── (uses RM)
        OR
    DPO / IPO   ────────────── (uses preference pairs directly; skips RM)
        Both optimise the policy to produce responses humans prefer.
            ↓
    RLVR (for reasoning models)
        RL with ground-truth verifiable rewards on math/code problems.
        Produces extended thinking behaviour (o1, DeepSeek-R1 style).
            ↓
    Safety Fine-Tuning
        Constitutional AI, red-teaming, refusal training.
        Often a separate preference optimisation pass focused on harmlessness.
            ↓
    Agentic Fine-Tuning
        Tool use, multi-step planning, and trajectory-level feedback.
        The hardest stage: rewards are sparse and trajectories are long.

    Note: This is ITERATIVE, not linear. Each stage generates new model
    outputs, which are rated by humans or AI, producing new training data
    for the next round. In practice, frontier labs run this loop 5–10+
    times over the lifetime of a model family.


##### PART 2 — SUPERVISED FINE-TUNING (SFT)

### What SFT Does

    SFT fine-tunes the pre-trained model on a curated dataset of
    (instruction, response) pairs. The training objective is IDENTICAL
    to pre-training: cross-entropy next-token prediction. The difference
    is entirely in the DATA:

        Pre-training data:  raw web/book corpus (trillions of tokens)
        SFT data:           curated demonstrations (tens of thousands of examples)

    During SFT, loss is computed ONLY on the response tokens, not the prompt:

        Tokens:  [<system>You are helpful</system><user>What is 2+2?</user>  <assistant>  2+2  equals  4  .]
        Loss:    [   ───────────────────────────────────────────────────────   YES         YES  YES    YES YES]

    The model learns: "given a conversational prefix, generate a helpful
    assistant-style continuation." This teaches format and tone in just
    a few hundred gradient steps.

### SFT Data Quality: The Critical Variable

    More SFT data is NOT always better. Quality dominates quantity.

    High-quality SFT data is:
        DIVERSE: covers many task types (summarisation, QA, code, math, ...)
        CONSISTENT: responses follow the same style and quality standards
        ACCURATE: factually correct and logically sound
        BALANCED: not biased toward verbose or terse responses by default

    Signs of poor SFT data:
        Model refuses everything (too many refusals in training data)
        Model is sycophantic (training data was written to please, not inform)
        Model has inconsistent persona (data from multiple uncoordinated writers)

    LIMA (Less is More for Alignment, Zhou et al. 2023) showed 1,000 high-quality
    examples can match the instruction-following of models trained on 50,000 noisy
    examples. Quality is the bottleneck, not quantity.

### Data Formats: Chat Templates

    SFT data is stored in a "chat template" — a structured format that the
    model learns to associate with assistant behaviour. Example:

        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Explain gradient descent.<|im_end|>
        <|im_start|>assistant
        Gradient descent is an optimisation algorithm ...<|im_end|>

    Different models use different templates:
        ChatML:      <|im_start|>role\ncontent<|im_end|>
        Llama-3:     <|begin_of_text|><|start_header_id|>role<|end_header_id|>...
        Mistral:     [INST] prompt [/INST] response
        Claude:      \n\nHuman: prompt\n\nAssistant: response

    The specific format doesn't matter — what matters is CONSISTENCY.
    The model must see the same template at inference time as during training.

### SFT Training Recipe

    HYPERPARAMETERS:
        Learning rate: 1e-5 to 2e-5 (10–50× smaller than pre-training)
        Epochs: 1–3 (more → overfitting; SFT data is tiny)
        Batch size: small (often 128 sequences, ~256K tokens effective batch)
        Warm-up: 3% of total steps
        LR schedule: cosine decay
        Gradient clip: 1.0

    The small LR and few epochs are critical: SFT should SHAPE the model,
    not overwrite its pre-training knowledge. Too many epochs causes
    "SFT collapse": the model forgets general knowledge and only answers
    in training format.

    Pack multiple short conversations per sequence to efficiency:
        [conv0 <EOS> conv1 <EOS> conv2 ... until ctx_len]
        Loss mask ensures no cross-conversation influence.


##### PART 3 — REWARD MODELING

### The Reward Modeling Problem

    We want to optimise a policy for "what humans prefer" — but we cannot
    query humans 10 million times during RL training. The reward model (RM)
    solves this: it is a neural network trained on human comparisons that
    serves as a differentiable proxy for human judgment.

    Architecture: take the SFT model, remove the language modelling head,
    add a LINEAR HEAD that outputs a single scalar:

        RM architecture:
            SFT backbone (frozen or lightly tuned)
                ↓
            [last token hidden state]   (shape: [batch, d_model])
                ↓
            Linear(d_model → 1)
                ↓
            reward scalar r ∈ ℝ

    The last token hidden state aggregates information about the entire
    sequence (causal attention means the last token attends to everything).

### The Bradley-Terry Preference Model

    Given two responses y_w (chosen/winner) and y_l (rejected/loser) to
    the same prompt x, the Bradley-Terry model defines:

        P(y_w ≻ y_l | x) = σ(r(x, y_w) − r(x, y_l))

    where σ is the sigmoid function and r is the reward model.

    This says: the probability that the human prefers y_w over y_l is
    determined by how much higher its reward is. The margin matters, not
    absolute values.

    LOSS FUNCTION (binary cross-entropy on the preference):

        L_RM = -E[ log σ( r(x, y_w) - r(x, y_l) ) ]

    Minimising this loss pushes:
        r(x, y_w) → higher   (chosen response gets higher score)
        r(x, y_l) → lower    (rejected response gets lower score)
        The MARGIN r(x, y_w) - r(x, y_l) → larger

    Example with numbers:
        r(x, y_w) = 2.1,  r(x, y_l) = -0.4
        margin = 2.5
        P(y_w preferred) = σ(2.5) = 0.92   ✓  (high confidence)

        r(x, y_w) = 0.3,  r(x, y_l) = 0.1
        margin = 0.2
        P(y_w preferred) = σ(0.2) = 0.55   ✗  (barely better)

### Reward Model Training Details

    DATA PIPELINE:
        1. Collect (prompt, response_A, response_B) from raters
        2. Raters pick which response is better (or use a Likert scale)
        3. Add metadata: rater confidence, task type, toxicity flags
        4. Balance across task categories to avoid reward hacking one type

    SCALE:
        Anthropic's Constitutional AI paper used ~300K comparisons.
        InstructGPT used ~33K comparisons.
        Llama RLHF used the OpenAssistant dataset (~161K comparisons).

    TRAINING:
        Init: from SFT checkpoint (same tokeniser, same architecture)
        LR: 1e-5 to 5e-5
        Epochs: 1–2 (overfitting to noisy labels is a real risk)
        Regularisation: margin penalty to prevent reward collapse

    MARGIN PENALTY (optional):
        L = -log σ(r_w - r_l) + λ · max(0, m - (r_w - r_l))
        The margin term penalises predictions with too-small margins.
        Prevents the model from trivially setting r_w ≈ r_l ≈ 0.

### Reward Hacking

    The RM is a PROXY for human preference, not a perfect oracle. If the
    policy is optimised too aggressively against the RM, it will find
    inputs that score highly on the RM but are NOT actually preferred
    by humans. This is reward hacking (Goodhart's Law: "when a measure
    becomes a target, it ceases to be a good measure").

    Classic hacking examples:
        LENGTH HACKING: longer responses score higher → model becomes verbose
        SYCOPHANCY: "Great question! I love your thinking. Here is..." → high reward
        FORMAT TRICKS: bullet points inflate RM scores → model always uses bullets

    Mitigations:
        KL PENALTY in PPO (hard-wires a constraint on how far the policy drifts)
        Ensemble RMs (harder to hack multiple independent models)
        Iterative RM retraining (re-collect human feedback on the new policy)
        Gold-standard human eval that is SEPARATE from the RM


##### PART 4 — RLHF WITH PPO

### The Four Models in PPO-Based RLHF

    PPO-RLHF requires keeping FOUR models in memory simultaneously:

        1. ACTOR (Policy) π_θ:
               The model being trained. Generates responses to prompts.
               Initialised from the SFT checkpoint.

        2. CRITIC (Value model) V_φ:
               Estimates expected return from a given state.
               Same architecture as the actor, with a scalar head.
               Initialised from the SFT checkpoint or the RM.

        3. REFERENCE POLICY π_ref:
               The FROZEN SFT model. Used to compute the KL divergence penalty.
               Never updated. Prevents the policy from drifting too far from SFT.

        4. REWARD MODEL r_ψ:
               The trained RM. Provides scalar reward for each response.
               Frozen during PPO. (Some pipelines fine-tune it iteratively.)

    Memory cost: approximately 4 × SFT model memory.
    For a 7B model at BF16: 4 × 14 GB = 56 GB minimum (before activations).
    In practice: requires multi-GPU setup even for 7B.

### The PPO Objective for LLMs

    At each step, the actor generates a response y to a prompt x.
    The reward is:

        R(x, y) = r_ψ(x, y)  −  β · KL[ π_θ(·|x) || π_ref(·|x) ]

    where:
        r_ψ(x, y):   reward model score (scalar for the whole response)
        β:            KL penalty coefficient (typically 0.01–0.1)
        KL term:      sum of token-level log-probability differences

    The KL penalty is CRITICAL:
        Without it: the policy will exploit the RM aggressively → reward hacking.
        With it: the policy stays "close" to the SFT model in distribution.
        β too small: reward hacking. β too large: model barely improves.

    The KL divergence expands per token as:

        KL[ π_θ || π_ref ] ≈ Σ_t log π_θ(y_t | x, y_{<t})
                                      − log π_ref(y_t | x, y_{<t})

    Each token's log-probability ratio becomes part of the reward signal.

### PPO Clipping: Preventing Large Policy Updates

    The PPO objective clips the probability ratio to prevent destructive
    large updates. For each token t:

        ratio_t = π_θ(y_t | ...) / π_θ_old(y_t | ...)

        L_CLIP = E[ min(ratio_t · A_t,
                        clip(ratio_t, 1−ε, 1+ε) · A_t) ]

    where A_t is the ADVANTAGE — how much better this token was than
    the critic's baseline expectation.

    If ratio > 1+ε (policy increased the token's probability too much):
        clip to 1+ε → limit the update
    If ratio < 1−ε (policy decreased it too much):
        clip to 1−ε → limit the update

    ε is typically 0.2. This means the policy can move at most 20% in
    probability ratio per step — preventing catastrophic forgetting.

### The Advantage Function and GAE

    The advantage A_t = "how much better was this action than expected?"

        A_t = R_t − V_φ(s_t)

    where R_t is the actual return and V_φ(s_t) is the critic's estimate.

    In practice, PPO for LLMs uses Generalised Advantage Estimation (GAE):

        δ_t = r_t + γ · V(s_{t+1}) − V(s_t)     (TD error)
        A_t = Σ_{k=0}^{T-t} (γλ)^k · δ_{t+k}    (exponentially weighted sum)

    γ (discount factor): typically 1.0 for LLMs (no time discounting).
    λ (GAE lambda): typically 0.95, balances bias vs variance.

    The CRITIC is trained to minimise:
        L_VF = (V_φ(s_t) − R_t)²    (mean-squared error against actual returns)

### Full PPO Training Loop for LLMs

    STEP 1: ROLLOUT (generate responses)
        For each prompt in the batch:
            Sample y ~ π_θ(· | x)   (the actor generates a response)
            Compute r = r_ψ(x, y)   (reward model scores it)
            Store (x, y, r, log_probs_old) in the rollout buffer

    STEP 2: COMPUTE RETURNS AND ADVANTAGES
        For each (x, y, r):
            Token-level reward: only the LAST token gets r_ψ score,
                                all tokens get the KL penalty per token.
            Compute V_φ(s_t) for each token position.
            Compute GAE advantages A_t.

    STEP 3: PPO OPTIMISATION (multiple epochs over rollout data)
        For each mini-batch from the rollout buffer:
            Compute new log_probs from π_θ (current policy).
            Compute ratio = exp(log_probs_new − log_probs_old).
            Compute clipped PPO loss.
            Compute value function loss.
            Update actor and critic.

    STEP 4: REPEAT from STEP 1 with updated policy.

    The key insight: the rollout data is collected once from the OLD policy,
    then used for multiple gradient updates. This is "on-policy" RL but
    with a small off-policy correction via the clipping.


##### PART 5 — DIRECT PREFERENCE OPTIMISATION (DPO)

### The Key Insight: Skipping the Reward Model

    In 2023, Rafailov et al. showed that the RLHF objective has a
    CLOSED-FORM SOLUTION. Instead of:
        1. Train a reward model
        2. Run PPO to optimise against it

    You can directly optimise the policy using preference pairs.

    The derivation starts from the RLHF optimisation objective:

        max_π  E_{x~D, y~π}[ r(x,y) ] − β · KL[π || π_ref]

    This has an optimal closed-form solution:

        π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)

    The reward can then be written in terms of the optimal policy:

        r(x,y) = β · log[ π*(y|x) / π_ref(y|x) ] + β · log Z(x)

    where Z(x) is a partition function that cancels in preference comparisons.

    Substituting into the Bradley-Terry preference model:

        P(y_w ≻ y_l | x) = σ( β · log[π*(y_w|x)/π_ref(y_w|x)]
                                − β · log[π*(y_l|x)/π_ref(y_l|x)] )

    Replacing π* with the learned policy π_θ gives the DPO loss:

        L_DPO = −E[ log σ( β · log[π_θ(y_w|x)/π_ref(y_w|x)]
                          − β · log[π_θ(y_l|x)/π_ref(y_l|x)] ) ]

### Understanding the DPO Loss

    The two terms inside σ are LOG-PROBABILITY RATIOS:

        log[π_θ(y_w|x)/π_ref(y_w|x)]  =  Σ_t log π_θ(y_w_t|...) − log π_ref(y_w_t|...)
        This is the implicit reward the policy assigns to the CHOSEN response.

        log[π_θ(y_l|x)/π_ref(y_l|x)]  =  Σ_t log π_θ(y_l_t|...) − log π_ref(y_l_t|...)
        This is the implicit reward for the REJECTED response.

    DPO directly maximises: (implicit reward of chosen) − (implicit reward of rejected)

    What the gradient pushes:
        → INCREASE π_θ(y_w|x) relative to π_ref (chosen becomes more likely)
        → DECREASE π_θ(y_l|x) relative to π_ref (rejected becomes less likely)
        Both relative to the REFERENCE POLICY, preventing collapse to trivial solutions.

### DPO vs RLHF: Trade-offs

    DPO ADVANTAGES:
        No reward model: fewer models, simpler pipeline
        Stable: no PPO instabilities (clipping, value head, etc.)
        Simpler: 2 models instead of 4 (policy + reference only)
        Offline: trains on static preference dataset

    DPO DISADVANTAGES:
        Offline only: cannot use freshly generated samples from the current policy
        Credit assignment: treats the whole response as one unit (coarse signal)
        β sensitivity: needs careful tuning; too small → collapse, too large → no change
        Reference model: must keep π_ref in memory (same cost as the policy)

    RLHF ADVANTAGES:
        Online: generates new samples each rollout → explores better
        Token-level credit assignment via the advantage function
        Can improve iteratively beyond the preference dataset coverage

    In practice:
        DPO works well for most instruction-following tasks (Llama-3, Mistral, Qwen).
        RLHF/PPO is still used when online exploration matters (coding, reasoning).

### DPO Variants

    IPO (Identity Preference Optimisation):
        Replaces the log-sigmoid with a squared loss.
        Less sensitive to β. Better theoretical guarantees.
        L_IPO = E[ (log[π_θ(y_w|x)/π_ref(y_w|x)] − log[π_θ(y_l|x)/π_ref(y_l|x)] − 1/β)² ]

    KTO (Kahneman-Tversky Optimisation):
        Uses only (prompt, response, good/bad label) — no paired comparisons needed.
        Motivated by prospect theory: losses feel worse than equal gains feel good.
        Useful when it's cheaper to label individual responses than compare pairs.

    SimPO (Simple Preference Optimisation):
        Removes the reference model entirely (no π_ref).
        Normalises by sequence length to avoid length bias.
        Even simpler pipeline than DPO.

    ONLINE DPO / ITERATIVE DPO:
        Generate responses from the current policy → rate them → train DPO.
        Repeat. Combines DPO simplicity with RLHF's online exploration.
        Used in Llama-3 and many recent models.


##### PART 6 — RLVR: REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS

### What Makes Reasoning Models Different

    Standard RLHF uses a LEARNED reward model — a neural network trained
    on human preferences. This has fundamental limits for reasoning tasks:
        Human raters cannot reliably judge correctness of hard math proofs.
        RM generalisation breaks down in novel problem domains.
        RM is itself a neural network that can be hacked.

    RLVR bypasses all of these: the reward is computed from a GROUND-TRUTH
    VERIFIER — a deterministic function that checks if the answer is correct.

        reward(response) = 1.0  if  answer == ground_truth_answer
                         = 0.0  otherwise

    This is only possible for domains with VERIFIABLE ANSWERS:
        Mathematics: can check if final numerical answer is correct
        Code: can run tests against a test suite
        Formal proofs: can check via a proof assistant (Lean, Coq)
        Logic puzzles: can verify the solution satisfies all constraints

### The Training Setup

    DATASET:
        Problems with known correct answers (GSM8K, MATH, Codeforces, etc.)
        Each problem has: question, ground-truth answer, optional test cases

    RESPONSE FORMAT:
        The model is prompted to reason BEFORE answering:
            "Think step by step. Put your final answer in \\boxed{}"
        or equivalently with a <thinking> / <answer> split.
        The reasoning trace is unconstrained — the model explores freely.

    REWARD:
        +1 if the extracted final answer matches ground truth
        0 otherwise (binary; some implementations use partial credit for code)

    ALGORITHM:
        DeepSeek-R1 and similar use GRPO (Group Relative Policy Optimisation):
            Sample K responses to the same problem.
            Reward each: r_i ∈ {0, 1}.
            Advantage: A_i = (r_i − mean(r)) / std(r)
            Policy gradient update weighted by A_i.
        This is simpler than PPO: no value head, no GAE, no critic model.

### What Emerges: Chain-of-Thought as RL Discovery

    A crucial insight: the THINKING FORMAT is not taught by SFT.
    It EMERGES from the RL training signal.

    Early in RLVR training: model writes short responses.
    As training proceeds:  model discovers that longer reasoning
                           leads to higher reward on hard problems.
    Late in training:      model writes thousands of tokens of working,
                           with self-correction, exploration, backtracking.

    This "aha moment" — where the model learns to think longer on hard
    problems and shorter on easy ones — is a qualitative phase transition
    that pure SFT on CoT data CANNOT replicate.

    The model is not mimicking human reasoning steps — it is discovering
    its own effective reasoning strategies via reward signal.

### RLVR vs RLHF: When to Use Each

    Use RLVR when:
        You have verifiable answers (math, code, logic)
        You want reasoning depth to emerge, not just mimic
        You care about OOD generalisation (RM doesn't generalise; verifier does)

    Use RLHF/DPO when:
        Task is subjective (writing quality, helpfulness, tone)
        No ground-truth verifier exists
        You want to align with specific human values or preferences


##### PART 7 — CONSTITUTIONAL AI AND RLAIF

### The Bottleneck: Human Labellers Are Expensive and Slow

    For RLHF to work, humans must:
        Read long model outputs (often 300–1000 tokens each)
        Compare pairs on subtle quality dimensions
        Do this 100,000–1,000,000 times
        Maintain consistent standards across diverse task types

    This is expensive ($5–50 per comparison) and slow (days to weeks).
    As models improve, comparisons become harder — humans struggle to
    evaluate outputs better than they could produce themselves.

    Constitutional AI (Bai et al., Anthropic 2022) and RLAIF solve this
    by using an AI model as the rater instead of humans.

### Constitutional AI: Principles-First Alignment

    Constitutional AI introduces a SET OF PRINCIPLES (a "constitution")
    that guide the AI critic's evaluations. Example principles:
        "Choose the response that is less likely to cause harm."
        "Choose the response that is more honest and accurate."
        "Choose the response that a thoughtful senior employee would prefer."

    TWO-PHASE PIPELINE:

    Phase 1: Supervised Learning from AI Feedback (SL-CAI)
        1. Sample a harmful response from the model (using adversarial prompts).
        2. Prompt a CRITIC MODEL with the response and a principle:
               "The above response contains harmful content.
                Rewrite it to be safe and helpful."
        3. Use the rewritten response as the SFT target.
        4. Fine-tune on (original_prompt, ai_revised_response) pairs.

    Phase 2: RL from AI Feedback (RLAIF)
        1. Sample pairs of responses from the current policy.
        2. Prompt the CRITIC MODEL:
               "Which response better follows the principle: {principle}?
                Response A: {y_w}  Response B: {y_l}
                Answer with A or B."
        3. Use the critic's preference as the training signal.
        4. Train a reward model on the AI-labelled pairs.
        5. Run PPO with this RM.

### RLAIF: Replacing Human Raters at Scale

    RLAIF (Reinforcement Learning from AI Feedback) generalises the above:
    use any sufficiently capable LLM as the preference judge.

    The AI rater can be prompted with different criteria:
        Helpfulness:  "Which response more directly addresses the user's need?"
        Honesty:      "Which response makes fewer unverifiable claims?"
        Harmlessness: "Which response is less likely to cause harm?"

    RLAIF at scale (Google's "RLAIF" paper, 2023):
        AI feedback nearly matches human feedback on helpfulness
        AI is consistent (same model, same prompts → same ratings)
        AI is fast and cheap (millions of comparisons at cents each)
        AI can evaluate domains humans cannot judge (technical, multilingual)

    Limitation: AI rater carries the biases and errors of its training.
    "Sycophantic rater" problem: if the critic is also a language model,
    it may prefer verbose, confident responses regardless of quality.


##### PART 8 — AGENTIC FINE-TUNING

### Why Agentic Tasks Are Harder

    Standard RLHF trains on SINGLE-TURN interactions: one prompt → one response.
    Agentic tasks require MULTI-STEP TRAJECTORIES:

        User: "Write a Python function to scrape product prices from Amazon."
        Agent: [call browser_tool("amazon.com/search?q=laptops")]
               [observe: HTML response with product listings]
               [call code_interpreter("parse prices from HTML")]
               [observe: list of prices]
               [write: "Here is the function: ...")
               [call run_tests("test_scraper.py")]
               [observe: 1 test FAILED]
               [rewrite function to fix bug]
               [call run_tests again]
               [observe: all tests pass]
               [write: "Here is the corrected function..."]

    The trajectory above has 8 steps. The reward (did the function work?)
    arrives only at the END. This is the SPARSE REWARD problem.

### The Sparse Reward Problem

    In single-turn RLHF:
        Reward is given after one response.
        Policy gradient can credit/blame that one response directly.

    In multi-step agentic RL:
        Reward arrives after potentially dozens of steps.
        Which step CAUSED success or failure?
        This is the credit assignment problem — much harder.

    A model that takes a wrong tool call at step 3 may still succeed at
    step 10 by recovering. Or a model that looks right up to step 9 may
    fail at the final step. Reward cannot tell us which decisions mattered.

    SOLUTIONS:
        Process Reward Models (PRM): reward at EACH step, not just the end.
            Train a model to predict "is this intermediate step correct?"
            Used in OpenAI's MATH verification work.
        Hindsight Relabelling: treat failed trajectories as demonstrations
            of how NOT to proceed (negative examples for DPO).
        Monte Carlo rollouts: run K completions from each state to estimate
            which states are "good" regardless of the final outcome.

### Tool Use and Function Calling

    Agentic models emit STRUCTURED TOOL CALLS during generation:

        {"tool": "web_search", "args": {"query": "latest Python releases"}}

    The model must learn:
        WHEN to call a tool (vs answer from memory)
        WHICH tool to call (from a catalogue of available tools)
        HOW to format the call (correct JSON schema, correct args)
        HOW to USE the result (parse observations, update plan)

    SFT for tool use:
        Collect expert demonstrations of tool-using trajectories.
        Typically 10,000–100,000 trajectories across different tool types.
        Format: system prompt with tool descriptions + multi-turn dialogue.

    RL for tool use:
        Generate rollouts: model picks tools, environment executes them.
        Reward: task completion (binary or graded).
        The model learns to prefer tool combinations that lead to success.

### Trajectory-Level Feedback

    For agentic RL on multi-step trajectories, the PPO formulation must
    account for the full trajectory:

        Trajectory τ = (s_0, a_0, o_0, s_1, a_1, o_1, ..., s_T, a_T, R)

        where:
            s_t = state (conversation history + observations so far)
            a_t = action (next LLM generation: text or tool call)
            o_t = observation (tool result or user message)
            R   = final reward (task success/failure)

    Token-level reward is SPARSE:
        Most tokens get reward = 0
        Only the terminal step gets R (e.g., +1 for task success)
        Plus KL penalty at every token (same as single-turn)

    The advantage function must propagate reward BACKWARDS through all
    intermediate steps — telling each early action how much it contributed
    to the final outcome. This requires:
        A value model that can estimate "expected success from this state"
        Long context capability (full trajectory fits in the context window)
        Careful discount factor γ (usually 1.0 — no discounting within a task)

### SFT vs RL for Agents: What Each Teaches

    SFT teaches:
        Correct tool call format
        Basic planning structure (search → analyse → respond)
        Common patterns in training demonstrations

    RL teaches:
        RECOVERY from mistakes (not in any SFT demonstration)
        EFFICIENT tool use (fewer calls to achieve the goal)
        GENERALISATION to new tool combinations not seen in SFT
        ROBUSTNESS to unexpected tool outputs (error handling)

    The combination is essential: SFT prevents random exploration from
    scratch; RL improves beyond the ceiling of human demonstrations.


##### PART 9 — THE ITERATIVE PIPELINE AND PRACTICAL CONSIDERATIONS

### Why the Pipeline Is Always Iterative

    Each post-training stage changes the policy distribution.
    A changed policy produces different outputs than the SFT model.
    Those new outputs expose new failure modes not in the original data.

    Frontier models iterate the full pipeline 5–15 times:
        Round 1: SFT on human demos → RLHF (data from SFT model)
        Round 2: Generate from Round 1 model → new human ratings → RLHF
        Round 3: Generate from Round 2 model → harder prompts → RLHF
        ...

    Each round: better prompts, better raters, harder edge cases.
    The "training data" evolves to track the model's improving capabilities.

### KL Budget: The Fundamental Tension

    Every preference optimisation stage pulls the model AWAY from its
    pre-training distribution. The KL divergence from the pre-trained
    model is a measure of how much the model has changed:

        KL[ π_final || π_pretrain ]

    Too little KL → the model is barely better than pre-training.
    Too much KL → the model loses general knowledge (catastrophic forgetting).

    Practical sign of too-much-KL:
        Model becomes verbose, sycophantic, or repetitive.
        Perplexity on general text (WikiText) increases significantly.
        Model refuses too many requests (if safety fine-tuning was too aggressive).

    The KL budget is split across stages:
        SFT: small KL (just format/tone)
        RLHF helpfulness: medium KL
        Safety fine-tuning: additional KL
        Total: keep total KL below a threshold empirically determined per model.

### Evaluation: How Do You Know the Pipeline Worked?

    AUTOMATIC METRICS:
        MT-Bench: multi-turn instruction following (GPT-4 as judge)
        AlpacaEval: win-rate vs text-davinci-003 (GPT-4 as judge)
        TruthfulQA: factual accuracy under pressure
        BBH, MMLU, GSM8K: general reasoning (check forgetting)
        HumanEval / SWE-Bench: coding capability

    HUMAN EVALUATION:
        Side-by-side comparisons vs the previous model version.
        Specifically probe for: helpfulness, safety, refusals, verbosity.
        Red-teaming: adversarial prompts designed to elicit failure.

    REGRESSION TESTING:
        Verify the model did NOT regress on pre-training benchmarks.
        A 5-point improvement on MT-Bench means nothing if MMLU dropped 10 points.

### Common Failure Modes

    SYCOPHANCY:
        Model agrees with whatever the user says, even when wrong.
        Cause: human raters prefer confident, agreeable responses.
        Fix: explicitly include "disagrees with user when user is wrong"
             examples in SFT and preference data.

    REWARD HACKING:
        Model discovers patterns that score high on RM but are actually bad.
        Fix: ensemble RMs, iterative RM retraining, independent human eval.

    CATASTROPHIC FORGETTING:
        Model loses capability on tasks not covered by preference data.
        Fix: KL penalty, mix original pre-training data into SFT, lower LR.

    REFUSAL OVER-GENERALISATION:
        Model refuses benign requests because they superficially resemble
        harmful ones in the safety training data.
        Fix: targeted refusal data that distinguishes harmful vs merely-sensitive.

    LENGTH BIAS:
        Model produces unnecessarily long responses.
        Cause: human raters often prefer longer, more detailed answers.
        Fix: normalise rewards by length in RM training.
"""


OPERATIONS = {

    "1 · SFT Data Pipeline, Chat Templates, and Reward Model Training": {
        "description": (
            "Build the SFT data pipeline from raw preference data to formatted training examples. "
            "Implement multiple chat templates (ChatML, Llama-3, Mistral). "
            "Compute per-token SFT loss with correct response masking. "
            "Implement the Bradley-Terry reward model loss from scratch. "
            "Simulate reward model training on preference pairs. "
            "Visualise reward distributions for chosen vs rejected responses."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 65)
print("  SFT DATA PIPELINE AND REWARD MODEL TRAINING")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Chat templates and SFT data formatting
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Chat templates and SFT loss masking")
print("━" * 65)
print()

def format_chatml(messages, tokenise=False):
    """Format a conversation in ChatML format."""
    result = ""
    for msg in messages:
        result += f"<|im_start|>{msg[\'role\']}\\n{msg[\'content\']}<|im_end|>\\n"
    result += "<|im_start|>assistant\\n"
    return result

def format_llama3(messages):
    """Format in Llama-3 header format."""
    result = "<|begin_of_text|>"
    for msg in messages:
        result += (f"<|start_header_id|>{msg[\'role\']}<|end_header_id|>\\n\\n"
                   f"{msg[\'content\']}<|eot_id|>")
    result += "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
    return result

def format_mistral(messages):
    """Format in Mistral [INST] format."""
    result = ""
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            result += f"[INST] {msg[\'content\']} [/INST]"
        elif msg["role"] == "assistant":
            result += f" {msg[\'content\']}</s>"
    return result.strip()

sample_conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is gradient descent?"},
]
sample_response = "Gradient descent is an optimisation algorithm that iteratively updates parameters in the direction of steepest loss reduction."

print("  Same conversation, three templates:")
print()
print("  ── ChatML ──────────────────────────────────────────────────")
chatml = format_chatml(sample_conversation)
print("  " + chatml.replace("\\n", "\\n  "))
print()
print("  ── Llama-3 ─────────────────────────────────────────────────")
print("  " + format_llama3(sample_conversation[:1]))  # no system in Llama-3 base
print()
print("  ── Mistral ─────────────────────────────────────────────────")
print("  " + format_mistral(sample_conversation[1:]))
print()

# Demonstrate loss masking
print("  SFT LOSS MASKING:")
print("  Compute loss ONLY on response tokens, not prompt tokens.")
print()

# Simulate token IDs
PROMPT_TOKENS   = [101, 234, 567, 890, 111]   # fake token IDs for the prompt
RESPONSE_TOKENS = [222, 333, 444, 555, 666, 777, 888]  # response

all_tokens   = PROMPT_TOKENS + RESPONSE_TOKENS
loss_mask    = [0] * len(PROMPT_TOKENS) + [1] * len(RESPONSE_TOKENS)

print(f"  Prompt tokens:    {PROMPT_TOKENS}  (mask=0, no gradient)")
print(f"  Response tokens:  {RESPONSE_TOKENS}  (mask=1, gradient flows)")
print(f"  Full sequence:    {all_tokens}")
print(f"  Loss mask:        {loss_mask}")
print()

# Simulate cross-entropy loss per token
np.random.seed(42)
per_token_loss = np.random.uniform(0.2, 3.5, len(all_tokens))
masked_loss    = per_token_loss * np.array(loss_mask)

# SFT loss = mean over RESPONSE tokens only
sft_loss = masked_loss.sum() / sum(loss_mask)

print(f"  Per-token CE loss (simulated): {np.round(per_token_loss, 2)}")
print(f"  After masking:                 {np.round(masked_loss, 2)}")
print(f"  SFT loss (mean over response): {sft_loss:.4f}")
print()
print("  Key: dividing by len(response), not len(full_sequence).")
print("  Otherwise prompt length would dilute the response gradient.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Bradley-Terry Reward Model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Bradley-Terry reward model: loss and training")
print("━" * 65)
print()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def rm_loss(r_w, r_l):
    """
    Bradley-Terry preference loss.
    r_w: reward for chosen  (winner)
    r_l: reward for rejected (loser)
    Loss = -log σ(r_w - r_l)
    """
    margin = r_w - r_l
    prob   = sigmoid(margin)
    loss   = -np.log(prob + 1e-8)
    return loss, prob, margin

print("  Bradley-Terry preference model:")
print("  P(y_w preferred) = σ(r_w - r_l)")
print("  L = -log σ(r_w - r_l)")
print()
print(f"  {'r_w':>6} {'r_l':>6} {'margin':>8} {'P(w wins)':>12} {'loss':>8}")
print(f"  {'─'*50}")

scenarios = [
    (3.0,  -1.0, "Strong preference"),
    (1.5,   0.5, "Moderate preference"),
    (0.6,   0.4, "Weak preference"),
    (0.0,   0.0, "No preference (ambiguous)"),
    (-0.5,  1.0, "Inverted (wrong!)"),
]

for r_w, r_l, label in scenarios:
    loss, prob, margin = rm_loss(r_w, r_l)
    flag = "✅" if prob > 0.7 else ("⚠️" if prob > 0.5 else "❌")
    print(f"  {r_w:>6.1f} {r_l:>6.1f} {margin:>8.1f} {prob:>11.1%} {loss:>8.4f}  {flag}  {label}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Training a reward model on preference data
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Simulated reward model training")
print("━" * 65)
print()

class SimpleRewardModel:
    """
    A minimal linear reward model.
    Input: feature vector representing a response.
    Output: scalar reward.
    """
    def __init__(self, n_features, lr=0.05):
        np.random.seed(0)
        self.W  = np.random.randn(n_features) * 0.01
        self.lr = lr

    def forward(self, features):
        return float(np.dot(self.W, features))

    def train_step(self, feat_w, feat_l):
        """
        One gradient step on a single preference pair.
        Uses autograd-free derivation:
          dL/dr_w = -σ(-margin)   (gradient w.r.t. r_w)
          dL/dr_l = +σ(-margin)   (gradient w.r.t. r_l)
        """
        r_w    = self.forward(feat_w)
        r_l    = self.forward(feat_l)
        margin = r_w - r_l

        # Gradient of loss w.r.t. rewards
        prob    = sigmoid(margin)
        loss    = -np.log(prob + 1e-8)
        dl_drw  = -(1.0 - prob)        # d(-log σ(m))/d(r_w)
        dl_drl  =  (1.0 - prob)        # d(-log σ(m))/d(r_l)

        # Gradient w.r.t. weights (linear model: dL/dW = dL/dr * x)
        grad_w  = dl_drw * feat_w + dl_drl * feat_l
        self.W -= self.lr * grad_w

        return loss, prob

# Simulate preference data
# Features: [length_score, fluency_score, factuality_score, helpfulness_score]
# Good responses have high scores; bad responses have lower scores
np.random.seed(123)

n_pairs    = 200
n_features = 4

chosen_feats  = np.random.randn(n_pairs, n_features) + np.array([0.5, 0.8, 0.7, 0.9])
rejected_feats = np.random.randn(n_pairs, n_features) + np.array([-0.3, 0.2, 0.1, -0.1])

rm = SimpleRewardModel(n_features, lr=0.1)

print("  Training reward model on 200 preference pairs...")
print(f"  Features: [length, fluency, factuality, helpfulness]")
print()

epoch_losses = []
for epoch in range(10):
    losses, probs = [], []
    perm = np.random.permutation(n_pairs)
    for idx in perm:
        loss, prob = rm.train_step(chosen_feats[idx], rejected_feats[idx])
        losses.append(loss)
        probs.append(prob)

    avg_loss = np.mean(losses)
    avg_acc  = np.mean([p > 0.5 for p in probs])
    epoch_losses.append(avg_loss)

    if epoch in [0, 2, 5, 9]:
        print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f}  accuracy={avg_acc:.1%}")

print()
print(f"  Final weights (what the RM learned to value):")
feature_names = ["Length", "Fluency", "Factuality", "Helpfulness"]
for name, w in zip(feature_names, rm.W):
    bar = "█" * int(abs(w) * 5) if w > 0 else "░" * int(abs(w) * 5)
    sign = "+" if w >= 0 else "-"
    print(f"    {name:<15} {sign}{abs(w):.3f}  {bar}")
print()

# Compute reward distributions on held-out data
test_chosen   = np.random.randn(50, n_features) + np.array([0.5, 0.8, 0.7, 0.9])
test_rejected = np.random.randn(50, n_features) + np.array([-0.3, 0.2, 0.1, -0.1])

r_chosen   = np.array([rm.forward(f) for f in test_chosen])
r_rejected = np.array([rm.forward(f) for f in test_rejected])

print(f"  Reward distributions on held-out test set (n=50 each):")
print(f"    Chosen   — mean: {r_chosen.mean():+.3f}  std: {r_chosen.std():.3f}")
print(f"    Rejected — mean: {r_rejected.mean():+.3f}  std: {r_rejected.std():.3f}")
print(f"    Separation: {r_chosen.mean() - r_rejected.mean():.3f}")
print(f"    % correctly ranked: {np.mean(r_chosen > r_rejected):.1%}")
print()

# ASCII reward distribution histogram
def ascii_hist(data, label, width=30, bins=10):
    lo, hi = data.min(), data.max()
    bin_edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(data, bins=bin_edges)
    max_count  = counts.max()
    print(f"  {label}:")
    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        left    = bin_edges[i]
        print(f"    {left:+.2f} │{'█' * bar_len}")
    print()

ascii_hist(r_chosen,   "Chosen  (preferred) rewards")
ascii_hist(r_rejected, "Rejected (dispreferred) rewards")
''',
    },

    "2 · PPO Training Loop, KL Penalty, and DPO Loss": {
        "description": (
            "Implement the full PPO objective for LLMs: KL penalty, advantage estimation, and clipping. "
            "Simulate a token-level rollout and compute per-token rewards. "
            "Implement the DPO loss and show the gradient interpretation. "
            "Compare PPO vs DPO loss landscapes numerically. "
            "Show reward hacking via a simulated RM that can be gamed. "
            "Implement the DPO variants: IPO and the length-normalised SimPO."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 65)
print("  PPO TRAINING LOOP, KL PENALTY, AND DPO LOSS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Token-level PPO reward with KL penalty
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — PPO reward structure: RM score + KL penalty")
print("━" * 65)
print()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_kl_divergence(log_probs_policy, log_probs_ref):
    """
    Token-level KL: KL[π_θ || π_ref] = Σ_t (log π_θ(t) - log π_ref(t))
    Both inputs: arrays of log-probabilities for each token.
    """
    return log_probs_policy - log_probs_ref  # per-token KL contribution

def compute_ppo_reward(rm_score, log_probs_policy, log_probs_ref, beta=0.05):
    """
    Full PPO reward for a response:
      r_t = KL_penalty_t  for all tokens except the last
      r_T = rm_score + KL_penalty_T  for the final token
    beta: KL penalty coefficient
    """
    T       = len(log_probs_policy)
    kl      = compute_kl_divergence(log_probs_policy, log_probs_ref)
    rewards = -beta * kl                  # KL penalty on every token
    rewards[-1] += rm_score               # RM reward only on final token
    return rewards, kl

np.random.seed(42)
T = 12  # response length in tokens

# Simulate log-probs for policy and reference
log_probs_ref    = np.random.uniform(-3.0, -0.5, T)
log_probs_policy_small_kl = log_probs_ref + np.random.normal(0, 0.1, T)
log_probs_policy_large_kl = log_probs_ref + np.random.normal(0, 0.8, T)

rm_score = 2.0  # reward model says this is a good response
beta = 0.05

print(f"  Response length: {T} tokens")
print(f"  RM score:        {rm_score:.1f}")
print(f"  KL penalty β:    {beta}")
print()

for label, log_probs_p in [
    ("Small KL (policy ≈ reference)", log_probs_policy_small_kl),
    ("Large KL (policy diverged)",    log_probs_policy_large_kl),
]:
    rewards, kl = compute_ppo_reward(rm_score, log_probs_p, log_probs_ref, beta)
    total_kl    = kl.sum()
    total_r     = rewards.sum()
    print(f"  ── {label} ──────────────────")
    print(f"     Total KL divergence: {total_kl:.4f}")
    print(f"     KL penalty (sum):    {(-beta * kl).sum():.4f}")
    print(f"     RM contribution:     {rm_score:.4f}")
    print(f"     Net total reward:    {total_r:.4f}")
    print(f"     Token rewards: {np.round(rewards, 3)}")
    print()

print("  Insight: large KL erodes the RM reward via the penalty.")
print(f"  β={beta}: 10 bits of KL costs {beta * 10:.2f} reward units.")
print(f"  If RM score={rm_score}, the policy can 'afford' at most "
      f"{rm_score/beta:.0f} bits of KL before the update becomes net-negative.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: PPO clipping and advantage computation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — PPO clipping: bounding the policy update")
print("━" * 65)
print()

def ppo_clipped_objective(advantage, log_prob_new, log_prob_old, epsilon=0.2):
    """
    Clipped PPO objective for a single token.
    L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    """
    ratio           = np.exp(log_prob_new - log_prob_old)
    clipped_ratio   = np.clip(ratio, 1 - epsilon, 1 + epsilon)
    unclipped_obj   = ratio * advantage
    clipped_obj     = clipped_ratio * advantage
    objective       = np.minimum(unclipped_obj, clipped_obj)   # take the pessimistic
    was_clipped     = (ratio != clipped_ratio)
    return objective, ratio, was_clipped

print("  PPO clipping with ε=0.2: ratio ∈ [0.8, 1.2]")
print()
print(f"  {'Advantage':>10} {'ratio':>7} {'unclipped':>11} {'clipped':>9} {'final':>8} {'clipped?':>9}")
print(f"  {'─'*60}")

test_cases = [
    # (advantage, ratio)
    ( 1.5, 1.5,  "ratio too high, pos advantage → clip"),
    ( 1.5, 0.5,  "ratio too low,  pos advantage → no clip (discourages action)"),
    (-1.0, 1.5,  "ratio too high, neg advantage → no clip (already punishing)"),
    (-1.0, 0.5,  "ratio too low,  neg advantage → clip"),
    ( 0.8, 1.1,  "within bounds → no clip"),
    ( 0.8, 0.9,  "within bounds → no clip"),
]

log_prob_old = -1.0   # reference

for adv, target_ratio, label in test_cases:
    log_prob_new  = log_prob_old + np.log(target_ratio)
    obj, ratio, clipped = ppo_clipped_objective(adv, log_prob_new, log_prob_old)
    print(f"  {adv:>10.1f} {ratio:>7.2f} {adv*ratio:>11.3f} "
          f"{adv*np.clip(ratio, 0.8, 1.2):>9.3f} {obj:>8.3f}  {'✂️ YES' if clipped else '   no'}")

print()
print("  The clip ensures: when ratio > 1+ε AND advantage > 0,")
print("  we do NOT increase the objective by raising the ratio further.")
print("  This prevents a single batch from making a destructively large update.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: DPO loss and gradient interpretation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — DPO loss: the policy as its own reward model")
print("━" * 65)
print()

def dpo_loss(log_prob_chosen_policy, log_prob_rejected_policy,
             log_prob_chosen_ref,    log_prob_rejected_ref,
             beta=0.1):
    """
    DPO loss for a single preference pair.
    L_DPO = -log σ(β * (log_ratio_chosen - log_ratio_rejected))

    log_ratio = log π_θ(y|x) - log π_ref(y|x)  (summed over tokens)
    """
    log_ratio_chosen   = log_prob_chosen_policy   - log_prob_chosen_ref
    log_ratio_rejected = log_prob_rejected_policy - log_prob_rejected_ref

    implicit_reward_diff = beta * (log_ratio_chosen - log_ratio_rejected)
    prob  = sigmoid(implicit_reward_diff)
    loss  = -np.log(prob + 1e-8)

    return loss, prob, log_ratio_chosen, log_ratio_rejected

print("  DPO loss: L = -log σ(β · (implicit_reward_w - implicit_reward_l))")
print("  where implicit_reward = log π_θ(y|x) - log π_ref(y|x)")
print()
print(f"  β=0.1 (temperature controlling how far from reference we go)")
print()
print(f"  {'Policy probs':^30} | {'Ref probs':^24} | {'Impl.Rew':^14} | {'P(w>l)':>7} | {'Loss':>7}")
print(f"  {'chosen':>12} {'rejected':>12}  | {'chosen':>10} {'rejected':>10}  | {'w':>6} {'l':>6} |         |")
print(f"  {'─'*80}")

beta = 0.1
dpo_scenarios = [
    # (lp_chosen_policy, lp_rejected_policy, lp_chosen_ref, lp_rejected_ref, label)
    (-2.0, -4.0, -2.5, -2.5, "Policy favours chosen (good training signal)"),
    (-3.0, -2.5, -2.5, -2.5, "Policy favours rejected (large loss, big update)"),
    (-2.5, -3.0, -2.5, -3.0, "Policy = reference (loss = -log 0.5 ≈ 0.693)"),
    (-1.5, -3.5, -2.5, -2.5, "Policy strongly favours chosen (small loss)"),
]

for lp_cp, lp_rp, lp_cr, lp_rr, label in dpo_scenarios:
    loss, prob, ir_c, ir_r = dpo_loss(lp_cp, lp_rp, lp_cr, lp_rr, beta)
    print(f"  {lp_cp:>12.1f} {lp_rp:>12.1f}  | {lp_cr:>10.1f} {lp_rr:>10.1f}  | "
          f"{ir_c:>6.2f} {ir_r:>6.2f} | {prob:>7.1%} | {loss:>7.4f}")

print()
print("  When policy already prefers chosen: loss is small, small gradient.")
print("  When policy prefers rejected:       loss is large, large gradient.")
print("  This is exactly what we want — the model has learned when it's already right.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: DPO variants — IPO and SimPO
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — DPO variants: IPO and SimPO")
print("━" * 65)
print()

def ipo_loss(log_prob_chosen_policy, log_prob_rejected_policy,
             log_prob_chosen_ref,    log_prob_rejected_ref,
             beta=0.1):
    """
    IPO: squared loss instead of log-sigmoid.
    L = (log_ratio_chosen - log_ratio_rejected - 1/(2β))²
    """
    log_ratio_chosen   = log_prob_chosen_policy   - log_prob_chosen_ref
    log_ratio_rejected = log_prob_rejected_policy - log_prob_rejected_ref
    margin = log_ratio_chosen - log_ratio_rejected
    loss   = (margin - 1.0 / (2 * beta)) ** 2
    return loss

def simpo_loss(log_prob_chosen_policy,  chosen_length,
               log_prob_rejected_policy, rejected_length,
               beta=2.0, gamma=0.5):
    """
    SimPO: no reference model, normalise by length, add margin γ.
    L = -log σ(β/|yw| * log π(yw) - β/|yl| * log π(yl) - γ)
    """
    avg_lp_chosen   = log_prob_chosen_policy   / chosen_length
    avg_lp_rejected = log_prob_rejected_policy / rejected_length
    margin          = beta * (avg_lp_chosen - avg_lp_rejected) - gamma
    prob            = sigmoid(margin)
    loss            = -np.log(prob + 1e-8)
    return loss, prob

print("  Comparing DPO, IPO, SimPO on the same preference pairs:")
print()
print(f"  {'Scenario':^35} | {'DPO loss':>9} | {'IPO loss':>9} | {'SimPO loss':>11}")
print(f"  {'─'*70}")

beta = 0.1
pairs = [
    # (lp_chosen_policy, lp_rejected_policy, lp_chosen_ref, lp_rejected_ref, len_c, len_r)
    (-2.0, -4.0, -2.5, -2.5, 10, 8),
    (-3.0, -2.5, -2.5, -2.5, 10, 8),
    (-2.5, -3.0, -2.5, -3.0, 10, 8),
    (-2.0, -2.5, -2.5, -2.5, 10, 4),   # rejected is shorter
]
labels = [
    "Policy favours chosen",
    "Policy favours rejected",
    "Policy = reference",
    "Short rejected (length bias test)",
]

for (lp_cp, lp_rp, lp_cr, lp_rr, lc, lr), label in zip(pairs, labels):
    l_dpo,  _, _, _ = dpo_loss(lp_cp, lp_rp, lp_cr, lp_rr, beta)
    l_ipo           = ipo_loss(lp_cp, lp_rp, lp_cr, lp_rr, beta)
    l_simpo, _      = simpo_loss(lp_cp, lc, lp_rp, lr, beta=2.0, gamma=0.5)
    print(f"  {label:<35} | {l_dpo:>9.4f} | {l_ipo:>9.4f} | {l_simpo:>11.4f}")

print()
print("  IPO: smooth squared loss, no saturation at extreme margins.")
print("  SimPO: no reference model needed + length normalisation.")
print("       Shorter rejected responses don't get a free pass.")
''',
    },

    "3 · RLVR Simulation, Reward Hacking, and Pipeline Comparison": {
        "description": (
            "Simulate GRPO training as used in RLVR (DeepSeek-R1 style). "
            "Show how thinking length emerges as a strategy under verifiable rewards. "
            "Demonstrate reward hacking: an RM that can be gamed vs a verifiable reward. "
            "Compare SFT → RLHF → DPO → RLVR on simulated task performance curves. "
            "Implement the iterative pipeline loop with reward model retraining. "
            "Print a complete post-training cookbook and decision guide."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 65)
print("  RLVR, REWARD HACKING, AND PIPELINE COMPARISON")
print("=" * 65)
print()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: RLVR and GRPO — Group Relative Policy Optimisation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — RLVR / GRPO: training on verifiable rewards")
print("━" * 65)
print()

print("  GRPO samples K responses to the same problem,")
print("  then uses their relative reward to estimate advantages.")
print("  No value model (critic) needed — the group IS the baseline.")
print()

def verifiable_reward(response_quality, problem_difficulty, thinking_tokens):
    """
    Simulates a verifiable reward (e.g., math problem).
    Probability of getting correct answer depends on:
       - The quality of the reasoning (thinking depth vs. problem difficulty)
       - Chance (some correct answers via shortcuts; some wrong despite good reasoning)
    thinking_tokens: how many tokens the model spent "thinking" before answering.
    """
    # More thinking helps on hard problems, not on easy ones
    thinking_bonus = min(thinking_tokens / 500.0, 1.0) * (problem_difficulty - 0.3)
    effective_quality = response_quality + max(0, thinking_bonus)
    p_correct = sigmoid(3.0 * (effective_quality - problem_difficulty + 0.3))
    correct   = np.random.random() < p_correct
    return float(correct)

def grpo_step(policy_quality, problem_difficulty, K=8, thinking_budget=200):
    """
    Simulate one GRPO step.
    K: number of responses to sample per problem.
    thinking_budget: max thinking tokens the policy will use.
    Returns the policy gradient update direction.
    """
    rewards     = []
    think_lens  = []

    for _ in range(K):
        # Policy decides how much to think (proportional to its quality + randomness)
        think_len = int(thinking_budget * policy_quality + np.random.normal(0, 30))
        think_len = max(0, think_len)
        think_lens.append(think_len)

        # Response quality with some noise
        resp_quality = np.random.normal(policy_quality, 0.15)
        r = verifiable_reward(resp_quality, problem_difficulty, think_len)
        rewards.append(r)

    rewards = np.array(rewards, dtype=float)
    mean_r  = rewards.mean()
    std_r   = rewards.std() + 1e-8

    # GRPO advantage: normalise within the group
    advantages = (rewards - mean_r) / std_r

    # Policy gradient signal: positive advantage → reinforce that response
    gradient_signal = advantages.mean()   # simplified

    return mean_r, gradient_signal, np.mean(think_lens)


print("  GRPO training on math problems of increasing difficulty:")
print(f"  K={8} responses per problem, binary reward (correct / incorrect)")
print()
print(f"  {'Step':>5} | {'Difficulty':>11} | {'Avg reward':>11} | "
      f"{'Avg think len':>14} | {'Grad signal':>12}")
print(f"  {'─'*65}")

np.random.seed(7)
policy_quality = 0.4    # starts not great
thinking_budget = 300

for step in range(15):
    difficulty    = 0.3 + step * 0.05      # problems get harder
    mean_r, grad, think_len = grpo_step(policy_quality, difficulty, K=8,
                                        thinking_budget=thinking_budget)

    # Policy improves with positive gradient; thinking budget adapts
    policy_quality  = min(0.95, policy_quality + 0.01 * max(grad, 0))
    thinking_budget = min(800, thinking_budget + 10 * max(grad, 0))

    bar = "█" * int(mean_r * 20)
    print(f"  {step+1:>5} | {difficulty:>11.2f} | {mean_r:>11.1%} | "
          f"{think_len:>14.0f} | {grad:>+12.4f}")

print()
print("  Observation: as problems get harder, the policy learns to allocate")
print("  MORE thinking tokens (longer chains of reasoning emerge from reward,")
print("  not from supervised imitation of longer CoT demonstrations).")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Reward hacking — the RM vs the verifier
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Reward hacking: gaming the RM vs a verifier")
print("━" * 65)
print()

class HackableRewardModel:
    """
    A reward model that assigns high scores based on surface features.
    It can be hacked by a policy that learns to exploit these features.
    """
    def score(self, length, uses_bullets, uses_hedges, actual_quality):
        """
        Simulated RM that overweights length and formatting.
        A hacky policy can get a high score without actual quality.
        """
        return (0.4 * min(length / 300, 1.0)   # longer = better (bias!)
              + 0.2 * uses_bullets              # bullets = better (bias!)
              + 0.1 * uses_hedges               # "it seems" = nuanced (bias!)
              + 0.3 * actual_quality)            # actual quality (underweighted)

class VerifiableReward:
    """Ground-truth verifier: correct answer or not. Cannot be gamed."""
    def score(self, actual_quality):
        return float(np.random.random() < actual_quality)

rm        = HackableRewardModel()
verifier  = VerifiableReward()

print("  RM weights: 40% length + 20% bullets + 10% hedges + 30% quality")
print("  Verifier:   100% actual correctness (binary)")
print()
print("  Simulating two policies: honest vs reward-hacker")
print()

np.random.seed(99)
n_eval = 200

# Honest policy: improves actual quality, ignores surface features
honest_rm_scores, honest_true_scores = [], []
# Hacker policy: optimises surface features, ignores actual quality
hacker_rm_scores, hacker_true_scores = [], []

for _ in range(n_eval):
    # Honest policy at quality=0.65
    honest_quality = np.random.normal(0.65, 0.1)
    honest_rm   = rm.score(length=150, uses_bullets=0,
                           uses_hedges=0, actual_quality=honest_quality)
    honest_true = verifier.score(honest_quality)
    honest_rm_scores.append(honest_rm)
    honest_true_scores.append(honest_true)

    # Hacker policy: writes long bullet-pointed responses; low actual quality
    hacker_quality = np.random.normal(0.35, 0.1)
    hacker_rm   = rm.score(length=400, uses_bullets=1,
                           uses_hedges=1, actual_quality=hacker_quality)
    hacker_true = verifier.score(hacker_quality)
    hacker_rm_scores.append(hacker_rm)
    hacker_true_scores.append(hacker_true)

print(f"  {'Metric':<25} | {'Honest policy':>15} | {'Hacker policy':>15}")
print(f"  {'─'*60}")
print(f"  {'RM score (mean)':<25} | {np.mean(honest_rm_scores):>15.3f} | {np.mean(hacker_rm_scores):>15.3f}")
print(f"  {'Actual correctness':<25} | {np.mean(honest_true_scores):>15.1%} | {np.mean(hacker_true_scores):>15.1%}")
print()
print("  ⚠️  The hacker gets a HIGHER RM score but a LOWER true accuracy.")
print("  This is reward hacking: optimising the proxy at the expense of truth.")
print()
print("  Mitigations:")
print("    1. Ensemble multiple RMs (harder to game all simultaneously)")
print("    2. Iterative RM retraining on samples from the hacker policy")
print("    3. Separate held-out evaluation that is NEVER used as training signal")
print("    4. Use verifiable rewards wherever possible (eliminate the RM)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Training stage comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Stage-by-stage performance gains (simulated)")
print("━" * 65)
print()

# Simulated benchmark trajectories for each post-training stage
# Scores are illustrative, inspired by published Llama / InstructGPT numbers
stages = [
    ("Pre-trained",          0.25,  0.48, 0.22,  0.08),
    ("+ SFT",                0.55,  0.62, 0.48,  0.22),
    ("+ RLHF (PPO)",         0.72,  0.70, 0.61,  0.31),
    ("+ DPO (instead)",      0.70,  0.69, 0.60,  0.30),
    ("+ RLVR (reasoning)",   0.74,  0.73, 0.79,  0.55),
    ("+ Safety FT",          0.71,  0.71, 0.77,  0.53),
    ("+ Agentic FT",         0.73,  0.73, 0.78,  0.54),
]

headers = ["Stage", "MT-Bench", "Helpful%", "GSM8K", "MATH"]
print(f"  {headers[0]:<28} | {headers[1]:>9} | {headers[2]:>9} | {headers[3]:>7} | {headers[4]:>7}")
print(f"  {'─'*70}")

for stage, mt, helpful, gsm, math_score in stages:
    print(f"  {stage:<28} | {mt:>9.2f} | {helpful:>9.1%} | {gsm:>7.1%} | {math_score:>7.1%}")

print()
print("  Notes:")
print("   • SFT: biggest jump in helpfulness; modest boost in math (CoT format)")
print("   • RLHF/DPO: similar helpfulness; RLHF slightly better (online exploration)")
print("   • RLVR: dramatic jump in math/reasoning; slight helpfulness dip (tradeoff)")
print("   • Safety FT: slight helpfulness reduction (refusals cost some benign cases)")
print("   • Agentic FT: small general improvement; large gain on agent benchmarks (not shown)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Post-training cookbook and decision guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Post-training cookbook")
print("━" * 65)
print()

COOKBOOK = """
  POST-TRAINING PIPELINE: COMPLETE DECISION GUIDE

  STEP 1: SFT
  ─────────────────────────────────────────────
  □ Collect/curate (instruction, response) pairs
  □ Prioritise quality over quantity (1K excellent > 50K mediocre)
  □ Format with consistent chat template
  □ Set loss mask to RESPONSE TOKENS ONLY
  □ Train: LR=1e-5, epochs=1-3, cosine decay, clip=1.0
  □ Verify: overfit single batch → loss → 0 (sanity check)

  STEP 2: CHOOSE YOUR RL PATH
  ─────────────────────────────────────────────
  Task has verifiable ground truth (math, code)?
      → RLVR (GRPO/PPO with binary verifier)
      → No RM needed. Much simpler and more principled.

  Task is subjective (writing, helpfulness, style)?
      → Collect preference pairs (chosen, rejected)
      → Start with DPO (simpler, stable)
      → Upgrade to online DPO / PPO if DPO plateaus

  Budget is very tight?
      → DPO only. Skip the RM entirely.

  STEP 3: PREFERENCE DATA COLLECTION
  ─────────────────────────────────────────────
  □ Sample 2 responses per prompt from the CURRENT model
  □ Have raters pick the better response (not rate absolutely)
  □ Aim for high agreement between raters (κ > 0.6)
  □ Balance: task types, difficulty levels, lengths
  □ Minimum: 10K pairs for DPO. 100K+ for RM training.

  STEP 4: REWARD MODEL (if using RLHF/PPO)
  ─────────────────────────────────────────────
  □ Init from SFT checkpoint
  □ Add linear head on last token hidden state
  □ Train with Bradley-Terry loss: -log σ(r_w - r_l)
  □ LR=1e-5, epochs=1-2, watch for overfitting on noisy pairs
  □ Validate: does the RM actually rank your gold examples correctly?
  □ Check: is RM reward correlated with human ratings (Pearson ≥ 0.6)?

  STEP 5: PREFERENCE OPTIMISATION
  ─────────────────────────────────────────────
  DPO:
  □ β = 0.1 (tune: too small → collapse, too large → no change)
  □ LR = 5e-7 to 2e-6 (very small — we are nudging, not rewriting)
  □ Epochs = 1-3
  □ Monitor: KL from reference (should grow slowly; alert if KL > 20)

  PPO:
  □ β (KL penalty) = 0.02 to 0.1
  □ ε (clip) = 0.2
  □ Value loss coeff = 0.1 to 0.5
  □ Rollout batch >> update batch (collect more data than you use each step)
  □ Monitor: reward_mean, kl_from_ref, grad_norm, response_length

  RLVR:
  □ GRPO: K=8-16 rollouts per problem, no value head
  □ Reward: binary (correct/incorrect), no partial credit initially
  □ Think format: prompt with <thinking>...</thinking><answer>...</answer>
  □ Watch for: reasoning collapse (model skips thinking), length explosion

  STEP 6: SAFETY FINE-TUNING
  ─────────────────────────────────────────────
  □ Run red-teaming to find refusal failures and over-refusals
  □ Collect (harmful_prompt, safe_refusal) SFT pairs
  □ Collect preference pairs: (safe_refusal, harmful_compliance)
  □ Run DPO with safety-specific preference data
  □ Evaluate: ToxiGen, TruthfulQA, in-house adversarial set
  □ Balance: check helpfulness did not regress significantly

  STEP 7: ITERATE
  ─────────────────────────────────────────────
  □ Generate new outputs from the post-trained model
  □ Collect human ratings on the NEW model's outputs
  □ Retrain RM on new ratings (old ratings become stale)
  □ Run another round of DPO/PPO
  □ Repeat until MT-Bench/human eval plateaus

  COMMON FAILURE MODES AND FIXES:
  ─────────────────────────────────────────────
  Sycophancy:       Add contrastive SFT where model DISAGREES correctly.
  Length inflation: Normalise rewards by response length in RM training.
  Reward hacking:   Ensemble RMs + iterative RM retraining.
  Forgetting:       Lower LR, add pre-training data to SFT mix, KL penalty.
  Over-refusal:     Add positive SFT examples for borderline benign queries.
  Repetition:       Presence penalty at inference; check for repetition in SFT data.
"""
print(COOKBOOK)

print("━" * 65)
print("  POST-TRAINING QUICK REFERENCE")
print("━" * 65)
print()
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ Question                         │ Answer                    │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ Models needed for PPO-RLHF?      │ 4: actor, critic, ref, RM │")
print("  │ Models needed for DPO?           │ 2: policy + reference     │")
print("  │ Models needed for RLVR?          │ 1-2: policy + ref (opt.)  │")
print("  │ SFT learning rate?               │ 1e-5 to 2e-5              │")
print("  │ DPO learning rate?               │ 5e-7 to 2e-6              │")
print("  │ DPO β range?                     │ 0.05 to 0.3               │")
print("  │ PPO KL penalty β range?          │ 0.01 to 0.1               │")
print("  │ GRPO rollouts per problem (K)?   │ 8 to 16                   │")
print("  │ Reward hacking first sign?       │ Responses get longer/     │")
print("  │                                  │ more sycophantic          │")
print("  │ DPO vs PPO: when to upgrade?     │ When DPO plateaus and     │")
print("  │                                  │ online data would help    │")
print("  │ Safety FT hurts helpfulness?     │ Slightly — balance via    │")
print("  │                                  │ harmless + helpful pairs  │")
print("  │ SFT data minimum (quality)?      │ 1K excellent examples     │")
print("  │ Preference data minimum?         │ 10K pairs (DPO min)       │")
print("  └──────────────────────────────────────────────────────────────┘")
''',
    },
}

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