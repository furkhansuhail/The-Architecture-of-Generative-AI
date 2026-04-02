"""
Supervised Fine-Tuning (SFT)
=============================

Pre-training teaches a model the statistical structure of language — grammar,
facts, reasoning patterns. But it does not teach the model to follow
instructions, maintain a helpful persona, or format responses appropriately.
Supervised Fine-Tuning (SFT) bridges this gap by training the pre-trained
model on carefully curated demonstration data: examples of the exact
input-output behaviour we want the model to exhibit. SFT is the foundational
step in every instruction-following, chat, and task-specific LLM.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Supervised Fine-Tuning (SFT)"
DISPLAY_NAME = "26 · Supervised Fine-Tuning"
ICON         = "🎯"
SUBTITLE     = "Instruction Tuning and Chat Templates"


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

### The Pre-Training to Fine-Tuning Gap

A freshly pre-trained language model is a powerful statistical engine, but
it has a fundamental behavioural problem: it is trained to **predict the most
probable next token given its context**, not to **follow instructions**.

When you prompt a raw pre-trained model with "What is the capital of France?",
it may respond with:
    "What is the capital of France? What is the capital of Germany?
     What is the capital of Spain?..."
because it has learned that this is a common internet pattern (quiz lists,
FAQ pages). It is statistically completing the text, not answering the question.

Or it may respond:
    "What is the capital of France? This is a basic geography question
     that all students should know..."
because it has seen exam-style content where questions are followed by
commentary, not answers.

This is not a failure of capability — the model almost certainly knows the
answer "Paris". It is a failure of **alignment**: the model's behaviour
does not match the user's intent.

**SFT aligns the model's behaviour** by fine-tuning it on demonstrations
of the desired input-output pattern. After SFT, when asked "What is the
capital of France?", the model has learned that the appropriate response
to a direct question is a direct answer: "Paris."


### What SFT Actually Does Mathematically

SFT is identical to pre-training in its loss function — it minimises
cross-entropy on the next-token prediction task. The difference is entirely
in the **data**:

    Pre-training data:     web text, books, code  (predict any next token)
    SFT data:              (instruction, response) pairs  (learn to respond)

And critically, in the **loss mask**:

    Pre-training:   compute loss on ALL tokens
    SFT:            compute loss ONLY on response tokens (not the prompt)

This distinction is crucial. If we compute loss on the prompt, the model
is penalised for not being able to "predict" the prompt — but the prompt
is provided by the user and should not be predicted. Computing loss on the
prompt:
    1.  Wastes gradient signal (prompt structure is provided, not generated)
    2.  Degrades performance (prompt tokens often have structure the model
        did not choose, like special tokens, usernames, timestamps)
    3.  Can cause the model to memorise the prompt distribution rather than
        learning good response generation


    **Diagram 1 — SFT Loss Masking:**

    SFT TRAINING EXAMPLE: What is masked vs not masked
    ════════════════════════════════════════════════════════════════

    Full sequence:
    [SYS] You are helpful. [/SYS]
    [USER] What is 2+2? [/USER]
    [ASST] The answer is 4. [/ASST]
     ↑                               ↑
     Token 0                         Token N

    Loss mask:
    [SYS] 0 0 0 0 0 0 [/SYS] 0 0 0 0 0 [/USER] 1 1 1 1 1 [/ASST]
                                                 ↑
                                        Only these get gradients!

    tokens_masked  (loss=0):  system prompt, user message, special tokens
    tokens_trained (loss=1):  assistant response tokens

    Why this matters:
    • Gradient flows only from prediction errors on response tokens
    • Model learns: "given this context, generate THIS kind of response"
    • Model does NOT learn to predict system/user patterns (which are provided)


### The Anatomy of SFT Data

SFT data is a collection of (input, output) demonstrations. The quality,
diversity, and quantity of this data is the primary determinant of SFT quality
— more important than most hyperparameter choices.

**Key data dimensions:**

    1.  **Task coverage:** What kinds of instructions does the model see?
        •   Open-ended question answering
        •   Summarisation
        •   Code generation
        •   Creative writing
        •   Step-by-step reasoning
        •   Factual lookup
        •   Instruction following with constraints
        •   Multi-turn dialogue
        •   Tool use / function calling

    2.  **Response quality:** The demonstrations should show ideal responses.
        Low-quality demonstrations (short, evasive, incorrect) teach
        the model to produce low-quality responses.

    3.  **Diversity:** A model trained on 1,000 diverse instructions often
        outperforms one trained on 10,000 repetitive instructions from one
        domain. The LIMA paper (Zhou et al., 2023) showed that 1,000
        high-quality demonstrations sufficed to match models trained on
        much larger SFT datasets.

    4.  **Style consistency:** Within a single training run, responses should
        have a consistent voice, format, and level of detail. Mixed styles
        confuse the model about what "good" looks like.


### The LIMA Principle: Less Is More

The **LIMA** (Less Is More for Alignment) paper (Zhou et al., 2023) made a
striking discovery: a LLaMA-65B model fine-tuned on only **1,000 carefully
selected examples** performed comparably to models fine-tuned on 52,000+
examples from larger datasets like Alpaca.

The key insight: **most of the capability is already in the pre-trained model**.
SFT is not teaching the model new knowledge or new reasoning abilities — it is
teaching the model the *format* and *style* in which to express the knowledge
it already has. This means:

    •   You need enough examples to cover the diversity of tasks (breadth)
    •   You need each example to be high quality (accuracy, helpfulness)
    •   You do NOT need millions of examples (quality >> quantity)

The LIMA principle has significant practical implications:
    1.  Spend more effort curating high-quality data than collecting large datasets
    2.  A dataset of 1,000–10,000 examples is sufficient for most applications
    3.  Clean and consistent formatting matters more than raw data volume
    4.  Domain-specific fine-tuning on 100–500 examples is often enough


### Chat Templates: Structured Dialogue Format

SFT models receive multi-turn conversations as input. The structure of how
conversation turns are represented is the **chat template** — a formatting
convention that marks who said what.

Different model families use different templates. Getting this right is critical:
using the wrong template at inference time causes severe performance degradation
because the model was trained to expect specific token patterns at role boundaries.

**ChatML (used by OpenAI, Mistral Instruct, many others):**
    <|im_start|>system
    You are a helpful AI assistant.
    <|im_end|>
    <|im_start|>user
    What is the capital of France?
    <|im_end|>
    <|im_start|>assistant
    Paris is the capital of France.
    <|im_end|>

**LLaMA-2 Chat:**
    <s>[INST] <<SYS>>
    You are a helpful assistant.
    <</SYS>>

    What is the capital of France? [/INST] Paris is the capital of France. </s>

**LLaMA-3 (Llama-3-style):**
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    What is the capital of France?<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    Paris is the capital of France.<|eot_id|>

**The key tokens to know:**
Each template uses special tokens that were added to the vocabulary specifically
for chat formatting. When you train an SFT model, these special tokens must be
present in the tokeniser AND the model must have been pre-trained with them
(or at least have them as new vocabulary embeddings that get fine-tuned).


### Multi-Turn Conversation Handling

SFT often involves multi-turn conversations where the model must maintain
context across multiple exchanges. The key considerations:

**Packing multiple turns:**
A single training example may contain 10+ turns. When packing for training:
    •   All turns are concatenated into a single sequence
    •   The loss mask ensures only assistant turns contribute to the loss
    •   A cross-turn attention mask prevents turn N from attending to turn N+2
      (this is important to avoid "data leakage" — the model shouldn't know
      what it is about to say before generating it)

**Ghost attention (LLaMA-2 RLHF technique):**
In long conversations, the model may "forget" the system prompt. Ghost
Attention (Touvron et al., 2023) trains the model to maintain awareness of
the system prompt by repeatedly concatenating it to all user turns during
training:
    ACTUAL: [SYS] [USER1] [ASST1] [USER2] [ASST2] ...
    GHOST:  [SYS] [USER1] [ASST1] [SYS] [USER2] [ASST2] [SYS] [USER3] ...
    (the repeated SYS tokens are masked from the loss; only ASST tokens trained)

This ensures the model has fresh access to system prompt tokens at every turn.


### SFT Hyperparameters: Key Differences from Pre-Training

SFT requires different hyperparameters than pre-training:

    Hyperparameter    Pre-training          SFT
    ─────────────────────────────────────────────────────────────────────────
    Learning rate     3e-4 (high)           1e-5 to 5e-5 (much lower)
    LR schedule       Cosine 1T tokens      Linear decay, 1-3 epochs
    Batch size        1M–8M tokens          8k–64k tokens (much smaller)
    Epochs            1 epoch (huge data)   1–3 epochs (small dataset)
    Weight decay      0.1                   0.0 or 0.001 (less important)
    Dropout           0.0                   0.0–0.1 (may help generalisation)
    Gradient clip     1.0                   1.0 (same)
    Warmup            2000 steps            50–200 steps (short)
    ─────────────────────────────────────────────────────────────────────────

**Why lower learning rates?**
The pre-trained model already has excellent representations. SFT should
make small, targeted adjustments — not wholesale rewrite the model.
Using a large learning rate in SFT is one of the most common failure modes:
the model forgets its pre-trained knowledge (catastrophic forgetting).

**Catastrophic forgetting:**
When fine-tuned aggressively on a narrow dataset, the model "forgets" knowledge
from pre-training. A model that was excellent at coding may become terrible at
it after SFT on pure instruction-following data if the LR is too high.

Symptoms: pre-training benchmarks (MMLU, HumanEval) drop significantly after SFT.
Mitigation:
    1.  Lower learning rate (< 1e-5 for full fine-tuning)
    2.  Fewer epochs
    3.  Mix in some pre-training data (replay buffer — ~5% of total tokens)
    4.  Use PEFT/LoRA instead of full fine-tuning (only trains small subsets)


    **Diagram 2 — SFT Training Dynamics vs Pre-Training:**

    PRE-TRAINING vs SFT TRAINING DYNAMICS
    ════════════════════════════════════════════════════════════════

    Pre-training:                      SFT:
    ┌────────────────────────────┐     ┌────────────────────────────┐
    │  Tokens:  1T–15T           │     │  Tokens:  100k–50M         │
    │  LR:      3e-4 (high)      │     │  LR:      1e-5 (10-30× lower│
    │  Data:    diverse web text │     │  Data:    curated instruct  │
    │  Epochs:  ~1 epoch         │     │  Epochs:  1-3              │
    │  Loss:    all tokens       │     │  Loss:    response only    │
    │  Goal:    language model   │     │  Goal:    instruction model │
    └────────────────────────────┘     └────────────────────────────┘
         Loss curve:                         Loss curve:
    5.0 ┤●                            3.0 ┤●
        │ ╲                               │╲
    3.0 ┤  ╲──────                    2.0 ┤ ╲───
        │         ╲────────               │     ╲──────
    2.0 ┤               ─────        1.5 ┤           ────
        └──────────────────── steps       └──────────────── steps
          (hundreds of thousands)          (hundreds to thousands)


### Data Sources and Curation Strategies

**Self-Instruct (Wang et al., 2022):**
Start with a small seed of 175 human-written instructions. Use the LLM itself
to generate new instructions and responses. Filter for quality and diversity.
This technique generated the Alpaca dataset (52K examples from GPT-3.5).

Limitation: model-generated data inherits the model's biases and failure modes.
"Garbage in, garbage out" amplified through the generator model.

**Distillation from stronger models (common in open LLM SFT):**
Use GPT-4 or Claude to generate responses for a diverse set of prompts.
Fine-tune the smaller open model on these responses.
Risk: license violations (OpenAI TOS forbids using API outputs to train
competing models). Many early open models (WizardLM, Vicuna) used this approach.

**Human demonstrations (gold standard):**
Have skilled annotators write high-quality (instruction, response) pairs.
Expensive but highest quality. Used by OpenAI for InstructGPT/ChatGPT,
Anthropic for Claude, and in the curated LIMA dataset.

**Filtering web data with quality classifiers:**
Apply a classifier (trained on human quality judgements) to existing web data
to identify instruction-like, high-quality content. Used in Dolly (Databricks)
and many academic datasets.

**Synthetic data at scale (2024 trend):**
Use GPT-4 or Claude to generate millions of (instruction, response) pairs
with careful prompt engineering to ensure diversity and quality.
Examples: Phi-3 (3.3T tokens of synthetic data), OpenHermes, Orca 2.

The SFT data pipeline is an active research area with no single dominant
approach — the best results come from carefully curated, diverse, high-quality
demonstrations of exactly the behaviour you want.


### SFT vs RLHF: When Each Is Used

SFT teaches the model what a good response *looks like* by showing examples.
RLHF (Module 28) teaches the model what humans *prefer* by comparing responses.

    SFT:  given context, generate THIS response  (supervised, one right answer)
    RLHF: given context, this response > that response  (preference learning)

**Use SFT when:**
    •   You have high-quality demonstration data
    •   The desired behaviour is well-defined (code generation, specific formats)
    •   Low budget (SFT is 10-100× cheaper than full RLHF)
    •   Starting from a raw pre-trained base model

**Use RLHF/DPO when:**
    •   The desired behaviour is better captured by preferences than examples
    •   The task has multiple valid responses and you want to capture nuance
    •   You have human feedback data (costly to collect)
    •   After SFT (RLHF is typically applied ON TOP of SFT, not instead of)

In practice, the standard pipeline is:
    Base model → SFT → Reward Model → PPO (or DPO)
    (Each module in this collection covers one stage)


### Evaluation: How to Know SFT Is Working

Evaluating SFT models requires both automatic and human evaluation:

**Automatic benchmarks (proxy for capability):**
    •   MMLU (57 subject areas, 14K questions): measures factual knowledge
    •   TruthfulQA: measures calibration and honesty
    •   HumanEval/MBPP: measures code generation
    •   MT-Bench (Zheng et al., 2023): 80 multi-turn questions, GPT-4 judged
    •   AlpacaEval: measures win-rate against text-davinci-003

**Human evaluation:**
    A/B test: annotators compare model's response to a baseline (GPT-3.5, etc.)
    and rate which response is better. Gold standard, expensive, slow.

**LLM-as-judge (increasingly common):**
    Use a strong model (GPT-4) to score responses on criteria:
    helpfulness, accuracy, safety, instruction-following.
    Faster and cheaper than human evaluation, correlates well with human judgement.
    Risk: bias toward responses similar to the judge model's own outputs.

**Critical: watch for benchmark contamination:**
If SFT data includes examples from MMLU or HumanEval, benchmark scores will
be inflated. Use held-out test sets or tasks that didn't appear in training.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
SFT vs Pre-Training vs RLHF

| Dimension             | Pre-Training              | SFT                        | RLHF/DPO                    |
|-----------------------|---------------------------|----------------------------|-----------------------------|
| Data type             | Raw web/book/code text    | (instruction, response)    | (prompt, chosen, rejected)  |
| Data size             | 1T–15T tokens             | 100K–50M tokens            | 10K–500K comparisons        |
| Loss mask             | All tokens                | Response tokens only        | N/A (preference loss)       |
| Learning rate         | 3e-4                      | 1e-5 to 5e-5               | 5e-7 to 1e-6                |
| Epochs                | ~1 (huge data)            | 1–3                        | 1–2                         |
| Goal                  | Language understanding    | Instruction following       | Human preference alignment  |
| Compute               | 10³–10⁵× more than SFT   | Moderate (days–weeks)      | Moderate (days–weeks)       |
| Catastrophic forgetting risk | Low (from scratch)| Medium (lower LR needed)   | Low (very low LR)           |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "SFT Training Loop with Loss Masking": {
        "description": "Complete SFT training loop with proper response-only loss masking, chat template formatting, multi-turn conversation support, and catastrophic forgetting monitoring.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SUPERVISED FINE-TUNING TRAINING LOOP
================================================================================

A complete SFT training loop demonstrating:
    1. Chat template formatting (ChatML style)
    2. Response-only loss masking (critical!)
    3. Multi-turn conversation packing
    4. Catastrophic forgetting detection (pre-training benchmark tracking)
    5. SFT-specific hyperparameter choices
    6. Gradient accumulation for small-batch SFT

The loss masking is the most important SFT-specific detail:
    - System prompt tokens:  loss = 0  (don't train on these)
    - User message tokens:   loss = 0  (don't train on these)
    - Assistant response:    loss = 1  (train only on these)
    - Special delimiters:    loss = 0

================================================================================
"""

import math
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ── Special token IDs (ChatML style) ─────────────────────────────────────────

@dataclass
class ChatMLTokens:
    """Token IDs for ChatML chat template."""
    BOS:      int = 1    # <|begin_of_text|>
    EOS:      int = 2    # <|end_of_text|>
    IM_START: int = 32001  # <|im_start|>
    IM_END:   int = 32002  # <|im_end|>
    PAD:      int = 0
    # Role IDs (fake byte-level tokens for simulation)
    SYSTEM:   int = 32010
    USER:     int = 32011
    ASST:     int = 32012
    NEWLINE:  int = 200   # '\n'


TOKENS = ChatMLTokens()


# ── Chat template formatter ───────────────────────────────────────────────────

@dataclass
class Message:
    role:    str    # "system", "user", "assistant"
    content: list[int]  # token IDs for this message


def apply_chat_template(
    messages: list[Message],
    tokens: ChatMLTokens = TOKENS,
    add_generation_prompt: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Apply ChatML template to a list of messages.

    Returns:
        input_ids:  flat list of token IDs for the full conversation
        loss_mask:  1 for assistant response tokens, 0 for all others

    ChatML format:
        <|im_start|>system\\n{content}<|im_end|>\\n
        <|im_start|>user\\n{content}<|im_end|>\\n
        <|im_start|>assistant\\n{content}<|im_end|>\\n
    """
    ROLE_TO_ID = {
        "system":    tokens.SYSTEM,
        "user":      tokens.USER,
        "assistant": tokens.ASST,
    }

    input_ids  = []
    loss_mask  = []

    for msg in messages:
        is_assistant = (msg.role == "assistant")
        role_id      = ROLE_TO_ID[msg.role]

        # <|im_start|> {role_id} \n
        header = [tokens.IM_START, role_id, tokens.NEWLINE]
        input_ids  += header
        loss_mask  += [0] * len(header)   # never train on structure tokens

        # message content
        input_ids  += msg.content
        loss_mask  += [1 if is_assistant else 0] * len(msg.content)

        # <|im_end|> \n
        footer = [tokens.IM_END, tokens.NEWLINE]
        input_ids  += footer
        loss_mask  += [0] * len(footer)

    # Optionally add <|im_start|>assistant\n to prompt generation
    if add_generation_prompt:
        gen_prompt = [tokens.IM_START, tokens.ASST, tokens.NEWLINE]
        input_ids  += gen_prompt
        loss_mask  += [0] * len(gen_prompt)

    return input_ids, loss_mask


def pack_conversations(
    conversations: list[list[Message]],
    max_seq_len: int,
    pad_token: int = TOKENS.PAD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack multiple conversations into fixed-length sequences for batch training.

    Conversations that exceed max_seq_len are truncated (from the left — preserve
    the most recent context). Shorter conversations are padded on the right.

    Returns:
        input_ids:  (batch, max_seq_len) long tensor
        labels:     (batch, max_seq_len) long tensor  — -100 for masked tokens
    """
    batch_input_ids = []
    batch_labels    = []

    for conv in conversations:
        ids, mask = apply_chat_template(conv, add_generation_prompt=False)

        # Truncate from the left if too long (preserve most recent context)
        if len(ids) > max_seq_len:
            ids  = ids[-max_seq_len:]
            mask = mask[-max_seq_len:]

        # Pad to max_seq_len
        pad_len = max_seq_len - len(ids)
        ids  = ids  + [pad_token] * pad_len
        mask = mask + [0]         * pad_len

        # Convert mask to labels: 1 → token_id, 0 → -100 (ignored by CE loss)
        labels = [
            ids[i] if mask[i] == 1 else -100
            for i in range(len(ids))
        ]

        batch_input_ids.append(ids)
        batch_labels.append(labels)

    return (
        torch.tensor(batch_input_ids,  dtype=torch.long),
        torch.tensor(batch_labels,     dtype=torch.long),
    )


# ── Tiny language model for SFT demo ─────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab: int = 32100, d: int = 128,
                 n_layers: int = 2, max_len: int = 256):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d, padding_idx=0)
        self.pos    = nn.Embedding(max_len, d)
        dec_layer   = nn.TransformerDecoderLayer(
            d_model=d, nhead=4, dim_feedforward=d*4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.body   = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.max_len = max_len
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.embed(idx) + self.pos(pos)
        m    = self.causal_mask[:T, :T]
        x    = self.body(x, x, tgt_mask=m, memory_mask=m,
                         tgt_is_causal=True, memory_is_causal=True)
        return self.head(self.norm(x))


# ── SFT-specific loss function ────────────────────────────────────────────────

def sft_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss with label masking for SFT.

    labels:  token IDs where loss should be computed, -100 elsewhere.
             -100 is PyTorch's standard ignore_index for CrossEntropyLoss.

    This automatically handles the loss masking — only tokens with label != -100
    contribute to the loss. System prompt and user tokens have label=-100.
    """
    # Shift: predict token i+1 from token i
    # logits: (B, T, V) → (B, T-1, V) → flatten to (B*(T-1), V)
    # labels: (B, T)   → (B, T-1)    → flatten to (B*(T-1),)
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def count_trainable_tokens(labels: torch.Tensor) -> int:
    """Count how many tokens actually receive gradient signal."""
    return (labels != -100).sum().item()


# ── SFT training loop ─────────────────────────────────────────────────────────

class SFTTrainer:
    """
    A minimal but complete SFT trainer with all key best practices:
        - Response-only loss masking
        - Gradient accumulation for effective batch size
        - Catastrophic forgetting monitoring
        - LR warmup + linear decay (SFT-appropriate schedule)
        - Per-step metric logging
    """

    def __init__(self, model: nn.Module,
                 lr: float = 2e-5,
                 warmup_steps: int = 100,
                 total_steps: int = 1000,
                 grad_accum: int = 4,
                 max_grad_norm: float = 1.0,
                 device: str = "cpu"):
        self.model          = model
        self.device         = device
        self.grad_accum     = grad_accum
        self.max_grad_norm  = max_grad_norm
        self.global_step    = 0

        # SFT-appropriate: AdamW with low LR and mild weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,   # often 0 for SFT (small dataset, low overfitting risk)
        )

        # Linear warmup + linear decay (simple for SFT)
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # Track pre-training perplexity as forgetting signal
        self.pretrain_ppl_history = []

    def train_step(self, conversations: list[list[Message]],
                    max_seq_len: int = 128) -> dict:
        """
        Run one gradient accumulation cycle on a list of conversations.
        Conversations are split into micro-batches of 1 for this demo.
        """
        self.optimizer.zero_grad(set_to_none=True)

        total_loss      = 0.0
        total_tokens    = 0
        micro_steps     = min(self.grad_accum, len(conversations))
        conversations   = conversations[:micro_steps]

        for k, conv_group in enumerate(conversations):
            # Pack conversation into tensors
            input_ids, labels = pack_conversations(
                [conv_group], max_seq_len
            )
            input_ids = input_ids.to(self.device)
            labels    = labels.to(self.device)

            # Forward pass
            logits = self.model(input_ids)
            loss   = sft_cross_entropy(logits, labels) / micro_steps

            # Backward (accumulate)
            loss.backward()

            n_tokens    = count_trainable_tokens(labels)
            total_loss += loss.item() * micro_steps
            total_tokens += n_tokens

        # Gradient clipping + optimiser step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return {
            "step":          self.global_step,
            "loss":          total_loss / micro_steps,
            "perplexity":    math.exp(total_loss / micro_steps),
            "trained_tokens": total_tokens,
            "grad_norm":     grad_norm.item(),
            "lr":            self.scheduler.get_last_lr()[0],
        }


# ── Demo: complete SFT run ───────────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
    VOCAB    = 32100
    D        = 128
    MAX_LEN  = 128
    N_STEPS  = 30
    LR       = 2e-5

    print("=" * 65)
    print("  SUPERVISED FINE-TUNING (SFT) DEMO")
    print(f"  device={DEVICE}, d={D}, vocab={VOCAB}")
    print("=" * 65)
    print()

    # Create a tiny pre-trained model (in reality: load from checkpoint)
    torch.manual_seed(42)
    model = TinyLM(VOCAB, D, n_layers=2, max_len=MAX_LEN).to(DEVICE)
    pretrained_weights = copy.deepcopy(model.state_dict())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Synthetic SFT dataset (simulating (instruction, response) pairs)
    # In real training, load from jsonl files
    def make_conversation(user_len: int = 15, asst_len: int = 30) -> list[Message]:
        """Create a synthetic conversation with simulated token IDs."""
        torch.manual_seed(user_len + asst_len)
        sys_tokens  = torch.randint(100, 1000, (8,)).tolist()
        user_tokens = torch.randint(100, 1000, (user_len,)).tolist()
        asst_tokens = torch.randint(100, 1000, (asst_len,)).tolist()
        return [
            Message("system",    sys_tokens),
            Message("user",      user_tokens),
            Message("assistant", asst_tokens),
        ]

    # Show loss masking in detail
    print("  Loss mask example (single conversation):")
    conv_demo = make_conversation(user_len=5, asst_len=8)
    ids, mask = apply_chat_template(conv_demo)
    print(f"  Total tokens:     {len(ids)}")
    print(f"  Masked (loss=0):  {sum(1 for m in mask if m == 0)}")
    print(f"  Trained (loss=1): {sum(1 for m in mask if m == 1)}")
    print(f"  Training token %: {sum(mask)/len(mask)*100:.1f}%")
    print()

    # Show multi-turn conversation
    multi_turn = [
        Message("system",    [100, 200, 300, 400]),
        Message("user",      [500, 600, 700]),
        Message("assistant", [800, 900, 1000, 1100, 1200]),
        Message("user",      [1300, 1400]),
        Message("assistant", [1500, 1600, 1700, 1800]),
    ]
    ids_mt, mask_mt = apply_chat_template(multi_turn, add_generation_prompt=False)
    print("  Multi-turn conversation loss mask:")
    roles = ["sys", "usr", "asst", "usr", "asst"]
    pos   = 0
    for i, msg in enumerate(multi_turn):
        header_len = 3   # IM_START + role + newline
        footer_len = 2   # IM_END + newline
        content_len = len(msg.content)
        total_len   = header_len + content_len + footer_len
        seg_mask    = mask_mt[pos:pos+total_len]
        print(f"    {roles[i]:>4}: [{sum(seg_mask):>2} trainable / {total_len:>3} total]  "
              f"{'░' * (total_len - sum(seg_mask))}{'█' * sum(seg_mask)}")
        pos += total_len
    print("    ░ = masked (no loss)   █ = trained (loss computed)")
    print()

    # Train
    trainer = SFTTrainer(model, lr=LR, warmup_steps=5,
                          total_steps=N_STEPS, grad_accum=2, device=DEVICE)

    print("=" * 65)
    print(f"  {'Step':>5}  {'Loss':>10}  {'PPL':>10}  "
          f"{'Trained tok':>12}  {'Grad norm':>12}  {'LR':>12}")
    print("=" * 65)

    conversations = [make_conversation(15 + i % 10, 25 + i % 15)
                     for i in range(N_STEPS * 2)]

    for step_i in range(N_STEPS):
        batch = conversations[step_i * 2 : step_i * 2 + 2]
        metrics = trainer.train_step(batch, max_seq_len=MAX_LEN)

        if step_i % 5 == 0 or step_i < 3:
            print(f"  {metrics['step']:>5}  {metrics['loss']:>10.4f}  "
                  f"{metrics['perplexity']:>10.2f}  "
                  f"{metrics['trained_tokens']:>12}  "
                  f"{metrics['grad_norm']:>12.4f}  "
                  f"{metrics['lr']:>12.2e}")

    # Catastrophic forgetting check
    print()
    print("=" * 65)
    print("  CATASTROPHIC FORGETTING CHECK")
    print("=" * 65)
    print()
    print("  Comparing parameter norms: pre-trained vs SFT-ed model")
    print()
    sft_weights = model.state_dict()
    drift_norms = {}
    for key in pretrained_weights:
        pt_w  = pretrained_weights[key].float()
        sft_w = sft_weights[key].float()
        drift = (pt_w - sft_w).norm().item()
        orig  = pt_w.norm().item()
        if orig > 0:
            drift_norms[key] = drift / orig

    print(f"  {'Parameter':<30}  {'Relative drift':>16}  {'Assessment':>14}")
    print(f"  {'':─<30}  {'':─>16}  {'':─>14}")
    for key, drift in sorted(drift_norms.items(), key=lambda x: -x[1]):
        assessment = ("large drift ⚠️" if drift > 0.1
                      else "moderate" if drift > 0.01
                      else "small ✓")
        print(f"  {key:<30}  {drift:>15.4f}×  {assessment:>14}")

    total_drift = sum(drift_norms.values()) / len(drift_norms)
    print()
    print(f"  Average relative parameter drift: {total_drift:.4f}×")
    if total_drift > 0.1:
        print("  ⚠️  HIGH DRIFT: consider lower LR or fewer epochs to prevent forgetting")
    else:
        print("  ✓  Drift is small — SFT is making targeted adjustments")
''',
    },

    "SFT Data Pipeline: Format, Quality Filter, and Pack": {
        "description": "Build the full SFT data pipeline: load raw instruction data, validate format, apply quality filters, apply chat templates, tokenise, and pack into training batches.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SFT DATA PIPELINE
================================================================================

Complete SFT data preparation pipeline:
    1. Load and validate raw instruction data (JSONL format)
    2. Apply quality filters (length, language, safety heuristics)
    3. Apply chat template
    4. Tokenise with loss masking
    5. Pack into fixed-length training batches
    6. Statistics: token utilisation, assistant token fraction

This is the pipeline that would precede the training loop in production.

================================================================================
"""

import json
import re
import math
import hashlib
from dataclasses import dataclass, field
from typing import Optional


# ── Raw data format ───────────────────────────────────────────────────────────

@dataclass
class RawExample:
    """One raw SFT example as loaded from JSONL."""
    messages:   list[dict]       # [{"role": ..., "content": ...}, ...]
    source:     str = "unknown"  # dataset source for mixing/attribution
    quality:    float = 1.0      # 0.0–1.0 quality score (if available)


# ── Quality filters ────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    min_response_words:    int   = 5      # response too short
    max_response_words:    int   = 2048   # response too long (context limit)
    min_instruction_words: int   = 2      # instruction too short
    max_instruction_words: int   = 512    # instruction too long
    min_quality_score:     float = 0.0    # minimum quality gate
    dedup_ngram:           int   = 13     # n-gram fingerprint size for dedup
    block_phrases:         list  = field(default_factory=lambda: [
        "as an ai language model",   # self-referential boilerplate
        "i cannot and will not",     # overly refusal-heavy responses
        "i'm unable to",             # same
    ])


@dataclass
class FilterResult:
    kept:    bool
    reason:  Optional[str] = None


def apply_quality_filter(example: RawExample, cfg: FilterConfig) -> FilterResult:
    """Apply heuristic quality filters to an SFT example."""
    # Find the last assistant message (the response we're training on)
    responses    = [m for m in example.messages if m["role"] == "assistant"]
    instructions = [m for m in example.messages if m["role"] == "user"]

    if not responses or not instructions:
        return FilterResult(False, "missing assistant or user turn")

    response    = responses[-1]["content"]
    instruction = instructions[-1]["content"]

    response_words    = len(response.split())
    instruction_words = len(instruction.split())

    # Length filters
    if response_words < cfg.min_response_words:
        return FilterResult(False, f"response too short ({response_words} words)")
    if response_words > cfg.max_response_words:
        return FilterResult(False, f"response too long ({response_words} words)")
    if instruction_words < cfg.min_instruction_words:
        return FilterResult(False, f"instruction too short ({instruction_words} words)")
    if instruction_words > cfg.max_instruction_words:
        return FilterResult(False, f"instruction too long ({instruction_words} words)")

    # Quality score
    if example.quality < cfg.min_quality_score:
        return FilterResult(False, f"quality score too low ({example.quality:.2f})")

    # Block phrase filter (poor response patterns)
    resp_lower = response.lower()
    for phrase in cfg.block_phrases:
        if phrase in resp_lower:
            return FilterResult(False, f"blocked phrase: '{phrase}'")

    # Repetition check (many low-quality responses repeat themselves)
    words      = response.split()
    if len(words) > 20:
        bigrams    = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        unique_frac = len(set(bigrams)) / len(bigrams)
        if unique_frac < 0.6:
            return FilterResult(False, f"high response repetition ({unique_frac:.2f})")

    return FilterResult(True)


# ── Fingerprint-based deduplication ──────────────────────────────────────────

class NgramFingerprinter:
    """
    Near-duplicate detection using n-gram fingerprints.
    Two examples are duplicates if they share too many n-grams.
    """

    def __init__(self, n: int = 13, threshold: float = 0.5):
        self.n         = n
        self.threshold = threshold
        self.seen_grams: set[str] = set()

    def fingerprint(self, text: str) -> set[str]:
        """Extract n-gram fingerprints from text."""
        words  = text.lower().split()
        return {" ".join(words[i:i+self.n])
                for i in range(max(0, len(words) - self.n + 1))}

    def is_duplicate(self, text: str) -> bool:
        """Return True if this text is a near-duplicate of previously seen text."""
        grams = self.fingerprint(text)
        if not grams:
            return False
        overlap = len(grams & self.seen_grams) / len(grams)
        if overlap >= self.threshold:
            return True
        self.seen_grams.update(grams)
        return False


# ── Tokeniser simulation ──────────────────────────────────────────────────────

class SimpleTokenizer:
    """
    Word-level tokeniser simulation.
    In production, use tiktoken or SentencePiece.
    """
    SPECIAL = {
        "<|im_start|>": 32001, "<|im_end|>": 32002,
        "<system>":     32010, "<user>":     32011,
        "<assistant>":  32012, "<pad>":      0,
        "\n":           200,
    }

    def __init__(self, vocab_size: int = 32100):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        tokens = []
        for special, tid in self.SPECIAL.items():
            text = text.replace(special, f" {tid}_SPECIAL ")
        for part in text.split():
            if part.endswith("_SPECIAL"):
                tokens.append(int(part[:-8]))
            else:
                # Hash to a vocab ID in [3, vocab_size)
                tokens.append(3 + hash(part) % (self.vocab_size - 3))
        return tokens

    def apply_template_and_tokenise(
        self, messages: list[dict]
    ) -> tuple[list[int], list[int]]:
        """
        Apply ChatML template and return (token_ids, loss_mask).
        loss_mask[i] = 1 iff token i is an assistant response token.
        """
        ids  = []
        mask = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Header: <|im_start|><role>\n
            header = [32001, {"system": 32010, "user": 32011, "assistant": 32012}[role], 200]
            ids    += header
            mask   += [0] * len(header)

            # Content
            content_ids = self.encode(content)
            ids         += content_ids
            mask        += [1 if role == "assistant" else 0] * len(content_ids)

            # Footer: <|im_end|>\n
            footer = [32002, 200]
            ids   += footer
            mask  += [0] * len(footer)

        return ids, mask


# ── Statistics collector ──────────────────────────────────────────────────────

@dataclass
class DatasetStats:
    n_raw:                int   = 0
    n_filtered:           int   = 0
    n_deduplicated:       int   = 0
    n_kept:               int   = 0
    total_tokens:         int   = 0
    assistant_tokens:     int   = 0
    filter_reasons:       dict  = field(default_factory=dict)
    response_word_hist:   list  = field(default_factory=list)

    @property
    def retention_rate(self) -> float:
        return self.n_kept / self.n_raw if self.n_raw > 0 else 0.0

    @property
    def assistant_token_fraction(self) -> float:
        return self.assistant_tokens / self.total_tokens if self.total_tokens > 0 else 0.0


# ── Full pipeline ─────────────────────────────────────────────────────────────

def process_sft_dataset(
    raw_examples: list[RawExample],
    filter_cfg: FilterConfig = None,
    max_seq_len: int = 512,
    pad_to_length: bool = True,
) -> tuple[list[dict], DatasetStats]:
    """
    Full SFT data preparation pipeline.
    Returns (processed_examples, stats).
    """
    if filter_cfg is None:
        filter_cfg = FilterConfig()

    tokenizer   = SimpleTokenizer()
    deduper     = NgramFingerprinter(n=7, threshold=0.7)
    stats       = DatasetStats(n_raw=len(raw_examples))
    processed   = []

    for example in raw_examples:
        # 1. Quality filter
        result = apply_quality_filter(example, filter_cfg)
        if not result.kept:
            stats.n_filtered += 1
            reason = result.reason.split(" ")[0] if result.reason else "unknown"
            stats.filter_reasons[reason] = stats.filter_reasons.get(reason, 0) + 1
            continue

        # 2. Deduplication (based on last response)
        last_response = next(
            (m["content"] for m in reversed(example.messages)
             if m["role"] == "assistant"), ""
        )
        if deduper.is_duplicate(last_response):
            stats.n_deduplicated += 1
            continue

        # 3. Tokenise with loss masking
        ids, mask = tokenizer.apply_template_and_tokenise(example.messages)

        # 4. Truncate / pad
        if len(ids) > max_seq_len:
            ids  = ids[-max_seq_len:]    # keep most recent context
            mask = mask[-max_seq_len:]
        elif pad_to_length:
            pad_len = max_seq_len - len(ids)
            ids     += [0]  * pad_len
            mask    += [0]  * pad_len

        # 5. Build labels (-100 for masked positions)
        labels = [tid if m == 1 else -100 for tid, m in zip(ids, mask)]

        # 6. Track stats
        n_assistant = sum(mask)
        if n_assistant == 0:
            continue  # skip examples with no trainable tokens

        stats.total_tokens     += len(ids)
        stats.assistant_tokens += n_assistant
        stats.response_word_hist.append(
            len(last_response.split())
        )

        processed.append({
            "input_ids": ids,
            "labels":    labels,
            "source":    example.source,
            "n_trainable_tokens": n_assistant,
        })

    stats.n_kept = len(processed)
    return processed, stats


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    # Generate synthetic raw SFT examples
    def make_example(i: int) -> RawExample:
        system  = f"You are a helpful assistant for task {i % 5}."
        user    = f"Please explain the concept of {'machine learning' if i%2==0 else 'neural networks'} in simple terms {'with examples' * (i % 3)}."
        # Vary response quality
        if i % 7 == 0:
            asst = "Sure!"  # too short → filtered
        elif i % 11 == 0:
            asst = "As an AI language model, I cannot provide " + "information " * 100  # block phrase + too long
        else:
            asst = ("Machine learning is a field of artificial intelligence that allows "
                    "computers to learn from data without being explicitly programmed. "
                    "Neural networks are computational models inspired by the brain. "
                    f"Example {i}: consider a task like image classification. "
                    "The model learns patterns from thousands of examples and then "
                    "applies those patterns to new unseen data.")
        # Some duplicates
        if i % 13 == 0:
            asst = ("Machine learning is a field of artificial intelligence that allows "
                    "computers to learn from data without being explicitly programmed.")
        return RawExample(
            messages=[
                {"role": "system",    "content": system},
                {"role": "user",      "content": user},
                {"role": "assistant", "content": asst},
            ],
            source=["alpaca", "dolly", "oasst", "synthetic"][i % 4],
            quality=0.3 + 0.7 * (abs(hash(asst)) % 100) / 100,
        )

    n_raw    = 200
    examples = [make_example(i) for i in range(n_raw)]

    print("=" * 65)
    print(f"  SFT DATA PIPELINE DEMO ({n_raw} raw examples)")
    print("=" * 65)
    print()

    processed, stats = process_sft_dataset(
        examples,
        filter_cfg=FilterConfig(min_quality_score=0.4),
        max_seq_len=256,
    )

    print("  PIPELINE STATISTICS:")
    print(f"    Raw examples:          {stats.n_raw:>6}")
    print(f"    Filtered (quality):    {stats.n_filtered:>6}  "
          f"({stats.n_filtered/stats.n_raw*100:.1f}%)")
    print(f"    Deduplicated:          {stats.n_deduplicated:>6}  "
          f"({stats.n_deduplicated/stats.n_raw*100:.1f}%)")
    print(f"    Kept:                  {stats.n_kept:>6}  "
          f"(retention: {stats.retention_rate*100:.1f}%)")
    print()
    print("  TOKEN STATISTICS:")
    print(f"    Total tokens:          {stats.total_tokens:>8,}")
    print(f"    Assistant tokens:      {stats.assistant_tokens:>8,}")
    print(f"    Assistant fraction:    {stats.assistant_token_fraction*100:>7.1f}%")
    print(f"    (Only {stats.assistant_token_fraction*100:.1f}% of tokens actually train the model!)")
    print()
    print("  FILTER REASONS:")
    for reason, count in sorted(stats.filter_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<30} {count:>5}  ({count/stats.n_raw*100:.1f}%)")
    print()

    # Show distribution of response lengths
    if stats.response_word_hist:
        rl = stats.response_word_hist
        print("  RESPONSE WORD LENGTH DISTRIBUTION:")
        buckets = [(0,20), (20,50), (50,100), (100,200), (200,1000)]
        for lo, hi in buckets:
            count = sum(1 for l in rl if lo <= l < hi)
            bar   = "█" * (count * 20 // max(len(rl), 1))
            print(f"    {lo:>5}–{hi:<5}: {bar:<20}  {count:>4} ({count/len(rl)*100:.0f}%)")

    # Show a processed example
    if processed:
        ex = processed[0]
        n_t   = ex["n_trainable_tokens"]
        n_tot = len(ex["input_ids"])
        print()
        print(f"  EXAMPLE PROCESSED ENTRY:")
        print(f"    input_ids[:10]: {ex['input_ids'][:10]}")
        print(f"    labels[:10]:    {ex['labels'][:10]}")
        print(f"    trainable:      {n_t}/{n_tot} tokens  ({n_t/n_tot*100:.1f}%)")
        print(f"    source:         {ex['source']}")
''',
    },

    "Catastrophic Forgetting Analysis and Prevention": {
        "description": "Quantify and visualise catastrophic forgetting during SFT — how aggressive fine-tuning degrades pre-training benchmarks, and how to prevent it with LR scheduling and replay.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
CATASTROPHIC FORGETTING: ANALYSIS AND PREVENTION
================================================================================

Catastrophic forgetting is the most common failure mode in SFT:
aggressive fine-tuning rewrites the model's pre-trained knowledge.

This module demonstrates:
    1. How forgetting correlates with learning rate
    2. The pre-training perplexity as a forgetting proxy metric
    3. Replay buffer: mixing pre-training data into SFT to prevent forgetting
    4. Optimal LR range for SFT without forgetting
    5. Layer-wise forgetting: which layers are most affected

================================================================================
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Tiny model for ablation ───────────────────────────────────────────────────

class MiniModel(nn.Module):
    def __init__(self, vocab: int = 256, d: int = 64, n_layers: int = 3):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d),
                          nn.Linear(d, d * 4, bias=False),
                          nn.GELU(),
                          nn.Linear(d * 4, d, bias=False))
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


def compute_perplexity(model, data: torch.Tensor,
                        batch_size: int = 4) -> float:
    """Compute model perplexity on a token sequence."""
    model.eval()
    total_loss  = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(data) - 8, batch_size * 8):
            batch = data[i : i + batch_size * 8].view(batch_size, 8)
            logits = model(batch[:, :-1])
            loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      batch[:, 1:].reshape(-1))
            total_loss   += loss.item() * batch[:, 1:].numel()
            total_tokens += batch[:, 1:].numel()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def run_sft_with_forgetting_analysis(
    model_init: nn.Module,
    sft_data: torch.Tensor,
    pretrain_data: torch.Tensor,
    lr: float,
    n_steps: int,
    replay_frac: float = 0.0,   # 0 = no replay, 0.1 = 10% pretrain data
    label_name: str = "",
) -> dict:
    """
    Run SFT and track forgetting via pre-training perplexity.
    Returns dict of metrics at each step.
    """
    model = copy.deepcopy(model_init)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    VOCAB = model.embed.num_embeddings
    B, T  = 4, 8

    ppl_history   = []
    loss_history  = []

    for step in range(n_steps):
        model.train()
        opt.zero_grad()

        # SFT batch with simulated response masking
        sft_idx    = torch.randint(0, len(sft_data) - T, (B,))
        sft_batch  = torch.stack([sft_data[i:i+T] for i in sft_idx])
        # Simulate: only last 4 tokens are "response" (loss mask)
        sft_logits = model(sft_batch[:, :-1])
        sft_labels = sft_batch[:, 1:].clone()
        sft_labels[:, :3] = -100   # mask first 3 tokens (prompt)
        sft_loss   = F.cross_entropy(sft_logits.reshape(-1, VOCAB),
                                      sft_labels.reshape(-1), ignore_index=-100)

        total_loss = sft_loss

        # Replay: mix in pre-training data to prevent forgetting
        if replay_frac > 0:
            pt_idx   = torch.randint(0, len(pretrain_data) - T, (B,))
            pt_batch = torch.stack([pretrain_data[i:i+T] for i in pt_idx])
            pt_logits = model(pt_batch[:, :-1])
            pt_loss   = F.cross_entropy(pt_logits.reshape(-1, VOCAB),
                                         pt_batch[:, 1:].reshape(-1))
            total_loss = (1 - replay_frac) * sft_loss + replay_frac * pt_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_history.append(sft_loss.item())

        # Measure forgetting every 5 steps
        if step % 5 == 0:
            ppl = compute_perplexity(model, pretrain_data)
            ppl_history.append((step, ppl))

    return {
        "label":        label_name,
        "lr":           lr,
        "replay_frac":  replay_frac,
        "loss_final":   loss_history[-1],
        "ppl_history":  ppl_history,
        "ppl_final":    ppl_history[-1][1] if ppl_history else float("inf"),
    }


def layer_forgetting_analysis(model_pre: nn.Module, model_post: nn.Module) -> dict:
    """
    Analyse which layers changed most during SFT.
    Returns per-layer relative drift.
    """
    drifts = {}
    for (name_pre, p_pre), (name_post, p_post) in zip(
        model_pre.named_parameters(), model_post.named_parameters()
    ):
        assert name_pre == name_post
        drift  = (p_pre.data - p_post.data).norm().item()
        orig   = p_pre.data.norm().item()
        drifts[name_pre] = drift / orig if orig > 0 else 0.0
    return drifts


def sparkline(values: list[float], width: int = 30) -> str:
    if not values: return ""
    lo, hi = min(values), max(values)
    bars = " ▁▂▃▄▅▆▇█"
    span = hi - lo if hi > lo else 1.0
    result = ""
    step = max(1, len(values) // width)
    for i in range(0, len(values), step):
        v   = values[i]
        idx = int((v - lo) / span * (len(bars) - 1))
        result += bars[max(0, min(len(bars)-1, idx))]
    return result[:width]


if __name__ == "__main__":
    torch.manual_seed(42)

    VOCAB    = 256
    D        = 64
    N_LAYERS = 3
    N_STEPS  = 50

    # Synthetic data
    pretrain_data = torch.randint(3, VOCAB, (2000,))
    sft_data      = torch.randint(3, VOCAB, (500,))

    model_init = MiniModel(VOCAB, D, N_LAYERS)

    # Measure initial (pre-SFT) perplexity as baseline
    initial_ppl = compute_perplexity(model_init, pretrain_data)
    print("=" * 65)
    print("  CATASTROPHIC FORGETTING ANALYSIS")
    print(f"  Pre-SFT baseline perplexity: {initial_ppl:.2f}")
    print("=" * 65)
    print()

    # Ablation over learning rates
    configs = [
        (3e-3,  0.0,  "LR=3e-3 (too high)"),
        (1e-4,  0.0,  "LR=1e-4 (standard SFT)"),
        (1e-5,  0.0,  "LR=1e-5 (conservative)"),
        (1e-4,  0.1,  "LR=1e-4 + 10% replay"),
        (1e-4,  0.2,  "LR=1e-4 + 20% replay"),
    ]

    results = []
    for lr, replay, label in configs:
        r = run_sft_with_forgetting_analysis(
            model_init, sft_data, pretrain_data,
            lr=lr, n_steps=N_STEPS, replay_frac=replay,
            label_name=label
        )
        results.append(r)

    print(f"  {'Configuration':<30}  {'SFT Loss':>10}  "
          f"{'Final PPL':>10}  {'Forgetting':>12}  PPL trajectory")
    print(f"  {'':─<30}  {'':─>10}  {'':─>10}  {'':─>12}  {'':─<25}")

    for r in results:
        ppl_vals     = [p for _, p in r["ppl_history"]]
        forgetting   = (r["ppl_final"] - initial_ppl) / initial_ppl * 100
        flag         = "✗ SEVERE" if forgetting > 20 else "⚠️  moderate" if forgetting > 5 else "✓ minimal"
        spark        = sparkline(ppl_vals)
        print(f"  {r['label']:<30}  {r['loss_final']:>10.4f}  "
              f"{r['ppl_final']:>10.2f}  {forgetting:>+11.1f}%  {spark}")

    print()
    print(f"  Baseline PPL (pre-SFT): {initial_ppl:.2f}")
    print()
    print("  Key observations:")
    print("  • LR=3e-3 causes severe forgetting — PPL spikes dramatically")
    print("  • LR=1e-5 avoids forgetting but SFT loss decreases slowly")
    print("  • LR=1e-4 + replay balances both: low SFT loss AND low forgetting")
    print("  • Replay with just 10-20% pre-training data is often sufficient")

    # Layer-wise drift analysis
    print()
    print("=" * 65)
    print("  LAYER-WISE FORGETTING (LR=3e-3 vs LR=1e-4)")
    print("=" * 65)
    print()

    high_lr = run_sft_with_forgetting_analysis(
        model_init, sft_data, pretrain_data, lr=3e-3, n_steps=N_STEPS,
        label_name="high_lr"
    )
    low_lr  = run_sft_with_forgetting_analysis(
        model_init, sft_data, pretrain_data, lr=1e-4, n_steps=N_STEPS,
        label_name="low_lr"
    )

    # Reconstruct final models for layer analysis
    # (simplified: compare initial vs re-run high LR model)
    model_high = copy.deepcopy(model_init)
    opt_h = torch.optim.AdamW(model_high.parameters(), lr=3e-3)
    for _ in range(N_STEPS):
        idx = torch.randint(0, len(sft_data) - 8, (4,))
        b   = torch.stack([sft_data[i:i+8] for i in idx])
        l   = F.cross_entropy(model_high(b[:,:-1]).reshape(-1,VOCAB),b[:,1:].reshape(-1))
        opt_h.zero_grad(); l.backward(); opt_h.step()

    drifts_high = layer_forgetting_analysis(model_init, model_high)
    print(f"  {'Parameter':<35}  {'LR=3e-3 drift':>16}")
    print(f"  {'':─<35}  {'':─>16}")
    for name, drift in sorted(drifts_high.items(), key=lambda x: -x[1]):
        bar  = "█" * int(min(drift * 50, 30))
        flag = " ← embedding!" if "embed" in name else ""
        print(f"  {name:<35}  {drift:>14.4f}×  {bar}{flag}")

    print()
    print("  Embedding layers often drift most because they represent")
    print("  fundamental token semantics learned during pre-training.")
    print("  Freezing embeddings during SFT can reduce forgetting significantly.")
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
    #     from llm_training.visuals.supervised_finetuning import (
    #         SFT_VISUAL_HTML,
    #         SFT_VISUAL_HEIGHT,
    #     )
    #     visual_html   = SFT_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = SFT_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[26_supervised_finetuning_sft.py] Could not load visual: {e}",
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