"""
Training Loop Fundamentals
===========================

The training loop is the engine that drives every LLM from a random
initialisation to a capable model. Understanding each phase — forward pass,
loss computation, backward pass, and parameter update — and the exact data
flow between them is the foundation for everything else in this collection:
optimisers, mixed precision, gradient clipping, and distributed training all
plug into this same core loop.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Training Loop Fundamentals"
DISPLAY_NAME = "06 · Training Loop"
ICON         = "🔄"
SUBTITLE     = "Forward Pass, Loss, Backward Pass, Update"


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

### The Big Picture: What a Training Loop Does

A language model is a parameterised function f(x; θ) that takes a sequence of
tokens x and predicts the probability distribution over the next token. The
parameters θ start as random noise. Training is the process of iteratively
adjusting θ to make f better at predicting next tokens on a large corpus.

The training loop repeats three phases indefinitely over the dataset:

    1.  Forward Pass  — run the model on a batch, produce predictions
    2.  Loss          — measure how wrong the predictions are
    3.  Backward Pass — compute how much each parameter contributed to the error
    4.  Update        — nudge parameters in the direction that reduces the error

That's it. Billions of parameters, trillions of tokens, months of GPU time —
all of it is this loop, repeated.


### Phase 1 — The Forward Pass

The forward pass runs a batch of input token sequences through the model and
produces logits (unnormalised scores over the vocabulary):

    Input:   idx    ∈ ℤ^(B × T)        — B sequences, each T tokens long
    Output:  logits ∈ ℝ^(B × T × V)   — for each of the B×T positions, a
                                          score for each of the V vocabulary items

During a forward pass, PyTorch's autograd engine records every operation in a
**computation graph** — a directed acyclic graph (DAG) where nodes are tensors
and edges are the functions that produced them. This graph is what makes
backpropagation automatic.

Memory during the forward pass:
    •   Activations (intermediate tensors) are stored for use in the backward pass
    •   For a large model, activations often consume MORE memory than the weights
    •   This is why activation checkpointing (Module 16) is critical


### Phase 2 — The Loss Function

For language model pre-training, the objective is **next-token prediction**
(also called the language modelling objective or causal language modelling, CLM).

Given a sequence [t₀, t₁, t₂, …, t_{T-1}], the model predicts:
    P(t₁ | t₀),  P(t₂ | t₀, t₁),  …,  P(t_{T-1} | t₀, …, t_{T-2})

All T-1 predictions are made simultaneously (thanks to the causal mask), and
the loss is the average cross-entropy over all positions:

    CE(logits, targets) = −(1/T) Σᵢ log P(tᵢ₊₁ | t₀..tᵢ)

In practice:
    •   Input  to model: [t₀, t₁, …, t_{T-1}]   (the prompt)
    •   Target for loss: [t₁, t₂, …, t_T]        (shift by one)
    •   Loss is averaged over the batch and sequence dimension

**Cross-entropy loss formula:**

    L = −(1 / (B·T)) Σ_{b,t} log( softmax(logits[b,t,:])[targets[b,t]] )

Or equivalently (numerically stable):

    L = (1 / (B·T)) Σ_{b,t} log_sum_exp(logits[b,t,:]) − logits[b,t,target[b,t]]

**Perplexity:** The standard metric for LM quality is perplexity = exp(loss).
A perplexity of 10 means the model is as "confused" as if it were choosing
uniformly among 10 equally likely options at each position.


    **Diagram 1 — The Language Modelling Objective:**

    NEXT-TOKEN PREDICTION OBJECTIVE
    ════════════════════════════════════════════════════════════════

    Input sequence:   "The  cat  sat  on   the  mat"
    Token IDs:         [47]  [82] [391] [25] [47]  [291]

    Model input:      [ 47,  82, 391,  25,  47]   (first 5 tokens)
    Model targets:    [ 82, 391,  25,  47, 291]   (next token at each position)

    Forward pass produces logits[0..4, 0..V]:
    Position 0: P( next=82  | 47)         = ?   ← predict "cat" given "The"
    Position 1: P( next=391 | 47, 82)     = ?   ← predict "sat" given "The cat"
    Position 2: P( next=25  | 47, 82, 391)= ?   ← predict "on" given "The cat sat"
    ...

    Loss = -mean(log P(correct token) at each position)

    If model assigns:
        P(cat  | The)        = 0.30  → contribution: -log(0.30) = 1.20
        P(sat  | The cat)    = 0.45  → contribution: -log(0.45) = 0.80
        P(on   | The cat sat)= 0.60  → contribution: -log(0.60) = 0.51
    Average loss = (1.20 + 0.80 + 0.51) / 3 = 0.84
    Perplexity   = exp(0.84) ≈ 2.3


### Phase 3 — The Backward Pass (Backpropagation)

Backpropagation computes ∂L/∂θ — the gradient of the loss with respect to
every trainable parameter — using the chain rule of calculus.

**The chain rule:** if L = f(g(θ)), then:

    dL/dθ = (dL/df) · (df/dg) · (dg/dθ)

In a deep network with many composed functions, the chain rule propagates
gradients layer by layer from the loss back to every parameter:

    dL/dθ_layer_1 = dL/d(out_N) · d(out_N)/d(out_{N-1}) · ... · d(out_1)/dθ_1

PyTorch's `.backward()` call traverses the computation graph in reverse
(from loss → input), computing and accumulating gradients for every parameter
that has `requires_grad=True`.

**Memory during the backward pass:**
    •   Gradients have the same shape as parameters: ~same memory as the model
    •   The computation graph (activations) must be kept in memory during
        the forward pass so they can be reused in the backward pass
    •   Total memory = parameters + gradients + activations + optimiser state
    •   For a 7B model in fp32: ~28GB params + ~28GB grads + activations + …


    **Diagram 2 — The Computation Graph and Backward Pass:**

    COMPUTATION GRAPH (simplified 2-layer network)
    ════════════════════════════════════════════════════════════════

    FORWARD PASS (builds the graph):

    x → [W₁] → h₁ = W₁·x → [ReLU] → a₁ → [W₂] → h₂ = W₂·a₁ → [Loss] → L

    Each arrow stores the local derivative for use in the backward pass.

    BACKWARD PASS (traverses graph in reverse):

    L ──────────────────────────────────────────────────────────► dL/dL = 1
       ↑
    ∂L/∂h₂ = 1 · ∂L/∂h₂                          ← from loss function
       ↑
    ∂L/∂W₂ = ∂L/∂h₂ · ∂h₂/∂W₂ = ∂L/∂h₂ · a₁ᵀ  ← gradient for W₂
    ∂L/∂a₁ = ∂L/∂h₂ · ∂h₂/∂a₁ = ∂L/∂h₂ · W₂ᵀ  ← pass gradient through
       ↑
    ∂L/∂h₁ = ∂L/∂a₁ · ∂a₁/∂h₁                    ← ReLU: 1 if h₁>0, 0 otherwise
       ↑
    ∂L/∂W₁ = ∂L/∂h₁ · xᵀ                          ← gradient for W₁

    Result: ∂L/∂W₁ and ∂L/∂W₂ — the gradient of loss w.r.t. every weight.


### Phase 4 — The Parameter Update

Once we have gradients ∂L/∂θ, we update parameters to reduce the loss.
The simplest update rule is **stochastic gradient descent (SGD)**:

    θ ← θ − lr · ∂L/∂θ

where lr (learning rate) controls the step size.

Modern LLMs use AdamW (covered in Module 07), but the principle is the same:
move parameters in the direction that reduces the loss, scaled by the learning
rate.

After the update:
    1.  Zero the gradients (`.zero_grad()`) — PyTorch accumulates gradients
        by default; if you don't zero them, the next backward pass adds on top
    2.  Optionally clip gradients (Module 09) before the update
    3.  Log metrics (loss, learning rate, gradient norm)
    4.  Advance to the next batch


### The Token Prediction — Sampling vs Greedy vs Beam Search

At training time: always use the ground-truth next token (teacher forcing).
At inference time, several strategies exist for picking the next token from
the predicted distribution:

    Greedy decoding:
        Always pick argmax(logits). Deterministic but repetitive.
        next_token = argmax(logits)

    Temperature sampling:
        Scale logits by temperature T before softmax.
        T < 1: sharper distribution (more confident)
        T > 1: flatter distribution (more random)
        next_token ~ Categorical(softmax(logits / T))

    Top-k sampling:
        Sample only from the k highest-probability tokens.
        Prevents sampling very rare/unlikely tokens.

    Top-p (nucleus) sampling:
        Sample from the smallest set of tokens whose cumulative probability ≥ p.
        Adapts the number of candidates to the entropy of the distribution.

    Beam search:
        Maintain k candidate sequences in parallel, always expanding the
        most likely ones. Common in translation but less used for open-ended
        generation (produces generic, repetitive text).


### Packed vs Padded Sequences

A batch must consist of sequences with the same length to be processed as a
matrix. Two strategies handle variable-length sequences:

    **Padding:** pad shorter sequences to the length of the longest with a
    special [PAD] token and mask out their loss contribution.

        Seq 1: "Hello world" → [101, 202, 0, 0, 0]   ← 3 PAD tokens
        Seq 2: "Hi there friend" → [88, 77, 66, 0, 0] ← 2 PAD tokens
        Loss mask:                  [1,  1,  0,  0,  0]

    Wasted compute: PAD tokens consume memory and FLOPs.

    **Packing (sequence packing):** concatenate multiple documents end-to-end
    to fill each context window completely. Documents are separated with [EOS].

        Packed:  [101, 202, EOS, 88, 77, 66, EOS, 55, 44, ...]   ← no waste

    Packing achieves near-100% token utilisation. It requires a 2D attention
    mask to prevent document boundaries from attending across each other
    (covered in Module 14).


    **Diagram 3 — Padded vs Packed Batching:**

    PADDED vs PACKED SEQUENCES
    ════════════════════════════════════════════════════════════════

    Context window = 8 tokens, 3 documents of lengths 3, 2, 5

    PADDED (3 sequences padded to length 5):
    ┌─────────────────────────────────────────────────┐
    │ doc1 doc1 doc1 PAD  PAD  PAD  PAD  PAD  │ 3/8 useful
    │ doc2 doc2 PAD  PAD  PAD  PAD  PAD  PAD  │ 2/8 useful
    │ doc3 doc3 doc3 doc3 doc3 PAD  PAD  PAD  │ 5/8 useful
    └─────────────────────────────────────────────────┘
    Token utilisation: (3+2+5)/(3×8) = 10/24 ≈ 42%

    PACKED (1 sequence, all 3 documents concatenated):
    ┌─────────────────────────────────────────────────┐
    │ doc1 doc1 doc1 EOS  doc2 doc2 EOS  doc3 │ 8/8 useful (with 2 docs cut)
    └─────────────────────────────────────────────────┘
    Token utilisation: 8/8 = 100%  ✓


### The Training Loop in Practice — Key Implementation Details

**1. Mixed precision (fp16/bf16):** Run the forward and backward pass in lower
precision to reduce memory and increase throughput. Covered in Module 10.

**2. Gradient accumulation:** If GPU memory limits batch size to B, run K
micro-batches of B/K before updating, accumulating gradients. Covered in Module 17.

**3. Gradient clipping:** Clip the global gradient norm before the update to
prevent exploding gradients. Covered in Module 09.

**4. Learning rate scheduling:** The learning rate is not constant — it warms
up linearly then decays following a cosine schedule. Covered in Module 08.

**5. Checkpointing:** Save model weights, optimiser state, and training progress
periodically. Covered in Module 33.

**6. Logging:** Track loss, perplexity, learning rate, gradient norm, and token
throughput (tokens/second). Covered in Module 34.


    **Diagram 4 — One Complete Training Step:**

    ONE COMPLETE TRAINING STEP
    ════════════════════════════════════════════════════════════════

    ① Load batch
       idx[B, T]  ←── DataLoader (Module 14)

    ② Forward pass (autocast fp16/bf16 — Module 10)
       logits[B, T, V] = model(idx)

    ③ Compute loss
       loss = cross_entropy(logits[:, :-1, :], idx[:, 1:])
       loss = loss / grad_accum_steps     ← if gradient accumulation (Module 17)

    ④ Backward pass
       scaler.scale(loss).backward()      ← GradScaler for fp16 (Module 10)

    ⑤ (Accumulate more micro-batches if grad_accum_steps > 1)

    ⑥ Unscale gradients
       scaler.unscale_(optimizer)

    ⑦ Clip gradient norm  (Module 09)
       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    ⑧ Optimiser step  (Module 07)
       scaler.step(optimizer)
       scaler.update()

    ⑨ Zero gradients
       optimizer.zero_grad(set_to_none=True)

    ⑩ Update LR schedule  (Module 08)
       scheduler.step()

    ⑪ Log metrics
       wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], ...})


### Weight Initialisation

Before training can begin, parameters must be initialised. Poor initialisation
leads to vanishing or exploding activations from the very first forward pass,
making training impossible regardless of the learning rate.

**Xavier / Glorot initialisation** (for tanh/sigmoid networks):
    σ = sqrt(2 / (fan_in + fan_out))

**Kaiming / He initialisation** (for ReLU networks):
    σ = sqrt(2 / fan_in)

**Modern LLM initialisation:** Most LLMs use a small standard deviation,
typically 0.02 (GPT-2 style) or scaled by 1/√(2·N·d) for residual connections.
The key is that activations at the start of training should be O(1) — not
exploding or vanishing — so that gradients flow cleanly.

LLaMA uses:
    Linear weights: N(0, 0.02)  or  N(0, 1/√d_model)
    Embedding weights: N(0, 0.02)
    Output projection (after residual): scaled by 1/√(2·n_layers)
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Memory Breakdown During Training (7B model, fp32, batch=1, T=2048)

| Component              | Size Formula              | Approx (7B, fp32)  |
|------------------------|---------------------------|--------------------|
| Model parameters       | P × 4 bytes               | ~28 GB             |
| Gradients              | P × 4 bytes               | ~28 GB             |
| Activations (no ckpt)  | ~ 12 × B × T × d × N × 2 | ~50+ GB            |
| Activations (ckpt)     | ~ sqrt(N) layers kept     | ~10 GB             |
| AdamW optimiser state  | P × 8 bytes (2 moments)   | ~56 GB             |
| Total (no tricks)      |                           | ~162 GB            |
| Total (ckpt + bf16)    |                           | ~30–50 GB          |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Training Loop From Scratch": {
        "description": "A complete, annotated training loop for a small language model — forward pass, cross-entropy loss, backward pass, SGD update, and logging.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
LANGUAGE MODEL TRAINING LOOP — FROM SCRATCH
================================================================================

A minimal but complete training loop for a character-level language model.
No HuggingFace, no trainer abstractions — every step is explicit.

Demonstrates:
    1. Data preparation (char-level tokenisation)
    2. Batching (random windows from the corpus)
    3. Forward pass
    4. Cross-entropy loss (with target shift)
    5. Backward pass
    6. SGD parameter update
    7. Metric logging (loss, perplexity, tokens/sec)
================================================================================
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Tiny LM for this demo ──────────────────────────────────────────────────────

class TinyLM(nn.Module):
    """A minimal decoder-only language model."""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Embedding(max_len,    d_model)
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.body  = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying
        self.max_len = max_len
        self._build_mask(max_len)

    def _build_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.embed(idx) + self.pos(pos)
        mask = self.causal_mask[:T, :T]
        # TransformerDecoder: use same x as both target and memory (self-attn only)
        x    = self.body(x, x, tgt_mask=mask, memory_mask=mask,
                         tgt_is_causal=True, memory_is_causal=True)
        x    = self.norm(x)
        return self.head(x)   # (B, T, vocab_size)


# ── Data ───────────────────────────────────────────────────────────────────────

CORPUS = (
    "The transformer architecture revolutionised natural language processing. "
    "Before transformers, recurrent networks processed tokens sequentially, "
    "making parallelisation during training impossible. Transformers replaced "
    "recurrence with self-attention, allowing every token to directly attend "
    "to every other token simultaneously. This enabled massive parallelism "
    "across the sequence dimension, making it practical to train on "
    "trillions of tokens using thousands of GPUs. The key insight is that "
    "language does not require sequential processing — the meaning of a word "
    "depends on context, not position, and self-attention captures this. "
) * 20  # repeat for a larger corpus


def build_vocab(text: str):
    chars     = sorted(set(text))
    stoi      = {c: i for i, c in enumerate(chars)}
    itos      = {i: c for i, c in enumerate(chars)}
    return stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    """Sample a random batch of (input, target) pairs from the corpus."""
    # Random start positions
    ix  = torch.randint(len(data) - seq_len, (batch_size,))
    x   = torch.stack([data[i    : i + seq_len    ] for i in ix])
    y   = torch.stack([data[i + 1: i + seq_len + 1] for i in ix])
    return x, y


# ── Training loop ──────────────────────────────────────────────────────────────

def train():
    # Hyperparameters
    BATCH_SIZE = 16
    SEQ_LEN    = 64
    D_MODEL    = 128
    N_HEADS    = 4
    N_LAYERS   = 2
    LR         = 3e-4
    N_STEPS    = 300
    LOG_EVERY  = 50
    EVAL_EVERY = 100

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data
    stoi, itos = build_vocab(CORPUS)
    vocab_size = len(stoi)
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data   = data[n_train:]

    print(f"Vocab size:    {vocab_size}")
    print(f"Train tokens:  {len(train_data):,}")
    print(f"Val   tokens:  {len(val_data):,}")
    print()

    # Model
    model = TinyLM(vocab_size, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params:  {n_params:,}")
    print()

    # Optimiser — simple SGD for clarity (AdamW shown in Module 07)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    print("=" * 60)
    print(f"  {'Step':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'PPL':>8}  "
          f"{'Tok/s':>10}")
    print("=" * 60)

    total_tokens = 0
    t_start      = time.perf_counter()

    for step in range(1, N_STEPS + 1):
        model.train()

        # ── ① Load batch ────────────────────────────────────────────────────
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN)
        x, y = x.to(device), y.to(device)

        # ── ② Forward pass ───────────────────────────────────────────────────
        logits = model(x)    # (B, T, V)

        # ── ③ Compute loss ────────────────────────────────────────────────────
        # Flatten: (B*T, V) vs (B*T,)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
        )

        # ── ④ Backward pass ───────────────────────────────────────────────────
        optimiser.zero_grad(set_to_none=True)
        loss.backward()

        # ── ⑤ Gradient clip (prevents exploding gradients) ────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # ── ⑥ Parameter update ────────────────────────────────────────────────
        optimiser.step()

        # Track throughput
        total_tokens += BATCH_SIZE * SEQ_LEN

        # ── ⑦ Logging ─────────────────────────────────────────────────────────
        if step % LOG_EVERY == 0 or step == 1:
            elapsed      = time.perf_counter() - t_start
            tok_per_sec  = total_tokens / elapsed
            train_loss   = loss.item()

            # Validation loss
            val_loss = None
            if step % EVAL_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    xv, yv   = get_batch(val_data, BATCH_SIZE * 2, SEQ_LEN)
                    xv, yv   = xv.to(device), yv.to(device)
                    val_logits = model(xv)
                    val_loss   = F.cross_entropy(
                        val_logits.view(-1, vocab_size), yv.view(-1)
                    ).item()

            ppl = math.exp(train_loss)
            val_str = f"{val_loss:.4f}" if val_loss is not None else "     -"
            print(f"  {step:>6}  {train_loss:>12.4f}  {val_str:>10}  "
                  f"{ppl:>8.2f}  {tok_per_sec:>10.0f}")

    print()
    print("Training complete.")

    # ── Generation demo ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  GENERATION SAMPLE (greedy, temperature=0.8)")
    print("=" * 60)

    model.eval()
    prompt      = "The transformer"
    prompt_ids  = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long).to(device)
    generated   = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(150):
            context = generated[:, -SEQ_LEN:]
            logits  = model(context)[:, -1, :]   # last position only
            probs   = F.softmax(logits / 0.8, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

    output_text = "".join(itos[i.item()] for i in generated[0])
    print(f"  {output_text}")


if __name__ == "__main__":
    train()
''',
    },

    "Cross-Entropy Loss — Deep Dive": {
        "description": "Implement cross-entropy from scratch, show numerical stability tricks, and visualise how loss relates to perplexity and token probability.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
CROSS-ENTROPY LOSS — DEEP DIVE
================================================================================

Cross-entropy is the sole loss function for LLM pre-training. This module
dissects it:
    1. Mathematical definition
    2. Naive vs numerically stable implementation
    3. The log-sum-exp trick
    4. Relationship to perplexity
    5. What the loss looks like at different training stages
================================================================================
"""

import math
import torch
import torch.nn.functional as F


# ── 1. From-scratch implementations ───────────────────────────────────────────

def cross_entropy_naive(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Naive cross-entropy: compute softmax explicitly, then take -log of target.
    Numerically unstable for large logits (exp overflows).
    """
    probs  = torch.exp(logits) / torch.exp(logits).sum(dim=-1, keepdim=True)
    target_probs = probs[torch.arange(len(targets)), targets]
    return -target_probs.log().mean()


def cross_entropy_stable(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable cross-entropy using the log-sum-exp trick.

    log(softmax(x)_i) = x_i - log(sum(exp(x)))
                      = x_i - (max(x) + log(sum(exp(x - max(x)))))
                                         ↑ this never overflows because
                                           x_i - max(x) ≤ 0 always
    """
    # Subtract max for numerical stability (doesn't change result)
    logits_shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_sum_exp    = torch.log(torch.exp(logits_shifted).sum(dim=-1))
    target_logits  = logits_shifted[torch.arange(len(targets)), targets]
    log_probs      = target_logits - log_sum_exp
    return -log_probs.mean()


# ── 2. Verify implementations match PyTorch ───────────────────────────────────

def verify():
    torch.manual_seed(0)
    B, V = 32, 1000
    logits  = torch.randn(B, V)
    targets = torch.randint(0, V, (B,))

    loss_pt     = F.cross_entropy(logits, targets)
    loss_stable = cross_entropy_stable(logits, targets)

    print("=" * 58)
    print("  IMPLEMENTATION COMPARISON")
    print("=" * 58)
    print(f"  PyTorch  F.cross_entropy: {loss_pt.item():.6f}")
    print(f"  Stable   (scratch):       {loss_stable.item():.6f}")
    print(f"  Match: {abs(loss_pt.item() - loss_stable.item()) < 1e-5}")

    # Demonstrate overflow with naive version on large logits
    print()
    print("  Overflow demo:")
    large_logits = torch.randn(B, V) * 100   # very large logits
    try:
        loss_naive = cross_entropy_naive(large_logits, targets)
        print(f"  Naive with large logits:  {loss_naive.item()}")
    except Exception as e:
        print(f"  Naive with large logits:  ERROR — {e}")

    loss_pt_large = F.cross_entropy(large_logits, targets)
    print(f"  PyTorch with large logits: {loss_pt_large.item():.6f}  (handles it ✓)")


# ── 3. Loss ↔ Perplexity relationship ─────────────────────────────────────────

def loss_perplexity_table():
    print()
    print("=" * 58)
    print("  LOSS ↔ PERPLEXITY ↔ INTERPRETATION")
    print("=" * 58)
    print()
    print(f"  {'Loss':>8}  {'Perplexity':>12}  Interpretation")
    print(f"  {'':─>8}  {'':─>12}  ─────────────────────────────────────")

    scenarios = [
        (math.log(50257),   "Random (GPT vocab), untrained"),
        (4.0,               "Very early training"),
        (3.0,               "Early training (~e³ ≈ 20 choices)"),
        (2.5,               "Mid training (~e^2.5 ≈ 12 choices)"),
        (2.0,               "Good (~e² ≈ 7 choices)"),
        (1.5,               "Strong (~e^1.5 ≈ 4.5 choices)"),
        (1.0,               "Very strong (~e choices)"),
        (0.5,               "Exceptional (nearly deterministic)"),
    ]

    for loss, label in scenarios:
        ppl = math.exp(loss)
        print(f"  {loss:>8.3f}  {ppl:>12.1f}  {label}")

    print()
    print("  Perplexity = exp(loss) = effective vocabulary size")
    print("  Perplexity 7 means: on average, model is as uncertain as")
    print("  choosing uniformly among 7 equally likely next tokens.")


# ── 4. Masked loss (for padding / instruction tuning) ─────────────────────────

def masked_loss_demo():
    """
    In instruction tuning, we only want to compute loss on the assistant
    response tokens, not the prompt/system tokens.
    """
    print()
    print("=" * 58)
    print("  MASKED LOSS (instruction tuning)")
    print("=" * 58)
    print()

    # Simulate a conversation:
    # [SYSTEM] You are helpful. [USER] What is 2+2? [ASST] It is 4.
    # Loss mask: 1 for assistant tokens only
    V    = 100
    B, T = 1, 10

    logits  = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    # Loss mask: only compute loss on assistant tokens (positions 7-9 in this example)
    loss_mask = torch.zeros(B, T)
    loss_mask[0, 7:] = 1.0   # positions 7, 8, 9 are assistant tokens

    print(f"  Sequence length: T={T}")
    print(f"  Loss mask (1=compute, 0=ignore):")
    print(f"    {loss_mask[0].int().tolist()}")
    print(f"    (positions 0-6 = system+user prompt, 7-9 = assistant response)")
    print()

    # Unmasked loss (wrong — includes prompt in loss)
    loss_all = F.cross_entropy(logits.view(-1, V), targets.view(-1))

    # Masked loss (correct — only assistant tokens)
    token_losses = F.cross_entropy(
        logits.view(-1, V), targets.view(-1), reduction="none"
    ).view(B, T)
    masked_loss = (token_losses * loss_mask).sum() / loss_mask.sum()

    print(f"  Unmasked loss (all tokens):   {loss_all.item():.4f}")
    print(f"  Masked loss   (asst only):    {masked_loss.item():.4f}")
    print()
    print("  Using masked loss is critical for instruction tuning:")
    print("  We don't want the model to 'learn' the system/user prompt.")
    print("  Only the assistant's response should shape the gradients.")


# ── 5. Label smoothing ────────────────────────────────────────────────────────

def label_smoothing_demo():
    """
    Label smoothing replaces hard 0/1 targets with soft targets that
    assign a small probability ε to all other tokens.
    """
    print()
    print("=" * 58)
    print("  LABEL SMOOTHING")
    print("=" * 58)
    print()
    print("  Hard target: [0, 0, 1, 0, 0]  (100% on correct token)")
    print("  Soft target (ε=0.1, V=5): [0.02, 0.02, 0.92, 0.02, 0.02]")
    print()
    print("  Effect:")
    print("  • Prevents model from becoming overconfident")
    print("  • Regularises training — especially useful for fine-tuning")
    print("  • Cross-entropy with label smoothing:")
    print("    L = (1-ε)·CE(logits, true_target) + ε·mean(CE(logits, uniform))")
    print()

    V = 5
    logits  = torch.tensor([[2.0, 0.5, 3.0, 0.1, 0.2]])
    targets = torch.tensor([2])   # correct token is index 2

    loss_hard   = F.cross_entropy(logits, targets, label_smoothing=0.0)
    loss_smooth = F.cross_entropy(logits, targets, label_smoothing=0.1)

    print(f"  Logits: {logits[0].tolist()}")
    print(f"  Target: {targets[0].item()} (token index 2)")
    print()
    print(f"  Hard   loss (ε=0.0): {loss_hard.item():.4f}")
    print(f"  Smooth loss (ε=0.1): {loss_smooth.item():.4f}")
    print()
    print("  Smooth loss is higher — the model is prevented from assigning")
    print("  all probability mass to the correct token.")


if __name__ == "__main__":
    verify()
    loss_perplexity_table()
    masked_loss_demo()
    label_smoothing_demo()
''',
    },

    "Gradient Flow and Vanishing/Exploding Gradients": {
        "description": "Visualise how gradients flow through a deep network, demonstrate the vanishing and exploding gradient problems, and show how residual connections fix them.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
GRADIENT FLOW ANALYSIS
================================================================================

Demonstrates three key concepts:
    1. How gradient magnitude changes layer-by-layer during backprop
    2. The vanishing gradient problem (sigmoid, tanh activations + depth)
    3. The exploding gradient problem (poor initialisation)
    4. How residual connections (skip connections) fix vanishing gradients
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Network architectures for comparison ──────────────────────────────────────

class DeepNetVanishing(nn.Module):
    """Deep network with sigmoid activations — exhibits vanishing gradients."""
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))   # sigmoid saturates → small grads
        return x.mean()


class DeepNetReLU(nn.Module):
    """Deep network with ReLU activations — better gradient flow."""
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        # Kaiming init for ReLU
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x.mean()


class DeepNetResidual(nn.Module):
    """Deep network with residual connections — near-perfect gradient flow."""
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        for layer in self.layers:
            x = x + F.relu(layer(x))   # skip connection: add input to output
        return x.mean()


# ── Analysis function ──────────────────────────────────────────────────────────

def analyse_gradient_flow(model: nn.Module, x: torch.Tensor, name: str):
    """
    Run a forward + backward pass and report gradient norms per layer.
    """
    # Reset gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss = model(x)
    loss.backward()

    grad_norms = []
    for i, layer in enumerate(model.layers):
        if layer.weight.grad is not None:
            norm = layer.weight.grad.norm().item()
            grad_norms.append(norm)
        else:
            grad_norms.append(0.0)

    return grad_norms


def bar_chart(value: float, max_val: float, width: int = 30) -> str:
    """ASCII bar proportional to value/max_val."""
    if max_val == 0:
        return " " * width
    filled = int(min(value / max_val, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


if __name__ == "__main__":
    N_LAYERS = 12
    D        = 64
    torch.manual_seed(0)

    x = torch.randn(16, D, requires_grad=False)

    models = [
        ("Sigmoid  (vanishing)", DeepNetVanishing(D, N_LAYERS)),
        ("ReLU     (better)",    DeepNetReLU(D, N_LAYERS)),
        ("Residual (ideal)",     DeepNetResidual(D, N_LAYERS)),
    ]

    print("=" * 65)
    print(f"  GRADIENT FLOW ANALYSIS — {N_LAYERS} layers, d={D}")
    print("=" * 65)
    print()
    print("  Layer index 0 = first layer (closest to input)")
    print("  Layer index N = last layer  (closest to loss)")
    print()
    print("  Expected pattern:")
    print("  • Sigmoid: gradients shrink toward 0 in early layers")
    print("  • ReLU:    gradients more stable but still shrink somewhat")
    print("  • Residual: gradients roughly equal across all layers")
    print()

    all_results = {}
    for name, model in models:
        grads = analyse_gradient_flow(model, x.clone().requires_grad_(False), name)
        all_results[name] = grads

        max_g = max(grads) if grads else 1.0
        print(f"  {name}")
        for i, g in enumerate(grads):
            bar = bar_chart(g, max_g)
            flag = " ⚠️  (vanishing!)" if g < 1e-6 else ""
            print(f"    Layer {i:2d}: {g:.2e}  │{bar}│{flag}")
        print(f"    Ratio (last/first): {grads[-1]/(grads[0]+1e-20):.4f}")
        print()

    # Summary table
    print("=" * 65)
    print("  SUMMARY: first-layer gradient norm")
    print("=" * 65)
    print()
    for name, grads in all_results.items():
        first  = grads[0]
        last   = grads[-1]
        ratio  = last / (first + 1e-20)
        health = "✓ healthy" if ratio > 0.1 else "⚠️  vanishing" if ratio < 0.01 else "ok"
        print(f"  {name:<30}  first={first:.2e}  last={last:.2e}  "
              f"ratio={ratio:.3f}  {health}")

    print()
    print("  The residual connection creates a 'gradient highway' that allows")
    print("  loss signals to flow directly to early layers without passing")
    print("  through N multiplicative transformations.")
    print()
    print("  This is why all modern LLMs use residual connections in every block.")

    # Exploding gradient demo
    print()
    print("=" * 65)
    print("  EXPLODING GRADIENTS — poor initialisation")
    print("=" * 65)
    print()

    class DeepNetBadInit(nn.Module):
        def __init__(self, d, n):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])
            for layer in self.layers:
                nn.init.normal_(layer.weight, std=2.0)  # too large init!

        def forward(self, x):
            for layer in self.layers:
                x = F.tanh(layer(x))
            return x.mean()

    bad_model = DeepNetBadInit(D, 6)
    grads_bad = analyse_gradient_flow(bad_model, x, "bad init")
    print("  Model with std=2.0 initialisation (too large):")
    for i, g in enumerate(grads_bad):
        flag = " 💥 EXPLODING!" if g > 100 else ""
        print(f"    Layer {i}: {g:.2e}{flag}")
    print()
    print("  Fix: use small σ (0.02) or Xavier/Kaiming initialisation.")
    print("  In practice, gradient clipping (Module 09) also prevents this.")
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
    try:
        from llm_training.visuals.training_loop import (
            TRAINING_LOOP_VISUAL_HTML,
            TRAINING_LOOP_VISUAL_HEIGHT,
        )
        visual_html   = TRAINING_LOOP_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = TRAINING_LOOP_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[06_training_loop_fundamentals.py] Could not load visual: {e}", stacklevel=2)

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