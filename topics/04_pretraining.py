"""Module: 04 · Pretraining"""

DISPLAY_NAME = "04 · Pretraining"
ICON = "⚡"
SUBTITLE = "How LLMs are trained at scale — objectives, data and distributed compute"

THEORY = """
## 04 · Pretraining

Pretraining is the foundational phase that turns a randomly initialised transformer into a model
that understands language, facts, and reasoning. Everything downstream — fine-tuning, RLHF,
prompting — stands on what is learned here.

---

### 1 · What Is Pretraining?

Pretraining is **self-supervised learning on massive text corpora**. The model is never given
human-written labels; instead, the training signal comes from the text itself. The model learns
to predict what comes next (or what is masked), and in doing so must implicitly learn grammar,
world knowledge, logical structure, and common-sense reasoning.

The result is a **foundation model** — a single set of weights that can be adapted cheaply
to almost any downstream task.

---

### 2 · Training Objectives

#### 2.1 Causal Language Modelling (CLM) — "Next-token prediction"
Used by GPT-style (decoder-only) models such as GPT-4, LLaMA, Mistral, Claude.

Given a sequence of tokens x₁, x₂, …, xₙ the objective is to maximise the log-likelihood
of every token given all tokens to its left:

```
L_CLM = − (1/N) · Σᵢ log P(xᵢ | x₁ … xᵢ₋₁ ; θ)
```

The model is trained with a **causal (upper-triangular) attention mask** so that position i
cannot attend to position j > i. This makes every forward pass simultaneously train on all
N prefix–next-token pairs in the sequence, which is extremely data-efficient.

#### 2.2 Masked Language Modelling (MLM) — "Fill in the blank"
Used by BERT-style (encoder-only) models.

A random 15 % of tokens are replaced with a [MASK] token, and the model must recover them:

```
L_MLM = − Σ_{masked i} log P(xᵢ | x₁ … x_{i−1}, [MASK], x_{i+1} … xₙ ; θ)
```

Because the model sees future tokens, MLM builds richer bidirectional representations but
**cannot be used directly for generation**.

#### 2.3 Span / Prefix LM objectives
T5 uses a **span-corruption** objective: contiguous spans are masked and the decoder must
regenerate them. This bridges encoder–decoder and pure CLM training, and works well for
seq-to-seq tasks like summarisation and translation.

---

### 3 · Training Data

#### 3.1 Data sources
| Source | Example datasets | Characteristics |
|---|---|---|
| Web crawls | Common Crawl, C4, FineWeb | Huge volume; noisy |
| Curated web | OpenWebText, RefinedWeb | Filtered for quality |
| Books | Books3, Gutenberg | Long-range coherence |
| Code | GitHub, The Stack | Structured, logical |
| Academic | arXiv, PubMed, S2ORC | Precise, technical |
| Wikipedia / Wikidata | English + multilingual | Factual, encyclopaedic |
| Conversations | Reddit, StackOverflow | Dialogue, QA style |

A typical large model trains on **trillions of tokens** — GPT-3 used ~300 B, LLaMA-2 used
2 T, and Llama-3 used 15 T.

#### 3.2 Data processing pipeline
```
Raw dumps → Language ID → Deduplication → Quality filtering
          → PII scrubbing → Tokenisation → Shuffling → Packing
```

Key steps in detail:

**Deduplication** — Exact and near-duplicate documents inflate apparent corpus size and cause
models to memorise rather than generalise. MinHash LSH is commonly used for fuzzy deduplication
at scale.

**Quality filtering** — Heuristic classifiers remove spam, boilerplate, toxic content, and
low-coherence text. C4 uses a simple "ends in punctuation, no JavaScript" filter; FineWeb
uses a heavier ML classifier trained on high-quality reference text.

**Domain mixing / upsampling** — Even though web data dominates by volume, high-quality
domains (code, books, maths) are upsampled to improve downstream capability. The mixing
ratios are hyperparameters tuned empirically.

**Tokenisation** — Text is converted to integer IDs using a BPE or Unigram tokeniser (e.g.
tiktoken for GPT models, SentencePiece for LLaMA). Vocabulary sizes are typically 32 K–128 K.

**Sequence packing** — Documents are concatenated and sliced into fixed-length windows
(e.g. 2 048, 4 096, 8 192 tokens) to maximise GPU utilisation. A special `<EOS>` boundary
token separates documents so the model learns document boundaries.

---

### 4 · The Training Loop

```
for each micro-batch:
    1. Forward pass   → compute logits for all positions in parallel
    2. Loss           → cross-entropy between logits and shifted input
    3. Backward pass  → compute gradients via backprop-through-time (BPTT)
    4. Gradient sync  → all-reduce across data-parallel workers
    5. Optimiser step → update weights (AdamW + LR schedule)
    6. Repeat
```

#### 4.1 Optimiser — AdamW
AdamW is the de-facto standard. It combines Adam's adaptive learning rates with **decoupled
weight decay** (L2 regularisation applied directly to weights, not to the gradient):

```
mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ          # 1st moment (momentum)
vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²         # 2nd moment (variance)
m̂ₜ = mₜ / (1 − β₁ᵗ)                      # bias correction
v̂ₜ = vₜ / (1 − β₂ᵗ)
θₜ = θₜ₋₁ − η · m̂ₜ/(√v̂ₜ + ε) − η·λ·θₜ₋₁  # update + weight decay
```

Typical hyperparameters: β₁=0.9, β₂=0.95, ε=1e-8, λ=0.1

#### 4.2 Learning Rate Schedule
Most large runs use a **warm-up + cosine decay** schedule:

- **Warm-up phase** (first ~1–2 % of steps): LR rises linearly from 0 to η_max.
  This prevents large early gradient steps that destabilise layer norms.
- **Cosine decay**: LR anneals from η_max to η_min ≈ η_max/10 over the rest of training.

```
η(t) = η_min + 0.5 · (η_max − η_min) · (1 + cos(π · t / T))
```

#### 4.3 Gradient Clipping
Gradient norm is clipped to a threshold (typically 1.0) to prevent training instabilities
("spikes") that are common at large scale:

```
if ‖g‖ > clip_threshold:
    g ← g · (clip_threshold / ‖g‖)
```

#### 4.4 Mixed Precision (BF16 / FP16)
Weights are stored in FP32 for numerical stability; forward and backward passes run in
BF16 (preferred over FP16 for LLMs because BF16 has the same exponent range as FP32,
avoiding overflow on large activations). This roughly halves memory and doubles throughput.

---

### 5 · Distributed Training

A 70 B-parameter model in BF16 requires ~140 GB just for weights — far beyond a single GPU.
Distributed training spreads the work across hundreds or thousands of GPUs.

#### 5.1 Data Parallelism (DP)
Each GPU holds a **full copy** of the model and processes a different mini-batch.
After the backward pass, gradients are averaged across all workers (all-reduce).

```
GPU 0: batch_A → grads_A ─┐
GPU 1: batch_B → grads_B ──┼─ all-reduce → avg_grad → each GPU updates its copy
GPU 2: batch_C → grads_C ─┘
```

**ZeRO** (Zero Redundancy Optimizer, used in DeepSpeed) partitions the optimiser state,
gradients, and/or parameters across DP workers, eliminating the memory redundancy of naive DP.

| ZeRO Stage | What is sharded | Memory saving |
|---|---|---|
| Stage 1 | Optimiser states | ~4× |
| Stage 2 | + Gradients | ~8× |
| Stage 3 | + Parameters | ~64× |

#### 5.2 Tensor Parallelism (TP)
Individual weight matrices are split across GPUs along one dimension. For an MLP with weight
matrix W ∈ ℝ^{d×4d}, column-parallel TP splits it into P chunks of size d×(4d/P).
A single matrix multiplication requires one all-reduce per layer but no replication of weights.

Used *within* a node (NVLink bandwidth >> InfiniBand).

#### 5.3 Pipeline Parallelism (PP)
Transformer layers are divided into **stages**, each stage living on a different set of GPUs.
Forward activations are passed between stages; each stage computes gradients for its layers.

**Micro-batching** (GPipe / 1F1B schedule) keeps GPUs busy by feeding multiple micro-batches
through the pipeline, hiding the inter-stage communication bubble.

#### 5.4 Sequence / Context Parallelism
For very long context (≥32 K tokens), the sequence dimension itself is sharded across GPUs,
with ring-attention used to compute attention scores without materialising the full attention
matrix on any single device.

#### 5.5 3D Parallelism
In practice, all three strategies are composed:
```
World = DP_degree × TP_degree × PP_degree
e.g.  64 nodes × 8-way TP × 8-way PP × 8-way DP = 4 096 GPUs
```

---

### 6 · Scaling Laws

Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") showed that loss follows
predictable power laws in model size N, dataset size D, and compute budget C = 6ND:

```
L(N, D) ≈ A/N^α + B/D^β + L_∞
```

**Chinchilla optimal**: for a given compute budget C, loss is minimised when
```
N_opt ∝ C^0.5   and   D_opt ∝ C^0.5   (roughly 20 tokens per parameter)
```

This showed that GPT-3 (175 B params, 300 B tokens) was significantly undertrained — a
70 B model trained on 1.4 T tokens would match it at 1/3 the inference cost.

Modern practice (LLaMA-3, Mistral) often trains **beyond** Chinchilla optimal because
inference is cheap relative to training, so smaller but more-trained models are preferred.

---

### 7 · Training Stability & Common Failure Modes

| Issue | Symptom | Mitigation |
|---|---|---|
| Loss spike | Sudden jump in loss mid-training | Rollback checkpoint; lower LR; clip gradients |
| Gradient explosion | Norm → ∞ | Gradient clipping; check LR warm-up |
| Loss plateau early | Loss stops decreasing | Check data pipeline; increase LR or batch size |
| NaN loss | Loss becomes NaN | BF16 overflow; bad data batch; check tokeniser |
| Rank collapse | Attention heads become identical | QK-Norm; better init; lower LR |

---

### 8 · Compute & Hardware

Modern pretraining runs on **GPU clusters** (NVIDIA H100 / A100) or **TPU pods** (Google).

A rough compute estimate:
```
FLOPs ≈ 6 · N · D
```
where N = parameters, D = training tokens.

For a 7B model on 1T tokens:
```
FLOPs ≈ 6 × 7×10⁹ × 10¹² = 4.2×10²² FLOPs
```

An H100 delivers ~1 000 TFLOPs (BF16), so at 50 % MFU:
```
Time ≈ 4.2×10²² / (500×10¹²) ≈ 84 000 GPU-seconds ≈ ~23 GPU-hours per GPU
      → with 1 000 GPUs ≈ ~84 seconds wall-clock  (unrealistically idealised)
```
Real runs on 1 000 GPUs typically take **weeks** due to communication overhead,
checkpointing, restarts, and lower MFU.

---

---

### 9 · How Training Is Actually Carried Out — Step by Step

This section walks through every stage of a single training step in mechanistic detail,
from raw token IDs all the way back to a weight update.

---

#### 9.1 Model Initialisation

Before the first step, every parameter is initialised carefully. Poor initialisation causes
gradients to explode or vanish in the very first forward pass.

**Embedding table** — drawn from N(0, 1/√d_model). The small scale prevents the first
attention logits from saturating the softmax.

**Attention projections (Q, K, V, O)** — typically N(0, σ) where σ = 0.02 or
σ = 1/√(d_model). The output projection O is often scaled down by 1/√(2 × n_layers)
(the "GPT-2 trick") so that residual additions don't explode at depth.

**MLP weights** — same as attention, with the second linear layer scaled by 1/√(2 × n_layers).

**Layer norms** — γ = 1, β = 0 (identity transform at initialisation).

**Biases** — set to zero.

---

#### 9.2 The Forward Pass in Detail

Given a batch of shape (B, T) token IDs, the forward pass proceeds as follows:

```
STEP 1 — Token Embedding
  x = E[tokens]           # (B, T, d_model)   — look up embedding table

STEP 2 — Positional Encoding
  x = x + PE              # add sinusoidal or learned positional vectors
                           # (both are (T, d_model) broadcast over B)

STEP 3 — L × Transformer Layers
  for each layer ℓ = 1 … L:
      x = x + Attention(LayerNorm(x))   # pre-norm residual
      x = x + MLP(LayerNorm(x))         # pre-norm residual

STEP 4 — Final LayerNorm
  x = LayerNorm(x)         # (B, T, d_model)

STEP 5 — Unembedding (logits)
  logits = x @ W_U         # (B, T, V)   — W_U is the vocab projection
                           # often tied to E (shared embedding weights)
```

##### Inside a Transformer Layer

**Pre-norm vs Post-norm** — Modern LLMs use *pre-norm* (LayerNorm before the sublayer,
not after). This stabilises gradients at depth; post-norm requires careful warm-up tuning.

**LayerNorm** normalises the d_model dimension of each token independently:
```
LN(x) = γ · (x − μ) / (σ + ε) + β
  where μ, σ = mean and std over d_model
```
RMSNorm (used in LLaMA) drops the mean subtraction (μ = 0) for ~10 % speed gain.

**Multi-Head Self-Attention (MHSA)**:
```
Q = x W_Q    (B, T, d_k × H)  — split into H heads after projection
K = x W_K
V = x W_V

For each head h:
  scores_h = Q_h @ K_hᵀ / √d_k                  (B, T, T)
  scores_h = scores_h + causal_mask               (−∞ for future positions)
  attn_h   = softmax(scores_h)                    (B, T, T)
  head_h   = attn_h @ V_h                         (B, T, d_k)

x_attn = concat(head_1 … head_H) @ W_O            (B, T, d_model)
```

The causal mask sets upper-triangular entries to −∞ before softmax, which forces
them to zero after softmax, so every token can only look at itself and earlier tokens.

**MLP (FFN)**:
```
h = GELU(x W_1 + b_1)    # (B, T, 4 · d_model)  — expand 4×
x = h W_2 + b_2           # (B, T, d_model)       — contract back
```
Modern models replace GELU with SwiGLU:
```
SwiGLU(x) = SiLU(x W_1) ⊙ (x W_3)    # element-wise gate
x_mlp     = SwiGLU(x) @ W_2
```
SwiGLU is empirically better and is used in LLaMA, Mistral, and PaLM.

---

#### 9.3 Computing the Loss

After the forward pass we have logits of shape (B, T, V). The CLM loss:

```
# Shift: predict token i+1 from position i
shift_logits = logits[:, :-1, :]     # (B, T-1, V)
shift_labels = tokens[:, 1:]         # (B, T-1)

# Cross-entropy (numerically stable via log-sum-exp)
log_probs = log_softmax(shift_logits, dim=-1)
loss = -log_probs[b, t, shift_labels[b,t]].mean()
```

The mean is taken over all (B × (T−1)) token positions simultaneously — this is what
makes CLM so data-efficient compared to MLM which only trains on ~15 % of positions.

---

#### 9.4 The Backward Pass

PyTorch's autograd (or JAX's grad) traces the computation graph built during the forward
pass and applies the chain rule backwards from the scalar loss to every leaf parameter.

The key gradient flows to understand:

**Through the unembedding**: ∂L/∂logits = softmax(logits) − one_hot(target)
This is the "error signal" — the probability difference between what was predicted and
what was correct. Large errors on common tokens drive the biggest weight updates.

**Through LayerNorm**: Gradients pass almost unchanged (γ ≈ 1 at init), which is why
pre-norm is stable. In post-norm, gradients must flow through LN at every residual
junction and can vanish if σ grows too large.

**Through attention**: ∂L/∂Q, ∂L/∂K, ∂L/∂V each require computing through the softmax.
The softmax Jacobian is (diag(a) − a aᵀ) where a is the attention distribution. When
attention is peaked (low temperature), gradients to Q and K become sparse and small.

**Through residual connections**: Because x_out = x_in + f(x_in), the gradient is:
```
∂L/∂x_in = ∂L/∂x_out · (1 + ∂f/∂x_in)
```
The "1" term creates a **gradient highway** — gradients always have a direct path back
to earlier layers with no multiplication. This is why deep transformers (100+ layers)
train stably while deep plain networks do not.

**Activation checkpointing (gradient checkpointing)** — Storing all activations from
a forward pass requires memory ∝ B × T × d_model × L. For a 7B model with B=4, T=4096,
this is ~50 GB. Recomputation trades memory for compute: activations are discarded after
the forward pass and recomputed during the backward pass. This reduces activation memory
by ~√L at a cost of ~33 % extra compute.

---

#### 9.5 Gradient Accumulation

In large-scale training, the effective batch size B_eff is a key hyperparameter.
Larger batches give better gradient estimates (lower variance) and enable faster convergence
per token, but require more GPU memory.

**Gradient accumulation** simulates a large batch without the memory cost:

```
optimizer.zero_grad()

for micro_step in range(grad_accum_steps):
    micro_batch = next(data_loader)
    loss = model(micro_batch) / grad_accum_steps   # scale loss!
    loss.backward()                                 # accumulate gradients

# Only after all micro-steps:
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

The effective batch size is: B_eff = micro_batch_size × grad_accum_steps × num_GPUs

Example: 8 GPUs × micro_batch=4 × accum_steps=8 = B_eff of 256 sequences.
At T=4096 tokens/seq → ~1M tokens per step, which is the typical target for large runs.

**Why scale the loss?** Each micro-batch loss is divided by grad_accum_steps so the
accumulated gradient equals the gradient you would get on the full batch. Forgetting
this is a common bug that inflates the effective learning rate by grad_accum_steps×.

---

#### 9.6 Gradient Clipping — Why It Matters

Gradient norm clipping is not just a safety net; it is an integral part of the training
recipe:

```
global_norm = √( Σ_p ‖∇p‖² )    # sum over ALL parameter tensors

if global_norm > clip_value:
    scale = clip_value / global_norm
    for each parameter p:
        ∇p ← ∇p · scale
```

**What triggers large gradients?**
- A "bad" batch with unusual token distributions (e.g., all-caps, code with rare symbols)
- Attention collapse (all heads attending to one position)
- Early training when loss is high and errors are large
- Residual branch outputs suddenly growing due to lucky/unlucky initialisation

**Monitoring gradient norms** reveals health of the run:
- Norm slowly decreasing over training → healthy convergence
- Norm suddenly spiking 5–10× → likely loss spike incoming
- Norm consistently at the clip threshold → LR may be too high
- Norm → 0 for certain layers → vanishing gradients, check LN or init

---

#### 9.7 The Optimiser Step — Inside AdamW

After clipping, each parameter tensor p is updated:

```
# Per-parameter update rule
m_p ← β₁ · m_p + (1 − β₁) · ∇p        # EMA of gradient
v_p ← β₂ · v_p + (1 − β₂) · (∇p)²     # EMA of squared gradient
m̂_p = m_p / (1 − β₁ᵗ)                  # bias correction (important early in training)
v̂_p = v_p / (1 − β₂ᵗ)
p   ← p · (1 − η·λ) − η · m̂_p / (√v̂_p + ε)   # update
```

The key insight is **per-parameter adaptive learning rates**: parameters with consistently
large gradients (high v_p) get a smaller effective step, while rarely-updated parameters
(low v_p) get relatively larger steps. This makes AdamW excellent for sparse features
like rare vocabulary tokens in the embedding table.

**Weight decay** (λ, decoupled in AdamW): Shrinks every weight toward zero each step.
This acts as L2 regularisation, preventing parameters from growing unboundedly. Bias
vectors and LayerNorm gain/bias are typically *excluded* from weight decay.

---

#### 9.8 A Complete Single Training Step — Putting It All Together

```
╔══════════════════════════════════════════════════════════╗
║           ONE TRAINING STEP (pseudocode)                ║
╚══════════════════════════════════════════════════════════╝

[SETUP — done once before training]
model   = Transformer(n_layers, d_model, n_heads, vocab_size)
init_weights(model)                     # careful initialisation
optimiser = AdamW(model.params(), lr=lr_max, betas=(0.9, 0.95),
                  weight_decay=0.1)
lr_schedule = CosineWithWarmup(total_steps, warmup_steps)

[TRAINING LOOP]
for global_step in range(total_steps):

  optimiser.zero_grad()

  total_loss = 0
  for micro_step in range(grad_accum_steps):            # ① DATA
      tokens = next(dataloader)                         #   fetch (B, T) token IDs

      with autocast(dtype=bfloat16):                    # ② FORWARD
          logits = model(tokens)                        #   (B, T, V)
          loss   = cross_entropy(                       # ③ LOSS
              logits[:, :-1], tokens[:, 1:]
          ) / grad_accum_steps

      loss.backward()                                   # ④ BACKWARD
      total_loss += loss.item()

  grad_norm = clip_grad_norm_(model.params(), 1.0)      # ⑤ CLIP

  lr = lr_schedule(global_step)
  set_lr(optimiser, lr)

  optimiser.step()                                      # ⑥ UPDATE

  if global_step % ckpt_interval == 0:                 # ⑦ CHECKPOINT
      save_checkpoint(model, optimiser, global_step)

  log(step=global_step, loss=total_loss,               # ⑧ LOG
      grad_norm=grad_norm, lr=lr, tokens_seen=...)
```

Each of the 8 numbered actions must happen in the correct order. A common mistake
is placing `zero_grad()` after `step()` in the wrong iteration, or forgetting to
divide the loss during accumulation — both silently corrupt training.

---

#### 9.9 What the Model Actually Learns — Layer by Layer

Research using probing classifiers, attention pattern visualisation, and mechanistic
interpretability reveals a rough curriculum that emerges across layers:

| Layer depth | What is learned |
|---|---|
| 1–2 (early) | Local syntax: POS tags, adjacent word dependencies |
| 3–6 | Phrase structure, named entity recognition |
| 7–12 | Coreference resolution, long-range subject–verb agreement |
| 13–20 | Factual recall, entity attributes, simple reasoning |
| 20+ (late) | Abstract reasoning, task adaptation, instruction following |

This is emergent — not designed. The gradient signal from next-token prediction
naturally organises representations this way because earlier layers need to build
the right features for later layers to predict well.

---

### Key Takeaways

- Pretraining = self-supervised next-token (or masked-token) prediction on trillions of tokens.
- Data quality and mixing ratios matter as much as model architecture.
- The forward pass flows: embed → L × (LN → attn → residual → LN → MLP → residual) → LN → unembed.
- Residual connections create a gradient highway that enables training at extreme depth.
- AdamW + cosine LR schedule + gradient clipping is the standard recipe.
- Gradient accumulation decouples effective batch size from GPU memory.
- Distributed training (DP + TP + PP + ZeRO) is necessary for models > ~1 B parameters.
- Scaling laws let practitioners predict loss and choose compute-optimal N and D before training.
- Training instabilities are inevitable at scale; robust checkpointing and monitoring are essential.
"""

# ---------------------------------------------------------------------------
# Interactive Operations
# ---------------------------------------------------------------------------

OPERATIONS = {

    # ------------------------------------------------------------------
    # Step 1 — Tokenisation & Sequence Packing
    # ------------------------------------------------------------------
    "Step 1 · Tokenisation & Sequence Packing": {
        "description": (
            "Demonstrates how raw text is tokenised with BPE and then packed into "
            "fixed-length windows — the first stage of every pretraining data pipeline."
        ),
        "language": "python",
        "code": """
import re
from collections import Counter, defaultdict

# ── Tiny BPE tokeniser (pure Python, no dependencies) ──────────────────────

def get_vocab(corpus: list[str]) -> dict[str, int]:
    \"\"\"Build initial character-level vocabulary with </w> end-of-word marker.\"\"\"
    vocab: dict[str, int] = defaultdict(int)
    for sentence in corpus:
        for word in sentence.strip().split():
            vocab[' '.join(list(word)) + ' </w>'] += 1
    return dict(vocab)

def get_pairs(vocab: dict[str, int]) -> Counter:
    \"\"\"Count all adjacent symbol pairs across the vocabulary.\"\"\"
    pairs: Counter = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for a, b in zip(symbols[:-1], symbols[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_vocab(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    \"\"\"Merge the most frequent pair into a new symbol.\"\"\"
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!' + re.escape(' ') + r')' + bigram + r'(?!' + re.escape(' ') + r')')
    for word in vocab:
        new_word = re.sub(' '.join(pair), ''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def bpe_train(corpus: list[str], num_merges: int = 20) -> tuple[dict, list]:
    vocab = get_vocab(corpus)
    merges = []
    for _ in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return vocab, merges

# ── Sequence packing ────────────────────────────────────────────────────────

def naive_tokenise(text: str) -> list[str]:
    \"\"\"Character-split for illustration purposes.\"\"\"
    return list(text.replace(' ', '_'))

def pack_sequences(documents: list[str], max_len: int = 16,
                   eos: str = '<EOS>') -> list[list[str]]:
    \"\"\"
    Concatenate tokenised documents with EOS separators and slice into
    fixed-length windows of `max_len` tokens.
    \"\"\"
    stream: list[str] = []
    for doc in documents:
        stream.extend(naive_tokenise(doc))
        stream.append(eos)

    windows = [stream[i:i + max_len] for i in range(0, len(stream) - max_len + 1, max_len)]
    return windows

# ── Demo ────────────────────────────────────────────────────────────────────

corpus = [
    "the cat sat on the mat",
    "the cat ate the rat",
    "a cat is a mammal",
    "rats and cats are animals",
]

print("=" * 60)
print("BPE TRAINING")
print("=" * 60)
vocab, merges = bpe_train(corpus, num_merges=15)
print(f"Top 10 BPE merges learned:")
for i, m in enumerate(merges[:10], 1):
    print(f"  {i:2d}. '{m[0]}' + '{m[1]}' → '{''.join(m)}'")

print()
print("Final vocabulary tokens (sorted by length):")
all_tokens = sorted({t for word in vocab for t in word.split()}, key=len)
print(" ", all_tokens)

print()
print("=" * 60)
print("SEQUENCE PACKING  (window size = 16 tokens)")
print("=" * 60)
documents = [
    "hello world",
    "the quick brown fox",
    "jumps over the lazy dog",
]
windows = pack_sequences(documents, max_len=16)
for i, w in enumerate(windows):
    print(f"  Window {i}: {w}")
print(f"\\n  {len(documents)} documents → {len(windows)} packed windows")
print("  (no padding waste — GPU sees full sequences)")
""",
    },

    # ------------------------------------------------------------------
    # Step 2 — Causal Language Modelling Objective
    # ------------------------------------------------------------------
    "Step 2 · Causal LM Objective (Next-Token Prediction)": {
        "description": (
            "Implements the causal (autoregressive) cross-entropy loss with a causal "
            "attention mask from scratch using only NumPy."
        ),
        "language": "python",
        "code": """
import numpy as np

np.random.seed(42)

# ── Hyper-parameters ────────────────────────────────────────────────────────
VOCAB_SIZE = 8        # tiny vocab for illustration
SEQ_LEN    = 6        # tokens per sequence
D_MODEL    = 16       # embedding dimension

# ── Dummy token sequence ────────────────────────────────────────────────────
# Represents the token IDs: [2, 5, 1, 3, 7, 0]
tokens = np.array([2, 5, 1, 3, 7, 0])

# ── Random weight matrix (embedding table + output projection) ───────────────
W_emb = np.random.randn(VOCAB_SIZE, D_MODEL) * 0.1  # embedding
W_out = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.1  # unembedding (tied weights in practice)

# ── Causal attention mask ────────────────────────────────────────────────────
# Lower-triangular matrix: position i may only attend to positions ≤ i
causal_mask = np.tril(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool))
print("Causal attention mask (True = allowed to attend):")
print(causal_mask.astype(int))

# ── Forward pass (simplified — no real attention, just linear) ──────────────
# In a real transformer, each position would attend only to previous positions.
# Here we directly compute logits from embeddings to illustrate the loss.

x = W_emb[tokens]          # (SEQ_LEN, D_MODEL) — look up embeddings
logits = x @ W_out          # (SEQ_LEN, VOCAB_SIZE) — project to vocab

# ── Softmax ──────────────────────────────────────────────────────────────────
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

probs = softmax(logits)     # (SEQ_LEN, VOCAB_SIZE)

# ── Cross-entropy loss ───────────────────────────────────────────────────────
# For CLM we predict x_{i+1} given x_{1..i}
# → inputs  = tokens[:-1]   (positions 0 … T-2)
# → targets = tokens[1:]    (positions 1 … T-1)

inputs  = tokens[:-1]
targets = tokens[1:]
pred_probs = probs[:-1]     # logits computed at input positions

# Gather probability assigned to the correct next token
correct_log_probs = np.log(pred_probs[np.arange(len(targets)), targets] + 1e-9)
loss = -correct_log_probs.mean()

print()
print(f"Token sequence : {tokens}")
print(f"Inputs  (x_0…x_{{T-2}}): {inputs}")
print(f"Targets (x_1…x_{{T-1}}): {targets}")
print()
print("Per-position log-probabilities assigned to correct token:")
for i, (inp, tgt, lp) in enumerate(zip(inputs, targets, correct_log_probs)):
    p = np.exp(lp)
    print(f"  pos {i}: token {inp} → predicts {tgt}  | P(correct) = {p:.4f}  | -log P = {-lp:.4f}")

print()
print(f"  Cross-entropy loss = {loss:.4f}")
print(f"  Perplexity         = {np.exp(loss):.2f}  "
      f"(random baseline for vocab={VOCAB_SIZE}: {VOCAB_SIZE:.2f})")
print()
print("Note: perplexity ≈ vocab size means the model has learned nothing yet.")
print("After pretraining on trillions of tokens it drops to ~10–20 for language.")
""",
    },

    # ------------------------------------------------------------------
    # Step 3 — AdamW Optimiser
    # ------------------------------------------------------------------
    "Step 3 · AdamW Optimiser": {
        "description": (
            "Implements AdamW from scratch and compares it to vanilla SGD on a "
            "simple quadratic loss surface, showing faster convergence."
        ),
        "language": "python",
        "code": """
import numpy as np
import math

# ── Loss surface: f(θ) = 0.5 * θᵀ A θ  (ill-conditioned quadratic) ──────────
# A has eigenvalues 1 and 100 — a narrow ravine that confounds SGD.
A = np.array([[100., 0.], [0., 1.]])

def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
    loss = 0.5 * theta @ A @ theta
    grad = A @ theta
    return float(loss), grad

# ── Vanilla SGD ──────────────────────────────────────────────────────────────
def sgd(steps=200, lr=0.01):
    theta = np.array([1.0, 1.0])
    history = [float(0.5 * theta @ A @ theta)]
    for _ in range(steps):
        _, g = loss_and_grad(theta)
        theta -= lr * g
        history.append(float(0.5 * theta @ A @ theta))
    return history

# ── AdamW ────────────────────────────────────────────────────────────────────
def adamw(steps=200, lr=1e-2, beta1=0.9, beta2=0.999,
          eps=1e-8, weight_decay=0.01):
    theta = np.array([1.0, 1.0])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = [float(0.5 * theta @ A @ theta)]
    for t in range(1, steps + 1):
        _, g = loss_and_grad(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        # AdamW: weight decay directly on θ, NOT added to gradient
        theta = theta * (1 - lr * weight_decay) - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(float(0.5 * theta @ A @ theta))
    return history

# ── LR schedule: warm-up + cosine decay ─────────────────────────────────────
def cosine_with_warmup(total_steps: int, warmup_steps: int,
                       lr_max: float, lr_min: float) -> list[float]:
    schedule = []
    for t in range(total_steps):
        if t < warmup_steps:
            lr = lr_max * (t + 1) / warmup_steps
        else:
            progress = (t - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
        schedule.append(lr)
    return schedule

# ── Run ──────────────────────────────────────────────────────────────────────
STEPS = 200
sgd_losses   = sgd(STEPS)
adamw_losses = adamw(STEPS)
schedule     = cosine_with_warmup(total_steps=1000, warmup_steps=50,
                                   lr_max=3e-4, lr_min=3e-5)

print("=" * 55)
print("OPTIMISER COMPARISON (ill-conditioned quadratic)")
print("=" * 55)
header = f"{'Step':>6}  {'SGD loss':>12}  {'AdamW loss':>12}"
print(header)
print("-" * len(header))
for step in [0, 10, 25, 50, 100, 200]:
    print(f"{step:>6}  {sgd_losses[step]:>12.6f}  {adamw_losses[step]:>12.6f}")

print()
print(f"SGD   final loss : {sgd_losses[-1]:.8f}")
print(f"AdamW final loss : {adamw_losses[-1]:.8f}")
print()
print("=" * 55)
print("COSINE LR SCHEDULE WITH LINEAR WARM-UP (1 000 steps)")
print("=" * 55)
checkpoints = [0, 25, 50, 100, 250, 500, 750, 999]
for s in checkpoints:
    bar = "█" * int(schedule[s] / 3e-4 * 30)
    print(f"  step {s:>4}: lr = {schedule[s]:.2e}  {bar}")
print()
print("AdamW's adaptive per-parameter LR makes it far more robust")
print("to ill-conditioned loss surfaces than vanilla SGD.")
""",
    },

    # ------------------------------------------------------------------
    # Step 4 — Scaling Laws
    # ------------------------------------------------------------------
    "Step 4 · Scaling Laws & Chinchilla Optimality": {
        "description": (
            "Implements the Chinchilla scaling law to predict model loss and find "
            "the compute-optimal (N, D) allocation for a given FLOPs budget."
        ),
        "language": "python",
        "code": """
import math

# ── Chinchilla scaling law ───────────────────────────────────────────────────
# L(N, D) ≈ A/N^α + B/D^β + L_inf
# Fitted constants from Hoffmann et al. 2022 (Table A3, Approach 1)
A     = 406.4
B     = 410.7
ALPHA = 0.34
BETA  = 0.28
L_INF = 1.69        # irreducible loss

def predict_loss(N: float, D: float) -> float:
    \"\"\"Predict cross-entropy loss given N parameters and D training tokens.\"\"\"
    return A / (N ** ALPHA) + B / (D ** BETA) + L_INF

# ── Compute budget → optimal N and D ────────────────────────────────────────
# C ≈ 6·N·D  (FLOPs for a dense transformer forward + backward)
# Under Chinchilla: N_opt ∝ C^0.5,  D_opt ∝ C^0.5
# More precise:  N_opt = (A·α / B·β)^{1/(α+β)} · (C/6)^{β/(α+β)}
#                D_opt = C / (6 · N_opt)

def chinchilla_optimal(C: float) -> tuple[float, float]:
    \"\"\"Return (N_opt, D_opt) for FLOPs budget C.\"\"\"
    ratio = (A * ALPHA) / (B * BETA)
    N_opt = ratio ** (1 / (ALPHA + BETA)) * (C / 6) ** (BETA / (ALPHA + BETA))
    D_opt = C / (6 * N_opt)
    return N_opt, D_opt

def fmt(x: float) -> str:
    \"\"\"Human-readable number (B / T).\"\"\"
    if x >= 1e12: return f"{x/1e12:.1f}T"
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.1f}M"
    return f"{x:.0f}"

# ── Known models ─────────────────────────────────────────────────────────────
models = [
    ("GPT-3",        175e9,  300e9),
    ("Gopher",       280e9,  300e9),
    ("Chinchilla",    70e9, 1400e9),
    ("LLaMA-1 7B",    7e9, 1000e9),
    ("LLaMA-2 70B",  70e9, 2000e9),
    ("LLaMA-3 8B",    8e9,   15e12),
    ("Mistral-7B",    7e9, 1000e9),
]

print("=" * 70)
print(f"{'Model':<20} {'N':>8} {'D':>8} {'C (FLOPs)':>14} {'Loss':>6}")
print("=" * 70)
for name, N, D in models:
    C    = 6 * N * D
    loss = predict_loss(N, D)
    print(f"{name:<20} {fmt(N):>8} {fmt(D):>8} {fmt(C):>14} {loss:>6.3f}")

print()
print("=" * 70)
print("CHINCHILLA-OPTIMAL (N, D) FOR DIFFERENT COMPUTE BUDGETS")
print("=" * 70)
print(f"{'Budget':>14}  {'N_opt':>10}  {'D_opt':>10}  {'Loss':>6}  tokens/param")
print("-" * 60)
budgets = [1e19, 1e20, 1e21, 1e22, 1e23, 1e24]
for C in budgets:
    N_opt, D_opt = chinchilla_optimal(C)
    loss = predict_loss(N_opt, D_opt)
    ratio = D_opt / N_opt
    print(f"{fmt(C):>14}  {fmt(N_opt):>10}  {fmt(D_opt):>10}  {loss:>6.3f}  {ratio:.1f}×")

print()
print("GPT-3 compute budget: ~3.1×10²³ FLOPs")
N_gpt3, D_gpt3 = chinchilla_optimal(3.1e23)
print(f"  Chinchilla-optimal would be: N={fmt(N_gpt3)}, D={fmt(D_gpt3)}")
print(f"  GPT-3 actual:                N=175B, D=300B  (severely undertrained!)")
print()
print("Key insight: for a fixed compute budget, you get *lower loss*")
print("by training a SMALLER model on MORE data than a large model on less data.")
""",
    },

    # ------------------------------------------------------------------
    # Step 5 — ZeRO Memory Partitioning
    # ------------------------------------------------------------------
    "Step 5 · ZeRO Memory Partitioning": {
        "description": (
            "Simulates ZeRO Stage 1/2/3 memory savings versus naive data parallelism, "
            "showing how large models can be trained across many GPUs."
        ),
        "language": "python",
        "code": """
# ZeRO (Zero Redundancy Optimizer) memory analysis
# Reference: Rajbhandari et al. 2020 — "ZeRO: Memory Optimizations Toward
#            Training Trillion Parameter Models"

def bytes_to_gb(b: float) -> str:
    return f"{b / 1e9:.2f} GB"

def analyse_zero(model_params: int, num_gpus: int,
                 mixed_precision: bool = True) -> None:
    \"\"\"
    Calculate per-GPU memory usage under different ZeRO stages.

    Memory breakdown per parameter (mixed precision BF16 training):
      - Parameters      : 2 bytes  (BF16)
      - Gradients       : 2 bytes  (BF16)
      - FP32 master copy: 4 bytes  (for optimiser)
      - Adam m (FP32)   : 4 bytes
      - Adam v (FP32)   : 4 bytes
      Total: 16 bytes/param  (often quoted as 16× parameter count in bytes)
    \"\"\"
    bytes_per_param     = 2     # BF16 param
    bytes_per_grad      = 2     # BF16 gradient
    bytes_per_opt_state = 4+4+4 # FP32 master + Adam m + Adam v

    total_param_bytes = model_params * bytes_per_param
    total_grad_bytes  = model_params * bytes_per_grad
    total_opt_bytes   = model_params * bytes_per_opt_state

    print(f"Model: {model_params/1e9:.1f}B parameters  |  {num_gpus} GPUs")
    print(f"Total weights :  {bytes_to_gb(total_param_bytes)}")
    print(f"Total grads   :  {bytes_to_gb(total_grad_bytes)}")
    print(f"Total opt st. :  {bytes_to_gb(total_opt_bytes)}")
    print(f"Full model mem:  {bytes_to_gb(total_param_bytes + total_grad_bytes + total_opt_bytes)}")
    print()

    # Per-GPU memory at each ZeRO stage
    stages = {
        "Naive DP (No ZeRO)": {
            "params": total_param_bytes,
            "grads":  total_grad_bytes,
            "opt":    total_opt_bytes,
            "note":   "Full copy on every GPU",
        },
        "ZeRO-1 (shard opt states)": {
            "params": total_param_bytes,
            "grads":  total_grad_bytes,
            "opt":    total_opt_bytes / num_gpus,
            "note":   "Opt states sharded across GPUs",
        },
        "ZeRO-2 (shard grads + opt)": {
            "params": total_param_bytes,
            "grads":  total_grad_bytes / num_gpus,
            "opt":    total_opt_bytes  / num_gpus,
            "note":   "Grads + opt states sharded",
        },
        "ZeRO-3 (shard everything)": {
            "params": total_param_bytes / num_gpus,
            "grads":  total_grad_bytes  / num_gpus,
            "opt":    total_opt_bytes   / num_gpus,
            "note":   "All tensors sharded; params gathered on-demand",
        },
    }

    h80_memory = 80e9  # H100 80 GB HBM
    print(f"{'Stage':<35} {'Params':>8} {'Grads':>8} {'Opt':>8} {'Total':>8}  {'Fits H100?':>10}")
    print("-" * 85)
    for stage, mem in stages.items():
        total = mem["params"] + mem["grads"] + mem["opt"]
        fits  = "✓" if total <= h80_memory else "✗"
        print(f"{stage:<35} "
              f"{bytes_to_gb(mem['params']):>8} "
              f"{bytes_to_gb(mem['grads']):>8} "
              f"{bytes_to_gb(mem['opt']):>8} "
              f"{bytes_to_gb(total):>8}  "
              f"{fits:>10}")

    print()
    naive  = total_param_bytes + total_grad_bytes + total_opt_bytes
    zero3  = (total_param_bytes + total_grad_bytes + total_opt_bytes) / num_gpus
    print(f"  ZeRO-3 memory reduction: {naive/zero3:.1f}× (≈ num_gpus = {num_gpus}×)")

print("=" * 85)
print("SCENARIO 1: 7B model on 8 GPUs  (single node, NVLink)")
print("=" * 85)
analyse_zero(model_params=7_000_000_000, num_gpus=8)

print("=" * 85)
print("SCENARIO 2: 70B model on 64 GPUs")
print("=" * 85)
analyse_zero(model_params=70_000_000_000, num_gpus=64)

print("=" * 85)
print("SCENARIO 3: 405B model on 512 GPUs  (LLaMA-3.1 scale)")
print("=" * 85)
analyse_zero(model_params=405_000_000_000, num_gpus=512)
""",
    },

    # ------------------------------------------------------------------
    # Step 6 — Training Dynamics & Loss Monitoring
    # ------------------------------------------------------------------
    "Step 6 · Training Dynamics & Loss Monitoring": {
        "description": (
            "Simulates a realistic pretraining loss curve including warm-up, "
            "stable training, and a mid-run loss spike with rollback recovery."
        ),
        "language": "python",
        "code": """
import math
import random

random.seed(7)

def simulate_pretraining(
    total_steps: int    = 1_000,
    warmup_steps: int   = 50,
    lr_max: float       = 3e-4,
    lr_min: float       = 3e-5,
    spike_step: int     = 420,   # inject a training instability here
    noise_scale: float  = 0.003,
) -> list[dict]:
    \"\"\"
    Simulate a pretraining loss curve with:
    - Cosine LR schedule + warm-up
    - Gaussian noise on gradients (realistic stochasticity)
    - An injected loss spike at `spike_step`
    - Automatic rollback to the last checkpoint before the spike
    \"\"\"
    records: list[dict] = []
    checkpoints: list[tuple[int, float]] = []   # (step, loss)
    CKPT_INTERVAL = 50

    # True underlying loss trajectory (exponential decay approximation)
    def true_loss(step: int) -> float:
        # Loss starts at ~10 (random init perplexity) and decays toward ~2.5
        progress = step / total_steps
        return 2.5 + 7.5 * math.exp(-4.5 * progress)

    loss = true_loss(0)
    spike_active  = False
    rollback_step = None

    for step in range(total_steps + 1):

        # --- LR schedule ------------------------------------------------
        if step < warmup_steps:
            lr = lr_max * (step + 1) / max(warmup_steps, 1)
        else:
            t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))

        # --- Simulate loss update ----------------------------------------
        target = true_loss(step)
        noise  = random.gauss(0, noise_scale)

        if step == spike_step:
            loss = loss * 3.8          # injected spike (bad data batch / LR too high)
            spike_active = True
        elif spike_active:
            # Training is still unstable post-spike
            loss = loss * 0.98 + noise * 0.5
        else:
            # Normal convergence toward true_loss
            loss = 0.85 * loss + 0.15 * target + noise

        # --- Checkpoint --------------------------------------------------
        if step % CKPT_INTERVAL == 0 and not spike_active:
            checkpoints.append((step, loss))

        # --- Spike detection & rollback ----------------------------------
        event = None
        if spike_active and len(checkpoints) >= 1:
            rollback_step, rollback_loss = checkpoints[-1]
            loss = rollback_loss          # restore weights
            spike_active  = False
            rollback_step = step
            event = f"ROLLBACK to step {rollback_step}"

        records.append({
            "step":     step,
            "loss":     loss,
            "lr":       lr,
            "ckpt":     step % CKPT_INTERVAL == 0,
            "event":    event,
        })

    return records


records = simulate_pretraining()

print("=" * 65)
print("PRETRAINING LOSS CURVE — KEY CHECKPOINTS")
print("=" * 65)
print(f"{'Step':>6}  {'Loss':>7}  {'LR':>9}  {'PPL':>8}  Event")
print("-" * 65)

display_steps = set(range(0, 1001, 100)) | {420, 421}
for r in records:
    if r["step"] in display_steps or r["event"]:
        ppl = math.exp(min(r["loss"], 20))   # cap for display
        event_str = r["event"] or ("★ CKPT" if r["ckpt"] else "")
        flag = "  ⚠ SPIKE" if r["step"] == 420 else ""
        print(f"{r['step']:>6}  {r['loss']:>7.4f}  {r['lr']:>9.2e}  {ppl:>8.2f}  {event_str}{flag}")

print()
print("=" * 65)
print("KEY METRICS")
print("=" * 65)
first_loss = records[0]["loss"]
last_loss  = records[-1]["loss"]
last_ppl   = math.exp(min(last_loss, 20))
print(f"  Initial loss       : {first_loss:.2f}  (perplexity ≈ {math.exp(first_loss):.0f})")
print(f"  Final loss         : {last_loss:.4f}  (perplexity ≈ {last_ppl:.1f})")
print(f"  Loss reduction     : {(1 - last_loss/first_loss)*100:.1f}%")
print()
print("  Training instability at step 420:")
print("    → loss spiked 3.8× above running value")
print("    → rolled back to checkpoint at step 400")
print("    → training resumed normally from rollback point")
print()
print("Best practice reminders:")
print("  • Save checkpoints every 50–200 steps during large runs")
print("  • Monitor gradient norm alongside loss (norm spike often precedes loss spike)")
print("  • If spike persists after rollback, reduce LR by 2–5×")
""",
    },

    # ------------------------------------------------------------------
    # Step 7 — Transformer Forward Pass from Scratch
    # ------------------------------------------------------------------
    "Step 7 · Transformer Forward Pass from Scratch": {
        "description": (
            "Implements a complete single-layer transformer forward pass in pure NumPy: "
            "token embedding, positional encoding, multi-head causal self-attention, "
            "MLP with GELU, layer norm, residual connections, and unembedding to logits."
        ),
        "language": "python",
        "code": """
import numpy as np

np.random.seed(0)

# ── Hyper-parameters (tiny model for illustration) ──────────────────────────
V       = 16     # vocab size
T       = 6      # sequence length
d_model = 32     # model dimension
d_k     = 8      # key/query dim per head
H       = 4      # number of attention heads  (H × d_k = d_model)
d_ff    = 64     # MLP hidden dim (4 × d_model)

assert H * d_k == d_model, "H * d_k must equal d_model"

# ── Random weight initialisation (mimicking GPT-2 scheme) ───────────────────
scale = 0.02
W_E  = np.random.randn(V, d_model)  * scale          # token embedding
W_PE = np.random.randn(T, d_model)  * scale          # positional embedding

# Attention projections (stacked for all heads)
W_Q  = np.random.randn(d_model, d_model) * scale
W_K  = np.random.randn(d_model, d_model) * scale
W_V  = np.random.randn(d_model, d_model) * scale
W_O  = np.random.randn(d_model, d_model) * scale / np.sqrt(2)   # GPT-2 output scale

# MLP weights
W_1  = np.random.randn(d_model, d_ff)   * scale
b_1  = np.zeros(d_ff)
W_2  = np.random.randn(d_ff, d_model)   * scale / np.sqrt(2)
b_2  = np.zeros(d_model)

# LayerNorm parameters (two sets: pre-attn and pre-mlp)
gamma_1 = np.ones(d_model);  beta_1 = np.zeros(d_model)
gamma_2 = np.ones(d_model);  beta_2 = np.zeros(d_model)

# Unembedding (tied weights = W_E.T in real models)
W_U  = W_E.T                                                    # (d_model, V)

# ── Helper functions ─────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    \"\"\"Normalise over d_model dimension for each token independently.\"\"\"
    mu  = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def gelu(x):
    \"\"\"Gaussian Error Linear Unit — smooth approximation.\"\"\"
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# ── Token + positional embedding ─────────────────────────────────────────────
tokens = np.array([3, 1, 4, 1, 5, 9])   # our input sequence
x = W_E[tokens] + W_PE[:T]              # (T, d_model)

print("=" * 60)
print("FORWARD PASS — shape trace")
print("=" * 60)
print(f"  Tokens              : {tokens}")
print(f"  After embedding     : {x.shape}")

# ── Pre-norm + Multi-Head Causal Self-Attention ───────────────────────────────
x_norm = layer_norm(x, gamma_1, beta_1)   # (T, d_model)

Q = x_norm @ W_Q   # (T, d_model)
K = x_norm @ W_K
V_mat = x_norm @ W_V

# Reshape into heads: (T, H, d_k) → (H, T, d_k)
Q_h = Q.reshape(T, H, d_k).transpose(1, 0, 2)       # (H, T, d_k)
K_h = K.reshape(T, H, d_k).transpose(1, 0, 2)
V_h = V_mat.reshape(T, H, d_k).transpose(1, 0, 2)

# Scaled dot-product attention with causal mask
scale_factor = np.sqrt(d_k)
attn_scores = Q_h @ K_h.transpose(0, 2, 1) / scale_factor   # (H, T, T)

# Causal mask: upper triangle → -inf → softmax → 0
mask = np.triu(np.ones((T, T), dtype=bool), k=1)
attn_scores[:, mask] = -np.inf

attn_weights = softmax(attn_scores, axis=-1)                 # (H, T, T)
head_outputs = attn_weights @ V_h                            # (H, T, d_k)

# Concat heads and project
concat = head_outputs.transpose(1, 0, 2).reshape(T, d_model) # (T, d_model)
x_attn = concat @ W_O                                        # (T, d_model)

# Residual add (first sublayer)
x = x + x_attn
print(f"  After attn + resid  : {x.shape}")

# Inspect attention patterns for head 0
print()
print("  Attention weights — Head 0 (rows=query positions, cols=key positions):")
print("  (upper-right = 0 due to causal mask)")
np.set_printoptions(precision=2, suppress=True, linewidth=120)
print("  " + str(attn_weights[0].round(2)))

# ── Pre-norm + MLP (FFN) with GELU ──────────────────────────────────────────
x_norm2 = layer_norm(x, gamma_2, beta_2)                     # (T, d_model)
h        = gelu(x_norm2 @ W_1 + b_1)                         # (T, d_ff)
x_mlp    = h @ W_2 + b_2                                     # (T, d_model)

# Residual add (second sublayer)
x = x + x_mlp
print(f"\\n  After MLP  + resid  : {x.shape}")

# ── Final LayerNorm + Unembedding ────────────────────────────────────────────
x_final = layer_norm(x, gamma_1, beta_1)                     # (T, d_model)
logits   = x_final @ W_U                                     # (T, V)

print(f"  Logits (T × V)      : {logits.shape}")
print()

# Show predictions at each position
print("=" * 60)
print("PER-POSITION PREDICTIONS")
print("=" * 60)
probs = softmax(logits)
targets = np.append(tokens[1:], [-1])   # next token at each position
print(f"  {'pos':>3}  {'input':>6}  {'target':>7}  {'pred':>5}  {'P(target)':>10}  {'entropy':>8}")
print(f"  {'-'*3}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*10}  {'-'*8}")
for i in range(T - 1):
    pred_tok = int(probs[i].argmax())
    p_target = probs[i, targets[i]]
    H_ent    = -float((probs[i] * np.log(probs[i] + 1e-9)).sum())
    print(f"  {i:>3}  {tokens[i]:>6}  {targets[i]:>7}  {pred_tok:>5}  {p_target:>10.4f}  {H_ent:>8.4f}")

loss = -np.log(probs[np.arange(T-1), targets[:T-1]] + 1e-9).mean()
print(f"\\n  Cross-entropy loss = {loss:.4f}  |  Perplexity = {np.exp(loss):.2f}")
print(f"  (untrained model; random baseline perplexity = {V:.0f})")
print()
print("Key observations:")
print("  1. Residual connections pass shape (T, d_model) cleanly through each sublayer")
print("  2. Causal mask zeros out upper-right attention weights (no future peeking)")
print("  3. High entropy ≈ uniform predictions — model has learned nothing yet")
print("  4. Backward pass will compute dL/d(every weight above) via chain rule")
""",
    },

    # ------------------------------------------------------------------
    # Step 8 — Backpropagation Through a Transformer Layer
    # ------------------------------------------------------------------
    "Step 8 · Backpropagation & Gradient Flow": {
        "description": (
            "Manually derives and numerically verifies the exact gradients flowing "
            "through a transformer residual sublayer — LayerNorm, attention output "
            "projection, and the residual bypass — then shows how much gradient "
            "each path carries."
        ),
        "language": "python",
        "code": """
import numpy as np

np.random.seed(42)

# ── Architecture: single residual block ──────────────────────────────────────
#   x_in → LN → W_proj → x_sub   (sublayer output)
#   x2   = x_in + x_sub           (residual add)
#   x3   = x2 @ W_U               (unembedding)
#   loss = cross_entropy(x3, tgt)
#
# This is the simplest residual unit that captures the key gradient behaviours.

d  = 8    # d_model
T  = 4    # sequence length
V  = 10   # vocab size

W_proj = np.random.randn(d, d) * 0.1   # attention / MLP output projection
W_U    = np.random.randn(d, V) * 0.1   # unembedding
gamma  = np.ones(d)
beta   = np.zeros(d)
x_in   = np.random.randn(T, d) * 0.3
targets = np.array([2, 7, 0, 5])

def layer_norm_fwd(x, g, b, eps=1e-5):
    mu    = x.mean(-1, keepdims=True)
    var   = x.var(-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return g * x_hat + b, x_hat, var

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def layer_norm_bwd(dL_dy, x_hat, var, g, eps=1e-5):
    \"\"\"Exact backward through LayerNorm (Ba et al. 2016).\"\"\"
    dL_dxhat  = dL_dy * g
    inv_std   = 1.0 / np.sqrt(var + eps)
    dL_dx     = inv_std * (
        dL_dxhat
        - dL_dxhat.mean(-1, keepdims=True)
        - x_hat * (dL_dxhat * x_hat).mean(-1, keepdims=True)
    )
    dL_dg     = (dL_dy * x_hat).sum(axis=0)
    dL_db     = dL_dy.sum(axis=0)
    return dL_dx, dL_dg, dL_db

# ── Forward pass ─────────────────────────────────────────────────────────────
ln_out, x_hat, var = layer_norm_fwd(x_in, gamma, beta)
x_sub = ln_out @ W_proj          # sublayer transform
x2    = x_in + x_sub             # residual connection
logits = x2 @ W_U                # (T, V)

e = np.exp(logits - logits.max(-1, keepdims=True))
probs = e / e.sum(-1, keepdims=True)
loss = -np.log(probs[np.arange(T), targets] + 1e-9).mean()

print("=" * 60)
print("GRADIENT DERIVATION — STEP BY STEP")
print("=" * 60)
print(f"  Forward loss = {loss:.6f}")
print()

# ── Analytical backward ───────────────────────────────────────────────────────
# Step 1: dL / d logits
dL_dlogits = probs.copy()
dL_dlogits[np.arange(T), targets] -= 1
dL_dlogits /= T

# Step 2: dL / d x2  and  dL / d W_U
dL_dx2 = dL_dlogits @ W_U.T           # (T, d)
dL_dWU = x2.T @ dL_dlogits           # (d, V)

# Step 3: Residual split — gradient fans into BOTH paths equally
#   x2 = x_in + x_sub
#   dL/dx_in = dL/dx2 · 1   (highway — the identity path)
#   dL/dx_sub= dL/dx2 · 1   (sublayer path)
dL_dx_in_residual = dL_dx2.copy()     # the "highway" portion
dL_dxsub          = dL_dx2.copy()     # the "learned sublayer" portion

# Step 4: dL / d W_proj  and  dL / d ln_out
dL_dWproj  = ln_out.T @ dL_dxsub     # (d, d)
dL_dlnout  = dL_dxsub @ W_proj.T    # (T, d)

# Step 5: dL / d x_in through LN
dL_dx_in_ln, dL_dgamma, dL_dbeta = layer_norm_bwd(dL_dlnout, x_hat, var, gamma)

# Step 6: total dL / d x_in  (sum of both paths!)
dL_dxin_total = dL_dx_in_residual + dL_dx_in_ln

# ── Numerical gradient check ──────────────────────────────────────────────────
eps_fd = 1e-5

def full_forward(wp, wu, xi):
    ln, xh, v = layer_norm_fwd(xi, gamma, beta)
    x_s = ln @ wp
    x_2 = xi + x_s
    lgt = x_2 @ wu
    e   = np.exp(lgt - lgt.max(-1, keepdims=True))
    pr  = e / e.sum(-1, keepdims=True)
    return -np.log(pr[np.arange(T), targets] + 1e-9).mean()

def fd_grad_ij(param, name, i=0, j=0):
    wp = W_proj.copy(); wu = W_U.copy(); xi = x_in.copy()
    if name == 'W_proj': wp[i,j] += eps_fd
    elif name == 'W_U':  wu[i,j] += eps_fd
    lp = full_forward(wp, wu, xi)
    if name == 'W_proj': wp[i,j] -= 2*eps_fd
    elif name == 'W_U':  wu[i,j] -= 2*eps_fd
    lm = full_forward(wp, wu, xi)
    return (lp - lm) / (2 * eps_fd)

print("GRADIENT VERIFICATION vs central finite differences")
print(f"  {'Parameter':<12} {'Numerical':>12} {'Analytical':>12} {'Rel error':>12}  Status")
print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}")
for name, analytical in [('W_proj', dL_dWproj), ('W_U', dL_dWU)]:
    num = fd_grad_ij(None, name, i=0, j=0)
    ana = analytical[0, 0]
    err = abs(num - ana) / (abs(ana) + 1e-10)
    ok  = "✓ PASS" if err < 1e-4 else "✗ FAIL"
    print(f"  {name:<12} {num:>12.8f} {ana:>12.8f} {err:>12.2e}  {ok}")

print()

# ── Gradient flow analysis ────────────────────────────────────────────────────
highway_norm  = np.linalg.norm(dL_dx_in_residual)    # identity path
sublayer_norm = np.linalg.norm(dL_dx_in_ln)          # path through LN + W_proj
total_norm    = np.linalg.norm(dL_dxin_total)

print("=" * 60)
print("GRADIENT FLOW DECOMPOSITION")
print("=" * 60)
print(f"  Total  ‖dL/dx_in‖     : {total_norm:.6f}")
print(f"  Highway path (resid)  : {highway_norm:.6f}  ({highway_norm/total_norm*100:.1f}%)")
print(f"  Sublayer path (LN+W)  : {sublayer_norm:.6f}  ({sublayer_norm/total_norm*100:.1f}%)")
print()

print("  Why residuals prevent vanishing gradients:")
print("  ─────────────────────────────────────────")
print("  Plain network (no residual): gradient ∝ ∏ᵢ Wᵢ   → can vanish or explode")
print("  Residual network:           gradient ∝ 1 + ∑ᵢ f'ᵢ → always at least 1")
print()
print("  At initialisation (sublayer outputs ≈ 0), the gradient is ENTIRELY")
print("  carried by the highway. As training progresses and the sublayer")
print("  learns useful transforms, its gradient contribution grows.")
print()

# Show how residual gradient norm stays bounded at many layers
print("  Simulated gradient norm through N stacked residual layers:")
print(f"  {'Layers':>8}  {'Plain net':>12}  {'Residual net':>14}")
print(f"  {'-'*8}  {'-'*12}  {'-'*14}")
W_norm = 0.95   # typical sub-1 weight norm per layer
for n in [1, 5, 10, 25, 50, 100]:
    plain    = W_norm**n
    residual = (1 + W_norm)**n / 2**n   # rough bound: shrinks much slower
    # More accurate: product of (1 + small_sublayer_contribution)
    resid_actual = 1.0   # lower bound — the identity path always carries 1×
    print(f"  {n:>8}  {plain:>12.6f}  {resid_actual:>14.6f} (lower bound)")
print()
print("  The residual lower bound of 1.0 means gradients NEVER vanish,")
print("  regardless of depth — this is the key insight behind ResNets and Transformers.")
""",
    },

    # ------------------------------------------------------------------
    # Step 9 — Gradient Accumulation
    # ------------------------------------------------------------------
    "Step 9 · Gradient Accumulation & Effective Batch Size": {
        "description": (
            "Proves that gradient accumulation over K micro-batches is mathematically "
            "equivalent to a single forward/backward on the full concatenated batch, "
            "and demonstrates how to compute the correct effective batch size for "
            "large-scale training."
        ),
        "language": "python",
        "code": """
import numpy as np

np.random.seed(99)

# ── Tiny 2-layer linear model  (input → hidden → logits) ────────────────────
d_in, d_hid, d_out = 8, 16, 4

W1 = np.random.randn(d_in, d_hid) * 0.1
W2 = np.random.randn(d_hid, d_out) * 0.1

def relu(x): return np.maximum(0, x)

def forward_and_loss(X, y):
    \"\"\"Returns loss and (dW1, dW2) analytically.\"\"\"
    h     = relu(X @ W1)                    # (B, d_hid)
    logits = h @ W2                          # (B, d_out)

    # Softmax cross-entropy
    e     = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    loss  = -np.log(probs[np.arange(len(y)), y] + 1e-9).mean()

    # Backward
    dlogits                 = probs.copy()
    dlogits[np.arange(len(y)), y] -= 1
    dlogits                /= len(y)

    dW2  = h.T @ dlogits                    # (d_hid, d_out)
    dh   = dlogits @ W2.T                   # (B, d_hid)
    dh  *= (h > 0).astype(float)            # ReLU mask
    dW1  = X.T @ dh                         # (d_in, d_hid)
    return loss, dW1, dW2

# ── Create a "full" batch by combining K micro-batches ───────────────────────
B_micro = 6      # samples per micro-batch
K       = 4      # number of accumulation steps
B_full  = B_micro * K

X_full = np.random.randn(B_full, d_in)
y_full = np.random.randint(0, d_out, size=B_full)

# ── Method A: single forward on full batch ───────────────────────────────────
loss_full, dW1_full, dW2_full = forward_and_loss(X_full, y_full)

# ── Method B: gradient accumulation over K micro-batches ─────────────────────
dW1_accum = np.zeros_like(W1)
dW2_accum = np.zeros_like(W2)
total_loss = 0.0

for k in range(K):
    X_micro = X_full[k * B_micro : (k+1) * B_micro]
    y_micro = y_full[k * B_micro : (k+1) * B_micro]

    loss_k, dW1_k, dW2_k = forward_and_loss(X_micro, y_micro)

    # CRITICAL: scale gradient contribution by B_micro / B_full
    # (forward_and_loss already divides by B_micro, we need division by B_full)
    scale = B_micro / B_full
    dW1_accum += dW1_k * scale
    dW2_accum += dW2_k * scale
    total_loss += loss_k * scale

# ── Compare ──────────────────────────────────────────────────────────────────
err_W1 = np.linalg.norm(dW1_full - dW1_accum)
err_W2 = np.linalg.norm(dW2_full - dW2_accum)

print("=" * 65)
print("GRADIENT ACCUMULATION EQUIVALENCE CHECK")
print("=" * 65)
print(f"  Full batch size    : {B_full}  (single pass)")
print(f"  Micro-batch size   : {B_micro}  ×  {K} steps")
print()
print(f"  Full batch loss    : {loss_full:.8f}")
print(f"  Accumulated loss   : {total_loss:.8f}")
print(f"  Loss difference    : {abs(loss_full - total_loss):.2e}  ← numerical noise only")
print()
print(f"  ‖dW1_full − dW1_accum‖ = {err_W1:.2e}   {'✓ equivalent' if err_W1 < 1e-10 else '✗ BUG'}")
print(f"  ‖dW2_full − dW2_accum‖ = {err_W2:.2e}   {'✓ equivalent' if err_W2 < 1e-10 else '✗ BUG'}")

print()
print("=" * 65)
print("COMMON BUG: forgetting to scale the loss")
print("=" * 65)

# Buggy version: just accumulates without scaling (as if loss.backward() called K times)
dW1_buggy = np.zeros_like(W1)
for k in range(K):
    X_micro = X_full[k * B_micro : (k+1) * B_micro]
    y_micro = y_full[k * B_micro : (k+1) * B_micro]
    _, dW1_k, _ = forward_and_loss(X_micro, y_micro)
    dW1_buggy += dW1_k    # NO scaling!

inflation = np.linalg.norm(dW1_buggy) / np.linalg.norm(dW1_full)
print(f"  Unscaled accumulation inflates gradient norm by {inflation:.2f}×")
print(f"  This is equivalent to multiplying the learning rate by {inflation:.2f}×")
print(f"  → Silent training instability / loss spike / divergence")

print()
print("=" * 65)
print("EFFECTIVE BATCH SIZE CALCULATOR")
print("=" * 65)
print(f"  {'GPUs':>6}  {'micro-B':>8}  {'accum':>6}  {'B_eff':>8}  {'tokens/step (T=4096)':>22}")
print(f"  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*22}")
configs = [
    (1,    4,  1),
    (1,    4,  8),
    (8,    4,  8),
    (64,   4,  8),
    (512,  2,  4),
    (1024, 2,  4),
]
for gpus, mb, acc in configs:
    beff = gpus * mb * acc
    tok  = beff * 4096
    print(f"  {gpus:>6}  {mb:>8}  {acc:>6}  {beff:>8}  {tok/1e6:>20.2f}M")

print()
print("  Most large LLM runs target 1–4M tokens per gradient step.")
print("  Below ~256K tokens/step, gradient noise degrades convergence.")
print("  Above ~8M tokens/step, diminishing returns and longer wall time.")
""",
    },

    # ------------------------------------------------------------------
    # Step 10 — Full End-to-End Training Loop
    # ------------------------------------------------------------------
    "Step 10 · Full End-to-End Mini Pretraining Loop": {
        "description": (
            "Ties everything together: a complete pretraining loop with a tiny "
            "transformer trained on character-level text. Covers data streaming, "
            "gradient accumulation, AdamW with cosine LR, gradient clipping, "
            "checkpointing, and per-step logging — all in pure Python + NumPy."
        ),
        "language": "python",
        "code": """
import numpy as np
import math
import copy

np.random.seed(2024)

# ══════════════════════════════════════════════════════════════════════════════
# 0. DATA — tiny character-level corpus
# ══════════════════════════════════════════════════════════════════════════════
TEXT = (
    "the quick brown fox jumps over the lazy dog "
    "pack my box with five dozen liquor jugs "
    "how vexingly quick daft zebras jump "
    "the five boxing wizards jump quickly "
    "sphinx of black quartz judge my vow "
) * 8   # repeat to get more tokens

chars  = sorted(set(TEXT))
V      = len(chars)
stoi   = {c: i for i, c in enumerate(chars)}
itos   = {i: c for c, i in stoi.items()}
tokens = np.array([stoi[c] for c in TEXT], dtype=np.int32)

print(f"Vocab size : {V} characters")
print(f"Corpus     : {len(tokens)} tokens")
print(f"Characters : {''.join(chars)}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL — single-layer transformer, pure NumPy
# ══════════════════════════════════════════════════════════════════════════════
T       = 16     # context window
d_model = 32
d_k     = 8
H       = 4      # heads
d_ff    = 64

def init_weights():
    s = 0.02
    return {
        'WE':  np.random.randn(V, d_model)  * s,
        'WPE': np.random.randn(T, d_model)  * s,
        'WQ':  np.random.randn(d_model, d_model) * s,
        'WK':  np.random.randn(d_model, d_model) * s,
        'WV':  np.random.randn(d_model, d_model) * s,
        'WO':  np.random.randn(d_model, d_model) * s / math.sqrt(2),
        'W1':  np.random.randn(d_model, d_ff)    * s,
        'b1':  np.zeros(d_ff),
        'W2':  np.random.randn(d_ff, d_model)    * s / math.sqrt(2),
        'b2':  np.zeros(d_model),
        'g1':  np.ones(d_model),  'b_1': np.zeros(d_model),
        'g2':  np.ones(d_model),  'b_2': np.zeros(d_model),
        'WU':  None,   # tied to WE, set after init
    }

params = init_weights()
params['WU'] = params['WE']   # tied embeddings

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

def softmax(x, ax=-1):
    e = np.exp(x - x.max(ax, keepdims=True))
    return e / e.sum(ax, keepdims=True)

CAUSAL = np.triu(np.ones((T, T), dtype=bool), k=1)

def forward(tok_ids, p):
    \"\"\"Forward pass. Returns logits (T, V) and loss (scalar) given token IDs (T+1,).\"\"\"
    x_in = tok_ids[:T]                               # inputs
    tgt  = tok_ids[1:T+1]                            # targets (shifted by 1)

    x = p['WE'][x_in] + p['WPE']                    # (T, d_model)

    # — Attention sublayer —
    xn = layer_norm(x, p['g1'], p['b_1'])
    Q  = xn @ p['WQ']; K = xn @ p['WK']; V = xn @ p['WV']
    Qh = Q.reshape(T, H, d_k).transpose(1,0,2)      # (H, T, d_k)
    Kh = K.reshape(T, H, d_k).transpose(1,0,2)
    Vh = V.reshape(T, H, d_k).transpose(1,0,2)
    sc = Qh @ Kh.transpose(0,2,1) / math.sqrt(d_k)  # (H, T, T)
    sc[:, CAUSAL] = -1e9
    aw = softmax(sc, ax=-1)
    oc = (aw @ Vh).transpose(1,0,2).reshape(T, d_model)
    x  = x + oc @ p['WO']

    # — MLP sublayer —
    xn2 = layer_norm(x, p['g2'], p['b_2'])
    x   = x + gelu(xn2 @ p['W1'] + p['b1']) @ p['W2'] + p['b2']

    # — Logits & loss —
    logits = x @ p['WE'].T                           # tied unembedding (T, V)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    pr = e / e.sum(-1, keepdims=True)
    loss = -np.log(pr[np.arange(T), tgt] + 1e-9).mean()
    return logits, loss, pr, tgt

# ══════════════════════════════════════════════════════════════════════════════
# 2. OPTIMISER — AdamW
# ══════════════════════════════════════════════════════════════════════════════
PARAM_KEYS  = ['WE','WQ','WK','WV','WO','W1','W2']  # weight decay applies here
NO_DECAY    = ['WPE','b1','b2','g1','g2','b_1','b_2']

def init_adamw_state(p):
    return {k: {'m': np.zeros_like(v), 'v': np.zeros_like(v)}
            for k, v in p.items() if v is not None and k != 'WU'}

def adamw_step(p, grads, state, step, lr, beta1=0.9, beta2=0.95,
               eps=1e-8, wd=0.1):
    bc1 = 1 - beta1**step
    bc2 = 1 - beta2**step
    for k in grads:
        if k == 'WU' or p.get(k) is None: continue
        g  = grads[k]
        st = state[k]
        st['m'] = beta1 * st['m'] + (1 - beta1) * g
        st['v'] = beta2 * st['v'] + (1 - beta2) * g**2
        mh = st['m'] / bc1
        vh = st['v'] / bc2
        decay = wd if k in PARAM_KEYS else 0.0
        p[k]  = p[k] * (1 - lr * decay) - lr * mh / (np.sqrt(vh) + eps)
    return p

# ══════════════════════════════════════════════════════════════════════════════
# 3. NUMERICAL GRADIENT — used instead of full backprop for clarity
# ══════════════════════════════════════════════════════════════════════════════
EPS_FD = 1e-4

def numerical_grads(p, tok_ids, keys, n_samples=4):
    \"\"\"Estimate gradients via central finite differences for first n_samples entries.\"\"\"
    grads = {k: np.zeros_like(p[k]) for k in keys if p[k] is not None}
    for k in keys:
        if p[k] is None: continue
        flat  = p[k].ravel()
        indices = np.random.choice(len(flat), size=min(n_samples, len(flat)), replace=False)
        for idx in indices:
            orig = flat[idx]
            flat[idx] = orig + EPS_FD
            _, lp, _, _ = forward(tok_ids, p)
            flat[idx] = orig - EPS_FD
            _, lm, _, _ = forward(tok_ids, p)
            flat[idx] = orig
            grads[k].ravel()[idx] = (lp - lm) / (2 * EPS_FD)
    return grads

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_STEPS    = 120
WARMUP_STEPS   = 12
LR_MAX         = 3e-3
LR_MIN         = 3e-4
ACCUM_STEPS    = 3       # gradient accumulation
CKPT_INTERVAL  = 40
CLIP_THRESHOLD = 1.0

opt_state  = init_adamw_state(params)
checkpoints = {}
log_records = []

def lr_schedule(step):
    if step < WARMUP_STEPS:
        return LR_MAX * (step + 1) / WARMUP_STEPS
    t = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))

def get_batch(offset):
    start = (offset * (T + 1)) % (len(tokens) - T - 1)
    return tokens[start : start + T + 1]

print("=" * 65)
print("TRAINING LOOP — character-level LM on pangrams")
print("=" * 65)
print(f"  {'Step':>5}  {'Loss':>7}  {'PPL':>7}  {'LR':>9}  {'GradNorm':>9}  Event")
print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*15}")

for step in range(1, TOTAL_STEPS + 1):
    lr = lr_schedule(step - 1)

    # — Gradient accumulation loop —
    accum_grads = {k: np.zeros_like(v) for k, v in params.items()
                   if v is not None and k != 'WU'}
    step_loss = 0.0

    for acc in range(ACCUM_STEPS):
        tok_ids = get_batch(offset=(step - 1) * ACCUM_STEPS + acc)
        grad_keys = ['WE', 'WO', 'W1', 'W2']
        grads_k   = numerical_grads(params, tok_ids, grad_keys, n_samples=6)
        _, loss_k, _, _ = forward(tok_ids, params)

        scale = 1.0 / ACCUM_STEPS
        for k in accum_grads:
            if k in grads_k:
                accum_grads[k] += grads_k[k] * scale
        step_loss += loss_k * scale

    # — Gradient norm & clipping —
    raw_norm = math.sqrt(sum(np.sum(g**2) for g in accum_grads.values()))
    if raw_norm > CLIP_THRESHOLD:
        clip_scale = CLIP_THRESHOLD / raw_norm
        accum_grads = {k: v * clip_scale for k, v in accum_grads.items()}
    clipped_norm = min(raw_norm, CLIP_THRESHOLD)

    # — Optimiser step —
    params = adamw_step(params, accum_grads, opt_state, step, lr)
    params['WU'] = params['WE']   # keep tied weights in sync

    # — Logging —
    ppl   = math.exp(min(step_loss, 20))
    event = ""

    if step % CKPT_INTERVAL == 0:
        checkpoints[step] = copy.deepcopy(params)
        event = "★ checkpoint saved"

    log_records.append(dict(step=step, loss=step_loss, ppl=ppl,
                            lr=lr, grad_norm=raw_norm))

    if step == 1 or step % 20 == 0 or step == TOTAL_STEPS:
        print(f"  {step:>5}  {step_loss:>7.4f}  {ppl:>7.2f}  {lr:>9.2e}  "
              f"{raw_norm:>9.4f}  {event}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION — sample from trained model
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("GENERATION — greedy decoding from trained model")
print("=" * 65)

def generate(params, seed_text, max_new=30):
    ctx = [stoi.get(c, 0) for c in seed_text]
    # Pad context to exactly T tokens (repeat seed if too short)
    while len(ctx) < T:
        ctx = ctx + ctx
    ctx = ctx[-T:]   # take the last T tokens
    out = list(seed_text)
    for _ in range(max_new):
        tok_ids_gen = np.array(ctx + [0])   # (T+1,) with dummy target
        logits, _, pr, _ = forward(np.array(tok_ids_gen), params)
        # Greedy: pick highest-prob token at last position
        next_tok = int(logits[-1].argmax())
        out.append(itos[next_tok])
        ctx = ctx[1:] + [next_tok]
    return ''.join(out)

seeds = ["the ", "fox ", "jump"]
for seed in seeds:
    generated = generate(params, seed, max_new=25)
    print('  Seed: ' + repr(seed) + ' -> ' + repr(generated))
print()
print("=" * 65)
print("TRAINING SUMMARY")
print("=" * 65)
first = log_records[0]
last  = log_records[-1]
best  = min(log_records, key=lambda r: r['loss'])
print(f"  Initial  loss / PPL : {first['loss']:.4f}  /  {first['ppl']:.2f}")
print(f"  Final    loss / PPL : {last['loss']:.4f}  /  {last['ppl']:.2f}")
print(f"  Best     loss / PPL : {best['loss']:.4f}  /  {best['ppl']:.2f}  (step {best['step']})")
print(f"  Checkpoints saved   : {list(checkpoints.keys())}")
print()
print("  Step-by-step what happened:")
print("  1. Token IDs fetched from corpus via sliding window")
print("  2. Forward pass: embed → attn (causal mask) → MLP → logits")
print("  3. Loss = cross-entropy(logits[:-1], tokens[1:])")
print("  4. Gradients estimated via finite differences (proxy for autograd)")
print("  5. Gradients accumulated over 3 micro-batches then scaled")
print("  6. Global gradient norm clipped to 1.0")
print("  7. AdamW updated W_E, W_O, W_1, W_2 with cosine LR + weight decay")
print("  8. Tied unembedding (WU = WE) kept in sync after each step")
print("  9. Checkpoint saved every 40 steps")
print(" 10. Generation uses greedy decoding over the trained logits")
""",
    },

}


# ---------------------------------------------------------------------------

def get_topic_data():
    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "operations": OPERATIONS,
    }