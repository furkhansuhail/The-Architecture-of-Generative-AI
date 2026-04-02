"""
LLM Training on a Single GPU — From Raw Text to a Trained Language Model
=========================================================================

Training a Large Language Model from scratch on a single GPU is the
foundational case that every practitioner must understand before moving to
distributed systems. Everything that happens across 1,000 GPUs is a
parallelised version of what happens on one.

This module walks through the complete single-GPU LLM training pipeline:
how raw internet text becomes tokenised training batches, how the transformer
forward pass works, how gradients flow backward, how the optimizer updates
weights, and how every memory and compute constraint on a single GPU forces
specific engineering decisions (gradient checkpointing, mixed precision,
gradient accumulation, learning rate schedules).

We treat a "single GPU" concretely — an NVIDIA A100 80GB — and work
through exactly what fits, what doesn't, and what tricks allow training
models that would otherwise exceed GPU memory. By the end, you will
understand why a 7-billion-parameter model fits on one A100 in BF16
but not in FP32, why gradient accumulation simulates a larger batch size,
and why the loss curve shape tells you whether training is progressing
correctly.

The architecture target is a GPT-style decoder-only transformer, which
is the architecture behind GPT-2, GPT-3, LLaMA, Mistral, Falcon, and
virtually every modern autoregressive language model.

"""

import textwrap
import re

TOPIC_NAME   = "LLM Training on a Single GPU"
DISPLAY_NAME = "03 · Single-GPU LLM Training"
ICON         = "🧠"
SUBTITLE     = "Tokenisation, Transformer Forward/Backward, Memory, and the Training Loop"


THEORY = """

##### PART 1 — WHAT AN LLM IS AND WHAT "TRAINING" MEANS

### The Core Objective: Next-Token Prediction

    A language model is trained to predict the next token given all previous
    tokens. Given a sequence [t_0, t_1, ..., t_{n-1}], the model outputs a
    probability distribution over the vocabulary for each position:

        P(t_i | t_0, t_1, ..., t_{i-1})  for all i in the sequence.

    This is the language modelling objective — also called causal language
    modelling or autoregressive language modelling.

    The training loss is the cross-entropy (negative log-likelihood) averaged
    over all token positions:

        L = -(1/T) × Σ_i log P(t_i | t_0, ..., t_{i-1})

    "Training" means: adjust the model's parameters (weights) to minimise L
    over the entire training dataset. After sufficient training, the model
    has learned to predict text so well that it has implicitly encoded
    grammar, facts, reasoning patterns, and language structure.

### What the Model Actually Learns

    The model NEVER sees explicit supervision labels like "this is a noun"
    or "this sentiment is positive". It only ever sees raw text sequences
    and tries to predict the next token. Yet from this signal alone, it
    learns:
        - Grammar and syntax (predicts grammatically correct continuations)
        - Facts (predicts "The capital of France is Paris" given context)
        - Reasoning (predicts correct logical conclusions)
        - Style and tone (predicts text matching the style of the context)

    This is the key insight of pre-training: raw next-token prediction on
    enough high-quality text is sufficient to learn a surprisingly general
    representation of language and knowledge.

### Architecture: Decoder-Only Transformer

    Modern LLMs use a DECODER-ONLY transformer (no encoder):

    Input tokens → Embedding lookup → N × TransformerBlock → LM Head → logits

    Each TransformerBlock contains:
        1. RMSNorm (or LayerNorm) before each sub-layer (pre-norm)
        2. Causal Multi-Head Self-Attention (MHSA)
        3. Residual connection: x = x + MHSA(RMSNorm(x))
        4. RMSNorm
        5. Feed-Forward Network (FFN, typically SwiGLU or GELU MLP)
        6. Residual connection: x = x + FFN(RMSNorm(x))

    LM Head: final linear projection from hidden_dim → vocab_size
    Followed by softmax to get probabilities.

    Why decoder-only (not encoder-decoder like the original Transformer)?
        Simpler architecture, easier to scale, and for language modelling
        (next-token prediction), you only need the causal (left-to-right)
        attention pattern. Encoder-decoder is needed for translation-style
        tasks with an explicit source and target, but not for generation.

### Parameter Count

    For a GPT-style model with:
        V = vocab_size (e.g., 32,000)
        d = hidden dimension (e.g., 4,096)
        h = number of attention heads (e.g., 32)
        d_kv = head dimension (d/h = 128)
        d_ff = FFN intermediate size (e.g., 11,008 for SwiGLU)
        N = number of layers (e.g., 32)

    Parameters per layer:
        Attention (Q, K, V, O):       4 × d × d            = 4 × d²
        FFN (SwiGLU: gate, up, down): 3 × d × d_ff
        RMSNorm (2 per block):        2 × d

    Total:
        Embedding:     V × d
        N layers:      N × (4d² + 3d×d_ff + 2d)
        LM head:       d × V  (often tied to embedding weights)

    For LLaMA-2 7B (V=32K, d=4096, d_ff=11008, N=32):
        Per layer: 4×4096² + 3×4096×11008 = 67M + 135M = 202M params
        32 layers: 32 × 202M = 6.46B params
        Embedding: 32K × 4096 = 131M params
        Total ≈ 6.7B ≈ 7B ✓


##### PART 2 — DATA PIPELINE: FROM RAW TEXT TO TRAINING BATCHES

### Step 1: Text Corpus Collection

    Pre-training data for modern LLMs:
        Common Crawl:     ~3TB of compressed raw web text (monthly snapshots)
        Books3 / BookCorpus: fiction and non-fiction books
        Wikipedia:         encyclopaedia articles (all languages)
        ArXiv:             scientific papers
        StackExchange:     Q&A technical content
        GitHub:            code (critical for code-capable models)
        C4 (Cleaned CC):   Colossal Clean Crawled Corpus

    Data quality matters more than quantity:
        Duplicate removal: many web pages appear in multiple crawl snapshots.
            MinHash + LSH deduplication removes near-duplicates.
        Language filtering: detect and optionally filter non-English text.
        Quality heuristics: remove pages with < 80% English, too many symbols,
            very short pages, boilerplate (cookie notices, nav bars).
        Toxic content filtering: classifier-based removal of harmful content.

    Typical pre-training scale:
        LLaMA-1 7B: 1 trillion tokens (1T)
        LLaMA-2 7B: 2 trillion tokens (2T)
        GPT-3:       300 billion tokens (300B)
        PaLM:        780 billion tokens (780B)

### Step 2: Tokenisation

    Tokenisation converts raw text into a sequence of integers (token IDs).

    BYTE PAIR ENCODING (BPE):
        Start: every byte (256 symbols) is a token.
        Repeatedly merge the most frequent adjacent pair of tokens.
        Stop when vocabulary reaches target size V.

        Example merges:
            ("t", "h") → "th"   (frequent in English)
            ("th", "e") → "the" (very frequent)
            ("ing", " ") → "ing " (common suffix)

        Common words become single tokens: "the", "and", "is", "are"
        Rare words split into sub-word pieces: "unbelievable" → "un", "believ", "able"
        Unknown words always encodable: fall back to byte-level tokens.

    SENTENCEPIECE (BPE variant):
        Treats entire input as a stream of Unicode characters.
        Used by: LLaMA, Mistral (sentencepiece with BPE).

    TIKTOKEN (OpenAI's BPE):
        Used by GPT-2, GPT-3, GPT-4, ChatGPT.
        cl100k_base: V=100,277 tokens.
        Handles code efficiently (identifiers as single tokens).

    Vocabulary sizes:
        GPT-2:    V=50,257
        LLaMA-2:  V=32,000
        GPT-4:    V=100,277
        Gemini:   V=256,000 (multilingual friendly)

### Step 3: Dataset Preparation

    After tokenisation, the entire corpus is concatenated into one long
    sequence of token IDs. Documents are separated by a special <EOS> token.

    The sequence is then cut into fixed-length CHUNKS of length context_length
    (also called max_seq_len or block_size):
        Typical values: 2048 (GPT-2), 4096 (LLaMA-2), 8192 (LLaMA-2-long)

    Each chunk becomes one TRAINING EXAMPLE:
        input_ids  = chunk[0:context_length-1]   (T tokens)
        labels     = chunk[1:context_length]      (T tokens, shifted by 1)

    The labels are the input_ids shifted right by 1 position:
        input_ids:  [t_0, t_1, t_2, ..., t_{T-1}]
        labels:     [t_1, t_2, t_3, ..., t_T    ]

    This is the "teacher forcing" setup: for each position i, the model
    sees [t_0,...,t_{i-1}] as input and must predict t_i as output.

    DataLoader setup:
        shuffle=True per epoch (or shuffle the corpus before chunking)
        batch_size=B (number of chunks per batch)
        Packed batching: no padding — all sequences are full length.
        This is critical: wasted positions from padding = wasted compute.

### Step 4: Data Mixing

    When training on multiple sources (web, code, books, etc.):
        Sample from each source with DIFFERENT PROBABILITIES.
        Over-sample high-quality sources (Wikipedia, books).
        Under-sample noisy sources (raw web crawl).

    Example mixture for a code-capable model:
        Web text:    50%
        Code:        20%
        Books:       15%
        Wikipedia:   10%
        Other:        5%

    Curriculum learning (optional):
        Start with easier, cleaner data.
        Gradually introduce more complex or noisier data.
        Can improve final quality and convergence speed.


##### PART 3 — THE FORWARD PASS: COMPUTING PREDICTIONS

### Token Embeddings and Positional Encoding

    INPUT: batch of token IDs, shape (B, T) where B=batch, T=sequence length.

    EMBEDDING LOOKUP: token_embeddings = Embedding(input_ids)
        Shape: (B, T, d)  — d = hidden dimension
        The embedding matrix E ∈ ℝ^{V × d} maps each token ID to a vector.

    POSITIONAL ENCODING: add position information.
        RoPE (Rotary Positional Embedding) — used by LLaMA, Mistral, GPT-NeoX:
            Instead of adding position embeddings, ROTATE the query and key
            vectors by a position-dependent angle.
            Applied INSIDE the attention mechanism (not to the residual stream).
            Key advantage: extrapolates to longer sequences than seen in training.

        ALiBi (Attention with Linear Biases):
            Add a position-dependent BIAS to attention logits: bias[i-j] × slope_h
            Different slopes per head. Zero memory overhead.

        Absolute learnable: standard in GPT-2, BERT.
            x = x + position_embedding[0:T]
            Learned by training. Fixed max length.

### Self-Attention: The Core Computation

    For each layer, given input X ∈ ℝ^{B × T × d}:

    MULTI-HEAD ATTENTION:
        For head h (out of H total heads, d_head = d/H):
            Q_h = X · W_Q_h    shape: (B, T, d_head)
            K_h = X · W_K_h    shape: (B, T, d_head)
            V_h = X · W_V_h    shape: (B, T, d_head)

        Attention scores:
            S_h = (Q_h · K_h^T) / sqrt(d_head)    shape: (B, T, T)

        Causal mask (autoregressive): upper triangle → -∞ (before softmax)
            This prevents attending to future tokens.

        Softmax:
            A_h = softmax(S_h + mask)              shape: (B, T, T)

        Attention output:
            O_h = A_h · V_h                        shape: (B, T, d_head)

    Concatenate heads: O = concat([O_1, ..., O_H])  shape: (B, T, d)
    Output projection: O = O · W_O                  shape: (B, T, d)

    Total FLOPS for attention in one layer:
        QKV projection: 2 × B × T × d × 3d = 6·B·T·d²
        Attention matrix: 2 × B × H × T² × d_head = 4·B·T²·d
        Output proj:      2 × B × T × d × d = 2·B·T·d²
        Per-layer total: (8·d² + 4·T·d) × B × T

    For LLaMA-2 7B (d=4096, T=4096, B=1):
        Per-layer: ≈ 8 × 4096² × 4096 ≈ 550 GFLOP/layer
        32 layers:  ≈ 17.6 TFLOP per forward pass (just attention)

### Feed-Forward Network (SwiGLU)

    LLaMA uses SwiGLU activation (Shazeer, 2020):
        x_gate = Linear(x, d_ff)   shape: (B, T, d_ff)
        x_up   = Linear(x, d_ff)   shape: (B, T, d_ff)
        x_ff   = SiLU(x_gate) × x_up
        output = Linear(x_ff, d)   shape: (B, T, d)

    SiLU(x) = x × σ(x)  (Sigmoid Linear Unit / Swish)

    Three weight matrices (hence the "Swi-GLU" name):
        W_gate: (d, d_ff), W_up: (d, d_ff), W_down: (d_ff, d)

    FLOPs per FFN layer: 2 × B × T × d × (2×d_ff + d_ff) = 6·B·T·d·d_ff

### Loss Computation

    After the final layer, apply:
        LM Head: Linear(hidden, V) → logits shape (B, T, V)
        Softmax: probabilities over vocabulary
        Cross-entropy: L = -log P(t_i | t_0...t_{i-1}) for each position

    Practical implementation:
        logits = logits.view(B*T, V)    # flatten batch × time
        labels = labels.view(B*T)       # flatten
        loss   = F.cross_entropy(logits, labels, ignore_index=-100)

    ignore_index=-100: mask out padding positions (if any) from loss.
    For packed batches (no padding), all positions contribute.


##### PART 4 — THE BACKWARD PASS: COMPUTING GRADIENTS

### Backpropagation Through a Transformer

    During the forward pass, PyTorch builds a COMPUTATION GRAPH:
        Each operation records: inputs, outputs, how to compute ∂output/∂input.
        This graph is the autograd tape.

    During backward: traverse the graph in reverse.
        For each operation: compute local gradient and propagate upstream.
        Chain rule: ∂L/∂w = ∂L/∂output × ∂output/∂w

    In a transformer:
        1. Backprop through loss → logits
        2. Backprop through LM head (linear layer)
        3. For each layer from N to 1:
            a. Backprop through residual connection (gradient flows through unchanged)
            b. Backprop through FFN (three linear layers + SiLU)
            c. Backprop through attention (softmax, QKV projections)
            d. Backprop through RMSNorm
        4. Backprop through embedding → gradient w.r.t. token embeddings

    Computational cost of backward:
        Backward pass ≈ 2× forward pass in FLOPs.
        Reason: for each weight, must compute both ∂L/∂output AND ∂L/∂weight.
        Total training FLOP per token ≈ 6 × model_params (rule of thumb):
            Forward:  2 × model_params
            Backward: 4 × model_params (backward + weight gradient)

### Gradient Accumulation

    Single-GPU memory limits the BATCH SIZE that fits at once.
    But optimal training often requires a LARGE effective batch.

    Gradient accumulation simulates a larger batch without more memory:
        Split the desired batch (B_target) into micro-batches of size B_micro.
        Run forward + backward for each micro-batch.
        Accumulate gradients: grad_total += grad_micro (do NOT zero_grad).
        After K steps: optimizer.step() + optimizer.zero_grad().
        Effective batch size = B_micro × K.

    PyTorch implementation:
        accumulation_steps = 8   # accumulate 8 micro-batches
        for step, (input_ids, labels) in enumerate(dataloader):
            loss = model(input_ids, labels) / accumulation_steps
            loss.backward()     # accumulate gradients
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

    Important: divide loss by accumulation_steps BEFORE backward.
    Otherwise gradients are K× too large relative to the intended batch size.

### Gradient Clipping

    During training, gradients can spike due to:
        - Difficult training examples
        - Learning rate that is slightly too high
        - Accumulated floating-point errors

    Without clipping: one bad gradient spike → parameters blow up → NaN loss.

    Global gradient norm clipping:
        grad_norm = sqrt(Σ_i ||∂L/∂w_i||²)  (L2 norm of all gradients)
        If grad_norm > clip_value:
            scale = clip_value / grad_norm
            For each gradient: g_i ← g_i × scale

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        This is called between loss.backward() and optimizer.step().
        max_norm=1.0 is a common default for LLM training.

    What it prevents:
        A single outlier gradient cannot derail training.
        Gradients are rescaled so the total norm is bounded.
        The DIRECTION of the gradient is preserved; only the magnitude changes.


##### PART 5 — THE OPTIMIZER: UPDATING WEIGHTS

### AdamW: The Standard LLM Optimizer

    AdamW (Adam with decoupled weight decay) is used by virtually every
    modern LLM. It combines adaptive learning rates with weight decay.

    For each parameter w with gradient g at step t:

        m_t = β_1 × m_{t-1} + (1 - β_1) × g       # 1st moment (momentum)
        v_t = β_2 × v_{t-1} + (1 - β_2) × g²       # 2nd moment (adaptive LR)

        m̂_t = m_t / (1 - β_1^t)   # bias correction (important early in training)
        v̂_t = v_t / (1 - β_2^t)

        w_t = w_{t-1} - lr × m̂_t / (sqrt(v̂_t) + ε) - lr × λ × w_{t-1}
                                                              ↑ weight decay

    Standard hyperparameters:
        β_1 = 0.9          (momentum coefficient)
        β_2 = 0.95–0.999   (second moment, often 0.95 for LLMs)
        ε   = 1e-8         (numerical stability)
        λ   = 0.01–0.1     (weight decay, often 0.1)

    Why AdamW, not Adam?
        Adam's weight decay is entangled with the adaptive learning rate.
        AdamW decouples them: weight decay applied to w directly, not scaled.
        Better generalisation in practice.

    Memory cost of AdamW:
        Weights:    1 × n_params × bpe (e.g., BF16 = 2 bytes)
        Gradients:  1 × n_params × bpe
        Momentum m: 1 × n_params × 4 bytes (FP32, must be high precision)
        Variance v: 1 × n_params × 4 bytes (FP32)
        Total:      (bpe + bpe + 4 + 4) × n_params

    For LLaMA-2 7B with BF16 weights:
        Weights:   7B × 2 = 14 GB
        Gradients: 7B × 2 = 14 GB
        Moments:   7B × 4 × 2 = 56 GB (FP32)
        Total:     84 GB — exceeds A100 80GB!
        → Only fits with BF16 weights AND gradient checkpointing AND optimizer sharding.

### Learning Rate Schedule

    The learning rate is not constant — it follows a schedule:

    WARMUP PHASE (first 1–2% of training):
        LR ramps from 0 to peak_lr linearly over warmup_steps.
        Why warmup? At step 0, Adam's moment estimates are zero → biased.
        Without warmup: first step takes a huge, poorly-directed update.
        Warmup gives time for m and v to accumulate accurate estimates.
        Typical: warmup_steps = 1000–2000.

    COSINE DECAY PHASE (remaining 98–99% of training):
        LR decays from peak_lr to min_lr following a cosine curve:
        lr(t) = min_lr + 0.5 × (peak_lr - min_lr) × (1 + cos(π × t/T))

    FINAL VALUE: min_lr ≈ 10% of peak_lr (e.g., 1e-5 if peak is 3e-4).

    Typical values:
        peak_lr:    3e-4 to 1e-4  (depends on model size — larger models need smaller LR)
        min_lr:     3e-5 to 1e-5
        LLaMA-2 7B: peak 3e-4, decay to 3e-5 over 2T tokens.


##### PART 6 — MEMORY MANAGEMENT ON A SINGLE GPU

### The Four Memory Consumers

    On a single GPU, the HBM memory is consumed by:

    1. MODEL WEIGHTS:
        Parameters × bytes_per_element
        LLaMA-2 7B at BF16: 7B × 2 = 14 GB

    2. GRADIENTS:
        Same shape as weights — one gradient per parameter.
        LLaMA-2 7B: 14 GB (BF16 gradients)

    3. OPTIMIZER STATES:
        AdamW: 2 × parameters × 4 bytes (FP32 momentum + variance)
        LLaMA-2 7B: 7B × 4 × 2 = 56 GB
        8-bit Adam (bitsandbytes): 7B × 1 byte × 2 = 14 GB (saves 42 GB!)

    4. ACTIVATIONS (the often-overlooked memory hog):
        Intermediate computations stored for backward pass.
        Scale: B × T × d × N × (factor depending on architecture)
        For B=1, T=4096, d=4096, N=32: ≈ 20–30 GB without checkpointing!
        With gradient checkpointing: reduced to ≈ 2–4 GB.

### Gradient Checkpointing (Activation Recomputation)

    Problem: backward pass needs activations from forward pass.
    Solution: DON'T store all activations. Recompute them on demand.

    Standard (no checkpointing):
        Forward: compute all activations and STORE them all.
        Backward: use stored activations to compute gradients.
        Memory: O(N × B × T × d) for all intermediate activations.

    With gradient checkpointing:
        Divide the N layers into √N segments (or checkpoint every K layers).
        Forward: only store activations at SEGMENT BOUNDARIES (checkpoints).
        Backward: when a segment is needed, RECOMPUTE its forward pass.
        Memory: O(√N × B × T × d) instead of O(N × ...).

    Trade-off:
        Memory: saves 60–80% of activation memory.
        Compute: costs one extra forward pass per layer → ~33% slower.
        For large models where memory is the constraint: always worth it.

    PyTorch implementation:
        from torch.utils.checkpoint import checkpoint
        def forward_with_ckpt(layer, x):
            return checkpoint(layer, x, use_reentrant=False)

### Mixed Precision Training (AMP)

    Train with FP16 or BF16 computations but FP32 optimizer states.

    Why NOT use pure FP16 for everything?
        FP16 range: ±65,504 — gradients can underflow to zero or overflow to Inf.
        Loss scaling (multiply loss by large scalar before backward) mitigates this.
        BF16 has FP32 range: ±3.4×10³⁸ — no overflow risk, no scaling needed.
        BF16 is the preferred format for LLM training on Ampere GPUs (A100).

    Automatic Mixed Precision (AMP) with BF16:
        from torch.amp import autocast
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss   = loss_fn(logits, labels)
        # backward in BF16 (fast, memory-efficient)
        loss.backward()
        # optimizer step in FP32 (precise weight updates)
        optimizer.step()

    Memory savings from AMP:
        Weights: FP32 → BF16: 4→2 bytes (2× saving)
        Activations: computed in BF16 → 2× saving
        Optimizer states: kept in FP32 (no saving there)
        Net: significant activation and weight savings.

### Memory Budget for LLaMA-2 7B on A100 80GB

    With BF16 + gradient checkpointing + 8-bit Adam:
        Weights:       7B × 2  =  14 GB
        Gradients:     7B × 2  =  14 GB
        8-bit optimizer: 7B × 2 = 14 GB (vs 56 GB for FP32 Adam)
        Activations:   ≈ 2 GB  (gradient checkpointing)
        Total:         ≈ 44 GB  → fits in A100 80GB ✅

    Without these tricks:
        Weights:       7B × 4  =  28 GB (FP32)
        Gradients:     7B × 4  =  28 GB
        Optimizer:     7B × 8  =  56 GB
        Activations:  ≈ 30 GB  (no checkpointing)
        Total:        ≈ 142 GB → doesn't fit in A100 80GB ❌


##### PART 7 — THE TRAINING LOOP AND CHECKPOINTING

### The Complete Training Loop

    for step, batch in enumerate(dataloader):
        # 1. Move data to GPU
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        # 2. Forward pass (BF16)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss   = F.cross_entropy(logits.view(-1, V), labels.view(-1))
            loss   = loss / gradient_accumulation_steps

        # 3. Backward pass (accumulate gradients)
        loss.backward()

        # 4. Optimizer step (every N accumulation steps)
        if (step + 1) % gradient_accumulation_steps == 0:
            # 4a. Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 4b. Update weights
            optimizer.step()
            # 4c. Update LR
            scheduler.step()
            # 4d. Zero gradients
            optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory

        # 5. Logging
        if step % log_every == 0:
            log_metrics({"loss": loss.item(), "lr": scheduler.get_lr(),
                         "step": step, "tokens_seen": step * B * T})

        # 6. Checkpoint
        if step % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step)

### Checkpointing Strategy

    What to save:
        model.state_dict()          — all weights and biases
        optimizer.state_dict()      — Adam moments for each parameter
        scheduler.state_dict()      — step count, current LR
        step, epoch, tokens_seen    — training progress metadata
        loss_history, config        — reproducibility and debugging

    checkpoint = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step":      global_step,
        "config":    model_config,
    }
    torch.save(checkpoint, f"checkpoint_step_{global_step}.pt")

    Save frequency: every 1000–5000 steps.
    Keep latest 3–5 checkpoints (delete older ones).
    Always save before and after risky operations.

    Loading a checkpoint:
        ckpt = torch.load("checkpoint.pt")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        # Resume training from start_step

### Tokens Per Second and Training Time Estimation

    Key metric: tokens/second (throughput).

    tokens_per_step = batch_size × sequence_length
    tokens_per_second = tokens_per_step / time_per_step

    For LLaMA-2 7B on one A100 80GB:
        batch_size=1, seq=4096, grad_ckpt, BF16:
        ~1000–2000 tokens/second

    To train on 2T tokens:
        time_seconds = 2e12 / 1500 ≈ 1.33 billion seconds → 15,000 days
        On ONE A100: 40+ years. Obviously requires multi-GPU.

    But for a SMALL model (e.g., 125M parameters) on 10B tokens:
        ~10,000 tokens/second
        time = 10e9 / 10000 = 1,000,000 seconds ≈ 11.6 days on one GPU.
        Feasible for experimentation!

    Model FLOP Utilisation (MFU):
        MFU = actual_FLOPS / theoretical_peak_FLOPS
        Computed as: 6 × N_params × tokens_per_second / peak_TFLOPs × 1e12
        Good training: MFU = 40–55%. World-class: 50–55%.
        Below 20%: something is wrong (data bottleneck, tiny batch size, etc.)


##### PART 8 — EVALUATING, MONITORING, AND DIAGNOSING TRAINING

### The Loss Curve: Reading the Signal

    TRAINING LOSS (per-step):
        Should decrease steadily with noise.
        Smooth if gradient accumulation is large (more stable).
        Typical final value: 1.5–2.5 (depends on data entropy).
        Perplexity = exp(loss): 4.5–12 for good models on English text.

    VALIDATION LOSS (every N steps on held-out data):
        Tracks generalisation (not just memorisation).
        Should track training loss closely (healthy gap < 0.1).
        Large gap: overfitting (reduce model or increase data).
        Validation loss RISING: overfitting or data distribution mismatch.

    WHAT HEALTHY LOOKS LIKE:
        Step 0:        loss ≈ log(V) = log(32000) ≈ 10.4 (random predictions)
        Step 100:      loss ≈ 6–7    (model learning common tokens)
        Step 1000:     loss ≈ 3–4    (basic language structure learned)
        Step 10,000:   loss ≈ 2.5–3  (language well-modelled)
        Step 100,000+: loss ≈ 2.0–2.5 (fine structure learned)

### Diagnosing Common Training Problems

    PROBLEM 1: LOSS IS NaN OR inf
        Causes:
            Learning rate too high → gradients explode → NaN.
            FP16 overflow (use BF16 or add loss scaling).
            Division by zero in some operation (LayerNorm, log).
        Fixes:
            Reduce learning rate by 10×.
            Switch to BF16 from FP16.
            Add gradient clipping (max_norm=1.0).
            Load a recent checkpoint and resume from there.

    PROBLEM 2: LOSS NOT DECREASING (flat curve)
        Causes:
            Learning rate too low → gradients too small → no progress.
            Data pipeline bug → model sees same batch repeatedly.
            Wrong loss implementation → loss not connected to parameters.
            Bug in gradient accumulation division.
        Fixes:
            Check that loss.requires_grad is True.
            Log grad_norm: should be non-zero and reasonable.
            Overfit on a tiny batch (10 samples) — loss should drop to ~0.
            Print input_ids for a few steps — verify data is different each step.

    PROBLEM 3: LOSS DECREASES THEN SPIKES
        Causes:
            Bad batch (corrupt data, extremely long sequence, all-zero input).
            Learning rate spike (bug in scheduler, especially cosine warmup).
        Fixes:
            Add data quality filtering: skip batches where all tokens are the same.
            Log loss before it spikes: identify the step and inspect the batch.
            Use gradient clipping to bound the effect of bad batches.

    PROBLEM 4: LOSS DECREASES TOO SLOWLY
        Causes:
            Batch size too small → noisy gradient direction.
            Learning rate too conservative.
            Data not shuffled → model sees similar data repeatedly.
        Fixes:
            Increase effective batch size via gradient accumulation.
            Tune learning rate with a short sweep (100 steps each).
            Verify data shuffling.

    PROBLEM 5: OUT OF MEMORY (OOM)
        Causes:
            Batch size too large.
            Sequence length too long.
            Activations not checkpointed.
            Optimizer in FP32 on large model.
        Fixes:
            Reduce batch size (then increase gradient accumulation).
            Enable gradient checkpointing.
            Use 8-bit Adam (bitsandbytes).
            Use BF16/FP16 for weights.

### Evaluation Benchmarks

    During and after training, evaluate on standard benchmarks:

    PERPLEXITY on held-out text:
        The primary training signal. Lower = better.
        Measure on WikiText-103, PTB, Pile-CC (from a holdout split).

    ZERO-SHOT BENCHMARKS (no fine-tuning):
        HellaSwag:    commonsense reasoning (completion selection)
        PIQA:         physical intuition Q&A
        Winogrande:   pronoun resolution (Winograd schema)
        ARC-Easy/Challenge: grade school science Q&A
        TriviaQA:     factual knowledge retrieval

    FEW-SHOT BENCHMARKS (5–10 examples as context):
        MMLU:         57 subjects (academic knowledge)
        BBH (BIG-Bench Hard): reasoning tasks
        GSM8K:        grade school math word problems

    How to run evaluations during training:
        Every 5000–10000 steps, pause training.
        Run lm-evaluation-harness (EleutherAI's standard evaluation suite).
        Log results to W&B/TensorBoard alongside loss.
        This tracks whether the model is generalising, not just memorising.

"""


OPERATIONS = {

    "1 · Building the LLM Training Pipeline from Scratch": {
        "description": (
            "Implement the complete single-GPU LLM training loop step by step. "
            "BPE tokenisation and dataset chunking. "
            "GPT-style transformer forward pass with causal attention. "
            "Loss computation, backward pass, gradient clipping. "
            "Learning rate schedule (warmup + cosine decay). "
            "Memory budget calculation for a given model and GPU."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import math

print("=" * 65)
print("  LLM TRAINING PIPELINE FROM SCRATCH")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Tokenisation simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Tokenisation: text → token IDs → training chunks")
print("━" * 65)
print()

class SimpleBPETokenizer:
    """
    Simplified BPE tokeniser to illustrate the concept.
    Real tokenisers (tiktoken, sentencepiece) are optimised in C++.
    """
    def __init__(self, vocab_size=256):
        # Start with byte-level vocabulary (256 bytes)
        self.vocab   = {i: bytes([i]) for i in range(256)}
        self.merges  = {}   # (pair) → merged_token_id
        self.vocab_size = vocab_size

    def get_pairs(self, ids):
        """Get all adjacent pairs in a sequence."""
        return set(zip(ids[:-1], ids[1:]))

    def train(self, text, target_size=300, verbose=True):
        """
        Run BPE training: repeatedly merge most frequent pair.
        """
        # Convert text to byte sequence
        ids = list(text.encode("utf-8"))
        current_vocab = dict(self.vocab)
        next_id = 256

        if verbose:
            print(f"  Starting vocab size: {len(current_vocab)}")
            print(f"  Training text: {len(ids)} bytes")
            print()

        while next_id < target_size:
            # Count all pairs
            pair_counts = {}
            for a, b in zip(ids[:-1], ids[1:]):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

            if not pair_counts:
                break

            # Most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                break   # no repeating pairs left

            # Merge: replace all occurrences of best_pair with next_id
            self.merges[best_pair] = next_id
            # Decode the merged token's bytes
            a_bytes = (current_vocab[best_pair[0]]
                       if best_pair[0] in current_vocab
                       else bytes([best_pair[0]]))
            b_bytes = (current_vocab[best_pair[1]]
                       if best_pair[1] in current_vocab
                       else bytes([best_pair[1]]))
            current_vocab[next_id] = a_bytes + b_bytes

            if verbose and (next_id - 256) < 10:
                merged_str = (a_bytes + b_bytes).decode("utf-8", errors="replace")
                print(f"  Merge {next_id-256+1}: "
                      f"({best_pair[0]}, {best_pair[1]}) → {next_id} "
                      f"= '{merged_str}'  (count={best_count})")

            # Apply merge to sequence
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids)-1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(next_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            self.vocab = current_vocab
            next_id += 1

        if verbose:
            print(f"\n  Final vocab size: {next_id}")
            print(f"  Sequence length: {len(ids)} tokens (compressed from original)")
        return ids


# Demo tokenisation
sample_text = ("The quick brown fox jumps over the lazy dog. " * 5 +
               "Language models learn by predicting the next token. " * 5)
print(f"  Training BPE on: '{sample_text[:60]}...'")
print(f"  Original length: {len(sample_text)} characters")
print()

tokenizer = SimpleBPETokenizer()
token_ids = tokenizer.train(sample_text, target_size=300, verbose=True)
print()
print(f"  Compression ratio: {len(sample_text)}/{len(token_ids)} = "
      f"{len(sample_text)/len(token_ids):.2f}x")
print()

# Show dataset chunking
print("  Dataset chunking (context_length=16 for demo):")
CONTEXT_LEN = 16
chunks       = []
for start in range(0, len(token_ids) - CONTEXT_LEN, CONTEXT_LEN):
    chunk = token_ids[start:start + CONTEXT_LEN + 1]
    if len(chunk) == CONTEXT_LEN + 1:
        chunks.append(chunk)

print(f"  Total tokens: {len(token_ids)}")
print(f"  Context length: {CONTEXT_LEN}")
print(f"  Number of chunks: {len(chunks)}")
print()
for i, chunk in enumerate(chunks[:3]):
    inp   = chunk[:-1]
    label = chunk[1:]
    print(f"  Chunk {i}: input_ids={inp[:8]}...  labels={label[:8]}...")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Transformer forward pass (NumPy for clarity)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Transformer forward pass (step by step)")
print("━" * 65)
print()

class TinyGPT:
    """
    Minimal GPT-style transformer for educational purposes.
    Uses NumPy so no GPU needed — shows the exact computations.
    """
    def __init__(self, vocab_size=300, d=32, n_heads=4, n_layers=2, ctx=16):
        self.V = vocab_size; self.d = d
        self.H = n_heads; self.d_h = d // n_heads
        self.N = n_layers; self.T = ctx
        np.random.seed(42)
        s = 0.02   # small init scale

        # Weights — in a real model these are all learnable parameters
        self.embed    = np.random.randn(vocab_size, d) * s         # token embeddings
        self.pos_emb  = np.random.randn(ctx, d) * s                # position embeddings
        self.layers   = []
        for _ in range(n_layers):
            layer = {
                "ln1_w": np.ones(d), "ln1_b": np.zeros(d),         # LayerNorm 1
                "W_qkv": np.random.randn(d, 3*d) * s,               # QKV projection
                "W_out": np.random.randn(d, d) * s,                  # attention output
                "ln2_w": np.ones(d), "ln2_b": np.zeros(d),          # LayerNorm 2
                "W_ff1": np.random.randn(d, 4*d) * s,               # FFN up
                "W_ff2": np.random.randn(4*d, d) * s,               # FFN down
            }
            self.layers.append(layer)
        self.ln_f_w   = np.ones(d); self.ln_f_b = np.zeros(d)     # final LN
        self.lm_head  = np.random.randn(d, vocab_size) * s         # LM head

    def layer_norm(self, x, w, b, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        return w * (x - mean) / np.sqrt(var + eps) + b

    def softmax(self, x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def attention(self, x, layer):
        T, d = x.shape
        # QKV projection
        qkv = x @ layer["W_qkv"]           # (T, 3d)
        Q, K, V = np.split(qkv, 3, axis=-1) # each (T, d)

        # Reshape for multi-head: (T, H, d_h) → (H, T, d_h)
        Q = Q.reshape(T, self.H, self.d_h).transpose(1,0,2)  # (H, T, d_h)
        K = K.reshape(T, self.H, self.d_h).transpose(1,0,2)
        V = V.reshape(T, self.H, self.d_h).transpose(1,0,2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_h)
        scores = Q @ K.transpose(0,2,1) * scale    # (H, T, T)

        # Causal mask: upper triangle → -inf
        mask = np.triu(np.full((T,T), -1e9), k=1)
        scores = scores + mask[None, :, :]          # broadcast over heads

        attn = self.softmax(scores, axis=-1)        # (H, T, T)
        out  = attn @ V                             # (H, T, d_h)

        # Reshape back: (H, T, d_h) → (T, H*d_h) = (T, d)
        out = out.transpose(1,0,2).reshape(T, self.d)
        return out @ layer["W_out"]                 # (T, d)

    def forward(self, token_ids):
        T    = len(token_ids)
        x    = self.embed[token_ids] + self.pos_emb[:T]   # (T, d)

        for layer in self.layers:
            # Attention sub-block
            x = x + self.attention(self.layer_norm(x, layer["ln1_w"], layer["ln1_b"]),
                                    layer)
            # FFN sub-block
            ln2 = self.layer_norm(x, layer["ln2_w"], layer["ln2_b"])
            ff  = self.gelu(ln2 @ layer["W_ff1"]) @ layer["W_ff2"]
            x   = x + ff

        x      = self.layer_norm(x, self.ln_f_w, self.ln_f_b)
        logits = x @ self.lm_head                          # (T, V)
        return logits

    def cross_entropy_loss(self, logits, labels):
        """Compute cross-entropy loss."""
        T, V   = logits.shape
        # Numerical stability: subtract max before exp
        logits = logits - logits.max(axis=-1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        loss = -log_probs[np.arange(T), labels].mean()
        return loss


print("  Creating TinyGPT (32-dim, 2-layer, 4-head)...")
model = TinyGPT(vocab_size=300, d=32, n_heads=4, n_layers=2, ctx=16)

# Count parameters
n_params = (300*32 + 16*32 +                    # embed + pos
            2 * (32*96 + 32*32 + 2*32 + 32*128 + 128*32) +  # 2 layers
            32 + 32*300)                          # final_ln + lm_head
print(f"  Approximate parameters: ~{n_params:,}")
print()

# Run a forward pass
sample_ids = token_ids[:16]
print(f"  Input tokens (first 16): {sample_ids}")
t0     = time.perf_counter()
logits = model.forward(sample_ids)
t_fwd  = (time.perf_counter() - t0) * 1000

labels = token_ids[1:17]
loss   = model.cross_entropy_loss(logits, labels)

print(f"  Forward pass: {t_fwd:.2f}ms")
print(f"  Logits shape: {logits.shape}  (T=16, V=300)")
print(f"  Loss: {loss:.4f}  (initial ≈ log(300) = {math.log(300):.4f} for random model)")
print()

# Show what logits look like
top3 = np.argsort(logits[0])[-3:][::-1]
print(f"  At position 0: top-3 predicted next tokens = {top3}")
print(f"  Their probabilities: {np.exp(logits[0, top3] - logits[0].max()) / np.exp(logits[0] - logits[0].max()).sum():.4f}")
print()
''',
    },

    "2 · Memory Budget, Mixed Precision, and Gradient Checkpointing": {
        "description": (
            "Compute exact memory requirements for any LLM configuration. "
            "Show how mixed precision (BF16) reduces memory vs FP32. "
            "Simulate gradient checkpointing memory savings. "
            "Implement gradient accumulation for large effective batch size. "
            "Implement the learning rate warmup + cosine decay schedule. "
            "Show training loop with all production best practices."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time

print("=" * 65)
print("  MEMORY BUDGET, MIXED PRECISION, AND TRAINING LOOP")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Memory budget calculator
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — GPU memory budget for LLM training")
print("━" * 65)
print()

def compute_memory_budget(
    n_params,           # number of parameters (e.g., 7e9)
    batch_size,         # micro-batch size
    seq_len,            # sequence length (tokens)
    n_layers,           # number of transformer layers
    hidden_dim,         # d_model
    weight_dtype="bf16",           # weights precision
    optimizer="adamw_fp32",        # optimizer type
    grad_checkpointing=False,      # gradient checkpointing
):
    GB = 1024**3

    bpe = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
    w_bpe = bpe[weight_dtype]

    # Weights
    weight_gb = n_params * w_bpe / GB

    # Gradients (same precision as weights)
    grad_gb = n_params * w_bpe / GB

    # Optimizer states
    if optimizer == "adamw_fp32":
        # Adam: 2 FP32 tensors per parameter (m + v)
        optim_gb = n_params * 4 * 2 / GB
    elif optimizer == "adam_bf16":
        optim_gb = n_params * 2 * 2 / GB
    elif optimizer == "adam_8bit":
        # 8-bit Adam: ~1 byte per state × 2
        optim_gb = n_params * 1 * 2 / GB
    elif optimizer == "sgd_fp32":
        optim_gb = n_params * 4 / GB  # only momentum
    else:
        optim_gb = 0

    # Activations
    # Per-layer: batch × seq_len × hidden_dim × (Q+K+V+attn+FFN ≈ 12)
    bytes_per_activation = 2  # BF16
    if grad_checkpointing:
        # Only store activations at checkpoints (approx sqrt(N) checkpoints)
        n_ckpts = math.ceil(math.sqrt(n_layers))
        act_per_layer = batch_size * seq_len * hidden_dim * bytes_per_activation
        act_gb = n_ckpts * act_per_layer / GB
    else:
        # All intermediate activations stored
        # Rough estimate: ~12× hidden_dim of intermediates per layer
        act_per_layer = batch_size * seq_len * hidden_dim * 12 * bytes_per_activation
        act_gb = n_layers * act_per_layer / GB

    total_gb = weight_gb + grad_gb + optim_gb + act_gb

    return {
        "weights_gb":   weight_gb,
        "gradients_gb": grad_gb,
        "optimizer_gb": optim_gb,
        "activations_gb": act_gb,
        "total_gb":     total_gb,
    }


def print_budget(config_name, n_params, **kwargs):
    mem = compute_memory_budget(n_params, **kwargs)
    print(f"  {config_name}")
    print(f"    Weights:     {mem['weights_gb']:6.1f} GB")
    print(f"    Gradients:   {mem['gradients_gb']:6.1f} GB")
    print(f"    Optimizer:   {mem['optimizer_gb']:6.1f} GB")
    print(f"    Activations: {mem['activations_gb']:6.1f} GB")
    print(f"    ─────────────────────────────")
    fits = "✅ FITS" if mem['total_gb'] < 80 else "❌ OOM"
    print(f"    TOTAL:       {mem['total_gb']:6.1f} GB   {fits} (A100 80GB)")
    print()


print("  Memory budget analysis for LLaMA-2 7B (7B params):")
print(f"  n_params=7B, n_layers=32, d=4096, seq=4096, batch=1")
print()

print_budget(
    "1. FP32 everything, no checkpointing (naive baseline)",
    7e9, batch_size=1, seq_len=4096, n_layers=32, hidden_dim=4096,
    weight_dtype="fp32", optimizer="adamw_fp32", grad_checkpointing=False
)

print_budget(
    "2. BF16 weights, FP32 Adam, no checkpointing",
    7e9, batch_size=1, seq_len=4096, n_layers=32, hidden_dim=4096,
    weight_dtype="bf16", optimizer="adamw_fp32", grad_checkpointing=False
)

print_budget(
    "3. BF16 + FP32 Adam + gradient checkpointing",
    7e9, batch_size=1, seq_len=4096, n_layers=32, hidden_dim=4096,
    weight_dtype="bf16", optimizer="adamw_fp32", grad_checkpointing=True
)

print_budget(
    "4. BF16 + 8-bit Adam + gradient checkpointing",
    7e9, batch_size=1, seq_len=4096, n_layers=32, hidden_dim=4096,
    weight_dtype="bf16", optimizer="adam_8bit", grad_checkpointing=True
)

# Smaller model: 125M params (GPT-2 small)
print()
print("  Memory for small model: 125M params (GPT-2 small):")
print(f"  n_layers=12, d=768, seq=1024")
print()
print_budget(
    "125M params, BF16, FP32 Adam, no checkpointing, batch=8",
    125e6, batch_size=8, seq_len=1024, n_layers=12, hidden_dim=768,
    weight_dtype="bf16", optimizer="adamw_fp32", grad_checkpointing=False
)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Learning rate schedule
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Learning rate schedule: warmup + cosine decay")
print("━" * 65)
print()

def get_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    """
    Cosine decay with linear warmup.
    - Linear warmup: step 0 → warmup_steps: LR goes from 0 to max_lr
    - Cosine decay:  warmup_steps → total_steps: LR goes from max_lr to min_lr
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# LLaMA-2 7B training schedule
MAX_LR      = 3e-4
MIN_LR      = 3e-5
WARMUP_STEPS = 2000
TOTAL_STEPS  = 500000   # 2T tokens / (4096 tokens/step)

print(f"  Schedule: max_lr={MAX_LR:.0e}, min_lr={MIN_LR:.0e}")
print(f"  Warmup: {WARMUP_STEPS} steps, Total: {TOTAL_STEPS} steps")
print()
print(f"  {'Step':>10} | {'LR':>12} | {'Phase':>20} | {'Visual'}")
print(f"  {'─'*60}")

for step in [0, 100, 500, 1000, 2000, 5000, 10000, 50000,
             100000, 250000, 499000, 500000]:
    lr    = get_lr(step, TOTAL_STEPS, WARMUP_STEPS, MAX_LR, MIN_LR)
    pct   = lr / MAX_LR
    bar   = "█" * int(pct * 20)
    phase = "WARMUP" if step < WARMUP_STEPS else "COSINE DECAY"
    print(f"  {step:>10} | {lr:>12.2e} | {phase:>20} | [{bar:<20}]")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Gradient accumulation simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Gradient accumulation: simulating large batches")
print("━" * 65)
print()

print("  Gradient accumulation splits a large batch into micro-batches.")
print("  Gradients are summed; optimizer step fires every N micro-batches.")
print()

def simulate_training(
    n_params=125_000_000,
    target_batch=512,     # desired effective batch
    micro_batch=4,        # what fits on GPU
    n_steps=100,
    max_lr=3e-4,
    warmup_steps=10,
):
    accum_steps = target_batch // micro_batch
    print(f"  Config: target_batch={target_batch}, micro_batch={micro_batch}")
    print(f"  Gradient accumulation steps: {accum_steps}")
    print(f"  Effective batch = {micro_batch} × {accum_steps} = {micro_batch*accum_steps}")
    print()

    # Simulate simplified training dynamics
    # Loss ≈ initial_loss × exp(-learning_rate_eff × step)
    initial_loss  = 10.4   # log(vocab_size=32000)
    loss_history  = []
    grad_norm_hist = []
    total_grad    = np.zeros(10)  # proxy for parameter gradients

    np.random.seed(0)
    optimizer_step = 0

    print(f"  {'Step':>6} | {'opt_step':>9} | {'loss':>8} | "
          f"{'grad_norm':>10} | {'lr':>10} | {'tokens_seen'}")
    print(f"  {'─'*68}")

    for micro_step in range(n_steps):
        # Simulate gradient for this micro-batch
        # Real gradient would come from loss.backward()
        noise      = np.random.randn(10) * 0.1
        true_grad  = -0.1 * np.ones(10) + noise   # gradient points toward lower loss
        total_grad += true_grad / accum_steps       # accumulate, scale by accum

        if (micro_step + 1) % accum_steps == 0:
            optimizer_step += 1
            lr = get_lr(optimizer_step, n_steps // accum_steps,
                        warmup_steps, max_lr, max_lr * 0.1)

            # Gradient clipping
            grad_norm = np.linalg.norm(total_grad)
            if grad_norm > 1.0:
                total_grad = total_grad / grad_norm * 1.0
                grad_norm  = 1.0

            # Simulate loss decay
            effective_lr = lr
            loss = initial_loss * math.exp(-0.05 * optimizer_step) + \
                   np.random.randn() * 0.05

            loss_history.append(loss)
            grad_norm_hist.append(grad_norm)
            tokens_seen = optimizer_step * micro_batch * accum_steps * 512

            print(f"  {micro_step+1:>6} | {optimizer_step:>9} | {loss:>8.4f} | "
                  f"{grad_norm:>10.4f} | {lr:>10.2e} | {tokens_seen:,}")

            total_grad = np.zeros(10)  # zero gradients

    return loss_history, grad_norm_hist


loss_hist, gnorm_hist = simulate_training(
    target_batch=512, micro_batch=4, n_steps=200, warmup_steps=5)
print()
print(f"  Final loss: {loss_hist[-1]:.4f}  (started at ~10.4)")
print(f"  Average grad_norm: {np.mean(gnorm_hist):.4f}")
print()
''',
    },

    "3 · Evaluation, Diagnostics, and Training Health Monitoring": {
        "description": (
            "Implement perplexity evaluation on a validation set. "
            "Build a training loss curve analyser that detects anomalies. "
            "Show gradient norm tracking to detect instability. "
            "Implement the MFU (Model FLOP Utilisation) metric. "
            "Build a training health dashboard. "
            "Show how to overfit on a small batch to verify gradient flow."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time

print("=" * 65)
print("  EVALUATION, DIAGNOSTICS, AND TRAINING HEALTH")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Perplexity evaluation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Perplexity: the primary evaluation metric")
print("━" * 65)
print()

print("  Perplexity = exp(average cross-entropy loss)")
print("  Lower = better. Random model: exp(log(V)) = V (vocab size)")
print("  Good LLM on English: perplexity 5–15")
print()

def compute_perplexity(losses):
    """Compute perplexity from a list of per-token losses."""
    avg_loss = np.mean(losses)
    return math.exp(avg_loss)

# Simulate loss values at different training stages
stages = [
    ("Untrained (random)",       10.4,  0.3),   # log(32000) + noise
    ("Early training (1k steps)", 6.0,  0.2),
    ("Mid training (10k steps)",  3.5,  0.15),
    ("Late training (100k steps)", 2.5, 0.1),
    ("Fully trained (500k steps)", 2.0, 0.08),
    ("GPT-2 (official)",           3.35, 0.05),
    ("LLaMA-2 7B (official)",      2.08, 0.03),
]

print(f"  {'Stage':<35} | {'Avg Loss':>10} | {'Perplexity':>12} | "
      f"{'Interpretation'}")
print(f"  {'─'*80}")
for name, loss, std in stages:
    ppl = compute_perplexity([loss])
    interp = ("useless" if ppl > 1000
               else "poor" if ppl > 100
               else "weak" if ppl > 30
               else "decent" if ppl > 15
               else "good" if ppl > 8
               else "strong")
    print(f"  {name:<35} | {loss:>10.4f} | {ppl:>12.2f} | {interp}")
print()

# Validation split demonstration
print("  Validation evaluation protocol:")
print()
VAL_PROTOCOL = """
  Standard evaluation procedure:

  1. HOLD OUT validation set (never seen during training):
     val_data = entire_dataset[last_5_percent]  # or a fixed held-out set
     # For deterministic comparison: use WikiText-103 or Pile-CC 1% holdout

  2. SET model to eval mode:
     model.eval()
     torch.set_grad_enabled(False)   # no gradients needed = saves memory

  3. COMPUTE loss on all validation batches:
     val_losses = []
     for batch in val_loader:
         with torch.no_grad():
             with autocast(device_type='cuda', dtype=torch.bfloat16):
                 logits = model(batch['input_ids'])
                 loss   = F.cross_entropy(logits.view(-1, V),
                                          batch['labels'].view(-1))
         val_losses.append(loss.item())

  4. AGGREGATE:
     val_loss = np.mean(val_losses)
     val_ppl  = math.exp(val_loss)
     print(f"Validation loss: {val_loss:.4f}  PPL: {val_ppl:.2f}")

  5. LOG AND COMPARE:
     if val_ppl < best_ppl:
         best_ppl = val_ppl
         save_best_checkpoint(model)

  Evaluation frequency: every 1000–5000 training steps.
  Too frequent: slows down training. Too infrequent: misses divergence early.
"""
print(VAL_PROTOCOL)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Training health dashboard
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Training health dashboard and anomaly detection")
print("━" * 65)
print()

class TrainingMonitor:
    """
    Monitors training metrics and detects anomalies.
    In production: pipe these to W&B, TensorBoard, or MLflow.
    """
    def __init__(self, window=50):
        self.window       = window
        self.train_losses = []
        self.val_losses   = []
        self.grad_norms   = []
        self.lrs          = []
        self.steps        = []
        self.alerts       = []

    def log(self, step, train_loss, grad_norm, lr, val_loss=None):
        self.steps.append(step)
        self.train_losses.append(train_loss)
        self.grad_norms.append(grad_norm)
        self.lrs.append(lr)
        if val_loss is not None:
            self.val_losses.append((step, val_loss))
        self._check_health(step, train_loss, grad_norm)

    def _check_health(self, step, loss, grad_norm):
        """Detect training anomalies."""
        alerts = []

        # NaN/Inf check
        if not math.isfinite(loss):
            alerts.append(f"🚨 CRITICAL: loss={loss} at step {step}")

        # Gradient explosion
        if grad_norm > 10.0:
            alerts.append(f"⚠️  GRAD EXPLOSION: norm={grad_norm:.2f} at step {step}")
        elif grad_norm < 1e-7 and step > 100:
            alerts.append(f"⚠️  VANISHING GRAD: norm={grad_norm:.2e} at step {step}")

        # Loss not decreasing
        if len(self.train_losses) >= self.window:
            recent     = self.train_losses[-self.window:]
            early_avg  = np.mean(recent[:self.window//2])
            late_avg   = np.mean(recent[self.window//2:])
            if late_avg > early_avg * 0.99 and step > 200:
                alerts.append(f"⚠️  PLATEAU: loss not decreasing "
                               f"({early_avg:.3f} → {late_avg:.3f})")

        # Loss spike
        if len(self.train_losses) >= 3:
            avg_recent = np.mean(self.train_losses[-10:])
            if loss > avg_recent * 1.5 and step > 10:
                alerts.append(f"⚠️  LOSS SPIKE: {loss:.3f} vs avg {avg_recent:.3f}")

        self.alerts.extend(alerts)
        return alerts

    def report(self):
        if not self.steps:
            return
        n = len(self.train_losses)
        print(f"  Training Monitor Report ({n} steps)")
        print(f"  {'─'*55}")
        print(f"  Steps logged:      {self.steps[-1]}")
        print(f"  Current loss:      {self.train_losses[-1]:.4f}")
        print(f"  Initial loss:      {self.train_losses[0]:.4f}")
        print(f"  Total decrease:    {self.train_losses[0]-self.train_losses[-1]:.4f}")
        print(f"  Current grad norm: {self.grad_norms[-1]:.4f}")
        print(f"  Avg grad norm:     {np.mean(self.grad_norms):.4f}")
        print(f"  Max grad norm:     {np.max(self.grad_norms):.4f}")

        if self.val_losses:
            best_val = min(v for _, v in self.val_losses)
            last_val = self.val_losses[-1][1]
            print(f"  Best val loss:     {best_val:.4f}")
            print(f"  Last val loss:     {last_val:.4f}")
            train_val_gap = self.train_losses[-1] - last_val
            print(f"  Train-val gap:     {train_val_gap:+.4f}  "
                  f"({'overfitting risk' if train_val_gap < -0.2 else 'healthy'})")

        if self.alerts:
            print(f"\n  Alerts ({len(self.alerts)} total):")
            for alert in self.alerts[-5:]:
                print(f"    {alert}")
        else:
            print(f"  Alerts: None ✅")


# Simulate a training run with some anomalies
np.random.seed(1)
monitor = TrainingMonitor(window=20)

print("  Simulating training run (400 steps, injecting anomalies):")
print()

initial_loss = 10.0
for step in range(1, 401):
    # Simulate loss curve: fast initial drop then slower
    base_loss = initial_loss * math.exp(-0.008 * step) + 1.8
    noise     = np.random.randn() * 0.05

    # Inject anomalies at specific steps
    if step == 150:
        noise += 1.5    # loss spike (bad batch)
    if step == 250:
        noise -= 0.0    # normal

    loss      = max(0.1, base_loss + noise)
    grad_norm = abs(np.random.randn() * 0.3 + 0.8)

    # Simulate gradient explosion
    if step == 150:
        grad_norm = 12.5

    lr = 3e-4 * min(1.0, step / 50) * (0.9 + 0.1 * math.cos(math.pi * step / 400))

    val_loss = loss + 0.05 if step % 50 == 0 else None
    monitor.log(step, loss, grad_norm, lr, val_loss)

print()
monitor.report()
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: MFU and throughput estimation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Model FLOP Utilisation (MFU) and throughput")
print("━" * 65)
print()

def compute_mfu(n_params, tokens_per_sec, peak_flops_tflops):
    """
    MFU = actual_FLOPS / theoretical_peak_FLOPS.
    Karpathy's approximation: FLOPs per token ≈ 6 × N_params.
    (2 FLOPs/param forward + 4 FLOPs/param backward)
    """
    flops_per_token    = 6 * n_params
    actual_tflops      = tokens_per_sec * flops_per_token / 1e12
    mfu                = actual_tflops / peak_flops_tflops
    return actual_tflops, mfu

def estimate_training_time(n_params, total_tokens, tokens_per_sec):
    time_seconds = total_tokens / tokens_per_sec
    time_hours   = time_seconds / 3600
    time_days    = time_hours / 24
    return time_seconds, time_hours, time_days


print("  A100 80GB (BF16 Tensor Core peak: 312 TFLOPS)")
print()
print(f"  {'Model':>15} | {'Params':>8} | {'Tok/s':>8} | "
      f"{'TFLOPS':>8} | {'MFU':>8} | {'1T tok time'}")
print(f"  {'─'*70}")

scenarios = [
    ("GPT-2 (125M)",    125e6,  20000,   "batch=16, seq=1024, bf16"),
    ("GPT-2 (125M)",    125e6,   5000,   "batch=4,  seq=1024, fp32"),
    ("LLaMA-2 7B",      7e9,    1800,   "batch=1,  seq=4096, bf16+ckpt"),
    ("LLaMA-2 7B",      7e9,     800,   "batch=1,  seq=4096, fp32"),
    ("GPT-3 (175B)",    175e9,    40,   "batch=1,  seq=2048, bf16+ckpt"),
]

for name, params, tok_s, config in scenarios:
    actual_tf, mfu = compute_mfu(params, tok_s, 312.0)
    _, _, days = estimate_training_time(params, 1e12, tok_s)
    quality = "✅ good" if mfu > 0.35 else ("⚠️ fair" if mfu > 0.15 else "❌ poor")
    print(f"  {name:>15} | {params/1e9:>7.1f}B | {tok_s:>8,} | "
          f"{actual_tf:>8.1f} | {mfu:>7.1%} | {days:>7.0f} days")
    print(f"  {'':>15}   {config}")

print()
print("  MFU interpretation:")
print("    > 45%: excellent (Megatron-LM, optimised training)")
print("    30-45%: good (typical well-configured training)")
print("    15-30%: fair (room for optimisation)")
print("    < 15%:  poor (likely data bottleneck or config issue)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Overfit test (verifying gradient flow)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Overfit test: verifying gradient flow")
print("━" * 65)
print()

print("  The overfit test: can the model memorise a tiny dataset?")
print("  If YES: gradients are flowing correctly.")
print("  If NO:  there's a bug in the loss computation or backpropagation.")
print()
print("  Recipe:")
OVERFIT_TEST = """
  # 1. Take 1–10 training batches
  tiny_dataset = [next(iter(train_loader)) for _ in range(5)]

  # 2. Train on ONLY these batches for many steps
  model.train()
  for step in range(500):
      for batch in tiny_dataset:
          input_ids = batch['input_ids'].to(device)
          labels    = batch['labels'].to(device)
          logits = model(input_ids)
          loss   = F.cross_entropy(logits.view(-1, V), labels.view(-1))
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

      if step % 50 == 0:
          print(f"Step {step}: loss = {loss.item():.4f}")

  # EXPECTED output (loss should drop to near 0):
  # Step 0:   loss = 10.4    ← random
  # Step 50:  loss = 5.2     ← learning
  # Step 100: loss = 2.1     ← strong
  # Step 200: loss = 0.8     ← memorising
  # Step 500: loss = 0.02    ← fully memorised ✅

  # If loss stays high (> 5.0 after 500 steps):
  #   → loss is not connected to parameters (check requires_grad)
  #   → learning rate is zero (check optimizer and scheduler)
  #   → gradient accumulation has a bug (check scaling)
  #   → data preprocessing is wrong (same batch every time, all-same tokens)
"""
print(OVERFIT_TEST)

# Simulate the overfit test numerically
print("  Simulated overfit test results:")
print()
print(f"  {'Step':>6} | {'Loss':>8} | {'Status'}")
print(f"  {'─'*35}")

loss = 10.4
for step in range(0, 501, 50):
    decay = math.exp(-0.015 * step)
    loss  = max(0.01, 10.4 * decay + np.random.randn() * 0.03 * decay)
    status = ("✅ memorised!" if loss < 0.1
               else "↓ learning" if loss < 3.0
               else "↓ decreasing")
    print(f"  {step:>6} | {loss:>8.4f} | {status}")
print()
print("  Loss dropped from 10.4 → ~0.01: gradient flow confirmed ✅")
print("  Model CAN memorise a small dataset → training pipeline is correct.")
print()

print("━" * 65)
print("  SINGLE-GPU TRAINING QUICK REFERENCE")
print("━" * 65)
print()
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │ Step                      │ Key decisions                   │")
print("  ├─────────────────────────────────────────────────────────────┤")
print("  │ Data prep                 │ Pack sequences, BPE tokenise    │")
print("  │ Precision                 │ BF16 weights + FP32 optimizer   │")
print("  │ Memory (big model)        │ Grad ckpt + 8-bit Adam          │")
print("  │ Batch size                │ Grad accumulation to target 512+│")
print("  │ LR schedule               │ Warmup 1-2% then cosine decay   │")
print("  │ Gradient stability        │ Clip at max_norm=1.0            │")
print("  │ Loss monitoring           │ Train + val every 1k steps      │")
print("  │ Health check              │ Overfit test before full run    │")
print("  │ Checkpoint                │ Every 1k-5k steps, keep last 3  │")
print("  │ MFU target                │ > 30% on A100 BF16              │")
print("  └─────────────────────────────────────────────────────────────┘")
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