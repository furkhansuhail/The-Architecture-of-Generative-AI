"""
Single-GPU LLM Training: Full Recipe
======================================

Training a real language model on a single GPU — from data preparation
through tokenisation, model construction, the training loop, gradient
accumulation, mixed precision, activation checkpointing, and evaluation —
without distributed training complexity. This is the canonical starting
point for anyone fine-tuning or training small LLMs on a single A100 or
consumer GPU. Every component is documented, every design decision is
explained, and the result is a complete, runnable training script.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Single-GPU LLM Training"
DISPLAY_NAME = "35 · Single-GPU Training"
ICON         = "🖥️"
SUBTITLE     = "Full Recipe: One GPU, Minimal Dependencies"


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

### The Single-GPU Starting Point

Most LLM training tutorials begin with multi-GPU distributed training,
which adds significant complexity before the core training loop is even
established. Starting with a single-GPU recipe offers:

    •   The complete training loop without distributed boilerplate
    •   Directly applicable to fine-tuning scenarios (LoRA on 1×A100)
    •   A debuggable environment (single process, standard Python debugger)
    •   A template that scales to multi-GPU by adding DDP/FSDP wrappers

The techniques in this module apply directly to:
    •   Fine-tuning a 7B model on a single A100 80GB (with LoRA or QLoRA)
    •   Pre-training a small model (125M–1B) from scratch
    •   Instruction-tuning on domain-specific data
    •   Continued pre-training on new domains


### Memory Budget on a Single GPU

Before writing any code, calculate whether your model fits. On an A100 80GB:

    Memory budget:  80 GB
    System reserve: ~2 GB (CUDA context, PyTorch allocator)
    Available:      ~78 GB

    Model weights (fp16/bf16):  P × 2 bytes
    Gradients (fp16):           P × 2 bytes  (same as weights)
    FP32 master weights:        P × 4 bytes
    AdamW moments (m, v):       P × 4 bytes × 2 = P × 8 bytes
    ─────────────────────────────────────────────────────────────
    Total model state:          P × 16 bytes

    For a 7B model (P = 7×10⁹):
        Model state: 7B × 16 = 112 GB → does NOT fit on A100 80GB!

    Solutions:
        LoRA: only trains adapter params (~0.5% of P)
              model state ≈ 2P + 0.005P × 16 = 14.08 GB ✓
        QLoRA: 4-bit base model + LoRA adapters
              model state ≈ 0.5P + small_lora = ~3.8 GB ✓

    For training from scratch (P ≤ 4B):
        4B × 16 = 64 GB ← fits with careful memory management ✓

    Activations (not counted above):
        With activation checkpointing: ~B × T × d × √N bytes
        Without: B × T × d × N bytes (much larger, often infeasible)


### The Complete Single-GPU Recipe

A production single-GPU training loop has the following components:

    **1. Data preparation:**
    •   Load text data from a JSONL file or Hugging Face datasets
    •   Tokenise with the model's tokeniser (BPE / SentencePiece)
    •   Pack sequences into fixed-length chunks (no padding waste)
    •   Create a DataLoader with prefetching and multiple workers

    **2. Model initialisation:**
    •   Instantiate the model architecture (from config or pretrained)
    •   Optionally load pretrained weights (for fine-tuning)
    •   Apply LoRA/QLoRA adapters if fine-tuning a large model
    •   Cast to bf16 (recommended over fp16 for training stability)
    •   Move to GPU with `.to(device)`

    **3. Optimiser and schedule:**
    •   AdamW with betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    •   Cosine LR schedule with linear warmup
    •   Gradient clipping at max_norm=1.0

    **4. Training loop:**
    •   Mixed precision with `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
    •   Gradient accumulation: K micro-batches → 1 optimizer step
    •   Gradient clipping after accumulation
    •   Logging every N steps
    •   Validation every M steps
    •   Checkpointing every K steps

    **5. Evaluation:**
    •   Per-token perplexity on held-out validation set
    •   Optional: sample generation to inspect quality


### Mixed Precision: BF16 vs FP16 on a Single GPU

Both bf16 and fp16 halve the memory compared to fp32. The choice matters:

    **BF16 (bfloat16):**
    •   Same exponent range as fp32 (8 bits exponent)
    •   Only 7 bits mantissa (less precision than fp16's 10 bits)
    •   Can represent very large/small numbers without overflow
    •   Does NOT need a GradScaler (no overflow risk)
    •   Available on: A100, H100, RTX 3090/4090, Apple M-series
    •   Recommended for: all modern GPU training

    **FP16 (half-precision):**
    •   Smaller exponent range (5 bits) → overflow is common during training
    •   Requires GradScaler to detect and rescale before overflow
    •   Available on: all CUDA GPUs (including older V100, T4)
    •   Use when BF16 is unavailable

    For single-GPU training on a modern GPU:
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    And NO GradScaler needed.

    For older GPUs that support only fp16:
        scaler = torch.cuda.amp.GradScaler()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


### Activation Checkpointing: The Memory-Speed Trade-off

During a forward pass with no activation checkpointing, every intermediate
activation is stored for the backward pass. For a Transformer with N layers:
    activation_memory ≈ B × T × d × N × (several tensors per layer)
    For B=4, T=2048, d=4096, N=32: ≈ 4 × 2048 × 4096 × 32 × 4 bytes ≈ 4 GB

With activation checkpointing (recomputation):
    Only checkpoint tensors at layer boundaries are stored.
    Intermediate activations are recomputed during backward pass.
    Memory: ≈ B × T × d × √N   (saves ~4–8× memory)
    Cost: ≈ 33% extra compute (one extra forward pass per layer)

For single-GPU training with a large model, activation checkpointing is
often mandatory. Without it, the model simply won't fit.

PyTorch interface:
    from torch.utils.checkpoint import checkpoint
    # In the forward pass:
    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(block, x)   # recompute during backward
        return x


### The Learning Rate: The Most Critical Hyperparameter

On a single GPU, getting the learning rate right is the most important
single decision. Rules of thumb:

    **Pre-training from scratch:**
        LR = 3e-4 for small models (125M–1B)
        LR = 1e-4 for medium models (1B–7B)
        Cosine decay to min_lr = 0.1 × max_lr
        Linear warmup: 1–2% of total steps (typical: 2000 steps)

    **Fine-tuning (full):**
        LR = 1e-5 to 5e-5 (10–30× lower than pre-training)
        Shorter warmup: 50–200 steps
        Fewer epochs: 1–3 (risk of catastrophic forgetting)

    **LoRA fine-tuning:**
        LR = 1e-4 to 5e-4 (higher than full FT, fewer params being trained)
        Warmup: 50–100 steps

    **The linear scaling rule for batch size:**
    If you increase effective batch size by 2× (via gradient accumulation),
    scale LR by √2 (conservative) or 2 (aggressive):
        base: batch=4, LR=3e-4
        4× accum: effective_batch=16, LR=6e-4 (√4 × 3e-4)


### Data Loading Efficiency on a Single GPU

The GPU should never wait for data. Common bottlenecks:
    •   `num_workers=0` (default): data loaded in main process → slow
    •   `pin_memory=False`: extra CPU→GPU copy step
    •   Not prefetching: GPU waits after each batch

Recommended DataLoader settings:
    DataLoader(
        dataset,
        batch_size       = batch_size,
        shuffle          = True,
        num_workers      = 4,         # background workers
        pin_memory       = True,      # faster GPU transfer
        prefetch_factor  = 2,         # 2 batches prefetched per worker
        persistent_workers = True,    # keep workers alive between epochs
        drop_last        = True,      # avoid variable-size last batch
    )

For token datasets (fixed-length sequences), `pin_memory=True` is particularly
important because the transfer of integer token tensors to GPU can be a
bottleneck at high training throughput.


### Gradient Accumulation: Simulating Large Batches

On a single GPU, the maximum micro-batch size is memory-limited (typically 1–8).
To achieve the large effective batch sizes needed for stable training:
    effective_batch = micro_batch × grad_accum_steps

For training a 7B model with LoRA:
    micro_batch = 2 (limited by GPU memory with LoRA activations)
    grad_accum  = 32 (accumulate 32 micro-batches)
    effective   = 64 sequences × T tokens ≈ 131K tokens/step

This is lower than the ideal 4M tokens/step but still trains well with
a correspondingly lower learning rate.

Critical: always divide the loss by grad_accum_steps before calling backward.
(See Module 17 for the full analysis.)


### Validation and Evaluation Protocol

Every training run needs a held-out validation set to detect:
    1.  Overfitting: train loss decreasing but val loss increasing
    2.  Training quality: absolute val perplexity as a quality measure
    3.  Best model selection: save checkpoint with lowest val loss

**Validation set sizing:**
    For language models: 5–10% of training data (or 10M tokens, whichever is smaller)
    For fine-tuning on small datasets: 10–20% of the fine-tuning data

**Validation loop:**
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch)
            total_loss += loss.item()
    model.train()
    val_ppl = math.exp(total_loss / n_val_batches)

Note: always use `model.eval()` during validation to disable dropout and
switch BatchNorm to inference mode. Don't forget `model.train()` afterward.


### Production Configuration: A Practical Checklist

Before starting a training run:
    ☐  Verify model fits in GPU memory (with activations and optimizer state)
    ☐  Set num_workers to 4+ for data loading
    ☐  Enable pin_memory=True in DataLoader
    ☐  Use bf16 autocast if on A100/H100 (no GradScaler needed)
    ☐  Enable activation checkpointing if needed
    ☐  Set gradient accumulation to reach target effective batch size
    ☐  Configure cosine LR schedule with warmup
    ☐  Set up checkpointing every N steps
    ☐  Enable experiment tracking (W&B or console)
    ☐  Create a validation split and monitor val_ppl
    ☐  Set torch.backends.cuda.matmul.allow_tf32 = True  (A100 speedup)
    ☐  Set torch.backends.cudnn.allow_tf32 = True
    ☐  Verify data pipeline is not the bottleneck (profile 10 steps)
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Single-GPU Training Configurations

| Scenario                      | Model size  | Technique           | GPU needed        | LR         |
|-------------------------------|-------------|---------------------|-------------------|------------|
| Scratch pre-training          | 125M        | Full, bf16          | 1× RTX 3090 24GB  | 6e-4       |
| Scratch pre-training          | 1B          | Full + act ckpt     | 1× A100 40GB      | 2e-4       |
| Scratch pre-training          | 3B          | Full + act ckpt     | 1× A100 80GB      | 1e-4       |
| Full fine-tuning              | 7B          | Full + act ckpt     | 1× A100 80GB      | 2e-5       |
| LoRA fine-tuning (r=16)       | 7B          | LoRA, bf16 base     | 1× RTX 3090 24GB  | 2e-4       |
| QLoRA fine-tuning             | 13B         | 4-bit base + LoRA   | 1× RTX 3090 24GB  | 2e-4       |
| QLoRA fine-tuning             | 70B         | 4-bit base + LoRA   | 1× A100 80GB      | 1e-4       |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Complete Single-GPU Training Script": {
        "description": "A full, production-quality single-GPU training loop: model, data, optimizer, mixed-precision, gradient accumulation, activation checkpointing, validation, and checkpointing — everything in one place.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
COMPLETE SINGLE-GPU TRAINING SCRIPT
================================================================================

A full training pipeline on a single GPU:
    1. Configuration dataclass (all hyperparameters in one place)
    2. Model with activation checkpointing support
    3. Token dataset with sequence packing
    4. Training loop with bf16 autocast, gradient accumulation, grad clipping
    5. Cosine LR schedule with linear warmup
    6. Periodic validation with perplexity computation
    7. Checkpointing with full training state
    8. Training metrics logging and reporting

This is the complete recipe for single-GPU LLM training.

================================================================================
"""

import math
import time
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """All hyperparameters and settings in one place."""

    # Model
    vocab_size:      int   = 512
    d_model:         int   = 128
    n_heads:         int   = 4
    n_layers:        int   = 4
    d_ff:            int   = 512       # d_model × 4
    max_seq_len:     int   = 64
    dropout:         float = 0.0

    # Training
    n_steps:         int   = 200
    micro_batch:     int   = 4
    grad_accum:      int   = 4        # effective_batch = micro_batch × grad_accum
    max_lr:          float = 3e-4
    min_lr:          float = 3e-5     # 0.1 × max_lr
    warmup_steps:    int   = 20
    weight_decay:    float = 0.1
    beta1:           float = 0.9
    beta2:           float = 0.95
    eps:             float = 1e-8
    grad_clip:       float = 1.0

    # Mixed precision
    use_amp:         bool  = True
    amp_dtype:       str   = "bfloat16"  # "bfloat16" or "float16"

    # Memory
    use_act_ckpt:    bool  = True     # activation checkpointing

    # Data
    n_train_tokens:  int   = 50_000
    n_val_tokens:    int   = 5_000

    # Logging / checkpointing
    log_every:       int   = 10
    val_every:       int   = 50
    save_every:      int   = 100
    checkpoint_dir:  str   = "/tmp/ckpts"

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum

    @property
    def amp_torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no mean-centering)."""
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: the activation used in LLaMA."""
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE-free sinusoidal positions."""
    def __init__(self, d: int, n_heads: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.qkv     = nn.Linear(d, 3 * d, bias=False)
        self.out     = nn.Linear(d, d, bias=False)
        self.dropout = dropout
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        h, dh   = self.n_heads, self.d_head
        qkv     = self.qkv(x).view(B, T, 3, h, dh).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)
        scale   = math.sqrt(dh)
        scores  = Q @ K.transpose(-2, -1) / scale
        scores  = scores.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0),
                                      float("-inf"))
        attn    = F.softmax(scores, dim=-1)
        attn    = F.dropout(attn, p=self.dropout, training=self.training)
        out     = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RMSNorm + SwiGLU (LLaMA-style)."""
    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        d  = cfg.d_model
        self.norm1 = RMSNorm(d)
        self.attn  = CausalSelfAttention(d, cfg.n_heads, cfg.max_seq_len, cfg.dropout)
        self.norm2 = RMSNorm(d)
        self.ffn   = SwiGLU(d, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LLM(nn.Module):
    """
    Compact LLaMA-style causal language model.
    Features: RMSNorm, SwiGLU, tied input/output embeddings.
    Supports activation checkpointing for memory efficiency.
    """
    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation with depth scaling (GPT-2 style)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        # Scale residual stream projections by 1/√(2N) for depth stability
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
            nn.init.normal_(block.ffn.down.weight, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    def forward(self, idx: torch.Tensor,
                 targets: Optional[torch.Tensor] = None) -> tuple:
        """
        idx:     (B, T) token indices
        targets: (B, T) target token indices (optional)
        Returns: (logits, loss) or (logits, None)
        """
        x = self.embed(idx)   # (B, T, d)

        if self.cfg.use_act_ckpt and self.training:
            # Activation checkpointing: recompute intermediate activations during backward
            # Saves ~4–8× memory at the cost of ~33% extra compute
            for block in self.blocks:
                x = gradient_checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        logits = self.head(self.norm(x))   # (B, T, vocab)

        if targets is not None:
            # Shift: predict token i+1 from position i
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.cfg.vocab_size),
                targets[:, 1:].reshape(-1),
            )
            return logits, loss

        return logits, None

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class PackedTokenDataset(Dataset):
    """
    Dataset that packs tokens into fixed-length sequences with no padding.
    All tokens are used; no sequence boundaries needed.

    In production: replace the token generation with actual tokenised data
    loaded from a file:
        tokens = np.memmap("tokens.bin", dtype=np.uint16, mode="r")
    """
    def __init__(self, n_tokens: int, vocab_size: int, seq_len: int, seed: int):
        torch.manual_seed(seed)
        # Synthetic data: random token IDs (replace with real tokenised text)
        self.tokens  = torch.randint(0, vocab_size, (n_tokens,), dtype=torch.long)
        self.seq_len = seq_len
        # Number of full sequences we can extract
        self.n_seqs  = (n_tokens - 1) // seq_len

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        # Return seq_len + 1 tokens so we can create input/target pairs
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk


def collate_packed(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate packed sequences into (input, target) tensors."""
    seqs    = torch.stack(batch)   # (B, T+1)
    inputs  = seqs[:, :-1]        # (B, T)
    targets = seqs[:, 1:]         # (B, T)  — shifted by 1
    return inputs, targets


# ─────────────────────────────────────────────────────────────────────────────
# 4. LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainingConfig) -> float:
    """
    Cosine LR schedule with linear warmup.

    0 → warmup_steps:    linear ramp from 0 to max_lr
    warmup → n_steps:    cosine decay from max_lr to min_lr
    """
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / cfg.warmup_steps
    if step >= cfg.n_steps:
        return cfg.min_lr
    # Cosine decay
    progress = (step - cfg.warmup_steps) / (cfg.n_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainingConfig) -> dict:
    """
    Complete single-GPU training loop.
    Returns training history dict.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype   = cfg.amp_torch_dtype
    use_amp = cfg.use_amp and device.type == "cuda"
    # On modern GPUs with bf16: no GradScaler needed (no overflow)
    # On old GPUs with fp16: would need scaler = torch.cuda.amp.GradScaler()
    scaler  = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

    # A100/H100 optimisation: allow TF32 for better throughput
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LLM(cfg).to(device)
    print(f"  Model: {model.n_params:,} parameters on {device}")
    print(f"  Activation checkpointing: {'✓' if cfg.use_act_ckpt else '✗'}")
    print(f"  Mixed precision: {'✓ ' + cfg.amp_dtype if use_amp else '✗ (fp32)'}")
    print(f"  Effective batch: {cfg.effective_batch} sequences × {cfg.max_seq_len} tokens"
          f" = {cfg.effective_batch * cfg.max_seq_len:,} tokens/step")
    print()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate weight decay and no-decay parameter groups
    # (biases and normalisation weights should not be decayed)
    decay_params    = [p for n, p in model.named_parameters()
                        if p.dim() >= 2 and "norm" not in n]
    no_decay_params = [p for n, p in model.named_parameters()
                        if p.dim() < 2 or "norm" in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr    = cfg.max_lr,
        betas = (cfg.beta1, cfg.beta2),
        eps   = cfg.eps,
        fused = (device.type == "cuda"),  # fused kernel for speed
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = PackedTokenDataset(cfg.n_train_tokens, cfg.vocab_size,
                                   cfg.max_seq_len, seed=0)
    val_ds   = PackedTokenDataset(cfg.n_val_tokens,   cfg.vocab_size,
                                   cfg.max_seq_len, seed=1)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.micro_batch, shuffle=True,
        collate_fn=collate_packed, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader   = DataLoader(
        val_ds, batch_size=cfg.micro_batch, shuffle=False,
        collate_fn=collate_packed, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── History ───────────────────────────────────────────────────────────────
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "grad_norm": []}
    best_val_loss = float("inf")
    t0     = time.perf_counter()
    n_tokens_total = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    train_iter  = iter(train_loader)
    global_step = 0
    accum_loss  = 0.0

    while global_step < cfg.n_steps:
        # LR update (manual, applied before each optimizer step)
        current_lr = get_lr(global_step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        # ── Gradient accumulation ─────────────────────────────────────────────
        for micro_step in range(cfg.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            n_tokens_total += x.numel()

            # Forward + loss under autocast (bf16 / fp16)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                _, loss = model(x, targets=y)
                loss    = loss / cfg.grad_accum   # ← IMPORTANT: normalise!

            # Backward (scaled if using fp16)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        # ── Gradient clipping + optimizer step ────────────────────────────────
        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        ).item()

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        global_step += 1
        step_loss    = accum_loss * cfg.grad_accum   # undo the normalisation for logging
        accum_loss   = 0.0

        # ── Logging ───────────────────────────────────────────────────────────
        if global_step % cfg.log_every == 0:
            elapsed  = time.perf_counter() - t0
            tok_ps   = n_tokens_total / elapsed
            history["step"].append(global_step)
            history["train_loss"].append(step_loss)
            history["lr"].append(current_lr)
            history["grad_norm"].append(grad_norm)
            print(f"  step {global_step:>5}/{cfg.n_steps}  "
                  f"loss={step_loss:.4f}  ppl={math.exp(step_loss):.1f}  "
                  f"lr={current_lr:.2e}  grad={grad_norm:.3f}  "
                  f"tok/s={tok_ps:,.0f}")

        # ── Validation ────────────────────────────────────────────────────────
        if global_step % cfg.val_every == 0:
            val_loss = evaluate(model, val_loader, device, dtype, use_amp)
            history["val_loss"].append((global_step, val_loss))
            val_ppl  = math.exp(val_loss)
            improved = "✓ new best!" if val_loss < best_val_loss else ""
            print(f"  >>> VAL step={global_step}: loss={val_loss:.4f}  "
                  f"ppl={val_ppl:.2f}  {improved}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            model.train()

    return history


def evaluate(model: nn.Module, val_loader: DataLoader,
              device: torch.device, dtype: torch.dtype,
              use_amp: bool) -> float:
    """Compute mean cross-entropy loss on the validation set."""
    model.eval()
    total_loss  = 0.0
    n_batches   = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                _, loss = model(x, targets=y)
            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = TrainingConfig(
        vocab_size   = 512,
        d_model      = 128,
        n_heads      = 4,
        n_layers     = 4,
        d_ff         = 512,
        max_seq_len  = 64,
        n_steps      = 120,
        micro_batch  = 4,
        grad_accum   = 4,
        max_lr       = 3e-4,
        min_lr       = 3e-5,
        warmup_steps = 12,
        use_amp      = True,
        use_act_ckpt = True,
        n_train_tokens = 40_000,
        n_val_tokens   = 5_000,
        log_every    = 20,
        val_every    = 60,
    )

    print("=" * 65)
    print("  SINGLE-GPU TRAINING — COMPLETE RECIPE")
    print(f"  device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 65)
    print()

    history = train(cfg)

    print()
    print("=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    if history["train_loss"]:
        print(f"  Initial train loss: {history['train_loss'][0]:.4f}")
        print(f"  Final train loss:   {history['train_loss'][-1]:.4f}")
        print(f"  Loss reduction:     {(1 - history['train_loss'][-1]/history['train_loss'][0])*100:.1f}%")
    if history["val_loss"]:
        best_val = min(v for _, v in history["val_loss"])
        print(f"  Best val loss:      {best_val:.4f}  (ppl={math.exp(best_val):.2f})")
    print()
    print("  ✓ This script is your template for single-GPU LLM training.")
    print("  To scale to multi-GPU: wrap model with DDP/FSDP (Module 18/24).")
''',
    },

    "Memory Profiling and Optimisation": {
        "description": "Profile GPU memory usage at each stage of a training step, identify the bottlenecks, and demonstrate the memory savings from each optimisation (bf16, activation checkpointing, gradient accumulation).",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
GPU MEMORY PROFILING AND OPTIMISATION
================================================================================

Profiles GPU memory at each stage of a training step and shows the
cumulative effect of memory optimisations:
    1. Baseline (fp32, no checkpointing)
    2. + bf16 mixed precision
    3. + activation checkpointing
    4. + gradient accumulation (smaller micro-batch)
    5. Final: maximum batch size achievable

Without an actual GPU, simulates memory using parameter counting.

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from dataclasses import dataclass


@dataclass
class MemSnapshot:
    """Memory usage at one point during training."""
    label:       str
    gpu_alloc:   float   # GB allocated
    gpu_reserved: float  # GB reserved (includes fragmentation)

    def __str__(self) -> str:
        return (f"{self.label:<40}: "
                f"alloc={self.gpu_alloc:.2f}GB  "
                f"reserved={self.gpu_reserved:.2f}GB")


def get_gpu_memory() -> tuple[float, float]:
    """Return (allocated, reserved) GPU memory in GB."""
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved()  / 1e9
        return alloc, reserved
    return 0.0, 0.0


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def snap(label: str) -> MemSnapshot:
    alloc, res = get_gpu_memory()
    return MemSnapshot(label, alloc, res)


# ── Configurable model ────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, d: int, d_ff: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(nn.Linear(d, d_ff, bias=False),
                                    nn.GELU(),
                                    nn.Linear(d_ff, d, bias=False))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ProfiledModel(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, d_ff, use_ckpt=False):
        super().__init__()
        self.embed   = nn.Embedding(vocab, d)
        self.blocks  = nn.ModuleList([Block(d, d_ff, heads) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d)
        self.head    = nn.Linear(d, vocab, bias=False)
        self.use_ckpt = use_ckpt

    def forward(self, x, y=None):
        h = self.embed(x)
        for block in self.blocks:
            if self.use_ckpt and self.training:
                h = grad_ckpt(block, h, use_reentrant=False)
            else:
                h = block(h)
        logits = self.head(self.norm(h))
        if y is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                    y[:, 1:].reshape(-1))
            return loss
        return logits


def profile_training_step(
    vocab=512, d=256, n_layers=4, heads=4, d_ff=1024,
    batch=4, seq_len=128,
    use_bf16=False, use_ckpt=False,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """
    Profile memory at each stage of one training step.
    Returns dict of {stage: (alloc_GB, reserved_GB)}.
    """
    device = torch.device(device_str)
    dtype  = torch.bfloat16 if use_bf16 and device_str == "cuda" else torch.float32

    reset_gpu_memory()
    snapshots = []

    # Stage 1: model allocation
    model = ProfiledModel(vocab, d, n_layers, heads, d_ff, use_ckpt=use_ckpt)
    model = model.to(device).to(dtype if use_bf16 else torch.float32)
    snapshots.append(snap("1. Model loaded"))

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Note: optimizer state not yet allocated (created on first step)

    # Stage 2: first forward pass (gradients allocated)
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    y = torch.randint(0, vocab, (batch, seq_len), device=device)

    use_amp = use_bf16 and device_str == "cuda"
    with torch.autocast(device_type=device_str, dtype=torch.bfloat16, enabled=use_amp):
        loss = model(x, y)
    snapshots.append(snap("2. After forward (activations in memory)"))

    # Stage 3: backward pass (gradients allocated)
    loss.backward()
    snapshots.append(snap("3. After backward (grads allocated)"))

    # Stage 4: optimizer step (optimizer state allocated)
    opt.step()
    opt.zero_grad(set_to_none=True)
    snapshots.append(snap("4. After optimizer step (moments allocated)"))

    peak_alloc, peak_res = get_gpu_memory()
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_res   = torch.cuda.max_memory_reserved()  / 1e9

    n_params = sum(p.numel() for p in model.parameters())

    # Cleanup
    del model, opt, x, y, loss
    reset_gpu_memory()

    return {
        "config": f"bf16={use_bf16}, ckpt={use_ckpt}, B={batch}",
        "n_params": n_params,
        "snapshots": snapshots,
        "peak_alloc_gb": peak_alloc,
        "peak_res_gb":   peak_res,
    }


# ── Theoretical memory estimation ─────────────────────────────────────────────

def theoretical_memory(n_params, batch, seq_len, d, n_layers,
                         use_bf16=False, use_ckpt=False) -> dict:
    """
    Theoretical memory breakdown without running the model.
    Useful for pre-planning before running out of GPU memory.
    """
    p_bytes  = 2 if use_bf16 else 4   # bytes per model param

    weights   = n_params * p_bytes
    gradients = n_params * p_bytes   # same as weights
    optim     = n_params * 8         # fp32 m + v  (always fp32)
    fp32_copy = n_params * 4         # fp32 master weights (if using fp16)

    # Activations (rough estimate)
    if use_ckpt:
        # Checkpointed: only layer boundaries stored
        act_per_tok = d * p_bytes    # one d-dim vector per token per boundary
        acts = batch * seq_len * act_per_tok * n_layers   # layer outputs only
    else:
        # Full activations: attention (T×T per head), FFN hiddens, etc.
        # Rough approximation: ~10× the embedding size per layer
        act_per_tok = d * 10 * p_bytes
        acts = batch * seq_len * act_per_tok * n_layers

    total = weights + gradients + optim + acts + (fp32_copy if use_bf16 else 0)

    return {
        "weights_gb":    weights / 1e9,
        "gradients_gb":  gradients / 1e9,
        "optimizer_gb":  optim / 1e9,
        "activations_gb": acts / 1e9,
        "fp32_copy_gb":  (fp32_copy / 1e9 if use_bf16 else 0),
        "total_gb":      total / 1e9,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    VOCAB, D, LAYERS, HEADS, D_FF = 512, 256, 4, 4, 1024
    BATCH, SEQ = 4, 64
    N_PARAMS = (VOCAB * D + LAYERS * (D * D * 4 + D * D_FF * 2) + D + D) // 1

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(f"  GPU MEMORY PROFILING  (device: {device_str})")
    print(f"  vocab={VOCAB}, d={D}, layers={LAYERS}, B={BATCH}, T={SEQ}")
    print("=" * 65)
    print()

    # Theoretical analysis
    print("  THEORETICAL MEMORY BREAKDOWN (estimated, no model run):")
    print()
    configs = [
        ("fp32, no ckpt",  False, False, BATCH),
        ("bf16, no ckpt",  True,  False, BATCH),
        ("bf16 + ckpt",    True,  True,  BATCH),
        ("bf16 + ckpt B=1",True,  True,  1),
    ]

    print(f"  {'Config':<22}  {'Weights':>8}  {'Grads':>7}  {'Optim':>7}  "
          f"{'Acts':>8}  {'Total':>8}  {'Fits 24GB?':>12}")
    print(f"  {'':─<22}  {'':─>8}  {'':─>7}  {'':─>7}  "
          f"{'':─>8}  {'':─>8}  {'':─>12}")

    # Estimate N params roughly
    est_params = VOCAB * D + LAYERS * (4 * D * D + 2 * D * D_FF + 4 * D) + D + D * VOCAB
    for label, bf16, ckpt, batch in configs:
        mem = theoretical_memory(est_params, batch, SEQ, D, LAYERS, bf16, ckpt)
        fits = "✓" if mem["total_gb"] < 24 else "✗"
        print(f"  {label:<22}  {mem['weights_gb']:>8.2f}  {mem['gradients_gb']:>7.2f}  "
              f"{mem['optimizer_gb']:>7.2f}  {mem['activations_gb']:>8.2f}  "
              f"{mem['total_gb']:>8.2f}  {fits:>12}")

    print()
    print("  KEY: activations are the dominant variable component (scales with B×T×d×N).")
    print("  Activation checkpointing is the most important memory optimisation.")
    print()

    # Live profiling (only meaningful with CUDA)
    if device_str == "cuda":
        print("  LIVE GPU MEMORY PROFILING:")
        print()
        for use_bf16, use_ckpt, b in [(False, False, BATCH), (True, False, BATCH),
                                        (True, True, BATCH), (True, True, 1)]:
            result = profile_training_step(VOCAB, D, LAYERS, HEADS, D_FF,
                                            b, SEQ, use_bf16, use_ckpt, device_str)
            print(f"  Config: {result['config']}")
            for snap_s in result["snapshots"]:
                print(f"    {snap_s}")
            print(f"    Peak: {result['peak_alloc_gb']:.2f} GB")
            print()
    else:
        print("  (Live GPU profiling skipped — no CUDA device)")
        print("  On a real GPU, each configuration would show different peak memory,")
        print("  demonstrating the ~4-8× savings from activation checkpointing.")
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
    #     from llm_training.visuals.singlegpu_training import (
    #         SINGLEGPU_VISUAL_HTML,
    #         SINGLEGPU_VISUAL_HEIGHT,
    #     )
    #     visual_html   = SINGLEGPU_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = SINGLEGPU_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[35_singlegpu_llm_training.py] Could not load visual: {e}",
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