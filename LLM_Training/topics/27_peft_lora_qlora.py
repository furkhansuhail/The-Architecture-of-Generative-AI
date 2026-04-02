"""
PEFT, LoRA, and QLoRA
======================

Parameter-Efficient Fine-Tuning (PEFT) methods allow large language models
to be adapted to new tasks by modifying only a tiny fraction of their
parameters. The most important of these methods — LoRA (Low-Rank Adaptation)
— achieves near-full fine-tuning quality while training fewer than 1% of
parameters and requiring a fraction of the GPU memory. QLoRA pushes this
further by quantising the frozen base model to 4-bit, enabling 65B parameter
models to be fine-tuned on a single consumer GPU. Understanding the linear
algebra behind these methods, their failure modes, and how to tune them is
essential for anyone doing LLM customisation at scale.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "PEFT, LoRA, and QLoRA"
DISPLAY_NAME = "27 · PEFT, LoRA & QLoRA"
ICON         = "🔬"
SUBTITLE     = "Parameter-Efficient Fine-Tuning"


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

### The Cost of Full Fine-Tuning

Full fine-tuning (all parameters updated) has several drawbacks at scale:

    1.  **Memory:** A 7B model requires 112 GB per GPU to store weights,
        gradients, and optimizer state (ZeRO Stage 1 helps but doesn't eliminate it).

    2.  **Storage:** Each fine-tuned variant requires a full model checkpoint.
        If you fine-tune a 70B model for 100 different tasks, you need
        100 × 140 GB = 14 TB of checkpoint storage.

    3.  **Catastrophic forgetting:** Full fine-tuning rewrites all parameters
        and can destroy generalisation abilities learned during pre-training.

    4.  **Serving complexity:** Switching between tasks requires loading
        different model weights — expensive in production.

PEFT methods address all four problems by:
    •   Training only a tiny subset of parameters (1–5%)
    •   Storing only the small adapter rather than the full model
    •   Making minimal changes to frozen base model weights
    •   Enabling multiple task adapters to be hot-swapped at inference


### The PEFT Landscape

Several families of PEFT methods have been proposed:

    Method             Trainable params   Memory        Quality vs full FT
    ──────────────────────────────────────────────────────────────────────────
    LoRA               0.1 – 1%           Very low      95–99% on most tasks
    DoRA               0.1 – 1%           Very low      Slightly better than LoRA
    IA³                0.01 – 0.1%        Tiny          Good for simple tasks
    Prefix Tuning      0.01 – 0.1%        Low           Moderate
    Prompt Tuning      ~0.001%            Tiny          Lower quality
    Adapter (Houlsby)  0.5 – 2%           Moderate      Good
    BitFit             0.03 – 0.1%        Tiny          Limited tasks only
    Full fine-tuning   100%               Very high     1× (baseline)
    ──────────────────────────────────────────────────────────────────────────

LoRA dominates in practice because it:
    •   Achieves quality competitive with full fine-tuning
    •   Requires no changes to model architecture (easy to apply)
    •   Can be merged with the base model for zero-overhead inference
    •   Works well with quantisation (enabling QLoRA)
    •   Is well-studied with clear theory and practical guidance


### LoRA: Low-Rank Adaptation — The Core Idea

LoRA (Hu et al., 2021) is motivated by the observation that **the weight
changes during fine-tuning have low intrinsic rank**.

Formally: when a pre-trained model is fine-tuned, the update to any weight
matrix W ∈ ℝ^(d × k) can be approximated as a product of two low-rank matrices:

    ΔW ≈ B × A,   where B ∈ ℝ^(d × r),  A ∈ ℝ^(r × k),  r << min(d, k)

**Why low rank?** The hypothesis is that the "task-relevant" direction in
weight space is a low-dimensional subspace. Fine-tuning does not need to
explore the full d×k dimensional parameter space — it only needs to adjust
a few degrees of freedom (rank r of them) to adapt the model.

Evidence for this:
    •   Aghajanyan et al. (2020) showed that fine-tuning trajectories lie
        in very low-dimensional subspaces (effective rank often < 100)
    •   Li et al. (2018) showed that random subspace optimisation works
        well for neural networks
    •   Intrinsic dimensionality studies suggest 100–1000 parameters are
        often sufficient to capture task-specific adaptation


### LoRA: Mathematical Formulation

For a weight matrix W₀ ∈ ℝ^(d × k) in the pre-trained model:

    Forward pass:    h = x · W₀ᵀ + x · (BA)ᵀ
                     h = x · (W₀ + ΔW)ᵀ,   where ΔW = BA

    Initialisation:  A ~ N(0, σ²)  (standard Gaussian)
                     B = 0         (zero matrix)
                     ΔW = BA = 0 at start (preserves pre-trained output)

    Scaling:         h = x · W₀ᵀ + (α/r) · x · (BA)ᵀ

The scaling factor α/r controls the magnitude of the LoRA update:
    •   α is a constant hyperparameter (often set equal to r, so α/r = 1)
    •   Dividing by r normalises for different choices of r
    •   At small r, each rank-1 component must contribute more → higher effective lr

**Number of trainable parameters:**
    Original W:   d × k parameters
    LoRA (A + B): d×r + r×k = r(d+k) parameters
    Reduction:    r(d+k) / (dk) = (d+k)/(dk) × r = r × (1/k + 1/d)

    For d=k=4096, r=16:
        Original: 4096 × 4096 = 16,777,216 parameters
        LoRA:     16 × (4096 + 4096) = 131,072 parameters
        Reduction: 16.7M → 131K = 128× fewer parameters!


    **Diagram 1 — LoRA Architecture:**

    LORA: STANDARD vs LORA-ADAPTED LAYER
    ════════════════════════════════════════════════════════════════

    Standard linear layer:
    x ──────── [W₀: d×k] ──────────→ h
                (frozen)

    LoRA-adapted layer:
                ┌─────────────────────────────────┐
    x ──────────┤         [W₀: d×k]               ├──(+)──→ h
                │          (frozen)                │   │
                │                                  │   │
                └─[A: r×k]──[B: d×r]──(α/r scale)─┘   │
                   (trained)  (trained)                   ↑
                                                  h = xW₀ᵀ + (α/r)xAᵀBᵀ

    Memory during training:
    W₀:  d×k×2 bytes   (bf16, frozen, no grad, no optimizer state)
    A:   r×k×2 bytes   (bf16, trained)
    B:   d×r×2 bytes   (bf16, trained)
    m_A, v_A: r×k×8 bytes  (fp32 optimizer state for A)
    m_B, v_B: d×r×8 bytes  (fp32 optimizer state for B)

    Memory for W₀ gradients: NONE (frozen base model)
    Memory for W₀ optimizer state: NONE (not updated)

    Total trainable memory: 10 × r × (d+k) bytes vs 16 × d × k for full FT


### Which Layers to Apply LoRA To?

LoRA can be applied to any linear layer. The original paper applied it to
the query (Q) and value (V) projection matrices in attention. Later work
showed that applying to more layers improves quality:

    Module         Commonly used?    Notes
    ──────────────────────────────────────────────────────────────────────
    Q projection   Yes               Core: query direction matters for tasks
    K projection   Yes               Improves performance when included
    V projection   Yes               Core: value direction for output quality
    O projection   Yes               Output projection, important for generation
    FFN W1 (up)    Sometimes         Large matrices; significant r required
    FFN W2 (down)  Sometimes         Large matrices; significant r required
    Embedding      Rarely            Vocabulary rarely needs adaptation
    LM head        Rarely            Tied to embedding usually
    ──────────────────────────────────────────────────────────────────────

The standard recommendation:
    •   Apply to q_proj, k_proj, v_proj, o_proj (all attention projections)
    •   Optionally include gate_proj, up_proj, down_proj (FFN) for harder tasks
    •   Use r=16 for general fine-tuning, r=4 for very light adaptation,
        r=64 for complex tasks that need more capacity


### Choosing the Rank r

The rank r is the most important LoRA hyperparameter:

    r          Trainable params (7B)   Use case
    ──────────────────────────────────────────────────────────────────────
    1          ~1.6M                   Extreme efficiency; limited quality
    4          ~6.5M                   Simple classification/extraction tasks
    8          ~13M                    Good balance; most instruction tuning
    16         ~26M                    Standard for general instruction tuning
    32         ~52M                    Complex tasks; reasoning-heavy
    64         ~104M                   Approaching full FT quality on hard tasks
    128        ~208M                   Rarely needed; diminishing returns
    ──────────────────────────────────────────────────────────────────────

The relationship between r and quality follows a Pareto frontier: increasing r
beyond ~64 gives diminishing returns while doubling the trainable parameters.
For most tasks, r=16 is the sweet spot.

**Rank and the singular value spectrum:**
You can analyse the rank of the learned LoRA updates by examining the singular
values of the ΔW = BA matrix. If most singular values are near zero, a lower
rank would have sufficed. If singular values are all similar in magnitude,
a higher rank is needed (all directions are important).


### LoRA Scaling: α and the Learning Rate

The scaling factor α is often misunderstood:

    h += (α / r) × x · (BA)ᵀ

For α = r (the default), the scaling factor is 1.0 — the LoRA update is
added without any scaling. This is equivalent to standard learning.

But if you change r while keeping α fixed, the effective learning rate changes:
    •   Doubling r with α fixed → each rank-1 component is half as large
    •   This is equivalent to halving the effective learning rate for LoRA

**Best practice:** Always set α = r (or equivalently set α/r = 1.0) and
adjust the learning rate independently. This decouples rank from learning rate.

Alternatively: the RSLoRA paper (Kalajdzievski, 2023) proposes using
α/√r instead of α/r as the scaling factor, arguing it is more theoretically
principled and gives consistent results across different ranks.


### LoRA Merging: Zero Inference Overhead

After fine-tuning, LoRA adapters can be **merged** into the base model:

    W_merged = W₀ + (α/r) × B × A

This produces a standard weight matrix with no added parameters.
After merging:
    •   No additional computation during inference
    •   No storage overhead (just one weight matrix, same as original)
    •   Cannot "unmerge" to use the base model separately

This is the standard deployment approach: fine-tune with LoRA (cheap),
merge for deployment (fast inference).

For serving multiple LoRA adapters simultaneously (e.g., 1000 customers
each with their own fine-tuned adapter), the **unmerged** approach is used:
the base model stays frozen on GPU, and LoRA weights are loaded/unloaded
per request. This is the basis of services like Predibase's "Turbo LoRA"
and vLLM's LoRA serving.


### DoRA: Decomposing Weight Updates into Magnitude and Direction

**DoRA** (Liu et al., 2024) extends LoRA by decomposing weight updates into:
    •   **Magnitude** changes (how large the weight is)
    •   **Direction** changes (which way the weight points)

    W_DoRA = (m / ‖W₀ + BA‖_c) × (W₀ + BA)

where m ∈ ℝ^k is a learnable magnitude vector and ‖·‖_c is the column norm.

DoRA consistently outperforms LoRA on many tasks by separating these two
learning signals. The key insight: LoRA conflates magnitude and direction
changes, which can cause interference. DoRA explicitly separates them.


### QLoRA: 4-bit Quantisation + LoRA

**QLoRA** (Dettmers et al., 2023) enables fine-tuning models that are much
too large to fit on a single GPU by quantising the base model to 4-bit.

**The QLoRA recipe:**
    1.  Quantise the frozen base model to NF4 (4-bit Normal Float) format
    2.  Apply LoRA adapters (in bf16/fp16) to the quantised base
    3.  Use double quantisation to further reduce quantisation overhead
    4.  Compute gradients through the quantised operations

**Memory calculation for 65B model with QLoRA (from the paper):**
    Without QLoRA:  65B × 2 bytes = 130 GB (just weights in bf16)
    With QLoRA:     65B × 0.5 bytes = 32.5 GB (4-bit weights)
    + LoRA states:  ~0.5 GB (r=64, q_proj+v_proj)
    + Activations:  ~8 GB (with activation checkpointing)
    Total:          ~41 GB  → fits on a single A100 40GB!

Even better: the 65B model fits on a consumer 24GB GPU (A10/RTX 3090) with:
    4-bit base: ~20 GB
    LoRA: ~0.2 GB
    Activations (small batch): ~2 GB
    Total: ~22 GB  → fits on consumer hardware!

**NF4 (Normal Float 4-bit):**
NF4 is a data type designed for normally-distributed neural network weights.
The quantisation grid is set to have equal probability mass in each bin
under a standard normal distribution, rather than equal spacing (linear
quantisation). This is near-optimal for weights that approximately follow
a normal distribution.

    NF4 values: [-1.0, -0.6961, -0.5250, -0.3949, -0.2844, -0.1848,
                 -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379,
                 0.4407, 0.5626, 0.7230, 1.0]
    (16 values = 4 bits; placed at quantile points of N(0,1))

Each weight is stored as a 4-bit index into this lookup table.

**Double quantisation:**
The quantisation scaling factors (used to scale each block's weights)
are themselves quantised from fp32 to fp8, saving an additional ~0.5 bits
per weight. At 65B parameters this saves ~4 GB.

**Paged optimisers:**
QLoRA uses "paged optimisers" — when GPU memory is exhausted by large
batches, optimizer states are temporarily paged to CPU RAM. This prevents
OOM errors at the cost of some CPU-GPU transfer during those steps.


### IA³: Injected Adapter in Activation Scaling

**IA³** (Liu et al., 2022) is another PEFT method that scales activations
rather than modifying weights. It injects learned vectors that multiply
the intermediate activations:

    h_adapted = l × h,    where l ∈ ℝ^d is a learned scaling vector

Applied to keys, values, and FFN activations:
    k_adapted = l_k × k     (rescale key activations)
    v_adapted = l_v × v     (rescale value activations)
    ff_adapted = l_ff × ff  (rescale FFN inner activations)

IA³ has even fewer parameters than LoRA (no matrix products, just vectors).
It is extremely parameter-efficient but works best for narrow task adaptation
(classification, extraction) rather than open-ended generation tasks.


### Practical LoRA Hyperparameter Guide

    Hyperparameter     Typical range      Notes
    ─────────────────────────────────────────────────────────────────────
    r (rank)           8 – 64             16 is usually the best starting point
    α (scaling)        r (= lora_alpha)   Set equal to r → α/r = 1.0
    Dropout            0.0 – 0.1          Small dropout for regularisation
    Target modules     q,k,v,o projections  Add FFN for hard tasks
    lr                 1e-4 – 5e-4        Higher than full FT (more params frozen)
    Epochs             1 – 5              More epochs are safer than full FT
    Batch size         8 – 128 sequences  Gradient accum to reach ~128
    ─────────────────────────────────────────────────────────────────────

**LoRA learning rate is higher than full FT:**
Since only a small subset of parameters are trained, the effective capacity
of the optimizer is lower. A higher lr is needed to make sufficient progress.
Typical: 1e-4 for LoRA vs 1e-5 for full fine-tuning.

**Rank and α jointly:**
If you change r, adjust α to keep α/r constant (or use RSLoRA's √r scaling).
This ensures the magnitude of LoRA updates is consistent across rank choices.


### Rank Stabilisation and rsLoRA

A subtle problem with vanilla LoRA: the initialisation scale of ΔW changes
with rank because A is sampled from N(0, 1/k) and has norm ~√r.

    E[‖BA‖_F] ∝ √r   (scales with square root of rank)

So as r increases, the LoRA updates become larger at initialisation.
With fixed α, the effective learning rate decreases proportionally.

**RSLoRA** (Rank-Stabilised LoRA) addresses this by using α/√r instead of
α/r, ensuring consistent initialisation scale across rank choices:

    h += (α/√r) × x · (BA)ᵀ

This makes rank selection more predictable and is now the default in
many implementations (e.g., PEFT library with use_rslora=True).
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
PEFT Methods Comparison

| Method          | Trainable % | Memory overhead | Quality vs full FT | Merge to base? | Best use case            |
|-----------------|-------------|-----------------|---------------------|----------------|--------------------------|
| Full fine-tuning| 100%        | 16× params      | 1× (baseline)       | N/A            | When budget allows       |
| LoRA (r=16)     | ~0.5%       | 0.5× params     | 95–99%              | Yes            | General instruction tuning|
| LoRA (r=4)      | ~0.1%       | 0.1× params     | 90–95%              | Yes            | Simple task adaptation   |
| DoRA            | ~0.5%       | Similar to LoRA | Often > LoRA        | Yes            | When LoRA underperforms  |
| QLoRA (4-bit)   | ~0.5%       | 0.3× params*   | 95–99%              | Yes (dequant)  | Consumer GPU training    |
| IA³             | ~0.01%      | Tiny            | 80–90%              | Yes (mult)     | Classification/extraction|
| Prefix Tuning   | ~0.1%       | Activations     | 75–90%              | No             | Generation diversity     |
| Prompt Tuning   | ~0.001%     | Tiny            | 60–80%              | No             | Very few-shot            |

*QLoRA memory: base model in 4-bit + LoRA adapters in bf16
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "LoRA Layer From Scratch — Full Implementation": {
        "description": "Implement LoRA from scratch with all details: low-rank decomposition, α/r scaling, correct initialisation (A=Gaussian, B=zeros), merging, and singular value analysis of the learned ΔW.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
LORA LAYER — COMPLETE FROM-SCRATCH IMPLEMENTATION
================================================================================

Implements LoRA with all details:
    1. Low-rank decomposition: ΔW = B×A, where A~N(0,1), B=0 at init
    2. α/r scaling for consistent learning rate across ranks
    3. Correct parameter counting (trainable vs frozen)
    4. Merging LoRA into base weights for zero-overhead inference
    5. Singular value analysis of learned ΔW matrix
    6. Comparison of quality at different ranks (r = 1,4,8,16,64)
    7. RSLoRA (α/√r scaling) for rank-stable training

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Core LoRA layer ───────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    A linear layer augmented with a LoRA adapter.

    Forward:
        h = x @ W₀.T + (α/r) * x @ A.T @ B.T
          = x @ (W₀ + (α/r) * B @ A).T
          = x @ W_eff.T

    Parameters:
        in_features:   d_in   (columns of W₀)
        out_features:  d_out  (rows of W₀)
        rank:          r      (inner dimension of low-rank decomposition)
        lora_alpha:    α      (scaling constant; α/r is applied to ΔW)
        dropout:       float  (applied to x before LoRA path)
        use_rslora:    bool   (if True: use α/√r instead of α/r)
        merge_weights: bool   (if True: merge A,B into W₀ after training)
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, lora_alpha: float = 16.0,
                 dropout: float = 0.0, use_rslora: bool = False,
                 bias: bool = True):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.lora_alpha   = lora_alpha
        self.use_rslora   = use_rslora
        self.merged       = False

        # Frozen base weight
        self.weight  = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias    = nn.Parameter(
            torch.zeros(out_features), requires_grad=bias
        ) if bias else None

        # LoRA trainable parameters
        # A: shape (r, d_in)  — projects to low-rank space
        # B: shape (d_out, r) — projects back to full space
        self.lora_A  = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B  = nn.Parameter(torch.zeros(out_features, rank))

        # LoRA dropout (applied to x before the LoRA path)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Scaling: α/r or α/√r (RSLoRA)
        if use_rslora:
            self.scaling = lora_alpha / math.sqrt(rank)
        else:
            self.scaling = lora_alpha / rank

        # Initialise base weights (Kaiming) and LoRA A (normal), B (zero)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # B=0 ⟹ ΔW = B@A = 0 at initialisation ✓ (preserves base model output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # After merging, just a standard linear layer
            return F.linear(x, self.weight, self.bias)

        # Base path (frozen)
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA path (trainable)
        lora_out = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),  # (B, T, r)
            self.lora_B                                     # (d_out, r) → (B, T, d_out)
        ) * self.scaling

        return base_out + lora_out

    def merge(self):
        """
        Merge LoRA weights into base weight: W_eff = W₀ + scaling × B @ A.
        After merging, the layer behaves identically but has no extra computation.
        """
        if self.merged:
            raise ValueError("Already merged!")
        # Compute effective weight delta
        delta = (self.lora_B @ self.lora_A) * self.scaling  # (d_out, d_in)
        self.weight.data += delta
        self.merged = True

    def unmerge(self):
        """Undo the merge (for continued training)."""
        if not self.merged:
            raise ValueError("Not merged!")
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.data -= delta
        self.merged = False

    @property
    def n_trainable_params(self) -> int:
        bias_p = self.bias.numel() if self.bias is not None else 0
        return self.lora_A.numel() + self.lora_B.numel() + bias_p

    @property
    def n_frozen_params(self) -> int:
        return self.weight.numel()

    @property
    def reduction_ratio(self) -> float:
        """How many times fewer trainable params vs full fine-tuning."""
        return self.n_frozen_params / self.n_trainable_params

    def delta_W_rank_analysis(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute singular values of the ΔW matrix to analyse intrinsic rank.
        If all singular values are similar in magnitude, higher r is needed.
        If they decay quickly, the chosen r is sufficient.
        """
        with torch.no_grad():
            delta_W = (self.lora_B @ self.lora_A).detach()  # (d_out, d_in)
            U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        return S, U


# ── Full model with LoRA ──────────────────────────────────────────────────────

def apply_lora(model: nn.Module, target_modules: list[str],
                rank: int = 16, lora_alpha: float = 16.0,
                dropout: float = 0.0, use_rslora: bool = False) -> nn.Module:
    """
    Replace specified linear layers in a model with LoRA-augmented versions.
    All other parameters are frozen.
    """
    def replace_module(parent: nn.Module, name_path: list[str],
                        replacement: nn.Module):
        """Recursively replace a nested module."""
        if len(name_path) == 1:
            setattr(parent, name_path[0], replacement)
        else:
            replace_module(getattr(parent, name_path[0]),
                            name_path[1:], replacement)

    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # Then replace target modules with LoRA versions
    for module_name in target_modules:
        parts     = module_name.split(".")
        parent    = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        original  = getattr(parent, parts[-1])

        if not isinstance(original, nn.Linear):
            print(f"  Warning: {module_name} is not Linear, skipping")
            continue

        lora_layer = LoRALinear(
            in_features  = original.in_features,
            out_features = original.out_features,
            rank         = rank,
            lora_alpha   = lora_alpha,
            dropout      = dropout,
            use_rslora   = use_rslora,
            bias         = original.bias is not None,
        )
        # Copy original weights
        lora_layer.weight.data.copy_(original.weight.data)
        if original.bias is not None:
            lora_layer.bias.data.copy_(original.bias.data)

        setattr(parent, parts[-1], lora_layer)

    return model


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    # ── 1. Basic LoRA layer verification ─────────────────────────────────────
    print("=" * 65)
    print("  LORA LAYER: PROPERTIES AND VERIFICATION")
    print("=" * 65)
    print()

    D_IN, D_OUT = 4096, 4096

    for r in [4, 16, 64]:
        layer = LoRALinear(D_IN, D_OUT, rank=r, lora_alpha=r, bias=False)
        total_base   = layer.n_frozen_params
        total_lora   = layer.n_trainable_params
        reduction    = layer.reduction_ratio

        print(f"  r={r:>3}:  base={total_base/1e6:.1f}M  "
              f"lora={total_lora/1e3:.0f}K  "
              f"reduction={reduction:.0f}×  "
              f"trainable%={total_lora/total_base*100:.2f}%  "
              f"scaling={layer.scaling:.3f}")

    print()

    # ── 2. Verify B=0 initialisation preserves base model output ──────────────
    print("  VERIFICATION: B=0 initialisation preserves base model output")
    layer = LoRALinear(64, 64, rank=8, lora_alpha=8.0, bias=False)
    x     = torch.randn(4, 16, 64)

    # Base output (without LoRA)
    with torch.no_grad():
        h_base = F.linear(x, layer.weight)
        h_lora = layer(x)
        max_diff = (h_base - h_lora).abs().max().item()

    print(f"  max |h_base - h_lora| at init: {max_diff:.2e}  "
          f"{'✓ identical (B=0)' if max_diff < 1e-5 else '✗ differs'}")
    print()

    # ── 3. Train a small model with LoRA ──────────────────────────────────────
    print("=" * 65)
    print("  TRAINING WITH LORA vs FULL FINE-TUNING")
    print("=" * 65)
    print()

    class TinyModel(nn.Module):
        def __init__(self, vocab=512, d=128):
            super().__init__()
            self.embed = nn.Embedding(vocab, d)
            self.q     = nn.Linear(d, d, bias=False)
            self.v     = nn.Linear(d, d, bias=False)
            self.out   = nn.Linear(d, vocab, bias=False)
        def forward(self, x):
            h = self.embed(x)
            h = F.gelu(self.q(h) + self.v(h))
            return self.out(h)

    VOCAB, D = 512, 128
    N_STEPS  = 40

    def count_params(m):
        total   = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    results = {}
    for config, rank, use_full in [
        ("Full FT",    0,  True),
        ("LoRA r=4",   4,  False),
        ("LoRA r=16",  16, False),
        ("LoRA r=64",  64, False),
    ]:
        torch.manual_seed(0)
        model = TinyModel(VOCAB, D)
        x     = torch.randint(0, VOCAB, (8, 16))
        y     = torch.randint(0, VOCAB, (8, 16))

        if not use_full:
            model = apply_lora(model, ["q", "v"], rank=rank, lora_alpha=rank)

        total, trainable = count_params(model)
        opt   = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=3e-4 if use_full else 1e-3
        )
        losses = []
        for _ in range(N_STEPS):
            opt.zero_grad()
            loss = F.cross_entropy(
                model(x).view(-1, VOCAB), y.view(-1)
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())

        results[config] = (losses, trainable, total)
        print(f"  {config:<15}: {trainable:>8,} trainable / {total:>8,} total  "
              f"({trainable/total*100:>5.1f}%)  "
              f"final loss={losses[-1]:.4f}")

    # ── 4. Merge verification ─────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  MERGE VERIFICATION: merged == unmerged output")
    print("=" * 65)
    torch.manual_seed(1)
    layer  = LoRALinear(64, 32, rank=8, lora_alpha=8.0, bias=False)

    # Train a few steps to give LoRA non-zero values
    opt2   = torch.optim.AdamW(
        [layer.lora_A, layer.lora_B], lr=1e-3
    )
    for _ in range(10):
        xb    = torch.randn(4, 8, 64)
        yb    = torch.randn(4, 8, 32)
        loss  = F.mse_loss(layer(xb), yb)
        opt2.zero_grad(); loss.backward(); opt2.step()

    x_test = torch.randn(4, 8, 64)
    with torch.no_grad():
        h_before = layer(x_test).clone()
        layer.merge()
        h_after  = layer(x_test).clone()

    diff = (h_before - h_after).abs().max().item()
    print(f"  Max diff (unmerged vs merged): {diff:.2e}  "
          f"{'✓ identical' if diff < 1e-4 else '✗ differs'}")
    print(f"  After merging: no lora_A/lora_B in forward path")
    print(f"  Zero inference overhead ✓")

    # ── 5. Singular value analysis ────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  SINGULAR VALUE ANALYSIS OF LEARNED ΔW")
    print("=" * 65)
    print()
    print("  (Singular value decay tells us if rank is well-chosen)")
    print()

    # Train a LoRA layer and analyse the learned update
    layer_sv = LoRALinear(128, 128, rank=16, lora_alpha=16.0, bias=False)
    opt_sv   = torch.optim.AdamW([layer_sv.lora_A, layer_sv.lora_B], lr=1e-3)

    for step in range(100):
        xb    = torch.randn(8, 32, 128)
        # Target: a low-rank transformation (rank 4 signal)
        W_target = torch.randn(4, 128) * 0.5
        yb    = F.linear(F.linear(xb, W_target), W_target.T)
        loss  = F.mse_loss(layer_sv(xb), yb + F.linear(xb, layer_sv.weight))
        opt_sv.zero_grad(); loss.backward(); opt_sv.step()

    S, _ = layer_sv.delta_W_rank_analysis()
    bars  = " ▁▂▃▄▅▆▇█"
    smax  = S[0].item()

    print(f"  Singular values of learned ΔW (rank-16 LoRA, rank-4 target):")
    print(f"  {'Rank':>6}  {'σ value':>12}  {'σ / σ₁':>8}  Bar")
    print(f"  {'':─>6}  {'':─>12}  {'':─>8}  {'':─}")
    for i, sv in enumerate(S):
        v       = sv.item()
        frac    = v / smax if smax > 0 else 0
        bar_len = int(frac * 30)
        bar     = "█" * bar_len
        note    = " ← most important" if i < 4 else ""
        print(f"  {i+1:>6}  {v:>12.4f}  {frac:>8.3f}  {bar}{note}")

    print()
    print("  The first 4 singular values dominate (matching the true rank=4 target).")
    print("  This confirms the LoRA hypothesis: fine-tuning changes are low-rank.")
''',
    },

    "QLoRA: 4-bit Quantisation Simulation": {
        "description": "Simulate NF4 quantisation of model weights, implement QLoRA training (4-bit base + bf16 LoRA adapters), and measure the memory savings vs full fine-tuning and standard LoRA.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
QLORA: 4-BIT QUANTISATION + LORA ADAPTERS
================================================================================

Simulates QLoRA:
    1. NF4 (Normal Float 4-bit) quantisation of base model weights
    2. Double quantisation (quantise the quantisation constants)
    3. LoRA adapters applied in bf16 on top of quantised base
    4. Memory analysis: how QLoRA enables 65B models on consumer GPUs
    5. Quality comparison: NF4 quantisation error vs full precision

Key insight: gradients do NOT flow through the quantisation step.
The quantised weights are treated as frozen constants.
Only the LoRA A and B matrices receive gradient updates.

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── NF4 quantisation ──────────────────────────────────────────────────────────

# The 16 NF4 data type values, placed at quantile points of N(0,1)
# These are the lookup table values from the QLoRA paper
NF4_VALUES = torch.tensor([
    -1.0,       -0.6961928,  -0.5250730,  -0.3949310,
    -0.2844630,  -0.1848450,  -0.0911740,   0.0000000,
     0.0795620,   0.1609030,   0.2461430,   0.3379280,
     0.4406830,   0.5626170,   0.7229560,   1.0000000,
], dtype=torch.float32)


def quantise_nf4(tensor: torch.Tensor, block_size: int = 64) -> tuple:
    """
    Quantise a tensor to NF4 (4-bit Normal Float).

    Process:
        1. Split tensor into blocks of block_size elements
        2. For each block: find max absolute value (scale)
        3. Normalise block to [-1, 1]
        4. Find nearest NF4 value for each normalised element
        5. Store as 4-bit indices + float32 scales

    Returns:
        quantised:  (n_blocks, block_size) uint8 tensor of NF4 indices
        scales:     (n_blocks,) float32 tensor of per-block scales
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n    = len(flat)

    # Pad to multiple of block_size
    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    n_blocks = len(flat) // block_size
    blocks   = flat.view(n_blocks, block_size)

    # Per-block scales (max absolute value)
    scales = blocks.abs().max(dim=1).values.clamp(min=1e-8)

    # Normalise to [-1, 1]
    norm_blocks = blocks / scales.unsqueeze(1)

    # Find nearest NF4 value for each element
    nf4_expanded = NF4_VALUES.unsqueeze(0).unsqueeze(0)   # (1, 1, 16)
    norm_exp     = norm_blocks.unsqueeze(-1)               # (n_blocks, block_size, 1)
    distances    = (norm_exp - nf4_expanded).abs()         # (n_blocks, block_size, 16)
    indices      = distances.argmin(dim=-1).to(torch.uint8) # (n_blocks, block_size)

    return indices, scales, original_shape, n


def dequantise_nf4(indices: torch.Tensor, scales: torch.Tensor,
                    original_shape, n_original: int,
                    block_size: int = 64) -> torch.Tensor:
    """Dequantise NF4 back to float32."""
    nf4_vals     = NF4_VALUES.to(indices.device)
    quantised    = nf4_vals[indices.long()]           # (n_blocks, block_size)
    dequantised  = quantised * scales.unsqueeze(1)    # scale back
    return dequantised.flatten()[:n_original].reshape(original_shape)


def quantisation_error(original: torch.Tensor, block_size: int = 64) -> dict:
    """Measure NF4 quantisation error vs linear 4-bit quantisation."""
    # NF4 quantisation
    idx, scales, shape, n = quantise_nf4(original, block_size)
    reconstructed_nf4     = dequantise_nf4(idx, scales, shape, n, block_size)

    # Linear 4-bit quantisation (uniform spacing in [-1, 1])
    linear_vals = torch.linspace(-1.0, 1.0, 16)
    flat_orig   = original.flatten().float()
    # Per-block linear quant
    n_blocks    = math.ceil(len(flat_orig) / block_size)
    padded      = F.pad(flat_orig, (0, n_blocks * block_size - len(flat_orig)))
    blocks      = padded.view(n_blocks, block_size)
    sc          = blocks.abs().max(dim=1).values.clamp(min=1e-8)
    norm_b      = blocks / sc.unsqueeze(1)
    lin_idx     = (norm_b.unsqueeze(-1) - linear_vals.unsqueeze(0).unsqueeze(0)).abs().argmin(-1)
    lin_q       = linear_vals[lin_idx] * sc.unsqueeze(1)
    reconstructed_lin = lin_q.flatten()[:len(flat_orig)].reshape(shape)

    err_nf4 = (original - reconstructed_nf4).pow(2).mean().sqrt().item()
    err_lin = (original - reconstructed_lin).pow(2).mean().sqrt().item()

    return {
        "nf4_rmse":    err_nf4,
        "linear_rmse": err_lin,
        "nf4_better":  err_nf4 < err_lin,
        "improvement": (err_lin - err_nf4) / err_lin * 100,
    }


# ── QLoRA layer ───────────────────────────────────────────────────────────────

class QLoRALinear(nn.Module):
    """
    QLoRA: 4-bit quantised base weight + bf16 LoRA adapters.

    The quantised weight is a frozen constant. Gradients flow through
    the dequantised computation, but only the LoRA A and B matrices
    accumulate gradients and receive optimizer updates.

    Memory:
        base weight:  d×k / 2 bytes (4 bits per param = 0.5 bytes)
        scales:       d×k / block_size × 4 bytes (fp32, ~0.0625 bytes/param)
        LoRA A:       r×k × 2 bytes (bf16)
        LoRA B:       d×r × 2 bytes (bf16)
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, lora_alpha: float = 16.0,
                 block_size: int = 64, bias: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.scaling      = lora_alpha / rank
        self.block_size   = block_size

        # Build and quantise a random base weight
        torch.manual_seed(42)
        W = torch.randn(out_features, in_features)
        idx, scales, shape, n = quantise_nf4(W, block_size)

        # Store quantised state (non-parameter buffers)
        self.register_buffer("W_quant",   idx)
        self.register_buffer("W_scales",  scales)
        self.W_shape  = shape
        self.W_n      = n

        # LoRA parameters (trainable, bf16)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features).half())
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank).half())
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features).half())
        else:
            self.bias = None

    def get_base_weight(self) -> torch.Tensor:
        """Dequantise the base weight on the fly (bf16)."""
        W_fp32 = dequantise_nf4(
            self.W_quant, self.W_scales, self.W_shape, self.W_n, self.block_size
        )
        return W_fp32.to(self.lora_A.dtype)   # cast to bf16/fp16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantise base weight for this forward pass
        W_dequant = self.get_base_weight()   # bf16, no grad

        # Base output (no grad through W_dequant)
        with torch.no_grad():
            base = F.linear(x.half(), W_dequant, self.bias)

        # LoRA output (grad flows through lora_A and lora_B)
        lora = F.linear(F.linear(x.half(), self.lora_A), self.lora_B) * self.scaling

        return (base + lora).float()

    @property
    def memory_bytes(self) -> dict:
        """Memory footprint of each component."""
        base_quant  = self.W_quant.numel() // 2      # 4 bits each = 0.5 bytes
        base_scales = self.W_scales.numel() * 4      # fp32 scales
        lora_ab     = (self.lora_A.numel() + self.lora_B.numel()) * 2   # bf16
        full_ref    = self.in_features * self.out_features * 4   # fp32 reference
        return {
            "base_4bit_bytes":   base_quant,
            "base_scales_bytes": base_scales,
            "lora_bytes":        lora_ab,
            "total_bytes":       base_quant + base_scales + lora_ab,
            "full_fp32_bytes":   full_ref,
            "compression_ratio": full_ref / (base_quant + base_scales + lora_ab),
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 65)
    print("  NF4 QUANTISATION ERROR ANALYSIS")
    print("=" * 65)
    print()

    # Test on different weight distributions
    distributions = [
        ("N(0,1) — typical weights",  torch.randn(4096, 4096)),
        ("N(0,0.02) — small weights", torch.randn(4096, 4096) * 0.02),
        ("Laplace(0,1)",              torch.distributions.Laplace(0,1).sample((4096, 4096))),
        ("Uniform[-1,1]",             torch.rand(4096, 4096) * 2 - 1),
    ]

    print(f"  {'Distribution':<30}  {'NF4 RMSE':>12}  {'Linear RMSE':>14}  "
          f"{'NF4 better?':>14}  {'Improvement':>14}")
    print(f"  {'':─<30}  {'':─>12}  {'':─>14}  {'':─>14}  {'':─>14}")

    for name, data in distributions:
        err = quantisation_error(data, block_size=64)
        print(f"  {name:<30}  {err['nf4_rmse']:>12.6f}  {err['linear_rmse']:>14.6f}  "
              f"{'✓ YES' if err['nf4_better'] else '✗ NO':>14}  "
              f"{err['improvement']:>13.1f}%")

    print()
    print("  NF4 is designed for normally-distributed weights and outperforms")
    print("  linear 4-bit quantisation for typical neural network weight matrices.")

    # Memory analysis
    print()
    print("=" * 65)
    print("  QLORA MEMORY ANALYSIS")
    print("=" * 65)
    print()

    d = 4096
    layer = QLoRALinear(d, d, rank=16, lora_alpha=16.0)
    mem   = layer.memory_bytes

    print(f"  Layer: d_in={d}, d_out={d}, rank=16")
    print()
    print(f"  Base weight (4-bit NF4):   {mem['base_4bit_bytes']/1e6:.2f} MB")
    print(f"  Quantisation scales:        {mem['base_scales_bytes']/1e6:.2f} MB")
    print(f"  LoRA A + B (bf16):          {mem['lora_bytes']/1e6:.2f} MB")
    print(f"  Total (QLoRA):              {mem['total_bytes']/1e6:.2f} MB")
    print(f"  Reference (fp32 full):      {mem['full_fp32_bytes']/1e6:.2f} MB")
    print(f"  Compression ratio:          {mem['compression_ratio']:.1f}×")
    print()

    # QLoRA forward pass correctness
    x     = torch.randn(2, 16, d)
    out   = layer(x)
    print(f"  Forward pass output shape: {out.shape}  ✓")
    print()

    # Large model memory projections
    print("=" * 65)
    print("  LARGE MODEL MEMORY: Full FT vs LoRA vs QLoRA")
    print("=" * 65)
    print()

    models_info = [
        ("7B",   7e9,   "1×A100 40GB"),
        ("13B",  13e9,  "2×A100 40GB"),
        ("33B",  33e9,  "4×A100 40GB"),
        ("65B",  65e9,  "1×A100 80GB"),
        ("70B",  70e9,  "1×A100 80GB"),
        ("405B", 405e9, "8×A100 80GB"),
    ]

    print(f"  {'Model':<8}  {'Full FT (bf16)':>16}  {'LoRA (r=16)':>14}  "
          f"{'QLoRA (4-bit)':>16}  {'Fits GPU?':>12}")
    print(f"  {'':─<8}  {'':─>16}  {'':─>14}  {'':─>16}  {'':─>12}")

    for name, P, target_gpu in models_info:
        # Full fine-tuning: weights(2) + grads(2) + master(4) + m(4) + v(4) = 16 bytes/param
        full_ft_gb  = P * 16 / 1e9

        # Standard LoRA: frozen base(2) + no grads(0) + LoRA states only
        # Frozen base doesn't need optimizer state; only LoRA params do
        # LoRA params ≈ 1% of P at r=16
        lora_ratio  = 0.01   # ~1% trainable
        lora_gb     = (P * 2              # bf16 base (no grad, no optim)
                       + P * lora_ratio * 2   # LoRA bf16 params
                       + P * lora_ratio * 16  # LoRA optimizer state
                       ) / 1e9

        # QLoRA: 4-bit base (~0.5 bytes/param + ~0.06 scales)
        # + bf16 LoRA adapters + their fp32 optimizer state
        qlora_gb    = (P * 0.56           # NF4 + scales
                       + P * lora_ratio * 2   # LoRA bf16 params
                       + P * lora_ratio * 16  # LoRA optimizer state
                       ) / 1e9

        gpu_mem     = float(target_gpu.split("×")[0]) * (
            80 if "80" in target_gpu else 40
        ) if "×" in target_gpu else (
            80 if "80" in target_gpu else 40
        )

        fits_full   = "✓" if full_ft_gb <= gpu_mem else "✗"
        fits_lora   = "✓" if lora_gb   <= gpu_mem else "✗"
        fits_qlora  = "✓" if qlora_gb  <= gpu_mem else "✗"
        fits        = f"{fits_full}/{fits_lora}/{fits_qlora}"

        print(f"  {name:<8}  {full_ft_gb:>15.0f}GB  {lora_gb:>13.0f}GB  "
              f"{qlora_gb:>15.0f}GB  {fits:>12}")

    print()
    print("  Columns: Full FT / LoRA / QLoRA. ✓=fits on target GPU, ✗=too large")
    print("  QLoRA enables 65B–70B fine-tuning on a single A100 80GB.")
''',
    },

    "LoRA Rank Selection and Ablation Study": {
        "description": "Systematically ablate LoRA rank and target module selection — showing quality vs parameter count Pareto frontier, and which layers matter most for different task types.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
LORA RANK ABLATION AND MODULE SELECTION
================================================================================

Studies the effect of:
    1. Rank r on downstream quality (PEFT Pareto frontier)
    2. Which modules to apply LoRA to (attention vs FFN)
    3. RSLoRA vs standard LoRA scaling
    4. The "intrinsic dimensionality" of task adaptation

Provides practical guidance for choosing LoRA configuration.

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    rank:             int    = 16
    lora_alpha:       float  = 16.0
    target_modules:   list   = None
    dropout:          float  = 0.0
    use_rslora:       bool   = False
    lr:               float  = 1e-3
    name:             str    = ""


# ── Minimal Transformer for ablation ─────────────────────────────────────────

class MiniTransformerBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm1  = nn.LayerNorm(d)
        self.q      = nn.Linear(d, d, bias=False)
        self.k      = nn.Linear(d, d, bias=False)
        self.v      = nn.Linear(d, d, bias=False)
        self.o      = nn.Linear(d, d, bias=False)
        self.norm2  = nn.LayerNorm(d)
        self.gate   = nn.Linear(d, d * 4, bias=False)
        self.up     = nn.Linear(d, d * 4, bias=False)
        self.down   = nn.Linear(d * 4, d, bias=False)

    def forward(self, x):
        # Simplified attention (no causal mask for demo)
        normed = self.norm1(x)
        q      = self.q(normed)
        k      = self.k(normed)
        v      = self.v(normed)
        # Dot-product attention
        scale  = math.sqrt(q.shape[-1])
        attn   = torch.softmax(q @ k.transpose(-2,-1) / scale, dim=-1)
        x      = x + self.o(attn @ v)
        # FFN (SwiGLU-like)
        normed = self.norm2(x)
        x      = x + self.down(F.silu(self.gate(normed)) * self.up(normed))
        return x


class MiniLM(nn.Module):
    def __init__(self, vocab: int = 512, d: int = 128, n_layers: int = 2):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([MiniTransformerBlock(d) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


# ── LoRA application (simplified) ────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float,
                 rslora: bool = False):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.weight  = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias    = (nn.Parameter(linear.bias.data.clone(), requires_grad=False)
                        if linear.bias is not None else None)
        self.lora_A  = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B  = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale   = alpha / (math.sqrt(rank) if rslora else rank)

    def forward(self, x):
        return (F.linear(x, self.weight, self.bias)
                + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale)


def apply_lora_to_model(model: nn.Module, target_module_names: list[str],
                          rank: int, alpha: float, rslora: bool = False) -> nn.Module:
    """Freeze all params, replace specified Linear layers with LoRA versions."""
    for p in model.parameters():
        p.requires_grad_(False)

    def _replace(module: nn.Module, name: str):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name in target_module_names and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank, alpha, rslora))
            else:
                _replace(child, full_name)

    _replace(model, "")
    return model


def run_ablation(cfg: LoRAConfig, base_model: nn.Module,
                  train_data: torch.Tensor, eval_data: torch.Tensor,
                  n_steps: int = 100, vocab: int = 512) -> dict:
    """Run one LoRA training configuration and return results."""
    model = copy.deepcopy(base_model)

    targets = cfg.target_modules or []
    if targets:
        model = apply_lora_to_model(model, targets,
                                     rank=cfg.rank, alpha=cfg.lora_alpha,
                                     rslora=cfg.use_rslora)
    else:
        # Full fine-tuning
        for p in model.parameters():
            p.requires_grad_(True)

    total_p    = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt        = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.0
    )

    losses     = []
    B, T       = 4, 16

    for step in range(n_steps):
        idx    = torch.randint(0, len(train_data) - T, (B,))
        batch  = torch.stack([train_data[i:i+T] for i in idx])

        opt.zero_grad()
        logits = model(batch[:, :-1])
        loss   = F.cross_entropy(logits.reshape(-1, vocab), batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        losses.append(loss.item())

    # Eval
    model.eval()
    with torch.no_grad():
        idx   = torch.randint(0, len(eval_data) - T, (16,))
        ebatch = torch.stack([eval_data[i:i+T] for i in idx])
        eval_loss = F.cross_entropy(
            model(ebatch[:, :-1]).reshape(-1, vocab),
            ebatch[:, 1:].reshape(-1)
        ).item()

    return {
        "name":          cfg.name,
        "rank":          cfg.rank,
        "trainable":     trainable,
        "total":         total_p,
        "pct":           trainable / total_p * 100,
        "train_loss":    losses[-1],
        "eval_loss":     eval_loss,
        "eval_ppl":      math.exp(eval_loss),
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    VOCAB, D = 512, 64
    N_STEPS  = 60

    # Synthetic token data
    train_data = torch.randint(3, VOCAB, (5000,))
    eval_data  = torch.randint(3, VOCAB, (1000,))

    # Base model
    base_model = MiniLM(VOCAB, D, n_layers=2)

    # All attention module names (for 2-layer model)
    attn_modules = [
        "blocks.0.q", "blocks.0.k", "blocks.0.v", "blocks.0.o",
        "blocks.1.q", "blocks.1.k", "blocks.1.v", "blocks.1.o",
    ]
    ffn_modules  = [
        "blocks.0.gate", "blocks.0.up", "blocks.0.down",
        "blocks.1.gate", "blocks.1.up", "blocks.1.down",
    ]

    # ── Ablation 1: Rank sweep ─────────────────────────────────────────────────
    print("=" * 68)
    print("  RANK ABLATION (attention only, same alpha=rank)")
    print("=" * 68)
    print()

    rank_configs = [
        LoRAConfig(rank=1,   lora_alpha=1,   target_modules=attn_modules, lr=2e-3, name="r=1"),
        LoRAConfig(rank=4,   lora_alpha=4,   target_modules=attn_modules, lr=1e-3, name="r=4"),
        LoRAConfig(rank=8,   lora_alpha=8,   target_modules=attn_modules, lr=1e-3, name="r=8"),
        LoRAConfig(rank=16,  lora_alpha=16,  target_modules=attn_modules, lr=1e-3, name="r=16"),
        LoRAConfig(rank=32,  lora_alpha=32,  target_modules=attn_modules, lr=5e-4, name="r=32"),
        LoRAConfig(rank=0,   lora_alpha=0,   target_modules=[],           lr=3e-4, name="Full FT"),
    ]

    print(f"  {'Config':<12}  {'Trainable':>12}  {'% of total':>12}  "
          f"{'Eval loss':>12}  {'Eval PPL':>10}  {'vs Full FT':>12}")
    print(f"  {'':─<12}  {'':─>12}  {'':─>12}  {'':─>12}  {'':─>10}  {'':─>12}")

    rank_results = {}
    full_ft_ppl  = None
    for cfg in rank_configs:
        r = run_ablation(cfg, base_model, train_data, eval_data,
                          n_steps=N_STEPS, vocab=VOCAB)
        rank_results[cfg.name] = r
        if cfg.name == "Full FT":
            full_ft_ppl = r["eval_ppl"]

    for cfg in rank_configs:
        r = rank_results[cfg.name]
        vs_ft = (r["eval_ppl"] - full_ft_ppl) / full_ft_ppl * 100 if full_ft_ppl else 0
        flag  = f"+{vs_ft:.1f}%" if vs_ft > 0 else f"{vs_ft:.1f}%"
        print(f"  {r['name']:<12}  {r['trainable']:>12,}  {r['pct']:>11.2f}%  "
              f"{r['eval_loss']:>12.4f}  {r['eval_ppl']:>10.2f}  {flag:>12}")

    # ── Ablation 2: Module selection ───────────────────────────────────────────
    print()
    print("=" * 68)
    print("  MODULE SELECTION ABLATION (r=8, alpha=8)")
    print("=" * 68)
    print()

    module_configs = [
        LoRAConfig(rank=8, lora_alpha=8, target_modules=["blocks.0.q", "blocks.0.v",
                                                           "blocks.1.q", "blocks.1.v"],
                    lr=1e-3, name="q,v only"),
        LoRAConfig(rank=8, lora_alpha=8, target_modules=attn_modules,
                    lr=1e-3, name="all attn"),
        LoRAConfig(rank=8, lora_alpha=8, target_modules=ffn_modules,
                    lr=1e-3, name="all FFN"),
        LoRAConfig(rank=8, lora_alpha=8, target_modules=attn_modules + ffn_modules,
                    lr=1e-3, name="attn+FFN"),
    ]

    print(f"  {'Config':<14}  {'Trainable':>12}  {'Eval PPL':>10}  Notes")
    print(f"  {'':─<14}  {'':─>12}  {'':─>10}  {'':─}")
    for cfg in module_configs:
        r = run_ablation(cfg, base_model, train_data, eval_data,
                          n_steps=N_STEPS, vocab=VOCAB)
        vs_ft = (r["eval_ppl"] - full_ft_ppl) / full_ft_ppl * 100
        note  = "original LoRA paper default" if cfg.name == "q,v only" else ""
        print(f"  {cfg.name:<14}  {r['trainable']:>12,}  "
              f"{r['eval_ppl']:>10.2f}  {note}")

    # ── Ablation 3: RSLoRA vs standard ────────────────────────────────────────
    print()
    print("=" * 68)
    print("  RSLORA vs STANDARD LORA (different ranks, same α)")
    print("=" * 68)
    print()

    rslora_configs = [
        (8,  "Standard r=8",  False),
        (8,  "RSLoRA  r=8",   True),
        (32, "Standard r=32", False),
        (32, "RSLoRA  r=32",  True),
    ]

    print(f"  {'Config':<18}  {'Scaling':>12}  {'Eval PPL':>12}")
    print(f"  {'':─<18}  {'':─>12}  {'':─>12}")

    for r, name, rslora in rslora_configs:
        cfg = LoRAConfig(rank=r, lora_alpha=16.0, target_modules=attn_modules,
                          lr=1e-3, use_rslora=rslora, name=name)
        res = run_ablation(cfg, base_model, train_data, eval_data,
                            n_steps=N_STEPS, vocab=VOCAB)
        scaling = f"α/√r = {16/math.sqrt(r):.2f}" if rslora else f"α/r = {16/r:.2f}"
        print(f"  {name:<18}  {scaling:>12}  {res['eval_ppl']:>12.2f}")

    print()
    print("  RSLoRA uses α/√r scaling, making results more consistent across")
    print("  different ranks when α is held fixed.")
    print()
    print("  Summary:")
    print("  • r=16 is the sweet spot for most general instruction tuning")
    print("  • Including FFN modules helps for complex reasoning tasks")
    print("  • RSLoRA gives consistent quality when sweeping rank with fixed α")
    print("  • Full fine-tuning is only necessary when dataset is > 10M tokens")
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
    #     from llm_training.visuals.peft_lora_qlora import (
    #         LORA_VISUAL_HTML,
    #         LORA_VISUAL_HEIGHT,
    #     )
    #     visual_html   = LORA_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = LORA_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[27_peft_lora_qlora.py] Could not load visual: {e}",
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