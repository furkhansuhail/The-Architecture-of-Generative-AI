"""
Mixed Precision Training — FP16, BF16, and Loss Scaling
=========================================================

Training large language models in full float32 precision is prohibitively
expensive: it doubles memory and halves throughput compared to 16-bit
formats. Mixed precision training runs the forward and backward passes in
a lower-precision format while keeping a full-precision master copy of the
weights for the optimiser update. Understanding the numerical properties of
each format, when to use fp16 vs bf16, and how loss scaling prevents
underflow is essential for stable large-scale training.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Mixed Precision Training — FP16, BF16, Loss Scaling"
DISPLAY_NAME = "10 · Mixed Precision"
ICON         = "🔢"
SUBTITLE     = "FP16, BF16, and Loss Scaling"


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

### Why Precision Matters

Every number stored in a GPU is represented in a floating-point format. The
format determines:

    •   Range: the largest and smallest representable values
    •   Precision: how many significant digits are stored
    •   Memory: bytes per value
    •   Speed: how fast arithmetic operations execute

Training a 7B parameter model in float32 requires ~112 GB (params + grads +
optimiser state). In bf16/fp16, the same training drops to ~56–84 GB. More
importantly, modern GPUs execute 16-bit operations 2–8× faster than 32-bit
(on Tensor Cores), so mixed precision is both a memory and a throughput win.


### Floating-Point Format Anatomy

Every IEEE floating-point number is stored as three fields:

    [sign | exponent | mantissa (significand)]

    format    bits   sign   exponent   mantissa   max val        min normal val
    ────────────────────────────────────────────────────────────────────────────
    float32   32      1       8          23        ~3.4 × 10³⁸    ~1.2 × 10⁻³⁸
    float16   16      1       5          10        ~6.5 × 10⁴     ~6.1 × 10⁻⁵
    bfloat16  16      1       8          7         ~3.4 × 10³⁸    ~1.2 × 10⁻³⁸
    float8 e4m3 8     1       4          3         ~448            ~0.002
    int8       8      1       —          7         127             —
    ────────────────────────────────────────────────────────────────────────────

The key architectural insight: **bfloat16 has the same exponent width as
float32 (8 bits), so it has the same dynamic range**. Float16 has only 5
exponent bits — its range is ~65,000× smaller than float32.

**The value of a floating-point number:**

    value = (−1)^sign × 2^(exponent − bias) × (1 + mantissa/2^M)

    where bias = 2^(E-1) − 1,  E = number of exponent bits,  M = mantissa bits


    **Diagram 1 — Bit Layout Comparison:**

    FLOATING-POINT FORMAT BIT LAYOUTS
    ════════════════════════════════════════════════════════════════

    float32  (32 bits):
    ┌─┬────────┬───────────────────────┐
    │S│EEEEEEEE│MMMMMMMMMMMMMMMMMMMMMMM│
    └─┴────────┴───────────────────────┘
     1    8              23
     sign exponent      mantissa (significand)

    float16  (16 bits):
    ┌─┬─────┬──────────┐
    │S│EEEEE│MMMMMMMMMM│
    └─┴─────┴──────────┘
     1   5        10
     sign exp  mantissa

    bfloat16 (16 bits):
    ┌─┬────────┬───────┐
    │S│EEEEEEEE│MMMMMMM│
    └─┴────────┴───────┘
     1    8        7
     sign exponent mantissa

    float32 → bfloat16:  truncate the lower 16 bits of the mantissa
                         (no exponent change → same range)
    float32 → float16:   re-encode with smaller exponent and mantissa
                         (range dramatically reduced)

    BF16 IS A TRUNCATED FLOAT32. It can represent the same range of values
    but with less precision (~2.3 significant decimal digits vs ~7.2 for fp32).


### FP16 vs BF16: Which to Use?

**Float16 (fp16):**
    •   Supported on all NVIDIA GPUs since Pascal (2016)
    •   5-bit exponent → max value ≈ 65,504
    •   Gradients easily overflow (common in training)
    •   Requires **loss scaling** to prevent underflow of small gradients
    •   Unstable for models with large activations or logits
    •   Still used on older hardware (V100, T4)

**BFloat16 (bf16):**
    •   Supported on Ampere (A100, 3090) and later; TPUs always
    •   8-bit exponent → same range as float32
    •   Gradients almost never overflow (same range as fp32)
    •   Loss scaling rarely needed (but sometimes used as extra safety)
    •   **Standard for modern LLM training** (LLaMA, GPT-4, PaLM, etc.)
    •   Slightly lower precision than fp16 (7 vs 10 mantissa bits)

**Rule of thumb:** Use bf16 if your hardware supports it. Use fp16 + loss
scaling if you are on V100 or older hardware.


    **Diagram 2 — Overflow and Underflow Regions:**

    FP16 OVERFLOW AND UNDERFLOW RISK
    ════════════════════════════════════════════════════════════════

    Number line (log scale):

    fp32:  ←──────────────────────────────────────────────────→
           1.2e-38                                        3.4e38

    fp16:  ←─────────────────────→     [OVERFLOW ZONE]
           6.1e-5            65504     (values > 65504 → ∞)

                  ↑
           [UNDERFLOW ZONE]
           (values < 6.1e-5 become 0 — gradients lost!)

    bf16:  ←──────────────────────────────────────────────────→
           1.2e-38                                        3.4e38
           Same range as fp32 — overflow extremely rare ✓

    In LLM training, typical gradient values span [1e-7, 1.0].
    fp16 underflows for gradients < 6.1e-5 → silent information loss!
    bf16 handles the full range → no loss scaling usually needed.


### Mixed Precision Training: The Standard Recipe

Mixed precision does NOT mean "everything is in fp16/bf16". The recipe is:

    Component                    Precision      Why
    ─────────────────────────────────────────────────────────────────────
    Model weights (compute)      bf16 / fp16    Speed + memory
    Activations                  bf16 / fp16    Speed + memory
    Gradients                    bf16 / fp16    Speed (during accumulation)
    Master weights               fp32           Precision for updates
    Optimiser state (m, v)       fp32           Precision for AdamW
    Loss value                   fp32           Stability
    LayerNorm, softmax           fp32           Numerical stability
    ─────────────────────────────────────────────────────────────────────

The core idea: keep a float32 "master copy" of the weights. For each
training step:
    1.  Cast master weights → bf16/fp16 for the forward/backward pass
    2.  Compute gradients in bf16/fp16 (fast)
    3.  Cast gradients → fp32 for the optimiser update
    4.  Update master weights in fp32 (precise)
    5.  (The bf16/fp16 copies are discarded — recast next step)

This gives the speed of 16-bit arithmetic while preserving the precision
of float32 for the weight updates that actually matter.


### Loss Scaling (Required for FP16, Optional for BF16)

The problem with fp16: gradients are often very small in magnitude
(< 6.1 × 10⁻⁵), which causes them to underflow to exactly 0 in fp16.
Zero gradients → no learning for those parameters → silent failure.

**Loss scaling** multiplies the loss by a large scalar S before backprop:

    scaled_loss = loss × S          (e.g. S = 65536)
    scaled_loss.backward()          → gradients are also ×S
    gradients ÷= S                  → back to true scale before update

Because gradients are S times larger in fp16, they no longer underflow.
The key is to unscale BEFORE gradient clipping and the optimiser step.

**Dynamic loss scaling (GradScaler):**
PyTorch's `torch.cuda.amp.GradScaler` adjusts S automatically:
    •   If no inf/NaN gradients: increase S (more headroom)
    •   If inf/NaN detected: halve S and skip the update step

The scale starts large (e.g. 65536) and adapts to keep gradients in the
representable range without overflow.

    scaler = GradScaler()

    with autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss   = criterion(logits, y)

    scaler.scale(loss).backward()    # backward with scaled loss
    scaler.unscale_(optimizer)       # unscale grads before clipping
    clip_grad_norm_(params, 1.0)     # clip unscaled grads
    scaler.step(optimizer)           # update (skipped if grads are inf/nan)
    scaler.update()                  # adjust scale factor for next step


### PyTorch `autocast` and the Mixed Precision Context

`torch.autocast` automatically casts operations to the target dtype within
its scope. Not all operations are cast — some are kept in fp32 for
numerical safety:

    Cast to bf16/fp16 (fast, numerically safe):
        •   Matrix multiplications (linear layers, attention)
        •   Convolutions
        •   Element-wise operations on tensors

    Kept in fp32 (numerically sensitive):
        •   LayerNorm, GroupNorm
        •   Softmax, log_softmax
        •   Cross-entropy loss
        •   Loss scaling operations

This means you don't need to manually manage precision for most operations
— `autocast` handles it correctly.


    **Diagram 3 — Mixed Precision Training Data Flow:**

    MIXED PRECISION TRAINING STEP
    ════════════════════════════════════════════════════════════════

    Master weights  W_fp32 ────────────────────────────────────┐
         │                                                       │ ← fp32 update
         ▼  cast to bf16                                        │
    Compute weights W_bf16                                       │
         │                                                       │
         ▼  forward pass (autocast)                             │
    Activations (bf16)  →  Logits (fp32)  →  Loss (fp32)       │
         │                                                       │
         ▼  backward pass (autocast)                            │
    Gradients G_bf16                                            │
         │                                                       │
         ▼  cast to fp32                                        │
    Gradients G_fp32  →  clip  →  AdamW update  ──────────────┘
         │
         [fp32 master weights updated, bf16 copies regenerated next step]

    Memory layout:
    W_fp32:  7B × 4 bytes = 28 GB   ← master copy (training only)
    W_bf16:  7B × 2 bytes = 14 GB   ← compute copy
    m, v:    7B × 8 bytes = 56 GB   ← optimiser state (fp32)
    Total:   ~98 GB  (vs ~112 GB for pure fp32 — modest saving)

    The real saving is throughput: bf16 matmuls are 2–4× faster on A100.


### FP8 — The Next Frontier

NVIDIA H100 GPUs introduce native FP8 (float8) support. The two common
variants:
    •   E4M3 (4 exponent bits, 3 mantissa): forward pass, higher precision
    •   E5M2 (5 exponent bits, 2 mantissa): backward pass, higher range

FP8 training is 2× faster than bf16 on H100 Tensor Cores and enables
even larger batch sizes. It requires:
    •   Gradient scaling (similar to fp16 loss scaling)
    •   Per-tensor or per-block scaling factors (to handle dynamic range)
    •   Careful placement: not all ops can safely run in fp8

Used by: Transformer Engine (NVIDIA), DeepSpeed, some HuggingFace recipes.
The adoption is growing rapidly as H100/H200 become the standard training
hardware.


### Numerical Stability Tips for Mixed Precision

    Issue                    Cause                          Fix
    ─────────────────────────────────────────────────────────────────────
    Loss = NaN from step 1   lr too high, bad init          Reduce lr, check init
    Gradients = inf in fp16  Overflow (activation too large) Reduce scale or use bf16
    Gradients = 0 in fp16    Underflow                       Increase loss scale (GradScaler)
    Loss oscillates badly    Scale factor too large          Let GradScaler auto-adjust
    Accuracy loss vs fp32    Precision insufficient          Check if norm/softmax in fp32
    Slow convergence         Master weights not fp32         Ensure optimiser state in fp32
    ─────────────────────────────────────────────────────────────────────

**Critical: optimiser state must stay in fp32.** Running AdamW entirely in
bf16 (including m and v buffers) causes severe quality degradation because
the weight updates involve small corrections (lr × gradient / √v) that
are below bf16 precision. The difference between a weight of 1.0000 and
1.0001 may be unrepresentable in bf16 (which has ~0.78% relative precision),
causing the update to be silently dropped.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Floating-Point Format Comparison for LLM Training

| Property            | float32     | float16         | bfloat16         | float8 (e4m3)   |
|---------------------|-------------|-----------------|------------------|-----------------|
| Total bits          | 32          | 16              | 16               | 8               |
| Exponent bits       | 8           | 5               | 8                | 4               |
| Mantissa bits       | 23          | 10              | 7                | 3               |
| Max value           | ~3.4e38     | ~65,504         | ~3.4e38          | ~448            |
| Underflow risk      | Very low    | High (< 6e-5)   | Very low         | Very high       |
| Overflow risk       | Very low    | Moderate        | Very low         | High            |
| Loss scaling needed | No          | Yes             | Rarely           | Yes             |
| Memory per param    | 4 bytes     | 2 bytes         | 2 bytes          | 1 byte          |
| Tensor Core speed   | 1×          | 2–4×            | 2–4×             | 4–8×            |
| Best use            | Optim state | Legacy hardware | Modern LLM train | H100 forward    |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Floating-Point Format Explorer": {
        "description": "Inspect the bit layout, range, precision, and rounding behaviour of fp32, fp16, and bf16. Show exactly which values overflow, underflow, and round.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
FLOATING-POINT FORMAT EXPLORER
================================================================================

Explores the numerical properties of fp32, fp16, and bf16:
    1. Bit layout and field sizes
    2. Representable range (max, min normal, min subnormal)
    3. Precision (spacing between adjacent values = ULP)
    4. Which gradient values are lost in fp16 vs bf16
    5. Rounding comparison: fp32 value → fp16 vs → bf16
================================================================================
"""

import struct
import math
import torch


# ── Format descriptors ────────────────────────────────────────────────────────

FORMATS = {
    "float32":  {"bits": 32, "exp_bits": 8,  "man_bits": 23, "dtype": torch.float32},
    "float16":  {"bits": 16, "exp_bits": 5,  "man_bits": 10, "dtype": torch.float16},
    "bfloat16": {"bits": 16, "exp_bits": 8,  "man_bits": 7,  "dtype": torch.bfloat16},
}


def fp_max(exp_bits: int, man_bits: int) -> float:
    """Maximum representable value."""
    bias    = 2 ** (exp_bits - 1) - 1
    max_exp = 2 ** exp_bits - 2   # all-ones exponent reserved for inf/nan
    return (2 - 2 ** (-man_bits)) * 2 ** (max_exp - bias)


def fp_min_normal(exp_bits: int, man_bits: int) -> float:
    """Minimum positive normal value."""
    bias = 2 ** (exp_bits - 1) - 1
    return 2.0 ** (1 - bias)


def fp_epsilon(man_bits: int) -> float:
    """Machine epsilon (spacing between 1.0 and next representable value)."""
    return 2.0 ** (-man_bits)


def float_to_bits(x: float, dtype=torch.float32) -> str:
    """Return binary representation of a float."""
    t = torch.tensor(x, dtype=dtype)
    if dtype == torch.float32:
        bits = struct.unpack("I", struct.pack("f", x))[0]
        return f"{bits:032b}"
    elif dtype == torch.float16:
        bits = t.view(torch.int16).item() & 0xFFFF
        return f"{bits:016b}"
    elif dtype == torch.bfloat16:
        bits = t.view(torch.int16).item() & 0xFFFF
        return f"{bits:016b}"
    return ""


def annotate_bits(bits_str: str, exp_bits: int, man_bits: int) -> str:
    """Format bit string with field separators."""
    sign = bits_str[0]
    exp  = bits_str[1: 1 + exp_bits]
    man  = bits_str[1 + exp_bits:]
    return f"{sign} | {exp} | {man}"


if __name__ == "__main__":
    # 1. Format overview
    print("=" * 70)
    print("  FLOATING-POINT FORMAT PROPERTIES")
    print("=" * 70)
    print()
    print(f"  {'Format':<12} {'Bits':>5}  {'Exp':>4}  {'Man':>4}  {'Max':>12}  "
          f"{'Min normal':>12}  {'ε (ULP)':>12}")
    print(f"  {'':─<12} {'':─>5}  {'':─>4}  {'':─>4}  {'':─>12}  "
          f"{'':─>12}  {'':─>12}")
    for name, f in FORMATS.items():
        e, m = f["exp_bits"], f["man_bits"]
        print(f"  {name:<12} {f['bits']:>5}  {e:>4}  {m:>4}  "
              f"{fp_max(e,m):>12.2e}  {fp_min_normal(e,m):>12.2e}  "
              f"{fp_epsilon(m):>12.2e}")

    # 2. Bit layouts for the value 3.14159
    print()
    print("=" * 70)
    print("  BIT LAYOUTS FOR π ≈ 3.14159")
    print("=" * 70)
    pi = 3.14159265358979
    for name, f in FORMATS.items():
        bits = float_to_bits(pi, f["dtype"])
        anno = annotate_bits(bits, f["exp_bits"], f["man_bits"])
        val  = torch.tensor(pi, dtype=f["dtype"]).item()
        err  = abs(val - pi) / pi * 100
        print(f"  {name:<12}  [{anno}]")
        print(f"              stored={val:.8f}  error={err:.4f}%")
        print()

    # 3. Gradient underflow analysis
    print("=" * 70)
    print("  GRADIENT UNDERFLOW: Which values are lost in fp16?")
    print("=" * 70)
    print()
    print("  Small gradient values common in LLM training:")
    print()
    print(f"  {'Value':>12}  {'fp32':>10}  {'fp16':>10}  {'bf16':>10}  "
          f"{'fp16 lost?':>12}  {'bf16 lost?':>12}")
    print(f"  {'':─>12}  {'':─>10}  {'':─>10}  {'':─>10}  "
          f"{'':─>12}  {'':─>12}")

    grad_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 5e-5, 1e-6, 1e-7, 1e-8]
    for v in grad_values:
        v32  = torch.tensor(v, dtype=torch.float32).item()
        v16  = torch.tensor(v, dtype=torch.float16).item()
        vb16 = torch.tensor(v, dtype=torch.bfloat16).item()
        lost16  = "✗ ZERO" if v16  == 0.0 else f"ok {abs(v16-v32)/v32*100:.1f}%"
        lostb16 = "✗ ZERO" if vb16 == 0.0 else f"ok {abs(vb16-v32)/v32*100:.1f}%"
        print(f"  {v:>12.2e}  {v32:>10.2e}  {v16:>10.2e}  {vb16:>10.2e}  "
              f"{lost16:>12}  {lostb16:>12}")

    print()
    print("  fp16 underflows for values < ~6.1e-5 → gradients silently zero!")
    print("  bf16 never underflows in this range (same exponent width as fp32) ✓")

    # 4. Overflow test
    print()
    print("=" * 70)
    print("  OVERFLOW TEST: Large logit values")
    print("=" * 70)
    print()
    large_values = [1000, 5000, 30000, 60000, 65504, 65505, 100000, 1e10]
    print(f"  {'Value':>10}  {'fp32 ok?':>10}  {'fp16 ok?':>12}  {'bf16 ok?':>12}")
    print(f"  {'':─>10}  {'':─>10}  {'':─>12}  {'':─>12}")
    for v in large_values:
        v32  = torch.tensor(float(v), dtype=torch.float32).item()
        v16  = torch.tensor(float(v), dtype=torch.float16).item()
        vb16 = torch.tensor(float(v), dtype=torch.bfloat16).item()
        ok32  = "✓ ok" if not math.isinf(v32)  else "∞ overflow"
        ok16  = "✓ ok" if not math.isinf(v16)  else "∞ OVERFLOW"
        okb16 = "✓ ok" if not math.isinf(vb16) else "∞ overflow"
        print(f"  {v:>10.0f}  {ok32:>10}  {ok16:>12}  {okb16:>12}")

    print()
    print("  fp16 overflows at 65,505. bf16 handles up to ~3.4e38.")
    print("  LLM logits can easily reach 10,000+, causing fp16 overflow in the")
    print("  softmax denominator — this is why bf16 is strongly preferred.")
''',
    },

    "Mixed Precision Training Loop": {
        "description": "Complete mixed precision training loop using torch.autocast and GradScaler — bf16 mode, fp16 mode with loss scaling, and a comparison against fp32.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MIXED PRECISION TRAINING LOOP — AUTOCAST + GRADSCALER
================================================================================

A complete training loop demonstrating:
    1. BF16 autocast  (recommended for Ampere+)
    2. FP16 autocast  + GradScaler  (required for V100/older hardware)
    3. Full FP32      (baseline)

Shows throughput comparison, loss curves, and GradScaler state tracking.
================================================================================
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import copy


# ── Model ────────────────────────────────────────────────────────────────────

class SmallTransformer(nn.Module):
    def __init__(self, vocab=512, d=256, n_heads=4, n_layers=3, max_len=64):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.pos    = nn.Embedding(max_len, d)
        layer       = nn.TransformerDecoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d*4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.body   = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.max_len = max_len
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.embed(idx) + self.pos(pos)
        mask = self.causal_mask[:T, :T]
        x    = self.body(x, x, tgt_mask=mask, memory_mask=mask,
                         tgt_is_causal=True, memory_is_causal=True)
        return self.head(self.norm(x))


# ── Training runner ───────────────────────────────────────────────────────────

def train_one_mode(model, mode: str, n_steps: int = 200,
                   device: str = "cpu"):
    """
    Train the model in the specified precision mode.

    mode:
        "fp32"   — standard float32, no autocast
        "bf16"   — autocast to bfloat16
        "fp16"   — autocast to float16 + GradScaler
    """
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    scaler = None
    if mode == "fp16":
        scaler = GradScaler()

    autocast_dtype = {
        "fp32": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[mode]

    x = torch.randint(0, 512, (16, 32)).to(device)

    loss_history   = []
    scale_history  = []
    t0             = time.perf_counter()

    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)

        # ── Forward ────────────────────────────────────────────────────────
        if autocast_dtype is not None:
            with autocast(device_type=device, dtype=autocast_dtype):
                logits = model(x)                        # (B, T, V)
                loss   = F.cross_entropy(
                    logits[:, :-1].reshape(-1, 512),
                    x[:, 1:].reshape(-1),
                )
        else:
            logits = model(x)
            loss   = F.cross_entropy(
                logits[:, :-1].reshape(-1, 512),
                x[:, 1:].reshape(-1),
            )

        # ── Backward ───────────────────────────────────────────────────────
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scale_history.append(scaler.get_scale())
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        loss_history.append(loss.item())

    elapsed = time.perf_counter() - t0
    return loss_history, scale_history, elapsed


def check_device():
    """Choose best available device."""
    if torch.cuda.is_available():
        return "cuda"
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    DEVICE  = check_device()
    N_STEPS = 300
    print(f"Device: {DEVICE}")
    print()

    base_model = SmallTransformer()
    n_params   = sum(p.numel() for p in base_model.parameters())
    print(f"Model: {n_params:,} parameters")
    print()

    modes_to_run = ["fp32"]
    if DEVICE in ("cuda", "mps"):
        modes_to_run += ["bf16"]
    if DEVICE == "cuda":
        modes_to_run += ["fp16"]
    else:
        print("Note: fp16 + GradScaler requires CUDA. Showing fp32 and bf16 only.")
        print()

    results = {}
    for mode in modes_to_run:
        model = copy.deepcopy(base_model)
        print(f"  Training in {mode.upper()} mode...")
        losses, scales, elapsed = train_one_mode(model, mode, N_STEPS, DEVICE)
        results[mode] = (losses, scales, elapsed)
        print(f"    Done in {elapsed:.2f}s  final_loss={losses[-1]:.4f}")

    # Comparison table
    print()
    print("=" * 65)
    print("  TRAINING MODE COMPARISON")
    print("=" * 65)
    print()
    print(f"  {'Mode':<10}  {'Final Loss':>12}  {'Time (s)':>10}  "
          f"{'Steps/s':>10}  {'Relative':>10}")
    print(f"  {'':─<10}  {'':─>12}  {'':─>10}  {'':─>10}  {'':─>10}")

    fp32_time = results.get("fp32", (None, None, 1.0))[2]
    for mode, (losses, scales, elapsed) in results.items():
        steps_per_s = N_STEPS / elapsed
        relative    = fp32_time / elapsed
        print(f"  {mode:<10}  {losses[-1]:>12.4f}  {elapsed:>10.2f}  "
              f"{steps_per_s:>10.1f}  {relative:>9.2f}×")

    # GradScaler dynamics (fp16 only)
    if "fp16" in results and results["fp16"][1]:
        scales = results["fp16"][1]
        print()
        print("=" * 65)
        print("  GRADSCALER SCALE FACTOR DYNAMICS (fp16 mode)")
        print("=" * 65)
        print()
        print("  GradScaler doubles scale when no overflow, halves on overflow.")
        print()
        print(f"  {'Step':>6}  {'Scale':>12}  {'Change':>10}")
        print(f"  {'':─>6}  {'':─>12}  {'':─>10}")
        prev = scales[0]
        for i, s in enumerate(scales):
            if i % 30 == 0 or s != prev:
                change = "↑ doubled" if s > prev else ("↓ halved" if s < prev else "—")
                print(f"  {i:>6}  {s:>12.0f}  {change:>10}")
            prev = s
        print(f"  Final scale: {scales[-1]:,.0f}")

    # Mixed precision memory estimate
    print()
    print("=" * 65)
    print("  MEMORY BREAKDOWN (fp32 master + bf16 compute)")
    print("=" * 65)
    print()
    for model_name, n_b in [("7B params", 7e9), ("13B params", 13e9), ("70B params", 70e9)]:
        fp32_w  = n_b * 4
        bf16_w  = n_b * 2
        fp32_g  = n_b * 4
        fp32_mv = n_b * 8   # m + v in fp32
        total   = fp32_w + bf16_w + fp32_g + fp32_mv
        pure32  = n_b * 16  # w + g + m + v all fp32
        saving  = (pure32 - total) / pure32 * 100
        print(f"  {model_name}:")
        print(f"    FP32 master:   {fp32_w/1e9:.0f} GB")
        print(f"    BF16 compute:  {bf16_w/1e9:.0f} GB")
        print(f"    FP32 grads:    {fp32_g/1e9:.0f} GB")
        print(f"    Optim m+v:     {fp32_mv/1e9:.0f} GB")
        print(f"    Total:         {total/1e9:.0f} GB  "
              f"(vs {pure32/1e9:.0f} GB pure fp32 → {saving:.0f}% saving)")
        print()
''',
    },

    "Loss Scaling Deep Dive": {
        "description": "Implement GradScaler from scratch, show gradient underflow without scaling, demonstrate dynamic scale adjustment, and compare to PyTorch's built-in.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
LOSS SCALING — FROM SCRATCH
================================================================================

Demonstrates:
    1. What happens to fp16 gradients WITHOUT loss scaling (underflow)
    2. How loss scaling prevents underflow
    3. A minimal GradScaler implementation (grow / skip / backoff)
    4. Verification against PyTorch's GradScaler
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Demonstrate fp16 gradient underflow ───────────────────────────────────────

def underflow_demo():
    """Show that small fp16 gradients become exactly zero without scaling."""
    print("=" * 62)
    print("  FP16 GRADIENT UNDERFLOW DEMO")
    print("=" * 62)
    print()

    # A weight that will receive a small gradient
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16,
                      requires_grad=True)

    # Construct a loss that produces a very small gradient (~1e-5)
    x     = torch.ones(2, 2, dtype=torch.float16) * 1e-3
    loss  = (w * x).sum()   # gradient = x ≈ 1e-3 in fp32 → underflow in fp16?

    loss.backward()

    print(f"  Input scale (x values):     {x[0,0].item():.2e}")
    print(f"  Loss value:                 {loss.item():.2e}")
    print(f"  Gradient (should be 1e-3): {w.grad[0,0].item():.2e}")
    if w.grad[0,0].item() == 0.0:
        print(f"  ✗  Gradient is ZERO — underflow! Gradient was lost in fp16")
    else:
        print(f"  ✓  Gradient survived (fp16 can represent this value)")

    print()
    print("  Now with loss scaling (scale = 1024):")
    w2    = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16,
                          requires_grad=True)
    loss2 = (w2 * x).sum() * 1024.0   # scaled loss
    loss2.backward()
    scaled_grad = w2.grad[0, 0].item()
    true_grad   = scaled_grad / 1024.0

    print(f"  Scaled loss:                {loss2.item():.2e}")
    print(f"  Scaled gradient:            {scaled_grad:.2e}")
    print(f"  True gradient (÷ 1024):     {true_grad:.2e}")
    if true_grad > 0:
        print(f"  ✓  Gradient recovered by scaling!")


# ── Minimal GradScaler ────────────────────────────────────────────────────────

class MiniGradScaler:
    """
    A minimal implementation of PyTorch's GradScaler logic.

    Algorithm:
        - Start with scale = init_scale
        - After each successful update: growth_interval steps → double scale
        - After any inf/nan gradient: halve scale, skip this update
    """

    def __init__(self, init_scale: float = 65536.0,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 enabled: bool = True):
        self.scale            = init_scale
        self.growth_factor    = growth_factor
        self.backoff_factor   = backoff_factor
        self.growth_interval  = growth_interval
        self.enabled          = enabled
        self._good_steps      = 0    # consecutive steps without inf/nan

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return loss
        return loss * self.scale

    def unscale_(self, optimizer):
        """Divide all gradients by the current scale (in-place)."""
        if not self.enabled:
            return
        inv_scale = 1.0 / self.scale
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(inv_scale)

    def _check_inf_nan(self, optimizer) -> bool:
        """Return True if any gradient is inf or nan."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        return True
        return False

    def step(self, optimizer) -> bool:
        """
        Perform the optimizer step if gradients are finite.
        Returns True if the step was taken, False if skipped.
        """
        if not self.enabled:
            optimizer.step()
            return True

        if self._check_inf_nan(optimizer):
            # Gradients contain inf/nan — skip update and shrink scale
            self.scale *= self.backoff_factor
            self._good_steps = 0
            return False
        else:
            optimizer.step()
            self._good_steps += 1
            return True

    def update(self):
        """Grow scale if we've had enough consecutive good steps."""
        if not self.enabled:
            return
        if self._good_steps >= self.growth_interval:
            self.scale *= self.growth_factor
            self._good_steps = 0

    def get_scale(self) -> float:
        return self.scale


# ── Verify against PyTorch ────────────────────────────────────────────────────

def verify_grads_match(model_a, model_b):
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        if pa.grad is not None and pb.grad is not None:
            diff = (pa.grad.float() - pb.grad.float()).abs().max().item()
            if diff > 1e-3:
                return False, diff
    return True, 0.0


if __name__ == "__main__":
    underflow_demo()

    print()
    print("=" * 62)
    print("  GRADSCALER: MINI vs PYTORCH VERIFICATION")
    print("=" * 62)
    print()

    # Only run CUDA demo if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  (Running on CPU — fp16 GradScaler works but is unusual)")
        print("  Demonstrating logic without actual cuda autocast...")
        print()

    torch.manual_seed(0)
    model_a = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1))
    model_b = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1))
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        pb.data.copy_(pa.data)

    opt_a   = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    opt_b   = torch.optim.AdamW(model_b.parameters(), lr=1e-3)

    mini_scaler = MiniGradScaler(init_scale=256.0, growth_interval=10)
    pt_scaler   = torch.cuda.amp.GradScaler(
        init_scale=256.0, growth_interval=10, enabled=(device=="cuda")
    ) if device == "cuda" else None

    x = torch.randn(32, 16)
    y = torch.randn(32, 1)

    print(f"  {'Step':>4}  {'Mini scale':>12}  {'Status':>12}")
    print(f"  {'':─>4}  {'':─>12}  {'':─>12}")

    for step in range(1, 51):
        # Mini scaler
        opt_a.zero_grad()
        loss_a = F.mse_loss(model_a(x), y)
        (mini_scaler.scale_loss(loss_a)).backward()
        mini_scaler.unscale_(opt_a)
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
        took_step = mini_scaler.step(opt_a)
        mini_scaler.update()

        if step % 10 == 0 or step <= 3:
            status = "✓ updated" if took_step else "✗ skipped"
            print(f"  {step:>4}  {mini_scaler.get_scale():>12.0f}  {status:>12}")

    print()
    print(f"  Final scale: {mini_scaler.get_scale():.0f}")
    print()
    print("  Scale growth: every 10 good steps → scale doubles")
    print("  Scale backoff: on inf/nan → scale halves, step skipped")
    print()

    # Show the scale trajectory
    print("  Scale trajectory (init=256, growth_interval=10):")
    s     = 256.0
    steps = []
    for i in range(1, 101):
        steps.append((i, s))
        s *= 2.0   # pretend every step succeeds
        if i % 10 == 0:
            # scale doubles
            pass
    # Print at doubling points
    for i, sc in [(10, 256*2), (20, 256*4), (30, 256*8), (50, 256*32), (100, 256*1024)]:
        actual = 256.0 * (2 ** (i // 10))
        print(f"    Step {i:>4}: scale ≈ {actual:>10.0f}")
    print()
    print("  If an inf/nan occurs at step 35:")
    print("    Step 35: scale halved → scale = 256 × 8 / 2 = 1024")
    print("    good_steps reset to 0 → need 10 more good steps to double again")
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
    #     from llm_training.visuals.mixed_precision import (
    #         MP_VISUAL_HTML,
    #         MP_VISUAL_HEIGHT,
    #     )
    #     visual_html   = MP_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = MP_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[10_mixed_precision_bf16_fp16.py] Could not load visual: {e}",
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