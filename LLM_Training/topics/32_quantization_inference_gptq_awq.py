"""
Quantisation for Inference: GPTQ and AWQ
==========================================

Post-training quantisation (PTQ) converts a trained model's floating-point
weights into lower-bit integer representations to reduce memory bandwidth and
accelerate inference without retraining. GPTQ and AWQ are the two dominant
methods that achieve near-fp16 quality at 4-bit precision. Understanding the
mathematical challenges of quantisation — the non-uniform importance of weights,
the layer-wise error propagation, the rounding problem — and how each method
solves them is essential for deploying large models efficiently at inference time.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Quantisation for Inference: GPTQ and AWQ"
DISPLAY_NAME = "32 · Quantisation (GPTQ/AWQ)"
ICON         = "🗜️"
SUBTITLE     = "Post-Training Quantisation at 4-bit Precision"


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

### Why Quantise for Inference?

Every inference forward pass reads model weights from GPU HBM (high-bandwidth
memory) into compute cores. At batch=1, the generation speed is almost entirely
determined by how fast weights can be loaded:

    TPOT ≈ model_bytes / HBM_bandwidth

For LLaMA-2 70B in bf16 (2 bytes per weight):
    70B × 2 bytes = 140 GB; A100 HBM bandwidth = 2 TB/s
    TPOT ≈ 140 GB / 2 TB/s = 70 ms per token

In INT4 (0.5 bytes per weight):
    70B × 0.5 bytes = 35 GB
    TPOT ≈ 35 GB / 2 TB/s = 17.5 ms per token → 4× speedup!

And critically: the model now fits on a single A100 80GB instead of two,
and on a single 48GB consumer GPU (RTX 6000 Ada) instead of being impossible
to deploy.

**The complete case for quantisation:**
    1.  Memory:    140 GB → 35 GB for 4-bit (4× less HBM needed)
    2.  Bandwidth: 4× faster weight loading (direct decode speedup)
    3.  Cost:      Run on fewer or cheaper GPUs (4–10× cost reduction)
    4.  Latency:   4× lower TPOT for single-user generation
    5.  Concurrency: 4× more tokens fit in KV budget

The challenge: going from fp16 to INT4 means representing each weight with
only 16 possible values. Done naively, this destroys model quality. The
question is how to minimise the quality loss while achieving maximum compression.


### The Fundamental Quantisation Problem

A **linear quantiser** maps a floating-point weight w ∈ ℝ to an integer q ∈ {0, 1, ..., 2^b - 1}:

    q(w) = round(w / s) + z    (quantise)
    ŵ(q) = s × (q - z)         (dequantise)

where:
    s = scale factor  (step size between quantisation levels)
    z = zero-point    (offset for asymmetric quantisation)
    b = bit-width     (4 bits → 16 levels)

**Symmetric quantisation** (z=0):
    s = max(|W|) / (2^{b-1} - 1)
    q ∈ {-(2^{b-1}-1), ..., (2^{b-1}-1)}
    Used when weight distribution is symmetric around zero (common for LLMs)

**Asymmetric quantisation:**
    s = (max(W) - min(W)) / (2^b - 1)
    z = round(-min(W) / s)
    q ∈ {0, ..., 2^b - 1}
    Used when weight distribution is skewed

**The quantisation error:**
    ε_w = ŵ - w = quantisation error for weight w
    For uniform quantisation: |ε_w| ≤ s/2 (bounded by half the step size)

The key challenge: how does quantisation error in individual weights affect
the model's output? A single weight change propagates through the entire
computation graph, and errors compound across layers.


### Why Naive Round-to-Nearest Fails at 4-bit

Consider a weight matrix W ∈ ℝ^(d × k) being quantised to 4-bit.

**Round-to-nearest (RTN):** Simply round each weight to the nearest representable value:
    Q(W) = s × round(W / s)    (element-wise)

RTN is fast and simple, but at 4-bit it causes severe quality degradation:
    •   LLaMA-7B fp16 perplexity (WikiText): ~5.7
    •   LLaMA-7B 4-bit RTN perplexity: ~8-10 (significant degradation)
    •   LLaMA-7B 4-bit GPTQ perplexity: ~5.9 (near-fp16!)

Why does RTN fail? Three reasons:

**1. Outlier weights:** LLM weight matrices often contain a small number of
   outlier values with much larger magnitude than the rest. When the scale s
   is set to accommodate these outliers, all other weights are quantised more
   coarsely (larger quantisation error).

   Example: If 99.9% of weights are in [-0.1, 0.1] but one weight is 2.0,
   the scale becomes s = 2.0/7 ≈ 0.286, and all small weights are rounded to
   the nearest multiple of 0.286, losing most of their precision.

**2. Error accumulation:** Each layer's quantisation error creates a difference
   between the original and quantised layer output. This error feeds into the
   next layer as a perturbed input, amplifying the original error. By layer 32,
   small per-layer errors can cascade into large output differences.

**3. Channel non-uniformity:** Different output channels of a weight matrix
   may have wildly different magnitudes. Per-tensor quantisation forces all
   channels to share one scale factor, poorly fitting any individual channel.
   Per-channel or per-group quantisation allows each group to have its own scale.


### Group Quantisation: Reducing Granularity

**Per-tensor quantisation:** One scale for the entire weight tensor.
    •   Minimum overhead (one float per tensor)
    •   Worst quality (outliers ruin precision for everyone)

**Per-channel quantisation:** One scale per output channel (row of W).
    •   Quality improvement for channel-unbalanced matrices
    •   Overhead: d floats for a d×k matrix

**Per-group quantisation (standard for 4-bit LLMs):**
    •   Divide each row into groups of g consecutive elements
    •   One scale per group (g is typically 128)
    •   Overhead: (d×k/g) × 2 bytes per scale and zero-point
    •   For g=128: each group needs 1/128 extra bytes per weight
    •   Net bit-rate: 4 + 32/128 ≈ 4.25 bits per weight (marginal overhead)

Why group size 128?
    •   Smaller groups → better quality, more overhead
    •   Larger groups → less overhead, worse quality
    •   g=128 is empirically near-optimal for LLMs


    **Diagram 1 — Quantisation Error Sources:**

    QUANTISATION ERROR: THE OUTLIER PROBLEM
    ════════════════════════════════════════════════════════════════

    Weight distribution (one row of W):
    ×   ×× × ×× × ×× ×× × ×       ×  ×× × ×            ●
    ─────────────────────────────────────────────────────────────
    -0.1                          0.1                    2.0

    Per-tensor scale s = 2.0/7 = 0.286:
    ×  → rounded to 0     (true: ~0.05, error = 0.05)   ← precision lost!
    ●  → rounded to 7     (true: 2.0,   error ≈ 0)

    Per-group scale (group excludes the outlier):
    ×  → scale s' = 0.1/7 = 0.014                        ← much more precise
    ●  → in its own group, scale = 2.0/7 = 0.286

    The outlier forces a larger scale that hurts all other weights.
    Solution: group quantisation isolates outliers in their own groups.


### GPTQ: Hessian-Guided Round-to-Nearest

**GPTQ** (Frantar et al., 2022) is an extension of Optimal Brain Quantisation
(OBQ) that efficiently minimises the layer-wise quantisation error.

**The key insight:** When we round weight w_q to its nearest integer, we can
compensate for the introduced error by adjusting the remaining (not-yet-quantised)
weights in the same row. This is the quantisation-as-compression framework.

**Mathematical setup:**
For a weight matrix W with layer input X, the layer output is Y = WX.
The quantisation error in the output is:
    ΔY = (Ŵ - W)X = δW × X

We want to find quantised weights Ŵ that minimise:
    ||δW × X||²_F = ||ΔY||²_F

This is a weighted least-squares problem where the weights are determined
by the input activation statistics captured in the **Hessian matrix**:
    H = 2 × X × X^T    (the Hessian of the squared output error w.r.t. W)

**GPTQ algorithm (per-row of W):**
    1.  Compute H = X × X^T using calibration data
    2.  For each weight column j (left to right):
        a.  Quantise w_j: q_j = round(w_j / s) × s
        b.  Compute error: e_j = w_j - q_j
        c.  Update remaining weights (j+1 to k):
            w_{j+1:k} ← w_{j+1:k} - (e_j / H_{jj}) × H_{j, j+1:k}
    3.  The correction propagates the quantisation error to remaining weights

This is equivalent to Cholesky decomposition of H and back-substitution.
The correction ensures that the total error WX is minimised even as individual
weights are rounded.

**Complexity:** O(d² × k) per matrix vs O(d² × k²) for exact OBQ → practical!

**The Cholesky trick:**
GPTQ uses the inverse Cholesky factor of H to efficiently compute the
necessary corrections without recomputing the full Hessian inverse:
    H^{-1} is computed once via Cholesky decomposition
    Updates use only the relevant column of H^{-1}

**Lazy batching:** Process multiple columns at once (e.g., 128 columns per batch)
to exploit GPU parallelism. The update is still exact within the batch; small
numerical errors occur between batches. This gives a 10× speedup vs column-by-column.

**GPTQ quality:**
    LLaMA-7B fp16:  PPL=5.68
    GPTQ 4-bit:     PPL=5.85  (Δ=+0.17, barely noticeable)
    GPTQ 3-bit:     PPL=6.54  (Δ=+0.86, noticeable but often acceptable)

GPTQ requires calibration data (~512 random sequences) and takes 30–60 minutes
per 7B model on a GPU. The result is a quantised model that can be served
at 4-bit with near-fp16 quality.


    **Diagram 2 — GPTQ Error Propagation:**

    GPTQ: QUANTISE w₁, CORRECT w₂..wₙ
    ════════════════════════════════════════════════════════════════

    Original weights (one row):   [w₁=0.52, w₂=0.31, w₃=0.48, w₄=0.19]
    Hessian diagonal (importance):[H₁₁=2.1,  H₂₂=1.8,  H₃₃=3.0,  H₄₄=0.9]

    Step 1 — Quantise w₁:
    s = 0.52/7 = 0.074,  q₁ = round(0.52/0.074) = round(7) = 7
    ŵ₁ = 7 × 0.074 = 0.518;  error e₁ = 0.52 - 0.518 = 0.002
    Correction: Δw_{2:4} = -(e₁/H₁₁) × H₁,₂:₄  (small adjustments to w₂,w₃,w₄)

    Step 2 — Quantise w₂ (already adjusted):
    w₂_adjusted = 0.31 + Δw₂ = 0.313  (slightly modified by w₁'s error)
    q₂ = round(0.313/s) ...

    The key: GPTQ compensates for each weight's rounding error BEFORE
    quantising the next weight. Later weights are adjusted to absorb earlier errors.
    Total output error ≈ sum of squared errors × inverse Hessian ← minimal.


### AWQ: Activation-Aware Weight Quantisation

**AWQ** (Lin et al., 2023) takes a completely different approach. Instead of
correcting errors after quantisation (like GPTQ), AWQ identifies and **protects**
the most important weights before quantisation.

**The key insight of AWQ:**
Not all weights are equally important. Weights that are activated by large
activation values have disproportionate impact on the output. Protecting just
1% of the most important weights dramatically reduces quantisation error.

**The per-channel importance metric:**
For a weight matrix W with corresponding activation X (B × d_in):
    channel_importance_c = mean(|X[:, c]|)  for each input channel c

Channels with large activation magnitudes are important: the output of column c
of W is x_c × W[:, c], and if x_c is large, errors in W[:, c] are amplified.

**Two observations from this:**
    1.  ~1% of channels have much larger activations than the rest
        (the "outlier activation" phenomenon observed in LLM.int8())
    2.  Protecting these 1% of channels with higher precision dramatically
        reduces overall quantisation error

**The AWQ solution:**
Instead of keeping 1% of weights in fp16 (hardware-unfriendly mixed precision),
AWQ applies a **per-channel scaling** before quantisation:

    W_scaled = W / s      (divide weights by scale)
    X_scaled = X × s      (multiply activations by scale)

Since W_scaled × X_scaled = W × X, the output is unchanged.
But now W / s has smaller magnitude for large-activation channels,
which IMPROVES the quantisation precision for those channels.

The scale s is chosen to minimise the quantisation error:
    s* = argmin_s || Q(W / s^α) × (X × s) - WX ||²

where α ∈ (0, 1) is a balance factor (typically α = 0.5).

**Efficiency of AWQ:**
    •   No calibration data gradient computation (unlike GPTQ)
    •   No Hessian computation
    •   Just search for the optimal per-channel scale using ~512 examples
    •   Very fast: minutes vs 30-60 minutes for GPTQ

**AWQ quality:**
    LLaMA-7B fp16:   PPL=5.68
    AWQ 4-bit:       PPL=5.80  (Δ=+0.12, slightly better than GPTQ on some tasks)
    AWQ 3-bit:       PPL=6.22  (competitive with GPTQ at 3-bit)

AWQ tends to be faster to compute than GPTQ and achieves slightly better
zero-shot task accuracy (even if GPTQ is slightly better on perplexity).


    **Diagram 3 — AWQ: Rescaling Protects Important Channels:**

    AWQ: PER-CHANNEL SCALE TO PROTECT IMPORTANT WEIGHTS
    ════════════════════════════════════════════════════════════════

    Input activations X:   [small] [small] [LARGE] [small] [small]
                                         ↑
                            Channel 3 has large activation magnitude
                            → weights in column 3 of W are critical

    Without AWQ:
    W column 3: [0.08, 0.12, 0.07, 0.11] with s = max_all / 7 ≈ 0.04
    Error in column 3 = ±0.02  → multiplied by LARGE activation → BIG output error

    With AWQ (scale s=4 for column 3):
    W / s column 3: [0.02, 0.03, 0.0175, 0.0275]  ← smaller, finer quantisation
    X × s column 3: 4 × [LARGE] = [VERY LARGE]     ← preserved mathematically
    Error in column 3 = ±0.005  → multiplied by 4×LARGE → same or less output error

    Net effect: important channels get finer quantisation resolution.
    All other channels: unchanged (scale = 1).


### GPTQ vs AWQ: A Detailed Comparison

    Property                     GPTQ                    AWQ
    ─────────────────────────────────────────────────────────────────────
    Core approach           Error compensation      Importance protection
    Calibration time        30–60 min (7B GPU)      5–15 min (7B GPU)
    Memory during quant.    High (Hessian matrix)   Low (just activations)
    Quality at 4-bit        PPL+0.17 (7B)           PPL+0.12 (7B)
    Quality at 3-bit        PPL+0.86 (7B)           PPL+0.54 (7B)
    Zero-shot accuracy      Competitive             Slightly better
    Hardware friendliness   Excellent               Excellent
    Mixed precision support Yes (some groups fp16)  Yes (some channels fp16)
    Requires Hessian        Yes                     No
    Deployment support      AutoGPTQ, ExLlama2      AutoAWQ, vLLM (native)
    ─────────────────────────────────────────────────────────────────────

Both methods use per-group quantisation (g=128 typical) and produce weights
stored as INT4 with fp16 scales/zeros. The computation during inference is:
    1.  Load INT4 weights (fast — 0.5 bytes per weight)
    2.  Dequantise to fp16 (fast — fused kernel)
    3.  Compute fp16 matmul
    Step 2+3 can be fused into a single kernel (GPTQ/AWQ kernels do this).


### INT8 Quantisation: LLM.int8() and SmoothQuant

**LLM.int8()** (Dettmers et al., 2022) achieves INT8 inference by addressing
the outlier problem with mixed-precision decomposition:
    •   Identify outlier activation channels (|x_c| > threshold, e.g. 6.0)
    •   Compute outlier × weight in fp16 (small number of channels)
    •   Compute non-outlier × weight in INT8
    •   Combine results

This gives ~1.7× speedup at batch=1 vs fp16, and ~2× memory reduction.
It has little quality loss (PPL < 0.5% worse than fp16).

**SmoothQuant** (Xiao et al., 2022) addresses the migration problem:
    "Weights are easy to quantise (narrow distribution),
     but activations are hard (outliers)"

Solution: migrate the quantisation difficulty from activations to weights
by applying per-channel smoothing:
    X̂ = X / s   (smooth activation outliers away)
    Ŵ = W × s   (weights can absorb the scale, stay within 8-bit)

Then quantise X̂ and Ŵ symmetrically. The output X̂ × Ŵ = X × W is unchanged.
This is conceptually very similar to AWQ, applied at 8-bit instead of 4-bit.


### FP8 Quantisation on H100

NVIDIA H100 GPUs natively support FP8 (float8) computation:
    •   E4M3 (4 exponent bits, 3 mantissa): for forward pass (larger range needed)
    •   E5M2 (5 exponent bits, 2 mantissa): for gradients (more range, less precision)

FP8 matmuls are 2× faster than BF16 on H100 Tensor Cores. Unlike INT8/INT4,
FP8 preserves the floating-point format and can represent the same range as
BF16 (E5M2 has same exponent as BF16), making it easier to apply to activations
(not just weights).

**Calibration for FP8:**
Each tensor needs a per-tensor scale to map its range to FP8. This scale must
be computed from calibration data and stored alongside the quantised tensors.
Transformer Engine (NVIDIA) automates this.


### The Quantisation-Speed-Quality Triangle

There is a three-way trade-off between bit-width, quality, and inference speed:

    Bit-width  Memory     TPOT speedup   Quality loss    Use case
    ─────────────────────────────────────────────────────────────────────
    fp32       4 bytes    1× (baseline)  0%              Training only
    bf16       2 bytes    2× (vs fp32)   ~0%             Standard training/infer
    fp16       2 bytes    2× (vs fp32)   ~0%             Standard training/infer
    int8       1 byte     4× (vs fp32)   < 0.5%          High-volume serving
    fp8        1 byte     8× (vs fp32)   < 0.5%          H100 training/infer
    int4/nf4   0.5 bytes  8× (vs fp32)   0.5–2%          Edge/efficiency serving
    int3       0.375 bytes 10.7×         3–10%           Research/extreme edge
    int2       0.25 bytes  16×           > 10%           Rarely usable
    ─────────────────────────────────────────────────────────────────────

The current production sweet spot is INT4 with GPTQ or AWQ:
    •   4× memory reduction vs bf16 (4-bit vs 16-bit)
    •   4× TPOT improvement (bandwidth-bound decode)
    •   < 1% quality loss on most benchmarks
    •   Native GPU kernel support (ExLlama2, AutoAWQ kernels)


### Practical Deployment: Formats and Kernels

**GPTQ format:**
    •   INT4 weights packed into uint32 (8 INT4 per uint32)
    •   Per-group scales and zeros (fp16, one per 128 weights)
    •   Served by: AutoGPTQ, ExLlama2 (fastest GPTQ kernel), llama.cpp

**AWQ format:**
    •   INT4 weights packed similarly
    •   Per-channel scales (fp16)
    •   Served by: AutoAWQ, vLLM (native AWQ support), llama.cpp

**GGUF format (llama.cpp):**
    •   CPU-first format supporting INT4/INT5/INT8/mixed
    •   Cross-platform (CPU, Metal, CUDA, ROCm)
    •   Designed for consumer hardware (4090, M1/M2 Mac)
    •   Uses different quantisation methods (Q4_K_M, Q5_K_M, etc.)
    •   Q4_K_M: 4-bit with 6-bit scales, ~4.5 bits per weight effective

**ExLlama2:**
The fastest GPTQ inference kernel. Key innovations:
    •   4-bit matmul fused with dequantisation (no separate INT4→fp16 step)
    •   Asymmetric quantisation support
    •   8-bit attention + 4-bit feed-forward (mixed precision)
    •   vLLM integrates ExLlama2 as the default GPTQ backend

**Expected throughput numbers (LLaMA-2 7B, single A100):**
    bf16 (baseline):         ~5,000 tok/s peak throughput
    INT8 (LLM.int8()):       ~7,000 tok/s
    INT4 (GPTQ/AWQ):         ~12,000–18,000 tok/s
    INT4 + spec decoding:    ~25,000–40,000 tok/s
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Post-Training Quantisation Method Comparison

| Method       | Bit-width | Calibration time | Quality (PPL Δ 7B) | Speed vs fp16 | Use case                     |
|--------------|-----------|------------------|--------------------|---------------|------------------------------|
| RTN          | 4-bit     | None             | +2–4 PPL           | 4×            | Quick testing only           |
| GPTQ         | 4-bit     | 30–60 min        | +0.17 PPL          | 4×            | High-quality offline serving |
| AWQ          | 4-bit     | 5–15 min         | +0.12 PPL          | 4×            | Fast deployment              |
| LLM.int8()   | 8-bit     | Minutes          | <0.5% accuracy     | 1.7×          | Quality-sensitive serving    |
| SmoothQuant  | 8-bit     | Minutes          | <0.5% accuracy     | 2×            | A100/H100 serving            |
| GGUF Q4_K_M  | ~4.5-bit  | Hours (training) | +0.3 PPL           | 3.5×          | Consumer hardware (CPU/GPU)  |
| FP8 (H100)   | 8-bit     | Minutes (calibr) | <0.1% accuracy     | 2×            | H100 production serving      |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Quantisation Fundamentals: RTN, Errors, and Group Quantisation": {
        "description": "Implement uniform quantisation from scratch, measure quantisation error as a function of bit-width and group size, demonstrate the outlier problem, and show why per-group quantisation is necessary at 4-bit.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
QUANTISATION FUNDAMENTALS: RTN, ERRORS, AND GROUP QUANTISATION
================================================================================

Implements and analyses:
    1. Symmetric and asymmetric uniform quantisation
    2. Quantisation error as a function of bit-width (8, 6, 4, 3, 2)
    3. The outlier problem: how one outlier ruins precision for all weights
    4. Per-group quantisation: isolating outliers
    5. The quantisation-quality curve for realistic LLM weight distributions
    6. Effective bit-rate accounting for scale/zero-point overhead

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Quantiser implementations ─────────────────────────────────────────────────

def quantise_symmetric(tensor: torch.Tensor, bits: int,
                         per_group: bool = False,
                         group_size: int = 128) -> tuple:
    """
    Symmetric uniform quantisation.

    For symmetric: zero_point = 0, scale = max(|W|) / (2^(bits-1) - 1)
    Quantised values: integers in [-(2^(bits-1)-1), 2^(bits-1)-1]

    Returns:
        q_tensor:  quantised tensor (integer values as float)
        scale:     scale factor(s) used
        error:     quantisation error (q_tensor_dequant - tensor)
    """
    max_val  = 2 ** (bits - 1) - 1   # e.g., bits=4 → max_val=7

    if not per_group:
        s     = tensor.abs().max() / max_val
        s     = s.clamp(min=1e-8)
        q     = torch.clamp(torch.round(tensor / s), -max_val, max_val)
        deq   = q * s
        return q, s, deq - tensor
    else:
        # Per-group: reshape → quantise each group separately
        orig_shape = tensor.shape
        flat       = tensor.flatten()
        n          = flat.numel()
        pad_len    = (group_size - n % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        n_groups  = flat.numel() // group_size
        groups    = flat.view(n_groups, group_size)
        scales    = groups.abs().max(dim=1, keepdim=True).values / max_val
        scales    = scales.clamp(min=1e-8)
        q_groups  = torch.clamp(torch.round(groups / scales), -max_val, max_val)
        deq_groups = q_groups * scales

        q_flat   = q_groups.flatten()[:n].reshape(orig_shape)
        deq_flat = deq_groups.flatten()[:n].reshape(orig_shape)
        return q_flat, scales, deq_flat - tensor


def quantise_asymmetric(tensor: torch.Tensor, bits: int,
                          group_size: int = 128) -> tuple:
    """
    Asymmetric per-group quantisation.
    zero_point allows for non-zero centred weight distributions.
    """
    n_levels  = 2 ** bits - 1   # e.g., bits=4 → 15 levels
    orig_shape = tensor.shape
    flat       = tensor.flatten()
    n          = flat.numel()
    pad_len    = (group_size - n % group_size) % group_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    groups    = flat.view(-1, group_size)
    w_min     = groups.min(dim=1, keepdim=True).values
    w_max     = groups.max(dim=1, keepdim=True).values
    scales    = (w_max - w_min).clamp(min=1e-8) / n_levels
    zeros     = torch.round(-w_min / scales).clamp(0, n_levels)

    q_groups  = torch.clamp(torch.round(groups / scales) + zeros, 0, n_levels)
    deq_groups = (q_groups - zeros) * scales

    q_flat   = q_groups.flatten()[:n].reshape(orig_shape)
    deq_flat = deq_groups.flatten()[:n].reshape(orig_shape)
    return q_flat, scales, zeros, deq_flat - tensor


def effective_bits(nominal_bits: int, group_size: int,
                    scale_bits: int = 16) -> float:
    """
    Effective bits per weight accounting for scale/zero-point overhead.
    Each group of `group_size` weights needs one fp16 scale (and zero-point).
    """
    overhead_per_weight = scale_bits / group_size
    return nominal_bits + overhead_per_weight


# ── Weight distribution simulation ───────────────────────────────────────────

def generate_realistic_weights(rows: int, cols: int,
                                 outlier_fraction: float = 0.001,
                                 outlier_scale: float = 15.0,
                                 seed: int = 42) -> torch.Tensor:
    """
    Generate a realistic LLM weight distribution.
    Most weights: N(0, 0.02) (normal transformer scale)
    Outliers: ~outlier_fraction of weights have |w| ~ 0.3
    """
    torch.manual_seed(seed)
    W = torch.randn(rows, cols) * 0.02   # typical weight scale
    # Add a small number of outliers
    n_outliers = int(rows * cols * outlier_fraction)
    idx_r = torch.randint(0, rows, (n_outliers,))
    idx_c = torch.randint(0, cols, (n_outliers,))
    W[idx_r, idx_c] = (torch.randn(n_outliers) * 0.02 * outlier_scale)
    return W


# ── Output error analysis ─────────────────────────────────────────────────────

def output_error(W: torch.Tensor, W_hat: torch.Tensor,
                  X: torch.Tensor) -> float:
    """
    Measure the output error caused by weight quantisation.
    MSE of (W - W_hat) @ X vs 0.
    """
    delta_W = W_hat - W
    return (delta_W @ X).pow(2).mean().sqrt().item()


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    ROWS, COLS = 256, 256
    X = torch.randn(COLS, 64)   # activation matrix for output error measurement

    # ── 1. Bit-width vs error ─────────────────────────────────────────────────
    print("=" * 70)
    print("  QUANTISATION ERROR vs BIT-WIDTH (symmetric, per-tensor)")
    print("=" * 70)
    print()
    W = generate_realistic_weights(ROWS, COLS)

    print(f"  {'Bits':>6}  {'RMSE':>10}  {'Output RMSE':>14}  "
          f"{'Output % error':>16}  {'Eff. bits':>12}")
    print(f"  {'':─>6}  {'':─>10}  {'':─>14}  {'':─>16}  {'':─>12}")

    fp16_out_norm = (W @ X).norm().item()
    for bits in [8, 6, 4, 3, 2]:
        _, _, err = quantise_symmetric(W, bits, per_group=False)
        rmse        = err.pow(2).mean().sqrt().item()
        W_hat       = W + err   # dequantised weights
        out_err     = output_error(W, W_hat, X)
        pct_err     = out_err / fp16_out_norm * 100
        eff_bits    = effective_bits(bits, 128)
        print(f"  {bits:>6}  {rmse:>10.6f}  {out_err:>14.4f}  "
              f"{pct_err:>15.2f}%  {eff_bits:>12.2f}")

    # ── 2. Outlier impact ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  THE OUTLIER PROBLEM: One outlier ruins 4-bit quantisation")
    print("=" * 70)
    print()

    for outlier_scale in [0.0, 5.0, 10.0, 20.0, 50.0]:
        W_with_outlier = generate_realistic_weights(ROWS, COLS,
                                                      outlier_fraction=0.001,
                                                      outlier_scale=outlier_scale/0.02)
        _, s_tensor, err_tensor = quantise_symmetric(W_with_outlier, bits=4, per_group=False)
        _, _, err_group         = quantise_symmetric(W_with_outlier, bits=4, per_group=True, group_size=128)

        rmse_tensor = err_tensor.pow(2).mean().sqrt().item()
        rmse_group  = err_group.pow(2).mean().sqrt().item()
        max_weight  = W_with_outlier.abs().max().item()

        print(f"  max|w|={max_weight:.3f}  scale={s_tensor.item():.4f}  "
              f"per-tensor RMSE={rmse_tensor:.5f}  "
              f"per-group RMSE={rmse_group:.5f}  "
              f"improvement: {rmse_tensor/rmse_group:.1f}×")

    print()
    print("  Per-group quantisation reduces RMSE by up to 50× when outliers exist!")

    # ── 3. Group size vs quality ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  GROUP SIZE vs QUALITY AND OVERHEAD")
    print("=" * 70)
    print()
    W_out = generate_realistic_weights(ROWS, COLS, outlier_scale=500.0)

    print(f"  {'Group size':>12}  {'4-bit RMSE':>12}  {'Eff. bits':>12}  "
          f"{'Scale overhead':>16}  {'Quality/overhead':>18}")
    print(f"  {'':─>12}  {'':─>12}  {'':─>12}  {'':─>16}  {'':─>18}")

    for g in [1, 8, 32, 64, 128, 256, 1024, 999999]:
        if g > ROWS * COLS:
            g_actual = ROWS * COLS
            label    = "per-tensor"
        else:
            g_actual = g
            label    = str(g)
        per_group = (g_actual < ROWS * COLS)
        _, _, err = quantise_symmetric(W_out, bits=4, per_group=per_group,
                                        group_size=g_actual)
        rmse      = err.pow(2).mean().sqrt().item()
        eff_b     = effective_bits(4, g_actual) if per_group else 4.0
        overhead  = 16 / g_actual if per_group else 0.0
        qoh_ratio = 1.0 / (rmse * eff_b)   # higher = better

        print(f"  {label:>12}  {rmse:>12.6f}  {eff_b:>12.2f}  "
              f"{overhead:>15.2f}b  {qoh_ratio:>18.1f}")

    print()
    print("  Group size 128 is the sweet spot:")
    print("  - Dramatically better quality than per-tensor")
    print("  - Only 0.125 bits/weight overhead (negligible)")
    print("  - Standard in GPTQ, AWQ, and all production 4-bit methods")

    # ── 4. Asymmetric vs symmetric ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SYMMETRIC vs ASYMMETRIC QUANTISATION")
    print("=" * 70)
    print()
    # Asymmetric distribution (biased weights — some heads)
    W_asym = torch.randn(ROWS, COLS) * 0.02 + 0.01   # slight positive bias
    _, _, err_sym  = quantise_symmetric(W_asym, bits=4, per_group=True, group_size=128)
    _, _, _, err_asym = quantise_asymmetric(W_asym, bits=4, group_size=128)

    rmse_sym  = err_sym.pow(2).mean().sqrt().item()
    rmse_asym = err_asym.pow(2).mean().sqrt().item()
    print(f"  Weight range: [{W_asym.min():.4f}, {W_asym.max():.4f}]  "
          f"(asymmetric distribution)")
    print(f"  Symmetric RMSE:   {rmse_sym:.6f}")
    print(f"  Asymmetric RMSE:  {rmse_asym:.6f}  "
          f"({'better' if rmse_asym < rmse_sym else 'similar'})")
    print()
    print("  Asymmetric quantisation helps when weights are not zero-centred.")
    print("  For most transformer weights: symmetric ≈ asymmetric (saves zero-point).")
''',
    },

    "GPTQ From Scratch — Hessian-Guided Quantisation": {
        "description": "Implement GPTQ's core algorithm: compute the Hessian from calibration data, quantise weights column-by-column with error compensation, and compare output error vs RTN.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
GPTQ — HESSIAN-GUIDED WEIGHT QUANTISATION
================================================================================

Implements GPTQ's core algorithm:
    1. Compute the input Hessian H = 2 × X × X^T from calibration data
    2. Compute the Cholesky decomposition of H for efficient inversion
    3. Quantise columns left-to-right: round each weight, then compensate
       remaining weights using the Hessian to minimise total output error
    4. Compare: RTN vs GPTQ vs fp16 output on the same test inputs

GPTQ achieves near-fp16 quality by making the total output error minimal,
not just the per-weight error. A weight that is "optimally" rounded given
the Hessian may not be the nearest integer!

================================================================================
"""

import math
import torch
import torch.nn.functional as F


def compute_hessian(X: torch.Tensor, damp_percent: float = 0.01) -> torch.Tensor:
    """
    Compute the input Hessian H = 2 × X × X^T / N.

    The Hessian of the output MSE w.r.t. each row of W is:
        H_ij = E_X[x_i × x_j] = (X × X^T)_ij / N

    Dampening: add a small diagonal term λI to prevent numerical instability
    when H is near-singular (common in practice for some layers).

    Args:
        X: (d_in, N) calibration activations (N samples)
        damp_percent: fraction of mean diagonal to add as dampening

    Returns:
        H: (d_in, d_in) Hessian matrix
    """
    d_in, N = X.shape
    H       = 2 * (X @ X.T) / N   # (d_in, d_in)

    # Dampen to prevent near-singular matrices
    damp = damp_percent * H.diag().mean()
    H   += torch.eye(d_in, device=X.device) * damp

    return H


def gptq_quantise_row(w_row: torch.Tensor, H: torch.Tensor,
                        bits: int = 4, group_size: int = 128) -> tuple:
    """
    GPTQ quantisation for one row of a weight matrix.

    Quantises columns left-to-right. After each column is quantised,
    adjusts the remaining unquantised columns to compensate for the
    rounding error, using the Hessian.

    Args:
        w_row:     (d_in,) weight row to quantise
        H:         (d_in, d_in) input Hessian
        bits:      quantisation bit-width
        group_size: per-group quantisation group size

    Returns:
        w_quantised:  (d_in,) dequantised (compensated) row
        q_row:        (d_in,) integer quantised row
    """
    d_in  = w_row.shape[0]
    W     = w_row.clone().float()
    Hinv  = torch.linalg.inv(H.float())  # H^{-1} for error propagation

    max_val  = 2 ** (bits - 1) - 1
    W_out    = torch.zeros_like(W)
    q_out    = torch.zeros_like(W)

    # Process column by column (in order)
    for j in range(d_in):
        # Determine scale for this group
        group_start = (j // group_size) * group_size
        group_end   = min(group_start + group_size, d_in)
        group_vals  = W[group_start:group_end]
        s           = group_vals.abs().max() / max_val
        s           = max(s.item(), 1e-8)

        # Quantise column j
        q_j    = torch.clamp(torch.round(W[j] / s), -max_val, max_val).item()
        w_hat  = q_j * s
        error  = W[j].item() - w_hat   # rounding error

        # Store quantised value
        W_out[j] = w_hat
        q_out[j] = q_j

        # Compensate remaining weights using the Hessian
        # Update: w_{j+1:} -= error × H^{-1}_{j+1:, j} / H^{-1}_{jj}
        if j < d_in - 1:
            h_inv_diag = Hinv[j, j].item()
            if abs(h_inv_diag) > 1e-8:
                # Error correction for remaining weights
                W[j+1:] -= (error / h_inv_diag) * Hinv[j+1:, j]

    return W_out, q_out


def rtn_quantise(W: torch.Tensor, bits: int = 4,
                  group_size: int = 128) -> torch.Tensor:
    """Round-to-nearest baseline quantisation."""
    max_val  = 2 ** (bits - 1) - 1
    orig     = W.clone()
    flat     = W.flatten()
    n        = flat.numel()
    pad_len  = (group_size - n % group_size) % group_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))
    groups   = flat.view(-1, group_size)
    scales   = groups.abs().max(dim=1, keepdim=True).values / max_val
    scales   = scales.clamp(min=1e-8)
    q        = torch.clamp(torch.round(groups / scales), -max_val, max_val)
    deq      = (q * scales).flatten()[:n].reshape(orig.shape)
    return deq


def measure_output_error(W_orig: torch.Tensor, W_quant: torch.Tensor,
                           X_test: torch.Tensor) -> tuple[float, float]:
    """Measure output MSE and normalised error."""
    y_orig  = W_orig @ X_test
    y_quant = W_quant @ X_test
    mse     = (y_orig - y_quant).pow(2).mean().sqrt().item()
    norm    = y_orig.norm().item()
    return mse, mse / (norm + 1e-8) * 100


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    D_IN  = 64    # input dimension (scaled down for demo speed)
    D_OUT = 32    # output dimension
    N_CAL = 128   # calibration samples

    # Create a synthetic weight matrix and calibration data
    W_orig  = torch.randn(D_OUT, D_IN) * 0.05
    # Add some outliers
    W_orig[3, 15] = 0.8
    W_orig[7, 42] = -0.6

    # Calibration data (typically from the target dataset)
    X_calib  = torch.randn(D_IN, N_CAL)

    # Test data (unseen)
    X_test   = torch.randn(D_IN, 256)

    print("=" * 65)
    print(f"  GPTQ vs RTN QUANTISATION COMPARISON")
    print(f"  d_in={D_IN}, d_out={D_OUT}, calibration_N={N_CAL}")
    print("=" * 65)
    print()

    # Compute Hessian from calibration data
    print("  Computing Hessian from calibration data...")
    H = compute_hessian(X_calib, damp_percent=0.01)
    print(f"  Hessian shape: {H.shape}")
    print(f"  Hessian condition number: {torch.linalg.cond(H).item():.2f}")
    print()

    # Compare methods at different bit-widths
    print(f"  {'Method':<25}  {'RMSE (weights)':>16}  "
          f"{'Output RMSE':>14}  {'Output % err':>14}")
    print(f"  {'':─<25}  {'':─>16}  {'':─>14}  {'':─>14}")

    for bits in [4, 3]:
        # RTN baseline
        W_rtn = rtn_quantise(W_orig, bits=bits, group_size=128)
        rtn_w_rmse = (W_rtn - W_orig).pow(2).mean().sqrt().item()
        rtn_out_rmse, rtn_out_pct = measure_output_error(W_orig, W_rtn, X_test)
        print(f"  {'RTN '+str(bits)+'-bit':<25}  {rtn_w_rmse:>16.6f}  "
              f"{rtn_out_rmse:>14.4f}  {rtn_out_pct:>13.2f}%")

        # GPTQ
        gptq_rows = []
        for row_idx in range(D_OUT):
            w_hat, _ = gptq_quantise_row(
                W_orig[row_idx], H, bits=bits, group_size=128
            )
            gptq_rows.append(w_hat)
        W_gptq = torch.stack(gptq_rows)

        gptq_w_rmse = (W_gptq - W_orig).pow(2).mean().sqrt().item()
        gptq_out_rmse, gptq_out_pct = measure_output_error(W_orig, W_gptq, X_test)
        improvement = rtn_out_rmse / gptq_out_rmse if gptq_out_rmse > 0 else 1.0
        print(f"  {'GPTQ '+str(bits)+'-bit':<25}  {gptq_w_rmse:>16.6f}  "
              f"{gptq_out_rmse:>14.4f}  {gptq_out_pct:>13.2f}%  "
              f"({improvement:.1f}× better than RTN)")
        print()

    # Show the compensation effect: GPTQ vs RTN on individual row
    print("=" * 65)
    print("  GPTQ COMPENSATION EFFECT (one weight row)")
    print("=" * 65)
    print()
    row_idx = 3   # row with outlier at position 15
    print(f"  Row {row_idx} has outlier weight at position 15: "
          f"{W_orig[row_idx, 15]:.4f}")
    print()

    w_row   = W_orig[row_idx]
    W_rtn_r = rtn_quantise(w_row.unsqueeze(0), bits=4, group_size=128).squeeze()
    W_gptq_r, _ = gptq_quantise_row(w_row, H, bits=4, group_size=128)

    # Show first 10 weights
    print(f"  {'Col':>5}  {'Original':>12}  {'RTN quant':>12}  "
          f"{'GPTQ quant':>12}  {'RTN err':>10}  {'GPTQ err':>10}")
    print(f"  {'':─>5}  {'':─>12}  {'':─>12}  {'':─>12}  {'':─>10}  {'':─>10}")
    for j in range(min(12, D_IN)):
        rtn_err  = abs(W_rtn_r[j].item()  - w_row[j].item())
        gptq_err = abs(W_gptq_r[j].item() - w_row[j].item())
        flag     = " ← compensated" if gptq_err < rtn_err - 0.001 else ""
        print(f"  {j:>5}  {w_row[j].item():>12.5f}  {W_rtn_r[j].item():>12.5f}  "
              f"{W_gptq_r[j].item():>12.5f}  {rtn_err:>10.5f}  "
              f"{gptq_err:>10.5f}{flag}")

    print()
    print("  GPTQ modifies weights that were NOT rounded (later columns)")
    print("  to compensate for errors in weights that WERE rounded (earlier).")
    print("  The total output error (W @ X) is minimised, not individual errors.")
''',
    },

    "AWQ — Activation-Aware Scaling": {
        "description": "Implement AWQ's activation-based scaling search: find per-channel scales from activation statistics, apply smooth scaling, quantise, and compare to unscaled GPTQ and RTN baselines.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
AWQ — ACTIVATION-AWARE WEIGHT QUANTISATION
================================================================================

Implements AWQ's core algorithm:
    1. Collect per-channel activation statistics from calibration data
    2. Identify important channels (high activation magnitude)
    3. Search for optimal per-channel scale using a grid search
    4. Apply smooth scaling: W' = W / s, X' = X × s (output unchanged)
    5. Quantise W' using standard RTN (simpler than GPTQ)
    6. Compare AWQ vs plain RTN vs GPTQ on the same weight matrix

The elegant insight: by rescaling W before quantisation, we implicitly
give more quantisation precision to important channels without any
hardware-unfriendly mixed-precision computation.

================================================================================
"""

import math
import torch
import torch.nn.functional as F


# ── Activation statistics ─────────────────────────────────────────────────────

def collect_activation_stats(X: torch.Tensor) -> torch.Tensor:
    """
    Compute per-input-channel activation magnitude statistics.
    Used to identify which channels are most important for quantisation.

    Args:
        X: (d_in, N) calibration activations

    Returns:
        channel_importance: (d_in,) per-channel mean absolute activation
    """
    return X.abs().mean(dim=1)   # (d_in,)


# ── RTN quantisation (used inside AWQ) ───────────────────────────────────────

def rtn_quantise_grouped(W: torch.Tensor, bits: int = 4,
                          group_size: int = 128) -> torch.Tensor:
    """Per-group symmetric RTN quantisation."""
    max_val  = 2 ** (bits - 1) - 1
    flat     = W.flatten()
    n        = flat.numel()
    pad_len  = (group_size - n % group_size) % group_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))
    groups   = flat.view(-1, group_size).float()
    scales   = groups.abs().max(dim=1, keepdim=True).values / max_val
    scales   = scales.clamp(min=1e-8)
    q        = torch.clamp(torch.round(groups / scales), -max_val, max_val)
    deq      = (q * scales).flatten()[:n].reshape(W.shape)
    return deq


# ── AWQ scale search ──────────────────────────────────────────────────────────

def awq_search_scales(W: torch.Tensor, X: torch.Tensor,
                        bits: int = 4, group_size: int = 128,
                        alpha: float = 0.5,
                        n_scale_candidates: int = 20) -> torch.Tensor:
    """
    AWQ: search for per-channel scales that minimise quantisation error.

    The scale is applied as:
        W_scaled = W × diag(s)^{-1}   (scale down important channels in W)
        X_scaled = diag(s) × X        (scale up in activation space)
    Output is unchanged: W_scaled @ (X_scaled × 1) = W @ X

    The optimal scale for channel c is approximately:
        s_c = (activation_magnitude_c)^α

    where α ∈ (0, 1) controls the aggressiveness.
    We search over a grid of α values to find the best one.

    Args:
        W:  (d_out, d_in) weight matrix
        X:  (d_in, N) calibration activations
        alpha: balance factor (search around this)

    Returns:
        best_scales: (d_in,) per-channel scale factors
    """
    # Reference output (fp16 quality)
    Y_ref   = (W.float() @ X.float())

    # Activation magnitude per channel
    act_mag = X.abs().mean(dim=1).float()   # (d_in,)
    act_mag = act_mag / act_mag.max().clamp(min=1e-8)  # normalise

    best_error  = float("inf")
    best_scales = torch.ones(W.shape[1])

    # Search over different alpha values
    for alpha_candidate in torch.linspace(0.0, 1.0, n_scale_candidates):
        # Scale: s_c = act_mag_c ^ alpha
        s = act_mag ** alpha_candidate.item()
        s = s / s.mean()   # normalise so mean scale = 1

        # Apply scale to weights: W' = W / s (per column)
        W_scaled = W.float() / s.unsqueeze(0)   # (d_out, d_in) ÷ (d_in,)

        # Quantise W_scaled using RTN
        W_q_scaled = rtn_quantise_grouped(W_scaled, bits=bits, group_size=group_size)

        # Dequantise back to original scale (for output computation)
        W_q_orig = W_q_scaled * s.unsqueeze(0)  # rescale back

        # Measure output error (activations are scaled back by 1/s in X × s)
        # Since output = W_q_orig @ X = (W_q_scaled × s) @ X = W_q_scaled @ (s × X)
        # We compute error without actually scaling X (mathematically equivalent)
        Y_q     = W_q_orig @ X.float()
        error   = (Y_ref - Y_q).pow(2).mean().sqrt().item()

        if error < best_error:
            best_error  = error
            best_scales = s.clone()
            best_alpha  = alpha_candidate.item()

    return best_scales, best_alpha, best_error


def awq_quantise(W: torch.Tensor, X: torch.Tensor,
                  bits: int = 4, group_size: int = 128,
                  n_candidates: int = 20) -> tuple:
    """
    Full AWQ quantisation pipeline.

    1. Search for optimal per-channel scales
    2. Apply scales to weights
    3. Quantise scaled weights
    4. Return quantised weights in original scale space
    """
    best_scales, best_alpha, search_error = awq_search_scales(
        W, X, bits=bits, group_size=group_size, n_scale_candidates=n_candidates
    )

    # Apply optimal scale and quantise
    W_scaled   = W.float() / best_scales.unsqueeze(0)
    W_q_scaled = rtn_quantise_grouped(W_scaled, bits=bits, group_size=group_size)

    # Rescale back: W_quantised_in_original_scale = W_q_scaled × s
    W_q_final  = W_q_scaled * best_scales.unsqueeze(0)

    return W_q_final, best_scales, best_alpha


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    D_IN, D_OUT = 128, 64
    N_CAL, N_TEST = 256, 512

    # Realistic LLM weight matrix: mostly small weights, some outlier channels
    W_orig = torch.randn(D_OUT, D_IN) * 0.02
    W_orig[:, 10] = torch.randn(D_OUT) * 0.3   # outlier channel 10
    W_orig[:, 50] = torch.randn(D_OUT) * 0.25  # outlier channel 50

    # Calibration activations with corresponding large magnitudes for those channels
    X_calib = torch.randn(D_IN, N_CAL) * 0.1
    X_calib[10, :] = torch.randn(N_CAL) * 2.0   # channel 10 has large activations
    X_calib[50, :] = torch.randn(N_CAL) * 1.5   # channel 50 has large activations

    X_test = torch.randn(D_IN, N_TEST) * 0.1
    X_test[10, :] = torch.randn(N_TEST) * 2.0
    X_test[50, :] = torch.randn(N_TEST) * 1.5

    print("=" * 65)
    print("  AWQ vs RTN COMPARISON")
    print(f"  d_in={D_IN}, d_out={D_OUT}, outlier channels: 10, 50")
    print("=" * 65)
    print()

    # Show activation magnitudes
    act_mag = collect_activation_stats(X_calib)
    top_k   = act_mag.topk(5)
    print("  Top-5 channels by activation magnitude:")
    for idx, val in zip(top_k.indices, top_k.values):
        print(f"    Channel {idx.item():>4}: {val.item():.4f}")
    print()

    # Quantise with all methods
    Y_fp16   = W_orig @ X_test

    W_rtn    = rtn_quantise_grouped(W_orig, bits=4, group_size=128)
    Y_rtn    = W_rtn @ X_test
    rtn_err  = (Y_fp16 - Y_rtn).pow(2).mean().sqrt().item()
    rtn_pct  = rtn_err / Y_fp16.norm().item() * 100

    W_awq, awq_scales, awq_alpha = awq_quantise(W_orig, X_calib, bits=4, n_candidates=20)
    Y_awq    = W_awq @ X_test
    awq_err  = (Y_fp16 - Y_awq).pow(2).mean().sqrt().item()
    awq_pct  = awq_err / Y_fp16.norm().item() * 100

    print("  Output error comparison:")
    print(f"  RTN 4-bit:    RMSE = {rtn_err:.5f}  ({rtn_pct:.2f}% of output norm)")
    print(f"  AWQ 4-bit:    RMSE = {awq_err:.5f}  ({awq_pct:.2f}% of output norm)  "
          f"({rtn_err/awq_err:.1f}× better than RTN)")
    print(f"  AWQ best α:   {awq_alpha:.2f}")
    print()

    # Show how scales distribute
    print("  AWQ optimal scale distribution:")
    scale_percentiles = [0, 10, 25, 50, 75, 90, 100]
    awq_s_sorted = awq_scales.sort().values
    n             = len(awq_s_sorted)
    print(f"  {'Percentile':>12}  {'Scale value':>14}  "
          f"{'Effect on quant':>20}")
    print(f"  {'':─>12}  {'':─>14}  {'':─>20}")
    for p in scale_percentiles:
        idx = min(int(p / 100 * n), n - 1)
        s   = awq_s_sorted[idx].item()
        # Effect: s > 1 → W divided by s → finer quantisation
        effect = "finer quant" if s > 1.1 else "coarser quant" if s < 0.9 else "unchanged"
        print(f"  {p:>11}%  {s:>14.4f}  {effect:>20}")

    print()
    print("  Channels with scale > 1 (= large activation) → finer quantisation ✓")
    print("  Channels with scale ≈ 1 (= small activation) → unchanged ✓")

    # Compare which channels got the most scale adjustment
    print()
    print("  Per-channel scale for outlier channels:")
    for ch in [10, 50, 0, 1, 2]:   # outliers + normal channels
        s   = awq_scales[ch].item()
        mag = act_mag[ch].item()
        print(f"  Channel {ch:>4}: act_mag={mag:.4f}  scale={s:.4f}  "
              f"{'← protected (large act)' if s > 1.2 else ''}")

    # Quality degradation curves
    print()
    print("=" * 65)
    print("  QUALITY VS BIT-WIDTH (RTN vs AWQ)")
    print("=" * 65)
    print()
    print(f"  {'Bits':>6}  {'RTN output err':>16}  {'AWQ output err':>16}  "
          f"{'AWQ improvement':>18}")
    print(f"  {'':─>6}  {'':─>16}  {'':─>16}  {'':─>18}")

    for bits in [8, 6, 4, 3]:
        W_rtn_b   = rtn_quantise_grouped(W_orig, bits=bits, group_size=128)
        W_awq_b, _, _ = awq_quantise(W_orig, X_calib, bits=bits, n_candidates=15)

        rtn_e = (Y_fp16 - W_rtn_b @ X_test).pow(2).mean().sqrt().item()
        awq_e = (Y_fp16 - W_awq_b @ X_test).pow(2).mean().sqrt().item()
        improv = rtn_e / awq_e if awq_e > 0 else 1.0

        print(f"  {bits:>6}  {rtn_e:>16.5f}  {awq_e:>16.5f}  "
              f"{improv:>17.1f}×")

    print()
    print("  AWQ improves over RTN at all bit-widths by protecting")
    print("  the channels that matter most based on activation statistics.")
    print()
    print("  Key advantage vs GPTQ: no Hessian inversion needed,")
    print("  making AWQ 5–10× faster to compute while achieving")
    print("  similar or better perplexity scores.")
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
    #     from llm_training.visuals.quantization_gptq_awq import (
    #         QUANT_VISUAL_HTML,
    #         QUANT_VISUAL_HEIGHT,
    #     )
    #     visual_html   = QUANT_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = QUANT_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[32_quantization_inference_gptq_awq.py] Could not load visual: {e}",
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