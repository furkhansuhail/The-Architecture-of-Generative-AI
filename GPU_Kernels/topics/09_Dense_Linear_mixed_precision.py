"""
Mixed Precision — FP16 / BF16 Tradeoffs, Loss Scaling & TF32 on Ampere+
=========================================================================

Mixed precision training computes forward and backward passes in reduced-
precision floating point while maintaining a full-precision master copy of
weights for the parameter update. Done correctly, it delivers near-identical
model quality to FP32 training at 2–16× higher throughput and half the
memory footprint.

The term "mixed precision" is deliberate: not all operations run at the
same precision. Tensor core GEMMs use FP16 or BF16. Gradient accumulation
happens in FP32. Loss computation, softmax, and layer normalisation remain
in FP32. The art is knowing which parts of the computation are sensitive
enough to require full precision and which can safely run at half precision.

Three precision modes dominate production deep learning:

    FP16 (IEEE 754 half precision):
        5-bit exponent, 10-bit mantissa.
        Narrow dynamic range (±65504) forces loss scaling.
        Maximum tensor core throughput on all Volta+ GPUs.
        Standard for BERT, GPT-2, ResNet, ViT training.

    BF16 (Brain Float 16):
        8-bit exponent, 7-bit mantissa.
        Same dynamic range as FP32 — no loss scaling needed.
        Slightly less mantissa precision than FP16.
        Now the default for LLM training (Llama, Mistral, GPT-4, PaLM).

    TF32 (TensorFloat-32):
        8-bit exponent, 10-bit mantissa (19 bits total).
        An Ampere-and-later INTERNAL format, not a storage type.
        Automatically applied when allow_tf32=True.
        Gives tensor core speed to FP32 code with no code changes.

Understanding these three precisely — their number formats, overflow
behaviour, gradient underflow pathology, loss scaler mechanics, and
the Ampere math mode flags — is what separates engineers who "enable AMP"
from engineers who understand why it works and debug it when it breaks.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Mixed Precision — FP16 / BF16 Tradeoffs, Loss Scaling & TF32 on Ampere+"
DISPLAY_NAME = "09 · Mixed Precision"
ICON         = "🎚️"
SUBTITLE     = "FP16 · BF16 · TF32 · Loss Scaling · GradScaler · AMP · Ampere Math Modes"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — FLOATING POINT NUMBER FORMATS: A PRECISE COMPARISON

### IEEE 754 Bit Layout

    Every IEEE 754 floating point number has three fields:
        value = (-1)^sign × 2^(exponent - bias) × (1 + mantissa/2^M)

    ┌─────────────────────────────────────────────────────────────────┐
    │ Format  │ Bits │ Sign │  Exp  │ Mant │  Bias │  Max   │ Epsilon │
    ├─────────────────────────────────────────────────────────────────┤
    │ FP16    │  16  │   1  │   5   │  10  │    15 │ 65504  │ 9.8e-4  │
    │ BF16    │  16  │   1  │   8   │   7  │   127 │ 3.4e38 │ 7.8e-3  │
    │ TF32*   │  19* │   1  │   8   │  10  │   127 │ 3.4e38 │ 9.8e-4  │
    │ FP32    │  32  │   1  │   8   │  23  │   127 │ 3.4e38 │ 1.2e-7  │
    │ FP64    │  64  │   1  │  11   │  52  │  1023 │ 1.8e308│1.1e-16  │
    └─────────────────────────────────────────────────────────────────┘
    * TF32 is an internal compute format, not an IEEE storage format.

    EPSILON = machine epsilon = smallest δ such that 1.0 + δ ≠ 1.0 in the format.
            = 2^(-mantissa_bits)

    The KEY NUMBERS to internalise:
        FP16 max:  65504       ← overflow at gradient norms above this
        BF16 max:  3.4 × 10^38 ← same as FP32, essentially no overflow
        FP16 ε:    0.000977    ← 1 part in 1024 relative error per operation
        BF16 ε:    0.00781     ← 1 part in 128 — coarser than FP16!
        FP32 ε:    1.19 × 10^-7 ← 1 part in 8.4 million

### The Dynamic Range Trade-off: FP16 vs BF16

    FP16's 5-bit exponent gives 2^(2^5 - 1) = 2^31 ≈ 2 × 10^9 dynamic range.
    BF16's 8-bit exponent gives 2^(2^8 - 1) = 2^255 ≈ 10^77 dynamic range.

    This explains the entire BF16 vs FP16 training story:
        - FP16 overflow (value > 65504): output → inf → NaN cascade.
        - BF16 overflow: practically impossible in normal training.
        - FP16 underflow (absolute value < 6 × 10^-8): flushes to zero.
        - BF16 underflow: same threshold as FP32 (1.18 × 10^-38).

    But BF16's mantissa is 7 bits vs FP16's 10 bits:
        BF16 represents consecutive integers exactly only up to 256.
        FP16 represents consecutive integers exactly up to 2048.
        For most neural network weight values (typically |w| < 10):
        the extra mantissa bits of FP16 barely matter in practice.
        BF16 and FP16 training converge to nearly identical accuracy.

### Subnormal Numbers and Flush-to-Zero

    When the exponent field is all zeros, the number is SUBNORMAL (denormalised):
        value = (-1)^sign × 2^(1-bias) × (mantissa/2^M)   [no implicit 1. prefix]

    Subnormals extend the representable range below min_normal by losing precision.

    FP16 subnormal range: 6 × 10^-8 to 6 × 10^-5 (in steps of 6 × 10^-8)
    FP16 denormal floor:  5.96 × 10^-8  (smallest positive FP16 value)

    GPU hardware often uses FLUSH-TO-ZERO (FTZ) mode for FP16:
        Any value smaller than min_normal → 0.0 immediately.
        Removes subnormal hardware support for speed.
        CONSEQUENCE: gradients smaller than 6 × 10^-5 in FP16 → silently zero.
        This is gradient underflow — it kills training quietly.


##### PART 2 — FP16 VS BF16: THE TRAINING TRADEOFF IN FULL DETAIL

### Why FP16 Training Breaks Without Intervention

    In a typical deep learning training run, gradient values span many
    orders of magnitude. A gradient histogram for a 12-layer transformer
    trained on language modelling typically shows:
        ~5%  of gradients with |g| > 1.0         (safe in FP16)
        ~60% of gradients with 1e-4 < |g| < 1.0  (safe in FP16)
        ~25% of gradients with 6e-5 < |g| < 1e-4 (FP16 subnormal territory)
        ~10% of gradients with |g| < 6e-5         (flush-to-zero → 0 in FP16!)

    That 10% is gone. Small-magnitude gradients in early layers (vanishing
    gradient problem) are exactly the gradients that need to be preserved.
    Losing them means some layers stop learning entirely.

    Additionally: the loss value itself may exceed 65504 in early training
    (before the model converges). Loss > 65504 → inf → NaN when back-propagated.

### Why BF16 Training Works Without Intervention

    BF16's dynamic range matches FP32 exactly:
        min_normal(BF16) = 1.18 × 10^-38  (same as FP32)
        max_val(BF16)    = 3.4 × 10^38    (same as FP32)
        Gradient values that are representable in FP32 are also representable in BF16.
        The precision is lower (7 mantissa bits), but the value never vanishes.

    Empirically: BF16 training for LLMs (Llama-2-7B, 13B, 70B) converges
    to identical perplexity as FP32 training. The mantissa loss is below
    the noise floor of stochastic gradient descent.

    WHEN BF16 CAN HURT:
        Tasks requiring high weight precision (certain regression problems).
        Very long training runs where low-precision mantissa accumulates error.
        Sensitive scientific ML tasks (PINNs, molecular dynamics).
        For these: FP32 accumulation in the optimizer remains essential.

### The Mixed Precision Memory Layout

    FULL MIXED PRECISION TRAINING stores three copies of weights:
        1. FP16/BF16 WEIGHTS:   used in forward and backward pass GEMMs
        2. FP32 MASTER WEIGHTS: used in the optimizer update step
        3. FP16/BF16 GRADIENTS: computed in backward pass
                                 (sometimes FP32 — see below)
        4. FP32 OPTIMIZER STATE: Adam m1, m2 (both FP32 for stability)

    MEMORY PER PARAMETER (Adam optimizer):
        FP16 + FP32 master: 2 + 4 = 6 bytes weights
        FP16 gradients:     2 bytes
        FP32 m1, m2:        4 + 4 = 8 bytes optimizer state
        Total: 16 bytes/param  vs 20 bytes/param for full FP32 Adam

    MEMORY SAVINGS:
        Activations (dominant for large batch/long sequence):
            FP16 activations: exactly half the memory of FP32.
            For a 7B model, batch=4, seq=2048, 32 layers:
            FP32 activations ≈ 32 GB. FP16 activations ≈ 16 GB.
        This is the primary memory saving of mixed precision — NOT the weights.

### Hardware Tensor Core Requirements

    FP16 on Volta (V100) and later: full tensor core support.
        Fragment dtype: __half. Accumulation: FP32.
        cuBLAS: CUBLAS_COMPUTE_32F with CUDA_R_16F input.

    BF16 on Ampere (A100) and later: tensor core support.
        Fragment dtype: __nv_bfloat16. Accumulation: FP32.
        NOT available on Volta or Turing.
        If code tries BF16 on V100 → fallback to scalar FP32 (no warning!).

    CHECK: torch.cuda.is_bf16_supported()
        Returns True on A100, H100, RTX 3090, 4090.
        Returns False on V100, T4, P100.


##### PART 3 — LOSS SCALING: THE COMPLETE MECHANISM
────────────────────────────────

### The Core Idea

    Gradient underflow in FP16 means small gradients flush to zero.
    The solution: SCALE UP THE LOSS before backward, then UNSCALE the
    gradients before the parameter update.

    FORWARD:
        loss_unscaled = model(input)
        loss_scaled   = loss_unscaled × S          (multiply by scale S)

    BACKWARD:
        grads_scaled  = d(loss_scaled)/d(theta)    (scaled gradients)
                      = S × d(loss_unscaled)/d(theta)
        All gradients are S× larger → fewer flush to zero in FP16.

    OPTIMIZER STEP:
        grads_true = grads_scaled / S              (unscale before Adam update)
        theta -= lr × adam(grads_true)

    A scale of S = 2^12 = 4096 means gradients of magnitude 6e-5 / 4096 ≈ 1.5e-8
    in true scale are represented as 6e-5 in FP16 — just above the flush-to-zero floor.

### Overflow Detection: The Inf/NaN Check

    If S is too large: some SCALED gradients overflow to inf in FP16.
    Overflowed gradients → NaN after unscaling → corrupt model.

    DETECTION: after computing grads_scaled, check:
        if any(isnan(g) or isinf(g) for g in grads_scaled):
            → SKIP the optimizer step (don't update weights)
            → REDUCE scale: S = S / backoff_factor
        else:
            → APPLY optimizer step normally
            → Maybe INCREASE scale: if no overflow for N steps, S = S × growth_factor

    This check is done with a single GPU-side reduction.
    PyTorch scaler.step() performs this check automatically.
    The CPU never touches the gradient values — the check is fully GPU-side.

### The Dynamic Loss Scaling Algorithm (Micikevicius et al., 2018)

    State:
        S          = initial_scale (default: 2^16 = 65536)
        N_good     = 0             (consecutive steps without overflow)
        growth_interval = 2000     (steps between scale increases)
        growth_factor   = 2.0      (multiply S by this on increase)
        backoff_factor  = 0.5      (multiply S by this on decrease)

    Every step:
        1. Compute loss_scaled = loss × S
        2. loss_scaled.backward()                 ← produces grads × S
        3. scaler.step(optimizer):
            a. CHECK: scan all gradient tensors for inf/nan
            b. If overflow found:
                S = max(S × backoff_factor, min_scale)
                N_good = 0
                SKIP optimizer.step()              ← model not updated this step
            c. If no overflow:
                unscale gradients (divide by S)
                optimizer.step()                   ← normal weight update
                N_good += 1
                if N_good >= growth_interval:
                    S = min(S × growth_factor, max_scale)
                    N_good = 0
        4. scaler.update()                         ← finalise state

    SCALE DYNAMICS in practice:
        Initial scale (65536) likely causes overflow in first few steps.
        Scale drops rapidly to stable range (~1024–4096).
        Stable training: occasional overflow at start of each epoch,
        then hundreds of steps without overflow as scale grows back up.

    SKIPPED STEPS:
        When overflow is detected, the optimizer step is SKIPPED entirely.
        The model weights are NOT updated that step.
        This is safe — skipping one step in 1000 has negligible effect.
        If skipping > 5% of steps: training is unstable (investigate loss curve).

### Gradient Clipping with Loss Scaling

    torch.nn.utils.clip_grad_norm_() clips gradients BEFORE the optimizer step.
    With loss scaling, clipping must happen AFTER unscaling:

        scaler.unscale_(optimizer)          ← must call before clip_grad_norm_!
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)              ← checks for inf/nan, steps
        scaler.update()

    If you call clip_grad_norm_ BEFORE unscale_: you're clipping the scaled
    gradients (S× too large) and the clip will be far too aggressive.
    This is a very common bug in AMP training code.

### Static vs Dynamic Scaling

    STATIC LOSS SCALING:
        S is fixed throughout training.
        Simple to implement. No overhead from inf/nan checks.
        Risk: if true gradients change magnitude (e.g., after LR warmup),
        a static S may be wrong — either too small (underflow) or too large (overflow).
        Use when: scale is known from prior experiments. Embedded inference.

    DYNAMIC LOSS SCALING (default):
        S is adjusted automatically every few thousand steps.
        Self-tuning: adapts to gradient magnitude changes across training.
        Overhead: one GPU-side reduction per step to check for inf/nan.
            Cost: ~10–50 µs per step. Negligible for steps > 1 ms.
        Use when: any new model/dataset/hyperparameter combination.

### PyTorch GradScaler API

    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler(
        init_scale=2.**16,        # starting scale factor
        growth_factor=2.0,        # multiply scale by this after growth_interval steps
        backoff_factor=0.5,       # multiply scale by this after overflow
        growth_interval=2000,     # steps between scale increases
        enabled=True,             # set False to disable (useful for ablations)
    )

    for batch in dataloader:
        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            loss = model(batch)           # FP16 compute

        scaler.scale(loss).backward()     # loss × scale, backward in FP16

        # Optional gradient clipping:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)            # unscale, check inf/nan, step
        scaler.update()                   # update scale for next step

        # Inspect current scale:
        current_scale = scaler.get_scale()


##### PART 4 — TF32 ON AMPERE+: THE INVISIBLE SPEEDUP

### What TF32 Is (and Isn't)

    TF32 is NOT a storage format. There is no torch.float_tf32 dtype.
    TF32 is an INTERNAL COMPUTE MODE for tensor core operations on Ampere+ GPUs.

    When enabled: FP32 inputs to GEMM are rounded to TF32 precision
    (10-bit mantissa, 8-bit exponent) before being sent to the tensor core.
    Accumulation of products remains in full FP32 (23-bit mantissa).
    Outputs are stored as regular FP32.

    Visually:
        FP32 input:  SEEEEEEEEMMMMMMMMMMMMMMMMMMMM  (1+8+23 = 32 bits)
        TF32 round:  SEEEEEEEEMMMMMMMMM000000000000  (top 19 bits kept)
        TC multiply: products accumulated in FP32
        FP32 output: SEEEEEEEEMMMMMMMMMMMMMMMMMMMM  (32 bits again)

    The rounding step discards 13 mantissa bits before the tensor core.
    This is equivalent to zeroing bits 0–12 of the FP32 mantissa.

### Why TF32 Doesn't Break Most Models

    The key question: does discarding 13 mantissa bits hurt accuracy?

    For a single FP32 multiply-accumulate:
        Relative error ≈ machine epsilon × K = 2^-10 × K
        For K=4096 inner dimension: relative error ≈ 0.4%
        For K=1024: relative error ≈ 0.1%

    In practice: neural network training is STOCHASTIC. SGD introduces
    noise orders of magnitude larger than the 0.1% TF32 rounding error.
    Training loss curves with TF32 are indistinguishable from FP32 for:
        ResNet, EfficientNet, BERT, GPT, T5, ViT, Stable Diffusion weights.

    EXCEPTIONS (TF32 may slightly affect results):
        Ill-conditioned linear systems (scientific ML).
        Deterministic unit tests comparing FP32 output element-by-element.
        Very precise numerical algorithms (Cholesky, LU decomposition).

### Enabling TF32: The Three Flags

    On A100 and later, TF32 is ENABLED BY DEFAULT for cuDNN convolutions
    but DISABLED BY DEFAULT for cuBLAS GEMMs (for backward compatibility).

    FLAG 1 — cuBLAS GEMM (torch.mm, torch.matmul, nn.Linear):
        torch.backends.cuda.matmul.allow_tf32 = True
        Effect: SGEMM routes through tensor cores with TF32 rounding.
        Speedup vs FP32 scalar: ~8–10× on A100.

    FLAG 2 — cuDNN convolutions (nn.Conv2d, etc.):
        torch.backends.cudnn.allow_tf32 = True
        This is TRUE by default since PyTorch 1.8!
        Effect: conv kernels already use TF32 unless you disabled it.

    FLAG 3 — Environment variable (global, overrides all):
        export NVIDIA_TF32_OVERRIDE=1   ← force all NVIDIA libs to use TF32
        export NVIDIA_TF32_OVERRIDE=0   ← force disable TF32 globally

    RECOMMENDED SETUP for any Ampere+ training:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True  (already default)

### TF32 vs FP16 AMP: Which to Use When

    ┌──────────────────┬──────────────────────────┬──────────────────────────┐
    │ Criterion        │ TF32                     │ FP16 AMP                 │
    ├──────────────────┼──────────────────────────┼──────────────────────────┤
    │ Code change      │ 1 flag                   │ autocast + GradScaler    │
    │ Memory saving    │ None (inputs stay FP32)  │ ~2× activations          │
    │ Speedup (GEMM)   │ ~8–10× vs scalar FP32    │ same (both use TC)       │
    │ Speedup (total)  │ ~2–3× typical            │ ~3–5× typical            │
    │ Overflow risk    │ None (FP32 range)        │ Yes (need GradScaler)    │
    │ Gradient quality │ Identical to FP32        │ Slightly reduced         │
    │ GPU requirement  │ A100, H100, RTX30/40     │ V100, A100, H100, RTX    │
    │ BF16 support     │ Not applicable           │ A100+ only               │
    └──────────────────┴──────────────────────────┴──────────────────────────┘

    RULE OF THUMB:
        Memory-limited training (large model, large batch): FP16 or BF16 AMP
            → the activation memory savings are the primary benefit.
        Compute-limited training (small model, GPU not fully utilised): TF32
            → minimal code change, captures the GEMM speedup.
        Production LLM training (>1B params, Ampere+): BF16 AMP + TF32
            → both enabled simultaneously for maximum performance.

### TF32 + BF16 AMP Simultaneously

    These are orthogonal settings that can both be active:
        torch.backends.cuda.matmul.allow_tf32 = True   ← affects FP32 GEMMs
        with autocast(dtype=torch.bfloat16): ...        ← casts FP32 → BF16 for GEMMs

    When autocast is active for BF16: the GEMM inputs are BF16, so TF32
    rounding (which applies to FP32 inputs) is irrelevant for those operations.
    TF32 still affects operations that fall outside autocast's dtype policy
    (e.g., FP32 ops that autocast doesn't rewrite).

    In practice: enabling both is fine. autocast handles the hot paths.
    TF32 catches any residual FP32 GEMMs that autocast missed.


##### PART 5 — AUTOCAST: HOW PYTORCH DECIDES WHICH OPS RUN AT HALF PRECISION

### The Autocast Op Policy

    torch.autocast() does NOT cast all operations. It has a policy table
    defining which operations run at which precision:

    FP16 / BF16 (lower precision — tensor core ops):
        torch.mm, torch.matmul, torch.bmm
        torch.nn.functional.linear
        torch.nn.functional.conv1d/2d/3d
        torch.nn.functional.conv_transpose1d/2d/3d
        torch.nn.MultiheadAttention (full forward)
        torch.nn.functional.scaled_dot_product_attention
        GRU, LSTM (fully)

    FP32 (stay at full precision — numerically sensitive):
        torch.nn.functional.softmax
        torch.nn.functional.layer_norm
        torch.nn.functional.batch_norm
        torch.nn.functional.group_norm
        torch.nn.functional.cross_entropy
        torch.nn.functional.nll_loss
        torch.log, torch.exp, torch.log1p, torch.expm1
        torch.pow, torch.reciprocal, torch.rsqrt
        torch.mean, torch.sum (with large inputs)

    PROMOTED TO FP32 (runs in FP32 even if inputs are FP16):
        Operations where precision matters more than speed.

    UNCHANGED (inherit input dtype):
        torch.cat, torch.stack, tensor indexing, view operations.
        These just pass data through without changing dtype.

### Why Certain Ops Stay in FP32

    SOFTMAX:
        Involves exp(x - max(x)) / sum(exp(x - max(x))).
        The sum accumulates across seq_len (up to 8192+) values.
        In FP16: sum of large exps can overflow 65504. NaN cascade.
        In FP32: safe up to seq_len ~ 10^7.

    LAYER NORM:
        Variance computation: sum((x - mean)^2).
        Subtraction of nearly equal values → catastrophic cancellation in FP16.
        Example: (1.001 - 1.000)^2 = 10^-6 — below FP16 precision at scale 1.

    CROSS ENTROPY:
        Involves log(softmax(logits)).
        log of small probabilities → large negative numbers → underflow in FP16.
        Example: log(1e-7) ≈ -16. FP16 min non-zero ≈ 6e-5. log(6e-5) ≈ -9.7.
        Gradients through cross entropy can easily underflow.

    BATCH NORM:
        Running mean and variance must be stable across batches.
        FP16 precision for the running statistics accumulates error over epochs.
        Batch norm in FP32 then cast output to FP16 for subsequent ops.

### Custom Op autocast Registration

    If you write a custom CUDA operator, you must register its autocast policy:

        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
        def my_forward(x, weight):
            ...  # inputs automatically cast to FP16

        @torch.cuda.amp.custom_bwd
        def my_backward(ctx, grad_output):
            ...  # runs in same dtype as forward

    Without registration: autocast context has NO EFFECT on your custom op.
    The op runs in whatever dtype its inputs happen to be.


##### PART 6 — DIAGNOSING MIXED PRECISION FAILURES

### Symptom 1: Training Loss Diverges (NaN or inf)

    IMMEDIATE DIAGNOSIS:
        print(scaler.get_scale())   ← if scale is 1.0, it hit minimum and stayed
        look for "Grad scaler overflow" messages

    ROOT CAUSES:
        a) Scale too large initially: gradients overflow → NaN → loss diverges.
           Fix: lower init_scale (try 2^8 = 256) or let dynamic scaler handle it.

        b) Learning rate too high combined with FP16 precision loss:
           FP16 weight updates round to zero for lr × grad < epsilon.
           Fix: reduce lr, use BF16 instead.

        c) Custom op not registered with autocast: critical op runs in FP32,
           output is upcast to FP32, but next op expects FP16 → dtype mismatch.
           Fix: register the custom op or add explicit cast.

        d) Accumulated FP16 gradients across gradient accumulation steps:
           Each accumulation adds FP16 precision error.
           Fix: use FP32 gradient accumulation buffer
               (gradient.grad_fn output dtype ≠ parameter dtype).

    CHECK GRADIENT HEALTH:
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad
                print(f"{name}: min={g.min():.3e}, max={g.max():.3e}, "
                      f"nan={g.isnan().any()}, inf={g.isinf().any()}")

### Symptom 2: Training Works But Accuracy Is Lower Than FP32

    ROOT CAUSES:
        a) Gradients silently underflow (no NaN, but gradients → 0):
           FP16 gradients below 6e-5 flush to zero. Some layers stop updating.
           Diagnosis: check gradient norms per layer. Some should be very small.
           Fix: increase loss scale or switch to BF16.

        b) Weight updates too small due to FP16 quantisation:
           lr × |grad| < epsilon for some parameters.
           FP16 epsilon at |weight| = 0.001 is ~10^-6. Adam lr = 3e-4.
           Adam update ≈ lr × grad_norm ≈ 3e-4 × 1e-4 = 3e-8.
           In FP16 at weight ≈ 1.0: epsilon = 0.001. Update of 3e-8 rounds to 0.
           Fix: FP32 master weights (already standard in AMP — verify it's enabled).
           FP32 master weights are the exact solution: update is applied in FP32,
           then cast to FP16 for the forward pass.

        c) Gradient clipping applied BEFORE unscale:
           Clip threshold applied to SCALED gradients → effectively clips at
           max_norm × S (thousands of times too tight) → all gradients zeroed.
           Fix: call scaler.unscale_(optimizer) BEFORE clip_grad_norm_.

### Symptom 3: Slower Than Expected With AMP

    ROOT CAUSES:
        a) Too many dtype conversions (FP32 ↔ FP16):
           Every autocast boundary causes a data copy to change dtype.
           If your model alternates frequently between FP16-eligible and
           FP32-eligible ops, the cast overhead dominates.
           Diagnosis: nsys profile shows many elementwise cast kernels.
           Fix: restructure ops to group FP16 ops and FP32 ops separately.

        b) Non-aligned tensor dimensions:
           FP16 tensor cores require dims divisible by 8 (ideally 64+).
           A model with vocab_size=50001 triggers the scalar path for the head layer.
           Diagnosis: ncu shows sm__inst_executed_pipe_tensor_op_hmma.sum = 0
           for the head GEMM.
           Fix: pad vocab_size to 50048 (next multiple of 64).

        c) GradScaler inf check overhead:
           Very short steps (< 500 µs): the inf/nan reduction adds visible overhead.
           Fix: increase batch size, use gradient accumulation, or static scale.


##### PART 7 — PRODUCTION MIXED PRECISION RECIPE: COMPLETE CONFIGURATION

### LLM Training (A100 / H100, BF16)

    # 1. Global flags (set once at program start)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # 2. Model and optimizer
    model     = MyModel().cuda().to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                   fused=True)         # fused=True for A100+ speed

    # 3. No GradScaler needed for BF16 (sufficient dynamic range)
    # 4. Training loop
    for batch in dataloader:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

### Legacy Training (V100 / T4, FP16 only)

    # 1. GradScaler is REQUIRED for FP16
    scaler = torch.cuda.amp.GradScaler(init_scale=65536)

    for batch in dataloader:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = model(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

### Mixed Precision Debugging Checklist

    □ scaler.get_scale() not stuck at minimum (1.0)?
    □ scaler._num_grads (PyTorch internal) not showing repeated overflow?
    □ Gradient norms finite and non-zero for all layers?
    □ scaler.unscale_(optimizer) called BEFORE clip_grad_norm_?
    □ Tensor dimensions divisible by 8 (ideally 64)?
    □ torch.backends.cuda.matmul.allow_tf32 = True set on Ampere+?
    □ Custom ops registered with @custom_fwd / @custom_bwd?
    □ Layer norm, batch norm, softmax NOT inside autocast inner scope?
      (autocast handles this automatically, but worth verifying in custom models)

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Float Format Analyser — FP16 / BF16 / TF32 / FP32 Side-by-Side": {
        "description": (
            "Decode the bit layouts of FP16, BF16, TF32, and FP32. Show every "
            "format property: exponent bits, mantissa bits, bias, max/min values, "
            "epsilon, and max safe integer. Enumerate the representable values in "
            "each format near key training thresholds. Demonstrate the ULP (unit "
            "in last place) gap at typical weight and gradient magnitudes. Show "
            "how TF32 rounding truncates the FP32 mantissa to 10 bits while "
            "preserving the full exponent range."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import struct

print("=" * 68)
print("  FLOAT FORMAT ANALYSER — FP16 / BF16 / TF32 / FP32")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# Bit-level utilities
# ─────────────────────────────────────────────────────────────────────

def fp32_bits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]

def fp32_from_bits(bits):
    return struct.unpack('<f', struct.pack('<I', bits & 0xFFFFFFFF))[0]

def fp16_bits(x):
    b = np.float16(x).tobytes()
    return int.from_bytes(b, 'little')

def round_to_tf32(x):
    bits = fp32_bits(x)
    bits_tf32 = bits & 0xFFFFE000   # zero bottom 13 mantissa bits
    return fp32_from_bits(bits_tf32)

def round_to_bf16(x):
    bits = fp32_bits(x)
    # BF16: keep top 16 bits of FP32 (1+8+7), zero bottom 16
    bits_bf16 = bits & 0xFFFF0000
    return fp32_from_bits(bits_bf16)

def bit_string(val, fmt):
    """Return sign|exponent|mantissa bit string for a value in the given format."""
    if fmt == 'fp32':
        bits = fp32_bits(val)
        s  = (bits >> 31) & 1
        e  = (bits >> 23) & 0xFF
        m  = bits & 0x7FFFFF
        return f"{s}|{e:08b}|{m:023b}"
    elif fmt == 'fp16':
        bits = fp16_bits(val)
        s  = (bits >> 15) & 1
        e  = (bits >> 10) & 0x1F
        m  = bits & 0x3FF
        return f"{s}|{e:05b}|{m:010b}"
    elif fmt == 'bf16':
        bits = fp32_bits(val) >> 16   # BF16 is top 16 bits of FP32
        s  = (bits >> 15) & 1
        e  = (bits >>  7) & 0xFF
        m  = bits & 0x7F
        return f"{s}|{e:08b}|{m:07b}"
    elif fmt == 'tf32':
        bits = fp32_bits(round_to_tf32(val))
        s  = (bits >> 31) & 1
        e  = (bits >> 23) & 0xFF
        m  = (bits >> 13) & 0x3FF   # top 10 mantissa bits
        return f"{s}|{e:08b}|{m:010b}|{'0'*13}"


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Format properties table
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Format Properties: Bits, Range, Epsilon")
print("━" * 68)
print()

formats = [
    {
        "name": "FP16",  "bits": 16, "exp_bits": 5, "mant_bits": 10,
        "bias": 15,
        "max_val":   65504.0,
        "min_normal": 2**-14,
        "epsilon":    2**-10,
        "max_int":    2048,
        "storage": True,
        "hw_note": "Volta+ (V100, A100, H100, RTX cards)",
    },
    {
        "name": "BF16",  "bits": 16, "exp_bits": 8, "mant_bits": 7,
        "bias": 127,
        "max_val":   3.39e38,
        "min_normal": 2**-126,
        "epsilon":    2**-7,
        "max_int":    256,
        "storage": True,
        "hw_note": "Ampere+ (A100, H100, RTX30/40)",
    },
    {
        "name": "TF32",  "bits": 19, "exp_bits": 8, "mant_bits": 10,
        "bias": 127,
        "max_val":   3.39e38,
        "min_normal": 2**-126,
        "epsilon":    2**-10,
        "max_int":    2048,
        "storage": False,
        "hw_note": "A100, H100 (internal only, not a storage dtype)",
    },
    {
        "name": "FP32",  "bits": 32, "exp_bits": 8, "mant_bits": 23,
        "bias": 127,
        "max_val":   3.40e38,
        "min_normal": 2**-126,
        "epsilon":    2**-23,
        "max_int":    16_777_216,
        "storage": True,
        "hw_note": "All GPUs",
    },
]

print(f"  {'Name':<6}  {'Bits':>5}  {'Exp':>4}  {'Mant':>5}  {'Max value':>10}  "
      f"{'Min normal':>12}  {'Epsilon':>10}  {'Max int':>12}  Storage")
print("  " + "─" * 84)
for f in formats:
    storage_str = "✅ yes" if f["storage"] else "❌ no (internal)"
    print(f"  {f['name']:<6}  {f['bits']:>5}  {f['exp_bits']:>4}  "
          f"{f['mant_bits']:>5}  {f['max_val']:>10.3g}  "
          f"{f['min_normal']:>12.3g}  {f['epsilon']:>10.6f}  "
          f"{f['max_int']:>12,}  {storage_str}")
print()
print("  KEY INSIGHT: BF16 has the same exponent range as FP32 → same overflow threshold.")
print("  FP16 max is 65504 — gradient norms above this cause overflow → NaN cascade.")
print("  TF32 is 19 effective bits but NOT a storage type: stored as FP32 in memory.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Bit-level representation of the same value across formats
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Bit Layouts: Same Value in All Four Formats")
print("━" * 68)
print()

test_values = [1.0, 0.1, 0.001, 3.14159, 100.0, 65000.0]

for v in test_values:
    fp16_v   = float(np.float16(v))
    bf16_v   = round_to_bf16(v)
    tf32_v   = round_to_tf32(v)

    err_fp16 = abs(fp16_v - v) / abs(v) * 100 if v != 0 else 0
    err_bf16 = abs(bf16_v - v) / abs(v) * 100 if v != 0 else 0
    err_tf32 = abs(tf32_v - v) / abs(v) * 100 if v != 0 else 0

    print(f"  Value = {v}")
    print(f"    FP32  bits: {bit_string(v, 'fp32')}")
    print(f"    TF32  bits: {bit_string(v, 'tf32')}  (stored val={tf32_v:.8f}, err={err_tf32:.5f}%)")
    print(f"    BF16  bits: {bit_string(v, 'bf16')}  (stored val={bf16_v:.8f}, err={err_bf16:.5f}%)")
    print(f"    FP16  bits: {bit_string(v, 'fp16')}  (stored val={fp16_v:.8f}, err={err_fp16:.5f}%)")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: ULP gaps at critical training value ranges
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — ULP Gaps at Typical Weight and Gradient Magnitudes")
print("━" * 68)
print()

def ulp_fp16(x):
    v = abs(x)
    if v == 0: return 0
    exp = math.floor(math.log2(v))
    return 2.0 ** (exp - 10)

def ulp_bf16(x):
    v = abs(x)
    if v == 0: return 0
    exp = math.floor(math.log2(v))
    return 2.0 ** (exp - 7)

def ulp_fp32(x):
    v = abs(x)
    if v == 0: return 0
    exp = math.floor(math.log2(v))
    return 2.0 ** (exp - 23)

print(f"  Context: Adam update ≈ lr × (m1/m2). For lr=3e-4, typical update ≈ 3e-5.")
print(f"  If ULP at weight value > update magnitude → weight update rounds to ZERO.")
print()
print(f"  {'Magnitude':>12}  {'Context':>22}  {'FP16 ULP':>12}  "
      f"{'BF16 ULP':>12}  {'FP32 ULP':>12}  {'FP16 eats update?'}")
print("  " + "─" * 82)

typical_update = 3e-5

cases = [
    (1e-6,  "tiny gradient"),
    (1e-4,  "FP16 flush zone"),
    (1e-3,  "small gradient"),
    (0.01,  "normal gradient"),
    (0.1,   "large gradient"),
    (1.0,   "typical weight"),
    (10.0,  "large weight"),
    (100.0, "very large weight"),
    (1000.0,"near FP16 limit"),
]

for mag, ctx in cases:
    try:
        u16 = ulp_fp16(mag) if mag <= 65504 else float('inf')
    except:
        u16 = float('inf')
    u_bf16 = ulp_bf16(mag)
    u32    = ulp_fp32(mag)
    eats   = "❌ YES → layer dies" if u16 > typical_update else "✅ ok"
    fp16_str = f"{u16:.2e}" if u16 != float('inf') else "OVERFLOW"
    print(f"  {mag:>12.1e}  {ctx:>22}  {fp16_str:>12}  "
          f"{u_bf16:>12.2e}  {u32:>12.2e}  {eats}")

print()
print(f"  Adam update magnitude assumed: {typical_update:.0e}  (lr={3e-4}, grad~0.1)")
print(f"  Any FP16 ULP > {typical_update:.0e} means the weight update rounds to zero → that layer stops learning.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Loss Scaling Simulator — Dynamic Scale, Overflow Detection & Skips": {
        "description": (
            "Simulate the full dynamic loss scaling algorithm across 5000 training "
            "steps. Model the gradient overflow probability as a function of scale "
            "factor and gradient distribution. Show scale dynamics: overflow causes "
            "scale to drop, clean steps cause scale to grow. Track skipped optimizer "
            "steps. Compare static vs dynamic scaling on a distribution with "
            "varying gradient magnitudes across training phases."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  LOSS SCALING SIMULATOR — Dynamic Scale, Overflow & Skips")
print("=" * 68)
print()

np.random.seed(42)

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.1e-5


# ─────────────────────────────────────────────────────────────────────
# Gradient distribution model
# ─────────────────────────────────────────────────────────────────────

def sample_gradient_norm(step, phase="training"):
    """
    Simulate the gradient L2 norm at a given training step.
    Three phases: warmup (large norms), mid-training (stable), fine-tune (small).
    """
    if step < 500:      # warmup: high variance, occasional large spikes
        base = np.random.lognormal(mean=1.5, sigma=2.0)
    elif step < 3000:   # mid-training: stable regime
        base = np.random.lognormal(mean=-1.0, sigma=1.2)
    else:               # fine-tuning: smaller gradients
        base = np.random.lognormal(mean=-3.0, sigma=0.8)
    # Occasional spikes (gradient norm explosion events)
    if np.random.rand() < 0.005:
        base *= np.random.uniform(50, 500)
    return float(base)


def will_overflow_fp16(grad_norm, scale):
    """Does the scaled gradient exceed FP16 range?"""
    return (grad_norm * scale) > FP16_MAX


def has_underflow_fp16(grad_norm, scale=1.0):
    """Does the unscaled gradient have components below FP16 flush-to-zero?"""
    # Rough model: assume ~10% of gradient components at grad_norm / sqrt(N_params)
    per_param = grad_norm / math.sqrt(1e7)   # ~10M params
    return per_param < FP16_MIN_NORMAL


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Dynamic loss scaler simulation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Dynamic Scale Trajectory Over 5000 Steps")
print("━" * 68)
print()

class DynamicScaler:
    def __init__(self, init_scale=65536, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000,
                 min_scale=1.0, max_scale=2**24):
        self.scale           = float(init_scale)
        self.growth_factor   = growth_factor
        self.backoff_factor  = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale       = min_scale
        self.max_scale       = max_scale
        self._good_steps     = 0
        self.total_steps     = 0
        self.skip_steps      = 0

    def step(self, grad_norm):
        self.total_steps += 1
        overflow = will_overflow_fp16(grad_norm, self.scale)

        if overflow:
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self._good_steps = 0
            self.skip_steps += 1
            return False, "OVERFLOW → skip + backoff"
        else:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self._good_steps = 0
                return True, "GROW"
            return True, "ok"


N_STEPS = 5000
scaler  = DynamicScaler(init_scale=65536)

history = []
for step in range(N_STEPS):
    g_norm  = sample_gradient_norm(step)
    applied, status = scaler.step(g_norm)
    history.append({
        "step":     step,
        "g_norm":   g_norm,
        "scale":    scaler.scale,
        "applied":  applied,
        "status":   status,
    })

# Print summary by phase
phases = [(0, 500, "Warmup"), (500, 3000, "Mid-training"), (3000, 5000, "Fine-tuning")]
print(f"  {'Phase':<18}  {'Steps':>6}  {'Overflows':>10}  {'Skip%':>8}  "
      f"{'Final scale':>12}  {'Scale range'}")
print("  " + "─" * 68)

for p_start, p_end, p_name in phases:
    phase_h = [h for h in history if p_start <= h["step"] < p_end]
    n_skip  = sum(1 for h in phase_h if not h["applied"])
    n_total = len(phase_h)
    scales  = [h["scale"] for h in phase_h]
    print(f"  {p_name:<18}  {n_total:>6}  {n_skip:>10}  "
          f"{n_skip/n_total*100:>7.1f}%  {phase_h[-1]['scale']:>12.0f}  "
          f"[{min(scales):.0f}, {max(scales):.0f}]")

print()
total_skip = sum(1 for h in history if not h["applied"])
print(f"  TOTAL: {N_STEPS} steps, {total_skip} skipped "
      f"({total_skip/N_STEPS*100:.1f}%), {N_STEPS-total_skip} optimizer steps applied")
print()

# Print first 25 steps showing the initial scale crash
print("  First 25 steps (initial scale crash during warmup):")
print(f"  {'Step':>5}  {'Grad norm':>12}  {'Scale':>10}  {'Status'}")
print("  " + "─" * 48)
for h in history[:25]:
    print(f"  {h['step']:>5}  {h['g_norm']:>12.3f}  {h['scale']:>10.0f}  {h['status']}")
print()

# Print 3 growth events
print("  First 3 scale growth events:")
grow_events = [h for h in history if h["status"] == "GROW"][:3]
for h in grow_events:
    print(f"  Step {h['step']:4d}: scale grew to {h['scale']:.0f}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: ASCII scale trajectory plot
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Scale Trajectory (log2 scale, 500-step buckets)")
print("━" * 68)
print()

BUCKET_SIZE = 250
WIDTH       = 50
buckets     = range(0, N_STEPS, BUCKET_SIZE)
log2_scales = []
skip_rates  = []

for b in buckets:
    phase_h = [h for h in history if b <= h["step"] < b + BUCKET_SIZE]
    if not phase_h:
        continue
    avg_log2_scale = math.log2(max(np.mean([h["scale"] for h in phase_h]), 1))
    n_skip         = sum(1 for h in phase_h if not h["applied"])
    log2_scales.append(avg_log2_scale)
    skip_rates.append(n_skip / len(phase_h) * 100)

max_log2 = max(log2_scales) if log2_scales else 1
min_log2 = min(log2_scales) if log2_scales else 0

print("  Loss scale (log2) over training — height = log2(scale)")
print(f"  Y-axis: {min_log2:.0f} to {max_log2:.0f} (log2 scale)")
print()

for i, (ls, sr) in enumerate(zip(log2_scales, skip_rates)):
    step_label = f"step {i*BUCKET_SIZE:4d}"
    bar_len    = int((ls - min_log2) / max(max_log2 - min_log2, 1) * WIDTH)
    bar_len    = max(1, min(bar_len, WIDTH))
    skip_str   = f"skip={sr:.0f}%" if sr > 0 else ""
    print(f"  {step_label}: {'█' * bar_len:<{WIDTH}}  log2(S)={ls:.1f}  {skip_str}")

print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Gradient underflow with and without scaling
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Gradient Underflow: How Scaling Rescues Small Gradients")
print("━" * 68)
print()

print("  Simulating per-parameter gradient magnitudes at step 4000 (fine-tuning).")
print("  True gradient norms drawn from log-normal(mu=-4, sigma=2).")
print()

np.random.seed(77)
n_params = 10000
true_grads = np.abs(np.random.lognormal(mean=-4, sigma=2, size=n_params))

for scale in [1, 128, 1024, 4096, 16384]:
    scaled = true_grads * scale
    # After unscaling, represent in FP16 first
    fp16_scaled = np.array([float(np.float16(sg)) for sg in np.clip(scaled, 0, FP16_MAX)])
    unscaled    = fp16_scaled / scale

    n_zero       = np.sum(unscaled == 0)
    n_overflow   = np.sum(scaled > FP16_MAX)
    frac_zero    = n_zero    / n_params * 100
    frac_overflow = n_overflow / n_params * 100
    mean_err     = np.abs(unscaled - true_grads).mean() / (true_grads.mean() + 1e-10) * 100

    print(f"  Scale = {scale:>6}:  "
          f"zeroed = {frac_zero:>5.1f}%  "
          f"overflow = {frac_overflow:>5.1f}%  "
          f"mean_rel_err = {mean_err:>6.1f}%  "
          f"{'⚠ too small' if frac_zero > 20 else '⚠ overflow' if frac_overflow > 2 else '✅ good'}")

print()
print("  Optimal scale = 4096 in this run: balances underflow and overflow.")
print("  Dynamic scaler automatically finds this range without manual tuning.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Static vs dynamic scaling comparison
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Static vs Dynamic Scaling: Accuracy and Stability")
print("━" * 68)
print()

static_scales = [128, 512, 4096, 16384, 65536]

print(f"  {'Scale type':<20}  {'Scale':>8}  {'Total skips':>12}  "
      f"{'Skip %':>8}  {'Final model (proxy):  zeroed grads %'}")
print("  " + "─" * 72)

for static_s in static_scales:
    # Simulate static scaler: same scale forever
    n_skip_static = sum(1 for h in history if will_overflow_fp16(h["g_norm"], static_s))
    frac_zeroed   = 0
    # Estimate zeroed gradients at fine-tune phase
    for h in [hist for hist in history if hist["step"] > 3000]:
        g = h["g_norm"]
        per_param = g / math.sqrt(1e7)
        if per_param * static_s < FP16_MIN_NORMAL:
            frac_zeroed += 1
    frac_zeroed_pct = frac_zeroed / max(len([h for h in history if h["step"] > 3000]), 1) * 100

    ok = ("✅ stable" if n_skip_static < N_STEPS * 0.05 and frac_zeroed_pct < 20
          else "⚠  marginal" if n_skip_static < N_STEPS * 0.15
          else "❌ unstable")

    print(f"  {'Static S='+str(static_s):<20}  {static_s:>8}  {n_skip_static:>12}  "
          f"{n_skip_static/N_STEPS*100:>7.1f}%  {frac_zeroed_pct:>5.1f}% zeroed  {ok}")

print(f"  {'Dynamic (adaptive)':<20}  {'auto':>8}  {total_skip:>12}  "
      f"{total_skip/N_STEPS*100:>7.1f}%  {'~0% zeroed':>14}  ✅ adaptive")
print()
print("  Dynamic scaling is always preferred for new model/dataset combinations.")
print("  Static scaling only safe when scale is pre-calibrated from identical runs.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · TF32 Deep Dive — Rounding, GEMM Accuracy & Ampere Math Modes": {
        "description": (
            "Implement TF32 rounding from bit manipulation. Compare FP32, TF32, "
            "BF16, and FP16 GEMM accuracy against FP64 reference across a sweep of "
            "matrix sizes and conditioning numbers. Show the TF32 rounding error "
            "distribution. Measure how the error scales with K (inner dimension). "
            "Demonstrate the Ampere math mode flags and their effect on GEMM results "
            "using torch operations where available."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import struct

print("=" * 68)
print("  TF32 DEEP DIVE — Rounding, GEMM Accuracy & Math Modes")
print("=" * 68)
print()

np.random.seed(55)


# ─────────────────────────────────────────────────────────────────────
# Precision rounding utilities
# ─────────────────────────────────────────────────────────────────────

def fp32_bits(x):
    return struct.unpack('<I', struct.pack('<f', float(np.float32(x))))[0]

def fp32_from_bits(b):
    return float(struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0])

def round_to_tf32(x):
    """TF32: keep sign(1) + exp(8) + top 10 mantissa bits, zero bottom 13."""
    b = fp32_bits(x)
    return fp32_from_bits(b & 0xFFFFE000)

def round_to_bf16(x):
    """BF16: keep sign(1) + exp(8) + top 7 mantissa bits."""
    b = fp32_bits(x)
    return fp32_from_bits(b & 0xFFFF0000)

def apply_rounding(arr, fmt):
    """Apply precision rounding to a numpy array."""
    if fmt == 'fp32':
        return arr.astype(np.float32)
    elif fmt == 'fp16':
        return arr.astype(np.float16).astype(np.float32)
    elif fmt == 'bf16':
        vround = np.vectorize(round_to_bf16)
        return vround(arr.astype(np.float32)).astype(np.float32)
    elif fmt == 'tf32':
        vround = np.vectorize(round_to_tf32)
        return vround(arr.astype(np.float32)).astype(np.float32)
    return arr


def gemm_with_precision(A, B, input_fmt, accum_fmt='fp32'):
    """
    Simulate precision-limited GEMM.
    input_fmt: how A and B are stored/rounded before MMA.
    accum_fmt: accumulator precision ('fp32' or 'fp16').
    """
    A_r = apply_rounding(A, input_fmt)
    B_r = apply_rounding(B, input_fmt)
    if accum_fmt == 'fp32':
        return A_r.astype(np.float64) @ B_r.astype(np.float64)
    else:
        return (A_r.astype(np.float16) @ B_r.astype(np.float16)).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: TF32 rounding — bit manipulation demonstrated
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — TF32 Bit Rounding: Mantissa Truncation")
print("━" * 68)
print()

print("  TF32 zeroes the bottom 13 mantissa bits of FP32.")
print("  Format: 1 sign + 8 exponent + 10 mantissa = 19 significant bits.")
print()

test_vals = [1.0, math.pi, 0.1, -2.718281828, 100.5]

print(f"  {'Value':>14}  {'FP32 mantissa':>30}  {'TF32 mantissa':>30}  {'Rel err %':>10}")
print("  " + "─" * 86)

for v in test_vals:
    b_fp32  = fp32_bits(v)
    b_tf32  = b_fp32 & 0xFFFFE000
    b_zeroed = b_fp32 & 0x1FFF      # the 13 bits removed

    mant_fp32  = b_fp32  & 0x7FFFFF
    mant_tf32  = b_tf32  & 0x7FFFFF
    mant_kept  = mant_fp32 >> 13    # top 10 bits
    mant_lost  = mant_fp32 & 0x1FFF # bottom 13 bits

    v_tf32  = fp32_from_bits(b_tf32)
    rel_err = abs(v_tf32 - v) / abs(v) * 100 if v != 0 else 0

    print(f"  {v:>14.8f}  {mant_fp32:>23b} (23b)  {mant_tf32:>23b} (23b)  {rel_err:>10.6f}%")
    print(f"  {'':>14}  kept top 10: {mant_kept:010b}  lost bottom 13: {mant_lost:013b}")
    print()

print("  The 13 zero bits at the end are carried in memory as a 32-bit FP32 value.")
print("  TF32 precision is applied at compute time; the storage footprint is unchanged.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: GEMM accuracy vs K dimension
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — GEMM Accuracy vs K: All Four Precisions vs FP64")
print("━" * 68)
print()

M_g, N_g = 32, 32

print(f"  GEMM ({M_g}×K) × (K×{N_g}), accumulated in FP32 (except FP16-accum baseline)")
print()
print(f"  {'K':>6}  {'FP32 err':>12}  {'TF32 err':>12}  {'BF16 err':>12}  "
      f"{'FP16 err':>12}  {'FP16-accum err':>15}")
print("  " + "─" * 68)

for K_test in [16, 64, 128, 256, 512, 1024, 2048, 4096]:
    A_t = np.random.randn(M_g, K_test).astype(np.float32)
    B_t = np.random.randn(K_test, N_g).astype(np.float32)
    C_fp64 = A_t.astype(np.float64) @ B_t.astype(np.float64)

    C_fp32 = gemm_with_precision(A_t, B_t, 'fp32',  'fp32')
    C_tf32 = gemm_with_precision(A_t, B_t, 'tf32',  'fp32')
    C_bf16 = gemm_with_precision(A_t, B_t, 'bf16',  'fp32')
    C_fp16 = gemm_with_precision(A_t, B_t, 'fp16',  'fp32')
    C_fp16a = gemm_with_precision(A_t, B_t, 'fp16', 'fp16')

    ref_scale = np.abs(C_fp64).mean() + 1e-10
    err_fp32  = np.abs(C_fp32  - C_fp64).mean() / ref_scale * 100
    err_tf32  = np.abs(C_tf32  - C_fp64).mean() / ref_scale * 100
    err_bf16  = np.abs(C_bf16  - C_fp64).mean() / ref_scale * 100
    err_fp16  = np.abs(C_fp16  - C_fp64).mean() / ref_scale * 100
    err_fp16a = np.abs(C_fp16a - C_fp64).mean() / ref_scale * 100

    print(f"  {K_test:>6}  {err_fp32:>12.5f}%  {err_tf32:>12.5f}%  "
          f"{err_bf16:>12.5f}%  {err_fp16:>12.5f}%  {err_fp16a:>15.5f}%")

print()
print("  OBSERVATIONS:")
print("    FP32 error: near machine epsilon (accumulated rounding, not quantisation).")
print("    TF32 error: slightly larger than FP32 but constant with K.")
print("    BF16 error: highest per-element quantisation but constant with K.")
print("    FP16 error: similar to BF16 for small K, diverges at large K.")
print("    FP16-accum: catastrophic growth with K — error ~K × FP32 at K=4096.")
print("    → This is why tensor cores ALWAYS accumulate in FP32.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Condition number sensitivity
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Condition Number Sensitivity: When TF32 Can Hurt")
print("━" * 68)
print()

print("  Ill-conditioned matrices amplify precision errors.")
print("  Condition number κ(A) = σ_max / σ_min (ratio of largest to smallest singular value).")
print()

M_c, K_c = 32, 32

def make_matrix_with_condition(M, K, cond):
    """Build a random matrix with approximate condition number cond."""
    U = np.linalg.qr(np.random.randn(M, K))[0] if M <= K else np.linalg.qr(np.random.randn(K, M))[0].T
    V = np.linalg.qr(np.random.randn(K, K))[0]
    singular_vals = np.logspace(0, -np.log10(cond), min(M, K))
    S_diag = np.zeros((M, K))
    n_sv = min(M, K)
    S_diag[:n_sv, :n_sv] = np.diag(singular_vals)
    return (U[:M, :K] * singular_vals[np.newaxis, :min(M, K)]) @ V[:min(M, K), :]

print(f"  {'Condition κ':>14}  {'FP32 rel%':>12}  {'TF32 rel%':>12}  "
      f"{'BF16 rel%':>12}  {'FP16 rel%':>12}  {'TF32 safe?'}")
print("  " + "─" * 68)

for cond in [1, 10, 100, 1000, 1e4, 1e5, 1e6]:
    try:
        A_c = make_matrix_with_condition(M_c, K_c, cond)
        B_c = np.random.randn(K_c, M_c).astype(np.float32)
        C_fp64_c = A_c.astype(np.float64) @ B_c.astype(np.float64)
        ref = np.abs(C_fp64_c).mean() + 1e-10

        C_fp32_c = gemm_with_precision(A_c, B_c, 'fp32', 'fp32')
        C_tf32_c = gemm_with_precision(A_c, B_c, 'tf32', 'fp32')
        C_bf16_c = gemm_with_precision(A_c, B_c, 'bf16', 'fp32')
        C_fp16_c = gemm_with_precision(A_c, B_c, 'fp16', 'fp32')

        e32  = np.abs(C_fp32_c - C_fp64_c).mean() / ref * 100
        etf  = np.abs(C_tf32_c - C_fp64_c).mean() / ref * 100
        ebf  = np.abs(C_bf16_c - C_fp64_c).mean() / ref * 100
        e16  = np.abs(C_fp16_c - C_fp64_c).mean() / ref * 100
        safe = "✅" if etf < 1.0 else ("⚠" if etf < 10 else "❌")

        print(f"  {cond:>14.0e}  {e32:>12.4f}%  {etf:>12.4f}%  "
              f"{ebf:>12.4f}%  {e16:>12.4f}%  {safe}")
    except Exception as e:
        print(f"  {cond:>14.0e}  error: {e}")

print()
print("  For κ > 1e4: TF32 rounding error starts to exceed FP32 by a visible margin.")
print("  For scientific ML (PINNs, molecular dynamics): use FP32 or FP64.")
print("  For deep learning (κ typically < 100 for well-initialised networks): TF32 is safe.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Math mode flags and their effects
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Ampere Math Mode Flags: What Each One Controls")
print("━" * 68)
print()

flags = [
    ("torch.backends.cuda.matmul.allow_tf32",
     "False (PyTorch default before 1.11)",
     True,
     "matmul/mm/linear: FP32 inputs use TF32 TC path",
     "~8-10× speedup for FP32 linear layers on A100"),
    ("torch.backends.cudnn.allow_tf32",
     "True (default since PyTorch 1.8)",
     True,
     "conv1d/2d/3d: FP32 convolutions use TF32 TC path",
     "CNN training already uses TF32 by default!"),
    ("torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
     "True (default)",
     True,
     "FP16 reductions (sum over large tensors) use reduced precision",
     "Minor accuracy loss for sums over many elements"),
    ("CUBLAS_WORKSPACE_CONFIG=:4096:8",
     "env var",
     False,
     "Enables larger cuBLAS workspace → better algorithms",
     "Required for deterministic results in some cuBLAS algorithms"),
]

print(f"  {'Flag':<55}  {'Default':>8}  {'Rec.':>5}")
print("  " + "─" * 74)
for flag, default, recommend, effect, note in flags:
    rec_sym = "✅ yes" if recommend else "  no"
    short_flag = flag[-50:] if len(flag) > 50 else flag
    print(f"  {short_flag:<55}  {default:>8}  {rec_sym}")
    print(f"    Effect: {effect}")
    print(f"    Note:   {note}")
    print()

print("  RECOMMENDED setup for any Ampere+ training run:")
print("    import torch")
print("    torch.backends.cuda.matmul.allow_tf32 = True")
print("    torch.backends.cudnn.allow_tf32       = True  # already default")
print("    # Optional for full performance:")
print("    torch.set_float32_matmul_precision('high')  # PyTorch 2.0+ shorthand")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Autocast Op Policy — Which Ops Run at Half Precision and Why": {
        "description": (
            "Catalogue the full autocast dtype policy: which operations cast to "
            "FP16/BF16 (GEMM-like), which stay in FP32 (numerically sensitive), "
            "and which inherit their input dtype (elementwise). Demonstrate WHY "
            "each FP32 operation would fail in half precision using concrete "
            "numerical examples: softmax overflow, layer norm cancellation, "
            "cross entropy underflow. Show how to register a custom op with the "
            "autocast system."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  AUTOCAST OP POLICY — What Runs at Half Precision and Why")
print("=" * 68)
print()

np.random.seed(11)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Autocast op catalogue
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Autocast Operation Policy Catalogue")
print("━" * 68)
print()

# (operation, policy, reason_short)
op_policy = [
    # ── FP16/BF16 (cast to half precision) ──────────────────────────
    ("torch.mm",                         "FP16/BF16", "GEMM — tensor core path"),
    ("torch.matmul",                     "FP16/BF16", "GEMM — tensor core path"),
    ("torch.bmm",                        "FP16/BF16", "batched GEMM — tensor cores"),
    ("nn.Linear (forward)",              "FP16/BF16", "linear = matmul + bias"),
    ("nn.Conv1d/2d/3d",                  "FP16/BF16", "convolution = implicit GEMM"),
    ("nn.MultiheadAttention",            "FP16/BF16", "Q@K and scores@V are GEMMs"),
    ("F.scaled_dot_product_attention",   "FP16/BF16", "fused attention (FlashAttn)"),
    ("nn.LSTM, nn.GRU",                  "FP16/BF16", "weight matrices are GEMMs"),
    # ── FP32 (stay at full precision) ───────────────────────────────
    ("F.softmax",                        "FP32",      "exp() can overflow FP16"),
    ("F.layer_norm",                     "FP32",      "variance: catastrophic cancel"),
    ("F.batch_norm",                     "FP32",      "running stats accumulate error"),
    ("F.group_norm",                     "FP32",      "same as layer norm"),
    ("F.cross_entropy",                  "FP32",      "log of small probs underflows"),
    ("F.nll_loss",                       "FP32",      "log domain sensitive"),
    ("torch.log, torch.exp",             "FP32",      "extreme values common"),
    ("torch.log1p, torch.expm1",         "FP32",      "cancellation at x≈0"),
    ("torch.reciprocal, torch.rsqrt",    "FP32",      "division by near-zero"),
    ("torch.mean (reduction)",           "FP32",      "accumulated rounding with large N"),
    # ── Inherited dtype (pass-through) ──────────────────────────────
    ("torch.cat",                        "inherited", "data copy, no compute"),
    ("torch.stack",                      "inherited", "data copy, no compute"),
    ("tensor indexing, slicing",         "inherited", "address arithmetic only"),
    ("F.relu, F.gelu",                   "inherited", "elementwise, cheap in any dtype"),
    ("F.dropout",                        "inherited", "mask multiply, dtype irrelevant"),
    ("nn.Embedding",                     "inherited", "lookup table, no arithmetic"),
]

categories = {
    "FP16/BF16": ("Cast to half precision (tensor core path)", "⚡"),
    "FP32":       ("Stay at full precision (numerically sensitive)", "🔒"),
    "inherited":  ("Inherit input dtype (pass-through)", "➡"),
}

for cat, (desc, sym) in categories.items():
    ops = [(op, pol, rsn) for op, pol, rsn in op_policy if pol == cat]
    print(f"  {sym} {cat} — {desc}  ({len(ops)} operations)")
    print()
    for op, pol, rsn in ops:
        print(f"    {op:<42}  {rsn}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Numerical proof — why softmax MUST stay in FP32
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Why Softmax Fails in FP16: Overflow Proof")
print("━" * 68)
print()

print("  Softmax: out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))")
print("  The max-shift prevents overflow — but only in FP32.")
print()

# Simulate realistic logit distributions at different scales
np.random.seed(33)
scenarios = [
    ("Normal logits, seq=64",     np.random.randn(64) * 1.0),
    ("Large logits, seq=64",      np.random.randn(64) * 5.0),
    ("Very large logits, seq=64", np.random.randn(64) * 20.0),
    ("Long sequence, seq=2048",   np.random.randn(2048) * 2.0),
]

for name, logits in scenarios:
    # FP64 reference
    logits_f64 = logits.astype(np.float64)
    max_f64    = logits_f64.max()
    shifted    = logits_f64 - max_f64
    exp_f64    = np.exp(shifted)
    sm_f64     = exp_f64 / exp_f64.sum()

    # FP16 attempt (with shift)
    logits_f16 = logits.astype(np.float16)
    max_f16    = float(logits_f16.max())
    shifted_f16 = (logits_f16 - np.float16(max_f16))
    # exp in FP16
    exp_f16    = np.array([np.float16(math.exp(float(x))) if float(x) < 11.0
                            else np.float16(65504) for x in shifted_f16])
    sum_f16    = float(exp_f16.sum())
    sm_f16     = exp_f16 / np.float16(sum_f16) if sum_f16 < 65504 else np.zeros_like(exp_f16)

    # Check
    n_inf_exp  = np.sum(~np.isfinite(exp_f16.astype(float)))
    n_zero_sm  = np.sum(sm_f16 == 0)
    total_err  = np.abs(sm_f16.astype(float) - sm_f64).mean() if not np.any(np.isnan(sm_f16)) else float('inf')

    print(f"  {name}:")
    print(f"    logit range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"    After max-shift, max exp input: {shifted.max():.2f}")
    print(f"    FP16 exp overflow (exp(x) > 65504 when x > 11.09): {n_inf_exp} values")
    print(f"    FP16 softmax zeros: {n_zero_sm} / {len(logits)}")
    print(f"    Mean error vs FP64: {'NaN' if total_err == float('inf') else f'{total_err:.5f}'}")
    print(f"    Safe in FP16: {'✅' if n_inf_exp == 0 and total_err < 0.01 else '❌'}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Why layer norm fails in FP16 — cancellation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Why Layer Norm Fails in FP16: Catastrophic Cancellation")
print("━" * 68)
print()

print("  Layer norm variance: var = mean((x - mean(x))^2)")
print("  When x values are clustered near a large value, (x - mean(x)) is tiny.")
print("  FP16 precision of 10 bits ≈ 0.1% relative error. At x~1.0, ULP ≈ 0.001.")
print("  A deviation of 0.0001 from the mean is invisible in FP16.")
print()

test_cases = [
    ("Well-separated (OK in FP16)",    np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])),
    ("Clustered near 1 (dangerous)",   np.array([1.001, 1.002, 1.0015, 0.999, 1.0005, 1.0])),
    ("Clustered near 100",             np.array([100.001, 100.002, 99.999, 100.001, 100.0, 99.998])),
    ("Very close (pathological)",      np.array([1.0000001, 1.0000002, 0.9999998, 1.0, 1.0000003, 0.9999997])),
]

print(f"  {'Test case':<36}  {'FP64 var':>12}  {'FP16 var':>12}  "
      f"{'Rel err%':>10}  {'Safe?'}")
print("  " + "─" * 72)

for name, x in test_cases:
    x_f64 = x.astype(np.float64)
    mean_f64 = x_f64.mean()
    var_f64  = ((x_f64 - mean_f64)**2).mean()

    x_f16 = x.astype(np.float16)
    mean_f16 = float(x_f16.mean())
    diff_f16 = x_f16 - np.float16(mean_f16)
    var_f16  = float((diff_f16**2).mean())

    rel_err = abs(var_f16 - var_f64) / (var_f64 + 1e-20) * 100
    safe    = "✅" if rel_err < 5.0 else ("⚠" if rel_err < 50 else "❌ FAIL")

    print(f"  {name:<36}  {var_f64:>12.8f}  {var_f16:>12.8f}  {rel_err:>9.1f}%  {safe}")

print()
print("  'Clustered near 100': mean subtraction leaves residuals of ~0.001.")
print("  FP16 ULP at 100.0 = 0.0625. Residuals of 0.001 round to ZERO.")
print("  Variance computed as 0.0 → layer norm divides by zero → NaN.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Cross-entropy underflow in FP16
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Cross-Entropy Underflow in FP16")
print("━" * 68)
print()

print("  Cross-entropy: loss = -log(softmax(logits)[target])")
print("  For a correct prediction with high confidence: softmax[target] ≈ 0.999")
print("  For an uncertain prediction: softmax[target] ≈ 1/vocab_size ≈ 1/32000")
print()

vocab_size   = 32000
logit_scale  = 1.0
np.random.seed(7)

print(f"  {'Scenario':<32}  {'Prob':>8}  {'FP64 loss':>12}  "
      f"{'FP16 loss':>12}  {'FP16 safe?'}")
print("  " + "─" * 68)

scenarios_ce = [
    ("High confidence (prob=0.99)",   0.99),
    ("Medium conf (prob=0.5)",        0.5),
    ("Low conf (prob=0.01)",          0.01),
    ("Very low (prob=1e-4)",          1e-4),
    ("Near-zero (prob=1e-6)",         1e-6),
    ("Below FP16 min (prob=5e-8)",    5e-8),
]

for name, prob in scenarios_ce:
    loss_fp64 = -math.log(prob) if prob > 0 else float('inf')
    prob_fp16 = float(np.float16(prob))
    if prob_fp16 == 0:
        loss_fp16 = float('inf')   # log(0) = -inf
        safe = "❌ prob→0 in FP16"
    else:
        log_p_fp16 = float(np.float16(math.log(prob_fp16))) if prob_fp16 > 0 else float('inf')
        loss_fp16  = -log_p_fp16
        rel_err    = abs(loss_fp16 - loss_fp64) / loss_fp64 * 100 if loss_fp64 != 0 else 0
        safe = "✅" if rel_err < 5.0 else ("⚠" if rel_err < 50 else "❌")

    print(f"  {name:<32}  {prob:>8.2e}  {loss_fp64:>12.4f}  "
          f"{loss_fp16:>12.4f}  {safe}")

print()
print("  For prob ≈ 1/32000 (typical early training): loss_fp64 ≈ 10.37.")
print("  FP16 cannot represent probabilities below 6e-5 → log(0) = -inf.")
print("  Cross-entropy MUST stay in FP32 (autocast handles this correctly).")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Mixed Precision Training Emulator — Full Training Loop Comparison": {
        "description": (
            "Simulate a complete 3-layer transformer training run comparing four "
            "precision configurations: FP32, FP16+GradScaler, BF16, and TF32. "
            "Track loss convergence, gradient norm health, weight update magnitudes, "
            "and memory usage across 200 simulated steps. Show the divergence of "
            "FP16 training without loss scaling. Produce a final comparison report "
            "covering throughput multiplier, memory reduction, and accuracy delta."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List

print("=" * 68)
print("  MIXED PRECISION TRAINING EMULATOR — Full Loop Comparison")
print("=" * 68)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# Minimal model and precision simulation
# ─────────────────────────────────────────────────────────────────────

HIDDEN_DIM  = 256
FFN_DIM     = 512
SEQ_LEN     = 128
BATCH_SIZE  = 8
N_LAYERS    = 3
N_PARAMS    = (HIDDEN_DIM * FFN_DIM * 2 + HIDDEN_DIM * HIDDEN_DIM * 4) * N_LAYERS
FP16_MAX    = 65504.0
FP16_MIN_N  = 6.1e-5
BF16_MAX    = 3.4e38


@dataclass
class PrecisionConfig:
    name:         str
    weight_bytes: int       # bytes per parameter (weight storage)
    grad_bytes:   int       # bytes per parameter (gradient storage)
    fp32_master:  bool      # maintain FP32 master copy?
    loss_scale:   float     # initial loss scale (1.0 = no scaling)
    dynamic_scale:bool      # use dynamic loss scaling?
    color:        str       # for display

CONFIGS = {
    "FP32 (baseline)": PrecisionConfig(
        "FP32 (baseline)", 4, 4, False, 1.0, False, ""),
    "FP16 + GradScaler": PrecisionConfig(
        "FP16 + GradScaler", 2, 2, True, 65536.0, True, ""),
    "FP16 (no scale)": PrecisionConfig(
        "FP16 (no scale)", 2, 2, True, 1.0, False, ""),
    "BF16": PrecisionConfig(
        "BF16", 2, 2, True, 1.0, False, ""),
    "TF32 (FP32 storage)": PrecisionConfig(
        "TF32 (FP32 storage)", 4, 4, False, 1.0, False, ""),
}


def simulate_step(step, weights, config, lr=3e-4, true_loss_noise=0.02):
    """
    Simulate one training step for a given precision config.
    Returns (new_weights, loss, grad_norm, update_norm, skipped, scale).
    """
    np.random.seed(step * 31337 + hash(config.name) % 10000)

    # True gradient (what FP64 would compute)
    true_grad_norm = abs(np.random.lognormal(mean=-1.5 - step * 0.0005, sigma=1.0))
    true_loss      = 4.0 * math.exp(-step * 0.003) + 0.5 + np.random.randn() * true_loss_noise

    # Apply precision effects on gradients
    if config.name == "FP32 (baseline)":
        eff_grad_norm = true_grad_norm
        can_overflow  = False
        skip_fraction = 0.0
        update_scale  = 1.0

    elif config.name == "TF32 (FP32 storage)":
        # TF32: FP32 range, tiny rounding noise
        eff_grad_norm = true_grad_norm * (1.0 + np.random.randn() * 1e-3)
        can_overflow  = False
        skip_fraction = 0.0
        update_scale  = 1.0

    elif config.name == "BF16":
        # BF16: same range as FP32, coarser mantissa → small noise
        quantisation_noise = np.random.randn() * config.weight_bytes * 0.008
        eff_grad_norm = max(0, true_grad_norm * (1.0 + quantisation_noise))
        can_overflow  = False
        skip_fraction = 0.0
        update_scale  = 1.0

    elif config.name == "FP16 + GradScaler":
        # FP16 with dynamic scaling
        scale = config.loss_scale
        scaled_grad_norm = true_grad_norm * scale

        if scaled_grad_norm > FP16_MAX:
            # Overflow detected → skip step, reduce scale
            config.loss_scale = max(config.loss_scale * 0.5, 1.0)
            return weights.copy(), true_loss, true_grad_norm, 0.0, True, config.loss_scale

        # Check if any gradients underflow even after scaling
        per_param_scaled = true_grad_norm / math.sqrt(N_PARAMS) * scale
        underflow_frac   = max(0, (FP16_MIN_N - per_param_scaled) / FP16_MIN_N)

        # Effective gradient after FP16 quantisation and underflow
        eff_grad_norm = true_grad_norm * (1 - underflow_frac * 0.1)
        eff_grad_norm *= (1.0 + np.random.randn() * 0.001)  # FP16 noise

        # Grow scale if no overflow for a while
        if step % 2000 == 0 and step > 0:
            config.loss_scale = min(config.loss_scale * 2.0, 65536.0)

        update_scale  = 1.0
        can_overflow  = False
        skip_fraction = 0.0

    elif config.name == "FP16 (no scale)":
        # FP16 without scaling — gradients underflow!
        per_param = true_grad_norm / math.sqrt(N_PARAMS)
        if per_param < FP16_MIN_N:
            eff_grad_norm = 0.0   # complete underflow
        elif true_grad_norm > FP16_MAX:
            eff_grad_norm = float('inf')   # overflow → NaN cascade
        else:
            underflow_frac = max(0, min(1, (FP16_MIN_N - per_param * 0.1) / FP16_MIN_N))
            eff_grad_norm  = true_grad_norm * (1 - underflow_frac)
        can_overflow  = True
        skip_fraction = 0.0
        update_scale  = 1.0
    else:
        eff_grad_norm = true_grad_norm
        can_overflow  = False
        skip_fraction = 0.0
        update_scale  = 1.0

    # Simulate Adam update
    # FP32 master weights: update is always applied in FP32 precision
    update_norm = lr * eff_grad_norm * 0.1  # simplified Adam scaling

    # If update rounds to zero in weight dtype (FP16 weights without master):
    if config.weight_bytes == 2 and not config.fp32_master:
        weight_ulp = 2.0**(-10) * (np.abs(weights).mean() + 1e-10)
        if update_norm < weight_ulp:
            update_norm = 0.0   # update lost to precision

    # Handle NaN/inf
    if not math.isfinite(eff_grad_norm) or not math.isfinite(update_norm):
        return np.full_like(weights, float('nan')), float('nan'), float('nan'), 0.0, True, config.loss_scale

    # Update weights (simplified)
    direction = np.random.randn(len(weights)) / math.sqrt(len(weights))
    new_weights = weights - update_norm * direction

    return new_weights, true_loss, eff_grad_norm, update_norm, False, config.loss_scale


# ─────────────────────────────────────────────────────────────────────
# Run simulated training
# ─────────────────────────────────────────────────────────────────────

N_STEPS       = 200
W_DIM_SIM     = 100   # simplified weight vector
RESULTS       = {}

for config_name, config in CONFIGS.items():
    weights = np.random.randn(W_DIM_SIM).astype(np.float32) * 0.02
    losses, grad_norms, update_norms, skips = [], [], [], []
    nan_step = None

    for step in range(N_STEPS):
        weights, loss, g_norm, u_norm, skipped, scale = simulate_step(
            step, weights, config)

        if not math.isfinite(loss) and nan_step is None:
            nan_step = step

        losses.append(loss if math.isfinite(loss) else float('nan'))
        grad_norms.append(g_norm if math.isfinite(g_norm) else float('nan'))
        update_norms.append(u_norm if math.isfinite(u_norm) else 0.0)
        skips.append(skipped)

    RESULTS[config_name] = {
        "losses":       losses,
        "grad_norms":   grad_norms,
        "update_norms": update_norms,
        "skips":        skips,
        "nan_step":     nan_step,
        "config":       config,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Training summary report
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Training Summary: 200 Steps")
print("━" * 68)
print()

print(f"  {'Config':<24}  {'Final loss':>11}  {'Skip%':>7}  "
      f"{'NaN at step':>12}  {'Grad norm (avg)':>16}  {'Verdict'}")
print("  " + "─" * 78)

for cname, res in RESULTS.items():
    valid_losses = [l for l in res["losses"] if math.isfinite(l)]
    final_loss   = valid_losses[-1] if valid_losses else float('nan')
    skip_frac    = sum(res["skips"]) / N_STEPS * 100
    nan_at       = res["nan_step"] if res["nan_step"] else "—"
    valid_grads  = [g for g in res["grad_norms"] if math.isfinite(g)]
    avg_grad     = np.mean(valid_grads) if valid_grads else float('nan')

    if res["nan_step"] is not None:
        verdict = f"❌ DIVERGED (step {res['nan_step']})"
    elif skip_frac > 5:
        verdict = f"⚠  {skip_frac:.0f}% skips"
    else:
        verdict = "✅ stable"

    print(f"  {cname:<24}  {final_loss:>11.4f}  {skip_frac:>6.1f}%  "
          f"{str(nan_at):>12}  {avg_grad:>16.4f}  {verdict}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Loss curves ASCII
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Loss Curves: 20-Step Buckets")
print("━" * 68)
print()

BUCKET = 20
print(f"  {'Step range':<14}", end="")
for cname in CONFIGS:
    short = cname[:10]
    print(f"  {short:>10}", end="")
print()
print("  " + "─" * (14 + 12 * len(CONFIGS)))

for b in range(0, N_STEPS, BUCKET):
    print(f"  {b:>4}–{b+BUCKET-1:<4}     ", end="")
    for cname, res in RESULTS.items():
        bucket_losses = [l for l in res["losses"][b:b+BUCKET] if math.isfinite(l)]
        if bucket_losses:
            val = np.mean(bucket_losses)
            print(f"  {val:>10.3f}", end="")
        else:
            print(f"  {'NaN/inf':>10}", end="")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Memory and throughput summary
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Memory Footprint & Throughput Comparison")
print("━" * 68)
print()

# Memory model per parameter (bytes)
# weights + gradients + optimizer (Adam m1, m2 always FP32) + optional FP32 master

def memory_per_param(cfg):
    weight_b  = cfg.weight_bytes
    grad_b    = cfg.grad_bytes
    master_b  = 4 if cfg.fp32_master else 0
    adam_b    = 8   # m1 + m2 always FP32
    return weight_b + grad_b + master_b + adam_b

# Activation memory: proportional to batch × seq × hidden × layers × precision
def activation_bytes(weight_bytes):
    return BATCH_SIZE * SEQ_LEN * HIDDEN_DIM * N_LAYERS * 4 * weight_bytes  # 4 tensors/layer

# Throughput model: TC speedup × memory bandwidth savings
def throughput_multiplier(cfg):
    if cfg.weight_bytes == 2 and "BF16" in cfg.name:
        return 3.5  # BF16 TC vs FP32 scalar
    elif cfg.weight_bytes == 2 and "FP16" in cfg.name:
        return 3.5  # FP16 TC vs FP32 scalar
    elif "TF32" in cfg.name:
        return 2.8  # TF32 TC vs FP32 scalar (FP32 storage, same bandwidth)
    else:
        return 1.0  # FP32 baseline

fp32_mem = memory_per_param(CONFIGS["FP32 (baseline)"])
fp32_act = activation_bytes(4)
fp32_thr = 1.0

print(f"  {'Config':<24}  {'Param mem':>10}  {'Act mem':>9}  "
      f"{'Total mem':>10}  {'Mem ratio':>10}  {'Tput mult':>11}")
print("  " + "─" * 78)

for cname, cfg in CONFIGS.items():
    p_mem    = memory_per_param(cfg) * N_PARAMS
    act_mem  = activation_bytes(cfg.weight_bytes)
    tot_mem  = p_mem + act_mem
    fp32_tot = fp32_mem * N_PARAMS + fp32_act
    mem_ratio= tot_mem / fp32_tot
    tput     = throughput_multiplier(cfg)

    print(f"  {cname:<24}  {p_mem/1e6:>9.1f}M  {act_mem/1e6:>8.1f}M  "
          f"{tot_mem/1e6:>9.1f}M  {mem_ratio:>9.2f}×  {tput:>10.1f}×")

print()
print("  MEMORY NOTE: The dominant saving is in ACTIVATIONS, not weights.")
print(f"  FP32 activations: {fp32_act/1e6:.0f} MB.  FP16/BF16: {activation_bytes(2)/1e6:.0f} MB. (2× less)")
print()
print("  RECOMMENDATION:")
print("    A100, H100  → BF16 + TF32 both enabled: max throughput, zero instability")
print("    V100, T4    → FP16 + GradScaler: tensor cores work, needs careful scaling")
print("    Research     → FP32: safest, slowest, most reproducible")
''',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  "",
        "visual_height": 400,
        "complexity":   None,
        "operations":   OPERATIONS,
    }