"""
Advanced Training Techniques
==============================

Label smoothing, knowledge distillation, quantisation, pruning, and
mixed precision are the tools that take a model from "works in research"
to "runs efficiently in production." They address the gap between
a model that achieves a target metric and one that can be served
at scale on constrained hardware.

"""

import textwrap
import re

TOPIC_NAME   = "Advanced Training Techniques"
DISPLAY_NAME = "05 · Advanced Training"
ICON         = "⚙️"
SUBTITLE     = "Label Smoothing · Distillation · Quantisation · Pruning · Mixed Precision"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — LABEL SMOOTHING

### The Overconfidence Problem

Standard cross-entropy loss is trained against ONE-HOT labels:

    y_hard = [0, 0, 1, 0, 0]    (only the true class gets probability 1)

This drives the model toward a dangerous state: to minimise cross-entropy
perfectly, the model must make the logit for the correct class
INFINITELY larger than all others. Because softmax never saturates at
exactly 0 or 1, the model keeps increasing the logit magnitude with
no natural stopping point.

    Loss = -log(softmax(z_true))
    To minimise: push z_true → +∞, push all other logits → -∞

    Consequences:
    1. The model becomes overconfident — it assigns near-zero probability
       to all classes except the predicted one.
    2. Generalisation suffers — the model has no incentive to learn
       meaningful structure in the logit space; it just amplifies the
       true class's signal without caring about relationships between classes.
    3. Calibration is poor — the predicted probability is 0.999 even for
       difficult, ambiguous examples.
    4. The model is brittle — a small perturbation to the input can flip
       the prediction because the decision boundary is razor-thin.


### The Label Smoothing Formula

    Proposed by Szegedy et al. (2016) in Inception v3. Instead of
    one-hot labels, use a SOFT distribution that is a mixture of the
    true label and a uniform distribution over all K classes:

    y_smooth(k) = (1 - α) · y_hard(k) + α / K

    Where:
        α = smoothing factor (typically 0.05 to 0.2)
        K = number of classes
        y_hard(k) = 1 if k is the true class, else 0

    For the TRUE class:   y_smooth = (1 - α) + α/K  = 1 - α(1 - 1/K)
    For all other classes: y_smooth = α/K

    Example with K=5, α=0.1:

    Hard label:   [0.000,  0.000,  1.000,  0.000,  0.000]
    Smooth label: [0.020,  0.020,  0.920,  0.020,  0.020]

    The true class still dominates, but wrong classes get a small floor.
    The model can never achieve zero loss — it always has some target
    probability to assign to non-true classes.


### Effect on the Cross-Entropy Gradient

    Cross-entropy loss:  L = -Σk y(k) log(p(k))

    Gradient with respect to logit zᵢ (with softmax):
    ∂L/∂zᵢ = pᵢ - yᵢ

    With hard labels: gradient for true class = pᵢ - 1
    → pushes logit until pᵢ = 1 (never actually achieved → infinite logit)

    With smooth labels: gradient for true class = pᵢ - (1 - α + α/K)
    → pushes logit until pᵢ = 1 - α + α/K  (FINITE target)
    → gradient is zero at a BOUNDED logit value

    This is why label smoothing prevents logit explosion and improves
    both calibration and representation quality.

    Diagram 1 — Hard vs Smooth Label Targets:

    Probability │ y_hard               y_smooth (α=0.1, K=5)
                │
          1.000 │    ████               ████   ← 0.920 (true class)
                │
          0.500 │
                │
          0.020 │              ░░░░░ ░░░░░ ░░░░░ ░░░░░  ← 0.020 (others)
          0.000 │ ░░░░ ░░░░░         ░░░░░ ░░░░░ ░░░░░
                └──────────────────────────────────────── classes
                   c1   c2    c3    c4    c5

    The model is still trained to maximise the true class, but non-true
    classes are no longer trained to exactly zero.


### Effect on Calibration vs Accuracy

    Label smoothing is primarily a REGULARISATION technique for
    classification. Its effects:

    Calibration:  IMPROVES significantly. The model can no longer be
                  infinitely confident. Predicted probabilities are more
                  meaningful (see Module 14 — Calibration).

    Accuracy:     Often slightly improves on the validation set due to
                  regularisation effect. Can be mixed on test sets.

    Representation quality: SIGNIFICANTLY IMPROVES. Hinton et al. (2019)
                  showed that label smoothing produces better intermediate
                  representations (tighter, more separated clusters in
                  the penultimate layer). This matters for transfer learning.

    Fine-tuning on top of a label-smoothed model: WORKS BETTER because
    the learned representations are more structured.

    Knowledge distillation: WORSE if the teacher was trained with label
    smoothing! The soft targets from a label-smoothed teacher contain
    less "dark knowledge" because the teacher was explicitly prevented
    from developing strong inter-class relationships. This is a known
    gotcha — if you plan to distil, train the teacher WITHOUT label smoothing.


### Label Smoothing as Implicit Regularisation

    Label smoothing is equivalent to adding a KL divergence penalty
    between the model's predicted distribution and the uniform distribution:

    L_smooth = L_hard + α · KL(Uniform || p)
             = L_hard + α · [H(Uniform) - H(Uniform, p)]
             = L_hard + α/K · Σk log(p(k)) + constant

    The penalty KL(Uniform || p) is large when the model is very
    confident (p far from uniform) and zero when p = Uniform.
    This is a soft constraint on the model's confidence.


### When to Use and When Not To

    USE label smoothing:
    ✓ Multi-class classification (image classification, NLP tagging)
    ✓ Training the final model for production deployment
    ✓ When calibration matters (medical, finance)
    ✓ As a transfer learning base (produces better representations)
    ✓ Noisy labels: smoothing absorbs some of the label noise

    DO NOT use label smoothing:
    ✗ When training a TEACHER for knowledge distillation
      (the soft targets lose inter-class structure)
    ✗ Object detection with bounding boxes
      (detection targets are already not one-hot; overlap can cause issues)
    ✗ Regression tasks (label smoothing is meaningless for continuous y)
    ✗ Binary classification (the effect is minimal; just use class weights)

    Typical α values:
    Image classification (ImageNet): α = 0.1  (Inception v3, ViT)
    Machine translation:             α = 0.1  (original Transformer paper)
    Speech recognition:              α = 0.05 to 0.1
    General default:                 α = 0.1


##### PART 2 — KNOWLEDGE DISTILLATION

### The Teacher-Student Framework

Introduced by Hinton, Vinyals & Dean (2015). The idea: train a large,
accurate "teacher" model first. Then train a smaller "student" model
to mimic the teacher's SOFT OUTPUT DISTRIBUTION, not just its hard
class predictions. The student learns from the teacher's "knowledge."

    Diagram 2 — Teacher-Student Distillation Pipeline:

    Training data
         │
         ├──────────────────────────────────────────────┐
         │                                              │
         ▼                                              ▼
    [TEACHER model]                              [STUDENT model]
    (large, accurate)                            (small, fast)
         │                                              │
         ▼                                              │
    Soft predictions                                    │
    (class probabilities at                             │
    temperature T)                                      │
         │                                              │
         └──────────────────────────────────► [Distillation loss]
                                                        │
                                              [Hard label loss]
                                                        │
                                                        ▼
                                               Update student weights


### Why Soft Targets Carry More Information

    Consider an image of a "3" written in a style close to an "8".
    Teacher predictions at T=1:   [0.000, 0.000, 0.001, 0.990, 0.000,
                                   0.000, 0.000, 0.000, 0.009, 0.000]

    The hard label [0,0,0,1,0,...] only tells the student "this is a 3."
    The soft distribution reveals: "this is a 3, but it slightly resembles
    an 8, and barely resembles a 2."

    This INTER-CLASS RELATIONSHIP is Hinton's "dark knowledge":
    the knowledge that certain classes are more similar to each other
    than to the rest. This structural information is absent from
    one-hot labels but is baked into the teacher's soft outputs.

    Quantifying the information gain:
    A hard label carries log₂(K) bits maximum (log₂(1000) = 10 bits).
    But on average, a hard label for ImageNet only carries ~1 bit
    (since the true class is almost always predicted correctly).
    The soft distribution carries information about ALL class relationships.


### Temperature Scaling for Distillation

    At inference temperature T=1, the teacher's softmax output is often
    peaked — most probability mass on one class, near-zero everywhere else.
    This gives the student very little information about wrong classes.

    Solution: RAISE the temperature T during training to make the
    distribution SOFTER (more uniform), revealing more inter-class structure:

    p_T(k) = exp(z_k / T) / Σⱼ exp(z_j / T)

    Low T (T → 0):  distribution collapses toward hard one-hot
    T = 1:          standard softmax
    High T (T → ∞): distribution approaches uniform

    Diagram 3 — Effect of Temperature on Teacher Soft Targets:

    Teacher logits: z = [3.2, 0.1, -0.4, 0.8, 0.2]

    T = 1:    [0.924, 0.022, 0.013, 0.028, 0.013]   ← peaked
    T = 2:    [0.716, 0.080, 0.049, 0.102, 0.053]   ← softer
    T = 4:    [0.462, 0.145, 0.111, 0.176, 0.106]   ← much softer
    T = 10:   [0.280, 0.192, 0.172, 0.211, 0.145]   ← nearly uniform

    At T=4, the student can clearly see "the 3 somewhat resembles an 8".
    At T=1, this signal is almost invisible (0.009 vs 0.990).


### Hinton's Combined Loss

    The student minimises a weighted combination of two losses:

    L = α · L_CE(y_hard, p_student_T1)     ← match the true label
      + (1-α) · T² · L_KL(p_teacher_T, p_student_T)   ← match the teacher

    Where:
        α ∈ [0,1] controls the balance (typically 0.1 to 0.5)
        T is the temperature (typically 2 to 8)
        T² factor: compensates for the softened gradients
                   (dividing by T in softmax reduces gradient magnitudes
                   by T², so we multiply back by T² to restore balance)

    The KL divergence term:
    KL(p_teacher || p_student) = Σk p_teacher(k) · log(p_teacher(k) / p_student(k))


### Dark Knowledge and Information in Wrong Answers

    The most surprising result from Hinton (2015): on MNIST, a student
    trained ONLY on soft targets from a teacher — with NO access to any
    true labels — achieves near-teacher accuracy.

    This is because the soft targets contain all the information needed:
        - Which class is correct (the highest probability)
        - How confident to be (the sharpness of the distribution)
        - Which other classes are similar (the second/third-highest)

    Practical insight: in some regimes, the ENTIRE training signal from
    hard labels can be replaced by teacher soft targets. This enables
    training a student even on UNLABELLED data, as long as you have
    a teacher that can generate soft predictions.


### Distillation Variants

    **Response-based (original Hinton)**:
        Distil from the teacher's final output logits.
        Simple, effective, widely used.

    **Feature-based (FitNets, Romero 2015)**:
        Train the student's intermediate representations to match
        the teacher's intermediate feature maps.
        Requires a projection layer if dimensions differ.
        Captures HOW the teacher processes inputs, not just what it outputs.

    **Relation-based**:
        Match pairwise or triplet relationships between examples,
        rather than individual activations.
        Preserves structural relationships in the embedding space.

    **Attention Transfer (Zagoruyko 2017)**:
        Match the spatial attention maps from the teacher's conv layers.
        Very effective for convolutional architectures.

    **Self-distillation / Born-Again Networks**:
        Train multiple generations: G1 is the original model. G2 is
        trained to distil from G1. G3 distils from G2. Ensembling all
        generations often beats any individual model.

    **Online distillation (DML - Zhang 2018)**:
        Teacher and student train simultaneously. Each model acts as
        the teacher for the other. No pre-trained teacher needed.


##### PART 3 — QUANTISATION

### Why Quantise?

A typical production motivation — deploying a 7B parameter LLM:

    FP32 (4 bytes/param): 7B × 4 = 28 GB  → needs 4× A100 (80GB each)
    FP16 (2 bytes/param): 7B × 2 = 14 GB  → fits on 1× A100 (80GB)
    INT8 (1 byte/param):  7B × 1 =  7 GB  → fits on 1× A100 (80GB) with room
    INT4 (0.5 bytes/param): 7B × 0.5 = 3.5 GB → fits on a consumer GPU (RTX 4090)

    Memory reduction:   8× vs FP32 with INT4
    Latency reduction:  2-4× (hardware can execute INT8 MACs faster)
    Energy reduction:   significant (integer arithmetic uses less power)
    Bandwidth reduction: loading weights from DRAM is the bottleneck for
                         LLM inference; INT4/INT8 reduces this directly.


### Number Format Anatomy

    Floating point numbers store three fields:
        sign (1 bit) | exponent | mantissa (fraction)

    ┌──────────────┬──────┬────────────┬────────────┬──────────────────────┐
    │ Format       │ Bits │ Exp. bits  │ Mant. bits │ Range / Precision    │
    ├──────────────┼──────┼────────────┼────────────┼──────────────────────┤
    │ FP32         │  32  │    8       │    23      │ ±3.4×10³⁸ / 7 dec.   │
    │ FP16         │  16  │    5       │    10      │ ±65,504 / 3-4 dec.   │
    │ BF16         │  16  │    8       │     7      │ ±3.4×10³⁸ / 2-3 dec. │
    │ FP8 (E4M3)   │   8  │    4       │     3      │ ±448 / 1 dec.        │
    │ FP8 (E5M2)   │   8  │    5       │     2      │ ±57,344 / <1 dec.    │
    │ INT8         │   8  │   N/A      │   N/A      │ -128 to 127          │
    │ INT4         │   4  │   N/A      │   N/A      │ -8 to 7              │
    └──────────────┴──────┴────────────┴────────────┴──────────────────────┘

    Key insight: BF16 has the SAME exponent range as FP32 (8 bits) but
    fewer mantissa bits (7 vs 23). This means BF16 can represent the
    SAME dynamic range as FP32, just with less precision. This is why
    BF16 requires no loss scaling (see Part 5) — it never overflows
    where FP32 would be fine.

    FP16 has only 5 exponent bits → dynamic range caps at ±65,504.
    Gradient values during training routinely exceed this → overflow/NaN.


### The Linear Quantisation Map

    Quantisation maps a real-valued float x to a low-bit integer x_q:

    QUANTISE:    x_q = clamp(round(x / S) + Z,  -2^(b-1),  2^(b-1) - 1)
    DEQUANTISE:  x_hat = S · (x_q - Z)

    Where:
        S = scale factor  (real-valued, one per tensor or per channel)
        Z = zero point    (integer, ensures zero maps to exactly 0)
        b = number of bits (e.g., 8 for INT8)

    Error:  x - x_hat = x - S·(round(x/S) + Z - Z)
                       = x - S·round(x/S)    [quantisation rounding error]
           Plus clipping error when x falls outside the representable range.


### Computing Scale and Zero-Point

    Given a tensor with values in range [x_min, x_max]:

    ASYMMETRIC (INT8 range: -128 to 127):
        S = (x_max - x_min) / 255
        Z = -round(x_min / S) - 128

    SYMMETRIC (INT8 range: -127 to 127):
        S = max(|x_min|, |x_max|) / 127
        Z = 0  (zero always maps to zero — simpler hardware)

    Diagram 4 — Symmetric vs Asymmetric Quantisation:

    Real value range: [-2.5, 8.0]

    ASYMMETRIC:                         SYMMETRIC:
    Maps [-2.5, 8.0] to [-128, 127]    Maps [-8.0, 8.0] to [-127, 127]
    Efficiently uses all 256 levels.    Wastes levels for the negative side.
    More complex hardware.              Simpler: Z=0, no zero-point add.
    Better for activations (often non-  Better for weights (often symmetric
    symmetric, e.g., post-ReLU ≥ 0).   around zero after regularisation).

    Asymmetric for activations + Symmetric for weights is the standard
    combination in quantisation frameworks like TensorRT and ONNX Runtime.


### Per-Tensor vs Per-Channel vs Per-Token Quantisation

    PER-TENSOR: one scale S for the entire weight matrix.
    Simplest, but misses the fact that different channels/rows may have
    very different value ranges. High quantisation error for outlier channels.

    PER-CHANNEL: one scale S per output channel (row of weight matrix).
    Captures channel-specific range → lower quantisation error.
    ~1% accuracy improvement over per-tensor for most models.
    Small overhead: one extra multiply per channel during dequant.

    PER-TOKEN (for LLM activations): one scale per token (row of activation).
    Important because LLM activations have MASSIVE outliers in specific
    channels that vary by token. Per-token reduces the impact of outliers
    on the scale factor shared with normal values.

    PER-GROUP: divide each weight row into groups of g=128 weights,
    one scale per group. G (group size) is a hyperparameter.
    GPTQ, AWQ, and llama.cpp use per-group quantisation.
    The most practical choice for 4-bit LLM quantisation today.


### Quantisation Error: Rounding and Clipping

    Total quantisation error has two additive components:

    1. ROUNDING ERROR: the gap between x and the nearest quantisation level.
       Cannot be eliminated; only reduced by using more bits.
       Roughly uniform in [-S/2, S/2] → expected squared error = S²/12.

    2. CLIPPING ERROR: values outside [x_min, x_max] are clamped to the
       boundary. MASSIVE error for outlier values.

    There is a fundamental trade-off:
        Wider range → smaller clipping error but larger rounding error (S is larger)
        Narrower range → smaller rounding error but larger clipping error

    The calibration step finds the optimal range by minimising total error,
    often using the mean squared quantisation error or KL divergence
    between the original and quantised activation distributions.


### Post-Training Quantisation (PTQ)

    Quantise a TRAINED float model without additional training.
    Requires only a small CALIBRATION DATASET (~100-1000 unlabelled examples).
    No gradient computation needed. Fast and cheap.

    Algorithm:
    1. Run the calibration dataset through the float model.
    2. Collect the activation distribution statistics at each layer.
    3. Choose scale and zero-point to minimise quantisation error.
    4. Apply the quantisation mapping to all weights and activations.
    5. Evaluate on a validation set. If accuracy is acceptable, done.

    Range calibration strategies:
        Min-Max: use the literal observed [min, max] of the calibration data.
                 Simple but sensitive to outliers.
        Percentile: use [p%, (100-p)%] percentiles (e.g., 0.1% to 99.9%).
                    More robust to outliers at the cost of some clipping.
        MSE minimisation: choose the range that minimises mean squared
                          quantisation error — optimal in theory.
        KL divergence: choose the range that minimises KL between the
                       float and quantised activation distributions.
                       Used by TensorRT.

    PTQ accuracy drop:
        FP32 → INT8: typically < 1% accuracy drop for CNNs, transformers
        FP32 → INT4: 1-5% drop for CNNs, larger for LLMs without calibration
        FP32 → INT4 with group quantisation: often < 1% on LLMs (GPTQ, AWQ)


### Quantisation-Aware Training (QAT)

    The model LEARNS to be robust to quantisation during training.
    Fake quantisation nodes are inserted: the forward pass simulates
    INT8 arithmetic while weights and activations remain in FP32.
    Gradients are computed through the fake quantisation using the
    STRAIGHT-THROUGH ESTIMATOR (STE).

    Fake quantisation node:
        Forward:  x_q = S · round(x / S)   [simulate INT8 rounding]
        Backward: ∂L/∂x = ∂L/∂x_q × 1     [pretend the round() has gradient 1]
        The STE ignores that round() has zero gradient almost everywhere.
        It is a biased gradient estimate but works very well in practice.

    Why QAT outperforms PTQ:
        The model adapts its weights to minimise loss GIVEN the
        quantisation constraints. Weights shift to use the full
        representable range efficiently. Layers compensate for each
        other's quantisation errors through end-to-end training.

    When to use QAT:
        INT4 or INT2 quantisation (PTQ accuracy drop is large)
        Domain-specific models where every percentage point matters
        Long-term production models worth the extra training cost

    QAT cost: requires retraining for 10-30% of the original training
    budget. Inference cost: identical to PTQ (no extra overhead).


### LLM-Specific Quantisation Algorithms

    Modern LLMs have a critical challenge: a small fraction of weight
    and activation channels have MASSIVE magnitudes (100-1000× the average).
    Standard quantisation sacrifices precision everywhere to accommodate
    these outliers.

    **GPTQ (Frantar et al., 2022)**:
        Layer-wise quantisation using second-order information (Hessian).
        Quantises each weight row by row, compensating downstream weights
        for the error introduced by quantising upstream ones.
        Achieves 4-bit quantisation with ~1% perplexity increase on LLMs.
        Widely used for running 70B models on consumer hardware.

    **AWQ (Lin et al., 2023) — Activation-aware Weight Quantisation**:
        Identifies the 1% of weight channels that correspond to the largest
        activations (the "important" channels). Scales these weights UP
        before quantisation (making them easier to represent accurately)
        and scales the corresponding activations DOWN (maintaining output).
        Reduces quantisation error on the most critical channels.
        Better than GPTQ on many benchmarks with less compute.

    **SmoothQuant (Xiao et al., 2023)**:
        Addresses activation quantisation (not just weights).
        Transfers the quantisation difficulty from activations to weights:
            X_new = X / s    (scale activations down by per-channel factor)
            W_new = W × s    (scale weights up by same factor)
        This mathematical equivalence makes both X and W easier to quantise.
        Enables W8A8 (weight 8-bit, activation 8-bit) with minimal accuracy loss.


### How Quantisation Error Accumulates Through Layers

    Each quantised layer introduces a small error ε in its output.
    This error is the INPUT to the next layer, where it gets amplified
    and combined with new quantisation error.

    For a chain of L layers:
    Approximate total error ≈ Σᵢ εᵢ · Πⱼ>ᵢ ||Wⱼ||

    In practice, if the weight norms are close to 1, errors accumulate
    roughly linearly. If norms are larger, earlier-layer errors are
    amplified exponentially. This is why:

    1. The first and last layers are often kept in FP32 or higher precision
       (they are not quantised, or quantised to a higher bit count).
    2. Calibration is done on the FULL model end-to-end, not layer by layer.
    3. QAT allows the model to learn to compensate for error accumulation.


##### PART 4 — PRUNING

### Why Prune?

Pruning removes redundant or unimportant parameters from a trained model.
Motivation:

    Smaller model file: fewer bytes on disk and in memory.
    Faster inference: fewer multiplications → lower latency.
    Lower energy: sparse arithmetic uses less power on compatible hardware.
    Deployment: models that fit on embedded hardware (microcontrollers,
                phones) with strict memory and compute budgets.

    Over-parameterisation insight: large neural networks are strongly
    over-parameterised. A 100M parameter model solving a task that
    "only requires" 10M parameters has 90M redundant parameters
    that can be removed with minimal accuracy loss.


### Unstructured Pruning

    Remove INDIVIDUAL WEIGHTS by setting them to zero. The weight matrix
    becomes SPARSE — most entries are zero, but the positions are arbitrary.

    Diagram 5 — Unstructured Pruning of a Weight Matrix (70% sparse):

    DENSE (original):           SPARSE (after pruning):
    ┌─────────────────────┐    ┌─────────────────────┐
    │ 0.8  0.3  0.7  0.2  │    │ 0.8  0.0  0.7  0.0  │
    │-0.5  0.1  0.4 -0.9  │    │ 0.0  0.0  0.4 -0.9  │
    │ 0.6 -0.2  0.3  0.5  │    │ 0.6  0.0  0.0  0.5  │
    └─────────────────────┘    └─────────────────────┘
    Every position used.        70% of weights are zero.

    Challenge: standard dense matrix multiply still processes zero weights.
    To get ACTUAL speedup, you need sparse matrix libraries and hardware
    support (or structured sparsity patterns — see below).

    When unstructured sparsity helps speed:
    - >90% sparsity: sparse formats (CSR, CSC) become more efficient
    - Dedicated sparse inference hardware (NVIDIA Ampere structured sparse)
    - CPU inference with NEON/AVX sparse kernels


### Structured Pruning

    Remove ENTIRE STRUCTURAL UNITS: neurons, filters (conv channels),
    attention heads, or even entire layers. The resulting model is DENSE
    but smaller — no special sparse hardware needed.

    Types:
    NEURON PRUNING: remove a neuron from a dense layer.
        The corresponding row in W[l] and column in W[l+1] are removed.
        Result: smaller dense weight matrices → direct speedup.

    FILTER PRUNING: remove an output filter from a conv layer.
        The feature map has fewer channels → directly reduces next layer's
        input channels → strict speedup proportional to pruning rate.

    ATTENTION HEAD PRUNING: remove an entire attention head in a Transformer.
        Michel et al. (2019): 20-40% of heads can be removed from BERT with
        <1% GLUE score drop. Many heads learn redundant patterns.

    LAYER PRUNING: remove an entire transformer block or residual block.
        Most aggressive; typically applied when compressing very deep models.
        DistilBERT (66% of original BERT) was effectively created this way.

    Diagram 6 — Structured vs Unstructured for Speedup:

    Unstructured (50% sparse):     Structured (50% filters removed):

    [0 W W 0 W 0 0 W]              [W W W W]
    [W 0 0 W 0 W W 0]              [W W W W]
    [0 W W 0 W 0 0 W]      vs      [W W W W]
    [W 0 0 W 0 W W 0]              [W W W W]

    Still 8 rows, sparse ops.       4 rows, DENSE ops.
    Slow on most hardware.          2x speedup immediately!


### Magnitude-Based Pruning

    The simplest and most widely used criterion: remove the weights
    with the SMALLEST ABSOLUTE VALUE.

    Rationale: weights with |w| ≈ 0 contribute little to the output.
    Removing them changes the network minimally.

    Algorithm (one-shot):
    1. Train the model to convergence.
    2. Compute the global magnitude threshold at the desired sparsity ratio.
       (e.g., for 70% sparsity: find the 70th percentile of all |w| values)
    3. Set all weights below the threshold to exactly zero.
    4. Fine-tune the remaining weights for a few epochs to recover accuracy.

    Limitation: magnitude does not equal importance. A small weight might
    be critical if it connects two important neurons. Gradient-based methods
    (see below) capture actual importance better.


### Gradual Magnitude Pruning (GMP)

    One-shot pruning at high sparsity causes severe accuracy drops because
    the network cannot recover with only a few fine-tuning steps.

    Gradual pruning (Zhu & Gupta, 2017) prunes incrementally over training:

    Algorithm:
    1. Start with a dense model (0% sparsity).
    2. Every K steps, increase sparsity target by a small delta:
       sparsity(t) = s_f · [1 - (1 - t/T)³]
       (cubic sparsity schedule: starts slow, ramps fast in the middle,
        slows down at the end to allow recovery near target sparsity s_f)
    3. After each pruning step, continue training — the model adapts
       its remaining weights to compensate for the removed ones.
    4. At the end of the schedule, fine-tune at fixed sparsity.

    Cubic schedule illustration:

    Sparsity │                      ╭─────────────────── s_f
             │                 ╭───╯
             │             ╭──╯
             │          ╭─╯
             │       ╭──╯
             │    ╭──╯
           0 │────╯
             └────────────────────────────────── Training step

    Result: 70-90% sparsity with <1% accuracy loss for many models.
    The network continuously adapts: when a weight is zeroed, the
    network reroutes the information through remaining weights.


### The Lottery Ticket Hypothesis

    Frankle & Carlin (2019) proved a striking result:

    "A randomly initialised, dense neural network contains a subnetwork
    (the 'winning ticket') that, when trained in isolation starting from
    its initialisation, can match the full network's test accuracy in
    at most the same number of training iterations."

    The finding: the SPARSE subnetwork that survives magnitude pruning
    was already "lucky" at initialisation — it had the right random
    weights to begin with. Only with those specific initial values
    does the sparse network train successfully.

    Algorithm to find the winning ticket:
    1. Randomly initialise the full network: θ₀
    2. Train to convergence: θ_T
    3. Prune the smallest-magnitude weights (create mask m)
    4. RESET the remaining weights to their initial values: θ₀ ∩ m
    5. Retrain the sparse network from this reset initialisation.
    6. Repeat steps 2-5 iteratively (iterative magnitude pruning)

    This reset to initial values is what makes it a "lottery ticket" —
    you need that original initialisation, not random re-initialisation.

    Implications:
    - Sparse networks CAN learn as well as dense networks — they just
      need the right initialisation.
    - This challenges the belief that over-parameterisation is needed
      during training. It is needed for FINDING the right subnetwork,
      but not for training it once found.
    - Finding the winning ticket is expensive (requires training the
      full network). Transfer learning approaches look for universal
      tickets that transfer across tasks.


### Movement Pruning

    Sanh et al. (2020) showed that for FINE-TUNING pre-trained models,
    magnitude is a poor importance criterion. The pre-trained weights
    have been arranged for a different task — their magnitudes reflect
    pre-training importance, not fine-tuning importance.

    Movement pruning scores weights by HOW MUCH THEY CHANGE during
    fine-tuning, not by their current magnitude:

    Score(wᵢⱼ) = wᵢⱼ · ∂L/∂wᵢⱼ    (weight × gradient)

    A positive score means the weight is moving in a direction that
    reduces loss — it is "important for learning." A near-zero score
    means the weight barely changes — it can be pruned.

    This is equivalent to the contribution of wᵢⱼ to the loss decrease,
    first-order approximated. Naturally suited to fine-tuning where the
    model starts from a pre-trained state and moves toward the new task.


### 2:4 Structured Sparsity (NVIDIA Ampere Architecture)

    A hardware-native sparsity pattern introduced with NVIDIA A100 GPUs:
    for every group of 4 consecutive weights, at most 2 can be non-zero.

    ┌──────────────────────────────────────┐
    │  Weight row:  [0.8, 0.0, 0.0, -0.5]  │  2 non-zero in every 4
    │               [0.0, 0.3, -0.7, 0.0]  │
    │               [0.2, 0.0, 0.0, 0.9]   │
    └──────────────────────────────────────┘

    This pattern (exactly 50% sparse) enables a 2x speedup in dense
    matrix multiply on Ampere/Hopper Tensor Cores through dedicated
    sparse matrix hardware. Standard unstructured sparsity does NOT
    achieve speedup on GPU; 2:4 structured sparsity DOES.

    NVIDIA's ASP (Automatic SParsity) applies 2:4 structured sparsity:
    1. Train the dense model normally.
    2. Apply 2:4 sparsification (choose the 2 largest in each group of 4).
    3. Fine-tune the sparse model with the 2:4 mask fixed.
    Accuracy loss: <1% for most tasks. Speedup: 1.5-2x on supported hardware.


##### PART 5 — MIXED PRECISION TRAINING

### Why Use Lower Precision During Training?

    Three complementary motivations:

    MEMORY: FP16 halves the memory used for activations and gradients.
    A batch that requires 40 GB in FP32 fits in 20 GB with FP16.
    This allows LARGER batches → better GPU utilisation.

    THROUGHPUT: NVIDIA Tensor Cores execute FP16/BF16 matrix multiplies
    4-8× faster than FP32 on the same hardware:
        V100:    FP32 = 14 TFLOPS    FP16 = 112 TFLOPS  (8× faster)
        A100:    FP32 = 19.5 TFLOPS  BF16 = 312 TFLOPS  (16× faster)
        H100:    FP32 = 67 TFLOPS    FP8  = 3,958 TFLOPS (59× faster)

    BANDWIDTH: loading weights from DRAM is often the bottleneck.
    FP16 weights are 2× smaller → 2× faster to load per batch.


### Number Format Anatomy: FP32 vs FP16 vs BF16

    FP32: 1 sign | 8 exponent | 23 mantissa
    Exponent bias 127. Range: ±3.4×10³⁸. Smallest normal: 1.2×10⁻³⁸.

    FP16: 1 sign | 5 exponent | 10 mantissa
    Exponent bias 15. Range: ±65,504. Smallest normal: 6.1×10⁻⁵.
    MAXIMUM value:  65,504  ← gradients can exceed this → OVERFLOW → NaN!
    MINIMUM normal: 6.1e-5  ← small gradients fall below this → UNDERFLOW → 0

    BF16: 1 sign | 8 exponent | 7 mantissa
    Same exponent as FP32! Range: ±3.4×10³⁸ (identical to FP32).
    Much less precision (7 mantissa bits vs 23 for FP32, 10 for FP16).
    Minimum normal: 1.2×10⁻³⁸ (same as FP32 → no underflow issue).

    Diagram 7 — FP16 vs BF16 Trade-offs:

    FP16:   WIDE range of representable fractions (10 mantissa bits)
            NARROW range of magnitudes (5 exponent bits)
            → Good for inference (model outputs are bounded)
            → Risky for training (gradient magnitudes can explode)

    BF16:   WIDE range of magnitudes (8 exponent bits, same as FP32)
            NARROW range of fractions (7 mantissa bits)
            → Good for training (no overflow) but less precise
            → Supported natively on A100/H100 (not on older GPUs)


### The Two Problems With FP16 in Training

    Problem 1 — OVERFLOW (gradient or activation exceeds ±65,504):
    Result: NaN or Inf propagates through backpropagation → training crashes.
    Common in early training when gradients are large, or with large batch sizes.

    Problem 2 — UNDERFLOW (small gradient rounds to zero):
    Gradients in range [10⁻⁷, 6×10⁻⁵] → underflow to 0 in FP16.
    Common in later training when gradients are small (near convergence).
    A weight that should receive a small gradient update gets ZERO update.
    This stalls learning and can cause the model to get stuck.

    Both problems must be solved for stable FP16 training.


### The Three-Component Recipe for FP16 Training

    Micikevicius et al. (2018) proposed the standard recipe:

    COMPONENT 1 — FP16 FORWARD AND BACKWARD PASS:
        Weights, activations, and gradients are stored and computed in FP16.
        The forward pass and backward pass both use FP16 arithmetic.
        Tensor Core accelerators are fully utilised.

    COMPONENT 2 — FP32 MASTER WEIGHTS:
        A FP32 copy of the weights is kept in memory throughout training.
        After each optimiser step, the FP32 master weights are updated.
        The FP16 weights used for computation are CAST from the FP32 masters.

        Why? Small gradient updates can be lost in FP16:
            FP32 weight: 0.123456789
            Gradient:    0.000001    (small update)
            FP16 result: 0.1234    (precision insufficient → update lost)
            FP32 result: 0.123457789 (update preserved)

    COMPONENT 3 — LOSS SCALING (for FP16 only, not BF16):
        Multiply the loss by a large scalar S before backpropagation.
        This shifts gradient values into FP16's representable range.
        After computing gradients, DIVIDE by S before the optimiser step.

        Without scaling:  gradient = 10⁻⁶ → underflows to 0 in FP16
        With scaling S=1024:  loss × 1024 → gradient = 10⁻³ → FP16 ok
        After unscaling:  gradient / 1024 = 10⁻⁶ → correct value in FP32

    Memory usage:
        FP16 weights:          2 bytes/param
        FP32 master weights:   4 bytes/param
        FP16 activations:      2 bytes/activation
        FP32 gradient in optim: 4 bytes/param × 2 (momentum + variance)
        Net vs FP32: roughly 40-60% memory reduction despite extra copies.


### Dynamic Loss Scaling

    Static loss scaling (fixed S): fragile. Too small → underflow.
    Too large → overflow (the scaled gradients exceed FP16 maximum).

    Dynamic loss scaling algorithm (automatically tunes S):

    Initialise: S = 2¹⁵ = 32,768  (large starting value)

    Each training step:
        1. Scale loss by S and compute FP16 gradients.
        2. Check if any gradient is Inf or NaN (overflow signal).

        IF any Inf/NaN:
            Skip the optimiser step (do not update weights).
            Reduce scale: S ← S / 2
            (halve S to prevent overflow next time)

        IF no Inf/NaN:
            Unscale gradients (divide by S).
            Perform optimiser step.
            Count consecutive "good" steps.
            Every 2000 good steps: S ← S × 2
            (increase S to maximise FP16 coverage)

    This algorithm converges to the largest S that keeps gradients in range.
    Typical stable S at convergence: 2¹² to 2²⁰.

    Diagram 8 — Dynamic Loss Scale Over Training:

    Scale │                       Grows every 2000 good steps
    2²⁰   │                   ╱──────────
          │               ╱──╯
    2¹⁶   │           ╱──╯
          │       ╱──╯
    2¹²   │   ╱──╯     Overflow at step N → halve scale
          │  ╱╲ ╲ ╲
    2⁸    │       ╲───
          └────────────────────────────────────────── Training step

    Scales grow slowly during stable training and halve quickly when
    overflow occurs. Eventually oscillates in a healthy range.


### Automatic Mixed Precision (AMP) in PyTorch

    PyTorch's AMP handles all three components transparently:

        from torch.cuda.amp import autocast, GradScaler

        scaler = GradScaler()   # handles dynamic loss scaling

        for X, y in dataloader:
            with autocast():    # forward + backward in FP16 automatically
                output = model(X)
                loss   = criterion(output, y)

            scaler.scale(loss).backward()    # scale then backward
            scaler.step(optimizer)           # unscale, check Inf, step
            scaler.update()                  # update scale factor

    AMP automatically decides which operations to run in FP16 vs FP32:

    FP16 (safe for precision reduction):     KEPT IN FP32:
    ┌────────────────────────────────────┐   ┌───────────────────────────────┐
    │ Linear layers (nn.Linear)          │   │ Softmax (numerical stability) │
    │ Convolutional layers               │   │ Log-softmax                   │
    │ BatchNorm (forward)                │   │ Loss computation              │
    │ Embedding lookup                   │   │ Layer norm (accumulation)     │
    │ Elementwise multiplications        │   │ Group norm                    │
    └────────────────────────────────────┘   └───────────────────────────────┘

    The whitelist/blacklist of operations is maintained by PyTorch and
    NVIDIA based on empirical stability research.


### BF16: The Modern Default

    On A100 and newer GPUs (TPUs also), BF16 is preferred over FP16:

        No loss scaling needed: same exponent range as FP32 → no overflow
        No FP32 master weights needed: BF16 has enough range to accumulate
        Simpler code: just cast to BF16, no GradScaler

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(X)
            loss   = criterion(output, y)
        loss.backward()
        optimizer.step()

    The tradeoff: BF16 has only 7 mantissa bits (vs 10 for FP16).
    This means BF16 rounds weights and activations more aggressively.
    In practice this rarely matters because gradients are noisy anyway,
    and the stability benefit far outweighs the precision cost.

    Rule of thumb:
        A100/H100/TPU available → BF16 (simpler, stable)
        V100/older GPU only → FP16 with loss scaling (BF16 not supported)
        Inference only → FP16 (wider range of representable fractions,
                                better for post-softmax probabilities)


### FP8: The Frontier

    NVIDIA Hopper (H100) and Transformer Engine introduce FP8 training:
        FP8 E4M3: 4 exponent + 3 mantissa bits (range ±448)
        FP8 E5M2: 5 exponent + 2 mantissa bits (range ±57,344)

    E4M3 is used for weights and activations (high precision needed).
    E5M2 is used for gradients (high range needed, less precision OK).

    FP8 requires per-tensor scaling with delayed scaling or just-in-time
    scaling to manage the tiny dynamic range. The Transformer Engine
    library from NVIDIA handles this automatically for attention and MLP.

    Speedup: FP8 GEMM is 2× faster than BF16, 4× faster than FP32.
    Memory: 8 bits = half of BF16 = quarter of FP32.
    Status (2024): supported in H100 + Transformer Engine, production-ready
    for LLM training at scale.


##### PART 6 — GRADIENT ACCUMULATION

### The Problem: Large Batch vs GPU Memory

    Research consistently shows that larger effective batch sizes improve
    training stability and allow higher learning rates (linear scaling rule):

        Learning rate scales linearly with batch size:
        lr_new = lr_base × (batch_size_new / batch_size_base)

    But GPU memory is finite. A 40 GB A100 may only fit batch size 8
    for a large language model. To achieve effective batch size 512,
    you would need 64 GPUs — or gradient accumulation.


### How Gradient Accumulation Works

    Run K "micro-batches" through the model, accumulating gradients
    without updating the weights. After K steps, perform one optimiser step.

    Pseudo-code:
        effective_batch = K × micro_batch_size
        optimizer.zero_grad()

        for step in range(K):
            micro_batch = next(dataloader)
            loss = model(micro_batch) / K        ← divide by K!
            loss.backward()                       ← ACCUMULATE gradients

        optimizer.step()                          ← ONE update per K steps

    The division by K ensures the gradient magnitude is the average
    over the full effective batch, not the sum (which would be K× too large).

    Mathematical equivalence:
        Σᵢ₌₁ᴷ ∂Lᵢ/∂θ / K  =  ∂(Σᵢ₌₁ᴷ Lᵢ/K)/∂θ

    Gradient accumulation IS mathematically equivalent to computing the
    gradient over the full batch — as long as the model and loss are
    the same (no stochastic elements between steps).


### Interaction With BatchNorm

    WARNING: gradient accumulation breaks BatchNorm.

    BatchNorm statistics are computed PER MICRO-BATCH:
        Running mean and variance are updated at each micro-batch.
        Different micro-batches produce different statistics.
        Gradients for the full effective batch are computed with
        INCONSISTENT normalisation statistics.

    This introduces noise and can destabilise training.

    Solutions:
    1. Use SyncBatchNorm across multiple GPUs (share statistics).
    2. Switch to Layer Norm or Group Norm (these normalise per-sample,
       not per-batch, so they are unaffected by accumulation).
    3. Freeze BatchNorm statistics during accumulation steps.

    This is why modern large models use Layer Norm / RMS Norm instead of
    Batch Norm — they work correctly with gradient accumulation.


### The Linear Scaling Rule for Learning Rate

    When the effective batch size is multiplied by K:
        Each gradient update sees K× more examples → K× more signal.
        A single weight update can afford to be K× larger → lr × K.

    Goyal et al. (2017) (Facebook, ResNet-50 in 1 hour):
        Linear scaling rule: lr_new = lr_base × K
        Warmup: use a lower lr for the first few epochs to allow the
        network to stabilise before using the large batch.

    Sqrt scaling (alternative for very large K):
        lr_new = lr_base × sqrt(K)
        More conservative; used when K is very large (>512).

    Practical note: gradient accumulation changes the effective batch
    size by K, so learning rate should also scale. Forgetting this
    is a common mistake when scaling up training.


### Interaction With Mixed Precision and Loss Scaling

    When combining gradient accumulation with FP16 + loss scaling:

        for step in range(K):
            with autocast():
                loss = model(micro_batch) / K
            scaler.scale(loss).backward()   ← accumulates SCALED gradients

        scaler.step(optimizer)   ← unscales, checks Inf, updates weights
        scaler.update()

    Important: the GradScaler should only call step() and update() ONCE
    per effective batch (after all K micro-batches). If you call step()
    inside the accumulation loop, you perform K updates instead of 1.

    PyTorch GradScaler handles this correctly when used as shown above.


### Gradient Accumulation vs Data Parallelism

    Both achieve larger effective batch size, but they are complementary:

    DATA PARALLELISM (multi-GPU):
        Each GPU holds a model copy and processes a micro-batch.
        Gradients are synchronised via all-reduce (summed across GPUs).
        Parallel compute → faster wall-clock time per step.
        Cost: multiple GPUs required.

    GRADIENT ACCUMULATION (single GPU):
        Sequential micro-batches on one GPU.
        No communication overhead.
        Wall-clock time scales with K (K× slower per effective batch).
        Cost: none (just time).

    COMBINED:
        Run data parallelism across N GPUs, each using gradient accumulation
        with K steps. Effective batch = N × K × micro_batch.
        Used by every large-scale LLM training run.

    Rule: data parallelism when you have multiple GPUs.
    Gradient accumulation when you want a larger batch on one GPU,
    or when increasing GPU count is not cost-effective.


##### PART 7 — COMBINED STRATEGIES & PRODUCTION TRAINING

### The Modern LLM Pre-Training Stack

    Every large language model pre-training run (GPT-4, LLaMA, Gemini)
    uses essentially the same combination of techniques:

    PRECISION:   BF16 mixed precision (no loss scaling needed on H100/A100)
    BATCH SIZE:  Gradient accumulation + data parallelism to reach
                 effective batch of 1M-8M tokens per step
    GRADIENTS:   Gradient clipping (clip_norm = 1.0) to prevent loss spikes
    OPTIMISER:   AdamW with linear warmup + cosine decay
    NORMALISATION: RMS Norm (works correctly with accumulation)
    PARALLELISM: Tensor parallel + pipeline parallel + data parallel

    Each technique solves one specific bottleneck:
    BF16 → 2× memory, 4× compute vs FP32
    Gradient accumulation → large batch without many GPUs
    Gradient clipping → prevents catastrophic loss spikes (loss cliff)
    RMS Norm → stable with variable batch statistics


### The Fine-Tuning / Compression Stack

    When deploying a pre-trained model to production with constraints:

    STEP 1 — Start with the pre-trained FP32 or BF16 model.

    STEP 2 — Fine-tune (if task-specific):
        Use label smoothing (α=0.1) for classification heads.
        Use distillation if the original model is your teacher.
        Use movement pruning during fine-tuning to identify important weights.

    STEP 3 — Prune:
        Apply structured pruning (remove attention heads, reduce layers).
        Or: apply 2:4 structured sparsity for NVIDIA GPU deployment.
        Fine-tune the pruned model for 10-20% of original fine-tuning steps.

    STEP 4 — Quantise:
        Apply PTQ with KL or percentile calibration.
        If accuracy drop is unacceptable: use QAT for 10-30% retraining.
        For LLMs: use GPTQ or AWQ for 4-bit quantisation.
        First and last layers often kept in higher precision (FP16 or INT8).

    STEP 5 — Validate and export:
        Export to TensorRT, ONNX, or platform-specific format.
        Validate accuracy on held-out test set.
        Measure latency and memory on target hardware.


### Decision Framework: Which Technique for Which Constraint

    ┌────────────────────────────────────────────────────────────────────┐
    │ CONSTRAINT              │ PRIMARY TECHNIQUE                        │
    ├────────────────────────────────────────────────────────────────────┤
    │ Model trains slowly on  │ Mixed precision (BF16/FP16)              │
    │ available GPUs          │                                          │
    ├────────────────────────────────────────────────────────────────────┤
    │ Batch size too small to │ Gradient accumulation                    │
    │ fit in GPU memory       │                                          │
    ├────────────────────────────────────────────────────────────────────┤
    │ Model too large for     │ Quantisation (INT8/INT4) for inference   │
    │ inference hardware      │                                          │
    ├────────────────────────────────────────────────────────────────────┤
    │ Inference is too slow   │ Structured pruning + quantisation        │
    │                         │ + hardware-aware compilation (TensorRT)  │
    ├────────────────────────────────────────────────────────────────────┤
    │ Model overfits / poor   │ Label smoothing + dropout                │
    │ calibration             │                                          │
    ├────────────────────────────────────────────────────────────────────┤
    │ Need smaller model with │ Knowledge distillation                   │
    │ similar accuracy        │                                          │
    ├────────────────────────────────────────────────────────────────────┤
    │ Pre-trained model is    │ Pruning + movement pruning during        │
    │ too large for fine-tune │ fine-tuning                              │
    └────────────────────────────────────────────────────────────────────┘


### Accuracy vs Speed vs Memory Trade-Off

    Starting from a FP32 dense baseline model:

    ┌────────────────────────────────────────────────────────────────────────┐
    │ Technique               │ Memory  │ Speed    │ Accuracy impact         │
    ├────────────────────────────────────────────────────────────────────────┤
    │ FP16 training           │ −50%    │ +2-4×    │ Neutral (same model)    │
    │ BF16 training           │ −50%    │ +4-8×    │ Neutral                 │
    │ Gradient accumulation   │ Neutral │ Neutral  │ Neutral (larger batch)  │
    │ Label smoothing         │ Neutral │ Neutral  │ +0 to +1% (reg. effect) │
    │ Knowledge distillation  │ −50-80% │ +2-5×    │ Student < Teacher by    │
    │ (smaller student)       │         │          │ ~1-5%                   │
    │ 50% unstructured pruning│ −50%    │ ~Neutral │ +0 to −1%               │
    │ 50% structured pruning  │ −50%    │ +2×      │ −1 to −3%               │
    │ INT8 quantisation (PTQ) │ −75%    │ +2-4×    │ <1% drop on most tasks  │
    │ INT4 quantisation       │ −87.5%  │ +4-8×    │ 1-5% drop               │
    │ INT4 + QAT              │ −87.5%  │ +4-8×    │ <1% drop                │
    │ 2:4 structured sparsity │ −50%    │ +1.5-2×  │ <1% drop                │
    └────────────────────────────────────────────────────────────────────────┘


### Order of Operations When Combining Techniques

    The order matters. Following wrong order creates inconsistencies:

    FOR TRAINING A NEW MODEL:
        1. Design architecture (can apply structured pruning at design time)
        2. Train with mixed precision (BF16/FP16) + gradient accumulation
        3. Apply label smoothing during training (if classification)
        4. Optionally apply gradual magnitude pruning during training
        5. Fine-tune on task-specific data if needed
        6. Quantise (PTQ or QAT) as the LAST step before deployment

    FOR COMPRESSING AN EXISTING MODEL:
        1. Start with the trained float model
        2. Fine-tune with distillation if a larger teacher is available
        3. Apply structured pruning → fine-tune (recover accuracy)
        4. Apply unstructured or 2:4 pruning → fine-tune (recover)
        5. Quantise (PTQ first; if insufficient, QAT)
        6. Validate, export, benchmark

    KEY RULE: Quantisation is ALWAYS the last step.
    Pruning after quantisation would change the model and invalidate
    the calibrated scale factors. Quantising first and then pruning
    can work but requires re-calibration.

    SECOND KEY RULE: Never prune then quantise then fine-tune.
    Fine-tuning after quantisation undoes the quantisation-aware
    weight arrangement. Always fine-tune BEFORE quantising.

"""


# ─────────────────────────────────────────────────────────────────────────────
# No OPERATIONS (theory-only module)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings — they're indented inside the dict literal,
# so each line has ~20 leading spaces. textwrap.dedent removes the common indent,
# producing clean left-aligned code that runs without IndentationError.
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()

# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_operations(st, scripts_dir=None, main_script=None):
    """Render all operations with code display and optional run buttons."""
    import streamlit as st  # local import so module stays importable without st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if scripts_dir is None:
        scripts_dir = None
    if main_script is None:
        main_script = None # _MAIN_SCRIPT

    scripts_available = main_script.exists()

    if "tok_step_status"  not in st.session_state:
        st.session_state.tok_step_status  = {}
    if "tok_step_outputs" not in st.session_state:
        st.session_state.tok_step_outputs = {}

    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**{op_data['description']}**")
            st.markdown("---")
            st.code(op_data["code"], language=op_data.get("language", "python"))


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
# render_operations() has been removed.  app.py owns all Streamlit rendering
# via its own render_operation() helper and strips callables from topic dicts
# inside load_topics_for() anyway — so a local render function is never called.

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────


def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 1150
    try:
        from Training_Core.visuals.Advtraining_visual import (
            ADVTRAINING_VISUAL_HTML,
            ADVTRAINING_VISUAL_HEIGHT,
        )
        visual_html   = ADVTRAINING_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = ADVTRAINING_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None,
        "operations":    OPERATIONS,
    }