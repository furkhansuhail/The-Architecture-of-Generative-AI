"""
Neural Networks — Supervised vs Unsupervised: What Makes the Difference?
=========================================================================

Both supervised and unsupervised neural networks are composed of identical
building blocks: matrix multiplications, nonlinear activations, and gradient
descent via backpropagation. The architecture of a single layer — the neuron,
the weight matrix, the ReLU — is IDENTICAL in both paradigms.

What differs is not the network. What differs is the training signal.

This module answers three questions precisely:
    1. What is the structural difference between supervised and unsupervised NNs?
    2. What makes a neural network supervised? What makes it unsupervised?
    3. How do the same building blocks produce such different behaviour?

By the end, the distinction will be crisp enough that you can classify any
neural network you encounter — and understand exactly why it falls where it does.

"""

import re
import textwrap
import numpy as np

DISPLAY_NAME = "01 · Supervised vs Unsupervised Neural Networks"
ICON         = "⚖️"
SUBTITLE     = "What makes the difference — a precise, side-by-side comparison"
TOPIC_NAME   = "Supervised vs Unsupervised Neural Networks"
VISUAL_HTML  = ""


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
What makes the difference ?

### The One-Sentence Answer

A neural network is supervised if a human-provided label y appears in its
loss function. It is unsupervised if the loss function contains only the
input data x and quantities derived from it.

That is the complete definition. Everything else follows from this.


---


### Part 1: What They Actually Share — The Common Foundation


Before the differences, understand what is identical:

    ┌──────────────────────────────────────────────────────────────────┐
    │   IDENTICAL IN BOTH SUPERVISED AND UNSUPERVISED NETWORKS         │
    │                                                                  │
    │   Neurons:          aˡ = σ( Wˡ aˡ⁻¹ + bˡ )                       │
    │   Weights:          Wˡ  (learned by gradient descent)            │
    │   Biases:           bˡ  (learned by gradient descent)            │
    │   Activations:      ReLU, GELU, sigmoid, tanh (same functions)   │
    │   Forward pass:     x → layer 1 → ... → layer L → output         │
    │   Backpropagation:  chain rule computing ∂L/∂W for every W       │
    │   Optimisers:       SGD, Adam, AdamW (same algorithms)           │
    │   Regularisation:   Dropout, L2 weight decay, BatchNorm          │
    │   Hyperparameters:  depth, width, learning rate, batch size      │
    └──────────────────────────────────────────────────────────────────┘

You cannot tell from looking at a single neuron — or even a single layer —
whether it belongs to a supervised or unsupervised network.

The difference lives entirely in:
    (1) What the output layer produces
    (2) What the loss function measures
    (3) What data is required to compute that loss


---


### Part 2: The Precise Definition — Loss Function Anatomy


Every neural network training loop has the form:

    Compute:    output = Network(input; W)
    Measure:    L = loss_function( output, ??? )
    Update:     W ← W − α · ∂L/∂W

The ??? is the critical question. What does the loss compare the output to?

    SUPERVISED:           L = loss( ŷ,  y  )
                                       ↑
                               human-provided label
                               (class name, price, bounding box, sentiment score)

    UNSUPERVISED:         L = loss( output, f(x) )
                                              ↑
                               something derived from x itself
                               (the input again, a masked part of x,
                                a similarity to another view of x)


### Formalising This:

Let D = {x₁, x₂, ..., xₙ} be unlabelled data.
Let D_L = {(x₁,y₁), ..., (xₙ,yₙ)} be labelled data.

    SUPERVISED loss:    L(W) = (1/n) Σᵢ loss( F(xᵢ; W),  yᵢ )
                                                          ↑
                                             yᵢ ∈ D_L required

    UNSUPERVISED loss:  L(W) = (1/n) Σᵢ loss( F(xᵢ; W),  g(xᵢ) )
                                                          ↑
                                             g(xᵢ) derived from xᵢ alone
                                             No yᵢ needed. D_L not required.


    **The Two Training Scenarios — Side by Side:**

    SUPERVISED TRAINING STEP:
    ─────────────────────────────────────────────────────────────────
    Input:    x = [image of a cat]
    Forward:  ŷ = Network(x) = [0.03, 0.94, 0.03]   ← 3-class softmax
    Label:    y = [0, 1, 0]                           ← "cat" (human provided)
    Loss:     L = CrossEntropy(ŷ, y) = −log(0.94)
    Signal:   ∂L/∂ŷ depends on y — gradient cannot be computed without y

    UNSUPERVISED TRAINING STEP (Autoencoder):
    ─────────────────────────────────────────────────────────────────
    Input:    x = [image of a cat]
    Forward:  x̂ = Decoder(Encoder(x))                ← reconstruction
    Target:   x itself (no label, no human involvement)
    Loss:     L = ||x − x̂||²
    Signal:   ∂L/∂x̂ = 2(x̂ − x) — gradient needs only x and x̂



---


### Part 3: What Makes a Network Supervised — The Five Markers


A neural network is supervised if and only if ANY of these are true:

    MARKER 1 — LABELLED DATASET REQUIRED
        Training requires a dataset of (x, y) pairs.
        If you cannot train it on raw x alone, it is supervised.
        Examples: image classification, sentiment analysis, regression,
                  object detection, named entity recognition.

    MARKER 2 — OUTPUT SPACE IS PREDEFINED BY LABELS
        The output layer size equals the number of human-defined classes.
        A 1000-class softmax output was defined because a human catalogued
        1000 ImageNet categories.
        An output is "cat" or "dog" only because a human decided these labels.

    MARKER 3 — LOSS REQUIRES y TO BE COMPUTABLE
        Cross-entropy:  L = −Σₖ yₖ log(ŷₖ)    ← needs yₖ
        MSE regression: L = (y − ŷ)²           ← needs y
        Hinge loss:     L = max(0, 1 − y·ŷ)    ← needs y
        Without y, the loss is undefined. Training cannot proceed.

    MARKER 4 — GRADIENTS CARRY LABEL INFORMATION
        ∂L/∂ŷ for cross-entropy: (ŷ − y)
        This gradient tells the network which direction is "wrong."
        The network moves toward what a human called correct.
        Label information propagates backwards through every layer.

    MARKER 5 — EVALUATION REQUIRES GROUND TRUTH
        Metrics: Accuracy, F1, AUC, RMSE, BLEU, IoU.
        All require known y to compare against.
        A supervised model cannot be evaluated on unlabelled data.


---


### Part 4: What Makes a Network Unsupervised — The Five Markers


A neural network is unsupervised if and only if ALL of these are true:

    MARKER 1 — RAW DATA ONLY
        Training requires only D = {x₁, ..., xₙ}.
        No human annotates any label. Any person could collect the data
        automatically (web scrape, sensor readings, text corpora).

    MARKER 2 — LOSS IS SELF-REFERENTIAL
        The loss compares the network's output to something derived
        entirely from x itself, with no external reference:
            Reconstruction:  compare x̂ to x
            Density:         evaluate log p(x) under the model
            Prediction:      compare predicted_mask to actual_mask_values
            Similarity:      compare two augmented views of x

    MARKER 3 — THE NETWORK DEFINES ITS OWN OBJECTIVE
        There is no "correct answer" supplied from outside.
        The network must construct a useful internal objective:
            Autoencoder: "reconstruct your input under compression"
            VAE: "reconstruct AND maintain a smooth latent distribution"
            GAN: "fool a simultaneously trained discriminator"
            Contrastive: "be invariant to augmentation"

    MARKER 4 — GRADIENTS CARRY NO LABEL INFORMATION
        ∂L/∂x̂ = 2(x̂ − x)   for MSE reconstruction
        This gradient only knows how much x̂ differs from x.
        No human ever told the network what the "right" output is.
        The direction of improvement comes from data geometry, not annotation.

    MARKER 5 — EVALUATION DOES NOT REQUIRE GROUND TRUTH
        Primary metrics: reconstruction error, FID (image quality),
        ELBO (VAE), perplexity (language), linear probe accuracy.
        Some metrics (linear probe) use a small held-out labelled set,
        but the training itself required none.


---


### Part 5: The Gradient — Where the Difference Lives in the Mathematics


Both supervised and unsupervised networks update weights by:
    W ← W − α · ∂L/∂W

The chain rule is identical. Backpropagation is identical.
The difference is in the first term of the chain:

    ∂L/∂W = ∂L/∂output × ∂output/∂W

    SUPERVISED:
        ∂L/∂output  =  ∂CrossEntropy(ŷ, y) / ∂ŷ  =  (ŷ − y)
        ↑ This quantity contains y. It points the network toward the human label.

    UNSUPERVISED (Autoencoder):
        ∂L/∂output  =  ∂MSE(x, x̂) / ∂x̂  =  2(x̂ − x)
        ↑ This quantity contains only x. It points the network toward the input.

    UNSUPERVISED (Contrastive):
        ∂L/∂z  =  ∂NT-Xent(zᵢ, zⱼ) / ∂z  =  function of (zᵢ, zⱼ, negatives)
        ↑ This quantity contains only embeddings of x. No y anywhere.

The gradient is the teacher. In supervised learning, the teacher is a human
(via the label y). In unsupervised learning, the teacher is the data itself.


    **Gradient Flow Comparison — Backpropagation Source:**

    SUPERVISED (Classification):
    ─────────────────────────────────────────────────────────────
    Human labels y
         ↓
    CrossEntropy(ŷ, y)  →  ∂L/∂ŷ = ŷ − y  [contains y]
         ↓
    Backprop through all layers
         ↓
    Every weight adjusted to predict y better

    UNSUPERVISED (Autoencoder):
    ─────────────────────────────────────────────────────────────
    Input data x
         ↓
    MSE(x̂, x)  →  ∂L/∂x̂ = 2(x̂ − x)  [contains only x]
         ↓
    Backprop through decoder AND encoder
         ↓
    Every weight adjusted to reconstruct x better

    SAME MACHINERY. DIFFERENT TEACHER.


---


### Part 6: How the Same Architecture Becomes Either


Consider a standard 3-layer feedforward network:
    Input → Hidden(64, ReLU) → Hidden(32, ReLU) → Output

This exact architecture can be trained in EITHER paradigm depending only
on the output layer design and the loss function:

    ┌────────────────────────────────────────────────────────────────────┐
    │  Architecture:  Input(p) → 64 → 32 → ???                           │
    │                                                                    │
    │  SUPERVISED (classification, p=784, 10 classes):                   │
    │    Output layer:   Dense(10, softmax)                              │
    │    Loss:           CrossEntropy(output, y)   y ∈ {0,...,9}         │
    │    Training data:  (image, digit_label) pairs                      │
    │    What it learns: which pixel patterns predict which digit        │
    │                                                                    │
    │  SUPERVISED (regression, p=8 features):                            │
    │    Output layer:   Dense(1, linear)                                │
    │    Loss:           MSE(output, y)             y ∈ ℝ                │
    │    Training data:  (features, price) pairs                         │
    │    What it learns: how features predict a continuous target        │
    │                                                                    │
    │  UNSUPERVISED (autoencoder encoder, p=784):                        │
    │    Output layer:   Dense(2, linear)   ← bottleneck                 │
    │    Decoder appended: 2 → 32 → 64 → Dense(784, sigmoid)             │
    │    Loss:           MSE(decoder_output, x)   — no y                 │
    │    Training data:  images only                                     │
    │    What it learns: the 2 most informative dimensions of the data   │
    │                                                                    │
    │  UNSUPERVISED (contrastive encoder, p=784):                        │
    │    Projection head appended: Dense(16, linear)                     │
    │    Loss:           NT-Xent(z_aug1, z_aug2)  — no y                 │
    │    Training data:  images only                                     │
    │    What it learns: invariant features across augmentations         │
    └────────────────────────────────────────────────────────────────────┘

The SAME 3-layer trunk learns completely different representations
depending only on what comes after layer 3 and how the loss is defined.


---


### Part 7: What Each Paradigm Learns — And Why It Differs


The loss function determines what the network optimises.
What it optimises determines what it learns.

    SUPERVISED — learns a decision boundary:
        The network learns the minimum information needed to map x to y.
        It may discard large parts of x that are irrelevant to the labels.
        Example: a digit classifier ignores ink colour, stroke width, exact
                 position — none of this predicts the digit class.
        The representation is task-specific. Transfer to other tasks is limited.

    UNSUPERVISED — learns data structure:
        The network must retain all information needed to reconstruct x,
        or to be invariant to augmentation, or to model the density.
        It cannot afford to throw away information it might need.
        Example: an autoencoder on digit images retains stroke width,
                 position, thickness — everything that reconstructs the pixel.
        The representation is task-agnostic. Transfer is often better.


    **What the Hidden Layers Learn in Each Case (MNIST example):**

    SUPERVISED (classifier trained on 60,000 labelled images):
    ─────────────────────────────────────────────────────────────
    Layer 1:  detects edges oriented at angles that discriminate digits
    Layer 2:  detects combinations of edges that look like digit parts
    Layer 3:  detects patterns specific to 0,1,2,...,9
    Output:   10-class softmax probability
    Discards: ink colour, position jitter, stroke smoothness

    UNSUPERVISED (autoencoder trained on 60,000 unlabelled images):
    ─────────────────────────────────────────────────────────────
    Encoder L1: detects all edges (not just label-discriminative ones)
    Encoder L2: detects all shapes (more general than classifier)
    Bottleneck: compresses to minimum sufficient representation
    Decoder:    reconstructs exact pixel values
    Retains:    digit identity AND style AND position AND stroke width

    CONSEQUENCE: The unsupervised encoder representation often transfers
    BETTER across tasks, because it lost less information.

---

### Part 8: The Data Requirements — The Practical Difference


This is the most operationally important distinction:

    SUPERVISED:
        Requires (xᵢ, yᵢ) pairs for every training example.
        Labelling cost is the bottleneck:
            - ImageNet:  1.2M images, ~$1M to label
            - Medical imaging: requires expert clinicians per image
            - Legal NER: requires trained annotators
        Label quality directly determines model quality.
        Training set size is bounded by labelling budget.

    UNSUPERVISED:
        Requires only {xᵢ} — raw, unlabelled data.
        Collection cost only:
            - Text: entire internet (hundreds of billions of tokens, ~$0 per token)
            - Images: billions of photographs, no annotation needed
            - Audio: streaming platforms, no transcription needed
        Training set size is bounded only by storage and compute.
        This is why GPT-4 trained on more data than any supervised model could.


    ┌──────────────────────────────────────────────────────────────────┐
    │  The Label Bottleneck:                                           │
    │                                                                  │
    │  Available unlabelled text:      ≈ 10¹² tokens    (virtually ∞)  │
    │  Available unlabelled images:    ≈ 10¹² images    (virtually ∞)  │
    │  Human-labelled training images: ≈ 10⁷ images     (expensive)    │
    │  Human-labelled medical scans:   ≈ 10⁵ images     (very costly)  │
    │                                                                  │
    │  Unsupervised learning scales to the first row.                  │
    │  Supervised learning is capped by the last two rows.             │
    └──────────────────────────────────────────────────────────────────┘


---


### Part 9: The Output — What Each Network Produces


The output type reveals the paradigm immediately:

    SUPERVISED OUTPUTS (mapped to a human-defined target space):
    ─────────────────────────────────────────────────────────────
    Classification:    ŷ ∈ [0,1]^K         (class probabilities)
    Regression:        ŷ ∈ ℝ               (continuous value)
    Segmentation:      ŷ ∈ [0,1]^{H×W×K}  (per-pixel class map)
    Bounding box:      ŷ ∈ ℝ⁴             (x, y, w, h)
    Named entity:      ŷ ∈ {B, I, O}^T    (per-token tag)

    Every output is defined by a human taxonomy. A "cat" class exists
    because a human decided to label cats.

    UNSUPERVISED OUTPUTS (derived from data structure):
    ─────────────────────────────────────────────────────────────
    Autoencoder:       x̂ ∈ ℝᵖ             (reconstruction of input)
    VAE encoder:       (μ, σ²) ∈ ℝᵈ       (parameters of latent Gaussian)
    GAN generator:     x̂ ∈ ℝᵖ             (synthesised sample)
    Contrastive enc:   z ∈ ℝᵈ             (embedding vector)
    Self-supervised:   x̂_masked ∈ ℝᵐ     (predicted masked values)

    No output is defined by a human. The output space is either the
    input space (reconstruction) or a learned latent space (embedding).

---

### Part 10: Decision Tree — Classifying Any Neural Network


Use this decision tree to classify any network you encounter:

    START: Look at the training procedure.
    │
    ├─ Does training require human-annotated labels y for every sample?
    │       YES ──► SUPERVISED
    │       NO  ──► Continue ↓
    │
    ├─ Does the loss function compare output to y directly?
    │       YES ──► SUPERVISED
    │       NO  ──► Continue ↓
    │
    ├─ Does the loss compare output to the input x (or a part of x)?
    │       YES ──► UNSUPERVISED (reconstruction family)
    │               → Autoencoder, VAE, Denoising AE, MAE
    │
    ├─ Does the loss compare embeddings of two views/augmentations of x?
    │       YES ──► UNSUPERVISED (contrastive family)
    │               → SimCLR, BYOL, DINO, CLIP
    │
    ├─ Does the loss arise from a game between two networks?
    │       YES ──► UNSUPERVISED (adversarial family)
    │               → GAN, WGAN, CycleGAN, StyleGAN
    │
    ├─ Does the loss predict a masked/corrupted/future portion of x?
    │       YES ──► UNSUPERVISED (self-supervised / predictive family)
    │               → BERT, GPT, MAE, wav2vec
    │
    └─ Does training use SOME labels but also large amounts of unlabelled data?
            YES ──► SEMI-SUPERVISED (hybrid — uses both paradigms)
                    → Label propagation, consistency regularisation, pseudo-labelling


    # =======================================================================================# 
    **Quick Classification Examples:**

    Network                     Loss Function               Verdict
    ──────────────────────────────────────────────────────────────────────────────────────────
    ResNet-50 (ImageNet)        CrossEntropy(ŷ, y_class)    SUPERVISED
    BERT (pretraining)          CrossEntropy(ŷ, x_masked)   UNSUPERVISED (self-supervised)
    BERT (fine-tuning)          CrossEntropy(ŷ, y_label)    SUPERVISED
    GPT-4 (pretraining)         CrossEntropy(ŷ, x_next)     UNSUPERVISED (self-supervised)
    GPT-4 (RLHF fine-tuning)    Reward from human rater     SUPERVISED (reward model)
    Autoencoder (anomaly det.)  MSE(x̂, x)                   UNSUPERVISED
    VAE (face generation)       MSE + KL(q||p)              UNSUPERVISED
    StyleGAN                    Adversarial (D vs G)        UNSUPERVISED
    SimCLR                      NT-Xent(z_i, z_j)           UNSUPERVISED (contrastive)
    CLIP (pretraining)          Contrastive(image, text)    UNSUPERVISED (contrastive)
    CLIP (zero-shot classif.)   Cosine sim to label text    SUPERVISED (uses label text)
    U-Net (medical segm.)       Dice + CrossEntropy(ŷ, y)   SUPERVISED
    ──────────────────────────────────────────────────────────────────────────────────────────
    
    Note: BERT and GPT are UNSUPERVISED during pretraining (predict tokens from
    tokens — no human wrote new labels) but SUPERVISED during fine-tuning
    (human labels y are introduced). The same model can switch paradigms.

---

### Part 11: Semi-Supervised Learning — The Hybrid


In practice, the two paradigms are often combined:

    SEMI-SUPERVISED:  train unsupervised on large D, supervised on small D_L

    Why?
        - Labels are expensive. D_L is always small.
        - Unlabelled data D is abundant and cheap.
        - Unsupervised pretraining extracts structure from D.
        - A small D_L then steers that structure toward the target task.

    The two-stage recipe:
        Stage 1: Pretrain encoder on D (unsupervised, no labels)
        Stage 2: Fine-tune on D_L (supervised, few labels)

    This is how BERT, GPT, CLIP, DALL-E, and AlphaFold are all built.
    The pretrain stage is unsupervised. The fine-tune stage is supervised.
    The combined system inherits the scale of unsupervised data and the
    precision of supervised labels.

    Result: consistently beats purely supervised models trained on D_L alone,
    often by large margins when |D_L| is small.


---


### Part 12: The Complete Comparison Table


    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                    SUPERVISED              UNSUPERVISED                             │
    ├─────────────────────────────────────────────────────────────────────────────────────┤
    │ Training data      (x, y) pairs            x only                                   │
    │ Loss depends on    x and y                 x only (and model output)                │
    │ Gradient source    Human labels y          Data structure in x                      │
    │ Output space       Human-defined classes   Latent space or input space              │
    │ Evaluation         Accuracy, F1, RMSE…     Recon error, FID, linear probe           │
    │ Data scalability   Bounded by label budget  Bounded by compute only                 │
    │ What it optimises  Prediction of y          Structure/density/similarity in x       │
    │ Representation     Task-specific            Task-agnostic, more general             │
    │ Transfer learning  Retrain output layer     Freeze encoder, add any head            │
    │ Typical use        When labels available    When labels scarce or expensive         │
    │ Modern examples    ResNet, BERT fine-tune   BERT pretrain, GPT, VAE, GAN, SimCLR    │
    │ Architecture diff  Classifier/regressor head Reconstruction/generation/embed head   │
    │ Backprop path      Output→label mismatch    Output→input mismatch (or similarity)   │
    └─────────────────────────────────────────────────────────────────────────────────────┘

---

### Part 13: Common Misconceptions — Cleared Up


    MISCONCEPTION 1: "Unsupervised networks don't use backpropagation."
    TRUTH: They use identical backpropagation. The chain rule is identical.
           Only the loss at the output end differs.

    MISCONCEPTION 2: "Unsupervised networks can't learn class structure."
    TRUTH: They frequently do — spontaneously. A VAE trained on digit images
           produces latent clusters corresponding to digit classes, despite
           never being told what a digit class is. The class structure exists
           in the data; unsupervised methods discover it.

    MISCONCEPTION 3: "Self-supervised learning is supervised because it
                      has labels (the masked tokens)."
    TRUTH: Self-supervised learning is a form of unsupervised learning.
           The "labels" (masked tokens, future tokens) are derived
           automatically from x — no human wrote them. The definition of
           supervised requires human-provided y, not any label whatsoever.

    MISCONCEPTION 4: "Supervised networks are always better."
    TRUTH: On the specific task they were trained for, yes.
           On everything else: unsupervised representations often transfer
           better because they retained more information.
           GPT-4 (largely unsupervised pretraining) outperforms supervised
           models on tasks those supervised models were explicitly trained for.

    MISCONCEPTION 5: "Unsupervised networks produce worse representations."
    TRUTH: The opposite is often true at scale. CLIP (contrastive,
           unsupervised) achieves better zero-shot ImageNet accuracy than
           supervised ResNets trained on the full labelled ImageNet dataset.

    MISCONCEPTION 6: "The architecture tells you whether it's supervised."
    TRUTH: No. The same architecture can be trained either way.
           A 3-layer MLP is supervised if trained with cross-entropy on labels,
           and unsupervised if trained as an autoencoder without labels.
           Architecture is shared infrastructure. Paradigm is training regime.

---

### Part 14: Summary — The Three Root Differences


Every other difference between supervised and unsupervised neural networks
follows from exactly three root distinctions:

    ROOT 1 — THE SOURCE OF THE TRAINING SIGNAL:
        Supervised:    human knowledge, encoded as labels y
        Unsupervised:  the data's own structure, encoded in x

    ROOT 2 — WHAT THE LOSS MEASURES:
        Supervised:    distance from the network's output to a human label
        Unsupervised:  distance from the network's output to a property of x

    ROOT 3 — THE DATA REQUIREMENT:
        Supervised:    needs (x, y) pairs — expensive, scarce
        Unsupervised:  needs x alone — cheap, abundant

    Everything else — output type, evaluation metrics, representation quality,
    scalability, transfer learning behaviour — is a consequence of these three.

    The neural network itself: weights, activations, backprop, Adam — all shared.
    The paradigm: determined entirely by the loss function and what it requires.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ─────────────────────────────────────────────────────────────────────────────────
    Dimension               Supervised                  Unsupervised
    ─────────────────────────────────────────────────────────────────────────────────
    Data required           O(n) labelled pairs         O(n) raw samples
    
    Loss computation        O(output_dim) + label       O(output_dim) or O(n²)
                                                        (contrastive: O(batch²))
                                                        
    Backprop cost           O(2 × forward pass)         O(2 × forward pass)
    
    Label annotation cost   O(n × cost_per_label)       O(0)
    
    Output dim              Fixed by #classes/targets   Chosen (latent dim)
    
    Evaluation complexity   O(n × metric)               O(n × metric) or linear probe
    
    ──────────────────────────────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Same Architecture, Two Paradigms": {
        "description": "Identical 3-layer MLP trained supervised vs unsupervised — direct comparison",
        "runnable": True,
        "code": '''
"""
================================================================================
SAME ARCHITECTURE — SUPERVISED vs UNSUPERVISED
================================================================================

We take an IDENTICAL 3-layer MLP and train it two ways:

    MODE A (Supervised):    Input → 64 → 32 → 3-class softmax
                            Loss: CrossEntropy(ŷ, y)
                            Requires: labelled (x, y) pairs

    MODE B (Unsupervised):  Input → 64 → 4 (bottleneck) → 64 → Input
                            Loss: MSE(x̂, x)
                            Requires: raw x only

The weights, activations, and backprop are IDENTICAL.
Only the output layer and loss change.
================================================================================
"""

import numpy as np
np.random.seed(42)

def relu(z):    return np.maximum(0, z)
def relu_g(z):  return (z > 0).astype(float)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Dataset: 3 clusters in 12D ────────────────────────────────────────────────

dim   = 12
n_per = 300
c0    = np.random.randn(n_per, dim) * 0.5 + np.array([3, 3] + [0]*(dim-2))
c1    = np.random.randn(n_per, dim) * 0.5 + np.array([-3, 3] + [0]*(dim-2))
c2    = np.random.randn(n_per, dim) * 0.5 + np.array([0, -4] + [0]*(dim-2))
X     = np.vstack([c0, c1, c2])
y     = np.array([0]*n_per + [1]*n_per + [2]*n_per)
X     = (X - X.mean(0)) / (X.std(0) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# MODE A: SUPERVISED — CrossEntropy(ŷ, y)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  MODE A: SUPERVISED  (CrossEntropy loss, labels required)")
print("=" * 65)
print(f"  Architecture: {dim} → 64 → 32 → 3  (softmax output)")
print(f"  Loss:         CrossEntropy(ŷ, y)")
print(f"  Data needed:  (x, y) labelled pairs\\n")

# Weights
SW1 = np.random.randn(dim, 64) * np.sqrt(2/dim);    Sb1 = np.zeros((1,64))
SW2 = np.random.randn(64, 32)  * np.sqrt(2/64);    Sb2 = np.zeros((1,32))
SW3 = np.random.randn(32, 3)   * np.sqrt(2/32);    Sb3 = np.zeros((1,3))
lr  = 0.01

Y_oh = np.zeros((len(y), 3)); Y_oh[np.arange(len(y)), y] = 1   # one-hot labels

for epoch in range(80):
    idx = np.random.permutation(len(X))
    for s in range(0, len(X), 64):
        Xb = X[idx[s:s+64]];  Yb = Y_oh[idx[s:s+64]]

        # Forward (same as any MLP forward pass)
        h1 = relu(Xb@SW1 + Sb1)
        h2 = relu(h1@SW2 + Sb2)
        yh = softmax(h2@SW3 + Sb3)

        # Loss: CrossEntropy — REQUIRES LABELS Yb
        ce_loss = -np.mean(np.sum(Yb * np.log(yh + 1e-15), axis=1))

        # Backward — gradient carries label info: (ŷ − y)
        n = Xb.shape[0]
        dz3 = (yh - Yb) / n          # ← THIS TERM CONTAINS y
        dSW3 = h2.T@dz3;  dSb3 = dz3.sum(0, keepdims=True)
        dh2 = dz3@SW3.T * relu_g(h1@SW2+Sb2)
        dSW2 = h1.T@dh2;  dSb2 = dh2.sum(0, keepdims=True)
        dh1 = dh2@SW2.T * relu_g(Xb@SW1+Sb1)
        dSW1 = Xb.T@dh1;  dSb1 = dh1.sum(0, keepdims=True)

        SW3 -= lr*dSW3; Sb3 -= lr*dSb3
        SW2 -= lr*dSW2; Sb2 -= lr*dSb2
        SW1 -= lr*dSW1; Sb1 -= lr*dSb1

# Evaluate supervised accuracy
h1_f = relu(X@SW1+Sb1); h2_f = relu(h1_f@SW2+Sb2)
preds_sup = softmax(h2_f@SW3+Sb3).argmax(1)
acc_sup   = (preds_sup == y).mean()
rep_sup   = h2_f   # 32-dim representations from layer 2

print(f"  Final accuracy (uses all labels): {acc_sup:.1%}")
print(f"  Representation: 32-dim, shaped by label supervision")
print(f"\\n  What the gradient contained: (ŷ - y) — label y in every update")
print(f"  What was thrown away: any feature irrelevant to the 3 classes")


# ─────────────────────────────────────────────────────────────────────────────
# MODE B: UNSUPERVISED — MSE Reconstruction (same shared trunk)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\\n{'='*65}")
print(f"  MODE B: UNSUPERVISED  (MSE reconstruction, NO labels)")
print(f"{'='*65}")
print(f"  Architecture: {dim} → 64 → 4 (bottleneck) → 64 → {dim}  (autoencoder)")
print(f"  Loss:         MSE(x̂, x)  ← no y anywhere")
print(f"  Data needed:  raw x only\\n")

# IDENTICAL first two layers to Mode A in spirit, but bottleneck at 4 + decoder
UW1 = np.random.randn(dim, 64) * np.sqrt(2/dim);    Ub1 = np.zeros((1,64))
UW2 = np.random.randn(64, 4)  * np.sqrt(2/64);     Ub2 = np.zeros((1,4))   # bottleneck
UW3 = np.random.randn(4, 64)  * np.sqrt(2/4);      Ub3 = np.zeros((1,64))  # decoder
UW4 = np.random.randn(64, dim) * np.sqrt(2/64);    Ub4 = np.zeros((1,dim))
lr_u = 0.005

for epoch in range(80):
    idx = np.random.permutation(len(X))
    for s in range(0, len(X), 64):
        Xb = X[idx[s:s+64]]   # NO labels used

        # Forward
        h1  = relu(Xb@UW1 + Ub1)
        z   = relu(h1@UW2  + Ub2)   # latent code
        d1  = relu(z@UW3   + Ub3)
        xh  = d1@UW4 + Ub4          # reconstruction (linear output)

        # Loss: MSE — compares x̂ to x, NO label y
        n   = Xb.shape[0]
        dxh = 2*(xh - Xb) / (n * dim)   # ← GRADIENT: contains only x, not y

        # Backward
        dUW4 = d1.T@dxh;   dUb4 = dxh.sum(0, keepdims=True)
        dd1  = dxh@UW4.T * relu_g(z@UW3+Ub3)
        dUW3 = z.T@dd1;    dUb3 = dd1.sum(0, keepdims=True)
        dz   = dd1@UW3.T * relu_g(h1@UW2+Ub2)
        dUW2 = h1.T@dz;    dUb2 = dz.sum(0, keepdims=True)
        dh1  = dz@UW2.T * relu_g(Xb@UW1+Ub1)
        dUW1 = Xb.T@dh1;   dUb1 = dh1.sum(0, keepdims=True)

        for w, g in [(UW1,dUW1),(Ub1,dUb1),(UW2,dUW2),(Ub2,dUb2),
                     (UW3,dUW3),(Ub3,dUb3),(UW4,dUW4),(Ub4,dUb4)]:
            w -= lr_u * g

# Unsupervised representations (latent codes)
h1_u = relu(X@UW1+Ub1)
rep_uns = relu(h1_u@UW2+Ub2)   # 4-dim latent codes

# Evaluate: linear probe with 5 labels per class (15 total)
from numpy.linalg import lstsq
few_idx = [i for c in range(3) for i in np.where(y==c)[0][:5]]
Xf, yf = rep_uns[few_idx], y[few_idx]
Yf_oh  = np.zeros((15, 3)); Yf_oh[np.arange(15), yf] = 1
W_lin, _, _, _ = lstsq(np.c_[Xf, np.ones(15)], Yf_oh, rcond=None)
preds_uns = (np.c_[rep_uns, np.ones(len(rep_uns))]@W_lin).argmax(1)
acc_uns   = (preds_uns == y).mean()

# Reconstruction error
xh_all = relu(rep_uns@UW3+Ub3)@UW4+Ub4
recon_err = np.mean((xh_all - X)**2)

print(f"  Reconstruction MSE:                  {recon_err:.4f}")
print(f"  Linear probe accuracy (15 labels):   {acc_uns:.1%}")
print(f"\\n  What the gradient contained: 2(x̂-x) — no label y ever")
print(f"  What was preserved: full data structure (all 3 clusters discoverable)")


# ─────────────────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print(f"\\n{'='*65}")
print(f"  SIDE-BY-SIDE COMPARISON")
print(f"{'='*65}")
rows = [
    ("Architecture",      "Input→64→32→3 (classifier)",  "Input→64→4→64→Input (AE)"),
    ("Output",            "Class probabilities (3)",       "Reconstruction (12D)"),
    ("Loss function",     "CrossEntropy(ŷ, y)",            "MSE(x̂, x)"),
    ("Labels required",   "YES — all 900 samples",         "NO — zero labels used"),
    ("Gradient contains", "(ŷ − y)  ← y present",          "2(x̂ − x)  ← no y"),
    ("Accuracy (full)",   f"{acc_sup:.1%}  (900 labels)",  f"N/A (no prediction task)"),
    ("Accuracy (probe)",  "N/A",                           f"{acc_uns:.1%}  (15 labels)"),
    ("Backprop algo",     "IDENTICAL",                     "IDENTICAL"),
    ("Optimiser",         "IDENTICAL (SGD)",               "IDENTICAL (SGD)"),
    ("Activation fn",     "IDENTICAL (ReLU)",              "IDENTICAL (ReLU)"),
]
print(f"\\n  {'Dimension':<22} {'SUPERVISED':<32} {'UNSUPERVISED'}")
print(f"  {'─'*80}")
for name, sup, uns in rows:
    print(f"  {name:<22} {sup:<32} {uns}")

print("  " + "-"*71)
print("  KEY INSIGHT:")
print("    The shared trunk (weights, activations, backprop, optimiser) is IDENTICAL.")
print("    The difference is entirely in:")
print("        (1) What the output layer produces")
print("        (2) What the loss function compares it to")
print("        (3) Whether that comparison needs y or just x")
print()
print("    Supervised:    loss needs y  ->  gradient carries label info  ->  decision boundary")
print("    Unsupervised:  loss needs x  ->  gradient carries data info   ->  data structure")
''',
    },

    "Loss Function Dissection": {
        "description": "Compute and compare every major supervised and unsupervised loss — see exactly what each measures",
        "runnable": True,
        "code": '''
"""
================================================================================
LOSS FUNCTION DISSECTION
================================================================================

The loss function IS what makes a network supervised or unsupervised.
We compute every major loss and show exactly what it measures and what it needs.
================================================================================
"""

import numpy as np
np.random.seed(42)

print("=" * 70)
print("  LOSS FUNCTION DISSECTION — SUPERVISED vs UNSUPERVISED")
print("=" * 70)

# ── Toy inputs ────────────────────────────────────────────────────────────────
p     = 6    # input dimension
K     = 3    # number of classes
d     = 2    # latent dimension (for unsupervised)
n     = 4    # batch size

x     = np.array([[0.8, 0.2, 0.5, 0.1, 0.9, 0.4],   # raw inputs
                   [0.1, 0.7, 0.3, 0.8, 0.2, 0.6],
                   [0.6, 0.4, 0.7, 0.2, 0.5, 0.8],
                   [0.3, 0.9, 0.2, 0.6, 0.1, 0.7]])

y_cls = np.array([0, 2, 1, 0])                         # class labels (SUPERVISED)
y_reg = np.array([2.3, -1.1, 0.8, 3.2])               # regression targets (SUPERVISED)

# Simulated network outputs
y_hat_cls = np.array([[0.7, 0.2, 0.1],                # softmax probabilities
                       [0.1, 0.1, 0.8],
                       [0.2, 0.7, 0.1],
                       [0.8, 0.1, 0.1]])

y_hat_reg = np.array([2.0, -0.8, 1.2, 2.9])           # regression predictions

x_hat     = x + np.random.randn(*x.shape) * 0.1       # imperfect reconstruction
mu        = np.random.randn(n, d) * 0.5                # VAE encoder mean
log_var   = np.random.randn(n, d) * 0.3 - 0.5         # VAE log variance

z_i = np.random.randn(n, d);  z_i /= np.linalg.norm(z_i, axis=1, keepdims=True)
z_j = z_i + np.random.randn(n, d) * 0.2;  z_j /= np.linalg.norm(z_j, axis=1, keepdims=True)

sep = "─" * 70


# ──────────────────────────────────────────────────────────────────────────────
# SUPERVISED LOSSES
# ──────────────────────────────────────────────────────────────────────────────

print(f"\\n{'SUPERVISED LOSSES':^70}")
print(f"  (All require label y — cannot be computed without it)\\n")


# 1. Cross-Entropy
print(sep)
print("  1. CROSS-ENTROPY LOSS  (multi-class classification)")
print(sep)
Y_oh = np.zeros((n, K)); Y_oh[np.arange(n), y_cls] = 1
ce_per_sample = -np.sum(Y_oh * np.log(y_hat_cls + 1e-15), axis=1)
ce_loss = ce_per_sample.mean()
grad_ce = (y_hat_cls - Y_oh) / n
print(f"  Formula:  L = -(1/n) Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)")
print(f"  Needs:    y (one-hot class labels) — SUPERVISED")
print(f"  Value:    {ce_loss:.4f}")
print(f"  Gradient: ∂L/∂ŷ = (ŷ - y)/n  →  contains y")
for i in range(n):
    print(f"    Sample {i}: y={y_cls[i]}  ŷ={y_hat_cls[i].round(2)}  loss={ce_per_sample[i]:.3f}  ∂L/∂ŷ={grad_ce[i].round(3)}")


# 2. MSE Regression
print(f"\\n{sep}")
print("  2. MEAN SQUARED ERROR  (regression)")
print(sep)
mse_per = (y_reg - y_hat_reg)**2
mse_loss = mse_per.mean()
grad_mse_sup = 2*(y_hat_reg - y_reg) / n
print(f"  Formula:  L = (1/n) Σᵢ (yᵢ - ŷᵢ)²")
print(f"  Needs:    y (continuous target) — SUPERVISED")
print(f"  Value:    {mse_loss:.4f}")
print(f"  Gradient: ∂L/∂ŷ = 2(ŷ - y)/n  →  contains y")
for i in range(n):
    print(f"    Sample {i}: y={y_reg[i]:.1f}  ŷ={y_hat_reg[i]:.1f}  loss={mse_per[i]:.3f}  ∂L/∂ŷ={grad_mse_sup[i]:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# UNSUPERVISED LOSSES
# ──────────────────────────────────────────────────────────────────────────────

print(f"\\n{'UNSUPERVISED LOSSES':^70}")
print(f"  (None require label y — all computed from x alone)\\n")


# 3. Reconstruction MSE (Autoencoder)
print(sep)
print("  3. RECONSTRUCTION MSE  (Autoencoder)")
print(sep)
recon_per = np.mean((x - x_hat)**2, axis=1)
recon_loss = recon_per.mean()
grad_recon = 2*(x_hat - x) / (n * p)
print(f"  Formula:  L = (1/n) Σᵢ ||xᵢ − x̂ᵢ||²")
print(f"  Needs:    x (the input itself) — NO LABEL — UNSUPERVISED")
print(f"  Value:    {recon_loss:.4f}")
print(f"  Gradient: ∂L/∂x̂ = 2(x̂ − x)/(n·p)  →  contains ONLY x")
for i in range(n):
    print(f"    Sample {i}: recon_loss={recon_per[i]:.4f}  ∂L/∂x̂ mean={grad_recon[i].mean():.4f}")


# 4. VAE ELBO
print(f"\\n{sep}")
print("  4. VAE ELBO = Reconstruction + KL Divergence")
print(sep)
recon_vae = np.mean((x - x_hat)**2, axis=1)
kl_per    = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)
elbo_per  = recon_vae + kl_per
elbo_loss = elbo_per.mean()
grad_mu   = mu / n
grad_lv   = 0.5*(np.exp(log_var) - 1) / n
print(f"  Formula:  L = E[||x−x̂||²] + KL[N(μ,σ²) || N(0,1)]")
print(f"          KL = -½ Σⱼ (1 + log σ²ⱼ - μ²ⱼ - σ²ⱼ)  (closed form)")
print(f"  Needs:    x (reconstruction target) + μ,σ² (encoder output) — NO LABEL — UNSUPERVISED")
print(f"  Total loss: {elbo_loss:.4f}   Recon: {recon_vae.mean():.4f}   KL: {kl_per.mean():.4f}")
print(f"  KL gradient: ∂KL/∂μ = μ/n,  ∂KL/∂log_σ² = ½(σ²-1)/n  →  no y")


# 5. Contrastive NT-Xent
print(f"\\n{sep}")
print("  5. NT-XENT CONTRASTIVE LOSS  (SimCLR)")
print(sep)
tau = 0.2
Z  = np.vstack([z_i, z_j])
sim = Z@Z.T / tau
np.fill_diagonal(sim, -1e9)
pos = np.concatenate([np.arange(n, 2*n), np.arange(n)])
sm  = np.exp(sim - sim.max(1, keepdims=True))
lse = np.log(sm.sum(1) + 1e-8) + sim.max(1)
pos_sim = sim[np.arange(2*n), pos]
nt_xent  = -(pos_sim - lse).mean()
print(f"  Formula:  L = -(1/2N) Σᵢ log[ sim(zᵢ,zⱼ)/τ / Σₖ≠ᵢ sim(zᵢ,zₖ)/τ ]")
print(f"  Needs:    two augmented views of x — NO LABEL — UNSUPERVISED")
print(f"  Value:    {nt_xent:.4f}   (τ={tau})")
print(f"  Gradient: ∂L/∂Z = function of cosine similarities only — no y")
print(f"  Positive pair similarity:  {np.exp(pos_sim[:n]).mean():.3f}  (should → 1)")
print(f"  Negative pair similarity:  {np.exp(sim[:n, :n].mean()):.3f}  (should → 0)")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\\n{'='*70}")
print(f"  SUMMARY: What Each Loss Contains")
print(f"{'='*70}")
print(f"\\n  {'Loss':<30} {'Uses y?':<10} {'Gradient contains'}")
print(f"  {'─'*65}")
rows = [
    ("CrossEntropy (classif.)",     "YES",  "(ŷ − y)  ← y present"),
    ("MSE / MAE (regression)",      "YES",  "2(ŷ − y) ← y present"),
    ("Hinge (SVM-style)",           "YES",  "−y if margin < 1 ← y present"),
    ("Reconstruction MSE (AE)",     "NO",   "2(x̂ − x) ← only x"),
    ("ELBO (VAE)",                  "NO",   "2(x̂−x) + KL terms ← only x"),
    ("Adversarial (GAN)",           "NO",   "∂log D(G(z))/∂θ_G ← only x"),
    ("NT-Xent (Contrastive)",       "NO",   "cosine sim grad ← only x"),
    ("Masked Pred. (Self-sup.)",    "NO",   "CE(predicted, x_masked) ← only x"),
]
for name, uses_y, grad in rows:
    marker = "◄ SUPERVISED" if uses_y == "YES" else "◄ UNSUPERVISED"
    print(f"  {name:<30} {uses_y:<10} {grad:<35} {marker}")

print(f"""
  THE RULE: if the loss formula contains y  →  SUPERVISED
            if the loss formula contains only x (and model outputs) →  UNSUPERVISED
""")
''',
    },

    "Gradient Signal Comparison": {
        "description": "Trace the gradient signal layer by layer — see exactly how label info vs data info flows",
        "runnable": True,
        "code": '''
"""
================================================================================
GRADIENT SIGNAL COMPARISON
================================================================================

We train the SAME single hidden layer network in both modes simultaneously
and track what information the gradient carries at each layer.

The goal: show concretely that in supervised mode the gradient "knows about"
the class label, while in unsupervised mode it knows only about the input.
================================================================================
"""

import numpy as np
np.random.seed(0)

def relu(z):   return np.maximum(0, z)
def relu_g(z): return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── Dataset: 2 well-separated clusters in 4D ──────────────────────────────────

n, dim = 200, 4
X = np.vstack([np.random.randn(n//2, dim)*0.3 + [2,2,0,0],
               np.random.randn(n//2, dim)*0.3 + [-2,-2,0,0]])
y = np.array([0]*(n//2) + [1]*(n//2))
X = (X - X.mean(0)) / X.std(0)

hidden = 16

# ── Supervised network ────────────────────────────────────────────────────────
SW1 = np.random.randn(dim, hidden) * np.sqrt(2/dim);  Sb1 = np.zeros((1,hidden))
SW2 = np.random.randn(hidden, 2)  * np.sqrt(2/hidden); Sb2 = np.zeros((1,2))

# ── Unsupervised AE network ───────────────────────────────────────────────────
AW1 = np.random.randn(dim, hidden) * np.sqrt(2/dim);  Ab1 = np.zeros((1,hidden))
AW2 = np.random.randn(hidden, dim) * np.sqrt(2/dim);  Ab2 = np.zeros((1,dim))

lr = 0.01
Y_oh = np.zeros((n,2)); Y_oh[np.arange(n), y] = 1

print("=" * 70)
print("  GRADIENT SIGNAL ANALYSIS — Layer by Layer")
print("=" * 70)
print(f"\\n  Network: {dim}D input → {hidden} hidden → output")
print(f"  Supervised:   output = 2-class softmax, loss = CrossEntropy(ŷ, y)")
print(f"  Unsupervised: output = {dim}D reconstruction, loss = MSE(x̂, x)\\n")

# Train for a few steps and capture gradient norms and content
print(f"  {'Step':<6} {'S: ∂L/∂W1 norm':<20} {'U: ∂L/∂W1 norm':<20} {'Cosine sim (S,U)'}")
print(f"  {'─'*65}")

for step in range(5):
    idx = np.random.permutation(n)[:32]
    Xb = X[idx];  Yb = Y_oh[idx];  yb = y[idx]

    # ── Supervised forward + backward ─────────────────────────────────────────
    sh1 = relu(Xb@SW1+Sb1)
    syh = softmax(sh1@SW2+Sb2)
    sdz2 = (syh - Yb)/32                    # ← contains y
    sdW2 = sh1.T@sdz2
    sdh1 = sdz2@SW2.T * relu_g(Xb@SW1+Sb1)
    sdW1 = Xb.T@sdh1                        # gradient of W1 in supervised mode

    # ── Unsupervised forward + backward ──────────────────────────────────────
    ah1  = relu(Xb@AW1+Ab1)
    axh  = sigmoid(ah1@AW2+Ab2)
    adxh = 2*(axh - Xb)/(32*dim)           # ← no y, only x
    adW2 = ah1.T@adxh
    adh1 = adxh@AW2.T * relu_g(Xb@AW1+Ab1)
    adW1 = Xb.T@adh1                        # gradient of W1 in unsupervised mode

    s_norm = np.linalg.norm(sdW1)
    u_norm = np.linalg.norm(adW1)
    cos_sim = (sdW1.flatten()@adW1.flatten()) / (s_norm*u_norm + 1e-8)
    print(f"  {step+1:<6} {s_norm:<20.4f} {u_norm:<20.4f} {cos_sim:.4f}")

    SW1 -= lr*sdW1; SW2 -= lr*sdW2
    AW1 -= lr*adW1; AW2 -= lr*adW2

print(f"""
  Interpretation:
    Both gradients have similar norms — both are learning.
    The cosine similarity shows how DIFFERENT the update directions are:
      Near 0 → the two paradigms are pushing weights in orthogonal directions
      This means they are learning DIFFERENT things from IDENTICAL architecture
      Supervised W1: aligns with class-discriminative directions
      Unsupervised W1: aligns with reconstruction-relevant directions
""")

# ── After training: compare what each paradigm learned ────────────────────────
print("=" * 70)
print("  WHAT DID EACH PARADIGM LEARN FROM THE SAME DATA?")
print("=" * 70)

# Supervised accuracy
sh1_f  = relu(X@SW1+Sb1)
syh_f  = softmax(sh1_f@SW2+Sb2)
acc    = (syh_f.argmax(1) == y).mean()

# Unsupervised: linear probe with 6 labels
from numpy.linalg import lstsq
ah1_f = relu(X@AW1+Ab1)
few   = [0,1,2, n//2, n//2+1, n//2+2]
Xf, yf = ah1_f[few], y[few]
Yf_oh  = np.zeros((6,2)); Yf_oh[np.arange(6),yf] = 1
W_lin, _, _, _ = lstsq(np.c_[Xf, np.ones(6)], Yf_oh, rcond=None)
probe_pred = (np.c_[ah1_f, np.ones(n)]@W_lin).argmax(1)
probe_acc  = (probe_pred == y).mean()

# Reconstruction error
axh_f = sigmoid(ah1_f@AW2+Ab2)
recon_err = np.mean((axh_f - X)**2)

print(f"""
  SUPERVISED NETWORK:
    Trained with: {n} (x,y) pairs (all labels used)
    Task accuracy: {acc:.1%}  — directly optimised for this
    What it learned: the single decision boundary separating class 0 from 1
    Gradient was: ∂L/∂W ∝ (ŷ − y)  — knew which class was right

  UNSUPERVISED NETWORK (Autoencoder):
    Trained with: {n} x samples, ZERO labels
    Reconstruction MSE: {recon_err:.4f}  — directly optimised for this
    Linear probe accuracy (6 labels): {probe_acc:.1%}  — not optimised for, but achieved
    What it learned: the data manifold (where points live in space)
    Gradient was: ∂L/∂W ∝ 2(x̂ − x)  — never saw a class label

  KEY CONCLUSION:
    Supervised: trains faster on the specific task, needs all labels
    Unsupervised: never sees labels, yet representations are linearly separable
    Same backprop. Same architecture. Entirely different teachers.
""")
''',
    },

    "The Classifier Hidden Inside an Autoencoder": {
        "description": "Train an autoencoder without labels, then show its latent space separates classes",
        "runnable": True,
        "code": '''
"""
================================================================================
THE CLASSIFIER HIDDEN INSIDE AN AUTOENCODER
================================================================================

This demo makes concrete a key theoretical claim:

    "Unsupervised networks can discover class structure without labels."

We train an autoencoder on 4-class data with ZERO labels.
Then we use only 4 labelled examples (one per class) to build a classifier.
We compare this against a supervised network trained on all labels.

The autoencoder never "knew" there were 4 classes. Yet the latent space
will organise itself so that classes cluster. This is because class membership
IS structure in the data — and unsupervised methods capture data structure.
================================================================================
"""

import numpy as np
from numpy.linalg import lstsq
np.random.seed(7)

def relu(z):   return np.maximum(0, z)
def relu_g(z): return (z > 0).astype(float)
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
def softmax(z):
    e = np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

# ── Dataset: 4 clusters in 20D ────────────────────────────────────────────────
n_per, dim = 150, 20
centres = [np.array([4,4]+[0]*(dim-2)),  np.array([-4,4]+[0]*(dim-2)),
           np.array([-4,-4]+[0]*(dim-2)), np.array([4,-4]+[0]*(dim-2))]
X_parts = [np.random.randn(n_per, dim)*0.6 + c for c in centres]
X = np.vstack(X_parts)
y = np.concatenate([[i]*n_per for i in range(4)])
X = (X - X.mean(0)) / X.std(0)

# ── UNSUPERVISED: Train autoencoder (latent dim=4) ─────────────────────────────
print("=" * 65)
print("  TRAINING AUTOENCODER — ZERO LABELS USED")
print("=" * 65)

h = 64; ld = 4
W1 = np.random.randn(dim,h)*np.sqrt(2/dim);  b1 = np.zeros((1,h))
W2 = np.random.randn(h,ld)*np.sqrt(2/h);     b2 = np.zeros((1,ld))
W3 = np.random.randn(ld,h)*np.sqrt(2/ld);    b3 = np.zeros((1,h))
W4 = np.random.randn(h,dim)*np.sqrt(2/h);    b4 = np.zeros((1,dim))
lr = 0.008

for ep in range(120):
    idx = np.random.permutation(len(X))
    for s in range(0, len(X), 64):
        Xb = X[idx[s:s+64]]
        # Forward
        h1 = relu(Xb@W1+b1); z = relu(h1@W2+b2)
        d1 = relu(z@W3+b3);  xh = d1@W4+b4
        # Backward — ZERO labels used
        n_b = len(Xb)
        g = 2*(xh-Xb)/(n_b*dim)
        dW4=d1.T@g;        db4=g.sum(0,keepdims=True)
        dd1=g@W4.T*relu_g(z@W3+b3)
        dW3=z.T@dd1;       db3=dd1.sum(0,keepdims=True)
        dz=dd1@W3.T*relu_g(h1@W2+b2)
        dW2=h1.T@dz;       db2=dz.sum(0,keepdims=True)
        dh1=dz@W2.T*relu_g(Xb@W1+b1)
        dW1=Xb.T@dh1;      db1=dh1.sum(0,keepdims=True)
        for w,g_ in [(W1,dW1),(b1,db1),(W2,dW2),(b2,db2),
                     (W3,dW3),(b3,db3),(W4,dW4),(b4,db4)]:
            w -= lr*g_

h1_f = relu(X@W1+b1);  z_all = relu(h1_f@W2+b2)   # latent codes, shape (n, 4)

# ── SUPERVISED: Train classifier (all labels) ─────────────────────────────────
print("\\n  Training supervised classifier — ALL labels used")

SW1=np.random.randn(dim,64)*np.sqrt(2/dim); Sb1=np.zeros((1,64))
SW2=np.random.randn(64,4)*np.sqrt(2/64);   Sb2=np.zeros((1,4))
Y_oh=np.zeros((len(y),4)); Y_oh[np.arange(len(y)),y]=1
lr_s = 0.01

for ep in range(120):
    idx = np.random.permutation(len(X))
    for s in range(0, len(X), 64):
        Xb=X[idx[s:s+64]]; Yb=Y_oh[idx[s:s+64]]
        sh1=relu(Xb@SW1+Sb1); syh=softmax(sh1@SW2+Sb2)
        n_b=len(Xb)
        dz=(syh-Yb)/n_b
        dSW2=sh1.T@dz; dSb2=dz.sum(0,keepdims=True)
        dh=dz@SW2.T*relu_g(Xb@SW1+Sb1)
        dSW1=Xb.T@dh; dSb1=dh.sum(0,keepdims=True)
        SW1-=lr_s*dSW1; Sb1-=lr_s*dSb1; SW2-=lr_s*dSW2; Sb2-=lr_s*dSb2

sh1_f=relu(X@SW1+Sb1); acc_full=(softmax(sh1_f@SW2+Sb2).argmax(1)==y).mean()

# ── LINEAR PROBE on AE latent codes (only 1 label per class = 4 labels) ──────
few_idx = [np.where(y==c)[0][0] for c in range(4)]   # 1 example per class
Xf, yf  = z_all[few_idx], y[few_idx]
Yf_oh   = np.zeros((4,4)); Yf_oh[np.arange(4),yf] = 1
W_lin, _,_,_ = lstsq(np.c_[Xf,np.ones(4)], Yf_oh, rcond=None)
probe_preds   = (np.c_[z_all, np.ones(len(z_all))]@W_lin).argmax(1)
probe_acc     = (probe_preds == y).mean()

# ── Measure cluster quality in latent space ────────────────────────────────────
cluster_sep = []
for c in range(4):
    m_c  = z_all[y==c].mean(0)
    m_all= z_all.mean(0)
    cluster_sep.append(np.linalg.norm(m_c - m_all))
avg_sep = np.mean(cluster_sep)
intra   = np.mean([np.mean(np.linalg.norm(z_all[y==c]-z_all[y==c].mean(0),axis=1))
                   for c in range(4)])

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\\n{'='*65}")
print(f"  RESULTS")
print(f"{'='*65}")
print(f"""
  AUTOENCODER (unsupervised):
    Labels used in training:       0  (zero)
    Labels used in probe:          4  (1 per class)
    Linear probe accuracy:         {probe_acc:.1%}
    Latent cluster separation:     {avg_sep:.3f}
    Latent intra-cluster spread:   {intra:.3f}
    Separation / spread ratio:     {avg_sep/intra:.2f}x

  SUPERVISED CLASSIFIER:
    Labels used in training:       {len(y)}  (all of them)
    Accuracy (full supervision):   {acc_full:.1%}

  COMPARISON:
    Unsupervised AE with 4 labels vs Supervised with {len(y)} labels:
    {probe_acc:.1%} vs {acc_full:.1%}

  EXPLANATION:
    The autoencoder's loss (MSE reconstruction) forced the network to
    preserve the cluster structure, because clusters ARE the dominant
    structure in the data.

    If your data has structure, unsupervised methods find it —
    even when the network never knew what to look for.

    The difference between supervised and unsupervised is not:
        "one learns and one doesn't"
    It is:
        "one learns what a human pointed at (y)"
        "one learns what the data contains (x structure)"
""")
''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  SUPERVISED vs UNSUPERVISED NEURAL NETWORKS")
    print("  What makes the difference — a direct comparison")
    print("=" * 70)
    print("""
  The ONE-SENTENCE rule:
    A neural network is SUPERVISED  if y (a human label) appears in its loss.
    A neural network is UNSUPERVISED if its loss contains only x and model outputs.

  Everything else — weights, activations, backprop, Adam — is IDENTICAL.
    """)

    np.random.seed(42)

    # ── Shared dataset ────────────────────────────────────────────────────────
    dim, n_per = 10, 200
    c0 = np.random.randn(n_per, dim) * 0.5 + np.array([3,3]+[0]*(dim-2))
    c1 = np.random.randn(n_per, dim) * 0.5 + np.array([-3,3]+[0]*(dim-2))
    c2 = np.random.randn(n_per, dim) * 0.5 + np.array([0,-4]+[0]*(dim-2))
    X  = np.vstack([c0,c1,c2])
    y  = np.array([0]*n_per + [1]*n_per + [2]*n_per)
    X  = (X - X.mean(0)) / X.std(0)

    def relu(z):    return np.maximum(0, z)
    def relu_g(z):  return (z > 0).astype(float)
    def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
    def softmax(z):
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

    # ── Train supervised ──────────────────────────────────────────────────────
    print("─"*70)
    print("  Training Supervised Network (CrossEntropy + labels)...")

    SW1=np.random.randn(dim,32)*np.sqrt(2/dim); Sb1=np.zeros((1,32))
    SW2=np.random.randn(32,3) *np.sqrt(2/32);  Sb2=np.zeros((1,3))
    Y_oh=np.zeros((len(y),3)); Y_oh[np.arange(len(y)),y]=1
    for _ in range(100):
        idx=np.random.permutation(len(X))
        for s in range(0,len(X),64):
            Xb=X[idx[s:s+64]]; Yb=Y_oh[idx[s:s+64]]
            h=relu(Xb@SW1+Sb1); yh=softmax(h@SW2+Sb2)
            n_b=len(Xb); dz=(yh-Yb)/n_b
            SW2-=0.01*h.T@dz; Sb2-=0.01*dz.sum(0,keepdims=True)
            dh=dz@SW2.T*relu_g(Xb@SW1+Sb1)
            SW1-=0.01*Xb.T@dh; Sb1-=0.01*dh.sum(0,keepdims=True)

    h_sup=relu(X@SW1+Sb1)
    acc_sup=(softmax(h_sup@SW2+Sb2).argmax(1)==y).mean()
    print(f"  Supervised accuracy: {acc_sup:.1%}  (used all {len(y)} labels)")

    # ── Train unsupervised ────────────────────────────────────────────────────
    print("\n  Training Unsupervised Autoencoder (MSE, zero labels)...")

    AW1=np.random.randn(dim,32)*np.sqrt(2/dim); Ab1=np.zeros((1,32))
    AW2=np.random.randn(32,3) *np.sqrt(2/32);  Ab2=np.zeros((1,3))   # bottleneck
    AW3=np.random.randn(3,32) *np.sqrt(2/3);   Ab3=np.zeros((1,32))
    AW4=np.random.randn(32,dim)*np.sqrt(2/32);  Ab4=np.zeros((1,dim))
    for _ in range(100):
        idx=np.random.permutation(len(X))
        for s in range(0,len(X),64):
            Xb=X[idx[s:s+64]]
            h1=relu(Xb@AW1+Ab1); z=relu(h1@AW2+Ab2)
            d1=relu(z@AW3+Ab3); xh=d1@AW4+Ab4
            n_b=len(Xb); g=2*(xh-Xb)/(n_b*dim)
            AW4-=0.008*d1.T@g; Ab4-=0.008*g.sum(0,keepdims=True)
            dd=g@AW4.T*relu_g(z@AW3+Ab3)
            AW3-=0.008*z.T@dd; Ab3-=0.008*dd.sum(0,keepdims=True)
            dz=dd@AW3.T*relu_g(h1@AW2+Ab2)
            AW2-=0.008*h1.T@dz; Ab2-=0.008*dz.sum(0,keepdims=True)
            dh=dz@AW2.T*relu_g(Xb@AW1+Ab1)
            AW1-=0.008*Xb.T@dh; Ab1-=0.008*dh.sum(0,keepdims=True)

    h1_u=relu(X@AW1+Ab1); z_u=relu(h1_u@AW2+Ab2)

    from numpy.linalg import lstsq
    few=[j for c in range(3) for j in np.where(y==c)[0][:5]]
    Xf,yf=z_u[few],y[few]
    Yf_oh=np.zeros((len(few),3)); Yf_oh[np.arange(len(few)),yf]=1
    Wl,_,_,_=lstsq(np.c_[Xf,np.ones(len(few))],Yf_oh,rcond=None)
    probe_acc=((np.c_[z_u,np.ones(len(z_u))]@Wl).argmax(1)==y).mean()
    print(f"  Unsupervised probe: {probe_acc:.1%}  (used only 15 labels after training)")

    # ── Print the canonical comparison ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"""
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │ DIMENSION            │ SUPERVISED               │ UNSUPERVISED                       │
  ├──────────────────────────────────────────────────────────────────────────────────────┤
  │ Labels in training   │ All {len(y)} samples     │ None (0 labels)                    │
  │ Loss formula         │ CE(ŷ, y)                 │ MSE(x̂, x)                          │
  │ Loss needs y?        │ YES                      │ NO                                 │
  │ Gradient = f(...)    │ f(ŷ, y)                  │ f(x̂, x)                            │
  │ Accuracy             │ {acc_sup:.1%}            │ {probe_acc:.1%} (15 labels probe)  │
  │ Weights/activations  │ IDENTICAL                │ IDENTICAL                          │
  │ Backpropagation      │ IDENTICAL                │ IDENTICAL                          │
  │ Optimiser (SGD)      │ IDENTICAL                │ IDENTICAL                          │
  └──────────────────────────────────────────────────────────────────────────────────────┘

  THE COMPLETE RULE (no exceptions):
    1. Find the loss function.
    2. Check if y (a human-provided label) appears in it.
    3. YES → supervised.  NO → unsupervised.
    That's it. Architecture, depth, width, activations — irrelevant to the
    classification. Only the loss function determines the paradigm.
    """)


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
        main_script = None

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
    visual_height = 400
    # try:
    #     from Supervised_Learning.visuals.regression_visual import (   # ← match your exact folder casing
    #         REG_VISUAL_HTML,
    #         REG_VISUAL_HEIGHT,
    #     )
    #     visual_html   = REG_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = REG_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[01_linear_regression.py] Could not load visual: {e}", stacklevel=2)

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