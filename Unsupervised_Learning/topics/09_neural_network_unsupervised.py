"""
Neural Networks — Unsupervised & Self-Supervised Learning
=========================================================

Unsupervised neural networks learn from data without any labels.
Instead of minimising the error between a prediction and a ground-truth
label, they minimise objectives derived entirely from the structure of the
data itself: reconstruction fidelity, latent-space geometry, probability
density, or similarity between augmented views.

This module covers the five major families that were deliberately excluded
from the supervised module:

    1. Autoencoders          — compress and reconstruct; learn representations
    2. Variational AEs (VAE) — probabilistic latent spaces; generative models
    3. GANs                  — adversarial generation; learning data distributions
    4. Self-Supervised       — generate labels from data structure (masking, prediction)
    5. Contrastive Learning  — learn by comparing similar vs dissimilar pairs

All five share one property: NO human-annotated labels required.
The loss signal comes from the data itself.

They are not competitors to supervised learning — they are its foundation.
The representations learned without labels are routinely fine-tuned with
a handful of labels to achieve state-of-the-art supervised performance.

"""

import re
import textwrap

DISPLAY_NAME = "09 · Neural Networks (Unsupervised)"
ICON         = "🔓"
SUBTITLE     = "Learning without labels — autoencoders, VAEs, GANs, self-supervised, contrastive"
TOPIC_NAME   = "Neural Networks (Unsupervised)"
VISUAL_HTML  = ""

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Why Unsupervised Neural Networks?

Supervised learning requires labels. Labels are expensive: a radiologist
must annotate each scan; a linguist must parse each sentence; a lawyer must
classify each contract clause. But the world generates unlabelled data in
virtually unlimited quantities.

Unsupervised neural networks answer the question:
    "What can a network learn if we never tell it the answer?"

The answer turns out to be: almost everything that matters.
    - The geometry of the data manifold
    - Which features co-occur and which are independent
    - How to compress information without losing it
    - How to generate plausible new examples
    - Which examples are similar and which are different

These learned representations power modern AI:
    BERT / GPT:          self-supervised on text → fine-tuned on any NLP task
    CLIP / DALL-E:       contrastive + generative → vision-language alignment
    Stable Diffusion:    VAE + diffusion → photorealistic image generation
    AlphaFold:           self-supervised on sequences → protein structure


### The Fundamental Difference: What Provides the Gradient?

    SUPERVISED:
        Loss = measure of error between prediction ŷ and label y
        ∂L/∂ŷ depends on y  →  a human provided the signal

    UNSUPERVISED:
        Loss = measure derived from the DATA ITSELF (no y needed)
        Examples:
            Reconstruction:   how well can we reconstruct x from its code?
            Density:          how probable is x under the learned distribution?
            Contrastive:      are these two representations similar or different?
            Predictive:       can we predict part of x from another part?

    The network is still trained by gradient descent via backpropagation.
    Only the source of the loss signal changes.


---


### Part 1: Autoencoders — Compress and Reconstruct


### Architecture: Encoder → Bottleneck → Decoder

An autoencoder learns an identity function under a bottleneck constraint:
    given input x, reproduce x at the output — but only via a compressed code z.

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │    INPUT x (dim p)                                               │
    │       ↓                                                          │
    │    ENCODER  E(x)  →  z  (dim d << p)     ← bottleneck            │
    │       ↓                                                          │
    │    DECODER  D(z)  →  x̂  (dim p)          ← reconstruction        │
    │       ↓                                                          │
    │    LOSS:  L = ||x − x̂||²                 ← no labels needed      │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

The encoder compresses x into a low-dimensional code z (the latent vector).
The decoder must reconstruct x from only z.

Because the bottleneck forces information through a narrow channel, the
encoder must learn to retain only the most important features.
Unimportant variation (noise, redundancy) is discarded.


### The Reconstruction Loss:

For continuous inputs (images, sensor data):
    MSE:   L = (1/p) Σⱼ (xⱼ − x̂ⱼ)²          ← squared pixel error

For binary inputs (black-and-white, binary features):
    BCE:   L = −(1/p) Σⱼ [xⱼ log(x̂ⱼ) + (1−xⱼ) log(1−x̂ⱼ)]

Critically: both losses compare x to x̂ — no label y appears anywhere.
The target IS the input. The network learns by trying to reproduce itself.


### What the Encoder Learns — The Latent Space:

After training, the encoder z = E(x) maps each input to a point in latent space.
The geometry of this latent space reflects the structure of the data:
    - Similar inputs map to nearby points
    - Different inputs map to distant points
    - Smooth directions in latent space correspond to meaningful variation

    Example: autoencoder on face images
        z-direction 1:  controls age
        z-direction 2:  controls pose (left/right)
        z-direction 3:  controls lighting

    No one told the network about age, pose, or lighting.
    The encoder discovered these as the most informative dimensions.


    # ======================================================================================= # 
    **Diagram 1 — Autoencoder Architecture:**

    Input x (p=784, MNIST digit):    28×28 pixel image flattened

    ENCODER (compresses):
    ─────────────────────────────────────────────────────────────
    Layer 1:   z¹ = ReLU( W¹ x  + b¹ )     shape: (784 → 256)
    Layer 2:   z² = ReLU( W² z¹ + b² )     shape: (256 → 64)
    Bottleneck:  z = W³ z² + b³             shape: (64  → 2)   ← latent code

    DECODER (reconstructs):
    ─────────────────────────────────────────────────────────────
    Layer 1:   h¹ = ReLU( W⁴ z  + b⁴ )     shape: (2   → 64)
    Layer 2:   h² = ReLU( W⁵ h¹ + b⁵ )     shape: (64  → 256)
    Output:    x̂  = σ( W⁶ h² + b⁶ )        shape: (256 → 784)  ← reconstruction

    Loss:  L = BCE(x, x̂) — averaged over all 784 pixels
    ─────────────────────────────────────────────────────────────
    Training:  minimise L over all training images
    No labels: the target x̂ is compared to the INPUT x, not a label
    # ======================================================================================= # 


### Autoencoder Variants:

    Standard AE:         deterministic encoder, MSE/BCE reconstruction loss
    Denoising AE:        corrupt input (add noise, mask pixels), reconstruct clean x
                         Forces encoder to learn robust features, not just copy input
    Sparse AE:           add sparsity penalty L += λ ||z||₁ → most hidden units off
                         Forces distributed, parts-based representations
    Contractive AE:      add Frobenius norm of Jacobian to loss
                         Penalises sensitivity to input perturbations


### What Autoencoders Are Used For:

    Dimensionality reduction:   z replaces PCA — nonlinear, captures manifold structure
    Anomaly detection:          train on normal data; anomalies have high reconstruction error
    Denoising:                  denoising AE learns to produce clean outputs from corrupted inputs
    Representation learning:    pretrain encoder, fine-tune on downstream tasks with few labels
    Data compression:           learned codes more compact than JPEG/MP3 for specific domains


---


### Part 2: Variational Autoencoders (VAEs)


### The Problem with Standard Autoencoders:

A standard AE maps each input to a single point z in latent space.
If we sample a random point z in that latent space and decode it, we often
get garbage — the latent space has "holes" between clusters.

    Why?  The encoder has no incentive to organise the latent space smoothly.
          It can pack representations anywhere, leaving large unstructured gaps.

The VAE fixes this by making z a probability distribution, not a point.


### VAE: Encoder Outputs a Distribution

Instead of z = E(x), the VAE encoder outputs:
    μ(x)  = mean of a Gaussian     (where the code "should" be)
    σ²(x) = variance of a Gaussian  (how uncertain we are)

Then z is sampled:
    z ~ N( μ(x), σ²(x) )   — z is a random variable, not a fixed vector

This means the decoder must be able to reconstruct x from a range of nearby
z values — forcing it to learn a smooth, continuous latent space.


### The VAE Loss — ELBO (Evidence Lower Bound):

    L_VAE = L_reconstruction + L_regularisation

    L_reconstruction = E[log p(x|z)]
        → How well does the decoder reconstruct x given the sampled z?
        → Same as autoencoder: MSE or BCE between x and x̂

    L_regularisation = KL[ q(z|x) || p(z) ]
        → KL divergence between the encoder distribution q(z|x) = N(μ,σ²)
          and a standard normal prior p(z) = N(0,I)
        → Closed form: KL = -½ Σⱼ (1 + log σ²ⱼ - μ²ⱼ - σ²ⱼ)
        → Pushes every encoder distribution toward N(0,I)
        → Forces the latent space to be smooth and well-organised

    TOTAL:  L = reconstruction + β · KL   (β=1 is standard VAE; β>1 is β-VAE)


### The Reparametrisation Trick:

Sampling z ~ N(μ, σ²) is not differentiable — we cannot backpropagate through it.

The reparametrisation trick rewrites the sample as:
    z = μ + σ · ε      where ε ~ N(0, I)

Now the randomness is in ε (a fixed sample from N(0,I)),
and the parameters μ and σ are computed by the encoder.
Gradients flow through μ and σ normally — ε is just a noise input.

    BEFORE reparametrisation:   z ← sample(N(μ, σ²))   ← NOT differentiable
    AFTER reparametrisation:    z = μ + σ · ε           ← differentiable in μ, σ


    # ======================================================================================= # 
    **Diagram 2 — VAE Architecture vs Standard AE:**

    STANDARD AUTOENCODER:
    ─────────────────────────────────────────────────────────────
    x → Encoder → z (single point) → Decoder → x̂
    Loss: ||x - x̂||²
    Latent space: unstructured, can have holes

    VARIATIONAL AUTOENCODER:
    ─────────────────────────────────────────────────────────────
    x → Encoder → μ(x), σ²(x)
                       ↓
                   z = μ + σ·ε   (ε ~ N(0,I), reparametrisation trick)
                       ↓
                   Decoder → x̂
    Loss: ||x - x̂||² + β · KL[N(μ,σ²) || N(0,I)]
    Latent space: continuous, smooth, organised
                  → CAN generate new samples by sampling z ~ N(0,I)

    Generation (no input needed):
    ─────────────────────────────────────────────────────────────
    Sample z ~ N(0, I) → Decoder(z) → new sample x̂
    # ======================================================================================= # 


### Why VAEs Are Generative:

Because the KL term forces the latent space to fill N(0,I), we can generate
new samples by sampling a random z ~ N(0,1) and passing it through the decoder.
The decoder has learned to map the entire standard normal region to valid outputs.


### VAE Applications:

    Image generation:     sample z ~ N(0,I) → decode → new face, digit, molecule
    Interpolation:        interpolate linearly between two codes z₁ and z₂ → smooth morph
    Anomaly detection:    high KL or reconstruction loss → novel/anomalous input
    Drug discovery:       encode molecules, sample latent space, decode → novel compounds
    Disentanglement:      β-VAE (large β) encourages each zⱼ to control one factor of variation


---


### Part 3: Generative Adversarial Networks (GANs)


### The Core Idea — An Adversarial Game:

A GAN has two neural networks playing a minimax game:

    GENERATOR  G(z):    takes random noise z ~ N(0,I) → produces fake samples x̂
    DISCRIMINATOR D(x): takes a sample → outputs probability it is real (not fake)

    Generator's goal:    fool the discriminator (make fakes look real)
    Discriminator's goal: correctly distinguish real from fake

They are trained simultaneously — a generator that is too easy to detect
pushes the discriminator to improve; a strong discriminator pushes the
generator to produce more convincing fakes.


### The GAN Objective:

    min_G  max_D  V(D, G) = E[log D(x)] + E[log(1 − D(G(z)))]

    Discriminator maximises V:
        → log D(x) should be large for real x (predicts "real" correctly)
        → log(1 − D(G(z))) should be large when G(z) is easy to detect (fake)

    Generator minimises V:
        → Wants D(G(z)) to be large (discriminator says "real" for fakes)
        → In practice: generator maximises log D(G(z)) (non-saturating loss)

    No labels appear in this objective.
    The "label" for each sample is just real (from dataset) or fake (from G).


### Training Dynamics:

At Nash Equilibrium (optimal solution):
    - Generator produces samples indistinguishable from real data
    - Discriminator outputs 0.5 for every input (cannot do better than random)
    - The generator has implicitly learned the true data distribution p(x)

In practice:
    - Train D for k steps, then G for 1 step (or alternate)
    - G and D losses should both decrease gradually
    - Instability is common: one side can "win" too early


    # =======================================================================================# 
    **Diagram 3 — GAN Training Loop:**

    REAL DATA:    x ~ p_data(x)     ───────────────────────────────┐
                                                                   ↓
                                                                  D(·)  →  real/fake?
                                                                   ↑
    GENERATOR:    z ~ N(0,I)  →  G(z)  =  x̂ (fake) ────────────────┘

    Step 1 — Train Discriminator (G frozen):
    ─────────────────────────────────────────────────────────────
        Sample batch of real x from dataset
        Sample batch of noise z → generate fake x̂ = G(z)
        L_D = −[log D(x) + log(1 − D(G(z)))]   ← maximise D's accuracy
        Update D weights via ∂L_D/∂θ_D

    Step 2 — Train Generator (D frozen):
    ─────────────────────────────────────────────────────────────
        Sample new noise z → generate fake x̂ = G(z)
        L_G = −log D(G(z))   ← generator wants D to output "real"
        Gradients flow: ∂L_G/∂θ_G  through D (D weights not updated)
        Update G weights via ∂L_G/∂θ_G

    Repeat until Nash equilibrium (D outputs 0.5 everywhere)
    # ======================================================================================= # 


### GAN Failure Modes:

    Mode collapse:      Generator produces only a few types of samples
                        (found one mode that fools D, ignores rest of distribution)
                        Fix: Wasserstein GAN (WGAN), minibatch discrimination

    Training instability: G and D loss oscillate wildly or diverge
                        Fix: spectral normalisation, gradient penalty

    Vanishing gradient: D becomes too strong → D(G(z)) ≈ 0 → log gradient ≈ 0
                        G receives no signal → stalls
                        Fix: non-saturating loss (maximise log D(G(z)) for G)


### Important GAN Variants:

    DCGAN:     Deep Convolutional GAN — convolutional G and D for images (2015)
    WGAN:      Wasserstein distance replaces JS divergence — more stable training
    StyleGAN:  Style-based generator — controls style at each layer → photorealistic faces
    CycleGAN:  Unpaired image-to-image translation — horse → zebra, summer → winter
    Conditional GAN (cGAN): condition G and D on a label y → controlled generation


---


### Part 4: Self-Supervised Learning


### The Core Idea — Create Labels from the Data Itself:

Self-supervised learning constructs a supervised signal automatically from
the structure of the data, without any human annotation:
    - Mask a part of x; predict the masked part from the rest
    - Permute patches of x; predict the correct order
    - Predict the next token in a sequence
    - Predict the rotation applied to an image

These are called "pretext tasks" — they are not the final task, but solving
them forces the network to learn general representations.


### Language: Masked Language Modelling (BERT, 2018)

    Input:     "The cat sat on the [MASK]"
    Target:    "mat"

    1. Take a sentence, randomly mask 15% of tokens
    2. Train a Transformer to predict masked tokens from context
    3. No labels needed — the answer is always the original token

    The model must learn syntax, semantics, world knowledge to predict well.
    After pretraining: add a small classifier head → fine-tune on any NLP task.


### Language: Causal Language Modelling (GPT series)

    Input:     "The cat sat on the"
    Target:    "mat"    ← next token prediction

    At each position, predict the next token given all previous tokens.
    The task IS autoregressive — the dataset itself provides all targets.

    All of GPT-2, GPT-3, GPT-4, Claude, Llama are trained this way.
    Billions of tokens, no human labels — just: predict the next word.


### Vision: Masked Autoencoders (MAE, He et al. 2021)

    1. Divide an image into 16×16 patches (e.g., 196 patches for 224×224 image)
    2. Randomly mask 75% of patches
    3. Encode only the visible patches with a Vision Transformer
    4. Decode: predict the pixel values of masked patches

    Loss: MSE between predicted and actual pixel values of masked patches

    Masking 75% is key: lower masking → too easy (interpolate neighbours);
                        75% → model must understand global structure.


    # ======================================================================================= # 
    **Diagram 4 — Self-Supervised Pretext Tasks:**

    MASKED TOKEN PREDICTION (BERT):
    ─────────────────────────────────────────────────────────────
    Original:   "The cat sat on the mat"
    Masked:     "The cat sat on the [M]"  → model predicts "mat"

    No human annotated this pair. The text itself provides the label.

    NEXT TOKEN PREDICTION (GPT):
    ─────────────────────────────────────────────────────────────
    For each position i in sequence:
        Input: x₁, x₂, ..., xᵢ   →   Target: xᵢ₊₁

    Training pairs generated automatically from any text corpus.

    MASKED IMAGE MODELLING (MAE):
    ─────────────────────────────────────────────────────────────
    Original image:   [P₁][P₂][P₃][P₄][P₅][P₆][P₇][P₈]
    75% masked:       [P₁][  ][P₃][  ][  ][P₆][  ][  ]
    Target:           reconstruct [P₂][P₄][P₅][P₇][P₈]

    Loss: MSE on pixel values of masked patches. No label needed.
    # ======================================================================================= # 


### Why Self-Supervised Representations Transfer:

To predict a masked word, the model must implicitly represent:
    - Grammatical structure (subject, verb, object)
    - Semantic relationships (cat → mat: rhyme? furniture? location?)
    - World knowledge (cats sit on things)

These representations, once learned, transfer to:
    Sentiment analysis, NER, question answering, translation, summarisation...

The pretext task (next word prediction) is not the final goal.
The representations learned while solving it are.


### The Two-Stage Training Paradigm:

    Stage 1:  PRETRAIN — self-supervised on massive unlabelled data
                         learns general representations
    Stage 2:  FINE-TUNE — supervised on small labelled dataset
                          adapts representations to specific task

    GPT-3 (175B params) pretrained on ~300B tokens (no labels)
    → fine-tune with a few hundred labelled examples → beats supervised
      baselines trained on millions of labelled examples

    This is why labels are becoming less central to AI progress.


---


### Part 5: Contrastive Learning


### The Core Idea — Learn by Comparing:

Contrastive learning teaches the network a similarity metric without labels:
    - Similar pairs (augmentations of the same image) → nearby in embedding space
    - Dissimilar pairs (different images) → far apart in embedding space

The network learns to represent what is the same across transformations
(semantic content) and ignore what is different (colour, crop, rotation).


### Data Augmentation — The Source of Pairs:

For each image x, two augmented views are created:
    x₁ = aug₁(x)    ← random crop + colour jitter + horizontal flip
    x₂ = aug₂(x)    ← different random crop + grayscale + blur

(x₁, x₂) are a POSITIVE pair: both came from the same image.
(x₁, xₖ) for k ≠ i are NEGATIVE pairs: from different images.

The network should output similar embeddings for (x₁, x₂)
and different embeddings for (x₁, xₖ).


### SimCLR (Chen et al. 2020) — NT-Xent Loss:

    Architecture:
        Encoder f(·):      backbone (ResNet, ViT) producing embedding h
        Projection head g(·):  small MLP producing z = g(h)
        Contrastive loss on z  (projection head discarded after training!)

    For a batch of N images → 2N augmented views → N positive pairs:

    Similarity:    sim(zᵢ, zⱼ) = zᵢᵀzⱼ / (||zᵢ|| ||zⱼ||)  ← cosine similarity

    NT-Xent loss (for one positive pair i, j):
        L_{i,j} = −log( exp(sim(zᵢ,zⱼ)/τ) / Σ_{k≠i} exp(sim(zᵢ,zₖ)/τ) )
                                                    ← sum over 2N−2 negatives

    τ = temperature (typically 0.1–0.5)
    Low τ → sharp distribution (strong push on hard negatives)
    High τ → soft distribution (treats all negatives equally)


    # =======================================================================================# 
    **Diagram 5 — SimCLR Contrastive Learning:**

    Image x  →  aug₁(x)  →  Encoder f  →  Projector g  →  z₁  ┐
                                                                  │  L_contrastive
    Image x  →  aug₂(x)  →  Encoder f  →  Projector g  →  z₂  ┘  (z₁ and z₂ should be close)

    Different image x'  →  aug(x')  →  f  →  g  →  z'            (z₁ and z' should be far)

    In embedding space:
    ─────────────────────────────────────────────────────────────
    Before training:   all embeddings randomly scattered
    After training:    embeddings cluster by semantic content
                       (dogs near dogs, cars near cars)
                       even though no labels were used

    Key insight: the network learns what is INVARIANT to augmentation.
    What survives cropping + flipping + colour jitter = semantic identity.
    # =======================================================================================# 


### BYOL — No Negative Pairs Needed:

Bootstrap Your Own Latent (Grill et al., 2020) achieves contrastive-quality
representations without negative pairs at all.

    Two networks: ONLINE network (updated by gradient) and TARGET network
                  (exponential moving average of online weights — slowly tracks online)

    Online encodes view 1, predicts the TARGET network's encoding of view 2.
    Loss: cosine similarity between online prediction and target representation.

    No negatives → no large batch size requirement
    Still learns useful representations because:
        - Online network predicts from one view what target sees in another
        - Target is a slow-moving average → prevents collapse (moving target)


### DINO — Self-Distillation:

DINO (Caron et al., 2021) applies BYOL-like self-distillation to Vision Transformers.
The student (online) is trained to match the teacher (EMA) on different crops.

    Key finding: DINO's attention maps spontaneously segment objects
                 without ever being trained on segmentation labels.
    The network learned to find objects purely from comparing crop views.


### Collapse — The Fundamental Danger:

Without careful design, contrastive learning collapses:
    Trivial solution: output z = constant for all inputs → loss = 0
    → Encoder is useless: it learned to ignore all input variation

Prevention strategies:
    SimCLR:    large batch of negatives actively pushes representations apart
    BYOL:      EMA target prevents both networks from collapsing together
    Barlow Twins:  decorrelation loss → minimise cross-correlation between dimensions
    VICReg:    variance, invariance, covariance regularisation


---


### Part 6: Latent Space Learning — The Unifying Theme


### What All Five Methods Share:

Every unsupervised neural network in this module learns a latent space z:
    Autoencoder:         z = E(x)                  (deterministic)
    VAE:                 z ~ N(μ(x), σ²(x))         (probabilistic)
    GAN Generator:       x = G(z), z ~ N(0,I)        (inverted)
    Self-supervised:     z = Encoder(x)              (used for downstream tasks)
    Contrastive:         z = g(f(x))                 (metric embedding)

The latent space captures the essential structure of the data:
    - Low-dimensional (bottleneck)
    - Smooth (nearby z → nearby x, meaningful interpolation)
    - Structured (clusters correspond to semantic categories)
    - Transferable (representations learned on unlabelled data work for labelled tasks)


### Visualising the Latent Space:

A 2D latent space z ∈ ℝ² can be visualised directly.
High-dimensional latent spaces are visualised via t-SNE or UMAP:
    - Project z to 2D preserving local neighbourhood structure
    - Clusters that emerge correspond to discovered categories

For MNIST digits (no labels given during training):
    AE latent space:        loose clusters, not well-separated
    VAE latent space:       smooth, continuous interpolation between digits
    Contrastive latent:     tight clusters, clean separation — best for classification


### The Downstream Task Pipeline:

    PRETRAIN unsupervised:   learn z = f(x) with no labels
    EVALUATE via linear probe:
        1. Freeze encoder f
        2. Train a linear classifier on top: ŷ = Wz + b
        3. Use a small labelled dataset (1% of original)
    RESULT: if the linear probe achieves high accuracy, the representation
            has captured class-relevant structure without seeing any labels.

    This is the gold standard for measuring unsupervised representation quality.


    # =======================================================================================# 
    **Diagram 6 — The Five Families: Loss Functions Compared:**

    Method          Loss Source              Loss Formula
    ──────────────────────────────────────────────────────────────────
    Autoencoder     Reconstruction           ||x − D(E(x))||²
    VAE             Reconstruction + KL      ||x − x̂||² + KL[q(z|x)||p(z)]
    GAN             Adversarial              E[log D(x)] + E[log(1−D(G(z)))]
    Self-Supervised Prediction               CrossEntropy(masked_token, predicted)
    Contrastive     Similarity metric        −log[ sim(z+) / Σ sim(z−) ]
    ──────────────────────────────────────────────────────────────────
    ALL five: no human labels y appear anywhere in the loss.
    # =======================================================================================# 


---


### Part 7: Where Each Method Excels


    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  AUTOENCODERS:                                                       │
    │    Best for:  anomaly detection, compression, denoising,             │
    │               tabular data representation                            │
    │    Weakness:  blurry reconstructions; no control over generation     │
    │                                                                      │
    │  VAEs:                                                               │
    │    Best for:  smooth generation, disentanglement, drug discovery,    │
    │               interpolation tasks                                    │
    │    Weakness:  blurrier than GANs; KL term can suppress latent info   │
    │                                                                      │
    │  GANs:                                                               │
    │    Best for:  photorealistic image synthesis, style transfer,        │
    │               data augmentation for rare classes                     │
    │    Weakness:  training instability, mode collapse; no encoder        │
    │                                                                      │
    │  SELF-SUPERVISED:                                                    │
    │    Best for:  language (GPT, BERT), vision (MAE, DINO),              │
    │               when massive unlabelled data is available              │
    │    Weakness:  requires large compute; pretext task must be           │
    │               well-chosen for representations to transfer            │
    │                                                                      │
    │  CONTRASTIVE:                                                        │
    │    Best for:  learning visual representations, few-shot learning,    │
    │               cross-modal alignment (CLIP: image ↔ text)             │
    │    Weakness:  requires large batches or careful negative mining;     │
    │               collapse risk without careful design                   │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘


### Relationship to the Supervised Module:

These methods do NOT replace supervised learning.
They COMPLEMENT it by providing better initialisation:

    Step 1: Pretrain unsupervised on 1,000,000 unlabelled images
    Step 2: Fine-tune supervised on 100 labelled images
    → Beats supervised training on 1,000,000 labelled images

    This is semi-supervised learning: the best of both worlds.
    Unlabelled data is cheap. Labels are expensive. Unsupervised pretraining
    stretches the value of every label.


---


### Part 8: Where Unsupervised Networks Sit in the Series


    LINEAR FEATURE LEARNING:
        PCA → Sparse Coding → ICA
        All learn linear latent representations. Closed-form or convex.

    NONLINEAR UNSUPERVISED:
        Autoencoder        ← this module, Part 1
        VAE                ← this module, Part 2
        GAN                ← this module, Part 3

    SELF-SUPERVISED:
        Self-supervised    ← this module, Part 4
        Contrastive        ← this module, Part 5

    LARGE SCALE:
        BERT / GPT / Claude:  self-supervised at scale → fine-tune
        CLIP:                  contrastive between images and text captions
        Stable Diffusion:      VAE + diffusion process + UNet
        AlphaFold:             self-supervised on protein sequences

The progression is: supervised networks (Module 11) → unsupervised networks
(this module) → architectures (CNNs, Transformers) → foundation models.

Unsupervised learning is the bridge from learning a mapping (supervised)
to learning the structure of the world itself.

"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ───────────────────────────────────────────────────────────────────────────────────────
    Method            Training Complexity               Notes
    ───────────────────────────────────────────────────────────────────────────────────────
    Autoencoder       O(epochs × n × forward_AE)        forward = encoder + decoder pass
    VAE               O(epochs × n × forward_VAE)       + KL computation (closed form)
    GAN               O(epochs × n × (fwd_D + fwd_G))   two networks, alternate updates
    Self-Supervised   O(epochs × n × forward_TF)        Transformer forward dominant
    Contrastive       O(epochs × n × 2 × forward)       2 augmented views per sample
    ───────────────────────────────────────────────────────────────────────────────────────
    Inference         O(forward pass only)              encoder pass → latent z
    Latent dim d      chosen: d << p                    p = input dim; bottleneck
    GAN stability     highly dependent on arch          requires careful tuning
    ───────────────────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Autoencoder from Scratch": {
        "description": "Encoder-bottleneck-decoder with MSE reconstruction loss — NumPy only",
        "runnable": True,
        "code": '''
"""
================================================================================
AUTOENCODER FROM SCRATCH — NUMPY ONLY
================================================================================

We build a symmetric encoder-decoder network that learns to compress MNIST
digits into a 2-dimensional latent space, then reconstruct them.

Architecture:
    Encoder:  784 → 128 → 64 → 2     (compress)
    Decoder:  2   → 64  → 128 → 784  (reconstruct)

Loss: Mean Squared Error between input x and reconstruction x̂
      No labels. The target IS the input.
================================================================================
"""

import numpy as np

np.random.seed(42)


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


# =============================================================================
# AUTOENCODER
# =============================================================================

class Autoencoder:
    """
    Symmetric autoencoder with a 2D latent space.

    Encoder:  784 → 128 → 64 → 2   (ReLU hidden, linear bottleneck)
    Decoder:  2   → 64 → 128 → 784  (ReLU hidden, sigmoid output for [0,1])

    Loss:     MSE = (1/n) Σ ||x - x̂||²
    No labels: target for reconstruction is the input itself.
    """

    def __init__(self, input_dim=784, latent_dim=2, lr=0.001):
        self.lr = lr

        # ── Encoder weights (He initialisation for ReLU layers) ─────────────
        e1 = np.sqrt(2.0 / input_dim)
        e2 = np.sqrt(2.0 / 128)
        e3 = np.sqrt(2.0 / 64)

        self.We1 = np.random.randn(input_dim, 128) * e1;  self.be1 = np.zeros((1, 128))
        self.We2 = np.random.randn(128, 64)         * e2;  self.be2 = np.zeros((1, 64))
        self.We3 = np.random.randn(64, latent_dim)  * e3;  self.be3 = np.zeros((1, latent_dim))

        # ── Decoder weights ───────────────────────────────────────────────────
        d1 = np.sqrt(2.0 / latent_dim)
        d2 = np.sqrt(2.0 / 64)
        d3 = np.sqrt(2.0 / 128)

        self.Wd1 = np.random.randn(latent_dim, 64)   * d1;  self.bd1 = np.zeros((1, 64))
        self.Wd2 = np.random.randn(64, 128)           * d2;  self.bd2 = np.zeros((1, 128))
        self.Wd3 = np.random.randn(128, input_dim)    * d3;  self.bd3 = np.zeros((1, input_dim))

        self.train_losses = []

    def encode(self, X):
        """Encoder forward pass → latent code z."""
        ze1 = X @ self.We1 + self.be1;  ae1 = relu(ze1)
        ze2 = ae1 @ self.We2 + self.be2;  ae2 = relu(ze2)
        ze3 = ae2 @ self.We3 + self.be3   # linear bottleneck
        cache = {"X": X, "ze1": ze1, "ae1": ae1, "ze2": ze2, "ae2": ae2, "ze3": ze3}
        return ze3, cache  # ze3 is the latent code z

    def decode(self, z):
        """Decoder forward pass → reconstruction x̂."""
        zd1 = z @ self.Wd1 + self.bd1;  ad1 = relu(zd1)
        zd2 = ad1 @ self.Wd2 + self.bd2;  ad2 = relu(zd2)
        zd3 = ad2 @ self.Wd3 + self.bd3;  x_hat = sigmoid(zd3)
        cache = {"z": z, "zd1": zd1, "ad1": ad1, "zd2": zd2, "ad2": ad2, "zd3": zd3}
        return x_hat, cache

    def forward(self, X):
        """Full forward pass: encode then decode."""
        z, enc_cache = self.encode(X)
        x_hat, dec_cache = self.decode(z)
        return x_hat, z, enc_cache, dec_cache

    def mse_loss(self, x, x_hat):
        """MSE reconstruction loss: (1/n) Σ ||x - x̂||²"""
        return np.mean((x - x_hat) ** 2)

    def backward_and_update(self, X, x_hat, enc_cache, dec_cache):
        """
        Backpropagation through decoder then encoder.

        Loss = MSE(x, x̂)  →  ∂L/∂x̂ = 2(x̂ - x) / (n × p)
        Then chain rule back through:
            Decoder:  sigmoid → ReLU → ReLU → latent z
            Encoder:  z → ReLU → ReLU → input x
        """
        n = X.shape[0]

        # ── Decoder backward ──────────────────────────────────────────────────
        dL_dxhat = 2 * (x_hat - X) / (n * X.shape[1])   # MSE gradient

        # Output layer (sigmoid)
        dL_dzd3  = dL_dxhat * sigmoid_grad(dec_cache["zd3"])
        dWd3     = dec_cache["ad2"].T @ dL_dzd3
        dbd3     = dL_dzd3.sum(axis=0, keepdims=True)
        dL_dad2  = dL_dzd3 @ self.Wd3.T

        # Decoder hidden 2 (ReLU)
        dL_dzd2  = dL_dad2 * relu_grad(dec_cache["zd2"])
        dWd2     = dec_cache["ad1"].T @ dL_dzd2
        dbd2     = dL_dzd2.sum(axis=0, keepdims=True)
        dL_dad1  = dL_dzd2 @ self.Wd2.T

        # Decoder hidden 1 (ReLU)
        dL_dzd1  = dL_dad1 * relu_grad(dec_cache["zd1"])
        dWd1     = dec_cache["z"].T @ dL_dzd1
        dbd1     = dL_dzd1.sum(axis=0, keepdims=True)
        dL_dz    = dL_dzd1 @ self.Wd1.T     # gradient reaching latent z

        # ── Encoder backward ──────────────────────────────────────────────────
        # Linear bottleneck: dL/dze3 = dL/dz (no activation)
        dWe3     = enc_cache["ae2"].T @ dL_dz
        dbe3     = dL_dz.sum(axis=0, keepdims=True)
        dL_dae2  = dL_dz @ self.We3.T

        dL_dze2  = dL_dae2 * relu_grad(enc_cache["ze2"])
        dWe2     = enc_cache["ae1"].T @ dL_dze2
        dbe2     = dL_dze2.sum(axis=0, keepdims=True)
        dL_dae1  = dL_dze2 @ self.We2.T

        dL_dze1  = dL_dae1 * relu_grad(enc_cache["ze1"])
        dWe1     = enc_cache["X"].T @ dL_dze1
        dbe1     = dL_dze1.sum(axis=0, keepdims=True)

        # ── SGD weight update ─────────────────────────────────────────────────
        for attr, grad in [("We1", dWe1), ("be1", dbe1),
                            ("We2", dWe2), ("be2", dbe2),
                            ("We3", dWe3), ("be3", dbe3),
                            ("Wd1", dWd1), ("bd1", dbd1),
                            ("Wd2", dWd2), ("bd2", dbd2),
                            ("Wd3", dWd3), ("bd3", dbd3)]:
            setattr(self, attr, getattr(self, attr) - self.lr * grad)

    def fit(self, X, epochs=30, batch_size=64):
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuf = X[idx]
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                Xb = X_shuf[start: start + batch_size]
                x_hat, z, enc_c, dec_c = self.forward(Xb)
                loss = self.mse_loss(Xb, x_hat)
                self.backward_and_update(Xb, x_hat, enc_c, dec_c)
                epoch_loss += loss
                batches += 1
            self.train_losses.append(epoch_loss / batches)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  |  Recon Loss: {self.train_losses[-1]:.4f}")


# =============================================================================
# DEMO
# =============================================================================

print("=" * 60)
print("  AUTOENCODER — UNSUPERVISED DIMENSIONALITY REDUCTION")
print("=" * 60)

# Synthetic data: 2 Gaussian clusters in high-dimensional space
# No labels used during training — only used to verify afterwards
n_per_class = 300
X_class0 = np.random.randn(n_per_class, 784) * 0.3 + 0.3
X_class1 = np.random.randn(n_per_class, 784) * 0.3 + 0.7
X_all    = np.vstack([X_class0, X_class1])
X_all    = np.clip(X_all, 0, 1)
labels   = np.array([0] * n_per_class + [1] * n_per_class)

print(f"\\nData: {X_all.shape[0]} samples, {X_all.shape[1]} dimensions")
print(f"      2 clusters — labels NOT used during training\\n")

ae = Autoencoder(input_dim=784, latent_dim=2, lr=0.005)
ae.fit(X_all, epochs=30, batch_size=64)

# ── Encode all samples → 2D latent space ─────────────────────────────────────
z_all, _ = ae.encode(X_all)

# ── Measure cluster separation in latent space ─────────────────────────────────
z0 = z_all[labels == 0]
z1 = z_all[labels == 1]
centroid0 = z0.mean(axis=0)
centroid1 = z1.mean(axis=0)
separation = np.linalg.norm(centroid1 - centroid0)
intra0     = np.mean(np.linalg.norm(z0 - centroid0, axis=1))
intra1     = np.mean(np.linalg.norm(z1 - centroid1, axis=1))

print(f"\\n{'─'*60}")
print(f"  LATENT SPACE ANALYSIS (after unsupervised training)")
print(f"{'─'*60}")
print(f"  Cluster 0 centroid: ({centroid0[0]:+.3f}, {centroid0[1]:+.3f})")
print(f"  Cluster 1 centroid: ({centroid1[0]:+.3f}, {centroid1[1]:+.3f})")
print(f"  Inter-cluster distance: {separation:.3f}")
print(f"  Intra-cluster spread:   {(intra0+intra1)/2:.3f}")
print(f"  Separation ratio:       {separation / ((intra0+intra1)/2):.2f}x")
print(f"\\n  → The autoencoder discovered cluster structure WITHOUT labels")
print(f"  → 784D space compressed to 2D; clusters are separable in latent space")
''',
    },

    "Variational Autoencoder (VAE)": {
        "description": "VAE with reparametrisation trick and ELBO loss — NumPy only",
        "runnable": True,
        "code": '''
"""
================================================================================
VARIATIONAL AUTOENCODER — NUMPY ONLY
================================================================================

Key differences from standard autoencoder:
    1. Encoder outputs μ AND log σ² (not a single point z)
    2. z sampled via reparametrisation:  z = μ + σ·ε,  ε ~ N(0,I)
    3. Loss = Reconstruction (MSE) + KL divergence from N(0,I)
    4. Can GENERATE new samples by sampling z ~ N(0,1) and decoding

The KL term forces the latent space to be smooth and well-organised.
================================================================================
"""

import numpy as np

np.random.seed(42)

def relu(z):     return np.maximum(0, z)
def relu_g(z):   return (z > 0).astype(float)
def sigmoid(z):  return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def sig_g(z):    s = sigmoid(z); return s * (1 - s)


class VAE:
    """
    Variational Autoencoder with a 2D latent space.

    Encoder:  p → 128 → (μ, log σ²)   both of dim 2
    Decoder:  2 → 128 → p

    ELBO loss = Reconstruction + β · KL
    KL (closed form) = -½ Σ (1 + log σ² - μ² - σ²)
    """

    def __init__(self, input_dim=20, latent_dim=2, lr=0.003, beta=1.0):
        self.lr, self.beta, self.latent_dim = lr, beta, latent_dim
        scale = np.sqrt(2.0 / input_dim)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.We1  = np.random.randn(input_dim, 128) * scale
        self.be1  = np.zeros((1, 128))
        self.Wmu  = np.random.randn(128, latent_dim) * 0.01
        self.bmu  = np.zeros((1, latent_dim))
        self.Wlv  = np.random.randn(128, latent_dim) * 0.01  # log variance
        self.blv  = np.zeros((1, latent_dim))

        # ── Decoder ──────────────────────────────────────────────────────────
        self.Wd1  = np.random.randn(latent_dim, 128) * np.sqrt(2.0 / latent_dim)
        self.bd1  = np.zeros((1, 128))
        self.Wd2  = np.random.randn(128, input_dim) * np.sqrt(2.0 / 128)
        self.bd2  = np.zeros((1, input_dim))

        self.losses = []

    def encode(self, X):
        ze1 = X @ self.We1 + self.be1;  ae1 = relu(ze1)
        mu  = ae1 @ self.Wmu + self.bmu
        lv  = ae1 @ self.Wlv + self.blv   # log σ²
        return mu, lv, {"X": X, "ze1": ze1, "ae1": ae1}

    def reparametrise(self, mu, lv):
        """
        Reparametrisation trick:  z = μ + σ · ε,  ε ~ N(0, I)
        ∂z/∂μ = 1,  ∂z/∂σ = ε   → gradients flow through μ and σ
        """
        eps = np.random.randn(*mu.shape)
        sigma = np.exp(0.5 * lv)          # σ = exp(log σ² / 2)
        z = mu + sigma * eps
        return z, eps, sigma

    def decode(self, z):
        zd1 = z @ self.Wd1 + self.bd1;  ad1 = relu(zd1)
        zd2 = ad1 @ self.Wd2 + self.bd2;  x_hat = sigmoid(zd2)
        return x_hat, {"z": z, "zd1": zd1, "ad1": ad1, "zd2": zd2}

    def kl_loss(self, mu, lv):
        """
        Closed-form KL divergence between N(μ, σ²) and N(0, 1):
            KL = -½ Σ (1 + log σ² - μ² - σ²)
        Averaged over the batch.
        """
        return -0.5 * np.mean(np.sum(1 + lv - mu**2 - np.exp(lv), axis=1))

    def train_step(self, X):
        n = X.shape[0]

        # ── Forward pass ──────────────────────────────────────────────────────
        mu, lv, enc_c = self.encode(X)
        z, eps, sigma  = self.reparametrise(mu, lv)
        x_hat, dec_c   = self.decode(z)

        # ── Loss ──────────────────────────────────────────────────────────────
        recon = np.mean((X - x_hat) ** 2)
        kl    = self.kl_loss(mu, lv)
        loss  = recon + self.beta * kl

        # ── Decoder backward ──────────────────────────────────────────────────
        dL_dxhat = 2 * (x_hat - X) / (n * X.shape[1])
        dL_dzd2  = dL_dxhat * sig_g(dec_c["zd2"])
        dWd2     = dec_c["ad1"].T @ dL_dzd2;  dbd2 = dL_dzd2.sum(0, keepdims=True)
        dL_dad1  = dL_dzd2 @ self.Wd2.T
        dL_dzd1  = dL_dad1 * relu_g(dec_c["zd1"])
        dWd1     = dec_c["z"].T @ dL_dzd1;    dbd1 = dL_dzd1.sum(0, keepdims=True)
        dL_dz    = dL_dzd1 @ self.Wd1.T

        # ── Reparametrisation backward ─────────────────────────────────────────
        # z = μ + σ·ε  →  ∂z/∂μ = 1,  ∂z/∂log σ² = ½ σ ε
        dL_dmu_recon = dL_dz
        dL_dlv_recon = dL_dz * (0.5 * sigma * eps)   # ∂z/∂lv chain

        # ── KL backward (closed form) ─────────────────────────────────────────
        # KL = -½ Σ(1 + lv - μ² - exp(lv))
        # ∂KL/∂μ = μ / n   ∂KL/∂lv = ½(exp(lv) - 1) / n
        dKL_dmu = mu / n
        dKL_dlv = 0.5 * (np.exp(lv) - 1) / n

        dL_dmu = dL_dmu_recon + self.beta * dKL_dmu
        dL_dlv = dL_dlv_recon + self.beta * dKL_dlv

        # ── Encoder backward ──────────────────────────────────────────────────
        dWmu = enc_c["ae1"].T @ dL_dmu;  dbmu = dL_dmu.sum(0, keepdims=True)
        dWlv = enc_c["ae1"].T @ dL_dlv;  dblv = dL_dlv.sum(0, keepdims=True)
        dL_dae1 = dL_dmu @ self.Wmu.T + dL_dlv @ self.Wlv.T
        dL_dze1 = dL_dae1 * relu_g(enc_c["ze1"])
        dWe1 = enc_c["X"].T @ dL_dze1;   dbe1 = dL_dze1.sum(0, keepdims=True)

        # ── Update ─────────────────────────────────────────────────────────────
        for attr, grad in [("We1",dWe1),("be1",dbe1),("Wmu",dWmu),("bmu",dbmu),
                            ("Wlv",dWlv),("blv",dblv),("Wd1",dWd1),("bd1",dbd1),
                            ("Wd2",dWd2),("bd2",dbd2)]:
            setattr(self, attr, getattr(self, attr) - self.lr * grad)

        return loss, recon, kl

    def fit(self, X, epochs=40, batch_size=64):
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            total_loss = total_recon = total_kl = 0.0; batches = 0
            for s in range(0, n, batch_size):
                Xb = X[idx[s:s+batch_size]]
                l, r, k = self.train_step(Xb)
                total_loss += l;  total_recon += r;  total_kl += k;  batches += 1
            avg = total_loss / batches
            self.losses.append(avg)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  |  "
                      f"ELBO: {avg:.4f}  Recon: {total_recon/batches:.4f}  "
                      f"KL: {total_kl/batches:.4f}")

    def generate(self, n_samples=5):
        """Generate new samples: sample z ~ N(0,I) → decode."""
        z = np.random.randn(n_samples, self.latent_dim)
        x_gen, _ = self.decode(z)
        return x_gen


# ── Demo ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  VARIATIONAL AUTOENCODER — GENERATIVE MODEL")
print("=" * 60)

# Synthetic dataset: two Gaussian clusters in 20D
n, p = 600, 20
X0 = np.random.randn(n//2, p) * 0.4 + np.array([1]*p)
X1 = np.random.randn(n//2, p) * 0.4 + np.array([-1]*p)
X_all = np.vstack([X0, X1])
X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min())  # scale to [0,1]

print(f"\\nData: {n} samples, {p} dimensions, 2 clusters, NO labels\\n")
vae = VAE(input_dim=p, latent_dim=2, lr=0.003, beta=1.0)
vae.fit(X_all, epochs=40, batch_size=64)

# ── Encode and inspect latent space ───────────────────────────────────────────
mu_all, lv_all, _ = vae.encode(X_all)
labels = np.array([0]*(n//2) + [1]*(n//2))
mu0 = mu_all[labels==0];  mu1 = mu_all[labels==1]
sep = np.linalg.norm(mu0.mean(0) - mu1.mean(0))
avg_kl = -0.5 * np.mean(np.sum(1 + lv_all - mu_all**2 - np.exp(lv_all), axis=1))

print(f"\\n{'─'*60}")
print(f"  LATENT SPACE INSPECTION")
print(f"{'─'*60}")
print(f"  Cluster 0 mean μ: {mu0.mean(axis=0).round(3)}")
print(f"  Cluster 1 mean μ: {mu1.mean(axis=0).round(3)}")
print(f"  Cluster centroid distance: {sep:.3f}")
print(f"  Average KL from N(0,1):    {avg_kl:.4f}")

# ── Generate new samples ──────────────────────────────────────────────────────
print(f"\\n{'─'*60}")
print(f"  GENERATING NEW SAMPLES (z ~ N(0,I) → Decoder)")
print(f"{'─'*60}")
generated = vae.generate(n_samples=4)
for i, s in enumerate(generated):
    print(f"  Sample {i+1}: min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}")
print(f"  → {len(generated)} new valid samples generated without any training input")
''',
    },

    "GAN (Generative Adversarial Network)": {
        "description": "Generator vs Discriminator adversarial training — NumPy only",
        "runnable": True,
        "code": '''
"""
================================================================================
GAN — GENERATIVE ADVERSARIAL NETWORK — NUMPY ONLY
================================================================================

Two networks trained simultaneously:
    Generator  G(z):  noise z ~ N(0,I) → fake samples
    Discriminator D:  sample → P(real)

Training alternates:
    1. Update D to distinguish real from fake (maximise log D(x) + log(1-D(G(z))))
    2. Update G to fool D (maximise log D(G(z)))

No labels: the only "labels" are real=1 (from dataset) and fake=0 (from G).
================================================================================
"""

import numpy as np

np.random.seed(42)

def relu(z):     return np.maximum(0, z)
def relu_g(z):   return (z > 0).astype(float)
def sigmoid(z):  return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))


class Generator:
    """G: z (noise_dim) → x (data_dim). Tries to produce real-looking samples."""

    def __init__(self, noise_dim, hidden, data_dim, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(noise_dim, hidden) * np.sqrt(2/noise_dim)
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, data_dim) * np.sqrt(2/hidden)
        self.b2 = np.zeros((1, data_dim))

    def forward(self, z):
        z1 = z @ self.W1 + self.b1;  a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2;  x_fake = np.tanh(z2)
        return x_fake, {"z": z, "z1": z1, "a1": a1, "z2": z2}

    def backward(self, cache, dL_dxfake):
        dL_dz2 = dL_dxfake * (1 - np.tanh(cache["z2"])**2)
        dW2    = cache["a1"].T @ dL_dz2
        db2    = dL_dz2.sum(0, keepdims=True)
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * relu_g(cache["z1"])
        dW1    = cache["z"].T @ dL_dz1
        db1    = dL_dz1.sum(0, keepdims=True)
        for attr, g in [("W1",dW1),("b1",db1),("W2",dW2),("b2",db2)]:
            setattr(self, attr, getattr(self, attr) - self.lr * g)


class Discriminator:
    """D: x (data_dim) → P(real). Binary classifier: real or fake?"""

    def __init__(self, data_dim, hidden, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(data_dim, hidden) * np.sqrt(2/data_dim)
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, 1) * np.sqrt(2/hidden)
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        z1 = x @ self.W1 + self.b1;  a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2;  prob = sigmoid(z2)
        return prob, {"x": x, "z1": z1, "a1": a1, "z2": z2}

    def backward(self, cache, dL_dprob):
        dL_dz2 = dL_dprob * (cache["prob"] if "prob" in cache else sigmoid(cache["z2"])) * (
                  1 - (cache["prob"] if "prob" in cache else sigmoid(cache["z2"])))
        # shortcut for combined sigmoid+BCE gradient
        n = cache["x"].shape[0]
        dL_dz2_bce = (sigmoid(cache["z2"]) - cache["target"]) / n
        dW2 = cache["a1"].T @ dL_dz2_bce
        db2 = dL_dz2_bce.sum(0, keepdims=True)
        da1 = dL_dz2_bce @ self.W2.T
        dz1 = da1 * relu_g(cache["z1"])
        dW1 = cache["x"].T @ dz1
        db1 = dz1.sum(0, keepdims=True)
        for attr, g in [("W1",dW1),("b1",db1),("W2",dW2),("b2",db2)]:
            setattr(self, attr, getattr(self, attr) - self.lr * g)
        return dL_dz2_bce @ self.W2.T @ np.diag(relu_g(cache["z1"]).mean(0)) @ self.W1.T

    def bce_forward(self, x, target):
        prob, cache = self.forward(x)
        cache["target"] = target
        y_c = np.clip(prob, 1e-15, 1-1e-15)
        loss = -np.mean(target * np.log(y_c) + (1-target) * np.log(1-y_c))
        return loss, prob, cache


# ── Training loop ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  GAN — ADVERSARIAL TRAINING")
print("=" * 60)

data_dim  = 8
noise_dim = 4
hidden    = 16
batch     = 64
epochs    = 200

# Real data: samples from a mixture of 2 Gaussians in 8D
def sample_real(n):
    half = n // 2
    c0 = np.random.randn(half, data_dim) * 0.3 + 1.0
    c1 = np.random.randn(n - half, data_dim) * 0.3 - 1.0
    return np.tanh(np.vstack([c0, c1]))   # tanh to match generator output range

G = Generator(noise_dim, hidden, data_dim, lr=0.002)
D = Discriminator(data_dim, hidden, lr=0.002)

print(f"\\nReal data: mixture of 2 Gaussians in {data_dim}D (no labels)\\n")

history = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}

for epoch in range(epochs):
    # ── Step 1: Train Discriminator ─────────────────────────────────────────
    x_real = sample_real(batch)
    z_noise = np.random.randn(batch, noise_dim)
    x_fake, g_cache = G.forward(z_noise)

    # D loss = BCE on real (label=1) + BCE on fake (label=0)
    d_loss_r, prob_r, cache_r = D.bce_forward(x_real, np.ones((batch, 1)))
    d_loss_f, prob_f, cache_f = D.bce_forward(x_fake, np.zeros((batch, 1)))
    d_loss = d_loss_r + d_loss_f

    D.backward(cache_r, None)
    D.backward(cache_f, None)

    # ── Step 2: Train Generator ──────────────────────────────────────────────
    z_noise2 = np.random.randn(batch, noise_dim)
    x_fake2, g_cache2 = G.forward(z_noise2)
    g_loss, prob_g, cache_g = D.bce_forward(x_fake2, np.ones((batch, 1)))  # want D to say "real"

    # Gradient from D back through G (D weights NOT updated here)
    n = batch
    dL_dz2 = (sigmoid(cache_g["z2"]) - np.ones((batch,1))) / n
    dL_dx_fake = (dL_dz2 @ D.W2.T) * relu_g(cache_g["z1"]) @ D.W1.T
    G.backward(g_cache2, dL_dx_fake)

    if (epoch + 1) % 40 == 0:
        history["d_loss"].append(d_loss)
        history["g_loss"].append(g_loss)
        history["d_real"].append(prob_r.mean())
        history["d_fake"].append(prob_f.mean())
        print(f"  Epoch {epoch+1:4d}  |  D loss: {d_loss:.3f}  G loss: {g_loss:.3f}  "
              f"| D(real)={prob_r.mean():.3f}  D(fake)={prob_f.mean():.3f}")

print(f"\\n{'─'*60}")
print(f"  TRAINING DYNAMICS")
print(f"{'─'*60}")
print(f"  Ideal Nash equilibrium: D(real) ≈ 0.5, D(fake) ≈ 0.5")
final_real = history["d_real"][-1]
final_fake = history["d_fake"][-1]
print(f"  Final D(real) = {final_real:.3f}  (should approach 0.5)")
print(f"  Final D(fake) = {final_fake:.3f}  (should approach 0.5)")
print(f"\\n  Generator quality metric: D(fake) rising toward 0.5 = improving G")
print(f"  → G learned to produce samples that confuse D, without any labels")
''',
    },

    "Self-Supervised: Masked Prediction": {
        "description": "Predict masked features from visible ones — the principle behind BERT and MAE",
        "runnable": True,
        "code": '''
"""
================================================================================
SELF-SUPERVISED LEARNING — MASKED FEATURE PREDICTION
================================================================================

The principle underlying BERT (masked token prediction) and MAE
(masked image patches), demonstrated on tabular data:

    1. Take an unlabelled sample x
    2. Randomly mask a fraction of its features
    3. Train a network to predict the masked values from the visible ones
    4. No human labels: the original feature values ARE the targets

After training, the encoder has learned to represent the full data structure.
The encoder can be frozen and a linear classifier fitted on top with few labels.
================================================================================
"""

import numpy as np

np.random.seed(42)


def relu(z):    return np.maximum(0, z)
def relu_g(z):  return (z > 0).astype(float)


class MaskedPredictor:
    """
    Self-supervised network: encode visible features → predict masked ones.

    Architecture:
        Encoder:  visible_features → hidden → representation z
        Decoder:  z → predict ALL features (loss only on masked ones)

    This is the tabular analogue of BERT's masked language modelling.
    """

    def __init__(self, input_dim, hidden=64, lr=0.003, mask_rate=0.3):
        self.input_dim = input_dim
        self.mask_rate = mask_rate
        self.lr = lr

        # Encoder: takes masked input (zeros for masked positions)
        self.We1 = np.random.randn(input_dim, hidden) * np.sqrt(2/input_dim)
        self.be1 = np.zeros((1, hidden))
        self.We2 = np.random.randn(hidden, hidden) * np.sqrt(2/hidden)
        self.be2 = np.zeros((1, hidden))

        # Decoder: predict all input features
        self.Wd1 = np.random.randn(hidden, input_dim) * np.sqrt(2/hidden)
        self.bd1 = np.zeros((1, input_dim))

        self.losses = []

    def create_mask(self, n):
        """Create random binary mask. 1 = visible, 0 = masked (to predict)."""
        mask = np.ones((n, self.input_dim))
        for i in range(n):
            masked_idx = np.random.choice(self.input_dim,
                                          size=int(self.input_dim * self.mask_rate),
                                          replace=False)
            mask[i, masked_idx] = 0
        return mask

    def forward(self, X, mask):
        """
        Forward pass with masking:
            Input to encoder: X * mask  (masked positions zeroed out)
            Encoder learns to infer masked values from context.
        """
        X_masked = X * mask   # zero out masked positions

        # Encoder
        ze1 = X_masked @ self.We1 + self.be1;  ae1 = relu(ze1)
        ze2 = ae1 @ self.We2 + self.be2;  ae2 = relu(ze2)   # representation

        # Decoder: predict full input
        pred = ae2 @ self.Wd1 + self.bd1

        cache = {"X_masked": X_masked, "ze1": ze1, "ae1": ae1,
                 "ze2": ze2, "ae2": ae2}
        return pred, ae2, cache

    def masked_mse(self, X, pred, mask):
        """
        MSE only on MASKED positions — the self-supervised objective.
        Masked positions: (1 - mask) = 1 where hidden.
        """
        invisible = 1 - mask   # 1 where masked, 0 where visible
        err = (pred - X) ** 2 * invisible
        count = invisible.sum() + 1e-8
        return err.sum() / count, invisible

    def train_step(self, X):
        n = X.shape[0]
        mask = self.create_mask(n)
        pred, rep, cache = self.forward(X, mask)
        loss, invisible = self.masked_mse(X, pred, mask)

        # Backprop: gradient only where masked
        dL_dpred = 2 * (pred - X) * invisible / (invisible.sum() + 1e-8)

        # Decoder backward
        dWd1 = cache["ae2"].T @ dL_dpred
        dbd1 = dL_dpred.sum(0, keepdims=True)
        dL_dae2 = dL_dpred @ self.Wd1.T

        # Encoder backward layer 2
        dL_dze2 = dL_dae2 * relu_g(cache["ze2"])
        dWe2 = cache["ae1"].T @ dL_dze2
        dbe2 = dL_dze2.sum(0, keepdims=True)
        dL_dae1 = dL_dze2 @ self.We2.T

        # Encoder backward layer 1
        dL_dze1 = dL_dae1 * relu_g(cache["ze1"])
        dWe1 = cache["X_masked"].T @ dL_dze1
        dbe1 = dL_dze1.sum(0, keepdims=True)

        for attr, g in [("We1",dWe1),("be1",dbe1),("We2",dWe2),("be2",dbe2),
                        ("Wd1",dWd1),("bd1",dbd1)]:
            setattr(self, attr, getattr(self, attr) - self.lr * g)

        return loss

    def encode(self, X):
        """Encode without masking — for downstream evaluation."""
        ze1 = X @ self.We1 + self.be1;  ae1 = relu(ze1)
        ze2 = ae1 @ self.We2 + self.be2;  ae2 = relu(ze2)
        return ae2

    def fit(self, X, epochs=50, batch_size=64):
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            total = 0; batches = 0
            for s in range(0, n, batch_size):
                Xb = X[idx[s:s+batch_size]]
                total += self.train_step(Xb);  batches += 1
            avg = total / batches
            self.losses.append(avg)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  |  Masked MSE: {avg:.4f}")


# ── Demo ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  SELF-SUPERVISED: MASKED FEATURE PREDICTION")
print("=" * 60)

# Dataset: 3 clusters in 16D — labels withheld during pretraining
n_per = 250;  dim = 16
c0 = np.random.randn(n_per, dim) * 0.5 + np.array( [2,  2] + [0]*(dim-2))
c1 = np.random.randn(n_per, dim) * 0.5 + np.array([-2,  2] + [0]*(dim-2))
c2 = np.random.randn(n_per, dim) * 0.5 + np.array([ 0, -3] + [0]*(dim-2))
X_all = np.vstack([c0, c1, c2])
y_all = np.array([0]*n_per + [1]*n_per + [2]*n_per)

print(f"\\nData: {X_all.shape[0]} samples, {dim} dimensions, 3 clusters")
print(f"Mask rate: 30% of features hidden per sample during pretraining")
print(f"Labels: NOT used during pretraining\\n")

model = MaskedPredictor(input_dim=dim, hidden=64, lr=0.003, mask_rate=0.30)
model.fit(X_all, epochs=50, batch_size=64)

# ── Linear probe: freeze encoder, train linear classifier with few labels ────
reps = model.encode(X_all)   # representations from self-supervised encoder

# Simulate few-label regime: only 10 labels per class
from numpy.linalg import lstsq
few_idx = []
for c in range(3):
    cls_idx = np.where(y_all == c)[0]
    few_idx.extend(cls_idx[:10])   # 10 labels per class = 30 total

X_few = reps[few_idx]
y_few = y_all[few_idx]

# One-hot encode
Y_oh = np.zeros((len(y_few), 3));  Y_oh[np.arange(len(y_few)), y_few] = 1
# Linear classifier (lstsq)
W_lin, _, _, _ = lstsq(np.hstack([X_few, np.ones((len(X_few),1))]),
                        Y_oh, rcond=None)
logits = np.hstack([reps, np.ones((len(reps),1))]) @ W_lin
pred_labels = logits.argmax(axis=1)
acc = (pred_labels == y_all).mean()

print(f"\\n{'─'*60}")
print(f"  LINEAR PROBE EVALUATION")
print(f"{'─'*60}")
print(f"  Training labels used:   30  ({10} per class)")
print(f"  Total samples:          {len(y_all)}")
print(f"  Linear probe accuracy:  {acc:.1%}")
print(f"\\n  → Self-supervised encoder learned class structure from")
print(f"    masked prediction — without any labels during pretraining.")
print(f"  → Linear classifier on frozen representations achieves {acc:.1%}")
print(f"    using only 30 out of {len(y_all)} labels.")
''',
    },

    "Contrastive Learning (SimCLR-style)": {
        "description": "NT-Xent contrastive loss with augmented pairs — the principle behind SimCLR and CLIP",
        "runnable": True,
        "code": '''
"""
================================================================================
CONTRASTIVE LEARNING — SimCLR-STYLE — NUMPY ONLY
================================================================================

For each sample x:
    1. Apply two random augmentations → (x₁, x₂)  [positive pair]
    2. All other samples in the batch are negatives
    3. NT-Xent loss: pull (x₁, x₂) together, push negatives apart

The network learns an embedding where similar inputs are close,
without ever being told which class anything belongs to.
================================================================================
"""

import numpy as np

np.random.seed(42)

def relu(z):   return np.maximum(0, z)
def relu_g(z): return (z > 0).astype(float)


def augment(X, noise=0.08, drop=0.15):
    """
    Two augmentation strategies applied independently:
        Gaussian noise + random feature dropout (zeroing)
    Mimics crop+colour jitter in vision contrastive learning.
    """
    mask = (np.random.rand(*X.shape) > drop).astype(float)
    return X * mask + np.random.randn(*X.shape) * noise


class ContrastiveEncoder:
    """
    Encoder f(·) + projection head g(·).
    Contrastive loss is computed on the projection z = g(f(x)).
    After training, the projection head is discarded; f(x) is used.
    """

    def __init__(self, input_dim, hidden=64, proj_dim=16, lr=0.003, tau=0.1):
        self.lr, self.tau = lr, tau

        # Encoder f
        self.We1 = np.random.randn(input_dim, hidden) * np.sqrt(2/input_dim)
        self.be1 = np.zeros((1, hidden))
        self.We2 = np.random.randn(hidden, hidden) * np.sqrt(2/hidden)
        self.be2 = np.zeros((1, hidden))

        # Projection head g
        self.Wp  = np.random.randn(hidden, proj_dim) * np.sqrt(2/hidden)
        self.bp  = np.zeros((1, proj_dim))

        self.losses = []

    def encode(self, X):
        ze1 = X @ self.We1 + self.be1;  ae1 = relu(ze1)
        ze2 = ae1 @ self.We2 + self.be2;  ae2 = relu(ze2)
        return ae2, {"X": X, "ze1": ze1, "ae1": ae1, "ze2": ze2}

    def project(self, h, enc_cache):
        zp = h @ self.Wp + self.bp   # linear projection (no activation)
        return zp

    def l2_normalise(self, z):
        norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        return z / norms

    def nt_xent_loss(self, z_i, z_j, tau):
        """
        NT-Xent (Normalised Temperature-scaled Cross Entropy):
            For each positive pair (i,j): maximise cosine sim(zᵢ,zⱼ)/τ
            relative to all 2N-2 negatives.

        z_i, z_j: L2-normalised projections, shape (N, proj_dim)
        """
        N = z_i.shape[0]
        # Concatenate both views: [z_i; z_j]  shape (2N, proj_dim)
        Z = np.vstack([z_i, z_j])   # (2N, proj_dim)

        # Similarity matrix: cosine similarity (vectors are L2-normalised)
        sim = Z @ Z.T / tau          # (2N, 2N)

        # Mask out self-similarity (diagonal)
        np.fill_diagonal(sim, -1e9)

        # For each sample i in [0,N):   positive is i+N
        # For each sample i in [N,2N):  positive is i-N
        pos_indices = np.concatenate([np.arange(N, 2*N), np.arange(N)])

        # Log-softmax trick for numerical stability
        sim_max = sim.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(sim - sim_max).sum(axis=1, keepdims=True)) + sim_max

        pos_sim = sim[np.arange(2*N), pos_indices]
        loss = -(pos_sim - log_sum_exp.squeeze()).mean()

        # Gradient ∂L/∂sim
        softmax = np.exp(sim - log_sum_exp) / 1.0   # (2N, 2N) but diag=-1e9
        softmax[np.arange(2*N), np.arange(2*N)] = 0.0
        grad_sim = softmax / (2 * N)
        grad_sim[np.arange(2*N), pos_indices] -= 1.0 / (2 * N)

        # ∂L/∂Z = (1/τ) * 2 * (grad_sim + grad_sim.T) @ Z  (symmetric)
        dL_dZ = (grad_sim + grad_sim.T) @ Z / tau
        dL_dzi = dL_dZ[:N]
        dL_dzj = dL_dZ[N:]

        return loss, dL_dzi, dL_dzj

    def train_step(self, X_batch):
        x_i = augment(X_batch)
        x_j = augment(X_batch)

        # Forward
        hi, enc_i = self.encode(x_i)
        hj, enc_j = self.encode(x_j)

        zi_raw = self.project(hi, enc_i)
        zj_raw = self.project(hj, enc_j)

        zi = self.l2_normalise(zi_raw)
        zj = self.l2_normalise(zj_raw)

        loss, dL_dzi, dL_dzj = self.nt_xent_loss(zi, zj, self.tau)

        # Backprop through L2 normalisation: ∂norm/∂z_raw = (I - z zᵀ) / ||z||
        def l2_norm_grad(z_norm, z_raw, dL_dz_norm):
            norms = np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-8
            dL_dz_raw = dL_dz_norm / norms - z_norm * (
                (dL_dz_norm * z_norm).sum(axis=1, keepdims=True)) / norms
            return dL_dz_raw

        dL_dzi_raw = l2_norm_grad(zi, zi_raw, dL_dzi)
        dL_dzj_raw = l2_norm_grad(zj, zj_raw, dL_dzj)

        def backward_one(dL_dzp, h, enc_cache):
            # Projection head backward
            dWp = h.T @ dL_dzp;    dbp = dL_dzp.sum(0, keepdims=True)
            dL_dh = dL_dzp @ self.Wp.T
            # Encoder backward
            dL_dze2 = dL_dh * relu_g(enc_cache["ze2"])
            dWe2 = enc_cache["ae1"].T @ dL_dze2;  dbe2 = dL_dze2.sum(0, keepdims=True)
            dL_dae1 = dL_dze2 @ self.We2.T
            dL_dze1 = dL_dae1 * relu_g(enc_cache["ze1"])
            dWe1 = enc_cache["X"].T @ dL_dze1;    dbe1 = dL_dze1.sum(0, keepdims=True)
            return dWe1, dbe1, dWe2, dbe2, dWp, dbp

        g_i = backward_one(dL_dzi_raw, hi, enc_i)
        g_j = backward_one(dL_dzj_raw, hj, enc_j)

        # Average gradients from both views
        for attr, gi, gj in zip(["We1","be1","We2","be2","Wp","bp"], g_i, g_j):
            setattr(self, attr, getattr(self, attr) - self.lr * (gi + gj) / 2)

        return loss

    def fit(self, X, epochs=60, batch_size=64):
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            total = 0; batches = 0
            for s in range(0, n, batch_size):
                Xb = X[idx[s:s+batch_size]]
                if len(Xb) < 4: continue
                total += self.train_step(Xb);  batches += 1
            avg = total / batches
            self.losses.append(avg)
            if (epoch + 1) % 15 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  |  NT-Xent Loss: {avg:.4f}")


# ── Demo ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  CONTRASTIVE LEARNING (SimCLR-style)")
print("=" * 60)

# 4 clusters in 20D, no labels during training
n_per = 200;  dim = 20
centres = [np.array([3,3]+[0]*(dim-2)), np.array([-3,3]+[0]*(dim-2)),
           np.array([-3,-3]+[0]*(dim-2)), np.array([3,-3]+[0]*(dim-2))]
X_parts = [np.random.randn(n_per, dim)*0.5 + c for c in centres]
X_all   = np.vstack(X_parts)
y_all   = np.concatenate([[i]*n_per for i in range(4)])

print(f"\\nData: {X_all.shape[0]} samples, {dim} dimensions, 4 clusters")
print(f"Augmentation: Gaussian noise + random feature dropout")
print(f"Labels: NOT used during training\\n")

enc = ContrastiveEncoder(input_dim=dim, hidden=64, proj_dim=16, lr=0.002, tau=0.15)
enc.fit(X_all, epochs=60, batch_size=64)

# ── Evaluate: linear probe on frozen encoder ───────────────────────────────────
h_all, _ = enc.encode(X_all)

from numpy.linalg import lstsq
few_idx = []
for c in range(4):
    cls_idx = np.where(y_all == c)[0][:5]   # only 5 labels per class
    few_idx.extend(cls_idx)

X_few = h_all[few_idx]
y_few = y_all[few_idx]
Y_oh = np.zeros((len(y_few),4));  Y_oh[np.arange(len(y_few)), y_few] = 1
W_lin, _, _, _ = lstsq(np.hstack([X_few, np.ones((len(X_few),1))]), Y_oh, rcond=None)
logits = np.hstack([h_all, np.ones((len(h_all),1))]) @ W_lin
pred = logits.argmax(axis=1)
acc = (pred == y_all).mean()

# ── Also compare: kNN in embedding space ──────────────────────────────────────
from numpy.linalg import norm
knn_preds = []
for i in range(len(h_all)):
    dists = norm(h_all[few_idx] - h_all[i], axis=1)
    knn_preds.append(y_few[dists.argmin()])
knn_acc = (np.array(knn_preds) == y_all).mean()

print(f"\\n{'─'*60}")
print(f"  LINEAR PROBE on CONTRASTIVE REPRESENTATIONS")
print(f"{'─'*60}")
print(f"  Training labels used:   20  (5 per class)")
print(f"  Total samples:          {len(y_all)}")
print(f"  Linear probe accuracy:  {acc:.1%}")
print(f"  1-NN probe accuracy:    {knn_acc:.1%}")
print(f"\\n  → Contrastive learning pulled same-class embeddings together")
print(f"    and pushed different-class embeddings apart — with no labels.")
print(f"  → {acc:.1%} accuracy using only 20 / {len(y_all)} labels demonstrates")
print(f"    how well the unsupervised representations captured structure.")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np

    print("\n" + "=" * 65)
    print("  UNSUPERVISED NEURAL NETWORKS — FIVE FAMILIES")
    print("=" * 65)
    print("""
  Key concepts demonstrated:
    • Autoencoder:       x → Encoder → z → Decoder → x̂   Loss: ||x−x̂||²
    • VAE:               z ~ N(μ(x), σ²(x)),  Loss: Recon + KL
    • GAN:               min_G max_D  E[log D(x)] + E[log(1−D(G(z)))]
    • Self-Supervised:   mask features → predict masked values (BERT/MAE principle)
    • Contrastive:       NT-Xent on augmented pairs (SimCLR principle)
    • ALL FIVE:          no human labels y used in the loss function
    """)

    np.random.seed(42)

    # ── Shared synthetic dataset: 3 clusters in 12D ──────────────────────────
    n_per = 200;  dim = 12
    c0 = np.random.randn(n_per, dim) * 0.4 + np.array([2,  2] + [0]*(dim-2))
    c1 = np.random.randn(n_per, dim) * 0.4 + np.array([-2,  2] + [0]*(dim-2))
    c2 = np.random.randn(n_per, dim) * 0.4 + np.array([ 0, -3] + [0]*(dim-2))
    X_all = np.vstack([c0, c1, c2])
    X_norm = (X_all - X_all.min()) / (X_all.max() - X_all.min())
    y_all = np.array([0]*n_per + [1]*n_per + [2]*n_per)

    print("=" * 65)
    print("  DATASET: 3 clusters in 12D  |  Labels withheld during training")
    print("=" * 65)
    print(f"  Samples: {X_all.shape[0]}   Dimensions: {dim}   Clusters: 3\n")

    # ── Simple helpers ────────────────────────────────────────────────────────
    def relu(z):    return np.maximum(0, z)
    def relu_g(z):  return (z > 0).astype(float)
    def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -500, 500)))

    def linear_probe_acc(X_reps, y_labels, n_per_class=5):
        """Train a linear classifier with very few labels, evaluate on all."""
        from numpy.linalg import lstsq
        n_cls = len(np.unique(y_labels))
        few_idx = []
        for c in range(n_cls):
            few_idx.extend(np.where(y_labels == c)[0][:n_per_class])
        X_few = X_reps[few_idx];  y_few = y_labels[few_idx]
        Y_oh = np.zeros((len(y_few), n_cls));  Y_oh[np.arange(len(y_few)), y_few] = 1
        A = np.hstack([X_few, np.ones((len(X_few),1))])
        W, _, _, _ = lstsq(A, Y_oh, rcond=None)
        preds = (np.hstack([X_reps, np.ones((len(X_reps),1))]) @ W).argmax(1)
        return (preds == y_labels).mean()

    results = {}

    # ── 1. Autoencoder ────────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  1/4  AUTOENCODER — reconstruction objective")
    print("─"*65)

    # Simple 2-layer AE
    We1 = np.random.randn(dim, 32) * np.sqrt(2/dim);   be1 = np.zeros((1,32))
    Wz  = np.random.randn(32, 4)  * np.sqrt(2/32);    bz  = np.zeros((1,4))
    Wd1 = np.random.randn(4, 32)  * np.sqrt(2/4);     bd1 = np.zeros((1,32))
    Wx  = np.random.randn(32, dim) * np.sqrt(2/32);   bx  = np.zeros((1,dim))
    lr_ae = 0.005

    for ep in range(60):
        idx = np.random.permutation(len(X_norm))
        for s in range(0, len(X_norm), 64):
            Xb = X_norm[idx[s:s+64]]
            # Forward
            h1 = relu(Xb @ We1 + be1)
            z  = relu(h1 @ Wz + bz)
            d1 = relu(z @ Wd1 + bd1)
            xh = sigmoid(d1 @ Wx + bx)
            # Backward
            n = Xb.shape[0]
            dxh = 2*(xh-Xb)/(n*dim)
            dWx_g = d1.T@(dxh*xh*(1-xh)); Wx -= lr_ae*dWx_g; bx -= lr_ae*(dxh*xh*(1-xh)).sum(0,keepdims=True)
            dd1   = (dxh*xh*(1-xh))@Wx.T * relu_g(z@Wd1+bd1)
            dWd1_g= z.T@dd1;  Wd1 -= lr_ae*dWd1_g; bd1 -= lr_ae*dd1.sum(0,keepdims=True)
            dz    = dd1@Wd1.T * relu_g(h1@Wz+bz)
            dWz_g = h1.T@dz;   Wz -= lr_ae*dWz_g;   bz -= lr_ae*dz.sum(0,keepdims=True)
            dh1   = dz@Wz.T * relu_g(Xb@We1+be1)
            dWe1_g= Xb.T@dh1;  We1-= lr_ae*dWe1_g;  be1-= lr_ae*dh1.sum(0,keepdims=True)

    h1 = relu(X_norm@We1+be1); reps_ae = relu(h1@Wz+bz)
    acc_ae = linear_probe_acc(reps_ae, y_all, n_per_class=5)
    results["Autoencoder"] = acc_ae
    print(f"  Latent dim: 4   Linear probe (5 labels/class): {acc_ae:.1%}")

    # ── 2. Self-Supervised Masking ────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  2/4  SELF-SUPERVISED — masked feature prediction")
    print("─"*65)

    Ws1 = np.random.randn(dim, 48) * np.sqrt(2/dim);   bs1 = np.zeros((1,48))
    Ws2 = np.random.randn(48, dim) * np.sqrt(2/48);    bs2 = np.zeros((1,dim))
    lr_ss = 0.005

    for ep in range(60):
        idx = np.random.permutation(len(X_norm))
        for s in range(0, len(X_norm), 64):
            Xb = X_norm[idx[s:s+64]]
            mask = (np.random.rand(*Xb.shape) > 0.3)
            Xm   = Xb * mask
            h    = relu(Xm @ Ws1 + bs1)
            pred = h @ Ws2 + bs2
            invis = ~mask
            n = Xb.shape[0]; cnt = invis.sum() + 1e-8
            dL = 2*(pred-Xb)*invis / cnt
            dWs2_g = h.T@dL;     Ws2 -= lr_ss*dWs2_g; bs2 -= lr_ss*dL.sum(0,keepdims=True)
            dh = dL@Ws2.T * relu_g(Xm@Ws1+bs1)
            dWs1_g = Xm.T@dh;   Ws1 -= lr_ss*dWs1_g; bs1 -= lr_ss*dh.sum(0,keepdims=True)

    reps_ss = relu(X_norm@Ws1+bs1)
    acc_ss = linear_probe_acc(reps_ss, y_all, n_per_class=5)
    results["Self-Supervised"] = acc_ss
    print(f"  Mask rate: 30%   Linear probe (5 labels/class): {acc_ss:.1%}")

    # ── 3. Contrastive Learning ───────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  3/4  CONTRASTIVE — NT-Xent on augmented pairs")
    print("─"*65)

    def augment_quick(X):
        return X * (np.random.rand(*X.shape) > 0.15) + np.random.randn(*X.shape)*0.06

    Wc1 = np.random.randn(dim, 48) * np.sqrt(2/dim);  bc1 = np.zeros((1,48))
    Wc2 = np.random.randn(48, 16) * np.sqrt(2/48);    bc2 = np.zeros((1,16))
    tau = 0.15;  lr_cl = 0.003

    def nt_xent_quick(zi, zj, tau):
        N = zi.shape[0]
        Z = np.vstack([zi, zj])
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        sim = Z@Z.T/tau;  np.fill_diagonal(sim, -1e9)
        pos = np.concatenate([np.arange(N, 2*N), np.arange(N)])
        sm = np.exp(sim - sim.max(1, keepdims=True))
        sm_sum = sm.sum(1, keepdims=True)
        loss = -(sim[np.arange(2*N), pos] - np.log(sm_sum.squeeze() + 1e-8)).mean()
        sft = sm/sm_sum; sft[np.arange(2*N),np.arange(2*N)] = 0
        gS = (sft - np.eye(2*N)[pos])/(2*N)
        dZ = ((gS + gS.T)@Z)/tau
        return loss, dZ[:N], dZ[N:]

    for ep in range(60):
        idx = np.random.permutation(len(X_norm))
        for s in range(0, len(X_norm), 64):
            Xb = X_norm[idx[s:s+64]]
            if len(Xb) < 4: continue
            xi, xj = augment_quick(Xb), augment_quick(Xb)
            hi = relu(xi@Wc1+bc1);  zi = hi@Wc2+bc2
            hj = relu(xj@Wc1+bc1);  zj = hj@Wc2+bc2
            _, dzi, dzj = nt_xent_quick(zi, zj, tau)
            for dz, h_enc, x_enc in [(dzi, hi, xi), (dzj, hj, xj)]:
                dWc2_g = h_enc.T@dz;   Wc2 -= lr_cl*dWc2_g; bc2 -= lr_cl*dz.sum(0,keepdims=True)
                dh = dz@Wc2.T * relu_g(x_enc@Wc1+bc1)
                dWc1_g = x_enc.T@dh;   Wc1 -= lr_cl*dWc1_g; bc1 -= lr_cl*dh.sum(0,keepdims=True)

    reps_cl = relu(X_norm@Wc1+bc1)
    acc_cl = linear_probe_acc(reps_cl, y_all, n_per_class=5)
    results["Contrastive"] = acc_cl
    print(f"  τ=0.15, aug=noise+dropout   Linear probe (5 labels/class): {acc_cl:.1%}")

    # ── 4. Random baseline ────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  4/4  BASELINE — random representations (control)")
    print("─"*65)
    reps_rand = np.random.randn(len(X_norm), 48)
    acc_rand = linear_probe_acc(reps_rand, y_all, n_per_class=5)
    results["Random (control)"] = acc_rand
    print(f"  Untrained random features   Linear probe (5 labels/class): {acc_rand:.1%}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY — Linear Probe Accuracy (5 labels per class = 15 total)")
    print(f"{'='*65}")
    print(f"  {'Method':<30} {'Accuracy':>10}")
    print(f"  {'─'*40}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 30)
        print(f"  {name:<30} {acc:>9.1%}  {bar}")

    print(f"""
  KEY TAKEAWAYS:
    1. All methods trained with ZERO labels — only the data structure
    2. Autoencoder: compress and reconstruct → learns efficient representations
    3. VAE: probabilistic bottleneck → smooth latent space → can generate
    4. GAN: adversarial loss → realistic generation without reconstruction
    5. Self-Supervised: mask+predict → BERT/MAE principle in one layer
    6. Contrastive: augmentation invariance → SimCLR/CLIP principle
    7. Representations transfer: few labels on frozen encoder → high accuracy
    8. Random control shows the gap: unsupervised learning adds real signal
    9. In practice: these methods scale to billions of parameters and
       outperform supervised models trained with millions of labels
   10. Modern AI (GPT, BERT, CLIP, Stable Diffusion) is built on these foundations
    """)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)

for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  VISUAL_HTML,
        "complexity":   COMPLEXITY,
        "operations":   OPERATIONS,
    }