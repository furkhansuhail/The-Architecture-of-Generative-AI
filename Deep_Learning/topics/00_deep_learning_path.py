"""Module 00 · Deep Learning — Architecture Reference Table"""

"""
Deep Learning — Complete Architecture Reference (ASIC Table)
=============================================================

A single-source reference for every major deep learning architecture and concept.
Each entry follows the ASIC format:

    A — Algorithm / Architecture name and family
    S — Summary (what problem it solves and how it works)
    I — Input / Output types
    C — Characteristics (assumptions, strengths, and limitations)

Use this as a map before diving into any individual architecture module.
"""

import os
import re
import textwrap

TOPIC_NAME = "Learning Path: Perceptron to LLMs"
DISPLAY_NAME = "00 · Deep Learning Path"
ICON         = "🧠"
SUBTITLE     = "Architecture · Summary · Input/Output · Characteristics — the complete deep learning map"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
What is Deep Learning?

Deep learning is the branch of machine learning that uses neural networks with many layers (hence "deep")
to learn hierarchical representations from raw data. Instead of hand-crafting features, the network
learns them automatically from examples via backpropagation and gradient descent.

Core idea:

    * Stack layers of learnable transformations: y = f_L( ... f_2( f_1(X; W_1); W_2) ... ; W_L)
    
    * Each layer learns increasingly abstract features (edges → shapes → objects for images)
    
    * Loss function measures prediction error; backprop computes gradients; optimiser updates weights

Key enabling components:
    * Activation functions  — introduce non-linearity (ReLU, GELU, Sigmoid, Tanh)
    * Loss functions        — measure error (Cross-Entropy, MSE, CTC, Contrastive)
    * Optimisers            — update weights (SGD, Adam, AdamW, RMSProp)
    * Regularisation        — prevent overfitting (Dropout, Batch Norm, Weight Decay)

The ASIC Table below maps every major deep learning family across four dimensions:
    A — Architecture      what the model is called and which family it belongs to
    S — Summary           what problem it solves and the core idea behind it
    I — Input / Output    what data it expects and what it produces
    C — Characteristics   key assumptions, strengths, and limitations


##### FAMILY 1 — FEEDFORWARD NETWORKS (MLP)

    The simplest deep networks. Input flows in one direction through fully connected layers.
    Each neuron computes a weighted sum of its inputs then applies a non-linear activation.
    Foundation of all other architectures.


    +------------------------------------+------------------------------------------------------------+---------------------------------------+----------------------------------------------------------------------------------------+
    | ARCHITECTURE                       | SUMMARY                                                    | INPUT / OUTPUT                        | CHARACTERISTICS                                                                        |
    +------------------------------------+------------------------------------------------------------+---------------------------------------+----------------------------------------------------------------------------------------+
    | Perceptron                         | Single-layer linear classifier. Computes a weighted sum    | Fixed-size feature vector     →       | Only linearly separable problems; historical origin of neural networks;                |
    |                                    | of inputs and applies a step activation function.          | Binary class                          | learning rule predates backprop; useful as conceptual baseline                         |
    +------------------------------------+------------------------------------------------------------+---------------------------------------+----------------------------------------------------------------------------------------+
    | Multi-Layer Perceptron (MLP)       | Stacks fully connected layers with non-linear activations. | Fixed-size feature vector     →       | Universal approximator (given enough neurons); requires feature engineering            |
    |                                    | Trained end-to-end via backpropagation.                    | Any output type                       | for raw data; struggles with spatial/temporal structure without CNN/RNN                |
    +------------------------------------+------------------------------------------------------------+---------------------------------------+----------------------------------------------------------------------------------------+
    | Deep MLP / Deep Feedforward Net    | MLP with many hidden layers (typically 5+).                | Fixed-size feature vector     →       | Can model highly complex functions; vanishing gradients without                        |
    |                                    | Requires careful initialisation and normalisation.         | Any output type                       | careful init / BatchNorm / residual connections; depth beats width                     |
    +------------------------------------+------------------------------------------------------------+---------------------------------------+----------------------------------------------------------------------------------------+
    | Radial Basis Function Network      | Hidden layer neurons compute Gaussian similarity to        | Fixed-size feature vector     →       | Fast to train (closed-form output layer); local receptive fields;                      |
    | (RBF Network)                      | learned prototype centres. Output is a weighted sum.       | Continuous / Class output             | number of centres is a hyperparameter; rarely used in modern practice                  |
    +------------------------------------+------------------------------------------------------------+---------------------------------------+----------------------------------------------------------------------------------------+
    | Mixture Density Network (MDN)      | MLP whose output is the parameters of a Gaussian           | Fixed-size feature vector     →       | Models multi-modal conditional distributions; avoids regression-to-mean;               |
    |                                    | Mixture Model — means, variances, and mixing weights.      | Distribution (GMM parameters)         | training can be unstable; used in robotics and speech synthesis                        |
    +------------------------------------+------------------------------------------------------------+---------------------------------------+----------------------------------------------------------------------------------------+


##### FAMILY 2 — CONVOLUTIONAL NEURAL NETWORKS (CNNs)

    CNNs exploit spatial locality and translation invariance via shared-weight filters.
    Each filter slides across the input, detecting a specific pattern regardless of position.
    Pooling reduces spatial dimensions, and stacked layers build increasingly abstract features.


    ARCHITECTURE                    SUMMARY                                                    INPUT / OUTPUT                        CHARACTERISTICS
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Vanilla CNN                     Alternates convolutional layers (feature extraction)       Image / 1D signal         →           Translation-equivariant; weight sharing massively reduces parameters
                                    with pooling layers (downsampling) and a final MLP.        Class / Feature map                   vs fully connected; requires large labelled datasets

    LeNet-5                         Early 5-layer CNN (conv → pool → conv → pool → FC).        Greyscale image (32×32)   →           Pioneered modern CNN structure; designed for digit recognition;
                                    Used tanh/sigmoid activations and average pooling.         Class label                           architecturally obsolete but historically important

    AlexNet                         First deep CNN to win ImageNet (2012). 8 layers,           RGB image (224×224)       →           Introduced ReLU, Dropout, and GPU training to mainstream DL;
                                    ReLU activations, Dropout, and data augmentation.          Class label (1000 classes)            sparked the deep learning revolution; now superseded

    VGGNet (VGG-16 / VGG-19)        Very deep networks using only 3×3 convolutions and         RGB image (224×224)       →           Simple, uniform architecture easy to understand and extend;
                                    max pooling. Depth (16–19 layers) drives performance.      Class label                           large number of parameters (~138M for VGG-16); slow to train

    GoogLeNet / Inception           Introduces Inception modules: parallel 1×1, 3×3, 5×5       RGB image (224×224)       →           Efficient use of computation; 1×1 convolutions reduce channels;
                                    and pooling paths concatenated along the channel axis.     Class label                           22 layers but fewer parameters than AlexNet; Inception v3/v4 refined

    ResNet (Residual Network)       Adds skip connections: output of block = F(x) + x.         RGB image (any size)      →           Enables very deep networks (50–152+ layers) without vanishing gradients;
                                    Learns residual functions rather than direct mappings.     Class label / Features                skip connections are now universal; backbone for transfer learning

    DenseNet (Dense Connections)    Each layer receives feature maps from all previous         RGB image                 →           Maximum feature reuse; alleviates vanishing gradients; fewer parameters
                                    layers via dense connections (concatenation).              Class label / Features                than ResNet; high memory usage due to concatenation

    EfficientNet                    Scales width, depth, and resolution simultaneously         RGB image (variable)      →           State-of-the-art accuracy / efficiency tradeoff; uses NAS to find
                                    using a compound scaling coefficient.                      Class label                           base architecture; EfficientNetV2 adds Fused-MBConv blocks

    MobileNet                       Uses depthwise separable convolutions to drastically       RGB image                 →           Designed for mobile / edge deployment; MobileNetV2 adds inverted
                                    reduce parameters and FLOPs.                               Class label                           residuals; V3 adds hard-swish and SE modules; very fast inference

    U-Net                           Encoder–decoder architecture with skip connections         Image (any modality)      →           Designed for biomedical segmentation; skip connections preserve
                                    from encoder to decoder at each resolution level.          Pixel segmentation mask               fine spatial detail; works well with very small datasets

    Feature Pyramid Network (FPN)   Builds a multi-scale pyramid of feature maps, fusing       Image                     →           Core component of many object detectors; strong at detecting
                                    bottom-up and top-down pathways with lateral connections.  Multi-scale feature maps              objects at different scales; used in Mask R-CNN, RetinaNet

    YOLO (You Only Look Once)       Single-pass object detection — predicts bounding boxes     Image                     →           Real-time detection; trades some accuracy for extreme speed;
                                    and class probabilities directly from the full image.      Bounding boxes + classes              YOLOv5/v8 are popular; anchor-free variants (YOLOX) improve recall

    Deformable Convolutional Net    Augments standard convolutions with learnable offsets,     Image                     →           Adapts receptive field to object geometry; improves detection
                                    allowing filters to sample non-grid positions.             Class / Detection / Segments          and segmentation of deformable objects; added complexity


##### FAMILY 3 — RECURRENT NETWORKS

    Recurrent networks process sequences by maintaining a hidden state that carries information
    from previous time steps. This makes them suited for language, audio, and time-series tasks.
    Largely superseded by Transformers for long sequences but still efficient on short ones.


    ARCHITECTURE                    SUMMARY                                                     INPUT / OUTPUT                      CHARACTERISTICS
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Vanilla RNN                     Computes hidden state hₜ = tanh(Wxhxₜ + Whhh_{t-1}).         Sequence →                         Captures short-range dependencies; suffers from vanishing / exploding
                                    Shares weights across all time steps.                       Sequence / Class / Value           gradients over long sequences; limited context window in practice
                                                                                                                                    
    Long Short-Term Memory (LSTM)   Introduces cell state and three gates (input, forget,      Sequence →                          Solves vanishing gradient problem; learns to retain / forget;
                                    output) to selectively preserve long-range information.    Sequence / Class / Value            widely used in NLP pre-Transformer era; slower than GRU

    Gated Recurrent Unit (GRU)      Simplified LSTM with two gates (reset, update).            Sequence →                          Fewer parameters than LSTM; often similar performance; faster
                                    Merges cell state and hidden state.                        Sequence / Class / Value            to train; good default when speed matters more than capacity

    Bidirectional RNN / LSTM        Runs two RNNs — one forward, one backward — and            Sequence (full context needed) →    Captures both past and future context at each position; cannot
                                    concatenates their hidden states at each step.             Sequence / Class                    be used for online / streaming prediction (needs full sequence)

    Deep RNN                        Stacks multiple recurrent layers. Each layer's output      Sequence →                          More expressive than single-layer; vanishing gradients harder to
                                    sequence becomes the next layer's input sequence.          Sequence / Class / Value            manage; dropout between layers mitigates overfitting

    Sequence-to-Sequence (Seq2Seq)  Encoder RNN compresses input sequence into a context       Sequence →                          Classic architecture for translation and summarisation;
                                    vector; decoder RNN generates output sequence from it.     Sequence                            information bottleneck at context vector; addressed by attention

    Seq2Seq with Attention          Decoder attends over all encoder hidden states at each     Sequence →                          Eliminates information bottleneck; precursor to Transformers;
                                    decoding step via a learnable alignment score.             Sequence                            Bahdanau (additive) and Luong (dot-product) variants

    Connectionist Temporal          Enables sequence-to-sequence mapping without requiring     Sequence (variable length) →        Used in speech recognition and OCR; handles alignment implicitly;
    Classification (CTC)            frame-level alignment labels. Sums over all alignments.    Sequence labels (shorter)           combined with RNN (CRNN) for text spotting in images


##### FAMILY 4 — ATTENTION MECHANISMS & TRANSFORMERS

    Transformers replaced recurrence with self-attention: every position attends to every other
    position simultaneously. This allows full parallelism and models arbitrarily long dependencies.
    The dominant paradigm for language, vision, audio, and multimodal tasks.


    ARCHITECTURE                    SUMMARY                                                     INPUT / OUTPUT                  CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Self-Attention                  Each token queries all other tokens with Q·Kᵀ/√d            Sequence of vectors →           O(n²) complexity in sequence length; full parallel computation;
                                    scores, then aggregates values V weighted by softmax.       Sequence of vectors             foundation of all Transformer variants

    Multi-Head Attention (MHA)      Runs h parallel attention heads in lower-dimensional        Sequence of vectors →           Each head can learn different relation types; outputs concatenated
                                    subspaces; outputs are concatenated and projected.          Sequence of vectors             and projected; used in every Transformer block

    Original Transformer            Encoder–decoder architecture with MHA + Feed-Forward        Sequence →                      Proposed "Attention is All You Need" (Vaswani 2017); encoder for
                                    sublayers, residual connections, and layer normalisation.   Sequence                        understanding, decoder for generation; foundation of BERT/GPT

    BERT                            Bidirectional Transformer encoder. Pre-trained with         Text tokens →                   Rich contextual representations; fine-tunes on downstream tasks;
                                    Masked Language Modelling (MLM) and NSP objectives.         Token embeddings / Class        bidirectional = cannot generate; excels at classification, NER, QA

    GPT (Generative Pre-Trained)    Causal (left-to-right) Transformer decoder. Pre-trained     Text tokens →                   Autoregressive generation; scales remarkably with data and compute;
    Transformer                     with next-token prediction on large text corpora.           Next token / Sequence           GPT-4 class models are state-of-the-art; fine-tuning via RLHF

    T5 (Text-to-Text Transfer)      Frames every NLP task as text-in → text-out.                Text →                          Unified framework for classification, translation, summarisation;
                                    Full encoder–decoder Transformer.                           Text                            easy to fine-tune any task; large model sizes needed for best results

    Vision Transformer (ViT)        Splits image into fixed-size patches, embeds each as        Image patches →                 Outperforms CNNs on large-scale image classification;
                                    a token, and applies a standard Transformer encoder.        Class label / Embeddings        needs large pre-training data; less inductive bias than CNN

    Swin Transformer                Computes self-attention within local shifted windows        Image patches →                 Linear complexity in image size; strong on detection/segmentation;
                                    rather than globally, then merges hierarchically.           Class / Feature hierarchy       hierarchical features match FPN-based detection pipelines

    CLIP (Contrastive Language-     Jointly trains an image encoder and text encoder with       Image + Text →                  Zero-shot image classification; powerful visual–language alignment;
    Image Pre-training)             contrastive loss to align matching image–text pairs.        Shared embedding space          used as backbone for multimodal models; sensitive to prompt wording

    DALL·E / Stable Diffusion       Text-conditioned image generation. Combines a text          Text prompt →                   Produces high-quality, diverse images; Stable Diffusion uses latent
    (Text-to-Image Transformers)    encoder (CLIP) with a generative image model.               Image                           diffusion for efficiency; controllable via prompting / LoRA

    Longformer / BigBird            Replace full self-attention with sparse attention           Long document (tokens) →        Reduces complexity from O(n²) to O(n); global tokens attend to all;
                                    patterns (sliding window + global tokens).                  Embeddings / Class              enables processing of very long documents (books, genomics)

    FlashAttention                  Exact attention computed with IO-aware tiling to avoid      Sequence →                      Not a new architecture — a faster implementation of standard MHA;
                                    materialising the full N×N attention matrix in HBM.         Sequence                        2–4× speedup and lower memory; enables longer context training


##### FAMILY 5 — GENERATIVE MODELS

    Generative models learn the data distribution P(X) or P(X|y) and can sample new,
    realistic examples. Central to image synthesis, drug discovery, speech generation,
    and large language models.


    ARCHITECTURE                    SUMMARY                                                     INPUT / OUTPUT                      CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Autoencoder (AE)                Encoder compresses input to a bottleneck z; decoder         Any data →                          Learns compressed representation; bottleneck forces meaningful
                                    reconstructs the original. Trained with reconstruction      Reconstruction                      latent codes; not generative by itself; decoder can hallucinate
                                    loss (MSE or BCE).

    Variational Autoencoder (VAE)   Encoder outputs a distribution q(z|x) = N(μ,σ²) rather      Any data →                          True generative model; samples from learned latent space;
                                    than a point. Trained with ELBO = recon loss + KL term.     Generated samples / Embeddings      generations can be blurry; latent space is smooth and interpolable

    Generative Adversarial          Generator G produces fake data from noise z;                Noise vector →                      Sharp, high-quality samples; training is unstable (mode collapse,
    Network (GAN)                   Discriminator D distinguishes real from fake.               Generated samples                   oscillation); many stabilisation tricks exist (WGAN, spectral norm)

    Conditional GAN (cGAN)          Conditions both G and D on extra information y              Noise + Condition →                 Generates samples from a specific class or matching a target;
                                    (class label, image, text).                                 Generated sample (cond. on y)       used in image-to-image translation (pix2pix)

    StyleGAN                        Learns a disentangled style space W; controls coarse        Noise →                             State-of-the-art for face / scene generation; style mixing;
                                    and fine features via adaptive instance normalisation.      High-res generated image            StyleGAN2/3 address aliasing; training requires many GPUs

    Normalising Flows               Series of invertible, differentiable transformations        Noise →                             Exact likelihood evaluation; fast exact inference; limited by
                                    maps simple prior to complex data distribution.             Generated samples / Density         requirement of invertibility; less flexible than VAE/GAN

    Diffusion Models (DDPM)         Gradually add Gaussian noise to data (forward process);     Noise → (denoising steps) →         State-of-the-art image quality; stable training vs GANs;
                                    train a network to reverse this process step by step.       Generated sample                    slow sampling (many denoising steps); DDIM and consistency
                                                                                                                                    models accelerate this

    Latent Diffusion Model (LDM)    Runs diffusion in the latent space of a VAE, not pixel      Noise (latent) →                    Much more compute-efficient than pixel-space diffusion;
                                    space. Text conditioning via cross-attention.               High-res generated image            backbone of Stable Diffusion; enables fast fine-tuning (LoRA)

    Energy-Based Model (EBM)        Assigns a scalar energy E(x) to every input; samples        Any data →                          Flexible density estimation; training requires MCMC or contrastive
                                    from the Boltzmann distribution ∝ exp(-E(x)/T).             Generated samples / Energy          divergence; computationally expensive; active research area

    Restricted Boltzmann Machine    Bipartite undirected graphical model. Visible and hidden    Binary / Continuous features →      Pre-deep-learning generative model; trained with contrastive
    (RBM)                           units connected; trained with contrastive divergence.       Generated samples                   divergence; stacked RBMs form Deep Belief Networks (DBN)


##### FAMILY 6 — GRAPH NEURAL NETWORKS (GNNs)

    GNNs operate on graph-structured data by passing messages between connected nodes.
    Used wherever data has relational structure: molecules, social networks, knowledge graphs,
    physics simulations, and recommendation systems.


    ARCHITECTURE                    SUMMARY                                                     INPUT / OUTPUT                      CHARACTERISTICS
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Graph Convolutional Network     Aggregates neighbour features with a normalised             Graph (nodes, edges, features) →    Spectral motivation; simple and effective; fixed graph structure
    (GCN)                           adjacency matrix: H = σ(D̃⁻¹ÃHW).                               Node / Graph embeddings             at inference; transductive (cannot generalise to new nodes)

    GraphSAGE                       Learns aggregation functions (mean, LSTM, pool)             Graph →                             Inductive — generalises to unseen nodes; samples fixed-size
                                    over sampled neighbourhoods, not the full graph.            Node embeddings                     neighbourhoods for scalability; used in Pinterest, Uber

    Graph Attention Network (GAT)   Applies attention coefficients when aggregating             Graph →                             Learns which neighbours matter; interpretable via attention;
                                    neighbour features — different neighbours weighted          Node / Graph embeddings             multi-head attention for stability; more expressive than GCN

    Message Passing Neural Net      Generalises GNNs as: nodes aggregate messages from          Graph (with edge features) →        Unifying framework for all GNN variants; edge features used in
    (MPNN)                          neighbours, then update their own state.                    Node / Edge / Graph output          molecular property prediction (QM9); expressive but may over-smooth

    Graph Transformer               Combines Transformer self-attention with graph              Graph →                             Captures global structure beyond local neighbourhoods; handles
                                    structure (relative position encoding on edges).            Node / Graph embeddings             long-range interactions in molecules and code; computationally heavy

    Heterogeneous GNN (HetGNN)      Handles graphs with multiple node and edge types            Heterogeneous graph →               Used in knowledge graphs and recommendation (users + items +
                                    via type-specific aggregation functions.                    Node / Graph embeddings             attributes); requires careful schema design


##### FAMILY 7 — SELF-SUPERVISED & CONTRASTIVE LEARNING

    Self-supervised learning creates supervision from the data itself, removing the need for
    manual labels. Models pre-trained this way learn powerful representations that transfer
    to many downstream tasks with minimal labelled data.


    ARCHITECTURE                    SUMMARY                                                     INPUT / OUTPUT                  CHARACTERISTICS
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    SimCLR                          Augments each image twice; trains encoder to maximise       Images (unlabelled) →           Requires large batch sizes to have enough negatives; simple and
                                    agreement between augmented views via contrastive loss.     Embeddings                      effective; linear probing on representations rivals supervised

    MoCo (Momentum Contrast)        Maintains a queue of negative examples updated with a       Images (unlabelled) →           Decouples batch size from number of negatives via queue;
                                    slow-moving (momentum) encoder copy.                        Embeddings                      memory efficient; MoCo v3 adapts for ViT backbones

    BYOL (Bootstrap Your Own        Uses two networks (online + target) with no negative        Images (unlabelled) →           No negative pairs needed; avoids collapse via stop-gradient
    Latent)                         examples. Online network predicts target network output.    Embeddings                      and EMA; simpler training; strong downstream performance

    DINO / DINOv2                   Self-distillation with no labels. Student network           Images (unlabelled) →           ViT features with DINO show emergent object segmentation;
                                    trained to match teacher (EMA) under augmentations.         Embeddings                      DINOv2 scales to large diverse datasets; strong transfer learning

    Masked Autoencoder (MAE)        Masks ~75% of image patches; trains ViT encoder +           Images (unlabelled) →           Extremely efficient — encoder only sees visible patches; strong
                                    lightweight decoder to reconstruct masked patches.          Embeddings / Reconstruction     fine-tuning performance; scales well; inspired by BERT masking

    SimSiam                         Similar to BYOL but uses stop-gradient only (no EMA,        Images (unlabelled) →           No negative pairs, no momentum encoder; simpler; avoids collapse
                                    no negative pairs). Encoder–predictor asymmetry.            Embeddings                      via stop-gradient; works with small batch sizes

    Contrastive Language-Image      Contrastive pre-training on 400M image–text pairs.          Image + Text pairs →            Zero-shot transfer to dozens of tasks; powerful shared embedding
    Pre-training (CLIP)             Aligns vision and language in a shared embedding space.     Shared embedding space          space; sensitive to distribution shift between training and test


##### FAMILY 8 — REGULARISATION & TRAINING TECHNIQUES

    Techniques that are not architectures in themselves, but critical decisions that determine
    whether a network trains successfully and generalises well.


    TECHNIQUE                       SUMMARY                                                     APPLIES TO                      CHARACTERISTICS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Batch Normalisation             Normalises activations within each mini-batch: zero         Any deep network                Accelerates training; acts as mild regulariser; problematic with
    (BatchNorm)                     mean, unit variance. Learns scale γ and shift β.                                            small batches; use LayerNorm for Transformers / small batches

    Layer Normalisation             Normalises across the feature dimension for each            Transformers, RNNs              Independent of batch size; preferred in NLP and sequential models;
    (LayerNorm)                     sample individually.                                                                        computationally equivalent to BatchNorm but more stable in DL

    Dropout                         During training, randomly zeros out each unit with          MLPs, CNNs, RNNs                Simple and effective; equivalent to ensembling many sub-networks;
                                    probability p. Removed at inference (scale by 1-p).                                         too high p hurts learning; use 0.1–0.5 depending on model size

    Weight Decay (L2 Regulariser)   Adds λ·‖W‖² penalty to the loss. Equivalent to a            All networks                    Prevents weight explosion; use AdamW for correct decoupled weight
                                    Gaussian prior over weights.                                                                decay with Adam optimiser; most common regularisation choice

    Data Augmentation               Artificially expands training set with label-preserving     Vision, NLP, Audio              Most impactful regulariser for vision; random crop/flip/colour
                                    transformations (flips, crops, cutout, mixup, cutmix).                                      jitter; Mixup and CutMix mix examples and labels

    Early Stopping                  Halt training when validation loss stops improving.         All networks                    Free regularisation; requires a held-out validation set;
                                    Restore weights from the best validation checkpoint.                                        watch out for noisy validation curves; use patience parameter

    Learning Rate Scheduling        Reduce LR during training (step decay, cosine annealing,    All networks                    Critical for convergence quality; cosine annealing + warm restarts
                                    linear warmup + cosine, cyclic LR).                                                         (SGDR) popular; linear warmup prevents early instability

    Gradient Clipping               Clip gradient norm to a maximum value before the            RNNs, Transformers              Essential for RNNs to prevent exploding gradients; clip-by-norm
                                    parameter update.                                                                           preferred over clip-by-value; common threshold: 1.0 or 5.0

    Mixup / CutMix                  Mixup: train on convex combinations of pairs (λx₁+(1-λ)     Vision (and beyond)             Improves calibration and generalisation; encourages linear behaviour
                                    x₂, λy₁+(1-λ)y₂). CutMix: paste crops between images.                                       between classes; especially useful with ViTs

    Label Smoothing                 Replace hard one-hot targets with (1-ε)·one-hot +           Classification tasks            Prevents overconfidence; improves calibration; small ε (0.1) works
                                    ε/K·uniform distribution.                                                                   well; standard practice in Transformer training

    Stochastic Depth (DropPath)     Randomly drops entire residual blocks during training.      Deep ResNets, ViTs              Different from Dropout; reduces effective depth at training time;
                                    At inference all blocks are used.                                                           improves regularisation; drop rate increases with layer depth


##### FAMILY 9 — OPTIMISERS

    The optimiser determines how gradients are used to update weights. Choice of optimiser
    significantly affects training speed, stability, and final performance.


    OPTIMISER                       SUMMARY                                                     APPLIES TO                      CHARACTERISTICS
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    SGD (Stochastic Gradient        Updates weights in the negative gradient direction          All networks                    Simple and well-understood; good asymptotic convergence; requires
    Descent)                        using a mini-batch estimate: w ← w - η∇L.                                                   careful LR tuning; use with momentum (0.9) and LR schedule

    SGD with Momentum               Accumulates a velocity vector in gradient direction:        All networks                    Dampens oscillations; accelerates convergence in ravines;
                                    v ← βv - η∇L, w ← w + v.                                                                    β=0.9 standard; Nesterov momentum (look-ahead) slightly better

    Adam (Adaptive Moment           Maintains per-parameter running mean of gradient (m)        Transformers, GANs, most DL     Fast convergence; little LR tuning needed; can generalise slightly
    Estimation)                     and squared gradient (v). Bias-corrected updates.                                           worse than SGD for CNNs; use AdamW (decoupled weight decay)

    AdamW                           Adam with decoupled L2 weight decay (not mixed into         Transformers, ViTs, LLMs        Standard optimiser for large-scale deep learning; correct weight
                                    the adaptive gradient scaling).                                                             decay; default choice for pre-training language models

    RMSProp                         Maintains running average of squared gradients to           RNNs, RL                        Historically popular for RNNs; adaptive learning rates per param;
                                    normalise gradients. No momentum term.                                                      often outperformed by Adam in modern practice

    Lion (EvoLved Sign Momentum)    Uses sign of momentum update: w ← w - η·sign(βm + g).       Large-scale vision, LLMs        More memory efficient than Adam (one state vs two); competitive
                                    Only the sign of the update is used.                                                        performance; lower LR than Adam typically needed

    LARS / LAMB                     Scales learning rate by the ratio of weight norm to         Large-batch CNN / BERT          Enables very large batch training (16k+ samples) without accuracy
                                    gradient norm, per layer.                                   pre-training                    loss; LAMB extends LARS to Adam; used for BERT pre-training


##### FAMILY 10 — TRANSFER LEARNING & FINE-TUNING PARADIGMS

    Training from scratch is prohibitively expensive for most practitioners. Transfer learning
    leverages models pre-trained on large datasets and adapts them to new tasks efficiently.


    PARADIGM                        SUMMARY                                                     APPLIES TO                      CHARACTERISTICS
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Feature Extraction              Freeze all pre-trained weights. Add and train only a        CNNs, BERT, ViT                 Fastest approach; appropriate when target domain ≈ source domain;
                                    new task-specific head (linear layer or MLP).                                               linear probe performance benchmarks representation quality

    Full Fine-Tuning                Unfreeze the entire pre-trained network and continue        All pre-trained models          Best accuracy when enough target data available; risk of catastrophic
                                    training on target task data with a low learning rate.                                      forgetting of pre-trained knowledge; expensive for large models

    Gradual Unfreezing              Unfreeze layers from top to bottom progressively during     CNNs, BERT                      Reduces catastrophic forgetting; ULMFiT popularised for NLP;
                                    fine-tuning. Higher layers adapt first.                                                     discriminative LR (lower for early layers) often combined

    LoRA (Low-Rank Adaptation)      Injects trainable low-rank matrices ΔW = AB into            LLMs, Diffusion models          <1% of parameters trained; no inference latency; adapters can be
                                    frozen pre-trained weight matrices.                                                         merged back; de facto standard for efficient LLM fine-tuning

    Prefix / Prompt Tuning          Prepends a small set of trainable token embeddings          LLMs                            Extremely parameter-efficient; leaves model weights frozen;
                                    to the input. Only these are trained.                                                       prompt tuning matches fine-tuning at very large scales (>10B)

    Adapters                        Inserts small bottleneck modules (down-project →            BERT, ViT, LLMs                 Modular: one adapter per task; share the base model;
                                    non-linearity → up-project) after each Transformer block.                                   slight inference overhead; precursor to LoRA

    Instruction Fine-Tuning         Fine-tunes LLM on diverse (instruction, response) pairs     LLMs                            Makes models more helpful and instruction-following; combined with
                                    to improve instruction following.                                                           RLHF for alignment; used in InstructGPT, Claude, Llama-2-Chat

    RLHF (Reinforcement Learning    Trains a reward model on human preference data; then        LLMs                            Key technique for aligning LLMs with human intent; PPO optimiser
    from Human Feedback)            fine-tunes LLM with PPO to maximise reward.                                                 commonly used; DPO is a simpler offline alternative


##### HOW TO CHOOSE — A DECISION GUIDE

    ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                │
    │  What is your data modality?                                                                   │
    │                                                                                                │
    │  Images / Video          → Start with a pre-trained CNN (ResNet, EfficientNet) or ViT          │
    │  Text / Language         → Pre-trained Transformer (BERT for understanding, GPT for gen)       │
    │  Audio / Speech          → Wav2Vec 2.0, Whisper, or 1D CNN + LSTM                              │
    │  Time Series             → Temporal CNN, LSTM, or Transformer with positional encoding         │
    │  Tabular / Mixed         → MLP or Gradient Boosting (DL rarely wins on pure tabular data)      │
    │  Graphs / Molecules      → GNN (GCN, GAT, MPNN, Graph Transformer)                             │
    │  Multi-modal             → CLIP-based or cross-attention fusion architecture                   │
    │                                                                                                │
    │  What is your task?                                                                            │
    │                                                                                                │
    │  Classification          → CNN (images), Transformer (text), MLP (features)                    │
    │  Object Detection        → YOLO, Faster R-CNN, DETR (Transformer-based detector)               │
    │  Segmentation            → U-Net (medical), Mask R-CNN, SAM (Segment Anything)                 │
    │  Generation (images)     → Stable Diffusion (LDM), StyleGAN3, DALL·E                           │
    │  Generation (text)       → GPT-class autoregressive Transformer                                │
    │  Translation / Seq2Seq   → Transformer encoder–decoder (T5, BART)                              │
    │  Representation learning → MAE, DINO, SimCLR, CLIP                                             │
    │  Graph tasks             → GCN (transductive), GraphSAGE (inductive), GAT                      │
    │                                                                                                │
    │  How much labelled data do you have?                                                           │
    │                                                                                                │
    │  None (unlabelled only)  → Self-supervised pre-training (MAE, DINO, SimCLR)                    │
    │  Very little (<1k)       → Few-shot with a large pre-trained model + linear probe              │
    │  Moderate (1k – 100k)    → Fine-tune a pre-trained model (LoRA, full fine-tune)                │
    │  Large (100k+)           → Fine-tune or train from scratch if domain is unique                 │
    │                                                                                                │
    │  What are your compute constraints?                                                            │
    │                                                                                                │
    │  Edge / Mobile           → MobileNet, EfficientNet-Lite, quantised models                      │
    │  Single GPU              → LoRA fine-tuning, distilled models, efficient ViTs                  │
    │  Multi-GPU cluster       → Full fine-tuning, pre-training, diffusion models                    │
    │  Unlimited compute       → Scale up model size + data (scaling laws apply)                     │
    │                                                                                                │
    │  Do you need interpretability?                                                                 │
    │                                                                                                │
    │  High interpretability   → Attention maps (ViT/BERT), saliency maps, SHAP on MLPs              │
    │  Moderate                → GradCAM for CNNs, attention rollout for Transformers                │
    │  Prediction only         → Largest / most accurate model regardless of interpretability        │
    │                                                                                                │
    └────────────────────────────────────────────────────────────────────────────────────────────────┘


##### QUICK REFERENCE — ARCHITECTURE FAMILIES AT A GLANCE

    Family                           Architectures
    ─────────────────────────────────────────────────────────────────────────────────────────────
    Feedforward Networks             Perceptron, MLP, Deep MLP, RBF Network, MDN

    Convolutional Networks           LeNet, AlexNet, VGG, GoogLeNet/Inception, ResNet,
                                     DenseNet, EfficientNet, MobileNet, U-Net, FPN, YOLO

    Recurrent Networks               Vanilla RNN, LSTM, GRU, Bidirectional RNN,
                                     Seq2Seq, Seq2Seq + Attention, CTC

    Attention & Transformers         Self-Attention, MHA, Original Transformer, BERT,
                                     GPT, T5, ViT, Swin, CLIP, Longformer, FlashAttention

    Generative Models                Autoencoder, VAE, GAN, cGAN, StyleGAN,
                                     Normalising Flows, DDPM, LDM (Stable Diffusion), EBM, RBM

    Graph Neural Networks            GCN, GraphSAGE, GAT, MPNN, Graph Transformer, HetGNN

    Self-Supervised / Contrastive    SimCLR, MoCo, BYOL, DINO/DINOv2, MAE, SimSiam, CLIP

    Regularisation Techniques        BatchNorm, LayerNorm, Dropout, Weight Decay,
                                     Data Augmentation, Early Stopping, Mixup/CutMix,
                                     Label Smoothing, Stochastic Depth

    Optimisers                       SGD, SGD+Momentum, Adam, AdamW, RMSProp, Lion, LARS/LAMB

    Transfer & Fine-Tuning           Feature Extraction, Full Fine-Tuning, Gradual Unfreezing,
                                     LoRA, Prefix Tuning, Adapters, Instruction Fine-Tuning, RLHF
    ─────────────────────────────────────────────────────────────────────────────────────────────


## The Road from Perceptron to Large Language Models

This application walks you through the **complete evolution of AI**, starting from
the simplest computational neuron and building all the way to modern generative
models. Each topic in the sidebar corresponds to a milestone on this journey.

---

### Phase 1 — Mathematical Foundations
Before touching any neural network, you need comfort with:
- **Linear Algebra**: vectors, matrices, dot products, matrix multiplication
- **Calculus**: derivatives, partial derivatives, chain rule
- **Probability & Statistics**: distributions, Bayes' theorem, expectation
- **Optimization**: gradient descent, learning rate, convergence

> *You don't need a PhD — you need intuition for what these tools do and why.*

---


### Phase 2 — The Perceptron & Single Neurons
The **perceptron** (Rosenblatt, 1958) is where it all begins:
- A single unit that takes inputs, multiplies by weights, adds a bias, and applies a step function
- Can learn linearly separable patterns (AND, OR) but fails on XOR
- Introduces the core loop: **forward pass → compute error → update weights**

Key concepts: *weights, bias, activation function, decision boundary, learning rule*

---

### Phase 3 — Multi-Layer Networks & Backpropagation
Stacking perceptrons into layers solves the XOR problem and much more:
- **Feedforward Neural Networks** (MLPs): input → hidden layers → output
- **Activation functions**: sigmoid, tanh, ReLU and why they matter
- **Backpropagation** (Rumelhart et al., 1986): the chain rule applied layer by layer
- **Loss functions**: MSE for regression, cross-entropy for classification

Key concepts: *hidden layers, vanishing gradients, weight initialization, epochs, batches*

---

### Phase 4 — Convolutional Neural Networks (CNNs)
Designed for spatial data (images, grids):
- **Convolution**: sliding a filter/kernel across input to detect features
- **Pooling**: reducing spatial dimensions while keeping important info
- **Architectures**: LeNet → AlexNet → VGG → ResNet → EfficientNet
- Concepts: *feature maps, stride, padding, receptive field, skip connections*

---

### Phase 5 — Recurrent Neural Networks (RNNs)
Designed for sequential data (text, time series):
- **Vanilla RNN**: hidden state passed from step to step — suffers from vanishing gradients
- **LSTM** (Hochreiter & Schmidhuber, 1997): gates (forget, input, output) to control memory
- **GRU**: simplified LSTM with fewer parameters
- **Bidirectional RNNs**: reading sequences both forward and backward

Key concepts: *hidden state, sequence-to-sequence, teacher forcing, beam search*

---

### Phase 6 — The Attention Mechanism
The bridge between RNNs and Transformers:
- **Problem**: RNNs compress entire sequences into a fixed-size vector (bottleneck)
- **Solution**: let the decoder *attend* to all encoder states, weighted by relevance
- **Bahdanau Attention** (2014): additive attention over encoder states
- **Luong Attention** (2015): dot-product variants

Key concepts: *query, key, value, attention weights, context vector*

---

### Phase 7 — The Transformer
"Attention Is All You Need" (Vaswani et al., 2017) — the architecture that changed everything:
- **Self-Attention**: every token attends to every other token in the sequence
- **Multi-Head Attention**: run attention in parallel across multiple representation subspaces
- **Positional Encoding**: inject sequence order since there's no recurrence
- **Encoder-Decoder structure**: encoder reads input, decoder generates output

Key concepts: *scaled dot-product attention, layer normalization, residual connections, feed-forward layers*

---

### Phase 8 — Pre-trained Language Models
Taking Transformers and training them on massive text corpora:
- **BERT** (2018): encoder-only, masked language modeling, bidirectional
- **GPT** (2018–2024): decoder-only, autoregressive, next-token prediction
- **T5** (2019): encoder-decoder, text-to-text framework
- **Scaling laws**: more parameters + more data = emergent capabilities

Key concepts: *pre-training, fine-tuning, tokenization (BPE, WordPiece), transfer learning*

---

### Phase 9 — Generative AI & LLMs
The current frontier:
- **In-Context Learning**: few-shot prompting without gradient updates
- **RLHF**: reinforcement learning from human feedback for alignment
- **Instruction Tuning**: training models to follow instructions
- **RAG** (Retrieval-Augmented Generation): grounding LLMs with external knowledge
- **Prompt Engineering**: crafting inputs to get optimal outputs
- **Agents & Tool Use**: LLMs that can call APIs, write code, browse the web

Key concepts: *temperature, top-k/top-p sampling, context window, hallucination, grounding*

---

### Phase 10 — Beyond Text
- **Diffusion Models**: Stable Diffusion, DALL-E — image generation via iterative denoising
- **Multimodal Models**: GPT-4V, Gemini — processing text + images + audio
- **Video Generation**: Sora, Runway — temporal coherence in generated content
- **Speech**: Whisper (ASR), TTS models
- **Mixture of Experts (MoE)**: activating only a subset of parameters per input

---

### Suggested Study Order in This App

| #  | Topic | Builds On |
|----|---------------------------------|---------------------------|
| 1  | Perceptron                      | —                         |
| 2  | Activation Functions            | Perceptron                |
| 3  | Loss Functions                  | Activation Functions      |
| 4  | Gradient Descent                | Loss Functions, Calculus  |
| 5  | Backpropagation                 | Gradient Descent          |
| 6  | Feedforward Neural Networks     | All above                 |
| 7  | CNNs                            | Feedforward NN            |
| 8  | RNNs / LSTM / GRU               | Feedforward NN            |
| 9  | Attention Mechanism             | RNNs                      |
| 10 | Transformer Architecture        | Attention                 |
| 11 | BERT & Encoder Models           | Transformer               |
| 12 | GPT & Decoder Models            | Transformer               |
| 13 | Fine-Tuning & Transfer Learning | Pre-trained Models        |
| 14 | RLHF & Alignment                | GPT                       |
| 15 | RAG                             | LLMs                      |
| 16 | Prompt Engineering              | LLMs                      |
| 17 | Diffusion Models                | Neural Nets, Probability  |
| 18 | Multimodal Models               | Transformer, Vision       |
| 19 | Agents & Tool Use               | LLMs, RAG                 |


---

"""


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
    try:
        from Deep_Learning.visuals.deep_learning_path import (   # ← match your exact folder casing
            DL_PATH_HTML,
            DL_PATH_HEIGHT,
        )
        visual_html   = DL_PATH_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = DL_PATH_HEIGHT
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
        "complexity":    None, # COMPLEXITY,
        "operations":    None, # OPERATIONS,
    }