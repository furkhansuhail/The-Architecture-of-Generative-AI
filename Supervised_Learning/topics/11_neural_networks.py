"""
Neural Networks — Universal Approximators
==========================================

A neural network is a computational graph of simple parameterised functions,
chained together to form a highly flexible, learnable mapping from inputs to
outputs. Each layer transforms its input into a new representation, and across
many layers the network learns to represent increasingly abstract features of
the data — edges, then shapes, then objects; or characters, then words, then
grammar; or raw measurements, then interactions, then decisions.

Neural networks are not magic. They are composed of nothing more than matrix
multiplications and elementwise nonlinearities, trained by gradient descent
via the chain rule. What makes them powerful is the combination of:
    (1) sufficient depth to represent complex functions
    (2) sufficient data to guide gradient descent toward useful solutions
    (3) effective regularisation to prevent memorisation
    (4) efficient hardware to make the computation tractable

They are the bridge from the classical algorithms in this series to the deep
learning models — CNNs, Transformers, LLMs — that power modern AI.

"""

import base64
import os
import re
import textwrap

DISPLAY_NAME = "11 · Neural Networks (Supervised)"
ICON         = "🧠"
SUBTITLE     = "Universal approximators — the bridge from supervised learning to deep learning"
TOPIC_NAME   = "Neural Networks (Supervised)"
VISUAL_HTML  = ""

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is a Neural Network?

A feedforward neural network is a function F: ℝᵖ → ℝᵏ built by composing
L layers, each of the form:

                    aˡ = σ( Wˡ aˡ⁻¹ + bˡ )

Where:
    aˡ⁻¹:   input to layer l  (the previous layer's output, or the raw features)
    Wˡ:     weight matrix of layer l  (learned during training)
    bˡ:     bias vector of layer l   (learned during training)
    σ(·):   activation function     (applied elementwise, introduces nonlinearity)
    aˡ:     output of layer l (the "activations" — also called hidden representations)

The full network computes:
    F(x) = σ_L( W_L · σ_{L-1}( W_{L-1} · ... σ_1( W_1 · x + b_1 ) ... + b_{L-1} ) + b_L )

This is just repeated matrix multiplication and function application.
The power comes from what the network learns to represent in each layer.

    Things that exist inside the model (learned during training):
        - All weight matrices W¹, W², ..., W_L
        - All bias vectors b¹, b², ..., b_L
        - For a depth-3 network with widths [p, 64, 32, k]:
          Total parameters = p×64 + 64 + 64×32 + 32 + 32×k + k

    Things you control before training (hyperparameters):
        - Number of layers (depth)
        - Number of neurons per layer (width)
        - Activation function (ReLU, sigmoid, tanh, GELU...)
        - Loss function (MSE, cross-entropy...)
        - Optimiser (SGD, Adam, AdamW...)
        - Learning rate and schedule
        - Batch size
        - Regularisation (dropout rate, L2 weight decay)


### Are Neurons Like Logic Gates? The Historical Origin

Yes — and this is exactly where neural networks came from.

In 1943, McCulloch and Pitts showed that a single neuron modelled as a binary
threshold function — outputting 1 if the weighted sum of inputs exceeds a
threshold, 0 otherwise — is equivalent to a logic gate:

    AND gate:  fire if ALL inputs are active    (high threshold, all weights = 1)
    OR gate:   fire if ANY input is active      (low threshold, all weights = 1)
    NOT gate:  fire if input is INACTIVE        (negative weight, threshold = -0.5)

A network of such neurons can compute ANY Boolean function — it is Turing complete.

    # =======================================================================================# 
    **Historical Neuron Types — Binary to Continuous:**

    McCulloch-Pitts (1943): binary threshold
    ─────────────────────────────────────────────────────────────────
    output = 1  if  Σ wᵢ xᵢ ≥ θ
             0  otherwise

    This IS a logic gate. The limitation: binary, not differentiable.
    You cannot compute gradients through a step function → cannot use
    gradient descent to learn the weights.

    Perceptron (Rosenblatt, 1958): same binary output, learnable weights
    ─────────────────────────────────────────────────────────────────
    output = sign( Σ wᵢ xᵢ + b )
    Update rule: w ← w + α (y - ŷ) x  (error-driven, not gradient-based)

    Modern neuron: continuous activation function (sigmoid, ReLU, tanh)
    ─────────────────────────────────────────────────────────────────
    output = σ( Σ wᵢ xᵢ + b )    where σ is smooth and differentiable

    The move from binary thresholds to continuous activations is what
    enabled backpropagation (1986) and made modern deep learning possible.

    MODERN NETWORKS:    not binary logic gates
    BIOLOGICAL NEURONS: not mathematical models
    MATHEMATICAL NEURONS: differentiable parameterised functions
    The analogy to biology is loose — the maths is the reality.
    # =======================================================================================# 


### Neural Networks in the ERM Framework

    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │  Hypothesis class:  H = { all functions representable by     │
    │                           a network with L layers and given  │
    │                           widths and activations }           │
    │                     (enormously expressive — Universal       │
    │                      Approximation Theorem: any continuous   │
    │                      function on a compact set can be        │
    │                      approximated by a 2-layer network)      │
    │                                                              │
    │  Loss function:     MSE (regression), BCE / CE (classif.)    │
    │                     Any differentiable loss is compatible    │
    │                                                              │
    │  Optimiser:         Gradient descent (SGD / Adam) via        │
    │                     BACKPROPAGATION — the chain rule applied │
    │                     efficiently through the computation graph│
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

The ERM comparison across all 9 modules:
    Linear Regression:    MSE,  gradient descent, closed-form optimal
    Logistic Regression:  BCE,  gradient descent, convex (global optimum)
    SVM:                  Hinge, QP solver, global optimum (convex)
    Decision Tree:        Gini, greedy split, local optimum only
    Random Forest:        0-1,  averaging B independent trees
    Gradient Boosting:    any,  gradient descent in function space
    KNN:                  none, memorisation
    Naive Bayes:          NLL,  MLE counting
    Neural Networks:      any,  gradient descent via backprop, NON-CONVEX


### The Universal Approximation Theorem

**Theorem (Cybenko 1989, Hornik 1991):** A feedforward network with a single
hidden layer containing a sufficient number of neurons can approximate any
continuous function on a compact subset of ℝᵖ to arbitrary precision.

    ANY function → ONE hidden layer → enough neurons → approximated

This is profoundly powerful. It means the hypothesis class of neural networks
is "everything" — given enough neurons, there is always a network that fits.

**The catch:** the theorem guarantees existence, not learnability.
A network wide enough to approximate any function might:
    - Require exponentially many neurons in the shallow case
    - Not be findable by gradient descent in reasonable time
    - Overfit the training data badly

Deep networks (many layers) are more efficient: they can represent exponentially
more functions than shallow networks with the same number of parameters, because
each layer builds on the compressed representations of the previous layer.


### The Inductive Bias of Neural Networks

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Neural networks encode relatively weak inductive bias:     │
    │                                                             │
    │  1. COMPOSITIONALITY — complex functions are built from     │
    │     simpler functions. Later layers build on earlier ones.  │
    │                                                             │
    │  2. SMOOTH INTERPOLATION — ReLU/GELU networks produce       │
    │     piecewise-linear / smooth functions (not jumpy).        │
    │                                                             │
    │  3. DISTRIBUTED REPRESENTATIONS — information is encoded    │
    │     across many neurons, not localised to one.              │
    │                                                             │
    │  Specialised architectures add stronger biases:             │
    │    CNN: TRANSLATION INVARIANCE (images)                     │
    │    RNN: SEQUENTIAL STRUCTURE (sequences)                    │
    │    Transformer: PAIRWISE ATTENTION (all positions)          │
    │    GNN: GRAPH TOPOLOGY (relational data)                    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘


---


### Part 1: The Forward Pass — How a Network Computes


### Layer-by-Layer Computation:

Start with raw input x (shape: p).
Apply each layer in sequence to produce the final output ŷ.

    INPUT  x:    shape (p,)           ← raw features
    LAYER 1:     z¹ = W¹x + b¹        ← linear (matrix multiply + bias)
                 a¹ = σ(z¹)           ← activation (nonlinearity)
    LAYER 2:     z² = W²a¹ + b²       ← linear again
                 a² = σ(z²)           ← activation again
    ...
    OUTPUT:      ŷ = W_L a_{L-1} + b_L  ← final linear layer
                 (+ softmax for multiclass, or sigmoid for binary)


### Why Nonlinear Activations Are Essential:

Without activations: σ(x) = x (identity)
    Layer 2 output: W²(W¹x + b¹) + b² = (W²W¹)x + (W²b¹ + b²) = W'x + b'
    → A deep linear network is exactly equivalent to a single linear layer
    → Can only represent linear functions regardless of depth

With nonlinear activations:
    Each layer can represent complex, nonlinear transformations
    Composing nonlinear layers creates exponentially richer function classes


    # =======================================================================================# 
    **Diagram 1 — Forward Pass Through a 3-Layer Network:**

    Input: x = [x₁, x₂]   (2 features)

    Layer 1 (hidden, 3 neurons):
    ─────────────────────────────────────────────────────────────
    z¹₁ = w¹₁₁x₁ + w¹₁₂x₂ + b¹₁    a¹₁ = ReLU(z¹₁)
    z¹₂ = w¹₂₁x₁ + w¹₂₂x₂ + b¹₂    a¹₂ = ReLU(z¹₂)
    z¹₃ = w¹₃₁x₁ + w¹₃₂x₂ + b¹₃    a¹₃ = ReLU(z¹₃)

    Layer 2 (hidden, 2 neurons):
    ─────────────────────────────────────────────────────────────
    z²₁ = w²₁₁a¹₁ + w²₁₂a¹₂ + w²₁₃a¹₃ + b²₁    a²₁ = ReLU(z²₁)
    z²₂ = w²₂₁a¹₁ + w²₂₂a¹₂ + w²₂₃a¹₃ + b²₂    a²₂ = ReLU(z²₂)

    Output (1 neuron, binary classification):
    ─────────────────────────────────────────────────────────────
    ŷ = σ( w³₁a²₁ + w³₂a²₂ + b³ )   ← sigmoid → probability in [0,1]

    ALL OF THIS is just matrix multiplications and elementwise functions.
    In matrix form:  a¹ = ReLU(W¹x + b¹),  a² = ReLU(W²a¹ + b²),  ŷ = σ(W³a² + b³)
    # =======================================================================================# 


### What Do the Hidden Layers Learn?

Each layer learns a transformation of the previous layer's representation.
For a network trained on images:
    Layer 1: detects edges (oriented lines at different angles)
    Layer 2: combines edges into shapes (corners, curves, textures)
    Layer 3: combines shapes into parts (eyes, wheels, branches)
    Layer 4: combines parts into objects (faces, cars, trees)

For a network trained on tabular data:
    Layer 1: detects individual feature thresholds and combinations
    Layer 2: detects interactions between features
    Layer 3: detects patterns in those interactions
    Final:   produces the prediction

The weights Wˡ are not hand-designed — they are learned from data to minimise
the loss. The network discovers which combinations of inputs are useful,
not us.


---


### Part 2: Activation Functions


### Sigmoid:   σ(z) = 1 / (1 + e⁻ᶻ)

    Output range: (0, 1)  — good for binary probability outputs
    Gradient:     σ'(z) = σ(z)(1−σ(z))  — maximum 0.25 at z=0
    Problems:     VANISHING GRADIENT — for large |z|, gradient ≈ 0
                  Gradients are multiplied across layers; near-zero gradients
                  make early layers learn extremely slowly (or not at all).
                  Also not zero-centred → slow convergence.
    Use now:      output layer only (binary classification)


### Tanh:      tanh(z) = (e^z − e^{-z}) / (e^z + e^{-z})

    Output range: (−1, 1)  — zero-centred (better than sigmoid)
    Gradient:     tanh'(z) = 1 − tanh²(z)  — maximum 1.0 at z=0
    Problems:     Still saturates for large |z|, still vanishing gradient
    Use now:      RNNs / LSTMs, some output layers


### ReLU:      ReLU(z) = max(0, z)

    Output range: [0, ∞)
    Gradient:     1 if z > 0,   0 if z ≤ 0   (a step function)
    Advantages:   No saturation for positive z → gradients flow freely
                  Computationally trivial (max operation)
                  Biologically plausible (sparse activation)
    Problems:     DYING ReLU — if z < 0 for all training examples, the
                  neuron always outputs 0, gradient is always 0 → never learns.
                  Not differentiable at z=0 (in practice, use subgradient).
    Use now:      Default for most hidden layers in feedforward networks


### Leaky ReLU:    f(z) = max(αz, z)  where α ≈ 0.01

    Gradient:     1 if z > 0,   α if z ≤ 0
    Fix:          dying ReLU — negative inputs still get a small gradient
    Use when:     Dying ReLU is a problem (large networks, small batch size)


### GELU:      Gaussian Error Linear Unit — approximate: z · σ(1.702z)

    Properties:   Smooth everywhere, approximately like ReLU for large z,
                  decays smoothly to 0 for large negative z.
    Advantage:    Works better than ReLU in Transformers and large models.
    Use now:      Default in BERT, GPT, most modern Transformer models.


    # =======================================================================================# 
    **Diagram 2 — Activation Functions Compared:**

     z →          -3      -1       0       1       3
    ─────────────────────────────────────────────────────────────
    Sigmoid:      0.047   0.269   0.500   0.731   0.953
    Tanh:        -0.995  -0.762   0.000   0.762   0.995
    ReLU:         0.000   0.000   0.000   1.000   3.000
    Leaky ReLU:  -0.030  -0.010   0.000   1.000   3.000
    GELU:        -0.004  -0.159   0.000   0.841   2.996
    ─────────────────────────────────────────────────────────────
    Sigmoid, Tanh: saturate at extremes → vanishing gradient
    ReLU:          zero for negative inputs → dying ReLU
    GELU:          smooth, no saturation for positive, smooth decay for negative
    # =======================================================================================# 


---


### Part 3: The Loss Function and What We're Optimising


### Choosing the Right Loss:

    ──────────────────────────────────────────────────────────────────────────────
    Task                Output activation    Loss function
    ──────────────────────────────────────────────────────────────────────────────
    Binary classif.     Sigmoid (output)     Binary cross-entropy BCE
    Multiclass          Softmax (output)     Categorical cross-entropy CE
    Regression          Linear (no activ.)   MSE or MAE or Huber
    Multilabel          Sigmoid per output   BCE per output (summed)
    Ranking             Sigmoid / linear     Pairwise / listwise ranking loss
    ──────────────────────────────────────────────────────────────────────────────

### Cross-Entropy Loss (Classification):

For binary classification with label y ∈ {0,1} and predicted probability ŷ:
    BCE(y, ŷ) = −[y · log(ŷ) + (1−y) · log(1−ŷ)]

For K-class classification with one-hot label yᵢ ∈ {0,1}^K and softmax output:
    CE(y, ŷ) = −Σₖ yₖ · log(ŷₖ)

These losses are maximally penalised when the model is confident and wrong
(predicts 0.99 for the wrong class), and minimally penalised when correct.
They are differentiable everywhere needed and connect to maximum likelihood
estimation (minimising CE = maximising log-likelihood under the model).


---


### Part 4: Backpropagation — How Weights Are Changed


### The Core Idea — Gradient Descent:

Training minimises the average loss over the training set:
    L(W) = (1/n) Σᵢ loss(yᵢ, F(xᵢ; W))

We update all weights in the direction of steepest descent:
    W ← W − α · ∇_W L

The challenge: how do we compute ∇_W L efficiently when W includes millions
of weights spread across many layers?

Answer: **Backpropagation** — the chain rule applied recursively from the
output layer back to the input layer. It computes all gradients in O(2 × forward pass),
which is remarkably efficient.


### The Chain Rule — The Foundation of Backpropagation:

If f = g(h(x)), then:    df/dx = dg/dh · dh/dx

For a network with two layers:
    z¹ = W¹x + b¹      a¹ = σ(z¹)
    z² = W²a¹ + b²     ŷ = σ(z²)
    L  = loss(y, ŷ)

To compute ∂L/∂W¹ (how the first layer's weights affect the loss):

    ∂L/∂W¹ = ∂L/∂ŷ · ∂ŷ/∂z² · ∂z²/∂a¹ · ∂a¹/∂z¹ · ∂z¹/∂W¹

Each term is easy to compute individually. Backpropagation computes them
efficiently by caching intermediate values during the forward pass.


    # =======================================================================================# 
    **Diagram 3 — Forward Pass → Loss → Backward Pass:**

    FORWARD PASS (left to right):
    ─────────────────────────────────────────────────────────────────────────────
    x  →  [z¹=W¹x+b¹]  →  [a¹=σ(z¹)]  →  [z²=W²a¹+b²]  →  [ŷ=σ(z²)]  →  L
    ─────────────────────────────────────────────────────────────────────────────
    Cache: x, z¹, a¹, z², ŷ  (needed for backward pass)

    BACKWARD PASS (right to left, chain rule):
    ─────────────────────────────────────────────────────────────────────────────
    ∂L/∂ŷ              = loss'(y, ŷ)              [easy: BCE derivative]
    ∂L/∂z²             = ∂L/∂ŷ × σ'(z²)           [chain rule through activation]
    ∂L/∂W²             = ∂L/∂z² × (a¹)ᵀ           [chain rule through matrix mult]
    ∂L/∂b²             = ∂L/∂z²                    [gradient of bias]
    ∂L/∂a¹             = (W²)ᵀ × ∂L/∂z²           [propagate gradient back]
    ∂L/∂z¹             = ∂L/∂a¹ × σ'(z¹)          [chain rule through activation]
    ∂L/∂W¹             = ∂L/∂z¹ × xᵀ              [gradient of first layer weights]
    ∂L/∂b¹             = ∂L/∂z¹                    [gradient of first bias]
    ─────────────────────────────────────────────────────────────────────────────
    ALL gradients computed in one backward pass = one forward pass in cost.

    KEY INSIGHT: Each layer's gradient depends only on the gradient from the
    layer above (the "error signal" flowing backward) and the cached activations.
    # =======================================================================================# 


### The Weight Update Step:

After computing gradients, weights are updated by an optimiser.
Three common choices:

**Vanilla SGD:**
    W ← W − α · ∂L/∂W

**SGD with Momentum:**
    v ← β·v + (1−β)·∂L/∂W        (momentum accumulates gradient history)
    W ← W − α·v
    β = 0.9 is the standard choice. Momentum smooths oscillations and
    accelerates movement in consistent gradient directions.

**Adam (Adaptive Moment Estimation):**
    m ← β₁·m + (1−β₁)·∂L/∂W      (1st moment = exponential moving average of gradient)
    v ← β₂·v + (1−β₂)·(∂L/∂W)²   (2nd moment = exponential moving average of gradient²)
    m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)  (bias correction for early steps)
    W ← W − α · m̂ / (√v̂ + ε)
    Defaults: β₁=0.9, β₂=0.999, ε=1e-8, α=0.001

    Adam adapts the learning rate per-parameter: parameters with consistently
    large gradients get a smaller effective learning rate; parameters with
    small or noisy gradients get a larger effective rate.


### Why the Loss Surface is Non-Convex:

For linear regression, the loss surface is a bowl — one global minimum.
For neural networks with nonlinear activations, the loss surface has:
    - Many local minima (but often similar loss to the global minimum)
    - Saddle points (gradient = 0 but not a minimum)
    - Flat regions (tiny gradients → slow learning)
    - Sharp vs flat minima (flat minima generalise better — implicit regularisation)

Neural networks are trained in the non-convex regime. We cannot guarantee
finding the global optimum, but in practice SGD/Adam find solutions that
generalise well. The reasons are an active area of research.


---


### Part 5: The Training Loop — Epochs, Batches, and Iterations


### Stochastic Gradient Descent (SGD):

Computing the gradient over ALL n training examples is expensive: O(n·p) per step.
Instead, at each step we sample a random mini-batch of B examples and compute
the gradient on just those B examples — an unbiased estimate of the true gradient.

    Mini-batch gradient descent (the standard):
        For each epoch:
            Shuffle training data
            For each batch of B examples:
                Compute forward pass on B examples
                Compute average loss over the B examples
                Backpropagate to get gradients
                Update weights with SGD / Adam

    Batch size B:
        B = 1:    pure SGD — noisy, slow per epoch, many updates
        B = n:    full batch GD — exact gradient, slow per update
        B = 32–256: mini-batch SGD — the standard  balance

    Epoch: one complete pass through the entire training set.
    Iteration: one weight update (on one mini-batch).


### Why Noise in SGD Helps:

Noisy gradient estimates from small batches act as regularisation:
    1. Escapes sharp local minima (noisy steps jump out)
    2. Finds flatter minima that generalise better
    3. Acts like data augmentation (different batch = different function estimate)

Large batches compute more accurate gradients but often find sharper minima
that generalise poorly. This is called the "generalization gap" of large batches.


---


### Part 6: Training, Validation, and Test — The Three Datasets


This is one of the most important concepts in all of machine learning.
Each dataset serves a completely different purpose.

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  TRAINING SET (typically 60–70% of data):                   │
    │    What happens: gradient descent runs on this data.        │
    │    The loss on training data is minimised directly.         │
    │    The model SEES this data and learns from it.             │
    │    Training loss tells you: is the model learning at all?   │
    │                                                             │
    │  VALIDATION SET (typically 10–20% of data):                 │
    │    What happens: no gradient updates. Forward pass only.    │
    │    Used to: select hyperparameters (depth, width, LR,       │
    │    dropout), detect overfitting, choose early-stopping epoch│
    │    The model does NOT see this during training.             │
    │    Validation loss tells you: is the model generalising?    │
    │                                                             │
    │  TEST SET (typically 10–20% of data):                       │
    │    What happens: evaluated ONCE, at the very end.           │
    │    Never used for any decisions during development.         │
    │    Provides an unbiased estimate of real-world performance. │
    │    Test loss tells you: how will it perform in deployment?  │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘


### Why You Cannot Use the Test Set for Decisions:

If you use the test set to decide which model to keep, you are performing
implicit optimisation over the test set — you are selecting the model that
happened to perform best on test, which inflates the apparent performance.

The validation set is the "held-out" data you use during development.
The test set is the "locked vault" you open only once.

    Anti-pattern: "I tried 30 models and picked the one with best test accuracy"
    → The test accuracy is now the best of 30 samples from a distribution
    → Expected to be optimistic by several percentage points
    → The real-world performance will be worse

    Correct pattern:
    → Use validation to pick the model
    → Use test ONCE to report the final result


### The Loss Curve — Diagnosing Training:


    # =======================================================================================# 
    **Diagram 4 — Loss Curves and Their Interpretation:**

    UNDERFITTING (high bias):              HEALTHY TRAINING:
    loss ↑                                 loss ↑
      │ ─────────────── train              │╲ train
      │ ─────────────── val                │  ╲_______________
      │                                    │   ╲ val
      │  both losses high and flat         │    ╲_____________
      └──────────────────→ epoch           └──────────────────→ epoch
    Fix: more capacity, more epochs        val loss converges and stays low

    OVERFITTING (high variance):           EARLY STOPPING:
    loss ↑                                 loss ↑
      │╲ train                             │╲ train
      │  ╲__________                       │  ╲__________
      │             ╲ val (rises!)         │   ╲ val
      │              ╲────                 │    ╲───* stop here
      └──────────────────→ epoch           └──────────────────→ epoch
    Fix: dropout, weight decay, less       Save weights at val minimum
    capacity, more data                    Use those weights for test

    THE GAP between train and val loss:
      Small gap:  good generalisation
      Large gap:  overfitting
    # =======================================================================================# 


### Overfitting — Why It Happens and How to Fix It:

Overfitting = the model fits the training data so well that it memorises
noise specific to the training set, rather than learning the general pattern.
Performance on unseen data is worse than on training data.

Signs of overfitting:
    - Training loss continues falling, validation loss starts rising
    - Training accuracy much higher than validation accuracy
    - Model predicts training examples perfectly but fails on new ones

**Regularisation techniques:**

    Dropout (Srivastava et al., 2014):
        During training: randomly zero out each neuron with probability p (typically 0.2–0.5)
        During inference: use all neurons, scale outputs by (1−p)
        Effect: forces the network to learn redundant representations —
                no single neuron can be relied upon, so all neurons must
                contribute meaningfully. Approximates training an ensemble.

    L2 Weight Decay:
        Add λ/2 · ||W||² to the loss.
        Gradient: ∂L/∂W += λ·W  → W ← W · (1−α·λ) − α · ∂L/∂W
        Effect: shrinks all weights toward zero. Large weights are penalised.
        Prevents any single pathway from dominating.

    Batch Normalisation (Ioffe & Szegedy, 2015):
        Normalise each mini-batch to zero mean, unit variance, then rescale.
        Effect: reduces internal covariate shift, allows higher learning rates,
                acts as regularisation.

    Early Stopping:
        Monitor validation loss during training. Stop when it stops improving.
        Save the weights at the best validation epoch.
        Simple and highly effective — no additional hyperparameter.


---


### Part 7: How Patterns Are Found Between Neurons


### What a Weight Actually Represents:

A weight wᵢⱼ between neuron i (in layer l−1) and neuron j (in layer l) encodes
how much neuron i's activation should influence neuron j's input.

    wᵢⱼ > 0:   neuron i EXCITES neuron j (more i → more j)
    wᵢⱼ < 0:   neuron i INHIBITS neuron j (more i → less j)
    wᵢⱼ ≈ 0:   neuron i is irrelevant to neuron j

After training, each neuron in a hidden layer responds to a particular pattern
in the input:
    - A neuron with large positive weights on features [tall, long_neck]
      and large negative weight on [small] → activates for giraffes
    - A neuron with large weight on [curve, dark] → activates for edges
    - A neuron connected to many "urgent", "free", "click" features → spam


### What Gradient Descent Discovers:

Gradient descent does not know in advance that "tall + long neck = giraffe".
It discovers which patterns are useful by observing:
    - When this neuron fired on training examples with label "giraffe",
      the loss was lower than when it didn't fire
    - Therefore: increase the weights connecting the "tall" and "long_neck"
      neurons to this neuron

This is the fundamental learning process:
    1. Make predictions with current weights
    2. Measure how wrong they are (loss)
    3. Backpropagate: for each weight, compute how much changing it would
       reduce the loss (the gradient)
    4. Adjust every weight by a small step in the gradient direction
    5. Repeat millions of times

The patterns emerge because features that correlate with the correct label
receive consistently positive gradient signal — their weights grow.
Features that are irrelevant or misleading receive inconsistent gradient signal
— their weights stay small or shrink.


    # =======================================================================================# 
    **Diagram 5 — How a Neuron Learns to Detect a Pattern:**

    Training example 1: cat (label=1)
    ─────────────────────────────────────────────────────────────────
    Input: [whiskers=1, tail=1, bark=0]
    Neuron output: 0.2 (too low for cat)
    Gradient: increase weights for whiskers and tail → cat signal
              decrease weight for bark (not active, but check sign)

    Training example 2: dog (label=0)
    ─────────────────────────────────────────────────────────────────
    Input: [whiskers=0, tail=1, bark=1]
    Neuron output: 0.6 (too high — neuron fires for dog)
    Gradient: decrease weight for bark → bark should suppress firing
              tail weight: conflicting signal (present in both) → stays small

    After many examples:
    ─────────────────────────────────────────────────────────────────
    w_whiskers → large positive  (whiskers → probably cat)
    w_bark     → large negative  (bark → probably NOT cat)
    w_tail     → near zero       (tail appears in both — not discriminative)

    The neuron has learned to detect "cat vs dog" patterns
    purely from gradient descent — no one programmed this.
    # =======================================================================================# 


---


### Part 8: What Happens During Testing


### The Testing Phase — No Learning:

When a trained model is evaluated on the test set:
    - All weights are FROZEN (no gradient computation, no weight updates)
    - Only the forward pass runs: x → layer 1 → layer 2 → ... → ŷ
    - The prediction ŷ is compared to the true label y to compute metrics

    model.eval()  in PyTorch  — disables dropout, uses running statistics in BatchNorm
    with torch.no_grad():     — disables gradient computation (saves memory and time)

Testing is just inference: compute the output of the frozen network.


### Dropout During Training vs Testing:

During TRAINING:
    Each neuron is randomly dropped with probability p = 0.5
    Different neurons are dropped each forward pass
    Network cannot rely on any one neuron → learns robust features

During TESTING:
    All neurons are active (no dropout applied)
    But outputs are scaled by (1−p) to keep expected values the same
    This is called "inverted dropout" — the most common implementation

    Inverted dropout (during training):
        mask = (random_uniform(shape) > p)   ← 1 with prob (1-p), 0 with prob p
        output = (input × mask) / (1 − p)   ← scale UP during training
    At test: just use output = input as-is (no masking, no scaling)


### Metrics for Classification vs Regression:

    Classification:   Accuracy, Precision, Recall, F1, AUC-ROC, Log-loss
    Regression:       MSE, RMSE, MAE, R², Huber loss
    Calibration:      Brier score, reliability diagram


---


### Part 9: The Architecture — Design Decisions


### Depth vs Width:

    WIDER network:   more neurons per layer → can represent more functions per layer
    DEEPER network:  more layers → can represent compositional functions efficiently

    Deep networks are better at:
        - Learning hierarchical representations
        - Representing exponentially more functions with same parameter count
        - Image recognition, natural language understanding

    Wide networks are better at:
        - Avoiding vanishing gradients (fewer multiplications)
        - Problems with less hierarchical structure
        - When depth causes training instability

### The Depth-Generalisation Relationship:

Deeper networks tend to find solutions with better generalisation properties,
even though they have more parameters. This is the "deep learning puzzle":
    - Classical statistics: more parameters → worse generalisation
    - Deep networks: more depth → often better generalisation

Reasons: weight sharing (CNNs), residual connections, implicit regularisation
from SGD, and the hierarchical nature of real-world data.


### Architecture Evolution:

    Fully Connected (MLP / Dense):   all-to-all connections
                                     Good for tabular data
    Convolutional (CNN):              weight sharing over spatial locations
                                     Good for images
    Recurrent (RNN/LSTM/GRU):         sequential processing with state
                                     Good for sequences, time series
    Transformer:                      self-attention, no sequential constraint
                                     Current state-of-the-art for text, images
    Graph Neural Network (GNN):       message passing on graph topology
                                     Good for molecules, social networks


---


### Part 10: Where Neural Networks Sit in the Series


    Perceptron → Logistic Regression → SVM → Decision Tree
         → Random Forest → Gradient Boosting → KNN → Naive Bayes → Neural Networks

    All previous models: fixed, manually designed feature space
                         Learning only adjusts the final combination
    Neural Networks:     learns the feature REPRESENTATION itself
                         Each layer transforms the feature space

    This is the crucial shift: representation learning.

    The models above this layer in modern AI:
        CNNs:         spatial representation learning (images)
        LSTMs:        sequential representation learning (text, audio)
        Transformers: contextual representation learning (text, images, code)
        LLMs:         world knowledge representation (GPT, Claude, Gemini)

    """

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ──────────────────────────────────────────────────────────────────────────
    Operation                   Complexity              Notes
    ──────────────────────────────────────────────────────────────────────────
    Forward pass (1 example)    O(Σ_l  n_l × n_{l-1})  matrix multiplications
    Backward pass               O(2 × forward)           chain rule through all layers
    One training step (B batch) O(B × forward + backward) per parameter
    Total training              O(epochs × n × forward)  n = training set size
    Inference (prediction)      O(forward pass only)     no gradient needed
    Parameters                  O(Σ_l  n_l × n_{l-1})   grows with depth × width
    ──────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Neural Network from Scratch": {
        "description": "Forward pass, backpropagation, and SGD/Adam — built with NumPy only",
        "runnable": True,
        "code": '''
"""
================================================================================
NEURAL NETWORK FROM SCRATCH — NUMPY ONLY
================================================================================

We build a complete feedforward neural network implementing:
    - Forward pass:   x → hidden layers → output
    - Loss function:  binary cross-entropy
    - Backpropagation: chain rule computing all gradients
    - Weight updates: SGD and Adam optimisers

No PyTorch, no TensorFlow, no sklearn — pure NumPy matrix operations.

Architecture: [p] → [hidden₁] → [hidden₂] → [1]
              Input   ReLU        ReLU       Sigmoid output
================================================================================
"""

import numpy as np


# =============================================================================
# ACTIVATION FUNCTIONS (with derivatives for backprop)
# =============================================================================

def sigmoid(z):
    """σ(z) = 1 / (1 + exp(-z)),  clipped for numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(z):
    """σ'(z) = σ(z)(1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """ReLU(z) = max(0, z)"""
    return np.maximum(0, z)

def relu_grad(z):
    """ReLU'(z) = 1 if z > 0, else 0"""
    return (z > 0).astype(float)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def bce_loss(y_true, y_pred):
    """
    Binary Cross-Entropy: L = -mean[y*log(ŷ) + (1-y)*log(1-ŷ)]
    Clip predictions to avoid log(0).
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_grad(y_true, y_pred):
    """
    ∂BCE/∂ŷ = (-y/ŷ + (1-y)/(1-ŷ)) / n
    Combined with sigmoid at output: ∂L/∂z_out = (ŷ - y) / n
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    n = len(y_true)
    return (y_pred - y_true) / n


# =============================================================================
# THE NEURAL NETWORK
# =============================================================================

class NeuralNetwork:
    """
    A 3-layer feedforward network for binary classification.

        Input → Dense(hidden1, ReLU) → Dense(hidden2, ReLU) → Dense(1, Sigmoid)

    Parameters (all weights and biases):
        W1, b1:  first hidden layer
        W2, b2:  second hidden layer
        W3, b3:  output layer

    Hyperparameters (chosen by you, not learned):
        hidden1, hidden2:  layer widths
        lr:                learning rate
        optimiser:         'sgd' or 'adam'
    """

    def __init__(self, input_dim, hidden1=32, hidden2=16,
                 lr=0.01, optimiser="adam", weight_decay=0.0):
        self.lr           = lr
        self.optimiser    = optimiser
        self.weight_decay = weight_decay

        # ── Weight initialisation (He initialisation for ReLU layers) ────────
        # He: W ~ N(0, sqrt(2/fan_in))  — prevents vanishing/exploding gradient
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden1)
        scale3 = np.sqrt(2.0 / hidden2)

        self.W1 = np.random.randn(input_dim, hidden1) * scale1
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * scale2
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, 1) * scale3
        self.b3 = np.zeros((1, 1))

        # ── Adam state (first and second moment estimates) ───────────────────
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.t = 0   # Adam timestep

        self.train_losses = []
        self.val_losses   = []

    def _params(self):
        return {"W1":self.W1,"b1":self.b1,"W2":self.W2,
                "b2":self.b2,"W3":self.W3,"b3":self.b3}

    def _set_param(self, key, val):
        setattr(self, key, val)

    # ── FORWARD PASS ──────────────────────────────────────────────────────────

    def forward(self, X):
        """
        Forward pass: compute layer activations and cache for backprop.

        For each layer:
            z = W·a_prev + b       (linear transformation)
            a = activation(z)       (nonlinear activation)

        Returns: final prediction ŷ (probabilities in [0,1])
        """
        self._cache = {}

        # Layer 1: Input → Hidden1  (ReLU)
        self._cache["A0"] = X
        self._cache["Z1"] = X @ self.W1 + self.b1      # (n, hidden1)
        self._cache["A1"] = relu(self._cache["Z1"])     # (n, hidden1)

        # Layer 2: Hidden1 → Hidden2  (ReLU)
        self._cache["Z2"] = self._cache["A1"] @ self.W2 + self.b2   # (n, hidden2)
        self._cache["A2"] = relu(self._cache["Z2"])                   # (n, hidden2)

        # Layer 3: Hidden2 → Output  (Sigmoid)
        self._cache["Z3"] = self._cache["A2"] @ self.W3 + self.b3   # (n, 1)
        A3                 = sigmoid(self._cache["Z3"])               # (n, 1)

        return A3

    # ── BACKWARD PASS ─────────────────────────────────────────────────────────

    def backward(self, y):
        """
        Backward pass: compute gradients via chain rule.

        We propagate the error signal from the output back through each layer.
        At each layer, we compute:
            dL/dW = dL/dz × (a_prev).T   (gradient w.r.t. weights)
            dL/db = dL/dz                  (gradient w.r.t. bias)
            dL/da_prev = W.T × dL/dz       (gradient to propagate further back)
        """
        cache = self._cache
        n     = y.shape[0]
        y_col = y.reshape(-1, 1)

        # ── Output layer gradient ──────────────────────────────────────────
        # For BCE + sigmoid combined: dL/dz3 = (ŷ - y) / n
        # (This is the "miraculous cancellation" from logistic regression)
        A3        = sigmoid(cache["Z3"])
        dZ3       = (A3 - y_col) / n           # (n, 1)
        dW3       = cache["A2"].T @ dZ3          # (hidden2, 1)
        db3       = dZ3.sum(axis=0, keepdims=True)  # (1, 1)
        dA2       = dZ3 @ self.W3.T              # (n, hidden2)

        # ── Layer 2 gradient ───────────────────────────────────────────────
        dZ2       = dA2 * relu_grad(cache["Z2"])   # chain rule through ReLU
        dW2       = cache["A1"].T @ dZ2             # (hidden1, hidden2)
        db2       = dZ2.sum(axis=0, keepdims=True)  # (1, hidden2)
        dA1       = dZ2 @ self.W2.T                 # (n, hidden1)

        # ── Layer 1 gradient ───────────────────────────────────────────────
        dZ1       = dA1 * relu_grad(cache["Z1"])   # chain rule through ReLU
        dW1       = cache["A0"].T @ dZ1             # (p, hidden1)
        db1       = dZ1.sum(axis=0, keepdims=True)  # (1, hidden1)

        # Collect gradients and apply weight decay (L2 regularisation)
        grads = {"W1": dW1 + self.weight_decay * self.W1,
                 "b1": db1,
                 "W2": dW2 + self.weight_decay * self.W2,
                 "b2": db2,
                 "W3": dW3 + self.weight_decay * self.W3,
                 "b3": db3}
        return grads

    # ── WEIGHT UPDATE ─────────────────────────────────────────────────────────

    def _update_sgd(self, grads):
        """Vanilla SGD: W ← W − α · ∇W"""
        for key in ["W1","b1","W2","b2","W3","b3"]:
            self._set_param(key, getattr(self, key) - self.lr * grads[key])

    def _update_adam(self, grads):
        """
        Adam: adapts learning rate per parameter using first and second moments.

        m_t = β₁ m_{t-1} + (1-β₁) g_t         (running mean of gradient)
        v_t = β₂ v_{t-1} + (1-β₂) g_t²         (running mean of gradient²)
        m̂_t = m_t / (1 - β₁ᵗ)                  (bias correction)
        v̂_t = v_t / (1 - β₂ᵗ)                  (bias correction)
        W_t = W_{t-1} - α · m̂_t / (√v̂_t + ε)
        """
        β1, β2, ε = 0.9, 0.999, 1e-8
        self.t += 1
        for key in ["W1","b1","W2","b2","W3","b3"]:
            g            = grads[key]
            self.m[key]  = β1 * self.m[key] + (1 - β1) * g
            self.v[key]  = β2 * self.v[key] + (1 - β2) * g ** 2
            m_hat        = self.m[key] / (1 - β1 ** self.t)
            v_hat        = self.v[key] / (1 - β2 ** self.t)
            self._set_param(key, getattr(self, key) - self.lr * m_hat / (np.sqrt(v_hat) + ε))

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=32, verbose=True):
        """
        Training loop: epochs of mini-batch gradient descent.

        For each epoch:
            Shuffle training data
            For each mini-batch:
                Forward pass  → compute predictions
                Compute loss
                Backward pass → compute gradients
                Update weights
            Evaluate on validation set (NO gradient update)
        """
        n = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data each epoch
            perm = np.random.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train[perm]

            # Mini-batch gradient descent
            for start in range(0, n, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]

                # Forward pass
                self.forward(X_batch)

                # Backward pass (compute gradients)
                grads = self.backward(y_batch)

                # Weight update
                if self.optimiser == "adam":
                    self._update_adam(grads)
                else:
                    self._update_sgd(grads)

            # ── End of epoch: compute metrics ─────────────────────────────
            y_hat_train = self.forward(X_train).ravel()
            y_hat_val   = self.forward(X_val).ravel()
            train_loss  = bce_loss(y_train, y_hat_train)
            val_loss    = bce_loss(y_val,   y_hat_val)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 20 == 0:
                tr_acc = ((y_hat_train >= 0.5) == y_train).mean()
                va_acc = ((y_hat_val   >= 0.5) == y_val).mean()
                print(f"  Epoch {epoch+1:>4} | "
                      f"train loss={train_loss:.4f} acc={tr_acc:.3f} | "
                      f"val loss={val_loss:.4f} acc={va_acc:.3f}")
        return self

    def predict_proba(self, X):
        return self.forward(X).ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# =============================================================================
# DEMONSTRATION
# =============================================================================

print("=" * 65)
print("  NEURAL NETWORK FROM SCRATCH")
print("=" * 65)

np.random.seed(42)
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Make moons: non-linear, cannot be solved by logistic regression alone
X, y = make_moons(n_samples=800, noise=0.2, random_state=42)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

X_train, X_tmp, y_train, y_tmp = train_test_split(X_sc, y, test_size=0.3, random_state=42)
X_val,   X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

print(f"\n  Dataset: make_moons  (n=800, 2 features, non-linear boundary)")
print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"\n  Training [2] → [32] → [16] → [1] network with Adam:\n")

nn = NeuralNetwork(input_dim=2, hidden1=32, hidden2=16,
                   lr=0.01, optimiser="adam", weight_decay=1e-4)
nn.fit(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

# Final test evaluation
test_acc  = accuracy_score(y_test, nn.predict(X_test))
train_acc = accuracy_score(y_train, nn.predict(X_train))
val_acc   = accuracy_score(y_val, nn.predict(X_val))

print(f"\n  Final evaluation:")
print(f"    Train accuracy: {train_acc:.4f}")
print(f"    Val   accuracy: {val_acc:.4f}")
print(f"    Test  accuracy: {test_acc:.4f}  ← reported only once, at the end")


# =============================================================================
# PARAMETER COUNT AND ARCHITECTURE SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("  ARCHITECTURE AND PARAMETER COUNT")
print("=" * 65)

p = X_train.shape[1]   # input dimension
print(f"""
  Architecture:   Input({p}) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Sigmoid)

  Layer-by-layer parameter breakdown:
  {'Layer':>20} | {'Shape W':>15} | {'Shape b':>10} | {'# Params':>10}
  {'-'*20}-+-{'-'*15}-+-{'-'*10}-+-{'-'*10}""")
layers = [
    ("Layer 1 (Dense 32)", f"({p}, 32)", "(1, 32)", p*32 + 32),
    ("Layer 2 (Dense 16)", "(32, 16)", "(1, 16)", 32*16 + 16),
    ("Layer 3 (Dense 1)",  "(16, 1)",  "(1, 1)",   16*1 + 1),
]
total = 0
for name, wshape, bshape, n_params in layers:
    print(f"  {name:>20} | {wshape:>15} | {bshape:>10} | {n_params:>10}")
    total += n_params
print(f"  {'TOTAL':>20} | {'':>15} | {'':>10} | {total:>10}")

print(f"""
  Each weight Wᵢⱼ says: "how much should neuron i influence neuron j?"
  Positive weight → neuron i excites neuron j
  Negative weight → neuron i inhibits neuron j
  Near-zero weight → neuron i is irrelevant to neuron j
""")


# =============================================================================
# TRAINING vs VALIDATION vs TEST — THE THREE DATASETS
# =============================================================================

print("=" * 65)
print("  TRAINING vs VALIDATION vs TEST — WHAT EACH IS FOR")
print("=" * 65)

print(f"""
  Visualising loss curves (stored during training):

  {'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | Status
  {'-'*6}-+-{'-'*11}-+-{'-'*10}-+-{'-'*20}""")

for ep in [0, 9, 19, 39, 59, 79, 99]:
    tl = nn.train_losses[ep]
    vl = nn.val_losses[ep]
    gap = vl - tl
    if gap < 0.02:
        status = "generalising well"
    elif gap < 0.05:
        status = "slight overfitting"
    else:
        status = "OVERFITTING"
    print(f"  {ep+1:>6} | {tl:>11.4f} | {vl:>10.4f} | {status}")

print(f"""
  TRAINING SET:   Gradient descent runs here. Loss is DIRECTLY minimised.
  VALIDATION SET: No gradient updates. Used to detect overfitting and
                  choose hyperparameters (depth, width, LR, dropout).
  TEST SET:       Evaluated ONCE at the very end. Never used for decisions.
                  Gives an unbiased estimate of real-world performance.

  Anti-pattern: using test accuracy to pick which model to deploy.
  This inflates apparent performance — the model was implicitly optimised
  to perform well on the test set (by your selection).
""")
''',
    },

    "Activation Functions and Vanishing Gradients": {
        "description": "Sigmoid vs tanh vs ReLU vs GELU — why activations matter for deep network training",
        "runnable": True,
        "code": '''
"""
================================================================================
ACTIVATION FUNCTIONS — THE NONLINEARITY THAT MAKES DEPTH WORK
================================================================================

This script demonstrates:
    1. Numerical comparison of all common activation functions + derivatives
    2. Vanishing gradient problem with sigmoid/tanh in deep networks
    3. Dying ReLU and how Leaky ReLU fixes it
    4. Why GELU became the default in modern Transformers
    5. How depth without the right activation = no learning

================================================================================
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =============================================================================
# ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# =============================================================================

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_d(z): s = sigmoid(z); return s * (1 - s)

def tanh(z): return np.tanh(z)
def tanh_d(z): return 1 - np.tanh(z) ** 2

def relu(z): return np.maximum(0, z)
def relu_d(z): return (z > 0).astype(float)

def leaky_relu(z, a=0.01): return np.where(z > 0, z, a * z)
def leaky_relu_d(z, a=0.01): return np.where(z > 0, 1.0, a)

def gelu(z):
    """Gaussian Error Linear Unit (approximate formula from the GELU paper)."""
    return z * sigmoid(1.702 * z)
def gelu_d(z):
    s = sigmoid(1.702 * z)
    return s + 1.702 * z * s * (1 - s)

def linear(z): return z
def linear_d(z): return np.ones_like(z)

activations = {
    "Sigmoid":     (sigmoid, sigmoid_d),
    "Tanh":        (tanh, tanh_d),
    "ReLU":        (relu, relu_d),
    "Leaky ReLU":  (leaky_relu, leaky_relu_d),
    "GELU":        (gelu, gelu_d),
    "Linear":      (linear, linear_d),
}


# =============================================================================
# PART 1: NUMERICAL TABLE
# =============================================================================

print("=" * 65)
print("  PART 1: ACTIVATION VALUES AND GRADIENTS")
print("=" * 65)

z_vals = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

for name, (f, df) in activations.items():
    print(f"\n  {name}:")
    print(f"  {'z':>6}", end="")
    for z in z_vals: print(f" | {z:>8.2f}", end="")
    print()
    print(f"  {'f(z)':>6}", end="")
    for z in z_vals: print(f" | {f(z):>8.4f}", end="")
    print()
    print(f"  {'f\'(z)':>6}", end="")
    for z in z_vals: print(f" | {df(z):>8.4f}", end="")
    print()

print(f"""
  KEY OBSERVATIONS:
    Sigmoid/Tanh: gradient ≈ 0 for large |z| → vanishing gradient in deep nets
    ReLU:         gradient = 0 for z < 0     → dying ReLU problem
    Leaky ReLU:   gradient = 0.01 for z < 0  → fixes dying ReLU
    GELU:         smooth everywhere, ≈ReLU for large z, smooth decay for z<0
    Linear:       gradient = 1 everywhere    → no nonlinearity = useless in hidden layers
""")


# =============================================================================
# PART 2: VANISHING GRADIENT — THE DEPTH PROBLEM
# =============================================================================

print("=" * 65)
print("  PART 2: VANISHING GRADIENT WITH DEPTH")
print("=" * 65)

print(f"""
  Backpropagation multiplies gradients across layers (chain rule).
  If each activation gradient ≈ 0.25 (sigmoid at z=0), after L layers:

    gradient at input = Π_l (∂a_l/∂z_l) ≈ 0.25^L

  The gradient VANISHES exponentially with depth — early layers
  receive almost no gradient signal and stop learning.

  Demonstration: gradient magnitude at the FIRST layer after backprop.
""")

np.random.seed(42)
n_examples, p_features = 200, 10
X_demo = np.random.randn(n_examples, p_features)
y_demo = (np.random.rand(n_examples) > 0.5).astype(float)

print(f"  Network: input({p_features}) → [width=20]×L → output(1)")
print(f"  L (depth) | {'Sigmoid':>10} | {'Tanh':>10} | {'ReLU':>10} | {'GELU':>10}")
print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

for n_layers in [1, 2, 4, 6, 8, 12]:
    row = []
    for actname, (f, df) in [("Sigmoid",sigmoid_d), ("Tanh",tanh_d),
                              ("ReLU",relu_d), ("GELU",gelu_d)]:
        if isinstance(df, tuple): df = df[1]
        # Simple forward+backward magnitude calculation
        W = [np.random.randn(p_features if i==0 else 20, 20) * 0.1 for i in range(n_layers)]
        W_out = np.random.randn(20, 1) * 0.1
        # Forward
        a = X_demo.copy()
        zs, acts = [], [a]
        for w in W:
            z = a @ w
            zs.append(z)
            if actname == "Sigmoid": a = sigmoid(z)
            elif actname == "Tanh":  a = tanh(z)
            elif actname == "ReLU":  a = relu(z)
            else:                    a = gelu(z)
            acts.append(a)
        out = a @ W_out
        # Backward: propagate gradient magnitude
        grad_mag = 1.0
        for l in range(n_layers - 1, -1, -1):
            if actname == "Sigmoid": d = np.mean(np.abs(sigmoid_d(zs[l])))
            elif actname == "Tanh":  d = np.mean(np.abs(tanh_d(zs[l])))
            elif actname == "ReLU":  d = np.mean(np.abs(relu_d(zs[l])))
            else:                    d = np.mean(np.abs(gelu_d(zs[l])))
            grad_mag *= d
        row.append(grad_mag)
    print(f"  {n_layers:>10} | {row[0]:>10.6f} | {row[1]:>10.6f} | {row[2]:>10.6f} | {row[3]:>10.6f}")

print(f"""
  RESULT:
    Sigmoid: gradient magnitude shrinks to ≈0 by L=4-6
    Tanh:    better (gradient up to 1.0), but still shrinks
    ReLU:    gradient stays constant! (either 0 or 1, not shrinking toward 0)
    GELU:    similar to ReLU, smooth version

  This is why ReLU replaced sigmoid in deep networks (2010s).
  For very deep networks (100+ layers), residual connections also help.
""")


# =============================================================================
# PART 3: EFFECT ON TRAINING — SIGMOID vs RELU
# =============================================================================

print("=" * 65)
print("  PART 3: TRAINING DEEP NETWORKS — SIGMOID vs RELU")
print("=" * 65)

from sklearn.datasets import make_classification

np.random.seed(42)
X_cls, y_cls = make_classification(n_samples=500, n_features=10, n_informative=6,
                                    random_state=42)
X_sc = StandardScaler().fit_transform(X_cls)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_cls, test_size=0.2, random_state=42)

# Simple MLP with numpy, varying activation
def train_mlp(X_tr, y_tr, X_te, y_te, activation="relu", n_layers=5,
              width=30, lr=0.01, epochs=50, seed=42):
    np.random.seed(seed)
    p = X_tr.shape[1]
    dims = [p] + [width] * n_layers + [1]
    W = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2/dims[i]) for i in range(len(dims)-1)]
    b = [np.zeros((1, dims[i+1])) for i in range(len(dims)-1)]
    best_acc = 0
    def act(z):
        if activation == "sigmoid": return sigmoid(z)
        elif activation == "tanh":  return tanh(z)
        else:                        return relu(z)
    def act_d(z):
        if activation == "sigmoid": return sigmoid_d(z)
        elif activation == "tanh":  return tanh_d(z)
        else:                        return relu_d(z)
    for _ in range(epochs):
        perm = np.random.permutation(len(X_tr))
        for s in range(0, len(X_tr), 64):
            Xb = X_tr[perm[s:s+64]]; yb = y_tr[perm[s:s+64]]
            zs, acts = [], [Xb]
            for i, (wi, bi) in enumerate(zip(W, b)):
                z = acts[-1] @ wi + bi; zs.append(z)
                acts.append(act(z) if i < len(W)-1 else sigmoid(z))
            dA = (acts[-1] - yb.reshape(-1,1)) / len(Xb)
            for i in range(len(W)-1, -1, -1):
                dZ = dA * (sigmoid_d(zs[i]) if i == len(W)-1 else act_d(zs[i]))
                W[i] -= lr * (acts[i].T @ dZ);  b[i] -= lr * dZ.sum(0, keepdims=True)
                dA = dZ @ W[i].T
        y_pred = (sigmoid(X_te @ W[0] + b[0]) if n_layers == 0
                  else None)
    # Evaluate
    a = X_te
    for i, (wi, bi) in enumerate(zip(W, b)):
        z = a @ wi + bi
        a = act(z) if i < len(W)-1 else sigmoid(z)
    return ((a.ravel() >= 0.5) == y_te).mean()

print(f"\n  Accuracy after 50 epochs, depth=5, width=30:")
print(f"  {'Activation':>12} | {'Test Acc':>10} | Note")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*30}")
for act_name in ["sigmoid", "tanh", "relu"]:
    acc = train_mlp(X_tr, y_tr, X_te, y_te, activation=act_name)
    note = ("vanishing gradient — early layers barely learn"
            if act_name == "sigmoid" else
            "better but still some vanishing" if act_name == "tanh"
            else "gradient flows freely — best result")
    print(f"  {act_name:>12} | {acc:>10.4f} | {note}")

print(f"""
  Even at depth=5, the vanishing gradient already hurts sigmoid.
  For depth=10+, sigmoid networks often fail to learn entirely.
  ReLU (and GELU in modern networks) enables training very deep networks.
""")


# =============================================================================
# PART 4: DYING RELU AND LEAKY RELU FIX
# =============================================================================

print("=" * 65)
print("  PART 4: DYING RELU PROBLEM")
print("=" * 65)

print(f"""
  Dying ReLU: if a ReLU neuron's pre-activation z is negative for
  ALL training examples, its gradient is always 0 — it never updates.

  This happens when:
    - Learning rate is too large (weights jump to negative region)
    - Poor weight initialisation (biases pushed too negative)
    - For a neuron that outputs 0 for all examples → gradient = 0 → stays at 0

  Diagnosis: check the fraction of neurons with zero output across the batch.
""")

np.random.seed(42)
W_demo = np.random.randn(10, 20) - 2.0   # shift negative to trigger dying ReLU
X_demo = np.random.randn(100, 10)
z_demo = X_demo @ W_demo
relu_output = relu(z_demo)
dead_fraction = (relu_output == 0).mean()
print(f"  With biased weights (mean = -2.0):")
print(f"    Fraction of ReLU outputs that are 0: {dead_fraction:.3f}")
print(f"    → {100*dead_fraction:.0f}% of neurons are 'dead' — contributing nothing")

leaky_out = leaky_relu(z_demo)
dead_leaky = (leaky_out == 0).mean()
print(f"\n  Leaky ReLU with same weights:")
print(f"    Fraction of zero outputs: {dead_leaky:.3f}")
print(f"    Gradient for negative z: 0.01 (not zero!)")
print(f"    → All neurons still contribute to learning, just weakly for negative inputs")

print(f"""
  Solutions to dying ReLU:
    1. Leaky ReLU:      gradient = 0.01 for z < 0
    2. PReLU:           learned α per neuron (parametric)
    3. ELU:             smooth negative exponential instead of hard zero
    4. GELU:            smooth approximation, mostly positive gradients
    5. Careful init:    He initialisation prevents weights from going too negative
    6. Lower LR:        reduces risk of large gradient steps that kill neurons
""")
''',
    },

    "Overfitting, Dropout, and Regularisation": {
        "description": "Diagnosing overfitting from loss curves, dropout mechanics, L2 weight decay, batch norm",
        "runnable": True,
        "code": '''
"""
================================================================================
OVERFITTING, DROPOUT, AND REGULARISATION IN NEURAL NETWORKS
================================================================================

Overfitting is the central challenge in training deep networks.
This script demonstrates:
    1. Reproducing overfitting — deliberately creating it
    2. Loss curves as a diagnostic tool
    3. Dropout mechanics — training vs inference
    4. L2 weight decay
    5. Early stopping
    6. How much data you need vs how much capacity you use

================================================================================
"""

import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)


# =============================================================================
# PART 1: DELIBERATELY CREATING OVERFITTING
# =============================================================================

print("=" * 65)
print("  PART 1: CREATING AND DIAGNOSING OVERFITTING")
print("=" * 65)

# Small training set + large network = guaranteed overfitting
X, y = make_classification(n_samples=2000, n_features=20, n_informative=8,
                            n_redundant=4, random_state=42)
X_sc = StandardScaler().fit_transform(X)

# Deliberately: tiny training set (n=100), large network, many epochs
X_tr_small, X_tmp, y_tr_small, y_tmp = train_test_split(X_sc, y, train_size=100, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

print(f"\n  Small training set (n=100), large network, no regularisation:")

for hidden in [(200, 200, 200), (32,), (8,)]:
    nn = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500,
                       random_state=42, learning_rate_init=0.001,
                       activation="relu", solver="adam", alpha=0.0)
    nn.fit(X_tr_small, y_tr_small)
    tr_acc = accuracy_score(y_tr_small, nn.predict(X_tr_small))
    te_acc = accuracy_score(y_test, nn.predict(X_test))
    gap    = tr_acc - te_acc
    status = "SEVERELY OVERFIT" if gap > 0.1 else ("overfit" if gap > 0.05 else "ok")
    print(f"  Network {str(hidden):>20}: train={tr_acc:.3f}, test={te_acc:.3f}, gap={gap:+.3f}  {status}")

print(f"""
  The large network memorises the 100 training examples perfectly.
  It has seen each example hundreds of times — it knows the training data
  by heart, not the underlying pattern.

  Signs of overfitting:
    • Training accuracy ≈ 1.0 (memorised training set)
    • Validation / test accuracy much lower
    • Val loss starts rising after an initial dip
    • The gap grows as the network trains more
""")


# =============================================================================
# PART 2: DROPOUT MECHANICS
# =============================================================================

print("=" * 65)
print("  PART 2: DROPOUT — HOW IT WORKS")
print("=" * 65)

print(f"""
  Dropout randomly zeros out neurons during each forward pass in training.
  The key insight: it forces the network to learn REDUNDANT representations —
  any given neuron might be absent, so all neurons must be useful on their own.

  Analogy: training a sports team where random players are absent each day.
  The team can't rely on any one star player — everyone must know the plays.

  Implementation (inverted dropout):
""")

def dropout_demo(a, p_keep=0.5, training=True):
    """
    Inverted dropout implementation.

    Training: randomly zero neurons, scale up by 1/p_keep to keep expected value.
    Inference: use all neurons as-is (no masking, no scaling needed).
    """
    if not training:
        return a, None   # at test time: all neurons active, no scaling
    mask   = (np.random.rand(*a.shape) < p_keep)   # 1 with prob p_keep
    a_out  = a * mask / p_keep                       # zero the dropped neurons, scale up
    return a_out, mask

np.random.seed(7)
p_keep = 0.5
neuron_outputs = np.array([0.8, 0.3, 0.9, 0.1, 0.7, 0.4, 0.6, 0.5])
print(f"  Original neuron outputs:   {neuron_outputs}")
print(f"  Dropout probability:       p_keep={p_keep} (50% kept, 50% dropped)")
print()
print(f"  {'Pass':>6} | {'After dropout':>40} | {'Mean (should ≈ original mean)'}")
print(f"  {'-'*6}-+-{'-'*40}-+-{'-'*30}")
for i in range(4):
    out, mask = dropout_demo(neuron_outputs.copy(), p_keep=p_keep, training=True)
    print(f"  Train{i+1} | {np.round(out,2)} | {out.mean():.4f} vs {neuron_outputs.mean():.4f}")

# Test time: no dropout
out_test, _ = dropout_demo(neuron_outputs.copy(), training=False)
print(f"  {'Test':>6} | {np.round(out_test,2)} | All neurons active")

print(f"""
  Scaling by 1/p_keep during training ensures the expected output is the same
  at test time without any scaling needed ("inverted dropout").

  Without scaling: training outputs have expected value p_keep × original.
  Test outputs have expected value = original → different distribution → bad.

  Typical dropout probabilities:
    Hidden layers: p_drop = 0.2–0.5   (keep 50–80%)
    Input layer:   p_drop = 0.1–0.2   (less aggressive — inputs are the raw features)
    Output layer:  never apply dropout to the output
""")


# =============================================================================
# PART 3: REGULARISATION COMPARISON
# =============================================================================

print("=" * 65)
print("  PART 3: REGULARISATION COMPARISON")
print("=" * 65)

# Use sklearn MLPClassifier for quick comparisons
# alpha = L2 weight decay,  early_stopping = early stopping on val loss

configs = [
    ("No regularisation",      dict(alpha=0.0,    early_stopping=False, max_iter=300)),
    ("L2 weight decay (α=0.01)",dict(alpha=0.01,  early_stopping=False, max_iter=300)),
    ("L2 weight decay (α=0.1)", dict(alpha=0.1,   early_stopping=False, max_iter=300)),
    ("Early stopping",          dict(alpha=0.0,   early_stopping=True,  validation_fraction=0.15, n_iter_no_change=15, max_iter=500)),
    ("L2 + Early stopping",     dict(alpha=0.01,  early_stopping=True,  validation_fraction=0.15, n_iter_no_change=15, max_iter=500)),
]

print(f"\n  Small training set (n=100), testing regularisation strategies:")
print(f"  {'Config':>32} | {'Train Acc':>10} | {'Test Acc':>10} | {'Gen Gap':>10}")
print(f"  {'-'*32}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

for name, params in configs:
    nn_cfg = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                           solver="adam", learning_rate_init=0.001,
                           random_state=42, **params)
    nn_cfg.fit(X_tr_small, y_tr_small)
    tr = accuracy_score(y_tr_small, nn_cfg.predict(X_tr_small))
    te = accuracy_score(y_test,     nn_cfg.predict(X_test))
    print(f"  {name:>32} | {tr:>10.4f} | {te:>10.4f} | {tr-te:>+10.4f}")

print(f"""
  EFFECTS:
    No regularisation:   train ≈ 1.0, test much lower → overfitting
    L2 (α=0.01):        slight gap reduction — shrinks large weights
    L2 (α=0.1):         too much regularisation → underfitting
    Early stopping:      stops before memorisation peaks → good gap reduction
    L2 + Early stopping: often the best combination
""")


# =============================================================================
# PART 4: HOW MUCH DATA vs NETWORK CAPACITY
# =============================================================================

print("=" * 65)
print("  PART 4: TRAINING SET SIZE vs NETWORK CAPACITY")
print("=" * 65)

print(f"""
  The generalisation gap depends on both the network size AND training set size.
  Key principle: gap = complexity of the model relative to the amount of data.
""")

print(f"\n  {'n_train':>8} | {'Small net (32,16)':>18} | {'Large net (256,128)':>20}")
print(f"  {'-'*8}-+-{'-'*18}-+-{'-'*20}")
print(f"  {'':>8} | {'Train Acc | Test Acc':>18} | {'Train Acc | Test Acc':>20}")
print(f"  {'-'*8}-+-{'-'*18}-+-{'-'*20}")

for n_tr in [50, 100, 200, 500, 1000, 1500]:
    X_sub, _, y_sub, _ = train_test_split(X_sc, y, train_size=n_tr, random_state=42)
    X_te_fixed, _, y_te_fixed, _ = train_test_split(X_sc, y, test_size=400, random_state=99)

    results = []
    for hidden in [(32, 16), (256, 128)]:
        nn_s = MLPClassifier(hidden_layer_sizes=hidden, max_iter=200,
                             random_state=42, alpha=0.001, solver="adam")
        nn_s.fit(X_sub, y_sub)
        tr_a = accuracy_score(y_sub, nn_s.predict(X_sub))
        te_a = accuracy_score(y_te_fixed, nn_s.predict(X_te_fixed))
        results.append((tr_a, te_a))
    print(f"  {n_tr:>8} | {results[0][0]:.3f}  |  {results[0][1]:.3f}  | {results[1][0]:.3f}  |  {results[1][1]:.3f}")

print(f"""
  Key observations:
    Small network:  low overfitting even with small n (less capacity to memorise)
    Large network:  severe overfitting with small n, converges with large n
    Enough data:    both networks achieve similar test accuracy

  Practical guideline:
    More data → bigger network is fine
    Limited data → use smaller network, stronger regularisation, data augmentation
""")
''',
    },

    "Neural Network vs Classical Models": {
        "description": "Where NNs win and lose vs logistic regression, SVM, gradient boosting, and random forest",
        "runnable": True,
        "code": '''
"""
================================================================================
NEURAL NETWORKS vs ALL PREVIOUS MODELS
================================================================================

The final comparison across all 9 models in the series.
Neural networks sit at the top of the flexibility hierarchy —
but flexibility comes at a cost (data, compute, tuning, interpretability).

This script compares across:
    1. Tabular data (where GBM often wins)
    2. Non-linear data (where NNs and kernel SVMs shine)
    3. Sample complexity (where NNs need more data than simpler models)
    4. Training and inference speed

================================================================================
"""

import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)


# =============================================================================
# PART 1: ALL MODELS ON MULTIPLE DATASETS
# =============================================================================

print("=" * 70)
print("  PART 1: ALL 9 MODEL TYPES — ACCURACY COMPARISON")
print("=" * 70)

datasets = {
    "Linear (easy)":        make_classification(500, 10, n_informative=5, class_sep=2.0, random_state=42),
    "Linear (noisy)":       make_classification(500, 10, n_informative=3, class_sep=0.7, random_state=42),
    "Moons (non-linear)":   make_moons(500, noise=0.2, random_state=42),
    "Circles (concentric)": make_circles(500, noise=0.1, factor=0.4, random_state=42),
    "High-dim (p=50)":      make_classification(500, 50, n_informative=10, random_state=42),
}

models = {
    "Logistic Reg":      LogisticRegression(C=1.0, max_iter=1000),
    "SVM (RBF)":         SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Decision Tree":     DecisionTreeClassifier(random_state=42),
    "Random Forest":     RandomForestClassifier(100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(100, learning_rate=0.1, max_depth=3, random_state=42),
    "KNN (K=5)":         KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":       GaussianNB(),
    "Neural Net (MLP)":  MLPClassifier((64, 32), max_iter=300, random_state=42,
                                        alpha=0.001, learning_rate_init=0.001),
}

print(f"\n  {'Dataset':>22}", end="")
for name in models:
    print(f" | {name:>16}", end="")
print()
print("  " + "-"*22, end="")
for _ in models:
    print(f"-+-{'-'*16}", end="")
print()

for ds_name, (X, y) in datasets.items():
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25, random_state=42)
    print(f"  {ds_name:>22}", end="")
    for clf in models.values():
        c = type(clf)(**clf.get_params())
        c.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, c.predict(X_te))
        print(f" | {acc:>16.4f}", end="")
    print()

print(f"""
  Key patterns:
    Linear data:     Logistic Regression matches or beats neural networks
                     (NN is overkill — it learns the same linear boundary)
    Non-linear:      SVM, RF, GB, and MLP all shine; LR fails
    High-dim:        GB and RF handle it better out-of-the-box; NN needs tuning

  Neural networks do not automatically win on tabular data.
  On structured/tabular data, Gradient Boosting (XGBoost/LightGBM) is
  usually the best baseline. NNs require more tuning and more data.
  NNs dominate on images, text, audio — where features must be learned.
""")


# =============================================================================
# PART 2: SAMPLE COMPLEXITY — HOW MUCH DATA EACH MODEL NEEDS
# =============================================================================

print("=" * 70)
print("  PART 2: SAMPLE COMPLEXITY — ACCURACY vs TRAINING SET SIZE")
print("=" * 70)

X_big, y_big = make_classification(n_samples=2000, n_features=20, n_informative=10,
                                    n_redundant=4, random_state=42)
X_big = StandardScaler().fit_transform(X_big)
X_te_fix, _, y_te_fix, _ = train_test_split(X_big, y_big, test_size=400, random_state=99)

sample_models = {
    "Logistic Reg": LogisticRegression(C=1.0, max_iter=1000),
    "Naive Bayes":  GaussianNB(),
    "Random Forest":RandomForestClassifier(100, random_state=42, n_jobs=-1),
    "Grad Boosting":GradientBoostingClassifier(50, random_state=42),
    "Neural Net":   MLPClassifier((64,32), max_iter=200, random_state=42,
                                   alpha=0.001, learning_rate_init=0.001),
}

print(f"\n  {'n_train':>8}", end="")
for name in sample_models:
    print(f" | {name:>14}", end="")
print()
print(f"  {'-'*8}", end="")
for _ in sample_models:
    print(f"-+-{'-'*14}", end="")
print()

for n_tr in [30, 50, 100, 200, 500, 1000, 1500]:
    X_s, _, y_s, _ = train_test_split(X_big, y_big, train_size=n_tr, random_state=42)
    print(f"  {n_tr:>8}", end="")
    for clf in sample_models.values():
        c = type(clf)(**clf.get_params())
        c.fit(X_s, y_s)
        acc = accuracy_score(y_te_fix, c.predict(X_te_fix))
        print(f" | {acc:>14.4f}", end="")
    print()

print(f"""
  Key observations:
    Very small n (<100): Naive Bayes and Logistic Regression often competitive
    Medium n (100-500):  Random Forest and Gradient Boosting pull ahead
    Large n (>500):      Neural Network catches up and can surpass others

  Neural Networks benefit most from large datasets.
  With limited data: use simpler models with strong priors (NB, LR, small RF).
  With abundant data: NN's advantage grows — it can learn complex representations.
""")


# =============================================================================
# PART 3: TRAINING AND INFERENCE SPEED
# =============================================================================

print("=" * 70)
print("  PART 3: TRAINING AND INFERENCE SPEED")
print("=" * 70)

X_sp, y_sp = make_classification(n_samples=1000, n_features=20,
                                   n_informative=8, random_state=42)
X_sp = StandardScaler().fit_transform(X_sp)
X_tr_sp, X_te_sp, y_tr_sp, y_te_sp = train_test_split(X_sp, y_sp, test_size=0.25, random_state=42)

speed_models = [
    ("Logistic Regression",       LogisticRegression(C=1.0, max_iter=1000)),
    ("Gaussian Naive Bayes",      GaussianNB()),
    ("KNN (K=5)",                 KNeighborsClassifier(n_neighbors=5)),
    ("Decision Tree",             DecisionTreeClassifier(random_state=42)),
    ("Random Forest (B=100)",     RandomForestClassifier(100, random_state=42, n_jobs=-1)),
    ("Gradient Boosting (B=100)", GradientBoostingClassifier(100, random_state=42)),
    ("SVM (RBF)",                 SVC(kernel="rbf", C=1.0, gamma="scale")),
    ("MLP (64, 32), 300 epochs",  MLPClassifier((64,32), max_iter=300, random_state=42, alpha=0.001)),
]

print(f"\n  n=1000, p=20 features")
print(f"  {'Model':>34} | {'Train ms':>10} | {'Predict ms':>12} | {'Test Acc':>10}")
print(f"  {'-'*34}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

for name, clf in speed_models:
    t0 = time.perf_counter()
    clf.fit(X_tr_sp, y_tr_sp)
    train_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    preds = clf.predict(X_te_sp)
    pred_ms = (time.perf_counter() - t0) * 1000
    acc = accuracy_score(y_te_sp, preds)
    print(f"  {name:>34} | {train_ms:>10.1f} | {pred_ms:>12.3f} | {acc:>10.4f}")

print(f"""
  Observations:
    Fastest training:    Naive Bayes (single pass counting)
    Slowest training:    MLP (many gradient descent iterations)
    Fastest inference:   Logistic Regression, Naive Bayes (dot product)
    Slowest inference:   KNN (must search all training examples)
    Best accuracy:       RF and GBM competitive; MLP similar

  The accuracy-speed tradeoff:
    Simple models (LR, NB): extremely fast, lower accuracy on complex data
    Ensemble models (RF, GBM): moderate speed, excellent accuracy
    Neural networks: slower training, excellent accuracy on sufficient data

  In production: inference speed matters most.
  Neural networks on GPU: milliseconds for millions of predictions.
  Neural networks on CPU: can be slower than tree models.
""")


# =============================================================================
# PART 4: THE FULL PICTURE — WHEN TO CHOOSE WHAT
# =============================================================================

print("=" * 70)
print("  PART 4: WHICH MODEL WHEN — DECISION GUIDE")
print("=" * 70)

print(f"""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │  START WITH LOGISTIC REGRESSION / LINEAR MODELS WHEN:                    │
  │    • Interpretability is critical (legal, medical requirements)          │
  │    • Data is linearly separable or close to it                           │
  │    • Feature engineering can create the right nonlinear features         │
  │    • Very small datasets (good with strong prior assumptions)            │
  │                                                                          │
  │  USE RANDOM FOREST WHEN:                                                 │
  │    • Good out-of-the-box performance with minimal tuning                 │
  │    • Mixed feature types, missing values                                 │
  │    • Moderate-sized tabular data                                         │
  │    • Fast training needed with reasonable accuracy                       │
  │                                                                          │
  │  USE GRADIENT BOOSTING (XGBoost/LightGBM) WHEN:                          │
  │    • Maximum accuracy on tabular data (Kaggle benchmark winner)          │
  │    • Can tune hyperparameters and use early stopping                     │
  │    • Large tabular datasets                                              │
  │                                                                          │
  │  USE NEURAL NETWORKS WHEN:                                               │
  │    • Data is images, text, audio, video — raw, unstructured data         │
  │    • Dataset is large (n > 10k–100k examples)                            │
  │    • Feature engineering would be impractical or impossible              │
  │    • Transfer learning is available (pretrained models)                  │
  │    • End-to-end learning: raw input to final output                      │
  │    • Multi-task learning: one model, multiple outputs                    │
  │                                                                          │
  │  ALL TREE MODELS FAIL WHEN:                                              │
  │    • Extrapolation required — NNs and linear models handle this          │
  │    • Sequential/temporal data — use RNNs, LSTMs, Transformers            │
  │    • Very high-dimensional raw data (images) — use CNNs                  │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘
""")
''',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 65)
    print("  NEURAL NETWORKS: UNIVERSAL APPROXIMATORS")
    print("=" * 65)
    print("""
  Key concepts demonstrated:
    • Neuron: z = W·x + b,  a = σ(z)   — linear transform + nonlinearity
    • Neurons ARE historically logic gates — McCulloch-Pitts (1943) were
      binary threshold units. Modern neurons use smooth activations.
    • Smooth activations (ReLU, GELU) enable backpropagation
    • Forward pass: compute predictions layer by layer
    • Loss function: measures how wrong the predictions are
    • Backpropagation: chain rule propagates gradients back through layers
    • Weight update: W ← W − α·∂L/∂W  (gradient descent)
    • Patterns found: weights grow for features that correlate with correct labels
    • Training/Validation/Test: three distinct purposes
    • Overfitting: memorises training data; fix with dropout, L2, early stopping
    """)

    np.random.seed(42)

    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_sc = StandardScaler().fit_transform(X)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_sc, y, test_size=0.3, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    print("=" * 65)
    print("  TRAINING A NEURAL NETWORK ON MAKE_MOONS")
    print("=" * 65)

    for hidden, desc in [
        ((2,), "minimal (2 neurons) — linear-ish"),
        ((8, 8), "small network"),
        ((64, 32), "medium network"),
    ]:
        nn = MLPClassifier(hidden_layer_sizes=hidden, max_iter=300,
                           random_state=42, alpha=0.001,
                           learning_rate_init=0.01, solver="adam")
        nn.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, nn.predict(X_tr))
        va_acc = accuracy_score(y_val, nn.predict(X_val))
        te_acc = accuracy_score(y_te, nn.predict(X_te))
        params = sum(w.size for w in nn.coefs_) + sum(b.size for b in nn.intercepts_)
        print(f"\n  {desc}  {str(hidden):>10}")
        print(f"    Parameters: {params:,}")
        print(f"    Train acc:  {tr_acc:.4f}")
        print(f"    Val   acc:  {va_acc:.4f}")
        print(f"    Test  acc:  {te_acc:.4f}  ← reported once, only at the end")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1.  A neuron = σ(Σ wᵢ xᵢ + b) — weighted sum + nonlinearity
  2.  YES — neurons were originally binary logic gates (McCulloch-Pitts 1943)
      Modern neurons use smooth activations for differentiability
  3.  Without nonlinear activations, any depth = one linear layer
  4.  Forward pass: input → layer 1 → layer 2 → ... → prediction
  5.  Loss: measures error between prediction and ground truth
  6.  Backprop: chain rule computes ∂L/∂W for every weight simultaneously
  7.  Weight update: W ← W − α · ∂L/∂W  (gradient descent step)
  8.  Adam adapts learning rate per-parameter (β₁=0.9, β₂=0.999)
  9.  Patterns emerge: weights strengthen for features that predict the label
  10. Training set: gradient descent runs here (model sees this data)
  11. Validation set: monitor generalisation, pick hyperparameters, early stop
  12. Test set: evaluated ONCE at the very end — never used for decisions
  13. Overfitting: train loss falls, val loss rises → fix with regularisation
  14. Dropout: randomly zero neurons during training → forces redundancy
  15. ReLU replaced sigmoid in hidden layers — vanishing gradient is solved
  16. Universal Approximation Theorem: any function can be approximated
  17. Deep > wide: compositional representations, efficient parameter use
  18. NNs dominate on images/text/audio; GBM often wins on tabular data
    """)


# # ─────────────────────────────────────────────────────────────────────────────
# # CONTENT EXPORT
# # ─────────────────────────────────────────────────────────────────────────────
#
# def get_content():
#     return {
#         "theory": THEORY,
#         "theory_raw": THEORY,
#         "complexity": COMPLEXITY,
#         "operations": OPERATIONS,
#         "interactive_components": [],
#     }


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
    visual_height = 400
    try:
        from Supervised_Learning.visuals.neural_network_visual import (   # ← match your exact folder casing
            NN_VISUAL_HTML,
            NN_VISUAL_HEIGHT,
        )
        visual_html   = NN_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = NN_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[09_neural_networks.py] Could not load visual: {e}", stacklevel=2)

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


