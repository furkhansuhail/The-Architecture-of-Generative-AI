"""
Backpropagation — Teaching Neural Networks to Learn
=====================================================

The algorithm that made deep learning possible.

"""
import base64
import os
import textwrap
import re

TOPIC_NAME = "04_Backpropagation"
DISPLAY_NAME = "04 . Backpropagation Explanation"
ICON         = "🚨"
SUBTITLE     = "How Backpropagation is carried out ?"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER — converts local images to base64 HTML for st.markdown()
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    """Convert a local image file to an HTML <img> tag with base64 data.
    This allows images to render inside st.markdown() with unsafe_allow_html=True.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="width:{width}; border-radius:8px; margin:12px 0;">'
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
### Backpropagation — The Algorithm That Taught Machines to Learn 🧠

---

## **Part 1: The Problem Backpropagation Solves**

### The Credit Assignment Problem

When a neural network makes a wrong prediction, we know the OUTPUT was wrong.
But the output neuron didn't act alone. It received signals from hidden neurons.
Those hidden neurons received signals from the layer before them. Ultimately,
every single weight in the entire network had *some* influence on the final mistake.

The question is: **How do we assign blame (or credit) to each individual weight?**

This is called the **Credit Assignment Problem**, and it was one of the central
unsolved challenges in AI for two decades (the late 1960s to mid-1980s).

Think of it like this: a football team loses a game. Who was at fault?

    • The goalkeeper who let in a goal?
    • The midfielder who lost possession?
    • The coach who picked the wrong formation?
    • All of them — in different proportions?

Backpropagation is the mathematician's answer to this question for neural networks.

It figures out exactly how much each weight was responsible for the error, and
adjusts each one by just the right amount.

---

### Why Simple Guessing Won't Work

One naive solution: randomly adjust a weight, see if the error improves, keep the change if it does.
This is called random search or hill climbing. For a network with:

    • 3 layers, 10 neurons each = ~200 weights

    If you test each weight by nudging it slightly, that's 200 tests just to
    complete ONE update step. Real networks have MILLIONS of parameters.
    ResNet-50 has 25 million. GPT-3 has 175 billion.

    At 200 tests per step × 1 million training examples × 100 epochs,
    you'd need 20 TRILLION tests. That's computationally impossible.

Backpropagation computes the exact direction to adjust ALL weights simultaneously
in a single backward pass — as fast as the forward pass itself.


---

## **Part 2: The Calculus Foundation — The Chain Rule**

### What Is a Derivative?

Before we can understand backpropagation, we need to understand what a derivative
*means* in this context.

A derivative answers: **"If I change this input by a tiny amount, how much does
the output change?"**

For a weight w in our network:

    ∂Loss/∂w  =  "If I change w by a tiny amount ε,
                  how much does the Loss change?"

If this value is positive, it means: "increasing w makes the loss WORSE."
If this value is negative, it means: "increasing w makes the loss BETTER."

The magnitude tells us HOW SENSITIVE the loss is to that particular weight.

    ∂Loss/∂w = +5.0   →  This weight has a big positive effect on loss.
                          Decrease it to reduce the loss.

    ∂Loss/∂w = -0.01  →  This weight has a tiny negative effect on loss.
                          This weight barely matters right now.


### The Chain Rule — Calculus in One Sentence

The chain rule states:

    If y depends on z, and z depends on x, then:
    dy/dx  =  (dy/dz) × (dz/dx)

In plain English: **"To find how x affects y, multiply all the little
effects along the path from x to y."**

A real-world analogy:
    You turn a gear → the gear turns a second gear → the second gear moves a lever.
    If gear 1 rotates 1° and causes gear 2 to rotate 2°, and rotating gear 2 by 1°
    moves the lever by 3cm, then rotating gear 1 by 1° moves the lever by 2×3 = 6cm.

    Total effect = product of effects along the chain.

This is exactly what backpropagation does — it traces the chain of computations
from the loss backward through every layer, multiplying the partial effects at
each step.


### The Computation Graph

Before computing derivatives, it helps to draw the computation as a graph.
For a simple 2-input, 1-hidden-layer, 1-output network:

         x1 ──────┐
                  │    w1
         x2 ──────┴──[ × w1 ]──[ + z1 ]──[ σ ]── h1 ──┐
                                                      │    w3
         x1 ──────┐                                   ├──[ × w3 ]──[ + z3 ]──[ σ ]── ŷ ──[ Loss ]
                  │    w2                             │
         x2 ──────┴──[ × w2 ]──[ + z2 ]──[ σ ]── h2 ──┘

Every arrow and box is a step in the computation. The chain rule lets us
flow gradients *backward* through every single arrow.


---

## **Part 3: Forward Pass — Setting the Stage**

Before we can go backward, we go forward. Let's trace a complete example
with fixed weights so we can do real numbers.

**Network Architecture:** 2 inputs → 2 hidden neurons → 1 output
**Training Example:** Input = [0, 1], Expected Output = [1]

    INITIAL WEIGHTS (hand-chosen for clarity):
    ─────────────────────────────────────────
    Hidden Neuron 1 (H1):
        w1  = 0.15  (weight from x1)
        w2  = 0.20  (weight from x2)
        b1  = 0.35  (bias)

    Hidden Neuron 2 (H2):
        w3  = 0.25  (weight from x1)
        w4  = 0.30  (weight from x2)
        b2  = 0.35  (bias)

    Output Neuron (O1):
        w5  = 0.40  (weight from H1)
        w6  = 0.45  (weight from H2)
        b3  = 0.60  (bias)


### Forward Pass — Step by Step

**STEP 1: Hidden Layer Computation**

    H1 pre-activation (z_h1):
        z_h1 = (0 × 0.15) + (1 × 0.20) + 0.35
             = 0.0 + 0.20 + 0.35
             = 0.55
        h1   = sigmoid(0.55) = 1 / (1 + e^-0.55) = 0.6343

    H2 pre-activation (z_h2):
        z_h2 = (0 × 0.25) + (1 × 0.30) + 0.35
             = 0.0 + 0.30 + 0.35
             = 0.65
        h2   = sigmoid(0.65) = 1 / (1 + e^-0.65) = 0.6570

**STEP 2: Output Layer Computation**

    O1 pre-activation (z_o):
        z_o  = (0.6343 × 0.40) + (0.6570 × 0.45) + 0.60
             = 0.2537 + 0.2957 + 0.60
             = 1.1493
        ŷ    = sigmoid(1.1493) = 1 / (1 + e^-1.1493) = 0.7594

**STEP 3: Compute the Loss**

    We use Mean Squared Error (MSE):
        Loss = (1/2) × (expected - actual)²
             = (1/2) × (1 - 0.7594)²
             = (1/2) × (0.2406)²
             = (1/2) × 0.0579
             = 0.02895

    The 1/2 factor is a mathematical convenience — it makes the derivative
    cleaner (the 2 from the power rule cancels with the 1/2).

---

## **Part 4: Backward Pass — Backpropagation in Action**

Now the real work begins. We go BACKWARD through the network, computing
how much each weight contributed to the error.

The goal for each weight w is to compute:  **∂Loss / ∂w**


### Phase 1: Output Layer Gradients

**∂Loss / ∂ŷ  (How does loss change with output?)**

    Loss = (1/2)(1 - ŷ)²

    ∂Loss/∂ŷ = -(expected - ŷ)
              = -(1 - 0.7594)
              = -0.2406

    (Negative because increasing ŷ would decrease the loss here)

**∂ŷ / ∂z_o  (How does output change with pre-activation?)**

    This is the sigmoid derivative:
    ∂ŷ/∂z_o = ŷ × (1 - ŷ)
             = 0.7594 × (1 - 0.7594)
             = 0.7594 × 0.2406
             = 0.1827

**Output Layer Delta (δ_o):**

    Combining the above two via chain rule:
    δ_o = ∂Loss/∂ŷ × ∂ŷ/∂z_o
        = -0.2406 × 0.1827          ← chain rule: multiply
        = -0.04397

    In implementation: δ_o = (expected - ŷ) × sigmoid_derivative(ŷ)
                           = (1 - 0.7594) × 0.1827
                           = 0.2406 × 0.1827
                           = 0.04397   (positive when using the + convention)

**∂Loss / ∂w5  (gradient for w5, the weight connecting H1 to O1):**

    By the chain rule:
    ∂Loss/∂w5 = δ_o × h1
              = 0.04397 × 0.6343
              = 0.02789

    Intuition: w5 contributed h1=0.6343 of signal to the output.
    The output was wrong by delta δ_o. So w5's responsibility is proportional
    to how much signal it carried.

**∂Loss / ∂w6  (gradient for w6, the weight connecting H2 to O1):**

    ∂Loss/∂w6 = δ_o × h2
              = 0.04397 × 0.6570
              = 0.02889

**∂Loss / ∂b3  (gradient for output bias):**

    ∂Loss/∂b3 = δ_o × 1    (bias input is always 1)
              = 0.04397


### Phase 2: Hidden Layer Gradients (The "Backward" Part)

This is where backpropagation earns its name. The hidden neurons had no
"expected" value — we didn't tell them what to output. So how do we update them?

Answer: we **propagate the error signal backward** from the output layer.

The output neuron's delta tells us how much the combined hidden layer output
was responsible for the error. We distribute that blame backward, weighted
by the connection strengths.

**Backward error signal to H1:**

    The error flowing back to H1 = δ_o × w5
                                 = 0.04397 × 0.40
                                 = 0.01759

    Why w5? Because that's the connection between H1 and O1.
    If w5 were 0, H1 had no influence on the output, so it gets no blame.
    If w5 is large, H1 had a large influence, so it gets more blame.

**Hidden Layer Delta for H1 (δ_h1):**

    δ_h1 = (error flowing back to H1) × sigmoid_derivative(h1)
         = 0.01759 × h1 × (1 - h1)
         = 0.01759 × 0.6343 × (1 - 0.6343)
         = 0.01759 × 0.6343 × 0.3657
         = 0.004085

**Backward error signal to H2:**

    Error to H2 = δ_o × w6
                = 0.04397 × 0.45
                = 0.01979

**Hidden Layer Delta for H2 (δ_h2):**

    δ_h2 = 0.01979 × h2 × (1 - h2)
         = 0.01979 × 0.6570 × (1 - 0.6570)
         = 0.01979 × 0.6570 × 0.3430
         = 0.004459

**Gradients for Hidden Layer Weights:**

    ∂Loss/∂w2 = δ_h1 × x2   = 0.004085 × 1 = 0.004085  (x2 = 1 for this example)
    ∂Loss/∂w1 = δ_h1 × x1   = 0.004085 × 0 = 0.000000  (x1 = 0 for this example)
    ∂Loss/∂b1 = δ_h1         = 0.004085

    ∂Loss/∂w4 = δ_h2 × x2   = 0.004459 × 1 = 0.004459
    ∂Loss/∂w3 = δ_h2 × x1   = 0.004459 × 0 = 0.000000
    ∂Loss/∂b2 = δ_h2         = 0.004459

    Notice: w1 and w3 (connecting x1 to hidden neurons) get ZERO gradient.
    This is because x1 = 0, so those weights contributed nothing to this
    particular forward pass. They won't be updated for this example.


---

## **Part 5: Weight Update — Gradient Descent**

Now that we have all gradients, we update every weight using gradient descent:

    new_weight = old_weight + learning_rate × gradient

    (We ADD because our delta already accounts for direction)

Using learning_rate (η) = 0.5:

    UPDATED OUTPUT LAYER WEIGHTS:
    ──────────────────────────────
    w5 (new) = 0.40 + 0.5 × 0.02789 = 0.40 + 0.01395 = 0.41395
    w6 (new) = 0.45 + 0.5 × 0.02889 = 0.45 + 0.01445 = 0.46445
    b3 (new) = 0.60 + 0.5 × 0.04397 = 0.60 + 0.02199 = 0.62199

    UPDATED HIDDEN LAYER WEIGHTS:
    ──────────────────────────────
    w1 (new) = 0.15 + 0.5 × 0.000000 = 0.15 + 0.000 = 0.15000  (unchanged — x1 was 0)
    w2 (new) = 0.20 + 0.5 × 0.004085 = 0.20 + 0.00243 = 0.20243
    b1 (new) = 0.35 + 0.5 × 0.004085 = 0.35 + 0.00243 = 0.35243

    w3 (new) = 0.25 + 0.5 × 0.000000 = 0.25 + 0.000 = 0.25000  (unchanged — x1 was 0)
    w4 (new) = 0.30 + 0.5 × 0.004459 = 0.30 + 0.00223 = 0.30223
    b2 (new) = 0.35 + 0.5 × 0.004459 = 0.35 + 0.00223 = 0.35223

This was ONE training step on ONE example.
Training repeats this cycle — forward → loss → backward → update — thousands of
times across all examples until the network's loss converges to a minimum.

---

## **Part 6: The Full Backpropagation Algorithm**

Here is the complete algorithm in pseudocode, generalized to any depth:

    ╔══════════════════════════════════════════════════════════════════╗
    ║            BACKPROPAGATION ALGORITHM (PSEUDOCODE)                ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  For each training example (x, y):                               ║
    ║                                                                  ║
    ║   ── FORWARD PASS ──────────────────────────────────────────     ║
    ║   1. Push x through the network layer by layer                   ║
    ║   2. At each neuron:  z = Σ(w_i × input_i) + b                   ║
    ║                       a = activation(z)                          ║
    ║   3. Cache each z and a (we'll need them in the backward pass)   ║
    ║   4. Compute Loss = (1/2)(y - ŷ)²                                ║
    ║                                                                  ║
    ║   ── BACKWARD PASS ─────────────────────────────────────────     ║
    ║   5. OUTPUT LAYER:                                               ║
    ║      For each output neuron i:                                   ║
    ║        δ_i = (y_i - ŷ_i) × σ'(z_i)                               ║
    ║                                                                  ║
    ║   6. HIDDEN LAYERS (from last hidden layer to first):            ║
    ║      For each hidden neuron j in layer L:                        ║
    ║        δ_j = σ'(z_j) × Σ(δ_k × w_jk)  for all k in layer L+1     ║
    ║                                                                  ║
    ║   ── WEIGHT UPDATE ──────────────────────────────────────────    ║
    ║   7. For every weight w_ij in the network:                       ║
    ║        w_ij += η × δ_j × a_i                                     ║
    ║      For every bias b_j:                                         ║
    ║        b_j  += η × δ_j                                           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝

    WHERE:
        η           = learning rate (e.g. 0.01 to 1.0)
        σ'(z)       = sigmoid_derivative = σ(z) × (1 - σ(z))
        δ           = delta (error signal for each neuron)
        a_i         = activation (output) of the neuron feeding INTO neuron j
        Σ(δ_k × w_jk) = sum of downstream deltas × connecting weights

---

## **Part 7: Why the Sigmoid Derivative Matters**

### The "Confidence Scaling" Effect

Look at the shape of sigmoid_derivative(output):

    output  →  sigmoid_derivative
    ───────────────────────────────
    0.00    →  0.00 × 1.00 = 0.000   (extreme)
    0.10    →  0.10 × 0.90 = 0.090
    0.25    →  0.25 × 0.75 = 0.188
    0.50    →  0.50 × 0.50 = 0.250   ← MAXIMUM sensitivity
    0.75    →  0.75 × 0.25 = 0.188
    0.90    →  0.90 × 0.10 = 0.090
    1.00    →  1.00 × 0.00 = 0.000   (extreme)

The derivative is LARGEST at output = 0.5 (maximum uncertainty) and approaches
ZERO as the output moves toward 0 or 1 (high confidence).

**What this means for training:**
    • A neuron firing near 0.5 ("I'm unsure") gets LARGER weight updates.
      It needs more adjustment because it's genuinely confused.
    • A neuron firing near 0 or 1 ("I'm confident") gets SMALLER weight updates.
      It already has a strong, committed opinion.

This is a natural regularization effect built directly into the sigmoid function.
However, it also causes a problem called the **Vanishing Gradient Problem**.


### The Vanishing Gradient Problem

In deep networks (many layers), the sigmoid derivative is always ≤ 0.25.
At each layer, the gradient is multiplied by this derivative.
After just 4 layers:

    0.25 × 0.25 × 0.25 × 0.25 = 0.0039

After 8 layers:  0.25^8 ≈ 0.000015
After 12 layers: 0.25^12 ≈ 0.000000006

The gradients shrink exponentially as they flow backward. By the time
they reach the first hidden layers, they are so tiny that the weights
barely change at all. The network effectively stops learning in the early layers.

This is called the **Vanishing Gradient Problem**, and it was the main barrier
to training deep networks in the 1990s and early 2000s.

**The Modern Solution: ReLU**

    ReLU (Rectified Linear Unit): f(z) = max(0, z)
    ReLU derivative: 1 if z > 0, else 0

    The ReLU derivative is either 1 (no shrinkage) or 0 (neuron is "dead").
    This prevents the exponential decay across layers.

    Most modern deep learning uses ReLU (or variants like LeakyReLU, GELU)
    instead of sigmoid for hidden layers precisely because of this.


---

## **Part 8: Batch Sizes, Epochs, and Learning Rate**

### Stochastic vs Batch vs Mini-Batch Gradient Descent

In a dataset of N examples, you have a choice of how to compute gradients:

    STOCHASTIC GRADIENT DESCENT (SGD):
        Update weights after EACH single example.
        Fast updates, noisy gradients.
        Good for escaping local minima (noise helps).

    BATCH GRADIENT DESCENT:
        Average gradients over ALL N examples, then update.
        Smooth gradients, but very slow for large datasets.
        One "step" requires processing the entire dataset.

    MINI-BATCH GRADIENT DESCENT (most common):
        Average gradients over a small batch (e.g. 32 or 64 examples).
        Balance of speed and smoothness.
        Efficient on GPU hardware (matrix operations are parallelized).

    Analogy:
        SGD = getting directions from one random stranger
        Batch = averaging directions from 10,000 people
        Mini-batch = averaging directions from 32 people nearby

### Epochs

An epoch is one full pass through the ENTIRE training dataset.

    Epoch 1: See all 1,000 examples, update weights 1,000 times (SGD)
    Epoch 2: See all 1,000 examples again (in different order), update again
    ...
    Epoch 100: By now, the network has seen each example 100 times

Training typically runs for tens to thousands of epochs depending on dataset size.
The loss should decrease monotonically (or at least trend downward).

### Learning Rate — The Most Sensitive Hyperparameter

    TOO HIGH (η = 10.0):   Weights overshoot the minimum and diverge.
                            Loss explodes → NaN values
                            ↑↑↑ huge jumps, bouncing chaotically

    TOO LOW  (η = 0.00001): Weights barely move. Training takes forever.
                             Could take millions of epochs to converge.
                             ↓↓↓ tiny steps, incredibly slow

    JUST RIGHT (η = 0.01): Weights smoothly descend toward the minimum.
                            Loss decreases steadily, convergence in reasonable time.

    TYPICAL RANGES:
        Raw SGD:    0.001 – 0.1
        Adam optimizer: 0.0001 – 0.001 (adaptive learning rates)

{{BACKPROP_IMAGE}}

---

## **Part 9: Numerical Verification (Gradient Checking)**

One classic technique to verify that a backpropagation implementation is
correct is **gradient checking** — computing numerical gradients and comparing
them to analytical gradients.

**Numerical gradient (finite difference approximation):**

    numerical_grad(w) ≈ [Loss(w + ε) - Loss(w - ε)] / (2ε)

    Where ε is a tiny number like 0.0001.

This is essentially asking: "If I nudge w by ε in each direction, how does
the loss change?" It's the definition of a derivative, computed numerically.

**Checking the implementation:**

    For each weight w in the network:
        1. Run a forward pass. Compute Loss(w).
        2. Add ε to w. Run forward pass. Get Loss(w + ε).
        3. Subtract ε from w. Run forward pass. Get Loss(w - ε).
        4. numerical_grad = (Loss(w + ε) - Loss(w - ε)) / (2ε)
        5. analytic_grad = gradient from backpropagation
        6. If |numerical_grad - analytic_grad| < 1e-5, the implementation is correct.

This technique is computationally expensive (requires 2 forward passes per weight)
so it's only used for debugging, not training. But it's a critical sanity check.

---

## **Part 10: Historical Context — Why Backpropagation Changed Everything**

### The AI Winter (1969–1986)

In 1969, Marvin Minsky and Seymour Papert published "Perceptrons," proving that
a single perceptron cannot learn XOR. This caused a dramatic loss of interest
(and funding) in neural network research.

The solution — multi-layer networks — was known in principle. But nobody knew
how to train them. Without a way to propagate error to hidden layers, you
couldn't credit or blame the hidden neurons, so you couldn't train them.

### The 1986 Breakthrough

In 1986, David Rumelhart, Geoffrey Hinton, and Ronald Williams published
"Learning representations by back-propagating errors" in Nature.

They showed that the chain rule of calculus could be applied systematically
to compute gradients for every weight in a multi-layer network simultaneously.
This was backpropagation.

The impact was immediate. Networks that had seemed untrainable suddenly worked.
XOR, non-linear classification, function approximation — all solvable.

### The Deep Learning Era (2006–Present)

Backpropagation is still the core learning algorithm in every neural network today.
GPT-4, DALL-E, AlphaFold, self-driving car models — all trained with backpropagation.

What changed over the decades:
    • Activation functions (ReLU replaced sigmoid for hidden layers → deeper networks)
    • Optimizers (Adam, RMSprop → adaptive learning rates)
    • Batch normalization (stabilizes training in very deep networks)
    • Skip connections (ResNets → gradients can flow past layers)
    • Larger datasets and faster hardware (GPUs → practical training)

But underneath every one of these advances, the chain rule is still computing
∂Loss/∂w for every weight in the network. Every modern AI system learns
by backpropagation.

---

### Summary of Key Equations

    ┌──────────────────────────────────────────────────────────────┐
    │  FORWARD PASS                                                │
    │  z   = Σ(w_i × input_i) + b       (weighted sum + bias)      │
    │  a   = σ(z) = 1/(1+e^-z)          (sigmoid activation)       │
    │  L   = (1/2)(y - ŷ)²              (mean squared error)       │
    ├──────────────────────────────────────────────────────────────┤
    │  BACKWARD PASS                                               │
    │  Output delta:  δ_out = (y - ŷ) × σ'(ŷ)                      │
    │  Hidden delta:  δ_h   = σ'(h) × Σ(δ_next × w_connecting)     │
    │  Sigmoid deriv: σ'(a) = a × (1 - a)                          │
    ├──────────────────────────────────────────────────────────────┤
    │  WEIGHT UPDATE                                               │
    │  w_new = w_old + η × δ × input_to_this_weight                │
    │  b_new = b_old + η × δ                                       │
    └──────────────────────────────────────────────────────────────┘
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """

| Concept                  | Simple Perceptron Update          | Backpropagation                                    |
|--------------------------|-----------------------------------|----------------------------------------------------|
| Error source             | Output only                       | Output, propagated backward through all layers     |
| Math required            | Arithmetic                        | Chain rule of calculus (partial derivatives)       |
| Hidden layer updates     | Not possible                      | Computed via downstream deltas × weights           |
| Number of passes         | 1 forward pass                    | 1 forward + 1 backward pass per example            |
| Weight update rule       | Δw = η × error × input            | Δw = η × δ × input (delta computed via chain rule) |
| Applicable to            | Linear problems only              | Any differentiable architecture                    |
| Gradient computation     | Direct                            | Recursive, layer by layer                          |
| Scalability              | Fixed single layer                | Scales to billions of parameters                   |
| Key risk                 | N/A                               | Vanishing / exploding gradients                    |
| Modern replacement of    | Step function                     | Sigmoid → ReLU in hidden layers (avoid vanishing)  |

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets / full runnable demos
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "Backpropagation from Scratch": {
        "description": "Full backpropagation implementation with step-by-step commentary",
        "runnable": True,
        "pipeline_cmd": "token",
        "code":
r'''
"""
================================================================================
BACKPROPAGATION FROM SCRATCH — FULL IMPLEMENTATION WITH COMMENTARY
================================================================================

This script implements backpropagation from the ground up with:
    1. Detailed forward pass (caching pre-activations and activations)
    2. Full backward pass with explicit chain rule application
    3. Live gradient checking (numerical vs analytical comparison)
    4. Training loop with loss visualization
    5. Solving XOR as a proof of correctness

Every calculation is printed in detail so you can see exactly what's happening.

Key insight: Backpropagation is just the CHAIN RULE applied recursively.
Every neuron learns how much it was responsible for the final error by
tracing the effect of its weights all the way to the loss.
================================================================================
"""

import math
import random


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def sigmoid(z):
    """
    σ(z) = 1 / (1 + e^(-z))

    Smooth, differentiable activation. Output is always in (0, 1).
    Key for backpropagation: the derivative only needs the output value,
    not the original z. This saves us from caching z separately.
    """
    z = max(-500, min(500, z))   # Prevent overflow
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(output):
    """
    σ'(z) = σ(z) × (1 - σ(z))

    We compute the derivative from the OUTPUT, not from z directly.
    This is a beautiful property of sigmoid — we already computed σ(z)
    in the forward pass and stored it. We don't need z again.

    The derivative is maximized at output=0.5 (most uncertainty)
    and approaches 0 at output=0 or output=1 (high confidence).

    This causes the VANISHING GRADIENT PROBLEM in deep networks, because
    each layer multiplies the gradient by a value ≤ 0.25.
    """
    return output * (1.0 - output)


def mse_loss(expected, actual):
    """
    Mean Squared Error: L = (1/2) × Σ(expected_i - actual_i)²

    The 1/2 factor is a convention — it cancels the 2 from the power rule
    when we differentiate, giving a clean derivative of (actual - expected).
    """
    total = 0.0
    for e, a in zip(expected, actual):
        total += (e - a) ** 2
    return 0.5 * total


# ==============================================================================
# NEURON CLASS — Stores state needed for both forward and backward passes
# ==============================================================================

class Neuron:
    """
    A single neuron that knows how to:
        1. Compute its output (forward pass)
        2. Store pre-activation (z) and activation (output) for backward use
        3. Receive a delta (error signal) and update its own weights
    """

    def __init__(self, num_inputs):
        """
        Initialize with random weights and zero bias.

        Why random weights? If all weights start at zero, every neuron in a
        layer computes the same output, receives the same gradient, and updates
        identically. They'd remain identical forever — the network effectively
        has only 1 neuron per layer. Random initialization "breaks symmetry."
        """
        # Xavier-inspired initialization: scale by 1/sqrt(num_inputs)
        # This keeps variance stable as signals pass through layers
        scale = 1.0 / math.sqrt(num_inputs)
        self.weights = [random.uniform(-scale, scale) for _ in range(num_inputs)]
        self.bias = 0.0

        # Cached values — set during forward pass, read during backward pass
        self.z      = 0.0    # Pre-activation: z = Σ(w_i × input_i) + b
        self.output = 0.0    # Post-activation: output = sigmoid(z)
        self.delta  = 0.0    # Error signal (set during backward pass)
        self.last_inputs = []  # The inputs this neuron received (for weight updates)

    def forward(self, inputs):
        """
        Forward pass: compute z = Σ(w × input) + b, then output = σ(z).

        We cache BOTH z and output because backward pass needs output (for
        the sigmoid derivative), and the weight update needs inputs.
        """
        self.last_inputs = inputs  # Cache inputs for weight update

        self.z = self.bias
        for w, x in zip(self.weights, inputs):
            self.z += w * x

        self.output = sigmoid(self.z)
        return self.output

    def update_weights(self, learning_rate):
        """
        Update this neuron's weights using its pre-computed delta.

        Rule:  w_new = w_old + η × δ × input_i
               b_new = b_old + η × δ

        This is gradient descent applied to each weight.
        δ encodes both the direction (sign) and magnitude of change.
        """
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * self.delta * self.last_inputs[i]
        self.bias += learning_rate * self.delta


# ==============================================================================
# LAYER CLASS
# ==============================================================================

class Layer:
    """
    A collection of neurons that all receive the same input and produce outputs.
    In a fully-connected (dense) layer, every neuron sees every input.
    """

    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def forward(self, inputs):
        """
        Run all neurons in this layer on the same inputs.
        Returns a list of outputs, one per neuron.
        """
        return [n.forward(inputs) for n in self.neurons]

    def update_weights(self, learning_rate):
        """
        Update all neurons' weights after deltas have been set.
        """
        for n in self.neurons:
            n.update_weights(learning_rate)


# ==============================================================================
# NEURAL NETWORK WITH EXPLICIT BACKPROPAGATION
# ==============================================================================

class NeuralNetwork:
    """
    Multi-layer neural network with explicit, step-by-step backpropagation.

    Architecture is defined by layer_sizes, e.g.:
        [2, 3, 1]  →  2 inputs, 3 hidden neurons, 1 output
        [2, 4, 4, 1]  →  2 inputs, two hidden layers of 4, 1 output
    """

    def __init__(self, layer_sizes, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.layers = []
        self.layer_sizes = layer_sizes

        # Build layers (skip the first entry — it's just the input size)
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i - 1]))

    # ------------------------------------------------------------------
    # FORWARD PASS
    # ------------------------------------------------------------------

    def forward(self, inputs):
        """
        Push data through the entire network, layer by layer.
        Each layer's output becomes the next layer's input.

        Returns the final output layer's activations.
        """
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    # ------------------------------------------------------------------
    # BACKWARD PASS — THE HEART OF THE ALGORITHM
    # ------------------------------------------------------------------

    def backward(self, expected):
        """
        Backpropagation: compute error signals (deltas) for every neuron,
        then update every weight in the network.

        PHASE 1 — Compute Deltas (going BACKWARD, right to left):

            Output Layer:
                δ_i = (expected_i - output_i) × σ'(output_i)

                "How wrong was I?" × "How sensitive am I to input changes?"
                The sigmoid derivative scales the correction by confidence.

            Hidden Layers:
                δ_j = σ'(output_j) × Σ_k(δ_k × w_jk)

                "How sensitive am I?" × "How much blame reaches me from downstream?"
                Each downstream neuron k sends back (its delta × the weight
                connecting j to k), because that weight determined j's influence.

        PHASE 2 — Update Weights (all layers, left to right or any order):

            w_ij_new = w_ij_old + η × δ_j × input_to_neuron_j
            b_j_new  = b_j_old  + η × δ_j
        """

        # ── PHASE 1: Compute Deltas ──────────────────────────────────────

        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]

            if layer_idx == len(self.layers) - 1:
                # OUTPUT LAYER: We know the expected values directly.
                for i, neuron in enumerate(layer.neurons):
                    error = expected[i] - neuron.output
                    # Chain rule: ∂Loss/∂z = ∂Loss/∂output × ∂output/∂z
                    #                      = -error           × σ'(output)
                    neuron.delta = error * sigmoid_derivative(neuron.output)

            else:
                # HIDDEN LAYER: Distribute blame from the next layer back.
                next_layer = self.layers[layer_idx + 1]
                for i, neuron in enumerate(layer.neurons):
                    # Sum up all error signals flowing back to this neuron.
                    # Each next-layer neuron k contributes:
                    #   δ_k × w[k][i]
                    # where w[k][i] is the weight FROM neuron i (this layer)
                    # TO neuron k (next layer).
                    downstream_error = sum(
                        next_neuron.delta * next_neuron.weights[i]
                        for next_neuron in next_layer.neurons
                    )
                    # Scale by how sensitive this neuron is to its own input
                    neuron.delta = downstream_error * sigmoid_derivative(neuron.output)

        # ── PHASE 2: Update Weights ──────────────────────────────────────

        for layer in self.layers:
            layer.update_weights(self.learning_rate)

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------

    def train(self, X, y, epochs=1000, verbose=True):
        """
        Full training loop: for each epoch, run every example through
        forward → loss → backward → update.
        """
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0.0

            for xi, yi in zip(X, y):
                outputs = self.forward(xi)
                total_loss += mse_loss(yi, outputs)
                self.backward(yi)

            avg_loss = total_loss / len(X)
            loss_history.append(avg_loss)

            if verbose and (epoch % (epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:5d} / {epochs}  |  Loss: {avg_loss:.6f}")

        return loss_history

    def predict(self, inputs):
        return self.forward(inputs)


# ==============================================================================
# DEMO 1: STEP-BY-STEP BACKPROPAGATION WALKTHROUGH
# ==============================================================================

def demo_step_by_step():
    """
    Trace ONE complete forward-and-backward pass with fixed weights,
    printing every intermediate value.

    Architecture: 2 inputs → 2 hidden neurons → 1 output
    Input: [0, 1], Expected: [1]
    """
    print("\n" + "=" * 65)
    print("  DEMO 1: STEP-BY-STEP BACKPROPAGATION WALKTHROUGH")
    print("=" * 65)
    print("""
  Architecture: [2 → 2 → 1]  (2 inputs, 2 hidden, 1 output)
  Training example: Input = [0, 1], Expected output = [1]
  Learning rate: 0.5
    """)

    # ── Fixed weights (same as the theory section) ──────────────────────
    input_data  = [0.0, 1.0]
    expected    = [1.0]
    lr          = 0.5

    # Hidden layer weights
    w1, w2, b1 = 0.15, 0.20, 0.35   # H1 weights from x1, x2
    w3, w4, b2 = 0.25, 0.30, 0.35   # H2 weights from x1, x2

    # Output layer weights
    w5, w6, b3 = 0.40, 0.45, 0.60   # O1 weights from H1, H2

    # ── FORWARD PASS ────────────────────────────────────────────────────
    print(f"  {'─' * 55}")
    print(f"  FORWARD PASS")
    print(f"  {'─' * 55}")

    z_h1 = w1 * input_data[0] + w2 * input_data[1] + b1
    h1   = sigmoid(z_h1)
    print(f"\n  Hidden Neuron 1 (H1):")
    print(f"    z_h1 = ({w1} × {input_data[0]}) + ({w2} × {input_data[1]}) + {b1}")
    print(f"         = {w1*input_data[0]:.4f} + {w2*input_data[1]:.4f} + {b1}")
    print(f"         = {z_h1:.4f}")
    print(f"    h1   = sigmoid({z_h1:.4f}) = {h1:.4f}")

    z_h2 = w3 * input_data[0] + w4 * input_data[1] + b2
    h2   = sigmoid(z_h2)
    print(f"\n  Hidden Neuron 2 (H2):")
    print(f"    z_h2 = ({w3} × {input_data[0]}) + ({w4} × {input_data[1]}) + {b2}")
    print(f"         = {w3*input_data[0]:.4f} + {w4*input_data[1]:.4f} + {b2}")
    print(f"         = {z_h2:.4f}")
    print(f"    h2   = sigmoid({z_h2:.4f}) = {h2:.4f}")

    z_o  = w5 * h1 + w6 * h2 + b3
    y_hat = sigmoid(z_o)
    print(f"\n  Output Neuron (O1):")
    print(f"    z_o  = ({w5} × {h1:.4f}) + ({w6} × {h2:.4f}) + {b3}")
    print(f"         = {w5*h1:.4f} + {w6*h2:.4f} + {b3}")
    print(f"         = {z_o:.4f}")
    print(f"    ŷ    = sigmoid({z_o:.4f}) = {y_hat:.4f}")

    loss = 0.5 * (expected[0] - y_hat) ** 2
    print(f"\n  Loss = (1/2) × ({expected[0]} - {y_hat:.4f})²")
    print(f"       = (1/2) × {(expected[0] - y_hat):.4f}²")
    print(f"       = {loss:.6f}")

    # ── BACKWARD PASS ───────────────────────────────────────────────────
    print(f"\n  {'─' * 55}")
    print(f"  BACKWARD PASS — Computing Deltas")
    print(f"  {'─' * 55}")

    # Output delta
    sd_o   = sigmoid_derivative(y_hat)
    delta_o = (expected[0] - y_hat) * sd_o
    print(f"\n  Output Layer Delta (δ_o):")
    print(f"    error         = expected - ŷ = {expected[0]} - {y_hat:.4f} = {expected[0]-y_hat:.4f}")
    print(f"    σ'(ŷ)         = ŷ × (1 - ŷ) = {y_hat:.4f} × {1-y_hat:.4f} = {sd_o:.4f}")
    print(f"    δ_o           = error × σ'(ŷ) = {expected[0]-y_hat:.4f} × {sd_o:.4f} = {delta_o:.5f}")

    # Gradients for output weights
    grad_w5 = delta_o * h1
    grad_w6 = delta_o * h2
    grad_b3 = delta_o
    print(f"\n  Output Weight Gradients:")
    print(f"    ∂L/∂w5 = δ_o × h1 = {delta_o:.5f} × {h1:.4f} = {grad_w5:.5f}")
    print(f"    ∂L/∂w6 = δ_o × h2 = {delta_o:.5f} × {h2:.4f} = {grad_w6:.5f}")
    print(f"    ∂L/∂b3 = δ_o       = {grad_b3:.5f}")

    # Hidden deltas
    sd_h1    = sigmoid_derivative(h1)
    sd_h2    = sigmoid_derivative(h2)
    error_h1 = delta_o * w5
    error_h2 = delta_o * w6
    delta_h1 = error_h1 * sd_h1
    delta_h2 = error_h2 * sd_h2

    print(f"\n  Hidden Layer Deltas:")
    print(f"    Backprop error to H1:")
    print(f"      = δ_o × w5 = {delta_o:.5f} × {w5} = {error_h1:.5f}")
    print(f"    σ'(h1) = h1 × (1 - h1) = {h1:.4f} × {1-h1:.4f} = {sd_h1:.4f}")
    print(f"    δ_h1   = {error_h1:.5f} × {sd_h1:.4f} = {delta_h1:.6f}")
    print(f"")
    print(f"    Backprop error to H2:")
    print(f"      = δ_o × w6 = {delta_o:.5f} × {w6} = {error_h2:.5f}")
    print(f"    σ'(h2) = h2 × (1 - h2) = {h2:.4f} × {1-h2:.4f} = {sd_h2:.4f}")
    print(f"    δ_h2   = {error_h2:.5f} × {sd_h2:.4f} = {delta_h2:.6f}")

    # Gradients for hidden weights
    grad_w1 = delta_h1 * input_data[0]
    grad_w2 = delta_h1 * input_data[1]
    grad_b1 = delta_h1
    grad_w3 = delta_h2 * input_data[0]
    grad_w4 = delta_h2 * input_data[1]
    grad_b2 = delta_h2

    print(f"\n  Hidden Weight Gradients:")
    print(f"    ∂L/∂w1 = δ_h1 × x1 = {delta_h1:.6f} × {input_data[0]} = {grad_w1:.6f}  (x1=0, so no update)")
    print(f"    ∂L/∂w2 = δ_h1 × x2 = {delta_h1:.6f} × {input_data[1]} = {grad_w2:.6f}")
    print(f"    ∂L/∂b1 = δ_h1       = {grad_b1:.6f}")
    print(f"    ∂L/∂w3 = δ_h2 × x1 = {delta_h2:.6f} × {input_data[0]} = {grad_w3:.6f}  (x1=0, so no update)")
    print(f"    ∂L/∂w4 = δ_h2 × x2 = {delta_h2:.6f} × {input_data[1]} = {grad_w4:.6f}")
    print(f"    ∂L/∂b2 = δ_h2       = {grad_b2:.6f}")

    # ── WEIGHT UPDATES ───────────────────────────────────────────────────
    print(f"\n  {'─' * 55}")
    print(f"  WEIGHT UPDATE  (new_w = old_w + η × gradient,  η = {lr})")
    print(f"  {'─' * 55}")

    new_w5 = w5 + lr * grad_w5
    new_w6 = w6 + lr * grad_w6
    new_b3 = b3 + lr * grad_b3
    new_w1 = w1 + lr * grad_w1
    new_w2 = w2 + lr * grad_w2
    new_b1 = b1 + lr * grad_b1
    new_w3 = w3 + lr * grad_w3
    new_w4 = w4 + lr * grad_w4
    new_b2 = b2 + lr * grad_b2

    print(f"\n  Output Neuron:")
    print(f"    w5: {w5:.5f} + {lr} × {grad_w5:.5f} = {new_w5:.5f}")
    print(f"    w6: {w6:.5f} + {lr} × {grad_w6:.5f} = {new_w6:.5f}")
    print(f"    b3: {b3:.5f} + {lr} × {grad_b3:.5f} = {new_b3:.5f}")

    print(f"\n  Hidden Neuron 1:")
    print(f"    w1: {w1:.5f} + {lr} × {grad_w1:.6f} = {new_w1:.5f}  (unchanged)")
    print(f"    w2: {w2:.5f} + {lr} × {grad_w2:.6f} = {new_w2:.5f}")
    print(f"    b1: {b1:.5f} + {lr} × {grad_b1:.6f} = {new_b1:.5f}")

    print(f"\n  Hidden Neuron 2:")
    print(f"    w3: {w3:.5f} + {lr} × {grad_w3:.6f} = {new_w3:.5f}  (unchanged)")
    print(f"    w4: {w4:.5f} + {lr} × {grad_w4:.6f} = {new_w4:.5f}")
    print(f"    b2: {b2:.5f} + {lr} × {grad_b2:.6f} = {new_b2:.5f}")

    # Verify improvement
    z_h1_new = new_w1 * input_data[0] + new_w2 * input_data[1] + new_b1
    h1_new   = sigmoid(z_h1_new)
    z_h2_new = new_w3 * input_data[0] + new_w4 * input_data[1] + new_b2
    h2_new   = sigmoid(z_h2_new)
    z_o_new  = new_w5 * h1_new + new_w6 * h2_new + new_b3
    y_hat_new = sigmoid(z_o_new)
    loss_new = 0.5 * (expected[0] - y_hat_new) ** 2

    print(f"\n  {'─' * 55}")
    print(f"  VERIFICATION — did loss improve?")
    print(f"  {'─' * 55}")
    print(f"    Before: ŷ = {y_hat:.6f}, Loss = {loss:.6f}")
    print(f"    After:  ŷ = {y_hat_new:.6f}, Loss = {loss_new:.6f}")
    improvement = (loss - loss_new) / loss * 100
    print(f"    Loss reduced by {improvement:.2f}%  ✓" if loss_new < loss else f"    Loss did not improve ✗")


# ==============================================================================
# DEMO 2: GRADIENT CHECKING — Verify our backprop is correct
# ==============================================================================

def demo_gradient_checking():
    """
    Numerical gradient checking: compare analytical gradients (from backprop)
    to numerical gradients (from finite differences).

    If they match within ~1e-5, the implementation is correct.
    """
    print("\n" + "=" * 65)
    print("  DEMO 2: GRADIENT CHECKING (Verifying Backprop is Correct)")
    print("=" * 65)
    print("""
  Gradient checking is the standard technique for verifying that a
  backpropagation implementation computes correct gradients.

  Method: For each weight w, compute:
      Numerical:   [Loss(w+ε) - Loss(w-ε)] / (2ε)
      Analytical:  gradient from backprop

  If |numerical - analytical| < 1e-5, the implementation is correct.
    """)

    epsilon = 1e-4
    random.seed(42)
    net = NeuralNetwork([2, 3, 1], learning_rate=0.0)  # lr=0 so weights don't change

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0],    [1],    [1],    [0]]

    xi, yi = X[1], y[1]   # Use [0, 1] → expected [1]

    # Compute analytical gradients via backprop
    net.forward(xi)
    net.backward(yi)

    print(f"  Input: {xi}, Expected: {yi}")
    print(f"  Checking gradients for all weights:\n")
    print(f"  {'Weight':<15} {'Analytical':>15} {'Numerical':>15} {'Match?':>10}")
    print(f"  {'-' * 58}")

    all_pass = True

    for li, layer in enumerate(net.layers):
        for ni, neuron in enumerate(layer.neurons):
            # Check each weight
            for wi in range(len(neuron.weights)):
                # Analytical gradient of the LOSS w.r.t. this weight.
                # The update rule is:  w += lr * delta * input
                # which means:  w -= lr * (-delta * input)
                # so:  ∂L/∂w = -delta * input
                analytic = -neuron.delta * neuron.last_inputs[wi]

                # Numerical gradient via finite differences:
                # [L(w+ε) - L(w-ε)] / (2ε)  ≈  ∂L/∂w
                original = neuron.weights[wi]

                neuron.weights[wi] = original + epsilon
                out_plus = net.forward(xi)
                loss_plus = mse_loss(yi, out_plus)

                neuron.weights[wi] = original - epsilon
                out_minus = net.forward(xi)
                loss_minus = mse_loss(yi, out_minus)

                neuron.weights[wi] = original  # Restore

                numerical = (loss_plus - loss_minus) / (2 * epsilon)

                diff = abs(analytic - numerical)
                match = "✓" if diff < 1e-4 else "✗"
                if diff >= 1e-4:
                    all_pass = False

                label = f"Layer{li+1}_N{ni+1}_w{wi+1}"
                print(f"  {label:<15} {analytic:>15.8f} {numerical:>15.8f} {match:>10}")

            # Also check bias (∂L/∂b = -delta)
            analytic_bias = -neuron.delta

            original = neuron.bias
            neuron.bias = original + epsilon
            out_plus  = net.forward(xi); loss_plus  = mse_loss(yi, out_plus)
            neuron.bias = original - epsilon
            out_minus = net.forward(xi); loss_minus = mse_loss(yi, out_minus)
            neuron.bias = original
            numerical_bias = (loss_plus - loss_minus) / (2 * epsilon)

            diff = abs(analytic_bias - numerical_bias)
            match = "✓" if diff < 1e-4 else "✗"
            if diff >= 1e-4:
                all_pass = False

            label = f"Layer{li+1}_N{ni+1}_bias"
            print(f"  {label:<15} {analytic_bias:>15.8f} {numerical_bias:>15.8f} {match:>10}")

    print(f"\n  {'All gradients match! Implementation is correct.' if all_pass else 'Some gradients do NOT match — bug in implementation!'}")


# ==============================================================================
# DEMO 3: TRAIN A NETWORK ON XOR
# ==============================================================================

def demo_xor():
    """
    Train a network to solve XOR. This is the canonical test of whether
    backpropagation is working — a single perceptron cannot solve XOR,
    but a network with one hidden layer can.
    """
    print("\n" + "=" * 65)
    print("  DEMO 3: TRAINING ON XOR WITH BACKPROPAGATION")
    print("=" * 65)
    print("""
  XOR cannot be solved by a single perceptron because it is not
  linearly separable. A 2-layer network with backpropagation can learn it.

  Architecture: [2 inputs → 4 hidden → 1 output]
  Activation: Sigmoid
  Loss: Mean Squared Error
  Optimizer: SGD (Stochastic Gradient Descent)
    """)

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0],    [1],    [1],    [0]]

    random.seed(42)
    net = NeuralNetwork([2, 4, 1], learning_rate=1.0)

    print(f"  --- Training ---")
    loss_history = net.train(X, y, epochs=5000, verbose=True)

    print(f"\n  --- Final Predictions ---")
    print(f"  {'Input':<10} {'Expected':<12} {'Output':<15} {'Rounded':<10} {'Correct?'}")
    print(f"  {'-' * 55}")

    all_correct = True
    for xi, yi in zip(X, y):
        output = net.predict(xi)
        raw = output[0]
        rounded = round(raw)
        correct = "✓" if rounded == yi[0] else "✗"
        if rounded != yi[0]:
            all_correct = False
        print(f"  {str(xi):<10} {yi[0]:<12} {raw:<15.6f} {rounded:<10} {correct}")

    print(f"\n  Final loss: {loss_history[-1]:.8f}")
    if all_correct:
        print(f"  Network correctly learned XOR via backpropagation! ✓")
    else:
        print(f"  Training not complete — try increasing epochs or adjusting lr.")


# ==============================================================================
# DEMO 4: VANISHING GRADIENT VISUALIZATION
# ==============================================================================

def demo_vanishing_gradient():
    """
    Demonstrate the vanishing gradient problem by showing how gradients
    shrink as they flow backward through multiple sigmoid layers.
    """
    print("\n" + "=" * 65)
    print("  DEMO 4: VANISHING GRADIENT PROBLEM")
    print("=" * 65)
    print("""
  As gradients flow backward through sigmoid activations, they are
  multiplied by the sigmoid derivative at each layer.

  The maximum value of sigmoid_derivative is 0.25 (at output = 0.5).
  After each layer, the gradient can shrink by up to 4x.

  Let's see what happens to a gradient after each layer:
    """)

    # Simulate gradient magnitude through N layers
    initial_gradient = 1.0
    print(f"  Starting gradient magnitude: {initial_gradient}")
    print(f"\n  {'Layer':>7}  {'Sigmoid Deriv':>14}  {'Gradient After':>15}  {'% of Original':>14}")
    print(f"  {'─' * 58}")

    g = initial_gradient
    for depth in range(1, 13):
        # Use the worst-case sigmoid derivative (0.25, when output = 0.5)
        sd = 0.25
        g = g * sd
        pct = (g / initial_gradient) * 100
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"  {depth:>7}  {sd:>14.4f}  {g:>15.10f}  {pct:>12.6f}%  {bar}")

    print(f"""
  After just 12 layers, the gradient is {0.25**12:.2e} of its original value.
  This is the VANISHING GRADIENT PROBLEM — early layers stop learning.

  ReLU (Rectified Linear Unit) solves this:
    f(z) = max(0, z)
    f'(z) = 1 if z > 0, else 0

  The gradient passes through unchanged (multiplied by 1) for active neurons,
  so deep networks can be trained effectively. This is why every modern
  deep learning architecture uses ReLU or its variants (LeakyReLU, GELU)
  in hidden layers.
    """)


# ==============================================================================
# RUN ALL DEMONSTRATIONS
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  BACKPROPAGATION — THE ALGORITHM THAT TEACHES NEURAL NETWORKS")
    print("=" * 65)
    print("""
  Backpropagation solves the "credit assignment problem":
  how do we know which weights in a deep network are responsible
  for an error, and by exactly how much should each one change?

  The answer: the chain rule of calculus, applied layer by layer,
  flowing error signals backward from output to input.

  This script covers:
    1. Step-by-step hand-traced backpropagation (with exact numbers)
    2. Gradient checking (verifying the implementation is correct)
    3. Training a network on XOR (proof that backprop works)
    4. Vanishing gradient visualization (the key limitation of sigmoid)
    """)

    demo_step_by_step()
    demo_gradient_checking()
    demo_xor()
    demo_vanishing_gradient()

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print(f"""
  1. CREDIT ASSIGNMENT: Backpropagation tells every weight in the
     network exactly how much it was responsible for the final error.

  2. CHAIN RULE: The mathematical engine of backpropagation.
     Total effect = product of partial effects along the path.

  3. TWO PHASES: Forward pass computes predictions (no updates).
     Backward pass propagates error and updates weights.

  4. DELTA = ERROR SIGNAL: Output layer deltas are computed from
     (expected - actual) × σ'(output). Hidden layer deltas come from
     the downstream deltas × connecting weights × σ'(output).

  5. SIGMOID DERIVATIVE = CONFIDENCE SCALER: Large at 0.5 (uncertain),
     near zero at 0 or 1 (confident). Causes vanishing gradients in
     deep networks — this is why ReLU replaced sigmoid in hidden layers.

  6. GRADIENT CHECKING: The standard way to verify your backprop
     is correct. Numerical gradients must match analytical gradients.

  7. THIS ALGORITHM RUNS EVERYTHING: GPT-4, DALL-E, AlphaFold,
     self-driving cars — all trained with backpropagation and gradient
     descent. The chain rule invented modern AI.
    """)
'''
    },
}


# # ─────────────────────────────────────────────────────────────────────────────
# # CONTENT EXPORT
# # ─────────────────────────────────────────────────────────────────────────────
#
# def get_content():
#     """Return all content for this topic module."""
#     from pathlib import Path
#
#     # Optional: drop in a backprop diagram image (same pattern as MLP module)
#     png_path = Path(__file__).parent.parent / "Required_Images" / "Backpropagation_Breakdown.png"
#     backprop_img = _image_to_html(
#         str(png_path),
#         alt="Backpropagation Flow Diagram",
#         width="80%"
#     )
#     theory_with_images = THEORY.replace("{{BACKPROP_IMAGE}}", backprop_img)
#
#     return {
#         "theory":      theory_with_images,
#         "theory_raw":  THEORY,
#         "complexity":  COMPLEXITY,
#         "operations":  OPERATIONS,
#     }


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
    """Return all content for this topic module."""
    from pathlib import Path
    from Deep_Learning.visuals.backpropagation_visual import BACKPROP_VISUAL_HTML, BACKPROP_VISUAL_HEIGHT

    # Build an absolute path to the PNG so it works regardless of the
    # working directory Streamlit is launched from.
    png_path = Path(__file__).parent.parent / "visuals" / "MultiLayerPreceptron_Breakdown.png"
    mlp_img = _image_to_html(
        str(png_path),
        alt="Multilayer Perceptron Architecture",
        width="80%"
    )
    theory_with_images = THEORY.replace("{{MLP_IMAGE}}", mlp_img)

    return {
        "theory": theory_with_images,       # Theory tab — inline PNG
        "theory_raw": THEORY,
        # Keys that app.py's "🎨 Visual Breakdown" tab reads
        "visual_html": BACKPROP_VISUAL_HTML,        # Interactive XOR component
        "visual_height": BACKPROP_VISUAL_HEIGHT,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }