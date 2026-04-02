"""
Perceptron - The simplest neural network unit
============================================

A Single Cell Neuron is the simplest type of artificial neuron and the fundamental building block of neural networks.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME = "Single Cell Neuron: Perceptron"
DISPLAY_NAME = "02 · Perceptron"
ICON         = "🚨"
SUBTITLE     = "Workings of a Single Neuron - Perceptron"


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

### What is a Perceptron?

A perceptron is the simplest type of artificial neuron and the fundamental building block of neural networks. 
It is inspired by the way biological neurons process information in the brain. At its core, a perceptron takes 
multiple input signals, processes them through a simple mathematical function, and produces a single output.

Think of it like a tiny decision-maker: it receives evidence (inputs), weighs how important each piece of evidence is (weights), 
and then makes a yes-or-no decision (output).

The Biological Analogy
A biological neuron receives electrical signals through its dendrites, processes them in the cell body (soma), 
and if the combined signal is strong enough, it "fires" an output signal through its axon. The perceptron mimics this:

    •	Inputs:   (x₁, x₂, ..., xₙ) - analogous to dendrites receiving signals
    
    •	Weights:  (w₁, w₂, ..., wₙ) - analogous to the strength of synaptic connections
    
    •	Summation function: - analogous to the cell body aggregating signals
    
    •	Activation function: - analogous to the threshold that determines whether the neuron fires
    
    •	Output: - analogous to the axon transmitting a signal


Things that exist inside the network (learnable parameters):

    - Weights — strength of connections between neurons, updated during training
    
    - Bias — constant that shifts the decision boundary, updated during training

Things that exist only at setup time (hyperparameters / configuration):

    - Initialization variance — controls the spread of random values assigned to weights before training starts, never stored or updated
    
    - Learning rate — controls how much weights change per update step
    
    - Number of layers, neurons per layer, activation function choices, etc.


### The Math Behind It
## The perceptron computes its output in two steps:


Step 1 — Weighted Sum:

    z = Σ(wᵢ · xᵢ) + b

Step 2 — Activation:

    y = activation(z)   


** ------------ Detailed Explanation ------------ **

### Step 1 — Weighted Sum:

                Z = (w1⋅x1) + (w2⋅x2) +⋯+ (wn⋅xn) + b

Or more compactly: 

                z = W · X + b

Where:
- X is the input vector
- W is the weight vector
- b is the bias term (a constant that shifts the decision boundary, similar to the y-intercept in a linear equation)


### ** Understanding the Dot Product Intuitively: **

The operation W · X is a dot product, and it has a powerful geometric meaning: 
it measures how aligned two vectors are. When the input vector X points in roughly the same direction as the weight vector W, 
the dot product is large and positive — the neuron is more likely to fire. When they are orthogonal (at 90° to each other), 
the dot product is zero. When they point in opposite directions, it's negative.

This gives us a beautiful intuition for what a perceptron is really doing: 
it's asking "how similar is this input to the pattern I've learned?" The weight vector W encodes the ideal pattern, 
and the dot product measures how closely the current input matches it.
Consider a simple example with two inputs. If the perceptron has learned weights W = [1, 1], 
it's looking for inputs where both features are high.

An input like X = [1, 1] produces a large dot product (1×1 + 1×1 = 2), while X = [-1, -1] produces a strongly negative one (1×(-1) + 1×(-1) = -2).

The perceptron is essentially a pattern matcher.

    # ======================================================================================= # 
    **Diagram 1 — Vector Alignment and Dot Product Value:**

    DOT PRODUCT: How "aligned" are W and X?

    W · X = |W| × |X| × cos(θ)     where θ = angle between them


    CASE 1: Same Direction (θ = 0°)          CASE 2: Perpendicular (θ = 90°)
    cos(0°) = 1 → LARGE POSITIVE             cos(90°) = 0 → ZERO

                    W                                       W
                ↗                                        ↑
              ↗                                          |
            ↗   X                                        |
          ↗   ↗                                          |
        ↗   ↗                                            |
      ↗   ↗                                              +--------→ X

                                                    "No alignment at all"
    "Strongly aligned"                               Neuron is indifferent
    Neuron fires! ✅                                    Output ≈ 0 ⚠️


    CASE 3: Opposite Direction (θ = 180°)     CASE 4: Partially Aligned (θ = 45°)
    cos(180°) = -1 → LARGE NEGATIVE           cos(45°) ≈ 0.71 → MODERATE POSITIVE

        W                                          W
            ↗                                          ↗
          ↗                                          ↗
        ↗                                          ↗
      ↗                                          ↗
    ↗                                          ↗---------------→ X
            ↙
              ↙  X                            "Partially aligned"
                ↙                               Neuron might fire
    "Completely opposed"                        depending on bias
    Neuron stays silent ❌


    **Diagram 2 — Concrete Example with W = [1, 1]:**

    PERCEPTRON PATTERN MATCHING
    ══════════════════════════════════════════════════════════
    Weight Vector W = [1, 1]  →  "I'm looking for BOTH features to be high"

                        x₂
                        ↑
                        |
            X=[-1,1]    |  X=[1,1]
            W·X = 0     |  W·X = 2 ✅ STRONG MATCH
            ⚠️ Neutral  |  ★ Fires!
                        |          ↗ W = [1,1]
    ────────────────────+────────↗──────────→ x₁
                        |      ↗
            X=[-1,-1]   |  X=[1,-1]
            W·X = -2    |  W·X = 0
            ❌ Opposed  |  ⚠️ Neutral
                        |
                        |
                        |

    ─────────────────────────────────────────────────────────────
    Input X        Dot Product W·X      Interpretation
    ─────────────────────────────────────────────────────────────
    [ 1,  1]       1(1)+1(1) =  2       Same direction → FIRE
    [ 1, -1]       1(1)+1(-1)=  0       Perpendicular  → MAYBE
    [-1,  1]       1(-1)+1(1)=  0       Perpendicular  → MAYBE
    [-1, -1]       1(-1)+1(-1)= -2      Opposite       → SILENT
    ─────────────────────────────────────────────────────────────

    **Diagram 3 — The Pattern Matching Analogy:**
    
    THE PERCEPTRON AS A PATTERN MATCHER
    ════════════════════════════════════

    The weight vector W is the "ideal pattern" the neuron has learned.
    The dot product asks: "How much does this input look like my pattern?"


        Learned Pattern          Input Signal           Match Score
        (Weight Vector W)        (Input Vector X)       (Dot Product)
    ┌─────────────┐         ┌─────────────┐
    │ w₁ = 0.8    │         │ x₁ = 0.9    │
    │ w₂ = 0.6    │    ·    │ x₂ = 0.7    │    =    W · X
    │ w₃ = -0.2   │         │ x₃ = 0.1    │
    └─────────────┘         └─────────────┘

    z = (0.8)(0.9) + (0.6)(0.7) + (-0.2)(0.1)
        =   0.72     +   0.42     +  -0.02
        =   1.12  ← High score! Input matches the learned pattern.


    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  HIGH W·X  ──→  "This input matches my pattern!"  ──→ 🔥 │
    │                                                          │
    │  ZERO W·X  ──→  "This input is irrelevant to me"  ──→ 😐 │
    │                                                          │
    │  LOW  W·X  ──→  "This is the OPPOSITE of my pattern"→ ❄️ │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    **Diagram 4 — Full Pipeline (Dot Product → Bias → Activation):**

    FULL PERCEPTRON PIPELINE
    ════════════════════════════════════════════════════════════

    Inputs X          Weights W         Dot Product    + Bias    Activation
    ┌──────┐         ┌──────┐
    │x₁=1.0├────×────┤w₁=0.7│──┐
    └──────┘         └──────┘  │
                               ├──► Σ (sum) ──►  z  ──►  step(z) ──► OUTPUT
    ┌──────┐         ┌──────┐  │      ▲
    │x₂=0.5├────×────┤w₂=0.3│──┘      │
    └──────┘         └──────┘      ┌──┴───┐
                                   │b=-0.5│
                                   └──────┘

    Step by step:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. MULTIPLY each input by its weight (element-wise)     │
    │    x₁×w₁ = 1.0 × 0.7 = 0.70                             │
    │    x₂×w₂ = 0.5 × 0.3 = 0.15                             │
    │                                                         │
    │ 2. SUM them up (this IS the dot product W·X)            │
    │    W·X = 0.70 + 0.15 = 0.85                             │
    │    #======================================#             │
    │    "How well does this input match my learned pattern?" │
    │                                                         │
    │ 3. ADD the bias                                         │
    │    z = 0.85 + (-0.5) = 0.35                             │
    │    #======================================#             │
    │    "Am I over my firing threshold?"                     │
    │                                                         │
    │ 4. ACTIVATE                                             │
    │    step(0.35) = 1  ← z ≥ 0, so neuron FIRES 🔥          │
    └─────────────────────────────────────────────────────────┘

    # ======================================================================================= # 

### Step 2 — Activation:

The weighted sum z is then passed through an activation function. In Rosenblatt's original perceptron, this was a simple step function:

                        output = 1 if z ≥ 0

                        output = 0 if z < 0

This means: if the combined weighted input exceeds a threshold, the perceptron "fires" (outputs 1); otherwise, it doesn't (outputs 0).

### The Role of the Bias:

The bias b is a crucial but often misunderstood component. 
Without a bias, the decision boundary of the perceptron must always pass through the origin. The bias allows the perceptron to shift its decision boundary, 
giving it much more flexibility. 

Think of it as how easily the neuron is "triggered" — a large positive bias means the neuron fires easily, 
while a large negative bias means it requires very strong input to fire.

### Bias as Threshold — A Historical Bridge

In Rosenblatt's original 1958 formulation, there was actually no bias term. Instead, there was an explicit threshold θ, 
and the neuron fired if W · X ≥ θ. The modern formulation rewrites this as W · X + b ≥ 0, where b = -θ. 
These are mathematically identical: 

                    W · X + b ≥ 0 is the same as W · X ≥ -b is the same as W · X ≥ θ

So the bias is simply the negative of the threshold. 
This is worth knowing because you'll encounter both formulations in textbooks and papers. 

The bias formulation is preferred today because it lets us treat the bias as just another learnable parameter, simplifying the math and code.

### What Can a Single Perceptron Do?

A single perceptron is a linear binary classifier. 

It can learn to separate data into two classes as long as the data is linearly separable — 
    meaning you can draw a straight line (or hyperplane in higher dimensions) between the two classes.

**A single perceptron can learn simple logical functions like AND and OR:**

### The Geometric Picture — Decision Boundaries:

This is easier to see than you might think. 
The equation w₁x₁ + w₂x₂ + b = 0 is literally the equation of a straight line in 2D space. 
Everything the perceptron does comes down to this line:

The weights w₁ and w₂ determine the orientation (slope) of the line. Specifically, 
the weight vector **W = [w₁, w₂]** is perpendicular to the decision boundary line, pointing toward the "positive" region.

The bias b determines where the line sits — how far from the origin it is shifted.

Everything on one side of the line (where **w₁x₁ + w₂x₂ + b ≥ 0**) gets classified as 1.
Everything on the other side (where **w₁x₁ + w₂x₂ + b < 0**) gets classified as 0.

In higher dimensions, this line becomes a plane (3D) or a hyperplane (4D+), but the principle is identical: the perceptron carves the input space in half with a flat surface.


A Single Perceptron Can Learn AND and OR:

AND Gate (both inputs must be 1):

                                AND  Gate : (Both must be active)    

                                x₁	x₂	->  Output
                                --------------------
                                0	0	    0
                                0	1	    0
                                1	0	    0
                                1	1	    1



    A perceptron with weights w₁=1, w₂=1, and bias b=-1.5 solves this perfectly.

**Concrete Walkthrough — Proving the AND Gate Works:**

    Let's verify every input by hand to make the formula tangible:
        Input (0, 0): z = (1×0) + (1×0) + (-1.5) = -1.5 → z < 0 → output = 0 ✓
        Input (0, 1): z = (1×0) + (1×1) + (-1.5) = -0.5 → z < 0 → output = 0 ✓
        Input (1, 0): z = (1×1) + (1×0) + (-1.5) = -0.5 → z < 0 → output = 0 ✓
        Input (1, 1): z = (1×1) + (1×1) + (-1.5) = 0.5 → z ≥ 0 → output = 1 ✓
    All four cases produce the correct AND output. Geometrically, 

the decision boundary is the line x₁ + x₂ - 1.5 = 0, which is the line x₂ = -x₁ + 1.5. If you plot the four input points on a 2D grid, 
this line passes between (1,1) on one side and the other three points on the other side.

                                    OR Gate
                                (At least one input must be 1)

                                x₁   x₂   →   Output
                                --------------------
                                0    0        0
                                0    1        1
                                1    0        1
                                1    1        1

    A perceptron with weights w₁=1, w₂=1, and bias b=-0.5 solves this:
        Input (0, 0): z = (1×0) + (1×0) + (-0.5) = -0.5 → output = 0 ✓
        Input (0, 1): z = (1×0) + (1×1) + (-0.5) = 0.5 → output = 1 ✓
        Input (1, 0): z = (1×1) + (1×0) + (-0.5) = 0.5 → output = 1 ✓
        Input (1, 1): z = (1×1) + (1×1) + (-0.5) = 1.5 → output = 1 ✓

Notice how only the bias changed between AND and OR. 
Both use w₁=1, w₂=1, but AND uses b=-1.5 (harder to fire — needs both inputs) while OR uses b=-0.5 (easier to fire — needs just one). 
This perfectly illustrates the role of the bias as a "trigger sensitivity" knob.

### Part 2: Learning and Training

**The Perceptron Learning Algorithm**
The perceptron doesn't come pre-loaded with the correct weights — it learns them from data. The training algorithm is elegantly simple:

1) Initialize weights and bias to small random values (or zeros).
2) For each training example (x, y_true):

        Compute the output: 

                            ŷ = activation(W · X + b)

        Calculate the error: 

                            error = y_true - ŷ

        Update the weights: 

                            wᵢ = wᵢ + η · error · xᵢ

        Update the bias: 
                            b = b + η · error

3) Repeat until the error is zero or a maximum number of iterations is reached.

Here, η (eta) is the learning rate, a small positive number (e.g., 0.01) that controls how much the weights are adjusted at each step. 
Too large and the model overshoots; too small and learning is painfully slow.

**Intuition Behind the Learning Rate**

Think of it this way: imagine you're blindfolded, standing on a hilly landscape, trying to find the lowest valley. 
Each step you take is guided only by the slope of the ground under your feet. The learning rate is your step size.

**Too large (e.g., η = 10): You take massive leaps.**

    You might jump right over the valley and land on the opposite hill, then leap back, oscillating wildly and never settling down.

**Too small (e.g., η = 0.0001): You take tiny, cautious steps. **

    You're definitely heading in the right direction, but it might take thousands of steps to get anywhere useful.

**Just right (e.g., η = 0.01 to 0.1): **

    You make steady progress toward the valley and converge in a reasonable number of steps.

Note: Unlike gradient descent in deeper networks where the learning rate interacts with smooth, continuous gradients, 
the perceptron learning rule makes discrete corrections     
    — the error is always -1, 0, or +1. So the learning rate here simply scales the size of each correction.

**The key insight is that the perceptron only updates when it makes a mistake.**
**If the prediction is correct, nothing changes. If it's wrong, the weights are nudged in the direction that would have produced the correct answer.**

### Why the Update Rule Works — Assigning Blame
If the prediction is correct, nothing changes. If it's wrong, the weights are nudged in the direction that would have produced the correct answer.
But the update rule wᵢ = wᵢ + η · error · xᵢ is doing something cleverly specific. 

Let's break down why it works:

1) Case 1 — False Negative (predicted 0, should be 1, error = +1):
    The perceptron didn't fire when it should have. 
    The fix: increase the weights for inputs that were active (xᵢ > 0), so that next time the perceptron sees a similar input, 
    the weighted sum will be larger and more likely to cross the threshold. Inactive inputs (xᵢ = 0) contributed nothing to the mistake, 
    so their weights don't change.

2) Case 2 — False Positive (predicted 1, should be 0, error = -1):
    The perceptron fired when it shouldn't have. 
    The fix: decrease the weights for inputs that were active, so that next time, the weighted sum will be smaller. 
    Again, inputs that were zero had no role in the mistake and are left alone.

3) Case 3 — Correct prediction (error = 0):
    Nothing changes. Don't fix what isn't broken.
    This is an elegant form of credit assignment — blame for each mistake is distributed proportionally to how much each input contributed to that mistake. 
    An input that was strongly active (large xᵢ) gets a larger weight adjustment. An input that was zero gets no adjustment at all.


### Concrete Training Walkthrough
Let's train a perceptron to learn the AND function from scratch. We'll start with all zeros and use a learning rate of η = 0.1.

    Initial state: w₁ = 0, w₂ = 0, b = 0
    --- Epoch 1 (first pass through all training examples) ---

    Example 1: 

        x = (0, 0), 
        y_true = 0

        z = (0×0) + (0×0) + 0 = 0 → ŷ = step(0) = 1 (since z ≥ 0, z = 0 → 0)

        error = 0 - 1 = -1

    w₁ = 0 + 0.1 × (-1) × 0 = 0

    w₂ = 0 + 0.1 × (-1) × 0 = 0

    b = 0 + 0.1 × (-1) = -0.1

    State: w₁ = 0, w₂ = 0, b = -0.1
    
    Example 2: 

        x = (0, 1), 

        y_true = 0

        z = (0×0) + (0×1) + (-0.1) = -0.1 → ŷ = step(-0.1) = 0

    error = 0 - 0 = 0 → correct! No update.

    State: w₁ = 0, w₂ = 0, b = -0.1


    Example 3: 
        x = (1, 0), 

        y_true = 0

        z = (0×1) + (0×0) + (-0.1) = -0.1 → ŷ = step(-0.1) = 0

    error = 0 - 0 = 0 → correct! No update.

    State: w₁ = 0, w₂ = 0, b = -0.1


    Example 4: 
        x = (1, 1), 

        y_true = 1

        z = (0×1) + (0×1) + (-0.1) = -0.1 → ŷ = step(-0.1) = 0

    error = 1 - 0 = 1

    w₁ = 0 + 0.1 × 1 × 1 = 0.1

    w₂ = 0 + 0.1 × 1 × 1 = 0.1

    b = -0.1 + 0.1 × 1 = 0

    State: w₁ = 0.1, w₂ = 0.1, b = 0

    **--- End of Epoch 1 ---**

The weights are already starting to take shape — both w₁ and w₂ increased because both inputs were active when the perceptron should have fired. 
But the perceptron hasn't converged yet. After several more epochs, the weights will continue to adjust, eventually settling on values that correctly classify
all four AND inputs. 

The exact final values depend on the learning rate and initialization, but the perceptron is guaranteed to find a solution because AND is linearly separable.

Notice how informative even one epoch is: the perceptron made a mistake on (0,0) and on (1,1), and the weight updates from those mistakes pushed it 
in the right direction. The mistakes on the zero-input examples affected only the bias (because the inputs were zero), 
while the mistake on (1,1) affected the weights too. This is the credit assignment principle at work.


### Convergence Theorem

If the data is linearly separable, the perceptron learning algorithm is guaranteed to converge to a solution in a finite number of steps. 
This is a mathematically proven result. 

The proof relies on the fact that each update reduces the angle between the weight vector and the optimal solution vector, so the algorithm must eventually find it.

However, if the data is NOT linearly separable, the perceptron will never converge — it will oscillate forever, 
endlessly adjusting weights without finding a perfect boundary. This is another reason why the XOR problem was so devastating: 
researchers saw the perceptron flailing on it and concluded the entire approach was flawed. 
    Note: The Zero Initialization Problem

If every weight in a neural network is initialized to zero, every neuron in a given layer will compute the exact same output, 
receive the exact same gradient during backpropagation, and update in the exact same way. This is called the symmetry problem.
The neurons never differentiate from each other — they all learn the same thing forever. You essentially have a network with one effective neuron per layer, 
no matter how wide it is.

For a single perceptron, zero initialization is actually fine (as we showed in our walkthrough above) because there's only one neuron. 
The symmetry problem only arises when you have multiple neurons in the same layer. This is why random initialization is critical for multi-layer networks.

### Part 3: From Perceptron to Deep Learning

**The Multi-Layer Perceptron (MLP)**
    The limitation of a single perceptron (inability to solve non-linear problems like XOR) led to the development of the Multi-Layer Perceptron (MLP), 
    which stacks multiple layers of perceptrons together:

    * Input Layer — receives the raw data
    * Hidden Layer(s) — intermediate layers that learn increasingly abstract representations
    * Output Layer — produces the final prediction

    By combining multiple perceptrons in layers, the network can learn non-linear decision boundaries. 
    For example, XOR can be solved by a network with just one hidden layer containing two neurons.

**How XOR Gets Solved — Building Intuition**
    Here's the key insight for why layers solve the XOR problem. Remember, a single perceptron draws one line. 
    Two perceptrons in a hidden layer draw two lines. The output neuron then combines these, effectively saying "fire if you're on the correct side of BOTH lines."

For XOR, imagine two lines cutting the 2D input space:

    * Line 1 (Neuron A): roughly separates "at least one input is on" from "both are off"
    * Line 2 (Neuron B): roughly separates "at least one input is on" from "both are on"

The output neuron then computes something like "Neuron A fires AND Neuron B fires" — which captures the XOR region. 
Each hidden neuron does a simple linear split, but by combining them, the network carves out a non-linear region. 
This is the fundamental mechanism of all deep learning: simple linear operations, composed through layers and non-linearities, produce complex non-linear behavior.


**The Critical Shift: From Perceptron Rule to Backpropagation**
The perceptron learning rule (which we walked through above) works perfectly for a single layer — but it cannot train multi-layer networks. 

Why? 

Because the step function is not differentiable. At z = 0 it has an infinite slope; everywhere else the slope is zero. 
You can't compute how much to adjust the weights of hidden neurons because you can't propagate error gradients through a step function.

This is the real reason modern networks switched to smooth activation functions. It wasn't just about "better" functions — 
it was about making the entire network trainable via calculus. 
The backpropagation algorithm (popularized by Rumelhart, Hinton, and Williams in 1986) uses the chain rule of calculus to compute how much each weight in
every layer contributed to the final error, then adjusts them all simultaneously. 
This requires every function in the chain to be differentiable, which the step function is not.


This insight — that you need smooth, differentiable activation functions to train deep networks — is what truly ended the AI Winter, 
not just the idea of stacking layers.

### Modern Activation Functions

In deep learning, the step function is almost never used. Instead, modern networks use smooth, differentiable activation functions that enable 
gradient-based optimization (backpropagation):

Sigmoid: σ(z) = 1 / (1 + e⁻ᶻ) — outputs between 0 and 1. 

* You can think of it as a "soft" version of the step function: instead of a hard 0-or-1 jump, it provides a smooth S-shaped curve. 
Historically popular but suffers from vanishing gradients — when z is very large or very small, the gradient approaches zero, 
making learning extremely slow in deep networks.

Tanh: tanh(z) — outputs between -1 and 1, zero-centered. 
* Mathematically, tanh is just a rescaled sigmoid: 
tanh(z) = 2σ(2z) - 1. The zero-centering helps with gradient flow in practice. This only holds approximately / in a loose sense.

σ(z) = (1 + tanh(z/2)) / 2, or equivalently, tanh(z) = 2σ(2z) − 1

ReLU: max(0, z) — the most widely used today. 
* Stunningly simple: output z if positive, 0 if negative. 
It's computationally fast, doesn't saturate for positive values (no vanishing gradient on that side), and empirically works remarkably well. 
However, it has a "dying ReLU" problem: if a neuron's inputs consistently produce z < 0, the gradient is always zero and the neuron
permanently stops learning.

Leaky ReLU, GELU, Swish — modern variants that address ReLU's limitations. 
* Leaky ReLU allows a small negative slope (e.g., 0.01z for z < 0) so neurons never fully die.
* GELU (Gaussian Error Linear Unit) is used in Transformers and provides a smooth approximation of ReLU. 
* Swish (z × σ(z)) was discovered by neural architecture search and slightly outperforms ReLU in some deep networks.

From Perceptrons to Deep Networks
Every neuron in a modern deep neural network (CNNs, Transformers, etc.) is fundamentally a generalized perceptron: 
it computes a weighted sum of its inputs, adds a bias, and passes the result through a non-linear activation function. 
The magic of deep learning comes from stacking thousands or millions of these simple units in sophisticated architectures and 
training them end-to-end with backpropagation.

The progression is clear:

1) Perceptron (1958): One neuron, step activation, perceptron learning rule. Can only learn linearly separable patterns.

2) Multi-Layer Perceptron (1986): Multiple layers, sigmoid activation, backpropagation. Can learn any continuous function (universal approximation theorem).

3) Modern Deep Learning (2012+): Hundreds of layers, ReLU/GELU activation, sophisticated architectures (convolutions, attention, residual connections), 
trained on massive data with GPU acceleration.


But at their core, all of these are just perceptrons — organized, stacked, and trained with increasingly clever techniques. 
Understanding the single perceptron means understanding the atom from which all of deep learning is built.
    """

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
None
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "Perceptron Implementation": {
        "description": "Single Percepton Implementation",
        "runnable": True,
        "pipeline_cmd": "token",
        "code": '''
"""
================================================================================
SINGLE PERCEPTRON IMPLEMENTATION FROM SCRATCH
================================================================================

A perceptron is the simplest artificial neuron. It takes inputs, multiplies each
by a weight, adds a bias, and passes the result through an activation function
to produce an output.

Architecture:

    x1 ──(w1)──┐
               │
    x2 ──(w2)──┼──► [ Σ weighted sum + bias ] ──► [ activation ] ──► output
               │
    x3 ──(w3)──┘

Math:
    z = (x1*w1) + (x2*w2) + ... + (xn*wn) + bias
    output = activation(z)

This script demonstrates:
    1. How a perceptron is structured
    2. How it makes predictions (forward pass)
    3. How it learns from mistakes (training)
    4. Training it to learn AND, OR, and attempting XOR
================================================================================
"""

import random
import math


class Perceptron:
    """
    A single perceptron (artificial neuron).

    Parameters stored inside the perceptron:
        - self.weights : list of floats (one weight per input feature)
        - self.bias    : single float (shifts the decision boundary)

    These are the ONLY learnable parameters. They get updated during training.
    """

    def __init__(self, num_inputs, learning_rate=0.1, activation="step"):
        """
        Initialize the perceptron.

        Args:
            num_inputs (int):
                How many input features the perceptron will receive.
                For a 2-input logic gate (like AND), this would be 2.

            learning_rate (float):
                Controls how much the weights are adjusted after each mistake.
                - Too high (e.g., 10.0): overshoots the correct weights, unstable
                - Too low (e.g., 0.0001): learns very slowly
                - Typical range: 0.01 to 0.1
                This is a HYPERPARAMETER — you set it, the model doesn't learn it.

            activation (str):
                Which activation function to use: "step" or "sigmoid"
        """

        # =====================================================================
        # WEIGHT INITIALIZATION
        # =====================================================================
        # We initialize weights to small random values to BREAK SYMMETRY.
        # If all weights were 0, a multi-layer network would never differentiate
        # its neurons. For a single perceptron it doesn't matter as much,
        # but we follow best practice.
        #
        # Here we use simple random initialization between -0.5 and 0.5.
        # In deep learning, you'd use He or Xavier initialization instead.
        # =====================================================================
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]

        # =====================================================================
        # BIAS INITIALIZATION
        # =====================================================================
        # Bias is typically initialized to 0. It will be learned during training.
        # The bias allows the decision boundary to shift away from the origin.
        # Without it, the perceptron can only learn boundaries passing through
        # the origin, which is very limiting.
        # =====================================================================
        self.bias = 0.0

        # Store hyperparameters (not learned, set by us)
        self.learning_rate = learning_rate
        self.activation_type = activation

    # =========================================================================
    # ACTIVATION FUNCTIONS
    # =========================================================================
    # The activation function decides whether the neuron "fires" or not.
    # It takes the weighted sum (z) and produces the output.
    # =========================================================================

    def activation(self, z):
        """
        Apply the activation function to the weighted sum.

        Args:
            z (float): The weighted sum = (inputs · weights) + bias

        Returns:
            float: The perceptron's output
        """
        if self.activation_type == "step":
            # -----------------------------------------------------------------
            # STEP FUNCTION (Original Perceptron - Rosenblatt 1958)
            # -----------------------------------------------------------------
            # Output is binary: 1 if z >= 0, else 0
            # Simple but NOT differentiable (can't use gradient descent)
            # Used with the basic perceptron learning rule instead
            #
            #   Output
            #     1 |          ________
            #       |         |
            #     0 |_________|
            #       +---------+---------
            #                 0        z
            # -----------------------------------------------------------------
            return 1 if z >= 0 else 0

        elif self.activation_type == "sigmoid":
            # -----------------------------------------------------------------
            # SIGMOID FUNCTION
            # -----------------------------------------------------------------
            # Output is continuous between 0 and 1: σ(z) = 1 / (1 + e^(-z))
            # Smooth and differentiable (needed for gradient descent)
            # Historically important but causes vanishing gradients in deep nets
            #
            #   Output
            #   1.0 |              ___-------
            #       |           /
            #   0.5 |         /
            #       |       /
            #   0.0 |------
            #       +------|------|------
            #             -5     0     5   z
            # -----------------------------------------------------------------
            return 1 / (1 + math.exp(-z))

    # =========================================================================
    # FORWARD PASS (Making a Prediction)
    # =========================================================================

    def predict(self, inputs):
        """
        Compute the perceptron's output for given inputs.

        This is the FORWARD PASS — data flows forward through the neuron.

        Steps:
            1. Multiply each input by its corresponding weight
            2. Sum all the weighted inputs
            3. Add the bias
            4. Pass through the activation function

        Args:
            inputs (list): Input values [x1, x2, ..., xn]

        Returns:
            float: The perceptron's output (0 or 1 for step, 0.0-1.0 for sigmoid)
        """

        # Step 1 & 2: Compute the weighted sum
        # z = (x1 * w1) + (x2 * w2) + ... + (xn * wn)
        weighted_sum = 0.0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]

        # Step 3: Add the bias
        # z = weighted_sum + bias
        weighted_sum += self.bias

        # Step 4: Apply activation function
        output = self.activation(weighted_sum)

        return output

    # =========================================================================
    # TRAINING (Learning from Mistakes)
    # =========================================================================

    def train(self, training_data, labels, epochs=100):
        """
        Train the perceptron using the Perceptron Learning Rule.

        The perceptron only learns when it makes a MISTAKE. If the prediction
        is correct, weights stay the same. If wrong, weights are nudged in the
        direction that would have given the correct answer.

        Update rules:
            error = true_label - predicted_output

            For each weight:
                w_i = w_i + (learning_rate × error × x_i)

            For bias:
                bias = bias + (learning_rate × error)

        WHY THIS WORKS:
            - If error = 0 (correct prediction): nothing changes
            - If error = 1 (predicted 0, should be 1): weights INCREASE for
              active inputs, making it more likely to fire next time
            - If error = -1 (predicted 1, should be 0): weights DECREASE for
              active inputs, making it less likely to fire next time

        Args:
            training_data (list of lists): Each inner list is one training example
            labels (list): The correct output for each training example
            epochs (int): Number of complete passes through the training data

        Returns:
            list: History of total errors per epoch (for tracking convergence)
        """

        error_history = []

        for epoch in range(epochs):
            total_errors = 0

            # Go through each training example
            for inputs, true_label in zip(training_data, labels):

                # ----- Forward Pass -----
                prediction = self.predict(inputs)

                # ----- Calculate Error -----
                error = true_label - prediction

                # ----- Update Weights and Bias (only if there's an error) -----
                if error != 0:
                    total_errors += 1

                    # Update each weight
                    for i in range(len(self.weights)):
                        # w_i = w_i + (learning_rate × error × x_i)
                        #
                        # Intuition:
                        #   - error is positive (should have fired but didn't)
                        #     → increase weights for inputs that were active (x_i > 0)
                        #   - error is negative (fired but shouldn't have)
                        #     → decrease weights for inputs that were active
                        #   - x_i = 0 → that input had no influence, don't change its weight
                        self.weights[i] += self.learning_rate * error * inputs[i]

                    # Update bias (bias input is always 1, so: bias += lr * error * 1)
                    self.bias += self.learning_rate * error

            error_history.append(total_errors)

            # If no errors in this epoch, we've converged — stop early
            if total_errors == 0:
                print(f"  ✓ Converged at epoch {epoch + 1} with 0 errors!")
                break

        return error_history

    # =========================================================================
    # DISPLAY THE PERCEPTRON'S CURRENT STATE
    # =========================================================================

    def display_state(self):
        """Print the current weights and bias of the perceptron."""
        print(f"  Weights: {[round(w, 4) for w in self.weights]}")
        print(f"  Bias:    {round(self.bias, 4)}")


# =============================================================================
# =============================================================================
#                           DEMONSTRATIONS
# =============================================================================
# =============================================================================


def demonstrate_logic_gate(gate_name, training_data, labels):
    """
    Train a perceptron to learn a logic gate and display results.

    Args:
        gate_name (str): Name of the gate (AND, OR, XOR, etc.)
        training_data (list): Input combinations
        labels (list): Expected outputs
    """
    print(f"\\n{'=' * 60}")
    print(f"  TRAINING PERCEPTRON TO LEARN: {gate_name} GATE")
    print(f"{'=' * 60}")

    # Create a perceptron with 2 inputs (for 2-input logic gates)
    perceptron = Perceptron(num_inputs=2, learning_rate=0.1, activation="step")

    # Show initial random state
    print(f"\\n  --- Before Training (random weights) ---")
    perceptron.display_state()

    print(f"\\n  --- Training... ---")
    error_history = perceptron.train(training_data, labels, epochs=100)

    # Show learned state
    print(f"\\n  --- After Training (learned weights) ---")
    perceptron.display_state()

    # Test on all inputs and show results
    print(f"\\n  --- Results ---")
    print(f"  {'Input':<15} {'Expected':<12} {'Predicted':<12} {'Correct?'}")
    print(f"  {'-' * 50}")

    all_correct = True
    for inputs, expected in zip(training_data, labels):
        predicted = perceptron.predict(inputs)
        correct = "✓" if predicted == expected else "✗"
        if predicted != expected:
            all_correct = False
        print(f"  {str(inputs):<15} {expected:<12} {predicted:<12} {correct}")

    if all_correct:
        print(f"\\n  ✅ Perceptron successfully learned the {gate_name} gate!")
    else:
        print(f"\\n  ❌ Perceptron FAILED to learn the {gate_name} gate!")
        print(f"     This is expected — {gate_name} is NOT linearly separable.")
        print(f"     A single perceptron can only learn linearly separable patterns.")
        print(f"     You would need a Multi-Layer Perceptron (MLP) to solve this.")

    # Show how errors decreased over training
    print(f"\\n  --- Error History (first 10 epochs) ---")
    for i, errors in enumerate(error_history[:10]):
        bar = "█" * errors + "░" * (4 - errors)
        print(f"  Epoch {i + 1:>3}: {bar} ({errors} errors)")

    return perceptron


# =============================================================================
# TRAINING DATA FOR LOGIC GATES
# =============================================================================
# Each gate takes two binary inputs and produces one binary output.
#
# Truth Tables:
#
#   AND: Both inputs must be 1         OR: At least one input must be 1
#   [0,0] → 0                          [0,0] → 0
#   [0,1] → 0                          [0,1] → 1
#   [1,0] → 0                          [1,0] → 1
#   [1,1] → 1                          [1,1] → 1
#
#   NAND: NOT AND                       XOR: Exactly one input must be 1
#   [0,0] → 1                          [0,0] → 0
#   [0,1] → 1                          [0,1] → 1
#   [1,0] → 1                          [1,0] → 1
#   [1,1] → 0                          [1,1] → 0  ← This makes it non-linear!
# =============================================================================

# Input combinations (same for all gates)
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

# Labels for each gate
and_labels = [0, 0, 0, 1]
or_labels = [0, 1, 1, 1]
nand_labels = [1, 1, 1, 0]
xor_labels = [0, 1, 1, 0]  # NOT linearly separable!

# =============================================================================
# RUN THE DEMONSTRATIONS
# =============================================================================

if __name__ == "__main__":

    print("\\n" + "=" * 60)
    print("  PERCEPTRON: THE BUILDING BLOCK OF DEEP LEARNING")
    print("=" * 60)
    print("""
  This script demonstrates a single perceptron learning logic gates.

  Key Concepts:
    • Weights determine how much each input matters
    • Bias shifts the decision boundary
    • The perceptron learns by adjusting weights when it makes errors
    • It can ONLY learn linearly separable patterns
    """)

    # ---- Learnable Gates (linearly separable) ----
    demonstrate_logic_gate("AND", inputs, and_labels)
    demonstrate_logic_gate("OR", inputs, or_labels)
    demonstrate_logic_gate("NAND", inputs, nand_labels)

    # ---- Non-Learnable Gate (NOT linearly separable) ----
    demonstrate_logic_gate("XOR", inputs, xor_labels)

    # ==========================================================================
    # BONUS: STEP-BY-STEP WALKTHROUGH OF ONE TRAINING ITERATION
    # ==========================================================================
    print(f"\\n{'=' * 60}")
    print(f"  BONUS: STEP-BY-STEP WALKTHROUGH OF ONE PREDICTION + UPDATE")
    print(f"{'=' * 60}")

    print("""
  Let's manually trace what happens when a perceptron processes one example.

  Setup:
    - Input:         [1, 0]
    - True Label:    1  (this should be the OR gate output for [1, 0])
    - Weights:       [0.2, -0.1]  (some initial random values)
    - Bias:          0.0
    - Learning Rate: 0.1
    """)

    # Manual walkthrough
    x = [1, 0]
    true_label = 1
    weights = [0.2, -0.1]
    bias = 0.0
    lr = 0.1

    # Forward pass
    z = (x[0] * weights[0]) + (x[1] * weights[1]) + bias
    prediction = 1 if z >= 0 else 0
    error = true_label - prediction

    print(f"  FORWARD PASS:")
    print(f"    z = (x1 × w1) + (x2 × w2) + bias")
    print(f"    z = ({x[0]} × {weights[0]}) + ({x[1]} × {weights[1]}) + {bias}")
    print(f"    z = {x[0] * weights[0]} + {x[1] * weights[1]} + {bias}")
    print(f"    z = {z}")
    print(f"    activation(z) = step({z}) = {prediction}")
    print(f"    prediction = {prediction}")
    print()
    print(f"  ERROR CALCULATION:")
    print(f"    error = true_label - prediction = {true_label} - {prediction} = {error}")
    print()

    if error == 0:
        print(f"  WEIGHT UPDATE:")
        print(f"    Error is 0 → prediction was correct → NO UPDATE needed!")
    else:
        new_w0 = weights[0] + lr * error * x[0]
        new_w1 = weights[1] + lr * error * x[1]
        new_bias = bias + lr * error

        print(f"  WEIGHT UPDATE:")
        print(f"    w1_new = w1 + (lr × error × x1)")
        print(f"    w1_new = {weights[0]} + ({lr} × {error} × {x[0]})")
        print(f"    w1_new = {new_w0}")
        print()
        print(f"    w2_new = w2 + (lr × error × x2)")
        print(f"    w2_new = {weights[1]} + ({lr} × {error} × {x[1]})")
        print(f"    w2_new = {new_w1}")
        print()
        print(f"    bias_new = bias + (lr × error)")
        print(f"    bias_new = {bias} + ({lr} × {error})")
        print(f"    bias_new = {new_bias}")
        print()
        print(f"  SUMMARY:")
        print(f"    Weights: {weights} → [{new_w0}, {new_w1}]")
        print(f"    Bias:    {bias} → {new_bias}")

    print(f"\\n{'=' * 60}")
    print(f"  END OF DEMONSTRATION")
    print(f"{'=' * 60}")
    print("""
  Key Takeaways:
    1. A perceptron is just: weighted sum → activation → output
    2. It learns by adjusting weights and bias when it makes mistakes  
    3. It can learn AND, OR, NAND (linearly separable)
    4. It CANNOT learn XOR (not linearly separable)
    5. To solve XOR, you need multiple perceptrons → Multi-Layer Perceptron
    6. Every neuron in modern deep learning is a generalized version of this
    """)
'''
    },


    "2. Perceptron Basic Implementation":{
     "description": "A Basic Implementation of Perceptron",
     "timeout":300,
     "runnable": True,
     "code": '''
     
     # =============================================================================
# Perceptron from Scratch
# =============================================================================
#
# WHAT IS A PERCEPTRON?
# -----------------------------------------------------------------------------
# The Perceptron is the simplest form of a neural network — a single artificial
# neuron that learns to classify data into two categories (binary classification).
#
# Invented by Frank Rosenblatt in 1958, it is the direct ancestor of every
# modern neural network. Understanding the Perceptron means understanding the
# fundamental building block of deep learning.
#
#
# HOW IS IT DIFFERENT FROM LOGISTIC REGRESSION?
# -----------------------------------------------------------------------------
#
#   Feature               Logistic Regression       Perceptron
#   ──────────────────────────────────────────────────────────────────────
#   Output                Probability (0 to 1)      Hard label (0 or 1)
#   Activation            Sigmoid → smooth curve    Step function → hard cut
#   Loss function         Binary Cross-Entropy      No explicit loss (rule-based)
#   Update rule           Gradient of loss          Mistake-driven only
#   Updates every step?   Yes — always              No — only on misclassifications
#   Convergence           Always converges           Only if data is linearly separable
#   Probabilistic?        Yes                       No
#
# In short:
#   - Logistic Regression minimises a smooth loss using gradient descent
#   - The Perceptron only updates weights when it makes a mistake
#
#
# THE CORE IDEA
# -----------------------------------------------------------------------------
# The Perceptron computes a weighted sum of its inputs, adds a bias, and
# passes the result through a step function to produce a binary output:
#
#       z     = w · x + b          (linear combination)
#       ŷ     = 1  if z >= 0
#               0  if z <  0       (step function — hard decision boundary)
#
# Unlike Logistic Regression, there is no probability — just a direct decision.
#
#
# HOW THE PERCEPTRON LEARNS — The Perceptron Update Rule
# -----------------------------------------------------------------------------
# The Perceptron does NOT use gradient descent. Instead it uses a simple
# mistake-driven rule discovered by Rosenblatt:
#
#   For each training sample (x, y):
#       1. Make a prediction:  ŷ = step(w · x + b)
#       2. If correct  (ŷ == y): do nothing — weights unchanged
#       3. If wrong    (ŷ != y): update weights to correct the mistake:
#
#               w ← w + learning_rate * (y - ŷ) * x
#               b ← b + learning_rate * (y - ŷ)
#
#   When the prediction is too high (ŷ=1, y=0): (y - ŷ) = -1 → weights decrease
#   When the prediction is too low  (ŷ=0, y=1): (y - ŷ) = +1 → weights increase
#
#
# THE PERCEPTRON CONVERGENCE THEOREM
# -----------------------------------------------------------------------------
# If the data is LINEARLY SEPARABLE (a straight line can separate the two classes),
# the Perceptron is guaranteed to find a perfect solution in a finite number of steps.
#
# If the data is NOT linearly separable, the Perceptron will never converge —
# it will keep cycling through mistakes forever. This is its fundamental limitation.
#
#   Linearly separable:     One straight line can divide all class 0 from class 1
#   Not separable:          No straight line can perfectly separate the classes
#                           → XOR is the classic example Perceptron cannot solve
#
#
# MAKING A PREDICTION
# -----------------------------------------------------------------------------
# Once trained, for any new input x:
#
#   1. Compute weighted sum: z = w · x + b
#   2. Apply step function:  if z >= 0 → predict class 1
#                            if z <  0 → predict class 0
#
#
# Example:
#       x_train = [2, 3, 4, 5, 6]   (input feature)
#       y_train = [0, 0, 1, 1, 1]   (binary labels)
#       → The Perceptron learns a threshold that separates 0s from 1s
# =============================================================================

import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.w             = None
        self.b             = 0.0
        self.errors_       = []   # mistakes per epoch — useful for diagnostics

    # ── Activation: hard step function ───────────────────────────────────────
    def _step(self, z):
        return np.where(z >= 0, 1, 0)

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, x, y):
        self.w = np.zeros(x.shape[1] if x.ndim > 1 else 1)

        for epoch in range(self.epochs):
            errors = 0

            for xi, yi in zip(x, y):
                prediction = self._step(np.dot(self.w, xi) + self.b)
                update     = self.learning_rate * (yi - prediction)

                # Only update if a mistake was made (update = 0 when correct)
                self.w += update * xi
                self.b += update

                if update != 0:
                    errors += 1

            self.errors_.append(errors)

            # Early stop: if no mistakes in this epoch, perfectly converged
            if errors == 0:
                print(f"Converged at epoch {epoch + 1} with 0 errors.")
                break

        else:
            print(f"Stopped after {self.epochs} epochs. "
                  f"Final mistakes per epoch: {self.errors_[-1]}")
            print("Data may not be linearly separable.")

        print(f"Weights: {self.w},  Bias: {self.b:.4f}")

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, x):
        if self.w is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._step(np.dot(x, self.w) + self.b)


if __name__ == '__main__':
    # ── Linearly separable example (Perceptron will converge perfectly) ───────
    x_train = np.array([[1], [2], [3], [4], [5], [6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = Perceptron(learning_rate=0.1, epochs=1000)
    model.fit(x_train, y_train)

    x_test = np.array([[2], [4], [7]])
    print(f"\\nPredictions for {x_test.flatten().tolist()}: "
          f"{model.predict(x_test).tolist()}")

    # ── XOR example (NOT linearly separable — Perceptron will fail) ───────────
    print("\\n--- XOR Problem (not linearly separable) ---")
    x_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    model_xor = Perceptron(learning_rate=0.1, epochs=20)
    model_xor.fit(x_xor, y_xor)
    print(f"XOR predictions: {model_xor.predict(x_xor).tolist()}  (true: [0,1,1,0])")
    print("Perceptron cannot solve XOR — a multi-layer network is required.")
     '''
    },

}


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
    visual_height = 1300
    try:
        from Deep_Learning.visuals.perceptron import (   # ← match your exact folder casing
            PERCEPTRON_VISUAL_HTML,
            PERCEPTRON_VISUAL_HEIGHT,
        )
        visual_html   = PERCEPTRON_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = PERCEPTRON_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[07_gmm.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None, # COMPLEXITY,
        "operations":    OPERATIONS,
    }
