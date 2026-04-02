"""
AI Frameworks — PyTorch, JAX, Ray, TensorFlow & the Modern ML Stack
=====================================================================

The infrastructure layer beneath every model you train and deploy.
Frameworks are not interchangeable tools that do the same job differently —
each occupies a distinct niche: PyTorch for research flexibility, JAX for
mathematical rigour and compiler-level speed, Ray for distributed scale,
TensorFlow/Keras for production pipelines, Hugging Face for the ecosystem.

Knowing WHICH framework to reach for, and WHY, is as important as knowing
how to use any one of them. Wrong choice = wrong abstractions for your problem.

"""

import textwrap
import re

TOPIC_NAME   = "AI Frameworks — PyTorch, JAX, Ray & the Modern ML Stack"
DISPLAY_NAME = "00 · AI Frameworks"
ICON         = "🧰"
SUBTITLE     = "Right Tool, Right Problem — The ML Infrastructure Layer"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY FRAMEWORKS EXIST: THE ABSTRACTION LADDER

### The Problem Frameworks Solve

    Without a framework, training a neural network means:
        - Manually computing every gradient via calculus
        - Writing CUDA kernels for GPU parallelism
        - Managing memory, synchronisation, numerical stability
        - Rebuilding the same conv/attention/norm layers from scratch

    Frameworks eliminate all of this boilerplate by providing:
        1. AUTOMATIC DIFFERENTIATION — compute ∂L/∂w for any computation graph
        2. GPU ACCELERATION         — tensors backed by CUDA, no CUDA code needed
        3. COMPOSABLE BUILDING BLOCKS — layers, losses, optimisers, data loaders
        4. DISTRIBUTED TRAINING     — split work across many GPUs/machines

### The Abstraction Ladder

    ┌─────────────────────────────────────────────────────────────┐
    │  Your model code (Python)                                   │
    ├─────────────────────────────────────────────────────────────┤
    │  Framework API  (PyTorch / JAX / TF)                        │
    │  ↑ where YOU work 90% of the time                           │
    ├─────────────────────────────────────────────────────────────┤
    │  Auto-differentiation engine  (Autograd / XLA / GradTape)   │
    ├─────────────────────────────────────────────────────────────┤
    │  Linear algebra primitives  (cuBLAS, cuDNN, XLA HLO)        │
    ├─────────────────────────────────────────────────────────────┤
    │  Hardware drivers  (CUDA, ROCm, TPU runtime, Metal)         │
    ├─────────────────────────────────────────────────────────────┤
    │  Hardware  (NVIDIA GPU, AMD GPU, Google TPU, Apple Silicon) │
    └─────────────────────────────────────────────────────────────┘

### The Two Core Operations Everything Builds On

    FORWARD PASS:
        Input x → [layer₁ → layer₂ → ... → layerₙ] → prediction ŷ
        Compute loss: L = loss_fn(ŷ, y)

    BACKWARD PASS (backpropagation):
        Compute ∂L/∂w for every weight w in the network
        This is what the auto-differentiation engine does automatically
        Update: w ← w − η · ∂L/∂w    (gradient descent step)

    Frameworks differ HUGELY in HOW they implement these two operations.
    That difference drives every other design decision.


##### PART 2 — PyTorch: THE RESEARCH STANDARD

### What Makes PyTorch Different

    PyTorch uses DEFINE-BY-RUN (eager execution):
        The computation graph is built AS your Python code runs.
        No separate "build graph" step. No sessions. No placeholders.

    This means:
        - You can use normal Python control flow (if, for, while)
        - You can print intermediate tensor values instantly
        - Debugging is identical to debugging regular Python
        - The graph is recreated fresh on every forward pass

    Compare to TensorFlow 1.x (define-and-run):
        graph = tf.Graph()              # define everything upfront
        sess  = tf.Session(graph=graph) # then execute the static graph
        → Error messages pointed to graph nodes, not your Python line numbers
        → PyTorch's eager mode was a revelation when released in 2017

### Autograd — PyTorch's Gradient Engine

    Every tensor has a .grad_fn attribute that records what operation created it.
    PyTorch builds a directed acyclic graph (DAG) of these operations.

    Diagram — Autograd DAG for z = (x * w) + b:

        x ──┐
            │  mul ──► z = x*w+b ──► loss
        w ──┘       ↑
                    add
        b ──────────┘

    When you call loss.backward():
        PyTorch walks this DAG in REVERSE (chain rule)
        Accumulates gradients into .grad attribute of leaf tensors (x, w, b)

    Key autograd concepts:

        requires_grad=True:  tell PyTorch to track this tensor for grad computation
        .detach():           stop tracking — useful for inference, not training
        torch.no_grad():     context manager — disables gradient tracking entirely
        .grad:               where accumulated gradients land after .backward()
        .zero_grad():        CLEAR gradients before next batch (they accumulate!)

    Why gradients accumulate by default:
        Useful for gradient accumulation (simulate larger batches on small GPUs)
        Dangerous if you forget to call zero_grad() — gradients build up silently!

### PyTorch Core Concepts

    torch.Tensor:           the fundamental data structure (N-dimensional array + grad)
    nn.Module:              base class for ALL models and layers
    nn.Parameter:           a Tensor that is automatically registered as a weight
    torch.optim.Optimizer:  SGD, Adam, AdamW, RMSprop, etc.
    DataLoader:             wraps a Dataset, handles batching, shuffling, prefetching
    torch.cuda.device():    move tensors/models to GPU

    The canonical training loop structure:

        for epoch in range(num_epochs):
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)   # move to GPU
                optimizer.zero_grad()                 # 1. clear gradients
                pred  = model(X)                      # 2. forward pass
                loss  = loss_fn(pred, y)              # 3. compute loss
                loss.backward()                       # 4. backpropagate
                optimizer.step()                      # 5. update weights

    These five lines are the heartbeat of every PyTorch training job.

### torch.compile — The JIT Revolution (PyTorch 2.0+)

    Problem:  eager mode is easy to debug but slower than static graph execution.
    Solution: torch.compile() — compiles your model into optimised machine code
              WITHOUT changing how you write the model.

    model = torch.compile(model)   # one line, same API, 1.5–3× faster
    pred  = model(x)               # now runs compiled kernels

    How it works internally:
        TorchDynamo   → captures Python bytecode, creates a graph
        AOTAutograd   → traces the backward pass ahead of time
        TorchInductor → generates optimised CUDA/CPU/Triton kernels

    When to use torch.compile:
        ✅ Production training — straightforward speedup with zero code change
        ✅ When your model is mostly standard ops (conv, linear, attention)
        ❌ Models with heavy Python control flow (graph breaks reduce benefit)
        ❌ Debugging sessions — stack traces become harder to read

### When to Use PyTorch

    ✅  Research and experimentation — dynamic graphs = fast iteration
    ✅  NLP, CV, RL — all major papers release PyTorch code
    ✅  Custom layers and architectures — full Python expressivity
    ✅  Prototyping to production (via TorchScript or torch.compile)
    ✅  Ecosystem: torchvision, torchaudio, torchtext, HuggingFace
    ❌  TPU training at scale — JAX is better optimised for TPUs
    ❌  Functional programming style — JAX's model is cleaner for this
    ❌  When you need mature serving infrastructure out of the box


##### PART 3 — JAX: COMPOSABLE FUNCTION TRANSFORMS

### The JAX Philosophy — Everything is a Transform

    JAX is NOT just "NumPy on GPU". It is a system of composable
    function transforms that can be applied to any Python function.

    The four core transforms:

        jax.grad(f)     → a new function: computes ∂f/∂x
        jax.jit(f)      → a new function: compiled, runs on XLA
        jax.vmap(f)     → a new function: vectorised over a batch dimension
        jax.pmap(f)     → a new function: parallelised across multiple devices

    These compose:
        jax.jit(jax.grad(f))         → compiled gradient function
        jax.vmap(jax.grad(f))        → per-example gradients (efficiently!)
        jax.pmap(jax.jit(f))         → compiled, runs in parallel on 8 GPUs
        jax.grad(jax.grad(f))        → second-order derivatives (Hessian diagonal)

    This composability is JAX's superpower. PyTorch has no equivalent.

### Functional Purity — The Key Constraint

    JAX functions must be PURE (no side effects):
        - Same inputs → always same outputs
        - No mutation of external state
        - No random numbers unless you pass the key explicitly

    This sounds restrictive. It enables everything:
        - jit can safely cache and reuse compiled functions
        - vmap can safely parallelise without shared state
        - grad can safely apply the chain rule knowing nothing changed

    Impure code (PyTorch style) that breaks JAX:
        global_state += 1             # SIDE EFFECT — breaks jit
        x[0] = 5                      # IN-PLACE MUTATION — breaks grad
        np.random.randn(3)            # STATEFUL RNG — breaks vmap

    Pure equivalent in JAX:
        new_state = global_state + 1  # return new value, don't mutate
        x = x.at[0].set(5)           # functional update syntax
        jax.random.normal(key, (3,)) # explicit key, stateless RNG

### JAX Random Number System

    Problem: Python/NumPy global random state is invisible to JIT.
    Solution: JAX uses EXPLICIT, SPLITTABLE keys.

        key         = jax.random.PRNGKey(seed=42)   # create key
        key, subkey = jax.random.split(key)          # split into two
        x           = jax.random.normal(subkey, (3,)) # use subkey

    Why split?  Each call gets a UNIQUE key → reproducible + jit-safe.
    Never reuse a key — results will be correlated.

    Pattern for training loop:
        key = jax.random.PRNGKey(0)
        for step in range(n_steps):
            key, subkey = jax.random.split(key)
            loss, grads = train_step(params, subkey, batch)
            params      = update(params, grads)

### XLA — The Compiler Behind JAX

    XLA (Accelerated Linear Algebra) is Google's compiler for tensor programs.
    When you call jax.jit(f)(x), JAX:
        1. Traces f with abstract values (shapes only, not data)
        2. Produces an XLA HLO (High-Level Operations) graph
        3. XLA compiles this to hardware-specific machine code
        4. Caches the compiled binary — second call is instant

    XLA optimisations:
        Operator fusion:    conv + bias + relu → one kernel (like TensorRT!)
        Memory planning:    allocates all buffers upfront, no GC pressure
        Layout optimisation: chooses optimal tensor memory layout per op

    Trace cache keyed on SHAPE + DTYPE — different shapes recompile!
    This is why JAX is fast but has a "cold start" on first call.

### Flax and Optax — JAX's Neural Network Libraries

    JAX itself has no layers or optimisers — it's pure math transforms.
    The ecosystem provides these:

    Flax (Google): Neural network library for JAX
        - nn.Module with a functional-style parameter system
        - Parameters are EXPLICIT Python dicts (not hidden in model.parameters())
        - Supports scan (efficient RNNs), lift (higher-order modules)

    Optax (Google): Gradient processing and optimisation for JAX
        - Composable gradient transformations (clip, normalise, scale)
        - All standard optimisers: adam, sgd, adamw, lion, adagrad
        - Chains: optax.chain(optax.clip(1.0), optax.adam(1e-3))

### When to Use JAX

    ✅  Numerical computing research (physics simulations, PDEs, RL)
    ✅  Second-order optimisation (Hessians, natural gradients, Newton's method)
    ✅  Per-example gradients (privacy-preserving ML, gradient clipping)
    ✅  TPU training at Google/cloud scale — JAX is Google's internal default
    ✅  When functional purity and mathematical clarity matter
    ✅  Protein folding (AlphaFold 2/3 is JAX), RL research (Brax, PGX)
    ❌  Teams new to functional programming — steep mental model shift
    ❌  Rich ecosystem of pre-trained models — PyTorch/HuggingFace wins here
    ❌  Production deployment — fewer serving options than PyTorch/TF


##### PART 4 — RAY: DISTRIBUTED COMPUTING FOR ML

### What Ray Is

    Ray is a distributed computing framework for Python.
    It is NOT a neural network library — it has no layers, no autograd.
    Instead, it makes Python functions and classes run ACROSS a cluster
    as easily as they run on a single machine.

    The core primitive:
        @ray.remote              ← turns any function into a distributed task
        def my_function(x):
            return x * 2

        result = my_function.remote(5)   # runs on a remote worker (non-blocking)
        value  = ray.get(result)         # retrieve result (blocks until done)

### Ray's ML Ecosystem — AIR (AI Runtime)

    Ray itself is the foundation. Built on top of it:

    Ray Train:    Distributed model training
                  Works with PyTorch, TensorFlow, XGBoost, Hugging Face
                  Handles data sharding, gradient synchronisation, checkpointing
                  Scales from 1 GPU to 1000+ GPUs with minimal code change

    Ray Tune:     Hyperparameter optimisation (HPO)
                  Runs hundreds of trials in parallel across a cluster
                  Implements state-of-the-art algorithms: ASHA, PBT, Bayesian
                  Integrates with Optuna, Ax, Hyperopt

    Ray Serve:    Model serving and inference
                  Deploys models as scalable HTTP endpoints
                  Supports batching, model composition, A/B testing
                  Works with any Python function — not framework-specific

    Ray Data:     Distributed data preprocessing
                  Streaming data loading for training pipelines
                  Handles datasets larger than RAM via streaming

    Diagram — Ray AIR Stack:

    ┌───────────────────────────────────────────────────────────┐
    │  YOUR APPLICATION                                         │
    ├──────────────┬──────────────┬─────────────┬───────────────┤
    │  Ray Train   │  Ray Tune    │  Ray Serve  │  Ray Data     │
    │  (training)  │  (HPO)       │  (serving)  │  (data)       │
    ├──────────────┴──────────────┴─────────────┴───────────────┤
    │  Ray Core (task scheduling, actor model, object store)    │
    ├───────────────────────────────────────────────────────────┤
    │  Cluster  (one laptop, or 1000 cloud machines)            │
    └───────────────────────────────────────────────────────────┘

### The Actor Model — Stateful Distributed Objects

    Ray extends Python classes into distributed stateful actors:

        @ray.remote
        class ModelServer:
            def __init__(self):
                self.model = load_model()   # runs ONCE on a remote worker

            def predict(self, x):
                return self.model(x)        # called repeatedly

        server  = ModelServer.remote()      # start the actor
        futures = [server.predict.remote(x) for x in batch]   # parallel calls
        results = ray.get(futures)          # collect results

    Why actors matter:
        - Model is loaded once, stays in GPU memory across calls
        - Multiple actors = multiple GPU workers, automatically load-balanced
        - Actors can hold state (counters, caches, session data)

### Distributed Training: Data Parallelism vs Model Parallelism

    Data Parallelism (most common):
        - Each GPU has a FULL COPY of the model
        - Dataset is split across GPUs (different batches per GPU)
        - Gradients are averaged (all-reduce) across GPUs after each step
        - Ray Train handles this automatically with ScalingConfig

        ScalingConfig(num_workers=8, use_gpu=True)  → 8 GPUs, data parallel

    Model Parallelism (large models):
        - Model is split ACROSS GPUs (layer 1 on GPU0, layer 2 on GPU1...)
        - Needed when model doesn't fit in a single GPU's VRAM
        - Pipeline parallelism: GPUs pass activations like an assembly line
        - Tensor parallelism: individual weight matrices are split across GPUs

    Tensor Parallelism illustration (splitting a matrix multiply):

        Full:    Y = X @ W    [huge W matrix, one GPU]

        Split:   Y = [X @ W₁ | X @ W₂]    [W split across 2 GPUs, concat output]
                       GPU0       GPU1

### When to Use Ray

    ✅  Training on multi-GPU or multi-node clusters
    ✅  Hyperparameter search at scale (Tune)
    ✅  Production model serving with high throughput (Serve)
    ✅  Data preprocessing pipelines too large for pandas/sklearn
    ✅  Any parallelisable Python workload — not just ML
    ✅  When you want ONE framework to cover training + HPO + serving
    ❌  Single-GPU training — DDP or torch.compile is simpler
    ❌  Maximum raw training speed — DeepSpeed/Megatron-LM beat Ray Train
    ❌  When you need GPU-level control (custom CUDA, custom collectives)


##### PART 5 — TENSORFLOW & KERAS: THE PRODUCTION ECOSYSTEM

### TensorFlow's History and Position

    2015: TensorFlow 1.x — define-and-run static graphs. Powerful but verbose.
    2019: TensorFlow 2.x — eager execution by default (matching PyTorch's UX).
    Today: Keras is the official high-level API, deeply integrated into TF.

    TF's strength has never been research ergonomics.
    Its strength is PRODUCTION INFRASTRUCTURE:

        TensorFlow Extended (TFX):    full ML pipeline orchestration
        TensorFlow Serving:           battle-tested model server (gRPC + REST)
        TensorFlow Lite:              on-device inference (mobile, embedded)
        TensorFlow.js:                inference in the browser (JavaScript)
        TensorBoard:                  training visualisation (also used by PyTorch)

### Keras — The Minimal High-Level API

    Keras is the simplest neural network API in the Python ecosystem.
    Trade-off: ease-of-use over flexibility.

    Three ways to build a model in Keras:

    1. Sequential (simple stacks):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10,  activation='softmax'),
        ])

    2. Functional API (arbitrary DAGs):
        x   = keras.Input(shape=(128,))
        h   = keras.layers.Dense(256, activation='relu')(x)
        out = keras.layers.Dense(10)(h)
        model = keras.Model(inputs=x, outputs=out)

    3. Subclassing (full flexibility, like PyTorch):
        class MyModel(keras.Model):
            def __init__(self): ...
            def call(self, x): ...

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X_train, y_train, epochs=10, validation_split=0.1)
    model.save("model.keras")    ← serialise (architecture + weights + optimizer)
    model.predict(X_test)

### When to Use TensorFlow/Keras

    ✅  Mobile and edge deployment (TFLite is best-in-class)
    ✅  Browser-based ML (TensorFlow.js)
    ✅  Enterprise pipelines needing TFX, TF Serving, Vertex AI
    ✅  Beginners — Keras is the most accessible entry point
    ✅  Google Cloud / TPU integration (native support)
    ❌  Research — PyTorch dominates; most papers don't release TF code
    ❌  Flexibility for novel architectures — Keras abstracts too much


##### PART 6 — HUGGING FACE: THE PRE-TRAINED MODEL ECOSYSTEM

### What Hugging Face Provides

    Hugging Face is not a training framework — it is an ECOSYSTEM on top of them.
    It sits at the intersection of PyTorch, JAX, and TensorFlow.

    Hub:            800,000+ pre-trained models, datasets, and Spaces
    Transformers:   unified API for every major architecture
    Datasets:       streaming access to thousands of datasets
    Tokenizers:     fast Rust-backed text tokenisation
    PEFT:           parameter-efficient fine-tuning (LoRA, Prefix, Adapter)
    TRL:            reinforcement learning from human feedback (RLHF, DPO)
    Accelerate:     one-line distributed training for PyTorch

### The Transformers API — One Interface for Everything

    Same three lines load GPT-2, BERT, LLaMA, Stable Diffusion, Whisper:

        from transformers import pipeline
        pipe = pipeline("text-generation", model="gpt2")
        pipe("The secret to good coffee is")

    Fine-tuning with Trainer:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir="out", num_train_epochs=3),
            train_dataset=dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()

### Why Hugging Face Changed Everything

    Before: reproducing a BERT result required:
        - Finding the original TF code (often incomplete)
        - Converting weights manually
        - Figuring out the exact tokenisation
        - Debugging shape mismatches for days

    After: from transformers import BertModel
           model = BertModel.from_pretrained("bert-base-uncased")
           → works in 30 seconds, identical to the original paper.

    The Hub created a commons: every model trained by anyone is potentially
    usable by everyone. This compressed the field's iteration speed enormously.

### When to Use Hugging Face

    ✅  Any NLP task — classification, generation, QA, translation
    ✅  Fine-tuning a pre-trained model on your data (the standard approach)
    ✅  Multi-modal tasks (vision-language: CLIP, LLaVA, Stable Diffusion)
    ✅  When you need a baseline in hours, not weeks
    ✅  Speech, audio (Whisper, Wav2Vec2)
    ❌  Training from scratch — use raw PyTorch/JAX + Transformers as a component
    ❌  Non-standard architectures that don't fit the AutoModel API


##### PART 7 — FRAMEWORK SELECTION GUIDE

### Decision Matrix

    ┌──────────────────────────────────────┬──────────────────────────────────┐
    │  Your situation                      │  Reach for                       │
    ├──────────────────────────────────────┼──────────────────────────────────┤
    │  Research / reproducing a paper      │  PyTorch                         │
    │  Custom architecture from scratch    │  PyTorch                         │
    │  Fine-tuning a transformer model     │  HuggingFace (Transformers)      │
    │  Mathematics / physics simulation    │  JAX                             │
    │  Protein structure / biology         │  JAX (AlphaFold standard)        │
    │  Second-order optimisation           │  JAX (hessian, natural grad)     │
    │  Multi-GPU / multi-node training     │  PyTorch DDP or Ray Train        │
    │  Hyperparameter search at scale      │  Ray Tune                        │
    │  Production model serving            │  Ray Serve or TF Serving         │
    │  Mobile / edge deployment            │  TensorFlow Lite or ONNX         │
    │  Browser deployment                  │  TensorFlow.js or ONNX.js        │
    │  Google Cloud / TPU                  │  JAX or TensorFlow               │
    │  Beginner, first model               │  Keras                           │
    │  Fastest inference on NVIDIA GPU     │  TensorRT (see TensorRT module)  │
    │  Data pipeline, feature engineering  │  Ray Data + Pandas/Polars        │
    └──────────────────────────────────────┴──────────────────────────────────┘

### The Modern Production Stack (Most Common Configuration)

    RESEARCH PHASE:
        PyTorch (raw or HuggingFace) → define and train the model

    OPTIMISATION PHASE:
        torch.compile() or TensorRT → optimise for inference speed

    SERVING PHASE:
        Ray Serve or TorchServe → scale to production traffic

    MONITORING PHASE:
        TensorBoard or Weights & Biases → track metrics, debug issues

    ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐
    │  PyTorch    │ → │ torch.compile│ → │ Ray Serve  │ → │  W&B / TB  │
    │  HuggingFace│   │ TensorRT     │   │ TF Serving │   │  Monitoring│
    │  (training) │   │ (optimise)   │   │ (deploy)   │   │            │
    └─────────────┘   └──────────────┘   └────────────┘   └────────────┘


##### PART 8 — AUTOGRAD DEEP DIVE: HOW FRAMEWORKS DIFFERENTIATE

### Three Strategies for Automatic Differentiation

    1. SYMBOLIC DIFFERENTIATION (Mathematica, early TF):
        Manipulate the symbolic expression algebraically.
        Problem: expression swell — d/dx(x^n) → n*x^(n-1), chains get huge fast.

    2. NUMERIC DIFFERENTIATION (finite differences):
        df/dx ≈ (f(x + ε) - f(x)) / ε
        Problem: slow (2 evaluations per parameter), accumulates floating point error.
        Use: gradient checking (comparing against autograd to verify correctness).

    3. AUTOMATIC DIFFERENTIATION (PyTorch, JAX, TF):
        Apply the CHAIN RULE mechanically to each primitive operation.
        Exact, fast, and generalises to any differentiable program.

        Two modes:
            FORWARD MODE (JAX jvp):  compute ∂output/∂input for ONE input at a time
                                     cheap when #inputs < #outputs (rare in ML)

            REVERSE MODE (all backprop): compute ∂output/∂ALL inputs in one pass
                                          cheap when #inputs >> #outputs (ML's case:
                                          millions of weights, one scalar loss)

### The Chain Rule — The Only Math Behind Backprop

    If z = f(g(x)), then:
        dz/dx = dz/dy · dy/dx    where y = g(x)

    For a 3-layer network: z = f₃(f₂(f₁(x)))
        dL/dw₁ = (dL/df₃) · (df₃/df₂) · (df₂/df₁) · (df₁/dw₁)

    Autograd does this automatically for arbitrary computation graphs.
    Each primitive operation (add, mul, matmul, relu, softmax) has a
    pre-programmed VJP (Vector-Jacobian Product) rule:

        op:     y  = f(x)
        VJP:    v̄ᵀ = v̄ᵀ · ∂f/∂x    [v̄ is the upstream gradient]

    These VJPs are composed by walking the DAG in reverse.
    This is ALL that backpropagation is — repeated chain rule via DAG traversal.

### Gradient Tape, grad(), and backward() — Three Interfaces, One Idea

    TensorFlow 2 (explicit tape):
        with tf.GradientTape() as tape:
            y = model(x)
            L = loss(y, target)
        grads = tape.gradient(L, model.trainable_variables)

    JAX (functional):
        grad_fn = jax.grad(loss_fn)          # returns a NEW function
        grads   = grad_fn(params, x, target) # call it like any function

    PyTorch (implicit graph):
        L = loss(model(x), target)
        L.backward()                          # gradients in param.grad

    All three compute identical gradients — the interface differs, not the math.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · PyTorch Core — Tensors, Autograd & the Training Loop": {
        "description": (
            "PyTorch fundamentals from the ground up. "
            "Tensor creation and operations, requires_grad and gradient accumulation, "
            "manual gradient computation (no nn.Module), then a full training loop "
            "with nn.Module, DataLoader, and Adam. Covers the five canonical training steps."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  PYTORCH CORE — TENSORS, AUTOGRAD & TRAINING LOOP")
print("=" * 65)
print()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

print(f"  PyTorch version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: TENSOR BASICS
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — TENSORS")
print("━" * 65)
print()

# Creation
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(3, 4)
c = torch.randn(2, 3)
d = torch.arange(0, 10, step=2, dtype=torch.float32)

print(f"  torch.tensor([1,2,3]):    {a}")
print(f"  torch.zeros(3,4).shape:   {b.shape}")
print(f"  torch.randn(2,3):"); print(c)
print(f"  torch.arange(0,10,2):     {d}")
print()

# Operations — elementwise and matrix
x  = torch.randn(3, 4)
W  = torch.randn(4, 5)
y  = x @ W          # matrix multiply
print(f"  x @ W  ({x.shape} @ {W.shape}) = {y.shape}")

# Broadcasting (numpy rules apply)
row_mean = x.mean(dim=1, keepdim=True)   # shape (3, 1)
centred  = x - row_mean                  # broadcasts (3,4) - (3,1) → (3,4)
print(f"  Centred x (broadcast): {centred.shape}")
print()

# GPU transfer
t = torch.randn(3)
t_gpu = t.to(device)
print(f"  CPU tensor device: {t.device}")
print(f"  GPU tensor device: {t_gpu.device}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: AUTOGRAD — GRADIENTS WITHOUT CALCULUS
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — AUTOGRAD (automatic differentiation)")
print("━" * 65)
print()

# Manual gradient computation for: L = (w*x - y)^2  (MSE, 1 sample)
x_val = torch.tensor(2.0)
y_val = torch.tensor(6.0)
w     = torch.tensor(1.0, requires_grad=True)   # ← track this!
b     = torch.tensor(0.0, requires_grad=True)

pred  = w * x_val + b          # forward
loss  = (pred - y_val) ** 2    # loss

print(f"  x={x_val.item()}, y={y_val.item()}, w={w.item()}, b={b.item()}")
print(f"  pred = w*x + b = {pred.item()}")
print(f"  loss = (pred - y)² = {loss.item()}")
print()

loss.backward()   # computes ∂loss/∂w and ∂loss/∂b

# Verify analytically:
# loss = (w*x + b - y)^2
# ∂loss/∂w = 2*(w*x+b-y)*x  = 2*(2-6)*2 = -16
# ∂loss/∂b = 2*(w*x+b-y)*1  = 2*(2-6)*1 = -8
print(f"  ∂loss/∂w (autograd):   {w.grad.item():.4f}  (analytic: {2*(pred.item()-y_val.item())*x_val.item():.4f})")
print(f"  ∂loss/∂b (autograd):   {b.grad.item():.4f}  (analytic: {2*(pred.item()-y_val.item()):.4f})")
print()
print("  Gradient accumulation warning:")
print("  Calling backward() again ADDS to existing .grad — call .zero_grad() first!")
print()

# Demonstrate accumulation bug
w.grad  # currently -16
loss2  = (w * x_val + b - y_val) ** 2
loss2.backward()
print(f"  After 2nd backward WITHOUT zero_grad: w.grad = {w.grad.item()}")  # -32!
w.grad.zero_()   # reset
loss3  = (w * x_val + b - y_val) ** 2
loss3.backward()
print(f"  After 2nd backward WITH    zero_grad: w.grad = {w.grad.item()}")  # -16 ✓
print()

# torch.no_grad — disable tracking for inference
with torch.no_grad():
    inference_pred = w * x_val + b
print(f"  Prediction under no_grad: {inference_pred.item():.4f}  (grad_fn: {inference_pred.grad_fn})")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: THE CANONICAL TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — FULL TRAINING LOOP (nn.Module + DataLoader + Adam)")
print("━" * 65)
print()

# ── Dataset: noisy linear y = 2x + 1 ─────────────────────────────────────
torch.manual_seed(42)
N      = 500
X_data = torch.randn(N, 1)
y_data = 2.0 * X_data + 1.0 + 0.3 * torch.randn(N, 1)  # y = 2x+1 + noise

dataset    = TensorDataset(X_data, y_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"  Dataset: y = 2x + 1 + noise   (N={N}, batch_size=32)")
print(f"  DataLoader batches per epoch: {len(dataloader)}")
print()

# ── Model ─────────────────────────────────────────────────────────────────
class LinearModel(nn.Module):
    """A single Linear layer — will learn w≈2, b≈1."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model     = LinearModel().to(device)
loss_fn   = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("  Model architecture:")
print(f"  {model}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"    Initial w: {model.linear.weight.item():.4f}  (random init)")
print(f"    Initial b: {model.linear.bias.item():.4f}  (random init)")
print()

# ── Training loop — THE FIVE CANONICAL STEPS ─────────────────────────────
print(f"  {'Epoch':>6} | {'Train Loss':>12} | {'w (→ 2.0)':>12} | {'b (→ 1.0)':>12}")
print(f"  {'─'*50}")

for epoch in range(1, 21):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # ① Clear gradients from the previous batch
        optimizer.zero_grad()

        # ② Forward pass — compute predictions
        pred = model(X_batch)

        # ③ Compute loss
        loss = loss_fn(pred, y_batch)

        # ④ Backward pass — compute gradients
        loss.backward()

        # ⑤ Update weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    if epoch in {1, 5, 10, 15, 20}:
        w_val = model.linear.weight.item()
        b_val = model.linear.bias.item()
        print(f"  {epoch:6d} | {avg_loss:12.6f} | {w_val:12.6f} | {b_val:12.6f}")

print()
w_final = model.linear.weight.item()
b_final = model.linear.bias.item()
print(f"  Final w: {w_final:.4f}  (true: 2.0000)  error: {abs(w_final-2):.4f}")
print(f"  Final b: {b_final:.4f}  (true: 1.0000)  error: {abs(b_final-1):.4f}")
print()

# ── Inference (eval mode) ─────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    test_x    = torch.tensor([[0.0], [1.0], [2.0]]).to(device)
    test_pred = model(test_x)
    true_y    = 2 * test_x + 1

print("  Evaluation (model.eval() + torch.no_grad()):")
print(f"  {'x':>6} | {'pred':>10} | {'true (2x+1)':>14} | {'error':>10}")
print(f"  {'─'*45}")
for xi, pi, ti in zip(test_x, test_pred, true_y):
    print(f"  {xi.item():6.1f} | {pi.item():10.4f} | {ti.item():14.4f} | {abs(pi-ti).item():10.4f}")

print()
print("  THE FIVE CANONICAL PYTORCH TRAINING STEPS (memorise these):")
print("  ① optimizer.zero_grad()    — clear accumulated gradients")
print("  ② pred = model(X)          — forward pass")
print("  ③ loss = loss_fn(pred, y)  — compute scalar loss")
print("  ④ loss.backward()          — backpropagate (fill .grad)")
print("  ⑤ optimizer.step()         — update weights with gradients")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · PyTorch Advanced — Custom Layers, LR Scheduling & Regularisation": {
        "description": (
            "Go beyond the basics. Build a custom nn.Module with He initialisation. "
            "Add dropout and batch normalisation. Use learning rate scheduling "
            "(cosine annealing, warmup). Demonstrate gradient clipping. "
            "Compare Adam vs AdamW on an overfit experiment."
        ),
        "language": "python",
        "code": '''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import TensorDataset, DataLoader

print("=" * 65)
print("  PYTORCH ADVANCED — CUSTOM LAYERS, LR SCHEDULING & REGULARISATION")
print("=" * 65)
print()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: CUSTOM nn.Module WITH PROPER INITIALISATION
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Custom nn.Module with He Init, BN, Dropout")
print("━" * 65)
print()

class MLP(nn.Module):
    """
    Multi-layer perceptron with:
    - He (Kaiming) initialisation for ReLU layers
    - Batch normalisation for stable training
    - Dropout for regularisation
    """
    def __init__(self, in_dim: int, hidden_dims: list, out_dim: int,
                 dropout_p: float = 0.3):
        super().__init__()

        dims   = [in_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))  # normalise activations
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))    # randomly zero activations

        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

        # He (Kaiming) initialisation — designed for ReLU activations
        # Var(w) = 2/fan_in prevents vanishing/exploding gradients
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# Inspect architecture
model = MLP(in_dim=20, hidden_dims=[128, 256, 128], out_dim=1, dropout_p=0.2)
model = model.to(device)

print("  MLP architecture (in=20, hidden=[128,256,128], out=1):")
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable:,}")
print()

# Verify He init
first_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][0]
w            = first_linear.weight.data
expected_std = np.sqrt(2.0 / w.shape[1])   # He init: std = sqrt(2/fan_in)
print(f"  First Linear weight stats:")
print(f"    std (actual):   {w.std().item():.4f}")
print(f"    std (He theory):{expected_std:.4f}")
print(f"    mean:           {w.mean().item():.4f}  (should be ~0)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: TRAINING MODES — model.train() vs model.eval()
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Training vs Eval Mode")
print("━" * 65)
print()
print("  BatchNorm and Dropout behave DIFFERENTLY in train vs eval:")
print()
print("  BatchNorm:")
print("    train(): normalises with the CURRENT BATCH stats (mean, var)")
print("    eval():  uses RUNNING stats accumulated during training")
print()
print("  Dropout:")
print("    train(): randomly zeroes p% of activations (adds noise, prevents overfit)")
print("    eval():  passes all activations through (scales by 1/(1-p) automatically)")
print()
print("  THE RULE: always call model.eval() before inference, model.train() before training.")
print()

x_test = torch.randn(4, 20).to(device)  # 4 samples
model.train()
out_train = model(x_test)
model.eval()
with torch.no_grad():
    out_eval1 = model(x_test)
    out_eval2 = model(x_test)

print(f"  Same input, train mode outputs:   {out_train.flatten()[:4].tolist()}")
print(f"  Same input, eval  mode output 1:  {out_eval1.flatten()[:4].tolist()}")
print(f"  Same input, eval  mode output 2:  {out_eval2.flatten()[:4].tolist()}")
print(f"  eval outputs identical: {torch.allclose(out_eval1, out_eval2)}")
print(f"  train != eval (dropout active): {not torch.allclose(out_train, out_eval1)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: LEARNING RATE SCHEDULING
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — LR Scheduling: Warmup → Cosine Annealing")
print("━" * 65)
print()

model_lr = MLP(20, [64, 64], 1, dropout_p=0.1).to(device)
BASE_LR  = 1e-3
WARMUP   = 5     # epochs of linear warmup
TOTAL    = 50    # total training epochs

optimizer = optim.AdamW(model_lr.parameters(), lr=BASE_LR, weight_decay=1e-2)

# Warmup: LR rises linearly from 0 → BASE_LR over WARMUP steps
warmup   = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=WARMUP)
# Cosine: LR decays smoothly from BASE_LR → 0 over remaining steps
cosine   = CosineAnnealingLR(optimizer, T_max=TOTAL - WARMUP, eta_min=1e-6)
# Chain them sequentially
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP])

print(f"  Schedule: linear warmup ({WARMUP} ep) → cosine anneal ({TOTAL-WARMUP} ep)")
print(f"  BASE_LR = {BASE_LR}")
print()
print(f"  {'Epoch':>6} | {'LR':>12} | Phase")
print(f"  {'─'*35}")

import warnings
lrs = []
for ep in range(1, TOTAL + 1):
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    phase = "warmup" if ep <= WARMUP else "cosine"
    if ep in {1, 3, 5, 10, 20, 35, 50}:
        print(f"  {ep:6d} | {current_lr:12.8f} | {phase}")
    # Demo loop: suppress the step-order warning — we are only
    # visualising the LR curve here, not performing real training.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler.step()

print()
print(f"  LR at start:  {lrs[0]:.2e}  (near zero — warmup begins)")
print(f"  LR at peak:   {max(lrs):.2e}  (end of warmup)")
print(f"  LR at end:    {lrs[-1]:.2e}  (cosine brings it near zero)")
print()
print("  WHY WARMUP MATTERS:")
print("  Early in training, weights are random → gradients are huge.")
print("  A large initial LR causes catastrophic parameter updates.")
print("  Warmup lets the model 'settle' before using the full LR.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: GRADIENT CLIPPING
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Gradient Clipping (prevents exploding gradients)")
print("━" * 65)
print()

torch.manual_seed(1)
N_SAMPLES = 400
X_ovf = torch.randn(N_SAMPLES, 20)
y_ovf = torch.randn(N_SAMPLES, 1)

model_clip = MLP(20, [64, 64], 1, dropout_p=0.0).to(device)
opt_clip   = optim.SGD(model_clip.parameters(), lr=10.0)  # deliberately huge LR
loss_fn    = nn.MSELoss()

ds     = TensorDataset(X_ovf.to(device), y_ovf.to(device))
loader = DataLoader(ds, batch_size=64)

print(f"  Using SGD with LR=10.0 (deliberately too large → exploding grads)")
print()
print(f"  {'Step':>5} | {'Loss':>12} | {'Grad norm (before)':>20} | {'After clip (1.0)':>18}")
print(f"  {'─'*65}")

model_clip.train()
for step, (xb, yb) in enumerate(loader):
    opt_clip.zero_grad()
    loss = loss_fn(model_clip(xb), yb)
    loss.backward()

    # Measure gradient norm BEFORE clipping
    total_norm_before = 0.0
    for p in model_clip.parameters():
        if p.grad is not None:
            total_norm_before += p.grad.data.norm(2).item() ** 2
    total_norm_before = total_norm_before ** 0.5

    # Clip gradients — rescale if norm > max_norm
    torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=1.0)

    # Measure AFTER clipping
    total_norm_after = 0.0
    for p in model_clip.parameters():
        if p.grad is not None:
            total_norm_after += p.grad.data.norm(2).item() ** 2
    total_norm_after = total_norm_after ** 0.5

    opt_clip.step()
    if step < 6:
        print(f"  {step:5d} | {loss.item():12.4f} | {total_norm_before:20.4f} | {total_norm_after:18.4f}")

print()
print("  clip_grad_norm_(params, max_norm=1.0):")
print("  If ||grads|| > max_norm: grads ← grads × (max_norm / ||grads||)")
print("  Prevents one bad batch from making catastrophically large weight updates.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: ADAM vs AdamW
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Adam vs AdamW (weight decay done right)")
print("━" * 65)
print()
print("  Adam:   weight_decay is applied via L2 penalty IN the gradient")
print("          → L2 penalty interacts with adaptive learning rates → INCORRECT")
print()
print("  AdamW:  weight_decay is applied SEPARATELY (decoupled decay)")
print("          → decoupled from gradient scaling → mathematically correct")
print("          → better generalisation, standard in modern transformers")
print()
print("  Decoupled weight decay formula:")
print("    Adam:   g_t = g_t + λ·w_t           (adds λw to gradient)")
print("    AdamW:  w_{t+1} = (1-λ)·w_t - η·m̂_t/√(v̂_t+ε)  (separate step)")
print()
print("  RULE: Always use AdamW for transformers and large models.")
print("        Adam is fine for CNNs and smaller architectures.")
print()

param_count = sum(p.numel() for p in model.parameters())
wd_adam  = optim.Adam(model.parameters(),  lr=1e-3, weight_decay=1e-4)
wd_adamw = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
print(f"  Both optimisers created for {param_count:,}-param model.")
print(f"  AdamW: optimizer.param_groups[0]['weight_decay'] = "
      f"{wd_adamw.param_groups[0]['weight_decay']}")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · JAX Fundamentals — jit, grad, vmap and Functional Style": {
        "description": (
            "JAX from first principles. Demonstrate jax.grad on a scalar function, "
            "higher-order derivatives (Hessian), jax.jit for compilation speedup, "
            "jax.vmap for vectorisation (per-example gradients), and the pure-function "
            "constraint. Compare JAX vs NumPy speed. Show the explicit random key system."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  JAX FUNDAMENTALS — jit, grad, vmap AND FUNCTIONAL STYLE")
print("=" * 65)
print()

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
except ImportError:
    print("  pip install jax jaxlib")
    print("  For GPU: pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    raise SystemExit(0)

print(f"  JAX version: {jax.__version__}")
print(f"  Devices: {jax.devices()}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: jax.grad — DIFFERENTIATION AS A FUNCTION TRANSFORM
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — jax.grad (differentiation as a transform)")
print("━" * 65)
print()

# JAX's key insight: grad() is a HIGHER-ORDER FUNCTION
# It takes a function and returns a NEW function that computes gradients
print("  In JAX, grad() transforms a function into its gradient function:")
print("  f(x) → grad(f)(x) → df/dx")
print()

def f(x):
    """A simple scalar function: f(x) = x³ - 2x² + x"""
    return x**3 - 2*x**2 + x

def f_prime_analytic(x):
    """Analytic derivative: f'(x) = 3x² - 4x + 1"""
    return 3*x**2 - 4*x + 1

df = grad(f)        # df is now a function: x → df/dx
d2f = grad(df)      # d2f is the SECOND derivative: x → d²f/dx²
d3f = grad(d2f)     # d3f is the THIRD derivative

test_points = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])

print(f"  f(x) = x³ - 2x² + x")
print(f"  f'(x) [analytic] = 3x² - 4x + 1")
print()
print(f"  {'x':>6} | {'f(x)':>10} | {'grad(f)':>12} | {'analytic':>12} | {'match':>8}")
print(f"  {'─'*55}")
for x in test_points:
    fx      = f(x)
    dfx     = df(x)
    analytic = f_prime_analytic(x)
    match   = jnp.allclose(dfx, analytic, atol=1e-5)
    print(f"  {x.item():6.2f} | {fx.item():10.4f} | {dfx.item():12.6f} | {analytic.item():12.6f} | {'✅' if match else '❌':>8}")

print()
print(f"  Second derivative d²f/dx² at x=1.0: {d2f(1.0):.4f}  (analytic: 6x-4 = {6*1-4:.1f})")
print(f"  Third  derivative d³f/dx³ at x=1.0: {d3f(1.0):.4f}  (analytic: 6)")
print()
print("  PyTorch equivalent would require autograd twice — messy and slower.")
print("  JAX: grad(grad(grad(f))) — trivially composes.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: GRADIENT DESCENT WITH JAX
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Manual Gradient Descent (functional style)")
print("━" * 65)
print()

# In JAX, model parameters are EXPLICIT Python dicts (not hidden in module)
# This is functional programming — no mutable state

def loss_fn(params, X, y):
    """MSE loss for linear regression: L = mean((X@w + b - y)²)"""
    pred = X @ params['w'] + params['b']
    return jnp.mean((pred - y) ** 2)

# Gradient w.r.t. FIRST argument (params) — other args are static
grad_fn  = jit(grad(loss_fn))          # compile the gradient function
loss_jit = jit(loss_fn)                # compile the loss function too

# Data: y = 3x₁ + -2x₂ + 0.5
key   = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
X_jax = jax.random.normal(k1, (200, 2))
y_jax = 3.0 * X_jax[:, 0] - 2.0 * X_jax[:, 1] + 0.5

# Initial parameters (pure Python/JAX dicts — no Module class needed)
params = {
    'w': jax.random.normal(k2, (2,)) * 0.1,
    'b': jnp.array(0.0)
}

print(f"  Task: learn y = 3x₁ - 2x₂ + 0.5 from {len(X_jax)} samples")
print(f"  Params are plain dicts: {{'w': array(shape=(2,)), 'b': scalar}}")
print()

LR     = 0.05
STEPS  = 100

print(f"  {'Step':>6} | {'Loss':>12} | {'w[0] (→ 3.0)':>14} | {'w[1] (→-2.0)':>14} | {'b (→ 0.5)':>12}")
print(f"  {'─'*65}")

for step in range(STEPS + 1):
    grads = grad_fn(params, X_jax, y_jax)   # ← compute gradient of loss w.r.t. params

    # Functional update — create NEW params dict, don't mutate the old one
    params = jax.tree_util.tree_map(
        lambda p, g: p - LR * g,   # update rule: p ← p - lr*g
        params, grads
    )

    if step % 20 == 0:
        loss = loss_jit(params, X_jax, y_jax)
        print(f"  {step:6d} | {loss.item():12.6f} | {params['w'][0].item():14.6f} | "
              f"{params['w'][1].item():14.6f} | {params['b'].item():12.6f}")

print()
print(f"  True coefficients: w=[3.0, -2.0], b=0.5")
print(f"  Learned:           w=[{params['w'][0]:.4f}, {params['w'][1]:.4f}], b={params['b']:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: jax.jit — COMPILATION SPEEDUP
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — jax.jit (trace once, run fast forever)")
print("━" * 65)
print()

def matmul_relu(A, B):
    return jnp.maximum(0, A @ B)

matmul_relu_jit = jit(matmul_relu)

key, k1, k2 = jax.random.split(key, 3)
A = jax.random.normal(k1, (512, 512))
B = jax.random.normal(k2, (512, 512))

N_BENCH = 50

# Un-JIT baseline
_ = matmul_relu(A, B)  # warmup
_ = matmul_relu(A, B).block_until_ready()
t0 = time.time()
for _ in range(N_BENCH):
    out = matmul_relu(A, B).block_until_ready()
t_nojit = (time.time() - t0) / N_BENCH * 1000

# JIT (first call traces and compiles)
print(f"  First jit call (traces + compiles)...")
_ = matmul_relu_jit(A, B).block_until_ready()  # trace + compile
t0 = time.time()
for _ in range(N_BENCH):
    out = matmul_relu_jit(A, B).block_until_ready()
t_jit = (time.time() - t0) / N_BENCH * 1000

print(f"  Without jit: {t_nojit:.4f} ms/call")
print(f"  With jit:    {t_jit:.4f} ms/call")
print(f"  Speedup:     {t_nojit/t_jit:.2f}×")
print()
print("  CRITICAL JIT RULES:")
print("  1. First call is slow — it traces and compiles the function")
print("  2. Recompiles if shape or dtype changes (traced per shape!)")
print("  3. Use .block_until_ready() for accurate timing (GPU is async)")
print("  4. Side effects inside jit are NOT guaranteed to happen")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: jax.vmap — VECTORISATION
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — jax.vmap (automatic vectorisation)")
print("━" * 65)
print()

print("  vmap turns a function that processes ONE sample into one that")
print("  processes a BATCH — without writing any loop or batching logic.")
print()

def single_sample_loss(params, x, y):
    """Loss for a SINGLE (x, y) pair — not a batch."""
    pred = jnp.dot(params['w'], x) + params['b']
    return (pred - y) ** 2

# Per-example gradient: gradient of loss for EACH individual sample
# This is expensive in PyTorch; in JAX it's one line
per_example_grad = vmap(grad(single_sample_loss), in_axes=(None, 0, 0))
#                        ↑ params: same for all   ↑ x: batch axis 0   ↑ y: batch axis 0

key, k1, k2 = jax.random.split(key, 3)
X_batch = jax.random.normal(k1, (8, 2))
y_batch = jax.random.normal(k2, (8,))
p_test  = {'w': jnp.ones(2), 'b': jnp.array(0.0)}

per_grads = per_example_grad(p_test, X_batch, y_batch)

print(f"  Batch size: {X_batch.shape[0]}, feature dim: {X_batch.shape[1]}")
print(f"  per_example_grad output shapes:")
print(f"    w gradients: {per_grads['w'].shape}  (one grad vector per sample)")
print(f"    b gradients: {per_grads['b'].shape}  (one scalar per sample)")
print()
print(f"  Per-example w-gradients (first 4 samples):")
for i in range(4):
    print(f"    sample {i}: {per_grads['w'][i].tolist()}")
print()
print("  USE CASE: differential privacy (DP-SGD) clips PER-EXAMPLE gradients")
print("  before averaging. vmap makes this trivial. In PyTorch, this requires")
print("  hacky loops or specialist libraries (Opacus).")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: JAX RANDOM KEY SYSTEM
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Explicit Random Keys (functional RNG)")
print("━" * 65)
print()

print("  JAX rule: NEVER reuse a key. Always split before using.")
print()
print("  WRONG (correlated outputs):")
print("    key = jax.random.PRNGKey(0)")
print("    x   = jax.random.normal(key, (3,))")
print("    y   = jax.random.normal(key, (3,))  ← SAME key → same output!")
print()

key = jax.random.PRNGKey(0)
x_wrong = jax.random.normal(key, (3,))
y_wrong = jax.random.normal(key, (3,))
print(f"    x = {x_wrong.tolist()}")
print(f"    y = {y_wrong.tolist()}")
print(f"    x == y: {jnp.allclose(x_wrong, y_wrong)}")
print()

print("  CORRECT (independent outputs via splitting):")
print("    key, subkey1 = jax.random.split(key)")
print("    key, subkey2 = jax.random.split(key)")
print()

key = jax.random.PRNGKey(0)
key, sk1 = jax.random.split(key)
key, sk2 = jax.random.split(key)
x_right = jax.random.normal(sk1, (3,))
y_right = jax.random.normal(sk2, (3,))
print(f"    x = {x_right.tolist()}")
print(f"    y = {y_right.tolist()}")
print(f"    x == y: {jnp.allclose(x_right, y_right)}")
print()
print("  Why explicit keys?")
print("  jit needs to trace the function deterministically.")
print("  Implicit global state breaks this — you can't JIT numpy.random.")
print("  Explicit keys make randomness a first-class, composable value.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · JAX Neural Network — Pure JAX + Optax MLP (Windows-compatible)": {
        "description": (
            "Build and train a real neural network in the JAX ecosystem using "
            "pure JAX + Optax — no Flax required (Windows-compatible). "
            "Parameters live in explicit Python dicts. Uses jax.grad + jax.jit "
            "for a compiled train step with AdamW via Optax. Mirrors what Flax "
            "does under the hood and explains the Flax design decisions."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  JAX NEURAL NETWORK — Pure JAX + Optax MLP")
print("=" * 65)
print()

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, random, value_and_grad
    import optax
except ImportError as e:
    print(f"  ImportError: {e}")
    print("  pip install jax jaxlib optax")
    raise SystemExit(0)

print(f"  JAX:   {jax.__version__}")
print(f"  Optax: {optax.__version__}")
print()
print("  NOTE: This operation uses pure JAX + Optax (no Flax).")
print("  Flax requires orbax-checkpoint -> uvloop, which does not")
print("  support Windows. Pure JAX achieves the same result.")
print()

# ─────────────────────────────────────────────────────────────────────────
# DATASET: synthetic binary classification
# ─────────────────────────────────────────────────────────────────────────
key = random.PRNGKey(42)

def make_dataset(key, n=1000, d=16):
    key, k1 = random.split(key)
    X = random.normal(k1, (n, d))
    threshold = d * 1.0
    y = (jnp.sum(X**2, axis=1) > threshold).astype(jnp.int32)
    return X, y

key, k_data = random.split(key)
X_all, y_all = make_dataset(k_data, n=2000, d=16)
X_train, y_train = X_all[:1600], y_all[:1600]
X_val,   y_val   = X_all[1600:], y_all[1600:]

print(f"  Task: binary classification (||x||² > threshold)")
print(f"  Features: 16   Train: {len(X_train)}   Val: {len(X_val)}")
print(f"  Class balance: {y_train.mean()*100:.1f}% positive")
print()

# ─────────────────────────────────────────────────────────────────────────
# PURE JAX MLP — parameters as explicit Python dicts (what Flax does internally)
# ─────────────────────────────────────────────────────────────────────────
print("  PURE JAX MLP — params are plain Python dicts, no hidden state")
print()

def init_mlp(key, layer_sizes):
    """
    Initialise all weights using He (Kaiming) init.
    Returns a LIST of {'w': ..., 'b': ...} dicts — one per layer.
    This is exactly what Flax stores internally.
    """
    params = []
    for i in range(len(layer_sizes) - 1):
        key, k1 = random.split(key)
        fan_in = layer_sizes[i]
        w = random.normal(k1, (fan_in, layer_sizes[i+1])) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros(layer_sizes[i+1])
        params.append({'w': w, 'b': b})
    return params

def mlp_forward(params, x):
    """Forward pass: ReLU hidden layers, linear output."""
    for layer in params[:-1]:          # all but last
        x = jnp.dot(x, layer['w']) + layer['b']
        x = jax.nn.relu(x)
    out = jnp.dot(x, params[-1]['w']) + params[-1]['b']  # output layer
    return out

# Layer sizes: 16 inputs → 64 → 128 → 64 → 2 outputs
LAYER_SIZES = [16, 64, 128, 64, 2]
key, k_init = random.split(key)
params = init_mlp(k_init, LAYER_SIZES)

total_params = sum(p['w'].size + p['b'].size for p in params)
print(f"  Architecture: {' → '.join(map(str, LAYER_SIZES))}")
print(f"  Total parameters: {total_params:,}")
print()
print("  Parameter structure (inspect directly — no model.parameters() needed):")
for i, layer in enumerate(params):
    print(f"    Layer {i}: w={layer['w'].shape}  b={layer['b'].shape}")
print()

# ─────────────────────────────────────────────────────────────────────────
# LOSS, METRICS AND OPTAX OPTIMISER
# ─────────────────────────────────────────────────────────────────────────
def cross_entropy_loss(params, x, y):
    logits   = mlp_forward(params, x)
    one_hot  = jax.nn.one_hot(y, 2)
    log_p    = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot * log_p, axis=-1))

def accuracy(params, x, y):
    logits = mlp_forward(params, x)
    return jnp.mean(jnp.argmax(logits, axis=-1) == y)

print("  Optax: composable gradient processing pipeline")
print("  optax.chain(clip, adamw) — clip gradients, then AdamW update")
print()

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=1e-3, weight_decay=1e-4),
)
opt_state = optimizer.init(params)

# ─────────────────────────────────────────────────────────────────────────
# JIT-COMPILED TRAIN STEP — pure function, no mutation
# ─────────────────────────────────────────────────────────────────────────
@jit
def train_step(params, opt_state, X, y):
    """
    Pure function: same inputs → same outputs.
    Nothing is mutated. Returns NEW params and opt_state.
    This functional discipline is what allows jit to cache the compiled binary.
    """
    loss, grads    = value_and_grad(cross_entropy_loss)(params, X, y)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params     = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

@jit
def eval_step(params, X, y):
    loss = cross_entropy_loss(params, X, y)
    acc  = accuracy(params, X, y)
    return loss, acc

# ─────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────
EPOCHS     = 20
BATCH_SIZE = 64
N_BATCHES  = len(X_train) // BATCH_SIZE

print(f"  Training: {EPOCHS} epochs, batch={BATCH_SIZE}, {N_BATCHES} steps/epoch")
print()
print(f"  {'Epoch':>6} | {'Train Loss':>12} | {'Train Acc':>11} | {'Val Loss':>10} | {'Val Acc':>9}")
print(f"  {'-'*60}")

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    key, k_shuf = random.split(key)
    perm   = random.permutation(k_shuf, len(X_train))
    X_shuf = X_train[perm]
    y_shuf = y_train[perm]

    ep_loss = []
    for i in range(N_BATCHES):
        Xb = X_shuf[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        yb = y_shuf[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        params, opt_state, loss = train_step(params, opt_state, Xb, yb)
        ep_loss.append(float(loss))

    val_loss, val_acc = eval_step(params, X_val, y_val)
    tr_acc = accuracy(params, X_train, y_train)

    if epoch in {1, 5, 10, 15, 20}:
        print(f"  {epoch:6d} | {np.mean(ep_loss):12.4f} | "
              f"{float(tr_acc)*100:10.2f}% | {float(val_loss):.6f} | {float(val_acc)*100:.2f}%")

jax.block_until_ready(params[0]['w'])
t_total = time.time() - t0
print()
print(f"  Total training time: {t_total:.2f}s  (includes JIT compile on first step)")
print()

# ─────────────────────────────────────────────────────────────────────────
# COMPARISON: PURE JAX vs FLAX vs PYTORCH
# ─────────────────────────────────────────────────────────────────────────
print("  DESIGN COMPARISON:")
print("  +--------------------------+----------------------+----------------------+")
print("  | Concept                  | PyTorch              | JAX (pure / Flax)    |")
print("  +--------------------------+----------------------+----------------------+")
print("  | Parameters               | Hidden in Module     | Explicit dict/list   |")
print("  | Optimiser state          | Inside optim object  | External opt_state   |")
print("  | Training mode            | model.train()        | Pass as argument     |")
print("  | Gradient computation     | loss.backward()      | jax.grad(loss_fn)    |")
print("  | Weight update            | optimizer.step()     | optax.apply_updates  |")
print("  | Side effects             | Yes (stateful)       | None (pure funcs)    |")
print("  +--------------------------+----------------------+----------------------+")
print()
print("  Flax adds on top of pure JAX:")
print("    - nn.Module class for reusable layer definitions")
print("    - @nn.compact for inline layer construction")
print("    - Automatic parameter tree management via model.init()")
print("    - BatchNorm running stats via 'batch_stats' pytree")
print("  All Flax params are still plain dicts — inspectable and serialisable.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Ray Core & Ray Tune — Parallel Tasks and Hyperparameter Search": {
        "description": (
            "Ray fundamentals: remote tasks, actors, and the object store. "
            "Then Ray Tune for hyperparameter optimisation: run 20 trials in parallel, "
            "use ASHA early stopping to kill unpromising trials, visualise results. "
            "Show how Ray scales from a laptop to a cluster with zero code change. "
            "Requires: ray, ray[tune], numpy, torch."
        ),
        "timeout":600,
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  RAY CORE & RAY TUNE — DISTRIBUTED TASKS + HPO")
print("=" * 65)
print()

try:
    import ray
except ImportError:
    print("  pip install 'ray[tune]' torch")
    raise SystemExit(0)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: RAY CORE — REMOTE FUNCTIONS AND THE OBJECT STORE
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Ray Core (remote tasks and actors)")
print("━" * 65)
print()

ray.init(ignore_reinit_error=True, num_cpus=4, log_to_driver=False)
print(f"  Ray initialised")
print(f"  Available resources: {ray.available_resources()}")
print()

# ── Remote functions — run in parallel ───────────────────────────────────
print("  REMOTE FUNCTIONS (stateless parallel tasks):")
print()

@ray.remote
def slow_computation(task_id: int, n: int = 500_000) -> float:
    """Simulate a CPU-bound computation (sum of squares)."""
    total = sum(i * i for i in range(n))
    return total / n

# SEQUENTIAL baseline
print("  Running 8 tasks SEQUENTIALLY...")
t0 = time.time()
seq_results = [slow_computation.remote.__wrapped__(i) if hasattr(slow_computation, '__wrapped__')
               else sum(j*j for j in range(500_000))/500_000
               for i in range(8)]
t_seq = time.time() - t0
print(f"  Sequential time: {t_seq:.2f}s")

# PARALLEL with Ray
print("  Running 8 tasks in PARALLEL with Ray...")
t0 = time.time()
futures  = [slow_computation.remote(i) for i in range(8)]  # launch all, non-blocking
par_results = ray.get(futures)                               # wait for all
t_par = time.time() - t0
print(f"  Parallel time:   {t_par:.2f}s  (speedup: {t_seq/max(t_par,0.001):.1f}×)")
print()

# ── Object store ───────────────────────────────────────────────────────
print("  OBJECT STORE (shared memory between tasks):")
print()

# put() stores an object in Ray's distributed object store
# All workers can access it without copying — huge data is shared efficiently
large_array = np.random.randn(1000, 1000)   # 8 MB
ref         = ray.put(large_array)          # put into object store

print(f"  Array shape: {large_array.shape}  ({large_array.nbytes/1e6:.1f} MB)")
print(f"  Object reference: {ref}")
print(f"  (Ray auto-dereferences ObjectRefs passed to .remote() — no ray.get() inside the task)")

@ray.remote
def process_chunk(array_data, start: int, end: int) -> float:
    """Access shared data by reference — zero copy.
    NOTE: Ray auto-dereferences ObjectRefs passed to remote() calls,
    so the argument arrives here already as a numpy array — no ray.get() needed.
    """
    return float(array_data[start:end].mean())

chunk_futures = [process_chunk.remote(ref, i*200, (i+1)*200) for i in range(5)]
chunk_means   = ray.get(chunk_futures)
print(f"  Chunk means (5 parallel workers, shared data): {[f'{m:.4f}' for m in chunk_means]}")
print()

# ── Actors — stateful distributed objects ─────────────────────────────
print("  ACTORS (stateful distributed objects):")
print()

@ray.remote
class ParameterServer:
    """
    A simple parameter server — holds model weights, workers push gradients.
    State (self.weights) persists across method calls.
    """
    def __init__(self, size: int):
        self.weights  = np.zeros(size)
        self.n_updates = 0

    def apply_gradient(self, gradient: np.ndarray, lr: float = 0.01):
        self.weights  -= lr * gradient
        self.n_updates += 1

    def get_weights(self):
        return self.weights.copy()

    def stats(self):
        return {'n_updates': self.n_updates, 'weight_norm': float(np.linalg.norm(self.weights))}

ps = ParameterServer.remote(size=10)

# Simulate 5 workers each pushing a gradient
grads = [np.random.randn(10) for _ in range(5)]
for g in grads:
    ps.apply_gradient.remote(g)

final_weights = ray.get(ps.get_weights.remote())
stats         = ray.get(ps.stats.remote())
print(f"  ParameterServer after 5 gradient updates:")
print(f"    Weights: {final_weights[:5].round(4).tolist()} ...")
print(f"    Stats:   {stats}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: RAY TUNE — HYPERPARAMETER OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Ray Tune (hyperparameter search at scale)")
print("━" * 65)
print()

from ray import tune
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn as nn

# ── Toy task: train a 2-layer net on sinusoidal regression ───────────────
def make_sin_data(n=500):
    x = np.linspace(-np.pi, np.pi, n)
    y = np.sin(x) + 0.1 * np.random.randn(n)
    X = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
    Y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    return X, Y

def trainable(config):
    """
    The function Ray Tune calls for each trial.
    config: dict of hyperparameters chosen by the search algorithm.
    """
    X, Y = make_sin_data()

    model = nn.Sequential(
        nn.Linear(1, config['hidden']),
        nn.Tanh(),
        nn.Linear(config['hidden'], config['hidden']),
        nn.Tanh(),
        nn.Linear(config['hidden'], 1),
    )

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config.get('wd', 0.0))
    loss_fn   = nn.MSELoss()

    for epoch in range(30):
        model.train()
        pred  = model(X)
        loss  = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report to Tune after each epoch — enables early stopping
        # New Ray API: pass a dict, not keyword arguments
        tune.report({"loss": loss.item(), "epoch": epoch})

# ── Search space ──────────────────────────────────────────────────────────
search_space = {
    'lr':     tune.loguniform(1e-4, 1e-1),   # log-uniform: good for LR
    'hidden': tune.choice([16, 32, 64, 128]), # categorical
    'wd':     tune.loguniform(1e-6, 1e-2),   # log-uniform: good for wd
}

# ── ASHA Scheduler — kills bad trials early ───────────────────────────────
# ASHA (Async Successive Halving Algorithm):
#   - Run all trials for 1 epoch
#   - Keep top 1/reduction_factor, kill the rest
#   - Repeat with 2× epochs, keep top 1/reduction_factor again
#   - Result: strong configs get many epochs, weak configs die early
asha = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=30,                  # max epochs per trial
    grace_period=3,            # run at least 3 epochs before stopping
    reduction_factor=2,        # halve survivors each round
)

print(f"  Search space:")
print(f"    lr:     loguniform(1e-4, 1e-1)   ← 4 orders of magnitude")
print(f"    hidden: choice([16, 32, 64, 128])")
print(f"    wd:     loguniform(1e-6, 1e-2)")
print()
print(f"  Scheduler: ASHA (kills bad trials early, saves compute)")
print(f"  Running 12 trials (Ray handles parallelism automatically)...")
print()

tuner = tune.Tuner(
    trainable,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        scheduler=asha,
        num_samples=12,     # total trials
        # metric/mode are already set on ASHAScheduler — do NOT repeat here
    ),
    run_config=tune.RunConfig(verbose=0),   # ray.air.RunConfig is deprecated
)

results  = tuner.fit()

print(f"  Search complete. Results (all trials):")
print()
print(f"  {'Trial':>6} | {'LR':>10} | {'Hidden':>8} | {'WD':>12} | {'Best Loss':>12}")
print(f"  {'─'*58}")

for i, r in enumerate(results):
    cfg  = r.config
    loss = r.metrics.get('loss', float('nan')) if r.metrics else float('nan')
    print(f"  {i:6d} | {cfg['lr']:10.6f} | {cfg['hidden']:8d} | "
          f"{cfg.get('wd', 0):12.2e} | {loss:12.6f}")

print()
try:
    best_res  = results.get_best_result(metric="loss", mode="min")
    best_cfg  = best_res.config
    best_loss = best_res.metrics['loss']
    print(f"  BEST CONFIG:")
    print(f"     lr:     {best_cfg['lr']:.6f}")
    print(f"     hidden: {best_cfg['hidden']}")
    print(f"     wd:     {best_cfg.get('wd', 0):.2e}")
    print(f"     loss:   {best_loss:.6f}")
except RuntimeError as e:
    print(f"  Could not determine best trial: {e}")
print()
print("  ASHA EFFICIENCY vs RANDOM SEARCH:")
print("  Random search:  all 12 trials run for 30 epochs = 360 epoch-budget")
print("  ASHA:           most trials killed after 3–6 epochs")
print("  Typical saving: 3–10× compute for same search quality")
print()
print("  SCALING WITH RAY:")
print("  Same code runs on: laptop (1 CPU) → cloud node (32 CPU/8 GPU)")
print("  → multi-node cluster (1000 CPUs) — zero code change.")
print("  Add resource annotations: @ray.remote(num_gpus=1) to use GPUs.")

ray.shutdown()
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · Framework Comparison — Same Task in PyTorch, JAX & Keras": {
        "description": (
            "Side-by-side implementation of identical logistic regression in "
            "PyTorch (imperative OOP), JAX (functional), and Keras (declarative). "
            "Compare philosophy, parameter management, training step verbosity, "
            "and performance. Serves as a framework decision reference."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  FRAMEWORK COMPARISON — LOGISTIC REGRESSION IN 3 FRAMEWORKS")
print("=" * 65)
print()
print("  Same task, same data, three philosophies.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SHARED: Generate classification dataset
# ─────────────────────────────────────────────────────────────────────────
try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except (ImportError, ValueError) as e:
    # Two common causes on numpy 2.x:
    #   1. scikit-learn < 1.4: "cannot import name 'ComplexWarning'"
    #   2. pandas binary incompatibility: "numpy.dtype size changed"
    #      (pandas was compiled against numpy 1.x, now numpy 2.x is installed)
    print(f"  sklearn/pandas import failed: {e}")
    print()
    print("  Root cause: pandas and/or scikit-learn were compiled against")
    print("  numpy 1.x but numpy 2.x is now installed — binary mismatch.")
    print()
    print("  Fix (run these in your venv):")
    print("    pip install --upgrade pandas scikit-learn")
    print("  Requires: pandas >= 2.2  and  scikit-learn >= 1.4")
    raise SystemExit(0)

np.random.seed(42)
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                             n_classes=2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler  = StandardScaler().fit(X_tr)
X_tr    = scaler.transform(X_tr).astype(np.float32)
X_te    = scaler.transform(X_te).astype(np.float32)
y_tr    = y_tr.astype(np.int64)
y_te    = y_te.astype(np.int64)

EPOCHS = 50
LR     = 1e-2
HIDDEN = 64

print(f"  Dataset: 2000 samples, 20 features, 2 classes")
print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")
print(f"  Hyperparams: hidden={HIDDEN}, lr={LR}, epochs={EPOCHS}")
print()

results = {}

# ─────────────────────────────────────────────────────────────────────────
# FRAMEWORK 1: PyTorch  — imperative, OOP
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  FRAMEWORK 1 — PyTorch (imperative, object-oriented)")
print("━" * 65)
print()
print("  Philosophy: define layers as objects, loop manually, mutate state.")
print()

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

X_tr_t = torch.tensor(X_tr); y_tr_t = torch.tensor(y_tr)
X_te_t = torch.tensor(X_te); y_te_t = torch.tensor(y_te)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 2)
        )
    def forward(self, x):
        return self.net(x)

model_pt   = Net()
opt_pt     = torch.optim.Adam(model_pt.parameters(), lr=LR)
loss_fn_pt = nn.CrossEntropyLoss()
loader_pt  = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)

t0 = time.time()
for epoch in range(EPOCHS):
    model_pt.train()
    for Xb, yb in loader_pt:
        opt_pt.zero_grad()                    # ← step 1
        logits = model_pt(Xb)                 # ← step 2
        loss   = loss_fn_pt(logits, yb)       # ← step 3
        loss.backward()                       # ← step 4
        opt_pt.step()                         # ← step 5
t_pt = time.time() - t0

model_pt.eval()
with torch.no_grad():
    preds_pt = model_pt(X_te_t).argmax(1).numpy()
acc_pt = (preds_pt == y_te).mean()

print(f"  PyTorch:")
print(f"    Code style:  Class inheriting nn.Module, explicit 5-step loop")
print(f"    Params:      Hidden in model.parameters() — not directly visible")
print(f"    Grad engine: Dynamic graph built on every forward() call")
print(f"    Train time:  {t_pt:.3f}s ({EPOCHS} epochs)")
print(f"    Test acc:    {acc_pt*100:.2f}%")
print()
results['PyTorch'] = (t_pt, acc_pt)

# ─────────────────────────────────────────────────────────────────────────
# FRAMEWORK 2: JAX  — functional, pure functions
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  FRAMEWORK 2 — JAX (functional, pure functions)")
print("━" * 65)
print()
print("  Philosophy: params are data, operations are pure functions, compose transforms.")
print()

try:
    import jax
    import jax.numpy as jnp
    import optax

    def init_params(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            'w1': jax.random.normal(k1, (20, HIDDEN)) * 0.1,
            'b1': jnp.zeros(HIDDEN),
            'w2': jax.random.normal(k2, (HIDDEN, 2)) * 0.1,
            'b2': jnp.zeros(2),
        }

    def forward(params, x):
        h = jax.nn.relu(x @ params['w1'] + params['b1'])
        return h @ params['w2'] + params['b2']

    def loss_fn(params, x, y):
        logits    = forward(params, x)
        log_probs = jax.nn.log_softmax(logits)
        one_hot   = jax.nn.one_hot(y, 2)
        return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    grad_fn   = jax.jit(jax.grad(loss_fn))
    optimizer = optax.adam(LR)

    key    = jax.random.PRNGKey(42)
    params = init_params(key)
    state  = optimizer.init(params)

    X_jax  = jnp.array(X_tr); y_jax = jnp.array(y_tr)

    @jax.jit
    def train_step(params, state, X, y):
        grads           = grad_fn(params, X, y)
        updates, state  = optimizer.update(grads, state, params)
        params          = optax.apply_updates(params, updates)
        return params, state

    t0 = time.time()
    for _ in range(EPOCHS):
        params, state = train_step(params, state, X_jax, y_jax)
    jax.block_until_ready(params['w1'])
    t_jax = time.time() - t0

    logits_jax = forward(params, jnp.array(X_te))
    preds_jax  = jnp.argmax(logits_jax, axis=1)
    acc_jax    = float((preds_jax == jnp.array(y_te)).mean())

    print(f"  JAX:")
    print(f"    Code style:  Plain functions, params as dicts, no class needed")
    print(f"    Params:      Explicit Python dict — inspect/modify directly")
    print(f"    Grad engine: Functional transform (jax.grad) — composes infinitely")
    print(f"    Train time:  {t_jax:.3f}s ({EPOCHS} epochs, jit-compiled)")
    print(f"    Test acc:    {acc_jax*100:.2f}%")
    results['JAX'] = (t_jax, acc_jax)
    jax_ok = True
except ImportError:
    print(f"  JAX not available — pip install jax jaxlib optax")
    jax_ok = False
print()

# ─────────────────────────────────────────────────────────────────────────
# FRAMEWORK 3: Keras  — declarative, high-level
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  FRAMEWORK 3 — Keras (declarative, minimal boilerplate)")
print("━" * 65)
print()
print("  Philosophy: describe WHAT you want, not HOW to do it.")
print()

try:
    import keras

    model_k = keras.Sequential([
        keras.layers.Dense(HIDDEN, activation='relu', input_shape=(20,)),
        keras.layers.Dense(2, activation='softmax'),
    ])

    model_k.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    t0 = time.time()
    model_k.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=64, verbose=0)
    t_keras = time.time() - t0

    _, acc_keras = model_k.evaluate(X_te, y_te, verbose=0)

    print(f"  Keras:")
    print(f"    Code style:  Declarative (compile + fit) — no explicit loop")
    print(f"    Params:      Managed by framework — model.get_weights()")
    print(f"    Grad engine: TensorFlow GradientTape (hidden inside fit())")
    print(f"    Train time:  {t_keras:.3f}s ({EPOCHS} epochs)")
    print(f"    Test acc:    {acc_keras*100:.2f}%")
    results['Keras'] = (t_keras, acc_keras)
    keras_ok = True
except ImportError:
    print(f"  Keras not available — pip install keras tensorflow")
    keras_ok = False
print()

# ─────────────────────────────────────────────────────────────────────────
# COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SUMMARY — Framework Comparison")
print("━" * 65)
print()

print(f"  {'Framework':>12} | {'Train time':>12} | {'Test acc':>10} | {'Lines of code':>14}")
print(f"  {'─'*55}")
loc = {'PyTorch': 14, 'JAX': 16, 'Keras': 6}
for fw, (t, acc) in results.items():
    print(f"  {fw:>12} | {t:12.3f}s | {acc*100:9.2f}% | {loc.get(fw, '?'):>14}")

print()
print("  DECISION GUIDE:")
print("  ┌───────────────────────────────────────────────────────────────┐")
print("  │ Situation                       → Best Choice                 │")
print("  ├───────────────────────────────────────────────────────────────┤")
print("  │ Research, custom architectures  → PyTorch                     │")
print("  │ Mathematical / physics research → JAX                         │")
print("  │ Quick prototype or baseline     → Keras                       │")
print("  │ Pre-trained transformers        → HuggingFace (on PyTorch)    │")
print("  │ Multi-GPU / cluster training    → Ray Train (wraps any above) │")
print("  │ Fastest GPU inference           → TensorRT (from any above)   │")
print("  │ Mobile / edge deployment        → TFLite (from Keras/TF)      │")
print("  └───────────────────────────────────────────────────────────────┘")
print()
print("  PHILOSOPHICAL DIFFERENCES (memorise these):")
print()
print("  PyTorch:  YOU manage everything explicitly.")
print("            Loop, gradient steps, mode switching — all visible.")
print("            Feels like Python. Great for debugging.")
print()
print("  JAX:      Everything is a PURE FUNCTION TRANSFORM.")
print("            grad(jit(vmap(f))) — compose infinitely.")
print("            Requires functional discipline. Rewards it with speed.")
print()
print("  Keras:    YOU describe the structure, Keras does the rest.")
print("            Fastest to working code. Least flexible when you")
print("            need to do something non-standard.")
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
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }