"""
TensorFlow & Keras — Deep Learning in Python
=============================================

TensorFlow is Google's open-source platform for building, training, and
deploying machine learning models at any scale. Keras is its high-level API.

"""

import re
import textwrap


TOPIC_NAME   = "TensorFlow & Keras: Deep Learning in Python"
DISPLAY_NAME = "02 · TensorFlow & Keras"
ICON         = "🧠"
SUBTITLE     = "Tensors, Neural Networks, and the Art of Deep Learning"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is TensorFlow?

TensorFlow is an open-source machine learning framework originally developed by
the Google Brain team in 2015. It provides the infrastructure for numerical
computation — turning mathematical operations on large arrays of numbers into
code that can run on CPUs, GPUs, and TPUs.

Keras is TensorFlow's official high-level API. It wraps TensorFlow's lower-level
operations into a clean, human-readable interface for building, training, and
evaluating neural networks. As of TensorFlow 2.0, Keras is fully integrated and
is the recommended way to build models.

    import tensorflow as tf          # the framework
    from tensorflow import keras     # the high-level API
    from tensorflow.keras import layers  # building blocks


### Why TensorFlow and Deep Learning?

Classical ML (sklearn) excels at structured, tabular data. Deep learning
excels when:
    • The data is unstructured (images, audio, text, video)
    • There are millions of examples to learn from
    • The features are too complex to engineer manually

The reason is representation learning: instead of hand-crafting features,
a neural network learns its own hierarchical representations of the data
from the raw pixels or characters themselves.

    Classical ML:  you extract features → model learns mapping
    Deep Learning: model learns features AND mapping end-to-end


### Tensors — The Fundamental Data Structure

A tensor is a generalisation of arrays to any number of dimensions. In
TensorFlow, everything — data, weights, gradients, activations — is a tensor.

    Rank / Dimensions   Name         Example                   ML Use
    ─────────────────────────────────────────────────────────────────────────
    0-D (rank 0)        Scalar       tf.constant(3.14)         A single loss
    1-D (rank 1)        Vector       tf.constant([1, 2, 3])    One sample
    2-D (rank 2)        Matrix       tf.constant([[1,2],[3,4]])Batch of 1-D
    3-D (rank 3)        Tensor       shape (batch, seq, feat)  Text / time
    4-D (rank 4)        Tensor       shape (b, h, w, c)        Image batches
    N-D (rank N)        Tensor       any shape                 Video, etc.

TensorFlow tensors are immutable (like NumPy arrays). Mutable tensors are
called Variables (tf.Variable) and are used for model weights.

Key tensor attributes:
    tensor.shape    → the dimensions, e.g. TensorShape([32, 224, 224, 3])
    tensor.dtype    → element type: float32, int32, bool, string, etc.
    tensor.ndim     → number of dimensions (rank)
    tensor.numpy()  → convert to NumPy array


### The TensorFlow Ecosystem

    tf.constant / tf.Variable     → tensor creation
    tf.GradientTape               → automatic differentiation
    tf.data.Dataset               → efficient input pipelines
    tf.keras                      → model building API
    tf.keras.layers               → layer building blocks
    tf.keras.losses               → loss functions
    tf.keras.optimizers           → optimizers (gradient descent variants)
    tf.keras.metrics              → evaluation metrics
    tf.keras.callbacks            → training hooks (early stopping, etc.)
    tf.saved_model / model.save() → serialisation


### Why GPUs Matter for Deep Learning

A neural network's dominant computation is matrix multiplication:
    output = weights @ input + bias

A single training step for a large model involves billions of these
multiplications. GPUs (Graphics Processing Units) contain thousands of
small cores designed to perform floating-point operations in parallel —
exactly what matrix multiplication requires. A modern GPU can be 50-100×
faster than a CPU for this workload.

TPUs (Tensor Processing Units) are Google's custom chips designed
specifically for deep learning — even faster than GPUs for large batches.

TensorFlow abstracts away device placement:
    with tf.device('/GPU:0'):    # force to GPU
        result = model(input)


### What is a Neural Network?

A neural network is a function approximator — a mathematical function with
millions of learnable parameters that can be shaped to model virtually any
relationship between inputs and outputs.

Biologically inspired by the brain, it is organised into layers:

    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │ INPUT      │    │ HIDDEN     │    │ OUTPUT     │
    │ LAYER      │───▶│ LAYER(S)   │───▶│ LAYER      │
    │ (data)     │    │ (features) │    │ (pred.)    │
    └────────────┘    └────────────┘    └────────────┘

Each connection has a weight. Each neuron has a bias. Learning means
finding the values of all weights and biases that minimise prediction error.

A single neuron computes:
    z = W · x + b          (weighted sum + bias)
    a = activation(z)      (non-linear transformation)

The non-linearity is critical. Without it, stacking many layers would be
equivalent to a single linear transformation — no matter how deep the
network.


### Activation Functions

Activation functions introduce non-linearity, allowing networks to learn
complex, non-linear patterns.

    Function    Formula                   Range       Use Case
    ─────────────────────────────────────────────────────────────────────────
    ReLU        max(0, z)                 [0, ∞)      Default for hidden
    Leaky ReLU  max(0.01z, z)            (-∞, ∞)     Avoids dying ReLU
    ELU         z if z>0 else α(eᶻ−1)   (-α, ∞)     Smooth negative
    Sigmoid     1 / (1 + e^−z)           (0, 1)      Binary output prob.
    Tanh        (eᶻ − e^−ᶻ)/(eᶻ + e^−ᶻ) (−1, 1)     Zero-centred probs.
    Softmax     eᶻᵢ / Σeᶻⱼ              (0,1),Σ=1   Multi-class output
    Linear      z                         (−∞, ∞)     Regression output

    The dying ReLU problem: if z is always negative, ReLU outputs 0 and
    its gradient is 0 — the neuron never updates. Leaky ReLU and ELU fix
    this by allowing a small negative output.

    Sigmoid saturates for very large or very small inputs (gradient ≈ 0).
    This causes the vanishing gradient problem in deep networks. ReLU
    largely solved this for hidden layers.


### Loss Functions

The loss function measures how wrong the model's predictions are. Training
minimises this value. Choosing the right loss depends on the problem type:

    Problem              Loss Function               Keras Name
    ─────────────────────────────────────────────────────────────────────────
    Binary classif.      Binary Crossentropy         binary_crossentropy
    Multi-class          Categorical Crossentropy    categorical_crossentropy
    Multi-class (int y)  Sparse Cat. Crossentropy    sparse_categorical_crossentropy
    Regression           Mean Squared Error          mse / mean_squared_error
    Regression           Mean Absolute Error         mae / mean_absolute_error

    Binary Crossentropy:
        L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
        Perfect when ŷ = y: L = −log(1) = 0
        Worst case when ŷ = 1−y: L → ∞

    Categorical Crossentropy:
        L = −Σ yᵢ · log(ŷᵢ)    (over all classes)
        Penalises confident wrong predictions most severely.


### Optimisers — How Models Learn

An optimiser updates the model's weights after each batch using the
gradients computed by backpropagation.

    Optimiser   Update Rule (simplified)                Notes
    ─────────────────────────────────────────────────────────────────────────
    SGD         w ← w − η·∇L                           Simple, noisy
    Momentum    v ← β·v − η·∇L; w ← w + v             Smoother than SGD
    RMSProp     v ← β·v + (1−β)·(∇L)²                 Adaptive lr/feature
                w ← w − η·∇L/√(v + ε)
    Adam        Combines Momentum + RMSProp             Default choice
                w ← w − η·m̂/(√v̂ + ε)

Adam (Adaptive Moment Estimation) is the default for most deep learning
because it adapts the learning rate per parameter and converges reliably.

    Learning rate (η) — the single most important hyperparameter:
        Too high  → overshoots minima, diverges (loss oscillates/explodes)
        Too low   → converges very slowly, may get stuck in local minima
        Typical   → 1e-3 (Adam), 1e-2 to 1e-1 (SGD)


### Backpropagation — How Gradients Flow

Backpropagation is the algorithm that computes how much each weight
contributed to the total error, so the optimiser knows how to update it.

It works using the chain rule of calculus:

    Forward pass:  input → layers → prediction → loss
    Backward pass: loss → ∂L/∂weights for every weight in every layer

    ∂L/∂w₁ = ∂L/∂a₂ × ∂a₂/∂z₂ × ∂z₂/∂a₁ × ∂a₁/∂z₁ × ∂z₁/∂w₁

In TensorFlow, this is automated by `tf.GradientTape`. You record the
forward pass inside a tape context; calling `tape.gradient(loss, weights)`
computes all gradients automatically via automatic differentiation.

    The vanishing gradient problem: in very deep networks, gradients can
    become exponentially small as they flow backward (especially with
    sigmoid/tanh). This means early layers learn extremely slowly.
    Solutions: ReLU, skip connections (ResNet), batch normalisation.

    The exploding gradient problem: gradients become exponentially large,
    causing weight updates to overshoot wildly. Solutions: gradient
    clipping, careful weight initialisation.


### Building Models — Sequential vs Functional API

Keras provides two main ways to define a model:

    Sequential API — for linear stacks of layers:
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax'),
        ])

    Functional API — for any directed acyclic graph of layers:
        inputs  = keras.Input(shape=(784,))
        x       = layers.Dense(128, activation='relu')(inputs)
        x       = layers.Dropout(0.2)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        model   = keras.Model(inputs=inputs, outputs=outputs)

    Use Sequential for simple, linear architectures.
    Use Functional for: multiple inputs/outputs, branching (Inception),
    skip connections (ResNet), shared layers, or layer reuse.


### Key Layers

    Layer Type          What it does                        Keras class
    ─────────────────────────────────────────────────────────────────────────
    Dense               Fully connected: y = Wx + b         layers.Dense
    Conv2D              Sliding-window feature detector      layers.Conv2D
    MaxPooling2D        Spatial downsampling (max)           layers.MaxPooling2D
    Flatten             Reshape 3-D → 1-D                   layers.Flatten
    Dropout             Zero random neurons (regularisation) layers.Dropout
    BatchNorm           Normalise activations per batch      layers.BatchNormalization
    Embedding           Map integers to dense vectors        layers.Embedding
    LSTM / GRU          Recurrent layers for sequences       layers.LSTM
    MultiHeadAttention  Attention mechanism (Transformers)   layers.MultiHeadAttention


### Convolutional Neural Networks (CNNs)

CNNs are the standard architecture for image tasks. Instead of connecting
every pixel to every neuron (which would require billions of parameters),
a convolutional layer slides a small filter (kernel) across the image and
computes the dot product at each position.

    A 3×3 filter applied to a 28×28 image:
        Output size = (28 − 3 + 1) × (28 − 3 + 1) = 26×26
        With padding='same': output stays 28×28

    Multiple filters → multiple feature maps. Each filter detects a
    different visual pattern: edges, textures, shapes.

    Typical CNN stack:
        Conv2D → ReLU → Conv2D → ReLU → MaxPool  (feature extraction)
        ↕ repeat N times
        Flatten → Dense → Softmax                 (classification head)

    Pooling reduces spatial dimensions, reducing computation and providing
    some translation invariance (the feature fires regardless of exact position).

    Why CNNs work: parameter sharing (same filter applied everywhere) and
    local connectivity (filters see local patches) dramatically reduce
    parameter count compared to fully connected networks.


### Overfitting and Regularisation

Overfitting: the model memorises training examples rather than learning
generalisable patterns. Signs: high train accuracy, much lower val accuracy.

Strategies to combat overfitting:

    Strategy            How it works                          Keras
    ─────────────────────────────────────────────────────────────────────────
    More data           The most reliable fix. Always try first.
    Data augmentation   Randomly transform training images     ImageDataGenerator
    Dropout             Randomly zero neurons during training  layers.Dropout(rate)
    L2 regularisation   Penalise large weights in loss         kernel_regularizer=l2(λ)
    L1 regularisation   Promotes sparsity in weights           kernel_regularizer=l1(λ)
    Batch Normalisation Normalises activations, stabilises     layers.BatchNormalization
    Early Stopping      Stop when val loss stops improving     callbacks.EarlyStopping
    Reduce model size   Fewer parameters = less capacity to overfit

    Dropout: at each training step, each neuron is randomly "dropped"
    (set to 0) with probability `rate`. At inference, all neurons are
    active and outputs are scaled by (1−rate). This prevents co-adaptation:
    neurons cannot rely on specific other neurons → forced to learn
    redundant, robust representations.

    L2 regularisation adds λ·Σw² to the loss. The gradient of this term
    (2λw) shrinks weights toward zero at every step — also called "weight
    decay". Prevents any single weight from dominating the prediction.


### Batch Normalisation

BatchNorm normalises the inputs to each layer within a mini-batch:

    x̂ = (x − μ_batch) / √(σ²_batch + ε)    (normalise)
    y  = γ·x̂ + β                            (scale + shift, learnable)

Benefits:
    • Reduces internal covariate shift (layer inputs stay in a stable range)
    • Acts as a mild regulariser
    • Allows higher learning rates
    • Reduces sensitivity to weight initialisation

BatchNorm is typically placed AFTER the linear operation and BEFORE the
activation function: Conv2D → BatchNorm → ReLU.


### Training Loop — Epochs, Batches, Steps

    Dataset size: N samples
    Batch size:   B samples per update
    Steps per epoch: N / B

    One epoch = one complete pass through all N training samples.
    One step  = one forward + backward pass on one batch of B samples.

    Total weight updates per epoch = N / B

    Large batch advantages:   more stable gradients, faster per-epoch training
    Small batch advantages:   noise helps escape local minima, less GPU memory
    Standard batch sizes:     32, 64, 128, 256 (powers of 2 for GPU efficiency)

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    Returns a History object: history.history is a dict of metric arrays
    (one value per epoch) — useful for plotting learning curves.


### Callbacks

Callbacks are objects that perform actions at various stages of training.

    Callback              Trigger                     Use
    ─────────────────────────────────────────────────────────────────────────
    EarlyStopping         val_loss stops decreasing   Prevent overfitting
    ModelCheckpoint       After each epoch            Save best weights
    ReduceLROnPlateau     val_loss stagnates          Reduce learning rate
    TensorBoard           End of each epoch           Visualise training
    LambdaCallback        Custom                      Any custom action

    EarlyStopping:
        monitor='val_loss'   → watch validation loss
        patience=5           → stop after 5 epochs of no improvement
        restore_best_weights → rollback to best checkpoint before stopping


### Transfer Learning

Training a deep CNN from scratch requires millions of labelled images and
days of GPU time. Transfer learning takes the weights of a model already
trained on a large dataset (e.g. ImageNet — 1.4M images, 1000 classes)
and reuses them.

Two modes:

    Feature Extraction (frozen base):
        Take a pretrained backbone (e.g. MobileNetV2, ResNet50).
        Freeze all its weights (base.trainable = False).
        Add a new classification head on top.
        Train ONLY the new head on your data.
        Fast, requires few samples, prevents catastrophic forgetting.

    Fine-tuning (unfrozen base):
        After training the head, unfreeze the top N layers of the base.
        Continue training with a very small learning rate.
        The base adapts its learned features to your specific domain.
        Requires more data, risks forgetting.

Why it works: early layers of CNNs learn universal features (edges, textures,
colours) that transfer to almost any vision task. Only the higher-level,
task-specific representations need to change.


### The tf.data.Dataset API

Loading all images into memory is impossible for large datasets. `tf.data`
provides a lazy, pipeline-based approach:

    tf.data.Dataset.from_tensor_slices(filenames)  # start from file list
        .map(preprocess_fn, num_parallel_calls=AUTOTUNE)  # preprocess in parallel
        .shuffle(buffer_size=1000)                 # random shuffle
        .batch(32)                                 # group into batches
        .prefetch(AUTOTUNE)                        # pre-load next batch while GPU trains

    AUTOTUNE = tf.data.AUTOTUNE  # TF chooses optimal parallelism

    prefetch() is especially important: while the GPU processes batch N,
    the CPU loads and preprocesses batch N+1 in parallel. Without prefetch,
    the GPU sits idle while the CPU catches up.


### Saving and Loading Models

TensorFlow supports two formats:

    SavedModel (recommended):
        model.save('my_model')           # saves directory
        model = tf.keras.models.load_model('my_model')

    HDF5 (.h5):
        model.save('my_model.h5')
        model = tf.keras.models.load_model('my_model.h5')

    SavedModel stores the architecture, weights, AND the computation graph.
    It is framework-agnostic (can be served by TF Serving, TF Lite, etc.).

    Save only weights:
        model.save_weights('weights.h5')
        model.load_weights('weights.h5')  # architecture must already exist

    ModelCheckpoint callback auto-saves the best model during training:
        callbacks.ModelCheckpoint('best.keras', save_best_only=True)

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 01 ─────────────────────────────────────────────────────────────────
    "01 · TensorFlow Setup and Tensor Basics": {
        "description": (
            "Import TensorFlow, inspect tensors and their key attributes. "
            "Tensors are the universal data structure — every input, weight, "
            "activation, gradient, and prediction in a neural network is a tensor."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            import numpy as np

            print(f"TensorFlow version : {tf.__version__}")
            print(f"Keras version      : {tf.keras.__version__}")
            print(f"GPU available      : {bool(tf.config.list_physical_devices('GPU'))}")

            # ── Creating tensors ──────────────────────────────────────────────
            scalar  = tf.constant(3.14)                          # rank-0
            vector  = tf.constant([1, 2, 3])                     # rank-1
            matrix  = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)  # rank-2
            tensor3 = tf.zeros([2, 3, 4])                        # rank-3
            tensor4 = tf.random.normal([8, 28, 28, 1])           # rank-4 (images)

            for name, t in [("scalar", scalar), ("vector", vector),
                            ("matrix", matrix), ("tensor3", tensor3), ("tensor4", tensor4)]:
                print(f"  {name:8s}: shape={str(t.shape):18s} ndim={t.ndim}  dtype={t.dtype.name}")

            # ── Key attributes ─────────────────────────────────────────────
            print("\\n--- Matrix details ---")
            print(f"  shape     : {matrix.shape}")
            print(f"  dtype     : {matrix.dtype}")
            print(f"  ndim      : {matrix.ndim}")
            print(f"  as numpy  : \\n{matrix.numpy()}")

            # ── Immutable tensors vs mutable Variables ──────────────────────
            w = tf.Variable([[1.0, 2.0], [3.0, 4.0]])   # learnable weights
            print(f"\\nVariable (mutable)  : {w.numpy()}")
            w.assign_add([[0.1, 0.1], [0.1, 0.1]])       # in-place update
            print(f"After assign_add    : {w.numpy()}")

            # ── NumPy interop ───────────────────────────────────────────────
            arr = np.array([1.0, 2.0, 3.0])
            tf_from_np = tf.constant(arr)           # numpy → tensor
            np_from_tf = tf_from_np.numpy()         # tensor → numpy
            print(f"\\nNumPy → Tensor → NumPy: {np_from_tf}")
        ''',
    },

    # ── 02 ─────────────────────────────────────────────────────────────────
    "02 · Tensor Operations and Broadcasting": {
        "description": (
            "All arithmetic, matrix, and mathematical operations on tensors. "
            "TensorFlow mirrors NumPy's broadcasting rules — smaller tensors "
            "are stretched to match larger ones without copying data."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf

            a = tf.constant([[1., 2., 3.], [4., 5., 6.]])   # shape (2, 3)
            b = tf.constant([[1., 0., 1.]])                  # shape (1, 3)

            # ── Element-wise operations ────────────────────────────────────
            print("a + b (broadcast):\\n", (a + b).numpy())
            print("a * 2.0 (scalar broadcast):\\n", (a * 2.0).numpy())
            print("tf.square(a):\\n", tf.square(a).numpy())
            print("tf.sqrt(a):\\n", tf.sqrt(a).numpy().round(3))

            # ── Matrix multiplication — the core neural network operation ───
            W = tf.constant([[1., 0.], [0., 1.], [2., -1.]])  # (3, 2)
            x = tf.constant([[1.], [2.], [3.]])                # (3, 1)
            print("\\nW shape:", W.shape, "  x shape:", x.shape)

            # (3,2)ᵀ @ (3,1) → (2,1)  — not valid
            # Transpose W first: (2,3) @ (3,1) → (2,1)
            result = tf.matmul(tf.transpose(W), x)
            print("Wᵀ @ x = ", result.numpy().flatten())

            # @-operator is shorthand for tf.matmul
            result2 = tf.transpose(W) @ x
            print("Same via @  =", result2.numpy().flatten())

            # ── Reduction operations ───────────────────────────────────────
            print("\\nReductions on a:")
            print(f"  tf.reduce_sum(a)          = {tf.reduce_sum(a).numpy():.1f}")
            print(f"  tf.reduce_mean(a)         = {tf.reduce_mean(a).numpy():.4f}")
            print(f"  tf.reduce_max(a, axis=1)  = {tf.reduce_max(a, axis=1).numpy()}")
            print(f"  tf.argmax(a, axis=1)      = {tf.argmax(a, axis=1).numpy()}")

            # ── Reshape / transpose ────────────────────────────────────────
            flat   = tf.reshape(a, [-1])           # -1 = infer
            col    = tf.reshape(a, [6, 1])
            transposed = tf.transpose(a)           # (2,3) → (3,2)
            print(f"\\nReshape to [-1]    : {flat.numpy()}")
            print(f"Transpose shape    : {transposed.shape}")

            # ── Activation functions from tf ───────────────────────────────
            z = tf.constant([-2., -1., 0., 1., 2.])
            print("\\nActivation functions on z =", z.numpy())
            print("  ReLU   :", tf.nn.relu(z).numpy())
            print("  Sigmoid:", tf.nn.sigmoid(z).numpy().round(3))
            print("  Tanh   :", tf.nn.tanh(z).numpy().round(3))
            print("  Softmax:", tf.nn.softmax(z).numpy().round(3))
        ''',
    },

    # ── 03 ─────────────────────────────────────────────────────────────────
    "03 · Automatic Differentiation with GradientTape": {
        "description": (
            "GradientTape records operations during a forward pass so that "
            "TensorFlow can compute exact gradients via automatic differentiation. "
            "This is the engine behind backpropagation — shown here both manually "
            "and in the context of a single-step gradient descent update."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            import numpy as np

            # ── Basic gradient: d(y)/d(x) where y = x² ────────────────────
            x = tf.Variable(3.0)
            with tf.GradientTape() as tape:
                y = x ** 2          # forward pass: y = x²
            dy_dx = tape.gradient(y, x)
            print(f"x = {x.numpy():.1f}   y = x² = {y.numpy():.1f}")
            print(f"dy/dx = 2x = {dy_dx.numpy():.1f}  (exact: 2×3 = 6)")

            # ── Gradient of a loss w.r.t. a weight matrix ──────────────────
            W = tf.Variable(tf.random.normal([3, 2], seed=42))
            x_in = tf.constant([[1.0, 2.0, 3.0]])    # shape (1, 3)
            y_true = tf.constant([[1.0, 0.0]])         # shape (1, 2)

            with tf.GradientTape() as tape:
                y_pred = tf.nn.softmax(x_in @ W)       # forward pass
                loss   = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-8))

            grad_W = tape.gradient(loss, W)
            print(f"\\nW shape      : {W.shape}")
            print(f"Loss before  : {loss.numpy():.4f}")
            print(f"grad_W shape : {grad_W.shape}")
            print(f"grad_W sample: {grad_W.numpy()[0].round(4)}")

            # ── One step of gradient descent ───────────────────────────────
            lr = 0.1
            W.assign_sub(lr * grad_W)   # w ← w - η·∇L

            # Recompute loss after update
            y_pred_after = tf.nn.softmax(x_in @ W)
            loss_after   = -tf.reduce_sum(y_true * tf.math.log(y_pred_after + 1e-8))
            print(f"Loss after 1 step : {loss_after.numpy():.4f}  (should be lower)")

            # ── Watching non-Variable tensors ──────────────────────────────
            # By default tape only watches tf.Variables. Use tape.watch() for tensors.
            x_const = tf.constant(2.0)
            with tf.GradientTape() as tape:
                tape.watch(x_const)
                y = x_const ** 3    # y = x³, dy/dx = 3x² = 12
            grad = tape.gradient(y, x_const)
            print(f"\\nd(x³)/dx at x=2 : {grad.numpy():.1f}  (exact: 3×4 = 12)")

            # ── Persistent tape for multiple gradients ─────────────────────
            w1 = tf.Variable(1.0)
            w2 = tf.Variable(2.0)
            with tf.GradientTape(persistent=True) as tape:
                z = w1 * w2 + w2 ** 2
            print(f"\\ndz/dw1 = {tape.gradient(z, w1).numpy():.1f}  (= w2 = 2)")
            print(f"dz/dw2 = {tape.gradient(z, w2).numpy():.1f}  (= w1 + 2w2 = 5)")
            del tape
        ''',
    },

    # ── 04 ─────────────────────────────────────────────────────────────────
    "04 · Building Models — Sequential API": {
        "description": (
            "The Sequential API builds a linear stack of layers — the simplest "
            "and most common pattern for feedforward networks. Covers layer types, "
            "input shapes, trainable parameters, and model.summary()."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models

            # ── Fully connected network for MNIST (28×28 grayscale images) ─
            model = models.Sequential([
                # Flatten converts the 2-D image to a 1-D vector
                layers.Flatten(input_shape=(28, 28)),           # 784 inputs

                # Dense layer: 784 → 256 neurons, ReLU activation
                layers.Dense(256, activation="relu"),

                # Dropout for regularisation (20% neurons dropped per step)
                layers.Dropout(0.2),

                # Dense layer: 256 → 128 neurons
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),

                # Output layer: 128 → 10 classes, Softmax gives probabilities
                layers.Dense(10, activation="softmax"),
            ], name="mnist_classifier")

            model.summary()

            # ── Understanding the parameter count ─────────────────────────
            # Dense(256): 784 inputs × 256 neurons + 256 biases = 200,960
            # Dense(128): 256 inputs × 128 neurons + 128 biases =  32,896
            # Dense(10) : 128 inputs × 10 neurons  + 10 biases  =   1,290
            params_manual = (784*256 + 256) + (256*128 + 128) + (128*10 + 10)
            print(f"\\nManual param count : {params_manual:,}")
            print(f"Model total params : {model.count_params():,}")

            # ── Layer introspection ────────────────────────────────────────
            print("\\nLayer details:")
            for layer in model.layers:
                cfg = layer.get_config()
                units = cfg.get("units", cfg.get("rate", "—"))
                print(f"  {layer.name:20s}  output={str(layer.output_shape):20s}  "
                      f"params={layer.count_params():,}")

            # ── Alternative: add layers one at a time ──────────────────────
            model2 = models.Sequential(name="alternative")
            model2.add(layers.Input(shape=(784,)))
            model2.add(layers.Dense(64, activation="relu"))
            model2.add(layers.Dense(10, activation="softmax"))
            print(f"\\nAlternative model params: {model2.count_params():,}")
        ''',
    },

    # ── 05 ─────────────────────────────────────────────────────────────────
    "05 · Building Models — Functional API": {
        "description": (
            "The Functional API treats layers as callable functions on tensors, "
            "enabling any directed acyclic graph — multiple inputs/outputs, "
            "branching paths, and skip connections. Essential for ResNet, "
            "Inception, and multi-task models."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models, Input

            # ── Simple functional model (equivalent to Sequential) ─────────
            inputs  = Input(shape=(784,), name="pixel_input")
            x       = layers.Dense(256, activation="relu", name="hidden_1")(inputs)
            x       = layers.Dropout(0.2)(x)
            x       = layers.Dense(128, activation="relu", name="hidden_2")(x)
            outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

            model = models.Model(inputs=inputs, outputs=outputs,
                                 name="functional_mnist")
            model.summary()

            # ── Multi-output model ─────────────────────────────────────────
            # Example: shared feature extractor with two task heads
            img_input = Input(shape=(128,), name="features")

            shared = layers.Dense(64, activation="relu")(img_input)
            shared = layers.Dense(32, activation="relu")(shared)

            # Head 1: classify (10 classes)
            class_out = layers.Dense(10, activation="softmax", name="class_output")(shared)

            # Head 2: predict a continuous value (regression)
            reg_out   = layers.Dense(1, activation="linear", name="reg_output")(shared)

            multi_model = models.Model(inputs=img_input,
                                       outputs=[class_out, reg_out],
                                       name="multi_task")

            print("\\n--- Multi-task model ---")
            print(f"Outputs: {[o.name for o in multi_model.outputs]}")
            print(f"Total params: {multi_model.count_params():,}")

            # ── Skip connection (ResNet-style residual block) ──────────────
            def residual_block(x, filters=64):
                shortcut = x                                       # skip
                x = layers.Dense(filters, activation="relu")(x)
                x = layers.Dense(filters, activation="relu")(x)
                x = layers.Add()([x, shortcut])                    # add skip
                x = layers.Activation("relu")(x)
                return x

            res_in  = Input(shape=(64,))
            res_out = residual_block(res_in)
            res_out = residual_block(res_out)
            res_out = layers.Dense(10, activation="softmax")(res_out)
            res_model = models.Model(res_in, res_out, name="residual_net")

            print("\\n--- Residual model ---")
            print(f"Total params: {res_model.count_params():,}")
            print("Skip connections bypass layers — gradients flow directly")
            print("to early layers, solving the vanishing gradient problem.")
        ''',
    },

    # ── 06 ─────────────────────────────────────────────────────────────────
    "06 · Activation Functions — Visualised and Compared": {
        "description": (
            "Activation functions add non-linearity — without them, any number "
            "of stacked linear layers collapses to a single linear transform. "
            "Each activation has different properties, gradients, and use cases."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            z = np.linspace(-4, 4, 200)
            zt = tf.constant(z, dtype=tf.float32)

            activations = {
                "ReLU":       tf.nn.relu(zt).numpy(),
                "Leaky ReLU": tf.nn.leaky_relu(zt, alpha=0.1).numpy(),
                "Sigmoid":    tf.nn.sigmoid(zt).numpy(),
                "Tanh":       tf.nn.tanh(zt).numpy(),
                "ELU":        tf.nn.elu(zt).numpy(),
                "Softplus":   tf.nn.softplus(zt).numpy(),
            }

            # ── Numerical properties ───────────────────────────────────────
            print("Activation properties at z=0:")
            print(f"  {'Name':<12} f(0)    f(−1)   f(1)    Range")
            for name, vals in activations.items():
                idx_0  = np.argmin(np.abs(z))
                idx_n1 = np.argmin(np.abs(z + 1))
                idx_p1 = np.argmin(np.abs(z - 1))
                rng = f"[{vals.min():.2f}, {vals.max():.2f}]"
                print(f"  {name:<12} {vals[idx_0]:.4f}  {vals[idx_n1]:.4f}  "
                      f"{vals[idx_p1]:.4f}  {rng}")

            # ── Softmax for multi-class (operates on a vector, not scalar) ──
            logits = tf.constant([2.0, 1.0, 0.5, -1.0])
            probs  = tf.nn.softmax(logits)
            print(f"\nSoftmax demo:")
            print(f"  logits: {logits.numpy()}")
            print(f"  probs : {probs.numpy().round(3)}")
            print(f"  sum   : {probs.numpy().sum():.4f}  (always 1.0)")

            # ── Plot ───────────────────────────────────────────────────────
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            colors = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9", "#8e44ad", "#16a085"]
            for ax, (name, vals), color in zip(axes.flat, activations.items(), colors):
                ax.plot(z, vals, color=color, lw=2)
                ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
                ax.set_title(name, fontsize=12, fontweight="bold")
                ax.set_xlim(-4, 4); ax.grid(alpha=0.3)
                # highlight gradient at z=1
                dv = np.gradient(vals, z)
                ax.annotate(f"f'(1)≈{dv[np.argmin(np.abs(z-1))]:.2f}",
                            xy=(1, vals[np.argmin(np.abs(z-1))]),
                            fontsize=8, color="gray")
            plt.suptitle("Activation Functions", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig("activations.png", dpi=100)
            plt.show()
            print("\nActivation plot saved to activations.png")
        ''',
    },

    # ── 07 ─────────────────────────────────────────────────────────────────
    "07 · Loss Functions — What Models Minimise": {
        "description": (
            "The loss function translates prediction error into a scalar that "
            "the optimiser minimises. Choosing the right loss is as important "
            "as choosing the right architecture — wrong loss → wrong objective."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from tensorflow.keras import losses

            print("=== CLASSIFICATION LOSSES ===")

            # ── Binary Crossentropy ────────────────────────────────────────
            bce = losses.BinaryCrossentropy()
            y_true_bin = tf.constant([1., 0., 1., 1.])
            # Perfect prediction
            bce_perfect = bce(y_true_bin, y_true_bin).numpy()
            # Reasonable prediction
            y_pred_bin  = tf.constant([0.9, 0.1, 0.8, 0.7])
            bce_good    = bce(y_true_bin, y_pred_bin).numpy()
            # Terrible prediction (high confidence wrong answer)
            y_pred_bad  = tf.constant([0.05, 0.95, 0.1, 0.1])
            bce_bad     = bce(y_true_bin, y_pred_bad).numpy()
            print(f"Binary CE — perfect: {bce_perfect:.4f}  "
                  f"good: {bce_good:.4f}  bad: {bce_bad:.4f}")

            # ── Categorical Crossentropy ────────────────────────────────────
            cce = losses.CategoricalCrossentropy()
            y_true_cat = tf.constant([[0., 0., 1.], [1., 0., 0.]])  # one-hot
            y_pred_cat = tf.constant([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])
            cce_good   = cce(y_true_cat, y_pred_cat).numpy()
            print(f"Categorical CE — good predictions: {cce_good:.4f}")

            # Sparse version: y_true as integers
            scce = losses.SparseCategoricalCrossentropy()
            y_true_int = tf.constant([2, 0])   # class indices (not one-hot)
            scce_val   = scce(y_true_int, y_pred_cat).numpy()
            print(f"Sparse Cat. CE — same result:      {scce_val:.4f}")

            print("\n=== REGRESSION LOSSES ===")

            y_true_reg = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
            y_pred_reg = tf.constant([1.1, 1.8, 3.2, 3.6, 5.5])
            errors     = (y_pred_reg - y_true_reg).numpy()
            print(f"Errors: {errors.round(2)}")

            mse_val = losses.MeanSquaredError()(y_true_reg, y_pred_reg).numpy()
            mae_val = losses.MeanAbsoluteError()(y_true_reg, y_pred_reg).numpy()
            print(f"MSE : {mse_val:.4f}  (squares errors → outlier-sensitive)")
            print(f"MAE : {mae_val:.4f}  (linear errors  → robust to outliers)")

            # Outlier effect: add one large error
            y_pred_outlier = tf.constant([1.1, 1.8, 3.2, 3.6, 15.0])  # last is 10 off
            mse_out = losses.MeanSquaredError()(y_true_reg, y_pred_outlier).numpy()
            mae_out = losses.MeanAbsoluteError()(y_true_reg, y_pred_outlier).numpy()
            print(f"\nWith large outlier (pred=15 for true=5):")
            print(f"  MSE jumped from {mse_val:.3f} to {mse_out:.3f}  (+{mse_out-mse_val:.3f})")
            print(f"  MAE jumped from {mae_val:.3f} to {mae_out:.3f}  (+{mae_out-mae_val:.3f})")
            print("MSE is far more sensitive to outliers due to squaring.")

            # ── Visualise CCE as a function of confidence ──────────────────
            confs = np.linspace(0.01, 0.99, 100)
            cce_curve = -np.log(confs)   # L = -log(p_true)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(confs, cce_curve, color="#e74c3c", lw=2)
            ax.set_xlabel("Predicted probability for true class")
            ax.set_ylabel("Crossentropy loss")
            ax.set_title("Categorical Crossentropy: L = −log(p_true)")
            ax.axvline(0.5, color="gray", linestyle="--", label="50% confidence")
            ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig("loss_curve.png", dpi=100); plt.show()
            print("\nLoss curve saved to loss_curve.png")
        ''',
    },

    # ── 08 ─────────────────────────────────────────────────────────────────
    "08 · Optimisers — Gradient Descent and its Variants": {
        "description": (
            "Optimisers update weights using gradients to minimise the loss. "
            "SGD is the conceptual foundation; Adam is the practical default. "
            "This operation demonstrates how each optimiser traces a different "
            "path through the loss landscape."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from tensorflow.keras import optimizers

            # ── Quadratic loss surface: L(w) = (w - 3)² ────────────────────
            def loss_fn(w):
                return (w - 3.0) ** 2

            def grad_fn(w):
                return 2.0 * (w - 3.0)   # dL/dw = 2(w - 3)

            # True minimum is at w = 3.0, L_min = 0

            optimiser_cfgs = {
                "SGD (lr=0.1)":             optimizers.SGD(learning_rate=0.1),
                "SGD+Momentum (lr=0.1)":    optimizers.SGD(learning_rate=0.1, momentum=0.9),
                "Adam (lr=0.5)":            optimizers.Adam(learning_rate=0.5),
                "RMSprop (lr=0.3)":         optimizers.RMSprop(learning_rate=0.3),
            }

            n_steps = 20
            histories = {}

            for name, opt in optimiser_cfgs.items():
                w = tf.Variable(-2.0)       # start far from minimum (w = 3)
                w_hist = [w.numpy()]
                for _ in range(n_steps):
                    with tf.GradientTape() as tape:
                        loss = loss_fn(w)
                    grad = tape.gradient(loss, w)
                    opt.apply_gradients([(grad, w)])
                    w_hist.append(w.numpy())
                histories[name] = w_hist
                print(f"{name:<30}: final w={w.numpy():.4f}  "
                      f"final loss={loss_fn(w).numpy():.6f}")

            # ── Plot convergence paths ──────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            colors = ["#e74c3c", "#e67e22", "#2980b9", "#27ae60"]

            for (name, hist), color in zip(histories.items(), colors):
                losses = [loss_fn(tf.constant(w)).numpy() for w in hist]
                axes[0].plot(hist,   label=name, color=color, marker="o", markersize=3)
                axes[1].plot(losses, label=name, color=color, marker="o", markersize=3)

            axes[0].axhline(3.0, color="k", linestyle="--", lw=1, label="Minimum w=3")
            axes[0].set_title("Weight value over steps"); axes[0].set_xlabel("Step")
            axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

            axes[1].set_title("Loss over steps"); axes[1].set_xlabel("Step")
            axes[1].set_yscale("log"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

            plt.suptitle("Optimiser Convergence Comparison", fontweight="bold")
            plt.tight_layout()
            plt.savefig("optimiser_comparison.png", dpi=100); plt.show()
            print("\nPlot saved to optimiser_comparison.png")

            # ── Key takeaways ──────────────────────────────────────────────
            print("\nKey points:")
            print("  Adam    : adaptive lr per parameter → usually fastest convergence")
            print("  SGD+Mom : reliable for vision tasks with tuned lr schedules")
            print("  RMSprop : good for RNNs and noisy gradients")
            print("  Learning rate is the most critical hyperparameter to tune")
        ''',
    },

    # ── 09 ─────────────────────────────────────────────────────────────────
    "09 · Compiling a Model — Loss, Optimiser, Metrics": {
        "description": (
            "Compilation binds the training configuration to the model: which "
            "loss to minimise, which optimiser to use, and which metrics to track. "
            "Covers all common combinations across classification and regression."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, losses, metrics

            # ── Binary classification ──────────────────────────────────────
            bin_model = models.Sequential([
                layers.Dense(64, activation="relu", input_shape=(20,)),
                layers.Dense(1, activation="sigmoid"),   # single output in (0,1)
            ])
            bin_model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-3),
                loss=losses.BinaryCrossentropy(),
                metrics=[metrics.BinaryAccuracy(), metrics.AUC()]
            )
            print("Binary classifier compiled:")
            print(f"  loss    : {bin_model.loss}")
            print(f"  metrics : {[m.name for m in bin_model.metrics]}")

            # ── Multi-class classification ─────────────────────────────────
            multi_model = models.Sequential([
                layers.Dense(128, activation="relu", input_shape=(784,)),
                layers.Dense(10, activation="softmax"),  # 10 class probabilities
            ])
            multi_model.compile(
                optimizer="adam",                          # string shorthand
                loss="sparse_categorical_crossentropy",    # y_true as integers
                metrics=["accuracy"]
            )
            print("\nMulti-class classifier compiled:")
            print(f"  loss    : {multi_model.loss}")
            print(f"  metrics : {[m.name for m in multi_model.metrics]}")

            # ── Regression ────────────────────────────────────────────────
            reg_model = models.Sequential([
                layers.Dense(64, activation="relu", input_shape=(8,)),
                layers.Dense(1, activation="linear"),   # unbounded output
            ])
            reg_model.compile(
                optimizer=optimizers.Adam(1e-3),
                loss="mse",                     # mean squared error
                metrics=["mae"]                 # mean absolute error (same units as y)
            )
            print("\nRegressor compiled:")
            print(f"  loss    : {reg_model.loss}")
            print(f"  metrics : {[m.name for m in reg_model.metrics]}")

            # ── Compile parameters reference ───────────────────────────────
            print("\n--- Compile quick-reference ---")
            print("optimizer shortcuts : 'adam', 'sgd', 'rmsprop', 'adagrad'")
            print("loss shortcuts      : 'mse', 'mae', 'binary_crossentropy',")
            print("                     'categorical_crossentropy',")
            print("                     'sparse_categorical_crossentropy'")
            print("metric shortcuts    : 'accuracy', 'mae', 'mse'")
            print("\nRule: output activation must match loss function:")
            print("  sigmoid + binary_crossentropy")
            print("  softmax + categorical_crossentropy  (one-hot y)")
            print("  softmax + sparse_categorical_crossentropy  (int y)")
            print("  linear  + mse or mae")
        ''',
    },

    # ── 10 ─────────────────────────────────────────────────────────────────
    "10 · Training on MNIST — Full Workflow": {
        "description": (
            "End-to-end training on the MNIST handwritten digit dataset: "
            "load and normalise data, build a model, train with fit(), "
            "and evaluate. MNIST is the 'hello world' of deep learning."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models
            import numpy as np

            # ── Load and preprocess MNIST ─────────────────────────────────
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            print(f"Training set : {X_train.shape}  labels: {y_train.shape}")
            print(f"Test set     : {X_test.shape}   labels: {y_test.shape}")
            print(f"Pixel range before normalisation: [{X_train.min()}, {X_train.max()}]")

            # Normalise pixel values to [0, 1]
            X_train = X_train.astype("float32") / 255.0
            X_test  = X_test.astype("float32")  / 255.0
            print(f"Pixel range after  normalisation: [{X_train.min():.1f}, {X_train.max():.1f}]")

            # Reserve 10,000 training samples for validation
            X_val, y_val   = X_train[-10000:], y_train[-10000:]
            X_train, y_train = X_train[:-10000], y_train[:-10000]
            print(f"\\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

            # ── Build model ───────────────────────────────────────────────
            model = models.Sequential([
                layers.Flatten(input_shape=(28, 28)),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(10, activation="softmax"),
            ], name="mnist_mlp")

            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            # ── Train ─────────────────────────────────────────────────────
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=128,
                validation_data=(X_val, y_val),
                verbose=1
            )

            # ── Evaluate on test set ──────────────────────────────────────
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"\\nTest loss     : {test_loss:.4f}")
            print(f"Test accuracy : {test_acc*100:.2f}%")

            # ── Training history summary ──────────────────────────────────
            final_train_acc = history.history["accuracy"][-1]
            final_val_acc   = history.history["val_accuracy"][-1]
            print(f"Final train accuracy : {final_train_acc*100:.2f}%")
            print(f"Final val accuracy   : {final_val_acc*100:.2f}%")
            gap = (final_train_acc - final_val_acc) * 100
            print(f"Train-val gap        : {gap:.2f}%  ", end="")
            print("(overfitting)" if gap > 3 else "(healthy)")
        ''',
    },

    # ── 11 ─────────────────────────────────────────────────────────────────
    "11 · Visualising Training — Learning Curves": {
        "description": (
            "Learning curves (loss and accuracy over epochs) are the primary "
            "diagnostic tool for understanding model training. They reveal "
            "underfitting, overfitting, and whether training has converged."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            # ── Train a model quickly on MNIST ────────────────────────────
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_train = X_train.astype("float32") / 255.0
            X_test  = X_test.astype("float32")  / 255.0
            X_val, y_val     = X_train[-5000:], y_train[-5000:]
            X_train, y_train = X_train[:-5000], y_train[:-5000]

            model = models.Sequential([
                layers.Flatten(input_shape=(28, 28)),
                layers.Dense(128, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ])
            model.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])

            history = model.fit(X_train, y_train, epochs=15, batch_size=256,
                                validation_data=(X_val, y_val), verbose=0)

            # ── Plot learning curves ───────────────────────────────────────
            h = history.history
            epochs = range(1, len(h["loss"]) + 1)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Loss subplot
            axes[0].plot(epochs, h["loss"],     "b-o", label="Train loss",  ms=4)
            axes[0].plot(epochs, h["val_loss"], "r-o", label="Val loss",    ms=4)
            axes[0].set_title("Loss over epochs"); axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

            # Accuracy subplot
            axes[1].plot(epochs, h["accuracy"],     "b-o", label="Train acc", ms=4)
            axes[1].plot(epochs, h["val_accuracy"], "r-o", label="Val acc",   ms=4)
            axes[1].set_title("Accuracy over epochs"); axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

            plt.suptitle("Learning Curves — Diagnosing Model Training",
                         fontweight="bold")
            plt.tight_layout()
            plt.savefig("learning_curves.png", dpi=100); plt.show()

            # ── Diagnose from numbers ─────────────────────────────────────
            print("\nDiagnostic guide:")
            best_epoch    = int(np.argmin(h["val_loss"])) + 1
            best_val_acc  = max(h["val_accuracy"])
            final_gap     = h["accuracy"][-1] - h["val_accuracy"][-1]
            print(f"  Best epoch (lowest val loss) : {best_epoch}")
            print(f"  Best val accuracy            : {best_val_acc*100:.2f}%")
            print(f"  Final train-val acc gap      : {final_gap*100:.2f}%")

            if final_gap > 0.05:
                print("  ⚠  Gap > 5% → likely overfitting. Try Dropout or more data.")
            elif h["accuracy"][-1] < 0.85:
                print("  ⚠  Low accuracy → likely underfitting. Try larger model or more epochs.")
            else:
                print("  ✓  Healthy training curves.")
            print("\nLearning curves saved to learning_curves.png")
        ''',
    },

    # ── 12 ─────────────────────────────────────────────────────────────────
    "12 · Callbacks — EarlyStopping, ModelCheckpoint, ReduceLROnPlateau": {
        "description": (
            "Callbacks intercept training at predefined points. EarlyStopping "
            "prevents overfitting; ModelCheckpoint saves the best weights; "
            "ReduceLROnPlateau automatically anneals the learning rate when "
            "the model plateaus — together they eliminate most manual tuning."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models, callbacks
            import tempfile, os

            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_train = X_train.astype("float32") / 255.0
            X_test  = X_test.astype("float32")  / 255.0
            X_val, y_val     = X_train[-5000:], y_train[-5000:]
            X_train, y_train = X_train[:-5000], y_train[:-5000]

            tmpdir = tempfile.mkdtemp()
            checkpoint_path = os.path.join(tmpdir, "best_model.keras")

            # ── Define callbacks ──────────────────────────────────────────
            early_stopping = callbacks.EarlyStopping(
                monitor="val_loss",       # watch validation loss
                patience=4,               # stop after 4 epochs of no improvement
                restore_best_weights=True # rollback to best weights on stop
            )

            model_checkpoint = callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,      # only overwrite if val_acc improved
                verbose=1
            )

            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,               # multiply lr by 0.5 when triggered
                patience=2,               # wait 2 epochs before reducing
                min_lr=1e-6,              # never go below this
                verbose=1
            )

            # ── Build and train ───────────────────────────────────────────
            model = models.Sequential([
                layers.Flatten(input_shape=(28, 28)),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ])
            model.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])

            print("Training with callbacks (EarlyStopping patience=4)...")
            history = model.fit(
                X_train, y_train,
                epochs=50,              # would train for 50 epochs without ES
                batch_size=256,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, model_checkpoint, reduce_lr],
                verbose=1
            )

            actual_epochs = len(history.history["loss"])
            print(f"\\nEarly stopping triggered at epoch {actual_epochs}/50")
            print(f"Saved best model to: {checkpoint_path}")

            # ── Load best saved model and evaluate ────────────────────────
            best_model = tf.keras.models.load_model(checkpoint_path)
            loss, acc  = best_model.evaluate(X_test, y_test, verbose=0)
            print(f"Best model test accuracy: {acc*100:.2f}%")

            # ── What each callback monitors ────────────────────────────────
            print("\n--- Callback reference ---")
            print("EarlyStopping       : stop training + restore weights")
            print("ModelCheckpoint     : save model after each epoch (best only)")
            print("ReduceLROnPlateau   : halve lr when val_loss plateaus")
            print("TensorBoard         : write logs for tensorboard dashboard")
            print("LearningRateScheduler: apply a custom lr schedule function")
        ''',
    },

    # ── 13 ─────────────────────────────────────────────────────────────────
    "13 · tf.data.Dataset — Efficient Input Pipelines": {
        "description": (
            "tf.data builds lazy, parallelised data pipelines that keep the GPU "
            "fed continuously. Operations like map(), shuffle(), batch(), and "
            "prefetch() are the building blocks of production data loading. "
            "prefetch() in particular eliminates the GPU idle time bottleneck."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            import numpy as np
            import time

            # ── Dataset from numpy arrays ─────────────────────────────────
            (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            X_train = X_train.astype("float32") / 255.0

            # from_tensor_slices pairs each X[i] with y[i]
            raw_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            print(f"Raw dataset element spec: {raw_dataset.element_spec}")
            print(f"Dataset length: {len(raw_dataset)}")

            # ── The full pipeline ─────────────────────────────────────────
            BATCH_SIZE   = 256
            AUTOTUNE     = tf.data.AUTOTUNE

            def augment(image, label):
                """Simple preprocessing / augmentation function."""
                image = tf.cast(image, tf.float32)
                # Random left-right flip (simulated — for demo)
                image = tf.image.random_flip_left_right(image[..., tf.newaxis])[..., 0]
                return image, label

            train_ds = (
                tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(buffer_size=10_000, seed=42)   # random order
                .map(augment, num_parallel_calls=AUTOTUNE) # parallel preprocessing
                .batch(BATCH_SIZE)                         # group into batches
                .prefetch(AUTOTUNE)                        # pre-load next batch
            )

            print(f"\\nBatched dataset element spec:")
            print(f"  {train_ds.element_spec}")

            # ── Iterate over a few batches ────────────────────────────────
            for i, (images, labels) in enumerate(train_ds.take(3)):
                print(f"Batch {i}: images={images.shape}  labels={labels.shape}  "
                      f"dtype={images.dtype}")

            # ── Why prefetch matters: timing comparison ────────────────────
            def time_pipeline(dataset, name, n_batches=50):
                t0 = time.time()
                for _ in dataset.take(n_batches):
                    pass   # simulate training step
                return time.time() - t0

            no_prefetch = (
                tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .batch(BATCH_SIZE)
            )
            with_prefetch = (
                tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE)
            )

            t_no  = time_pipeline(no_prefetch,  "No prefetch")
            t_yes = time_pipeline(with_prefetch, "With prefetch")
            print(f"\\n50-batch iteration:")
            print(f"  Without prefetch : {t_no:.3f}s")
            print(f"  With prefetch    : {t_yes:.3f}s")
            speedup = t_no / max(t_yes, 1e-6)
            print(f"  Speedup          : {speedup:.1f}x")
            print("\nprefetch() overlaps data loading with GPU computation.")
            print("On real image data, the speedup is dramatic.")
        ''',
    },

    # ── 14 ─────────────────────────────────────────────────────────────────
    "14 · Regularisation — Dropout and L1/L2 Weight Decay": {
        "description": (
            "Regularisation techniques constrain a model to prevent it memorising "
            "training data. Dropout randomly silences neurons each step. L2 weight "
            "decay adds a penalty for large weights. Both are demonstarted "
            "with before/after training curves showing overfitting being tamed."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models, regularizers
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # ── Create a small, noisy dataset → easy to overfit ────────────
            np.random.seed(42)
            N = 500
            X = np.random.randn(N, 20).astype("float32")
            y = (X[:, 0] + X[:, 1] > 0).astype("float32")  # simple decision boundary
            X_test = np.random.randn(200, 20).astype("float32")
            y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype("float32")

            def build_and_train(name, use_dropout=False, l2_lambda=0.0, epochs=50):
                reg = regularizers.l2(l2_lambda) if l2_lambda > 0 else None
                model = models.Sequential([
                    layers.Dense(512, activation="relu", input_shape=(20,),
                                 kernel_regularizer=reg),
                    layers.Dropout(0.5) if use_dropout else layers.Lambda(lambda x: x),
                    layers.Dense(512, activation="relu", kernel_regularizer=reg),
                    layers.Dropout(0.5) if use_dropout else layers.Lambda(lambda x: x),
                    layers.Dense(1, activation="sigmoid"),
                ])
                model.compile(optimizer="adam", loss="binary_crossentropy",
                              metrics=["accuracy"])
                h = model.fit(X, y, epochs=epochs, batch_size=32,
                              validation_split=0.2, verbose=0)
                test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
                return h.history, test_acc

            print("Training 3 models (no reg, dropout, L2)...")
            h_none,    acc_none    = build_and_train("No regularisation")
            h_dropout, acc_dropout = build_and_train("Dropout",  use_dropout=True)
            h_l2,      acc_l2     = build_and_train("L2",       l2_lambda=0.01)

            print(f"No regularisation — test acc: {acc_none*100:.1f}%")
            print(f"Dropout           — test acc: {acc_dropout*100:.1f}%")
            print(f"L2 (λ=0.01)      — test acc: {acc_l2*100:.1f}%")

            # ── Plot train vs val accuracy gap ─────────────────────────────
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
            configs = [("No Regularisation", h_none), ("Dropout 0.5", h_dropout),
                       ("L2 (λ=0.01)",      h_l2)]
            for ax, (name, h) in zip(axes, configs):
                ep = range(1, len(h["accuracy"]) + 1)
                ax.plot(ep, h["accuracy"],     "b-", label="Train", lw=1.5)
                ax.plot(ep, h["val_accuracy"], "r-", label="Val",   lw=1.5)
                gap = (h["accuracy"][-1] - h["val_accuracy"][-1]) * 100
                ax.set_title(f"{name}\\nFinal gap: {gap:.1f}%")
                ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
            axes[0].set_ylabel("Accuracy")
            plt.suptitle("Regularisation: Closing the Train-Val Gap", fontweight="bold")
            plt.tight_layout()
            plt.savefig("regularisation.png", dpi=100); plt.show()
            print("\nPlot saved to regularisation.png")
        ''',
    },

    # ── 15 ─────────────────────────────────────────────────────────────────
    "15 · Batch Normalisation — Stabilising Deep Networks": {
        "description": (
            "Batch Normalisation normalises activations within each mini-batch, "
            "reducing internal covariate shift. It stabilises training, allows "
            "higher learning rates, and acts as a mild regulariser — making "
            "it near-universal in modern deep network architectures."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_train = X_train.astype("float32") / 255.0
            X_test  = X_test.astype("float32")  / 255.0
            X_val, y_val     = X_train[-5000:], y_train[-5000:]
            X_train, y_train = X_train[:-5000], y_train[:-5000]

            def build_model(use_batchnorm=False, name=""):
                model = models.Sequential(name=name)
                model.add(layers.Flatten(input_shape=(28, 28)))
                for units in [512, 256, 128]:
                    model.add(layers.Dense(units))
                    if use_batchnorm:
                        # BN before activation is the common convention
                        model.add(layers.BatchNormalization())
                    model.add(layers.Activation("relu"))
                model.add(layers.Dense(10, activation="softmax"))
                return model

            model_no_bn = build_model(use_batchnorm=False, name="no_batchnorm")
            model_bn    = build_model(use_batchnorm=True,  name="with_batchnorm")

            for model in [model_no_bn, model_bn]:
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                              loss="sparse_categorical_crossentropy",
                              metrics=["accuracy"])

            print("Training WITHOUT BatchNorm (lr=0.01)...")
            h_no = model_no_bn.fit(X_train, y_train, epochs=10, batch_size=256,
                                    validation_data=(X_val, y_val), verbose=0)

            print("Training WITH    BatchNorm (lr=0.01)...")
            h_bn = model_bn.fit(X_train, y_train, epochs=10, batch_size=256,
                                validation_data=(X_val, y_val), verbose=0)

            # ── Compare ────────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            ep = range(1, 11)
            for ax, metric, ylabel in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
                ax.plot(ep, h_no.history[metric],         "b--", label="Train no BN")
                ax.plot(ep, h_no.history[f"val_{metric}"],"b-",  label="Val no BN")
                ax.plot(ep, h_bn.history[metric],         "r--", label="Train BN")
                ax.plot(ep, h_bn.history[f"val_{metric}"],"r-",  label="Val BN")
                ax.set_title(f"{ylabel} (lr=0.01)"); ax.set_xlabel("Epoch")
                ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(alpha=0.3)
            plt.suptitle("Batch Normalisation Effect (high lr=0.01)", fontweight="bold")
            plt.tight_layout()
            plt.savefig("batchnorm.png", dpi=100); plt.show()

            final_no = h_no.history["val_accuracy"][-1] * 100
            final_bn = h_bn.history["val_accuracy"][-1] * 100
            print(f"\nFinal val accuracy — No BN: {final_no:.2f}%  | With BN: {final_bn:.2f}%")
            print("BatchNorm allows higher learning rates → faster, stabler training.")
            print("Placement: Dense/Conv → BatchNorm → Activation")
        ''',
    },

    # ── 16 ─────────────────────────────────────────────────────────────────
    "16 · Convolutional Neural Network (CNN) for Image Classification": {
        "description": (
            "CNNs are the standard architecture for images. Conv2D layers apply "
            "learned filters to detect local patterns. MaxPooling downsamples "
            "spatially. This operation builds and trains a CNN on CIFAR-10 "
            "(60k colour images, 10 classes) and visualises the learned filters."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # ── CIFAR-10: 32×32 colour images, 10 classes ─────────────────
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
            class_names = ["airplane","automobile","bird","cat","deer",
                           "dog","frog","horse","ship","truck"]
            print(f"Train: {X_train.shape}  Test: {X_test.shape}")

            # Normalise to [0, 1]
            X_train = X_train.astype("float32") / 255.0
            X_test  = X_test.astype("float32")  / 255.0
            y_train = y_train.flatten()   # squeeze from (50000,1) to (50000,)
            y_test  = y_test.flatten()

            # ── CNN Architecture ──────────────────────────────────────────
            # Conv block: Conv2D → BatchNorm → ReLU → MaxPool
            model = models.Sequential([
                # Block 1: detect low-level features (edges, colours)
                layers.Conv2D(32, (3,3), padding="same", activation="relu",
                              input_shape=(32, 32, 3)),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3,3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),     # 32×32 → 16×16
                layers.Dropout(0.25),

                # Block 2: detect higher-level features
                layers.Conv2D(64, (3,3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3,3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),     # 16×16 → 8×8
                layers.Dropout(0.25),

                # Classification head
                layers.Flatten(),               # 8×8×64 = 4096 → 4096
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(10, activation="softmax"),
            ], name="cifar10_cnn")

            model.summary()

            model.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])

            # ── Train for a few epochs (full training would need ~50+ epochs) ─
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=128,
                validation_split=0.1,
                verbose=1
            )

            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"\nTest accuracy after 10 epochs: {test_acc*100:.2f}%")
            print("(Full training with data augmentation reaches ~90%+)")

            # ── Visualise a few predictions ───────────────────────────────
            preds = model.predict(X_test[:9], verbose=0)
            fig, axes = plt.subplots(3, 3, figsize=(7, 7))
            for i, ax in enumerate(axes.flat):
                ax.imshow(X_test[i])
                pred = class_names[np.argmax(preds[i])]
                true = class_names[y_test[i]]
                color = "green" if pred == true else "red"
                ax.set_title(f"P:{pred}\\nT:{true}", fontsize=8, color=color)
                ax.axis("off")
            plt.suptitle("CNN Predictions (green=correct, red=wrong)", fontweight="bold")
            plt.tight_layout()
            plt.savefig("cnn_predictions.png", dpi=100); plt.show()
            print("Predictions saved to cnn_predictions.png")
        ''',
    },

    # ── 17 ─────────────────────────────────────────────────────────────────
    "17 · Transfer Learning — Feature Extraction with MobileNetV2": {
        "description": (
            "Transfer learning reuses a model trained on ImageNet and adapts it "
            "to a new task. Feature extraction freezes the pretrained backbone "
            "and trains only a new classification head — achieving strong results "
            "with very few new training examples."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models
            from tensorflow.keras.applications import MobileNetV2
            import numpy as np

            # ── Use CIFAR-10 as our "new task" ────────────────────────────
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
            X_train = X_train.astype("float32") / 255.0
            X_test  = X_test.astype("float32")  / 255.0
            y_train, y_test = y_train.flatten(), y_test.flatten()

            # Use only 5000 training samples to simulate data-scarce scenario
            X_small, y_small = X_train[:5000], y_train[:5000]

            # ── CIFAR-10 images are 32×32 but MobileNetV2 expects 96×96 min ─
            # Resize on the fly using a Lambda layer
            resize_layer = layers.Resizing(96, 96)

            # ── Load MobileNetV2 pretrained on ImageNet ────────────────────
            base_model = MobileNetV2(
                input_shape=(96, 96, 3),
                include_top=False,          # remove ImageNet classifier head
                weights="imagenet"          # pretrained weights
            )
            base_model.trainable = False    # FREEZE: do not update during training
            print(f"Base model layers    : {len(base_model.layers)}")
            print(f"Base trainable params: {base_model.count_params():,}")

            # ── Add a new classification head ──────────────────────────────
            inputs  = layers.Input(shape=(32, 32, 3))
            x       = resize_layer(inputs)                         # 32→96
            x       = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255)
            x       = base_model(x, training=False)                # frozen
            x       = layers.GlobalAveragePooling2D()(x)           # (b,3,3,1280)→(b,1280)
            x       = layers.Dense(256, activation="relu")(x)
            x       = layers.Dropout(0.3)(x)
            outputs = layers.Dense(10, activation="softmax")(x)

            model = models.Model(inputs, outputs, name="transfer_learning")

            # Only the new head is trainable
            trainable = sum(v.numpy().size for v in model.trainable_variables)
            total     = model.count_params()
            print(f"Total params         : {total:,}")
            print(f"Trainable params     : {trainable:,}  ({trainable/total*100:.1f}%)")

            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])

            print("\nTraining new head only (feature extraction)...")
            history = model.fit(X_small, y_small,
                                epochs=5, batch_size=64,
                                validation_split=0.2,
                                verbose=1)

            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"\nTest accuracy (feature extraction, 5000 samples): {test_acc*100:.2f}%")

            # ── Fine-tuning: unfreeze top layers ──────────────────────────
            print("\n--- Fine-tuning top 20 layers of base model ---")
            for layer in base_model.layers[-20:]:
                layer.trainable = True

            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # much smaller lr!
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])

            fine_history = model.fit(X_small, y_small,
                                     epochs=3, batch_size=64,
                                     validation_split=0.2,
                                     verbose=1)

            test_loss2, test_acc2 = model.evaluate(X_test, y_test, verbose=0)
            print(f"Test accuracy (after fine-tuning)               : {test_acc2*100:.2f}%")
            print(f"Improvement from fine-tuning: +{(test_acc2-test_acc)*100:.2f}%")
            print("\nKey rules for fine-tuning:")
            print("  1. Always train head first with frozen base")
            print("  2. Unfreeze top layers only (preserve universal low-level features)")
            print("  3. Use a very small learning rate (e.g. 1e-5) to avoid catastrophic forgetting")
        ''',
    },

    # ── 18 ─────────────────────────────────────────────────────────────────
    "18 · Making and Evaluating Predictions": {
        "description": (
            "After training, the model is used to make predictions. This covers "
            "predict(), predict_proba (via softmax), converting logit outputs to "
            "labels, confidence scores, and a full classification report with "
            "a visualised confusion matrix."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # ── Train a quick model ────────────────────────────────────────
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_train = X_train.astype("float32") / 255.0
            X_test  = X_test.astype("float32")  / 255.0

            model = models.Sequential([
                layers.Flatten(input_shape=(28, 28)),
                layers.Dense(128, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ])
            model.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            model.fit(X_train, y_train, epochs=5, batch_size=256,
                      validation_split=0.1, verbose=0)

            # ── Making predictions ─────────────────────────────────────────
            # predict() returns softmax probabilities (10 values per sample)
            y_probs = model.predict(X_test, verbose=0)
            print(f"y_probs shape         : {y_probs.shape}")
            print(f"y_probs[0] (probs)    : {y_probs[0].round(3)}")
            print(f"Sum of probs (must=1) : {y_probs[0].sum():.6f}")

            # Convert probabilities → hard class labels
            y_preds = np.argmax(y_probs, axis=1)
            confs   = np.max(y_probs, axis=1)
            print(f"\nFirst 10 predicted labels : {y_preds[:10]}")
            print(f"First 10 true labels      : {y_test[:10]}")
            print(f"First 10 confidences      : {confs[:10].round(3)}")

            # ── Overall accuracy ───────────────────────────────────────────
            accuracy = np.mean(y_preds == y_test)
            print(f"\nOverall test accuracy : {accuracy*100:.2f}%")

            # ── Per-class accuracy ─────────────────────────────────────────
            print("\nPer-class accuracy:")
            for cls in range(10):
                mask      = y_test == cls
                cls_acc   = np.mean(y_preds[mask] == y_test[mask])
                cls_conf  = confs[mask].mean()
                print(f"  Digit {cls}: acc={cls_acc*100:.1f}%  avg_conf={cls_conf:.3f}")

            # ── Confusion matrix ───────────────────────────────────────────
            conf_mat = tf.math.confusion_matrix(y_test, y_preds).numpy()

            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(conf_mat, cmap="Blues")
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(10)); ax.set_yticks(range(10))
            ax.set_xticklabels(range(10)); ax.set_yticklabels(range(10))
            ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
            ax.set_title("Confusion Matrix — MNIST Test Set")
            # Add numbers to cells
            for i in range(10):
                for j in range(10):
                    ax.text(j, i, conf_mat[i,j], ha="center", va="center",
                            fontsize=7, color="white" if conf_mat[i,j] > 500 else "black")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=100); plt.show()
            print("\nConfusion matrix saved to confusion_matrix.png")

            # ── Visualise errors ───────────────────────────────────────────
            errors    = np.where(y_preds != y_test)[0]
            print(f"\nTotal errors: {len(errors)} out of {len(y_test)}")
            fig2, axes = plt.subplots(2, 5, figsize=(12, 5))
            for i, ax in enumerate(axes.flat):
                idx = errors[i]
                ax.imshow(X_test[idx], cmap="gray")
                ax.set_title(f"True:{y_test[idx]} Pred:{y_preds[idx]}\nConf:{confs[idx]:.2f}",
                             fontsize=8, color="red")
                ax.axis("off")
            plt.suptitle("Misclassified Examples", fontweight="bold")
            plt.tight_layout()
            plt.savefig("misclassified.png", dpi=100); plt.show()
            print("Misclassified examples saved to misclassified.png")
        ''',
    },

    # ── 19 ─────────────────────────────────────────────────────────────────
    "19 · Saving and Loading Models": {
        "description": (
            "Trained models must be saved to avoid retraining. TensorFlow supports "
            "SavedModel (recommended, framework-agnostic) and HDF5 (.h5) formats. "
            "Covers full model save/load, weights-only save/load, and the "
            "ModelCheckpoint callback for saving during training."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models, callbacks
            import numpy as np
            import tempfile, os

            # ── Train a small model ────────────────────────────────────────
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_train = X_train.astype("float32") / 255.0
            X_test  = X_test.astype("float32")  / 255.0

            def build_model():
                m = models.Sequential([
                    layers.Flatten(input_shape=(28, 28)),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(10, activation="softmax"),
                ])
                m.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
                return m

            model = build_model()
            model.fit(X_train, y_train, epochs=3, batch_size=256,
                      validation_split=0.1, verbose=0)

            original_acc = model.evaluate(X_test, y_test, verbose=0)[1]
            print(f"Original model test accuracy: {original_acc*100:.2f}%")

            tmpdir = tempfile.mkdtemp()

            # ── Method 1: SavedModel format (recommended) ──────────────────
            saved_path = os.path.join(tmpdir, "my_model")
            model.save(saved_path)
            contents = os.listdir(saved_path)
            print(f"\n[SavedModel] Saved to: {saved_path}")
            print(f"  Contents: {contents}")

            loaded_saved = tf.keras.models.load_model(saved_path)
            acc_saved = loaded_saved.evaluate(X_test, y_test, verbose=0)[1]
            print(f"  Loaded model accuracy: {acc_saved*100:.2f}%  "
                  f"(match: {acc_saved == original_acc})")

            # ── Method 2: HDF5 format ─────────────────────────────────────
            h5_path = os.path.join(tmpdir, "my_model.h5")
            model.save(h5_path)
            size_kb = os.path.getsize(h5_path) / 1024
            print(f"\n[HDF5] Saved to: {h5_path}  ({size_kb:.1f} KB)")

            loaded_h5 = tf.keras.models.load_model(h5_path)
            acc_h5 = loaded_h5.evaluate(X_test, y_test, verbose=0)[1]
            print(f"  Loaded model accuracy: {acc_h5*100:.2f}%")

            # ── Method 3: Weights only ────────────────────────────────────
            weights_path = os.path.join(tmpdir, "weights.h5")
            model.save_weights(weights_path)

            # Must re-create the architecture before loading weights
            fresh_model = build_model()
            fresh_model.load_weights(weights_path)
            acc_weights = fresh_model.evaluate(X_test, y_test, verbose=0)[1]
            print(f"\n[Weights only] Loaded into fresh model accuracy: {acc_weights*100:.2f}%")

            # ── Method 4: ModelCheckpoint callback (during training) ───────
            best_path = os.path.join(tmpdir, "best.keras")
            checkpoint_cb = callbacks.ModelCheckpoint(
                filepath=best_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )
            new_model = build_model()
            new_model.fit(X_train, y_train, epochs=5, batch_size=256,
                          validation_split=0.1, callbacks=[checkpoint_cb], verbose=0)
            best_loaded = tf.keras.models.load_model(best_path)
            best_acc = best_loaded.evaluate(X_test, y_test, verbose=0)[1]
            print(f"\n[Checkpoint] Best model test accuracy: {best_acc*100:.2f}%")

            # ── Summary ────────────────────────────────────────────────────
            print("\n--- Save format summary ---")
            print("SavedModel  : default, serves on TF Serving, TFLite, TFjs")
            print("HDF5 (.h5)  : single file, widely portable, slightly slower")
            print("Weights only: smallest, requires architecture to be defined first")
            print("Checkpoint  : automatic best-model saving during long training runs")
        ''',
    },

    # ── 20 ─────────────────────────────────────────────────────────────────
    "20 · Custom Training Loop with GradientTape": {
        "description": (
            "model.fit() is convenient but sometimes you need full control: "
            "custom loss calculations, gradient clipping, multiple optimisers, "
            "or printing internals per step. A custom training loop with "
            "GradientTape gives complete flexibility — it's also how PyTorch "
            "works, making this pattern transferable knowledge."
        ),
        "language": "python",
        "code": r'''
            import tensorflow as tf
            from tensorflow.keras import layers, models
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # ── Dataset ────────────────────────────────────────────────────
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_train = X_train.astype("float32").reshape(-1, 784) / 255.0
            X_test  = X_test.astype("float32").reshape(-1, 784)  / 255.0

            # One-hot encode labels
            y_train_oh = tf.one_hot(y_train, 10)
            y_test_oh  = tf.one_hot(y_test,  10)

            # tf.data pipeline
            BATCH_SIZE = 256
            train_ds = (
                tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
                .shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            )

            # ── Model ──────────────────────────────────────────────────────
            model = models.Sequential([
                layers.Dense(256, activation="relu", input_shape=(784,)),
                layers.Dropout(0.2),
                layers.Dense(128, activation="relu"),
                layers.Dense(10),      # raw logits — no softmax (handled in loss)
            ])

            # ── Loss and optimiser ─────────────────────────────────────────
            loss_fn   = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

            # ── Metrics (stateful — accumulate over batches) ───────────────
            train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
            train_acc_metric  = tf.keras.metrics.CategoricalAccuracy(name="train_acc")
            val_loss_metric   = tf.keras.metrics.Mean(name="val_loss")
            val_acc_metric    = tf.keras.metrics.CategoricalAccuracy(name="val_acc")

            # ── @tf.function compiles the step to a graph (10× faster) ─────
            @tf.function
            def train_step(x_batch, y_batch):
                with tf.GradientTape() as tape:
                    logits = model(x_batch, training=True)      # forward pass
                    loss   = loss_fn(y_batch, logits)
                    # L2 regularisation (manual)
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                                        if "kernel" in v.name]) * 1e-4
                    total_loss = loss + l2_loss

                grads = tape.gradient(total_loss, model.trainable_variables)

                # Gradient clipping: prevent exploding gradients
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                train_loss_metric.update_state(loss)
                train_acc_metric.update_state(y_batch, logits)

            @tf.function
            def val_step(x_batch, y_batch):
                logits = model(x_batch, training=False)         # no dropout
                loss   = loss_fn(y_batch, logits)
                val_loss_metric.update_state(loss)
                val_acc_metric.update_state(y_batch, logits)

            # ── Custom training loop ───────────────────────────────────────
            EPOCHS = 8
            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

            val_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test_oh))
                      .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

            for epoch in range(1, EPOCHS + 1):
                # Training
                for x_batch, y_batch in train_ds:
                    train_step(x_batch, y_batch)

                # Validation
                for x_batch, y_batch in val_ds:
                    val_step(x_batch, y_batch)

                # Collect metrics
                tr_l = train_loss_metric.result().numpy()
                tr_a = train_acc_metric.result().numpy()
                vl_l = val_loss_metric.result().numpy()
                vl_a = val_acc_metric.result().numpy()
                history["train_loss"].append(tr_l)
                history["train_acc"].append(tr_a)
                history["val_loss"].append(vl_l)
                history["val_acc"].append(vl_a)
                print(f"Epoch {epoch:2d}/{EPOCHS} | "
                      f"loss={tr_l:.4f}  acc={tr_a*100:.2f}% | "
                      f"val_loss={vl_l:.4f}  val_acc={vl_a*100:.2f}%")

                # Reset metrics after each epoch
                for m in [train_loss_metric, train_acc_metric,
                          val_loss_metric, val_acc_metric]:
                    m.reset_state()

            # ── Plot ───────────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            ep = range(1, EPOCHS + 1)
            axes[0].plot(ep, history["train_loss"], "b-o", ms=4, label="Train")
            axes[0].plot(ep, history["val_loss"],   "r-o", ms=4, label="Val")
            axes[0].set_title("Custom Loop — Loss"); axes[0].legend()
            axes[1].plot(ep, [a*100 for a in history["train_acc"]], "b-o", ms=4, label="Train")
            axes[1].plot(ep, [a*100 for a in history["val_acc"]],   "r-o", ms=4, label="Val")
            axes[1].set_title("Custom Loop — Accuracy (%)"); axes[1].legend()
            for ax in axes: ax.grid(alpha=0.3); ax.set_xlabel("Epoch")
            plt.suptitle("Custom Training Loop with GradientTape", fontweight="bold")
            plt.tight_layout()
            plt.savefig("custom_loop.png", dpi=100); plt.show()

            final_acc = history["val_acc"][-1]
            print(f"\nFinal val accuracy : {final_acc*100:.2f}%")
            print("Custom loop features demonstrated:")
            print("  • GradientTape for manual backprop")
            print("  • Manual L2 regularisation added to loss")
            print("  • Gradient clipping via clip_by_global_norm")
            print("  • @tf.function for graph compilation (10× speedup)")
            print("  • Stateful keras.metrics for clean epoch aggregation")
        ''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings
# ─────────────────────────────────────────────────────────────────────────────
def _dedent_code(code):
    """
    Dedent a code block whose lines may include zero-indent continuation lines
    (caused by literal \n inside string constants in the source file).
    Uses the FIRST non-empty line's indent as the strip width, then removes
    exactly that many leading spaces from every line that starts with them.
    """
    lines = code.split("\n")
    first_indent = 0
    for l in lines:
        if l.strip():
            first_indent = len(l) - len(l.lstrip())
            break
    if first_indent == 0:
        return code.strip()
    result = []
    for l in lines:
        if l.startswith(" " * first_indent):
            result.append(l[first_indent:])
        else:
            result.append(l)
    return "\n".join(result).strip()

for _op in OPERATIONS.values():
    _op["code"] = _dedent_code(_op["code"])


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
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 600,
        "complexity":    None,
        "operations":    OPERATIONS,
    }