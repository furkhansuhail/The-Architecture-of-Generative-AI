"""
Keras — The High-Level Deep Learning API
=========================================

Keras is a deep learning API created by François Chollet at Google in 2015
and now maintained as a standalone multi-backend library. Where PyTorch gives
you maximum control and TensorFlow gives you production infrastructure, Keras
gives you the fastest path from idea to working model.

Its design philosophy is captured in a single sentence from Chollet: "Being
able to go from idea to result with the least possible delay is key to doing
good research." Keras achieves this through consistent abstraction: no matter
how complex your architecture, the same patterns — layer stacking, model
compilation, fit loops, callbacks — always apply.

What makes Keras architecturally distinctive is its three-tier abstraction
system. Sequential handles linear pipelines in three lines. Functional API
handles any directed acyclic graph — multi-input, multi-output, skip
connections, shared weights — without losing the simplicity guarantee. Model
subclassing gives you full Python control identical to raw PyTorch, for the
rare cases that truly need it. Each tier is additive: you never have to
rewrite code when upgrading from Sequential to Functional.

Keras 3 (released 2023) extended this philosophy to a multi-backend world:
the same Keras code runs transparently on TensorFlow, JAX, or PyTorch as
backends. This is the first time in the framework's history that you can
write your model once and benchmark it across all three execution engines.

This module covers Keras's full architecture: all three model-building APIs,
the Layer class and its build/call lifecycle, the compile/fit training system,
the preprocessing layers pipeline, the full callback system, custom training
loops via train_step override, mixed precision, and the complete deployment
stack from SavedModel to TFLite to TF Serving.

"""

import textwrap
import re

TOPIC_NAME   = "Keras — The High-Level Deep Learning API"
DISPLAY_NAME = "07 · Keras"
ICON         = "🧠"
SUBTITLE     = "From Sequential Prototypes to Multi-Backend Production Models"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — KERAS HISTORY, PHILOSOPHY AND THE MULTI-BACKEND ERA

### The Origins

    Keras 1.x (2015–2017): standalone library that ran on top of Theano or
        TensorFlow as optional backends. Chollet's insight: the framework that
        wins is the one that collapses the distance between idea and result.

    Keras 2.x (2017–2023): became TensorFlow's official high-level API,
        absorbed into tf.keras. The tradeoff — deep TF integration — made it
        the default entry point for most TensorFlow users worldwide. Keras
        became the first ML framework taught in most university courses.

    Keras 3 (2023–present): Chollet and the team separated Keras back into
        a standalone library with pluggable backends. The three supported
        backends are TensorFlow, JAX, and PyTorch. The same model code runs
        on all three without modification.

### Why a Multi-Backend API Makes Sense

    The three backends have complementary strengths:

        TensorFlow:  production infrastructure — TF Serving, TFLite, TFX,
                     TPU first-class support, TensorFlow.js, enterprise tools.
        JAX:         fastest research iteration — functional transformations,
                     vmap (vectorised batching), pmap (multi-device),
                     grad (clean functional autodiff), JIT via XLA.
        PyTorch:     research ecosystem — Hugging Face, torchvision, torchaudio,
                     the widest selection of third-party models and datasets.

    With Keras 3, you write the model once and choose the backend at runtime:
        import os
        os.environ['KERAS_BACKEND'] = 'jax'   # or 'tensorflow', 'torch'
        import keras

    This matters for benchmarking: the same architecture may run 40% faster
    on JAX+XLA than on TF for certain workloads. You can now measure this
    without rewriting your model.

### The Keras Design Principles

    1. USER EXPERIENCE FIRST:
        Every API decision prioritises the developer experience.
        The learning curve is gentle and consistent.

    2. PROGRESSIVE DISCLOSURE:
        Simple things are simple (Sequential).
        Complex things are possible (subclassing).
        You never hit a wall where you must abandon the framework.

    3. SANE DEFAULTS:
        Xavier initialisation for weights. Zeros for biases.
        Adam optimiser with lr=0.001 is a reasonable default.
        The defaults are chosen to work for the vast majority of cases.

    4. COMPOSABILITY:
        Layers compose into models. Models are layers.
        A trained model can be inserted as a layer into another model.
        There is no special "submodel" type — the abstraction is uniform.

    5. STATEFUL METRICS:
        Metrics accumulate state across batches correctly.
        No naive batch-averaging bugs. Keras handles this automatically.

### Keras vs TF/Keras vs tf.keras

    keras (pip install keras):     standalone multi-backend Keras 3.
    tf.keras:                      TensorFlow's bundled Keras 2 (not Keras 3).
    tensorflow.keras:              same as tf.keras.

    For new projects: import keras (standalone) and set KERAS_BACKEND.
    For legacy TF projects: tf.keras works but is Keras 2 (fewer features).

    This module uses standalone keras (Keras 3). All code also works with
    tf.keras / tensorflow.keras on the TensorFlow backend.


##### PART 2 — THE THREE MODEL-BUILDING APIS

### API 1: Sequential — Linear Stacks

    Sequential is a special case of Model where layers form a single
    ordered chain: one input → one output, no branching.

        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10),
        ])

    When to use Sequential:
        - Linear pipeline: each layer feeds exactly into the next
        - Quick prototyping and experimentation
        - Teaching and demonstration purposes

    When NOT to use Sequential:
        - Your model has multiple inputs or multiple outputs
        - Any layer has more than one input (e.g. Add, Concatenate)
        - You need skip connections / residual connections
        - You want to reuse layers across different parts of the graph

    Sequential with input() for explicit shape propagation:
        model = keras.Sequential()
        model.add(keras.Input(shape=(784,)))      # explicit input layer
        model.add(keras.layers.Dense(128, activation='relu'))
        # Now model.output_shape is known before any data is seen

    model.summary() before fitting — always check:
        model.summary()   # prints layer names, output shapes, param counts
        # Total params, trainable params, non-trainable params

### API 2: Functional API — Directed Acyclic Graphs

    The Functional API treats layers as callable objects on symbolic tensors.
    Any DAG of layers can be expressed this way.

    The pattern:
        inputs  = keras.Input(shape=(32,), name='features')
        x       = keras.layers.Dense(64, activation='relu')(inputs)
        skip    = x                                 # save for residual
        x       = keras.layers.Dense(64, activation='relu')(x)
        x       = keras.layers.Add()([x, skip])     # skip connection
        x       = keras.layers.LayerNormalization()(x)
        outputs = keras.layers.Dense(10, name='logits')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='resnet_mlp')

    Why the Functional API is the practical default for production:
        1. The model is a STATIC DAG — shape inference is complete at build time.
           Call model.summary() and keras.utils.plot_model() to visualise it.
        2. Input/output shapes are validated automatically.
        3. You get intermediate activations for free:
               layer_output = model.get_layer('dense').output
               extractor    = keras.Model(inputs, layer_output)
        4. Multiple inputs and outputs are natural:
               inputs  = [keras.Input((32,)), keras.Input((8,))]
               merged  = keras.layers.Concatenate()(inputs)
               outputs = keras.layers.Dense(10)(merged)
               model   = keras.Model(inputs, outputs)

    The key rule: each keras.Input() creates a symbolic tensor. Calling
    layers on these symbols traces the graph without executing any computation.
    The actual forward pass happens only when you call model(data).

### API 3: Model Subclassing — Full Python Control

    For models that genuinely require dynamic behaviour: recurrent loops
    that vary per sample, mixture of experts routing, models that build
    subgraphs conditionally based on Python logic.

        class TransformerEncoder(keras.Model):
            def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
                super().__init__()
                self.layers_list = [
                    TransformerBlock(d_model, n_heads, d_ff, dropout)
                    for _ in range(n_layers)
                ]
                self.norm = keras.layers.LayerNormalization()

            def call(self, x, training=False, mask=None):
                for layer in self.layers_list:
                    x = layer(x, training=training, mask=mask)
                return self.norm(x)

    Rules for subclassing:
        - Define all sublayers in __init__
        - Define the forward computation in call()
        - Pass training=False to call() — Keras routes it to Dropout/BN
        - Do NOT call self.build() — Keras calls it automatically on first use

    Subclassing trade-offs vs Functional:
        GAINS:   full Python control, dynamic shapes, data-dependent logic
        LOSSES:  no automatic shape inference before the first call,
                 model.summary() shows less detail,
                 keras.utils.plot_model() cannot visualise the graph,
                 harder to extract intermediate activations

### Choosing the Right API

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Scenario                               │ Use API                     │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Linear stack, quick prototype          │ Sequential                  │
    │ Multiple inputs/outputs, skip conn.    │ Functional                  │
    │ Residual networks, attention models    │ Functional                  │
    │ Shared layers (Siamese networks)       │ Functional                  │
    │ Dynamic control flow per sample        │ Subclassing                 │
    │ Mixture of experts, tree-RNNs          │ Subclassing                 │
    │ All other production models            │ Functional (default)        │
    └──────────────────────────────────────────────────────────────────────┘

    When in doubt: use the Functional API.
    It gives 90% of the flexibility of subclassing with none of the
    shape-inference drawbacks.


##### PART 3 — LAYERS: THE FUNDAMENTAL BUILDING BLOCKS

### The Layer Lifecycle

    A Keras Layer is the atomic unit of computation. Every component —
    Dense, Conv2D, BatchNormalization, your custom layer — follows the
    same lifecycle:

    1. __init__(self, ...):
        Define hyperparameters and sub-layer objects.
        Do NOT create weights here (input shape is unknown).

    2. build(self, input_shape):
        Called ONCE automatically on first call, with the actual input shape.
        Create weight tensors here using self.add_weight().
        This is where input-shape-dependent initialisation happens.

    3. call(self, inputs, training=False):
        The forward pass. Run on every call.
        Use training flag to switch Dropout/BN behaviour.
        Return the output tensor.

    4. get_config(self):
        Return a dict of constructor arguments for serialisation.
        Required for keras.models.model_from_config() to work.

    Example custom layer:

        class ScaledDotProductLayer(keras.Layer):
            def __init__(self, units, use_bias=True, **kwargs):
                super().__init__(**kwargs)
                self.units    = units
                self.use_bias = use_bias

            def build(self, input_shape):
                # input_shape[-1] = fan_in — known only at build time
                self.W = self.add_weight(
                    name        = 'kernel',
                    shape       = (input_shape[-1], self.units),
                    initializer = 'glorot_uniform',
                    trainable   = True,
                )
                if self.use_bias:
                    self.b = self.add_weight(
                        name        = 'bias',
                        shape       = (self.units,),
                        initializer = 'zeros',
                        trainable   = True,
                    )
                self.scale = self.add_weight(
                    name        = 'scale',
                    shape       = (1,),
                    initializer = 'ones',
                    trainable   = True,
                )
                super().build(input_shape)  # marks layer as built

            def call(self, inputs, training=False):
                output = inputs @ self.W * self.scale
                if self.use_bias:
                    output = output + self.b
                return output

            def get_config(self):
                config = super().get_config()
                config.update({'units': self.units, 'use_bias': self.use_bias})
                return config

### add_weight Arguments

    self.add_weight(
        name        = 'kernel',           # used in weight names and summaries
        shape       = (in_dim, out_dim),
        dtype       = 'float32',          # inherits layer dtype by default
        initializer = 'glorot_uniform',   # or keras.initializers.GlorotUniform()
        regularizer = None,               # or keras.regularizers.L2(1e-4)
        constraint  = None,               # or keras.constraints.NonNeg()
        trainable   = True,               # False = frozen, not updated
    )

    Common initializers:
        'glorot_uniform'  (Xavier uniform) — default for Dense weight matrices
        'glorot_normal'   (Xavier normal)  — Dense, smoother than uniform
        'he_uniform'      (He uniform)     — ReLU networks
        'he_normal'       (He normal)      — ReLU, widely used in ResNets
        'lecun_normal'                     — SELU activations
        'orthogonal'                       — RNNs, transformers
        'zeros', 'ones'                    — biases, gates

### Non-Trainable Weights and the training Flag

    Weights can be non-trainable (updated by logic, not by gradients):
        self.running_mean = self.add_weight(
            name='running_mean', shape=(dim,), trainable=False
        )

    Update non-trainable weights in call() using:
        self.running_mean.assign(new_value)
        self.running_mean.assign_add(delta)

    The training flag in call() is critical for:
        Dropout:        skip dropout connections during inference
        BatchNorm:      use batch stats (train) vs running stats (inference)
        Any stochastic computation that should only happen during training

    Always route training down to sub-layers:
        def call(self, x, training=False):
            x = self.bn(x, training=training)    # route training
            x = self.dropout(x, training=training)
            return x

### Built-In Layer Reference

    Core:
        Dense(units, activation, use_bias, kernel_regularizer)
        Activation('relu'), ReLU(), LeakyReLU(0.2), GELU(), Swish()
        Lambda(lambda x: x * 2)            inline computation

    Convolutional:
        Conv1D(filters, kernel_size, strides, padding, dilation_rate)
        Conv2D(filters, kernel_size, strides, padding='same')
        Conv2DTranspose(filters, kernel_size, strides)   upsampling / decoder
        SeparableConv2D(filters, kernel_size)            depthwise-separable
        DepthwiseConv2D(kernel_size)                     channel-independent

    Pooling:
        MaxPooling2D(pool_size, strides), AveragePooling2D(...)
        GlobalAveragePooling2D()                         spatial → vector
        GlobalMaxPooling2D()

    Normalisation:
        BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3)
        LayerNormalization(axis=-1)
        GroupNormalization(groups=8)                     between BN and LN
        InstanceNormalization()                          per-sample per-channel

    Regularisation:
        Dropout(rate=0.5)
        SpatialDropout2D(rate=0.5)                       drop entire feature maps
        GaussianNoise(stddev=0.1)                        additive Gaussian noise
        ActivityRegularization(l1=0.0, l2=1e-4)

    Recurrent:
        LSTM(units, return_sequences, return_state, go_backwards)
        GRU(units, return_sequences)
        SimpleRNN(units)
        Bidirectional(LSTM(units))                       wrap any RNN

    Attention:
        MultiHeadAttention(num_heads, key_dim, value_dim, dropout)
        Attention(use_scale=True)                        dot-product attention

    Merging (require list input in Functional API):
        Add(), Subtract(), Multiply()
        Concatenate(axis=-1)
        Average(), Maximum(), Minimum()
        Dot(axes=(1, 1))

    Reshaping:
        Flatten()
        Reshape(target_shape)
        RepeatVector(n)
        Permute(dims)
        Cropping2D, ZeroPadding2D

    Embedding:
        Embedding(input_dim, output_dim, embeddings_regularizer)


##### PART 4 — COMPILE AND FIT: THE DECLARATIVE TRAINING SYSTEM

### model.compile() — Configuring the Training

    compile() binds three things to the model: the optimiser, the loss
    function, and the metrics to track. This information is used by the
    Trainer (model.fit()) to automate the training loop.

        model.compile(
            optimizer = keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-2),
            loss      = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics   = [
                keras.metrics.SparseCategoricalAccuracy(name='acc'),
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc'),
            ],
        )

    Loss specification options:
        String:             loss='sparse_categorical_crossentropy'
        Class instance:     loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        Python function:    loss=my_loss_fn
        Multiple outputs:   loss={'output_a': 'mse', 'output_b': 'binary_crossentropy'}
        Loss weights:       loss_weights={'output_a': 1.0, 'output_b': 0.5}

    The from_logits flag — the most common compile bug:
        SparseCategoricalCrossentropy(from_logits=True):
            Expects RAW logits. Internally applies softmax + log + negative.
            Numerically stable. Use this.
        SparseCategoricalCrossentropy(from_logits=False):
            Expects probabilities (softmax output). Use only if your model
            explicitly applies softmax in the final layer.
        RULE: Do not put softmax in the final Dense layer.
              Set from_logits=True in the loss. Always.

### model.fit() — The Automated Training Loop

    model.fit(
        x                  = train_dataset,    # or (X, y), or generator
        validation_data    = (X_val, y_val),   # or val_dataset
        epochs             = 50,
        batch_size         = 32,               # ignored if x is a Dataset
        class_weight       = {0: 1.0, 1: 5.0}, # reweight minority classes
        sample_weight      = sample_weights,    # per-sample importance
        shuffle            = True,             # shuffle at each epoch start
        initial_epoch      = 0,                # resume from checkpoint
        steps_per_epoch    = None,             # None = full epoch
        validation_steps   = None,             # None = full val set
        callbacks          = [es, ckpt, tb],
        verbose            = 1,                # 0=silent, 1=bar, 2=epoch only
    )

    fit() returns a History object:
        history = model.fit(...)
        history.history              # dict: {'loss': [...], 'val_loss': [...]}
        history.history['val_acc']   # list of val accuracy per epoch
        history.params               # {'epochs': 50, 'steps': 40, ...}

### Loss Functions

    Classification:
        SparseCategoricalCrossentropy(from_logits=True)
            Targets: integer class indices (shape [batch,])
        CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
            Targets: one-hot encoded (shape [batch, n_classes])
            label_smoothing=0.1: replace hard 0/1 targets with 0.1/0.9
        BinaryCrossentropy(from_logits=True)
            Binary classification; targets: 0 or 1

    Regression:
        MeanSquaredError()                MSE loss
        MeanAbsoluteError()               MAE, robust to outliers
        Huber(delta=1.0)                  quadratic near 0, linear for |e|>delta
        LogCosh()                         smooth approximation to MAE
        MeanAbsolutePercentageError()     scale-independent

    Ranking / Metric learning:
        CosineSimilarity(axis=-1)         maximise cosine similarity
        KLDivergence()                    KL(P || Q), used in VAEs, knowledge distil.

    Custom loss function (two patterns):

        # Pattern 1: simple function (no state needed)
        def contrastive_loss(y_true, y_pred, margin=1.0):
            d = y_pred  # pairwise distance
            loss = (1 - y_true) * 0.5 * d**2 \
                 + y_true * 0.5 * keras.ops.maximum(margin - d, 0)**2
            return keras.ops.mean(loss)
        model.compile(loss=contrastive_loss, ...)

        # Pattern 2: class with get_config (serialisable, recommended)
        class ContrastiveLoss(keras.losses.Loss):
            def __init__(self, margin=1.0, **kwargs):
                super().__init__(**kwargs)
                self.margin = margin
            def call(self, y_true, y_pred):
                d = y_pred
                return keras.ops.mean(
                    (1-y_true)*0.5*d**2 + y_true*0.5*keras.ops.maximum(self.margin-d,0)**2
                )
            def get_config(self):
                return {**super().get_config(), 'margin': self.margin}

### Optimisers

    SGD family:
        keras.optimizers.SGD(learning_rate=0.1, momentum=0.9,
                              nesterov=True, weight_decay=1e-4)

    Adaptive:
        keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
        keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-2)   ← preferred
        keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)

    Modern:
        keras.optimizers.Adamax(learning_rate=2e-3)
        keras.optimizers.Nadam(learning_rate=2e-3)      Nesterov + Adam
        keras.optimizers.Adagrad(learning_rate=0.01)

    Learning rate schedules (pass as learning_rate argument):
        ExponentialDecay:   lr × decay_rate every decay_steps
        CosineDecay:        cosine curve from initial_lr to alpha
        CosineDecayRestarts: warm restarts (SGDR — Loshchilov & Hutter)
        PolynomialDecay:    lr decays as (1 - step/max_steps)^power
        PiecewiseConstantDecay: different lr for different step ranges

        Example:
            schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate = 1e-3,
                decay_steps           = 10000,
                alpha                 = 1e-5,   # minimum lr
            )
            optimizer = keras.optimizers.AdamW(learning_rate=schedule)

### Built-In Metrics

    Classification:
        Accuracy()
        SparseCategoricalAccuracy()       integer targets
        CategoricalAccuracy()             one-hot targets
        TopKCategoricalAccuracy(k=5)
        AUC(curve='ROC'), AUC(curve='PR') area under ROC / PR curve
        Precision(), Recall(), F1Score()

    Regression:
        MeanSquaredError()
        MeanAbsoluteError()
        RootMeanSquaredError()
        R2Score()                         coefficient of determination

    Custom metric (stateful accumulation — required for correctness):
        class MeanIoU(keras.metrics.Metric):
            def __init__(self, num_classes, **kwargs):
                super().__init__(**kwargs)
                self.total = self.add_weight('total', initializer='zeros')
                self.count = self.add_weight('count', initializer='zeros')
                self.num_classes = num_classes

            def update_state(self, y_true, y_pred, sample_weight=None):
                # accumulate state from each batch
                iou = compute_iou(y_true, y_pred, self.num_classes)
                self.total.assign_add(keras.ops.sum(iou))
                self.count.assign_add(keras.ops.cast(keras.ops.shape(y_true)[0], 'float32'))

            def result(self):
                return self.total / self.count   # epoch-level result

            def reset_state(self):
                self.total.assign(0.0)
                self.count.assign(0.0)


##### PART 5 — DATA PIPELINES: tf.data AND KERAS PREPROCESSING LAYERS

### tf.data Integration

    Keras model.fit() accepts:
        NumPy arrays:        (X_train, y_train) — simple, fits in RAM
        TensorFlow Datasets: tf.data.Dataset    — streaming, efficient, scalable
        Python generators:   must yield (X_batch, y_batch)
        Keras Sequence:      subclass keras.utils.PyDataset

    The canonical tf.data pipeline for image training:

        train_ds = (
            tf.data.Dataset
            .from_tensor_slices((X_train, y_train))
            .shuffle(buffer_size=10_000)
            .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )

    AUTOTUNE: TensorFlow dynamically determines the optimal parallelism level.
    Always use AUTOTUNE for both num_parallel_calls and prefetch.

### Keras Preprocessing Layers — The Training/Serving Consistency Guarantee

    The most important innovation in Keras 2.6+:
    Preprocessing layers adapt FROM the data during fit, then become
    baked into the model for inference.

    This solves training-serving skew:
        Problem: you normalise data with mean/std computed on train set.
                 At serving time you must replicate this computation.
                 If you do it differently, predictions are wrong.
        Solution: bake the normalisation into the model as a layer.
                  The same computation runs at serving time automatically.

    Normalization layer (feature scaling):
        normalizer = keras.layers.Normalization(axis=-1)
        normalizer.adapt(X_train)     # computes mean and variance from data
        # Now: normalizer.mean, normalizer.variance are fixed

        model = keras.Sequential([
            normalizer,               # first layer: auto-normalises input
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10),
        ])
        # At inference: model(raw_X) auto-normalises, no manual preprocessing

    TextVectorization (string → integer sequences):
        vectoriser = keras.layers.TextVectorization(
            max_tokens       = 20_000,
            output_mode      = 'int',         # 'multi_hot', 'tf_idf', 'int'
            output_sequence_length = 256,
        )
        vectoriser.adapt(text_dataset)        # builds vocabulary from data

        model = keras.Sequential([
            vectoriser,                       # raw strings in → token IDs out
            keras.layers.Embedding(20_000, 128),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10),
        ])

    Image augmentation layers (training only):
        augmentation = keras.Sequential([
            keras.layers.RandomFlip('horizontal'),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.2),
        ])

        # Augmentation only during training (use training=True/False):
        class AugmentedModel(keras.Model):
            def call(self, x, training=False):
                if training:
                    x = self.augmentation(x, training=True)
                return self.backbone(x)

    Other preprocessing layers:
        Rescaling(scale=1./255)                  pixel normalisation
        CenterCrop(height, width)                centre crop
        RandomCrop(height, width)                random crop
        CategoryEncoding(num_tokens, mode)       OHE / count encoding
        Hashing(num_bins)                        hash bucketing
        IntegerLookup, StringLookup              vocabulary mapping
        Discretisation(bin_boundaries)           continuous → buckets

### PyDataset — The Keras Dataset Interface

    For datasets that don't fit in memory or require complex preprocessing:

        class ImageDataset(keras.utils.PyDataset):
            def __init__(self, file_paths, labels, batch_size, **kwargs):
                super().__init__(**kwargs)
                self.file_paths = file_paths
                self.labels     = labels
                self.batch_size = batch_size

            def __len__(self):
                return math.ceil(len(self.file_paths) / self.batch_size)

            def __getitem__(self, idx):
                batch_files  = self.file_paths[idx * self.batch_size:
                                               (idx+1) * self.batch_size]
                batch_labels = self.labels[idx * self.batch_size:
                                           (idx+1) * self.batch_size]
                images = [load_and_preprocess(f) for f in batch_files]
                return np.array(images), np.array(batch_labels)

        train_gen = ImageDataset(train_files, train_labels, batch_size=32,
                                  workers=4, use_multiprocessing=True)
        model.fit(train_gen, epochs=10)


##### PART 6 — CALLBACKS: THE TRAINING HOOK SYSTEM

### What Callbacks Are

    Callbacks are objects that are called at specific points in the training
    loop. They receive the model, logs, and epoch/batch number at each hook
    point and can read or modify any model state.

    They are passed to model.fit(callbacks=[...]).
    Multiple callbacks run in list order at each hook point.

    The key insight: ALL training customisation that doesn't modify the loss
    or gradient computation belongs in a callback, not in the model.
    The model stays clean. The callback is reusable across models.

### Built-In Callbacks

    ModelCheckpoint:
        Saves the model (or just weights) at specified intervals.

        keras.callbacks.ModelCheckpoint(
            filepath           = 'ckpts/model_{epoch:02d}_{val_loss:.4f}.keras',
            monitor            = 'val_loss',
            mode               = 'min',           # save when val_loss decreases
            save_best_only     = True,            # only keep the best checkpoint
            save_weights_only  = False,           # save full model vs just weights
            save_freq          = 'epoch',         # or integer (every N batches)
        )

        Restore from checkpoint:
            model = keras.models.load_model('ckpts/model_05_0.2341.keras')
            # or, for weights only:
            model.load_weights('ckpts/best_weights.weights.h5')

    EarlyStopping:
        Stops training when a monitored metric stagnates.

        keras.callbacks.EarlyStopping(
            monitor              = 'val_loss',
            patience             = 10,
            min_delta            = 1e-4,
            mode                 = 'min',
            restore_best_weights = True,     # revert to best epoch on stop
            baseline             = None,     # must improve past this threshold
        )

    ReduceLROnPlateau:
        Reduces learning rate when a metric stops improving.

        keras.callbacks.ReduceLROnPlateau(
            monitor   = 'val_loss',
            factor    = 0.5,         # new_lr = lr × factor
            patience  = 5,           # wait 5 epochs before reducing
            min_lr    = 1e-7,        # never reduce below this
            min_delta = 1e-4,
            cooldown  = 2,           # wait N epochs before resuming monitoring
        )

    TensorBoard:
        Logs metrics, weight histograms, computation graphs to TensorBoard.

        keras.callbacks.TensorBoard(
            log_dir          = './tb_logs',
            histogram_freq   = 1,        # log weight histograms every N epochs
            write_graph      = True,     # log computation graph
            write_images     = False,    # log weight images
            update_freq      = 'epoch',  # 'epoch' or 'batch' or integer
            profile_batch    = 0,        # 0=off, N=profile batch N
        )

    LearningRateScheduler:
        Applies a custom Python function to compute lr at each epoch.

        def cosine_schedule(epoch, lr):
            return lr_max * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))

        keras.callbacks.LearningRateScheduler(cosine_schedule, verbose=1)

    BackupAndRestore:
        Backs up the training state at the end of each epoch.
        Automatically restores if training is interrupted (fault tolerance).

        keras.callbacks.BackupAndRestore(backup_dir='./backup')

    TerminateOnNaN:
        Stops training immediately if loss becomes NaN or Inf.
        Simple but critical for stability debugging.

### The Full Callback Hook Interface

    Every Callback can implement these hooks:

        class MyCallback(keras.callbacks.Callback):
            # ── Fit lifecycle ─────────────────────────────────────
            def on_train_begin(self, logs=None): ...
            def on_train_end(self, logs=None): ...

            # ── Epoch lifecycle ───────────────────────────────────
            def on_epoch_begin(self, epoch, logs=None): ...
            def on_epoch_end(self, epoch, logs=None):
                # logs contains: 'loss', 'acc', 'val_loss', 'val_acc', etc.
                ...

            # ── Batch lifecycle (training) ────────────────────────
            def on_train_batch_begin(self, batch, logs=None): ...
            def on_train_batch_end(self, batch, logs=None): ...

            # ── Batch lifecycle (validation) ──────────────────────
            def on_test_begin(self, logs=None): ...
            def on_test_batch_begin(self, batch, logs=None): ...
            def on_test_batch_end(self, batch, logs=None): ...
            def on_test_end(self, logs=None): ...

            # ── Prediction lifecycle ──────────────────────────────
            def on_predict_begin(self, logs=None): ...
            def on_predict_batch_begin(self, batch, logs=None): ...
            def on_predict_batch_end(self, batch, logs=None): ...
            def on_predict_end(self, logs=None): ...

    Access model internals inside any callback:
        self.model                           the Keras model
        self.model.optimizer                 the current optimiser
        self.model.optimizer.learning_rate   current lr (as a tensor)
        self.model.get_layer('name')         a specific layer
        self.model.trainable_weights         list of weight tensors


##### PART 7 — CUSTOM TRAINING LOOPS: train_step AND GRADIENTTAPE

### The Three Levels of Training Loop Control

    Level 1 — model.compile() + model.fit():
        Everything automated. No training code. For standard supervised tasks.

    Level 2 — Override train_step():
        Keep model.fit() with all its automation (callbacks, metrics, logging)
        but replace the inner computation for one step.
        Use for: GANs, knowledge distillation, custom loss logic, multi-task.

    Level 3 — Manual GradientTape loop:
        Write the entire training loop manually.
        Use for: research that requires complete loop control,
                 custom distributed training, RL, meta-learning.

### Overriding train_step() — The Sweet Spot

    The most powerful under-used feature in Keras. You keep all of
    model.fit()'s benefits — callbacks, progress bars, validation,
    metric accumulation — while replacing the inner per-step computation.

        class CustomTrainingModel(keras.Model):
            def train_step(self, data):
                x, y = data      # unpack however your data is structured

                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    loss   = self.compute_loss(y=y, y_pred=y_pred)
                    # add_metric, add_loss can also be called here

                gradients = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_weights)
                )

                # Update metrics (Keras handles averaging automatically)
                for metric in self.metrics:
                    if metric.name == 'loss':
                        metric.update_state(loss)
                    else:
                        metric.update_state(y, y_pred)

                return {m.name: m.result() for m in self.metrics}

            def test_step(self, data):
                x, y   = data
                y_pred = self(x, training=False)
                loss   = self.compute_loss(y=y, y_pred=y_pred)
                for metric in self.metrics:
                    if metric.name == 'loss':
                        metric.update_state(loss)
                    else:
                        metric.update_state(y, y_pred)
                return {m.name: m.result() for m in self.metrics}

    The key invariant: train_step() must return a dict of metric names →
    metric values. These values appear in the History object and progress bar.

### GAN train_step — The Canonical Non-Trivial Example

    GANs require two separate optimisers and two separate gradient tapes:

        class GAN(keras.Model):
            def __init__(self, generator, discriminator, latent_dim):
                super().__init__()
                self.generator     = generator
                self.discriminator = discriminator
                self.latent_dim    = latent_dim
                self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
                self.d_loss_tracker = keras.metrics.Mean(name='d_loss')

            @property
            def metrics(self):
                return [self.g_loss_tracker, self.d_loss_tracker]

            def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
                super().compile()
                self.d_optimizer = d_optimizer
                self.g_optimizer = g_optimizer
                self.d_loss_fn   = d_loss_fn
                self.g_loss_fn   = g_loss_fn

            def train_step(self, real_data):
                batch_size = tf.shape(real_data)[0]
                noise      = tf.random.normal([batch_size, self.latent_dim])

                # Step 1: Train discriminator
                with tf.GradientTape() as d_tape:
                    fake_data = self.generator(noise, training=True)
                    d_real    = self.discriminator(real_data, training=True)
                    d_fake    = self.discriminator(fake_data, training=True)
                    d_loss    = self.d_loss_fn(d_real, d_fake)

                d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
                self.d_optimizer.apply_gradients(
                    zip(d_grads, self.discriminator.trainable_weights)
                )

                # Step 2: Train generator (new tape — D weights frozen)
                noise2 = tf.random.normal([batch_size, self.latent_dim])
                with tf.GradientTape() as g_tape:
                    fake_data2 = self.generator(noise2, training=True)
                    d_fake2    = self.discriminator(fake_data2, training=True)
                    g_loss     = self.g_loss_fn(d_fake2)

                g_grads = g_tape.gradient(g_loss, self.generator.trainable_weights)
                self.g_optimizer.apply_gradients(
                    zip(g_grads, self.generator.trainable_weights)
                )

                self.d_loss_tracker.update_state(d_loss)
                self.g_loss_tracker.update_state(g_loss)
                return {'d_loss': self.d_loss_tracker.result(),
                        'g_loss': self.g_loss_tracker.result()}

    Call model.fit(real_dataset, epochs=100) — everything else works normally.

### Knowledge Distillation train_step

    The student learns from soft labels (teacher logits), not hard labels:

        class Distiller(keras.Model):
            def __init__(self, student, teacher):
                super().__init__()
                self.student     = student
                self.teacher     = teacher
                self.loss_tracker = keras.metrics.Mean(name='loss')
                self.acc_metric   = keras.metrics.SparseCategoricalAccuracy(name='acc')

            def compile(self, optimizer, student_loss_fn, distill_loss_fn,
                         alpha=0.1, temperature=5.0):
                super().compile(optimizer=optimizer)
                self.student_loss_fn = student_loss_fn    # hard label loss
                self.distill_loss_fn = distill_loss_fn    # KL divergence
                self.alpha           = alpha               # mix weight
                self.temperature     = temperature

            def train_step(self, data):
                x, y = data
                teacher_pred = self.teacher(x, training=False)  # frozen teacher

                with tf.GradientTape() as tape:
                    student_pred = self.student(x, training=True)
                    # Hard label loss (cross-entropy with true labels)
                    student_loss = self.student_loss_fn(y, student_pred)
                    # Soft label loss (KL divergence between soft distributions)
                    soft_targets = tf.nn.softmax(teacher_pred / self.temperature, axis=1)
                    soft_preds   = tf.nn.softmax(student_pred / self.temperature, axis=1)
                    distill_loss = self.distill_loss_fn(soft_targets, soft_preds)
                    # Combined loss
                    loss = self.alpha * student_loss \
                         + (1 - self.alpha) * distill_loss * self.temperature**2

                grads = tape.gradient(loss, self.student.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads, self.student.trainable_weights)
                )
                self.loss_tracker.update_state(loss)
                self.acc_metric.update_state(y, student_pred)
                return {'loss': self.loss_tracker.result(),
                        'acc': self.acc_metric.result()}

### Manual GradientTape Loop — Full Control

    For research that needs complete loop customisation:

        optimizer = keras.optimizers.AdamW(learning_rate=3e-4)
        loss_fn   = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        @tf.function    # compile for speed
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss   = loss_fn(y, logits)
                loss  += sum(model.losses)          # regularisation losses
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return loss

        for epoch in range(epochs):
            for x_batch, y_batch in train_dataset:
                loss = train_step(x_batch, y_batch)


##### PART 8 — MIXED PRECISION, SAVING, AND DEPLOYMENT

### Mixed Precision

    Enable globally (affects all models created after this call):
        keras.mixed_precision.set_global_policy('mixed_float16')
        # or:
        keras.mixed_precision.set_global_policy('mixed_bfloat16')

    Under the hood:
        - Compute (matmul, conv) runs in float16/bfloat16
        - Weights are stored in float32 (numerical stability)
        - Keras automatically wraps optimiser with LossScaleOptimiser
          when using float16 (handles gradient underflow)
        - bfloat16: no loss scaling needed (same exponent range as float32)

    Per-layer dtype override (when global policy isn't appropriate):
        output_layer = keras.layers.Dense(10, dtype='float32')   # keep final layer in f32

    Check current policy:
        keras.mixed_precision.global_policy()   # returns Policy('mixed_float16')

### Model Saving Formats

    Keras 3 supports three saving formats:

    1. Keras format (.keras) — RECOMMENDED for Keras 3:
        model.save('model.keras')
        model = keras.models.load_model('model.keras')

        Saves: architecture, weights, optimiser state, compile config.
        Backend-agnostic: saved on TF, loadable on JAX backend.
        Human-readable ZIP archive containing JSON + weights HDF5.

    2. SavedModel format — for TF Serving / TFLite / production:
        model.export('saved_model_dir/')
        # Creates: saved_model.pb + variables/ directory

        Load for inference (TF serving, or Python):
            loaded = tf.saved_model.load('saved_model_dir/')
            output = loaded.serve(input_tensor)

        Limitations: saves only inference graph (@tf.function traces).
        Cannot resume training from a SavedModel.

    3. HDF5 format (.h5) — legacy, avoid for new projects:
        model.save('model.h5')
        model = keras.models.load_model('model.h5')

        Works but: larger files, slower to save/load, TF-backend only.

    Save / load weights only (for fine-tuning from a checkpoint):
        model.save_weights('weights.weights.h5')
        model.load_weights('weights.weights.h5')

        For partial loading (load backbone but not head):
            backbone_model.save_weights('backbone.h5')
            full_model.get_layer('backbone').set_weights(
                backbone_model.get_weights()
            )

### TFLite Conversion

    TFLite is TensorFlow Lite — the edge/mobile inference runtime.
    Conversion pipeline: Keras model → SavedModel → .tflite flat buffer.

        # FP32 conversion (no quantisation)
        converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir/')
        tflite_model = converter.convert()

        # Dynamic range quantisation (weights INT8, activations FP32)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        # Full integer quantisation (requires representative dataset)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tflite_model = converter.convert()

        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)

    Inference with TFLite interpreter:
        interp = tf.lite.Interpreter('model.tflite')
        interp.allocate_tensors()
        in_idx  = interp.get_input_details()[0]['index']
        out_idx = interp.get_output_details()[0]['index']
        interp.set_tensor(in_idx, input_data)
        interp.invoke()
        result = interp.get_tensor(out_idx)

### Quantisation-Aware Training (QAT)

    QAT trains the model with fake quantisation nodes inserted.
    The model learns to compensate for quantisation error during training.
    Result: higher INT8 accuracy than post-training quantisation.

        import tensorflow_model_optimization as tfmot

        quant_aware_model = tfmot.quantization.keras.quantize_model(model)
        quant_aware_model.compile(optimizer='adam',
                                   loss='sparse_categorical_crossentropy',
                                   metrics=['acc'])
        quant_aware_model.fit(train_ds, epochs=5)   # fine-tune with QAT

        converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        qat_tflite = converter.convert()

### Deployment Decision Guide

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Target                  │ Format           │ Notes                   │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Continue training       │ .keras           │ Full state, portable    │
    │ Python inference server │ SavedModel       │ tf.saved_model.load()   │
    │ TF Serving (Docker)     │ SavedModel       │ REST + gRPC endpoints   │
    │ Android / iOS           │ .tflite          │ TFLite runtime          │
    │ Microcontroller         │ .tflite          │ TF Micro                │
    │ Browser (JavaScript)    │ TFJS format      │ tensorflowjs_converter  │
    │ NVIDIA GPU (prod.)      │ ONNX → TensorRT  │ max GPU throughput      │
    │ Intel CPU               │ ONNX → OpenVINO  │ max CPU throughput      │
    └──────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Three Model APIs & Custom Layers — Sequential, Functional, Subclass": {
        "description": (
            "Complete tour of all three Keras model-building APIs. "
            "Sequential API for linear stacks with regularisation. "
            "Functional API: residual blocks, skip connections, multi-output. "
            "Siamese network (shared weights) with Functional API. "
            "Model subclassing: Transformer encoder with dynamic masks. "
            "Custom Layer with build() / call() / add_weight() / get_config(). "
            "Non-trainable weights updated via assign(). "
            "Parameter counting, summary, and plot_model introspection."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

try:
    import tensorflow as tf
    import keras
    print(f"  Keras {keras.__version__} | TF {tf.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'tensorflow', 'keras', '--quiet'], check=True)
    import tensorflow as tf
    import keras

print("=" * 65)
print("  THREE MODEL APIs & CUSTOM LAYERS")
print("=" * 65)
print()

tf.random.set_seed(42)
np.random.seed(42)

# ── Shared dataset ─────────────────────────────────────────────────────
N_TRAIN, N_VAL = 2000, 400
FEAT, CLS      = 32, 5

X_tr = np.random.randn(N_TRAIN, FEAT).astype('float32')
y_tr = np.random.randint(0, CLS, N_TRAIN)
X_va = np.random.randn(N_VAL,   FEAT).astype('float32')
y_va = np.random.randint(0, CLS, N_VAL)

print(f"  Dataset: {N_TRAIN} train / {N_VAL} val | {FEAT} features | {CLS} classes")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Sequential API
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Sequential API (linear stack)")
print("━" * 65)
print()

model_seq = keras.Sequential([
    keras.Input(shape=(FEAT,), name='features'),
    keras.layers.Dense(128, activation='relu',
                       kernel_regularizer=keras.regularizers.L2(1e-4),
                       name='dense1'),
    keras.layers.BatchNormalization(name='bn1'),
    keras.layers.Dropout(0.3, name='drop1'),
    keras.layers.Dense(64, activation='relu', name='dense2'),
    keras.layers.Dropout(0.2, name='drop2'),
    keras.layers.Dense(CLS, name='logits'),          # NO activation — use from_logits=True
], name='sequential_model')

model_seq.compile(
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-2),
    loss      = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics   = [keras.metrics.SparseCategoricalAccuracy(name='acc')],
)

total_p     = model_seq.count_params()
trainable_p = sum(np.prod(w.shape) for w in model_seq.trainable_weights)
print(f"  Total params:     {total_p:,}")
print(f"  Trainable params: {trainable_p:,}")
print(f"  Layer names:      {[l.name for l in model_seq.layers]}")
print()

hist_seq = model_seq.fit(
    X_tr, y_tr,
    validation_data = (X_va, y_va),
    epochs          = 8,
    batch_size      = 64,
    verbose         = 0,
)
final_val_acc = hist_seq.history['val_acc'][-1]
print(f"  Sequential trained 8 epochs → val_acc = {final_val_acc:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Functional API — residual MLP
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Functional API (residual connections)")
print("━" * 65)
print()

def residual_block(x, units, dropout=0.2):
    """Pre-norm residual block: LayerNorm → Dense → GELU → Dense → Add."""
    skip = x
    if keras.ops.shape(x)[-1] != units:
        skip = keras.layers.Dense(units, use_bias=False)(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dense(units, activation='gelu')(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(units)(x)
    return keras.layers.Add()([x, skip])

# Build a 3-block residual MLP using Functional API
inputs = keras.Input(shape=(FEAT,), name='features')
x      = keras.layers.Dense(64, activation='gelu', name='embed')(inputs)
x      = residual_block(x, 64)
x      = residual_block(x, 64)
x      = residual_block(x, 32)
x      = keras.layers.LayerNormalization(name='final_norm')(x)
outputs = keras.layers.Dense(CLS, name='logits')(x)

model_func = keras.Model(inputs=inputs, outputs=outputs, name='residual_mlp')
model_func.compile(
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3),
    loss      = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics   = ['sparse_categorical_accuracy'],
)

print(f"  Residual MLP — params: {model_func.count_params():,}")
print(f"  Layers:  {len(model_func.layers)}")
layer_types = sorted({type(l).__name__ for l in model_func.layers})
print(f"  Types:   {layer_types}")
print()

hist_func = model_func.fit(
    X_tr, y_tr,
    validation_data = (X_va, y_va),
    epochs          = 10,
    batch_size      = 64,
    verbose         = 0,
)
print(f"  Residual MLP trained 10 epochs → val_acc = {hist_func.history['sparse_categorical_accuracy'][-1]:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Functional API — multi-output and shared layers (Siamese)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Functional API: Siamese (shared weights) network")
print("━" * 65)
print()

# Shared encoder — SAME layer object called on both branches
shared_encoder = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.LayerNormalization(),
], name='shared_encoder')

input_a = keras.Input(shape=(FEAT,), name='input_a')
input_b = keras.Input(shape=(FEAT,), name='input_b')

# SAME layer called twice → weights are truly shared
emb_a   = shared_encoder(input_a)
emb_b   = shared_encoder(input_b)

diff     = keras.layers.Subtract()([emb_a, emb_b])
distance = keras.layers.Dense(1, activation='sigmoid', name='distance')(diff)

siamese = keras.Model(
    inputs  = [input_a, input_b],
    outputs = distance,
    name    = 'siamese_network',
)
siamese.compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy'],
)

print(f"  Siamese network — params: {siamese.count_params():,}")
print(f"  shared_encoder trainable weights: {len(shared_encoder.trainable_weights)}")
print(f"  (called twice, but weights exist only ONCE — this is shared weights)")
print()

# Generate pair data: pairs from the same class → label 1, else 0
n_pairs = 1000
idx_a   = np.random.randint(0, N_TRAIN, n_pairs)
idx_b   = np.random.randint(0, N_TRAIN, n_pairs)
labels  = (y_tr[idx_a] == y_tr[idx_b]).astype('float32')

hist_siam = siamese.fit(
    [X_tr[idx_a], X_tr[idx_b]], labels,
    epochs = 6, batch_size = 64, verbose = 0,
    validation_split = 0.2,
)
print(f"  Siamese trained 6 epochs → val_acc = {hist_siam.history['val_accuracy'][-1]:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Model Subclassing — Transformer encoder with masking
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Subclassing: Transformer encoder block")
print("━" * 65)
print()

class TransformerBlock(keras.Layer):
    """Pre-norm transformer block: MHA + FFN with residual connections."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=dropout
        )
        self.ffn   = keras.Sequential([
            keras.layers.Dense(d_ff, activation='gelu'),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(d_model),
        ])
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        # Pre-norm: normalise before each sub-layer (GPT-2 / LLaMA style)
        attn_out = self.attn(
            self.norm1(x), self.norm1(x),
            attention_mask=mask,
            training=training,
        )
        x = x + self.drop1(attn_out, training=training)   # residual
        ffn_out = self.ffn(self.norm2(x), training=training)
        return x + self.drop2(ffn_out, training=training)  # residual

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.attn.key_dim * self.attn.num_heads,
            'n_heads': self.attn.num_heads,
            'd_ff':    self.ffn.layers[0].units,
        })
        return config


class TransformerClassifier(keras.Model):
    def __init__(self, vocab_size, d_model, n_heads, d_ff,
                 n_layers, n_classes, max_len=64, dropout=0.1):
        super().__init__()
        self.embed   = keras.layers.Embedding(vocab_size, d_model)
        self.pos_emb = keras.layers.Embedding(max_len,    d_model)  # learned pos encoding
        self.blocks  = [TransformerBlock(d_model, n_heads, d_ff, dropout)
                        for _ in range(n_layers)]
        self.norm    = keras.layers.LayerNormalization()
        self.pool    = keras.layers.GlobalAveragePooling1D()
        self.drop    = keras.layers.Dropout(dropout)
        self.head    = keras.layers.Dense(n_classes)

    def call(self, token_ids, training=False):
        seq_len  = tf.shape(token_ids)[1]
        pos_ids  = tf.range(seq_len)[tf.newaxis, :]
        x        = self.embed(token_ids) + self.pos_emb(pos_ids)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        x = self.pool(x)      # mean-pool across sequence length
        x = self.drop(x, training=training)
        return self.head(x)

clf = TransformerClassifier(
    vocab_size=1000, d_model=64, n_heads=4, d_ff=256,
    n_layers=2, n_classes=5
)
dummy_tokens = tf.constant(np.random.randint(0, 1000, (8, 32)))
out = clf(dummy_tokens, training=False)
print(f"  TransformerClassifier:")
print(f"    Input:  {tuple(dummy_tokens.shape)}  (batch, seq_len)")
print(f"    Output: {tuple(out.shape)}  (batch, n_classes)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Custom Layer — build() / call() / add_weight()
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Custom Layer with build/call/add_weight")
print("━" * 65)
print()

class FeatureWiseLinearModulation(keras.Layer):
    """
    FiLM: Feature-wise Linear Modulation.
    Modulates a feature tensor x by learned scale (gamma) and shift (beta)
    computed from a conditioning input c: output = gamma(c) * x + beta(c).
    Used in: conditional image generation, multi-task learning, meta-learning.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units       = units
        self.gamma_dense = keras.layers.Dense(units)   # sub-layers defined in __init__
        self.beta_dense  = keras.layers.Dense(units)

    def build(self, input_shape):
        # input_shape is a list: [x_shape, condition_shape]
        # sub-layers build themselves on first call — nothing extra needed here
        # but we can add weights if needed:
        self.scale_init = self.add_weight(
            name='global_scale', shape=(1,), initializer='ones', trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x, condition = inputs                # two inputs: feature + conditioning
        gamma = self.gamma_dense(condition)  # (batch, units)
        beta  = self.beta_dense(condition)   # (batch, units)
        return self.scale_init * gamma * x + beta

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# Build a FiLM-conditioned model using Functional API
feat_input = keras.Input(shape=(FEAT,), name='features')
cond_input = keras.Input(shape=(16,), name='condition')

x    = keras.layers.Dense(64, activation='relu')(feat_input)
x    = FeatureWiseLinearModulation(64)([x, cond_input])   # condition modulates x
x    = keras.layers.Dense(32, activation='relu')(x)
out  = keras.layers.Dense(CLS)(x)

film_model = keras.Model(inputs=[feat_input, cond_input], outputs=out, name='film_model')
film_model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'],
)

cond_data = np.random.randn(N_TRAIN, 16).astype('float32')
cond_val  = np.random.randn(N_VAL,   16).astype('float32')

hist_film = film_model.fit(
    [X_tr, cond_data], y_tr,
    validation_data = ([X_va, cond_val], y_va),
    epochs = 6, batch_size = 64, verbose = 0,
)
print(f"  FiLM-conditioned model — params: {film_model.count_params():,}")
print(f"  Trained 6 epochs → val_acc = {hist_film.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print()

# Custom non-trainable weight updated in call()
class ExponentialMovingMean(keras.Layer):
    """Demonstrates non-trainable weight updated via assign (not by gradient)."""
    def __init__(self, momentum=0.99, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum

    def build(self, input_shape):
        self.ema = self.add_weight(
            name='ema', shape=(input_shape[-1],),
            initializer='zeros', trainable=False    # NOT trained by gradient
        )
        super().build(input_shape)

    def call(self, x, training=False):
        if training:
            batch_mean = tf.reduce_mean(x, axis=0)
            new_ema    = self.momentum * self.ema + (1 - self.momentum) * batch_mean
            self.ema.assign(new_ema)               # update non-trainable weight
        return x - self.ema                         # centre by running mean

ema_layer = ExponentialMovingMean(momentum=0.95)
x_test    = tf.constant(np.random.randn(16, 8).astype('float32'))
_ = ema_layer(x_test, training=True)               # first call triggers build
_ = ema_layer(x_test, training=True)
_ = ema_layer(x_test, training=True)
print(f"  ExponentialMovingMean after 3 training calls:")
print(f"    ema (running mean): {ema_layer.ema.numpy().round(4)}")
print(f"    trainable_weights:  {len(ema_layer.trainable_weights)}  (EMA is non-trainable)")
print(f"    non_trainable_weights: {len(ema_layer.non_trainable_weights)}")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · compile/fit Full Stack — Losses, Metrics, Callbacks & Checkpointing": {
        "description": (
            "Complete compile/fit pipeline with advanced configuration. "
            "Custom loss function (label-smoothed focal loss). "
            "Custom stateful metric (per-class accuracy via update_state). "
            "LR schedule (cosine decay) baked into the optimiser. "
            "ModelCheckpoint: save best, templated filename, restore. "
            "EarlyStopping with restore_best_weights. "
            "ReduceLROnPlateau and LearningRateScheduler comparison. "
            "Custom Callback: gradient norm logger + sample generator. "
            "History object analysis: plotting loss and metric curves. "
            "class_weight for imbalanced datasets."
        ),
        "language": "python",
        "code": '''
import numpy as np
import os, tempfile, time
import tensorflow as tf
import keras

print("=" * 65)
print("  compile/fit FULL STACK — LOSSES, METRICS & CALLBACKS")
print("=" * 65)
print()

tf.random.set_seed(42)
np.random.seed(42)
print(f"  Keras: {keras.__version__} | TF: {tf.__version__}")
print()

# ── Shared dataset: intentionally imbalanced ──────────────────────────
N_TRAIN, N_VAL = 2400, 600
FEAT, CLS      = 32, 4
X_tr = np.random.randn(N_TRAIN, FEAT).astype('float32')
# Class imbalance: class 0 → 60%, others → ~13% each
class_probs = [0.60, 0.15, 0.15, 0.10]
y_tr = np.random.choice(CLS, N_TRAIN, p=class_probs).astype('int32')
X_va = np.random.randn(N_VAL, FEAT).astype('float32')
y_va = np.random.choice(CLS, N_VAL, p=class_probs).astype('int32')

counts = np.bincount(y_tr, minlength=CLS)
print(f"  Class distribution: {counts}  (imbalanced — class 0 dominates)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Custom loss — Focal Loss
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Custom loss: Focal Loss")
print("━" * 65)
print()

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss (Lin et al., 2017): down-weights easy examples, focuses
    on hard/misclassified examples. Excellent for class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=0: standard cross-entropy
    gamma=2: standard focal (penalises easy examples more)
    """
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma       = gamma
        self.alpha       = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        if self.from_logits:
            probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            probs = y_pred + 1e-7

        # Gather probability of the true class for each sample
        n_classes = tf.shape(y_pred)[-1]
        y_onehot  = tf.one_hot(y_true, n_classes)
        p_t       = tf.reduce_sum(probs * y_onehot, axis=-1)

        focal_weight = (1.0 - p_t) ** self.gamma
        ce           = -tf.math.log(p_t + 1e-7)
        return tf.reduce_mean(self.alpha * focal_weight * ce)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha,
                       'from_logits': self.from_logits})
        return config

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Custom metric — per-class accuracy
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Custom stateful metric (per-class accuracy)")
print("━" * 65)
print()

class PerClassAccuracy(keras.metrics.Metric):
    """
    Computes per-class accuracy for the MINORITY class (class 0 here).
    Demonstrates stateful metric accumulation: update_state, result, reset_state.
    """
    def __init__(self, target_class=0, **kwargs):
        super().__init__(**kwargs)
        self.target_class = target_class
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total   = self.add_weight(name='total',   initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true  = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred  = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        mask    = tf.equal(y_true, self.target_class)  # only target class
        correct = tf.cast(tf.equal(tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask)), tf.float32)
        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Cosine LR decay built into optimiser
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Cosine LR decay schedule in optimiser")
print("━" * 65)
print()

STEPS_PER_EPOCH = N_TRAIN // 64
MAX_EPOCHS      = 30
TOTAL_STEPS     = MAX_EPOCHS * STEPS_PER_EPOCH

cosine_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = 3e-3,
    decay_steps           = TOTAL_STEPS,
    alpha                 = 1e-5,           # final lr = alpha * initial_lr
)
optimizer = keras.optimizers.AdamW(
    learning_rate = cosine_schedule,
    weight_decay  = 1e-2,
)

# Sample a few schedule values to verify decay
lrs_sampled = [float(cosine_schedule(s * STEPS_PER_EPOCH)) for s in [0, 5, 10, 20, 30]]
print(f"  Cosine LR at epochs [0, 5, 10, 20, 30]:")
print(f"    {[f'{lr:.2e}' for lr in lrs_sampled]}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Build model and compile
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Model, compile, class_weight for imbalance")
print("━" * 65)
print()

def build_model(in_feat, n_cls):
    inp = keras.Input(shape=(in_feat,))
    x   = keras.layers.Dense(128, activation='gelu')(inp)
    x   = keras.layers.LayerNormalization()(x)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.Dense(64,  activation='gelu')(x)
    x   = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(n_cls)(x)
    return keras.Model(inp, out)

model = build_model(FEAT, CLS)

# Compute class weights inversely proportional to class frequency
total     = len(y_tr)
cw        = {i: total / (CLS * c) for i, c in enumerate(counts)}
print(f"  Class weights: { {k: round(v,3) for k, v in cw.items()} }")
print(f"  (down-weights majority class 0, up-weights minority class 3)")
print()

model.compile(
    optimizer = optimizer,
    loss      = FocalLoss(gamma=2.0, alpha=0.25),
    metrics   = [
        keras.metrics.SparseCategoricalAccuracy(name='acc'),
        PerClassAccuracy(target_class=3, name='minority_acc'),   # track minority
    ],
)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Full callback suite
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Full callback suite")
print("━" * 65)
print()

class GradNormCallback(keras.callbacks.Callback):
    """Log per-epoch gradient norms. Hook: on_train_batch_end."""
    def __init__(self):
        super().__init__()
        self._norms  = []
        self.history = []

    def on_train_batch_end(self, batch, logs=None):
        # self.model.optimizer has .get_gradients() in some Keras versions
        # Here: use trainable_weights norms as a proxy (gradient norm not
        # directly accessible post-step without custom train_step override)
        w_norms = [tf.norm(w).numpy() for w in self.model.trainable_weights[:4]]
        self._norms.append(np.mean(w_norms))

    def on_epoch_end(self, epoch, logs=None):
        if self._norms:
            self.history.append(np.mean(self._norms))
            self._norms.clear()


class LRLoggerCallback(keras.callbacks.Callback):
    """Log current learning rate to History at end of each epoch."""
    def __init__(self):
        super().__init__()
        self.lr_history = []

    def on_epoch_end(self, epoch, logs=None):
        lr_val = float(self.model.optimizer.learning_rate)
        self.lr_history.append(lr_val)
        if logs is not None:
            logs['lr'] = lr_val


with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, 'model_{epoch:02d}_{val_acc:.4f}.keras')
    best_path = os.path.join(tmp, 'best_model.keras')

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath       = best_path,
            monitor        = 'val_acc',
            mode           = 'max',
            save_best_only = True,
            verbose        = 0,
        ),
        keras.callbacks.EarlyStopping(
            monitor              = 'val_loss',
            patience             = 8,
            min_delta            = 5e-4,
            mode                 = 'min',
            restore_best_weights = True,
            verbose              = 0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor   = 'val_loss',
            factor    = 0.5,
            patience  = 4,
            min_lr    = 1e-7,
            verbose   = 0,
        ),
        keras.callbacks.TerminateOnNaN(),
        GradNormCallback(),
        LRLoggerCallback(),
    ]

    t0   = time.perf_counter()
    hist = model.fit(
        X_tr, y_tr,
        validation_data = (X_va, y_va),
        epochs          = MAX_EPOCHS,
        batch_size      = 64,
        class_weight    = cw,
        callbacks       = callbacks,
        verbose         = 0,
    )
    t_fit = time.perf_counter() - t0

    stopped_at  = len(hist.history['loss'])
    best_val_acc = max(hist.history['val_acc'])
    best_min_acc = max(hist.history['val_minority_acc'])

    print(f"  Training: {stopped_at}/{MAX_EPOCHS} epochs in {t_fit:.2f}s")
    print(f"  Best val_acc:       {best_val_acc:.4f}")
    print(f"  Best minority_acc:  {best_min_acc:.4f}  (class 3 per-class accuracy)")
    print()

    # ── History analysis ───────────────────────────────────────────────
    print(f"  Last 5 epochs:")
    print(f"  {'Ep':>4} {'loss':>10} {'val_loss':>10} {'val_acc':>10} {'min_acc':>12}")
    print(f"  {'─'*50}")
    for i in range(max(0, stopped_at-5), stopped_at):
        print(f"  {i+1:>4} "
              f"{hist.history['loss'][i]:>10.4f} "
              f"{hist.history['val_loss'][i]:>10.4f} "
              f"{hist.history['val_acc'][i]:>10.4f} "
              f"{hist.history['val_minority_acc'][i]:>12.4f}")
    print()

    # ── Reload best model ──────────────────────────────────────────────
    if os.path.exists(best_path):
        best_model = keras.models.load_model(
            best_path,
            custom_objects={'FocalLoss': FocalLoss,
                            'PerClassAccuracy': PerClassAccuracy}
        )
        test_scores = best_model.evaluate(X_va, y_va, verbose=0)
        metric_names = ['loss', 'acc', 'minority_acc']
        print(f"  Best model reloaded and evaluated:")
        for name, val in zip(metric_names, test_scores):
            print(f"    {name}: {val:.4f}")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Custom Training Loops — train_step Override, GAN & Distillation": {
        "description": (
            "Advanced training patterns using train_step override. "
            "Override train_step to keep model.fit() automation. "
            "Custom metrics property and @property metrics pattern. "
            "GAN with two optimisers in a single train_step. "
            "Knowledge distillation: soft labels from teacher model. "
            "Multi-task learning: shared backbone with task-specific heads. "
            "Manual GradientTape training loop with @tf.function compilation. "
            "Gradient clipping inside GradientTape. "
            "Higher-order gradients (Jacobian penalty example)."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import tensorflow as tf
import keras

print("=" * 65)
print("  CUSTOM TRAINING LOOPS — train_step OVERRIDE")
print("=" * 65)
print()

tf.random.set_seed(42)
np.random.seed(42)
print(f"  Keras: {keras.__version__} | TF: {tf.__version__}")
print()

N, FEAT, CLS = 2000, 32, 5
X = np.random.randn(N, FEAT).astype('float32')
y = np.random.randint(0, CLS, N).astype('int32')
train_ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(2000).batch(64)
val_ds   = tf.data.Dataset.from_tensor_slices(
    (X[:400], y[:400])).batch(128)
print(f"  Dataset: {N} samples | {FEAT} features | {CLS} classes")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Custom train_step — label smoothing + mixup augmentation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Custom train_step: Mixup augmentation")
print("━" * 65)
print()

class MixupClassifier(keras.Model):
    """
    Implements Mixup (Zhang et al., 2018) training inside train_step.
    Mixup: randomly interpolates pairs of training examples and their labels.
    x_mix = lam * x_a + (1 - lam) * x_b
    y_mix = lam * y_a + (1 - lam) * y_b   (soft labels)
    Improves calibration and generalisation without external data.
    """
    def __init__(self, network, alpha=0.4, n_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.network  = network
        self.alpha    = alpha
        self.n_classes = n_classes
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.acc_tracker  = keras.metrics.SparseCategoricalAccuracy(name='acc')

    @property
    def metrics(self):
        # MUST declare metrics here for Keras to reset them between epochs
        return [self.loss_tracker, self.acc_tracker]

    def train_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]

        # Sample mixup coefficient lambda from Beta(alpha, alpha)
        if self.alpha > 0:
            lam = float(np.random.beta(self.alpha, self.alpha))
        else:
            lam = 1.0

        # Shuffle indices for mixing
        indices = tf.random.shuffle(tf.range(batch_size))
        x_b     = tf.gather(x, indices)
        y_b     = tf.gather(y, indices)

        # Create mixed samples
        x_mix = lam * x + (1.0 - lam) * x_b

        # Convert labels to one-hot for soft mixing
        y_onehot   = tf.one_hot(y,   self.n_classes)
        y_b_onehot = tf.one_hot(y_b, self.n_classes)
        y_mix      = lam * y_onehot + (1.0 - lam) * y_b_onehot   # soft labels

        with tf.GradientTape() as tape:
            logits = self.network(x_mix, training=True)
            # CategoricalCrossentropy accepts soft labels
            loss   = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(y_mix, logits)
            )

        grads = tape.gradient(loss, self.trainable_weights)
        # Gradient clipping inside train_step
        grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, logits)     # acc on original labels
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y   = data
        logits = self.network(x, training=False)
        loss   = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)
        )
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


net = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64,  activation='relu'),
    keras.layers.Dense(CLS),
])

mixup_model = MixupClassifier(net, alpha=0.4, n_classes=CLS)
mixup_model.compile(optimizer=keras.optimizers.AdamW(1e-3))

hist_mx = mixup_model.fit(
    train_ds, validation_data=val_ds,
    epochs=12, verbose=0,
)
print(f"  Mixup classifier trained 12 epochs:")
print(f"    Final train_acc: {hist_mx.history['acc'][-1]:.4f}")
print(f"    Final val_acc:   {hist_mx.history['val_acc'][-1]:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: GAN with train_step override
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — GAN with train_step (two optimisers)")
print("━" * 65)
print()

class VanillaGAN(keras.Model):
    """
    Vanilla GAN using train_step override.
    Two separate GradientTape contexts → two separate optimisers.
    Demonstrates: compile() accepting custom optimisers by keyword.
    """
    def __init__(self, generator, discriminator, latent_dim):
        super().__init__()
        self.generator     = generator
        self.discriminator = discriminator
        self.latent_dim    = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.real_acc       = keras.metrics.BinaryAccuracy(name='real_acc')
        self.fake_acc       = keras.metrics.BinaryAccuracy(name='fake_acc')

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.real_acc, self.fake_acc]

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, real_data):
        batch_size  = tf.shape(real_data)[0]
        noise       = tf.random.normal([batch_size, self.latent_dim])
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # ── Train discriminator (real + fake) ──────────────────────
        with tf.GradientTape() as d_tape:
            fake_data = self.generator(noise, training=True)
            d_real    = self.discriminator(real_data,  training=True)
            d_fake    = self.discriminator(fake_data,  training=True)
            loss_real = keras.losses.binary_crossentropy(real_labels, d_real, from_logits=True)
            loss_fake = keras.losses.binary_crossentropy(fake_labels, d_fake, from_logits=True)
            d_loss    = tf.reduce_mean(loss_real) + tf.reduce_mean(loss_fake)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights))

        # ── Train generator (fool discriminator) ───────────────────
        noise2 = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as g_tape:
            fake_data2 = self.generator(noise2, training=True)
            d_out      = self.discriminator(fake_data2, training=False)
            g_loss     = tf.reduce_mean(
                keras.losses.binary_crossentropy(real_labels, d_out, from_logits=True)
            )
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.real_acc.update_state(real_labels, tf.nn.sigmoid(d_real))
        self.fake_acc.update_state(fake_labels, tf.nn.sigmoid(d_fake))
        return {m.name: m.result() for m in self.metrics}


LATENT_DIM = 16
DATA_DIM   = 32

gen = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(LATENT_DIM,)),
    keras.layers.Dense(DATA_DIM),
])
disc = keras.Sequential([
    keras.layers.Dense(64, activation='leaky_relu', input_shape=(DATA_DIM,)),
    keras.layers.Dense(1),
])

gan = VanillaGAN(gen, disc, LATENT_DIM)
gan.compile(
    d_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
)

# "Real" data: samples from N(2, 1)
real_ds = tf.data.Dataset.from_tensor_slices(
    tf.random.normal([2000, DATA_DIM], mean=2.0)
).batch(64)

hist_gan = gan.fit(real_ds, epochs=15, verbose=0)

g_final = hist_gan.history['g_loss'][-1]
d_final = hist_gan.history['d_loss'][-1]
print(f"  GAN trained 15 epochs:")
print(f"    G loss: {g_final:.4f}  (target ≈ log 2 = {np.log(2):.4f})")
print(f"    D loss: {d_final:.4f}  (target ≈ log 2)")
print(f"    Real acc: {hist_gan.history['real_acc'][-1]:.3f}  "
      f"Fake acc: {hist_gan.history['fake_acc'][-1]:.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Knowledge distillation train_step
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Knowledge distillation (teacher → student)")
print("━" * 65)
print()

# Teacher: large model, trained
teacher_net = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(FEAT,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64,  activation='relu'),
    keras.layers.Dense(CLS),
], name='teacher')
teacher_net.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'],
)
teacher_net.fit(X, y, epochs=10, batch_size=64, verbose=0)
teacher_train_acc = teacher_net.evaluate(X, y, verbose=0)[1]
print(f"  Teacher (256→128→64→5):  train_acc = {teacher_train_acc:.4f}")

# Student: small model, learns from teacher
student_net = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(FEAT,)),
    keras.layers.Dense(CLS),
], name='student')
student_net.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'],
)
student_net.fit(X, y, epochs=10, batch_size=64, verbose=0)
student_base_acc = student_net.evaluate(X, y, verbose=0)[1]
print(f"  Student baseline (32→5): train_acc = {student_base_acc:.4f}")
print()

class KnowledgeDistiller(keras.Model):
    """
    Student learns from:
      - Hard labels (true class) via sparse cross-entropy
      - Soft labels (teacher logits / temperature) via KL divergence
    Combined: loss = alpha * student_loss + (1-alpha) * T^2 * kl_loss
    """
    def __init__(self, student, teacher, temperature=5.0, alpha=0.3):
        super().__init__()
        self.student     = student
        self.teacher     = teacher
        self.temperature = temperature
        self.alpha       = alpha
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.acc_tracker  = keras.metrics.SparseCategoricalAccuracy(name='acc')

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def train_step(self, data):
        x, y = data
        # Teacher produces soft labels (frozen — no tape)
        teacher_logits = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)

            # Hard label loss
            student_loss = keras.losses.sparse_categorical_crossentropy(
                y, student_logits, from_logits=True
            )

            # Soft label loss — KL divergence at temperature T
            soft_teacher  = tf.nn.softmax(teacher_logits / self.temperature)
            soft_student  = tf.nn.log_softmax(student_logits / self.temperature)
            distill_loss  = tf.keras.losses.KLDivergence()(
                soft_teacher, tf.exp(soft_student)
            )

            # Combined loss (T^2 scales gradients back to original magnitude)
            total_loss = (self.alpha * tf.reduce_mean(student_loss)
                        + (1 - self.alpha) * self.temperature**2 * distill_loss)

        grads = tape.gradient(total_loss, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        self.acc_tracker.update_state(y, student_logits)
        return {m.name: m.result() for m in self.metrics}


# Reinitialise student for fair comparison
student_distilled = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(FEAT,)),
    keras.layers.Dense(CLS),
], name='student_distilled')

distiller = KnowledgeDistiller(
    student=student_distilled,
    teacher=teacher_net,
    temperature=5.0,
    alpha=0.3,
)
distiller.compile(optimizer=keras.optimizers.AdamW(1e-3))
hist_kd = distiller.fit(train_ds, epochs=15, verbose=0)

student_kd_acc = student_distilled.evaluate(X, y, verbose=0)[1]
print(f"  Distilled student:       train_acc = {student_kd_acc:.4f}")
print(f"  Improvement over baseline: {(student_kd_acc - student_base_acc)*100:+.2f}%")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Manual GradientTape loop with @tf.function
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Manual GradientTape loop with @tf.function")
print("━" * 65)
print()

manual_model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(FEAT,)),
    keras.layers.Dense(64,  activation='relu'),
    keras.layers.Dense(CLS),
])
manual_opt  = keras.optimizers.AdamW(1e-3)
loss_fn     = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc   = keras.metrics.SparseCategoricalAccuracy(name='acc')

@tf.function   # compile to TF graph — eliminates Python overhead
def train_step_manual(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = manual_model(x_batch, training=True)
        loss   = loss_fn(y_batch, logits)
        loss  += tf.add_n(manual_model.losses) if manual_model.losses else 0.0
    grads = tape.gradient(loss, manual_model.trainable_weights)
    grads, _  = tf.clip_by_global_norm(grads, 1.0)
    manual_opt.apply_gradients(zip(grads, manual_model.trainable_weights))
    train_acc.update_state(y_batch, logits)
    return loss

@tf.function
def val_step_manual(x_batch, y_batch):
    logits = manual_model(x_batch, training=False)
    return loss_fn(y_batch, logits)

print(f"  Manual GradientTape loop (10 epochs):")
print(f"  {'Epoch':>6} {'Train loss':>12} {'Train acc':>12} {'Val loss':>12}")
print(f"  {'─'*46}")

for epoch in range(10):
    train_acc.reset_state()
    epoch_losses = []

    for x_b, y_b in train_ds:
        loss_val = train_step_manual(x_b, y_b)
        epoch_losses.append(float(loss_val))

    # Validation
    val_losses = [float(val_step_manual(x_b, y_b)) for x_b, y_b in val_ds]
    print(f"  {epoch+1:>6} {np.mean(epoch_losses):>12.4f} "
          f"{float(train_acc.result()):>12.4f} {np.mean(val_losses):>12.4f}")

print()
print("  @tf.function: eliminates Python overhead by tracing the step once.")
print("  First call is slow (tracing). Subsequent calls run compiled XLA graph.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Preprocessing Layers, Mixed Precision & Deployment": {
        "description": (
            "Production-quality Keras preprocessing and deployment pipeline. "
            "Normalization layer: adapt() from training data, bake into model. "
            "TextVectorization: vocabulary building and sequence encoding. "
            "Image augmentation layers: only active during training. "
            "Mixed precision: set_global_policy('mixed_float16'). "
            "Model saving: .keras format and SavedModel format. "
            "TFLite conversion: FP32 vs dynamic range vs full INT8. "
            "Serving signature for TF Serving deployment. "
            "Transfer learning: freeze backbone, unfreeze for fine-tuning."
        ),
        "language": "python",
        "code": '''
import numpy as np
import os, tempfile, time
import tensorflow as tf
import keras

print("=" * 65)
print("  PREPROCESSING LAYERS, MIXED PRECISION & DEPLOYMENT")
print("=" * 65)
print()

tf.random.set_seed(42)
np.random.seed(42)
print(f"  Keras: {keras.__version__} | TF: {tf.__version__}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Normalization layer — adapt + bake into model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Normalization layer (adapt + bake into model)")
print("━" * 65)
print()

N_FEAT, CLS = 64, 8
X_raw = np.random.randn(3000, N_FEAT).astype('float32') * 10 + 5   # mean≈5, std≈10
y_raw = np.random.randint(0, CLS, 3000).astype('int32')
X_tr, X_va = X_raw[:2400], X_raw[2400:]
y_tr, y_va = y_raw[:2400], y_raw[2400:]

# Step 1: Create Normalization layer and adapt to TRAINING data
normalizer = keras.layers.Normalization(axis=-1, name='normalizer')
normalizer.adapt(X_tr)   # computes mean and variance from training data

print(f"  Raw data: mean≈{X_tr.mean():.1f}, std≈{X_tr.std():.1f}")
print(f"  Normalizer: adapted mean[0:3] = {normalizer.mean.numpy()[:3].round(3)}")
print(f"  Normalizer: adapted var[0:3]  = {normalizer.variance.numpy()[:3].round(3)}")
print()

# Step 2: Build model with normalizer as FIRST layer
inp = keras.Input(shape=(N_FEAT,), name='raw_features')
x   = normalizer(inp)    # raw input auto-normalised — no external preprocessing!
x   = keras.layers.Dense(128, activation='gelu')(x)
x   = keras.layers.LayerNormalization()(x)
x   = keras.layers.Dropout(0.3)(x)
x   = keras.layers.Dense(64, activation='gelu')(x)
out = keras.layers.Dense(CLS, name='logits')(x)

norm_model = keras.Model(inp, out, name='normalised_model')
norm_model.compile(
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3),
    loss      = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics   = [keras.metrics.SparseCategoricalAccuracy(name='acc')],
)

hist_norm = norm_model.fit(
    X_tr, y_tr,           # raw un-normalised input — model normalises internally
    validation_data=(X_va, y_va),
    epochs=10, batch_size=64, verbose=0,
)
print(f"  Model with baked Normalization layer:")
print(f"    val_acc = {hist_norm.history['val_acc'][-1]:.4f}")
print(f"    At inference: pass RAW data. Model normalises automatically.")
print()

# Verify: raw inference vs manual normalisation should be identical
raw_batch     = tf.constant(X_va[:8])
raw_preds     = norm_model(raw_batch, training=False).numpy()

manual_norm   = (X_va[:8] - normalizer.mean.numpy()) / np.sqrt(normalizer.variance.numpy() + 1e-3)
# Can't directly compare since rest of model also applies; just check normalization layer output
norm_layer_out = normalizer(raw_batch).numpy()
manual_out     = (X_va[:8] - normalizer.mean.numpy()) / np.sqrt(normalizer.variance.numpy() + 1e-3)
max_diff = np.abs(norm_layer_out - manual_out).max()
print(f"  Normalization layer vs manual: max diff = {max_diff:.2e}  ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Image augmentation layers (training only)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Image augmentation layers (training-only)")
print("━" * 65)
print()

# Simulate small image dataset
N_IMG = 800
IMG_H, IMG_W, C_IMG = 32, 32, 3
X_img = np.random.rand(N_IMG, IMG_H, IMG_W, C_IMG).astype('float32')
y_img = np.random.randint(0, 4, N_IMG).astype('int32')

# Data augmentation pipeline — only applied during training
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    keras.layers.RandomContrast(factor=0.2),
], name='augmentation')

# Pattern: apply augmentation conditionally in call()
class AugmentedCNN(keras.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.augment = data_augmentation
        self.conv1   = keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool    = keras.layers.MaxPooling2D()
        self.conv2   = keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.gap     = keras.layers.GlobalAveragePooling2D()
        self.drop    = keras.layers.Dropout(0.3)
        self.head    = keras.layers.Dense(n_classes)

    def call(self, x, training=False):
        if training:
            x = self.augment(x, training=True)   # augment only during training!
        x = self.conv1(x, training=training)
        x = self.pool(x)
        x = self.conv2(x, training=training)
        x = self.gap(x)
        x = self.drop(x, training=training)
        return self.head(x)

aug_model = AugmentedCNN(n_classes=4)
aug_model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'],
)
hist_aug = aug_model.fit(
    X_img, y_img,
    validation_split=0.2,
    epochs=5, batch_size=32, verbose=0,
)
print(f"  AugmentedCNN trained 5 epochs:")
print(f"    val_acc = {hist_aug.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print(f"    Augmentation: RandomFlip + Rotation + Zoom + Contrast")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Mixed precision — set_global_policy
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Mixed precision training")
print("━" * 65)
print()

def build_bench_model(in_feat, n_cls):
    inp = keras.Input(shape=(in_feat,))
    x   = keras.layers.Dense(256, activation='gelu')(inp)
    x   = keras.layers.Dense(256, activation='gelu')(x)
    x   = keras.layers.Dense(128, activation='gelu')(x)
    out = keras.layers.Dense(n_cls, dtype='float32')(x)  # final layer stays float32
    return keras.Model(inp, out)

N_BENCH, F_BENCH = 3000, 64
X_b = np.random.randn(N_BENCH, F_BENCH).astype('float32')
y_b = np.random.randint(0, 8, N_BENCH)

results_prec = {}

for policy_name in ['float32', 'mixed_float16']:
    keras.mixed_precision.set_global_policy(policy_name)
    m = build_bench_model(F_BENCH, 8)
    m.compile(
        optimizer=keras.optimizers.AdamW(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'],
    )
    t0   = time.perf_counter()
    hist = m.fit(X_b, y_b, epochs=5, batch_size=64, verbose=0)
    t_s  = time.perf_counter() - t0
    final_acc = hist.history['sparse_categorical_accuracy'][-1]
    results_prec[policy_name] = {'time_s': t_s, 'acc': final_acc}

# Reset to float32
keras.mixed_precision.set_global_policy('float32')

print(f"  Mixed precision benchmark (5 epochs, {N_BENCH} samples, {F_BENCH} features):")
print(f"  {'Policy':<20} {'Time (s)':>12} {'Final acc':>12}")
print(f"  {'─'*46}")
for policy, r in results_prec.items():
    print(f"  {policy:<20} {r['time_s']:>12.2f} {r['acc']:>12.4f}")
print()
print("  Note: float16 speedup is primarily on CUDA GPUs with Tensor Cores.")
print("  On CPU, float16 may be slower (no dedicated hardware).")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Model saving — .keras format and weights
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Model saving: .keras format and weights")
print("━" * 65)
print()

deploy_model = build_bench_model(F_BENCH, 8)
deploy_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
deploy_model.fit(X_b, y_b, epochs=5, batch_size=64, verbose=0)

with tempfile.TemporaryDirectory() as tmp:
    keras_path  = os.path.join(tmp, 'model.keras')
    h5_path     = os.path.join(tmp, 'model.h5')
    weights_path = os.path.join(tmp, 'weights.weights.h5')
    sm_path     = os.path.join(tmp, 'saved_model')

    # ── .keras format (recommended) ────────────────────────────────
    deploy_model.save(keras_path)
    keras_size = os.path.getsize(keras_path) / 1024
    loaded_keras = keras.models.load_model(keras_path)

    # ── H5 format (legacy) ─────────────────────────────────────────
    deploy_model.save(h5_path)
    h5_size = os.path.getsize(h5_path) / 1024

    # ── Weights only ───────────────────────────────────────────────
    deploy_model.save_weights(weights_path)
    w_size = os.path.getsize(weights_path) / 1024

    # ── SavedModel (TF Serving compatible) ─────────────────────────
    deploy_model.export(sm_path)
    sm_size = sum(os.path.getsize(os.path.join(r, f))
                  for r, _, fs in os.walk(sm_path) for f in fs) / 1024

    print(f"  Saving format comparison:")
    print(f"  {'Format':<20} {'Size (KB)':>12} {'Notes'}")
    print(f"  {'─'*55}")
    print(f"  {'.keras':<20} {keras_size:>12.1f}  Full model, backend-agnostic")
    print(f"  {'.h5 (legacy)':<20} {h5_size:>12.1f}  Full model, TF-backend only")
    print(f"  {'weights only':<20} {w_size:>12.1f}  Weights only, no architecture")
    print(f"  {'SavedModel dir':<20} {sm_size:>12.1f}  TF Serving / inference only")
    print()

    # ── Verify .keras round-trip ───────────────────────────────────
    x_check = tf.constant(X_b[:8])
    out_orig   = deploy_model(x_check, training=False).numpy()
    out_loaded = loaded_keras(x_check, training=False).numpy()
    diff = np.abs(out_orig - out_loaded).max()
    print(f"  .keras round-trip max diff: {diff:.2e}  ✅")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: TFLite conversion comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — TFLite conversion: FP32 vs dynamic range vs INT8")
print("━" * 65)
print()

# Train a small model for TFLite conversion
tflite_model_src = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(5),
], name='tflite_source')
tflite_model_src.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])

X_tfl = np.random.randn(2000, 32).astype('float32')
y_tfl = np.random.randint(0, 5, 2000)
tflite_model_src.fit(X_tfl, y_tfl, epochs=3, batch_size=64, verbose=0)
ref_preds = tflite_model_src.predict(X_tfl[:100], verbose=0).argmax(1)

def representative_dataset():
    for i in range(0, 200, 10):
        yield [X_tfl[i:i+10]]

with tempfile.TemporaryDirectory() as tmp:
    sm_dir = os.path.join(tmp, 'sm')
    tflite_model_src.export(sm_dir)

    tflite_results = {}

    for mode, label in [
        ('fp32',    'FP32 (no quant)'),
        ('dynrng',  'Dynamic range (weights INT8)'),
        ('full_int','Full INT8 (acts + weights)'),
    ]:
        converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
        if mode == 'dynrng':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif mode == 'full_int':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type  = tf.float32
            converter.inference_output_type = tf.float32

        try:
            tflite_flat = converter.convert()
            tfl_path    = os.path.join(tmp, f'{mode}.tflite')
            with open(tfl_path, 'wb') as f:
                f.write(tflite_flat)
            size_kb = os.path.getsize(tfl_path) / 1024

            interp = tf.lite.Interpreter(model_content=tflite_flat)
            interp.allocate_tensors()
            in_idx  = interp.get_input_details()[0]['index']
            out_idx = interp.get_output_details()[0]['index']

            preds   = []
            t0      = time.perf_counter()
            X_test_np = X_tfl[:100].astype('float32')
            for i in range(100):
                interp.set_tensor(in_idx, X_test_np[i:i+1])
                interp.invoke()
                preds.append(interp.get_tensor(out_idx).argmax())
            t_ms = (time.perf_counter() - t0) / 100 * 1000

            agreement = np.mean(np.array(preds) == ref_preds) * 100
            tflite_results[label] = {'kb': size_kb, 'ms': t_ms, 'agree': agreement}
        except Exception as e:
            tflite_results[label] = {'kb': 0, 'ms': 0, 'agree': 0, 'err': str(e)[:40]}

    print(f"  {'Mode':<30} {'Size KB':>10} {'ms/sample':>11} {'Agreement':>11}")
    print(f"  {'─'*65}")
    for label, r in tflite_results.items():
        if 'err' in r:
            print(f"  {label:<30} {'ERROR':>10}  {r['err']}")
        else:
            print(f"  {label:<30} {r['kb']:>10.1f} {r['ms']:>11.4f} {r['agree']:>10.1f}%")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Transfer learning — freeze / unfreeze backbone
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Transfer learning: freeze backbone, then fine-tune")
print("━" * 65)
print()

# Simulated pre-trained backbone
backbone = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(32,)),
    keras.layers.Dense(64,  activation='relu'),
    keras.layers.Dense(32,  activation='relu'),
], name='backbone')
# Simulate pre-training: already trained
backbone.compile('adam', 'mse')
backbone.fit(X_tfl, np.random.randn(2000, 32).astype('float32'),
              epochs=3, verbose=0)

# Phase 1: freeze backbone, train only the head
backbone.trainable = False   # freeze ALL backbone layers

inp        = keras.Input(shape=(32,))
features   = backbone(inp)
head_out   = keras.layers.Dense(64, activation='relu')(features)
head_out   = keras.layers.Dense(5)(head_out)
transfer_m = keras.Model(inp, head_out, name='transfer_model')

trainable_before = sum(np.prod(w.shape) for w in transfer_m.trainable_weights)
frozen_before    = sum(np.prod(w.shape) for w in transfer_m.non_trainable_weights)
print(f"  Phase 1 (backbone FROZEN):")
print(f"    Trainable params:     {trainable_before:,}")
print(f"    Non-trainable params: {frozen_before:,}")

transfer_m.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'],
)
h1 = transfer_m.fit(X_tfl, y_tfl, epochs=5, validation_split=0.2,
                     batch_size=64, verbose=0)
print(f"    Phase 1 val_acc: {h1.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print()

# Phase 2: unfreeze backbone, fine-tune everything at lower lr
backbone.trainable = True    # unfreeze ALL backbone layers
transfer_m.compile(
    optimizer=keras.optimizers.Adam(1e-5),   # 100× smaller lr for fine-tuning
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'],
)
trainable_after = sum(np.prod(w.shape) for w in transfer_m.trainable_weights)

print(f"  Phase 2 (backbone UNFROZEN, fine-tuning at lr=1e-5):")
print(f"    Trainable params:     {trainable_after:,}")

h2 = transfer_m.fit(X_tfl, y_tfl, epochs=5, validation_split=0.2,
                     batch_size=64, verbose=0)
print(f"    Phase 2 val_acc: {h2.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print()
print("  Layer-level freezing (fine-tune only top N layers):")
print("    for layer in backbone.layers[:-2]:  # freeze all but last 2")
print("        layer.trainable = False")
print("    backbone.layers[-1].trainable = True   # unfreeze last layer only")
print()
print("  DEPLOYMENT DECISION GUIDE:")
print("  ┌─────────────────────────────────────────────────────────────────┐")
print("  │ Target                  │ Format           │ Tools              │")
print("  ├─────────────────────────────────────────────────────────────────┤")
print("  │ Checkpoint / retrain    │ .keras           │ keras.save/load    │")
print("  │ Python inference        │ .keras / .h5     │ keras.load_model   │")
print("  │ TF Serving (Docker)     │ SavedModel       │ model.export()     │")
print("  │ Android / iOS           │ .tflite          │ TFLiteConverter    │")
print("  │ Microcontroller         │ .tflite (INT8)   │ TF Micro runtime   │")
print("  │ Browser (JS)            │ TFJS format      │ tensorflowjs_conv  │")
print("  │ NVIDIA GPU (prod)       │ ONNX → TensorRT  │ tf2onnx + trt      │")
print("  └─────────────────────────────────────────────────────────────────┘")
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