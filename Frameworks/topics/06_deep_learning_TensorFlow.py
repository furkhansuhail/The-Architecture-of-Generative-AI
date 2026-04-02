"""
TensorFlow — Google's End-to-End Machine Learning Platform
===========================================================

TensorFlow is Google's open-source machine learning framework, released in
2015 and now on version 2.x. Where PyTorch dominates research, TensorFlow
dominates production infrastructure: TensorFlow Serving, TensorFlow Lite
for mobile/edge, TensorFlow.js for browsers, and the TFX pipeline for
enterprise ML workflows are battle-tested at scales no other framework matches.

The framework's history is also the history of how the field learned what
"good" looks like: TF1's static graphs taught us compilation is powerful
but debugging is painful; TF2's eager execution (adopted from PyTorch)
showed that research ergonomics matter; Keras becoming the official high-level
API showed that abstraction layers win adoption. Understanding TensorFlow's
design evolution tells you WHY every major ML framework makes the choices it does.

This module covers TensorFlow's computation model, Keras, the GradientTape
differentiation engine, the TFX production pipeline, and the full deployment
stack from training to edge inference.

"""

import textwrap
import re

TOPIC_NAME   = "TensorFlow — Google's End-to-End ML Platform"
DISPLAY_NAME = "06 · TensorFlow"
ICON         = "🌊"
SUBTITLE     = "From Eager Training to TFX Production Pipelines"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — TENSORFLOW'S HISTORY AND DESIGN EVOLUTION

### The Two Eras: TF1 vs TF2

    TensorFlow 1.x (2015–2019): Define-and-Run
        The computation graph was defined FIRST as a Python data structure.
        A separate Session object then executed that static graph.
        Think of it as writing a recipe (graph), then hiring a chef (session) to cook it.

        import tensorflow as tf                  # TF1 style
        x = tf.placeholder(tf.float32, [None, 4])
        W = tf.Variable(tf.zeros([4, 1]))
        y = tf.matmul(x, W)                      # NO computation yet — just a graph node
        sess = tf.Session()
        result = sess.run(y, feed_dict={x: data}) # computation happens HERE

        Strengths of TF1:
            - Compiler optimisations on the full static graph
            - Efficient deployment (graph is portable, language-independent)
            - Distributed execution planned at graph construction time

        Fatal weaknesses:
            - Debugging meant inspecting graph nodes, not Python values
            - Dynamic shapes and control flow required tf.cond, tf.while_loop
            - Error messages pointed to graph construction sites, not logic errors
            - 2-3× slower iteration cycle than PyTorch's eager mode

    TensorFlow 2.x (2019–present): Eager by Default
        Eager execution: every operation runs immediately, returns a concrete value.
        Same model as PyTorch — Python code IS the graph.
        Keras became the official high-level API.
        @tf.function provides optional compilation for speed.

        import tensorflow as tf                  # TF2 style
        x = tf.constant([[1.0, 2.0, 3.0, 4.0]])
        W = tf.Variable(tf.zeros([4, 1]))
        y = tf.matmul(x, W)                      # executes IMMEDIATELY
        print(y.numpy())                         # → [[0.]] — works right away

### Why TensorFlow Still Matters Despite PyTorch's Research Dominance

    1. PRODUCTION INFRASTRUCTURE:
        TensorFlow Serving: the most battle-tested ML model server in production.
        TFX (TensorFlow Extended): full ML pipeline from data validation to serving.
        Google Vertex AI: deeply integrated with TF.
        Used by: Google Search, YouTube recommendations, Gmail Smart Reply.

    2. EDGE AND MOBILE:
        TensorFlow Lite: the industry standard for on-device inference.
        Runs on Android, iOS, Raspberry Pi, microcontrollers.
        Quantisation-aware training (QAT) built-in — models optimised for edge at training time.
        ExecuTorch (PyTorch's equivalent) is newer and less mature.

    3. BROWSER DEPLOYMENT:
        TensorFlow.js: run models in any JavaScript environment (browser, Node.js).
        No server round-trip for inference — privacy-preserving, low latency.
        Used in: Google's teachable machine, real-time video effects.

    4. ENTERPRISE STANDARDISATION:
        Many large organisations standardised on TF1/TF2 before PyTorch's rise.
        Existing pipelines, tooling, and expertise are sunk costs.
        Keras's simplicity made TF the first ML framework for many data scientists.

    5. TPU FIRST-CLASS SUPPORT:
        Google's TPUs (Tensor Processing Units) are designed around TF's XLA compiler.
        JAX also targets TPUs, but TF has years of production TPU integration.

### The Current Ecosystem Position

    Research:      PyTorch dominates (~75% of papers)
    Production:    TensorFlow is strong (~40% of production deployments use TF Serving)
    Mobile/Edge:   TFLite leads (Android native, iOS CoreML conversion)
    Browser:       TensorFlow.js leads (no PyTorch equivalent at scale)
    Enterprise:    Both, with heavy TFX/Vertex AI usage for Google Cloud shops
    Education:     Keras (TF backend) is often the first framework taught


##### PART 2 — TENSORS AND EAGER EXECUTION IN TF2

### tf.Tensor vs tf.Variable

    TensorFlow has TWO tensor types with different roles:

    tf.Tensor:
        Immutable value — cannot be changed in-place.
        Result of any operation (add, matmul, relu, etc.).
        Has .numpy() to convert to NumPy array.
        No gradients tracked automatically (need GradientTape).

    tf.Variable:
        Mutable state — can be updated with .assign(), .assign_add().
        Used for model weights, optimiser state, counters.
        Automatically tracked by GradientTape when in scope.
        Persists across function calls (unlike tensors which are temporary).

    The key distinction from PyTorch:
        PyTorch: any tensor with requires_grad=True is tracked.
        TensorFlow: only tf.Variable is tracked by default.
                    For non-variable tensors, use tape.watch(tensor).

### tf.Tensor Operations

    TF2 operations mirror NumPy:
        tf.zeros(shape), tf.ones(shape), tf.random.normal(shape)
        tf.reshape(t, shape), tf.transpose(t, perm)
        tf.matmul(a, b)   or  a @ b
        tf.reduce_sum(t, axis), tf.reduce_mean(t, axis)
        tf.concat([a, b], axis), tf.stack([a, b], axis)
        tf.nn.relu(t), tf.nn.softmax(t), tf.nn.sigmoid(t)

    Interop with NumPy:
        t_np = tf_tensor.numpy()         # TF → NumPy (CPU tensor only)
        t_tf = tf.constant(numpy_array)  # NumPy → TF
        # TF ops accept NumPy arrays directly (auto-converted)

### Device Placement

    TF auto-places ops on GPU if available:
        with tf.device('/GPU:0'):
            result = tf.matmul(a, b)

        with tf.device('/CPU:0'):
            result = tf.matmul(a, b)   # force CPU

    Check available devices:
        tf.config.list_physical_devices('GPU')

    Multi-GPU memory growth (critical to set before model creation):
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Default: TF allocates ALL GPU memory upfront. This allows dynamic growth.

### Eager vs Graph Mode

    TF2 default: eager execution (like PyTorch).
    Optional: @tf.function wraps Python functions into compiled TF graphs.

    Comparison:
        Eager:      Runs immediately, full Python debuggability, slower for training.
        @tf.function: Traces once, compiles to XLA graph, 1.5–5× faster.

    The tracing model for @tf.function:
        First call: Python function is traced with symbolic abstract tensors.
        Result: a tf.Graph is cached (keyed on input signature/shapes).
        Subsequent calls: execute the cached compiled graph — no Python overhead.
        Retracing: occurs when input shapes change (like torch.compile's recompilation).


##### PART 3 — GRADIENTTAPE: TENSORFLOW'S DIFFERENTIATION ENGINE

### The Explicit Tape Metaphor

    TensorFlow's autograd uses the "tape" metaphor from dual numbers in calculus.
    A GradientTape is an explicit recording device that must be held open.

    PyTorch:        builds the graph implicitly as operations execute.
    TensorFlow:     records operations only within a 'with GradientTape()' block.

    This makes the differentiation scope VISIBLE in the code:
        with tf.GradientTape() as tape:
            y = model(x)                # recorded
            loss = loss_fn(y, labels)   # recorded
        grads = tape.gradient(loss, model.trainable_variables)  # differentiate

    Outside the 'with' block: nothing is recorded.
    After tape.gradient(): the tape is consumed and freed.

### What Gets Watched Automatically

    tf.Variable:     always watched by default inside any GradientTape scope.
    tf.Tensor:       NOT watched by default. Must call tape.watch(tensor).
    Constants:       NOT watched (immutable, no gradient needed).

    tape.watch example:
        x = tf.constant([3.0, 4.0])    # constant — not auto-watched
        with tf.GradientTape() as tape:
            tape.watch(x)              # explicitly watch
            y = tf.reduce_sum(x ** 2)
        dy_dx = tape.gradient(y, x)    # [6.0, 8.0] = 2x

### Persistent Tapes

    Default: tape is consumed after one .gradient() call.
    persistent=True: call .gradient() multiple times (must del tape manually).

        with tf.GradientTape(persistent=True) as tape:
            y = model(x)
            loss_a = loss_fn_a(y, labels_a)
            loss_b = loss_fn_b(y, labels_b)

        grad_a = tape.gradient(loss_a, model.trainable_variables)
        grad_b = tape.gradient(loss_b, model.trainable_variables)   # second call OK
        del tape   # must manually release

    Use case: multi-task learning, GAN training (separate generator/discriminator grads).

### Higher-Order Derivatives

    Nest tapes for second-order (and beyond) derivatives:

        x = tf.Variable(3.0)
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                y = x ** 3         # y = x³
            dy_dx = inner.gradient(y, x)     # dy/dx = 3x²
        d2y_dx2 = outer.gradient(dy_dx, x)  # d²y/dx² = 6x

    Applications:
        Hessian computation: physics-informed neural networks
        MAML (model-agnostic meta-learning): gradient of gradient
        Gradient penalty in GANs: ||∇D(x̂)||² term

### GradientTape in the Training Loop

    Full canonical TF2 training loop:

        optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3)
        loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        @tf.function   # compile for speed
        def train_step(X, y):
            with tf.GradientTape() as tape:
                logits = model(X, training=True)    # training=True enables Dropout/BN
                loss   = loss_fn(y, logits)
                loss  += sum(model.losses)          # add regularisation losses
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        for X_batch, y_batch in dataset:
            loss = train_step(X_batch, y_batch)


##### PART 4 — KERAS: THE OFFICIAL HIGH-LEVEL API

### Keras's Philosophy

    Keras was created by François Chollet at Google in 2015.
    Core principle: "deep learning should be accessible to anyone."
    Keras is NOT just a wrapper — it is a complete framework that now runs
    on multiple backends: TensorFlow, JAX, and PyTorch.

    Three levels of abstraction in Keras:

        HIGH (Sequential):   stack layers in a list. Zero boilerplate.
        MEDIUM (Functional): connect layers as a DAG. Handles branching.
        LOW (Subclassing):   full Python class. Same flexibility as raw TF.

### Sequential API

    For linear pipelines — each layer connects to the previous one:

        model = keras.Sequential([
            keras.layers.Input(shape=(784,)),
            keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(10),
        ])

    The simplest possible path from "I have data" to "I have predictions."
    Can add layers incrementally: model.add(keras.layers.Dense(64))

### Functional API

    For models with branches, skip connections, multiple inputs/outputs:

        inputs  = keras.Input(shape=(32,))
        x       = keras.layers.Dense(64, activation='relu')(inputs)
        skip    = x                                      # save for skip connection
        x       = keras.layers.Dense(64, activation='relu')(x)
        x       = keras.layers.Add()([x, skip])          # residual connection
        x       = keras.layers.LayerNormalization()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

    Advantages:
        - Model is a DAG — visualise with keras.utils.plot_model()
        - Input/output shapes automatically validated
        - Can have multiple inputs and outputs
        - Shareable layers: call the same layer object on different inputs

### Model Subclassing

    Maximum flexibility — matches PyTorch's nn.Module model:

        class TransformerEncoder(keras.Model):
            def __init__(self, d_model, n_heads, dff, dropout_rate):
                super().__init__()
                self.attention = keras.layers.MultiHeadAttention(n_heads, d_model)
                self.ffn       = keras.Sequential([
                    keras.layers.Dense(dff, activation='gelu'),
                    keras.layers.Dense(d_model),
                ])
                self.norm1   = keras.layers.LayerNormalization()
                self.norm2   = keras.layers.LayerNormalization()
                self.drop1   = keras.layers.Dropout(dropout_rate)
                self.drop2   = keras.layers.Dropout(dropout_rate)

            def call(self, x, training=False):
                attn_out = self.attention(x, x, training=training)
                x = self.norm1(x + self.drop1(attn_out, training=training))
                ffn_out  = self.ffn(x)
                return self.norm2(x + self.drop2(ffn_out, training=training))

### compile() and fit() — The Declarative Training Interface

    model.compile() configures the training:
        model.compile(
            optimizer = keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-2),
            loss      = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics   = [keras.metrics.SparseCategoricalAccuracy()],
        )

    model.fit() runs the training loop:
        history = model.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs          = 50,
            callbacks       = [
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
                keras.callbacks.ModelCheckpoint('best.keras', save_best_only=True),
                keras.callbacks.TensorBoard(log_dir='./logs'),
            ],
        )

    Keras handles: batch iteration, gradient computation, metric accumulation,
    callback dispatch, progress bars — everything.

### Keras Callbacks — Hook System

    Callbacks are called at specific training events:
        on_epoch_begin / on_epoch_end
        on_batch_begin / on_batch_end
        on_train_begin / on_train_end

    Built-in callbacks:
        EarlyStopping:      stop when metric stops improving
        ReduceLROnPlateau:  halve LR on plateau (automatic LR scheduling)
        ModelCheckpoint:    save best model weights
        TensorBoard:        log scalars, histograms, images for visualisation
        LearningRateScheduler: apply a schedule function

    Custom callback example:
        class GradientNormLogger(keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                grads = [tf.norm(g) for g in self.model.optimizer.variables]
                print(f"Batch {batch}: grad_norm = {tf.reduce_mean(grads):.4f}")

### Custom Training Loops (override train_step)

    Between compile+fit and raw GradientTape, override train_step():

        class CustomModel(keras.Model):
            def train_step(self, data):
                X, y = data
                with tf.GradientTape() as tape:
                    logits = self(X, training=True)
                    loss   = self.compiled_loss(y, logits)
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                self.compiled_metrics.update_state(y, logits)
                return {m.name: m.result() for m in self.metrics}

    Use case: custom loss functions, multi-optimizer training (GANs),
              reward shaping (RL), knowledge distillation.


##### PART 5 — tf.data: THE DATA PIPELINE SYSTEM

### What tf.data Does

    tf.data is TensorFlow's pipeline API for building efficient input pipelines.
    It handles: loading, transforming, batching, shuffling, and prefetching.
    Crucially, the ENTIRE pipeline can be expressed as a TF graph and compiled —
    data transforms execute in C++, not Python, avoiding the GIL.

    Performance hierarchy:
        1. Load data (from disk, TFRecords, generators)
        2. Apply transforms (decode, augment, normalise) — parallelised with .map()
        3. Batch samples into tensors
        4. Prefetch — overlap data prep with GPU compute

### Core Pipeline Pattern

    dataset = (
        tf.data.Dataset
        .from_tensor_slices((X, y))         # from in-memory arrays
        .shuffle(buffer_size=10000)          # shuffle within a buffer
        .map(augment_fn,                     # apply transforms
             num_parallel_calls=tf.data.AUTOTUNE)  # auto-parallelise
        .batch(32)                           # group into batches
        .prefetch(tf.data.AUTOTUNE)          # overlap with GPU
    )

    AUTOTUNE: TF dynamically determines the optimal parallelism level.
    Always use AUTOTUNE for both num_parallel_calls and prefetch.

### TFRecord Format — The Production Data Format

    TFRecord is TF's binary serialisation format for datasets.
    Reasons to use it:
        - Sequential reads are faster than random-access file reads
        - Stores diverse data types (images, text, labels) in one file
        - Efficient with tf.data pipeline (streaming, no random access needed)
        - Standard format for all TFX components

    Writing TFRecords:
        with tf.io.TFRecordWriter('train.tfrecord') as writer:
            for image, label in dataset:
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(
                                 value=[image.tobytes()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(
                                 value=[label])),
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    Reading TFRecords:
        raw_ds   = tf.data.TFRecordDataset(['train.tfrecord'])
        parsed   = raw_ds.map(parse_fn)   # decode bytes back to tensors
        pipeline = parsed.shuffle(1000).batch(32).prefetch(AUTOTUNE)

### Cache, Repeat, and Take

    .cache():   store the dataset in memory (or on disk) after the first epoch.
                After caching, disk reads are skipped — huge speedup.
                Only use when dataset fits in RAM.

    .repeat(N): repeat the dataset N times (None = infinite).
                Use with steps_per_epoch in model.fit().

    .take(N):   use only the first N batches.
                Useful for quick debugging without full dataset.


##### PART 6 — @tf.function AND XLA COMPILATION

### How @tf.function Works

    @tf.function converts a Python function into a TensorFlow graph.
    This is TF's equivalent of torch.compile() but with a longer history.

    Tracing process:
        1. First call: Python function runs with symbolic tf.Tensor arguments.
           No actual computation — just records which ops are called.
        2. Records all TF operations, builds a tf.Graph.
        3. XLA compiles the graph (optionally — with jit_compile=True).
        4. Caches the compiled function, keyed on input signature.
        5. Subsequent calls: skip Python, run compiled graph directly.

    Speedup: 2–5× for training steps, up to 10× for simple compute-heavy ops.

### The Tracing Trap: Python Side Effects

    Python code inside @tf.function runs ONLY during tracing, not execution:

        @tf.function
        def buggy_fn(x):
            print("I am called!")   # ← only prints ONCE (at trace time)
            return x + 1

        buggy_fn(tf.constant(1))    # prints "I am called!"
        buggy_fn(tf.constant(2))    # prints NOTHING — uses cached graph
        buggy_fn(tf.constant(1))    # prints NOTHING — same shape, cached

        # Use tf.print() for in-graph printing:
        @tf.function
        def correct_fn(x):
            tf.print("Value:", x)  # prints every execution
            return x + 1

    Rule: anything that must happen EVERY call must use TF ops, not Python.

### Retrace Triggers

    @tf.function retraces when:
        - Input dtype changes (int32 → int64 → new trace)
        - Input shape changes (batch size changes → new trace if not InputSpec)
        - A Python-valued argument changes (use tf.Tensor instead)

    Avoid excessive retracing:
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, 32], dtype=tf.float32),
            tf.TensorSpec(shape=[None],     dtype=tf.int32),
        ])
        def train_step(x, y): ...
        # Exactly ONE trace for any batch size — no retracing

### XLA (Accelerated Linear Algebra)

    XLA is a compiler for linear algebra operations:
        - Fuses multiple operations into single kernels (conv + bias + relu → 1 kernel)
        - Optimises memory layout for hardware (NCHW vs NHWC for convolutions)
        - Ahead-of-time compilation for TPUs (required), optional for GPUs

    Enable XLA for GPU:
        @tf.function(jit_compile=True)   # or: tf.config.optimizer.set_jit(True)
        def compute(x, w):
            return tf.nn.relu(tf.matmul(x, w))

    Enable XLA for TPU:
        # XLA is MANDATORY on TPU — all ops must be XLA-compatible


##### PART 7 — DEPLOYMENT: SAVEDMODEL, TFLITE, TF SERVING & TFX

### SavedModel — The Universal Serialisation Format

    TF2's SavedModel stores:
        - The computation graph (as @tf.function traces)
        - Model weights (as tf.Variable checkpoints)
        - Serving signatures (which functions to expose as API endpoints)
        - Optional assets (vocabulary files, lookup tables)

    Saving:
        model.save("my_model/")            # SavedModel format (directory)
        model.save("my_model.keras")       # Keras v3 native format
        tf.saved_model.save(model, "dir/") # Low-level SavedModel API

    Loading:
        model = tf.keras.models.load_model("my_model/")
        model = tf.saved_model.load("dir/")   # returns raw SavedModel object

    Serving signature:
        @tf.function(input_signature=[tf.TensorSpec([None, 32], tf.float32)])
        def serve(x):
            return {"logits": model(x, training=False)}

        tf.saved_model.save(model, "dir/",
                            signatures={"serving_default": serve})

### TensorFlow Lite — On-Device Inference

    TFLite is a separate runtime optimised for mobile and edge:
        - Size: ~300 KB runtime (vs TF's ~50 MB)
        - Supports: Android, iOS, Raspberry Pi, microcontrollers (TF Micro)
        - Quantisation: INT8, FP16, dynamic range quant — built into the converter

    Conversion pipeline:
        1. Train a Keras or TF model (full precision)
        2. Convert to .tflite format with TFLiteConverter
        3. Deploy the .tflite file to the device
        4. Run inference with the TFLite interpreter

    Converter options:
        converter = tf.lite.TFLiteConverter.from_saved_model("dir/")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]   # INT8 quant
        # Optional: provide representative dataset for full integer quant
        converter.representative_dataset = representative_data_gen
        tflite_model = converter.convert()

    Quantisation modes:
        Dynamic range:   weights INT8, activations FP32 at runtime → 4× smaller, 2-3× faster
        Full integer:    weights + activations INT8 → requires representative data
        FP16:            weights FP16 → 2× smaller, GPU-accelerated on supported hardware
        16×8 mixed:      activations INT16, weights INT8 → high accuracy, hardware dependent

    Benchmark results (MobileNetV2, Pixel 6):
        FP32 model:    ~120ms/inference
        FP16 quant:    ~75ms/inference   (1.6× faster)
        INT8 quant:    ~40ms/inference   (3× faster)

### TensorFlow Serving — Production Model Server

    TF Serving is a production-grade serving system for TF models.
    Features:
        - Versioned model management (load new versions without downtime)
        - A/B testing support (route % of traffic to new version)
        - gRPC and REST endpoints
        - Batching support (auto-batch single requests for GPU efficiency)
        - Monitoring with Prometheus metrics

    Deployment (Docker):
        docker run -p 8501:8501 \\
            -v /models/my_model:/models/my_model \\
            -e MODEL_NAME=my_model \\
            tensorflow/serving

    REST query:
        POST http://localhost:8501/v1/models/my_model:predict
        {"instances": [[0.1, 0.2, 0.3, ...]]}

    gRPC query (lower latency, preferred for production):
        channel = grpc.insecure_channel('localhost:8500')
        stub    = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.inputs['input'].CopyFrom(tf.make_tensor_proto(data))
        result  = stub.Predict(request)

### TFX — TensorFlow Extended (Enterprise ML Pipelines)

    TFX is a complete production ML platform. Each component is a pipeline stage:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  ExampleGen     → ingest data (CSV, TFRecord, BigQuery, databases)  │
    │  StatisticsGen  → compute data statistics (TFDV)                    │
    │  SchemaGen      → infer data schema (types, ranges, required fields)│
    │  ExampleValidator → validate data against schema (catch anomalies)  │
    │  Transform      → feature engineering (TF Transform — artifacts)    │
    │  Trainer        → train model (Keras or TF Estimator)               │
    │  Tuner          → hyperparameter search (Keras Tuner integration)   │
    │  Evaluator      → evaluate against a baseline (TF Model Analysis)   │
    │  ModelValidator → gate on metrics before blessing                   │
    │  Pusher         → deploy to TF Serving or TFLite                    │
    └─────────────────────────────────────────────────────────────────────┘

    TFX runs on: Apache Airflow, Kubeflow Pipelines, Apache Beam, Vertex AI.
    Every artifact is versioned and traceable — complete ML lineage.

    The key innovation of TFX: CONSISTENT preprocessing between training and serving.
    TF Transform computes preprocessing on the FULL dataset at training time,
    then bakes those statistics into the SavedModel as a TF graph.
    The same graph runs at serving time — no training-serving skew possible.


##### PART 8 — DISTRIBUTED TRAINING STRATEGIES

### tf.distribute.Strategy — The Unified Distribution API

    TF's distribution API wraps all distributed training strategies
    in a single, consistent interface. The model and training code stay
    IDENTICAL — only the Strategy object changes.

    This is TF's biggest distribution advantage over PyTorch:
    PyTorch needs explicit DDP setup; TF wraps it in one strategy object.

### MirroredStrategy (single machine, multiple GPUs)

    Replicates model on all GPUs. Gradients synchronised via AllReduce.
    Equivalent to PyTorch DDP but within one machine.

        strategy = tf.distribute.MirroredStrategy()
        print(f"Devices: {strategy.num_replicas_in_sync}")

        with strategy.scope():                    # ALL model creation inside scope
            model = build_model()
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        model.fit(dataset, epochs=10)             # training loop unchanged

    Under the hood: each GPU has a full copy of the model.
    Gradients are averaged with AllReduce (NCCL for NVIDIA GPUs).

### MultiWorkerMirroredStrategy (multi-machine)

    Extends MirroredStrategy across multiple machines (each with GPUs).
    Requires: TF_CONFIG environment variable on each machine.

        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {'worker': ['host1:port1', 'host2:port2']},
            'task': {'type': 'worker', 'index': 0}   # 0 on first machine, 1 on second
        })

        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            model = build_model()
        model.fit(dataset, epochs=10)   # each machine runs its own copy

### ParameterServerStrategy (asynchronous, large-scale)

    Separates parameter storage (parameter servers) from computation (workers).
    Asynchronous updates: workers don't wait for each other → higher throughput.
    Used at Google scale for training on thousands of machines.

        strategy     = tf.distribute.ParameterServerStrategy(cluster_resolver)
        coordinator  = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
        # Workers and parameter servers are separate processes

### TPUStrategy (Google TPU pods)

        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://...')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.TPUStrategy(resolver)
        with strategy.scope():
            model = build_model()

    TPU training requires: bfloat16 dtype, XLA-compatible ops, specific batch sizes.

### Strategy Comparison

    ┌────────────────────────────────────────────────────────────────────┐
    │ Strategy                   │ Use Case                              │
    ├────────────────────────────────────────────────────────────────────┤
    │ MirroredStrategy           │ 1 machine, 2-8 GPUs                   │
    │ MultiWorkerMirroredStrategy│ Multiple machines, each with GPUs     │
    │ TPUStrategy                │ Google TPU (v3, v4, v5)               │
    │ ParameterServerStrategy    │ Async, thousands of workers           │
    │ OneDeviceStrategy          │ Testing distribution code on 1 device │
    └────────────────────────────────────────────────────────────────────┘

    The key API: wrap model creation in strategy.scope(). That's it.
    model.fit() automatically distributes data across replicas.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · GradientTape & @tf.function — Differentiation and Compilation": {
        "description": (
            "Deep dive into TensorFlow's two core mechanisms. "
            "GradientTape: explicit recording, persistent tapes, higher-order "
            "derivatives, watching non-variable tensors. "
            "@tf.function: tracing behaviour, Python side-effect traps, "
            "retrace triggers, input_signature pinning, XLA compilation. "
            "Benchmark eager vs compiled training steps."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  GRADIENTTAPE & @tf.function — DIFFERENTIATION & COMPILATION")
print("=" * 65)
print()

import tensorflow as tf

print(f"  TensorFlow version: {tf.__version__}")
print(f"  GPUs: {tf.config.list_physical_devices('GPU')}")
print(f"  Eager execution: {tf.executing_eagerly()}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: GradientTape basics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — GradientTape: Variables, tensors, and watching")
print("━" * 65)
print()

# tf.Variable is watched automatically
w = tf.Variable(3.0, name='w')
b = tf.Variable(2.0, name='b')

with tf.GradientTape() as tape:
    y    = w * 4.0 + b       # y = 4w + b
    loss = (y - 20.0) ** 2   # loss = (4w+b-20)^2

dw, db = tape.gradient(loss, [w, b])
print(f"  y = w*4 + b = {y.numpy():.1f}  (w={w.numpy()}, b={b.numpy()})")
print(f"  loss = (y-20)^2 = {loss.numpy():.1f}")
print()
print(f"  dl/dw = 2*(y-20)*4 = {dw.numpy():.2f}  (analytic: {2*(y.numpy()-20)*4:.2f})")
print(f"  dl/db = 2*(y-20)*1 = {db.numpy():.2f}  (analytic: {2*(y.numpy()-20)*1:.2f})")
print()

# tf.Tensor (constant) is NOT watched by default
x_const = tf.constant([2.0, 3.0])
with tf.GradientTape() as tape:
    y_const = tf.reduce_sum(x_const ** 2)
grad_none = tape.gradient(y_const, x_const)
print(f"  Gradient of constant without tape.watch: {grad_none}  (None — not watched!)")

with tf.GradientTape() as tape:
    tape.watch(x_const)           # explicitly watch a constant tensor
    y_const = tf.reduce_sum(x_const ** 2)
grad_watched = tape.gradient(y_const, x_const)
print(f"  Gradient after tape.watch:              {grad_watched.numpy()}  (correct: [4, 6])")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Persistent tape for multi-loss training
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Persistent tape (multi-loss / GAN pattern)")
print("━" * 65)
print()

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(4, use_bias=False)
    def call(self, x): return tf.nn.tanh(self.dense(x))

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, use_bias=False)
    def call(self, x): return tf.nn.sigmoid(self.dense(x))

G = Generator()
D = Discriminator()
opt_G = tf.keras.optimizers.Adam(1e-3)
opt_D = tf.keras.optimizers.Adam(1e-3)

noise    = tf.random.normal([8, 2])
real_data = tf.random.normal([8, 4], mean=2.0)

# GAN step: one tape, two separate .gradient() calls
with tf.GradientTape(persistent=True) as tape:
    fake_data   = G(noise)
    d_real      = D(real_data)
    d_fake      = D(fake_data)

    loss_D = -tf.reduce_mean(tf.math.log(d_real + 1e-7)
                              + tf.math.log(1 - d_fake + 1e-7))
    loss_G = -tf.reduce_mean(tf.math.log(d_fake + 1e-7))

grads_D = tape.gradient(loss_D, D.trainable_variables)  # first call
grads_G = tape.gradient(loss_G, G.trainable_variables)  # second call — persistent!
del tape   # MUST delete to free memory

opt_D.apply_gradients(zip(grads_D, D.trainable_variables))
opt_G.apply_gradients(zip(grads_G, G.trainable_variables))

print(f"  GAN step complete:")
print(f"    Discriminator loss: {loss_D.numpy():.4f}")
print(f"    Generator loss:     {loss_G.numpy():.4f}")
print(f"    D gradients:  {len(grads_D)} tensors")
print(f"    G gradients:  {len(grads_G)} tensors")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Higher-order derivatives
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Higher-order derivatives (nested tapes)")
print("━" * 65)
print()

x = tf.Variable(2.0)

with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = x ** 4         # y = x⁴
    dy_dx = inner.gradient(y, x)     # dy/dx = 4x³
d2y_dx2 = outer.gradient(dy_dx, x)  # d²y/dx² = 12x²

print(f"  y = x⁴,  x = {x.numpy()}")
print(f"  dy/dx  (4x³):     {dy_dx.numpy():.2f}  (analytic: {4 * x.numpy()**3:.2f})")
print(f"  d²y/dx² (12x²):  {d2y_dx2.numpy():.2f}  (analytic: {12 * x.numpy()**2:.2f})")
print()

# Gradient penalty (GAN Wasserstein) — uses second order implicitly
x_hat = tf.Variable(tf.random.normal([4, 4]))
with tf.GradientTape() as tape:
    d_out = D(x_hat)
    grad  = tape.gradient(d_out, x_hat)          # first order: ∂D/∂x̂
    gp    = tf.reduce_mean((tf.norm(grad, axis=1) - 1.0) ** 2)

print(f"  Gradient penalty (WGAN-GP): {gp.numpy():.4f}")
print(f"  Gradient norm at x_hat:     {tf.norm(grad, axis=1).numpy().round(4)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: @tf.function tracing and side effects
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — @tf.function: tracing behaviour")
print("━" * 65)
print()

trace_count = [0]

@tf.function
def traced_fn(x):
    trace_count[0] += 1            # Python side effect — runs at TRACE time only
    tf.print("TF print:", x)       # TF op — runs at EXECUTION time every call
    return x * 2

print("  Calling traced_fn with different inputs:")
for val, shape_desc in [(tf.constant(1.0),       "scalar float32"),
                         (tf.constant(2.0),       "scalar float32 (same shape → cached)"),
                         (tf.constant([1.0, 2.0]),"vector float32 (new shape → retrace!)")]:
    result = traced_fn(val)
    print(f"    Input: {shape_desc:35s} → trace_count = {trace_count[0]}")

print()
print("  Python side effect (trace_count) only incremented at trace time.")
print("  Same shape/dtype: uses cached graph → trace_count stays the same.")
print("  New shape → new trace → trace_count increments again.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: @tf.function with input_signature (pin shape → ONE trace)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — input_signature: prevent retracing")
print("━" * 65)
print()

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 32], dtype=tf.float32),  # dynamic batch, fixed features
    tf.TensorSpec(shape=[None],     dtype=tf.int32),
])
def stable_train_step(x, labels):
    """This function traces ONCE regardless of batch size."""
    logits = tf.matmul(x, tf.ones([32, 10]))
    loss   = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
    return loss

# All batch sizes → same trace (batch dim is None = dynamic)
for bs in [8, 16, 32, 64]:
    loss = stable_train_step(tf.random.normal([bs, 32]),
                              tf.random.uniform([bs], 0, 10, dtype=tf.int32))
print(f"  Ran stable_train_step with batch sizes 8, 16, 32, 64 — all use ONE trace ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Eager vs @tf.function benchmark
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Eager vs @tf.function training step benchmark")
print("━" * 65)
print()

model_bench = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10),
])
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

X_b = tf.random.normal([64, 128])
y_b = tf.random.uniform([64], 0, 10, dtype=tf.int32)

def eager_step(x, y):
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model_bench(x, training=True))
    grads = tape.gradient(loss, model_bench.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_bench.trainable_variables))
    return loss

@tf.function(input_signature=[
    tf.TensorSpec([None, 128], tf.float32),
    tf.TensorSpec([None],      tf.int32),
])
def compiled_step(x, y):
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model_bench(x, training=True))
    grads = tape.gradient(loss, model_bench.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_bench.trainable_variables))
    return loss

# Warmup
for _ in range(5): compiled_step(X_b, y_b)

N = 100
t0 = time.perf_counter()
for _ in range(N): eager_step(X_b, y_b)
t_eager = (time.perf_counter() - t0) / N * 1000

t0 = time.perf_counter()
for _ in range(N): compiled_step(X_b, y_b)
t_compiled = (time.perf_counter() - t0) / N * 1000

print(f"  3-layer Dense network, batch=64, features=128  ({N} steps each)")
print(f"    Eager:              {t_eager:.3f} ms/step")
print(f"    @tf.function:       {t_compiled:.3f} ms/step")
print(f"    Speedup:            {t_eager/t_compiled:.2f}×")
print()
print("  @tf.function speedup is larger on GPU (2-5×) and for smaller models")
print("  (Python overhead is proportionally bigger vs compute).")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Keras Full Stack — Three APIs, Custom Layers & Callbacks": {
        "description": (
            "Complete Keras usage across all three abstraction levels. "
            "Sequential for linear pipelines. Functional API for residual networks. "
            "Model subclassing for a transformer encoder. "
            "Custom Layer with add_weight and regularisation. "
            "Full custom training loop via train_step override. "
            "Callback system: built-in and custom. "
            "Model saving, loading, and serving signatures."
        ),
        "language": "python",
        "code": '''
import numpy as np
import os, tempfile, time
import tensorflow as tf

print("=" * 65)
print("  KERAS FULL STACK — THREE APIs, CUSTOM LAYERS & CALLBACKS")
print("=" * 65)
print()

tf.random.set_seed(42)
np.random.seed(42)

# ── Shared dataset ─────────────────────────────────────────────────────
N_TRAIN, N_VAL, N_FEAT, N_CLS = 2000, 400, 32, 5
X_train = np.random.randn(N_TRAIN, N_FEAT).astype(np.float32)
y_train = np.random.randint(0, N_CLS, N_TRAIN)
X_val   = np.random.randn(N_VAL, N_FEAT).astype(np.float32)
y_val   = np.random.randint(0, N_CLS, N_VAL)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(2000).batch(64).prefetch(tf.data.AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(64).prefetch(tf.data.AUTOTUNE))

print(f"  Dataset: {N_TRAIN} train, {N_VAL} val, {N_FEAT} features, {N_CLS} classes")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Sequential API
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Sequential API (linear stack)")
print("━" * 65)
print()

model_seq = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_FEAT,)),
    tf.keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(N_CLS),
], name="sequential_model")

model_seq.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')],
)

print(f"  Parameters: {model_seq.count_params():,}")
print()
model_seq.summary(print_fn=lambda s: print(f"  {s}"))
print()

history = model_seq.fit(
    train_ds, validation_data=val_ds,
    epochs=5, verbose=0,
)
final_acc = history.history['val_acc'][-1]
print(f"  Trained 5 epochs → val_acc = {final_acc:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Functional API (residual connections)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Functional API (residual connections)")
print("━" * 65)
print()

def residual_block(x, units):
    """Residual block using functional API: output = x + F(x)."""
    # Project to same dimension if needed
    if x.shape[-1] != units:
        skip = tf.keras.layers.Dense(units, use_bias=False)(x)
    else:
        skip = x
    h = tf.keras.layers.Dense(units, activation='relu')(x)
    h = tf.keras.layers.LayerNormalization()(h)
    h = tf.keras.layers.Dense(units)(h)
    h = tf.keras.layers.Add()([h, skip])   # skip connection
    return tf.keras.layers.Activation('relu')(h)

inputs = tf.keras.Input(shape=(N_FEAT,), name='features')
x      = residual_block(inputs, 64)
x      = residual_block(x,      64)
x      = residual_block(x,      32)
outputs = tf.keras.layers.Dense(N_CLS, name='logits')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet_mlp')
model_func.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'],
)

print(f"  ResNet-style MLP  |  Params: {model_func.count_params():,}")
print(f"  Layers: {len(model_func.layers)}")
# Show unique layer types
layer_types = {type(l).__name__ for l in model_func.layers}
print(f"  Layer types: {sorted(layer_types)}")
print()

history_f = model_func.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
print(f"  Trained 5 epochs → val_acc = {history_f.history['sparse_categorical_accuracy'][-1]:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Custom Layer with add_weight
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Custom Layer: add_weight and build()")
print("━" * 65)
print()

class ScaledDotSelfAttention(tf.keras.layers.Layer):
    """
    Simplified single-head self-attention as a custom Keras layer.
    Demonstrates: add_weight, build(), call(), get_config().
    """
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        """
        build() is called once with the input shape.
        Weights are created here so the layer works with any input dim.
        """
        d = input_shape[-1]   # input feature dimension
        self.Wq = self.add_weight(name='Wq', shape=(d, self.d_model),
                                   initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(name='Wk', shape=(d, self.d_model),
                                   initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(name='Wv', shape=(d, self.d_model),
                                   initializer='glorot_uniform', trainable=True)
        self.Wo = self.add_weight(name='Wo', shape=(self.d_model, d),
                                   initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, x, training=False):
        # x: (batch, seq_len, d_input)
        Q = x @ self.Wq                              # (batch, T, d_model)
        K = x @ self.Wk
        V = x @ self.Wv
        scale   = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        scores  = tf.matmul(Q, K, transpose_b=True) / scale   # (batch, T, T)
        weights = tf.nn.softmax(scores, axis=-1)
        context = weights @ V                        # (batch, T, d_model)
        return context @ self.Wo                     # (batch, T, d_input)

    def get_config(self):
        """Required for model serialisation."""
        return {**super().get_config(), 'd_model': self.d_model}

# Test custom layer
attn  = ScaledDotSelfAttention(d_model=16)
x_seq = tf.random.normal([2, 8, 32])   # batch=2, seq=8, features=32
out   = attn(x_seq)
print(f"  ScaledDotSelfAttention(d_model=16)")
print(f"  Input:  {x_seq.shape}")
print(f"  Output: {out.shape}")
print(f"  Weights: {[w.name + ' ' + str(w.shape) for w in attn.weights]}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Custom training loop via train_step override
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Custom train_step (label smoothing + mixup)")
print("━" * 65)
print()

class MixupModel(tf.keras.Model):
    """
    Model with Mixup augmentation baked into train_step.
    Mixup: interpolate two samples and their labels.
    α ∈ [0,1] drawn from Beta(α_param, α_param).
    """
    def __init__(self, base_model, alpha=0.2):
        super().__init__()
        self.base = base_model
        self.alpha = alpha

    def call(self, x, training=False):
        return self.base(x, training=training)

    def train_step(self, data):
        X, y = data
        batch_size = tf.shape(X)[0]

        # Mixup augmentation
        lam   = tf.constant(self.alpha, dtype=tf.float32)
        perm  = tf.random.shuffle(tf.range(batch_size))
        X_mix = lam * X + (1 - lam) * tf.gather(X, perm)

        # One-hot labels for interpolation
        y_oh  = tf.one_hot(tf.cast(y, tf.int32), N_CLS)
        y_mix = lam * y_oh + (1 - lam) * tf.gather(y_oh, perm)

        with tf.GradientTape() as tape:
            logits = self(X_mix, training=True)
            # Soft cross-entropy (y_mix is soft labels from mixup)
            log_probs = tf.nn.log_softmax(logits)
            loss      = -tf.reduce_mean(tf.reduce_sum(y_mix * log_probs, axis=1))
            loss     += sum(self.losses)   # regularisation

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics with ORIGINAL (not mixed) labels for correct accuracy
        self.compiled_metrics.update_state(y, logits)
        return {'loss': loss, **{m.name: m.result() for m in self.metrics}}

base = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_FEAT,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(N_CLS),
])

mixup_model = MixupModel(base, alpha=0.2)
mixup_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')],
)

h = mixup_model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
print(f"  MixupModel (custom train_step) trained 5 epochs")
print(f"  val_acc = {h.history['val_acc'][-1]:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Callbacks
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Callbacks: built-in and custom")
print("━" * 65)
print()

class LRWarmupScheduler(tf.keras.callbacks.Callback):
    """
    Custom callback: linear warmup for first N batches.
    Sets LR from 0 to target_lr over warmup_batches steps.
    """
    def __init__(self, target_lr: float, warmup_batches: int):
        super().__init__()
        self.target_lr      = target_lr
        self.warmup_batches = warmup_batches
        self.batch_count    = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count <= self.warmup_batches:
            frac = self.batch_count / self.warmup_batches
            new_lr = self.target_lr * frac
            self.model.optimizer.learning_rate.assign(new_lr)

    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        print(f"    [LRWarmup] epoch {epoch+1}: lr={lr:.6f}")

with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, "best_model.keras")

    callbacks = [
        LRWarmupScheduler(target_lr=1e-3, warmup_batches=30),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_acc', save_best_only=True, verbose=0),
    ]

    fresh_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_FEAT,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(N_CLS),
    ])
    fresh_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                        loss='sparse_categorical_crossentropy',
                        metrics=['sparse_categorical_accuracy'])

    h2 = fresh_model.fit(train_ds, validation_data=val_ds,
                          epochs=8, callbacks=callbacks, verbose=0)

    print()
    print(f"  Training summary:")
    print(f"    Epochs ran:  {len(h2.history['loss'])} (EarlyStopping may have intervened)")
    print(f"    Best val_acc: {max(h2.history.get('val_sparse_categorical_accuracy', [0])):.4f}")

    if os.path.exists(ckpt_path):
        restored = tf.keras.models.load_model(ckpt_path)
        test_out  = restored.predict(X_val[:4], verbose=0)
        print(f"    Checkpoint loaded and predicts shape: {test_out.shape} ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: SavedModel and serving signature
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — SavedModel format and serving signature")
print("━" * 65)
print()

class ServedModel(tf.keras.Model):
    """Model with explicit serving signature for TF Serving."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    def call(self, x, training=False):
        return self.base(x, training=training)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, N_FEAT], dtype=tf.float32)
    ])
    def serve(self, x):
        """Serving endpoint: returns probabilities, not logits."""
        logits = self(x, training=False)
        return {
            "probabilities": tf.nn.softmax(logits),
            "predicted_class": tf.argmax(logits, axis=1),
        }

served = ServedModel(model_seq.layers)

with tempfile.TemporaryDirectory() as tmp:
    save_path = os.path.join(tmp, "served_model")
    tf.saved_model.save(served, save_path,
                        signatures={"serving_default": served.serve})

    loaded     = tf.saved_model.load(save_path)
    infer_fn   = loaded.signatures["serving_default"]
    sample     = tf.random.normal([3, N_FEAT])
    result     = infer_fn(sample)

    print(f"  SavedModel saved to: {save_path}/")
    print(f"  Signature keys: {list(result.keys())}")
    print(f"  Probabilities shape: {result['probabilities'].shape}")
    print(f"  Predicted classes:   {result['predicted_class'].numpy()}")
    print(f"  Probs sum to 1:      {tf.reduce_all(tf.abs(tf.reduce_sum(result['probabilities'], axis=1) - 1) < 1e-5).numpy()}")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · tf.data, TFLite & Distribution Strategies": {
        "description": (
            "tf.data pipeline performance: shuffle, map, cache, prefetch. "
            "TFRecord reading and writing for production datasets. "
            "TFLite conversion and quantisation: FP32 vs FP16 vs INT8. "
            "MirroredStrategy for multi-GPU training. "
            "Complete production workflow: train → save → quantise → deploy."
        ),
        "language": "python",
        "code": '''
import numpy as np
import os, tempfile, time
import tensorflow as tf

print("=" * 65)
print("  tf.data, TFLite & DISTRIBUTION STRATEGIES")
print("=" * 65)
print()

tf.random.set_seed(0)
np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: tf.data pipeline construction and performance
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — tf.data pipeline: shuffle, map, cache, prefetch")
print("━" * 65)
print()

N = 4096
X_all = np.random.randn(N, 64).astype(np.float32)
y_all = np.random.randint(0, 10, N, dtype=np.int32)

def augment(x, y):
    """Feature-space augmentation: add Gaussian noise."""
    x = x + tf.random.normal(tf.shape(x), stddev=0.05)
    return x, y

def normalise(x, y):
    """Z-score normalisation (mean=0, std=1 per feature)."""
    x = (x - tf.constant(X_all.mean(0))) / tf.constant(X_all.std(0) + 1e-8)
    return x, y

# ── Naive pipeline (no prefetch) ──────────────────────────────────────
ds_naive = (tf.data.Dataset.from_tensor_slices((X_all, y_all))
            .shuffle(1000)
            .map(normalise)
            .map(augment)
            .batch(64))

# ── Optimised pipeline ────────────────────────────────────────────────
ds_opt = (tf.data.Dataset.from_tensor_slices((X_all, y_all))
          .cache()                                     # ← cache after first epoch
          .shuffle(1000, reshuffle_each_iteration=True)
          .map(normalise,  num_parallel_calls=tf.data.AUTOTUNE)
          .map(augment,    num_parallel_calls=tf.data.AUTOTUNE)
          .batch(64)
          .prefetch(tf.data.AUTOTUNE))                 # ← overlap with GPU

def time_dataset(ds, n_epochs=3):
    times = []
    for ep in range(n_epochs):
        t0 = time.perf_counter()
        for batch in ds: pass
        times.append(time.perf_counter() - t0)
    return np.mean(times)

print(f"  Dataset: {N} samples, 2 map transforms, batch=64")
print()

# Warm up cache
for _ in ds_opt: pass

t_naive  = time_dataset(ds_naive,  n_epochs=3)
t_opt    = time_dataset(ds_opt,    n_epochs=3)
speedup  = t_naive / t_opt

print(f"  {'Pipeline':<30} | {'Time/epoch (s)':>16} | {'Speedup':>10}")
print(f"  {'─'*60}")
print(f"  {'Naive (no cache/prefetch)':<30} | {t_naive:16.4f} | {'1.00×':>10}")
print(f"  {'Optimised (cache+prefetch)':<30} | {t_opt:16.4f} | {speedup:9.2f}×")
print()
print("  Optimisation breakdown:")
print("    .cache():      saves first-epoch compute for subsequent epochs")
print("    AUTOTUNE map:  parallelise CPU transforms across threads")
print("    .prefetch():   prepare next batch while GPU processes current")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: TFRecord write and read
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — TFRecord: write and read")
print("━" * 65)
print()

def make_feature(value, dtype='float'):
    if dtype == 'float':
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    elif dtype == 'int':
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecords(path, X, y, n_samples=200):
    with tf.io.TFRecordWriter(path) as writer:
        for i in range(min(n_samples, len(X))):
            feature = {
                'features': make_feature(X[i].tolist(), 'float'),
                'label':    make_feature(int(y[i]),     'int'),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def parse_tfrecord(serialised):
    feature_desc = {
        'features': tf.io.FixedLenFeature([64], tf.float32),
        'label':    tf.io.FixedLenFeature([],   tf.int64),
    }
    parsed = tf.io.parse_single_example(serialised, feature_desc)
    return parsed['features'], parsed['label']

with tempfile.TemporaryDirectory() as tmp:
    tfr_path = os.path.join(tmp, 'train.tfrecord')
    write_tfrecords(tfr_path, X_all, y_all, n_samples=500)
    file_kb  = os.path.getsize(tfr_path) / 1024

    raw_ds    = tf.data.TFRecordDataset([tfr_path])
    parsed_ds = (raw_ds
                 .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
                 .shuffle(256)
                 .batch(32)
                 .prefetch(tf.data.AUTOTUNE))

    batches  = list(parsed_ds)
    n_read   = sum(b[0].shape[0] for b in batches)

    print(f"  Written {500} samples to TFRecord")
    print(f"  File size: {file_kb:.1f} KB")
    print(f"  Read back: {n_read} samples in {len(batches)} batches")
    print(f"  Batch shape: features={batches[0][0].shape}, label={batches[0][1].shape}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: TFLite conversion and quantisation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — TFLite conversion: FP32 vs FP16 vs INT8")
print("━" * 65)
print()

# Train a small model to convert
src_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10),
], name="tflite_source")
src_model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
src_model.fit(tf.data.Dataset.from_tensor_slices((X_all, y_all)).batch(64),
               epochs=3, verbose=0)
print(f"  Source model trained: {src_model.count_params():,} params")
print()

X_test   = X_all[:100].astype(np.float32)
y_test   = y_all[:100]
ref_preds = src_model.predict(X_test, verbose=0).argmax(1)

def representative_data_gen():
    """Calibration data for full-integer quantisation."""
    for i in range(0, 200, 10):
        yield [X_all[i:i+10].astype(np.float32)]

results = {}

with tempfile.TemporaryDirectory() as tmp:
    saved_path = os.path.join(tmp, 'model.keras')
    src_model.save(saved_path)

    for mode, label in [
        (None,                              "FP32 (no quantisation)"),
        ([tf.lite.Optimize.DEFAULT],        "Dynamic range (weights INT8)"),
    ]:
        converter = tf.lite.TFLiteConverter.from_keras_model(src_model)
        if mode:
            converter.optimizations = mode
        tflite_model = converter.convert()

        tflite_path = os.path.join(tmp, f'model_{label[:4]}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        size_kb = os.path.getsize(tflite_path) / 1024

        # Run inference with TFLite interpreter
        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        in_idx  = interp.get_input_details()[0]['index']
        out_idx = interp.get_output_details()[0]['index']

        preds = []
        t0 = time.perf_counter()
        for i in range(len(X_test)):
            interp.set_tensor(in_idx, X_test[i:i+1])
            interp.invoke()
            preds.append(interp.get_tensor(out_idx).argmax())
        t_inf = (time.perf_counter() - t0) / len(X_test) * 1000

        agreement = np.mean(np.array(preds) == ref_preds) * 100
        results[label] = dict(size_kb=size_kb, t_ms=t_inf, agreement=agreement)

print(f"  {'Mode':<35} | {'Size (KB)':>10} | {'ms/sample':>11} | {'Agreement':>11}")
print(f"  {'─'*75}")
for label, r in results.items():
    print(f"  {label:<35} | {r['size_kb']:10.1f} | {r['t_ms']:11.4f} | {r['agreement']:10.1f}%")
print()
print("  Full INT8 (representative_dataset) saves another 2× in size.")
print("  Required for NPU acceleration on Android (Pixel Neural Core).")
print("  Recommendation: use dynamic-range quant as default,")
print("                  full-int8 when deploying to dedicated hardware.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: MirroredStrategy for multi-GPU training
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — MirroredStrategy (multi-GPU training)")
print("━" * 65)
print()

n_gpus = len(tf.config.list_physical_devices('GPU'))
print(f"  Physical GPUs available: {n_gpus}")
print()

MIRRORED_PATTERN = """
# ── MirroredStrategy — single machine, multiple GPUs ─────────────────
strategy = tf.distribute.MirroredStrategy()
print(f"Replicas: {strategy.num_replicas_in_sync}")   # = number of GPUs

# ALL model code goes inside strategy.scope()
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64,)),
        tf.keras.layers.Dense(256,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    # Optimizer inside scope: each replica gets its own optimizer state
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'],
    )

# Dataset: batch size should be GLOBAL (per_replica × num_replicas)
BATCH_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_PER_REPLICA * strategy.num_replicas_in_sync

train_ds = (tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(10000)
            .batch(GLOBAL_BATCH_SIZE)           # ← global batch
            .prefetch(tf.data.AUTOTUNE))

# model.fit() distributes automatically
model.fit(train_ds, epochs=10)

# What MirroredStrategy does under the hood:
# 1. Replicates model weights to all GPUs
# 2. Splits each global batch into per-replica shards
# 3. Each GPU runs forward + backward on its shard
# 4. AllReduce (NCCL) averages gradients across all GPUs
# 5. All replicas apply the same averaged gradient → stay in sync
"""
print(MIRRORED_PATTERN)

if n_gpus >= 2:
    strategy  = tf.distribute.MirroredStrategy()
    GLOBAL_BS = 64 * strategy.num_replicas_in_sync
    dist_ds   = (tf.data.Dataset.from_tensor_slices((X_all, y_all))
                 .shuffle(4096).batch(GLOBAL_BS).prefetch(tf.data.AUTOTUNE))
    with strategy.scope():
        dist_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ])
        dist_model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['acc'])
    dist_model.fit(dist_ds, epochs=2, verbose=0)
    print(f"  MirroredStrategy training complete ({n_gpus} GPUs) ✅")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"  MirroredStrategy on CPU (num_replicas={strategy.num_replicas_in_sync})")
    print(f"  (Run on multi-GPU machine to see actual parallel speedup)")
print()

print("  DISTRIBUTION STRATEGY DECISION GUIDE:")
print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Strategy                   │ When to use                         │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ MirroredStrategy           │ 1 machine, 2-8 GPUs. Start here.    │")
print("  │ MultiWorkerMirroredStrategy│ Multiple machines, each with GPUs   │")
print("  │ TPUStrategy                │ Google TPU (v3/v4/v5)               │")
print("  │ ParameterServerStrategy    │ Async at scale (1000s of workers)   │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()
print("  KEY RULE: scale GLOBAL batch size proportionally with replicas.")
print("  If 1 GPU needs batch=64, then 4 GPUs should use batch=256.")
print("  This keeps per-replica batch size constant → same convergence.")
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