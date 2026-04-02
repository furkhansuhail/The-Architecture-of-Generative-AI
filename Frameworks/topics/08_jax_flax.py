"""
JAX + Flax — Functional Deep Learning with XLA Compilation
===========================================================

JAX is a numerical computing library from Google Research, released in 2018,
built around a single radical idea: make NumPy composable with program
transformations. The four core transformations — jit (JIT compilation via
XLA), grad (automatic differentiation), vmap (vectorisation), and pmap
(parallel execution) — each turn any Python function into a faster, smarter,
or more parallel version of itself. And they compose: jit(vmap(grad(f))) works.

This functional design is not cosmetic. It reflects a deep insight: if a
function is pure (no side effects, same input always gives same output),
then the compiler can do things to it that are simply impossible with stateful
code. XLA can fuse entire computation graphs into single GPU kernels. Gradient
checkpointing becomes a one-line annotation. Data-parallel training requires
no communication primitives — just pmap your training step.

Flax is Google's neural network library built on JAX, providing the Module
abstraction that JAX itself deliberately omits. Flax 0.7+ introduces NNX
(Neural Network eXtended), a module system with standard Python mutable state
that feels natural compared to the purely functional Linen API it builds upon.
Flax sits between raw JAX (maximum flexibility, maximum boilerplate) and
Keras (maximum convenience, constrained flexibility).

Optax completes the stack — a composable gradient transformation library.
Instead of a monolithic Optimizer class, Optax chains gradient transforms:
clip → scale by lr → add weight decay → apply momentum. Any gradient
processing you can describe algorithmically, you can compose in Optax.

Understanding JAX teaches you what every modern ML framework converges on
under the hood: functional transforms, XLA compilation, and explicit state
management. JAX is where the field does its most radical research because
it gives you the lowest-level control with the highest-level abstractions.

This module covers JAX's transformation system, JIT compilation mechanics,
the pytree system, Flax NNX and Linen modules, Optax gradient transforms,
the full training loop with TrainState, custom VJP/JVP rules, scan-based
RNNs, and distributed training with device sharding.

"""

import textwrap
import re

TOPIC_NAME   = "JAX + Flax — Functional Deep Learning with XLA Compilation"
DISPLAY_NAME = "08 · JAX + Flax"
ICON         = "🔬"
SUBTITLE     = "From XLA JIT Compilation to Distributed Functional Training"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — JAX PHILOSOPHY: PURE FUNCTIONS AND COMPOSABLE TRANSFORMS

### Why Purity Enables Power

    JAX is built on a constraint that initially seems limiting but unlocks
    everything: functions must be pure and arrays must be immutable.

    Pure function: no side effects, no global state, same input → same output.

        # PURE — JAX loves this
        def relu(x):
            return jnp.maximum(x, 0.0)

        # IMPURE — JAX cannot transform this
        count = 0
        def bad_relu(x):
            global count
            count += 1          # side effect: modifies global state
            return jnp.maximum(x, 0.0)

    When a function is pure, JAX can:
        1. jit-compile it: trace the computation graph, compile to XLA,
           cache the compiled version — no Python overhead on subsequent calls.
        2. grad: compute exact gradients via reverse-mode AD through the graph.
        3. vmap: vectorise it — turn a function on a single example into a
           batched function with zero for-loop overhead.
        4. pmap: shard it across devices — run on N GPUs/TPUs in parallel.
        5. COMPOSE all of the above: jit(vmap(grad(f))) is perfectly valid.

### The Four Core Transformations

    jit(f):
        JIT-compiles f via XLA. First call traces f with abstract values,
        compiles to optimised machine code, caches result.
        Subsequent calls with same-shape inputs skip Python and run compiled code.
        Speedup: 2–100× depending on operation mix and hardware.

    grad(f):
        Returns a function that computes the gradient of f with respect to
        its first argument (by default). Uses reverse-mode automatic differentiation.
        f must return a scalar.

        loss_fn = lambda params, x, y: cross_entropy(model(params, x), y)
        grad_fn = jax.grad(loss_fn)        # dL/d(params)
        grads   = grad_fn(params, x, y)   # same structure as params

    vmap(f):
        Vectorises f over a leading batch dimension.
        No for loop, no manual batch dimension management.

        # f takes a single example (shape [D])
        def f_single(x):
            return jnp.dot(W, x)

        # vmap(f) takes a batch (shape [B, D]) and applies f_single to each row
        f_batched = jax.vmap(f_single)     # → [B] output

        in_axes argument controls which argument gets batched:
            vmap(f, in_axes=(0, None))(batched_x, shared_W)
            # batch over first arg (axis 0), broadcast second arg

    pmap(f):
        Distributes f across multiple devices (GPUs/TPUs).
        Each device gets a different shard of the input along axis 0.
        Gradients are synchronised via collective operations (AllReduce).

        # Same API as vmap — just runs on N devices instead of batching
        parallel_step = jax.pmap(train_step)

### JAX vs PyTorch: The Fundamental Model Difference

    PyTorch (and TF2 eager):
        - Mutable model objects with .parameters()
        - Optimiser has internal state (momentum buffers)
        - .backward() mutates .grad attributes in place
        - Device management via .to(device)

    JAX:
        - Parameters are plain pytrees (dicts of arrays) — no model objects
        - Optimiser state is an explicit pytree — passed in, returned out
        - Gradients are return values — never mutate anything
        - Device placement is explicit or automatic via XLA

    The JAX training loop:

        # Everything is explicit: params and opt_state are values, not objects
        params, opt_state = init(key)               # explicit initialisation

        def train_step(params, opt_state, batch):   # pure function: in → out
            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, new_opt_state = optimiser.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss  # return new state

        for batch in dataloader:
            params, opt_state, loss = train_step(params, opt_state, batch)

    Notice: train_step is a PURE function. It takes state in, returns new state
    out. This is why jit(train_step) works perfectly.

### JAX's Relationship to XLA

    XLA (Accelerated Linear Algebra) is a compiler for linear algebra computations
    developed by Google. It is the backend for both TensorFlow and JAX.

    When you call jit(f):
        1. JAX traces f with abstract ShapedArray values (shapes known, values unknown)
        2. The trace produces a Jaxpr (JAX expression) — a dataflow graph
        3. XLA compiles the Jaxpr to machine code for the target device
        4. XLA optimisations:
               - Operator fusion: conv + bias + relu → single kernel
               - Memory layout optimisation (NCHW vs NHWC)
               - Algebraic simplification (x * 1 → x)
               - Common subexpression elimination

    The first call to a jitted function is slow (tracing + compilation).
    Every subsequent call with the same input shapes is fast (compiled code).
    Shape change → retracing and recompilation.

    This is why shape dynamism is harder in JAX than PyTorch:
    JAX requires static shapes to compile. Use padding for variable-length inputs.


##### PART 2 — JAX ARRAYS, DEVICES, AND THE NUMPY API

### jax.numpy vs numpy

    jax.numpy (jnp) mirrors numpy's API almost exactly. Most numpy code
    can be replaced by jnp with no changes, gaining JIT compilability:

        import jax.numpy as jnp

        # Creation
        jnp.zeros((3, 4))
        jnp.ones((3, 4))
        jnp.arange(10)
        jnp.linspace(0, 1, 100)

        # Manipulation
        jnp.reshape(a, (2, -1))
        jnp.transpose(a, (1, 0, 2))
        jnp.concatenate([a, b], axis=0)
        jnp.stack([a, b], axis=0)

        # Reduction
        jnp.sum(a, axis=-1, keepdims=True)
        jnp.mean(a), jnp.std(a)
        jnp.max(a, axis=0)

        # Neural network ops
        jax.nn.relu(x), jax.nn.sigmoid(x), jax.nn.softmax(x, axis=-1)
        jax.nn.gelu(x), jax.nn.swish(x), jax.nn.silu(x)

    Key difference from numpy:
        JAX arrays are IMMUTABLE. a[0] = 5 raises an error.
        Use functional update:  a = a.at[0].set(5)
                                a = a.at[1:3].add(1.0)
                                a = a.at[indices].mul(2.0)

        This immutability enables safe JIT compilation and correct autodiff.

### Random Number Generation in JAX

    This is the most surprising JAX design decision for NumPy users.
    JAX does NOT have a global random state (no np.random.seed()).

    Instead: explicit PRNG keys. Every random operation takes an explicit key.

        key = jax.random.PRNGKey(42)       # create a key from an integer seed

        # Split to get subkeys for different operations
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Use subkeys for random operations
        x = jax.random.normal(subkey1, shape=(3, 4))
        y = jax.random.uniform(subkey2, shape=(2,), minval=0, maxval=1)

    Why explicit keys?
        1. Reproducibility: the same key always produces the same output.
        2. Purity: random functions are now pure (key in → numbers out).
        3. JIT compatibility: pure functions can be compiled.
        4. Parallelism: each pmap replica gets its own key — no global state contention.

    The key splitting pattern:
        key, subkey = jax.random.split(key)   # consume one key, get one subkey
        result = some_random_fn(subkey, ...)   # use subkey for the operation
        # key is now "used up" conceptually — never reuse a key

    Folded-in keys (for loop-local keys):
        for step in range(n):
            step_key = jax.random.fold_in(key, step)   # deterministic per-step key
            data     = augment(step_key, data)

### Devices and Sharding

    List available devices:
        jax.devices()              # all available devices
        jax.devices('gpu')         # all GPU devices
        jax.local_device_count()   # GPUs on this machine

    Place an array on a specific device:
        arr = jax.device_put(arr, jax.devices('gpu')[0])

    Check where an array lives:
        arr.device()   # device object

    Move to CPU:
        arr_np = np.array(arr)   # implicit copy to CPU + conversion to numpy

    For multi-device (pmap), arrays have a leading device axis:
        # Shape [8, 32, 128] on 8 devices = 8 shards of [32, 128]
        x_sharded = jax.device_put_replicated(x, jax.devices())  # replicate
        x_sharded = x.reshape(8, -1, x.shape[-1])                 # shard


### The Pytree System

    JAX's universal data structure. A pytree is any nested combination of:
        - Python dicts      {'a': array, 'b': {'c': array}}
        - Python lists      [array1, array2]
        - Python tuples     (array1, array2)
        - NamedTuples, dataclasses (with registration)
        - Leaves: JAX arrays, Python scalars

    WHY pytrees matter:
        grad, jit, vmap, pmap all operate on pytrees transparently.
        Model parameters are naturally a pytree. Gradients have the SAME
        structure as parameters — you can add them directly (params + grads).

        params = {
            'layer1': {'W': jnp.zeros((32, 64)), 'b': jnp.zeros(64)},
            'layer2': {'W': jnp.zeros((64, 10)), 'b': jnp.zeros(10)},
        }
        # grad(loss_fn)(params, ...) returns a dict with SAME structure as params
        grads = jax.grad(loss_fn)(params, x, y)
        # grads['layer1']['W'] has same shape as params['layer1']['W']

    Pytree operations:
        jax.tree.map(fn, tree)                 # apply fn to every leaf
        jax.tree.leaves(tree)                  # get all leaves as a list
        jax.tree.structure(tree)               # get the treedef (structure without values)
        jax.tree.unflatten(treedef, leaves)    # reconstruct from structure + leaves

    Common uses:
        # Update params with gradients (elementwise subtract)
        new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

        # Compute global gradient norm
        leaves = jax.tree.leaves(grads)
        global_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))

        # Count parameters
        n_params = sum(x.size for x in jax.tree.leaves(params))


##### PART 3 — JAX TRANSFORMATIONS IN DEPTH: jit, grad, vmap, pmap

### jit: JIT Compilation and the Tracing Model

    jax.jit(f) returns a lazily-compiled version of f.
    The first call traces f — Python runs, but with abstract ShapedArray
    inputs instead of real arrays. The trace produces a Jaxpr.

        @jax.jit
        def matmul_relu(W, x):
            return jax.nn.relu(x @ W)

        # First call: slow (tracing + XLA compilation)
        y = matmul_relu(W, x)

        # Subsequent calls with same shapes: fast (compiled code)
        y = matmul_relu(W, x2)   # same shapes → cached, no Python

    Static arguments (values known at trace time):
        @functools.partial(jax.jit, static_argnums=(2,))
        def f(x, y, training: bool):
            if training:                 # Python if — only works if training is static
                return x + y
            return x * y

        f(x, y, True)   # traces with training=True → one compiled version
        f(x, y, False)  # traces again with training=False → second compiled version

    The tracing trap — side effects:
        @jax.jit
        def buggy(x):
            print("I run!")    # only at trace time, NOT on every call
            return x + 1

        buggy(jnp.array(1))   # prints "I run!" (trace)
        buggy(jnp.array(2))   # prints NOTHING (cached)
        # Use jax.debug.print("value: {x}", x=x) for in-graph printing

    Donate buffers (in-place update optimisation):
        @functools.partial(jax.jit, donate_argnums=(0,))
        def update(params, grads):
            return jax.tree.map(lambda p, g: p - 0.01*g, params, grads)
        # XLA may reuse params memory buffer for new_params — avoids allocation

### grad: Automatic Differentiation

    jax.grad(f) computes the gradient of f with respect to its FIRST argument.
    f must return a scalar.

        f    = lambda x: jnp.sum(x ** 2)
        df   = jax.grad(f)
        d2f  = jax.grad(jax.grad(f))    # second derivative — just compose

        x    = jnp.array([1.0, 2.0, 3.0])
        df(x)    # [2., 4., 6.]  ← ∂(Σx²)/∂x = 2x
        d2f(x)   # [2., 2., 2.] ← ∂²(Σx²)/∂x² = 2

    Gradient with respect to multiple arguments:
        jax.grad(f, argnums=(0, 1))   # returns tuple of (df/d_arg0, df/d_arg1)
        jax.grad(f, argnums=1)        # gradient only with respect to arg 1

    value_and_grad — get loss AND gradients in one forward pass:
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        # Equivalent to loss=f(params,...) + grads=grad(f)(params,...) but ~2× faster

    has_aux — return auxiliary data alongside the gradient:
        def loss_with_logits(params, x, y):
            logits = model(params, x)
            loss   = cross_entropy(logits, y)
            return loss, logits                 # (scalar, aux_data)

        grad_fn = jax.value_and_grad(loss_with_logits, has_aux=True)
        (loss, logits), grads = grad_fn(params, x, y)

    Jacobian (for non-scalar outputs):
        jax.jacrev(f)   # Jacobian via reverse-mode  (efficient for many inputs, few outputs)
        jax.jacfwd(f)   # Jacobian via forward-mode  (efficient for few inputs, many outputs)
        jax.hessian(f)  # jax.jacfwd(jax.jacrev(f))  — Hessian matrix

### vmap: Vectorised Batch Processing

    vmap lifts a function from single-example space into batch space.
    It is not syntactic sugar — it compiles a genuinely vectorised operation.

        # Write for a single example:
        def predict_single(params, x):    # x: [D]
            return jnp.dot(params['W'], x) + params['b']

        # vmap over the batch dimension:
        predict_batch = jax.vmap(predict_single, in_axes=(None, 0))
        # None: don't batch params (broadcast across batch)
        # 0:    batch x along axis 0 (treat first axis as batch)

        y = predict_batch(params, X)    # X: [B, D], y: [B]

    vmap with out_axes:
        jax.vmap(f, in_axes=0, out_axes=0)   # output is stacked along axis 0 (default)
        jax.vmap(f, in_axes=0, out_axes=1)   # output stacked along axis 1

    vmap + grad (per-sample gradients — impossible to do efficiently in PyTorch):
        per_sample_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
        # Returns gradient for EACH sample in the batch independently
        # Critical for: DP-SGD (differential privacy), influence functions, meta-learning

    vmap composition:
        # Batch over two dimensions simultaneously (e.g. cross-batch comparisons)
        pairwise = jax.vmap(jax.vmap(similarity_fn, in_axes=(None, 0)), in_axes=(0, None))

### pmap: Single-Program Multiple-Data (SPMD)

    pmap distributes a function across multiple devices.
    Each device receives a different slice of the input along axis 0.

        n_devices = jax.device_count()

        # Reshape data to have a leading device dimension
        x_sharded = x.reshape(n_devices, -1, *x.shape[1:])   # [N_dev, B/N, ...]

        @jax.pmap
        def parallel_forward(params, x):
            return model(params, x)

        # params must be replicated across devices
        params_rep = jax.tree.map(lambda p: jnp.stack([p] * n_devices), params)
        y = parallel_forward(params_rep, x_sharded)   # [N_dev, B/N, ...]

    Collective operations inside pmap:
        jax.lax.pmean(grads, axis_name='batch')    # AllReduce: average
        jax.lax.psum(grads,  axis_name='batch')    # AllReduce: sum
        jax.lax.pmax(x,      axis_name='batch')    # AllReduce: max

        @functools.partial(jax.pmap, axis_name='batch')
        def parallel_train_step(params, opt_state, batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            grads = jax.lax.pmean(grads, axis_name='batch')   # sync gradients!
            updates, new_state = optimiser.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state, loss

### scan: Efficient Sequential Computation

    jax.lax.scan replaces Python for-loops over a sequence.
    It compiles into a single XLA while-loop — O(1) memory, no loop unrolling.

        def scan_fn(carry, x):
            # carry: accumulated state,  x: current input → returns (new_carry, output)
            new_carry = carry + x
            output    = new_carry * 2
            return new_carry, output

        final_carry, stacked_outputs = jax.lax.scan(scan_fn, init=0.0, xs=jnp.arange(10))
        # final_carry: scalar (accumulated sum)
        # stacked_outputs: [10] array

    scan for RNNs (the key use case):
        def rnn_step(h, x_t):          # h: hidden state, x_t: input at time t
            h_new = jnp.tanh(h @ Wh + x_t @ Wx + b)
            return h_new, h_new        # (new carry, output to stack)

        h_final, all_hidden = jax.lax.scan(rnn_step, init=h0, xs=x_seq)
        # x_seq: [T, D], all_hidden: [T, H] — all timesteps' hidden states

    scan vs Python loop:
        Python loop:   O(T) compilations, O(T) XLA nodes, gradient needs O(T) memory
        scan:          O(1) XLA nodes, gradient uses O(log T) memory via rematerialisation


##### PART 4 — FLAX NNX: THE MODERN MODULE SYSTEM

### Two Flax APIs: NNX and Linen

    Flax has two module systems:

    Linen (nn.Module — the original Flax):
        Purely functional. Parameters live outside the module in a separate
        params dict. The module is a stateless transform. This is conceptually
        clean but requires passing params everywhere explicitly.

        class Linear(nn.Module):
            features: int
            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.features)(x)

        model  = Linear(features=4)
        params = model.init(key, x)['params']    # params live here
        y      = model.apply({'params': params}, x)

    NNX (introduced in Flax 0.8 — the modern API):
        Mutable Python objects. Parameters live INSIDE the module as attributes.
        Feels like PyTorch's nn.Module or Keras.
        Under the hood, NNX splits state from structure for JIT compatibility.

        class Linear(nnx.Module):
            def __init__(self, in_f, out_f, rngs):
                self.w = nnx.Param(jax.random.normal(rngs.params(), (in_f, out_f)))
                self.b = nnx.Param(jnp.zeros(out_f))

            def __call__(self, x):
                return x @ self.w.value + self.b.value

        model = Linear(4, 8, rngs=nnx.Rngs(0))   # params inside model
        y     = model(x)                           # just call it

    This module covers both, with NNX as the primary API (modern default).

### NNX Module Fundamentals

    The NNX.Module class:
        - Subclass nnx.Module
        - Create parameters in __init__ as nnx.Param, nnx.Variable subtypes
        - Define __call__ for the forward pass
        - Parameters are accessed as .value (e.g. self.w.value)

    The five NNX variable types:
        nnx.Param:          trainable parameter (updated by optimiser)
        nnx.BatchStat:      non-trainable running statistics (BatchNorm mean/var)
        nnx.Cache:          KV-cache (attention, autoregressive inference)
        nnx.Intermediate:   values returned from intermediate computations
        nnx.RngState:       PRNG key state

    NNX Rngs — the random key management system:
        rngs = nnx.Rngs(params=0, dropout=1)    # separate keys per stream
        # Usage inside module:
        x = nnx.Dropout(rate=0.5)(x, deterministic=False, rngs=rngs)

    Built-in NNX layers:
        nnx.Linear(in_features, out_features, rngs=rngs)
        nnx.Conv(in_features, out_features, kernel_size, rngs=rngs)
        nnx.BatchNorm(num_features, rngs=rngs)
        nnx.Dropout(rate=0.5)
        nnx.LayerNorm(num_features, rngs=rngs)
        nnx.Embed(num_embeddings, features, rngs=rngs)
        nnx.MultiHeadAttention(num_heads, in_features, rngs=rngs)
        nnx.LSTMCell(in_features, hidden_features, rngs=rngs)

### NNX and JIT: The State Split

    The key challenge: JIT requires pure functions, but NNX modules have
    mutable state. NNX solves this with the graph-state split:

        graphdef, state = nnx.split(model)
        # graphdef: the static structure (layer types, shapes) — hashable, static
        # state:    the mutable values (all arrays) — pytree of Variable.value

        # Use graphdef + state in a jitted function:
        @jax.jit
        def forward(graphdef, state, x):
            model = nnx.merge(graphdef, state)   # reconstruct module from parts
            y = model(x)
            _, new_state = nnx.split(model)
            return y, new_state

        y, new_state = forward(graphdef, state, x)
        nnx.update(model, new_state)             # write new state back to module

    Or use nnx.jit directly — it handles the split/merge automatically:
        @nnx.jit
        def train_step(model, optimizer, batch):
            # model and optimizer are NNX objects — nnx.jit manages state
            def loss_fn(model):
                logits = model(batch['x'])
                return cross_entropy(logits, batch['y'])
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
            return loss

### Linen API (for reference and legacy code)

    Linen is still used widely (most existing Flax code is Linen).
    Key concepts:

    @nn.compact decorator:
        Defines all sub-layers inline in __call__. Sub-layers are created
        on first call and their parameters are stored in the params dict.

        class MLP(nn.Module):
            features: int
            @nn.compact
            def __call__(self, x, training=False):
                x = nn.Dense(self.features)(x)
                x = nn.BatchNorm(use_running_average=not training)(x)
                x = nn.relu(x)
                x = nn.Dense(self.features // 2)(x)
                return x

    model.init — returns a full variables dict:
        variables = model.init(key, x)   # {params: {...}, batch_stats: {...}}
        params    = variables['params']
        batch_stats = variables['batch_stats']

    model.apply — stateless forward pass:
        y = model.apply({'params': params}, x)

        # With mutable state (BatchNorm running stats):
        y, updates = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x,
            training=True,
            mutable=['batch_stats'],     # declare which collections are mutable
        )
        batch_stats = updates['batch_stats']

    bind — alternative to apply for interactive use:
        bounded_model = model.bind({'params': params})
        y = bounded_model(x)   # no explicit params passing


##### PART 5 — OPTAX: COMPOSABLE GRADIENT TRANSFORMATIONS

### Optax's Design Philosophy

    Optax models optimisers not as monolithic objects but as compositions
    of gradient transformations (GradientTransformation). Each transformation
    is a pure function that takes gradients + state → updates + new_state.

    A GradientTransformation has two methods:
        init(params) → state              # initialise per-parameter state
        update(grads, state, params)      # transform grads, return (updates, new_state)

    Standard optimisers are compositions of primitives:
        Adam = chain(scale_by_adam(), scale(-lr))
        SGD  = chain(trace(decay=momentum), scale(-lr))
        AdamW = chain(scale_by_adam(), add_decayed_weights(wd), scale(-lr))

### Built-In Optimisers

    SGD family:
        optax.sgd(learning_rate=0.1, momentum=0.9, nesterov=True)

    Adaptive:
        optax.adam(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8)
        optax.adamw(learning_rate=3e-4, weight_decay=1e-2)   ← preferred
        optax.rmsprop(learning_rate=1e-3, decay=0.9)
        optax.adagrad(learning_rate=0.01)
        optax.lamb(learning_rate=1e-3)    # large-batch BERT-style training
        optax.lion(learning_rate=1e-4)    # sign-based, memory-efficient

### Learning Rate Schedules

    Schedules are callables that take a step count and return a scalar lr.
    Pass them as the learning_rate argument:

        schedule = optax.cosine_decay_schedule(
            init_value  = 1e-3,
            decay_steps = 10_000,
            alpha       = 1e-5,   # final lr = alpha * init_value
        )
        optimizer = optax.adamw(learning_rate=schedule)

    Available schedules:
        optax.constant_schedule(value)
        optax.linear_schedule(init, end, transition_steps)
        optax.cosine_decay_schedule(init, decay_steps, alpha)
        optax.warmup_cosine_decay_schedule(warmup_steps, decay_steps, ...)
        optax.piecewise_constant_schedule(init, {step: factor, ...})
        optax.exponential_decay(init, transition_steps, decay_rate)

    Linear warmup + cosine decay (standard for transformers):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value       = 0.0,           # start at 0
            peak_value       = 3e-4,          # warm up to peak
            warmup_steps     = 1000,
            decay_steps      = 50_000,
            end_value        = 1e-6,          # cool down to near-zero
        )

### Composing Gradient Transforms

    optax.chain(*transforms):
        Applies transforms sequentially. The standard way to build optimisers.

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),          # 1: clip gradients
            optax.scale_by_adam(b1=0.9, b2=0.999),  # 2: Adam scaling
            optax.add_decayed_weights(1e-2),          # 3: weight decay
            optax.scale(-3e-4),                       # 4: apply lr (negative = descent)
        )

    optax.masked(transform, mask):
        Apply a transform only to a subset of parameters.

        # Apply weight decay only to non-bias parameters
        mask = jax.tree.map(lambda p: p.ndim != 1, params)   # True for matrices
        wd_transform = optax.masked(optax.add_decayed_weights(1e-2), mask)

    optax.multi_transform(transforms_and_masks):
        Different optimisers for different parameter groups (fine-tuning).

        label_fn = lambda path, _: 'slow' if 'backbone' in '/'.join(path) else 'fast'
        tx = optax.multi_transform(
            {'slow': optax.adam(1e-5), 'fast': optax.adam(1e-3)},
            param_labels=optax.apply_if_finite(label_fn, params),
        )

### Using an Optax Optimiser

    The three-step pattern:

        # 1. Initialise optimiser state (pytree matching params structure)
        opt_state = optimizer.init(params)

        # 2. Compute updates from gradients
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        # params argument is optional — needed for weight decay transforms

        # 3. Apply updates to parameters
        new_params = optax.apply_updates(params, updates)

    Note: optax.apply_updates is just:
        jax.tree.map(lambda p, u: p + u, params, updates)

### Useful Optax Utilities

    Gradient clipping:
        optax.clip(max_delta)                    # clip each gradient element
        optax.clip_by_global_norm(max_norm)      # clip by total gradient norm

    Gradient accumulation:
        optax.MultiSteps(inner_optimizer, every_k_schedule=4)
        # Accumulates gradients for 4 steps before applying update

    Exponential moving average of parameters (model averaging):
        ema = optax.ema(decay=0.999)
        ema_state = ema.init(params)
        _, ema_state = ema.update(params, ema_state)
        ema_params  = ema_state.ema     # the smoothed params


##### PART 6 — THE JAX TRAINING LOOP: TRAINSTATE AND FULL PIPELINE

### The Core Training Loop Pattern

    JAX training loops are explicit about state management. Every step
    is a pure function that takes (state, batch) → (new_state, metrics).

    With Linen + TrainState (the classic pattern):

        from flax.training import train_state

        # TrainState bundles params + opt_state into a single pytree
        class TrainState(train_state.TrainState):
            batch_stats: Any = None   # extend for BatchNorm

        # Initialise
        variables = model.init(key, dummy_x)
        state = TrainState.create(
            apply_fn   = model.apply,
            params     = variables['params'],
            tx         = optimizer,              # optax optimiser
            batch_stats = variables.get('batch_stats'),
        )

        # One training step — pure function
        @jax.jit
        def train_step(state, batch):
            def loss_fn(params):
                logits, updates = state.apply_fn(
                    {'params': params, 'batch_stats': state.batch_stats},
                    batch['x'],
                    training=True,
                    mutable=['batch_stats'],
                )
                loss = cross_entropy(logits, batch['y'])
                return loss, (logits, updates)

            (loss, (logits, updates)), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)

            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates['batch_stats'])
            return state, loss

        # Training loop
        for batch in dataloader:
            state, loss = train_step(state, batch)

### TrainState: The State Pytree

    flax.training.train_state.TrainState is a dataclass with:
        apply_fn:      the model's apply function (model.apply)
        params:        current parameter pytree
        tx:            the Optax GradientTransformation
        opt_state:     the Optax optimiser state (managed internally)
        step:          int — incremented on every apply_gradients call

    Key methods:
        state.apply_gradients(grads=grads):
            Calls optimizer.update(grads, opt_state, params), applies updates,
            increments step. Returns a new TrainState.

        state.replace(key=value):
            Returns a new TrainState with the specified field replaced.
            Pure — no mutation. Used to update batch_stats, etc.

    Extending TrainState for custom fields:
        class CustomState(train_state.TrainState):
            batch_stats: Any
            ema_params:  Any
            # Additional fields are included in state checkpointing

### Checkpointing with Orbax

    Orbax is the modern JAX checkpointing library:

        import orbax.checkpoint as ocp

        checkpointer = ocp.StandardCheckpointer()

        # Save
        checkpointer.save('/path/to/ckpt', state)

        # Restore (must provide structure for abstract restore)
        restored_state = checkpointer.restore('/path/to/ckpt',
                                               target=state)  # provides structure

    Async checkpointing (saves in background while training continues):
        async_checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        async_checkpointer.save('/path/ckpt', args=ocp.args.StandardSave(state))
        async_checkpointer.wait_until_finished()   # flush before next epoch

### Metrics with Clu

    Common Loop Utilities (clu) provides JAX-native metric computation:

        import clu.metrics as clu_metrics

        @flax.struct.dataclass
        class Metrics(clu_metrics.Collection):
            accuracy: clu_metrics.Accuracy
            loss:     clu_metrics.Average.from_output('loss')

        # Inside train_step, compute metrics
        metrics = Metrics.single_from_model_output(
            logits=logits, labels=batch['y'], loss=loss
        )

        # Accumulate across batches
        batch_metrics_np = jax.device_get(jax.tree.map(lambda x: x[0], metrics))
        epoch_metrics = batch_metrics_np.merge()     # aggregate across all batches


##### PART 7 — ADVANCED JAX: CUSTOM DERIVATIVES, CHECKPOINTING, AND MIXED PRECISION

### Custom VJP and JVP Rules

    For non-differentiable operations, discontinuous functions, or numerical
    stability improvements, you can define custom gradient rules.

    Custom VJP (vector-Jacobian product — reverse mode):

        @jax.custom_vjp
        def relu(x):
            return jnp.maximum(x, 0.0)

        def relu_fwd(x):
            return relu(x), x > 0       # return (output, residuals for backward)

        def relu_bwd(residuals, g):     # g: gradient flowing back
            mask = residuals            # the residuals saved from forward
            return (g * mask,)          # gradient of relu = g if x>0 else 0

        relu.defvjp(relu_fwd, relu_bwd)

    Custom JVP (forward mode — for physics simulations, spectral methods):

        @jax.custom_jvp
        def f(x):
            return jnp.sin(x)

        @f.defjvp
        def f_jvp(primals, tangents):
            x, = primals
            xdot, = tangents
            return f(x), jnp.cos(x) * xdot   # primal out, tangent out

    Use cases for custom derivatives:
        - Straight-through estimator (binary neurons, quantisation)
        - Implicit function theorem (differentiating through optimisation)
        - Numerically stable log-sum-exp
        - Operations that JAX cannot differentiate (CUDA kernels)

### Gradient Checkpointing (Rematerialisation)

    For very deep networks, the memory cost of storing all activations for
    the backward pass is O(depth). Gradient checkpointing trades compute
    for memory: recompute activations during the backward pass.

        from jax.checkpoint import checkpoint

        @checkpoint
        def layer(params, x):
            return jnp.tanh(x @ params['W'] + params['b'])

        # Or checkpoint specific sub-expressions:
        y = jax.checkpoint(intermediate_fn)(x)

    Fine-grained checkpointing policy:
        policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
        # Only save dot-product outputs where no batch dim is involved
        y = jax.checkpoint(f, policy=policy)(x)

    Savings: from O(depth) memory → O(sqrt(depth)) with sqrt-depth checkpointing.

### Mixed Precision in JAX

    JAX handles mixed precision through explicit dtype casting.
    There is no global policy — you control dtypes manually or via Flax.

    Manual casting:
        x_f16  = x.astype(jnp.float16)
        x_bf16 = x.astype(jnp.bfloat16)

    Flax mixed precision (Linen):
        class DenseBlock(nn.Module):
            features: int
            dtype: jnp.dtype = jnp.float32

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(self.features, dtype=self.dtype)(x)
                return x

        # Use bfloat16 for compute
        model = DenseBlock(features=128, dtype=jnp.bfloat16)

    Common pattern: compute in bfloat16, keep params in float32:
        def train_step(state, batch):
            def loss_fn(params):
                # Cast params to bf16 for compute, keep as f32 for update
                params_bf16 = jax.tree.map(lambda p: p.astype(jnp.bfloat16), params)
                logits = state.apply_fn({'params': params_bf16}, batch['x'])
                return cross_entropy(logits.astype(jnp.float32), batch['y'])
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            # grads are f32 (from the f32 params); updates in f32 — stable
            return state.apply_gradients(grads=grads), loss

### Debugging in JAX

    The tracing model makes traditional debugging harder:

    jax.debug.print (in-graph printing):
        @jax.jit
        def f(x):
            jax.debug.print("x = {x}", x=x)   # prints on every call, not just trace
            return x + 1

    jax.debug.breakpoint:
        Drops into an interactive debugger when reached inside a jitted function.

    disable_jit context manager:
        with jax.disable_jit():
            y = f(x)   # runs in pure Python, no JIT — breakpoints work

    Check for NaN:
        jax.config.update('jax_debug_nans', True)   # raises on NaN production

    jax.make_jaxpr:
        jax.make_jaxpr(f)(x)   # print the Jaxpr (compiled graph representation)


##### PART 8 — DISTRIBUTED TRAINING AND DEPLOYMENT

### Device Sharding with jax.sharding (Mesh-Based)

    Modern JAX uses a mesh-based sharding model (jax.sharding) rather than
    pmap for production multi-device training. It is more flexible and
    works identically on 1 or 1000 devices.

        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
        import numpy as np

        # Create a logical device mesh
        devices = np.array(jax.devices()).reshape(2, 4)   # 2×4 mesh
        mesh    = Mesh(devices, axis_names=('data', 'model'))

        # Define sharding — how to partition each tensor
        data_sharding   = NamedSharding(mesh, P('data', None))  # shard along 'data' axis
        model_sharding  = NamedSharding(mesh, P(None, 'model')) # shard along 'model' axis
        rep_sharding    = NamedSharding(mesh, P())               # fully replicated

        # Place arrays on the mesh
        x_sharded = jax.device_put(x, data_sharding)

        # JIT with sharding constraints (compiler respects them)
        @jax.jit
        def forward(params, x):
            return model(params, x)

        with mesh:
            y = forward(params, x_sharded)

### Data Parallelism vs Model Parallelism

    Data parallelism (DDP equivalent):
        Each device holds a full copy of the model.
        Each device processes a different data shard.
        Gradients AllReduced across devices.

        data_sharding = NamedSharding(mesh, P('data', None))
        # Shard batch axis, replicate model — pure data parallelism

    Tensor parallelism (model parallel):
        Large weight matrices split across devices.
        Each device holds a shard of the model weights.

        tp_sharding = NamedSharding(mesh, P(None, 'model'))
        # Replicate batch, shard weight's column/feature dimension

    Full pipeline parallelism:
        Different transformer layers on different devices.
        Requires pipeline scheduling (micro-batches).

### SPMD Compilation Strategy

    JAX uses a single-program multiple-data (SPMD) model via XLA GSPMD.
    You write the program ONCE as if for a single device.
    XLA inserts collective operations automatically based on sharding specs.

    This contrasts with PyTorch DDP/FSDP where you explicitly wrap models
    in DDP, set up process groups, and call dist.init_process_group().

    The JAX approach scales from 1→10000 devices with the same source code.
    Google trains LLMs with hundreds of billions of parameters this way.

### Deployment: Export to StableHLO / ONNX

    JAX models can be exported via:

    1. StableHLO (Google's stable XLA IR — for TPU/GPU serving):
        from jax.export import export
        exported = export(jax.jit(model.apply))(
            {'params': params}, dummy_x
        )
        with open('model.stablehlo', 'wb') as f:
            f.write(exported.serialize())

    2. ONNX via jax2onnx or tf2onnx (JAX → TF → ONNX):
        # Convert JAX function to ONNX for cross-framework deployment

    3. TFLite via JAX → TF conversion:
        tf_fn = jax2tf.convert(jax.jit(model.apply), enable_xla=False)
        tf_module = tf.Module()
        tf_module.f = tf.function(tf_fn)
        converter = tf.lite.TFLiteConverter.from_module(tf_module)
        tflite_flat = converter.convert()

### JAX vs PyTorch vs TensorFlow Summary

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Dimension            │ JAX              │ PyTorch        │ TF/Keras  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Core abstraction     │ Function → fn    │ nn.Module      │ Layer     │
    │ State management     │ Explicit pytree  │ Mutable obj.   │ Mutable   │
    │ Compilation          │ jit (XLA)        │ torch.compile  │ @tf.fn    │
    │ Autodiff             │ grad (any order) │ backward()     │ GradTape  │
    │ Batching             │ vmap             │ manual/DataLdr │ DataLdr   │
    │ Multi-device         │ pmap/sharding    │ DDP/FSDP       │ tf.distrib│
    │ Random state         │ Explicit keys    │ Global seed    │ Global    │
    │ Research flexibility │ ★★★★★           │ ★★★★☆        │ ★★★☆☆    │
    │ Production tooling   │ ★★★☆☆           │ ★★★★☆        │ ★★★★★    │
    │ LLM training         │ ★★★★★ (TPU)     │ ★★★★★ (GPU)  │ ★★★☆☆    │
    └──────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · JAX Transformations Deep Dive — jit, grad, vmap, scan & pytrees": {
        "description": (
            "Complete tour of JAX's four core transformations. "
            "jit: tracing model, static_argnums, compilation caching, jax.debug.print. "
            "grad: scalar gradients, value_and_grad, has_aux, argnums, higher-order. "
            "vmap: per-sample gradients, in_axes/out_axes, vmap+grad composition. "
            "scan: replacing for-loops, RNN over sequences, O(1) XLA nodes. "
            "Pytrees: tree_map, tree_leaves, custom pytree registration. "
            "PRNG keys: split pattern, fold_in for loop keys. "
            "Benchmarking eager vs jit across problem sizes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import functools

try:
    import jax
    import jax.numpy as jnp
    print(f"  JAX version: {jax.__version__}")
    print(f"  Devices: {jax.devices()}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'jax', '--quiet'], check=True)
    import jax
    import jax.numpy as jnp
    print(f"  JAX version: {jax.__version__}")

print("=" * 65)
print("  JAX TRANSFORMATIONS DEEP DIVE — jit, grad, vmap, scan")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: PRNG keys — the JAX random model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — PRNG keys: explicit random state")
print("━" * 65)
print()

key = jax.random.PRNGKey(42)
print(f"  Initial key:  {key}")

# Split pattern: consume key, produce fresh subkeys
key, subkey1, subkey2 = jax.random.split(key, 3)
x1 = jax.random.normal(subkey1, (3,))
x2 = jax.random.normal(subkey2, (3,))
print(f"  Normal sample 1: {x1.round(4)}")
print(f"  Normal sample 2: {x2.round(4)}")
print(f"  Same subkey → same numbers always: {jax.random.normal(subkey1, (3,)).round(4)}")
print()

# fold_in: deterministic per-step key without consuming the master key
base_key = jax.random.PRNGKey(0)
step_keys = [jax.random.fold_in(base_key, i) for i in range(5)]
print(f"  fold_in keys for steps 0–4 (first element):")
print(f"    {[k[0].item() for k in step_keys]}")
print(f"  (deterministic per step, no master key mutation)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: jit — tracing model and compilation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — jit: tracing, caching, static args")
print("━" * 65)
print()

trace_count = [0]   # mutable container to observe tracing

@jax.jit
def traced_matmul(W, x):
    trace_count[0] += 1           # runs ONLY at trace time (Python side-effect)
    jax.debug.print("  → jax.debug.print runs every call: sum={s}", s=x.sum())
    return jax.nn.relu(x @ W)

W = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
x_a = jax.random.normal(jax.random.PRNGKey(2), (4,))
x_b = jax.random.normal(jax.random.PRNGKey(3), (4,))
x_c = jax.random.normal(jax.random.PRNGKey(4), (8,))   # different shape

print("  Calling traced_matmul with same shape twice:")
_ = traced_matmul(W, x_a)
print(f"  trace_count after call 1: {trace_count[0]}  (traced once)")
_ = traced_matmul(W, x_b)
print(f"  trace_count after call 2: {trace_count[0]}  (cached — no retrace)")
print()

# Static arguments — allow Python control flow inside jitted functions
@functools.partial(jax.jit, static_argnums=(2,))
def conditional_layer(W, x, use_relu: bool):
    y = x @ W
    return jax.nn.relu(y) if use_relu else jnp.tanh(y)   # Python if on static arg

out_relu = conditional_layer(W, x_a, True)
out_tanh = conditional_layer(W, x_a, False)
print(f"  Static arg True  → relu branch: {out_relu[:3].round(4)}")
print(f"  Static arg False → tanh branch: {out_tanh[:3].round(4)}")
print(f"  (Two compiled versions, one per static value)")
print()

# Benchmark: eager vs jit
def matmul_fn(A, B):
    return jnp.sum(A @ B)

A_big = jax.random.normal(jax.random.PRNGKey(5), (512, 512))
B_big = jax.random.normal(jax.random.PRNGKey(6), (512, 512))
jit_matmul = jax.jit(matmul_fn)

# Warmup (trigger compilation)
_ = jit_matmul(A_big, B_big).block_until_ready()

N = 100
t0 = time.perf_counter()
for _ in range(N): matmul_fn(A_big, B_big).block_until_ready()
t_eager = (time.perf_counter() - t0) / N * 1000

t0 = time.perf_counter()
for _ in range(N): jit_matmul(A_big, B_big).block_until_ready()
t_jit = (time.perf_counter() - t0) / N * 1000

print(f"  512×512 matmul benchmark ({N} runs):")
print(f"    Eager:    {t_eager:.3f} ms/call")
print(f"    jit:      {t_jit:.3f} ms/call")
print(f"    Speedup:  {t_eager / t_jit:.2f}×")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: grad — automatic differentiation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — grad: scalar gradients, value_and_grad, has_aux")
print("━" * 65)
print()

# Basic grad
f    = lambda x: jnp.sum(x ** 3)       # f(x) = Σ xᵢ³
df   = jax.grad(f)                      # df/dx = 3x²
# jax.grad requires scalar output; grad(f) returns a vector, so compose via hessian diagonal
d2f  = lambda x: jnp.diag(jax.hessian(f)(x))   # d²f/dx² = 6x (diagonal of Hessian)
x_v  = jnp.array([1.0, 2.0, 3.0])
print(f"  f(x) = Σxᵢ³,  x = {x_v}")
print(f"  df/dx   = 3x²:   {df(x_v).round(4)}")
print(f"  d²f/dx² = 6x:    {d2f(x_v).round(4)}")
print()

# value_and_grad — 2× more efficient than calling f and grad(f) separately
def loss_fn(params, x, y_true):
    logits = x @ params['W'] + params['b']
    log_sm = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(log_sm[jnp.arange(len(y_true)), y_true])

key  = jax.random.PRNGKey(7)
params = {
    'W': jax.random.normal(key, (16, 5)) * 0.01,
    'b': jnp.zeros(5),
}
X_demo = jax.random.normal(jax.random.PRNGKey(8), (32, 16))
y_demo = jax.random.randint(jax.random.PRNGKey(9), (32,), 0, 5)

loss_and_grad = jax.value_and_grad(loss_fn)
loss_val, grads = loss_and_grad(params, X_demo, y_demo)
print(f"  value_and_grad demo:")
print(f"    loss = {loss_val:.4f}")
print(f"    grad['W'] shape: {grads['W'].shape},  norm: {jnp.linalg.norm(grads['W']):.4f}")
print(f"    grad['b'] shape: {grads['b'].shape}")
print()

# has_aux — return auxiliary outputs alongside the gradient
def loss_with_logits(params, x, y_true):
    logits  = x @ params['W'] + params['b']
    log_sm  = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    loss    = -jnp.mean(log_sm[jnp.arange(len(y_true)), y_true])
    acc     = jnp.mean(logits.argmax(-1) == y_true)
    return loss, {'logits': logits, 'acc': acc}    # (scalar, aux_dict)

grad_fn = jax.value_and_grad(loss_with_logits, has_aux=True)
(loss_v, aux), grads = grad_fn(params, X_demo, y_demo)
print(f"  has_aux demo:")
print(f"    loss = {loss_v:.4f},  acc = {aux['acc']:.4f}")
print(f"    logits shape: {aux['logits'].shape}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: vmap — vectorisation and per-sample gradients
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — vmap: vectorisation and per-sample gradients")
print("━" * 65)
print()

# vmap: transform a single-example function to batched
def single_predict(W, x_single):   # x_single: [D]
    return W @ x_single            # [K]

W_layer = jax.random.normal(jax.random.PRNGKey(10), (5, 16))
x_batch = jax.random.normal(jax.random.PRNGKey(11), (32, 16))

# Without vmap: must write matmul explicitly (x_batch @ W_layer.T)
manual_out = x_batch @ W_layer.T            # [32, 5]

# With vmap: apply single_predict to every row of x_batch
batched_predict = jax.vmap(single_predict, in_axes=(None, 0))
# None: W is shared (not batched), 0: x is batched along axis 0
vmap_out = batched_predict(W_layer, x_batch)  # [32, 5]

print(f"  vmap vs manual matmul difference: {jnp.max(jnp.abs(manual_out - vmap_out)):.2e}")
print()

# Per-sample gradients via vmap(grad(f)) — impossible to do efficiently in PyTorch
def sample_loss(params, x_i, y_i):
    logits = x_i @ params['W'] + params['b']
    return jnp.sum((logits - jax.nn.one_hot(y_i, 5)) ** 2)

# Gradient for a SINGLE example
single_grad_fn = jax.grad(sample_loss)

# vmap over the batch — per-sample gradients with no Python loop
batch_grad_fn = jax.vmap(single_grad_fn, in_axes=(None, 0, 0))
per_sample_grads = batch_grad_fn(params, X_demo, y_demo)

print(f"  Per-sample gradients via vmap(grad):")
print(f"    Batch size: {X_demo.shape[0]}")
print(f"    Per-sample W grads shape: {per_sample_grads['W'].shape}  (batch, out, in)")
print(f"    Per-sample b grads shape: {per_sample_grads['b'].shape}  (batch, out)")
print(f"  → Used in DP-SGD: clip each sample's gradient individually")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: scan — O(1) memory sequential loops
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — scan: memory-efficient sequential computation")
print("━" * 65)
print()

# Basic scan: cumulative sum
def cumsum_step(carry, x):
    new_carry = carry + x
    return new_carry, new_carry   # (new carry, stacked output)

xs     = jnp.arange(1, 6, dtype=float)    # [1, 2, 3, 4, 5]
final, all_sums = jax.lax.scan(cumsum_step, init=0.0, xs=xs)
print(f"  scan cumulative sum of {xs.tolist()}:")
print(f"    All prefix sums: {all_sums.tolist()}")   # [1, 3, 6, 10, 15]
print(f"    Final carry:     {final}")
print()

# Scan for RNN — the primary use case
T, D, H = 20, 8, 16       # sequence length, input dim, hidden dim
key_r = jax.random.PRNGKey(12)
Wx    = jax.random.normal(key_r, (D, H)) * 0.1
Wh    = jax.random.normal(jax.random.fold_in(key_r, 1), (H, H)) * 0.1
b_rnn = jnp.zeros(H)
x_seq = jax.random.normal(jax.random.PRNGKey(13), (T, D))   # [T, D]
h0    = jnp.zeros(H)

rnn_params = {'Wx': Wx, 'Wh': Wh, 'b': b_rnn}

def rnn_cell(h, x_t):
    """Pure RNN step: (h, x_t) → (h_new, h_new)."""
    h_new = jnp.tanh(x_t @ rnn_params['Wx'] + h @ rnn_params['Wh'] + rnn_params['b'])
    return h_new, h_new   # carry = output here

# scan over T timesteps — O(1) XLA nodes regardless of T
h_final, all_hidden = jax.lax.scan(rnn_cell, init=h0, xs=x_seq)

print(f"  RNN over T={T} timesteps via scan:")
print(f"    Input sequence:  {x_seq.shape}  (T, D)")
print(f"    All hidden:      {all_hidden.shape}  (T, H)")
print(f"    Final hidden:    {h_final.shape}  (H,)")
print()

# Python loop vs scan memory: scan has O(1) XLA nodes
# Verify scan produces same result as manual loop
h = h0
h_loop_all = []
for t in range(T):
    h = jnp.tanh(x_seq[t] @ Wx + h @ Wh + b_rnn)
    h_loop_all.append(h)
h_loop_stack = jnp.stack(h_loop_all)

diff = jnp.max(jnp.abs(h_loop_stack - all_hidden))
print(f"  scan vs Python loop max diff: {float(diff):.2e}  ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Pytrees — universal data structure
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Pytrees: tree_map, tree_leaves, custom operations")
print("━" * 65)
print()

params_tree = {
    'embed':   {'W': jnp.ones((100, 16))},
    'layer1':  {'W': jnp.ones((16, 32)), 'b': jnp.zeros(32)},
    'layer2':  {'W': jnp.ones((32, 5)),  'b': jnp.zeros(5)},
}
grads_tree = jax.tree.map(lambda p: p * 0.01, params_tree)   # mock gradients

# SGD update: params - lr * grads — works on any pytree structure
updated = jax.tree.map(lambda p, g: p - 0.1 * g, params_tree, grads_tree)
print(f"  params_tree leaves: {len(jax.tree.leaves(params_tree))}")
print(f"  jax.tree.map(p - lr*g) produces same structure ✅")

# Count total parameters
total = sum(x.size for x in jax.tree.leaves(params_tree))
print(f"  Total parameters: {total:,}")
print()

# Global gradient norm using tree_map + tree_leaves
squares   = jax.tree.map(lambda g: jnp.sum(g ** 2), grads_tree)
sq_leaves = jax.tree.leaves(squares)
gnorm     = jnp.sqrt(sum(sq_leaves))
print(f"  Global gradient norm: {float(gnorm):.4f}")

# Clip gradients to max norm
clip_norm = 1.0
scale     = jnp.minimum(1.0, clip_norm / (gnorm + 1e-8))
clipped   = jax.tree.map(lambda g: g * scale, grads_tree)
gnorm_clipped = jnp.sqrt(sum(jax.tree.leaves(
    jax.tree.map(lambda g: jnp.sum(g**2), clipped)
)))
print(f"  After clipping (max_norm={clip_norm}): {float(gnorm_clipped):.4f}  ✅")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Flax NNX — Modules, Custom Layers & NNX Training Loop": {
        "description": (
            "Complete Flax NNX module system from minimal to advanced. "
            "nnx.Module: Param, BatchStat, Dropout, Rngs. "
            "Sequential and residual MLP with NNX. "
            "Custom NNX module: Scaled Multi-Head Attention from scratch. "
            "nnx.split / nnx.merge / nnx.jit for JIT-compatible training. "
            "nnx.value_and_grad for NNX-native gradient computation. "
            "Optax integration with NNX Optimizer wrapper. "
            "Full NNX training loop with validation and metric tracking. "
            "BatchNorm with training/inference mode switching."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import functools

try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    print(f"  JAX: {jax.__version__}")
    try:
        import flax
        print(f"  Flax: {flax.__version__}")
    except Exception:
        pass
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'jax', 'flax', 'optax', '--quiet', '--prefer-binary'], check=True)
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

print("=" * 65)
print("  FLAX NNX — MODULES, CUSTOM LAYERS & TRAINING LOOP")
print("=" * 65)
print()

key = jax.random.PRNGKey(42)
np.random.seed(42)

# ── Shared dataset ─────────────────────────────────────────────────────
N, FEAT, CLS = 2000, 32, 5
X_np = np.random.randn(N, FEAT).astype('float32')
y_np = np.random.randint(0, CLS, N).astype('int32')
X_tr, X_va = jnp.array(X_np[:1600]), jnp.array(X_np[1600:])
y_tr, y_va = jnp.array(y_np[:1600]), jnp.array(y_np[1600:])
print(f"  Dataset: {len(y_tr)} train / {len(y_va)} val | {FEAT} features | {CLS} classes")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Minimal NNX Module
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Minimal NNX module")
print("━" * 65)
print()

class LinearLayer(nnx.Module):
    """Minimal NNX module: single linear transformation."""
    def __init__(self, in_f, out_f, rngs: nnx.Rngs):
        self.w = nnx.Param(
            jax.random.normal(rngs.params(), (in_f, out_f)) * 0.01
        )
        self.b = nnx.Param(jnp.zeros(out_f))

    def __call__(self, x):
        return x @ self.w.value + self.b.value

linear = LinearLayer(FEAT, 16, rngs=nnx.Rngs(0))
x_test = jnp.ones((4, FEAT))
out    = linear(x_test)
print(f"  LinearLayer output shape: {out.shape}")
print(f"  w shape: {linear.w.value.shape},  b shape: {linear.b.value.shape}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Residual MLP with NNX built-in layers
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Residual MLP with built-in NNX layers")
print("━" * 65)
print()

class ResBlock(nnx.Module):
    """Pre-norm residual block using nnx.Linear, nnx.LayerNorm, nnx.Dropout."""
    def __init__(self, d_model: int, rngs: nnx.Rngs, dropout: float = 0.1):
        self.norm    = nnx.LayerNorm(d_model, rngs=rngs)
        self.dense1  = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dense2  = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout)

    def __call__(self, x, *, deterministic: bool = True):
        skip = x
        x    = self.norm(x)
        x    = jax.nn.gelu(self.dense1(x))
        x    = self.dropout(x, deterministic=deterministic)
        x    = self.dense2(x)
        return x + skip    # residual connection


class ResidualMLP(nnx.Module):
    def __init__(self, in_f, hidden, n_blocks, n_classes, rngs: nnx.Rngs):
        self.embed  = nnx.Linear(in_f, hidden, rngs=rngs)
        self.blocks = [ResBlock(hidden, rngs=rngs) for _ in range(n_blocks)]
        self.norm   = nnx.LayerNorm(hidden, rngs=rngs)
        self.head   = nnx.Linear(hidden, n_classes, rngs=rngs)

    def __call__(self, x, *, deterministic: bool = True):
        x = jax.nn.gelu(self.embed(x))
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        x = self.norm(x)
        return self.head(x)


rngs  = nnx.Rngs(params=0, dropout=1)
model = ResidualMLP(FEAT, hidden=64, n_blocks=3, n_classes=CLS, rngs=rngs)

# Count parameters by inspecting the state
graphdef, state = nnx.split(model)
n_params = sum(x.size for x in jax.tree.leaves(state))
print(f"  ResidualMLP (3 blocks, hidden=64) params: {n_params:,}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: nnx.jit training loop
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — NNX training loop with nnx.jit and Optax")
print("━" * 65)
print()

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3, weight_decay=1e-2))

def cross_entropy(logits, labels):
    log_softmax = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(log_softmax[jnp.arange(len(labels)), labels])

def accuracy(logits, labels):
    return jnp.mean(logits.argmax(-1) == labels)

@nnx.jit   # nnx.jit: handles split/merge of NNX state automatically
def train_step(model, optimizer, x_batch, y_batch):
    def loss_fn(model):
        logits = model(x_batch, deterministic=False)
        return cross_entropy(logits, y_batch)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)   # updates model in-place via NNX variable graph
    return loss

@nnx.jit
def eval_step(model, x_batch, y_batch):
    logits = model(x_batch, deterministic=True)    # deterministic=True → no dropout
    return cross_entropy(logits, y_batch), accuracy(logits, y_batch)

BATCH = 64
EPOCHS = 15

print(f"  Training {EPOCHS} epochs (batch={BATCH}):")
print(f"  {'Epoch':>6} {'Train loss':>12} {'Val loss':>12} {'Val acc':>12}")
print(f"  {'─'*46}")

for epoch in range(EPOCHS):
    # Shuffle training data
    perm   = np.random.permutation(len(y_tr))
    X_shuf = X_tr[perm]
    y_shuf = y_tr[perm]

    train_losses = []
    for i in range(0, len(y_tr), BATCH):
        xb = X_shuf[i:i+BATCH]
        yb = y_shuf[i:i+BATCH]
        loss = train_step(model, optimizer, xb, yb)
        train_losses.append(float(loss))

    val_loss, val_acc = eval_step(model, X_va, y_va)

    if (epoch + 1) % 3 == 0 or epoch == 0:
        print(f"  {epoch+1:>6} {np.mean(train_losses):>12.4f} "
              f"{float(val_loss):>12.4f} {float(val_acc):>12.4f}")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Custom NNX module — Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Custom NNX module: Scaled Dot-Product Attention")
print("━" * 65)
print()

class ScaledDotProductAttention(nnx.Module):
    """
    Multi-head self-attention implemented from scratch as an NNX module.
    Demonstrates: nn.Param arrays, attention math, causal masking option.
    """
    def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        # Projection matrices stored as nnx.Param
        init = lambda key, shape: jax.random.normal(key, shape) * (d_model ** -0.5)
        self.Wq = nnx.Param(init(rngs.params(), (d_model, d_model)))
        self.Wk = nnx.Param(init(rngs.params(), (d_model, d_model)))
        self.Wv = nnx.Param(init(rngs.params(), (d_model, d_model)))
        self.Wo = nnx.Param(init(rngs.params(), (d_model, d_model)))

    def __call__(self, x, mask=None):
        # x: (batch, seq, d_model)
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        def project_and_split(W, x):
            proj = x @ W.value                     # (B, T, D)
            proj = proj.reshape(B, T, H, Dh)       # (B, T, H, Dh)
            return proj.transpose(0, 2, 1, 3)      # (B, H, T, Dh)

        Q = project_and_split(self.Wq, x)
        K = project_and_split(self.Wk, x)
        V = project_and_split(self.Wv, x)

        # Scaled dot-product: (B, H, T, T)
        scale  = Dh ** -0.5
        scores = (Q @ K.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)

        attn    = jax.nn.softmax(scores, axis=-1)
        context = (attn @ V).transpose(0, 2, 1, 3).reshape(B, T, D)
        return context @ self.Wo.value

attn_layer = ScaledDotProductAttention(d_model=32, n_heads=4,
                                        rngs=nnx.Rngs(params=42))
x_seq = jnp.ones((2, 8, 32))   # (batch, seq, d_model)
out_attn = attn_layer(x_seq)
print(f"  ScaledDotProductAttention:")
print(f"    Input:  {x_seq.shape}  (batch, seq, d_model)")
print(f"    Output: {out_attn.shape}  (batch, seq, d_model)")
params_attn = sum(p.size for p in jax.tree.leaves(nnx.split(attn_layer)[1]))
print(f"    Params: {params_attn:,}  (4×d_model² = 4×{32}² = {4*32**2})")
print()

# Causal mask (lower triangular) — for autoregressive models
T_causal = 6
causal_mask = jnp.tril(jnp.ones((T_causal, T_causal), dtype=bool))
causal_mask = causal_mask[None, None, :, :]    # (1, 1, T, T)
x_causal    = jnp.ones((1, T_causal, 32))
out_causal  = attn_layer(x_causal, mask=causal_mask)
print(f"  With causal mask ({T_causal}×{T_causal} lower-triangular):")
print(f"    Output: {out_causal.shape}  ✅  (positions only attend to past)")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Flax Linen + TrainState — Classic API, BatchNorm & Checkpointing": {
        "description": (
            "Complete Flax Linen training pipeline (the original Flax API). "
            "Linen nn.Module with @nn.compact decorator. "
            "model.init: variables dict with params and batch_stats. "
            "model.apply: stateless forward pass with mutable collections. "
            "BatchNorm: training=True vs use_running_average=True pattern. "
            "flax.training.TrainState: bundling params + opt_state. "
            "Extending TrainState with batch_stats field. "
            "jitted train_step with state.apply_gradients + state.replace. "
            "Orbax checkpoint saving and restoration. "
            "Warmup cosine LR schedule with Optax."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import functools
import tempfile
import os
from typing import Any

try:
    import jax
    import jax.numpy as jnp
    import optax
    import flax.linen as nn
    from flax.training import train_state
    import flax
    print(f"  JAX: {jax.__version__}  |  Flax: {flax.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'jax', 'flax', 'optax', '--quiet', '--prefer-binary'], check=True)
    import jax
    import jax.numpy as jnp
    import optax
    import flax.linen as nn
    from flax.training import train_state
    import flax
    print(f"  JAX: {jax.__version__}  |  Flax: {flax.__version__}")

print("=" * 65)
print("  FLAX LINEN + TRAINSTATE — CLASSIC API & FULL PIPELINE")
print("=" * 65)
print()

key = jax.random.PRNGKey(0)
np.random.seed(0)

# ── Dataset ───────────────────────────────────────────────────────────
N, FEAT, CLS = 2400, 64, 8
X_np = np.random.randn(N, FEAT).astype('float32')
y_np = np.random.randint(0, CLS, N).astype('int32')
X_tr, X_va = jnp.array(X_np[:2000]), jnp.array(X_np[2000:])
y_tr, y_va = jnp.array(y_np[:2000]), jnp.array(y_np[2000:])
print(f"  Dataset: {len(y_tr)}/{len(y_va)} train/val | {FEAT} features | {CLS} classes")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Linen module with @nn.compact
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Linen @nn.compact: inline sub-layer definition")
print("━" * 65)
print()

class MLPBlock(nn.Module):
    """
    Linen module using @nn.compact: sub-layers are defined inline in __call__.
    First call creates sub-layers and stores params; subsequent calls reuse them.
    """
    features: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.features)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.features)(x)
        if residual.shape[-1] != self.features:
            residual = nn.Dense(self.features, use_bias=False)(residual)
        return x + residual


class ResNetMLP(nn.Module):
    """Full ResNet-style MLP using setup() for explicit sub-layer declaration."""
    hidden: int
    n_blocks: int
    n_classes: int

    def setup(self):    # setup() is the alternative to @nn.compact
        # Sub-layers declared here are registered as attributes
        self.embed  = nn.Dense(self.hidden)
        self.blocks = [MLPBlock(self.hidden) for _ in range(self.n_blocks)]
        self.norm   = nn.LayerNorm()
        self.head   = nn.Dense(self.n_classes)

    def __call__(self, x, training: bool = False):
        x = nn.gelu(self.embed(x))
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        return self.head(x)


model = ResNetMLP(hidden=128, n_blocks=3, n_classes=CLS)

# Initialise — returns a variables dict
dummy_x    = jnp.ones((1, FEAT))
variables  = model.init(key, dummy_x)
print(f"  variables keys: {list(variables.keys())}")
print(f"  params tree structure:")
for layer, weights in variables['params'].items():
    shapes = {k: v.shape for k, v in weights.items()} if isinstance(weights, dict) else {}
    print(f"    {layer}: {shapes}")
print()

# Count parameters
n_params = sum(x.size for x in jax.tree.leaves(variables['params']))
print(f"  Total params: {n_params:,}")
print()

# model.apply — stateless forward pass
y_test = model.apply(variables, dummy_x, training=False)
print(f"  model.apply output shape: {y_test.shape}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: BatchNorm — the mutable collection pattern
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — BatchNorm with mutable batch_stats collection")
print("━" * 65)
print()

class MLPWithBN(nn.Module):
    """Model with BatchNorm — requires mutable='batch_stats' during training."""
    hidden: int
    n_classes: int

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Dense(self.hidden)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)  # ← key: use_running_average
        x = nn.relu(x)
        x = nn.Dense(self.hidden // 2)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        return nn.Dense(self.n_classes)(x)

bn_model = MLPWithBN(hidden=64, n_classes=CLS)
bn_vars  = bn_model.init(key, dummy_x[:, :FEAT])

print(f"  BatchNorm variables keys: {list(bn_vars.keys())}")
print(f"  batch_stats structure:")
for layer, stats in bn_vars['batch_stats'].items():
    shapes = {k: v.shape for k, v in stats.items()}
    print(f"    {layer}: {shapes}")
print()

# During training: mutable=['batch_stats'] → returns (output, updates)
x_batch = jnp.array(X_np[:8])
y_out, updates = bn_model.apply(
    bn_vars,
    x_batch,
    training    = True,
    mutable     = ['batch_stats'],    # declare which collections are mutable
)
new_batch_stats = updates['batch_stats']
print(f"  Training forward (mutable): y_out shape = {y_out.shape}")
print(f"  Updated batch_stats available ✅")
print()

# During inference: no mutable, use_running_average=True
y_inf = bn_model.apply(
    {'params': bn_vars['params'], 'batch_stats': new_batch_stats},
    x_batch,
    training = False,
)
print(f"  Inference forward (no mutable): y_inf shape = {y_inf.shape}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: TrainState + extended TrainState with batch_stats
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — TrainState: bundled params + opt_state")
print("━" * 65)
print()

# Extended TrainState: add batch_stats field for BatchNorm
class TrainStateWithBN(train_state.TrainState):
    batch_stats: Any   # any pytree — added on top of the base TrainState

# Warmup cosine decay schedule
EPOCHS      = 20
STEPS_EPOCH = len(y_tr) // 64
TOTAL_STEPS = EPOCHS * STEPS_EPOCH

schedule = optax.warmup_cosine_decay_schedule(
    init_value   = 0.0,
    peak_value   = 3e-3,
    warmup_steps = STEPS_EPOCH * 2,   # 2 epoch warmup
    decay_steps  = TOTAL_STEPS,
    end_value    = 1e-6,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=1e-2),
)

# Initialise the full model variables
full_vars = bn_model.init(key, jnp.ones((1, FEAT)))

# Create extended TrainState
state = TrainStateWithBN.create(
    apply_fn    = bn_model.apply,
    params      = full_vars['params'],
    tx          = optimizer,
    batch_stats = full_vars['batch_stats'],
)

print(f"  TrainState fields: step, params, tx, opt_state, batch_stats")
print(f"  Initial step: {state.step}")
print(f"  LR at step 0: {float(schedule(0)):.2e}")
print(f"  LR at warmup peak ({STEPS_EPOCH*2}): {float(schedule(STEPS_EPOCH*2)):.2e}")
print(f"  LR at end ({TOTAL_STEPS}): {float(schedule(TOTAL_STEPS)):.2e}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Full jitted train_step and eval_step
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Full training pipeline with TrainState")
print("━" * 65)
print()

@jax.jit
def train_step(state: TrainStateWithBN, x_batch, y_batch):
    """Pure function: (state, batch) → (new_state, metrics_dict)."""

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x_batch,
            training = True,
            mutable  = ['batch_stats'],
        )
        log_sm = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        loss   = -jnp.mean(log_sm[jnp.arange(len(y_batch)), y_batch])
        return loss, (logits, updates)

    (loss, (logits, updates)), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    acc = jnp.mean(logits.argmax(-1) == y_batch)
    return state, {'loss': loss, 'acc': acc}

@jax.jit
def eval_step(state: TrainStateWithBN, x_batch, y_batch):
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        x_batch,
        training = False,
    )
    log_sm = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    loss   = -jnp.mean(log_sm[jnp.arange(len(y_batch)), y_batch])
    acc    = jnp.mean(logits.argmax(-1) == y_batch)
    return {'loss': float(loss), 'acc': float(acc)}

print(f"  Training {EPOCHS} epochs:")
print(f"  {'Epoch':>6} {'Train loss':>12} {'Train acc':>12} {'Val acc':>12}")
print(f"  {'─'*46}")

for epoch in range(EPOCHS):
    perm   = np.random.permutation(len(y_tr))
    losses, accs = [], []

    for i in range(0, len(y_tr), 64):
        idx    = perm[i:i+64]
        xb, yb = X_tr[idx], y_tr[idx]
        state, metrics = train_step(state, xb, yb)
        losses.append(float(metrics['loss']))
        accs.append(float(metrics['acc']))

    val_m = eval_step(state, X_va, y_va)

    if (epoch + 1) % 4 == 0 or epoch == 0:
        print(f"  {epoch+1:>6} {np.mean(losses):>12.4f} "
              f"{np.mean(accs):>12.4f} {val_m['acc']:>12.4f}")

print()
print(f"  Final step count: {int(state.step)}")
print(f"  Final LR: {float(schedule(state.step)):.2e}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Checkpointing with msgpack (lightweight, no orbax needed)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Checkpointing (flax.serialization)")
print("━" * 65)
print()

with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, 'state.msgpack')

    # Serialize state to bytes (flax built-in, no orbax dependency)
    state_bytes = flax.serialization.to_bytes(state)
    with open(ckpt_path, 'wb') as f:
        f.write(state_bytes)

    size_kb = os.path.getsize(ckpt_path) / 1024
    print(f"  Checkpoint saved: {size_kb:.1f} KB")

    # Restore — provide a target (template) for structure
    with open(ckpt_path, 'rb') as f:
        restored_bytes = f.read()

    restored_state = flax.serialization.from_bytes(state, restored_bytes)

    # Verify restored state matches
    orig_leaves     = jax.tree.leaves(state.params)
    restored_leaves = jax.tree.leaves(restored_state.params)
    max_diff = max(float(jnp.max(jnp.abs(o - r)))
                   for o, r in zip(orig_leaves, restored_leaves))
    print(f"  Restored params max diff: {max_diff:.2e}  ✅")
    print(f"  Restored step: {int(restored_state.step)}")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Advanced JAX — Custom VJP, Sharding, Mixed Precision & Benchmarks": {
        "description": (
            "Advanced JAX features and production patterns. "
            "Custom VJP: straight-through estimator for binary networks. "
            "Gradient checkpointing (jax.checkpoint) for memory reduction. "
            "Mixed precision: bf16 compute with f32 param storage. "
            "Optax gradient transforms: masking, multi-transform, EMA params. "
            "pmap pattern for data-parallel training across devices. "
            "jax.sharding: NamedSharding and Mesh-based data parallelism. "
            "Comprehensive JIT/vmap/grad benchmark against NumPy. "
            "jax.make_jaxpr for inspecting compiled computation graphs."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import functools
from typing import Any

try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    import flax.linen as nn
    from flax.training import train_state
    print(f"  JAX: {jax.__version__}  |  Optax: {optax.__version__}")
    print(f"  Devices: {jax.devices()}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'jax', 'flax', 'optax', '--quiet', '--prefer-binary'], check=True)
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    import flax.linen as nn
    from flax.training import train_state

print("=" * 65)
print("  ADVANCED JAX — CUSTOM VJP, MIXED PRECISION & SHARDING")
print("=" * 65)
print()

key = jax.random.PRNGKey(99)
np.random.seed(99)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Custom VJP — Straight-Through Estimator
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Custom VJP: Straight-Through Estimator")
print("━" * 65)
print()

@jax.custom_vjp
def hard_sigmoid(x):
    """Hard sigmoid: 0 if x<0, 1 if x>1, else x. Not differentiable everywhere."""
    return jnp.clip(x, 0.0, 1.0).round()   # binary output (0 or 1)

def hard_sigmoid_fwd(x):
    out = hard_sigmoid(x)
    return out, x    # residuals = original x (for backward)

def hard_sigmoid_bwd(residuals, g):
    x, = residuals if isinstance(residuals, tuple) else (residuals,)
    # Straight-through: pass gradient through as if function were identity
    # But only where the original input is in [0, 1] (not saturated)
    ste_grad = g * jnp.where((x >= 0) & (x <= 1), 1.0, 0.0)
    return (ste_grad,)

hard_sigmoid.defvjp(hard_sigmoid_fwd, hard_sigmoid_bwd)

x_ste = jnp.array([-0.5, 0.2, 0.7, 1.3, 0.5])
out   = hard_sigmoid(x_ste)
grad  = jax.grad(lambda x: hard_sigmoid(x).sum())(x_ste)

print(f"  Input:    {x_ste.tolist()}")
print(f"  Output:   {out.tolist()}  (binary 0/1)")
print(f"  STE grad: {grad.tolist()}  (1.0 where not saturated, 0 at extremes)")
print()
print("  Use case: binary/ternary weight networks, VQ-VAE codebook commitment.")
print()

# Custom VJP for numerically stable log-sum-exp gradient
@jax.custom_vjp
def stable_logsumexp(x):
    """Numerically stable log-sum-exp: log Σ exp(xᵢ)."""
    c = jnp.max(x)
    return c + jnp.log(jnp.sum(jnp.exp(x - c)))

def stable_logsumexp_fwd(x):
    out = stable_logsumexp(x)
    return out, x   # save x for backward

def stable_logsumexp_bwd(saved, g):
    x, = saved if isinstance(saved, tuple) else (saved,)
    softmax = jax.nn.softmax(x)   # ∂(log Σ exp(x))/∂xᵢ = softmax(x)ᵢ
    return (g * softmax,)

stable_logsumexp.defvjp(stable_logsumexp_fwd, stable_logsumexp_bwd)

x_lse = jnp.array([1.0, 2.0, 3.0])
print(f"  Stable log-sum-exp({x_lse.tolist()}) = {float(stable_logsumexp(x_lse)):.4f}")
print(f"  Gradient = softmax = {jax.grad(stable_logsumexp)(x_lse).round(4).tolist()}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Gradient checkpointing — memory vs compute trade-off
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Gradient checkpointing (rematerialisation)")
print("━" * 65)
print()

D = 64

def deep_layer(params, x):
    """A single layer of a deep network — simulates expensive activation."""
    W, b = params
    return jax.nn.tanh(x @ W + b)

# Build a deep chain (10 layers)
keys    = [jax.random.fold_in(key, i) for i in range(10)]
params_chain = [(jax.random.normal(k, (D, D)) * 0.1, jnp.zeros(D)) for k in keys]
x_chain = jax.random.normal(key, (32, D))

# Without checkpointing: all 10 activations stored in memory during backward
def forward_no_ckpt(params_chain, x):
    for W, b in params_chain:
        x = jax.nn.tanh(x @ W + b)
    return x.sum()

# With checkpointing: recompute activations during backward, save memory
@jax.checkpoint
def checkpointed_layer(params, x):
    W, b = params
    return jax.nn.tanh(x @ W + b)

def forward_with_ckpt(params_chain, x):
    for p in params_chain:
        x = checkpointed_layer(p, x)    # each layer's activation is recomputed
    return x.sum()

# Both produce same gradients
grad_no_ckpt   = jax.jit(jax.grad(forward_no_ckpt))(params_chain, x_chain)
grad_with_ckpt = jax.jit(jax.grad(forward_with_ckpt))(params_chain, x_chain)

flat_no   = jax.tree.leaves(grad_no_ckpt)
flat_ckpt = jax.tree.leaves(grad_with_ckpt)
max_diff  = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(flat_no, flat_ckpt))
print(f"  10-layer chain: checkpointed vs unchecked gradients max diff: {max_diff:.2e}  ✅")
print(f"  Memory saved: checkpointing stores 0 intermediate activations")
print(f"  Cost: each activation recomputed once during backward pass (~2× compute)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Mixed precision — bf16 compute, f32 params
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Mixed precision: bfloat16 compute, float32 params")
print("━" * 65)
print()

class MixedPrecisionModel(nnx.Module):
    """
    Params stored as float32 (stable training).
    Compute happens in bfloat16 (fast Tensor Core on modern GPUs/TPUs).
    Pattern: cast params → bf16 at start of forward, cast outputs → f32.
    """
    def __init__(self, in_f, hidden, out_f, rngs: nnx.Rngs,
                 compute_dtype=jnp.bfloat16):
        self.compute_dtype = compute_dtype
        self.dense1 = nnx.Linear(in_f, hidden, rngs=rngs)
        self.dense2 = nnx.Linear(hidden, out_f, rngs=rngs)

    def __call__(self, x):
        # Cast activations to compute dtype for fast math
        x = x.astype(self.compute_dtype)
        # Weights are float32 internally; jnp matmul will use the compute dtype
        x = jax.nn.gelu(self.dense1(x.astype(jnp.float32)).astype(self.compute_dtype))
        # Cast output back to float32 for loss computation (stable)
        return self.dense2(x.astype(jnp.float32))

mp_model = MixedPrecisionModel(32, 128, 8, rngs=nnx.Rngs(0), compute_dtype=jnp.bfloat16)
x_mp = jax.random.normal(jax.random.PRNGKey(1), (16, 32))  # f32 input
out_mp = mp_model(x_mp)
print(f"  MixedPrecision model:")
print(f"    Input dtype:  {x_mp.dtype}")
print(f"    Output dtype: {out_mp.dtype}  (f32 for stable loss)")
print(f"    Param dtype:  {mp_model.dense1.kernel.value.dtype}  (f32 for stable updates)")
print()

# Show manual per-layer dtype casting pattern (explicit control)
params_f32 = {'W': jax.random.normal(key, (32, 8)), 'b': jnp.zeros(8)}
x_input_f32 = jax.random.normal(jax.random.PRNGKey(2), (4, 32))

def mp_forward(params, x):
    # Cast params to bf16 for compute only
    params_bf16 = jax.tree.map(lambda p: p.astype(jnp.bfloat16), params)
    x_bf16      = x.astype(jnp.bfloat16)
    out_bf16    = x_bf16 @ params_bf16['W'] + params_bf16['b']
    return out_bf16.astype(jnp.float32)   # cast back to f32 before returning

loss_and_grad = jax.value_and_grad(lambda p, x: mp_forward(p, x).sum())
loss_v, grads_v = loss_and_grad(params_f32, x_input_f32)

print(f"  Manual bf16 compute pattern:")
print(f"    Loss dtype:   {loss_v.dtype}  (f32 — stable)")
print(f"    Grads dtype:  {grads_v['W'].dtype}  (f32 — params stay f32)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Optax advanced — masking, multi-transform, EMA
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Optax: masking, chain composition, EMA params")
print("━" * 65)
print()

# Demonstrate optax.chain with gradient clipping + AdamW
tx_composed = optax.chain(
    optax.clip_by_global_norm(1.0),          # first: clip gradient norm
    optax.scale_by_adam(b1=0.9, b2=0.999),  # second: Adam second-moment scaling
    optax.add_decayed_weights(weight_decay=1e-2),  # third: weight decay
    optax.scale(-3e-4),                       # fourth: apply lr (negative for descent)
)

params_demo = {'W': jnp.ones((8, 4)), 'b': jnp.zeros(4)}
opt_state   = tx_composed.init(params_demo)

# Simulate a gradient update
grads_demo  = {'W': jnp.ones((8, 4)) * 0.1, 'b': jnp.ones(4) * 0.1}
updates, new_state = tx_composed.update(grads_demo, opt_state, params_demo)
new_params  = optax.apply_updates(params_demo, updates)

print(f"  Composed optax.chain (clip → adam → weight_decay → scale):")
print(f"    params['W'][0,0]: {params_demo['W'][0,0]:.6f}")
print(f"    update['W'][0,0]: {float(updates['W'][0,0]):.6f}")
print(f"    new_params['W'][0,0]: {float(new_params['W'][0,0]):.6f}")
print()

# optax masked: apply weight decay only to weight matrices, not biases
no_decay_mask = jax.tree.map(lambda p: p.ndim == 1, params_demo)
# True = apply to bias (ndim==1), False = don't apply here
# We want to SKIP bias → invert for add_decayed_weights
weight_decay_mask = jax.tree.map(lambda m: not m, no_decay_mask)

tx_masked = optax.chain(
    optax.adamw(3e-4, weight_decay=1e-2, mask=weight_decay_mask),
)
print(f"  Masked weight decay (skips bias):")
print(f"    mask (apply WD): W={weight_decay_mask['W']}, b={weight_decay_mask['b']}")
print()

# EMA of parameters (model averaging for better generalisation)
ema_transform = optax.ema(decay=0.999)
ema_state     = ema_transform.init(params_demo)

# Simulate 5 steps of EMA tracking
p = jax.tree.map(lambda x: x.copy(), params_demo)
for step in range(5):
    p = jax.tree.map(lambda x: x + 0.01, p)   # simulate param updates
    _, ema_state = ema_transform.update(p, ema_state)

ema_params = ema_state.ema   # smoothed parameter estimate
diff = float(jnp.max(jnp.abs(p['W'] - ema_params['W'])))
print(f"  EMA params after 5 steps:")
print(f"    params['W'][0,0]:     {float(p['W'][0,0]):.6f}")
print(f"    ema_params['W'][0,0]: {float(ema_params['W'][0,0]):.6f}")
print(f"    Diff (EMA lags):      {diff:.6f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: pmap pattern and sharding concepts
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — pmap and sharding patterns")
print("━" * 65)
print()

n_devices = jax.local_device_count()
print(f"  Local device count: {n_devices}")
print()

if n_devices >= 2:
    # Data-parallel training with pmap
    params_rep = jax.tree.map(
        lambda p: jnp.stack([p] * n_devices), params_demo
    )

    @functools.partial(jax.pmap, axis_name='batch')
    def parallel_step(params, x):
        def loss_fn(params):
            return jnp.sum((x @ params['W'] + params['b']) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        # AllReduce: average gradients across all devices
        grads = jax.lax.pmean(grads, axis_name='batch')
        return loss, grads

    x_sharded = jnp.ones((n_devices, 4, 8))   # [N_dev, B, D]
    losses_p, grads_p = parallel_step(params_rep, x_sharded)
    print(f"  pmap data-parallel step:")
    print(f"    Input shape: {x_sharded.shape}  (devices, batch, features)")
    print(f"    Loss per device: {[f'{float(l):.4f}' for l in losses_p]}")
    print(f"    Grads synced across devices via pmean ✅")
else:
    print("  pmap requires ≥2 devices. Showing pattern only:")
    PMAP_PATTERN = """
    # ── pmap data-parallel training ──────────────────────────────────
    @functools.partial(jax.pmap, axis_name='batch')
    def parallel_train_step(params, opt_state, x, y):
        def loss_fn(params):
            logits = model_apply(params, x)
            return cross_entropy(logits, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        # AllReduce: average gradients across all N devices
        grads = jax.lax.pmean(grads, axis_name='batch')
        loss  = jax.lax.pmean(loss,  axis_name='batch')

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Replicate params + opt_state to all devices
    params_rep    = flax.jax_utils.replicate(params)
    opt_state_rep = flax.jax_utils.replicate(opt_state)

    # Shard batch along first axis (one shard per device)
    x_sharded = x.reshape(n_devices, -1, *x.shape[1:])   # [N_dev, B/N, ...]

    # Identical API to non-pmap version
    params_rep, opt_state_rep, losses = parallel_train_step(
        params_rep, opt_state_rep, x_sharded, y_sharded
    )
    """
    print(PMAP_PATTERN)

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Comprehensive benchmark — NumPy vs JAX across sizes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Benchmark: NumPy vs JAX eager vs JAX jit")
print("━" * 65)
print()

def numpy_forward(W, x):
    return np.maximum(x @ W, 0)

def jax_forward(W, x):
    return jax.nn.relu(x @ W)

jax_forward_jit = jax.jit(jax_forward)

print(f"  {'Size':>8} {'NumPy ms':>12} {'JAX eager ms':>14} {'JAX jit ms':>12} {'Speedup':>10}")
print(f"  {'─'*60}")

for size in [64, 256, 512, 1024]:
    W_np = np.random.randn(size, size).astype('float32')
    x_np = np.random.randn(256, size).astype('float32')
    W_jx = jnp.array(W_np)
    x_jx = jnp.array(x_np)

    # Warmup JIT
    jax_forward_jit(W_jx, x_jx).block_until_ready()

    N_REPS = 50

    t0 = time.perf_counter()
    for _ in range(N_REPS): _ = numpy_forward(W_np, x_np)
    t_np = (time.perf_counter() - t0) / N_REPS * 1000

    t0 = time.perf_counter()
    for _ in range(N_REPS): jax_forward(W_jx, x_jx).block_until_ready()
    t_eager = (time.perf_counter() - t0) / N_REPS * 1000

    t0 = time.perf_counter()
    for _ in range(N_REPS): jax_forward_jit(W_jx, x_jx).block_until_ready()
    t_jit = (time.perf_counter() - t0) / N_REPS * 1000

    speedup = t_np / t_jit
    print(f"  {size:>8} {t_np:>12.3f} {t_eager:>14.3f} {t_jit:>12.3f} {speedup:>10.2f}×")

print()
print("  JAX jit includes async dispatch overhead; block_until_ready() measures")
print("  true compute time. On GPU, speedup is dramatically larger (10–100×).")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 7: jax.make_jaxpr — inspect the compiled graph
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 7 — jax.make_jaxpr: inspect compiled representations")
print("━" * 65)
print()

def simple_fn(x, w):
    return jnp.sum(jax.nn.relu(x @ w))

W_small = jnp.ones((4, 2))
x_small = jnp.ones((3, 4))

jaxpr = jax.make_jaxpr(simple_fn)(x_small, W_small)
jaxpr_str = str(jaxpr)
lines = [l for l in jaxpr_str.split('\\n') if l.strip()]
print(f"  simple_fn Jaxpr ({len(lines)} lines):")
for line in lines[:8]:
    print(f"    {line}")
if len(lines) > 8:
    print(f"    ... ({len(lines)-8} more lines)")
print()
print("  The Jaxpr shows:")
print("    - invars:  input symbolic variables (shapes + dtypes)")
print("    - outvars: output symbolic variables")
print("    - eqns:    primitive equations (dot_general, reduce_sum, etc.)")
print("  XLA compiles the Jaxpr to machine code — this is the 'graph'.")
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