"""
MXNet — Apache's Scalable Deep Learning Framework
===================================================

Apache MXNet is a deep learning framework engineered from the ground up for
efficiency, portability, and scale. Originally developed at Carnegie Mellon
University, the University of Washington, and multiple other institutions,
it became the framework of choice for Amazon Web Services and powered
production systems at AWS, Baidu, and Microsoft before the rise of PyTorch 2.x.

MXNet's engineering contributions to the field are substantial and often
underappreciated: the parameter server architecture for distributed training
was pioneered here, the Gluon API introduced a hybrid eager/graph paradigm
before PyTorch or TF2 had solved it cleanly, and MXNet's memory efficiency
made it the only framework capable of running large models on single machines
at a time when GPU memory was scarce.

Understanding MXNet means understanding the design choices that PyTorch and
TensorFlow later adopted — and the unique strengths that still make MXNet
relevant in production at AWS (SageMaker) and for edge/IoT deployment
through TVM and the Model Zoo.

This module covers MXNet's dual ndarray/symbol system, the Gluon API,
the parameter server, mixed precision, and the complete AWS/SageMaker
deployment workflow.

"""

import textwrap
import re

TOPIC_NAME   = "MXNet — Apache's Scalable Deep Learning Framework"
DISPLAY_NAME = "19 · MXNet"
ICON         = "⚙️"
SUBTITLE     = "Hybrid Eager/Graph, Parameter Servers, and AWS-Native Scaling"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — MXNET'S HISTORY, CONTRIBUTIONS AND CURRENT POSITION

### Origins and Key Milestones

    2015:   MXNet published by Tianqi Chen, Mu Li, et al. at CMU and UW.
            Core insight: a single framework should support both symbolic
            (static graph, compilable) and imperative (eager) execution.
    2016:   Amazon Web Services adopts MXNet as its official DL framework.
            MXNet powers AWS DeepLens, SageMaker, and Amazon's internal ML.
    2017:   Gluon API introduced — a high-level, Python-native API that
            offered eager execution in MXNet before TF2 existed.
            This was a direct influence on TensorFlow 2.0's design.
    2017:   Apache Software Foundation incubates MXNet → Apache MXNet.
    2019:   MXNet 1.6 with Horovod integration, full Python 3 support.
    2021:   MXNet 2.0 — unified NumPy-compatible API, simplified backends.
    2023:   AWS announces reduced MXNet investment; community transitions
            some workloads to PyTorch, but MXNet remains in production at
            large scale within AWS infrastructure.

### MXNet's Engineering Contributions

    1. HYBRID COMPUTATION (Gluon's HybridBlock):
        Before PyTorch 2.0's torch.compile and before TF2's @tf.function,
        MXNet's HybridBlock solved the eager-vs-compiled dilemma:
        - Write code eagerly (easy to debug)
        - Call .hybridize() once → runs as a compiled static graph (fast)
        - Same Python code, two execution modes
        This influenced TF2's tf.function and was the design that
        PyTorch 2.0's torch.compile eventually matched.

    2. PARAMETER SERVER ARCHITECTURE:
        MXNet's distributed training via a parameter server (PS) was the
        reference implementation for asynchronous gradient descent at scale.
        The PS architecture separates workers (compute gradients) from
        servers (store and update parameters).
        This enabled training on 1000s of machines before NCCL AllReduce
        became the standard approach.

    3. MEMORY EFFICIENCY:
        MXNet pioneered memory-sharing across computation graph nodes —
        tensors that are no longer needed are immediately reused.
        At a time when a GTX 980 Ti had 6 GB VRAM, this enabled training
        models that would OOM in every other framework.

    4. CROSS-PLATFORM PORTABILITY:
        MXNet's C++ backend compiles for: Linux, Windows, macOS, Android,
        iOS, Raspberry Pi, and bare-metal MCUs.
        TVM integration makes MXNet models deployable on any hardware target.

### Where MXNet Is Used Today

    ┌──────────────────────────────────────────────────────────────────┐
    │  AWS SageMaker:    MXNet is a first-class built-in framework     │
    │  AWS DeepLens:     edge computer vision device runs MXNet        │
    │  AWS Inferentia:   MXNet models compiled via AWS Neuron          │
    │  Amazon internal:  product recommendations, search ranking       │
    │  GluonCV/NLP:      production-ready model zoos for CV and NLP    │
    │  TVM ecosystem:    MXNet → TVM → any hardware target             │
    │  IoT/edge:         ARM microcontrollers via TVM compiled models  │
    └──────────────────────────────────────────────────────────────────┘


##### PART 2 — THE DUAL ENGINE: NDArray AND SYMBOL

### Two Execution Paradigms in One Framework

    MXNet was designed from the start to support BOTH computation styles:

    NDArray (Imperative / Eager):
        Operations execute immediately, just like NumPy.
        Each op returns a concrete value.
        Debug with Python print(), inspect intermediate results.
        The programming model feels like Python.

        import mxnet as mx
        a = mx.nd.array([[1, 2], [3, 4]])
        b = mx.nd.array([[5, 6], [7, 8]])
        c = mx.nd.dot(a, b)   # executes immediately
        print(c.asnumpy())    # → [[19, 22], [43, 50]]

    Symbol (Declarative / Graph):
        Builds a computation graph WITHOUT executing it.
        The graph is then compiled and optimised.
        Portable across languages (Python, Scala, R, Julia).
        No Python overhead at inference time.

        x = mx.sym.Variable('data')
        W = mx.sym.Variable('weight')
        y = mx.sym.dot(x, W)    # no computation — just a graph node
        # Must bind to an Executor with actual data to run

### Why Both Exist: The Fundamental Tradeoff

    Imperative (NDArray):
        PRO:  Easy to debug, natural Python control flow, flexible.
        CON:  Python overhead per operation, graph not globally visible
              for cross-op optimisations.

    Symbolic (Symbol):
        PRO:  Global graph visibility → layer fusion, memory reuse.
        CON:  Hard to debug (errors come from graph execution, not Python).
              Dynamic control flow is awkward (must use special ops).

    PyTorch chose imperative-only, accepting the performance trade-off.
    TensorFlow 1.x chose symbolic-only, accepting the usability trade-off.
    MXNet was the first to offer BOTH and let the user choose — or to
    write code once and transparently switch between them (HybridBlock).

### NDArray Internals

    mx.nd.NDArray is MXNet's primary tensor class:
        - Backed by C++ storage, device-agnostic (CPU, GPU, multiple GPUs)
        - Supports lazy evaluation: operations can be queued and fused
        - All operations are ASYNCHRONOUS by default — the CPU schedules
          work and the GPU executes it; Python continues immediately.
        - .wait_to_read() or .asnumpy() forces synchronisation.

    Device management:
        cpu_arr = mx.nd.array([1, 2, 3])                     # mx.cpu()
        gpu_arr = mx.nd.array([1, 2, 3], ctx=mx.gpu(0))      # first GPU
        gpu1_arr = mx.nd.array([1, 2, 3], ctx=mx.gpu(1))     # second GPU
        ctx = mx.cpu() if mx.context.num_gpus() == 0 else mx.gpu(0)

    Asynchronous execution example:
        a = mx.nd.random.normal(shape=(1000, 1000))
        b = mx.nd.random.normal(shape=(1000, 1000))
        c = mx.nd.dot(a, b)    # Python continues; GPU queues the work
        # c is a "promise" at this point — computation may not be done
        c_np = c.asnumpy()     # THIS forces synchronisation — wait until done

### MXNet 2.0: The NumPy-Compatible API

    MXNet 2.0 introduced mxnet.numpy (np) and mxnet.numpy_extension (npx):
        import mxnet.numpy as np
        import mxnet.numpy_extension as npx

        a = np.array([1.0, 2.0, 3.0])        # drop-in NumPy replacement
        b = np.dot(a, a)                      # runs on GPU if available
        npx.set_np()                          # enable NumPy compatibility mode

    This was a strategic move to make the transition from NumPy to GPU
    computing trivial — write NumPy code, run it on a GPU cluster.


##### PART 3 — GLUON: MXNET'S HIGH-LEVEL API

### What Gluon Is

    Gluon is MXNet's high-level neural network API, introduced in 2017.
    It was co-developed by Amazon and Microsoft Research.
    Its design directly influenced Keras 2.x and PyTorch's nn.Module system.

    Gluon's three core design principles:
        1. Friendly:   Write neural networks in intuitive Python, with
                       real Python control flow.
        2. Fast:       Optional hybridization converts Python code to a
                       compiled graph for production speed.
        3. Flexible:   Build any architecture — from standard CNNs to
                       custom differentiable programs.

### gluon.nn.Block — The Base Class

    Every Gluon model and layer inherits from gluon.nn.Block:

        class MyLayer(gluon.nn.Block):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                # Parameters registered via self.params or sub-blocks
                with self.name_scope():             # namespacing for params
                    self.dense = gluon.nn.Dense(units)

            def forward(self, x):
                return mx.nd.relu(self.dense(x))

    Key features:
        - Parameters are ParameterDict objects (lazy, deferred initialisation)
        - Sub-blocks are automatically discovered and their params included
        - .collect_params() returns all parameters recursively
        - .save_parameters(path) / .load_parameters(path) for checkpointing

### Deferred (Lazy) Initialisation

    One of Gluon's most distinctive features:
    You can define a layer WITHOUT specifying the input dimension.
    MXNet infers it from the first batch of data.

        net = gluon.nn.Sequential()
        net.add(gluon.nn.Dense(128, activation='relu'))  # no input_dim!
        net.add(gluon.nn.Dense(64,  activation='relu'))
        net.add(gluon.nn.Dense(10))

        # Parameters are UNINITIALIZED at this point
        net.initialize(mx.init.Xavier())    # Xavier init registered but not applied

        # FIRST forward pass triggers shape inference and actual initialisation:
        X = mx.nd.random.normal(shape=(32, 784))
        output = net(X)   # Gluon infers Dense(784→128), etc. HERE

    Why this matters:
        - Define architectures without knowing input shape upfront
        - Build general-purpose blocks that adapt to any input
        - PyTorch's LazyLinear is a later, partial adoption of this idea

### HybridBlock — Hybrid Eager/Graph Execution

    HybridBlock is MXNet's killer feature: write eager Python, then
    call .hybridize() to compile it to a static computation graph.

        class MyNet(gluon.nn.HybridBlock):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                with self.name_scope():
                    self.fc1 = gluon.nn.Dense(128, activation='relu')
                    self.fc2 = gluon.nn.Dense(10)

            def hybrid_forward(self, F, x):
                # F is mx.nd (eager) or mx.sym (symbolic) — same code!
                x = self.fc1(x)
                return self.fc2(x)

        net = MyNet()
        net.initialize()

        # Mode 1: Eager (debug-friendly, Python overhead per op)
        output = net(X)           # uses mx.nd, Python control flow active

        # Mode 2: Compiled (fast, no Python overhead)
        net.hybridize()           # compile on next forward call
        output = net(X)           # traces with mx.sym, builds static graph
        output = net(X)           # runs compiled graph — very fast

    The F argument in hybrid_forward:
        In eager mode:  F = mxnet.ndarray  (real computation)
        In graph mode:  F = mxnet.symbol   (graph construction)
        Code is IDENTICAL — one implementation, two execution modes.

    This was the cleanest solution to the eager/graph dilemma until
    torch.compile appeared in 2022.

### Exporting Hybridized Models

    After hybridization, the static graph can be exported as a portable
    binary that runs WITHOUT Python:

        net.hybridize()
        net(dummy_input)                      # trigger graph tracing
        net.export("my_model", epoch=0)
        # Creates: my_model-0000.params (weights)
        #          my_model-symbol.json  (graph)

        # Load in ANY language (Python, Scala, Java, R, Go, C++)
        sym, arg_params, aux_params = mx.model.load_checkpoint("my_model", 0)
        # Deploy on mobile, edge, servers — no Python required


##### PART 4 — AUTOGRAD: MXNET'S DIFFERENTIATION ENGINE

### mxnet.autograd — Record and Backward

    MXNet's autograd API is similar in concept to TensorFlow's GradientTape
    but uses a context manager called autograd.record():

        with autograd.record():
            output = net(X)
            loss   = loss_fn(output, y)
        loss.backward()              # computes gradients

    Key differences from PyTorch:
        1. No requires_grad flag on tensors — all NDArrays can be differentiated.
        2. The recording context must be EXPLICITLY entered.
        3. backward() is called on the loss, not via the graph implicitly.

### Training Mode vs Prediction Mode

    In MXNet, layers like Dropout and BatchNorm need to know if they're
    training or predicting. This is done through the autograd context:

        # Training mode: dropout active, BN uses batch statistics
        with autograd.record():
            output = net(X)

        # Prediction mode: dropout inactive, BN uses running statistics
        output = net(X)          # outside autograd.record() → prediction mode

    Alternatively, call net.hybridize() with train=True/False, or use
    autograd.is_training() inside forward() to branch:

        def hybrid_forward(self, F, x):
            if autograd.is_training():
                x = F.Dropout(x, p=0.5)
            return self.fc(x)

### Gradient Computation and update()

    Full MXNet training step pattern:

        trainer = gluon.Trainer(
            net.collect_params(),
            'adam',
            {'learning_rate': 1e-3, 'wd': 1e-4}
        )

        with autograd.record():
            output = net(data)
            loss   = loss_fn(output, label)
        loss.backward()
        trainer.step(batch_size)     # scales gradients by 1/batch_size, applies update

    trainer.step(batch_size):
        Divides gradients by batch_size (normalises for varying batch sizes),
        then applies the optimiser update rule.
        This differs from PyTorch/TF where the loss is already mean-reduced.


##### PART 5 — DISTRIBUTED TRAINING: THE PARAMETER SERVER

### What a Parameter Server Is

    The parameter server (PS) architecture separates the cluster into:
        WORKERS:  compute forward and backward passes, produce gradients.
        SERVERS:  store model parameters, aggregate gradients, apply updates.

    Workflow:
        1. Worker pulls current parameters from servers.
        2. Worker computes forward + backward on its data shard.
        3. Worker pushes gradients to servers.
        4. Server aggregates gradients (sum or average).
        5. Server applies optimizer update → new parameters.
        6. Goto 1.

    Diagram — Parameter Server topology:

        Worker 0  ←→  |               |  ←→ GPU 0
        Worker 1  ←→  | Parameter     |  ←→ GPU 1
        Worker 2  ←→  | Server(s)     |  ←→ GPU 2
        Worker 3  ←→  |               |  ←→ GPU 3
                       (hold params,
                        aggregate grads)

### Synchronous vs Asynchronous PS

    Synchronous (BSP — Bulk Synchronous Parallel):
        Server waits for gradients from ALL workers before updating.
        All workers see the same model version — mathematically equivalent
        to single-GPU training with a larger batch.
        Bottleneck: the slowest worker ("straggler") delays everyone.

    Asynchronous (ASP — Asynchronous Parallel):
        Server updates parameters immediately when ANY worker pushes gradients.
        Workers pull the latest parameters without waiting for others.
        Faster wall-clock time — no straggler delay.
        Trade-off: "stale gradients" — worker updates based on old parameters.
        More training steps needed; convergence is noisier.

    Stale gradient problem:
        Worker 0 pulls params at step t.
        Worker 0 computes gradient ∇L(θₜ).
        Meanwhile, server updates params to θₜ₊₃ (from other workers).
        Worker 0 pushes ∇L(θₜ) against θₜ₊₃ → effectively wrong gradient.
        With staleness τ = 3: gradient is based on params 3 steps behind.

### MXNet KVStore — The PS Implementation

    MXNet's KVStore is the key-value store that implements the PS:

        kv = mx.kv.create('dist_sync')   # synchronous PS
        kv = mx.kv.create('dist_async')  # asynchronous PS
        kv = mx.kv.create('local')       # multi-GPU, single machine

    The KVStore API:
        kv.init(key, value)     # initialise a key with a parameter tensor
        kv.push(key, grads)     # send gradients to server
        kv.pull(key, params)    # retrieve latest parameters from server

    Gluon's Trainer uses KVStore internally — usually transparent to the user.
    For multi-machine training, set the kvstore type:
        trainer = gluon.Trainer(params, 'sgd', {'learning_rate': 0.01},
                                 kvstore='dist_sync')

### Modern Alternative: Horovod + MXNet

    Horovod (developed at Uber) provides AllReduce-based distributed training
    for MXNet, TensorFlow, and PyTorch with a uniform API:

        import horovod.mxnet as hvd
        hvd.init()
        ctx = mx.gpu(hvd.local_rank())   # each process gets its GPU

        trainer = gluon.Trainer(net.collect_params(), optimizer,
                                 kvstore=None)
        # Wrap trainer with Horovod's DistributedTrainer:
        trainer = hvd.DistributedTrainer(net.collect_params(), optimizer)

    Horovod is now preferred over the native PS for MXNet because:
        - AllReduce (ring-based) is more bandwidth-efficient than PS
        - Unified API across frameworks (same code for MXNet, PyTorch, TF)
        - Better NCCL integration for NVLink clusters


##### PART 6 — MIXED PRECISION AND MEMORY OPTIMISATION

### AMP in MXNet (Automatic Mixed Precision)

    MXNet introduced AMP before TensorFlow 2.x with a very clean API:

        from mxnet.contrib import amp
        amp.init()                                    # enable AMP globally
        # From this point: eligible ops run in FP16 automatically

        with autograd.record():
            with amp.scale_loss(loss, trainer) as scaled_loss:
                scaled_loss.backward()
        trainer.step(batch_size)

    Key components:
        amp.init():         registers FP16-safe ops (matmul, conv) and
                            keeps FP32-required ops (softmax, batch norm update).
        scale_loss:         multiplies loss by a dynamic scale factor
                            to prevent FP16 underflow during backward.

### Memory Pool and Workspace

    MXNet uses a memory pool to avoid repeated malloc/free:
        mx.nd.waitall()                   # sync all pending GPU ops
        mx.nd.free_memory(ctx)            # free unused pooled memory

    Workspace: temporary buffer for convolution computations.
    Larger workspace → more algorithm choices → potentially faster convolution.
        mx.set_np_shape(1)                # enable variable-shape arrays
        # workspace controlled via environment: MXNET_GPU_WORK_SPACE_TYPE

### Memory-Efficient Training Techniques

    Gradient checkpointing (recompute):
        MXNet supports SymbolBlock with memory-efficient backprop.
        Checkpointing divides the graph into segments, recomputing
        activations segment by segment during backward.

    Inplace operations:
        Some MXNet ops support inplace computation (reuse input memory):
        mx.nd.relu(data, out=data)    # write relu output back to input
        This eliminates intermediate tensor allocation.

    Operator fusion:
        When a model is hybridized, MXNet's graph executor automatically
        fuses compatible operations (conv+bn+relu → single kernel).
        Fusion reduces memory bandwidth and kernel launch overhead.


##### PART 7 — THE GLUONCV AND GLUONNLP MODEL ZOOS

### GluonCV — Computer Vision Toolkit

    GluonCV provides production-ready implementations and pre-trained weights
    for major CV tasks, all built in MXNet/Gluon:

    Object detection:
        - SSD (Single Shot Detector), YOLOv3, Faster R-CNN, DETR
        - Pre-trained on COCO, Pascal VOC, ImageNet

    Image classification:
        - ResNet (all variants), VGG, DenseNet, MobileNet, EfficientNet
        - Calibrated uncertainty — not just accuracy

    Semantic segmentation:
        - FCN, DeepLab v3+, PSPNet, ICNet

    Instance segmentation:
        - Mask R-CNN

    Pose estimation:
        - Simple Baselines, AlphaPose

    Action recognition (video):
        - SlowFast, C3D, TSN, I3D

    Usage:
        import gluoncv
        net = gluoncv.model_zoo.get_model('resnet50_v1', pretrained=True)
        net.hybridize()      # compile for inference speed
        net.export("resnet50_v1")

### GluonNLP — NLP Toolkit

    GluonNLP mirrors GluonCV for NLP tasks:

    Pre-trained language models:
        - BERT (base, large), RoBERTa, ALBERT, XLNet, GPT-2
        - Pre-trained on: Wikipedia + Books, CC-News, OpenWebText

    Datasets:
        - GLUE, SQuAD, WMT, CoNLL, SST

    Usage:
        from gluonnlp.model import BERTModel, BERTEncoder
        model, vocab = gluonnlp.model.get_model('bert_12_768_12',
                                                  dataset_name='book_corpus_wiki_en_uncased',
                                                  pretrained=True)


##### PART 8 — DEPLOYMENT: AWS SAGEMAKER, INFERENTIA AND TVM

### AWS SageMaker — Native MXNet Integration

    SageMaker provides managed infrastructure for training and serving.
    MXNet is a first-class citizen with pre-built containers.

    Training on SageMaker:
        from sagemaker.mxnet import MXNet as SageMakerMXNet

        estimator = SageMakerMXNet(
            entry_point='train.py',          # your training script
            role=iam_role,
            instance_type='ml.p3.2xlarge',   # 1 NVIDIA V100
            framework_version='1.9.0',
            py_version='py3',
            hyperparameters={'epochs': 20, 'batch_size': 64},
        )
        estimator.fit({'train': s3_train_uri, 'val': s3_val_uri})

    Multi-GPU training:
        instance_type='ml.p3.16xlarge'        # 8 V100s
        distribution={'parameter_server': {'enabled': True}}

    Serving on SageMaker:
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge',
        )
        result = predictor.predict(data)

### AWS Inferentia — Custom ML Inference Chip

    AWS Inferentia is Amazon's custom ASIC for ML inference.
    Up to 40× lower cost and 4× higher throughput vs GPU for inference.
    Uses AWS Neuron SDK — MXNet models compile to Inferentia via:

        import mx.neuron as mx_neuron
        net.hybridize()
        net(dummy)                          # trace
        net.export("model")                 # standard MXNet export
        # Neuron compiler takes model-symbol.json + model-0000.params
        # and produces a Neuron-optimised binary

### Apache TVM — Cross-Platform Compilation

    TVM is a compiler stack for deep learning models:
        - Input: MXNet Relay IR (or ONNX, TensorFlow, PyTorch)
        - Output: optimised native code for ANY target hardware

    Supported targets:
        NVIDIA GPU (CUDA), AMD GPU (ROCm), Intel CPU (AVX2/AVX512),
        ARM CPU (Cortex-A, Cortex-M), Mali GPU, FPGA, custom accelerators.

    MXNet → TVM workflow:
        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor

        # Import MXNet model to TVM's relay IR
        shape_dict = {"data": (1, 3, 224, 224)}
        mod, params = relay.frontend.from_mxnet(block, shape_dict)

        # Compile for target
        target  = tvm.target.Target("llvm -mcpu=cascadelake")  # Intel CPU
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        # Deploy
        dev  = tvm.cpu(0)
        gmod = graph_executor.GraphModule(lib["default"](dev))
        gmod.set_input("data", tvm.nd.array(img))
        gmod.run()
        output = gmod.get_output(0).numpy()

    TVM's AutoTVM / AutoScheduler auto-tunes kernel parameters:
        - Matrix multiplication tile sizes
        - Loop unrolling factors
        - Memory access patterns
        Tuning finds the optimal implementation for YOUR specific hardware.

### Deployment Format Comparison

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Format          │ Runtime needed │ Target           │ Best for       │
    ├──────────────────────────────────────────────────────────────────────┤
    │ MXNet symbol    │ MXNet C++ lib  │ CPU, GPU, cloud  │ General serve  │
    │ ONNX            │ ONNX Runtime   │ Platform-agnostic│ Cross-framework│
    │ TVM compiled    │ TVM runtime    │ Any hardware     │ Edge, custom   │
    │ Neuron binary   │ AWS Neuron     │ Inferentia only  │ AWS cloud      │
    │ TFLite (via ONNX│ TFLite runtime │ Mobile, MCU      │ On-device      │
    └──────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · NDArray, Autograd & the Gluon Training Loop": {
        "description": (
            "MXNet fundamentals from the ground up. "
            "NDArray creation, device management, and asynchronous execution. "
            "mxnet.autograd.record() for differentiation. "
            "Full Gluon training loop: Block, Trainer, loss, metrics. "
            "Deferred initialisation demonstrated. "
            "Compare the autograd API with PyTorch and TensorFlow."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  MXNET NDARRAY, AUTOGRAD & GLUON TRAINING LOOP")
print("=" * 65)
print()

try:
    import mxnet as mx
    from mxnet import gluon, autograd, nd
    from mxnet.gluon import nn
    print(f"  MXNet version: {mx.__version__}")
except ImportError:
    print("  MXNet not installed. Install with:")
    print("  pip install mxnet          (CPU)")
    print("  pip install mxnet-cu118    (CUDA 11.8)")
    print()
    print("  This module shows MXNet patterns annotated for study.")
    print("  All code is syntactically correct and will run with MXNet installed.")
    print()
    # ── Annotated reference code follows ─────────────────────────────────
    NDARRAY_REFERENCE = """
# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: NDArray — MXNet's tensor type
# ─────────────────────────────────────────────────────────────────────────
import mxnet as mx
from mxnet import nd

# Device context
ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

# Creation
a = nd.array([[1.0, 2.0], [3.0, 4.0]], ctx=ctx)
b = nd.zeros((3, 4), ctx=ctx)
c = nd.random.normal(shape=(2, 3), ctx=ctx)

print(f"a = {a}")
print(f"a.shape = {a.shape}")
print(f"a.dtype = {a.dtype}")
print(f"a.context = {a.context}")

# Operations — all asynchronous by default!
d = nd.dot(a, a.T)    # Python returns immediately; GPU queues the work
# d is a "pending" computation at this point

e = d.asnumpy()       # THIS forces synchronisation — wait until GPU finishes
print(f"a @ a.T = {e}")

# Asynchronous execution is a performance feature:
# Python can submit many ops to the GPU queue without waiting.
# This keeps the GPU busy continuously rather than waiting for Python.

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: MXNet vs PyTorch vs TensorFlow — autograd comparison
# ─────────────────────────────────────────────────────────────────────────

# PyTorch style:
import torch
x_pt = torch.tensor(3.0, requires_grad=True)  # mark for grad tracking
y_pt = x_pt ** 2
y_pt.backward()
print(f"PyTorch dy/dx at x=3: {x_pt.grad}")    # 6.0

# TensorFlow style:
import tensorflow as tf
x_tf = tf.Variable(3.0)                         # Variable auto-tracked
with tf.GradientTape() as tape:
    y_tf = x_tf ** 2
print(f"TF dy/dx at x=3: {tape.gradient(y_tf, x_tf)}")  # 6.0

# MXNet style:
from mxnet import autograd
x_mx = nd.array([3.0])
x_mx.attach_grad()                              # attach gradient buffer
with autograd.record():                         # explicit recording scope
    y_mx = x_mx ** 2
y_mx.backward()
print(f"MXNet dy/dx at x=3: {x_mx.grad}")      # 6.0

# Summary:
# PyTorch: requires_grad=True flag on tensor → implicit recording
# TF:      explicit GradientTape scope + Variable auto-watched
# MXNet:   attach_grad() + explicit autograd.record() scope
"""
    print("  NDARRAY AND AUTOGRAD REFERENCE CODE:")
    print(NDARRAY_REFERENCE)

    GLUON_REFERENCE = """
# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Gluon Block — MXNet's nn.Module equivalent
# ─────────────────────────────────────────────────────────────────────────
from mxnet.gluon import nn

class MLP(nn.Block):
    \"\"\"
    Multi-layer perceptron in Gluon.
    Key Gluon feature: deferred initialisation — no input_dim needed!
    \"\"\"
    def __init__(self, hidden_units, n_classes, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():           # namespaces all sub-blocks
            self.hidden = nn.Sequential()
            for units in hidden_units:
                self.hidden.add(nn.Dense(units, activation='relu'))
                self.hidden.add(nn.Dropout(dropout))
            self.output = nn.Dense(n_classes)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP([128, 64], n_classes=10)

# Initialise parameters — shapes inferred on first forward pass
net.initialize(mx.init.Xavier(), ctx=ctx)
# No Dense(?, 128) needed — Gluon infers input dim from data!

# First forward pass triggers shape inference:
X = nd.random.normal(shape=(32, 784), ctx=ctx)
output = net(X)    # Gluon now knows: Dense(784→128), Dense(128→64), Dense(64→10)
print(f"Output shape: {output.shape}")  # (32, 10)

# Inspect parameters
for name, param in net.collect_params().items():
    print(f"  {name}: shape={param.shape}")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Training loop
# ─────────────────────────────────────────────────────────────────────────
from mxnet.gluon import loss as gloss
from mxnet.gluon.data import DataLoader, ArrayDataset

# Dataset and DataLoader
X_data = nd.random.normal(shape=(1000, 784))
y_data = nd.array(np.random.randint(0, 10, 1000))
dataset = ArrayDataset(X_data, y_data)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

# Loss and Trainer
loss_fn = gloss.SoftmaxCrossEntropyLoss()    # includes softmax
trainer = gluon.Trainer(
    net.collect_params(),
    'adam',
    {'learning_rate': 1e-3, 'wd': 1e-4}     # weight decay in optimizer config
)

# ── THE CANONICAL MXNET TRAINING STEP ─────────────────────────────────
# Four steps (compare to PyTorch's five):
# 1. autograd.record()  — open recording scope
# 2. forward + loss     — compute the loss
# 3. loss.backward()    — compute gradients
# 4. trainer.step()     — apply gradients (scaled by batch_size)

# NOTE: No explicit zero_grad() needed!
# MXNet resets gradients automatically at the start of backward().

for epoch in range(5):
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.as_in_context(ctx)
        y_batch = y_batch.as_in_context(ctx)

        with autograd.record():              # ① open recording scope
            output = net(X_batch)            # ② forward pass
            loss   = loss_fn(output, y_batch) # ② compute loss
        loss.backward()                      # ③ backpropagate
        trainer.step(X_batch.shape[0])       # ④ update (normalise by batch size)

        total_loss += loss.mean().asscalar()
    print(f"Epoch {epoch+1}: avg_loss = {total_loss/len(loader):.4f}")

# KEY DIFFERENCE FROM PYTORCH:
# trainer.step(batch_size) divides gradients by batch_size before applying.
# PyTorch's loss is already mean-reduced, so no division is needed.
# MXNet allows loss to be summed over batch, then step() normalises.
"""
    print("\n  GLUON TRAINING LOOP REFERENCE CODE:")
    print(GLUON_REFERENCE)

    COMPARISON_TABLE = """
# ─────────────────────────────────────────────────────────────────────────
# COMPARISON: Training loop differences across frameworks
# ─────────────────────────────────────────────────────────────────────────

# Framework  │ Grad recording    │ Zero grad?    │ Update call
# ───────────┼───────────────────┼───────────────┼──────────────────────
# PyTorch    │ requires_grad=True│ zero_grad()   │ optimizer.step()
#            │ (implicit)        │ REQUIRED       │ (no batch scaling)
# ───────────┼───────────────────┼───────────────┼──────────────────────
# TensorFlow │ GradientTape()    │ Not needed    │ apply_gradients()
#            │ (explicit context)│ (tape resets) │ (no batch scaling)
# ───────────┼───────────────────┼───────────────┼──────────────────────
# MXNet      │ autograd.record() │ Not needed    │ trainer.step(bs)
#            │ (explicit context)│ (auto-reset)  │ (DIVIDES by bs!)
"""
    print(COMPARISON_TABLE)
    raise SystemExit(0)   # nothing more to run without MXNet

# ─────────────────────────────────────────────────────────────────────────
# If MXNet IS available — run the code
# ─────────────────────────────────────────────────────────────────────────

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()
print(f"  Context: {ctx}")
print()

# SECTION 1: NDArray basics
print("━" * 65)
print("  SECTION 1 — NDArray creation and async execution")
print("━" * 65)
print()

a = nd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ctx=ctx)
b = nd.random.normal(shape=(3, 4), ctx=ctx)

print(f"  a = {a}")
print(f"  a.shape={a.shape}, dtype={a.dtype}, context={a.context}")
print()

t0 = time.perf_counter()
for _ in range(1000):
    c = nd.dot(a, b)    # async — queues without waiting
t_async = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(1000):
    c = nd.dot(a, b)
    c.wait_to_read()    # sync — wait for completion
t_sync = time.perf_counter() - t0

print(f"  1000 matmuls (async, no wait): {t_async*1000:.2f} ms")
print(f"  1000 matmuls (sync, wait):     {t_sync*1000:.2f} ms")
print(f"  Async overhead: {t_async/t_sync:.2f}×  (async faster because GPU runs ahead)")
print()

# SECTION 2: Autograd
print("━" * 65)
print("  SECTION 2 — autograd.record() differentiation")
print("━" * 65)
print()

x = nd.array([2.0, 3.0, 4.0], ctx=ctx)
x.attach_grad()

with autograd.record():
    y = (x ** 3).sum()   # y = sum(x³), dy/dx = 3x²

y.backward()
print(f"  x        = {x.asnumpy()}")
print(f"  dy/dx (MXNet) = {x.grad.asnumpy()}   (analytic 3x²: {(3*x**2).asnumpy()})")
match = np.allclose(x.grad.asnumpy(), (3 * x**2).asnumpy())
print(f"  Match: {'✅' if match else '❌'}")
print()

# SECTION 3: Gluon Block and deferred init
print("━" * 65)
print("  SECTION 3 — Gluon Block with deferred initialisation")
print("━" * 65)
print()

class DeferredMLP(nn.Block):
    def __init__(self, hidden, n_out, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(hidden, activation='relu')
            self.fc2 = nn.Dense(hidden // 2, activation='relu')
            self.fc3 = nn.Dense(n_out)
            self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.fc3(x)

net = DeferredMLP(hidden=128, n_out=10)
net.initialize(mx.init.Xavier(), ctx=ctx)

print("  Before first forward: parameters NOT yet shaped")
for name, p in net.collect_params().items():
    try:
        print(f"    {name}: shape={p.shape}")
    except mx.base.MXNetError:
        print(f"    {name}: shape=DEFERRED (not yet inferred)")

X = nd.random.normal(shape=(32, 64), ctx=ctx)
out = net(X)
print()
print("  After first forward with shape (32, 64): parameters inferred")
for name, p in net.collect_params().items():
    print(f"    {name}: shape={p.shape}")
print()

# SECTION 4: Full training loop
print("━" * 65)
print("  SECTION 4 — Full training loop")
print("━" * 65)
print()

from mxnet.gluon import loss as gloss
from mxnet.gluon.data import DataLoader, ArrayDataset

N = 1000
X_data = nd.random.normal(shape=(N, 64))
y_data = nd.array(np.random.randint(0, 10, N))
loader = DataLoader(ArrayDataset(X_data, y_data), batch_size=64, shuffle=True)

loss_fn = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam',
                         {'learning_rate': 1e-3, 'wd': 1e-4})

print(f"  {'Epoch':>6} | {'Loss':>10}")
print(f"  {'─'*20}")
for epoch in range(1, 6):
    ep_loss = 0.0
    for Xb, yb in loader:
        Xb = Xb.as_in_context(ctx)
        yb = yb.as_in_context(ctx)
        with autograd.record():
            loss = loss_fn(net(Xb), yb)
        loss.backward()
        trainer.step(Xb.shape[0])
        ep_loss += loss.mean().asscalar()
    print(f"  {epoch:6d} | {ep_loss/len(loader):10.6f}")
print()
print("  KEY: trainer.step(batch_size) divides gradients by batch_size.")
print("  PyTorch uses mean loss already → no division needed in step().")
print("  MXNet lets you sum the loss and normalise in step() instead.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · HybridBlock — Eager to Compiled in One .hybridize() Call": {
        "description": (
            "MXNet's unique contribution to the eager/graph debate. "
            "Build a HybridBlock and show hybrid_forward with the F argument. "
            "Benchmark eager vs hybridized speed. "
            "Export the hybridized model to a portable symbol+params format. "
            "Reload and run without Python interpreter. "
            "Compare to torch.compile and @tf.function."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time, os, tempfile

print("=" * 65)
print("  HYBRIDBLOCK — EAGER TO COMPILED WITH .hybridize()")
print("=" * 65)
print()

try:
    import mxnet as mx
    from mxnet import gluon, nd
    from mxnet.gluon import nn
    HAS_MXNET = True
except ImportError:
    HAS_MXNET = False

if not HAS_MXNET:
    print("  MXNet not installed — showing annotated reference code.")
    print()

HYBRID_CONCEPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  THE HYBRIDBLOCK CONCEPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HybridBlock solves the eagerness-vs-speed dilemma by making the
  SAME code run in two modes:

  Mode 1 — EAGER (before hybridize()):
    F = mxnet.ndarray
    Every F.relu(), F.Dense() etc. executes immediately.
    Python debugger works. Print statements work. Dynamic shapes work.

  Mode 2 — SYMBOLIC (after hybridize()):
    F = mxnet.symbol
    Operations BUILD a computation graph instead of running.
    Graph is compiled and cached.
    Subsequent calls skip Python entirely.

  The magic: hybrid_forward(self, F, x) uses F as a namespace.
  In eager mode:  F.relu(x) = nd.relu(x)   (immediate computation)
  In graph mode:  F.relu(x) = sym.relu(x)  (graph node creation)

  YOUR CODE IS IDENTICAL — the framework switches F behind the scenes.
"""

print(HYBRID_CONCEPT)

HYBRID_CODE = """
from mxnet.gluon import nn
from mxnet import gluon

class ResidualBlock(nn.HybridBlock):
    \"\"\"
    Residual block as a HybridBlock.
    Note: hybrid_forward takes F as the first argument after self.
    The code inside must use F.* for all operations (not nd.* or mx.*)
    because F will be either ndarray or symbol depending on mode.
    \"\"\"
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.bn1   = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, use_bias=False)
            self.bn2   = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1, use_bias=False)

    def hybrid_forward(self, F, x):
        # F can be nd (eager) or sym (graph) — code is IDENTICAL
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')   # use F.Activation, not nd.relu
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)
        return x + residual                    # skip connection

class SmallResNet(nn.HybridBlock):
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.stem   = nn.Conv2D(32, kernel_size=3, padding=1, use_bias=False)
            self.block1 = ResidualBlock(32)
            self.block2 = ResidualBlock(32)
            self.pool   = nn.GlobalAvgPool2D()
            self.fc     = nn.Dense(n_classes)

    def hybrid_forward(self, F, x):
        x = F.Activation(self.stem(x), act_type='relu')
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        return self.fc(x)


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()
net = SmallResNet(n_classes=10)
net.initialize(mx.init.Xavier(), ctx=ctx)

# ── Benchmark: eager vs hybridized ─────────────────────────────────
X = mx.nd.random.normal(shape=(16, 32, 32, 32), ctx=ctx)  # NCHW format

# Eager mode
N = 50
t0 = time.perf_counter()
for _ in range(N):
    out = net(X)
    out.wait_to_read()
t_eager = (time.perf_counter() - t0) / N * 1000

# Hybridize
net.hybridize()
_ = net(X); _.wait_to_read()   # warmup: trigger tracing

t0 = time.perf_counter()
for _ in range(N):
    out = net(X)
    out.wait_to_read()
t_hyb = (time.perf_counter() - t0) / N * 1000

print(f"  Eager:      {t_eager:.3f} ms/forward")
print(f"  Hybridized: {t_hyb:.3f} ms/forward")
print(f"  Speedup:    {t_eager/t_hyb:.2f}×")

# ── Export ──────────────────────────────────────────────────────────
net.export("small_resnet", epoch=0)
# Creates: small_resnet-symbol.json   (computation graph)
#          small_resnet-0000.params   (all weights)
# These two files run in ANY language without Python!

# ── Load back as SymbolBlock (Python-less execution) ────────────────
loaded_sym, loaded_args, loaded_aux = mx.model.load_checkpoint("small_resnet", 0)
exec_block = gluon.nn.SymbolBlock(
    outputs=loaded_sym,
    inputs=mx.sym.var('data'),
    params=gluon.ParameterDict(),
)
# The loaded model needs no Python class definition —
# the symbol.json IS the architecture.
"""

print("  HYBRIDBLOCK CODE:")
print(HYBRID_CODE)

if HAS_MXNET:
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()
    print()
    print("━" * 65)
    print("  RUNNING LIVE: HybridBlock vs Block benchmark")
    print("━" * 65)
    print()

    # Simple Block (not hybrid)
    class EagerBlock(nn.Block):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            with self.name_scope():
                self.layers = nn.Sequential()
                for _ in range(6):
                    self.layers.add(nn.Dense(256, activation='relu'))
                self.out = nn.Dense(10)
        def forward(self, x):
            return self.out(self.layers(x))

    # HybridBlock version
    class HybridNet(nn.HybridBlock):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            with self.name_scope():
                self.layers = nn.HybridSequential()
                for _ in range(6):
                    self.layers.add(nn.Dense(256, activation='relu'))
                self.out = nn.Dense(10)
        def hybrid_forward(self, F, x):
            return self.out(self.layers(x))

    X_bench = nd.random.normal(shape=(64, 128), ctx=ctx)
    N_BENCH  = 200

    def bench(net, x, n):
        for _ in range(10): net(x).wait_to_read()   # warmup
        t0 = time.perf_counter()
        for _ in range(n):
            net(x).wait_to_read()
        return (time.perf_counter() - t0) / n * 1000

    eager_net  = EagerBlock();  eager_net.initialize(mx.init.Xavier(), ctx=ctx)
    hybrid_net = HybridNet();   hybrid_net.initialize(mx.init.Xavier(), ctx=ctx)
    hybrid_net.hybridize()
    hybrid_net(X_bench).wait_to_read()   # trigger compilation

    t_eager  = bench(eager_net,  X_bench, N_BENCH)
    t_hybrid = bench(hybrid_net, X_bench, N_BENCH)

    print(f"  6-layer Dense (256 units), batch=64, input=128")
    print(f"    Block (eager):          {t_eager:.4f} ms/forward")
    print(f"    HybridBlock (compiled): {t_hybrid:.4f} ms/forward")
    print(f"    Speedup:                {t_eager/t_hybrid:.2f}×")
    print()

    # Export
    with tempfile.TemporaryDirectory() as tmp:
        export_path = os.path.join(tmp, "hybrid_net")
        hybrid_net(X_bench)           # ensure tracing done
        hybrid_net.export(export_path, epoch=0)
        sym_size = os.path.getsize(export_path + "-symbol.json") / 1024
        par_size = os.path.getsize(export_path + "-0000.params") / 1024
        print(f"  Exported hybridized model:")
        print(f"    symbol.json:  {sym_size:.1f} KB  (computation graph — language-agnostic)")
        print(f"    0000.params:  {par_size:.1f} KB  (all weights)")
        print(f"    Total:        {sym_size+par_size:.1f} KB")
        print()
        print(f"  These files can be loaded in C++, Java, Scala, R, Go")
        print(f"  without Python or any MXNet Python bindings.")

print()
COMPARISON_TABLE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EAGER → COMPILED: COMPARISON ACROSS FRAMEWORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Framework  │ Mechanism          │ When               │ Trigger
  ───────────┼────────────────────┼────────────────────┼─────────────────
  MXNet      │ HybridBlock        │ Python+eager→graph │ .hybridize()
             │                    │ Same code, 2 modes │ (one call)
  ───────────┼────────────────────┼────────────────────┼─────────────────
  PyTorch    │ torch.compile()    │ Eager→TorchInductor│ torch.compile(m)
             │                    │ Trace on first call│ (one call)
  ───────────┼────────────────────┼────────────────────┼─────────────────
  TensorFlow │ @tf.function       │ Python→TF graph    │ Decorator
             │                    │ Traces on 1st call │ or context mgr
  ───────────┼────────────────────┼────────────────────┼─────────────────

  MXNet's HybridBlock was the FIRST production-ready hybrid solution (2017).
  PyTorch's torch.compile (2022) and TF2's tf.function (2019) came later.
  All three solve the same problem with slightly different APIs.

  Unique to MXNet's approach:
    - The F argument makes the dual-mode explicit in the code signature
    - Exported symbol is truly language-independent (no Python runtime needed)
    - Hybridization works transitively: hybridize() compiles all sub-blocks
"""
print(COMPARISON_TABLE)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Parameter Server, Data Pipeline & AWS SageMaker Deployment": {
        "description": (
            "MXNet's distributed training via the KVStore parameter server. "
            "Simulate synchronous vs asynchronous gradient aggregation. "
            "MXNet data iterators vs gluon.data for production pipelines. "
            "Complete AWS SageMaker training and deployment reference. "
            "TVM cross-compilation for edge targets."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from collections import defaultdict

print("=" * 65)
print("  PARAMETER SERVER, DATA PIPELINE & SAGEMAKER DEPLOYMENT")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Parameter Server — Synchronous vs Asynchronous simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Parameter Server simulation: sync vs async")
print("━" * 65)
print()

print("  The Parameter Server (PS) architecture separates:")
print("  WORKERS: compute gradients on data shards")
print("  SERVERS: store parameters, aggregate gradients, apply updates")
print()

class ParameterServer:
    """
    Simulated parameter server with synchronous and asynchronous modes.
    In production MXNet this is the KVStore abstraction.
    """
    def __init__(self, params: np.ndarray, lr: float = 0.01, mode: str = 'sync'):
        self.params   = params.copy().astype(np.float64)
        self.lr       = lr
        self.mode     = mode                  # 'sync' or 'async'
        self.grad_buf = []                    # buffer for sync mode
        self.version  = 0                     # parameter version counter
        self.n_workers = 0                    # how many workers registered
        self.staleness_log = []               # track gradient staleness

    def register(self, n_workers: int):
        self.n_workers = n_workers

    def push(self, grad: np.ndarray, worker_id: int,
             worker_version: int) -> None:
        """Worker pushes gradients to server."""
        staleness = self.version - worker_version
        self.staleness_log.append(staleness)

        if self.mode == 'async':
            # ASYNC: update immediately regardless of other workers
            self.params -= self.lr * grad
            self.version += 1
        else:
            # SYNC: buffer until all workers have pushed
            self.grad_buf.append(grad)
            if len(self.grad_buf) >= self.n_workers:
                avg_grad      = np.mean(self.grad_buf, axis=0)
                self.params  -= self.lr * avg_grad
                self.grad_buf = []
                self.version += 1

    def pull(self) -> np.ndarray:
        """Worker pulls current parameters from server."""
        return self.params.copy(), self.version


def simulate_training(mode: str, n_workers: int = 4,
                       n_rounds: int = 20, straggler_prob: float = 0.3,
                       true_optimum: float = 3.0):
    """
    Simulate distributed optimisation of a simple quadratic:
    L(w) = (w - true_optimum)²
    Gradient: dL/dw = 2*(w - true_optimum)

    Straggler: with straggler_prob, a worker uses a stale gradient
    (from 2 rounds ago instead of current params).
    """
    np.random.seed(42)
    ps     = ParameterServer(np.array([0.0]), lr=0.05, mode=mode)
    ps.register(n_workers)

    history = []

    for round_num in range(n_rounds):
        current_params, current_version = ps.pull()
        worker_grads = []

        for wid in range(n_workers):
            # Straggler uses stale params
            if np.random.random() < straggler_prob and round_num >= 2:
                stale_params  = history[max(0, len(history)-2)]['param']
                stale_version = max(0, current_version - 2)
                grad          = 2 * (stale_params - true_optimum)
                worker_grads.append((grad, wid, stale_version))
            else:
                noise = np.random.normal(0, 0.1)  # gradient noise
                grad  = 2 * (current_params - true_optimum) + noise
                worker_grads.append((grad, wid, current_version))

        if mode == 'async':
            # Workers push one at a time in random order
            for grad, wid, version in worker_grads:
                ps.push(grad, wid, version)
        else:
            # Workers push, but all must arrive before server updates
            for grad, wid, version in worker_grads:
                ps.push(grad, wid, version)

        history.append({'param': ps.params[0], 'version': ps.version})

    return history, ps.staleness_log

print(f"  Optimising L(w) = (w - 3)², starting at w=0")
print(f"  4 workers, 20 rounds, 30% straggler probability")
print()

hist_sync,  stale_sync  = simulate_training('sync',  straggler_prob=0.3)
hist_async, stale_async = simulate_training('async', straggler_prob=0.3)

print(f"  {'Round':>6} | {'Sync param':>12} | {'Async param':>13}")
print(f"  {'─'*35}")
for i in range(0, 20, 4):
    print(f"  {i+1:6d} | {hist_sync[i]['param']:12.4f} | {hist_async[i]['param']:13.4f}")
print()

final_sync  = hist_sync[-1]['param']
final_async = hist_async[-1]['param']
print(f"  True optimum:         3.0000")
print(f"  Sync final w:         {final_sync:.4f}  (error: {abs(final_sync-3):.4f})")
print(f"  Async final w:        {final_async:.4f}  (error: {abs(final_async-3):.4f})")
print()
print(f"  Mean gradient staleness:")
print(f"    Sync:  {np.mean(stale_sync):.2f} rounds  (always 0 for strict sync)")
print(f"    Async: {np.mean(stale_async):.2f} rounds  (non-zero due to stragglers)")
print()
print("  Sync:  slower wall-clock (waits for all workers), cleaner convergence")
print("  Async: faster wall-clock (no straggler wait), noisier convergence")
print("  For most DL: use Horovod AllReduce (sync, efficient) over async PS")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: MXNet data pipeline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — MXNet data pipeline patterns")
print("━" * 65)
print()

print("  MXNet has two data pipeline APIs:")
print()
print("  1. mxnet.io.DataIter (legacy, C++ level):")
print("     Used in Symbol API training. Highly optimised but inflexible.")
print("     Example: mx.io.ImageRecordIter — reads ImageNet .rec files.")
print()
print("  2. gluon.data (modern, Python level):")
print("     Mirrors PyTorch's Dataset/DataLoader. Composable and flexible.")
print()

GLUON_DATA = """
from mxnet.gluon.data import Dataset, DataLoader
from mxnet.gluon.data.vision import transforms

# ── Map-style Dataset ──────────────────────────────────────────────────
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   # returns NDArrays

# ── Transforms (image augmentation) ───────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── DataLoader ─────────────────────────────────────────────────────────
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,         # parallel data loading (like PyTorch)
    last_batch='discard',  # or 'keep', 'rollover'
    pin_memory=True,       # pin to page-locked memory for faster GPU transfer
)

# ── ImageRecordIter (legacy, for ImageNet-scale datasets) ─────────────
# First, pack images into .rec format:
# im2rec.py image_list.lst /path/to/images train.rec --resize 256
# Then load with:
train_iter = mx.io.ImageRecordIter(
    path_imgrec   = "train.rec",
    data_shape    = (3, 224, 224),         # NCHW format (MXNet default)
    batch_size    = 64,
    shuffle       = True,
    num_parts     = 1, part_index = 0,     # for multi-worker data splitting
    preprocess_threads = 4,                # parallel preprocessing
)
# ImageRecordIter is extremely fast for large datasets —
# reads .rec directly in C++ with multi-threaded augmentation.
"""
print("  Data pipeline code:")
print(GLUON_DATA)

# Simulate performance of gluon.data DataLoader
class NumpyDataset:
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

N_DS = 2048
X_np = np.random.randn(N_DS, 128).astype(np.float32)
y_np = np.random.randint(0, 10, N_DS).astype(np.float32)

try:
    from mxnet.gluon.data import DataLoader as GluonLoader, ArrayDataset
    ds = ArrayDataset(mx.nd.array(X_np), mx.nd.array(y_np))
    dl = GluonLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    t0 = time.perf_counter()
    for _ in dl: pass
    t_epoch = time.perf_counter() - t0
    print(f"  Gluon DataLoader: {N_DS} samples, batch=64 → {t_epoch*1000:.1f} ms/epoch")
except Exception:
    print("  (Gluon DataLoader requires MXNet installation)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: AWS SageMaker deployment reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — AWS SageMaker: training and deployment")
print("━" * 65)
print()

SAGEMAKER_TRAIN = """
# ── train.py (entry point script, runs on SageMaker) ──────────────────
import os, argparse
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn

def train(args):
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

    # Build model
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'),
            nn.Dropout(0.3),
            nn.Dense(args.num_classes))
    net.initialize(mx.init.Xavier(), ctx=ctx)
    net.hybridize()               # compile for speed

    # Data: SageMaker provides S3 data at os.environ['SM_CHANNEL_TRAIN']
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', args.data_dir)

    trainer  = gluon.Trainer(net.collect_params(), 'adam',
                              {'learning_rate': args.lr})
    loss_fn  = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(args.epochs):
        # ... training loop ...
        pass

    # Save model to SageMaker's expected output directory
    model_dir = os.environ.get('SM_MODEL_DIR', args.model_dir)
    net.export(os.path.join(model_dir, 'model'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num-classes', type=int,   default=10)
    parser.add_argument('--data-dir',    type=str,   default='/opt/ml/input/data')
    parser.add_argument('--model-dir',   type=str,   default='/opt/ml/model')
    train(parser.parse_args())
"""

print("  train.py — SageMaker entry point:")
print(SAGEMAKER_TRAIN)

SAGEMAKER_LAUNCH = """
# ── launcher.py (runs locally, launches on SageMaker) ─────────────────
from sagemaker.mxnet import MXNet as SageMakerMXNet

estimator = SageMakerMXNet(
    entry_point     = 'train.py',
    source_dir      = './src',            # directory with train.py
    role            = 'arn:aws:iam::...:role/SageMakerRole',
    instance_type   = 'ml.p3.2xlarge',   # 1× V100 (16 GB)
    instance_count  = 1,
    framework_version = '1.9.0',
    py_version        = 'py3',
    hyperparameters = {
        'epochs':      20,
        'lr':          0.001,
        'num-classes': 10,
    },
    # For multi-GPU single node:
    #   instance_type = 'ml.p3.16xlarge'  (8× V100)
    # For multi-node:
    #   instance_count = 4                (4 machines)
    #   distribution = {'parameter_server': {'enabled': True}}
)

estimator.fit({
    'train': 's3://my-bucket/data/train',
    'val':   's3://my-bucket/data/val',
})
print(f"Model artifact: {estimator.model_data}")

# ── Deploy as a REST endpoint ──────────────────────────────────────────
predictor = estimator.deploy(
    initial_instance_count = 1,
    instance_type          = 'ml.m5.xlarge',    # CPU inference endpoint
    serializer   = sagemaker.serializers.NumpySerializer(),
    deserializer = sagemaker.deserializers.NumpyDeserializer(),
)

# Query the endpoint
result = predictor.predict(test_data_numpy)   # returns numpy array
print(f"Predictions: {result}")

# ── SageMaker pricing estimate (2024) ─────────────────────────────────
# ml.p3.2xlarge:  $3.83/hour  training
# ml.p3.16xlarge: $24.48/hour training
# ml.m5.xlarge:   $0.23/hour  inference endpoint
# Use Spot Instances for 60-90% training cost reduction:
#   use_spot_instances = True
#   max_wait = 3600  (wait up to 1h for spot)
"""
print("  SageMaker launch and deploy:")
print(SAGEMAKER_LAUNCH)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: TVM cross-compilation reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — TVM cross-compilation for edge targets")
print("━" * 65)
print()

TVM_REFERENCE = """
# ── Compile a hybridized MXNet model for any hardware via TVM ──────────
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# 1. Export model from MXNet
net.hybridize()
net(dummy_input)                       # trigger tracing
net.export("my_model", epoch=0)

# 2. Load into TVM's relay IR
shape_dict = {"data": (1, 3, 224, 224)}
mod, params = relay.frontend.from_mxnet(
    symbol    = mx.sym.load("my_model-symbol.json"),
    arg_params = mx.nd.load("my_model-0000.params"),
    shape     = shape_dict,
)

# 3. Compile for target hardware

# Target: x86 Intel Skylake CPU
target = tvm.target.Target("llvm -mcpu=skylake-avx512")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
lib.export_library("model_x86.so")

# Target: ARM Cortex-A (Raspberry Pi)
target = tvm.target.Target("llvm -mtriple=armv7l-linux-gnueabihf")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
lib.export_library("model_arm.tar")

# Target: NVIDIA GPU
target = tvm.target.cuda()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Target: Apple Silicon (Metal GPU)
target = tvm.target.Target("metal")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# 4. AutoTVM tuning — find the FASTEST implementation on YOUR hardware
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

tasks   = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
tuner   = XGBTuner(tasks[0])
tuner.tune(
    n_trial=1000,
    measure_option=autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10),
    ),
    callbacks=[autotvm.callback.log_to_file("tune.log")],
)
# After tuning: rebuild with the tuning log for maximum speed
with autotvm.apply_history_best("tune.log"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

# Typical AutoTVM speedup: 1.5-3x over untuned baseline
# Works for ANY hardware — finds optimal tile sizes, unroll factors, layouts
"""

print("  TVM compilation code:")
print(TVM_REFERENCE)

print()
print("━" * 65)
print("  MXNET ECOSYSTEM SUMMARY")
print("━" * 65)
print()
print("  ┌────────────────────────────────────────────────────────────────┐")
print("  │ Component           │ What it does               │ Replaces   │")
print("  ├────────────────────────────────────────────────────────────────┤")
print("  │ NDArray             │ Eager tensor operations    │ numpy+CUDA │")
print("  │ Symbol              │ Static graph construction  │ TF1 graphs │")
print("  │ Gluon Block         │ Neural network module      │ nn.Module  │")
print("  │ HybridBlock         │ Eager→compiled in one call │ torch.comp.│")
print("  │ KVStore (PS)        │ Distributed param server   │ DDP (async)│")
print("  │ mxnet.autograd      │ Gradient computation       │ GradTape   │")
print("  │ GluonCV / GluonNLP  │ Pre-trained model zoos     │ timm/HF    │")
print("  │ SageMaker MXNet     │ Managed AWS training/serve │ Self-hosted│")
print("  │ Neuron SDK          │ Inferentia compilation     │ TensorRT   │")
print("  │ TVM backend         │ Cross-platform compile     │ ONNX Rt    │")
print("  └────────────────────────────────────────────────────────────────┘")
print()
print("  WHEN TO CHOOSE MXNET:")
print("    ✅ Your infrastructure is AWS-native (SageMaker, Inferentia)")
print("    ✅ Deploying to diverse hardware targets via TVM")
print("    ✅ Need a language-independent exported model (C++, Java, Scala)")
print("    ✅ Async parameter server training at extreme scale")
print("    ✅ Memory-constrained training (MXNet's pool is very efficient)")
print("    ❌ Research / reproducing recent papers (PyTorch ecosystem)")
print("    ❌ Rapid prototyping (Keras/PyTorch are faster to iterate)")
print("    ❌ Google Cloud / TPU deployment (use JAX or TensorFlow)")
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