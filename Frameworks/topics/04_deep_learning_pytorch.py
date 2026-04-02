"""
PyTorch — Meta's Research-First Deep Learning Framework
========================================================

PyTorch is Meta's open-source deep learning framework, released in 2016
as a Python-first reimagining of Torch (a Lua library). Where TensorFlow
once dominated production, PyTorch has won the research world decisively —
roughly 75–80% of papers at NeurIPS, ICML, and ICLR use PyTorch — and is
rapidly closing the production gap through TorchServe, torch.export, and
the ExecuTorch edge runtime.

The framework's core philosophy is "define-by-run" (dynamic computation
graphs): Python IS the graph. There is no separate compilation step, no
session, no placeholder. You write Python, you debug Python, and the
gradients flow through whatever Python you wrote. This makes PyTorch feel
like a natural extension of NumPy with GPU acceleration and autodiff.

Understanding PyTorch's design teaches you the principles every modern
deep learning framework now converges on: eager-first execution, dynamic
graphs, autograd as a first-class citizen, and staged compilation for
production performance.

This module covers PyTorch's tensor engine, the Autograd differentiation
system, nn.Module model building, the DataLoader/Dataset pipeline, the
torch.compile JIT system, and the full deployment stack from TorchScript
to ONNX to ExecuTorch.

"""

import textwrap
import re

TOPIC_NAME   = "PyTorch — Meta's Research-First Deep Learning Framework"
DISPLAY_NAME = "04 · PyTorch"
ICON         = "🔥"
SUBTITLE     = "From Dynamic Autograd to torch.compile Production Pipelines"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — PYTORCH'S HISTORY AND DESIGN PHILOSOPHY

### The Define-by-Run Revolution

    Torch (pre-PyTorch): Lua-based, fast C/CUDA backend, but Lua made it
        niche. Researchers wanted Python.

    PyTorch 0.1 (2016): Soumith Chintala and team at Meta (then Facebook AI)
        rewrote Torch's core in Python. The central innovation was the
        dynamic computation graph — "define-by-run" rather than
        TensorFlow 1.x's "define-then-run."

    Dynamic graph means:
        The computation graph is built IMPLICITLY as Python executes.
        Every torch operation records itself on a graph in real time.
        No placeholder, no session, no compile step.

        import torch

        x = torch.tensor([3.0], requires_grad=True)
        y = x ** 2 + 2 * x + 1    # graph built HERE, as these lines run
        y.backward()               # differentiate the graph that was built
        print(x.grad)              # → tensor([8.])  (dy/dx = 2x+2 at x=3)

        # The graph is DISCARDED after backward(). Next forward pass builds
        # a NEW graph — this is what "dynamic" means.

    Contrast with TF1's static graph:
        TF1 defined graph first, then ran it in a session.
        PyTorch: graph is the program's execution trace.
        This means: if statements, for loops, recursion all work naturally.

### Why Dynamic Graphs Win for Research

    1. NATURAL CONTROL FLOW:
        # This is impossible in TF1 without tf.cond / tf.while_loop
        def forward(x, depth):
            for _ in range(depth):          # depth can vary per sample!
                x = self.layer(x)
            return x

    2. PRINT DEBUGGING:
        # Inspect any intermediate value directly — it's a Python float
        y = model(x)
        print(y.shape, y.mean().item())     # just works

    3. VARIABLE-LENGTH SEQUENCES:
        # Sequence length can differ per batch item — no padding tricks needed
        for token_seq in batch:             # different lengths
            hidden = rnn(token_seq)

    4. RESEARCH SPEED:
        Faster to prototype → iterate → discard.
        Paper ideas are often impossible to express in static graph frameworks.

### PyTorch's Evolution

    v0.1 (2016): Initial release. Dynamic graphs. Limited production tooling.
    v1.0 (2018): TorchScript introduced — JIT compilation for deployment.
    v1.1 (2019): torch.utils.tensorboard. Improved ONNX export.
    v1.6 (2020): Automatic Mixed Precision (AMP). TorchScript improvements.
    v1.8 (2021): torch.fx: structured graph transformation. Functorch (vmap).
    v2.0 (2023): torch.compile introduced — 30–200% speedup via Dynamo + Inductor.
    v2.1 (2023): torch.export for ahead-of-time compilation. FlexAttention.
    v2.2 (2024): SDPA (Scaled Dot-Product Attention) kernel, FlashAttention-2.
    v2.3+ (2024-25): ExecuTorch for edge/mobile. Improved torch.compile coverage.

### The Current Ecosystem Position

    Research:      PyTorch dominates (~80% of ML papers use PyTorch)
    Production:    Growing rapidly (TorchServe, torch.export, ExecuTorch)
    Mobile/Edge:   ExecuTorch (newer than TFLite but rapidly maturing)
    Browser:       Limited (ONNX.js / ONNX Runtime Web covers this gap)
    Enterprise:    Hugging Face ecosystem built on PyTorch (transformers library)
    LLMs/GenAI:    The de facto standard — GPT, LLaMA, Stable Diffusion all PyTorch


##### PART 2 — TENSORS: PYTORCH'S FUNDAMENTAL DATA STRUCTURE

### What Is a Tensor?

    A tensor is an n-dimensional array with:
        - A dtype    (torch.float32, torch.int64, torch.bool, torch.bfloat16…)
        - A shape    (tuple of dimensions, e.g. [batch, channels, height, width])
        - A device   (cpu, cuda:0, cuda:1, mps for Apple Silicon)
        - A storage  (the actual contiguous memory block)
        - A grad_fn  (the backward function that created it, if from autograd)
        - requires_grad flag (whether to track gradients through it)

    PyTorch tensors are nearly API-identical to NumPy arrays, but can live
    on GPU and participate in automatic differentiation.

### Tensor Creation

    From Python / NumPy:
        torch.tensor([1.0, 2.0, 3.0])              # from list (copies data)
        torch.from_numpy(np_array)                  # shares memory with NumPy!
        torch.as_tensor(np_array, dtype=torch.float32)   # zero-copy if possible

    Filled tensors:
        torch.zeros(3, 4)                           # shape (3,4) all zeros
        torch.ones(3, 4)                            # all ones
        torch.full((3, 4), fill_value=7.0)          # all 7.0
        torch.eye(4)                                # identity matrix

    Random tensors:
        torch.rand(3, 4)                            # uniform [0, 1)
        torch.randn(3, 4)                           # standard normal N(0,1)
        torch.randint(0, 10, size=(3, 4))           # integer uniform [0, 10)
        torch.manual_seed(42)                       # reproducibility

    Like-tensor creation (same shape, device, dtype):
        torch.zeros_like(existing_tensor)
        torch.ones_like(existing_tensor)
        torch.rand_like(existing_tensor)

### Tensor dtypes and Memory Layout

    Float types (for neural network weights):
        torch.float32  (f32)  — default, full precision
        torch.float16  (f16)  — half precision, GPU-optimised (less stable)
        torch.bfloat16 (bf16) — brain float, same exponent range as f32, 
                                 preferred for training LLMs on modern hardware
        torch.float64  (f64)  — double precision (rarely used in DL)

    Integer types:
        torch.int32  / torch.int64  — indices, class labels
        torch.uint8                 — image pixels (before normalisation)

    Boolean:
        torch.bool                  — masks, attention masks

    The rule:
        Use float32 for training unless you know your hardware supports bf16/f16.
        Use bf16 with AMP on A100/H100/RTX 4090 for ~2× throughput.
        Use int64 for class indices passed to loss functions.

### Essential Tensor Operations

    Shape manipulation:
        t.reshape(new_shape)           # returns a view if possible
        t.view(new_shape)              # ALWAYS returns a view (contiguous required)
        t.permute(0, 2, 1)             # reorder dimensions (like np.transpose)
        t.squeeze(dim)                 # remove size-1 dimension
        t.unsqueeze(dim)               # add size-1 dimension
        t.flatten(start_dim, end_dim)  # collapse dimensions
        t.contiguous()                 # force C-contiguous layout

    Aggregation:
        t.sum(dim=1, keepdim=True)     # sum along dim
        t.mean(), t.std(), t.var()
        t.max(dim=1).values            # max values along dim
        t.argmax(dim=1)                # index of max
        t.topk(k=5, dim=1)            # top-k values and indices

    Elementwise:
        t + other, t * other, t ** 2   # standard Python operators
        torch.relu(t), torch.sigmoid(t), torch.softmax(t, dim=-1)
        torch.clamp(t, min=0, max=1)   # clip values

    Matrix operations:
        t @ other                      # matrix multiplication (batched)
        torch.matmul(a, b)             # same, clearer for non-2D
        torch.bmm(a, b)                # batched matmul (3D tensors)
        torch.einsum('bij,bjk->bik', a, b)  # Einstein summation (universal)

    Concatenation and stacking:
        torch.cat([a, b], dim=0)       # concatenate along existing dim
        torch.stack([a, b], dim=0)     # create NEW dim and stack

### Device Management

    PyTorch tensors are CPU by default. Moving to GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = t.to(device)               # returns new tensor on device
        t = t.cuda()                   # shorthand for .to('cuda:0')
        t = t.cpu()                    # move back to CPU

    Check GPU availability:
        torch.cuda.is_available()          # True if CUDA GPU present
        torch.cuda.device_count()          # number of GPUs
        torch.cuda.get_device_name(0)      # GPU model name

    Apple Silicon (MPS):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        t = t.to(device)                   # runs on Apple's GPU via Metal

    Rule: model AND data must be on the SAME device.
        model = model.to(device)
        X, y = X.to(device), y.to(device)

### Views vs Copies

    Many reshape operations return VIEWS (same memory, different strides):
        a = torch.zeros(4, 4)
        b = a.view(16)          # b shares memory with a
        b[0] = 99               # modifies a too!

    Non-contiguous tensors cannot be viewed:
        a_t = a.T               # transpose — non-contiguous
        a_t.view(-1)            # RuntimeError! Must call .contiguous() first

    Check:
        t.is_contiguous()       # True/False
        t.data_ptr()            # memory address (same for views of same tensor)

    Use .clone() to force a copy:
        b = a.clone()           # new memory, gradient history preserved
        b = a.detach().clone()  # new memory, no gradient history


##### PART 3 — AUTOGRAD: PYTORCH'S DIFFERENTIATION ENGINE

### The Computational Graph

    Every time you apply an operation to a tensor that has requires_grad=True,
    PyTorch records that operation in a directed acyclic graph (DAG).

    Nodes: tensors
    Edges: Function objects (grad_fn) that know how to compute the backward pass

    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2          # y.grad_fn = PowBackward0
    z = y + 2           # z.grad_fn = AddBackward0

    The forward pass builds this graph implicitly.
    .backward() traverses it in reverse to compute gradients.

### Forward Pass vs Backward Pass

    Forward pass: compute the output (loss), build the graph.
    Backward pass: traverse the graph in reverse (chain rule), accumulate gradients.

    Chain rule reminder:
        If  z = f(y),  y = g(x)
        Then  dz/dx = dz/dy * dy/dx
        PyTorch applies this automatically at each grad_fn node.

    loss.backward()            # computes ∂loss/∂(all leaves with requires_grad)
    x.grad                     # the accumulated gradient ∂loss/∂x

    The graph is freed after .backward() by default (retain_graph=False).
    Call .backward(retain_graph=True) to call backward multiple times.

### requires_grad and Leaf Tensors

    Leaf tensor: a tensor created directly (not from an operation).
    Model parameters (nn.Parameter) are always leaf tensors with requires_grad=True.

    Non-leaf tensors (results of operations) are also tracked but their .grad
    is NOT stored by default (saves memory). Use .retain_grad() to keep it.

    Disabling gradient tracking:
        torch.no_grad():               # context manager — no graph built
            with torch.no_grad():
                output = model(x)      # fast inference, no graph overhead

        tensor.detach():               # new tensor sharing data, no grad
            target = prediction.detach()  # stop gradients from flowing back

        @torch.no_grad()               # decorator version
        def evaluate(model, loader): ...

    When to use:
        - Inference / evaluation loops: always wrap in torch.no_grad()
        - Computing targets in RL or self-supervised learning: .detach()
        - Freezing a subnetwork: set param.requires_grad = False

### Gradient Accumulation

    By default, .backward() ACCUMULATES gradients into .grad (adds to existing).
    Before each training step, call optimizer.zero_grad():

        optimizer.zero_grad()           # clear accumulated gradients
        output = model(x)
        loss = criterion(output, y)
        loss.backward()                 # accumulate into .grad
        optimizer.step()                # update weights using .grad

    Why accumulate? Gradient accumulation is used to simulate larger batches:
        # Simulate batch_size=256 with only 64 samples in memory
        ACCUM_STEPS = 4
        for i, (x, y) in enumerate(loader):
            loss = criterion(model(x), y) / ACCUM_STEPS   # scale loss
            loss.backward()                                # accumulate grads
            if (i + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

### Higher-Order Derivatives

    create_graph=True preserves the backward graph for second-order differentiation:

        x = torch.tensor(2.0, requires_grad=True)
        y = x ** 3
        dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]    # 3x²
        d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]                 # 6x

    Applications:
        MAML (meta-learning): gradient of a gradient for fast adaptation
        Physics-informed NNs: enforce PDE constraints (∂²u/∂x²)
        Gradient penalty: WGAN-GP requires ||∇D(x̂)||₂ → 1 constraint

### torch.autograd.grad vs .backward()

    .backward():
        - Accumulates into .grad attribute of leaf tensors
        - Standard training loop pattern
        - Frees the graph after (unless retain_graph=True)

    torch.autograd.grad(outputs, inputs, ...):
        - Returns gradient tensors directly (doesn't modify .grad)
        - Required for higher-order derivatives
        - More explicit — prefer for custom differentiation logic

    Example:
        grads = torch.autograd.grad(loss, model.parameters())
        # returns a tuple of gradient tensors, one per parameter


##### PART 4 — nn.MODULE: BUILDING NEURAL NETWORKS

### nn.Module as the Universal Building Block

    nn.Module is PyTorch's base class for all neural network components.
    A Module is a container that:
        - Holds parameters (nn.Parameter) and submodules (nested Modules)
        - Defines a forward() method that computes the output
        - Tracks all parameters recursively for optimizer and serialisation
        - Has training/eval mode flags (affects BatchNorm, Dropout)

    The invariant:
        EVERYTHING in your model — each layer, each block, the whole model —
        is an nn.Module. They compose recursively without special wiring.

### Pattern 1: nn.Sequential — Linear Stacks

    For pipelines where each layer feeds directly into the next:

        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    Named sequential (access layers by name):
        from collections import OrderedDict
        model = nn.Sequential(OrderedDict([
            ('fc1',   nn.Linear(784, 256)),
            ('relu1', nn.ReLU()),
            ('fc2',   nn.Linear(256, 10)),
        ]))
        model.fc1       # access by name

    Limitation: Cannot express skip connections, multiple inputs/outputs,
    or conditional logic. Use subclassing for those.

### Pattern 2: Module Subclassing — Full Flexibility

    The standard pattern for any model beyond a linear stack:

        class TransformerBlock(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()                          # MUST call super().__init__()
                self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                self.ffn   = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.drop  = nn.Dropout(dropout)

            def forward(self, x, attn_mask=None):          # define computation here
                # Pre-norm transformer (GPT-style)
                attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
                x = self.norm1(x + self.drop(attn_out))    # residual + norm
                ffn_out = self.ffn(x)
                return self.norm2(x + self.drop(ffn_out))  # residual + norm

    Rules for subclassing:
        1. Always call super().__init__() in __init__.
        2. Assign all submodules as attributes (self.layer = ...).
           PyTorch tracks them only if assigned to self.
        3. Define forward(self, x, ...) — never call forward() directly;
           call the module as a function: output = module(input).
        4. Do NOT do computation in __init__ — only define structure.

### nn.Parameter vs Buffers

    nn.Parameter:
        A tensor that IS a model parameter. Included in model.parameters().
        Updated by the optimizer. Saved/loaded with state_dict.

        class ScaledAttention(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(1))    # learnable scalar
                self.proj  = nn.Linear(d_model, d_model)

    nn.Buffer:
        A tensor that is NOT a parameter. Not updated by optimizer.
        But IS moved to GPU with .to(device) and saved in state_dict.

        class RunningNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('running_mean', torch.zeros(1))  # tracked buffer
                self.register_buffer('running_var',  torch.ones(1))

        Use for: BatchNorm running statistics, positional encodings,
                 class prior probabilities, any fixed tensor that belongs
                 to the model but isn't trained.

### Module Introspection

    model.parameters()          # iterator over all trainable parameters
    model.named_parameters()    # (name, param) pairs — for selective freezing
    model.children()            # direct submodules
    model.modules()             # all submodules recursively
    model.state_dict()          # OrderedDict of all param+buffer tensors

    Count parameters:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    Freeze a submodule:
        for p in model.encoder.parameters():
            p.requires_grad = False

    Training vs eval mode:
        model.train()    # enables Dropout, sets BatchNorm to update running stats
        model.eval()     # disables Dropout, BatchNorm uses running stats (fixed)

### Built-in Layer Reference

    Linear layers:
        nn.Linear(in, out, bias=True)          # dense / fully-connected
        nn.Bilinear(in1, in2, out)             # bilinear: y = x1 @ A @ x2 + b

    Convolutional:
        nn.Conv1d(in_ch, out_ch, kernel)       # 1D convolution (sequences)
        nn.Conv2d(in_ch, out_ch, kernel)       # 2D convolution (images)
        nn.Conv3d(in_ch, out_ch, kernel)       # 3D convolution (video/volume)
        nn.ConvTranspose2d(in, out, kernel)    # transposed / "deconv"
        nn.DepthwiseSeparable (from torchvision)

    Recurrent:
        nn.RNN, nn.LSTM, nn.GRU                # standard recurrent cells
        # All accept batch_first=True to use (batch, seq, feat) layout

    Attention:
        nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    Normalisation:
        nn.BatchNorm1d/2d/3d                   # batch statistics normalisation
        nn.LayerNorm(normalised_shape)         # per-sample normalisation
        nn.GroupNorm(num_groups, num_channels) # between BN and LN
        nn.InstanceNorm2d                      # per-sample per-channel (style transfer)

    Activations:
        nn.ReLU(), nn.LeakyReLU(0.2)
        nn.GELU(), nn.SiLU()                   # modern activations (transformers)
        nn.Sigmoid(), nn.Tanh(), nn.Softmax(dim=1)

    Regularisation:
        nn.Dropout(p=0.5)                      # zero p fraction of elements
        nn.Dropout2d(p=0.5)                    # zero entire feature maps

    Pooling:
        nn.MaxPool2d(kernel_size, stride)
        nn.AvgPool2d(kernel_size, stride)
        nn.AdaptiveAvgPool2d((1, 1))           # global average pool to any size


##### PART 5 — THE TRAINING LOOP: DATASETS, DATALOADERS, OPTIMIZERS

### PyTorch's Explicit Training Loop Philosophy

    Unlike Keras's model.fit(), PyTorch gives you a raw training loop.
    You write the loop yourself. This is intentional: it makes every step
    visible, debuggable, and customisable.

    The canonical training loop:

        for epoch in range(num_epochs):
            model.train()                                # enable dropout, BN
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()                    # 1. clear old gradients
                logits = model(X_batch)                  # 2. forward pass
                loss = criterion(logits, y_batch)        # 3. compute loss
                loss.backward()                          # 4. backward pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()                         # 5. update weights

            model.eval()                                 # disable dropout, BN
            with torch.no_grad():
                val_loss = evaluate(model, val_loader)

### Dataset and DataLoader

    torch.utils.data.Dataset: abstract base class for your data.
        Must implement __len__() and __getitem__(idx).

        class TabularDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.long)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

    torch.utils.data.DataLoader: wraps Dataset with batching, shuffling,
        multiprocessing, and collation.

        loader = DataLoader(
            dataset,
            batch_size  = 64,
            shuffle     = True,          # re-shuffle every epoch
            num_workers = 4,             # parallel data loading processes
            pin_memory  = True,          # faster CPU→GPU transfer (host pinned memory)
            drop_last   = True,          # discard last incomplete batch
            persistent_workers = True,   # keep worker processes alive between epochs
        )

    num_workers advice:
        0: everything in main process (simplest, no fork overhead)
        4–8: typical for image datasets (I/O bound)
        Set to os.cpu_count() // 2 as a starting heuristic
        On Windows: multiprocessing requires if __name__ == '__main__' guard

### Transforms and Augmentation

    torchvision.transforms (image pipelines):
        from torchvision import transforms

        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),                        # [0,255] → [0,1] float32
            transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                                  [0.229, 0.224, 0.225]), # ImageNet std
        ])

        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    v2 transforms (torchvision.transforms.v2 — preferred for new code):
        - Support for bounding boxes, segmentation masks, keypoints
        - GPU-accelerated transforms via .to(device)
        - Works on both PIL images and tensors

### Loss Functions (Criteria)

    Classification:
        nn.CrossEntropyLoss()      # expects RAW LOGITS (no softmax before!)
                                   # internally: softmax + NLL
                                   # targets: class indices (int64)
        nn.BCEWithLogitsLoss()     # binary; expects logits, not sigmoid(logits)
        nn.NLLLoss()               # expects log_softmax output

    Regression:
        nn.MSELoss()               # mean squared error
        nn.L1Loss()                # mean absolute error
        nn.HuberLoss(delta=1.0)    # robust: L2 near 0, L1 for large errors

    Embedding:
        nn.TripletMarginLoss()     # metric learning
        nn.CosineEmbeddingLoss()   # cosine similarity objective

    CRITICAL: CrossEntropyLoss expects LOGITS (no activation on final layer).
    The most common bug: accidentally applying softmax before CrossEntropyLoss,
    which gives log(softmax(logits)) — numerically unstable and wrong.

### Optimizers

    SGD family:
        optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4,
                  nesterov=True)                        # classic SGD + Nesterov

    Adaptive:
        optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)
        optim.AdamW(params, lr=1e-3, weight_decay=1e-2)  # decoupled weight decay
                                                          # PREFERRED over Adam
        optim.RMSprop(params, lr=1e-3, alpha=0.99)

    Modern:
        optim.NAdam(params)        # Adam + Nesterov lookahead
        optim.RAdam(params)        # Rectified Adam (warm-up free)

    Param groups (different lr per layer — critical for fine-tuning):
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-5},  # frozen/slow
            {'params': model.head.parameters(),     'lr': 1e-3},  # fast
        ], weight_decay=1e-2)

### Learning Rate Schedulers

    torch.optim.lr_scheduler:

        StepLR(optimizer, step_size=30, gamma=0.1)
            # lr *= 0.1 every 30 epochs

        CosineAnnealingLR(optimizer, T_max=100)
            # lr follows cosine curve from initial to near-zero over T_max steps

        OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(loader), epochs=10)
            # Smith's 1-cycle policy: warm up → peak → cool down
            # Often gives faster convergence than constant lr

        ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            # Reduce lr when val_loss stops improving
            # Must call scheduler.step(val_loss) instead of scheduler.step()

    Scheduler step placement:
        # Most schedulers: step AFTER each epoch
        for epoch in range(N):
            train(...)
            scheduler.step()

        # OneCycleLR: step AFTER each BATCH
        for x, y in loader:
            train_step(x, y)
            scheduler.step()

### Saving and Loading Models

    Save/load only weights (recommended):
        torch.save(model.state_dict(), 'model.pt')
        model.load_state_dict(torch.load('model.pt', map_location=device))

    Save full checkpoint (for resuming training):
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':                 loss,
        }, 'checkpoint.pt')

        # Resume:
        ckpt = torch.load('checkpoint.pt')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    Why save only state_dict, not the whole model?
        torch.save(model, 'model.pt') also saves the class definition.
        This creates brittle dependencies on exact Python file paths.
        state_dict + model class definition = portable, safe, recommended.


##### PART 6 — AUTOMATIC MIXED PRECISION AND PERFORMANCE

### Why Mixed Precision?

    Modern GPUs (Volta, Turing, Ampere, Hopper) have dedicated Tensor Core
    hardware that runs bfloat16/float16 operations 2–8× faster than float32.

    The challenge: training in pure float16 leads to:
        - Underflow: small gradients round to zero (vanishing gradients)
        - Overflow: large activations produce inf/nan

    Solution: Automatic Mixed Precision (AMP):
        - Keeps weights in float32 (full precision)
        - Performs forward/backward in float16/bfloat16 (speed)
        - Uses a GradScaler to prevent gradient underflow

### torch.cuda.amp — The AMP API

    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()          # manages the loss scale factor

    for x, y in loader:
        optimizer.zero_grad()

        with autocast(dtype=torch.float16):      # forward in f16
            output = model(x)
            loss   = criterion(output, y)

        scaler.scale(loss).backward()            # scale loss before backward
        scaler.unscale_(optimizer)               # unscale grads before clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)                   # update (unscales internally)
        scaler.update()                          # adjust scale for next iter

    On Ampere (A100) or Hopper (H100): use bfloat16 instead — no GradScaler needed:
        with autocast(dtype=torch.bfloat16):
            output = model(x)                    # bf16: same exponent as f32
            loss   = criterion(output, y)        # no overflow/underflow risk
        loss.backward()                          # normal backward, no scaler

    Typical speedups:
        float16 AMP on V100: 1.5–2× throughput
        bfloat16 on A100: 2–3× throughput
        Memory reduction: ~50% (weights remain f32, activations in f16/bf16)


##### PART 7 — torch.compile: JIT COMPILATION IN PYTORCH 2.x

### The Problem torch.compile Solves

    PyTorch's eager execution is fast and debuggable, but pays overhead:
        - Python interpreter overhead per operation
        - Kernel launch overhead for each individual op
        - No cross-operation fusion (relu + matmul → separate kernels)

    TF1's static graph solved this by compiling everything upfront,
    but lost Python flexibility.

    torch.compile solves it differently: capture, compile, keep Python.

### The torch.compile Stack

    Dynamo (graph capture):
        Bytecode-level hooking into CPython's evaluation loop.
        Captures PyTorch operations as FX graphs.
        Falls back to eager for anything it can't capture (Python control flow,
        external libraries). Fallback is SILENT — model still runs correctly.

    AOTAutograd (ahead-of-time autograd):
        Traces through both forward AND backward passes before execution.
        Produces a joint graph for forward + backward that can be optimised together.

    Inductor (backend compiler):
        Takes the FX graph and generates optimised GPU code.
        On CUDA: generates Triton kernels (Python-like GPU kernels).
        On CPU: generates C++ code via OpenMP.
        Key optimisations: operator fusion, memory planning, layout optimisation.

### Using torch.compile

    One line change:
        model = torch.compile(model)            # that's it

    Compile modes (speed vs warmup trade-off):
        model = torch.compile(model, mode='default')       # balanced (default)
        model = torch.compile(model, mode='reduce-overhead') # lowest kernel launch overhead
        model = torch.compile(model, mode='max-autotune')  # slowest compile, fastest runtime

    Selective compilation (compile only the hot path):
        @torch.compile
        def train_step(model, x, y):
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            return loss

    Warmup: the first 1–3 batches trigger compilation (compilation is deferred).
    Subsequent batches run the compiled graph — typically 1.5–3× faster.

    torch.compile vs @tf.function:
        Both compile Python→graph at first call.
        @tf.function: traces using TF ops only; Python side-effects vanish.
        torch.compile: captures via bytecode; handles more Python patterns
                       (tries harder before falling back to eager).

### TorchScript — Ahead-of-Time Compilation for Deployment

    torch.compile is TRAINING and inference acceleration. It still requires Python.
    TorchScript produces a Python-INDEPENDENT serialised model.

    Two modes:
        Scripting (preferred): statically type-checks and compiles the model:
            scripted = torch.jit.script(model)
            scripted.save('model.pt')           # portable, no Python needed

            # Load in C++ (or Python):
            scripted = torch.jit.load('model.pt')
            output = scripted(input)

        Tracing: records operations on a concrete input (misses control flow):
            traced = torch.jit.trace(model, example_input)
            # Simpler but WRONG if model has data-dependent control flow

    TorchScript limitations:
        - Only a subset of Python is supported
        - No arbitrary Python objects or libraries
        - Type annotations required for complex signatures
        - Debugging compiled scripts is harder

    When to use TorchScript:
        Deploying to production C++ servers, Android, iOS, embedded systems
        where Python is not available.

### ONNX Export — The Universal Interchange Format

    ONNX (Open Neural Network Exchange) is an open format for ML models.
    Export once, run everywhere: ONNX Runtime, TensorRT, OpenVINO, CoreML.

        torch.onnx.export(
            model,
            args = example_input,
            f    = "model.onnx",
            input_names  = ['input'],
            output_names = ['output'],
            dynamic_axes = {'input':  {0: 'batch_size'},
                            'output': {0: 'batch_size'}},
            opset_version = 17,
        )

        # Run with ONNX Runtime (10-50% faster inference on CPU than PyTorch):
        import onnxruntime as ort
        sess = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
        output = sess.run(None, {'input': np_array})


##### PART 8 — DISTRIBUTED TRAINING: DDP AND FSDP

### DistributedDataParallel (DDP)

    DDP is the standard multi-GPU training strategy: model is replicated
    on each GPU, each GPU processes a different shard of the batch,
    gradients are synchronised via AllReduce (NCCL) before weight update.

    Identical to TF's MirroredStrategy, but requires explicit setup:

        # Launch script: python -m torch.distributed.launch --nproc_per_node=4 train.py

        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        dist.init_process_group(backend='nccl')        # initialise communication
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

        model = MyModel().to(local_rank)
        model = DDP(model, device_ids=[local_rank])    # wrap with DDP

        # DistributedSampler ensures each GPU sees different data
        sampler = DistributedSampler(dataset)
        loader  = DataLoader(dataset, sampler=sampler, batch_size=64)

        for epoch in range(N):
            sampler.set_epoch(epoch)                   # re-shuffle each epoch
            for x, y in loader:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()                        # AllReduce happens here automatically
                optimizer.step()

    Under the hood:
        1. Each GPU runs forward + backward independently on its data shard
        2. After backward(), DDP automatically AllReduces gradients across all GPUs
        3. All GPUs apply the same averaged gradient → weights stay identical
        4. Effective batch size = per_gpu_batch × num_gpus (scale lr accordingly)

### FSDP — Fully Sharded Data Parallel

    DDP keeps a FULL model copy on each GPU. For very large models (billions
    of parameters), this is impossible — the model doesn't fit on one GPU.

    FSDP shards model weights, gradients, AND optimiser states across GPUs:
        - Each GPU holds only 1/N of the parameters
        - Parameters are all-gathered just-in-time for each forward/backward layer
        - After use, parameters are discarded (sharded back to 1/N)

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Auto-wrap policy: each TransformerBlock gets its own FSDP unit
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock}
        )

        model = FSDP(model, auto_wrap_policy=wrap_policy,
                     mixed_precision=MixedPrecision(
                         param_dtype=torch.bfloat16,
                         reduce_dtype=torch.float32,
                     ))

    FSDP memory savings: 8 GPUs → each holds ~1/8 of parameters
        vs DDP: each GPU holds full model copy

    When to use DDP vs FSDP:
        Model fits on one GPU?    → DDP (simpler, same convergence)
        Model fits on one GPU?    → DDP with ZeRO stage 1/2 for optimizer states
        Model doesn't fit at all? → FSDP (necessary)

### Distributed Strategy Decision Guide

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Strategy       │ Model fits on GPU? │ When to use                    │
    ├──────────────────────────────────────────────────────────────────────┤
    │ DataParallel   │ Yes                │ Quick test, 1 machine only     │
    │                │                    │ Legacy: don't use for training │
    │ DDP            │ Yes                │ Standard multi-GPU training    │
    │                │                    │ Best throughput for most models│
    │ FSDP           │ No                 │ Large models (7B+ params)      │
    │                │                    │ Shards everything across GPUs  │
    │ Pipeline Par.  │ No (even FSDP)     │ Extreme scale (100B+ params)   │
    │                │                    │ Splits model layers across GPUs│
    └──────────────────────────────────────────────────────────────────────┘

    KEY RULE: when scaling from 1→N GPUs with DDP, multiply global batch
    size by N and multiply base learning rate by √N (square root rule)
    or by N with a linear warmup phase (linear scaling rule from the
    Facebook ResNet paper).

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Autograd Deep Dive — Dynamic Graphs, Backward & Higher-Order Grads": {
        "description": (
            "Deep dive into PyTorch's Autograd engine. "
            "Dynamic graph construction, requires_grad, grad_fn chain. "
            "Manual training step vs torch.optim. "
            "torch.no_grad and detach for inference. "
            "Gradient accumulation for large batches. "
            "Higher-order derivatives with create_graph. "
            "Custom autograd Functions via torch.autograd.Function. "
            "Benchmark eager vs torch.compile training step."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 65)
print("  AUTOGRAD DEEP DIVE — DYNAMIC GRAPHS & DIFFERENTIATION")
print("=" * 65)
print()
print(f"  PyTorch version:   {torch.__version__}")
print(f"  CUDA available:    {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device:            {device}")
print()

torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Tensor grad_fn chain — the dynamic graph
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — grad_fn chain: the dynamic computational graph")
print("━" * 65)
print()

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2         # y = x²
z = y + 2 * x + 1  # z = x² + 2x + 1 = (x+1)²

print(f"  x = {x.item()},  requires_grad={x.requires_grad},  grad_fn={x.grad_fn}")
print(f"  y = x²  = {y.item()},  grad_fn={y.grad_fn}")
print(f"  z = y+2x+1 = {z.item()},  grad_fn={z.grad_fn}")
print()

z.backward()       # dz/dx = 2x + 2 = 2*3+2 = 8
print(f"  z.backward() → x.grad = {x.grad.item():.4f}  (analytic 2x+2 = {2*3+2})")
print()

# Graph is freed after backward — x.grad accumulates if called again
x2 = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
loss = (x2 ** 3).sum()   # sum of cubes
loss.backward()
print(f"  x = {x2.detach().numpy()}")
print(f"  d(sum(x³))/dx = 3x² = {(3 * x2**2).detach().numpy()}")
print(f"  x.grad       =       {x2.grad.numpy()}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Manual parameter update vs torch.optim
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Manual gradient descent vs torch.optim")
print("━" * 65)
print()

# Simple linear regression: y = 2x + 3
torch.manual_seed(0)
X_data = torch.linspace(0, 1, 100).unsqueeze(1)
y_data = 2.0 * X_data + 3.0 + 0.1 * torch.randn_like(X_data)

w_manual = torch.tensor([0.0], requires_grad=True)
b_manual = torch.tensor([0.0], requires_grad=True)

lr = 0.5
for step in range(200):
    pred = X_data * w_manual + b_manual
    loss = ((pred - y_data) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w_manual -= lr * w_manual.grad
        b_manual -= lr * b_manual.grad
    w_manual.grad.zero_()     # MUST clear gradients
    b_manual.grad.zero_()

print(f"  Manual SGD (200 steps, lr={lr}):")
print(f"    w = {w_manual.item():.4f}  (target 2.0)")
print(f"    b = {b_manual.item():.4f}  (target 3.0)")
print()

# Same with torch.optim (cleaner):
model_lr = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model_lr.parameters(), lr=0.5)

for step in range(200):
    optimizer.zero_grad()                          # clear grads
    pred = model_lr(X_data)
    loss = F.mse_loss(pred, y_data)
    loss.backward()
    optimizer.step()

print(f"  torch.optim.SGD (200 steps, lr=0.5):")
print(f"    w = {model_lr.weight.item():.4f}  (target 2.0)")
print(f"    b = {model_lr.bias.item():.4f}  (target 3.0)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: torch.no_grad and detach
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — torch.no_grad() and detach()")
print("━" * 65)
print()

model_eval = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
x_eval = torch.randn(32, 8)

# Without no_grad: builds graph (wasted memory during inference)
t0 = time.perf_counter()
for _ in range(1000):
    out = model_eval(x_eval)
t_eager = (time.perf_counter() - t0) / 1000 * 1000

# With no_grad: skips graph construction
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(1000):
        out = model_eval(x_eval)
t_nograd = (time.perf_counter() - t0) / 1000 * 1000

print(f"  Inference timing (1000 passes, batch=32):")
print(f"    With grad:     {t_eager:.4f} ms/pass")
print(f"    no_grad:       {t_nograd:.4f} ms/pass")
print(f"    Speedup:       {t_eager/t_nograd:.2f}×")
print()

# detach(): stop gradients from flowing into a sub-graph
teacher = nn.Linear(8, 4)
student = nn.Linear(8, 4)
x_kd = torch.randn(16, 8)

teacher_out = teacher(x_kd).detach()    # stop gradients — teacher is frozen
student_out = student(x_kd)
kd_loss = F.mse_loss(student_out, teacher_out)   # gradients only flow through student
kd_loss.backward()
print(f"  Knowledge distillation detach demo:")
print(f"    teacher.weight.grad: {teacher.weight.grad}  (None — detached!)")
print(f"    student.weight.grad max: {student.weight.grad.abs().max().item():.6f}  ✓")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Gradient accumulation (simulate larger batch)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Gradient accumulation (simulate large batch)")
print("━" * 65)
print()

model_acc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
opt_acc = torch.optim.AdamW(model_acc.parameters(), lr=1e-3)

ACCUM_STEPS = 4    # effective batch = 16 * 4 = 64

losses_accum = []
for step in range(20):
    opt_acc.zero_grad()
    accum_loss = 0.0
    for micro in range(ACCUM_STEPS):
        X_micro = torch.randn(16, 64)                         # micro-batch 16
        y_micro = torch.randint(0, 10, (16,))
        logits  = model_acc(X_micro)
        loss    = F.cross_entropy(logits, y_micro) / ACCUM_STEPS   # scale!
        loss.backward()                                        # grads ACCUMULATE
        accum_loss += loss.item()
    opt_acc.step()                                             # update once per full batch
    losses_accum.append(accum_loss)

print(f"  Gradient accumulation: 4 micro-batches × 16 = effective batch 64")
print(f"  Steps 1-5 losses: {[f'{l:.4f}' for l in losses_accum[:5]]}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Higher-order derivatives
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Higher-order derivatives (create_graph)")
print("━" * 65)
print()

x_ho = torch.tensor(2.0, requires_grad=True)
y_ho = x_ho ** 4   # y = x⁴

# First derivative: dy/dx = 4x³
dy_dx = torch.autograd.grad(y_ho, x_ho, create_graph=True)[0]

# Second derivative: d²y/dx² = 12x²
d2y_dx2 = torch.autograd.grad(dy_dx, x_ho, create_graph=True)[0]

# Third derivative: d³y/dx³ = 24x
d3y_dx3 = torch.autograd.grad(d2y_dx2, x_ho)[0]

xv = x_ho.item()
print(f"  y = x⁴,   x = {xv}")
print(f"  dy/dx    = 4x³   → {dy_dx.item():.2f}   (analytic: {4*xv**3:.2f})")
print(f"  d²y/dx²  = 12x²  → {d2y_dx2.item():.2f}  (analytic: {12*xv**2:.2f})")
print(f"  d³y/dx³  = 24x   → {d3y_dx3.item():.2f}  (analytic: {24*xv:.2f})")
print()

# WGAN-GP gradient penalty (practical higher-order use)
D_gp = nn.Linear(4, 1)
x_hat = torch.randn(8, 4, requires_grad=True)
d_out = D_gp(x_hat).sum()
grad_d = torch.autograd.grad(d_out, x_hat, create_graph=True)[0]
gp = ((grad_d.norm(dim=1) - 1.0) ** 2).mean()
print(f"  WGAN-GP gradient penalty: {gp.item():.4f}")
print(f"  Gradient norms at x_hat:  {grad_d.norm(dim=1).detach().numpy().round(3)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Custom autograd.Function
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Custom autograd.Function (straight-through estimator)")
print("━" * 65)
print()

class StraightThroughHeaviside(torch.autograd.Function):
    """
    Forward: Heaviside step function (not differentiable at 0).
    Backward: Straight-through estimator — pass gradient as-is.
    Used in: binary neural networks, VQ-VAE codebook commitment.
    """
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output   # gradient passes through unchanged

heaviside_ste = StraightThroughHeaviside.apply

x_ste = torch.tensor([-0.5, 0.1, -0.3, 0.8], requires_grad=True)
out_ste = heaviside_ste(x_ste)
loss_ste = out_ste.sum()
loss_ste.backward()

print(f"  Input x:                   {x_ste.detach().numpy()}")
print(f"  Heaviside(x):              {out_ste.detach().numpy()}  (binary)")
print(f"  Gradient (STE):            {x_ste.grad.numpy()}  (passed through)")
print(f"  Without STE, gradient would be 0 everywhere (non-differentiable)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 7: Eager vs torch.compile benchmark
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 7 — Eager vs torch.compile training step benchmark")
print("━" * 65)
print()

bench_model = nn.Sequential(
    nn.Linear(128, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 10),
).to(device)

bench_opt = torch.optim.AdamW(bench_model.parameters(), lr=1e-3)
X_bench = torch.randn(64, 128, device=device)
y_bench = torch.randint(0, 10, (64,), device=device)

def eager_train_step():
    bench_opt.zero_grad()
    loss = F.cross_entropy(bench_model(X_bench), y_bench)
    loss.backward()
    bench_opt.step()
    return loss.item()

# Attempt torch.compile (PyTorch 2.0+)
try:
    compiled_model = torch.compile(bench_model, mode='default')
    compiled_opt   = torch.optim.AdamW(compiled_model.parameters(), lr=1e-3)

    def compiled_train_step():
        compiled_opt.zero_grad()
        loss = F.cross_entropy(compiled_model(X_bench), y_bench)
        loss.backward()
        compiled_opt.step()
        return loss.item()

    # Warmup (compilation happens here)
    for _ in range(5):
        compiled_train_step()

    N = 200
    t0 = time.perf_counter()
    for _ in range(N): eager_train_step()
    t_eager_ms = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N): compiled_train_step()
    t_compiled_ms = (time.perf_counter() - t0) / N * 1000

    print(f"  3-layer MLP, batch=64, features=128  ({N} steps each, device={device})")
    print(f"    Eager:           {t_eager_ms:.3f} ms/step")
    print(f"    torch.compile:   {t_compiled_ms:.3f} ms/step")
    print(f"    Speedup:         {t_eager_ms / t_compiled_ms:.2f}×")
    print()
    print("  torch.compile speedup is larger on GPU (1.5–3×).")
    print("  On CPU, overhead from Triton codegen may reduce gains.")
    print("  Best gains: small models where Python overhead dominates compute.")
except Exception as e:
    print(f"  torch.compile not available ({e}), showing eager only.")
    N = 200
    t0 = time.perf_counter()
    for _ in range(N): eager_train_step()
    t_eager_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Eager: {t_eager_ms:.3f} ms/step  (install PyTorch 2.0+ for torch.compile)")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · nn.Module Full Stack — Three Patterns, Custom Layers & Training Loop": {
        "description": (
            "Complete nn.Module usage across all three abstraction levels. "
            "nn.Sequential for linear stacks. Functional-style with custom "
            "residual blocks (skip connections, layer norm). "
            "Full subclassing for a transformer encoder block. "
            "Custom nn.Module with nn.Parameter and register_buffer. "
            "Explicit PyTorch training loop with validation, early stopping, "
            "lr scheduling. Model saving and loading via state_dict."
        ),
        "language": "python",
        "code": '''
import numpy as np
import os, tempfile, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

print("=" * 65)
print("  nn.MODULE FULL STACK — PATTERNS, LAYERS & TRAINING LOOP")
print("=" * 65)
print()

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  PyTorch: {torch.__version__} | Device: {device}")
print()

# ── Shared dataset ─────────────────────────────────────────────────────
N_TRAIN, N_VAL, N_FEAT, N_CLS = 2000, 400, 32, 5
X_train = torch.randn(N_TRAIN, N_FEAT)
y_train = torch.randint(0, N_CLS, (N_TRAIN,))
X_val   = torch.randn(N_VAL, N_FEAT)
y_val   = torch.randint(0, N_CLS, (N_VAL,))

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val),
                          batch_size=64, shuffle=False)

print(f"  Dataset: {N_TRAIN} train / {N_VAL} val | {N_FEAT} features | {N_CLS} classes")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: nn.Sequential
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — nn.Sequential (linear stack)")
print("━" * 65)
print()

model_seq = nn.Sequential(
    nn.Linear(N_FEAT, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, N_CLS),
)

total_params = sum(p.numel() for p in model_seq.parameters())
trainable    = sum(p.numel() for p in model_seq.parameters() if p.requires_grad)
print(f"  Total params:    {total_params:,}")
print(f"  Trainable:       {trainable:,}")
print(f"  Layer types:     {set(type(l).__name__ for l in model_seq)}")
print()

# Quick 5-epoch training
opt_seq = torch.optim.AdamW(model_seq.parameters(), lr=1e-3)

def train_epoch(model, loader, opt):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n

def val_epoch(model, loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            loss   = F.cross_entropy(logits, y)
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += len(y)
    return total_loss / n, correct / n

print(f"  {'Epoch':<8} {'Train Loss':>12} {'Train Acc':>12} {'Val Acc':>12}")
print(f"  {'─'*48}")
for epoch in range(5):
    tr_l, tr_a = train_epoch(model_seq, train_loader, opt_seq)
    vl_l, vl_a = val_epoch(model_seq, val_loader)
    print(f"  {epoch+1:<8} {tr_l:12.4f} {tr_a:12.4f} {vl_a:12.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Subclassed model with residual blocks
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Subclassed model with residual blocks")
print("━" * 65)
print()

class ResidualBlock(nn.Module):
    """Pre-activation residual block: norm → relu → linear → residual add."""
    def __init__(self, d_model):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.drop   = nn.Dropout(0.1)

    def forward(self, x):
        # Pre-norm: normalise BEFORE the transformation (GPT-2 style)
        return x + self.drop(self.linear(F.relu(self.norm(x))))

class ResNetMLP(nn.Module):
    def __init__(self, in_features, hidden_dim, n_blocks, n_classes):
        super().__init__()
        self.embed  = nn.Linear(in_features, hidden_dim)      # project to hidden dim
        self.blocks = nn.ModuleList(                           # nn.ModuleList tracks params!
            [ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        )
        self.head   = nn.Linear(hidden_dim, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.embed(x))           # embed to hidden dim
        for block in self.blocks:           # dynamic loop — pure Python
            x = block(x)
        return self.head(x)                 # logits

model_res = ResNetMLP(N_FEAT, 64, n_blocks=4, n_classes=N_CLS)
opt_res   = torch.optim.AdamW(model_res.parameters(), lr=1e-3, weight_decay=1e-2)
sched_res = torch.optim.lr_scheduler.CosineAnnealingLR(opt_res, T_max=10)

print(f"  ResNetMLP (4 residual blocks) | Params: {sum(p.numel() for p in model_res.parameters()):,}")
print()
print(f"  {'Epoch':<8} {'Train Loss':>12} {'Val Acc':>12} {'LR':>14}")
print(f"  {'─'*50}")
for epoch in range(10):
    tr_l, tr_a = train_epoch(model_res, train_loader, opt_res)
    vl_l, vl_a = val_epoch(model_res, val_loader)
    current_lr = opt_res.param_groups[0]['lr']
    sched_res.step()
    print(f"  {epoch+1:<8} {tr_l:12.4f} {vl_a:12.4f} {current_lr:14.6f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Transformer encoder block (multihead attention)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Transformer encoder block")
print("━" * 65)
print()

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),                          # GELU: smoother than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch, seq_len, d_model)
        attn_out, attn_weights = self.attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
        )
        x = self.norm1(x + self.drop(attn_out))    # residual + layer norm
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x, attn_weights

class TransformerClassifier(nn.Module):
    def __init__(self, n_tokens, d_model, n_heads, n_layers, n_classes, max_len=128):
        super().__init__()
        self.embed    = nn.Embedding(n_tokens, d_model)
        self.pos_emb  = nn.Embedding(max_len, d_model)    # learned positional encoding
        self.layers   = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, n_classes)

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos  = torch.arange(T, device=token_ids.device).unsqueeze(0)  # (1, T)
        x    = self.embed(token_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm(x)
        cls_token = x[:, 0, :]               # use [CLS] token representation
        return self.head(cls_token)

tformer = TransformerClassifier(
    n_tokens=1000, d_model=64, n_heads=4, n_layers=2, n_classes=N_CLS
)
dummy_tokens = torch.randint(0, 1000, (8, 32))   # batch=8, seq_len=32
out = tformer(dummy_tokens)
print(f"  TransformerClassifier:")
print(f"    Params:          {sum(p.numel() for p in tformer.parameters()):,}")
print(f"    Input shape:     {tuple(dummy_tokens.shape)}  (batch, seq_len)")
print(f"    Output shape:    {tuple(out.shape)}  (batch, n_classes)")
print(f"    Output (logits): {out.detach().numpy().round(3)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Custom nn.Module with nn.Parameter and register_buffer
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Custom nn.Parameter and register_buffer")
print("━" * 65)
print()

class ScaledDotProductAttention(nn.Module):
    """
    Demonstrates nn.Parameter (learnable) vs register_buffer (fixed but
    moves with .to(device) and is saved in state_dict).
    """
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Parameter(torch.randn(d_model, d_model) * 0.02)   # learnable
        self.W_k = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_v = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        # Fixed scale factor: 1/sqrt(d_model)
        self.register_buffer('scale', torch.tensor(d_model ** -0.5))    # buffer: not trained

    def forward(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        scores = (Q @ K.transpose(-2, -1)) * self.scale    # buffer used here
        attn   = F.softmax(scores, dim=-1)
        return attn @ V

sdpa = ScaledDotProductAttention(d_model=16)
print(f"  ScaledDotProductAttention(d_model=16):")
for name, p in sdpa.named_parameters():
    print(f"    Parameter: {name:<15}  shape={tuple(p.shape)}  requires_grad={p.requires_grad}")
print()
for name, b in sdpa.named_buffers():
    print(f"    Buffer:    {name:<15}  value={b.item():.4f}  (not in parameters())")
print()

x_sdpa = torch.randn(2, 4, 16)   # (batch, seq, d_model)
out_sdpa = sdpa(x_sdpa)
print(f"    Input  shape: {tuple(x_sdpa.shape)}")
print(f"    Output shape: {tuple(out_sdpa.shape)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Model saving and loading via state_dict
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Model saving and loading (state_dict)")
print("━" * 65)
print()

with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, 'model.pt')

    # Save checkpoint (weights + optimiser state for resuming)
    torch.save({
        'model_state_dict': model_res.state_dict(),
        'optimizer_state_dict': opt_res.state_dict(),
        'epoch': 10,
        'val_acc': vl_a,
    }, ckpt_path)

    size_kb = os.path.getsize(ckpt_path) / 1024
    print(f"  Saved checkpoint: {size_kb:.1f} KB")

    # Load and verify
    model_loaded = ResNetMLP(N_FEAT, 64, n_blocks=4, n_classes=N_CLS)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_loaded.load_state_dict(ckpt['model_state_dict'])
    model_loaded.eval()

    # Verify identical outputs
    with torch.no_grad():
        x_check = torch.randn(4, N_FEAT)
        out_orig   = model_res(x_check)
        out_loaded = model_loaded(x_check)
        max_diff   = (out_orig - out_loaded).abs().max().item()

    print(f"  Loaded from checkpoint (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    print(f"  Max output difference (orig vs loaded): {max_diff:.2e}  ✅")
print()

print("  OPTIMIZER DECISION GUIDE:")
print("  ┌─────────────────────────────────────────────────────────────────┐")
print("  │ Optimizer     │ Best for                                        │")
print("  ├─────────────────────────────────────────────────────────────────┤")
print("  │ SGD+momentum  │ ConvNets, image classification (well-tuned lr)  │")
print("  │ AdamW         │ Transformers, NLP, most default scenarios        │")
print("  │ RMSprop       │ RNNs, RL (adaptive per-weight lr)               │")
print("  │ LBFGS         │ Scientific computing, small models, full batch   │")
print("  └─────────────────────────────────────────────────────────────────┘")
print()
print("  SCHEDULER DECISION GUIDE:")
print("  ┌─────────────────────────────────────────────────────────────────┐")
print("  │ Scheduler          │ Best for                                   │")
print("  ├─────────────────────────────────────────────────────────────────┤")
print("  │ CosineAnnealingLR  │ Most training runs, smooth decay           │")
print("  │ OneCycleLR         │ Fast convergence, less hypertuning         │")
print("  │ ReduceLROnPlateau  │ When you don't know how long to train      │")
print("  │ LinearWarmup+Cosine│ Transformers (warm up then decay)          │")
print("  └─────────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · DataLoader & Dataset — Custom Data, Transforms & Samplers": {
        "description": (
            "Deep dive into PyTorch's data pipeline. "
            "Custom Dataset with __getitem__ and __len__. "
            "Image dataset with torchvision transforms pipeline. "
            "Collate functions for variable-length sequences. "
            "WeightedRandomSampler for imbalanced datasets. "
            "Automatic Mixed Precision (AMP) training loop. "
            "Performance profiling: num_workers vs pin_memory. "
            "TorchScript export and ONNX export for deployment."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset, DataLoader, WeightedRandomSampler, Subset
)
from torch.amp import autocast, GradScaler

print("=" * 65)
print("  DATALOADER & DATASET — DATA PIPELINES, AMP & EXPORT")
print("=" * 65)
print()

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  PyTorch: {torch.__version__} | Device: {device}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Custom Dataset
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Custom Dataset and DataLoader")
print("━" * 65)
print()

class SyntheticTabularDataset(Dataset):
    """
    A synthetic tabular dataset with features and multi-class labels.
    Demonstrates the minimal Dataset interface: __len__ and __getitem__.
    """
    def __init__(self, n_samples=1000, n_features=64, n_classes=5,
                 noise=0.3, seed=42):
        rng = np.random.RandomState(seed)

        # Generate cluster centres, one per class
        centres = rng.randn(n_classes, n_features)
        labels  = rng.randint(0, n_classes, n_samples)
        X = centres[labels] + noise * rng.randn(n_samples, n_features)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def class_counts(self):
        return torch.bincount(self.y)

full_dataset = SyntheticTabularDataset(n_samples=2000, n_features=64, n_classes=5)
n_train = int(0.8 * len(full_dataset))

train_set = Subset(full_dataset, range(n_train))
val_set   = Subset(full_dataset, range(n_train, len(full_dataset)))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False)

print(f"  Full dataset:  {len(full_dataset)} samples")
print(f"  Train / Val:   {len(train_set)} / {len(val_set)}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Class counts:  {full_dataset.class_counts().numpy()}")
print()

# Inspect a batch
X_sample, y_sample = next(iter(train_loader))
print(f"  Sample batch — X: {tuple(X_sample.shape)}, y: {tuple(y_sample.shape)}")
print(f"  X dtype: {X_sample.dtype}, y dtype: {y_sample.dtype}")
print(f"  X range: [{X_sample.min():.2f}, {X_sample.max():.2f}]")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: WeightedRandomSampler (imbalanced datasets)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — WeightedRandomSampler for class imbalance")
print("━" * 65)
print()

# Create an imbalanced dataset (class 0 has 10× more samples)
imbalanced = SyntheticTabularDataset(n_samples=2000, n_features=16, n_classes=3, seed=1)
# Force imbalance: keep all of class 0, subsample others
imbalanced.y[:1500] = 0   # 75% class 0
imbalanced.y[1500:1750] = 1  # 12.5% class 1
imbalanced.y[1750:] = 2  # 12.5% class 2

counts = imbalanced.class_counts().float()
print(f"  Imbalanced class distribution: {counts.int().numpy()}")

# Compute per-sample weights: inverse of class frequency
class_weights = 1.0 / counts
sample_weights = class_weights[imbalanced.y]

sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(imbalanced),
    replacement = True,          # with replacement (standard)
)

balanced_loader = DataLoader(imbalanced, batch_size=64, sampler=sampler)

# Verify: check class distribution in a few batches
all_labels = []
for i, (_, y) in enumerate(balanced_loader):
    all_labels.append(y)
    if i >= 9: break

y_all = torch.cat(all_labels)
balanced_counts = torch.bincount(y_all, minlength=3)
print(f"  Sampled 10 batches — class distribution: {balanced_counts.numpy()}")
print(f"  (Should be approximately balanced)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Custom collate_fn (variable-length sequences)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Custom collate_fn (variable-length sequences)")
print("━" * 65)
print()

class VariableLengthDataset(Dataset):
    """Each sample has a different sequence length — simulates text / time series."""
    def __init__(self, n=200, vocab_size=50, min_len=4, max_len=16, n_classes=3):
        self.data = [(
            torch.randint(0, vocab_size, (torch.randint(min_len, max_len+1, ()).item(),)),
            torch.randint(0, n_classes, ()).item()
        ) for _ in range(n)]

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def pad_collate(batch):
    """Pad variable-length sequences and pack into a batch tensor."""
    sequences, labels = zip(*batch)
    lengths  = torch.tensor([len(s) for s in sequences])
    max_len  = lengths.max().item()
    # Pad all sequences to max_len with zeros
    padded   = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded, torch.tensor(labels), lengths

vl_dataset = VariableLengthDataset(n=200)
vl_loader  = DataLoader(vl_dataset, batch_size=8, collate_fn=pad_collate, shuffle=True)

padded_batch, labels_batch, lengths_batch = next(iter(vl_loader))
print(f"  Variable-length batch:")
print(f"    Padded shape: {tuple(padded_batch.shape)}  (batch, max_seq_len)")
print(f"    Lengths:      {lengths_batch.numpy()}")
print(f"    Labels:       {labels_batch.numpy()}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Automatic Mixed Precision (AMP) training
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Automatic Mixed Precision (AMP) training")
print("━" * 65)
print()

model_amp = nn.Sequential(
    nn.Linear(64, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 5),
).to(device)

N_AMP_STEPS = 50

def run_training(use_amp, use_bf16=False, n_steps=N_AMP_STEPS):
    model = nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 5),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler('cuda', enabled=(use_amp and not use_bf16 and device.type == 'cuda'))
    losses = []

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    t0 = time.perf_counter()
    for step in range(n_steps):
        X = torch.randn(64, 64, device=device)
        y = torch.randint(0, 5, (64,), device=device)

        opt.zero_grad()
        if use_amp and device.type == 'cuda':
            with autocast('cuda', dtype=amp_dtype):
                logits = model(X)
                loss   = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(X)
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        losses.append(loss.item())

    t_total = time.perf_counter() - t0
    return t_total / n_steps * 1000, np.mean(losses)

t_fp32, l_fp32   = run_training(use_amp=False)
t_amp_f16, l_f16 = run_training(use_amp=True, use_bf16=False)
t_amp_bf16, l_bf16 = run_training(use_amp=True, use_bf16=True)

print(f"  AMP benchmark ({N_AMP_STEPS} steps, batch=64, device={device}):")
print(f"  {'Mode':<20} {'ms/step':>10} {'Avg Loss':>12}")
print(f"  {'─'*44}")
print(f"  {'FP32 (baseline)':<20} {t_fp32:10.3f} {l_fp32:12.4f}")
print(f"  {'AMP float16':<20} {t_amp_f16:10.3f} {l_f16:12.4f}")
print(f"  {'AMP bfloat16':<20} {t_amp_bf16:10.3f} {l_bf16:12.4f}")
print()
if device.type == 'cpu':
    print("  (AMP speedup is primarily on CUDA GPUs with Tensor Cores)")
    print("  On A100/H100 with bfloat16: expect 2–3× throughput improvement")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: DataLoader performance — num_workers and pin_memory
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — DataLoader performance: num_workers & pin_memory")
print("━" * 65)
print()

PERF_PATTERN = """
# DataLoader performance settings and their effects:

loader_slow = DataLoader(
    dataset,
    batch_size  = 64,
    num_workers = 0,        # single-process: I/O blocks training
    pin_memory  = False,    # pageable memory: slower GPU transfer
)

loader_fast = DataLoader(
    dataset,
    batch_size         = 64,
    num_workers        = 4,     # parallel prefetch workers (OS processes)
    pin_memory         = True,  # page-locked memory: faster CUDA transfer
    persistent_workers = True,  # keep workers alive between epochs
    prefetch_factor    = 2,     # batches prefetched per worker (default 2)
    drop_last          = True,  # avoid variable last-batch overhead
)

# num_workers heuristic:
#   num_workers = 0  → simple datasets, debugging (no fork overhead)
#   num_workers = 2  → light preprocessing (tabular, small tensors)
#   num_workers = 4-8 → image datasets (heavy I/O + augmentation)
#   num_workers > 8  → rarely helps; bottleneck shifts to CPU decode

# pin_memory=True:
#   Allocates CPU tensors in page-locked (pinned) memory.
#   .to(device) uses async DMA transfer → overlaps with GPU compute.
#   Only relevant when transferring to CUDA. Ignore for MPS/CPU.

# On Windows: multiprocessing requires:
#   if __name__ == '__main__':
#       train(...)    # all DataLoader code inside this guard
"""
print(PERF_PATTERN)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: ONNX export
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — ONNX export and inference")
print("━" * 65)
print()

import tempfile, os

export_model = nn.Sequential(
    nn.Linear(32, 64), nn.ReLU(),
    nn.Linear(64, 10),
)
export_model.eval()
dummy_input = torch.randn(1, 32)

with tempfile.TemporaryDirectory() as tmp:
    onnx_path = os.path.join(tmp, 'model.onnx')
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        input_names    = ['input'],
        output_names   = ['logits'],
        dynamic_axes   = {'input':  {0: 'batch_size'},
                          'logits': {0: 'batch_size'}},
        opset_version  = 17,
    )

    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  ONNX model exported: {size_kb:.1f} KB")

    # Run inference with onnxruntime (if available)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        X_test_np = np.random.randn(4, 32).astype(np.float32)
        ort_out = sess.run(None, {'input': X_test_np})[0]

        with torch.no_grad():
            pt_out = export_model(torch.from_numpy(X_test_np)).numpy()

        max_diff = np.abs(ort_out - pt_out).max()
        print(f"  ORT inference output shape: {ort_out.shape}")
        print(f"  Max difference PyTorch vs ORT: {max_diff:.2e}  ✅")

    except ImportError:
        print("  onnxruntime not installed. Export successful — run with:")
        print("    pip install onnxruntime")
        print("    sess = ort.InferenceSession('model.onnx')")
        print("    out  = sess.run(None, {'input': X_np})[0]")
        print()
        # Verify the ONNX file is valid via torch
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"  ONNX model validation: PASSED ✅")
        except ImportError:
            print("  (Install onnx or onnxruntime to validate/run the model)")

print()
print("  ONNX is the bridge to:")
print("    ONNX Runtime     → fastest CPU/GPU inference (Microsoft)")
print("    TensorRT         → NVIDIA GPU inference (CUDA graphs + INT8)")
print("    OpenVINO         → Intel CPU/VPU inference (Intel)")
print("    CoreML Tools     → Apple device deployment (iOS / macOS)")
print("    ONNX.js Runtime  → browser inference (JavaScript)")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Distributed Training & Deployment — DDP, TorchScript & Profiling": {
        "description": (
            "Multi-GPU training with DistributedDataParallel (DDP) patterns. "
            "DistributedSampler for correct data sharding. "
            "Gradient clipping and norm monitoring. "
            "torch.jit.script (TorchScript) for Python-free deployment. "
            "torch.jit.trace and its limitations. "
            "torch.profiler for identifying bottlenecks. "
            "Quantization-aware training (QAT) with torch.quantization. "
            "Full production checklist: compile, export, benchmark."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time, os, tempfile, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

print("=" * 65)
print("  DISTRIBUTED TRAINING & DEPLOYMENT — DDP, SCRIPT & PROFILE")
print("=" * 65)
print()

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  PyTorch: {torch.__version__} | Device: {device}")
print(f"  GPUs:    {torch.cuda.device_count()}")
print()

# ── Shared dataset and model ───────────────────────────────────────────
N, N_FEAT, C = 2000, 64, 10   # renamed F→N_FEAT: F is already imported as torch.nn.functional
X_all = torch.randn(N, N_FEAT)
y_all = torch.randint(0, C, (N,))
dataset = TensorDataset(X_all, y_all)

class DeepMLP(nn.Module):
    def __init__(self, in_features=64, hidden=256, n_classes=10, n_layers=4):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.LayerNorm(hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU()]
        layers += [nn.Linear(hidden, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # type annotations needed for TorchScript
        return self.net(x)

model = DeepMLP().to(device)
print(f"  DeepMLP params: {sum(p.numel() for p in model.parameters()):,}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: DDP pattern (shown as runnable single-process mock)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — DDP pattern and DistributedSampler")
print("━" * 65)
print()

DDP_PATTERN = """
# ── Full DDP launch pattern ───────────────────────────────────────────
# Save as train_ddp.py, launch with:
#   torchrun --nproc_per_node=4 train_ddp.py
# (torchrun is the modern replacement for torch.distributed.launch)

import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def train():
    # torchrun sets LOCAL_RANK automatically
    local_rank  = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size  = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend='nccl')          # NCCL: fastest GPU comms
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Model: same code, wrapped with DDP
    model = DeepMLP().to(device)
    model = DDP(model, device_ids=[local_rank])
    # model.module  → access the unwrapped model

    # Dataset: DistributedSampler shards data across workers
    sampler = DistributedSampler(dataset,
                                  num_replicas=world_size,
                                  rank=global_rank,
                                  shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler,
                        pin_memory=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 * world_size)
    # Scale lr by world_size (linear scaling rule) or sqrt(world_size)

    for epoch in range(10):
        sampler.set_epoch(epoch)                      # different shuffle each epoch
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()                           # AllReduce happens here
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Only rank 0 saves checkpoints
        if global_rank == 0 and (epoch + 1) % 5 == 0:
            torch.save(model.module.state_dict(), f'ckpt_epoch{epoch+1}.pt')

    dist.destroy_process_group()

if __name__ == '__main__':
    train()

# What DDP does:
# 1. Each process (rank) loads a different shard of data (DistributedSampler)
# 2. All ranks run forward + backward on their shard independently
# 3. After backward(), DDP hooks AllReduce to average gradients via NCCL
# 4. All ranks apply the SAME averaged gradient → weights stay in sync
# 5. Effective batch = per_rank_batch × world_size
"""
print(DDP_PATTERN)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Gradient clipping and norm monitoring
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Gradient clipping and norm monitoring")
print("━" * 65)
print()

model_clip = DeepMLP().to(device)
opt_clip   = torch.optim.AdamW(model_clip.parameters(), lr=1e-3)
loader_clip = DataLoader(dataset, batch_size=64, shuffle=True)

grad_norms, clipped_norms = [], []
for i, (X, y) in enumerate(loader_clip):
    if i >= 20: break
    X, y = X.to(device), y.to(device)

    opt_clip.zero_grad()
    loss = F.cross_entropy(model_clip(X), y)
    loss.backward()

    # Measure gradient norm BEFORE clipping
    raw_norm = torch.nn.utils.get_total_norm(
        [p.grad for p in model_clip.parameters() if p.grad is not None]
    ) if hasattr(torch.nn.utils, 'get_total_norm') else torch.sqrt(
        sum(p.grad.norm()**2 for p in model_clip.parameters() if p.grad is not None)
    )

    # Clip gradients (in-place) — returns norm BEFORE clip
    clip_norm = torch.nn.utils.clip_grad_norm_(
        model_clip.parameters(), max_norm=1.0
    )
    grad_norms.append(float(raw_norm))
    clipped_norms.append(float(clip_norm))

    opt_clip.step()

avg_raw     = np.mean(grad_norms)
pct_clipped = np.mean([n > 1.0 for n in grad_norms]) * 100
print(f"  Gradient monitoring over 20 steps (max_norm=1.0):")
print(f"    Avg raw gradient norm:   {avg_raw:.4f}")
print(f"    % steps where clipping activated: {pct_clipped:.1f}%")
print(f"    Min / Max gradient norm: {min(grad_norms):.4f} / {max(grad_norms):.4f}")
print()
print("  Rule: if >50% of steps are clipped, your model is exploding.")
print("  Consider: lower lr, better weight init, gradient accumulation.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: TorchScript (torch.jit.script)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — TorchScript: Python-free deployment")
print("━" * 65)
print()

# Script mode: statically analyses Python code
model_script_src = DeepMLP().eval()
try:
    scripted = torch.jit.script(model_script_src)

    with tempfile.TemporaryDirectory() as tmp:
        script_path = os.path.join(tmp, 'model_scripted.pt')
        scripted.save(script_path)
        size_kb = os.path.getsize(script_path) / 1024

        # Reload — no Python class needed!
        loaded_script = torch.jit.load(script_path)
        loaded_script.eval()

    X_check = torch.randn(4, 64)
    with torch.no_grad():
        out_orig   = model_script_src(X_check)
        out_script = scripted(X_check)
        out_loaded = loaded_script(X_check)

    print(f"  TorchScript model saved: {size_kb:.1f} KB")
    print(f"  Max diff (orig vs scripted): {(out_orig - out_script).abs().max():.2e}")
    print(f"  Max diff (orig vs loaded):   {(out_orig - out_loaded).abs().max():.2e}  ✅")
    print()

    # Speed comparison: eager vs scripted
    X_bench = torch.randn(64, 64)
    N_BENCH = 500
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(N_BENCH): model_script_src(X_bench)
        t_eager = (time.perf_counter() - t0) / N_BENCH * 1000

        t0 = time.perf_counter()
        for _ in range(N_BENCH): scripted(X_bench)
        t_script = (time.perf_counter() - t0) / N_BENCH * 1000

    print(f"  Inference benchmark ({N_BENCH} steps, batch=64):")
    print(f"    Eager:      {t_eager:.4f} ms")
    print(f"    Scripted:   {t_script:.4f} ms")
    print(f"    Ratio:      {t_eager / t_script:.2f}×")
    print()

except Exception as e:
    print(f"  TorchScript: {e}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: torch.jit.trace — and when it goes wrong
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — torch.jit.trace: limitations with dynamic control flow")
print("━" * 65)
print()

class DynamicModel(nn.Module):
    """Uses data-dependent control flow — tracing will bake in one branch."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        if x.sum() > 0:     # DATA-DEPENDENT branch — trace can't capture both!
            return self.linear(x) * 2
        else:
            return self.linear(x) * -1

dyn_model = DynamicModel().eval()
example   = torch.tensor([1.0, 2.0, 3.0, 4.0])
negative  = torch.tensor([-1.0, -2.0, -3.0, -4.0])

# Trace with positive input — bakes in the x.sum() > 0 = True branch
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    traced_dyn = torch.jit.trace(dyn_model, example)

out_eager_pos = dyn_model(example)
out_eager_neg = dyn_model(negative)
out_trace_pos = traced_dyn(example)
out_trace_neg = traced_dyn(negative)    # WRONG: always takes the True branch!

print("  Model with data-dependent branch:")
print(f"    Eager  positive: {out_eager_pos.detach().numpy().round(4)}")
print(f"    Trace  positive: {out_trace_pos.detach().numpy().round(4)}")
print(f"    Eager  negative: {out_eager_neg.detach().numpy().round(4)}")
print(f"    Trace  negative: {out_trace_neg.detach().numpy().round(4)}  ← WRONG! (baked in positive branch)")
print()
print("  Rule: Use torch.jit.script for models with control flow.")
print("  Use torch.jit.trace only for fully static computation (no if/for on data).")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Production deployment checklist
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Production deployment checklist")
print("━" * 65)

PRODUCTION_CHECKLIST = """
Production Deployment Checklist for PyTorch Models:

STEP 1: TRAINING BEST PRACTICES
    ✓  Use AdamW over Adam (decoupled weight decay)
    ✓  Apply gradient clipping (max_norm=1.0 for most models)
    ✓  Use AMP (bfloat16 on Ampere/Hopper, float16 + GradScaler on Volta/Turing)
    ✓  Save checkpoints every N epochs with optimizer state
    ✓  Use model.eval() + torch.no_grad() for all validation/inference

STEP 2: MODEL OPTIMISATION
    ✓  torch.compile(model, mode='max-autotune') — 1.5–3× speedup, no code changes
    ✓  Review model for inference-only paths (remove training-only ops)
    ✓  Consider quantisation for latency-critical edge deployment

STEP 3: SERIALISATION STRATEGY
    ┌──────────────────────────────────────────────────────────────────┐
    │ Target             │ Method              │ Notes                 │
    ├──────────────────────────────────────────────────────────────────┤
    │ Python server      │ state_dict + class  │ simplest, flexible    │
    │ C++ server         │ torch.jit.script    │ no Python required    │
    │ Mobile (Android)   │ ExecuTorch          │ Meta's edge runtime   │
    │ Cross-platform     │ ONNX export         │ use opset 17+         │
    │ NVIDIA GPU server  │ ONNX → TensorRT     │ maximum GPU speed     │
    │ Intel CPU          │ ONNX → OpenVINO     │ maximum CPU speed     │
    │ Apple Silicon      │ CoreML Tools        │ via coremltools       │
    └──────────────────────────────────────────────────────────────────┘

STEP 4: SERVING INFRASTRUCTURE
    TorchServe:  torch-model-archiver + torchserve (REST + gRPC)
    FastAPI:     load model once at startup, serve via endpoint
    Triton:      NVIDIA Triton Inference Server (multi-model, batching)
    BentoML:     framework-agnostic serving with packaging

STEP 5: MONITORING
    ✓  Log inference latency (p50, p95, p99) via Prometheus/Grafana
    ✓  Track input distribution shifts (embedding drift, feature statistics)
    ✓  A/B test new model versions before full rollout
    ✓  Set up alerts on error rate and latency degradation
"""
print(PRODUCTION_CHECKLIST)
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