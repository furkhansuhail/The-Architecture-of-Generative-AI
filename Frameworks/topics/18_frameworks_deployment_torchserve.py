"""
TorchServe — Production Serving for PyTorch Models
====================================================

TorchServe is the official, production-grade model serving framework
for PyTorch, jointly developed by AWS and Facebook (Meta) and released
in April 2020. It was built to bridge the gap that existed between
PyTorch's dominance in research and its historically weak story in
production deployment: PyTorch was the most popular research framework,
but TensorFlow Serving had a years-long head start for serving models in
production. TorchServe was the direct response — a feature-rich,
cloud-native serving solution designed specifically around PyTorch's idioms.

By 2024, TorchServe powers inference workloads across AWS SageMaker,
Azure ML, Hugging Face Inference Endpoints, and thousands of independent
deployments. It is the reference serving framework whenever PyTorch
documentation discusses productionising a model.

### The Core Problem TorchServe Solves

Before TorchServe, serving a PyTorch model in production required one of:
    1. Flask/FastAPI wrapper:     Simple but slow, no batching, no scaling
    2. ONNX + ORT:                Requires export step, loses dynamic shapes
    3. TorchScript + custom:      Complex, framework-specific engineering
    4. TensorFlow Serving:        Required re-implementing in TensorFlow

TorchServe provides what was missing: a purpose-built, batteries-included
serving system that works directly with PyTorch model files — with zero
framework switching, built-in batching, multi-worker scaling, REST and
gRPC APIs, model versioning, A/B testing, metrics, and a management API.

### The .mar File: The TorchServe Packaging Unit

At the heart of TorchServe is the Model Archive (MAR) file — a ZIP archive
containing everything needed to serve a model:
    - model.pt / model.bin:  The serialised model weights
    - handler.py:            The pre/post-processing and inference logic
    - extra files:           Tokenizer configs, class label maps, etc.
    - MAR-INF/MANIFEST.json: Metadata (name, version, handler reference)

The MAR file is the single deployable artefact. You build it once and
deploy it to any TorchServe instance, any cloud provider, any environment.
This is the key architectural innovation: separating the model artefact
from the serving infrastructure.

### Architecture Overview

    ┌─────────────────────────────────────────────────────────────────┐
    │  Clients (REST / gRPC)                                          │
    ├─────────────────────────────────────────────────────────────────┤
    │  TorchServe Frontend (Java / Netty)                             │
    │    Inference API  :8080  │  Management API :8081 │ Metrics :8082│
    ├─────────────────────────────────────────────────────────────────┤
    │  Backend Workers (Python processes)                             │
    │    Worker 1  │  Worker 2  │  Worker N                           │
    │    Each runs: Handler → Model → Pre/Post-processing             │
    ├─────────────────────────────────────────────────────────────────┤
    │  Model Store (filesystem or S3)                                 │
    │    model_a.mar  │  model_b.mar  │  model_c.mar                  │
    └─────────────────────────────────────────────────────────────────┘

Key design principles:
    - Frontend in Java (Netty): high-throughput request routing
    - Backend in Python: full PyTorch ecosystem compatibility
    - Decoupled: frontend and backend communicate via domain socket
    - Pluggable: handlers are pure Python, fully customisable
    - Observable: built-in Prometheus metrics on port 8082

### TorchServe vs Alternatives

    TorchServe:      Native PyTorch. Best for PyTorch-only, needs MAR packaging
    Triton:          Multi-framework, C++, best raw GPU performance, complex
    TF Serving:      TensorFlow only, mature, production-proven
    ONNX Runtime:    Framework-agnostic, requires ONNX export
    BentoML:         Python-native, flexible, good for complex pipelines
    FastAPI + ORT:   Maximum flexibility, most DIY work

TorchServe's sweet spot: production PyTorch deployment with minimal
infrastructure engineering. Teams that want managed batching, versioning,
A/B testing, and metrics without building it themselves.

"""

import textwrap
import re

TOPIC_NAME   = "TorchServe — Production Serving for PyTorch Models"
DISPLAY_NAME = "18 · TorchServe"
ICON         = "🔥"
SUBTITLE     = "From PyTorch Models to Scalable Production Inference Services"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — TORCHSERVE ARCHITECTURE: THE FRONTEND-BACKEND MODEL

### The Two-Tier Architecture

    TorchServe has a clean two-tier architecture that separates high-speed
    request routing from Python-based inference:

    FRONTEND (Java / Netty HTTP server):
        - Receives HTTP/gRPC requests from clients
        - Routes requests to the appropriate model's worker queue
        - Handles request queuing, batching wait, and response assembly
        - Exposes three APIs on separate ports:
            :8080  Inference API    (POST /predictions/{model_name})
            :8081  Management API   (register/unregister/scale/version models)
            :8082  Metrics API      (Prometheus endpoint)
        - Written in Java for high concurrency (Netty non-blocking I/O)
        - Communicates with workers via Unix domain sockets or TCP

    BACKEND (Python workers):
        - Each worker is a separate Python process
        - Loads and holds the model in memory (or GPU VRAM)
        - Executes the Handler — your custom pre/post/inference code
        - Reports metrics (latency, batch size, queue depth) back to frontend
        - Each model can have a configurable number of workers
        - Workers can be on CPU, GPU, or multiple GPUs

    The design choice to use Java for the frontend is deliberate:
        Python's GIL prevents true multi-thread concurrency.
        Java handles thousands of concurrent HTTP connections natively.
        Python workers are separate processes — no GIL contention.
        This gives TorchServe the best of both worlds: Java's concurrency
        for I/O and Python's ecosystem for ML.

### Request Flow: From HTTP to Prediction

    A complete request lifecycle:
        1. Client sends POST /predictions/resnet50
        2. Netty frontend receives the HTTP request
        3. Frontend serialises the request into a binary message
        4. Frontend places message onto the model's worker queue
        5. An available Python worker picks up the message
        6. Worker calls handler.preprocess() → tensor
        7. Worker calls handler.inference(tensor) → model output
        8. Worker calls handler.postprocess(output) → response dict
        9. Worker sends serialised response back via domain socket
        10. Frontend assembles the HTTP response and sends to client

    Batched request flow:
        Multiple requests (steps 1-4) accumulate in the queue.
        After max_batch_delay milliseconds or after max_batch_size requests
        have queued, the worker processes all of them as a single batch:
            Step 6: preprocess() called for each sample (or vectorised)
            Step 7: inference(batch_tensor) — single forward pass for all
            Step 8: postprocess() called per-sample or for the batch
        This is critical for GPU efficiency: one CUDA kernel launch for
        N requests instead of N separate kernel launches.

### The Worker Pool

    TorchServe manages a pool of workers per model. Key configuration:
        min_workers:      minimum workers always running (default: 1)
        max_workers:      maximum workers at peak load (default: 1)
        num_gpu:          GPUs per worker (default: 0 = CPU)
        max_batch_size:   accumulate up to N requests per batch
        batch_timeout_ms: max wait time to fill a batch (default: 100ms)

    Worker autoscaling:
        TorchServe monitors the queue depth. When requests pile up, it
        starts new workers (up to max_workers). When the queue is empty,
        it scales back down (below a configurable idle timeout).
        This is managed entirely by the TorchServe Management API.

    GPU assignment:
        Workers are assigned GPUs in round-robin order.
        If you have 4 GPUs and max_workers=8, workers 0,4 share GPU 0;
        workers 1,5 share GPU 1; etc.
        For exclusive GPU ownership: max_workers = num_gpus.


##### PART 2 — HANDLERS: THE CORE OF TORCHSERVE

### What Is a Handler?

    A Handler is a Python class that defines how TorchServe should process
    requests for a specific model. It is the central customisation point —
    everything that happens between receiving a raw HTTP request and sending
    back a response is implemented in the handler.

    Every handler must implement (or inherit) four methods:
        initialize(context):          load the model and any resources
        preprocess(data):             raw bytes → tensor(s)
        inference(data):              tensor(s) → model output(s)
        handle(data, context):        the entry point called by TorchServe

    The handler receives:
        data:    list of dicts [{"body": <bytes>}, {"body": <bytes>}, ...]
                 One element per request in the current batch.
        context: the ModelContext object with:
                 manifest:      the model's MANIFEST.json
                 system_properties: worker_id, gpu_id, model_dir
                 metrics:       the MetricsApiUtil for custom metrics

### BaseHandler: The Standard Starting Point

    TorchServe ships with `BaseHandler` — a default implementation that
    covers common patterns:

        from ts.torch_handler.base_handler import BaseHandler

        class MyHandler(BaseHandler):
            def initialize(self, context):
                # Called once when the worker starts
                super().initialize(context)
                # super() loads model from context.manifest["model"]["serializedFile"]
                # self.model is now the loaded nn.Module
                # self.device is "cuda:0" or "cpu"
                self.tokenizer = load_tokenizer(context.model_dir)

            def preprocess(self, data):
                # data: list of {"body": bytes} dicts (one per request in batch)
                texts = [item["body"].decode("utf-8") for item in data]
                tokens = self.tokenizer(texts, return_tensors="pt",
                                         truncation=True, max_length=512)
                return tokens.input_ids.to(self.device)

            def postprocess(self, output):
                # output: the raw model output tensor
                # Must return a list (one element per input request)
                probs  = output.softmax(-1).cpu().tolist()
                labels = ["negative", "positive"]
                return [{"label": labels[p.index(max(p))],
                          "confidence": max(p)} for p in probs]

    BaseHandler provides:
        self.model:         the loaded nn.Module (from initialize)
        self.device:        "cuda:0" or "cpu" (based on num_gpu config)
        self.manifest:      parsed MANIFEST.json
        self.map_location:  torch.device for loading weights
        self.context:       the ModelContext

### Built-in Handlers

    TorchServe ships with pre-built handlers for common tasks:
        image_classifier:   loads image, runs through model, returns top-k classes
        image_segmenter:    returns segmentation masks
        object_detector:    returns bounding boxes and class labels
        text_classifier:    text → class label
        base_handler:       minimal base for custom implementations

    Using a built-in handler:
        torch-model-archiver --handler image_classifier ...
        # No handler.py needed — built-in handles everything

    When to write a custom handler:
        - Custom preprocessing (tokenisation, normalisation, augmentation)
        - Multi-modal models (image + text)
        - Models with custom output formats
        - Ensemble models
        - Models needing external resources (vocab files, class maps)

### The Handle Method: Batch Orchestration

    The handle() method is what TorchServe actually calls. In BaseHandler:

        def handle(self, data, context):
            # data: list of {"body": bytes} — one per request in batch
            # context: ModelContext

            # Step 1: Preprocess all inputs
            model_input = self.preprocess(data)

            # Step 2: Run inference on the batch
            with torch.no_grad():
                model_output = self.inference(model_input)

            # Step 3: Postprocess all outputs
            return self.postprocess(model_output)

    Custom handle() for complex logic (streaming, multi-pass):
        def handle(self, data, context):
            # Example: two-stage pipeline
            text_input  = self.preprocess_text(data)
            embeddings  = self.model_a(text_input)           # stage 1
            predictions = self.model_b(embeddings)           # stage 2
            return self.postprocess(predictions)


##### PART 3 — MODEL PACKAGING: THE .MAR FORMAT

### The torch-model-archiver Tool

    torch-model-archiver packages a model and its handler into a .mar file:

        torch-model-archiver \\
            --model-name        resnet50 \\
            --version           1.0 \\
            --model-file        models/resnet50.py \\
            --serialized-file   weights/resnet50.pt \\
            --handler           handlers/image_handler.py \\
            --extra-files       config/imagenet_classes.txt \\
            --export-path       model_store/ \\
            --requirements-file requirements.txt

    Parameters explained:
        --model-name:         Name used in API endpoints (case-sensitive)
        --version:            Semantic version (1.0, 2.1, etc.)
        --model-file:         Python file defining the nn.Module class
        --serialized-file:    The .pt file (state dict or full model)
        --handler:            Handler Python file (or built-in name)
        --extra-files:        Additional files handler needs at runtime
        --export-path:        Directory where the .mar will be saved
        --requirements-file:  Python dependencies to install in the worker

### MAR File Structure

    A .mar file is a ZIP archive with this structure:

        resnet50.mar (ZIP)
        ├── MAR-INF/
        │   └── MANIFEST.json      ← model metadata and file inventory
        ├── resnet50.pt            ← model weights (from --serialized-file)
        ├── resnet50.py            ← model class definition (from --model-file)
        ├── image_handler.py       ← handler (from --handler)
        ├── imagenet_classes.txt   ← extra files (from --extra-files)
        └── requirements.txt       ← dependencies

    MANIFEST.json (example):
        {
            "createdOn": "03/01/2024 10:30:45",
            "runtime": "python",
            "model": {
                "modelName": "resnet50",
                "serializedFile": "resnet50.pt",
                "handler": "image_handler",
                "modelVersion": "1.0"
            },
            "archiverVersion": "0.8.0"
        }

### Serialisation Formats

    TorchServe supports multiple ways to save the model:

    TorchScript (recommended for production):
        scripted = torch.jit.script(model)
        torch.jit.save(scripted, "model.pt")
        # No model-file needed — the class is embedded in the .pt file
        # Portable, serialisation-safe, no class definition needed at runtime

    State dict (most flexible):
        torch.save(model.state_dict(), "model_state.pt")
        # Requires --model-file with the nn.Module class definition
        # Handler's initialize() does: model.load_state_dict(torch.load(...))

    Full model (convenient, fragile):
        torch.save(model, "model_full.pt")
        # Requires model class to be importable at runtime
        # Breaks if class definition changes — avoid for production

    Eager mode with state dict is the most common production pattern:
        Allows model code updates without repackaging the weights.

### Model Store

    TorchServe watches a directory (--model-store) for .mar files.
    Models must be registered (via Management API or config.properties)
    before they can serve requests.

    File-system model store (development):
        torchserve --start --model-store ./model_store

    S3 model store (production):
        TorchServe can download MAR files from S3 URLs:
        POST /models?url=s3://my-bucket/models/resnet50-v2.mar


##### PART 4 — THE THREE APIs: INFERENCE, MANAGEMENT, AND METRICS

### Inference API (port 8080)

    The primary client-facing API for running predictions:

    Single model inference:
        POST /predictions/{model_name}
        Body: raw bytes (image), JSON, or form-data
        Headers: Content-Type appropriate for the input

    Versioned inference:
        POST /predictions/{model_name}/{version}
        # Always routes to the specified version regardless of default

    Explanation (Captum integration):
        POST /explanations/{model_name}
        # Runs integrated gradients or other attribution methods

    Batch inference (explicit HTTP batching):
        POST /predictions/{model_name}  (multiple requests in flight simultaneously)
        # TorchServe's adaptive batching collects concurrent requests

    Example request:
        curl -X POST http://localhost:8080/predictions/resnet50 \\
            -T ./cat.jpg \\
            -H "Content-Type: image/jpeg"

        Response: {"tabby": 0.42, "tiger_cat": 0.31, ...}

### Management API (port 8081)

    The operator API for managing the lifecycle of models:

    Register a model (load it into TorchServe):
        POST /models?url=resnet50.mar&initial_workers=2&batch_size=16

    List registered models:
        GET  /models
        GET  /models/{model_name}   → version list and worker stats

    Scale workers for a model:
        PUT  /models/{model_name}?min_worker=2&max_worker=8

    Set default version:
        PUT  /models/{model_name}/{version}/set-default

    Unregister a model:
        DELETE /models/{model_name}/{version}

    Worker health:
        GET /models/{model_name} returns per-worker status:
        {"workers": [{"id": "9000", "state": "READY", "gpu": "0"}, ...]}

    Blue/green deployment pattern:
        1. Register new version: POST /models?url=resnet50-v2.mar
        2. Scale up new:         PUT  /models/resnet50/2.0?min_worker=4
        3. Set as default:       PUT  /models/resnet50/2.0/set-default
        4. Scale down old:       PUT  /models/resnet50/1.0?min_worker=0
        5. Unregister old:       DELETE /models/resnet50/1.0

### Metrics API (port 8082)

    Prometheus-compatible metrics endpoint:
        GET /metrics

    Built-in metrics exposed by TorchServe:
        ts_inference_requests_total{model_name, status}:
            Counter: total inference requests (success/error)
        ts_inference_latency_microseconds{model_name, level}:
            Histogram: end-to-end inference latency
        ts_queue_latency_microseconds{model_name}:
            Histogram: time spent waiting in the request queue
        ts_worker_memory_used_megabytes:
            Gauge: RAM used by each worker process
        ts_gpu_utilization_percentage{gpu_id}:
            Gauge: GPU compute utilisation (via NVML)
        ts_gpu_memory_used_megabytes{gpu_id}:
            Gauge: GPU VRAM consumption

    Scraping with Prometheus + Grafana:
        Add to prometheus.yml:
            - job_name: 'torchserve'
              static_configs: [{targets: ['localhost:8082']}]
        Import TorchServe Grafana dashboard (ID: 17829)

    Custom metrics (from handler code):
        from ts.metrics.metrics_store import MetricsStore

        def inference(self, data):
            output = self.model(data)
            # Emit a custom metric
            self.context.metrics.add_metric(
                name="custom_prediction_score",
                value=float(output.max()),
                unit="Dimensionless",
                metric_type="gauge",
            )
            return output


##### PART 5 — BATCHING: THE KEY TO GPU EFFICIENCY

### Why Batching Matters for GPU Throughput

    A GPU processes matrix operations most efficiently when the matrices
    are large. The overhead of launching a CUDA kernel (scheduling,
    memory transfer, dispatch) is the same whether you multiply a
    [1, 768] vector or a [64, 768] matrix — but the [64, 768] result
    is 64× more useful work per kernel launch.

    Without batching:
        64 requests × (5ms kernel overhead + 1ms compute) = 384ms total
        Throughput: ~167 req/s, GPU utilisation: ~17%

    With batch_size=64 (all arrive within batch_timeout):
        1 batch × (5ms kernel overhead + 64ms compute) = 69ms total
        Throughput: ~928 req/s, GPU utilisation: ~93%

    The batch_timeout creates a tradeoff:
        Long timeout:   more batching, higher throughput, higher latency
        Short timeout:  less batching, lower throughput, lower latency
        Typical:        50-200ms for most workloads

### Configuring Batching in TorchServe

    Global defaults in config.properties:
        batch_size=32           # max requests per batch
        max_batch_delay=100     # milliseconds to wait for batch to fill
        default_workers_per_model=1

    Per-model at registration:
        POST /models?url=model.mar&batch_size=64&max_batch_delay=200ms

    Per-model in config.properties:
        [resnet50]
        batch_size=64
        max_batch_delay=100
        initial_workers=2
        max_workers=8

### Handler Implementation for Batching

    When batching is enabled, the handler receives a list of inputs:

        def preprocess(self, data):
            # data has len == current batch size (1 to max_batch_size)
            images = []
            for item in data:
                img = decode_image(item["body"])
                img = self.transforms(img)
                images.append(img)
            return torch.stack(images).to(self.device)  # [B, C, H, W]

        def inference(self, batch_tensor):
            with torch.no_grad():
                return self.model(batch_tensor)          # [B, n_classes]

        def postprocess(self, output):
            # Must return a list of length == batch size
            probs = output.softmax(-1).cpu().tolist()
            return [{"top_class": int(np.argmax(p)),
                     "confidence": max(p)}
                    for p in probs]                      # len == batch_size

    CRITICAL: postprocess MUST return a list of the same length as the
    input batch. TorchServe uses this to route each result back to the
    correct waiting client.

### Adaptive Batching vs Fixed Batching

    Fixed batching (TorchServe default):
        Process batch when EITHER batch_size requests OR max_batch_delay
        milliseconds have elapsed (whichever comes first).

    Continuous Batching (for LLMs, TorchServe 0.9+):
        New requests join the batch in-flight.
        Completed requests are removed from the batch mid-generation.
        More efficient for autoregressive generation where different
        sequences complete at different times.
        Enabled via: continuous_batching=true in config


##### PART 6 — TORCHSCRIPT AND OPTIMISATION

### TorchScript: The Recommended Serialisation

    TorchScript compiles a PyTorch model to an Intermediate Representation
    (IR) that can be:
        - Saved and loaded without the original Python class definition
        - Executed in C++ without a Python interpreter
        - Optimised by the TorchScript compiler (constant folding, etc.)
        - Deployed to mobile (Android/iOS via torch.jit)

    Two ways to create TorchScript:

    TRACING (simpler, less flexible):
        scripted = torch.jit.trace(model, example_input)
        # Runs model once with example_input, records operations
        # Problem: control flow (if/for) is frozen at trace time
        # Use for: CNNs, standard transformers, models without dynamic shapes

    SCRIPTING (more robust):
        scripted = torch.jit.script(model)
        # Compiles the model by analysing the Python AST
        # Handles control flow, loops, conditionals
        # Requires type annotations for non-trivial code
        # Use for: models with dynamic shapes, custom RNNs, ensemble models

    Combining (for complex models):
        @torch.jit.script
        def forward_with_logic(x: torch.Tensor, threshold: float) -> torch.Tensor:
            out = model(x)
            if out.max() > threshold:
                return out.softmax(-1)
            return torch.zeros_like(out)

    Saving TorchScript for TorchServe:
        scripted = torch.jit.script(model)
        scripted.save("model_scripted.pt")
        # --model-file is NOT needed with TorchScript
        # The class definition is embedded in the .pt file

### Torch Compile (PyTorch 2.x): Faster Inference

    torch.compile() applies further optimisations via Inductor backend:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        # mode options:
        #   "default":         balanced speed/compile time
        #   "reduce-overhead": aggressive optimisation (longer compile)
        #   "max-autotune":    maximum speed via kernel auto-tuning

    In TorchServe handler:
        def initialize(self, context):
            super().initialize(context)
            self.model = torch.compile(self.model, mode="reduce-overhead")
            # First inference: slow (compilation)
            # Subsequent: 20-50% faster than eager on most models

### quantisation in TorchServe

    Dynamic quantisation (handler-level):
        def initialize(self, context):
            super().initialize(context)
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            # No calibration data needed
            # Weights quantised to INT8, activations dynamically

    Static quantisation (calibrated):
        Requires calibration data. More accurate than dynamic.
        Run calibration → generate quantisation parameters → save model.
        Load quantised model in handler exactly like any other model.

    FP16 on GPU:
        def inference(self, data):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model(data)
            # ~2× throughput on Tensor Core GPUs, minimal accuracy drop


##### PART 7 — MULTI-MODEL SERVING AND A/B TESTING

### Serving Multiple Models Simultaneously

    TorchServe can serve many models at once from the same process:

        torchserve --start \\
            --model-store model_store/ \\
            --models resnet50=resnet50.mar,bert=bert.mar,yolo=yolo.mar

    Or via config.properties:
        model_store=./model_store
        load_models=resnet50.mar,bert-base.mar,yolo-v8.mar

    Each model gets its own:
        - Worker pool (independent scaling)
        - Request queue (no cross-model interference)
        - Metrics (per-model latency and throughput)
        - API endpoint (/predictions/resnet50, /predictions/bert-base)

    Resource sharing:
        Workers share GPUs (by GPU ID assignment).
        Multiple models can share the same GPU if they fit in VRAM.
        Use batch_size and worker counts to control VRAM allocation.

### Model Versioning and A/B Testing

    TorchServe supports multiple versions of the same model simultaneously:

        # Register version 1.0
        POST /models?url=resnet50-v1.mar
        # Registers as resnet50/1.0

        # Register version 2.0
        POST /models?url=resnet50-v2.mar
        # Registers as resnet50/2.0

        # Access specific version:
        POST /predictions/resnet50/1.0   # always v1
        POST /predictions/resnet50/2.0   # always v2
        POST /predictions/resnet50       # routes to default version

        # Set default (where non-versioned requests go):
        PUT /models/resnet50/2.0/set-default

    Traffic splitting (A/B testing via load balancer):
        TorchServe itself does not do percentage-based traffic splitting.
        Use a load balancer (NGINX, Istio, AWS ALB) in front:

        upstream torchserve_ab {
            server ts-v1:8080 weight=80;  # 80% traffic to v1
            server ts-v2:8080 weight=20;  # 20% traffic to v2
        }

    Or use separate endpoints in your application:
        def route_request(request, traffic_split=0.2):
            if random.random() < traffic_split:
                return call_model("resnet50/2.0", request)
            return call_model("resnet50/1.0", request)

### Workflow Handlers (Sequential Models)

    For complex pipelines that chain multiple models:

        class PipelineHandler(BaseHandler):
            def initialize(self, context):
                # Load two models
                self.detector  = load_model("detector.pt")
                self.classifier = load_model("classifier.pt")

            def handle(self, data, context):
                # Stage 1: detect objects
                images   = self.preprocess(data)
                boxes    = self.detector(images)

                # Stage 2: classify each detected region
                crops    = extract_crops(images, boxes)
                classes  = self.classifier(crops)

                return self.postprocess(boxes, classes)


##### PART 8 — CONFIGURATION, DEPLOYMENT, AND OBSERVABILITY

### config.properties: The Central Configuration

    TorchServe's main configuration file:

        # Network
        inference_address=http://0.0.0.0:8080
        management_address=http://0.0.0.0:8081
        metrics_address=http://0.0.0.0:8082

        # Model store
        model_store=/home/model-server/model-store

        # Initial models to load on startup
        load_models=resnet50.mar,bert-base.mar

        # Default batch settings
        batch_size=32
        max_batch_delay=100

        # Worker settings
        default_workers_per_model=1

        # Timeouts (milliseconds)
        default_response_timeout=120000
        unregister_model_timeout=120

        # GPU
        number_of_netty_threads=32
        job_queue_size=1000

        # Security
        enable_cors=true
        allowed_urls=https://my-app.company.com

    Starting with config:
        torchserve --start --model-store ./model_store \\
            --ts-config config.properties

### Docker Deployment (Production Standard)

    Official TorchServe Docker images from AWS:
        pytorch/torchserve:latest            # CPU
        pytorch/torchserve:latest-gpu        # NVIDIA GPU (CUDA 12.x)
        pytorch/torchserve:0.10.0-gpu        # pinned version (use in prod!)

    Minimal Dockerfile:
        FROM pytorch/torchserve:latest-gpu

        # Copy model archive and config
        COPY model_store/ /home/model-server/model-store/
        COPY config.properties /home/model-server/config.properties

        # Expose ports
        EXPOSE 8080 8081 8082

        # Start TorchServe
        CMD ["torchserve", "--start", "--ncs", \\
             "--model-store", "/home/model-server/model-store", \\
             "--ts-config", "/home/model-server/config.properties"]

    Run with GPU:
        docker run --gpus all \\
            -p 8080:8080 -p 8081:8081 -p 8082:8082 \\
            -v $(pwd)/model_store:/home/model-server/model-store \\
            pytorch/torchserve:latest-gpu

### Kubernetes Deployment

    TorchServe on Kubernetes with GPU:
        # torchserve-deployment.yaml
        apiVersion: apps/v1
        kind: Deployment
        spec:
          replicas: 3
          template:
            spec:
              containers:
              - name: torchserve
                image: pytorch/torchserve:0.10.0-gpu
                ports: [{containerPort: 8080}, {containerPort: 8081}]
                resources:
                  limits:
                    nvidia.com/gpu: "1"
                    memory: "8Gi"
                livenessProbe:
                  httpGet: {path: "/ping", port: 8080}
                  initialDelaySeconds: 30
                readinessProbe:
                  httpGet: {path: "/models", port: 8081}
                  initialDelaySeconds: 60

    Horizontal Pod Autoscaler (scale on GPU utilisation):
        kubectl autoscale deployment torchserve \\
            --min=2 --max=10 \\
            --cpu-percent=70

### Observability Best Practices

    Latency SLOs:
        p50 < 50ms (median request should be fast)
        p95 < 200ms (95% of requests)
        p99 < 500ms (99% of requests — catch slow batches)

    Key metrics to monitor:
        ts_inference_latency_microseconds (p50/p95/p99) — end-to-end latency
        ts_queue_latency_microseconds — time waiting in queue (→ scale up if high)
        ts_inference_requests_total{status="error"} — error rate
        ts_gpu_utilization_percentage — GPU efficiency
        Worker count vs queue depth — autoscaling trigger

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Tuning Lever          │ When to adjust                               │
    ├──────────────────────────────────────────────────────────────────────┤
    │ max_batch_size ↑      │ Queue depth high, GPU util low               │
    │ max_batch_delay ↑     │ Low traffic, tolerate latency for efficiency │
    │ max_workers ↑         │ CPU bound, queue depth consistently high     │
    │ num_gpu ↑             │ GPU OOM or GPU utilisation < 60%             │
    │ FP16 inference        │ Latency target not met on GPU                │
    │ torch.compile()       │ High throughput needed, first-call warm-up ok│
    └──────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Handlers — Writing, Testing, and Validating Custom Handlers": {
        "description": (
            "Complete TorchServe handler development without a running server. "
            "BaseHandler interface: initialize, preprocess, inference, postprocess. "
            "Custom handler for image classification: decode, transform, predict. "
            "Custom handler for NLP text classification. "
            "Batch-aware preprocess/postprocess: list in, list out. "
            "ModelContext simulation: manifest, system_properties, model_dir. "
            "Handler unit testing: mock context, raw bytes input. "
            "Error handling in handlers: try/except, error responses. "
            "Extra files access: loading class maps and vocab files. "
            "Multi-model handler: loading two models in one handler. "
            "Custom metrics emission from handler code. "
            "Handler performance profiling: preprocessing bottlenecks."
        ),
        "language": "python",
        "code": '''
import os
import io
import json
import time
import math
import struct
import hashlib
import tempfile
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"  PyTorch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "torch", "--quiet"], check=True)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"  PyTorch: {torch.__version__}")

print("=" * 65)
print("  HANDLERS — WRITING AND TESTING CUSTOM TORCHSERVE HANDLERS")
print("=" * 65)
print()

rng = np.random.default_rng(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: ModelContext simulation (mimics TorchServe runtime)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — ModelContext: the runtime environment for handlers")
print("━" * 65)
print()

@dataclass
class MockMetrics:
    """Simulates ts.metrics.metrics_store.MetricsStore"""
    _log: List[Dict] = field(default_factory=list)

    def add_metric(self, name: str, value: float, unit: str = "ms",
                   metric_type: str = "gauge"):
        self._log.append({"name": name, "value": value,
                           "unit": unit, "type": metric_type})

    def add_counter(self, name: str, value: int = 1):
        self._log.append({"name": name, "value": value, "type": "counter"})


@dataclass
class MockContext:
    """
    Simulates ts.context.Context.
    In real TorchServe, Context is created by the backend worker.
    """
    model_name:        str
    model_dir:         str
    gpu_id:            Optional[int] = None
    batch_size:        int = 1
    manifest:          Dict = field(default_factory=dict)
    system_properties: Dict = field(default_factory=dict)
    metrics:           MockMetrics = field(default_factory=MockMetrics)

    def __post_init__(self):
        if not self.manifest:
            self.manifest = {
                "model": {
                    "modelName":      self.model_name,
                    "serializedFile": f"{self.model_name}.pt",
                    "handler":        "handler",
                    "modelVersion":   "1.0",
                },
                "runtime": "python",
            }
        if not self.system_properties:
            self.system_properties = {
                "model_dir":  self.model_dir,
                "gpu_id":     self.gpu_id,
                "batch_size": self.batch_size,
            }

    def get_manifest(self): return self.manifest


def make_context(model_name: str, model_dir: str,
                 gpu_id: int = None) -> MockContext:
    return MockContext(
        model_name = model_name,
        model_dir  = model_dir,
        gpu_id     = gpu_id,
    )

print(f"  ModelContext fields that handlers rely on:")
ctx = make_context("resnet50", "/tmp/models/resnet50")
print(f"    context.model_name:        {ctx.model_name}")
print(f"    context.manifest:          {ctx.manifest['model']}")
print(f"    context.system_properties: {ctx.system_properties}")
print(f"    context.metrics:           <MetricsStore> (emit custom metrics)")
print(f"    context.gpu_id:            {ctx.gpu_id}  (None = CPU)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: BaseHandler pattern — the standard four methods
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — BaseHandler: the four-method contract")
print("━" * 65)
print()

class BaseHandler:
    """
    Faithful reimplementation of TorchServe's ts.torch_handler.base_handler.
    Real TorchServe imports from ts.torch_handler.base_handler.BaseHandler.
    We reimplement here so the demo runs without TorchServe installed.
    """
    def __init__(self):
        self.model    = None
        self.device   = None
        self.context  = None
        self.manifest = None
        self._initialized = False

    def initialize(self, context: MockContext):
        """
        Called once when the worker starts.
        Load model and any auxiliary resources.
        """
        self.context  = context
        self.manifest = context.get_manifest()
        self.device   = torch.device(
            f"cuda:{context.gpu_id}"
            if context.gpu_id is not None and torch.cuda.is_available()
            else "cpu"
        )
        # Load model — in real TorchServe, this loads from the .pt file
        # Here we just create the model
        self.model = self._load_model(context)
        self.model.eval()
        self._initialized = True

    def _load_model(self, context: MockContext) -> nn.Module:
        """Override to load the actual model. Default raises."""
        raise NotImplementedError("Subclass must implement _load_model")

    def preprocess(self, data: List[Dict]) -> Any:
        """Convert raw request bytes to model input tensor(s)."""
        raise NotImplementedError

    def inference(self, data: Any) -> Any:
        """Run model forward pass."""
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, inference_output: Any) -> List[Any]:
        """Convert model output to JSON-serialisable list."""
        raise NotImplementedError

    def handle(self, data: List[Dict], context: MockContext) -> List[Any]:
        """Entry point called by TorchServe for each batch."""
        if not self._initialized:
            self.initialize(context)

        t0 = time.perf_counter()
        model_input  = self.preprocess(data)
        t_pre = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        model_output = self.inference(model_input)
        t_inf = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        response = self.postprocess(model_output)
        t_post = (time.perf_counter() - t0) * 1000

        # Emit metrics
        context.metrics.add_metric("preprocess_latency",  t_pre,  "ms")
        context.metrics.add_metric("inference_latency",   t_inf,  "ms")
        context.metrics.add_metric("postprocess_latency", t_post, "ms")
        context.metrics.add_counter("inference_requests", len(data))

        assert len(response) == len(data), (
            f"postprocess must return list of same length as input batch. "
            f"Got {len(response)}, expected {len(data)}"
        )
        return response


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Image classification handler
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Image classification handler: decode → tensor → predict")
print("━" * 65)
print()

class TinyImageCNN(nn.Module):
    """Small CNN that fits a 16×16 image → 10 classes (for fast demo)."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))


class ImageClassifierHandler(BaseHandler):
    """
    Handler for image classification.
    Input:  raw image bytes (JPEG/PNG or raw float32 pixels)
    Output: top-3 class labels with probabilities
    """
    IMAGE_SIZE = 16   # 16×16 for demo (real: 224×224)
    N_CLASSES  = 10
    IMAGENET_CLASSES = [f"class_{i:02d}" for i in range(10)]

    def _load_model(self, context: MockContext) -> nn.Module:
        model = TinyImageCNN(self.N_CLASSES).to(self.device)
        # In production: model.load_state_dict(torch.load(weights_path))
        return model

    def _decode_image(self, raw_bytes: bytes) -> torch.Tensor:
        """
        Decode image bytes to a [3, H, W] float tensor.
        Real handler would use PIL.Image.open(io.BytesIO(raw_bytes)).
        We simulate with a deterministic hash of the bytes.
        """
        h = int(hashlib.md5(raw_bytes).hexdigest(), 16)
        rng_local = np.random.default_rng(h % (2**32))
        pixels = rng_local.uniform(0, 1, (3, self.IMAGE_SIZE, self.IMAGE_SIZE))
        return torch.tensor(pixels, dtype=torch.float32)

    def preprocess(self, data: List[Dict]) -> torch.Tensor:
        """data: [{"body": <bytes>}, ...]  →  [B, 3, H, W] tensor"""
        tensors = []
        for item in data:
            raw  = item.get("body", item.get("data", b""))
            img  = self._decode_image(raw)
            # Normalise (ImageNet mean/std)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img  = (img - mean) / std
            tensors.append(img)
        return torch.stack(tensors).to(self.device)  # [B, 3, H, W]

    def postprocess(self, output: torch.Tensor) -> List[Dict]:
        """[B, n_classes] logits → list of top-3 predictions per image"""
        probs   = output.softmax(-1).cpu().tolist()
        results = []
        for prob_row in probs:
            top3_idx = sorted(range(len(prob_row)), key=lambda i: -prob_row[i])[:3]
            results.append({
                "predictions": [
                    {"label": self.IMAGENET_CLASSES[i],
                     "probability": round(prob_row[i], 4)}
                    for i in top3_idx
                ]
            })
        return results


# Test the image classification handler
with tempfile.TemporaryDirectory() as model_dir:
    ctx = make_context("resnet50", model_dir)
    handler = ImageClassifierHandler()
    handler.initialize(ctx)

    # Simulate a batch of 4 image requests
    fake_images = [
        {"body": bytes(rng.integers(0, 256, 100, dtype=np.uint8).tolist())}
        for _ in range(4)
    ]

    t0 = time.perf_counter()
    results = handler.handle(fake_images, ctx)
    t_total = (time.perf_counter() - t0) * 1000

    print(f"  ImageClassifierHandler: batch of {len(fake_images)} images")
    print(f"  Total handle() time: {t_total:.1f}ms")
    print()
    for i, result in enumerate(results):
        top_pred = result["predictions"][0]
        print(f"  Image {i}: top={top_pred['label']}  "
              f"prob={top_pred['probability']:.4f}")
    print()

    # Check batch size invariant
    assert len(results) == len(fake_images), "Batch size invariant violated!"
    print(f"  Batch size invariant: len(output)==len(input) ✅")
    print()

    # Check metrics were emitted
    emitted = ctx.metrics._log
    print(f"  Emitted {len(emitted)} metrics:")
    for m in emitted:
        print(f"    {m['name']:<28}: {m['value']:.3f} {m.get('unit','')}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: NLP text classification handler
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — NLP handler: tokenise → classify → label")
print("━" * 65)
print()

class TinyTextClassifier(nn.Module):
    """Embedding bag → linear head for fast text classification."""
    def __init__(self, vocab_size=1000, embed_dim=32, n_classes=3):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc        = nn.Linear(embed_dim, n_classes)
    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))


class TextClassifierHandler(BaseHandler):
    """
    Handler for text sentiment classification.
    Input:  UTF-8 text bytes
    Output: sentiment label + confidence
    """
    VOCAB_SIZE  = 1000
    EMBED_DIM   = 32
    N_CLASSES   = 3
    LABELS      = ["negative", "neutral", "positive"]
    # Simple character-level "tokeniser" for demo
    # Real handler: load tokenizer from extra files

    def _load_model(self, context: MockContext) -> nn.Module:
        model = TinyTextClassifier(
            self.VOCAB_SIZE, self.EMBED_DIM, self.N_CLASSES
        ).to(self.device)
        return model

    def _tokenize(self, text: str) -> List[int]:
        """Simple char-level tokeniser → token IDs (demo only)."""
        return [ord(c) % self.VOCAB_SIZE for c in text[:128]]

    def preprocess(self, data: List[Dict]):
        """Tokenise all texts and pack into (text_tensor, offsets)."""
        all_tokens = []
        offsets    = [0]
        for item in data:
            raw    = item.get("body", b"")
            text   = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
            offsets.append(offsets[-1] + len(tokens))

        # EmbeddingBag expects: token_ids, offsets (start of each sequence)
        text_t    = torch.tensor(all_tokens, dtype=torch.long).to(self.device)
        offset_t  = torch.tensor(offsets[:-1], dtype=torch.long).to(self.device)
        return text_t, offset_t

    def inference(self, data):
        text_t, offset_t = data
        with torch.no_grad():
            return self.model(text_t, offset_t)

    def postprocess(self, output: torch.Tensor) -> List[Dict]:
        """[B, n_classes] logits → list of {label, confidence, scores}"""
        probs = output.softmax(-1).cpu().tolist()
        return [
            {
                "label":      self.LABELS[int(np.argmax(p))],
                "confidence": round(max(p), 4),
                "scores":     {self.LABELS[i]: round(p[i], 4)
                               for i in range(len(p))},
            }
            for p in probs
        ]


with tempfile.TemporaryDirectory() as model_dir:
    ctx      = make_context("sentiment", model_dir)
    handler  = TextClassifierHandler()
    handler.initialize(ctx)

    texts = [
        "This product is absolutely amazing! Best purchase ever.",
        "Terrible experience, completely broken on arrival.",
        "It's okay, does what it says but nothing special.",
        "Five stars! Would highly recommend to everyone.",
        "Waste of money, very disappointed.",
    ]
    batch = [{"body": t.encode("utf-8")} for t in texts]

    results = handler.handle(batch, ctx)

    print(f"  TextClassifierHandler: batch of {len(texts)} texts")
    print()
    print(f"  {'Text (truncated)':<45} {'Label':<12} {'Conf':>8}")
    print(f"  {'─'*68}")
    for text, result in zip(texts, results):
        print(f"  {text[:43]:<45} {result['label']:<12} "
              f"{result['confidence']:>8.4f}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Error handling in handlers
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Error handling: robust handler patterns")
print("━" * 65)
print()

class RobustImageHandler(ImageClassifierHandler):
    """
    Demonstrates production-grade error handling in handlers.
    TorchServe will return HTTP 500 if handle() raises.
    Better: catch expected errors and return structured error responses.
    """
    def preprocess(self, data: List[Dict]) -> torch.Tensor:
        tensors = []
        for i, item in enumerate(data):
            try:
                raw = item.get("body", b"")
                if not raw:
                    raise ValueError(f"Empty body in request {i}")
                img = self._decode_image(raw)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                tensors.append((img - mean) / std)
            except Exception as e:
                # Log the error and substitute a zero tensor
                self.context.metrics.add_counter("preprocess_errors")
                tensors.append(torch.zeros(3, self.IMAGE_SIZE, self.IMAGE_SIZE))
        return torch.stack(tensors).to(self.device)

    def postprocess(self, output: torch.Tensor) -> List[Dict]:
        try:
            probs = output.softmax(-1).cpu().tolist()
            results = []
            for prob_row in probs:
                top3 = sorted(range(len(prob_row)), key=lambda i: -prob_row[i])[:3]
                results.append({
                    "status": "ok",
                    "predictions": [
                        {"label": self.IMAGENET_CLASSES[i],
                         "probability": round(prob_row[i], 4)}
                        for i in top3
                    ],
                })
            return results
        except Exception as e:
            # Return error for every item in the batch
            return [{"status": "error", "message": str(e)}] * output.shape[0]


with tempfile.TemporaryDirectory() as model_dir:
    ctx     = make_context("resnet50", model_dir)
    handler = RobustImageHandler()
    handler.initialize(ctx)

    # Mix of valid and empty requests
    mixed_batch = [
        {"body": bytes(rng.integers(0, 256, 100, dtype=np.uint8).tolist())},
        {"body": b""},  # empty body
        {"body": bytes(rng.integers(0, 256, 100, dtype=np.uint8).tolist())},
    ]
    results = handler.handle(mixed_batch, ctx)

    print(f"  RobustImageHandler: 3 requests (one empty body)")
    for i, r in enumerate(results):
        status = r.get("status", "ok")
        pred   = r.get("predictions", [{}])[0]
        print(f"    Request {i}: status={status}  "
              f"top={pred.get('label','N/A')}  "
              f"prob={pred.get('probability','N/A')}")

    errors = [m for m in ctx.metrics._log if "error" in m["name"]]
    print(f"  Error metrics emitted: {len(errors)}")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Model Packaging — .mar Files, TorchScript, and Archiver": {
        "description": (
            "Complete model packaging workflow for TorchServe. "
            "TorchScript: tracing vs scripting comparison. "
            "Scripting benefits: control flow, dynamic shapes. "
            "Tracing limitations: frozen branches. "
            "State dict vs full model vs TorchScript serialisation. "
            "MANIFEST.json structure and content. "
            "torch-model-archiver CLI simulation. "
            "MAR file contents: what goes inside the archive. "
            "Extra files: class maps, tokenizer configs, vocab files. "
            "requirements.txt in a MAR: dependency specification. "
            "Model versioning in the archiver (--version flag). "
            "Loading and validating a packaged model."
        ),
        "language": "python",
        "code": '''
import os
import io
import json
import time
import math
import zipfile
import tempfile
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.jit
    print(f"  PyTorch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "torch", "--quiet"], check=True)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.jit
    print(f"  PyTorch: {torch.__version__}")

print("=" * 65)
print("  MODEL PACKAGING — .MAR FILES, TORCHSCRIPT, AND ARCHIVER")
print("=" * 65)
print()

rng = np.random.default_rng(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: TorchScript — tracing vs scripting
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — TorchScript: trace vs script comparison")
print("━" * 65)
print()

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim=32, hidden=64, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicClassifier(nn.Module):
    """Model with control flow — tracing will miss branches."""
    def __init__(self, in_dim=32, hidden=64, out_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        # Data-dependent control flow — tracing freezes this branch!
        if logits.max() > threshold:
            return logits.softmax(-1)
        return logits   # return raw logits for uncertain predictions


model_simple  = SimpleClassifier()
model_dynamic = DynamicClassifier()
model_simple.eval()
model_dynamic.eval()

dummy32 = torch.randn(1, 32)

# Tracing
t0             = time.perf_counter()
traced_simple  = torch.jit.trace(model_simple, dummy32)
t_trace_simple = (time.perf_counter() - t0) * 1000

# Scripting
t0             = time.perf_counter()
scripted_simple = torch.jit.script(model_simple)
t_script_simple = (time.perf_counter() - t0) * 1000

t0              = time.perf_counter()
scripted_dynamic = torch.jit.script(model_dynamic)
t_script_dynamic = (time.perf_counter() - t0) * 1000

print(f"  Compilation times:")
print(f"  {'Method':<30} {'Time ms':>10}")
print(f"  {'─'*42}")
print(f"  {'trace(SimpleClassifier)':<30} {t_trace_simple:>10.2f}")
print(f"  {'script(SimpleClassifier)':<30} {t_script_simple:>10.2f}")
print(f"  {'script(DynamicClassifier)':<30} {t_script_dynamic:>10.2f}")
print()

# Verify numerical equivalence
with torch.no_grad():
    x_test = torch.randn(4, 32)
    orig_out   = model_simple(x_test)
    traced_out = traced_simple(x_test)
    scripted_out = scripted_simple(x_test)

    diff_trace  = (orig_out - traced_out).abs().max().item()
    diff_script = (orig_out - scripted_out).abs().max().item()
    print(f"  Numerical equivalence (max abs diff):")
    print(f"    Original vs Traced:   {diff_trace:.2e}  ✅" if diff_trace < 1e-5 else f"    ❌ {diff_trace:.2e}")
    print(f"    Original vs Scripted: {diff_script:.2e}  ✅" if diff_script < 1e-5 else f"    ❌ {diff_script:.2e}")
print()

# Show the TorchScript IR for the simple model
print(f"  TorchScript IR (scripted simple model, first few lines):")
graph_str = str(scripted_simple.graph).split("\\n")
for line in graph_str[:12]:
    print(f"    {line}")
print(f"    ... ({len(graph_str)} lines total)")
print()

# Demonstrate tracing freeze issue
print(f"  Tracing control flow limitation:")
# Trace with threshold=0.5 (logits.max() likely > 0.5 at init)
traced_dyn = torch.jit.trace(
    model_dynamic,
    (dummy32, torch.tensor(0.5)),
)
# Now call with threshold=100.0 — traced version ignores the threshold!
with torch.no_grad():
    x_dyn   = torch.randn(1, 32)
    script_out_high = scripted_dynamic(x_dyn, threshold=100.0)
    traced_out_high = traced_dyn(x_dyn, torch.tensor(100.0))

# With threshold=100, logits.max() < threshold → should return raw logits
script_is_proba = bool((script_out_high.sum() - 1.0).abs() < 0.01)
traced_is_proba = bool((traced_out_high.sum() - 1.0).abs() < 0.01)
print(f"  With threshold=100 (very high, should return raw logits):")
print(f"    Scripted output is proba (sum≈1): {script_is_proba}  "
      f"({'WRONG: took softmax' if script_is_proba else 'CORRECT: raw logits'})")
print(f"    Traced  output is proba (sum≈1): {traced_is_proba}  "
      f"({'WRONG: branch frozen at trace time' if traced_is_proba else 'CORRECT'})")
print(f"  → Scripting correctly handles data-dependent branches; tracing does not.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Serialisation formats and size comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Serialisation formats: state dict vs script vs full")
print("━" * 65)
print()

with tempfile.TemporaryDirectory() as tmp:
    # 1. State dict
    state_path = os.path.join(tmp, "model_state.pt")
    torch.save(model_simple.state_dict(), state_path)
    state_size = os.path.getsize(state_path)

    # 2. Full model (pickle-based)
    full_path = os.path.join(tmp, "model_full.pt")
    torch.save(model_simple, full_path)
    full_size = os.path.getsize(full_path)

    # 3. TorchScript (traced)
    traced_path = os.path.join(tmp, "model_traced.pt")
    traced_simple.save(traced_path)
    traced_size = os.path.getsize(traced_path)

    # 4. TorchScript (scripted)
    scripted_path = os.path.join(tmp, "model_scripted.pt")
    scripted_simple.save(scripted_path)
    scripted_size = os.path.getsize(scripted_path)

    # Count model parameters
    n_params = sum(p.numel() for p in model_simple.parameters())
    raw_size = n_params * 4   # float32

    print(f"  Model: SimpleClassifier  ({n_params:,} params, {raw_size} raw bytes)")
    print()
    print(f"  {'Format':<22} {'File size':>12} {'Overhead':>12} {'Loads without class def'}")
    print(f"  {'─'*65}")
    formats = [
        ("State dict",      state_size,   False),
        ("Full model",      full_size,    False),
        ("TorchScript (traced)",  traced_size, True),
        ("TorchScript (scripted)",scripted_size, True),
    ]
    for name, size, no_class in formats:
        overhead = size - raw_size
        print(f"  {name:<22} {size:>12,}B {overhead:>+12,}B "
              f"{'✓' if no_class else '✗ (needs class)'}")
    print()

    # Verify state dict reload
    model_reloaded = SimpleClassifier()
    model_reloaded.load_state_dict(torch.load(state_path, weights_only=True))
    model_reloaded.eval()
    with torch.no_grad():
        r_orig   = model_simple(x_test)
        r_reload = model_reloaded(x_test)
    reload_diff = (r_orig - r_reload).abs().max().item()
    print(f"  State dict reload accuracy: diff={reload_diff:.2e} ✅")

    # Verify TorchScript reload (no class needed)
    scripted_reloaded = torch.jit.load(scripted_path)
    scripted_reloaded.eval()
    with torch.no_grad():
        r_scripted = scripted_reloaded(x_test)
    scripted_diff = (r_orig - r_scripted).abs().max().item()
    print(f"  TorchScript reload accuracy: diff={scripted_diff:.2e} ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Building the MANIFEST.json
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — MANIFEST.json: the MAR file identity card")
print("━" * 65)
print()

def build_manifest(
    model_name:      str,
    version:         str,
    serialized_file: str,
    handler:         str,
    model_file:      Optional[str] = None,
    archiver_ver:    str = "0.10.0",
) -> Dict:
    """Generate the MAR MANIFEST.json."""
    manifest = {
        "createdOn":      time.strftime("%m/%d/%Y %H:%M:%S"),
        "runtime":        "python",
        "model": {
            "modelName":      model_name,
            "serializedFile": serialized_file,
            "handler":        handler.replace(".py", ""),
            "modelVersion":   version,
        },
        "archiverVersion": archiver_ver,
    }
    if model_file:
        manifest["model"]["modelFile"] = model_file
    return manifest


manifest_example = build_manifest(
    model_name      = "resnet50",
    version         = "2.1",
    serialized_file = "resnet50_scripted.pt",
    handler         = "image_handler",
)

print(f"  MANIFEST.json example:")
print(json.dumps(manifest_example, indent=2))
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Simulating torch-model-archiver
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — torch-model-archiver simulation: building a .mar")
print("━" * 65)
print()

def build_mar(
    model_name:      str,
    version:         str,
    serialized_file: str,
    handler_file:    str,
    export_path:     str,
    extra_files:     Optional[List[str]] = None,
    model_file:      Optional[str] = None,
    requirements:    Optional[str] = None,
) -> str:
    """
    Simulates torch-model-archiver.
    Creates a .mar (ZIP) file with the correct structure.
    """
    os.makedirs(export_path, exist_ok=True)
    mar_path = os.path.join(export_path, f"{model_name}.mar")

    manifest = build_manifest(
        model_name, version,
        os.path.basename(serialized_file),
        os.path.basename(handler_file),
        os.path.basename(model_file) if model_file else None,
    )

    with zipfile.ZipFile(mar_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. MANIFEST
        zf.writestr("MAR-INF/MANIFEST.json", json.dumps(manifest, indent=2))

        # 2. Serialised model
        if os.path.exists(serialized_file):
            zf.write(serialized_file, os.path.basename(serialized_file))

        # 3. Handler
        if os.path.exists(handler_file):
            zf.write(handler_file, os.path.basename(handler_file))

        # 4. Model class file (if any)
        if model_file and os.path.exists(model_file):
            zf.write(model_file, os.path.basename(model_file))

        # 5. Extra files
        for ef in (extra_files or []):
            if os.path.exists(ef):
                zf.write(ef, os.path.basename(ef))

        # 6. Requirements
        if requirements and os.path.exists(requirements):
            zf.write(requirements, "requirements.txt")

    return mar_path


with tempfile.TemporaryDirectory() as workspace:
    # Create all the files that would go into the MAR
    model_store = os.path.join(workspace, "model_store")

    # 1. Save TorchScript model
    model_pt_path = os.path.join(workspace, "resnet50_scripted.pt")
    scripted_simple.save(model_pt_path)

    # 2. Write handler file
    handler_code = (
        "from ts.torch_handler.base_handler import BaseHandler\n"
        "import torch\n\n"
        "class ImageClassifierHandler(BaseHandler):\n"
        "    def preprocess(self, data):\n"
        "        import torchvision.transforms as T\n"
        "        transform = T.Compose([T.Resize(224), T.CenterCrop(224),\n"
        "                                T.ToTensor(), T.Normalize([0.485],[0.229])])\n"
        "        return torch.stack([transform(item['body']) for item in data])\n\n"
        "    def postprocess(self, output):\n"
        "        probs = output.softmax(-1)\n"
        "        return [{'label': int(p.argmax()), 'prob': float(p.max())}\n"
        "                for p in probs]\n"
    )
    handler_path = os.path.join(workspace, "image_handler.py")
    with open(handler_path, "w") as f:
        f.write(handler_code)

    # 3. Class labels file (extra file)
    classes_path = os.path.join(workspace, "imagenet_classes.txt")
    with open(classes_path, "w") as f:
        f.write("\\n".join([f"class_{i:04d}" for i in range(5)]))

    # 4. Requirements
    reqs_path = os.path.join(workspace, "requirements.txt")
    with open(reqs_path, "w") as f:
        f.write("torchvision>=0.15.0\\nPillow>=9.0.0\\n")

    # Build the MAR
    mar_path = build_mar(
        model_name     = "resnet50",
        version        = "2.1",
        serialized_file = model_pt_path,
        handler_file   = handler_path,
        export_path    = model_store,
        extra_files    = [classes_path],
        requirements   = reqs_path,
    )

    # Inspect the MAR
    mar_size = os.path.getsize(mar_path)
    print(f"  Built: {os.path.basename(mar_path)}")
    print(f"  Size:  {mar_size:,} bytes ({mar_size/1024:.1f} KB)")
    print()

    print(f"  MAR contents (ZIP listing):")
    with zipfile.ZipFile(mar_path, "r") as zf:
        file_list = zf.infolist()
        print(f"  {'File':<40} {'Size':>10}")
        print(f"  {'─'*52}")
        for info in file_list:
            print(f"  {info.filename:<40} {info.file_size:>10,}B")
        print()

        # Read and show MANIFEST
        manifest_content = json.loads(zf.read("MAR-INF/MANIFEST.json"))
        print(f"  MANIFEST.json contents:")
        print(json.dumps(manifest_content, indent=4))

    print()
    print(f"  CLI equivalent command:")
    print(f"  torch-model-archiver \\")
    print(f"    --model-name     resnet50 \\")
    print(f"    --version        2.1 \\")
    print(f"    --serialized-file {os.path.basename(model_pt_path)} \\")
    print(f"    --handler        {os.path.basename(handler_path)} \\")
    print(f"    --extra-files    {os.path.basename(classes_path)} \\")
    print(f"    --requirements-file {os.path.basename(reqs_path)} \\")
    print(f"    --export-path    model_store/")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Batching, Performance & Optimisation": {
        "description": (
            "Batching mechanics and performance analysis for TorchServe. "
            "Batch accumulation: max_batch_size and max_batch_delay dynamics. "
            "GPU efficiency: throughput vs latency trade-off curves. "
            "Batch-size sensitivity: throughput at 1, 8, 32, 64, 128. "
            "Simulated concurrent request arrival and batching decisions. "
            "torch.compile() integration in handler initialize(). "
            "FP16 autocast: latency and memory impact analysis. "
            "Dynamic quantisation: model size and latency comparison. "
            "Preprocess vectorisation: list-of-images vs single stack. "
            "Worker pool sizing: CPU-bound vs GPU-bound recommendations. "
            "Profiling handler phases: pre/infer/post breakdown. "
            "Latency percentile analysis: p50/p95/p99 under load."
        ),
        "language": "python",
        "code": '''
import time
import math
import threading
import queue
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import statistics

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"  PyTorch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "torch", "--quiet"], check=True)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"  PyTorch: {torch.__version__}")

print("=" * 65)
print("  BATCHING, PERFORMANCE & OPTIMISATION")
print("=" * 65)
print()

rng = np.random.default_rng(42)
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}")
print()

# ── Model for all benchmarks ──────────────────────────────────────────────
class BenchModel(nn.Module):
    def __init__(self, in_dim=512, hidden=1024, out_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, out_dim),
        )
    def forward(self, x): return self.layers(x)

model = BenchModel().to(DEVICE)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Benchmark model: 512→1024→1024→512→10  ({n_params:,} params)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Batch size → throughput curve
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Batch size vs throughput: the GPU efficiency curve")
print("━" * 65)
print()

N_REPS = 100

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(torch.randn(8, 512).to(DEVICE))

print(f"  Throughput at different batch sizes ({N_REPS} reps each):")
print(f"  {'Batch':>8} {'p50 ms':>10} {'p95 ms':>10} {'Req/s':>12} "
      f"{'ms/req':>10} {'Efficiency'}")
print(f"  {'─'*65}")

baseline_thr = None
for bs in [1, 4, 8, 16, 32, 64, 128, 256]:
    x = torch.randn(bs, 512).to(DEVICE)
    times = []
    with torch.no_grad():
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            _ = model(x)
            if DEVICE == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    p50     = statistics.median(times)
    p95     = sorted(times)[int(N_REPS * 0.95)]
    rps     = bs / (p50 / 1000)
    ms_req  = p50 / bs

    if baseline_thr is None: baseline_thr = rps
    eff_pct = rps / (baseline_thr * bs) * 100
    bar     = "█" * min(20, int(eff_pct / 5))

    print(f"  {bs:>8} {p50:>10.2f} {p95:>10.2f} {rps:>12.0f} "
          f"{ms_req:>10.4f} {eff_pct:>8.1f}%  {bar}")
print()
print(f"  Efficiency = actual throughput / (single-sample × batch_size)")
print(f"  Higher batch sizes → better GPU utilisation (approach 100%)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Batching dynamics simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Batching dynamics: request arrival and dispatch")
print("━" * 65)
print()

@dataclass
class BatchingSimulator:
    """
    Simulates TorchServe's batching logic:
    - Requests arrive at a Poisson rate
    - Dispatcher waits up to max_batch_delay or until max_batch_size filled
    """
    max_batch_size:   int   = 32
    max_batch_delay:  float = 0.1    # seconds
    model_latency_fn: Any  = None    # callable: batch_size → latency_ms

    def __post_init__(self):
        if self.model_latency_fn is None:
            # Simulate GPU latency: base + per-sample (sublinear)
            self.model_latency_fn = lambda bs: 5.0 + 0.3 * math.sqrt(bs)

    def simulate(self, arrival_rps: float, n_requests: int) -> Dict:
        """Simulate n_requests arriving at arrival_rps."""
        arrival_interval = 1.0 / arrival_rps   # seconds between arrivals
        results = {
            "batch_sizes":     [],
            "queue_latencies": [],
            "total_latencies": [],
        }

        t  = 0.0
        pending = []
        served  = 0

        while served < n_requests:
            # Arrival event
            pending.append({"arrival": t})
            t += arrival_interval + rng.exponential(arrival_interval * 0.1)

            # Dispatch condition: batch full OR timeout
            oldest_arrival = pending[0]["arrival"] if pending else t
            timeout_reached = (t - oldest_arrival) >= self.max_batch_delay
            batch_full      = len(pending) >= self.max_batch_size

            if batch_full or timeout_reached:
                batch       = pending[:self.max_batch_size]
                pending     = pending[self.max_batch_size:]
                batch_size  = len(batch)
                infer_ms    = self.model_latency_fn(batch_size)
                dispatch_t  = t

                for req in batch:
                    queue_lat = (dispatch_t - req["arrival"]) * 1000
                    total_lat = queue_lat + infer_ms
                    results["batch_sizes"].append(batch_size)
                    results["queue_latencies"].append(queue_lat)
                    results["total_latencies"].append(total_lat)
                    served += 1

        return results


print(f"  Simulating different request rates (max_batch=32, timeout=100ms):")
print()
print(f"  {'RPS':>6} {'Avg batch':>12} {'p50 lat ms':>12} {'p95 lat ms':>12} "
      f"{'Queue lat':>12}")
print(f"  {'─'*58}")

simulator = BatchingSimulator(max_batch_size=32, max_batch_delay=0.1)
for rps in [5, 20, 50, 100, 200, 500]:
    results = simulator.simulate(arrival_rps=rps, n_requests=500)
    avg_bs  = statistics.mean(results["batch_sizes"])
    p50_tot = statistics.median(results["total_latencies"])
    p95_tot = sorted(results["total_latencies"])[int(len(results["total_latencies"])*0.95)]
    avg_q   = statistics.mean(results["queue_latencies"])
    print(f"  {rps:>6} {avg_bs:>12.1f} {p50_tot:>12.1f} {p95_tot:>12.1f} {avg_q:>12.1f}")

print()
print(f"  Observations:")
print(f"  - Low RPS: small batches, low queue latency, higher GPU waste")
print(f"  - High RPS: large batches, some queue wait, excellent GPU efficiency")
print(f"  - At very high RPS: queue grows → latency spikes → need more workers")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: torch.compile and FP16 optimisation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — torch.compile and FP16: latency improvements")
print("━" * 65)
print()

model_fp32    = BenchModel().to(DEVICE)
model_fp32.eval()

# FP16 model (if CUDA available)
if DEVICE == "cuda":
    model_fp16 = BenchModel().to(DEVICE).half()
    model_fp16.eval()

# torch.compile (skip on some environments — safe fallback)
try:
    model_compiled = torch.compile(BenchModel().to(DEVICE), mode="reduce-overhead")
    model_compiled.eval()
    # warmup compile
    with torch.no_grad():
        for _ in range(5):
            _ = model_compiled(torch.randn(32, 512).to(DEVICE))
    HAS_COMPILE = True
except Exception:
    HAS_COMPILE = False

BS = 32
x_fp32 = torch.randn(BS, 512).to(DEVICE)

N_BENCH = 200
def bench_model(m, x, n=N_BENCH):
    with torch.no_grad():
        for _ in range(10): m(x)   # warmup
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            _ = m(x)
            if DEVICE == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)

times_fp32 = bench_model(model_fp32, x_fp32)
p50_fp32   = statistics.median(times_fp32)

rows = [("FP32 (baseline)", times_fp32, 1.0)]

if DEVICE == "cuda":
    x_fp16 = x_fp32.half()
    times_fp16 = bench_model(model_fp16, x_fp16)
    rows.append(("FP16 (autocast)", times_fp16, None))

if HAS_COMPILE:
    times_comp = bench_model(model_compiled, x_fp32)
    rows.append(("torch.compile", times_comp, None))

print(f"  Inference optimisation comparison (batch={BS}, {N_BENCH} runs):")
print(f"  {'Method':<25} {'p50 ms':>10} {'p95 ms':>10} {'Speedup':>10} "
      f"{'Memory'}")
print(f"  {'─'*60}")
for name, times, _ in rows:
    p50 = statistics.median(times)
    p95 = sorted(times)[int(N_BENCH * 0.95)]
    sp  = p50_fp32 / p50
    mem = "~100%" if "FP32" in name else "~50%" if "FP16" in name else "~100%"
    print(f"  {name:<25} {p50:>10.3f} {p95:>10.3f} {sp:>10.2f}× {mem}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Dynamic quantisation impact
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Dynamic quantisation: size and latency analysis")
print("━" * 65)
print()

import tempfile, os

model_for_quant = BenchModel().cpu()
model_for_quant.eval()

# Dynamic quantise
model_int8 = torch.quantization.quantize_dynamic(
    model_for_quant,
    {nn.Linear},
    dtype=torch.qint8,
)

# Save both to measure size
with tempfile.TemporaryDirectory() as tmp:
    fp32_path = os.path.join(tmp, "fp32.pt")
    int8_path = os.path.join(tmp, "int8.pt")
    torch.save(model_for_quant.state_dict(), fp32_path)
    torch.save(model_int8.state_dict(),      int8_path)

    fp32_size = os.path.getsize(fp32_path)
    int8_size = os.path.getsize(int8_path)

# Benchmark on CPU
x_cpu = torch.randn(BS, 512).cpu()
times_fp32_cpu = bench_model(model_for_quant, x_cpu)
times_int8_cpu = bench_model(model_int8, x_cpu)
p50_fp32_cpu   = statistics.median(times_fp32_cpu)
p50_int8_cpu   = statistics.median(times_int8_cpu)

print(f"  Dynamic INT8 quantisation (CPU inference, batch={BS}):")
print(f"  {'Metric':<25} {'FP32':>12} {'INT8':>12} {'Ratio':>10}")
print(f"  {'─'*62}")
print(f"  {'State dict size (KB)':<25} {fp32_size/1024:>12.1f} "
      f"{int8_size/1024:>12.1f} {int8_size/fp32_size:>10.2f}×")
print(f"  {'p50 latency (ms)':<25} {p50_fp32_cpu:>12.3f} "
      f"{p50_int8_cpu:>12.3f} {p50_fp32_cpu/p50_int8_cpu:>10.2f}×")
print(f"  {'Throughput (req/s)':<25} "
      f"{BS/(p50_fp32_cpu/1000):>12.0f} "
      f"{BS/(p50_int8_cpu/1000):>12.0f} "
      f"{p50_fp32_cpu/p50_int8_cpu:>10.2f}×")
print()

# Accuracy check
with torch.no_grad():
    out_fp32 = model_for_quant(x_cpu[:4])
    out_int8 = model_int8(x_cpu[:4])
    diff     = (out_fp32 - out_int8).abs().max().item()
    argmax_match = (out_fp32.argmax(-1) == out_int8.argmax(-1)).float().mean()

print(f"  Quantisation accuracy:")
print(f"    Max output diff:    {diff:.4f}")
print(f"    Argmax agreement:   {argmax_match*100:.1f}%")
print()
print(f"  TorchServe handler integration:")
print(f"    In initialize():")
print(f"      self.model = torch.quantization.quantize_dynamic(")
print(f"          self.model, {{nn.Linear}}, dtype=torch.qint8)")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Management API, Multi-Model Serving & Production Patterns": {
        "description": (
            "TorchServe management API simulation and production patterns. "
            "Management API: register, list, scale, version, unregister. "
            "Model versioning: register v1 and v2 simultaneously. "
            "Blue/green deployment: zero-downtime model promotion. "
            "Worker autoscaling: queue-depth triggered scaling. "
            "Multi-model serving: independent resource allocation. "
            "config.properties: complete production configuration. "
            "Health and readiness probe endpoints: /ping, /models. "
            "Latency SLO monitoring: p50/p95/p99 with alert thresholds. "
            "Docker deployment configuration. "
            "TorchServe vs alternatives comparison. "
            "Production deployment checklist."
        ),
        "language": "python",
        "code": '''
import os
import json
import time
import math
import statistics
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

print("=" * 65)
print("  MANAGEMENT API, MULTI-MODEL SERVING & PRODUCTION PATTERNS")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Management API simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Management API: register, scale, version, unregister")
print("━" * 65)
print()

@dataclass
class WorkerInfo:
    worker_id:   int
    state:       str    # READY, LOADING, BUSY
    gpu_id:      Optional[int] = None
    pid:         int = 0
    start_time:  float = field(default_factory=time.time)


@dataclass
class ModelVersion:
    version:     str
    mar_url:     str
    min_workers: int = 1
    max_workers: int = 4
    batch_size:  int = 32
    batch_delay: int = 100    # ms
    is_default:  bool = False
    workers:     List[WorkerInfo] = field(default_factory=list)
    total_requests: int = 0
    total_errors:   int = 0

    def scale(self, desired: int):
        """Scale worker count toward desired."""
        current = len(self.workers)
        if desired > current:
            for _ in range(desired - current):
                self.workers.append(WorkerInfo(
                    worker_id = len(self.workers),
                    state     = "READY",
                    gpu_id    = len(self.workers) % max(1, 4),  # round-robin GPU
                    pid       = 10000 + len(self.workers),
                ))
        elif desired < current:
            self.workers = self.workers[:desired]

    def status(self) -> Dict:
        return {
            "version":    self.version,
            "is_default": self.is_default,
            "batch_size": self.batch_size,
            "workers":    [
                {"id": str(w.pid), "state": w.state, "gpu": str(w.gpu_id)}
                for w in self.workers
            ],
            "total_requests": self.total_requests,
        }


class MockManagementAPI:
    """Simulates the TorchServe Management API (port 8081)."""

    def __init__(self, model_store: str = "./model_store"):
        self.model_store = model_store
        self._models: Dict[str, List[ModelVersion]] = {}

    def register_model(
        self,
        url:             str,
        model_name:      str    = None,
        initial_workers: int    = 1,
        max_workers:     int    = 4,
        batch_size:      int    = 32,
        batch_delay:     int    = 100,
        version:         str    = None,
        set_default:     bool   = True,
    ) -> Dict:
        # Extract name/version from URL if not provided
        mar_name = os.path.basename(url).replace(".mar", "")
        if model_name is None:
            model_name = mar_name
        if version is None:
            version = "1.0"

        if model_name not in self._models:
            self._models[model_name] = []

        # Check for duplicate version
        for v in self._models[model_name]:
            if v.version == version:
                return {"error": f"Version {version} already exists", "code": 409}

        mv = ModelVersion(
            version     = version,
            mar_url     = url,
            min_workers = initial_workers,
            max_workers = max_workers,
            batch_size  = batch_size,
            batch_delay = batch_delay,
            is_default  = set_default or not self._models[model_name],
        )

        # Clear other defaults if this is set as default
        if mv.is_default:
            for v in self._models[model_name]:
                v.is_default = False

        mv.scale(initial_workers)
        self._models[model_name].append(mv)
        return {
            "status": "Model registered",
            "name":   model_name,
            "version": version,
            "workers": len(mv.workers),
        }

    def list_models(self) -> Dict:
        return {
            "models": [
                {"modelName": name, "modelUrl": f"{name}.mar"}
                for name in self._models
            ]
        }

    def describe_model(self, name: str, version: str = None) -> Dict:
        if name not in self._models:
            return {"error": f"Model {name} not found", "code": 404}
        versions = self._models[name]
        if version:
            versions = [v for v in versions if v.version == version]
        return {"name": name, "versions": [v.status() for v in versions]}

    def scale_workers(self, name: str, version: str,
                      min_worker: int, max_worker: int) -> Dict:
        for v in self._models.get(name, []):
            if v.version == version:
                v.min_workers = min_worker
                v.max_workers = max_worker
                v.scale(min_worker)
                return {"status": "Workers updated", "workers": len(v.workers)}
        return {"error": "Version not found", "code": 404}

    def set_default(self, name: str, version: str) -> Dict:
        for v in self._models.get(name, []):
            v.is_default = (v.version == version)
        return {"status": f"Default set to {version}"}

    def unregister_model(self, name: str, version: str) -> Dict:
        if name not in self._models:
            return {"error": "Model not found", "code": 404}
        self._models[name] = [v for v in self._models[name]
                               if v.version != version]
        if not self._models[name]:
            del self._models[name]
        return {"status": f"{name}/{version} unregistered"}


# Demonstrate the Management API
api = MockManagementAPI()

print(f"  Management API demo (port 8081 in production):")
print()

# Register v1
r = api.register_model("model_store/resnet50-v1.mar", model_name="resnet50",
                        version="1.0", initial_workers=2, max_workers=4)
print(f"  POST /models  (register v1.0)")
print(f"    Response: {r}")

# Register v2 alongside v1
r = api.register_model("model_store/resnet50-v2.mar", model_name="resnet50",
                        version="2.0", initial_workers=1, set_default=False)
print(f"\n  POST /models  (register v2.0, not default yet)")
print(f"    Response: {r}")

# Describe both versions
desc = api.describe_model("resnet50")
print(f"\n  GET /models/resnet50  (describe all versions)")
for vinfo in desc["versions"]:
    print(f"    v{vinfo['version']} (default={vinfo['is_default']}): "
          f"{len(vinfo['workers'])} workers  |  "
          f"batch_size={vinfo['batch_size']}")

# Blue-green promotion
print(f"\n  === Blue/green deployment sequence ===")

print(f"\n  1. Scale up v2.0 for traffic")
r = api.scale_workers("resnet50", "2.0", min_worker=4, max_worker=8)
print(f"     PUT /models/resnet50/2.0?min_worker=4  → {r}")

print(f"\n  2. Set v2.0 as default")
r = api.set_default("resnet50", "2.0")
print(f"     PUT /models/resnet50/2.0/set-default   → {r}")

print(f"\n  3. Scale down v1.0")
r = api.scale_workers("resnet50", "1.0", min_worker=0, max_worker=0)
print(f"     PUT /models/resnet50/1.0?min_worker=0  → {r}")

print(f"\n  4. Unregister v1.0")
r = api.unregister_model("resnet50", "1.0")
print(f"     DELETE /models/resnet50/1.0             → {r}")

desc = api.describe_model("resnet50")
print(f"\n  Final state:")
for vinfo in desc["versions"]:
    print(f"    v{vinfo['version']} (default={vinfo['is_default']}): "
          f"{len(vinfo['workers'])} workers  "
          f"[{', '.join(w['state'] for w in vinfo['workers'][:3])}]")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Multi-model resource analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Multi-model serving: resource allocation")
print("━" * 65)
print()

models_config = [
    {"name": "resnet50",    "workers": 2, "gpu_mem_gb": 0.8, "batch": 64,  "rps_target": 200},
    {"name": "bert-base",   "workers": 2, "gpu_mem_gb": 1.5, "batch": 32,  "rps_target": 80},
    {"name": "yolov8",      "workers": 1, "gpu_mem_gb": 2.5, "batch": 16,  "rps_target": 30},
    {"name": "whisper-s",   "workers": 1, "gpu_mem_gb": 1.2, "batch": 4,   "rps_target": 15},
    {"name": "gpt2-med",    "workers": 1, "gpu_mem_gb": 3.0, "batch": 8,   "rps_target": 10},
]

gpu_vram_gb = 24.0   # A100-24GB

print(f"  Serving {len(models_config)} models on 1× A100 24GB GPU:")
print()
print(f"  {'Model':<14} {'Workers':>8} {'GPU VRAM':>10} {'Batch':>7} "
      f"{'RPS target':>12} {'Cumulative VRAM'}")
print(f"  {'─'*65}")

cumulative_vram = 0.0
for cfg in models_config:
    vram_total = cfg["workers"] * cfg["gpu_mem_gb"]
    cumulative_vram += vram_total
    fits = "✅" if cumulative_vram <= gpu_vram_gb else "⚠️ OOM"
    print(f"  {cfg['name']:<14} {cfg['workers']:>8} "
          f"{cfg['gpu_mem_gb']:>9.1f}GB {cfg['batch']:>7} "
          f"{cfg['rps_target']:>12} "
          f"{cumulative_vram:>10.1f}/{gpu_vram_gb:.0f}GB {fits}")
print()
print(f"  Recommendations:")
print(f"    - Keep total VRAM < 80% of GPU memory (leave headroom for activation)")
print(f"    - Use separate GPUs for models with very different latency SLOs")
print(f"    - Heavy models (GPT-2) should have exclusive GPU access")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: config.properties and production configuration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — config.properties: production configuration reference")
print("━" * 65)
print()

config_production = """# TorchServe Production Configuration
# ────────────────────────────────────────────────────────────────

# Network binding
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

# Model store location (filesystem or S3 URI)
model_store=/home/model-server/model-store

# Models to load on startup (comma-separated)
load_models=resnet50.mar,bert-base.mar

# ─── Default serving parameters ──────────────────────────────────
batch_size=32
max_batch_delay=100
default_workers_per_model=1
job_queue_size=1000

# ─── Performance ──────────────────────────────────────────────────
# Netty worker threads (HTTP request handling — Java layer)
number_of_netty_threads=32
# Response timeout in milliseconds
default_response_timeout=120000
# Max connection backlog
backlog=1000

# ─── GPU ──────────────────────────────────────────────────────────
# Number of GPUs to use (0 = CPU only)
# This is overridden per-model via Management API
# Handled via --gpus docker flag or num_gpu in management API

# ─── Security ─────────────────────────────────────────────────────
enable_cors=true
cors_allowed_origin=https://my-app.company.com
cors_allowed_methods=GET, POST, PUT, OPTIONS
enable_metrics_api=true

# ─── Model-specific settings ─────────────────────────────────────
# [resnet50]
# batch_size=64
# max_batch_delay=50
# initial_workers=2
# max_workers=8

# ─── Logging ──────────────────────────────────────────────────────
# log4j2.xml path for custom log configuration
# Enable access logs:
# enable_access_log=true
"""

print(config_production)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Production deployment and ecosystem summary
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Production deployment guide and ecosystem")
print("━" * 65)
print()

print(f"  TorchServe deployment commands:")
print()
COMMANDS = [
    ("Start (local)",
     "torchserve --start --model-store ./model_store"),
    ("Start (with models)",
     "torchserve --start --model-store ./model_store --models resnet50.mar"),
    ("Start (with config)",
     "torchserve --start --model-store ./model_store --ts-config config.properties"),
    ("Stop",
     "torchserve --stop"),
    ("Register model",
     "curl -X POST 'http://localhost:8081/models?url=resnet50.mar&initial_workers=2'"),
    ("Predict (JSON)",
     "curl -X POST http://localhost:8080/predictions/resnet50 -H 'Content-Type: application/json' -d '...'"),
    ("Predict (image file)",
     "curl -X POST http://localhost:8080/predictions/resnet50 -T image.jpg"),
    ("Scale workers",
     "curl -X PUT 'http://localhost:8081/models/resnet50?min_worker=4&max_worker=8'"),
    ("Docker (GPU)",
     "docker run --gpus all -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest-gpu"),
]
for desc, cmd in COMMANDS:
    print(f"  [{desc}]")
    print(f"    {cmd[:80]}")
    print()

print(f"  TorchServe vs alternative serving frameworks:")
print(f"  {'Framework':<18} {'PyTorch Native':>16} {'Batching':>10} "
      f"{'Multi-model':>13} {'Complexity'}")
print(f"  {'─'*65}")
for name, native, batching, multi, complexity in [

    ("TorchServe",     "✓ (built for it)",   "Built-in",  "✓",    "Medium"),
    ("Triton",         "Via ONNX/TS",        "Built-in",  "✓✓",   "High"),
    ("BentoML",        "✓",                  "✓",         "✓",    "Low"),
    ("FastAPI + ORT",  "Via ONNX",           "Custom",    "✓",    "Low"),
    ("TF Serving",     "Via ONNX",           "Built-in",  "✓",    "Medium"),
    ("Ray Serve",      "✓",                  "✓",         "✓✓",   "Medium"),
    
]:
    print(f"  {name:<18} {native:>16} {batching:>10} {multi:>13} {complexity}")
print()

print(f"  Production checklist:")
checklist = [
    ("MAR packaging",       "TorchScript serialised model inside .mar"),
    ("Handler tested",      "Unit test handler locally before deployment"),
    ("Batch size tuned",    "Benchmark batch 1/8/32/64 on production hardware"),
    ("Workers configured",  "min_workers=N where N = GPU count"),
    ("Timeout set",         "default_response_timeout matches SLO budget"),
    ("Health probes",       "/ping (liveness) + /models (readiness) in K8s"),
    ("Metrics scraped",     "Prometheus scraping port 8082"),
    ("Grafana dashboard",   "Import TorchServe dashboard (ID: 17829)"),
    ("Version registered",  "Explicit --version for rollback capability"),
    ("MAR on S3",           "Centralised artefact storage for all environments"),
    ("Docker image pinned", "pytorch/torchserve:0.10.0-gpu not :latest"),
    ("Resource limits set", "Kubernetes memory/GPU limits prevent OOM cascade"),
]
for item, desc in checklist:
    print(f"  ✓ {item:<25} {desc}")
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