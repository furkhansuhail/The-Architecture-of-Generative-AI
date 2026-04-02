"""
FastAPI — High-Performance APIs for Machine Learning Systems
=============================================================

FastAPI is a modern, high-performance Python web framework for building
APIs, created by Sebastián Ramírez (tiangolo) and first released in
December 2018. It has become the dominant choice for building ML serving
APIs and data science web services, consistently ranking as one of the
fastest-growing Python frameworks.

FastAPI is built on three foundations:
    Starlette:   The ASGI web framework (routing, middleware, WebSockets)
    Pydantic:    Data validation and serialisation using Python type hints
    Uvicorn:     Lightning-fast ASGI server based on uvloop and httptools

The result is a framework that is:
    FAST:     Comparable to NodeJS and Go in throughput benchmarks.
              Powered by Starlette's async request handling and uvloop's
              event loop, FastAPI can handle tens of thousands of
              concurrent requests per second.
    CORRECT:  Pydantic v2 validates every request and response at the
              Python type-hint level. A wrong input type raises a 422
              error with a clear message — before it ever reaches your
              model inference code.
    DOCUMENTED: Automatic OpenAPI (Swagger) and ReDoc UI generated from
              your code. Every endpoint, every parameter, every response
              schema is documented without writing a single line of
              documentation code.
    PRODUCTIVE: Python type hints serve as the single source of truth
              for validation, serialisation, documentation, and IDE
              autocomplete simultaneously.

For ML systems specifically, FastAPI has become the standard because:
    1. Async support: inference can be non-blocking; I/O while GPU runs
    2. Background tasks: post-request processing (logging, monitoring)
    3. Dependency injection: shared resources (models, DB pools, config)
    4. WebSockets: streaming predictions, LLM token streaming
    5. Middleware: authentication, rate limiting, CORS, tracing
    6. Testing: TestClient for synchronous testing without a server

FastAPI powers the inference APIs of Hugging Face Spaces, Stability AI,
Cohere, and hundreds of ML startups. The pattern of "load model at startup,
serve predictions via FastAPI" is the dominant architecture for ML
microservices globally.

This module covers the complete FastAPI stack for ML: the ASGI programming
model and async/await fundamentals, Pydantic models for request/response
validation with ML data types, path and query parameters, dependency
injection for model lifecycle management, async inference patterns,
background tasks for monitoring and logging, WebSocket streaming for LLM
responses, middleware for authentication and rate limiting, error handling
and custom exception handlers, health checks and readiness probes,
testing with TestClient and pytest, and production deployment with Uvicorn
and Gunicorn.

"""

import textwrap
import re

TOPIC_NAME   = "FastAPI — High-Performance APIs for Machine Learning Systems"
DISPLAY_NAME = "17 · FastAPI"
ICON         = "⚡"
SUBTITLE     = "From Async Inference Endpoints to Production ML Microservices"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — ASGI, ASYNC/AWAIT, AND THE FASTAPI ARCHITECTURE

### WSGI vs ASGI: Why It Matters for ML

    The traditional Python web stack (Flask, Django) uses WSGI (Web Server
    Gateway Interface), which is synchronous: one request occupies one
    thread from start to finish.

    For a simple web page, this is fine. For ML inference:
        - GPU inference: GPU runs while CPU is completely idle, but the
          thread is BLOCKED — it cannot serve other requests
        - External services: if inference calls a model API (OpenAI, etc.),
          the network I/O blocks the entire thread
        - Database lookups: feature retrieval blocks the thread

    With a thread pool of N workers (typical: 4-16), WSGI can handle at
    most N concurrent blocking operations. Under load, requests queue up.

    ASGI (Asynchronous Server Gateway Interface) solves this:
        - Requests are handled as coroutines (async functions)
        - When a coroutine awaits I/O, the event loop serves other requests
        - A single process with a single thread can handle thousands of
          concurrent connections — as long as I/O is awaited properly

    The caveat for ML: CPU-bound work (running model inference locally) is
    NOT improved by async alone — it still occupies the CPU.
    Solutions:
        - Run blocking inference in a thread pool:
          await asyncio.get_event_loop().run_in_executor(None, model.predict, x)
        - Use GPU which is inherently concurrent with CPU
        - Use multiple worker processes (Gunicorn with Uvicorn workers)

### The Event Loop and Coroutines

    Python's asyncio event loop manages coroutines (async def functions).
    A coroutine is a function that can be suspended (at await points) and
    resumed later, without blocking the thread.

    Key async primitives:
        async def handler():     define a coroutine function
        await some_coroutine()   suspend until coroutine completes
        asyncio.gather(a, b)     run multiple coroutines concurrently
        asyncio.sleep(n)         non-blocking sleep (yields control)
        run_in_executor(...)     run blocking code in a thread pool

    In FastAPI:
        @app.post("/predict")
        async def predict(request: PredictRequest):
            # This runs in the event loop
            result = await run_inference(request.data)   # I/O bound
            return {"prediction": result}

        @app.post("/predict-sync")
        def predict_sync(request: PredictRequest):
            # FastAPI auto-detects sync functions and runs them
            # in a thread pool to avoid blocking the event loop
            result = model.predict(request.data)
            return {"prediction": result}

### Uvicorn: The ASGI Server

    Uvicorn is the standard ASGI server for FastAPI in production.
    It is built on uvloop (Cython-based event loop, 2-4× faster than
    asyncio) and httptools (fast HTTP parser).

    Development:
        uvicorn app.main:app --reload --port 8000
        # --reload: hot-reload on code change (development only)

    Production (single process):
        uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
        # --workers: multiple processes (bypasses GIL for CPU-bound work)

    Production (Gunicorn + Uvicorn workers):
        gunicorn app.main:app \\
            --workers 4 \\
            --worker-class uvicorn.workers.UvicornWorker \\
            --bind 0.0.0.0:8000
        # Gunicorn manages process lifecycle (restart on crash, signals)
        # Each worker is a full Uvicorn ASGI server
        # Rule of thumb: workers = 2 × CPU_cores + 1

### FastAPI Application Structure

    Minimal FastAPI app:
        from fastapi import FastAPI

        app = FastAPI(
            title       = "ML Inference API",
            description = "Serves predictions from our fraud detection model",
            version     = "2.1.0",
            docs_url    = "/docs",    # Swagger UI
            redoc_url   = "/redoc",   # ReDoc UI
        )

        @app.get("/")
        async def root():
            return {"message": "OK"}

    Lifespan (startup/shutdown):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup: runs before the first request
            app.state.model = load_model()      # load ML model once
            app.state.db    = await connect_db() # open DB pool
            yield
            # Shutdown: runs after last request
            await app.state.db.close()

        app = FastAPI(lifespan=lifespan)
        # Much cleaner than @app.on_event("startup") (deprecated)

    Router organisation (large apps):
        from fastapi import APIRouter

        router = APIRouter(prefix="/v1/models", tags=["inference"])

        @router.post("/{model_name}/predict")
        async def predict(model_name: str, ...): ...

        app.include_router(router)


##### PART 2 — PYDANTIC MODELS: VALIDATION AND SERIALISATION

### Pydantic v2: The Validation Engine

    Pydantic v2 (2023+) is a complete rewrite in Rust via pydantic-core.
    It is 5-50× faster than Pydantic v1 for validation and serialisation.

    Every FastAPI request body, query parameter, and response is validated
    by Pydantic. Invalid inputs return a 422 Unprocessable Entity response
    with a detailed error message before your code runs.

    Basic model:
        from pydantic import BaseModel, Field, field_validator
        from typing import Optional, List
        import numpy as np

        class PredictRequest(BaseModel):
            features:    List[float]
            model_name:  str          = "default"
            threshold:   float        = Field(0.5, ge=0.0, le=1.0)
            batch_id:    Optional[str] = None

        class PredictResponse(BaseModel):
            prediction:  int
            probability: float
            model_name:  str
            latency_ms:  float

### Field Validation for ML Data

    Constraining numeric fields:
        from pydantic import Field

        class ImageRequest(BaseModel):
            width:   int   = Field(..., ge=1, le=4096)   # required, 1-4096
            height:  int   = Field(..., ge=1, le=4096)
            pixels:  List[float] = Field(..., min_length=1, max_length=16777216)

    Custom validators (Pydantic v2 style):
        from pydantic import field_validator, model_validator

        class EmbeddingRequest(BaseModel):
            text:      str
            model:     str = "bert-base"
            max_length: int = 512

            @field_validator("text")
            @classmethod
            def text_not_empty(cls, v: str) -> str:
                if not v.strip():
                    raise ValueError("text must not be empty or whitespace")
                return v.strip()

            @field_validator("max_length")
            @classmethod
            def valid_max_length(cls, v: int) -> int:
                allowed = [128, 256, 512, 1024]
                if v not in allowed:
                    raise ValueError(f"max_length must be one of {allowed}")
                return v

            @model_validator(mode="after")
            def check_text_fits(self) -> "EmbeddingRequest":
                # Approximate token count (rough: chars/4)
                approx_tokens = len(self.text) // 4
                if approx_tokens > self.max_length:
                    raise ValueError(
                        f"Text too long (~{approx_tokens} tokens) "
                        f"for max_length={self.max_length}"
                    )
                return self

    Numpy/tensor types (requires custom serialisation):
        from pydantic import BaseModel
        import numpy as np
        from typing import Annotated

        class BatchPredictRequest(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            # For numpy arrays in requests, use List[List[float]] and convert
            features:   List[List[float]]    # JSON-serialisable
            labels:     Optional[List[int]] = None

            def to_numpy(self) -> np.ndarray:
                return np.array(self.features, dtype=np.float32)

### Response Models and Serialisation

    Defining response schemas:
        class ModelInfo(BaseModel):
            name:      str
            version:   str
            n_params:  int
            framework: str

        class HealthResponse(BaseModel):
            status:      str
            model_loaded: bool
            uptime_sec:  float
            gpu_available: bool

        @app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(
                status       = "ok",
                model_loaded = app.state.model is not None,
                uptime_sec   = time.time() - app.state.start_time,
                gpu_available = torch.cuda.is_available(),
            )

    Excluding fields from responses:
        class UserResponse(BaseModel):
            username:  str
            email:     str
            password:  str = Field(exclude=True)   # never serialised to response


##### PART 3 — ROUTING, PARAMETERS, AND REQUEST HANDLING

### HTTP Methods and ML Endpoints

    Standard ML API endpoint design:
        GET  /health              → server and model health status
        GET  /models              → list available models
        GET  /models/{name}       → model info (version, architecture)
        POST /models/{name}/predict → run inference on this model
        POST /predict             → run inference on default model
        POST /batch               → batch inference (multiple inputs)
        GET  /models/{name}/metrics → model performance metrics

    Endpoint parameters:
        Path parameters:    /models/{model_name}/predict
        Query parameters:   /predict?threshold=0.7&format=proba
        Request body:       POST with JSON body (PredictRequest)
        Headers:            Authorization, X-Request-ID, X-API-Key

### Path Parameters

    Strongly typed path parameters:
        @app.get("/models/{model_name}/version/{version_id}")
        async def get_model_info(
            model_name: str,
            version_id: int,       # FastAPI converts "3" → 3 automatically
        ):
            return {"model": model_name, "version": version_id}

    Enum-constrained path parameters:
        from enum import Enum

        class ModelName(str, Enum):
            bert    = "bert"
            roberta = "roberta"
            gpt2    = "gpt2"

        @app.post("/predict/{model_name}")
        async def predict(model_name: ModelName, request: PredictRequest):
            # model_name is validated against the enum values
            # Invalid values → 422 error automatically
            pass

### Query Parameters

    Optional with defaults:
        @app.post("/predict")
        async def predict(
            request:    PredictRequest,
            threshold:  float = Query(0.5, ge=0.0, le=1.0),
            top_k:      int   = Query(1,   ge=1,   le=100),
            format:     str   = Query("label", pattern="^(label|proba|logit)$"),
        ):
            ...

    Combining body + query + path:
        @app.post("/models/{model_name}/predict")
        async def predict(
            model_name: str,                          # path
            batch_size: int = Query(32),              # query
            request: BatchPredictRequest = Body(...), # body
            x_api_key: str = Header(...),             # header
        ):
            ...

### File Uploads for ML

    Image inference endpoint:
        from fastapi import UploadFile, File
        import io

        @app.post("/predict/image")
        async def predict_image(
            file:      UploadFile = File(...),
            threshold: float      = Query(0.5),
        ):
            if file.content_type not in ["image/jpeg", "image/png"]:
                raise HTTPException(400, "Only JPEG and PNG supported")

            image_bytes = await file.read()
            # Process image...
            image       = Image.open(io.BytesIO(image_bytes))
            prediction  = model.predict(preprocess(image))
            return {"prediction": prediction}

    Multiple file upload (batch image inference):
        @app.post("/predict/images/batch")
        async def predict_images(
            files: List[UploadFile] = File(...),
        ):
            images  = [Image.open(io.BytesIO(await f.read())) for f in files]
            results = [model.predict(preprocess(img)) for img in images]
            return {"predictions": results}


##### PART 4 — DEPENDENCY INJECTION: THE ML MODEL LIFECYCLE

### What Is Dependency Injection in FastAPI?

    FastAPI's Depends() system provides a clean way to:
        - Share expensive resources (models, DB pools, config)
        - Apply authentication checks to multiple endpoints
        - Inject request-scoped or application-scoped objects
        - Test endpoints by replacing real dependencies with mocks

    A dependency is just a Python callable (function or class) that
    FastAPI calls and injects into the endpoint function.

        from fastapi import Depends

        def get_model():
            return app.state.model   # returns the loaded model

        @app.post("/predict")
        async def predict(
            request: PredictRequest,
            model = Depends(get_model),   # model injected automatically
        ):
            return model.predict(request.features)

### Model Registry Dependency

    For multi-model serving, use a dependency that resolves the right model:

        class ModelRegistry:
            def __init__(self):
                self._models: Dict[str, Any] = {}

            def register(self, name: str, model: Any):
                self._models[name] = model

            def get(self, name: str) -> Any:
                if name not in self._models:
                    raise HTTPException(
                        status_code = 404,
                        detail      = f"Model '{name}' not found. "
                                      f"Available: {list(self._models.keys())}"
                    )
                return self._models[name]

        registry = ModelRegistry()

        def get_registry() -> ModelRegistry:
            return registry

        @app.post("/models/{model_name}/predict")
        async def predict(
            model_name: str,
            request:    PredictRequest,
            reg:        ModelRegistry = Depends(get_registry),
        ):
            model  = reg.get(model_name)
            result = model.predict(request.to_numpy())
            return {"prediction": result.tolist()}

### Authentication Dependency

    API key authentication:
        from fastapi.security import APIKeyHeader

        API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

        async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
            if api_key not in VALID_API_KEYS:
                raise HTTPException(
                    status_code = 401,
                    detail      = "Invalid API key",
                    headers     = {"WWW-Authenticate": "ApiKey"},
                )
            return api_key

        @app.post("/predict", dependencies=[Depends(verify_api_key)])
        async def predict(request: PredictRequest): ...

    OAuth2 / JWT Bearer token:
        from fastapi.security import OAuth2PasswordBearer
        import jwt

        oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

        async def get_current_user(token: str = Depends(oauth2_scheme)):
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            return payload["user_id"]

### Caching Dependency (lru_cache)

    For expensive-to-create dependencies:
        from functools import lru_cache

        @lru_cache(maxsize=1)
        def get_settings() -> Settings:
            return Settings()   # reads from env vars, .env file, etc.

        @app.get("/config")
        async def show_config(settings: Settings = Depends(get_settings)):
            return {"env": settings.environment, "debug": settings.debug}

        # get_settings() called once, cached forever (singleton)


##### PART 5 — ASYNC INFERENCE, STREAMING, AND WEBSOCKETS

### Async Inference Patterns

    The three inference patterns and their concurrency implications:

    PATTERN 1 — Sync function (FastAPI runs in thread pool):
        @app.post("/predict")
        def predict_sync(request: PredictRequest):
            # FastAPI detects this is NOT async and runs it in threadpool
            # Safe for CPU-bound inference — doesn't block event loop
            return model.predict(request.to_numpy())

    PATTERN 2 — Async with run_in_executor (for blocking code):
        import asyncio

        @app.post("/predict")
        async def predict_async(request: PredictRequest):
            loop = asyncio.get_event_loop()
            # run_in_executor runs model.predict in a thread pool
            # The event loop can serve other requests while waiting
            result = await loop.run_in_executor(
                None,               # None = default thread pool
                model.predict,
                request.to_numpy(),
            )
            return {"prediction": result.tolist()}

    PATTERN 3 — Truly async inference (e.g. calling an API):
        import httpx

        @app.post("/predict")
        async def predict_via_api(request: PredictRequest):
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "http://model-server:8501/v1/models/bert:predict",
                    json={"instances": request.features},
                )
            return resp.json()

### Streaming Responses (LLM Token Streaming)

    Server-Sent Events (SSE) for LLM streaming:

        from fastapi.responses import StreamingResponse
        import asyncio

        async def token_generator(prompt: str):
            tokens = generate_tokens(prompt)   # your LLM
            for token in tokens:
                # SSE format: "data: {json}\\n\\n"
                yield f"data: {json.dumps({'token': token})}\\n\\n"
                await asyncio.sleep(0)   # yield control to event loop
            yield f"data: {json.dumps({'done': True})}\\n\\n"

        @app.post("/generate")
        async def generate_stream(request: GenerateRequest):
            return StreamingResponse(
                token_generator(request.prompt),
                media_type = "text/event-stream",
                headers    = {
                    "Cache-Control":     "no-cache",
                    "X-Accel-Buffering": "no",  # disable Nginx buffering
                },
            )

    JavaScript client (fetch API):
        const response = await fetch("/generate", {...});
        const reader   = response.body.getReader();
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            const text = new TextDecoder().decode(value);
            // Parse SSE format and display token
        }

### WebSocket Endpoints

    For bidirectional streaming (real-time inference):

        from fastapi import WebSocket, WebSocketDisconnect

        @app.websocket("/ws/inference")
        async def websocket_inference(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    # Receive input from client
                    data    = await websocket.receive_json()
                    request = PredictRequest(**data)

                    # Run inference and stream result
                    result  = model.predict(request.to_numpy())
                    await websocket.send_json({
                        "prediction": result.tolist(),
                        "status":     "ok",
                    })
            except WebSocketDisconnect:
                # Client disconnected — clean up
                pass

    WebSockets are ideal for:
        - Real-time interactive applications
        - Streaming partial results as they become available
        - High-frequency inference (avoiding HTTP overhead per request)
        - Bidirectional communication (client can cancel requests)


##### PART 6 — MIDDLEWARE, ERROR HANDLING, AND OBSERVABILITY

### Middleware: Request/Response Processing Pipeline

    Middleware wraps every request-response cycle:

        import time
        from fastapi import Request
        from starlette.middleware.base import BaseHTTPMiddleware

        class TimingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                t0       = time.perf_counter()
                response = await call_next(request)
                latency  = (time.perf_counter() - t0) * 1000
                response.headers["X-Process-Time-Ms"] = f"{latency:.2f}"
                return response

        app.add_middleware(TimingMiddleware)

    Built-in middleware:
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware

        app.add_middleware(CORSMiddleware,
            allow_origins  = ["https://app.company.com"],
            allow_methods  = ["GET", "POST"],
            allow_headers  = ["*"],
        )
        app.add_middleware(GZipMiddleware, minimum_size=1000)

    Rate limiting middleware:
        from collections import defaultdict
        import asyncio

        class RateLimitMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, max_per_minute: int = 60):
                super().__init__(app)
                self.max = max_per_minute
                self.counts = defaultdict(list)

            async def dispatch(self, request: Request, call_next):
                ip  = request.client.host
                now = time.time()
                # Keep only requests from the last 60 seconds
                self.counts[ip] = [t for t in self.counts[ip] if now - t < 60]
                if len(self.counts[ip]) >= self.max:
                    return Response(
                        content  = "Rate limit exceeded",
                        status_code = 429,
                        headers  = {"Retry-After": "60"},
                    )
                self.counts[ip].append(now)
                return await call_next(request)

### Exception Handlers

    Custom exception types for ML services:
        class ModelNotFoundError(Exception):
            def __init__(self, model_name: str):
                self.model_name = model_name

        class InputValidationError(Exception):
            def __init__(self, field: str, message: str):
                self.field, self.message = field, message

        @app.exception_handler(ModelNotFoundError)
        async def model_not_found_handler(request, exc):
            return JSONResponse(
                status_code = 404,
                content     = {
                    "error":   "ModelNotFound",
                    "model":   exc.model_name,
                    "message": f"Model '{exc.model_name}' is not registered.",
                },
            )

    Handling Pydantic validation errors:
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse

        @app.exception_handler(RequestValidationError)
        async def validation_error_handler(request, exc):
            return JSONResponse(
                status_code = 422,
                content     = {
                    "error":   "ValidationError",
                    "details": exc.errors(),  # Pydantic error details
                    "body":    exc.body,
                },
            )

### Logging and Observability

    Structured JSON logging:
        import logging
        import json

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    "level":      record.levelname,
                    "time":       record.asctime,
                    "message":    record.getMessage(),
                    "module":     record.module,
                    "request_id": getattr(record, "request_id", None),
                })

    Request ID propagation (for distributed tracing):
        import uuid

        @app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

    Prometheus metrics:
        from prometheus_client import Counter, Histogram, generate_latest
        import time

        REQUEST_COUNT    = Counter("api_requests_total", "Total requests",
                                    ["method", "endpoint", "status"])
        REQUEST_LATENCY  = Histogram("api_request_latency_seconds",
                                      "Request latency", ["endpoint"])

        @app.get("/metrics")
        async def prometheus_metrics():
            return Response(generate_latest(), media_type="text/plain")


##### PART 7 — BACKGROUND TASKS AND BATCH PROCESSING

### Background Tasks: Post-Request Work

    Background tasks run AFTER the response is sent to the client.
    They run in the same process, in the same event loop.

    Use cases: logging, monitoring, async database writes, alerting.

        from fastapi import BackgroundTasks

        async def log_inference_to_db(
            request_id: str,
            features:   List[float],
            prediction: int,
            latency_ms: float,
        ):
            await db.execute(
                "INSERT INTO inference_log VALUES ($1, $2, $3, $4)",
                request_id, features, prediction, latency_ms
            )

        @app.post("/predict")
        async def predict(
            request:    PredictRequest,
            background: BackgroundTasks,
        ):
            t0         = time.perf_counter()
            prediction = model.predict(request.to_numpy())
            latency    = (time.perf_counter() - t0) * 1000

            # Schedule post-response work
            background.add_task(
                log_inference_to_db,
                request_id = str(uuid.uuid4()),
                features   = request.features,
                prediction = int(prediction),
                latency_ms = latency,
            )
            # Response sent to client BEFORE log_inference_to_db runs
            return {"prediction": int(prediction), "latency_ms": latency}

    Important: Background tasks run AFTER the response. If the server
    restarts before they complete, they are lost. For critical work
    (payment processing, important audit logs), use a proper queue
    (Celery + Redis/RabbitMQ, or cloud queues like SQS).

### Batch Inference Optimisation

    Adaptive batching: accumulate requests and process as a batch:

        import asyncio
        from collections import deque

        class BatchInferenceService:
            def __init__(self, model, max_batch=32, max_wait_ms=10):
                self.model        = model
                self.max_batch    = max_batch
                self.max_wait_ms  = max_wait_ms
                self.queue        = deque()
                self.processing   = False

            async def predict(self, features: List[float]) -> int:
                future = asyncio.get_event_loop().create_future()
                self.queue.append((features, future))

                if not self.processing:
                    asyncio.create_task(self._process_batch())

                return await future

            async def _process_batch(self):
                self.processing = True
                await asyncio.sleep(self.max_wait_ms / 1000)  # wait to accumulate

                batch, futures = [], []
                while self.queue and len(batch) < self.max_batch:
                    features, future = self.queue.popleft()
                    batch.append(features)
                    futures.append(future)

                # Single model call for the whole batch
                predictions = self.model.predict_batch(np.array(batch))

                for future, pred in zip(futures, predictions):
                    future.set_result(int(pred))

                self.processing = False


##### PART 8 — TESTING, PRODUCTION DEPLOYMENT, AND BEST PRACTICES

### Testing with TestClient

    FastAPI's TestClient is a synchronous test client wrapping httpx.
    It starts the app in the same process — no server needed.

        from fastapi.testclient import TestClient
        import pytest

        @pytest.fixture
        def client():
            with TestClient(app) as c:
                yield c   # lifespan events triggered

        def test_health(client):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

        def test_predict_valid(client):
            response = client.post("/predict",
                json={"features": [1.2, 3.4, 5.6, 7.8]})
            assert response.status_code == 200
            assert "prediction" in response.json()

        def test_predict_invalid(client):
            response = client.post("/predict",
                json={"features": "not_a_list"})
            assert response.status_code == 422  # Pydantic validation error

    Dependency overrides for testing (mock the model):
        def mock_model_dependency():
            return MockModel()   # deterministic fake model

        app.dependency_overrides[get_model] = mock_model_dependency

        with TestClient(app) as client:
            response = client.post("/predict", json={...})
            # Uses MockModel, not the real loaded model

### Health Checks and Kubernetes Probes

    Kubernetes expects two probes:
        Liveness:   Is the process running and not deadlocked?
        Readiness:  Is the model loaded and ready to serve traffic?

        @app.get("/health/live")
        async def liveness():
            return {"status": "alive"}   # just return 200 if process is up

        @app.get("/health/ready")
        async def readiness():
            model = getattr(app.state, "model", None)
            if model is None:
                raise HTTPException(503, "Model not loaded")
            # Optionally: run a tiny inference to confirm GPU is working
            try:
                _ = model.predict([[0.0] * n_features])
                return {"status": "ready", "model": "ok"}
            except Exception as e:
                raise HTTPException(503, f"Model inference failed: {e}")

### Production Configuration

    Environment-based settings with Pydantic Settings:
        from pydantic_settings import BaseSettings

        class Settings(BaseSettings):
            model_path:      str   = "models/fraud_detector.pkl"
            api_key:         str
            max_batch_size:  int   = 32
            timeout_seconds: float = 30.0
            environment:     str   = "production"
            debug:           bool  = False

            class Config:
                env_file = ".env"   # reads from .env file

    Docker deployment:
        FROM python:3.11-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt --no-cache-dir
        COPY . .
        EXPOSE 8000
        CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0",
             "--port", "8000", "--workers", "4"]

    Performance tuning:
        # Gunicorn with Uvicorn workers (production standard)
        gunicorn app.main:app \\
            --workers       $(( 2 * $(nproc) + 1 )) \\
            --worker-class  uvicorn.workers.UvicornWorker \\
            --timeout       120 \\
            --keep-alive    5 \\
            --log-level     info \\
            --access-logfile -

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Use Case                       │ FastAPI Pattern                     │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Single-model REST API          │ POST /predict + Pydantic I/O        │
    │ Multi-model serving            │ Registry dependency + path param    │
    │ Authentication                 │ API key or OAuth2 dependency        │
    │ LLM token streaming            │ StreamingResponse + SSE             │
    │ Real-time bidirectional        │ WebSocket endpoint                  │
    │ GPU batch throughput           │ Adaptive batching service           │
    │ Post-inference logging         │ BackgroundTasks                     │
    │ Rate limiting                  │ Middleware                          │
    │ Health/readiness probes        │ /health/live + /health/ready        │
    │ Testing without server         │ TestClient + dependency_overrides   │
    │ Production deployment          │ Gunicorn + UvicornWorker            │
    └──────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · FastAPI Core — Routing, Pydantic, and ML Endpoints": {
        "description": (
            "Core FastAPI patterns for ML inference APIs. "
            "App creation with metadata: title, version, docs_url. "
            "Pydantic request models with field validation for ML data. "
            "Response models with type-safe serialisation. "
            "Path parameters, query parameters, request body patterns. "
            "GET /health, GET /models, POST /predict endpoint design. "
            "Custom validators: feature range checks, shape validation. "
            "HTTP status codes: 200, 422, 404, 503 usage in ML context. "
            "Auto-documentation: OpenAPI schema from type hints. "
            "Lifespan events: model loading at startup. "
            "TestClient: testing endpoints without a running server. "
            "Multiple response types: JSON, plain text, streaming."
        ),
        "language": "python",
        "code": '''
import time
import math
import json
import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Query, Path, status
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field, field_validator, model_validator
    print(f"  FastAPI and Pydantic available")
    import fastapi, pydantic
    print(f"  FastAPI: {fastapi.__version__}  Pydantic: {pydantic.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "fastapi[all]", "httpx", "--quiet"], check=True)
    from fastapi import FastAPI, HTTPException, Query, Path, status
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field, field_validator, model_validator
    import fastapi, pydantic
    print(f"  FastAPI: {fastapi.__version__}  Pydantic: {pydantic.__version__}")

print("=" * 65)
print("  FASTAPI CORE — ROUTING, PYDANTIC, AND ML ENDPOINTS")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Pydantic models for ML request/response validation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Pydantic models: validation for ML data")
print("━" * 65)
print()

# ── Request models with rich validation ──────────────────────────────────
class PredictRequest(BaseModel):
    """Single-sample inference request."""
    features:   List[float]    = Field(..., min_length=1, max_length=10000,
                                        description="Feature vector")
    model_name: str            = Field("default", max_length=64)
    threshold:  float          = Field(0.5, ge=0.0, le=1.0,
                                        description="Classification threshold")
    return_proba: bool         = Field(False, description="Return probabilities")

    @field_validator("features")
    @classmethod
    def features_finite(cls, v: List[float]) -> List[float]:
        if any(not math.isfinite(x) for x in v):
            raise ValueError("All features must be finite (no NaN or Inf)")
        return v

    def to_numpy(self) -> np.ndarray:
        return np.array(self.features, dtype=np.float32).reshape(1, -1)


class BatchPredictRequest(BaseModel):
    """Batch inference request."""
    samples:    List[List[float]] = Field(..., min_length=1, max_length=512)
    model_name: str               = Field("default")

    @model_validator(mode="after")
    def uniform_feature_length(self) -> "BatchPredictRequest":
        lengths = {len(s) for s in self.samples}
        if len(lengths) > 1:
            raise ValueError(
                f"All samples must have the same number of features. "
                f"Found lengths: {sorted(lengths)}"
            )
        return self

    def to_numpy(self) -> np.ndarray:
        return np.array(self.samples, dtype=np.float32)


class PredictResponse(BaseModel):
    """Inference response."""
    prediction:  int
    probability: Optional[float] = None
    model_name:  str
    latency_ms:  float
    request_id:  str


class ModelInfo(BaseModel):
    """Model metadata."""
    name:         str
    version:      str
    n_features:   int
    n_classes:    int
    framework:    str
    loaded:       bool
    load_time_ms: float


class HealthResponse(BaseModel):
    status:       str
    n_models:     int
    uptime_sec:   float


# ── Demonstrate Pydantic validation ──────────────────────────────────────
print(f"  Pydantic model validation examples:")
print()

# Valid request
try:
    req = PredictRequest(features=[1.2, 3.4, 5.6], threshold=0.7)
    print(f"  Valid request:           {req.model_dump()}")
except Exception as e:
    print(f"  Unexpected error: {e}")

# NaN features
try:
    bad = PredictRequest(features=[1.2, float("nan"), 3.4])
    print(f"  NaN features: not caught")
except Exception as e:
    print(f"  NaN features:            ValidationError: {e.errors()[0]['msg']}")

# Invalid threshold
try:
    bad = PredictRequest(features=[1.0], threshold=1.5)
    print(f"  threshold=1.5: not caught")
except Exception as e:
    print(f"  threshold=1.5:           ValidationError: {e.errors()[0]['msg']}")

# Batch shape mismatch
try:
    bad = BatchPredictRequest(samples=[[1.0, 2.0], [3.0, 4.0, 5.0]])
except Exception as e:
    print(f"  Jagged batch:            ValidationError: {e.errors()[0]['msg']}")

# Valid batch
batch = BatchPredictRequest(samples=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"  Valid batch:             shape={batch.to_numpy().shape}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Building the FastAPI application
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — FastAPI app: lifespan, routes, and dependencies")
print("━" * 65)
print()

# ── Simulated ML model ───────────────────────────────────────────────────
class MockMLModel:
    """Simulates a real sklearn / PyTorch model for demo purposes."""
    def __init__(self, name: str, n_features: int, n_classes: int):
        self.name       = name
        self.n_features = n_features
        self.n_classes  = n_classes
        self.version    = "1.2.0"
        self.framework  = "sklearn"
        self.load_time_ms = rng.uniform(200, 800)
        self._weights   = rng.standard_normal((n_features, n_classes)).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Simulated softmax classifier
        logits = X @ self._weights
        exp_l  = np.exp(logits - logits.max(1, keepdims=True))
        probs  = exp_l / exp_l.sum(1, keepdims=True)
        return probs


# ── Application state ────────────────────────────────────────────────────
class AppState:
    models:     Dict[str, MockMLModel] = {}
    start_time: float = 0.0


_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up at shutdown."""
    _state.start_time = time.time()
    # Load models into memory (happens once at startup)
    _state.models["default"]   = MockMLModel("default",   n_features=10, n_classes=3)
    _state.models["fraud"]     = MockMLModel("fraud",     n_features=20, n_classes=2)
    _state.models["sentiment"] = MockMLModel("sentiment", n_features=768, n_classes=5)
    print(f"  [startup] Loaded {len(_state.models)} models")
    yield
    # Shutdown cleanup
    _state.models.clear()
    print(f"  [shutdown] Models unloaded")


# ── FastAPI application ──────────────────────────────────────────────────
app = FastAPI(
    title       = "ML Inference API",
    description = "Multi-model ML inference service with full validation",
    version     = "2.1.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(
        status    = "ok",
        n_models  = len(_state.models),
        uptime_sec = time.time() - _state.start_time if _state.start_time else 0,
    )


@app.get("/models", tags=["models"])
async def list_models() -> List[str]:
    return list(_state.models.keys())


@app.get("/models/{model_name}", response_model=ModelInfo, tags=["models"])
async def get_model_info(
    model_name: str = Path(..., description="Name of the model"),
):
    if model_name not in _state.models:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"Model '{model_name}' not found. "
                          f"Available: {list(_state.models.keys())}",
        )
    model = _state.models[model_name]
    return ModelInfo(
        name         = model.name,
        version      = model.version,
        n_features   = model.n_features,
        n_classes    = model.n_classes,
        framework    = model.framework,
        loaded       = True,
        load_time_ms = model.load_time_ms,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest):
    model_name = request.model_name
    if model_name not in _state.models:
        raise HTTPException(
            status_code = 404,
            detail      = f"Model '{model_name}' not found",
        )

    model = _state.models[model_name]

    # Validate feature count
    if len(request.features) != model.n_features:
        raise HTTPException(
            status_code = 422,
            detail      = f"Expected {model.n_features} features, "
                          f"got {len(request.features)}",
        )

    t0      = time.perf_counter()
    X       = request.to_numpy()
    probs   = model.predict(X)[0]
    latency = (time.perf_counter() - t0) * 1000

    pred_class = int(probs.argmax())
    pred_prob  = float(probs[pred_class])

    return PredictResponse(
        prediction  = pred_class if probs[pred_class] >= request.threshold else -1,
        probability = pred_prob if request.return_proba else None,
        model_name  = model_name,
        latency_ms  = round(latency, 3),
        request_id  = "demo-" + str(int(time.time() * 1000))[-8:],
    )


@app.post("/batch", tags=["inference"])
async def batch_predict(request: BatchPredictRequest) -> Dict:
    model_name = request.model_name
    if model_name not in _state.models:
        raise HTTPException(404, f"Model '{model_name}' not found")

    model = _state.models[model_name]
    X     = request.to_numpy()
    if X.shape[1] != model.n_features:
        raise HTTPException(
            422,
            f"Expected {model.n_features} features per sample, got {X.shape[1]}",
        )

    t0     = time.perf_counter()
    probs  = model.predict(X)
    t_ms   = (time.perf_counter() - t0) * 1000

    return {
        "predictions":  probs.argmax(axis=1).tolist(),
        "n_samples":    len(X),
        "model_name":   model_name,
        "latency_ms":   round(t_ms, 3),
    }


@app.get("/metrics", tags=["system"])
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    lines = [
        "# HELP ml_models_loaded Number of models currently loaded",
        "# TYPE ml_models_loaded gauge",
        f"ml_models_loaded {len(_state.models)}",
        "# HELP ml_api_uptime_seconds API uptime in seconds",
        "# TYPE ml_api_uptime_seconds counter",
        f"ml_api_uptime_seconds {time.time() - _state.start_time:.1f}",
    ]
    return PlainTextResponse("\\n".join(lines))


print(f"  FastAPI app created:")
print(f"    Endpoints:   {len([r for r in app.routes])}")
print(f"    Docs at:     /docs  (Swagger UI)")
print(f"    ReDoc at:    /redoc")
print(f"    OpenAPI at:  /openapi.json")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: TestClient — testing without a running server
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — TestClient: automated endpoint testing")
print("━" * 65)
print()

with TestClient(app) as client:
    test_results = []

    # Test 1: Health check
    resp = client.get("/health")
    test_results.append(("GET /health",    resp.status_code == 200,
                          resp.json()["status"]))

    # Test 2: List models
    resp = client.get("/models")
    test_results.append(("GET /models",    resp.status_code == 200,
                          f"{resp.json()}"))

    # Test 3: Get model info
    resp = client.get("/models/default")
    test_results.append(("GET /models/default", resp.status_code == 200,
                          f"n_feat={resp.json()['n_features']}"))

    # Test 4: Model not found
    resp = client.get("/models/nonexistent")
    test_results.append(("GET /models/nonexistent", resp.status_code == 404,
                          "404 expected"))

    # Test 5: Valid prediction
    resp = client.post("/predict", json={
        "features":   [rng.uniform(0, 1) for _ in range(10)],
        "model_name": "default",
        "return_proba": True,
    })
    test_results.append(("POST /predict (valid)", resp.status_code == 200,
                          f"pred={resp.json()['prediction']}"))

    # Test 6: Wrong feature count
    resp = client.post("/predict", json={
        "features":   [1.0, 2.0],    # default model needs 10 features
        "model_name": "default",
    })
    test_results.append(("POST /predict (wrong n_feat)", resp.status_code == 422,
                          "422 expected"))

    # Test 7: Invalid threshold
    resp = client.post("/predict", json={
        "features":   [1.0] * 10,
        "threshold":  2.0,            # > 1.0 is invalid
    })
    test_results.append(("POST /predict (bad threshold)", resp.status_code == 422,
                          "422 expected"))

    # Test 8: Batch prediction
    resp = client.post("/batch", json={
        "samples":   [[rng.uniform(0,1) for _ in range(10)] for _ in range(5)],
        "model_name": "default",
    })
    test_results.append(("POST /batch (5 samples)", resp.status_code == 200,
                          f"n={resp.json()['n_samples']}"))

    # Test 9: Prometheus metrics
    resp = client.get("/metrics")
    test_results.append(("GET /metrics", resp.status_code == 200,
                          "text/plain"))

    # Test 10: OpenAPI schema
    resp = client.get("/openapi.json")
    schema = resp.json()
    test_results.append(("GET /openapi.json", resp.status_code == 200,
                          f"{len(schema.get('paths', {}))} paths documented"))

print(f"  Test results:")
print(f"  {'Test':<40} {'Pass':>6} {'Result'}")
print(f"  {'─'*65}")
passed = 0
for name, ok, result in test_results:
    status_str = "✅" if ok else "❌"
    if ok: passed += 1
    print(f"  {name:<40} {status_str:>6} {str(result)[:22]}")
print(f"  {'─'*65}")
print(f"  {passed}/{len(test_results)} tests passed")
print()

# Inspect the OpenAPI schema
with TestClient(app) as client:
    schema = client.get("/openapi.json").json()
    paths  = schema.get("paths", {})
    print(f"  Auto-generated OpenAPI schema:")
    print(f"    title:    {schema['info']['title']}")
    print(f"    version:  {schema['info']['version']}")
    print(f"    paths:    {len(paths)}")
    for path, methods in sorted(paths.items()):
        method_list = [m.upper() for m in methods.keys() if m != "parameters"]
        print(f"      {' '.join(method_list):<8} {path}")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Dependency Injection — Model Registry, Auth & Caching": {
        "description": (
            "FastAPI dependency injection for ML production patterns. "
            "Depends() system: injectable callables and class dependencies. "
            "Model registry as a dependency: multi-model serving. "
            "API key authentication dependency with HTTPException. "
            "Request-scoped vs application-scoped dependencies. "
            "lru_cache for singleton resources: settings, heavy objects. "
            "Dependency override for testing: mock model injection. "
            "Nested dependencies: auth → user → permissions chain. "
            "Async dependencies: database connections, token validation. "
            "Dependency with cleanup: yield-based context dependencies. "
            "Rate limiting dependency using in-memory state. "
            "Benchmarking dependency overhead."
        ),
        "language": "python",
        "code": '''
import time
import math
import json
import asyncio
import numpy as np
from typing import List, Dict, Optional, Any
from functools import lru_cache
from collections import defaultdict
from contextlib import asynccontextmanager

try:
    from fastapi import (
        FastAPI, Depends, HTTPException, Header,
        Request, status, Query
    )
    from fastapi.testclient import TestClient
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field
    from pydantic_settings import BaseSettings
    import fastapi
    print(f"  FastAPI: {fastapi.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "fastapi[all]", "httpx", "pydantic-settings",
                    "--quiet"], check=True)
    from fastapi import (
        FastAPI, Depends, HTTPException, Header,
        Request, status, Query
    )
    from fastapi.testclient import TestClient
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field
    from pydantic_settings import BaseSettings
    import fastapi
    print(f"  FastAPI: {fastapi.__version__}")

print("=" * 65)
print("  DEPENDENCY INJECTION — MODEL REGISTRY, AUTH & CACHING")
print("=" * 65)
print()

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Model registry as a dependency
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Model registry dependency: multi-model serving")
print("━" * 65)
print()

class MockModel:
    """Simulated ML model."""
    def __init__(self, name, n_in, n_out):
        self.name = name; self.n_in = n_in; self.n_out = n_out
        self.W    = rng.standard_normal((n_in, n_out)).astype(np.float32) * 0.1
        self.call_count = 0
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.call_count += 1
        logits = X @ self.W
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


class ModelRegistry:
    """Central registry of all loaded models."""
    def __init__(self):
        self._store: Dict[str, MockModel] = {}
        self._load_times: Dict[str, float] = {}

    def load(self, name: str, n_in: int, n_out: int):
        t0 = time.perf_counter()
        self._store[name] = MockModel(name, n_in, n_out)
        self._load_times[name] = (time.perf_counter() - t0) * 1000

    def get(self, name: str) -> MockModel:
        if name not in self._store:
            raise HTTPException(
                status_code=404,
                detail={"error": "ModelNotFound", "name": name,
                        "available": list(self._store.keys())}
            )
        return self._store[name]

    def list_models(self) -> List[Dict]:
        return [
            {"name": k, "n_in": m.n_in, "n_out": m.n_out,
             "calls": m.call_count, "load_ms": self._load_times[k]}
            for k, m in self._store.items()
        ]


# Application-level registry (created once)
registry = ModelRegistry()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models at startup
    registry.load("bert-small",  n_in=256, n_out=5)
    registry.load("resnet-tiny", n_in=512, n_out=10)
    registry.load("gpt-nano",    n_in=128, n_out=50000)
    yield

# Dependency function — returns the shared registry
def get_registry() -> ModelRegistry:
    return registry


# Pydantic models
class InferRequest(BaseModel):
    features: List[float] = Field(..., min_length=1)

class InferResponse(BaseModel):
    model_name:  str
    prediction:  int
    probability: float
    latency_ms:  float
    call_count:  int


app2 = FastAPI(lifespan=lifespan, title="Multi-Model API")

@app2.get("/models")
def list_models(reg: ModelRegistry = Depends(get_registry)):
    return {"models": reg.list_models()}

@app2.post("/models/{model_name}/predict", response_model=InferResponse)
def predict(
    model_name: str,
    request:    InferRequest,
    reg:        ModelRegistry = Depends(get_registry),
):
    model = reg.get(model_name)
    if len(request.features) != model.n_in:
        raise HTTPException(422, f"Expected {model.n_in} features, "
                                 f"got {len(request.features)}")
    X      = np.array(request.features, np.float32).reshape(1, -1)
    t0     = time.perf_counter()
    probs  = model.predict(X)[0]
    t_ms   = (time.perf_counter() - t0) * 1000
    return InferResponse(
        model_name  = model_name,
        prediction  = int(probs.argmax()),
        probability = float(probs.max()),
        latency_ms  = t_ms,
        call_count  = model.call_count,
    )


with TestClient(app2) as client:
    print(f"  Multi-model serving tests:")
    print()

    # List models
    r = client.get("/models")
    models = r.json()["models"]
    print(f"  Available models:")
    print(f"  {'Name':<15} {'n_in':>6} {'n_out':>7} {'load_ms':>10}")
    print(f"  {'─'*42}")
    for m in models:
        print(f"  {m['name']:<15} {m['n_in']:>6} {m['n_out']:>7} {m['load_ms']:>10.3f}")
    print()

    # Predict on each model
    print(f"  Inference on each model:")
    for m in models:
        features = [rng.uniform(-1, 1) for _ in range(m["n_in"])]
        r = client.post(
            f"/models/{m['name']}/predict",
            json={"features": features},
        )
        data = r.json()
        print(f"    {m['name']:<15} pred={data['prediction']:>6} "
              f"prob={data['probability']:.4f} "
              f"lat={data['latency_ms']:.3f}ms")

    # Wrong feature count
    r = client.post("/models/bert-small/predict",
                    json={"features": [1.0, 2.0]})
    print(f"\n  Wrong n_features: status={r.status_code} ✅")

    # Model not found
    r = client.post("/models/nonexistent/predict",
                    json={"features": [1.0] * 10})
    print(f"  Model not found:  status={r.status_code} ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Authentication dependency
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — API key authentication dependency")
print("━" * 65)
print()

VALID_KEYS = {"sk-prod-abc123", "sk-dev-xyz789", "sk-test-000000"}
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)):
    """Dependency: validates API key from X-API-Key header."""
    if api_key is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Missing API key. Provide X-API-Key header.",
        )
    if api_key not in VALID_KEYS:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Invalid API key.",
        )
    # Extract tier from key prefix
    tier = "prod" if api_key.startswith("sk-prod") else "dev"
    return {"key": api_key, "tier": tier}


app3 = FastAPI(title="Auth Demo")

@app3.post("/secure/predict")
async def secure_predict(
    auth: Dict = Depends(verify_api_key),
    request: InferRequest = ...,
):
    return {
        "tier":       auth["tier"],
        "prediction": 0,
        "authorized": True,
    }

@app3.get("/public/health")
async def public_health():
    return {"status": "ok"}   # no auth needed


with TestClient(app3) as client:
    print(f"  Authentication tests:")

    # No key
    r = client.post("/secure/predict", json={"features": [1.0, 2.0]})
    print(f"    No key:           {r.status_code} {r.json().get('detail','')[:40]}")

    # Wrong key
    r = client.post("/secure/predict",
                    json={"features": [1.0, 2.0]},
                    headers={"X-API-Key": "bad-key"})
    print(f"    Invalid key:      {r.status_code} {r.json().get('detail','')[:30]}")

    # Valid prod key
    r = client.post("/secure/predict",
                    json={"features": [1.0, 2.0]},
                    headers={"X-API-Key": "sk-prod-abc123"})
    print(f"    Valid prod key:   {r.status_code} tier={r.json().get('tier','?')}")

    # Valid dev key
    r = client.post("/secure/predict",
                    json={"features": [1.0, 2.0]},
                    headers={"X-API-Key": "sk-dev-xyz789"})
    print(f"    Valid dev key:    {r.status_code} tier={r.json().get('tier','?')}")

    # Public endpoint — no key needed
    r = client.get("/public/health")
    print(f"    Public endpoint:  {r.status_code} {r.json()['status']}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Settings dependency with lru_cache
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — lru_cache settings: singleton dependency pattern")
print("━" * 65)
print()

class AppSettings(BaseSettings):
    app_name:       str   = "ML API"
    environment:    str   = "development"
    max_batch_size: int   = 32
    timeout_ms:     float = 5000.0
    debug:          bool  = False
    model_path:     str   = "models/"

    class Config:
        env_prefix = "ML_"    # reads ML_APP_NAME, ML_ENVIRONMENT, etc.


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Returns the same Settings instance on every call (singleton)."""
    return AppSettings()


app4 = FastAPI()

@app4.get("/settings")
def show_settings(settings: AppSettings = Depends(get_settings)):
    return settings.model_dump()

@app4.post("/predict/limited")
def predict_limited(
    request:  InferRequest,
    settings: AppSettings = Depends(get_settings),
):
    if len(request.features) > settings.max_batch_size * 100:
        raise HTTPException(
            status_code = 413,
            detail      = f"Request too large. Max features: "
                          f"{settings.max_batch_size * 100}",
        )
    return {"received": len(request.features), "env": settings.environment}


with TestClient(app4) as client:
    r = client.get("/settings")
    settings = r.json()
    print(f"  Settings (loaded from environment / defaults):")
    for k, v in settings.items():
        print(f"    {k:<20}: {v}")
    print()

    # Verify singleton — same object returned each time
    s1 = get_settings()
    s2 = get_settings()
    print(f"  lru_cache singleton: s1 is s2 → {s1 is s2}  "
          f"(called twice, created once)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Dependency override for testing (mock model)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Dependency overrides: mock injection for testing")
print("━" * 65)
print()

# Original dependency
def get_real_model():
    return registry.get("bert-small")   # loads actual model

# Test override — deterministic mock
class DeterministicModel:
    """Always predicts class 0 with probability 0.99. For testing."""
    n_in = 256
    call_count = 0
    def predict(self, X):
        self.call_count += 1
        out = np.zeros((len(X), 5), dtype=np.float32)
        out[:, 0] = 0.99
        out[:, 1:] = 0.0025
        return out


app5 = FastAPI(lifespan=lifespan)

@app5.post("/predict")
def predict_with_dep(
    request: InferRequest,
    model = Depends(get_real_model),
):
    X     = np.array(request.features[:model.n_in], np.float32).reshape(1,-1)
    probs = model.predict(X)[0]
    return {"prediction": int(probs.argmax()), "probability": float(probs.max())}


# Test with real model
mock_instance = DeterministicModel()
app5.dependency_overrides[get_real_model] = lambda: mock_instance

with TestClient(app5) as client:
    results = []
    for _ in range(5):
        r = client.post("/predict", json={"features": [rng.uniform() for _ in range(256)]})
        results.append(r.json()["prediction"])

    print(f"  With DeterministicModel override:")
    print(f"    Predictions (should all be 0): {results}")
    print(f"    All == 0: {all(p == 0 for p in results)} ✅")
    print(f"    Mock call count: {mock_instance.call_count}")
    print()

# Remove override and test with real model
app5.dependency_overrides.clear()
with TestClient(app5) as client:
    r = client.post("/predict",
                    json={"features": [rng.uniform() for _ in range(256)]})
    print(f"  With real model (override cleared):")
    print(f"    Prediction: {r.json()['prediction']} (can be any class)")
    print(f"    Status: {r.status_code} ✅")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Middleware, Error Handling & Streaming": {
        "description": (
            "FastAPI middleware and error handling for production ML APIs. "
            "Custom middleware: timing, request ID injection. "
            "CORS middleware: cross-origin resource sharing configuration. "
            "Rate limiting middleware: per-IP request counting. "
            "Custom exception handlers: ModelNotFoundError, ValidationError. "
            "HTTPException with custom detail and headers. "
            "StreamingResponse: LLM token streaming with SSE. "
            "Background tasks: post-request logging and monitoring. "
            "Request/response logging middleware with structured JSON. "
            "Exception handler for all unhandled errors. "
            "Latency percentile tracking with middleware. "
            "Testing streaming responses and error handlers."
        ),
        "language": "python",
        "code": '''
import time
import uuid
import math
import asyncio
import json
import numpy as np
from typing import List, Dict, Optional, AsyncGenerator
from collections import defaultdict, deque
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from pydantic import BaseModel, Field
    import fastapi
    print(f"  FastAPI: {fastapi.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "fastapi[all]", "httpx", "--quiet"], check=True)
    from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from pydantic import BaseModel, Field
    import fastapi
    print(f"  FastAPI: {fastapi.__version__}")

print("=" * 65)
print("  MIDDLEWARE, ERROR HANDLING & STREAMING")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Custom middleware
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Middleware: timing, request IDs, and rate limiting")
print("━" * 65)
print()

# Shared latency history for demonstration
latency_history: List[float] = []
request_log:     List[Dict]  = []
ip_request_counts = defaultdict(deque)  # IP → deque of timestamps


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects a unique request ID into every request and response."""
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Adds X-Process-Time-Ms header and records latency history."""
    async def dispatch(self, request: Request, call_next):
        t0       = time.perf_counter()
        response = await call_next(request)
        latency  = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{latency:.2f}"
        latency_history.append(latency)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs structured JSON for every request."""
    async def dispatch(self, request: Request, call_next):
        t0       = time.perf_counter()
        response = await call_next(request)
        latency  = (time.perf_counter() - t0) * 1000
        request_log.append({
            "method":     request.method,
            "path":       request.url.path,
            "status":     response.status_code,
            "latency_ms": round(latency, 2),
            "client_ip":  request.client.host if request.client else "unknown",
            "request_id": getattr(request.state, "request_id", "?"),
        })
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple token-bucket rate limiter: max N requests per minute per IP."""
    def __init__(self, app, max_per_minute: int = 100):
        super().__init__(app)
        self.max = max_per_minute

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        ip  = request.client.host if request.client else "unknown"
        now = time.time()
        # Prune requests older than 60 seconds
        while ip_request_counts[ip] and now - ip_request_counts[ip][0] > 60:
            ip_request_counts[ip].popleft()

        if len(ip_request_counts[ip]) >= self.max:
            return JSONResponse(
                status_code = 429,
                content     = {
                    "error":       "RateLimitExceeded",
                    "limit":       self.max,
                    "window_sec":  60,
                    "retry_after": 60,
                },
                headers     = {"Retry-After": "60"},
            )
        ip_request_counts[ip].append(now)
        return await call_next(request)


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Custom exception handlers
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Custom exception handlers")
print("━" * 65)
print()

# Custom exception types
class ModelError(Exception):
    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason     = reason

class InferenceFailed(Exception):
    def __init__(self, details: str):
        self.details = details


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Background tasks
# ─────────────────────────────────────────────────────────────────────────
inference_audit: List[Dict] = []

async def audit_log_inference(
    request_id:  str,
    model_name:  str,
    n_features:  int,
    prediction:  int,
    latency_ms:  float,
    client_ip:   str,
):
    """Background task: records every inference for compliance/monitoring."""
    await asyncio.sleep(0)   # yield control
    inference_audit.append({
        "request_id":  request_id,
        "model":       model_name,
        "n_features":  n_features,
        "prediction":  prediction,
        "latency_ms":  latency_ms,
        "client_ip":   client_ip,
        "timestamp":   time.time(),
    })


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Full app with all middleware and error handling
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Complete app: middleware stack + streaming + BG tasks")
print("━" * 65)
print()

class MockModelForMiddleware:
    call_count = 0
    n_in = 8

    def predict(self, X):
        self.call_count += 1
        e = np.exp(X @ rng.standard_normal((self.n_in, 3)) * 0.1)
        return e / e.sum(1, keepdims=True)


_model = MockModelForMiddleware()


app_full = FastAPI(title="Production-Ready ML API", version="3.0.0")

# Register middleware (order matters: first added = outermost)
app_full.add_middleware(RequestIDMiddleware)
app_full.add_middleware(TimingMiddleware)
app_full.add_middleware(RequestLoggingMiddleware)
app_full.add_middleware(RateLimitMiddleware, max_per_minute=5)   # low for demo
app_full.add_middleware(
    CORSMiddleware,
    allow_origins  = ["http://localhost:3000", "https://app.example.com"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["Content-Type", "X-API-Key", "X-Request-ID"],
)


# Exception handlers
@app_full.exception_handler(ModelError)
async def model_error_handler(request: Request, exc: ModelError):
    return JSONResponse(
        status_code = 503,
        content     = {
            "error":      "ModelError",
            "model_name": exc.model_name,
            "reason":     exc.reason,
        },
    )

@app_full.exception_handler(InferenceFailed)
async def inference_error_handler(request: Request, exc: InferenceFailed):
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":   "InferenceFailed",
            "details": exc.details,
        },
    )

@app_full.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":   "InternalServerError",
            "message": str(exc)[:100],
        },
    )


# Endpoints
class SimpleRequest(BaseModel):
    features: List[float] = Field(..., min_length=1, max_length=8)

@app_full.get("/health")
async def health():
    return {"status": "ok", "model_calls": _model.call_count}

@app_full.post("/predict")
async def predict_with_bg(
    request:    SimpleRequest,
    background: BackgroundTasks,
    req:        Request = None,
):
    n = len(request.features)
    features_padded = request.features[:_model.n_in] + [0.0] * max(0, _model.n_in - n)
    X = np.array(features_padded, np.float32).reshape(1, -1)

    t0     = time.perf_counter()
    probs  = _model.predict(X)[0]
    t_ms   = (time.perf_counter() - t0) * 1000
    pred   = int(probs.argmax())

    background.add_task(
        audit_log_inference,
        request_id = getattr(req.state, "request_id", "?") if req else "?",
        model_name = "default",
        n_features = n,
        prediction = pred,
        latency_ms = t_ms,
        client_ip  = req.client.host if req and req.client else "test",
    )
    return {"prediction": pred, "latency_ms": round(t_ms, 3)}

@app_full.get("/predict/error")
async def trigger_model_error():
    raise ModelError(model_name="bert", reason="GPU OOM on allocation")

@app_full.get("/predict/crash")
async def trigger_generic_error():
    raise RuntimeError("Unexpected model state — weights corrupted")


# Streaming endpoint (LLM-style token streaming)
@app_full.post("/generate")
async def generate_stream(request: SimpleRequest):
    async def token_stream() -> AsyncGenerator[str, None]:
        words = ["The", "quick", "brown", "fox", "jumps", "over",
                 "the", "lazy", "dog", "END"]
        for i, word in enumerate(words):
            await asyncio.sleep(0.005)    # simulate token generation delay
            payload = json.dumps({"token": word, "index": i,
                                   "done": word == "END"})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        token_stream(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache",
                      "X-Accel-Buffering": "no"},
    )

@app_full.get("/audit")
async def get_audit_log():
    return {"total": len(inference_audit), "recent": inference_audit[-5:]}

@app_full.get("/latencies")
async def get_latency_stats():
    if not latency_history:
        return {"count": 0}
    arr = sorted(latency_history)
    return {
        "count": len(arr),
        "p50":   arr[len(arr)//2],
        "p95":   arr[int(len(arr)*0.95)],
        "p99":   arr[int(len(arr)*0.99)],
        "mean":  sum(arr)/len(arr),
    }


# ── Run tests ─────────────────────────────────────────────────────────────
with TestClient(app_full) as client:

    # Make several requests to build up history
    test_cases = []
    for i in range(4):
        features = [rng.uniform(-1,1) for _ in range(rng.integers(3, 9))]
        r = client.post("/predict", json={"features": features},
                        headers={"X-Request-ID": f"req-{i:04d}"})
        test_cases.append(r)

    print(f"  Made 4 predict requests:")
    for i, r in enumerate(test_cases):
        lat = r.headers.get("X-Process-Time-Ms", "?")
        rid = r.headers.get("X-Request-ID", "?")
        print(f"    req-{i}: status={r.status_code} "
              f"pred={r.json().get('prediction','?')} "
              f"latency={lat}ms  request_id={rid}")

    # Test rate limiting (5 requests/min limit, already used 4)
    r5 = client.post("/predict", json={"features": [1.0]*5})
    r6 = client.post("/predict", json={"features": [1.0]*5})  # should be 429
    print(f"\n  Rate limit (max 5/min):")
    print(f"    5th request: {r5.status_code}")
    print(f"    6th request: {r6.status_code} "
          f"{'(rate limited ✅)' if r6.status_code == 429 else '(not limited)'}")

    # Test exception handlers
    r_err1 = client.get("/predict/error")
    r_err2 = client.get("/predict/crash")
    print(f"\n  Exception handlers:")
    print(f"    ModelError:  {r_err1.status_code} "
          f"error={r_err1.json().get('error','?')}")
    print(f"    RuntimeError: {r_err2.status_code} "
          f"error={r_err2.json().get('error','?')}")

    # Test streaming
    import time as _t
    t_stream = _t.perf_counter()
    with client.stream("POST", "/generate",
                       json={"features": [1.0]*4}) as stream:
        chunks = list(stream.iter_lines())
    t_stream = (_t.perf_counter() - t_stream) * 1000
    data_chunks = [c for c in chunks if c.startswith("data:")]
    tokens = [json.loads(c[6:])["token"] for c in data_chunks if c]
    print(f"\n  Streaming response:")
    print(f"    Tokens received:  {len(tokens)}")
    print(f"    Tokens:           {tokens[:5]}...")
    print(f"    Total time:       {t_stream:.0f}ms")

    # Latency stats
    lat_stats = client.get("/latencies").json()
    print(f"\n  Request latency stats:")
    for k, v in lat_stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.3f}ms")
        else:
            print(f"    {k}: {v}")

    # Request log
    print(f"\n  Request log ({len(request_log)} entries):")
    print(f"  {'Method':<8} {'Path':<20} {'Status':>7} {'Latency':>10}")
    print(f"  {'─'*50}")
    for entry in request_log[-6:]:
        print(f"  {entry['method']:<8} {entry['path']:<20} "
              f"{entry['status']:>7} {entry['latency_ms']:>9.2f}ms")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Production Patterns — Benchmarking, Testing & Deployment": {
        "description": (
            "Production-grade FastAPI ML service patterns. "
            "Adaptive batch inference: accumulate requests for GPU efficiency. "
            "Throughput benchmark: requests/sec at varying concurrency. "
            "pytest patterns: fixtures, parametrize, dependency overrides. "
            "Health check endpoints: liveness vs readiness probes. "
            "Graceful shutdown: finishing in-flight requests. "
            "Docker and Gunicorn configuration for production. "
            "OpenAPI schema introspection: endpoint documentation. "
            "Request size limits and timeout middleware. "
            "Multi-worker concurrency: threading model explained. "
            "Complete ML serving architecture comparison. "
            "FastAPI vs Flask vs Django REST performance."
        ),
        "language": "python",
        "code": '''
import time
import math
import asyncio
import threading
import numpy as np
import json
import concurrent.futures
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from collections import deque

try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import fastapi
    print(f"  FastAPI: {fastapi.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "fastapi[all]", "httpx", "--quiet"], check=True)
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import fastapi
    print(f"  FastAPI: {fastapi.__version__}")

print("=" * 65)
print("  PRODUCTION PATTERNS — BENCHMARKING, TESTING & DEPLOYMENT")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Adaptive batch inference service
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Adaptive batching: maximise GPU utilisation")
print("━" * 65)
print()

print(f"  Adaptive batching concept:")
print(f"    Problem: GPU processes large batches far more efficiently")
print(f"             than individual samples (10ms for 1 vs 12ms for 32)")
print(f"    Solution: accumulate requests for up to max_wait_ms,")
print(f"              then process as a single batch")
print()
print(f"  Implementation:")
print(f"    1. Each request adds itself to an async queue")
print(f"    2. A background task waits max_wait_ms for more requests")
print(f"    3. After the wait, dispatches the accumulated batch")
print(f"    4. Each waiting coroutine receives its individual result")
print()

# Simulate the throughput improvement from batching
def simulate_batched_inference(
    n_requests: int,
    model_latency_base_ms: float = 5.0,
    model_latency_per_sample_ms: float = 0.5,
    max_batch: int = 32,
    max_wait_ms: float = 10.0,
):
    """
    Compare individual vs adaptive-batched inference throughput.
    Returns (individual_throughput, batched_throughput) in req/s.
    """
    # Individual: each request pays base latency
    total_time_individual = n_requests * (model_latency_base_ms +
                                          model_latency_per_sample_ms) / 1000
    throughput_individual = n_requests / total_time_individual

    # Batched: requests accumulate, pay base once per batch
    n_batches  = math.ceil(n_requests / max_batch)
    batch_time = (model_latency_base_ms +
                  min(max_batch, n_requests / n_batches) * model_latency_per_sample_ms)
    total_time_batched = n_batches * (batch_time + max_wait_ms) / 1000
    throughput_batched = n_requests / total_time_batched

    return throughput_individual, throughput_batched

print(f"  Throughput analysis (100 requests):")
print(f"  {'Batch size':>12} {'Individual req/s':>18} {'Batched req/s':>16} "
      f"{'Speedup':>10}")
print(f"  {'─'*60}")
for batch_sz in [1, 8, 16, 32, 64]:
    ind_thr, bat_thr = simulate_batched_inference(
        100, model_latency_base_ms=5, model_latency_per_sample_ms=0.5,
        max_batch=batch_sz, max_wait_ms=10,
    )
    speedup = bat_thr / ind_thr
    print(f"  {batch_sz:>12} {ind_thr:>18.0f} {bat_thr:>16.0f} {speedup:>10.2f}×")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Throughput benchmarking with TestClient
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Throughput benchmark: requests/sec measurement")
print("━" * 65)
print()

# Build a minimal but realistic inference app
class InfRequest(BaseModel):
    features: List[float] = Field(..., min_length=4, max_length=128)

class MockFastModel:
    n_in = 16
    W    = rng.standard_normal((16, 4)).astype(np.float32) * 0.1
    def predict(self, X):
        e = np.exp(X @ self.W)
        return e / e.sum(1, keepdims=True)

_bench_model = MockFastModel()
app_bench    = FastAPI()

@app_bench.get("/ping")
def ping(): return "pong"

@app_bench.post("/infer")
def infer(request: InfRequest):
    features = (request.features * (_bench_model.n_in // len(request.features) + 1)
               )[:_bench_model.n_in]
    X = np.array(features, np.float32).reshape(1, -1)
    p = _bench_model.predict(X)[0]
    return {"prediction": int(p.argmax()), "probability": float(p.max())}

with TestClient(app_bench) as bench_client:
    # Warmup
    for _ in range(20):
        bench_client.get("/ping")
        bench_client.post("/infer", json={"features": [1.0]*16})

    # Benchmark /ping (minimal overhead)
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        bench_client.get("/ping")
    t_ping = (time.perf_counter() - t0) / N * 1000

    # Benchmark /infer with different input sizes
    print(f"  TestClient benchmark ({N} requests each):")
    print(f"  {'Endpoint':<25} {'Features':>10} {'Latency ms':>12} {'Req/s':>10}")
    print(f"  {'─'*60}")

    print(f"  {'GET /ping':<25} {'—':>10} {t_ping:>12.3f} "
          f"{1000/t_ping:>10.0f}")

    for n_feat in [4, 16, 64, 128]:
        features = [rng.uniform(-1, 1) for _ in range(n_feat)]
        times    = []
        for _ in range(N):
            t0 = time.perf_counter()
            r  = bench_client.post("/infer", json={"features": features})
            times.append((time.perf_counter() - t0) * 1000)
        p50 = sorted(times)[N//2]
        print(f"  {'POST /infer':<25} {n_feat:>10} {p50:>12.3f} {1000/p50:>10.0f}")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: pytest patterns
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — pytest patterns for ML APIs")
print("━" * 65)
print()

# Simulate what pytest tests would look like
test_app = app_bench

class TestMLAPI:
    """Simulates a pytest test class."""

    def setup_method(self):
        self.client = TestClient(test_app)

    def test_health(self):
        r = self.client.get("/ping")
        assert r.status_code == 200
        assert r.text == '"pong"'
        return True

    def test_predict_valid(self):
        r = self.client.post("/infer", json={"features": [1.0]*16})
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert "probability" in data
        assert 0 <= data["prediction"] <= 3
        assert 0.0 <= data["probability"] <= 1.0
        return True

    def test_predict_too_few_features(self):
        r = self.client.post("/infer", json={"features": [1.0, 2.0]})
        assert r.status_code == 422   # Pydantic: min_length=4 violated
        return True

    def test_predict_too_many_features(self):
        r = self.client.post("/infer", json={"features": [1.0]*200})
        assert r.status_code == 422   # Pydantic: max_length=128 violated
        return True

    def test_predict_non_numeric(self):
        r = self.client.post("/infer", json={"features": ["a", "b", "c", "d"]})
        assert r.status_code == 422
        return True

    def test_response_schema(self):
        r = self.client.post("/infer", json={"features": [1.0]*16})
        data = r.json()
        # All required fields present
        assert set(data.keys()) >= {"prediction", "probability"}
        # Correct types
        assert isinstance(data["prediction"], int)
        assert isinstance(data["probability"], float)
        return True

    def test_deterministic(self):
        """Same input should always give same output."""
        features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                    0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        r1 = self.client.post("/infer", json={"features": features})
        r2 = self.client.post("/infer", json={"features": features})
        assert r1.json()["prediction"] == r2.json()["prediction"]
        return True

    def test_parametrize(self):
        """Tests multiple valid feature lengths."""
        for n_feat in [4, 8, 16, 32, 64, 100, 128]:
            features = [1.0] * n_feat
            r = self.client.post("/infer", json={"features": features})
            assert r.status_code == 200, f"Failed for n_feat={n_feat}"
        return True


# Run all tests
suite = TestMLAPI()
suite.setup_method()

tests = [
    ("test_health",                 suite.test_health),
    ("test_predict_valid",          suite.test_predict_valid),
    ("test_predict_too_few",        suite.test_predict_too_few_features),
    ("test_predict_too_many",       suite.test_predict_too_many_features),
    ("test_predict_non_numeric",    suite.test_predict_non_numeric),
    ("test_response_schema",        suite.test_response_schema),
    ("test_deterministic",          suite.test_deterministic),
    ("test_parametrize (7 lengths)",suite.test_parametrize),
]

print(f"  Running test suite:")
passed = 0
for name, test_fn in tests:
    t0 = time.perf_counter()
    try:
        result = test_fn()
        t_ms   = (time.perf_counter() - t0) * 1000
        print(f"    ✅  {name:<40} {t_ms:.1f}ms")
        passed += 1
    except AssertionError as e:
        print(f"    ❌  {name:<40} FAILED: {e}")
    except Exception as e:
        print(f"    💥  {name:<40} ERROR: {e}")

print(f"\n  Results: {passed}/{len(tests)} tests passed")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Production deployment reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Production deployment guide")
print("━" * 65)
print()

print(f"  Deployment configurations:")
print()
DEPLOY_CONFIGS = {
    "Development": {
        "cmd": "uvicorn app.main:app --reload --port 8000",
        "workers": 1, "notes": "Hot-reload, single process",
    },
    "Single server": {
        "cmd": "uvicorn app.main:app --host 0.0.0.0 --workers 4",
        "workers": 4, "notes": "4 processes, recommended: 2×cores+1",
    },
    "Gunicorn (prod)": {
        "cmd": "gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker",
        "workers": 4, "notes": "Process manager + ASGI workers",
    },
    "Docker": {
        "cmd": "docker run -p 8000:8000 ml-api:v2.1",
        "workers": 4, "notes": "Containerised, orchestrated by K8s",
    },
}
for env, cfg in DEPLOY_CONFIGS.items():
    print(f"  [{env}]")
    print(f"    Command: {cfg['cmd']}")
    print(f"    Notes:   {cfg['notes']}")
    print()

print(f"  Framework performance comparison (approximate, CPU inference):")
print(f"  {'Framework':<20} {'Req/s':>10} {'Latency':>12} {'Async':>8} {'Auto-docs'}")
print(f"  {'─'*60}")
for name, rps, lat, asyn, docs in [
    ("FastAPI + Uvicorn", "~50,000", "sub-ms",   "✓", "✓"),
    ("Flask + Gunicorn",  "~10,000", "1-5ms",    "✗", "✗"),
    ("Django REST",       "~8,000",  "2-10ms",   "✗", "✗"),
    ("aiohttp",           "~40,000", "sub-ms",   "✓", "✗"),
    ("Tornado",           "~15,000", "sub-ms",   "✓", "✗"),
]:
    print(f"  {name:<20} {rps:>10} {lat:>12} {asyn:>8} {docs}")
print()

print(f"  FastAPI production checklist:")
items = [
    ("Lifespan events",      "Load model once at startup, not per-request"),
    ("Pydantic validation",  "Define strict schemas with Field constraints"),
    ("Dependency injection", "Use Depends() for shared resources"),
    ("Async endpoints",      "Use async def; run_in_executor for blocking code"),
    ("Exception handlers",   "Register custom handlers, not bare 500 errors"),
    ("Health probes",        "/health/live and /health/ready for K8s"),
    ("Request IDs",          "Inject X-Request-ID for distributed tracing"),
    ("Structured logging",   "JSON logs with request_id, latency, status"),
    ("Rate limiting",        "Protect against abuse and runaway clients"),
    ("Timeout middleware",   "Kill slow requests before they pile up"),
    ("Gunicorn workers",     "Multiple processes to bypass Python GIL"),
    ("OOM protection",       "Limit request body size with max_length"),
]
print(f"  {'Item':<25} {'Description'}")
print(f"  {'─'*70}")
for item, desc in items:
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