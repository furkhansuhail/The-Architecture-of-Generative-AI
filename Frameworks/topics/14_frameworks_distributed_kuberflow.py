"""
Kubeflow — The MLOps Platform for Kubernetes
=============================================

Kubeflow is an open-source machine learning platform designed to make
deploying, orchestrating, and managing machine learning workloads on
Kubernetes simple, portable, and scalable. It was originally created by
Google engineers in 2017 as an internal project to run TensorFlow jobs
on Kubernetes, and was open-sourced in December 2017 at KubeCon Austin.

The founding insight: Kubernetes solves the operational challenges of
running containerised workloads at scale — service discovery, health
checking, autoscaling, rolling deployments, resource scheduling. Machine
learning has the same operational challenges, plus the additional complexity
of distributed training, hyperparameter tuning, experiment tracking, and
model serving. Kubeflow extends Kubernetes natively to handle all of this
under one unified platform.

By 2024, Kubeflow is the industry-standard MLOps platform for organisations
running ML at scale on Kubernetes — used at Google, Bloomberg, IBM, Cisco,
Canonical, and thousands of other enterprises. It underpins Google Cloud's
Vertex AI, and the patterns it established (pipeline DSL, training operators,
model registry) have been adopted across the ML infrastructure landscape.

Kubeflow is structured as a collection of loosely coupled, independently
deployable components, each solving a specific ML lifecycle stage:

    Kubeflow Pipelines (KFP):
        Orchestrate multi-step ML workflows as directed acyclic graphs (DAGs).
        Each step is a containerised function. Steps can run on CPUs, GPUs,
        or specialised hardware. Pipelines are version-controlled, reproducible,
        and can be triggered on a schedule or by events.

    Kubeflow Training Operator:
        Run distributed training jobs on Kubernetes using TFJob (TensorFlow),
        PyTorchJob, MPIJob, PaddleJob, and XGBoostJob custom resources.
        Manages worker/parameter-server topologies, failure recovery,
        and GPU scheduling.

    Katib:
        Automated hyperparameter tuning and Neural Architecture Search (NAS)
        as a Kubernetes-native service. Supports Bayesian, grid, random,
        Hyperband, and evolutionary search algorithms.

    KServe (formerly KFServing):
        Production model serving with canary deployments, A/B testing,
        autoscaling, multi-model serving, and explainability hooks.
        Built on Knative Serving and Istio for traffic management.

    Kubeflow Notebooks:
        Managed JupyterLab environments running as Kubernetes pods,
        with persistent storage, GPU access, and direct cluster API access.

    Central Dashboard:
        Unified UI connecting all Kubeflow components with multi-user,
        multi-namespace isolation.

This module covers the complete Kubeflow stack with deep theory: Kubernetes
fundamentals for ML, the Kubeflow architecture and component interactions,
Kubeflow Pipelines DSL in depth (components, artifacts, caching), the
Training Operator pattern and distributed training topologies, Katib
hyperparameter search, KServe serving, and production MLOps patterns
including CI/CD for ML and observability.

"""

import textwrap
import re

TOPIC_NAME   = "Kubeflow — The MLOps Platform for Kubernetes"
DISPLAY_NAME = "14 · Kubeflow"
ICON         = "☸️"
SUBTITLE     = "From Kubernetes Fundamentals to Production ML Pipelines"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — KUBERNETES FOUNDATIONS FOR MACHINE LEARNING

### Why Kubernetes for ML?

    Machine learning workloads share the exact operational challenges that
    Kubernetes was designed to solve, plus unique ML-specific ones:

    General challenges (shared with web services):
        - Reproducible, environment-isolated execution (containers)
        - Resource scheduling across a heterogeneous cluster
        - Health monitoring, automatic restart on failure
        - Horizontal scaling based on load
        - Rolling deployments and rollbacks

    ML-specific challenges:
        - Gang scheduling: all N distributed training workers must start
          simultaneously, or the job cannot proceed
        - GPU and TPU allocation: specialised hardware with complex topology
          (NVLink, InfiniBand) affects performance dramatically
        - Shared storage: training data, checkpoints, and model artefacts
          must be accessible across many pods
        - Long-running jobs: training may run for hours or days; eviction
          must trigger checkpoint recovery, not restart from scratch
        - Multi-step pipelines: data prep → training → evaluation → serving
          with complex dependencies and conditional branches
        - Experiment tracking: each run is an experiment with different
          hyperparameters; results must be recorded and comparable

    Kubernetes solves the first category natively; Kubeflow extends it
    for the second category through Custom Resource Definitions (CRDs).

### Kubernetes Core Concepts for ML Engineers

    POD:
        The smallest deployable unit. One or more containers sharing:
            - Network namespace (same IP, communicate via localhost)
            - Storage volumes (shared filesystem mounts)
        A training worker is a Pod. A model server is a Pod.
        Pods are ephemeral — if deleted, they do not restart automatically.

    DEPLOYMENT:
        Manages a set of identical Pod replicas. Ensures N replicas are
        always running. Handles rolling updates. Used for stateless services
        (model servers, UI components).

    STATEFULSET:
        Like a Deployment, but Pods have stable network identities
        (pod-0, pod-1, ...) and stable persistent storage.
        Required for distributed training where workers reference each
        other by stable hostnames.

    SERVICE:
        A stable network endpoint (DNS name + virtual IP) that routes
        traffic to Pods matching a label selector.
        Even as Pods come and go, the Service IP/DNS is constant.
        worker-0.myjob.default.svc.cluster.local — how training
        workers discover each other.

    PERSISTENT VOLUME CLAIM (PVC):
        A request for persistent storage. The PVC is fulfilled by a
        Persistent Volume (physical storage: NFS, EBS, GCS, etc.).
        ML use: attach a PVC to every training worker Pod to share
        training data without copying it to each container.

    CUSTOM RESOURCE DEFINITION (CRD):
        Extends the Kubernetes API with new resource types.
        This is how Kubeflow works: KFP, PyTorchJob, Katib Experiment
        are all CRDs. `kubectl apply -f myjob.yaml` works just like
        applying a native Kubernetes resource.

    NAMESPACE:
        Virtual cluster within a cluster. Kubeflow uses namespaces
        to provide user and team isolation — each user gets their own
        namespace where their pipelines, jobs, and notebooks live.

    RESOURCE REQUESTS AND LIMITS:
        resources:
          requests:
            cpu:    "4"
            memory: "16Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu:    "8"
            memory: "32Gi"
            nvidia.com/gpu: "1"
        requests: guaranteed minimum (scheduler places pod based on this)
        limits:   maximum allowed (pod killed if exceeded for CPU-throttled,
                  or OOMKilled for memory)

### The Container Image in ML

    Every Kubeflow component runs in a Docker container. The container image
    contains:
        - Python version and packages (pinned in requirements.txt)
        - CUDA and cuDNN versions (must match the GPU driver on the node)
        - Your training code

    Best practices:
        - Pin all dependencies to exact versions
        - Use multi-stage builds to keep image size minimal
        - Tag images with git commit SHA for reproducibility
        - Store images in a registry (GCR, ECR, DockerHub, private registry)

    Example Dockerfile for a training job:
        FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt --no-cache-dir
        COPY src/ ./src/
        ENTRYPOINT ["python", "src/train.py"]


##### PART 2 — KUBEFLOW ARCHITECTURE AND COMPONENT INTERACTIONS


### The Kubeflow Component Map

    Kubeflow is not a single monolithic system — it is a platform composed
    of independently deployable components that integrate via Kubernetes APIs:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  Central Dashboard (Kubeflow UI — port-forward or Ingress)           │
    ├──────────────────────────────────────────────────────────────────────┤
    │  KF Notebooks  │  KF Pipelines  │  Katib  │  KServe  │  Training Op  │
    ├──────────────────────────────────────────────────────────────────────┤
    │  Kubernetes API Server (CRDs for each component)                     │
    ├──────────────────────────────────────────────────────────────────────┤
    │  Kubernetes (pods, services, PVCs, namespaces, RBAC)                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │  Infrastructure (GKE, EKS, AKS, on-premise bare metal)               │
    └──────────────────────────────────────────────────────────────────────┘

    Supporting infrastructure:
        Istio:     Service mesh. Handles mTLS between services, traffic
                   routing for canary deployments, observability.
        Knative:   Serverless framework. Powers KServe's autoscaling
                   (scale-to-zero, scale-from-zero on request arrival).
        Argo Workflows: Workflow engine underlying KFP (in KFP v1).
        MinIO/Cloud Storage: Object store for pipeline artefacts and
                             model checkpoints.
        MySQL:     Metadata database for KFP (pipeline run history,
                   artefact lineage).

### Multi-User Isolation with Profiles

    Kubeflow Profiles provide multi-tenancy:
        - Each user gets a Profile (which creates a Namespace)
        - Role-based access control (RBAC) is scoped to the namespace
        - Users can share a Profile with others (as Owner or Contributor)
        - Notebooks, pipeline runs, and training jobs are namespace-scoped

    Authentication flow:
        User → Dex (identity provider) → Istio Gateway → OIDC token
        → namespace-scoped access to Kubeflow components

### Pipeline Run Execution Model

    When you submit a KFP pipeline run:
        1. The KFP backend stores the pipeline YAML and creates a run record in MySQL
        2. For each pipeline step, the pipeline controller creates a Pod
        3. Each Pod runs the step's container image with its arguments
        4. The Pod writes outputs to MinIO/cloud storage (as artefacts)
        5. When the step completes, the controller reads outputs and passes
           them as inputs to downstream steps
        6. Artefact lineage is tracked in ML Metadata (MLMD)

    Key insight: KFP COMPILES a Python pipeline function into a YAML spec
    (a static description of the DAG). The Python code never runs in the
    cluster — only the compiled YAML does. The cluster runs container images
    you point to in the YAML.


##### PART 3 — KUBEFLOW PIPELINES: ORCHESTRATING ML WORKFLOWS

### What Is a Kubeflow Pipeline?

    A KFP pipeline is a directed acyclic graph (DAG) of containerised steps.
    Each step is called a COMPONENT — a reusable unit with:
        - Defined inputs and outputs (typed)
        - A container image and command to run
        - Resource requirements (CPU, memory, GPU)
        - Caching behaviour (skip if inputs unchanged)

    The pipeline DAG encodes data dependencies: if step B uses the output
    of step A, then B must wait for A to complete. Steps with no dependencies
    between them run in parallel automatically.

### Component Types in KFP v2

    Python Component (@dsl.component):
        The simplest type. Write a Python function; KFP wraps it in a
        lightweight container (uses the kfp-component base image).

        from kfp import dsl
        from kfp.dsl import Input, Output, Dataset, Model, Metrics

        @dsl.component(
            base_image    = "python:3.11-slim",
            packages_to_install = ["pandas==2.1.0", "scikit-learn==1.3.2"],
        )
        def preprocess_data(
            raw_data_path: str,
            output_dataset: Output[Dataset],
            test_fraction: float = 0.2,
        ):
            import pandas as pd
            from sklearn.model_selection import train_test_split

            df    = pd.read_csv(raw_data_path)
            train, test = train_test_split(df, test_size=test_fraction)
            train.to_csv(output_dataset.path + "/train.csv", index=False)

    Container Component (@dsl.container_component):
        For arbitrary containerised workloads — any language, any tool.

        @dsl.container_component
        def run_spark_job(input_path: str, output_path: str):
            return dsl.ContainerSpec(
                image   = "apache/spark:3.5.0",
                command = ["spark-submit"],
                args    = ["--master", "local[*]",
                           "my_job.py", input_path, output_path],
            )

    Importer Component:
        Imports external artefacts (models, datasets) into the pipeline
        lineage system without reprocessing.

### Artefact Types and the ML Metadata System

    KFP v2 has a rich typed artefact system. Every component input/output
    is either a primitive (str, int, float, bool, list, dict) or an Artefact:

        Dataset:   tabular data, feature tables, raw data
        Model:     trained model weights + metadata
        Metrics:   scalar evaluation metrics (accuracy, F1, etc.)
        ClassificationMetrics:  confusion matrix, ROC curve
        HTML:      visualisation output (renders in KFP UI)
        Markdown:  documentation output (renders in KFP UI)
        Artifact:  generic untyped artefact

    The ML Metadata (MLMD) system records:
        - Which component produced each artefact
        - Which run and execution produced it
        - What the artefact URI (storage path) is
        - Properties and custom metadata

    This enables full DATA LINEAGE: given a model, trace back to:
        the training job → the training dataset → the preprocessing step →
        the raw data source. Critical for debugging, compliance, and
        reproducing results.

    Accessing artefacts in a component:
        def my_component(
            model: Input[Model],        # read an existing model
            metrics: Output[Metrics],   # write metrics
        ):
            model_path = model.uri        # storage URI
            model_name = model.name       # registered name
            metrics.log_metric("accuracy", 0.95)   # write metrics
            metrics.log_metric("f1", 0.93)

### Pipeline Structure and Control Flow

    Defining a pipeline:
        @dsl.pipeline(
            name        = "training-pipeline",
            description = "End-to-end training pipeline",
        )
        def training_pipeline(
            data_path:    str,
            n_epochs:     int   = 10,
            learning_rate: float = 3e-4,
        ):
            # Each step call returns a task object
            preprocess_task = preprocess_data(
                raw_data_path = data_path
            )
            train_task = train_model(
                train_dataset  = preprocess_task.outputs["output_dataset"],
                n_epochs       = n_epochs,
                learning_rate  = learning_rate,
            )
            # train_task implicitly depends on preprocess_task (via data flow)

            evaluate_task = evaluate_model(
                model   = train_task.outputs["model"],
                test_ds = preprocess_task.outputs["test_dataset"],
            )

    Conditional execution:
        from kfp import dsl

        with dsl.If(evaluate_task.output > 0.9):
            deploy_model(model=train_task.outputs["model"])

    Parallel loop (fan-out):
        with dsl.ParallelFor(["us-east", "eu-west", "ap-south"]) as region:
            train_regional_model(region=region, data=preprocess_task.output)

    Exit handler (always runs, even on failure):
        with dsl.ExitHandler(notify_slack(message="Pipeline finished")):
            # Main pipeline steps here
            ...

### Caching: Skipping Unchanged Steps

    KFP caches the output of each step. If:
        - The container image has not changed
        - The step's inputs have not changed
        - Caching is enabled for the step

    Then the step is SKIPPED on the next run and its cached output is
    reused directly. This dramatically speeds up iterative development:
    change only the training code → only the training step reruns.

    Controlling caching:
        # Disable caching for a specific task
        train_task.set_caching_options(enable_caching=False)

        # Disable caching for an entire pipeline run (at submission time)
        run = client.create_run_from_pipeline_func(
            pipeline_fn,
            enable_caching=False,
        )

### Compiling and Submitting Pipelines

    Compilation: Python → YAML (the Intermediate Representation):
        from kfp import compiler

        compiler.Compiler().compile(
            pipeline_func = training_pipeline,
            package_path  = "pipeline.yaml",
        )

    The YAML contains:
        - The DAG structure
        - Container images and commands for each step
        - Resource requests
        - Input/output artefact definitions

    Submission via the KFP Python client:
        import kfp

        client = kfp.Client(host="http://kubeflow.example.com")

        run = client.create_run_from_pipeline_func(
            pipeline_func = training_pipeline,
            arguments     = {
                "data_path":    "gs://my-bucket/data/train.csv",
                "n_epochs":     20,
                "learning_rate": 1e-3,
            },
            run_name       = "training-run-v2",
            experiment_name = "production-experiments",
            enable_caching  = True,
        )
        print(f"Run URL: {run.ui_url}")

    Recurring runs (scheduled pipelines):
        client.create_recurring_run(
            pipeline_func    = training_pipeline,
            cron_expression  = "0 2 * * *",   # every day at 2am
            max_concurrency  = 1,
        )


##### PART 4 — TRAINING OPERATOR: DISTRIBUTED TRAINING ON KUBERNETES

### What Is the Kubeflow Training Operator?

    The Training Operator is a Kubernetes operator (a controller that manages
    custom resources) for running distributed machine learning training jobs.
    It implements custom resource types for different frameworks:

        TFJob:        Distributed TensorFlow training
        PyTorchJob:   Distributed PyTorch training (DDP, PS topology)
        MXNetJob:     Distributed MXNet training
        XGBoostJob:   Distributed XGBoost
        MPIJob:       General MPI-based distributed jobs (Horovod)
        PaddleJob:    Distributed PaddlePaddle training
        JAXJob:       Distributed JAX training (newer)

    Each custom resource defines the DESIRED STATE. The operator's controller
    reconciles the actual state toward the desired state:
        - Creates worker and parameter-server Pods
        - Sets up networking (headless Services for stable DNS names)
        - Monitors job status and handles failures
        - Cleans up resources when the job completes

### PyTorchJob: The Standard for PyTorch Training

    A PyTorchJob spec defines the distributed training topology:

        apiVersion: kubeflow.org/v1
        kind: PyTorchJob
        metadata:
          name: bert-finetune
          namespace: user-alice
        spec:
          nprocPerNode: "4"   # GPUs per node (processes per pod)
          pytorchReplicaSpecs:
            Master:           # exactly 1 Master — coordinates and reports status
              replicas: 1
              template:
                spec:
                  containers:
                  - name: pytorch
                    image: my-registry/bert-trainer:v1.2.3
                    resources:
                      limits:
                        nvidia.com/gpu: "4"
                    command: ["torchrun", "--standalone", "train.py"]
            Worker:           # N Worker replicas
              replicas: 3
              template:
                spec:
                  containers:
                  - name: pytorch
                    image: my-registry/bert-trainer:v1.2.3
                    resources:
                      limits:
                        nvidia.com/gpu: "4"

    The Training Operator automatically:
        - Creates 4 Pods (1 Master + 3 Workers)
        - Creates a headless Service so Workers find each other by DNS
        - Sets environment variables: MASTER_ADDR, MASTER_PORT, WORLD_SIZE,
          RANK, LOCAL_RANK for PyTorch DDP
        - Monitors Pod health; if any Pod fails, marks job as Failed

    torchrun (PyTorch's distributed launcher) inside the container:
        - Reads MASTER_ADDR and MASTER_PORT from environment
        - Launches nproc_per_node processes per Pod
        - Sets up the NCCL/Gloo process group for gradient all-reduce

### Distributed Training Topologies

    DATA PARALLEL (DDP — most common):
        All workers hold the full model.
        Each worker processes a different data shard.
        After each backward pass: AllReduce gradients (average across workers).
        Effective batch size = per_worker_batch × world_size.

        Total GPUs: N workers × G GPUs per worker.
        Throughput scales linearly with GPU count (ideal).
        Memory: same as single-GPU (model fits in one GPU).

    PARAMETER SERVER (PS):
        Some replicas are "workers" (compute gradients).
        Others are "parameter servers" (store and update weights).
        Workers compute gradients on their data shard.
        Workers push gradients to PS; PS applies update; Workers pull new weights.

        Advantage: asynchronous updates → no waiting for slow workers.
        Disadvantage: communication bottleneck at PS; stale gradients.
        Used in: TFJob (supports PS topology natively), older DL systems.

    FULLY SHARDED DATA PARALLEL (FSDP):
        Model weights, gradients, AND optimiser states sharded across GPUs.
        Enables training models larger than any single GPU's VRAM.
        PyTorch FSDP: wraps model layers in FSDP modules, handles sharding.
        Used for: LLM fine-tuning (7B+ parameter models).

    PIPELINE PARALLEL:
        Split model layers across GPUs along the depth dimension.
        GPU 0 processes layers 1-12, GPU 1 processes layers 13-24, etc.
        Micro-batching fills the pipeline to prevent idle GPUs.
        Used for: very deep models, combined with tensor parallelism.

    TENSOR PARALLEL:
        Split individual weight matrices across GPUs.
        Each GPU computes a slice of each matrix multiplication.
        Results combined via AllReduce at the end of each layer.
        Requires fast interconnect (NVLink). Used in Megatron-LM.

### Gang Scheduling: The ML-Specific Kubernetes Challenge

    Gang scheduling means: ALL pods in a distributed job must be scheduled
    simultaneously, or NONE of them should run.

    The problem without gang scheduling:
        Worker-0 starts and holds a GPU, waiting for Worker-1.
        Worker-1 cannot be scheduled (cluster is full).
        Deadlock: Worker-0 holds a GPU but cannot progress;
        other jobs cannot get that GPU.

    Solutions:
        Volcano Scheduler: Kubernetes-native gang scheduler. Knows about
        PyTorchJob/TFJob and schedules all pods atomically.

        Coscheduling plugin: Alternative, lighter-weight gang scheduling.

        Pod Groups (Volcano): Defines the minimum number of pods that
        must be schedulable simultaneously:
            spec:
              minMember: 4   # only schedule if 4 pods can start together

### Resource Management: GPU Scheduling

    Kubernetes treats GPUs as integer resources:
        nvidia.com/gpu: "2"   # request exactly 2 GPUs

    GPU isolation: by default, a container with 1 GPU cannot see other GPUs.
    CUDA_VISIBLE_DEVICES is set automatically by the device plugin.

    Multi-Instance GPU (MIG):
        NVIDIA A100/H100 can be partitioned into 7 independent instances.
        Each instance has dedicated compute, memory, cache.
        Kubernetes sees them as separate resources:
            nvidia.com/mig-1g.10gb: "1"   # request one 1/7 A100 slice

    Time-slicing:
        Multiple pods share a GPU with time multiplexing.
        No memory isolation! All pods can OOM each other.
        Use only for inference workloads with well-known memory usage.

    RDMA / InfiniBand:
        High-bandwidth, low-latency interconnect between nodes.
        Required for efficient multi-node training with NCCL.
        Requested as: rdma/hca: "1"
        Without RDMA: inter-node gradient sync over Ethernet is the bottleneck.


##### PART 5 — KATIB: AUTOMATED HYPERPARAMETER TUNING

### Katib Architecture

    Katib provides Kubernetes-native hyperparameter tuning and Neural
    Architecture Search (NAS). It is modelled after Google Vizier.

    Core resources:
        Experiment: defines the search space, objective, and algorithm
        Trial:      one hyperparameter configuration being evaluated
        Suggestion: the component that generates new HP configurations

    How Katib works:
        1. User submits an Experiment CRD
        2. The Experiment controller creates Trial objects
        3. Each Trial controller creates a training job (PyTorchJob, etc.)
        4. The job runs, logs metrics to stdout or a metrics collector
        5. The Katib metrics collector scrapes the metrics from the pod logs
        6. The Suggestion service receives the metrics, updates its model,
           and proposes the next set of hyperparameters
        7. New Trials are created; repeat until budget exhausted

### The Katib Experiment Spec

        apiVersion: kubeflow.org/v1beta1
        kind: Experiment
        metadata:
          name: bert-hpo
          namespace: user-alice
        spec:
          objective:
            type:         maximize       # or minimize
            goal:         0.95           # stop when this is reached
            objectiveMetricName: val-accuracy

          algorithm:
            algorithmName: bayesianoptimization   # or random, grid, hyperband, cma

          parallelTrialCount: 4     # run 4 trials simultaneously
          maxTrialCount:       30   # stop after 30 trials total
          maxFailedTrialCount: 5    # stop if 5 trials fail

          parameters:
          - name:         learning-rate
            parameterType: double
            feasibleSpace:
              min: "0.0001"
              max: "0.01"
          - name:         batch-size
            parameterType: categorical
            feasibleSpace:
              list: ["32", "64", "128", "256"]
          - name:         num-layers
            parameterType: int
            feasibleSpace:
              min: "2"
              max: "8"

          trialTemplate:
            trialParameters:
            - name:      learningRate
              reference: learning-rate
            - name:      batchSize
              reference: batch-size
            trialSpec:   # the training job to run for each trial
              apiVersion: kubeflow.org/v1
              kind: PyTorchJob
              spec: ...  # PyTorchJob spec with ${trialParameters.learningRate} substitution

### Search Algorithms in Katib

    Random Search:
        Each trial is an independently random draw from the search space.
        No model of the objective function.
        Perfectly parallelisable (no dependency between trials).
        Good baseline; competitive for high-parallelism settings.

    Grid Search:
        Exhaustive evaluation of a Cartesian product of discrete values.
        Total trials = ∏ |options per parameter|. Scales exponentially.
        Only feasible for 2-3 parameters with few options each.

    Bayesian Optimisation (GP-based):
        Fits a Gaussian Process to the (HP config → objective) surface.
        Uses an acquisition function (Expected Improvement, UCB) to select
        the next trial that maximises expected gain.
        Very sample-efficient — finds good configs with fewer trials.
        Sequential (each trial informs the next) → less parallel benefit.

    TPE (Tree-structured Parzen Estimator):
        Models good and bad hyperparameter regions separately.
        More scalable than GP-based BO for high-dimensional spaces.
        Used by default in many Katib deployments.

    Hyperband:
        Early-stops unpromising trials to focus resources on promising ones.
        Runs trials at multiple budget levels (epochs).
        Brackets: at each level, promotes the top-η fraction of trials.
        Katib implements ASHA (asynchronous Hyperband) for better GPU util.

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy):
        Evolutionary algorithm that adapts a multivariate Gaussian to
        the high-scoring region of the search space.
        Excellent for continuous high-dimensional spaces.
        Often finds better solutions than BO for complex parameter landscapes.

    DARTS (Differentiable Architecture Search):
        NAS method that relaxes the discrete architecture choice to a
        continuous mixture, optimises jointly with standard gradient descent.
        Finds novel neural architectures without exhaustive enumeration.

### Metrics Collection

    Katib collects metrics from training pods in three ways:

    StdOut collector (default):
        Training code prints metrics in a specific format to stdout:
            print(f"val-accuracy={val_acc:.4f}")
        Katib's sidecar container scrapes the pod logs.

    File collector:
        Write metrics to a file at a known path:
            with open("/var/log/katib/metrics.json", "w") as f:
                json.dump({"val-accuracy": 0.95}, f)

    Prometheus collector:
        Expose a /metrics endpoint in Prometheus format.
        Katib collects from the endpoint at the end of the trial.

    TensorBoard collector:
        Reads metrics from TensorBoard event files.


##### PART 6 — KSERVE: PRODUCTION MODEL SERVING

### What Is KServe?

    KServe (formerly KFServing) is a Kubernetes-native model inference server
    built on Knative Serving and Istio. It provides:
        - Serverless inference: scale to zero when idle, scale up on traffic
        - Pre/post-processing: transform requests before/after model inference
        - Canary deployments: gradually shift traffic to new model versions
        - A/B testing: split traffic between multiple model versions
        - Model explainability: LIME, SHAP, ALIBI integration
        - Batch inference: queue-based large-scale prediction jobs
        - Multi-model serving: serve thousands of models from one server

### The InferenceService CRD

    KServe's primary resource is the InferenceService:

        apiVersion: serving.kserve.io/v1beta1
        kind: InferenceService
        metadata:
          name: bert-sentiment
          namespace: user-alice
        spec:
          predictor:
            model:
              modelFormat:
                name: pytorch   # or tensorflow, sklearn, xgboost, onnx
              storageUri: "gs://my-models/bert-sentiment/v3/"
              resources:
                requests:
                  nvidia.com/gpu: "1"

    This creates:
        - A Knative Service (auto-scales based on request traffic)
        - An Istio VirtualService (traffic routing)
        - A KNative Revision per model version

    Default pre-built serving runtimes:
        TorchServe:     PyTorch models (.pt, .mar format)
        TensorFlow Serving: SavedModel format
        Triton:         NVIDIA Triton (all frameworks, GPU-optimised)
        MLServer:       scikit-learn, XGBoost, LightGBM, ONNX
        Hugging Face:   Transformers models from the Hub

### Canary Deployments and Traffic Management

    Rolling out a new model version safely:

        spec:
          predictor:
            canaryTrafficPercent: 10   # 10% → new version
            model:
              modelFormat: {name: pytorch}
              storageUri: "gs://models/bert/v4/"   # new version

          # Original (v3) gets 90% of traffic automatically

    As confidence grows:
        canaryTrafficPercent: 30   → 50   → 80  → 100 (full rollout)

    This enables testing a new model on real traffic with limited blast
    radius — if the new model degrades latency or accuracy, roll back.

### KServe Inference Protocol

    KServe supports two protocols:

    V1 protocol (scikit-learn compatible):
        POST /v1/models/{name}:predict
        Body: {"instances": [[1.2, 3.4, 5.6], [7.8, 9.0, 1.1]]}
        Response: {"predictions": [0, 1]}

    V2 protocol (Open Model Interface — standard):
        POST /v2/models/{name}/infer
        Body: {"inputs": [{"name": "input", "shape": [2, 3],
                            "datatype": "FP32",
                            "data": [1.2, 3.4, 5.6, 7.8, 9.0, 1.1]}]}

### Transformer: Pre/Post-Processing Pipelines

    A Transformer is a component that runs BEFORE the predictor (pre-process)
    and AFTER (post-process), without baking preprocessing into the model:

        spec:
          transformer:
            containers:
            - image: my-registry/bert-preprocess:v1
              name: transformer

          predictor:
            model:
              storageUri: "gs://models/bert/v3/"

    The transformer receives the raw HTTP request, transforms it (tokenise,
    normalise, etc.), calls the predictor, receives the raw output, and
    transforms it back (decode labels, format response).

    This separation means:
        - Preprocessing logic is versioned independently of the model
        - The predictor handles only tensor I/O (efficient)
        - You can reuse the same predictor with different transformers


##### PART 7 — MLOPS PATTERNS: CI/CD FOR ML WITH KUBEFLOW

### The ML Lifecycle and Kubeflow's Role

    The full ML lifecycle has stages that Kubeflow maps to:

        Data Collection & Versioning → outside Kubeflow (DVC, Delta Lake)
        Data Validation              → KFP component (Great Expectations, TFDV)
        Feature Engineering          → KFP component (Feast feature store)
        Model Training               → KFP + Training Operator
        Hyperparameter Tuning        → Katib (inside KFP via TrainingClient)
        Model Evaluation             → KFP component
        Model Registry               → KFP artefact lineage / MLflow / Model Registry
        Model Serving                → KServe (triggered by KFP)
        Monitoring                   → Prometheus + KServe metrics
        Retraining Trigger           → KFP recurring runs / event-based triggers

### CI/CD for ML: The Four Levels of Automation

    Level 0 (Manual):
        Scientists train models manually in notebooks.
        Manual deployment. No automation.
        Problem: not reproducible, not scalable.

    Level 1 (Pipeline Automation):
        ML pipeline is automated (Kubeflow Pipelines).
        The pipeline runs automatically on new data.
        Code changes still deployed manually.
        Problem: model is automated but code changes are not.

    Level 2 (CI/CD for ML):
        Code changes trigger automated testing of the pipeline.
        New model versions automatically deployed if metrics pass thresholds.
        Infrastructure as code (pipeline YAML in git).
        Full GitOps: merge to main → test → build image → compile pipeline →
        run pipeline → evaluate → deploy if metrics pass.

    Level 3 (Continuous Training with Monitoring):
        Production model is monitored for data drift and performance decay.
        Automated retraining trigger when drift detected.
        Champion/challenger comparison framework.
        Full closed loop: production → monitoring → retraining → evaluation → deploy.

### A Complete MLOps Pipeline Pattern

    The standard production pipeline in Kubeflow:

        @dsl.pipeline
        def production_ml_pipeline(
            data_version: str,       # e.g. "2024-01-15"
            min_accuracy: float = 0.92,
            run_hpo: bool = False,
        ):
            # 1. Validate incoming data
            validate_task = validate_data(data_version=data_version)

            # 2. Optionally run HPO (expensive, run weekly)
            with dsl.If(run_hpo == True):
                hpo_task = run_katib_experiment(
                    n_trials=50,
                    dataset=validate_task.output,
                )

            # 3. Train with best known config
            train_task = train_model(
                dataset        = validate_task.output,
                learning_rate  = 3e-4,   # or hpo_task.output if run_hpo
                n_epochs       = 20,
            )

            # 4. Evaluate against held-out test set
            eval_task = evaluate_model(
                model   = train_task.outputs["model"],
                dataset = validate_task.output,
            )

            # 5. Register in model registry if metrics pass threshold
            with dsl.If(eval_task.outputs["accuracy"] > min_accuracy):
                register_task = register_model(
                    model        = train_task.outputs["model"],
                    metrics      = eval_task.outputs["metrics"],
                    data_version = data_version,
                )
                # 6. Deploy to KServe (canary first)
                deploy_task = deploy_to_kserve(
                    model_uri      = register_task.output,
                    canary_percent = 20,
                )

### Model Versioning and the Model Registry

    A model registry tracks:
        - Model name and version (e.g. "fraud-detector/v23")
        - Storage location (S3/GCS path to model weights)
        - Training metrics (accuracy, F1, AUC)
        - Training data version used
        - Pipeline run that produced it
        - Serving configuration
        - Stage: staging, production, archived

    KFP artefact lineage provides basic tracking.
    Full model registries: MLflow Model Registry, Vertex AI Model Registry,
    BentoML, or the upcoming Kubeflow Model Registry component.

    Promotion flow:
        Candidate → Staging (automated tests) → Production (A/B) → Champion


##### PART 8 — OBSERVABILITY, DEBUGGING, AND PRODUCTION OPERATIONS

### Observability Stack for Kubeflow

    Prometheus + Grafana:
        Kubeflow exposes Prometheus metrics for:
            - GPU utilisation per pod (DCGM exporter)
            - Training job duration and failure rates
            - KServe request latency, throughput, error rate
            - KFP pipeline run duration and step failure rates
        Grafana dashboards visualise these over time.

    Loki + Grafana:
        Log aggregation. Every Pod's stdout/stderr is indexed.
        Query: show all logs from the last run of "bert-finetune".

    Jaeger / Tempo:
        Distributed tracing for KServe inference requests.
        Traces the full path: HTTP request → transformer → predictor → response.
        Identifies where latency is introduced.

    Weights & Biases / MLflow / Kubeflow Metadata:
        Training experiment tracking within pipeline runs.
        Log metrics, hyperparameters, and artefacts per training run.

### Debugging Kubeflow Pipelines

    Common failure modes and their diagnosis:

    ImagePullBackOff:
        The container image cannot be pulled.
        Fix: check registry credentials (imagePullSecrets), image tag.
        kubectl describe pod <pod-name> -n <namespace>

    OOMKilled:
        The container exceeded its memory limit.
        Fix: increase memory limit, reduce batch size, or use gradient checkpointing.
        kubectl get events -n <namespace>

    CrashLoopBackOff:
        The container repeatedly crashes on start.
        Fix: check logs → kubectl logs <pod-name> -n <namespace>

    Pending (scheduling):
        Insufficient resources on the cluster.
        Fix: scale cluster, reduce resource requests, check node taints.
        kubectl describe pod <pod-name> → shows scheduling failure reason.

    Pipeline step stuck "Running":
        Often a deadlock in distributed training (worker waiting for master).
        Fix: check all pods in the job → kubectl get pods -n <namespace>
        One pod may have failed silently.

### Resource Optimisation for ML Workloads

    Spot/Preemptible instances:
        Use cheap spot instances for training; checkpoint frequently.
        PyTorchJob supports preemption via elastic training (torchrun --max-restarts).
        Cost reduction: 60-80% cheaper than on-demand.

    Right-sizing:
        Profile actual GPU/CPU/memory usage during a short training run.
        Set requests = typical usage; limits = 120% of requests.
        Avoid over-requesting (wastes cluster capacity).

    Cluster autoscaling:
        Kubernetes Cluster Autoscaler provisions new nodes when pending pods exist.
        Set up node pools with different GPU types:
            - t4-pool: cheap inference GPUs
            - a100-pool: expensive training GPUs
        Node selectors and tolerations route jobs to the right pool.

    Vertical Pod Autoscaler (VPA):
        Automatically recommends resource request adjustments based on
        historical usage. Apply in "recommendation only" mode first.

    Priority classes:
        production-serving: highest priority (never preempted)
        training-batch:     medium priority
        experimentation:    lowest priority (can be preempted by others)

### Production Checklist

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Category              │ Checklist item                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Reproducibility       │ Pipeline YAML in git                        │
    │                       │ Container images tagged with git SHA        │
    │                       │ Data versions tracked (not file paths)      │
    │ Reliability           │ Checkpoints every N steps (training)        │
    │                       │ maxFailedTrialCount set (Katib)             │
    │                       │ KServe canary before full rollout           │
    │ Security              │ RBAC per namespace (Profile)                │
    │                       │ Secrets in Vault/Kubernetes secrets         │
    │                       │ No hardcoded credentials in pipeline code   │
    │ Observability         │ Metrics logged to Prometheus                │
    │                       │ Structured JSON logs from all pods          │
    │                       │ Alerts on training job failure              │
    │ Cost                  │ Spot instances for training                 │
    │                       │ Scale-to-zero for serving (Knative)         │
    │                       │ Resource requests match actual usage        │
    └─────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · KFP Components & Pipelines — Building ML Workflows in Python": {
        "description": (
            "Complete Kubeflow Pipelines v2 (KFP SDK) tour from scratch. "
            "Installing kfp and inspecting the DSL. "
            "@dsl.component: Python function → containerised step. "
            "Input/Output types: primitives, Dataset, Model, Metrics. "
            "Component resource specification: CPU, memory, GPU. "
            "Component caching: enable/disable per step. "
            "@dsl.pipeline: wiring steps into a DAG. "
            "Data flow: passing artefacts between steps. "
            "Control flow: dsl.If, dsl.ParallelFor, dsl.ExitHandler. "
            "Pipeline compilation: Python → YAML IR. "
            "Inspecting compiled YAML: DAG structure and component specs. "
            "Simulated local pipeline execution for offline demo."
        ),
        "language": "python",
        "code": '''
import json
import time
import yaml
import tempfile
import os
import hashlib
from typing import NamedTuple, List, Dict, Any
from dataclasses import dataclass, field

try:
    import kfp
    from kfp import dsl, compiler
    from kfp.dsl import (
        Input, Output, Dataset, Model, Metrics,
        ClassificationMetrics, HTML, Artifact,
    )
    print(f"  KFP SDK version: {kfp.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "kfp>=2.0.0", "--quiet"], check=True)
    import kfp
    from kfp import dsl, compiler
    from kfp.dsl import (
        Input, Output, Dataset, Model, Metrics,
        ClassificationMetrics, HTML, Artifact,
    )
    print(f"  KFP SDK version: {kfp.__version__}")

print("=" * 65)
print("  KFP COMPONENTS & PIPELINES — BUILDING ML WORKFLOWS")
print("=" * 65)
print()

import numpy as np
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Defining components with @dsl.component
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — @dsl.component: typed Python function → pipeline step")
print("━" * 65)
print()

@dsl.component(
    base_image          = "python:3.11-slim",
    packages_to_install = ["pandas==2.2.0", "scikit-learn==1.4.0",
                            "numpy==1.26.4"],
)
def ingest_data(
    n_samples:       int,
    n_features:      int,
    noise:           float,
    output_dataset:  Output[Dataset],
) -> None:
    """
    Step 1 — Data ingestion.
    In production: reads from a database, S3, or feature store.
    Here: generates synthetic regression data.
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        noise=noise, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["target"] = y

    import os
    os.makedirs(output_dataset.path, exist_ok=True)
    df.to_parquet(os.path.join(output_dataset.path, "data.parquet"), index=False)
    output_dataset.metadata["n_samples"]  = n_samples
    output_dataset.metadata["n_features"] = n_features
    output_dataset.metadata["noise"]      = noise
    print(f"Generated {n_samples} samples with {n_features} features.")


@dsl.component(
    base_image          = "python:3.11-slim",
    packages_to_install = ["pandas==2.2.0", "scikit-learn==1.4.0",
                            "numpy==1.26.4"],
)
def split_and_preprocess(
    input_dataset:   Input[Dataset],
    train_dataset:   Output[Dataset],
    val_dataset:     Output[Dataset],
    test_fraction:   float = 0.2,
    scale_features:  bool  = True,
) -> None:
    """Step 2 — Train/val split and feature normalisation."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import os, pickle

    df = pd.read_parquet(os.path.join(input_dataset.path, "data.parquet"))
    X  = df.drop("target", axis=1)
    y  = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_fraction, random_state=42
    )
    if scale_features:
        scaler  = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_val   = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

    for ds_output, X_part, y_part in [
        (train_dataset, X_train, y_train),
        (val_dataset,   X_val,   y_val),
    ]:
        os.makedirs(ds_output.path, exist_ok=True)
        part = X_part.copy()
        part["target"] = y_part.values
        part.to_parquet(os.path.join(ds_output.path, "data.parquet"), index=False)
        ds_output.metadata["n_rows"] = len(part)

    print(f"Train: {len(X_train)} rows | Val: {len(X_val)} rows")


@dsl.component(
    base_image          = "python:3.11-slim",
    packages_to_install = ["pandas==2.2.0", "scikit-learn==1.4.0",
                            "numpy==1.26.4"],
)
def train_model(
    train_dataset:  Input[Dataset],
    trained_model:  Output[Model],
    alpha:          float = 1.0,
    fit_intercept:  bool  = True,
) -> float:
    """Step 3 — Train a Ridge regression model. Returns train R²."""
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import Ridge
    import os, pickle

    df     = pd.read_parquet(os.path.join(train_dataset.path, "data.parquet"))
    X      = df.drop("target", axis=1).values
    y      = df["target"].values

    model  = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X, y)
    r2     = model.score(X, y)

    os.makedirs(trained_model.path, exist_ok=True)
    with open(os.path.join(trained_model.path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    trained_model.metadata["alpha"]         = alpha
    trained_model.metadata["train_r2"]      = r2
    trained_model.metadata["n_features"]    = X.shape[1]
    trained_model.metadata["framework"]     = "scikit-learn"
    trained_model.metadata["model_class"]   = "Ridge"

    print(f"Trained Ridge(alpha={alpha}), train R² = {r2:.4f}")
    return r2


@dsl.component(
    base_image          = "python:3.11-slim",
    packages_to_install = ["pandas==2.2.0", "scikit-learn==1.4.0",
                            "numpy==1.26.4"],
)
def evaluate_model(
    val_dataset:    Input[Dataset],
    trained_model:  Input[Model],
    metrics:        Output[Metrics],
    clf_metrics:    Output[ClassificationMetrics],
) -> NamedTuple("EvalResults", [("r2", float), ("mse", float), ("mae", float)]):
    """Step 4 — Evaluate model on validation set. Log metrics to KFP."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import os, pickle
    from collections import namedtuple

    df     = pd.read_parquet(os.path.join(val_dataset.path, "data.parquet"))
    X, y   = df.drop("target", axis=1).values, df["target"].values

    with open(os.path.join(trained_model.path, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    r2     = float(r2_score(y, y_pred))
    mse    = float(mean_squared_error(y, y_pred))
    mae    = float(mean_absolute_error(y, y_pred))

    # Log scalar metrics — visible in KFP UI
    metrics.log_metric("val_r2",  r2)
    metrics.log_metric("val_mse", mse)
    metrics.log_metric("val_mae", mae)
    metrics.log_metric("n_val_samples", len(y))

    # For classification tasks: confusion matrix / ROC (demo with regression bins)
    y_bins    = (y > np.median(y)).astype(int)
    pred_bins = (y_pred > np.median(y)).astype(int)
    clf_metrics.log_confusion_matrix(
        categories = ["below_median", "above_median"],
        matrix     = [[int(((y_bins==0) & (pred_bins==0)).sum()),
                       int(((y_bins==0) & (pred_bins==1)).sum())],
                      [int(((y_bins==1) & (pred_bins==0)).sum()),
                       int(((y_bins==1) & (pred_bins==1)).sum())]],
    )
    print(f"Val R²={r2:.4f}  MSE={mse:.4f}  MAE={mae:.4f}")
    EvalResults = namedtuple("EvalResults", ["r2", "mse", "mae"])
    return EvalResults(r2=r2, mse=mse, mae=mae)


@dsl.component(
    base_image = "python:3.11-slim",
    packages_to_install = ["pandas==2.2.0"],
)
def register_model(
    trained_model:  Input[Model],
    metrics:        Input[Metrics],
    registry_name:  str,
    version_tag:    str,
) -> str:
    """Step 5 — Register model in the model registry (simulated)."""
    import json
    # In production: call MLflow / Vertex AI Model Registry / KServe API
    registration = {
        "registry":     registry_name,
        "version":      version_tag,
        "model_uri":    trained_model.uri,
        "framework":    trained_model.metadata.get("framework", "unknown"),
        "model_class":  trained_model.metadata.get("model_class", "unknown"),
        "val_r2":       metrics.metadata.get("val_r2", 0),
        "status":       "registered",
    }
    print(f"Registered: {json.dumps(registration, indent=2)}")
    return json.dumps(registration)


# Inspect component metadata
print(f"  Defined {5} pipeline components:")
components = [ingest_data, split_and_preprocess, train_model,
              evaluate_model, register_model]
for comp in components:
    name        = comp.component_spec.name
    n_inputs    = len(comp.component_spec.inputs or [])
    n_outputs   = len(comp.component_spec.outputs or [])
    image       = (comp.component_spec.implementation.container.image
                   if hasattr(comp.component_spec.implementation, "container") else "N/A")
    print(f"  [{name:<28}] inputs={n_inputs}  outputs={n_outputs}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Building the pipeline DAG
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Pipeline DAG: wiring components with data flow")
print("━" * 65)
print()

@dsl.pipeline(
    name        = "regression-training-pipeline",
    description = "End-to-end regression training: ingest → split → train → evaluate → register",
)
def regression_pipeline(
    n_samples:       int   = 2000,
    n_features:      int   = 20,
    noise:           float = 25.0,
    test_fraction:   float = 0.2,
    ridge_alpha:     float = 1.0,
    min_r2_to_register: float = 0.7,
    registry_name:   str   = "regression-model-registry",
    version_tag:     str   = "v1",
):
    # Step 1: ingest
    ingest_task = ingest_data(
        n_samples  = n_samples,
        n_features = n_features,
        noise      = noise,
    )
    ingest_task.set_display_name("Ingest Data")
    ingest_task.set_caching_options(enable_caching=True)

    # Step 2: preprocess (depends on ingest via data flow)
    split_task = split_and_preprocess(
        input_dataset  = ingest_task.outputs["output_dataset"],
        test_fraction  = test_fraction,
        scale_features = True,
    )
    split_task.set_display_name("Split & Normalise")
    split_task.set_caching_options(enable_caching=True)

    # Step 3: train (depends on split)
    train_task = train_model(
        train_dataset = split_task.outputs["train_dataset"],
        alpha         = ridge_alpha,
        fit_intercept = True,
    )
    train_task.set_display_name("Train Ridge Model")
    train_task.set_caching_options(enable_caching=False)  # always retrain

    # Step 4: evaluate (depends on split AND train)
    eval_task = evaluate_model(
        val_dataset   = split_task.outputs["val_dataset"],
        trained_model = train_task.outputs["trained_model"],
    )
    eval_task.set_display_name("Evaluate on Validation")

    # Step 5: conditional registration
    with dsl.If(
        eval_task.outputs["r2"] > min_r2_to_register,
        name="r2-threshold-check",
    ):
        register_task = register_model(
            trained_model = train_task.outputs["trained_model"],
            metrics       = eval_task.outputs["metrics"],
            registry_name = registry_name,
            version_tag   = version_tag,
        )
        register_task.set_display_name("Register Model")


print(f"  Pipeline: regression_pipeline")
print(f"  Parameters:")
import inspect
sig = inspect.signature(regression_pipeline.pipeline_func)
for param_name, param in sig.parameters.items():
    default = param.default if param.default != inspect.Parameter.empty else "required"
    print(f"    {param_name:<25}: default={default}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Compile the pipeline to YAML
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Pipeline compilation: Python → YAML IR")
print("━" * 65)
print()

with tempfile.TemporaryDirectory() as tmp:
    yaml_path = os.path.join(tmp, "regression_pipeline.yaml")

    compiler.Compiler().compile(
        pipeline_func = regression_pipeline,
        package_path  = yaml_path,
    )

    yaml_size = os.path.getsize(yaml_path)
    print(f"  Compiled pipeline → {yaml_path}")
    print(f"  YAML size: {yaml_size:,} bytes")
    print()

    with open(yaml_path) as f:
        pipeline_yaml = yaml.safe_load(f)

    # Inspect the compiled YAML structure
    spec = pipeline_yaml.get("pipelineInfo", {})
    print(f"  Pipeline YAML structure:")
    print(f"    pipelineInfo.name:  {spec.get('name', 'N/A')}")

    components_section = pipeline_yaml.get("components", {})
    print(f"    components:         {len(components_section)} defined")

    deploy_graph = pipeline_yaml.get("deploymentSpec", {}).get("executors", {})
    print(f"    executors:          {len(deploy_graph)}")
    print()

    # Show executor details (container images, commands)
    print(f"  Executor specs (container image per component):")
    print(f"  {'Executor':<30} {'Image':<35}")
    print(f"  {'─'*68}")
    for exec_name, exec_spec in list(deploy_graph.items())[:6]:
        container = exec_spec.get("container", {})
        image     = container.get("image", "N/A")
        print(f"  {exec_name[:28]:<30} {image[:33]}")
    print()

    # Show DAG structure from the root component
    root_spec = pipeline_yaml.get("root", {})
    dag       = root_spec.get("dag", {}).get("tasks", {})
    print(f"  Pipeline DAG tasks ({len(dag)} steps):")
    print(f"  {'Task name':<30} {'Depends on'}")
    print(f"  {'─'*60}")
    for task_name, task_spec in dag.items():
        deps      = task_spec.get("dependentTasks", [])
        dep_str   = ", ".join(deps) if deps else "→ (starts immediately)"
        print(f"  {task_name:<30} {dep_str}")
    print()

    # Show one full task spec
    first_task_name = list(dag.keys())[0]
    first_task      = dag[first_task_name]
    print(f"  Full spec for first task ({first_task_name}):")
    print(f"    component ref: {first_task.get('componentRef', {}).get('name', 'N/A')}")
    inputs = first_task.get("inputs", {}).get("parameters", {})
    if inputs:
        print(f"    parameters ({len(inputs)}):")
        for pname, pval in list(inputs.items())[:3]:
            print(f"      {pname}: {pval}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Simulated local execution
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Simulated local pipeline execution")
print("━" * 65)
print()

print(f"  Note: In production, the pipeline runs in Kubernetes pods.")
print(f"  Here we simulate the same execution graph locally using the")
print(f"  compiled component functions directly (without containers).")
print()

# Execute each component in topological order (manually, for demo)
import sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

with tempfile.TemporaryDirectory() as workspace:
    # Replicate what each KFP pod would do
    step_results = {}

    # ── STEP 1: Ingest ────────────────────────────────────────────────────
    print(f"  [Step 1/5] Ingest Data")
    t0 = time.perf_counter()
    N, F = 2000, 20
    X_raw, y_raw = make_regression(n_samples=N, n_features=F, noise=25, random_state=42)
    df_raw = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(F)])
    df_raw["target"] = y_raw
    data_path = os.path.join(workspace, "raw_data.parquet")
    df_raw.to_parquet(data_path, index=False)
    t1 = (time.perf_counter() - t0) * 1000
    print(f"    ✓ Generated {N} rows × {F} features  ({t1:.0f}ms)")
    step_results["ingest"] = {"path": data_path, "rows": N, "cols": F+1}

    # ── STEP 2: Split & Normalise ─────────────────────────────────────────
    print(f"  [Step 2/5] Split & Normalise (cache hit: False)")
    t0 = time.perf_counter()
    df   = pd.read_parquet(data_path)
    X    = df.drop("target", axis=1)
    y    = df["target"]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler   = StandardScaler()
    X_tr_sc  = pd.DataFrame(scaler.fit_transform(X_tr), columns=X.columns)
    X_va_sc  = pd.DataFrame(scaler.transform(X_va), columns=X.columns)
    train_df = X_tr_sc.copy(); train_df["target"] = y_tr.values
    val_df   = X_va_sc.copy(); val_df["target"]   = y_va.values
    train_path = os.path.join(workspace, "train.parquet")
    val_path   = os.path.join(workspace, "val.parquet")
    train_df.to_parquet(train_path); val_df.to_parquet(val_path)
    t2 = (time.perf_counter() - t0) * 1000
    print(f"    ✓ Train={len(train_df)} rows | Val={len(val_df)} rows  ({t2:.0f}ms)")
    step_results["split"] = {"train_rows": len(train_df), "val_rows": len(val_df)}

    # ── STEP 3: Train ─────────────────────────────────────────────────────
    print(f"  [Step 3/5] Train Ridge Model (cache: disabled)")
    t0 = time.perf_counter()
    import pickle
    df_train = pd.read_parquet(train_path)
    X_t = df_train.drop("target", axis=1).values
    y_t = df_train["target"].values
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_t, y_t)
    train_r2 = model.score(X_t, y_t)
    model_path = os.path.join(workspace, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    t3 = (time.perf_counter() - t0) * 1000
    print(f"    ✓ Train R²={train_r2:.4f}  |  model saved ({os.path.getsize(model_path):,} bytes)  ({t3:.0f}ms)")
    step_results["train"] = {"train_r2": train_r2, "model_path": model_path}

    # ── STEP 4: Evaluate ──────────────────────────────────────────────────
    print(f"  [Step 4/5] Evaluate on Validation")
    t0 = time.perf_counter()
    with open(model_path, "rb") as f:
        model_loaded = pickle.load(f)
    df_val = pd.read_parquet(val_path)
    X_v = df_val.drop("target", axis=1).values
    y_v = df_val["target"].values
    y_pred  = model_loaded.predict(X_v)
    val_r2  = float(r2_score(y_v, y_pred))
    val_mse = float(mean_squared_error(y_v, y_pred))
    val_mae = float(mean_absolute_error(y_v, y_pred))
    t4 = (time.perf_counter() - t0) * 1000
    print(f"    ✓ val_r2={val_r2:.4f}  val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  ({t4:.0f}ms)")
    step_results["evaluate"] = {"val_r2": val_r2, "val_mse": val_mse, "val_mae": val_mae}

    # ── STEP 5: Conditional Register ──────────────────────────────────────
    min_r2 = 0.7
    print(f"  [Step 5/5] Register Model (condition: val_r2 > {min_r2})")
    t0 = time.perf_counter()
    if val_r2 > min_r2:
        registration = {
            "registry": "regression-model-registry",
            "version": "v1",
            "val_r2": val_r2, "val_mse": val_mse,
            "framework": "scikit-learn", "status": "registered",
        }
        t5 = (time.perf_counter() - t0) * 1000
        print(f"    ✓ Condition met (val_r2={val_r2:.4f} > {min_r2})  ({t5:.0f}ms)")
        print(f"    ✓ Registered: {json.dumps(registration, indent=8)}")
        step_results["register"] = registration
    else:
        print(f"    ✗ Condition NOT met (val_r2={val_r2:.4f} ≤ {min_r2}) → step skipped")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    total_ms = t1 + t2 + t3 + t4
    print(f"  Pipeline execution summary:")
    print(f"  {'Step':<25} {'Time':>10} {'Key result'}")
    print(f"  {'─'*55}")
    step_times = [t1, t2, t3, t4]
    step_names = ["Ingest", "Split/Preprocess", "Train", "Evaluate"]
    step_keys  = [f"{N} rows ingested",
                  f"train={len(train_df)}, val={len(val_df)}",
                  f"train_r2={train_r2:.4f}",
                  f"val_r2={val_r2:.4f}"]
    for name, t, key in zip(step_names, step_times, step_keys):
        print(f"  {name:<25} {t:>8.0f}ms {key}")
    print(f"  {'─'*55}")
    print(f"  {'Total':<25} {total_ms:>8.0f}ms")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Katib — Hyperparameter Search with Bayesian & ASHA Schedulers": {
        "description": (
            "Complete Katib hyperparameter tuning simulation. "
            "Experiment spec: objective, algorithm, parameters, budget. "
            "Parameter types: continuous, integer, categorical, log-scale. "
            "Algorithm comparison: Grid, Random, Bayesian (GP/TPE), CMA-ES. "
            "ASHA early stopping: bracket mechanics and compute savings. "
            "Metrics collection: stdout parser simulation. "
            "Trial management: parallel trials, failure handling. "
            "Results analysis: parameter importance, Pareto frontier. "
            "Best trial selection: objective metric ranking. "
            "Visualisation: parallel coordinates for HP analysis. "
            "Integration with KFP: Katib-inside-pipeline pattern. "
            "Production Katib Experiment YAML generation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import json
import time
import math
import itertools
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

print("=" * 65)
print("  KATIB — HYPERPARAMETER SEARCH ALGORITHMS")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Search space and objective function
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Search space definition and objective landscape")
print("━" * 65)
print()

@dataclass
class SearchSpace:
    """Mirrors Katib's parameter spec."""
    params: Dict[str, Any] = field(default_factory=dict)

    def sample_random(self) -> Dict[str, Any]:
        config = {}
        for name, spec in self.params.items():
            kind = spec["type"]
            if kind == "double":
                lo, hi = float(spec["min"]), float(spec["max"])
                if spec.get("log_scale"):
                    config[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                else:
                    config[name] = float(rng.uniform(lo, hi))
            elif kind == "int":
                lo, hi = int(spec["min"]), int(spec["max"])
                config[name] = int(rng.integers(lo, hi + 1))
            elif kind == "categorical":
                config[name] = rng.choice(spec["list"])
        return config

    def grid_configs(self, grid_pts: int = 3) -> List[Dict]:
        grids = {}
        for name, spec in self.params.items():
            kind = spec["type"]
            if kind == "double":
                lo, hi = float(spec["min"]), float(spec["max"])
                if spec.get("log_scale"):
                    grids[name] = list(np.exp(np.linspace(np.log(lo), np.log(hi), grid_pts)))
                else:
                    grids[name] = list(np.linspace(lo, hi, grid_pts))
            elif kind == "int":
                lo, hi = int(spec["min"]), int(spec["max"])
                step = max(1, (hi - lo) // (grid_pts - 1))
                grids[name] = list(range(lo, hi + 1, step))[:grid_pts]
            elif kind == "categorical":
                grids[name] = spec["list"][:grid_pts]
        keys   = list(grids.keys())
        combos = list(itertools.product(*[grids[k] for k in keys]))
        return [{keys[i]: c[i] for i in range(len(keys))} for c in combos]


# The Kubeflow Katib search space definition
search_space = SearchSpace(params={
    "learning-rate": {
        "type": "double", "min": "1e-5", "max": "1e-1", "log_scale": True,
    },
    "batch-size": {
        "type": "categorical", "list": ["32", "64", "128", "256"],
    },
    "num-layers": {
        "type": "int", "min": "2", "max": "8",
    },
    "dropout": {
        "type": "double", "min": "0.0", "max": "0.5", "log_scale": False,
    },
    "weight-decay": {
        "type": "double", "min": "1e-6", "max": "1e-2", "log_scale": True,
    },
})

print(f"  Search space ({len(search_space.params)} hyperparameters):")
print(f"  {'Parameter':<20} {'Type':<12} {'Range/Options'}")
print(f"  {'─'*60}")
for name, spec in search_space.params.items():
    if spec["type"] == "categorical":
        opts = spec["list"]
        print(f"  {name:<20} {'categorical':<12} {opts}")
    elif spec.get("log_scale"):
        print(f"  {name:<20} {spec['type']+' (log)':<12} [{spec['min']}, {spec['max']}]")
    else:
        print(f"  {name:<20} {spec['type']:<12} [{spec['min']}, {spec['max']}]")
print()

def objective_function(config: Dict, max_epochs: int = 30) -> Tuple[float, List[float]]:
    """
    Simulated training loss curve for a given HP configuration.
    Returns (final_val_accuracy, loss_history).
    Models typical neural network training dynamics realistically.
    """
    lr      = float(config["learning-rate"])
    bs      = int(config["batch-size"])
    layers  = int(config["num-layers"])
    dropout = float(config["dropout"])
    wd      = float(config["weight-decay"])

    # Optimal values (ground truth of the black-box function)
    lr_opt, bs_opt = 3e-3, 64
    layers_opt, dropout_opt = 4, 0.15
    wd_opt = 1e-4

    # Penalty for deviation from optimum
    def log_dist(x, opt):
        return abs(math.log(max(x, 1e-10)) - math.log(max(opt, 1e-10)))

    penalty = (
        log_dist(lr, lr_opt)      * 0.25 +
        log_dist(bs, bs_opt)      * 0.10 +
        abs(layers - layers_opt)  * 0.08 +
        abs(dropout - dropout_opt)* 0.30 +
        log_dist(wd, wd_opt)      * 0.12
    )
    peak_accuracy = max(0.50, 0.96 - penalty * 0.18)
    noise_scale   = 0.015 + penalty * 0.005

    # Generate training curve: fast rise → plateau with noise
    curve = []
    seed  = int(sum(ord(c) for c in str(config)))
    loc_rng = np.random.default_rng(seed)
    for epoch in range(max_epochs):
        progress   = 1.0 - math.exp(-4.0 * (epoch + 1) / max_epochs)
        noisy_acc  = peak_accuracy * progress + loc_rng.normal(0, noise_scale * (1 - progress + 0.3))
        curve.append(max(0.0, min(1.0, noisy_acc)))

    return curve[-1], curve


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Grid search and random search baselines
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Grid search vs Random Search baselines")
print("━" * 65)
print()

# Grid search (limited to 3 options per param to keep it finite)
grid_configs = search_space.grid_configs(grid_pts=2)
print(f"  Grid search (2 values per param):")
print(f"    Total configs: {len(grid_configs)}  ({2}^{len(search_space.params)-1} × 2 categorical)")

t0 = time.perf_counter()
grid_results = []
for cfg in grid_configs[:20]:   # limit to 20 for speed
    acc, _ = objective_function(cfg, max_epochs=20)
    grid_results.append((acc, cfg))
grid_results.sort(key=lambda x: -x[0])
t_grid = (time.perf_counter() - t0) * 1000

best_grid_acc, best_grid_cfg = grid_results[0]
print(f"    Best accuracy: {best_grid_acc:.4f}")
print(f"    Compute time:  {t_grid:.0f}ms  (simulated)")
print()

# Random search
N_RANDOM = 40
t0 = time.perf_counter()
random_results = []
for _ in range(N_RANDOM):
    cfg = search_space.sample_random()
    acc, _ = objective_function(cfg, max_epochs=20)
    random_results.append((acc, cfg))
random_results.sort(key=lambda x: -x[0])
t_rand = (time.perf_counter() - t0) * 1000

best_rand_acc = random_results[0][0]
print(f"  Random Search ({N_RANDOM} trials):")
print(f"    Best accuracy:   {best_rand_acc:.4f}")
print(f"    Median accuracy: {np.median([r[0] for r in random_results]):.4f}")
print(f"    Compute time:    {t_rand:.0f}ms")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Bayesian Optimisation (GP surrogate)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Bayesian Optimisation: GP surrogate model")
print("━" * 65)
print()

class GPBayesianSearch:
    """
    Simplified Gaussian Process Bayesian Optimisation.
    Demonstrates the acquisition function (Expected Improvement) concept.
    """
    def __init__(self, search_space: SearchSpace, n_init: int = 5):
        self.space     = search_space
        self.observed  = []   # list of (config_vector, accuracy)
        self.n_init    = n_init
        self._all_candidates = None

    def _config_to_vector(self, config: Dict) -> np.ndarray:
        vec = []
        for name, spec in self.space.params.items():
            val = config[name]
            if spec["type"] == "categorical":
                opts = spec["list"]
                idx  = opts.index(str(val)) if str(val) in opts else 0
                vec.append(idx / max(len(opts) - 1, 1))
            elif spec.get("log_scale"):
                lo = math.log(float(spec["min"]))
                hi = math.log(float(spec["max"]))
                vec.append((math.log(max(float(val), 1e-12)) - lo) / max(hi - lo, 1e-8))
            else:
                lo, hi = float(spec["min"]), float(spec["max"])
                vec.append((float(val) - lo) / max(hi - lo, 1e-8))
        return np.array(vec)

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray, l: float = 0.5) -> float:
        return float(np.exp(-np.sum((x1 - x2)**2) / (2 * l**2)))

    def _gp_predict(self, x_new: np.ndarray) -> Tuple[float, float]:
        """Returns (mean, std) of GP posterior at x_new."""
        if not self.observed:
            return 0.5, 0.1
        X_obs = np.array([self._config_to_vector(c) for c, _ in self.observed])
        y_obs = np.array([y for _, y in self.observed])
        sigma_n = 1e-3  # noise variance
        n = len(X_obs)
        # Build kernel matrix K
        K = np.array([[self._rbf_kernel(X_obs[i], X_obs[j])
                       for j in range(n)] for i in range(n)])
        K += sigma_n * np.eye(n)
        k_star = np.array([self._rbf_kernel(x_new, X_obs[i]) for i in range(n)])
        try:
            K_inv  = np.linalg.inv(K)
            mu     = float(k_star @ K_inv @ y_obs)
            sigma2 = max(0.0, float(1.0 - k_star @ K_inv @ k_star))
        except np.linalg.LinAlgError:
            mu, sigma2 = float(np.mean(y_obs)), 0.01
        return mu, math.sqrt(max(sigma2, 1e-10))

    def _expected_improvement(self, mu: float, sigma: float, y_best: float,
                               xi: float = 0.01) -> float:
        """EI acquisition function."""
        from scipy.stats import norm as sp_norm
        z = (mu - y_best - xi) / max(sigma, 1e-10)
        return float((mu - y_best - xi) * sp_norm.cdf(z) + sigma * sp_norm.pdf(z))

    def suggest_next(self, n_candidates: int = 200) -> Dict:
        """Use EI to select the most promising next config."""
        # Initial random exploration
        if len(self.observed) < self.n_init:
            return self.space.sample_random()

        y_best     = max(y for _, y in self.observed)
        candidates = [self.space.sample_random() for _ in range(n_candidates)]
        best_ei, best_cfg = -1, candidates[0]

        for cfg in candidates:
            x_vec     = self._config_to_vector(cfg)
            mu, sigma = self._gp_predict(x_vec)
            ei        = self._expected_improvement(mu, sigma, y_best)
            if ei > best_ei:
                best_ei  = ei
                best_cfg = cfg

        return best_cfg

    def update(self, config: Dict, accuracy: float):
        self.observed.append((config, accuracy))


N_BAYES = 30
gp_search = GPBayesianSearch(search_space, n_init=5)

t0 = time.perf_counter()
bayes_results = []
print(f"  Bayesian Optimisation ({N_BAYES} trials, GP with EI acquisition):")
print(f"  {'Trial':>7} {'Acc':>8} {'Best so far':>12} {'Phase':>15}")
print(f"  {'─'*46}")

for trial in range(N_BAYES):
    cfg = gp_search.suggest_next()
    acc, _ = objective_function(cfg, max_epochs=20)
    gp_search.update(cfg, acc)
    bayes_results.append((acc, cfg))
    best_so_far = max(r[0] for r in bayes_results)
    phase = "explore (random)" if trial < 5 else "exploit (EI guided)"
    if trial % 5 == 0 or trial == N_BAYES - 1:
        print(f"  {trial+1:>7} {acc:>8.4f} {best_so_far:>12.4f} {phase:>15}")

t_bayes = (time.perf_counter() - t0) * 1000
best_bayes_acc = max(r[0] for r in bayes_results)
print()
print(f"  Best accuracy:  {best_bayes_acc:.4f}")
print(f"  Compute time:   {t_bayes:.0f}ms")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: ASHA early stopping
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — ASHA: Asynchronous Successive Halving")
print("━" * 65)
print()

class ASHAScheduler:
    """
    ASHA: trials that don't rank in the top 1/eta fraction at each rung
    are stopped early. Asynchronous: no waiting for bracket completion.
    """
    def __init__(self, max_t: int = 30, grace_period: int = 5,
                 reduction_factor: int = 3):
        self.max_t   = max_t
        self.grace   = grace_period
        self.eta     = reduction_factor
        # Compute rungs: grace, grace*eta, grace*eta^2, ..., max_t
        self.rungs   = []
        r = grace_period
        while r <= max_t:
            self.rungs.append(r)
            r *= reduction_factor
        if self.rungs[-1] < max_t:
            self.rungs.append(max_t)
        self.rung_data = {r: [] for r in self.rungs}

    def should_stop(self, trial_id: int, epoch: int, metric: float) -> bool:
        """Returns True if this trial should be stopped at this epoch."""
        # At each rung, check if metric is in the bottom (1 - 1/eta) fraction
        for rung in self.rungs:
            if epoch == rung:
                self.rung_data[rung].append((metric, trial_id))
                n = len(self.rung_data[rung])
                n_keep = max(1, n // self.eta)
                sorted_metrics = sorted([m for m, _ in self.rung_data[rung]], reverse=True)
                threshold = sorted_metrics[n_keep - 1] if n >= self.eta else -float("inf")
                return metric < threshold
        return False

N_ASHA = 40
MAX_EPOCHS = 30
GRACE_PERIOD = 5
ETA = 3

asha = ASHAScheduler(max_t=MAX_EPOCHS, grace_period=GRACE_PERIOD,
                      reduction_factor=ETA)

print(f"  ASHA config: max_t={MAX_EPOCHS}, grace={GRACE_PERIOD}, η={ETA}")
print(f"  Rung schedule: {asha.rungs}")
print(f"  At each rung: keep top 1/{ETA} = {100//ETA:.0f}% of trials")
print()

t0 = time.perf_counter()
asha_results  = []
total_epochs_used = 0
total_epochs_naive = N_ASHA * MAX_EPOCHS

for trial_id in range(N_ASHA):
    cfg     = search_space.sample_random()
    _, curve = objective_function(cfg, max_epochs=MAX_EPOCHS)
    stopped_at = MAX_EPOCHS
    stopped    = False

    for epoch, acc in enumerate(curve, start=1):
        total_epochs_used += 1
        if asha.should_stop(trial_id, epoch, acc):
            stopped_at = epoch
            stopped    = True
            break

    final_acc = curve[stopped_at - 1]
    asha_results.append({
        "trial_id":   trial_id,
        "final_acc":  final_acc,
        "stopped_at": stopped_at,
        "stopped":    stopped,
        "config":     cfg,
    })

t_asha = (time.perf_counter() - t0) * 1000

# Analysis
stopped_trials = [r for r in asha_results if r["stopped"]]
full_trials    = [r for r in asha_results if not r["stopped"]]
best_asha      = max(asha_results, key=lambda x: x["final_acc"])

print(f"  ASHA results ({N_ASHA} trials):")
print(f"    Trials stopped early:      {len(stopped_trials)}/{N_ASHA}")
print(f"    Trials completed:          {len(full_trials)}/{N_ASHA}")
print(f"    Total epochs used:         {total_epochs_used:,}")
print(f"    Naive (no stopping):       {total_epochs_naive:,}")
compute_savings = (1 - total_epochs_used / total_epochs_naive) * 100
print(f"    Compute savings:           {compute_savings:.1f}%")
print()

# Epoch distribution of stopped trials
stopped_epochs = [r["stopped_at"] for r in stopped_trials]
if stopped_epochs:
    print(f"  Stopping epoch distribution:")
    for rung in asha.rungs:
        n_at_rung = sum(1 for e in stopped_epochs if e == rung)
        bar = "█" * n_at_rung
        print(f"    Rung epoch={rung:>3}: {n_at_rung:>3} trials stopped  {bar}")
    print()

print(f"  Best trial:")
print(f"    Accuracy:     {best_asha['final_acc']:.4f}")
print(f"    Stopped at:   epoch {best_asha['stopped_at']}")
best_lr = best_asha['config']['learning-rate']
print(f"    lr={best_lr:.4e}  batch={best_asha['config']['batch-size']}  "
      f"layers={best_asha['config']['num-layers']}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Algorithm comparison and Katib YAML generation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Algorithm comparison and Katib Experiment YAML")
print("━" * 65)
print()

all_accs = {
    "Grid (20 trials)":      [r[0] for r in grid_results[:20]],
    f"Random ({N_RANDOM})":  [r[0] for r in random_results],
    f"Bayesian GP ({N_BAYES})": [r[0] for r in bayes_results],
    f"ASHA ({N_ASHA})":      [r["final_acc"] for r in asha_results],
}

print(f"  Algorithm comparison (best, mean, p75):")
print(f"  {'Algorithm':<25} {'Best':>8} {'Mean':>8} {'p75':>8} {'Compute'}")
print(f"  {'─'*62}")
compute_map = {
    "Grid (20 trials)":     f"~{20*20}ep",
    f"Random ({N_RANDOM})": f"~{N_RANDOM*20}ep",
    f"Bayesian GP ({N_BAYES})": f"~{N_BAYES*20}ep",
    f"ASHA ({N_ASHA})":     f"~{total_epochs_used}ep",
}
for name, accs in all_accs.items():
    arr = np.array(accs)
    print(f"  {name:<25} {arr.max():>8.4f} {arr.mean():>8.4f} "
          f"{np.percentile(arr,75):>8.4f} {compute_map[name]:>10}")
print()

# Generate Katib Experiment YAML
katib_yaml = {
    "apiVersion": "kubeflow.org/v1beta1",
    "kind":       "Experiment",
    "metadata":   {"name": "nn-hpo-experiment", "namespace": "user-alice"},
    "spec": {
        "objective": {
            "type":                "maximize",
            "goal":                0.95,
            "objectiveMetricName": "val-accuracy",
        },
        "algorithm": {"algorithmName": "bayesianoptimization"},
        "parallelTrialCount": 4,
        "maxTrialCount":      30,
        "maxFailedTrialCount": 5,
        "parameters": [
            {"name": "learning-rate", "parameterType": "double",
             "feasibleSpace": {"min": "1e-5", "max": "1e-1",
                               "step": "0", "list": []}},
            {"name": "batch-size", "parameterType": "categorical",
             "feasibleSpace": {"list": ["32", "64", "128", "256"]}},
            {"name": "num-layers", "parameterType": "int",
             "feasibleSpace": {"min": "2", "max": "8"}},
            {"name": "dropout", "parameterType": "double",
             "feasibleSpace": {"min": "0.0", "max": "0.5"}},
        ],
        "metricsCollectorSpec": {
            "collector": {"kind": "StdOut"},
            "source": {
                "filter": {"metricsFormat":
                           ["val-accuracy=([0-9\\.]+)",
                            "val-loss=([0-9\\.]+)"]},
            }
        },
        "trialTemplate": {
            "primaryContainerName": "training-container",
            "trialParameters": [
                {"name": "learningRate",  "reference": "learning-rate"},
                {"name": "batchSize",     "reference": "batch-size"},
                {"name": "numLayers",     "reference": "num-layers"},
                {"name": "dropout",       "reference": "dropout"},
            ],
            "trialSpec": {
                "apiVersion": "kubeflow.org/v1",
                "kind":       "PyTorchJob",
                "spec": {
                    "pytorchReplicaSpecs": {
                        "Master": {
                            "replicas": 1,
                            "template": {
                                "spec": {
                                    "containers": [{
                                        "name":    "training-container",
                                        "image":   "my-registry/nn-trainer:v1.2.0",
                                        "command": ["python", "train.py"],
                                        "args": [
                                            "--lr=$(trialParameters.learningRate)",
                                            "--batch-size=$(trialParameters.batchSize)",
                                            "--num-layers=$(trialParameters.numLayers)",
                                            "--dropout=$(trialParameters.dropout)",
                                        ],
                                        "resources": {
                                            "limits": {"nvidia.com/gpu": "1"}
                                        }
                                    }]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

yaml_str = yaml.dump(katib_yaml, default_flow_style=False, sort_keys=False)
lines    = yaml_str.split("\n")
print(f"  Generated Katib Experiment YAML ({len(lines)} lines):")
print(f"  {'─'*55}")
for line in lines[:35]:
    print(f"  {line}")
if len(lines) > 35:
    print(f"  ... ({len(lines)-35} more lines)")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Training Operator — PyTorchJob Distributed Training Patterns": {
        "description": (
            "Kubeflow Training Operator distributed training deep dive. "
            "PyTorchJob YAML: Master/Worker replica specs. "
            "Environment variables: MASTER_ADDR, WORLD_SIZE, RANK, LOCAL_RANK. "
            "torchrun integration: nproc_per_node, NCCL backend. "
            "DDP gradient synchronisation simulation across workers. "
            "Effective batch size scaling with world size. "
            "Checkpoint recovery: saving/loading in distributed context. "
            "TFJob topology: PS vs AllReduce comparison. "
            "Gang scheduling: why all workers must start together. "
            "Resource utilisation analysis: GPU memory and compute. "
            "Multi-node training topology YAML generation. "
            "Failure recovery: max_restarts and fault tolerance patterns."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import json
import yaml
import math
import os
import tempfile
import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    print(f"  PyTorch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "torch", "--quiet"], check=True)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

print("=" * 65)
print("  TRAINING OPERATOR — PYTORCH DISTRIBUTED TRAINING")
print("=" * 65)
print()

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: PyTorchJob YAML and environment variable injection
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — PyTorchJob: YAML spec and env variable injection")
print("━" * 65)
print()

def generate_pytorch_job_yaml(
    job_name:      str,
    namespace:     str,
    image:         str,
    n_workers:     int,
    gpus_per_pod:  int = 1,
    cpus_per_pod:  int = 4,
    memory_gi:     int = 16,
    n_proc_per_node: int = None,
) -> dict:
    n_proc = n_proc_per_node or gpus_per_pod

    def make_replica_spec(replica_type: str) -> dict:
        return {
            "replicas": 1 if replica_type == "Master" else n_workers,
            "restartPolicy": "OnFailure",
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"   # skip service mesh for training
                    }
                },
                "spec": {
                    "containers": [{
                        "name":  "pytorch",
                        "image": image,
                        "imagePullPolicy": "Always",
                        "command": ["torchrun"],
                        "args": [
                            f"--nproc_per_node={n_proc}",
                            "train.py",
                            "--epochs=50",
                            "--checkpoint-dir=/checkpoints",
                        ],
                        "resources": {
                            "requests": {
                                "cpu":               f"{cpus_per_pod}",
                                "memory":            f"{memory_gi}Gi",
                                **({"nvidia.com/gpu": str(gpus_per_pod)} if gpus_per_pod > 0 else {}),
                            },
                            "limits": {
                                "cpu":               f"{cpus_per_pod * 2}",
                                "memory":            f"{memory_gi * 2}Gi",
                                **({"nvidia.com/gpu": str(gpus_per_pod)} if gpus_per_pod > 0 else {}),
                            },
                        },
                        "volumeMounts": [
                            {"name": "checkpoints", "mountPath": "/checkpoints"},
                            {"name": "training-data", "mountPath": "/data"},
                        ],
                        "env": [
                            {"name": "NCCL_DEBUG", "value": "INFO"},
                            {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
                            {"name": "CUDA_VISIBLE_DEVICES",
                             "value": ",".join(str(i) for i in range(gpus_per_pod))},
                        ],
                    }],
                    "volumes": [
                        {"name": "checkpoints",
                         "persistentVolumeClaim": {"claimName": f"{job_name}-checkpoints"}},
                        {"name": "training-data",
                         "persistentVolumeClaim": {"claimName": "training-data-pvc"}},
                    ],
                }
            }
        }

    return {
        "apiVersion": "kubeflow.org/v1",
        "kind":       "PyTorchJob",
        "metadata":   {"name": job_name, "namespace": namespace},
        "spec": {
            "nprocPerNode": str(n_proc),
            "pytorchReplicaSpecs": {
                "Master": make_replica_spec("Master"),
                "Worker": make_replica_spec("Worker"),
            }
        }
    }

# Single-node multi-GPU
job_single = generate_pytorch_job_yaml(
    "bert-finetune-singlenode", "user-alice",
    "my-registry/bert-trainer:v1.0.0",
    n_workers=0, gpus_per_pod=4, n_proc_per_node=4,
)
# Multi-node multi-GPU
job_multi = generate_pytorch_job_yaml(
    "llm-pretrain-multinode", "user-alice",
    "my-registry/llm-trainer:v2.1.0",
    n_workers=7, gpus_per_pod=8, cpus_per_pod=96,
    memory_gi=512, n_proc_per_node=8,
)

print(f"  PyTorchJob configurations:")
print()
configs_to_show = [
    ("Single-node (4 GPU)",  job_single, 1,  4,  4),
    ("Multi-node (8×8 GPU)", job_multi,  8, 64, 64),
]
for label, job, n_nodes, total_gpus, world_size in configs_to_show:
    print(f"  Config: {label}")
    print(f"    Job name:      {job['metadata']['name']}")
    print(f"    Nodes:         {n_nodes}")
    print(f"    Total GPUs:    {total_gpus}")
    print(f"    World size:    {world_size}  (processes)")
    master_spec = job["spec"]["pytorchReplicaSpecs"]["Master"]
    worker_spec = job["spec"]["pytorchReplicaSpecs"].get("Worker", master_spec)
    cpu   = worker_spec["template"]["spec"]["containers"][0]["resources"]["requests"]["cpu"]
    mem   = worker_spec["template"]["spec"]["containers"][0]["resources"]["requests"]["memory"]
    print(f"    CPU/pod:       {cpu}")
    print(f"    Memory/pod:    {mem}")
    print()

# Environment variables injected by the Training Operator
print(f"  Environment variables injected by Training Operator:")
env_vars = {
    "MASTER_ADDR":    "bert-finetune-singlenode-master-0.bert-finetune-singlenode.svc",
    "MASTER_PORT":    "23456",
    "WORLD_SIZE":     "4",
    "RANK":           "0 (master), 1-3 (workers)",
    "LOCAL_RANK":     "GPU index within the pod (0 to nproc_per_node-1)",
    "PET_NNODES":     "Number of nodes",
    "PET_NODE_RANK":  "This node's rank among nodes",
}
print(f"  {'Variable':<18} {'Value'}")
print(f"  {'─'*65}")
for k, v in env_vars.items():
    print(f"  {k:<18} {v}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Simulated DDP training across multiple workers
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — DDP simulation: gradient AllReduce across workers")
print("━" * 65)
print()

class SimpleNet(nn.Module):
    def __init__(self, in_dim=32, hidden=64, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.ln  = nn.LayerNorm(hidden)

    def forward(self, x):
        x = F.gelu(self.ln(self.fc1(x)))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


def simulate_ddp_step(
    world_size:   int,
    model:        nn.Module,
    global_batch: torch.Tensor,
    targets:      torch.Tensor,
    lr:           float = 1e-3,
) -> Dict:
    """
    Simulate one DDP training step:
    1. Each worker gets its own mini-batch (shard of global batch)
    2. Each worker computes gradients independently
    3. Gradients are AllReduced (averaged) across workers
    4. All workers apply the same gradient update
    Returns diagnostics.
    """
    # Split global batch across world_size workers
    shard_size = len(global_batch) // world_size
    results    = []
    grad_buffers = {name: [] for name, _ in model.named_parameters()}

    for rank in range(world_size):
        # Each rank processes its own shard
        start = rank * shard_size
        end   = start + shard_size
        x_shard = global_batch[start:end]
        y_shard = targets[start:end]

        # Forward + backward on shard
        model.zero_grad()
        preds = model(x_shard).squeeze(-1)
        loss  = F.mse_loss(preds, y_shard)
        loss.backward()

        # Collect gradients from this rank
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_buffers[name].append(param.grad.clone())
        results.append(float(loss.item()))

    # AllReduce: average gradients across all workers
    grad_norms_before = {}
    grad_norms_after  = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in grad_buffers and grad_buffers[name]:
                grads = torch.stack(grad_buffers[name])
                grad_norms_before[name] = float(grads[0].norm())
                # Average (AllReduce SUM / world_size)
                averaged_grad = grads.mean(dim=0)
                param.grad = averaged_grad
                grad_norms_after[name] = float(averaged_grad.norm())

    # Apply update (same on all workers since gradients are now identical)
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= lr * param.grad

    return {
        "per_worker_losses":  results,
        "global_loss":        np.mean(results),
        "grad_norm_before":   float(np.mean(list(grad_norms_before.values()))),
        "grad_norm_after":    float(np.mean(list(grad_norms_after.values()))),
    }


# Generate synthetic data
IN_DIM, N_SAMPLES = 32, 2048
X_global = torch.randn(N_SAMPLES, IN_DIM)
true_W   = torch.randn(IN_DIM)
y_global = (X_global @ true_W + 0.1 * torch.randn(N_SAMPLES)).detach()

# Compare training across different world sizes
print(f"  Comparing DDP across world sizes (same global batch={N_SAMPLES}):")
print()
print(f"  {'World size':>12} {'Effective bs':>15} {'Steps/epoch':>13} {'Global loss':>14} {'Time/step':>12}")
print(f"  {'─'*70}")

for world_size in [1, 2, 4, 8]:
    model = SimpleNet(IN_DIM, 64, 1)
    times = []

    for step in range(5):
        t0 = time.perf_counter()
        result = simulate_ddp_step(world_size, model, X_global, y_global, lr=1e-3)
        times.append((time.perf_counter() - t0) * 1000)

    eff_batch   = N_SAMPLES // world_size * world_size   # all workers combined
    steps_epoch = N_SAMPLES // (N_SAMPLES // world_size * world_size)
    avg_t       = np.mean(times)
    print(f"  {world_size:>12} {eff_batch:>15} {steps_epoch:>13} "
          f"{result['global_loss']:>14.6f} {avg_t:>12.2f}ms")

print()
print(f"  Key insight: with DDP, effective batch = per_worker_batch × world_size.")
print(f"  Larger effective batch → may need LR scaling: lr = base_lr × world_size")
print(f"  Linear scaling rule (Goyal et al., Facebook, 2017)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Checkpointing in distributed training
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Checkpoint management: save, recover, resume")
print("━" * 65)
print()

def save_checkpoint(model, optimizer, epoch, step, loss, path):
    """Save full training state for recovery after pod restart."""
    state = {
        "epoch":          epoch,
        "step":           step,
        "loss":           loss,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_version":  torch.__version__,
    }
    torch.save(state, path)
    return os.path.getsize(path)


def load_checkpoint(model, optimizer, path):
    """Restore training state from a checkpoint."""
    state     = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["epoch"], state["step"], state["loss"]


model     = SimpleNet(IN_DIM, 64, 1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Simulate training and save checkpoints
with tempfile.TemporaryDirectory() as tmp:
    checkpoint_paths = []
    losses           = []

    print(f"  Training with checkpoint every 5 epochs:")
    print(f"  {'Epoch':>7} {'Loss':>10} {'Checkpoint':>15} {'Size (KB)':>12}")
    print(f"  {'─'*48}")

    for epoch in range(20):
        # Mini training step
        preds = model(X_global[:256]).squeeze(-1)
        loss  = F.mse_loss(preds, y_global[:256])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(tmp, f"checkpoint_epoch_{epoch+1:04d}.pt")
            size_bytes = save_checkpoint(model, optimizer, epoch+1,
                                          (epoch+1)*8, float(loss), ckpt_path)
            checkpoint_paths.append(ckpt_path)
            print(f"  {epoch+1:>7} {float(loss):>10.6f} {'saved':>15} "
                  f"{size_bytes/1024:>12.1f}")

    # Simulate pod restart: reload from last checkpoint
    print()
    print(f"  Simulating pod restart (worker failure):")
    fresh_model     = SimpleNet(IN_DIM, 64, 1)
    fresh_optimizer = optim.AdamW(fresh_model.parameters(), lr=1e-3)
    start_epoch, start_step, ckpt_loss = load_checkpoint(
        fresh_model, fresh_optimizer, checkpoint_paths[-1]
    )
    print(f"    Loaded checkpoint: epoch={start_epoch}, loss={ckpt_loss:.6f}")

    # Verify restored model matches original
    with torch.no_grad():
        orig_out    = model(X_global[:4]).squeeze(-1)
        restored_out = fresh_model(X_global[:4]).squeeze(-1)
    diff = (orig_out - restored_out).abs().max().item()
    print(f"    Output difference (should be 0): {diff:.2e}  ✅")
    print(f"    Resume training from epoch {start_epoch + 1}")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Resource utilisation analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — GPU memory analysis for distributed strategies")
print("━" * 65)
print()

def estimate_memory_gb(
    model_params_b:  float,
    batch_size:      int,
    seq_len:         int = 512,
    hidden:          int = 1024,
    strategy:        str = "DDP",
    world_size:      int = 8,
    dtype_bytes:     int = 2,   # bf16/fp16
) -> Dict:
    """Estimate per-GPU memory for different parallelism strategies."""
    # Model weights
    weights_gb = model_params_b * dtype_bytes / 1e9

    # Gradients (same size as weights for DDP)
    grads_gb   = weights_gb

    # Optimiser states (Adam: 2× model size in fp32)
    optim_gb   = model_params_b * 4 * 2 / 1e9  # fp32 m + v

    # Activations (rough estimate: batch × seq × hidden × n_layers × dtype)
    activations_gb = batch_size * seq_len * hidden * 4 / 1e9  # simplified

    if strategy == "DDP":
        per_gpu = weights_gb + grads_gb + optim_gb + activations_gb
    elif strategy == "FSDP":
        # Shard weights, grads, optim across world_size
        per_gpu = (weights_gb + grads_gb + optim_gb) / world_size + activations_gb
    elif strategy == "Inference":
        per_gpu = weights_gb + activations_gb / 4
    else:
        per_gpu = weights_gb + grads_gb + optim_gb + activations_gb

    return {
        "strategy":      strategy,
        "world_size":    world_size,
        "weights_gb":    weights_gb,
        "grads_gb":      grads_gb,
        "optim_gb":      optim_gb,
        "activations_gb": activations_gb,
        "per_gpu_total": per_gpu,
    }

models_to_analyse = [
    ("BERT-base (110M)",   110e6,  8),
    ("LLaMA-3 8B",         8e9,   64),
    ("LLaMA-3 70B",        70e9, 512),
]

print(f"  GPU memory estimates (bf16, batch=32, seq=512):")
print()
print(f"  {'Model':<22} {'Strategy':<10} {'World size':>12} {'Per GPU (GB)':>14} {'Fits on'}")
print(f"  {'─'*72}")
for model_name, params, seq_l in models_to_analyse:
    for strategy, ws in [("DDP", 8), ("FSDP", 8), ("FSDP", 64)]:
        mem = estimate_memory_gb(params, batch_size=4, seq_len=seq_l,
                                  hidden=4096, strategy=strategy,
                                  world_size=ws, dtype_bytes=2)
        per_gpu = mem["per_gpu_total"]
        if per_gpu <= 24:
            fits = "RTX 4090 (24GB)"
        elif per_gpu <= 40:
            fits = "A100-40GB"
        elif per_gpu <= 80:
            fits = "A100-80GB / H100"
        elif per_gpu <= 141:
            fits = "H100 SXM5"
        else:
            fits = "Need more shards"
        print(f"  {model_name:<22} {strategy:<10} {ws:>12} {per_gpu:>14.1f} {fits}")
    print()
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · KServe & MLOps — Model Serving, CI/CD Pipelines & Observability": {
        "description": (
            "KServe model serving and production MLOps patterns. "
            "InferenceService YAML: predictor, transformer, explainer. "
            "Canary deployment: traffic shifting across model versions. "
            "Knative autoscaling: scale-to-zero and scale-from-zero. "
            "V2 inference protocol: request/response format. "
            "Batch inference: throughput vs latency tradeoff. "
            "Model monitoring: input drift detection simulation. "
            "Complete MLOps pipeline: train → evaluate → register → serve. "
            "CI/CD for ML: trigger → test → build → deploy pattern. "
            "Observability: Prometheus metrics, Grafana dashboards. "
            "Production checklist verification. "
            "Kubeflow component decision guide."
        ),
        "language": "python",
        "code": '''
import numpy as np
import json
import yaml
import time
import math
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

print("=" * 65)
print("  KSERVE & MLOPS — MODEL SERVING AND PRODUCTION PATTERNS")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: InferenceService YAML and canary deployment
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — KServe InferenceService: canary deployment pattern")
print("━" * 65)
print()

def generate_inference_service(
    name:            str,
    namespace:       str,
    model_uri:       str,
    model_format:    str,
    canary_pct:      Optional[int] = None,
    min_replicas:    int = 0,    # 0 = scale-to-zero
    max_replicas:    int = 10,
    target_rps:      int = 5,    # scale when > target reqs/replica
    n_gpus:          int = 0,
    transformer_img: Optional[str] = None,
) -> dict:
    """Generate a KServe InferenceService manifest."""
    predictor_spec = {
        "model": {
            "modelFormat": {"name": model_format},
            "storageUri":  model_uri,
            "resources": {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits":   {
                    "cpu": "4", "memory": "8Gi",
                    **({"nvidia.com/gpu": str(n_gpus)} if n_gpus else {}),
                },
            },
        }
    }
    if canary_pct is not None:
        predictor_spec["canaryTrafficPercent"] = canary_pct

    spec = {
        "predictor": predictor_spec,
    }

    if transformer_img:
        spec["transformer"] = {
            "containers": [{
                "name":  "transformer",
                "image": transformer_img,
                "env": [
                    {"name": "PREDICTOR_URL", "value": f"http://{name}-predictor-default"},
                ],
                "resources": {
                    "requests": {"cpu": "0.5", "memory": "512Mi"},
                    "limits":   {"cpu": "2",   "memory": "2Gi"},
                },
            }]
        }

    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind":       "InferenceService",
        "metadata": {
            "name":      name,
            "namespace": namespace,
            "annotations": {
                "autoscaling.knative.dev/target":        str(target_rps),
                "autoscaling.knative.dev/minScale":      str(min_replicas),
                "autoscaling.knative.dev/maxScale":      str(max_replicas),
                "autoscaling.knative.dev/class":         "kpa.autoscaling.knative.dev",
            }
        },
        "spec": spec,
    }

# V3 model — current production
isvc_v3 = generate_inference_service(
    "fraud-detector", "user-alice",
    model_uri    = "gs://models/fraud/v3/",
    model_format = "sklearn",
    min_replicas = 1,   # always-on for production
    max_replicas = 20,
    target_rps   = 10,
)

# V4 model — canary rollout (10% → 30% → 50% → 100%)
isvc_v4_canary = generate_inference_service(
    "fraud-detector", "user-alice",
    model_uri    = "gs://models/fraud/v4/",
    model_format = "sklearn",
    canary_pct   = 10,   # send 10% of traffic here
    min_replicas = 0,    # can scale to zero when traffic is low
    max_replicas = 20,
    target_rps   = 10,
    transformer_img = "my-registry/fraud-transformer:v2",
)

print(f"  Canary deployment pattern:")
print(f"  {'Stage':<20} {'Canary %':>10} {'Action'}")
print(f"  {'─'*55}")
stages = [
    ("Initial rollout",   10, "Monitor error rate and latency"),
    ("Validation",        30, "Compare accuracy on live traffic"),
    ("Broader test",      50, "Check resource scaling behaviour"),
    ("Near-full rollout", 80, "Final latency/cost validation"),
    ("Full promotion",   100, "Decommission v3"),
]
for stage, pct, action in stages:
    print(f"  {stage:<20} {pct:>10}%  {action}")
print()

# Show the generated YAML
print(f"  Generated InferenceService YAML (v4 canary):")
yaml_str  = yaml.dump(isvc_v4_canary, default_flow_style=False, sort_keys=False)
yaml_lines = yaml_str.split("\n")
print(f"  {'─'*55}")
for line in yaml_lines[:30]:
    print(f"  {line}")
if len(yaml_lines) > 30:
    print(f"  ... ({len(yaml_lines)-30} more lines)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Knative autoscaling simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Knative autoscaling: scale-to-zero simulation")
print("━" * 65)
print()

@dataclass
class KnativeAutoscaler:
    """Simulates Knative Pod Autoscaler (KPA) behaviour."""
    min_replicas:     int   = 0
    max_replicas:     int   = 10
    target_rps:       float = 5.0      # target requests/replica/second
    stable_window_s:  int   = 60
    panic_window_s:   int   = 6
    scale_down_delay_s: int = 30
    current_replicas: int   = 0
    _last_nonzero_t:  float = 0.0

    def get_desired_replicas(self, requests_per_sec: float, t: float) -> int:
        """Compute desired replica count given current RPS."""
        if requests_per_sec > 0:
            self._last_nonzero_t = t

        # Scale down only after delay
        time_since_traffic = t - self._last_nonzero_t
        if requests_per_sec == 0 and time_since_traffic < self.scale_down_delay_s:
            return max(self.min_replicas, self.current_replicas)

        # Desired = ceil(rps / target_rps)
        if requests_per_sec == 0:
            desired = self.min_replicas   # scale to zero
        else:
            desired = math.ceil(requests_per_sec / self.target_rps)
        return max(self.min_replicas, min(self.max_replicas, desired))


# Simulate a 24-hour traffic pattern
def simulate_traffic(hour: float) -> float:
    """Realistic daily traffic pattern: off-hours = low, business hours = high."""
    if 2 <= hour < 6:
        return 0.0        # near-zero traffic at night
    elif 9 <= hour < 11:
        return 45.0       # morning spike
    elif 14 <= hour < 16:
        return 60.0       # afternoon peak
    elif 21 <= hour < 22:
        return 80.0       # evening spike (fraud attempts)
    elif 6 <= hour < 22:
        return 20.0       # normal business hours
    else:
        return 2.0        # light overnight traffic

autoscaler = KnativeAutoscaler(
    min_replicas=1, max_replicas=20, target_rps=10,
    scale_down_delay_s=60,
)

time_points     = np.linspace(0, 24, 145)   # every 10 minutes
rps_values      = [simulate_traffic(t) for t in time_points]
replicas_values = []
t0_sim          = 0.0

for t, rps in zip(time_points, rps_values):
    t_sec = t * 3600
    desired = autoscaler.get_desired_replicas(rps, t_sec)
    autoscaler.current_replicas = desired
    replicas_values.append(desired)

print(f"  Simulated 24-hour autoscaling (target_rps={autoscaler.target_rps}):")
print()
print(f"  {'Time':>6} {'RPS':>6} {'Replicas':>10} {'Cost indicator'}")
print(f"  {'─'*42}")
for i, (t, rps, reps) in enumerate(zip(time_points, rps_values, replicas_values)):
    if i % 18 == 0:   # show every 30 minutes
        bar = "█" * reps
        print(f"  {t:>4.0f}h  {rps:>6.0f} {reps:>10}  {bar}")

print()
total_replica_hours = sum(r * (24/len(replicas_values)) for r in replicas_values)
always_on_cost      = autoscaler.max_replicas * 24
savings_pct         = (1 - total_replica_hours / always_on_cost) * 100
print(f"  Autoscaling cost analysis:")
print(f"    Always-on (max_replicas=20): {always_on_cost:.0f} replica-hours")
print(f"    Autoscaled:                  {total_replica_hours:.1f} replica-hours")
print(f"    Cost savings:                {savings_pct:.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Input drift detection (production monitoring)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Production monitoring: input drift detection")
print("━" * 65)
print()

def compute_psi(expected: np.ndarray, actual: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI): measures distribution shift.
    PSI < 0.1:  no significant change
    PSI < 0.25: moderate change, monitor
    PSI >= 0.25: significant shift, retrain
    """
    bins     = np.histogram(expected, bins=n_bins)[1]
    exp_hist = np.histogram(expected, bins=bins)[0].astype(float) + 1e-6
    act_hist = np.histogram(actual,   bins=bins)[0].astype(float) + 1e-6
    exp_pct  = exp_hist / exp_hist.sum()
    act_pct  = act_hist / act_hist.sum()
    psi      = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def detect_drift(training_dist: Dict[str, np.ndarray],
                 production_dist: Dict[str, np.ndarray]) -> Dict:
    """Compute PSI for each feature and flag drifted ones."""
    results = {}
    for feature in training_dist:
        if feature not in production_dist:
            continue
        psi = compute_psi(training_dist[feature], production_dist[feature])
        results[feature] = {
            "psi":      psi,
            "status":   "OK" if psi < 0.1 else "WARN" if psi < 0.25 else "DRIFT",
            "train_mean": float(np.mean(training_dist[feature])),
            "prod_mean":  float(np.mean(production_dist[feature])),
        }
    return results


# Simulate a fraud detection model with features
N_TRAIN, N_PROD = 10000, 5000
n_features = 8
feature_names = [
    "transaction_amount", "merchant_category_code",
    "hour_of_day", "day_of_week",
    "transactions_last_hour", "distance_from_home_km",
    "card_age_days", "country_code_numeric",
]

# Training distribution (historical data)
training_data = {
    "transaction_amount":     rng.exponential(50, N_TRAIN),
    "merchant_category_code": rng.integers(1, 100, N_TRAIN).astype(float),
    "hour_of_day":            rng.integers(0, 24, N_TRAIN).astype(float),
    "day_of_week":            rng.integers(0, 7, N_TRAIN).astype(float),
    "transactions_last_hour": rng.poisson(2, N_TRAIN).astype(float),
    "distance_from_home_km":  rng.exponential(10, N_TRAIN),
    "card_age_days":          rng.normal(365*2, 365, N_TRAIN),
    "country_code_numeric":   rng.integers(1, 50, N_TRAIN).astype(float),
}

# Production distribution — some features have drifted
production_data = {
    "transaction_amount":     rng.exponential(50, N_PROD),           # no drift
    "merchant_category_code": rng.integers(1, 100, N_PROD).astype(float),  # no drift
    "hour_of_day":            rng.integers(0, 24, N_PROD).astype(float),    # no drift
    "day_of_week":            rng.integers(0, 7, N_PROD).astype(float),     # no drift
    "transactions_last_hour": rng.poisson(5, N_PROD).astype(float),   # DRIFTED (fraud spike)
    "distance_from_home_km":  rng.exponential(35, N_PROD),            # DRIFTED (travel patterns)
    "card_age_days":          rng.normal(365*2, 365, N_PROD),         # no drift
    "country_code_numeric":   rng.integers(30, 80, N_PROD).astype(float),  # DRIFTED (new markets)
}

drift_results = detect_drift(training_data, production_data)

print(f"  Input drift monitoring ({N_TRAIN} train / {N_PROD} production samples):")
print(f"  {'Feature':<28} {'PSI':>8} {'Status':>8} {'Train μ':>10} {'Prod μ':>10}")
print(f"  {'─'*68}")
retraining_needed = False
for feature, result in drift_results.items():
    status   = result["status"]
    status_emoji = "✅" if status == "OK" else "⚠️ " if status == "WARN" else "🚨"
    if status == "DRIFT":
        retraining_needed = True
    print(f"  {feature:<28} {result['psi']:>8.4f} {status_emoji+status:>10} "
          f"{result['train_mean']:>10.2f} {result['prod_mean']:>10.2f}")

print()
print(f"  PSI thresholds: < 0.10 OK | 0.10-0.25 WARN | > 0.25 DRIFT")
print(f"  Retraining trigger: {'🚨 YES — drift detected' if retraining_needed else '✅ NO'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Complete MLOps pipeline simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Complete MLOps CI/CD pipeline simulation")
print("━" * 65)
print()

@dataclass
class PipelineRun:
    name:       str
    trigger:    str
    status:     str = "pending"
    steps_done: List[str] = field(default_factory=list)
    metrics:    Dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0

def simulate_mlops_pipeline(
    trigger:        str,
    data_version:   str,
    min_accuracy:   float = 0.90,
    run_hpo:        bool  = False,
    existing_model_acc: float = 0.88,
) -> PipelineRun:
    """Simulate a complete MLOps pipeline run."""
    run = PipelineRun(name=f"run-{data_version}", trigger=trigger)
    run.status = "running"
    t0 = time.perf_counter()

    print(f"  Pipeline triggered by: {trigger}")
    print(f"  Data version: {data_version}")
    print()

    # Step 1: Data validation
    data_issues = rng.random() < 0.05  # 5% chance of data issues
    if data_issues:
        print(f"    [Step 1/6] Validate Data    ✗ FAILED: schema drift detected")
        run.status = "failed"
        run.steps_done.append("validate_data:FAILED")
        return run
    time.sleep(0.01)
    print(f"    [Step 1/6] Validate Data    ✓ OK  ({N_TRAIN} rows, 8 features)")
    run.steps_done.append("validate_data:OK")

    # Step 2: Optionally run HPO
    if run_hpo:
        time.sleep(0.05)
        best_lr = float(10 ** rng.uniform(-4, -2))
        print(f"    [Step 2/6] Run Katib HPO    ✓ OK  best_lr={best_lr:.2e} (50 trials)")
        run.steps_done.append("katib_hpo:OK")
    else:
        print(f"    [Step 2/6] Run Katib HPO    — SKIPPED (run_hpo=False)")
        run.steps_done.append("katib_hpo:SKIPPED")

    # Step 3: Train model
    time.sleep(0.02)
    val_acc = float(0.87 + rng.uniform(0, 0.08))
    run.metrics["val_accuracy"] = val_acc
    print(f"    [Step 3/6] Train Model      ✓ OK  val_accuracy={val_acc:.4f}")
    run.steps_done.append("train_model:OK")

    # Step 4: Evaluate
    time.sleep(0.01)
    val_auc = val_acc - rng.uniform(0, 0.05)
    val_f1  = val_acc - rng.uniform(0, 0.03)
    run.metrics["val_auc"] = float(val_auc)
    run.metrics["val_f1"]  = float(val_f1)
    print(f"    [Step 4/6] Evaluate Model   ✓ OK  auc={val_auc:.4f}  f1={val_f1:.4f}")
    run.steps_done.append("evaluate_model:OK")

    # Step 5: Conditional registration + promotion decision
    improves_existing = val_acc > existing_model_acc
    meets_threshold   = val_acc > min_accuracy
    should_deploy     = improves_existing and meets_threshold

    if should_deploy:
        time.sleep(0.01)
        version = f"v{data_version.replace('-','')[:8]}"
        print(f"    [Step 5/6] Register Model   ✓ OK  {version}  "
              f"({val_acc:.4f} > {existing_model_acc:.4f} existing)")
        run.steps_done.append("register_model:OK")

        # Step 6: Deploy canary
        time.sleep(0.01)
        print(f"    [Step 6/6] Deploy Canary    ✓ OK  10% traffic → fraud-detector-{version}")
        print(f"               Monitoring: 6h before promoting to 100%")
        run.steps_done.append("deploy_canary:OK")
        run.status = "succeeded"
    else:
        reason = (f"val_acc={val_acc:.4f} does not improve "
                  f"existing={existing_model_acc:.4f}" if not improves_existing
                  else f"val_acc={val_acc:.4f} < min_threshold={min_accuracy:.4f}")
        print(f"    [Step 5/6] Register Model   — SKIPPED: {reason}")
        print(f"    [Step 6/6] Deploy Canary    — SKIPPED (no registration)")
        run.steps_done.append("register_model:SKIPPED")
        run.steps_done.append("deploy_canary:SKIPPED")
        run.status = "succeeded (model not deployed)"

    run.duration_s = time.perf_counter() - t0
    return run


# Run different pipeline scenarios
scenarios = [
    ("Nightly scheduled run",     "2024-06-01", False, 0.88),
    ("Data drift retraining",     "2024-06-15", False, 0.89),
    ("Weekly HPO + retrain",      "2024-06-22", True,  0.90),
]

for trigger, data_ver, do_hpo, existing_acc in scenarios:
    print(f"  {'─'*60}")
    run = simulate_mlops_pipeline(
        trigger             = trigger,
        data_version        = data_ver,
        min_accuracy        = 0.89,
        run_hpo             = do_hpo,
        existing_model_acc  = existing_acc,
    )
    print(f"  Result: {run.status}  ({run.duration_s*1000:.0f}ms)")
    if run.metrics:
        print(f"  Metrics: {run.metrics}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Kubeflow component decision guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Kubeflow decision guide and production checklist")
print("━" * 65)
print()

print(f"  Kubeflow component selection guide:")
print(f"  ┌──────────────────────────────────────────────────────────────────┐")
print(f"  │ Task                                │ Kubeflow Component         │")
print(f"  ├──────────────────────────────────────────────────────────────────┤")
print(f"  │ Orchestrate multi-step ML workflow  │ Kubeflow Pipelines (KFP)   │")
print(f"  │ Distributed PyTorch training        │ PyTorchJob (Training Op)   │")
print(f"  │ Distributed TensorFlow training     │ TFJob (Training Operator)  │")
print(f"  │ MPI / Horovod training              │ MPIJob (Training Operator) │")
print(f"  │ Hyperparameter tuning at scale      │ Katib                      │")
print(f"  │ Neural Architecture Search          │ Katib (DARTS/ENAS)         │")
print(f"  │ Model serving (REST/gRPC)           │ KServe InferenceService    │")
print(f"  │ Canary / A/B model deployment       │ KServe traffic split       │")
print(f"  │ Batch inference (large dataset)     │ KServe ClusterServingRuntime│")
print(f"  │ Interactive development             │ Kubeflow Notebooks         │")
print(f"  │ Data preprocessing (scale)          │ KFP + Spark/Ray step       │")
print(f"  │ Experiment tracking                 │ KFP Metadata + MLflow      │")
print(f"  └──────────────────────────────────────────────────────────────────┘")
print()

print(f"  Production readiness checklist:")
items = [
    ("Pipelines",    [
        "Pipeline YAML committed to git (GitOps)",
        "Container images use pinned digest tags (not :latest)",
        "Caching enabled for expensive steps (preprocess, embeddings)",
        "Exit handler for cleanup/notification on failure",
        "Resource requests match profiled actual usage",
    ]),
    ("Training",     [
        "Checkpoint every N steps (handle pod eviction on spot instances)",
        "MaxRestarts set on PyTorchJob (auto-retry on transient failures)",
        "Gang scheduling via Volcano (avoid deadlock on GPU allocation)",
        "Gradient clipping enabled (avoid NaN on distributed training)",
        "NCCL_SOCKET_IFNAME set to correct network interface",
    ]),
    ("Serving",      [
        "Canary rollout before 100% traffic promotion",
        "MinReplicas >= 1 for latency-sensitive production endpoints",
        "Liveness and readiness probes configured",
        "Request/response logging enabled (for drift monitoring)",
        "Prometheus metrics scraped (latency, throughput, error rate)",
    ]),
    ("Observability",[
        "Structured JSON logging from all training and serving pods",
        "Grafana dashboard for GPU utilisation and training loss",
        "PSI monitoring for input feature drift (retraining trigger)",
        "Alerts on: pipeline failure, serving error rate > 1%, p99 > SLA",
        "Model lineage tracked (data version → pipeline run → model)",
    ]),
]

for category, checklist in items:
    print(f"  [{category}]")
    for item in checklist:
        print(f"    ✓ {item}")
    print()
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