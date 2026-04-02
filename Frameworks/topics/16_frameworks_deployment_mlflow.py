"""
MLflow — The Open-Source Platform for the ML Lifecycle
=======================================================

MLflow is an open-source platform for managing the end-to-end machine
learning lifecycle. It was created by Databricks and released in June 2018
by Matei Zaharia (creator of Apache Spark) and colleagues. Despite being
only a few years old, MLflow has become the dominant experiment tracking
and model management platform, with over 17 million monthly downloads,
used by Netflix, Microsoft, Facebook, Booking.com, H&M, and thousands
of other organisations.

The problem MLflow solves is fundamental: machine learning development is
messy. A data scientist runs hundreds of experiments with different
hyperparameters, datasets, preprocessing strategies, and model architectures.
Without tracking, the questions "which run produced that good model?" and
"what exactly did I do to get those results?" become nearly unanswerable.
Reproducing a result from three weeks ago — with the same data, the same
code, the same random seed — is a daily frustration in ML teams.

Beyond tracking, there is the deployment problem: once you have a good
model, how do you package it, version it, register it, and serve it
consistently across different environments (laptop, staging cluster,
production GPU fleet)?

MLflow addresses these challenges with four tightly integrated components:

    MLflow Tracking:
        The experiment tracking system. Log parameters, metrics, tags,
        and artefacts (model files, plots, datasets) from any ML code
        with a simple Python API. Every run is automatically timestamped,
        tagged with the git commit, and stored in a queryable backend.

    MLflow Models:
        A universal packaging format for ML models. Wraps trained models
        from any framework (PyTorch, TensorFlow, sklearn, XGBoost, etc.)
        in a standard directory with a model signature, environment
        specification, and multiple flavours (python_function, pytorch,
        sklearn, ONNX, ...). A packaged MLflow model can be deployed
        to any MLflow-compatible serving infrastructure without rewriting.

    MLflow Model Registry:
        A centralised model store with version control, stage management
        (Staging → Production → Archived), and governance workflows.
        Think of it as Git for model weights: every model version is
        stored, annotated with metrics and tags, and can be promoted
        through stages with review and approval workflows.

    MLflow Projects:
        A reproducible packaging format for ML code. Defines the
        environment (conda or Docker), entry points, and parameters
        so that any MLflow Project can be reproduced exactly with
        mlflow run github.com/username/repo.

MLflow is framework-agnostic and backend-agnostic:
    Works with:    PyTorch, TensorFlow, JAX, sklearn, XGBoost, LightGBM,
                   Hugging Face, Spark MLlib, statsmodels, Prophet
    Backends:      Local filesystem, SQLite, PostgreSQL, MySQL, cloud
                   storage (S3, GCS, Azure Blob), Databricks Managed MLflow

This module covers the complete MLflow stack with deep theory: the tracking
data model (runs, experiments, metrics, params, artefacts), the autologging
system, custom logging patterns, the Models packaging format and flavour
system, the Model Registry workflow, MLflow Model Serving and inference,
comparing and querying runs programmatically, integrating MLflow with
hyperparameter tuning, and production MLOps patterns with MLflow.

"""

import textwrap
import re

TOPIC_NAME   = "MLflow — The Open-Source Platform for the ML Lifecycle"
DISPLAY_NAME = "16 · MLflow"
ICON         = "🔬"
SUBTITLE     = "From Experiment Tracking to Model Registry and Production Serving"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE ML LIFECYCLE PROBLEM AND HOW MLFLOW SOLVES IT

### The Experiment Chaos Problem

    A typical ML project involves hundreds of experiments. Consider a team
    tuning a BERT fine-tuning pipeline for 3 weeks:
        - 8 learning rate values
        - 4 batch sizes
        - 3 warmup schedules
        - 2 dropout settings
        - Multiple dataset versions

    That is 8 × 4 × 3 × 2 × 3 = 576 potential runs, each potentially taking
    30-60 minutes on a GPU. Without systematic tracking:
        - "Which run achieved 94.2% accuracy?" → check 576 log files
        - "What was the exact code when we hit that result?" → unknown
        - "Can we reproduce run #47?" → maybe, if nothing changed
        - "Which preprocessing version was used?" → unclear
        - "What was the GPU memory usage?" → not recorded

    This is not an edge case. According to the 2020 State of MLOps report,
    over 60% of ML teams could not reliably reproduce their best models.

### The Four Components in Context

    MLflow TRACKING solves experiment chaos:
        Every metric, parameter, and artefact from every run is recorded.
        Runs are searchable, comparable, and rerunnable.
        Integration with git records the exact code version.

    MLflow MODELS solves the "it works on my laptop" problem:
        Packages the model with its conda/pip environment.
        Provides a standard pyfunc interface regardless of framework.
        The same packaged model can be served locally, in Docker, on
        SageMaker, Azure ML, or any MLflow-compatible serving system.

    MLflow MODEL REGISTRY solves model governance:
        Which model version is in production right now?
        Who approved this model for production?
        What was its validation accuracy?
        Registry provides answers with full audit trail.

    MLflow PROJECTS solves reproducibility:
        Encapsulates code, environment, and entry points.
        Any collaborator (or CI/CD pipeline) can reproduce exactly.

### MLflow vs Alternatives

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Tool              │ Tracking │ Registry │ Serving  │ Open Source     │
    ├──────────────────────────────────────────────────────────────────────┤
    │ MLflow            │ ✓        │ ✓        │ ✓        │ ✓ (Apache 2.0)  │
    │ Weights & Biases  │ ✓✓       │ ✓        │ ✗        │ ✗ (commercial)  │
    │ Neptune.ai        │ ✓✓       │ ✓        │ ✗        │ ✗ (commercial)  │
    │ Comet ML          │ ✓        │ ✓        │ ✗        │ ✗ (commercial)  │
    │ ClearML           │ ✓        │ ✓        │ ✓        │ ✓ (AGPL)        │
    │ DVC               │ ✗        │ ✗        │ ✗        │ ✓ (data only)   │
    │ Vertex AI         │ ✓        │ ✓        │ ✓        │ ✗ (GCP only)    │
    │ SageMaker         │ ✓        │ ✓        │ ✓        │ ✗ (AWS only)    │
    └──────────────────────────────────────────────────────────────────────┘

    MLflow's positioning: the only fully open-source platform covering the
    full lifecycle. Can self-host for free, no vendor lock-in, integrates
    with any cloud provider.

### The MLflow Tracking Server Architecture

    Local (default, development):
        Runs stored in ./mlruns/ directory on disk.
        SQLite database or plain YAML files.
        Perfect for individual development.
        mlflow.set_tracking_uri("./my_mlruns")

    Remote (team/production):
        Tracking server process handles HTTP API.
        Backend store: PostgreSQL or MySQL (metadata: params, metrics, tags)
        Artefact store: S3, GCS, Azure Blob, HDFS (large binary files)

        mlflow server \\
            --backend-store-uri postgresql://user:pw@host/mlflow \\
            --default-artifact-root s3://my-bucket/mlflow/ \\
            --host 0.0.0.0 --port 5000

    Client configuration:
        import mlflow
        mlflow.set_tracking_uri("http://mlflow.company.com:5000")
        mlflow.set_experiment("fraud-detection-v2")

    Managed (zero-ops):
        Databricks Managed MLflow: enterprise-grade, no setup needed.
        Azure Machine Learning: uses MLflow protocol, Azure storage.


##### PART 2 — MLFLOW TRACKING: THE EXPERIMENT DATA MODEL

### Core Entities

    EXPERIMENT:
        A named collection of runs. The top-level organisational unit.
        Experiments correspond to a project, dataset version, or major
        modelling approach.
        experiment_id: unique integer assigned by MLflow
        name:          human-readable ("bert-finetune-imdb")
        lifecycle_stage: active or deleted

    RUN:
        A single execution of your training code within an experiment.
        run_id: UUID (e.g. "3b8d2e1a4f5c6d7890ab12cd34ef5678")
        A run stores four types of data:
            Parameters:    key-value pairs, static hyperparameters
            Metrics:       key-value-step triples, scalar measurements
            Tags:          key-value pairs, metadata and annotations
            Artefacts:     files (model weights, plots, datasets, configs)

    METRIC:
        A numeric measurement recorded during a run.
        Has a key (name), value (float), step (integer, for time series),
        and timestamp.
        Metrics can be logged at every epoch → full training curve stored.
        mlflow.log_metric("val_accuracy", 0.946, step=epoch)

    PARAMETER:
        A static hyperparameter of the run. String key, string value.
        Logged once per run (not time-varying).
        mlflow.log_param("learning_rate", 3e-4)
        mlflow.log_params({"lr": 3e-4, "batch_size": 64, "epochs": 20})

    ARTEFACT:
        A file or directory stored in the artefact store.
        Model weights, ONNX files, confusion matrices, requirements.txt,
        feature importance plots — anything file-based.
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifacts("./plots/", artifact_path="evaluation")

    TAG:
        Arbitrary key-value metadata. Unlike params, can be set/updated
        after the run. Used for: model type, team, environment, status.
        mlflow.set_tag("model_type", "transformer")
        mlflow.set_tag("team", "nlp-core")

### Run Lifecycle

    Runs have three lifecycle states:
        RUNNING:   actively executing (in progress)
        FINISHED:  completed successfully
        FAILED:    encountered an error
        KILLED:    manually terminated

    Context manager pattern (recommended):
        with mlflow.start_run(run_name="experiment-v3"):
            mlflow.log_params(config)
            model = train(config)
            mlflow.log_metric("val_acc", val_acc)
            mlflow.sklearn.log_model(model, "model")
        # Run automatically set to FINISHED on context exit,
        # FAILED if an exception is raised

    Manual pattern:
        run = mlflow.start_run()
        try:
            # training code
            mlflow.end_run(status="FINISHED")
        except Exception:
            mlflow.end_run(status="FAILED")
            raise

    Nested runs (parent-child hierarchy):
        with mlflow.start_run(run_name="hpo-sweep") as parent_run:
            for config in configs:
                with mlflow.start_run(run_name=f"trial-{i}",
                                      nested=True) as child_run:
                    val_acc = train_and_evaluate(config)
                    mlflow.log_params(config)
                    mlflow.log_metric("val_acc", val_acc)
            # Parent run aggregates children

### Autologging: Zero-Code Tracking

    MLflow autologging automatically captures parameters, metrics, and
    models without any manual log calls. Activate with one line:

        mlflow.autolog()               # enable for all supported frameworks
        mlflow.sklearn.autolog()       # sklearn only
        mlflow.pytorch.autolog()       # PyTorch only
        mlflow.tensorflow.autolog()    # TensorFlow/Keras only
        mlflow.xgboost.autolog()       # XGBoost only
        mlflow.lightgbm.autolog()      # LightGBM only

    What autologging captures:
        sklearn:
            Parameters: all estimator parameters (from get_params())
            Metrics:    training score, test score
            Artefacts:  pickled model, input example, model signature
            Tags:       estimator class, sklearn version

        PyTorch Lightning:
            Parameters: Trainer init parameters, model hparams
            Metrics:    every metric from trainer.callback_metrics per epoch
            Artefacts:  model checkpoint, requirements

        TensorFlow/Keras:
            Parameters: optimizer, loss, metrics, epochs
            Metrics:    loss and all metrics per epoch
            Artefacts:  saved model, summary, TensorBoard logs

    Autologging configuration:
        mlflow.sklearn.autolog(
            log_input_examples   = True,   # log a sample of input data
            log_model_signatures = True,   # infer input/output schema
            log_models           = True,   # log the trained model
            max_tuning_runs      = 5,      # for HPO: log top-5 child runs
        )

### Searching and Querying Runs

    The MLflow search API uses a SQL-like syntax:

        runs = mlflow.search_runs(
            experiment_names = ["fraud-detection-v2"],
            filter_string    = "metrics.val_accuracy > 0.92 AND params.lr < 0.001",
            order_by         = ["metrics.val_accuracy DESC"],
            max_results      = 20,
        )
        # Returns a pandas DataFrame with all run data

    Filter string operators:
        = != < > <= >=   (numeric comparisons for metrics/params)
        LIKE 'pattern%'  (string matching with wildcards)
        IN ('a','b','c') (membership test)
        AND OR NOT       (logical operators)

    Useful search patterns:
        # Find the best run by metric:
        "metrics.val_f1 = (SELECT MAX(metrics.val_f1) FROM runs)"
        # This is not supported — use order_by + max_results=1 instead

        # Find runs from a specific git commit:
        "tags.mlflow.source.git.commit = 'abc123def456'"

        # Find runs that used a specific dataset:
        "tags.dataset_version = 'v3.2'"

        # Find all completed runs in the last 7 days:
        "attributes.status = 'FINISHED' AND attributes.start_time > 1704067200000"


##### PART 3 — MLFLOW MODELS: THE UNIVERSAL PACKAGING FORMAT

### The MLflow Model Directory Structure

    An MLflow Model is a directory with a standard layout:

        my_model/
        ├── MLmodel           ← the model metadata manifest (YAML)
        ├── model.pkl         ← the actual model file (framework-specific)
        ├── conda.yaml        ← conda environment specification
        ├── requirements.txt  ← pip requirements
        ├── python_env.yaml   ← Python version and virtual env spec
        └── input_example.json ← example input for signature validation

    The MLmodel manifest (YAML) contains:
        flavors:              all the ways to load this model
        signature:            input/output schema with types and shapes
        utc_time_created:     when the model was logged
        run_id:               the training run that produced it
        model_uuid:           globally unique identifier
        mlflow_version:       MLflow version used to save

### The Flavour System: Framework Polymorphism

    A single MLflow model can be loaded in multiple ways, called FLAVOURS.
    This is the key to MLflow's framework-agnostic serving.

    Every model has the python_function (pyfunc) flavour — a generic
    interface that wraps any model:
        loaded = mlflow.pyfunc.load_model("runs:/run_id/model_path")
        predictions = loaded.predict(input_df)   # always a DataFrame

    Framework-specific flavours preserve native behaviour:
        mlflow.sklearn.load_model()     → returns sklearn estimator
        mlflow.pytorch.load_model()     → returns torch.nn.Module
        mlflow.tensorflow.load_model()  → returns tf.keras.Model
        mlflow.onnx.load_model()        → returns onnx.ModelProto
        mlflow.xgboost.load_model()     → returns xgboost.Booster
        mlflow.lightgbm.load_model()    → returns lgb.Booster

    Logging models with flavours:
        mlflow.sklearn.log_model(
            sk_model       = pipeline,
            artifact_path  = "model",              # path in artefact store
            signature      = infer_signature(X_train, y_pred),
            input_example  = X_train[:5],          # sample input
            registered_model_name = "fraud-detector",  # auto-register
        )

    Custom pyfunc model (wrap any code as an MLflow model):
        class MyModel(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                import pickle
                with open(context.artifacts["model_path"], "rb") as f:
                    self.model = pickle.load(f)

            def predict(self, context, model_input, params=None):
                return self.model.predict(model_input)

        mlflow.pyfunc.log_model(
            artifact_path  = "my_custom_model",
            python_model   = MyModel(),
            artifacts      = {"model_path": "/path/to/model.pkl"},
            conda_env      = {"channels": ["defaults"],
                               "dependencies": ["python=3.11", "scikit-learn"]},
        )

### Model Signatures: Input/Output Contracts

    A model signature defines the expected input and output schema.
    This is used for:
        - Validation at inference time (catch wrong input shapes/types)
        - Documentation in the UI
        - Automatic schema enforcement in MLflow Serving

    Inferring signatures automatically:
        from mlflow.models import infer_signature

        X_train = pd.DataFrame({"feature_1": [1.2, 3.4], "feature_2": [5.6, 7.8]})
        y_pred  = model.predict(X_train)

        signature = infer_signature(X_train, y_pred)
        # Signature records: inputs=[feature_1: double, feature_2: double]
        #                    outputs=[prediction: long]

    Defining signatures manually:
        from mlflow.types.schema import Schema, ColSpec, TensorSpec

        input_schema  = Schema([
            ColSpec("double", "age"),
            ColSpec("string", "category"),
            ColSpec("float",  "amount"),
        ])
        output_schema = Schema([TensorSpec(np.dtype("float32"), (-1, 5))])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)


##### PART 4 — THE MODEL REGISTRY: VERSION CONTROL FOR MODELS

### Why a Model Registry?

    A model registry solves the "which model is in production?" problem.
    Without a registry:
        - Production model is a file on a server somewhere
        - No record of when it was deployed or by whom
        - Rollback requires manually tracking old model files
        - No comparison between the model in staging vs production

    With a model registry:
        - Every model version has a unique identifier (name + version number)
        - Each version has a stage: None → Staging → Production → Archived
        - Full audit trail: who registered it, when, with what metrics
        - Rollback = change stage on an older version
        - Champion/challenger A/B testing = two versions in Production

### Registry Concepts

    REGISTERED MODEL:
        A named model entity. Think of it as a git repository for model
        weights. Name: "fraud-detector", "sentiment-classifier", etc.

    MODEL VERSION:
        A specific version of a registered model (auto-incremented).
        Version 1, Version 2, ... Each version points to a run artefact.
        Versions are immutable once created.

    STAGE:
        The lifecycle stage of a model version:
            None:       just registered, not yet evaluated
            Staging:    under evaluation, not yet serving traffic
            Production: serving live traffic
            Archived:   retired, no longer active

        Transitions require explicit action (programmatic or UI).
        In enterprises: transitions can require approval workflows.

    ALIAS (MLflow 2.x):
        Named pointers to specific versions. More flexible than stages.
        @production, @champion, @challenger, @latest
        Load by alias: mlflow.pyfunc.load_model("models:/fraud-detector@production")

### Registry API

    Register a model from a completed run:
        from mlflow import MlflowClient

        client = MlflowClient()

        # Option 1: Register when logging (creates version 1)
        mlflow.sklearn.log_model(
            model,
            artifact_path         = "model",
            registered_model_name = "fraud-detector",
        )

        # Option 2: Register a run artefact post-hoc
        result = mlflow.register_model(
            model_uri      = "runs:/3b8d2e1a4f/model",
            name           = "fraud-detector",
        )
        print(f"Version: {result.version}")

    Transition between stages:
        client.transition_model_version_stage(
            name    = "fraud-detector",
            version = 3,
            stage   = "Staging",
        )
        # Later, after validation:
        client.transition_model_version_stage(
            name               = "fraud-detector",
            version            = 3,
            stage              = "Production",
            archive_existing_versions = True,   # demote old Production to Archived
        )

    Loading models by stage (deployment code):
        model = mlflow.pyfunc.load_model("models:/fraud-detector/Production")
        # Always loads the current Production version
        # Code does not need to know the version number

    Loading models by alias (MLflow 2.x preferred):
        model = mlflow.pyfunc.load_model("models:/fraud-detector@champion")

    Annotating versions with metrics and descriptions:
        client.update_model_version(
            name        = "fraud-detector",
            version     = 3,
            description = "Retrained on Jan 2024 data. F1=0.947, AUC=0.981."
        )
        client.set_model_version_tag(
            name    = "fraud-detector",
            version = 3,
            key     = "val_f1",
            value   = "0.947",
        )

### Model Registry Workflow in Production

    The champion/challenger pattern:

        # Week 1: Train challenger model
        with mlflow.start_run():
            train_challenger()
            mlflow.sklearn.log_model(model, "model",
                registered_model_name="fraud-detector")
            # → creates version 4

        # Register metrics for review
        client.set_model_version_tag("fraud-detector", 4, "auc", "0.985")

        # After validation: promote challenger to staging
        client.transition_model_version_stage("fraud-detector", 4, "Staging")

        # A/B test: run both Production and Staging with traffic split
        # Load both versions:
        champion   = mlflow.pyfunc.load_model("models:/fraud-detector/Production")
        challenger = mlflow.pyfunc.load_model("models:/fraud-detector/Staging")

        # After A/B test confirms challenger wins:
        client.transition_model_version_stage(
            "fraud-detector", 4, "Production",
            archive_existing_versions=True  # demotes version 3 → Archived
        )


##### PART 5 — MLFLOW SERVING AND DEPLOYMENT

### MLflow Models Serve

    The simplest deployment: spin up a REST server from any MLflow model:

        mlflow models serve -m "models:/fraud-detector/Production" --port 5001

    This starts a Flask server with two endpoints:
        POST /invocations:    run model inference
        GET  /ping:           health check

    Request format (JSON):
        Content-Type: application/json

        # DataFrame split format:
        {"dataframe_split": {"columns": ["feat1", "feat2"],
                              "data": [[1.2, 3.4], [5.6, 7.8]]}}

        # DataFrame records format:
        {"dataframe_records": [{"feat1": 1.2, "feat2": 3.4}]}

        # Inputs format (for tensor inputs):
        {"inputs": [[1.2, 3.4, 5.6]]}

    Python client:
        import requests, json
        resp = requests.post(
            "http://localhost:5001/invocations",
            json={"dataframe_records": [{"feat1": 1.2, "feat2": 3.4}]},
            headers={"Content-Type": "application/json"},
        )
        print(resp.json())

### Deployment Targets

    Docker deployment:
        mlflow models build-docker -m "models:/fraud-detector/Production" \\
            -n fraud-detector:v3 --enable-mlserver
        docker run -p 5001:8080 fraud-detector:v3

    SageMaker deployment:
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("sagemaker")
        client.create_deployment(
            name       = "fraud-detector-prod",
            model_uri  = "models:/fraud-detector/3",
            config     = {
                "instance_type": "ml.m5.large",
                "region_name":   "us-east-1",
                "execution_role_arn": "arn:aws:iam::...",
            }
        )

    Azure ML deployment:
        client = get_deploy_client("azureml")
        client.create_deployment(
            name      = "fraud-detector-v3",
            model_uri = "models:/fraud-detector/3",
        )

    Kubernetes with KServe:
        apiVersion: serving.kserve.io/v1beta1
        kind: InferenceService
        spec:
          predictor:
            model:
              modelFormat: {name: mlflow}
              storageUri: "s3://mlflow/artifacts/3/model"
              # KServe uses MLflow's pyfunc loader

### MLserver Backend

    MLserver is a production-grade ML model server that natively supports
    MLflow models via the MLflow runtime:
        - gRPC and REST with Open Model Interface (v2)
        - Adaptive batching
        - Parallel inference workers
        - Prometheus metrics
        - Multiple model serving

    pip install mlserver mlserver-mlflow

    model-settings.json:
        {
          "name": "fraud-detector",
          "implementation": "mlserver_mlflow.MLflowRuntime",
          "parameters": {
            "uri": "models:/fraud-detector/Production"
          }
        }

    mlserver start .   # starts the server


##### PART 6 — MLFLOW WITH HYPERPARAMETER TUNING

### MLflow + Optuna

    Optuna is the most popular HPO library in Python. The standard pattern
    logs each trial as a child run of the parent HPO sweep:

        import optuna
        import mlflow

        def objective(trial):
            with mlflow.start_run(nested=True):
                lr     = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
                n_layers = trial.suggest_int("n_layers", 2, 8)
                dropout  = trial.suggest_float("dropout", 0.0, 0.5)

                mlflow.log_params({"lr": lr, "n_layers": n_layers, "dropout": dropout})
                val_acc = train_and_evaluate(lr, n_layers, dropout)
                mlflow.log_metric("val_acc", val_acc)

            return val_acc

        with mlflow.start_run(run_name="optuna-hpo-sweep"):
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)

            best_params = study.best_params
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_acc", study.best_value)

### MLflow + Ray Tune

    Ray Tune integrates directly with MLflow via the MLflowLoggerCallback:

        from ray.air.integrations.mlflow import MLflowLoggerCallback
        from ray import tune

        tuner = tune.Tuner(
            trainable,
            param_space = search_space,
            run_config  = tune.RunConfig(
                callbacks = [MLflowLoggerCallback(
                    tracking_uri     = "http://mlflow:5000",
                    experiment_name  = "ray-tune-hpo",
                    save_artifact    = True,
                )],
            ),
        )
        results = tuner.fit()
        # Each trial automatically logged as a separate MLflow run

### Metric History and Comparison

    Access the full metric history (every epoch, not just final):
        client  = MlflowClient()
        history = client.get_metric_history(run_id, "val_loss")
        # Returns list of Metric(key, value, timestamp, step) objects

        steps  = [m.step  for m in history]
        losses = [m.value for m in history]
        # → plot the full training curve


##### PART 7 — MLFLOW RECIPES AND EVALUATION

### MLflow Evaluate: Model Quality Assessment

    mlflow.evaluate() provides a unified interface for computing and
    logging model evaluation metrics automatically:

        import mlflow

        with mlflow.start_run():
            mlflow.log_params({"model": "rf", "n_estimators": 100})

            result = mlflow.evaluate(
                model         = "runs:/run_id/model",    # or a pyfunc model
                data          = X_test_df,               # pandas DataFrame
                targets       = "label",                 # column name
                model_type    = "classifier",            # or "regressor"
                evaluators    = "default",
                evaluator_config = {
                    "log_model_explainability": True,    # SHAP feature importance
                    "explainability_nsamples":  100,
                },
            )

    Automatically computes and logs:
        Classifier:    accuracy, F1, precision, recall, AUC-ROC, log_loss,
                       confusion matrix (as artefact), ROC curve (as artefact)
        Regressor:     MAE, MSE, RMSE, R², MAPE

    Custom metrics:
        from mlflow.models.evaluation import make_metric

        def my_metric(eval_df, builtin_metrics):
            # eval_df has columns: prediction, target
            specificity = ...
            return MetricValue(aggregate_results={"specificity": specificity})

        result = mlflow.evaluate(
            ...,
            extra_metrics = [make_metric(eval_fn=my_metric, name="specificity")],
        )

### Baseline Models

    mlflow.evaluate() can compare your model against a simple baseline:

        result = mlflow.evaluate(
            model        = "runs:/run_id/model",
            data         = test_data,
            targets      = "label",
            model_type   = "classifier",
            baseline_model = "runs:/baseline_run_id/model",  # e.g. DummyClassifier
        )
        # Logs all metrics for both model and baseline
        # UI shows side-by-side comparison

### MLflow Recipes (formerly Pipelines)

    MLflow Recipes provide opinionated, production-ready ML workflow templates:
        - Classification Recipe
        - Regression Recipe

    The recipe defines a fixed pipeline structure:
        ingest → split → transform → train → evaluate → register

    Configured via YAML files:
        recipe.yaml:         which steps to run, dataset location
        profiles/dev.yaml:   development-specific config
        profiles/prod.yaml:  production-specific config

    Run the full pipeline:
        mlflow recipes run --profile dev

    Benefits: reproducibility, best practices enforced, full MLflow tracking
    integrated at every step.


##### PART 8 — PRODUCTION MLOPS PATTERNS WITH MLFLOW

### CI/CD Integration Pattern

    A complete MLOps pipeline triggered by a git push:

        # .github/workflows/train.yml
        on: [push]
        jobs:
          train:
            steps:
              - name: Train and evaluate
                env:
                  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
                run: python train.py

              - name: Promote if better than Production
                run: python scripts/promote_if_better.py

    The promotion script:
        def promote_if_better(model_name, candidate_run_id, metric, threshold):
            client = MlflowClient()
            # Get current Production metrics
            prod_versions = client.get_latest_versions(model_name, ["Production"])
            if prod_versions:
                prod_run_id = prod_versions[0].run_id
                prod_metric = client.get_run(prod_run_id).data.metrics[metric]
            else:
                prod_metric = 0.0  # no Production yet

            # Get candidate metrics
            cand_metric = client.get_run(candidate_run_id).data.metrics[metric]

            if cand_metric > prod_metric + threshold:
                # Register and promote
                version = mlflow.register_model(
                    f"runs:/{candidate_run_id}/model", model_name
                )
                client.transition_model_version_stage(
                    model_name, version.version, "Production",
                    archive_existing_versions=True,
                )
                return True
            return False

### Data Lineage with MLflow

    Track which data version was used for each run:
        mlflow.log_input(
            mlflow.data.from_pandas(train_df, source="gs://data/train_v3.parquet",
                                     name="training_data", targets="label"),
            context="training",
        )
        # Stores: dataset name, source URI, schema, digest (hash of data)
        # Enables: "which runs used this dataset version?"

### Model Monitoring Integration

    Log production inference metrics back to MLflow:
        # In the serving layer, periodically log monitoring metrics
        with mlflow.start_run(run_id=production_run_id):
            mlflow.log_metric("prod_accuracy_week_3",  0.934, step=3)
            mlflow.log_metric("prod_accuracy_week_4",  0.921, step=4)  # drift!
            mlflow.log_metric("input_feature_psi",     0.31,  step=4)  # alert

    Trigger retraining when PSI > threshold or accuracy drops below SLA.

### Best Practices

    1. Always use named experiments (not the Default experiment):
       mlflow.set_experiment("project-name/component-name")

    2. Log params BEFORE metrics (so failed runs still have params):
       mlflow.log_params(config)
       # training...
       mlflow.log_metric("val_acc", val_acc)

    3. Log input examples for signature validation:
       mlflow.sklearn.log_model(model, "model", input_example=X_train[:3])

    4. Use tags for non-numeric metadata:
       mlflow.set_tags({"model_type": "ensemble", "dataset": "v3.2", "env": "prod"})

    5. Log the full training curve (step parameter):
       for epoch, val_loss in enumerate(history):
           mlflow.log_metric("val_loss", val_loss, step=epoch)

    6. Log dataset hash for reproducibility:
       import hashlib
       data_hash = hashlib.md5(X_train.tobytes()).hexdigest()[:8]
       mlflow.set_tag("data_hash", data_hash)

    7. Use register_model with registered_model_name at log time:
       mlflow.sklearn.log_model(..., registered_model_name="model-name")

    8. Archive old Production versions when promoting:
       client.transition_model_version_stage(..., archive_existing_versions=True)

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Scenario                    │ MLflow Component(s)                    │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Track an experiment         │ mlflow.start_run + log_*               │
    │ Compare many runs           │ UI or search_runs() DataFrame          │
    │ Zero-code tracking          │ mlflow.autolog()                       │
    │ Package model for serving   │ mlflow.<framework>.log_model()         │
    │ Version a trained model     │ register_model() + Model Registry      │
    │ Promote to production       │ transition_model_version_stage()       │
    │ Load production model       │ load_model("models:/name/Production")  │
    │ Serve via REST API          │ mlflow models serve -m ...             │
    │ Deploy to cloud             │ mlflow deployments create              │
    │ Evaluate model quality      │ mlflow.evaluate()                      │
    │ Reproduce an old run        │ mlflow.projects.run() or run_id        │
    └──────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · MLflow Tracking — Experiments, Runs, Metrics & Artefacts": {
        "description": (
            "Complete MLflow tracking API from first principles. "
            "Experiment creation, run lifecycle, context managers. "
            "log_param, log_params, log_metric (with step), log_metrics. "
            "Metric time series: full epoch-by-epoch training curves. "
            "log_artifact, log_artifacts: files and directories. "
            "set_tag, set_tags: metadata annotation. "
            "log_dict, log_text, log_figure, log_image utilities. "
            "Nested runs: parent HPO sweep with child trial runs. "
            "MlflowClient: programmatic run access and search. "
            "search_runs: filter string and result DataFrame analysis. "
            "Run comparison: finding best runs by metric. "
            "Simulated multi-framework tracking pipeline."
        ),
        "language": "python",
        "code": '''
import os
import time
import math
import json
import tempfile
import shutil
import hashlib
from typing import Dict, List, Any

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    print(f"  MLflow version: {mlflow.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "mlflow", "--quiet"], check=True)
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    print(f"  MLflow version: {mlflow.__version__}")

import numpy as np
rng = np.random.default_rng(42)

print("=" * 65)
print("  MLFLOW TRACKING — EXPERIMENTS, RUNS & ARTEFACTS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# Setup: local tracking directory
# ─────────────────────────────────────────────────────────────────────────
TRACKING_DIR = tempfile.mkdtemp(prefix="mlflow_demo_")
mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")
client = MlflowClient(tracking_uri=f"file://{TRACKING_DIR}")

print(f"  Tracking URI: file://{TRACKING_DIR}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Creating experiments
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Experiments: creation and configuration")
print("━" * 65)
print()

# Create experiments programmatically
exp_id = mlflow.create_experiment(
    name   = "classification-benchmark",
    tags   = {
        "project":    "fraud-detection",
        "team":       "ml-core",
        "created_by": "demo-script",
    },
)

exp = client.get_experiment(exp_id)
print(f"  Created experiment:")
print(f"    experiment_id:   {exp.experiment_id}")
print(f"    name:            {exp.name}")
print(f"    lifecycle_stage: {exp.lifecycle_stage}")
print(f"    artifact_loc:    {exp.artifact_location[-40:]}...")
print(f"    tags:            {exp.tags}")
print()

# Set as active experiment for subsequent runs
mlflow.set_experiment("classification-benchmark")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Basic run with all log types
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Run lifecycle: params, metrics, tags, artefacts")
print("━" * 65)
print()

def simulate_training(lr, n_layers, dropout, n_epochs=20, seed=42):
    """Simulate a realistic training curve."""
    _rng = np.random.default_rng(seed)
    lr_penalty = abs(math.log10(lr / 3e-3)) * 0.2
    arch_penalty = abs(n_layers - 4) * 0.05
    drop_penalty = abs(dropout - 0.15) * 0.3
    best_val = max(0.55, 0.96 - lr_penalty - arch_penalty - drop_penalty)
    train_losses, val_losses, val_accs = [], [], []
    for epoch in range(n_epochs):
        progress = 1.0 - math.exp(-4 * (epoch+1) / n_epochs)
        noise    = _rng.normal(0, 0.015 * (1 - progress + 0.2))
        val_acc  = min(0.999, max(0.4, best_val * progress + noise))
        val_loss = max(0.01, 1.5 * (1 - progress) + abs(noise))
        train_loss = val_loss * 0.85
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return train_losses, val_losses, val_accs


with mlflow.start_run(run_name="baseline-logistic") as run:
    # ── Log hyperparameters ───────────────────────────────────────────
    mlflow.log_params({
        "model_type":      "logistic_regression",
        "learning_rate":   1e-3,
        "n_layers":        3,
        "dropout":         0.1,
        "batch_size":      64,
        "max_epochs":      20,
        "optimizer":       "adam",
        "dataset_version": "v2.1",
    })

    # ── Log tags (metadata, not hyperparameters) ──────────────────────
    mlflow.set_tags({
        "team":          "ml-core",
        "framework":     "sklearn",
        "run_type":      "baseline",
        "data_hash":     hashlib.md5(b"dummy_data_v2.1").hexdigest()[:8],
        "git_branch":    "main",
    })

    # ── Simulate training and log metric time series ──────────────────
    train_losses, val_losses, val_accs = simulate_training(
        lr=1e-3, n_layers=3, dropout=0.1, n_epochs=20
    )
    for epoch, (tl, vl, va) in enumerate(zip(train_losses, val_losses, val_accs)):
        mlflow.log_metrics({
            "train_loss": tl,
            "val_loss":   vl,
            "val_acc":    va,
        }, step=epoch)

    # Log final summary metrics
    mlflow.log_metrics({
        "final_val_acc":  val_accs[-1],
        "best_val_acc":   max(val_accs),
        "final_val_loss": val_losses[-1],
    })

    # ── Log artefacts (files) ─────────────────────────────────────────
    with tempfile.TemporaryDirectory() as artefact_tmp:
        # Config file
        config_path = os.path.join(artefact_tmp, "config.json")
        with open(config_path, "w") as f:
            json.dump({"lr": 1e-3, "n_layers": 3, "dropout": 0.1}, f, indent=2)
        mlflow.log_artifact(config_path)

        # Training curve summary
        curve_path = os.path.join(artefact_tmp, "training_summary.txt")
        with open(curve_path, "w") as f:
            f.write(f"Epoch | Train Loss | Val Loss | Val Acc\\n")
            f.write("-" * 45 + "\\n")
            for i in range(0, 20, 4):
                f.write(f"{i:>5} | {train_losses[i]:>10.4f} | "
                        f"{val_losses[i]:>8.4f} | {val_accs[i]:>7.4f}\\n")
        mlflow.log_artifact(curve_path, artifact_path="reports")

        # Metrics as a dict
        mlflow.log_dict(
            {"val_accs": val_accs, "val_losses": val_losses},
            artifact_file="metrics_history.json",
        )
        # Text note
        mlflow.log_text(
            "This is the baseline logistic regression run. "
            "No feature engineering applied.",
            artifact_file="notes.txt",
        )

    run_id = run.info.run_id
    print(f"  Run completed: {run_id[:16]}...")
    print(f"  Logged: params={len(run.data.params)} "
          f"metrics={len(run.data.metrics)} "
          f"tags={len(run.data.tags)}")

# Inspect what was logged
finished_run = client.get_run(run_id)
print()
print(f"  Run data breakdown:")
print(f"  {'Type':<12} {'Count':>8} {'Keys'}")
print(f"  {'─'*60}")
print(f"  {'Parameters':<12} {len(finished_run.data.params):>8} "
      f"{list(finished_run.data.params.keys())[:5]}")
print(f"  {'Metrics':<12} {len(finished_run.data.metrics):>8} "
      f"{list(finished_run.data.metrics.keys())[:5]}")
print(f"  {'Tags':<12} {len(finished_run.data.tags):>8} "
      f"{[k for k in finished_run.data.tags if not k.startswith('mlflow.')][:4]}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Nested runs — HPO sweep with child trials
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Nested runs: HPO sweep as parent-child hierarchy")
print("━" * 65)
print()

# Define search space
hpo_configs = [
    {"lr": 1e-4, "n_layers": 2, "dropout": 0.05},
    {"lr": 3e-3, "n_layers": 4, "dropout": 0.15},
    {"lr": 1e-2, "n_layers": 6, "dropout": 0.30},
    {"lr": 5e-3, "n_layers": 3, "dropout": 0.10},
    {"lr": 8e-4, "n_layers": 5, "dropout": 0.20},
]

sweep_run_id = None
with mlflow.start_run(run_name="hpo-grid-sweep") as sweep_run:
    sweep_run_id = sweep_run.info.run_id
    mlflow.set_tags({
        "run_type":    "hpo-sweep",
        "n_trials":    str(len(hpo_configs)),
        "search_type": "grid",
    })

    trial_results = []
    for i, config in enumerate(hpo_configs):
        with mlflow.start_run(
            run_name = f"trial-{i:02d}",
            nested   = True,
        ) as trial_run:
            mlflow.log_params(config)
            mlflow.set_tag("trial_index", str(i))

            # Simulate training (fewer epochs for speed)
            _, val_losses, val_accs = simulate_training(
                lr=config["lr"], n_layers=config["n_layers"],
                dropout=config["dropout"], n_epochs=10, seed=i,
            )
            for epoch, (vl, va) in enumerate(zip(val_losses, val_accs)):
                mlflow.log_metrics({"val_loss": vl, "val_acc": va}, step=epoch)

            final_acc = val_accs[-1]
            mlflow.log_metric("final_val_acc", final_acc)
            trial_results.append({
                "run_id": trial_run.info.run_id,
                "config": config,
                "val_acc": final_acc,
            })

    # Log summary on the parent run
    best_trial = max(trial_results, key=lambda x: x["val_acc"])
    mlflow.log_params({f"best_{k}": v for k, v in best_trial["config"].items()})
    mlflow.log_metric("best_val_acc", best_trial["val_acc"])

print(f"  HPO sweep: {len(hpo_configs)} trials (parent + {len(hpo_configs)} children)")
print()
print(f"  {'Trial':>7} {'LR':>10} {'Layers':>8} {'Dropout':>10} {'Val Acc':>10}")
print(f"  {'─'*50}")
for i, r in enumerate(sorted(trial_results, key=lambda x: -x["val_acc"])):
    marker = " ← BEST" if r["val_acc"] == best_trial["val_acc"] else ""
    print(f"  {i+1:>7} {r['config']['lr']:>10.4e} {r['config']['n_layers']:>8} "
          f"{r['config']['dropout']:>10.3f} {r['val_acc']:>10.4f}{marker}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: search_runs — programmatic run comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — search_runs: querying and comparing runs")
print("━" * 65)
print()

# Add more runs to make search interesting
extra_configs = [
    {"lr": 2e-3, "n_layers": 4, "dropout": 0.12},
    {"lr": 4e-4, "n_layers": 5, "dropout": 0.08},
]
for config in extra_configs:
    with mlflow.start_run(run_name="additional-trial"):
        mlflow.log_params(config)
        _, _, accs = simulate_training(**config, n_epochs=15)
        mlflow.log_metric("final_val_acc", accs[-1])
        mlflow.set_tag("run_type", "additional")

# Search all runs in the experiment
all_runs = mlflow.search_runs(
    experiment_names = ["classification-benchmark"],
    filter_string    = "attributes.status = 'FINISHED'",
    order_by         = ["metrics.final_val_acc DESC"],
)

print(f"  Total finished runs: {len(all_runs)}")
print()

# Show top runs
print(f"  Top runs by final_val_acc:")
top_cols = ["run_id", "metrics.final_val_acc", "params.lr",
            "params.n_layers", "params.dropout"]
available = [c for c in top_cols if c in all_runs.columns]
top_runs  = all_runs[available].head(6)
print(f"  {'Rank':>5} {'Run ID':>18} {'Val Acc':>10} {'LR':>10} "
      f"{'Layers':>8} {'Dropout':>10}")
print(f"  {'─'*65}")
for rank, (_, row) in enumerate(top_runs.iterrows(), 1):
    rid   = str(row.get("run_id", "?"))[:12] + "..."
    acc   = row.get("metrics.final_val_acc", float("nan"))
    lr    = row.get("params.lr",      "?")
    nl    = row.get("params.n_layers", "?")
    drop  = row.get("params.dropout",  "?")
    lr_f  = f"{float(lr):.2e}" if lr != "?" else "?"
    print(f"  {rank:>5} {rid:>18} {acc:>10.4f} {lr_f:>10} {str(nl):>8} {str(drop):>10}")

print()

# Filter runs by metric threshold
good_runs = mlflow.search_runs(
    experiment_names = ["classification-benchmark"],
    filter_string    = "metrics.final_val_acc > 0.85 AND attributes.status = 'FINISHED'",
    order_by         = ["metrics.final_val_acc DESC"],
)
print(f"  Runs with val_acc > 0.85: {len(good_runs)}")
print()

# Access metric history for the best run
if len(all_runs) > 0:
    best_run_id = all_runs.iloc[0]["run_id"]
    history     = client.get_metric_history(best_run_id, "val_acc")
    if history:
        steps  = [m.step  for m in history]
        values = [m.value for m in history]
        print(f"  Best run val_acc training curve (first 5 epochs):")
        print(f"  {'Epoch':>7} {'Val Acc':>12}")
        print(f"  {'─'*22}")
        for s, v in list(zip(steps, values))[:5]:
            bar = "█" * int(v * 20)
            print(f"  {s:>7} {v:>12.4f}  {bar}")
        print(f"  ... (total {len(history)} steps recorded)")
print()

# Cleanup
shutil.rmtree(TRACKING_DIR, ignore_errors=True)
print(f"  Tracking directory cleaned up ✅")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · MLflow Models — Flavours, Signatures & Custom PyFunc": {
        "description": (
            "MLflow Models packaging and the flavour system. "
            "mlflow.sklearn.log_model: package with environment. "
            "mlflow.sklearn.load_model: framework-native reload. "
            "mlflow.pyfunc.load_model: framework-agnostic pyfunc. "
            "infer_signature: automatic input/output schema detection. "
            "ModelSignature: manual schema with ColSpec and TensorSpec. "
            "Input example: validation and UI display. "
            "Custom PythonModel: wrap arbitrary code as MLflow model. "
            "MLmodel manifest inspection: flavours and metadata. "
            "Multi-flavour model: pytorch + onnx in one package. "
            "log_model with registered_model_name: auto-registration. "
            "Model URI formats: runs:/ and models:/ schemes."
        ),
        "language": "python",
        "code": '''
import os
import json
import math
import time
import tempfile
import shutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.models import infer_signature
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, ColSpec, TensorSpec
    from mlflow.tracking import MlflowClient
    print(f"  MLflow version: {mlflow.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "mlflow", "--quiet"], check=True)
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.models import infer_signature
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, ColSpec, TensorSpec
    from mlflow.tracking import MlflowClient

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

rng = np.random.default_rng(42)
print("=" * 65)
print("  MLFLOW MODELS — FLAVOURS, SIGNATURES & CUSTOM PYFUNC")
print("=" * 65)
print()

TRACKING_DIR = tempfile.mkdtemp(prefix="mlflow_models_")
mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")
mlflow.set_experiment("model-packaging-demo")
client = MlflowClient(tracking_uri=f"file://{TRACKING_DIR}")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Sklearn model logging with full metadata
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — sklearn model: log, signature, input_example")
print("━" * 65)
print()

if HAS_SKLEARN:
    # Generate dataset
    N, F, C = 1000, 15, 3
    X, y = make_classification(n_samples=N, n_features=F, n_classes=C,
                                n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y, test_size=0.2, random_state=42
    )

    # Build feature names for schema
    feature_names = [f"feature_{i:02d}" for i in range(F)]
    X_train_df    = pd.DataFrame(X_train, columns=feature_names)
    X_test_df     = pd.DataFrame(X_test,  columns=feature_names)

    # Train model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=30, max_depth=5,
                                           random_state=42)),
    ])
    pipeline.fit(X_train_df, y_train)
    train_acc = pipeline.score(X_train_df, y_train)
    test_acc  = pipeline.score(X_test_df,  y_test)

    print(f"  Trained RandomForest Pipeline:")
    print(f"    Features: {F}  |  Classes: {C}  |  Train samples: {len(X_train)}")
    print(f"    Train accuracy: {train_acc:.4f}  |  Test accuracy: {test_acc:.4f}")
    print()

    # Infer model signature automatically
    y_pred_proba = pipeline.predict_proba(X_test_df[:5])
    signature    = infer_signature(
        model_input  = X_train_df[:5],    # sample input
        model_output = y_pred_proba[:5],   # corresponding output
    )
    print(f"  Inferred signature:")
    print(f"    inputs:  {signature.inputs}")
    print(f"    outputs: {signature.outputs}")
    print()

    # Log the model
    with mlflow.start_run(run_name="rf-pipeline-v1") as run:
        mlflow.log_params({
            "n_estimators": 30,
            "max_depth":    5,
            "scaler":       "standard",
        })
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "test_accuracy":  test_acc,
        })

        model_info = mlflow.sklearn.log_model(
            sk_model          = pipeline,
            artifact_path     = "model",
            signature         = signature,
            input_example     = X_train_df[:3],   # for UI preview
        )
        run_id = run.info.run_id

    print(f"  Model logged:")
    print(f"    model_uri:   {model_info.model_uri}")
    print(f"    flavors:     {list(model_info.flavors.keys())}")
    print()

    # Inspect the MLmodel manifest
    artefact_path = client.download_artifacts(run_id, "model")
    mlmodel_path  = os.path.join(artefact_path, "MLmodel")
    if os.path.exists(mlmodel_path):
        with open(mlmodel_path) as f:
            mlmodel_content = f.read()
        print(f"  MLmodel manifest ({os.path.getsize(mlmodel_path)} bytes):")
        print(f"  {'─'*55}")
        for line in mlmodel_content.split("\\n")[:20]:
            print(f"  {line}")
        print()

    # ── Loading in different flavours ────────────────────────────────
    print(f"  Loading model in different flavours:")

    # sklearn flavour → returns actual sklearn estimator
    sk_loaded   = mlflow.sklearn.load_model(model_info.model_uri)
    sk_pred     = sk_loaded.predict(X_test_df[:3])
    print(f"    sklearn flavour:  {type(sk_loaded).__name__}")
    print(f"      predict() → {sk_pred}")

    # pyfunc flavour → framework-agnostic interface
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    pyfunc_pred  = pyfunc_model.predict(X_test_df[:3])
    print(f"    pyfunc flavour:  {type(pyfunc_model).__name__}")
    print(f"      predict() shape={pyfunc_pred.shape}  dtype={pyfunc_pred.dtype}")
    print()

    # Signature validation (pyfunc enforces schema)
    print(f"  Signature validation:")
    try:
        # Wrong type → should warn or raise
        bad_input = X_test_df[:2].copy()
        bad_input.columns = [f"wrong_{i}" for i in range(F)]
        _ = pyfunc_model.predict(bad_input)
        print(f"    Wrong column names: warned (non-breaking)")
    except Exception as e:
        print(f"    Wrong column names: {type(e).__name__}")
    print(f"    Correct input:      {pyfunc_model.predict(X_test_df[:1]).shape}  ✅")

else:
    print(f"  sklearn not available. Install: pip install scikit-learn")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Custom PythonModel — wrap any code as MLflow model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Custom PythonModel: wrap arbitrary logic")
print("━" * 65)
print()

class EnsembleModel(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow model that wraps an ensemble of numpy-based models.
    Demonstrates the PythonModel interface:
        load_context(): called once at model load time
        predict():      called for every inference request
    """
    def __init__(self, model_weights: np.ndarray, feature_means: np.ndarray,
                 feature_stds: np.ndarray, n_classes: int):
        self.model_weights  = model_weights    # (n_models, n_features, n_classes)
        self.feature_means  = feature_means
        self.feature_stds   = feature_stds
        self.n_classes      = n_classes

    def load_context(self, context):
        """Load any additional artefacts from context.artifacts."""
        # context.artifacts is a dict: {"key": "/path/to/file"}
        import numpy as np
        config_path = context.artifacts.get("config")
        if config_path:
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}
        print(f"  [EnsembleModel] Loaded with config: {self.config}")

    def predict(self, context, model_input: pd.DataFrame,
                params: Optional[Dict] = None) -> np.ndarray:
        """
        model_input: DataFrame where each column is a feature
        Returns:     numpy array of class probabilities
        """
        # 1. Convert to numpy
        X = model_input.values.astype(np.float32)

        # 2. Normalise
        X_norm = (X - self.feature_means) / (self.feature_stds + 1e-8)

        # 3. Forward pass through each model in ensemble
        logits_all = []
        for w in self.model_weights:
            logits = X_norm @ w
            logits_all.append(logits)

        # 4. Average ensemble
        avg_logits = np.stack(logits_all).mean(axis=0)

        # 5. Softmax
        exp_l  = np.exp(avg_logits - avg_logits.max(1, keepdims=True))
        probs  = exp_l / exp_l.sum(1, keepdims=True)
        return probs


# Build a simple ensemble
N_FEAT, N_CLS, N_MODELS = 10, 3, 5
X_sample = rng.standard_normal((100, N_FEAT)).astype(np.float32)
y_sample = rng.integers(0, N_CLS, 100)

feat_means = X_sample.mean(axis=0)
feat_stds  = X_sample.std(axis=0)

ensemble_weights = [
    (rng.standard_normal((N_FEAT, N_CLS)) * 0.1).astype(np.float32)
    for _ in range(N_MODELS)
]
ensemble_model = EnsembleModel(
    model_weights  = ensemble_weights,
    feature_means  = feat_means,
    feature_stds   = feat_stds,
    n_classes      = N_CLS,
)

# Create config artefact
with tempfile.TemporaryDirectory() as art_tmp:
    config_path = os.path.join(art_tmp, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({"n_models": N_MODELS, "n_features": N_FEAT,
                   "n_classes": N_CLS, "version": "1.0"}, f)

    # Define input/output signature
    input_schema  = Schema([ColSpec("float", f"feat_{i:02d}") for i in range(N_FEAT)])
    output_schema = Schema([TensorSpec(np.dtype("float32"), (-1, N_CLS), "probabilities")])
    signature     = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Sample input for validation
    X_df = pd.DataFrame(X_sample[:5], columns=[f"feat_{i:02d}" for i in range(N_FEAT)])

    with mlflow.start_run(run_name="custom-ensemble") as run:
        mlflow.log_params({"n_models": N_MODELS, "n_features": N_FEAT})

        model_info = mlflow.pyfunc.log_model(
            artifact_path = "ensemble_model",
            python_model  = ensemble_model,
            artifacts     = {"config": config_path},
            signature     = signature,
            input_example = X_df[:3],
        )
        custom_run_id = run.info.run_id

print(f"  Custom EnsembleModel logged:")
print(f"    model_uri: {model_info.model_uri[-50:]}...")
print(f"    flavors:   {list(model_info.flavors.keys())}")
print()

# Reload and test
loaded_ensemble = mlflow.pyfunc.load_model(model_info.model_uri)
probs = loaded_ensemble.predict(X_df)
print(f"  Reloaded and ran inference:")
print(f"    Input shape:   {X_df.shape}")
print(f"    Output shape:  {probs.shape}  (batch × n_classes)")
print(f"    Output dtype:  {probs.dtype}")
print(f"    Row sums:      {probs.sum(axis=1).round(4).tolist()}  (should be 1.0)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Model URI formats
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Model URI formats and loading patterns")
print("━" * 65)
print()

print(f"  MLflow model URI schemes:")
uri_examples = [
    ("runs:/<run_id>/model",               "Load from run artefact (most common)"),
    ("runs:/<run_id>/reports/model",       "Load from nested artefact path"),
    ("models:/<name>/Production",          "Load by stage (registry)"),
    ("models:/<name>/3",                   "Load by version number (registry)"),
    ("models:/<name>@champion",            "Load by alias (MLflow 2.x+)"),
    ("file:///local/path/model",           "Load from local filesystem"),
    ("s3://bucket/prefix/model",           "Load from S3 (with credentials)"),
    ("gs://bucket/prefix/model",           "Load from GCS"),
    ("azureml://registries/name/models/v1","Load from Azure ML registry"),
]
print(f"  {'URI Pattern':<44} {'Description'}")
print(f"  {'─'*75}")
for uri, desc in uri_examples:
    print(f"  {uri:<44} {desc}")
print()

# Show run:/ URI for our logged model
if HAS_SKLEARN:
    print(f"  Our logged models:")
    print(f"    sklearn:  runs:/{run_id[:16]}.../<hash>/model")
    print(f"    ensemble: runs:/{custom_run_id[:16]}.../<hash>/ensemble_model")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Model flavour inspection
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — MLmodel manifest: flavours and environment")
print("━" * 65)
print()

print(f"  The MLmodel manifest is the model's ID card:")
print()

MLMODEL_EXAMPLE = {
    "artifact_path": "model",
    "flavors": {
        "python_function": {
            "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
            "loader_module": "mlflow.sklearn",
            "model_path": "model.pkl",
            "predict_fn": "predict",
            "python_version": "3.11.0",
        },
        "sklearn": {
            "code": None,
            "pickled_model": "model.pkl",
            "sklearn_version": "1.4.0",
            "serialization_format": "cloudpickle",
        },
    },
    "mlflow_version":    "2.10.0",
    "model_uuid":        "3b8d2e1a4f5c6d7890abcdef",
    "run_id":            "abc123def456789",
    "saved_input_example_info": {
        "artifact_path": "input_example.json",
        "pandas_orient": "split",
        "serving_input_path": "serving_input_example.json",
        "type": "dataframe",
    },
    "signature": {
        "inputs":  "[{\"name\": \"feature_00\", \"type\": \"float\"}]",
        "outputs": "[{\"tensor-spec\": {\"dtype\": \"float64\", \"shape\": [-1, 3]}}]",
    },
    "utc_time_created": "2024-01-15 10:30:45.123456",
}

for section, value in MLMODEL_EXAMPLE.items():
    if isinstance(value, dict):
        print(f"  {section}:")
        for k, v in value.items():
            if isinstance(v, dict):
                print(f"    {k}: {{...}}")
            else:
                print(f"    {k}: {str(v)[:50]}")
    else:
        print(f"  {section}: {str(value)[:60]}")
print()

# Environment files
print(f"  Environment files included with every MLflow model:")
print(f"    conda.yaml:       conda channels + pip dependencies")
print(f"    requirements.txt: plain pip dependencies list")
print(f"    python_env.yaml:  Python version + pip version")
print()
print(f"  Loading with environment recreation:")
print(f"    mlflow models predict -m models:/name/Production \\")
    # This is a command reference, not executable Python
print(f"        --input-path input.csv --env-manager conda")

# Cleanup
shutil.rmtree(TRACKING_DIR, ignore_errors=True)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Model Registry — Version Control, Stages & Governance": {
        "description": (
            "MLflow Model Registry: full lifecycle governance workflow. "
            "Registering models: from log_model and register_model(). "
            "Model versions: auto-increment, immutable records. "
            "Stage transitions: None → Staging → Production → Archived. "
            "Programmatic promotion with MlflowClient. "
            "Champion/challenger pattern: A/B testing two versions. "
            "Model version tagging: attach metrics for audit trail. "
            "Model aliases (MLflow 2.x): named pointers to versions. "
            "Registry search: get_latest_versions, search_model_versions. "
            "Automated CI/CD promotion: promote_if_better function. "
            "Registry audit trail: version history and annotations. "
            "Loading by stage vs version vs alias comparison."
        ),
        "language": "python",
        "code": '''
import os
import time
import json
import math
import tempfile
import shutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
    from mlflow.models import infer_signature
    from mlflow.exceptions import MlflowException
    print(f"  MLflow version: {mlflow.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "mlflow", "--quiet"], check=True)
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
    from mlflow.models import infer_signature
    from mlflow.exceptions import MlflowException

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

rng = np.random.default_rng(42)

print("=" * 65)
print("  MODEL REGISTRY — VERSION CONTROL, STAGES & GOVERNANCE")
print("=" * 65)
print()

TRACKING_DIR = tempfile.mkdtemp(prefix="mlflow_registry_")
mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")
mlflow.set_experiment("registry-demo")
client = MlflowClient(tracking_uri=f"file://{TRACKING_DIR}")

MODEL_NAME = "fraud-detector"

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Register multiple model versions
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Registering model versions from runs")
print("━" * 65)
print()

def simulate_model_performance(seed: int, model_type: str) -> Dict:
    """Simulate training metrics for different model configurations."""
    _rng = np.random.default_rng(seed)
    base = {"logistic": 0.83, "random_forest": 0.91, "gradient_boost": 0.94}
    noise = _rng.uniform(-0.02, 0.02)
    acc   = min(0.99, max(0.70, base.get(model_type, 0.88) + noise))
    return {
        "val_accuracy": acc,
        "val_f1":       acc - _rng.uniform(0.01, 0.04),
        "val_auc":      min(0.999, acc + _rng.uniform(0.01, 0.03)),
        "val_loss":     1.5 * (1 - acc) + _rng.uniform(0, 0.1),
    }


# Register three versions of the fraud-detector model
model_configs = [
    {"model_type": "logistic",      "description": "Baseline logistic regression",    "seed": 1},
    {"model_type": "random_forest", "description": "Random forest with 100 trees",    "seed": 2},
    {"model_type": "gradient_boost","description": "GBM with early stopping",         "seed": 3},
]

registered_versions = []
print(f"  Registering {len(model_configs)} versions of '{MODEL_NAME}':")
print()

for cfg in model_configs:
    metrics = simulate_model_performance(cfg["seed"], cfg["model_type"])

    with mlflow.start_run(run_name=f"train-{cfg['model_type']}") as run:
        mlflow.log_params({
            "model_type": cfg["model_type"],
            "dataset_version": "v3.1",
            "n_features": 20,
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "framework": "sklearn",
            "description": cfg["description"],
        })

        # Build a tiny pyfunc model (no sklearn needed — uses custom wrapper)
        class TinyModel(mlflow.pyfunc.PythonModel):
            def __init__(self, metrics): self.metrics = metrics
            def predict(self, context, model_input, params=None):
                return np.zeros((len(model_input), 1))

        model_info = mlflow.pyfunc.log_model(
            artifact_path         = "model",
            python_model          = TinyModel(metrics),
            registered_model_name = MODEL_NAME,  # auto-register!
        )
        version_number = model_info.registered_model_version
        run_id         = run.info.run_id

        # Annotate the version with metrics
        client.update_model_version(
            name        = MODEL_NAME,
            version     = version_number,
            description = f"{cfg['description']}. "
                          f"val_accuracy={metrics['val_accuracy']:.4f}, "
                          f"val_auc={metrics['val_auc']:.4f}",
        )
        for metric_key, metric_val in metrics.items():
            client.set_model_version_tag(
                MODEL_NAME, version_number,
                key=metric_key, value=f"{metric_val:.6f}",
            )

        registered_versions.append({
            "version":    version_number,
            "model_type": cfg["model_type"],
            "metrics":    metrics,
            "run_id":     run_id,
        })
        print(f"    v{version_number}: {cfg['model_type']:<18} "
              f"acc={metrics['val_accuracy']:.4f}  "
              f"auc={metrics['val_auc']:.4f}")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Stage transitions
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Stage transitions: None → Staging → Production")
print("━" * 65)
print()

print(f"  Initial state: all versions in stage=None")
versions_info = client.search_model_versions(f"name='{MODEL_NAME}'")
for v in versions_info:
    print(f"    v{v.version}: {v.current_stage:<12} {v.description[:50]}...")
print()

# Promote v2 (random_forest) to Staging for evaluation
client.transition_model_version_stage(
    name    = MODEL_NAME,
    version = "2",
    stage   = "Staging",
)
print(f"  → Transitioned v2 (random_forest) to Staging")

# Promote v1 (logistic) to Production as initial deployment
client.transition_model_version_stage(
    name    = MODEL_NAME,
    version = "1",
    stage   = "Production",
)
print(f"  → Transitioned v1 (logistic) to Production (initial deployment)")
print()

# Show current state
versions_info = client.search_model_versions(f"name='{MODEL_NAME}'")
print(f"  Current stage state:")
for v in sorted(versions_info, key=lambda x: int(x.version)):
    print(f"    v{v.version}: {v.current_stage:<15} {v.run_id[:12]}...")
print()

# After validation: promote v2 to Production, demote v1
client.transition_model_version_stage(
    name                      = MODEL_NAME,
    version                   = "2",
    stage                     = "Production",
    archive_existing_versions = True,   # v1 → Archived automatically
)
print(f"  → Promoted v2 to Production (archive_existing=True)")
print()

versions_info = client.search_model_versions(f"name='{MODEL_NAME}'")
print(f"  Final stage state:")
stage_counts = {}
for v in sorted(versions_info, key=lambda x: int(x.version)):
    stage_counts[v.current_stage] = stage_counts.get(v.current_stage, 0) + 1
    marker = " ← LIVE" if v.current_stage == "Production" else ""
    print(f"    v{v.version}: {v.current_stage:<15}{marker}")
print()
print(f"  Stage distribution: {stage_counts}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: CI/CD promotion function
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Automated CI/CD: promote_if_better()")
print("━" * 65)
print()

def promote_if_better(
    model_name:        str,
    candidate_run_id:  str,
    metric:            str  = "val_auc",
    min_improvement:   float = 0.005,
    verbose:           bool  = True,
) -> bool:
    """
    CI/CD gate: register candidate model and promote to Production
    only if it beats the current Production version by min_improvement.

    Returns True if promoted, False otherwise.
    """
    _client = MlflowClient()

    # Get current Production metric
    prod_versions = _client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        prod_run_id     = prod_versions[0].run_id
        prod_run        = _client.get_run(prod_run_id)
        prod_metric_val = prod_run.data.metrics.get(metric, 0.0)
        prod_version    = prod_versions[0].version
    else:
        prod_metric_val = 0.0
        prod_version    = None

    # Get candidate metric
    cand_run        = _client.get_run(candidate_run_id)
    cand_metric_val = cand_run.data.metrics.get(metric, 0.0)

    if verbose:
        print(f"  Candidate {metric}: {cand_metric_val:.6f}")
        print(f"  Production {metric}: {prod_metric_val:.6f} (v{prod_version})")
        improvement = cand_metric_val - prod_metric_val
        print(f"  Improvement:        {improvement:+.6f}  "
              f"(threshold: {min_improvement:+.4f})")

    should_promote = cand_metric_val > prod_metric_val + min_improvement

    if should_promote:
        # Register candidate
        new_version = mlflow.register_model(
            model_uri = f"runs:/{candidate_run_id}/model",
            name      = model_name,
        )
        # Annotate with metrics
        for k, v in cand_run.data.metrics.items():
            _client.set_model_version_tag(
                model_name, new_version.version, key=k, value=f"{v:.6f}"
            )
        # Transition to Production
        _client.transition_model_version_stage(
            model_name, new_version.version, "Production",
            archive_existing_versions=True,
        )
        if verbose:
            print(f"  ✅ PROMOTED: v{new_version.version} is now Production")
        return True
    else:
        if verbose:
            print(f"  ✗  NOT PROMOTED: improvement below threshold")
        return False


# Simulate a CI/CD pipeline: train v4 (gradient boost) and promote if better
print(f"  Simulating CI/CD pipeline with gradient boosting model:")
print()
metrics_v4 = simulate_model_performance(seed=42, model_type="gradient_boost")
with mlflow.start_run(run_name="ci-cd-candidate") as candidate_run:
    mlflow.log_params({"model_type": "gradient_boost", "n_estimators": 200})
    mlflow.log_metrics(metrics_v4)
    mlflow.pyfunc.log_model(
        "model",
        python_model=type("M", (mlflow.pyfunc.PythonModel,), {
            "predict": lambda s, c, d, p=None: np.zeros((len(d), 1))
        })(),
    )
    candidate_run_id = candidate_run.info.run_id

was_promoted = promote_if_better(
    MODEL_NAME, candidate_run_id,
    metric="val_auc", min_improvement=0.005,
)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Registry search and audit trail
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Registry search, aliases, and audit trail")
print("━" * 65)
print()

# Get all versions of the model
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
print(f"  All versions of '{MODEL_NAME}':")
print(f"  {'Ver':>5} {'Stage':<15} {'Description':<45} {'Val AUC':>10}")
print(f"  {'─'*78}")
for v in sorted(all_versions, key=lambda x: int(x.version)):
    auc_tag = v.tags.get("val_auc", "N/A")
    desc    = (v.description or "")[:43]
    print(f"  {v.version:>5} {v.current_stage:<15} {desc:<45} {auc_tag:>10}")
print()

# Get latest by stage
for stage in ["Production", "Staging", "Archived"]:
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if versions:
        v = versions[0]
        print(f"  Latest in {stage}: v{v.version}  ({v.description[:40]}...)")
    else:
        print(f"  Latest in {stage}: none")
print()

# Model aliases (MLflow 2.x)
try:
    # Set alias on current Production version
    prod_v = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if prod_v:
        client.set_registered_model_alias(MODEL_NAME, "champion", prod_v[0].version)
        print(f"  Alias set: @champion → v{prod_v[0].version}")
        # Load by alias
        model_uri_alias = f"models:/{MODEL_NAME}@champion"
        print(f"  Load by alias: mlflow.pyfunc.load_model('{model_uri_alias}')")
except Exception as e:
    print(f"  Aliases require MLflow 2.x (current: {mlflow.__version__})")
    print(f"  Usage: client.set_registered_model_alias(name, 'champion', version)")
print()

# Registered model metadata
reg_model = client.get_registered_model(MODEL_NAME)
print(f"  Registered model metadata:")
print(f"    name:             {reg_model.name}")
print(f"    creation_time:    {reg_model.creation_timestamp}")
print(f"    last_updated:     {reg_model.last_updated_timestamp}")
print(f"    latest_versions:  {len(reg_model.latest_versions)} stage(s) with versions")
for v in reg_model.latest_versions:
    print(f"      {v.current_stage}: v{v.version}")
print()

# Cleanup
shutil.rmtree(TRACKING_DIR, ignore_errors=True)
print(f"  Registry demo complete. Tracking directory cleaned up ✅")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · MLflow Evaluation, Autologging & Production Patterns": {
        "description": (
            "mlflow.evaluate() for automated model quality assessment. "
            "Built-in metrics: accuracy, F1, AUC, confusion matrix. "
            "Custom metric functions with MetricValue. "
            "Baseline model comparison: champion vs challenger. "
            "mlflow.autolog(): zero-code tracking for sklearn, PyTorch. "
            "Autolog configuration: selective capture, input examples. "
            "Training curve capture: per-epoch metrics via autolog. "
            "Optuna integration: HPO trial logging as child runs. "
            "Production serving pattern: mlflow models serve reference. "
            "Batch inference via loaded pyfunc model. "
            "Complete MLOps pipeline: train → evaluate → register → serve. "
            "Production best practices checklist."
        ),
        "language": "python",
        "code": '''
import os
import time
import json
import math
import tempfile
import shutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    print(f"  MLflow version: {mlflow.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "mlflow", "--quiet"], check=True)
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        precision_score, recall_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

rng = np.random.default_rng(42)
print("=" * 65)
print("  MLFLOW EVALUATION, AUTOLOGGING & PRODUCTION PATTERNS")
print("=" * 65)
print()

TRACKING_DIR = tempfile.mkdtemp(prefix="mlflow_eval_")
mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")
client = MlflowClient(tracking_uri=f"file://{TRACKING_DIR}")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: mlflow.autolog()
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — mlflow.autolog(): zero-code experiment tracking")
print("━" * 65)
print()

if HAS_SKLEARN:
    mlflow.set_experiment("autolog-demo")
    mlflow.sklearn.autolog(
        log_input_examples   = True,
        log_model_signatures = True,
        log_models           = True,
        max_tuning_runs      = 3,
        silent               = True,
    )

    N, F, C = 800, 12, 2
    X, y = make_classification(n_samples=N, n_features=F, n_classes=C,
                                n_informative=8, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X.astype(np.float32), y,
                                                test_size=0.2, random_state=42)
    feature_names = [f"f{i:02d}" for i in range(F)]
    X_tr_df = pd.DataFrame(X_tr, columns=feature_names)
    X_te_df = pd.DataFrame(X_te, columns=feature_names)

    print(f"  Training with autolog enabled...")
    print(f"  (No manual mlflow.log_* calls — all captured automatically)")
    print()

    with mlflow.start_run(run_name="autolog-rf") as auto_run:
        # Just train normally — autolog captures everything
        model = RandomForestClassifier(
            n_estimators = 50,
            max_depth    = 6,
            min_samples_split = 5,
            random_state = 42,
        )
        model.fit(X_tr_df, y_tr)
        # autolog automatically logs: all params, train score, test score,
        # pickled model, input example, model signature

    # Inspect what was captured
    auto_run_data = client.get_run(auto_run.info.run_id)
    params   = auto_run_data.data.params
    metrics  = auto_run_data.data.metrics
    tags     = {k: v for k, v in auto_run_data.data.tags.items()
                if not k.startswith("mlflow.")}

    print(f"  Autolog captured automatically:")
    print(f"  {'Category':<15} {'Count':>8} {'Keys (sample)'}")
    print(f"  {'─'*65}")
    print(f"  {'Parameters':<15} {len(params):>8} "
          f"{list(params.keys())[:4]}")
    print(f"  {'Metrics':<15} {len(metrics):>8} "
          f"{list(metrics.keys())}")
    print(f"  {'Tags':<15} {len(tags):>8} "
          f"{list(tags.keys())[:3]}")
    print()

    print(f"  Captured parameters (from estimator):")
    for k, v in list(params.items())[:8]:
        print(f"    {k:<30}: {v}")
    print()

    # Disable autolog for subsequent sections
    mlflow.sklearn.autolog(disable=True)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: mlflow.evaluate() — automated metrics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — mlflow.evaluate(): automated model assessment")
print("━" * 65)
print()

if HAS_SKLEARN:
    mlflow.set_experiment("evaluation-demo")

    # Train champion and challenger models
    champion   = LogisticRegression(C=0.1, random_state=42)
    challenger = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
    dummy      = DummyClassifier(strategy="most_frequent")

    champion.fit(X_tr_df, y_tr)
    challenger.fit(X_tr_df, y_tr)
    dummy.fit(X_tr_df, y_tr)

    # Create evaluation dataset with targets
    eval_df        = X_te_df.copy()
    eval_df["target"] = y_te

    eval_results = {}

    for name, model in [
        ("champion",   champion),
        ("challenger", challenger),
        ("baseline",   dummy),
    ]:
        with mlflow.start_run(run_name=f"eval-{name}") as run:
            model_info = mlflow.sklearn.log_model(model, "model")

            # Custom metric: specificity = TN / (TN + FP)
            def specificity_metric(eval_df, builtin_metrics):
                from mlflow.models.evaluation import MetricValue
                y_true = eval_df["target"]
                y_pred = eval_df["prediction"]
                tn = ((y_true == 0) & (y_pred == 0)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                spec = tn / (tn + fp + 1e-8)
                return MetricValue(aggregate_results={"specificity": float(spec)})

            try:
                result = mlflow.evaluate(
                    model           = model_info.model_uri,
                    data            = eval_df,
                    targets         = "target",
                    model_type      = "classifier",
                    evaluators      = "default",
                    extra_metrics   = [
                        mlflow.models.make_metric(
                            eval_fn   = specificity_metric,
                            name      = "specificity",
                            greater_is_better = True,
                        )
                    ],
                )
                eval_results[name] = result.metrics
            except Exception as e:
                # mlflow.evaluate may not work perfectly in all local environments
                # Fall back to manual metric computation
                y_pred = model.predict(X_te_df)
                y_proba = model.predict_proba(X_te_df)[:, 1]
                eval_results[name] = {
                    "accuracy_score":  accuracy_score(y_te, y_pred),
                    "f1_score":        f1_score(y_te, y_pred),
                    "roc_auc_score":   roc_auc_score(y_te, y_proba),
                    "precision_score": precision_score(y_te, y_pred),
                    "recall_score":    recall_score(y_te, y_pred),
                }
                for k, v in eval_results[name].items():
                    mlflow.log_metric(k, v)

    print(f"  Model evaluation comparison:")
    all_keys = set()
    for v in eval_results.values():
        all_keys.update(v.keys())
    key_map = {
        "accuracy_score":  "accuracy_score",
        "f1_score":        "f1_score",
        "roc_auc_score":   "roc_auc_score",
    }
    display_keys = [k for k in key_map if k in all_keys]

    print(f"  {'Metric':<25}", end="")
    for name in ["champion", "challenger", "baseline"]:
        print(f" {name:>14}", end="")
    print()
    print(f"  {'─'*70}")

    for dk in display_keys:
        print(f"  {dk:<25}", end="")
        for name in ["champion", "challenger", "baseline"]:
            val = eval_results.get(name, {}).get(dk, float("nan"))
            print(f" {val:>14.4f}", end="")
        print()
    print()

    # Identify winner
    champ_acc = eval_results.get("champion",   {}).get("accuracy_score", 0)
    chall_acc = eval_results.get("challenger", {}).get("accuracy_score", 0)
    winner    = "challenger" if chall_acc > champ_acc else "champion"
    print(f"  Winner: {winner.upper()} "
          f"(accuracy: {max(champ_acc, chall_acc):.4f})")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Optuna + MLflow: HPO with full tracking
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Optuna + MLflow: HPO with nested runs")
print("━" * 65)
print()

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    import subprocess, sys
    try:
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "optuna", "--quiet"], check=True)
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        HAS_OPTUNA = True
    except Exception:
        pass

if HAS_SKLEARN and HAS_OPTUNA:
    mlflow.set_experiment("optuna-hpo-demo")

    def objective(trial):
        with mlflow.start_run(nested=True,
                               run_name=f"trial-{trial.number:03d}"):
            # Suggest hyperparameters
            n_est = trial.suggest_int("n_estimators", 10, 100)
            depth = trial.suggest_int("max_depth",    2, 8)
            feat  = trial.suggest_float("max_features", 0.3, 1.0)
            lr    = trial.suggest_float("lr_proxy", 1e-4, 1e-1, log=True)  # proxy

            mlflow.log_params({
                "n_estimators": n_est,
                "max_depth":    depth,
                "max_features": round(feat, 3),
            })

            # Train and evaluate
            model = RandomForestClassifier(
                n_estimators = n_est,
                max_depth    = depth,
                max_features = feat,
                random_state = trial.number,
                n_jobs       = 1,
            )
            model.fit(X_tr_df, y_tr)
            acc = model.score(X_te_df, y_te)
            f1  = f1_score(y_te, model.predict(X_te_df))

            mlflow.log_metrics({"val_accuracy": acc, "val_f1": f1})
            return acc

    N_TRIALS = 10
    with mlflow.start_run(run_name=f"optuna-sweep-{N_TRIALS}trials") as sweep:
        sweep_run_id = sweep.info.run_id
        mlflow.log_params({"n_trials": N_TRIALS, "direction": "maximize",
                           "sampler": "TPE"})

        study = optuna.create_study(
            direction   = "maximize",
            sampler     = optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        # Log best result on sweep run
        mlflow.log_params({f"best_{k}": v
                           for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_accuracy", study.best_value)

    print(f"  Optuna HPO: {N_TRIALS} trials completed")
    print(f"  Best value: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print()
    print(f"  Structure: 1 parent sweep run + {N_TRIALS} child trial runs")
    print(f"  All trials searchable via mlflow.search_runs()")
    print()

    # Verify parent-child structure
    parent_run     = client.get_run(sweep_run_id)
    child_runs_df  = mlflow.search_runs(
        experiment_names = ["optuna-hpo-demo"],
        filter_string    = (
            f"tags.mlflow.parentRunId = '{sweep_run_id}' "
            "AND attributes.status = 'FINISHED'"
        ),
        order_by = ["metrics.val_accuracy DESC"],
    )
    print(f"  Parent run:  {sweep_run_id[:16]}...")
    print(f"  Child runs:  {len(child_runs_df)} trials")
    if len(child_runs_df) > 0:
        top3 = child_runs_df[["metrics.val_accuracy",
                               "params.n_estimators",
                               "params.max_depth"]].head(3)
        print(f"  Top 3 trials:")
        print(f"  {'Rank':>5} {'Val Acc':>10} {'n_est':>8} {'depth':>8}")
        print(f"  {'─'*35}")
        for rank, (_, row) in enumerate(top3.iterrows(), 1):
            acc   = row.get("metrics.val_accuracy", float("nan"))
            n_est = row.get("params.n_estimators", "?")
            depth = row.get("params.max_depth", "?")
            print(f"  {rank:>5} {acc:>10.4f} {str(n_est):>8} {str(depth):>8}")

else:
    print(f"  sklearn or optuna not available.")
    print(f"  Pattern: wrap optuna objective with mlflow.start_run(nested=True)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Complete MLOps pipeline and production checklist
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Production MLOps pipeline summary")
print("━" * 65)
print()

PIPELINE_STEPS = [
    ("1. Experiment tracking",  "mlflow.start_run() + log_params/metrics/artefacts"),
    ("2. Autologging",          "mlflow.<framework>.autolog() — zero-code capture"),
    ("3. Model packaging",      "mlflow.<framework>.log_model(signature=...)"),
    ("4. Model evaluation",     "mlflow.evaluate(model, data, targets, model_type)"),
    ("5. Version registration", "mlflow.register_model() or registered_model_name=..."),
    ("6. Stage transition",     "client.transition_model_version_stage(...)"),
    ("7. Model serving",        "mlflow models serve -m models:/name/Production"),
    ("8. Batch inference",      "mlflow.pyfunc.load_model(...).predict(df)"),
    ("9. Production monitoring","log back prod metrics to run for drift detection"),
    ("10. Retraining trigger",  "CI/CD gate: promote_if_better()"),
]

print(f"  End-to-end MLOps pipeline:")
print(f"  {'Step':<28} {'MLflow API'}")
print(f"  {'─'*72}")
for step, api in PIPELINE_STEPS:
    print(f"  {step:<28} {api}")
print()

print(f"  MLflow serving command reference:")
print()
SERVE_CMDS = [
    ("Local REST server",  "mlflow models serve -m models:/fraud/Production -p 5001"),
    ("Docker container",   "mlflow models build-docker -m models:/fraud/Production -n fraud:v3"),
    ("Batch predict (CSV)","mlflow models predict -m models:/fraud/2 -i data.csv"),
    ("SageMaker deploy",   "mlflow deployments create -t sagemaker -m models:/fraud/3"),
]
for desc, cmd in SERVE_CMDS:
    print(f"  [{desc}]")
    print(f"    $ {cmd}")
    print()

print(f"  Production best practices:")
checklist = [
    ("Named experiments",          "Use project/component hierarchy, not Default"),
    ("Log params before metrics",  "Params persist even if training crashes"),
    ("Full training curves",       "log_metric(..., step=epoch) every epoch"),
    ("Input examples",             "input_example= for schema validation in UI"),
    ("Data lineage",               "mlflow.log_input() with dataset hash"),
    ("Git commit tag",             "mlflow.set_tag('git_commit', sha[:8])"),
    ("Register from run",          "Use registered_model_name= at log_model time"),
    ("Archive old Production",     "archive_existing_versions=True on promote"),
    ("Version descriptions",       "client.update_model_version(description=...)"),
    ("Stage-based loading",        "load_model('models:/name/Production')"),
]
for item, desc in checklist:
    print(f"    ✓ {item:<30} {desc}")

# Cleanup
shutil.rmtree(TRACKING_DIR, ignore_errors=True)
print()
print(f"  Tracking directory cleaned up ✅")
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