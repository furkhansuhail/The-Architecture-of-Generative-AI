"""
ML System Design, Monitoring & Distribution Shift
===================================================

Building a model that works in a notebook is the easy part.
Deploying it, keeping it working as the world changes, and knowing
when it has stopped working — that is where most production ML
projects succeed or fail. This module covers the full lifecycle:
from system design through deployment, monitoring, and drift handling.

"""

import textwrap
import re

TOPIC_NAME   = "ML System Design, Monitoring & Distribution Shift"
DISPLAY_NAME = "05 · ML System Design & Monitoring"
ICON         = "🏗️"
SUBTITLE     = "Design · Serving · Monitoring · Distribution Shift · Feedback Loops"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — ML SYSTEM DESIGN FUNDAMENTALS

### The ML System Design Interview Framework

Any ML system design question should be approached in a structured order.
Jumping straight to model architecture is the most common mistake.

    STEP 1 — CLARIFY THE PROBLEM (5 minutes):
        What is the business objective? (increase revenue, reduce churn)
        What are the constraints? (latency < 100ms, budget, data privacy)
        What does success look like, and how will it be measured?
        Who are the users and what is the traffic volume?

    STEP 2 — DEFINE THE ML TASK (5 minutes):
        Translate the business objective into an ML objective.
        Classification, regression, ranking, generation, clustering?
        What is the ground truth label, and when does it arrive?
        Online or batch inference? Real-time or pre-computed features?

    STEP 3 — DATA (10 minutes):
        What data is available? Where does it live?
        What is the label strategy? (human labelling, implicit feedback)
        Estimate the scale: rows, features, update frequency.
        What are the data quality risks?

    STEP 4 — MODEL AND FEATURES (10 minutes):
        Feature engineering: raw signal → numerical representation.
        Model choice and justification.
        Training pipeline: how often, on how much data?

    STEP 5 — SERVING AND INFRASTRUCTURE (5 minutes):
        How is the model called at inference time?
        Latency requirements and how to meet them.
        Fallback if the model fails?

    STEP 6 — MONITORING (5 minutes):
        What metrics indicate the model is working?
        What triggers a retrain or rollback?
        How is ground truth obtained post-deployment?


### Business Metric vs ML Metric

    The most important (and most often skipped) translation:

    ┌────────────────────────────────────────────────────────────────────┐
    │ Business objective         │ ML metric                            │
    ├────────────────────────────────────────────────────────────────────┤
    │ Reduce fraud losses        │ Recall at 1% FPR (catch fraud,       │
    │                            │ minimise false declines)              │
    ├────────────────────────────────────────────────────────────────────┤
    │ Increase click-through     │ Normalised DCG (ranking quality)      │
    ├────────────────────────────────────────────────────────────────────┤
    │ Reduce customer churn      │ AUC-PR (imbalanced binary classif.)   │
    ├────────────────────────────────────────────────────────────────────┤
    │ Better search results      │ MRR / NDCG@K                          │
    ├────────────────────────────────────────────────────────────────────┤
    │ Faster medical diagnosis   │ Sensitivity at 95% specificity        │
    └────────────────────────────────────────────────────────────────────┘

    The gap between the business metric and the ML metric is where
    most deployed models disappoint stakeholders. A model with 99% AUC
    may deliver no business value if its recall at the operating threshold
    is low, or if it creates false positives that damage customer trust.


### The Training-Serving Skew

    Training-serving skew is the most pervasive silent failure in ML systems.
    It occurs when the data seen during SERVING is not drawn from the same
    distribution as TRAINING data — specifically when the features computed
    at serving time differ from the features used during training.

    Common causes:
    1. FEATURE COMPUTATION INCONSISTENCY:
       Training: user_age = (date_of_training - birth_date)
       Serving:  user_age = stored integer from user profile (last updated 2019)
       The model was trained on computed ages; it serves on stale stored ages.

    2. TEMPORAL LEAKAGE IN TRAINING FEATURES:
       Training uses a "7-day average spend" feature computed from future data.
       Serving can only use a "7-day average spend" from the past 7 days.
       The feature semantics differ.

    3. MISSING FEATURES HANDLED DIFFERENTLY:
       Training: missing values imputed with training set median.
       Serving: missing values imputed with 0 (different code path).

    Diagnostic: log both training and serving feature distributions.
    Compare them regularly. Unexplained divergence = training-serving skew.
    Prevention: use the SAME feature computation code for training and serving
    (feature store with point-in-time correct lookups).


### Feature Stores

    A feature store is a centralised repository for ML features that
    ensures the same feature values are used consistently across:
        - Different models that use the same features
        - Training and serving (eliminates training-serving skew)
        - Different teams in the organisation

    Architecture:
    OFFLINE STORE (batch features):
        Historical feature values stored as tables (Hive, BigQuery, S3).
        Used for training: point-in-time correct feature lookup.
        "Give me user_x's average spend as of 2023-01-15" — no future leakage.

    ONLINE STORE (real-time features):
        Low-latency key-value store (Redis, DynamoDB).
        Used for serving: returns current feature values in milliseconds.
        Pre-computed and kept up-to-date by streaming pipelines.

    Point-in-time correctness:
        A critical property of offline feature stores.
        When assembling a training example for an event at time t,
        all feature values must be as they were at time t, not later.
        Failing this is temporal leakage → inflated training metrics.

    When to use a feature store:
        Multiple models sharing the same features (amortises compute)
        Serving latency requirements that rule out real-time feature computation
        Regulatory requirements to audit what features were used per prediction
        Team larger than ~5 ML engineers (coordination overhead)

    When a feature store is overhead:
        Single model, small team, features computed from request-time data only.
        Starting a project — add the feature store when you have the complexity.


### Online vs Offline Serving

    BATCH (OFFLINE) INFERENCE:
        Pre-compute predictions for all entities and store in a database.
        The serving layer is just a database lookup — extremely fast.
        Latency: microseconds to milliseconds.
        Freshness: predictions are stale (as old as the last batch run).
        Use when: predictions can be computed ahead of time (daily churn,
                  content recommendations pre-computed nightly).

    REAL-TIME (ONLINE) INFERENCE:
        The model is called live at request time.
        Latency depends on model complexity: 10ms (linear), 50ms (forest),
        100-500ms (small NN), seconds (large LLM).
        Freshness: uses the most current features.
        Use when: the request context cannot be known in advance (search,
                  fraud detection, ad ranking).

    Hybrid pattern (common in practice):
        Pre-compute expensive features offline (user embeddings, item stats).
        At serving time: lookup pre-computed features + compute cheap
        request-specific features → run fast model.


##### PART 2 — DESIGNING THE DATA PIPELINE

### Label Strategy

    Labels are often the most expensive and fragile part of the system.

    EXPLICIT LABELS (human annotation):
        Pros: high quality, unambiguous, can define any task.
        Cons: expensive (crowd-sourcing), slow (days to weeks lag),
              inter-annotator disagreement for subjective tasks.
        Mitigation: annotation guidelines, quality control with
        overlapping annotators, calibration tests.

    IMPLICIT LABELS (behavioural signals):
        Click, dwell time, purchase, like, share — cheap and plentiful.
        Cons: NOISY PROXIES. A click does not equal "user found this valuable."
              Exposure bias: items not shown can never be clicked.
              Position bias: items ranked first are clicked more regardless of quality.
        Debiasing techniques: inverse propensity scoring (IPS).

    PROGRAMMATIC LABELS (weak supervision, Snorkel):
        Write labelling functions (heuristics, regular expressions, knowledge bases).
        Combine multiple noisy labellers with a generative model.
        Cons: weaker than human labels; requires validation.

    SELF-SUPERVISED LABELS (from data structure):
        Masked language modelling: hide a word, predict it.
        Next sentence prediction, contrastive learning, rotation prediction.
        Free at massive scale — the foundation of modern LLMs.


### Stream vs Batch Ingestion

    BATCH INGESTION (Spark, Hadoop):
        Process data in large chunks at scheduled intervals (hourly, daily).
        High throughput, low complexity, suitable for slowly changing data.
        Latency: hours to days.
        When: training data pipelines, analytical feature computation.

    STREAM INGESTION (Kafka + Flink/Spark Streaming):
        Process events in real-time as they arrive (milliseconds to seconds).
        Higher complexity, requires exactly-once or at-least-once guarantees.
        When: fraud detection features, live recommendation signals,
              session-level features that must respond to recent actions.

    Lambda Architecture (hybrid):
        Batch layer: processes all historical data for accuracy.
        Speed layer: processes recent streaming data for freshness.
        Serving layer: merges both views.
        Drawback: maintaining two code paths (batch and streaming)
        for the same feature computation is complex and error-prone.

    Kappa Architecture (streaming only):
        A single streaming pipeline handles both historical and real-time.
        Reprocess historical data by replaying from a durable event log (Kafka).
        Preferred for new systems — one code path, no synchronisation issues.


### Data Versioning and the Lakehouse

    For reproducibility, training data must be versioned alongside code.

    DVC (Data Version Control):
        Git-like version control for datasets and model artefacts.
        Stores large files in cloud storage (S3, GCS), tracks metadata in Git.
        Enables: checkout the exact dataset that produced a specific model.

    Delta Lake / Apache Iceberg (table-level versioning):
        ACID transactions on data lakes (append, update, delete, merge).
        Time-travel: query the data as it was at any past timestamp.
        Z-ordering: cluster data by frequently queried columns for fast scans.
        Schema enforcement: prevents schema drift from corrupting pipelines.

    Lakehouse architecture:
        Combines the scalability of a data lake (cheap object storage)
        with the reliability of a data warehouse (ACID, schema enforcement).
        Training pipeline reads a SPECIFIC VERSION of a Delta table.
        Ensures that rerunning a training job produces the same result.


### Handling Delayed Labels

    In many production systems, the ground truth label arrives AFTER
    the prediction was made. This creates several challenges:

    FRAUD DETECTION:
        A transaction is predicted as fraudulent at t=0.
        The actual fraud outcome is confirmed at t = 3-90 days.
        Naive training: use all confirmed labels → training data is stale.

    CHURN PREDICTION:
        Predict churn at t=0. Ground truth at t = 30-90 days.
        How do you know a user has not churned yet vs not churned ever?

    Solutions:
    1. FIXED OBSERVATION WINDOW: only use labels confirmed within X days.
       Pros: simple. Cons: wastes recent data; biased if delay varies.

    2. CENSORING-AWARE TRAINING: treat unresolved examples as censored.
       Use survival analysis models (Kaplan-Meier, Cox) that model
       time-to-event rather than binary outcomes.

    3. LABEL UPDATES: start with a default negative label, update when
       the true label arrives. Track which model version saw which label.
       Requires careful version bookkeeping in the label store.


### The Data Flywheel

    A data flywheel is a virtuous cycle where deploying a model generates
    data that improves the model, which improves the product, which
    attracts more users, which generates more data.

    Diagram 1 — The Data Flywheel:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Better model  ──────►  Better product  ──────►  More users   │
    │       ▲                                              │          │
    │       │                                              ▼          │
    │  More data  ◄────────────────────────────────  More interactions│
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Examples:
    Google Search: more queries → better click feedback → better ranker.
    TikTok: more views → richer preference signals → better recommendations.
    Gmail spam filter: more labelled spam → fewer spam → more trust → more mail.

    Risks:
    Feedback loops (covered in Part 8): the model affects the data it trains on.
    Cold-start problem: the flywheel does not spin until there is enough data.
    Echo chambers: the model increasingly serves what it already knows.


##### PART 3 — MODEL TRAINING AT SCALE

### Experiment Tracking

    Every training run should log: hyperparameters, dataset version,
    code version, metrics, artefacts, and environment.

    MLflow:
        Open-source. Runs on any infrastructure.
        Tracking, Model Registry, Projects, Deployments.
        MLflow UI: compare runs by metric across hyperparameter sweeps.
        Model Registry: stages (Staging → Production → Archived).

    Weights & Biases (WandB):
        Hosted or self-hosted. Excellent visualisations.
        System metrics (GPU utilisation, memory) alongside training metrics.
        Sweeps: hyperparameter search integrated with logging.
        Artefacts: versioned datasets and models linked to runs.

    What to log:
        CONFIG: all hyperparameters, model architecture, dataset paths.
        METRICS: loss, task metrics per epoch, at fixed step intervals.
        ARTEFACTS: model checkpoints, confusion matrices, SHAP plots.
        SYSTEM: GPU utilisation, memory, training throughput.
        CODE: commit hash, diff, environment (requirements.txt).

    The goal: any result should be exactly reproducible from its logged config.


### Training Pipelines: Orchestration

    Training pipelines automate the sequence: data prep → feature eng →
    training → evaluation → registration → deployment.

    Apache Airflow:
        DAG-based workflow scheduler. Runs Python tasks on a schedule.
        Mature, widely deployed. Good for ETL and periodic training jobs.
        Weakness: no built-in ML concepts (experiments, models, artefacts).

    Kubeflow Pipelines:
        Kubernetes-native ML pipeline. Docker containers per step.
        Native integration with experiment tracking, model serving.
        Strength: reproducibility (each step is a versioned container).
        Weakness: high operational overhead (requires Kubernetes expertise).

    Metaflow (Netflix):
        Python-first. Each step is a Python class method decorated with @step.
        Seamlessly runs locally or on AWS/GCP/Azure.
        Versioning built-in: every run is a first-class object with full lineage.


### Model Registry and Promotion Workflow

    The model registry tracks all trained models and their metadata,
    enabling controlled progression from training to production.

    Diagram 2 — Model Promotion Workflow:

    ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │  Training  │───►│  Staging   │───►│  Shadow    │───►│ Production │
    │  Run (any) │    │ (validated)│    │ (A/B test) │    │  (live)    │
    └────────────┘    └────────────┘    └────────────┘    └────────────┘
          │                 ▲                  ▲                  │
          │                 │ Passes offline   │ Passes online    │
          │                 │ evaluation       │ evaluation       ▼
          └─────────────────┴──────────────────┴──────────  Rollback if
                                                             metrics drop

    STAGING: automated offline evaluation against a held-out test set.
             Comparison against the current production model.
             Must beat production by a minimum margin to proceed.

    SHADOW: runs alongside production but does NOT affect users.
            Logs predictions for analysis without real consequences.
            Used to measure online latency, memory, CPU before exposure.

    A/B: serves a fraction of real traffic (e.g., 5%).
         Measures business metrics (CTR, conversion) on real users.
         Statistical significance determines promotion or rollback.

    ROLLBACK: if production metrics drop below a threshold, the registry
              supports one-command rollback to the previous model version.


### Reproducibility

    The three things that must be identical to reproduce a result:
    1. CODE: exact commit hash (not just branch name).
    2. DATA: exact dataset version (DVC hash, Delta table version, S3 URI).
    3. ENVIRONMENT: exact library versions (requirements.txt, conda env, Docker).

    Additionally for neural networks:
    4. RANDOM SEEDS: torch.manual_seed, numpy random seed,
       Python random seed, CUDA determinism flags.

    Note: even with all four identical, floating-point non-determinism
    in GPU operations (cuDNN convolutions, reductions) means results may
    differ by small amounts across runs. "Reproducibility" in practice
    means statistical consistency, not bit-for-bit identical results.


##### PART 4 — SERVING INFRASTRUCTURE

### Model Serving Patterns

    REST API (JSON over HTTP):
        Universal, easy to test with curl/Postman, works through firewalls.
        Overhead: JSON serialisation/deserialisation adds ~1-5ms per call.
        Frameworks: FastAPI, Flask, BentoML, TorchServe, Triton (REST mode).
        Use when: general-purpose APIs, browser clients, cross-language calls.

    gRPC (Protocol Buffers):
        Binary protocol, 5-10× less serialisation overhead than REST.
        Strongly typed via .proto schemas → client code auto-generated.
        Supports streaming (server sends multiple responses to one request).
        Use when: high-throughput internal service-to-service calls,
                  streaming predictions, latency-critical paths.

    Streaming Inference:
        Model output is streamed token-by-token (LLMs) or frame-by-frame.
        The client receives partial results before the full output is ready.
        Critical for LLM UX: users see text appearing rather than waiting.
        Implementation: server-sent events (SSE) or gRPC bidirectional streaming.


### Dynamic Batching: Throughput vs Latency

    Individual inference requests arrive sporadically.
    GPUs are most efficient with large batches (high parallelism).
    Dynamic batching groups multiple arriving requests into one GPU batch.

    STATIC BATCHING: wait for exactly B requests, then run.
        Guarantees high GPU utilisation.
        Problem: a request arriving when the batch is almost full may wait
        longer than its SLA allows.

    DYNAMIC BATCHING (NVIDIA Triton, TorchServe):
        Specify a max batch size and a max wait time.
        When either condition is met (batch full OR timeout), dispatch.
        The server handles batching transparently to clients.

    CONTINUOUS BATCHING (for LLMs):
        Different from dynamic batching for fixed-size models.
        In LLM generation, different requests complete at different times
        (some prompts generate 10 tokens, others 1000).
        Continuous batching immediately replaces a completed request in the
        batch with a new one, keeping GPU occupancy high at all times.
        Used by vLLM, TGI (HuggingFace Text Generation Inference).


### Model Optimisation for Serving

    Trained models are often not in their most efficient form for inference.

    TORCHSCRIPT:
        Converts PyTorch models to a static graph (no Python overhead).
        Two modes: tracing (records operations on example inputs) and
        scripting (compiles Python code to static graph).
        2-3× inference speedup on CPU; modest gain on GPU.

    ONNX (Open Neural Network Exchange):
        Open format for interoperable model exchange.
        Export from PyTorch/TensorFlow → import to TensorRT, ONNX Runtime.
        ONNX Runtime: optimised execution across hardware (CPU, GPU, ARM).

    TENSORRT (NVIDIA):
        Hardware-specific optimisation for NVIDIA GPUs.
        Optimisations: layer fusion, kernel auto-tuning, precision reduction.
        Can achieve 2-8× speedup over PyTorch on the same hardware.
        Limitations: only for NVIDIA GPUs; compilation is slow (minutes to hours);
        must recompile when input shapes change.

    KV CACHE (for LLMs):
        In transformer inference, the key-value projections from previous
        tokens are reused rather than recomputed.
        Without KV cache: O(n²) computation per generated token.
        With KV cache: O(n) computation per token; memory O(n) per token.
        Critical for any practical LLM deployment.


### SLAs and Latency Targets

    SLA = Service Level Agreement: contractual commitments on availability,
    latency, and throughput.

    KEY METRICS:
    P50 (median): half of requests are faster.
    P99: 99% of requests are faster. The "tail latency."
    P99.9: 1 in 1000 requests — a single slow model call is acceptable;
           1 in 1000 means something in the infrastructure is very wrong.

    Why P99 matters more than P50:
        A user who hits a slow request has a worse experience than
        your P50 suggests. Google found that a 400ms slowdown on search
        caused a 0.6% drop in search volume — from users who encountered
        tail latency, not median latency.

    Latency budget decomposition:
        Total latency = feature retrieval + inference + postprocessing
        Typical fraud detection system:
            Feature lookup (Redis):     1-5ms
            Feature computation:        1-3ms
            Model inference (XGBoost): 5-20ms
            Network overhead:          1-5ms
            Total:                     8-33ms  (SLA: < 100ms)


### A/B Testing vs Multi-Armed Bandit

    A/B TESTING:
        Split traffic: 50% sees model A, 50% sees model B.
        Run for a fixed duration; analyse results; promote the winner.
        Pros: clean experimental design, well-understood statistics.
        Cons: during the experiment, 50% of traffic uses the inferior model.
              Opportunity cost if model B is clearly winning early.

    MULTI-ARMED BANDIT (Exploration-Exploitation):
        Dynamically adjust traffic allocation based on observed performance.
        Thompson Sampling: assign traffic proportional to the probability
        that each model is the best (Bayesian posterior over performance).
        Models that perform better get more traffic automatically.
        Pros: reduces opportunity cost compared to fixed A/B split.
        Cons: harder to interpret; statistical analysis is more complex;
              can be fooled by non-stationarity (a model might look good
              early but plateau or degrade).

    When to use which:
        A/B: important decisions, regulatory environments, when you need
             clean statistical evidence for stakeholders.
        Bandit: recommendation systems, content optimisation, anywhere
                the cost of exposing users to the inferior option is high
                and you want the system to self-optimise.


##### PART 5 — MONITORING IN PRODUCTION

### The Four Types of Production Failures

    Understanding failure mode determines what you monitor.

    TYPE 1 — OPERATIONAL FAILURE:
        The model is not reachable or crashes.
        Symptoms: HTTP 5xx errors, timeouts, null predictions.
        Monitoring: service uptime, error rate, latency P99.
        Cause: infrastructure (OOM, deployment bug, dependency outage).

    TYPE 2 — DATA PIPELINE FAILURE:
        The model receives malformed or stale features.
        Symptoms: predictions seem frozen, performance drops suddenly.
        Monitoring: feature schema validation, null rates, value range checks.
        Cause: upstream data source changed, ETL job failed, schema migration.

    TYPE 3 — MODEL PERFORMANCE DEGRADATION:
        The model is operational but making worse predictions.
        Symptoms: business metrics declining; model metrics degrading.
        Monitoring: accuracy, F1, AUC on labelled recent data.
        Cause: distribution shift, concept drift, model staleness.

    TYPE 4 — SILENT FAILURES:
        The model is operational, predictions look normal in distribution,
        but quality has degraded in ways that do not immediately show up.
        Symptoms: none apparent until business metrics decline weeks later.
        Monitoring: output distribution, prediction confidence, cohort analysis.
        This is the hardest failure type to catch.


### Operational Monitoring Layer

    Every ML serving system should have standard software monitoring:

    AVAILABILITY: percentage of time the service is responsive.
        Target: 99.9% (8.7 hours downtime/year) or 99.99% (52 min/year).

    LATENCY: P50, P95, P99 response times.
        Alert: P99 > 2× normal; any P99.9 spike.

    THROUGHPUT: requests per second (RPS). Both current value and trend.
        Alert: RPS drops to near zero (service may be unreachable).
        Alert: RPS unexpectedly spikes (potential abuse or test leak).

    ERROR RATE: fraction of requests returning errors.
        Alert: error rate > 0.1% sustained for >5 minutes.

    SATURATION: CPU, GPU, memory utilisation of serving instances.
        Alert: >80% sustained → scale out or the service will soon degrade.

    Tools: Prometheus + Grafana, Datadog, CloudWatch, New Relic.


### The Ground Truth Problem

    The fundamental challenge in ML monitoring: to measure whether the
    model is correct, you need the true label. In production, the true
    label often arrives late, never, or in a biased way.

    IMMEDIATE GROUND TRUTH (rare and easy):
        Classification of an email the user reads now.
        The outcome is observable immediately after the prediction.

    DELAYED GROUND TRUTH (common):
        Loan default (weeks to months later).
        Churn (30-90 days later).
        Fraud outcome (days to weeks later).
        Solution: monitor PROXY METRICS (input distribution, prediction
        distribution) while awaiting ground truth.

    NEVER-OBSERVED GROUND TRUTH (hard):
        Medical diagnostic model: no surgery means no histology.
        Content moderation: items removed by moderators are biased
        sample (only the flagged content gets reviewed).
        Solution: active labelling, human-in-the-loop spot checking.

    OBSERVABILITY BIAS:
        Only predictions above a threshold are acted upon; only those
        actions generate labels.
        A fraud model that blocks transactions → we never know the
        outcome for blocked transactions (maybe they would have been fine).
        This creates SELECTION BIAS in the label distribution.


### Input Data Monitoring

    Monitor the FEATURES that enter the model, not just the outputs.
    Data problems upstream cause model failures downstream.

    SCHEMA VALIDATION:
        Every feature has an expected type, range, and cardinality.
        Alert if a feature changes type (string vs int), exceeds range,
        or has unexpected cardinality (new category values).
        Tool: Great Expectations, TFX Data Validation, Evidently AI.

    NULL/MISSING RATE MONITORING:
        Track the fraction of null values per feature over time.
        A feature that suddenly has 20% nulls (up from 1%) indicates
        a pipeline failure upstream.

    DISTRIBUTION MONITORING:
        Track summary statistics (mean, std, percentiles) over time.
        Plot the distribution daily/weekly using histograms.
        Sudden distribution shifts indicate data source changes or bugs.


### Output Distribution Monitoring

    Even without ground truth, the MODEL'S OUTPUT DISTRIBUTION contains
    signal about whether the model is healthy.

    PREDICTION DISTRIBUTION:
        The distribution of the model's raw scores or predicted probabilities.
        A well-calibrated model should have a stable distribution over time
        (assuming the input distribution is stable).
        Sudden shift: model may have been retrained, input features changed,
        or the population being scored has shifted.

    PREDICTION CONFIDENCE:
        Track the average confidence of predictions.
        Sudden drop in confidence: model is encountering unfamiliar inputs.
        Sudden spike in confidence: possible data leakage or degenerate input.

    DOWNSTREAM ACTION RATE:
        If the model drives a binary action (block/allow, show/hide),
        track the rate of each action.
        An unexpected jump in block rate → either fraud is actually up
        (check other signals) or the model is misfiring (check data).


### Statistical Drift Tests

    Formal statistical tests quantify whether two distributions are the same.
    Apply to: features, model outputs, error rates across cohorts.

    KOLMOGOROV-SMIRNOV (KS) TEST (for continuous variables):
        Tests whether two samples come from the same continuous distribution.
        KS statistic D = max|F₁(x) - F₂(x)| (max difference of CDFs).
        p-value < 0.05 → distributions are significantly different.
        Sensitive to both location and shape differences.
        Limitation: sensitive to sample size — with enough data, even
        trivial differences are "significant."

    POPULATION STABILITY INDEX (PSI) (for continuous and categorical):
        PSI = Σᵢ (Actual% - Expected%) × ln(Actual% / Expected%)
        Ranges:
            PSI < 0.10: no significant shift
            PSI 0.10-0.25: moderate shift; investigate
            PSI > 0.25: major shift; model likely needs retraining
        Widely used in financial services and credit risk.

    CHI-SQUARED TEST (for categorical variables):
        Tests whether observed category frequencies match expected.
        Use for monitoring categorical feature distributions.

    MAXIMUM MEAN DISCREPANCY (MMD) (multivariate):
        Measures discrepancy between two distributions in a kernel-induced
        feature space. Captures multivariate drift that univariate tests miss.
        Computationally more expensive but more sensitive.

    Diagram 3 — PSI Interpretation:

    Production   │ ██████  ← expected (training)
    distribution │ █████   ← actual (current)
                 │ ████
                 │ ███             PSI = 0.08  → No shift
    Bin:           1  2  3  4  5  6

    Production   │ ██████
    distribution │    ████
                 │       ███
                 │          ██    PSI = 0.35  → Major shift!
    Bin:           1  2  3  4  5  6


### Alert Design

    Alert when:  something meaningful has changed and action is required.
    Alert fatigue: too many alerts → oncall ignores them → real issues missed.

    THRESHOLD ALERTS: trigger when a metric crosses a fixed threshold.
        Simple, easy to understand.
        Problem: what is the right threshold? May trigger on seasonal patterns.

    STATISTICAL CONTROL CHARTS (Western Electric rules):
        Track rolling mean and standard deviation.
        Alert when: value > μ + 3σ (99.7% rule), or any 2 of last 3 values
        are > μ + 2σ, or any 5 of last 7 are on the same side of μ.
        Pros: adapts to normal variation; catches sustained shifts.
        Cons: requires historical data to establish μ and σ.

    ANOMALY DETECTION ALERTS:
        Train an unsupervised model (Isolation Forest, LSTM) on the
        metric time series. Alert on out-of-distribution values.
        Better for complex multivariate drift.

    Alert routing:
        P0 (immediate): service is down, error rate > 5% → page oncall.
        P1 (urgent): latency P99 > 2×, feature null rate spike → Slack alert.
        P2 (informational): PSI > 0.1 on a non-critical feature → email digest.


##### PART 6 — DISTRIBUTION SHIFT

### What Is Distribution Shift?

    A model trained on data from distribution P_train is deployed in an
    environment where the data comes from distribution P_serve.

    Distribution shift: P_train ≠ P_serve

    Why it matters: all guarantees about model performance (accuracy, AUC,
    calibration) are conditional on P_train. When the serving distribution
    differs, those guarantees are void.

    Distribution shift is not a rare failure mode — it is the default
    state of any deployed model over time. The question is not whether
    shift will happen but when and how severely.

    Decomposition of the joint distribution:
        P(X, Y) = P(X) × P(Y|X) = P(Y) × P(X|Y)

    Any shift in the joint distribution can be decomposed into shifts in
    these factors. Different components change at different rates for
    different reasons. This decomposition is the key to diagnosing shift.


### Covariate Shift: P(X) Changes, P(Y|X) Stays the Same

    The INPUTS change but the underlying relationship between inputs
    and outputs remains the same.

    Examples:
    - A spam classifier trained on 2020 emails deployed in 2024.
      The writing style of emails has changed (shorter, more emojis).
      But spam is still identifiable by the same features — there are
      just different proportions of each type of email.

    - A credit model trained on data from the US deployed in Germany.
      The distribution of income, age, employment differs (P(X) changes).
      But the relationship between creditworthiness and these features
      (P(Y|X)) is broadly similar.

    - A computer vision model trained on daytime images deployed at night.
      The pixel distribution changes dramatically. The objects themselves
      have the same appearance structure — just different lighting.

    Effect on model performance:
        The model may have poor coverage in regions of the new X distribution
        where training data was sparse. It extrapolates rather than interpolates.
        Calibration degrades even if the DECISION BOUNDARY is still correct.

    Detection: compare the marginal feature distributions P_train(X) vs P_serve(X)
    using KS test, PSI, or classifier two-sample test.

    Fix: importance weighting (see Part 7).


### Label Shift (Prior Shift): P(Y) Changes, P(X|Y) Stays the Same

    The prevalence of each class changes, but the features of each class
    look the same.

    Examples:
    - A pandemic flu predictor trained on pre-COVID data.
      During COVID, many more people have flu-like symptoms (P(Y) changes).
      But the feature profile of flu vs non-flu is the same.

    - Fraud prevalence increases during a promotional event.
      Each fraudulent transaction looks the same as before — there are
      just more of them relative to legitimate transactions.

    Effect: a model trained at 1% fraud rate will be under-calibrated
    if fraud rate rises to 5%. The model's decision threshold (0.5)
    is no longer appropriate. Recall drops, false negative rate rises.

    Detection: monitor the model's PREDICTION DISTRIBUTION.
    If more predictions are near 0 or 1, or the proportion of high-risk
    scores changes, label shift may be occurring. The Bayesian estimator
    method can estimate P_serve(Y) from unlabelled data.

    Fix: recalibrate the decision threshold; update prior probabilities.


### Concept Drift: P(Y|X) Changes

    The most dangerous form of shift: the underlying relationship
    between inputs and outputs has changed.

    Examples:
    - A credit model trained before a recession.
      After the recession, the same income and credit score that
      indicated a reliable borrower now indicates higher default risk.
      The feature-to-outcome mapping has changed.

    - A medical diagnostic model trained on pre-treatment images.
      A new treatment changes how a disease presents in imaging.
      The same visual features no longer mean the same diagnosis.

    - A sentiment classifier trained on 2020 social media.
      Slang and expressions shift — words that were positive in 2020
      have ironic or negative connotations by 2024.

    Effect on model: the model's learned function f(X) is now a mapping
    from features to WRONG outputs. No amount of recalibration or
    threshold adjustment can fix this — the model must be retrained.

    Detection: requires observing ground truth (or proxy ground truth) to
    detect that P(Y|X) has changed. Unlike covariate shift, P(X) can be
    stable while concept drift is happening. This is why concept drift
    is harder to detect than covariate shift.


### Temporal Drift Patterns

    Drift does not always happen the same way. Understanding the pattern
    determines the retraining strategy.

    GRADUAL DRIFT:
        The distribution shifts slowly and continuously over time.
        Example: language evolving, customer preferences slowly changing.
        Detection: trends in performance metrics, slow PSI increase.
        Strategy: continuous training on a rolling window of recent data.

    SUDDEN DRIFT (Concept Shift):
        An abrupt change due to an external event.
        Example: COVID-19 changing travel patterns instantly.
                 A competitor launching a product that changes user behaviour.
                 A regulatory change affecting what transactions are legal.
        Detection: sudden drop in performance metrics, sudden PSI spike.
        Strategy: immediate retrain on post-event data; may need to
                  treat pre-event data as out-of-domain.

    RECURRING DRIFT (Seasonal):
        Patterns that repeat periodically.
        Example: holiday shopping patterns, seasonal disease patterns,
                 weekly website traffic patterns.
        Detection: decompose the metric time series for seasonality.
        Strategy: train season-specific models, or include time features.

    INCREMENTAL DRIFT:
        Slow drift with occasional sudden jumps.
        Most real-world systems exhibit this pattern.
        Strategy: rolling window training + drift detection triggers.


### How Different Model Types Respond to Shift

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Model Type      │ Covariate Shift     │ Label Shift    │ Concept Drift│
    ├──────────────────────────────────────────────────────────────────────┤
    │ Linear / Logistic│ Moderate degr.     │ Threshold adj. │ Full retrain │
    │ regression       │ (extrapolates OK)  │ sufficient     │ needed       │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Decision Tree /  │ Significant degr.  │ Threshold adj. │ Full retrain │
    │ Random Forest    │ (can't extrapolate)│                │ needed       │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Gradient Boosting│ Significant degr.  │ Threshold adj. │ Full retrain │
    │ (XGBoost, etc.)  │                    │                │ needed       │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Neural Network   │ Moderate — better  │ Fine-tune head │ Fine-tune or │
    │                  │ generalisation via │ sufficient     │ full retrain │
    │                  │ representation     │                │              │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Large Pretrained │ Most robust —      │ Fine-tune head │ Fine-tune on │
    │ Model (GPT, ViT) │ generalises well   │ sufficient     │ new data     │
    └──────────────────────────────────────────────────────────────────────┘


##### PART 7 — HANDLING DISTRIBUTION SHIFT

### Importance Weighting for Covariate Shift

    Covariate shift is correctable in principle by reweighting training examples.

    Key insight: an example with features X that are COMMON in P_serve
    but RARE in P_train should be weighted more during training.
    This makes the effective training distribution match P_serve.

    Importance weights:
        w(x) = P_serve(X=x) / P_train(X=x)

    The ideal training objective:
        L_weighted = Σᵢ w(xᵢ) × L(f(xᵢ), yᵢ)

    In practice, P_serve is not known explicitly. Estimate it:
    CLASSIFIER TWO-SAMPLE TEST:
        Train a binary classifier to distinguish training vs serving samples.
        If it achieves AUC close to 0.5: no detectable shift.
        If AUC > 0.7: significant shift.
        The classifier's predicted probability p̂(is_serving | x) gives:
            w(x) ≈ p̂(is_serving | x) / p̂(is_training | x)
        These are the importance weights.

    Limitations:
        High weights for rare serving samples can cause gradient instability.
        Cap weights at a maximum (e.g., 10× or 95th percentile).
        Weights are only valid for the covariate shift assumption
        (P(Y|X) must truly be stable).


### Continuous Training: Retraining Pipelines

    The most common strategy: retrain the model on a rolling window
    of recent data at a scheduled frequency.

    SCHEDULE-BASED RETRAINING:
        Retrain every day/week/month regardless of detected drift.
        Pros: simple, predictable, no drift detector needed.
        Cons: wastes compute when nothing has changed; may be too slow
              if sudden drift occurs between retraining intervals.

    DATA FRESHNESS WINDOW:
        How much historical data to include in the training window?
        All data: stable model, good on long-term patterns, stale.
        Recent data only: adapts fast, but may lose rare events.
        Weighted window: recent data weighted more (exponential decay).

    FINE-TUNING vs FULL RETRAINING:
        For neural networks, fine-tuning on recent data with a small
        learning rate is faster than retraining from scratch.
        Risk: catastrophic forgetting (old patterns are forgotten).
        Mitigation: replay a fraction of old data alongside new data.

    TRIGGERING RETRAINING:
        Cron schedule: simplest.
        Data volume trigger: retrain when N new labelled examples arrive.
        Performance trigger: retrain when a monitored metric drops below threshold.
        Drift trigger: retrain when PSI > 0.25 or KS p-value < 0.01.


### Online Learning: Incremental Updates

    Instead of periodic retraining, update model weights continuously
    as each new example arrives.

    Suitable models:
        Linear/logistic regression: SGD updates per example.
        Neural networks: mini-batch streaming SGD.
        Gradient boosted trees: not easily done online (tree structure fixed).

    Challenges:
    1. CATASTROPHIC FORGETTING: recent data completely overwrites old patterns.
    2. CONCEPT DRIFT ADAPTATION vs NOISE: small fluctuations are noise,
       large shifts are real drift. Hard to distinguish without history.
    3. DELAYED LABELS: online learning needs immediate labels, which are
       often not available.
    4. DEBUGGING: online models are hard to introspect — their weights
       change continuously, making reproducibility very difficult.

    In practice: online learning is used for recommendation systems and
    ad ranking (where labels are nearly immediate — a click is observed
    within seconds) but rarely for complex models where labels are delayed.


### Catastrophic Forgetting in Continual Learning

    When a neural network is trained on new data, gradient descent
    updates weights to minimise loss on NEW examples. These updates
    can overwrite the weights that encoded knowledge of OLD examples.

    Diagram 4 — Catastrophic Forgetting:

    Weights after Task 1 ─────► Fine-tune on Task 2 ─────► Evaluate on Task 1
    (high accuracy on T1)        (weights drift away)        (accuracy collapses)

    Mitigation strategies:
    ELASTIC WEIGHT CONSOLIDATION (EWC):
        Add a penalty for changing weights that were important for Task 1.
        Importance estimated by the Fisher Information Matrix (diagonal approx).
        L_EWC = L_new + λ × Σᵢ Fᵢ × (θᵢ - θ₁ᵢ)²
        where Fᵢ = Fisher information of weight i from Task 1.

    EXPERIENCE REPLAY:
        Store a small buffer of examples from old tasks.
        Mix old and new examples in each training batch.
        Simple and effective. Buffer size is the main hyperparameter.

    PROGRESSIVE NEURAL NETWORKS:
        Add new network columns for new tasks; freeze old columns.
        Old columns can still be accessed via lateral connections.
        Memory grows linearly with tasks — not scalable indefinitely.

    REGULARISATION DECAY:
        When fine-tuning, use a high learning rate for the task-specific
        head and a very small learning rate for the shared backbone.
        Slows forgetting without explicit EWC terms.


### Root Cause Analysis of Distribution Shift

    When drift is detected, the question is WHY — to choose the correct fix.

    DIAGNOSTIC FRAMEWORK:
    1. Is it operational (pipeline failure) or distributional?
       Check feature null rates, range violations → if yes, fix the pipeline.

    2. Which features are drifting?
       Compute PSI for every feature. Rank by PSI descending.
       The top-drifting features often explain the model performance drop.

    3. Is it covariate shift (P(X) changes) or concept drift (P(Y|X) changes)?
       Train a classifier to distinguish old vs new feature distributions.
       If it achieves high AUC: P(X) has changed (covariate shift).
       If the classifier is near-chance: P(X) is stable, suspect concept drift.

    4. What real-world event could explain the shift?
       Timeline: when did the shift start?
       Cross-reference with: product changes, policy changes, external events,
       competitor actions, seasonality.

    5. Is it permanent or temporary?
       If the cause is a one-time event: retrain on post-event data.
       If the cause is ongoing (new user population): continual training.
       If seasonal: add time features or train seasonal models.


##### PART 8 — FEEDBACK LOOPS AND DATA QUALITY IN PRODUCTION

### Closed-Loop vs Open-Loop Systems

    OPEN LOOP: the model makes predictions, but those predictions do not
    affect the data that will be used to train future versions of the model.
    Example: a model that classifies images uploaded to a platform.
    The prediction does not change which images people upload.

    CLOSED LOOP: the model makes predictions that influence user behaviour,
    which generates the data used to train the next model version.
    Example: a recommendation system. Recommendations change what users
    see and click on. Future training data is shaped by current recommendations.

    In closed-loop systems, standard statistical assumptions break down.
    The training data is NOT an i.i.d. sample from the true population —
    it is a BIASED sample filtered through the current model's behaviour.


### Exposure Bias and Feedback Loop Amplification

    EXPOSURE BIAS:
        A recommendation model can only observe interactions with items
        it CHOSE TO SHOW. Items never shown cannot be clicked, liked, or
        rated. Future training data has zero examples for these items.
        The model becomes increasingly confident that unshown items are
        not interesting — reinforcing the decision to not show them.

    POPULARITY BIAS AMPLIFICATION:
        A content ranking system shows popular items more → they get more
        clicks → they look more popular → they get shown even more.
        Long-tail items (niche but relevant) are progressively buried.
        The model is optimising for popularity, not relevance.

    FILTER BUBBLE:
        A personalisation model shows content matching the user's existing
        preferences → user only sees confirming information → model learns
        these are the only relevant topics for this user → more filtering.
        Each iteration of the flywheel narrows the content universe.

    THE SELF-FULFILLING PROPHECY:
        A fraud detection model flags account X as high risk.
        The bank places restrictions on account X.
        The user of account X changes behaviour (more cash, less card).
        Future model sees restricted account with unusual behaviour → more risky.
        The model's prediction helped create the evidence supporting it.


### Counterfactual Learning and Offline Policy Evaluation

    To evaluate what would have happened under a DIFFERENT model (the
    counterfactual), without running a live A/B test, we use offline
    policy evaluation methods.

    INVERSE PROPENSITY SCORING (IPS):
        Assign each observed interaction an importance weight:
            w(x, a) = P_new(a | x) / P_old(a | x)
        where P_old is the policy that generated the logged data and
        P_new is the candidate policy being evaluated.
        Reweight the observed rewards by w to estimate new policy performance.
        Works when the logging policy has non-zero probability for all actions.
        Problem: variance is high when the policies are very different
        (some weights can be enormous).

    DOUBLY ROBUST ESTIMATION:
        Combines IPS with a reward model (direct method).
        Reduces variance while maintaining unbiasedness.
        More complex but more reliable for production evaluation.

    CAUSAL INFERENCE IN ML:
        When model predictions cause actions that affect the label,
        standard correlation-based ML conflates cause and effect.
        Structural Causal Models (SCMs) can separate the predictive
        signal from the interventional effect of the model's actions.
        Used in uplift modelling (treatment effect prediction).


### Logging Requirements for Counterfactual Evaluation

    For offline evaluation methods to work, production systems must log:

    1. FEATURES at prediction time (not at label time).
    2. MODEL OUTPUT (the score or decision that was acted on).
    3. ACTION TAKEN (which item was shown, which decision was made).
    4. OUTCOME (click, purchase, fraud, churn — whenever it arrives).
    5. MODEL VERSION that made the prediction.
    6. EXPLORATION PROBABILITY: if any random exploration was applied,
       log the probability that this action was chosen at random.

    This logging discipline is called a "reward log" or "feedback log."
    Without it, counterfactual evaluation is impossible and A/B testing
    is the only option for evaluating model changes.


##### PART 9 — ML SYSTEM DESIGN CASE STUDIES

### Case 1: Real-Time Fraud Detection

    REQUIREMENTS:
    - Decision must be made within 100ms of transaction initiation.
    - False positive rate must be < 0.5% (customer experience).
    - Recall on fraud must be > 90% (business loss).
    - Model must adapt to new fraud patterns within days.

    SYSTEM DESIGN:
    Feature store:
        Offline features (pre-computed, low latency lookup):
            User account age, historical fraud rate, average transaction amount.
        Online streaming features (updated in real-time by Kafka + Flink):
            Number of transactions in last 5 minutes, current session risk.
        Feature freshness: streaming features updated every 1-2 seconds.

    Model:
        Gradient boosted trees (XGBoost): interpretable, fast inference (<5ms),
        handles heterogeneous tabular features well.
        Threshold tuned on validation set to meet the 0.5% FPR constraint.

    Serving:
        REST API, SLA: P99 < 80ms.
        Inference: 5ms. Feature lookup: 3ms. Network: ~10ms. Buffer: 62ms.
        Deployed in the same AWS region as the payment processor.
        Fallback: rule-based system if the ML service is unavailable.

    Monitoring:
        Input drift: PSI on all features daily. Alert if PSI > 0.1.
        Output drift: track daily fraud rate predicted vs observed.
        Ground truth: fraud chargebacks arrive within 60 days.
        Retraining trigger: PSI > 0.25 OR confirmed recall drops below 85%.

    Heterogeneity / shift risks:
        New fraud patterns (concept drift): retrain weekly on rolling 90-day window.
        Geographic expansion (covariate shift): importance weighting on new region data.


### Case 2: Content Recommendation with Feedback Loops

    REQUIREMENTS:
    - Maximise long-term user satisfaction (not just short-term clicks).
    - Avoid filter bubbles and popularity bias.
    - Handle cold start (new users, new items).

    SYSTEM DESIGN:
    Two-stage architecture (common in Netflix, YouTube, Spotify):

    STAGE 1 — CANDIDATE GENERATION (recall):
        Retrieve top-K relevant candidates from millions of items.
        Methods: embedding-based retrieval (two-tower model), collaborative
        filtering, content-based filtering.
        Latency budget: 30ms for retrieval from FAISS/ScaNN index.

    STAGE 2 — RANKING (precision):
        Score and re-rank the K candidates using a rich feature model.
        Features: user context, item features, user-item interaction history.
        Model: gradient boosted trees or deep neural network.
        Latency budget: 20ms.

    Feedback loop mitigation:
        Exploration: force 10-20% of recommendations to be exploratory
        (unexplored items, long-tail items). Log exploration probability.
        Diversity constraints: ensure no more than 3 items from the
        same creator per page.
        Inverse propensity scoring: debias the click signal using the
        probability each item was shown.

    Metrics:
        Online A/B: long-session rate, day-7 retention (not just CTR).
        Diversity metric: intra-list diversity of recommended items.
        Coverage: fraction of catalogue that receives impressions.


### Case 3: LLM-Powered Search with Retrieval Augmentation

    REQUIREMENTS:
    - Answer user queries with cited sources.
    - Latency: first token in 2 seconds, full response in 10 seconds.
    - Knowledge freshness: new documents available within 24 hours.

    SYSTEM DESIGN:
    RAG (Retrieval-Augmented Generation) pipeline:

    INDEXING (offline, daily or event-triggered):
        Documents → chunking (512-token windows with overlap) →
        embedding model (e.g., text-embedding-3-large) →
        vector store (Pinecone, Weaviate, pgvector).

    RETRIEVAL (online, per query):
        Query → embedding → approximate nearest-neighbour search (ANN) →
        top-K chunks (typically K=5-20) → reranking with a cross-encoder →
        top-N chunks (N=3-5) passed to the LLM.

    GENERATION (online):
        LLM (GPT-4, LLaMA) generates a response grounded in the retrieved chunks.
        Response includes citations to source documents.

    MONITORING specific to RAG:
        Retrieval quality: are retrieved chunks relevant to the query?
        Groundedness: does the response contradict the retrieved documents?
        Hallucination rate: how often does the model add facts not in the chunks?
        Latency: P99 of full pipeline (embedding + ANN + rerank + generation).

    DISTRIBUTION SHIFT in RAG:
        Query distribution shift: new topics users ask about.
        Document quality drift: the indexed corpus changes over time.
        Embedding model updates: new embedding model → index must be rebuilt.


### Case 4: Computer Vision on Edge Devices

    REQUIREMENTS:
    - Model runs on embedded camera (ARM Cortex-A CPU, no GPU).
    - Inference: < 200ms per frame at 5 FPS.
    - Model update: over-the-air, must fit in 20 MB.
    - Robustness: must work in varying lighting and weather.

    SYSTEM DESIGN:
    Architecture: EfficientNet-Lite or MobileNetV3 (designed for edge).
    Quantisation: INT8 PTQ for 4× model size reduction.
    Pruning: 50% structured filter pruning for 2× latency reduction.
    Compilation: TFLite or ONNX Runtime with ARM backend optimisations.

    Distribution shift on edge:
        Lighting variation: data augmentation during training (brightness,
        contrast, gamma jitter).
        Domain shift: fine-tune on data from the deployment environment.
        Drift detection without connectivity: store prediction confidence
        locally; upload to cloud when connected; detect anomalous confidence drops.

    OTA update pipeline:
        Train on server → evaluate → compress → sign → push to devices.
        A/B test: push to 5% of devices, monitor for regressions before full rollout.
        Rollback: if P99 confidence drops > 20%, automatically revert to previous model.


### Common ML System Design Interview Mistakes

    MISTAKE 1 — JUMPING TO THE MODEL:
        Starting with "I'd use a transformer" before defining the problem.
        Fix: spend at least 5 minutes clarifying requirements before any
        model discussion. Show you understand the business problem.

    MISTAKE 2 — IGNORING LATENCY AND SCALE:
        Designing a perfect model that cannot serve in the required SLA.
        Fix: ask for latency requirements early; design the serving
        architecture with latency budgets in mind.

    MISTAKE 3 — FORGETTING ABOUT DATA:
        "We'll just use the existing logs."
        Fix: explicitly discuss label quality, label delay, class balance,
        data volume, and whether the logs are biased (exposure bias).

    MISTAKE 4 — NO MONITORING PLAN:
        Treating deployment as the end of the problem.
        Fix: always include a monitoring section. Describe what metrics
        indicate health, what triggers retraining, and how you detect
        distribution shift.

    MISTAKE 5 — IGNORING FAILURE MODES:
        "The model will be 95% accurate."
        Fix: discuss what happens when the model is WRONG. What is the
        cost of a false positive vs false negative? Is there a fallback
        system? How do you handle the model being unavailable?

    MISTAKE 6 — FEEDBACK LOOPS IGNORED:
        For any system where the model influences future data (recommendations,
        content moderation, fraud scoring), not discussing feedback loops is a
        serious gap. Interviewers at senior levels specifically probe for this.

    MISTAKE 7 — TREATING THE FIRST MODEL AS FINAL:
        A good ML system is iterative. Start with a simple baseline
        (rule-based or logistic regression), establish monitoring and
        evaluation infrastructure, then iterate to more complex models.
        The baseline tells you what "good" looks like before you spend
        months training a transformer.


### Reference: The ML System Design Checklist

    PROBLEM FRAMING:
    ✓ Business metric defined and understood
    ✓ ML metric chosen and justified
    ✓ Ground truth label defined + latency of label arrival
    ✓ Constraints: latency SLA, throughput, budget, data privacy

    DATA:
    ✓ Data sources identified, volume estimated
    ✓ Label strategy defined (explicit/implicit/programmatic)
    ✓ Temporal split strategy (no data leakage)
    ✓ Handling of missing labels and delayed labels
    ✓ Data versioning approach

    TRAINING:
    ✓ Feature engineering approach
    ✓ Model choice justified for latency/accuracy/interpretability
    ✓ Training pipeline and orchestration
    ✓ Experiment tracking and model registry

    SERVING:
    ✓ Online vs offline serving decision
    ✓ Latency budget decomposed (feature retrieval + inference + network)
    ✓ Fallback if model unavailable
    ✓ A/B test infrastructure

    MONITORING:
    ✓ Operational metrics (latency, error rate, throughput)
    ✓ Data quality monitoring (schema, nulls, distribution)
    ✓ Model performance monitoring (with ground truth plan)
    ✓ Drift detection method and thresholds
    ✓ Alert routing (P0/P1/P2)
    ✓ Retraining trigger criteria

"""


# ─────────────────────────────────────────────────────────────────────────────
# No OPERATIONS (theory-only module)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {}


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