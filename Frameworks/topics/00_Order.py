import textwrap
import re

TOPIC_NAME = "Order for Frameworks"
DISPLAY_NAME = "00 · Order for Frameworks"
ICON = "⚖️"
SUBTITLE = "Order for Frameworks Core with priorities"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### Frameworks in AI/ML

A framework is the layer between your mathematical ideas and working code — it handles the infrastructure 
so you can focus on the model. Choosing the right framework shapes everything from how fast you iterate 
to how your model eventually runs in production.

## The Core Problem

Training a neural network requires differentiating through millions of operations, managing GPU memory, 
batching data efficiently, checkpointing state, and eventually serving predictions at scale. 
Writing all of this from scratch for every project is not just tedious — it's a source of subtle bugs that 
corrupt gradients, leak memory, or silently degrade performance. Frameworks encode decades of collective 
engineering knowledge into reusable, tested abstractions.

## Two Levels of the Stack

Low-level frameworks give you direct control over computation graphs, tensor operations, and gradient flow. 
PyTorch and JAX live here — you're close to the metal, which means flexibility but also responsibility. 
High-level frameworks sit on top of these, abstracting away training loops, logging, and device placement. 
PyTorch Lightning, Keras, and Hugging Face Transformers live here — you trade some control for speed and 
standardization.

Neither level is strictly better. Research favors low-level control; production and team environments often 
favor high-level consistency.

## Classical ML Frameworks

**scikit-learn** is the foundation of applied ML. Its unified API — fit, transform, predict — works across 
hundreds of algorithms and makes pipelines composable. If your problem can be solved without deep learning, 
sklearn is almost always the right starting point.

**XGBoost** and **LightGBM** are gradient boosting libraries that consistently win structured/tabular data 
competitions. XGBoost pioneered the space and is extremely robust; LightGBM trades some interpretability for 
dramatically faster training on large datasets through histogram-based splitting and leaf-wise tree growth.

## Deep Learning Frameworks

**PyTorch** has become the dominant research framework. Its define-by-run (eager execution) model means 
your computation graph is built dynamically as Python executes — debugging feels natural because you can 
drop a breakpoint anywhere in the forward pass. Virtually all frontier model research is now done in PyTorch.

**PyTorch Lightning** wraps PyTorch to enforce a clean separation between research code (the LightningModule) 
and engineering code (the Trainer). It handles distributed training, checkpointing, logging, and mixed 
precision without you writing boilerplate, while keeping your model logic untouched.

**TensorFlow** was the dominant framework before PyTorch's rise. Its static graph model (define-then-run) 
made deployment straightforward — a frozen TF graph is self-contained and portable. TensorFlow 2.x added 
eager execution, closing the gap with PyTorch for research use.

**Keras** is TensorFlow's high-level API, designed for rapid prototyping. Its Sequential and Functional APIs 
let you build standard architectures in dozens of lines. Keras 3 now runs on top of PyTorch, TensorFlow, 
or JAX interchangeably, making it a genuinely backend-agnostic option.

**JAX** takes a different philosophy — it treats ML as functional programming over NumPy-like arrays. 
Pure functions get compiled via XLA, vectorized via vmap, and parallelized via pmap. JAX is increasingly 
used for frontier research where maximum performance and mathematical clarity matter, but its functional 
style has a steeper learning curve.

**Flax** is JAX's primary neural network library, providing the layer abstractions and parameter management 
that JAX deliberately omits.

## NLP and LLM Frameworks

**Hugging Face Transformers** is the central hub of modern NLP. It provides pretrained model weights, 
tokenizers, and training utilities for virtually every major architecture — BERT, GPT, T5, LLaMA, Mistral, 
and hundreds more. Fine-tuning a state-of-the-art language model on a custom dataset went from a research 
project to an afternoon of work because of this library.

**LangChain** addresses a different problem — orchestrating LLMs into applications. 
It provides abstractions for chains (sequences of LLM calls), agents (LLMs that choose which tools to invoke), 
memory (persisting context across turns), and retrieval. It's most useful when your application involves 
multi-step reasoning or combining an LLM with external data sources.

**LlamaIndex** specializes in Retrieval-Augmented Generation (RAG). Where LangChain is a general orchestration 
layer, LlamaIndex is specifically designed for indexing your own documents and querying them through an LLM — 
handling chunking strategies, embedding, vector store integration, and query routing.

## Computer Vision

**OpenCV** is the bedrock of computer vision engineering. Before deep learning reshaped the field, OpenCV 
was how you did vision — it contains classical algorithms for edge detection, optical flow, camera calibration, 
feature matching (SIFT, ORB), and much more. It remains essential for preprocessing, augmentation pipelines, 
and any task where you're working directly with image data rather than training models.

## Reinforcement Learning

**Gymnasium** (the maintained fork of OpenAI Gym) is the standard interface for RL environments. 
It defines a universal API — reset, step, render — that lets you swap environments without changing your 
agent code. Whether you're training on CartPole or a custom simulation, your agent interacts with the same 
interface. The ecosystem around Gymnasium includes thousands of environments and clean integrations with 
libraries like Stable-Baselines3.

## Distributed Training

**Ray** is a distributed computing framework built for ML workloads. Ray Core handles task parallelism; 
Ray Train abstracts distributed deep learning across multiple GPUs or nodes; Ray Tune handles 
hyperparameter search at scale; Ray Serve handles model serving. It's a cohesive ecosystem for scaling 
every stage of the ML lifecycle.

**Kubeflow** brings ML workflows to Kubernetes. Where Ray is developer-facing and relatively easy to start 
locally, Kubeflow is infrastructure-facing — it orchestrates multi-step pipelines (data prep, training, 
evaluation, serving) as Kubernetes-native workflows. It's the choice when your organization already runs 
on Kubernetes and needs reproducible, auditable pipelines at enterprise scale.

## Deployment

**ONNX** (Open Neural Network Exchange) is a model interchange format. You train in PyTorch, export to ONNX, 
and then run inference using any ONNX-compatible runtime — including ONNX Runtime, TensorRT, or CoreML. 
It's the glue layer that decouples training frameworks from inference infrastructure.

**MLflow** handles the experiment tracking, model registry, and deployment packaging problems that 
frameworks deliberately leave out. It records hyperparameters, metrics, and artifacts for every run, 
lets you compare experiments visually, and provides a model registry for promoting models from staging 
to production with governance.

**TorchServe** is PyTorch's native model serving solution. It handles batching incoming requests, 
managing multiple model versions, scaling across GPUs, and exposing REST and gRPC endpoints. 
It's the production inference layer for organizations whose models live in PyTorch.

## How the Layers Connect

A typical production ML system uses all of these layers simultaneously. 
A model trained in PyTorch Lightning with Hugging Face Transformers gets tracked in MLflow, 
exported to ONNX for portability, and served via TorchServe — with Ray orchestrating distributed training 
upstream and Kubeflow running the end-to-end pipeline in production.

Understanding frameworks isn't about memorizing APIs. It's about knowing which layer of the stack 
each tool owns, what tradeoffs it was designed to make, and how they compose into a system that can 
move a model from a research idea to a reliable production service.


    interpretability/
    ├── scope_and_taxonomy/
    ├── intrinsic_models/
    │   ├── linear_models
    │   ├── decision_trees
    │   └── rule_lists
    ├── post_hoc_agnostic/
    │   ├── feature_importance/
    │   │   ├── permutation_importance
    │   │   └── integrated_gradients
    │   ├── local_explanations/
    │   │   ├── SHAP
    │   │   ├── LIME
    │   │   └── anchors
    │   └── global_visualizations/
    │       ├── PDP
    │       ├── ICE
    │       └── ALE
    ├── nn_specific_xai/
    │   ├── gradient_based/
    │   │   ├── saliency_maps
    │   │   ├── GradCAM
    │   │   ├── LRP
    │   │   └── DeepLIFT
    │   └── attention_analysis/
    │       └── BERTViz
    ├── counterfactual/
    ├── concept_based/
    ├── mechanistic_interpretability/
    │   ├── circuits
    │   ├── probing_classifiers
    │   └── superposition
    ├── model_calibration
    ├── evaluation_of_explanations/
    └── fairness_and_bias/

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }

