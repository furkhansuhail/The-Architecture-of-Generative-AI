"""
Learning Path — Generative AI Roadmap
==============
A structured path from foundational concepts for Gen AI.

This module serves as both a guide and the first topic in the app.
"""
"""Module 00: Tokenization & Embeddings"""
from Required_Images.tokenization_visual import get_visual_html

import os
import re
import sys
import subprocess
from pathlib import Path
import base64
import textwrap


# ── Display name (shown in sidebar and as page title) ──────────────────────
TOPIC_NAME   = "00 · Roadmap for Generative AI"
DISPLAY_NAME = "00 · Roadmap for Generative AI"
ICON         = "📖"
SUBTITLE     = "This is the suggested roadmap to follow to understand Generative AI Roadmap."

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif",  "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">️ Image not found: {path}</p>'
# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif",  "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">️ Image not found: {path}</p>'

# ── Theory ─────────────────────────────────────────────────────────────────
THEORY = """

This roadmap walks you through the complete landscape of Generative AI — from core foundations all the way to 
building production-grade systems. Each phase builds on the last. 

Whether you're an engineer, researcher, or practitioner, this is the order that makes sense.

### Phase 1 — Foundations You Must Have

Before writing a single prompt or building a single pipeline, you need a working understanding of:

    - Machine Learning Basics: supervised vs. unsupervised learning, loss functions, overfitting, evaluation metrics
    
    - Deep Learning Literacy: what a neural network is, what training means, what parameters do — 
      you don't need to implement backprop, but you need the intuition
    
    - Linear Algebra & Probability: vectors, matrices, dot products, probability distributions, Bayes' theorem
    
    - APIs & HTTP: how to call REST APIs, handle JSON, manage auth — you'll be calling model APIs constantly

Key mindset: you don't need to be a researcher. You need enough depth to debug, design, and make informed decisions.

---

### Phase 2 — How Language Models Actually Work

Before building with LLMs you need to understand what's happening under the hood:

    - Tokenization: how text becomes numbers (BPE, WordPiece, SentencePiece) — token ≠ word

    - The Transformer Architecture: self-attention, multi-head attention, positional encoding, encoder vs. decoder
    
    - Autoregressive Generation: how models predict one token at a time; why order and context matter

    - Pre-training vs. Fine-tuning: what a base model is, what instruction tuning does, why they behave differently
    
    - Scaling Laws: why bigger models on more data produce qualitatively different behavior (emergent capabilities)
    
    - Context Windows: what they are, why they're a hard constraint, and how they affect everything you build

Key concepts: temperature, top-k, top-p (nucleus sampling), softmax, logits, perplexity

---

### Phase 3 — Prompt Engineering

The first practical skill and one of the highest-leverage ones:

    - Zero-shot Prompting: asking the model directly with no examples

    - Few-shot Prompting: including examples in the prompt to guide behavior

    - Chain-of-Thought (CoT): instructing the model to reason step-by-step before answering
    
    - System Prompts: setting role, tone, constraints, and context at the system level
    
    - Instruction Formatting: XML tags, JSON structures, markdown — how formatting affects output quality
    
    - Output Control: asking for specific formats, lengths, and structures
    
    - Prompt Failure Modes: hallucination, instruction drift, sycophancy, over-refusal

Key skill: the ability to decompose a task into a prompt that reliably produces the output you need.

---

### Phase 4 — Working with Model APIs
Moving from experimentation to actual software:

API Fundamentals: messages array structure, roles (system/user/assistant), max tokens, stop sequences

    - Streaming: receiving tokens as they generate rather than waiting for a full response
    
    - Latency vs. Quality Tradeoffs: choosing between frontier models and smaller/faster ones
    
    - Rate Limits & Cost Management: tokens-per-minute, tokens-per-dollar, batching strategies
    
    - Error Handling: timeouts, retries with exponential backoff, fallback strategies
    
    - Model Comparison: OpenAI GPT-4o, Anthropic Claude, Google Gemini, Meta LLaMA — strengths and tradeoffs

Key tools: OpenAI SDK, Anthropic SDK, LiteLLM (multi-provider abstraction)

---

### Phase 5 — Retrieval-Augmented Generation (RAG)

The single most important production pattern in Gen AI today:

    - Why RAG: LLMs have fixed knowledge cutoffs and limited context — RAG grounds them in current, specific data
    
    - The RAG Pipeline: document ingestion → chunking → embedding → vector storage → retrieval → generation
    
    - Embeddings: what they are, how semantic similarity works, cosine distance
    
    - Vector Databases: Pinecone, Weaviate, Chroma, pgvector — when to use each
    
    - Chunking Strategies: fixed-size, sentence-aware, recursive, semantic chunking — chunking quality drives retrieval quality
    
    - Retrieval Methods: dense retrieval (ANN search), sparse retrieval (BM25), hybrid approaches
    
    - Reranking: using a cross-encoder to re-score retrieved chunks before passing to the LLM
    
    - RAG Failure Modes: chunk boundaries cutting context, retrieval fetching irrelevant docs, LLM ignoring retrieved content

Key tools: LangChain, LlamaIndex, Haystack, OpenAI Embeddings, Cohere Rerank

---

### Phase 6 — Fine-Tuning & Adaptation

When prompt engineering and RAG aren't enough:

    - When to Fine-Tune: style/tone consistency, domain-specific vocabulary, reducing prompt length at scale
    
    - Instruction Fine-Tuning: training on (instruction, response) pairs to steer model behavior
    
    - RLHF (Reinforcement Learning from Human Feedback): how models like ChatGPT and Claude were aligned — reward modeling, PPO
    
    - RLAIF: using AI feedback instead of human feedback to scale alignment
    
##### Parameter-Efficient Fine-Tuning (PEFT):

    - LoRA (Low-Rank Adaptation): inserting low-rank matrices — the dominant method today
    
    - QLoRA: quantized LoRA, fine-tuning large models on consumer hardware
    
    - Prefix Tuning / Prompt Tuning: prepending trainable tokens instead of updating weights

    - Dataset Construction: quality matters more than quantity; annotation consistency; avoiding bias
    
    - Evaluation After Fine-Tuning: benchmark drift, alignment tax, regression testing

Key tools: Hugging Face PEFT, Axolotl, Unsloth, Modal, Together AI fine-tuning API

---

### Phase 7 — Agents & Tool Use

Where LLMs stop being text completers and start being autonomous systems:

    - What is an Agent: an LLM in a loop that can take actions, observe results, and iterate
    
    - Tool Calling / Function Calling: giving the model a schema of available functions it can invoke
    
    - The ReAct Pattern: Reason → Act → Observe → repeat — the core agentic loop
    
    - Planning: task decomposition, subgoal generation, handling multi-step dependencies
    
    Memory Systems:
    * In-context memory (within the conversation window)
    * External memory (vector stores, databases)
    * Episodic memory (summarizing past interactions)


* Multi-Agent Systems: orchestrator agents, specialist subagents, agent-to-agent communication

* Agent Failure Modes: hallucinated tool calls, infinite loops, compounding errors, over-autonomy

Key tools: LangGraph, AutoGen, CrewAI, OpenAI Assistants API, Anthropic tool use

---

### Phase 8 — Evaluation & Reliability

The most under-invested area in Gen AI — and the one that breaks production systems:

    - Why Evaluation is Hard: outputs are open-ended, subjective quality, no single ground truth
    
    - LLM-as-Judge: using a model to evaluate another model's output — fast but has biases
    
    - Human Evaluation: gold standard but expensive and slow; necessary for calibrating automated evals
    
    - Benchmark Types: task-specific (MMLU, HumanEval), capability (HELM), safety (TruthfulQA)
    
    - RAG-Specific Evals: RAGAS — faithfulness, answer relevancy, context recall
    
    - Regression Testing: ensuring new model versions or prompt changes don't break existing behavior
    
    - Red-Teaming: adversarial testing for jailbreaks, prompt injection, data leakage

Key concepts: precision/recall for retrieval, faithfulness vs. relevance, eval-driven development

---

### Phase 9 — Production & MLOps for Gen AI

Taking systems from prototype to reliable, scalable product:

    - Prompt Management: versioning prompts, A/B testing, separating prompts from application code
    
    - Observability & Tracing: logging inputs, outputs, latency, token counts, costs — end-to-end traces
    
    - Caching: semantic caching to reduce cost and latency on repeated or similar queries
    
    - Guardrails: input/output filtering, topic restriction, PII detection, content moderation
    
    - Latency Optimization: streaming, model distillation, speculative decoding, request batching
    
    - Cost Optimization: prompt compression, model routing (use small model when possible), caching
    
    - Infrastructure Choices: managed APIs vs. self-hosted open-source models (Llama, Mistral, Qwen)

Key tools: LangSmith, Weights & Biases, Helicone, Braintrust, Guardrails AI, Nemo Guardrails

---

Phase 10 — Multimodal & Frontier Systems

The expanding frontier beyond text:

    - Vision-Language Models: GPT-4o, Claude 3.5, Gemini — understanding and reasoning over images
    
    - Image Generation: Diffusion models (Stable Diffusion, DALL-E, Flux) — denoising as generation
    
    - Audio & Speech: Whisper for ASR, ElevenLabs/Cartesia for TTS, Suno/Udio for music generation
    
    - Video Generation: Sora, Runway Gen-3, Kling — temporal coherence as the core challenge
    
    - Code Generation: Copilot, Cursor, Codestral — LLMs as a coding layer
    
    - Mixture of Experts (MoE): routing inputs to specialist sub-networks — Mixtral, GPT-4 (rumored)
    
    - Long Context Models: Gemini 1.5 (1M tokens), Claude (200K) — implications for RAG and agents
    
    - Reasoning Models: OpenAI o1/o3, DeepSeek-R1 — extended chain-of-thought at inference time

---

### Phase 11 — Safety, Ethics & Alignment

Not optional. Not last. Should be considered from Phase 1:

    -  Hallucination: why models confabulate, how to detect it, mitigation strategies (RAG, citations, CoT)
    
    - Prompt Injection: attackers hijacking your agent via malicious content in retrieved data
    
    - Bias & Fairness: training data biases reflected in outputs, demographic disparities in model behavior
    
    - Privacy: training data memorization, PII risks in RAG pipelines, data retention policies
    
    - Alignment Approaches: RLHF, Constitutional AI, RLAIF — how values are instilled in models
    
    - Responsible Deployment: disclosure of AI use, human-in-the-loop design, meaningful oversight
    
    - Regulatory Landscape: EU AI Act, US executive orders, emerging compliance requirements

Key reading: Anthropic's Responsible Scaling Policy, DeepMind's safety research, AI safety benchmarks

---

### Learning Path by Role

    Role                            Priority Phases
    AI Engineer / Builder           1 → 2 → 3 → 4 → 5 → 7 → 9
    ML Researcher                   1 → 2 → 6 → 8 → 10 → 11
    Product Manager                 2 → 3 → 5 → 8 → 11
    Data Scientist                  1 → 2 → 5 → 6 → 8
    Security / Trust & Safety       2 → 3 → 8 → 11


## Suggested Resources

    * Books: Designing Large Language Model Applications (Chip Huyen), The Art of Prompt Engineering
    * Courses: DeepLearning.AI short courses (free), Fast.ai Practical Deep Learning, Hugging Face NLP Course
    * Papers: Attention Is All You Need, RLHF (Ouyang et al. 2022), LoRA (Hu et al. 2021), RAG (Lewis et al. 2020)
    * Hands-On: Build a RAG pipeline. Fine-tune a small model. Deploy one agent. Evaluate it rigorously.

"""

# ── Step-by-Step Operations ─────────────────────────────────────────────────
OPERATIONS = {}

# ── Entry point called by topics/__init__.py ────────────────────────────────
def get_content():
    """Return all content for this topic module."""
    try:
        from Required_Images.tokenization_visual import get_visual_html, VISUAL_HEIGHT
        visual_html   = get_visual_html()
        visual_height = VISUAL_HEIGHT
    except Exception:
        visual_html   = ""
        visual_height = 400

    tok_img = _image_to_html(
        "Required_Images/Tokenization_Breakdown.png",
        alt="Tokenization & Embedding Pipeline",
        width="80%",
    )
    theory_with_images = THEORY.replace("{{TOKENIZATION_IMAGE}}", tok_img)

    interactive_components = [
        {
            "placeholder": "{{TOKENIZATION_IMAGE}}",
            "html":        visual_html,
            "height":      visual_height,
        }
    ]

    return {
        "display_name":          DISPLAY_NAME,
        "icon":                  ICON,
        "subtitle":              SUBTITLE,
        "theory":                theory_with_images,
        "theory_raw":            THEORY,
        "visual_html":           visual_html,
        "complexity":            None,
        "operations":            OPERATIONS,
        "render_operations":     None,
        "interactive_components": interactive_components,
    }
