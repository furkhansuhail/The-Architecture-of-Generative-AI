# 🤖 The Architecture of Generative AI

> A structured, progressive reference hub for Generative AI — from tokenization to multi-agent systems.  
> Companion project to [The Architecture of Intelligence](../The-Architecture-of-Intelligence) (Deep Learning).

---

## 🗺️ Learning Path

```
01 · Tokenization & Embeddings
        ↓
02 · Language Modeling
        ↓
03 · Transformer LLMs  (GPT → BERT → T5)
        ↓
04 · Pretraining  (objectives, data, scale)
        ↓
05 · Fine-Tuning  (Full FT → LoRA → QLoRA)
        ↓
06 · Alignment  (RLHF → DPO → Constitutional AI)
        ↓
07 · Prompt Engineering  (Zero-shot → CoT → ReAct)
        ↓
08 · RAG  (Naive → Advanced → GraphRAG)
        ↓
09 · Inference Optimization  (Quantization → vLLM)
        ↓
10 · AI Agents  (Tool Use → Planning → Memory)
        ↓
11 · Multi-Agent Systems  (LangGraph, CrewAI)
        ↓
12 · Generative Models  (VAE → GAN → Diffusion)
        ↓
13 · Multimodal AI  (CLIP → VLMs)
        ↓
14 · Architecture Innovations  (MoE → Mamba)
        ↓
15 · Evaluation & Safety
```

---

## 📁 Project Structure

```
The-Architecture-of-GenAI/
│
├── app.py                          # Streamlit app entry point
│
├── topics/                         # Core concept modules (auto-discovered)
│   ├── __init__.py                 # Auto-discovery logic
│   ├── topic_template.py           # Blueprint for new topics
│   ├── 01_tokenization_embeddings.py
│   ├── 02_language_modeling.py
│   ├── 03_transformer_llms.py
│   ├── 04_pretraining.py
│   ├── 05_fine_tuning.py
│   ├── 06_alignment.py
│   ├── 07_prompt_engineering.py
│   ├── 08_rag.py
│   ├── 09_inference_optimization.py
│   ├── 10_ai_agents.py
│   ├── 11_multi_agent_systems.py
│   ├── 12_generative_models.py
│   ├── 13_multimodal_ai.py
│   ├── 14_architecture_innovations.py
│   └── 15_evaluation_safety.py
│
├── Implementation/                 # Runnable implementation files
│   ├── 01_Tokenization_and_Embeddings/
│   │   └── 01_Tokenization.py
│   ├── 02_Language_Modeling/
│   ├── 03_Transformer_LLMs/
│   ├── 04_Pretraining/
│   ├── 05_Fine_Tuning/
│   │   ├── 05a_Full_Fine_Tuning.py
│   │   ├── 05b_LoRA_From_Scratch.py
│   │   └── 05c_QLoRA.py
│   ├── 06_Alignment/
│   ├── 07_Prompt_Engineering/
│   ├── 08_RAG/
│   │   ├── 08a_Naive_RAG.py
│   │   ├── 08b_Advanced_RAG.py
│   │   └── 08c_GraphRAG.py
│   ├── 09_Inference_Optimization/
│   ├── 10_AI_Agents/
│   ├── 11_Multi_Agent_Systems/
│   ├── 12_Generative_Models/
│   ├── 13_Multimodal_AI/
│   ├── 14_Architecture_Innovations/
│   └── 15_Evaluation_Safety/
│
├── Required_Images/                # HTML visual breakdowns (rendered as iframes)
│   ├── tokenization_visual.py
│   ├── transformer_llm_visual.py
│   ├── rag_visual.py
│   └── ...
│
└── README.md
```

---

## 🚀 Running the App

```bash
# Install dependencies
pip install streamlit transformers torch gensim

# Run
streamlit run app.py
```

---

## 📐 Adding a New Topic

1. Copy `topics/topic_template.py`
2. Rename it with a numeric prefix (e.g. `16_new_topic.py`)
3. Fill in `THEORY`, `OPERATIONS`, and `visual_html`
4. Add a visual file to `Required_Images/new_topic_visual.py`
5. The app auto-discovers it — no registration needed

---

## 🔗 Connection to Architecture of Intelligence

| Deep Learning Project | Gen AI Project                  |
|----------------------|----------------------------------|
| Perceptron → MLP     | Tokenization → Language Modeling |
| CNN, RNN             | Transformer LLM families         |
| Transformer          | Pretraining paradigms            |
| Full Fine-Tuning     | Instruction Fine-Tuning          |
| LoRA / QLoRA         | LoRA / QLoRA (extended)          |
| —                    | Alignment (RLHF, DPO)            |
| —                    | RAG, Agents, Multi-Agent         |
| —                    | Diffusion, Multimodal            |

---

*Built as a structured learning reference — theory, visuals, and runnable code for every concept.*
