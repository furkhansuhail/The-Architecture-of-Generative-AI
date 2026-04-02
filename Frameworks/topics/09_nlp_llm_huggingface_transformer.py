"""
HuggingFace Transformers — Industrial-Strength NLP and Beyond
=============================================================

HuggingFace is a company founded in 2016 that has become the central
infrastructure layer of modern machine learning. Their open-source
`transformers` library, launched in 2019, democratised access to
state-of-the-art pretrained models and is now the most starred ML
repository on GitHub with over 130,000 stars.

Before HuggingFace, using a pretrained BERT or GPT model required
navigating each research group's bespoke codebase, custom tokenisers,
and idiosyncratic checkpoints. HuggingFace standardised all of this
behind three consistent abstractions — Model, Tokenizer, and Trainer —
that work identically whether you are doing sentiment analysis, machine
translation, image classification, or audio transcription.

At the core of this ecosystem is the Transformer architecture itself:
a model built entirely on self-attention mechanisms, first introduced
in the landmark 2017 paper "Attention Is All You Need" (Vaswani et al.).
Understanding the Transformer is not optional background knowledge — it
is the foundation upon which all of modern language modelling, computer
vision (ViT), multi-modal AI (CLIP, Flamingo), and code generation
(Codex, StarCoder) is built.

This module covers the full HuggingFace ecosystem: the theory of the
Transformer architecture from first principles (attention, positional
encoding, encoder vs decoder), the major model families and when to
use each, the tokenization pipeline, the Pipeline API for zero-code
inference, fine-tuning with the Trainer API, parameter-efficient
fine-tuning with LoRA/PEFT, quantization for deployment, and the
Model Hub as a component registry for modern ML.

"""

import textwrap
import re

TOPIC_NAME   = "HuggingFace Transformers — Industrial-Strength NLP and Beyond"
DISPLAY_NAME = "09 · HuggingFace Transformers"
ICON         = "🤗"
SUBTITLE     = "From Attention Fundamentals to Fine-Tuning and Deployment"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT IS A TRANSFORMER? MOTIVATION AND HISTORY

### The Problem Before Transformers

    Before 2017, sequence modelling was dominated by Recurrent Neural Networks
    (RNNs) and their variants (LSTMs, GRUs). These architectures process
    sequences token by token, left to right, maintaining a hidden state that
    carries information forward in time.

    The fundamental pathology of RNNs is the sequential bottleneck:

        x₁ → h₁ → x₂ → h₂ → x₃ → h₃ → ... → xₙ → hₙ → output

        - Information from x₁ must travel through n-1 hidden states to
          influence the output at position n.
        - Long-range dependencies either vanish (vanishing gradients) or
          explode. LSTMs ameliorate this but cannot eliminate it.
        - Steps must execute sequentially — no parallelism during training.
          A sequence of 1,000 tokens requires 1,000 sequential operations.
        - The fixed-size hidden state is an information bottleneck.

    Attention mechanisms were introduced as add-ons to encoder-decoder RNNs
    (Bahdanau et al., 2015) — the decoder could "look back" at all encoder
    states instead of relying solely on a final context vector. This worked
    dramatically better for machine translation.

### The Transformer: Attention Is All You Need (2017)

    Vaswani et al. asked a radical question: what if we removed the recurrence
    entirely and built a model using ONLY attention?

    The key insight: attention mechanisms compute relationships between ALL
    positions in a sequence simultaneously. No sequential dependency.
    Every token can directly attend to every other token in O(1) steps.

    Why this was revolutionary:
        1. PARALLELISM: the entire sequence is processed simultaneously.
           Training a 1,000-token sequence is the same number of operations
           as training a 10-token sequence (in terms of sequential depth).
        2. LONG-RANGE DEPENDENCIES: any two positions are one attention step
           apart, regardless of their distance in the sequence.
        3. INTERPRETABILITY: attention weights show what the model "looks at",
           providing some degree of explainability.
        4. SCALABILITY: without sequential bottlenecks, models could be made
           arbitrarily wide and deep, and scaled with data and compute.

### From Transformer to Modern LLMs: The Timeline

    2017: "Attention Is All You Need" — original Transformer for translation.
    2018: GPT-1 (OpenAI) — Transformer decoder pretrained on language modelling.
          BERT (Google) — Transformer encoder pretrained with masked language model.
    2019: GPT-2 (OpenAI) — 1.5B params, zero-shot text generation.
          T5 (Google) — Encoder-decoder, "text-to-text" unification.
          RoBERTa (Facebook) — Optimised BERT pretraining.
    2020: GPT-3 (OpenAI) — 175B params, in-context learning, few-shot.
          BART (Facebook) — Denoising encoder-decoder.
    2021: CLIP (OpenAI) — Contrastive vision-language pretraining.
          ViT (Google) — Vision Transformer — images as sequences of patches.
          T0 / FLAN (Google) — Instruction-tuned T5 variants.
    2022: InstructGPT (OpenAI) — RLHF alignment. ChatGPT basis.
          OPT / BLOOM — Open large language models.
          Stable Diffusion — Latent diffusion with transformer text encoder.
    2023: LLaMA / LLaMA-2 (Meta) — Efficient open-weight LLMs.
          Mistral 7B — Sliding window attention, grouped query attention.
          GPT-4 (OpenAI) — Multimodal.
          Falcon, Phi-1/2 — Alternative open models.
    2024: LLaMA-3, Mistral variants, Gemma (Google), Qwen (Alibaba) — dense.
          Mixture of Experts: Mixtral, DeepSeek. Command R+ (Cohere).
    2025: LLaMA-3.3, DeepSeek-R1 (chain-of-thought reasoning), Gemma-3.

    The common thread: all of these are Transformers. The architecture
    scales across orders of magnitude of parameter count, modalities,
    and tasks without fundamental redesign.


##### PART 2 — ATTENTION MECHANISM: THE MATHEMATICAL CORE

### What Attention Computes

    Attention is a function that maps a query and a set of key-value pairs
    to an output. The output is a weighted sum of the values, where the
    weight for each value is determined by the compatibility (similarity)
    between the query and its corresponding key.

    Intuitively:
        QUERY (Q):  "What am I looking for?"
                    The representation of the current position.
        KEY   (K):  "What do I contain?"
                    The representation that gets compared against queries.
        VALUE (V):  "What do I provide?"
                    The information that gets aggregated when attention fires.

    The analogy is a fuzzy dictionary lookup:
        In a hard dictionary: exact(key) → value
        In attention:         weighted similarity(query, all keys) → weighted sum of values

### Scaled Dot-Product Attention: The Formula

    Given matrices Q, K, V of shape (seq_len, d_k):

        Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V

    Step by step:
        1. Q Kᵀ:           Compute similarity between every query and every key.
                           Result: (seq_len × seq_len) matrix of raw scores.
                           Score[i, j] = how much position i attends to position j.

        2. / √d_k:         Scale by the square root of key dimension.
                           Why? Without scaling, dot products grow large as d_k
                           increases, pushing softmax into regions of very small
                           gradients (vanishing gradient in softmax).
                           √d_k normalises the variance of the dot products.

        3. softmax(·):     Normalise scores into a probability distribution.
                           Each row sums to 1. Now it's a valid attention distribution.
                           Masking is applied here: set forbidden positions to -∞
                           before softmax so they get ~0 attention weight.

        4. · V:            Weighted sum of values.
                           Result: (seq_len, d_v) — each position gets a mixture
                           of all values, weighted by how strongly it attends to them.

    Example:
        "The cat sat on the mat because it was tired."
        When computing the representation of "it", Q for "it" will have
        high attention weights on "cat" (K for "cat") because they are
        semantically compatible in this context. The value V of "cat"
        contributes heavily to the output representation of "it".

### Why Scale by √d_k?

    If Q and K have components with unit variance, then Q·K (dot product
    of d_k-dimensional vectors) has variance d_k. For large d_k (e.g. 512
    or 1024), the dot products are very large in magnitude.

    softmax(x) ≈ one-hot when x contains large values — the gradient
    ∂softmax/∂x ≈ 0 everywhere except the peak. This kills learning.

    Dividing by √d_k brings variance back to ~1 regardless of d_k.

### Causal (Masked) Self-Attention

    In autoregressive language models (GPT family), a token at position i
    must NOT attend to positions j > i — it has not been generated yet.

    Implementation: before the softmax, add a mask matrix:
        mask[i, j] = 0     if j <= i   (allowed — past or current)
        mask[i, j] = -∞    if j > i    (forbidden — future)

    After softmax, forbidden positions get weight e^(-∞) = 0.
    This is called causal masking or the autoregressive mask.
    It allows training on all positions simultaneously while preserving
    the autoregressive property.

### Multi-Head Attention

    Single attention head: one set of Q, K, V projections.
    Multi-head attention: h parallel attention heads, each learning
    different aspects of the relationship structure.

        head_i = Attention(Q · Wᵢ_Q,  K · Wᵢ_K,  V · Wᵢ_V)

    Each head has its own learned projection matrices Wᵢ_Q, Wᵢ_K, Wᵢ_V.
    Heads can specialise: one head tracks syntactic dependencies, another
    tracks coreference, another tracks positional offsets.

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

    Dimensions:
        d_model:     total model dimension (e.g. 512, 768, 1024, 4096)
        h:           number of heads (e.g. 8, 12, 16, 32)
        d_k = d_v:   per-head dimension = d_model / h

    The total parameter count of multi-head attention stays the same
    as single-head with d_model dimensions: h × (d_k projections × 3 + d_v)
    ≈ 4 × d_model² per attention layer.

### Modern Attention Variants

    Grouped Query Attention (GQA — Llama-3, Mistral):
        Fewer key-value heads than query heads. Multiple query heads share
        one KV head. Reduces KV-cache memory during inference by n_kv_ratio.
        n_kv_heads = 8 with n_query_heads = 32 → 4× smaller KV-cache.

    Multi-Query Attention (MQA — Falcon):
        Extreme case of GQA: single KV head shared by all query heads.
        Fastest inference, least memory, slight quality reduction.

    Sliding Window Attention (SWA — Mistral):
        Each token only attends to w tokens around it (window size w).
        Enables constant-memory attention for very long sequences.
        Global attention layers interspersed for long-range information.

    Flash Attention (Dao et al., 2022):
        NOT a different attention formula — exact same computation.
        A hardware-aware algorithm that tiles the attention computation
        to fit in GPU SRAM, avoiding expensive HBM reads/writes.
        Result: same output, 2–8× faster, O(N) memory vs O(N²).


##### PART 3 — THE TRANSFORMER ARCHITECTURE IN FULL

### From Token to Vector: Embedding and Positional Encoding

    Input pipeline before the Transformer layers:

        1. TOKENISATION:  raw text → integer token IDs
           "Hello world" → [101, 7592, 2088, 102]

        2. TOKEN EMBEDDING: integer IDs → dense vectors
           Embedding table: shape (vocab_size, d_model)
           token_id=7592 → d_model-dimensional vector

        3. POSITIONAL ENCODING: add position information
           Self-attention is permutation-invariant — "the cat sat" and
           "sat the cat" would produce the same attention scores without
           positional information.

    Positional encoding schemes:
        Sinusoidal (original 2017):
            PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
            Fixed, not learned. Generalises to unseen sequence lengths.

        Learned absolute (BERT, GPT-2):
            A learned embedding table indexed by position (0, 1, 2, ...).
            Cannot generalise beyond max training sequence length.

        Relative positional encoding (T5, DeBERTa):
            Encodes the relative offset (j - i) between positions.
            Generalises better to unseen lengths. T5 uses a bias table.

        RoPE — Rotary Position Embedding (LLaMA, Mistral, GPT-NeoX):
            Encodes position by rotating Q and K vectors in 2D subspaces.
            Relative positions fall out naturally from the dot product.
            Currently the dominant scheme in modern LLMs.
            Supports length extrapolation via YaRN / LongRoPE.

        ALiBi (MPT, BLOOM):
            Adds a negative bias proportional to token distance to attention
            scores before softmax. No learned position parameters.

### The Transformer Block (Encoder Style)

    Each Transformer layer applies two sub-layers with residual connections
    and normalisation around each:

        x → [LayerNorm] → [Multi-Head Self-Attention] → + residual
          → [LayerNorm] → [Feed-Forward Network]       → + residual

    Pre-norm vs Post-norm:
        Post-norm (original 2017):
            x = LayerNorm(x + Sublayer(x))   — original paper
        Pre-norm (modern default — GPT-2, LLaMA, Mistral):
            x = x + Sublayer(LayerNorm(x))   — more stable training
            Gradients flow unchanged through the residual path.

    Feed-Forward Network (FFN):
        Applied position-wise (same MLP independently at each token):

            FFN(x) = activation(x · W₁ + b₁) · W₂ + b₂

        W₁: (d_model, d_ff)   — expansion (d_ff ≈ 4 × d_model)
        W₂: (d_ff, d_model)   — contraction

        Activations:
            Original 2017:  ReLU
            BERT / GPT-2:   GELU
            LLaMA / Llama2: SwiGLU — uses gated linear units:
                            FFN(x) = (x W₁ ⊙ SiLU(x W_gate)) W₂
                            Requires three matrices (W₁, W₂, W_gate).
                            Better performance at same parameter count.

### Three Transformer Variants: Encoder, Decoder, Encoder-Decoder

    ENCODER-ONLY (BERT family):
        Stack of encoder blocks with bidirectional self-attention.
        Every token attends to ALL other tokens (past and future).
        Output: contextualised embedding for every input token.

        Pretraining objectives:
            Masked Language Modelling (MLM): randomly mask 15% of tokens,
            predict masked tokens from context. Forces bidirectional understanding.
            Next Sentence Prediction (NSP, BERT): predict if two sentences
            are consecutive (abandoned in RoBERTa as unhelpful).

        Use cases: classification, NER, token labelling, extractive QA,
                   semantic search (dense retrieval), sentence embeddings.
        Models: BERT, RoBERTa, ALBERT, DeBERTa, DistilBERT, ELECTRA.

    DECODER-ONLY (GPT family):
        Stack of decoder blocks with CAUSAL self-attention.
        Each token only attends to past tokens (autoregressive).
        Output: next-token probability distribution.

        Pretraining objective:
            Causal Language Modelling (CLM): predict the next token
            given all previous tokens. Pure unsupervised learning on text.
            Scales remarkably: more data + more compute = better language model.

        Use cases: text generation, code generation, chat, few-shot learning,
                   in-context learning, instruction following (with RLHF).
        Models: GPT-2, GPT-3/4, LLaMA, Mistral, Falcon, Phi, Gemma.

    ENCODER-DECODER (Seq2Seq):
        Encoder processes the input sequence (bidirectional attention).
        Decoder generates the output sequence token by token.
        Cross-attention: decoder queries attend to encoder key-value pairs.

        This is the original 2017 Transformer design for translation.
        Encoder and decoder have separate weights.

        Use cases: translation, summarisation, abstractive QA, dialogue,
                   code generation from natural language (text-to-code).
        Models: T5, BART, mBART, mT5, PEGASUS, MarianMT.

### Choosing the Right Architecture

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Task                          │ Architecture  │ Example Models       │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Text classification / NER     │ Encoder       │ BERT, RoBERTa        │
    │ Sentiment analysis            │ Encoder       │ DistilBERT, ALBERT   │
    │ Semantic search / embeddings  │ Encoder       │ BGE, E5, sentence-t5 │
    │ Extractive QA (span)          │ Encoder       │ BERT, DeBERTa        │
    │ Translation                   │ Enc-Dec       │ MarianMT, mBART, T5  │
    │ Summarisation                 │ Enc-Dec       │ BART, PEGASUS, T5    │
    │ Generative QA / dialogue      │ Decoder       │ LLaMA, Mistral, GPT  │
    │ Code completion               │ Decoder       │ CodeLlama, StarCoder │
    │ Image classification          │ Enc (ViT)     │ ViT, DeiT, CLIP      │
    │ Image captioning              │ Enc-Dec       │ BLIP-2, LLaVA        │
    │ Audio transcription           │ Enc-Dec       │ Whisper              │
    └──────────────────────────────────────────────────────────────────────┘

### Parameter Count and Scaling

    A rough parameter count for a Transformer:
        Embedding table:    vocab_size × d_model         (~30M for BERT-base)
        Per-layer params:   ~12 × d_model²              (attention + FFN)
        Output head:        varies by task

    Representative sizes:
        DistilBERT:    66M params    — 6 layers, d_model=768,  12 heads
        BERT-base:    110M params    — 12 layers, d_model=768,  12 heads
        BERT-large:   340M params    — 24 layers, d_model=1024, 16 heads
        GPT-2 small:  124M params    — 12 layers, d_model=768,  12 heads
        GPT-2 xl:     1.5B params    — 48 layers, d_model=1600, 25 heads
        LLaMA-3 8B:   8B params      — 32 layers, d_model=4096, 32 heads
        LLaMA-3 70B:  70B params     — 80 layers, d_model=8192, 64 heads
        GPT-3:       175B params     — 96 layers, d_model=12288, 96 heads
        GPT-4:       estimated ~1T params (Mixture of Experts, ~8 experts)

    Scaling laws (Chinchilla, 2022):
        Optimal training: N_params × 20 ≈ N_tokens (compute-optimal frontier)
        LLaMA-3 8B trained on 15T tokens — significantly over-trained vs
        compute-optimal, producing a smaller model that runs cheaply at inference.


##### PART 4 — TOKENIZATION: FROM TEXT TO NUMBERS

### Why Tokenization Matters

    Tokenizers convert raw text into integer token IDs that the model
    processes. They are not trivial: the tokenization algorithm determines
    the vocabulary size, how rare words are handled, how languages are
    supported, and the effective context window.

    Character-level: a=1, b=2, ... → extremely long sequences, no prior
    Word-level:      "running"=1234 → huge vocab, unknown words ("UNK")
    Subword (modern): "running" → ["run", "##ning"] → balance length/vocab

### Byte-Pair Encoding (BPE) — GPT family

    Start with individual characters as the vocabulary.
    Iteratively merge the most frequent adjacent pair of tokens.
    Stop when vocab_size is reached.

    Example (simplified):
        Corpus: "low lower lowest slow slower"
        Initial: l o w _ l o w e r _ l o w e s t _ s l o w _ s l o w e r
        Iteration 1: merge "l o" → "lo":   lo w _ lo w e r _ ...
        Iteration 2: merge "lo w" → "low": low _ low e r _ ...
        ...until vocab_size merges are applied.

    GPT-2/3/4, LLaMA, Mistral: BPE on bytes (byte-level BPE).
    Byte-level means every possible byte (0–255) is in the base vocabulary.
    No unknown tokens ever — any input can be encoded.
    Average: ~0.75 tokens per English character ≈ 1 token ≈ 4 characters.

### WordPiece — BERT family

    Similar to BPE but uses a different merge criterion:
    Maximise the language model likelihood of the training data, rather
    than raw pair frequency.

    Subword tokens beyond the first in a word are prefixed with "##":
        "tokenizing" → ["token", "##izing"]

    Special tokens added by default:
        [CLS]: classification token — prepended to every input.
               Its final hidden state is used as the sequence representation.
        [SEP]: separator token — marks end of each sentence.
        [PAD]: padding token — fills short sequences to batch length.
        [MASK]: mask token — replaces tokens during MLM pretraining.
        [UNK]: unknown token — used when a character is unrepresentable.

### SentencePiece — T5, LLaMA family

    Language-agnostic tokenizer that treats the input as a raw stream of
    Unicode characters, with no language-specific pre-tokenization.
    Works directly on whitespace — good for CJK languages.

    Two algorithms: BPE or Unigram Language Model.
    LLaMA uses SentencePiece BPE with a 32,000 token vocabulary.
    T5 uses SentencePiece Unigram with 32,128 tokens.

    Special token: ▁ (U+2581) marks word beginnings (replaces spaces).
        "hello world" → ["▁hello", "▁world"]

### The HuggingFace Tokenizer API

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    Basic encoding:
        ids    = tokenizer.encode("Hello world")   # returns list of ints
        tokens = tokenizer.tokenize("Hello world") # returns list of strings
        result = tokenizer("Hello world")          # returns dict

    The tokenizer return dict (key output type):
        input_ids:       token ID integers        [1, 101, 7592, 2088, 102]
        attention_mask:  1 for real, 0 for pad     [1, 1, 1, 1, 1]
        token_type_ids:  0 for first sentence, 1 for second (BERT only)

    Batch encoding:
        result = tokenizer(
            ["Hello world", "Another sentence"],
            padding        = True,         # pad to longest in batch
            truncation     = True,         # truncate to max_length
            max_length     = 128,
            return_tensors = 'pt',         # or 'tf', 'np'
        )

    Padding strategies:
        padding=True         — pad to longest in batch
        padding='max_length' — pad to max_length
        padding=False        — no padding (default)

    Truncation strategies:
        truncation=True          — truncate to model's max_length
        truncation='only_first'  — truncate first sequence (for pairs)
        truncation='only_second' — truncate second sequence

    Decoding:
        tokenizer.decode([101, 7592, 2088, 102])  # "hello world"
        tokenizer.batch_decode(batch_ids)          # list of strings
        tokenizer.convert_ids_to_tokens([101, ...])# include special tokens as strings

### Fast vs Slow Tokenizers

    Slow tokenizers: Python implementations. Always available.
    Fast tokenizers: Rust implementations via HuggingFace Tokenizers library.
        - 100× faster for large batches.
        - Return offset mappings (character → token index mapping).
        - Required for token-level tasks (NER, QA span extraction).

    AutoTokenizer automatically loads fast if available.
    Check: tokenizer.is_fast → True/False

    Offset mapping (critical for NER / extractive QA):
        result = tokenizer("Hello world", return_offsets_mapping=True)
        result['offset_mapping']  # [(0,0), (0,5), (6,11), (0,0)]
        # Maps each token to its character span in the original string


##### PART 5 — THE HUGGINGFACE ECOSYSTEM

### The Hub: Central Model Registry

    hub.huggingface.co hosts:
        - 750,000+ public models (as of 2024)
        - 150,000+ datasets
        - 300,000+ Spaces (interactive demos)

    Each model card contains:
        - Architecture description
        - Training data and procedure
        - Evaluation results and benchmarks
        - Usage examples
        - License information
        - Safety and bias disclosures

    Naming convention:
        "organisation/model-name"
        "bert-base-uncased"       ← HuggingFace official
        "meta-llama/Llama-3.1-8B-Instruct"
        "mistralai/Mistral-7B-v0.1"
        "sentence-transformers/all-MiniLM-L6-v2"

### The Four Core Libraries

    transformers:      model architectures, tokenizers, training
    datasets:          dataset loading, processing, streaming
    peft:              parameter-efficient fine-tuning (LoRA, prefix tuning)
    accelerate:        training abstraction for multi-GPU, TPU, mixed precision

    tokenizers:        fast Rust tokenizers (dependency of transformers)
    evaluate:          metrics computation (BLEU, ROUGE, accuracy, F1)
    trl:               reinforcement learning from human feedback (RLHF, SFT, DPO)
    optimum:           hardware-optimised inference (ONNX, TensorRT, OpenVINO)

### Auto Classes: The Consistent Entry Point

    The AutoModel* family automatically loads the correct architecture:

        from transformers import (
            AutoConfig,
            AutoTokenizer,
            AutoModel,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoModelForQuestionAnswering,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForMaskedLM,
        )

        # Automatic architecture detection from the Hub
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model     = AutoModelForSequenceClassification.from_pretrained(
                        "bert-base-uncased", num_labels=2)

    from_pretrained key arguments:
        model_name_or_path: Hub model name or local path
        num_labels:          number of output classes (classification)
        ignore_mismatched_sizes: True to load without the classification head
        torch_dtype:        dtype override (torch.float16, torch.bfloat16)
        device_map:         'auto' for automatic multi-GPU placement
        load_in_8bit:       True for bitsandbytes 8-bit quantisation
        load_in_4bit:       True for bitsandbytes 4-bit quantisation

### Task-Specific Model Heads

    The base model produces contextualised embeddings.
    Task heads convert these into task outputs:

        AutoModel:                          raw hidden states
        AutoModelForSequenceClassification: pooled output → Dense(num_labels)
        AutoModelForTokenClassification:    each token hidden → Dense(num_labels)
        AutoModelForQuestionAnswering:      each token → start/end logits
        AutoModelForCausalLM:               each token → vocab-size logits
        AutoModelForSeq2SeqLM:              encoder-decoder, generates sequences
        AutoModelForMaskedLM:               each token → vocab-size logits (for MLM)
        AutoModelForMultipleChoice:         each choice pooled → scalar score

### The Pipeline API: Zero-Code Inference

    pipeline() is the highest-level abstraction in HuggingFace.
    It handles tokenisation, batching, model inference, and post-processing.

        from transformers import pipeline

        # Text classification
        clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        clf("I loved this movie!")   # [{'label': 'POSITIVE', 'score': 0.9998}]

        # Text generation
        gen = pipeline("text-generation", model="gpt2")
        gen("The future of AI is", max_new_tokens=50)

        # NER (token classification)
        ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        ner("My name is Sarah and I live in London")

        # Question answering
        qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
        qa(question="Who wrote Hamlet?", context="Hamlet was written by Shakespeare.")

        # Summarisation
        summ = pipeline("summarization", model="facebook/bart-large-cnn")
        summ(long_article, max_length=130, min_length=30)

        # Translation
        tl = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        tl("Hello world")

        # Zero-shot classification
        zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        zsc("I love programming", candidate_labels=["sport", "tech", "arts"])

        # Conversational (chat)
        chat = pipeline("conversational", model="microsoft/DialoGPT-medium")

    Pipeline batch processing:
        clf(["sentence 1", "sentence 2", ...], batch_size=32)

    Pipeline with GPU:
        clf = pipeline("text-classification", device=0)       # GPU 0
        clf = pipeline("text-classification", device_map="auto")  # auto placement


##### PART 6 — FINE-TUNING: ADAPTING PRETRAINED MODELS

### Why Fine-Tune?

    Pretraining on vast corpora gives models broad language understanding.
    Fine-tuning adapts this to a specific task or domain with far less data.

    The fundamental property: TRANSFER LEARNING.
    A model trained on 1 trillion tokens of text has learned:
        - Grammar and syntax across many languages
        - World knowledge (facts, entities, relationships)
        - Reasoning patterns and common sense
        - Writing styles and conventions

    Fine-tuning takes ~100–10,000 labelled examples to achieve excellent
    performance on specific tasks. Without pretraining, the same task
    would require millions of examples.

### Three Fine-Tuning Regimes

    1. FULL FINE-TUNING:
        Update ALL model weights on the task-specific dataset.
        Highest task performance. Requires full GPU memory for gradients.
        Risk: catastrophic forgetting of pretraining knowledge.
        Practical for models up to ~7B parameters with sufficient GPU.
        Models: BERT fine-tuning on classification, RoBERTa on QA.

    2. FEATURE EXTRACTION (head-only):
        Freeze ALL pretrained weights. Train only the task head (1–2 layers).
        Fastest. Works when pretraining distribution matches task distribution.
        Limited adaptation — the frozen representations may not be ideal.
        Use when: data is scarce (<1000 examples), compute is very limited.

    3. PARAMETER-EFFICIENT FINE-TUNING (PEFT):
        Freeze most weights. Add small trainable adapter modules.
        Trainable parameters: 0.1–1% of total.
        Performance approaches full fine-tuning. Much less memory.
        Multiple tasks can be served from one base model + multiple adapters.
        This is the current standard for fine-tuning LLMs.

### The Trainer API

    HuggingFace Trainer handles the full training loop:
        - Gradient accumulation
        - Mixed precision (fp16/bf16)
        - Gradient clipping
        - Learning rate scheduling
        - Evaluation on a schedule
        - Checkpointing
        - Distributed training (via Accelerate)
        - Logging to TensorBoard, WandB, etc.

    TrainingArguments — the configuration:

        from transformers import TrainingArguments

        args = TrainingArguments(
            output_dir             = './results',
            num_train_epochs       = 3,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size  = 32,
            learning_rate          = 2e-5,
            weight_decay           = 0.01,
            warmup_ratio           = 0.1,          # 10% warmup
            lr_scheduler_type      = 'cosine',
            evaluation_strategy    = 'epoch',
            save_strategy          = 'epoch',
            load_best_model_at_end = True,
            metric_for_best_model  = 'f1',
            fp16                   = True,          # or bf16=True
            gradient_accumulation_steps = 4,
            logging_steps          = 50,
            report_to              = 'tensorboard',
        )

    Trainer usage:

        from transformers import Trainer

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(-1)
            return {'accuracy': (preds == labels).mean()}

        trainer = Trainer(
            model           = model,
            args            = args,
            train_dataset   = train_ds,
            eval_dataset    = val_ds,
            tokenizer       = tokenizer,
            compute_metrics = compute_metrics,
            callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()
        trainer.evaluate()
        trainer.save_model('./final_model')

### Data Collators — Dynamic Padding

    DataCollatorWithPadding: pads a batch to its longest element.
    This is more efficient than padding all samples to max_length upfront.

        from transformers import DataCollatorWithPadding
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # Pass to Trainer as data_collator=collator

    DataCollatorForLanguageModeling: randomly masks tokens for MLM training.
    DataCollatorForSeq2Seq: handles encoder-decoder padding.
    DataCollatorForTokenClassification: aligns word labels with subword tokens.


##### PART 7 — PARAMETER-EFFICIENT FINE-TUNING (PEFT) AND LoRA

### Why PEFT?

    Fine-tuning a 7B parameter model with full gradients requires:
        - ~28 GB for weights (fp32) or ~14 GB (fp16)
        - ~56 GB for gradients + optimiser states (Adam: 3× weights)
        - Total: ~84–168 GB — requires multiple high-end GPUs

    PEFT methods reduce trainable parameters by 100–1000×:
        LoRA on LLaMA-3 8B: 0.1% trainable = 8M trainable parameters
        Fits in 10–16 GB GPU with 4-bit quantised base model

### LoRA: Low-Rank Adaptation (Hu et al., 2021)

    The key insight: weight updates during fine-tuning have low intrinsic rank.
    The pre-trained weight matrix W₀ ∈ ℝ^(d×k) doesn't need to change by
    a full-rank matrix. The DELTA W can be approximated as a low-rank product:

        W = W₀ + ΔW = W₀ + B · A

        W₀:  original pretrained weights (frozen, d × k)
        A:   down-projection (r × k, small) — initialised from random Gaussian
        B:   up-projection   (d × r, small) — initialised as zeros
        r:   rank (hyperparameter, typically 4–64)

    During training: only A and B are updated (W₀ is frozen).
    During inference: W = W₀ + BA — merge into a single matrix, zero overhead.

    Why A initialised random and B as zeros?
        At training start, BA = 0, so ΔW = 0.
        The LoRA module starts as an identity pass-through.
        Training then nudges it toward the optimal delta.

    LoRA scaling factor (α):
        The effective update is (α/r) × BA.
        α is usually set to r (so the scale = 1) or to a fixed value like 16.
        Allows separate control of rank and update scale.

    Where to apply LoRA:
        Original paper: query and value projection matrices in attention.
        Modern practice: all attention projections (Q, K, V, O) + FFN layers.
        HuggingFace PEFT: target_modules specifies which layers get LoRA.

        target_modules = ['q_proj', 'v_proj']            # minimal (original)
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']  # attention
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'gate_proj', 'up_proj', 'down_proj']    # full

### LoRA in Practice with PEFT

        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r                = 16,              # rank of the update matrices
            lora_alpha       = 32,              # scaling factor (α)
            target_modules   = ['q_proj', 'v_proj'],  # which layers get LoRA
            lora_dropout     = 0.05,
            bias             = 'none',           # 'none', 'all', 'lora_only'
            task_type        = 'CAUSAL_LM',      # or 'SEQ_CLS', 'SEQ_2_SEQ_LM'
        )

        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        # trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622

    Merging LoRA weights back into base model (inference efficiency):
        merged_model = model.merge_and_unload()
        # Now a regular model with LoRA baked in — zero overhead at inference

    Saving and loading LoRA adapters:
        model.save_pretrained('./lora_adapter')   # saves adapter_config.json + weights
        # Loading:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, './lora_adapter')

### Other PEFT Methods

    QLoRA (Dettmers et al., 2023):
        LoRA + 4-bit NormalFloat quantisation of the base model.
        Fine-tune 65B models on a single A100 GPU.
        Use bitsandbytes 4-bit quantisation:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, load_in_4bit=True, bnb_4bit_quant_type='nf4'
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)

    Prefix Tuning:
        Prepend learned "soft prompt" tokens to every attention layer.
        No modification to attention weights.

    Prompt Tuning:
        Simpler version of prefix tuning — learned tokens only at the input.
        Works well for very large models (GPT-3 scale).

    IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations):
        Learn small vectors that rescale keys, values, and FFN activations.
        Even fewer parameters than LoRA — 0.01% trainable.


##### PART 8 — TEXT GENERATION AND QUANTIZATION FOR DEPLOYMENT

### Text Generation Strategies

    Autoregressive models generate one token at a time, sampling from:
        P(x_t | x_1, ..., x_{t-1})

    The decoding strategy dramatically affects output quality:

    Greedy decoding:
        Always pick the highest-probability token.
        Fast, deterministic, but tends toward repetitive/generic text.
        model.generate(input_ids, do_sample=False)

    Beam search:
        Maintain k (beam width) candidate sequences simultaneously.
        Pick the sequence with highest overall probability at the end.
        Less repetition than greedy. Still deterministic. k=4–8 is typical.
        model.generate(input_ids, num_beams=4, early_stopping=True)

    Sampling (do_sample=True):
        Randomly sample from the probability distribution.
        More diverse/creative outputs. Can be incoherent.

    Temperature scaling:
        Divide logits by temperature T before softmax.
        T < 1: sharper distribution → more focused, less random.
        T > 1: flatter distribution → more diverse, more random.
        T = 1: unchanged distribution.
        model.generate(input_ids, do_sample=True, temperature=0.7)

    Top-k sampling:
        Sample only from the k highest-probability tokens (e.g. k=50).
        Eliminates long tail of low-probability tokens.
        model.generate(input_ids, do_sample=True, top_k=50)

    Top-p (nucleus) sampling:
        Sample from the smallest set of tokens whose cumulative
        probability ≥ p (e.g. p=0.9). Dynamic vocabulary size.
        model.generate(input_ids, do_sample=True, top_p=0.9)

    Repetition penalty:
        Penalise tokens that have already appeared in the sequence.
        model.generate(input_ids, repetition_penalty=1.3)

    Typical production combination:
        model.generate(
            input_ids,
            do_sample          = True,
            temperature        = 0.7,
            top_p              = 0.9,
            top_k              = 50,
            repetition_penalty = 1.1,
            max_new_tokens     = 512,
        )

### Quantization for Efficient Inference

    Full precision (fp32): 4 bytes per parameter
    fp16/bf16:             2 bytes per parameter → 2× smaller
    INT8:                  1 byte per parameter  → 4× smaller
    INT4 (NF4):            0.5 bytes per parameter → 8× smaller

    BitsAndBytes (bitsandbytes library):
        8-bit:  model = AutoModelForCausalLM.from_pretrained(id, load_in_8bit=True)
        4-bit:  model = AutoModelForCausalLM.from_pretrained(
                            id, load_in_4bit=True,
                            bnb_4bit_quant_type='nf4',           # NormalFloat4
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,      # double quantisation
                        )

    GPTQ (post-training quantization for inference):
        Calibrate on a small dataset. More accurate than naive rounding.
        4-bit GPTQ models available directly on the Hub:
            "TheBloke/Llama-2-7B-GPTQ"

    AWQ (Activation-aware Weight Quantization):
        Identifies and preserves important channels during quantisation.
        Better accuracy than GPTQ at 4-bit.
        "TheBloke/Llama-2-7B-AWQ"

    GGUF (llama.cpp format):
        CPU-friendly quantisation for local inference.
        Runs on consumer hardware (M2 Mac, gaming laptop).
        "TheBloke/Llama-2-7B-GGUF"

### KV-Cache and Efficient Inference

    During autoregressive generation, attention keys and values for all
    previous tokens are recomputed every step — O(n²) total cost.

    KV-cache: store computed K, V tensors for all past tokens.
    Each new token only needs to compute K, V for itself.
    Memory cost: 2 × n_layers × 2 × n_kv_heads × d_head × seq_len × dtype

    For LLaMA-3 8B at 4096 tokens:
        2 × 32 × 2 × 8 × 128 × 4096 × 2 bytes (fp16) = 2 GB

    Grouped Query Attention (GQA) reduces this by n_query_heads / n_kv_heads.
    LLaMA-3 8B: 32 query heads, 8 KV heads → 4× smaller KV-cache.

    In HuggingFace:
        outputs = model.generate(input_ids, use_cache=True)   # default True

### Deployment Decision Guide

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Scenario                  │ Approach                                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Local inference (<7B)     │ Transformers + bfloat16                  │
    │ Local inference (7B+)     │ 4-bit BnB or GGUF + llama.cpp            │
    │ Production API server     │ vLLM or TGI (Text Generation Inference)  │
    │ BERT classification       │ ONNX export → ONNX Runtime               │
    │ Mobile / edge             │ ONNX → CoreML / TFLite via Optimum       │
    │ Fine-tune + serve small   │ Full fine-tune + ONNX                    │
    │ Fine-tune + serve large   │ LoRA → merge → GPTQ quantise → vLLM      │
    │ Many task variants        │ Base model + LoRA adapters (swap at rt)  │
    └──────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Tokenizers & Pipelines — Text Processing and Zero-Shot Inference": {
        "description": (
            "Deep dive into HuggingFace tokenizers and the Pipeline API. "
            "Tokenizer internals: BPE vs WordPiece, special tokens, vocabulary. "
            "Encoding: input_ids, attention_mask, token_type_ids. "
            "Batch encoding with padding and truncation strategies. "
            "Offset mapping for NER span alignment. "
            "Fast tokenizer performance benchmark. "
            "Pipeline API: text-classification, NER, QA, summarisation, "
            "zero-shot classification, text generation with sampling strategies. "
            "Batch inference with pipelines. "
            "Manual inference loop vs pipeline comparison."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, logging as hf_logging
    )
    import torch
    hf_logging.set_verbosity_error()
    print(f"  transformers: installed | torch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'transformers', 'torch', '--quiet'], check=True)
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, logging as hf_logging
    )
    import torch
    hf_logging.set_verbosity_error()

print("=" * 65)
print("  TOKENIZERS & PIPELINES — TEXT PROCESSING AND INFERENCE")
print("=" * 65)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Tokenizer internals
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Tokenizer internals: vocab, IDs, special tokens")
print("━" * 65)
print()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

print(f"  Model: bert-base-uncased")
print(f"  Vocab size:      {tokenizer.vocab_size:,}")
print(f"  Max length:      {tokenizer.model_max_length}")
print(f"  Fast tokenizer:  {tokenizer.is_fast}")
print(f"  Special tokens:  {tokenizer.all_special_tokens[:8]}")
print()

# Show tokenisation of various inputs
test_sentences = [
    "Hello, world!",
    "The tokenizer splits words into subwords.",
    "HuggingFace makes NLP accessible to everyone.",
    "Tokenization: running → [run, ##ning]",
    "Unknown: asjdhfkajshdfkjasdhf",   # rare chars → subword fragments
]

print(f"  Tokenisation examples:")
print(f"  {'Input':<40} {'Tokens'}")
print(f"  {'─'*65}")
for sent in test_sentences:
    tokens = tokenizer.tokenize(sent)
    print(f"  {sent[:38]:<40} {tokens[:8]}{'...' if len(tokens)>8 else ''}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Encoding outputs — input_ids, attention_mask, token_type_ids
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Encoding: input_ids, attention_mask, offset_mapping")
print("━" * 65)
print()

text  = "The cat sat on the mat."
enc   = tokenizer(
    text,
    return_offsets_mapping=True,    # fast tokenizers only
    return_token_type_ids=True,
)
tokens_decoded = tokenizer.convert_ids_to_tokens(enc['input_ids'])

print(f"  Text: '{text}'")
print(f"  {'Token':<15} {'ID':>6} {'Offset':>12} {'Attn':>6}")
print(f"  {'─'*42}")
for tok, tid, offset, attn in zip(
        tokens_decoded,
        enc['input_ids'],
        enc['offset_mapping'],
        enc['attention_mask']):
    print(f"  {tok:<15} {tid:>6} {str(offset):>12} {attn:>6}")
print()

# Sentence pair encoding (BERT NSP task format)
q = "Where does the cat sit?"
ctx = "The cat sits on the mat in the garden."
pair_enc = tokenizer(q, ctx, return_token_type_ids=True)
print(f"  Sentence pair encoding:")
print(f"    Q:   '{q}'")
print(f"    Ctx: '{ctx}'")
pair_tokens = tokenizer.convert_ids_to_tokens(pair_enc['input_ids'])
print(f"    Tokens: {pair_tokens}")
print(f"    token_type_ids: {pair_enc['token_type_ids']}")
print(f"    (0=first sentence/question, 1=second sentence/context)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Batch encoding with padding and truncation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Batch encoding: padding, truncation, return_tensors")
print("━" * 65)
print()

sentences = [
    "Short sentence.",
    "This is a medium length sentence with more words.",
    "This sentence is much longer and contains many more tokens than the previous ones, demonstrating how padding works in batched tokenization.",
]

# Dynamic padding — pad to longest in batch
batch = tokenizer(
    sentences,
    padding        = True,          # pad to longest in batch
    truncation     = True,          # truncate to model max length
    max_length     = 64,
    return_tensors = 'pt',          # PyTorch tensors
)

print(f"  Batch of {len(sentences)} sentences (max_length=64):")
print(f"  input_ids shape:      {batch['input_ids'].shape}   (batch, max_seq_len)")
print(f"  attention_mask shape: {batch['attention_mask'].shape}")
print()
print(f"  Actual lengths (non-padding tokens):")
for i, sent in enumerate(sentences):
    n_real = batch['attention_mask'][i].sum().item()
    n_pad  = batch['input_ids'].shape[1] - n_real
    print(f"    [{i}] {sent[:35]:<35} → {n_real:>3} real tokens, {n_pad:>2} pad")
print()

# Tokenizer performance benchmark
print(f"  Fast tokenizer benchmark (1000 sentences):")
test_corpus = ["The transformer architecture has revolutionised NLP."] * 1000
t0 = time.perf_counter()
_ = tokenizer(test_corpus, padding=True, truncation=True, max_length=64)
t_fast = (time.perf_counter() - t0) * 1000
print(f"    Fast (Rust):  {t_fast:.0f}ms for 1000 sentences")
print(f"    Per sentence: {t_fast/1000:.3f}ms")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Pipeline API — text classification
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Pipeline API: text classification")
print("━" * 65)
print()

clf_pipeline = pipeline(
    "text-classification",
    model     = "distilbert-base-uncased-finetuned-sst-2-english",
    device    = 0 if device == 'cuda' else -1,
    top_k     = None,           # return all class scores
)

test_texts = [
    "I absolutely loved this film — a masterpiece!",
    "The movie was terrible and a complete waste of time.",
    "It was neither particularly good nor bad, just mediocre.",
    "One of the best performances I have ever seen.",
]

print(f"  Model: distilbert-base-uncased-finetuned-sst-2-english")
print(f"  {'Text':<45} {'Label':<12} {'Score':>8}")
print(f"  {'─'*68}")
for text, result in zip(test_texts, clf_pipeline(test_texts)):
    # result is a list of dicts (top_k=None)
    best = max(result, key=lambda x: x['score'])
    print(f"  {text[:43]:<45} {best['label']:<12} {best['score']:>8.4f}")
print()

# Batch processing benchmark
print(f"  Batch inference benchmark (batch_size=16):")
batch_texts = test_texts * 25    # 100 samples
t0   = time.perf_counter()
outs = clf_pipeline(batch_texts, batch_size=16)
t_ms = (time.perf_counter() - t0) * 1000
print(f"    Classified {len(batch_texts)} texts in {t_ms:.0f}ms")
print(f"    Throughput: {len(batch_texts) / (t_ms/1000):.0f} texts/second")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Named Entity Recognition pipeline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — NER Pipeline with aggregation strategies")
print("━" * 65)
print()

ner_pipeline = pipeline(
    "ner",
    model                = "dslim/bert-base-NER",
    aggregation_strategy = "simple",   # merge B- and I- tokens
    device               = 0 if device == 'cuda' else -1,
)

ner_text = "Elon Musk founded SpaceX in 2002 and Tesla in 2003. He was born in Pretoria, South Africa."
entities = ner_pipeline(ner_text)

print(f"  Text: '{ner_text}'")
print()
print(f"  {'Entity':<20} {'Type':<8} {'Start':>6} {'End':>5} {'Score':>8}")
print(f"  {'─'*52}")
for ent in entities:
    print(f"  {ent['word']:<20} {ent['entity_group']:<8} "
          f"{ent['start']:>6} {ent['end']:>5} {ent['score']:>8.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Question Answering pipeline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Question Answering pipeline (extractive)")
print("━" * 65)
print()

qa_pipeline = pipeline(
    "question-answering",
    model  = "deepset/roberta-base-squad2",
    device = 0 if device == 'cuda' else -1,
)

context = """
The Transformer architecture was introduced in the seminal 2017 paper
'Attention Is All You Need' by Vaswani et al. at Google Brain. It replaced
recurrent neural networks with a self-attention mechanism, enabling
parallelisation during training. BERT was then developed by Google in 2018
and uses the encoder portion of the Transformer, pretrained with masked
language modelling. GPT models by OpenAI use only the decoder with
causal attention for autoregressive text generation.
"""

questions = [
    "Who introduced the Transformer architecture?",
    "What year was the Transformer paper published?",
    "What does BERT use for pretraining?",
    "What part of the Transformer does GPT use?",
]

print(f"  Context: [paragraph about Transformers and BERT/GPT history]")
print()
for q in questions:
    answer = qa_pipeline(question=q, context=context)
    print(f"  Q: {q}")
    print(f"  A: {answer['answer']}  (score: {answer['score']:.4f})")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 7: Zero-shot classification
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 7 — Zero-shot classification (no task-specific training)")
print("━" * 65)
print()

zsc = pipeline(
    "zero-shot-classification",
    model  = "facebook/bart-large-mnli",
    device = 0 if device == 'cuda' else -1,
)

zsc_samples = [
    ("I love playing football and watching the Champions League.",
     ["sports", "technology", "politics", "cooking"]),
    ("The central bank raised interest rates by 25 basis points.",
     ["economics", "sports", "medicine", "arts"]),
    ("She trained her neural network on a cluster of 64 GPUs.",
     ["machine learning", "cooking", "music", "travel"]),
]

print(f"  Multi-label zero-shot classification (no fine-tuning required):")
print()
for text, labels in zsc_samples:
    result = zsc(text, candidate_labels=labels, multi_label=False)
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    print(f"  Text:   '{text[:60]}'")
    print(f"  Labels: {list(zip(result['labels'], [f'{s:.3f}' for s in result['scores']]))}")
    print(f"  → Predicted: '{top_label}' ({top_score:.4f})")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 8: Manual inference with model.forward()
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 8 — Manual inference: beyond the Pipeline API")
print("━" * 65)
print()

manual_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
manual_model     = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english'
)
manual_model.eval()

import torch.nn.functional as F

texts = ["This is great!", "This is terrible!", "This is okay."]
inputs = manual_tokenizer(texts, padding=True, truncation=True,
                           return_tensors='pt', max_length=128)

with torch.no_grad():
    outputs   = manual_model(**inputs)     # ** unpacks dict as keyword args
    logits    = outputs.logits             # (batch, num_labels)
    probs     = F.softmax(logits, dim=-1)  # (batch, num_labels)
    preds     = logits.argmax(dim=-1)      # (batch,)

label_names = manual_model.config.id2label    # {0: 'NEGATIVE', 1: 'POSITIVE'}
print(f"  Manual forward pass (batch of {len(texts)}):")
print(f"  {'Text':<25} {'Pred':<12} {'NEGATIVE':>10} {'POSITIVE':>10}")
print(f"  {'─'*60}")
for text, pred, p in zip(texts, preds, probs):
    label = label_names[pred.item()]
    print(f"  {text:<25} {label:<12} {p[0].item():>10.4f} {p[1].item():>10.4f}")
print()
print(f"  model.config.id2label: {label_names}")
print(f"  outputs.logits shape:  {logits.shape}  (batch, num_labels)")
print(f"  Hidden states: outputs.hidden_states — available with output_hidden_states=True")
print(f"  Attentions:    outputs.attentions    — available with output_attentions=True")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Fine-Tuning with Trainer API — Classification, NER & Seq2Seq": {
        "description": (
            "Full fine-tuning pipeline with the HuggingFace Trainer API. "
            "Dataset loading with datasets library. "
            "Tokenization with map() and dynamic padding collator. "
            "TrainingArguments: all key hyperparameters, lr schedules, fp16. "
            "compute_metrics with sklearn for accuracy, F1, precision, recall. "
            "Training, evaluation, and save_model. "
            "Resuming from checkpoint. "
            "Custom Trainer subclass: overriding compute_loss for focal loss. "
            "EarlyStoppingCallback and custom training callbacks. "
            "Push to Hub workflow."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
        EarlyStoppingCallback, TrainerCallback, logging as hf_logging,
    )
    from datasets import Dataset
    import evaluate
    hf_logging.set_verbosity_error()
    print(f"  transformers + datasets: installed | torch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'transformers', 'datasets', 'evaluate',
                    'scikit-learn', 'torch', '--quiet'], check=True)
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
        EarlyStoppingCallback, TrainerCallback, logging as hf_logging,
    )
    from datasets import Dataset
    import evaluate
    hf_logging.set_verbosity_error()

print("=" * 65)
print("  FINE-TUNING WITH TRAINER API — CLASSIFICATION PIPELINE")
print("=" * 65)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  PyTorch: {torch.__version__} | Device: {device}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Synthetic dataset construction
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Dataset construction with HuggingFace datasets")
print("━" * 65)
print()

np.random.seed(42)

# Synthetic 4-class text classification dataset
TEMPLATES = {
    0: ["The stock market {v} today as investors {v2}.",
        "Earnings reports show {adj} quarterly results for {company}.",
        "The central bank {v} interest rates amid inflation concerns.",
        "Investors are {adj} about the latest economic data."],
    1: ["Scientists {v} a new {adj} discovery in quantum computing.",
        "The research team published findings on {adj} gene editing.",
        "A breakthrough in {adj} materials science was announced.",
        "Researchers at {uni} demonstrated {adj} results in AI."],
    2: ["The team {v} the championship with a {adj} performance.",
        "{player} scored the winning goal in {adj} fashion.",
        "The {adj} match ended in a draw after {n} minutes.",
        "The coach praised the {adj} effort of the players."],
    3: ["The {adj} film received {n} Oscar nominations.",
        "The concert was a {adj} event attended by thousands.",
        "Critics praised the {adj} performance in the new show.",
        "The album debuted at number one with {adj} reviews."],
}
FILL = dict(v=["surged","fell","rose","declined"], v2=["sold","bought","hedged"],
            adj=["strong","weak","record","unprecedented","remarkable","stunning"],
            company=["Apple","Tesla","Google"], uni=["MIT","Stanford","Oxford"],
            player=["Messi","Ronaldo","Mbappé"], n=["three","five","seven","ninety"])

def fill_template(tmpl):
    import re
    for k, vals in FILL.items():
        tmpl = re.sub(r'\{' + k + r'\}', np.random.choice(vals), tmpl)
    return tmpl

label_names = {0: "finance", 1: "science", 2: "sports", 3: "entertainment"}
all_texts, all_labels = [], []
for label_id, templates in TEMPLATES.items():
    for _ in range(200):
        tmpl  = np.random.choice(templates)
        all_texts.append(fill_template(tmpl))
        all_labels.append(label_id)

# Shuffle
idx = np.random.permutation(len(all_texts))
all_texts  = [all_texts[i]  for i in idx]
all_labels = [all_labels[i] for i in idx]

# Split
n_train, n_val = 600, 200
raw = {
    'train': Dataset.from_dict({'text': all_texts[:n_train],   'label': all_labels[:n_train]}),
    'val':   Dataset.from_dict({'text': all_texts[n_train:n_train+n_val], 'label': all_labels[n_train:n_train+n_val]}),
}

print(f"  Created synthetic 4-class dataset:")
print(f"    Train: {len(raw['train'])} samples  |  Val: {len(raw['val'])} samples")
print(f"    Classes: {label_names}")
print(f"  Sample (train[0]):")
print(f"    text:  '{raw['train'][0]['text']}'")
print(f"    label:  {raw['train'][0]['label']} ({label_names[raw['train'][0]['label']]})")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Tokenize dataset with .map()
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Tokenizing with Dataset.map() and DataCollatorWithPadding")
print("━" * 65)
print()

MODEL_NAME = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples['text'],
        truncation = True,
        max_length = 128,
        # NO padding here — DataCollatorWithPadding handles it dynamically
    )

tokenized = {
    split: ds.map(tokenize_fn, batched=True, remove_columns=['text'])
    for split, ds in raw.items()
}
# Set PyTorch format (returns tensors in DataLoader)
for split in tokenized:
    tokenized[split].set_format('torch')

print(f"  Tokenized columns: {tokenized['train'].column_names}")
print(f"  Sample token lengths (first 5 train examples):")
for i in range(5):
    n = tokenized['train'][i]['input_ids'].shape[0]
    print(f"    [{i}] {n} tokens: '{raw['train'][i]['text'][:50]}...'")
print()

# DataCollatorWithPadding: dynamic padding to longest in each batch
collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(f"  DataCollatorWithPadding: pads each batch to its longest sequence")
print(f"  (more efficient than padding all samples to max_length=128)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: compute_metrics function
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — compute_metrics with evaluate library")
print("━" * 65)
print()

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)['accuracy']
    f1  = f1_metric.compute(predictions=preds, references=labels,
                             average='macro')['f1']
    # Per-class F1
    f1_per_class = f1_metric.compute(predictions=preds, references=labels,
                                      average=None)['f1']
    metrics = {'accuracy': acc, 'f1_macro': f1}
    for cls_id, cls_name in label_names.items():
        metrics[f'f1_{cls_name}'] = f1_per_class[cls_id]
    return metrics

print(f"  compute_metrics returns: accuracy + macro F1 + per-class F1")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Custom Callback for monitoring
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Custom TrainerCallback")
print("━" * 65)
print()

class TrainingMonitorCallback(TrainerCallback):
    """
    Custom callback that logs training metrics at the end of each epoch
    and detects when val_loss stops improving.
    """
    def __init__(self):
        self.epoch_logs  = []
        self.best_metric = -float('inf')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            ep_info = {
                'epoch':    state.epoch,
                'val_acc':  metrics.get('eval_accuracy', 0),
                'val_f1':   metrics.get('eval_f1_macro', 0),
                'val_loss': metrics.get('eval_loss', 0),
            }
            self.epoch_logs.append(ep_info)
            improved = ep_info['val_f1'] > self.best_metric
            if improved:
                self.best_metric = ep_info['val_f1']
            symbol = "✓" if improved else " "
            print(f"  [{symbol}] Epoch {ep_info['epoch']:.0f}: "
                  f"val_loss={ep_info['val_loss']:.4f}  "
                  f"val_acc={ep_info['val_acc']:.4f}  "
                  f"val_f1={ep_info['val_f1']:.4f}")

monitor_cb = TrainingMonitorCallback()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: TrainingArguments and Trainer
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — TrainingArguments + Trainer: full training run")
print("━" * 65)
print()

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = len(label_names),
    id2label   = label_names,
    label2id   = {v: k for k, v in label_names.items()},
)

print(f"  Model: {MODEL_NAME}")
print(f"  Total params:     {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print()

with tempfile.TemporaryDirectory() as tmp:
    training_args = TrainingArguments(
        output_dir                  = tmp,
        num_train_epochs            = 5,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 64,
        learning_rate               = 2e-5,
        weight_decay                = 0.01,
        warmup_ratio                = 0.1,
        lr_scheduler_type           = 'cosine',
        evaluation_strategy         = 'epoch',
        save_strategy               = 'epoch',
        load_best_model_at_end      = True,
        metric_for_best_model       = 'f1_macro',
        greater_is_better           = True,
        fp16                        = (device == 'cuda'),
        logging_steps               = 999999,    # suppress step logs
        save_total_limit            = 2,
        report_to                   = 'none',
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized['train'],
        eval_dataset    = tokenized['val'],
        tokenizer       = tokenizer,
        data_collator   = collator,
        compute_metrics = compute_metrics,
        callbacks       = [
            monitor_cb,
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    t0 = time.perf_counter()
    trainer.train()
    t_fit = time.perf_counter() - t0

    print()
    print(f"  Training complete in {t_fit:.1f}s")
    print(f"  Best val F1: {monitor_cb.best_metric:.4f}")
    print()

    # Final evaluation
    final_metrics = trainer.evaluate()
    print(f"  Final evaluation metrics:")
    for k, v in final_metrics.items():
        if not k.startswith('eval_runtime'):
            print(f"    {k}: {v:.4f}")
    print()

    # Save and reload
    model_save_path = os.path.join(tmp, 'final_model')
    trainer.save_model(model_save_path)
    print(f"  Model saved to: {model_save_path}")

    # Reload and verify
    loaded_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
    loaded_model.eval()
    loaded_tok   = AutoTokenizer.from_pretrained(model_save_path)

    test_inputs = loaded_tok(["The stock market crashed today.",
                               "Scientists discovered water on Mars."],
                               return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        loaded_out = loaded_model(**test_inputs)
    loaded_preds = loaded_out.logits.argmax(-1).tolist()
    for text, pred in zip(["Stock: ...", "Science: ..."], loaded_preds):
        print(f"  {text:15} → predicted class: {label_names[pred]}")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Custom Trainer subclass — focal loss for imbalance
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 6 — Custom Trainer: overriding compute_loss (Focal Loss)")
print("━" * 65)
print()

import torch.nn.functional as F

class FocalLossTrainer(Trainer):
    """
    Trainer subclass that overrides compute_loss to use Focal Loss.
    Useful for class-imbalanced classification tasks.
    """
    def __init__(self, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits  = outputs.logits

        # Focal loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
        ce_loss    = F.cross_entropy(logits, labels, reduction='none')
        pt         = torch.exp(-ce_loss)                         # p_t
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return (focal_loss, outputs) if return_outputs else focal_loss

print(f"  FocalLossTrainer: overrides compute_loss with Focal Loss (gamma=2.0)")
print(f"  Pattern: inputs.pop('labels') + custom loss + return focal_loss")
print(f"  Use cases: class imbalance, hard-example mining, medical NLP")
print()
print(f"  Key TrainingArguments for production fine-tuning:")
print(f"  ┌─────────────────────────────────────────────────────────────────┐")
print(f"  │ Parameter                │ Typical value     │ Effect           │")
print(f"  ├─────────────────────────────────────────────────────────────────┤")
print(f"  │ learning_rate            │ 1e-5 to 5e-5      │ #1 hyperparameter│")
print(f"  │ num_train_epochs         │ 3–5               │ more = risk overfit│")
print(f"  │ warmup_ratio             │ 0.06–0.1          │ stable start     │")
print(f"  │ lr_scheduler_type        │ linear or cosine  │ final lr = 0     │")
print(f"  │ weight_decay             │ 0.01–0.1          │ L2 regularisation│")
print(f"  │ fp16 / bf16              │ True on GPU       │ 2× speed, <2× mem│")
print(f"  │ gradient_accumulation    │ 4–16              │ simulate large bs│")
print(f"  │ per_device_train_bs      │ 8–32              │ GPU memory limit │")
print(f"  └─────────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · PEFT & LoRA — Parameter-Efficient Fine-Tuning of Large Models": {
        "description": (
            "Complete PEFT pipeline: LoRA configuration, training, merging. "
            "LoRA math: low-rank decomposition ΔW = BA visualised. "
            "print_trainable_parameters: see 0.06% trainable. "
            "target_modules: attention only vs full model. "
            "Comparing LoRA ranks: r=4 vs r=16 vs r=64 quality/size. "
            "merge_and_unload: bake LoRA into base model for deployment. "
            "save_pretrained / PeftModel.from_pretrained for adapter I/O. "
            "QLoRA pattern with 4-bit quantisation (bitsandbytes). "
            "Custom generation with fine-tuned causal LM. "
            "PEFT methods decision guide: LoRA vs prefix vs prompt tuning."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForCausalLM, logging as hf_logging,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )
    from peft import (
        LoraConfig, get_peft_model, TaskType,
        PeftModel, PeftConfig,
    )
    from datasets import Dataset
    hf_logging.set_verbosity_error()
    print(f"  transformers + peft: installed | torch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'transformers', 'peft', 'datasets',
                    'torch', '--quiet'], check=True)
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForCausalLM, logging as hf_logging,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )
    from peft import (
        LoraConfig, get_peft_model, TaskType,
        PeftModel, PeftConfig,
    )
    from datasets import Dataset
    hf_logging.set_verbosity_error()

print("=" * 65)
print("  PEFT & LoRA — PARAMETER-EFFICIENT FINE-TUNING")
print("=" * 65)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: LoRA mathematics visualised
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — LoRA math: low-rank decomposition ΔW = B·A")
print("━" * 65)
print()

# Simulate LoRA weight decomposition
d_model, rank = 768, 16
torch.manual_seed(42)

W0 = torch.randn(d_model, d_model)    # pretrained weight (frozen)
A  = torch.randn(rank, d_model) * 0.01  # down-project  (init: random Gaussian)
B  = torch.zeros(d_model, rank)          # up-project    (init: zeros → ΔW = 0 at start)

# At training start: B·A = 0, so effective W = W0 (identical to base model)
delta_W_init = B @ A
print(f"  Weight shapes:")
print(f"    W0 (frozen):  {tuple(W0.shape)}  ({W0.numel():,} params)")
print(f"    A  (LoRA):    {tuple(A.shape)}   ({A.numel():,} params)")
print(f"    B  (LoRA):    {tuple(B.shape)}   ({B.numel():,} params)")
print(f"    ΔW = B·A:     {tuple(delta_W_init.shape)}")
print()

lora_params  = A.numel() + B.numel()
total_params = W0.numel() + lora_params
reduction    = lora_params / W0.numel()

print(f"  Trainable fraction for this layer:")
print(f"    Full W0 params:   {W0.numel():,}")
print(f"    LoRA A+B params:  {lora_params:,}")
print(f"    Ratio:            {reduction:.4f}  ({reduction*100:.2f}%)")
print(f"    ΔW rank:          {rank}  (out of possible {d_model})")
print()
print(f"  At init: ||B·A||_F = {delta_W_init.norm().item():.6f}  (≈0, pure base model)")

# After some training, B is no longer zero
B_trained = torch.randn(d_model, rank) * 0.01
A_trained = torch.randn(rank, d_model) * 0.01
delta_W   = B_trained @ A_trained
W_final   = W0 + delta_W           # effective weight = original + learned delta
print(f"  After training: ||ΔW||_F = {delta_W.norm().item():.6f}")
print(f"  Singular values of ΔW (first 5): {torch.linalg.svdvals(delta_W)[:5].tolist()}")
print(f"  → ΔW is indeed low-rank: {rank} significant singular values out of {d_model}")
print()

# Compare parameter efficiency at different ranks
print(f"  Parameter efficiency at different ranks (d_model={d_model}):")
print(f"  {'rank':>6} {'LoRA params':>14} {'% of W0':>10} {'Approx quality'}")
print(f"  {'─'*50}")
for r in [4, 8, 16, 32, 64, 128]:
    lp  = 2 * r * d_model
    pct = lp / W0.numel() * 100
    quality = "low" if r <= 4 else "good" if r <= 16 else "high" if r <= 64 else "diminishing"
    print(f"  {r:>6} {lp:>14,} {pct:>10.2f}%  {quality}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: LoRA on a classification model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — LoRA on DistilBERT for sequence classification")
print("━" * 65)
print()

base_model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=4
)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Count base model parameters
base_params = sum(p.numel() for p in base_model.parameters())
print(f"  Base model (DistilBERT): {base_params:,} parameters")
print()

# Configure LoRA for BERT-style encoder
lora_config = LoraConfig(
    r              = 16,                   # rank of the update matrices
    lora_alpha     = 32,                   # scaling factor (α)
    target_modules = ['q_lin', 'v_lin'],   # DistilBERT's attention projections
    lora_dropout   = 0.05,
    bias           = 'none',
    task_type      = TaskType.SEQ_CLS,     # sequence classification
)

lora_model = get_peft_model(base_model, lora_config)

# Print trainable parameters (the PEFT magic)
trainable   = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total       = sum(p.numel() for p in lora_model.parameters())
print(f"  After applying LoRA (r=16, target=[q_lin, v_lin]):")
print(f"    All params:       {total:,}")
print(f"    Trainable params: {trainable:,}")
print(f"    Trainable %:      {100 * trainable / total:.4f}%")
print()

# Show which layers are frozen vs trainable
print(f"  Parameter status per module:")
print(f"  {'Module':<50} {'Params':>10} {'Trainable':>12}")
print(f"  {'─'*75}")
for name, module in lora_model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        n      = module.weight.numel()
        is_tr  = module.weight.requires_grad
        status = "✓ trainable" if is_tr else "  frozen"
        if 'lora' in name.lower() or is_tr:
            print(f"  {name:<50} {n:>10,} {status:>12}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: LoRA training with Trainer
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Training LoRA model with Trainer")
print("━" * 65)
print()

# Quick synthetic dataset (reuse same pattern)
texts  = (["Great financial results for the company."] * 50 +
          ["Scientists discovered a new protein structure."] * 50 +
          ["The team won the championship game."] * 50 +
          ["The film received critical acclaim."] * 50)
labels = [0]*50 + [1]*50 + [2]*50 + [3]*50
idx    = np.random.permutation(200)
texts  = [texts[i] for i in idx]
labels = [labels[i] for i in idx]

train_ds = Dataset.from_dict({'text': texts[:160], 'label': labels[:160]})
val_ds   = Dataset.from_dict({'text': texts[160:], 'label': labels[160:]})

def tokenize_batch(examples):
    return tokenizer(examples['text'], truncation=True, max_length=64)

train_tok = train_ds.map(tokenize_batch, batched=True, remove_columns=['text'])
val_tok   = val_ds.map(tokenize_batch, batched=True, remove_columns=['text'])
train_tok.set_format('torch')
val_tok.set_format('torch')
collator = DataCollatorWithPadding(tokenizer)

with tempfile.TemporaryDirectory() as tmp:
    args = TrainingArguments(
        output_dir              = tmp,
        num_train_epochs        = 5,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 64,
        learning_rate           = 1e-3,      # LoRA can use larger lr than full fine-tuning
        warmup_ratio            = 0.1,
        weight_decay            = 0.01,
        evaluation_strategy     = 'epoch',
        save_strategy           = 'no',
        fp16                    = (device == 'cuda'),
        logging_steps           = 999999,
        report_to               = 'none',
    )

    trainer = Trainer(
        model           = lora_model,
        args            = args,
        train_dataset   = train_tok,
        eval_dataset    = val_tok,
        data_collator   = collator,
    )

    t0 = time.perf_counter()
    trainer.train()
    t_fit = time.perf_counter() - t0
    print(f"  LoRA training: 5 epochs in {t_fit:.1f}s")

    # Evaluate
    metrics = trainer.evaluate()
    print(f"  Final val_loss: {metrics['eval_loss']:.4f}")
    print()

    # ── Save and load LoRA adapter ──────────────────────────────────
    adapter_path = os.path.join(tmp, 'lora_adapter')
    lora_model.save_pretrained(adapter_path)
    adapter_files = os.listdir(adapter_path)
    sizes = {f: os.path.getsize(os.path.join(adapter_path, f))/1024
             for f in adapter_files}

    print(f"  LoRA adapter files saved:")
    for fname, kb in sizes.items():
        print(f"    {fname}: {kb:.1f} KB")
    print()

    # Reload adapter on top of fresh base model
    fresh_base  = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=4)
    loaded_lora = PeftModel.from_pretrained(fresh_base, adapter_path)
    loaded_lora.eval()
    print(f"  Loaded LoRA adapter onto fresh base model ✅")
    print()

    # ── Merge LoRA into base model (zero inference overhead) ────────
    merged = loaded_lora.merge_and_unload()
    merged.eval()

    test_input = tokenizer("The market rose on strong earnings.",
                            return_tensors='pt', padding=True)
    with torch.no_grad():
        out_lora   = loaded_lora(**test_input).logits
        out_merged = merged(**test_input).logits

    diff = (out_lora - out_merged).abs().max().item()
    print(f"  merge_and_unload(): bakes LoRA into base model weights")
    print(f"    Output difference (lora vs merged): {diff:.2e}  ✅ identical")
    print(f"    Merged model has NO extra overhead at inference")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: LoRA rank comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — LoRA rank ablation: r=4 vs r=16 vs r=64")
print("━" * 65)
print()

rank_results = {}
for rank in [4, 16, 64]:
    fresh = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=4)
    cfg = LoraConfig(r=rank, lora_alpha=rank*2,
                     target_modules=['q_lin', 'v_lin'],
                     task_type=TaskType.SEQ_CLS)
    m   = get_peft_model(fresh, cfg)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    rank_results[rank] = {'trainable': trainable, 'pct': 100*trainable/total}

print(f"  {'Rank':>6} {'Trainable':>14} {'%':>10}")
print(f"  {'─'*35}")
for r, info in rank_results.items():
    print(f"  {r:>6} {info['trainable']:>14,} {info['pct']:>10.4f}%")
print()

print(f"  PEFT method decision guide:")
print(f"  ┌─────────────────────────────────────────────────────────────────┐")
print(f"  │ Method          │ Params │ Quality │ Use case                   │")
print(f"  ├─────────────────────────────────────────────────────────────────┤")
print(f"  │ Full fine-tune  │ 100%   │ ★★★★★  │ <7B, plenty GPU            │")
print(f"  │ LoRA r=64       │ ~0.5%  │ ★★★★☆  │ General purpose, best PEFT │")
print(f"  │ LoRA r=16       │ ~0.1%  │ ★★★★☆  │ Default, best efficiency   │")
print(f"  │ LoRA r=4        │ ~0.03% │ ★★★☆☆  │ Very limited GPU memory    │")
print(f"  │ Prompt tuning   │ <0.01% │ ★★★☆☆  │ Very large models (>10B)   │")
print(f"  │ Prefix tuning   │ <0.1%  │ ★★★☆☆  │ Generation tasks           │")
print(f"  └─────────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Generation Strategies, Quantization & ONNX Deployment": {
        "description": (
            "Text generation deep dive: greedy, beam, sampling strategies. "
            "Temperature, top-k, top-p nucleus sampling comparison. "
            "Repetition penalty and length control. "
            "model.generate() full argument reference. "
            "chat_template for instruction-tuned models. "
            "BitsAndBytes 8-bit and 4-bit quantization (load_in_8bit / load_in_4bit). "
            "Memory footprint: fp32 vs fp16 vs int8 vs int4 models. "
            "ONNX export via optimum and onnxruntime inference benchmark. "
            "Batched generation with padding_side='left'. "
            "Streaming generation with TextStreamer."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import tempfile
import os

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TextStreamer, logging as hf_logging,
    )
    hf_logging.set_verbosity_error()
    print(f"  transformers: installed | torch: {torch.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'transformers', 'torch', '--quiet'], check=True)
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TextStreamer, logging as hf_logging,
    )
    hf_logging.set_verbosity_error()

print("=" * 65)
print("  GENERATION STRATEGIES, QUANTIZATION & DEPLOYMENT")
print("=" * 65)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Load a small generative model (GPT-2)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Loading GPT-2: causal language model basics")
print("━" * 65)
print()

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.padding_side = 'left'    # CRITICAL for batched generation with causal LM
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no PAD by default

model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  GPT-2 (small): {n_params/1e6:.0f}M parameters")
print(f"  Architecture:  12 layers, 768 d_model, 12 heads, 50257 vocab")
print(f"  Context length: {model.config.max_position_embeddings} tokens")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Greedy vs beam search vs sampling
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Decoding strategies: greedy, beam, sampling")
print("━" * 65)
print()

prompt      = "Artificial intelligence will transform"
input_ids   = tokenizer.encode(prompt, return_tensors='pt')

strategies = {
    'Greedy': dict(do_sample=False, num_beams=1),
    'Beam (k=4)': dict(do_sample=False, num_beams=4, early_stopping=True),
    'Sampling T=1.0': dict(do_sample=True, temperature=1.0, top_k=0),
    'Sampling T=0.7': dict(do_sample=True, temperature=0.7, top_k=0),
    'Top-k (k=50)': dict(do_sample=True, temperature=1.0, top_k=50),
    'Nucleus p=0.9': dict(do_sample=True, top_p=0.9, top_k=0, temperature=1.0),
    'Production mix': dict(do_sample=True, temperature=0.7, top_p=0.9,
                           top_k=50, repetition_penalty=1.1),
}

print(f"  Prompt: '{prompt}'")
print()
torch.manual_seed(42)
for name, params in strategies.items():
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens   = 20,
            pad_token_id     = tokenizer.eos_token_id,
            **params
        )
    # Only show newly generated tokens
    new_ids = out[0, input_ids.shape[1]:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"  [{name:<18}] {generated}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: generation config parameters reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — model.generate() parameters reference")
print("━" * 65)
print()

GEN_REFERENCE = """
model.generate(
    input_ids,                     # encoded prompt: (batch, seq_len)

    # ── Output length control ─────────────────────────────────────
    max_new_tokens     = 200,      # max tokens to GENERATE (preferred)
    max_length         = None,     # max TOTAL sequence length (input + output)
    min_new_tokens     = 10,       # min tokens to generate before EOS
    min_length         = None,     # min total length before EOS allowed

    # ── Decoding strategy ─────────────────────────────────────────
    do_sample          = True,     # False=greedy/beam, True=sampling
    num_beams          = 1,        # beam width (1=greedy, 4-8=beam search)
    num_return_sequences = 1,      # how many complete sequences to return

    # ── Sampling parameters ───────────────────────────────────────
    temperature        = 0.7,      # < 1 = focused, > 1 = diverse
    top_k              = 50,       # sample from top-k tokens only
    top_p              = 0.9,      # sample from top-p cumulative probability
    typical_p          = None,     # typical sampling (alternative to top-p)

    # ── Anti-repetition ───────────────────────────────────────────
    repetition_penalty = 1.1,      # > 1.0 penalises repeated tokens
    no_repeat_ngram_size = 3,      # forbid repeating n-grams of this size
    encoder_no_repeat_ngram_size=3,# for seq2seq: forbid input n-grams in output

    # ── Special tokens ────────────────────────────────────────────
    eos_token_id       = tokenizer.eos_token_id,    # stop token
    pad_token_id       = tokenizer.pad_token_id,    # padding token
    forced_bos_token_id = None,    # force first generated token
    forced_eos_token_id = None,    # force last generated token
    suppress_tokens    = None,     # list of token IDs never to generate

    # ── Return options ────────────────────────────────────────────
    return_dict_in_generate = True,  # return GenerateOutput object
    output_scores      = True,       # return per-token logit scores
    output_attentions  = False,      # return attention weights
    output_hidden_states = False,    # return hidden states

    # ── Performance ───────────────────────────────────────────────
    use_cache          = True,     # use KV-cache (default True, always on)
)
"""
print(GEN_REFERENCE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Batched generation (padding_side='left')
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Batched generation (padding_side=left is critical)")
print("━" * 65)
print()

batch_prompts = [
    "The history of Python programming language",
    "Climate change is a",
    "In the year 2050, robots will",
]

# For causal LM batching: padding on the LEFT so real tokens are at the end
# and model generates right after the actual prompt end (not after padding)
batch_inputs = tokenizer(
    batch_prompts,
    return_tensors = 'pt',
    padding        = True,
    truncation     = True,
    max_length     = 64,
)

print(f"  padding_side='left' — padding tokens are prepended, not appended.")
print(f"  This ensures generation continues from the REAL end of each prompt.")
print()
print(f"  Attention masks (1=real, 0=pad):")
for i, (prompt, mask) in enumerate(zip(batch_prompts, batch_inputs['attention_mask'])):
    n_pad  = (mask == 0).sum().item()
    n_real = (mask == 1).sum().item()
    print(f"    [{i}] {n_real} real + {n_pad} pad | '{prompt}'")
print()

torch.manual_seed(0)
with torch.no_grad():
    batch_out = model.generate(
        **batch_inputs,
        max_new_tokens   = 15,
        do_sample        = True,
        temperature      = 0.7,
        top_p            = 0.9,
        pad_token_id     = tokenizer.eos_token_id,
    )

print(f"  Batched generation results:")
for i, (prompt, out_ids) in enumerate(zip(batch_prompts, batch_out)):
    # Extract only newly generated tokens
    n_input  = batch_inputs['input_ids'].shape[1]
    new_ids  = out_ids[n_input:]
    new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"  [{i}] {prompt}")
    print(f"      → {new_text}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Memory footprint across precision levels
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Memory footprint: fp32 vs fp16 vs int8 vs int4")
print("━" * 65)
print()

# Measure actual GPT-2 memory in different precision
def count_model_bytes(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total

fp32_model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32)
fp16_model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float16)

fp32_bytes = count_model_bytes(fp32_model)
fp16_bytes = count_model_bytes(fp16_model)

print(f"  GPT-2 (124M params) memory footprint:")
print(f"  {'Precision':>12} {'Bytes per param':>18} {'Total (MB)':>12} {'Relative':>12}")
print(f"  {'─'*56}")
for label, nbytes, bpp in [
    ('float32',  fp32_bytes, 4),
    ('float16',  fp16_bytes, 2),
    ('int8 est', fp32_bytes//4, 1),     # estimated
    ('int4 est', fp32_bytes//8, 0.5),   # estimated
]:
    mb = nbytes / 1e6
    rel = nbytes / fp32_bytes
    print(f"  {label:>12} {bpp:>18.1f} {mb:>12.1f} {rel:>12.2f}×")
print()

# Extrapolate to larger models
print(f"  Estimated VRAM for larger models (approx params × bytes/param):")
print(f"  {'Model':>25} {'fp32':>10} {'fp16':>10} {'int8':>10} {'int4':>10}")
print(f"  {'─'*68}")
models_info = [
    ('DistilBERT', 66),
    ('BERT-base', 110),
    ('RoBERTa-large', 355),
    ('GPT-2 XL', 1500),
    ('LLaMA-3 8B', 8_000),
    ('LLaMA-3 70B', 70_000),
]
for name, params_m in models_info:
    fp32_gb = params_m * 4 / 1024
    fp16_gb = params_m * 2 / 1024
    int8_gb = params_m * 1 / 1024
    int4_gb = params_m * 0.5 / 1024
    print(f"  {name:>25} {fp32_gb:>9.1f}G {fp16_gb:>9.1f}G {int8_gb:>9.1f}G {int4_gb:>9.1f}G")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: BitsAndBytes quantisation pattern (code reference)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — BitsAndBytes quantisation patterns")
print("━" * 65)
print()

QUANTIZATION_PATTERNS = """
# ── 8-bit quantisation (bitsandbytes) ───────────────────────────────────
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_8bit = True,          # activates LLM.int8()
    device_map   = "auto",        # auto-shard across available GPUs
)
# Memory: ~8GB for 8B model (vs ~16GB fp16)

# ── 4-bit NF4 quantisation (QLoRA style) ────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",          # NormalFloat4: optimal for LLMs
    bnb_4bit_compute_dtype    = torch.bfloat16, # compute in bf16 for speed
    bnb_4bit_use_double_quant = True,           # second quantisation: saves ~0.4 bits
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config = bnb_config,
    device_map          = "auto",
)
# Memory: ~5GB for 8B model with double quantisation

# ── QLoRA: 4-bit base + LoRA fine-tuning ────────────────────────────────
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Step 1: Load quantised base model
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# Step 2: Prepare for k-bit training (gradient checkpointing, cast norms to f32)
model = prepare_model_for_kbit_training(model)

# Step 3: Apply LoRA on top of quantised model
lora_config = LoraConfig(
    r              = 64,
    lora_alpha     = 128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# Trainable params: ~65M (0.8% of 8B) — fits in 10GB GPU for training!
"""
print(QUANTIZATION_PATTERNS)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 7: ONNX export for production deployment
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 7 — ONNX export via torch.onnx")
print("━" * 65)
print()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

clf_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
clf_model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english')
clf_model.eval()

with tempfile.TemporaryDirectory() as tmp:
    onnx_path = os.path.join(tmp, 'classifier.onnx')
    dummy_text = "This is a test sentence for ONNX export."
    dummy_inputs = clf_tokenizer(dummy_text, return_tensors='pt',
                                  padding='max_length', max_length=64, truncation=True)

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            clf_model,
            (dummy_inputs['input_ids'], dummy_inputs['attention_mask']),
            onnx_path,
            input_names    = ['input_ids', 'attention_mask'],
            output_names   = ['logits'],
            dynamic_axes   = {
                'input_ids':      {0: 'batch', 1: 'seq_len'},
                'attention_mask': {0: 'batch', 1: 'seq_len'},
                'logits':         {0: 'batch'},
            },
            opset_version  = 14,
        )
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  ONNX export complete: {size_kb:.0f} KB")

    # Benchmark: PyTorch vs ONNX Runtime
    test_texts  = ["I love this product!"] * 32
    test_inputs = clf_tokenizer(test_texts, return_tensors='pt',
                                 padding=True, truncation=True, max_length=64)

    N = 20
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(N): _ = clf_model(**test_inputs).logits
        t_pt = (time.perf_counter() - t0) / N * 1000

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        ort_inputs = {
            'input_ids':      test_inputs['input_ids'].numpy(),
            'attention_mask': test_inputs['attention_mask'].numpy(),
        }
        t0 = time.perf_counter()
        for _ in range(N): _ = sess.run(None, ort_inputs)
        t_ort = (time.perf_counter() - t0) / N * 1000

        print(f"  Inference benchmark (batch=32, max_len=64, {N} runs):")
        print(f"    PyTorch:       {t_pt:.1f}ms/batch")
        print(f"    ONNX Runtime:  {t_ort:.1f}ms/batch")
        print(f"    Speedup:       {t_pt/t_ort:.2f}×")
    except ImportError:
        print(f"  PyTorch inference: {t_pt:.1f}ms/batch")
        print(f"  (install onnxruntime for benchmark comparison)")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 8: Chat templates for instruction-tuned models
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 8 — Chat templates and instruction formatting")
print("━" * 65)
print()

# Show chat template formatting WITHOUT loading a full chat model
# (demonstrate the pattern using the tokenizer's apply_chat_template)
CHAT_TEMPLATE_REFERENCE = """
# Chat template: formats messages into model-specific prompt strings.
# Different models have different prompt formats — templates standardise this.

# LLaMA-3 / LLaMA-3.1 format:
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
messages  = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user",      "content": "What is its population?"},
]
# apply_chat_template converts messages dict into the correct formatted string:
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize       = False,         # return string, not token IDs
    add_generation_prompt = True,   # add "Assistant:" signal to trigger generation
)
# prompt looks like:
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# What is its population?<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>

# Generate response:
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200,
                         do_sample=True, temperature=0.7)
response = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
# response: "The population of Paris is approximately 2.1 million..."

# Different models, different formats (apply_chat_template handles all):
# Mistral:   "[INST] {user} [/INST] {assistant} </s>[INST] {user2} [/INST]"
# ChatML:    "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
# Gemma:     "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
"""
print(CHAT_TEMPLATE_REFERENCE)

print(f"  DEPLOYMENT STRATEGY GUIDE:")
print(f"  ┌─────────────────────────────────────────────────────────────────┐")
print(f"  │ Use case                 │ Recommended approach                │")
print(f"  ├─────────────────────────────────────────────────────────────────┤")
print(f"  │ BERT classification      │ Fine-tune → ONNX export → ORT       │")
print(f"  │ NER / token labelling    │ Fine-tune → ONNX (with offset map)  │")
print(f"  │ Small LLM (<7B) local    │ Transformers + fp16/bf16            │")
print(f"  │ Large LLM (7B+) local    │ 4-bit BnB or GGUF + llama.cpp       │")
print(f"  │ Production API (LLM)     │ vLLM (continuous batching)          │")
print(f"  │ Mobile/edge deployment   │ Optimum → CoreML or TFLite          │")
print(f"  │ High-throughput BERT     │ TensorRT via Optimum NVIDIA          │")
print(f"  │ Multi-task from one base │ PEFT adapters, swap at request time │")
print(f"  └─────────────────────────────────────────────────────────────────┘")
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