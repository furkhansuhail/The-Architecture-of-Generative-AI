import textwrap
import re

TOPIC_NAME = "Order for LLM Training"
DISPLAY_NAME = "00 · Order for LLM Training"
ICON = "🧠"
SUBTITLE = "LLM Training curriculum order with priorities"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### LLM Training in AI/ML

Training a large language model is not a single problem — it is a stack of interlocking
engineering and research decisions, each of which can silently bottleneck or corrupt
everything above it. Getting the architecture right matters nothing if the optimizer
diverges. Getting the optimizer right matters nothing if the data pipeline starves the
GPU. Getting all of that right matters nothing if gradient synchronization across nodes
introduces nans or the checkpoint strategy loses a week of compute to a disk failure.
This curriculum is ordered so that each topic builds on the last — starting from the
transformer as a mathematical object, moving through single-GPU training mechanics,
scaling to multi-GPU and multi-node systems, and finishing with fine-tuning, inference,
and production infrastructure.

## The Core Problem

A language model is a function over sequences. Given tokens [t₁, t₂, ..., tₙ], it
produces a distribution over the next token tₙ₊₁. Training it means adjusting billions
of parameters to make this distribution match the true distribution of human language —
which we approximate by minimizing cross-entropy loss over trillions of tokens. The
challenge is not the math; the math is a few lines. The challenge is that this process
must run continuously for weeks or months across hundreds or thousands of GPUs, with
no silent correctness bugs, no wasted compute, and no single point of failure. Every
topic in this curriculum is a specific engineering problem that stands between you and
that goal.

## Two Levels of the Stack

Model-level concerns are about the architecture and the learning dynamics: how the
transformer is constructed, how attention scales, which positional encoding scheme works
for long contexts, how the optimizer handles the loss landscape, and how data quality
affects what the model learns. System-level concerns are about making that computation
run efficiently at scale: how weight tensors are sharded across devices, how gradients
flow back through a pipeline, how collective communication overlaps with backward passes,
and how the training loop is hardened against hardware failures. Both levels must be
understood — a researcher who ignores systems will be bottlenecked by infrastructure;
an engineer who ignores model-level dynamics will optimize the wrong things.

## Transformer Architecture Fundamentals

The **transformer architecture** is the substrate on which every other topic in this
curriculum builds. The residual stream, multi-head attention, and feedforward sublayers
are not just components — they impose specific mathematical structure on gradients,
activations, and weight updates that determines what optimization strategies work and
what hardware layouts are efficient. Understanding the forward pass at the level of
tensor shapes and matrix operations — not just conceptually — is the prerequisite for
understanding why FlashAttention works, why tensor parallelism shards the way it does,
and why certain learning rate schedules are needed.

**Tokenization and vocabulary** sit at the boundary between raw text and the model's
mathematical world. Byte Pair Encoding (BPE), SentencePiece, and Unigram Language Model
tokenizers each make different tradeoffs between vocabulary coverage, token fertility
(how many tokens a given text costs), and cross-lingual generalization. Vocabulary size
directly affects the embedding matrix size and the unembedding projection — a 128K
vocabulary adds 128K × d_model parameters to both. Tokenization bugs are among the
hardest to detect: a misconfigured tokenizer silently changes the training distribution
in ways that only manifest in downstream evaluations.

**Positional encodings** are how the model learns that token 5 comes after token 4.
Absolute sinusoidal encodings (the original transformer) are simple but don't generalize
to sequences longer than those seen in training. Rotary Position Embeddings (RoPE) encode
position as a rotation in embedding space, allowing relative position information to be
computed directly in the attention dot product — which is why every modern LLM uses RoPE
or a variant of it. ALiBi (Attention with Linear Biases) applies a fixed penalty to
attention scores based on distance, providing length generalization without learned
parameters. The choice of positional encoding determines the model's extrapolation
behavior and is one of the decisions hardest to change after pretraining begins.

**Attention variants — MHA, MQA, GQA** — address the inference-time memory bottleneck
of the KV cache. Multi-Head Attention (MHA) gives each head its own key and value
projections; at inference time, storing keys and values for all heads across all layers
dominates memory for long sequences. Multi-Query Attention (MQA) shares a single KV
head across all query heads, reducing KV cache size by the number of heads — a dramatic
memory saving with modest quality impact. Grouped-Query Attention (GQA) is the
intermediate: groups of query heads share a KV head, balancing quality and cache size.
Llama 2 and 3, Mistral, and Gemma all use GQA. The choice affects both training memory
and inference serving architecture.

**Modern architecture variants** — LLaMA, Mistral, Mixtral, Phi, Gemma, and Falcon —
share a common blueprint but differ in the details that matter most for training and
inference: RMSNorm instead of LayerNorm (saves a mean subtraction), SwiGLU instead of
ReLU in the feedforward (better empirical performance per parameter), GQA for KV cache
efficiency, sliding window attention for long contexts, and mixture-of-experts (MoE)
layers for parameter efficiency at scale. Understanding why each variant makes the
choices it does — not just what the choices are — is how you evaluate whether a new
architecture claim is a genuine advance or a hyperparameter difference.

## Training Loop and Optimization

**The training loop** is the innermost loop of everything in this curriculum. At its
core: forward pass, loss computation, backward pass, optimizer step, repeat. But the
production version of this loop handles gradient accumulation across micro-batches, loss
scaling for mixed precision, gradient clipping before the optimizer step, learning rate
scheduling, and checkpointing at the right cadence — all while maintaining exact numerical
reproducibility for debugging. Building this loop correctly from scratch, without a
high-level framework hiding the details, is the foundation everything else depends on.

**Optimizers — Adam and AdamW** — are the default for LLM training. Adam maintains
per-parameter first and second moment estimates, providing adaptive learning rates that
work across the heterogeneous parameter types in a transformer (embedding rows that
rarely update alongside attention projection matrices that update every step). AdamW
corrects Adam's weight decay implementation — standard Adam applies weight decay as a
gradient penalty, which interacts badly with the adaptive scaling; AdamW applies it
directly to the weights, which is the mathematically correct decoupled form. At scale,
optimizer state (two FP32 moments per parameter) often consumes more memory than the
model weights themselves — a critical consideration for ZeRO and CPU offloading strategies.

**Learning rate schedules — warmup and cosine decay** — are not optional. Training
without warmup fails: at initialization, gradient magnitudes are large and the Adam
second-moment estimates are uninitialised, so the first few hundred steps need a
conservatively small effective learning rate. Linear warmup over 1000–2000 steps
standard practice. Cosine decay over the rest of training smoothly reduces the learning
rate to near zero, empirically outperforming step-decay schedules. The Chinchilla optimal
scaling laws specifically assume cosine decay. For continued pretraining or fine-tuning,
choosing where to resume on the cosine schedule — or whether to restart it — significantly
affects final performance.

**Gradient clipping** prevents loss spikes from sending parameters to extreme values
that the model never recovers from. By rescaling the gradient vector whenever its L2
norm exceeds a threshold (typically 1.0), gradient clipping bounds the maximum parameter
update size without discarding gradient direction information. Loss spikes are common
in LLM training and almost always trace to a few pathological batches with unusually
long sequences or rare token combinations. Monitoring the pre-clip gradient norm is an
essential training health metric — a norm that's consistently near the clip threshold
indicates that clipping is doing real work and may signal that the learning rate is
too high.

**Mixed precision — BF16 and FP16** reduce memory and increase throughput by representing
activations and weights in 16-bit formats while keeping optimizer state in FP32. BF16
(Brain Floating Point) has the same exponent range as FP32 but fewer mantissa bits;
it handles the wide dynamic range of gradients without overflow, making it the preferred
format for LLM training on Ampere and later GPUs. FP16 has a smaller exponent range and
requires loss scaling to prevent gradient underflow. Training in BF16 typically halves
activation memory and doubles throughput for attention and GEMM operations, at the cost
of slightly noisier gradient estimates that rarely matter in practice.

**Activation checkpointing** (gradient checkpointing) is the technique for reducing
activation memory at the cost of additional compute. During the forward pass, instead
of storing all intermediate activations needed for the backward pass, only a subset of
"checkpoint" activations are stored — typically one per transformer layer. During the
backward pass, the forward pass is recomputed from the nearest checkpoint to recover the
discarded activations. This trades roughly 33% extra compute for a dramatic reduction
in peak memory (from O(layers × sequence_length) to O(√layers × sequence_length) with
optimal placement). Activation checkpointing is what makes training very deep models
or very long sequences feasible on finite GPU memory.

**Gradient accumulation** simulates a large batch size by accumulating gradients across
multiple micro-batches before performing an optimizer step. If you want an effective
batch size of 4M tokens but each GPU can only process 256K tokens per step, you
accumulate gradients over 16 micro-batches before calling `optimizer.step()`. This
requires careful handling of gradient normalization — dividing by the number of
accumulation steps — and interacts with mixed precision loss scaling, distributed
training gradient synchronization, and per-step learning rate warmup. Getting
accumulation numerically equivalent to a single large batch step is non-trivial.

## Data: Preparation, Loading, and Mixing

**Dataset preparation and preprocessing** determines what the model can possibly learn.
Web-scale pretraining datasets (The Pile, RedPajama, DCLM, FineWeb) are assembled from
diverse sources — Common Crawl, GitHub, arXiv, books, Wikipedia — and require extensive
filtering: deduplication (exact and near-duplicate removal), quality filtering (perplexity
filtering, heuristic rules for boilerplate removal), personally identifiable information
(PII) redaction, and toxic content removal. The preprocessing pipeline must be both
correct — preserving the true distribution of high-quality text — and efficient enough
to process terabytes of data in reasonable time. Data quality decisions made here have
larger effects on final model quality than most architectural choices.

**Data loading, batching, and sequence packing** is the engineering challenge of feeding
hundreds of GPUs without creating a CPU bottleneck. Sequences of different lengths must
be packed into fixed-length training chunks to avoid wasted padding tokens — a packed
sequence of 4096 tokens may contain 5 short documents concatenated with special separator
tokens, maximizing GPU utilization. The data loader must deliver the next batch to the
GPU before the current batch finishes training, which requires asynchronous prefetching,
multiple worker processes, and careful handling of epoch boundaries and dataset sharding
across distributed workers. A poorly implemented data pipeline can bottleneck the entire
training run by starving GPUs between steps.

**Data mixing and curriculum** are how you combine sources (code, books, web text,
scientific papers) in proportions that produce the best model. Mixing ratios are
hyperparameters: too much code produces a model weak on prose; too much web text produces
a model weak on reasoning. Domain weights are often determined by ablation on small proxy
models. Curriculum — changing the data distribution over training — is less well-studied
but increasingly important: some practitioners upsample high-quality data toward the
end of training (the "cool-down" phase) to improve instruction-following without the
cost of full fine-tuning. Getting mixing right requires a fast evaluation pipeline to
detect regressions early.

## Parallelism Strategies

**Data Parallel Training and DDP** is the foundation of distributed training. Each GPU
holds a complete copy of the model and processes a different shard of each batch. After
the backward pass, gradients are synchronized across all replicas via an all-reduce
operation (typically ring all-reduce using NCCL), ensuring all copies receive identical
gradient updates and stay in sync. PyTorch DDP overlaps gradient communication with
the backward pass by launching all-reduce as soon as each gradient bucket is ready,
rather than waiting for the full backward pass to complete. DDP is the first parallelism
strategy to reach for — it scales perfectly to the point where a single model replica
no longer fits in one GPU's memory.

**Tensor Parallelism** shards individual weight matrices across multiple GPUs. An
attention projection matrix W of shape [d_model, d_model] can be column-sharded so each
GPU holds [d_model, d_model/N] columns. Megatron-LM's tensor parallelism scheme shards
attention heads and feedforward matrices in a way that requires exactly two all-reduce
operations per transformer layer — one after the QKV projection and one after the
feedforward. Because tensor parallelism requires communication for every forward and
backward pass, it works best within a single node connected by NVLink, where bandwidth
is 600+ GB/s. Across nodes on InfiniBand, the communication overhead makes tensor
parallelism inefficient for typical transformer shapes.

**Pipeline Parallelism** assigns consecutive groups of transformer layers to consecutive
GPUs: GPU 0 holds layers 1–8, GPU 1 holds layers 9–16, and so on. The pipeline processes
micro-batches, with each GPU working on a different micro-batch at any given time. Without
careful scheduling, pipeline parallelism creates "bubbles" — idle time when a GPU is
waiting for the previous stage to finish. GPipe and PipeDream-Flush (1F1B scheduling)
minimize bubble fraction. Pipeline parallelism communicates only activations at stage
boundaries (rather than full weight gradients), making it efficient across nodes —
Megatron-LM uses tensor + pipeline + data parallelism (3D parallelism) to train models
at hundreds of billions of parameters.

**Sequence Parallelism** distributes the sequence dimension across GPUs, complementing
tensor parallelism. In standard tensor parallelism, the dropout and layer norm operations
are replicated on each GPU (they don't benefit from model sharding). Sequence parallelism
shards these operations along the sequence dimension, so each GPU processes a subsequence.
This reduces activation memory proportional to the number of GPUs and is critical for
very long sequence training. Megatron's sequence parallelism combines with tensor
parallelism using all-gather and reduce-scatter operations that achieve the same
communication volume as a single all-reduce.

**Expert Parallelism and Mixture-of-Experts (MoE)** replace dense feedforward layers
with a routing mechanism that sends each token to one or a few "expert" sub-networks
out of a larger pool. A model with 8 experts and top-2 routing activates only 2/8 of
the feedforward parameters for each token, dramatically reducing compute per token while
maintaining total parameter count. Expert parallelism places different experts on different
GPUs, requiring all-to-all communication to route tokens to the right device. Mixtral 8×7B
and the Switch Transformer are production examples. The engineering challenges are load
balancing (ensuring tokens are distributed evenly across experts) and handling variable-
length expert inputs efficiently.

**3D Parallelism — Megatron-DeepSpeed** combines data, tensor, and pipeline parallelism
in a single coherent strategy. A cluster of 1024 GPUs might use data parallelism across
groups of 8, tensor parallelism within each group of 8 (NVLink-connected), and pipeline
parallelism across groups of 4 nodes — achieving 8 × 4 × 32 = 1024 GPU training. The
interaction between these three strategies is complex: the order in which gradients are
reduced, the assignment of micro-batches to pipeline stages, and the communication pattern
of tensor parallelism all affect both correctness and performance. Megatron-LM and
DeepSpeed both implement this combination; understanding their implementation is necessary
for training at this scale.

## Memory Optimization

**ZeRO optimizer stages** (Zero Redundancy Optimizer, Rajbhandari et al., 2020) eliminate
the memory redundancy in standard data parallel training. In DDP, every GPU holds full
copies of model parameters, gradients, and optimizer state. For a 7B parameter model in
FP16 with Adam, this is roughly 7B × (2 + 2 + 8) bytes = 84 GB per GPU — far exceeding
most GPU memory. ZeRO Stage 1 shards optimizer state across GPUs (8× memory reduction
for Adam). ZeRO Stage 2 additionally shards gradients. ZeRO Stage 3 shards the parameters
themselves — every GPU holds only 1/N of the weights and fetches the rest during the
forward and backward pass via all-gather operations. DeepSpeed implements all three
stages; Fully Sharded Data Parallel (FSDP) in PyTorch implements ZeRO Stage 3.

**CPU offloading** extends ZeRO further by moving optimizer state or parameters to CPU
RAM when they exceed GPU memory. During the forward and backward pass, needed parameter
shards are streamed from CPU to GPU via PCIe; after the optimizer step, updated weights
are streamed back. CPU offloading makes it possible to train models that don't fit in
GPU memory at the cost of lower throughput — PCIe bandwidth (~16 GB/s) is much slower
than HBM bandwidth (~2 TB/s). DeepSpeed ZeRO-Infinity extends this to NVMe storage.
CPU offloading is most useful for fine-tuning very large models on limited hardware,
where throughput is less critical than fitting the model at all.

## Fine-Tuning

**Supervised Fine-Tuning (SFT)** adapts a pretrained base model to follow instructions
by training on demonstration data: (instruction, response) pairs where the response is
the ground truth the model should learn to produce. SFT involves masking the instruction
tokens from the loss (computing loss only on response tokens), using a small learning rate
(1e-5 to 1e-6) relative to pretraining, and training for 1–3 epochs to avoid overfitting
to the fine-tuning distribution. The quality of fine-tuning data matters more than
quantity — 1000 carefully curated examples often outperform 100K noisy ones. This is the
first stage of the alignment pipeline that produces chat and instruction-following models.

**Parameter-Efficient Fine-Tuning (PEFT) — LoRA and QLoRA** adapt large models by
training a small number of additional parameters rather than all weights. LoRA (Low-Rank
Adaptation) freezes the original weight matrix W and adds a low-rank decomposition
ΔW = BA, where B ∈ ℝ^{d×r} and A ∈ ℝ^{r×k} with r << min(d,k). Only A and B are trained,
reducing trainable parameters by 10–10,000× depending on rank. At inference, the LoRA
weights can be merged back into the base model with no overhead. QLoRA combines LoRA
with 4-bit quantization of the base model weights, making fine-tuning a 65B parameter
model feasible on a single 80GB GPU. PEFT is the dominant approach for domain adaptation
and task-specific fine-tuning in resource-constrained settings.

**RLHF — Reward Modeling and PPO** is the learning-from-human-feedback pipeline that
produces models with strong human preference alignment. A reward model is trained on
human preference comparisons: given two responses to the same prompt, a human labels
which is better, and the reward model learns to predict this preference. The policy (the
LLM being fine-tuned) is then trained via Proximal Policy Optimization (PPO) to maximize
reward model scores while a KL penalty prevents it from drifting too far from the SFT
model. RLHF is what separates instruction-following models from chat-aligned models —
it teaches the model not just to complete instructions but to produce outputs that humans
actually prefer, handling tone, format, safety, and helpfulness simultaneously.

**DPO — Direct Preference Optimization** is a mathematically equivalent but practically
simpler alternative to RLHF that fine-tunes on preference data without training a separate
reward model or running PPO. It reparameterizes the RLHF objective to express the optimal
policy directly as a function of a reference model, reducing the entire pipeline to a
supervised learning problem on (prompt, chosen_response, rejected_response) triples.
DPO is more stable and cheaper than PPO-based RLHF, which is why it's now the dominant
alignment fine-tuning method. SimPO, KTO, and IPO are variants that address specific
weaknesses of the original DPO formulation.

## Inference Optimization

**Speculative decoding** accelerates autoregressive generation by using a small "draft"
model to propose multiple tokens in parallel, then verifying them with the large model
in a single forward pass. The draft model generates a sequence of k tokens speculatively;
the target model evaluates all k+1 tokens simultaneously (since it can process them in
parallel, unlike during standard generation). Any draft tokens that the target model
would have generated identically are accepted; the first rejected token triggers a
correction and the process repeats. With a good draft model (same architecture family,
3–10× smaller), speculative decoding achieves 2–3× generation speedup with zero quality
degradation — the outputs are mathematically identical to standard sampling from the
target model.

**Inference optimization and serving** covers the systems engineering of deploying a
model to handle real user traffic efficiently. Continuous batching (used in vLLM and
TGI) dynamically inserts new requests into an ongoing generation batch without waiting
for all sequences to complete, dramatically improving GPU utilization compared to static
batching. PagedAttention manages KV cache memory using a virtual memory scheme inspired
by OS paging, allowing efficient multiplexing of variable-length sequences and eliminating
memory fragmentation. Tensor parallelism and pipeline parallelism are used at inference
time differently than at training time — inference has different latency/throughput
tradeoffs, and the optimal sharding strategy for a 4ms SLA is different from the optimal
strategy for maximizing training throughput.

**Quantization for inference** compresses model weights to INT8, INT4, or lower precision,
reducing memory bandwidth requirements and exploiting integer arithmetic units. GPTQ
(post-training quantization via approximate second-order information) and AWQ (activation-
aware weight quantization, which identifies and protects salient weights from aggressive
quantization) are the two dominant PTQ methods for LLMs. bitsandbytes provides 8-bit and
4-bit quantization with minimal quality degradation for most models. The engineering
challenge is implementing dequantization efficiently at the kernel level: weights are
stored in INT4 but must be multiplied with FP16 activations, requiring on-the-fly
conversion that must not become a bottleneck.

## Infrastructure and Production

**Checkpoint management** is the system that prevents weeks of compute from being lost
to a hardware failure. A checkpoint saves model weights, optimizer state, learning rate
schedule state, and data loader position so training can resume from exactly the last
saved point. For a 70B parameter model with Adam, a single checkpoint is ~1 TB in FP32.
At this scale, checkpointing to shared storage must itself be parallelized (each rank
saves its own shard), checkpoints must be validated before deleting the previous one,
and the checkpoint frequency must balance fault recovery granularity against I/O overhead.
Distributed checkpointing (PyTorch's `torch.distributed.checkpoint`) and async
checkpointing (writing in a background thread without blocking training) are now standard
practice for large-scale runs.

**Experiment tracking — Weights & Biases and MLflow** are the observability layer for
training runs. At minimum, every run should log training loss, validation loss, gradient
norm, learning rate, and throughput (tokens per second) at each step. W&B and MLflow
both provide experiment comparison, hyperparameter search visualization, and alert
systems for training anomalies. For multi-node training, all ranks typically log, but
only rank 0 writes to the tracking server. Reproducibility — logging the exact git
commit, configuration file, and random seeds used for each run — is what separates
experiments you can build on from experiments you can't explain.

**Training stability and debugging** is the art of diagnosing why a training run
diverged, stalled, or produced a bad model. Loss spikes almost always trace to pathological
batches — sequences with unusual token distributions that produce large gradient magnitudes.
The diagnostic toolkit includes gradient norm monitoring (spike before the loss spike),
loss decomposition by data source (which domain's loss is driving the divergence), and
attention entropy monitoring (collapsed attention heads indicate pathological softmax
saturation). NaN propagation through mixed-precision training requires tracking exactly
where the NaN originated — a single overflowing activation early in the forward pass
can corrupt the entire backward pass in ways that take hours to localize.

**ML System Design** encompasses the architectural decisions that determine whether a
training infrastructure can be built, maintained, and scaled by a team. Storage
throughput (streaming terabytes of data to hundreds of GPUs without disk I/O becoming
the bottleneck), network topology (how nodes are connected and what collective operations
the topology supports efficiently), job scheduling (how training jobs share a cluster
with other workloads), and cost optimization (spot instances, preemptible jobs, partial
failure recovery) are all system design problems distinct from model development. A team
that solves the model problems but not the infrastructure problems will find their
research cycle limited by queue times and hardware costs rather than ideas.

## How the Layers Connect

A complete LLM training run uses every layer of this curriculum simultaneously. A
transformer built with RoPE and GQA (architecture) is tokenized with a BPE vocabulary
of 128K tokens (tokenization) and trained in BF16 with AdamW and cosine warmup
(optimization) on 2T packed tokens assembled from 15 data sources with a tuned mixing
ratio (data). Single-GPU training validates the implementation; then DDP scales to 8
GPUs per node, tensor parallelism shards across 8 GPUs within each node, pipeline
parallelism spans 4 nodes (3D parallelism), with ZeRO Stage 2 reducing gradient memory
and activation checkpointing handling sequence length. After pretraining, SFT and DPO
align the model to human preferences (fine-tuning), LoRA adapts it to a specific domain,
and speculative decoding + continuous batching serve it at scale (inference). W&B logs
every metric; checkpoints are saved every 1000 steps with async writes (infrastructure).

Understanding LLM training isn't about memorizing API calls to Hugging Face or DeepSpeed.
It's about building a mental model deep enough to know why a run is slow, why a model
is underperforming, and which lever to pull next. Every abstraction in this curriculum
— from GQA to ZeRO to DPO — is a specific engineering or research decision made to
solve a specific problem. Understanding the problem is how you know when the solution
applies, when it doesn't, and how to adapt it when the standard approach isn't enough.


    08_llm_training/
    ├── 00_order.py
    ├── 01_transformer_architecture_fundamentals.py
    ├── 02_tokenization_vocabulary_bpe.py
    ├── 03_positional_encodings_rope_alibi.py
    ├── 04_attention_variants_mha_mqa_gqa.py
    ├── 05_modern_architectures_llama_mistral_phi.py
    ├── 06_training_loop_fundamentals.py
    ├── 07_optimizers_adam_adamw.py
    ├── 08_lr_schedules_warmup_cosine.py
    ├── 09_gradient_clipping_stability.py
    ├── 10_mixed_precision_bf16_fp16.py
    ├── 11_Multi_GPU_Training.py
    ├── 12_mlsystemdesign.py
    ├── 13_dataset_preparation_preprocessing.py
    ├── 14_data_loading_batching_packing.py
    ├── 15_data_mixing_curriculum.py
    ├── 16_activation_checkpointing.py
    ├── 17_gradient_accumulation.py
    ├── 18_data_parallel_ddp.py
    ├── 19_tensor_parallelism.py
    ├── 20_pipeline_parallelism.py
    ├── 21_sequence_parallelism.py
    ├── 22_expert_parallelism_moe.py
    ├── 23_3D_parallelism_megatron_deepspeed.py
    ├── 24_zero_optimizer_stages.py
    ├── 25_cpu_offloading.py
    ├── 26_supervised_finetuning_sft.py
    ├── 27_peft_lora_qlora.py
    ├── 28_rlhf_reward_modeling_ppo.py
    ├── 29_dpo_direct_preference_optimization.py
    ├── 30_speculative_decoding.py
    ├── 31_inference_optimization_serving.py
    ├── 32_quantization_inference_gptq_awq.py
    ├── 33_checkpoint_management.py
    ├── 34_experiment_tracking_wandb_mlflow.py
    ├── 35_singlegpu_llm_training.py
    ├── 36_multigpu_llm_training.py
    ├── 37_training_stability_debugging.py
    └── 38_evaluation_benchmarking.py

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


# """
# LLM Training — Module Registry
# ===============================
#
# This file is the single source of truth for the ordering, metadata, and
# discovery of every topic module in the llm_training/ collection.
#
# app.py (or any host) imports ORDER_LIST to know which modules exist, what
# order to show them in, and what display metadata to use in the sidebar /
# navigation before the individual module files are even imported.
#
# Usage:
#     from llm_training.00_order import ORDER_LIST, get_module
#
#     for entry in ORDER_LIST:
#         print(entry["index"], entry["display_name"], entry["icon"])
# """
#
# import importlib
# import os
#
# # ─────────────────────────────────────────────────────────────────────────────
# # MODULE REGISTRY
# # Each entry maps to exactly one .py file in this directory.
# # Keys
# #   index        : zero-padded string used in the filename prefix
# #   module_file  : bare filename (no .py extension) relative to this package
# #   display_name : short label shown in navigation / sidebar
# #   icon         : single emoji shown next to the label
# #   subtitle     : one-line description shown below the title
# #   category     : broad grouping for optional section headers
# # ─────────────────────────────────────────────────────────────────────────────
#
# ORDER_LIST = [
#
#     # ── FOUNDATIONS ────────────────────────────────────────────────────────
#     {
#         "index":        "01",
#         "module_file":  "01_transformer_architecture_fundamentals",
#         "display_name": "01 · Transformer Architecture",
#         "icon":         "🏗️",
#         "subtitle":     "Encoder, Decoder, and the Full Forward Pass",
#         "category":     "Foundations",
#     },
#     {
#         "index":        "02",
#         "module_file":  "02_tokenization_vocabulary_bpe",
#         "display_name": "02 · Tokenization & BPE",
#         "icon":         "✂️",
#         "subtitle":     "How Text Becomes Numbers",
#         "category":     "Foundations",
#     },
#     {
#         "index":        "03",
#         "module_file":  "03_positional_encodings_rope_alibi",
#         "display_name": "03 · Positional Encodings",
#         "icon":         "📍",
#         "subtitle":     "Sinusoidal, RoPE, and ALiBi",
#         "category":     "Foundations",
#     },
#     {
#         "index":        "04",
#         "module_file":  "04_attention_variants_mha_mqa_gqa",
#         "display_name": "04 · Attention Variants",
#         "icon":         "👁️",
#         "subtitle":     "MHA, MQA, and Grouped-Query Attention",
#         "category":     "Foundations",
#     },
#     {
#         "index":        "05",
#         "module_file":  "05_modern_architectures_llama_mistral_phi",
#         "display_name": "05 · Modern Architectures",
#         "icon":         "🦙",
#         "subtitle":     "LLaMA, Mistral, and Phi Design Choices",
#         "category":     "Foundations",
#     },
#
#     # ── TRAINING LOOP ──────────────────────────────────────────────────────
#     {
#         "index":        "06",
#         "module_file":  "06_training_loop_fundamentals",
#         "display_name": "06 · Training Loop",
#         "icon":         "🔄",
#         "subtitle":     "Forward Pass, Loss, Backward Pass, Update",
#         "category":     "Training Loop",
#     },
#     {
#         "index":        "07",
#         "module_file":  "07_optimizers_adam_adamw",
#         "display_name": "07 · Optimizers: Adam & AdamW",
#         "icon":         "⚙️",
#         "subtitle":     "Adaptive Moment Estimation and Weight Decay",
#         "category":     "Training Loop",
#     },
#     {
#         "index":        "08",
#         "module_file":  "08_lr_schedules_warmup_cosine",
#         "display_name": "08 · LR Schedules",
#         "icon":         "📈",
#         "subtitle":     "Warmup, Cosine Decay, and WSD",
#         "category":     "Training Loop",
#     },
#     {
#         "index":        "09",
#         "module_file":  "09_gradient_clipping_stability",
#         "display_name": "09 · Gradient Clipping",
#         "icon":         "✂️",
#         "subtitle":     "Preventing Exploding Gradients",
#         "category":     "Training Loop",
#     },
#     {
#         "index":        "10",
#         "module_file":  "10_mixed_precision_bf16_fp16",
#         "display_name": "10 · Mixed Precision",
#         "icon":         "🔢",
#         "subtitle":     "FP16, BF16, and Loss Scaling",
#         "category":     "Training Loop",
#     },
#
#     # ── HARDWARE & SYSTEMS ─────────────────────────────────────────────────
#     {
#         "index":        "11",
#         "module_file":  "11_Multi_GPU_Training",
#         "display_name": "11 · Multi-GPU Training",
#         "icon":         "🖥️",
#         "subtitle":     "NCCL, All-Reduce, and GPU Topology",
#         "category":     "Hardware & Systems",
#     },
#     {
#         "index":        "12",
#         "module_file":  "12_mlsystemdesign",
#         "display_name": "12 · ML System Design",
#         "icon":         "🏛️",
#         "subtitle":     "End-to-End Training Infrastructure",
#         "category":     "Hardware & Systems",
#     },
#
#     # ── DATA ───────────────────────────────────────────────────────────────
#     {
#         "index":        "13",
#         "module_file":  "13_dataset_preparation_preprocessing",
#         "display_name": "13 · Dataset Preparation",
#         "icon":         "🗂️",
#         "subtitle":     "Cleaning, Deduplication, and Filtering",
#         "category":     "Data",
#     },
#     {
#         "index":        "14",
#         "module_file":  "14_data_loading_batching_packing",
#         "display_name": "14 · Data Loading & Packing",
#         "icon":         "📦",
#         "subtitle":     "Efficient DataLoaders, Sequence Packing",
#         "category":     "Data",
#     },
#     {
#         "index":        "15",
#         "module_file":  "15_data_mixing_curriculum",
#         "display_name": "15 · Data Mixing & Curriculum",
#         "icon":         "🎓",
#         "subtitle":     "Domain Weights and Curriculum Strategies",
#         "category":     "Data",
#     },
#
#     # ── MEMORY EFFICIENCY ──────────────────────────────────────────────────
#     {
#         "index":        "16",
#         "module_file":  "16_activation_checkpointing",
#         "display_name": "16 · Activation Checkpointing",
#         "icon":         "💾",
#         "subtitle":     "Trading Compute for Memory",
#         "category":     "Memory Efficiency",
#     },
#     {
#         "index":        "17",
#         "module_file":  "17_gradient_accumulation",
#         "display_name": "17 · Gradient Accumulation",
#         "icon":         "🪣",
#         "subtitle":     "Simulating Large Batches on Small Memory",
#         "category":     "Memory Efficiency",
#     },
#
#     # ── PARALLELISM ────────────────────────────────────────────────────────
#     {
#         "index":        "18",
#         "module_file":  "18_data_parallel_ddp",
#         "display_name": "18 · Data Parallelism & DDP",
#         "icon":         "🔀",
#         "subtitle":     "DistributedDataParallel in PyTorch",
#         "category":     "Parallelism",
#     },
#     {
#         "index":        "19",
#         "module_file":  "19_tensor_parallelism",
#         "display_name": "19 · Tensor Parallelism",
#         "icon":         "🧩",
#         "subtitle":     "Splitting Weight Matrices Across GPUs",
#         "category":     "Parallelism",
#     },
#     {
#         "index":        "20",
#         "module_file":  "20_pipeline_parallelism",
#         "display_name": "20 · Pipeline Parallelism",
#         "icon":         "🚰",
#         "subtitle":     "Micro-batches, Bubbles, and Schedules",
#         "category":     "Parallelism",
#     },
#     {
#         "index":        "21",
#         "module_file":  "21_sequence_parallelism",
#         "display_name": "21 · Sequence Parallelism",
#         "icon":         "🔗",
#         "subtitle":     "Distributing the Sequence Dimension",
#         "category":     "Parallelism",
#     },
#     {
#         "index":        "22",
#         "module_file":  "22_expert_parallelism_moe",
#         "display_name": "22 · Expert Parallelism & MoE",
#         "icon":         "🧠",
#         "subtitle":     "Mixture of Experts and Routing",
#         "category":     "Parallelism",
#     },
#     {
#         "index":        "23",
#         "module_file":  "23_3D_parallelism_megatron_deepspeed",
#         "display_name": "23 · 3D Parallelism",
#         "icon":         "🌐",
#         "subtitle":     "Megatron-LM and DeepSpeed Combined",
#         "category":     "Parallelism",
#     },
#
#     # ── OPTIMIZER STATE & OFFLOADING ───────────────────────────────────────
#     {
#         "index":        "24",
#         "module_file":  "24_zero_optimizer_stages",
#         "display_name": "24 · ZeRO Optimizer Stages",
#         "icon":         "🅾️",
#         "subtitle":     "ZeRO-1, ZeRO-2, ZeRO-3 Memory Partitioning",
#         "category":     "Optimizer State & Offloading",
#     },
#     {
#         "index":        "25",
#         "module_file":  "25_cpu_offloading",
#         "display_name": "25 · CPU Offloading",
#         "icon":         "💿",
#         "subtitle":     "Moving States to CPU/NVMe",
#         "category":     "Optimizer State & Offloading",
#     },
#
#     # ── FINE-TUNING & ALIGNMENT ────────────────────────────────────────────
#     {
#         "index":        "26",
#         "module_file":  "26_supervised_finetuning_sft",
#         "display_name": "26 · Supervised Fine-Tuning",
#         "icon":         "🎯",
#         "subtitle":     "Instruction Tuning and Chat Templates",
#         "category":     "Fine-Tuning & Alignment",
#     },
#     {
#         "index":        "27",
#         "module_file":  "27_peft_lora_qlora",
#         "display_name": "27 · PEFT: LoRA & QLoRA",
#         "icon":         "🪡",
#         "subtitle":     "Parameter-Efficient Fine-Tuning Methods",
#         "category":     "Fine-Tuning & Alignment",
#     },
#     {
#         "index":        "28",
#         "module_file":  "28_rlhf_reward_modeling_ppo",
#         "display_name": "28 · RLHF & PPO",
#         "icon":         "🏆",
#         "subtitle":     "Reward Modeling and Policy Optimization",
#         "category":     "Fine-Tuning & Alignment",
#     },
#     {
#         "index":        "29",
#         "module_file":  "29_dpo_direct_preference_optimization",
#         "display_name": "29 · DPO",
#         "icon":         "🎖️",
#         "subtitle":     "Direct Preference Optimization",
#         "category":     "Fine-Tuning & Alignment",
#     },
#
#     # ── INFERENCE ──────────────────────────────────────────────────────────
#     {
#         "index":        "30",
#         "module_file":  "30_speculative_decoding",
#         "display_name": "30 · Speculative Decoding",
#         "icon":         "🔭",
#         "subtitle":     "Draft Models and Parallel Verification",
#         "category":     "Inference",
#     },
#     {
#         "index":        "31",
#         "module_file":  "31_inference_optimization_serving",
#         "display_name": "31 · Inference & Serving",
#         "icon":         "🚀",
#         "subtitle":     "KV Cache, Continuous Batching, vLLM",
#         "category":     "Inference",
#     },
#     {
#         "index":        "32",
#         "module_file":  "32_quantization_inference_gptq_awq",
#         "display_name": "32 · Quantization",
#         "icon":         "🗜️",
#         "subtitle":     "GPTQ, AWQ, and INT4/INT8 Inference",
#         "category":     "Inference",
#     },
#
#     # ── TOOLING & OPERATIONS ───────────────────────────────────────────────
#     {
#         "index":        "33",
#         "module_file":  "33_checkpoint_management",
#         "display_name": "33 · Checkpoint Management",
#         "icon":         "💽",
#         "subtitle":     "Saving, Resuming, and Sharding Checkpoints",
#         "category":     "Tooling & Operations",
#     },
#     {
#         "index":        "34",
#         "module_file":  "34_experiment_tracking_wandb_mlflow",
#         "display_name": "34 · Experiment Tracking",
#         "icon":         "📊",
#         "subtitle":     "W&B, MLflow, and TensorBoard",
#         "category":     "Tooling & Operations",
#     },
#
#     # ── END-TO-END RECIPES ─────────────────────────────────────────────────
#     {
#         "index":        "35",
#         "module_file":  "35_singlegpu_llm_training",
#         "display_name": "35 · Single-GPU LLM Training",
#         "icon":         "1️⃣",
#         "subtitle":     "Full Recipe: One GPU, Minimal Dependencies",
#         "category":     "End-to-End Recipes",
#     },
#     {
#         "index":        "36",
#         "module_file":  "36_multigpu_llm_training",
#         "display_name": "36 · Multi-GPU LLM Training",
#         "icon":         "🔢",
#         "subtitle":     "Full Recipe: DDP + ZeRO + Mixed Precision",
#         "category":     "End-to-End Recipes",
#     },
#     {
#         "index":        "37",
#         "module_file":  "37_training_stability_debugging",
#         "display_name": "37 · Training Stability & Debugging",
#         "icon":         "🐛",
#         "subtitle":     "Loss Spikes, NaN Gradients, and Fixes",
#         "category":     "End-to-End Recipes",
#     },
#     {
#         "index":        "38",
#         "module_file":  "38_evaluation_benchmarking",
#         "display_name": "38 · Evaluation & Benchmarking",
#         "icon":         "📏",
#         "subtitle":     "Perplexity, MMLU, and Harness Evals",
#         "category":     "End-to-End Recipes",
#     },
# ]
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # CATEGORY ORDER  (controls section-header sequence in the sidebar)
# # ─────────────────────────────────────────────────────────────────────────────
#
# CATEGORY_ORDER = [
#     "Foundations",
#     "Training Loop",
#     "Hardware & Systems",
#     "Data",
#     "Memory Efficiency",
#     "Parallelism",
#     "Optimizer State & Offloading",
#     "Fine-Tuning & Alignment",
#     "Inference",
#     "Tooling & Operations",
#     "End-to-End Recipes",
# ]
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────────────────────
#
# def get_module(module_file: str):
#     """
#     Dynamically import and return the Python module object for a given file stem.
#
#     Example:
#         mod = get_module("01_transformer_architecture_fundamentals")
#         content = mod.get_content()
#     """
#     package = __name__.rsplit(".", 1)[0] if "." in __name__ else ""
#     full_name = f"{package}.{module_file}" if package else module_file
#     return importlib.import_module(full_name)
#
#
# def get_all_content():
#     """
#     Import every module in ORDER_LIST and return a list of (entry, content) pairs.
#     Useful for bulk indexing or search.
#
#     Skips modules that fail to import (logs a warning instead of crashing).
#     """
#     import warnings
#     results = []
#     for entry in ORDER_LIST:
#         try:
#             mod = get_module(entry["module_file"])
#             results.append((entry, mod.get_content()))
#         except Exception as exc:
#             warnings.warn(
#                 f"[00_order] Could not load '{entry['module_file']}': {exc}",
#                 stacklevel=2,
#             )
#     return results
#
#
# def modules_by_category():
#     """
#     Return an OrderedDict mapping category name → list of ORDER_LIST entries.
#     Preserves the section order defined in CATEGORY_ORDER.
#     """
#     from collections import OrderedDict
#     buckets = OrderedDict((cat, []) for cat in CATEGORY_ORDER)
#     for entry in ORDER_LIST:
#         cat = entry.get("category", "Uncategorised")
#         buckets.setdefault(cat, []).append(entry)
#     return buckets
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # QUICK SANITY CHECK  (run this file directly to verify the registry)
# # ─────────────────────────────────────────────────────────────────────────────
#
# if __name__ == "__main__":
#     print(f"{'─'*60}")
#     print(f"  LLM Training Module Registry — {len(ORDER_LIST)} modules")
#     print(f"{'─'*60}")
#
#     current_cat = None
#     for entry in ORDER_LIST:
#         cat = entry.get("category", "")
#         if cat != current_cat:
#             print(f"\n  ── {cat} ──")
#             current_cat = cat
#         print(f"  {entry['icon']}  [{entry['index']}]  {entry['display_name']}")
#         print(f"            {entry['subtitle']}")
#
#     print(f"\n{'─'*60}")
#     print(f"  {len(CATEGORY_ORDER)} categories: {', '.join(CATEGORY_ORDER)}")
#     print(f"{'─'*60}")