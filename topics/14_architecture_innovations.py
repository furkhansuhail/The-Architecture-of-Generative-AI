"""Module: 14 · Architecture Innovations"""

DISPLAY_NAME = "14 · Architecture Innovations"
ICON         = "⚙️"
SUBTITLE     = "MoE, Mamba and long-context scaling — beyond the vanilla Transformer"

THEORY = """
## 14 · Architecture Innovations

The vanilla Transformer (Vaswani et al. 2017) revolutionised AI, but its O(n²) attention
complexity, dense parameter usage, and fixed-length context window created bottlenecks
at scale.  This module traces the architectural innovations that address these limits —
from efficient attention to sparse experts to state-space models — explaining the *why*
before the *how* at each step.

---

## 1 · The Vanilla Transformer's Bottlenecks

### 1.1 Recap: The Standard Transformer Block

Each Transformer layer applies:
```
h'  = x + MultiHeadAttention(LayerNorm(x))
h   = h' + FFN(LayerNorm(h'))
```

where FFN is a 2-layer MLP:  `FFN(x) = W₂ · ReLU(W₁x + b₁) + b₂`

For a model with hidden dim `d` and FFN expansion ratio `r=4`:
- Attention: `4d²` parameters per layer (Q, K, V, O projections)
- FFN:        `8d²` parameters per layer (W₁ ∈ ℝ^{4d×d}, W₂ ∈ ℝ^{d×4d})

**FFN dominates** — in a 70B parameter model, ~⅔ of all parameters are in FFN layers.

### 1.2 The Three Core Bottlenecks

**Bottleneck 1 — Quadratic Attention:**
```
Attention cost = O(n² · d)    where n = sequence length
```
For n=128K tokens, attention alone requires ~130 billion operations per layer.
Memory also scales O(n²) to store the attention matrix — 128K² = 16 billion floats.

**Bottleneck 2 — Dense Computation:**
Every input token activates every parameter.  For a 70B model, each token requires
70B multiply-adds regardless of token content — computationally wasteful for diverse
inputs that might only require a small subset of knowledge.

**Bottleneck 3 — Fixed Context Window:**
Standard attention attends over all tokens in the context — but the KV-cache grows
linearly with context length, making very long contexts (1M+ tokens) prohibitively
expensive in memory during inference.

---

## 2 · Efficient Attention Mechanisms

### 2.1 The KV-Cache and Inference Memory

During autoregressive generation, the model generates one token at a time.  At step t:
- The query is the new token's embedding.
- Keys and values of ALL previous tokens must be recomputed — or cached.

The **KV-cache** stores all previous K and V tensors:
```
Memory per token = 2 · n_layers · n_heads · d_head · bytes_per_element
For GPT-3 (96 layers, 96 heads, 128 d_head, fp16):
= 2 × 96 × 96 × 128 × 2 = 4.7 MB per token
For 8K context: 4.7MB × 8000 = 37.6 GB — just for the KV-cache!
```

### 2.2 Multi-Query Attention (MQA)

**MQA (Shazeer 2019)**: All attention heads **share a single K and V projection**
while each has its own Q projection.

```
Standard MHA:  Q_h = x W_Q^h,  K_h = x W_K^h,  V_h = x W_V^h   (h = 1…H)
MQA:           Q_h = x W_Q^h,  K   = x W_K,     V   = x W_V     (K,V shared)
```

**Effect on KV-cache:**
```
MHA KV-cache:  n_layers × 2 × H × d_head × n_tokens
MQA KV-cache:  n_layers × 2 × 1 × d_head × n_tokens   ← H× smaller!
```

For GPT-3-scale: KV-cache shrinks from 37.6 GB → ~0.4 GB for 8K context.
Quality impact: marginal — the V heads benefit least from separation; K separation
is more important but still reducible.

### 2.3 Grouped Query Attention (GQA)

**GQA (Ainslie et al. 2023)**: Interpolates between MHA and MQA by grouping H query
heads into G groups (G < H), with one K and V per group.

```
G = H → Multi-Head Attention (MHA)   — full quality, large KV-cache
G = 1 → Multi-Query Attention (MQA)  — minimal KV-cache, slight quality drop
G ∈ (1, H) → Grouped Query Attention — best trade-off
```

Used in: **LLaMA-2/3** (G=8), **Mistral** (G=8), **Gemma**, **Falcon**.

Mathematical formulation for group g with queries Q_{g,1}, …, Q_{g,H/G}:
```
Attn_h(x) = softmax(Q_{g,h} K_g^T / √d) V_g
```

### 2.4 FlashAttention

FlashAttention (Dao et al. 2022) doesn't change the **mathematical result** of attention
— it changes **how it's computed** by exploiting GPU memory hierarchy.

**GPU Memory Hierarchy:**
```
HBM (High Bandwidth Memory): 40-80 GB, ~2 TB/s bandwidth
SRAM (on-chip):               ~20 MB,   ~20 TB/s bandwidth (10× faster)
```

Standard attention materialises the full N×N attention matrix in HBM:
```
Standard: HBM reads/writes = O(N² + Nd)   ← N² dominates
```

FlashAttention uses **tiling** — computes attention in small blocks that fit in SRAM:
```
FlashAttention: HBM reads/writes = O(N²d / M)   where M = SRAM size
```
For M >> d: this is dramatically fewer HBM accesses → **2-4× wall-clock speedup**.

Key insight: the softmax over the full row can be computed incrementally using the
**online softmax** trick (maintaining running max and normalisation constant):

```
m_i = max(m_{i-1}, rowmax(S_i))
ℓ_i = e^{m_{i-1} - m_i} · ℓ_{i-1} + rowsum(e^{S_i - m_i})
```

FlashAttention-2 (2023): further optimises work partitioning across warps.
FlashAttention-3 (2024): overlaps computation with memory transfers on H100s.

### 2.5 Sliding Window Attention (SWA)

**Longformer / Mistral**: Each token only attends to the `w` nearest tokens:
```
Attn(x_i) = softmax({x_j : |i-j| ≤ w/2} W_Q x_i · W_K x_j^T) W_V x_j
```

Complexity drops from O(n²) → O(n·w).

For w=4096 and n=32768: 8× speedup over full attention.

**Information propagates globally** through stacking layers: after L layers with window
w, token i can attend to tokens up to `L×w` positions away.

Mistral 7B uses w=4096 (sliding) in lower layers for local context and full attention
in upper layers for global reasoning — a hybrid approach.

### 2.6 ALiBi and RoPE — Positional Embeddings for Length Generalisation

**Problem:** Models trained at context length n_train often fail on longer sequences.

**ALiBi (Press et al. 2022):** Adds a linear bias to attention scores based on distance:
```
Attn(i,j) = x_i W_Q · (x_j W_K)^T / √d  −  m_h · (i − j)
```
where `m_h` is a per-head slope (geometric sequence: 2^{−8/H}, 2^{−16/H}, …).
The penalty grows linearly with distance → recency bias without learned embeddings.
Generalises to longer contexts at inference even when trained on shorter ones.

**RoPE (Su et al. 2021 — Rotary Position Embedding):**
Encodes absolute position by rotating the Q and K vectors:
```
Q_m = R_m Q,   K_n = R_n K
where R_θ = block-diag(R(mθ₁), R(mθ₂), …)
and R(mθ) = [[cos(mθ), −sin(mθ)], [sin(mθ), cos(mθ)]]
```

The dot product Q_m · K_n = f(Q, K, m−n) depends only on **relative position** (m−n),
giving relative position encoding through absolute position rotation.

**RoPE scaling (YaRN, LongRoPE):** Extends pre-trained models to longer contexts by
interpolating or extrapolating the rotation frequencies — used to extend LLaMA-2's
4K context to 128K in Code Llama and 1M+ in LLaMA-3.1.

---

## 3 · Mixture of Experts (MoE)

### 3.1 The Dense-Sparse Trade-off

Dense models: every parameter is used for every token → wasteful for diverse tasks.

**Key insight**: A 70B parameter model might only need 7B parameters to answer a
specific maths question, but different 7B parameters to write a poem.  Can we build
a 70B model where only 7B parameters activate per token?

### 3.2 MoE Architecture

Replace each dense FFN layer with `E` expert FFN networks and a **router** (gating
network) that selects `K` experts per token:

```
MoE-FFN(x) = Σ_{k ∈ TopK(G(x))} G_k(x) · Expert_k(x)

where:
G(x)    = Softmax(x W_g)             ← router logits, W_g ∈ ℝ^{d × E}
TopK(.) → select K largest values
G_k(x)  = softmax(top-K logits)_k   ← renormalised gate weight for expert k
Expert_k(x) = W₂_k · ReLU(W₁_k x)  ← standard FFN
```

**Parameter count:**
```
Dense FFN:   8d²  parameters
MoE FFN:     E × 8d²  total,  but only K × 8d²  active per token
```

Typical: E=8 experts, K=2 active → same compute as 2×FFN, but 8× the parameters.
**Total params ↑, Active params constant** — inference cost ≈ dense at fraction of size.

### 3.3 The Load Balancing Problem

Without constraints, the router collapses — it learns to route everything to a few
"elite" experts and ignores the rest.  This is called **expert collapse**.

**Auxiliary load balancing loss:**

```
L_balance = α · E · Σ_{i=1}^{E} f_i · P_i

where:
f_i = fraction of tokens routed to expert i  (computed without gradients)
P_i = mean of router probabilities for expert i  (differentiable)
```

Minimising L_balance pushes `f_i → 1/E` (uniform expert utilisation).
`α` is a small coefficient (typically 0.01–0.1) to avoid dominating the main loss.

**Expert Capacity:**
```
Capacity = (tokens_per_batch / E) × capacity_factor
```

If expert i receives more than Capacity tokens, excess tokens are **dropped** (or
handled with a fallback mechanism). Capacity factor > 1 provides overflow buffer.

### 3.4 Token Choice vs Expert Choice Routing

**Token choice (standard):** Each token selects its top-K experts.
Risk: some experts overloaded, some underutilised.

**Expert choice (Zhou et al. 2022):** Each expert selects its top-C tokens.
Guarantees perfect load balancing.
Risk: some tokens are processed by no expert (dropped).

**Modern hybrid:** Buffer tokens, auxiliary loss, and expert choice are all combined
in systems like Switch Transformer, Mixtral, and DeepSeek-MoE.

### 3.5 Switch Transformer

Switch Transformer (Fedus et al. 2021) simplifies MoE by using **K=1** (one expert
per token):

```
Switch Router:  i* = argmax(W_g x)     (just the top-1 expert)
```

Benefits:
- Half the router computation vs K=2.
- Simpler implementation.
- Still achieves 7× speedup over T5-11B at equivalent compute.

Trained a 1.6T parameter model with 2048 experts at 1/6 the cost of a dense T5-11B.

### 3.6 Mixtral 8×7B

Mixtral (Mistral AI, 2023): 8 experts, K=2, standard FFN structure.
- Total parameters: ~46.7B
- Active parameters per forward pass: ~12.9B (2 experts of 7B-scale each)
- Performance: matches or exceeds LLaMA-2-70B at 1/5 the inference cost.

**DeepSeek-MoE (2024)**: Introduces **fine-grained experts** (E=64 small experts,
K=6 active) instead of few large experts.  More granular specialisation, better
load balancing.  Shared expert (always active) + routed experts.

### 3.7 MoE in Practice: Training Challenges

- **Communication overhead** (in distributed training): different experts live on
  different GPUs → all-to-all communication for routing is expensive.
- **Expert specialisation** (beneficial): experts develop interpretable specialties
  (e.g., expert for code, expert for mathematical reasoning).
- **Batch size sensitivity**: small batches → uneven load → more drops.
  MoE needs large batch training.

---

## 4 · State Space Models (SSMs) and Mamba

### 4.1 Motivation: The Recurrence-Attention Trade-off

| Property         | RNN/LSTM        | Transformer     | Ideal            |
|-----------------|-----------------|-----------------|------------------|
| Training        | Sequential      | Parallelisable  | Parallelisable   |
| Inference       | O(1) per token  | O(n) per token  | O(1) per token   |
| Long-range deps | Vanishing grad  | Full attention  | Linear in n      |
| Memory          | O(state size)   | O(n·d) KV-cache | O(state size)    |

Can we combine the best of both worlds?  State Space Models (SSMs) achieve this.

### 4.2 Continuous State Space Models

A linear time-invariant (LTI) system in continuous time:
```
h'(t) = A h(t) + B u(t)      ← state equation
y(t)  = C h(t) + D u(t)      ← output equation
```

where:
- `u(t)` ∈ ℝ: input signal
- `h(t)` ∈ ℝᴺ: hidden state (N is state size)
- `y(t)` ∈ ℝ: output
- `A` ∈ ℝ^{N×N}: state transition matrix (how state evolves)
- `B` ∈ ℝ^{N×1}: input projection
- `C` ∈ ℝ^{1×N}: output projection

### 4.3 Discretisation

For sequence models, discretise the continuous system with step size Δ using **ZOH**
(zero-order hold):

```
Ā = exp(ΔA)
B̄ = (ΔA)⁻¹(exp(ΔA) − I) · ΔB  ≈  ΔB   (for small Δ)

Discrete recurrence:
h_t = Ā h_{t-1} + B̄ u_t
y_t = C h_t
```

This is a linear recurrence — computable as a RNN in O(N) per step.
But it's also computable as a **global convolution** (for training):

```
y = K * u,   where K = (CB̄, CĀB̄, CĀ²B̄, …)  ← SSM convolution kernel
```

This is the key duality: **recurrence for inference, convolution for training**.

### 4.4 S4 and HiPPO Initialisation

Vanilla SSMs fail to capture long-range dependencies because `Aᵏ → 0` as k grows
(eigenvalues of A < 1).

**HiPPO (High-order Polynomial Projection Operator)** initialises A as:
```
A_{nk} = −(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
        = −(n+1)                        if n = k
        = 0                             if n < k
```

This projects the input history onto a basis of Legendre polynomials — optimal for
remembering the past.  The resulting S4 (Gu et al. 2021) achieves near-perfect scores
on Long Range Arena, surpassing Transformers on tasks requiring 4K–16K range.

### 4.5 Mamba (Selective State Spaces)

Mamba (Gu & Dao, 2023) identifies the key weakness of S4: the matrices A, B, C are
**input-independent** (LTI).  This means the model cannot selectively focus on or
ignore specific inputs — it treats all tokens equally.

**Mamba's insight:** Make B, C, and Δ **functions of the input**:
```
B_t = s_B(x_t) = Linear_B(x_t)
C_t = s_C(x_t) = Linear_C(x_t)
Δ_t = softplus(s_Δ(x_t)) = softplus(Linear_Δ(x_t))
```

Now the SSM parameters change at every timestep based on the input — **selective**
state space model.  This breaks the convolution parallelism, but Mamba introduces a
**hardware-aware parallel scan** algorithm that restores training efficiency.

**Selective mechanism intuition:**
- Large Δ_t → slow state transition → current token has large influence → "focus"
- Small Δ_t → fast state transition → current token has small influence → "forget"

B and C selectivity allow the model to write relevant information to state and read
relevant information from state.

### 4.6 Mamba Architecture

Mamba replaces the Transformer's attention + FFN with a single SSM block:

```
Input x ∈ ℝ^{L × d}
    ↓ Linear(d → 2·d_inner)  [expand]
   split into x_path and z_path
x_path:
    → Conv1d (local context, width 4)
    → SiLU activation
    → SSM (selective state space)
z_path:
    → SiLU activation (gate)
Element-wise product: SSM_out ⊙ gate
    ↓ Linear(d_inner → d)  [project back]
```

The **SiLU gate** (Sigmoid Linear Unit) acts like a content-based filter: the gate
decides which parts of the SSM output are passed through.

### 4.7 Mamba Computational Complexity

```
Training:  O(L · d · N)    (parallel scan, L = sequence length, N = state dim)
Inference: O(N)            (recurrence, constant per step regardless of context!)

Transformer (for comparison):
Training:  O(L² · d)
Inference: O(L · d)        (KV-cache grows with L)
```

For L=1M tokens:
- Transformer attention: 1M² = 1T operations
- Mamba: 1M × d × N ≈ 1M × 1024 × 16 = 16B operations — 60× less

### 4.8 Mamba-2 and Hybrid Models

**Mamba-2 (2024)**: Connects SSMs to attention by showing that SSMs can be written as
a form of **structured masked attention** — enabling GPU-optimised implementations.
Introduces state space duality (SSD).

**Hybrid SSM-Attention (Jamba, Zamba, etc.)**: Interleave Mamba and Attention layers:
- Mamba layers for cheap long-range sequence modelling.
- Sparse attention layers every k layers for precise information retrieval.
- Achieves better quality than pure Mamba with much lower cost than pure Transformer.

---

## 5 · Long-Context Scaling

### 5.1 The Challenge of Very Long Contexts

Modern tasks require very long contexts:
- RAG over entire codebases: 100K–1M tokens
- Legal document analysis: entire contracts
- Multi-turn conversations with long history
- Long video understanding

### 5.2 Position Interpolation (PI)

Extending a model trained at context length L_train to L_test > L_train:

**Naive extrapolation**: Apply RoPE at positions > L_train → **catastrophic failure**.
The model has never seen these rotation angles during training.

**Position Interpolation (Chen et al. 2023)**: Scale all positions by L_train / L_test:
```
Position p → p · (L_train / L_test)
```
Now positions stay within the trained range [0, L_train] but are more densely packed.
A short fine-tuning (1000 steps) on long-context data restores full performance.

### 5.3 YaRN (Yet Another RoPE Extension)

YaRN (Peng et al. 2023) improves on PI by applying different scaling to different
frequency components of RoPE:

```
For each dimension i:
- High-frequency (short wavelength): interpolate (scale down)
- Low-frequency (long wavelength): extrapolate (no scaling needed)
- Medium: blend (linear interpolation between the two)
```

Wavelength λ_i = 2π / θᵢ where θᵢ = θ_base^{-2i/d}.

Additionally, YaRN multiplies attention scores by a **temperature factor** t to
compensate for the reduced magnitude of long-context attention scores:
```
Attention = softmax(QKᵀ / (√d · t))
```

YaRN extends LLaMA-2-7B from 4K → 128K context with only 400 training steps.

### 5.4 Ring Attention

For contexts that don't fit in a single GPU's memory (1M+ tokens):

**Ring Attention (Liu et al. 2023)**: Split the sequence across devices in a ring.
Each device holds a chunk of the sequence.  Devices pass K,V chunks around the ring
while computing local attention blocks — using FlashAttention locally.

```
Device 1: Q_1, K_1, V_1  →  Device 2: Q_2, K_2, V_2  →  ... → Device 1
During each ring step, each device computes attention with a different K,V shard.
After N steps (N = number of devices), each device has computed its full output.
```

This enables training on sequences with **millions of tokens** across many GPUs with
no additional communication overhead vs standard distributed training.

### 5.5 Sparse Attention Patterns

For tasks where not all tokens need to attend to all others:

**Strided attention (Sparse Transformer)**: Attend every s-th token.
**Local + global**: Some tokens have global receptive field (BigBird).
**Learned sparsity**: Use router to select attended tokens (Routing Transformer).

---

## 6 · Architectural Efficiency Techniques

### 6.1 Mixture of Depths (MoD)

**MoD (Raposo et al. 2024)**: Instead of applying every layer to every token, route
tokens to layers adaptively:

```
For each layer ℓ:
  - Router decides which tokens need this layer
  - Unselected tokens skip the layer via residual
  - Selected tokens go through full MHA + FFN
```

If only C/n tokens (C = capacity) are processed at each layer:
- FLOP reduction: proportional to 1 − C/n
- For C/n = 0.5: 50% FLOP reduction with minimal quality loss

### 6.2 Parameter Sharing and Recycling

**ALBERT**: Shares weights across all Transformer layers (same weights used L times).
- Parameters: O(d²) instead of O(L·d²)
- Computation unchanged — still L passes through the shared weights
- Works surprisingly well for BERT-scale models

**Mixture of Layers**: Different tokens use different subsets of layers.

### 6.3 Speculative Decoding

LLM inference is memory-bandwidth bound: the bottleneck is loading model weights,
not arithmetic.  For a 70B model: ~140 GB of weights loaded per generated token.

**Speculative decoding (Leviathan et al. 2022)**:
1. A small **draft model** (e.g., 7B) generates γ candidate tokens cheaply.
2. The large **target model** verifies all γ+1 tokens in **one forward pass** (parallel).
3. If draft tokens are accepted, we've generated γ tokens for the cost of ~1.
4. On mismatch, fall back to the target model's distribution.

**Acceptance rate** depends on draft-target agreement:
- If draft matches target distribution, accept all γ tokens.
- Expected speedup: `γ · α / (1 + γ · (1−α))` where α = acceptance rate.
- For γ=4, α=0.8: ~3.2× speedup with **identical** output distribution as target model.

### 6.4 Continuous Batching / Iteration-Level Scheduling

Standard batching: all sequences in a batch finish together → GPU idles waiting for
longest sequences.

**Continuous batching (Orca, 2022)**: Insert new requests and remove finished ones
at every generation step (iteration level).  GPU utilisation approaches 100%.

Used in: vLLM, TGI, TensorRT-LLM.

### 6.5 PagedAttention and vLLM

KV-cache fragments GPU memory — standard allocation wastes 20–30% due to fragmentation.

**PagedAttention (Kwon et al. 2022)**: Manages KV-cache like OS virtual memory:
- Divide KV-cache into fixed-size **pages** (blocks of tokens).
- A logical sequence maps to non-contiguous physical pages.
- Different sequences can **share** KV pages (prefix caching, parallel sampling).

Result: near-zero memory waste, 24× higher throughput vs naive KV-cache management.

---

## 7 · Modern Architecture Zoo

### 7.1 Mistral 7B

Key innovations:
- **GQA** (8 KV heads instead of 32) → 8× smaller KV-cache
- **Sliding Window Attention** (window=4096) with **rolling buffer** KV-cache
- **ALiBi-free** — uses RoPE without positional extrapolation
- Outperforms LLaMA-2-13B at 7B scale

### 7.2 LLaMA-3 / Llama Architecture

- Dense, no MoE (LLaMA-3 70B / 405B)
- GQA everywhere (8 KV heads)
- RoPE with extended base (θ_base=500,000 vs 10,000 in LLaMA-2)
- 128K context (RoPE interpolation + fine-tuning)
- SwiGLU activation in FFN: `FFN(x) = (xW₁ ⊙ SiLU(xW₃)) W₂`
- Trained on 15T+ tokens

### 7.3 Mixture of Experts at Frontier Scale

**Mixtral 8×22B**: 8 experts, K=2, 141B total / 39B active.
**GPT-4**: Believed to be MoE with ~8 experts (not confirmed by OpenAI).
**Gemini 1.5**: MoE with ~1M token context via ring attention.
**DeepSeek-V2**: 236B total / 21B active, 64 experts, K=6.

### 7.4 SwiGLU and Gated Linear Units

**GLU (Gated Linear Unit, Dauphin et al. 2017)**:
```
GLU(x, W, V, b, c) = σ(xW + b) ⊙ (xV + c)
```

**SwiGLU (Shazeer 2020)**:
```
SwiGLU(x) = (xW₁) ⊙ SiLU(xW₃)  · W₂
```
where `SiLU(x) = x · σ(x)` (smooth, non-monotonic activation).

Benefits vs ReLU FFN:
- Gating allows content-dependent computation
- Better gradient flow (no dying ReLU)
- Empirically +1-2% on language modelling perplexity

Used in: LLaMA, Mistral, Gemma, PaLM-2.

### 7.5 Norm Placement: Pre-LN vs Post-LN

**Original Transformer (Post-LN):**
```
x = LayerNorm(x + Sublayer(x))
```
Problem: gradients in deep networks vanish at bottom layers during training.

**Pre-LN (modern standard):**
```
x = x + Sublayer(LayerNorm(x))
```
Benefits: stable training without warmup; gradients flow directly through residual.

**RMSNorm (Zhang & Sennrich 2019)**: Replace LayerNorm with:
```
RMSNorm(x) = x / RMS(x) · γ,   RMS(x) = √(1/d Σ xᵢ²)
```
Removes mean-centering (simpler), ~15% faster than LayerNorm, same performance.
Used in: LLaMA, Mistral, Gemma, T5.

---

## 8 · Architecture Design Principles

### 8.1 Scaling Laws and Architecture Choice

Chinchilla (Hoffmann et al. 2022) established that for a given compute budget C:
```
Optimal N* = 0.1 · C^{0.5}     (parameters)
Optimal D* = 20 · N*            (training tokens)
```

Architecture innovations shift the Pareto frontier:
- MoE: more params at same FLOP → better quality per inference FLOP
- Efficient attention: same quality at less memory → can use longer context
- State space models: linear scaling → quality-cost trade-off improves with length

### 8.2 The Residual Stream View

Every modern architecture can be viewed as writing to and reading from a **residual
stream** of dimension d:

```
Stream: x_0 = Embed(tokens)
        x_1 = x_0 + Attn₁(x_0)
        x_2 = x_1 + FFN₁(x_1)
        ...
        x_{2L} = logits (via unembedding)
```

Each layer reads from the stream, computes a delta, and adds it back.  This **additive
structure** is what enables stable training via direct gradient flow.

Architectural innovations are essentially different ways of defining what each component
reads from and writes to the stream.

### 8.3 Inductive Biases

| Architecture   | Key inductive bias                    | Strength          |
|----------------|---------------------------------------|-------------------|
| Transformer    | Pairwise attention (full)             | Flexible, global  |
| CNN            | Local, shift-equivariant              | Image/audio       |
| SSM/Mamba      | Linear time-invariance + selectivity  | Long sequences    |
| GNN            | Permutation-equivariant + graph       | Molecular, relational |
| MoE            | Sparse conditional computation        | Scale efficiency  |

The trend: as data and compute scale, weaker inductive biases (more flexible) win.

---

## 9 · Key Takeaways

- The vanilla Transformer has three core bottlenecks: **O(n²) attention**, **dense
  computation**, and **fixed context windows**.  Modern architectures attack each.

- **GQA / MQA** reduce the KV-cache by 8–96× by sharing key/value heads across
  query heads — with minimal quality loss.  Now standard in all frontier LLMs.

- **FlashAttention** achieves the same mathematical result as standard attention but
  uses tiling to reduce HBM memory accesses by an order of magnitude — 2-4× faster.

- **MoE** decouples total parameters from active parameters: Mixtral 8×7B has 46.7B
  parameters but only activates ~12.9B per token, matching 70B-scale quality at
  13B-scale inference cost.  The key challenge is load balancing.

- **Mamba** replaces attention entirely with selective state space models: constant
  memory per step at inference, linear training complexity — ideal for very long
  sequences but worse at precise information retrieval than attention.

- **RoPE / YaRN / Position Interpolation** extend context lengths from training range
  to 128K–1M tokens post-hoc with minimal fine-tuning.

- **SwiGLU + RMSNorm + Pre-LN + GQA + RoPE** is now the dominant LLM "recipe"
  (LLaMA, Mistral, Gemma, Falcon) after ablations showed each component's benefit.

- The field is converging on **hybrid architectures**: MoE + dense layers, Mamba + 
  attention layers, sparse + full attention — combining the best of each paradigm.
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────
OPERATIONS = {

    # ── 1 ────────────────────────────────────────────────────────────────────
    "1 · KV-Cache Memory: MHA vs GQA vs MQA": {
        "description": (
            "Computes and visualises KV-cache memory requirements for Multi-Head "
            "Attention (MHA), Grouped Query Attention (GQA), and Multi-Query Attention "
            "(MQA) across different context lengths and model scales."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

# ── Model configurations ──────────────────────────────────────────────────
models = {
    'GPT-3 (175B)': dict(n_layers=96, n_heads=96, d_head=128, n_kv_gqa=96, n_kv_mqa=1),
    'LLaMA-2-13B':  dict(n_layers=40, n_heads=40, d_head=128, n_kv_gqa=40, n_kv_mqa=1),
    'LLaMA-3-70B':  dict(n_layers=80, n_heads=64, d_head=128, n_kv_gqa=8,  n_kv_mqa=1),
    'Mistral-7B':   dict(n_layers=32, n_heads=32, d_head=128, n_kv_gqa=8,  n_kv_mqa=1),
}

def kv_cache_gb(n_layers, n_kv_heads, d_head, seq_len, dtype_bytes=2):
    \"\"\"KV-cache size in GB.\"\"\"\
    return (2 * n_layers * n_kv_heads * d_head * seq_len * dtype_bytes) / 1e9

context_lengths = [1024, 4096, 8192, 32768, 65536, 131072]

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.flatten()

for ax, (model_name, cfg) in zip(axes, models.items()):
    mha_mem = [kv_cache_gb(cfg['n_layers'], cfg['n_heads'],    cfg['d_head'], L)
               for L in context_lengths]
    gqa_mem = [kv_cache_gb(cfg['n_layers'], cfg['n_kv_gqa'],  cfg['d_head'], L)
               for L in context_lengths]
    mqa_mem = [kv_cache_gb(cfg['n_layers'], cfg['n_kv_mqa'],  cfg['d_head'], L)
               for L in context_lengths]

    xl = [L / 1000 for L in context_lengths]
    ax.plot(xl, mha_mem, 'o-', lw=2.5, color='tomato',      label=f'MHA ({cfg["n_heads"]} KV heads)')
    ax.plot(xl, gqa_mem, 's-', lw=2.5, color='royalblue',   label=f'GQA ({cfg["n_kv_gqa"]} KV heads)')
    ax.plot(xl, mqa_mem, '^-', lw=2.5, color='forestgreen', label=f'MQA (1 KV head)')

    # 80 GB GPU memory line
    ax.axhline(80, color='k', ls='--', lw=1.2, alpha=0.6, label='80 GB GPU limit')
    ax.axhline(24, color='grey', ls=':', lw=1.0, alpha=0.6, label='24 GB GPU limit')

    ax.set_title(f'{model_name}\n{cfg["n_layers"]} layers, d_head={cfg["d_head"]}',
                 fontweight='bold')
    ax.set_xlabel('Context Length (K tokens)')
    ax.set_ylabel('KV-Cache Memory (GB)')
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)
    ax.set_xlim(0, max(xl) + 5)

    # Annotate reduction
    ratio_gqa = mha_mem[-1] / max(gqa_mem[-1], 1e-9)
    ratio_mqa = mha_mem[-1] / max(mqa_mem[-1], 1e-9)
    ax.text(0.05, 0.97, f'GQA: {ratio_gqa:.0f}× smaller\\nMQA: {ratio_mqa:.0f}× smaller',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('KV-Cache Memory: MHA vs GQA vs MQA\\n(fp16 weights, varies by context)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('kv_cache_memory.png', dpi=130, bbox_inches='tight')
plt.show()

print("KV-cache at 128K context (fp16):")
print(f"{'Model':20s} {'MHA':>10s} {'GQA':>10s} {'MQA':>10s} {'GQA ratio':>12s}")
print("-" * 70)
for model_name, cfg in models.items():
    L = 131072
    mha = kv_cache_gb(cfg['n_layers'], cfg['n_heads'],   cfg['d_head'], L)
    gqa = kv_cache_gb(cfg['n_layers'], cfg['n_kv_gqa'],  cfg['d_head'], L)
    mqa = kv_cache_gb(cfg['n_layers'], cfg['n_kv_mqa'],  cfg['d_head'], L)
    print(f"{model_name:20s} {mha:>9.2f}G {gqa:>9.2f}G {mqa:>9.2f}G {mha/gqa:>10.1f}×")
""",
    },

    # ── 2 ────────────────────────────────────────────────────────────────────
    "2 · FlashAttention: Tiling and Memory Access Analysis": {
        "description": (
            "Compares standard attention vs FlashAttention in terms of HBM memory "
            "reads/writes and theoretical speedup.  Visualises how tiling reduces "
            "the memory bottleneck as sequence length grows."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

# ── Memory access model ───────────────────────────────────────────────────
# Standard attention: materialises full N×N matrix in HBM
# FlashAttention:     tiles into blocks that fit in SRAM

def standard_attention_hbm(N, d, bytes_per_el=2):
    \"\"\"HBM bytes read/written by standard attention.\"\"\"\
    # Read Q,K,V: 3*N*d
    # Write S (NxN): N^2
    # Read S for softmax: N^2
    # Write P (NxN): N^2
    # Read P,V to compute O: N^2 + N*d
    # Write O: N*d
    total = 3*N*d + 3*(N**2) + 2*N*d
    return total * bytes_per_el

def flash_attention_hbm(N, d, M=20*1024*1024, bytes_per_el=2):
    \"\"\"HBM bytes for FlashAttention (tiled, SRAM size M bytes).\"\"\"\
    # block size Bc = min(M / (4d), N)
    Bc = min(M // (4 * d * bytes_per_el), N)
    Br = min(M // (4 * d * bytes_per_el), d)
    # Number of outer/inner loops
    Tc = int(np.ceil(N / Bc))
    Tr = int(np.ceil(N / Br))
    # Each iteration: read Q block (Br*d), K/V block (Bc*d), write O block (Br*d)
    hbm = Tc * (Br*d + Bc*d) * bytes_per_el + Tr * Br*d * bytes_per_el
    # Approximate: O(N^2 * d / M)
    return max(hbm, N**2 * d // M * bytes_per_el)

d = 64       # head dimension
M_sram = 20 * 1024 * 1024  # 20 MB SRAM
seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

std_hbm   = [standard_attention_hbm(N, d) / 1e9 for N in seq_lens]
flash_hbm = [flash_attention_hbm(N, d, M_sram) / 1e9 for N in seq_lens]
speedup   = [s/f for s, f in zip(std_hbm, flash_hbm)]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# HBM access comparison
axes[0].loglog(seq_lens, std_hbm,   'o-', color='tomato',      lw=2.5, ms=7,
               label='Standard Attention')
axes[0].loglog(seq_lens, flash_hbm, 's-', color='royalblue',   lw=2.5, ms=7,
               label='FlashAttention')
# Slope reference lines
x_ref = np.array([128, 16384], dtype=float)
axes[0].loglog(x_ref, std_hbm[0] * (x_ref/seq_lens[0])**2,
               'k--', lw=0.8, alpha=0.5, label='O(N²) slope')
axes[0].loglog(x_ref, flash_hbm[0] * (x_ref/seq_lens[0])**2 * 0.1,
               'k:', lw=0.8, alpha=0.5, label='O(N²/M) slope')
axes[0].set_title('HBM Memory Access vs Sequence Length\n(d=64, SRAM=20MB)',
                   fontweight='bold')
axes[0].set_xlabel('Sequence Length N (log)'); axes[0].set_ylabel('HBM Access (GB, log)')
axes[0].legend(fontsize=9); axes[0].grid(alpha=.3)

# Speedup
axes[1].plot(seq_lens, speedup, 'D-', color='forestgreen', lw=2.5, ms=7)
axes[1].axhline(1, color='k', ls='--', lw=1)
axes[1].fill_between(seq_lens, 1, speedup, alpha=0.15, color='forestgreen')
axes[1].set_title('Theoretical HBM Speedup\nStandard / FlashAttention',
                   fontweight='bold')
axes[1].set_xlabel('Sequence Length N'); axes[1].set_ylabel('Speedup ×')
axes[1].grid(alpha=.3)
for x, y in zip(seq_lens[::2], speedup[::2]):
    axes[1].annotate(f'{y:.1f}×', (x, y+0.15), fontsize=8, ha='center')

# Memory breakdown at N=4096
N_demo = 4096
breakdown_std = {
    'Q,K,V Read':     3*N_demo*d*2 / 1e6,
    'Attn Matrix W':  N_demo**2*2 / 1e6,
    'Softmax R/W':    2*N_demo**2*2 / 1e6,
    'Output Write':   N_demo*d*2 / 1e6,
}
breakdown_flash = {
    'Q,K,V (tiled)':  3*N_demo*d*2 / 1e6 * 2,   # read multiple times but small
    'O (incremental)':N_demo*d*2*2 / 1e6,
    'No N² matrix!':  0.0001,
}
ax3a = axes[2]
cats_s = list(breakdown_std.keys()); vals_s = list(breakdown_std.values())
cats_f = list(breakdown_flash.keys()); vals_f = list(breakdown_flash.values())
x = np.arange(max(len(cats_s), len(cats_f)))
ax3a.barh(range(len(cats_s)), vals_s, color='tomato', alpha=0.8, label='Standard')
ax3a.barh([i + 0.4 for i in range(len(cats_f))], vals_f, color='royalblue',
          alpha=0.8, label='FlashAttention')
ax3a.set_yticks(range(len(cats_s)))
ax3a.set_yticklabels(cats_s, fontsize=9)
ax3a.set_title(f'HBM Access Breakdown\n(N={N_demo}, d={d})', fontweight='bold')
ax3a.set_xlabel('Memory Access (MB)'); ax3a.legend(fontsize=9)
ax3a.grid(axis='x', alpha=.3)

plt.suptitle('FlashAttention: Trading Compute for Memory Bandwidth',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('flashattention.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Speedup at N=8192: {speedup[6]:.1f}×")
print(f"Standard attention at N=8192: {std_hbm[6]:.2f} GB HBM access")
print(f"FlashAttention at N=8192:     {flash_hbm[6]:.2f} GB HBM access")
""",
    },

    # ── 3 ────────────────────────────────────────────────────────────────────
    "3 · MoE Router: Load Balancing Simulation": {
        "description": (
            "Simulates a Mixture of Experts router with and without the auxiliary "
            "load balancing loss.  Shows expert utilisation, token dropping, and "
            "how the balance loss prevents expert collapse."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)

def top_k_gate(logits, k=2):
    \"\"\"Top-K gating: select K experts per token.\"\"\"\
    topk_vals  = np.sort(logits, axis=-1)[:, -k:][:, ::-1]
    topk_idx   = np.argsort(logits, axis=-1)[:, -k:][:, ::-1]
    gates      = softmax(topk_vals)      # renormalise over selected experts
    return topk_idx, gates

def load_balance_loss(router_probs, topk_idx, E):
    \"\"\"Auxiliary load balancing loss (Switch Transformer).\"\"\"\
    T = router_probs.shape[0]
    f = np.zeros(E)     # fraction of tokens to each expert
    for i, row in enumerate(topk_idx):
        for e in row: f[e] += 1
    f /= (T * topk_idx.shape[1])     # normalise by total routes

    P = router_probs.mean(axis=0)    # mean router probability per expert
    return E * float(np.sum(f * P))

# ── Simulate training with/without balance loss ───────────────────────────
E      = 8      # number of experts
K      = 2      # top-K experts per token
T      = 256    # batch tokens
n_epochs = 100
lr     = 0.02
alpha  = 0.01   # balance loss weight

def simulate_router(use_balance_loss, n_epochs, E, K, T):
    # Initialise router weights (intentionally biased toward expert 0)
    W_g = rng.standard_normal((16, E)) * 0.5
    W_g[:, 0] += 2.0   # bias toward expert 0
    X = rng.standard_normal((T, 16))

    utilisation_hist = []
    loss_hist = []

    for ep in range(n_epochs):
        logits = X @ W_g                   # (T, E) router logits
        probs  = softmax(logits)           # router probabilities
        idx, gates = top_k_gate(logits, K)

        # Expert utilisation
        util = np.zeros(E)
        for row in idx:
            for e in row: util[e] += 1
        util /= (T * K)
        utilisation_hist.append(util.copy())

        # Main loss (simplified: push toward uniform —simulates LM loss)
        main_loss = -np.mean(np.log(np.mean(probs, axis=0) + 1e-9))

        # Balance loss
        bal_loss = load_balance_loss(probs, idx, E) if use_balance_loss else 0.0
        total_loss = main_loss + alpha * bal_loss
        loss_hist.append(total_loss)

        # Gradient (approximate): push toward uniform distribution
        target_probs = np.ones((T, E)) / E
        grad_main = -(target_probs - probs) / T
        if use_balance_loss:
            # Push router to be more uniform
            grad_bal = alpha * (probs - 1/E)
        else:
            grad_bal = 0.0
        grad = grad_main + grad_bal if use_balance_loss else grad_main
        W_g -= lr * X.T @ grad

    return np.array(utilisation_hist), loss_hist

util_no_bal,   loss_no_bal  = simulate_router(False, n_epochs, E, K, T)
util_with_bal, loss_with_bal = simulate_router(True,  n_epochs, E, K, T)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
epochs = np.arange(n_epochs)
colors = plt.cm.tab10(np.linspace(0, 0.9, E))
ideal  = 1.0 / E   # ideal uniform utilisation

# Expert utilisation over time — no balance loss
for e in range(E):
    axes[0, 0].plot(epochs, util_no_bal[:, e], color=colors[e],
                    lw=1.5, label=f'Expert {e}')
axes[0, 0].axhline(ideal, color='k', ls='--', lw=1.5, label='Ideal (1/E)')
axes[0, 0].set_title('Without Balance Loss\nExpert Utilisation', fontweight='bold')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Token fraction')
axes[0, 0].legend(fontsize=7, ncol=2); axes[0, 0].grid(alpha=.3)

# Expert utilisation over time — with balance loss
for e in range(E):
    axes[0, 1].plot(epochs, util_with_bal[:, e], color=colors[e], lw=1.5)
axes[0, 1].axhline(ideal, color='k', ls='--', lw=1.5, label='Ideal')
axes[0, 1].set_title('With Balance Loss\nExpert Utilisation', fontweight='bold')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=.3)

# Final utilisation comparison (bar chart)
x = np.arange(E)
w = 0.35
axes[0, 2].bar(x - w/2, util_no_bal[-1],   width=w, color='tomato',    alpha=0.85,
               label='No balance loss')
axes[0, 2].bar(x + w/2, util_with_bal[-1], width=w, color='royalblue', alpha=0.85,
               label='With balance loss')
axes[0, 2].axhline(ideal, color='k', ls='--', lw=1.5, label=f'Ideal = {ideal:.3f}')
axes[0, 2].set_title('Final Expert Utilisation\n(Epoch 100)', fontweight='bold')
axes[0, 2].set_xlabel('Expert ID'); axes[0, 2].set_ylabel('Token fraction')
axes[0, 2].legend(fontsize=9); axes[0, 2].grid(axis='y', alpha=.3)

# Utilisation entropy over training (higher = better balanced)
def entropy(u): return -np.sum(u * np.log(u + 1e-9))
ent_no_bal   = [entropy(u) for u in util_no_bal]
ent_with_bal = [entropy(u) for u in util_with_bal]
max_ent = np.log(E)

axes[1, 0].plot(epochs, ent_no_bal,   color='tomato',    lw=2, label='No balance')
axes[1, 0].plot(epochs, ent_with_bal, color='royalblue', lw=2, label='With balance')
axes[1, 0].axhline(max_ent, color='k', ls='--', lw=1.5, label=f'Max entropy = log({E})')
axes[1, 0].set_title('Load Entropy Over Training\n(Higher = Better Balanced)',
                      fontweight='bold')
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Entropy')
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=.3)

# Loss curves
axes[1, 1].plot(epochs, loss_no_bal,   color='tomato',    lw=2, label='No balance')
axes[1, 1].plot(epochs, loss_with_bal, color='royalblue', lw=2, label='With balance')
axes[1, 1].set_title('Total Loss Over Training', fontweight='bold')
axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=.3)

# Summary
axes[1, 2].axis('off')
gini_no  = 1 - np.sum(util_no_bal[-1]**2)
gini_yes = 1 - np.sum(util_with_bal[-1]**2)
cv_no    = np.std(util_no_bal[-1]) / (np.mean(util_no_bal[-1]) + 1e-9)
cv_yes   = np.std(util_with_bal[-1]) / (np.mean(util_with_bal[-1]) + 1e-9)
summary  = (
    f"MoE Load Balancing Summary\n"
    f"{'─'*30}\n"
    f"Experts (E): {E}   K={K}\n"
    f"Batch tokens: {T}\n"
    f"Ideal util: {ideal:.4f} per expert\n\n"
    f"WITHOUT balance loss:\n"
    f"  Final entropy : {ent_no_bal[-1]:.4f} / {max_ent:.4f}\n"
    f"  CV (load)     : {cv_no:.4f}\n"
    f"  Expert collapse: {'YES ⚠️' if cv_no > 0.3 else 'No'}\n\n"
    f"WITH balance loss:\n"
    f"  Final entropy : {ent_with_bal[-1]:.4f} / {max_ent:.4f}\n"
    f"  CV (load)     : {cv_yes:.4f}\n"
    f"  Expert collapse: {'YES ⚠️' if cv_yes > 0.3 else 'No ✓'}\n\n"
    f"Balance loss weight α = {alpha}"
)
axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                fontsize=10.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.9))

plt.suptitle('MoE Router: Load Balancing with Auxiliary Loss',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('moe_balance.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Without balance: entropy = {ent_no_bal[-1]:.3f} / {max_ent:.3f}")
print(f"With balance:    entropy = {ent_with_bal[-1]:.3f} / {max_ent:.3f}")
""",
    },

    # ── 4 ────────────────────────────────────────────────────────────────────
    "4 · MoE Parameter Count vs Active Compute": {
        "description": (
            "Illustrates the MoE trade-off: total parameters scale with number of "
            "experts while per-token FLOPs remain constant.  Compares MoE vs "
            "equivalent dense models on quality-vs-compute Pareto curves."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

# ── MoE parameter analysis ────────────────────────────────────────────────
d        = 4096      # model hidden dim
L        = 32        # transformer layers
K        = 2         # active experts per token
E_values = [1, 2, 4, 8, 16, 32, 64]

def dense_ffn_params(d, expansion=4):
    return 2 * d * (d * expansion)   # W1, W2

def attn_params(d, n_heads=32, kv_heads=8):
    d_head = d // n_heads
    return (n_heads + 2*kv_heads) * d * d_head + d * d   # QKV + O

attn_p   = attn_params(d)
dense_p  = dense_ffn_params(d)
base_p   = (attn_p + dense_p) * L   # dense baseline (E=1)

total_params, active_params, flops_ratio = [], [], []
for E in E_values:
    ffn_total  = E * dense_ffn_params(d) * L
    total      = attn_p * L + ffn_total
    active     = attn_p * L + K * dense_ffn_params(d) * L
    total_params.append(total / 1e9)
    active_params.append(active / 1e9)
    flops_ratio.append(active / base_p)

# ── Quality proxy: log(total_params) ─────────────────────────────────────
# Empirically, quality scales with log(total_params) for MoE
quality_moe   = [np.log(p) * 12 - 40 + 0.5 * np.random.randn() for p in total_params]
quality_dense = [np.log(active_params[0] * (k/2)) * 12 - 40 for k in E_values]
# Seed to make it deterministic
np.random.seed(7)
quality_moe = [np.log(p) * 12 - 38 + rng for p, rng in
               zip(total_params, np.random.randn(len(E_values)) * 0.3)]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Total vs active params
axes[0].plot(E_values, total_params,  'o-', color='tomato',    lw=2.5, ms=7,
             label='Total params')
axes[0].plot(E_values, active_params, 's--', color='royalblue', lw=2.5, ms=7,
             label=f'Active params (K={K})')
for i, E in enumerate(E_values):
    axes[0].annotate(f'{total_params[i]:.0f}B\n({active_params[i]:.0f}B active)',
                     (E, total_params[i]+1), fontsize=7, ha='center', color='tomato')
axes[0].set_title('Total vs Active Parameters\nvs Number of Experts',
                   fontweight='bold')
axes[0].set_xlabel('Number of Experts E')
axes[0].set_ylabel('Parameters (B)')
axes[0].legend(fontsize=10); axes[0].grid(alpha=.3)
axes[0].set_xticks(E_values)

# 2. FLOPs ratio relative to dense
axes[1].bar(range(len(E_values)), flops_ratio,
            color=[plt.cm.RdYlGn(1 - f/max(flops_ratio)) for f in flops_ratio],
            alpha=0.85)
axes[1].set_xticks(range(len(E_values)))
axes[1].set_xticklabels([f'E={e}' for e in E_values], rotation=30)
axes[1].axhline(1.0, color='k', ls='--', lw=1.5, label='Dense baseline (E=1)')
axes[1].set_title(f'Active FLOPs Ratio vs Dense\n(K={K} active experts)',
                   fontweight='bold')
axes[1].set_ylabel('FLOPs / Dense FLOPs')
axes[1].legend(fontsize=9); axes[1].grid(axis='y', alpha=.3)
for i, r in enumerate(flops_ratio):
    axes[1].text(i, r + 0.02, f'{r:.2f}×', ha='center', fontsize=8, fontweight='bold')

# 3. Quality-Compute Pareto
axes[2].scatter(active_params, quality_moe,
                c=E_values, cmap='viridis', s=150, zorder=5, label='MoE', marker='*')
for i, (x, y, E) in enumerate(zip(active_params, quality_moe, E_values)):
    axes[2].annotate(f'E={E}\n({total_params[i]:.0f}B total)',
                     (x, y), textcoords='offset points', xytext=(8, -5), fontsize=7.5)

# Dense comparison: same active compute, scale up dense model
dense_scales = np.linspace(active_params[0], active_params[-1] * 0.7, 8)
quality_dense_curve = np.log(dense_scales) * 12 - 38
axes[2].plot(dense_scales, quality_dense_curve, 'r--', lw=2, label='Dense Pareto')

axes[2].set_title('Quality vs Active Compute\nMoE vs Dense Pareto',
                   fontweight='bold')
axes[2].set_xlabel('Active Params (B) = Inference Cost')
axes[2].set_ylabel('Quality Proxy (higher = better)')
axes[2].legend(fontsize=9); axes[2].grid(alpha=.3)

plt.suptitle('MoE: Total Parameters ↑  Active FLOPs Constant  Quality ↑',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('moe_params.png', dpi=130, bbox_inches='tight')
plt.show()

print("E | Total Params | Active Params | Active/Total | Speedup vs Dense")
for E, tot, act in zip(E_values, total_params, active_params):
    print(f"{E:2d} | {tot:11.1f}B | {act:13.1f}B | "
          f"{act/tot*100:10.1f}% | {(act/active_params[0]):.2f}×")
""",
    },

    # ── 5 ────────────────────────────────────────────────────────────────────
    "5 · SSM Recurrence vs Convolution Duality": {
        "description": (
            "Demonstrates the recurrence-convolution duality that makes SSMs efficient "
            "to train.  Shows that the same SSM output can be computed via sequential "
            "recurrence (for inference) or parallel convolution (for training)."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# ── SSM parameters ────────────────────────────────────────────────────────
N  = 8     # state dimension
L  = 128   # sequence length

# Diagonal A (for simplicity; S4 uses structured complex A)
# Eigenvalues in (0,1) for stability
A_diag = rng.uniform(0.7, 0.98, N)        # diagonal elements of A
A      = np.diag(A_diag)

B = rng.standard_normal(N)                # (N,)  input projection
C = rng.standard_normal(N)                # (N,)  output projection

# Discretise: step size Δ=1 (already discrete), Ā = diag(exp(-Δ/A_diag))
# For simplicity use direct discrete system: Ā = A, B̄ = B
A_bar = A
B_bar = B.copy()

# ── Method 1: Recurrence ─────────────────────────────────────────────────
u = rng.standard_normal(L)    # input signal

h = np.zeros(N)
y_recurrence = np.zeros(L)
for t in range(L):
    h = A_bar @ h + B_bar * u[t]
    y_recurrence[t] = C @ h

# ── Method 2: Convolution (parallel) ─────────────────────────────────────
# Convolutional kernel: K = [CB̄, CĀB̄, CĀ²B̄, ...]
K_kernel = np.zeros(L)
A_pow    = np.eye(N)   # Ā^k
for k in range(L):
    K_kernel[k] = C @ (A_pow @ B_bar)
    A_pow = A_pow @ A_bar

# Compute output via FFT-based convolution
y_convolution = np.real(np.fft.ifft(
    np.fft.fft(K_kernel) * np.fft.fft(u)
))

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
t_ax = np.arange(L)

# Input signal
axes[0, 0].plot(t_ax, u, lw=1.2, color='grey')
axes[0, 0].set_title('Input Signal u(t)', fontweight='bold')
axes[0, 0].grid(alpha=.3); axes[0, 0].set_xlabel('Timestep')

# SSM outputs — should be identical
axes[0, 1].plot(t_ax, y_recurrence, lw=2, color='royalblue', label='Recurrence')
axes[0, 1].plot(t_ax, y_convolution, lw=1.5, color='tomato', ls='--',
                label='Convolution (FFT)')
axes[0, 1].set_title('SSM Output: Recurrence = Convolution\n(numerically identical)',
                      fontweight='bold')
axes[0, 1].legend(fontsize=9); axes[0, 1].grid(alpha=.3); axes[0, 1].set_xlabel('Timestep')

# Error
error = np.abs(y_recurrence - y_convolution)
axes[0, 2].semilogy(t_ax, error + 1e-16, color='purple', lw=2)
axes[0, 2].set_title('Absolute Difference\n(should be ~machine epsilon)',
                      fontweight='bold')
axes[0, 2].set_ylabel('|recurrence - convolution|')
axes[0, 2].grid(alpha=.3); axes[0, 2].set_xlabel('Timestep')

# SSM convolution kernel
axes[1, 0].plot(t_ax, K_kernel, lw=1.5, color='darkorange')
axes[1, 0].set_title('SSM Convolution Kernel K_t = CĀᵗB̄\n(decaying due to |A|<1)',
                      fontweight='bold')
axes[1, 0].grid(alpha=.3); axes[1, 0].set_xlabel('Lag k')
# Compare exponential decay rates
t50 = np.arange(50)
for i in range(min(3, N)):
    axes[1, 0].plot(t50, C[i] * B_bar[i] * A_diag[i]**t50,
                    ls='--', alpha=0.5, lw=1, label=f'Mode {i}: λ={A_diag[i]:.2f}')
axes[1, 0].legend(fontsize=7)

# State evolution
h_hist = np.zeros((L, N))
h = np.zeros(N)
for t in range(L):
    h = A_bar @ h + B_bar * u[t]
    h_hist[t] = h
im = axes[1, 1].imshow(h_hist.T, aspect='auto', cmap='RdBu_r',
                        extent=[0, L, 0, N])
axes[1, 1].set_title('State h_t over Time\n(N=8 state dimensions)', fontweight='bold')
axes[1, 1].set_xlabel('Timestep'); axes[1, 1].set_ylabel('State dimension')
plt.colorbar(im, ax=axes[1, 1])

# Eigenvalue spectrum
eigs = np.sort(A_diag)[::-1]
axes[1, 2].bar(range(N), eigs, color=plt.cm.RdYlGn(eigs), alpha=0.85)
axes[1, 2].axhline(1.0, color='k', ls='--', lw=1.5, label='|λ|=1 (boundary)')
axes[1, 2].set_title('SSM Eigenvalue Spectrum\n(all < 1 → stable)',
                      fontweight='bold')
axes[1, 2].set_xlabel('Mode index'); axes[1, 2].set_ylabel('Eigenvalue λ')
axes[1, 2].legend(fontsize=9); axes[1, 2].grid(axis='y', alpha=.3)
axes[1, 2].set_ylim(0, 1.1)

plt.suptitle('SSM: Recurrence (Inference) = Convolution (Training)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ssm_duality.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Max absolute error (recurrence vs convolution): {error.max():.2e}")
print("→ Numerically identical (floating-point precision only)")
""",
    },

    # ── 6 ────────────────────────────────────────────────────────────────────
    "6 · Mamba Selective State Space — Input-Dependent Dynamics": {
        "description": (
            "Demonstrates Mamba's selective mechanism: B, C, and Δ vary with input. "
            "Compares an LTI (S4-style) vs selective (Mamba-style) SSM on a task "
            "requiring selective memory — remembering only specific input tokens."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(9)

# ── Task: selective copy ──────────────────────────────────────────────────
# Input: sequence with regular values + rare "important" tokens (marked)
# Goal: output should only retain the "important" token values

L  = 64
N  = 4    # state dimension

# Input signal: mostly noise, with 5 important tokens
u = rng.standard_normal(L) * 0.1    # background noise
is_important = np.zeros(L, dtype=bool)
important_positions = [8, 20, 35, 45, 58]
for p in important_positions:
    u[p] = rng.choice([-2.0, 2.0])   # strong signal
    is_important[p] = True

# ── LTI SSM (S4-style): fixed A, B, C, Δ ────────────────────────────────
A_fixed = 0.9
B_fixed = 1.0
C_fixed = 1.0
Δ_fixed = 0.5
A_bar_fixed = np.exp(-Δ_fixed)
B_bar_fixed = (1 - A_bar_fixed) * B_fixed

h_lti = 0.0
y_lti = np.zeros(L)
for t in range(L):
    h_lti = A_bar_fixed * h_lti + B_bar_fixed * u[t]
    y_lti[t] = C_fixed * h_lti

# ── Selective SSM (Mamba-style): B, C, Δ depend on input ─────────────────
# Selectivity: when input is large (important), Δ is large (remember)
# When input is small (noise), Δ is small (forget)

def softplus(x): return np.log1p(np.exp(x))
def sigmoid(x):  return 1 / (1 + np.exp(-x))

# Δ_t = softplus(w_Δ * u_t + b_Δ)
w_delta = 3.0;   b_delta = -1.0   # tuned: large |u| → large Δ
w_B     = 1.0;   b_B     = 0.5
w_C     = 1.0;   b_C     = 0.5

h_mamba   = np.zeros(N)
y_mamba   = np.zeros(L)
delta_hist = np.zeros(L)
B_hist     = np.zeros(L)

A_diag = np.array([0.9, 0.85, 0.8, 0.75])  # N=4 modes

for t in range(L):
    # Input-dependent parameters
    delta_t = softplus(w_delta * abs(u[t]) + b_delta)
    B_t     = sigmoid(w_B * u[t] + b_B)
    C_t     = sigmoid(w_C * u[t] + b_C)

    # Discretise
    A_bar_t = np.exp(-delta_t * A_diag)
    B_bar_t = (1 - A_bar_t) * B_t

    # Update state
    h_mamba = A_bar_t * h_mamba + B_bar_t * u[t]
    y_mamba[t] = np.sum(C_t * h_mamba)

    delta_hist[t] = delta_t
    B_hist[t]     = B_t

# ── Ideal output: only important token values, zeroed elsewhere ───────────
y_ideal = np.where(is_important, u, 0.0)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 11))

# Input
axes[0, 0].bar(range(L), u, color=['tomato' if is_important[t] else 'steelblue'
                                    for t in range(L)], alpha=0.8, width=0.9)
axes[0, 0].set_title('Input Signal\n(red = important tokens)', fontweight='bold')
axes[0, 0].set_xlabel('Timestep'); axes[0, 0].set_ylabel('u_t')
axes[0, 0].grid(axis='y', alpha=.3)

# LTI output
axes[0, 1].plot(range(L), y_lti, lw=2, color='grey', label='LTI (S4-style)')
axes[0, 1].plot(range(L), y_ideal, 's', color='tomato', ms=8, label='Ideal output')
axes[0, 1].set_title('LTI SSM Output\n(cannot selectively filter)',
                      fontweight='bold')
axes[0, 1].legend(fontsize=9); axes[0, 1].grid(alpha=.3)

# Mamba output
axes[1, 0].plot(range(L), y_mamba, lw=2, color='royalblue', label='Mamba (selective)')
axes[1, 0].plot(range(L), y_ideal, 's', color='tomato', ms=8, label='Ideal output')
for p in important_positions:
    axes[1, 0].axvline(p, color='tomato', lw=1, ls='--', alpha=0.5)
axes[1, 0].set_title('Mamba Selective SSM Output\n(better tracks important tokens)',
                      fontweight='bold')
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=.3)

# Delta (step size) — key to selectivity
axes[1, 1].bar(range(L), delta_hist, color=['tomato' if is_important[t] else 'steelblue'
                                             for t in range(L)], alpha=0.8, width=0.9)
axes[1, 1].set_title('Δ_t (Step Size) vs Timestep\nLarge Δ = "pay attention"',
                      fontweight='bold')
axes[1, 1].set_xlabel('Timestep'); axes[1, 1].set_ylabel('Δ_t')
axes[1, 1].grid(axis='y', alpha=.3)

# MSE comparison
mse_lti   = np.mean((y_lti - u)**2)
mse_mamba = np.mean((y_mamba - u)**2)
mse_important_lti   = np.mean((y_lti[is_important] - u[is_important])**2)
mse_important_mamba = np.mean((y_mamba[is_important] - u[is_important])**2)

bars = axes[2, 0].bar(['LTI (all)', 'Mamba (all)',
                         'LTI (important)', 'Mamba (important)'],
                        [mse_lti, mse_mamba, mse_important_lti, mse_important_mamba],
                        color=['grey', 'royalblue', 'lightgrey', 'steelblue'], alpha=0.85)
axes[2, 0].set_title('Reconstruction MSE\n(lower = better memory of important tokens)',
                      fontweight='bold')
axes[2, 0].set_ylabel('MSE')
for bar, v in zip(bars, [mse_lti, mse_mamba, mse_important_lti, mse_important_mamba]):
    axes[2, 0].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[2, 0].grid(axis='y', alpha=.3)

# Architecture comparison table
axes[2, 1].axis('off')
summary = (
    "Selective SSM (Mamba) Key Properties\n"
    "─────────────────────────────────────\n"
    "  Δ_t = softplus(Linear_Δ(x_t))\n"
    "  B_t = sigmoid(Linear_B(x_t))\n"
    "  C_t = sigmoid(Linear_C(x_t))\n\n"
    "Large Δ_t:\n"
    "  A_bar_t = exp(-Δ·A) ≈ 0  (fast decay)\n"
    "  B_bar_t = (1 - A_bar_t)·B ≈ B\n"
    "  → Current token DOMINATES state\n\n"
    "Small Δ_t:\n"
    "  A_bar_t = exp(-Δ·A) ≈ 1  (slow decay)\n"
    "  B_bar_t ≈ 0\n"
    "  → State IGNORES current token\n\n"
    "Key: breaks LTI → enables selection"
)
axes[2, 1].text(0.05, 0.95, summary, transform=axes[2, 1].transAxes,
                fontsize=10, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0fff4', alpha=0.9))

plt.suptitle('Mamba Selectivity: Input-Dependent vs Fixed SSM Dynamics',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('mamba_selective.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"LTI MSE on important tokens:   {mse_important_lti:.4f}")
print(f"Mamba MSE on important tokens: {mse_important_mamba:.4f}")
print(f"Selectivity advantage: {mse_important_lti/mse_important_mamba:.2f}×")
""",
    },

    # ── 7 ────────────────────────────────────────────────────────────────────
    "7 · RoPE Positional Encoding and Length Extrapolation": {
        "description": (
            "Implements Rotary Position Embeddings (RoPE) and visualises how they "
            "encode relative positions via rotation.  Shows why naive extrapolation "
            "fails and how YaRN/position interpolation extends context length."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

# ── RoPE implementation ───────────────────────────────────────────────────
def build_rope_matrix(seq_len, d_head, theta_base=10000.0):
    \"\"\"Build RoPE rotation matrices for all positions.\"\"\"\
    positions = np.arange(seq_len, dtype=float)
    # Frequencies for each pair of dimensions
    dim_pairs = np.arange(0, d_head, 2, dtype=float)
    thetas    = theta_base ** (-dim_pairs / d_head)   # (d_head/2,)

    # angle[m, k] = m · θ_k
    angles = np.outer(positions, thetas)  # (seq_len, d_head/2)
    return angles

def rope_apply(x, angles):
    \"\"\"Apply RoPE to a query/key vector x of shape (seq_len, d_head).\"\"\"\
    seq_len, d = x.shape
    assert d % 2 == 0
    # Reshape to pairs: (seq_len, d/2, 2)
    x_pairs = x.reshape(seq_len, d // 2, 2)
    cos_ang  = np.cos(angles)[:, :, None]   # (seq_len, d/2, 1)
    sin_ang  = np.sin(angles)[:, :, None]

    # Rotate each pair: [x1, x2] → [x1·cos - x2·sin, x1·sin + x2·cos]
    x1 = x_pairs[:, :, 0:1]
    x2 = x_pairs[:, :, 1:2]
    x_rot = np.concatenate([x1*cos_ang - x2*sin_ang,
                              x1*sin_ang + x2*cos_ang], axis=-1)
    return x_rot.reshape(seq_len, d)

def rope_inner_product(pos_q, pos_k, d_head, theta_base=10000.0):
    \"\"\"Expected inner product between two RoPE-encoded unit vectors.\"\"\"\
    # cos(m·θ_k - n·θ_k) averaged over dimensions
    rel   = pos_q - pos_k
    thetas = theta_base ** (-np.arange(0, d_head, 2, dtype=float) / d_head)
    return float(np.mean(np.cos(rel * thetas)))

# ── 1. Rotation angle per dimension at various positions ─────────────────
d_head = 64
L_train = 2048
theta_base = 10000.0
angles_train = build_rope_matrix(L_train, d_head, theta_base)

# ── 2. Relative position dependence ───────────────────────────────────────
rel_dists  = np.arange(0, 500, 1)
inner_prods_standard = [rope_inner_product(500, 500-r, d_head, 10000.0)
                         for r in rel_dists]

# ── 3. Extrapolation: standard vs interpolated ────────────────────────────
L_extend = 8192   # want to extend to 8× training length
scale    = L_extend / L_train   # = 4.0

# Standard: use original positions (> L_train are unseen during training)
positions_ext = np.arange(L_extend)
angles_ext    = build_rope_matrix(L_extend, d_head, theta_base)

# Position Interpolation (PI): scale positions down
angles_pi = build_rope_matrix(L_extend, d_head, theta_base)
# positions go 0, 1/scale, 2/scale, ... L_train-1 → stay in trained range
angles_pi = build_rope_matrix(L_train, d_head, theta_base)  # scaled to fit
# For PI: angles[m'] = angles[m * (L_train / L_extend)]
pi_positions = np.linspace(0, L_train - 1, L_extend)
pi_angles    = np.outer(pi_positions,
                         theta_base ** (-np.arange(0, d_head, 2, dtype=float) / d_head))

# ── 4. Attention score vs position (trained range vs extrapolation) ───────
rng = np.random.default_rng(42)
q = rng.standard_normal((1, d_head)); q /= np.linalg.norm(q)
k = rng.standard_normal((1, d_head)); k /= np.linalg.norm(k)
angles_all = build_rope_matrix(L_extend, d_head, theta_base)

scores_naive = []
scores_pi    = []
for pos in range(0, L_extend, 32):
    # Naive extrapolation
    q_rot_n = rope_apply(np.tile(q, (1,1)),
                          angles_all[pos:pos+1])
    k_rot_n = rope_apply(k, angles_all[0:1])
    scores_naive.append(float(q_rot_n @ k_rot_n.T))

    # Position Interpolation
    pi_pos = pos * (L_train / L_extend)
    pi_ang_q = np.outer([pi_pos], theta_base**(-np.arange(0,d_head,2,dtype=float)/d_head))
    pi_ang_k = np.zeros_like(pi_ang_q)
    q_rot_pi = rope_apply(np.tile(q, (1,1)), pi_ang_q)
    k_rot_pi = rope_apply(k, pi_ang_k)
    scores_pi.append(float(q_rot_pi @ k_rot_pi.T))

pos_vals = np.arange(0, L_extend, 32)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. Rotation angle per dimension
dim_idx = [0, 4, 16, 31]
for idx in dim_idx:
    axes[0, 0].plot(range(min(200, L_train)), angles_train[:200, idx],
                    lw=1.5, label=f'dim {idx*2}: θ={theta_base**(-idx*2/d_head):.3f}')
axes[0, 0].set_title('RoPE Rotation Angles per Position\n(different frequency per dim)',
                      fontweight='bold')
axes[0, 0].set_xlabel('Position m'); axes[0, 0].set_ylabel('Angle (radians)')
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=.3)

# 2. Relative position inner product (should depend only on m-n)
axes[0, 1].plot(rel_dists, inner_prods_standard, color='royalblue', lw=2)
axes[0, 1].axhline(0, color='k', ls='--', lw=0.8)
axes[0, 1].set_title('RoPE: Attention Score vs Relative Distance\n'
                      '(depends only on m − n → relative position encoding)',
                      fontweight='bold')
axes[0, 1].set_xlabel('Relative position (m − n)'); axes[0, 1].set_ylabel('E[cos similarity]')
axes[0, 1].grid(alpha=.3)

# 3. Attention score heatmap (pos × dim)
n_pos   = 20
pos_grid = np.linspace(0, L_train-1, n_pos, dtype=int)
att_map  = np.zeros((n_pos, n_pos))
for i, pi in enumerate(pos_grid):
    for j, pj in enumerate(pos_grid):
        att_map[i, j] = rope_inner_product(pi, pj, d_head, theta_base)
im = axes[0, 2].imshow(att_map, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[0, 2].set_title('RoPE Attention Score Heatmap\n(query pos × key pos)',
                      fontweight='bold')
axes[0, 2].set_xlabel('Key position'); axes[0, 2].set_ylabel('Query position')
plt.colorbar(im, ax=axes[0, 2])

# 4. Extrapolation failure
axes[1, 0].plot(pos_vals, scores_naive, lw=2, color='tomato',    label='Naive extrapolation')
axes[1, 0].plot(pos_vals, scores_pi,    lw=2, color='royalblue', label='Position Interpolation')
axes[1, 0].axvline(L_train, color='k', ls='--', lw=1.5, label=f'Train limit ({L_train})')
axes[1, 0].set_title('Attention Scores Beyond Training Length\nExtrapolation vs PI',
                      fontweight='bold')
axes[1, 0].set_xlabel('Position'); axes[1, 0].set_ylabel('Attention Score (q·k)')
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=.3)

# 5. Wavelength per dimension (θ_base sensitivity)
dim_pairs = np.arange(0, d_head, 2, dtype=float)
for base in [500, 1000, 10000, 500000]:
    thetas = base ** (-dim_pairs / d_head)
    wavelengths = 2 * np.pi / thetas
    axes[1, 1].semilogy(dim_pairs, wavelengths, lw=1.8, label=f'θ_base={base:,}')
axes[1, 1].axhline(L_train, color='k', ls='--', lw=1.2, label=f'L_train={L_train}')
axes[1, 1].axhline(128000,  color='grey', ls=':', lw=1.2, label='128K context')
axes[1, 1].set_title('RoPE Wavelength per Dimension\nLarger θ_base → longer context',
                      fontweight='bold')
axes[1, 1].set_xlabel('Dimension pair index')
axes[1, 1].set_ylabel('Wavelength (tokens, log scale)')
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=.3)

# 6. Summary table
axes[1, 2].axis('off')
summary = (
    "RoPE & Context Extension Summary\n"
    "──────────────────────────────────────\n"
    "RoPE formula:\n"
    "  Q_m = R(mΘ)Q,  K_n = R(nΘ)K\n"
    "  Q·K = f(Q, K, m-n)  (relative pos!)\n\n"
    "Context Extension Methods:\n\n"
    "Position Interpolation (PI):\n"
    "  pos → pos × (L_train / L_extend)\n"
    "  + fine-tuning on long sequences\n"
    "  LLaMA-2 4K → 32K\n\n"
    "YaRN:\n"
    "  High-freq dims: interpolate\n"
    "  Low-freq dims:  extrapolate\n"
    "  + attention temp. scaling 1/√t\n"
    "  LLaMA-2 4K → 128K (400 steps!)\n\n"
    "Larger θ_base (LLaMA-3):\n"
    "  θ_base: 10K → 500K\n"
    "  Longer wavelengths → no saturation\n"
    "  LLaMA-3 handles 128K natively"
)
axes[1, 2].text(0.03, 0.97, summary, transform=axes[1, 2].transAxes,
                fontsize=9.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#fffbf0', alpha=0.9))

plt.suptitle('RoPE: Rotary Position Embeddings & Context Length Extension',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('rope_positions.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"RoPE dependency on relative position confirmed: "
      f"inner_prod(500,499) = {inner_prods_standard[1]:.4f}, "
      f"inner_prod(100,99) = {rope_inner_product(100, 99, d_head):.4f}")
""",
    },

    # ── 8 ────────────────────────────────────────────────────────────────────
    "8 · Speculative Decoding: Speedup Analysis": {
        "description": (
            "Simulates speculative decoding with a draft and target model.  "
            "Computes acceptance rates, expected speedup, and shows how "
            "the speedup varies with acceptance probability and draft length γ."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

# ── Theoretical speedup formula ───────────────────────────────────────────
# E[tokens per step] = (1 - α^{γ+1}) / (1 - α)    where α = acceptance rate
# E[forward passes]  = 1  (one pass through target, plus γ draft passes)
# But draft is cheap: draft_cost = c × target_cost  (c << 1)

def expected_speedup(alpha, gamma, draft_cost_ratio=0.1):
    \"\"\"Expected tokens generated per unit of target-model compute.\"\"\"\
    # E[accepted tokens] per speculative step
    e_accepted = (1 - alpha**(gamma+1)) / (1 - alpha + 1e-9)
    # Cost of one speculative step: γ draft passes + 1 target pass
    step_cost   = gamma * draft_cost_ratio + 1.0
    # Baseline: 1 target pass per token
    return e_accepted / step_cost

# ── Acceptance rate simulation ────────────────────────────────────────────
# Simulate draft and target token distributions
vocab_size = 1000
gamma_sim  = 4      # draft γ tokens

def simulate_spec_decoding(target_probs, draft_probs, gamma, n_steps=1000):
    \"\"\"Simulate speculative decoding and measure acceptance rate.\"\"\"\
    accepted = 0
    total    = 0

    for _ in range(n_steps):
        # Draft generates γ tokens
        draft_tokens = [rng.choice(vocab_size, p=dp) for dp in draft_probs[:gamma]]

        # Target model verifies all γ tokens in parallel
        n_accepted = 0
        for j, tok in enumerate(draft_tokens):
            # Acceptance probability: min(1, p_target / p_draft)
            p_t = target_probs[j][tok]
            p_d = draft_probs[j][tok]
            if rng.random() < min(1.0, p_t / (p_d + 1e-10)):
                n_accepted += 1
            else:
                break

        accepted += n_accepted
        total    += gamma  # total draft tokens

    return accepted / max(total, 1)

# Create target and draft distributions with varying similarity
# High similarity → high acceptance rate
def make_distributions(vocab, similarity, n_tokens):
    \"\"\"Create target and draft distributions with given similarity.\"\"\"\
    target_probs = []
    draft_probs  = []
    for _ in range(n_tokens):
        # Target: peaked distribution
        t = np.abs(rng.standard_normal(vocab)) + 0.01
        t /= t.sum()
        # Draft: adds noise proportional to (1 - similarity)
        noise = np.abs(rng.standard_normal(vocab)) + 0.01
        noise /= noise.sum()
        d = similarity * t + (1 - similarity) * noise
        d /= d.sum()
        target_probs.append(t)
        draft_probs.append(d)
    return target_probs, draft_probs

similarities = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
measured_alphas = []
for sim in similarities:
    tp, dp = make_distributions(100, sim, gamma_sim)   # smaller vocab for speed
    alpha  = simulate_spec_decoding(tp, dp, gamma_sim, n_steps=200)
    measured_alphas.append(alpha)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. Speedup vs acceptance rate for different γ
alpha_range = np.linspace(0.5, 0.99, 100)
for gamma in [1, 2, 4, 7, 10]:
    su = [expected_speedup(a, gamma, 0.1) for a in alpha_range]
    axes[0, 0].plot(alpha_range, su, lw=2, label=f'γ = {gamma}')
axes[0, 0].axhline(1.0, color='k', ls='--', lw=1, label='No speedup')
axes[0, 0].set_title('Expected Speedup vs Acceptance Rate\n(draft_cost = 10% of target)',
                      fontweight='bold')
axes[0, 0].set_xlabel('Token Acceptance Rate α')
axes[0, 0].set_ylabel('Speedup ×')
axes[0, 0].legend(fontsize=9); axes[0, 0].grid(alpha=.3)

# 2. Speedup vs γ at different acceptance rates
gamma_range = np.arange(1, 16)
for alpha in [0.6, 0.7, 0.8, 0.9, 0.95]:
    su = [expected_speedup(alpha, g, 0.1) for g in gamma_range]
    axes[0, 1].plot(gamma_range, su, 'o-', lw=2, ms=5, label=f'α = {alpha}')
    # Find optimal γ
    best_g = gamma_range[np.argmax(su)]
    axes[0, 1].scatter([best_g], [max(su)], s=80, zorder=5)
axes[0, 1].set_title('Speedup vs Draft Length γ\n(★ = optimal γ per α)',
                      fontweight='bold')
axes[0, 1].set_xlabel('Draft length γ'); axes[0, 1].set_ylabel('Speedup ×')
axes[0, 1].legend(fontsize=9); axes[0, 1].grid(alpha=.3)

# 3. Speedup heatmap (α × γ)
alpha_g = np.linspace(0.5, 0.99, 40)
gamma_g = np.arange(1, 15)
heatmap = np.array([[expected_speedup(a, g, 0.1) for g in gamma_g] for a in alpha_g])
im = axes[0, 2].pcolormesh(gamma_g, alpha_g, heatmap, cmap='viridis', shading='auto')
axes[0, 2].set_title('Speedup Heatmap\n(α × γ, draft_cost=10%)', fontweight='bold')
axes[0, 2].set_xlabel('γ (draft tokens)'); axes[0, 2].set_ylabel('Acceptance rate α')
plt.colorbar(im, ax=axes[0, 2], label='Speedup ×')

# 4. Simulated vs theoretical acceptance rate
theoretical_alpha = similarities   # approximation
axes[1, 0].scatter(similarities, measured_alphas, s=80, color='tomato', zorder=5,
                    label='Simulated α')
axes[1, 0].plot([0.5, 1.0], [0.5, 1.0], 'k--', lw=1.5, label='y = x')
axes[1, 0].set_title('Distribution Similarity vs Acceptance Rate\n'
                      '(simulated with γ=4, vocab=100)', fontweight='bold')
axes[1, 0].set_xlabel('Draft-Target Distribution Similarity')
axes[1, 0].set_ylabel('Measured Acceptance Rate α')
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=.3)

# 5. Expected tokens per step
for gamma in [2, 4, 7]:
    e_tok = [(1 - a**(gamma+1)) / (1 - a + 1e-9) for a in alpha_range]
    axes[1, 1].plot(alpha_range, e_tok, lw=2, label=f'γ = {gamma}')
axes[1, 1].axhline(1.0, color='k', ls=':', lw=1, label='Baseline (no spec)')
axes[1, 1].set_title('Expected Accepted Tokens per Speculative Step',
                      fontweight='bold')
axes[1, 1].set_xlabel('Acceptance rate α')
axes[1, 1].set_ylabel('E[accepted tokens]')
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=.3)

# 6. Summary
axes[1, 2].axis('off')
gamma_opt = 4
alpha_typ = 0.8
su_typ = expected_speedup(alpha_typ, gamma_opt, 0.1)
summary = (
    "Speculative Decoding Summary\n"
    "──────────────────────────────\n"
    "Algorithm:\n"
    "  1. Draft model generates γ tokens\n"
    "  2. Target verifies all in 1 pass\n"
    "  3. Accept each: p_accept = \n"
    "     min(1, p_target / p_draft)\n"
    "  4. Fall back at first rejection\n\n"
    "Key properties:\n"
    "  • Output = identical to target only\n"
    "  • No quality loss (mathematically\n"
    "    equivalent)\n"
    "  • Memory: draft + target in GPU\n\n"
    f"Typical settings:\n"
    f"  γ = {gamma_opt} draft tokens\n"
    f"  α = {alpha_typ} acceptance rate\n"
    f"  → {su_typ:.1f}× speedup\n\n"
    "Used in: SGD, TGI, vLLM, Gemini"
)
axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                fontsize=10, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8fff8', alpha=0.9))

plt.suptitle('Speculative Decoding: Speedup Without Quality Loss',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('speculative_decoding.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Expected speedup (α={alpha_typ}, γ={gamma_opt}, draft_cost=10%): {su_typ:.2f}×")
""",
    },

    # ── 9 ────────────────────────────────────────────────────────────────────
    "9 · SwiGLU vs ReLU vs GELU Activations": {
        "description": (
            "Compares activation functions used in modern LLMs: ReLU, GELU, SiLU, "
            "GLU, and SwiGLU.  Shows function shapes, gradient profiles, and "
            "simulates FFN quality differences on a simple fitting task."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

rng = np.random.default_rng(17)

# ── Activation functions ──────────────────────────────────────────────────
def relu(x):    return np.maximum(0, x)
def gelu(x):    return 0.5 * x * (1 + erf(x / np.sqrt(2)))
def silu(x):    return x * (1 / (1 + np.exp(-x)))    # = x · sigmoid(x)
def swiglu(x, W1, W3):
    \"\"\"SwiGLU(x, W1, W3) = SiLU(x @ W1) ⊙ (x @ W3)\"\"\"\
    return silu(x @ W1) * (x @ W3)

def relu_ffn(x, W1, W2):
    return relu(x @ W1) @ W2

def gelu_ffn(x, W1, W2):
    return gelu(x @ W1) @ W2

def swiglu_ffn(x, W1, W2, W3):
    return swiglu(x, W1, W3) @ W2

# ── 1. Activation function shapes ────────────────────────────────────────
x_range = np.linspace(-4, 4, 400)
activations = {
    'ReLU':    relu(x_range),
    'GELU':    gelu(x_range),
    'SiLU':    silu(x_range),
}
gradients = {
    'ReLU':    (x_range > 0).astype(float),
    'GELU':    np.gradient(gelu(x_range), x_range),
    'SiLU':    np.gradient(silu(x_range), x_range),
}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

colors = {'ReLU': 'tomato', 'GELU': 'royalblue', 'SiLU': 'forestgreen'}
for name, vals in activations.items():
    axes[0, 0].plot(x_range, vals, lw=2.5, color=colors[name], label=name)
axes[0, 0].axhline(0, color='k', lw=0.6); axes[0, 0].axvline(0, color='k', lw=0.6)
axes[0, 0].set_title('Activation Functions', fontweight='bold')
axes[0, 0].set_xlabel('x'); axes[0, 0].legend(fontsize=10); axes[0, 0].grid(alpha=.3)

for name, vals in gradients.items():
    axes[0, 1].plot(x_range, vals, lw=2.5, color=colors[name], label=name)
axes[0, 1].set_title('Gradient (dActivation/dx)', fontweight='bold')
axes[0, 1].set_xlabel('x'); axes[0, 1].legend(fontsize=10); axes[0, 1].grid(alpha=.3)
axes[0, 1].axhline(0, color='k', lw=0.6)

# GLU gating visualisation
x_demo  = np.linspace(-3, 3, 200)
W1_d    = np.array([[2.0]])   # amplify
W3_d    = np.array([[1.0]])   # gate
gate_vals  = x_demo.reshape(-1,1) @ W3_d.T
linear_vals = x_demo.reshape(-1,1) @ W1_d.T
swiglu_vals = silu(linear_vals) * gate_vals

axes[0, 2].plot(x_demo, gate_vals.flatten(),   color='grey',     lw=2, label='Gate: x·W₃')
axes[0, 2].plot(x_demo, silu(linear_vals).flatten(), color='steelblue', lw=2, label='SiLU(x·W₁)')
axes[0, 2].plot(x_demo, swiglu_vals.flatten(), color='darkorange', lw=2.5, label='SwiGLU output')
axes[0, 2].set_title('SwiGLU: Gate × Activation\n(Content-Dependent FFN)',
                      fontweight='bold')
axes[0, 2].set_xlabel('x'); axes[0, 2].legend(fontsize=9); axes[0, 2].grid(alpha=.3)

# ── 2. Fitting task: ReLU vs GELU vs SwiGLU FFN ──────────────────────────
din, dout, dhid = 8, 1, 32
n_samples = 200

X_fit = rng.standard_normal((n_samples, din))
y_fit = np.sin(X_fit[:, 0]) * np.cos(X_fit[:, 1]) + X_fit[:, 2]**2   # target function

def train_ffn(ffn_type, X, y, n_epochs=300, lr=0.005):
    W1 = rng.standard_normal((din, dhid)) * 0.1
    W2 = rng.standard_normal((dhid, dout)) * 0.1
    W3 = rng.standard_normal((din, dhid)) * 0.1  # for SwiGLU

    losses = []
    for ep in range(n_epochs):
        if ffn_type == 'relu':
            h = relu(X @ W1); out = h @ W2
        elif ffn_type == 'gelu':
            h = gelu(X @ W1); out = h @ W2
        elif ffn_type == 'swiglu':
            h = swiglu(X, W1, W3); out = h @ W2
        else:
            raise ValueError

        loss = np.mean((out.flatten() - y)**2)
        losses.append(float(loss))

        # Backprop
        dy   = 2 * (out.flatten() - y) / n_samples
        dW2  = h.T @ dy[:, None]
        dh   = dy[:, None] * W2.T

        if ffn_type == 'relu':
            d_act = dh * (X @ W1 > 0).astype(float)
            dW1   = X.T @ d_act
            W1   -= lr * dW1
        elif ffn_type == 'gelu':
            gelu_g = np.gradient(gelu(X @ W1), axis=0) + 1e-8
            d_act  = dh * np.sign(gelu_g)
            dW1    = X.T @ d_act
            W1    -= lr * dW1
        elif ffn_type == 'swiglu':
            # SwiGLU: ∂/∂W1 SiLU(xW1)*(xW3) and ∂/∂W3 SiLU(xW1)*(xW3)
            silu_out = silu(X @ W1)
            gate_out = X @ W3
            dsilu = dh * gate_out  # backprop through silu branch
            dgate = dh * silu_out  # backprop through gate branch
            dW1  = X.T @ (dsilu * np.gradient(silu_out, axis=0))
            dW3  = X.T @ dgate
            W1  -= lr * dW1; W3 -= lr * dW3

        W2 -= lr * dW2

    return losses

losses_relu   = train_ffn('relu',   X_fit, y_fit)
losses_gelu   = train_ffn('gelu',   X_fit, y_fit)
losses_swiglu = train_ffn('swiglu', X_fit, y_fit)

ep_ax = np.arange(300)
axes[1, 0].plot(ep_ax, losses_relu,   lw=2, color='tomato',      label='ReLU FFN')
axes[1, 0].plot(ep_ax, losses_gelu,   lw=2, color='royalblue',   label='GELU FFN')
axes[1, 0].plot(ep_ax, losses_swiglu, lw=2, color='forestgreen', label='SwiGLU FFN')
axes[1, 0].set_title('Training Loss: ReLU vs GELU vs SwiGLU FFN',
                      fontweight='bold')
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('MSE Loss')
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=.3)

# Dead neuron analysis (ReLU)
W1_test  = rng.standard_normal((din, dhid)) * 0.1
X_test   = rng.standard_normal((500, din))
activs   = relu(X_test @ W1_test)
dead_frac = np.mean(activs == 0, axis=0)

axes[1, 1].bar(range(dhid), dead_frac, color='tomato', alpha=0.8)
axes[1, 1].axhline(np.mean(dead_frac), color='k', ls='--', lw=1.5,
                    label=f'Mean dead = {np.mean(dead_frac)*100:.1f}%')
axes[1, 1].set_title('ReLU Dead Neurons\n(fraction of inputs producing 0 output)',
                      fontweight='bold')
axes[1, 1].set_xlabel('Neuron index'); axes[1, 1].set_ylabel('Dead fraction')
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(axis='y', alpha=.3)

# Final loss comparison
finals = [losses_relu[-1], losses_gelu[-1], losses_swiglu[-1]]
names  = ['ReLU', 'GELU', 'SwiGLU']
cols   = ['tomato', 'royalblue', 'forestgreen']
bars = axes[1, 2].bar(names, finals, color=cols, alpha=0.85)
axes[1, 2].set_title('Final MSE Loss Comparison\n(lower = better)',
                      fontweight='bold')
axes[1, 2].set_ylabel('MSE Loss')
for bar, f in zip(bars, finals):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, f + 0.005,
                    f'{f:.4f}', ha='center', fontweight='bold', fontsize=11)
axes[1, 2].grid(axis='y', alpha=.3)

plt.suptitle('Modern FFN Activations: ReLU → GELU → SwiGLU',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('activations.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Final losses:  ReLU={losses_relu[-1]:.4f}  GELU={losses_gelu[-1]:.4f}  "
      f"SwiGLU={losses_swiglu[-1]:.4f}")
print(f"Dead ReLU neurons: {np.mean(dead_frac)*100:.1f}%")
""",
    },

    # ── 10 ───────────────────────────────────────────────────────────────────
    "10 · Computational Complexity: Attention vs SSM vs MoE": {
        "description": (
            "Comprehensive comparison of computational complexity across "
            "Transformer (dense + MoE) and Mamba (SSM) architectures on "
            "training FLOPs, inference memory, and throughput vs sequence length."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

# ── Model parameters ──────────────────────────────────────────────────────
d      = 2048    # hidden dim
N_ssm  = 16      # SSM state dim
n_heads = 16
d_head  = d // n_heads
L_vals  = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

E_moe  = 8    # MoE experts
K_moe  = 2    # active experts

# ── FLOPs per layer per token ─────────────────────────────────────────────
def attn_flops_train(L, d, n_heads):
    \"\"\"Attention FLOPs (training, per layer): O(L² × d).\"\"\"\
    # QKV: 3 * 2*L*d*d
    # Attention matrix: 2*L*L*d
    # Output: 2*L*d*d
    return L * (3 * 2*d*d + 2*L*d + 2*d*d)

def ffn_flops(L, d, expansion=4):
    \"\"\"Dense FFN FLOPs per layer: O(L × d²).\"\"\"\
    return L * (2 * d * d*expansion + 2 * d*expansion * d)

def moe_ffn_flops(L, d, K, expansion=4):
    \"\"\"MoE FFN FLOPs per layer (only K experts active).\"\"\"\
    return L * K * (2 * d * d*expansion + 2 * d*expansion * d)

def ssm_flops_train(L, d, N):
    \"\"\"Mamba SSM FLOPs (parallel scan training): O(L × d × N).\"\"\"\
    return L * d * N * 6   # input projections + scan + output

def ssm_inference_flops(d, N):
    \"\"\"Mamba inference: O(d × N) per step, constant in L.\"\"\"\
    return d * N * 6

# ── Training FLOPs ────────────────────────────────────────────────────────
attn_flops   = [attn_flops_train(L, d, n_heads) / 1e9 for L in L_vals]
dense_total  = [(attn_flops_train(L, d, n_heads) + ffn_flops(L, d)) / 1e9 for L in L_vals]
moe_total    = [(attn_flops_train(L, d, n_heads) + moe_ffn_flops(L, d, K_moe)) / 1e9
                for L in L_vals]
mamba_total  = [ssm_flops_train(L, d, N_ssm) / 1e9 for L in L_vals]

# ── Inference memory (KV-cache) ───────────────────────────────────────────
n_layers = 32
kv_mha   = [2 * n_layers * n_heads * d_head * L * 2 / 1e9 for L in L_vals]  # fp16
kv_gqa8  = [2 * n_layers * 2       * d_head * L * 2 / 1e9 for L in L_vals]  # 8 KV heads
kv_mamba = [n_layers * d * N_ssm * 2 / 1e9 for _ in L_vals]                 # constant!

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# 1. Training FLOPs
axes[0, 0].loglog(L_vals, dense_total,  'o-', lw=2.5, color='tomato',    ms=6,
                   label='Dense Transformer')
axes[0, 0].loglog(L_vals, moe_total,    's-', lw=2.5, color='darkorange', ms=6,
                   label=f'MoE ({E_moe} experts, K={K_moe})')
axes[0, 0].loglog(L_vals, mamba_total,  '^-', lw=2.5, color='royalblue',  ms=6,
                   label='Mamba (SSM)')
# Reference slopes
L_ref = np.array([L_vals[0], L_vals[-1]], dtype=float)
axes[0, 0].loglog(L_ref, [dense_total[0]*(l/L_vals[0])**2 for l in L_ref],
                   'k--', lw=0.8, alpha=0.5, label='O(L²) slope')
axes[0, 0].loglog(L_ref, [mamba_total[0]*(l/L_vals[0]) for l in L_ref],
                   'k:', lw=0.8, alpha=0.5, label='O(L) slope')
axes[0, 0].set_title('Training FLOPs per Layer\n(log-log scale)', fontweight='bold')
axes[0, 0].set_xlabel('Sequence Length L'); axes[0, 0].set_ylabel('GFLOPs (log)')
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=.3)

# 2. Mamba vs Transformer FLOP ratio
ratio = [d/m for d, m in zip(dense_total, mamba_total)]
axes[0, 1].semilogx(L_vals, ratio, 'D-', lw=2.5, color='forestgreen', ms=7)
axes[0, 1].axhline(1, color='k', ls='--', lw=1.2)
axes[0, 1].fill_between(L_vals, 1, ratio, alpha=0.15, color='forestgreen')
axes[0, 1].set_title('Transformer FLOPs / Mamba FLOPs\n(>1 = Mamba cheaper)',
                      fontweight='bold')
axes[0, 1].set_xlabel('Sequence Length L'); axes[0, 1].set_ylabel('FLOP Ratio')
axes[0, 1].grid(alpha=.3)
for L, r in zip(L_vals[::2], ratio[::2]):
    axes[0, 1].annotate(f'{r:.0f}×', (L, r+5), fontsize=8, ha='center')

# 3. Inference KV-cache memory
axes[0, 2].loglog(L_vals, kv_mha,   'o-', lw=2.5, color='tomato',      ms=6,
                   label=f'MHA (n_kv={n_heads})')
axes[0, 2].loglog(L_vals, kv_gqa8,  's-', lw=2.5, color='darkorange',  ms=6,
                   label='GQA (n_kv=2)')
axes[0, 2].loglog(L_vals, kv_mamba, '^--', lw=2.5, color='royalblue',  ms=6,
                   label='Mamba (constant!)')
axes[0, 2].axhline(24, color='grey', ls=':', lw=1.5, alpha=0.7, label='24 GB GPU')
axes[0, 2].axhline(80, color='k',    ls=':', lw=1.5, alpha=0.7, label='80 GB GPU')
axes[0, 2].set_title('Inference Memory (KV-Cache)\nvs Context Length', fontweight='bold')
axes[0, 2].set_xlabel('Context Length L'); axes[0, 2].set_ylabel('GB (log)')
axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=.3)

# 4. Architecture comparison table
axes[1, 0].axis('off')
table_data = [
    ['Architecture', 'Train FLOPs', 'Infer Memory', 'Parallelism', 'Long-range'],
    ['Dense Attn',   'O(L²d)',      'O(Ld) KV',     'Full',        'Full attn'],
    ['MoE',          'O(L²d/E)',    'O(Ld) KV',     'Full',        'Full attn'],
    ['Mamba',        'O(LdN)',      'O(dN) state',   'Scan',        'Linear SSM'],
    ['Hybrid',       'O(L²d/k)',    'O(Ld/k) KV',   'Full',        'Mix'],
]
tbl = axes[1, 0].table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.85])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1a1a2e'); cell.set_text_props(color='white', fontweight='bold')
    elif c == 0:
        cat_colors = ['#ffcccc', '#ffe0cc', '#ccf0cc', '#cce0ff']
        cell.set_facecolor(cat_colors[r-1]); cell.set_text_props(fontweight='bold')
    else:
        cell.set_facecolor('#f9f9f9' if r % 2 == 0 else 'white')
    cell.set_edgecolor('#cccccc')
axes[1, 0].set_title('Complexity Comparison', fontsize=11, fontweight='bold', pad=10)

# 5. Throughput proxy (tokens/sec ∝ 1/FLOPs)
throughput_dense  = [1/f for f in dense_total]
throughput_mamba  = [1/f for f in mamba_total]
# Normalise to throughput at L=512
t0_d = throughput_dense[0]; t0_m = throughput_mamba[0]
axes[1, 1].semilogx(L_vals, [t/t0_d for t in throughput_dense],
                     'o-', lw=2.5, color='tomato', ms=6, label='Dense Transformer')
axes[1, 1].semilogx(L_vals, [t/t0_m for t in throughput_mamba],
                     '^-', lw=2.5, color='royalblue', ms=6, label='Mamba')
axes[1, 1].set_title('Relative Throughput vs Sequence Length\n(normalised to L=512)',
                      fontweight='bold')
axes[1, 1].set_xlabel('Sequence Length L'); axes[1, 1].set_ylabel('Relative Throughput')
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=.3)

# 6. Crossover analysis
axes[1, 2].axis('off')
crossover_L = None
for i, (d_f, m_f) in enumerate(zip(dense_total, mamba_total)):
    if d_f > m_f * 5:
        crossover_L = L_vals[i]
        break
ratio_128k = dense_total[-1] / mamba_total[-1]

summary = (
    f"Complexity Summary (d={d}, N={N_ssm})\n"
    f"{'─'*38}\n"
    f"At L=1024:\n"
    f"  Dense: {dense_total[1]:.1f} GFLOPs/layer\n"
    f"  Mamba: {mamba_total[1]:.1f} GFLOPs/layer\n"
    f"  Ratio: {dense_total[1]/mamba_total[1]:.1f}×\n\n"
    f"At L=131072 (128K):\n"
    f"  Dense: {dense_total[-1]:.0f} GFLOPs/layer\n"
    f"  Mamba: {mamba_total[-1]:.1f} GFLOPs/layer\n"
    f"  Ratio: {ratio_128k:.0f}×\n\n"
    f"Inference KV-cache at L=131072:\n"
    f"  MHA:   {kv_mha[-1]:.1f} GB\n"
    f"  GQA-2: {kv_gqa8[-1]:.2f} GB\n"
    f"  Mamba: {kv_mamba[-1]*1000:.1f} MB (constant!)\n\n"
    f"Winner by context length:\n"
    f"  L < 4K:  Transformer (full attn)\n"
    f"  L > 16K: Mamba / Hybrid preferred\n"
    f"  L > 64K: Mamba / Ring-Attn only"
)
axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                fontsize=10, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f5f5ff', alpha=0.9))

plt.suptitle('Architecture Complexity: Attention vs MoE vs Mamba',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('complexity_comparison.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"FLOP ratio (Dense/Mamba) at 128K: {ratio_128k:.0f}×")
print(f"KV-cache ratio (MHA/Mamba) at 128K: {kv_mha[-1]/kv_mamba[-1]:.0f}×")
""",
    },

    # ── 11 ───────────────────────────────────────────────────────────────────
    "11 · RMSNorm vs LayerNorm and Normalisation Placement": {
        "description": (
            "Compares LayerNorm vs RMSNorm mathematically and empirically. "
            "Shows the Pre-LN vs Post-LN gradient flow difference — why Pre-LN "
            "leads to more stable training of deep networks."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(33)

# ── LayerNorm vs RMSNorm ───────────────────────────────────────────────────
def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    x_n  = (x - mean) / (std + eps)
    if gamma is not None: x_n = gamma * x_n
    if beta  is not None: x_n = x_n + beta
    return x_n

def rms_norm(x, gamma=None, eps=1e-5):
    rms  = np.sqrt(np.mean(x**2, axis=-1, keepdims=True))
    x_n  = x / (rms + eps)
    if gamma is not None: x_n = gamma * x_n
    return x_n

# ── 1. Normalisation demo on skewed input ────────────────────────────────
d_demo = 128
x_raw  = rng.exponential(2.0, (16, d_demo)) + rng.standard_normal((16, d_demo)) * 0.5
gamma  = np.ones(d_demo); beta = np.zeros(d_demo)

x_ln  = layer_norm(x_raw, gamma, beta)
x_rms = rms_norm(x_raw, gamma)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Input distribution
axes[0, 0].hist(x_raw.flatten(),  bins=60, color='grey',       alpha=0.7, label='Raw input',  density=True)
axes[0, 0].hist(x_ln.flatten(),   bins=60, color='royalblue',  alpha=0.7, label='LayerNorm',  density=True)
axes[0, 0].hist(x_rms.flatten(),  bins=60, color='darkorange', alpha=0.7, label='RMSNorm',    density=True)
axes[0, 0].set_title('Input Distribution After Normalisation', fontweight='bold')
axes[0, 0].legend(fontsize=9); axes[0, 0].grid(alpha=.3); axes[0, 0].set_xlabel('Value')

# Per-sample statistics
sample_means_raw = x_raw.mean(axis=1)
sample_stds_raw  = x_raw.std(axis=1)
sample_means_ln  = x_ln.mean(axis=1)
sample_stds_ln   = x_ln.std(axis=1)
sample_rms_raw   = np.sqrt(np.mean(x_raw**2, axis=1))
sample_rms_norm  = np.sqrt(np.mean(x_rms**2, axis=1))

idx = np.arange(16)
w   = 0.3
axes[0, 1].bar(idx - w, sample_stds_raw,  width=w, color='grey',       alpha=0.8, label='Raw std')
axes[0, 1].bar(idx,     sample_stds_ln,   width=w, color='royalblue',  alpha=0.8, label='LN std')
axes[0, 1].bar(idx + w, sample_rms_norm,  width=w, color='darkorange', alpha=0.8, label='RMS (after RMSNorm)')
axes[0, 1].axhline(1.0, color='k', ls='--', lw=1.5, label='Target=1')
axes[0, 1].set_title('Per-Sample Std/RMS\n(should be ~1 after normalisation)',
                      fontweight='bold')
axes[0, 1].set_xlabel('Sample index'); axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(axis='y', alpha=.3)

# Speed comparison (ops count)
ops_ln  = d_demo * 4   # mean, var, normalize, scale+shift
ops_rms = d_demo * 2   # rms, normalize+scale
axes[0, 2].bar(['LayerNorm', 'RMSNorm'], [ops_ln, ops_rms],
               color=['royalblue', 'darkorange'], alpha=0.85, width=0.4)
axes[0, 2].set_title(f'Operation Count (d={d_demo})\nRMSNorm ≈ 15% fewer ops',
                      fontweight='bold')
axes[0, 2].set_ylabel('Approximate operations')
axes[0, 2].annotate(f'{ops_ln} ops', (0, ops_ln + 3), ha='center', fontweight='bold')
axes[0, 2].annotate(f'{ops_rms} ops', (1, ops_rms + 3), ha='center', fontweight='bold')
axes[0, 2].grid(axis='y', alpha=.3)
axes[0, 2].set_ylim(0, ops_ln * 1.3)

# ── 2. Pre-LN vs Post-LN gradient flow ──────────────────────────────────
# Simulate gradient magnitudes through L layers

def simulate_gradient_flow(n_layers, placement='pre', d=64, n_steps=60):
    \"\"\"Simulate gradient flow — track gradient norm at each layer.\"\"\"\
    rng2 = np.random.default_rng(7)
    grad_norms_all = []

    for step in range(n_steps):
        x = rng2.standard_normal((32, d)) * 0.3
        # Forward pass (simplified)
        xs = [x]
        for ℓ in range(n_layers):
            W = rng2.standard_normal((d, d)) * 0.1 / np.sqrt(d)
            if placement == 'pre':
                x_n = rms_norm(x)
                h   = x_n @ W
                x   = x + h   # residual AFTER sublayer
            else:  # post
                h  = x @ W
                x  = rms_norm(x + h)   # norm AFTER residual
            xs.append(x)

        # Simulate backward: gradient = 1 at top, propagate back
        grad = np.ones_like(xs[-1]) * 0.1
        grad_norms = [np.linalg.norm(grad)]
        for ℓ in range(n_layers - 1, -1, -1):
            if placement == 'pre':
                # Residual: grad flows through both paths
                # Sublayer path attenuates by weight norm
                W_norm = rng2.standard_normal((d, d)) * 0.1 / np.sqrt(d)
                grad_sub = grad @ W_norm.T * 0.95   # slight attenuation
                grad     = grad + grad_sub           # residual adds gradients
            else:
                # LayerNorm normalises gradient → can shrink small gradients
                grad = grad / (np.linalg.norm(grad) / (d**0.5) + 1e-4)
            grad_norms.append(np.linalg.norm(grad))

        grad_norms_all.append(grad_norms)

    return np.array(grad_norms_all)

n_layers = 24
grad_pre  = simulate_gradient_flow(n_layers, 'pre')
grad_post = simulate_gradient_flow(n_layers, 'post')

layers = np.arange(n_layers + 1)[::-1]   # top (output) to bottom (input)
axes[1, 0].plot(layers, grad_pre.mean(axis=0),  lw=2.5, color='royalblue', label='Pre-LN')
axes[1, 0].fill_between(layers,
                          grad_pre.mean(axis=0) - grad_pre.std(axis=0),
                          grad_pre.mean(axis=0) + grad_pre.std(axis=0),
                          alpha=0.2, color='royalblue')
axes[1, 0].plot(layers, grad_post.mean(axis=0), lw=2.5, color='tomato',    label='Post-LN')
axes[1, 0].fill_between(layers,
                          grad_post.mean(axis=0) - grad_post.std(axis=0),
                          grad_post.mean(axis=0) + grad_post.std(axis=0),
                          alpha=0.2, color='tomato')
axes[1, 0].set_title(f'Gradient Norm Through Layers\n({n_layers} layers, Pre-LN vs Post-LN)',
                      fontweight='bold')
axes[1, 0].set_xlabel('Layer (from output to input)')
axes[1, 0].set_ylabel('Gradient L2 Norm')
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=.3)

# Gradient variance (stability indicator)
axes[1, 1].semilogy(layers, grad_pre.std(axis=0) + 1e-6,  lw=2, color='royalblue',
                    label='Pre-LN std')
axes[1, 1].semilogy(layers, grad_post.std(axis=0) + 1e-6, lw=2, color='tomato',
                    label='Post-LN std')
axes[1, 1].set_title('Gradient Variance Through Layers\n(lower = more stable)',
                      fontweight='bold')
axes[1, 1].set_xlabel('Layer (from output)'); axes[1, 1].set_ylabel('Gradient Std (log)')
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=.3)

# Summary
axes[1, 2].axis('off')
summary = (
    "Normalisation Summary\n"
    "──────────────────────────────────\n"
    "LayerNorm:\n"
    "  x_n = (x - μ) / (σ + ε) · γ + β\n"
    "  Ops: mean + var + norm + scale+shift\n"
    "  Used: original Transformer, BERT\n\n"
    "RMSNorm (LLaMA standard):\n"
    "  x_n = x / RMS(x) · γ\n"
    "  RMS(x) = √(1/d Σ xᵢ²)\n"
    "  Ops: rms + scale  (~15% faster)\n"
    "  Used: LLaMA, Mistral, Gemma, T5\n\n"
    "Placement:\n"
    "Post-LN (original):\n"
    "  x = LayerNorm(x + Sublayer(x))\n"
    "  Problem: vanishing gradients in deep\n\n"
    "Pre-LN (modern standard):\n"
    "  x = x + Sublayer(LayerNorm(x))\n"
    "  Gradient flows through residual directly\n"
    "  No warmup needed, stable deep training\n"
    "  Used: GPT-2+, LLaMA, all modern LLMs"
)
axes[1, 2].text(0.03, 0.97, summary, transform=axes[1, 2].transAxes,
                fontsize=9.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#fafff5', alpha=0.9))

plt.suptitle('RMSNorm vs LayerNorm & Pre-LN vs Post-LN Gradient Flow',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('normalization.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Pre-LN grad norm variance at bottom layer:  {grad_pre.std(axis=0)[-1]:.4f}")
print(f"Post-LN grad norm variance at bottom layer: {grad_post.std(axis=0)[-1]:.4f}")
print(f"Pre-LN more stable by: {grad_post.std(axis=0)[-1]/grad_pre.std(axis=0)[-1]:.1f}×")
""",
    },

    # ── 12 ───────────────────────────────────────────────────────────────────
    "12 · Architecture Innovation Timeline Dashboard": {
        "description": (
            "Interactive timeline and capability comparison of major architecture "
            "innovations from vanilla Transformer (2017) to modern hybrids (2024). "
            "Radar chart comparison and a structured innovation history."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Architecture timeline ─────────────────────────────────────────────────
innovations = [
    # (year, name, category, description)
    (2017, 'Transformer',        'Attention',   'Multi-head self-attention + FFN'),
    (2018, 'BERT / GPT',         'Pre-training','Masked/causal LM pre-training'),
    (2019, 'MQA',                'Efficiency',  'Multi-query attention (Shazeer)'),
    (2020, 'Reformer',           'Attention',   'LSH attention O(n log n)'),
    (2021, 'Switch Transformer', 'MoE',         'Sparse MoE with K=1 routing'),
    (2021, 'RoPE',               'Position',    'Rotary position embeddings'),
    (2021, 'S4',                 'SSM',         'Structured state spaces (HiPPO)'),
    (2022, 'FlashAttention',     'Efficiency',  'IO-aware exact attention'),
    (2022, 'ALiBi',              'Position',    'Linear attention bias'),
    (2022, 'Chinchilla',         'Scaling',     'Compute-optimal scaling laws'),
    (2023, 'GQA',                'Efficiency',  'Grouped-query attention'),
    (2023, 'Mamba',              'SSM',         'Selective state spaces'),
    (2023, 'Mixtral 8×7B',       'MoE',         'Open-source MoE surpassing LLaMA-2-70B'),
    (2023, 'Mistral 7B',         'Architecture','GQA + SWA, beats LLaMA-2-13B'),
    (2023, 'LLaMA-2',            'Architecture','Open base model (4K→32K via PI)'),
    (2023, 'FlashAttention-2',   'Efficiency',  'Better parallelism, 2× faster'),
    (2023, 'YaRN',               'Position',    '4K→128K context, 400 fine-tune steps'),
    (2024, 'Mamba-2',            'SSM',         'State Space Duality (SSD)'),
    (2024, 'LLaMA-3',            'Architecture','θ=500K RoPE, 128K context, GQA'),
    (2024, 'DeepSeek-V2',        'MoE',         'Fine-grained MoE, 236B/21B active'),
    (2024, 'Jamba',              'Hybrid',      'Mamba + Transformer hybrid'),
]

cat_colors = {
    'Attention':    '#4e79a7',
    'Efficiency':   '#f28e2b',
    'Pre-training': '#59a14f',
    'MoE':          '#e15759',
    'SSM':          '#76b7b2',
    'Position':     '#b07aa1',
    'Architecture': '#ff9da7',
    'Scaling':      '#9c755f',
    'Hybrid':       '#bab0ac',
}

fig = plt.figure(figsize=(20, 12))

# ── Timeline ──────────────────────────────────────────────────────────────
ax_tl = fig.add_axes([0.02, 0.05, 0.48, 0.90])

years = [inn[0] for inn in innovations]
y_positions = {}   # year → current y offset
for inn in innovations:
    yr = inn[0]
    if yr not in y_positions:
        y_positions[yr] = []
    y_positions[yr].append(inn)

# Draw timeline
min_yr, max_yr = 2016.5, 2024.8
ax_tl.axvline(0, color='k', lw=2, zorder=1)
for yr in range(2017, 2025):
    ax_tl.axhline(yr, color='grey', lw=0.4, alpha=0.4, ls='--', zorder=0)
    ax_tl.text(-0.15, yr, str(yr), ha='right', va='center', fontsize=9.5, fontweight='bold')

x_offset = 0.0
for yr, entries in sorted(y_positions.items()):
    for i, (year, name, cat, desc) in enumerate(entries):
        xpos  = 0.05 + i * 0.0
        color = cat_colors.get(cat, '#888888')
        ax_tl.scatter(xpos, year, s=120, color=color, zorder=5, clip_on=False)
        ax_tl.annotate(
            f'{name}', (xpos + 0.04, year),
            fontsize=8.5, va='center',
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=0.8, alpha=0.9)
        )
        if i == 0:  # connect to timeline
            ax_tl.plot([0, xpos], [year, year], '-', color=color, lw=1.2, alpha=0.7)

ax_tl.set_xlim(-0.2, 1.6)
ax_tl.set_ylim(min_yr, max_yr)
ax_tl.axis('off')
ax_tl.set_title('Architecture Innovation Timeline (2017–2024)',
                 fontsize=12, fontweight='bold', pad=10)

# Legend
legend_patches = [mpatches.Patch(color=c, label=k) for k, c in cat_colors.items()]
ax_tl.legend(handles=legend_patches, fontsize=8, loc='lower right',
              ncol=2, title='Category', title_fontsize=8)

# ── Radar chart: architecture capability comparison ───────────────────────
ax_r = fig.add_axes([0.52, 0.40, 0.46, 0.58], polar=True)
categories = ['Quality\n(dense)', 'Long\nContext', 'Inference\nSpeed', 'Train\nStab.',
              'Memory\nEff.', 'Param\nEff.']
archs = {
    'Vanilla Transformer': [7, 3, 5, 5, 4, 5],
    'Transformer + GQA':   [7, 6, 7, 6, 7, 5],
    'MoE (Mixtral)':       [9, 5, 8, 7, 5, 9],
    'Mamba':               [6, 9, 9, 8, 9, 7],
    'Hybrid MoE+Mamba':    [9, 9, 8, 8, 8, 9],
}
arch_colors = ['grey', 'royalblue', 'tomato', 'forestgreen', 'darkorange']
N_cat = len(categories)
angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]

for (arch, scores), col in zip(archs.items(), arch_colors):
    vals = scores + scores[:1]
    ax_r.plot(angles, vals, 'o-', color=col, lw=2, ms=4, label=arch)
    ax_r.fill(angles, vals, alpha=0.07, color=col)

ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(categories, fontsize=9)
ax_r.set_ylim(0, 10); ax_r.set_yticks([3,6,9]); ax_r.set_yticklabels(['3','6','9'], fontsize=7)
ax_r.set_title('Architecture Capability Radar\n(2024 State of the Art)',
               fontsize=11, fontweight='bold', pad=20)
ax_r.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=8.5)

# ── Key principles box ────────────────────────────────────────────────────
ax_p = fig.add_axes([0.52, 0.02, 0.46, 0.36])
ax_p.axis('off')
principles = (
    "Key Architectural Principles (2024)\n"
    "────────────────────────────────────────────────────────\n"
    "The Modern LLM Recipe (LLaMA-3, Mistral, Gemma):\n"
    "  ✓ Pre-LN with RMSNorm (stable, fast)\n"
    "  ✓ SwiGLU activations in FFN\n"
    "  ✓ GQA (8 KV heads) — 8× smaller KV-cache\n"
    "  ✓ RoPE (θ=500K) — long context out of the box\n"
    "  ✓ FlashAttention-2/3 for training efficiency\n\n"
    "For Scale → MoE:  Total params ↑, FLOPs constant\n"
    "For Long Context → Mamba:  O(L) train, O(1) infer\n"
    "For Best of Both → Hybrid:  Mamba + sparse attention\n\n"
    "The residual stream view unifies all:\n"
    "  Every component reads from and writes to a shared\n"
    "  d-dimensional stream — enabling stable gradient flow."
)
ax_p.text(0.02, 0.98, principles, transform=ax_p.transAxes,
           fontsize=9.5, va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#f0f0ff', alpha=0.95))

plt.suptitle('Architecture Innovations: Beyond the Vanilla Transformer',
             fontsize=15, fontweight='bold', y=1.01)
plt.savefig('architecture_timeline.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Innovations tracked: {len(innovations)}")
print(f"Categories covered: {len(cat_colors)}")
print("Modern LLM recipe: RMSNorm + Pre-LN + SwiGLU + GQA + RoPE + FlashAttention")
""",
    },
}


def get_topic_data():
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  "",
        "operations":   OPERATIONS,
    }