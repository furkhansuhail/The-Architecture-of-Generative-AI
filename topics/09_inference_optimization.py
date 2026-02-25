"""Module: 09 · Inference Optimization"""

DISPLAY_NAME = "09 · Inference Optimization"
ICON         = "🚀"
SUBTITLE     = "Quantization, KV cache, speculative decoding, distillation — making LLMs fast and cheap"

THEORY = """
## 09 · Inference Optimization

Training a language model is expensive but done once. Inference is done billions of times
per day. A single GPT-4-scale model serving production traffic can cost millions of dollars
per month in GPU time. Inference optimization is the discipline of making that cost —
in latency, memory, and dollars — as low as possible without degrading output quality.
This module covers every major technique, from the physics of GPU memory to the algorithms
that let a small draft model accelerate a large one.

---

### 1 · The Inference Cost Problem

**1.1 Why inference is expensive.** Transformer inference has two phases:

- **Prefill:** Process the entire input prompt in one forward pass. Highly parallelisable —
  all input tokens are known, so all attention computations can run in parallel. This phase
  is *compute-bound*: the GPU's arithmetic units are the bottleneck.

- **Decode:** Generate tokens one at a time via autoregression. Each new token requires a
  full forward pass through all layers. This phase is *memory-bandwidth-bound*: for each
  token generated, every weight matrix must be loaded from HBM (High Bandwidth Memory) into
  SRAM/registers, used for a tiny amount of arithmetic, then discarded. The weights do far
  more travelling than computing.

**1.2 Arithmetic intensity.** The ratio of floating point operations to bytes read from
memory is called *arithmetic intensity* (FLOP/byte). GPU hardware has two limits:

```
Peak throughput = min(FLOPS / arithmetic_intensity, Memory_bandwidth)
```

For decoding a 7B model:
- Weights: 14 GB (FP16)
- Per-token decode FLOPS: ~14 GFLOPs
- Arithmetic intensity: 14 GFLOPs / 14 GB = 1 FLOP/byte

An A100 achieves ~312 TFLOPS but only ~2 TB/s memory bandwidth. The roofline breakeven
is 312 / 2,000 = 156 FLOPs/byte. Single-token decode sits at 1 FLOP/byte — 156× below
the compute roofline. We are almost entirely memory-bandwidth-bound.

**1.3 The memory wall.** Modern LLMs require:
- Model weights: Llama 3 8B = 16 GB (FP16), Llama 3 70B = 140 GB
- KV cache: grows linearly with batch size × sequence length
- Activations: ~2 GB for a 7B model at batch=1
- Peak GPU memory is 40–80 GB on H100/A100, 24 GB on consumer cards

This is why quantization, KV cache management, and model parallelism are not academic
curiosities — they are prerequisites for deploying large models at all.

---

### 2 · Floating Point Formats — The Representation Trade-off

A floating point number uses three fields: sign (S), exponent (E), mantissa/fraction (M).
The exponent field determines *dynamic range* (how large/small numbers can be);
the mantissa field determines *precision* (how finely spaced the numbers are).

```
FP32:  S[1] E[8]  M[23]  range: +/-3.4e38, precision: ~7 decimal digits
FP16:  S[1] E[5]  M[10]  range: +/-6.5e4,  precision: ~3 decimal digits
BF16:  S[1] E[8]  M[ 7]  range: +/-3.4e38, precision: ~2 decimal digits
TF32:  S[1] E[8]  M[10]  (internal NVIDIA compute format, not storage)
FP8:   S[1] E[4]  M[ 3]  range: +/-448,    precision: ~1 decimal digit
INT8:  8-bit integer      range: -128..127 (signed) or 0..255 (unsigned)
INT4:  4-bit integer      range: -8..7
```

**BF16 vs FP16.** BF16 (Brain Float 16) sacrifices mantissa bits to match FP32's exponent
range. This is crucial for training: gradient updates span many orders of magnitude and
overflow FP16's limited range (hence the FP16 loss scaling hacks). BF16 eliminates overflow
at the cost of coarser precision. For inference, both work well.

**Why INT quantisation works.** Neural network weights after training are approximately
normally distributed, centred near zero with most values in a small range. Quantising
to INT8 or INT4 applies a scale factor to map this range to integers. The quantisation
error (the rounding noise added) is small relative to the signal — models are naturally
robust to this noise because they were never trained to use the precise 32-bit values.

---

### 3 · Quantization

Quantization reduces model weight (and sometimes activation) precision. This reduces
memory footprint, reduces memory bandwidth requirements, and on modern hardware
enables faster integer matrix multiplications.

**3.1 Uniform linear quantization.** Map a float value x in range [alpha, beta] to an
integer in range [q_min, q_max]:

```
scale      = (beta - alpha) / (q_max - q_min)
zero_point = round(q_min - alpha / scale)

quantize:   q = round(x / scale) + zero_point   (clamped to [q_min, q_max])
dequantize: x_hat = scale * (q - zero_point)

quantization_error = x - x_hat
```

For INT8: q_min = -128, q_max = 127, so scale = range / 255.

**3.2 Granularity.** The quantization parameters (scale, zero_point) can be shared at
different granularities:

- *Per-tensor:* one scale for the entire weight matrix. Fastest; highest error.
- *Per-channel (per-output-neuron):* one scale per row of the weight matrix. Better accuracy.
- *Per-group (blockwise):* one scale per group of G consecutive weights (G = 32, 64, 128).
  Used in GPTQ, AWQ, GGUF. Best accuracy at 4-bit.

**3.3 PTQ vs QAT.**

*Post-Training Quantization (PTQ):* Quantize a pre-trained model without retraining.
Requires only a small calibration dataset to compute activation statistics. Fast — works
in minutes. Works well for INT8; moderate accuracy loss at INT4.

*Quantization-Aware Training (QAT):* Simulate quantization during fine-tuning.
Forward pass uses quantized weights; backward pass uses straight-through estimator (STE)
to pass gradients through the rounding operation. Best accuracy; requires full training run.

**3.4 Advanced PTQ methods.**

*GPTQ (Frantar et al., 2022):* Layer-wise, second-order PTQ. For each weight in a layer,
uses the Hessian of the layer's reconstruction error to decide optimal rounding.
Achieves INT4 with ~1% accuracy loss on LLMs. The key insight: minimise the layer-wise
output reconstruction error ||WX - W_hat X||^2 using optimal quantization of each weight
column, compensating rounding errors in subsequent columns.

*AWQ (Lin et al., 2023):* Activation-aware Weight Quantization. Identifies "salient"
weight channels (those multiplied by large activations) and protects them with higher
precision. Remaining 99% of weights are aggressively quantized. Outperforms GPTQ
at 4-bit with faster calibration.

*GGUF/llama.cpp:* Mixed precision: layers near the embedding and output use higher
precision; middle layers use INT4 or INT3. Optimised for CPU inference.

**3.5 Quantization memory savings:**

| Format | Bits/param | 7B model | 70B model | vs FP16 |
|---|---|---|---|---|
| FP32 | 32 | 28 GB | 280 GB | 2x |
| FP16/BF16 | 16 | 14 GB | 140 GB | 1x (baseline) |
| INT8 | 8 | 7 GB | 70 GB | 0.5x |
| INT4 (GPTQ) | 4 | 3.5 GB | 35 GB | 0.25x |
| INT3 | 3 | 2.6 GB | 26 GB | 0.19x |
| INT2 | 2 | 1.75 GB | 17.5 GB | 0.125x |

---

### 4 · KV Cache

**4.1 The recomputation problem.** In autoregressive decoding, at step t the model
attends over all previous tokens 0..t-1. Without caching, each new token forces
recomputation of all prior keys and values — quadratic total computation in sequence length.

**4.2 KV cache.** Store the key and value projections for every previously generated token.
On step t, compute K and V only for the new token, then concatenate with the cache.

```
Attention(Q_t, [K_0..K_t], [V_0..V_t]) -- uses cached K/V
```

Each token in the cache requires storing 2 vectors (K and V) per layer, per head.

**4.3 KV cache memory formula.**

```
KV_cache_bytes = 2 x layers x n_kv_heads x head_dim x seq_len x batch_size x bytes_per_element
```

For Llama 3 8B (32 layers, 8 kv_heads, 128 head_dim, FP16=2 bytes):
```
At seq=4096, batch=1: 2 x 32 x 8 x 128 x 4096 x 1 x 2 = 536 MB
At seq=4096, batch=8: 4.3 GB
At seq=32768, batch=8: 34 GB
```

The KV cache is the primary limit on serving long sequences and large batches.

**4.4 Multi-Query Attention (MQA) and Grouped-Query Attention (GQA).**

Standard MHA: each head has its own K and V projections. n_heads KV pairs per layer.

MQA (Shazeer 2019): all attention heads share a single K and V. Reduces KV cache by n_heads.

GQA (Ainslie et al., 2023): n_kv_heads groups, each shared by n_heads/n_kv_heads heads.
Llama 3 8B uses GQA with n_kv_heads=8 (vs n_heads=32), reducing KV cache 4x.

```
MHA KV size:  n_heads    x head_dim x seq x layers
GQA KV size:  n_kv_heads x head_dim x seq x layers   (n_kv_heads < n_heads)
MQA KV size:  1          x head_dim x seq x layers
```

**4.5 Paged KV cache (vLLM).** Standard KV cache allocates a contiguous block per
sequence, sized for the maximum possible length. This wastes memory when sequences finish
early. vLLM introduces paged attention: the KV cache is divided into fixed-size blocks
(pages). A page table maps each sequence's logical positions to physical memory pages.
Pages are allocated on demand and freed when sequences complete — like virtual memory for
an operating system. This eliminates internal fragmentation and enables up to 24x higher
throughput than static allocation.

---

### 5 · FlashAttention

**5.1 Standard attention memory complexity.** Naive attention computes:
```
S = Q K^T / sqrt(d_k)        # (seq x seq) matrix
P = softmax(S)                # (seq x seq) attention weights
O = P V                       # (seq x d_v) output
```

This materialises the full (seq x seq) attention matrix in HBM — O(N^2) memory. At
seq=16,384, this is 256M float16 values = 512 MB per layer. Prohibitive for long contexts.

**5.2 FlashAttention algorithm (Dao et al., 2022).** The key insight: softmax requires
knowledge of the max value across all columns (for numerical stability) and the sum of
exponentials (for normalisation). FlashAttention computes a numerically-stable online
softmax without ever materialising the full attention matrix, using a tiling approach:

1. Divide Q, K, V into blocks that fit in SRAM.
2. For each block: compute partial attention scores, update a running max and running sum.
3. Merge partial results using the numerically-stable online softmax update.
4. Write the final output to HBM once.

**5.3 IO complexity comparison.**

| Method | HBM reads | HBM writes | Memory |
|---|---|---|---|
| Standard attention | O(Nd + N^2) | O(N^2) | O(N^2) |
| FlashAttention | O(N^2 d / M) | O(Nd) | O(N) |

Where N = sequence length, d = head dimension, M = SRAM size.

For N=4096, d=64, on A100 SRAM=20MB: FlashAttention uses ~15x less HBM I/O.

FlashAttention 2 and 3 further reduce non-matrix operations, improve parallelisation
across query blocks, and achieve ~70-85% of theoretical peak A100 GPU throughput.

---

### 6 · Knowledge Distillation

**6.1 The core idea.** A large "teacher" model contains more knowledge than is expressed
in its hard label predictions. When the teacher assigns 70% probability to "cat", 20% to
"dog", 10% to "lion", it reveals that "cat" and "dog" are more similar than "cat" and
"house." A small "student" model can learn from these *soft labels* — which encode
inter-class relationships — rather than just the ground-truth hard labels.

**6.2 Temperature scaling.** To make the teacher's distribution more informative (less
peaked at the correct class), scale the logits by temperature T before softmax:

```
p_T(k) = exp(z_k / T) / sum_j exp(z_j / T)
```

High T gives a flatter distribution, more entropy, more "dark knowledge" shared.
At inference, T = 1.0. During distillation, T = 3-6 is common.

**6.3 Distillation loss.** Combined objective:

```
L_KD = alpha * L_CE(y_hard, sigma(z_student)) + (1-alpha) * T^2 * KL(p_T^teacher || p_T^student)
```

The T^2 factor corrects for the fact that the KL divergence with temperature T produces
gradients that are 1/T^2 of the normal gradient magnitude.

alpha is typically 0.1-0.5; larger alpha weights the ground-truth more.

**6.4 Feature distillation.** Beyond output logits, students can also learn from:
- Intermediate layer activations (FitNets)
- Attention patterns (TinyBERT attention transfer)
- Relation matrices between examples (RKD)

**6.5 Task-agnostic vs task-specific distillation.**
DistilBERT (Sanh et al., 2019): distilled from BERT during pretraining. 40% smaller,
60% faster, retaining 97% of BERT's performance on GLUE.
TinyLlama (Zhang et al., 2024): 1.1B parameters distilled from Llama 2 on 3T tokens.

---

### 7 · Speculative Decoding

**7.1 The bottleneck.** Standard autoregressive decoding generates one token per model
forward pass. For a large target model (70B), this is slow. The memory-bandwidth
bottleneck means even a powerful GPU idles while waiting for weights to be loaded.

**7.2 The core insight.** It is much cheaper to *verify* multiple tokens in parallel than
to *generate* them one at a time. Given a sequence of K candidate tokens, the target
model can verify all K in a single forward pass (because the inputs are all known at
once, like prefill). If the draft tokens match the target's distribution, we accept them;
otherwise we resample from the correct distribution.

**7.3 Algorithm (Chen et al., 2023; Leviathan et al., 2023).**

1. Draft model generates K tokens autoregressively: x_1, x_2, ..., x_K
2. Target model processes input + all K draft tokens in one parallel forward pass.
3. For each draft token x_i, accept with probability min(1, q(x_i)/p(x_i))
   where p = draft probability, q = target probability.
4. If token i is rejected, resample from adjusted distribution; discard tokens i+1..K.
5. Regardless, one bonus token is always accepted from the target at the last position.

**7.4 Expected tokens per step.**

If all draft tokens are accepted with probability alpha:
```
E[accepted tokens] = (1 - alpha^(K+1)) / (1 - alpha)
```

For alpha=0.8, K=4: E = (1 - 0.8^5)/(1 - 0.8) = (1 - 0.328)/0.2 = 3.36 tokens per step
vs 1.0 token per step for standard decoding — 3.36x speedup.

**7.5 Acceptance rate in practice.** Alpha depends on how well the draft model's
distribution matches the target. Common draft/target pairings:
- Llama 8B (draft) + Llama 70B (target): alpha ~ 0.6-0.8
- Target model with fewer layers as draft (self-speculative): alpha ~ 0.7-0.85

**7.6 Correctness guarantee.** Speculative decoding is *lossless* — the output
distribution is mathematically identical to sampling from the target model alone,
regardless of draft quality. Bad drafts waste compute (reject and resample) but
never corrupt the output distribution.

**7.7 Extensions.**
- *Medusa (Cai et al., 2024):* adds multiple independent draft "heads" to the target
  model itself, each predicting 1 step ahead, 2 steps ahead, etc. No separate draft model.
- *SpecInfer:* tree-structured speculative decoding with multiple draft models.

---

### 8 · Batching Strategies

**8.1 Why batching.** A single-token decode uses a tiny amount of GPU compute relative
to the memory bandwidth spent loading weights. Batching N requests together amortises
this weight-loading cost: N tokens are generated in the same time it takes to generate 1,
because the weights are loaded once and multiplied against N vectors simultaneously.
This turns memory-bandwidth-bound inference toward compute-bound.

**8.2 Static batching.** The naive approach: wait until a full batch is assembled, process
all sequences together until all complete, return all results. GPU utilisation: excellent
when all sequences are the same length. In practice, sequences finish at different times,
leaving idle GPU bubbles when short sequences complete while long ones continue.

**8.3 Dynamic (continuous) batching.** As soon as a sequence completes, remove it from the
batch and add a new waiting request. The batch is repacked every decode step. This is the
key innovation in systems like vLLM (Yu et al., 2022), TGI, and TensorRT-LLM. Throughput
improvement over static batching: 2-4x at typical production traffic patterns.

**8.4 Chunked prefill.** Long prefill requests monopolise the GPU for many milliseconds,
blocking decode steps for other requests (high time-to-first-token for those requests).
Chunked prefill splits long prompts into fixed-size chunks interleaved with decode steps,
reducing P99 TTFT (time to first token) dramatically while keeping throughput high.

---

### 9 · Structured Pruning

**9.1 Unstructured vs structured pruning.**

*Unstructured pruning:* Set individual weights to zero based on magnitude. Achieves
high sparsity (90%+) with minimal accuracy loss. Problem: irregular sparse matrices
have poor GPU utilisation — current hardware is not optimised for random sparsity.

*Structured pruning:* Remove entire rows, columns, heads, or layers. The result is a
smaller dense model that runs efficiently on standard hardware. Less sparsity achievable
for same accuracy loss, but actual wallclock speedup is real.

**9.2 Magnitude pruning.** Simplest method: sort all weights by |w|, zero out the
smallest P%. With fine-tuning afterward (gradual magnitude pruning), models can recover
most accuracy at 40-70% sparsity. Without fine-tuning, accuracy drops significantly
beyond 20-30% sparsity.

**9.3 Attention head pruning.** Not all attention heads are equally important.
Michel et al. (2019) showed that 20-40% of heads in BERT can be removed with minimal
accuracy loss, because many heads are redundant. Head importance score:
```
Importance(h) = |dL/d xi_h| (expected absolute gradient w.r.t. head gate)
```

**9.4 Layer dropping.** In very deep models (>32 layers), some layers contribute minimally.
Dropping 10-20% of layers gives 10-20% latency reduction at ~1-2% accuracy loss.
DistilBERT uses layer interleaving: keep every other layer, fine-tune.

**9.5 Width pruning.** LLM-Pruner (Ma et al., 2023): prune coupled structures (attention
heads + FFN neurons) together, maintaining structural consistency. Achieves 20% size
reduction in Llama with ~3% accuracy loss.

---

### 10 · Hardware-Aware Optimization and Serving Systems

**10.1 Fused CUDA kernels.** Each GPU operation (matrix multiply, softmax, ReLU, layer
norm) incurs a kernel launch and HBM read/write. Fusing multiple operations into a single
kernel eliminates intermediate HBM roundtrips. FlashAttention is the canonical example:
fuses the Q*K^T, softmax, and softmax*V operations into one kernel.

**10.2 Triton.** OpenAI's Triton is a Python-based GPU programming language that compiles
to PTX, making custom CUDA kernels accessible to ML researchers. xFormers, FlashAttention,
and many quantized GEMM kernels are implemented in Triton.

**10.3 Torch.compile.** PyTorch 2.0's JIT compiler traces and compiles a model's forward
pass into optimised CUDA code, automatically applying operator fusion. 20-40% speedup
on transformer workloads with one line: model = torch.compile(model).

**10.4 TensorRT.** NVIDIA's inference engine: converts trained models to optimised
TensorRT engines with layer fusion, precision calibration (INT8/FP16), and kernel
autotuning. Achieves 2-8x speedup over native PyTorch.

**10.5 Key serving systems:**

| System | Key innovation | Best for |
|---|---|---|
| vLLM | Paged attention, continuous batching | High-throughput serving |
| TGI (Hugging Face) | Continuous batching, tensor parallelism | Open model serving |
| TensorRT-LLM | Quantization + kernel fusion | NVIDIA GPU max throughput |
| llama.cpp | CPU inference, GGUF quantization | Consumer hardware, edge |
| Ollama | llama.cpp wrapper with model management | Local deployment |
| DeepSpeed-FastGen | Dynamic SplitFuse batching | Large model serving |

**10.6 Tensor parallelism at inference.** Shard weight matrices across GPUs:
Column-parallel: split output dimensions. Row-parallel: split input dimensions.
Megatron-LM style: each MLP layer requires one all-reduce per GPU per layer.
For inference, tensor parallelism across 2-8 GPUs enables 70B models on accessible hardware.

---

### Key Takeaways

- Inference is memory-bandwidth-bound during decode. Single-token arithmetic intensity
  is ~1 FLOP/byte vs a GPU roofline of 150+ FLOP/byte. Every optimization must increase
  arithmetic intensity or reduce memory pressure.
- Quantization (INT8/INT4) reduces memory 2-4x with minimal accuracy loss. AWQ and
  GPTQ achieve 4-bit with <1% accuracy degradation on LLMs. INT8 GEMMs are also 2x
  faster than FP16 on modern hardware — compression and speedup combined.
- The KV cache is the second largest memory consumer after weights, growing linearly
  with batch x sequence length. GQA/MQA reduce it 4-32x. Paged attention eliminates
  wasted allocation. These are prerequisites for long-context and high-batch serving.
- FlashAttention eliminates the O(N^2) memory bottleneck of standard attention by tiling
  computation to stay within SRAM. It is now standard in all production LLM training
  and inference code.
- Speculative decoding is a rigorous, lossless technique that achieves 2-4x speedup by
  using a cheap draft model to propose tokens, then verifying multiple in parallel.
- Knowledge distillation compresses teacher knowledge into smaller students via soft
  labels. DistilBERT retains 97% of BERT performance at 40% smaller.
- Continuous batching is the single highest-impact serving optimization for throughput.
  It eliminates GPU idle bubbles by repacking the batch every decode step.
- Production-grade inference stacks layer all of these: INT4 weights + GQA +
  FlashAttention + speculative decoding + continuous batching + tensor parallelism.
  Each technique is complementary; combining them achieves 10-50x throughput vs
  naive FP32 inference.
"""

OPERATIONS = {
    "1 · Floating Point Format Anatomy": {
        "description": (
            "Dissect FP32, FP16, BF16, INT8, and INT4 bit layouts. Show dynamic range, "
            "precision, model memory at each format, and the information-loss curve from FP32."
        ),
        "language": "python",
        "code": """\
import math, struct

def fp32_fields(f):
    packed = struct.pack('>f', f)
    bits   = int.from_bytes(packed, 'big')
    sign   = (bits >> 31) & 1
    exp    = (bits >> 23) & 0xFF
    mant   = bits & 0x7FFFFF
    return sign, exp, mant

def fp16_fields(f):
    s, e, m = fp32_fields(float(f))
    new_e = e - 127 + 15
    new_m = m >> 13
    if new_e <= 0:
        new_e, new_m = 0, 0
    elif new_e >= 31:
        new_e, new_m = 31, 0
    return s, new_e, new_m

def bf16_fields(f):
    s, e, m = fp32_fields(float(f))
    return s, e, m >> 16

def reconstruct_fp16(s, e, m):
    if e == 0:
        return 0.0
    return (-1)**s * 2**(e - 15) * (1 + m / 1024.0)

def reconstruct_bf16(s, e, m):
    new_m_fp32 = m << 16
    bits = (s << 31) | (e << 23) | new_m_fp32
    return struct.unpack('>f', bits.to_bytes(4, 'big'))[0]

formats = [
    ("FP32",      32, 8,  23, 3.4e38,  "Standard training / high-precision"),
    ("FP16",      16, 5,  10, 6.5e4,   "Training (with loss scaling), inference"),
    ("BF16",      16, 8,   7, 3.4e38,  "Training (preferred), inference"),
    ("FP8-E4M3",   8, 4,   3, 448.0,   "Cutting-edge training (H100+)"),
    ("INT8",       8, 0,   0, 127.0,   "Quantized inference — 2x speedup"),
    ("INT4",       4, 0,   0, 7.0,     "Quantized inference — 4x compression"),
]

print("=" * 80)
print("Floating Point and Integer Format Comparison")
print("=" * 80)
print(f"  {'Format':<12} {'Bits':>5} {'Exp':>5} {'Mant':>6} {'Max Value':>14} {'Distinct':>14}")
print("  " + "-" * 60)
for name, bits, exp_b, mant_b, max_val, note in formats:
    distinct = 2 ** bits
    print(f"  {name:<12} {bits:>5} {exp_b:>5} {mant_b:>6} {max_val:>14.3e} {distinct:>14,}")
print()
print("  Note: INT8/INT4 are integers — no exponent/mantissa breakdown.")
print("  BF16 matches FP32 exponent width (8 bits), preventing overflow in training.")
print()

print("=" * 80)
print("Model Memory by Format (GB)")
print("=" * 80)
models_params = [("Llama 3 8B", 8e9), ("Llama 3 70B", 70e9), ("GPT-3 175B", 175e9)]
bits_list     = [("FP32", 32), ("FP16", 16), ("INT8", 8), ("INT4", 4), ("INT2", 2)]

header = f"  {'Model':<16}"
for bname, _ in bits_list:
    header += f" {bname:>10}"
print(header)
print("  " + "-" * 68)
for mname, params in models_params:
    row = f"  {mname:<16}"
    for _, bits in bits_list:
        gb = params * bits / 8 / 1e9
        row += f" {gb:>9.1f}G"
    print(row)
print()

print("=" * 80)
print("Precision Test: Representing pi = 3.14159265358979...")
print("=" * 80)
pi = math.pi
s32, e32, m32 = fp32_fields(pi)
fp32_r = (-1)**s32 * 2**(e32 - 127) * (1 + m32 / 2**23)
s16, e16, m16 = fp16_fields(pi)
fp16_r = reconstruct_fp16(s16, e16, m16)
sb, eb, mb = bf16_fields(pi)
bf16_r = reconstruct_bf16(sb, eb, mb)

scale_i8 = pi / 127.0
int8_r   = round(pi / scale_i8) * scale_i8
scale_i4 = pi / 7.0
int4_r   = round(pi / scale_i4) * scale_i4

print(f"  True value:  {pi:.15f}")
print(f"  FP32:        {fp32_r:.15f}  error={abs(pi-fp32_r):.2e}  (32 bits)")
print(f"  FP16:        {fp16_r:.15f}  error={abs(pi-fp16_r):.2e}  (16 bits)")
print(f"  BF16:        {bf16_r:.15f}  error={abs(pi-bf16_r):.2e}  (16 bits)")
print(f"  INT8:        {int8_r:.15f}  error={abs(pi-int8_r):.2e}  ( 8 bits)")
print(f"  INT4:        {int4_r:.15f}  error={abs(pi-int4_r):.2e}  ( 4 bits)")
print()
print("  INT4 error (~7% of value) is acceptable for weights but not activations.")
print("  BF16 has less precision than FP16 but equal dynamic range to FP32.")
""",
    },

    "2 · Post-Training Quantization Simulation": {
        "description": (
            "Implement INT8 and INT4 quantization from scratch: per-tensor, per-channel, "
            "and per-group granularities. Measure quantization error and show clipping effects."
        ),
        "language": "python",
        "code": """\
import math, random
random.seed(42)

def rand_normal(n, sigma=0.02, seed=42):
    rng = random.Random(seed)
    return [rng.gauss(0, sigma) for _ in range(n)]

def quantize_tensor(weights, bits=8):
    q_min = -(2**(bits-1))
    q_max  =  (2**(bits-1)) - 1
    w_min, w_max = min(weights), max(weights)
    scale = (w_max - w_min) / (q_max - q_min)
    if scale == 0:
        return list(weights), 1.0, 0
    zp  = round(q_min - w_min / scale)
    dq  = [scale * (max(q_min, min(q_max, round(w/scale) + zp)) - zp) for w in weights]
    return dq, scale, zp

def quantize_channel(weights, rows, cols, bits=8):
    q_min = -(2**(bits-1))
    q_max  =  (2**(bits-1)) - 1
    result = []
    for r in range(rows):
        row = weights[r*cols:(r+1)*cols]
        w_min, w_max = min(row), max(row)
        scale = (w_max - w_min) / (q_max - q_min)
        if scale == 0:
            result.extend(row)
            continue
        zp  = round(q_min - w_min / scale)
        dq  = [scale * (max(q_min, min(q_max, round(w/scale)+zp)) - zp) for w in row]
        result.extend(dq)
    return result

def quantize_group(weights, group_size=32, bits=4):
    q_min = -(2**(bits-1))
    q_max  =  (2**(bits-1)) - 1
    result = []
    for g in range(len(weights) // group_size):
        chunk = weights[g*group_size:(g+1)*group_size]
        w_min, w_max = min(chunk), max(chunk)
        scale = (w_max - w_min) / (q_max - q_min)
        if scale == 0:
            result.extend(chunk)
            continue
        zp  = round(q_min - w_min / scale)
        dq  = [scale * (max(q_min, min(q_max, round(w/scale)+zp)) - zp) for w in chunk]
        result.extend(dq)
    rem = len(weights) % group_size
    if rem:
        result.extend(weights[-rem:])
    return result

def mse(a, b):
    return sum((x-y)**2 for x, y in zip(a, b)) / len(a)

def max_abs(a, b):
    return max(abs(x-y) for x, y in zip(a, b))

ROWS, COLS = 64, 64
N = ROWS * COLS
weights = rand_normal(N, sigma=0.015, seed=7)
rng = random.Random(99)
for _ in range(20):
    i = rng.randint(0, N-1)
    weights[i] = rng.choice([-1, 1]) * rng.uniform(0.15, 0.25)

print("=" * 68)
print("Post-Training Quantization: Error Analysis")
print(f"  {ROWS}x{COLS}={N} weights | Normal(0, 0.015) + 20 outliers")
print("=" * 68)
print()
print(f"  {'Method':<22} {'MSE':>14} {'MaxAbsErr':>12} {'Overhead':>14}")
print("  " + "-" * 66)

configs = [
    ("INT8 Per-Tensor",   "tensor",  8),
    ("INT8 Per-Channel",  "channel", 8),
    ("INT4 Per-Tensor",   "tensor",  4),
    ("INT4 Per-Group-32", "group",   4),
]
for name, mode, bits in configs:
    if mode == "tensor":
        dq, _, _ = quantize_tensor(weights, bits)
        overhead = "1 scale"
    elif mode == "channel":
        dq = quantize_channel(weights, ROWS, COLS, bits)
        overhead = f"{ROWS} scales"
    else:
        dq = quantize_group(weights, 32, bits)
        overhead = f"{N//32} scales"
    print(f"  {name:<22} {mse(weights, dq):>14.6e} {max_abs(weights, dq):>12.6f} {overhead:>14}")

print()
print("=" * 68)
print("Clipping: Effect of Removing Top Outliers Before Quantizing")
print("=" * 68)
print()
sorted_abs = sorted(abs(w) for w in weights)
print(f"  {'Clip%':>8} {'Threshold':>12} {'INT8 MSE':>14} {'INT4 MSE':>14}")
print("  " + "-" * 52)
for pct in [100.0, 99.9, 99.5, 99.0, 98.0]:
    thresh   = sorted_abs[int(len(sorted_abs) * pct / 100) - 1]
    clipped  = [max(-thresh, min(thresh, w)) for w in weights]
    dq8, _, _ = quantize_tensor(clipped, 8)
    dq4, _, _ = quantize_tensor(clipped, 4)
    print(f"  {pct:>7.1f}% {thresh:>12.4f} {mse(weights, dq8):>14.6e} {mse(weights, dq4):>14.6e}")

print()
print("  Clipping top 1% outliers reduces INT4 MSE by allowing a finer scale.")
print("  Per-group quantization has the best accuracy because each group gets")
print("  its own scale, isolating outliers to their local neighbourhood.")
""",
    },

    "3 · KV Cache Memory Modelling": {
        "description": (
            "Model KV cache memory for real LLM architectures. Compare MHA vs GQA vs MQA, "
            "show memory vs sequence length and batch size, and compute max supported batches."
        ),
        "language": "python",
        "code": """\
import math

MODELS = {
    "Llama 3 8B":  {"layers": 32, "n_heads": 32, "n_kv": 8,  "head_dim": 128, "params_b": 8},
    "Llama 3 70B": {"layers": 80, "n_heads": 64, "n_kv": 8,  "head_dim": 128, "params_b": 70},
    "Mistral 7B":  {"layers": 32, "n_heads": 32, "n_kv": 8,  "head_dim": 128, "params_b": 7},
    "Falcon 7B":   {"layers": 32, "n_heads": 71, "n_kv": 1,  "head_dim": 64,  "params_b": 7},
    "GPT-J 6B":    {"layers": 28, "n_heads": 16, "n_kv": 16, "head_dim": 256, "params_b": 6},
}
BYTES_FP16 = 2

def kv_gb(layers, n_kv, head_dim, seq, batch, bpe=BYTES_FP16):
    return 2 * layers * n_kv * head_dim * seq * batch * bpe / 1e9

def weight_gb(params_b, bits=16):
    return params_b * 1e9 * bits / 8 / 1e9

print("=" * 72)
print("KV Cache: Bytes per Token per Request (batch=1, FP16)")
print("=" * 72)
print(f"  {'Model':<16} {'Type':>10} {'B/tok':>10} {'KB@1K':>10} {'MB@32K':>12}")
print("  " + "-" * 62)
for name, cfg in MODELS.items():
    l, nh, nkv, hd = cfg["layers"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"]
    attn_type = "MHA" if nkv == nh else ("MQA" if nkv == 1 else f"GQA-{nkv}")
    bpt   = 2 * l * nkv * hd * BYTES_FP16
    kb1k  = bpt * 1024 / 1024
    mb32k = bpt * 32768 / 1e6
    print(f"  {name:<16} {attn_type:>10} {bpt:>10,} {kb1k:>10.1f} {mb32k:>12.1f}")

print()
print("=" * 72)
print("GPU Memory Budget: Model Weights + KV Cache on 80GB A100")
print("=" * 72)
GPU_GB = 80.0
for mname in ["Llama 3 8B", "Llama 3 70B"]:
    cfg = MODELS[mname]
    l, nh, nkv, hd, pb = cfg["layers"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["params_b"]
    wgt = weight_gb(pb)
    print(f"  {mname}  (weights={wgt:.0f}GB FP16, batch=8)")
    print(f"  {'SeqLen':>10} {'KV Cache':>12} {'Total':>10} {'Headroom':>12} {'OK?':>8}")
    print("  " + "-" * 58)
    for sl in [2048, 8192, 32768]:
        kv  = kv_gb(l, nkv, hd, sl, 8)
        tot = wgt + kv
        rem = GPU_GB - tot
        ok  = "YES" if rem >= 2.0 else "NO (OOM)"
        print(f"  {sl:>10,} {kv:>11.2f}G {tot:>9.1f}G {rem:>11.1f}G {ok:>8}")
    print()

print("=" * 72)
print("MHA vs GQA vs MQA: KV Cache Comparison at seq=8192, batch=8")
print("=" * 72)
BASE = {"layers": 32, "head_dim": 128, "params_b": 7}
variants = [
    ("MHA  (32 kv)", 32), ("GQA-8  (8 kv)", 8), ("GQA-4  (4 kv)", 4),
    ("GQA-2  (2 kv)", 2), ("MQA   (1 kv)", 1),
]
sl, bs = 8192, 8
mha_kv = kv_gb(BASE["layers"], 32, BASE["head_dim"], sl, bs)
wgt_gb = weight_gb(BASE["params_b"])

print(f"  {'Variant':<22} {'KV Cache':>12} {'vs MHA':>10} {'MaxSeq@80GB':>14}")
print("  " + "-" * 62)
for vname, nkv in variants:
    kv  = kv_gb(BASE["layers"], nkv, BASE["head_dim"], sl, bs)
    ratio = kv / mha_kv
    avail = 80.0 - wgt_gb
    max_seq = int(avail * 1e9 / (2 * BASE["layers"] * nkv * BASE["head_dim"] * bs * BYTES_FP16))
    bar = "=" * int((1.0/max(ratio, 0.05)) * 8)
    print(f"  {vname:<22} {kv:>11.2f}G {ratio:>9.2f}x {max_seq:>12,}  {bar}")

print()
print("  Llama3-8B uses GQA-8: 4x KV reduction vs MHA with minimal quality loss.")
print("  MQA (Falcon) achieves 32x KV reduction but with some quality degradation.")
""",
    },

    "4 · Speculative Decoding Simulation": {
        "description": (
            "Implement the full speculative decoding loop: draft proposal, parallel verification, "
            "rejection sampling, and expected speedup. Compare K values and acceptance rates."
        ),
        "language": "python",
        "code": """\
import math, random
random.seed(2024)

VOCAB = 50

def make_probs(rng, vocab=VOCAB, temperature=1.0):
    raw   = [rng.gauss(0, 1) / temperature for _ in range(vocab)]
    mx    = max(raw)
    exps  = [math.exp(x - mx) for x in raw]
    total = sum(exps)
    return [e / total for e in exps]

def sample(probs, rng):
    r, cum = rng.random(), 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1

def speculative_step(draft_rng, target_rng, K):
    draft_tokens, draft_probs_list = [], []
    for k in range(K):
        p = make_probs(draft_rng, temperature=0.9)
        t = sample(p, draft_rng)
        draft_tokens.append(t)
        draft_probs_list.append(p)

    target_probs_list = [make_probs(target_rng) for _ in range(K+1)]

    accepted = 0
    for k in range(K):
        tok    = draft_tokens[k]
        p_d    = draft_probs_list[k][tok]
        q_t    = target_probs_list[k][tok]
        if random.random() < min(1.0, q_t / max(p_d, 1e-12)):
            accepted += 1
        else:
            diff  = [max(0.0, qt - pd) for qt, pd in zip(target_probs_list[k], draft_probs_list[k])]
            dsum  = sum(diff)
            if dsum > 0:
                norm = [d/dsum for d in diff]
                sample(norm, random.Random())
            return accepted, 1, True   # rejected at k

    sample(target_probs_list[K], random.Random())
    return accepted + 1, 1, False

def simulate(N_tokens, K, seed=2024):
    random.seed(seed)
    draft_rng  = random.Random(seed + 1)
    target_rng = random.Random(seed + 2)
    tokens_gen, target_calls = 0, 0
    total_accepted, total_rejected = 0, 0
    while tokens_gen < N_tokens:
        acc, tc, rejected = speculative_step(draft_rng, target_rng, K)
        tokens_gen    += acc
        target_calls  += tc
        total_accepted += acc - (0 if not rejected else 1)
        if rejected:
            total_rejected += 1
    return tokens_gen, target_calls, total_accepted, total_rejected

N_TOKENS = 80
print("=" * 70)
print(f"Speculative Decoding: {N_TOKENS} tokens")
print("=" * 70)
print()
print(f"  {'K':>4} {'TgtCalls':>10} {'Accepted':>10} {'Rejected':>10} {'AccRate':>10} {'Speedup':>10}")
print("  " + "-" * 60)

baseline_calls = N_TOKENS
for K in [1, 2, 3, 4, 6, 8]:
    toks, tcalls, acc, rej = simulate(N_TOKENS, K)
    total_proposals = acc + rej
    alpha   = acc / max(total_proposals, 1)
    speedup = N_TOKENS / tcalls
    bar     = "=" * int(speedup * 5)
    print(f"  {K:>4} {tcalls:>10} {acc:>10} {rej:>10} {alpha:>10.3f} {speedup:>9.2f}x  {bar}")

print()
print(f"  Standard decoding: {baseline_calls} target calls = 1.00x baseline")
print()

print("=" * 70)
print("Expected Speedup Table: E[tokens/step] = (1 - a^(K+1)) / (1 - a)")
print("=" * 70)
print()
print(f"  {'Alpha':>8}  ", end="")
for K in [1, 2, 4, 6, 8]:
    print(f" K={K:>2}", end="")
print()
print("  " + "-" * 48)
for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    row = f"  {alpha:>8.2f}  "
    for K in [1, 2, 4, 6, 8]:
        e = (1 - alpha**(K+1)) / (1 - alpha)
        row += f" {e:>5.2f}"
    print(row)

print()
print("  Key insight: gains diminish past K=4. At alpha=0.8, K=4 gives 3.36x speedup.")
print("  At alpha=0.5 (poor draft), even K=8 only gives 1.99x. Draft quality matters.")
""",
    },

    "5 · Knowledge Distillation — Soft Labels and KD Loss": {
        "description": (
            "Implement temperature-scaled softmax, compare information content of soft vs "
            "hard labels, and compute the full KD loss with alpha weighting."
        ),
        "language": "python",
        "code": """\
import math, random
random.seed(42)

def softmax(logits, T=1.0):
    sc  = [z / T for z in logits]
    mx  = max(sc)
    exp = [math.exp(x - mx) for x in sc]
    tot = sum(exp)
    return [e / tot for e in exp]

def entropy(probs):
    return -sum(p * math.log(p + 1e-12) for p in probs)

def kl_div(p, q):
    return sum(pi * math.log((pi + 1e-12) / (qi + 1e-12)) for pi, qi in zip(p, q))

def cross_entropy(target, pred_logits):
    pp = softmax(pred_logits)
    return -sum(t * math.log(p + 1e-12) for t, p in zip(target, pp))

def kd_loss(hard, teacher_lgts, student_lgts, T=4.0, alpha=0.3):
    ce_h      = cross_entropy(hard, student_lgts)
    tch_T     = softmax(teacher_lgts, T=T)
    stu_T     = softmax(student_lgts, T=T)
    kl        = kl_div(tch_T, stu_T)
    total     = alpha * ce_h + (1 - alpha) * T * T * kl
    return total, ce_h, kl

N_CLASSES  = 8
CLASS_NAMES = ["cat", "dog", "lion", "tiger", "fish", "bird", "car", "house"]

def teacher_lgts_for(true_cls, confidence=4.0, seed=0):
    rng = random.Random(seed)
    z   = [rng.gauss(0, 0.5) for _ in range(N_CLASSES)]
    z[true_cls] += confidence
    return z

true_cls   = 0
tch_lgts   = teacher_lgts_for(true_cls, seed=42)
hard_labels = [1.0 if i == true_cls else 0.0 for i in range(N_CLASSES)]

print("=" * 70)
print("Temperature Effect: Teacher Soft Labels at Different Temperatures")
print(f"  True class: '{CLASS_NAMES[true_cls]}'")
print("=" * 70)
print()

temps = [0.5, 1.0, 2.0, 4.0, 8.0]
print(f"  {'Class':<10}", end="")
for T in temps:
    print(f"  T={T:<3.0f}", end="")
print()
print("  " + "-" * 52)
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<10}", end="")
    for T in temps:
        p = softmax(tch_lgts, T=T)[i]
        print(f"  {p:.3f}", end="")
    print()
print()
print("  Entropy:", end="")
for T in temps:
    H = entropy(softmax(tch_lgts, T=T))
    print(f"  {H:.3f}", end="")
print()
print()
print("  High T: flatter distribution encodes inter-class similarities.")
print("  At T=8 'dog' gets 15%+, revealing cat-dog semantic closeness.")
print()

print("=" * 70)
print("KD Loss Components vs Alpha (weight on hard labels)")
print("=" * 70)
print()
rng = random.Random(7)
stu_lgts = [l + rng.gauss(0, 1.5) for l in tch_lgts]

print(f"  {'Alpha':>8} {'Total':>12} {'CE-Hard':>12} {'KL-Soft':>12} {'Dominant':>12}")
print("  " + "-" * 60)
for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    tot, ce_h, kl = kd_loss(hard_labels, tch_lgts, stu_lgts, T=4.0, alpha=alpha)
    dom = "Soft KL" if alpha < 0.5 else ("Hard CE" if alpha > 0.5 else "Equal")
    print(f"  {alpha:>8.1f} {tot:>12.4f} {ce_h:>12.4f} {kl:>12.4f} {dom:>12}")

print()
print("=" * 70)
print("Information Content: Hard vs Soft Labels")
print("=" * 70)
print()
print(f"  Hard label entropy: {entropy(hard_labels):.4f}  (one class, no similarity info)")
print()
for T in [1.0, 2.0, 4.0, 8.0]:
    soft = softmax(tch_lgts, T=T)
    H    = entropy(soft)
    top3 = sorted(enumerate(soft), key=lambda x: x[1], reverse=True)[:3]
    top3_str = "  ".join(f"{CLASS_NAMES[i]}:{p:.3f}" for i, p in top3)
    print(f"  T={T:.0f}  H={H:.3f}  top-3: {top3_str}")

print()
print("  Soft labels encode that 'dog' and 'lion' are more like 'cat' than 'car' is.")
print("  DistilBERT uses T=4 distillation to retain 97% of BERT performance at 40% size.")
""",
    },

    "6 · Continuous vs Static Batching Throughput": {
        "description": (
            "Simulate static and continuous batching. Show GPU utilisation, idle bubbles, "
            "tokens-per-step throughput, and the 2-4x improvement from continuous batching."
        ),
        "language": "python",
        "code": """\
import random, math
random.seed(123)

def make_requests(n, lo=50, hi=500, seed=42):
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n)]

def simulate_static(requests, batch_size=8):
    total_time, useful, wasted, tokens = 0, 0, 0, 0
    n_batches = math.ceil(len(requests) / batch_size)
    batch_utils = []
    for b in range(n_batches):
        batch   = requests[b*batch_size:(b+1)*batch_size]
        max_len = max(batch)
        tokens += sum(batch)
        total_batch_slots = max_len * len(batch)
        useful_slots      = sum(batch)
        wasted_slots      = total_batch_slots - useful_slots
        useful  += useful_slots
        wasted  += wasted_slots
        total_time += max_len
        batch_utils.append(useful_slots / max(total_batch_slots, 1))
    return total_time, useful, wasted, tokens, batch_utils

def simulate_continuous(requests, max_batch=8):
    remaining  = list(requests)
    tokens     = sum(requests)
    pending    = list(range(len(requests)))
    active     = pending[:max_batch]
    pending    = pending[max_batch:]
    steps, useful, wasted = 0, 0, 0
    while active:
        steps  += 1
        useful += len(active)
        wasted += max_batch - len(active)
        for i in list(active):
            remaining[i] -= 1
            if remaining[i] <= 0:
                active.remove(i)
                if pending:
                    active.append(pending.pop(0))
    return steps, useful, wasted, tokens

requests = make_requests(64, lo=20, hi=400, seed=42)
print("=" * 72)
print(f"Batching Simulation: {len(requests)} requests, output tokens 20-400")
print(f"  min={min(requests)} max={max(requests)} avg={sum(requests)//len(requests)}")
print("=" * 72)
print()
print(f"  {'Method':<34} {'Time':>8} {'Util%':>8} {'Tput':>12} {'vs Static8':>12}")
print("  " + "-" * 78)

static8_t, _, _, toks8, _ = simulate_static(requests, 8)
baseline_tput = toks8 / static8_t

for bs in [4, 8, 16]:
    t, us, ws, toks, _ = simulate_static(requests, bs)
    util  = 100 * us / (us + ws)
    tput  = toks / t
    ratio = tput / baseline_tput
    print(f"  Static batch={bs:<3}                    {t:>8.0f} {util:>8.1f} {tput:>12.2f} {ratio:>11.2f}x")

print()
for bs in [4, 8, 16]:
    t, us, ws, toks = simulate_continuous(requests, bs)
    util  = 100 * us / (us + ws)
    tput  = toks / t
    ratio = tput / baseline_tput
    bar   = "+" * int((ratio - 1) * 20)
    print(f"  Continuous max_batch={bs:<3}              {t:>8.0f} {util:>8.1f} {tput:>12.2f} {ratio:>11.2f}x  {bar}")

print()
print("=" * 72)
print("Batch Waste Visualisation: 8 requests, static batching")
print("=" * 72)
sample_batch = [120, 80, 310, 45, 200, 160, 95, 280]
max_l = max(sample_batch)
print(f"  Sequence lengths: {sample_batch}")
print(f"  Batch runs until longest: {max_l} tokens")
print()
print("  Timeline (= active, . = idle padding):")
for i, sl in enumerate(sample_batch):
    active = int(sl / max_l * 50)
    idle   = 50 - active
    waste  = 100 * (1 - sl / max_l)
    print(f"  Seq{i} [{sl:>3}t] {'='*active}{'.'*idle}  waste={waste:.0f}%")

useful = sum(sample_batch)
total  = max_l * len(sample_batch)
print()
print(f"  Batch utilisation: {useful}/{total} = {100*useful/total:.1f}%")
print(f"  Continuous batching fills the dots immediately with new requests.")
""",
    },

    "7 · FlashAttention IO Complexity Analysis": {
        "description": (
            "Compute standard vs FlashAttention HBM reads/writes and memory footprint. "
            "Show the N^2 memory wall, tiling mechanics, and estimated speedup at various sequence lengths."
        ),
        "language": "python",
        "code": """\
import math

HBM_BW_GBS     = 2000.0
SRAM_MB        = 20.0
TFLOPS_FP16    = 312.0
BYTES_FP16     = 2
HEAD_DIM       = 64

def std_attn_io(N, d, bpe=BYTES_FP16):
    reads  = (4 * N * d + 3 * N * N) * bpe
    writes = (2 * N * N + N * d) * bpe
    return reads, writes

def flash_attn_io(N, d, sram_mb=SRAM_MB, bpe=BYTES_FP16):
    sram_elems = sram_mb * 1e6 / bpe
    block_size = max(1, int(math.sqrt(sram_elems / 4)))
    n_blocks   = math.ceil(N / block_size)
    reads  = n_blocks * n_blocks * block_size * d * 3 * bpe
    writes = N * d * bpe
    return reads, writes

def attn_flops(N, d):
    return 4 * N * N * d + 5 * N * N

def io_time_ms(reads, writes):
    return (reads + writes) / (HBM_BW_GBS * 1e9) * 1e3

def compute_time_ms(flops):
    return flops / (TFLOPS_FP16 * 1e12) * 1e3

def total_time_ms(reads, writes, flops):
    return max(io_time_ms(reads, writes), compute_time_ms(flops))

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

print("=" * 88)
print(f"Standard Attention: HBM IO and Memory  (A100 SXM4, head_dim={HEAD_DIM})")
print("=" * 88)
print(f"  {'N':>7} {'ReadMB':>10} {'WriteMB':>10} {'AttnMatrix':>14} {'Time(ms)':>12} {'Bound':>12}")
print("  " + "-" * 64)
for N in SEQ_LENS:
    r, w = std_attn_io(N, HEAD_DIM)
    fl   = attn_flops(N, HEAD_DIM)
    attn_mb = N * N * BYTES_FP16 / 1e6
    t_ms = total_time_ms(r, w, fl)
    bd   = "IO" if io_time_ms(r, w) > compute_time_ms(fl) else "Compute"
    attn_str = f"{attn_mb:.0f}MB" if attn_mb < 1000 else f"{attn_mb/1e3:.1f}GB"
    print(f"  {N:>7,} {r/1e6:>10.1f} {w/1e6:>10.1f} {attn_str:>14} {t_ms:>12.4f} {bd:>12}")

print()
print("=" * 88)
print("FlashAttention: HBM IO, Memory, and Speedup vs Standard")
print("=" * 88)
print(f"  {'N':>7} {'ReadMB':>10} {'WriteMB':>10} {'Memory':>14} {'Time(ms)':>12} {'Speedup':>10}")
print("  " + "-" * 64)
for N in SEQ_LENS:
    r_s, w_s = std_attn_io(N, HEAD_DIM)
    r_f, w_f = flash_attn_io(N, HEAD_DIM)
    fl       = attn_flops(N, HEAD_DIM)
    t_std    = total_time_ms(r_s, w_s, fl)
    t_flash  = total_time_ms(r_f, w_f, fl)
    mem_mb   = N * HEAD_DIM * BYTES_FP16 / 1e6
    speedup  = t_std / max(t_flash, 1e-12)
    print(f"  {N:>7,} {r_f/1e6:>10.1f} {w_f/1e6:>10.1f} {mem_mb:>13.2f}M {t_flash:>12.4f} {speedup:>9.1f}x")

print()
print("=" * 88)
print("Tiling: How FlashAttention fits attention computation inside SRAM")
print("=" * 88)
print()
for N in [2048, 8192, 32768]:
    sram_elems = SRAM_MB * 1e6 / BYTES_FP16
    block_size = max(1, int(math.sqrt(sram_elems / 4)))
    n_blocks   = math.ceil(N / block_size)
    full_mb    = N * N * BYTES_FP16 / 1e6
    sram_used  = block_size * HEAD_DIM * BYTES_FP16 * 4 / 1e6
    saving     = full_mb / max(sram_used, 0.001)
    print(f"  N={N:<6}: tile={block_size}x{block_size}  {n_blocks}x{n_blocks} tiles  "
          f"full attn={full_mb:.0f}MB vs SRAM/tile={sram_used:.1f}MB  ({saving:.0f}x saving)")

print()
print("  FlashAttention NEVER writes the NxN attention matrix to HBM.")
print("  Each tile is computed entirely in SRAM and discarded after writing output.")
""",
    },

    "8 · GPTQ Layer-Wise Quantization": {
        "description": (
            "Implement the GPTQ core: per-column quantization with Hessian-based error "
            "compensation. Compare RTN (round-to-nearest) vs GPTQ output reconstruction MSE."
        ),
        "language": "python",
        "code": """\
import math, random
random.seed(42)

def mse_list(a, b):
    return sum((x-y)**2 for x, y in zip(a, b)) / max(len(a), 1)

def compute_scale(vals, bits=4):
    q_max = 2**(bits-1) - 1
    absmax = max(abs(v) for v in vals)
    return absmax / q_max if absmax > 0 else 1.0

def qround(w, scale, bits=4):
    q_min = -(2**(bits-1))
    q_max =  (2**(bits-1)) - 1
    q     = max(q_min, min(q_max, round(w / scale)))
    return q * scale

def mat_mul_flat(W, X, out_dim, in_dim, n_cal):
    Y = [[0.0]*n_cal for _ in range(out_dim)]
    for o in range(out_dim):
        for c in range(n_cal):
            s = sum(W[o][j] * X[j][c] for j in range(in_dim))
            Y[o][c] = s
    return Y

def mat_mse(A, B):
    err = 0.0
    n   = 0
    for ra, rb in zip(A, B):
        for a, b in zip(ra, rb):
            err += (a - b)**2
            n   += 1
    return err / max(n, 1)

OUT_DIM = 8
IN_DIM  = 16
N_CAL   = 32

rng = random.Random(42)
W   = [[rng.gauss(0, 0.1) for _ in range(IN_DIM)] for _ in range(OUT_DIM)]
X   = [[rng.gauss(0, 1.0) for _ in range(N_CAL)] for _ in range(IN_DIM)]

# Hessian H = X X^T / N  (IN x IN)
H = [[0.0]*IN_DIM for _ in range(IN_DIM)]
for i in range(IN_DIM):
    for j in range(IN_DIM):
        H[i][j] = sum(X[i][c] * X[j][c] for c in range(N_CAL)) / N_CAL

Y_ref = mat_mul_flat(W, X, OUT_DIM, IN_DIM, N_CAL)

# ── Method 1: Round-to-Nearest (RTN) ─────────────────────────────────────────
def rtn_quant(W, bits=4):
    W_q = []
    for row in W:
        scale = compute_scale(row, bits)
        W_q.append([qround(w, scale, bits) for w in row])
    return W_q

# ── Method 2: GPTQ (column-wise error propagation) ───────────────────────────
def gptq_quant(W, H, bits=4):
    W_q = [list(row) for row in W]
    for o in range(OUT_DIM):
        row   = list(W[o])
        scale = compute_scale(row, bits)
        for j in range(IN_DIM):
            w_orig  = row[j]
            w_quant = qround(row[j], scale, bits)
            err     = w_orig - w_quant
            row[j]  = w_quant
            h_jj    = H[j][j] + 1e-6
            for k in range(j+1, IN_DIM):
                row[k] -= err * H[j][k] / h_jj
        W_q[o] = row
    return W_q

W_rtn  = rtn_quant(W,  bits=4)
W_gptq = gptq_quant(W, H, bits=4)

Y_rtn  = mat_mul_flat(W_rtn,  X, OUT_DIM, IN_DIM, N_CAL)
Y_gptq = mat_mul_flat(W_gptq, X, OUT_DIM, IN_DIM, N_CAL)

print("=" * 68)
print(f"GPTQ vs RTN: {OUT_DIM}x{IN_DIM} layer, INT4, {N_CAL} calibration samples")
print("=" * 68)
print()
print(f"  {'Method':<16} {'Weight MSE':>14} {'Output MSE':>14} {'Improvement':>14}")
print("  " + "-" * 60)

mse_rtn_w  = mat_mse(W, W_rtn)
mse_gptq_w = mat_mse(W, W_gptq)
mse_rtn_y  = mat_mse(Y_ref, Y_rtn)
mse_gptq_y = mat_mse(Y_ref, Y_gptq)
gain       = (mse_rtn_y - mse_gptq_y) / mse_rtn_y * 100

print(f"  {'RTN (naive)':<16} {mse_rtn_w:>14.6e} {mse_rtn_y:>14.6e} {'baseline':>14}")
print(f"  {'GPTQ':<16} {mse_gptq_w:>14.6e} {mse_gptq_y:>14.6e} {gain:>13.1f}%")
print()

print("=" * 68)
print("Per-neuron output MSE: GPTQ vs RTN")
print("=" * 68)
print()
print(f"  {'Row':>5} {'RTN out-MSE':>14} {'GPTQ out-MSE':>14} {'Gain':>10}")
print("  " + "-" * 46)
for o in range(OUT_DIM):
    rtn_e  = mse_list(Y_ref[o], Y_rtn[o])
    gptq_e = mse_list(Y_ref[o], Y_gptq[o])
    gain_r = (rtn_e - gptq_e) / (rtn_e + 1e-15) * 100
    bar    = "+" * max(0, int(gain_r / 5))
    print(f"  {o:>5} {rtn_e:>14.6e} {gptq_e:>14.6e} {gain_r:>9.1f}%  {bar}")

print()
print("  GPTQ propagates each rounding error to remaining columns,")
print("  minimising per-layer output MSE instead of per-weight MSE.")
print("  In practice on LLMs: INT4 GPTQ achieves <1% perplexity increase.")
""",
    },

    "9 · Structured Pruning and Head/Layer Analysis": {
        "description": (
            "Simulate magnitude pruning, structured attention head pruning, and layer dropping. "
            "Show sparsity vs accuracy and real speedup from structured vs unstructured pruning."
        ),
        "language": "python",
        "code": """\
import math, random
random.seed(0)

N_LAYERS = 12
N_HEADS  = 12

rng_h = random.Random(7)
head_imp = [[rng_h.random() * (1 + 2 * rng_h.random()) for _ in range(N_HEADS)]
            for _ in range(N_LAYERS)]
layer_imp = [sum(head_imp[l])/N_HEADS + rng_h.gauss(0, 0.1) for l in range(N_LAYERS)]

def unstructured_prune(weights, sparsity):
    thresh = sorted(abs(w) for w in weights)[int(len(weights) * sparsity)]
    return [w if abs(w) > thresh else 0.0 for w in weights]

def mag_accuracy(sparsity, noise_rng):
    acc = 90.0 * (1 - sparsity) ** 0.4 + noise_rng.gauss(0, 0.4)
    return min(90.0, max(0.0, acc))

def prune_heads(head_imp, frac):
    all_h = [(head_imp[l][h], l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    all_h.sort()
    n_prune = int(len(all_h) * frac)
    pruned  = {(l, h) for _, l, h in all_h[:n_prune]}
    return pruned

def head_acc_proxy(frac, noise_rng):
    return min(90.0, max(0.0, 90.0 * (1 - frac)**0.5 + noise_rng.gauss(0, 0.3)))

def head_speedup(frac):
    remaining = 1 - frac
    mha_speedup = 1 / max(remaining, 0.01)
    return 1 + 0.45 * (mha_speedup - 1)

def layer_drop(layer_imp, frac):
    scores  = sorted(enumerate(layer_imp), key=lambda x: x[1])
    n_drop  = int(len(scores) * frac)
    dropped = {i for i, _ in scores[:n_drop]}
    kept    = [i for i in range(len(layer_imp)) if i not in dropped]
    return dropped, kept

noise_rng = random.Random(99)

print("=" * 72)
print("1. Unstructured Magnitude Pruning (no real GPU speedup on dense hardware)")
print("=" * 72)
print()
print(f"  {'Sparsity':>10} {'Nonzero%':>10} {'Acc Proxy':>12} {'GPU Speedup':>14} {'Mem Save':>10}")
print("  " + "-" * 58)
for sp in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    nz_pct = 100 * (1 - sp)
    acc    = mag_accuracy(sp, noise_rng)
    gpu_sp = "2x" if abs(sp - 0.5) < 0.01 else "1x"   # only NVIDIA 2:4 gives speedup
    mem    = f"{100*sp:.0f}%"
    print(f"  {100*sp:>9.0f}% {nz_pct:>10.1f}% {acc:>11.1f}% {gpu_sp:>14} {mem:>10}")

print()
print("  NVIDIA 2:4 structured sparsity (50%) gives 2x speedup on Ampere+.")
print("  Random sparsity does NOT speed up dense GPU kernels.")
print()

print("=" * 72)
print("2. Structured Attention Head Pruning (12 layers x 12 heads)")
print("=" * 72)
print()
print(f"  {'PruneFrac':>10} {'Pruned':>10} {'Kept':>8} {'Acc':>12} {'Speedup':>12}")
print("  " + "-" * 56)
for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    pruned_set = prune_heads(head_imp, frac)
    acc        = head_acc_proxy(frac, noise_rng)
    speedup    = head_speedup(frac)
    bar        = "=" * int(speedup * 6)
    print(f"  {frac:>10.0%} {len(pruned_set):>10} {N_LAYERS*N_HEADS - len(pruned_set):>8} {acc:>11.1f}% {speedup:>11.2f}x  {bar}")

print()
print("=" * 72)
print("3. Layer Dropping (remove least-important transformer layers)")
print("=" * 72)
print()
print(f"  {'DropFrac':>10} {'Dropped':>10} {'Kept':>8} {'Latency':>12} {'EstAccDrop':>14}")
print("  " + "-" * 58)
for frac in [0.0, 0.08, 0.17, 0.25, 0.33, 0.42]:
    dropped, kept = layer_drop(layer_imp, frac)
    latency_frac  = len(kept) / N_LAYERS
    acc_drop      = frac * 100 * 0.5 + noise_rng.gauss(0, 0.2)
    bar           = "=" * int(latency_frac * 15)
    print(f"  {frac:>10.0%} {len(dropped):>10} {len(kept):>8} {latency_frac:>11.2f}x {acc_drop:>13.1f}%  {bar}")

print()
print("  Layer dropping gives proportional latency reduction and compiles to")
print("  a smaller dense model — no sparse hardware required.")
print("  DistilBERT drops half the layers + fine-tunes: 97% of BERT at 60% speed.")
""",
    },

    "10 · Roofline Model and Optimization Stack Impact": {
        "description": (
            "Build the full inference latency model: arithmetic intensity, roofline analysis, "
            "prefill vs decode phases, and combined optimization stack speedup with cost per 1M tokens."
        ),
        "language": "python",
        "code": """\
import math

HARDWARE = {
    "A100 80GB": {"flops_fp16": 312e12, "bw":  2000e9, "mem_gb": 80,  "usd_hr": 3.0},
    "H100 80GB": {"flops_fp16": 989e12, "bw":  3350e9, "mem_gb": 80,  "usd_hr": 5.0},
    "RTX4090":   {"flops_fp16": 82e12,  "bw":  1008e9, "mem_gb": 24,  "usd_hr": 0.6},
}

# Llama 3 8B architecture
MODEL = {"params": 8e9, "layers": 32, "hidden": 4096, "ffn": 14336,
         "n_heads": 32, "n_kv": 8, "head_dim": 128}

BYTES = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}

def decode_flops(m):
    h, f, l = m["hidden"], m["ffn"], m["layers"]
    per_layer = 2 * (3*h*h + h*h + 2*h*f + 4*h)
    return per_layer * l

def decode_bytes_loaded(m, fmt="fp16"):
    bpe = BYTES[fmt]
    h, f, l = m["hidden"], m["ffn"], m["layers"]
    weights_per_layer = 3*h*h + h*h + 2*h*f
    return weights_per_layer * l * bpe

def arith_intensity(m, fmt="fp16", batch=1):
    return (decode_flops(m) * batch) / decode_bytes_loaded(m, fmt)

def roofline_tps(m, fmt, batch, hw, spec_mult=1.0, flash_mult=1.0):
    intensity    = arith_intensity(m, fmt, batch) * flash_mult
    hw_roof      = hw["flops_fp16"] / hw["bw"]
    flops        = decode_flops(m)
    if intensity >= hw_roof:
        peak_tps = hw["flops_fp16"] / flops
    else:
        peak_tps = hw["bw"] * intensity / flops
    return peak_tps * batch * spec_mult

hw_name = "A100 80GB"
hw      = HARDWARE[hw_name]
hw_roof = hw["flops_fp16"] / hw["bw"]

print("=" * 74)
print(f"Roofline Analysis: Llama 3 8B on {hw_name}")
print(f"  Peak FLOPs: {hw['flops_fp16']/1e12:.0f} TFLOPs  "
      f"BW: {hw['bw']/1e9:.0f} GB/s  "
      f"Roofline: {hw_roof:.1f} FLOPs/byte")
print("=" * 74)
print()
print(f"  {'Format':<10} {'Batch':>8} {'Intensity':>12} {'vs Roof':>10} {'Bottleneck':>14}")
print("  " + "-" * 58)
for fmt in ["fp16", "int8", "int4"]:
    for batch in [1, 8, 32]:
        intensity  = arith_intensity(MODEL, fmt, batch)
        frac_roof  = intensity / hw_roof
        bottleneck = "Compute" if frac_roof >= 1.0 else f"Mem BW ({100*frac_roof:.0f}%)"
        bar        = "=" * min(20, int(frac_roof * 20))
        print(f"  {fmt:<10} {batch:>8} {intensity:>12.1f} {frac_roof:>9.2f}x {bottleneck:>14}  {bar}")
    print()

print("=" * 74)
print("Optimization Stack: Cumulative Throughput Impact")
print(f"  Llama 3 8B on {hw_name}, decode phase")
print("=" * 74)
print()

configs = [
    ("Baseline FP16 batch=1",    "fp16",  1, 1.0, 1.0),
    ("+ INT8 quantization",      "int8",  1, 1.0, 1.0),
    ("+ INT4 quantization",      "int4",  1, 1.0, 1.0),
    ("+ Continuous batch=8",     "int4",  8, 1.0, 1.0),
    ("+ Continuous batch=32",    "int4", 32, 1.0, 1.0),
    ("+ Speculative dec (3.5x)", "int4", 32, 3.5, 1.0),
    ("+ FlashAttention (1.4x)",  "int4", 32, 3.5, 1.4),
]

baseline_tps = roofline_tps(MODEL, "fp16", 1, hw, 1.0, 1.0)
print(f"  {'Configuration':<34} {'TPS':>10} {'Speedup':>10} {'$/1M tok':>12}")
print("  " + "-" * 70)
for name, fmt, batch, spec, flash in configs:
    tps     = roofline_tps(MODEL, fmt, batch, hw, spec, flash)
    speedup = tps / baseline_tps
    cost    = (1e6 / tps) / 3600 * hw["usd_hr"]
    bar     = "=" * min(30, int(math.log2(speedup + 1) * 8))
    print(f"  {name:<34} {tps:>10.0f} {speedup:>9.1f}x {cost:>11.4f}  {bar}")

print()
print("=" * 74)
print("Hardware Comparison at INT4 batch=32 + speculative decoding")
print("=" * 74)
print()
print(f"  {'Hardware':<16} {'TPS':>10} {'$/1M tok':>12} {'Mem (GB)':>10}")
print("  " + "-" * 52)
for hw_n, hw_c in HARDWARE.items():
    tps  = roofline_tps(MODEL, "int4", 32, hw_c, 3.5, 1.4)
    cost = (1e6 / tps) / 3600 * hw_c["usd_hr"]
    print(f"  {hw_n:<16} {tps:>10.0f} {cost:>11.4f} {hw_c['mem_gb']:>10}GB")

print()
print("  The roofline ceiling means INT4+batching eventually saturates compute,")
print("  not memory. Beyond that point, only better hardware or speculative")
print("  decoding can further improve throughput.")
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