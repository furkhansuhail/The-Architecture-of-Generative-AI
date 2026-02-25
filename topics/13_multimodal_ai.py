"""Module: 13 · Multimodal AI"""

DISPLAY_NAME = "13 · Multimodal AI"
ICON         = "👁️"
SUBTITLE     = "CLIP, ViT and VLMs — models that see, hear and reason across modalities"

THEORY = """
## 13 · Multimodal AI

Multimodal AI refers to systems that process, align, and reason over **multiple modalities
simultaneously** — images, text, audio, video, depth maps, etc.  The central challenge is
that each modality has its own structure (pixel grids, token sequences, waveforms) yet
they describe the same world and must be brought into a **shared representational space**.

---

## 1 · Why Multimodal?

### 1.1 The Unimodal Bottleneck

Language models are powerful, but language alone is ambiguous:

- "It's raining cats and dogs" has no literal visual grounding.
- Medical diagnosis from text alone misses visual pathology.
- Robotics needs visual-spatial understanding that text cannot encode.

Humans are inherently multimodal — we use sight, sound, language, and touch together.
AI systems need to do the same to be truly general.

### 1.2 The Alignment Problem

Raw data across modalities is on completely different scales and formats:

| Modality   | Raw format                   | Typical dimension             |
|------------|------------------------------|-------------------------------|
| Image      | H × W × C pixel grid         | 224×224×3 = 150,528 values    |
| Text       | Discrete token IDs           | Variable length sequences     |
| Audio      | 1-D waveform or spectrogram  | 16,000 samples/sec            |
| Video      | T frames × H × W × C         | Millions of values per clip   |

**Goal:** Learn a function `f_m : X_m → ℝᵈ` for each modality `m` such that semantically
related inputs across modalities map to **nearby points** in the shared embedding space ℝᵈ.

### 1.3 Taxonomy of Multimodal Tasks

```
Input → Output Examples
─────────────────────────────────────────────────────────────────────────────────
Image  →    Text          : Image captioning, VQA, OCR
Text   →    Image         : Text-to-image generation (Stable Diffusion, DALL-E)
Image  +    Text → Text   : Visual question answering, visual reasoning
Audio  →    Text          : Automatic Speech Recognition (ASR)
Text   →    Audio         : Text-to-speech synthesis
Video  →    Text          : Video captioning, action recognition
Any    →    Any           : General multimodal models (GPT-4V, Gemini, Claude)
```

---

## 2 · Vision Transformer (ViT)

### 2.1 Motivation: Applying Transformers to Images

CNNs encode images with **local** receptive fields — they struggle with long-range spatial
dependencies.  The Transformer's self-attention is inherently global, but it operates on
sequences.  **ViT (Dosovitskiy et al. 2020)** makes images into sequences by patching.

### 2.2 Patch Embedding

Given an image `x ∈ ℝ^{H×W×C}`, divide it into `N` non-overlapping patches of size `P×P`:

```
N = (H × W) / P²
```

For H=W=224, P=16:  N = (224 × 224) / 256 = **196 patches**.

Each patch is flattened to a vector of dimension `P² × C = 16² × 3 = 768` and linearly
projected to the model dimension `D`:

```
z_i^0 = E · x_i^patch + e_i^pos,     E ∈ ℝ^{D × (P²C)}
```

where `e_i^pos` is a **learnable positional embedding** encoding patch position `i`.

### 2.3 Class Token

A special learnable `[CLS]` token is prepended to the sequence:

```
z^0 = [x_class ; z_1^0 ; z_2^0 ; … ; z_N^0]    ∈ ℝ^{(N+1) × D}
```

After `L` Transformer layers, the `[CLS]` output `z_L^0` aggregates global information
from all patches and is used as the image representation for downstream tasks.

### 2.4 Transformer Layers

Each layer applies:

```
z'_ℓ  = z_{ℓ-1} + MSA(LayerNorm(z_{ℓ-1}))
z_ℓ   = z'_ℓ   + MLP(LayerNorm(z'_ℓ))
```

**Multi-head Self-Attention (MSA)** computes attention across all (N+1) patch tokens:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

where Q, K, V ∈ ℝ^{(N+1)×d_k} are linear projections of the token embeddings.

This means every patch can attend to every other patch — global context from layer 1.

### 2.5 ViT Variants

| Model    | Layers | Heads | D    | Params | Notes                      |
|----------|--------|-------|------|--------|----------------------------|
| ViT-S/16 | 12     | 6     | 384  | 22M    | Small                      |
| ViT-B/16 | 12     | 12    | 768  | 86M    | Base (most common)         |
| ViT-L/16 | 24     | 16    | 1024 | 307M   | Large                      |
| ViT-H/14 | 32     | 16    | 1280 | 632M   | Huge (14×14 patches)       |

**Data hunger**: ViT-B trained only on ImageNet (1.3M images) underperforms ResNet.
Trained on JFT-300M (300M images), it substantially outperforms.  DINO and MAE later
showed self-supervised training can bridge this gap.

### 2.6 Positional Embeddings in ViT

Unlike CNNs, the Transformer has no built-in spatial inductive bias.  Positional
embeddings provide this:

- **Learnable 1-D**: Assign a learnable vector to each position 0…N. ✅ Most common.
- **2-D sin/cos**: Encode row and column separately with sinusoids.
- **Relative**: Encode relative distances between patches (better for different resolutions).
- **RoPE**: Rotary positional embeddings — used in modern VLMs for flexible resolution.

Ablation studies show positional embeddings matter significantly — without them, ViT
treats the image as a bag-of-patches with no spatial structure.

---

## 3 · CLIP — Contrastive Language-Image Pre-training

### 3.1 Core Idea

CLIP (Radford et al. 2021, OpenAI) learns aligned image and text embeddings by training
on **400 million (image, caption) pairs** from the internet.  The learning signal is
purely **contrastive**: matching pairs should be close, mismatched pairs should be far.

### 3.2 Architecture

**Image Encoder** `f(x)`:  ViT-B/32 or ResNet-50/101.  Outputs `ℝᵈ`.
**Text Encoder** `g(t)`:   Transformer (GPT-style, 12 layers, 512-D).  Outputs `ℝᵈ`.

Both encoders project to the **same** embedding space via linear projection heads:
```
v_i = f(x_i) / ‖f(x_i)‖      (L2-normalised image embedding)
u_i = g(t_i) / ‖g(t_i)‖      (L2-normalised text embedding)
```

### 3.3 InfoNCE Contrastive Loss

Given a batch of N (image, text) pairs, form the N×N similarity matrix:

```
S_{ij} = v_i · u_j  · exp(τ)        τ = learnable temperature
```

For each image `i`, the corresponding text `i` is the positive; all others are negatives.
The loss is symmetric cross-entropy:

```
L_image = −(1/N) Σᵢ log [exp(S_{ii}) / Σⱼ exp(S_{ij})]   ← image-to-text
L_text  = −(1/N) Σⱼ log [exp(S_{jj}) / Σᵢ exp(S_{ij})]   ← text-to-image
L_CLIP  = (L_image + L_text) / 2
```

This is **InfoNCE** (Noise-Contrastive Estimation) — it maximises the mutual information
between image and text representations (van den Oord et al. 2018).

### 3.4 Zero-Shot Classification

CLIP's killer feature: classify images **without any labelled training data for new tasks**.

For a dataset with labels `[cat, dog, car, …]`:
1. Encode each label as text: `g("a photo of a {label}")` → text embeddings.
2. Encode the query image: `f(x)` → image embedding.
3. Predict the label with maximum cosine similarity.

```
ŷ = argmax_c  cos(f(x), g("a photo of a " + c))
```

On ImageNet zero-shot, CLIP-ViT-L/14 achieves **76.2% top-1 accuracy** — matching a
supervised ResNet-50 trained on 1.28M labelled ImageNet images.

### 3.5 Prompt Engineering for CLIP

The text template dramatically affects performance:
- "a photo of a {label}" → better than just "{label}" (+3–5%)
- Ensembling 80 templates: "a photo of a big {label}", "a blurry photo of {label}", etc.
  → additional +3.5% on ImageNet zero-shot.

This prompted the field of **prompt engineering** — crafting text inputs to maximise
model performance without gradient updates.

### 3.6 Embedding Space Geometry

Because of L2 normalisation, all embeddings lie on the **unit hypersphere** Sᵈ⁻¹.
Dot product equals cosine similarity.  The contrastive loss pushes:
- Same-concept image/text pairs → cosine similarity close to 1.
- Different-concept pairs → cosine similarity close to 0 (or negative).

Interesting emergent property: arithmetic works in CLIP space (analogous to word2vec):

```
f("king") − f("man") + f("woman") ≈ f("queen")
```

### 3.7 CLIP Limitations

- **Distribution shift**: trained on internet images → biased toward Western content.
- **Compositionality**: "a red cube on top of a blue sphere" ≠ "a blue cube on top of a
  red sphere" — CLIP often confuses spatial relations and attribute bindings.
- **Negation**: CLIP is poor at "a photo without a cat" — negation is hard to learn
  from image-text contrastive training alone.
- **Fine-grained tasks**: struggles with counting, precise localization, text reading.

---

## 4 · Visual Question Answering (VQA)

### 4.1 Task Definition

Given an image `I` and a natural language question `Q`, produce an answer `A`:

```
A = f(I, Q)
```

Examples:
- "What colour is the car?" → "red"
- "How many people are in the image?" → "3"
- "Is the person wearing glasses?" → "yes"

### 4.2 Early Fusion Approaches

**Early (feature-level) fusion:**
1. Extract image features `v = CNN(I)` → ℝ^d
2. Extract question features `q = RNN(Q)` → ℝ^d
3. Combine: `h = v ⊙ q` (element-wise product) or `h = [v; q]`
4. Classify: `A = softmax(W · h)`

The choice of fusion operator matters:
- **Concatenation**: `[v; q]` → simple, misses fine-grained interaction.
- **Element-wise product (Hadamard)**: captures multiplicative interaction.
- **Bilinear**: `v^T W q` → full pairwise interaction but O(d²) parameters.
- **MLB / MFH**: Low-rank bilinear approximations to reduce parameters.

### 4.3 Attention-Based Fusion

Co-attention (Lu et al. 2016): compute attention over image regions conditioned on the
question, and vice versa, iteratively refining both representations.

```
α_v = softmax(W_a [v; q])      ← attend over image regions
α_q = softmax(W_b [q; ṽ])      ← attend over question words  (ṽ = attended image)
```

This allows the model to focus on the image region most relevant to the question.

---

## 5 · Vision-Language Models (VLMs)

### 5.1 The Paradigm Shift

Modern VLMs (2022–present) don't treat vision and language as separate streams to fuse.
Instead, they process visual tokens directly through the **language model's transformer**,
treating image patches as "visual words".

### 5.2 Flamingo (DeepMind, 2022)

Flamingo (Alayrac et al. 2022) introduced **few-shot in-context visual learning**:

**Architecture:**
- Frozen large language model (Chinchilla 70B)
- Frozen CLIP vision encoder
- **Perceiver Resampler**: cross-attention that compresses N visual tokens → fixed 64 tokens
- **Gated cross-attention layers** inserted between LLM layers — only these are trained

```
Visual tokens → Perceiver Resampler → 64 tokens
                                          ↓
[Text] → LM layer → Gated Cross-Attn → LM layer → ... → Output
```

**Gated cross-attention:**
```
y = x + tanh(α) · Attn(x, visual_tokens)
```
where `α` is a learnable scalar initialised to 0 — ensures the model starts as a
pure language model and gradually incorporates visual information.

**In-context learning**: By interleaving `<image> question answer` pairs in the prompt,
Flamingo achieves strong few-shot performance on VQA without any fine-tuning.

### 5.3 LLaVA (Visual Instruction Tuning, 2023)

LLaVA (Liu et al. 2023) showed that a **simple projection** + **instruction tuning** is
surprisingly effective:

```
Image → CLIP ViT-L/14 → Linear Projection W → Visual tokens
Visual tokens + Text tokens → LLaMA/Vicuna → Response
```

Training has two stages:
1. **Stage 1 — Feature alignment**: Freeze LLM + CLIP, train only the projection W
   on image-caption pairs. W learns to map visual features to the LLM's token space.
2. **Stage 2 — Instruction tuning**: Unfreeze LLM, train on visual instruction pairs
   (GPT-4 generated Q&A from image descriptions).

Despite its simplicity (a single linear layer bridging ViT and LLM), LLaVA matched or
exceeded much more complex architectures on several benchmarks.

### 5.4 BLIP-2 (Salesforce, 2023)

BLIP-2 introduces the **Q-Former** (Querying Transformer) — a lightweight bridge module:

```
Frozen Image Encoder → Q-Former (32 learnable query tokens) → Frozen LLM
```

The Q-Former has two attention pathways:
- **Self-attention** among 32 query tokens (can attend to each other).
- **Cross-attention** between query tokens and image features (extracts relevant info).

The 32 output query tokens become a compressed visual representation — much smaller than
the full 196 ViT patch tokens, reducing the computational burden on the LLM.

Training is three-stage:
1. Vision-language representation learning (ITC + ITG + ITM losses).
2. Bootstrap from frozen LLM with Q-Former output.
3. Full instruction tuning.

### 5.5 Visual Tokenisation: Continuous vs Discrete

**Continuous tokens (most VLMs):**
- Image features are real-valued vectors passed directly as soft tokens to the LLM.
- No information loss from quantisation.
- Cannot be stored or transmitted as text.

**Discrete tokens (e.g. DALL-E, Chameleon):**
- VQ-VAE or VQGAN discretises image patches into token IDs.
- Image becomes a sequence of integers, just like text tokens.
- A single unified vocabulary covers both modalities.
- Enables truly unified autoregressive modelling.

### 5.6 Instruction Tuning for VLMs

Raw pre-training gives a model that completes sequences — not one that follows instructions.
**Instruction tuning** fine-tunes on (instruction, input, output) triples:

```
System: "You are a helpful visual assistant."
User:   [image] "What is unusual about this image?"
Assistant: "The cat appears to be wearing sunglasses and reading a newspaper."
```

LLaVA-1.5 (2023), InstructBLIP, and MiniGPT-4 all demonstrate that the quality and
diversity of instruction data matters far more than its quantity.

---

## 6 · Cross-Modal Attention

### 6.1 How VLMs Fuse Modalities

Once visual tokens are in the LLM's embedding space, the standard **transformer
self-attention** handles cross-modal fusion naturally:

```
Sequence: [IMG_1, IMG_2, ..., IMG_N, TEXT_1, TEXT_2, ...]
```

Each token attends to every other token regardless of modality. The attention pattern
reveals which image regions the model focuses on when generating each word.

### 6.2 Cross-Attention vs Self-Attention Fusion

| Approach              | How                                        | Used in            |
|-----------------------|--------------------------------------------|--------------------|
| Self-attention fusion | Concatenate visual + text tokens → unified | LLaVA, GPT-4V      |
| Cross-attention fusion| Text queries attend to visual keys/values  | Flamingo, BLIP-2   |

**Self-attention fusion** is simpler and leverages the full LLM capacity.
**Cross-attention fusion** is more efficient — frozen LLM is not burdened with raw
image tokens; the Q-Former extracts what it needs.

### 6.3 Attention Patterns in VLMs

When a VLM answers "What colour is the dog?":
- Text token "colour" attends strongly to patch tokens covering the dog.
- Text token "dog" attends to patches with high-level object semantics.
- Later layers show more focused, task-relevant attention.

This can be visualised via **attention rollout** — multiplying attention matrices across
layers to get the effective receptive field of each output token.

---

## 7 · Audio and Speech Modalities

### 7.1 Audio Representation

Raw audio (16 kHz waveform) → **log-mel spectrogram**:
1. Frame the signal into overlapping windows (~25ms, 10ms hop).
2. Apply FFT to each window → power spectrum.
3. Apply 80 mel-filterbank → 80-D vector per frame.
4. Take log → log-mel spectrogram of shape (T_frames × 80).

The log-mel spectrogram is a 2-D "image" of frequency vs time — so 2-D CNNs or ViTs
apply directly.

### 7.2 Whisper (OpenAI, 2022)

Whisper is a **weakly supervised speech recognition** model trained on 680,000 hours of
internet audio with transcription labels.

**Architecture:** Encoder-Decoder Transformer.
- **Encoder**: ViT-style — 2-layer CNN for initial feature extraction, then Transformer
  on 30-second log-mel spectrograms (1500 time frames after striding).
- **Decoder**: GPT-style autoregressive Transformer generating text tokens.

**Key design choices:**
- A single model handles: multilingual ASR, translation (speech → English), language ID,
  and voice activity detection — all via special task tokens in the decoder prompt.
- No language model fusion at inference — the model end-to-end maps audio → text.

### 7.3 Audio-Visual Learning

**AV-HuBERT**: Learns lip-reading and speech together. The model is forced to predict
masked audio features using visual lip motion — and vice versa.

**AudioCLIP**: Extends CLIP to audio — trains a three-way contrastive loss between
images, text, and audio embeddings so that a dog image, the word "dog", and a barking
sound all map to nearby points.

---

## 8 · Multimodal Large Language Models (MLLMs)

### 8.1 GPT-4V and Successors

GPT-4V (2023) and successors (Claude 3, Gemini, LLaVA-1.6, etc.) represent the
convergence of LLMs and multimodal perception:

- Accept interleaved text and images in context.
- Support multiple images per conversation.
- Reason about relationships between images.
- Handle document understanding (charts, tables, screenshots).

Architectural details are proprietary, but the general approach follows:
- High-resolution image encoding (tiling: split into crops + global thumbnail).
- A large number of visual tokens per image.
- Instruction-tuned on diverse visual reasoning datasets.

### 8.2 High-Resolution Strategies

Naive ViT at 224×224 loses fine details needed for OCR, chart reading, etc.
Modern VLMs use tiling:

```
Original image
    ↓
Split into 2×2 (or adaptive) crops + 1 global thumbnail
    ↓
Each crop → ViT encoder → visual tokens
    ↓
All crops + thumbnail tokens concatenated → LLM
```

LLaVA-1.5 and InternVL use up to 672×672 resolution this way.

### 8.3 Unified Multimodal Models

**Gemini (Google, 2023)**: Natively multimodal — trained on interleaved text, image,
audio, and video from the start (not retrofitted from a text-only LLM).  Processes up to
1M tokens in context (Gemini 1.5 Pro).

**Chameleon (Meta, 2024)**: Fully discrete — image patches tokenised with VQVAE and
treated identically to text tokens.  A single causal Transformer autoregressively
generates both text and image tokens.  True modality-agnostic architecture.

---

## 9 · Training Objectives for Multimodal Models

### 9.1 Contrastive (ITC — Image-Text Contrastive)

Pull matching pairs together, push non-matching apart.
Used in: CLIP, ALIGN, SigLIP.

```
L_ITC = InfoNCE(image_embeds, text_embeds)
```

**SigLIP** (Zhai et al. 2023) replaces softmax-normalised InfoNCE with **sigmoid loss**:
```
L_SigLIP = −Σ_{ij} [ y_{ij} log σ(z_{ij}) + (1−y_{ij}) log(1−σ(z_{ij})) ]
```
where `y_{ij} = 1` if (i,j) is a matching pair. This removes the dependency on batch
size normalisation and scales better to very large batches.

### 9.2 Image-Text Matching (ITM)

Binary classification: given an (image, text) pair, predict whether they match.
Used in: BLIP, BLIP-2, X-VLM.

```
L_ITM = CrossEntropy(Match(image, text), y ∈ {0,1})
```

Hard negative mining: use in-batch negatives with the highest ITC similarity scores as
hard negatives — these are semantically close but don't match.

### 9.3 Image-Grounded Text Generation (ITG / Captioning)

Autoregressive language modelling conditioned on the image:
```
L_ITG = −Σ_t log p(w_t | w_{<t}, image)
```

This forces the model to learn grounded language generation — using visual content to
produce relevant text.

### 9.4 Masked Image Modelling (MIM)

Extend BERT-style masked LM to images — mask patches and predict:
- **BEiT**: Predict discrete DALL-E token IDs of masked patches.
- **MAE**: Predict raw pixel values of masked patches (75% mask ratio).
- **IBOT / iBOT**: Predict online teacher's patch features (self-distillation).

MAE (He et al. 2022) shows that 75% masking ratio forces genuinely semantic
reconstruction (vs 15% for BERT) because so little spatial context remains.

### 9.5 Combined Objectives (BLIP Framework)

BLIP trains jointly with ITC + ITM + ITG — the three objectives are complementary:
- ITC aligns the embedding space globally.
- ITM provides fine-grained matching discrimination.
- ITG grounds language generation in visual content.

---

## 10 · Evaluation Benchmarks

### 10.1 Vision-Language Benchmarks

| Benchmark    | Task                            | Metric       |
|--------------|---------------------------------|--------------|
| VQAv2        | Open-ended VQA (balanced)       | Accuracy     |
| GQA          | Compositional scene graph QA    | Accuracy     |
| TextVQA      | VQA requiring OCR               | Accuracy     |
| POPE         | Hallucination (object presence) | F1           |
| MMBench      | Multi-task VLM evaluation       | Accuracy     |
| MMMU         | College-level multimodal reasoning | Accuracy  |
| NoCaps       | Image captioning (zero-shot)    | CIDEr        |

### 10.2 CLIP Evaluation

- **Zero-shot ImageNet**: Top-1 accuracy without any ImageNet training.
- **Linear probing**: Train a linear classifier on frozen CLIP features.
- **Retrieval**: Image-to-text and text-to-image R@1/R@5/R@10 on MSCOCO/Flickr30k.

### 10.3 Hallucination in VLMs

A critical problem: VLMs confidently describe objects, attributes, or relationships that
are **not present** in the image — hallucinating based on language priors.

Example: An image with no banana → "there is a yellow banana on the table" (language
prior from "fruit bowl" context).

**CHAIR metric** (Caption Hallucination Assessment with Image Relevance):
```
CHAIR_s = |hallucinated objects in captions| / |total objects mentioned|
```

Mitigation strategies:
- POPE benchmark (binary presence/absence questions, harder to hallucinate on).
- RLHF-style feedback on factual grounding.
- Contrastive decoding: subtract language-only model's logits from VLM logits.

---

## 11 · Key Takeaways

- **Multimodal AI** bridges different data modalities — images, text, audio — by learning
  shared embedding spaces where semantically related content is geometrically close.

- **ViT** makes Transformers applicable to images via patch tokenisation.  Each image
  becomes a sequence of patch embeddings with positional information, and global
  self-attention captures spatial relationships from the first layer.

- **CLIP** uses contrastive learning on 400M image-text pairs to learn aligned embeddings
  enabling powerful zero-shot transfer via cosine similarity between text and image encodings.

- **VLMs** (Flamingo, LLaVA, BLIP-2) connect a frozen or fine-tuned vision encoder to a
  large language model.  The bridge module (linear projection, Q-Former, Perceiver) is
  key — it must compress visual information into a form the LLM can use.

- **Modern MLLMs** (GPT-4V, Gemini, Claude 3) use high-resolution tiling, large context
  windows, and instruction tuning to achieve human-level visual reasoning on many tasks.

- **Training objectives** — contrastive (ITC), matching (ITM), and generative (ITG) —
  are typically combined because they are complementary: alignment, discrimination, and
  grounded generation.

- **Hallucination** remains an open challenge: models confuse language priors with visual
  evidence, generating plausible but factually wrong descriptions.

- The field is converging toward **unified models** (Gemini, Chameleon) where modalities
  are first-class citizens in the same architecture rather than add-ons to a text LLM.
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────
OPERATIONS = {

    # ── 1 ────────────────────────────────────────────────────────────────────
    "1 · ViT Patch Embedding Visualisation": {
        "description": (
            "Demonstrates how ViT divides an image into fixed-size patches and "
            "creates a sequence of patch embeddings.  Shows the effect of patch size "
            "on sequence length and information granularity."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

# ── Synthetic 'image' (gradient + texture to show patch structure) ───────
H, W, C = 224, 224, 3
x_img = np.zeros((H, W, C))
for i in range(H):
    for c in range(C):
        x_img[i, :, c] = (i / H) * [0.8, 0.3, 0.1][c] + \
                          (np.arange(W) / W) * [0.1, 0.4, 0.9][c]
# Add structured "features" (blobs)
for _ in range(12):
    cy, cx = np.random.randint(20, 200, 2)
    r = np.random.randint(8, 25)
    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy)**2 + (xx - cx)**2 < r**2
    col  = np.random.rand(3)
    x_img[mask] = col
x_img = np.clip(x_img, 0, 1)

# ── Patch sizes to compare ───────────────────────────────────────────────
patch_sizes = [32, 16, 8]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(x_img)
axes[0].set_title('Original Image\n224×224×3', fontweight='bold')
axes[0].axis('off')

for ax, P in zip(axes[1:], patch_sizes):
    N = (H // P) * (W // P)
    ax.imshow(x_img)
    # Draw patch grid
    for y in range(0, H, P):
        ax.axhline(y, color='white', lw=0.6, alpha=0.8)
    for x in range(0, W, P):
        ax.axvline(x, color='white', lw=0.6, alpha=0.8)
    # Highlight a few patches
    for py in range(0, H, P*4):
        for px in range(0, W, P*4):
            rect = mpatches.Rectangle((px, py), P, P,
                                       linewidth=2, edgecolor='yellow',
                                       facecolor='yellow', alpha=0.25)
            ax.add_patch(rect)
    ax.set_title(f'Patch size = {P}×{P}\\n'
                 f'N = {N} patches\\n'
                 f'Seq len = {N+1} (+ [CLS])', fontweight='bold')
    ax.axis('off')

plt.suptitle('ViT Patch Tokenisation — Effect of Patch Size on Sequence Length',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('vit_patches.png', dpi=130, bbox_inches='tight')
plt.show()

print("Patch size | N patches | Sequence length | Patch dim (RGB)")
print("-----------|-----------|-----------------|----------------")
for P in [32, 16, 8]:
    N = (224 // P)**2
    d = P * P * 3
    print(f"   {P:3d}×{P:3d} |   {N:5d}   |     {N+1:4d}        | {d:5d}")
""",
    },

    # ── 2 ────────────────────────────────────────────────────────────────────
    "2 · Positional Encoding — 2D Sinusoidal vs Learnable": {
        "description": (
            "Compares 2-D sinusoidal positional encodings (row + column) with "
            "random learnable encodings.  Visualises the encoding similarity "
            "matrix to show spatial structure captured by each approach."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

# ── 14×14 grid of patches (ViT-L/14 style) ──────────────────────────────
G  = 14          # grid size
N  = G * G       # 196 patches
D  = 128         # encoding dimension

positions = [(r, c) for r in range(G) for c in range(G)]  # (row, col) for each patch

# ── 2-D Sinusoidal encoding ───────────────────────────────────────────────
def sin2d_encoding(positions, D):
    '''Encode (row, col) with sinusoids of different frequencies.'''
    enc = np.zeros((len(positions), D))
    D_half = D // 2
    for i, (r, c) in enumerate(positions):
        for k in range(D_half // 2):
            freq = 1 / (10000 ** (2 * k / D_half))
            enc[i, 4*k]     = np.sin(r * freq)
            enc[i, 4*k + 1] = np.cos(r * freq)
            enc[i, 4*k + 2] = np.sin(c * freq)
            enc[i, 4*k + 3] = np.cos(c * freq)
    return enc

# ── Random (untrained) learnable ─────────────────────────────────────────
np.random.seed(7)
sin_enc  = sin2d_encoding(positions, D)
rand_enc = np.random.randn(N, D) * 0.02    # initialised small, as is standard

# Normalise for fair comparison
sin_enc  = sin_enc  / (np.linalg.norm(sin_enc,  axis=1, keepdims=True) + 1e-8)
rand_enc = rand_enc / (np.linalg.norm(rand_enc, axis=1, keepdims=True) + 1e-8)

# ── Similarity matrices ───────────────────────────────────────────────────
S_sin  = sin_enc  @ sin_enc.T     # (196, 196)
S_rand = rand_enc @ rand_enc.T

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im0 = axes[0].imshow(S_sin, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_title('2D Sinusoidal Encoding\nCosine Similarity Matrix',
                  fontweight='bold')
axes[0].set_xlabel('Patch index'); axes[0].set_ylabel('Patch index')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(S_rand, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title('Random Learnable Encoding\nCosine Similarity Matrix',
                  fontweight='bold')
axes[1].set_xlabel('Patch index')
plt.colorbar(im1, ax=axes[1])

# Show the similarity of one centre patch to all others (spatial structure)
centre_idx = (G//2) * G + (G//2)    # patch at (7,7)
sim_sin_2d  = S_sin[centre_idx].reshape(G, G)
im2 = axes[2].imshow(sim_sin_2d, cmap='hot')  # capture return value for colorbar
axes[2].set_title(f'Sinusoidal: Similarity of\nPatch ({G//2},{G//2}) to All Others',
                  fontweight='bold')
axes[2].set_xlabel('Column'); axes[2].set_ylabel('Row')
plt.colorbar(im2, ax=axes[2])  # use im2 so colorbar reflects actual data range

plt.suptitle('ViT Positional Encoding Comparison (14×14 patch grid)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=130, bbox_inches='tight')
plt.show()
print("Sinusoidal encoding shows clear spatial structure (nearby patches similar).")
print("Random encoding shows near-orthogonal vectors (no spatial structure until trained).")
""",
    },

    # ── 3 ────────────────────────────────────────────────────────────────────
    "3 · InfoNCE Contrastive Loss (CLIP Objective)": {
        "description": (
            "Implements the symmetric InfoNCE loss used in CLIP training. "
            "Shows the N×N similarity matrix, how loss evolves as embeddings align, "
            "and the role of the temperature parameter τ."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def infonce_loss(img_emb, txt_emb, tau=0.07):
    '''Symmetric InfoNCE (CLIP) loss.  Embeddings should be L2-normalised.'''
    N   = img_emb.shape[0]
    # Cosine similarity matrix (already L2-normalised)
    S   = img_emb @ txt_emb.T / tau        # (N, N)
    # Image-to-text loss
    labels = np.arange(N)
    def cross_entropy(logits, labels):
        logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
        log_softmax = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        return -np.mean(log_softmax[np.arange(N), labels])
    L_i2t = cross_entropy(S,   labels)
    L_t2i = cross_entropy(S.T, labels)
    return (L_i2t + L_t2i) / 2, S

# ── Simulate training: embeddings start random, gradually align ──────────
N = 16
D = 64

# "True" image/text embeddings (perfect alignment) — ground truth target
img_true = rng.standard_normal((N, D))
img_true /= np.linalg.norm(img_true, axis=1, keepdims=True)
txt_true  = img_true.copy()    # perfectly aligned

# Start from random misaligned embeddings
img_emb = rng.standard_normal((N, D)) * 0.5 + img_true * 0.5
txt_emb = rng.standard_normal((N, D)) * 0.5 + txt_true * 0.5
img_emb /= np.linalg.norm(img_emb, axis=1, keepdims=True)
txt_emb /= np.linalg.norm(txt_emb, axis=1, keepdims=True)

lr = 0.05
losses, sim_diag = [], []
n_steps = 120
tau_default = 0.07   # CLIP default temperature — referenced everywhere below

for step in range(n_steps):
    L, S = infonce_loss(img_emb, txt_emb, tau=tau_default)
    losses.append(float(L))
    sim_diag.append(float(np.mean(np.diag(S) * tau_default)))  # undo /tau to recover cosine sims

    # Gradient step toward true embeddings (simulating optimiser)
    alpha  = lr * (1 - step / n_steps)
    img_emb = (1 - alpha) * img_emb + alpha * img_true
    txt_emb = (1 - alpha) * txt_emb + alpha * txt_true
    img_emb /= np.linalg.norm(img_emb, axis=1, keepdims=True)
    txt_emb /= np.linalg.norm(txt_emb, axis=1, keepdims=True)

# ── Temperature sensitivity ───────────────────────────────────────────────
taus = np.logspace(-2, 0.5, 60)
tau_losses = []
for tau in taus:
    L, _ = infonce_loss(img_emb, txt_emb, tau=float(tau))
    tau_losses.append(float(L))

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Final similarity matrix
_, S_final = infonce_loss(img_emb, txt_emb, tau=tau_default)
im = axes[0].imshow(S_final * tau_default, cmap='viridis')  # *tau_default recovers cosine sims
axes[0].set_title('Final Similarity Matrix S_ij\n(diagonal = matching pairs)',
                  fontweight='bold')
axes[0].set_xlabel('Text index'); axes[0].set_ylabel('Image index')
plt.colorbar(im, ax=axes[0])
# Highlight diagonal
for i in range(N):
    axes[0].add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                                     fill=False, edgecolor='red', lw=2))

axes[1].plot(losses, color='royalblue', lw=2)
axes[1].set_title('InfoNCE Loss vs Training Step', fontweight='bold')
axes[1].set_xlabel('Step'); axes[1].set_ylabel('Loss')
axes[1].grid(alpha=.3)
ax2 = axes[1].twinx()
ax2.plot(sim_diag, color='tomato', lw=1.5, ls='--', label='Diag similarity')
ax2.set_ylabel('Avg diagonal cosine sim', color='tomato')

axes[2].semilogx(taus, tau_losses, color='forestgreen', lw=2.5)
axes[2].axvline(tau_default, color='k', ls='--', lw=1.2, label=f'τ={tau_default} (CLIP default)')
axes[2].set_title('Temperature τ Sensitivity', fontweight='bold')
axes[2].set_xlabel('Temperature τ (log scale)')
axes[2].set_ylabel('InfoNCE Loss')
axes[2].legend(); axes[2].grid(alpha=.3)

plt.suptitle('CLIP: InfoNCE Contrastive Loss', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('infonce_clip.png', dpi=130, bbox_inches='tight')
plt.show()

print(f"Initial loss:  {losses[0]:.4f}  (random embeddings)")
print(f"Final loss:    {losses[-1]:.4f}  (aligned embeddings)")
print(f"Optimal τ (approx): {taus[np.argmin(tau_losses)]:.4f}")
""",
    },

    # ── 4 ────────────────────────────────────────────────────────────────────
    "4 · Zero-Shot Classification with CLIP Embeddings": {
        "description": (
            "Simulates CLIP zero-shot classification: encodes class names as text, "
            "encodes a query 'image' as a visual feature, and classifies via "
            "cosine similarity — no training on the target classes."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

rng = np.random.default_rng(7)

# ── Simulate a CLIP-like embedding space ─────────────────────────────────
# 10 visual concepts, each at a known location in 64-D space
D = 64
concepts = [
    'cat', 'dog', 'car', 'airplane', 'banana',
    'guitar', 'mountain', 'ocean', 'fire', 'book'
]
K = len(concepts)

# "True" concept embeddings (orthogonal-ish directions)
concept_vecs = rng.standard_normal((K, D))
concept_vecs /= np.linalg.norm(concept_vecs, axis=1, keepdims=True)

def encode_text(label, noise=0.3):
    '''Simulate text encoder: concept embedding + small noise.'''\
    idx = concepts.index(label)
    v   = concept_vecs[idx] + rng.standard_normal(D) * noise
    return v / np.linalg.norm(v)

def encode_image(true_label, noise=0.4):
    '''Simulate image encoder: concept embedding + larger noise.'''\
    idx = concepts.index(true_label)
    v   = concept_vecs[idx] + rng.standard_normal(D) * noise
    return v / np.linalg.norm(v)

# ── Prompt templates (CLIP ensembling trick) ──────────────────────────────
templates = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of {}",
    "a photograph of {}",
    "a {} in the wild",
]

def encode_text_ensemble(label):
    '''Average embedding across multiple prompt templates (noise variation simulates template diversity).'''\
    vecs = np.stack([encode_text(label, noise=0.2) for _ in templates])  # pass label directly; each call varies noise
    avg  = vecs.mean(axis=0)
    return avg / np.linalg.norm(avg)

# ── Zero-shot classification ──────────────────────────────────────────────
n_queries = 100
single_correct, ensemble_correct = 0, 0
results = []

for _ in range(n_queries):
    true_label = concepts[rng.integers(K)]
    img_emb    = encode_image(true_label, noise=0.5)

    # Single-template scores
    txt_single = np.stack([encode_text("a photo of a " + c) for c in concepts])
    scores_s   = img_emb @ txt_single.T
    pred_s     = concepts[np.argmax(scores_s)]

    # Ensemble-template scores
    txt_ens    = np.stack([encode_text_ensemble(c) for c in concepts])
    scores_e   = img_emb @ txt_ens.T
    pred_e     = concepts[np.argmax(scores_e)]

    single_correct   += (pred_s == true_label)
    ensemble_correct += (pred_e == true_label)
    results.append((true_label, pred_s, pred_e, scores_e))

# ── Confusion-style analysis on final batch ───────────────────────────────
# Show similarity matrix for a fixed test image of each class
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Class-by-class similarity heatmap
sim_matrix = np.zeros((K, K))
for i, concept in enumerate(concepts):
    img_emb = encode_image(concept, noise=0.35)
    txt_ens = np.stack([encode_text_ensemble(c) for c in concepts])
    sim_matrix[i] = img_emb @ txt_ens.T

im = axes[0].imshow(sim_matrix, cmap='Blues')
axes[0].set_xticks(range(K)); axes[0].set_xticklabels(concepts, rotation=45, ha='right', fontsize=8)
axes[0].set_yticks(range(K)); axes[0].set_yticklabels(concepts, fontsize=8)
axes[0].set_title('Image-Text Cosine Similarities\\n(diagonal = correct match)',
                  fontweight='bold')
plt.colorbar(im, ax=axes[0])
# Highlight diagonal
for i in range(K):
    axes[0].add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                                     fill=False, edgecolor='red', lw=2.5))

# 2. Accuracy comparison
bar_vals = [single_correct, ensemble_correct]
bars = axes[1].bar(['Single Template', 'Ensembled (5×)'], bar_vals,
                    color=['steelblue', 'darkorange'], width=0.45)
axes[1].set_ylim(0, 110)
axes[1].set_ylabel('Correct / 100 queries')
axes[1].set_title('Zero-Shot Accuracy:\\nSingle Template vs Ensembling',
                  fontweight='bold')
for bar, val in zip(bars, bar_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
                 ha='center', fontweight='bold', fontsize=12)
axes[1].grid(axis='y', alpha=.3)

# 3. Per-class retrieval accuracy
per_class_acc = {c: 0 for c in concepts}
per_class_n   = {c: 0 for c in concepts}
rng2 = np.random.default_rng(99)
for _ in range(500):
    true_label = concepts[rng2.integers(K)]
    img_emb    = encode_image(true_label, noise=0.5)
    txt_ens    = np.stack([encode_text_ensemble(c) for c in concepts])
    pred       = concepts[np.argmax(img_emb @ txt_ens.T)]
    per_class_n[true_label] += 1
    per_class_acc[true_label] += (pred == true_label)
accs = [per_class_acc[c] / max(per_class_n[c], 1) * 100 for c in concepts]
bars2 = axes[2].barh(concepts, accs, color=cm.viridis(np.array(accs)/100))
axes[2].set_xlim(0, 110)
axes[2].set_xlabel('Zero-Shot Accuracy (%)')
axes[2].set_title('Per-Class Zero-Shot Accuracy', fontweight='bold')
axes[2].axvline(np.mean(accs), color='red', ls='--', lw=1.5,
                label=f'Mean = {np.mean(accs):.1f}%')
axes[2].legend(fontsize=9); axes[2].grid(axis='x', alpha=.3)

plt.suptitle('CLIP Zero-Shot Classification Simulation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('clip_zeroshot.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Single-template accuracy : {single_correct}%")
print(f"Ensemble (5×) accuracy   : {ensemble_correct}%")
print(f"Improvement from ensembling: +{ensemble_correct - single_correct}%")
""",
    },

    # ── 5 ────────────────────────────────────────────────────────────────────
    "5 · Self-Attention Patterns in ViT": {
        "description": (
            "Visualises how multi-head self-attention works in ViT. "
            "Computes attention weights for a synthetic image and shows "
            "which patches the [CLS] token attends to — simulating attention rollout."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ── Synthetic 7×7 patch grid (49 patches + CLS) ──────────────────────────
G   = 7      # grid size
N   = G * G  # 49 patches
D   = 64     # embedding dim
n_heads = 4
d_head  = D // n_heads

# ── Create patch embeddings with spatial structure ────────────────────────
# Place an "object" (high activation) at patches (2,2)–(4,4)
emb = rng.standard_normal((N + 1, D)) * 0.1  # background noise

# Simulate a salient object in top-left quadrant
for r in range(2, 5):
    for c in range(2, 5):
        idx = r * G + c + 1   # +1 for CLS token
        emb[idx] += rng.standard_normal(D) * 1.5  # strong signal

# CLS token
emb[0] = rng.standard_normal(D) * 0.5

# Add positional encodings
for pos in range(N):
    r, c = divmod(pos, G)
    emb[pos+1] += np.array([np.sin(r/7), np.cos(r/7),
                              np.sin(c/7), np.cos(c/7)] * (D//4))

# ── Compute multi-head self-attention ────────────────────────────────────
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)

head_attentions = []
for h in range(n_heads):
    Wq = rng.standard_normal((D, d_head)) * 0.1
    Wk = rng.standard_normal((D, d_head)) * 0.1
    Q  = emb @ Wq       # (N+1, d_head)
    K  = emb @ Wk
    A  = softmax(Q @ K.T / np.sqrt(d_head))   # (N+1, N+1)
    head_attentions.append(A)

# ── Attention rollout (multiply attention matrices across heads) ──────────
avg_att  = np.mean(head_attentions, axis=0)     # average across heads
# Add residual connection (identity + attention) / 2
rollout  = 0.5 * avg_att + 0.5 * np.eye(N + 1)
rollout /= rollout.sum(axis=-1, keepdims=True)

# CLS token's effective attention to all patches after rollout
cls_att = rollout[0, 1:]    # (N,) — ignore CLS→CLS
cls_att_grid = cls_att.reshape(G, G)

# Per-head CLS attention
head_cls = [A[0, 1:].reshape(G, G) for A in head_attentions]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Full attention matrix
im = axes[0, 0].imshow(avg_att, cmap='Blues')
axes[0, 0].set_title('Avg Attention Matrix\n(all tokens)', fontweight='bold')
axes[0, 0].set_xlabel('Key token'); axes[0, 0].set_ylabel('Query token')
plt.colorbar(im, ax=axes[0, 0])

# Per-head CLS attention maps
for h in range(n_heads):
    ax = axes[0, h+1] if h < 3 else axes[1, 0]
    im = ax.imshow(head_cls[h], cmap='hot', vmin=0)
    ax.set_title(f'Head {h+1}: CLS Attention Map', fontweight='bold')
    ax.set_xlabel('Column'); ax.set_ylabel('Row')
    # Mark the "object" region
    rect = plt.Rectangle((1.5, 1.5), 3, 3, fill=False, edgecolor='cyan', lw=2)
    ax.add_patch(rect)
    plt.colorbar(im, ax=ax)

# Rollout
im2 = axes[1, 1].imshow(cls_att_grid, cmap='hot')
axes[1, 1].set_title('Attention Rollout\\n(effective receptive field of CLS)',
                      fontweight='bold')
axes[1, 1].set_xlabel('Column'); axes[1, 1].set_ylabel('Row')
rect2 = plt.Rectangle((1.5, 1.5), 3, 3, fill=False, edgecolor='cyan', lw=2.5,
                        label='Object region')
axes[1, 1].add_patch(rect2)
axes[1, 1].legend(fontsize=8)
plt.colorbar(im2, ax=axes[1, 1])

# CLS attention bar chart
axes[1, 2].bar(range(N), cls_att, color='steelblue', alpha=0.7)
axes[1, 2].set_title('CLS Token Attention to Each Patch', fontweight='bold')
axes[1, 2].set_xlabel('Patch index'); axes[1, 2].set_ylabel('Attention weight')
# Highlight object patches
obj_patches = [r*G + c for r in range(2,5) for c in range(2,5)]
for p in obj_patches:
    axes[1, 2].axvspan(p-0.5, p+0.5, alpha=0.3, color='orange')
axes[1, 2].grid(axis='y', alpha=.3)

# Head agreement (std across heads)
head_std = np.std(head_cls, axis=0)
im3 = axes[1, 3].imshow(head_std, cmap='RdPu')
axes[1, 3].set_title('Std Across Heads\\n(high = heads disagree)', fontweight='bold')
plt.colorbar(im3, ax=axes[1, 3])

plt.suptitle('ViT Self-Attention Patterns & Attention Rollout',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('vit_attention.png', dpi=130, bbox_inches='tight')
plt.show()

top_k = np.argsort(cls_att)[-5:][::-1]
print("Top-5 patches attended to by [CLS] token:")
for rank, idx in enumerate(top_k):
    r, c = divmod(idx, G)
    in_obj = "← IN OBJECT" if idx in obj_patches else ""
    print(f"  Rank {rank+1}: patch {idx:2d} (row={r}, col={c})  "
          f"att={cls_att[idx]:.4f}  {in_obj}")
""",
    },

    # ── 6 ────────────────────────────────────────────────────────────────────
    "6 · Multimodal Embedding Space Geometry": {
        "description": (
            "Simulates a shared image-text embedding space. Demonstrates "
            "cross-modal retrieval (image query → top-k texts), arithmetic "
            "in embedding space, and the semantic neighbourhood structure."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as skPCA

rng = np.random.default_rng(0)

# ── Build a toy shared CLIP-like embedding space ──────────────────────────
D = 128
concepts = {
    'animals':   ['cat', 'dog', 'bird', 'fish', 'lion', 'elephant'],
    'vehicles':  ['car', 'truck', 'airplane', 'bicycle', 'boat', 'train'],
    'food':      ['apple', 'pizza', 'sushi', 'cake', 'bread', 'coffee'],
    'nature':    ['mountain', 'ocean', 'forest', 'desert', 'river', 'sky'],
}
category_colors = {'animals': 'royalblue', 'vehicles': 'tomato',
                   'food': 'forestgreen', 'nature': 'darkorange'}

# Category prototypes — well separated
cat_centers = {
    'animals':  rng.standard_normal(D),
    'vehicles': rng.standard_normal(D),
    'food':     rng.standard_normal(D),
    'nature':   rng.standard_normal(D),
}
# Orthogonalise (approximate)
keys = list(cat_centers.keys())
for i, k1 in enumerate(keys):
    for k2 in keys[:i]:
        proj = np.dot(cat_centers[k1], cat_centers[k2]) * cat_centers[k2]
        cat_centers[k1] -= proj
    cat_centers[k1] /= np.linalg.norm(cat_centers[k1])
# Scale to radius 3
for k in cat_centers: cat_centers[k] *= 3

# Build per-concept embeddings
all_labels, all_cats, all_embs = [], [], []

for cat, words in concepts.items():
    for w in words:
        v = cat_centers[cat] + rng.standard_normal(D) * 0.6
        v /= np.linalg.norm(v)
        # text embedding
        all_labels.append(w)
        all_cats.append(cat)
        all_embs.append(v)
        # paired image embedding (slight noise)
        img_v = v + rng.standard_normal(D) * 0.15
        img_v /= np.linalg.norm(img_v)
        all_labels.append(f'[IMG] {w}')
        all_cats.append(cat)
        all_embs.append(img_v)

all_embs = np.array(all_embs)

# ── PCA to 2D ────────────────────────────────────────────────────────────
pca = skPCA(n_components=2)
emb_2d = pca.fit_transform(all_embs)

# ── Cross-modal retrieval: image of 'dog' → top-5 text matches ───────────
query_label = '[IMG] dog'
query_idx   = all_labels.index(query_label)
query_emb   = all_embs[query_idx]
sims        = all_embs @ query_emb                         # cosine (L2-normed)
# Exclude itself and other [IMG] items
text_mask   = np.array([not l.startswith('[IMG]') for l in all_labels])
sims_text   = sims.copy(); sims_text[~text_mask] = -999
top5_idx    = np.argsort(sims_text)[::-1][:5]

# ── Embedding arithmetic: king - man + woman ≈ queen ─────────────────────
# Simulate: mountain - 'high' + 'low' → valley?
def get_emb(label):
    return all_embs[all_labels.index(label)]

# ocean - fish + eagle ≈ sky?
v_ocean  = get_emb('ocean')
v_fish   = get_emb('[IMG] fish')
v_bird   = get_emb('bird')
v_arith  = v_ocean - v_fish + v_bird
v_arith /= np.linalg.norm(v_arith)
sims_arith = all_embs @ v_arith
top5_arith = np.argsort(sims_arith)[::-1][:5]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))

# Embedding space scatter
for cat, color in category_colors.items():
    mask_txt = np.array([l not in [f'[IMG] {w}' for w in concepts[cat]]
                          and l in concepts[cat] and all_cats[i] == cat
                          for i, l in enumerate(all_labels)])
    mask_img = np.array([l.startswith('[IMG]') and all_cats[i] == cat
                          for i, l in enumerate(all_labels)])
    axes[0].scatter(emb_2d[mask_txt, 0], emb_2d[mask_txt, 1],
                    c=color, s=60, alpha=0.9, label=cat, marker='o', zorder=4)
    axes[0].scatter(emb_2d[mask_img, 0], emb_2d[mask_img, 1],
                    c=color, s=60, alpha=0.5, marker='^', zorder=3)
    # Draw lines between matched pairs
    for w in concepts[cat]:
        if f'[IMG] {w}' in all_labels and w in all_labels:
            ti = all_labels.index(w)
            ii = all_labels.index(f'[IMG] {w}')
            axes[0].plot([emb_2d[ti,0], emb_2d[ii,0]],
                          [emb_2d[ti,1], emb_2d[ii,1]],
                          c=color, lw=0.7, alpha=0.4)
axes[0].set_title('Shared Embedding Space (PCA 2D)\n○=Text  △=Image  lines=paired',
                   fontweight='bold')
axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)

# Retrieval results
labels_r5  = [all_labels[i] for i in top5_idx]
sims_r5    = [sims_text[i]  for i in top5_idx]
colors_r5  = [category_colors[all_cats[i]] for i in top5_idx]
bars = axes[1].barh(range(5), sims_r5[::-1], color=colors_r5[::-1])
axes[1].set_yticks(range(5))
axes[1].set_yticklabels(labels_r5[::-1], fontsize=10)
axes[1].set_title(f'Cross-Modal Retrieval\nImage query: "{query_label}"',
                   fontweight='bold')
axes[1].set_xlabel('Cosine Similarity')
axes[1].axvline(0, color='k', lw=0.8); axes[1].grid(axis='x', alpha=.3)

# Arithmetic results
labels_a5 = [all_labels[i] for i in top5_arith]
sims_a5   = [sims_arith[i]  for i in top5_arith]
colors_a5 = [category_colors[all_cats[i]] for i in top5_arith]
axes[2].barh(range(5), sims_a5[::-1], color=colors_a5[::-1])
axes[2].set_yticks(range(5))
axes[2].set_yticklabels(labels_a5[::-1], fontsize=10)
axes[2].set_title('Embedding Arithmetic\nocean − fish + bird ≈ ?',
                   fontweight='bold')
axes[2].set_xlabel('Cosine Similarity')
axes[2].grid(axis='x', alpha=.3)

plt.suptitle('Multimodal Embedding Space: Geometry & Retrieval',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('multimodal_embedding.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Top-5 text matches for image query '{query_label}':")
for rank, (lbl, sim) in enumerate(zip(labels_r5, sims_r5), 1):
    print(f"  {rank}. {lbl:15s}  sim={sim:.4f}")
print(f"\\nEmbedding arithmetic (ocean − fish + bird):")
for rank, (lbl, sim) in enumerate(zip(labels_a5, sims_a5), 1):
    print(f"  {rank}. {lbl:15s}  sim={sim:.4f}")
""",
    },

    # ── 7 ────────────────────────────────────────────────────────────────────
    "7 · Q-Former / Perceiver Resampler (Cross-Attention Bridge)": {
        "description": (
            "Implements the Q-Former bridge module used in BLIP-2 and Flamingo. "
            "Shows how a small set of learnable query tokens can compress N visual "
            "patch tokens into a fixed-size representation via cross-attention."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)

# ── Dimensions ────────────────────────────────────────────────────────────
N_patches = 196    # ViT-L/14 image tokens
N_queries  = 32    # Q-Former learnable queries (BLIP-2 uses 32)
D_vis      = 128   # visual feature dim
D_query    = 64    # query dim (smaller, as in practice)
n_heads    = 4
d_head     = D_query // n_heads

# ── Synthetic visual features (196 patch embeddings) ─────────────────────
# Simulate: most patches are background (low variance), a few are salient
vis_feats = rng.standard_normal((N_patches, D_vis)) * 0.3   # background
# Salient region: patches 30–60
vis_feats[30:60] += rng.standard_normal((30, D_vis)) * 2.0
feature_norms = np.linalg.norm(vis_feats, axis=1)           # save BEFORE normalising — salient norms ~22×, background ~3×
vis_feats /= np.linalg.norm(vis_feats, axis=1, keepdims=True)

# ── Learnable query tokens ────────────────────────────────────────────────
query_tokens = rng.standard_normal((N_queries, D_query)) * 0.1

# ── Cross-attention: queries attend to visual features ───────────────────
# Project visual features to key/value space
Wk = rng.standard_normal((D_vis, D_query)) * 0.1
Wv = rng.standard_normal((D_vis, D_query)) * 0.1
Wq = rng.standard_normal((D_query, D_query)) * 0.1

K = vis_feats @ Wk       # (N_patches, D_query)
V = vis_feats @ Wv       # (N_patches, D_query)
Q = query_tokens @ Wq    # (N_queries, D_query)

# Multi-head cross-attention
head_atts = []
out_per_head = []
for h in range(n_heads):
    start, end = h * d_head, (h+1) * d_head
    A_h = softmax(Q[:, start:end] @ K[:, start:end].T / np.sqrt(d_head))
    O_h = A_h @ V[:, start:end]
    head_atts.append(A_h)
    out_per_head.append(O_h)

A_avg = np.mean(head_atts, axis=0)         # (N_queries, N_patches)
output = np.concatenate(out_per_head, axis=-1)  # (N_queries, D_query)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. Input visual features (salient region highlighted)
# feature_norms already computed before normalisation — correctly shows background ~3, salient ~22
axes[0, 0].bar(range(N_patches), feature_norms, color='steelblue', alpha=0.7)
axes[0, 0].axvspan(29.5, 59.5, alpha=0.25, color='orange', label='Salient region')
axes[0, 0].set_title(f'Input: {N_patches} Visual Patch Features\n(L2 norms)',
                      fontweight='bold')
axes[0, 0].set_xlabel('Patch index'); axes[0, 0].set_ylabel('‖feature‖')
axes[0, 0].legend(fontsize=9); axes[0, 0].grid(alpha=.3)

# 2. Cross-attention matrix (queries × patches)
im = axes[0, 1].imshow(A_avg, cmap='Blues', aspect='auto')
axes[0, 1].set_title(f'Cross-Attention: {N_queries} Queries × {N_patches} Patches\n'
                      f'(avg over {n_heads} heads)', fontweight='bold')
axes[0, 1].set_xlabel('Patch index (visual tokens)')
axes[0, 1].set_ylabel('Query index')
plt.colorbar(im, ax=axes[0, 1])
axes[0, 1].axvline(29.5, color='orange', lw=1.5, ls='--')
axes[0, 1].axvline(59.5, color='orange', lw=1.5, ls='--', label='Salient region')
axes[0, 1].legend(fontsize=8)

# 3. Which patches does each query attend to most?
top_patch = np.argmax(A_avg, axis=1)
axes[0, 2].scatter(top_patch, range(N_queries), c=top_patch,
                    cmap='RdYlGn', s=60, zorder=4)
axes[0, 2].axvspan(29.5, 59.5, alpha=0.2, color='orange', label='Salient region')
axes[0, 2].set_title('Top Attended Patch per Query', fontweight='bold')
axes[0, 2].set_xlabel('Patch index (argmax attention)')
axes[0, 2].set_ylabel('Query index'); axes[0, 2].legend(fontsize=9)
axes[0, 2].grid(alpha=.3)

# 4. Per-head attention for query 0
for h in range(n_heads):
    axes[1, 0].plot(head_atts[h][0], label=f'Head {h+1}', alpha=0.8, lw=1.5)
axes[1, 0].axvspan(29.5, 59.5, alpha=0.15, color='orange')
axes[1, 0].set_title('Query 0: Per-Head Attention over Patches', fontweight='bold')
axes[1, 0].set_xlabel('Patch index'); axes[1, 0].set_ylabel('Attention weight')
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=.3)

# 5. Output query features
out_norms = np.linalg.norm(output, axis=1)
axes[1, 1].bar(range(N_queries), out_norms, color='forestgreen', alpha=0.8)
axes[1, 1].set_title(f'Output: {N_queries} Compressed Query Representations\n'
                      f'(L2 norms after cross-attention)', fontweight='bold')
axes[1, 1].set_xlabel('Query index'); axes[1, 1].set_ylabel('‖output‖')
axes[1, 1].grid(alpha=.3)

# 6. Compression summary
compression = N_patches / N_queries
axes[1, 2].axis('off')
summary = (
    f"Q-Former Compression Summary\n"
    f"{'─'*32}\n"
    f"Input visual tokens :  {N_patches}\n"
    f"Learnable queries   :  {N_queries}\n"
    f"Compression ratio   :  {compression:.1f}×\n"
    f"Visual feature dim  :  {D_vis}\n"
    f"Query dim           :  {D_query}\n"
    f"Attention heads     :  {n_heads}\n\n"
    f"Tokens fed to LLM   :  {N_queries} (fixed)\n"
    f"Tokens from ViT     :  {N_patches} (variable)\n\n"
    f"Key insight: The {N_queries} query tokens learn\n"
    f"to extract task-relevant information\n"
    f"from all {N_patches} visual patches via\n"
    f"cross-attention — like a learned\n"
    f"'visual summariser' for the LLM."
)
axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                fontsize=10.5, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.9))

plt.suptitle('Q-Former / Perceiver Resampler: Compressing Visual Tokens',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('qformer.png', dpi=130, bbox_inches='tight')
plt.show()

frac_salient = np.mean((top_patch >= 30) & (top_patch < 60))
print(f"Queries attending to salient region: {frac_salient*100:.1f}%")
print(f"Compression: {N_patches} visual tokens → {N_queries} query tokens ({compression:.0f}×)")
""",
    },

    # ── 8 ────────────────────────────────────────────────────────────────────
    "8 · Log-Mel Spectrogram (Audio Representation for Whisper)": {
        "description": (
            "Generates a synthetic audio signal (speech-like formants) and "
            "computes a log-mel spectrogram — the input representation used by "
            "Whisper and most audio transformers."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

# ── Synthetic audio signal (sum of formant-like sinusoids) ───────────────
sr       = 16000   # 16 kHz
duration = 2.0     # seconds
t        = np.linspace(0, duration, int(sr * duration))

# "Vowel-like" signal: fundamental + harmonics with amplitude envelope
f0       = 180     # fundamental frequency (Hz)
envelope = np.exp(-t * 0.8) * (1 - np.exp(-t * 15))   # attack + decay
signal   = np.zeros_like(t)

# First vowel (0–0.8s): /a/ formants ~800, 1200, 2600 Hz
mask1 = t < 0.8
for f_k in [f0, 2*f0, 3*f0, 4*f0, 800, 1200, 2600]:
    amp  = 1.0 / (f_k / f0)**0.6
    signal[mask1] += amp * np.sin(2 * np.pi * f_k * t[mask1])

# Second vowel (0.9–1.7s): /i/ formants ~280, 2800, 3100 Hz
mask2 = (t > 0.9) & (t < 1.7)
for f_k in [f0, 2*f0, 3*f0, 280, 2800, 3100]:
    amp  = 1.0 / (f_k / f0)**0.5
    signal[mask2] += amp * np.sin(2 * np.pi * f_k * t[mask2])

signal *= envelope
signal += rng.standard_normal(len(t)) * 0.03   # background noise

# ── STFT → Power Spectrogram ─────────────────────────────────────────────
n_fft     = 512
hop_len   = 160    # 10ms at 16kHz
win_len   = 400    # 25ms window
window    = np.hanning(win_len)
n_frames  = 1 + (len(signal) - win_len) // hop_len

spec = np.zeros((n_fft // 2 + 1, n_frames))
for i in range(n_frames):
    start = i * hop_len
    frame = signal[start : start + win_len]
    if len(frame) < win_len:
        frame = np.pad(frame, (0, win_len - len(frame)))
    frame_w = frame * window
    fft_out = np.fft.rfft(frame_w, n=n_fft)
    spec[:, i] = np.abs(fft_out)**2

# ── Mel filterbank ────────────────────────────────────────────────────────
n_mels   = 80
f_min, f_max = 0, sr / 2
def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
def mel_to_hz(m): return 700 * (10**(m / 2595) - 1)

mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
hz_points  = mel_to_hz(mel_points)
bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

fbank = np.zeros((n_mels, n_fft // 2 + 1))
for m in range(1, n_mels + 1):
    b1, b2, b3 = bin_points[m-1], bin_points[m], bin_points[m+1]
    for k in range(b1, b2):
        fbank[m-1, k] = (k - b1) / max(b2 - b1, 1)
    for k in range(b2, min(b3, n_fft // 2 + 1)):
        fbank[m-1, k] = (b3 - k) / max(b3 - b2, 1)

mel_spec = fbank @ spec                             # (80, T)
log_mel  = np.log(mel_spec + 1e-9)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

# Waveform
axes[0].plot(t, signal, lw=0.6, color='royalblue')
axes[0].set_title('Waveform (synthetic vowel /a/ then /i/)', fontweight='bold')
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Amplitude')
axes[0].axvspan(0, 0.8,  alpha=0.1, color='green',  label='/a/')
axes[0].axvspan(0.9, 1.7, alpha=0.1, color='orange', label='/i/')
axes[0].legend(fontsize=9); axes[0].grid(alpha=.3)

# Power spectrogram
freqs  = np.linspace(0, sr/2, n_fft//2 + 1)
times  = np.arange(n_frames) * hop_len / sr
im1 = axes[1].pcolormesh(times, freqs/1000, 10*np.log10(spec + 1e-9),
                           cmap='magma', shading='gouraud', vmin=-50)
axes[1].set_title('Power Spectrogram (STFT)', fontweight='bold')
axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Frequency (kHz)')
axes[1].set_ylim(0, 4)
plt.colorbar(im1, ax=axes[1], label='dB')

# Log-mel spectrogram
mel_times = np.arange(n_frames) * hop_len / sr
im2 = axes[2].pcolormesh(mel_times, np.arange(n_mels), log_mel,
                           cmap='magma', shading='gouraud')
axes[2].set_title(f'Log-Mel Spectrogram ({n_mels} mel bins)  ← Whisper Input',
                   fontweight='bold')
axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('Mel bin')
plt.colorbar(im2, ax=axes[2], label='Log energy')

plt.suptitle('Audio Representation for Multimodal Models (Whisper-style)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('logmel_spectrogram.png', dpi=130, bbox_inches='tight')
plt.show()

print(f"Log-mel spectrogram shape : {log_mel.shape}  (n_mels × n_frames)")
print(f"Each frame                : {hop_len/sr*1000:.1f} ms hop, "
      f"{win_len/sr*1000:.1f} ms window")
print(f"Total input to model      : {n_mels} × {n_frames} = "
      f"{n_mels * n_frames} values (vs {len(signal)} raw samples)")
print(f"Compression ratio         : {len(signal) / (n_mels * n_frames):.1f}×")
""",
    },

    # ── 9 ────────────────────────────────────────────────────────────────────
    "9 · Hallucination Detection (CHAIR-style Metric)": {
        "description": (
            "Simulates VLM hallucination: the model generates captions with "
            "plausible but non-existent objects.  Computes the CHAIR_s metric "
            "and shows how hallucination correlates with confidence and context."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

rng = np.random.default_rng(11)

# ── Simulated VLM captions and ground truth objects ───────────────────────
# Each image has a set of true objects + a VLM-generated caption mentioning objects

dataset = [
    # (true_objects, predicted_objects_in_caption)
    (['dog', 'ball', 'grass'],            ['dog', 'ball', 'grass', 'tree']),     # 1 hallucination
    (['cat', 'sofa'],                      ['cat', 'sofa', 'book', 'lamp']),      # 2 hallucinations
    (['pizza', 'table', 'fork'],           ['pizza', 'table', 'fork']),           # 0
    (['car', 'road', 'building'],          ['car', 'road', 'building', 'sky']),   # 1
    (['person', 'bicycle', 'helmet'],      ['person', 'bicycle', 'dog', 'cat']), # 2
    (['apple', 'bowl', 'table'],           ['apple', 'bowl', 'banana', 'table']),# 1
    (['airplane', 'sky'],                  ['airplane', 'sky']),                  # 0
    (['elephant', 'grass', 'tree'],        ['elephant', 'grass', 'tree', 'river']),# 1
    (['book', 'desk', 'lamp'],             ['book', 'desk', 'lamp', 'cup', 'plant']),# 2
    (['fish', 'water'],                    ['fish', 'water', 'boat', 'child']),   # 2
    (['train', 'station', 'crowd'],        ['train', 'station', 'crowd']),        # 0
    (['bird', 'branch', 'leaves'],         ['bird', 'branch', 'leaves', 'nest']),# 1
    (['computer', 'keyboard', 'mouse'],    ['computer', 'keyboard', 'mouse', 'monitor', 'coffee']),# 2
    (['cake', 'candles', 'plate'],         ['cake', 'candles', 'plate', 'balloons', 'gift']),# 2
    (['mountain', 'snow', 'sky'],          ['mountain', 'snow', 'sky']),          # 0
]

# Compute CHAIR_s per image
def chair_s(true_objs, pred_objs):
    '''CHAIR_s = |hallucinated objects| / |total mentioned objects|'''
    hallucinated = [o for o in pred_objs if o not in true_objs]
    return len(hallucinated) / max(len(pred_objs), 1), hallucinated

chairs, n_hall_per_img, total_mentioned = [], [], []
all_hallucinated = []

for true_o, pred_o in dataset:
    ch, hall = chair_s(true_o, pred_o)
    chairs.append(ch)
    n_hall_per_img.append(len(hall))
    total_mentioned.append(len(pred_o))
    all_hallucinated.extend(hall)

# Hallucination frequency by object
hall_counter = Counter(all_hallucinated)

# ── Language prior analysis ───────────────────────────────────────────────
# Objects commonly mentioned together (co-occurrence priors)
cooccurrence = {
    'tree':   ['dog', 'ball', 'grass'],
    'book':   ['cat', 'sofa', 'lamp'],
    'sky':    ['car', 'road', 'building', 'airplane'],
    'banana': ['apple', 'bowl'],
    'dog':    ['person', 'bicycle'],
    'cat':    ['person', 'bicycle'],
    'river':  ['elephant', 'grass'],
    'plant':  ['book', 'desk'],
    'cup':    ['book', 'desk'],
    'boat':   ['fish', 'water'],
    'child':  ['fish', 'water'],
    'nest':   ['bird', 'branch'],
    'monitor':['computer', 'keyboard'],
    'coffee': ['computer', 'keyboard'],
    'balloons':['cake', 'candles'],
    'gift':   ['cake', 'candles'],
}

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. CHAIR_s per image
colors_ch = ['tomato' if c > 0 else 'forestgreen' for c in chairs]
axes[0, 0].bar(range(len(chairs)), chairs, color=colors_ch, alpha=0.85)
axes[0, 0].axhline(np.mean(chairs), color='k', ls='--', lw=1.5,
                    label=f'Mean CHAIR_s = {np.mean(chairs):.3f}')
axes[0, 0].set_title('CHAIR_s Score per Image', fontweight='bold')
axes[0, 0].set_xlabel('Image index'); axes[0, 0].set_ylabel('CHAIR_s (0=no hallucination)')
axes[0, 0].legend(fontsize=9); axes[0, 0].grid(axis='y', alpha=.3)
axes[0, 0].set_ylim(0, 0.75)

# 2. Hallucinated objects frequency
if hall_counter:
    objs_h, cnts_h = zip(*hall_counter.most_common(10))
    axes[0, 1].barh(objs_h, cnts_h, color='tomato', alpha=0.8)
    axes[0, 1].set_title('Most Frequently Hallucinated Objects', fontweight='bold')
    axes[0, 1].set_xlabel('Count across dataset'); axes[0, 1].grid(axis='x', alpha=.3)

# 3. Total mentioned vs hallucinated per image
x_pos = np.arange(len(dataset))
correct_counts = [t - h for t, h in zip(total_mentioned, n_hall_per_img)]  # non-hallucinated portion
axes[0, 2].bar(x_pos, correct_counts,  label='Correct',      color='steelblue', alpha=0.85)
axes[0, 2].bar(x_pos, n_hall_per_img,  label='Hallucinated', color='tomato',    alpha=0.85,
               bottom=correct_counts)  # stack ON TOP of correct — total bar height = total_mentioned
axes[0, 2].set_title('Objects Mentioned vs Hallucinated per Image', fontweight='bold')
axes[0, 2].set_xlabel('Image index'); axes[0, 2].set_ylabel('Object count')
axes[0, 2].legend(fontsize=9); axes[0, 2].grid(axis='y', alpha=.3)

# 4. Correlation: more objects in scene → more hallucinations?
axes[1, 0].scatter(total_mentioned, n_hall_per_img, c=chairs,
                    cmap='RdYlGn_r', s=80, zorder=5)
corr = np.corrcoef(total_mentioned, n_hall_per_img)[0, 1]
axes[1, 0].set_title(f'Scene Complexity vs Hallucinations\ncorr = {corr:.3f}',
                      fontweight='bold')
axes[1, 0].set_xlabel('Total objects mentioned'); axes[1, 0].set_ylabel('# Hallucinated')
axes[1, 0].grid(alpha=.3)

# 5. Co-occurrence heatmap (which objects trigger which hallucinations)
all_true_objs = list(set(o for (t, _) in dataset for o in t))[:10]
all_hall_objs = list(set(all_hallucinated))[:10]
co_matrix     = np.zeros((len(all_hall_objs), len(all_true_objs)))
for i, hall_obj in enumerate(all_hall_objs):
    for j, true_obj in enumerate(all_true_objs):
        if hall_obj in cooccurrence and true_obj in cooccurrence.get(hall_obj, []):
            co_matrix[i, j] = 1
im = axes[1, 1].imshow(co_matrix, cmap='Blues', aspect='auto')
axes[1, 1].set_xticks(range(len(all_true_objs)))
axes[1, 1].set_xticklabels(all_true_objs, rotation=45, ha='right', fontsize=8)
axes[1, 1].set_yticks(range(len(all_hall_objs)))
axes[1, 1].set_yticklabels(all_hall_objs, fontsize=8)
axes[1, 1].set_title('Language Prior Co-occurrence\n(hallucinated ← triggered by)',
                      fontweight='bold')
plt.colorbar(im, ax=axes[1, 1])

# 6. Summary text
axes[1, 2].axis('off')
n_clean = sum(1 for c in chairs if c == 0)
n_hall  = len(dataset) - n_clean
summary = (
    f"Hallucination Analysis Summary\n"
    f"{'─'*34}\n"
    f"Images analysed     : {len(dataset)}\n"
    f"Clean (CHAIR=0)     : {n_clean} ({n_clean/len(dataset)*100:.1f}%)\n"
    f"Hallucinated (>0)   : {n_hall} ({n_hall/len(dataset)*100:.1f}%)\n"
    f"Mean CHAIR_s        : {np.mean(chairs):.3f}\n"
    f"Max CHAIR_s         : {max(chairs):.3f}\n\n"
    f"Total objects mentioned : {sum(total_mentioned)}\n"
    f"Total hallucinated      : {sum(n_hall_per_img)}\n"
    f"Overall halluc. rate    : {sum(n_hall_per_img)/sum(total_mentioned)*100:.1f}%\n\n"
    f"Root cause:\n"
    f"  Language priors over-ride\n"
    f"  visual evidence — the model\n"
    f"  fills in 'expected' objects\n"
    f"  from training co-occurrences."
)
axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#fff4f4', alpha=0.9))

plt.suptitle('VLM Hallucination: CHAIR Metric & Analysis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hallucination.png', dpi=130, bbox_inches='tight')
plt.show()

print(f"Overall CHAIR_s: {np.mean(chairs):.4f}")
print(f"Hallucination rate: {sum(n_hall_per_img)/sum(total_mentioned)*100:.1f}%")
print(f"Most hallucinated objects: {dict(hall_counter.most_common(5))}")
""",
    },

    # ── 10 ───────────────────────────────────────────────────────────────────
    "10 · VLM Architecture Comparison Dashboard": {
        "description": (
            "Comprehensive comparison of CLIP, Flamingo, BLIP-2, LLaVA and "
            "modern MLLMs across architecture choices, training strategies, "
            "capabilities and compute requirements — radar chart + table."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

# ── Scores (0–10) for each model on each axis ─────────────────────────────
categories = [
    'Zero-shot\\nTransfer',
    'VQA\\nAccuracy',
    'Image\\nCaptioning',
    'OCR /\\nDocument',
    'Few-shot\\nLearning',
    'Training\\nEfficiency',
    'Open-source',
    'Multiimage\\nReasoning',
]
models = {
    'CLIP':      [9.5, 5.0, 3.0, 3.5, 6.0, 8.5, 9.5, 1.0],
    'Flamingo':  [7.5, 8.0, 7.5, 4.0, 9.5, 4.5, 3.0, 7.0],
    'BLIP-2':    [8.0, 8.5, 8.5, 5.0, 7.0, 7.0, 9.0, 4.0],
    'LLaVA-1.5': [7.0, 8.5, 8.0, 6.5, 6.5, 8.5, 9.5, 5.5],
    'GPT-4V':    [9.0, 9.5, 9.0, 9.5, 8.5, 5.0, 1.0, 9.5],
    'Gemini':    [9.0, 9.5, 9.0, 9.0, 8.5, 5.5, 2.0, 9.0],
}
colors = {
    'CLIP':       '#4e79a7',
    'Flamingo':   '#f28e2b',
    'BLIP-2':     '#59a14f',
    'LLaVA-1.5':  '#e15759',
    'GPT-4V':     '#76b7b2',
    'Gemini':     '#b07aa1',
}

fig = plt.figure(figsize=(18, 8))

# ── Radar chart ───────────────────────────────────────────────────────────
ax_r = fig.add_subplot(131, polar=True)
N_cat = len(categories)
angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]

for model, scores in models.items():
    vals = scores + scores[:1]
    ax_r.plot(angles, vals, 'o-', color=colors[model], lw=2.0, ms=4, label=model)
    ax_r.fill(angles, vals, color=colors[model], alpha=0.07)

ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels(categories, fontsize=7.5)
ax_r.set_ylim(0, 10); ax_r.set_yticks([2,4,6,8,10])
ax_r.set_yticklabels(['2','4','6','8','10'], fontsize=7)
ax_r.set_title('Capability Radar', fontsize=11, fontweight='bold', pad=20)
ax_r.legend(loc='upper right', bbox_to_anchor=(1.5, 1.15), fontsize=8.5)

# ── Grouped bar chart ─────────────────────────────────────────────────────
ax_b = fig.add_subplot(132)
x   = np.arange(N_cat)
bw  = 0.12
for i, (model, scores) in enumerate(models.items()):
    offset = (i - len(models)/2 + 0.5) * bw
    ax_b.bar(x + offset, scores, width=bw, color=colors[model], alpha=0.85, label=model)
ax_b.set_xticks(x)
ax_b.set_xticklabels([c.replace('\n',' ') for c in categories],
                      rotation=40, ha='right', fontsize=7.5)
ax_b.set_ylim(0, 11.5); ax_b.set_ylabel('Score (0–10)')
ax_b.set_title('Score per Capability', fontsize=11, fontweight='bold')
ax_b.legend(fontsize=7.5, ncol=2); ax_b.grid(axis='y', alpha=.3)

# ── Architecture summary table ────────────────────────────────────────────
ax_t = fig.add_subplot(133)
ax_t.axis('off')

arch_info = {
    'CLIP':       ('ViT + Trans.', 'Contrastive', '400M pairs', '—'),
    'Flamingo':   ('CLIP + Chinchilla', 'Cross-attn', '~2.3B pairs', 'Gated X-Attn'),
    'BLIP-2':     ('ViT-G + T5/OPT', 'ITC+ITM+ITG', '129M pairs', 'Q-Former (32q)'),
    'LLaVA-1.5':  ('ViT-L + Vicuna', 'Inst. Tuning', '~665K instruct', 'Linear proj'),
    'GPT-4V':     ('Unknown', 'RLHF+SFT', 'Proprietary', 'Unknown'),
    'Gemini':     ('Native MM', 'Native MM', 'Proprietary', 'Unified arch'),
}

col_labels = ['Model', 'Architecture', 'Training', 'Scale', 'Bridge']
rows = [[m] + list(v) for m, v in arch_info.items()]

tbl = ax_t.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center', loc='center',
    bbox=[0, 0.05, 1, 0.92]
)
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='white', fontweight='bold', fontsize=8.5)
    elif c == 0 and r > 0:
        model_name = rows[r-1][0]
        cell.set_facecolor(colors.get(model_name, '#eee'))
        cell.set_text_props(fontweight='bold', fontsize=8.5)
        cell.set_alpha(0.7)
    else:
        cell.set_facecolor('#f9f9f9' if r % 2 == 0 else 'white')
    cell.set_edgecolor('#cccccc')
ax_t.set_title('Architecture Summary', fontsize=11, fontweight='bold', pad=10)

plt.suptitle('Vision-Language Model Family Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('vlm_comparison.png', dpi=130, bbox_inches='tight')
plt.show()

print("\\nOverall capability score (mean across all axes):")
for model, scores in models.items():
    print(f"  {model:12s}: {np.mean(scores):.2f} / 10")
""",
    },

    # ── 11 ───────────────────────────────────────────────────────────────────
    "11 · ITC + ITM + ITG Combined Training (BLIP Framework)": {
        "description": (
            "Simulates the three complementary training objectives used in BLIP: "
            "Image-Text Contrastive (ITC), Image-Text Matching (ITM), and "
            "Image-Grounded Text Generation (ITG).  Shows how each shapes the embedding "
            "space differently and why combining them outperforms any single objective."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(55)

# ── Toy dataset: N (image, text) pairs with binary match labels ───────────
N   = 20   # positive pairs
D   = 32   # embedding dim
tau = 0.07

# Generate matched image/text embeddings (positive pairs)
true_vecs = rng.standard_normal((N, D))
true_vecs /= np.linalg.norm(true_vecs, axis=1, keepdims=True)

def make_embeddings(noise_img=1.0, noise_txt=1.0):
    img = true_vecs + rng.standard_normal((N, D)) * noise_img
    txt = true_vecs + rng.standard_normal((N, D)) * noise_txt
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    return img, txt

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)

def itc_loss(img, txt):
    '''InfoNCE contrastive loss'''\
    S    = img @ txt.T / tau
    labs = np.arange(N)
    def ce(logits):
        sm = softmax(logits)
        return -np.mean(np.log(sm[np.arange(N), labs] + 1e-9))
    return (ce(S) + ce(S.T)) / 2

def itm_loss(img, txt, hard_neg_ratio=0.3):
    '''Binary classification: match vs non-match'''\
    # Positive pairs
    pos_scores = np.sum(img * txt, axis=1)         # (N,)
    # Hard negatives: highest-similarity non-matching pairs
    S = img @ txt.T
    np.fill_diagonal(S, -999)
    hard_neg_idx = np.argmax(S, axis=1)             # hardest text neg per image
    neg_txt = txt[hard_neg_idx]
    neg_scores = np.sum(img * neg_txt, axis=1)

    # Binary cross-entropy
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(N), np.zeros(N)])
    logits = all_scores
    loss = -np.mean(all_labels * np.log(1/(1+np.exp(-logits)) + 1e-9) +
                    (1-all_labels) * np.log(1 - 1/(1+np.exp(-logits)) + 1e-9))
    acc  = np.mean((logits > 0) == all_labels.astype(bool))
    return float(loss), float(acc)

def itg_loss(img, txt):
    '''Simulated captioning loss: cross-entropy of generating correct text conditioned on image.'''
    # img@txt.T: each image should predict its matching text (diagonal = positive pairs)
    cos_sim   = img @ txt.T      # cross-modal similarity — diagonal is the generation target
    log_probs = np.log(softmax(cos_sim / 0.1) + 1e-9)
    return float(-np.mean(np.diag(log_probs)))

# ── Training simulation: 3 conditions ────────────────────────────────────
epochs = 80
conditions = {
    'ITC only':      {'use_itc': True,  'use_itm': False, 'use_itg': False},
    'ITC + ITM':     {'use_itc': True,  'use_itm': True,  'use_itg': False},
    'ITC + ITM + ITG (BLIP)': {'use_itc': True, 'use_itm': True, 'use_itg': True},
}
cond_colors = {'ITC only': 'steelblue', 'ITC + ITM': 'darkorange',
               'ITC + ITM + ITG (BLIP)': 'forestgreen'}
results = {k: {'itc': [], 'itm_acc': [], 'total': []} for k in conditions}

for cond_name, flags in conditions.items():
    # Simulate: embeddings start misaligned and improve over epochs
    img_emb, txt_emb = make_embeddings(noise_img=2.0, noise_txt=2.0)

    for ep in range(epochs):
        # Compute losses
        L_itc = itc_loss(img_emb, txt_emb) if flags['use_itc'] else 0.0
        L_itm, acc_itm = itm_loss(img_emb, txt_emb) if flags['use_itm'] else (0.0, 0.5)
        L_itg = itg_loss(img_emb, txt_emb) if flags['use_itg'] else 0.0  # pass txt_emb for cross-modal generation proxy

        n_objectives = sum(flags.values())
        total = (L_itc + L_itm + L_itg) / max(n_objectives, 1)

        results[cond_name]['itc'].append(float(itc_loss(img_emb, txt_emb)))
        results[cond_name]['itm_acc'].append(float(itm_loss(img_emb, txt_emb)[1]))
        results[cond_name]['total'].append(total)

        # Simulate gradient descent: move embeddings toward ground truth
        # More objectives → faster convergence
        lr = 0.04 * (1 + 0.5 * n_objectives) * (1 - ep / epochs)
        img_emb = (1-lr) * img_emb + lr * true_vecs
        txt_emb = (1-lr) * txt_emb + lr * true_vecs
        img_emb /= np.linalg.norm(img_emb, axis=1, keepdims=True)
        txt_emb /= np.linalg.norm(txt_emb, axis=1, keepdims=True)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ep_ax = np.arange(epochs)

# ITC loss convergence
for cond, col in cond_colors.items():
    axes[0].plot(ep_ax, results[cond]['itc'], color=col, lw=2, label=cond)
axes[0].set_title('ITC Loss vs Epoch\n(Image-Text Contrastive)', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('InfoNCE Loss')
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

# ITM accuracy
for cond, col in cond_colors.items():
    axes[1].plot(ep_ax, results[cond]['itm_acc'], color=col, lw=2, label=cond)
axes[1].axhline(0.5, color='k', ls='--', lw=1, label='Random (50%)')
axes[1].set_title('ITM Accuracy vs Epoch\n(Image-Text Matching)', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Binary Accuracy')
axes[1].set_ylim(0.4, 1.05)
axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)

# Final comparison
final_itc  = {c: results[c]['itc'][-1]    for c in conditions}
final_itm  = {c: results[c]['itm_acc'][-1] for c in conditions}
x = np.arange(len(conditions))
bw = 0.3
bars1 = axes[2].bar(x - bw/2, [final_itc[c]  for c in conditions],
                     width=bw, color='steelblue', alpha=0.85, label='Final ITC Loss ↓')
ax2b = axes[2].twinx()
bars2 = ax2b.bar(x + bw/2, [final_itm[c] for c in conditions],
                  width=bw, color='darkorange', alpha=0.85, label='Final ITM Acc ↑')
axes[2].set_xticks(x)
axes[2].set_xticklabels([c.replace(' (BLIP)', '') for c in conditions],
                         rotation=12, ha='right', fontsize=8.5)
axes[2].set_ylabel('ITC Loss (↓ better)', color='steelblue')
ax2b.set_ylabel('ITM Accuracy (↑ better)', color='darkorange')
axes[2].set_title('Final Performance by Training Objective\nCombining all 3 is best',
                   fontweight='bold')
lines1, labs1 = axes[2].get_legend_handles_labels()
lines2, labs2 = ax2b.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labs1 + labs2, fontsize=8.5, loc='upper left')
axes[2].grid(axis='y', alpha=.3)

plt.suptitle('BLIP Training Objectives: ITC + ITM + ITG are Complementary',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('blip_objectives.png', dpi=130, bbox_inches='tight')
plt.show()

print("Final results:")
for cond in conditions:
    print(f"  {cond:35s} ITC={results[cond]['itc'][-1]:.3f}  "
          f"ITM_acc={results[cond]['itm_acc'][-1]:.3f}")
""",
    },

    # ── 12 ───────────────────────────────────────────────────────────────────
    "12 · High-Resolution Tiling Strategy": {
        "description": (
            "Demonstrates the tiling strategy used in modern VLMs (LLaVA-1.5, InternVL) "
            "to handle high-resolution images.  Shows how a large image is split into "
            "crops + global thumbnail and how token count scales with resolution."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

rng = np.random.default_rng(7)

# ── Synthetic high-resolution image ──────────────────────────────────────
H, W = 672, 672
img  = np.zeros((H, W, 3))

# Background gradient
for i in range(H):
    img[i, :, 0] = i / H * 0.4 + 0.1
    img[i, :, 2] = (H - i) / H * 0.4 + 0.1
img[:, :, 1] = 0.15

# Add "objects" at various scales (important for high-res)
objects = [
    (100, 80, 60,  [0.9, 0.2, 0.2]),     # large red circle
    (300, 400, 35, [0.2, 0.8, 0.2]),     # green circle
    (520, 200, 20, [0.2, 0.2, 0.9]),     # small blue circle
    (150, 500, 15, [0.9, 0.9, 0.1]),     # tiny yellow
    (480, 520, 10, [0.8, 0.3, 0.8]),     # tiny purple
]
for (cy, cx, r, col) in objects:
    yy, xx = np.ogrid[:H, :W]
    mask = (yy-cy)**2 + (xx-cx)**2 < r**2
    img[mask] = col

# ── Tiling strategies ─────────────────────────────────────────────────────
def tile_image(img, grid_size, crop_size=224):
    '''Split image into grid_size×grid_size crops + global thumbnail.'''\
    H, W = img.shape[:2]
    crops = []
    for r in range(grid_size):
        for c in range(grid_size):
            y0 = r * (H // grid_size); y1 = y0 + (H // grid_size)
            x0 = c * (W // grid_size); x1 = x0 + (W // grid_size)
            crop = img[y0:y1, x0:x1]
            crops.append(crop)
    # Global thumbnail (downsample)
    step = H // crop_size
    thumbnail = img[::max(step,1), ::max(step,1)][:crop_size, :crop_size]
    return crops, thumbnail

strategies = [
    ('Naive (224×224)', 1, 196),
    ('2×2 crops + thumb', 2, 5 * 196),
    ('3×3 crops + thumb', 3, 10 * 196),
]

fig, axes = plt.subplots(2, 4, figsize=(17, 9))

# Original high-res
axes[0, 0].imshow(np.clip(img, 0, 1))
axes[0, 0].set_title(f'Original Image\n{H}×{W} pixels', fontweight='bold')
axes[0, 0].axis('off')

# Show tiling for each strategy
for idx, (name, grid, n_tokens) in enumerate(strategies):
    crops, thumb = tile_image(img, grid)

    ax = axes[0, idx + 1]
    ax.imshow(np.clip(img, 0, 1))

    # Draw tile boundaries
    tile_h = H // grid; tile_w = W // grid
    for r in range(grid + 1):
        ax.axhline(r * tile_h, color='yellow', lw=2.5, alpha=0.9)
    for c in range(grid + 1):
        ax.axvline(c * tile_w, color='yellow', lw=2.5, alpha=0.9)

    # Label tiles
    for r in range(grid):
        for c in range(grid):
            ax.text(c * tile_w + tile_w//2, r * tile_h + tile_h//2,
                    f'Crop\n{r*grid+c+1}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
    ax.set_title(f'{name}\n{n_tokens:,} visual tokens', fontweight='bold')
    ax.axis('off')

# Bottom row: show individual crops for 2×2 case
crops_2x2, thumb_2x2 = tile_image(img, 2)
for c_idx, (crop, ax_label) in enumerate(
        zip(crops_2x2 + [thumb_2x2], ['Crop 1\n(TL)', 'Crop 2\n(TR)',
                                        'Crop 3\n(BL)', 'Crop 4\n(BR)', 'Thumbnail\n(global)'])):
    ax = axes[1, c_idx] if c_idx < 4 else axes[1, 4] if 4 < len(axes[1]) else None
    if c_idx >= 4: break
    axes[1, c_idx].imshow(np.clip(crop, 0, 1))
    axes[1, c_idx].set_title(f'{ax_label}\n→ ViT → 196 tokens', fontweight='bold', fontsize=9)
    axes[1, c_idx].axis('off')

# ── Token count comparison ────────────────────────────────────────────────
grids  = [1, 2, 3, 4]
tokens = [196, 5*196, 10*196, 17*196]
labels = ['1×1\n(224px)', '2×2\n(448px)', '3×3\n(672px)', '4×4\n(896px)']
bars = axes[1, 3].bar(grids, tokens, color=['steelblue','darkorange','forestgreen','tomato'],
                       alpha=0.85, tick_label=labels)
axes[1, 3].set_title('Visual Token Count\nvs Tiling Grid', fontweight='bold')
axes[1, 3].set_xlabel('Grid size'); axes[1, 3].set_ylabel('Total visual tokens')
for bar, tok in zip(bars, tokens):
    axes[1, 3].text(bar.get_x() + bar.get_width()/2, tok + 100,
                    f'{tok:,}', ha='center', fontsize=9, fontweight='bold')
axes[1, 3].grid(axis='y', alpha=.3)

plt.suptitle('High-Resolution Tiling Strategy for VLMs\n'
             '(More crops → More tokens → Better detail, higher cost)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hires_tiling.png', dpi=130, bbox_inches='tight')
plt.show()

print("Resolution vs token count trade-off:")
for g, t, lbl in zip(grids, tokens, labels):
    cost_mult = t / 196
    print(f"  Grid {g}×{g} ({lbl.replace(chr(10),' ')}) : {t:5d} tokens  "
          f"({cost_mult:.0f}× base cost)")
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