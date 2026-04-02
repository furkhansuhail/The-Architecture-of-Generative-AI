"""
Positional Encodings — Sinusoidal, RoPE, and ALiBi
====================================================

Self-attention is permutation-invariant: shuffle the tokens and you get the
same result (up to shuffled outputs). Positional encodings inject the notion
of order into the model. The choice of encoding determines how well the model
generalises to sequence lengths it has never seen — one of the central
challenges in modern LLM design.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Positional Encodings — Sinusoidal, RoPE, ALiBi"
DISPLAY_NAME = "03 · Positional Encodings"
ICON         = "📍"
SUBTITLE     = "Sinusoidal, RoPE, and ALiBi"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Why Positional Information Is Necessary

Consider the sentence "The cat sat on the mat." Shuffled: "mat the on sat cat
The." The words are identical; the meaning is destroyed. A model that cannot
tell position apart cannot distinguish these.

Self-attention computes a weighted sum over value vectors, where the weights
depend on query-key dot products. Because every token's query and key are
computed from the same embedding table — with no index information baked in —
the attention score between token i and token j is identical regardless of
whether they are adjacent or 1000 positions apart.

**The positional encoding solves this by making the representation of each
token a function of both its content and its position.**

There are two broad approaches:

    1.  Absolute positional encodings — assign a fixed vector to each position
        and add it to the token embedding. The model learns what position 0,
        1, 2, … look like.

    2.  Relative positional encodings — instead of tagging each token with
        its absolute position, bias the attention score between tokens i and j
        based on their distance |i − j|. The model learns what "3 positions
        apart" means.

The modern LLM world has largely converged on relative encodings — specifically
RoPE — because they generalise much better to lengths longer than seen during
training.


### Generation 1 — Sinusoidal Positional Encoding (Vaswani et al., 2017)

The original Transformer paper proposed a **fixed** (non-learned) encoding:

    PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )

where:
    pos  = token position  (0, 1, 2, …, T-1)
    i    = dimension index (0, 1, 2, …, d/2 - 1)
    d    = model dimension

Intuition: think of each pair of dimensions (2i, 2i+1) as a "clock" running
at a different frequency. Low-index dimensions oscillate quickly (every few
tokens); high-index dimensions oscillate very slowly (over thousands of tokens).
Together they create a unique fingerprint for every position.


    **Diagram 1 — Sinusoidal PE as a Multi-Frequency Clock:**

    SINUSOIDAL POSITIONAL ENCODING
    ════════════════════════════════════════════════════════════════

    Dimension pair i=0  (fastest):   period ≈ 6 tokens
    │ sin │ 0.0  0.84  0.91  0.14 -0.76 -0.96 -0.28  0.66  0.99 ...
    │ cos │ 1.0  0.54 -0.42 -0.99 -0.65  0.28  0.96  0.75 -0.15 ...

    Dimension pair i=1  (slower):    period ≈ 63 tokens
    │ sin │ 0.0  0.10  0.20  0.30  0.39  0.48  0.56  0.64  0.72 ...
    │ cos │ 1.0  0.99  0.98  0.96  0.92  0.88  0.83  0.77  0.70 ...

    Dimension pair i=d/4 (slowest):  period ≈ 10,000 tokens
    │ sin │ 0.0  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.01 ...
    │ cos │ 1.0  1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00 ...

    Low dimensions:  fast oscillation → local position (nearby tokens differ)
    High dimensions: slow oscillation → global position (far tokens differ)

    Together: every (pos, dimension) cell has a unique value.


The **key mathematical property**: the dot product between PE(pos) and
PE(pos + k) depends only on k, not on the absolute value of pos. This means
the model can learn "how far apart" without memorising absolute positions —
a form of relative awareness baked into absolute encodings.

**Learned absolute encodings** (GPT-2, early BERT) replace the fixed sinusoids
with a learned embedding table of shape (max_len, d_model). This is flexible
but has a hard ceiling: positions beyond max_len have no embedding and the
model cannot extrapolate.

**Limitation of both:** Training on sequences up to length T, the model sees
positions 0 … T-1. At inference with T+100 tokens, the embeddings for positions
T … T+99 were never optimised. Performance degrades sharply — often to random
— beyond the training length. This is the **length extrapolation problem**.


### Generation 2 — Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) is the positional encoding used by LLaMA, Mistral,
Falcon, Qwen, and most modern LLMs. It is not added to the token embedding —
instead it **rotates the query and key vectors** before computing attention
scores.

**Core idea:** Encode position information as a rotation in the complex plane.
Rotating both Q and K by their respective positions means the dot product
Q_m · K_n naturally captures only the *relative* rotation between positions
m and n, i.e., only their distance m − n.

For a 2D subspace (dimensions 2i and 2i+1), the rotation for position m is:

    R(m, θᵢ) = [ cos(m·θᵢ)  -sin(m·θᵢ) ]
               [ sin(m·θᵢ)   cos(m·θᵢ) ]

    where θᵢ = 1 / 10000^(2i/d)    (same base frequencies as sinusoidal PE)

The full RoPE transformation applies this rotation independently to each pair
of dimensions in the query/key vector.

**Why dot products encode relative position:**

    Q_m · K_n = (R(m) · q) · (R(n) · k)
              = q · (R(m)ᵀ R(n)) · k
              = q · R(n - m) · k

The product of the two rotation matrices gives a rotation by (n − m) — the
*relative* displacement. The absolute positions m and n cancel out completely.
The attention score depends only on the distance between tokens.


    **Diagram 2 — RoPE: Rotation by Position:**

    ROPE: ROTATING QUERY AND KEY VECTORS
    ════════════════════════════════════════════════════════════════

    Each query/key vector is split into d/2 pairs of dimensions.
    Each pair is rotated by a position-dependent angle.

    Pair i at position m:

        [q₂ᵢ  ]       [cos(mθᵢ)  -sin(mθᵢ)] [q₂ᵢ  ]
        [q₂ᵢ₊₁]  ←   [sin(mθᵢ)   cos(mθᵢ)] [q₂ᵢ₊₁]

    Position 0:  θ = 0°  →  no rotation   (identity matrix)
    Position 1:  θ = θ₀  →  small rotation
    Position 10: θ = 10θ₀ →  larger rotation
    Position 100: θ = 100θ₀ → much larger rotation

    When we compute Q_m · K_n:
        The rotations compose:  R(m)ᵀ · R(n) = R(n - m)
        → Result depends only on (n − m), the RELATIVE distance. ✓
        → Absolute positions m and n are never directly compared. ✓


    **Diagram 3 — Why RoPE Generalises Better:**

    LENGTH GENERALISATION: LEARNED ABSOLUTE vs RoPE
    ════════════════════════════════════════════════════════════════

    Learned absolute PE (GPT-2 style):

    Training:    [pos0] [pos1] [pos2] ... [pos2047]
                    ↓      ↓      ↓              ↓
                 emb0   emb1   emb2   ...   emb2047   ← trained embeddings

    Inference at length 3000:
                    ... [pos2048] [pos2049] ...
                           ↓         ↓
                        ???       ???       ← NEVER SEEN → garbage

    RoPE:
    Training:    positions 0..2047 → rotation angles 0..2047·θ
    Inference at position 2500:
                 rotation angle = 2500·θ    ← just a larger rotation
                                              The math still works! ✓

    The sinusoidal functions are smooth — larger positions extrapolate
    naturally as "more rotation". Absolute embeddings are lookup tables —
    missing indices have no valid entry.


**RoPE and Long-Context Extensions**

A subtle issue: even though RoPE can mathematically handle larger positions,
the model was only trained on certain rotation magnitudes. Very large rotation
angles (long contexts) push the model into unfamiliar territory.

Several techniques extend RoPE to longer contexts:

    •   **Linear RoPE scaling** (simple): divide all position indices by a
        constant factor s > 1. This "compresses" positions so that the
        original training length maps to a longer actual context.
        If trained on 4k tokens with scale=4, positions 0..16k are mapped
        to the rotation range of 0..4k. Fast but imprecise.

    •   **NTK-aware RoPE scaling** (better): instead of scaling all
        frequencies uniformly, scale only the high-frequency components.
        Derived from Neural Tangent Kernel theory. Used in many community
        long-context models.

    •   **YaRN** (Peng et al., 2023): combines NTK-aware scaling with an
        attention temperature correction. Used to extend LLaMA-2 to 128k.

    •   **LongRoPE**: dynamically adjust per-dimension scale factors after
        profiling which dimensions are most sensitive to long contexts.


### Generation 3 — ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2022) takes a fundamentally different approach: instead
of encoding position into the tokens themselves, it **biases the attention
scores directly** with a penalty proportional to the distance between tokens.

    Attention score (i, j) = Q_i · K_j / √d_k  −  m · |i − j|

where m is a head-specific slope (a fixed hyperparameter, not learned).
Different heads use different slopes, giving the model multiple "views" of
distance sensitivity.

**The slope schedule:** For h attention heads, slopes are set to:

    m_k = 2^(−8k/h)    for k = 1, 2, …, h

So head 1 (steepest slope) penalises distance most aggressively (attends
mostly to very nearby tokens), while head h (shallowest slope) can still
attend globally.


    **Diagram 4 — ALiBi: Attention Score Modification:**

    ALIBI ATTENTION BIAS
    ════════════════════════════════════════════════════════════════

    Raw attention scores for a sequence of 5 tokens:

                 t0    t1    t2    t3    t4
    from t4:  [ 1.2   0.8   0.3  -0.1   2.1 ]  ← raw Q·K scores

    ALiBi penalty (slope m = 0.5, distances from t4):
    distances:   4     3     2     1     0
    penalty:  [-2.0  -1.5  -1.0  -0.5  -0.0 ]

    After bias:
              [-0.8  -0.7  -0.7  -0.6   2.1 ]

    After softmax: t4 dominates (nearby tokens get boosted; far tokens
                   are suppressed proportional to distance)

    Compare heads with different slopes:
    ┌──────────────────────────────────────────────────────────┐
    │  Steep slope (m = 1.0): only looks at last 5-10 tokens  │
    │  Medium slope (m = 0.25): looks at last 20-40 tokens    │
    │  Shallow slope (m = 0.0625): can look at 100+ tokens    │
    └──────────────────────────────────────────────────────────┘


**ALiBi's superpower: zero-shot length extrapolation.** Because the bias is a
simple linear penalty computed at inference time (no lookup table, no angle
that "runs out"), ALiBi models trained on 1024-token sequences can often
handle 2048-token or 4096-token sequences at inference with graceful
(not catastrophic) degradation. MPT-7B, BLOOM, and BloombergGPT all use ALiBi.

**ALiBi's limitation:** The linear penalty is a strong inductive bias that may
not suit all tasks. Very long-range dependencies (where the model genuinely
needs to attend to a document beginning from the end) are penalised. Empirically
RoPE + YaRN tends to outperform ALiBi for true long-context tasks.


### Comparison: Which Encoding Should You Use?

    Property                   Sinusoidal   Learned Abs.   RoPE       ALiBi
    ─────────────────────────────────────────────────────────────────────────
    Learnable parameters?      No           Yes (V×d)      No         No
    Relative position?         Partial      No             Yes ✓      Yes ✓
    Length extrapolation?       Poor         Very poor      Good       Very good
    KV-cache compatible?        Yes          Yes            Yes ✓      Yes ✓
    Requires retraining to      N/A          Yes            No (scale) No
      extend context?
    Used by                    Orig. paper  GPT-2          LLaMA,     MPT,
                                            BERT           Mistral    BLOOM
    ─────────────────────────────────────────────────────────────────────────


### RoPE Implementation Details (What the Code Actually Does)

In practice, applying RoPE to a query tensor q of shape (B, T, n_heads, d_head):

    1.  Precompute cos and sin tables for all positions 0..max_len:
            θᵢ = 1 / 10000^(2i / d_head)   for i = 0..d_head//2-1
            freqs[pos, i] = pos × θᵢ
            cos_table = cos(freqs)    shape: (max_len, d_head//2)
            sin_table = sin(freqs)    shape: (max_len, d_head//2)

    2.  For each position, split q into even and odd indices:
            q_even = q[..., 0::2]   (dimensions 0, 2, 4, …)
            q_odd  = q[..., 1::2]   (dimensions 1, 3, 5, …)

    3.  Apply the 2D rotation block-wise:
            q_rotated_even = q_even × cos − q_odd  × sin
            q_rotated_odd  = q_even × sin + q_odd  × cos

    4.  Interleave back:
            q_out[..., 0::2] = q_rotated_even
            q_out[..., 1::2] = q_rotated_odd

This is identical for keys. The cos/sin tables are precomputed once and cached;
during inference with KV-cache, only the new position's cos/sin values are needed.

The full complexity cost is O(T × d_head) — essentially free compared to the
O(T² × d_head) cost of the attention itself.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Positional Encoding Comparison

| Property                  | Sinusoidal (fixed) | Learned Absolute | RoPE            | ALiBi           |
|---------------------------|--------------------|------------------|-----------------|-----------------|
| Extra parameters          | 0                  | max_len × d      | 0               | 0               |
| Where applied             | Added to embed     | Added to embed   | Rotates Q and K | Biases attn scores|
| Encodes relative distance?| Partially          | No               | Yes             | Yes             |
| Context length extension  | Not needed (fixed) | Hard limit       | Scale/YaRN      | Zero-shot       |
| Inference length > train  | Degrades           | Fails            | Degrades slowly | Degrades slowly |
| Modern usage              | Legacy             | BERT, GPT-2      | LLaMA 1/2/3     | MPT, BLOOM      |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Sinusoidal PE — From Scratch": {
        "description": "Implement and visualise the original Transformer sinusoidal positional encoding with ASCII heatmap.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SINUSOIDAL POSITIONAL ENCODING — IMPLEMENTATION AND VISUALISATION
================================================================================

Implements the exact formula from "Attention Is All You Need" (Vaswani 2017):
    PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )

Shows:
    1. The encoding matrix (positions × dimensions)
    2. The multi-frequency "clock" property
    3. That dot products encode relative distance
================================================================================
"""

import math


def sinusoidal_pe(max_len: int, d_model: int) -> list[list[float]]:
    """
    Build the sinusoidal positional encoding matrix.
    Returns a list of shape (max_len, d_model).
    """
    pe = [[0.0] * d_model for _ in range(max_len)]
    for pos in range(max_len):
        for i in range(d_model // 2):
            angle = pos / (10_000 ** (2 * i / d_model))
            pe[pos][2 * i]     = math.sin(angle)
            pe[pos][2 * i + 1] = math.cos(angle)
    return pe


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def ascii_heatmap(matrix, title, rows=16, cols=32):
    """Print a simple ASCII heatmap using block characters."""
    shades = " ·░▒▓█"
    print(f"\n  {title}")
    print(f"  Rows = positions (0..{rows-1}), Cols = dimensions (0..{cols-1})")
    print(f"  Value range: dark=negative, light=positive")
    print()
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            v = matrix[r][c]           # value in [-1, 1]
            idx = int((v + 1) / 2 * (len(shades) - 1))
            idx = max(0, min(len(shades) - 1, idx))
            row_str += shades[idx] * 2
        print(f"  pos {r:3d} │{row_str}│")


if __name__ == "__main__":
    MAX_LEN = 64
    D_MODEL = 32

    pe = sinusoidal_pe(MAX_LEN, D_MODEL)

    # 1. Show encoding matrix as ASCII heatmap
    ascii_heatmap(pe, "Sinusoidal PE Matrix (pos × dim)", rows=20, cols=D_MODEL)

    print()
    print("=" * 62)
    print("  FREQUENCY ANALYSIS: Individual Dimension Pairs")
    print("=" * 62)

    # 2. Show values for specific dimension pairs
    for i in [0, 2, 8, 15]:
        period = 2 * math.pi * (10_000 ** (2 * i / D_MODEL))
        values_sin = [round(pe[pos][2 * i], 3)     for pos in range(12)]
        values_cos = [round(pe[pos][2 * i + 1], 3) for pos in range(12)]
        print(f"\n  Dim pair i={i:2d}  (period ≈ {period:.0f} tokens)")
        print(f"  sin: {values_sin}")
        print(f"  cos: {values_cos}")

    print()
    print("=" * 62)
    print("  RELATIVE DISTANCE PROPERTY")
    print("=" * 62)
    print()
    print("  If the encoding truly encodes relative position,")
    print("  PE(pos) · PE(pos+k) should depend only on k, not pos.")
    print()

    k = 5   # fixed offset
    print(f"  Dot products PE(pos) · PE(pos+{k}) for various pos:")
    for base in [0, 10, 20, 30, 40]:
        if base + k < MAX_LEN:
            d = dot(pe[base], pe[base + k])
            print(f"    pos={base:3d} to pos={base+k:3d}: {d:+.4f}")

    print(f"\n  All values are approximately equal → relative distance ✓")
    print()

    # 3. Unique fingerprint: are all position vectors distinct?
    print("=" * 62)
    print("  UNIQUENESS: Is every position vector distinct?")
    print("=" * 62)
    collisions = 0
    for i in range(MAX_LEN):
        for j in range(i + 1, MAX_LEN):
            sim = dot(pe[i], pe[j]) / math.sqrt(
                dot(pe[i], pe[i]) * dot(pe[j], pe[j]) + 1e-9
            )
            if abs(sim) > 0.999:
                collisions += 1
                print(f"  ⚠️  pos {i} and pos {j} are nearly identical! (sim={sim:.4f})")
    if collisions == 0:
        print(f"  ✓  All {MAX_LEN} position vectors are distinct (no collisions).")
''',
    },

    "RoPE — Full Implementation": {
        "description": "Complete RoPE implementation in PyTorch: precompute cos/sin tables, rotate Q and K, verify that attention scores encode relative position.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ROTARY POSITION EMBEDDINGS (RoPE) — FULL PYTORCH IMPLEMENTATION
================================================================================

Implements RoPE exactly as used in LLaMA:
    1. Precompute frequency tables (cos/sin for each position × dim pair)
    2. Apply rotation to Q and K tensors before attention
    3. Verify that Q_m · K_n depends only on (n - m)

Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           Su et al., 2021
================================================================================
"""

import math
import torch


# ── Core RoPE implementation ───────────────────────────────────────────────────

def precompute_rope_freqs(d_head: int, max_seq_len: int, base: float = 10_000.0):
    """
    Precompute cos and sin tables for all positions and dimension pairs.

    Returns:
        cos_table:  (max_seq_len, d_head//2)
        sin_table:  (max_seq_len, d_head//2)
    """
    # θᵢ = 1 / 10000^(2i / d_head)  for i = 0..d_head//2-1
    i      = torch.arange(0, d_head, 2, dtype=torch.float32)   # [0, 2, 4, ...]
    thetas = 1.0 / (base ** (i / d_head))                       # (d_head//2,)

    # positions × thetas → angle matrix
    positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
    freqs     = torch.outer(positions, thetas)                   # (T, d_head//2)

    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE rotation to a query or key tensor.

    Args:
        x:   (B, T, n_heads, d_head)  — query or key tensor
        cos: (T, d_head//2)            — precomputed cosines
        sin: (T, d_head//2)            — precomputed sines

    Returns:
        x_rotated: same shape as x
    """
    # Split into even and odd dimension indices
    x_even = x[..., 0::2]   # (B, T, n_heads, d_head//2)
    x_odd  = x[..., 1::2]   # (B, T, n_heads, d_head//2)

    # Broadcast cos/sin: (T, d_head//2) → (1, T, 1, d_head//2)
    cos = cos[None, :, None, :]   # unsqueeze batch and heads dims
    sin = sin[None, :, None, :]

    # 2D rotation for each dimension pair:
    #   [x_even_rot]   [cos  -sin] [x_even]
    #   [x_odd_rot ] = [sin   cos] [x_odd ]
    x_rot_even = x_even * cos - x_odd  * sin
    x_rot_odd  = x_even * sin + x_odd  * cos

    # Interleave even and odd back together
    x_rotated = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (..., d_head//2, 2)
    x_rotated = x_rotated.flatten(-2)                          # (..., d_head)

    return x_rotated


# ── RoPE-enhanced attention layer ─────────────────────────────────────────────

class RoPEAttention(torch.nn.Module):
    """Single-head attention with RoPE applied to Q and K."""

    def __init__(self, d_model: int, d_head: int, max_seq_len: int):
        super().__init__()
        self.d_head  = d_head
        self.Wq      = torch.nn.Linear(d_model, d_head, bias=False)
        self.Wk      = torch.nn.Linear(d_model, d_head, bias=False)
        self.Wv      = torch.nn.Linear(d_model, d_head, bias=False)

        cos, sin = precompute_rope_freqs(d_head, max_seq_len)
        self.register_buffer("cos_table", cos)
        self.register_buffer("sin_table", sin)

    def forward(self, x: torch.Tensor, mask=None):
        B, T, _ = x.shape

        q = self.Wq(x).unsqueeze(2)   # (B, T, 1, d_head) — fake n_heads dim
        k = self.Wk(x).unsqueeze(2)
        v = self.Wv(x)

        # Apply RoPE
        cos = self.cos_table[:T]
        sin = self.sin_table[:T]
        q   = apply_rope(q, cos, sin).squeeze(2)   # (B, T, d_head)
        k   = apply_rope(k, cos, sin).squeeze(2)

        # Attention
        scale  = math.sqrt(self.d_head)
        scores = (q @ k.transpose(-1, -2)) / scale  # (B, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        return weights @ v, scores


# ── Verification ──────────────────────────────────────────────────────────────

def verify_relative_encoding():
    """
    The key property of RoPE:
        Q_m · K_n should depend only on (n - m), not on absolute m or n.
    We verify this by computing attention logits at different (m, n) pairs
    with the same distance.
    """
    d_head    = 16
    max_len   = 64
    torch.manual_seed(0)

    # Fixed random q and k vectors (not position-dependent — pure content)
    q_content = torch.randn(d_head)
    k_content = torch.randn(d_head)

    cos_table, sin_table = precompute_rope_freqs(d_head, max_len)

    def rotated_dot(pos_q: int, pos_k: int) -> float:
        """Compute (R(pos_q) · q) · (R(pos_k) · k)."""
        q = q_content.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,d)
        k = k_content.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        q_rot = apply_rope(q, cos_table[pos_q:pos_q+1], sin_table[pos_q:pos_q+1])
        k_rot = apply_rope(k, cos_table[pos_k:pos_k+1], sin_table[pos_k:pos_k+1])

        return (q_rot.squeeze() @ k_rot.squeeze()).item()

    print("=" * 60)
    print("  ROPE: VERIFYING RELATIVE POSITION PROPERTY")
    print("=" * 60)
    print()
    print("  For a fixed (q_content, k_content), the dot product")
    print("  should depend only on distance = pos_k - pos_q.")
    print()

    for distance in [0, 1, 3, 7]:
        print(f"  Distance = {distance}:")
        scores = []
        for base in [0, 5, 10, 20, 30]:
            pos_q = base
            pos_k = base + distance
            if pos_k < max_len:
                s = rotated_dot(pos_q, pos_k)
                scores.append(s)
                print(f"    pos_q={pos_q:3d}, pos_k={pos_k:3d}  →  score = {s:+.5f}")
        variance = max(scores) - min(scores)
        print(f"    Range across base positions: {variance:.6f}  (≈0 → relative ✓)\n")


if __name__ == "__main__":
    # Basic shape test
    B, T, n_heads, d_head = 2, 12, 4, 32
    cos, sin = precompute_rope_freqs(d_head, max_seq_len=128)

    x = torch.randn(B, T, n_heads, d_head)
    x_rot = apply_rope(x, cos[:T], sin[:T])
    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {x_rot.shape}  (unchanged ✓)")
    print()

    # Verify relative position property
    verify_relative_encoding()

    # RoPE scaling demo (linear)
    print("=" * 60)
    print("  ROPE LINEAR SCALING FOR LONG CONTEXT")
    print("=" * 60)
    print()
    original_max  = 4096
    target_max    = 32768
    scale_factor  = target_max / original_max
    print(f"  Original training length: {original_max}")
    print(f"  Target inference length:  {target_max}")
    print(f"  Linear scale factor:      {scale_factor:.1f}×")
    print()
    print(f"  Without scaling: position {target_max} uses angle = {target_max}·θ")
    print(f"  With    scaling: position {target_max} maps to angle = {original_max}·θ")
    print(f"  → Model sees rotation magnitudes it was trained on ✓")
    print()
    print(f"  Implementation: divide all positions by {scale_factor:.1f} before lookup")
    print(f"    cos_scaled = cos_table[pos // {scale_factor:.1f}]")
''',
    },

    "ALiBi — Implementation and Comparison": {
        "description": "Implement ALiBi attention biases and compare attention weight distributions (locality vs global) against unbiased attention.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ALIBI (ATTENTION WITH LINEAR BIASES) — IMPLEMENTATION AND COMPARISON
================================================================================

ALiBi replaces positional encodings with a fixed linear penalty on the
attention scores based on token distance:

    score(i, j) = Q_i · K_j / sqrt(d) - m_head * |i - j|

where m_head is a head-specific slope: steeper = more local, shallower = global.

Reference: "Train Short, Test Long: Attention with Linear Biases Enables
            Input Length Extrapolation" (Press et al., 2022)
================================================================================
"""

import math
import torch
import torch.nn.functional as F


# ── ALiBi slope schedule ───────────────────────────────────────────────────────

def get_alibi_slopes(n_heads: int) -> list[float]:
    """
    Compute the ALiBi slope for each attention head.

    The schedule is: m_k = 2^(-8k/n_heads) for k = 1..n_heads
    This gives slopes from 2^(-8/n_heads) (shallowest) to 2^(-8) (steepest).

    Steeper slope → stronger distance penalty → more local attention head.
    Shallower slope → weaker penalty → more global attention head.
    """
    slopes = []
    for k in range(1, n_heads + 1):
        slope = 2 ** (-8 * k / n_heads)
        slopes.append(slope)
    return slopes


def build_alibi_bias(seq_len: int, n_heads: int) -> torch.Tensor:
    """
    Build the ALiBi bias matrix for all heads.

    Returns:
        bias: (n_heads, seq_len, seq_len)
              bias[h, i, j] = -slope_h * |i - j|  if j <= i (causal)
                             = -inf                 if j >  i (future, masked)
    """
    slopes = get_alibi_slopes(n_heads)  # (n_heads,)

    # Distance matrix: dist[i, j] = |i - j|
    positions = torch.arange(seq_len)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()  # (T, T)

    # ALiBi bias per head: -(slope × distance)
    # shape: (n_heads, T, T)
    slopes_t = torch.tensor(slopes).view(-1, 1, 1)
    alibi_bias = -slopes_t * dist.unsqueeze(0)   # (n_heads, T, T)

    # Apply causal mask (future → -inf)
    causal = torch.tril(torch.ones(seq_len, seq_len))
    alibi_bias = alibi_bias.masked_fill(causal.unsqueeze(0) == 0, float("-inf"))

    return alibi_bias


# ── ALiBi attention ───────────────────────────────────────────────────────────

def alibi_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    alibi_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention with ALiBi positional bias.

    Args:
        q, k, v:     (B, n_heads, T, d_head)
        alibi_bias:  (n_heads, T, T)

    Returns:
        output:  (B, n_heads, T, d_head)
        weights: (B, n_heads, T, T)
    """
    d_head = q.shape[-1]
    scale  = math.sqrt(d_head)

    # Raw attention scores
    scores = (q @ k.transpose(-1, -2)) / scale  # (B, n_heads, T, T)

    # Add ALiBi bias (broadcast over batch)
    scores = scores + alibi_bias.unsqueeze(0)

    weights = F.softmax(scores, dim=-1)
    output  = weights @ v
    return output, weights


# ── Visualisation helpers ─────────────────────────────────────────────────────

def print_attention_row(weights: list[float], label: str, width: int = 40):
    """Display one row of attention weights as a bar chart."""
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    bar_chars = "▁▂▃▄▅▆▇█"
    bars = ""
    for w in weights:
        idx = min(int(w * len(bar_chars) * 4), len(bar_chars) - 1)
        bars += bar_chars[idx]
    print(f"  {label:<20} │{bars}│  peak at pos {weights.index(max(weights))}")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    T       = 16   # sequence length
    n_heads = 4
    d_head  = 8
    B       = 1

    torch.manual_seed(7)

    # ── Slope schedule ────────────────────────────────────────────────────────
    slopes = get_alibi_slopes(n_heads)
    print("=" * 60)
    print("  ALIBI SLOPE SCHEDULE")
    print("=" * 60)
    print()
    print(f"  n_heads = {n_heads}")
    print()
    for h, s in enumerate(slopes):
        locality = "very local" if s > 0.3 else "local" if s > 0.1 else "semi-global" if s > 0.03 else "global"
        bar = "█" * max(1, int(s * 100))
        print(f"  Head {h}:  slope = {s:.4f}  {bar:<20}  ({locality})")
    print()
    print("  Steeper slope → stronger distance penalty → more local attention")
    print("  Shallower slope → weaker penalty → global / long-range attention")

    # ── Build bias matrix and run attention ───────────────────────────────────
    alibi_bias = build_alibi_bias(T, n_heads)

    q = torch.randn(B, n_heads, T, d_head)
    k = torch.randn(B, n_heads, T, d_head)
    v = torch.randn(B, n_heads, T, d_head)

    output, weights = alibi_attention(q, k, v, alibi_bias)

    print()
    print("=" * 60)
    print("  ATTENTION WEIGHT DISTRIBUTION — LAST TOKEN")
    print("  (which positions does the last token attend to?)")
    print("=" * 60)
    print()
    print(f"  Sequence length = {T}.  Showing weights for position {T-1}.")
    print()
    for h in range(n_heads):
        w = weights[0, h, T-1, :].tolist()
        print_attention_row(w, f"Head {h} (m={slopes[h]:.3f})")
    print()
    print("  Expected: steeply-sloped heads peak near last token (local);")
    print("            shallowly-sloped heads spread weight further back (global).")

    # ── Extrapolation demo ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  LENGTH EXTRAPOLATION DEMO")
    print("=" * 60)
    print()
    print("  ALiBi requires no positional parameters — the bias is computed")
    print("  on-the-fly from the distance matrix at any length.")
    print()
    for test_len in [16, 64, 256, 1024]:
        bias = build_alibi_bias(test_len, n_heads)
        print(f"  Length {test_len:5d}: bias shape = {tuple(bias.shape)}  ✓ (no lookup table)")

    print()
    print("  Compare with learned absolute PE:")
    print("  → Embedding table is fixed at training length; longer sequences")
    print("    have NO valid positional embedding → catastrophic failure.")
    print()
    print("  ALiBi just computes a larger distance matrix at inference time.")
    print("  The penalty formula works for any distance. Zero extra parameters.")
''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 600
    # try:
    #     from llm_training.visuals.positional_encodings import (
    #         PE_VISUAL_HTML,
    #         PE_VISUAL_HEIGHT,
    #     )
    #     visual_html   = PE_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = PE_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[03_positional_encodings_rope_alibi.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    COMPLEXITY,
        "operations":    OPERATIONS,
    }