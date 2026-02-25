"""Module: 12 · Generative Models"""

DISPLAY_NAME = "12 · Generative Models"
ICON         = "🎨"
SUBTITLE     = "VAEs, GANs and Diffusion Models — the math of content generation"

THEORY = """
## 12 · Generative Models

Generative models learn the underlying **probability distribution** of data so they can
synthesise new samples that look as if they came from the same source.  Unlike
discriminative models (which learn `P(y | x)`), generative models learn `P(x)` — or a
way to sample from it — where `x` is the raw data itself (images, audio, text, molecules).

---

## 1 · The Generative Modelling Problem

### 1.1 What Does "Generative" Mean?

Given a dataset `D = {x₁, x₂, …, xₙ}` drawn i.i.d. from some unknown distribution
`p_data(x)`, we want to learn a model distribution `p_θ(x)` such that:

- `p_θ(x) ≈ p_data(x)` — samples from our model look real.
- We can efficiently **sample** new `x ~ p_θ(x)`.
- Optionally, we can **evaluate likelihoods** `p_θ(x)` for new points.

### 1.2 The Curse of Dimensionality

A 256×256 RGB image lives in ℝ^(196,608).  Directly modelling this density is
intractable — the space is astronomically large yet real images occupy a vanishingly
small, highly structured manifold within it.

**Solution:** All modern generative models assume the data lies near a *low-dimensional
latent manifold* and learn a mapping between a simple latent space `z ∈ ℝᵈ` (d ≪ D)
and data space.

### 1.3 Taxonomy of Approaches

```
Generative Models
├── Explicit Density
│   ├── Tractable   → Autoregressive (PixelCNN, GPT), Normalising Flows
│   └── Approximate → Variational (VAE)
└── Implicit Density
    ├── GANs  (adversarial training, no explicit likelihood)
    └── Score-based / Diffusion Models
```

The three families we cover — **VAE**, **GAN**, **Diffusion** — each make different
trade-offs between sample quality, diversity, training stability, and the ability to
compute exact likelihoods.

---

## 2 · Variational Autoencoders (VAEs)

### 2.1 The Autoencoder Baseline

A plain autoencoder learns:
- **Encoder** `q_φ(z | x)` — compresses data to a bottleneck `z`.
- **Decoder** `p_θ(x | z)` — reconstructs from `z`.

Training minimises reconstruction loss `‖x − x̂‖²`.  The latent space has *no structure*
imposed, so you cannot sample new points from it meaningfully — interpolating between
latent codes produces garbage.

### 2.2 Variational Inference Background

We want to maximise the log-likelihood of our data:

```
log p_θ(x) = log ∫ p_θ(x | z) p(z) dz
```

This integral is **intractable** for deep networks.  Variational inference introduces an
approximate posterior `q_φ(z | x)` and derives the **Evidence Lower BOund (ELBO)**:

```
log p_θ(x) ≥ E_{q_φ(z|x)}[log p_θ(x | z)] − KL(q_φ(z | x) ‖ p(z))
           = ELBO(θ, φ; x)
```

The ELBO is a **lower bound** on the true log-likelihood (equality when `q = p`).
Maximising ELBO simultaneously:
1. Maximises the expected reconstruction quality.
2. Minimises the KL divergence between the approximate posterior and the prior.

### 2.3 The VAE Model

**Prior:**        `p(z) = N(0, I)`  — standard Gaussian
**Encoder:**      `q_φ(z | x) = N(μ_φ(x), diag(σ²_φ(x)))`  — diagonal Gaussian
**Decoder:**      `p_θ(x | z) = N(μ_θ(z), σ²I)`  — or Bernoulli for binary data

The encoder outputs two vectors `μ` and `log σ²` (same dimension as `z`).

**KL Term (closed form for Gaussians):**

```
KL(q_φ(z|x) ‖ N(0,I)) = −½ Σⱼ (1 + log σ²ⱼ − μⱼ² − σ²ⱼ)
```

This has a beautiful interpretation:
- `−log σ²ⱼ` → penalises shrinking variance (keeps spread)
- `μⱼ²`      → penalises mean moving away from 0 (keeps centred)
- `σ²ⱼ`      → penalises variance inflating past 1

### 2.4 The Reparameterisation Trick

The ELBO requires gradients to flow through the sampling step `z ~ q_φ(z|x)`.
Sampling is **not differentiable**.  The trick:

```
z = μ_φ(x) + σ_φ(x) ⊙ ε,    ε ~ N(0, I)
```

Now randomness lives in `ε` (no parameters), and gradients flow cleanly through `μ` and
`σ` via backprop.  This simple trick is why VAEs are trainable end-to-end.

### 2.5 Full ELBO Loss

```
L_VAE = E_{ε~N(0,I)}[ ‖x − Decoder(μ + σ⊙ε)‖² ]  +  β · KL(q_φ ‖ p)
```

The `β` hyperparameter (β-VAE, Higgins et al. 2017) controls the disentanglement–
reconstruction trade-off:
- `β = 1`  → standard VAE
- `β > 1`  → more disentangled latent dimensions, blurrier reconstructions
- `β < 1`  → sharper reconstructions, less structured latent space

### 2.6 Latent Space Properties

Because the KL term forces `q_φ(z|x)` toward `N(0,I)`:
- The entire latent space is **covered** — no holes.
- You can sample `z ~ N(0,I)` and decode to get coherent outputs.
- Nearby points in `z`-space decode to similar-looking outputs → smooth interpolation.

### 2.7 VAE Limitations

- **Blurry samples**: The pixel-wise reconstruction loss (MSE) is equivalent to a
  Gaussian decoder — it averages over modes, producing blurry images.
- **Posterior collapse**: Individual latent dimensions can be ignored by the decoder;
  the encoder collapses them to the prior.
- **Approximate posterior**: The diagonal Gaussian is a weak approximation to the true
  posterior for complex data.

---

## 3 · Generative Adversarial Networks (GANs)

### 3.1 The Adversarial Framework

Instead of maximising likelihood, GANs (Goodfellow et al. 2014) frame generation as a
**two-player zero-sum game**:

- **Generator G**: Maps noise `z ~ p(z)` → fake data `G(z)`.  Goal: fool D.
- **Discriminator D**: Maps data `x` → probability of being real.  Goal: detect fakes.

### 3.2 The Minimax Objective

```
min_G  max_D  V(G, D) = E_{x~p_data}[log D(x)]  +  E_{z~p(z)}[log(1 − D(G(z)))]
```

At the **Nash equilibrium**:
- `D(x) = 0.5` everywhere — discriminator cannot distinguish real from fake.
- `p_G = p_data` — generator perfectly replicates the data distribution.

### 3.3 Theoretical Analysis

**Optimal Discriminator** (for fixed G):

```
D*_G(x) = p_data(x) / (p_data(x) + p_G(x))
```

Substituting back, the generator's loss becomes:

```
C(G) = −log(4) + 2 · JSD(p_data ‖ p_G)
```

where **JSD** is the Jensen-Shannon Divergence.  The global minimum `C(G) = −log(4)` is
achieved iff `p_G = p_data`.

### 3.4 The Non-Saturating Trick

In practice, `log(1 − D(G(z)))` saturates early in training (D is too good).
The generator is instead trained to **maximise** `log D(G(z))`:

```
L_G = −E_z[log D(G(z))]     ← non-saturating (NS-GAN)
L_D = −E_x[log D(x)] − E_z[log(1 − D(G(z)))]
```

This provides stronger gradients early in training.

### 3.5 Wasserstein GAN (WGAN)

JS divergence is problematic when `p_G` and `p_data` have disjoint support (common early
in training) — it equals `log 2` regardless of how far apart the distributions are →
vanishing gradients.

WGAN (Arjovsky et al. 2017) uses the **Earth Mover's (Wasserstein-1) distance**:

```
W(p, q) = inf_{γ ∈ Π(p,q)} E_{(x,y)~γ}[‖x − y‖]
```

By the Kantorovich-Rubinstein duality:

```
W(p_data, p_G) = sup_{‖f‖_L ≤ 1} E_{x~p_data}[f(x)] − E_{z~p(z)}[f(G(z))]
```

where `f` is any **1-Lipschitz function**.  The discriminator (now called the "critic")
approximates this `f`.  The WGAN objective:

```
L_critic = −E_x[f(x)] + E_z[f(G(z))]
L_G      = −E_z[f(G(z))]
```

**Enforcing the Lipschitz constraint:**
- **Weight clipping** (original WGAN): clip weights to `[−c, c]` — crude.
- **Gradient Penalty** (WGAN-GP): add `λ E_{x̂}[(‖∇f(x̂)‖₂ − 1)²]` to critic loss.
  `x̂` is sampled uniformly along lines between real and fake samples.

### 3.6 Progressive Growing & StyleGAN

**Progressive GAN** (Karras et al. 2018): Start training at 4×4 resolution, progressively
add layers to grow to full resolution.  This stabilises training dramatically.

**StyleGAN** (Karras et al. 2019): Separates style from structure via:
- **Mapping network**: `z → w` (8-layer MLP maps noise to intermediate space W).
- **Adaptive Instance Normalisation (AdaIN)**: style vector `w` controls each layer's
  scale and shift separately.
- **Stochastic variation**: per-pixel noise added at each layer for fine details.

### 3.7 Training Instabilities & Failure Modes

**Mode collapse**: G maps all `z` to a single or few modes.  Entire diversity of
`p_data` is never captured.

**Oscillation**: D and G cycle without converging — neither reaches Nash equilibrium.

**Gradient vanishing**: When D is too strong, its gradients w.r.t. G's parameters
vanish — G stops learning.

**Remedies:**
- Unrolled GANs (backprop through D's update steps)
- Minibatch discrimination (D sees statistics across a batch)
- Feature matching (G matches D's intermediate feature statistics)
- Label smoothing (replace `1` labels with `0.9`)
- Spectral normalisation of D's weights

---

## 4 · Diffusion Models

### 4.1 Motivation

Diffusion models (Ho et al. 2020 — DDPM) achieve the best image quality of any
generative model as of 2023+, while being easier to train than GANs (no adversarial
instability).  The core idea: learn to **reverse a corruption process**.

### 4.2 The Forward Process (Noising)

Define a Markov chain that gradually adds Gaussian noise to data over `T` steps:

```
q(x_t | x_{t-1}) = N(x_t; √(1 − β_t) x_{t-1},  β_t I)
```

- `β_t ∈ (0, 1)` is a **noise schedule** (small, slowly increasing).
- After enough steps, `x_T ≈ N(0, I)` — pure noise.

**Key insight — closed form for arbitrary t:**

```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0,  (1 − ᾱ_t) I)
```

where `αₜ = 1 − βₜ` and `ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ`.

So we can sample any noisy version directly:

```
x_t = √ᾱ_t · x_0  +  √(1 − ᾱ_t) · ε,    ε ~ N(0, I)
```

This is the **reparameterisation** that makes training efficient.

### 4.3 The Reverse Process (Denoising)

The reverse process `p_θ(x_{t-1} | x_t)` is also Gaussian (for small `β_t`):

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t),  Σ_θ(x_t, t))
```

A neural network learns the reverse transition at each step.

### 4.4 Training Objective

The ELBO for diffusion simplifies remarkably.  After algebra, the loss reduces to:

```
L_simple = E_{t, x_0, ε} [ ‖ε − ε_θ(x_t, t)‖² ]
```

**The network `ε_θ` simply predicts the noise `ε` that was added!**

Training procedure:
1. Sample clean image `x_0 ~ q(x_0)`
2. Sample timestep `t ~ Uniform(1, T)`
3. Sample noise `ε ~ N(0, I)`
4. Compute `x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε`
5. Predict `ε_θ(x_t, t)` and minimise `‖ε − ε_θ‖²`

### 4.5 Sampling (DDPM)

To generate, start from pure noise and iteratively denoise:

```
x_T ~ N(0, I)
for t = T, T−1, …, 1:
    z ~ N(0, I)  if t > 1 else 0
    x_{t-1} = (1/√αₜ)(x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t)) + σ_t z
```

This requires **T forward passes** through the network (typically T = 1000).

### 4.6 DDIM — Faster Sampling

DDIM (Song et al. 2020) reformulates the reverse process as a **non-Markovian**
deterministic process:

```
x_{t−1} = √ᾱ_{t−1} · (x_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
         + √(1−ᾱ_{t−1}) · ε_θ(x_t, t)
```

This allows sampling with as few as **10–50 steps** with minimal quality loss.
The same trained DDPM model is reused — DDIM only changes the sampling procedure.

### 4.7 The U-Net Architecture

The denoising network `ε_θ(x_t, t)` is typically a **U-Net** with:
- **Encoder**: Residual blocks + downsampling (halving spatial dims).
- **Bottleneck**: Self-attention (captures global context).
- **Decoder**: Residual blocks + upsampling + skip connections from encoder.
- **Time conditioning**: Sinusoidal timestep embedding, injected via AdaIN or addition.

The skip connections are crucial — they preserve fine spatial details through the
bottleneck, matching the structure needed to denoise accurately.

### 4.8 Classifier-Free Guidance (CFG)

To generate conditioned on a label `y` (e.g., text prompt), train a single model that
sometimes receives `y` and sometimes a null token `∅` (drop 10–20% of conditioning):

At inference, extrapolate away from the unconditional prediction:

```
ε̃_θ(x_t, t, y) = (1 + w) · ε_θ(x_t, t, y)  −  w · ε_θ(x_t, t, ∅)
```

- `w = 0`  → standard conditional sampling
- `w > 0`  → stronger adherence to `y`, slightly less diversity
- `w ≈ 7.5` is typical for image generation

### 4.9 Latent Diffusion Models (LDMs / Stable Diffusion)

Running diffusion in pixel space at high resolution is expensive.  LDMs (Rombach et al.
2022) run diffusion in the **latent space of a pre-trained VAE**:

1. **Encode**: `z = Encoder(x)`  → compress 512×512×3 → 64×64×4
2. **Diffuse**: Run DDPM entirely on `z`.
3. **Decode**: `x̂ = Decoder(z_0)`  → recover pixel-space image.

The VAE handles perceptual compression; the diffusion model handles semantic generation.
This reduces compute by **~48×** while maintaining quality.

Text conditioning comes via a **CLIP text encoder** that produces token embeddings
injected into the U-Net via **cross-attention layers**.

### 4.10 Score-Based Perspective

Diffusion models are deeply connected to **score matching**.  The score of a
distribution is `∇_x log p(x)`.  Tweedie's formula shows:

```
E[x_0 | x_t] = (x_t + (1−ᾱ_t) · ∇_{x_t} log q(x_t)) / √ᾱ_t
```

The noise prediction `ε_θ` is proportional to the **negative score**:

```
ε_θ ≈ −√(1−ᾱ_t) · ∇_{x_t} log q(x_t)
```

This connects DDPM to continuous-time score-based generative models (Song & Ermon 2020)
and stochastic differential equations (SDEs).

---

## 5 · Comparing the Three Families

| Property              | VAE                | GAN                   | Diffusion                  |
|-----------------------|--------------------|-----------------------|------------------------|
| Training stability    | ✅ Stable          | ⚠️ Unstable           | ✅ Very stable        |
| Sample quality        | 🔶 Blurry          | ✅ Sharp              | ✅✅ Best             |
| Sample diversity      | ✅ Good            | ⚠️ Mode collapse      | ✅ Excellent          |
| Likelihood evaluation | ✅ (ELBO)          | ❌ No                 | ✅ (ELBO)             |
| Latent space          | ✅ Structured      | ⚠️ Entangled          | ❌ No explicit latent |
| Inference speed       | ✅ One pass        | ✅ One pass           | ❌ Iterative (slow)   |
| Conditioning ease     | 🔶 Moderate        | 🔶 Moderate           | ✅ CFG is elegant     |

---

## 6 · Key Takeaways

- Generative models learn `p(x)` to synthesise new data.  The central challenge is that
  this distribution is intractable for high-dimensional data.

- **VAEs** solve intractability with the ELBO and the reparameterisation trick.  They
  produce a structured, continuous latent space suitable for interpolation and
  disentanglement, at the cost of blurry samples.

- **GANs** bypass likelihood entirely via adversarial training.  The generator learns by
  fooling a discriminator.  This produces sharp images but training is notoriously
  unstable; WGAN and its variants greatly improve stability by using Wasserstein distance.

- **Diffusion models** learn to reverse a Gaussian noising process.  The noise-prediction
  objective is simple and stable.  With DDIM and CFG they achieve state-of-the-art
  quality and flexible conditioning, at the cost of slower sampling.

- Latent Diffusion (Stable Diffusion) moves diffusion to a VAE's latent space, making
  high-resolution generation tractable.

- All three approaches rest on the same principle: model a simple distribution (`N(0,I)`)
  and learn a **neural mapping** to the complex data distribution.
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────
OPERATIONS = {
    # ── 1 ──────────────────────────────────────────────────────────────────
    "1 · Forward Diffusion (Closed-Form Noising)": {
        "description": (
            "Demonstrates the closed-form forward process q(x_t | x_0). "
            "Given a clean signal x_0, directly compute any noisy version x_t "
            "using the cumulative noise schedule ᾱ_t — no iterative stepping needed."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

# ── Noise schedule ──────────────────────────────────────────────────────────
T = 1000
beta = np.linspace(1e-4, 0.02, T)          # linear schedule
alpha = 1.0 - beta
alpha_bar = np.cumprod(alpha)              # ᾱ_t = ∏ αᵢ

# ── Clean 1-D "image" (a simple cosine wave) ────────────────────────────────
np.random.seed(42)
N = 256
x0 = np.cos(np.linspace(0, 4 * np.pi, N))

# ── Sample x_t at various timesteps ─────────────────────────────────────────
timesteps = [0, 100, 250, 500, 750, 999]
noisy = {}
for t in timesteps:
    eps = np.random.randn(N)
    noisy[t] = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 5), sharex=True, sharey=True)
axes = axes.flatten()
for i, t in enumerate(timesteps):
    axes[i].plot(noisy[t], lw=1, color='royalblue')
    axes[i].set_title(f't = {t}   ᾱ_t = {alpha_bar[t]:.4f}')
    axes[i].set_ylim(-3.5, 3.5)
    axes[i].axhline(0, color='k', lw=0.4, ls='--')
plt.suptitle('Forward Diffusion: q(x_t | x_0)  —  gradual Gaussian corruption',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('forward_diffusion.png', dpi=130, bbox_inches='tight')
plt.show()
print("Signal-to-noise ratio at each t:")
for t in timesteps:
    snr = alpha_bar[t] / (1 - alpha_bar[t])
    print(f"  t={t:4d}  ᾱ_t={alpha_bar[t]:.4f}  SNR={snr:.4f}")
""",
    },

    # ── 2 ──────────────────────────────────────────────────────────────────
    "2 · Noise Schedules (Linear vs Cosine)": {
        "description": (
            "Compares the original DDPM linear β-schedule with the cosine schedule "
            "(Nichol & Dhariwal 2021). The cosine schedule preserves more signal at "
            "early timesteps and is empirically superior for images."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

T = 1000
t_vals = np.arange(T)

# ── Linear schedule ───────────────────────────────────────────────────────
beta_lin = np.linspace(1e-4, 0.02, T)
abar_lin = np.cumprod(1 - beta_lin)

# ── Cosine schedule ───────────────────────────────────────────────────────
s = 0.008
f_t = np.cos((((t_vals + 1) / T + s) / (1 + s)) * np.pi / 2) ** 2  # +1 so t=0 is the first noisy step, matching Nichol & Dhariwal 2021
f_0 = np.cos((s / (1 + s)) * np.pi / 2) ** 2
abar_cos = f_t / f_0
abar_cos = np.clip(abar_cos, 1e-9, 1.0)

# ── Signal & noise fractions ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(t_vals, abar_lin, label='Linear',  color='steelblue')
axes[0].plot(t_vals, abar_cos, label='Cosine',  color='darkorange')
axes[0].set_title('Cumulative ᾱ_t  (signal fraction)')
axes[0].set_xlabel('Timestep t'); axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(t_vals, 1 - abar_lin, color='steelblue', label='Linear')
axes[1].plot(t_vals, 1 - abar_cos, color='darkorange', label='Cosine')
axes[1].set_title('1 − ᾱ_t  (noise fraction)')
axes[1].set_xlabel('Timestep t'); axes[1].legend(); axes[1].grid(alpha=.3)

# SNR
snr_lin = abar_lin / (1 - abar_lin + 1e-9)
snr_cos = abar_cos / (1 - abar_cos + 1e-9)
axes[2].semilogy(t_vals, snr_lin, color='steelblue', label='Linear')
axes[2].semilogy(t_vals, snr_cos, color='darkorange', label='Cosine')
axes[2].set_title('SNR  (log scale)')
axes[2].set_xlabel('Timestep t'); axes[2].legend(); axes[2].grid(alpha=.3)

plt.suptitle('Noise Schedule Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('noise_schedules.png', dpi=130, bbox_inches='tight')
plt.show()
print("Cosine schedule keeps more signal early → better gradient flow at small t.")
""",
    },

    # ── 3 ──────────────────────────────────────────────────────────────────
    "3 · VAE — ELBO Decomposition": {
        "description": (
            "Builds a minimal VAE in NumPy and plots how the reconstruction loss "
            "and KL divergence (ELBO terms) evolve during training on toy 2-D Gaussian data."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
rng = np.random.default_rng(0)

# ── Toy dataset: mixture of two 2-D Gaussians ────────────────────────────
N = 800
X = np.vstack([
    rng.multivariate_normal([2, 2],  [[1, .5],[.5, 1]], N//2),
    rng.multivariate_normal([-2, -2], [[1,-.4],[-.4,1]], N//2),
])

# ── Minimal 1-layer VAE ─────────────────────────────────────────────────
D, Z = 2, 2        # data dim, latent dim
lr   = 5e-3
epochs = 400

# Parameters: encoder (D→Z for mu and logvar), decoder (Z→D)
W_enc_mu  = rng.standard_normal((D, Z)) * 0.1
b_enc_mu  = np.zeros(Z)
W_enc_lv  = rng.standard_normal((D, Z)) * 0.1
b_enc_lv  = np.zeros(Z)
W_dec     = rng.standard_normal((Z, D)) * 0.1
b_dec     = np.zeros(D)

rec_losses, kl_losses = [], []

for ep in range(epochs):
    # ── Forward ──────────────────────────────────────────────────────────
    mu   = X @ W_enc_mu + b_enc_mu          # (N, Z)
    logv = X @ W_enc_lv + b_enc_lv          # (N, Z)
    sig  = np.exp(0.5 * logv)

    eps  = rng.standard_normal(mu.shape)
    z    = mu + sig * eps                   # reparameterisation

    x_hat = z @ W_dec + b_dec              # decoder

    # ── Losses ────────────────────────────────────────────────────────────
    diff  = X - x_hat
    rec   = 0.5 * np.mean(np.sum(diff**2, axis=1))
    kl    = -0.5 * np.mean(np.sum(1 + logv - mu**2 - np.exp(logv), axis=1))
    loss  = rec + kl

    rec_losses.append(float(rec))
    kl_losses.append(float(kl))

    # ── Backward (manual gradients) ───────────────────────────────────────
    dxhat = -(diff) / N                                  # ∂rec/∂x̂
    dWdec = z.T @ dxhat;  dbdec = dxhat.mean(axis=0)
    dz    = dxhat @ W_dec.T

    dmu_rec  = dz
    dsig     = dz * eps
    dlogv    = dsig * 0.5 * sig

    dmu_kl   = mu / N
    dlogv_kl = (-0.5 * (1 - np.exp(logv))) / N
    dlogv   += dlogv_kl

    dWemu = X.T @ dmu_rec + X.T @ (dmu_kl);  dbemu = (dmu_rec + dmu_kl).mean(0)
    dWelv = X.T @ dlogv;                      dbelv = dlogv.mean(0)

    W_dec -= lr * dWdec;   b_dec -= lr * dbdec
    W_enc_mu -= lr * dWemu; b_enc_mu -= lr * dbemu
    W_enc_lv -= lr * dWelv; b_enc_lv -= lr * dbelv

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ep_ax = np.arange(epochs)

axes[0].plot(ep_ax, rec_losses, color='royalblue',  label='Reconstruction')
axes[0].plot(ep_ax, kl_losses,  color='tomato',     label='KL Divergence')
axes[0].plot(ep_ax, np.array(rec_losses)+np.array(kl_losses),
             color='forestgreen', label='Loss / −ELBO (total)', lw=2)
axes[0].set_title('Loss / −ELBO Terms vs Epoch'); axes[0].legend(); axes[0].grid(alpha=.3)
axes[0].set_xlabel('Epoch')

# Latent space scatter
mu_final  = X @ W_enc_mu + b_enc_mu
axes[1].scatter(mu_final[:N//2, 0], mu_final[:N//2, 1],
                c='royalblue', s=10, alpha=.5, label='Cluster 1')
axes[1].scatter(mu_final[N//2:, 0], mu_final[N//2:, 1],
                c='tomato',    s=10, alpha=.5, label='Cluster 2')
axes[1].set_title('Learned Latent Space μ'); axes[1].legend(); axes[1].grid(alpha=.3)

# Reconstructions vs originals
axes[2].scatter(X[:, 0],    X[:, 1],    c='grey',  s=8, alpha=.3, label='Original')
x_hat_f = (X @ W_enc_mu + b_enc_mu) @ W_dec + b_dec
axes[2].scatter(x_hat_f[:, 0], x_hat_f[:, 1], c='darkorange', s=8, alpha=.3, label='Reconstructed')
axes[2].set_title('Original vs Reconstructed'); axes[2].legend(); axes[2].grid(alpha=.3)

plt.suptitle('VAE on 2-D Gaussian Mixture  (NumPy)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('vae_elbo.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Final reconstruction loss : {rec_losses[-1]:.4f}")
print(f"Final KL divergence       : {kl_losses[-1]:.4f}")
print(f"Final Loss / −ELBO (lower = better): {rec_losses[-1]+kl_losses[-1]:.4f}")
""",
    },

    # ── 4 ──────────────────────────────────────────────────────────────────
    "4 · Reparameterisation Trick — Gradient Flow Demo": {
        "description": (
            "Visualises why the reparameterisation trick is necessary. "
            "Shows that gradients cannot flow through a naive 'sample' node, "
            "but can flow when we rewrite z = μ + σ·ε with ε ~ N(0,1) fixed."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# ── Objective: minimise E_z[z²] where z ~ N(μ, 1), should converge μ → 0 ──
lr = 0.1
n_steps = 80
n_samples = 512

# ── Method A: naive sampling (no reparameterisation) ─────────────────────
# Estimate gradient via REINFORCE (score-function estimator)
mu_naive = np.array([3.0])
loss_naive = []

for _ in range(n_steps):
    z = rng.normal(mu_naive, 1.0, n_samples)
    f_z = z ** 2
    grad = np.mean(f_z * (z - mu_naive))           # score-function estimator: E[f(z) * ∇log p(z|μ)]
    loss_naive.append(float(np.mean(f_z)))
    mu_naive -= lr * grad

# ── Method B: reparameterisation ─────────────────────────────────────────
mu_reparam = np.array([3.0])
loss_reparam = []

for _ in range(n_steps):
    eps = rng.standard_normal(n_samples)
    z   = mu_reparam + eps                         # z = μ + ε
    f_z = z ** 2
    grad = np.mean(2 * z)                          # ∂(z²)/∂μ = 2z, dz/dμ=1
    loss_reparam.append(float(np.mean(f_z)))
    mu_reparam -= lr * grad

# ── Variance of the gradient estimator ───────────────────────────────────
epsilons = rng.standard_normal(2000)
mu_test  = 3.0
z_test   = mu_test + epsilons

grad_sf   = (z_test**2) * (z_test - mu_test)       # score-function
grad_rt   = 2 * z_test                              # reparameterisation

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(loss_naive,   color='tomato',      lw=2, label='Score-Function (REINFORCE)')
axes[0].plot(loss_reparam, color='royalblue',   lw=2, label='Reparameterisation')
axes[0].set_title('Loss Convergence'); axes[0].set_xlabel('Step')
axes[0].set_ylabel('E[z²]'); axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].hist(grad_sf, bins=60, color='tomato',    alpha=.7, label=f'SF   var={np.var(grad_sf):.1f}')
axes[1].hist(grad_rt, bins=60, color='royalblue', alpha=.7, label=f'RT   var={np.var(grad_rt):.2f}')
axes[1].set_title('Gradient Estimator Variance')
axes[1].legend(); axes[1].grid(alpha=.3)
axes[1].set_xlabel('Gradient value')

steps = np.arange(n_steps)
axes[2].plot(steps, [3.0 - lr * sum(
    [np.mean(2 * (3.0 - lr * sum([np.mean(2 * (3.0 + rng.standard_normal(n_samples)))
     for _ in range(k)]) + rng.standard_normal(n_samples)))
     for _ in range(k)]) for k in range(n_steps)],  # approximation for illustration
     color='grey', lw=1, ls=':')
# Replace with tracked mu values for a cleaner convergence plot
mu_naive_track    = [3.0]
mu_reparam_track  = [3.0]
_mu_n = np.array([3.0]); _mu_r = np.array([3.0])
_rng2 = np.random.default_rng(99)
for _ in range(n_steps):
    _z = _rng2.normal(_mu_n, 1.0, n_samples)
    _mu_n -= lr * np.mean(_z**2 * (_z - _mu_n))
    mu_naive_track.append(float(_mu_n))
    _eps = _rng2.standard_normal(n_samples)
    _z2  = _mu_r + _eps
    _mu_r -= lr * np.mean(2 * _z2)
    mu_reparam_track.append(float(_mu_r))

axes[2].plot(mu_naive_track,   color='tomato',    lw=1.5, label='Score-Function  μ')
axes[2].plot(mu_reparam_track, color='royalblue', lw=1.5, label='Reparam  μ')
axes[2].axhline(0, color='k', ls='--', lw=.8, label='Target μ=0')
axes[2].set_title('μ Convergence to 0 Over Steps')
axes[2].legend(); axes[2].grid(alpha=.3); axes[2].set_xlabel('Step'); axes[2].set_ylabel('μ')

plt.suptitle('Reparameterisation Trick vs Score-Function Estimator',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reparam_trick.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Score-function gradient variance : {np.var(grad_sf):.2f}")
print(f"Reparameterisation gradient variance: {np.var(grad_rt):.4f}")
print(f"Variance reduction factor: {np.var(grad_sf)/np.var(grad_rt):.1f}×")
""",
    },

    # ── 5 ──────────────────────────────────────────────────────────────────
    "5 · GAN — Optimal Discriminator & JS Divergence": {
        "description": (
            "Demonstrates the theoretical GAN analysis. Given a generator distribution p_G "
            "and the true data distribution p_data, computes the optimal discriminator D* "
            "analytically and visualises how the generator's loss equals the JS divergence."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-6, 6, 1000)

# ── Three snapshots of training ───────────────────────────────────────────
configs = [
    ("Early training",  0.0, 2.0),    # (label, G_mean, G_std)
    ("Mid training",    1.0, 1.5),
    ("Near convergence",2.0, 1.0),
]

p_data  = norm.pdf(x, loc=2.0, scale=1.0)   # fixed true distribution

fig, axes = plt.subplots(2, 3, figsize=(14, 7))

for col, (title, gm, gs) in enumerate(configs):
    p_G     = norm.pdf(x, loc=gm, scale=gs)
    p_total = p_data + p_G

    D_star  = p_data / (p_total + 1e-10)     # D*(x) = p_data / (p_data + p_G)

    # JS divergence
    m = 0.5 * (p_data + p_G)
    # avoid log(0)
    kl_pm = np.where(p_data > 1e-12, p_data * np.log(p_data / (m + 1e-12)), 0)
    kl_qm = np.where(p_G    > 1e-12, p_G    * np.log(p_G    / (m + 1e-12)), 0)
    jsd = 0.5 * np.trapz(kl_pm + kl_qm, x)

    # ── Top row: distributions ───────────────────────────────────────────
    axes[0, col].plot(x, p_data, color='royalblue', lw=2, label='p_data')
    axes[0, col].plot(x, p_G,    color='tomato',    lw=2, label='p_G (generator)')
    axes[0, col].fill_between(x, 0, np.minimum(p_data, p_G), alpha=.2, color='purple',
                              label='Overlap')
    axes[0, col].set_title(f'{title}\nJSD = {jsd:.4f}')
    axes[0, col].legend(fontsize=8); axes[0, col].grid(alpha=.3)
    axes[0, col].set_ylim(0, 0.55)

    # ── Bottom row: optimal discriminator ────────────────────────────────
    axes[1, col].plot(x, D_star, color='forestgreen', lw=2.5)
    axes[1, col].axhline(0.5, color='k', ls='--', lw=1, label='D*=0.5 (Nash eq.)')
    axes[1, col].set_title(f'Optimal D*(x)  (Nash: D*=0.5 everywhere)')
    axes[1, col].set_ylim(0, 1); axes[1, col].legend(fontsize=8)
    axes[1, col].grid(alpha=.3)

plt.suptitle('GAN Analysis: Optimal Discriminator & JS Divergence',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('gan_analysis.png', dpi=130, bbox_inches='tight')
plt.show()
print("At Nash equilibrium, D*(x)=0.5 everywhere and JSD=0 (p_G = p_data).")
""",
    },

    # ── 6 ──────────────────────────────────────────────────────────────────
    "6 · Wasserstein Distance vs JS Divergence": {
        "description": (
            "Shows why JS divergence fails for disjoint distributions (returns log2 "
            "regardless of separation), while the Wasserstein-1 distance scales "
            "continuously — motivating the WGAN objective."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

separations = np.linspace(0, 6, 120)

# p_data = N(0, 0.3), p_G = N(sep, 0.3)
sigma = 0.3
js_vals, w1_vals = [], []

x = np.linspace(-3, 10, 5000)
dx = x[1] - x[0]

for sep in separations:
    p  = norm.pdf(x, loc=0,   scale=sigma)
    q  = norm.pdf(x, loc=sep, scale=sigma)

    # JS divergence
    m    = 0.5 * (p + q)
    kl_p = np.where(p > 1e-15, p * np.log(p / (m + 1e-15)), 0)
    kl_q = np.where(q > 1e-15, q * np.log(q / (m + 1e-15)), 0)
    jsd  = float(0.5 * np.sum((kl_p + kl_q) * dx))

    # Wasserstein-1 for 1-D = integral of |CDF_p - CDF_q|
    F_p = np.cumsum(p) * dx
    F_q = np.cumsum(q) * dx
    w1  = float(np.sum(np.abs(F_p - F_q)) * dx)

    js_vals.append(jsd)
    w1_vals.append(w1)

js_vals = np.array(js_vals)
w1_vals = np.array(w1_vals)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(separations, js_vals, color='tomato', lw=2.5)
axes[0].axhline(np.log(2), color='k', ls='--', lw=1, label='log 2  (max / plateau)')
axes[0].set_title('JS Divergence vs Separation')
axes[0].set_xlabel('Separation between means')
axes[0].set_ylabel('JSD'); axes[0].legend(); axes[0].grid(alpha=.3)
axes[0].annotate('Saturates to log2\n→ vanishing gradients!',
                 xy=(1.5, np.log(2)-.01), fontsize=9, color='tomato')

axes[1].plot(separations, w1_vals, color='royalblue', lw=2.5)
axes[1].set_title('Wasserstein-1 Distance vs Separation')
axes[1].set_xlabel('Separation between means')
axes[1].set_ylabel('W₁'); axes[1].grid(alpha=.3)
axes[1].annotate('Scales linearly\n→ stable gradients everywhere',
                 xy=(2, separations[-1]*0.35), fontsize=9, color='royalblue')

plt.suptitle('Why WGAN? JSD Fails for Disjoint Distributions',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('wgan_motivation.png', dpi=130, bbox_inches='tight')
plt.show()
print("JSD plateaus to log(2) ≈", round(np.log(2), 4), "for separated distributions.")
print("W1 increases linearly → meaningful gradient everywhere.")
""",
    },

    # ── 7 ──────────────────────────────────────────────────────────────────
    "7 · 1-D GAN Training Simulation": {
        "description": (
            "Trains a tiny GAN on 1-D data (target: N(2, 0.5)) using NumPy "
            "to illustrate the adversarial dynamics: the generator distribution "
            "chasing the real distribution step by step."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(7)

# ── Tiny 1-layer models as parameterised Gaussians ───────────────────────
# Generator:     G(z) = g_mu + exp(g_logs) * z,  z ~ N(0,1)
# Discriminator: D(x) = sigmoid(d_w * x + d_b)

g_mu, g_logs = np.array([-1.0]),  np.array([0.0])   # init far from target
d_w,  d_b    = np.array([0.5]),   np.array([0.0])

lr_d, lr_g = 0.05, 0.02
n_epochs    = 300
batch       = 256
snapshots   = {}

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

for ep in range(n_epochs):
    # ── Sample ──────────────────────────────────────────────────────────
    x_real = rng.normal(2.0, 0.5, batch)
    z      = rng.standard_normal(batch)
    x_fake = g_mu + np.exp(g_logs) * z    # generator

    # ── Discriminator update (maximise log D(real) + log(1-D(fake))) ────
    D_real = sigmoid(d_w * x_real + d_b)
    D_fake = sigmoid(d_w * x_fake + d_b)

    grad_dw = (np.mean((1 - D_real) * x_real) - np.mean(D_fake * x_fake))
    grad_db = (np.mean(1 - D_real)            - np.mean(D_fake))
    d_w += lr_d * grad_dw
    d_b += lr_d * grad_db

    # ── Generator update (non-saturating: maximise log D(fake)) ─────────
    D_fake2 = sigmoid(d_w * x_fake + d_b)
    # ∂log D(fake)/∂g_mu   = (1-D_fake) * d_w  (chain rule)
    grad_gmu  = np.mean((1 - D_fake2) * d_w)
    grad_glogs= np.mean((1 - D_fake2) * d_w * z * np.exp(g_logs))
    g_mu   += lr_g * grad_gmu
    g_logs += lr_g * grad_glogs

    if ep in [0, 20, 60, 150, 299]:
        snapshots[ep] = (float(g_mu), float(np.exp(g_logs)))

# ── Plot ─────────────────────────────────────────────────────────────────
x_plot = np.linspace(-4, 6, 500)
p_true = norm.pdf(x_plot, 2.0, 0.5)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))
for (ep, (mu, sig)), col in zip(sorted(snapshots.items()), colors):
    p_gen = norm.pdf(x_plot, mu, sig)
    axes[0].plot(x_plot, p_gen, color=col, lw=1.8,
                 label=f'Epoch {ep:3d}: N({mu:.2f}, {sig:.2f})')

axes[0].plot(x_plot, p_true, 'r--', lw=2.5, label='Target N(2.0, 0.5)')
axes[0].set_title('Generator Distribution Over Training')
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

ep_list = sorted(snapshots.keys())
mus  = [snapshots[e][0] for e in ep_list]
sigs = [snapshots[e][1] for e in ep_list]
axes[1].plot(ep_list, mus,  'o-', color='royalblue',  label='Generator μ')
axes[1].axhline(2.0, color='royalblue',  ls='--', lw=1)
axes[1].plot(ep_list, sigs, 's-', color='darkorange', label='Generator σ')
axes[1].axhline(0.5, color='darkorange', ls='--', lw=1)
axes[1].set_title('Generator Parameters vs Epoch')
axes[1].legend(); axes[1].grid(alpha=.3)

plt.suptitle('1-D GAN: Generator Chasing the Target Distribution',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('gan_1d_training.png', dpi=130, bbox_inches='tight')
plt.show()
final_mu, final_sig = snapshots[299]
print(f"Target:  μ=2.000, σ=0.500")
print(f"Learned: μ={final_mu:.3f}, σ={final_sig:.3f}")
""",
    },

    # ── 8 ──────────────────────────────────────────────────────────────────
    "8 · Classifier-Free Guidance (CFG) Visualisation": {
        "description": (
            "Simulates CFG in a toy 2-D conditional diffusion setting. "
            "Shows how increasing the guidance scale w sharpens the conditional "
            "distribution at the cost of sample diversity."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

rng = np.random.default_rng(42)

# ── Toy conditional: class y ∈ {0,1,2} each at different 2-D location ───
# We approximate the score as a Gaussian score for illustration.

class_centers = np.array([[0, 2], [2, -1], [-2, -1]], dtype=float)
class_stds    = [0.6, 0.6, 0.6]

# Unconditional = mixture of all classes
# CFG formula:  ε̃ = (1+w)·ε_cond − w·ε_uncond

def score_gaussian(x, mu, sigma):
    '''Gaussian score: -(x - mu)/sigma^2  (points toward mean)'''
    return -(x - mu) / sigma**2

def sample_cfg(y, w, n=500):
    # Start from N(0,3) noise
    x = rng.standard_normal((n, 2)) * 3
    # Simulate many denoising steps via gradient ascent on score
    mu_cond = class_centers[y]
    sigma   = class_stds[y]

    # Unconditional: mixture score  (approx as score of nearest component)
    # For clarity: uncond score = score toward global mean
    mu_uncond = class_centers.mean(axis=0)
    sigma_uncond = 2.0

    steps = 80
    for _ in range(steps):
        s_cond   = score_gaussian(x, mu_cond,   sigma)
        s_uncond = score_gaussian(x, mu_uncond, sigma_uncond)
        s_guided = (1 + w) * s_cond - w * s_uncond
        x = x + 0.05 * s_guided + rng.standard_normal(x.shape) * 0.01
    return x

guidance_scales = [0.0, 1.0, 3.0, 7.0]
fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex='row', sharey='row')

for row, y in enumerate(range(3)):
    for col, w in enumerate(guidance_scales):
        samples = sample_cfg(y, w)
        axes[row, col].scatter(samples[:, 0], samples[:, 1],
                               s=8, alpha=0.5, color=f'C{y}')
        axes[row, col].scatter(*class_centers[y], s=200, marker='*',
                               color='black', zorder=5, label='True centre')
        if row == 0:
            axes[row, col].set_title(f'w = {w}', fontsize=12, fontweight='bold')
        if col == 0:
            axes[row, col].set_ylabel(f'Class {y}', fontsize=11)
        axes[row, col].set_xlim(-5, 5); axes[row, col].set_ylim(-5, 5)
        axes[row, col].grid(alpha=.2)

plt.suptitle(
    'Classifier-Free Guidance: increasing w sharpens distribution\n'
    '(★ = true class centre)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('cfg_visualisation.png', dpi=130, bbox_inches='tight')
plt.show()
print("Higher guidance scale → samples closer to class centre, less spread.")
print("Trade-off: diversity ↓ as w ↑.")
""",
    },

    # ── 9 ──────────────────────────────────────────────────────────────────
    "9 · Latent Space Interpolation (VAE-style)": {
        "description": (
            "Demonstrates smooth latent space interpolation — a key advantage of VAEs. "
            "Linearly and spherically interpolates between two points in a 2-D latent "
            "space and decodes along the path, showing the transition."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

rng = np.random.default_rng(3)

# ── Toy decoder: latent z → 2-D data ────────────────────────────────────
# True mapping: x = A·z + b  (affine for illustration)
np.random.seed(5)
A = np.array([[1.5, -0.8], [0.6, 1.2]])
b = np.array([0.3, -0.2])

def decode(z):
    return z @ A.T + b          # (N, 2)

# ── Two latent endpoints ──────────────────────────────────────────────────
z_a = np.array([-2.0,  1.5])
z_b = np.array([ 2.0, -1.5])

# ── Linear interpolation ──────────────────────────────────────────────────
alphas = np.linspace(0, 1, 20)
z_lin  = np.outer(1 - alphas, z_a) + np.outer(alphas, z_b)

# ── Spherical linear interpolation (SLERP) ────────────────────────────────
def slerp(z0, z1, alphas):
    z0_n  = z0 / np.linalg.norm(z0)
    z1_n  = z1 / np.linalg.norm(z1)
    omega = np.arccos(np.clip(np.dot(z0_n, z1_n), -1, 1))
    if np.abs(omega) < 1e-6:
        return np.outer(1 - alphas, z0) + np.outer(alphas, z1)
    r = np.linalg.norm(z0) * (1-alphas) + np.linalg.norm(z1) * alphas  # lerp norms
    return r[:, None] * (
        np.sin((1-alphas)*omega)[:, None] / np.sin(omega) * z0_n +
        np.sin(   alphas *omega)[:, None] / np.sin(omega) * z1_n
    )

z_slerp = slerp(z_a, z_b, alphas)

x_lin   = decode(z_lin)
x_slerp = decode(z_slerp)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Latent space
axes[0].set_title('Latent Space Paths')
theta = np.linspace(0, 2*np.pi, 200)
axes[0].plot(2*np.cos(theta), 2*np.sin(theta), 'k--', lw=.6, alpha=.4, label='Unit circle × 2')
axes[0].plot(z_lin[:,0],   z_lin[:,1],   'o-', color='royalblue', ms=4, label='Linear lerp')
axes[0].plot(z_slerp[:,0], z_slerp[:,1], 's-', color='tomato',    ms=4, label='SLERP')
axes[0].scatter(*z_a, s=120, zorder=5, color='black'); axes[0].annotate('z_a', z_a+.1)
axes[0].scatter(*z_b, s=120, zorder=5, color='black'); axes[0].annotate('z_b', z_b+.1)
axes[0].legend(fontsize=9); axes[0].set_aspect('equal'); axes[0].grid(alpha=.3)

# Data space — linear
axes[1].plot(x_lin[:,0], x_lin[:,1], 'o-', color='royalblue', ms=5)
axes[1].scatter(*decode(z_a[None])[0], s=150, zorder=5, color='black', label='Start')
axes[1].scatter(*decode(z_b[None])[0], s=150, zorder=5, color='red',   label='End')
for i, alpha in enumerate(alphas):
    axes[1].annotate(f'{alpha:.1f}', x_lin[i]+.02, fontsize=6, color='royalblue')
axes[1].set_title('Decoded: Linear Interpolation')
axes[1].legend(fontsize=9); axes[1].grid(alpha=.3)

# Data space — slerp
axes[2].plot(x_slerp[:,0], x_slerp[:,1], 's-', color='tomato', ms=5)
axes[2].scatter(*decode(z_a[None])[0], s=150, zorder=5, color='black', label='Start')
axes[2].scatter(*decode(z_b[None])[0], s=150, zorder=5, color='red',   label='End')
axes[2].set_title('Decoded: SLERP Interpolation')
axes[2].legend(fontsize=9); axes[2].grid(alpha=.3)

plt.suptitle('VAE Latent Space Interpolation  (Linear vs SLERP)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('latent_interpolation.png', dpi=130, bbox_inches='tight')
plt.show()
print("SLERP respects the geometry of the latent sphere — produces more uniform speed.")
""",
    },

    # ── 10 ─────────────────────────────────────────────────────────────────
    "10 · DDPM Reverse Process Simulation": {
        "description": (
            "Simulates DDPM's reverse denoising process with a known clean signal. "
            "Uses the analytical reverse posterior (since we know x_0) to visualise "
            "how the signal is progressively reconstructed from pure noise."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(99)

# ── Schedule ─────────────────────────────────────────────────────────────
T     = 200
beta  = np.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
abar  = np.concatenate([[1.0], np.cumprod(alpha)])  # abar[0]=ᾱ₀=1, abar[t]=ᾱ_t; prevents off-by-one at t=1

N = 128
x0 = np.sin(np.linspace(0, 3*np.pi, N))   # clean "image"

# ── Forward: compute x_T ─────────────────────────────────────────────────
eps_true = rng.standard_normal(N)
xT = np.sqrt(abar[T]) * x0 + np.sqrt(1 - abar[T]) * eps_true  # abar[T] = ᾱ_T

# ── Reverse: use DDPM posterior mean (oracle — knows x0) ─────────────────
# μ̃_t(x_t, x_0) = (√ᾱ_{t-1}β_t / (1-ᾱ_t)) x_0  +  (√αₜ(1-ᾱ_{t-1}) / (1-ᾱ_t)) x_t
# β̃_t = (1-ᾱ_{t-1}) / (1-ᾱ_t) · β_t

snapshots = {T: xT.copy()}
x_t = xT.copy()

for t in range(T-1, 0, -1):
    abar_t  = abar[t]      # ᾱ_t
    abar_tm = abar[t-1]    # ᾱ_{t-1}; abar[0]=1.0 so t=1 correctly gives ᾱ₀=1.0
    bt      = beta[t]
    bt_tild = (1 - abar_tm) / (1 - abar_t) * bt
    bt_tild = max(bt_tild, 1e-10)

    mu_tild = (np.sqrt(abar_tm) * bt / (1 - abar_t)) * x0 + \
              (np.sqrt(alpha[t]) * (1 - abar_tm) / (1 - abar_t)) * x_t

    z   = rng.standard_normal(N) if t > 1 else np.zeros(N)
    x_t = mu_tild + np.sqrt(bt_tild) * z

    if t in [T-1, 150, 100, 50, 20, 1]:
        snapshots[t] = x_t.copy()

# ── Plot ──────────────────────────────────────────────────────────────────
plot_steps = [T, T-1, 150, 100, 50, 20, 1]
fig, axes  = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
axes = axes.flatten()

for i, t in enumerate(plot_steps[:8]):
    axes[i].plot(snapshots[t], lw=1.2, color='steelblue', label=f't={t}')
    axes[i].plot(x0, lw=1, color='tomato', ls='--', label='x₀ (true)')
    axes[i].set_title(f't = {t}')
    axes[i].legend(fontsize=7); axes[i].grid(alpha=.3)
    rmse = np.sqrt(np.mean((snapshots[t] - x0)**2))
    axes[i].set_xlabel(f'RMSE = {rmse:.3f}')

axes[-1].axis('off')
plt.suptitle('DDPM Reverse Process: Recovering Clean Signal from Noise',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ddpm_reverse.png', dpi=130, bbox_inches='tight')
plt.show()

rmse_T = np.sqrt(np.mean((xT - x0)**2))
rmse_1 = np.sqrt(np.mean((snapshots[1] - x0)**2))
print(f"RMSE at t=T (pure noise)      : {rmse_T:.4f}")
print(f"RMSE at t=1  (reconstructed)  : {rmse_1:.4f}")
print(f"Improvement factor            : {rmse_T/rmse_1:.1f}×")
""",
    },

    # ── 11 ─────────────────────────────────────────────────────────────────
    "11 · β-VAE: Disentanglement vs Reconstruction Trade-off": {
        "description": (
            "Trains VAEs with different β values on a 2-D dataset and shows how "
            "increasing β forces the posterior toward N(0,I) (more disentangled) "
            "at the cost of reconstruction accuracy."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(21)

# ── Dataset: correlated 2-D Gaussian ─────────────────────────────────────
N   = 1000
cov = [[1.0, 0.85], [0.85, 1.0]]
X   = rng.multivariate_normal([0, 0], cov, N)

D, Z = 2, 2
lr   = 3e-3
epochs = 300

beta_values = [0.1, 1.0, 4.0, 10.0]
results = {}

for beta in beta_values:
    # Re-initialise parameters
    rng2 = np.random.default_rng(42)
    Wemu = rng2.standard_normal((D, Z)) * 0.1; bemu = np.zeros(Z)
    Welv = rng2.standard_normal((D, Z)) * 0.1; belv = np.zeros(Z)
    Wdec = rng2.standard_normal((Z, D)) * 0.1; bdec = np.zeros(D)

    rec_hist, kl_hist = [], []

    for ep in range(epochs):
        eps  = rng2.standard_normal((N, Z))
        mu   = X @ Wemu + bemu
        logv = X @ Welv + belv
        sig  = np.exp(0.5 * logv)
        z    = mu + sig * eps
        xh   = z @ Wdec + bdec

        diff  = X - xh
        rec   = 0.5 * np.mean(np.sum(diff**2, axis=1))
        kl    = -0.5 * np.mean(np.sum(1 + logv - mu**2 - np.exp(logv), axis=1))
        rec_hist.append(float(rec)); kl_hist.append(float(kl))

        dxh  = -(diff) / N
        dWd  = z.T @ dxh;  dbd = dxh.mean(0)
        dz   = dxh @ Wdec.T
        dm   = dz + beta * mu / N
        dlv  = (dz * eps * 0.5 * sig) + beta * (-0.5 * (1 - np.exp(logv))) / N  # eps included: chain rule z=mu+sig*eps → dL/dlogv = dL/dz * eps * 0.5*sig

        Wemu -= lr * (X.T @ dm);  bemu -= lr * dm.mean(0)
        Welv -= lr * (X.T @ dlv); belv -= lr * dlv.mean(0)
        Wdec -= lr * dWd;         bdec -= lr * dbd

    final_mu   = X @ Wemu + bemu
    final_logv = X @ Welv + belv

    results[beta] = {
        'rec':    rec_hist,
        'kl':     kl_hist,
        'mu':     final_mu,
        'logv':   final_logv,
        'Wdec':   Wdec,
        'bdec':   bdec,
    }

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
colors = ['royalblue', 'forestgreen', 'darkorange', 'tomato']

for col, (beta, res) in enumerate(results.items()):
    ep_ax = np.arange(epochs)

    axes[0, col].plot(ep_ax, res['rec'], color=colors[col], lw=2, label='Rec loss')
    axes[0, col].set_title(f'β = {beta}', fontsize=11, fontweight='bold')
    axes[0, col].set_xlabel('Epoch')
    if col == 0: axes[0, col].set_ylabel('Loss')
    axes[0, col].grid(alpha=.3)

    ax2 = axes[0, col].twinx()
    ax2.plot(ep_ax, res['kl'], color='grey', lw=1.5, ls='--', label='KL')
    ax2.set_ylabel('KL', color='grey')

    # Latent scatter
    mu = res['mu']
    axes[1, col].scatter(mu[:, 0], mu[:, 1], s=8, alpha=.4, color=colors[col])
    # Draw unit circle
    th = np.linspace(0, 2*np.pi, 200)
    axes[1, col].plot(np.cos(th), np.sin(th), 'k--', lw=.8, alpha=.5)
    axes[1, col].set_title(f'Latent μ  (β={beta})', fontsize=10)
    axes[1, col].set_aspect('equal')
    axes[1, col].set_xlim(-4, 4); axes[1, col].set_ylim(-4, 4)
    axes[1, col].grid(alpha=.2)
    corr = np.corrcoef(mu[:, 0], mu[:, 1])[0, 1]
    axes[1, col].set_xlabel(f'Latent correlation = {corr:.2f}')

plt.suptitle('β-VAE: Higher β → More Disentangled (but Blurrier)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('beta_vae.png', dpi=130, bbox_inches='tight')
plt.show()

print("β     | Final Rec Loss | Final KL  | Latent Correlation")
print("------|----------------|-----------|-------------------")
for beta, res in results.items():
    mu = res['mu']
    corr = np.corrcoef(mu[:,0], mu[:,1])[0,1]
    print(f"β={beta:5.1f} | {res['rec'][-1]:.4f}          | {res['kl'][-1]:.4f}    | {corr:.4f}")
""",
    },

    # ── 12 ─────────────────────────────────────────────────────────────────
    "12 · Model Comparison Dashboard": {
        "description": (
            "Comprehensive comparison of VAE, GAN and Diffusion Models across "
            "key axes: sample quality, diversity, training stability, inference speed, "
            "and likelihood availability — rendered as a radar chart and score table."
        ),
        "language": "python",
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Scores (0–10) for each model on each axis ─────────────────────────────
categories = [
    'Sample\nQuality',
    'Sample\nDiversity',
    'Training\nStability',
    'Inference\nSpeed',
    'Likelihood\nEst.',
    'Structured\nLatent',
    'Conditioning\nEase',
]
models = {
    'VAE':       [5.5, 7.5, 9.5, 9.5, 8.0, 9.5, 7.0],
    'GAN':       [8.5, 5.5, 4.5, 9.5, 1.0, 4.0, 6.0],
    'Diffusion': [9.5, 9.0, 9.0, 3.5, 7.5, 3.0, 9.0],
}
colors = {'VAE': 'royalblue', 'GAN': 'tomato', 'Diffusion': 'forestgreen'}

fig = plt.figure(figsize=(16, 6))

# ── Radar chart ───────────────────────────────────────────────────────────
ax_radar = fig.add_subplot(131, polar=True)
N_cat = len(categories)
angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]

for model, scores in models.items():
    vals = scores + scores[:1]
    ax_radar.plot(angles, vals, 'o-', color=colors[model], lw=2, label=model)
    ax_radar.fill(angles, vals, color=colors[model], alpha=0.12)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, fontsize=8.5)
ax_radar.set_ylim(0, 10)
ax_radar.set_yticks([2, 4, 6, 8, 10])
ax_radar.set_yticklabels(['2','4','6','8','10'], fontsize=7)
ax_radar.set_title('Model Comparison\n(Radar)', fontsize=11, fontweight='bold', pad=15)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

# ── Bar chart per category ────────────────────────────────────────────────
ax_bar = fig.add_subplot(132)
x = np.arange(N_cat)
w = 0.25
for i, (model, scores) in enumerate(models.items()):
    ax_bar.bar(x + i*w, scores, width=w, color=colors[model], alpha=0.85, label=model)
ax_bar.set_xticks(x + w)
ax_bar.set_xticklabels([c.replace('\n',' ') for c in categories],
                        rotation=35, ha='right', fontsize=8)
ax_bar.set_ylim(0, 11)
ax_bar.set_ylabel('Score (0–10)')
ax_bar.set_title('Score per Axis', fontsize=11, fontweight='bold')
ax_bar.legend(fontsize=9); ax_bar.grid(axis='y', alpha=.3)

# ── Summary table ─────────────────────────────────────────────────────────
ax_tbl = fig.add_subplot(133)
ax_tbl.axis('off')
table_data = [['Model', 'Best for', 'Weakness']]
summaries = {
    'VAE':       ('Structured latent,\nsmooth interpolation', 'Blurry samples'),
    'GAN':       ('Sharp images,\nfast inference', 'Mode collapse,\nunstable'),
    'Diffusion': ('Highest quality,\ndiverse + conditioned', 'Slow sampling'),
}
for model, (best, weak) in summaries.items():
    table_data.append([model, best, weak])

tbl = ax_tbl.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc='center', loc='center',
    bbox=[0, 0.25, 1, 0.7]
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2d2d2d'); cell.set_text_props(color='white', fontweight='bold')
    elif c == 0:
        cell.set_facecolor(list(colors.values())[r-1]); cell.set_alpha(0.6)
    cell.set_edgecolor('grey')
ax_tbl.set_title('Summary', fontsize=11, fontweight='bold', pad=10)

plt.suptitle('Generative Model Family Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=130, bbox_inches='tight')
plt.show()

print("\\nOverall scores (mean across all axes):")
for model, scores in models.items():
    print(f"  {model:12s}: {np.mean(scores):.2f} / 10")
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