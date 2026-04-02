import textwrap
import re

"""Module: 06 · Autoencoders"""

TOPIC_NAME   ="Autoencoders"
DISPLAY_NAME = "06 · Autoencoders"
ICON         = "🔄"
SUBTITLE     = "Neural encoder-decoder for unsupervised representation learning"

THEORY = """
## 06 · Autoencoders

---

### The Core Idea

An autoencoder is a neural network trained to do something that sounds trivial:
**reproduce its own input as its output**.

    Input x  →  [Encoder]  →  z  →  [Decoder]  →  x̂  ≈  x

If the network had no constraints, the trivial solution would be to copy the
input directly — an identity mapping. The constraint that makes autoencoders
useful is the **bottleneck**: the hidden representation z (the latent code)
must be much smaller than the input x.

Forcing all information through this narrow channel makes the network learn
a compressed representation that captures the essential structure of the data.
The encoder learns to extract the most important features; the decoder learns
to reconstruct the input from those features alone.

This is unsupervised learning — no labels needed. The input itself serves as
its own target.

---

### Architecture

A standard autoencoder has three parts:

**Encoder** — maps input x (dimension p) to latent code z (dimension d, d << p):

    z = f_enc(x) = activation(W_enc · x + b_enc)

In a deep encoder, this is a stack of layers that progressively compress:

    x (p) → h₁ (p/2) → h₂ (p/4) → z (d)

**Bottleneck (Latent Space)** — the compressed representation z. Its
dimensionality d is the key hyperparameter. Too large: trivial copying.
Too small: the network cannot reconstruct the input accurately. Typically
d is chosen as 2–10% of the input dimension, or tuned by reconstruction error.

**Decoder** — maps latent code z back to reconstructed input x̂:

    x̂ = f_dec(z) = activation(W_dec · z + b_dec)

Mirror architecture to the encoder:

    z (d) → h₁ (p/4) → h₂ (p/2) → x̂ (p)

The full autoencoder:

    x̂ = f_dec(f_enc(x))

---

### Training Objective

The autoencoder minimises **reconstruction loss** — the difference between
the input and its reconstruction:

**Mean Squared Error (for continuous inputs):**

    L = (1/n) · Σᵢ ||xᵢ − x̂ᵢ||² = (1/n) · Σᵢ Σⱼ (xᵢⱼ − x̂ᵢⱼ)²

**Binary Cross-Entropy (for binary inputs, e.g. pixel intensities in [0,1]):**

    L = −(1/n) · Σᵢ Σⱼ [xᵢⱼ · log(x̂ᵢⱼ) + (1−xᵢⱼ) · log(1−x̂ᵢⱼ)]

Trained with standard backpropagation and gradient descent. The encoder and
decoder weights are optimised jointly — the gradient flows from the output
loss all the way back through the decoder, through the bottleneck, through
the encoder to the input layer.

---

### A Worked Example (Tiny Autoencoder, 4D → 2D → 4D)

Architecture: Linear(4→2) → ReLU → Linear(2→4)

Data point: x = [1.0, 0.5, 0.8, 0.2]

**Forward pass:**

Encoder weights W_enc (2×4), biases b_enc (2):
    W_enc = [[0.4, -0.2,  0.7,  0.1],
             [-0.3, 0.8, -0.1,  0.5]]
    b_enc = [0.1, -0.1]

Pre-activation h:
    h[0] = 0.4(1.0) + (−0.2)(0.5) + 0.7(0.8) + 0.1(0.2) + 0.1
         = 0.4 − 0.1 + 0.56 + 0.02 + 0.1 = 0.98
    h[1] = (−0.3)(1.0) + 0.8(0.5) + (−0.1)(0.8) + 0.5(0.2) − 0.1
         = −0.3 + 0.4 − 0.08 + 0.1 − 0.1 = 0.02

Latent code z = ReLU(h) = [max(0, 0.98), max(0, 0.02)] = [0.98, 0.02]

Decoder weights W_dec (4×2), biases b_dec (4):
    W_dec = [[ 0.6,  0.3],
             [ 0.2,  0.7],
             [ 0.8, -0.1],
             [-0.1,  0.4]]
    b_dec = [0.0, 0.1, 0.0, 0.0]

Reconstruction x̂:
    x̂[0] = 0.6(0.98) + 0.3(0.02) + 0.0 = 0.588 + 0.006 = 0.594
    x̂[1] = 0.2(0.98) + 0.7(0.02) + 0.1 = 0.196 + 0.014 + 0.1 = 0.310
    x̂[2] = 0.8(0.98) − 0.1(0.02) + 0.0 = 0.784 − 0.002 = 0.782
    x̂[3] = −0.1(0.98) + 0.4(0.02) + 0.0 = −0.098 + 0.008 = −0.090

**Reconstruction loss (MSE):**
    L = [(1.0−0.594)² + (0.5−0.310)² + (0.8−0.782)² + (0.2−(−0.090))²] / 4
      = [0.165 + 0.036 + 0.000 + 0.084] / 4
      = 0.285 / 4 = 0.071

Backpropagation adjusts all weights to reduce this loss. After many iterations
on many data points, the encoder learns a meaningful 2D compression of the data.

---

### Variants of Autoencoders

#### Denoising Autoencoder (DAE)

The standard autoencoder can overfit by learning a near-identity mapping even
with a small bottleneck (by memorising training examples). The denoising
autoencoder prevents this by **corrupting the input** and training the network
to reconstruct the original clean version:

    x̃ = x + ε    where ε ~ N(0, σ²)   (Gaussian noise)
    or:  x̃ = x ⊙ mask                  (randomly zero out features)

    Minimise: L = ||x − f_dec(f_enc(x̃))||²

By learning to remove noise, the network is forced to learn the underlying
data manifold rather than memorising individual points. The denoising objective
also gives the autoencoder a probabilistic interpretation: it is learning to
map noisy observations back to the clean data manifold.

---

#### Sparse Autoencoder

Rather than constraining the bottleneck size, a sparse autoencoder encourages
most latent neurons to be inactive (near zero) for any given input. This is
enforced by adding a sparsity penalty to the loss:

    L = ||x − x̂||² + β · Σⱼ KL(ρ || ρ̂ⱼ)

where:
- ρ is the target sparsity (e.g. 0.05 — each neuron should activate only 5% of the time)
- ρ̂ⱼ = (1/n) Σᵢ hⱼ(xᵢ) is the average activation of neuron j over the training set
- KL(ρ || ρ̂) = ρ·log(ρ/ρ̂) + (1−ρ)·log((1−ρ)/(1−ρ̂)) is the KL divergence penalty

Sparse autoencoders tend to learn **interpretable features** — each latent
neuron activates for a specific, recognisable pattern in the input. Used
extensively in neuroscience-inspired AI research and mechanistic interpretability.

---

#### Variational Autoencoder (VAE)

The VAE is the most important variant. Instead of encoding x to a single
point z in latent space, the VAE encodes x to a **distribution**:

    q(z|x) = N(μ(x), σ²(x))

The encoder outputs two vectors: μ (the mean) and log σ² (the log variance).
The latent code z is sampled from this distribution during training:

    z = μ + σ ⊙ ε    where ε ~ N(0, I)

This **reparameterisation trick** allows gradients to flow through the sampling
operation — the randomness is pushed into ε, which is not a function of the
network parameters.

The VAE training objective is the **ELBO (Evidence Lower BOund)**:

    ELBO = E_q[log p(x|z)]  −  KL(q(z|x) || p(z))
         = Reconstruction term  −  Regularisation term

The first term maximises reconstruction quality (same as standard AE).
The second term is the KL divergence between the learned posterior q(z|x)
and the prior p(z) = N(0, I). It encourages the latent space to be smooth
and continuous — similar inputs should map to nearby distributions.

The minus sign: the total loss minimised = −ELBO = Reconstruction loss + KL term.

**Why the KL term matters:**
Without it, the encoder could map each training point to a tiny region of
latent space, leaving vast empty gaps. With the KL penalty, the encoder must
spread its distributions across the prior, ensuring that random samples from
N(0, I) decode to realistic outputs — enabling **generation**.

---

#### Contractive Autoencoder (CAE)

Adds a penalty on the Frobenius norm of the Jacobian of the encoder:

    L = ||x − x̂||² + λ · ||∂f_enc(x)/∂x||²_F

This forces the encoder's representation to be insensitive to small
perturbations of the input — making the latent code robust and local
variations are ignored. Used for learning invariant feature representations.

---

### What the Latent Space Learns

A well-trained autoencoder's latent space has structure that reflects the
underlying data manifold:

**Clustering** — Points from the same class cluster together in latent space
even though no class labels were provided. The encoder discovers class structure
through reconstruction pressure alone.

**Interpolation** — Moving smoothly through latent space produces smooth
transitions in the decoded output. This is weak in standard autoencoders
(the latent space may have gaps) but strong in VAEs.

**Disentanglement** — Ideally, each dimension of z corresponds to an independent
factor of variation in the data (shape, colour, orientation, etc.). Standard
autoencoders rarely achieve this without additional constraints, but
Beta-VAE (with a larger KL weight β > 1) encourages disentangled representations.

**Dimensionality** — The intrinsic dimensionality of the learned latent space
often matches the true complexity of the data, independent of the chosen
bottleneck size (though reconstruction quality degrades if the bottleneck is
too small).

---

### Autoencoders vs PCA

Both compress data into a lower-dimensional representation. The key difference:

| Property                  | PCA                              | Autoencoder                        |
|---------------------------|----------------------------------|------------------------------------|
| Compression type          | Linear projection                | Nonlinear mapping                  |
| Reconstruction            | Optimal linear reconstruction    | Can be near-perfect (nonlinear)    |
| Latent space structure    | Orthogonal, ordered by variance  | Arbitrary, learned                 |
| Interpretability          | Components = variance directions | Latent dims may lack clear meaning |
| Training                  | Closed-form (SVD)                | Iterative gradient descent         |
| Scalability               | O(np²) or O(n²p)                 | Scales with batch size             |
| Generation                | Not supported                    | VAE enables generation             |
| Handles nonlinear manifolds| No                               | Yes                                |

---

### Applications

**Anomaly Detection** — Train on normal data only. At inference, compute
reconstruction error for each sample. Anomalous samples lie far from the
learned manifold → high reconstruction error → flagged as anomalies. Used in
network intrusion detection, fraud detection, medical imaging, manufacturing QC.

**Noise Removal / Image Denoising** — Denoising autoencoders trained on
clean images learn to project noisy inputs back onto the image manifold.

**Dimensionality Reduction for Downstream Tasks** — Use the encoder as a
feature extractor. The latent code z is a dense, semantically meaningful
representation suitable for clustering, classification, or retrieval.

**Data Compression** — The encoder–decoder pair is a learnable lossy
compression codec. Neural image codecs based on autoencoders now outperform
JPEG at low bit rates.

**Generative Modelling (VAE)** — Sample z ~ N(0, I) and decode to generate
new data points. VAEs can interpolate between data points by mixing their
latent codes.

**Representation Learning** — Pretrain an encoder on a large unlabelled
dataset, then fine-tune the encoder (or use its output as features) for a
supervised downstream task with limited labels.

---

### Loss Landscape and Training Tips

**Vanishing gradients** — Deep autoencoders suffer from vanishing gradients
(the gradient is multiplied by weights at each layer on the way back). Use
ReLU or LeakyReLU activations, batch normalisation, and residual connections.

**Posterior collapse (VAE)** — The KL term can push the encoder to ignore z
entirely (posterior = prior, all information carried by the prior mean).
Solutions: KL annealing (gradually increase the KL weight from 0 to 1 during
training), free bits (allow a minimum KL before penalising).

**Mode collapse** — The decoder learns only a few outputs regardless of z.
Address with capacity-controlled architectures and careful learning rate tuning.

**Checkerboard artefacts** — Deconvolution layers can produce grid artefacts.
Use nearest-neighbour upsampling + convolution instead of transposed convolutions.

**Reconstruction blur (VAE)** — VAEs with MSE loss produce blurry reconstructions
because MSE averages over all plausible outputs. Use perceptual loss (feature
matching against a pretrained network) or adversarial training (VAE-GAN) for
sharper results.

---

### Key Takeaways

1. An autoencoder is an encoder–decoder network trained to reconstruct its
   input through a narrow bottleneck. No labels needed — the input is the target.

2. The bottleneck forces the network to learn a compressed representation z
   that captures the essential structure of the data.

3. The denoising AE prevents overfitting by corrupting inputs; the sparse AE
   adds a sparsity penalty; the VAE encodes to distributions and adds a KL
   regularisation term enabling smooth, generatable latent spaces.

4. The VAE's ELBO = Reconstruction term − KL(posterior || prior). The KL
   term ensures the latent space is smooth and prevents gaps between clusters.

5. Autoencoders are nonlinear — they strictly generalise PCA. They are used
   for anomaly detection, denoising, compression, representation learning,
   and (in the VAE form) generative modelling.
"""


OPERATIONS = {

    "▶ Run: Autoencoder From Scratch (Full Training Loop)": {
        "description": "Complete autoencoder with encoder, decoder, MSE loss, backpropagation, and gradient descent. Trains on synthetic 6D data with 2D latent space. Prints loss curve and final reconstruction quality.",
        "code": """
import math
import random

random.seed(42)

# ── Activation functions ───────────────────────────────────────────────────────
def relu(x):  return [max(0.0, v) for v in x]
def relu_d(x): return [1.0 if v > 0 else 0.0 for v in x]

def sigmoid(x):
    return [1.0/(1.0+math.exp(-min(max(v,-20),20))) for v in x]
def sigmoid_d(s): return [v*(1-v) for v in s]

# ── Dense layer ────────────────────────────────────────────────────────────────
def make_layer(in_dim, out_dim, scale=0.1):
    W = [[random.gauss(0, scale) for _ in range(in_dim)] for _ in range(out_dim)]
    b = [0.0] * out_dim
    return {'W': W, 'b': b, 'in_dim': in_dim, 'out_dim': out_dim}

def forward_layer(layer, x):
    out = [sum(layer['W'][i][j]*x[j] for j in range(len(x))) + layer['b'][i]
           for i in range(layer['out_dim'])]
    return out

def backward_layer(layer, x, grad_out, lr):
    # Compute grad w.r.t. input
    grad_in = [sum(layer['W'][j][i]*grad_out[j] for j in range(layer['out_dim']))
               for i in range(layer['in_dim'])]
    # Update weights
    for i in range(layer['out_dim']):
        for j in range(layer['in_dim']):
            layer['W'][i][j] -= lr * grad_out[i] * x[j]
        layer['b'][i] -= lr * grad_out[i]
    return grad_in

# ── Dataset: 3 clusters in 6D ─────────────────────────────────────────────────
def blob(cx, n):
    return [[cx[d]+random.gauss(0,0.4) for d in range(len(cx))] for _ in range(n)]

X = (blob([2,2,1,0,0,0], 30)
   + blob([-2,0,0,2,1,0], 30)
   + blob([0,-2,0,0,-1,2], 30))
true_labels = [0]*30 + [1]*30 + [2]*30
n, p = len(X), len(X[0])

# Normalise to [-1,1]
for j in range(p):
    mn = min(X[i][j] for i in range(n))
    mx = max(X[i][j] for i in range(n))
    rng = mx-mn or 1.0
    for i in range(n):
        X[i][j] = 2*(X[i][j]-mn)/rng - 1

# ── Architecture: 6 → 4 → 2 → 4 → 6 ─────────────────────────────────────────
LATENT_DIM = 2
LR         = 0.01
EPOCHS     = 300
BATCH_SIZE = 15

enc1 = make_layer(6, 4)
enc2 = make_layer(4, LATENT_DIM)
dec1 = make_layer(LATENT_DIM, 4)
dec2 = make_layer(4, 6)
layers = [enc1, enc2, dec1, dec2]

print(f"Autoencoder: 6 → 4 → {LATENT_DIM} → 4 → 6  |  n={n}")
print(f"Training: {EPOCHS} epochs, lr={LR}, batch={BATCH_SIZE}")
print()
print(f"  {'Epoch':>6}  {'MSE Loss':>10}  Progress")
print("  " + "─"*45)

loss_history = []

for epoch in range(1, EPOCHS+1):
    # Mini-batch SGD
    indices = list(range(n))
    random.shuffle(indices)
    epoch_loss = 0.0
    n_batches  = 0

    for start in range(0, n, BATCH_SIZE):
        batch = indices[start:start+BATCH_SIZE]
        batch_loss = 0.0

        for idx in batch:
            x = X[idx]

            # ── Forward pass ──────────────────────────────────────────────────
            h1_pre = forward_layer(enc1, x)
            h1     = relu(h1_pre)
            z_pre  = forward_layer(enc2, h1)
            z      = relu(z_pre)
            h2_pre = forward_layer(dec1, z)
            h2     = relu(h2_pre)
            xhat   = forward_layer(dec2, h2)  # linear output

            # ── MSE loss ──────────────────────────────────────────────────────
            loss = sum((x[j]-xhat[j])**2 for j in range(p)) / p
            batch_loss += loss

            # ── Backward pass ─────────────────────────────────────────────────
            # dL/dxhat = 2*(xhat - x) / p
            dL_dxhat = [2.0*(xhat[j]-x[j])/p for j in range(p)]

            # Decoder layer 2 (linear output → no activation grad)
            dL_dh2 = backward_layer(dec2, h2, dL_dxhat, LR)

            # ReLU at h2
            dL_dh2_pre = [dL_dh2[k]*relu_d(h2_pre)[k] for k in range(len(h2_pre))]

            # Decoder layer 1
            dL_dz = backward_layer(dec1, z, dL_dh2_pre, LR)

            # ReLU at z
            dL_dz_pre = [dL_dz[k]*relu_d(z_pre)[k] for k in range(len(z_pre))]

            # Encoder layer 2
            dL_dh1 = backward_layer(enc2, h1, dL_dz_pre, LR)

            # ReLU at h1
            dL_dh1_pre = [dL_dh1[k]*relu_d(h1_pre)[k] for k in range(len(h1_pre))]

            # Encoder layer 1
            backward_layer(enc1, x, dL_dh1_pre, LR)

        epoch_loss += batch_loss / len(batch)
        n_batches  += 1

    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)

    if epoch in (1, 25, 50, 100, 150, 200, 250, 300):
        bar = "█" * int((1-min(avg_loss,1.0))*20)
        print(f"  {epoch:>6}  {avg_loss:>10.5f}  {bar}")

# ── Encode all points and show latent space structure ─────────────────────────
print()
print("Latent space cluster analysis:")
latent_codes = []
for idx in range(n):
    h1 = relu(forward_layer(enc1, X[idx]))
    z  = relu(forward_layer(enc2, h1))
    latent_codes.append(z)

# Compute within/between cluster distances in latent space
from collections import defaultdict
class_pts = defaultdict(list)
for i, lbl in enumerate(true_labels):
    class_pts[lbl].append(latent_codes[i])

def edist(a, b): return math.sqrt(sum((a[k]-b[k])**2 for k in range(len(a))))

intra_dists = []
for lbl, pts in class_pts.items():
    d = [edist(pts[a],pts[b]) for a in range(len(pts)) for b in range(a+1,len(pts))]
    intra_dists.append(sum(d)/len(d) if d else 0)

inter_dists = []
lbls = list(class_pts.keys())
for i in range(len(lbls)):
    for j in range(i+1, len(lbls)):
        pa, pb = class_pts[lbls[i]], class_pts[lbls[j]]
        d = [edist(a,b) for a in pa for b in pb]
        inter_dists.append(sum(d)/len(d) if d else 0)

print(f"  Mean intra-cluster dist (latent): {sum(intra_dists)/len(intra_dists):.4f}")
print(f"  Mean inter-cluster dist (latent): {sum(inter_dists)/len(inter_dists):.4f}")
print(f"  Separation ratio                : {(sum(inter_dists)/len(inter_dists))/(sum(intra_dists)/len(intra_dists)):.2f}x")
print()

# ── Final reconstruction error ─────────────────────────────────────────────────
total_mse = 0
for idx in range(n):
    h1   = relu(forward_layer(enc1, X[idx]))
    z    = relu(forward_layer(enc2, h1))
    h2   = relu(forward_layer(dec1, z))
    xhat = forward_layer(dec2, h2)
    total_mse += sum((X[idx][j]-xhat[j])**2 for j in range(p)) / p
avg_mse = total_mse / n
print(f"Final average reconstruction MSE : {avg_mse:.5f}")
print(f"Initial loss (epoch 1)           : {loss_history[0]:.5f}")
print(f"Reduction                        : {loss_history[0]/avg_mse:.1f}x")
""",
        "runnable": True,
    },

    "▶ Run: Denoising Autoencoder": {
        "description": "Train a denoising autoencoder — corrupts input with Gaussian noise, learns to reconstruct the clean version. Compares reconstruction quality on clean vs noisy inputs before and after denoising training.",
        "code": """
import math
import random

random.seed(0)

# ── Dataset: 5D with clear structure ──────────────────────────────────────────
def blob(cx, n):
    return [[cx[d]+random.gauss(0,0.3) for d in range(len(cx))] for _ in range(n)]

X_clean = (blob([1,1,0,0,0], 25)
         + blob([-1,0,1,0,0], 25)
         + blob([0,-1,-1,0,0], 25))
n, p = len(X_clean), len(X_clean[0])

# Normalise
for j in range(p):
    mn=min(X_clean[i][j] for i in range(n)); mx=max(X_clean[i][j] for i in range(n))
    r=mx-mn or 1.0
    for i in range(n): X_clean[i][j]=2*(X_clean[i][j]-mn)/r-1

# ── Shared model utilities ─────────────────────────────────────────────────────
def relu(x):   return [max(0.0,v) for v in x]
def relu_d(x): return [1.0 if v>0 else 0.0 for v in x]

def make_layer(in_d, out_d, scale=0.1):
    return {'W':[[random.gauss(0,scale) for _ in range(in_d)] for _ in range(out_d)],
            'b':[0.0]*out_d, 'in_dim':in_d, 'out_dim':out_d}

def fwd(layer, x):
    return [sum(layer['W'][i][j]*x[j] for j in range(len(x)))+layer['b'][i]
            for i in range(layer['out_dim'])]

def bwd(layer, x, g, lr):
    gi=[sum(layer['W'][j][i]*g[j] for j in range(layer['out_dim']))
        for i in range(layer['in_dim'])]
    for i in range(layer['out_dim']):
        for j in range(layer['in_dim']): layer['W'][i][j]-=lr*g[i]*x[j]
        layer['b'][i]-=lr*g[i]
    return gi

def train_ae(X_input, X_target, n_epochs=200, lr=0.01, noise_tag=""):
    random.seed(1)
    e1=make_layer(p,3); e2=make_layer(3,2)
    d1=make_layer(2,3); d2=make_layer(3,p)
    losses=[]
    for epoch in range(1,n_epochs+1):
        idx=list(range(n)); random.shuffle(idx)
        ep_loss=0
        for i in idx:
            xi=X_input[i]; xt=X_target[i]
            # Forward
            h1_p=fwd(e1,xi); h1=relu(h1_p)
            z_p =fwd(e2,h1); z =relu(z_p)
            h2_p=fwd(d1,z);  h2=relu(h2_p)
            xh  =fwd(d2,h2)
            loss=sum((xt[j]-xh[j])**2 for j in range(p))/p
            ep_loss+=loss
            # Backward
            g=bwd(d2,h2,[2*(xh[j]-xt[j])/p for j in range(p)],lr)
            g=bwd(d1,z, [g[k]*relu_d(h2_p)[k] for k in range(len(h2_p))],lr)
            g=bwd(e2,h1,[g[k]*relu_d(z_p)[k]  for k in range(len(z_p))],lr)
            bwd(e1,xi,[g[k]*relu_d(h1_p)[k]  for k in range(len(h1_p))],lr)
        losses.append(ep_loss/n)
    return {'e1':e1,'e2':e2,'d1':d1,'d2':d2}, losses

def predict(model, xi):
    h1=relu(fwd(model['e1'],xi)); z=relu(fwd(model['e2'],h1))
    h2=relu(fwd(model['d1'],z));  return fwd(model['d2'],h2)

def mse(a,b): return sum((a[j]-b[j])**2 for j in range(len(a)))/len(a)

NOISE_STD = 0.5

# Add noise to inputs
X_noisy = [[X_clean[i][j]+random.gauss(0,NOISE_STD) for j in range(p)]
           for i in range(n)]

print(f"Denoising Autoencoder  |  5D → 2D → 5D  |  n={n}")
print(f"Noise σ = {NOISE_STD}  (added to inputs during denoising training)")
print()

# ── Train 1: Standard AE (clean → clean) ─────────────────────────────────────
print("Training standard autoencoder (clean input → clean target)...")
model_std, losses_std = train_ae(X_clean, X_clean, n_epochs=200)
print(f"  Final loss: {losses_std[-1]:.5f}")

# ── Train 2: Denoising AE (noisy → clean) ─────────────────────────────────────
print("Training denoising autoencoder (noisy input → clean target)...")
model_dae, losses_dae = train_ae(X_noisy, X_clean, n_epochs=200)
print(f"  Final loss: {losses_dae[-1]:.5f}")
print()

# ── Evaluate both on fresh noisy test inputs ──────────────────────────────────
def eval_model(model, X_clean, noise_std, n_eval=10):
    noisy_mse_list, denoised_mse_list = [], []
    for i in range(n_eval):
        xi = X_clean[i]
        xn = [xi[j]+random.gauss(0,noise_std) for j in range(p)]
        xh = predict(model, xn)
        noisy_mse_list.append(mse(xi, xn))
        denoised_mse_list.append(mse(xi, xh))
    return (sum(noisy_mse_list)/len(noisy_mse_list),
            sum(denoised_mse_list)/len(denoised_mse_list))

random.seed(77)
noisy_err, std_recon   = eval_model(model_std, X_clean, NOISE_STD)
_,          dae_recon  = eval_model(model_dae, X_clean, NOISE_STD)

print(f"Reconstruction error on held-out noisy inputs:")
print(f"  {'Method':<30}  {'MSE':>10}  Note")
print("  " + "─"*55)
print(f"  {'Noisy input (no model)':<30}  {noisy_err:>10.5f}  Baseline — no denoising")
print(f"  {'Standard AE on noisy input':<30}  {std_recon:>10.5f}  Not trained for noise")
print(f"  {'Denoising AE on noisy input':<30}  {dae_recon:>10.5f}  Trained to remove noise")
print()
improvement = (std_recon - dae_recon) / std_recon * 100
print(f"  Denoising AE reduces reconstruction error by {improvement:.1f}%")
print(f"  compared to standard AE on noisy inputs.")
print()
print("Intuition:")
print("  Standard AE trained on clean data has never seen noise — it")
print("  treats noise as signal and reconstructs the noisy input.")
print("  Denoising AE learned that the noise is irrelevant and projects")
print("  each input back onto the clean data manifold.")
""",
        "runnable": True,
    },

    "▶ Run: VAE — ELBO Loss and Reparameterisation": {
        "description": "Implement a Variational Autoencoder from scratch. Encoder outputs mean + log-variance, reparameterisation trick samples z, decoder reconstructs. Tracks reconstruction loss and KL divergence separately.",
        "code": """
import math
import random

random.seed(3)

# ── Utilities ─────────────────────────────────────────────────────────────────
def relu(x):   return [max(0.0,v) for v in x]
def relu_d(x): return [1.0 if v>0 else 0.0 for v in x]

def gauss(): # Box-Muller
    u=random.random() or 1e-9; v=random.random()
    return math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v)

def make_layer(in_d,out_d,scale=0.05):
    return {'W':[[random.gauss(0,scale) for _ in range(in_d)] for _ in range(out_d)],
            'b':[0.0]*out_d,'in_dim':in_d,'out_dim':out_d}

def fwd(layer,x):
    return [sum(layer['W'][i][j]*x[j] for j in range(len(x)))+layer['b'][i]
            for i in range(layer['out_dim'])]

def bwd(layer,x,g,lr):
    gi=[sum(layer['W'][j][i]*g[j] for j in range(layer['out_dim']))
        for i in range(layer['in_dim'])]
    for i in range(layer['out_dim']):
        for j in range(layer['in_dim']): layer['W'][i][j]-=lr*g[i]*x[j]
        layer['b'][i]-=lr*g[i]
    return gi

# ── Dataset: 4D, 2 clusters ───────────────────────────────────────────────────
def blob(cx,n,s=0.4):
    return [[cx[d]+random.gauss(0,s) for d in range(len(cx))] for _ in range(n)]
X_raw=(blob([2,2,0,0],40)+blob([-2,0,2,0],40)+blob([0,-2,-2,0],40))
n,p=len(X_raw),len(X_raw[0])
labels=[0]*40+[1]*40+[2]*40

# Normalise
for j in range(p):
    mn=min(X_raw[i][j] for i in range(n)); mx=max(X_raw[i][j] for i in range(n))
    r=mx-mn or 1.0
    for i in range(n): X_raw[i][j]=2*(X_raw[i][j]-mn)/r-1

LATENT=2; LR=0.008; EPOCHS=300; KL_WEIGHT=0.5

# Encoder: x → h → (mu, logvar)  [two separate heads]
enc1   = make_layer(p, 6)
enc_mu = make_layer(6, LATENT)
enc_lv = make_layer(6, LATENT)  # log-variance head
# Decoder: z → h → x
dec1   = make_layer(LATENT, 6)
dec2   = make_layer(6, p)

print(f"VAE  |  {p}D → 6 → ({LATENT}D μ, {LATENT}D logσ²) → {LATENT}D z → 6 → {p}D")
print(f"Loss = Reconstruction MSE + {KL_WEIGHT} × KL(N(μ,σ²) || N(0,I))")
print()
print(f"  {'Epoch':>6}  {'Recon Loss':>12}  {'KL Loss':>10}  {'Total':>10}")
print("  " + "─"*50)

for epoch in range(1,EPOCHS+1):
    idx=list(range(n)); random.shuffle(idx)
    ep_recon=0; ep_kl=0

    for i in idx:
        x=X_raw[i]

        # ── Encoder forward ───────────────────────────────────────────────────
        h_pre=fwd(enc1,x); h=relu(h_pre)
        mu     = fwd(enc_mu, h)
        logvar = fwd(enc_lv, h)
        sigma  = [math.exp(0.5*lv) for lv in logvar]

        # ── Reparameterisation: z = mu + sigma * epsilon ──────────────────────
        eps = [gauss() for _ in range(LATENT)]
        z   = [mu[k] + sigma[k]*eps[k] for k in range(LATENT)]

        # ── Decoder forward ───────────────────────────────────────────────────
        dh_pre=fwd(dec1,z); dh=relu(dh_pre)
        xhat = fwd(dec2, dh)

        # ── ELBO loss ─────────────────────────────────────────────────────────
        # Reconstruction: MSE
        recon_loss = sum((x[j]-xhat[j])**2 for j in range(p))/p

        # KL divergence: KL(N(mu,sigma^2) || N(0,1))
        # = 0.5 * sum(mu^2 + sigma^2 - logvar - 1)
        kl_loss = 0.5*sum(mu[k]**2+sigma[k]**2-logvar[k]-1.0 for k in range(LATENT))

        total_loss = recon_loss + KL_WEIGHT * kl_loss
        ep_recon += recon_loss; ep_kl += kl_loss

        # ── Backward through decoder ──────────────────────────────────────────
        dL_dxhat=[2*(xhat[j]-x[j])/p for j in range(p)]
        g=bwd(dec2,dh,dL_dxhat,LR)
        g=bwd(dec1,z,[g[k]*relu_d(dh_pre)[k] for k in range(len(dh_pre))],LR)

        # ── Gradient through reparameterisation to mu and logvar ──────────────
        # dL/dz = g (from decoder)
        # dL/dmu     = dL/dz + KL_WEIGHT * mu      (KL grad w.r.t. mu)
        # dL/dlogvar = dL/dz * sigma/2 * eps + KL_WEIGHT * 0.5*(sigma^2 - 1)
        dL_dmu  = [g[k] + KL_WEIGHT*mu[k] for k in range(LATENT)]
        dL_dlv  = [g[k]*sigma[k]*0.5*eps[k] + KL_WEIGHT*0.5*(sigma[k]**2-1)
                   for k in range(LATENT)]

        # ── Backward through encoder heads ────────────────────────────────────
        g_mu = bwd(enc_mu, h, dL_dmu, LR)
        g_lv = bwd(enc_lv, h, dL_dlv, LR)

        # Combine gradients into shared encoder body
        g_h = [g_mu[k]+g_lv[k] for k in range(len(g_mu))]
        bwd(enc1, x, [g_h[k]*relu_d(h_pre)[k] for k in range(len(h_pre))], LR)

    if epoch in (1,50,100,150,200,250,300):
        print(f"  {epoch:>6}  {ep_recon/n:>12.5f}  {ep_kl/n:>10.5f}  {(ep_recon/n+KL_WEIGHT*ep_kl/n):>10.5f}")

# ── Check latent space ────────────────────────────────────────────────────────
print()
print("Latent space (mu values per class):")
class_mus = {0:[], 1:[], 2:[]}
for i in range(n):
    h=relu(fwd(enc1,X_raw[i]))
    mu=fwd(enc_mu,h)
    class_mus[labels[i]].append(mu)

for c, mus in class_mus.items():
    mean_mu=[sum(m[d] for m in mus)/len(mus) for d in range(LATENT)]
    print(f"  Class {c}: mean_mu = [{', '.join(f'{v:.3f}' for v in mean_mu)}]")

print()
print("If classes have different mean_mu values → encoder separates them.")
print("If mean_mu ≈ [0,0] for all → KL term collapsed the latent (posterior collapse).")
print()
print("KL term forces latent means toward 0 and variances toward 1 (prior N(0,I)).")
print("Trade-off: too high KL_WEIGHT → posterior collapse. Too low → no regularisation.")
""",
        "runnable": True,
    },

    "▶ Run: Anomaly Detection via Reconstruction Error": {
        "description": "Train an autoencoder on normal data only. Then compute reconstruction error for normal and anomalous test samples. Shows how high reconstruction error flags anomalies.",
        "code": """
import math
import random

random.seed(42)

# ── Normal data: 5D, tight cluster ───────────────────────────────────────────
def blob(cx, n, s):
    return [[cx[d]+random.gauss(0,s) for d in range(len(cx))] for _ in range(n)]

X_normal = blob([1,1,1,1,1], 80, 0.3)  # normal training data
n, p = len(X_normal), len(X_normal[0])

# Normalise w.r.t. training data
for j in range(p):
    mn=min(X_normal[i][j] for i in range(n))
    mx=max(X_normal[i][j] for i in range(n))
    r=mx-mn or 1.0
    for i in range(n): X_normal[i][j]=(X_normal[i][j]-mn)/r

def relu(x):   return [max(0.0,v) for v in x]
def relu_d(x): return [1.0 if v>0 else 0.0 for v in x]
def make_layer(a,b,s=0.1):
    return {'W':[[random.gauss(0,s) for _ in range(a)] for _ in range(b)],'b':[0.0]*b,'in_dim':a,'out_dim':b}
def fwd(l,x):
    return [sum(l['W'][i][j]*x[j] for j in range(len(x)))+l['b'][i] for i in range(l['out_dim'])]
def bwd(l,x,g,lr):
    gi=[sum(l['W'][j][i]*g[j] for j in range(l['out_dim'])) for i in range(l['in_dim'])]
    for i in range(l['out_dim']):
        for j in range(l['in_dim']): l['W'][i][j]-=lr*g[i]*x[j]
        l['b'][i]-=lr*g[i]
    return gi

# ── Train on NORMAL data only ─────────────────────────────────────────────────
e1=make_layer(p,4); e2=make_layer(4,2)
d1=make_layer(2,4); d2=make_layer(4,p)

LR=0.015; EPOCHS=300
for epoch in range(1,EPOCHS+1):
    idx=list(range(n)); random.shuffle(idx)
    for i in idx:
        x=X_normal[i]
        h1_p=fwd(e1,x); h1=relu(h1_p)
        z_p =fwd(e2,h1); z=relu(z_p)
        h2_p=fwd(d1,z);  h2=relu(h2_p)
        xh  =fwd(d2,h2)
        g=bwd(d2,h2,[2*(xh[j]-x[j])/p for j in range(p)],LR)
        g=bwd(d1,z, [g[k]*relu_d(h2_p)[k] for k in range(len(h2_p))],LR)
        g=bwd(e2,h1,[g[k]*relu_d(z_p)[k]  for k in range(len(z_p))],LR)
        bwd(e1,x,  [g[k]*relu_d(h1_p)[k]  for k in range(len(h1_p))],LR)

def reconstruct_error(x):
    h1=relu(fwd(e1,x)); z=relu(fwd(e2,h1))
    h2=relu(fwd(d1,z));  xh=fwd(d2,h2)
    return sum((x[j]-xh[j])**2 for j in range(p))/p

# ── Create test sets ──────────────────────────────────────────────────────────
# Test normals: same distribution as training
X_test_normal = blob([1,1,1,1,1], 20, 0.3)
for i in range(len(X_test_normal)):
    for j in range(p):
        mn=min(X_normal[k][j] for k in range(n))
        mx=max(X_normal[k][j] for k in range(n))
        X_test_normal[i][j]=(X_test_normal[i][j]-mn)/(mx-mn or 1.0)

# Test anomalies: far from normal cluster
X_anomaly_near = blob([1.8,1.8,1.8,1.8,1.8], 10, 0.3)   # moderate anomaly
X_anomaly_far  = blob([5,5,5,5,5],             10, 0.3)   # extreme anomaly
for X_set in [X_anomaly_near, X_anomaly_far]:
    for i in range(len(X_set)):
        for j in range(p):
            mn=min(X_normal[k][j] for k in range(n))
            mx=max(X_normal[k][j] for k in range(n))
            X_set[i][j]=(X_set[i][j]-mn)/(mx-mn or 1.0)

normal_errs = [reconstruct_error(x) for x in X_test_normal]
near_errs   = [reconstruct_error(x) for x in X_anomaly_near]
far_errs    = [reconstruct_error(x) for x in X_anomaly_far]

mean_normal = sum(normal_errs)/len(normal_errs)
mean_near   = sum(near_errs)/len(near_errs)
mean_far    = sum(far_errs)/len(far_errs)

# Set threshold at mean + 3*std of normal errors
std_normal = math.sqrt(sum((e-mean_normal)**2 for e in normal_errs)/len(normal_errs))
threshold  = mean_normal + 3*std_normal

print("Anomaly Detection via Reconstruction Error")
print(f"Trained on {n} normal points  |  5D → 2D → 5D")
print()
print(f"  {'Category':<22}  {'Mean Error':>12}  {'Flagged':>8}  {'Bar'}")
print("  " + "─"*70)

def bar(val, max_val=0.5):
    return "█" * int(min(val/max_val,1.0)*30)

for name, errs in [("Normal test", normal_errs),
                   ("Near anomaly (1.8x)", near_errs),
                   ("Far anomaly (5x)", far_errs)]:
    flagged = sum(1 for e in errs if e > threshold)
    mean_e  = sum(errs)/len(errs)
    print(f"  {name:<22}  {mean_e:>12.5f}  {flagged:>5}/{len(errs)}  {bar(mean_e)}")

print()
print(f"  Threshold (μ + 3σ of normals): {threshold:.5f}")
print()

# Show point-by-point for 5 of each
print("  Sample errors (first 5 per class):")
print(f"  {'Normal':>10}  {'Near anomaly':>14}  {'Far anomaly':>12}  Threshold={threshold:.4f}")
print("  " + "─"*52)
for k in range(5):
    n_flag="*" if normal_errs[k]>threshold else " "
    a_flag="*" if near_errs[k]>threshold else " "
    f_flag="*" if far_errs[k]>threshold else " "
    print(f"  {normal_errs[k]:>9.5f}{n_flag}  {near_errs[k]:>13.5f}{a_flag}  {far_errs[k]:>11.5f}{f_flag}")
print("  (* = flagged as anomaly)")
""",
        "runnable": True,
    },

    "▶ Run: Autoencoder vs PCA — Nonlinear Compression": {
        "description": "Compare reconstruction quality of a nonlinear autoencoder vs linear PCA at the same bottleneck dimension. Shows autoencoders outperform PCA on curved, nonlinear data structures.",
        "code": """
import math
import random

random.seed(5)

# ── Dataset: data on a nonlinear curve (embedded in 6D) ──────────────────────
# True structure: 1D curve t → 6D via trig functions  (nonlinear manifold)
n, p = 80, 6
ts = [i/(n-1)*2*math.pi for i in range(n)]
X_raw = [[math.sin(t), math.cos(t), math.sin(2*t), math.cos(2*t),
          math.sin(t)*math.cos(t), t/math.pi - 1]
         for t in ts]
# Add small noise
X_raw = [[X_raw[i][j]+random.gauss(0,0.05) for j in range(p)] for i in range(n)]

# Normalise
for j in range(p):
    mn=min(X_raw[i][j] for i in range(n)); mx=max(X_raw[i][j] for i in range(n))
    r=mx-mn or 1.0
    for i in range(n): X_raw[i][j]=2*(X_raw[i][j]-mn)/r-1

def relu(x):   return [max(0.0,v) for v in x]
def relu_d(x): return [1.0 if v>0 else 0.0 for v in x]
def make_layer(a,b,s=0.1):
    return {'W':[[random.gauss(0,s) for _ in range(a)] for _ in range(b)],'b':[0.0]*b,'in_dim':a,'out_dim':b}
def fwd(l,x):
    return [sum(l['W'][i][j]*x[j] for j in range(len(x)))+l['b'][i] for i in range(l['out_dim'])]
def bwd(l,x,g,lr):
    gi=[sum(l['W'][j][i]*g[j] for j in range(l['out_dim'])) for i in range(l['in_dim'])]
    for i in range(l['out_dim']):
        for j in range(l['in_dim']): l['W'][i][j]-=lr*g[i]*x[j]
        l['b'][i]-=lr*g[i]
    return gi

# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder: 6 → 4 → 2 → 4 → 6
# ─────────────────────────────────────────────────────────────────────────────
LATENT=2; LR=0.01; EPOCHS=500
e1=make_layer(p,4); e2=make_layer(4,LATENT)
d1=make_layer(LATENT,4); d2=make_layer(4,p)

for epoch in range(1,EPOCHS+1):
    idx=list(range(n)); random.shuffle(idx)
    for i in idx:
        x=X_raw[i]
        h1_p=fwd(e1,x); h1=relu(h1_p)
        z_p =fwd(e2,h1); z=relu(z_p)
        h2_p=fwd(d1,z);  h2=relu(h2_p)
        xh  =fwd(d2,h2)
        g=bwd(d2,h2,[2*(xh[j]-x[j])/p for j in range(p)],LR)
        g=bwd(d1,z, [g[k]*relu_d(h2_p)[k] for k in range(len(h2_p))],LR)
        g=bwd(e2,h1,[g[k]*relu_d(z_p)[k]  for k in range(len(z_p))],LR)
        bwd(e1,x,  [g[k]*relu_d(h1_p)[k]  for k in range(len(h1_p))],LR)

ae_mse = sum(
    sum((X_raw[i][j] - fwd(d2,relu(fwd(d1,relu(fwd(e2,relu(fwd(e1,X_raw[i])))))))[j])**2
        for j in range(p)) / p
    for i in range(n)
) / n

# ─────────────────────────────────────────────────────────────────────────────
# PCA: 6D → 2D → 6D
# ─────────────────────────────────────────────────────────────────────────────
means=[sum(X_raw[i][j] for i in range(n))/n for j in range(p)]
Xc=[[X_raw[i][j]-means[j] for j in range(p)] for i in range(n)]
C=[[sum(Xc[i][j]*Xc[i][k] for i in range(n))/(n-1) for k in range(p)] for j in range(p)]

def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def power_iter(M,seed_v):
    v=seed_v[:]; nrm=math.sqrt(sum(x**2 for x in v)); v=[x/nrm for x in v]
    for _ in range(3000):
        w=mat_vec(M,v); nrm=math.sqrt(sum(x**2 for x in w))
        if nrm<1e-12: break
        v2=[x/nrm for x in w]
        if math.sqrt(sum((v2[k]-v[k])**2 for k in range(p)))<1e-10: v=v2; break
        v=v2
    lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam,0.0),v

epairs=[]; Cd=[row[:] for row in C]
seeds=[[float(k==j) for k in range(p)] for j in range(p)]
for s in seeds:
    lam,vec=power_iter(Cd,s); epairs.append((lam,vec))
    Cd=[[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]
epairs.sort(reverse=True,key=lambda x:x[0])
W_pca=[ep[1] for ep in epairs[:LATENT]]

def pca_recon(x_raw):
    xc=[x_raw[j]-means[j] for j in range(p)]
    z=[sum(xc[j]*W_pca[k][j] for j in range(p)) for k in range(LATENT)]
    rec=[sum(z[k]*W_pca[k][j] for k in range(LATENT))+means[j] for j in range(p)]
    return rec

pca_mse=sum(sum((X_raw[i][j]-pca_recon(X_raw[i])[j])**2 for j in range(p))/p
            for i in range(n))/n

total_var=sum(epairs[k][0] for k in range(p))
pca_var_kept=sum(epairs[k][0] for k in range(LATENT))/total_var*100

print(f"Autoencoder vs PCA  |  Bottleneck dim = {LATENT}  |  6D nonlinear manifold")
print(f"Data: 1D trig curve embedded in 6D — curved, nonlinear structure")
print()
print(f"  {'Method':<22}  {'Reconstruction MSE':>20}  {'Notes'}")
print("  " + "─"*65)
print(f"  {'PCA (linear, 2 PCs)':<22}  {pca_mse:>20.5f}  "
      f"{pca_var_kept:.1f}% variance captured")
print(f"  {'Autoencoder (nonlinear)':<22}  {ae_mse:>20.5f}  "
      f"Nonlinear mapping, {EPOCHS} epochs")
print()
improvement = (pca_mse-ae_mse)/pca_mse*100
if improvement > 0:
    print(f"  Autoencoder reduces MSE by {improvement:.1f}% vs PCA")
    print(f"  → Nonlinear compression captures the trig curve better than a flat projection.")
else:
    print(f"  PCA performs comparably (or better) here.")
    print(f"  → More AE training epochs or a wider network would improve results.")
print()
print("Key insight:")
print("  PCA can only find the best flat 2D plane through the 6D space.")
print("  The autoencoder can fold and curve its encoding to match the true")
print("  1D trig manifold — a fundamentally nonlinear structure that no")
print("  flat projection can represent faithfully.")
""",
        "runnable": True,
    },

}


# Dedent all operation code strings — they're indented inside the dict literal,
# so each line has ~20 leading spaces. textwrap.dedent removes the common indent,
# producing clean left-aligned code that runs without IndentationError.
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()

# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
def render_operations(st, scripts_dir=None, main_script=None):
    """Render all operations with code display and optional run buttons."""
    import streamlit as st  # local import so module stays importable without st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if scripts_dir is None:
        scripts_dir = None
    if main_script is None:
        main_script = None

    scripts_available = main_script.exists()

    if "tok_step_status"  not in st.session_state:
        st.session_state.tok_step_status  = {}
    if "tok_step_outputs" not in st.session_state:
        st.session_state.tok_step_outputs = {}

    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**{op_data['description']}**")
            st.markdown("---")
            st.code(op_data["code"], language=op_data.get("language", "python"))


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
# render_operations() has been removed.  app.py owns all Streamlit rendering
# via its own render_operation() helper and strips callables from topic dicts
# inside load_topics_for() anyway — so a local render function is never called.

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 400
    try:
        from Unsupervised_Learning.visuals.autoencoder_visual import (   # ← match your exact folder casing
            VISUAL_HTML,
            VISUAL_HEIGHT,
        )
        visual_html   = VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None, # COMPLEXITY,
        "operations":    OPERATIONS,
    }

