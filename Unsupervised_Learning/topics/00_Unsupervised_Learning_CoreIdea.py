"""Module 01 · Unsupervised Learning Core Idea"""
import textwrap
import re

DISPLAY_NAME = "01 · Unsupervised Learning Core Idea"
ICON         = "🔍"
SUBTITLE     = "The Fundamentals of Unsupervised Learning"

THEORY = """

## Unsupervised Learning in Machine Learning

**Formal Definition:**
Unsupervised learning is the task of learning the underlying structure, distribution, or organisation of data
without any labeled examples. Given only inputs x, the model must discover patterns, groupings, or compact
representations entirely on its own — there are no correct answers provided during training.

The fundamental idea behind unsupervised learning is finding hidden structure in unlabeled data.
Instead of learning a mapping from inputs to known outputs, the model must uncover the latent organisation
that exists within the data itself — clusters, patterns, densities, or lower-dimensional representations.

Think of it like an explorer dropped into an unknown city with no map and no guide.
They have to walk the streets, notice which buildings are near each other, which neighbourhoods feel similar,
and gradually build their own internal map of how the city is organised — entirely from observation,
without anyone telling them what they're looking at.

---

#### How It Works

The process generally follows these steps:

    1. Collect unlabeled data — You have inputs (features) but no associated labels or targets.
       For example, millions of customer purchase records with no pre-assigned segments.

    2. Define a structural objective — Instead of minimising error against known answers, the model
       optimises a structural objective: minimise within-cluster distance, maximise data likelihood,
       minimise reconstruction error, or preserve neighbourhood relationships.

    3. Discover structure — The algorithm iteratively reorganises its internal representation to better
       satisfy that objective. Patterns emerge not from supervision, but from the geometry and statistics
       of the data itself.

    4. Interpret and apply — The discovered structure (clusters, components, embeddings) is used for
       downstream tasks: visualisation, anomaly detection, compression, or as input to supervised models.


#### Why "Unsupervised"?

The term contrasts with supervised learning, where a human supervisor provides correct answers. In unsupervised
learning, there is no supervisor. The model receives no feedback signal telling it whether its output is right
or wrong in an absolute sense — it can only optimise relative to a structural criterion it defines for itself.

This is in contrast to:
    * Supervised learning — labeled data, learns a mapping from input to known output.
    * Reinforcement learning — no labels, but a reward signal guides behaviour over time.
    * Self-supervised learning — a hybrid where the model generates its own labels from structure in the data
      (e.g. predicting the next word in a sentence). This sits on the boundary between supervised and unsupervised.

---

#### Three Main Goals of Unsupervised Learning

Rather than the single goal of supervised learning (predict the correct label), unsupervised learning pursues
three distinct structural goals depending on the method used:

**Clustering** — Group similar data points together so that items within a group are more similar to each
other than to items in other groups. The output is a discrete assignment of each point to a cluster.

**Dimensionality Reduction** — Find a compact, lower-dimensional representation of the data that preserves
as much meaningful structure as possible. The output is a transformed version of the data in fewer dimensions.

**Density Estimation** — Learn the underlying probability distribution that generated the data. The output is
a model that can assign a likelihood to any data point — useful for anomaly detection and data generation.

---

#### Three Things to Keep Separate

Just as in supervised learning, it helps to keep three distinct layers separate:

**The learning paradigm** — Unsupervised learning is defined by the absence of labels and the goal of
discovering latent structure. This is the category of problem, not the method of solving it.

**The structural objective** — What the algorithm is actually optimising. K-Means minimises within-cluster
variance. PCA maximises explained variance. Autoencoders minimise reconstruction error. VAEs maximise the
evidence lower bound (ELBO). The objective defines what "good structure" means.

**The model class** — The type of function or representation used: centroid-based, probabilistic,
neural network-based, graph-based, and so on. Each makes different assumptions about what structure looks like.

Keeping these three layers separate prevents a very common confusion: thinking that unsupervised learning
means K-Means, or that dimensionality reduction and clustering are the same thing.

---

#### The Core Challenge: No Ground Truth

The deepest difference between supervised and unsupervised learning is that there is no objective way to
measure whether an unsupervised model is "correct."

In supervised learning, you can always ask: does the model's prediction match the label?
In unsupervised learning, there is no label to compare against.

This creates a fundamental evaluation problem. You can measure internal quality (e.g. how tight are the
clusters?), but that doesn't necessarily mean the structure is meaningful or useful for your actual task.

This is why unsupervised learning requires more domain knowledge to interpret, and why human judgement
plays a larger role in validating the output.

---

## Part 1 — Clustering

Clustering is the task of partitioning data into groups (clusters) such that points within the same cluster
are more similar to each other than to points in different clusters.

#### K-Means Clustering

K-Means is the most widely used clustering algorithm. It partitions N data points into K clusters by
iteratively assigning each point to its nearest cluster centre and then recomputing the centres.

**Objective — Inertia (Within-Cluster Sum of Squares):**

    Minimise: Σ_k Σ_{x ∈ C_k} ||x − μ_k||²

Where:
    K    = number of clusters (chosen by the user)
    C_k  = the set of points assigned to cluster k
    μ_k  = the centroid (mean) of cluster k
    ||·||= Euclidean distance

The goal is to find centroids μ_k and assignments such that this total within-cluster variance is as small
as possible.

**The Algorithm (Lloyd's Algorithm):**

    1. Initialise — Place K centroids randomly (or using K-Means++ for smarter initialisation)
    2. Assignment Step — Assign each point to the nearest centroid:
           cluster(x_i) = argmin_k ||x_i − μ_k||²
    3. Update Step — Recompute each centroid as the mean of all assigned points:
           μ_k = (1/|C_k|) Σ_{x ∈ C_k} x
    4. Repeat Steps 2–3 until assignments no longer change (convergence)

**Important properties:**
    * Guaranteed to converge (assignments can only improve or stay the same each step)
    * NOT guaranteed to find the global optimum — different initialisations give different results
    * Sensitive to the choice of K — you must decide K in advance
    * Assumes spherical, similarly-sized clusters — fails on elongated or irregular shapes
    * Sensitive to outliers — a single extreme point can pull a centroid far from the true cluster centre

---

#### Mathematical Walkthrough — K-Means Step by Step

Let's trace through K-Means by hand on a tiny dataset.

**Setup:**

We have 6 data points in 2D space (x₁, x₂) and want K = 2 clusters.

    Points:
        A = (1, 1)
        B = (1, 2)
        C = (2, 1)
        D = (8, 8)
        E = (8, 9)
        F = (9, 8)

The true structure here is obvious — A, B, C form one natural group and D, E, F form another.
K-Means should discover this from the data alone.

---

**Iteration 0 — Initialise Centroids:**

Let's say K-Means randomly picks two starting centroids:

    μ₁ = (1, 1)    (happens to land on point A)
    μ₂ = (8, 8)    (happens to land on point D)

---

**Iteration 1 — Assignment Step:**

For each point, compute the Euclidean distance to each centroid and assign to the nearest one.

    Euclidean distance formula:  d = √((x₁ − c₁)² + (x₂ − c₂)²)

    Point A = (1,1):
        d(A, μ₁) = √((1-1)² + (1-1)²) = 0.00
        d(A, μ₂) = √((1-8)² + (1-8)²) = √(49+49) = 9.90
        → Assign to Cluster 1  ✓

    Point B = (1,2):
        d(B, μ₁) = √((1-1)² + (2-1)²) = 1.00
        d(B, μ₂) = √((1-8)² + (2-8)²) = √(49+36) = 9.22
        → Assign to Cluster 1  ✓

    Point C = (2,1):
        d(C, μ₁) = √((2-1)² + (1-1)²) = 1.00
        d(C, μ₂) = √((2-8)² + (1-8)²) = √(36+49) = 9.22
        → Assign to Cluster 1  ✓

    Point D = (8,8):
        d(D, μ₁) = √((8-1)² + (8-1)²) = 9.90
        d(D, μ₂) = √((8-8)² + (8-8)²) = 0.00
        → Assign to Cluster 2  ✓

    Point E = (8,9):
        d(E, μ₁) = √((8-1)² + (9-1)²) = √(49+64) = 10.63
        d(E, μ₂) = √((8-8)² + (9-8)²) = 1.00
        → Assign to Cluster 2  ✓

    Point F = (9,8):
        d(F, μ₁) = √((9-1)² + (8-1)²) = √(64+49) = 10.63
        d(F, μ₂) = √((9-8)² + (8-8)²) = 1.00
        → Assign to Cluster 2  ✓

    Cluster 1: {A, B, C}
    Cluster 2: {D, E, F}

---

**Iteration 1 — Update Step:**

Recompute each centroid as the mean of its assigned points.

    μ₁ = mean of {(1,1), (1,2), (2,1)}
       = ((1+1+2)/3, (1+2+1)/3)
       = (4/3, 4/3)
       = (1.33, 1.33)

    μ₂ = mean of {(8,8), (8,9), (9,8)}
       = ((8+8+9)/3, (8+9+8)/3)
       = (25/3, 25/3)
       = (8.33, 8.33)

---

**Iteration 2 — Assignment Step (with updated centroids):**

Check if any point changes cluster.

    Point A = (1,1):
        d(A, μ₁) = √((1-1.33)² + (1-1.33)²) = √(0.11+0.11) = 0.47
        d(A, μ₂) = √((1-8.33)² + (1-8.33)²) = 10.37
        → Cluster 1 (no change)

    (All other assignments also remain the same — the clusters have stabilised.)

**Convergence.** Assignments did not change. K-Means has found the solution in 2 iterations.

**Final Inertia (Within-Cluster Sum of Squares):**

    Cluster 1 (μ₁ = 1.33, 1.33):
        d(A)² = 0.22,  d(B)² = 0.22 + 0.44 = 0.67,  d(C)² = 0.44 + 0.22 = 0.67
        Inertia₁ = 0.22 + 0.67 + 0.67 = 1.56  (approximately)

    Cluster 2 (μ₂ = 8.33, 8.33):
        d(D)² = 0.22,  d(E)² = 0.22 + 0.44 = 0.67,  d(F)² = 0.44 + 0.22 = 0.67
        Inertia₂ = 1.56  (symmetric)

    Total Inertia = 1.56 + 1.56 = 3.12

The model has recovered the true two-cluster structure from the data with no labels whatsoever.

---

#### How Do You Choose K? — The Elbow Method

Since K is a hyperparameter you set manually, you need a principled way to choose it.

The Elbow Method works as follows:
    1. Run K-Means for K = 1, 2, 3, ..., N
    2. Record the total inertia for each K
    3. Plot inertia vs. K and look for an "elbow" — the point where adding another cluster
       yields diminishing returns in inertia reduction

    K = 1   →   Inertia = very high (all points in one cluster)
    K = 2   →   Inertia = drops sharply (the natural split is found)
    K = 3   →   Inertia = smaller drop
    K = 4+  →   Inertia = tiny further drops (no more real structure to find)

The elbow at K = 2 signals the natural number of clusters in the data.

Alternative methods include the Silhouette Score (measuring how similar each point is to its own
cluster vs. other clusters) and the Gap Statistic (comparing inertia to a random baseline).

---

#### DBSCAN — Density-Based Clustering

K-Means struggles when clusters are non-spherical, have different densities, or when there are outliers.
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) addresses all three.

**The core idea:** A cluster is a region of high point density. Points in sparse regions are noise (outliers).

**Two hyperparameters:**
    * ε (epsilon) — the neighbourhood radius around each point
    * MinPts — the minimum number of points required to form a dense region

**Three types of points:**
    * Core point — has at least MinPts neighbours within radius ε
    * Border point — within radius ε of a core point, but fewer than MinPts neighbours itself
    * Noise point (outlier) — not within radius ε of any core point

**Algorithm:**
    1. For each unvisited point, find all points within radius ε
    2. If fewer than MinPts neighbours → mark as noise (tentatively)
    3. If MinPts or more neighbours → start a new cluster, recursively expand it
       by including all density-reachable points
    4. Border points are absorbed into whichever cluster they are density-reachable from
    5. Noise points remain unassigned

**Advantages over K-Means:**
    * Does not require specifying K in advance
    * Can find arbitrarily shaped clusters
    * Naturally identifies and ignores outliers
    * Handles clusters of varying density (with care)

**Weakness:** Performance degrades on datasets with highly variable density across clusters.

---

## Part 2 — Dimensionality Reduction

High-dimensional data is difficult to visualise, computationally expensive, and often contains redundant
information. Dimensionality reduction finds a compact representation that preserves the most meaningful
structure.

#### The Curse of Dimensionality

As the number of features (dimensions) grows, data becomes increasingly sparse.
In high dimensions, the concept of "distance" breaks down — all points become approximately equidistant
from each other, making clustering and similarity measures unreliable.

This is why reducing dimensionality is often a prerequisite for effective analysis and modelling.

---

#### Principal Component Analysis (PCA)

PCA is the most widely used dimensionality reduction technique. It finds the directions (principal components)
along which the data varies the most, and projects the data onto a lower-dimensional subspace defined by those directions.

**Objective — Maximise Explained Variance:**

    Find a set of orthogonal unit vectors (principal components) v₁, v₂, ..., vₖ
    such that projecting the data onto these directions preserves the maximum variance.

    Equivalently: minimise the reconstruction error when projecting to k dimensions.

**The Algorithm:**

    1. Centre the data — Subtract the mean from each feature so the data is centred at the origin.

           x_centred = x − mean(x)

    2. Compute the Covariance Matrix — Captures how each pair of features varies together.

           Cov = (1/N) × X^T × X       (where X is the centred data matrix, N is number of samples)

    3. Eigendecomposition — Find the eigenvectors and eigenvalues of the covariance matrix.

           Cov × v = λ × v

           Each eigenvector v is a principal component (a direction in feature space).
           Its eigenvalue λ tells you how much variance is explained by that direction.

    4. Rank and Select — Sort eigenvectors by eigenvalue (largest first). Select the top k components.

    5. Project — Multiply the centred data by the selected eigenvectors to get the reduced representation.

           X_reduced = X_centred × V_k     (where V_k is the matrix of top k eigenvectors)

---

#### Mathematical Walkthrough — PCA Step by Step

**Setup:**

We have 4 data points in 2D (x₁, x₂) and want to reduce to 1 dimension while keeping maximum variance.

    Data:
        P1 = (2, 4)
        P2 = (3, 5)
        P3 = (5, 7)
        P4 = (8, 9)

**Step 1 — Centre the Data:**

    mean_x1 = (2 + 3 + 5 + 8) / 4 = 18 / 4 = 4.5
    mean_x2 = (4 + 5 + 7 + 9) / 4 = 25 / 4 = 6.25

    Centred data:
        P1 = (2-4.5, 4-6.25)   = (-2.5, -2.25)
        P2 = (3-4.5, 5-6.25)   = (-1.5, -1.25)
        P3 = (5-4.5, 7-6.25)   = ( 0.5,  0.75)
        P4 = (8-4.5, 9-6.25)   = ( 3.5,  2.75)

**Step 2 — Compute the Covariance Matrix:**

    Var(x1)       = ((-2.5)² + (-1.5)² + (0.5)² + (3.5)²) / 4
                  = (6.25 + 2.25 + 0.25 + 12.25) / 4
                  = 21.00 / 4 = 5.25

    Var(x2)       = ((-2.25)² + (-1.25)² + (0.75)² + (2.75)²) / 4
                  = (5.0625 + 1.5625 + 0.5625 + 7.5625) / 4
                  = 14.75 / 4 = 3.6875

    Cov(x1, x2)   = ((-2.5×-2.25) + (-1.5×-1.25) + (0.5×0.75) + (3.5×2.75)) / 4
                  = (5.625 + 1.875 + 0.375 + 9.625) / 4
                  = 17.5 / 4 = 4.375

    Covariance Matrix:
        | 5.2500   4.375 |
        | 4.375   3.6875 |

**Step 3 — Eigendecomposition:**

For a 2×2 matrix, eigenvalues λ satisfy: det(Cov − λI) = 0

    (5.25 − λ)(3.6875 − λ) − (4.375)² = 0

    λ² − 8.9375λ + (5.25 × 3.6875 − 19.140625) = 0
    λ² − 8.9375λ + (19.359375 − 19.140625) = 0
    λ² − 8.9375λ + 0.21875 = 0

Using the quadratic formula:
    λ = (8.9375 ± √(8.9375² − 4 × 0.21875)) / 2
    λ = (8.9375 ± √(79.879 − 0.875)) / 2
    λ = (8.9375 ± √79.004) / 2
    λ = (8.9375 ± 8.8886) / 2

    λ₁ = (8.9375 + 8.8886) / 2 = 8.913   ← first principal component
    λ₂ = (8.9375 − 8.8886) / 2 = 0.024   ← second principal component

**Explained Variance:**

    Total variance = λ₁ + λ₂ = 8.913 + 0.024 = 8.937

    Explained by PC1 = 8.913 / 8.937 = 99.7%
    Explained by PC2 = 0.024 / 8.937 =  0.3%

PC1 alone captures 99.7% of all variance in this dataset. Reducing from 2D to 1D loses almost nothing.
This makes sense — the four points lie almost perfectly along a single diagonal line.

**First Principal Component (Eigenvector for λ₁ ≈ 8.913):**

Solving (Cov − λ₁I) × v = 0 gives approximately:

    v₁ ≈ (0.766, 0.643)    (normalised direction of maximum variance)

This vector points in the direction that the data varies the most — diagonally from bottom-left to top-right.

**Step 4 — Project the Data onto PC1:**

Each point's projection onto v₁ = (0.766, 0.643):

    z = x_centred · v₁    (dot product)

    P1: (-2.5 × 0.766) + (-2.25 × 0.643) = -1.915 − 1.447 = -3.362
    P2: (-1.5 × 0.766) + (-1.25 × 0.643) = -1.149 − 0.804 = -1.953
    P3: ( 0.5 × 0.766) + ( 0.75 × 0.643) =  0.383 + 0.482 =  0.865
    P4: ( 3.5 × 0.766) + ( 2.75 × 0.643) =  2.681 + 1.768 =  4.449

    1D representation:   [-3.362,  -1.953,  0.865,  4.449]

The original 2D data has been compressed into 4 scalar values. The ordering and relative spacing of
the points is faithfully preserved. PCA found the direction that tells the most coherent "story" about
the data.

---

#### t-SNE — Visualising High-Dimensional Data

PCA is a linear method — it finds straight-line directions. But sometimes the meaningful structure
in high-dimensional data is non-linear. t-SNE (t-distributed Stochastic Neighbour Embedding) is
designed specifically for visualisation, projecting high-dimensional data into 2D or 3D while
preserving local neighbourhood structure.

**The core idea:**
    * In high-dimensional space, compute the probability that two points are "neighbours"
      (based on a Gaussian distribution over distances)
    * In the low-dimensional embedding, define a similar probability using a heavier-tailed
      t-distribution (this prevents the "crowding problem" where everything collapses to the centre)
    * Minimise the KL divergence between the two probability distributions:

          KL(P || Q) = Σ_{i≠j} p_{ij} × log(p_{ij} / q_{ij})

    * The 2D positions are optimised via gradient descent until nearby points in high-dimensional
      space are also nearby in the 2D projection.

**Key properties:**
    * Excellent for visualisation — reveals cluster structure that PCA misses
    * Non-deterministic — different runs produce different layouts (though cluster structure is consistent)
    * Not suitable for downstream feature extraction — the embedding space has no linear meaning
    * Computationally expensive — O(N²) naive, O(N log N) with Barnes-Hut approximation
    * The perplexity hyperparameter controls the effective number of neighbours (typically 5–50)

**When to use t-SNE vs PCA:**
    Use PCA when you need a faithful linear compression, when you want to retain global structure,
    or when you're using the reduced representation as input to another model.
    Use t-SNE when you want to visualise cluster structure and are willing to sacrifice global geometry.

---

#### UMAP — Uniform Manifold Approximation and Projection

UMAP is a newer dimensionality reduction technique that has largely superseded t-SNE for many use cases.
Like t-SNE it preserves local structure, but it also better preserves global structure, is significantly
faster, and produces a more stable embedding that can be re-applied to new data.

**Core idea:** Model the high-dimensional data as a topological manifold and find a low-dimensional
representation that is as topologically equivalent as possible. It uses fuzzy simplicial sets to represent
local connectivity, then minimises the cross-entropy between the high- and low-dimensional representations.

**Practical advantages over t-SNE:**
    * Scales much better to large datasets
    * Preserves global cluster relationships (not just local)
    * Deterministic and reusable — can transform new points into the existing embedding
    * Controlled by two main hyperparameters: n_neighbors (local vs. global balance) and
      min_dist (how tightly points are packed in the embedding)

---

## Part 3 — Generative Models and Density Estimation

A third class of unsupervised methods aims not just to organise data but to learn the underlying
probability distribution that generated it. Such models can assign likelihoods to new data points
(anomaly detection) and generate entirely new samples that look like the training data.

#### Autoencoders

An autoencoder is a neural network trained to reproduce its own input. It consists of two parts:

    Encoder: x → z     (compresses input to a bottleneck representation z — the latent code)
    Decoder: z → x̂    (reconstructs the input from the latent code)

**Objective — Minimise Reconstruction Error:**

    Loss = ||x − x̂||²    (for MSE reconstruction loss)

Because the bottleneck z has far fewer dimensions than x, the network is forced to learn the most
essential features of the data in order to reconstruct it accurately. The encoder learns a compressed
representation; the decoder learns to reverse it.

**What the bottleneck does:**
    * Compression — Forces the network to learn what matters and discard noise
    * Feature extraction — The latent code z is a learned representation useful for downstream tasks
    * Anomaly detection — If a new input reconstructs poorly (high loss), it is likely anomalous

**Autoencoder forward pass example:**

    Input:    x = image of the digit "7" (e.g. 784 pixel values in a 28×28 image)
    Encoder:  784 → 256 → 64 → 16 dimensions  (z: 16-dimensional latent code)
    Decoder:  16 → 64 → 256 → 784 dimensions  (x̂: reconstructed image)
    Loss:     MSE between x and x̂

The network learns to represent the entire image in just 16 numbers and reconstruct it from them.
A normal digit will reconstruct cleanly. An unusual pattern will reconstruct poorly — flagging it as anomalous.

---

#### Variational Autoencoders (VAE)

A VAE extends the standard autoencoder in a critical way: instead of encoding each input to a single
point in latent space, it encodes it to a probability distribution — specifically, a Gaussian with
a learned mean μ and standard deviation σ.

    Encoder outputs: μ(x) and σ(x)   (parameters of a distribution over z)
    Latent sample:   z ~ N(μ(x), σ(x))   (sample from that distribution)
    Decoder:         z → x̂

**Objective — Evidence Lower Bound (ELBO):**

    Loss = Reconstruction Loss + KL Divergence

    Reconstruction Loss = E[log p(x|z)]   (how well does the decoder reproduce x from z?)
    KL Divergence       = KL(q(z|x) || p(z))   (how close is the learned distribution to a standard normal?)

    ELBO = Reconstruction Loss − KL Divergence    (maximised during training)

The KL divergence term acts as a regulariser — it forces the latent space to be smooth and continuous
rather than a collection of isolated points. This is what makes VAEs generative:
you can sample z ~ N(0,1) and decode it to get a plausible new data point.

**VAE vs standard autoencoder:**
    Standard autoencoder — learns a compressed representation; cannot generate new data
    VAE — learns a continuous, structured latent space; can sample and generate new data

---

#### Gaussian Mixture Models (GMM)

A Gaussian Mixture Model is a probabilistic clustering method that assumes the data was generated
by a mixture of K Gaussian distributions, each with its own mean and covariance.

Unlike K-Means (which gives hard, discrete cluster assignments), GMM gives soft probabilities:

    P(cluster k | x_i) = probability that point x_i belongs to cluster k

**Objective — Maximise Data Likelihood:**

    L(θ) = Σ_i log [ Σ_k π_k × N(x_i | μ_k, Σ_k) ]

Where:
    π_k = mixing coefficient (how large cluster k is, Σπ_k = 1)
    μ_k = mean of Gaussian k
    Σ_k = covariance matrix of Gaussian k

**Fitted using the EM Algorithm (Expectation-Maximisation):**

    E-Step (Expectation): Given current parameters, compute the probability that each point
    belongs to each cluster:
        r_{ik} = π_k × N(x_i | μ_k, Σ_k) / Σ_j [π_j × N(x_i | μ_j, Σ_j)]

    M-Step (Maximisation): Update parameters to maximise the expected log-likelihood
    given those responsibilities:
        π_k_new  = (1/N) Σ_i r_{ik}
        μ_k_new  = Σ_i r_{ik} × x_i / Σ_i r_{ik}
        Σ_k_new  = Σ_i r_{ik} × (x_i − μ_k)(x_i − μ_k)^T / Σ_i r_{ik}

    Repeat until log-likelihood converges.

**Advantages over K-Means:**
    * Soft assignments — each point gets a probability for each cluster, not a binary assignment
    * Captures elliptical cluster shapes via the full covariance matrix
    * Provides a complete probabilistic model — you can compute exact likelihoods
    * Can detect anomalies: points with very low likelihood under all components are outliers

---

## Part 4 — Anomaly Detection

Anomaly (or outlier) detection is one of the most important practical applications of unsupervised
learning. The task: given data where the vast majority of points are normal, identify the rare
points that are anomalous — without ever having labeled examples of what "anomalous" looks like.

**Why unsupervised?**
Because anomalies are often rare, novel, and unpredictable. In fraud detection, a new attack pattern
has never been seen before. In industrial monitoring, a novel failure mode has no labeled training data.
You can't use supervised learning when you don't have examples of the thing you're trying to find.

**Common approaches:**

**Isolation Forest** — Randomly partitions the data using decision tree splits. Normal points, which
exist in dense regions, require many splits to isolate. Anomalous points, which exist in sparse
regions, are isolated in very few splits. The anomaly score is inversely proportional to the
average number of splits required.

**One-Class SVM** — Learns a boundary around the normal data. Points outside the boundary are anomalous.

**Autoencoder-based** — Train an autoencoder on normal data only. At inference time, anomalous data
will reconstruct poorly (high MSE loss) because the autoencoder has never learned to represent it.
The reconstruction error becomes the anomaly score.

**Statistical methods (Z-score, IQR)** — Assume a distribution (e.g. Gaussian) and flag points
that fall more than N standard deviations from the mean. Simple but effective for univariate data.

---

#### The Bias-Variance Equivalent in Unsupervised Learning

Unsupervised learning has its own version of the underfitting/overfitting tension:

    Too few clusters (K too small)    →   Underfitting the structure
                                           Distinct groups merged into one
                                           Model misses real patterns

    Too many clusters (K too large)   →   Overfitting the noise
                                           Every outlier gets its own cluster
                                           Model finds "structure" that isn't there

    Optimal K                         →   Recovers natural structure in the data
                                           Clusters correspond to real meaningful groups

For dimensionality reduction:
    Too many dimensions retained   →   Noise preserved, no real compression
    Too few dimensions retained    →   Real structure is destroyed
    Optimal                        →   Maximum information, minimum noise

This tension is managed through techniques like the elbow method, silhouette scores, validation
on downstream tasks, and domain knowledge about what the data should contain.

---

#### Evaluating Unsupervised Models — The Hard Problem

Because there are no labels, standard evaluation metrics like accuracy are unavailable.
Practitioners use a combination of internal and external measures:

**Internal measures (no labels required):**
    * Inertia / Within-Cluster Sum of Squares — lower is better (K-Means)
    * Silhouette Score — ranges from -1 to 1; higher means points are well-matched to their own cluster
      and well-separated from others
    * Davies-Bouldin Index — ratio of within-cluster scatter to between-cluster separation; lower is better
    * Calinski-Harabasz Index (Variance Ratio) — higher means denser, more separated clusters

**External measures (require ground truth labels, used for benchmarking):**
    * Adjusted Rand Index (ARI) — compares clustering to true labels; 1.0 = perfect, 0 = random
    * Normalised Mutual Information (NMI) — measures mutual information between clusters and true labels
    * Homogeneity, Completeness, V-Measure — decompose clustering quality into different aspects

**Downstream task evaluation (the most practical measure):**
    Train a simple supervised model on the learned representations and measure its accuracy.
    If the unsupervised features are capturing meaningful structure, downstream performance will be high.
    This is the gold standard for evaluating representation learning and dimensionality reduction.

---

#### Unsupervised Learning in Practice — The Pipeline

The practical pipeline for an unsupervised task differs meaningfully from supervised learning:

**1. Collect and clean data** — More important here than in supervised learning, because the model
   has no labels to correct for data quality issues. Outliers and noise directly shape what
   "structure" the model finds.

**2. Preprocess carefully** — Normalise/standardise features. Unsupervised methods based on
   distance (K-Means, DBSCAN, t-SNE) are highly sensitive to feature scale. A feature measured
   in thousands will dominate one measured in units unless scaled.

**3. Choose method based on goal:**
       Clustering needed?        → K-Means, DBSCAN, GMM, Hierarchical
       Visualisation needed?     → t-SNE, UMAP, PCA (for linear)
       Compression needed?       → PCA, Autoencoder
       Generation needed?        → VAE, GAN
       Anomaly detection needed? → Isolation Forest, Autoencoder, One-Class SVM

**4. Tune hyperparameters** — K (number of clusters), ε and MinPts (DBSCAN), number of components
   (PCA), perplexity (t-SNE), n_neighbors (UMAP), bottleneck size (Autoencoder).

**5. Evaluate internally** — Use silhouette scores, inertia curves, or reconstruction error.

**6. Validate externally** — Visualise, apply domain knowledge, or test on a downstream supervised
   task to confirm the discovered structure is meaningful.

**7. Iterate** — Unlike supervised learning, convergence of the algorithm does not mean the solution
   is correct. Human judgment is essential.

---

#### The Relationship Between Supervised and Unsupervised Learning

These two paradigms are not isolated — they frequently work together in practice.

**Pre-training** — An autoencoder or PCA is used to learn a good representation of unlabeled data.
That representation is then used as input features for a downstream supervised model, dramatically
reducing the amount of labeled data needed.

**Semi-supervised learning** — A small amount of labeled data combined with a large amount of
unlabeled data. The unsupervised component helps the supervised model generalise better.

**Clustering as feature engineering** — K-Means cluster assignments can be used as a new categorical
feature in a supervised model, adding structure the model might not find on its own.

**Self-supervised learning** — The model creates its own supervisory signal from unlabeled data
(e.g. predict the next word, predict the missing patch in an image). This is how large language
models and vision transformers are pre-trained. It sits on the boundary of both paradigms.

---

## Unsupervised Learning Algorithms — Comprehensive List

**Clustering Algorithms**
    * K-Means (Lloyd's Algorithm)
    * K-Means++ (smarter initialisation)
    * K-Medoids (PAM — Partitioning Around Medoids)
    * Mini-Batch K-Means
    * DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    * HDBSCAN (Hierarchical DBSCAN)
    * OPTICS (Ordering Points To Identify the Clustering Structure)
    * Mean Shift
    * Agglomerative Hierarchical Clustering (Single, Complete, Average, Ward linkage)
    * Divisive Hierarchical Clustering
    * BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
    * CURE (Clustering Using Representatives)
    * Spectral Clustering
    * Affinity Propagation
    * Gaussian Mixture Models (GMM) — via EM
    * Fuzzy C-Means (soft cluster assignments)
    * Self-Organising Maps (SOM)
    * CLARANS (Clustering Large Applications based on RANdomized Search)

**Dimensionality Reduction — Linear**
    * Principal Component Analysis (PCA)
    * Kernel PCA (non-linear extension via kernel trick)
    * Truncated SVD (Singular Value Decomposition)
    * Linear Discriminant Analysis (LDA) — technically semi-supervised
    * Factor Analysis
    * Independent Component Analysis (ICA)
    * Non-negative Matrix Factorisation (NMF)
    * Sparse PCA
    * Random Projections (Johnson-Lindenstrauss)

**Dimensionality Reduction — Non-Linear (Manifold Learning)**
    * t-SNE (t-distributed Stochastic Neighbour Embedding)
    * UMAP (Uniform Manifold Approximation and Projection)
    * Isomap (Isometric Mapping)
    * Locally Linear Embedding (LLE)
    * Laplacian Eigenmaps
    * Diffusion Maps
    * Multidimensional Scaling (MDS)
    * Hessian LLE
    * Modified LLE (MLLE)
    * LTSA (Local Tangent Space Alignment)

**Generative / Density Estimation Models**
    * Gaussian Mixture Models (GMM)
    * Kernel Density Estimation (KDE)
    * Variational Autoencoders (VAE)
    * Generative Adversarial Networks (GAN)
    * Normalising Flows
    * Energy-Based Models
    * Restricted Boltzmann Machines (RBM)
    * Deep Belief Networks (DBN)
    * PixelCNN / PixelRNN (autoregressive generation)
    * Diffusion Models (DDPM)

**Autoencoders**
    * Vanilla Autoencoder
    * Sparse Autoencoder
    * Denoising Autoencoder
    * Contractive Autoencoder
    * Variational Autoencoder (VAE)
    * Convolutional Autoencoder
    * Recurrent Autoencoder (LSTM-based)
    * Vector Quantised VAE (VQ-VAE)

**Anomaly / Outlier Detection**
    * Isolation Forest
    * One-Class SVM
    * Local Outlier Factor (LOF)
    * Elliptic Envelope (Minimum Covariance Determinant)
    * Autoencoder Reconstruction Error
    * COPOD (Copula-Based Outlier Detection)
    * LODA (Lightweight Online Detector of Anomalies)
    * Statistical methods: Z-score, Grubbs Test, IQR method

**Graph / Relational Methods**
    * Graph Clustering (Community Detection)
    * Spectral Clustering
    * Louvain Method (modularity optimisation)
    * Leiden Algorithm
    * Label Propagation
    * Node2Vec / DeepWalk (graph embeddings)
    * Markov Clustering (MCL)

**Representation / Embedding Learning**
    * Word2Vec (Skip-Gram, CBOW)
    * GloVe (Global Vectors for Word Representation)
    * FastText
    * Doc2Vec
    * BERT (masked language modelling — self-supervised)
    * SimCLR / MoCo (contrastive learning)
    * DINO (self-distillation for vision)

---

Quick Summary by Goal

    Goal                        │  Common Algorithms
    ────────────────────────────┼──────────────────────────────────────────────────
    Group similar points        │  K-Means, DBSCAN, GMM, Hierarchical, Spectral
    Find structure in noise     │  HDBSCAN, OPTICS, Mean Shift
    Reduce dimensions linearly  │  PCA, SVD, ICA, NMF, Factor Analysis
    Visualise high-dim data     │  t-SNE, UMAP, MDS, Isomap
    Compress and reconstruct    │  Autoencoder, VAE, PCA
    Generate new data           │  VAE, GAN, Diffusion Models, Normalising Flows
    Detect anomalies            │  Isolation Forest, LOF, One-Class SVM, Autoencoders
    Learn word/node embeddings  │  Word2Vec, GloVe, Node2Vec, BERT
    ────────────────────────────┼──────────────────────────────────────────────────

"""

OPERATIONS = {
}

VISUAL_HTML = ""  # Add your HTML visual breakdown here


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
        main_script = None # _MAIN_SCRIPT

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
        from Unsupervised_Learning.visuals.unsupervised_learning_visual import (   # ← match your exact folder casing
            UL_VISUAL_HTML,
            UL_VISUAL_HEIGHT,
        )
        visual_html   = UL_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = UL_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[09_neural_networks.py] Could not load visual: {e}", stacklevel=2)

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

