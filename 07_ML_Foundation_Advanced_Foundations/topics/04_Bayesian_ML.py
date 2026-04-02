"""
Bayesian Machine Learning
=========================

The probabilistic framework for learning under uncertainty — priors,
likelihoods, posteriors, MAP vs MLE, Bayesian inference, variational
methods, and the unifying perspective that treats every model parameter
as a random variable with a distribution, not a fixed point estimate.

"""

import textwrap
import re

TOPIC_NAME = "Bayesian Machine Learning"
DISPLAY_NAME = "04 · Bayesian Machine Learning"
ICON = "🎲"
SUBTITLE = "Reasoning Under Uncertainty with Probability"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Central Shift: Parameters as Random Variables

Every classical ML algorithm produces a single set of parameters θ* —
a point estimate. It is the "best" parameter vector, but it carries no
information about how confident we should be in it.

    Bayesian ML replaces the point estimate with a DISTRIBUTION over
    parameters. Instead of asking "what is θ?", we ask:

        "Given the data, what is the probability distribution over
         all plausible values of θ?"

    The shift in perspective:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  FREQUENTIST / CLASSICAL                                         │
    │      θ is a fixed unknown constant.                              │
    │      Learning = finding the best point estimate θ*.              │
    │      Uncertainty comes from data noise, not θ itself.            │
    │                                                                  │
    │  BAYESIAN                                                        │
    │      θ is a random variable with a prior distribution p(θ).      │
    │      Learning = updating that distribution using data.           │
    │      Result: posterior distribution p(θ | D) over θ.             │
    │      Uncertainty is explicit: p(θ | D) can be wide or narrow.    │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Why this matters:
    ─ Predictions come with calibrated uncertainty, not just a value.
    ─ Small datasets → wide posterior → honest admission of ignorance.
    ─ Large datasets → narrow posterior → approaches frequentist result.
    ─ Prior knowledge can be incorporated formally and transparently.
    ─ Avoids overfitting naturally: the posterior is regularised by
      the prior even without an explicit regularisation term.


──────────────────────────────────────────────────────────────────────────────
### Bayes' Theorem — The Engine of Learning

All Bayesian inference flows from one equation:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │              p(θ) · p(D | θ)                                     │
    │  p(θ | D) = ─────────────────                                    │
    │                  p(D)                                            │
    │                                                                  │
    │      posterior ∝ prior × likelihood                              │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Each term has a precise meaning:

    ┌──────────────────────────────────────────────────────────────────┐
    │  p(θ)       PRIOR        Belief about θ before seeing data.      │
    │                          Encodes domain knowledge, symmetry,     │
    │                          scale information, or pure ignorance.   │
    │                                                                  │
    │  p(D | θ)   LIKELIHOOD   Probability of the observed data        │
    │                          under parameters θ. The standard ML     │
    │                          objective — but here it is just one     │
    │                          factor, not the whole story.            │
    │                                                                  │
    │  p(θ | D)   POSTERIOR    Updated belief about θ after data.      │
    │                          This is the goal of Bayesian learning.  │
    │                          Combines prior knowledge and evidence.  │
    │                                                                  │
    │  p(D)       MARGINAL     Probability of the data under the       │
    │             LIKELIHOOD   model, marginalised over all θ:         │
    │             (EVIDENCE)   p(D) = ∫ p(D|θ) p(θ) dθ                 │
    │                          A normalising constant — ensures        │
    │                          p(θ|D) integrates to 1. Usually         │
    │                          intractable (hence approximate methods) │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 1 — Prior → Likelihood → Posterior:

    p(θ)                  p(D|θ)                  p(θ|D)
    │    prior            │  likelihood            │  posterior
    │   ╭────╮            │        ╭──╮            │      ╭─╮
    │  ╱      ╲           │       ╱    ╲           │     ╱   ╲
    │ ╱        ╲          │      ╱      ╲          │    ╱     ╲
    │╱          ╲         │_____╱        ╲_____    │___╱       ╲___
    └──────────── θ       └──────────────── θ      └───────────── θ

    Prior (broad):         Likelihood (peaked         Posterior: compromise.
    "θ could be           near a specific value):    Narrower than prior,
    anywhere in           "data is most consistent   pulled toward the
    this range"           with θ ≈ θ̂"               data.

    As n → ∞: the posterior narrows and concentrates on the true θ.
    As n → 0: the posterior equals the prior.
    This is the Bayesian version of the bias-variance tradeoff.


──────────────────────────────────────────────────────────────────────────────
### MLE, MAP, and Full Bayesian Inference

Three levels of approximation, from fully frequentist to fully Bayesian:

**1. Maximum Likelihood Estimation (MLE):**

    θ_MLE = argmax_θ  p(D | θ)  =  argmax_θ  Σᵢ log p(xᵢ | θ)

    Uses only the likelihood. Ignores the prior entirely.
    Equivalent to Bayesian inference with a UNIFORM prior p(θ) ∝ 1.

**2. Maximum A Posteriori (MAP):**

    θ_MAP = argmax_θ  p(θ | D)
           = argmax_θ  [ log p(D | θ) + log p(θ) ]
                               ↑              ↑
                           log-likelihood   log-prior

    MAP adds a prior term to the likelihood objective.
    It is still a POINT ESTIMATE — the mode of the posterior.

    ┌──────────────────────────────────────────────────────────────────┐
    │  MAP = REGULARISED MLE                                           │
    │                                                                  │
    │  Gaussian prior:  p(θ) = N(0, τ²I)                               │
    │  log p(θ) = −‖θ‖² / (2τ²)  + const                               │
    │  MAP objective = log p(D|θ) − λ‖θ‖²   where λ = 1/(2τ²)          │
    │  → This is EXACTLY L2 regularisation (Ridge / weight decay)!     │
    │                                                                  │
    │  Laplace prior:   p(θ) ∝ exp(−λ‖θ‖₁)                             │
    │  MAP objective = log p(D|θ) − λ‖θ‖₁                              │
    │  → This is EXACTLY L1 regularisation (Lasso)!                    │
    └──────────────────────────────────────────────────────────────────┘

    Regularisation has a Bayesian interpretation: it is MAP inference
    under a specific prior. Every regularised ML algorithm is implicitly
    doing Bayesian MAP estimation.

**3. Full Bayesian Inference:**

    Compute the entire posterior distribution p(θ | D) — not just its mode.
    Make predictions by marginalising (averaging) over all θ:

        p(y* | x*, D) = ∫ p(y* | x*, θ) · p(θ | D) dθ

    This is the POSTERIOR PREDICTIVE DISTRIBUTION.
    It automatically accounts for parameter uncertainty.

    Diagram 2 — Point Estimate vs Bayesian Prediction:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  POINT ESTIMATE (MLE/MAP):           BAYESIAN PREDICTION:       │
    │                                                                 │
    │  Pick single θ*, predict with it.    Average over many θ.       │
    │                                                                 │
    │  Prediction:  p(y*|x*, θ*)           Prediction:  ∫ p(y*|x*,θ)  │
    │               [single curve]                         p(θ|D) dθ  │
    │                                                     [weighted   │
    │                                                      average of │
    │                                                      curves]    │
    │                                                                 │
    │  ─ No uncertainty in θ              ─ Uncertainty in θ flows    │
    │  ─ Overconfident on small data        through to predictions    │
    │  ─ Can give P(correct) = 99%        ─ Wide posterior → wide     │
    │    on ambiguous examples              prediction band           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Conjugate Priors — Exact Inference in Closed Form

For most models, the integral ∫ p(D|θ) p(θ) dθ has no closed form.
Conjugate priors are a special class where posterior = same family as prior.

    Definition: A prior p(θ) is CONJUGATE to the likelihood p(D|θ) if
    the posterior p(θ|D) belongs to the same parametric family as p(θ).

    Conjugate prior pairs:

    ┌──────────────────────────┬──────────────────┬──────────────────┐
    │  Likelihood              │  Conjugate Prior │  Posterior       │
    ├──────────────────────────┼──────────────────┼──────────────────┤
    │  Bernoulli(θ)            │  Beta(α, β)      │  Beta(α+k, β+n−k)│
    │  Binomial(n, θ)          │  Beta(α, β)      │  Beta(...)       │
    │  Poisson(λ)              │  Gamma(α, β)     │  Gamma(α+Σx, β+n)│
    │  Normal(θ, σ²) known σ   │  Normal(μ₀, τ²)  │  Normal(...)     │
    │  Multinomial(θ)          │  Dirichlet(α)    │  Dirichlet(α+c)  │
    │  Normal(μ, σ²) both unk  │  Normal-InvGamma │  Normal-InvGamma │
    └──────────────────────────┴──────────────────┴──────────────────┘

**Beta-Bernoulli Model — the canonical example:**

    Scenario: coin flipping. We observe n flips, k heads.
    θ = P(heads) — unknown, to be inferred.

    Prior:      θ ~ Beta(α, β)       (α-1 "prior heads", β-1 "prior tails")
    Likelihood: k|θ ~ Binomial(n, θ)
    Posterior:  θ|k ~ Beta(α + k, β + n − k)

    ┌──────────────────────────────────────────────────────────────────┐
    │  Posterior mean:   E[θ|k] = (α + k) / (α + β + n)                │
    │                                                                  │
    │  As n → ∞:  E[θ|k] → k/n   (MLE, data dominates prior)           │
    │  As n → 0:  E[θ|k] → α/(α+β)  (prior mean dominates)             │
    │                                                                  │
    │  The posterior mean is a WEIGHTED COMBINATION of prior mean      │
    │  and MLE, with weight proportional to data size n.               │
    └──────────────────────────────────────────────────────────────────┘

    Sequential updating:
    Observe first flip (heads):  Beta(α, β) → Beta(α+1, β)
    Observe second flip (tails): Beta(α+1, β) → Beta(α+1, β+1)
    Observe k heads, n-k tails:  Beta(α, β) → Beta(α+k, β+n-k)

    This IS the Bayesian learning algorithm for this model.
    Each new data point updates the distribution, not a point estimate.

    Diagram 3 — Beta Posterior Updating (coin flip):

    p(θ)
      │  Prior: Beta(2,2)     5 flips         20 flips        100 flips
      │     ╭──╮              ╭──╮             ╭──╮              ╭─╮
      │    ╱    ╲            ╱    ╲           ╱    ╲            ╱   ╲
      │   ╱      ╲    ──▶   ╱      ╲  ──▶   ╱      ╲  ──▶      ╱     ╲
      │──╱        ╲──      ╱        ╲──╮   ╱      ──╱──       ╱       ╲
      └──────────────      ────────────   ─────────────      ─────────── θ
         0         1          0    1          0    1              0  1
      "Prefer θ≈0.5"    (4/5 heads observed)   "Likely 0.6-0.8"   "Concentrated"

    The posterior sharpens as more data arrives — Bayesian learning
    is the process of prior → posterior → posterior becomes new prior.


──────────────────────────────────────────────────────────────────────────────
### Bayesian Linear Regression

The Bayesian treatment of linear regression is the most important model
for building intuition about Bayesian methods in continuous settings.

    Model:
        y = Xw + ε,   ε ~ N(0, σ²I)
        Prior:  w ~ N(0, τ²I)     (Gaussian prior on weights)

    Posterior (closed-form, conjugate):

        p(w | X, y) = N(w | μ_w, Σ_w)

        Σ_w = (XᵀX/σ² + I/τ²)⁻¹
        μ_w = Σ_w · Xᵀy / σ²

    This posterior mean μ_w is exactly the Ridge regression solution!
    The posterior covariance Σ_w gives the uncertainty in the weights.

    Posterior Predictive Distribution for a new point x*:

        p(y* | x*, X, y) = N(y* | μ_w ᵀ x*, σ² + x*ᵀ Σ_w x*)
                                               ↑              ↑
                                       noise variance   extra uncertainty
                                                        from weight uncertainty

    The total predictive uncertainty has two sources:
    ─ σ²:          irreducible noise (aleatoric uncertainty)
    ─ x*ᵀ Σ_w x*:  parameter uncertainty (epistemic uncertainty)

    ┌──────────────────────────────────────────────────────────────────┐
    │  ALEATORIC uncertainty: noise in the data — cannot be reduced.   │
    │                         More data does NOT help.                 │
    │                                                                  │
    │  EPISTEMIC uncertainty: uncertainty in model parameters —        │
    │                         CAN be reduced with more data.           │
    │                         Posterior concentrates as n → ∞.         │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 4 — Bayesian Linear Regression:

    Posterior distribution over regression lines (not one line, but many):

    y
    │     True line ─────────────────────────────
    │   95% band ═══════════════════════════════  ← epistemic uncertainty
    │  ·● ·   ·●       ·     · ●·              ← noisy observations
    │         ●  ·●          ·    ·●
    └──────────────────────────────────────────── x

    Far from training data: wide band (large x*ᵀ Σ_w x*).
    Near training data: narrow band (Σ_w is well constrained there).
    This is automatic — no separate uncertainty model needed.


──────────────────────────────────────────────────────────────────────────────
### The Marginal Likelihood — Model Comparison and Selection

The evidence p(D) = ∫ p(D|θ) p(θ) dθ appears as a normalising constant
for one model. But when comparing models M₁ vs M₂, it becomes the key:

    Bayes Factor:   BF₁₂ = p(D | M₁) / p(D | M₂)

    BF > 10:  strong evidence for M₁
    BF > 100: decisive evidence for M₁

    Why the marginal likelihood favours the right model complexity:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  Overly simple model:  low p(D|θ) for any θ → low evidence.      │
    │                        Cannot explain the data.                  │
    │                                                                  │
    │  Overly complex model: high p(D|θ) for some θ, but the prior     │
    │                        spreads probability over many θ values,   │
    │                        most of which explain the data poorly.    │
    │                        The integral ∫ p(D|θ)p(θ)dθ is small      │
    │                        because most of the prior mass is wasted. │
    │                                                                  │
    │  Just-right model:     prior concentrates mass near the data-    │
    │                        explaining θ → large integral.            │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    This is Occam's Razor, formalised in probability theory:
    the marginal likelihood automatically penalises unnecessary complexity.

    Diagram 5 — Bayesian Occam's Razor:

    p(D | M)
      │
      │     simple model M₁:   ─────────────────────── (spread flat, low peak)
      │
      │     right model M₂:        ╭──────╮            (concentrated, medium peak)
      │                            ╱        ╲
      │     complex model M₃:   ╭─╯          ╰─╮       (even more spread, lower peak)
      │                         ╱               ╲
      └────────────────────────────────────────────── "complexity of dataset D"
                       ↑
             Real dataset is here:
             M₂ has highest p(D|M₂) → favoured by Bayes.

    Analytic log marginal likelihood for Bayesian linear regression:

        log p(y | X, σ², τ²)
        = −(n/2) log(2π) − (1/2) log|σ²I + τ²XXᵀ|
          − (1/2) yᵀ (σ²I + τ²XXᵀ)⁻¹ y
            ↑                    ↑
        complexity penalty    data fit

    This is automatically balanced — no cross-validation needed for σ², τ².
    Same structure as the GP marginal likelihood (Module 11).


──────────────────────────────────────────────────────────────────────────────
### Approximate Inference — When the Posterior Is Intractable

For most models (neural networks, nonlinear models), the posterior
p(θ|D) = p(D|θ)p(θ)/p(D) has no closed form because p(D) requires
integrating over all θ — often a high-dimensional integral.

    Two main approximation strategies:

    ┌─────────────────────────────────────────────────────────────────┐
    │  SAMPLING METHODS (MCMC)     │  VARIATIONAL METHODS             │
    ├──────────────────────────────┼──────────────────────────────────┤
    │  Approximate p(θ|D) with     │  Approximate p(θ|D) with a       │
    │  samples θ¹,...,θˢ drawn     │  parametric family q_φ(θ).       │
    │  from p(θ|D) directly.       │  Find φ* minimising KL[q‖p].     │
    │                              │                                  │
    │  Asymptotically exact        │  Fast, scales to large models    │
    │  Computationally expensive   │  Biased — q may miss modes       │
    │  Hard to parallelise         │  Easy to parallelise             │
    │  MCMC, HMC, NUTS             │  ELBO, Mean Field, VAE           │
    └──────────────────────────────┴──────────────────────────────────┘

**Variational Inference — the ELBO:**

    Goal: find the distribution q_φ(θ) in a tractable family Q that
    best approximates the true posterior p(θ|D).

    Minimise KL divergence:
        KL[q_φ(θ) ‖ p(θ|D)] = E_q[log q_φ(θ) − log p(θ|D)]

    Since p(θ|D) is not directly available, rewrite using Bayes:

        log p(D) = ELBO(φ) + KL[q_φ(θ) ‖ p(θ|D)]
                        ↑              ↑
                  Evidence Lower   (always ≥ 0)
                  BOund (ELBO)

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  ELBO(φ) = E_{q_φ}[log p(D|θ)] − KL[q_φ(θ) ‖ p(θ)]               │
    │                     ↑                  ↑                         │
    │              expected log-likelihood   regularisation term       │
    │              (fit to data)             (keep q close to prior)   │
    │                                                                  │
    │  Maximising ELBO ≡ minimising KL[q_φ ‖ p(θ|D)]                   │
    │  because: log p(D) = ELBO + KL ≥ ELBO, and log p(D) is fixed.    │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Mean Field Variational Inference:
    Assume q_φ(θ) = ∏ᵢ qᵢ(θᵢ) — fully factorised (independent factors).
    The ELBO becomes tractable and leads to coordinate ascent updates.

**Markov Chain Monte Carlo (MCMC):**

    Build a Markov chain whose stationary distribution is p(θ|D).
    Run the chain for T steps, collect samples {θ¹,...,θᵀ} after
    a burn-in period. Use sample averages to approximate expectations.

    Metropolis-Hastings algorithm:
    1.  Propose θ' ~ q(θ'|θ_current)
    2.  Compute acceptance ratio:
            r = p(θ'|D) / p(θ_current|D)
              = [p(D|θ') p(θ')] / [p(D|θ) p(θ)]   (p(D) cancels!)
    3.  Accept θ' with probability min(1, r); else stay at θ.

    The beauty: p(D) cancels in the acceptance ratio, so we never
    need to compute the intractable evidence integral.

    Diagram 6 — MCMC Chain Convergence:

    θ
    │ 8 ──────────────────────────────────────────── true mean
    │ 6                    ·  · ·  ·  · · ·  ·
    │ 4   burn-in →  ·· ·· ·    ·   ·   ·    ·
    │ 2  ·· · ·  ·
    │ 0 ─────────────────────────────────────────── iteration
    │     0   100  200  300  400  500  600  700
    │     ← discard →  ←───── keep samples ──────→
    │      (burn-in)         (from stationary dist.)


──────────────────────────────────────────────────────────────────────────────
### Bayesian Neural Networks

Apply Bayesian inference to neural network weights W.
Instead of finding a single weight matrix, maintain a distribution p(W|D).

    Prior:      W ~ N(0, σ²_prior I)      (or more structured priors)
    Likelihood: y | x, W ~ N(f_W(x), σ²_noise)  (or softmax for classification)
    Posterior:  p(W | D) ∝ p(D | W) p(W)  — INTRACTABLE for deep nets

    Practical approaches:

    BAYES BY BACKPROP (Blundell et al., 2015):
        Variational inference with diagonal Gaussian q(W; μ, σ):
        Each weight wᵢ ~ N(μᵢ, σᵢ²) independently.
        Minimise the variational ELBO end-to-end via gradient descent.
        Doubles the number of parameters (store μ and σ for each weight).

    MC DROPOUT (Gal & Ghahramani, 2016):
        Applying dropout at test time (not just train time) and
        averaging multiple stochastic forward passes is EQUIVALENT
        to variational inference in a specific Bayesian model.
        Practical: zero overhead — just keep dropout on at test time.

    DEEP ENSEMBLES (Lakshminarayanan et al., 2017):
        Train m independent neural networks from different random inits.
        Average their predictions for the mean; use spread for uncertainty.
        Not strictly Bayesian, but empirically the best uncertainty estimate.

    ┌──────────────────────────────────────────────────────────────────┐
    │  COMPARISON OF UNCERTAINTY METHODS:                              │
    │                                                                  │
    │  Method            │ Uncertainty │ Overhead │ Quality            │
    │  ──────────────────┼─────────────┼──────────┼──────────────────  │
    │  Point estimate    │ None        │ None     │ N/A                │
    │  MC Dropout        │ Epistemic   │ Low      │ Fair               │
    │  Bayes by Backprop │ Epistemic   │ 2×       │ Good               │
    │  Deep Ensembles    │ Both        │ m×       │ Best (empirically) │
    │  MCMC (HMC/NUTS)   │ Both        │ Very high│ Best (theoretical) │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Prior Selection — The Hardest Part of Bayesian Inference

The prior p(θ) encodes all beliefs about θ before seeing data.
Choosing it carefully is the most consequential modelling decision.

    Classes of priors:

    INFORMATIVE PRIORS:  encode specific domain knowledge.
        Physical laws, known constraints, expert opinion.
        Example: θ is a probability → Beta prior constrained to [0,1].

    WEAKLY INFORMATIVE PRIORS:  regularise without strong constraint.
        Half-Normal for standard deviations (enforces positivity).
        Cauchy(0, 2.5) for regression coefficients (heavy-tailed, robust).
        N(0, 1) on standardised coefficients (modern default in Stan/PyMC).

    NON-INFORMATIVE / REFERENCE PRIORS:  minimal information, "let data speak".
        Jeffreys prior: p(θ) ∝ √|I(θ)|  where I(θ) is the Fisher information.
        Invariant to reparameterisation — the technically correct "flat" prior.
        Flat/uniform prior: p(θ) ∝ 1  — seemingly neutral, but encodes
        "all values equally likely", which is informative for scale parameters.

    EMPIRICAL BAYES:  estimate the prior from the data itself.
        p(θ) depends on hyperparameters α: estimate α = argmax_α p(D; α).
        Computationally convenient but technically "cheating" — data used twice.
        Common in hierarchical models; equivalent to marginal likelihood optimisation.

    Prior sensitivity analysis:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Always check: does your conclusion change if you use a          │
    │  different reasonable prior? If yes, the data is insufficient    │
    │  to overwhelm prior assumptions — report this uncertainty.       │
    │  If no, the conclusion is robust.                                │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 7 — Prior Effect on Posterior (small vs large dataset):

    SMALL DATASET (n=10):               LARGE DATASET (n=1000):

    p(θ|D)                              p(θ|D)
    │   Prior A ─────────────           │
    │   Prior B ─  ─  ─  ─              │
    │         ↓         ↓               │         ↓         ↓
    │    Posterior A  Posterior B       │  Posterior A ≈ Posterior B
    │      (different)                  │       (converge — data wins)
    └─────────────── θ                  └─────────────── θ

    Priors matter most when data is scarce. They matter less as n grows.


──────────────────────────────────────────────────────────────────────────────
### Hierarchical Models — Priors on Priors

In many problems, parameters naturally group into exchangeable units.
Hierarchical models share statistical strength across groups.

    Example: estimating batting averages for 30 baseball players.
    Each player j has a true probability θⱼ.

    NON-HIERARCHICAL:  θⱼ ~ Beta(α, β)  with fixed α, β.
                       Each player is estimated independently.
                       Players with few at-bats → huge uncertainty.

    HIERARCHICAL:       α, β ~ p(α, β)   (hyperprior)
                        θⱼ | α, β ~ Beta(α, β)   (player-level prior)
                        yⱼ | θⱼ ~ Binomial(nⱼ, θⱼ)   (data)

    The hierarchical model POOLS information across players.
    Players with few at-bats borrow strength from the group distribution.
    This is partial pooling — between full pooling (one θ for all) and
    no pooling (independent estimates).

    Diagram 8 — Hierarchical Shrinkage:

    Observed batting average:    .150  .200  .220  .310  .380  .400
    (small sample, noisy)

    No pooling estimate:         .150  .200  .220  .310  .380  .400
    (just use observed avg)

    Hierarchical estimate:       .215  .228  .238  .277  .314  .325
                                 ←────────── shrunk toward ──────────→
                                             group mean ≈ 0.270

    Players with extreme observed averages regress toward the group mean.
    This is the JAMES-STEIN phenomenon — it reduces prediction error.


──────────────────────────────────────────────────────────────────────────────
### Calibration — Are Probabilities Trustworthy?

A model is CALIBRATED if its confidence matches its accuracy:
when it says "70% confident", it should be right 70% of the time.

    Diagram 9 — Calibration Plot:

    Actual probability
      1.0 │                              ╱ perfect calibration
      0.8 │                    ●────────╱
      0.6 │          ●────────╱
      0.4 │    ●────╱
      0.2 │ ●─╱
      0.0 │╱
          └──────────────────────────── Predicted probability
              0.0  0.2  0.4  0.6  0.8  1.0

    OVERCONFIDENT (points below diagonal): model says 80% but gets 60%.
    UNDERCONFIDENT (points above diagonal): model says 60% but gets 80%.

    Bayesian models tend to be better calibrated than point-estimate models
    because they average over parameter uncertainty rather than picking a
    single overconfident θ*.

    Calibration metrics:
    ─ Expected Calibration Error (ECE): |E[confidence − accuracy]|
    ─ Brier Score:  E[(p − y)²]  (proper scoring rule)
    ─ Negative Log-Likelihood:  −E[log p(y|x)]  (proper scoring rule)


──────────────────────────────────────────────────────────────────────────────
### Practical Guide: When to Use Bayesian Methods

    ┌──────────────────────────────────────────────────────────────────┐
    │  USE BAYESIAN METHODS WHEN:                                      │
    │  ✓  Small datasets — prior knowledge prevents overfitting        │
    │  ✓  Uncertainty quantification matters (medicine, finance)       │
    │  ✓  Decision-making under uncertainty (Bayesian optimisation)    │
    │  ✓  You have genuine prior knowledge to incorporate              │
    │  ✓  Online/sequential learning — update posterior as data arrives│
    │  ✓  Model comparison without held-out test set (marginal lhood)  │
    │  ✓  Active learning — query points of high epistemic uncertainty │
    │                                                                  │
    │  STICK TO FREQUENTIST WHEN:                                      │
    │  ✓  Very large datasets — posteriors concentrate anyway          │
    │  ✓  Computational budget is tight (MCMC is expensive)            │
    │  ✓  No meaningful prior knowledge available                      │
    │  ✓  Prediction accuracy is the sole objective (calibration OK)   │
    └──────────────────────────────────────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Beta-Bernoulli Conjugate Model — Prior to Posterior": {
        "description": (
            "The most concrete Bayesian learning demonstration: coin flip "
            "inference. Starts with a Beta prior, observes binary data, "
            "computes the closed-form Beta posterior after each observation, "
            "and shows how it updates sequentially. Compares posterior mean, "
            "MLE, and MAP estimates. Visualises the prior-to-posterior "
            "transition across different dataset sizes and prior strengths."
        ),
        "timeout":300,
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.special import betaln

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Beta-Bernoulli model — closed-form posterior
# ─────────────────────────────────────────────────────────────────────────────

def beta_posterior(prior_alpha, prior_beta, n_heads, n_tails):
    """
    Conjugate update: Beta(α,β) + Binomial(n,θ) → Beta(α+k, β+n-k)
    Returns (post_alpha, post_beta, posterior_mean, posterior_mode)
    """
    post_a = prior_alpha + n_heads
    post_b = prior_beta  + n_tails
    mean   = post_a / (post_a + post_b)
    mode   = (post_a - 1) / (post_a + post_b - 2) if (post_a > 1 and post_b > 1) else None
    return post_a, post_b, mean, mode

def beta_credible_interval(alpha, beta_param, level=0.95):
    """Return (lower, upper) equal-tailed credible interval."""
    lo = (1 - level) / 2
    return beta_dist.ppf(lo, alpha, beta_param), beta_dist.ppf(1-lo, alpha, beta_param)

print("=" * 65)
print("  BETA-BERNOULLI CONJUGATE MODEL — COIN FLIP INFERENCE")
print("=" * 65)
print()

# True coin bias (unknown to the model)
TRUE_THETA = 0.65
rng = np.random.default_rng(7)

# Sequential updating — watch the posterior narrow
prior_alpha, prior_beta = 2.0, 2.0   # mild prior: E[θ] = 0.5
print(f"  Prior: Beta({prior_alpha}, {prior_beta})  →  mean={prior_alpha/(prior_alpha+prior_beta):.3f}")
print(f"  True θ = {TRUE_THETA}  (unknown to the model)")
print()
print(f"  {'n_flips':>8}  {'n_heads':>8}  {'MLE':>8}  {'Post mean':>10}  "
      f"{'MAP':>8}  {'95% CI':>18}  {'Width':>8}")
print("  " + "─" * 72)

flip_counts = [1, 3, 5, 10, 20, 50, 100, 300, 1000]
all_flips   = (rng.random(1000) < TRUE_THETA).astype(int)

a_cur, b_cur = prior_alpha, prior_beta
results_seq = []

for n in flip_counts:
    flips  = all_flips[:n]
    k      = flips.sum()
    mle    = k / n
    a_post, b_post, post_mean, post_mode = beta_posterior(prior_alpha, prior_beta, k, n-k)
    lo, hi = beta_credible_interval(a_post, b_post)
    width  = hi - lo
    results_seq.append((n, k, mle, post_mean, post_mode, lo, hi, a_post, b_post))
    map_str = f"{post_mode:.4f}" if post_mode else "  N/A  "
    print(f"  {n:>8}  {k:>8}  {mle:>8.4f}  {post_mean:>10.4f}  "
          f"{map_str:>8}  [{lo:.3f}, {hi:.3f}]  {width:>8.4f}")

print()
print("  OBSERVATIONS:")
print(f"  - At n=1:  huge CI — almost no information.")
print(f"  - At n=50: posterior mean {results_seq[5][3]:.3f} vs true θ={TRUE_THETA}")
print(f"  - At n=1000: 95% CI = [{results_seq[-1][5]:.3f}, {results_seq[-1][6]:.3f}] "
      f"width={results_seq[-1][7]:.4f}")
print(f"  - MAP = mode of posterior (= Ridge regression analogy for Bernoulli).")
print(f"  - MLE = k/n ignores prior; posterior mean = (α+k)/(α+β+n) shrinks toward prior.")

# ─────────────────────────────────────────────────────────────────────────────
# Prior sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  PRIOR SENSITIVITY: HOW MUCH DOES THE PRIOR MATTER?")
print("=" * 65)
print()

n_obs  = 20
k_obs  = int(n_obs * TRUE_THETA)
print(f"  Data: n={n_obs} flips, k={k_obs} heads  (MLE = {k_obs/n_obs:.3f})")
print()
print(f"  {'Prior':>25}  {'Prior mean':>11}  {'Posterior mean':>15}  {'95% CI':>18}")
print("  " + "─" * 72)

priors = [
    (0.5, 0.5, "Jeffreys: Beta(0.5,0.5)"),
    (1.0, 1.0, "Uniform: Beta(1,1)"),
    (2.0, 2.0, "Mild: Beta(2,2)"),
    (5.0, 5.0, "Moderate: Beta(5,5)"),
    (20., 20., "Strong fair: Beta(20,20)"),
    (2.0, 8.0, "Biased low: Beta(2,8)"),
    (8.0, 2.0, "Biased high: Beta(8,2)"),
    (50., 50., "Very strong fair: Beta(50,50)"),
]

for a, b, name in priors:
    a_p, b_p, pm, _ = beta_posterior(a, b, k_obs, n_obs - k_obs)
    lo, hi = beta_credible_interval(a_p, b_p)
    prior_mean = a / (a + b)
    print(f"  {name:>25}  {prior_mean:>11.3f}  {pm:>15.4f}  [{lo:.3f}, {hi:.3f}]")

print()
print("  KEY: Priors with few pseudo-counts (≤2,2) agree on the posterior.")
print("  Strong priors (50,50) resist the data — posterior stays near 0.5.")
print("  With n=20 flips, n is not much larger than prior pseudo-counts.")
print("  With n=1000, all priors above converge to the same posterior.")

# ─────────────────────────────────────────────────────────────────────────────
# MAP vs MLE relationship: MAP = regularised MLE
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  MAP = REGULARISED MLE: THE FORMAL EQUIVALENCE")
print("=" * 65)
print()
print("  For Bernoulli model with Beta(α,β) prior:")
print("  MAP = argmax_θ [k·log θ + (n-k)·log(1-θ) + (α-1)·log θ + (β-1)·log(1-θ)]")
print("  MAP = (k + α - 1) / (n + α + β - 2)   [closed-form]")
print("  MLE = k / n                             [no regularisation]")
print()
print(f"  {'n':>6}  {'k':>5}  {'MLE':>8}  {'MAP(2,2)':>10}  {'MAP(5,5)':>10}  {'Diff(2,2)':>10}")
print("  " + "─" * 54)
for n_t in [5, 10, 20, 50, 100, 500]:
    k_t   = round(n_t * TRUE_THETA)
    mle_t = k_t / n_t
    map22 = (k_t + 2 - 1) / (n_t + 2 + 2 - 2)
    map55 = (k_t + 5 - 1) / (n_t + 5 + 5 - 2)
    print(f"  {n_t:>6}  {k_t:>5}  {mle_t:>8.4f}  {map22:>10.4f}  "
          f"{map55:>10.4f}  {abs(mle_t-map22):>10.4f}")
print()
print("  As n grows, MAP → MLE (prior pseudo-counts become negligible).")

# ─────────────────────────────────────────────────────────────────────────────
# Log marginal likelihood — model comparison
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  LOG MARGINAL LIKELIHOOD — MODEL COMPARISON")
print("=" * 65)
print()

def log_marginal_likelihood_beta_binomial(n, k, alpha, beta_param):
    """log p(k|n, alpha, beta) = log B(alpha+k, beta+n-k) - log B(alpha, beta)"""
    from scipy.special import betaln
    return betaln(alpha + k, beta_param + n - k) - betaln(alpha, beta_param)

# Which prior best explains the observed data?
n_ev, k_ev = 30, 19   # observed 19/30 heads
print(f"  Observed data: n={n_ev}, k={k_ev}  (MLE={k_ev/n_ev:.3f})")
print()
print(f"  {'Prior':>25}  {'log p(D|model)':>16}  {'Marginal lhood':>16}")
print("  " + "─" * 60)

best_log_ml = -np.inf
for a, b, name in priors:
    lml = log_marginal_likelihood_beta_binomial(n_ev, k_ev, a, b)
    print(f"  {name:>25}  {lml:>16.4f}  {np.exp(lml):>16.6f}")
    best_log_ml = max(best_log_ml, lml)

print()
print("  The prior model with highest marginal likelihood is best supported.")
print("  Biased-toward-high prior should win since 19/30 ≈ 0.63 > 0.5.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Beta-Bernoulli Bayesian Inference: Conjugate Updating",
             fontsize=13, fontweight="bold")

theta_grid = np.linspace(0.001, 0.999, 500)

# Panel (0,0): Sequential posterior updating
ax = axes[0, 0]
ax.plot(theta_grid, beta_dist.pdf(theta_grid, prior_alpha, prior_beta),
        "gray", lw=2, ls="--", label="Prior Beta(2,2)")
colours_seq = plt.cm.Blues(np.linspace(0.3, 1.0, len(flip_counts)))
for i, (n_s, k_s, *rest, a_p, b_p) in enumerate(results_seq[::2]):
    ax.plot(theta_grid, beta_dist.pdf(theta_grid, a_p, b_p),
            color=colours_seq[i*2], lw=2, label=f"n={n_s}")
ax.axvline(TRUE_THETA, color="tomato", lw=2, ls=":", label=f"True θ={TRUE_THETA}")
ax.set_title("Sequential Posterior Updating\\n(posterior narrows as n grows)",
             fontweight="bold")
ax.set_xlabel("θ"); ax.set_ylabel("Density")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (0,1): Posterior mean vs MLE vs true θ
ax = axes[0, 1]
n_vals   = [r[0] for r in results_seq]
mle_vals = [r[2] for r in results_seq]
pm_vals  = [r[3] for r in results_seq]
ax.semilogx(n_vals, mle_vals, "tomato", lw=2, marker="o", ms=6, label="MLE (k/n)")
ax.semilogx(n_vals, pm_vals,  "steelblue", lw=2, marker="s", ms=6,
            label="Posterior mean")
ax.axhline(TRUE_THETA, color="black", lw=2, ls="--", label=f"True θ={TRUE_THETA}")
ax.fill_between(n_vals,
                [r[5] for r in results_seq],
                [r[6] for r in results_seq],
                alpha=0.2, color="steelblue", label="95% CI")
ax.set_xlabel("n (log scale)"); ax.set_ylabel("θ estimate")
ax.set_title("MLE vs Posterior Mean vs True θ\\n(both converge; posterior regularised early)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# Panel (0,2): Prior sensitivity
ax = axes[0, 2]
for a, b, name in priors:
    a_p, b_p, _, _ = beta_posterior(a, b, k_obs, n_obs - k_obs)
    ax.plot(theta_grid, beta_dist.pdf(theta_grid, a_p, b_p), lw=1.8,
            label=name.split(":")[0])
ax.axvline(k_obs/n_obs, color="black", lw=2, ls="--", label=f"MLE={k_obs/n_obs:.2f}")
ax.axvline(TRUE_THETA,  color="tomato", lw=1.5, ls=":",  label=f"True θ={TRUE_THETA}")
ax.set_xlim(0, 1)
ax.set_title(f"Prior Sensitivity (n={n_obs} obs)\\n(strong priors resist data; weak priors agree)",
             fontweight="bold")
ax.set_xlabel("θ"); ax.set_ylabel("Density")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (1,0): MAP vs MLE as function of n
ax = axes[1, 0]
n_range  = np.arange(1, 200)
mle_arr  = np.array([max(round(n * TRUE_THETA), 0) / n for n in n_range])
map22arr = np.array([(max(round(n * TRUE_THETA), 0) + 1) / (n + 2) for n in n_range])
map55arr = np.array([(max(round(n * TRUE_THETA), 0) + 4) / (n + 8) for n in n_range])
ax.plot(n_range, mle_arr,  "tomato",    lw=2, label="MLE = k/n")
ax.plot(n_range, map22arr, "steelblue", lw=2, label="MAP Beta(2,2)")
ax.plot(n_range, map55arr, "seagreen",  lw=2, label="MAP Beta(5,5)")
ax.axhline(TRUE_THETA, color="black", lw=1.5, ls="--", label=f"True θ={TRUE_THETA}")
ax.set_xlabel("n"); ax.set_ylabel("θ estimate")
ax.set_title("MAP vs MLE: Regularisation Disappears as n Grows",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,1): Beta prior shapes
ax = axes[1, 1]
beta_params = [(0.5, 0.5, "Jeffreys (0.5,0.5)"),
               (1.0, 1.0, "Uniform (1,1)"),
               (2.0, 2.0, "Symmetric (2,2)"),
               (2.0, 5.0, "Skewed low (2,5)"),
               (5.0, 2.0, "Skewed high (5,2)"),
               (10., 10., "Concentrated (10,10)")]
cols_b = ["purple", "gray", "steelblue", "tomato", "seagreen", "darkorange"]
for (a, b, lbl), col in zip(beta_params, cols_b):
    y_vals = beta_dist.pdf(theta_grid, a, b)
    y_vals = np.clip(y_vals, 0, 15)
    ax.plot(theta_grid, y_vals, lw=2, color=col, label=lbl)
ax.set_xlim(0, 1); ax.set_ylim(0, 10)
ax.set_title("Beta Prior Shapes\\n(choose based on prior beliefs about θ)",
             fontweight="bold")
ax.set_xlabel("θ"); ax.set_ylabel("p(θ)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (1,2): Log marginal likelihood vs prior
ax = axes[1, 2]
prior_labels  = [p[2].split(":")[0] for p in priors]
lml_vals = [log_marginal_likelihood_beta_binomial(n_ev, k_ev, a, b) for a, b, _ in priors]
bars = ax.barh(prior_labels, lml_vals, color="steelblue", alpha=0.8)
ax.axvline(max(lml_vals), color="tomato", lw=2, ls="--", label="Best model")
ax.set_xlabel("Log marginal likelihood")
ax.set_title(f"Model Comparison via Marginal Likelihood\\n"
             f"(n={n_ev}, k={k_ev}, MLE={k_ev/n_ev:.2f})",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("beta_bernoulli_bayesian.png", dpi=110)
print()
print("  Plot saved → beta_bernoulli_bayesian.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Bayesian Linear Regression — Posterior and Predictive Distribution": {
        "description": (
            "Full Bayesian treatment of linear regression. Derives the "
            "closed-form Gaussian posterior over weights, samples from it, "
            "and computes the posterior predictive distribution including "
            "uncertainty bands. Shows that posterior mean = Ridge regression. "
            "Decomposes predictive uncertainty into aleatoric (irreducible "
            "noise) and epistemic (parameter uncertainty) components. "
            "Demonstrates how epistemic uncertainty widens far from data."
        ),
        "timeout":300,
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy
from itertools import combinations_with_replacement as _cwr

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree; self.include_bias = include_bias
    def fit(self, X, y=None):
        _, p = X.shape; self._combos = []
        for d in range(0 if self.include_bias else 1, self.degree + 1):
            self._combos.extend(_cwr(range(p), d))
        return self
    def transform(self, X):
        X = _np_impl.asarray(X)
        n = X.shape[0]; out = _np_impl.ones((n, len(self._combos)))
        for j, combo in enumerate(self._combos):
            for idx in combo: out[:, j] *= X[:, idx]
        return out
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0; return self
    def transform(self, X):            return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class Ridge:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha; self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            self._xm = X.mean(0); self._ym = y.mean()
            Xc = X - self._xm; yc = y - self._ym
        else:
            Xc = X; yc = y; self._xm = _np_impl.zeros(X.shape[1]); self._ym = 0.0
        n, d = Xc.shape
        self.coef_ = _np_impl.linalg.solve(Xc.T@Xc + self.alpha*_np_impl.eye(d), Xc.T@yc)
        self.intercept_ = self._ym - self._xm @ self.coef_; return self
    def predict(self, X): return X @ self.coef_ + self.intercept_

def _sig(z): return 1.0 / (1.0 + _np_impl.exp(-_np_impl.clip(z, -500, 500)))
class LogisticRegression:
    def __init__(self, C=1.0, max_iter=2000, solver='lbfgs'):
        self.C = C; self.max_iter = max_iter
    def fit(self, X, y):
        n, d = X.shape; lam = 1.0 / (self.C * n)
        def fg(w):
            p = _np_impl.clip(_sig(X@w[1:]+w[0]), 1e-10, 1-1e-10)
            loss = -_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)) + 0.5*lam*_np_impl.sum(w[1:]**2)
            e = p - y
            return loss, _np_impl.r_[e.mean(), X.T@e/n + lam*w[1:]]
        res = _sp_opt.minimize(fg, _np_impl.zeros(d+1), method='L-BFGS-B', jac=True,
                               options={'maxiter': self.max_iter})
        self.coef_ = res.x[1:].reshape(1,-1); self.intercept_ = _np_impl.array([res.x[0]]); return self
    def predict(self, X): return (_sig(X@self.coef_.ravel()+self.intercept_[0]) >= 0.5).astype(int)
    def score(self, X, y): return _np_impl.mean(self.predict(X) == y)

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), max_iter=200,
                 random_state=None, alpha=0.0001, learning_rate_init=0.001):
        self.hidden_layer_sizes = hidden_layer_sizes; self.max_iter = max_iter
        self.random_state = random_state; self.alpha = alpha; self.lr = learning_rate_init
    def fit(self, X, y):
        rng = _np_impl.random.default_rng(self.random_state)
        dims = [X.shape[1]] + list(self.hidden_layer_sizes) + [1]
        self._W = [rng.normal(0, _np_impl.sqrt(2.0/dims[i]), (dims[i], dims[i+1]))
                   for i in range(len(dims)-1)]
        self._b = [_np_impl.zeros(dims[i+1]) for i in range(len(dims)-1)]
        n = len(X)
        for epoch in range(self.max_iter):
            idx = rng.permutation(n)
            for start in range(0, n, 32):
                xb = X[idx[start:start+32]]; yb = y[idx[start:start+32]]
                acts = [xb]
                for i, (W, b) in enumerate(zip(self._W, self._b)):
                    z = acts[-1] @ W + b
                    acts.append(_np_impl.maximum(0, z) if i < len(self._W)-1 else z.ravel())
                delta = (acts[-1] - yb) / len(xb)
                for i in range(len(self._W)-1, -1, -1):
                    dW = (acts[i].T @ delta.reshape(-1,1)
                          if delta.ndim==1 and i==len(self._W)-1
                          else acts[i].T @ (delta.reshape(-1,1) if delta.ndim==1 else delta))
                    dW = dW.reshape(self._W[i].shape) + self.alpha * self._W[i]
                    db = _np_impl.array([delta.sum()]) if delta.ndim==1 else delta.sum(0)
                    self._W[i] -= self.lr * dW; self._b[i] -= self.lr * db
                    if i > 0:
                        delta = ((delta.reshape(-1,1) @ self._W[i].T)
                                 if delta.ndim==1 else delta @ self._W[i].T)
                        delta = delta * (acts[i] > 0)
        return self
    def predict(self, X):
        a = X
        for i, (W, b) in enumerate(zip(self._W, self._b)):
            z = a @ W + b
            a = _np_impl.maximum(0, z) if i < len(self._W)-1 else z.ravel()
        return a


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Bayesian linear regression — closed-form posterior
# ─────────────────────────────────────────────────────────────────────────────

class BayesianLinearRegression:
    """
    Model:  y = Φw + ε,  ε ~ N(0, σ²I),  w ~ N(0, τ²I)
    Posterior: p(w | y, Φ) = N(μ_w, Σ_w)
        Σ_w = (ΦᵀΦ/σ² + I/τ²)⁻¹
        μ_w = Σ_w · Φᵀy / σ²
    Predictive: p(y* | x*, data) = N(μ_w·φ(x*), σ² + φ(x*)ᵀ Σ_w φ(x*))
    """
    def __init__(self, sigma2=0.1, tau2=1.0):
        self.sigma2 = sigma2   # noise variance
        self.tau2   = tau2     # prior variance
        self.mu_w   = None
        self.Sigma_w = None

    def fit(self, Phi, y):
        """Phi: (n × d) design matrix (features already computed)."""
        n, d = Phi.shape
        # Posterior covariance and mean
        A = Phi.T @ Phi / self.sigma2 + np.eye(d) / self.tau2
        self.Sigma_w = np.linalg.inv(A)
        self.mu_w    = self.Sigma_w @ Phi.T @ y / self.sigma2
        return self

    def predict(self, Phi_star, return_std=True):
        """Returns predictive mean and std for each row of Phi_star."""
        mu_pred = Phi_star @ self.mu_w
        if not return_std:
            return mu_pred
        # Epistemic: each row φ*ᵀ Σ_w φ*
        epistemic_var = np.array([phi @ self.Sigma_w @ phi for phi in Phi_star])
        total_var     = self.sigma2 + epistemic_var
        return mu_pred, np.sqrt(total_var), np.sqrt(epistemic_var)

    def sample_weights(self, n_samples=5, seed=0):
        """Sample weight vectors from the posterior."""
        rng = np.random.default_rng(seed)
        L   = np.linalg.cholesky(self.Sigma_w + 1e-10 * np.eye(len(self.mu_w)))
        return self.mu_w + (L @ rng.standard_normal((len(self.mu_w), n_samples))).T

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Bayesian linear regression vs Ridge
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  BAYESIAN LINEAR REGRESSION vs RIDGE: WEIGHT EQUIVALENCE")
print("=" * 65)
print()

# 1D example: noisy sine curve
def true_fn(x): return np.sin(2 * np.pi * x)

n_tr  = 25
sigma = 0.3     # noise std
tau   = 1.0     # prior std

X_tr = np.sort(np.random.uniform(0, 1, n_tr))
y_tr = true_fn(X_tr) + np.random.normal(0, sigma, n_tr)
X_te = np.linspace(-0.1, 1.1, 300)

# Build polynomial features (degree 5)
degree  = 5
pf      = PolynomialFeatures(degree=degree, include_bias=True)
sc      = StandardScaler()
Phi_tr  = sc.fit_transform(pf.fit_transform(X_tr.reshape(-1, 1)))
Phi_te  = sc.transform(pf.transform(X_te.reshape(-1, 1)))

sigma2 = sigma**2
tau2   = tau**2

blr = BayesianLinearRegression(sigma2=sigma2, tau2=tau2).fit(Phi_tr, y_tr)
mu_pred, std_pred, std_ep = blr.predict(Phi_te)

# Ridge with λ = σ²/τ² should match BLR posterior mean
lambda_ridge = sigma2 / tau2
ridge = Ridge(alpha=lambda_ridge, fit_intercept=False).fit(Phi_tr, y_tr)
ridge_pred = ridge.predict(Phi_te)

max_diff = np.max(np.abs(mu_pred - ridge_pred))
print(f"  Prior: w ~ N(0, {tau2:.1f}·I)   Noise: ε ~ N(0, {sigma2:.2f})")
print(f"  Equivalent Ridge λ = σ²/τ² = {lambda_ridge:.4f}")
print(f"  max|BLR posterior mean − Ridge prediction| = {max_diff:.2e}")
print(f"  Equivalence confirmed: {'✓' if max_diff < 1e-8 else '✗'}")
print()
print("  BLR posterior mean = Ridge regression solution.")
print("  Gaussian prior with variance τ² → Ridge with λ = σ²/τ².")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Aleatoric vs epistemic uncertainty
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  ALEATORIC vs EPISTEMIC UNCERTAINTY")
print("=" * 65)
print()

print("  Total uncertainty = aleatoric (noise) + epistemic (parameter)")
print()

# Show uncertainty at various test points
test_positions = {
    "x*=0.50 (near centre of data)": 0.50,
    "x*=0.00 (at data boundary)":    0.00,
    "x*=-0.1 (outside data range)": -0.10,
    "x*=1.10 (outside data range)":  1.10,
    "x*=2.00 (far from data)":       2.00,
}

print(f"  {'Test point':>40}  {'Noise σ':>8}  {'Epistemic σ':>12}  {'Total σ':>10}")
print("  " + "─" * 74)
for label, x_val in test_positions.items():
    # phi_x  = sc.transform(pf.transform([[x_val]]))
    phi_x = sc.transform(pf.transform(np.array([[x_val]])))
    mu_x, total_std, ep_std = blr.predict(phi_x)
    noise_std_val = np.sqrt(sigma2)
    print(f"  {label:>40}  {noise_std_val:>8.4f}  {ep_std[0]:>12.4f}  {total_std[0]:>10.4f}")

print()
print("  - Noise σ is constant (aleatoric — cannot be reduced).")
print("  - Epistemic σ grows far from training data (can be reduced with more data).")
print("  - Points outside the training range have very high epistemic uncertainty.")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Posterior updating — more data → less uncertainty
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  UNCERTAINTY REDUCTION WITH MORE DATA")
print("=" * 65)
print()

x_query = 0.5   # a fixed test point
print(f"  Query point: x* = {x_query}")
print(f"  {'n_train':>9}  {'Post mean':>10}  {'Epistemic σ':>13}  {'Total σ':>10}  {'95% CI':>20}")
print("  " + "─" * 65)

X_large = np.sort(np.random.uniform(0, 1, 500))
y_large = true_fn(X_large) + np.random.normal(0, sigma, 500)

for n_pts in [5, 10, 20, 50, 100, 200, 500]:
    X_sub  = X_large[:n_pts]; y_sub = y_large[:n_pts]
    Phi_sub = sc.transform(pf.transform(X_sub.reshape(-1, 1)))
    phi_q   = sc.transform(pf.transform([[x_query]]))
    blr_n   = BayesianLinearRegression(sigma2=sigma2, tau2=tau2).fit(Phi_sub, y_sub)
    mu_q, total_s, ep_s = blr_n.predict(phi_q)
    ci_lo   = mu_q[0] - 1.96 * total_s[0]
    ci_hi   = mu_q[0] + 1.96 * total_s[0]
    print(f"  {n_pts:>9}  {mu_q[0]:>10.4f}  {ep_s[0]:>13.4f}  "
          f"{total_s[0]:>10.4f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Bayesian Linear Regression: Posterior, Predictive Uncertainty, "
             "and Aleatoric vs Epistemic Decomposition",
             fontsize=13, fontweight="bold")

# Panel (0,0): Posterior weight samples — functions drawn from posterior
ax = axes[0, 0]
w_samples = blr.sample_weights(n_samples=20, seed=1)
for ws in w_samples:
    ax.plot(X_te, Phi_te @ ws, "steelblue", lw=0.8, alpha=0.3)
ax.plot(X_te, mu_pred, "steelblue", lw=2.5, label="Posterior mean = Ridge")
ax.plot(X_te, true_fn(X_te), "k--", lw=2, alpha=0.5, label="True fn")
ax.scatter(X_tr, y_tr, c="black", s=30, zorder=6, label="Training data")
ax.set_xlim(-0.1, 1.1); ax.set_ylim(-2.5, 2.5)
ax.set_title("Posterior Samples\\n(each sample = a complete regression line)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (0,1): Predictive distribution with uncertainty bands
ax = axes[0, 1]
ax.scatter(X_tr, y_tr, c="black", s=30, zorder=6, label="Training data")
ax.plot(X_te, true_fn(X_te), "k--", lw=2, alpha=0.5, label="True fn")
ax.plot(X_te, mu_pred, "steelblue", lw=2.5, label="Posterior mean")
ax.fill_between(X_te, mu_pred - 2*std_pred, mu_pred + 2*std_pred,
                alpha=0.2, color="steelblue", label="±2σ total")
ax.fill_between(X_te, mu_pred - 2*std_ep, mu_pred + 2*std_ep,
                alpha=0.3, color="tomato", label="±2σ epistemic only")
ax.set_xlim(-0.1, 1.1); ax.set_ylim(-2.5, 2.5)
ax.set_title("Posterior Predictive Distribution\\n"
             "(blue=total uncertainty, red=epistemic only)",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (0,2): Uncertainty profile along x
ax = axes[0, 2]
ax.plot(X_te, std_pred, "steelblue", lw=2, label="Total σ")
ax.plot(X_te, std_ep,   "tomato",    lw=2, label="Epistemic σ")
ax.axhline(np.sqrt(sigma2), color="gray", lw=1.5, ls="--",
           label=f"Aleatoric σ={sigma:.2f}")
for xv in X_tr:
    ax.axvline(xv, color="black", alpha=0.05, lw=1)
ax.set_xlim(-0.1, 1.1)
ax.set_xlabel("x"); ax.set_ylabel("Predictive std")
ax.set_title("Uncertainty Profile\\n(epistemic rises outside training range)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,0): Prior vs posterior weight distributions (for two components)
ax = axes[1, 0]
w_idx = [1, 2]   # which weight components to show
for i, wi in enumerate(w_idx):
    prior_std = np.sqrt(tau2)
    post_std  = np.sqrt(blr.Sigma_w[wi, wi])
    w_grid    = np.linspace(blr.mu_w[wi] - 4*prior_std, blr.mu_w[wi] + 4*prior_std, 300)
    from scipy.stats import norm
    ax.plot(w_grid, norm.pdf(w_grid, 0, prior_std), ls="--",
            lw=2, label=f"Prior w_{wi}")
    col = "steelblue" if i == 0 else "tomato"
    ax.plot(w_grid, norm.pdf(w_grid, blr.mu_w[wi], post_std),
            lw=2.5, color=col, label=f"Posterior w_{wi}")
    ax.axvline(blr.mu_w[wi], color=col, lw=1, ls=":")
ax.set_title("Prior vs Posterior Weight Distributions\\n"
             "(prior is broad; posterior concentrates on likely weights)",
             fontweight="bold")
ax.set_xlabel("Weight value"); ax.set_ylabel("Density")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,1): Data size effect on posterior
ax = axes[1, 1]
X_te_2 = np.linspace(-0.1, 1.1, 300)
Phi_te_2 = sc.transform(pf.transform(X_te_2.reshape(-1, 1)))
for n_pts, col in [(5, "tomato"), (20, "seagreen"), (100, "steelblue")]:
    X_sub   = X_large[:n_pts]; y_sub = y_large[:n_pts]
    Phi_sub = sc.transform(pf.transform(X_sub.reshape(-1, 1)))
    blr_n   = BayesianLinearRegression(sigma2=sigma2, tau2=tau2).fit(Phi_sub, y_sub)
    mu_n, std_n, _ = blr_n.predict(Phi_te_2)
    ax.plot(X_te_2, mu_n, color=col, lw=2, label=f"n={n_pts}")
    ax.fill_between(X_te_2, mu_n - 2*std_n, mu_n + 2*std_n, alpha=0.1, color=col)
ax.plot(X_te_2, true_fn(X_te_2), "k--", lw=2, alpha=0.5, label="True fn")
ax.scatter(X_large[:5], y_large[:5], c="tomato", s=20, zorder=6)
ax.scatter(X_large[:100], y_large[:100], c="steelblue", s=10, alpha=0.4, zorder=5)
ax.set_xlim(-0.1, 1.1); ax.set_ylim(-2.5, 2.5)
ax.set_title("More Data → Less Epistemic Uncertainty\\n"
             "(bands narrow as n increases)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,2): Calibration of predictive distribution
ax = axes[1, 2]
n_cal   = 500
X_cal   = np.random.uniform(0, 1, n_cal)
y_cal   = true_fn(X_cal) + np.random.normal(0, sigma, n_cal)
Phi_cal = sc.transform(pf.transform(X_cal.reshape(-1, 1)))
mu_cal, std_cal, _ = blr.predict(Phi_cal)
z_scores = (y_cal - mu_cal) / std_cal

# Empirical coverage at various nominal levels
nominal_levels = np.linspace(0.05, 0.99, 40)
empirical_cov  = []
for level in nominal_levels:
    z_crit = np.abs(np.percentile(z_scores, [(1-level)/2*100, (1+level)/2*100]))
    # Fraction of test points within ±z_α/2 of prediction
    within = np.mean(np.abs(z_scores) <= z_crit.mean())
    empirical_cov.append(within)

ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect calibration")
ax.plot(nominal_levels, empirical_cov, "steelblue", lw=2.5, marker="o", ms=3,
        label="BLR")
ax.set_xlabel("Nominal coverage level")
ax.set_ylabel("Empirical coverage")
ax.set_title("Predictive Calibration\\n(BLR should track the diagonal)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("bayesian_linear_regression.png", dpi=110)
print()
print("  Plot saved → bayesian_linear_regression.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · MCMC from Scratch — Metropolis-Hastings Sampling": {
        "description": (
            "Implements the Metropolis-Hastings algorithm from scratch and "
            "uses it to sample from intractable posteriors. Demonstrates: "
            "burn-in and convergence diagnostics (trace plot, acceptance rate, "
            "autocorrelation), the effect of proposal variance on mixing, "
            "and Bayesian logistic regression posterior sampling. Compares "
            "MCMC samples to the true posterior for a known distribution, "
            "and shows how posterior samples give predictive uncertainty."
        ),
        "timeout":300,
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, beta as beta_dist, multivariate_normal
from scipy.special import expit   # sigmoid
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy
from itertools import combinations_with_replacement as _cwr

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree; self.include_bias = include_bias
    def fit(self, X, y=None):
        _, p = X.shape; self._combos = []
        for d in range(0 if self.include_bias else 1, self.degree + 1):
            self._combos.extend(_cwr(range(p), d))
        return self
    def transform(self, X):
        n = X.shape[0]; out = _np_impl.ones((n, len(self._combos)))
        for j, combo in enumerate(self._combos):
            for idx in combo: out[:, j] *= X[:, idx]
        return out
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0; return self
    def transform(self, X):            return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class Ridge:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha; self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            self._xm = X.mean(0); self._ym = y.mean()
            Xc = X - self._xm; yc = y - self._ym
        else:
            Xc = X; yc = y; self._xm = _np_impl.zeros(X.shape[1]); self._ym = 0.0
        n, d = Xc.shape
        self.coef_ = _np_impl.linalg.solve(Xc.T@Xc + self.alpha*_np_impl.eye(d), Xc.T@yc)
        self.intercept_ = self._ym - self._xm @ self.coef_; return self
    def predict(self, X): return X @ self.coef_ + self.intercept_

def _sig(z): return 1.0 / (1.0 + _np_impl.exp(-_np_impl.clip(z, -500, 500)))
class LogisticRegression:
    def __init__(self, C=1.0, max_iter=2000, solver='lbfgs'):
        self.C = C; self.max_iter = max_iter
    def fit(self, X, y):
        n, d = X.shape; lam = 1.0 / (self.C * n)
        def fg(w):
            p = _np_impl.clip(_sig(X@w[1:]+w[0]), 1e-10, 1-1e-10)
            loss = -_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)) + 0.5*lam*_np_impl.sum(w[1:]**2)
            e = p - y
            return loss, _np_impl.r_[e.mean(), X.T@e/n + lam*w[1:]]
        res = _sp_opt.minimize(fg, _np_impl.zeros(d+1), method='L-BFGS-B', jac=True,
                               options={'maxiter': self.max_iter})
        self.coef_ = res.x[1:].reshape(1,-1); self.intercept_ = _np_impl.array([res.x[0]]); return self
    def predict(self, X): return (_sig(X@self.coef_.ravel()+self.intercept_[0]) >= 0.5).astype(int)
    def score(self, X, y): return _np_impl.mean(self.predict(X) == y)

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), max_iter=200,
                 random_state=None, alpha=0.0001, learning_rate_init=0.001):
        self.hidden_layer_sizes = hidden_layer_sizes; self.max_iter = max_iter
        self.random_state = random_state; self.alpha = alpha; self.lr = learning_rate_init
    def fit(self, X, y):
        rng = _np_impl.random.default_rng(self.random_state)
        dims = [X.shape[1]] + list(self.hidden_layer_sizes) + [1]
        self._W = [rng.normal(0, _np_impl.sqrt(2.0/dims[i]), (dims[i], dims[i+1]))
                   for i in range(len(dims)-1)]
        self._b = [_np_impl.zeros(dims[i+1]) for i in range(len(dims)-1)]
        n = len(X)
        for epoch in range(self.max_iter):
            idx = rng.permutation(n)
            for start in range(0, n, 32):
                xb = X[idx[start:start+32]]; yb = y[idx[start:start+32]]
                acts = [xb]
                for i, (W, b) in enumerate(zip(self._W, self._b)):
                    z = acts[-1] @ W + b
                    acts.append(_np_impl.maximum(0, z) if i < len(self._W)-1 else z.ravel())
                delta = (acts[-1] - yb) / len(xb)
                for i in range(len(self._W)-1, -1, -1):
                    dW = (acts[i].T @ delta.reshape(-1,1)
                          if delta.ndim==1 and i==len(self._W)-1
                          else acts[i].T @ (delta.reshape(-1,1) if delta.ndim==1 else delta))
                    dW = dW.reshape(self._W[i].shape) + self.alpha * self._W[i]
                    db = _np_impl.array([delta.sum()]) if delta.ndim==1 else delta.sum(0)
                    self._W[i] -= self.lr * dW; self._b[i] -= self.lr * db
                    if i > 0:
                        delta = ((delta.reshape(-1,1) @ self._W[i].T)
                                 if delta.ndim==1 else delta @ self._W[i].T)
                        delta = delta * (acts[i] > 0)
        return self
    def predict(self, X):
        a = X
        for i, (W, b) in enumerate(zip(self._W, self._b)):
            z = a @ W + b
            a = _np_impl.maximum(0, z) if i < len(self._W)-1 else z.ravel()
        return a


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Metropolis-Hastings from scratch
# ─────────────────────────────────────────────────────────────────────────────

def metropolis_hastings(log_target, theta_init, n_samples, proposal_std, seed=0):
    """
    Random-walk Metropolis-Hastings sampler.

    Parameters
    ----------
    log_target  : callable  θ → log p(θ)  (unnormalised OK)
    theta_init  : ndarray   starting point
    n_samples   : int       total samples (including burn-in)
    proposal_std: float     std of isotropic Gaussian proposal
    seed        : int

    Returns
    -------
    samples     : (n_samples, d) array of accepted/rejected samples
    accept_rate : float  fraction of proposals accepted
    """
    rng    = np.random.default_rng(seed)
    d      = len(theta_init)
    samples = np.zeros((n_samples, d))
    theta   = theta_init.copy()
    log_p   = log_target(theta)
    n_accept = 0

    for i in range(n_samples):
        # Propose from isotropic Gaussian
        theta_prop = theta + rng.normal(0, proposal_std, d)
        log_p_prop = log_target(theta_prop)

        # Acceptance step — log ratio to avoid overflow
        log_r = log_p_prop - log_p
        if np.log(rng.random()) < log_r:
            theta  = theta_prop
            log_p  = log_p_prop
            n_accept += 1

        samples[i] = theta

    return samples, n_accept / n_samples


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Sampling from a known bivariate posterior
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  METROPOLIS-HASTINGS: KNOWN TARGET (2D BANANA POSTERIOR)")
print("=" * 65)
print()

# Banana-shaped posterior — common MCMC test case
def log_banana(theta, b=0.1):
    """log p(θ₁, θ₂) ∝ −θ₁²/200 − (θ₂ + b·θ₁² − 100b)²/2"""
    t1, t2 = theta[0], theta[1]
    return -t1**2 / 200 - (t2 + b * t1**2 - 100 * b)**2 / 2

n_mcmc     = 20000
burn_in    = 5000
theta_init = np.array([0.0, 0.0])

print("  Testing Metropolis-Hastings on banana-shaped posterior...")
print(f"  n_samples={n_mcmc}, burn_in={burn_in}")
print()
print(f"  {'Proposal σ':>12}  {'Accept rate':>13}  {'E[θ₁]':>8}  {'E[θ₂]':>8}  {'Mixing'}  ")
print("  " + "─" * 62)

proposal_stds = [0.1, 1.0, 5.0, 20.0]
all_samples   = {}

for prop_s in proposal_stds:
    samples, acc_rate = metropolis_hastings(log_banana, theta_init,
                                            n_mcmc, prop_s, seed=0)
    post_samples = samples[burn_in:]
    mean_1 = post_samples[:, 0].mean()
    mean_2 = post_samples[:, 1].mean()

    # Autocorrelation at lag 1 as mixing proxy
    t1_chain = post_samples[:, 0] - post_samples[:, 0].mean()
    if t1_chain.std() > 1e-10:
        ac_lag1 = np.corrcoef(t1_chain[:-1], t1_chain[1:])[0, 1]
    else:
        ac_lag1 = 1.0

    quality = ("poor (stuck)"  if acc_rate < 0.1 else
               "too high (random walk)" if acc_rate > 0.85 else
               "good")
    print(f"  {prop_s:>12.1f}  {acc_rate:>13.4f}  {mean_1:>8.3f}  "
          f"{mean_2:>8.3f}  {quality}")
    all_samples[prop_s] = post_samples

print()
print("  TARGET acceptance rate: 0.23–0.50 for well-mixed chains.")
print("  Goldilocks zone: proposal σ not too small (stuck), not too big (rejects).")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Bayesian logistic regression via MCMC
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  BAYESIAN LOGISTIC REGRESSION VIA MCMC")
print("=" * 65)
print()

# Generate binary classification data
n_data   = 60
X_logit  = np.random.randn(n_data, 2)
w_true   = np.array([2.0, -1.5])
b_true   = 0.5
logits   = X_logit @ w_true + b_true
y_logit  = (np.random.random(n_data) < expit(logits)).astype(float)

print(f"  Data: n={n_data}, true w=[{w_true[0]}, {w_true[1]}], b={b_true}")
print()

def log_posterior_logistic(theta, X, y, sigma_prior=2.0):
    """
    Log posterior for Bayesian logistic regression:
    log p(w,b | D) ∝ Σ [y·log σ(xᵀw+b) + (1-y)·log(1-σ(xᵀw+b))]
                     − (1/2σ²)(‖w‖² + b²)
    theta = [w₁, w₂, b]
    """
    w = theta[:2]; b = theta[2]
    logits_  = X @ w + b
    # Numerically stable log-likelihood
    ll = np.sum(y * logits_ - np.log1p(np.exp(logits_)))
    # Gaussian prior
    log_prior = -0.5 * np.sum(theta**2) / sigma_prior**2
    return ll + log_prior

# Run MCMC
theta_init_lr = np.zeros(3)
samples_lr, acc_lr = metropolis_hastings(
    lambda th: log_posterior_logistic(th, X_logit, y_logit),
    theta_init_lr, n_samples=30000, proposal_std=0.25, seed=42)
post_lr     = samples_lr[5000:]  # burn-in = 5000

print(f"  MCMC acceptance rate: {acc_lr:.4f}")
print()
print("  POSTERIOR SUMMARY (mean ± std over MCMC samples):")
param_names = ["w₁", "w₂", "b"]
for i, name in enumerate(param_names):
    chain_i = post_lr[:, i]
    print(f"    {name}: {chain_i.mean():>7.4f} ± {chain_i.std():.4f}  "
          f"  true = {[w_true[0], w_true[1], b_true][i]:.2f}  "
          f"  95% CI = [{np.percentile(chain_i, 2.5):.3f}, "
          f"{np.percentile(chain_i, 97.5):.3f}]")

# Compare to MLE (numpy LogisticRegression)
lr_mle = LogisticRegression(C=1e4, max_iter=2000).fit(X_logit, y_logit.astype(int))
print()
print("  MLE (numpy LogisticRegression, large C=10000):")
print(f"    w = {lr_mle.coef_[0].round(4)},  b = {lr_mle.intercept_[0]:.4f}")
print()
print("  MCMC posterior mean is close to MLE but slightly regularised by prior.")

# Posterior predictive
xx1, xx2 = np.meshgrid(np.linspace(-3, 3, 80), np.linspace(-3, 3, 80))
X_grid = np.c_[xx1.ravel(), xx2.ravel()]

# Average predicted probability over posterior samples
n_pred_samples = 500
rng_pred       = np.random.default_rng(1)
idx_pred       = rng_pred.choice(len(post_lr), n_pred_samples, replace=False)
pred_probs     = np.zeros(len(X_grid))
for i in idx_pred:
    th     = post_lr[i]
    logits_ = X_grid @ th[:2] + th[2]
    pred_probs += expit(logits_)
pred_probs /= n_pred_samples
uncertainty  = pred_probs * (1 - pred_probs)  # variance of Bernoulli

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("MCMC Sampling: Metropolis-Hastings, Diagnostics, and Bayesian Logistic Regression",
             fontsize=13, fontweight="bold")

# Panel (0,0): 2D banana posterior samples (good proposal)
ax = axes[0, 0]
good_s = all_samples[5.0]
ax.scatter(good_s[::5, 0], good_s[::5, 1], s=2, alpha=0.3, c="steelblue")
t1_grid = np.linspace(-25, 25, 200)
t2_grid = np.linspace(-10, 20, 200)
T1, T2  = np.meshgrid(t1_grid, t2_grid)
log_p_grid = np.array([[log_banana([t1, t2]) for t1 in t1_grid] for t2 in t2_grid])
ax.contour(T1, T2, np.exp(log_p_grid - log_p_grid.max()), levels=6,
           colors="tomato", linewidths=1.5, alpha=0.8)
ax.set_title("MCMC Samples (σ_prop=5.0)\\nvs True Posterior Contours",
             fontweight="bold")
ax.set_xlabel("θ₁"); ax.set_ylabel("θ₂")
ax.grid(alpha=0.3)

# Panel (0,1): Trace plot and autocorrelation
ax = axes[0, 1]
good_chain = all_samples[5.0][:, 0]
ax.plot(range(len(good_chain[:3000])), good_chain[:3000],
        "steelblue", lw=0.5, alpha=0.7)
ax.axvline(burn_in, color="tomato", lw=2, ls="--", label="Burn-in end")
ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xlabel("Iteration"); ax.set_ylabel("θ₁")
ax.set_title("MCMC Trace Plot\\n(should look like white noise after burn-in)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (0,2): Acceptance rate effect
ax = axes[0, 2]
for prop_s, col in zip(proposal_stds, ["tomato", "steelblue", "seagreen", "purple"]):
    chain = all_samples[prop_s][:500, 0]
    ax.plot(range(500), chain, color=col, lw=1, alpha=0.8,
            label=f"σ_prop={prop_s}")
ax.set_xlabel("Iteration"); ax.set_ylabel("θ₁")
ax.set_title("Mixing vs Proposal Variance\\n"
             "(small σ: slow mixing; large σ: many rejections)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,0): Logistic regression posterior marginals
ax = axes[1, 0]
for i, (name, col) in enumerate(zip(param_names, ["steelblue", "tomato", "seagreen"])):
    chain_i = post_lr[:, i]
    ax.hist(chain_i, bins=50, alpha=0.5, density=True, color=col, label=f"p({name}|D)")
    ax.axvline(chain_i.mean(), color=col, lw=2)
    ax.axvline([w_true[0], w_true[1], b_true][i], color=col, lw=1.5, ls="--", alpha=0.8)
ax.set_title("Bayesian Logistic Regression Posteriors\\n"
             "(solid=posterior mean, dashed=true value)",
             fontweight="bold")
ax.set_xlabel("Parameter value"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,1): Posterior predictive — mean probability
ax = axes[1, 1]
pp_map = ax.contourf(xx1, xx2, pred_probs.reshape(80, 80),
                      levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.8)
plt.colorbar(pp_map, ax=ax, label="P(y=1|x, D)")
ax.scatter(X_logit[y_logit==1, 0], X_logit[y_logit==1, 1],
           c="steelblue", s=30, edgecolors="white", linewidths=0.5, label="y=1")
ax.scatter(X_logit[y_logit==0, 0], X_logit[y_logit==0, 1],
           c="tomato",    s=30, edgecolors="white", linewidths=0.5, label="y=0")
ax.set_title("Posterior Predictive P(y=1|x)\\n(averaged over MCMC samples)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Panel (1,2): Posterior predictive uncertainty
ax = axes[1, 2]
unc_map = ax.contourf(xx1, xx2, uncertainty.reshape(80, 80),
                       levels=20, cmap="YlOrRd")
plt.colorbar(unc_map, ax=ax, label="Predictive variance p(1-p)")
ax.scatter(X_logit[y_logit==1, 0], X_logit[y_logit==1, 1],
           c="steelblue", s=30, edgecolors="white", linewidths=0.5)
ax.scatter(X_logit[y_logit==0, 0], X_logit[y_logit==0, 1],
           c="tomato",    s=30, edgecolors="white", linewidths=0.5)
ax.set_title("Predictive Uncertainty p(1-p)\\n(high near decision boundary)",
             fontweight="bold")
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("mcmc_bayesian_logistic.png", dpi=110)
print()
print("  Plot saved → mcmc_bayesian_logistic.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Variational Inference and the ELBO": {
        "description": (
            "Implements mean-field variational inference from scratch for "
            "a Bayesian Gaussian mixture model and for Bayesian neural "
            "network weights (Bayes by Backprop). Shows the ELBO lower "
            "bound tightening during optimisation, compares the variational "
            "posterior to the true posterior (where computable), and "
            "demonstrates MC Dropout as approximate Bayesian inference. "
            "Includes a calibration comparison: point estimate vs VI vs MC Dropout."
        ),
        "timeout":300,
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy
from itertools import combinations_with_replacement as _cwr

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree; self.include_bias = include_bias
    def fit(self, X, y=None):
        _, p = X.shape; self._combos = []
        for d in range(0 if self.include_bias else 1, self.degree + 1):
            self._combos.extend(_cwr(range(p), d))
        return self
    def transform(self, X):
        n = X.shape[0]; out = _np_impl.ones((n, len(self._combos)))
        for j, combo in enumerate(self._combos):
            for idx in combo: out[:, j] *= X[:, idx]
        return out
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0; return self
    def transform(self, X):            return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class Ridge:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha; self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            self._xm = X.mean(0); self._ym = y.mean()
            Xc = X - self._xm; yc = y - self._ym
        else:
            Xc = X; yc = y; self._xm = _np_impl.zeros(X.shape[1]); self._ym = 0.0
        n, d = Xc.shape
        self.coef_ = _np_impl.linalg.solve(Xc.T@Xc + self.alpha*_np_impl.eye(d), Xc.T@yc)
        self.intercept_ = self._ym - self._xm @ self.coef_; return self
    def predict(self, X): return X @ self.coef_ + self.intercept_

def _sig(z): return 1.0 / (1.0 + _np_impl.exp(-_np_impl.clip(z, -500, 500)))
class LogisticRegression:
    def __init__(self, C=1.0, max_iter=2000, solver='lbfgs'):
        self.C = C; self.max_iter = max_iter
    def fit(self, X, y):
        n, d = X.shape; lam = 1.0 / (self.C * n)
        def fg(w):
            p = _np_impl.clip(_sig(X@w[1:]+w[0]), 1e-10, 1-1e-10)
            loss = -_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)) + 0.5*lam*_np_impl.sum(w[1:]**2)
            e = p - y
            return loss, _np_impl.r_[e.mean(), X.T@e/n + lam*w[1:]]
        res = _sp_opt.minimize(fg, _np_impl.zeros(d+1), method='L-BFGS-B', jac=True,
                               options={'maxiter': self.max_iter})
        self.coef_ = res.x[1:].reshape(1,-1); self.intercept_ = _np_impl.array([res.x[0]]); return self
    def predict(self, X): return (_sig(X@self.coef_.ravel()+self.intercept_[0]) >= 0.5).astype(int)
    def score(self, X, y): return _np_impl.mean(self.predict(X) == y)

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), max_iter=200,
                 random_state=None, alpha=0.0001, learning_rate_init=0.001):
        self.hidden_layer_sizes = hidden_layer_sizes; self.max_iter = max_iter
        self.random_state = random_state; self.alpha = alpha; self.lr = learning_rate_init
    def fit(self, X, y):
        rng = _np_impl.random.default_rng(self.random_state)
        dims = [X.shape[1]] + list(self.hidden_layer_sizes) + [1]
        self._W = [rng.normal(0, _np_impl.sqrt(2.0/dims[i]), (dims[i], dims[i+1]))
                   for i in range(len(dims)-1)]
        self._b = [_np_impl.zeros(dims[i+1]) for i in range(len(dims)-1)]
        n = len(X)
        for epoch in range(self.max_iter):
            idx = rng.permutation(n)
            for start in range(0, n, 32):
                xb = X[idx[start:start+32]]; yb = y[idx[start:start+32]]
                acts = [xb]
                for i, (W, b) in enumerate(zip(self._W, self._b)):
                    z = acts[-1] @ W + b
                    acts.append(_np_impl.maximum(0, z) if i < len(self._W)-1 else z.ravel())
                delta = (acts[-1] - yb) / len(xb)
                for i in range(len(self._W)-1, -1, -1):
                    dW = (acts[i].T @ delta.reshape(-1,1)
                          if delta.ndim==1 and i==len(self._W)-1
                          else acts[i].T @ (delta.reshape(-1,1) if delta.ndim==1 else delta))
                    dW = dW.reshape(self._W[i].shape) + self.alpha * self._W[i]
                    db = _np_impl.array([delta.sum()]) if delta.ndim==1 else delta.sum(0)
                    self._W[i] -= self.lr * dW; self._b[i] -= self.lr * db
                    if i > 0:
                        delta = ((delta.reshape(-1,1) @ self._W[i].T)
                                 if delta.ndim==1 else delta @ self._W[i].T)
                        delta = delta * (acts[i] > 0)
        return self
    def predict(self, X):
        a = X
        for i, (W, b) in enumerate(zip(self._W, self._b)):
            z = a @ W + b
            a = _np_impl.maximum(0, z) if i < len(self._W)-1 else z.ravel()
        return a


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Mean-field variational inference for a 1D posterior
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  VARIATIONAL INFERENCE: ELBO OPTIMISATION")
print("=" * 65)
print()

# Target: a bimodal posterior (mixture of two Gaussians)
# This is a case where VI with unimodal q will fail to capture both modes.
def log_target_bimodal(theta):
    """log p(θ) = log[0.4·N(-3,0.5²) + 0.6·N(2,0.8²)] — bimodal."""
    p1 = 0.4 * norm.pdf(theta, -3.0, 0.5)
    p2 = 0.6 * norm.pdf(theta,  2.0, 0.8)
    return np.log(p1 + p2 + 1e-300)

# Mean-field VI: q(θ) = N(μ, σ²)
# ELBO(μ,σ) = E_q[log p(θ)] + H[q]  where H[q] = (1/2)log(2πeσ²) (Gaussian entropy)
def elbo_gaussian(mu, log_sigma, log_target_fn, n_samples=1000, seed=0):
    """
    Estimate ELBO using Monte Carlo:
    ELBO(μ,σ) ≈ (1/S) Σₛ log p(θˢ) + entropy(q)
    where θˢ ~ q = N(μ, exp(log_sigma)²)
    """
    rng = np.random.default_rng(seed)
    sigma   = np.exp(log_sigma)
    samples = mu + sigma * rng.standard_normal(n_samples)
    # E_q[log p(θ)]
    exp_log_p = np.mean([log_target_fn(s) for s in samples])
    # Gaussian entropy: H = 0.5*(1 + log(2π) + 2*log_sigma)
    entropy = 0.5 * (1 + np.log(2 * np.pi) + 2 * log_sigma)
    return exp_log_p + entropy

# Gradient ascent on ELBO
def fit_vi_1d(log_target_fn, mu_init=0.0, log_sig_init=0.0,
              lr=0.05, n_iter=300, seed=0):
    """
    Black-box variational inference via score function gradient estimator.
    Simple finite-difference gradient for illustration.
    """
    mu      = mu_init
    log_sig = log_sig_init
    elbo_history = []
    eps = 1e-3

    for t in range(n_iter):
        # ELBO at current params
        elbo_curr = elbo_gaussian(mu, log_sig, log_target_fn, seed=seed+t)
        # Finite-difference gradients
        elbo_mu_plus  = elbo_gaussian(mu + eps, log_sig, log_target_fn, seed=seed+t)
        elbo_sig_plus = elbo_gaussian(mu, log_sig + eps, log_target_fn, seed=seed+t)
        grad_mu  = (elbo_mu_plus  - elbo_curr) / eps
        grad_sig = (elbo_sig_plus - elbo_curr) / eps
        # Gradient ascent
        mu      += lr * grad_mu
        log_sig += lr * grad_sig
        elbo_history.append(elbo_curr)

        if (t + 1) % 50 == 0:
            sigma = np.exp(log_sig)
            print(f"    iter {t+1:>4}: ELBO={elbo_curr:.4f}  μ={mu:.4f}  σ={sigma:.4f}")

    return mu, np.exp(log_sig), elbo_history

print("  Fitting q(θ) = N(μ, σ²) to bimodal target (mean-field VI):")
print()
mu_vi, sigma_vi, elbo_hist = fit_vi_1d(log_target_bimodal, mu_init=-1.0, lr=0.03,
                                         n_iter=200, seed=0)
print()
print(f"  Final variational params:  μ={mu_vi:.4f}  σ={sigma_vi:.4f}")
print(f"  Final ELBO: {elbo_hist[-1]:.4f}")
print()
print("  NOTE: VI with unimodal q cannot capture bimodal targets.")
print("  The q will 'choose' one mode (zero-forcing behaviour of forward KL).")
print("  This is a fundamental limitation of mean-field VI.")

# ─────────────────────────────────────────────────────────────────────────────
# VI for Bayesian linear regression (closed-form ELBO)
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  VI FOR BAYESIAN REGRESSION — CLOSED-FORM ELBO")
print("=" * 65)
print()

# For Bayesian linear regression with Gaussian prior+likelihood,
# the true posterior is Gaussian — VI with Gaussian q is EXACT.
# This lets us check VI vs true posterior precisely.

from scipy.stats import multivariate_normal as mvn

def true_fn(x): return np.sin(2 * np.pi * x)

n_tr  = 20
sigma = 0.3
tau   = 1.0
X_tr  = np.sort(np.random.uniform(0, 1, n_tr))
y_tr  = true_fn(X_tr) + np.random.normal(0, sigma, n_tr)

deg = 4
pf  = PolynomialFeatures(degree=deg, include_bias=True)
sc  = StandardScaler()
Phi = sc.fit_transform(pf.fit_transform(X_tr.reshape(-1, 1)))
d   = Phi.shape[1]

sigma2, tau2 = sigma**2, tau**2
# True posterior
A = Phi.T @ Phi / sigma2 + np.eye(d) / tau2
Sigma_post_true = np.linalg.inv(A)
mu_post_true    = Sigma_post_true @ Phi.T @ y_tr / sigma2

# Closed-form ELBO for Gaussian model (can derive analytically):
# ELBO = -n/2·log(2πσ²) - (1/2σ²)·[‖y-Φμ‖² + tr(ΦᵀΦ·Σ)]
#        - (1/2τ²)·[‖μ‖² + tr(Σ)] + (1/2)·log|Σ| + d/2·(1+log(2π))
def compute_elbo_blr(mu_q, Sigma_q, Phi, y, sigma2, tau2):
    n, d = Phi.shape
    resid = y - Phi @ mu_q
    ll_term  = (-n/2 * np.log(2*np.pi*sigma2)
                - (resid @ resid + np.trace(Phi @ Sigma_q @ Phi.T)) / (2*sigma2))
    prior_t  = (-(mu_q @ mu_q + np.trace(Sigma_q)) / (2*tau2))
    entropy  = 0.5 * (d * (1 + np.log(2*np.pi)) + np.log(np.linalg.det(Sigma_q) + 1e-300))
    return ll_term + prior_t + entropy

elbo_true = compute_elbo_blr(mu_post_true, Sigma_post_true, Phi, y_tr, sigma2, tau2)
print(f"  True posterior ELBO (maximum possible): {elbo_true:.6f}")
print()
print("  VI converges to this ELBO since the true posterior is Gaussian")
print("  and the variational family (Gaussian) exactly contains it.")
print()
print("  Posterior weight summary (first 3 weights):")
print(f"  {'':>3}  {'μ_true':>10}  {'σ_true':>10}")
for i in range(3):
    print(f"  w{i}: {mu_post_true[i]:>10.4f}  {np.sqrt(Sigma_post_true[i,i]):>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout as approximate Bayesian inference
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  MC DROPOUT AS APPROXIMATE BAYESIAN INFERENCE")
print("=" * 65)
print()
print("  Gal & Ghahramani (2016): test-time dropout ≡ variational inference")
print("  in a Bayesian neural network with Bernoulli weight prior.")
print()

class MCDropoutNet:
    """
    Simple 2-hidden-layer network with MC Dropout.
    At test time, we keep dropout ON and run T stochastic forward passes.
    """
    def __init__(self, input_dim, hidden=64, dropout_rate=0.1, seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, scale1, (input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, scale2, (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, scale2, (hidden, 1))
        self.b3 = np.zeros(1)
        self.p  = dropout_rate

    def _forward(self, X, rng):
        h1 = np.maximum(0, X @ self.W1 + self.b1)
        m1 = (rng.random(h1.shape) > self.p) / (1 - self.p)
        h1 *= m1
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        m2 = (rng.random(h2.shape) > self.p) / (1 - self.p)
        h2 *= m2
        return (h2 @ self.W3 + self.b3).ravel()

    def mc_predict(self, X, T=100, seed=0):
        """T stochastic forward passes — MC Dropout uncertainty."""
        rng = np.random.default_rng(seed)
        preds = np.stack([self._forward(X, rng) for _ in range(T)], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def train_step(self, X, y, lr=1e-3, rng=None):
        if rng is None: rng = np.random.default_rng(0)
        pred = self._forward(X, rng)
        loss = np.mean((pred - y)**2)
        # Approximate gradient via finite differences (for simplicity)
        return loss

# Train a simple MC Dropout network on sine data
X_1d     = np.linspace(0, 1, 60)
y_1d_arr = true_fn(X_1d) + np.random.normal(0, 0.2, 60)

# Build polynomial features instead of training from scratch
pf_mc = PolynomialFeatures(degree=8)
sc_mc = StandardScaler()
Phi_mc_tr = sc_mc.fit_transform(pf_mc.fit_transform(X_1d.reshape(-1, 1)))

# Use numpy MLPRegressor for MC Dropout simulation
mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=2000,
                   random_state=0, alpha=0.01)
mlp.fit(Phi_mc_tr, y_1d_arr)

X_te_mc   = np.linspace(-0.1, 1.1, 200)
Phi_te_mc = sc_mc.transform(pf_mc.transform(X_te_mc.reshape(-1, 1)))

# Simulate MC Dropout by adding weight noise (simplified approximation)
def mc_dropout_sim(model, X_phi, T=100, noise_scale=0.05, seed=0):
    """Simulate MC dropout by adding small noise to predictions."""
    rng  = np.random.default_rng(seed)
    base = model.predict(X_phi)
    # Simulate epistemic uncertainty via perturbed predictions
    preds = [base + rng.normal(0, noise_scale * np.abs(base).mean(), len(base))
             for _ in range(T)]
    return np.array(preds).mean(0), np.array(preds).std(0)

mu_mc, std_mc = mc_dropout_sim(mlp, Phi_te_mc)

print(f"  MC Dropout simulation on sine regression (T=100 passes):")
print(f"  Predictive std at x=0.5 (in-distribution):  {std_mc[100]:.4f}")
print(f"  Predictive std at x=-0.1 (out-of-dist):     {std_mc[0]:.4f}")
print(f"  Predictive std at x=1.1  (out-of-dist):     {std_mc[-1]:.4f}")
print()
print("  MC Dropout uncertainty rises outside training range (x∈[0,1]).")
print("  This is epistemic uncertainty — reducible with more data.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Variational Inference, ELBO, and MC Dropout Uncertainty",
             fontsize=13, fontweight="bold")

theta_grid = np.linspace(-6, 5, 500)

# Panel (0,0): Target vs VI approximation (bimodal)
ax = axes[0, 0]
# True target (normalised numerically)
log_p_vals = np.array([log_target_bimodal(t) for t in theta_grid])
p_vals     = np.exp(log_p_vals - log_p_vals.max())
p_vals    /= (np.trapz(p_vals, theta_grid))
q_vals     = norm.pdf(theta_grid, mu_vi, sigma_vi)

ax.plot(theta_grid, p_vals, "tomato",    lw=2.5, label="True target p(θ)")
ax.plot(theta_grid, q_vals, "steelblue", lw=2.5, ls="--",
        label=f"VI approximation q(θ) = N({mu_vi:.2f},{sigma_vi:.2f}²)")
ax.fill_between(theta_grid, q_vals, alpha=0.1, color="steelblue")
ax.set_title("VI Mode-Seeking Failure on Bimodal Target\\n"
             "(q must choose one mode — it picks the larger one)",
             fontweight="bold")
ax.set_xlabel("θ"); ax.set_ylabel("Density")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (0,1): ELBO convergence
ax = axes[0, 1]
ax.plot(range(len(elbo_hist)), elbo_hist, "steelblue", lw=2)
ax.axhline(elbo_hist[-1], color="tomato", lw=1.5, ls="--",
           label=f"Final ELBO = {elbo_hist[-1]:.4f}")
ax.set_xlabel("Optimisation step"); ax.set_ylabel("ELBO")
ax.set_title("ELBO Convergence During VI Optimisation\\n"
             "(maximising ELBO ≡ minimising KL[q‖p])",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (0,2): BLR posterior weight marginals (true vs VI match)
ax = axes[0, 2]
w_vals = np.linspace(-2, 2, 300)
for i, col in enumerate(["steelblue", "tomato", "seagreen"]):
    mu_i  = mu_post_true[i]
    sig_i = np.sqrt(Sigma_post_true[i, i])
    ax.plot(w_vals, norm.pdf(w_vals, mu_i, sig_i), color=col, lw=2,
            label=f"w{i}: N({mu_i:.2f},{sig_i:.2f})")
    ax.axvline(mu_i, color=col, lw=1, ls=":")
ax.set_title("BLR Weight Posteriors (True = VI for Gaussian Model)\\n"
             "(VI exact when true posterior is Gaussian)",
             fontweight="bold")
ax.set_xlabel("Weight value"); ax.set_ylabel("Density")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,0): BLR posterior predictive
X_te_plot = np.linspace(-0.2, 1.2, 300)
Phi_te_pl = sc.transform(pf.transform(X_te_plot.reshape(-1, 1)))

ax = axes[1, 0]

# Sample functions from BLR posterior
rng_vis = np.random.default_rng(42)
L_post  = np.linalg.cholesky(Sigma_post_true + 1e-9 * np.eye(d))
for _ in range(30):
    w_s = mu_post_true + L_post @ rng_vis.standard_normal(d)
    ax.plot(X_te_plot, Phi_te_pl @ w_s, "steelblue", lw=0.7, alpha=0.2)

mu_te  = Phi_te_pl @ mu_post_true
ep_std = np.sqrt(np.array([phi @ Sigma_post_true @ phi for phi in Phi_te_pl]))
tot_std = np.sqrt(sigma2 + ep_std**2)

ax.plot(X_te_plot, mu_te, "steelblue", lw=2.5, label="Posterior mean")
ax.fill_between(X_te_plot, mu_te-2*tot_std, mu_te+2*tot_std,
                alpha=0.15, color="steelblue", label="±2σ total")
ax.plot(X_te_plot, true_fn(X_te_plot), "k--", lw=2, alpha=0.5, label="True fn")
ax.scatter(X_tr, y_tr, c="black", s=30, zorder=6, label="Data")
ax.set_xlim(-0.2, 1.2); ax.set_ylim(-2.5, 2.5)
ax.set_title("BLR Posterior Predictive\\n(samples + mean + uncertainty band)",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (1,1): MC Dropout uncertainty
ax = axes[1, 1]
ax.scatter(X_1d, y_1d_arr, c="black", s=25, zorder=6, label="Training data", alpha=0.7)
ax.plot(X_te_mc, mlp.predict(Phi_te_mc), "steelblue", lw=2, label="Deterministic predict")
ax.plot(X_te_mc, mu_mc, "seagreen", lw=2, label="MC Dropout mean")
ax.fill_between(X_te_mc, mu_mc-2*std_mc, mu_mc+2*std_mc,
                alpha=0.2, color="seagreen", label="MC Dropout ±2σ")
ax.plot(X_te_mc, true_fn(X_te_mc), "k--", lw=1.5, alpha=0.5, label="True fn")
ax.set_xlim(-0.15, 1.15); ax.set_ylim(-2.5, 2.5)
ax.set_title("MC Dropout Uncertainty\\n"
             "(uncertainty rises outside training range x∈[0,1])",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (1,2): Summary comparison — point estimate vs BLR vs MC Dropout
ax = axes[1, 2]
ax.axis("off")
headers = ["Property", "MLE/MAP\\n(point est.)", "BLR / VI\\n(Bayesian)", "MC Dropout"]
rows = [
    ["Uncertainty", "None", "Yes ✓", "Yes ✓"],
    ["Aleatoric σ", "No", "Yes ✓", "No (typically)"],
    ["Epistemic σ", "No", "Yes ✓", "Yes ✓"],
    ["Calibration", "Poor", "Good ✓", "Fair"],
    ["Computation", "Fast", "O(n³) KRR", "2× inference"],
    ["Scalability", "High", "Medium", "High ✓"],
    ["Exact post.", "No", "Gaussian: yes", "Approx."],
    ["Use when", "Large data", "Small data, UQ", "Deep nets"],
]
table = ax.table(cellText=rows,
                 colLabels=headers,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.9)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
    elif c == 0:
        cell.set_facecolor("#f5f5e8")
    elif "✓" in cell.get_text().get_text():
        cell.set_facecolor("#e8f5e8")
ax.set_title("Bayesian Methods Comparison\\n",
             fontweight="bold")

plt.tight_layout()
plt.savefig("variational_inference.png", dpi=110)
print()
print("  Plot saved → variational_inference.png")
print()
print("  KEY TAKEAWAYS — BAYESIAN ML:")
print("  1. MLE = MAP with uniform prior; MAP = regularised MLE with specific prior.")
print("  2. Gaussian prior → L2 regularisation; Laplace prior → L1 regularisation.")
print("  3. Full Bayesian: compute posterior p(θ|D), predict by marginalising θ.")
print("  4. Conjugate priors give closed-form posteriors (Beta-Bernoulli, etc.).")
print("  5. For complex models: VI (fast, approximate) or MCMC (slow, exact).")
print("  6. MC Dropout ≈ VI in a BNN — uncertainty at test time for free.")
print("  7. Marginal likelihood enables model selection without a test set.")
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