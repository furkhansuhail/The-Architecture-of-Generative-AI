"""
Hyperparameter Tuning
=====================

A hyperparameter is any value that controls the learning process but
is not itself learned by gradient descent. Learning rate, batch size,
number of layers, dropout rate, weight decay — none of these are
optimised by the loss function. They must be chosen by the practitioner.
The process of choosing them well is hyperparameter tuning: the outer
optimisation loop that sits above gradient descent and shapes the
entire training trajectory.

Hyperparameter tuning is not guesswork made systematic. It is a
principled search problem with its own algorithms, failure modes, and
best practices. Done badly, it wastes compute and produces misleading
results. Done well, it is often the difference between a model that
works and one that does not.

"""
import textwrap
import re

TOPIC_NAME   = "Hyperparameter Tuning"
DISPLAY_NAME = "10 · Hyperparameter Tuning"
ICON         = "🎛️"
SUBTITLE     = "Grid · Random · Bayesian · LR Finder · Population · Schedules"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — THE HYPERPARAMETER LANDSCAPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Taxonomy of Hyperparameters

Not all hyperparameters are equal. They differ in sensitivity, interaction
effects, and the cost of getting them wrong:

    TIER 1 — Critical (largest impact, tune first):
    Learning rate:          The single most important hyperparameter.
                            An order-of-magnitude error kills training.
    Batch size:             Interacts with LR via the linear scaling rule.
    Weight decay (λ):       Primary regularisation strength.
    Architecture depth/width: Determines model capacity.

    TIER 2 — Important (tune after Tier 1 is stable):
    LR schedule type:       Cosine, linear, step, warmup duration.
    Dropout rate:           Secondary regularisation.
    Optimiser choice:       Adam vs AdamW vs SGD (rarely changes outcome
                            once Tier 1 is set).
    β₁, β₂ (Adam moments): Almost always left at defaults (0.9, 0.999).
    Activation function:    Usually determined by architecture convention.

    TIER 3 — Fine-tuning (marginal impact, tune last):
    ε (Adam epsilon):       Default 1e-8 rarely needs changing.
    Gradient clip value:    Default 1.0 works for most transformers.
    Warmup steps:           Tune within a factor of 2 of the default.
    Label smoothing α:      Default 0.1 is robust.
    Augmentation magnitude: Tune after architecture and LR are fixed.


### The Hyperparameter Search Space

Each hyperparameter lives in a SEARCH SPACE — the range of values
considered during tuning. Defining the search space correctly is as
important as choosing the search algorithm.

    CONTINUOUS hyperparameters (real-valued):
    Examples: learning rate, weight decay, dropout rate.
    Key insight: these are almost always best searched on a LOG SCALE.

    Learning rate example:
    Linear scale search:  [0.0001, 0.0002, 0.0003, ..., 0.001]
    Log scale search:     [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    On a linear scale, most of the budget is spent in the high end of
    the range (e.g., 0.0005 to 0.001) where differences matter less.
    On a log scale, each order of magnitude gets equal coverage —
    which is correct, since LR sensitivity is multiplicative, not additive.

    Rule: ALWAYS search LR, weight decay, and dropout on log scale.

    DISCRETE hyperparameters (integer-valued):
    Examples: number of layers, hidden dimension, attention heads,
    batch size.
    Search over a finite set of valid values.
    Powers of 2 for batch size and hidden dimension (hardware efficiency).

    CATEGORICAL hyperparameters:
    Examples: optimiser type, activation function, normalisation type.
    Search over a discrete unordered set.
    Cannot interpolate between categorical values.

    CONDITIONAL hyperparameters:
    A hyperparameter that only exists if another takes a specific value.
    Example: β₁ and β₂ only exist if the optimiser is Adam.
    α (negative slope) only exists if the activation is Leaky ReLU.
    Search algorithms must respect these dependencies.


### The Cost of Hyperparameter Evaluation

Every hyperparameter configuration must be EVALUATED — trained to
convergence (or to a proxy metric) and the result measured.
The cost of one evaluation determines the entire tuning strategy:

    Cheap evaluation (seconds to minutes):
    Small model, small dataset, or proxy metric (few epochs).
    Can afford: grid search, random search, many trials.

    Moderate evaluation (minutes to hours):
    Medium model, medium dataset, full training run.
    Must afford: random search, Bayesian optimisation, early stopping.

    Expensive evaluation (hours to days):
    Large model, large dataset, full pre-training run.
    Can only afford: 5-20 total trials.
    Use: expert priors, learning rate finder, population-based training.

    The tuning strategy must match the evaluation cost.
    Bayesian optimisation on a problem where each eval takes 3 days
    allows only ~5 trials over a 2-week budget — Bayesian optimisation
    cannot fit a good surrogate from 5 points. Expert initialisation wins.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — GRID SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### How Grid Search Works

Grid search exhaustively evaluates every combination of a discrete
set of hyperparameter values:

    Define a grid:
    learning_rate = [1e-4, 1e-3, 1e-2]
    weight_decay  = [1e-4, 1e-3, 1e-2]
    dropout       = [0.1, 0.3, 0.5]

    Total trials = 3 × 3 × 3 = 27

    Diagram 1 — Grid Search Coverage (2D example):

    weight_decay
    1e-2  │  ●  ●  ●
          │
    1e-3  │  ●  ●  ●
          │
    1e-4  │  ●  ●  ●
          └──────────────── learning_rate
             1e-4 1e-3 1e-2

    Every combination is evaluated. The grid covers the space uniformly.


### Why Grid Search Fails in High Dimensions

The CURSE OF DIMENSIONALITY makes grid search impractical for more
than 2-3 hyperparameters:

    2 hyperparameters, 5 values each: 5² = 25 trials
    3 hyperparameters, 5 values each: 5³ = 125 trials
    4 hyperparameters, 5 values each: 5⁴ = 625 trials
    5 hyperparameters, 5 values each: 5⁵ = 3,125 trials
    10 hyperparameters, 5 values each: 5¹⁰ = ~10 million trials

    The combinatorial explosion makes grid search intractable.

    Additionally, grid search WASTES evaluations on unimportant dimensions.
    Research (Bergstra & Bengio, 2012) showed that in practice, only
    1-2 hyperparameters have strong impact on performance. The others
    are largely irrelevant within their typical ranges.

    In grid search, if 8 out of 10 hyperparameters are unimportant,
    8/10 of the grid dimensions contribute nothing useful — every
    unique value for the important dimensions is repeated many times
    with different values of unimportant ones.

    Diagram 2 — Grid Search Waste (unimportant dimension):

    weight_decay (unimportant)
    1e-2  │  ●  ●  ●   ← same performance as row below
          │
    1e-3  │  ●  ●  ●   ← 3 points at this LR; only 1 needed
          │
    1e-4  │  ●  ●  ●   ← same performance as row above
          └──────────────── learning_rate (important)
             1e-4 1e-3 1e-2

    If weight_decay doesn't matter in this range, all rows produce
    the same result. Grid search used 9 evaluations to get 3
    unique LR datapoints. Random search would use 9 trials to get
    9 unique LR values — 3× more informative.


### When to Use Grid Search

    ✓ Very few hyperparameters (≤ 2-3)
    ✓ Small, well-understood search space
    ✓ Evaluation is cheap (seconds per trial)
    ✓ You need to reproduce a specific grid from a paper
    ✓ Ablation studies: "try all combinations of these 2 options"

    ✗ More than 3 hyperparameters
    ✗ Expensive evaluation
    ✗ Continuous search spaces


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — RANDOM SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### How Random Search Works

Random search independently samples each hyperparameter from its
search distribution for each trial. No structure or memory between trials:

    For each trial:
        lr    ~ LogUniform(1e-4, 1e-1)
        wd    ~ LogUniform(1e-5, 1e-1)
        drop  ~ Uniform(0.0, 0.5)
        depth ~ UniformInt(2, 8)
    Train and evaluate. Repeat for N trials.
    Return the configuration with the best validation metric.


### Why Random Search Beats Grid Search: The Bergstra-Bengio Result

Bergstra & Bengio (2012) proved analytically and empirically that
RANDOM SEARCH is more efficient than grid search when only a few
hyperparameters are important:

    Diagram 3 — Grid vs Random for One Important Dimension:

    unimportant
    dimension
          │
    high  │  ●  ●  ●      Random search:
          │               9 trials → 9 UNIQUE values of
    mid   │  ●  ●  ●      the important dimension
          │
    low   │  ●  ●  ●
          └──────────────── important dimension
           3 vals covered    9 unique values covered

    GRID search:                        RANDOM search:
    ┌─────────────────────────────┐     ┌─────────────────────────────┐
    │  ●    ●    ●    ●    ●     │     │ ●  ●       ●  ●   ●        │
    │  ●    ●    ●    ●    ●     │     │    ●  ●  ●       ●  ●      │
    │  ●    ●    ●    ●    ●     │     │ (no structure — random)     │
    │  ●    ●    ●    ●    ●     │     │                             │
    │  ●    ●    ●    ●    ●     │     └─────────────────────────────┘
    └─────────────────────────────┘     25 unique x-values sampled
    5 unique x-values, 5 repeats each   → 5× more coverage of the
    → 25 trials for 5 x-values             important dimension

    With 1 important dimension (x) and 1 unimportant dimension (y):
    Grid search with 25 trials:   5 unique values of x
    Random search with 25 trials: ~25 unique values of x (uniformly)

    In practice, with k out of d hyperparameters being important (k < d):
    Random search uses its budget entirely on unique combinations of the
    important dimensions. Grid search wastes most of its budget repeating
    combinations that differ only in unimportant dimensions.

    Empirical result: random search typically finds equally good or
    better configurations than grid search with the same budget,
    and requires far fewer trials to do so.


### Sampling Distributions for Random Search

Choosing the right SAMPLING DISTRIBUTION is as important as choosing
the search algorithm. The distribution should reflect your prior
belief about where good values lie:

    LOG-UNIFORM distribution:
    Sample log(x) ~ Uniform(log(a), log(b))
    ↔ x ~ Uniform(a, b) on the log scale.
    Use for: learning rate, weight decay, epsilon — any parameter
    where relative changes (×2, ×10) matter, not absolute changes (+0.001).

    Implementation: x = 10 ** np.random.uniform(np.log10(a), np.log10(b))

    Example: LR ~ LogUniform(1e-4, 1e-1)
    Samples equally from: [1e-4, 1e-3], [1e-3, 1e-2], [1e-2, 1e-1]
    Each order of magnitude gets ~equal probability.

    UNIFORM distribution:
    x ~ Uniform(a, b)
    Use for: dropout rate, layer sizes (when scale doesn't matter),
    mixing coefficients.

    INTEGER-UNIFORM:
    x ~ UniformInt(a, b)
    Use for: number of layers, attention heads, vocabulary size.

    CATEGORICAL:
    x ~ Categorical({opt1: p1, opt2: p2, ...})
    Use for: optimiser type, activation function, normalisation type.
    Can weight by prior probability (e.g., Adam more likely than SGD).

    TRUNCATED NORMAL:
    x ~ TruncatedNormal(μ, σ, a, b)
    Use when you have a strong prior about the best value (μ) and want
    to explore around it but not too far.


### Practical Random Search Protocol

    1. Set the budget: decide the maximum number of trials.
       Rule of thumb:
       ≤ 20 trials:  can only tune 1-2 hyperparameters reliably.
       20-100 trials: can tune 3-5 hyperparameters.
       100+ trials:   can tune 5+ hyperparameters or use Bayesian opt.

    2. Define the search space using informed priors:
       Do not search blindly. Use published results, the LR finder,
       or scaling laws to narrow the space before random search.

    3. Run trials in parallel (if compute is available):
       Independent trials parallelise perfectly. 100 trials on 100
       machines takes the same wall time as 1 trial on 1 machine.

    4. Use EARLY STOPPING within each trial to kill unpromising runs:
       A trial that has not improved by epoch 5 is unlikely to
       be the best by epoch 100. Stopping it early releases compute
       for more trials.
       → See Successive Halving / Hyperband (Part 6).

    5. After random search finds a good region, do a LOCAL refinement:
       Narrow the search space around the best trial found.
       Re-run random search or a coarse grid within the narrowed space.

    6. Report the best VAL metric from random search, but evaluate
       the final selected model on the TEST set exactly once.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — BAYESIAN OPTIMISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Core Idea

Bayesian optimisation (BO) treats hyperparameter tuning as a
SEQUENTIAL DECISION PROBLEM. It maintains a PROBABILISTIC MODEL
(the surrogate) of how performance varies across the search space,
and uses this model to choose which configuration to evaluate next.

    Unlike random search (which ignores all previous results when
    choosing the next point), BO USES past results to intelligently
    direct future search:

    Diagram 4 — Bayesian Optimisation Loop:

    ┌────────────────────────────────────────────────────────────────┐
    │ 1. Evaluate a few initial configurations (random seed trials)  │
    │                  │                                             │
    │                  ▼                                             │
    │ 2. Fit a surrogate model to all (config, performance) pairs    │
    │    The surrogate predicts: E[f(x)] and Var[f(x)]              │
    │                  │                                             │
    │                  ▼                                             │
    │ 3. Use an acquisition function to select the next config:      │
    │    find x* = argmax Acquisition(x; surrogate)                 │
    │                  │                                             │
    │                  ▼                                             │
    │ 4. Evaluate f(x*) — actually train with that config           │
    │                  │                                             │
    │                  ▼                                             │
    │ 5. Add (x*, f(x*)) to the dataset. Return to step 2.         │
    └────────────────────────────────────────────────────────────────┘

    The key insight: the surrogate model is MUCH CHEAPER to evaluate
    than the real objective (no training required). The acquisition
    function can be maximised with gradient descent or dense sampling
    over the surrogate in milliseconds, allowing intelligent selection
    of the next configuration without burning GPU hours.


### The Surrogate Model: Gaussian Processes

The standard surrogate is a GAUSSIAN PROCESS (GP):

    A Gaussian Process defines a distribution over functions:
    f(x) ~ GP(μ(x), k(x, x'))

    Where:
        μ(x) = mean function (often zero)
        k(x, x') = covariance kernel (measures similarity between configs)

    After observing data D = {(x₁, f₁), ..., (xₙ, fₙ)}, the GP gives
    a posterior distribution over f(x*) for any new point x*:

    p(f(x*) | D) = N(μ_post(x*), σ²_post(x*))

    μ_post(x*): our best ESTIMATE of performance at x*
    σ²_post(x*): our UNCERTAINTY about performance at x*

    Diagram 5 — Gaussian Process Surrogate Model:

    Performance
       │
       │         ┌───┐           ┌───┐
       │         │   │           │   │  ← uncertainty bounds (2σ)
    0.9│    ●────●   ●───────────●   │
       │                             │   ● = observed trials
    0.7│  ●                          ●─
       │                               ← mean prediction (surrogate)
    0.5│
       └──────────────────────────────── hyperparameter x
              ↑ known region           ↑ unexplored region
           low uncertainty            high uncertainty

    The GP is confident (narrow uncertainty) where it has seen data.
    It is uncertain (wide uncertainty) where it has not.
    The acquisition function uses both the predicted mean AND the
    uncertainty to decide where to evaluate next.


### Acquisition Functions

The acquisition function converts the surrogate's predictions into
a score for each candidate configuration:

    EXPECTED IMPROVEMENT (EI) — the standard choice:
    EI(x) = E[max(0, f(x) - f_best)]

    At each point x, EI computes the expected amount by which f(x)
    would improve over the current best observed value f_best.

    EI(x) is HIGH when:
    - μ_post(x) is much better than f_best (exploitation — confident improvement)
    - σ_post(x) is large (exploration — uncertain but potentially better)

    EI(x) is LOW when:
    - μ_post(x) is worse than f_best AND σ_post(x) is small
      (confident it's bad — don't evaluate)

    EXPLORATION-EXPLOITATION TRADEOFF:
    The acquisition function balances:
    EXPLOITATION: try configs where the model predicts high performance.
    EXPLORATION:  try configs where uncertainty is high (could be better
                  than expected, or could reveal important information).

    Diagram 6 — EI Acquisition Surface:

    Performance
       │    surrogate mean
       │     ●─────●─────────────●──●
       │   ●     uncertainty band ↑
       │         ┌────────────────┐
       │         │                │
       └─────────────────────────────── hyperparameter x

    EI │                     ●    ← maximum EI: uncertain + potentially good
       │              ●
       │        ●
       │  ●
       └─────────────────────────────── hyperparameter x
              ↑ evaluated             ↑ next point to evaluate

    Other acquisition functions:
    UCB (Upper Confidence Bound):  μ(x) + κ·σ(x)  [simpler, explicit κ tradeoff]
    PI  (Probability of Improvement): P(f(x) > f_best + ε)

    EI is the most widely used and is the default in most libraries.


### Tree-structured Parzen Estimator (TPE)

The dominant alternative to GP-based BO for high-dimensional spaces:

    Instead of modelling p(f | x) directly (like a GP does),
    TPE models TWO distributions:
    l(x) = p(x | f < f_best_quantile)  [density of configs in the "good" region]
    g(x) = p(x | f ≥ f_best_quantile)  [density of configs in the "bad" region]

    The acquisition function becomes: l(x) / g(x)
    → Choose x where good configs are dense and bad configs are sparse.

    TPE advantages over GP:
    - Scales to higher dimensions (GPs scale as O(n³) with data points)
    - Handles categorical and conditional hyperparameters naturally
    - Faster to fit and evaluate
    - Works well with even imprecise or noisy evaluations

    TPE is the default algorithm in Optuna and was the algorithm
    behind the highly successful Hyperopt library.
    For most practical deep learning tuning, TPE (via Optuna) is the
    recommended Bayesian optimisation approach.


### When Bayesian Optimisation Wins (and When It Doesn't)

    BO EXCELS when:
    ✓ Each evaluation is expensive (hours)
    ✓ The search space has ≤ 20 dimensions
    ✓ The objective is smooth (performance varies smoothly with hyperparams)
    ✓ You can afford 20-200 total evaluations
    ✓ The search is sequential (one trial at a time)

    BO STRUGGLES when:
    ✗ Very few evaluations are available (< 10) — not enough data to fit
      a useful surrogate. Use expert priors instead.
    ✗ Very many hyperparameters (> 20 dimensions) — GP surrogate fails.
      Use TPE or random search.
    ✗ Highly noisy objectives (random seed variance dominates signal).
    ✗ Parallel evaluation at scale — BO is inherently sequential.
      Parallel BO exists but is complex; random search parallelises trivially.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — THE LEARNING RATE FINDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why LR Is Different From Other Hyperparameters

The learning rate stands apart from other hyperparameters because:
1. It is the most sensitive hyperparameter by far.
2. It has a characteristic LOSS VS LR CURVE that reveals the optimal
   value directly, without requiring a full training run.
3. The LR finder gives a reliable starting point in a few minutes,
   collapsing a major axis of the hyperparameter search.


### The LR Range Test (Smith, 2017)

The learning rate range test (LR finder) runs a short training pass
over a few batches while INCREASING the LR exponentially from a very
small value to a very large one. The loss is recorded at each step.

    Procedure:
    1. Start with LR_min = 1e-7 (or similar very small value)
    2. For each mini-batch:
       a. Run forward pass and compute loss
       b. Log (current_LR, loss)
       c. Multiply LR by a constant factor: LR ← LR × (LR_max/LR_min)^(1/n_steps)
    3. Plot loss vs LR on a log scale
    4. Read the optimal LR from the curve

    Diagram 7 — LR Range Test Loss Curve:

    Loss
      │
      │────────────               ← flat region: LR too small, no learning
      │            ╲
      │             ╲             ← loss declining: productive range
      │              ╲ ●          ← minimum loss
      │               ╲────╱     ← loss rising: LR too large
      │                    ───── ← divergence: LR far too large
      └────────────────────────── LR (log scale)
           1e-7  1e-5  1e-3  1e-1

    Reading the curve:
    The STEEP DESCENT REGION is where the network is learning fastest.
    The MINIMUM of the curve marks the LR where training is most efficient
    but may be unstable (right at the edge of the basin).

    Recommended LR to use:
    For SGD with momentum: 1/10 of the LR at the minimum
    For Adam:              the LR at the beginning of the steep descent
                           (approximately one order of magnitude below the minimum)

    The Adam recommendation is LOWER because Adam already adapts per-parameter
    learning rates. The raw LR in Adam acts more like a step size bound
    than an absolute learning rate. Using the minimum LR directly with Adam
    often leads to instability.


### LR Finder Implementation

    def lr_finder(model, train_loader, optimiser,
                  start_lr=1e-7, end_lr=10, num_iter=100, smooth=0.05):
        '''
        Run the LR range test and return (lrs, losses).
        Call model.train() before this function.
        '''
        model.train()
        lrs, losses, avg_loss = [], [], 0.0
        lr = start_lr
        mult = (end_lr / start_lr) ** (1 / num_iter)

        # Set initial LR
        for pg in optimiser.param_groups:
            pg['lr'] = lr

        best_loss = float('inf')
        for step, batch in enumerate(train_loader):
            if step >= num_iter:
                break
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Exponential moving average smoothing
            avg_loss = smooth * loss.item() + (1 - smooth) * avg_loss
            smoothed_loss = avg_loss / (1 - (1 - smooth) ** (step + 1))

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            # Stop if loss has grown too much (diverged)
            if step > 10 and smoothed_loss > 4 * best_loss:
                break

            lrs.append(lr)
            losses.append(smoothed_loss)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            # Increase LR
            lr *= mult
            for pg in optimiser.param_groups:
                pg['lr'] = lr

        return lrs, losses

    After running: plt.plot(lrs, losses); plt.xscale('log')
    Look for the steepest descent. Pick the LR at its beginning.


### Cyclical Learning Rates as a Byproduct

The LR finder revealed that loss IMPROVES when LR increases (in the
steep descent region). This inspired CYCLICAL LEARNING RATES (CLR):

    Instead of monotonically decaying LR, cycle it between LR_min and LR_max:

    Triangular CLR:
    LR oscillates linearly between LR_min and LR_max over a fixed cycle.

    lr(t) = LR_min + (LR_max - LR_min) × max(0, 1 - |t/stepsize - 2k - 1|)
    where k = floor(t / (2 × stepsize))

    Benefits of CLR:
    - The periodic LR increase helps the model escape sharp minima
      (a temporary large LR can push the model out of a narrow basin
       into a broader, flatter region that generalises better)
    - Often achieves the same accuracy as fixed LR + careful tuning
      in fewer epochs, with less tuning required
    - LR_min and LR_max can be found directly from the LR range test


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — EFFICIENT SEARCH: HYPERBAND AND SUCCESSIVE HALVING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Early Stopping Insight

Many hyperparameter configurations can be identified as bad early
in training — they are clearly inferior by epoch 3 even if they
eventually converge to different values by epoch 100. Evaluating them
to completion wastes compute.

    Observation: a configuration that is in the top 20% after 5 epochs
    is likely to be in the top 20% after 100 epochs. Not always true,
    but true often enough to make early stopping a useful heuristic.

    The question: how to allocate a fixed compute budget across many
    configurations, stopping inferior ones early and giving more
    compute to promising ones?


### Successive Halving

The simplest early-stopping-based search strategy:

    Given: n initial configurations, total budget B, max iterations R.
    η = elimination ratio (typically η=3: keep 1/3 at each round).

    Round 1: train all n configs for R/n^log_η(n) iterations.
             Evaluate. Keep top 1/η fraction.
    Round 2: train surviving configs for η× longer.
             Evaluate. Keep top 1/η.
    Round 3: continue until 1 config remains.

    Diagram 8 — Successive Halving with n=8, η=2:

    Iteration:  1    2    4    8   16
    Configs:
    Config 1:  ●────●
    Config 2:  ●────●────●
    Config 3:  ●────●
    Config 4:  ●────●────●────●
    Config 5:  ●────●
    Config 6:  ●────●────●────●────●  ← winner (most compute)
    Config 7:  ●────●
    Config 8:  ●────●────●

    Half eliminated at each doubling. The winner gets 16 iterations.
    Total compute: 8×1 + 4×2 + 2×4 + 1×8 = 8+8+8+8 = 32
    Compared to running all 8 to 16 iterations: 8×16 = 128
    Compute savings: 4×, at the cost of eliminating some configs early.

    The limitation of Successive Halving: it requires choosing upfront
    whether to run many configurations with small budgets OR few
    configurations with large budgets. Getting this tradeoff wrong
    wastes the entire budget.


### Hyperband: Adaptive Bracket Selection

Hyperband (Li et al., 2018) runs Successive Halving with MULTIPLE
brackets — different tradeoffs between n (number of configs) and
the minimum training budget:

    For max resource R and η=3, Hyperband runs:
    Bracket 3 (explore): 27 configs, min budget = R/27 each
    Bracket 2:           9 configs,  min budget = R/9 each
    Bracket 1:           3 configs,  min budget = R/3 each
    Bracket 0 (exploit): 1 config,   budget = R (full training)

    Each bracket is a separate Successive Halving race.
    The config that wins across all brackets is the final result.

    Hyperband eliminates the bracket-selection problem by running
    all brackets. The total compute is ~5× what a single Successive
    Halving run would use, but it is robust to whether early performance
    is predictive of late performance.

    ASHA (Asynchronous Successive Halving Algorithm):
    A parallel version of Successive Halving where workers immediately
    start the next configuration when they finish one, rather than
    waiting for synchronisation at each round. ASHA is the practical
    parallelisable version used in most modern tuning frameworks.


### BOHB: Combining Bayesian Optimisation and Hyperband

BOHB (Falkner et al., 2018) integrates Bayesian optimisation (TPE)
with Hyperband:

    In standard Hyperband: configurations in each bracket are sampled
    randomly. This ignores information from previous brackets.

    In BOHB: configurations are sampled using TPE (fitted to the
    observed (config, performance) pairs from ALL previous brackets).
    The search becomes progressively smarter as more evaluations
    accumulate, while still using Hyperband's efficient early stopping.

    BOHB is the current state-of-the-art method for deep learning
    hyperparameter search when compute is available for 50-500 trials.
    Available in: SMAC3, Ray Tune (BOHBSearch), Optuna.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — POPULATION-BASED TRAINING (PBT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Core Idea

Population-Based Training (Jaderberg et al., 2017, DeepMind) trains
a POPULATION of models simultaneously, periodically:
1. Evaluating which models are performing best.
2. Copying the best models' weights into the worst-performing models.
3. Randomly perturbing the hyperparameters of the promoted models.

    This is a continuous, online hyperparameter search — hyperparameters
    evolve throughout training, not just at the start.

    Diagram 9 — PBT: Exploit and Explore Steps:

    Models:     t=0      t=T       t=2T      t=3T
    Model A:  LR=0.1 ──● ────── promoted ─── LR=0.05
    Model B:  LR=0.01──●── dead  copied from A with LR=0.11
    Model C:  LR=0.05──● ────── promoted ─── LR=0.06
    Model D:  LR=0.001──●─ dead  copied from C with LR=0.04

    EXPLOIT step: replace underperforming models with copies
                  of well-performing models (including their weights).
    EXPLORE step: randomly perturb the copied model's hyperparameters
                  (multiply LR by 1.2 or 0.8, etc.).

    PBT effectively searches the TRAJECTORY of hyperparameters, not
    just the initial configuration. A model might start with LR=0.1,
    then be inherited by another model that adapts to LR=0.02 as
    training progresses. This is adaptive scheduling that no fixed
    schedule could match.


### Why PBT Is Especially Powerful for LR Schedules

Fixed LR schedules (cosine, step decay) are designed once and applied
to all runs. They cannot adapt to what the model is doing during training.

PBT discovers ADAPTIVE SCHEDULES by exploiting the natural variation
in the population:

    Discovery: in PBT experiments on image classification, the model
    population collectively discovers warm-up → high LR → gradual decay
    without ever being told this is the right schedule. The population
    evolves toward it because it is the shape that maximises performance.

    This makes PBT especially valuable when the right LR schedule is
    unknown — it discovers it empirically.


### PBT in Practice

    PBT requires running many models SIMULTANEOUSLY — it is a
    PARALLEL algorithm. Each model in the population needs its own GPU
    (or compute allocation).

    Population size: typically 10-20 models.
    Evaluation frequency: every 20-100 training steps.
    Exploit fraction: bottom 20-25% of models are replaced each round.
    Perturbation: multiply by random factor in {0.8, 1.2} for continuous
                  hyperparameters; resample categoricals.

    PBT is implemented in:
    Ray Tune:        PopulationBasedTraining scheduler
    Google Vizier:   Population-based search
    DeepMind's Acme: For RL specifically

    PBT is primarily used for:
    ✓ Reinforcement learning (where training is inherently parallel
      and the right schedule is very hard to predict in advance)
    ✓ Long training runs where the LR schedule matters greatly
    ✓ When many GPUs are available and can run in parallel
    ✗ Single-GPU training (PBT is a parallel algorithm)
    ✗ Short training runs (not enough time for the population to evolve)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — LEARNING RATE SCHEDULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why Schedule the Learning Rate?

The optimal learning rate changes throughout training:

    Early training: large LR is beneficial — the model is far from any
    good minimum and can take large steps in the right direction without
    overshooting. Larger LR also helps escape the initial bad region faster.

    Mid training: as the model approaches a minimum, a large LR causes
    it to oscillate around the minimum without converging. The LR must
    decrease to allow stable convergence.

    Late training: a very small LR allows fine-grained adjustment of weights
    to settle into the minimum precisely.

    Warmup phase: at the very start, the model is poorly initialised.
    A large LR immediately causes chaotic updates. A gradual warmup
    (small LR increasing to the target) stabilises the first few steps.

    Optimal training thus requires a non-constant LR:
    warmup → peak → decay (the shape of virtually every successful schedule).


### Warmup

Linear warmup is standard in all transformer training:

    LR(step) = LR_max × (step / warmup_steps)   for step ≤ warmup_steps
    LR(step) = LR_decay_schedule(step)           for step > warmup_steps

    Typical warmup duration:
    Fine-tuning (BERT-scale):     warmup_steps = 0.06 × total_steps
    Pre-training (GPT-scale):     warmup_steps = 2,000 to 10,000 steps
    Image training (ViT):         warmup_steps = 10-50 epochs

    WHY WARMUP IS ESSENTIAL for transformers:
    Without warmup, Adam's variance estimate (v̂ₜ) is unreliable in the
    first few steps. The effective per-parameter LR = η/sqrt(v̂ₜ) can be
    arbitrarily large when v̂ₜ ≈ 0 (very early in training).
    Large LR in the first few steps corrupts the model's internal
    representations before any meaningful patterns can be learned.
    Warmup avoids this by keeping the global LR small until the
    variance estimates stabilise.


### Cosine Annealing

The most widely used LR decay schedule in modern deep learning:

    LR(step) = LR_min + 0.5 × (LR_max - LR_min) × (1 + cos(π × step/T))

    Where T = total training steps (or steps per cycle for SGDR).

    Diagram 10 — Cosine Annealing Shape:

    LR
    LR_max │╲
           │ ╲
           │  ╲
           │   ╲───╮
           │       ╰───────╮
           │               ╰─────────╮
    LR_min │                         ╰──── (approaches 0 smoothly)
           └──────────────────────────── Training step

    Properties:
    - Smooth (no sudden jumps unlike step decay)
    - Starts fast (high LR when far from minimum)
    - Ends slow (small LR for precise convergence)
    - The cosine shape was derived from the observation that it
      approximates the theoretical optimal schedule for quadratic
      loss landscapes

    Cosine annealing with warmup (standard transformer schedule):
    LR(step) = LR_warmup(step)           for step < warmup_steps
               cosine_anneal(step)        for step ≥ warmup_steps


### Cosine Annealing with Warm Restarts (SGDR)

Loshchilov & Hutter (2017) extended cosine annealing with periodic
RESTARTS — the LR is reset to LR_max periodically and then decays again:

    LR(step) = LR_min + 0.5 × (LR_max - LR_min) × (1 + cos(π × t_cur/T_cur))

    Where T_cur resets to 0 at each restart.

    Diagram 11 — Cosine Annealing with Warm Restarts:

    LR
    LR_max │╲    ╲    ╲         ╲
           │ ╲    ╲    ╲         ╲
           │  ╰╮   ╰╮   ╰──╮     ╰──────╮
    LR_min │   ╰   ╰      ╰            ╰── (longer cycles each restart)
           └──────────────────────────── step

    The cycle length can grow at each restart (multiply by T_mult=2):
    Restart 1: cycle length T₁
    Restart 2: cycle length T₁ × T_mult = 2T₁
    Restart 3: cycle length 4T₁
    ...

    Motivation: each restart helps the model escape local minima
    (the LR spike pushes it out). After each restart, it settles
    into a slightly different — potentially better — basin.
    Ensembling models from the snapshots just before each restart
    often outperforms a single final model.


### Step Decay and Exponential Decay

    STEP DECAY (most common in CNNs before ~2018):
    Reduce LR by a fixed factor at predetermined milestones.

    LR(epoch) = LR_base × γ^(floor(epoch / step_size))

    Example: LR=0.1, γ=0.1, step_size=30
    Epoch 0-29:  LR = 0.1
    Epoch 30-59: LR = 0.01
    Epoch 60-89: LR = 0.001

    Standard for ResNets (divide by 10 at epoch 30, 60, 90 for 90 epochs).
    Produces sudden LR changes rather than smooth decay.
    Less preferred for transformers (abrupt changes disrupt Adam's
    variance estimates).

    EXPONENTIAL DECAY:
    LR(step) = LR_base × γ^step    (continuous decay every step)

    Smooth but decays to near-zero long before training ends.
    Requires careful tuning of γ. Rarely used in modern practice.

    REDUCE ON PLATEAU (ReduceLROnPlateau):
    Monitor the validation metric. If it has not improved in
    patience epochs, multiply LR by a factor (e.g., 0.5).

    When to use: when you don't know how many total training steps
    to allocate upfront. The schedule adapts to actual training dynamics.
    When not to use: when training for a fixed number of steps and
    cosine annealing is applicable. ReduceLROnPlateau is reactive;
    cosine annealing is proactive.

    In PyTorch:
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle)
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — PRACTICAL TUNING STRATEGY AND DECISION FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Three-Phase Tuning Protocol

A structured approach that avoids wasted compute:

    PHASE 1 — COARSE SEARCH (1-2 days):
    Goal: identify the viable order of magnitude for each Tier-1 hyperparameter.
    Method: Random search with 20-50 trials, wide search space, short training runs
            (10-20% of full training budget, using early stopping).
    Hyperparameters: learning rate, weight decay.
    Outcome: identify the LR range [LR_low, LR_high] where val loss decreases.

    PHASE 2 — MEDIUM SEARCH (3-5 days):
    Goal: find good combinations of Tier-1 and Tier-2 hyperparameters.
    Method: Bayesian optimisation (Optuna/BOHB) or random search with 30-100 trials.
            Narrow the search space using Phase 1 results.
            Medium training runs (50% of full budget).
    Hyperparameters: LR (narrowed), weight decay, dropout, schedule type.
    Outcome: 3-5 strong candidate configurations.

    PHASE 3 — FINE SEARCH (2-3 days):
    Goal: confirm the best configuration and run at full budget.
    Method: Train each Phase 2 candidate to FULL convergence.
            Use 3 different random seeds for each to estimate variance.
    Outcome: select the best candidate. Report test metrics once.

    Total compute: ~10-20× a single full training run.
    For a 4-hour training run, this is a 2-3 day tuning campaign.


### Algorithm Selection Guide

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Situation                       │ Recommended algorithm              │
    ├──────────────────────────────────────────────────────────────────────┤
    │ 1-2 hyperparameters             │ Grid search (exhaustive)           │
    │                                 │                                    │
    │ 3-10 hyperparameters,           │ Random search with LR finder       │
    │ < 20 evaluations                │ for LR initialisation              │
    │                                 │                                    │
    │ 3-10 hyperparameters,           │ Bayesian optimisation (Optuna TPE) │
    │ 20-200 evaluations              │ with early stopping (ASHA)         │
    │                                 │                                    │
    │ > 10 hyperparameters or         │ Random search + ASHA, or           │
    │ very noisy objective            │ Hyperband                          │
    │                                 │                                    │
    │ Many GPUs available,            │ BOHB (Bayesian + Hyperband)        │
    │ 50-500 evaluations              │ via Ray Tune                       │
    │                                 │                                    │
    │ Very expensive evaluations      │ LR finder + expert priors +        │
    │ (days per trial), < 10 trials   │ scaling laws                       │
    │                                 │                                    │
    │ LR schedule only, known         │ LR finder → cosine annealing       │
    │ architecture                    │ (no search needed)                 │
    │                                 │                                    │
    │ Parallel training, adaptive     │ Population-Based Training (PBT)    │
    │ schedule needed                 │ via Ray Tune                       │
    └──────────────────────────────────────────────────────────────────────┘


### Hyperparameter Sensitivity Analysis

After any tuning campaign, estimate the SENSITIVITY of performance to
each hyperparameter. This guides where to invest future tuning:

    FANOVA (Functional ANOVA, Hutter et al., 2014):
    Decomposes performance variance into contributions from each
    hyperparameter and pair interactions.
    Output: each hyperparameter's fraction of total variance.

    Example fANOVA result:
    Learning rate:             62% of variance  ← tune most carefully
    Weight decay:              18% of variance
    LR schedule:               11% of variance
    Dropout:                    5% of variance
    β₁ (Adam):                  2% of variance  ← safe to fix at default
    β₂ (Adam):                  1% of variance  ← safe to fix at default

    This tells you: for this task, the LR and weight decay account for
    80% of hyperparameter-induced variance in performance. All future
    tuning should focus there. Time spent tuning β₁ or dropout is
    almost certainly wasted.

    Available in: SMAC3 (built-in), manual approximation by comparing
    random search results grouped by each hyperparameter value.


### Common Hyperparameter Tuning Mistakes

    MISTAKE 1 — Tuning on the test set:
    Evaluating on the test set during hyperparameter search and
    selecting the best config by test performance.
    Result: reported test performance is optimistically biased.
    Fix: use a validation set for all tuning decisions.

    MISTAKE 2 — Not controlling random seeds:
    Training with a single random seed and reporting variance
    as hyperparameter sensitivity.
    Result: the random seed may contribute more variance than
    the hyperparameter being studied.
    Fix: fix a seed for tuning (for comparability), then re-train the
    top configs with multiple seeds to estimate true variance.

    MISTAKE 3 — Searching in the wrong order:
    Tuning Tier-3 hyperparameters (e.g., Adam β₁) before Tier-1 (LR).
    Result: you find the best β₁ for a suboptimal LR — not transferable.
    Fix: always tune Tier-1 first. Fix those before exploring Tier-2.

    MISTAKE 4 — Using linear scale for log-distributed hyperparameters:
    Searching LR in [0.0001, 0.001] on a linear scale.
    99% of the search budget goes to the range [0.0005, 0.001].
    Fix: ALWAYS use log scale for LR, weight decay, epsilon.

    MISTAKE 5 — Forgetting that the optimal LR depends on batch size:
    Tuning LR at B=64 and using the same LR at B=256.
    Result: 4× suboptimal LR (should scale by 4 per linear scaling rule).
    Fix: re-run LR tuning (or LR finder) whenever batch size changes.

    MISTAKE 6 — Comparing models tuned with different budgets:
    Model A was tuned with 100 trials; Model B with 10 trials.
    Claiming Model A is "better" because its val metric is higher
    may just reflect better tuning, not a better architecture.
    Fix: control the tuning budget when comparing models.

"""


# ─────────────────────────────────────────────────────────────────────────────
# No OPERATIONS (theory-only module)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
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
    visual_height = 1150
    try:
        from Training_Core.visuals.HP_Tuning_Visual import (
            HP_TUNING_VISUAL_HTML,
            HP_TUNING_VISUAL_HEIGHT,
        )
        visual_html   = HP_TUNING_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = HP_TUNING_VISUAL_HEIGHT
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
        "complexity":    None,
        "operations":    OPERATIONS,
    }