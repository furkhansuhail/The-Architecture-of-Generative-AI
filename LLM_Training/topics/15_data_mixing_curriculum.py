"""
Data Mixing and Curriculum Learning
=====================================

What you train on matters as much as how you train. A 7B model trained
on carefully weighted domain mixtures outperforms the same model trained
on an unweighted crawl. Curriculum learning — ordering or reweighting the
data over time — can further improve sample efficiency. This module covers
how to set domain mixing weights, implement weighted sampling, and design
training curricula that match the data to the model's current capability.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Data Mixing and Curriculum Learning"
DISPLAY_NAME = "15 · Data Mixing & Curriculum"
ICON         = "🎓"
SUBTITLE     = "Domain Weights and Curriculum Strategies"


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

### Why Mixing Ratios Matter

The internet is not a balanced corpus. An unweighted Common Crawl sample
contains orders of magnitude more English text than Swahili, more celebrity
gossip than scientific papers, and more SEO spam than textbooks.

If you train naively on the raw distribution:
    •   The model will be good at generating spam and mediocre prose
    •   Scientific reasoning and mathematics will be underrepresented
    •   Non-English languages will be weak despite the multilingual data
    •   Code generation will suffer if code data is < 1% of the corpus

Data mixing deliberately **overrepresents high-quality, diverse sources**
and **underrepresents low-quality, redundant sources**, decoupled from
their natural frequency in the raw corpus.


### The Standard Mixing Approach

Each domain d has:
    •   A raw dataset size S_d (tokens available)
    •   A mixing weight w_d (target fraction of training tokens from domain d)
    •   Σ w_d = 1 across all domains

At each training step, a batch is assembled by sampling:
    •   w_d × (global_batch_size) tokens from domain d

In practice, this is implemented as a **weighted sampler** that samples
batches from a mixture of domain-specific DataLoaders.

**Oversample vs undersample:**
    •   If w_d × total_train_tokens > S_d: the domain is **oversampled**
        (data is seen multiple times — careful of overfitting small domains)
    •   If w_d × total_train_tokens < S_d: the domain is **undersampled**
        (some data is never seen)


### Typical Domain Mixing Recipes

    Domain              LLaMA-2     LLaMA-3 (est.)   Falcon     Phi-3
    ─────────────────────────────────────────────────────────────────────
    Web text (CC)       ~67%         ~55%            ~79%        ~40%
    Code                ~~8%         ~17%            ~3%         ~30%
    Books               ~~7%         ~~5%            ~~7%        ~15%
    Wikipedia/encyclop. ~~4%         ~~3%            ~~3%        ~3%
    Scientific papers   ~~2%         ~~3%            ~2%         ~2%
    Other               ~12%         ~17%            ~6%         ~10%
    ─────────────────────────────────────────────────────────────────────

Key observations:
    •   Code is heavily overweighted relative to its natural frequency
      (code is ~5% of the internet but gets 8–30% mixing weight)
    •   Books are overweighted (high information density)
    •   Pure web text is underweighted relative to its raw size
    •   Phi-3's heavy code/book weighting reflects its "textbook" thesis


### The Mathematics of Weighted Sampling

Given K domains with weights w_1, …, w_K (Σw_k = 1) and datasets
D_1, …, D_K of sizes |D_1|, …, |D_K|:

**Effective epochs per domain:**
    e_k = (w_k × N_total) / |D_k|

where N_total is the total training token budget.

    If e_k > 1: domain k is oversampled (repeated during training)
    If e_k < 1: domain k is undersampled (some data never used)
    If e_k = 1: domain k is used exactly once (Chinchilla ratio)

**Example — LLaMA-3 8B (15T tokens, ~5.8M token Wikipedia):**
    w_wiki = 0.03  →  e_wiki = 0.03 × 15e12 / 5.8e6 ≈ 77,000 epochs!

Wikipedia is so small relative to the training budget that even at a 3%
mixing weight, it is seen ~77,000 times. This level of repetition risks
memorisation but Wikipedia is high-enough quality that it is accepted.

**The oversampling threshold:** Empirically, domains can be safely
oversampled up to ~4–8× without causing significant memorisation problems
for large, diverse domains. Small, repetitive domains (e.g., a single
book) should not be oversampled more than ~3×.


    **Diagram 1 — Domain Mixing Conceptual Diagram:**

    WEIGHTED DOMAIN SAMPLING
    ════════════════════════════════════════════════════════════════

    Available data:                    Mixing weights (desired):

    ┌──────────────────────────────┐   Web:       ▓▓▓▓▓▓▓▓▓▓▓▓  60%
    │   Web text   (very large)    │   Code:      ▓▓▓▓           20%
    │██████████████████████████████│   Books:     ▓▓▓            12%
    │   Code        (medium)       │   Wiki:       ▓              5%
    │████████                      │   Papers:    ▒               3%
    │   Books       (small)        │
    │████                          │
    │   Wikipedia   (tiny)         │   ← Wikipedia is oversampled
    │█                             │     ~thousands of times
    └──────────────────────────────┘

    At each step, a batch is assembled by sampling proportionally
    to the mixing weights — not to raw dataset sizes.


### Curriculum Learning

**Curriculum learning** (Bengio et al., 2009) is the idea that presenting
training examples in a structured order — easy to hard — can improve
both convergence speed and final quality.

For LLMs, several curriculum strategies have proven effective:

**1. Quality curriculum (most impactful):**
Train on progressively higher-quality data. Start with broad web text,
then increase the fraction of curated, high-quality sources as training
progresses. The model first learns basic language statistics cheaply from
raw data, then refines on quality.

LLaMA-3 used a "late-stage data boost": near the end of pre-training, the
mixing weights were shifted heavily toward high-quality sources (code,
math, curated text). This produced large quality gains for the compute spent.

**2. Difficulty curriculum:**
Start with easier (shorter, simpler) sequences; introduce longer and more
complex text later. The training loss at the beginning of training is
dominated by random prediction; using harder examples too early provides
noisy gradient signal.

**3. Length curriculum:**
Start with shorter sequences (e.g., 512 tokens), progressively extend
to the target context length (e.g., 8192). This allows stable training
at a lower memory/compute cost for most of training, with only the final
phase needing the full context length. Used by LLaMA-2, Mistral.

**4. Capability-targeted annealing (LLaMA-3 style):**
In the final stage of pre-training (the WSD decay phase), concentrate
the data mix on capabilities you want to improve:
    •   Add more math data → improves mathematical reasoning
    •   Add more code → improves coding ability
    •   Add more recent web text → freshens knowledge cutoff
    •   Add more multilingual data → improves non-English performance

This is one of the most cost-effective techniques: the last 5–10% of
training can be disproportionately influential on the model's final
capability profile.


    **Diagram 2 — Curriculum Stages:**

    TRAINING CURRICULUM STAGES
    ════════════════════════════════════════════════════════════════

    Stage 1 — Broad base (0% → ~85% of training):
    ─────────────────────────────────────────────
    Data mix:   60% web, 20% code, 12% books, 5% wiki, 3% papers
    Seq len:    Start at 2048, gradually increase
    Purpose:    Learn general language, broad world knowledge

    Stage 2 — Quality boost (~85% → ~95% of training):
    ────────────────────────────────────────────────────
    Data mix:   40% high-quality web, 30% code, 15% curated text
                10% math/science, 5% other
    Seq len:    Full context length
    Purpose:    Refine on quality, improve reasoning

    Stage 3 — Capability annealing (~95% → 100%):
    ───────────────────────────────────────────────
    Data mix:   Targeted by desired output capability profile
                (math heavy for reasoning, code heavy for coding, etc.)
    Seq len:    Full or extended
    Purpose:    Final capability fine-tuning before SFT

    LR schedule: WSD decay phase aligns with Stage 3.
    (See Module 08: learning rate drops during this final phase,
     allowing careful settling into the quality-focused minimum)


### Domain Reweighting Strategies

**Uniform mixing:** Equal weight to all domains.
    Simple, but ignores domain quality and size disparities.

**Size-proportional:** Weight proportional to raw corpus size.
    Follows the natural internet distribution. Biased toward spam/noise.

**Capability-targeted (DoReMi / DoReMi-like):**
    Optimise mixing weights by running a small "reference" model on each
    domain and upweighting domains where the main model is worst.
    Used by: Google (DoReMi paper), some recent open recipes.

**Manual ablation-based:**
    Run small-scale training experiments (e.g., 1B model, 10B tokens) with
    different mixing ratios. Measure downstream task performance for each
    ablation. Select the best-performing mix.
    Most common in practice for large-scale training.

**Perplexity-based:**
    Train a reference model on "gold" data (Wikipedia, books). Use this
    model's perplexity on each domain as a quality signal. Higher perplexity
    on a domain = more surprising/informative to the model = upweight.
    This is related to the Phi approach of selecting "textbook-quality" text.


### DoReMi — Automatic Domain Reweighting

**DoReMi** (Xie et al., 2023) is an algorithm that automatically learns
domain weights:

    1.  Train a small reference model θ_ref on uniform domain mixture
    2.  For each domain d at each step, compute per-token excess loss:
            l_t(d) = max(0, L(θ, d) - L(θ_ref, d))
    3.  Update domain weights proportional to excess loss:
            w_d ← w_d × exp(η × l_t(d))
            Normalise so Σw_d = 1
    4.  The main model is then trained with these learned weights

Domains where the main model is underperforming relative to the reference
get higher weights. Domains where it is performing well get lower weights.

DoReMi was shown to match or exceed human-tuned mixing ratios on several
benchmarks, providing a principled alternative to manual ablations.


### The Multilingual Mixing Challenge

For multilingual LLMs, mixing ratios have a direct impact on
cross-lingual transfer and non-English capability:

**The imbalance problem:**
    English:  ~50% of web text
    Chinese:  ~8%
    German:   ~5%
    Arabic:   ~2%
    Hindi:    ~0.5%
    Swahili:  ~0.01%

If trained proportionally, rare languages are effectively ignored. But
oversampling them too much reduces English quality (capacity competition).

**The upsample strategy:**
Languages with < threshold fraction are upsampled. A common formula:
    w_L = (S_L)^α / Σ_L (S_L)^α

where α ∈ (0, 1) is a temperature parameter:
    α = 1.0: proportional to raw size (favours large languages)
    α = 0.7: moderate upsampling of small languages (used by XLM-R)
    α = 0.5: more aggressive upsampling
    α = 0.0: uniform across languages (extreme upsampling of rare)

LLaMA-3 uses a higher code fraction (17%) partly to improve multilingual
code generation, which is a proxy for multilingual reasoning.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Curriculum Strategy Comparison

| Strategy                  | Implementation effort | Quality gain | Best for                         |
|---------------------------|-----------------------|--------------|----------------------------------|
| Uniform mix               | Trivial               | Baseline     | Ablations / baselines            |
| Manual ablation tuning    | Moderate              | High         | Production pre-training          |
| Quality curriculum        | Moderate              | Very high    | Improving reasoning/knowledge    |
| Length curriculum         | Low                   | Moderate     | Long-context training            |
| Capability annealing      | Low                   | High         | Final pre-training stage         |
| DoReMi (auto-reweight)    | High                  | High         | Large scale, limited ablation $  |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Weighted Domain Sampler": {
        "description": "Implement a weighted domain sampler from scratch — interleave batches from multiple domain datasets according to configured mixing weights.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
WEIGHTED DOMAIN SAMPLER — FROM SCRATCH
================================================================================

Implements a production-style weighted domain sampler:
    1. Multiple domain datasets with different sizes
    2. Configurable mixing weights (independent of dataset sizes)
    3. Oversampling (repeat small high-quality domains)
    4. Undersampling (skip some data from large noisy domains)
    5. Epoch-level tracking and statistics

This is the core mechanism used in all major LLM training pipelines.
================================================================================
"""

import random
import math
from dataclasses import dataclass, field
from typing import Iterator
from collections import defaultdict


@dataclass
class DomainConfig:
    """Configuration for one data domain."""
    name:         str
    n_tokens:     int       # total tokens available
    weight:       float     # desired fraction of training tokens
    quality:      float = 1.0   # subjective quality score (for logging)


@dataclass
class MixingConfig:
    """Overall mixing configuration."""
    domains:           list[DomainConfig]
    total_train_tokens: int
    batch_tokens:      int  = 4_096   # tokens per batch step

    def __post_init__(self):
        total_w = sum(d.weight for d in self.domains)
        if abs(total_w - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_w:.4f}")

    @property
    def tokens_per_domain(self) -> dict[str, int]:
        return {d.name: int(d.weight * self.total_train_tokens)
                for d in self.domains}

    @property
    def epochs_per_domain(self) -> dict[str, float]:
        tpd = self.tokens_per_domain
        return {d.name: tpd[d.name] / d.n_tokens
                for d in self.domains}

    def summary(self) -> str:
        tpd = self.tokens_per_domain
        epd = self.epochs_per_domain
        lines = [
            f"  Total training: {self.total_train_tokens/1e12:.1f}T tokens",
            f"  Batch size:     {self.batch_tokens:,} tokens",
            f"  Total steps:    {self.total_train_tokens//self.batch_tokens:,}",
            "",
            f"  {'Domain':<18} {'Weight':>8}  {'Tokens':>12}  "
            f"{'Available':>12}  {'Epochs':>8}  {'Status':>12}",
            f"  {'':─<18} {'':─>8}  {'':─>12}  {'':─>12}  {'':─>8}  {'':─>12}",
        ]
        for d in self.domains:
            t   = tpd[d.name]
            e   = epd[d.name]
            status = ("oversample" if e > 1.5
                      else "undersample" if e < 0.3
                      else "≈ 1 epoch")
            lines.append(
                f"  {d.name:<18} {d.weight:>8.1%}  {t/1e9:>11.1f}B  "
                f"{d.n_tokens/1e9:>11.1f}B  {e:>8.1f}  {status:>12}"
            )
        return "\n".join(lines)


# ── Synthetic domain datasets (token stream simulators) ───────────────────────

class DomainDataset:
    """Simulates an infinite token stream for one domain."""

    def __init__(self, domain: DomainConfig, batch_tokens: int, seed: int = 0):
        self.name         = domain.name
        self.n_tokens     = domain.n_tokens
        self.batch_tokens = batch_tokens
        self.rng          = random.Random(seed)
        self.tokens_seen  = 0
        self.n_epochs     = 0

    def get_batch(self) -> dict:
        """Return one batch of tokens (simulated as random integers)."""
        if self.tokens_seen + self.batch_tokens > self.n_tokens:
            self.n_epochs   += 1
            self.tokens_seen = 0   # wrap around (oversampling)

        tokens = [self.rng.randint(0, 31999) for _ in range(self.batch_tokens)]
        self.tokens_seen += self.batch_tokens
        return {
            "input_ids": tokens,
            "domain":    self.name,
        }


# ── Weighted sampler ──────────────────────────────────────────────────────────

class WeightedDomainSampler:
    """
    At each training step, samples a batch from one domain according to
    the configured mixing weights.

    Two strategies:
        "stochastic": randomly sample domain at each step (matches weights
                      in expectation over many steps)
        "cyclic":     deterministic cycling that respects weights exactly
                      (better for reproducibility and exact budgets)
    """

    def __init__(self, cfg: MixingConfig, strategy: str = "stochastic",
                 seed: int = 42):
        self.cfg      = cfg
        self.strategy = strategy
        self.rng      = random.Random(seed)

        # Build domain datasets
        self.datasets = {
            d.name: DomainDataset(d, cfg.batch_tokens, seed=i)
            for i, d in enumerate(cfg.domains)
        }

        # Pre-compute cumulative weights for efficient sampling
        names    = [d.name for d in cfg.domains]
        weights  = [d.weight for d in cfg.domains]
        self.cum_weights = []
        total    = 0.0
        for w in weights:
            total += w
            self.cum_weights.append(total)
        self.domain_names = names

        # Tracking
        self.step_count:    int           = 0
        self.domain_counts: dict[str,int] = defaultdict(int)
        self.token_counts:  dict[str,int] = defaultdict(int)

    def _sample_domain_stochastic(self) -> str:
        r = self.rng.random()
        for name, cum in zip(self.domain_names, self.cum_weights):
            if r <= cum:
                return name
        return self.domain_names[-1]

    def _sample_domain_cyclic(self) -> str:
        """Deterministically cycle through domains matching exact weights."""
        # Use step count to determine which domain
        total_steps = self.cfg.total_train_tokens // self.cfg.batch_tokens
        step_mod    = self.step_count % 1000   # cycle length = 1000 steps
        target      = step_mod / 1000.0

        total = 0.0
        for name, w in zip(self.domain_names,
                            [d.weight for d in self.cfg.domains]):
            total += w
            if target < total:
                return name
        return self.domain_names[-1]

    def __iter__(self) -> Iterator[dict]:
        total_steps = self.cfg.total_train_tokens // self.cfg.batch_tokens
        for _ in range(total_steps):
            yield self.next_batch()

    def next_batch(self) -> dict:
        """Get the next batch from the weighted mixture."""
        if self.strategy == "stochastic":
            domain_name = self._sample_domain_stochastic()
        else:
            domain_name = self._sample_domain_cyclic()

        batch = self.datasets[domain_name].get_batch()
        self.step_count                     += 1
        self.domain_counts[domain_name]     += 1
        self.token_counts[domain_name]      += self.cfg.batch_tokens
        return batch

    def statistics(self) -> str:
        """Report actual vs target mixing fractions."""
        total_tokens = sum(self.token_counts.values())
        if total_tokens == 0:
            return "No steps taken yet."

        lines = [
            f"  Steps:          {self.step_count:,}",
            f"  Total tokens:   {total_tokens/1e6:.1f}M",
            "",
            f"  {'Domain':<18} {'Target':>8}  {'Actual':>8}  {'Δ':>8}  {'Batches':>10}",
            f"  {'':─<18} {'':─>8}  {'':─>8}  {'':─>8}  {'':─>10}",
        ]

        for d in self.cfg.domains:
            actual = self.token_counts[d.name] / total_tokens
            delta  = actual - d.weight
            flag   = " ⚠️" if abs(delta) > 0.02 else " ✓"
            lines.append(
                f"  {d.name:<18} {d.weight:>8.1%}  {actual:>8.1%}  "
                f"{delta:>+8.1%}  {self.domain_counts[d.name]:>10,}{flag}"
            )

        return "\n".join(lines)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Define a realistic LLaMA-3-style mixing config
    domains = [
        DomainConfig("web_text",    n_tokens=10_000_000_000_000, weight=0.55, quality=0.6),
        DomainConfig("code",        n_tokens=    500_000_000_000, weight=0.17, quality=0.9),
        DomainConfig("books",       n_tokens=    200_000_000_000, weight=0.08, quality=0.95),
        DomainConfig("wikipedia",   n_tokens=      5_800_000_000, weight=0.03, quality=0.98),
        DomainConfig("papers",      n_tokens=    100_000_000_000, weight=0.05, quality=0.92),
        DomainConfig("other",       n_tokens=    300_000_000_000, weight=0.12, quality=0.7),
    ]

    cfg = MixingConfig(
        domains=domains,
        total_train_tokens=15_000_000_000_000,   # 15T (LLaMA-3)
        batch_tokens=4_096,
    )

    print("=" * 65)
    print("  DOMAIN MIXING CONFIGURATION (LLaMA-3 style)")
    print("=" * 65)
    print()
    print(cfg.summary())

    print()
    print("=" * 65)
    print("  STOCHASTIC SAMPLER — 10,000 STEPS")
    print("=" * 65)
    print()

    sampler = WeightedDomainSampler(cfg, strategy="stochastic", seed=42)
    N_STEPS = 10_000

    for _ in range(N_STEPS):
        batch = sampler.next_batch()

    print(sampler.statistics())

    print()
    print("=" * 65)
    print("  OVERSAMPLING ANALYSIS")
    print("=" * 65)
    print()

    epd = cfg.epochs_per_domain
    for d in domains:
        e = epd[d.name]
        if e > 1:
            print(f"  ⚠️  {d.name:<18} seen {e:.0f}× — risk of memorisation if e >> 4")
        else:
            print(f"  ✓  {d.name:<18} seen {e:.3f}× — some data never used")

    print()
    print("  Key insight: Wikipedia (5.8B tokens) is seen ~2,586 times!")
    print("  Despite the repetition, Wikipedia quality is high enough to accept.")
    print("  Rule of thumb: oversampling > 4× risks memorisation for small datasets.")
''',
    },

    "Curriculum Implementation: Quality and Length Stages": {
        "description": "Implement a multi-stage training curriculum that shifts domain weights over time — broad base → quality boost → capability annealing.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MULTI-STAGE CURRICULUM IMPLEMENTATION
================================================================================

Implements a three-stage training curriculum:
    Stage 1 (0–85%):   Broad base — general web + code + books
    Stage 2 (85–95%):  Quality boost — shift toward curated sources
    Stage 3 (95–100%): Capability annealing — targeted domain boost

Also implements a length curriculum: progressively increase sequence length
from short to the target context length.

This matches the strategy used in LLaMA-3 and many 2024 LLMs.
================================================================================
"""

import math
from dataclasses import dataclass, field


@dataclass
class CurriculumStage:
    name:         str
    start_frac:   float       # start of stage as fraction of total training
    end_frac:     float       # end of stage
    domain_weights: dict      # domain → weight (must sum to 1.0)
    seq_len:      int         # sequence length for this stage
    description:  str = ""


class CurriculumScheduler:
    """
    Manages domain weights and sequence length as a function of training progress.

    Given the current step and total steps, returns:
        - current domain weights (dict)
        - current sequence length
        - current stage name
    """

    def __init__(self, stages: list[CurriculumStage], total_steps: int):
        self.stages      = sorted(stages, key=lambda s: s.start_frac)
        self.total_steps = total_steps
        self._validate()

    def _validate(self):
        for s in self.stages:
            total_w = sum(s.domain_weights.values())
            assert abs(total_w - 1.0) < 1e-6, \
                f"Stage '{s.name}' weights sum to {total_w:.4f}, not 1.0"

    def get_stage(self, step: int) -> CurriculumStage:
        """Return the active stage for the given step."""
        frac = step / self.total_steps
        for stage in reversed(self.stages):
            if frac >= stage.start_frac:
                return stage
        return self.stages[0]

    def get_weights(self, step: int) -> dict:
        return self.get_stage(step).domain_weights

    def get_seq_len(self, step: int) -> int:
        return self.get_stage(step).seq_len

    def transition_steps(self) -> list[int]:
        """Return the step numbers at which stage transitions occur."""
        return [int(s.start_frac * self.total_steps) for s in self.stages]

    def visualise(self, n_points: int = 40) -> str:
        """ASCII plot of domain weight evolution across training."""
        domains  = list(self.stages[0].domain_weights.keys())
        domain_colors = {d: c for d, c in zip(domains, "░▒▓█●○◆◇▲△▼▽")}

        lines    = ["\n  CURRICULUM DOMAIN WEIGHTS OVER TRAINING\n"]
        n_stages = len(self.stages)

        # Header: stage boundaries
        header = "  step: "
        for i in range(n_points + 1):
            frac = i / n_points
            if any(abs(frac - s.start_frac) < 0.5 / n_points for s in self.stages):
                header += "│"
            else:
                header += "─"
        lines.append(header)

        # One row per domain
        for domain in domains:
            row = f"  {domain:<12}"
            for i in range(n_points):
                frac    = (i + 0.5) / n_points
                step    = int(frac * self.total_steps)
                stage   = self.get_stage(step)
                w       = stage.domain_weights.get(domain, 0.0)
                # Map weight to a character
                if w >= 0.50:    row += "█"
                elif w >= 0.30:  row += "▓"
                elif w >= 0.15:  row += "▒"
                elif w >= 0.05:  row += "░"
                elif w > 0:      row += "·"
                else:            row += " "
            row += f"  (final: {self.get_stage(self.total_steps-1).domain_weights.get(domain, 0.0):.0%})"
            lines.append(row)

        # Stage labels
        stage_row = "  stages: "
        chars_per_step = n_points / 100
        for stage in self.stages:
            start_chars = int(stage.start_frac * n_points)
            end_chars   = int(stage.end_frac   * n_points)
            label       = stage.name[:end_chars - start_chars - 1]
            stage_row  += f"{stage_row:>{start_chars}}" if False else ""
        # Simple stage label row
        row2 = "  stage:  "
        for i in range(n_points):
            frac  = (i + 0.5) / n_points
            step  = int(frac * self.total_steps)
            stage = self.get_stage(step)
            row2 += str(self.stages.index(stage) + 1)
        lines.append(row2)

        return "\n".join(lines)


# ── Length curriculum ──────────────────────────────────────────────────────────

class LengthCurriculum:
    """
    Linearly or cosine-increases sequence length from min_len to max_len.

    This allows stable training at lower memory cost for most of training,
    with the full context only needed in the final phase.
    """

    def __init__(self, min_len: int, max_len: int, total_steps: int,
                 schedule: str = "linear", warmup_frac: float = 0.5):
        self.min_len     = min_len
        self.max_len     = max_len
        self.total_steps = total_steps
        self.schedule    = schedule
        self.warmup_frac = warmup_frac  # fraction of steps for length increase

    def get_len(self, step: int) -> int:
        """Return the sequence length for the given step."""
        frac      = step / self.total_steps
        ramp_frac = min(frac / self.warmup_frac, 1.0)

        if self.schedule == "linear":
            t = ramp_frac
        elif self.schedule == "cosine":
            t = 0.5 * (1 - math.cos(math.pi * ramp_frac))
        elif self.schedule == "step":
            # Step function: double at each quartile
            if ramp_frac < 0.25:   t = 0.0
            elif ramp_frac < 0.5:  t = 0.33
            elif ramp_frac < 0.75: t = 0.67
            else:                   t = 1.0
        else:
            t = ramp_frac

        raw_len = self.min_len + t * (self.max_len - self.min_len)
        # Round to nearest power of 2 or multiple of 256
        return max(self.min_len, min(self.max_len,
                                     int(raw_len // 256) * 256))


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TOTAL_STEPS = 100_000

    # Define three-stage curriculum (LLaMA-3 inspired)
    stages = [
        CurriculumStage(
            name="broad_base",
            start_frac=0.00,
            end_frac=0.85,
            domain_weights={
                "web":     0.55,
                "code":    0.15,
                "books":   0.10,
                "wiki":    0.03,
                "papers":  0.05,
                "other":   0.12,
            },
            seq_len=2048,
            description="General-purpose base learning",
        ),
        CurriculumStage(
            name="quality_boost",
            start_frac=0.85,
            end_frac=0.95,
            domain_weights={
                "web":     0.35,
                "code":    0.25,
                "books":   0.18,
                "wiki":    0.05,
                "papers":  0.10,
                "other":   0.07,
            },
            seq_len=4096,
            description="Shift toward high-quality curated sources",
        ),
        CurriculumStage(
            name="capability_annealing",
            start_frac=0.95,
            end_frac=1.00,
            domain_weights={
                "web":     0.20,
                "code":    0.35,
                "books":   0.20,
                "wiki":    0.08,
                "papers":  0.15,
                "other":   0.02,
            },
            seq_len=8192,
            description="Targeted: code + math + reasoning heavy",
        ),
    ]

    scheduler = CurriculumScheduler(stages, TOTAL_STEPS)

    print("=" * 65)
    print("  THREE-STAGE TRAINING CURRICULUM")
    print("=" * 65)

    for stage in stages:
        start_step = int(stage.start_frac * TOTAL_STEPS)
        end_step   = int(stage.end_frac   * TOTAL_STEPS)
        print()
        print(f"  ── {stage.name.upper()} "
              f"(steps {start_step:,}–{end_step:,}, {stage.description}) ──")
        for domain, w in stage.domain_weights.items():
            bar = "█" * int(w * 40)
            print(f"    {domain:<8}  {bar:<40}  {w:.0%}")
        print(f"    seq_len: {stage.seq_len}")

    print(scheduler.visualise())

    # Show weight changes at key milestones
    print()
    print("=" * 65)
    print("  DOMAIN WEIGHTS AT KEY TRAINING MILESTONES")
    print("=" * 65)
    milestones = [0, 50_000, 85_000, 90_000, 95_000, 99_000]
    domains    = list(stages[0].domain_weights.keys())
    print()
    print(f"  {'Step':>8}  {'Stage':<22}", end="")
    for d in domains:
        print(f"  {d:>8}", end="")
    print()
    print(f"  {'':─>8}  {'':─<22}", end="")
    for _ in domains:
        print(f"  {'':─>8}", end="")
    print()

    for step in milestones:
        stage   = scheduler.get_stage(step)
        weights = scheduler.get_weights(step)
        print(f"  {step:>8,}  {stage.name:<22}", end="")
        for d in domains:
            print(f"  {weights.get(d, 0):>8.1%}", end="")
        print()

    # Length curriculum demo
    print()
    print("=" * 65)
    print("  LENGTH CURRICULUM (512 → 8192 tokens)")
    print("=" * 65)
    print()
    lc = LengthCurriculum(
        min_len=512, max_len=8192, total_steps=TOTAL_STEPS,
        schedule="cosine", warmup_frac=0.7
    )

    print(f"  {'Step':>8}  {'% done':>8}  {'seq_len':>10}  {'memory rel.':>14}")
    print(f"  {'':─>8}  {'':─>8}  {'':─>10}  {'':─>14}")
    for step in [0, 10_000, 25_000, 50_000, 70_000, 85_000, 100_000]:
        if step >= TOTAL_STEPS:
            step = TOTAL_STEPS - 1
        l    = lc.get_len(step)
        pct  = step / TOTAL_STEPS * 100
        # Memory scales roughly quadratically with seq_len (attention)
        mem_rel = (l / 512) ** 2
        print(f"  {step:>8,}  {pct:>8.1f}%  {l:>10,}  {mem_rel:>14.1f}×")

    print()
    print("  Memory usage is ~quadratic in seq_len (attention scores).")
    print("  Training at 512 tokens uses 256× less attention memory than 8192.")
    print("  Length curriculum cuts training cost while maintaining final quality.")
''',
    },

    "DoReMi-style Domain Reweighting": {
        "description": "Implement a simplified DoReMi-like algorithm that automatically adjusts domain weights based on per-domain training loss relative to a reference model.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DOREMI-STYLE AUTOMATIC DOMAIN REWEIGHTING
================================================================================

Implements a simplified version of the DoReMi algorithm:
    1. Define a reference model loss per domain (simulated as constant baselines)
    2. At each step, compute excess loss per domain vs reference
    3. Update domain weights: upweight domains where model underperforms
    4. Downweight domains where model already performs well
    5. Visualise weight evolution and convergence

Reference: "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining"
           Xie et al., 2023
================================================================================
"""

import math
import random


# ── Simulated domain loss oracle ──────────────────────────────────────────────

class DomainLossOracle:
    """
    Simulates per-domain losses for a training model.
    In reality these would come from actual model training.

    The model starts uniformly bad (high loss on all domains) and improves
    faster on high-weight domains (more training data → faster learning).
    """

    def __init__(self, domains: list[str], ref_losses: dict[str, float],
                 seed: int = 42):
        self.domains    = domains
        self.ref_losses = ref_losses
        self.rng        = random.Random(seed)
        # Model starts 2.0 units above reference on all domains
        self.model_losses = {d: ref_losses[d] + 2.0 for d in domains}

    def step(self, domain_weights: dict[str, float]):
        """
        Update model losses based on current domain weights.
        Domains with higher weight improve faster (more training data).
        """
        for d in self.domains:
            w   = domain_weights.get(d, 0.0)
            gap = self.model_losses[d] - self.ref_losses[d]
            # Learning rate on this domain proportional to its weight
            decay = 0.002 * (w / (1.0 / len(self.domains)))  # relative to uniform
            self.model_losses[d] = self.ref_losses[d] + max(0.0, gap - decay)
            # Add small noise
            self.model_losses[d] += self.rng.gauss(0, 0.01)

    def get_excess_losses(self) -> dict[str, float]:
        """Excess loss = max(0, model_loss - ref_loss)."""
        return {d: max(0.0, self.model_losses[d] - self.ref_losses[d])
                for d in self.domains}


# ── DoReMi weight updater ─────────────────────────────────────────────────────

class DoReMiWeightUpdater:
    """
    Updates domain weights using the DoReMi algorithm:
        w_d ← w_d × exp(η × excess_loss_d)
        normalise so Σ w_d = 1
    """

    def __init__(self, domains: list[str], initial_weights: dict[str, float],
                 eta: float = 0.1, smoothing: float = 1e-3):
        self.domains  = domains
        self.weights  = initial_weights.copy()
        self.eta      = eta        # step size
        self.smoothing = smoothing  # minimum weight floor (prevent collapse)

    def update(self, excess_losses: dict[str, float]):
        """Update weights based on excess losses."""
        # Multiplicative weight update
        new_weights = {}
        for d in self.domains:
            loss = excess_losses.get(d, 0.0)
            new_weights[d] = self.weights[d] * math.exp(self.eta * loss)

        # Normalise
        total = sum(new_weights.values())
        for d in self.domains:
            new_weights[d] = new_weights[d] / total

        # Apply floor (prevent any domain from collapsing to 0)
        for d in self.domains:
            new_weights[d] = max(new_weights[d], self.smoothing)

        # Renormalise after floor
        total2 = sum(new_weights.values())
        self.weights = {d: w / total2 for d, w in new_weights.items()}

    def get_weights(self) -> dict[str, float]:
        return self.weights.copy()


# ── Full DoReMi simulation ────────────────────────────────────────────────────

def run_doremi_simulation(n_steps: int = 1000) -> tuple:
    """
    Simulate DoReMi domain weight evolution over N training steps.
    Returns history of (step, weights, excess_losses, model_losses).
    """
    domains = ["web", "code", "books", "wiki", "papers"]

    # Reference model losses (lower = easier/better domain for reference model)
    ref_losses = {
        "web":    2.5,   # easier (lots of data in reference model)
        "code":   3.5,   # harder (less in reference, domain-specific)
        "books":  2.8,   # medium
        "wiki":   2.6,   # medium-easy
        "papers": 3.8,   # hardest (technical domain)
    }

    # Start with uniform weights
    initial_weights = {d: 1.0 / len(domains) for d in domains}

    oracle  = DomainLossOracle(domains, ref_losses)
    updater = DoReMiWeightUpdater(domains, initial_weights, eta=0.05)

    history = []

    for step in range(n_steps):
        weights      = updater.get_weights()
        excess       = oracle.get_excess_losses()
        model_losses = oracle.model_losses.copy()

        if step % 50 == 0:
            history.append((step, weights.copy(), excess.copy(), model_losses.copy()))

        # Update oracle (model trains on data according to weights)
        oracle.step(weights)
        # Update weights based on new excess losses
        excess_new = oracle.get_excess_losses()
        updater.update(excess_new)

    return history, domains, ref_losses


def bar(val: float, max_val: float, width: int = 20) -> str:
    filled = int(val / max_val * width) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


if __name__ == "__main__":
    print("=" * 65)
    print("  DOREMI-STYLE AUTOMATIC DOMAIN REWEIGHTING")
    print("=" * 65)
    print()
    print("  DoReMi upweights domains where the model underperforms")
    print("  relative to a reference model trained on uniform data.")
    print()

    history, domains, ref_losses = run_doremi_simulation(n_steps=1000)

    # Print initial state
    step0, w0, ex0, ml0 = history[0]
    print(f"  Initial state (step 0, uniform weights):")
    print(f"  {'Domain':<10} {'Ref loss':>10}  {'Model loss':>12}  "
          f"{'Excess':>10}  {'Weight':>8}")
    print(f"  {'':─<10} {'':─>10}  {'':─>12}  {'':─>10}  {'':─>8}")
    for d in domains:
        print(f"  {d:<10} {ref_losses[d]:>10.3f}  {ml0[d]:>12.3f}  "
              f"{ex0[d]:>10.3f}  {w0[d]:>8.1%}")

    # Print weight evolution
    print()
    print("  Weight evolution (first 3 logged steps):")
    print()
    for step, w, ex, ml in history[:4]:
        print(f"  Step {step:>5}:")
        for d in domains:
            b = bar(w[d], max(w.values()))
            print(f"    {d:<8}  {b}  {w[d]:.1%}  (excess={ex[d]:.3f})")
        print()

    # Print final state and convergence
    step_f, wf, exf, mlf = history[-1]
    print(f"  Final state (step {step_f}):")
    print(f"  {'Domain':<10} {'Ref loss':>10}  {'Model loss':>12}  "
          f"{'Excess':>10}  {'Final w':>8}  {'Δ from uniform':>16}")
    print(f"  {'':─<10} {'':─>10}  {'':─>12}  {'':─>10}  {'':─>8}  {'':─>16}")
    uniform_w = 1.0 / len(domains)
    for d in domains:
        delta = wf[d] - uniform_w
        print(f"  {d:<10} {ref_losses[d]:>10.3f}  {mlf[d]:>12.3f}  "
              f"{exf[d]:>10.3f}  {wf[d]:>8.1%}  {delta:>+15.1%}")

    print()
    print("  Interpretation:")
    print("  • Domains with higher initial excess loss (code, papers)")
    print("    get higher final weights → model spends more time on them")
    print("  • Domains already well-modelled (web, wiki) get lower weights")
    print("  • DoReMi converges to a mix that reduces per-domain excess loss")
    print()
    print("  This is the key DoReMi insight: optimise weights to reduce")
    print("  the gap between the model and a reference trained uniformly.")
    print("  The resulting mix is more 'balanced' across domains than uniform.")
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
    #     from llm_training.visuals.data_mixing import (
    #         MIXING_VISUAL_HTML,
    #         MIXING_VISUAL_HEIGHT,
    #     )
    #     visual_html   = MIXING_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = MIXING_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[15_data_mixing_curriculum.py] Could not load visual: {e}",
    #         stacklevel=2,
    #     )

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