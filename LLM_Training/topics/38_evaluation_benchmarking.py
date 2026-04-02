"""
Evaluation and Benchmarking
============================

Evaluating language models rigorously is as important as training them
well. A model's benchmark scores determine its perceived quality, guide
research decisions, and drive deployment choices. Yet evaluation is rife
with subtle pitfalls: benchmark contamination, gaming through prompting,
task-format sensitivity, and the gap between automated metrics and human
preference. This module covers the full evaluation pipeline — from
perplexity computation to few-shot benchmarking with the LM Evaluation
Harness — and the deeper interpretability of what each metric actually
measures.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Evaluation and Benchmarking"
DISPLAY_NAME = "38 · Evaluation & Benchmarking"
ICON         = "📐"
SUBTITLE     = "Perplexity, MMLU, and LM Harness"


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

### The Evaluation Landscape

LLM evaluation spans a hierarchy of measurements, each capturing different
aspects of model quality:

    Metric class          What it measures               When to use
    ─────────────────────────────────────────────────────────────────────────
    Perplexity            Language modelling quality      Pre-training progress
    Multiple-choice acc   World knowledge, reasoning      Research comparison
    Generation metrics    Response quality (ROUGE, BERTScore) Summarisation
    Code evaluation       Functional correctness          Code models
    LLM-as-judge          Human preference proxy          Instruction models
    Human evaluation      True quality                    Final deployment
    ─────────────────────────────────────────────────────────────────────────

No single metric captures everything. The state of the art is to report
multiple metrics across diverse tasks, and to be explicit about what is
and is not measured.


### Perplexity: The Foundation Metric

**Perplexity** (PPL) is the exponential of the average per-token cross-entropy
loss on a held-out text corpus:

    PPL = exp(H) = exp(-1/N × Σᵢ log P(xᵢ | x₁,...,xᵢ₋₁))

where:
    •   N is the total number of tokens
    •   P(xᵢ | x₁,...,xᵢ₋₁) is the model's assigned probability
    •   H is the average negative log-likelihood (the language model loss)

**Interpretation:**
    PPL = k means the model assigns, on average, probability 1/k to the
    correct next token (as if it were choosing uniformly among k tokens).
    Lower PPL = better model.

**Typical values for English text (WikiText-103):**
    Random (50,000-word vocabulary):   ~50,000
    Unigram language model:            ~1,000
    GPT-2 (117M params):               ~29.4
    GPT-3 (175B params):               ~20.1
    LLaMA-2 7B:                        ~5.7
    LLaMA-2 70B:                       ~3.3
    Human (rough estimate):            ~15–30

**Critical: perplexity is corpus-specific.** A model's perplexity on
WikiText-103 cannot be compared to its perplexity on Pile or C4. The
comparison is only meaningful when both models are evaluated on exactly
the same held-out corpus.

**Stride-based evaluation for long contexts:**
Perplexity should be evaluated with a sliding window, not independently
on non-overlapping chunks. The issue: for a chunk starting at position T,
the model has no context from positions 0 to T-1. The perplexity at the
chunk boundaries is artificially high (models can't use context they
haven't seen).

Correct evaluation with stride S:
    For each position t: evaluate PPL using context from max(0, t - max_context) to t
    This gives every token maximum context.
    Computational cost: (N / S) forward passes instead of 1.


### Few-Shot Benchmarking: The Standard Evaluation Protocol

The modern standard for comparing LLMs is **few-shot prompting on
multiple-choice benchmarks**, evaluated with the LM Evaluation Harness
(EleutherAI, 2021).

**How few-shot evaluation works:**
For a multiple-choice question with answers (A, B, C, D):
    1.  Construct a prompt with k demonstrations (k=0,1,2,5,25 typical)
    2.  Present the test question and measure the model's log-probability
        of each answer choice
    3.  Select the answer with the highest log-probability
    4.  Accuracy = fraction of correct selections

Example (0-shot MMLU, "high school mathematics" category):
    Prompt: "What is 7² - 4?\nA. 45  B. 53  C. 41  D. 49"
    Model log P(A): -2.1,  log P(B): -1.8,  log P(C): -2.4,  log P(D): -2.9
    Prediction: B (highest log-prob) → Correct: B ✓

**Why log-probabilities instead of generation?**
Multiple choice with log-probs is more reliable than generation because:
    •   Generation is sensitive to answer format (the model may generate
        "The answer is B" vs just "B" vs "45")
    •   Log-probability gives a continuous signal even for wrong answers
    •   Results are fully reproducible (no temperature/sampling variance)


### Key Benchmarks and What They Measure

**MMLU (Massive Multitask Language Understanding):**
    57 subject areas, 14,000 questions, 4-way multiple choice.
    Coverage: STEM, humanities, social sciences, professional domains.
    Random baseline: 25% (4 choices)
    Human expert: ~90%
    LLaMA-2 7B: ~45%,  LLaMA-3 70B: ~82%,  GPT-4: ~87%

    What it measures: breadth of world knowledge and reading comprehension.
    Caveat: heavily depends on whether the training data contained these
    specific questions (contamination risk).

**HellaSwag:**
    70,000 multiple-choice questions about completing everyday scenarios.
    Tests common-sense reasoning about physical and social situations.
    Random baseline: 25%
    Human: ~95%
    LLaMA-2 7B: ~77%

    What it measures: physical commonsense reasoning, anti-NLI (naturalness).
    Caveat: many models now exceed human performance on HellaSwag, meaning
    it no longer discriminates between top models.

**ARC (AI2 Reasoning Challenge):**
    ARC-Easy: standard elementary science questions
    ARC-Challenge: hard questions that failed retrieval systems
    Tests factual knowledge and basic scientific reasoning.

**TruthfulQA:**
    817 questions designed to elicit common human misconceptions.
    Tests whether models are truthful vs. confidently wrong.
    Human: ~94%,  GPT-4: ~60%
    Most models do surprisingly poorly (they have been trained on human-
    generated text that contains misconceptions).

    What it measures: calibration and resistance to confident falsehoods.

**HumanEval:**
    164 Python programming problems with unit tests.
    Pass@k metric: probability that at least one of k generated solutions
    passes all unit tests.
    GPT-4: ~67% Pass@1,  LLaMA-3 70B: ~81% Pass@1

    What it measures: functional code generation quality.
    Critical: evaluation requires executing generated code safely.

**MT-Bench:**
    80 multi-turn conversation questions, GPT-4 rated.
    Evaluates instruction following, coherence, creativity.
    Score: 1–10, averaged across 80 conversations.
    GPT-4: ~8.9,  LLaMA-2 70B Chat: ~6.3

    What it measures: instruction-following and conversational quality.
    Caveat: GPT-4-as-judge is biased toward GPT-4-like responses.


### Benchmark Contamination: The Critical Reliability Issue

**Benchmark contamination** occurs when evaluation data (questions, answers,
or similar text) appears in the model's training data. A contaminated
model "knows" the answers not from understanding but from memorisation.

**How contamination happens:**
    •   Web scrapes include pages from academic papers that describe benchmarks
    •   GitHub repos containing benchmark data were scraped for training
    •   The model was fine-tuned on data that includes benchmark examples
    •   Test answers were accidentally included in the training set

**Detection methods:**
    1.  N-gram overlap: Check if any k-grams from test examples appear in
        training data. High overlap → likely contamination.
    2.  Temporal filtering: Use benchmarks released AFTER training cutoff.
    3.  Perplexity test: A model that assigns unusually low PPL to specific
        benchmark examples may have memorised them.
    4.  Variation test: Create paraphrased versions of benchmark questions.
        If accuracy drops significantly, contamination is likely.

**The solution:** Use held-out benchmarks with known collection dates,
or use benchmarks specifically designed to be contamination-resistant
(e.g., GPQA Diamond, which requires graduate-level knowledge and is
difficult to include in standard web scrapes).


### The LM Evaluation Harness

The **LM Evaluation Harness** (Gao et al., 2021) is the standard framework
for evaluating language models across hundreds of tasks in a unified way.

**Architecture:**
    •   Task definitions: YAML files specifying prompts, metrics, few-shot
    •   Model backends: HuggingFace, OpenAI API, vLLM, etc.
    •   Metrics: accuracy, perplexity, BLEU, ROUGE, pass@k

**Usage:**
    pip install lm-eval
    lm_eval --model hf --model_args pretrained=meta-llama/Llama-3-8B \
            --tasks mmlu,hellaswag,arc_challenge \
            --num_fewshot 5 \
            --batch_size 8

**Key evaluation modes:**
    •   loglikelihood: score each answer choice, pick highest (most benchmarks)
    •   generate_until: generate text until stopping token (open-ended tasks)
    •   perplexity: compute PPL on a text corpus

**Custom task example (YAML):**
    task: my_custom_benchmark
    dataset_path: /path/to/data.jsonl
    doc_to_text: "Question: {{question}}\\nAnswer:"
    doc_to_target: "{{answer}}"
    metric_list:
      - metric: acc
        aggregation: mean
    num_fewshot: 5


### Calibration: Does the Model Know What It Doesn't Know?

A well-calibrated model assigns probability p to answers that are correct p%
of the time. Poorly calibrated models are:

    **Overconfident:** Assign 90% probability to answers they get right only 70% of the time.
    **Underconfident:** Assign 60% probability to answers they get right 90% of the time.

**Expected Calibration Error (ECE):**
    Divide confidence scores into bins [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0).
    For each bin: compute |mean_accuracy - mean_confidence|.
    ECE = weighted sum across bins.
    Lower ECE = better calibrated.

**Reliability diagram:** Plot accuracy vs confidence for 10 equal-width bins.
A perfectly calibrated model lies on the diagonal.

**Why calibration matters:**
    •   A calibrated model's confidence is meaningful to users
    •   Under confidence-weighted sampling, calibrated models improve pass@k
    •   Calibration is especially important for medical/legal applications


    **Diagram 1 — Calibration and Reliability:**

    RELIABILITY DIAGRAM: WELL-CALIBRATED vs OVERCONFIDENT
    ════════════════════════════════════════════════════════════════

    Accuracy
    1.0 ┤                        ● (perfect)     ○ (overconfident)
    0.9 ┤                   ●         ○
    0.8 ┤              ●         ○
    0.7 ┤         ●         ○
    0.6 ┤    ●         ○
    0.5 ┤●         ○
    0.4 ┤     ○
    0.3 ┤
        └──────────────────────────────────────────────── Confidence
          0.5   0.6   0.7   0.8   0.9   1.0

    Well-calibrated (●): accuracy ≈ confidence at each bin
    Overconfident (○): model is right only 50% when it claims 70% confidence


### Evaluating Instruction-Following Models

Standard benchmarks (MMLU, HellaSwag) measure base model capabilities.
For instruction-tuned models, additional evaluations are needed:

**AlpacaEval (Dubois et al., 2023):**
    805 prompts, GPT-4 judges which response is better (vs. text-davinci-003).
    Win rate measures: how often does the model win vs. the baseline?
    Caveat: favours verbose responses (length bias in GPT-4 judgements).

**MT-Bench (Zheng et al., 2023):**
    80 multi-turn questions in 8 categories.
    GPT-4 rates responses 1–10 scale.
    More systematic than AlpacaEval.

**IFEval (Zhou et al., 2023):**
    Instruction-following evaluation: 500 prompts with verifiable instructions
    ("respond in at most 3 sentences", "include the word 'telescope'").
    Fully automated: no LLM judge needed.

**LMSYS Chatbot Arena:**
    Crowdsourced head-to-head comparisons between models.
    Elo rating system.
    Gold standard for human preference, but slow and expensive.


### Pass@k Metric for Code Generation

For code benchmarks, accuracy@1 underestimates model capabilities since
models are sampling stochastically. The **pass@k** metric asks:
    "If we generate k code samples, what is the probability that at least one
    of them passes all unit tests?"

**Calculation:**
    pass@k = 1 - C(n-c, k) / C(n, k)

where n = total samples generated, c = correct samples, k = target k-value.
This estimator is unbiased and accounts for sample diversity.

For k=1: use the single-sample accuracy.
For k=10: typically 1.5–2× higher than k=1 for capable models.
For k=100: an upper bound on the model's capability for a task.


### Building Your Own Evaluation Suite

For custom LLM applications, you need custom evaluations:

**Step 1: Define success criteria**
    What exactly does a "correct" response look like for your task?
    Is it a factual answer? A specific format? A style?

**Step 2: Create test cases**
    Collect 100–500 representative examples.
    Include edge cases and failure modes you care about.
    Split into a small dev set (for debugging) and a larger test set.

**Step 3: Define metrics**
    •   Exact match: works for factual answers
    •   Substring match: works for structured outputs
    •   ROUGE/BERTScore: works for generation tasks
    •   LLM-as-judge: works for complex quality assessments
    •   Human evaluation: gold standard, expensive

**Step 4: Run and report**
    Run at least 3 seeds for generative tasks (sample variance).
    Report confidence intervals, not just point estimates.
    Include baseline (random, majority class, simple heuristic).

**Step 5: Check for data contamination**
    Verify your test cases don't appear in training data.
    Consider generating synthetic test cases after training.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
LLM Evaluation Benchmarks Reference

| Benchmark   | Task type          | Metric       | Random baseline | Human  | What it measures              |
|-------------|--------------------|--------------|-----------------|-----------|-----------------------------|
| MMLU        | Multi-choice QA    | Accuracy     | 25%             | ~90%   | World knowledge, reasoning    |
| HellaSwag   | Multi-choice compl.| Accuracy     | 25%             | 95%    | Commonsense reasoning         |
| ARC-C       | Multi-choice QA    | Accuracy     | 25%             | ~80%   | Science knowledge             |
| TruthfulQA  | Multi-choice QA    | Accuracy     | 25%             | 94%    | Truthfulness / calibration    |
| HumanEval   | Code generation    | Pass@1       | ~0%             | ~90%   | Programming ability           |
| MT-Bench    | Multi-turn chat    | 1–10 score   | N/A             | ~9.3   | Instruction following         |
| GPQA        | Multi-choice QA    | Accuracy     | 25%             | ~60%   | Graduate-level science        |
| AlpacaEval  | Open generation    | Win rate     | 50%             | N/A    | Instruction quality vs GPT-3  |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Perplexity Evaluation with Stride": {
        "description": "Implement correct stride-based perplexity evaluation that gives every token maximum context, compare against naive chunked evaluation, and analyse the difference.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
PERPLEXITY EVALUATION — STRIDE-BASED FOR LONG CONTEXTS
================================================================================

Correct perplexity evaluation for language models:
    1. Naive chunked evaluation (incorrect for long contexts)
    2. Stride-based evaluation (correct: each token sees maximum context)
    3. Comparison showing the difference between the two approaches
    4. Bitwise entropy analysis: which positions are easiest/hardest to predict

The key insight: when evaluating PPL on a corpus, every token should
have as much context as possible. Chunking the corpus creates artificial
"context boundaries" where the model has no information.

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Simple language model for demo ────────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab=256, d=128, n_layers=3, max_len=128):
        super().__init__()
        self.embed   = nn.Embedding(vocab, d)
        self.layers  = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*4, bias=False),
                          nn.GELU(), nn.Linear(d*4, d, bias=False))
            for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d)
        self.head    = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.vocab   = vocab
        self.max_len = max_len

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers: h = h + layer(h)
        return self.head(self.norm(h))

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities over vocabulary for each position."""
        with torch.no_grad():
            logits = self.forward(x)
        return F.log_softmax(logits, dim=-1)


# ── Perplexity evaluation functions ──────────────────────────────────────────

def ppl_chunked(model: nn.Module, tokens: torch.Tensor,
                 chunk_size: int) -> float:
    """
    NAIVE (incorrect) perplexity: split corpus into non-overlapping chunks,
    evaluate each independently. Each chunk's first few tokens have no context.

    This underestimates PPL quality — the model does worse than it should
    at chunk boundaries because it has no context to work with.
    """
    total_nll  = 0.0
    total_toks = 0

    for start in range(0, len(tokens) - 1, chunk_size):
        chunk = tokens[start : start + chunk_size + 1]
        if len(chunk) < 2:
            break
        inp, tgt = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
        log_p    = model.log_probs(inp)  # (1, T, V)
        # Gather log-probability of actual next token
        nll = F.nll_loss(log_p[0], tgt[0], reduction="sum")
        total_nll  += nll.item()
        total_toks += tgt.shape[1]

    return math.exp(total_nll / max(total_toks, 1))


def ppl_stride(model: nn.Module, tokens: torch.Tensor,
                max_len: int, stride: int) -> tuple[float, list[float]]:
    """
    CORRECT perplexity with sliding window.

    For each target position t, provide context from max(0, t - max_len) to t.
    The stride controls how many positions we advance per forward pass.
    A smaller stride = more overlap = more context = better estimate.
    A stride = max_len reproduces the chunked approach (no overlap).

    Args:
        tokens:   full token sequence
        max_len:  maximum context window
        stride:   how many positions to advance per step

    Returns:
        ppl:         overall perplexity
        pos_nll:     per-position NLL (for analysis)
    """
    total_nll  = 0.0
    total_toks = 0
    pos_nll    = []   # NLL at each position

    N = len(tokens)

    for begin in range(0, N - 1, stride):
        # Context window: [begin, begin + max_len)
        end        = min(begin + max_len, N)
        window     = tokens[begin : end]

        if len(window) < 2:
            break

        inp  = window[:-1].unsqueeze(0)
        tgt  = window[1:].unsqueeze(0)
        T    = inp.shape[1]

        log_p = model.log_probs(inp)  # (1, T, V)

        # Which positions to actually count (the new ones, not the overlap)?
        # On the first window: count all T positions
        # On subsequent windows: only count the new `stride` positions
        #   (positions [stride, T) were already counted in the previous window)
        if begin == 0:
            count_start = 0
        else:
            count_start = max(0, T - stride)

        for t in range(count_start, T):
            nll_t = -log_p[0, t, tgt[0, t].item()].item()
            pos_nll.append(nll_t)
            total_nll  += nll_t
            total_toks += 1

    ppl = math.exp(total_nll / max(total_toks, 1))
    return ppl, pos_nll


# ── Per-position analysis ─────────────────────────────────────────────────────

def analyse_position_entropy(pos_nll: list[float],
                               n_bins: int = 10) -> dict:
    """
    Analyse how difficulty varies across positions in a sequence.
    Early positions (less context) should have higher NLL.
    """
    bin_size  = max(1, len(pos_nll) // n_bins)
    bin_means = []
    for b in range(n_bins):
        start  = b * bin_size
        end    = min(start + bin_size, len(pos_nll))
        chunk  = pos_nll[start:end]
        if chunk:
            bin_means.append((start, sum(chunk) / len(chunk)))
    return bin_means


# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    VOCAB    = 256
    D        = 128
    N_LAYERS = 2
    MAX_LEN  = 64     # context window
    N_TOKENS = 2000   # length of eval corpus

    print("=" * 65)
    print("  PERPLEXITY EVALUATION: CHUNKED vs STRIDE-BASED")
    print(f"  vocab={VOCAB}, max_context={MAX_LEN}, corpus_size={N_TOKENS}")
    print("=" * 65)
    print()

    model  = TinyLM(VOCAB, D, N_LAYERS, MAX_LEN)
    # Simulate "trained" model: run a few training steps
    opt    = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(50):
        x = torch.randint(0, VOCAB, (4, 32))
        y = x[:, 1:]
        loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    # Create evaluation corpus
    torch.manual_seed(0)
    eval_tokens = torch.randint(0, VOCAB, (N_TOKENS,))

    print("  Evaluating with different methods:")
    print()

    # Chunked (incorrect)
    ppl_chunk_64  = ppl_chunked(model, eval_tokens, chunk_size=64)
    ppl_chunk_128 = ppl_chunked(model, eval_tokens, chunk_size=128)

    # Stride-based (correct)
    ppl_s64, nll_s64   = ppl_stride(model, eval_tokens, MAX_LEN, stride=64)
    ppl_s32, nll_s32   = ppl_stride(model, eval_tokens, MAX_LEN, stride=32)
    ppl_s8,  nll_s8    = ppl_stride(model, eval_tokens, MAX_LEN, stride=8)
    ppl_s1,  nll_s1    = ppl_stride(model, eval_tokens, MAX_LEN, stride=1)

    print(f"  {'Method':<35}  {'PPL':>10}  Notes")
    print(f"  {'':─<35}  {'':─>10}  {'':─}")
    print(f"  {'Chunked (chunk=64, no overlap)':<35}  {ppl_chunk_64:>10.2f}  ← incorrect")
    print(f"  {'Chunked (chunk=128)':<35}  {ppl_chunk_128:>10.2f}  ← incorrect")
    print(f"  {'Stride=64 (same as chunked)':<35}  {ppl_s64:>10.2f}")
    print(f"  {'Stride=32 (50% overlap)':<35}  {ppl_s32:>10.2f}")
    print(f"  {'Stride=8  (87% overlap)':<35}  {ppl_s8:>10.2f}")
    print(f"  {'Stride=1  (max overlap, correct)':<35}  {ppl_s1:>10.2f}  ← correct")
    print()
    print(f"  Difference (chunked vs stride=1): {ppl_chunk_64/ppl_s1:.2f}× worse")
    print()

    # Per-position analysis
    bins = analyse_position_entropy(nll_s1, n_bins=8)
    print("  Per-position NLL (stride=1):")
    print("  (Higher NLL = harder to predict = less context available)")
    print()
    print(f"  {'Position range':>16}  {'Avg NLL':>10}  Bar")
    print(f"  {'':─>16}  {'':─>10}  {'':─}")
    max_nll_bin = max(m for _, m in bins)
    bars = " ▁▂▃▄▅▆▇█"
    for pos, mean_nll in bins:
        bar = "█" * int(mean_nll / max_nll_bin * 25)
        ppl_b = math.exp(mean_nll)
        print(f"  pos {pos:>6}–{pos+len(nll_s1)//8:<6}  {mean_nll:>10.3f}  {bar}")

    print()
    print("  Key insight: early positions have higher NLL (less context).")
    print("  Chunked evaluation artificially re-creates these high-NLL positions")
    print("  at every chunk boundary, inflating the measured perplexity.")
    print()
    print("  Always use stride ≤ max_len/2 for accurate perplexity evaluation.")
''',
    },

    "Few-Shot Benchmark Evaluator": {
        "description": "Implement a few-shot multiple-choice evaluator that computes log-probability-based accuracy on benchmark questions — the same method used by the LM Evaluation Harness.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
FEW-SHOT BENCHMARK EVALUATOR
================================================================================

Implements the evaluation method used by lm-eval-harness:
    1. Multiple-choice questions with log-probability scoring
    2. Few-shot prompting (0-shot, 5-shot, 25-shot)
    3. Length-normalised scoring (prevents bias toward short answers)
    4. Calibration measurement (ECE)
    5. Confidence interval computation via bootstrap

This is the gold standard evaluation methodology for language models.

================================================================================
"""

import math
import random
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class MultipleChoiceQuestion:
    """One multiple-choice question."""
    question:   str
    choices:    list[str]       # ["A. Paris", "B. London", "C. Berlin", "D. Rome"]
    answer_idx: int             # 0-indexed correct answer
    category:   str = "general"


@dataclass
class EvalResult:
    """Result for one question."""
    correct:        bool
    predicted_idx:  int
    true_idx:       int
    log_probs:      list[float]
    confidence:     float   # max normalised probability


# ── Log-probability scoring ───────────────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab=512, d=128, n_layers=3):
        super().__init__()
        self.embed   = nn.Embedding(vocab, d)
        self.layers  = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*4, bias=False),
                          nn.GELU(), nn.Linear(d*4, d, bias=False))
            for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d)
        self.head    = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        h = self.embed(x)
        for l in self.layers: h = h + l(h)
        return self.head(self.norm(h))

    def sequence_log_prob(self, input_ids: torch.Tensor,
                           target_ids: torch.Tensor) -> float:
        """
        Compute sum of log-probs for target_ids given input_ids as context.
        This is how we score each answer choice.
        """
        with torch.no_grad():
            full_seq = torch.cat([input_ids, target_ids]).unsqueeze(0)
            logits   = self.forward(full_seq)
            log_probs = F.log_softmax(logits, dim=-1)

        # We want log-probs of target_ids tokens, conditioned on all preceding
        total_lp = 0.0
        ctx_len  = len(input_ids)
        for t, target_tok in enumerate(target_ids.tolist()):
            # logits[ctx_len + t - 1] predicts ctx_len + t (the t-th target token)
            total_lp += log_probs[0, ctx_len + t - 1, target_tok].item()
        return total_lp


# ── Tokeniser simulation ──────────────────────────────────────────────────────

def simple_tokenise(text: str, vocab_size: int = 512) -> torch.Tensor:
    """Word-level tokeniser simulation."""
    tokens = []
    for word in text.lower().split():
        tok = 3 + abs(hash(word)) % (vocab_size - 3)
        tokens.append(tok)
    return torch.tensor(tokens, dtype=torch.long)


# ── Few-shot evaluator ────────────────────────────────────────────────────────

class FewShotEvaluator:
    """
    Evaluates a model on multiple-choice benchmarks using the log-probability method.
    Matches the methodology of EleutherAI's lm-eval-harness.
    """

    def __init__(self, model: nn.Module, vocab_size: int = 512):
        self.model      = model
        self.vocab_size = vocab_size

    def build_prompt(self,
                      question: MultipleChoiceQuestion,
                      few_shot_examples: list[MultipleChoiceQuestion],
                      n_shot: int = 0) -> str:
        """
        Build a few-shot prompt.

        0-shot: just the question and choices
        k-shot: k examples with answers, then the question
        """
        prompt = ""
        # Add k-shot examples with answers
        for ex in few_shot_examples[:n_shot]:
            prompt += f"Question: {ex.question}\n"
            prompt += "\n".join(ex.choices) + "\n"
            prompt += f"Answer: {chr(65 + ex.answer_idx)}\n\n"

        # Add the test question (without answer)
        prompt += f"Question: {question.question}\n"
        prompt += "\n".join(question.choices) + "\n"
        prompt += "Answer:"
        return prompt

    def score_choices(self,
                       context: str,
                       choices: list[str],
                       length_normalise: bool = True) -> list[float]:
        """
        Score each answer choice by computing its log-probability.

        context: the prompt up to and including "Answer:"
        choices: list of answer strings to score (e.g., ["A", "B", "C", "D"])
        length_normalise: divide by number of tokens (prevents bias toward short answers)
        """
        context_ids = simple_tokenise(context, self.vocab_size)
        scores = []

        for choice in choices:
            choice_ids = simple_tokenise(choice, self.vocab_size)
            if len(choice_ids) == 0:
                scores.append(float("-inf"))
                continue

            log_prob = self.model.sequence_log_prob(context_ids, choice_ids)

            if length_normalise and len(choice_ids) > 0:
                log_prob = log_prob / len(choice_ids)

            scores.append(log_prob)

        return scores

    def evaluate_question(self,
                            question: MultipleChoiceQuestion,
                            few_shot_examples: list[MultipleChoiceQuestion],
                            n_shot: int = 0,
                            length_normalise: bool = True) -> EvalResult:
        """Evaluate one question and return the result."""
        context = self.build_prompt(question, few_shot_examples, n_shot)

        # Score each choice letter (A, B, C, D, ...)
        choice_letters = [chr(65 + i) for i in range(len(question.choices))]
        scores         = self.score_choices(context, choice_letters, length_normalise)

        # Convert to probabilities for confidence
        scores_t  = torch.tensor(scores, dtype=torch.float32)
        probs     = F.softmax(scores_t, dim=0).tolist()

        predicted = scores.index(max(scores))
        correct   = (predicted == question.answer_idx)
        confidence = probs[predicted]

        return EvalResult(
            correct       = correct,
            predicted_idx = predicted,
            true_idx      = question.answer_idx,
            log_probs     = scores,
            confidence    = confidence,
        )

    def evaluate_dataset(self,
                          questions: list[MultipleChoiceQuestion],
                          few_shot_pool: list[MultipleChoiceQuestion],
                          n_shot: int = 0,
                          length_normalise: bool = True) -> dict:
        """
        Evaluate on a full dataset and return aggregate metrics.
        """
        results    = []
        by_category = {}

        for q in questions:
            # Don't include q itself in few-shot pool
            pool = [ex for ex in few_shot_pool if ex is not q]
            result = self.evaluate_question(q, pool, n_shot, length_normalise)
            results.append(result)

            by_category.setdefault(q.category, []).append(result)

        n          = len(results)
        accuracy   = sum(r.correct for r in results) / n
        avg_conf   = sum(r.confidence for r in results) / n

        # Expected Calibration Error (ECE)
        ece = self._compute_ece(results)

        # Per-category accuracy
        cat_acc = {
            cat: sum(r.correct for r in rs) / len(rs)
            for cat, rs in by_category.items()
        }

        # Bootstrap confidence interval on accuracy
        ci = self._bootstrap_ci([r.correct for r in results])

        return {
            "accuracy":         accuracy,
            "n_questions":      n,
            "n_correct":        sum(r.correct for r in results),
            "avg_confidence":   avg_conf,
            "ece":              ece,
            "ci_95":            ci,
            "by_category":      cat_acc,
        }

    def _compute_ece(self, results: list[EvalResult], n_bins: int = 5) -> float:
        """Compute Expected Calibration Error."""
        bins = [[] for _ in range(n_bins)]
        for r in results:
            bin_idx = min(n_bins - 1, int(r.confidence * n_bins))
            bins[bin_idx].append(r)

        ece = 0.0
        n   = len(results)
        for b in bins:
            if not b:
                continue
            acc  = sum(r.correct for r in b) / len(b)
            conf = sum(r.confidence for r in b) / len(b)
            ece += (len(b) / n) * abs(acc - conf)

        return ece

    def _bootstrap_ci(self, correct_list: list[bool],
                        n_boot: int = 1000, alpha: float = 0.05) -> tuple:
        """Bootstrap 95% CI on accuracy."""
        n     = len(correct_list)
        accs  = []
        rng   = random.Random(42)
        for _ in range(n_boot):
            sample = [rng.choice(correct_list) for _ in range(n)]
            accs.append(sum(sample) / n)
        accs.sort()
        lo = accs[int(n_boot * alpha / 2)]
        hi = accs[int(n_boot * (1 - alpha / 2))]
        return (lo, hi)


# ── Synthetic benchmark ────────────────────────────────────────────────────────

def make_synthetic_benchmark(n_questions: int = 60,
                               seed: int = 42) -> list[MultipleChoiceQuestion]:
    """Create a synthetic benchmark for demonstration."""
    rng = random.Random(seed)
    categories = ["math", "science", "history", "language"]
    questions  = []

    for i in range(n_questions):
        cat    = rng.choice(categories)
        n_ch   = 4
        # The "correct" answer has a higher first-word hash (simulating model knowledge)
        correct = rng.randint(0, n_ch - 1)
        choices = [f"{chr(65+j)}. Option {rng.randint(100,999)}" for j in range(n_ch)]

        questions.append(MultipleChoiceQuestion(
            question   = f"Synthetic question {i+1} in {cat}",
            choices    = choices,
            answer_idx = correct,
            category   = cat,
        ))
    return questions


# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import copy

    torch.manual_seed(42)
    random.seed(42)

    VOCAB = 512
    model = TinyLM(VOCAB, 128, 2)
    # Brief training to give model some structure
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(30):
        x = torch.randint(0, VOCAB, (4, 24))
        y = x[:, 1:]; loss = F.cross_entropy(model(x[:,:-1]).view(-1, VOCAB), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    evaluator = FewShotEvaluator(model, vocab_size=VOCAB)
    benchmark = make_synthetic_benchmark(n_questions=60)

    train_pool  = benchmark[:40]   # few-shot examples
    test_set    = benchmark[40:]   # evaluation questions

    print("=" * 65)
    print("  FEW-SHOT BENCHMARK EVALUATION")
    print(f"  {len(test_set)} test questions, {len(train_pool)} in few-shot pool")
    print("=" * 65)
    print()

    for n_shot in [0, 1, 5]:
        results = evaluator.evaluate_dataset(
            test_set, train_pool, n_shot=n_shot, length_normalise=True
        )
        ci = results["ci_95"]
        print(f"  {n_shot}-shot accuracy: {results['accuracy']*100:.1f}%  "
              f"(95% CI: {ci[0]*100:.1f}%–{ci[1]*100:.1f}%)  "
              f"ECE: {results['ece']:.3f}")

    print()
    print("  Per-category accuracy (5-shot):")
    results_5shot = evaluator.evaluate_dataset(
        test_set, train_pool, n_shot=5, length_normalise=True
    )
    for cat, acc in sorted(results_5shot["by_category"].items()):
        bar = "█" * int(acc * 20)
        print(f"    {cat:<12}: {acc*100:>5.1f}%  |{bar:<20}|")

    print()
    print(f"  Random baseline:   {100/4:.1f}%  (4 choices)")
    print(f"  Model 5-shot:      {results_5shot['accuracy']*100:.1f}%")
    print(f"  ECE (calibration): {results_5shot['ece']:.3f}  "
          f"(0 = perfect calibration)")
    print()
    print("  Notes on interpretation:")
    print("  • Always compare against random baseline for multiple-choice")
    print("  • Report confidence intervals, not just point estimates")
    print("  • Check for benchmark contamination in your training data")
    print("  • ECE < 0.05 is considered well-calibrated")
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
    #     from llm_training.visuals.evaluation_benchmarking import (
    #         EVAL_VISUAL_HTML,
    #         EVAL_VISUAL_HEIGHT,
    #     )
    #     visual_html   = EVAL_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = EVAL_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[38_evaluation_benchmarking.py] Could not load visual: {e}",
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