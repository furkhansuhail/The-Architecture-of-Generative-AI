"""
Expert Parallelism and Mixture of Experts
==========================================

Mixture of Experts (MoE) replaces the dense FFN in each Transformer block
with a collection of specialist "expert" networks, activating only a small
subset for each token. This allows the total parameter count to scale
dramatically without a proportional increase in computation per token.
Expert parallelism then distributes these experts across GPUs, making
MoE models tractable on large clusters.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Expert Parallelism and Mixture of Experts"
DISPLAY_NAME = "22 · Expert Parallelism & MoE"
ICON         = "🧠"
SUBTITLE     = "Mixture of Experts and Routing"


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

### The Dense FFN Bottleneck

In a standard dense Transformer, every token passes through every weight in
the FFN at every layer. A 70B parameter model applies all 70B weights to
every token. This means:
    •   The compute cost scales as O(P × T)  (P params, T tokens)
    •   Training 1T tokens requires 70B × 1T ≈ 7×10²² FLOPs
    •   Doubling parameters doubles compute for the same data

The **conditional computation** hypothesis: not every parameter is needed for
every input. A model of code needs different knowledge than a model of
biomedical text. If we could selectively apply different subsets of parameters
to different inputs, we could scale parameters without scaling compute.

This is exactly what **Mixture of Experts (MoE)** achieves.


### The MoE Architecture

In an MoE Transformer, each FFN layer is replaced by:
    1.  A **router** (a small linear layer + softmax)
    2.  E **expert networks** (each a standard FFN of the same size)

For each token, the router selects the top-k experts (typically k=1 or k=2)
and routes the token's hidden state to only those experts. The experts'
outputs are then combined with the routing weights.

    h' = Σₑ (router_weight_e × Expert_e(h))     for the top-k selected experts

**Key parameters:**
    •   E = total number of experts (e.g., 8 or 64)
    •   k = active experts per token (typically 1 or 2)
    •   Sparsity ratio = k/E  (e.g., 2/8 = 25% of experts activated)

**Example — Mixtral 8×7B:**
    Total parameters:    46B (8 experts × ~5.8B each)
    Active per token:    2 experts  → ~13B active per token
    Active compute:      ~2× a 7B dense model's compute per token

Despite having 46B total parameters, Mixtral 8×7B runs at the compute cost
of a ~13B dense model — it is 3.5× more parameter-efficient.


    **Diagram 1 — Dense FFN vs MoE FFN:**

    DENSE FFN:
    ════════════════════════════════════════════════════════════════

    Input h → [FFN (d × d_ff × d)] → h'

    Every token goes through ALL weights (d × d_ff × d).
    Compute: O(B × T × d × d_ff)

    MoE FFN (E=8, k=2):
    ════════════════════════════════════════════════════════════════

    Input h → [Router: h → scores ∈ ℝ^E] → top-2 experts selected
                   │
                   ├─→ Expert 1 (idle)
                   ├─→ Expert 2 (active) ─┐
                   ├─→ Expert 3 (idle)     │
                   ├─→ Expert 4 (idle)     │ weighted sum
                   ├─→ Expert 5 (active) ─┤
                   ├─→ Expert 6 (idle)     │
                   ├─→ Expert 7 (idle)     │
                   └─→ Expert 8 (idle)     │
                                           ▼
                                          h'

    Each expert is an independent FFN.
    Only 2/8 experts compute per token.
    Compute: O(B × T × d × d_ff × k/E)  ← k/E of dense cost


### The Router: Softmax + Top-k Selection

The router is a linear layer W_g ∈ ℝ^(d × E):

    logits = h · W_g                         (scores for each expert)
    probs  = Softmax(logits)                 (routing probabilities)
    top-k  = indices of k largest probs
    weights = probs[top-k] / sum(probs[top-k])  (renormalise top-k)

    h' = Σᵢ∈top-k  weights[i] × Expert_i(h)

**Router noise (auxiliary loss trick):**
A problem with vanilla top-k routing: the router tends to "collapse" —
it learns to always route to a small subset of experts (expert collapse).
This wastes the capacity of most experts.

**Auxiliary load-balancing loss** (Shazeer et al., 2017):
Add a loss term that encourages equal load across experts:

    L_aux = α × (E / T) × Σₑ f_e × p_e

where:
    f_e = fraction of tokens routed to expert e
    p_e = average routing probability for expert e
    α   = auxiliary loss weight (typically 0.01)

This penalises any expert from receiving too many or too few tokens.


### Expert Parallelism

In expert parallelism (EP), each GPU hosts a subset of experts:
    •   With E=8 experts and EP=8, each GPU hosts 1 expert
    •   When tokens are routed to expert i, they must be gathered on GPU i
    •   After expert computation, results are scattered back to origin GPUs

The communication pattern is **All-to-All**:
    •   Tokens from all GPUs are sent to whichever GPU holds their expert
    •   Results are returned to the originating GPUs

**All-to-All vs All-Reduce:**
    All-to-All sends different data to different destinations.
    Each GPU sends (and receives) B × T / E × d data.
    For E=8 GPUs and T=2048 tokens: each GPU sends 256 × d tokens.
    In contrast, all-reduce sends all parameters to all GPUs (P × 2 bytes).

All-to-All is token-routing: each token goes to its designated expert GPU.


    **Diagram 2 — Expert Parallelism Communication:**

    EXPERT PARALLELISM (E=4 experts, 4 GPUs, each GPU holds 1 expert)
    ════════════════════════════════════════════════════════════════

    Batch: 8 tokens,  routing assigns:
    Token 0 → Expert 2    Token 4 → Expert 0
    Token 1 → Expert 0    Token 5 → Expert 1
    Token 2 → Expert 3    Token 6 → Expert 2
    Token 3 → Expert 1    Token 7 → Expert 3

    All-to-All SEND:
    GPU0 (Expert 0): sends tokens {1,4} to GPU0 (stays local)... wait
    Actually: ALL GPUs have ALL tokens (replicated or split by DP)
    Each token goes to the GPU hosting its selected expert.

    GPU0 → GPU2: token 0 (to Expert 2)
    GPU0 → GPU0: token 1 (to Expert 0, stays local)
    GPU0 → GPU3: token 2 (to Expert 3)
    GPU0 → GPU1: token 3 (to Expert 1)
    ...

    Compute (each GPU computes its expert for assigned tokens):
    GPU0: Expert 0  processes tokens {1, 4}
    GPU1: Expert 1  processes tokens {3, 5}
    GPU2: Expert 2  processes tokens {0, 6}
    GPU3: Expert 3  processes tokens {2, 7}

    All-to-All RETURN: results sent back to origin GPUs


### Load Balancing: The Capacity Factor

With top-k routing, each expert receives a variable number of tokens.
Some experts may be overloaded; others underloaded. To bound the memory
and compute for each expert, a **capacity factor** C is used:

    Expert capacity = C × (B × T) / E   (maximum tokens per expert)

Tokens that would exceed this capacity are **dropped** (not processed by
any expert — they pass through with a zero contribution). A capacity factor
of C=1.0 means each expert gets exactly its fair share; C=1.25 allows 25%
overhead for imbalanced routing.

**The drop-token problem:** Dropped tokens receive zero contribution from
the expert layer, degrading quality. This is why load-balancing losses are
important — they reduce routing imbalance and thus reduce token dropping.


### Token Choice vs Expert Choice Routing

**Token Choice (top-k, standard):**
Each token independently selects its top-k experts.
    •   Tokens compete for expert capacity → overflow possible
    •   Each expert receives variable load
    •   Requires capacity factor and auxiliary loss

**Expert Choice (Zoph et al., 2022):**
Each expert independently selects the top-s tokens it wants to process.
    •   Experts receive exactly s tokens (perfect load balance)
    •   Some tokens may not be processed by any expert (or processed by all)
    •   No dropped tokens, no auxiliary loss needed
    •   Used in: Mixtral 8×22B alternative experiments, some Google models


### MoE Architectures in Production

    Model               Experts   k   Active params   Total params
    ──────────────────────────────────────────────────────────────────
    GShard (Google)     2048      2   N/A             600B
    Switch Transformer  4096      1   N/A             1.6T
    Mixtral 8×7B        8         2   ~13B            46B
    Mixtral 8×22B       8         2   ~39B            141B
    Grok-1 (xAI)        8         2   unknown         314B
    DBRX (Databricks)   16        4   36B             132B
    DeepSeek-V2         160       6   21B             236B
    Llama-4 (Meta)      16        1   ~17B            109B
    ──────────────────────────────────────────────────────────────────

**Key trend:** models are moving toward more experts with fewer active at once.
DeepSeek-V2's 160 experts / 6 active configuration provides extreme parameter
efficiency: 6/160 = 3.75% of experts are active per token.


### Shared Expert vs Routed Expert

**DeepSeek-V2** introduced "shared experts": some experts are always activated
for all tokens (like a dense FFN baseline), while other experts are routing-
dependent. This prevents the experts from learning only specialist knowledge
at the cost of losing common knowledge.

    h' = SharedExpert(h) + Σᵢ∈top-k weights[i] × RoutedExpert_i(h)

This hybrid design improves both quality and training stability.


### Fine-Grained Expert Design (DeepSeek-V3)

**Fine-grained MoE:** instead of E large experts, use E×G small experts,
each 1/G the size of a standard expert. Select k×G of them per token.
This provides more flexibility: with more but smaller experts, the routing
can express more nuanced specialisation.

    DeepSeek-V3: 256 fine-grained experts (each ~1/32 of standard FFN),
    top-8 selected per token. Effective active compute ≈ standard FFN.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Dense FFN vs Sparse MoE Comparison

| Property                | Dense FFN             | Sparse MoE (E experts, k active)  |
|-------------------------|-----------------------|------------------------------------|
| Parameters per FFN      | d × d_ff              | E × (d × d_ff)                     |
| Active params per token | d × d_ff              | k × (d × d_ff) = k/E of dense      |
| FLOPs per token         | 2 × d × d_ff          | 2 × k × d × d_ff = k/E of dense    |
| Training efficiency     | 1× baseline           | k/E × (parameter efficiency)       |
| Communication           | None (within GPU)     | All-to-All (expert routing)        |
| Load balancing needed?  | No                    | Yes (auxiliary loss)                |
| Dropped tokens?         | Never                 | Possible (capacity overflow)       |
| Used by                 | LLaMA, GPT-4, etc.   | Mixtral, DeepSeek, Grok, DBRX      |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "MoE Layer — From Scratch": {
        "description": "Complete MoE FFN implementation: top-k router with load-balancing loss, expert networks, and capacity-based token dropping.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MIXTURE OF EXPERTS (MoE) FFN — FROM SCRATCH
================================================================================

Implements a complete MoE FFN layer:
    1. Top-k router with learned routing probabilities
    2. E independent expert FFN networks
    3. Token-level routing with capacity factor
    4. Load-balancing auxiliary loss
    5. Comparison with dense FFN

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class MoEConfig:
    d_model:         int   = 128
    d_expert:        int   = 512    # d_ff per expert (matches dense d_ff / k for fair comparison)
    n_experts:       int   = 8
    k:               int   = 2      # top-k experts per token
    capacity_factor: float = 1.25   # allow 25% overflow per expert
    aux_loss_weight: float = 0.01   # λ for load-balancing loss


# ── Expert network ────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """One expert: a standard 2-layer FFN."""
    def __init__(self, d_model: int, d_expert: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_expert, bias=False)
        self.fc2 = nn.Linear(d_expert, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ── Router ────────────────────────────────────────────────────────────────────

class TopKRouter(nn.Module):
    """
    Top-k router for MoE.

    For each token, computes routing probabilities over all experts
    and selects the top-k experts. Returns:
        - dispatch_mask: (B*T, E, k) — which token goes to which expert slot
        - combine_weights: (B*T, E) — routing weights for combining outputs
        - aux_loss: load-balancing auxiliary loss
    """

    def __init__(self, d_model: int, n_experts: int, k: int):
        super().__init__()
        self.n_experts = n_experts
        self.k         = k
        self.W_gate    = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, h: torch.Tensor) -> tuple:
        """
        h: (B, T, d_model) — hidden states
        Returns: (topk_indices, topk_weights, aux_loss)
        """
        N, d = h.shape   # N = B*T flattened

        # Routing logits and probabilities
        logits = self.W_gate(h)          # (N, E)
        probs  = F.softmax(logits, dim=-1)  # (N, E)

        # Top-k selection
        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)  # (N, k)

        # Renormalise top-k weights
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # (N, k)

        # Auxiliary load-balancing loss
        # f_e = fraction of tokens routing to expert e
        # p_e = mean routing probability for expert e
        f_e = torch.zeros(self.n_experts, device=h.device)
        for k_idx in range(self.k):
            counts = torch.bincount(topk_idx[:, k_idx], minlength=self.n_experts)
            f_e += counts.float() / N

        p_e = probs.mean(dim=0)  # (E,)

        aux_loss = self.n_experts * (f_e * p_e).sum()

        return topk_idx, topk_weights, aux_loss


# ── MoE FFN Layer ─────────────────────────────────────────────────────────────

class MoEFFN(nn.Module):
    """
    Sparse Mixture of Experts FFN.

    Replaces a dense FFN with E expert networks + a top-k router.
    Tokens are dispatched to their selected experts, computed,
    then recombined with routing weights.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg     = cfg
        self.router  = TopKRouter(cfg.d_model, cfg.n_experts, cfg.k)
        self.experts = nn.ModuleList([
            Expert(cfg.d_model, cfg.d_expert) for _ in range(cfg.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, d_model)
        Returns: (output, aux_loss)
        """
        B, T, d = x.shape
        N       = B * T
        h       = x.view(N, d)

        # 1. Router: get expert assignments and weights
        topk_idx, topk_weights, aux_loss = self.router(h)
        # topk_idx:     (N, k) — expert indices
        # topk_weights: (N, k) — routing weights

        # 2. Dispatch tokens to experts and compute
        output = torch.zeros(N, d, device=x.device, dtype=x.dtype)

        for k_i in range(self.cfg.k):
            # Process one expert assignment at a time
            for e_idx in range(self.cfg.n_experts):
                # Find tokens assigned to expert e_idx in slot k_i
                token_mask = (topk_idx[:, k_i] == e_idx)
                if not token_mask.any():
                    continue

                # Apply capacity factor: limit tokens per expert
                n_tokens   = token_mask.sum().item()
                max_tokens = int(self.cfg.capacity_factor * N / self.cfg.n_experts)

                if n_tokens > max_tokens:
                    # Drop overflow tokens (keep first max_tokens)
                    token_indices = token_mask.nonzero(as_tuple=True)[0]
                    kept = token_indices[:max_tokens]
                    expert_mask = torch.zeros(N, dtype=torch.bool, device=x.device)
                    expert_mask[kept] = True
                else:
                    expert_mask = token_mask

                tokens = h[expert_mask]                       # (n, d)
                expert_out = self.experts[e_idx](tokens)      # (n, d)

                # Weight by routing probability and add to output
                weights = topk_weights[expert_mask, k_i].unsqueeze(-1)  # (n, 1)
                output[expert_mask] += weights * expert_out

        return output.view(B, T, d), aux_loss * self.cfg.aux_loss_weight


# ── Dense FFN for comparison ──────────────────────────────────────────────────

class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = MoEConfig(d_model=128, d_expert=256, n_experts=8, k=2)
    moe = MoEFFN(cfg)

    # Dense FFN with same total parameters (for comparison)
    # MoE has E × (d × d_expert) parameters
    # Dense with same total: d × d_ff_dense where d_ff_dense = E × d_expert
    d_ff_dense = cfg.n_experts * cfg.d_expert  # same total params
    d_ff_equiv = cfg.k * cfg.d_expert           # same active params (fair FLOP comparison)
    dense_same_total  = DenseFFN(cfg.d_model, d_ff_dense)
    dense_same_active = DenseFFN(cfg.d_model, d_ff_equiv)

    B, T = 4, 16
    x    = torch.randn(B, T, cfg.d_model)

    print("=" * 62)
    print("  MIXTURE OF EXPERTS FFN")
    print(f"  E={cfg.n_experts} experts, k={cfg.k} active, d_expert={cfg.d_expert}")
    print("=" * 62)
    print()

    # Forward pass
    out_moe,  aux_loss = moe(x)
    out_dense_total    = dense_same_total(x)
    out_dense_active   = dense_same_active(x)

    moe_params    = sum(p.numel() for p in moe.parameters())
    dense_p_total = sum(p.numel() for p in dense_same_total.parameters())
    dense_p_act   = sum(p.numel() for p in dense_same_active.parameters())

    print(f"  Output shapes: all {out_moe.shape}")
    print()
    print(f"  {'Model':<28} {'Params':>10}  {'Active/token':>14}")
    print(f"  {'':─<28} {'':─>10}  {'':─>14}")
    print(f"  {'MoE (E=8, k=2)':<28} {moe_params:>10,}  "
          f"{cfg.k * cfg.d_model * cfg.d_expert * 2:>14,}")
    print(f"  {'Dense (same total params)':<28} {dense_p_total:>10,}  "
          f"{dense_p_total:>14,}")
    print(f"  {'Dense (same active FLOPs)':<28} {dense_p_act:>10,}  "
          f"{dense_p_act:>14,}")

    print()
    print(f"  MoE compute efficiency: {cfg.k}/{cfg.n_experts} = "
          f"{cfg.k/cfg.n_experts:.1%} of dense (same total params)")
    print(f"  Auxiliary loss: {aux_loss.item():.4f}")

    # Load balancing analysis
    print()
    print("=" * 62)
    print("  ROUTING DISTRIBUTION ANALYSIS")
    print("=" * 62)
    print()

    with torch.no_grad():
        logits      = moe.router.W_gate(x.view(-1, cfg.d_model))
        probs       = F.softmax(logits, dim=-1)
        topk_idx, _, _ = moe.router(x.view(-1, cfg.d_model))

    # Count how many tokens went to each expert
    expert_counts = torch.zeros(cfg.n_experts)
    for k_i in range(cfg.k):
        for e in range(cfg.n_experts):
            expert_counts[e] += (topk_idx[:, k_i] == e).sum().float()

    total_assignments = expert_counts.sum()
    print(f"  Total token-expert assignments: {int(total_assignments)}")
    print(f"  Fair share per expert: {int(total_assignments / cfg.n_experts)}")
    print()
    print(f"  {'Expert':>8}  {'Tokens assigned':>18}  {'Load fraction':>15}  {'vs fair':>10}")
    print(f"  {'':─>8}  {'':─>18}  {'':─>15}  {'':─>10}")
    fair_share = total_assignments / cfg.n_experts
    for e in range(cfg.n_experts):
        count = expert_counts[e].item()
        frac  = count / total_assignments
        ratio = count / fair_share
        flag  = " ⚠️ overloaded" if ratio > 1.5 else " ✓" if ratio > 0.5 else " ⚠️ underused"
        print(f"  {e:>8}  {int(count):>18}  {frac:>14.1%}  {ratio:>9.2f}×{flag}")

    print()
    print("  Auxiliary loss penalises deviation from fair share.")
    print("  Without aux loss, routers collapse to 1-2 experts (expert collapse).")
''',
    },

    "Expert Parallelism All-to-All Communication": {
        "description": "Simulate the All-to-All communication pattern used in expert parallelism — tokens are routed to expert GPUs and results returned, with load analysis.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
EXPERT PARALLELISM: ALL-TO-ALL COMMUNICATION SIMULATION
================================================================================

Simulates the Expert Parallelism (EP) communication pattern:
    1. Each GPU has a subset of tokens (from data parallelism)
    2. Each GPU sends tokens to the GPU hosting their selected expert
    3. Each GPU computes its expert on received tokens
    4. Results are returned via All-to-All

This is the core communication primitive distinguishing MoE from dense models.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import math


# ── All-to-All barrier ────────────────────────────────────────────────────────

class AllToAllBarrier:
    """
    Simulates All-to-All: each rank sends different data to each destination.
    Each rank receives different data from each source.
    """

    def __init__(self, n: int):
        self.n       = n
        self.lock    = threading.Lock()
        self.barrier = threading.Barrier(n)
        self.outbox  = {}   # (src, dst) → tensor
        self.inbox   = {}   # rank → list of received tensors

    def send(self, src: int, dst: int, data: torch.Tensor):
        """Send data from src to dst."""
        with self.lock:
            self.outbox[(src, dst)] = data.clone()

    def exchange(self, rank: int):
        """Wait for all sends, then collect all data destined for this rank."""
        self.barrier.wait()
        received = {}
        with self.lock:
            for (src, dst), data in self.outbox.items():
                if dst == rank:
                    received[src] = data
        self.barrier.wait()
        return received

    def clear(self):
        with self.lock:
            self.outbox.clear()


# ── Expert simulation ─────────────────────────────────────────────────────────

class ExpertNet(nn.Module):
    """One expert FFN."""
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def ep_rank_forward(rank: int, ep: int,
                     tokens: torch.Tensor,           # (n_local_tokens, d)
                     routing: torch.Tensor,           # (n_local_tokens,) — expert indices
                     weights: torch.Tensor,           # (n_local_tokens,) — routing weights
                     expert: ExpertNet,               # this rank's expert
                     barrier: AllToAllBarrier,
                     results: dict, step: int):
    """
    One EP rank's execution:
        1. Send local tokens to appropriate expert GPUs
        2. Receive tokens destined for this rank's expert
        3. Compute expert on received tokens
        4. Return results to origin GPUs
        5. Collect returned results, apply routing weights
    """
    n_local, d = tokens.shape

    # ── Phase 1: Route tokens to expert GPUs ──────────────────────────────────
    # Group tokens by destination expert GPU
    for dst_rank in range(ep):
        mask   = (routing == dst_rank)
        if mask.any():
            to_send = tokens[mask]   # tokens destined for this expert
            # Also send routing weights and original indices for reconstruction
            orig_idx = torch.where(mask)[0]
            barrier.send(rank, dst_rank,
                          torch.cat([to_send,
                                     weights[mask].unsqueeze(1),
                                     orig_idx.float().unsqueeze(1)], dim=1))

    # Wait and collect tokens destined for this rank's expert
    received = barrier.exchange(rank)

    # ── Phase 2: Compute expert on received tokens ────────────────────────────
    expert_outputs = {}   # src_rank → output tensor
    for src_rank, data in received.items():
        recv_tokens = data[:, :d]                          # (n, d)
        recv_weights = data[:, d]                          # (n,)
        recv_orig_idx = data[:, d+1].long()                # (n,)

        with torch.no_grad():
            out = expert(recv_tokens)    # (n, d)

        # Weight outputs by routing probabilities
        out_weighted = out * recv_weights.unsqueeze(1)
        expert_outputs[src_rank] = (out_weighted, recv_orig_idx)

    # ── Phase 3: Return results to origin GPUs ────────────────────────────────
    # This is another All-to-All (the "return" direction)
    # For simplicity, we store results indexed by (origin_rank, token_idx)
    with barrier.lock:
        if step not in barrier.outbox:
            barrier.outbox[f"return_{step}"] = {}
        for src_rank, (out_weighted, orig_idx) in expert_outputs.items():
            barrier.outbox[f"return_{step}"][(rank, src_rank)] = (out_weighted, orig_idx)

    barrier.barrier.wait()

    # ── Phase 4: Collect returned results ────────────────────────────────────
    output = torch.zeros(n_local, d)
    return_data = barrier.outbox.get(f"return_{step}", {})
    for (expert_rank, origin_rank), (out_w, orig_idx) in return_data.items():
        if origin_rank == rank:
            for i, idx in enumerate(orig_idx):
                output[idx] += out_w[i]

    barrier.barrier.wait()

    results[rank] = output


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_ep_demo():
    torch.manual_seed(42)

    EP         = 4       # expert parallelism degree (= number of experts for simplicity)
    D          = 32
    D_FF       = 64
    TOKENS_PER_GPU = 8   # tokens each GPU starts with (from data parallelism)

    print("=" * 62)
    print(f"  EXPERT PARALLELISM ALL-TO-ALL SIMULATION")
    print(f"  EP={EP} GPUs, each hosting 1 expert")
    print(f"  {TOKENS_PER_GPU} tokens per GPU, {EP*TOKENS_PER_GPU} total tokens")
    print("=" * 62)
    print()

    # Each GPU has its local tokens
    local_tokens = {r: torch.randn(TOKENS_PER_GPU, D) for r in range(EP)}

    # Simulate routing: random expert assignment per token
    torch.manual_seed(0)
    local_routing = {r: torch.randint(0, EP, (TOKENS_PER_GPU,)) for r in range(EP)}
    local_weights = {r: torch.ones(TOKENS_PER_GPU) / 1.0 for r in range(EP)}

    # Print routing before exchange
    print("  Token routing assignments (token → expert GPU):")
    for r in range(EP):
        assignments = {e: [] for e in range(EP)}
        for i, e in enumerate(local_routing[r].tolist()):
            assignments[e].append(i)
        print(f"  GPU{r} local tokens → "
              f"{', '.join(f'E{e}:{len(assignments[e])}toks' for e in range(EP))}")

    # Create experts (one per GPU)
    experts = {r: ExpertNet(D, D_FF) for r in range(EP)}

    barrier  = AllToAllBarrier(EP)
    results  = {}

    threads = [
        threading.Thread(
            target=ep_rank_forward,
            args=(r, EP, local_tokens[r], local_routing[r],
                  local_weights[r], experts[r], barrier, results, 0)
        )
        for r in range(EP)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    print()
    print("  Results after All-to-All exchange:")
    for r in range(EP):
        print(f"  GPU{r}: output shape = {results[r].shape}  "
              f"norm = {results[r].norm():.3f}")

    # Communication volume analysis
    print()
    print("=" * 62)
    print("  ALL-TO-ALL COMMUNICATION VOLUME ANALYSIS")
    print("=" * 62)
    print()

    configs = [
        ("Mixtral 8×7B",   8,   1,  4096, 4096),
        ("DBRX",           16,  4,  4096, 6144),
        ("DeepSeek-V2",    160, 6,  4096, 5120),
    ]

    print(f"  {'Model':<18} {'E':>4}  {'k':>3}  {'Active E':>9}  "
          f"{'A2A vol/step (GB)':>20}  {'vs DDP (unk)':>14}")
    print(f"  {'':─<18} {'':─>4}  {'':─>3}  {'':─>9}  {'':─>20}  {'':─>14}")

    for name, E, k, d, B_T in configs:
        # Tokens per GPU (rough estimate)
        tokens_per_gpu = 1024  # B * T / n_gpus

        # Each token sends its hidden state to k expert GPUs and receives k results
        # Volume: 2 (fwd+bwd) × k × tokens_per_gpu × d × dtype_bytes
        a2a_vol_gb = 2 * k * tokens_per_gpu * d * 2 / 1e9

        # Active experts fraction
        active_frac = k / E

        print(f"  {name:<18} {E:>4}  {k:>3}  {active_frac:>8.1%}  "
              f"{a2a_vol_gb:>20.4f}  "
              f"{'(model-dependent)':>14}")

    print()
    print("  All-to-All volume scales with tokens × d × k — NOT with total params.")
    print("  Dense DDP all-reduce scales with total params × 2.")
    print("  For large MoE models, All-to-All is typically much cheaper than DDP.")


if __name__ == "__main__":
    run_ep_demo()
''',
    },

    "MoE Training Stability and Load Balancing": {
        "description": "Demonstrate expert collapse without auxiliary loss, show how the load-balancing loss prevents it, and analyse router entropy as a training health signal.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MOE TRAINING STABILITY: LOAD BALANCING AND EXPERT COLLAPSE
================================================================================

Demonstrates:
    1. Expert collapse without auxiliary loss (router learns to ignore most experts)
    2. Load balancing with auxiliary loss (equal expert utilisation)
    3. Router entropy as a training health metric
    4. The effect of auxiliary loss weight (too small → collapse, too large → noise)

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Minimal MoE for ablation ──────────────────────────────────────────────────

class MiniMoE(nn.Module):
    def __init__(self, d: int, n_experts: int, k: int,
                 aux_loss_weight: float = 0.01):
        super().__init__()
        self.n_experts       = n_experts
        self.k               = k
        self.aux_loss_weight = aux_loss_weight

        self.gate    = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d*4, bias=False), nn.GELU(),
                          nn.Linear(d*4, d, bias=False))
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple:
        N, d = x.shape

        logits = self.gate(x)               # (N, E)
        probs  = F.softmax(logits, dim=-1)  # (N, E)

        # Top-k routing
        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # Expert computation
        output = torch.zeros_like(x)
        for e in range(self.n_experts):
            for ki in range(self.k):
                mask = (topk_idx[:, ki] == e)
                if mask.any():
                    w   = topk_weights[mask, ki].unsqueeze(-1)
                    out = self.experts[e](x[mask])
                    output[mask] += w * out

        # Auxiliary load-balancing loss
        f_e   = torch.zeros(self.n_experts, device=x.device)
        for ki in range(self.k):
            counts = torch.bincount(topk_idx[:, ki], minlength=self.n_experts).float()
            f_e += counts / N
        p_e = probs.mean(dim=0)

        aux_loss = self.n_experts * (f_e * p_e).sum()

        return output, aux_loss * self.aux_loss_weight


def router_entropy(model: MiniMoE, x: torch.Tensor) -> float:
    """Compute mean entropy of routing distribution over a batch."""
    with torch.no_grad():
        probs = F.softmax(model.gate(x), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        return entropy.mean().item()


def expert_utilisation(model: MiniMoE, x: torch.Tensor) -> list:
    """Fraction of tokens routed to each expert."""
    with torch.no_grad():
        probs = F.softmax(model.gate(x), dim=-1)
        _, topk = torch.topk(probs, model.k, dim=-1)
        counts  = torch.zeros(model.n_experts)
        for ki in range(model.k):
            for e in range(model.n_experts):
                counts[e] += (topk[:, ki] == e).float().sum()
    return (counts / (x.shape[0] * model.k)).tolist()


def train_epoch(model, x, y, n_steps: int = 50, lr: float = 1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        out, aux = model(x)
        loss = F.mse_loss(out, y) + aux
        loss.backward()
        opt.step()
        losses.append(F.mse_loss(out.detach(), y).item())
    return losses


if __name__ == "__main__":
    torch.manual_seed(0)

    D, E, K = 64, 8, 2
    N       = 256  # tokens in batch
    x       = torch.randn(N, D)
    y       = torch.randn(N, D)

    print("=" * 62)
    print("  EXPERT COLLAPSE DEMONSTRATION")
    print(f"  E={E} experts, k={K}, d={D}, N={N} tokens")
    print("=" * 62)
    print()

    # Compare three auxiliary loss weights
    configs = [
        ("No aux loss (α=0)",     0.00),
        ("Standard (α=0.01)",     0.01),
        ("Too large (α=1.0)",     1.00),
    ]

    results = {}
    for name, alpha in configs:
        model  = MiniMoE(D, E, K, aux_loss_weight=alpha)
        losses = train_epoch(model, x, y, n_steps=200, lr=3e-4)
        util   = expert_utilisation(model, x)
        ent    = router_entropy(model, x)
        results[name] = (losses, util, ent)

    print(f"  {'Config':<26} {'Final loss':>12}  {'Router entropy':>16}  "
          f"{'Max expert load':>16}  {'Min expert load':>16}")
    print(f"  {'':─<26} {'':─>12}  {'':─>16}  {'':─>16}  {'':─>16}")

    max_entropy = math.log(E)  # uniform distribution entropy
    for name, (losses, util, ent) in results.items():
        print(f"  {name:<26} {losses[-1]:>12.4f}  "
              f"{ent:>15.3f} ({ent/max_entropy:.0%})  "
              f"{max(util):>15.1%}  "
              f"{min(util):>15.1%}")

    print()
    print("  Expert utilisation breakdown per expert:")
    print()
    print(f"  {'Expert':>8}", end="")
    for name, _ in configs:
        print(f"  {name[:14]:>16}", end="")
    print()
    print(f"  {'':─>8}" + "".join(f"  {'':─>16}" for _ in configs))

    for e in range(E):
        print(f"  {e:>8}", end="")
        for name, (_, util, _) in results.items():
            bar = "█" * int(util[e] * 20)
            print(f"  {util[e]:>5.1%} {bar:<10}", end="")
        print()

    print()
    print("  Key observations:")
    print("  • No aux loss: router collapses to 1-2 experts (>50% load on one)")
    print("  • Standard α=0.01: roughly equal load, high entropy")
    print("  • Too large α=1.0: forced equal load but poor reconstruction loss")
    print()
    print("  Router entropy:")
    print(f"  • Uniform distribution max entropy: {max_entropy:.3f}")
    print("  • Collapsed router:    entropy near 0 (always same experts)")
    print("  • Healthy router:      entropy near max (diverse routing)")
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
    #     from llm_training.visuals.expert_parallelism_moe import (
    #         MOE_VISUAL_HTML,
    #         MOE_VISUAL_HEIGHT,
    #     )
    #     visual_html   = MOE_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = MOE_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[22_expert_parallelism_moe.py] Could not load visual: {e}",
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