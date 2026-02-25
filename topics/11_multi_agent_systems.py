"""Module: 11 · Multi-Agent Systems"""

DISPLAY_NAME = "11 · Multi-Agent Systems"
ICON         = "🕸️"
SUBTITLE     = "Topologies, message passing, state graphs, consensus, debate, and production patterns"

THEORY = """
## 11 · Multi-Agent Systems

Module 10 established the single-agent loop: one LLM, one tool set, one context window.
This module asks what happens when you connect *multiple* such agents. The answer is not
merely "more of the same" — new phenomena emerge: specialisation, parallel throughput,
mutual error-checking, and collective intelligence that no single agent can achieve alone.
New failure modes also emerge: cascading errors, deadlock, message storms, and context
contamination. This module maps the full landscape.

---

### 1 · Why Multi-Agent Systems?

**1.1 The limits of a single agent revisited.**

A capable single agent still has four hard constraints:

| Constraint | Impact | Multi-agent remedy |
|---|---|---|
| Context window | Long tasks overflow memory | Each agent has its own fresh context |
| Serial execution | One step at a time | Parallel workers |
| Uniform capability | One model, one style | Specialised agents per subtask |
| Self-review blindspot | Same model that erred reviews its own error | Independent critics |

**1.2 Empirical evidence for collaboration.** Several studies (Wang et al., 2023;
Du et al., 2023; Liang et al., 2023) show that having multiple LLMs debate, critique,
or iteratively revise each other's outputs outperforms a single model on:
- Mathematical reasoning (GSM8K, MATH benchmarks)
- Factual accuracy (TriviaQA)
- Code correctness (HumanEval)
- Creative writing quality

The gain comes from *diversity*: different agents make different errors; cross-checking
catches errors that self-review misses.

**1.3 The coordination cost.** Every agent interaction incurs overhead: message
serialisation, LLM call latency for the orchestrator, and the risk of misaligned
context. Multi-agent systems are only worth it when the collaboration gain exceeds
the coordination cost. The break-even point depends on task complexity and independence
of subtasks.

---

### 2 · Agent Topologies

The *topology* defines which agents can communicate with which, and in what direction.
Topology choice is arguably the most important architectural decision in a multi-agent
system.

**2.1 Star (hub-and-spoke).** A central orchestrator communicates with all workers;
workers never communicate with each other.

```
        Orchestrator
       /      |      \\
   Worker  Worker  Worker
```

*Pros:* Simple coordination; single point of control; easy to debug; orchestrator
maintains global state. *Cons:* orchestrator is a bottleneck; its context fills with
all worker results; single point of failure.

*Best for:* Tasks where subtasks are independent and the orchestrator must synthesise
all results (e.g., parallel research then report writing).

**2.2 Chain (pipeline).** Agents are arranged sequentially; each agent transforms the
output of the previous agent.

```
Agent A → Agent B → Agent C → Result
```

*Pros:* Each agent specialises in one transformation; intermediate results are explicit;
easy to test stages independently. *Cons:* errors propagate forward; total latency is the
sum of all stages; a failed middle agent breaks the whole pipeline.

*Best for:* Multi-stage document processing (extract → clean → analyse → format),
code generation pipelines (write → review → test → document).

**2.3 Ring.** Each agent communicates with the next; the last agent communicates back
to the first. Used in iterative refinement loops.

```
A → B → C → D
↑           ↓
└───────────┘
```

*Pros:* enables iterative improvement across the full pipeline. *Cons:* convergence not
guaranteed; hard to know when to stop; cycles add latency.

**2.4 Mesh (fully connected).** Any agent can communicate with any other agent.

```
A ─── B
│  ╲╱  │
│  ╱╲  │
C ─── D
```

*Pros:* maximum flexibility; any agent can delegate to any other. *Cons:* communication
complexity is O(N²); risk of message storms; extremely hard to debug.

*Best for:* Small groups of peer agents (2-4) in debate or co-creation patterns.

**2.5 Hierarchical (tree).** Agents form a tree: top-level orchestrator breaks goal into
mid-level tasks dispatched to sub-orchestrators, each of which dispatches to leaf workers.

```
          CEO-Agent
         /         \\
   Manager-A    Manager-B
   /      \\         \\
Worker  Worker    Worker
```

*Pros:* scales to complex tasks; each level handles appropriate abstraction;
context stays localised. *Cons:* deep hierarchies multiply latency; errors at
high levels propagate to all subtrees; requires careful role design.

*Best for:* Large software projects, complex research tasks, business workflows.

**2.6 Market-based (dynamic allocation).** A broker publishes tasks; agents bid on
tasks based on their capabilities and current load; the broker assigns tasks to
the lowest-cost or highest-capability bidder. No static topology — connections
form dynamically.

*Pros:* efficient resource utilisation; naturally load-balances; agents can specialise
without pre-wiring. *Cons:* requires a pricing/evaluation mechanism; bid manipulation
is possible; overhead of auction rounds.

---

### 3 · Communication and Message Passing

**3.1 Message structure.** Every inter-agent message needs a minimal envelope:

```json
{
  "id":          "msg-uuid",
  "from":        "orchestrator",
  "to":          "researcher-1",
  "type":        "task",
  "content":     { "task": "...", "context": "..." },
  "reply_to":    null,
  "priority":    1,
  "timestamp":   "2024-01-01T00:00:00Z",
  "ttl_seconds": 60
}
```

The `type` field is critical: agents need to distinguish *tasks* (do something) from
*results* (here is output) from *errors* (something failed) from *queries* (give me
information). Without explicit typing, agents misinterpret messages and produce
cascading errors.

**3.2 Synchronous vs asynchronous communication.**

*Synchronous (blocking):* Agent A sends a message and waits for Agent B's response
before continuing. Simple to reason about; wastes time waiting; can deadlock.

*Asynchronous (non-blocking):* Agent A sends a message and continues. Responses arrive
as events. Higher throughput; parallel execution; requires careful state management
(which request does this response correspond to?).

Most production multi-agent systems use async internally but expose sync interfaces to
the user (the user waits for the final answer, not intermediate steps).

**3.3 Message queues and brokers.** In large systems, agents don't communicate
directly — they publish to and subscribe from a message queue (Redis, Kafka, RabbitMQ).
This decouples producers from consumers, enables persistence (messages survive agent
restarts), and provides backpressure (slow agents don't overwhelm fast ones).

**3.4 Context sharing.** The most expensive part of inter-agent communication is
*context*. A worker needs enough context to perform its task, but copying the full
orchestrator context into every worker message is wasteful and often exceeds context
limits. Best practices:
- Pass only the *relevant* subset of context (task-specific).
- Pass a *summary* of background context, not raw history.
- Use shared external state (a blackboard) that agents can query rather than receive.
- Include the original goal in every message — workers lose track of the big picture.

**3.5 Broadcast, multicast, unicast.** Agents may need to:
- *Unicast:* send to one specific agent (most common).
- *Multicast:* send to a group (all research workers get the same brief).
- *Broadcast:* send to all agents (coordinator announces task completion).

Broadcast should be rare — it grows message volume as O(N) and can trigger redundant
work if multiple agents respond.

---

### 4 · Shared State and the Blackboard Architecture

**4.1 The blackboard metaphor.** A blackboard is a shared data store that all agents
can read from and write to, like a whiteboard in a conference room. Agents don't need
to know about each other — they only interact with the blackboard.

```
Agent A ──writes──▶ Blackboard ◀──reads── Agent B
Agent C ──reads──▶ Blackboard ◀──writes── Agent D
```

**4.2 Blackboard structure.** A well-designed blackboard has:
- *Task queue:* pending work items any agent can claim.
- *Results store:* completed outputs keyed by task ID.
- *Shared context:* facts, retrieved documents, intermediate summaries available to all.
- *Agent registry:* which agents exist, their capabilities, their current status.
- *Coordination signals:* semaphores, locks, completion flags.

**4.3 Concurrency and conflicts.** When multiple agents write to the same blackboard
key simultaneously, conflicts arise. Strategies:
- *Last-write-wins:* simple, may lose data.
- *Optimistic locking:* read with version number, write only if version unchanged.
- *Event sourcing:* never overwrite; append new events; reconstruct state from event log.
- *CRDTs (Conflict-free Replicated Data Types):* data structures that merge automatically
  without conflicts (e.g., append-only sets, counters).

**4.4 State graphs (LangGraph pattern).** Rather than a free-form blackboard, LangGraph
(Harrison Chase, 2024) structures multi-agent state as a typed graph:
- *Nodes* are agent functions (each takes state, returns updated state).
- *Edges* define the flow between agents (who runs after whom).
- *Conditional edges* implement branching (run the critic only if quality score < 0.8).
- *Reducers* specify how to merge parallel writes (append vs overwrite vs custom merge).

The typed state forces explicit schema design upfront, which dramatically improves
debuggability. Every state transition is logged as a graph traversal.

```python
class AgentState(TypedDict):
    goal:      str
    drafts:    list[str]
    scores:    list[float]
    final:     str
    iteration: int
```

Nodes can only add/modify keys; the reducer decides how list keys accumulate
(default: replace; annotated with `operator.add`: append).

---

### 5 · Orchestrator Patterns in Depth

**5.1 Static orchestration.** The orchestrator has a fixed execution plan: always runs
agents in the same order with the same assignments. Predictable, easy to test, brittle
when subtask results affect the plan.

**5.2 Dynamic orchestration.** The orchestrator is itself an LLM that decides which
worker to call next based on the current state. More flexible; consumes more LLM calls;
harder to predict and test.

**5.3 Supervisor pattern.** A supervisor monitors all worker agents and intervenes when
quality drops below a threshold, a worker gets stuck, or a deadline approaches. The
supervisor can reassign tasks, restart workers, or escalate to a human.

**5.4 Map-reduce for agents.**

```
Input → Map(split into chunks) → [Worker₁, Worker₂, ... Workerₙ] → Reduce(aggregate)
```

Each worker processes an independent chunk; the reducer combines results. Works for
document analysis (process each section in parallel), data processing (analyse each
data slice), and research (search multiple angles simultaneously).

Reduce strategies: concatenation (join text), aggregation (sum/avg numbers),
election (pick best output by score), merging (combine unique facts from all outputs).

**5.5 Reflection pattern (self-critique loop).**

```
Generator → Output → Critic → Score
              ↑                  ↓
              └──── Revise ←─────┘ (if score < threshold)
```

The generator and critic can be the same model (prompted differently) or different models.
Multiple rounds of refinement typically plateau after 3-5 iterations — diminishing returns
from more iterations are well-documented (Saunders et al., 2022).

---

### 6 · Consensus and Voting

**6.1 Why consensus?** A single LLM produces a single answer — which may be wrong.
Having N agents independently produce answers and combining them reduces variance.
This is the ensemble principle from machine learning applied to language model outputs.

**6.2 Majority voting.** Run N agents independently. Take the most common answer.

For discrete answers (e.g., classification, multiple choice), majority voting works
well. Accuracy improves from p₁ to P_majority as:

```
P_majority = Σ_{k=⌈N/2⌉}^{N} C(N,k) × pᵏ × (1-p)^(N-k)
```

For p = 0.7 (single agent accuracy) and N = 5 agents:
P_majority = 1 - P(0 or 1 or 2 correct) ≈ 0.84 — a significant improvement.
For N = 9: P_majority ≈ 0.90.

**6.3 Weighted voting.** Not all agents are equal. Assign each agent a weight based
on demonstrated accuracy on past tasks. The final answer is the weighted majority.
Weights can be updated dynamically as agents are evaluated.

**6.4 Borda count.** For ranking problems, each agent submits a ranked list of
candidates. Borda count assigns points (N for rank 1, N-1 for rank 2, ..., 1 for last).
The candidate with the highest total score wins. More robust than plurality voting for
rank aggregation tasks.

**6.5 LLM-as-judge.** After N agents produce independent answers, a "judge" LLM
evaluates all answers and selects or synthesises the best one. The judge has more
information than a simple voting rule — it can assess reasoning quality, factual
consistency, and completeness.

Key concern: the judge LLM has its own biases — it tends to prefer verbose answers and
answers that match its own reasoning style, regardless of correctness (Zheng et al., 2023).

---

### 7 · Debate and Adversarial Collaboration

**7.1 The debate pattern.** Two or more agents argue opposing positions. A judge evaluates
the arguments and selects a winner. The debate forces agents to construct explicit
arguments rather than simply asserting answers — this surfaces reasoning errors.

```
Proposer: "The answer is X because ..."
Challenger: "The answer is NOT X because ... The proposer's reasoning fails because ..."
Proposer: "Rebuts: ..."
Judge: evaluates both arguments → decision
```

Du et al. (2023) showed that debate improves factual accuracy on tasks where the model
has the knowledge but fails to correctly apply it — the challenge forces the proposer to
sharpen its reasoning.

**7.2 Socratic dialogue.** One agent plays the questioner (Socrates), persistently asking
"why?" and "how do you know?" The responder is forced to make implicit assumptions explicit.
Effective for requirement elicitation, assumption surfacing, and logical gap detection.

**7.3 Red-team / blue-team.** Blue-team builds a system or argument. Red-team tries to
break it. Red-team findings are fed back to blue-team for hardening. Iterates until the
red-team cannot find new vulnerabilities. Used in security analysis, policy writing,
and argument stress-testing.

**7.4 Constitutional AI pattern.** The generator produces output. A critique agent
evaluates it against a list of principles (the "constitution"). The generator revises
based on the critique. The critique agent scores the revision. This is literally how
Anthropic trains Claude (Bai et al., 2022).

---

### 8 · Specialisation and Role Design

**8.1 The case for specialisation.** A generalist system prompt tries to be everything;
specialist prompts excel at one thing. Specialisation works because:
- The system prompt budget (tokens) is finite; deep expertise crowds out breadth.
- Tool sets can be tailored — a researcher gets search tools; a coder gets a REPL.
- Evaluation is easier — judge a coder's output on test pass rates, not vague quality.
- Fine-tuned specialist models (code model, math model) outperform generalists on their domains.

**8.2 Role taxonomy.** Common agent roles in production multi-agent systems:

| Role | Responsibility | Key tools |
|---|---|---|
| Orchestrator | Decompose goal, assign tasks, synthesise | Agent spawning, state management |
| Researcher | Gather information, retrieve facts | Web search, document retrieval |
| Analyst | Interpret data, identify patterns | Calculator, data query, code runner |
| Coder | Write and test code | Code interpreter, file system |
| Critic/Reviewer | Evaluate quality, find errors | Rubric evaluation, test runner |
| Writer | Produce final narrative | Text templates, style checkers |
| Memory Manager | Store and retrieve relevant context | Vector DB, summariser |
| Planner | Decompose complex goals into task DAGs | None (reasoning only) |

**8.3 Handoff protocols.** When one agent hands off to another, it must pass:
1. *The original goal* — never let it get lost.
2. *What has been done* — avoid redundant work.
3. *What is known* — relevant facts, retrieved documents.
4. *What is needed* — the specific subtask for the recipient.
5. *Constraints* — format, length, deadline, quality bar.

Poorly designed handoffs are the #1 cause of agent system failures. The receiving
agent proceeds without knowing what was already tried, duplicates work, or misunderstands
its scope.

---

### 9 · Failure Modes, Deadlock, and Livelock

**9.1 Cascading failures.** Agent A produces a slightly wrong output. Agent B uses it
as context and amplifies the error. Agent C builds on B's error. By the time the
orchestrator sees C's output, the error is deeply embedded and the original cause is
invisible. Prevention: validate at each handoff, not just at the final output.

**9.2 Deadlock.** Agent A is waiting for Agent B's result. Agent B is waiting for
Agent C's result. Agent C is waiting for Agent A's result. The system halts. Prevention:
timeout every agent call; detect waiting cycles; escalate to orchestrator on timeout.

**9.3 Livelock.** Agent A sends work back to Agent B for revision. Agent B revises and
sends back to A. Neither terminates because neither's quality threshold is met. They
spin indefinitely. Prevention: maximum revision count; monotonic quality requirement
(each round must improve by ε or terminate).

**9.4 Context contamination.** Agent B receives Agent A's output as context and begins
to mimic A's errors, style, or false beliefs. The contamination spreads to all downstream
agents. Prevention: include original goal prominently; use XML tags to clearly label
external content vs agent reasoning; fresh-context workers for critical decisions.

**9.5 Sycophancy in multi-agent systems.** When a junior agent presents its result to
an orchestrator, and the orchestrator expresses any preference, the junior often revises
its answer to match — even if the original was more correct. This is LLM sycophancy
operating at the inter-agent level. Prevention: have critics submit before seeing the
orchestrator's evaluation; blind review patterns.

**9.6 Cost explosion.** Each orchestrator turn spawns N workers. If workers are themselves
orchestrators that spawn N sub-workers, cost grows as O(Nᵈᵉᵖᵗʰ). Unbounded recursion
is catastrophic. Prevention: hard budget limits; depth limits; single-agent fallback
for simple subtasks.

---

### 10 · Production Frameworks Deep-Dive

**10.1 LangGraph.** Graph-based stateful orchestration (Langchain, 2024). Key abstractions:
- *StateGraph:* nodes are Python functions; edges are directed connections; state is a
  TypedDict shared across all nodes.
- *Conditional edges:* routing function decides which node to visit next.
- *Checkpointing:* state is persisted at each step; enables resumability and time-travel debugging.
- *Human-in-the-loop:* execution can pause at any node for human review (interrupt_before/after).
- *Streaming:* state updates stream to the client as they happen.

LangGraph's killer feature is *built-in cycles* — most DAG-based orchestrators can't
express iterative refinement without awkward hacks. LangGraph's graphs can have cycles
naturally (agent loops back to a previous node).

**10.2 AutoGen (Microsoft Research).** Conversation-based multi-agent framework (Wu et al., 2023).
Core concepts:
- *ConversableAgent:* any LLM or human that can participate in a conversation.
- *GroupChat:* multiple agents in a conversation; a GroupChatManager selects who speaks next.
- *Code execution:* built-in Docker-sandboxed Python execution; agents write code that
  actually runs and the output comes back as a message.
- *Human proxy:* a human can participate in the conversation as an agent, enabling HITL.

AutoGen excels at code generation + testing workflows because of its first-class code
execution support.

**10.3 CrewAI.** Role-based multi-agent framework. Agents have *roles* (job title),
*goals* (what they try to achieve), and *backstories* (personality). Tasks have
expected_output specifications. Crews can be sequential (each task depends on previous)
or hierarchical (manager delegates to workers).

CrewAI's differentiator is the *human-readable role specification* — you describe agents
as you would describe human team members. This makes the system accessible to non-engineers.

**10.4 Custom orchestration.** For production systems, frameworks often introduce more
problems than they solve: hidden abstractions make debugging hard; version upgrades break
pipelines; framework overhead adds latency. Many teams build a 300-500 line custom
orchestrator with:
- A typed message dataclass.
- An agent registry with capabilities metadata.
- A simple async dispatch loop.
- Structured logging of every call.
- Hard budget checks before every LLM call.

The tradeoff: frameworks give you a 1-day prototype; custom code gives you a maintainable
production system.

---

### Key Takeaways

- Topology is architecture. Star, chain, pipeline, hierarchical, and market-based
  topologies each fit different task structures. Choose based on task dependency graph,
  not personal preference.
- Message structure matters. Every inter-agent message needs type, sender/receiver,
  content, and the original goal. Poorly structured messages cause more failures than
  poor reasoning.
- Shared state (blackboard) decouples agents; direct messaging couples them. Use shared
  state for facts that multiple agents need; use direct messages for task handoffs.
- LangGraph models multi-agent state as a typed graph with explicit reducers, conditional
  edges, and built-in cycles — the right abstraction for iterative agent workflows.
- Consensus (voting, debate, LLM-as-judge) reduces individual agent error at the cost of
  N× more LLM calls. Worth it when single-agent reliability is below ~70%.
- Specialisation compounds: a specialist agent with specialist tools and specialist
  evaluation outperforms a generalist by a large margin on domain tasks.
- The characteristic failure modes — cascade, deadlock, livelock, context contamination,
  sycophancy, cost explosion — each require specific mitigations, not generic error handling.
- Production systems need hard budget limits, depth limits, per-step validation, and
  structured observability. Framework choice matters less than these fundamentals.
"""

OPERATIONS = {
    "1 · Agent Topology Simulator": {
        "description": (
            "Simulate and compare all five major topologies — star, chain, ring, "
            "hierarchical, and mesh — on message routing, latency, and failure propagation."
        ),
        "language": "python",
        "code": """\
import math, random
random.seed(1)

# ── Agent and topology models ─────────────────────────────────────────────────
class Agent:
    def __init__(self, name, role, latency_ms=100, fail_rate=0.05):
        self.name       = name
        self.role       = role
        self.latency_ms = latency_ms
        self.fail_rate  = fail_rate
        self.messages_received = 0
        self.messages_sent     = 0
        self.errors            = 0

    def process(self, message, rng):
        self.messages_received += 1
        if rng.random() < self.fail_rate:
            self.errors += 1
            return None  # failure
        self.messages_sent += 1
        return f"{self.name} processed: {message[:40]}"

class Topology:
    def __init__(self, name, agents):
        self.name    = name
        self.agents  = agents
        self.edges   = []   # (from_name, to_name)
        self.log     = []

    def add_edge(self, a, b):
        self.edges.append((a, b))

    def send(self, from_name, to_name, message, rng):
        sender    = next(a for a in self.agents if a.name == from_name)
        receiver  = next(a for a in self.agents if a.name == to_name)
        result    = receiver.process(message, rng)
        self.log.append({"from": from_name, "to": to_name,
                         "success": result is not None, "latency": receiver.latency_ms})
        return result

    def total_latency(self):
        return sum(e["latency"] for e in self.log if e["success"])

    def failure_count(self):
        return sum(1 for e in self.log if not e["success"])

    def message_count(self):
        return len(self.log)

# ── Build five topologies with 5 workers each ─────────────────────────────────
def make_agents(n, prefix, base_latency=100, fail_rate=0.05):
    return [Agent(f"{prefix}-{i}", "worker", base_latency + i*10, fail_rate)
            for i in range(n)]

rng = random.Random(42)
TASK = "Analyse the quarterly sales data and produce a summary"

# 1. STAR
star_agents = [Agent("hub", "orchestrator", 120, 0.02)] + make_agents(4, "W", 90)
star = Topology("Star (hub-and-spoke)", star_agents)
for w in star_agents[1:]:
    star.add_edge("hub", w.name)
    star.add_edge(w.name, "hub")
# Simulate: hub sends to each worker, workers return to hub
for w in star_agents[1:]:
    r = star.send("hub", w.name, TASK, rng)
    if r:
        star.send(w.name, "hub", r, rng)

# 2. CHAIN
chain_agents = [Agent(f"C{i}", "worker", 80 + i*15, 0.06) for i in range(5)]
chain = Topology("Chain (pipeline)", chain_agents)
for i in range(len(chain_agents) - 1):
    chain.add_edge(chain_agents[i].name, chain_agents[i+1].name)
# Simulate: message flows sequentially
msg = TASK
for i in range(len(chain_agents) - 1):
    result = chain.send(chain_agents[i].name, chain_agents[i+1].name, msg, rng)
    if result is None:
        break
    msg = result

# 3. RING
ring_agents = [Agent(f"R{i}", "worker", 95, 0.05) for i in range(5)]
ring = Topology("Ring (iterative)", ring_agents)
N = len(ring_agents)
for i in range(N):
    ring.add_edge(ring_agents[i].name, ring_agents[(i+1) % N].name)
# Simulate: 2 full rounds
msg = TASK
for _ in range(2):
    for i in range(N):
        nxt = ring_agents[(i+1) % N].name
        result = ring.send(ring_agents[i].name, nxt, msg, rng)
        if result:
            msg = result

# 4. HIERARCHICAL (3 levels: 1 CEO, 2 managers, 4 workers)
h_ceo    = Agent("CEO", "orchestrator", 150, 0.02)
h_mgrs   = [Agent(f"Mgr{i}", "manager", 110, 0.04) for i in range(2)]
h_workers= [Agent(f"Emp{i}", "worker",  80, 0.06) for i in range(4)]
hier     = Topology("Hierarchical (3 levels)", [h_ceo] + h_mgrs + h_workers)
# CEO → each manager
for mgr in h_mgrs:
    r = hier.send("CEO", mgr.name, TASK, rng)
    if r:
        # manager → 2 workers each
        for w in h_workers[:2] if mgr.name == "Mgr0" else h_workers[2:]:
            wr = hier.send(mgr.name, w.name, r, rng)
            if wr:
                hier.send(w.name, mgr.name, wr, rng)
        hier.send(mgr.name, "CEO", r, rng)

# 5. MESH (5 agents, all-to-all, 3 rounds)
mesh_agents = [Agent(f"M{i}", "peer", 100, 0.05) for i in range(4)]
mesh = Topology("Mesh (fully connected)", mesh_agents)
for a in mesh_agents:
    for b in mesh_agents:
        if a.name != b.name:
            mesh.add_edge(a.name, b.name)
for _ in range(2):  # 2 rounds
    for a in mesh_agents:
        for b in mesh_agents:
            if a.name != b.name:
                mesh.send(a.name, b.name, TASK, rng)

# ── Results table ─────────────────────────────────────────────────────────────
topologies = [star, chain, ring, hier, mesh]
SEP = "=" * 72
print(SEP)
print("Topology Comparison: 5-Agent Task Simulation")
print(f"Task: '{TASK[:50]}...'")
print(SEP)
print()
print(f"  {'Topology':<30} {'Messages':>10} {'Failures':>10} {'Total Latency':>14}")
print("  " + "-" * 68)
for t in topologies:
    n_msg  = t.message_count()
    n_fail = t.failure_count()
    latency= t.total_latency()
    print(f"  {t.name:<30} {n_msg:>10} {n_fail:>10} {latency:>12} ms")

print()
print(SEP)
print("Failure Propagation Analysis:")
print(SEP)
for t in topologies:
    fails = t.failure_count()
    msgs  = t.message_count()
    pct   = 100 * fails / max(msgs, 1)
    bar   = "X" * fails + "." * (msgs - fails)
    print(f"  {t.name:<30}  [{bar[:30]}]  {pct:.0f}% fail")

print()
print(SEP)
print("Topology Selection Guide:")
print(SEP)
guide = [
    ("Star",          "Independent parallel subtasks, single synthesis step"),
    ("Chain",         "Sequential transformations: extract → clean → analyse → write"),
    ("Ring",          "Iterative refinement where each pass improves the whole"),
    ("Hierarchical",  "Complex goals requiring multi-level decomposition"),
    ("Mesh",          "Small peer groups (2-4) doing debate or co-creation"),
]
for name, use_case in guide:
    print(f"  {name:<18} {use_case}")
""",
    },

    "2 · Message Passing and Routing System": {
        "description": (
            "Build a typed message bus with routing, priority queues, TTL expiry, "
            "delivery tracking, and dead-letter handling for dropped messages."
        ),
        "language": "python",
        "code": """\
import uuid, time, math
from collections import defaultdict, deque

# ── Message dataclass ─────────────────────────────────────────────────────────
class Message:
    TYPES = ("task", "result", "error", "query", "signal", "broadcast")

    def __init__(self, from_id, to_id, msg_type, content,
                 priority=1, ttl_s=60, reply_to=None):
        self.id        = str(uuid.uuid4())[:8]
        self.from_id   = from_id
        self.to_id     = to_id
        self.msg_type  = msg_type
        self.content   = content
        self.priority  = priority   # 1=low, 2=normal, 3=high, 4=critical
        self.ttl_s     = ttl_s
        self.reply_to  = reply_to
        self.created   = time.monotonic()
        self.delivered = False
        self.retries   = 0

    def is_expired(self):
        return (time.monotonic() - self.created) > self.ttl_s

    def summary(self):
        pnames = {1: "LOW", 2: "NORMAL", 3: "HIGH", 4: "CRITICAL"}
        return (f"[{self.id}] {self.from_id}->{self.to_id} "
                f"type={self.msg_type} P={pnames.get(self.priority,'?')}")

# ── Message bus ───────────────────────────────────────────────────────────────
class MessageBus:
    def __init__(self, max_retries=3):
        self.queues      = defaultdict(list)  # agent_id -> [Message]
        self.dead_letter = []
        self.delivered   = []
        self.max_retries = max_retries
        self.stats       = defaultdict(int)

    def publish(self, msg):
        self.queues[msg.to_id].append(msg)
        self.queues[msg.to_id].sort(key=lambda m: -m.priority)  # higher priority first
        self.stats["published"] += 1
        return msg.id

    def broadcast(self, from_id, recipients, msg_type, content, priority=2):
        ids = []
        for recipient in recipients:
            msg = Message(from_id, recipient, msg_type, content, priority)
            ids.append(self.publish(msg))
        self.stats["broadcasts"] += 1
        return ids

    def consume(self, agent_id, max_n=5):
        queue   = self.queues[agent_id]
        results = []
        while queue and len(results) < max_n:
            msg = queue.pop(0)
            if msg.is_expired():
                msg.retries += 1
                if msg.retries >= self.max_retries:
                    self.dead_letter.append(msg)
                    self.stats["dead_lettered"] += 1
                else:
                    # re-queue (reset TTL for demo)
                    msg.created = time.monotonic()
                    queue.append(msg)
                    queue.sort(key=lambda m: -m.priority)
                continue
            msg.delivered = True
            self.delivered.append(msg)
            self.stats["delivered"] += 1
            results.append(msg)
        return results

    def queue_depth(self, agent_id):
        return len(self.queues[agent_id])

    def all_depths(self):
        return {aid: len(q) for aid, q in self.queues.items() if q}

    def report(self):
        return dict(self.stats)

# ── Simulate a 6-agent research pipeline ─────────────────────────────────────
bus = MessageBus(max_retries=2)
AGENTS = ["orchestrator", "researcher-1", "researcher-2",
          "analyst", "critic", "writer"]

# Orchestrator broadcasts the task brief to researchers
brief = "Analyse the impact of transformer models on NLP benchmarks 2020-2024"
bus.broadcast("orchestrator", ["researcher-1", "researcher-2"], "task",
              {"task": brief, "goal": brief}, priority=3)

# Orchestrator sends analysis task to analyst
bus.publish(Message("orchestrator", "analyst", "task",
                    {"task": "Compute aggregate improvement statistics"}, priority=2))

# Researchers send results back
for i in (1, 2):
    bus.publish(Message(f"researcher-{i}", "analyst", "result",
                        {"finding": f"Researcher {i} finding: BERT/GPT class models "
                                    f"improved GLUE by 20-30 points from 2020-2022"},
                        priority=2))

# Analyst sends to critic
bus.publish(Message("analyst", "critic", "task",
                    {"draft": "Transformers improved NLP by avg 25 GLUE points"},
                    priority=2))

# Critic sends back with feedback
bus.publish(Message("critic", "analyst", "result",
                    {"score": 0.72, "feedback": "Missing citation for 25pt claim"},
                    priority=3))

# Add an expired message (low TTL)
old_msg = Message("researcher-1", "writer", "result",
                  {"stale": "outdated finding"}, ttl_s=0)
bus.publish(old_msg)

# Writer gets final signal
bus.publish(Message("orchestrator", "writer", "signal",
                    {"action": "begin_draft", "context": brief}, priority=4))

# ── Report state ───────────────────────────────────────────────────────────────
SEP = "=" * 72
print(SEP)
print("Message Bus State: Before Consumption")
print(SEP)
depths = bus.all_depths()
for agent in AGENTS:
    depth = depths.get(agent, 0)
    bar   = "=" * depth
    print(f"  {agent:<20} queue depth = {depth:>2}  [{bar}]")

print()
print(SEP)
print("Consuming Messages by Agent:")
print(SEP)
print()
for agent in AGENTS:
    msgs = bus.consume(agent, max_n=10)
    if msgs:
        print(f"  {agent} consumed {len(msgs)} message(s):")
        for m in msgs:
            content_preview = str(m.content)[:55]
            pnames = {1:"LOW", 2:"NORMAL", 3:"HIGH", 4:"CRITICAL"}
            print(f"    [{pnames.get(m.priority,'?'):<8}] {m.msg_type:<10} from={m.from_id:<15} {content_preview}")
    else:
        print(f"  {agent}: no messages")

print()
print(SEP)
print("Bus Statistics:")
print(SEP)
for k, v in bus.report().items():
    print(f"  {k:<20} {v}")
print(f"  {'dead_letter_queue':<20} {len(bus.dead_letter)} message(s)")
if bus.dead_letter:
    for dm in bus.dead_letter:
        print(f"    expired: {dm.summary()}")

print()
print(SEP)
print("Message Type Breakdown (delivered):")
print(SEP)
type_counts = {}
for m in bus.delivered:
    type_counts[m.msg_type] = type_counts.get(m.msg_type, 0) + 1
for t, c in sorted(type_counts.items()):
    bar = "=" * c
    print(f"  {t:<12} {c:>3}  {bar}")
""",
    },

    "3 · Blackboard Architecture with Conflict Resolution": {
        "description": (
            "Implement a blackboard shared state store: versioned writes, "
            "optimistic locking, event log, and three conflict resolution strategies."
        ),
        "language": "python",
        "code": """\
import time, random, copy
random.seed(3)

# ── Blackboard with versioned entries ─────────────────────────────────────────
class BlackboardEntry:
    def __init__(self, key, value, author, version=1):
        self.key     = key
        self.value   = value
        self.author  = author
        self.version = version
        self.ts      = time.monotonic()
        self.history = [(version, value, author)]

    def update(self, new_value, author, strategy="last_write_wins"):
        new_version = self.version + 1
        if strategy == "last_write_wins":
            self.value   = new_value
            self.version = new_version
        elif strategy == "append":
            if isinstance(self.value, list):
                self.value = self.value + (new_value if isinstance(new_value, list) else [new_value])
            else:
                self.value = [self.value, new_value]
            self.version = new_version
        elif strategy == "highest_confidence":
            # value is (content, confidence); keep whichever has higher confidence
            old_conf = self.value[1] if isinstance(self.value, tuple) else 0
            new_conf = new_value[1]  if isinstance(new_value, tuple)  else 0
            if new_conf > old_conf:
                self.value = new_value
            self.version = new_version
        self.author = author
        self.history.append((new_version, copy.deepcopy(new_value), author))

class Blackboard:
    def __init__(self):
        self.store       = {}
        self.event_log   = []
        self.write_count = 0
        self.conflict_count = 0

    def write(self, key, value, author, strategy="last_write_wins",
              expected_version=None):
        if key in self.store:
            entry = self.store[key]
            # Optimistic locking check
            if expected_version is not None and entry.version != expected_version:
                self.conflict_count += 1
                self.event_log.append({
                    "op": "conflict", "key": key, "author": author,
                    "expected": expected_version, "actual": entry.version,
                    "strategy": strategy,
                })
                if strategy == "abort":
                    return False, f"Version conflict on '{key}': expected {expected_version}, got {entry.version}"
            entry.update(value, author, strategy)
        else:
            self.store[key] = BlackboardEntry(key, value, author)
        self.write_count += 1
        self.event_log.append({
            "op": "write", "key": key, "author": author,
            "version": self.store[key].version, "strategy": strategy,
        })
        return True, "ok"

    def read(self, key, default=None):
        entry = self.store.get(key)
        if entry is None:
            return default, 0  # value, version
        self.event_log.append({"op": "read", "key": key, "author": "?"})
        return entry.value, entry.version

    def read_versioned(self, key):
        entry = self.store.get(key)
        if entry is None:
            return None, None, 0
        return entry.value, entry.author, entry.version

    def history(self, key):
        entry = self.store.get(key)
        return entry.history if entry else []

    def task_queue_push(self, task, author):
        val, ver = self.read("task_queue")
        if val is None:
            self.write("task_queue", [task], author, strategy="last_write_wins")
        else:
            self.write("task_queue", task, author, strategy="append")

    def task_queue_pop(self, agent):
        val, ver = self.read("task_queue")
        if val and isinstance(val, list) and val:
            task = val[0]
            ok, _ = self.write("task_queue", val[1:], agent, strategy="last_write_wins")
            return task if ok else None
        return None

    def stats(self):
        return {"keys": len(self.store), "writes": self.write_count,
                "conflicts": self.conflict_count, "events": len(self.event_log)}

# ── Simulate concurrent agent writes ─────────────────────────────────────────
bb = Blackboard()

# 1. Simple key-value writes from multiple agents
bb.write("goal", "Analyse LLM market trends for Q3 2024", "orchestrator")
bb.write("status", "in_progress", "orchestrator")
bb.write("search_results", [], "researcher-1")

# 2. Append strategy: multiple researchers contribute findings
findings = [
    ("researcher-1", "GPT-4 released in March 2023; multimodal; significant MMLU gains"),
    ("researcher-2", "Claude 3 family released Feb 2024; Opus tops MMLU benchmark"),
    ("researcher-1", "Gemini 1.5 Pro introduced 1M token context window in Feb 2024"),
]
for author, finding in findings:
    bb.write("findings", finding, author, strategy="append")

# 3. Confidence-based resolution: two analysts produce different quality estimates
bb.write("market_size_estimate", ("$35B by 2026", 0.60), "analyst-1",
         strategy="highest_confidence")
bb.write("market_size_estimate", ("$42B by 2026", 0.85), "analyst-2",
         strategy="highest_confidence")
bb.write("market_size_estimate", ("$38B by 2026", 0.70), "analyst-3",
         strategy="highest_confidence")

# 4. Optimistic locking: agent reads version, then writes conditionally
_, ver = bb.read("status")
ok, msg = bb.write("status", "analysis_complete", "analyst-1",
                   expected_version=ver)    # succeeds: version matches

_, ver2 = bb.read("status")
ok2, msg2 = bb.write("status", "writing", "writer",
                     expected_version=ver2 - 1)  # fails: stale version

# 5. Task queue
for task in ["research_transformers", "research_gpts", "analyse_data", "write_report"]:
    bb.task_queue_push(task, "orchestrator")
popped = [bb.task_queue_pop(f"worker-{i}") for i in range(3)]

# ── Report ─────────────────────────────────────────────────────────────────────
SEP = "=" * 72
print(SEP)
print("Blackboard State Snapshot")
print(SEP)
print()
for key, entry in bb.store.items():
    if key == "task_queue":
        val_str = str(entry.value)[:60]
    elif isinstance(entry.value, list):
        val_str = f"[{len(entry.value)} items]"
    elif isinstance(entry.value, tuple):
        val_str = f"({entry.value[0]}, conf={entry.value[1]})"
    else:
        val_str = str(entry.value)[:60]
    print(f"  {key:<25} v{entry.version:<3} by {entry.author:<15} = {val_str}")

print()
print(SEP)
print("Conflict Resolution Results:")
print(SEP)
print(f"  Last-write-wins  status: {bb.read('status')[0]!r}")
mkt_val, mkt_author, mkt_ver = bb.read_versioned("market_size_estimate")
print(f"  Confidence-based market_size: {mkt_val}  (winner: {mkt_author})")
print(f"  Optimistic lock success: {ok}   msg={msg!r}")
print(f"  Optimistic lock failure: {not ok2}  msg={msg2!r}")
print(f"  Task queue pops: {popped}")

print()
print(SEP)
print("Write History for 'market_size_estimate':")
print(SEP)
for ver, val, auth in bb.history("market_size_estimate"):
    if isinstance(val, tuple):
        print(f"  v{ver}  {auth:<15}  {val[0]}  confidence={val[1]}")
    else:
        print(f"  v{ver}  {auth:<15}  {val!r}")

print()
s = bb.stats()
print(SEP)
print("Blackboard Statistics:")
print(SEP)
for k, v in s.items():
    print(f"  {k:<15} {v}")
""",
    },

    "4 · LangGraph-Style State Graph Engine": {
        "description": (
            "Implement a state graph engine with typed state, nodes, conditional edges, "
            "reducers, cycles, and checkpointing — the core LangGraph abstraction."
        ),
        "language": "python",
        "code": """\
import copy, json

# ── State definition (TypedDict-style) ────────────────────────────────────────
def make_state(goal="", draft="", critique="", score=0.0,
               iteration=0, messages=None, final="", done=False):
    return {"goal": goal, "draft": draft, "critique": critique,
            "score": score, "iteration": iteration,
            "messages": messages if messages is not None else [],
            "final": final, "done": done}

# ── Reducer: how to merge list fields ────────────────────────────────────────
def reduce_state(old_state, updates):
    new_state = copy.deepcopy(old_state)
    for key, val in updates.items():
        if key == "messages" and isinstance(val, list):
            new_state["messages"] = new_state.get("messages", []) + val  # append
        else:
            new_state[key] = val
    return new_state

# ── Graph engine ──────────────────────────────────────────────────────────────
class StateGraph:
    def __init__(self, state_factory):
        self.nodes          = {}       # name -> fn(state) -> updates
        self.edges          = {}       # name -> next_name or conditional fn
        self.state_factory  = state_factory
        self.checkpoint_log = []
        self.entry_node     = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry(self, name):
        self.entry_node = name

    def add_edge(self, from_name, to_name):
        self.edges[from_name] = to_name

    def add_conditional_edge(self, from_name, condition_fn):
        self.edges[from_name] = condition_fn

    def checkpoint(self, node_name, state):
        self.checkpoint_log.append({
            "node": node_name, "iteration": state.get("iteration", 0),
            "score": state.get("score", 0.0),
            "draft_len": len(state.get("draft", "")),
            "done": state.get("done", False),
        })

    def run(self, initial_state, max_steps=20, verbose=True):
        state        = copy.deepcopy(initial_state)
        current_node = self.entry_node
        step         = 0
        SEP = "=" * 68

        if verbose:
            print(SEP)
            print(f"StateGraph execution")
            print(f"Entry: {current_node} | Goal: {state.get('goal','')[:50]}")
            print(SEP)
            print()

        while current_node and current_node != "END" and step < max_steps:
            fn = self.nodes.get(current_node)
            if fn is None:
                break

            updates = fn(state)
            state   = reduce_state(state, updates)
            self.checkpoint(current_node, state)
            step   += 1

            if verbose:
                score  = state.get("score", 0.0)
                itr    = state.get("iteration", 0)
                draft  = state.get("draft", "")[:50]
                crit   = state.get("critique", "")[:40]
                print(f"  Step {step:>2} [{current_node:<16}] "
                      f"iter={itr} score={score:.2f} draft='{draft}'")
                if crit:
                    print(f"            critique='{crit}'")

            # Resolve next node
            edge = self.edges.get(current_node)
            if callable(edge):
                current_node = edge(state)
            else:
                current_node = edge

        if verbose:
            print()
            final = state.get("final") or state.get("draft", "")
            print(SEP)
            print(f"Final output (after {step} steps):")
            print(f"  {final}")
            print(SEP)

        return state, step

# ── Nodes for a draft → critique → revise loop ────────────────────────────────
DRAFTS = [
    "Transformers have improved NLP.",
    "Transformers, introduced in 2017, have dramatically improved NLP tasks including translation and QA.",
    "Transformers (Vaswani et al., 2017) use self-attention and have improved SOTA on translation, QA, and summarisation by 20-40%.",
    "Transformers (Vaswani et al., 2017) use multi-head self-attention and positional encodings. They have improved SOTA on translation (+4 BLEU), reading comprehension (+15% F1), and summarisation (+8 ROUGE) over previous RNN-based models.",
]

def node_planner(state):
    return {"messages": [{"role": "planner", "content": f"Plan: draft about '{state['goal']}'"}]}

def node_writer(state):
    itr   = state.get("iteration", 0)
    draft = DRAFTS[min(itr, len(DRAFTS) - 1)]
    return {"draft": draft, "messages": [{"role": "writer", "content": draft[:40]}]}

def node_critic(state):
    draft = state.get("draft", "")
    # Score by word count and presence of citations
    words     = len(draft.split())
    has_cite  = "et al." in draft or "2017" in draft
    has_stats = any(c.isdigit() for c in draft)
    score     = min(0.3 + words * 0.008 + (0.2 if has_cite else 0) + (0.15 if has_stats else 0), 1.0)
    if score < 0.9:
        critique = f"Score {score:.2f}: needs {'citations' if not has_cite else ''} {'statistics' if not has_stats else ''} more depth".strip()
    else:
        critique = f"Score {score:.2f}: publication ready"
    return {"score": round(score, 2), "critique": critique,
            "messages": [{"role": "critic", "content": critique[:40]}]}

def node_revise(state):
    itr = state.get("iteration", 0) + 1
    return {"iteration": itr, "messages": [{"role": "reviser", "content": f"revision #{itr}"}]}

def node_finalise(state):
    draft = state.get("draft", "")
    final = f"[FINAL v{state.get('iteration',0)+1}] {draft}"
    return {"final": final, "done": True}

def route_after_critic(state):
    if state.get("score", 0) >= 0.90:
        return "finalise"
    if state.get("iteration", 0) >= 3:
        return "finalise"   # max iterations guard
    return "revise"

# ── Build and run the graph ────────────────────────────────────────────────────
graph = StateGraph(make_state)
graph.add_node("planner",  node_planner)
graph.add_node("writer",   node_writer)
graph.add_node("critic",   node_critic)
graph.add_node("revise",   node_revise)
graph.add_node("finalise", node_finalise)

graph.set_entry("planner")
graph.add_edge("planner",  "writer")
graph.add_edge("writer",   "critic")
graph.add_conditional_edge("critic", route_after_critic)
graph.add_edge("revise",   "writer")   # ← CYCLE: revise loops back to writer
graph.add_edge("finalise", "END")

initial = make_state(goal="impact of transformers on NLP")
final_state, total_steps = graph.run(initial)

# Checkpoint log
print()
print("=" * 68)
print("Checkpoint Log (state at each node):")
print("=" * 68)
print(f"  {'Step':>4} {'Node':<16} {'Iter':>5} {'Score':>7} {'DraftLen':>9}")
print("  " + "-" * 50)
for i, cp in enumerate(graph.checkpoint_log, 1):
    print(f"  {i:>4} {cp['node']:<16} {cp['iteration']:>5} {cp['score']:>7.2f} {cp['draft_len']:>9}")

print()
print("=" * 68)
print(f"Messages accumulated in state ({len(final_state['messages'])} total):")
print("=" * 68)
for msg in final_state["messages"]:
    print(f"  [{msg['role']:<10}] {msg['content'][:60]}")
""",
    },

    "5 · Consensus Voting: Majority, Weighted, Borda": {
        "description": (
            "Implement and compare three consensus algorithms — majority vote, "
            "weighted vote, and Borda count — with accuracy analysis across agent counts."
        ),
        "language": "python",
        "code": """\
import math, random
random.seed(9)

# ── Consensus functions ───────────────────────────────────────────────────────
def majority_vote(answers):
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    winner = max(counts, key=lambda k: counts[k])
    total  = len(answers)
    return winner, counts[winner] / total

def weighted_vote(answers, weights):
    scores = {}
    total  = sum(weights)
    for a, w in zip(answers, weights):
        scores[a] = scores.get(a, 0) + w
    winner = max(scores, key=lambda k: scores[k])
    return winner, scores[winner] / total

def borda_count(ranked_lists, candidates):
    n = len(candidates)
    scores = {c: 0 for c in candidates}
    for ranking in ranked_lists:
        for pos, cand in enumerate(ranking):
            if cand in scores:
                scores[cand] += (n - pos)
    winner = max(scores, key=lambda k: scores[k])
    total  = sum(scores.values())
    return winner, scores[winner] / max(total, 1), scores

def llm_judge(answers, correct_answer):
    # Simulated judge: picks majority answer but with 10% chance of wrong pick
    maj, conf = majority_vote(answers)
    if random.random() < 0.10:
        alternatives = [a for a in set(answers) if a != maj]
        if alternatives:
            return random.choice(alternatives), 0.4
    return maj, conf

# ── Simulate agents answering a question ─────────────────────────────────────
CORRECT_ANSWER  = "Paris"
WRONG_ANSWERS   = ["London", "Berlin", "Madrid", "Rome"]

def simulate_agents(n_agents, p_correct, rng):
    answers = []
    for _ in range(n_agents):
        if rng.random() < p_correct:
            answers.append(CORRECT_ANSWER)
        else:
            answers.append(rng.choice(WRONG_ANSWERS))
    return answers

# ── Accuracy experiments ───────────────────────────────────────────────────────
N_TRIALS = 2000
SEP = "=" * 72
print(SEP)
print("Consensus Mechanisms: Accuracy Analysis")
print(f"Question: 'What is the capital of France?'")
print(f"Correct answer: '{CORRECT_ANSWER}'")
print(SEP)
print()

# 1. Majority vote: accuracy vs N agents for different base p
print("Majority Vote: Accuracy vs Agent Count")
print(f"  (Each cell = % correct across {N_TRIALS} trials)")
print()
agent_counts = [1, 3, 5, 7, 9]
p_values     = [0.55, 0.65, 0.75, 0.85]
header = "  p(correct)  " + "  ".join(f"N={n:>2}" for n in agent_counts)
print(header)
print("  " + "-" * (len(header) - 2))
for p in p_values:
    rng  = random.Random(42)
    row  = f"  p={p:.2f}       "
    for n in agent_counts:
        wins = sum(
            1 for _ in range(N_TRIALS)
            if majority_vote(simulate_agents(n, p, rng))[0] == CORRECT_ANSWER
        )
        row += f"  {100*wins/N_TRIALS:>5.1f}%"
    print(row)

print()

# 2. Weighted vote experiment
print("=" * 72)
print("Weighted Vote: Effect of Weight Distribution (N=5 agents, p=0.70)")
print("=" * 72)
rng   = random.Random(7)
p     = 0.70
n_ag  = 5
WEIGHT_SCHEMES = {
    "Uniform (1,1,1,1,1)":          [1.0, 1.0, 1.0, 1.0, 1.0],
    "Expert first (3,2,1,1,1)":     [3.0, 2.0, 1.0, 1.0, 1.0],
    "Expert first (5,3,1,1,1)":     [5.0, 3.0, 1.0, 1.0, 1.0],
    "Strongly skewed (10,1,1,1,1)": [10., 1.0, 1.0, 1.0, 1.0],
}
for scheme_name, weights in WEIGHT_SCHEMES.items():
    wins = 0
    rng2 = random.Random(99)
    for _ in range(N_TRIALS):
        # Expert agent has p+0.15 accuracy; others have p
        ans = [CORRECT_ANSWER if rng2.random() < (p + 0.15) else rng2.choice(WRONG_ANSWERS)]
        for i in range(1, n_ag):
            ans.append(CORRECT_ANSWER if rng2.random() < p else rng2.choice(WRONG_ANSWERS))
        winner, _ = weighted_vote(ans, weights)
        if winner == CORRECT_ANSWER:
            wins += 1
    print(f"  {scheme_name:<35} {100*wins/N_TRIALS:.1f}%")

print()

# 3. Borda count on a ranking task
print("=" * 72)
print("Borda Count: Ranking Task (which framework is best for production?)")
print("=" * 72)
candidates = ["LangGraph", "AutoGen", "CrewAI", "Custom"]
agent_rankings = [
    ["LangGraph", "AutoGen",  "Custom",    "CrewAI"],   # agent 1: prefers structured
    ["LangGraph", "Custom",   "AutoGen",   "CrewAI"],   # agent 2: prefers structured
    ["AutoGen",   "LangGraph","Custom",    "CrewAI"],   # agent 3: AutoGen fan
    ["Custom",    "LangGraph","AutoGen",   "CrewAI"],   # agent 4: prefers minimal deps
    ["LangGraph", "CrewAI",   "AutoGen",   "Custom"],   # agent 5: likes high-level APIs
    ["AutoGen",   "Custom",   "LangGraph", "CrewAI"],   # agent 6: code-exec focus
    ["Custom",    "AutoGen",  "LangGraph", "CrewAI"],   # agent 7: minimal deps
]
winner, winner_frac, all_scores = borda_count(agent_rankings, candidates)
print(f"  Rankings submitted by {len(agent_rankings)} agents:")
print()
for agent_i, ranking in enumerate(agent_rankings, 1):
    print(f"  Agent {agent_i}: {' > '.join(ranking)}")
print()
print(f"  Borda scores:")
total_pts = sum(all_scores.values())
for c in candidates:
    pts = all_scores[c]
    pct = 100 * pts / max(total_pts, 1)
    bar = "=" * int(pct / 3)
    print(f"    {c:<12} {pts:>4} pts  {pct:>5.1f}%  {bar}")
print()
print(f"  Winner: {winner}  ({winner_frac*100:.1f}% of Borda points)")
print()
print(f"  Majority vote winner:  {majority_vote([r[0] for r in agent_rankings])[0]}")
print(f"  Borda winner:          {winner}")
print(f"  (Majority only considers first-choice; Borda considers all rankings)")
""",
    },

    "6 · Multi-Agent Debate and Critic Pattern": {
        "description": (
            "Implement the full debate pattern: proposer, challenger, rebuttal, "
            "and LLM-as-judge — with win-rate analysis showing when debate helps."
        ),
        "language": "python",
        "code": """\
import random, math
random.seed(11)

# ── Simulated agents (scripted for deterministic demo) ────────────────────────
class DebateAgent:
    def __init__(self, name, role, accuracy=0.75, verbosity=1.0):
        self.name     = name
        self.role     = role
        self.accuracy = accuracy  # prob of holding correct position
        self.verbosity= verbosity
        self.wins     = 0
        self.calls    = 0

    def argue(self, position, evidence, round_num=1):
        self.calls += 1
        ARGUMENTS = {
            "transformers_superior": [
                "Transformers achieve state-of-the-art on every major NLP benchmark including GLUE, SuperGLUE, and SQuAD.",
                "The attention mechanism allows direct token-to-token comparison regardless of distance, solving the vanishing gradient problem of RNNs.",
                "Transfer learning via pretraining (BERT, GPT) enables few-shot performance that RNNs cannot match without full fine-tuning.",
                "Parallelism during training allows transformers to scale to 100B+ parameters; RNNs are inherently sequential.",
            ],
            "transformers_not_superior": [
                "RNNs such as LSTM still outperform transformers on long-sequence time-series with limited compute budgets.",
                "Transformers have O(N²) attention complexity; for sequences >100K tokens, RNNs and SSMs are more efficient.",
                "On streaming inference tasks requiring constant memory, RNNs are strictly more efficient than transformers.",
                "RWKV and Mamba (2023) achieve transformer-level quality with RNN-like O(N) complexity, challenging the premise.",
            ],
        }
        key  = position.replace("-", "_").lower()
        args = ARGUMENTS.get(key, [f"Evidence supporting {position}"])
        idx  = min(round_num - 1, len(args) - 1)
        return args[idx]

class Judge:
    def __init__(self, name, bias=0.0):
        self.name     = name
        self.bias     = bias  # +ve biases toward proposer
        self.decisions= []

    def evaluate(self, proposition, pro_args, con_args, true_answer):
        # Simulated judge scoring based on argument quality proxy (length + keywords)
        pro_score = sum(len(a.split()) for a in pro_args) * 0.01
        con_score = sum(len(a.split()) for a in con_args) * 0.01
        pro_score += self.bias
        # Keywords that signal strong evidence
        strong_kws = ["benchmark", "complexity", "SOTA", "state-of-the-art", "scales", "prove"]
        pro_score += sum(1 for kw in strong_kws if any(kw.lower() in a.lower() for a in pro_args)) * 0.15
        con_score += sum(1 for kw in strong_kws if any(kw.lower() in a.lower() for a in con_args)) * 0.15
        winner  = "proposer" if pro_score > con_score else "challenger"
        correct = winner == ("proposer" if true_answer == "pro" else "challenger")
        decision = {"winner": winner, "pro_score": round(pro_score,2),
                    "con_score": round(con_score,2), "correct": correct}
        self.decisions.append(decision)
        return decision

# ── Run a structured debate ───────────────────────────────────────────────────
proposition = "Transformers are strictly superior to RNNs for all NLP tasks"
TRUE_ANSWER  = "con"   # actually false: see RNN advantages above

proposer   = DebateAgent("Alex",  "proposer",  accuracy=0.80)
challenger = DebateAgent("Morgan","challenger", accuracy=0.75)
judge      = Judge("Dr. Lee")

SEP = "=" * 72
print(SEP)
print("Multi-Agent Debate")
print(f"Proposition: '{proposition}'")
print(SEP)
print()

pro_args = []
con_args = []

for round_num in range(1, 4):
    print(f"  --- Round {round_num} ---")
    pro_arg = proposer.argue("transformers_superior", {}, round_num)
    con_arg = challenger.argue("transformers_not_superior", {}, round_num)
    pro_args.append(pro_arg)
    con_args.append(con_arg)
    print(f"  Proposer:   {pro_arg[:80]}{'...' if len(pro_arg)>80 else ''}")
    print(f"  Challenger: {con_arg[:80]}{'...' if len(con_arg)>80 else ''}")
    print()

decision = judge.evaluate(proposition, pro_args, con_args, TRUE_ANSWER)
print(SEP)
print("Judge's Decision:")
print(SEP)
print(f"  Pro score:    {decision['pro_score']}")
print(f"  Con score:    {decision['con_score']}")
print(f"  Winner:       {decision['winner'].upper()}")
print(f"  Correct:      {decision['correct']}  (True answer: '{TRUE_ANSWER}')")

# ── Debate vs single-agent accuracy: Monte Carlo ──────────────────────────────
print()
print(SEP)
print("Debate vs Single-Agent: Accuracy Across 500 Simulated Questions")
print(SEP)
print()

N_QUESTIONS = 500
rng = random.Random(42)

# Single agent: correct with p=0.72
single_agent_correct = sum(1 for _ in range(N_QUESTIONS) if rng.random() < 0.72)

# Debate: two agents argue; judge picks. If both are right, judge almost certainly
# picks right (0.95). If one is right, judge picks right 0.75 of the time.
# If both wrong, judge picks wrong.
debate_correct = 0
for _ in range(N_QUESTIONS):
    p1_right = rng.random() < 0.72
    p2_right = rng.random() < 0.68  # challenger slightly weaker
    if p1_right and p2_right:
        debate_correct += 1 if rng.random() < 0.95 else 0
    elif p1_right or p2_right:
        debate_correct += 1 if rng.random() < 0.75 else 0
    # else: both wrong → debate doesn't help

print(f"  {'Method':<30} {'Correct':>8} {'Accuracy':>10}")
print("  " + "-" * 52)
print(f"  {'Single agent (p=0.72)':<30} {single_agent_correct:>8} {100*single_agent_correct/N_QUESTIONS:>9.1f}%")
print(f"  {'Debate (two agents + judge)':<30} {debate_correct:>8} {100*debate_correct/N_QUESTIONS:>9.1f}%")
gain = debate_correct - single_agent_correct
print(f"  {'Gain from debate':<30} {gain:>+8} {100*gain/N_QUESTIONS:>+9.1f}%")

print()
print("  When does debate help most?")
conditions = [
    ("Both agents correct",    0.72, 0.68, 0.95, "High-knowledge domain"),
    ("Mixed correctness",      0.55, 0.55, 0.75, "Moderate-difficulty task"),
    ("Both often wrong",       0.35, 0.35, 0.75, "Hard task (debate less useful)"),
    ("One expert, one novice", 0.90, 0.40, 0.80, "Expert + generalist pair"),
]
print(f"  {'Scenario':<30} {'p1':>5} {'p2':>5} {'Debate acc':>12}")
print("  " + "-" * 58)
for scenario, p1, p2, judge_p, label in conditions:
    rng3 = random.Random(0)
    wins = 0
    for _ in range(N_QUESTIONS):
        r1 = rng3.random() < p1
        r2 = rng3.random() < p2
        if r1 and r2:
            wins += 1 if rng3.random() < 0.95 else 0
        elif r1 or r2:
            wins += 1 if rng3.random() < judge_p else 0
    print(f"  {scenario:<30} {p1:>5.2f} {p2:>5.2f} {100*wins/N_QUESTIONS:>11.1f}%")
""",
    },

    "7 · Map-Reduce for Parallel Document Analysis": {
        "description": (
            "Implement agent map-reduce: split a corpus into chunks, analyse in parallel "
            "with specialised workers, then aggregate with four reduce strategies."
        ),
        "language": "python",
        "code": """\
import math, random
random.seed(7)

# ── Simulated document corpus ──────────────────────────────────────────────────
CORPUS = [
    {"id": "doc-1", "title": "Attention Is All You Need",
     "year": 2017, "citations": 91000,
     "text": "We propose a new simple network architecture, the Transformer. "
             "The model achieves 28.4 BLEU on WMT 2014 English-to-German translation. "
             "Multi-head attention allows the model to jointly attend to information "
             "from different representation subspaces.", "domain": "architecture"},
    {"id": "doc-2", "title": "BERT: Pre-training of Deep Bidirectional Transformers",
     "year": 2018, "citations": 82000,
     "text": "We introduce BERT, which stands for Bidirectional Encoder Representations "
             "from Transformers. BERT obtains new state-of-the-art results on eleven NLP tasks. "
             "The key idea is masked language modelling and next sentence prediction.", "domain": "pretraining"},
    {"id": "doc-3", "title": "Language Models are Few-Shot Learners",
     "year": 2020, "citations": 48000,
     "text": "We train GPT-3, an autoregressive language model with 175 billion parameters. "
             "GPT-3 achieves strong performance on many NLP datasets in the zero-shot, "
             "one-shot, and few-shot settings without fine-tuning.", "domain": "scaling"},
    {"id": "doc-4", "title": "Training language models to follow instructions",
     "year": 2022, "citations": 19000,
     "text": "We train InstructGPT models using RLHF. Human evaluators prefer InstructGPT "
             "outputs to GPT-3 despite GPT-3 having 100× more parameters. "
             "RLHF significantly reduces harmful, dishonest, and unhelpful outputs.", "domain": "alignment"},
    {"id": "doc-5", "title": "Constitutional AI: Harmlessness from AI Feedback",
     "year": 2022, "citations": 3800,
     "text": "We present Constitutional AI, a method for training harmless AI. "
             "The method uses a set of principles to guide revision of outputs. "
             "A supervised learning phase and RL phase combine to produce helpful, harmless models.", "domain": "alignment"},
    {"id": "doc-6", "title": "Scaling Laws for Neural Language Models",
     "year": 2020, "citations": 6500,
     "text": "Performance of language models scales as a power-law with compute, dataset size, "
             "and model size. Optimal allocation: for a 10× compute increase, "
             "model size and tokens should both grow roughly 3×.", "domain": "scaling"},
]

# ── Map functions (worker agents) ─────────────────────────────────────────────
def map_extract_claims(doc):
    text   = doc["text"]
    words  = text.split()
    # Extract sentences with numbers or comparisons
    claims = [s.strip() for s in text.split(".") if any(c.isdigit() for c in s) and len(s.split()) > 5]
    return {"doc_id": doc["id"], "title": doc["title"][:30],
            "claims": claims[:2], "word_count": len(words)}

def map_citation_analysis(doc):
    return {"doc_id": doc["id"], "title": doc["title"][:30],
            "year": doc["year"], "citations": doc["citations"], "domain": doc["domain"],
            "impact_score": round(doc["citations"] / max(2024 - doc["year"], 1), 0)}

def map_keyword_frequency(doc):
    text  = doc["text"].lower()
    words = text.split()
    kws   = ["attention", "model", "training", "performance", "language",
             "transformer", "bert", "gpt", "rlhf", "scaling"]
    freq  = {kw: text.count(kw) for kw in kws if kw in text}
    return {"doc_id": doc["id"], "keywords": freq, "total_words": len(words)}

# ── Reduce functions ───────────────────────────────────────────────────────────
def reduce_concatenate(map_results):
    all_claims = []
    for r in map_results:
        for claim in r.get("claims", []):
            all_claims.append({"source": r["title"], "claim": claim})
    return {"total_claims": len(all_claims), "claims": all_claims}

def reduce_aggregate_stats(map_results):
    citations   = [r["citations"]  for r in map_results]
    impact      = [r["impact_score"] for r in map_results]
    by_domain   = {}
    for r in map_results:
        d = r["domain"]
        by_domain[d] = by_domain.get(d, 0) + 1
    return {"total_docs": len(map_results),
            "total_citations": sum(citations), "avg_citations": sum(citations)//len(citations),
            "max_citations": max(citations), "avg_impact": round(sum(impact)/len(impact)),
            "by_domain": by_domain}

def reduce_keyword_merge(map_results):
    global_freq = {}
    for r in map_results:
        for kw, cnt in r.get("keywords", {}).items():
            global_freq[kw] = global_freq.get(kw, 0) + cnt
    return {"global_keyword_freq": dict(sorted(global_freq.items(), key=lambda x: -x[1]))}

def reduce_election(map_results, score_key="impact_score"):
    best = max(map_results, key=lambda r: r.get(score_key, 0))
    return {"winner": best["title"], "score": best.get(score_key, 0)}

# ── Run map-reduce pipeline ────────────────────────────────────────────────────
SEP = "=" * 72
print(SEP)
print(f"Map-Reduce Agent Pipeline: {len(CORPUS)} documents, 3 map functions")
print(SEP)
print()

# MAP phase (would be parallel in production)
claims_results  = [map_extract_claims(doc)    for doc in CORPUS]
citation_results= [map_citation_analysis(doc) for doc in CORPUS]
keyword_results = [map_keyword_frequency(doc) for doc in CORPUS]

print(f"  MAP phase complete:")
print(f"    {len(claims_results):>3} claim extractions")
print(f"    {len(citation_results):>3} citation analyses")
print(f"    {len(keyword_results):>3} keyword analyses")
print()

# REDUCE phase
r_claims  = reduce_concatenate(claims_results)
r_stats   = reduce_aggregate_stats(citation_results)
r_keywords= reduce_keyword_merge(keyword_results)
r_winner  = reduce_election(citation_results, "impact_score")

print(SEP)
print("REDUCE: Claims (concatenation strategy)")
print(SEP)
print(f"  Total claims extracted: {r_claims['total_claims']}")
for c in r_claims["claims"][:4]:
    print(f"    [{c['source'][:25]:<25}] {c['claim'][:60]}...")
print()

print(SEP)
print("REDUCE: Citation Stats (aggregation strategy)")
print(SEP)
for k, v in r_stats.items():
    if k != "by_domain":
        print(f"  {k:<25} {v}")
print("  Domain distribution:")
for domain, count in r_stats["by_domain"].items():
    bar = "=" * (count * 4)
    print(f"    {domain:<15} {count}  {bar}")
print()

print(SEP)
print("REDUCE: Global Keyword Frequency (merge strategy)")
print(SEP)
for kw, freq in list(r_keywords["global_keyword_freq"].items())[:8]:
    bar = "=" * freq
    print(f"  {kw:<15} {freq:>3}  {bar}")
print()

print(SEP)
print("REDUCE: Highest-Impact Paper (election strategy)")
print(SEP)
print(f"  Winner:  {r_winner['winner']}")
print(f"  Score:   {r_winner['score']:.0f} citations/year")
print()

print(SEP)
print("Map-Reduce Strategy Guide:")
print(SEP)
guide = [
    ("Concatenation", "Collect all outputs as a list",             "Claim extraction, fact gathering"),
    ("Aggregation",   "Compute statistics across all outputs",     "Numeric analysis, counting"),
    ("Merge",         "Combine overlapping data structures",       "Keyword maps, entity sets"),
    ("Election",      "Pick the single best output by a score",    "Best answer, top document"),
    ("Summary",       "LLM synthesises all outputs into prose",    "Final report generation"),
]
for strat, desc, use_case in guide:
    print(f"  {strat:<15} {desc:<42} eg: {use_case}")
""",
    },

    "8 · Market-Based Task Allocation (Agent Auction)": {
        "description": (
            "Simulate a Contract Net Protocol auction where agents bid on tasks "
            "based on capability and load; the broker assigns to the best bidder."
        ),
        "language": "python",
        "code": """\
import random, math
random.seed(13)

# ── Agent with skills and load ────────────────────────────────────────────────
class AuctionAgent:
    def __init__(self, name, skills, capacity=3):
        self.name       = name
        self.skills     = skills   # dict: skill -> proficiency (0-1)
        self.capacity   = capacity # max concurrent tasks
        self.queue      = []
        self.completed  = 0
        self.earnings   = 0.0

    def bid(self, task):
        skill_needed = task["required_skill"]
        proficiency  = self.skills.get(skill_needed, 0.0)
        load_factor  = len(self.queue) / self.capacity
        if proficiency == 0 or load_factor >= 1.0:
            return None  # cannot take this task
        # Bid = cost (lower is better); skilled agents charge more but deliver better
        base_cost    = task.get("base_cost", 1.0)
        skill_bonus  = proficiency
        load_penalty = load_factor * 0.5
        # Quality-adjusted cost: skilled agents bid competitively
        quality_adj_cost = base_cost / max(proficiency, 0.01) * (1 + load_penalty)
        return {"agent": self.name, "cost": round(quality_adj_cost, 3),
                "quality": proficiency, "load": round(load_factor, 2),
                "delivery_estimate": round(1.0 / max(proficiency, 0.01) * (1 + load_penalty), 2)}

    def accept_task(self, task):
        self.queue.append(task)

    def complete_task(self, task_id, payment):
        self.queue = [t for t in self.queue if t["id"] != task_id]
        self.completed += 1
        self.earnings  += payment

    @property
    def utilisation(self):
        return len(self.queue) / self.capacity

# ── Broker ────────────────────────────────────────────────────────────────────
class Broker:
    def __init__(self, agents, selection="best_quality"):
        self.agents    = agents
        self.selection = selection  # "cheapest", "best_quality", "balanced"
        self.auctions  = []
        self.failed    = []

    def auction(self, task):
        bids   = [(a, a.bid(task)) for a in self.agents]
        bids   = [(a, b) for a, b in bids if b is not None]
        if not bids:
            self.failed.append(task)
            return None, None, []
        if self.selection == "cheapest":
            winner, bid = min(bids, key=lambda x: x[1]["cost"])
        elif self.selection == "best_quality":
            winner, bid = max(bids, key=lambda x: x[1]["quality"])
        else:  # balanced: score = quality / cost
            winner, bid = max(bids, key=lambda x: x[1]["quality"] / max(x[1]["cost"], 0.01))
        winner.accept_task(task)
        winner.complete_task(task["id"], bid["cost"])
        self.auctions.append({"task_id": task["id"], "task": task["name"],
                              "winner": winner.name, "bid": bid,
                              "n_bidders": len(bids)})
        return winner, bid, bids

# ── Define agents and tasks ────────────────────────────────────────────────────
agents = [
    AuctionAgent("CodeBot-Pro",    {"coding": 0.95, "testing": 0.85, "writing": 0.30}, capacity=4),
    AuctionAgent("ResearchBot",    {"research": 0.92, "writing": 0.80, "coding": 0.40}, capacity=3),
    AuctionAgent("WriterBot",      {"writing": 0.93, "research": 0.70, "coding": 0.20}, capacity=5),
    AuctionAgent("TestBot",        {"testing": 0.96, "coding": 0.75, "research": 0.30}, capacity=4),
    AuctionAgent("GeneralistBot",  {"coding": 0.60, "research": 0.65, "writing": 0.62,
                                    "testing": 0.58}, capacity=6),
]

tasks = [
    {"id": "T1",  "name": "Write API integration tests",    "required_skill": "testing",  "base_cost": 1.5, "priority": 3},
    {"id": "T2",  "name": "Research LLM benchmark papers",  "required_skill": "research", "base_cost": 1.2, "priority": 2},
    {"id": "T3",  "name": "Implement vector store module",  "required_skill": "coding",   "base_cost": 2.0, "priority": 3},
    {"id": "T4",  "name": "Write user documentation",       "required_skill": "writing",  "base_cost": 1.0, "priority": 2},
    {"id": "T5",  "name": "Benchmark inference latency",    "required_skill": "testing",  "base_cost": 1.3, "priority": 2},
    {"id": "T6",  "name": "Analyse competitor products",    "required_skill": "research", "base_cost": 1.5, "priority": 1},
    {"id": "T7",  "name": "Refactor attention module",      "required_skill": "coding",   "base_cost": 1.8, "priority": 3},
    {"id": "T8",  "name": "Draft product announcement",     "required_skill": "writing",  "base_cost": 0.9, "priority": 2},
    {"id": "T9",  "name": "Integration test suite",         "required_skill": "testing",  "base_cost": 1.6, "priority": 3},
    {"id": "T10", "name": "Summarise literature review",    "required_skill": "writing",  "base_cost": 1.1, "priority": 1},
]
tasks.sort(key=lambda t: -t["priority"])

SEP = "=" * 72
print(SEP)
print("Contract Net Protocol: Agent Task Auction")
print(f"  {len(agents)} agents, {len(tasks)} tasks, selection=best_quality")
print(SEP)
print()

broker = Broker(agents, selection="best_quality")

print(f"  {'Task':<35} {'Winner':<18} {'Quality':>8} {'Cost':>8} {'Bidders':>8}")
print("  " + "-" * 82)
for task in tasks:
    winner, bid, all_bids = broker.auction(task)
    if winner:
        print(f"  {task['name']:<35} {winner.name:<18} {bid['quality']:>8.2f} {bid['cost']:>8.3f} {len(all_bids):>8}")
    else:
        print(f"  {task['name']:<35} {'NO BID':<18}")

print()
print(SEP)
print("Agent Performance Summary:")
print(SEP)
print(f"  {'Agent':<20} {'Completed':>10} {'Earnings':>10} {'Utilisation':>13}")
print("  " + "-" * 58)
for a in agents:
    util_bar = "=" * int(a.utilisation * 10) + "." * (10 - int(a.utilisation * 10))
    print(f"  {a.name:<20} {a.completed:>10} {a.earnings:>10.3f} {a.utilisation*100:>11.0f}%  [{util_bar}]")

print()
print(SEP)
print("Strategy Comparison (same tasks, 3 broker strategies):")
print(SEP)
strategies = ["cheapest", "best_quality", "balanced"]
for strat in strategies:
    agents2 = [
        AuctionAgent("CodeBot-Pro",   {"coding": 0.95, "testing": 0.85, "writing": 0.30}, capacity=4),
        AuctionAgent("ResearchBot",   {"research": 0.92, "writing": 0.80, "coding": 0.40}, capacity=3),
        AuctionAgent("WriterBot",     {"writing": 0.93, "research": 0.70, "coding": 0.20}, capacity=5),
        AuctionAgent("TestBot",       {"testing": 0.96, "coding": 0.75, "research": 0.30}, capacity=4),
        AuctionAgent("GeneralistBot", {"coding": 0.60, "research": 0.65, "writing": 0.62, "testing": 0.58}, capacity=6),
    ]
    b2 = Broker(agents2, selection=strat)
    for task in tasks:
        b2.auction(task)
    total_cost = sum(a["bid"]["cost"] for a in b2.auctions)
    avg_qual   = sum(a["bid"]["quality"] for a in b2.auctions) / max(len(b2.auctions), 1)
    print(f"  {strat:<15} total_cost={total_cost:>7.3f}  avg_quality={avg_qual:.3f}  assigned={len(b2.auctions)}/{len(tasks)}")

print()
print(f"  Trade-off: cheapest minimises cost; best_quality maximises output quality;")
print(f"  balanced is the Pareto-optimal middle ground for most production systems.")
""",
    },

    "9 · Reflection and Self-Critique Loop": {
        "description": (
            "Implement the generator → critic → revise reflection loop with "
            "convergence tracking, diminishing-returns analysis, and quality scoring."
        ),
        "language": "python",
        "code": """\
import math, random
random.seed(17)

# ── Simulated quality rubric ───────────────────────────────────────────────────
def score_output(text, rubric):
    total, weights = 0.0, 0.0
    for criterion, weight in rubric.items():
        if criterion == "length":
            s = min(len(text.split()) / 100, 1.0)
        elif criterion == "citations":
            s = min(text.count("et al.") * 0.25 + text.count("2017") * 0.1 +
                    text.count("2022") * 0.1 + text.count("2023") * 0.1, 1.0)
        elif criterion == "technical_depth":
            kws = ["attention", "parameter", "BLEU", "benchmark", "training", "layer",
                   "token", "embedding", "softmax", "gradient"]
            s   = min(sum(1 for kw in kws if kw.lower() in text.lower()) * 0.1, 1.0)
        elif criterion == "structure":
            has_detail = "." in text[20:]
            has_multi  = len(text.split(".")) > 2
            s = 0.5 + 0.25 * has_detail + 0.25 * has_multi
        elif criterion == "accuracy":
            # Proxy: absence of vague/wrong claim markers
            bad = ["always", "never", "all models", "impossible", "trivially"]
            s   = 1.0 - 0.2 * sum(1 for b in bad if b in text.lower())
            s   = max(s, 0.0)
        else:
            s = 0.5
        total   += s * weight
        weights += weight
    return round(total / max(weights, 1.0), 3)

RUBRIC = {"length": 0.20, "citations": 0.25, "technical_depth": 0.30,
          "structure": 0.15, "accuracy": 0.10}

# ── Simulated drafts (progressively better) ───────────────────────────────────
DRAFT_POOL = [
    "Transformers are very powerful models used in NLP.",
    ("Transformer models have become dominant in NLP. "
     "They use attention mechanisms to process sequences."),
    ("Transformer models (Vaswani et al., 2017) are now dominant in NLP. "
     "They use self-attention to relate all tokens in a sequence. "
     "This enables parallelism during training, unlike RNNs."),
    ("Transformer models (Vaswani et al., 2017) have transformed NLP performance. "
     "Using multi-head self-attention and positional encodings, they achieved "
     "28.4 BLEU on WMT14 en-de translation. BERT (Devlin et al., 2018) applied "
     "bidirectional transformers to obtain SOTA on 11 NLP benchmarks."),
    ("Transformer models (Vaswani et al., 2017) have become the dominant architecture "
     "in NLP, outperforming RNNs and CNNs on every major benchmark. "
     "The self-attention mechanism computes pairwise token relationships in O(N^2) "
     "time, enabling parallelism but limiting context length. "
     "BERT (2018) and GPT-3 (Brown et al., 2020) demonstrated that pretraining "
     "on large corpora produces representations that transfer to diverse tasks. "
     "Scaling laws (Kaplan et al., 2020) showed power-law improvement with compute. "
     "Current frontier models reach >90% on MMLU and >50 BLEU on translation tasks."),
]

# ── Critique generator (scripted feedback per score range) ─────────────────────
def generate_critique(text, score, rubric):
    issues = []
    if len(text.split()) < 50:
        issues.append("Too brief — expand with more technical detail")
    if "et al." not in text and "2017" not in text:
        issues.append("Missing citations — add at least 2 specific paper references")
    kws = ["attention", "benchmark", "training", "parameter"]
    missing = [kw for kw in kws if kw.lower() not in text.lower()]
    if missing:
        issues.append(f"Missing key technical terms: {', '.join(missing[:2])}")
    if len(text.split(".")) < 3:
        issues.append("Single-sentence structure — write at least 3 distinct claims")
    if score >= 0.88:
        return f"Score {score:.3f}: Output is publication-ready. Minor polish only."
    return f"Score {score:.3f}. Issues: " + "; ".join(issues[:3]) if issues else f"Score {score:.3f}: Needs expansion."

# ── Reflection loop ────────────────────────────────────────────────────────────
class ReflectionAgent:
    def __init__(self, max_iterations=6, quality_threshold=0.88):
        self.max_iterations    = max_iterations
        self.quality_threshold = quality_threshold
        self.history           = []   # (iteration, draft, score, critique)

    def run(self, initial_draft, rubric, verbose=True):
        draft = initial_draft
        SEP   = "=" * 68
        if verbose:
            print(SEP)
            print("Reflection Loop: Generator → Critic → Revise")
            print(f"Threshold: {self.quality_threshold}  Max iterations: {self.max_iterations}")
            print(SEP)
            print()

        for iteration in range(self.max_iterations + 1):
            score    = score_output(draft, rubric)
            critique = generate_critique(draft, score, rubric)
            self.history.append((iteration, draft, score, critique))

            if verbose:
                draft_preview = draft[:70] + ("..." if len(draft) > 70 else "")
                print(f"  Iter {iteration}  score={score:.3f}  [{draft_preview}]")
                if score < self.quality_threshold:
                    print(f"          critique: {critique[:70]}...")
                else:
                    print(f"          ACCEPTED — quality threshold reached")
            if score >= self.quality_threshold:
                break
            if iteration < len(DRAFT_POOL) - 1:
                draft = DRAFT_POOL[iteration + 1]

        if verbose:
            print()
            print(SEP)
            print(f"Final output (iteration {iteration}):")
            print(SEP)
            for line in draft.split(". "):
                if line.strip():
                    print(f"  {line.strip()}.")
            print()
        return draft, self.history

agent = ReflectionAgent(max_iterations=6, quality_threshold=0.88)
final_draft, history = agent.run(DRAFT_POOL[0], RUBRIC)

# ── Analysis ───────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("Score Progression and Diminishing Returns:")
print("=" * 68)
scores = [h[2] for h in history]
gains  = [scores[i] - scores[i-1] for i in range(1, len(scores))]
print()
print(f"  {'Iter':>5} {'Score':>8} {'Gain':>8} {'Progress bar'}")
print("  " + "-" * 50)
for i, (itr, draft, score, crit) in enumerate(history):
    gain    = gains[i-1] if i > 0 else 0.0
    bar_len = int(score * 30)
    bar     = "=" * bar_len + "." * (30 - bar_len)
    print(f"  {itr:>5} {score:>8.3f} {gain:>+8.3f}  [{bar}]")

print()
total_gain = scores[-1] - scores[0]
print(f"  Total improvement: {total_gain:+.3f}  ({scores[0]:.3f} → {scores[-1]:.3f})")
if len(gains) > 1:
    early  = sum(gains[:len(gains)//2]) / max(len(gains)//2, 1)
    late   = sum(gains[len(gains)//2:]) / max(len(gains) - len(gains)//2, 1)
    print(f"  Early iterations avg gain: {early:+.4f}")
    print(f"  Late iterations avg gain:  {late:+.4f}")
    print(f"  Diminishing returns: {abs(late/max(abs(early),0.0001)):.1%} of early gain")
print()
print(f"  Research finding: reflection plateaus after 3-5 iterations (Saunders et al., 2022).")
print(f"  Beyond that: more rounds add latency without quality improvement.")
""",
    },

    "10 · Full Production Multi-Agent System": {
        "description": (
            "Assemble a complete production multi-agent system: typed message bus, "
            "blackboard state, role-specialised agents, orchestrator, debate critic, "
            "observability tracing, and cost accounting."
        ),
        "language": "python",
        "code": """\
import random, json, math, time
random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
class SharedState:
    def __init__(self):
        self.data     = {}
        self.lock_map = {}
        self.log      = []

    def set(self, key, value, author):
        self.data[key] = {"value": value, "author": author}
        self.log.append({"op": "set", "key": key, "author": author})

    def get(self, key, default=None):
        entry = self.data.get(key)
        return entry["value"] if entry else default

    def append_list(self, key, item, author):
        current = self.get(key) or []
        self.set(key, current + [item], author)

class CostLedger:
    GPT4_INPUT_PER_M  = 30.0
    GPT4_OUTPUT_PER_M = 60.0
    def __init__(self, budget_usd=5.0):
        self.budget  = budget_usd
        self.spent   = 0.0
        self.calls   = []

    def charge(self, agent, in_tok, out_tok, model="gpt-4"):
        cost = (in_tok * self.GPT4_INPUT_PER_M + out_tok * self.GPT4_OUTPUT_PER_M) / 1e6
        self.spent += cost
        self.calls.append({"agent": agent, "in": in_tok, "out": out_tok, "cost": cost})
        return cost

    def budget_ok(self):
        return self.spent < self.budget

    def summary(self):
        by_agent = {}
        for c in self.calls:
            a = c["agent"]
            by_agent[a] = {"calls": by_agent.get(a, {}).get("calls", 0) + 1,
                           "cost":  by_agent.get(a, {}).get("cost",  0.0) + c["cost"]}
        return {"total_calls": len(self.calls), "total_cost": round(self.spent, 5),
                "budget_pct": round(100 * self.spent / self.budget, 1),
                "by_agent": {k: {"calls": v["calls"], "cost": round(v["cost"], 5)}
                             for k, v in by_agent.items()}}

# ══════════════════════════════════════════════════════════════════════════════
# SPECIALISED AGENTS
# ══════════════════════════════════════════════════════════════════════════════
class BaseAgent:
    def __init__(self, name, role, state, ledger):
        self.name   = name
        self.role   = role
        self.state  = state
        self.ledger = ledger
        self.log    = []

    def emit(self, msg):
        self.log.append(msg)

    def run(self, task_context):
        raise NotImplementedError

class ResearchAgent(BaseAgent):
    FINDINGS = {
        "llm market": {"finding": "Global LLM market: $6.5B in 2023, projected $85B by 2030 (CAGR 33%)",
                       "confidence": 0.82},
        "key players": {"finding": "Top vendors: OpenAI (GPT-4), Anthropic (Claude), Google (Gemini), Meta (Llama)",
                        "confidence": 0.91},
        "enterprise adoption": {"finding": "72% of Fortune 500 piloting LLMs; 31% in production (McKinsey, 2024)",
                                "confidence": 0.78},
    }
    def run(self, task_context):
        topic  = task_context.get("topic", "llm market")
        result = self.FINDINGS.get(topic, {"finding": f"Research on {topic}", "confidence": 0.5})
        cost   = self.ledger.charge(self.name, 800, 200)
        self.state.append_list("research_findings", result, self.name)
        self.emit(f"Research complete: {result['finding'][:60]}... (conf={result['confidence']})")
        return result

class AnalystAgent(BaseAgent):
    def run(self, task_context):
        findings = self.state.get("research_findings") or []
        if not findings:
            return {"analysis": "No research findings available", "score": 0.0}
        avg_conf = sum(f.get("confidence", 0) for f in findings) / max(len(findings), 1)
        cost     = self.ledger.charge(self.name, 1200, 400)
        analysis = {
            "n_sources":   len(findings),
            "avg_confidence": round(avg_conf, 2),
            "summary": f"Based on {len(findings)} sources (avg confidence {avg_conf:.0%}): "
                       f"LLM market is large, fast-growing, and dominated by 4 major players. "
                       f"Enterprise adoption is accelerating.",
            "recommendation": "Strong buy signal for LLM infrastructure investment",
        }
        self.state.set("analysis", analysis, self.name)
        self.emit(f"Analysis: {analysis['summary'][:70]}...")
        return analysis

class CriticAgent(BaseAgent):
    def run(self, task_context):
        analysis = self.state.get("analysis") or {}
        if not analysis:
            return {"score": 0.0, "issues": ["No analysis to critique"]}
        cost      = self.ledger.charge(self.name, 900, 300)
        n_src     = analysis.get("n_sources", 0)
        avg_conf  = analysis.get("avg_confidence", 0)
        issues    = []
        score     = 0.7
        if n_src < 3:
            issues.append("Insufficient sources (need ≥3)")
            score -= 0.1
        if avg_conf < 0.80:
            issues.append(f"Average source confidence low ({avg_conf:.0%})")
            score -= 0.05
        if "market size" not in analysis.get("summary", "").lower():
            issues.append("Missing explicit market size figure")
        score     = max(round(score, 2), 0.0)
        verdict   = "APPROVED" if score >= 0.65 and len(issues) == 0 else "NEEDS_REVISION"
        critique  = {"score": score, "verdict": verdict, "issues": issues,
                     "confidence": avg_conf}
        self.state.set("critique", critique, self.name)
        self.emit(f"Critique: {verdict} score={score} issues={issues}")
        return critique

class WriterAgent(BaseAgent):
    def run(self, task_context):
        analysis  = self.state.get("analysis") or {}
        critique  = self.state.get("critique") or {}
        cost      = self.ledger.charge(self.name, 1500, 600)
        verdict   = critique.get("verdict", "APPROVED")
        issues_str= "; ".join(critique.get("issues", [])) or "none"
        report    = (
            "MARKET INTELLIGENCE REPORT: Large Language Models" + chr(10)
            + "-" * 50 + chr(10)
            + analysis.get("summary", "Analysis not available") + chr(10)
            + chr(10)
            + "RECOMMENDATION: " + analysis.get("recommendation", "") + chr(10)
            + "QUALITY CHECK: " + verdict + chr(10)
            + "RESIDUAL ISSUES: " + issues_str + chr(10)
            + "SOURCES: " + str(analysis.get("n_sources", 0)) + " research reports"
        )
        self.state.set("final_report", report, self.name)
        self.emit(f"Report drafted ({len(report.split())} words)")
        return {"report": report, "word_count": len(report.split())}

# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
class Orchestrator:
    def __init__(self, state, ledger, verbose=True):
        self.state   = state
        self.ledger  = ledger
        self.verbose = verbose
        self.agents  = {}
        self.pipeline= []

    def register(self, agent):
        self.agents[agent.name] = agent

    def add_step(self, agent_name, task_context):
        self.pipeline.append((agent_name, task_context))

    def run(self, goal):
        SEP = "=" * 72
        if self.verbose:
            print(SEP)
            print("Multi-Agent Orchestrator")
            print(f"Goal: {goal}")
            print(f"Pipeline: {' → '.join(n for n,_ in self.pipeline)}")
            print(f"Budget: ${self.ledger.budget:.2f}")
            print(SEP)
            print()

        self.ledger.charge("orchestrator", 500, 150)  # planner call
        results = {}
        for step_num, (agent_name, task_ctx) in enumerate(self.pipeline, 1):
            if not self.ledger.budget_ok():
                if self.verbose:
                    print(f"  BUDGET EXCEEDED — stopping at step {step_num}")
                break
            agent  = self.agents.get(agent_name)
            if not agent:
                if self.verbose:
                    print(f"  Step {step_num}: agent '{agent_name}' not found — SKIPPED")
                continue
            if self.verbose:
                print(f"  Step {step_num}: [{agent.role.upper():<12}] {agent_name}")
            result = agent.run(task_ctx)
            results[agent_name] = result
            for log_line in agent.log[-1:]:
                if self.verbose:
                    print(f"             → {log_line}")
            if self.verbose:
                print()

        # Orchestrator synthesises
        self.ledger.charge("orchestrator", 800, 300)
        final_report = self.state.get("final_report") or "Report not generated"

        if self.verbose:
            print(SEP)
            print("Final Report:")
            print(SEP)
            for line in final_report.split(chr(10)):
                print(f"  {line}")
            print()

        return results, final_report

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
state  = SharedState()
ledger = CostLedger(budget_usd=5.0)
orch   = Orchestrator(state, ledger, verbose=True)

researcher = ResearchAgent("researcher-1", "researcher", state, ledger)
analyst    = AnalystAgent("analyst-1",     "analyst",    state, ledger)
critic     = CriticAgent("critic-1",       "critic",     state, ledger)
writer     = WriterAgent("writer-1",       "writer",     state, ledger)

for agent in [researcher, analyst, critic, writer]:
    orch.register(agent)

# Research three angles in sequence (would be parallel in async production)
orch.add_step("researcher-1", {"topic": "llm market"})
orch.add_step("researcher-1", {"topic": "key players"})
orch.add_step("researcher-1", {"topic": "enterprise adoption"})
orch.add_step("analyst-1",    {})
orch.add_step("critic-1",     {})
orch.add_step("writer-1",     {})

results, report = orch.run("Produce an LLM market intelligence report")

# ── Cost and observability summary ────────────────────────────────────────────
print("=" * 72)
print("Cost and Observability Summary:")
print("=" * 72)
s = ledger.summary()
print(f"  Total LLM calls:  {s['total_calls']}")
print(f"  Total cost:       ${s['total_cost']:.5f}  ({s['budget_pct']}% of ${ledger.budget:.2f} budget)")
print()
print(f"  {'Agent':<20} {'Calls':>8} {'Cost':>12}")
print("  " + "-" * 44)
for agent_name, stats in s["by_agent"].items():
    print(f"  {agent_name:<20} {stats['calls']:>8} ${stats['cost']:>11.5f}")
print()
print(f"  Shared state keys written: {len(state.data)}")
for k in state.data:
    v = state.data[k]["value"]
    v_str = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
    print(f"    {k:<25} by={state.data[k]['author']:<15}  {v_str}")
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