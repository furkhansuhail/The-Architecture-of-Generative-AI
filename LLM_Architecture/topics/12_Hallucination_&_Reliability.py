"""
Tutorial Template — Automation & Infrastructure
=================================================
Copy this file, rename it, and fill in the sections.

The TOPIC_NAME and CATEGORY are parsed by the app.
OPERATIONS entries can specify "language" as "bash", "yaml", "python", "json", etc.
"""

TOPIC_NAME = "LLM - Hallucination & Reliability"
DISPLAY_NAME = "LLM - Hallucination & Reliability"
ICON = "📖"
SUBTITLE = "Explanation about LLM Hallucination & Reliability"
CATEGORY = "LLM Architecture"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = r"""

##### 1) What Is Hallucination?

In the context of large language models (LLMs), a hallucination is the confident generation 
of information that is factually incorrect, fabricated, or entirely unsupported by any 
grounding source. 

The term is borrowed loosely from psychology, where hallucinations describe perceptions with no 
external stimulus — and it fits: the model outputs something that sounds real, authoritative, 
and coherent, yet has no correspondence to reality.

---

### 1.1 Definition

Core definition: An LLM hallucination occurs when a model produces output that is presented 
with apparent confidence but is factually wrong, invented, or unverifiable — without the model 
flagging its own uncertainty.


## Key Characteristic

The defining feature of a hallucination is NOT just that the model is wrong — it is that the model
is confidently wrong. A model that says 'I'm not sure, but possibly...' and gets the fact wrong
is expressing calibrated uncertainty. A hallucinating model gives the wrong answer as though
it were established fact.

---

### 1.2 Taxonomy of Hallucination Types

    +-------------------------------+----------------------------------------------------------------------------------------------------------------------------+
    | Type                          | Description                                                                                                                |
    +-------------------------------+----------------------------------------------------------------------------------------------------------------------------+
    | Factual Hallucination         | The model states incorrect facts — wrong dates, wrong names, wrong statistics (e.g., citing a study that does not exist).  |
    | Entity Fabrication            | The model invents people, companies, publications, legal cases, or other named entities that do not exist.                 |
    | Temporal Hallucination        | The model misattributes when events occurred, confuses timelines, or applies outdated information as if it were current.   |
    | Attribution Hallucination     | The model falsely attributes a quote, finding, or claim to the wrong person or source.                                     |
    | Logical Hallucination         | The model's reasoning chain contains steps that do not follow, producing a wrong conclusion even from correct premises.    |
    | Instruction Hallucination     | The model claims it followed an instruction it did not (e.g., 'As I mentioned above' when no such mention exists).         |
    | Self-Knowledge Hallucination  | The model makes incorrect claims about its own capabilities, training, or architecture.                                    |
    +-------------------------------+----------------------------------------------------------------------------------------------------------------------------+

---

### 1.3 Illustrative Example

USER:  
        
        Who wrote the 1987 paper on transformer attention?

MODEL (hallucinating):

        The foundational 1987 paper on transformer attention was written by
        Dr. Alan H. Morris of MIT, published in the Journal of Computational
        Linguistics, volume 14, pp. 233-251.

REALITY:

        No such paper or author exists. The seminal 'Attention Is All You
        Need' paper was from 2017, not 1987, and was authored by Vaswani
        et al. at Google.


The model did not say 'I don't know' or hedge its answer. 
It produced specific, plausible-sounding, entirely fabricated details — the hallmark of hallucination.

---

### 1.4 Why 'Hallucination' vs. Other Terms

Some researchers prefer alternative terms like 'confabulation' (borrowed from neuropsychology, 
where brain-damaged patients fabricate memories without intent to deceive), 'fabrication', or 
simply 'model errors'. 

The term hallucination persists because it captures both dimensions: the model's internal state 
produces an output (like a perception) that has no grounding in external reality. 

It is not lying — the model has no intent — it is producing the statistically expected next 
tokens regardless of truth.

---

### 2. Why Do Models Hallucinate?

Understanding why hallucination occurs requires understanding the fundamental mechanism of 
autoregressive language models. 

The short answer: the model was never trained to know whether it knows something. 
It was trained to predict the next token.

---

### 2.1 The Next-Token Prediction Paradigm

**Core mechanism:** LLMs are trained via next-token prediction — given a sequence of tokens, 
predict the most probable next token. 

This objective has no built-in concept of 'true', 'false', or 'unknown'. 
A token that sounds right and fits the context is just as rewarded as a token that is 
factually accurate.


    Training objective (simplified):
    
      maximize P(token_t | token_1, token_2, ..., token_{t-1})
    
    This is a PURELY DISTRIBUTIONAL objective.
    The model learns: 'given this context, what text tends to follow?'
    NOT: 'given this context, what is the factually correct continuation?'
    
    ┌─────────────────────────────────────────────────────────────┐
    │  Input:  'The capital of Australia is...'                   │
    │  Model:  Finds completion that fits the pattern             │
    │  Output: 'Canberra'  (correct — commonly seen pattern)      │
    │                                                             │
    │  Input:  'The 1987 neural network paper by Morris...'       │
    │  Model:  Finds a plausible-sounding completion              │
    │  Output: '...vol 14, pp 233-251' (fabricated — no check)    │
    └─────────────────────────────────────────────────────────────┘

---
  
### 2.2 There Is No 'I Don't Know' in Vanilla Training

During pretraining on internet text, the model never encounters a training signal that says 
'stop generating and say you don't know'. 

Text on the internet rarely says 'I have no information about this' — it just proceeds to 
provide information. The model learns to imitate this behavior.

## The Silence Problem: 

The training corpus has an asymmetry: authoritative text fills in details,
even uncertain ones. The model learns to fill in gaps confidently because
that is what 'good text' looks like in the data. Epistemic hedges exist in
the corpus, but they are sparse relative to confident declarative sentences.

---

### 2.3 Causal Diagram of Hallucination


                    PRETRAINING
                        │
         ┌──────────────┴──────────────┐
         │                             │
   Learned patterns              Learned patterns
   from factual text             from plausible text
   (grounded in reality)         (not truth-checked)
         │                             │
         └──────────────┬──────────────┘
                        │
              Model weights encode
              statistical associations
                        │
              ┌─────────┴──────────┐
              │                    │
         Query in               Query about
         training distribution  sparse/missing knowledge
              │                    │
         Likely correct         Model interpolates
         answer                 or extrapolates
                                   │
                                HALLUCINATION

---

### 2.4 Architectural Factors

Beyond the training objective, several architectural properties make hallucination more likely:

---

## 2.4.1 No External Memory or Lookup

LLMs store 'knowledge' in their weights — billions of floating-point numbers learned during training. 
There is no dictionary lookup, no external database call, no retrieval mechanism in a base LLM. 
When the model lacks knowledge, it cannot fall back to a 'not found' signal; it must generate 
from the probability distribution it has learned.

---

## 2.4.2 Compression Artifacts

The training corpus is vast (trillions of tokens) but the model weights are finite. 
The model must compress world knowledge into parameters. 
This compression is lossy — rare facts are poorly retained, conflicting facts get blended, 
and the model may output confident-but-blended half-memories of multiple real things that don't 
match any single true thing.

---

## 2.4.3 Exposure Bias During Training

During training, the model always sees its own output conditioned on the true previous 
tokens (teacher forcing). 
At inference time, it generates token-by-token based on its own previous outputs. 
Small distributional errors early in generation compound and can lead the model into 
increasingly hallucinated territory.

---

## 2.4.4 Context Window Limitations

As context length grows, attention mechanisms may fail to maintain equal focus on all 
relevant parts of the input. 
For very long documents, information at certain positions (typically in the middle) 
receives less attention, potentially causing the model to 'forget' grounding information 
and fill the gap with a hallucination.

---

## 2.5 The Sycophancy Problem

RLHF (Reinforcement Learning from Human Feedback) training, used to align models, 
can inadvertently increase hallucination. 
Human raters often prefer confident, authoritative, well-structured answers — 
even when those answers are wrong. The model learns that confident-sounding completions 
receive better scores, reinforcing the tendency to generate assertive (and sometimes false) 
statements.

---

### 3. Factual vs. Reasoning Errors

Not all model errors are hallucinations. 
It is critical to distinguish between two fundamentally different failure modes: 
the model retrieving or generating the wrong facts, versus the model applying flawed 
logic to arrive at the wrong conclusion.

---

## 3.1 Factual Errors

A factual error occurs when the model asserts something about the world that is untrue — 
wrong names, dates, numbers, relationships, or attributions. 
The reasoning may be valid, but the premises are wrong.

    FACTUAL ERROR EXAMPLE:
    
      Q: Who invented the telephone?
      A: The telephone was invented by Thomas Edison in 1872.
    
      ERROR TYPE: Factual — Edison did not invent the telephone.
                  Alexander Graham Bell received the patent in 1876.
                  The reasoning form is fine; the stored fact is wrong.

---

## 3.2 Reasoning Errors

A reasoning error occurs when the model applies incorrect logical steps, even if the facts 
it starts with are correct. 
This is a different failure mode — the knowledge is there, but the inference engine is broken.

    REASONING ERROR EXAMPLE:
    
      Q: If all mammals are warm-blooded, and whales are mammals,
         are whales warm-blooded?
    
      A (with reasoning error):
         All mammals are warm-blooded.
         Whales live in cold water.
         Therefore, whales must be cold-blooded.
    
      ERROR TYPE: Reasoning — the model introduced an irrelevant
                  premise ('lives in cold water') and made an
                  invalid inference, ignoring the valid syllogism.

--- 

## 3.3 Comparison Table
    
    +--------------+--------------------------------------------------+
    |Factual Errors|                                                  |  
    +--------------+--------------------------------------------------+
    | Source       | Errors in stored/retrieved knowledge             |
    +--------------+--------------------------------------------------+
    | Mechanism    | Wrong facts, names, dates, attributions          |
    +--------------+--------------------------------------------------+
    | Logic        | May be structurally valid                        |
    +--------------+--------------------------------------------------+
    | Detection    | Fact-checking, external verification             |
    +--------------+--------------------------------------------------+
    | Mitigation   | RAG, citations, up-to-date training data         |
    +--------------+--------------------------------------------------+
    | Example      | 'Einstein won the 1921 Nobel Prize in Medicine'  |
    |              | (wrong field)                                    |
    +--------------+--------------------------------------------------+
    

    +------------------+------------------------------------------------------------+
    | Reasoning Errors |                                                            |
    +------------------+------------------------------------------------------------+
    | Source           | Errors in inference or logical steps                       |
    +------------------+------------------------------------------------------------+
    | Mechanism        | Invalid deductions, false premises injected mid-chain      |
    +------------------+------------------------------------------------------------+
    | Logic            | Structurally invalid                                       |
    +------------------+------------------------------------------------------------+
    | Detection        | Chain-of-thought inspection, logic verification            |
    +------------------+------------------------------------------------------------+
    | Mitigation       | Step-by-step prompting, formal verification, self-critique |
    +------------------+------------------------------------------------------------+
    | Example          | 'If X > 5 and X < 3, then X = 4'                           |
    |                  | (impossible constraint ignored)                            |
    +------------------+------------------------------------------------------------+

---

### 3.4 Mixed Errors

In practice, many model failures combine both types. 
A factual error can seed a reasoning chain that compounds into further errors. 
A reasoning error can generate false premises that then appear as factual outputs.

    MIXED ERROR CHAIN:
    
      Step 1 (Factual error):   'Python was created in 1989'   ← wrong (1991)
      Step 2 (Reasoning error): 'Since Python predates the web',
                                 'it was designed for pre-web scripting'
      Step 3 (Conclusion):      'Therefore Python has no native HTTP support'  ← wrong
    
      The initial factual error triggered a plausible-but-false
      reasoning chain, compounding into a false conclusion.
    
---

### 4. Hallucination in RAG Systems

Retrieval-Augmented Generation (RAG) is one of the most widely deployed architectures for 
grounding LLM outputs in external knowledge. 
While RAG significantly reduces hallucination rates in many settings, it also introduces 
its own category of failure modes that practitioners must understand.

---

## 4.1 RAG Architecture Overview

          RAG PIPELINE
        ═══════════════════════════════════════════════════════════
    
          User Query
              │
              ▼
          ┌─────────────────┐
          │   RETRIEVER     │   (dense / sparse / hybrid)
          │  Vector DB /    │
          │  Search Index   │
          └────────┬────────┘
                   │  top-k relevant chunks
                   ▼
          ┌─────────────────┐
          │   CONTEXT       │
          │   ASSEMBLY      │   (reranking, deduplication, truncation)
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  LLM GENERATOR  │   prompt = [query + retrieved context]
          └────────┬────────┘
                   │
                   ▼
          Grounded Response (ideally citing retrieved chunks)

---

## 4.2 How RAG Reduces Hallucination

In the ideal case, RAG reduces hallucination by:

    •	Providing ground truth at inference time: The model no longer has to rely solely on its parametric 
        memory. It can 'read' relevant passages at generation time.
    
    •	Enabling source attribution: The model can cite the document chunk it drew information from, 
        making the output verifiable.
        
    •	Keeping information current: The knowledge base can be updated without retraining the model, 
        eliminating knowledge cutoff hallucinations.
        
    •	Constraining generation scope: Prompts can instruct the model to only answer using the retrieved 
        context, rejecting queries it cannot ground.
        
---

### 4.3 RAG-Specific Hallucination Modes

    Despite its benefits, RAG introduces new failure modes:

## 4.3.1 Retrieval Failure (False Negatives)

If the retriever fails to fetch the relevant document, the model falls back to its parametric 
memory — or worse, blends retrieved and memorized content into a confused output.

    Query:  'What is our refund policy for digital products?'
    Retrieved:  [general refund policy, physical products only]
    Model generates:  '...digital products follow the same 30-day policy...'
    Reality:  Digital products have no refunds.
    Failure:  Retrieval miss → model extrapolated from wrong chunk.


## 4.3.2 Context Infidelity (Unfaithful Summarization)

The model correctly retrieves relevant content but then fails to faithfully represent it — 
paraphrasing inaccurately, omitting critical caveats, or blending multiple chunks in ways 
that change meaning.

## 4.3.3 Chunk Boundary Hallucination

When a document is split into chunks, key context may be in adjacent chunks that are not 
retrieved. The model receives a fragment and hallucinates the missing surrounding context.

## 4.3.4 Contradiction Hallucination

If the knowledge base contains conflicting documents (e.g., outdated and updated policies), 
the model may blend them into a response that matches neither, or confidently select the wrong one.

## 4.3.5 Over-Reliance on Retrieved Context

If the retrieved document itself contains errors (misquotes, outdated facts, intentional misinformation), 
the model may faithfully reproduce those errors while appearing well-grounded.


RAG Hallucination Summary:

    Without RAG: Model hallucinates from parametric memory.
    With RAG (retrieval miss): Model still hallucinates from memory.
    With RAG (faithfulness failure): Model misrepresents retrieved content.
    With RAG (bad source): Model correctly cites bad information.
    RAG reduces but does not eliminate hallucination risk.

---

### 5. Grounding Strategies

Grounding refers to anchoring model outputs to verifiable, external information sources. 
Properly implemented grounding transforms the model from a free-generating system into a constrained, 
accountable one.

## 5.1 Source Attribution and Citation
The most direct grounding strategy is requiring the model to cite specific sources for every factual claim. 
This serves two purposes: it forces the model to reason from retrievable evidence, and it provides a 
path for human verification.


## 5.1.1 Inline Citation Format

    PROMPT INSTRUCTION:
        Answer using only the provided documents. For each claim,
        include a citation in the format [DOC-N, paragraph P].
        If you cannot find support in the documents, say so explicitly.
    
    MODEL OUTPUT (good):
        The activation function ReLU was popularized for deep networks in
        2010 [DOC-3, para 2]. It has the property of being computationally
        efficient while avoiding the vanishing gradient problem [DOC-3, para 4].
    
    MODEL OUTPUT (bad — hallucination despite citations):
        ReLU was invented by Alan Turing in 1952 [DOC-3, para 2].
            → The citation reference is correct but the content is fabricated.
            → This is why citations must be verified, not just present.

## 5.2 Chain-of-Thought Grounding

Requiring the model to show its reasoning step-by-step increases both reasoning 
accuracy and groundedness. 
Each step can be individually verified, and the model is less able to 'jump' to a 
fabricated conclusion.

    STANDARD PROMPTING (prone to hallucination):
      Q: Is aspirin safe for children?
      A: Yes, children can take aspirin for fevers.  ← WRONG (Reye syndrome risk)
    
    CHAIN-OF-THOUGHT (more grounded):
        Q: Is aspirin safe for children?
            Think through this carefully, step by step.
        A: Step 1: Aspirin is an NSAID and analgesic.
            Step 2: In children, aspirin has been associated with Reye syndrome,
                    a rare but serious condition affecting the liver and brain.
            Step 3: Medical guidelines recommend against aspirin for children
                    under 16 for most conditions.
            Conclusion: Aspirin is generally NOT recommended for children.

## 5.3 Constitutional / Instruction-Based Grounding

System prompts can impose hard constraints on generation — explicitly forbidding claims not 
found in the context, requiring uncertainty expressions, or mandating a 'no answer' response 
for out-of-scope queries.

Example Grounding System Prompt

    You are a factual assistant. Rules you must follow:
    1. Only make claims that are directly supported by the provided context.
    2. If the context does not answer the question, respond: 'I cannot find this in the provided documents.'
    3. Never generate statistics, names, dates, or specific facts not present in the context.
    4. For each factual claim, append the document ID and sentence number in [brackets].


## 5.4 Tool-Augmented Generation

Giving the model access to external tools (web search, code executors, calculators, databases) 
allows it to verify claims or perform operations that its internal weights cannot reliably handle.

    +-----------------+---------------------------------------------------------------+
    | Tool            | Grounding Benefit                                             |
    +-----------------+---------------------------------------------------------------+
    | Calculator      | Prevents arithmetic hallucinations by executing computation   |
    |                 | externally                                                    |
    +-----------------+---------------------------------------------------------------+
    | Web Search      | Allows retrieval of current facts beyond training cutoff      |
    +-----------------+---------------------------------------------------------------+
    | Code Executor   | Verifies that generated code actually runs and produces       |
    |                 | expected output                                               |
    +-----------------+---------------------------------------------------------------+
    | Database Lookup | Grounds entity information (prices, specifications)           |
    |                 | in live data                                                  |
    +-----------------+---------------------------------------------------------------+
    | Fact-Check API  | Flags claims against curated fact databases at generation     |
    |                 | time                                                          |
    +-----------------+---------------------------------------------------------------+

## 5.5 Structured Output Grounding

Forcing the model to emit structured outputs (JSON, XML, typed schemas) with explicit fields 
for 'claim', 'source', and 'confidence' creates a natural accountability layer. 
Each claim is separated and attributed individually, making mass hallucination harder to hide.


### 6. Calibration

Calibration is the alignment between a model's expressed confidence and its actual accuracy. 
A perfectly calibrated model is right 90% of the time when it says it is 90% confident, 
right 50% of the time when it expresses 50% confidence, and so on. 
Miscalibration is a core contributor to the harm of hallucination.

## 6.1 What Is Calibration?

PERFECT CALIBRATION:

        Expressed Confidence │ Empirical Accuracy
        ────────────────────────────────────────
             10%             │    ~10%
             30%             │    ~30%
             50%             │    ~50%
             70%             │    ~70%
             90%             │    ~90%
            100%             │   ~100%
        
        OVERCONFIDENT MODEL (common in LLMs):
        
        
        Expressed Confidence │ Empirical Accuracy
        ────────────────────────────────────────
             90%             │    ~60%  ← dangerous gap
             70%             │    ~50%
             50%             │    ~45%
    
    
## 6.2 Why LLMs Are Often Overconfident
    
    •	Training data bias: Most text is written in declarative form. 
        The corpus rarely models uncertainty as naturally as confidence.
        
    •	RLHF reward shaping: Human raters often prefer decisive, confident answers, inadvertently 
        training the model to sound more certain than warranted.
        
    •	No epistemic scoring: The model is not explicitly rewarded for saying 'I don't know' — 
        that signal is absent from standard pretraining.
        
    •	Generation dynamics: Once a confident-sounding token is generated, subsequent tokens are 
        conditioned on that confident phrasing, reinforcing the confident tone throughout the response.


## 6.3 Measuring Calibration

The most common calibration metric is the Expected Calibration Error (ECE):

    Expected Calibration Error (ECE):
    
        ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|
              m
    
    Where:
        B_m  = set of predictions in confidence bucket m
        n    = total number of predictions
        acc  = accuracy within that bucket
        conf = average confidence within that bucket
    
    ECE = 0.0  →  Perfect calibration
    ECE > 0.1  →  Significantly miscalibrated (common for LLMs)
    
    Reliability diagram: plot acc(B_m) vs conf(B_m)
    Perfect model lies on the diagonal y = x

## 6.4 Calibration vs. Accuracy

These are independent dimensions. A model can be:


    +------------------------------+---------------------------------------------------------------+
    | Calibration Profile          | Characteristics                                               |
    +------------------------------+---------------------------------------------------------------+
    | High accuracy + well         | The ideal — correct and appropriately confident.              |
    | calibrated                   | Best-in-class models with calibration training.               |
    +------------------------------+---------------------------------------------------------------+
    | High accuracy + overconfident| Often the case — the model gets most things right but         |
    |                              | expresses certainty even when it might be wrong.              |
    +------------------------------+---------------------------------------------------------------+
    | Low accuracy + well          | The model knows it doesn't know much — rare, but useful for   |
    | calibrated                   | safety-critical applications.                                 |
    +------------------------------+---------------------------------------------------------------+
    | Low accuracy + overconfident | The worst case — wrong and certain. The hallucinating model   |
    |                              | archetype.                                                    |
    +------------------------------+---------------------------------------------------------------+
    
## 6.5 Improving Calibration

    1.	Temperature scaling: Post-hoc recalibration by dividing logits by a learned temperature T. T > 1 
        softens probabilities, T < 1 sharpens them.
        
    2.	Verbalized uncertainty: Training the model to output uncertainty in natural language 
        ('I am fairly confident...', 'This is uncertain but...') correlates expressed confidence 
        with accuracy.
        
    3.	Ensemble methods: Running multiple model instances and measuring agreement provides a natural 
        confidence signal — disagreement implies uncertainty.
        
    4.	Platt scaling and isotonic regression: Statistical post-processing methods that map raw model 
        outputs to calibrated probabilities using a held-out validation set.


### 7. Refusal vs. Hallucination

Refusal — the model declining to answer or expressing that it does not know — represents a 
direct trade-off against hallucination. 

Understanding when to refuse and when to answer is one of the core design challenges in deploying 
reliable LLM systems.

## 7.1 The Trade-off Defined

    ┌────────────────────────────────────────────────────────────────┐
    │                    THE REFUSAL SPECTRUM                        │
    │                                                                │
    │  ALWAYS REFUSE                              NEVER REFUSE       │
    │  ─────────────────────────────────────────────────────────     │
    │       │                                           │            │
    │  Useless model                           Hallucinating model   │
    │  (won't help with anything)              (answers everything   │
    │                                           confidently, wrong)  │
    │                     ●                                          │
    │               Optimal zone                                     │
    │         (refuses when uncertain,                               │
    │          answers when confident)                               │
    └────────────────────────────────────────────────────────────────┘

## 7.2 Types of Refusal

    +-------------------+---------------------------------------------------------------+
    | Refusal Type      | Description                                                   |
    +-------------------+---------------------------------------------------------------+
    | Epistemic Refusal | 'I don't have reliable information about this.' — Honest      |
    |                   | acknowledgment of knowledge limitations.                      |
    +-------------------+---------------------------------------------------------------+
    | Scope Refusal     | 'This question falls outside what I can verify.' — Used in    |
    |                   | grounded systems where context doesn't cover the query.       |
    +-------------------+---------------------------------------------------------------+
    | Safety Refusal    | 'I'm not able to help with this request.' — Policy-based      |
    |                   | refusal unrelated to factual uncertainty.                     |
    +-------------------+---------------------------------------------------------------+
    | Partial Refusal   | 'I can speak to X but not Y in this question.' — Selective    |
    |                   | answering where confidence varies across sub-questions.       |
    +-------------------+---------------------------------------------------------------+


## 7.3 Costs and Benefits

Benefits of refusing: 

    •	Prevents false information from reaching the user
    
    •	Maintains trust — users learn to interpret 'I don't know' as reliable
    
    •	Supports calibration — models that refuse when uncertain are better calibrated

Costs of refusing: 

    •	Reduces utility — an overly refusing model fails its core purpose
    
    •	Can be patronizing or unhelpful in contexts where approximations are acceptable
    
    •	May be gamed — some refusals are driven by caution rather than genuine uncertainty, 
        over-refusing on safe topics

## 7.4 Designing for Appropriate Refusal

System designers can tune the refusal threshold through several mechanisms:

    5.	Confidence thresholds: Set a minimum confidence level below which the model outputs 
        an uncertainty hedge or refusal phrase.
        
    6.	Retrieval-gated generation: In RAG systems, if no sufficiently relevant document is 
        retrieved (below a similarity threshold), the model defaults to refusal rather than 
        generating from memory.
        
    7.	Explicit uncertainty prompting: System prompt instructions like 'If you are not certain 
        about a fact, say so explicitly before stating it' shift model behavior toward appropriate 
        hedging.
        
    8.	Two-pass verification: Generate a response, then have the same (or a separate) model 
        evaluate whether each claim is grounded. 
        If ungrounded claims exceed a threshold, replace with a refusal.
        
        
### 8. How Are Hallucinations Detected ?

Detection is the first line of defense in hallucination management. 
Without reliable detection, you cannot measure the problem, cannot trigger mitigations, 
and cannot give users reliable feedback about output quality.

## Human Evaluation

The gold standard — human domain experts read model output and verify each factual claim 
against trusted sources. 
High accuracy but expensive, slow, and does not scale.

    +-------------+---------------------------------------------------------------+
    | Dimension   | Human Evaluation                                              |
    +-------------+---------------------------------------------------------------+
    | Precision   | High — experts catch subtle errors including nuanced          |
    |             | misrepresentations                                            |
    +-------------+---------------------------------------------------------------+
    | Recall      | Limited by expert bandwidth — large outputs may be skimmed    |
    +-------------+---------------------------------------------------------------+
    | Scalability | Very low — cannot run on every inference                      |
    +-------------+---------------------------------------------------------------+
    | Cost        | High — requires domain knowledge and significant time         |
    +-------------+---------------------------------------------------------------+
    | Best use    | Benchmark creation, model evaluation, post-deployment audits  |
    +-------------+---------------------------------------------------------------+


## 8.2 Automated NLP-Based Detection

## 8.2.1 Entailment-Based Detection (NLI)

Natural Language Inference models classify whether a generated claim is entailed by, neutral to, 
or contradicted by a reference passage. Used widely in faithfulness evaluation.


    NLI DETECTION PIPELINE:
    
        Reference text:  'Paris is the capital of France.'
        Model output:    'The capital of France is Lyon.'
           │
           ▼
        NLI Model:  (reference, hypothesis) → CONTRADICTION  ✗
        
        
        Reference text:  'Paris is the capital of France.'
        Model output:    'France's capital, Paris, is known for art.'
           │
           ▼
        NLI Model:  (reference, hypothesis) → ENTAILMENT  ✓

## 8.2.2 QA-Based Detection (QAFactEval)

Generate question-answer pairs from the reference document, then check whether the model's output 
answers those questions consistently. 
If the model's answer conflicts with the reference, a hallucination is flagged.

## 8.2.3 Entity and Fact Extraction

Extract named entities, numbers, and dates from the model output. Cross-reference them 
against a knowledge base (Wikidata, internal DB). 
Flag entities that don't match.

    ENTITY EXTRACTION PIPELINE:

        Output:  'Neil Armstrong landed on the moon on July 20, 1969.'
               │
        Extract: entity='Neil Armstrong', event='moon landing', date='July 20 1969'
               │
        Lookup:  KB check → Armstrong + moon landing + July 20 1969 → MATCH ✓
        
        Output:  'Buzz Aldrin was the first human on the moon.'
               │
        Extract: entity='Buzz Aldrin', claim='first human on moon'
               │
        Lookup:  KB check → first human on moon = Neil Armstrong → MISMATCH ✗


## 8.3 LLM-as-Judge

Use a second (often larger or more capable) LLM to evaluate the output of the first. 
The judge is prompted to identify unsupported claims, factual errors, or consistency failures.

    LLM-JUDGE PROMPT TEMPLATE:
        System  :   You are a rigorous fact-checker.
        User    :   Given this reference document: [DOCUMENT]
                    And this model response: [RESPONSE]
                    List every claim in the response that is:
                    (a) Not supported by the document
                    (b) Contradicted by the document
                    (c) Plausible but unverifiable from the document
                    Format: CLAIM | STATUS | REASON


## 8.4 Self-Consistency Checking

Sample multiple responses from the model for the same query. 
If answers are consistent across samples, confidence in accuracy increases. 
If answers vary significantly, a hallucination (or genuine uncertainty) is likely.

    SELF-CONSISTENCY DETECTION:
    
        Query: 'In what year was the Eiffel Tower completed?'
        
        Sample 1: '1889'
        Sample 2: '1889'
        Sample 3: '1889'
        Verdict:  HIGH AGREEMENT → likely correct
        
        ---------------------------------------------------------------
        
        Query: 'Who was the 24th Prime Minister of Canada?'
        
        Sample 1: 'Pierre Trudeau'
        Sample 2: 'John Turner'
        Sample 3: 'Brian Mulroney'
        Verdict:  LOW AGREEMENT → uncertain / flag for verification


## 8.5 Perplexity and Probability Monitoring

At the token level, a model's internal probability distribution can hint at uncertainty. 
When the model assigns low probability to its own generated tokens, that may indicate the 
model is in a low-confidence region of its learned distribution — a potential hallucination signal.

**Caveat**

    Low probability does not reliably predict hallucination in all cases. A model can generate a
    rare-but-true token with low probability, or generate a common-but-false token with high
    probability. Perplexity is a weak but computationally cheap signal.


### 9. Steps and Methods to Avoid Hallucinations

Hallucination mitigation is a multi-layer engineering and design challenge. 
No single technique eliminates hallucination; effective systems stack multiple approaches.

## 9.1 Mitigation Layers Overview
    
    HALLUCINATION MITIGATION LAYERS
    ═════════════════════════════════════════════════════════════════
    
        Layer 1: TRAINING                                              
        ─────────────────                                              
        • Better pretraining data quality                              
        • Uncertainty-aware fine-tuning                                
        • RLHF with accuracy-based rewards                             
                                                                     
        Layer 2: ARCHITECTURE                                          
        ─────────────────────                                          
        • Retrieval augmentation (RAG)                                 
        • Tool use (calculators, search, code execution)               
        • Structured output schemas                                    
                                                                     
        Layer 3: PROMPTING                                             
        ──────────────────                                             
        • Chain-of-thought / scratchpad                                
        • Explicit grounding instructions                              
        • Citation requirements                                        
        • 'Say I don't know' instructions                              
                                                                     
        Layer 4: INFERENCE                                             
        ─────────────────                                              
        • Self-consistency sampling                                    
        • Confidence thresholding                                      
        • Reranking / verification passes                              
                                                                     
        Layer 5: POST-GENERATION                                       
        ────────────────────────                                       
        • NLI-based faithfulness checking                              
        • LLM-as-judge pipelines                                       
        • Human review workflows                                     
    ═════════════════════════════════════════════════════════════════  


### 9.2 Training-Time Mitigations

## 9.2.1 High-Quality Training Data Curation

The model can only know what it has seen. Training on noisy, incorrect, or contradictory 
data directly increases hallucination rates. 

Key curation steps:

    •	Deduplication — repeated false information in training data increases the model's 
        learned confidence in falsehoods
        
    •	Source quality filtering — prioritize peer-reviewed, authoritative sources over 
        forums and unverified content
        
    •	Contradiction detection — identify and resolve conflicting claims about the same facts
    
    •	Temporal filtering — exclude outdated information that may conflict with current facts

### 9.2.2 Factuality-Focused Fine-Tuning

Fine-tuning on datasets where factual accuracy is explicitly labeled and rewarded can shift 
model behavior toward more grounded outputs. 

Approaches include:

    •	TruthfulQA-style fine-tuning: Fine-tune on datasets designed to elicit calibrated, 
        honest answers over persuasive but false ones.
        
    •	Negative examples: Train the model to recognize hallucinations in example outputs 
        and penalize them during supervised fine-tuning.
        
    •	Attribution training: Fine-tune the model to cite sources for factual claims, so 
        that citation becomes a habit during generation.

### 9.3 Prompting Strategies

## 9.3.1 Explicit Uncertainty Instructions

    PROMPT TEMPLATE FOR UNCERTAINTY:
    
        'When answering factual questions:',
        '- If you are highly confident (>90%), state the fact directly.',
        '- If you are moderately confident (50-90%), prefix with: "I believe..."',
        '- If you are uncertain (<50%), say: "I am not sure, but..."',
        '- If you do not know, say: "I don't have reliable information on this."',
        'Never fabricate specific names, dates, or numbers you are not certain of.'

## 9.3.2 Context-Constrained Generation

    CONTEXT CONSTRAINT PROMPT:
    
        Context: [RETRIEVED DOCUMENTS]

        Question: [USER QUERY]
        
        Instructions: Answer the question using ONLY information
        from the context above. If the context does not contain
        sufficient information to answer, respond with:
        'The provided documents do not address this question.'
        Do not use your background knowledge to supplement the context.

        
### 9.4 Inference-Time Mitigations

## 9.4.1 Self-Consistency Decoding

Sample k responses (e.g., k=5) using high temperature, then aggregate answers via majority 
vote or semantic clustering. 
The most consistent answer across samples is the one to return.

## 9.4.2 Chain-of-Thought with Verification Step

    EXTENDED COT WITH SELF-CHECK:
    
        Step 1: Generate full reasoning chain
        
        Step 2: For each factual claim in the chain, ask:
                'Is this claim directly supported by the context/my
                certain knowledge? If not, mark as [UNCERTAIN].'
                
        Step 3: If any [UNCERTAIN] claim is load-bearing for the
                conclusion, revise the conclusion or add a caveat.


### 9.5 System Architecture Mitigations

## 9.5.1 Retrieval-Augmented Generation (RAG)

See Section 4. RAG provides the single largest reduction in hallucination for knowledge-intensive 
tasks when implemented well. 

Best practices:

    •	Use hybrid retrieval (dense + sparse) to maximize recall of relevant chunks
    
    •	Implement a confidence threshold on retrieval — if no chunk exceeds it, route to refusal
    
    •	Use reranking (cross-encoder) to improve chunk relevance before feeding to the generator
    
    •	Chunk documents with appropriate overlap to avoid boundary artifacts

## 9.5.2 Tool Use

Offloading computation-heavy, lookup-heavy, or currency-sensitive tasks to external tools 
removes them from the model's unreliable memory:

    +------------------------+---------------------------------------------------------------+
    | Task Type              | Tool Strategy                                                 |
    +------------------------+---------------------------------------------------------------+
    | Arithmetic             | Use a code execution tool instead of letting the model        |
    |                        | compute — LLMs make frequent arithmetic errors                |
    +------------------------+---------------------------------------------------------------+
    | Current events         | Use web search to retrieve up-to-date facts before generating |
    +------------------------+---------------------------------------------------------------+
    | Entity details         | Query a structured knowledge base (e.g., Wikidata API) for    |
    |                        | specific entity properties                                    |
    +------------------------+---------------------------------------------------------------+
    | Technical documentation| Retrieve official docs at query time rather than relying on   |
    |                        | training-time knowledge                                       |
    +------------------------+---------------------------------------------------------------+

## 9.5.3 Two-Stage Generate-Then-Verify

GENERATE-THEN-VERIFY PIPELINE:

  ┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
  │  Generator    │────▶│   Verifier       │────▶│   Output     │
  │  LLM          │     │   (NLI / LLM-    │     │   Router     │
  │               │     │    judge / rules)│     │              │
  └───────────────┘     └──────────────────┘     └──────┬───────┘
                                                        │
                                  ┌─────────────────────┤
                                  │                     │
                            PASS  ▼                FAIL ▼
                          Return response         Return refusal /
                                                  request clarification



### 10. Root Causes of Hallucinations

A structured taxonomy of the root causes helps prioritize mitigation strategies — 
addressing root causes is more effective than treating symptoms.

## 10.1 Training Data Causes

## 10.1.1 Knowledge Gaps

The model was simply never exposed to certain information during training. 
When queried about facts outside its training distribution, it interpolates from 
related patterns — producing plausible but wrong answers.

EXAMPLE:
    Query about an obscure regional court ruling from 2019.
    Not in training data → model generates a plausible ruling
    structure with fabricated case number and judge's name.

## 10.1.2 Misinformation in Training Data

Web-scale training corpora inevitably contain misinformation — conspiracy theories, 
satire mistaken for fact, outdated information, and deliberate falsehoods. 
These co-train with factual data, and the model may have learned incorrect facts 
with high confidence if they appeared frequently.

## 10.1.3 Knowledge Cutoff Effects

Models have a training cutoff date. For queries about events after the cutoff, 
the model lacks factual ground truth entirely and is forced to extrapolate from 
patterns — a direct invitation to hallucinate. 
This is especially dangerous because the model often does not know it lacks the 
relevant knowledge.

## 10.1.4 Conflicting Information in Training Data

The internet contains contradictory claims about the same facts (different sources 
cite different statistics, outdated information exists alongside current information). 
The model learns a blended representation that matches no single authoritative source, 
producing confident outputs that are a mixture of conflicting claims.

### 10.2 Model Architecture Causes10.2 Model Architecture Causes

    ARCHITECTURE CAUSES TREE
    
        LLM Architecture
        │
        ├── No external memory
        │        └── Must generate from weights alone for factual recall
        │
        ├── Autoregressive generation
        │        └── Each token conditioned on previous (possibly wrong) tokens
        │            → Error compounding
        │
        ├── Finite context window
        │        └── Long documents: middle-position attention degradation
        │
        └── Attention mechanism
                 └── Soft relevance weighting → partial information blending
               
## 10.2.1 Parametric Memory as Knowledge Store

LLM 'knowledge' is distributed across billions of weight parameters — not stored as 
addressable facts. 
Retrieval from parametric memory is approximate and associative, not exact. 
This is fundamentally different from a database lookup and introduces reconstruction errors.

## 10.2.2 Attention Drift in Long Contexts

As context length increases, attention scores for tokens far from the current position decay. 
The model may effectively ignore a key grounding statement at position 500 when generating 
output at position 3000, reverting to its parametric knowledge — which may contradict the context.

## 10.3 Training Procedure Causes

## 10.3.1 Teacher Forcing Leads to Exposure Bias

During training, the model always sees the correct previous token as context (teacher forcing). 
At inference time, errors in generated tokens compound because the model sees its own potentially 
wrong tokens as context. This distributional shift (called exposure bias) means the model was 
never trained to recover from its own errors.

## 10.3.2 RLHF Sycophancy

RLHF using human feedback tends to reward confident, well-structured, authoritative-sounding responses. 
Human raters often cannot tell when a confident answer is wrong. 
This creates a training signal that rewards confident generation regardless of accuracy — a direct cause 
of confident hallucination.

The Sycophancy Feedback Loop

    1. Model generates a confident-but-wrong answer.
    
    2. Human rater finds the answer well-written and confident → gives high reward.
    
    3. RLHF updates weights toward more confident generation.
    
    4. Next model version is more confident — and possibly more prone to hallucinating.
    
Breaking this loop requires raters who are domain experts and can distinguish
confidence from correctness — expensive and hard to scale.


## 10.4 Inference Causes

## 10.4.1 Decoding Strategy Effects

The way tokens are sampled from the probability distribution affects hallucination rates:


    +--------------------------+---------------------------------------------------------------+
    | Decoding Strategy        | Hallucination Characteristics                                 |
    +--------------------------+---------------------------------------------------------------+
    | Greedy decoding          | Always takes highest-probability token. Lower diversity.      |
    |                          | Can get 'locked in' to a confident but wrong path.            |
    +--------------------------+---------------------------------------------------------------+
    | High-temperature sampling| More diverse outputs but may sample less likely (possibly     |
    |                          | wrong) tokens more often.                                     |
    +--------------------------+---------------------------------------------------------------+
    | Nucleus sampling (top-p) | Truncates the distribution. Better calibrated in practice     |
    |                          | but hallucination still occurs in low-frequency knowledge     |
    |                          | regions.                                                      |
    +--------------------------+---------------------------------------------------------------+
    | Beam search              | Generates multiple candidate sequences. Can reduce            |
    |                          | hallucination on short outputs but may still hallucinate      |
    |                          | plausibly.                                                    |
    +--------------------------+---------------------------------------------------------------+


## 10.4.2 Prompt Injection and Adversarial Inputs

Carefully crafted inputs can steer the model into hallucination. 
Adversarial prompts that set false premises ('Given that Einstein discovered relativity in 1880...') 
can cause the model to hallucinate downstream facts consistent with the false premise, even if it
would normally get them right.

## 10.5 Summary: Root Cause Map

    ROOT CAUSE MAP
    ════════════════════════════════════════════════════════════════
    
        TRAINING DATA
          ├─ Knowledge gaps          → Facts outside training distribution
          ├─ Misinformation           → False data trained with high frequency
          ├─ Temporal cutoff          → Post-cutoff queries have no ground truth
          └─ Contradictions           → Blended/averaged conflicting claims
        
        ARCHITECTURE
          ├─ Parametric memory        → Approximate associative recall
          ├─ No 'I don't know'        → Generation always proceeds
          ├─ Autoregressive errors    → Error propagation through tokens
          └─ Context length limits    → Attention drift on long inputs
        
        TRAINING PROCEDURE
          ├─ Teacher forcing          → Exposure bias at inference
          ├─ RLHF sycophancy          → Confidence rewarded over accuracy
          └─ Loss function            → No explicit factuality term
        
        INFERENCE
          ├─ Decoding dynamics        → High temperature increases errors
          ├─ Prompt framing           → False premises induce hallucination
          └─ Adversarial inputs       → Deliberate steering into error
        
        DEPLOYMENT
          ├─ No retrieval             → Forces parametric knowledge use
          └─ No verification          → Errors reach user undetected
          
    ════════════════════════════════════════════════════════════════

## 10.6 The Compounding Effect

In practice, multiple root causes interact. 
A training corpus with contradictory facts (training data cause) produces a model with 
uncertain parametric knowledge (architecture cause). Deployed without RAG (deployment cause) 
and with RLHF that rewards confidence (training procedure cause), the model confidently 
generates blended misinformation — a cascade of causes producing the worst-case 
hallucination outcome.

Key Takeaway

    Hallucination is not a bug to be patched — it is a structural property of current LLM
    architectures. A language model trained on next-token prediction has no intrinsic notion
    of truth, no external memory to fall back on, and no reliable epistemic signal.
    
    Effective mitigation requires addressing causes at multiple layers: better data, better
    training objectives, retrieval augmentation, grounded inference strategies, and post-
    generation verification. Defense-in-depth is the only reliable approach.



"""

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = """

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Step-by-step tutorials with runnable commands
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
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